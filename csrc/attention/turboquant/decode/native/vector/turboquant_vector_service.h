#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_VECTOR_SERVICE_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_VECTOR_SERVICE_H

#include "../../../common/turboquant_codec_950.h"
#include "../../../common/turboquant_common.h"
#include "../../../common/turboquant_mode.h"
#include "routine_vector_cube_softmax.hpp"
#include "../../../common/turboquant_fused_decode_opt.hpp"

namespace vllm_ascend {
namespace turboquant {

constexpr float kVectorNegInf = -3.4028235e38f;

__aicore__ inline void NormalizeHeads(const AscendC::LocalTensor<float> &acc, const AscendC::LocalTensor<float> &runSum,
                                      const uint32_t heads, const uint32_t headSize,
                                      const AscendC::LocalTensor<float> &sums, const AscendC::LocalTensor<float> &invs)
{
    ascend_micro::vector_cube_normalize_heads(acc, runSum, sums, invs, heads, headSize, TurboQuantCodec4::kEps);
}

template <typename scalar_t>
__aicore__ inline void WriteHeads(const AscendC::LocalTensor<float> &acc, const uint32_t heads, const uint32_t headSize,
                                  const AscendC::LocalTensor<scalar_t> &out, AscendC::GlobalTensor<scalar_t> &outputGm,
                                  const uint64_t outputElem)
{
    AscendC::Cast(out, acc, AscendC::RoundMode::CAST_RINT, heads * headSize);
    SyncVectorToMte3();
    AscendC::DataCopy(outputGm[outputElem], out, heads * headSize);
    SyncMte3ToVector();
}

template <typename scalar_t>
__aicore__ inline void WriteNormalizedHeads(const AscendC::LocalTensor<float> &acc,
                                            const AscendC::LocalTensor<float> &runSum, const uint32_t heads,
                                            const uint32_t headSize, const AscendC::LocalTensor<float> &sums,
                                            const AscendC::LocalTensor<float> &invs,
                                            const AscendC::LocalTensor<scalar_t> &out,
                                            AscendC::GlobalTensor<scalar_t> &outputGm, const uint64_t outputElem)
{
    NormalizeHeads(acc, runSum, heads, headSize, sums, invs);
    WriteHeads<scalar_t>(acc, heads, headSize, out, outputGm, outputElem);
}

struct TurboQuantRotatedOutput {
    __aicore__ static inline void FinishHeads(const AscendC::LocalTensor<float> &acc, const uint32_t token,
                                              const uint32_t firstHead, const uint32_t heads)
    {
        static_cast<void>(acc);
        static_cast<void>(token);
        static_cast<void>(firstHead);
        static_cast<void>(heads);
    }
};

template <typename scalar_t>
class TurboQuantPartialReducer {
public:

    static_assert(sizeof(scalar_t) == 2,
                  "turboquant_fused_decode_opt.hpp's kUbEnd/kBasisUbBytes/kReducerUbBytes assume a "
                  "16-bit output element");

    __aicore__ inline explicit TurboQuantPartialReducer(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *workspace, __gm__ void *output, const uint32_t numHeads,
                                const uint32_t headSize, const uint32_t numSplits)
    {
        numHeads_      = numHeads;
        headSize_      = headSize;
        numSplits_     = numSplits;
        partialStride_ = headSize + kPartialTail;

        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(output));

        pipe_->InitBuffer(accBuf_, (2 * headSize_ + kPartialTail) * sizeof(float));
        pipe_->InitBuffer(stateBuf_, kMergeStateLanes * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(blockBuf_, (2 * kBrcbDstLanes + 2 * kFp32PerBlock) * sizeof(float));
        pipe_->InitBuffer(outBuf_, headSize_ * sizeof(scalar_t));

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Reduce(const uint32_t token, const uint32_t head)
    {
        TurboQuantRotatedOutput rotated;
        Reduce(token, head, rotated);
    }

    template <typename Finisher>
    __aicore__ inline void Reduce(const uint32_t token, const uint32_t head, Finisher &finisher)
    {
        const AscendC::LocalTensor<float> acc       = accBuf_.Get<float>();
        const AscendC::LocalTensor<float> partAcc   = acc[headSize_];
        const AscendC::LocalTensor<float> partState = acc[2 * headSize_];
        const AscendC::LocalTensor<float> state     = stateBuf_.Get<float>();
        const AscendC::LocalTensor<float> blocks    = blockBuf_.Get<float>();

        ascend_micro::NativeMergeEngine merge{acc,
                                              partAcc,
                                              state[kPartialMaxLane],
                                              state[kPartialSumLane],
                                              partState[kPartialMaxLane],
                                              partState[kPartialSumLane],
                                              state[kMergeNewMaxLane],
                                              state[kMergeAlphaLane],
                                              state[kMergeBetaLane],
                                              blocks[kMergeAlphaBlock],
                                              blocks[kMergeBetaBlock],
                                              headSize_};

        AscendC::Duplicate(acc, 0.0f, headSize_);
        AscendC::Duplicate(state, 0.0f, kMergeStateLanes * kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(merge.run_max, kVectorNegInf, 1);

        for (uint32_t split = 0; split < numSplits_; ++split) {
            SyncVectorToMte2();
            AscendC::DataCopy(partAcc, workspaceGm_[PartialOffset(token, head, split)], partialStride_);
            SyncMte2ToVector();
            ascend_micro::vector_merge_partials(merge);
        }

        NormalizeHeads(acc, merge.run_sum, 1, headSize_, blocks[kMergeSums], blocks[kMergeInvs]);
        finisher.FinishHeads(acc, token, head, 1);
        WriteHeads<scalar_t>(acc, 1, headSize_, outBuf_.Get<scalar_t>(), outputGm_,
                             (static_cast<uint64_t>(token) * numHeads_ + head) * headSize_);
    }

private:

    static constexpr uint32_t kMergeNewMaxLane = 2 * kFp32PerBlock;
    static constexpr uint32_t kMergeAlphaLane  = 3 * kFp32PerBlock;
    static constexpr uint32_t kMergeBetaLane   = 4 * kFp32PerBlock;
    static constexpr uint32_t kMergeStateLanes = 5;

    static constexpr uint32_t kMergeAlphaBlock = 0;
    static constexpr uint32_t kMergeBetaBlock  = kBrcbDstLanes;
    static constexpr uint32_t kMergeSums       = 2 * kBrcbDstLanes;
    static constexpr uint32_t kMergeInvs       = 2 * kBrcbDstLanes + kFp32PerBlock;

    __aicore__ inline uint64_t PartialOffset(const uint32_t token, const uint32_t head, const uint32_t split) const
    {
        return ((static_cast<uint64_t>(token) * numHeads_ + head) * numSplits_ + split) * partialStride_;
    }

    AscendC::TPipe                            *pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stateBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> blockBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outBuf_;
    AscendC::GlobalTensor<float>                 workspaceGm_;
    AscendC::GlobalTensor<scalar_t>              outputGm_;
    uint32_t                                     numHeads_      = 0;
    uint32_t                                     headSize_      = 0;
    uint32_t                                     numSplits_     = 1;
    uint32_t                                     partialStride_ = 0;
};

template <TurboQuantMode MODE>
__aicore__ inline constexpr float OperandMax()
{
    return TurboQuantModeTraits<MODE>::kOperand == TurboQuantOperand::kFp4E2m1 ? 6.0f : 448.0f;
}

template <TurboQuantMode MODE>
constexpr bool kBatchedCubeDecode = MODE == TurboQuantMode::KV4_FP8;

struct TurboQuantTaskHeads {
    uint32_t first = 0;
    uint32_t rows  = 0;
    uint32_t base  = 0;
    uint32_t mine  = 0;
};

template <TurboQuantMode MODE, typename scalar_t, typename Mm, typename Codec,
          DecodeUnpack UNPACK = DecodeUnpack::kNative, uint32_t RING_SLOTS = kCubeSlots,
          uint32_t SCORE_WINDOWS = 1, uint32_t BANK_SKEW = 0>
class TurboQuantVectorDecodeService {
public:
    using OperandT = typename Mm::OperandT;
    static constexpr bool kBatched = kBatchedCubeDecode<MODE>;
    static constexpr bool kNzTiled = kStoresNzTiles<MODE>;
    static_assert(MODE == TurboQuantMode::KV4_FP8,
                  "the only decode kernels declared are kv4fp8; another mode needs its ingest and "
                  "softmax verified here before this service will carry it");
    static_assert(UNPACK == DecodeUnpack::kNative || UNPACK == DecodeUnpack::kLloydMaxLut,
                  "the tile pipeline carries the native expand and the Lloyd-Max gather expand; "
                  "another variant is added as a routine here, not as a branch nobody runs");
    static_assert(kBatched, "kv4fp8 batches its heads; the per-head softmax path is gone");
    static_assert(kNzTiled, "kv4fp8 streams NZ-tiled planes; the row-major burst path is gone");
    static_assert(kNzTiled == Codec::kIsAffine, "only the byte-wise affine expand streams an NZ-tiled plane");
    static_assert(Codec::kGatherLut == DecodeUnpackGathers(UNPACK) &&
                      Codec::kLloydMaxLut == DecodeUnpackIsLloydMax(UNPACK),
                  "the service and its codec disagree on the expand");

    static_assert(sizeof(scalar_t) == 2,
                  "turboquant_fused_decode_opt.hpp's UB map assumes a 16-bit output element");

    static constexpr uint32_t kIngestSlots = RING_SLOTS;

    static constexpr uint32_t kScoreWindows = SCORE_WINDOWS;
    static constexpr uint32_t kBankSkew     = BANK_SKEW;
    static_assert(RING_SLOTS >= kCubeSlots, "the ingest ring cannot be shallower than the L1 ring");
    static_assert(SCORE_WINDOWS >= 1, "the Cube needs somewhere to put the scores");
    static_assert(BANK_SKEW % (kFp32PerBlock * sizeof(float)) == 0,
                  "a skew that is not a whole UB block would misalign every buffer behind it");

    __aicore__ inline void Init(AscendC::TPipe *pipe, __gm__ void *queryRot, __gm__ void *keyCache,
                                __gm__ void *valueCache, __gm__ void *scaleCache, __gm__ void *modeTables,
                                __gm__ void *workspace, __gm__ void *output, const uint32_t numHeads,
                                const uint32_t numKvHeads, const uint32_t headSize, const uint32_t blockSize,
                                const uint32_t numSplits, const float scale, const float invSqrtLen)
    {
        ComputeLayout(numHeads, numKvHeads, headSize, blockSize, numSplits, scale);
        InitGlobalTensors(queryRot, keyCache, valueCache, scaleCache, modeTables, workspace, output);
        InitBuffers(pipe);
        codec_.Init(pipe, headSize_, kUnpackChunkRows, invSqrtLen, modeTablesGm_);
        InitIngestEvents();
    }

    __aicore__ inline AscendC::TBufPool<AscendC::TPosition::VECCALC> &TransientPool()
    {
        return transientPool_;
    }
    static constexpr uint32_t kTransientPoolBytes = ascend_micro::fused_decode_opt::kTransientPoolBytes;

    __aicore__ inline AscendC::LocalTensor<float> Scores(const uint32_t window = 0)
    {
        return scoreBuf_[window].template Get<float>();
    }
    __aicore__ inline AscendC::LocalTensor<float> Context() { return contextBuf_.Get<float>(); }

    __aicore__ static inline TurboQuantTaskHeads Heads(const uint32_t first, const uint32_t rows)
    {
        TurboQuantTaskHeads heads;
        heads.first                = first;
        heads.rows                 = rows;
        const uint32_t firstHalf   = CeilDiv(rows, kVectorSubcoresPerBlock);
        heads.mine                 = firstHalf;
        if ASCEND_IS_AIV {
            if (AscendC::GetSubBlockIdx() != 0) {
                heads.base = firstHalf;
                heads.mine = rows - firstHalf;
            }
        }
        return heads;
    }

    __aicore__ inline void BeginTask(const TurboQuantTaskHeads &heads)
    {
        if (heads.mine == 0) {
            return;
        }
        AscendC::Duplicate(accBuf_.Get<float>(), 0.0f, heads.mine * headSize_);
        AscendC::Duplicate(stateBuf_.Get<float>(), 0.0f, kStateFields * kStateField);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(StateField(kStateRunMax), kVectorNegInf, heads.mine * kFp32PerBlock);
    }

    __aicore__ inline event_t PostQueryRead(const uint32_t token, const TurboQuantTaskHeads &heads)
    {
        if (heads.mine == 0) {
            return {};
        }
        AscendC::DataCopy(queryInBuf_.Get<float>(),
                          queryRotGm_[(static_cast<uint64_t>(token) * numHeads_ + heads.first + heads.base) *
                                      headSize_],
                          heads.mine * headSize_);

        const auto ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ready);
        return ready;
    }

    __aicore__ inline void DiscardPrologueIngress(const TurboQuantTaskHeads &heads, const event_t queryReady)
    {
        if (heads.mine > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(queryReady);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(readEvents_[0]);
    }

    __aicore__ inline void StageQuery(Mm &mm, const uint32_t kvHead, const TurboQuantTaskHeads &heads,
                                      const event_t queryReady)
    {
        if (heads.mine == 0) {
            return;
        }
        const AscendC::LocalTensor<float>    queryIn      = queryInBuf_.Get<float>();
        const AscendC::LocalTensor<OperandT> queryOperand = queryOperandBuf_.Get<OperandT>();
        const AscendC::LocalTensor<OperandT> queryL1      = mm.L1Query();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(queryReady);

        ascend_micro::NativeCubeQueryEngine<Codec, OperandT> query{queryIn,
                                                                   queryWorkBuf_.Get<float>(),
                                                                   scratchBuf_.Get<float>(),
                                                                   StateField(kStateRowReduce),
                                                                   StateField(kStateAmax),
                                                                   StateField(kStateQScale),
                                                                   StateField(kStateQInv),
                                                                   queryOperand,
                                                                   codec_,
                                                                   heads.mine,
                                                                   headSize_,
                                                                   OperandMax<MODE>(),
                                                                   TurboQuantCodec4::kEps,
                                                                   rowRepeatParams_};
        ascend_micro::vector_cube_quantize_query(query);

        SyncVectorToMte3();
        for (uint32_t head = 0; head < heads.mine; ++head) {
            AscendC::DataCopy(queryL1[OperandIndex((heads.base + head) * kOperandC0)],
                              queryOperand[OperandIndex(head * operandElems_)], queryRowToL1Params_);
        }
        BuildScaleIndex(kvHead);
    }

    __aicore__ inline void ComputeTailMask(const uint32_t valid)
    {
        const AscendC::LocalTensor<float> mask = maskBuf_.Get<float>();
        ascend_micro::vector_cube_tail_mask(mask, mask[kCubeTileRows].ReinterpretCast<int32_t>(), valid,
                                            kMaskMidpoint, kMaskGain);

        AscendC::Compares(mask[kCubeTileRows].ReinterpretCast<uint8_t>(), mask, 0.0f,
                          AscendC::CMPMODE::GT, static_cast<uint32_t>(kCubeTileRows));
    }

    __aicore__ inline void BeginArgmaxProbe(const TurboQuantTaskHeads &heads)
    {
#if defined(ASCEND_TQ_DECODE_TELEMETRY) && ASCEND_TQ_DECODE_TELEMETRY
        if (heads.mine == 0) {
            return;
        }
        const AscendC::LocalTensor<float> arena = queryInBuf_.Get<float>();
        AscendC::Duplicate(arena[kProbePos], kTelemetryInvalid, heads.mine * kFp32PerBlock);
        AscendC::Duplicate(arena[kProbePage], kTelemetryInvalid, heads.mine * kFp32PerBlock);
        AscendC::ArithProgression(arena[kProbeSel].ReinterpretCast<int32_t>(), 0, 1,
                                  static_cast<int32_t>(kCubeTileRows));
        AscendC::Cast(arena[kProbeRamp], arena[kProbeSel].ReinterpretCast<int32_t>(),
                      AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(kCubeTileRows));
        AscendC::Compares(arena[kProbePred].ReinterpretCast<uint8_t>(), arena[kProbeRamp], 1.0f,
                          AscendC::CMPMODE::EQ, static_cast<uint32_t>(kPartialTail));
        AscendC::Compares(arena[kProbePred2].ReinterpretCast<uint8_t>(), arena[kProbeRamp], 2.0f,
                          AscendC::CMPMODE::EQ, static_cast<uint32_t>(kPartialTail));
#else
        static_cast<void>(heads);
#endif
    }

    template <bool PACKED = true, bool SCALES = true>
    __aicore__ inline void ReadTileToSlot(const uint32_t physical, const uint32_t rowBase, const uint32_t kvHead,
                                          const uint32_t slot)
    {
        static_assert(PACKED || SCALES);
        ReadPlane<PACKED, SCALES>(physical, rowBase, kvHead, slot);
        if constexpr (SCALES) {

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(readEvents_[slot]);
        }
    }

    __aicore__ inline void StageSlotToL1(Mm &mm, const uint32_t kvHead, const uint32_t slot)
    {
        static_cast<void>(kvHead);
        const AscendC::LocalTensor<OperandT> l1Dst = PlaneOperand(mm, slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(readEvents_[slot]);
        StageTileToL1(kvBuf_[slot].template Get<int8_t>(), l1Dst);
    }

    template <uint32_t WINDOW, bool PREPARED>
    __aicore__ inline void ComputeSoftmaxAndStageProbs(Mm &mm, const uint32_t valid,
                                                       const TurboQuantTaskHeads &heads, const uint32_t slot,
                                                       const uint32_t contextBase, const uint32_t logicalPage)
    {
        if (heads.mine == 0) {
            return;
        }
        static_assert(!PREPARED || (!Mm::kIsFp4 && WINDOW < 2));
        constexpr uint32_t panel = PREPARED ? WINDOW * kPairProbabilityPanelStride : 0;
        const auto probOperand = probOperandBuf_.Get<OperandT>()[panel];
        const auto probsL1 = mm.L1Probs();
        const uint32_t probElems = Mm::OperandElems(kCubeTileRows);
        if constexpr (!PREPARED) {
            const auto reduce = reduceBuf_.Get<float>();
            const auto keyScale = reduce[kReduceKeyScale];
            const auto valueScale = reduce[kReduceValueScale];
            ascend_micro::vector_cube_tile_scales(keyScale, valueScale,
                scaleTileBuf_[slot].template Get<float>(),
                scaleIndexBuf_.Get<int32_t>().ReinterpretCast<uint32_t>(), scoreScale_, invGain_);
            RunTileSoftmax(Scores(WINDOW), valid, heads.mine, keyScale, valueScale, contextBase, logicalPage);
        }
        SyncVectorToMte3();
        for (uint32_t head = 0; head < heads.mine; ++head) {
            AscendC::DataCopy(probsL1[OperandIndex((heads.base + head) * kOperandC0)],
                              probOperand[OperandIndex(head * probElems)], probRowToL1Params_);
        }
        if constexpr (!PREPARED) {
            SyncMte3ToVector();
        }
    }

    template <uint32_t WINDOW>
    __aicore__ inline void PreparePairLogits()
    {
        static_assert(WINDOW < 2);
        const auto keyScale = PairKeyScale<WINDOW>();
        const auto scaleTile = scaleTileBuf_[WINDOW].template Get<float>();
        const auto offsets = scaleIndexBuf_.Get<int32_t>().ReinterpretCast<uint32_t>();
        AscendC::Gather(keyScale, scaleTile, offsets, 0, kCubeTileRows);
        AscendC::Muls(keyScale, keyScale, scoreScale_, kCubeTileRows);
    }

    __aicore__ inline void ComputePreparedSoftmaxPair(const uint32_t firstContext, const uint32_t firstPage,
                                                     const uint32_t secondContext, const uint32_t secondPage,
                                                     const uint32_t secondValid)
    {
        static_assert(kScoreWindows == 2);
        const AscendC::LocalTensor<float> arena = queryInBuf_.Get<float>();
        const AscendC::LocalTensor<float> reduce = reduceBuf_.Get<float>();
        const auto firstValueScale = reduce[kReducePairFirstValueScale];
        const auto secondValueScale = reduce[kReducePairSecondValueScale];
        const auto valueOffsets = scaleIndexBuf_.Get<int32_t>().ReinterpretCast<uint32_t>()[kCubeTileRows];
        AscendC::Gather(firstValueScale, scaleTileBuf_[0].template Get<float>(), valueOffsets, 0, kCubeTileRows);
        AscendC::Gather(secondValueScale, scaleTileBuf_[1].template Get<float>(), valueOffsets, 0, kCubeTileRows);
        AscendC::Muls(firstValueScale, firstValueScale, invGain_, kCubeTileRows);
        AscendC::Muls(secondValueScale, secondValueScale, invGain_, kCubeTileRows);

        const auto packed = probOperandBuf_.Get<fp8_e4m3fn_t>();
        ascend_micro::native_cube_softmax_pair_vf(
            reinterpret_cast<__ubuf__ float*>(Scores(0).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(Scores(1).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(firstValueScale.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(secondValueScale.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(StateField(kStateRunMax).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(StateField(kStateRunSum).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(PairState(0, kPairMaximum).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(PairState(0, kPairAlpha).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(PairState(1, kPairMaximum).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(PairState(1, kPairAlpha).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(arena[kProbePos].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(arena[kProbePage].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(arena[kProbeRamp].GetPhyAddr()),
            firstContext, firstPage, secondContext, secondPage,
            reinterpret_cast<__ubuf__ float*>(PairKeyScale<0>().GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(PairKeyScale<1>().GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(StateField(kStateQInv).GetPhyAddr()),
            secondValid,
            reinterpret_cast<__ubuf__ float*>(StateField(kStateProbScale).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(StateField(kStatePart).GetPhyAddr()),
            reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(packed.GetPhyAddr()),
            reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(packed[kPairProbabilityPanelStride].GetPhyAddr()),
            OperandMax<MODE>(), TurboQuantCodec4::kEps);
    }

    __aicore__ inline void ComputePreparedAccumulatorDecay(const TurboQuantTaskHeads &heads,
                                                            const uint32_t window)
    {
        ascend_micro::vector_cube_decay_accumulator(accBuf_.Get<float>(), PairState(window, kPairAlpha),
                                                    StateField(kStateRunMax), PairState(window, kPairMaximum),
                                                    heads.mine, headSize_, rowRepeatParams_);
    }

    __aicore__ inline void ComputeAccumulatorDecay(const TurboQuantTaskHeads &heads)
    {
        if (heads.mine == 0) {
            return;
        }
        ascend_micro::vector_cube_decay_accumulator(accBuf_.Get<float>(), StateField(kStateAlpha),
                                                    StateField(kStateRunMax), StateField(kStateNewMax), heads.mine,
                                                    headSize_, rowRepeatParams_);
    }

    template <uint32_t PROBABILITY_WINDOW = 0>
    __aicore__ inline void ComputeAccumulate(const TurboQuantTaskHeads &heads)
    {
        if (heads.mine == 0) {
            return;
        }
        static_assert(PROBABILITY_WINDOW < 2);
        constexpr uint32_t probabilityScale = PROBABILITY_WINDOW == 1 ? kStatePart : kStateProbScale;
        ascend_micro::vector_cube_accumulate(accBuf_.Get<float>(), contextBuf_.Get<float>(),
                                             StateField(probabilityScale), StateField(kStateAccInv), heads.mine,
                                             headSize_, rowRepeatParams_);
    }

    template <typename Finisher>
    __aicore__ inline void StageTaskOutput(const uint32_t token, const uint32_t split, const TurboQuantTaskHeads &heads,
                                           const bool fused, Finisher &finisher)
    {
        if (heads.mine == 0) {
            return;
        }
        if (fused) {
            const AscendC::LocalTensor<float> acc       = accBuf_.Get<float>();
            const AscendC::LocalTensor<float> reduce    = reduceBuf_.Get<float>();
            const uint32_t                    firstHead = heads.first + heads.base;
            NormalizeHeads(acc, StateField(kStateRunSum), heads.mine, headSize_, reduce[kReduceNormalSums],
                           reduce[kReduceNormalInvs]);
            finisher.FinishHeads(acc, token, firstHead, heads.mine);
            WriteHeads<scalar_t>(acc, heads.mine, headSize_, outBuf_.Get<scalar_t>(), outputGm_,
                                 (static_cast<uint64_t>(token) * numHeads_ + firstHead) * headSize_);
            WriteProbeRecords(token, firstHead, heads.mine);
            return;
        }
        StagePartials(token, split, heads);
    }

    __aicore__ inline void WriteNeutralPartial(const uint32_t token, const uint32_t split,
                                               const TurboQuantTaskHeads &heads)
    {
        if ASCEND_IS_AIV {
            if (heads.mine == 0) return;
            BeginTask(heads);
            AscendC::Duplicate(StateField(kStateRunMax), kNeutralPartialMax, heads.mine * kFp32PerBlock);
            AscendC::Duplicate(StateField(kStateRunSum), 0.0f, heads.mine * kFp32PerBlock);
            StagePartials(token, split, heads);
        }
    }

private:
    static constexpr uint32_t kHalfRows = kCubeTileM / 2;

    static constexpr uint32_t kScoreBytes = kHalfRows * kCubeTileRows * sizeof(float);
    static_assert(kScoreBytes % (kFp32PerBlock * sizeof(float)) == 0,
                  "the context buffer has to start on a UB block, or the Fixpipe window is misaligned");
    static constexpr uint32_t kUnpackChunkRows   = kCubeTileRows / 2;
    static constexpr uint32_t kKeyPlane          = 0;
    static constexpr uint32_t kValuePlane        = 1;
    static constexpr float    kMaskGain          = -5.0e36f;
    static constexpr float    kMaskMidpoint      = 0.5f;
    static constexpr float    kNeutralPartialMax = -1.0e9f;

    static constexpr uint32_t kStateRunMax    = 0;
    static constexpr uint32_t kStateRunSum    = 1;
    static constexpr uint32_t kStateTileMax   = 2;
    static constexpr uint32_t kStateNewMax    = 3;
    static constexpr uint32_t kStateAlpha     = 4;
    static constexpr uint32_t kStateProbScale = 5;
    static constexpr uint32_t kStateAmax      = 6;
    static constexpr uint32_t kStateQScale    = 7;

    static constexpr uint32_t kStateQInv      = 8;
    static constexpr uint32_t kStateRowReduce = 9;
    static constexpr uint32_t kStatePart      = 10;
    static constexpr uint32_t kStateAccInv    = 11;
    static constexpr uint32_t kStateFields    = 12;
    static constexpr uint32_t kStateField     = kHalfRows * kFp32PerBlock;

    static constexpr uint32_t kPairProbabilityPanelStride = 4 * kCubeTileRows;
    static constexpr uint32_t kPairProbabilityPanelBytes = ascend_micro::kRegisterSoftmaxHeads * kCubeTileRows;
    static_assert(sizeof(fp8_e4m3fn_t) == 1);
    static_assert(kPairProbabilityPanelStride % 32 == 0);
    static_assert(kPairProbabilityPanelBytes <= kPairProbabilityPanelStride);
    static_assert(kPairProbabilityPanelStride + kPairProbabilityPanelBytes <=
                  ascend_micro::fused_decode_opt::kUbSizes[20]);

    static_assert(ascend_micro::kRegisterSoftmaxHeads * kFp32PerBlock <= kStateField);
    static_assert(kStateProbScale < kStatePart && kStatePart < kStateAccInv);

    static constexpr uint8_t  kTileBlocks     = static_cast<uint8_t>(kCubeTileRows / kFp32PerBlock);

    static constexpr uint32_t kReduceFloats     = 512;
    static constexpr uint32_t kReduceKeyScale   = 0;
    static constexpr uint32_t kReduceValueScale = kCubeTileRows;
    static constexpr uint32_t kReducePairFirstValueScale = 2 * kCubeTileRows;
    static constexpr uint32_t kReducePairSecondValueScale = 7 * kCubeTileRows;
    static constexpr uint32_t kReduceNormalSums = 3 * kCubeTileRows + kFp32PerBlock;
    static constexpr uint32_t kReduceNormalInvs = kReduceNormalSums + 2 * kCubeTileRows;

    static constexpr uint32_t kPairSecondKeyScale = kReduceValueScale;
    static_assert(kReduceKeyScale + kCubeTileRows <= kPairSecondKeyScale);
    static_assert(kPairSecondKeyScale + kCubeTileRows <= kReduceNormalSums);
    static_assert(kPairSecondKeyScale + kCubeTileRows <= kReducePairFirstValueScale);

    template <uint32_t WINDOW>
    __aicore__ inline AscendC::LocalTensor<float> PairKeyScale()
    {
        static_assert(WINDOW < 2);
        constexpr uint32_t offset = WINDOW == 0 ? kReduceKeyScale : kPairSecondKeyScale;
        return reduceBuf_.Get<float>()[offset];
    }

    static_assert(kReduceValueScale + kCubeTileRows <= kReduceNormalSums,
                  "the tile scales must not reach the normalisation broadcast");
    static_assert(kReduceValueScale + kCubeTileRows == kReducePairFirstValueScale);
    static_assert(kReducePairFirstValueScale + kCubeTileRows <= kReduceNormalSums);
    static_assert(kReduceNormalInvs + kHalfRows * kFp32PerBlock <= kReducePairSecondValueScale);
    static_assert(kReducePairSecondValueScale + kCubeTileRows == kReduceFloats);
    static_assert(kReduceNormalSums + (kHalfRows - 1) * kFp32PerBlock + kBrcbDstLanes <= kReduceNormalInvs,
                  "NormalizeHeads' broadcast must not reach its own reciprocals");
    static_assert(kReduceNormalInvs + kHalfRows * kFp32PerBlock <= kReduceFloats,
                  "the reciprocals must not run off the end of the buffer");

    static constexpr uint32_t kProbePos      = 0;
    static constexpr uint32_t kProbePage     = 64;
    static constexpr uint32_t kProbeRamp     = 128;
    static constexpr uint32_t kProbeBc       = 192;
    static constexpr uint32_t kProbeDiff     = 256;
    static constexpr uint32_t kProbeSel      = 320;
    static constexpr uint32_t kProbeStage    = 384;
    static constexpr uint32_t kProbePred = TelemetryPredicateLayout::kPosition;
    static constexpr uint32_t kProbePred2 = TelemetryPredicateLayout::kPage;
    static constexpr uint32_t kProbeCandPos  = 408;
    static constexpr uint32_t kProbeCandPage = 472;
    static constexpr uint32_t kProbeArgmaxPred = TelemetryPredicateLayout::kArgmax;
    static constexpr uint32_t kProbeArenaFloats = TelemetryPredicateLayout::kArenaFloats;
    static_assert(kProbeCandPage + kHalfRows * kFp32PerBlock <= kProbeArgmaxPred);

    static constexpr uint32_t kPairMaximum = 0;
    static constexpr uint32_t kPairAlpha = ascend_micro::kRegisterSoftmaxHeads * kFp32PerBlock;
    static constexpr uint32_t kPairStride = 2 * kPairAlpha;
    static_assert(kProbeRamp + kCubeTileRows <= kProbeBc);
    static_assert(kProbeBc + 2 * kPairStride <= kProbeSel);

    __aicore__ inline AscendC::LocalTensor<float> PairState(uint32_t window, uint32_t field)
    {
        return queryInBuf_.Get<float>()[kProbeBc + window * kPairStride + field];
    }

    __aicore__ inline void ComputeLayout(const uint32_t numHeads, const uint32_t numKvHeads, const uint32_t headSize,
                                         const uint32_t blockSize, const uint32_t numSplits, const float scale)
    {
        numHeads_         = numHeads;
        numKvHeads_       = numKvHeads;
        headSize_         = headSize;
        blockSize_        = blockSize;
        numSplits_        = numSplits;
        packedBytes_      = Codec::PackedBytes(headSize);
        scaleSlot_        = ScaleSlotFloats(numKvHeads_);
        partialStride_    = headSize + kPartialTail;
        operandElems_     = Mm::OperandElems(headSize_);
        tilePlaneBytes_   = kCubeTileRows * packedBytes_;
        unpackChunkBytes_ = kUnpackChunkRows * packedBytes_;
        nzTileToL1Params_ =
            AscendC::DataCopyParams{1, static_cast<uint16_t>(kCubeTileRows * operandElems_ / kOperandC0), 0, 0};
        queryRowToL1Params_ = AscendC::DataCopyParams{static_cast<uint16_t>(operandElems_ / kOperandC0), 1, 0,
                                                      static_cast<uint16_t>(kCubeTileM - 1)};
        probRowToL1Params_  = AscendC::DataCopyParams{
            static_cast<uint16_t>(Mm::OperandElems(kCubeTileRows) / kOperandC0), 1, 0,
            static_cast<uint16_t>(kCubeTileM - 1)};
        const uint8_t rowBlocks = static_cast<uint8_t>(headSize_ / kFp32PerBlock);
        rowRepeatParams_        = AscendC::BinaryRepeatParams{1, 1, 0, rowBlocks, rowBlocks, 1};
        tileByBlockParams_      = AscendC::BinaryRepeatParams{1, 1, 0, kTileBlocks, kTileBlocks, 1};
        tileByVectorParams_     = AscendC::BinaryRepeatParams{1, 1, 1, kTileBlocks, kTileBlocks, 0};

        scoreScale_ = scale;
        invGain_    = 1.0f;
    }

    __aicore__ static constexpr uint32_t OperandIndex(const uint32_t byteOffset)
    {
        return Mm::kIsFp4 ? byteOffset * 2 : byteOffset;
    }

    __aicore__ inline void InitGlobalTensors(__gm__ void *queryRot, __gm__ void *keyCache, __gm__ void *valueCache,
                                             __gm__ void *scaleCache, __gm__ void *modeTables, __gm__ void *workspace,
                                             __gm__ void *output)
    {
        queryRotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(queryRot));
        keyCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(keyCache));
        valueCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(valueCache));
        scaleCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scaleCache));
        modeTablesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(modeTables));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(output));
    }

    __aicore__ inline void InitBuffers(AscendC::TPipe *pipe)
    {
        static_assert(RING_SLOTS == 3 && SCORE_WINDOWS == 2 && BANK_SKEW == 32);
        static_assert(kStateFields * kStateField * sizeof(float) == ascend_micro::fused_decode_opt::kUbSizes[21]);
        static_assert(kReduceFloats * sizeof(float) == ascend_micro::fused_decode_opt::kUbSizes[22]);
        static_assert(kProbeArenaFloats * sizeof(float) <= ascend_micro::fused_decode_opt::kUbSizes[18],
                      "the telemetry probe arena must fit queryInBuf_, the dead buffer it is "
                      "carved from; a larger arena moves the UB map and must go to "
                      "turboquant_fused_decode_opt.hpp first");

        const auto *sizes = ascend_micro::fused_decode_opt::kUbSizes;
        for (uint32_t window = 0; window < kScoreWindows; ++window) pipe->InitBuffer(scoreBuf_[window], sizes[window]);
        pipe->InitBuffer(contextBuf_, sizes[2]);
        for (uint32_t slot = 0; slot < kIngestSlots; ++slot) {
            pipe->InitBuffer(kvBuf_[slot], sizes[3 + 2 * slot]);
            SkewBank(pipe, slot);
        }
        for (uint32_t slot = 0; slot < kIngestSlots; ++slot) {
            pipe->InitBuffer(scaleTileBuf_[slot], sizes[9 + 2 * slot]);
            SkewBank(pipe, kIngestSlots + slot);
        }

        pipe->InitBufPool(transientPool_, kTransientPoolBytes);
        transientPool_.InitBuffer(nzOperandBuf_, sizes[15]);
        transientPool_.InitBuffer(accBuf_, sizes[16]);
        transientPool_.InitBuffer(queryWorkBuf_, sizes[17]);
        pipe->InitBuffer(queryInBuf_, sizes[18]);
        pipe->InitBuffer(queryOperandBuf_, sizes[19]);

        pipe->InitBuffer(probabilitySkewBuf_, 32);
        pipe->InitBuffer(probOperandBuf_, sizes[20]);
        pipe->InitBuffer(stateBuf_, sizes[21]);
        pipe->InitBuffer(reduceBuf_, sizes[22]);
        pipe->InitBuffer(scaleIndexBuf_, sizes[23]);
        pipe->InitBuffer(scratchBuf_, sizes[24]);
        pipe->InitBuffer(outBuf_, sizes[25]);
        pipe->InitBuffer(maskBuf_, sizes[26]);
    }

    __aicore__ inline void SkewBank(AscendC::TPipe *pipe, const uint32_t pad)
    {
        if constexpr (BANK_SKEW > 0) {
            pipe->InitBuffer(bankSkewBuf_[pad], BANK_SKEW);
        }
    }

    __aicore__ inline void InitIngestEvents()
    {
        if ASCEND_IS_AIV {
            for (uint32_t slot = 0; slot < kIngestSlots; ++slot) {
                readEvents_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
            }
        }
    }

    __aicore__ inline AscendC::LocalTensor<float> StateField(const uint32_t field)
    {
        return stateBuf_.Get<float>()[field * kStateField];
    }

    __aicore__ inline void BuildScaleIndex(const uint32_t kvHead)
    {
        const int32_t                       slotBytes  = static_cast<int32_t>(scaleSlot_ * sizeof(float));
        const AscendC::LocalTensor<int32_t> scaleIndex = scaleIndexBuf_.Get<int32_t>();
        AscendC::ArithProgression(scaleIndex, static_cast<int32_t>(kvHead * sizeof(float)), slotBytes,
                                  static_cast<int32_t>(kCubeTileRows));
        AscendC::ArithProgression(scaleIndex[kCubeTileRows],
                                  static_cast<int32_t>((numKvHeads_ + kvHead) * sizeof(float)), slotBytes,
                                  static_cast<int32_t>(kCubeTileRows));
    }

    __aicore__ static inline uint32_t Plane() { return AscendC::GetSubBlockIdx() == 0 ? kKeyPlane : kValuePlane; }

    __aicore__ inline AscendC::LocalTensor<OperandT> PlaneOperand(Mm &mm, const uint32_t slot)
    {
        return Plane() == kKeyPlane ? mm.L1Key(slot) : mm.L1Value(slot);
    }

    template <bool PACKED, bool SCALES>
    __aicore__ inline void ReadPlane(const uint32_t physical, const uint32_t rowBase, const uint32_t kvHead,
                                     const uint32_t slot)
    {
        const uint64_t                 row     = static_cast<uint64_t>(physical) * blockSize_ + rowBase;
        if constexpr (PACKED) {
            const uint64_t tile = (row * numKvHeads_ +
                                   static_cast<uint64_t>(kvHead) * kCubeTileRows) * packedBytes_;
            AscendC::GlobalTensor<int8_t> &cacheGm = Plane() == kKeyPlane ? keyCacheGm_ : valueCacheGm_;
            AscendC::DataCopy(kvBuf_[slot].template Get<int8_t>(), cacheGm[tile], tilePlaneBytes_);
        }
        if constexpr (SCALES) {
            AscendC::DataCopy(scaleTileBuf_[slot].template Get<float>(), scaleCacheGm_[row * scaleSlot_],
                              kCubeTileRows * scaleSlot_);
        }
    }

    __aicore__ inline void StageTileToL1(const AscendC::LocalTensor<int8_t>   &packed,
                                         const AscendC::LocalTensor<OperandT> &l1Dst)
    {
        const AscendC::LocalTensor<OperandT> nzLow  = nzOperandBuf_.Get<OperandT>();
        const AscendC::LocalTensor<OperandT> nzHigh = nzLow[tilePlaneBytes_];
        for (uint32_t base = 0; base < tilePlaneBytes_; base += unpackChunkBytes_) {
            if constexpr (Codec::kGatherLut) {
                codec_.template UnpackAffineGather<OperandT>(nzLow[base], nzHigh[base], packed[base],
                                                             unpackChunkBytes_);
            } else {
                codec_.template UnpackAffine<OperandT>(nzLow[base], nzHigh[base], packed[base],
                                                       unpackChunkBytes_);
            }
        }
        SyncVectorToMte3();
        AscendC::DataCopy(l1Dst, nzLow, nzTileToL1Params_);
        SyncMte3ToVector();
    }

    __aicore__ inline void RunTileSoftmax(const AscendC::LocalTensor<float> &scores, const uint32_t valid,
                                          const uint32_t heads, const AscendC::LocalTensor<float> &keyScale,
                                          const AscendC::LocalTensor<float> &valueScale,
                                          const uint32_t contextBase, const uint32_t logicalPage)
    {
        const AscendC::LocalTensor<float> arena = queryInBuf_.Get<float>();
        ascend_micro::NativeCubeSoftmaxEngine<Codec, OperandT> engine{scores,
                                                                     keyScale,
                                                                     valueScale,
                                                                     StateField(kStateQInv),
                                                                     maskBuf_.Get<float>(),
                                                                     StateField(kStateRunMax),
                                                                     StateField(kStateRunSum),
                                                                     StateField(kStateTileMax),
                                                                     StateField(kStateNewMax),
                                                                     StateField(kStateAlpha),
                                                                     StateField(kStateProbScale),
                                                                     StateField(kStatePart),
                                                                     StateField(kStateRowReduce),
                                                                     probOperandBuf_.Get<OperandT>(),
                                                                     codec_,
                                                                     heads,
                                                                     valid,
                                                                     OperandMax<MODE>(),
                                                                     TurboQuantCodec4::kEps,
                                                                     tileByVectorParams_,
                                                                     tileByBlockParams_,
                                                                     arena[kProbePos],
                                                                     arena[kProbePage],
                                                                     arena[kProbeRamp],
                                                                     arena[kProbeBc],
                                                                     arena[kProbeDiff],
                                                                     arena[kProbeSel],
                                                                     arena[kProbeStage],
                                                                     arena[kProbeCandPos],
                                                                     arena[kProbeCandPage],
                                                                     arena[kProbeArgmaxPred],
                                                                     contextBase,
                                                                     logicalPage};
        ascend_micro::vector_online_softmax_probabilities_probed(engine);
        engine.quantize_probabilities();
    }

    __aicore__ inline uint64_t PartialOffset(const uint32_t token, const uint32_t head, const uint32_t split) const
    {
        return ((static_cast<uint64_t>(token) * numHeads_ + head) * numSplits_ + split) * partialStride_;
    }

    __aicore__ inline void StagePartials(const uint32_t token, const uint32_t split, const TurboQuantTaskHeads &heads)
    {
        const AscendC::LocalTensor<float> acc   = accBuf_.Get<float>();
        const AscendC::LocalTensor<float> tails = Scores(0);
        ascend_micro::vector_cube_pack_partial_tails(tails, StateField(kStateRunMax), StateField(kStateRunSum),
                                                     heads.mine, kPartialTail, kPartialMaxLane, kPartialSumLane);
#if defined(ASCEND_TQ_DECODE_TELEMETRY) && ASCEND_TQ_DECODE_TELEMETRY

        const AscendC::LocalTensor<float>   arena = queryInBuf_.Get<float>();
        const AscendC::LocalTensor<uint8_t> lane1 = arena[kProbePred].ReinterpretCast<uint8_t>();
        const AscendC::LocalTensor<uint8_t> lane2 = arena[kProbePred2].ReinterpretCast<uint8_t>();
        for (uint32_t head = 0; head < heads.mine; ++head) {
            AscendC::Brcb(arena[kProbeBc], arena[kProbePos + head * kFp32PerBlock], 1, {1, 8});
            AscendC::Select(tails[head * kPartialTail], lane1, arena[kProbeBc],
                            tails[head * kPartialTail], AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                            static_cast<uint32_t>(kPartialTail));
            AscendC::Brcb(arena[kProbeBc], arena[kProbePage + head * kFp32PerBlock], 1, {1, 8});
            AscendC::Select(tails[head * kPartialTail], lane2, arena[kProbeBc],
                            tails[head * kPartialTail], AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                            static_cast<uint32_t>(kPartialTail));
        }
#endif
        SyncVectorToMte3();
        for (uint32_t head = 0; head < heads.mine; ++head) {
            const uint64_t offset = PartialOffset(token, heads.first + heads.base + head, split);
            AscendC::DataCopy(workspaceGm_[offset], acc[head * headSize_], headSize_);
            AscendC::DataCopy(workspaceGm_[offset + headSize_], tails[head * kPartialTail], kPartialTail);
        }
        SyncMte3ToVector();
    }

    __aicore__ inline void WriteProbeRecords(const uint32_t token, const uint32_t firstHead,
                                             const uint32_t heads)
    {
#if defined(ASCEND_TQ_DECODE_TELEMETRY) && ASCEND_TQ_DECODE_TELEMETRY
        const AscendC::LocalTensor<float>   arena  = queryInBuf_.Get<float>();
        const AscendC::LocalTensor<float>   record = Scores(0);
        const AscendC::LocalTensor<float>   runMax = StateField(kStateRunMax);
        const AscendC::LocalTensor<uint8_t> lane1  = arena[kProbePred].ReinterpretCast<uint8_t>();
        const AscendC::LocalTensor<uint8_t> lane2  = arena[kProbePred2].ReinterpretCast<uint8_t>();
        for (uint32_t head = 0; head < heads; ++head) {
            const uint32_t at = head * kPartialTail;
            AscendC::Duplicate(record[at], 0.0f, static_cast<uint32_t>(kPartialTail));
            AscendC::Adds(record[at + kTelemetryMaxLane], runMax[head * kFp32PerBlock], 0.0f, 1);
            AscendC::Brcb(arena[kProbeBc], arena[kProbePos + head * kFp32PerBlock], 1, {1, 8});
            AscendC::Select(record[at], lane1, arena[kProbeBc], record[at],
                            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                            static_cast<uint32_t>(kPartialTail));
            AscendC::Brcb(arena[kProbeBc], arena[kProbePage + head * kFp32PerBlock], 1, {1, 8});
            AscendC::Select(record[at], lane2, arena[kProbeBc], record[at],
                            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                            static_cast<uint32_t>(kPartialTail));
        }
        SyncVectorToMte3();
        for (uint32_t head = 0; head < heads; ++head) {
            AscendC::DataCopy(workspaceGm_[PartialOffset(token, firstHead + head, 0) + headSize_],
                              record[head * kPartialTail], static_cast<uint32_t>(kPartialTail));
        }
        SyncMte3ToVector();
#else
        static_cast<void>(token);
        static_cast<void>(firstHead);
        static_cast<void>(heads);
#endif
    }

    Codec                                        codec_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> kvBuf_[kIngestSlots];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleTileBuf_[kIngestSlots];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bankSkewBuf_[BANK_SKEW > 0 ? 2 * kIngestSlots : 1];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> nzOperandBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> queryWorkBuf_;

    AscendC::TBufPool<AscendC::TPosition::VECCALC> transientPool_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> queryInBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> queryOperandBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scoreBuf_[kScoreWindows];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> probabilitySkewBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> probOperandBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> contextBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stateBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleIndexBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scratchBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> maskBuf_;
    AscendC::GlobalTensor<float>                 queryRotGm_;
    AscendC::GlobalTensor<int8_t>                keyCacheGm_;
    AscendC::GlobalTensor<int8_t>                valueCacheGm_;
    AscendC::GlobalTensor<float>                 scaleCacheGm_;
    AscendC::GlobalTensor<int32_t>               modeTablesGm_;
    AscendC::GlobalTensor<float>                 workspaceGm_;
    AscendC::GlobalTensor<scalar_t>              outputGm_;
    AscendC::DataCopyParams                      nzTileToL1Params_;
    AscendC::DataCopyParams                      queryRowToL1Params_;
    AscendC::DataCopyParams                      probRowToL1Params_;
    AscendC::BinaryRepeatParams                  rowRepeatParams_;
    AscendC::BinaryRepeatParams                  tileByBlockParams_;
    AscendC::BinaryRepeatParams                  tileByVectorParams_;
    AscendC::TEventID                            readEvents_[kIngestSlots] = {};
    float                                        scoreScale_       = 0.0f;
    float                                        invGain_          = 1.0f;
    uint32_t                                     numHeads_         = 0;
    uint32_t                                     numKvHeads_       = 0;
    uint32_t                                     headSize_         = 0;
    uint32_t                                     blockSize_        = 0;
    uint32_t                                     numSplits_        = 1;
    uint32_t                                     packedBytes_      = 0;
    uint32_t                                     scaleSlot_        = 0;
    uint32_t                                     partialStride_    = 0;
    uint32_t                                     operandElems_     = 0;
    uint32_t                                     tilePlaneBytes_   = 0;
    uint32_t                                     unpackChunkBytes_ = 0;
};

}
}

#endif
