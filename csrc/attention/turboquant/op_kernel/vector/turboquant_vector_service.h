/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Vector-unit services of the TurboQuant decode.
//
//   NormalizeHeads / WriteHeads    online softmax close-out: 1 / (sum + eps), then cast and direct GM write;
//                                  WriteNormalizedHeads is the two back to back
//   TurboQuantRotatedOutput        the output stage of a decode that writes the rotated basis: nothing
//   TurboQuantPartialReducer       the in-launch reduction of per-split partials (context > 4096)
//   TurboQuantVectorDecodeService  the AIV half of the Cube decode: query and KV staging into L1,
//                                  the online softmax over the Cube's score rows, the accumulation
//
// EVERY PIECE OF ARITHMETIC HERE IS A CALL INTO ascend_micro. This file owns buffers, DMA, hard
// events and the GM ABI; the routine_*.hpp leaf routines own the vector instructions and may not
// touch an event or a flag. That split is what lets the recurrence be audited on the host
// (tests/cpu/test_contract_gates.cpp drives the engines with a duck-typed phase recorder) while
// the pipeline around it is only meaningful on a device.
//
// The pipeline ITSELF lives in turboquant_fused_decode.cpp, which schedules these calls as a
// prologue, a modulo-scheduled steady state and a drain. The public surface below is grouped by
// the stage it serves so the two files read against each other.

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_VECTOR_SERVICE_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_VECTOR_SERVICE_H

#include "../common/turboquant_codec_950.h"
#include "../common/turboquant_common.h"
#include "../common/turboquant_mode.h"
#include "routine_vector_cube_softmax.hpp"
#include "../../include/turboquant_fused_decode_opt.hpp"

namespace vllm_ascend {
namespace turboquant {

constexpr float kVectorNegInf = -3.4028235e38f;

// ----------------------------------------------------------------------------------------------------
// Online softmax close-out
// ----------------------------------------------------------------------------------------------------

// Normalises `heads` accumulators by 1 / (runSum + eps) in place. runSum holds one kFp32PerBlock lane per
// head; sums and invs need heads * kFp32PerBlock floats each. Requires headSize to be a multiple of
// kFp32PerRepeat.
//
// A shim over ascend_micro::vector_cube_normalize_heads, kept at this signature because
// cube/turboquant_gemm_reduce_service.h re-exports the name and static_asserts the Brcb footprint
// of the `sums` operand against its own UB lane map. The routine's own comment states that
// footprint; do not narrow either broadcast operand on either side of this call.
__aicore__ inline void NormalizeHeads(const AscendC::LocalTensor<float> &acc, const AscendC::LocalTensor<float> &runSum,
                                      const uint32_t heads, const uint32_t headSize,
                                      const AscendC::LocalTensor<float> &sums, const AscendC::LocalTensor<float> &invs)
{
    ascend_micro::vector_cube_normalize_heads(acc, runSum, sums, invs, heads, headSize, TurboQuantCodec4::kEps);
}

// Casts `heads` fp32 rows to the output type in UB and writes them to GM in one burst.
//
// THE OPERATOR'S ONLY OUTPUT ROUNDING, and on the pre-rotated path the only place the element type
// is used for arithmetic at all: the query arrives as fp32 (the rotated image), the scores, the
// online-softmax recurrence, the probabilities, the accumulator and the Split-K merge are all fp32,
// and the result meets the element type exactly here. `float -> half` and `float -> bfloat16_t` are
// both listed CAST_RINT pairs on 950PR (type_conversion/Cast.md), so the same call serves both
// instances and the bf16 instance differs from the fp16 one by this rounding and nothing else.
// bf16's 8 significand bits against fp16's 11 make the step coarser, not the arithmetic different.
//
// Not a leaf routine, and deliberately not one: it owns the V -> MTE3 and MTE3 -> V hand-offs
// around its GM burst, which is exactly what a routine_*.hpp routine may not do.
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

// The output stage of a decode whose output stays in the rotated basis. A finisher is handed the
// normalised fp32 rows of `heads` heads of `token`, starting at `firstHead`, before they are cast and written.
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

// ----------------------------------------------------------------------------------------------------
// In-launch reduction of split partials
// ----------------------------------------------------------------------------------------------------

// A token whose context exceeds the fused limit is decoded in splits, each leaving
// [acc (headSize), runMax lane, runSum lane] in the workspace. This merges a (token, head)'s splits
// with the same online-softmax recurrence the split used and writes the output token. Runs in the
// same launch as the splits, after AscendC::SyncAll.
template <typename scalar_t>
class TurboQuantPartialReducer {
public:
    // The operator is instantiated at half and at bfloat16_t. Both are 16-bit and the UB map in
    // ../include/turboquant_fused_decode_opt.hpp is pinned on that: a third element type has to go
    // there first, not arrive here and silently move three cores' Fixpipe landings apart.
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
        // Init boundary: drains every pipe before the reduction's first GM read of the split partials.
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Reduce(const uint32_t token, const uint32_t head)
    {
        TurboQuantRotatedOutput rotated;
        Reduce(token, head, rotated);
    }

    // `finisher` sees the merged, normalised head before it is cast and written (TurboQuantRotatedOutput).
    //
    // The PipeBarrier<PIPE_V> below is an overlapping write and not a hand-off: the runMax lane lies
    // inside the state block the Duplicate above it has just zeroed.
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
    // The merge's state block, one lane per field: the running pair the partials fold into, the
    // new maximum, and the two exponential weights.
    static constexpr uint32_t kMergeNewMaxLane = 2 * kFp32PerBlock;
    static constexpr uint32_t kMergeAlphaLane  = 3 * kFp32PerBlock;
    static constexpr uint32_t kMergeBetaLane   = 4 * kFp32PerBlock;
    static constexpr uint32_t kMergeStateLanes = 5;
    // The two Brcb destinations, then NormalizeHeads' own pair.
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

// ----------------------------------------------------------------------------------------------------
// The vector half of the Cube decode
// ----------------------------------------------------------------------------------------------------

template <TurboQuantMode MODE>
__aicore__ inline constexpr float OperandMax()
{
    return TurboQuantModeTraits<MODE>::kOperand == TurboQuantOperand::kFp4E2m1 ? 6.0f : 448.0f;
}

// The ring-buffered tile ingest and the head-batched softmax, validated on kv4fp8 only. kv4fp8
// batches even one head per subcore: a per-head path's Level-2 reduces cost more than the row
// repeats. No other mode has a verified ingest here -- see the class static_asserts.
template <TurboQuantMode MODE>
constexpr bool kBatchedCubeDecode = MODE == TurboQuantMode::KV4_FP8;

// The heads one task covers, and the half of them this vector subcore owns. The Cube's dual-destination
// Fixpipe splits M rows in half, so subcore 0 owns rows [0, ceil(M/2)) and subcore 1 the rest.
struct TurboQuantTaskHeads {
    uint32_t first = 0;
    uint32_t rows  = 0;
    uint32_t base  = 0;
    uint32_t mine  = 0;
};

// Mm is the Cube BMM service (TurboQuantCubeMm<MODE>) that owns the L1 operand slots, and Codec the
// mode's KV codec. Subcore 0 stages the K plane of every tile and subcore 1 the V plane. The cache is
// NZ-tiled: one (tile, kv head) is read per burst and unpacked straight into NZ order.
//
// The decode keeps every per-head scalar of the softmax broadcast over its head's 8-lane block: one
// wide op advances all heads, the same field is the block operand of a row op, and lane j * 8 still
// holds head j's value for the per-task readers.
//
// THIS SERVICE IS PINNED TO ONE SHAPE, and the static_asserts below say so rather than carrying
// branches nothing instantiates. turboquant_fused_decode.cpp declares KV4_FP8 / bfloat16_t
// kernels only, and both host launchers refuse any other mode before the grid is sized. Two
// expands are admitted: kNative (the shipping affine code cast) and kLloydMaxLut (the same tile
// path with the codec's 16-entry Gather expand, decoding the Lloyd-Max codebook the cache was
// written on). They differ ONLY inside StageTileToL1, which dispatches on the codec's kGatherLut
// -- an `if constexpr` on the codec, not a branch on the tile path. The kGatherLut pricing leg
// and the bypass ablation stay uninstantiated: an unreachable path on a kernel this hard to test
// reads as verified and is not. A fifth mode or another expand comes back the same way -- a
// routine plus a static_assert that admits it, not a branch nobody runs.
//
// RING_SLOTS, SCORE_WINDOWS and BANK_SKEW are the pipeline's shape. The modulo-scheduled decode
// takes kCubePipelineSlots ingest slots, kCubeScoreWindows score windows so the Cube's next score
// GEMM does not wait on this tile's softmax, and kBankSkewBytes between the ingest buffers.
// Everything that is per-slot is indexed by slot; what the window count adds is a second Fixpipe
// destination for the scores.
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
    // scalar_t is the OUTPUT element type only. The Cube operands are OperandT, which comes from MODE
    // (KV4/FP8) and not from scalar_t, so the ring ingest, the dequant expand, the score and context
    // GEMMs and every tile descriptor are identical in the half and bfloat16_t instances. The width,
    // though, is pinned: see ../include/turboquant_fused_decode_opt.hpp.
    static_assert(sizeof(scalar_t) == 2,
                  "turboquant_fused_decode_opt.hpp's UB map assumes a 16-bit output element");
    // Ring slots of the tile ingest. Tile t reads into slot t % kIngestSlots, the slot of its L1 operands.
    static constexpr uint32_t kIngestSlots = RING_SLOTS;
    // Fixpipe score windows. Tile t's scores land in window t % kScoreWindows.
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

    // The attention phase's transient UB, as a pool the reduction phase can alias onto. Handing
    // out a reference rather than a size is what keeps the two in step: the reduction asks the
    // pool how long it is, so shrinking the pool is a compile error at the reduction's own
    // static_assert rather than an overlap nobody notices. See InitBuffers.
    __aicore__ inline AscendC::TBufPool<AscendC::TPosition::VECCALC> &TransientPool()
    {
        return transientPool_;
    }
    static constexpr uint32_t kTransientPoolBytes = ascend_micro::fused_decode_opt::kTransientPoolBytes;

    // The Cube's two Fixpipe destinations. The AIC names them by the offsets its own TPipe handed out, so
    // both halves of the launch have to allocate the same list in the same order -- see InitBuffers.
    __aicore__ inline AscendC::LocalTensor<float> Scores(const uint32_t window = 0)
    {
        return scoreBuf_[window].template Get<float>();
    }
    __aicore__ inline AscendC::LocalTensor<float> Context() { return contextBuf_.Get<float>(); }

    // Which of the task's heads this subcore owns. On the AIC every row is "rows" and nothing is mine.
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

    // ------------------------------------------------------------------------------------------------
    // PROLOGUE. Per-task state, the query operand into L1, the tail mask, and the ring priming reads.
    // The task queues query and the first packed read before preparing recurrence state. It joins
    // the query before staging its operand and arming telemetry in the same UB storage.
    // ------------------------------------------------------------------------------------------------

    // The PipeBarrier<PIPE_V> is an overlapping write and not a hand-off: the runMax field lies inside
    // the state the Duplicate above it has just zeroed.
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

    // Quantises this subcore's query heads onto the operand grid, stages them into the Cube's L1 query slot
    // in NZ order, one row per head, and indexes the task's kv head in the tile scale lanes.
    __aicore__ inline event_t PostQueryRead(const uint32_t token, const TurboQuantTaskHeads &heads)
    {
        if (heads.mine == 0) {
            return {};
        }
        AscendC::DataCopy(queryInBuf_.Get<float>(),
                          queryRotGm_[(static_cast<uint64_t>(token) * numHeads_ + heads.first + heads.base) *
                                      headSize_],
                          heads.mine * headSize_);
        // Slot reads use reserved IDs until StageQuery or cancellation joins this transient ID.
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

    // The mask a partial last tile is Min-ed with: 0-crossing at `valid`, +huge before, -huge after.
    __aicore__ inline void ComputeTailMask(const uint32_t valid)
    {
        const AscendC::LocalTensor<float> mask = maskBuf_.Get<float>();
        ascend_micro::vector_cube_tail_mask(mask, mask[kCubeTileRows].ReinterpretCast<int32_t>(), valid,
                                            kMaskMidpoint, kMaskGain);
        // DIRTY-DATA (D2): the ramp above is positive on the live lanes and negative on the dead
        // ones. Turn that sign into a bit predicate ONCE here, at the head of the task, so the
        // softmax can SELECT the dead rows away instead of relying on a Min whose NaN behaviour
        // the documentation does not define. The predicate lives in the mask buffer's own tail,
        // which is 32-byte aligned (256 bytes into the 512-byte buffer).
        AscendC::Compares(mask[kCubeTileRows].ReinterpretCast<uint8_t>(), mask, 0.0f,
                          AscendC::CMPMODE::GT, static_cast<uint32_t>(kCubeTileRows));
    }

    // Arm the attention telemetry probe for one task (ASCEND_TQ_DECODE_TELEMETRY). MUST be
    // called after StageQuery: the arena is carved out of queryInBuf_, whose last use in a task
    // is StageQuery's query staging, so the buffer is dead for the whole tile loop -- and is
    // overwritten by the NEXT task's StageQuery, which is why the arena is re-armed per task
    // rather than once at Init. Arms the running (position, page) pair with the sentinel, the
    // 0..63 lane index ramp the per-tile fold selects against, and the two single-lane
    // predicates that place the record into the statistics tail's dead lanes 1 and 2 (those
    // lanes are 4-byte aligned, so a vector write reaches them only through a Select; see
    // StagePartials). Also called for a task that will never enter the pipeline (zero context,
    // unmapped pages), so its StageTaskOutput still publishes the sentinel record.
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

    // The MTE2 half of the ring ingest: one tile's plane and scale lanes into ring slot `slot`, posted to
    // the vector unit on the slot's own event id so the stage can wait for exactly this read. The caller
    // orders it after the vector unit's last read of the slot's previous tile.
    template <bool PACKED = true, bool SCALES = true>
    __aicore__ inline void ReadTileToSlot(const uint32_t physical, const uint32_t rowBase, const uint32_t kvHead,
                                          const uint32_t slot)
    {
        static_assert(PACKED || SCALES);
        ReadPlane<PACKED, SCALES>(physical, rowBase, kvHead, slot);
        if constexpr (SCALES) {
            // The same MTE2 queue completes both halves before this event.
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(readEvents_[slot]);
        }
    }

    // The vector half: waits for slot `slot`'s read, unpacks it in UB in NZ order and stages it into the
    // slot's L1 operand in one MTE3 burst. `kvHead` selected a head out of a row-major row plane; an
    // NZ-tiled read already covers exactly one (tile, kv head), so the plane arrives selected.
    __aicore__ inline void StageSlotToL1(Mm &mm, const uint32_t kvHead, const uint32_t slot)
    {
        static_cast<void>(kvHead);
        const AscendC::LocalTensor<OperandT> l1Dst = PlaneOperand(mm, slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(readEvents_[slot]);
        StageTileToL1(kvBuf_[slot].template Get<int8_t>(), l1Dst);
    }

    // ------------------------------------------------------------------------------------------------
    // STEADY STATE. One phase is softmax, next tile's read, accumulator decay, next slot's stage,
    // accumulate -- six phases closing the slot ring against the window ring. The ingest pair above is
    // reused here; the cross-core flags that order a phase against the Cube live in the caller.
    // ------------------------------------------------------------------------------------------------

    // Scores for this subcore's heads arrived from the Cube: finish the logits, advance the running
    // max / log-sum per head, and stage the probability rows (on the operand grid) into L1. `slot` is the
    // ingest slot holding this tile's scale lanes.
    //
    // ONE BODY ON PURPOSE. Split into member helpers, the per-tile tensor views are re-derived in each
    // and the AIV issues more scalar instructions per tile. The ascend_micro calls below are leaf
    // routines taking already-derived views, so they inline without re-deriving anything.
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
        // Both FP8 panels and their scales must survive until their respective PV replays.
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

    // Decays the accumulators by alpha = exp(runMax - newMax) before this tile's context lands, and
    // commits the new running maximum in the same routine call.
    __aicore__ inline void ComputeAccumulatorDecay(const TurboQuantTaskHeads &heads)
    {
        if (heads.mine == 0) {
            return;
        }
        ascend_micro::vector_cube_decay_accumulator(accBuf_.Get<float>(), StateField(kStateAlpha),
                                                    StateField(kStateRunMax), StateField(kStateNewMax), heads.mine,
                                                    headSize_, rowRepeatParams_);
    }

    // The Cube's context rows are probs * V on the operand grid: undo each head's prob scale and add.
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

    // ------------------------------------------------------------------------------------------------
    // EPILOGUE. The drain's last three phases run the steady-state calls above with no further reads;
    // this is what closes the task once they are done.
    // ------------------------------------------------------------------------------------------------

    // A fused token writes its output straight to GM, after `finisher` has seen its normalised rows; a split
    // token leaves its partials for the in-launch reduction, which calls the finisher instead.
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

    // Synthetic Split-K network probe: publish a neutral partial for one split without
    // entering the page/tile pipeline. This uses the same production partial packing and
    // MTE3 publication as StageTaskOutput, so the Cube reducer reads the normal workspace ABI.
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
    // The Cube's score rows: the first half of the dual-destination Fixpipe window, and the offset the
    // context half starts at. See InitBuffers for why the two are pinned to the head of the UB pool.
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
    // The per-head reciprocals, the row-reduce outputs (one lane per head) and two working blocks.
    static constexpr uint32_t kStateQInv      = 8;
    static constexpr uint32_t kStateRowReduce = 9;
    static constexpr uint32_t kStatePart      = 10;
    static constexpr uint32_t kStateAccInv    = 11;
    static constexpr uint32_t kStateFields    = 12;
    static constexpr uint32_t kStateField     = kHalfRows * kFp32PerBlock;
    // Two prepared generations share the existing padded FP8 probability allocation.
    static constexpr uint32_t kPairProbabilityPanelStride = 4 * kCubeTileRows;
    static constexpr uint32_t kPairProbabilityPanelBytes = ascend_micro::kRegisterSoftmaxHeads * kCubeTileRows;
    static_assert(sizeof(fp8_e4m3fn_t) == 1);
    static_assert(kPairProbabilityPanelStride % 32 == 0);
    static_assert(kPairProbabilityPanelBytes <= kPairProbabilityPanelStride);
    static_assert(kPairProbabilityPanelStride + kPairProbabilityPanelBytes <=
                  ascend_micro::fused_decode_opt::kUbSizes[20]);
    // kStatePart is not produced by the prepared FP8 replay. Its separate
    // eight-lane blocks hold strip 1's scales until that strip accumulates.
    static_assert(ascend_micro::kRegisterSoftmaxHeads * kFp32PerBlock <= kStateField);
    static_assert(kStateProbScale < kStatePart && kStatePart < kStateAccInv);

    static constexpr uint8_t  kTileBlocks     = static_cast<uint8_t>(kCubeTileRows / kFp32PerBlock);

    // reduceBuf_'s lanes. The two tile scales are per-tile and live only across one softmax; the
    // normalisation pair is per-task and read once, at the epilogue. They are kept apart rather than
    // overlaid because the epilogue's Brcb destination is 64 floats wide whatever the head count --
    // see vector_cube_normalize_heads -- and these offsets are what the asserts below pin.
    static constexpr uint32_t kReduceFloats     = 512;
    static constexpr uint32_t kReduceKeyScale   = 0;
    static constexpr uint32_t kReduceValueScale = kCubeTileRows;
    static constexpr uint32_t kReducePairFirstValueScale = 2 * kCubeTileRows;
    static constexpr uint32_t kReducePairSecondValueScale = 7 * kCubeTileRows;
    static constexpr uint32_t kReduceNormalSums = 3 * kCubeTileRows + kFp32PerBlock;
    static constexpr uint32_t kReduceNormalInvs = kReduceNormalSums + 2 * kCubeTileRows;
    // The generic per-tile V-scale row is dead before the paired VF.
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

    // The telemetry probe arena, in floats of queryInBuf_. StageQuery is that buffer's only
    // other user and it runs before the tile loop, so the arena is dead exactly over the span
    // BeginArgmaxProbe .. StageTaskOutput of one task. Every operand is a whole number of
    // 32-byte blocks: vector reads and writes on this part are block-aligned, which is also why
    // the record's lanes 1 and 2 are written with a Select against a precomputed predicate
    // rather than addressed directly. Offsets are part of the kernel's UB contract the same way
    // the reduce lanes above are; kUbSizes[18] pins the capacity.
    static constexpr uint32_t kProbePos      = 0;    // 64 floats: running argmax position
    static constexpr uint32_t kProbePage     = 64;   // 64 floats: running logical page
    static constexpr uint32_t kProbeRamp     = 128;  // 64 floats: the 0..63 lane index ramp
    static constexpr uint32_t kProbeBc       = 192;  // 64 floats: per-head broadcast scratch
    static constexpr uint32_t kProbeDiff     = 256;  // 64 floats: row diff / improvement test
    static constexpr uint32_t kProbeSel      = 320;  // 64 floats: selected indices
    static constexpr uint32_t kProbeStage    = 384;  // 8 floats: MIN-reduce landing
    static constexpr uint32_t kProbePred = TelemetryPredicateLayout::kPosition;
    static constexpr uint32_t kProbePred2 = TelemetryPredicateLayout::kPage;
    static constexpr uint32_t kProbeCandPos  = 408;  // 64 floats: candidate position blocks
    static constexpr uint32_t kProbeCandPage = 472;  // 64 floats: candidate page blocks
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
        // The writers' dequantisation contract (turboquant_codec_mx.h): a row reconstructs as
        // code * stored_scale with no centroid offset and no codebook gain, so the score scale is
        // the bare attention scale and the value scale is 1.
        scoreScale_ = scale;
        invGain_    = 1.0f;
    }

    // LocalTensor<fp4x2> indexes individual 4-bit values, while all operand sizes and DMA
    // strides above are measured in bytes. Convert byte offsets before slicing FP4 tensors.
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

    // The UB layout is allocation order, so this order is part of the kernel's memory contract.
    //
    // The score and context buffers come first, and they come first on purpose. The two of them are the
    // Cube's dual-destination Fixpipe window: the AIC names them by the offsets its own TPipe handed out
    // and writes them into BOTH vector subcores' UB, so every core of the launch has to place them at the
    // same offsets, and nothing in the toolchain checks that. A UB buffer allocated on one core and not
    // the other -- here, in the codec, or in TurboQuantQueryBasis -- anywhere ahead of them on the shared
    // pipe would shift the destination on that core alone, and the decode would read another buffer's
    // bytes with nothing reporting an error. Allocated first they sit at UB offsets 0 and kScoreBytes
    // whatever else the list below grows, so the contract narrows to one rule: this service's InitBuffers
    // is the first UB allocation the pipe sees (TurboQuantCubeMm::Init runs before it but allocates in the
    // L1 and L0 pools only), and these two lines stay at the top of it.
    // The score windows are allocated as one run, so window w sits at UB offset w * kScoreBytes and the
    // context window right behind them, on both cores, whatever the window count is.
    //
    // AKA3003, the second rule of this list: with BANK_SKEW set, every buffer of the tile ingest is
    // followed by a kBankSkewBytes pad, so no two ring slots start on the same UB bank. MTE2 is filling
    // one slot while the vector ALU reads another throughout the steady state, and two slots that are an
    // exact multiple of the bank stride apart present the same bank to both ports. The pads sit after the
    // Fixpipe windows, never between them, so the dual-destination contract above is untouched.
    __aicore__ inline void InitBuffers(AscendC::TPipe *pipe)
    {
        static_assert(RING_SLOTS == 3 && SCORE_WINDOWS == 2 && BANK_SKEW == 32);
        static_assert(kStateFields * kStateField * sizeof(float) == ascend_micro::fused_decode_opt::kUbSizes[21]);
        static_assert(kReduceFloats * sizeof(float) == ascend_micro::fused_decode_opt::kUbSizes[22]);
        static_assert(kProbeArenaFloats * sizeof(float) <= ascend_micro::fused_decode_opt::kUbSizes[18],
                      "the telemetry probe arena must fit queryInBuf_, the dead buffer it is "
                      "carved from; a larger arena moves the UB map and must go to "
                      "turboquant_fused_decode_opt.hpp first");
        // The exact allocation list is shared with the host/compile-time budget.
        // The launch ABI gates D=256 and scaleSlot=8 before reaching this body.
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
        // THE TRANSIENT POOL, and the third rule of this list.
        //
        // These three buffers come out of a TBufPool instead of straight off the pipe, so that
        // the Cube GEMM reduction can alias its own pool onto the same bytes once the attention
        // phase is over. TPipe::InitBufPool(pool, len, shareBuf) sets the new pool's startAddr_
        // to shareBuf's and does NOT advance the hardware pool, so the reduction's panels cost
        // nothing; without it they need 23904 bytes the operator does not have.
        //
        // Three properties make this safe, and all three are why it is these three buffers:
        //
        //   * They are dead by then. nzOperandBuf_ is the tile ingest's NZ staging operand,
        //     queryWorkBuf_ the query staging, and accBuf_ the per-task online-softmax
        //     accumulator whose result StageTaskOutput has already written to the GM workspace.
        //     AscendC::SyncAll<false>() stands between Process() and the reduction, so nothing
        //     reads any of them afterwards. TurboQuantQueryBasis's buffers are NOT in here on
        //     purpose: the output-stage finisher runs during the reduction and reads the Pi
        //     signs in signBuf_, which are content, not scratch.
        //   * The pool is exactly as long as the three buffers it holds, so kUbEnd and every
        //     later buffer's offset are unchanged. That matters for the first rule above: the
        //     Cube's Fixpipe windows sit at UB offsets 0 and kScoreBytes and nothing here moves
        //     them, and the reduction's own dual-destination panels land at the same offset on
        //     every core because InitBuffers runs on all three.
        //   * It is allocated on every role, like the rest of this list. A pool created on one
        //     core and not another would put the reduction's panels at different addresses on
        //     the AIC and the AIVs, which is the exact failure the first rule warns about.
        pipe->InitBufPool(transientPool_, kTransientPoolBytes);
        transientPool_.InitBuffer(nzOperandBuf_, sizes[15]);
        transientPool_.InitBuffer(accBuf_, sizes[16]);
        transientPool_.InitBuffer(queryWorkBuf_, sizes[17]);
        pipe->InitBuffer(queryInBuf_, sizes[18]);
        pipe->InitBuffer(queryOperandBuf_, sizes[19]);
        // Shift probability and following buffers by one UB bank.
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

    // One hard MTE2 -> V id per ingest slot, taken once for the launch instead of once per tile. The ring
    // sets and waits a slot's id in strict alternation -- a slot's next read is ordered behind its previous
    // stage's vector work by the caller's V -> MTE2 edge -- so a fixed id per slot is all the pipeline needs,
    // and the tile loop issues no event bookkeeping at all.
    //
    // Holding the ids for the launch is also what keeps them out of the way of SyncEvent<MTE2_V>: TPipe's
    // FetchEventID only peeks at the first *free* id, so an id that stays allocated can never be handed to a
    // set/wait pair while a slot's read is in flight. kIngestSlots of the 8 the core has, and only on the AIV.
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

    // Byte offsets of the task's kv head in each tile row's scale slot: K lanes first, V lanes after.
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

    // Subcore 0 stages the K plane and subcore 1 the V plane.
    __aicore__ static inline uint32_t Plane() { return AscendC::GetSubBlockIdx() == 0 ? kKeyPlane : kValuePlane; }

    __aicore__ inline AscendC::LocalTensor<OperandT> PlaneOperand(Mm &mm, const uint32_t slot)
    {
        return Plane() == kKeyPlane ? mm.L1Key(slot) : mm.L1Value(slot);
    }

    // This subcore's plane of one tile and the tile's scale lanes, into ingest slot `slot`. rowBase is a
    // whole number of tiles into the block, so an NZ-tiled (tile, kv head) starts at
    // (row * kv heads + kvHead * kCubeTileRows) * packedBytes (turboquant_layout.h, NzTiledPackedByte).
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

    // An NZ-tiled plane is already the NZ image of its bytes, and the expand is byte-wise, so a chunk's
    // low nibbles land in place on NZ groups [0, groups) and its high nibbles on [groups, 2 * groups).
    // The staging buffer is the L1 operand as unpacked, and one MTE3 burst stages it.
    //
    // The two admitted expands differ only in how a widened nibble becomes an operand: kNative casts
    // the signed code itself (the uniform grid), a gathering codec resolves it through the UB table
    // its Init loaded from modeTables -- the Lloyd-Max centroids when UNPACK is kLloydMaxLut.
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

    // The recurrence over every row at once: a row op repeats over the heads, a per-head scalar is a head
    // block, and the row sums and maxima are whole-row reduces with no scalar readback. Logits, one
    // online-softmax step (newMax, alpha, runSum), then the probability rows on the operand grid.
    //
    // `contextBase` is the context index of this tile's first row and `logicalPage` the logical
    // page index of that row in the token's block table row; the telemetry probe folds them into
    // the running argmax record (turboquant_layout.h). They cost the unprobed path nothing: the
    // engine only reads them inside the probe phase.
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

    // The partial tails borrow score window 0 as scratch. The task's tile loop is over, so every window's
    // rows are dead; window 0 is the one that exists whatever kScoreWindows is.
    __aicore__ inline void StagePartials(const uint32_t token, const uint32_t split, const TurboQuantTaskHeads &heads)
    {
        const AscendC::LocalTensor<float> acc   = accBuf_.Get<float>();
        const AscendC::LocalTensor<float> tails = Scores(0);
        ascend_micro::vector_cube_pack_partial_tails(tails, StateField(kStateRunMax), StateField(kStateRunSum),
                                                     heads.mine, kPartialTail, kPartialMaxLane, kPartialSumLane);
#if defined(ASCEND_TQ_DECODE_TELEMETRY) && ASCEND_TQ_DECODE_TELEMETRY
        // The probe record rides the tail's dead lanes 1 and 2 (turboquant_layout.h). Both are
        // 4-byte-aligned lanes inside a 64-byte record, so they are written with a Select
        // against the single-lane predicates BeginArgmaxProbe precomputed -- a direct Adds to
        // an unaligned lane would be a misaligned vector store.
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

    // The fused path's telemetry record (ASCEND_TQ_DECODE_TELEMETRY): the same 16-lane
    // statistics tail a split partial carries, published at the (token, head, split 0) slot
    // whose whole slab is dead on this path -- a fused token never stages a partial. Lane
    // kTelemetryMaxLane is the running maximum logit, lanes kTelemetryPosLane and
    // kTelemetryPageLane the probe pair, and every other lane zero -- the run_sum lane
    // included, which is how the host tells a fused record from a split one. Scores(0) is the
    // staging window for the same reason as in StagePartials.
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
    // The three above live in this pool; the Cube GEMM reduction aliases onto it. See
    // InitBuffers for why it is exactly these three and why its length is exactly their sum.
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
