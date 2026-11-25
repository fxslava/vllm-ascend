#ifndef ASCEND_MICRO_TURBOQUANT_GEMM_REDUCE_SERVICE_H
#define ASCEND_MICRO_TURBOQUANT_GEMM_REDUCE_SERVICE_H

#include "kernel_operator.h"

#include "../../../common/turboquant_gemm_reduce.hpp"
#include "../../../common/turboquant_common.h"
#include "../vector/turboquant_vector_service.h"

namespace ascend_micro {
namespace gemm_reduce {

using vllm_ascend::turboquant::kFp32PerBlock;
using vllm_ascend::turboquant::kSubBlockSyncMode;
using vllm_ascend::turboquant::kVectorNegInf;
using vllm_ascend::turboquant::MixBlockIdx;
using vllm_ascend::turboquant::NormalizeHeads;
using vllm_ascend::turboquant::SyncEvent;
using vllm_ascend::turboquant::SyncMatrixToFixpipe;
using vllm_ascend::turboquant::SyncMte1ToMatrix;
using vllm_ascend::turboquant::kGatherSrcBase;
using vllm_ascend::turboquant::SyncMte2ToVector;
using vllm_ascend::turboquant::SyncScalarToVector;
using vllm_ascend::turboquant::SyncVectorToMte2;
using vllm_ascend::turboquant::SyncVectorToMte3;
using vllm_ascend::turboquant::WriteHeads;

__aicore__ inline constexpr uint16_t FlagWeightsReady(uint32_t slot)
{
    return static_cast<uint16_t>(kReduceFlagWeightsReady + slot);
}
__aicore__ inline constexpr uint16_t FlagProductReady(uint32_t slot)
{
    return static_cast<uint16_t>(kReduceFlagProductReady + slot);
}

template <typename scalar_t, bool GEMV_DISABLED = kGemvDisabledDefault>
class TurboQuantGemmReducer {
public:

    static_assert(sizeof(scalar_t) == 2,
                  "turboquant_fused_decode_opt.hpp's kUbEnd/kBasisUbBytes/kReducerUbBytes assume a "
                  "16-bit output element");

    __aicore__ inline explicit TurboQuantGemmReducer(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *workspace, __gm__ void *output, const uint32_t numHeads,
                                const uint32_t headSize, const uint32_t numSplits)
    {
        numHeads_ = numHeads;
        headSize_ = headSize;
        numSplits_ = numSplits;
        alignedSplits_ = kAlignedSplits(numSplits);

        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(output));

        for (uint32_t slot = 0; slot < kReduceSlots; ++slot) {
            pipe_->InitBuffer(weightL1Buf_[slot], kWeightL1Bytes);
            pipe_->InitBuffer(partialL1Buf_[slot], kPartialL1Bytes);
        }
        if ASCEND_IS_AIC {
            pipe_->InitBuffer(operandL0aBuf_, kReduceL0aBytes);
            pipe_->InitBuffer(operandL0bBuf_, kReduceL0bBytes);
            pipe_->InitBuffer(productL0cBuf_, kReduceL0cBytes);
        }

        for (uint32_t slot = 0; slot < kReduceSlots; ++slot) {

            pipe_->InitBuffer(productUbBuf_[slot], kReduceProductUbBytes);
        }
        pipe_->InitBuffer(statsUbBuf_, kStatsUbBytes);
        pipe_->InitBuffer(outUbBuf_, headSize * sizeof(scalar_t));

        AscendC::PipeBarrier<PIPE_ALL>();
        PrimeVectorState();
    }

    template <typename Select, typename Finisher>
    __aicore__ inline void Reduce(const uint32_t total, const uint32_t perBlock, Select &select,
                                  Finisher &finisher)
    {
        const uint32_t start = MixBlockIdx() * perBlock;
        if (start >= total) {

            return;
        }
        uint32_t end = start + perBlock;
        if (end > total) {
            end = total;
        }

        uint32_t tasks = 0;
        for (uint32_t task = start; task < end && tasks < kMaxTasksPerBlock; ++task) {
            if (select.NeedsCombine(task / numHeads_)) {
                taskIds_[tasks++] = task;
            }
        }
        if (tasks == 0) {
            return;
        }

        const uint32_t prologue = kReducePrologue(tasks);
        for (uint32_t r = 0; r < prologue; ++r) {
            Fetch(r);
            Stats(r);
        }
        for (uint32_t r = 0; r < tasks; ++r) {
            Gemm(r, tasks);
            Finish(r, finisher);
            if (r + kReduceSlots < tasks) {
                Fetch(r + kReduceSlots);
                Stats(r + kReduceSlots);
            }
        }
    }

private:

    static constexpr uint32_t kBrcbDstBlocks = 8;
    static constexpr uint32_t kLaneMax = 0 * kFp32PerBlock;
    static constexpr uint32_t kLaneSum = 1 * kFp32PerBlock;
    static constexpr uint32_t kLaneWeight = 2 * kFp32PerBlock;
    static constexpr uint32_t kLaneWeighted = 3 * kFp32PerBlock;
    static constexpr uint32_t kLaneScratch = 4 * kFp32PerBlock;
    static constexpr uint32_t kLaneTotal = 6 * kFp32PerBlock;
    static constexpr uint32_t kLaneStar = 8 * kFp32PerBlock;
    static constexpr uint32_t kLaneSums = 16 * kFp32PerBlock;
    static constexpr uint32_t kLaneInvs = 24 * kFp32PerBlock;
    static constexpr uint32_t kLaneMaxOffsets = 32 * kFp32PerBlock;
    static constexpr uint32_t kLaneSumOffsets = 33 * kFp32PerBlock;
    static constexpr uint32_t kLaneTail = 34 * kFp32PerBlock;

    static constexpr uint32_t kTailBlocksPerSplit = 2;
    static constexpr uint32_t kStatsUbFloats =
        kLaneTail + kMaxSplits * kTailBlocksPerSplit * kFp32PerBlock;
    static constexpr uint32_t kStatsUbBytes = kStatsUbFloats * sizeof(float);

    static_assert(kLaneStar + kBrcbDstBlocks * kFp32PerBlock <= kLaneSums,
                  "the m* broadcast must not reach the next allocation");
    static_assert(kLaneSums + kBrcbDstBlocks * kFp32PerBlock <= kLaneInvs,
                  "NormalizeHeads' broadcast must not reach its own reciprocals");
    static_assert(kLaneInvs + kBrcbDstBlocks * kFp32PerBlock <= kLaneMaxOffsets,
                  "the reciprocals must not reach the Gather tables");
    static_assert(kLaneSumOffsets + kFp32PerBlock <= kLaneTail,
                  "the Gather tables must not reach the DMA landings");
    static_assert(kLaneTotal + kReduceSlots * kFp32PerBlock <= kLaneStar,
                  "the per-slot l* blocks must not be inside a broadcast destination");
    static_assert(kStatsUbBytes % 32 == 0, "every UB allocation is a whole 32-byte block");
    static_assert(kStatsUbBytes >= kReduceStatsUbBytes, "the contract's stats budget is a floor");

    static constexpr uint32_t kTailStrideBytes =
        kTailBlocksPerSplit * kFp32PerBlock * sizeof(float);

    static constexpr uint32_t kMaxTasksPerBlock = 64;

    __aicore__ inline AscendC::LocalTensor<float> Stats() { return statsUbBuf_.template Get<float>(); }

    __aicore__ inline void PrimeVectorState()
    {
        if ASCEND_IS_AIV {
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        const AscendC::LocalTensor<float> stats = Stats();
        AscendC::Duplicate(stats, 0.0f, kStatsUbFloats);
        SyncVectorToMte3();
        for (uint32_t slot = 0; slot < kReduceSlots; ++slot) {
            AscendC::DataCopy(weightL1Buf_[slot].template Get<float>(), stats,
                              kWeightL1Bytes / sizeof(float));
        }
        SyncEvent<AscendC::HardEvent::MTE3_S>();

        const AscendC::LocalTensor<uint32_t> maxOffsets =
            stats[kLaneMaxOffsets].template ReinterpretCast<uint32_t>();
        const AscendC::LocalTensor<uint32_t> sumOffsets =
            stats[kLaneSumOffsets].template ReinterpretCast<uint32_t>();
        for (uint32_t split = 0; split < kMaxSplits; ++split) {
            const uint32_t base = split * kTailStrideBytes;
            maxOffsets.SetValue(split, base + kPartialMaxLane * sizeof(float));
            sumOffsets.SetValue(split, base + kPartialSumLane * sizeof(float));
        }
        SyncScalarToVector();
        }
    }

    __aicore__ inline uint64_t PartialRow(const uint32_t task, const uint32_t split) const
    {
        return kPartialRowOffset(task / numHeads_, task % numHeads_, split, numHeads_, numSplits_);
    }

    __aicore__ inline void Fetch(const uint32_t r)
    {
        if ASCEND_IS_AIC {
        const uint32_t slot = kReduceSlotOf(r);
        if (r >= kReduceSlots) {

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(slot));
        }
        AscendC::Dn2NzParams params;
        params.dnNum = kDn2NzDnNum;
        params.nValue = static_cast<uint16_t>(headSize_);

        params.dValue = numSplits_;
        params.srcDnMatrixStride = 0;

        params.srcDValue = kPartialStride;
        params.dstNzC0Stride = static_cast<uint16_t>(headSize_);
        params.dstNzNStride = kDn2NzDstNzNStride;
        params.dstNzMatrixStride = 0;
        AscendC::DataCopy(partialL1Buf_[slot].template Get<float>(),
                          workspaceGm_[PartialRow(taskIds_[r], 0)], params);
        SyncEvent<AscendC::HardEvent::MTE2_MTE1>();
        }
    }

    __aicore__ inline void Stats(const uint32_t r)
    {
        if ASCEND_IS_AIV {
        const uint32_t slot = kReduceSlotOf(r);
        if (AscendC::GetSubBlockIdx() != 0) {

            AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(FlagWeightsReady(slot));
            return;
        }

        const AscendC::LocalTensor<float> stats = Stats();
        const AscendC::LocalTensor<float> maxes = stats[kLaneMax];
        const AscendC::LocalTensor<float> sums = stats[kLaneSum];
        const AscendC::LocalTensor<float> weight = stats[kLaneWeight];
        const AscendC::LocalTensor<float> star = stats[kLaneStar];
        const AscendC::LocalTensor<float> weighted = stats[kLaneWeighted];
        const AscendC::LocalTensor<float> scratch = stats[kLaneScratch];
        const AscendC::LocalTensor<float> tail = stats[kLaneTail];

        AscendC::Duplicate(maxes, kVectorNegInf, kFp32PerBlock);
        AscendC::Duplicate(sums, 0.0f, kFp32PerBlock);

        SyncVectorToMte2();
        for (uint32_t split = 0; split < numSplits_; ++split) {
            const uint64_t row = PartialRow(taskIds_[r], split) + headSize_;

            AscendC::DataCopy(tail[split * kTailBlocksPerSplit * kFp32PerBlock], workspaceGm_[row],
                              kTailBlocksPerSplit * kFp32PerBlock);
        }
        SyncMte2ToVector();

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Gather(maxes, tail, stats[kLaneMaxOffsets].template ReinterpretCast<uint32_t>(),
                        kGatherSrcBase, numSplits_);
        AscendC::Gather(sums, tail, stats[kLaneSumOffsets].template ReinterpretCast<uint32_t>(),
                        kGatherSrcBase, numSplits_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::ReduceMax<float>(scratch, maxes, scratch[kFp32PerBlock], kFp32PerBlock, false);
        AscendC::Brcb(star, scratch, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
        AscendC::Sub(weight, maxes, star, kFp32PerBlock);
        AscendC::Exp(weight, weight, kFp32PerBlock);
        AscendC::Mul(weighted, weight, sums, kFp32PerBlock);

        AscendC::ReduceSum<float>(stats[kLaneTotal + slot * kFp32PerBlock], weighted, scratch,
                                  kFp32PerBlock);

        SyncVectorToMte3();
        AscendC::DataCopy(weightL1Buf_[slot].template Get<float>(), weight, kFp32PerBlock);

        SyncEvent<AscendC::HardEvent::MTE3_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(FlagWeightsReady(slot));
        }
    }

    __aicore__ inline void Gemm(const uint32_t r, const uint32_t tasks)
    {
        if ASCEND_IS_AIC {
        const uint32_t slot = kReduceSlotOf(r);

        AscendC::CrossCoreWaitFlag(FlagWeightsReady(slot));

        const AscendC::LocalTensor<float> leftL0a = operandL0aBuf_.template Get<float>();
        const AscendC::LocalTensor<float> rightL0b = operandL0bBuf_.template Get<float>();
        const AscendC::LocalTensor<float> product = productL0cBuf_.template Get<float>();

        AscendC::LoadData2DParamsV2 aParams;
        aParams.mStartPosition = 0;
        aParams.kStartPosition = 0;
        aParams.mStep = 1;
        aParams.kStep = 1;
        aParams.srcStride = 1;
        aParams.dstStride = 1;
        aParams.ifTranspose = false;
        AscendC::LoadData(leftL0a, weightL1Buf_[slot].template Get<float>(), aParams);

        AscendC::LoadData2DParamsV2 bParams;
        bParams.mStartPosition = 0;
        bParams.kStartPosition = 0;
        bParams.mStep = static_cast<uint16_t>(headSize_ / kFractalRows);
        bParams.kStep = 1;
        bParams.srcStride = static_cast<uint16_t>(headSize_ / kFractalRows);
        bParams.dstStride = static_cast<uint16_t>(headSize_ / kFractalRows);
        bParams.ifTranspose = false;
        AscendC::LoadData(rightL0b, partialL1Buf_[slot].template Get<float>(), bParams);

        SyncMte1ToMatrix();

        AscendC::MmadParams mmad;
        mmad.m = 1;
        mmad.n = static_cast<uint16_t>(headSize_);

        mmad.k = static_cast<uint16_t>(numSplits_);

        mmad.cmatrixInitVal = true;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)

        mmad.disableGemv = GEMV_DISABLED;
#else
        static_cast<void>(GEMV_DISABLED);
#endif
        AscendC::Mmad(product, leftL0a, rightL0b, mmad);

        if (r + kReduceSlots < tasks) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(slot));
        }

        SyncMatrixToFixpipe();
        AscendC::Fixpipe<float, float, vllm_ascend::turboquant::kFixpipeToUb>(
            productUbBuf_[slot].template Get<float>(), product,
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>(
                static_cast<uint16_t>(headSize_), 1, static_cast<uint16_t>(kFractalRows),
                headSize_));

        AscendC::PipeBarrier<PIPE_FIX>();
        SyncEvent<AscendC::HardEvent::FIX_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(FlagProductReady(slot));
        }
    }

    template <typename Finisher>
    __aicore__ inline void Finish(const uint32_t r, Finisher &finisher)
    {
        if ASCEND_IS_AIV {
        const uint32_t slot = kReduceSlotOf(r);
        AscendC::CrossCoreWaitFlag(FlagProductReady(slot));
        if (AscendC::GetSubBlockIdx() != 0) {

            return;
        }

        const AscendC::LocalTensor<float> stats = Stats();
        const AscendC::LocalTensor<float> acc = productUbBuf_[slot].template Get<float>();
        const uint32_t task = taskIds_[r];
        const uint32_t token = task / numHeads_;
        const uint32_t head = task % numHeads_;

        NormalizeHeads(acc, stats[kLaneTotal + slot * kFp32PerBlock], 1, headSize_,
                       stats[kLaneSums], stats[kLaneInvs]);
        finisher.FinishHeads(acc, token, head, 1);
        WriteHeads<scalar_t>(acc, 1, headSize_, outUbBuf_.template Get<scalar_t>(), outputGm_,
                             (static_cast<uint64_t>(token) * numHeads_ + head) * headSize_);
        }
    }

    AscendC::TPipe *pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> weightL1Buf_[kReduceSlots];
    AscendC::TBuf<AscendC::TPosition::B1> partialL1Buf_[kReduceSlots];
    AscendC::TBuf<AscendC::TPosition::A2> operandL0aBuf_;
    AscendC::TBuf<AscendC::TPosition::B2> operandL0bBuf_;
    AscendC::TBuf<AscendC::TPosition::CO1> productL0cBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> productUbBuf_[kReduceSlots];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> statsUbBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outUbBuf_;
    AscendC::GlobalTensor<float> workspaceGm_;
    AscendC::GlobalTensor<scalar_t> outputGm_;
    uint32_t taskIds_[kMaxTasksPerBlock] = {};
    uint32_t numHeads_ = 0;
    uint32_t headSize_ = 0;
    uint32_t numSplits_ = 1;
    uint32_t alignedSplits_ = kC0Floats;
};

template <typename scalar_t, bool GEMV_DISABLED = kGemvDisabledDefault>
class TurboQuantBatchedGemmReducer {
public:

    static_assert(sizeof(scalar_t) == 2,
                  "turboquant_fused_decode_opt.hpp's kUbEnd/kBasisUbBytes/kReducerUbBytes assume a "
                  "16-bit output element");

    __aicore__ inline explicit TurboQuantBatchedGemmReducer(AscendC::TPipe *pipe) : pipe_(pipe) {}

    template <typename SharedPool>
    __aicore__ inline void Init(__gm__ void *workspace, __gm__ void *output, const uint32_t numHeads,
                                const uint32_t headSize, const uint32_t numSplits,
                                SharedPool &sharedPool)
    {
        pipe_->InitBufPool(ubPool_, kGemmReducerUbBytes, sharedPool);
        pooled_ = true;
        InitCommon(workspace, output, numHeads, headSize, numSplits);
    }

    __aicore__ inline void Init(__gm__ void *workspace, __gm__ void *output, const uint32_t numHeads,
                                const uint32_t headSize, const uint32_t numSplits)
    {
        pooled_ = false;
        InitCommon(workspace, output, numHeads, headSize, numSplits);
    }

private:

    template <typename Buf>
    __aicore__ inline void AllocUb(Buf &buf, const uint32_t bytes)
    {
        if (pooled_) {
            ubPool_.InitBuffer(buf, bytes);
        } else {
            pipe_->InitBuffer(buf, bytes);
        }
    }

    __aicore__ inline void InitCommon(__gm__ void *workspace, __gm__ void *output,
                                      const uint32_t numHeads, const uint32_t headSize,
                                      const uint32_t numSplits)
    {
        numHeads_ = numHeads;
        headSize_ = headSize;
        numSplits_ = numSplits;

        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(output));

        for (uint32_t slot = 0; slot < kBatchSlots; ++slot) {
            pipe_->InitBuffer(weightL1Buf_[slot], kWeightCubeBytes);
        }
        for (uint32_t ss = 0; ss < kSliceSlots; ++ss) {
            pipe_->InitBuffer(sliceL1Buf_[ss], kSliceL1Bytes);
        }
        if ASCEND_IS_AIC {
            pipe_->InitBuffer(operandL0aBuf_, kGemmL0aBytes);
            for (uint32_t ss = 0; ss < kSliceSlots; ++ss) {
                pipe_->InitBuffer(operandL0bBuf_[ss], kSliceL1Bytes);
                pipe_->InitBuffer(productL0cBuf_[ss], kGemmL0cSlotBytes);
            }
        }

        for (uint32_t slot = 0; slot < kBatchSlots; ++slot) {
            AllocUb(panelUbBuf_[slot], kGemmPanelUbBytes);
        }
        AllocUb(statsUbBuf_, kStatsUbBytes);
        AllocUb(zeroUbBuf_, kSliceFractalFloats * sizeof(float));
        AllocUb(outUbBuf_, kGemmPanelRows * headSize * sizeof(scalar_t));
        AscendC::PipeBarrier<PIPE_ALL>();
        PrimeVectorState();
    }

public:

    template <typename Select, typename Finisher>
    __aicore__ inline void Reduce(const uint32_t total, const uint32_t perBlock, Select &select,
                                  Finisher &finisher)
    {
        const uint32_t start = MixBlockIdx() * perBlock;
        if (start >= total) {

            return;
        }
        uint32_t end = start + perBlock;
        if (end > total) {
            end = total;
        }

        uint32_t batches = 0;
        uint32_t rows = 0;
        uint32_t previous = 0;
        for (uint32_t task = start; task < end; ++task) {
            if (!select.NeedsCombine(task / numHeads_)) {
                continue;
            }
            if (batches == 0 || kGemmBatchBreaks(previous, task, rows)) {
                if (batches == kMaxGemmBatchesPerBlock) {
                    break;
                }
                batchTask_[batches] = task;
                batchLive_[batches] = 1;
                ++batches;
                rows = 1;
            } else {
                ++batchLive_[batches - 1];
                ++rows;
            }
            previous = task;
        }
        if (batches == 0) {
            return;
        }

        PrimeCubeRing();
        const uint32_t prologue = kReducePrologue(batches);
        for (uint32_t b = 0; b < prologue; ++b) {
            Stats(b);
        }
        for (uint32_t b = 0; b < batches; ++b) {
            Gemm(b);
            Finish(b, finisher);

            if (b + kBatchSlots < batches) {
                Stats(b + kBatchSlots);
            }
        }
        DrainCubeRing();
    }

private:

    static constexpr uint32_t kBrcbDstBlocks = 8;
    static constexpr uint32_t kSumsBlocks = 16;
    static constexpr uint32_t kLaneMax = 0 * kFp32PerBlock;
    static constexpr uint32_t kLaneSum = 1 * kFp32PerBlock;
    static constexpr uint32_t kLaneWeighted = 2 * kFp32PerBlock;
    static constexpr uint32_t kLaneScratch = 4 * kFp32PerBlock;
    static constexpr uint32_t kLaneStar = 8 * kFp32PerBlock;
    static constexpr uint32_t kLaneWeight = 16 * kFp32PerBlock;
    static constexpr uint32_t kLaneTotal = 24 * kFp32PerBlock;
    static constexpr uint32_t kLaneSums = 40 * kFp32PerBlock;
    static constexpr uint32_t kLaneInvs = 56 * kFp32PerBlock;
    static constexpr uint32_t kLaneMaxOffsets = 64 * kFp32PerBlock;
    static constexpr uint32_t kLaneSumOffsets = 65 * kFp32PerBlock;
    static constexpr uint32_t kLaneTail = 66 * kFp32PerBlock;

    static constexpr uint32_t kTailBlocksPerSplit = 2;
    static constexpr uint32_t kTailBlocks = kGemmPanelRows * kMaxSplits * kTailBlocksPerSplit;
    static constexpr uint32_t kStatsUbFloats = kLaneTail + kTailBlocks * kFp32PerBlock;
    static constexpr uint32_t kStatsUbBytes = kStatsUbFloats * sizeof(float);

    static_assert(kLaneTail == kGemmStatsHeaderBlocks * kFp32PerBlock,
                  "the contract header's stats header block count must match this layout");
    static_assert(kStatsUbBytes == kGemmStatsUbBytes,
                  "the contract header's stats arena size must match this allocation");
    static constexpr uint32_t kTailStrideBytes = kTailBlocksPerSplit * kFp32PerBlock * sizeof(float);

    static constexpr uint32_t kTailGapBlocks =
        (kPartialStride - kPartialTail) / kFp32PerBlock;
    static_assert((kPartialStride - kPartialTail) % kFp32PerBlock == 0,
                  "the tail gap must be a whole number of 32-byte blocks");
    static_assert(kTailGapBlocks == 32);

    static_assert(kLaneStar + kBrcbDstBlocks * kFp32PerBlock <= kLaneWeight,
                  "the m* broadcast must not reach the weight rows");
    static_assert(kLaneWeight + kGemmPanelRows * kFp32PerBlock <= kLaneTotal,
                  "every panel row owns a weight block, so MTE3 reads what V is not rewriting");
    static_assert(kLaneTotal + kBatchSlots * kGemmPanelRows * kFp32PerBlock <= kLaneSums,
                  "the per-slot l* lanes must not reach the next allocation");
    static_assert(kLaneSums + kSumsBlocks * kFp32PerBlock <= kLaneInvs,
                  "NormalizeHeads' broadcast must not reach its own reciprocals");
    static_assert((kGemmPanelRows - 1) * kFp32PerBlock + kBrcbDstBlocks * kFp32PerBlock <=
                      kSumsBlocks * kFp32PerBlock,
                  "sums needs (rows - 1) * 8 + 64 floats, not rows * 8");
    static_assert(kLaneInvs + kGemmPanelRows * kFp32PerBlock <= kLaneMaxOffsets,
                  "the reciprocals must not reach the Gather tables");
    static_assert(kLaneSumOffsets + kFp32PerBlock <= kLaneTail,
                  "the Gather tables must not reach the DMA landing");
    static_assert(kStatsUbBytes % 32 == 0, "every UB allocation is a whole 32-byte block");

    static constexpr uint8_t kGemmDualDstSplitM = 1;

    static_assert(kBatchSlots * kGemmPanelUbBytes + kStatsUbBytes +
                          kSliceFractalFloats * sizeof(float) +
                          kGemmPanelRows * kHeadSize * sizeof(scalar_t) ==
                      kGemmReducerUbBytes,
                  "the contract header's UB claim must equal what Init() actually allocates");

    __aicore__ inline AscendC::LocalTensor<float> Stats() { return statsUbBuf_.template Get<float>(); }

    __aicore__ inline uint64_t PartialRowOf(const uint32_t task) const
    {
        return kPartialRowOffset(task / numHeads_, task % numHeads_, 0, numHeads_, numSplits_);
    }

    __aicore__ inline void PrimeVectorState()
    {
        if ASCEND_IS_AIV {
        const uint32_t sub = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
        const AscendC::LocalTensor<float> stats = Stats();
        const AscendC::LocalTensor<float> zero = zeroUbBuf_.template Get<float>();
        AscendC::Duplicate(stats, 0.0f, kStatsUbFloats);
        AscendC::Duplicate(zero, 0.0f, kSliceFractalFloats);
        SyncVectorToMte3();
        const uint32_t base = kGemmFirstRowOf(sub) * kSliceFractalFloats;
        for (uint32_t slot = 0; slot < kBatchSlots; ++slot) {
            const AscendC::LocalTensor<float> cube = weightL1Buf_[slot].template Get<float>();
            for (uint32_t row = 0; row < kGemmPanelRows; ++row) {
                AscendC::DataCopy(cube[base + row * kSliceFractalFloats], zero,
                                  kSliceFractalFloats);
            }
        }
        SyncEvent<AscendC::HardEvent::MTE3_S>();

        const AscendC::LocalTensor<uint32_t> maxOffsets =
            stats[kLaneMaxOffsets].template ReinterpretCast<uint32_t>();
        const AscendC::LocalTensor<uint32_t> sumOffsets =
            stats[kLaneSumOffsets].template ReinterpretCast<uint32_t>();
        for (uint32_t split = 0; split < kMaxSplits; ++split) {
            const uint32_t byteBase = split * kTailStrideBytes;
            maxOffsets.SetValue(split, byteBase + kPartialMaxLane * sizeof(float));
            sumOffsets.SetValue(split, byteBase + kPartialSumLane * sizeof(float));
        }
        SyncScalarToVector();
        }
    }

    __aicore__ inline void Stats(const uint32_t b)
    {
        if ASCEND_IS_AIV {
        const uint32_t slot = kReduceSlotOf(b);
        const uint32_t sub = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
        const uint32_t live = batchLive_[b];
        const uint32_t mine = kGemmLiveRowsOf(live, sub);
        const uint32_t firstRow = kGemmFirstRowOf(sub);
        const uint32_t firstTask = batchTask_[b] + kGemmFirstTaskOf(live, sub);

        const AscendC::LocalTensor<float> stats = Stats();
        const AscendC::LocalTensor<float> cube = weightL1Buf_[slot].template Get<float>();
        const AscendC::LocalTensor<float> zero = zeroUbBuf_.template Get<float>();

        if (mine) {

            SyncVectorToMte2();
            AscendC::DataCopyParams tails;
            tails.blockCount = static_cast<uint16_t>(mine * numSplits_);
            tails.blockLen = static_cast<uint16_t>(kTailBlocksPerSplit);
            tails.srcStride = static_cast<uint16_t>(kTailGapBlocks);
            tails.dstStride = 0;
            AscendC::DataCopy(stats[kLaneTail], workspaceGm_[PartialRowOf(firstTask) + headSize_],
                              tails);
            SyncMte2ToVector();

            for (uint32_t i = 0; i < mine; ++i) {
                ReduceRow(stats, i, slot);
            }

            SyncVectorToMte3();
            for (uint32_t i = 0; i < mine; ++i) {
                AscendC::DataCopy(cube[kWeightRowOffsetFloats(firstRow + i)],
                                  stats[kLaneWeight + i * kFp32PerBlock], kFp32PerBlock);
            }
        }

        for (uint32_t i = mine; i < kGemmPanelRows; ++i) {
            AscendC::DataCopy(cube[kWeightRowOffsetFloats(firstRow + i)], zero, kFp32PerBlock);
        }

        SyncEvent<AscendC::HardEvent::MTE3_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(FlagWeightsReady(slot));
        }
    }

    __aicore__ inline void ReduceRow(const AscendC::LocalTensor<float> &stats, const uint32_t row,
                                     const uint32_t slot)
    {
        const AscendC::LocalTensor<float> maxes = stats[kLaneMax];
        const AscendC::LocalTensor<float> sums = stats[kLaneSum];

        const AscendC::LocalTensor<float> weight = stats[kLaneWeight + row * kFp32PerBlock];
        const AscendC::LocalTensor<float> star = stats[kLaneStar];
        const AscendC::LocalTensor<float> weighted = stats[kLaneWeighted];
        const AscendC::LocalTensor<float> scratch = stats[kLaneScratch];

        const AscendC::LocalTensor<float> landing =
            stats[kLaneTail + row * numSplits_ * kTailBlocksPerSplit * kFp32PerBlock];

        AscendC::Duplicate(maxes, kVectorNegInf, kFp32PerBlock);
        AscendC::Duplicate(sums, 0.0f, kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Gather(maxes, landing, stats[kLaneMaxOffsets].template ReinterpretCast<uint32_t>(),
                        kGatherSrcBase, numSplits_);
        AscendC::Gather(sums, landing, stats[kLaneSumOffsets].template ReinterpretCast<uint32_t>(),
                        kGatherSrcBase, numSplits_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::ReduceMax<float>(scratch, maxes, scratch[kFp32PerBlock], kFp32PerBlock, false);
        AscendC::Brcb(star, scratch, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
        AscendC::Sub(weight, maxes, star, kFp32PerBlock);
        AscendC::Exp(weight, weight, kFp32PerBlock);
        AscendC::Mul(weighted, weight, sums, kFp32PerBlock);
        AscendC::ReduceSum<float>(
            stats[kLaneTotal + (slot * kGemmPanelRows + row) * kFp32PerBlock], weighted, scratch,
            kFp32PerBlock);
    }

    __aicore__ inline void PrimeCubeRing()
    {
        if ASCEND_IS_AIC {

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(kGemmL0aEvent));
        }
    }

    __aicore__ inline void DrainCubeRing()
    {
        if ASCEND_IS_AIC {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(kGemmL0aEvent));
        }
    }

    __aicore__ inline void Gemm(const uint32_t b)
    {
        if ASCEND_IS_AIC {
        const uint32_t slot = kReduceSlotOf(b);
        const uint32_t live = batchLive_[b];
        const uint32_t rows0 = kGemmLiveRowsOf(live, 0);
        const uint32_t rows1 = kGemmLiveRowsOf(live, 1);
        const uint32_t firstTask = batchTask_[b];

        AscendC::CrossCoreWaitFlag(FlagWeightsReady(slot));

        SyncEvent<AscendC::HardEvent::MTE2_S>();

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(kGemmL0aEvent));
        AscendC::LoadData2DParamsV2 aParams;
        aParams.mStartPosition = 0;
        aParams.kStartPosition = 0;
        aParams.mStep = 1;
        aParams.kStep = static_cast<uint16_t>(kGemmBatchRows);
        aParams.srcStride = 1;
        aParams.dstStride = 1;
        aParams.ifTranspose = false;
        AscendC::LoadData(operandL0aBuf_.template Get<float>(),
                          weightL1Buf_[slot].template Get<float>(), aParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<event_t>(kGemmL0aEvent));

        for (uint32_t ss = 0; ss < kSliceSlots; ++ss) {
            const event_t id = static_cast<event_t>(kGemmSliceEventFirst + ss);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(id);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(id);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(id);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<event_t>(kGemmL0aEvent));

        for (uint32_t j = 0; j < kGemmSlices; ++j) {
            const uint32_t ss = j % kSliceSlots;
            const event_t id = static_cast<event_t>(kGemmSliceEventFirst + ss);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(id);
            FetchSlice(ss, firstTask, rows0, rows1, j);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(id);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(id);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(id);
            LoadSlice(ss);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(id);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(id);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(id);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(id);
            MmadSlice(ss);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(id);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(id);

            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(id);
            FixSlice(ss, slot, j);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(id);
        }

        for (uint32_t ss = 0; ss < kSliceSlots; ++ss) {
            const event_t id = static_cast<event_t>(kGemmSliceEventFirst + ss);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(id);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(id);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(id);
        }

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(kGemmL0aEvent));

        AscendC::PipeBarrier<PIPE_FIX>();
        SyncEvent<AscendC::HardEvent::FIX_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(FlagProductReady(slot));
        }
    }

    __aicore__ inline void FetchSlice(const uint32_t ss, const uint32_t firstTask,
                                      const uint32_t rows0, const uint32_t rows1,
                                      const uint32_t j)
    {
        const AscendC::LocalTensor<float> sliceL1 = sliceL1Buf_[ss].template Get<float>();
        AscendC::Dn2NzParams params;
        params.nValue = static_cast<uint16_t>(kSliceDn2NzNValue);

        params.dValue = numSplits_;
        params.srcDnMatrixStride = kSliceSrcMatrixStrideFloats(numSplits_);

        params.srcDValue = kPartialStride;
        params.dstNzC0Stride = static_cast<uint16_t>(kSliceDn2NzDstNzC0Stride);
        params.dstNzNStride = static_cast<uint16_t>(kSliceDn2NzDstNzNStride);
        params.dstNzMatrixStride = kSliceDn2NzDstMatrixStrideFloats;
        const uint32_t column = j * kGemmSliceCols;

        if (j < kSliceSlots) {
            const uint32_t live = rows0 + rows1;
            for (uint32_t sub = 0; sub < 2; ++sub) {
                const uint32_t padding = kGemmPaddingRowsOf(live, sub);
                if (padding == 0) continue;
                AscendC::InitConstValueParams<float> zero;
                zero.repeatTimes = 1;
                zero.blockNum = static_cast<uint16_t>(padding * kFractalBytes / kC0Bytes);
                zero.dstGap = 0;
                zero.initValue = 0.0f;
                AscendC::Fill(sliceL1[kGemmPaddingFirstRowOf(live, sub) * kSliceFractalFloats], zero);
            }
        }
        params.dnNum = static_cast<uint16_t>(rows0);
        AscendC::DataCopy(sliceL1, workspaceGm_[PartialRowOf(firstTask) + column], params);
        if (rows1) {
            params.dnNum = static_cast<uint16_t>(rows1);
            AscendC::DataCopy(sliceL1[kGemmFirstRowOf(1) * kSliceFractalFloats],
                              workspaceGm_[PartialRowOf(firstTask + rows0) + column], params);
        }
    }

    __aicore__ inline void LoadSlice(const uint32_t ss)
    {
        AscendC::LoadData2DParamsV2 bParams;
        bParams.mStartPosition = 0;
        bParams.kStartPosition = 0;
        bParams.mStep = 1;
        bParams.kStep = static_cast<uint16_t>(kGemmBatchRows);
        bParams.srcStride = 1;
        bParams.dstStride = 1;
        bParams.ifTranspose = false;
        AscendC::LoadData(operandL0bBuf_[ss].template Get<float>(),
                          sliceL1Buf_[ss].template Get<float>(), bParams);
    }

    __aicore__ inline void MmadSlice(const uint32_t ss)
    {
        AscendC::MmadParams mmad;

        mmad.m = static_cast<uint16_t>(kGemmBatchRows);
        mmad.n = static_cast<uint16_t>(kGemmSliceCols);
        mmad.k = static_cast<uint16_t>(kGemmKMax);

        mmad.cmatrixInitVal = true;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)

        mmad.disableGemv = GEMV_DISABLED;
#else
        static_cast<void>(GEMV_DISABLED);
#endif
        AscendC::Mmad(productL0cBuf_[ss].template Get<float>(),
                      operandL0aBuf_.template Get<float>(),
                      operandL0bBuf_[ss].template Get<float>(), mmad);
    }

    __aicore__ inline void FixSlice(const uint32_t ss, const uint32_t slot, const uint32_t j)
    {
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fix(
            static_cast<uint16_t>(kGemmSliceCols), static_cast<uint16_t>(kGemmBatchRows),
            static_cast<uint16_t>(kFractalRows), headSize_);
        fix.dualDstCtl = kGemmDualDstSplitM;
        fix.subBlockId = false;
        AscendC::Fixpipe<float, float, vllm_ascend::turboquant::kFixpipeToUb>(
            panelUbBuf_[slot].template Get<float>()[j * kGemmSliceCols],
            productL0cBuf_[ss].template Get<float>(), fix);
    }

    template <typename Finisher>
    __aicore__ inline void Finish(const uint32_t b, Finisher &finisher)
    {
        if ASCEND_IS_AIV {
        const uint32_t slot = kReduceSlotOf(b);
        const uint32_t sub = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
        AscendC::CrossCoreWaitFlag(FlagProductReady(slot));

        SyncEvent<AscendC::HardEvent::MTE3_S>();

        const uint32_t live = batchLive_[b];
        const uint32_t mine = kGemmLiveRowsOf(live, sub);
        if (mine == 0) {

            return;
        }

        const AscendC::LocalTensor<float> stats = Stats();
        const AscendC::LocalTensor<float> panel = panelUbBuf_[slot].template Get<float>();
        const uint32_t firstTask = batchTask_[b] + kGemmFirstTaskOf(live, sub);

        NormalizeHeads(panel, stats[kLaneTotal + slot * kGemmPanelRows * kFp32PerBlock], mine,
                       headSize_, stats[kLaneSums], stats[kLaneInvs]);

        uint32_t i = 0;
        while (i < mine) {
            const uint32_t task = firstTask + i;
            const uint32_t head = task % numHeads_;
            uint32_t run = numHeads_ - head;
            if (run > mine - i) {
                run = mine - i;
            }
            finisher.FinishHeads(panel[i * headSize_], task / numHeads_, head, run);
            i += run;
        }

        WriteHeads<scalar_t>(panel, mine, headSize_, outUbBuf_.template Get<scalar_t>(),
                             outputGm_, static_cast<uint64_t>(firstTask) * headSize_);
        }
    }

    AscendC::TPipe *pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> weightL1Buf_[kBatchSlots];
    AscendC::TBuf<AscendC::TPosition::B1> sliceL1Buf_[kSliceSlots];
    AscendC::TBuf<AscendC::TPosition::A2> operandL0aBuf_;
    AscendC::TBuf<AscendC::TPosition::B2> operandL0bBuf_[kSliceSlots];
    AscendC::TBuf<AscendC::TPosition::CO1> productL0cBuf_[kSliceSlots];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> panelUbBuf_[kBatchSlots];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> statsUbBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> zeroUbBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outUbBuf_;

    AscendC::TBufPool<AscendC::TPosition::VECCALC> ubPool_;
    bool pooled_ = false;
    AscendC::GlobalTensor<float> workspaceGm_;
    AscendC::GlobalTensor<scalar_t> outputGm_;
    uint32_t batchTask_[kMaxGemmBatchesPerBlock] = {};
    uint32_t batchLive_[kMaxGemmBatchesPerBlock] = {};
    uint32_t numHeads_ = 0;
    uint32_t headSize_ = 0;
    uint32_t numSplits_ = 1;
};

}
}

#endif
