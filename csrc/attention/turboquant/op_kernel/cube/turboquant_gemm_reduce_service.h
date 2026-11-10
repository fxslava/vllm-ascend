/*
 * The Split-K final reduction on the Cube. CAMODEL-VALIDATED, NO SILICON RUN.
 *
 * Two reducers live here, and only one of them is wired into a production path:
 *
 *   TurboQuantGemmReducer          M = 1 Cube GEMV, one task per Mmad. Probe only; nothing in
 *                                  a production entry instantiates it. Kept because it is the
 *                                  bisection control for the batched form and because its
 *                                  trace is what established the m2:2 ordering defect.
 *   TurboQuantBatchedGemmReducer   M = kGemmBatchRows packed query rows against 16-channel
 *                                  slices, both vector subcores reducing half the batch. This
 *                                  is the one TurboQuantFusedDecodeOpt::ReduceGemm() drives,
 *                                  behind ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE.
 *
 * Both call the production vector service for every piece of arithmetic they share with
 * TurboQuantPartialReducer -- NormalizeHeads, WriteHeads and the output-stage finisher -- and
 * reimplement nothing but the combine itself.
 *
 * The geometry, flag ids and budgets come from include/turboquant_gemm_reduce.hpp, which the
 * evidence about this code's arithmetic and not about a transcription of it.
 *
 * READ THE UB BUDGET IN THAT HEADER BEFORE ENABLING THE SWITCH. The batched form's panels do
 * not fit the operator's remaining UB headroom, and the static_assert there names the deficit.
 */

#ifndef ASCEND_MICRO_TURBOQUANT_GEMM_REDUCE_SERVICE_H
#define ASCEND_MICRO_TURBOQUANT_GEMM_REDUCE_SERVICE_H

#include "kernel_operator.h"

#include "../../include/turboquant_gemm_reduce.hpp"
#include "../common/turboquant_common.h"
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

// The stage's cross-core ids. They RECYCLE the attention schedule's: a mode-4 id is four bits,
// the toolkit reserves 11..14, the attention ring already spends 0..9 and the task join is 10,
// so there is no free id to take. Recycling is sound here only because SyncAll<false>() and
// the attention phase's own epilogue stand between the phases and every attention place is
// drained to zero first -- and it drags in the fence precondition, which is why every
// announcement below carries its x_S fence and why the trace audit checks tick order per id
// rather than per-id counts.
// One id per slot, per direction. Mode 2 is a block-wide notification and not a per-subcore
// mailbox: AIV -> AIC, BOTH vector subcores set the id and the Cube waits ONCE (which is what
// turboquant_fused_decode_opt.cpp's CubeScores does on kOptFlagSlotReady); AIC -> AIV, the
// Cube sets once and each subcore waits once. The token model keeps a place per subcore
// because that is what forbids a double set; the hardware wait count is this.
__aicore__ inline constexpr uint16_t FlagWeightsReady(uint32_t slot)
{
    return static_cast<uint16_t>(kReduceFlagWeightsReady + slot);
}
__aicore__ inline constexpr uint16_t FlagProductReady(uint32_t slot)
{
    return static_cast<uint16_t>(kReduceFlagProductReady + slot);
}

// Ascend 950PR is the one product whose Mmad can switch GEMV off. The two legs land different
// cubes in L0C, so the Fixpipe has to match: with GEMV on, C is 1 x N in sub-vectors of 1024
// bytes each holding 64 valid bytes; with it off, C is a conventional M-unaligned cube with M
// padded to 16, which is the shape this operator's audited Fixpipe path already emits. The
// default is off for exactly that reason; the enabled leg is a measurement variant.
template <typename scalar_t, bool GEMV_DISABLED = kGemvDisabledDefault>
class TurboQuantGemmReducer {
public:
    // The operator is instantiated at half and at bfloat16_t. Both are 16-bit and the UB map in
    // ../include/turboquant_fused_decode_opt.hpp is pinned on that: a third element type has to go
    // there first, not arrive here and silently move three cores' Fixpipe landings apart.
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

        // L1, allocated by BOTH roles so the two cores agree on the addresses: the vector
        // subcore writes the weight row by MTE3 and the Cube reads it by MTE1.
        for (uint32_t slot = 0; slot < kReduceSlots; ++slot) {
            pipe_->InitBuffer(weightL1Buf_[slot], kWeightL1Bytes);
            pipe_->InitBuffer(partialL1Buf_[slot], kPartialL1Bytes);
        }
        if ASCEND_IS_AIC {
            pipe_->InitBuffer(operandL0aBuf_, kReduceL0aBytes);
            pipe_->InitBuffer(operandL0bBuf_, kReduceL0bBytes);
            pipe_->InitBuffer(productL0cBuf_, kReduceL0cBytes);
        }
        // Allocated by BOTH roles, like the L1 buffers above and for the same reason: the
        // Cube's Fixpipe destination is an address in the vector core's UB, so the AIC has to
        // walk the same TPipe allocation to know where it is. Guarding these behind
        // `if ASCEND_IS_AIV` is what the first Cube run of this probe did, and the Fixpipe
        // then targeted an unallocated tensor -- su_ccu_mpu_err_t0, a B32 scalar access at a
        // null base on cubecore0 -- while the arithmetic still came out right, which is
        // exactly why a run is not evidence until its exception dumps are read.
        for (uint32_t slot = 0; slot < kReduceSlots; ++slot) {
            // kReduceProductUbBytes carries kReduceBankSkewBytes of tail past the headSize
            // payload, so the slot the Fixpipe is filling and the slot the vector unit is
            // normalising do not present the same UB bank (AKA3003). The skew is never
            // payload.
            pipe_->InitBuffer(productUbBuf_[slot], kReduceProductUbBytes);
        }
        pipe_->InitBuffer(statsUbBuf_, kStatsUbBytes);
        pipe_->InitBuffer(outUbBuf_, headSize * sizeof(scalar_t));
        // Init boundary: drains every pipe before the stage's first GM read, and before the
        // recycled ids are primed.
        AscendC::PipeBarrier<PIPE_ALL>();
        PrimeVectorState();
    }

    // `total` is numTokens * numHeads; `perBlock` the production tiling's reduceTasksPerBlock.
    // `select` answers "does this (token, head) need a combine at all", so a fused token is
    // skipped exactly as production's Reduce() skips it. `finisher` is the shared output stage.
    template <typename Select, typename Finisher>
    __aicore__ inline void Reduce(const uint32_t total, const uint32_t perBlock, Select &select,
                                  Finisher &finisher)
    {
        const uint32_t start = MixBlockIdx() * perBlock;
        if (start >= total) {
            // An empty worker still drains: "this queue is empty" is what the worker boundary
            // needs established, and nothing below was primed, so there is nothing to retire.
            return;
        }
        uint32_t end = start + perBlock;
        if (end > total) {
            end = total;
        }

        // Compact the block's live tasks first, so the ring's slot arithmetic is over a dense
        // index. A fused token contributes no task and must not consume a slot generation.
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
    // One UB arena for the statistics, in 8-float DataBlocks. Two rules shape it, and both
    // were learned the hard way on CAModel rather than read off first:
    //
    //   * Every vector destination is a whole-block base. A vector store to a 4-byte-aligned
    //     address raises vec_err_instr_misalign_t0.
    //   * A Brcb destination is EIGHT DataBlocks, not one. "Eight elements are read from src
    //     in each iteration" and they are filled into eight contiguous DataBlocks, so one
    //     repeat writes 64 floats. Giving the broadcast of m* a single block silently
    //     overwrote the seven blocks behind it, which at one reduction task is invisible --
    //     the clobber happens after the only read -- and at two tasks zeroed the first task's
    //     l*, so 1/(l* + eps) came out 1e20 and every channel of task 0 exported inf. Both
    //     Brcb destinations below therefore own eight blocks, with nothing live behind them.
    //
    //   0      gathered m_s, padding lanes kVectorNegInf
    //   1      gathered l_s, padding lanes zero
    //   2      the weight row w_s, the 32 bytes MTE3 stores into L1
    //   3      w_s * l_s
    //   4-5    ReduceMax / ReduceSum work
    //   6-7    l*, one block per slot: produced by Stats(r) and read by Finish(r) two tasks
    //          later, so a single shared block would be overwritten before its read
    //   8-15   m* broadcast -- a Brcb destination, eight blocks
    //   16-23  NormalizeHeads' sums -- also a Brcb destination, eight blocks
    //   24-31  NormalizeHeads' invs
    //   32     Gather byte offsets of the maxima
    //   33     Gather byte offsets of the masses
    //   34+    the 64-byte landing of each split's statistics tail, kMaxSplits of them
    static constexpr uint32_t kBrcbDstBlocks = 8;
    static constexpr uint32_t kLaneMax = 0 * kFp32PerBlock;
    static constexpr uint32_t kLaneSum = 1 * kFp32PerBlock;
    static constexpr uint32_t kLaneWeight = 2 * kFp32PerBlock;
    static constexpr uint32_t kLaneWeighted = 3 * kFp32PerBlock;
    static constexpr uint32_t kLaneScratch = 4 * kFp32PerBlock;
    static constexpr uint32_t kLaneTotal = 6 * kFp32PerBlock;  // + slot
    static constexpr uint32_t kLaneStar = 8 * kFp32PerBlock;
    static constexpr uint32_t kLaneSums = 16 * kFp32PerBlock;
    static constexpr uint32_t kLaneInvs = 24 * kFp32PerBlock;
    static constexpr uint32_t kLaneMaxOffsets = 32 * kFp32PerBlock;
    static constexpr uint32_t kLaneSumOffsets = 33 * kFp32PerBlock;
    static constexpr uint32_t kLaneTail = 34 * kFp32PerBlock;
    // Each split's tail gets its own 64-byte landing before anything reads it, so an unmasked
    // register read of one split's block cannot touch another split's DMA landing.
    static constexpr uint32_t kTailBlocksPerSplit = 2;
    static constexpr uint32_t kStatsUbFloats =
        kLaneTail + kMaxSplits * kTailBlocksPerSplit * kFp32PerBlock;
    static constexpr uint32_t kStatsUbBytes = kStatsUbFloats * sizeof(float);
    // The two Brcb destinations each own their eight blocks outright.
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
    // The byte stride between two splits' tail landings, which is what the Gather tables step.
    static constexpr uint32_t kTailStrideBytes =
        kTailBlocksPerSplit * kFp32PerBlock * sizeof(float);
    // The ring index arithmetic is over a dense live-task index, so the compacted id list is
    // what bounds a block's share.
    static constexpr uint32_t kMaxTasksPerBlock = 64;

    __aicore__ inline AscendC::LocalTensor<float> Stats() { return statsUbBuf_.template Get<float>(); }

    // Two one-time jobs on the vector subcore that owns the stage.
    //
    // The weight slot is a whole 512-byte fractal because GEMV wants its 1 x K row 512-byte
    // aligned, but only the first numSplits_ floats are ever rewritten. Zeroing the rest once
    // matters for correctness and not tidiness: section 3 floods GM with 0x7FC0BEEF, an fp32 quiet
    // NaN, and a stale NaN in a lane the Mmad reads would poison the whole product.
    //
    // The Gather tables are byte offsets into the statistics landings, one per split. They are
    // written by the SCALAR unit and handed to the vector pipe through S_V, which is this
    // repository's idiom for "a lane the vector pipe cannot address because it is not on a
    // 32-byte boundary" (SyncScalarToVector). This is table setup at Init, not per-element
    // arithmetic, so it is the address scheduling section 2 calls control flow rather than the scalar
    // dequant/GEMV/softmax computation section 2 forbids.
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

    // ------------------------------------------------------------------------------------
    // AIC MTE2: the partial cube, GM -> L1, in the layout Mmad's B operand needs
    // ------------------------------------------------------------------------------------
    //
    // Mmad wants B as K x N in Zn. The workspace already holds [S, D] row-major, which IS
    // K x N, so only the fractal layout is in question, and the inline DN2NZ conversion
    // settles it: the API reference states DN->NZ is "equivalent to the ND-to-ZN format
    // conversion", so feeding the ND rows lands the ZN image of that same matrix in L1 with
    // the split axis as C0. A non-transposed Load2DV2 then leaves it alone (Zn2Zn).
    //
    // That is the whole reason this operand is viable. The transposed b32 path merges two
    // consecutive 16x8 fractals into a 16x16 square and so needs a source Zn cube with
    // Col == 16, which a K = S <= 8 operand cannot provide; it also constrains kStep to a
    // multiple of 2. Avoiding the transpose avoids all of it.
    __aicore__ inline void Fetch(const uint32_t r)
    {
        if ASCEND_IS_AIC {
        const uint32_t slot = kReduceSlotOf(r);
        if (r >= kReduceSlots) {
            // The cube of task r - 2 must have reached L0B before this overwrites it. Posted
            // by Gemm only when a later fetch will consume it.
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(slot));
        }
        AscendC::Dn2NzParams params;
        params.dnNum = kDn2NzDnNum;
        params.nValue = static_cast<uint16_t>(headSize_);
        // dValue is the C0 axis: the splits. Below kC0Floats the conversion zero-fills the
        // destination to the 32-byte boundary, which is the second of the two independent
        // guards on the padding lanes (the first is the zeroed weight lanes).
        params.dValue = numSplits_;
        params.srcDnMatrixStride = 0;
        // Not headSize_: the statistics tail shares the row, so the source row stride is the
        // full partial stride.
        params.srcDValue = kPartialStride;
        params.dstNzC0Stride = static_cast<uint16_t>(headSize_);
        params.dstNzNStride = kDn2NzDstNzNStride;
        params.dstNzMatrixStride = 0;
        AscendC::DataCopy(partialL1Buf_[slot].template Get<float>(),
                          workspaceGm_[PartialRow(taskIds_[r], 0)], params);
        SyncEvent<AscendC::HardEvent::MTE2_MTE1>();
        }
    }

    // ------------------------------------------------------------------------------------
    // AIV0: the statistics merge, and the weight row into L1
    // ------------------------------------------------------------------------------------
    //
    // The sequential online rescale the Vector reducer runs is replaced by a two-pass form:
    // every m_s is known up front, so m* is one reduction and the S weights are ONE 8-lane
    // Exp rather than S dependent Exp pairs. That restructuring is available to a vector-only
    // reducer too; it is not what the Cube buys. What the Cube buys is the 256-wide part --
    // the two BroadcastMuls and the Add per split, four repeats each -- which leaves the
    // vector pipe entirely.
    __aicore__ inline void Stats(const uint32_t r)
    {
        if ASCEND_IS_AIV {
        const uint32_t slot = kReduceSlotOf(r);
        if (AscendC::GetSubBlockIdx() != 0) {
            // AIV1 carries no data. It still posts the credit that paces the Cube against it:
            // the Cube waits once per subcore, so without this post it could run two slot
            // generations ahead of AIV1's outstanding product wait and drop a SET.
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

        // Padding lanes: a finite -FLT_MAX, not -inf, so that m_s - m* underflows to -inf and
        // Exp gives exactly 0 instead of the NaN an inf - inf would produce when every split
        // of a task is empty.
        AscendC::Duplicate(maxes, kVectorNegInf, kFp32PerBlock);
        AscendC::Duplicate(sums, 0.0f, kFp32PerBlock);

        // The landings are shared across tasks, and the previous task's Gather, Sub and Exp
        // may still be reading them: this task's MTE2 burst has to land BEHIND that vector
        // work. Without this fence the two-task case already returns inf, because task r+1's
        // statistics overwrite task r's while they are being reduced. The shipping
        // TurboQuantPartialReducer carries the same SyncVectorToMte2 before each of its GM
        // reads. This is a WAR hazard across two pipes, not a reordering the compiler sees.
        SyncVectorToMte2();
        for (uint32_t split = 0; split < numSplits_; ++split) {
            const uint64_t row = PartialRow(taskIds_[r], split) + headSize_;
            // One 64-byte burst per split carries both lanes: the max sits at kPartialMaxLane
            // and the sum at kPartialSumLane, 8 floats apart, inside the same tail.
            AscendC::DataCopy(tail[split * kTailBlocksPerSplit * kFp32PerBlock], workspaceGm_[row],
                              kTailBlocksPerSplit * kFp32PerBlock);
        }
        SyncMte2ToVector();

        // The Duplicates above wrote the lanes the Gathers below overwrite.
        AscendC::PipeBarrier<PIPE_V>();

        // Pack lane 0 of each split's landing into contiguous lanes.
        //
        // The obvious way to do this -- one single-element Adds per split, writing maxes[s] --
        // is ILLEGAL on this core, and the simulator says so rather than silently working: a
        // vector store whose destination is not 32-byte aligned raises
        // vec_err_instr_misalign_t0, and maxes[1] is four bytes in. Vector repeat strides are
        // per-repeat and in whole blocks, so no repeat pattern expresses a 4-byte pick either.
        // Gather does: it takes a table of BYTE offsets and writes its result contiguously
        // from a block-aligned base, which is the primitive the KV codec's expand already uses.
        AscendC::Gather(maxes, tail, stats[kLaneMaxOffsets].template ReinterpretCast<uint32_t>(),
                        kGatherSrcBase, numSplits_);
        AscendC::Gather(sums, tail, stats[kLaneSumOffsets].template ReinterpretCast<uint32_t>(),
                        kGatherSrcBase, numSplits_);
        AscendC::PipeBarrier<PIPE_V>();

        // m* over all kC0Floats lanes: the padding lanes are kVectorNegInf, so reducing the
        // whole block is the same answer as reducing the live prefix and needs no mask.
        AscendC::ReduceMax<float>(scratch, maxes, scratch[kFp32PerBlock], kFp32PerBlock, false);
        AscendC::Brcb(star, scratch, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
        AscendC::Sub(weight, maxes, star, kFp32PerBlock);
        AscendC::Exp(weight, weight, kFp32PerBlock);
        AscendC::Mul(weighted, weight, sums, kFp32PerBlock);
        // l* lands in this slot's own block: it is produced here but consumed by Finish two
        // tasks later, so a single shared lane would be overwritten before its read.
        AscendC::ReduceSum<float>(stats[kLaneTotal + slot * kFp32PerBlock], weighted, scratch,
                                  kFp32PerBlock);

        SyncVectorToMte3();
        AscendC::DataCopy(weightL1Buf_[slot].template Get<float>(), weight, kFp32PerBlock);
        // the fence: the announcement must be preceded by the x_S pair on the pipe whose
        // work it reports. On a recycled id this is the precondition, not a nicety -- an
        // unfenced announcement balances every count and still releases the Cube onto a
        // weight row MTE3 has not landed.
        SyncEvent<AscendC::HardEvent::MTE3_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(FlagWeightsReady(slot));
        }
    }

    // ------------------------------------------------------------------------------------
    // AIC: the GEMV
    // ------------------------------------------------------------------------------------
    __aicore__ inline void Gemm(const uint32_t r, const uint32_t tasks)
    {
        if ASCEND_IS_AIC {
        const uint32_t slot = kReduceSlotOf(r);
        // One wait, not one per subcore: a mode-2 notification is satisfied when both vector
        // subcores have set the id.
        AscendC::CrossCoreWaitFlag(FlagWeightsReady(slot));

        const AscendC::LocalTensor<float> leftL0a = operandL0aBuf_.template Get<float>();
        const AscendC::LocalTensor<float> rightL0b = operandL0bBuf_.template Get<float>();
        const AscendC::LocalTensor<float> product = productL0cBuf_.template Get<float>();

        // A: 1 x K, M x K Nz, not transposed. mStep counts 16-element M steps and kStep
        // 32-byte K steps, so both are 1 for a single 1 x 8 float row.
        AscendC::LoadData2DParamsV2 aParams;
        aParams.mStartPosition = 0;
        aParams.kStartPosition = 0;
        aParams.mStep = 1;
        aParams.kStep = 1;
        aParams.srcStride = 1;
        aParams.dstStride = 1;
        aParams.ifTranspose = false;
        AscendC::LoadData(leftL0a, weightL1Buf_[slot].template Get<float>(), aParams);

        // B: the Zn cube DN2NZ already built. Non-transposed, so the b32 transpose
        // constraints do not apply: mStep walks N in 16s, kStep walks K in C0 blocks, and
        // K <= kC0Floats makes that exactly one.
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
        // The LOGICAL depth, numSplits_, and not the aligned one. kAlignedSplits describes the
        // OPERAND LAYOUT -- a float K block is 8 elements, so every legal split count occupies
        // one C0 block and needs no K tiling -- and passing it here instead told the Cube to
        // accumulate eight products, which at splits < 8 multiplied the L1 rows DN2NZ had not
        // written and returned inf. With k at the real depth those rows are not reached; the
        // zeroed weight lanes remain the second guard for any lane the hardware reads anyway.
        mmad.k = static_cast<uint16_t>(numSplits_);
        // cmatrixInitVal, not the doc's ccubeInitVal: this toolkit's MmadParams names it the
        // former. true zeroes C, which is what a single-Mmad GEMV wants.
        mmad.cmatrixInitVal = true;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        // disableGemv exists only on the arch that has it to disable. dav-3510 is the target,
        // so the knob is real here; the #if keeps the variant compilable elsewhere.
        mmad.disableGemv = GEMV_DISABLED;
#else
        static_cast<void>(GEMV_DISABLED);
#endif
        AscendC::Mmad(product, leftL0a, rightL0b, mmad);

        // The cube credit, posted only when a later fetch will consume it. Unconditional, it
        // leaves the id set on a short worker for the next worker's prologue to find.
        if (r + kReduceSlots < tasks) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(slot));
        }

        SyncMatrixToFixpipe();
        AscendC::Fixpipe<float, float, vllm_ascend::turboquant::kFixpipeToUb>(
            productUbBuf_[slot].template Get<float>(), product,
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>(
                static_cast<uint16_t>(headSize_), 1, static_cast<uint16_t>(kFractalRows),
                headSize_));
        // Drains the Fixpipe before the announcement releases the vector unit onto its rows:
        // the section 6.6 fence for this id, on the pipe that did the work.
        AscendC::PipeBarrier<PIPE_FIX>();
        SyncEvent<AscendC::HardEvent::FIX_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(FlagProductReady(slot));
        }
    }

    // ------------------------------------------------------------------------------------
    // AIV0: normalise, finish, write -- all shared production code
    // ------------------------------------------------------------------------------------
    template <typename Finisher>
    __aicore__ inline void Finish(const uint32_t r, Finisher &finisher)
    {
        if ASCEND_IS_AIV {
        const uint32_t slot = kReduceSlotOf(r);
        AscendC::CrossCoreWaitFlag(FlagProductReady(slot));
        if (AscendC::GetSubBlockIdx() != 0) {
            // AIV1's whole job: consume its copy of the token. Without this the phase ends
            // with the flag set, which the next worker's prologue would then mistake for its
            // own credit.
            return;
        }

        const AscendC::LocalTensor<float> stats = Stats();
        const AscendC::LocalTensor<float> acc = productUbBuf_[slot].template Get<float>();
        const uint32_t task = taskIds_[r];
        const uint32_t token = task / numHeads_;
        const uint32_t head = task % numHeads_;

        // The same close-out TurboQuantPartialReducer runs, on the same shared helpers: the
        // 1/(l* + eps) scale, then the output stage, then the cast and the GM burst.
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

// =============================================================================================
// The dense batched form: M = kGemmBatchRows packed query rows, N = 16-channel slices
// =============================================================================================
//
// The algebra, the block-diagonal A operand and the reason the subcore halves are static are
// all in include/turboquant_gemm_reduce.hpp, which the host gate compiles too. What this file
// adds is the schedule.
//
// Per batch of up to kGemmBatchRows consecutive live tasks:
//
//   AIV_c  Stats(b)   ONE DataCopy for every (row, split) statistics tail of the batch, then
//                     per row: m* = max m_s, w_s = exp(m_s - m*), l* = sum w_s l_s, and a
//                     32-byte MTE3 store of w into the A cube's row. Both subcores do this,
//                     for their own half of the batch, and each announces weights_ready --
//                     so the Cube's single wait is a genuine two-producer mode-2 quorum.
//   AIC    Gemm(b)    the A cube into L0A ONCE, then kGemmSlices iterations of
//                     {batched DN2NZ ingest, Load2DV2, Mmad(16x16x128), Fixpipe}, the inner
//                     slice ring carrying real split credits so the fetch of slice j + 1
//                     overlaps the Mmad of slice j.
//   AIV_c  Finish(b)  its own eight panel rows: NormalizeHeads, the shared finisher per token
//                     run, WriteHeads -- one GM burst for the whole half, because a batch is a
//                     run of consecutive tasks.
//
// WHAT THIS CHANGES AND WHAT IT DOES NOT
//
// It does NOT reduce MTE2 bytes: the combine must read every partial, and the M = 1 form
// already fetched a whole task in one burst, so neither form moves less. It does not reduce MAC
// count either -- the zeros in the block-diagonal A cost exactly what M = 1's row padding cost.
// What it reduces is the vector critical path (both subcores now carry half the combines each,
// where M = 1 left all of them on subcore 0), the cross-core handshake count (two
// announcements per BATCH rather than per task, so kGemmBatchRows times fewer), and MTE1 (one
// A load per batch plus one slice load per slice, against two loads per task). The MTE2
// instruction count actually RISES, from one burst per task to two per slice, which is the
// price of the static subcore halves; the bursts are 4 KB each and the bytes are unchanged.
template <typename scalar_t, bool GEMV_DISABLED = kGemvDisabledDefault>
class TurboQuantBatchedGemmReducer {
public:
    // The operator is instantiated at half and at bfloat16_t. Both are 16-bit and the UB map in
    // ../include/turboquant_fused_decode_opt.hpp is pinned on that: a third element type has to go
    // there first, not arrive here and silently move three cores' Fixpipe landings apart.
    static_assert(sizeof(scalar_t) == 2,
                  "turboquant_fused_decode_opt.hpp's kUbEnd/kBasisUbBytes/kReducerUbBytes assume a "
                  "16-bit output element");

    __aicore__ inline explicit TurboQuantBatchedGemmReducer(AscendC::TPipe *pipe) : pipe_(pipe) {}

    // `sharedPool`, when given, is the attention phase's transient UB pool
    // (TurboQuantVectorDecodeService::TransientPool()). Every UB buffer below is then allocated
    // from a pool aliased onto it, which costs zero bytes -- InitBufPool's shared overload sets
    // the new pool's startAddr_ to the shared one's and does not advance the hardware pool. The
    // bytes are safe to reuse because AscendC::SyncAll<false>() stands between Process() and
    // the reduction; see the budget in include/turboquant_gemm_reduce.hpp for why it is those
    // three buffers and not a bare TPipe::Reset().
    //
    // Without a pool -- the standalone probe -- the buffers come straight off the pipe, which
    // is what every CAModel rung of this reducer measures.
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
    // Every UB allocation goes through here, so the pooled and unpooled legs cannot drift in
    // the ORDER they allocate -- which matters more than it looks: the panels are a
    // dual-destination Fixpipe destination, and the AIC names them by the offset its own TPipe
    // handed out, so all three roles have to walk the same list.
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

        // L1, allocated by BOTH roles so the three cores agree on the addresses: each vector
        // subcore writes its half of the weight cube by MTE3 and the Cube reads all of it by
        // MTE1.
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
        // Allocated by BOTH roles, for the reason fault 2 of the GEMV report records: the
        // Fixpipe's destination is an address in the vector cores' UB, so the AIC has to walk
        // the same TPipe allocation to know where it is. Guarding these behind `if ASCEND_IS_AIV`
        // makes the Cube target an unallocated tensor -- su_ccu_mpu_err_t0 on cubecore0 -- while
        // the arithmetic still comes out right.
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
    // `total` is numTokens * numHeads; `perBlock` the production tiling's reduceTasksPerBlock.
    template <typename Select, typename Finisher>
    __aicore__ inline void Reduce(const uint32_t total, const uint32_t perBlock, Select &select,
                                  Finisher &finisher)
    {
        const uint32_t start = MixBlockIdx() * perBlock;
        if (start >= total) {
            // An empty worker still drains, but nothing here was primed, so there is nothing
            // to retire: the phase join is the caller's.
            return;
        }
        uint32_t end = start + perBlock;
        if (end > total) {
            end = total;
        }

        // Batch the block's live tasks into runs of CONSECUTIVE ids. Consecutive is required,
        // not preferred: dnNum walks one uniform source stride, so a gap -- which a fused token
        // leaves, exactly as production's Reduce() skips it -- has to end the batch. Every role
        // runs this loop and gets the same answer, which is what lets the Cube address the
        // batch without being told.
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
            // Only now may batch slot b % kBatchSlots be restaged: Finish(b) has read the UB
            // panel, so the weight store for b + kBatchSlots is what frees it.
            if (b + kBatchSlots < batches) {
                Stats(b + kBatchSlots);
            }
        }
        DrainCubeRing();
    }

private:
    // One UB arena for the statistics, in 8-float DataBlocks. Same two rules as the M = 1
    // form's arena -- every vector destination is a whole-block base, and a Brcb destination is
    // EIGHT DataBlocks and not one -- plus one the batched form adds: NormalizeHeads' own Brcb
    // writes 64 floats from `sums[j * 8]` for j up to heads - 1, so `sums` needs
    // (heads - 1) * 8 + 64 floats and not the heads * 8 its doc comment says. At
    // kGemmPanelRows = 8 rows that is 120 floats, so sixteen blocks are reserved for it. The
    // shipping reducer sizes that buffer by the doc comment and overruns it by 192 bytes; it is
    // benign there because WriteHeads overwrites the bytes before anyone reads them, which is
    // not a property to inherit.
    //
    //   0        gathered m_s, padding lanes kVectorNegInf
    //   1        gathered l_s, padding lanes zero
    //   2        w_s * l_s
    //   3        pad
    //   4-5      ReduceMax / ReduceSum work
    //   6-7      pad, so nothing live sits behind the broadcast below
    //   8-15     m* broadcast -- a Brcb destination, eight blocks
    //   16-23    the weight rows w_s, ONE BLOCK PER PANEL ROW -- see below
    //   24-39    l*, one lane per panel row per batch slot: produced by Stats(b) and read by
    //            Finish(b), which is two batches later, so a shared lane would be overwritten
    //   40-55    NormalizeHeads' sums -- a Brcb destination, sixteen blocks (see above)
    //   56-63    NormalizeHeads' invs
    //   64       Gather byte offsets of the maxima
    //   65       Gather byte offsets of the masses
    //   66+      the statistics landing: one 64-byte block pair per (panel row, split), filled
    //            by ONE strided DataCopy for the whole batch
    //
    // The weight rows get a block EACH, and that is a WAR fix rather than convenience. With one
    // shared block the loop is {ReduceRow writes it on V, SyncVectorToMte3, MTE3 reads it}, and
    // the next row's ReduceRow -- on V, a different in-order queue -- can overwrite the block
    // while MTE3's 32-byte copy of the previous row is still outstanding. No rung exposed it:
    // the exact fixture gives every row the same statistics, so the wrong weights are the right
    // ones, and the varied fixture's window is a handful of ticks wide. Per-row destinations
    // remove the hazard by construction, which is cheaper than arguing about the window, and
    // they let the batch's stores go out as one run behind a single fence.
    static constexpr uint32_t kBrcbDstBlocks = 8;
    static constexpr uint32_t kSumsBlocks = 16;
    static constexpr uint32_t kLaneMax = 0 * kFp32PerBlock;
    static constexpr uint32_t kLaneSum = 1 * kFp32PerBlock;
    static constexpr uint32_t kLaneWeighted = 2 * kFp32PerBlock;
    static constexpr uint32_t kLaneScratch = 4 * kFp32PerBlock;
    static constexpr uint32_t kLaneStar = 8 * kFp32PerBlock;
    static constexpr uint32_t kLaneWeight = 16 * kFp32PerBlock;  // + row
    static constexpr uint32_t kLaneTotal = 24 * kFp32PerBlock;   // + slot * rows + row
    static constexpr uint32_t kLaneSums = 40 * kFp32PerBlock;
    static constexpr uint32_t kLaneInvs = 56 * kFp32PerBlock;
    static constexpr uint32_t kLaneMaxOffsets = 64 * kFp32PerBlock;
    static constexpr uint32_t kLaneSumOffsets = 65 * kFp32PerBlock;
    static constexpr uint32_t kLaneTail = 66 * kFp32PerBlock;
    // Each (row, split) tail gets its own 64-byte landing, so an unmasked register read of one
    // split's block cannot touch another's.
    static constexpr uint32_t kTailBlocksPerSplit = 2;
    static constexpr uint32_t kTailBlocks = kGemmPanelRows * kMaxSplits * kTailBlocksPerSplit;
    static constexpr uint32_t kStatsUbFloats = kLaneTail + kTailBlocks * kFp32PerBlock;
    static constexpr uint32_t kStatsUbBytes = kStatsUbFloats * sizeof(float);
    // The contract header's integration budget is computed from kGemmStatsHeaderBlocks, so the
    // two have to agree or that budget is describing a different arena than this one.
    static_assert(kLaneTail == kGemmStatsHeaderBlocks * kFp32PerBlock,
                  "the contract header's stats header block count must match this layout");
    static_assert(kStatsUbBytes == kGemmStatsUbBytes,
                  "the contract header's stats arena size must match this allocation");
    static constexpr uint32_t kTailStrideBytes = kTailBlocksPerSplit * kFp32PerBlock * sizeof(float);
    // The gap, in 32-byte blocks, between one (row, split) tail and the next in GM: the whole
    // partial stride less the tail itself. This is what lets ONE DataCopy gather the batch.
    static constexpr uint32_t kTailGapBlocks =
        (kPartialStride - kPartialTail) / kFp32PerBlock;
    static_assert((kPartialStride - kPartialTail) % kFp32PerBlock == 0,
                  "the tail gap must be a whole number of 32-byte blocks");
    static_assert(kTailGapBlocks == 32);
    // Every broadcast destination owns its blocks outright, and nothing live sits behind one.
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

    // 1: the dual-destination Fixpipe splits mSize rows between the two subcores' UB, each
    // half landing at its own subcore's base. Production's own idiom
    // (TurboQuantCubeMm::ComputeProductToUb).
    static constexpr uint8_t kGemmDualDstSplitM = 1;

    // Every UB byte Init() allocates, against the contract header's integration claim. This is
    // what makes kGemmReducerUbBytes the claim rather than an estimate of it, and it is the
    // number the operator's UB deficit is computed from.
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

    // Two one-time jobs, done by EACH vector subcore for its own half.
    //
    // The A cube's off-diagonal rows have to be zero for the diagonal row's output to be
    // right, they are written once here, and they are never rewritten. Each subcore zeroes its
    // OWN eight fractals, which is why the halves are static: nothing orders a block's two
    // vector subcores against each other at Init, so a cube whose fractal ownership moved with
    // the live count could have its zero-fill land after the weight it is meant to precede.
    //
    // The Gather tables are byte offsets into this subcore's own statistics landing, one per
    // split, written by the SCALAR unit and handed to the vector pipe through S_V -- the
    // repository's idiom for a lane the vector pipe cannot address because it is not on a
    // 32-byte boundary. Table setup at Init, not per-element arithmetic.
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

    // ------------------------------------------------------------------------------------
    // AIV_c: the statistics merge for its own half, and its rows of the block-diagonal cube
    // ------------------------------------------------------------------------------------
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
            // ONE strided burst for every (row, split) tail of this subcore's half. The tails
            // sit kPartialStride floats apart and the rows are consecutive tasks, so the whole
            // half is one arithmetic progression -- which is the one thing the batched form
            // does buy on MTE2.
            //
            // The landing is shared across batches and the previous batch's Gather, Sub and Exp
            // may still be reading it, so this burst has to land BEHIND that vector work. This
            // is a WAR hazard across two pipes, not a reordering the compiler sees; without it
            // the two-batch case returns inf.
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
            // One fence for the whole half, then the stores: each row wrote its own weight
            // block, so there is nothing for MTE3 to lose to the next row's vector work.
            SyncVectorToMte3();
            for (uint32_t i = 0; i < mine; ++i) {
                AscendC::DataCopy(cube[kWeightRowOffsetFloats(firstRow + i)],
                                  stats[kLaneWeight + i * kFp32PerBlock], kFp32PerBlock);
            }
        }
        // A dead row's diagonal block is zeroed rather than left holding a previous batch's
        // weights. Its product row is garbage either way -- the block-diagonal A confines a
        // dead fractal's contribution to its own output row, and no vector instruction reads
        // that row -- but garbage that does not depend on which batch ran before it is the
        // difference between a reproducible run and a puzzling one.
        for (uint32_t i = mine; i < kGemmPanelRows; ++i) {
            AscendC::DataCopy(cube[kWeightRowOffsetFloats(firstRow + i)], zero, kFp32PerBlock);
        }
        // §6.6's fence: the announcement must be preceded by the x_S pair on the pipe whose
        // work it reports. On a recycled id this is the precondition and not a nicety -- an
        // unfenced announcement balances every count and still releases the Cube onto a weight
        // row MTE3 has not landed.
        SyncEvent<AscendC::HardEvent::MTE3_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(FlagWeightsReady(slot));
        }
    }

    // One row's online-softmax statistics, from its own landing into the weight block and the
    // slot's l* lane. The sequential rescale the vector reducer runs is replaced by a two-pass
    // form: every m_s is known up front, so m* is one reduction and the S weights are ONE
    // 8-lane Exp rather than S dependent Exp pairs.
    __aicore__ inline void ReduceRow(const AscendC::LocalTensor<float> &stats, const uint32_t row,
                                     const uint32_t slot)
    {
        const AscendC::LocalTensor<float> maxes = stats[kLaneMax];
        const AscendC::LocalTensor<float> sums = stats[kLaneSum];
        // This row's OWN weight block: the 32 bytes MTE3 will store into the A cube's row.
        const AscendC::LocalTensor<float> weight = stats[kLaneWeight + row * kFp32PerBlock];
        const AscendC::LocalTensor<float> star = stats[kLaneStar];
        const AscendC::LocalTensor<float> weighted = stats[kLaneWeighted];
        const AscendC::LocalTensor<float> scratch = stats[kLaneScratch];
        // The landing's row stride is numSplits_ tail pairs and NOT kMaxSplits: the one
        // DataCopy that fills it has dstStride = 0, so it packs (row, split) contiguously at
        // the LOGICAL split count. Indexing by kMaxSplits instead agrees only at
        // numSplits_ == kMaxSplits, and at splits < 8 it reads row numSplits_ and beyond off
        // the end of the filled region -- which is zeroed, so m* came out 0, every weight
        // exp(0) = 1, l* = 0, and 1 / (l* + eps) = 1e20 exported inf for exactly row
        // numSplits_ of every batch. The split-count sweep is what found it; at splits = 8 the
        // two strides coincide and every rung passes.
        const AscendC::LocalTensor<float> landing =
            stats[kLaneTail + row * numSplits_ * kTailBlocksPerSplit * kFp32PerBlock];

        // Padding lanes: a finite -FLT_MAX, not -inf, so m_s - m* underflows to -inf and Exp
        // gives exactly 0 instead of the NaN an inf - inf would produce for an empty row.
        AscendC::Duplicate(maxes, kVectorNegInf, kFp32PerBlock);
        AscendC::Duplicate(sums, 0.0f, kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();

        // Pack lane 0 of each split's landing into contiguous lanes. The obvious way -- one
        // single-element Adds per split writing maxes[s] -- is illegal on this core: a vector
        // store whose destination is not 32-byte aligned raises vec_err_instr_misalign_t0, and
        // maxes[1] is four bytes in. Repeat strides are per-repeat and in whole blocks, so no
        // repeat pattern expresses a 4-byte pick either. Gather does.
        AscendC::Gather(maxes, landing, stats[kLaneMaxOffsets].template ReinterpretCast<uint32_t>(),
                        kGatherSrcBase, numSplits_);
        AscendC::Gather(sums, landing, stats[kLaneSumOffsets].template ReinterpretCast<uint32_t>(),
                        kGatherSrcBase, numSplits_);
        AscendC::PipeBarrier<PIPE_V>();

        // m* over all kC0Floats lanes: the padding lanes are kVectorNegInf, so reducing the
        // whole block is the same answer as reducing the live prefix and needs no mask. The
        // weight block's padding lanes then come out exactly 0, which is the first of the two
        // guards on a split count below 8 (DN2NZ's zero fill is the second).
        AscendC::ReduceMax<float>(scratch, maxes, scratch[kFp32PerBlock], kFp32PerBlock, false);
        AscendC::Brcb(star, scratch, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
        AscendC::Sub(weight, maxes, star, kFp32PerBlock);
        AscendC::Exp(weight, weight, kFp32PerBlock);
        AscendC::Mul(weighted, weight, sums, kFp32PerBlock);
        AscendC::ReduceSum<float>(
            stats[kLaneTotal + (slot * kGemmPanelRows + row) * kFp32PerBlock], weighted, scratch,
            kFp32PerBlock);
    }

    // ------------------------------------------------------------------------------------
    // AIC: one A cube load, then kGemmSlices dense Mmads over 16-channel slices
    // ------------------------------------------------------------------------------------
    __aicore__ inline void PrimeCubeRing()
    {
        if ASCEND_IS_AIC {
        // The A cube's L0A region is reused by the next batch, so its free credit is a
        // feedback cycle of its own and has to be marked before it is waited on.
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

        // ONE wait for a genuine quorum: a mode-2 notification is satisfied when both vector
        // subcores have set the id, and in this form both of them really staged half the cube.
        AscendC::CrossCoreWaitFlag(FlagWeightsReady(slot));
        // The drain that makes this wait's release tick readable in the trace. A cross-core
        // wait blocks only its own pipe's issue, so the x_S pair behind it stalls instruction
        // issue until that pipe has retired past the wait -- and that scalar WaitFlag's logged
        // tick IS when it ran. Without it the auditor reads a floor, not a stall.
        SyncEvent<AscendC::HardEvent::MTE2_S>();

        // The A operand: M = 16 is one fractal row, K = 128 is kGemmBatchRows C0 blocks, and
        // the cube is already in exactly that layout, so srcStride = dstStride = 1 walks the
        // fractals in order. Loaded ONCE and reused by all kGemmSlices Mmads.
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

        // The inner slice ring is primed and drained INSIDE the batch, so no credit crosses a
        // batch boundary (§1a) and a short worker cannot leave one set for the next to find.
        for (uint32_t ss = 0; ss < kSliceSlots; ++ss) {
            const event_t id = static_cast<event_t>(kGemmSliceEventFirst + ss);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(id);  // slice_free
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(id);     // l0b_free
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(id);      // l0c_free
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
        // The A cube's region is free for the next batch only now, after its last Mmad.
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(kGemmL0aEvent));

        // Drains the Fixpipe before the announcement releases either vector core onto its
        // panel: the §6.6 fence for this id, on the pipe that did the work.
        AscendC::PipeBarrier<PIPE_FIX>();
        SyncEvent<AscendC::HardEvent::FIX_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(FlagProductReady(slot));
        }
    }

    // The batched ingest. Mmad wants B as K x N in Zn; the workspace holds [S, D] row-major and
    // DataCopy's inline DN2NZ conversion is "equivalent to the ND-to-ZN format conversion", so
    // feeding the ND rows of ONE 16-channel column band lands that band's ZN image with the
    // split axis as C0 -- one 512-byte fractal per packed row. dnNum repeats that for the whole
    // half at a uniform source stride, which is the batched MTE2 burst, and dstNzMatrixStride
    // of one fractal lands them contiguously so Load2DV2 needs no stride arithmetic.
    __aicore__ inline void FetchSlice(const uint32_t ss, const uint32_t firstTask,
                                      const uint32_t rows0, const uint32_t rows1,
                                      const uint32_t j)
    {
        const AscendC::LocalTensor<float> sliceL1 = sliceL1Buf_[ss].template Get<float>();
        AscendC::Dn2NzParams params;
        params.nValue = static_cast<uint16_t>(kSliceDn2NzNValue);
        // dValue is the C0 axis: the splits. Below kC0Floats the conversion zero-fills the
        // destination to the 32-byte boundary, which is the second of the two guards on the
        // padding lanes; the zeroed weight lanes are the first.
        params.dValue = numSplits_;
        params.srcDnMatrixStride = kSliceSrcMatrixStrideFloats(numSplits_);
        // Not headSize_: the statistics tail shares the row, so the source d-axis stride is the
        // full partial stride.
        params.srcDValue = kPartialStride;
        params.dstNzC0Stride = static_cast<uint16_t>(kSliceDn2NzDstNzC0Stride);
        params.dstNzNStride = static_cast<uint16_t>(kSliceDn2NzDstNzNStride);
        params.dstNzMatrixStride = kSliceDn2NzDstMatrixStrideFloats;
        const uint32_t column = j * kGemmSliceCols;
        // Two bursts, because the subcore halves are static: subcore 0's rows occupy fractals
        // 0.. and subcore 1's fractals kGemmPanelRows.., with the dead fractals between them
        // untouched. The contract header says why that is the right trade.
        params.dnNum = static_cast<uint16_t>(rows0);
        AscendC::DataCopy(sliceL1, workspaceGm_[PartialRowOf(firstTask) + column], params);
        if (rows1) {
            params.dnNum = static_cast<uint16_t>(rows1);
            AscendC::DataCopy(sliceL1[kGemmFirstRowOf(1) * kSliceFractalFloats],
                              workspaceGm_[PartialRowOf(firstTask + rows0) + column], params);
        }
    }

    // Non-transposed, so none of the b32 transpose constraints apply: that path merges two
    // consecutive 16x8 fractals into a 16x16 square and needs a source Zn cube with Col == 16,
    // which this operand cannot provide. N = 16 is one fractal row and K = 128 is
    // kGemmBatchRows C0 blocks, laid out in fractal order by the ingest above.
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
        // Dense on both axes, every batch: kGemmBatchRows output rows and kGemmSliceCols
        // channels. m and k are constant because the subcore halves are static -- a ragged
        // batch's dead fractals are read and their products confined to their own rows, which
        // is the block-diagonal containment the contract header states.
        mmad.m = static_cast<uint16_t>(kGemmBatchRows);
        mmad.n = static_cast<uint16_t>(kGemmSliceCols);
        mmad.k = static_cast<uint16_t>(kGemmKMax);
        // cmatrixInitVal, not the doc's ccubeInitVal: this toolkit's MmadParams names it the
        // former. true zeroes C, which is what a single Mmad per slice wants.
        mmad.cmatrixInitVal = true;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        // disableGemv exists only on the arch that has it to disable. With M = 16 there is no
        // GEMV case for it to affect, so the two legs should be indistinguishable here -- which
        // is itself the measurement: in the M = 1 form they were not.
        mmad.disableGemv = GEMV_DISABLED;
#else
        static_cast<void>(GEMV_DISABLED);
#endif
        AscendC::Mmad(productL0cBuf_[ss].template Get<float>(),
                      operandL0aBuf_.template Get<float>(),
                      operandL0bBuf_[ss].template Get<float>(), mmad);
    }

    // The dual-destination Fixpipe: mSize rows split in half, each half landing at the matching
    // subcore's own UB base as rows 0..kGemmPanelRows-1 of its panel. dstStride is the PANEL's
    // row stride and not nSize, because a slice is a 16-channel column band of a D-wide row.
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

    // ------------------------------------------------------------------------------------
    // AIV_c: normalise, finish and write its own half -- all shared production code
    // ------------------------------------------------------------------------------------
    template <typename Finisher>
    __aicore__ inline void Finish(const uint32_t b, Finisher &finisher)
    {
        if ASCEND_IS_AIV {
        const uint32_t slot = kReduceSlotOf(b);
        const uint32_t sub = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
        AscendC::CrossCoreWaitFlag(FlagProductReady(slot));
        // Both subcores drain this channel, with rows or without: the scalar pair behind the
        // wait is what makes its release tick measurable, and it is issued before the
        // zero-row early return so an idle subcore participates in the drain too (§B).
        SyncEvent<AscendC::HardEvent::MTE3_S>();

        const uint32_t live = batchLive_[b];
        const uint32_t mine = kGemmLiveRowsOf(live, sub);
        if (mine == 0) {
            // live == 1: this subcore has no query row to reduce. It still waited, still
            // drained, and still posts the weights credit that paces the Cube behind it.
            return;
        }

        const AscendC::LocalTensor<float> stats = Stats();
        const AscendC::LocalTensor<float> panel = panelUbBuf_[slot].template Get<float>();
        const uint32_t firstTask = batchTask_[b] + kGemmFirstTaskOf(live, sub);

        // The same close-out TurboQuantPartialReducer runs, on the same shared helpers, over
        // this subcore's whole half at once rather than one row at a time.
        NormalizeHeads(panel, stats[kLaneTotal + slot * kGemmPanelRows * kFp32PerBlock], mine,
                       headSize_, stats[kLaneSums], stats[kLaneInvs]);
        // The finisher is handed one token's rows at a time, exactly as production's Reduce()
        // hands it (token, firstHead, heads). A batch is a run of consecutive tasks, so its
        // token runs are contiguous and each is one call.
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
        // One GM burst for the half: the rows are consecutive tasks, so their output slices
        // are contiguous.
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
    // The overlay: when Init() was given the attention phase's transient pool, every UB buffer
    // above comes out of this one, aliased onto those bytes at zero cost.
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

}  // namespace gemm_reduce
}  // namespace ascend_micro

#endif
