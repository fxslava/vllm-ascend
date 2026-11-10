#pragma once

// The Split-K final reduction as a Cube GEMV: the layout, flag and budget contract.
//
// Dependency-free on purpose. The device kernel and the host gate include THIS
// declaration rather than each carrying a copy, so the host compile of
// tests/cpu/test_contract_gates.cpp is evidence about the kernel's arithmetic and
// not about a transcription of it.
//
// WHAT THE STAGE COMPUTES
//
// TurboQuantPartialReducer (fused_decode/vector/turboquant_vector_service.h) merges S split
// partials for one (token, head) with a sequential online-softmax rescale. Each split s wrote
// an unnormalised accumulator o_s[0..D) plus its own statistics m_s (running max) and
// l_s (sum of exp(x - m_s)) into the workspace row
//
//     workspace[((token * numHeads + head) * numSplits + s) * kPartialStride]
//
// whose first D floats are o_s and whose tail carries m_s at lane kPartialMaxLane and l_s at
// lane kPartialSumLane. The merged head is
//
//     m*     = max_s m_s
//     w_s    = exp(m_s - m*)
//     out[d] = (sum_s w_s * o_s[d]) / (sum_s w_s * l_s)
//
// so the NUMERATOR is exactly an M=1, K=S, N=D matmul of the [1,S] weight row against the
// [S,D] partial matrix, and the denominator is a [1,S]x[S,1] dot the vector unit does in one
// 8-lane multiply plus a reduce. Note w_s = exp(m_s - m*) and NOT exp(m_s - m*) * l_s: the
// partials this operator stores are unnormalised, so l_s belongs in the denominator only.
//
// This is algebraically the same value as the Vector reducer's two-term recurrence, but it is
// NOT the same floating-point expression: the recurrence rescales the running accumulator S-1
// times and sums in split order, the GEMV sums S products in the Cube's own order inside one
// fp32 accumulator. The difference is measured, not asserted -- see the report.
//
// WHY THE CUBE IS FREE TO DO IT
//
// In src/decode/turboquant_fused_decode.cpp the reduction phase is
//
//     op.Process(tasksPerBlock);
//     if (op.NeedsReduction()) { AscendC::SyncAll<false>(); if ASCEND_IS_AIV { ... Reduce ... } }
//
// The reduction is guarded by `if ASCEND_IS_AIV`, so the Cube is idle for the whole phase
// while the busier engine does every rescale. That idleness, not a FLOP shortage, is what
// this stage spends.
//
// WHAT IT DOES NOT DO
//
// It does not overlap the reduction with the attention phase. Token t's combine needs every
// split of token t complete, which is a grid-wide dependency that the SyncAll above is
// discharging; removing it needs a per-token completion counter in L2, which is a different
// probe. The overlap this stage buys is WITHIN the reduction phase: the Cube's GEMV for task r
// runs against the vector unit's statistics merge for task r+2 and its normalise of r-2, and
// against the AIC's own MTE2 fetch of task r+1's partials.

#include <cstdint>

// The helpers below are compiled into BOTH the kernel and the host gate, which is the whole
// point of section 6.1: the two share this declaration instead of each carrying a copy. BiSheng treats
// a plain function as __host__ only, so a device caller needs the attribute, and the host gate
// must not see it.
#if defined(__CCE_AICORE__)
#define ASCEND_MICRO_REDUCE_FN __aicore__ inline constexpr
#else
#define ASCEND_MICRO_REDUCE_FN constexpr
#endif

namespace ascend_micro::gemm_reduce {

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

// A Cube C0 block is 32 bytes on every DaVinci generation, so a b32 operand carries 8 elements
// along K per block and the b32 fractal is 16 x 8 (512 bytes). This is the whole reason the
// K = S operand is viable: the decode's split count is capped at kMaxSequenceSplits = 8
// (fused_decode/common/turboquant_layout.h), so K = S <= 8 lands inside ONE C0 block and the
// GEMV needs no K tiling and no multi-block zero padding at all. At S = 8 it is exact.
constexpr std::uint32_t kC0Bytes = 32;
constexpr std::uint32_t kFloatBytes = 4;
constexpr std::uint32_t kC0Floats = kC0Bytes / kFloatBytes;  // 8
constexpr std::uint32_t kFractalRows = 16;
constexpr std::uint32_t kFractalBytes = kFractalRows * kC0Bytes;  // 512

// The operator's real geometry, not an illustrative shape.
constexpr std::uint32_t kHeadSize = 256;
constexpr std::uint32_t kMaxSplits = 8;
constexpr std::uint32_t kPartialTail = 16;  // kPartialTail = 2 * kFp32PerBlock
constexpr std::uint32_t kPartialMaxLane = 0;
constexpr std::uint32_t kPartialSumLane = 8;
constexpr std::uint32_t kPartialStride = kHeadSize + kPartialTail;  // 272 floats

// Mmad's K direction for float aligns to ceil(K/8)*8 on Ascend 950PR -- kDirectionAlign
// supports only `false` there, which is that rule. So a split count below kC0Floats leaves
// K lanes S..7 live in the hardware's eyes.
ASCEND_MICRO_REDUCE_FN std::uint32_t kAlignedSplits(std::uint32_t splits)
{
    return (splits + kC0Floats - 1) / kC0Floats * kC0Floats;
}
static_assert(kAlignedSplits(1) == 8 && kAlignedSplits(8) == 8);
// Every legal split count is one C0 block: no K tiling, one Mmad.
static_assert(kAlignedSplits(kMaxSplits) == kC0Floats);

// Those padding lanes are a CORRECTNESS hazard and not merely waste, because section 3 floods every
// GM payload with 0x7FC0BEEF (an fp32 quiet NaN) and 0 * NaN is NaN, not 0. Two independent
// guards, both exercised by the S < 8 rungs of the ladder:
//   * the weight row's lanes S..7 are zeroed, so a padding partial cannot reach the sum, and
//   * DataCopy's DN2NZ conversion zero-fills the destination when dValue * sizeof(T) is not
//     32-byte aligned, which is exactly the S < 8 case.
// Relying on either alone would be relying on an undocumented half of the pair.
constexpr bool kZeroPadWeightLanes = true;

// ---------------------------------------------------------------------------------------------
// The B operand's road into L1
// ---------------------------------------------------------------------------------------------
//
// Mmad wants B as K x N in Zn in L0B. The workspace already holds [S, D] row-major, which IS
// K x N, so the only question is the fractal layout. Two facts settle it without a transposing
// load:
//
//   * DataCopy GM->L1 with Dn2NzParams performs an inline DN->NZ conversion, which the API
//     reference states is "equivalent to the ND-to-ZN format conversion". Feeding the ND
//     [S, D] rows therefore lands the ZN image of that same K x N matrix in L1, with the S
//     axis as C0. It is an Ascend 950PR/950DT-only path and it supports float.
//   * Load2DV2 with ifTranspose = false leaves both the intra- and inter-fractal layout alone
//     (Zn2Zn), so the L0B operand is the Zn cube Mmad asked for.
//
// So the reduction's B operand takes the SAME non-transposed load the score GEMM's N x K key
// tile takes (TurboQuantCubeMm::StageRightToL0bFromNk), and none of the transposed-load
// constraints apply to it. That matters: for b32 WITH transpose, Load2DV2 requires kStep to be
// a multiple of 2, and the b32 transpose merges two consecutive 16x8 fractals into a 16x16
// square, which demands a source Zn cube with Col == 16 -- a condition a K = S <= 8 operand
// cannot meet. Avoiding the transpose avoids all of it.
//
// The resulting L1 image is [N rows][C0 floats] contiguous.
constexpr std::uint32_t kPartialL1Floats = kHeadSize * kC0Floats;              // 2048
constexpr std::uint32_t kPartialL1Bytes = kPartialL1Floats * kFloatBytes;      // 8192
static_assert(kPartialL1Bytes == kHeadSize * kC0Bytes);
static_assert(kPartialL1Bytes % kFractalBytes == 0, "Load2DV2 moves whole 512-byte fractals");
// Exactly the payload: 8 splits x 256 floats, nothing padded away at the maximum split count.
static_assert(kPartialL1Bytes == kMaxSplits * kHeadSize * kFloatBytes);

// Dn2NzParams for one (token, head): dValue is the C0 axis (the splits), nValue the N axis,
// srcDValue the source row stride, which is kPartialStride and not kHeadSize because the
// statistics tail shares the row.
// This toolkit's Dn2NzParams names the per-matrix strides srcDnMatrixStride and
// dstNzMatrixStride, not the API reference's ...CubeStride; both are 0 for a single matrix.
constexpr std::uint32_t kDn2NzDnNum = 1;
constexpr std::uint32_t kDn2NzNValue = kHeadSize;
constexpr std::uint32_t kDn2NzSrcDValue = kPartialStride;
constexpr std::uint32_t kDn2NzDstNzC0Stride = kHeadSize;
constexpr std::uint32_t kDn2NzDstNzNStride = 1;
static_assert(kDn2NzNValue <= 16384 && kDn2NzDstNzC0Stride <= 16384 && kDn2NzSrcDValue >= 1,
              "Dn2NzParams value ranges");

// The A operand. GEMV requires the 1 x K row to be 512-byte aligned with K contiguous
// elements, so the weight slot is a whole fractal even though only S floats of it are read.
constexpr std::uint32_t kWeightL1Bytes = kFractalBytes;  // 512
static_assert(kWeightL1Bytes >= kMaxSplits * kFloatBytes);
static_assert(kWeightL1Bytes % 512 == 0, "GEMV's 1xK left cube needs a 512-byte base");

// The C operand. M = 1 is GEMV's own case: Mmad enables it automatically, and Ascend 950PR is
// the one product that can switch it off with MmadParams::disableGemv. The prototype carries
// both legs because they land DIFFERENT cubes in L0C and the Fixpipe has to match:
//   * GEMV on  -- C is 1 x N split into N/16 sub-vectors, each occupying 1024 bytes of L0C of
//     which only the low 64 bytes are valid.
//   * GEMV off -- C is a conventional M-unaligned cube, M padded to kFractalRows, which is the
//     shape the production Fixpipe path in this operator already emits.
// kGemvDisabled is the default precisely because it reuses the audited Fixpipe parameters; the
// enabled leg is a measurement variant.
constexpr bool kGemvDisabledDefault = true;
constexpr std::uint32_t kProductL0cBytes = kFractalRows * kHeadSize * kFloatBytes;  // 16384
constexpr std::uint32_t kGemvL0cBytes = (kHeadSize / kFractalRows) * 1024;           // 16384
static_assert(kProductL0cBytes == kGemvL0cBytes,
              "both legs reserve the same L0C, so the slot budget does not depend on the knob");

// ---------------------------------------------------------------------------------------------
// The stage's ring and its flags
// ---------------------------------------------------------------------------------------------

// Two slots: one the Cube is multiplying while the vector unit prepares the next task's
// weights and retires the previous task's output. A third buys nothing -- the Cube's GEMV is
// one Mmad and the stage is bound by the GM fetch, not by Cube occupancy.
constexpr std::uint32_t kReduceSlots = 2;

// One AIV->AIC place and one AIC->AIV place per slot, and no more, because each doubles as the
// other direction's free credit:
//
//   weights_ready[s]  AIV -> AIC, initially 0, primed by the prologue.
//                     Posted by the AIV's MTE3 store of the weight row into L1 slot s, which
//                     the AIV reaches only AFTER its Finish of the task two earlier read UB
//                     product slot s. So this one token says both "the weights are in L1" and
//                     "product slot s is yours again".
//   product_ready[s]  AIC -> AIV, initially 0.
//                     Posted by the AIC's Fixpipe into UB product slot s. Consumed by the
//                     AIV's Finish, after which the same AIV step is what frees the weights.
//
// The reuse cycle weights_ready[s] -> product_ready[s] -> weights_ready[s] is primed by the
// prologue's unconditional sets for the first min(R, kReduceSlots) tasks, so it is marked
// before it is ever waited on, and both places drain to zero.
constexpr std::uint32_t kReducePlacesPerSlot = 2;
constexpr std::uint32_t kReduceFlagCount = kReduceSlots * kReducePlacesPerSlot;  // 4

// WHERE THOSE IDS COME FROM, which is the stage's sharpest constraint.
//
// A mode-4 flag id is four bits and the toolkit reserves 11..14, leaving 0..10. The attention
// schedule already spends 0..9 (3 slot-ready + 3 slot-free + 2 scores + context + probs) and
// kTaskJoinFlag is 10. There is NO free id. The reduction stage therefore RECYCLES ids, which
// is legal here only because AscendC::SyncAll<false>() and the attention phase's own epilogue
// stand between the two phases and every attention place is drained to zero before the first
// reduction set -- and which drags in the fence precondition: on a recycled id, every
// announcement needs its SetFlag/WaitFlag<HardEvent::x_S> fence on the pipe whose work it
// reports, a priming set excepted, and the audit that matters is tick order per id rather than
// per-id counts, because an unfenced announcement balances every count and still releases a
// wait early.
constexpr std::uint16_t kReduceFlagFirst = 0;
constexpr std::uint16_t kReduceFlagWeightsReady = kReduceFlagFirst;                      // 0..1
constexpr std::uint16_t kReduceFlagProductReady = kReduceFlagWeightsReady + kReduceSlots;  // 2..3
constexpr std::uint16_t kReduceFlagReservedFirst = 11;
constexpr std::uint16_t kReduceFlagLast = kReduceFlagProductReady + kReduceSlots - 1;
static_assert(kReduceFlagLast < kReduceFlagReservedFirst,
              "a longer schedule must not renumber a channel into a library-reserved id");
static_assert(kReduceFlagCount == kReduceFlagLast - kReduceFlagFirst + 1,
              "the id budget is an object, not a comment: every place owns exactly one id");

// ---------------------------------------------------------------------------------------------
// Budgets
// ---------------------------------------------------------------------------------------------
//
// These are what the stage ADDS. TPipe allocates for the whole kernel, so the reduction's
// buffers coexist with the attention phase's even though the phases do not.

constexpr std::uint32_t kReduceL1Bytes = kReduceSlots * (kPartialL1Bytes + kWeightL1Bytes);
constexpr std::uint32_t kReduceL0aBytes = kFractalBytes;
constexpr std::uint32_t kReduceL0bBytes = kPartialL1Bytes;
constexpr std::uint32_t kReduceL0cBytes = kProductL0cBytes;
// Per slot: the product landing the Fixpipe writes. Plus one shared statistics arena holding
// the S maxima, the S sums, the weight row and its reduction scratch.
constexpr std::uint32_t kReduceProductPayloadBytes = kHeadSize * kFloatBytes;  // 1024
// AKA3003. The payload alone is exactly 32 UB blocks, so floor(offset/32) mod 16 repeats and
// the two product slots would present the same bank to both ports: the slot the Fixpipe is
// filling and the slot the vector unit is normalising. One block of skew per slot breaks it,
// the same remedy kBankSkewBytes applies to the attention ring's ingest buffers. The skew is
// an allocation tail, never payload.
constexpr std::uint32_t kReduceBankSkewBytes = kC0Bytes;
constexpr std::uint32_t kReduceProductUbBytes =
    kReduceProductPayloadBytes + kReduceBankSkewBytes;  // 1056
constexpr std::uint32_t kReduceStatsUbBytes = 4 * kC0Bytes;  // 128
constexpr std::uint32_t kReduceUbBytes =
    kReduceSlots * kReduceProductUbBytes + kReduceStatsUbBytes;

// The attention phase's own L1 claim, restated from turboquant_fused_decode_opt.hpp's assert so
// the sum is checked here rather than inferred.
constexpr std::uint32_t kAttentionL1Bytes = 16 * 256 + 16 * 64 + 2 * 3 * 64 * 256;
static_assert(kAttentionL1Bytes + kReduceL1Bytes <= 512 * 1024, "L1 budget");
static_assert(kReduceL0aBytes <= 64 * 1024 && kReduceL0bBytes <= 64 * 1024, "L0A/L0B budget");
static_assert(kReduceL0cBytes <= 128 * 1024, "L0C budget");
// Every allocation is a whole 32-byte block, and the product slots land in different UB banks
// (AKA3003: floor(offset/32) mod 16 must differ for a paired slot).
static_assert(kReduceProductUbBytes % kC0Bytes == 0 && kReduceStatsUbBytes % kC0Bytes == 0);
static_assert(kReduceProductUbBytes / kC0Bytes % 16 != 0,
              "paired product slots must not present the same bank");
// The skew is checked against the payload it separates, which is what makes it a skew and not
// a number: it is not a universal bank-conflict proof for every repeat/stride pattern.
static_assert(kReduceProductUbBytes > kReduceProductPayloadBytes &&
                  kReduceBankSkewBytes % kC0Bytes == 0,
              "the skew is a whole block of tail past the payload");

// ---------------------------------------------------------------------------------------------
// The schedule, as arithmetic both roles compute identically
// ---------------------------------------------------------------------------------------------

ASCEND_MICRO_REDUCE_FN std::uint32_t kReducePrologue(std::uint32_t tasks)
{
    return tasks < kReduceSlots ? tasks : kReduceSlots;
}
ASCEND_MICRO_REDUCE_FN std::uint32_t kReduceSlotOf(std::uint32_t task) { return task % kReduceSlots; }
// Number of (token, head) combines this block owns, clamped like production's Reduce().
ASCEND_MICRO_REDUCE_FN std::uint32_t kReduceTaskCount(std::uint32_t total, std::uint32_t block,
                                         std::uint32_t perBlock)
{
    const std::uint64_t start = std::uint64_t{block} * perBlock;
    if (start >= total) return 0;
    const std::uint64_t end = start + perBlock;
    return static_cast<std::uint32_t>((end > total ? total : end) - start);
}
static_assert(kReduceTaskCount(16, 0, 8) == 8 && kReduceTaskCount(16, 2, 8) == 0);
static_assert(kReduceTaskCount(5, 0, 4) == 4 && kReduceTaskCount(5, 1, 4) == 1);
// An empty Split-K worker is a real rung: it must still drain, so a zero task count is legal
// and is not a reason to skip the phase's join.
static_assert(kReducePrologue(0) == 0 && kReducePrologue(1) == 1 && kReducePrologue(9) == 2);

ASCEND_MICRO_REDUCE_FN std::uint64_t kPartialRowOffset(std::uint32_t token, std::uint32_t head,
                                          std::uint32_t split, std::uint32_t numHeads,
                                          std::uint32_t numSplits)
{
    return ((std::uint64_t{token} * numHeads + head) * numSplits + split) * kPartialStride;
}

ASCEND_MICRO_REDUCE_FN bool kValidReduceGeometry(std::uint32_t headSize, std::uint32_t splits)
{
    return headSize == kHeadSize && splits >= 1 && splits <= kMaxSplits &&
           kAlignedSplits(splits) == kC0Floats;
}
static_assert(kValidReduceGeometry(kHeadSize, 1) && kValidReduceGeometry(kHeadSize, 8));
static_assert(!kValidReduceGeometry(kHeadSize, 9) && !kValidReduceGeometry(128, 4));

// =============================================================================================
// THE DENSE BATCHED FORM: M = 16 packed query rows, N = 16-channel slices
// =============================================================================================
//
// The M = 1 stage above leaves fifteen of the Cube's sixteen output rows padding, and it issues
// one MTE2 burst per (token, head), which is where it serialised. This form packs up to
// kGemmBatchRows query rows into M and sweeps N over kGemmSlices 16-channel slices of D, so the
// Cube's 16 x 16 output line is full on both axes and the GM fetch is ONE batched DN2NZ burst
// per slice instead of one per task.
//
// WHAT THE CONTRACTION ACTUALLY IS, AND WHY K IS NOT S
//
// The obvious reading of "[16, S] x [S, 16]" is a GEMM whose 16 output rows share one
// [S, 16] partials operand. That is NOT this operator's algebra. Writing q for a packed query
// row, the value wanted is
//
//     out_q[c] = sum_s w_q[s] * P_q[s, c]
//
// and the partials P_q are indexed BY q: split s of (token, head) lands at
// kPartialRowOffset(token, head, s, ...), so every (token, head) owns its own [S, D] block.
// Query heads inside one GQA group share the KV cache, not the Split-K partials -- each head
// has its own attention probabilities and therefore its own o_s. A single GEMM Out = A x B
// shares B across all M rows, so a dense [16, S] x [S, 16] with one shared B computes
// sum_s w_q[s] * P[s, c] for a P that does not exist. It is a batched GEMV with 16 independent
// (A, B) pairs, and no 2-D GEMM expresses that without either batching or padding.
//
// So the K axis carries the batch: K = kGemmBatchRows * kGemmKPerRow, A is the BLOCK-DIAGONAL
// weight matrix
//
//     A[q, q' * 8 + s] = (q == q') ? w_q[s] : 0
//
// and B stacks the 16 rows' own channel slices, B[q' * 8 + s, c] = P_q'[s, c0 + c]. Then
//
//     Out[q, c] = sum_{q', s} A[q, q'*8+s] * B[q'*8+s, c] = sum_s w_q[s] * P_q[s, c0 + c]
//
// which is the wanted value. Everything the requested shape asks for survives: M = 16 and
// N = 16 are both full, A is one cube loaded into L0A ONCE per batch and reused across all
// kGemmSlices iterations, B is a 16-fractal slice streamed into L0B with a ping-pong, and the
// sweep is kGemmSlices = D / 16 = 16 Mmads per batch. What changes is the honest accounting:
// the zeros in A cost exactly what M = 1's padding cost, so this form does not reduce MAC
// count. It reduces INSTRUCTION count on the engines that were serialising -- one DN2NZ per
// slice instead of one per task, one L0A load per batch instead of one per task -- and it
// gives the Fixpipe sixteen live rows to split between the two vector subcores, which is what
// §5's idle-AIV1 defect needed.
//
// TWO PROPERTIES OF THE BLOCK-DIAGONAL A THAT ARE GATES, NOT REMARKS
//
//  * A dead row cannot contaminate a live one. Out[q, c] reads only K block q, so garbage in
//    B's block q -- a ragged batch's padding row, whose L1 fractal still holds the 0x7FC0BEEF
//    flood of §3 -- reaches output row q and nothing else. The M = 1 form had no such
//    guarantee: there, one NaN lane in K poisoned every channel of the product.
//  * The padding lane guards of the M = 1 form still apply WITHIN a row. For splits < 8 the
//    weight row's lanes s..7 are zero and DN2NZ zero-fills the destination to the 32-byte
//    boundary, which is the S < 8 case; both halves of that pair are kept.
constexpr std::uint32_t kGemmBatchRows = 16;  // M: packed query rows per batch
constexpr std::uint32_t kGemmSliceCols = 16;  // N: channels per slice, one fractal row
constexpr std::uint32_t kGemmSlices = kHeadSize / kGemmSliceCols;  // 16 Mmads per batch
// One C0 block of K per packed row, so a row's K block IS its fractal and the block-diagonal
// structure lines up with the fractal grid exactly. kAlignedSplits(S) == kC0Floats for every
// legal split count, which is what makes this exact rather than a rounding.
constexpr std::uint32_t kGemmKPerRow = kC0Floats;  // 8
constexpr std::uint32_t kGemmKMax = kGemmBatchRows * kGemmKPerRow;  // 128
static_assert(kGemmSlices * kGemmSliceCols == kHeadSize);
static_assert(kGemmSliceCols == kFractalRows, "a slice is one fractal row of the B operand");
static_assert(kGemmKMax <= 65535, "MmadParams::k is uint16_t");
static_assert(kAlignedSplits(kMaxSplits) == kGemmKPerRow,
              "a packed row's K block is one C0 block at every legal split count");

// The A operand's cube, in L1. kGemmBatchRows fractals of [16 M rows][8 K]; fractal q carries
// row q's weights in ITS row q and zeros everywhere else. A[m, 8k + s] sits at float
// k * kFractalRows * kC0Floats + m * kC0Floats + s, so row q's own 8 weights are at
// q * (kFractalRows * kC0Floats) + q * kC0Floats -- a 32-byte-aligned base, which a vector
// store needs (an unaligned vector destination raises vec_err_instr_misalign_t0).
constexpr std::uint32_t kWeightCubeBytes = kGemmBatchRows * kFractalBytes;  // 8192
ASCEND_MICRO_REDUCE_FN std::uint32_t kWeightRowOffsetFloats(std::uint32_t row)
{
    return row * kFractalRows * kC0Floats + row * kC0Floats;
}
static_assert(kWeightRowOffsetFloats(0) == 0 && kWeightRowOffsetFloats(1) == 136);
static_assert(kWeightRowOffsetFloats(1) * kFloatBytes % kC0Bytes == 0,
              "every weight row base is a whole 32-byte block");
// The last row's block ends exactly at the cube's end: nothing to overrun into.
static_assert(kWeightRowOffsetFloats(kGemmBatchRows - 1) * kFloatBytes + kC0Bytes ==
                  kWeightCubeBytes,
              "the block-diagonal cube is exactly kGemmBatchRows fractals");

// The B operand's slice, in L1: kGemmBatchRows fractals, fractal q being row q's
// [16 channels][8 splits]. The batched DN2NZ lands them contiguously, one matrix per packed
// row, so Load2DV2 takes the whole slice as kGemmBatchRows consecutive fractals with no stride
// arithmetic -- which is the reason the ingest is grouped BY SLICE and not by row.
constexpr std::uint32_t kSliceL1Bytes = kGemmBatchRows * kFractalBytes;  // 8192
constexpr std::uint32_t kSliceFractalFloats = kFractalRows * kC0Floats;  // 128
static_assert(kSliceL1Bytes == kGemmBatchRows * kSliceFractalFloats * kFloatBytes);
// Two inner slots: DN2NZ fetches slice j + 1 while the Mmad of slice j is in flight. That
// overlap is the whole point of the change, so the inner ring carries real split credits
// rather than a paired hand-off.
constexpr std::uint32_t kSliceSlots = 2;

// Dn2NzParams for the batched slice ingest. dnNum is the batch's live row count, so a ragged
// batch fetches exactly its live rows and the padding fractals are never written -- which is
// safe only because of the block-diagonal containment above.
constexpr std::uint32_t kSliceDn2NzNValue = kGemmSliceCols;
constexpr std::uint32_t kSliceDn2NzDstNzC0Stride = kGemmSliceCols;
constexpr std::uint32_t kSliceDn2NzDstNzNStride = 1;
constexpr std::uint32_t kSliceDn2NzDstMatrixStrideFloats = kSliceFractalFloats;
static_assert(kSliceDn2NzNValue <= 16384 && kGemmBatchRows <= 65535,
              "Dn2NzParams nValue and dnNum ranges");
// The source stride between two packed rows is one (token, head)'s whole partial block, which
// is why a batch must be a run of CONSECUTIVE live tasks: dnNum walks a uniform stride.
ASCEND_MICRO_REDUCE_FN std::uint64_t kSliceSrcMatrixStrideFloats(std::uint32_t numSplits)
{
    return std::uint64_t{numSplits} * kPartialStride;
}
static_assert(kSliceSrcMatrixStrideFloats(8) == 8 * kPartialStride);

// HOW THE TWO VECTOR SUBCORES SPLIT THE BATCH, AND WHY THE HALVES ARE STATIC
//
// A dual-destination Fixpipe (dualDstCtl = 1) divides mSize rows in half and lands each half at
// the matching subcore's own UB base, so with mSize = kGemmBatchRows the split point is fixed
// at kGemmPanelRows: rows 0..7 reach subcore 0 and rows 8..15 subcore 1, each as rows 0..7 of
// its own panel. The LIVE rows are then distributed ceil/floor, subcore 0 taking rows
// 0..a-1 and subcore 1 rows 8..8+b-1, so both have real work for every live >= 2. That is the
// whole content of §B: the GEMV form's M = 1 left subcore 1 with nothing to reduce, and a
// subcore that only consumes tokens is not paced by the channel that releases it.
//
// The halves are STATIC -- fractals 0..7 belong to subcore 0 and 8..15 to subcore 1 whatever
// the live count -- and that is a correctness requirement, not a tidiness one. A fractal's
// off-diagonal rows must be zero for ITS row's output to be right, they are written once at
// Init, and they are never rewritten. If the half boundary moved with the live count, the core
// that zero-fills fractal q at Init and the core that writes q's diagonal block in a later
// batch could be different cores, and nothing orders a block's two vector subcores against
// each other at Init -- so the zero-fill could land after the weight it is supposed to precede
// and erase it. With static halves each core only ever touches its own eight fractals and the
// race cannot be constructed.
//
// The cost of static halves is that mSize is always kGemmBatchRows and k always kGemmKMax, so
// a ragged batch's dead fractals ARE read by the Mmad. That is safe only because of the
// block-diagonal containment above: output row m reads K block m alone, so a dead fractal's
// stale or flooded content reaches its own output row and no live one, and no vector
// instruction ever reads a dead row (NormalizeHeads and WriteHeads are given the live count).
// It is also why each core zeroes its dead rows' diagonal blocks per batch: a dead row's
// garbage must not depend on which batch ran before it.
constexpr std::uint32_t kGemmPanelRows = kGemmBatchRows / 2;  // 8, the Fixpipe's split point

// Live rows subcore `sub` must normalise and write, and the batch row its first one occupies.
ASCEND_MICRO_REDUCE_FN std::uint32_t kGemmLiveRowsOf(std::uint32_t live, std::uint32_t sub)
{
    const std::uint32_t first = (live + 1) / 2;  // ceil: subcore 0 takes the larger half
    return sub == 0 ? first : live - first;
}
ASCEND_MICRO_REDUCE_FN std::uint32_t kGemmFirstRowOf(std::uint32_t sub)
{
    return sub == 0 ? 0 : kGemmPanelRows;
}
// The batch-local index of subcore `sub`'s first live task, which is what turns a panel row
// back into a (token, head) and a GM output offset.
ASCEND_MICRO_REDUCE_FN std::uint32_t kGemmFirstTaskOf(std::uint32_t live, std::uint32_t sub)
{
    return sub == 0 ? 0 : kGemmLiveRowsOf(live, 0);
}
static_assert(kGemmLiveRowsOf(16, 0) == 8 && kGemmLiveRowsOf(16, 1) == 8);
static_assert(kGemmLiveRowsOf(5, 0) == 3 && kGemmLiveRowsOf(5, 1) == 2);
static_assert(kGemmLiveRowsOf(2, 0) == 1 && kGemmLiveRowsOf(2, 1) == 1);
// live = 1 is the one geometry with nothing for subcore 1 to reduce. It still waits, drains and
// announces; that is §B's "even when an idle subcore has no active query rows".
static_assert(kGemmLiveRowsOf(1, 0) == 1 && kGemmLiveRowsOf(1, 1) == 0);
// Neither half ever overflows its eight panel rows, which is what bounds the panel.
static_assert(kGemmLiveRowsOf(kGemmBatchRows, 0) == kGemmPanelRows &&
              kGemmLiveRowsOf(kGemmBatchRows, 1) == kGemmPanelRows);
static_assert(kGemmFirstTaskOf(5, 1) == 3 && kGemmFirstRowOf(1) == 8);

// Budgets the batched form adds. Two batch slots (the AIV stages batch b + 1's weights while
// the Cube multiplies b and normalises b - 1) and two inner slice slots.
constexpr std::uint32_t kBatchSlots = kReduceSlots;
constexpr std::uint32_t kGemmL1Bytes =
    kBatchSlots * kWeightCubeBytes + kSliceSlots * kSliceL1Bytes;
// ONE L0A region. The A cube is loaded once per batch and read by all kGemmSlices Mmads, so a
// second region would buy at most one slice of MTE1 prefetch; the single region costs one
// credit (l0a_free) instead of two places and halves the L0A claim.
constexpr std::uint32_t kGemmL0aBytes = kWeightCubeBytes;
constexpr std::uint32_t kGemmL0bBytes = kSliceSlots * kSliceL1Bytes;
// One [16 M][16 N] fp32 cube per inner slot. M pads to kFractalRows, so a slot is one fractal
// of C: 16 x 16 x 4 bytes.
constexpr std::uint32_t kGemmL0cSlotBytes = kFractalRows * kGemmSliceCols * kFloatBytes;  // 1024
constexpr std::uint32_t kGemmL0cBytes = kSliceSlots * kGemmL0cSlotBytes;
static_assert(kAttentionL1Bytes + kGemmL1Bytes <= 512 * 1024, "L1 budget, batched form");
static_assert(kGemmL0aBytes <= 64 * 1024 && kGemmL0bBytes <= 64 * 1024, "L0A/L0B budget");
static_assert(kGemmL0cBytes <= 128 * 1024, "L0C budget");

// The product panel each subcore receives: its own half of the batch, [8 rows][D] fp32 dense.
// Dense matters -- NormalizeHeads' Mul walks rows with dstRepStride = headSize / kFp32PerBlock
// blocks, so the rows must be exactly headSize floats apart and the AKA3003 skew goes at the
// END of the slot, never between rows.
constexpr std::uint32_t kGemmPanelPayloadBytes = kGemmPanelRows * kHeadSize * kFloatBytes;
constexpr std::uint32_t kGemmPanelUbBytes = kGemmPanelPayloadBytes + kReduceBankSkewBytes;
static_assert(kGemmPanelPayloadBytes == 8192);
static_assert(kGemmPanelUbBytes % kC0Bytes == 0);
static_assert(kGemmPanelUbBytes / kC0Bytes % 16 != 0,
              "paired panel slots must not present the same UB bank");
// The statistics arena, in 8-float DataBlocks. The header blocks are laid out in the service
// header; the tail is one 64-byte landing per (panel row, split), filled by ONE strided burst.
constexpr std::uint32_t kGemmStatsHeaderBlocks = 66;
constexpr std::uint32_t kGemmStatsUbBytes =
    (kGemmStatsHeaderBlocks + kGemmPanelRows * kMaxSplits * 2) * kC0Bytes;  // 6208
constexpr std::uint32_t kGemmOutStageBytes = kGemmPanelRows * kHeadSize * 2;  // fp16 staging
constexpr std::uint32_t kGemmZeroStageBytes = kSliceFractalFloats * kFloatBytes;  // 512

// Every UB byte TurboQuantBatchedGemmReducer allocates. The service header static_asserts its
// own InitBuffer sizes against these, so this is the claim and not an estimate of it.
constexpr std::uint32_t kGemmReducerUbBytes = kBatchSlots * kGemmPanelUbBytes +
                                              kGemmStatsUbBytes + kGemmZeroStageBytes +
                                              kGemmOutStageBytes;
static_assert(kGemmReducerUbBytes == 27264, "the batched reducer's UB claim, pinned");

// =============================================================================================
// THE INTEGRATION BUDGET: THE OVERLAY, AND WHAT IT HAD TO BE MEASURED AGAINST
// =============================================================================================
//
// The batched form's panels do not fit the operator's LEFTOVER UB -- the arithmetic below says
// by how much -- so they do not use leftover UB. They come from a TBufPool aliased onto the
// attention phase's transient pool, which SyncAll<false>() has made dead by the time the
// reduction runs. TPipe::InitBufPool(pool, len, shareBuf) sets the new pool's startAddr_ to
// shareBuf's and does NOT advance the hardware pool, so the alias costs zero bytes.
//
// The deficit arithmetic is kept rather than deleted, because it is what makes the overlay a
// decision instead of a habit: without it the reduction needs 23904 bytes the operator does
// not have, and the next person to change a size here has to be able to see that.
//
// turboquant_fused_decode_opt.hpp asserts
//     kUbEnd + kBasisUbBytes + kReducerUbBytes + 256 <= 192 * 1024
// and those three terms are restated here so the sum is checked rather than inferred; that
// header static_asserts the restatement against its own constants, so a change on either side
// is a compile error and not a drift.
constexpr std::uint32_t kUbWindowBytes = 192 * 1024;              // 196608
constexpr std::uint32_t kOperatorUbEndBytes = 142304;             // kUbEnd
constexpr std::uint32_t kOperatorBasisUbBytes = 46432;            // kBasisUbBytes
constexpr std::uint32_t kVectorReducerUbBytes = 3360;             // kReducerUbBytes
constexpr std::uint32_t kUbReadPadBytes = 256;                    // never payload
constexpr std::uint32_t kOperatorUbClaimBytes =
    kOperatorUbEndBytes + kOperatorBasisUbBytes + kVectorReducerUbBytes + kUbReadPadBytes;
static_assert(kOperatorUbClaimBytes <= kUbWindowBytes, "the operator's own UB assert, restated");
// What is left once the attention phase, the basis and the read pad are paid for.
constexpr std::uint32_t kOperatorUbHeadroomBytes = kUbWindowBytes - kOperatorUbClaimBytes;
static_assert(kOperatorUbHeadroomBytes == 4256, "headroom, pinned so a change is visible");
// The batched reducer REPLACES the vector reducer, so the net cost is the difference.
constexpr std::uint32_t kGemmReducerNetUbBytes =
    kGemmReducerUbBytes - kVectorReducerUbBytes;  // 23904
constexpr std::uint32_t kGemmReducerUbDeficitBytes =
    kGemmReducerNetUbBytes - kOperatorUbHeadroomBytes;  // 19648
static_assert(kGemmReducerNetUbBytes > kOperatorUbHeadroomBytes,
              "the deficit is what the overlay below exists for; if this ever fails the "
              "reducer fits the leftover headroom and the overlay can be dropped");
static_assert(kGemmReducerUbDeficitBytes == 19648, "the deficit, pinned");

// THE OVERLAY, which is the way out that was taken.
//
// TurboQuantVectorDecodeService::InitBuffers allocates three of the attention phase's transient
// buffers -- the tile ingest's NZ staging operand, the query staging and the per-task
// online-softmax accumulator -- from a TBufPool of exactly their summed length, and the
// reduction aliases its own pool onto it. Restated here, like the three terms above, because
// this header is dependency-free; the vector service static_asserts the restatement against its
// own kTransientPoolBytes, so a change on either side is a compile error.
//
// Why those three and not a bare TPipe::Reset(). Reset() is a real allocator rewind --
// ResetPool() sets maxAddr = 0 for every hardware pool -- but it rewinds EVERYTHING, and
// TurboQuantQueryBasis::FinishHeads runs during the reduction and reads the Pi signs in
// signBuf_, which are content and not scratch. A rewind invalidates them. The pool is the
// narrower tool: it names the bytes being reused instead of reusing all of them.
//
// Why the pool's length is exactly the sum of what it holds: kUbEnd and every buffer allocated
// after it keep their offsets, so the Cube's dual-destination Fixpipe windows at UB offsets 0
// and kScoreBytes do not move -- and neither do the reduction's own dual-destination panels,
// which land at the same address on every core because InitBuffers runs on all three roles.
constexpr std::uint32_t kGemmSharedPoolBytes = 16384 + 8192 + 8192;  // kUbSizes[15..17]
static_assert(kGemmSharedPoolBytes == 32768, "the transient pool, pinned");
static_assert(kGemmReducerUbBytes <= kGemmSharedPoolBytes,
              "the Cube GEMM reduction allocates from the attention phase's transient pool, so "
              "it must fit inside it. If this fails, either shrink the reducer (a band-wise "
              "panel halves it) or widen the pool by moving another dead attention buffer into "
              "it -- and widening it is NOT free: the pool must stay exactly as long as what it "
              "holds, or kUbEnd moves and the dual-destination Fixpipe offsets move with it.");
// Headroom the overlay leaves unspent, for the next thing that needs UB here.
constexpr std::uint32_t kGemmSharedPoolSpareBytes =
    kGemmSharedPoolBytes - kGemmReducerUbBytes;  // 5504
static_assert(kGemmSharedPoolSpareBytes == 5504, "spare in the pool, pinned");

// The two ways out that were NOT taken, kept because they are what to reach for if the reducer
// ever outgrows the pool:
//
//   1. Band-wise normalise. The panel only has to hold the channels being normalised, so a
//      64-channel band drops it from 8 KB to 2 KB a slot; NormalizeHeads is elementwise per
//      row and WriteHeads becomes a strided GM burst. One panel slot then fits in 2880 bytes
//      net, inside even the leftover headroom -- but a single slot removes the Cube/AIV panel
//      overlap, which is most of what the form buys.
//   2. A smaller kGemmBatchRows. This does NOT work as a way to fit the leftover headroom: at
//      8 rows the net cost is still 11616 bytes and at 4 rows 5472, both over 4256, and M = 4
//      gives up the dense line. It would shrink the pool requirement, though.
constexpr std::uint32_t kGemmUbBytes = kGemmReducerUbBytes;
static_assert(kGemmUbBytes + kUbReadPadBytes <= kUbWindowBytes,
              "the reducer alone fits the window; the overlay is what makes the SUM fit");

// ---------------------------------------------------------------------------------------------
// Batch formation
// ---------------------------------------------------------------------------------------------
//
// A batch is a run of CONSECUTIVE live task ids, at most kGemmBatchRows long. Consecutive is
// not a convenience: dnNum walks one uniform source stride, so a gap in the task ids -- which a
// fused token leaves, since production's Reduce() skips it -- has to end the batch. It also
// makes the batch's outputs contiguous in GM, so one WriteHeads burst per subcore covers its
// rows instead of one per row.
ASCEND_MICRO_REDUCE_FN bool kGemmBatchBreaks(std::uint32_t previousTask, std::uint32_t task,
                                             std::uint32_t rowsSoFar)
{
    return rowsSoFar >= kGemmBatchRows || task != previousTask + 1;
}
static_assert(kGemmBatchBreaks(3, 5, 1), "a task-id gap ends the batch");
static_assert(!kGemmBatchBreaks(3, 4, 1), "a consecutive task extends it");
static_assert(kGemmBatchBreaks(3, 4, kGemmBatchRows), "a full batch ends regardless");

// Batches a dense run of `tasks` consecutive combines needs, which is what the depth and wrap
// rungs are counted in: the ring is over BATCHES now, so N = kBatchSlots is depth and
// N = 2 * kBatchSlots is wrap, in units of kGemmBatchRows tasks.
ASCEND_MICRO_REDUCE_FN std::uint32_t kGemmDenseBatches(std::uint32_t tasks)
{
    return (tasks + kGemmBatchRows - 1) / kGemmBatchRows;
}
static_assert(kGemmDenseBatches(0) == 0 && kGemmDenseBatches(1) == 1 &&
              kGemmDenseBatches(16) == 1 && kGemmDenseBatches(17) == 2 &&
              kGemmDenseBatches(32) == 2 && kGemmDenseBatches(64) == 4);

// The worst case is every live task isolated by a fused neighbour, which is one batch each.
constexpr std::uint32_t kMaxGemmTasksPerBlock = 64;
constexpr std::uint32_t kMaxGemmBatchesPerBlock = kMaxGemmTasksPerBlock;

// Intra-core credit ids on the AIC. Routes are keyed by (source pipe, trigger pipe, id), so the
// same id on two different routes is two different places; what must not collide is two places
// on ONE route. The auditor additionally requires every user HardEvent id in 0..5.
//
//   slice_ready[ss]  MTE2 -> MTE1   ids 3..4   the DN2NZ burst landed in L1 slice slot ss
//   slice_free[ss]   MTE1 -> MTE2   ids 3..4   Load2DV2 has drained that slot
//   l0b_ready[ss]    MTE1 -> M      ids 3..4   the slice cube reached L0B
//   l0b_free[ss]     M -> MTE1      ids 3..4   the Mmad has read it
//   l0a_ready        MTE1 -> M      id   5     the A cube reached L0A
//   l0a_free         M -> MTE1      id   5     the batch's last Mmad has read the A cube
//   l0c_ready[ss]    M -> FIX       ids 3..4   the product cube is in L0C
//   l0c_free[ss]     FIX -> M       ids 3..4   the Fixpipe has drained it
//
// WHY THESE START AT 3 AND NOT AT 0, WHICH IS A DEVICE FINDING AND NOT A STYLE CHOICE
//
// `M_MTE1` ids 0..2 are the RUNTIME's, not the kernel's. kernel_tpipe_impl.h says so in its own
// words -- "for matmul macro, set flag M_MTE1 at the begining of operator, and also wait flag
// at the end. matmul macro only use M_MTE1 event id 0 1 currently" -- and this toolkit primes
// three of them: a trace of the first batched run shows SET_FLAG CUBE->MTE1 for ids 0, 1 and 2
// at ticks 1421, 1426 and 1430, before a single instruction of the stage has issued. A kernel
// that primes its own M_MTE1 ids 0..1 on top of that is a double SET on a one-bit flag, and the
// simulator says so: `pem_ccu.cc:2270 execute_set_flag already has same set_flag`.
//
// It is worth being precise about what that fault costs, because the first run of this form had
// it and still printed worst_fp16_ulp = 0 with every canary intact. A dropped MTE2->MTE1 credit
// does not corrupt data here; it removes an ordering guarantee, so the fetch of slice j + 2
// could overwrite the L1 slot the Mmad of slice j is still loading. It did not on that run and
// it is not meant to be left to luck.
//
// Starting at 3 also keeps every explicit id clear of the ids `FetchEventID` hands out. That
// allocator returns the first free bit of its occupancy mask, and an id set with a bare
// SetFlag is never marked occupied, so every SyncEvent<> in the drain path picks id 0 -- which
// on MTE1_M is an id this schedule would otherwise be using.
constexpr std::uint16_t kGemmRuntimeOwnedEvents = 3;  // M_MTE1 ids 0..2, primed by TPipe
constexpr std::uint16_t kGemmSliceEventFirst = kGemmRuntimeOwnedEvents;              // 3
constexpr std::uint16_t kGemmL0aEvent = kGemmSliceEventFirst + kSliceSlots;          // 5
constexpr std::uint16_t kGemmEventLast = kGemmL0aEvent;
static_assert(kGemmEventLast <= 5,
              "the trace auditor requires every user HardEvent id in 0..5");
static_assert(kGemmSliceEventFirst >= kGemmRuntimeOwnedEvents,
              "M_MTE1 ids below this are primed and drained by TPipe's matmul macro");
// M -> MTE1 is the one route carrying two distinct place families, so its ids must be disjoint.
static_assert(kGemmL0aEvent >= kGemmSliceEventFirst + kSliceSlots,
              "l0a_free and l0b_free share the M -> MTE1 route and must not share an id");

ASCEND_MICRO_REDUCE_FN bool kValidBatchedGeometry(std::uint32_t headSize, std::uint32_t splits)
{
    return kValidReduceGeometry(headSize, splits) && headSize % kGemmSliceCols == 0 &&
           headSize / kGemmSliceCols == kGemmSlices;
}
static_assert(kValidBatchedGeometry(kHeadSize, 1) && kValidBatchedGeometry(kHeadSize, 8));
static_assert(!kValidBatchedGeometry(kHeadSize, 9));

// ---------------------------------------------------------------------------------------------
// The probe ABI
// ---------------------------------------------------------------------------------------------

// Legs over one workspace. kGemm and kGemmGemvMode share a body and differ in nothing but
// MmadParams::disableGemv, so a behavioural difference between them isolates that knob; kVector
// is the shipping TurboQuantPartialReducer and is the numerical reference. kBatched and
// kBatchedGemvMode are the dense form above, and are likewise one body and one knob apart.
enum class ReduceLeg : std::uint32_t {
    kGemm = 0,
    kGemmGemvMode = 1,
    kVector = 2,
    kBatched = 3,
    kBatchedGemvMode = 4,
};

struct ReduceLaunch {
    std::uint32_t grid = 1;
    std::uint32_t tokens = 1;
    std::uint32_t heads = 4;
    std::uint32_t head_size = kHeadSize;
    std::uint32_t splits = 8;
    std::uint32_t per_block = 4;
    // Diagnostic only: a negative limit combines every token, which is what the pure
    // multi-split rungs want; a real limit drives the fused-vs-combined mix.
    std::int32_t fused_limit = -1;
    ReduceLeg leg = ReduceLeg::kGemm;
};

struct ReduceBuffers {
    void *workspace = nullptr;
    void *output = nullptr;
    // Optional: when null every (token, head) is combined.
    void *lengths = nullptr;
};

}  // namespace ascend_micro::gemm_reduce
