#pragma once

#include <cstdint>

#if defined(__CCE_AICORE__)
#define ASCEND_MICRO_REDUCE_FN __aicore__ inline constexpr
#else
#define ASCEND_MICRO_REDUCE_FN constexpr
#endif

namespace ascend_micro::gemm_reduce {

constexpr std::uint32_t kC0Bytes = 32;
constexpr std::uint32_t kFloatBytes = 4;
constexpr std::uint32_t kC0Floats = kC0Bytes / kFloatBytes;
constexpr std::uint32_t kFractalRows = 16;
constexpr std::uint32_t kFractalBytes = kFractalRows * kC0Bytes;

constexpr std::uint32_t kHeadSize = 256;
constexpr std::uint32_t kMaxSplits = 8;
constexpr std::uint32_t kPartialTail = 16;
constexpr std::uint32_t kPartialMaxLane = 0;
constexpr std::uint32_t kPartialSumLane = 8;
constexpr std::uint32_t kPartialStride = kHeadSize + kPartialTail;

ASCEND_MICRO_REDUCE_FN std::uint32_t kAlignedSplits(std::uint32_t splits)
{
    return (splits + kC0Floats - 1) / kC0Floats * kC0Floats;
}
static_assert(kAlignedSplits(1) == 8 && kAlignedSplits(8) == 8);

static_assert(kAlignedSplits(kMaxSplits) == kC0Floats);

constexpr bool kZeroPadWeightLanes = true;

constexpr std::uint32_t kPartialL1Floats = kHeadSize * kC0Floats;
constexpr std::uint32_t kPartialL1Bytes = kPartialL1Floats * kFloatBytes;
static_assert(kPartialL1Bytes == kHeadSize * kC0Bytes);
static_assert(kPartialL1Bytes % kFractalBytes == 0, "Load2DV2 moves whole 512-byte fractals");

static_assert(kPartialL1Bytes == kMaxSplits * kHeadSize * kFloatBytes);

constexpr std::uint32_t kDn2NzDnNum = 1;
constexpr std::uint32_t kDn2NzNValue = kHeadSize;
constexpr std::uint32_t kDn2NzSrcDValue = kPartialStride;
constexpr std::uint32_t kDn2NzDstNzC0Stride = kHeadSize;
constexpr std::uint32_t kDn2NzDstNzNStride = 1;
static_assert(kDn2NzNValue <= 16384 && kDn2NzDstNzC0Stride <= 16384 && kDn2NzSrcDValue >= 1,
              "Dn2NzParams value ranges");

constexpr std::uint32_t kWeightL1Bytes = kFractalBytes;
static_assert(kWeightL1Bytes >= kMaxSplits * kFloatBytes);
static_assert(kWeightL1Bytes % 512 == 0, "GEMV's 1xK left cube needs a 512-byte base");

constexpr bool kGemvDisabledDefault = true;
constexpr std::uint32_t kProductL0cBytes = kFractalRows * kHeadSize * kFloatBytes;
constexpr std::uint32_t kGemvL0cBytes = (kHeadSize / kFractalRows) * 1024;
static_assert(kProductL0cBytes == kGemvL0cBytes,
              "both legs reserve the same L0C, so the slot budget does not depend on the knob");

constexpr std::uint32_t kReduceSlots = 2;

constexpr std::uint32_t kReducePlacesPerSlot = 2;
constexpr std::uint32_t kReduceFlagCount = kReduceSlots * kReducePlacesPerSlot;

constexpr std::uint16_t kReduceFlagFirst = 0;
constexpr std::uint16_t kReduceFlagWeightsReady = kReduceFlagFirst;
constexpr std::uint16_t kReduceFlagProductReady = kReduceFlagWeightsReady + kReduceSlots;
constexpr std::uint16_t kReduceFlagReservedFirst = 11;
constexpr std::uint16_t kReduceFlagLast = kReduceFlagProductReady + kReduceSlots - 1;
static_assert(kReduceFlagLast < kReduceFlagReservedFirst,
              "a longer schedule must not renumber a channel into a library-reserved id");
static_assert(kReduceFlagCount == kReduceFlagLast - kReduceFlagFirst + 1,
              "the id budget is an object, not a comment: every place owns exactly one id");

constexpr std::uint32_t kReduceL1Bytes = kReduceSlots * (kPartialL1Bytes + kWeightL1Bytes);
constexpr std::uint32_t kReduceL0aBytes = kFractalBytes;
constexpr std::uint32_t kReduceL0bBytes = kPartialL1Bytes;
constexpr std::uint32_t kReduceL0cBytes = kProductL0cBytes;

constexpr std::uint32_t kReduceProductPayloadBytes = kHeadSize * kFloatBytes;

constexpr std::uint32_t kReduceBankSkewBytes = kC0Bytes;
constexpr std::uint32_t kReduceProductUbBytes =
    kReduceProductPayloadBytes + kReduceBankSkewBytes;
constexpr std::uint32_t kReduceStatsUbBytes = 4 * kC0Bytes;
constexpr std::uint32_t kReduceUbBytes =
    kReduceSlots * kReduceProductUbBytes + kReduceStatsUbBytes;

constexpr std::uint32_t kAttentionL1Bytes = 16 * 256 + 16 * 64 + 2 * 3 * 64 * 256;
static_assert(kAttentionL1Bytes + kReduceL1Bytes <= 512 * 1024, "L1 budget");
static_assert(kReduceL0aBytes <= 64 * 1024 && kReduceL0bBytes <= 64 * 1024, "L0A/L0B budget");
static_assert(kReduceL0cBytes <= 128 * 1024, "L0C budget");

static_assert(kReduceProductUbBytes % kC0Bytes == 0 && kReduceStatsUbBytes % kC0Bytes == 0);
static_assert(kReduceProductUbBytes / kC0Bytes % 16 != 0,
              "paired product slots must not present the same bank");

static_assert(kReduceProductUbBytes > kReduceProductPayloadBytes &&
                  kReduceBankSkewBytes % kC0Bytes == 0,
              "the skew is a whole block of tail past the payload");

ASCEND_MICRO_REDUCE_FN std::uint32_t kReducePrologue(std::uint32_t tasks)
{
    return tasks < kReduceSlots ? tasks : kReduceSlots;
}
ASCEND_MICRO_REDUCE_FN std::uint32_t kReduceSlotOf(std::uint32_t task) { return task % kReduceSlots; }

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

constexpr std::uint32_t kGemmBatchRows = 16;
constexpr std::uint32_t kGemmSliceCols = 16;
constexpr std::uint32_t kGemmSlices = kHeadSize / kGemmSliceCols;

constexpr std::uint32_t kGemmKPerRow = kC0Floats;
constexpr std::uint32_t kGemmKMax = kGemmBatchRows * kGemmKPerRow;
static_assert(kGemmSlices * kGemmSliceCols == kHeadSize);
static_assert(kGemmSliceCols == kFractalRows, "a slice is one fractal row of the B operand");
static_assert(kGemmKMax <= 65535, "MmadParams::k is uint16_t");
static_assert(kAlignedSplits(kMaxSplits) == kGemmKPerRow,
              "a packed row's K block is one C0 block at every legal split count");

constexpr std::uint32_t kWeightCubeBytes = kGemmBatchRows * kFractalBytes;
ASCEND_MICRO_REDUCE_FN std::uint32_t kWeightRowOffsetFloats(std::uint32_t row)
{
    return row * kFractalRows * kC0Floats + row * kC0Floats;
}
static_assert(kWeightRowOffsetFloats(0) == 0 && kWeightRowOffsetFloats(1) == 136);
static_assert(kWeightRowOffsetFloats(1) * kFloatBytes % kC0Bytes == 0,
              "every weight row base is a whole 32-byte block");

static_assert(kWeightRowOffsetFloats(kGemmBatchRows - 1) * kFloatBytes + kC0Bytes ==
                  kWeightCubeBytes,
              "the block-diagonal cube is exactly kGemmBatchRows fractals");

constexpr std::uint32_t kSliceL1Bytes = kGemmBatchRows * kFractalBytes;
constexpr std::uint32_t kSliceFractalFloats = kFractalRows * kC0Floats;
static_assert(kSliceL1Bytes == kGemmBatchRows * kSliceFractalFloats * kFloatBytes);

constexpr std::uint32_t kSliceSlots = 2;

constexpr std::uint32_t kSliceDn2NzNValue = kGemmSliceCols;
constexpr std::uint32_t kSliceDn2NzDstNzC0Stride = kGemmSliceCols;
constexpr std::uint32_t kSliceDn2NzDstNzNStride = 1;
constexpr std::uint32_t kSliceDn2NzDstMatrixStrideFloats = kSliceFractalFloats;
static_assert(kSliceDn2NzNValue <= 16384 && kGemmBatchRows <= 65535,
              "Dn2NzParams nValue and dnNum ranges");

ASCEND_MICRO_REDUCE_FN std::uint64_t kSliceSrcMatrixStrideFloats(std::uint32_t numSplits)
{
    return std::uint64_t{numSplits} * kPartialStride;
}
static_assert(kSliceSrcMatrixStrideFloats(8) == 8 * kPartialStride);

constexpr std::uint32_t kGemmPanelRows = kGemmBatchRows / 2;

ASCEND_MICRO_REDUCE_FN std::uint32_t kGemmLiveRowsOf(std::uint32_t live, std::uint32_t sub)
{
    const std::uint32_t first = (live + 1) / 2;
    return sub == 0 ? first : live - first;
}
ASCEND_MICRO_REDUCE_FN std::uint32_t kGemmFirstRowOf(std::uint32_t sub)
{
    return sub == 0 ? 0 : kGemmPanelRows;
}

ASCEND_MICRO_REDUCE_FN std::uint32_t kGemmFirstTaskOf(std::uint32_t live, std::uint32_t sub)
{
    return sub == 0 ? 0 : kGemmLiveRowsOf(live, 0);
}
static_assert(kGemmLiveRowsOf(16, 0) == 8 && kGemmLiveRowsOf(16, 1) == 8);
static_assert(kGemmLiveRowsOf(5, 0) == 3 && kGemmLiveRowsOf(5, 1) == 2);
static_assert(kGemmLiveRowsOf(2, 0) == 1 && kGemmLiveRowsOf(2, 1) == 1);

static_assert(kGemmLiveRowsOf(1, 0) == 1 && kGemmLiveRowsOf(1, 1) == 0);

static_assert(kGemmLiveRowsOf(kGemmBatchRows, 0) == kGemmPanelRows &&
              kGemmLiveRowsOf(kGemmBatchRows, 1) == kGemmPanelRows);
static_assert(kGemmFirstTaskOf(5, 1) == 3 && kGemmFirstRowOf(1) == 8);

ASCEND_MICRO_REDUCE_FN std::uint32_t kGemmPaddingFirstRowOf(std::uint32_t live, std::uint32_t sub)
{
    return kGemmFirstRowOf(sub) + kGemmLiveRowsOf(live, sub);
}
ASCEND_MICRO_REDUCE_FN std::uint32_t kGemmPaddingRowsOf(std::uint32_t live, std::uint32_t sub)
{
    return kGemmPanelRows - kGemmLiveRowsOf(live, sub);
}
static_assert(kGemmPaddingFirstRowOf(2, 0) == 1 && kGemmPaddingRowsOf(2, 0) == 7);
static_assert(kGemmPaddingFirstRowOf(2, 1) == 9 && kGemmPaddingRowsOf(2, 1) == 7);
static_assert(kGemmPaddingRowsOf(kGemmBatchRows, 0) == 0 &&
              kGemmPaddingRowsOf(kGemmBatchRows, 1) == 0);

constexpr std::uint32_t kBatchSlots = kReduceSlots;
constexpr std::uint32_t kGemmL1Bytes =
    kBatchSlots * kWeightCubeBytes + kSliceSlots * kSliceL1Bytes;

constexpr std::uint32_t kGemmL0aBytes = kWeightCubeBytes;
constexpr std::uint32_t kGemmL0bBytes = kSliceSlots * kSliceL1Bytes;

constexpr std::uint32_t kGemmL0cSlotBytes = kFractalRows * kGemmSliceCols * kFloatBytes;
constexpr std::uint32_t kGemmL0cBytes = kSliceSlots * kGemmL0cSlotBytes;
static_assert(kAttentionL1Bytes + kGemmL1Bytes <= 512 * 1024, "L1 budget, batched form");
static_assert(kGemmL0aBytes <= 64 * 1024 && kGemmL0bBytes <= 64 * 1024, "L0A/L0B budget");
static_assert(kGemmL0cBytes <= 128 * 1024, "L0C budget");

constexpr std::uint32_t kGemmPanelPayloadBytes = kGemmPanelRows * kHeadSize * kFloatBytes;
constexpr std::uint32_t kGemmPanelUbBytes = kGemmPanelPayloadBytes + kReduceBankSkewBytes;
static_assert(kGemmPanelPayloadBytes == 8192);
static_assert(kGemmPanelUbBytes % kC0Bytes == 0);
static_assert(kGemmPanelUbBytes / kC0Bytes % 16 != 0,
              "paired panel slots must not present the same UB bank");

constexpr std::uint32_t kGemmStatsHeaderBlocks = 66;
constexpr std::uint32_t kGemmStatsUbBytes =
    (kGemmStatsHeaderBlocks + kGemmPanelRows * kMaxSplits * 2) * kC0Bytes;
constexpr std::uint32_t kGemmOutStageBytes = kGemmPanelRows * kHeadSize * 2;
constexpr std::uint32_t kGemmZeroStageBytes = kSliceFractalFloats * kFloatBytes;

constexpr std::uint32_t kGemmReducerUbBytes = kBatchSlots * kGemmPanelUbBytes +
                                              kGemmStatsUbBytes + kGemmZeroStageBytes +
                                              kGemmOutStageBytes;
static_assert(kGemmReducerUbBytes == 27264, "the batched reducer's UB claim, pinned");

constexpr std::uint32_t kUbWindowBytes = 192 * 1024;
constexpr std::uint32_t kOperatorUbEndBytes = 142336;
constexpr std::uint32_t kOperatorBasisUbBytes = 46432;
constexpr std::uint32_t kVectorReducerUbBytes = 3360;
constexpr std::uint32_t kUbReadPadBytes = 256;
constexpr std::uint32_t kOperatorUbClaimBytes =
    kOperatorUbEndBytes + kOperatorBasisUbBytes + kVectorReducerUbBytes + kUbReadPadBytes;
static_assert(kOperatorUbClaimBytes <= kUbWindowBytes, "the operator's own UB assert, restated");

constexpr std::uint32_t kOperatorUbHeadroomBytes = kUbWindowBytes - kOperatorUbClaimBytes;
static_assert(kOperatorUbHeadroomBytes == 4224, "headroom, pinned so a change is visible");

constexpr std::uint32_t kGemmReducerNetUbBytes =
    kGemmReducerUbBytes - kVectorReducerUbBytes;
constexpr std::uint32_t kGemmReducerUbDeficitBytes =
    kGemmReducerNetUbBytes - kOperatorUbHeadroomBytes;
static_assert(kGemmReducerNetUbBytes > kOperatorUbHeadroomBytes,
              "the deficit is what the overlay below exists for; if this ever fails the "
              "reducer fits the leftover headroom and the overlay can be dropped");
static_assert(kGemmReducerUbDeficitBytes == 19680, "the deficit, pinned");

constexpr std::uint32_t kGemmSharedPoolBytes = 16384 + 8192 + 8192;
static_assert(kGemmSharedPoolBytes == 32768, "the transient pool, pinned");
static_assert(kGemmReducerUbBytes <= kGemmSharedPoolBytes,
              "the Cube GEMM reduction allocates from the attention phase's transient pool, so "
              "it must fit inside it. If this fails, either shrink the reducer (a band-wise "
              "panel halves it) or widen the pool by moving another dead attention buffer into "
              "it -- and widening it is NOT free: the pool must stay exactly as long as what it "
              "holds, or kUbEnd moves and the dual-destination Fixpipe offsets move with it.");

constexpr std::uint32_t kGemmSharedPoolSpareBytes =
    kGemmSharedPoolBytes - kGemmReducerUbBytes;
static_assert(kGemmSharedPoolSpareBytes == 5504, "spare in the pool, pinned");

constexpr std::uint32_t kGemmUbBytes = kGemmReducerUbBytes;
static_assert(kGemmUbBytes + kUbReadPadBytes <= kUbWindowBytes,
              "the reducer alone fits the window; the overlay is what makes the SUM fit");

ASCEND_MICRO_REDUCE_FN bool kGemmBatchBreaks(std::uint32_t previousTask, std::uint32_t task,
                                             std::uint32_t rowsSoFar)
{
    return rowsSoFar >= kGemmBatchRows || task != previousTask + 1;
}
static_assert(kGemmBatchBreaks(3, 5, 1), "a task-id gap ends the batch");
static_assert(!kGemmBatchBreaks(3, 4, 1), "a consecutive task extends it");
static_assert(kGemmBatchBreaks(3, 4, kGemmBatchRows), "a full batch ends regardless");

ASCEND_MICRO_REDUCE_FN std::uint32_t kGemmDenseBatches(std::uint32_t tasks)
{
    return (tasks + kGemmBatchRows - 1) / kGemmBatchRows;
}
static_assert(kGemmDenseBatches(0) == 0 && kGemmDenseBatches(1) == 1 &&
              kGemmDenseBatches(16) == 1 && kGemmDenseBatches(17) == 2 &&
              kGemmDenseBatches(32) == 2 && kGemmDenseBatches(64) == 4);

constexpr std::uint32_t kMaxGemmTasksPerBlock = 64;
constexpr std::uint32_t kMaxGemmBatchesPerBlock = kMaxGemmTasksPerBlock;

constexpr std::uint16_t kGemmRuntimeOwnedEvents = 3;
constexpr std::uint16_t kGemmSliceEventFirst = kGemmRuntimeOwnedEvents;
constexpr std::uint16_t kGemmL0aEvent = kGemmSliceEventFirst + kSliceSlots;
constexpr std::uint16_t kGemmEventLast = kGemmL0aEvent;
static_assert(kGemmEventLast <= 5,
              "the trace auditor requires every user HardEvent id in 0..5");
static_assert(kGemmSliceEventFirst >= kGemmRuntimeOwnedEvents,
              "M_MTE1 ids below this are primed and drained by TPipe's matmul macro");

static_assert(kGemmL0aEvent >= kGemmSliceEventFirst + kSliceSlots,
              "l0a_free and l0b_free share the M -> MTE1 route and must not share an id");

ASCEND_MICRO_REDUCE_FN bool kValidBatchedGeometry(std::uint32_t headSize, std::uint32_t splits)
{
    return kValidReduceGeometry(headSize, splits) && headSize % kGemmSliceCols == 0 &&
           headSize / kGemmSliceCols == kGemmSlices;
}
static_assert(kValidBatchedGeometry(kHeadSize, 1) && kValidBatchedGeometry(kHeadSize, 8));
static_assert(!kValidBatchedGeometry(kHeadSize, 9));

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

    std::int32_t fused_limit = -1;
    ReduceLeg leg = ReduceLeg::kGemm;
};

struct ReduceBuffers {
    void *workspace = nullptr;
    void *output = nullptr;

    void *lengths = nullptr;
};

}
