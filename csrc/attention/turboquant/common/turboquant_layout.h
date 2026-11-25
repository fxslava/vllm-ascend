#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_LAYOUT_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_LAYOUT_H

#include <cstdint>

namespace vllm_ascend {
namespace turboquant {

constexpr uint32_t kFp32PerBlock = 8;
constexpr uint32_t kFp32PerRepeat = 64;
constexpr uint32_t kBrcbDstLanes = kFp32PerBlock * kFp32PerBlock;
constexpr uint32_t kGatherSrcBase = 0;
constexpr uint32_t kMaxRepeatTimes = 255;
constexpr uint32_t kVectorSubcoresPerBlock = 2;

constexpr uint32_t kPackFactor = 2;
constexpr uint32_t kCodecLevels = 16;

constexpr uint32_t kStripTokens = 16;

constexpr uint32_t kMaxSequenceSplits = 8;
constexpr uint32_t kFusedContextLimit = 4096;
constexpr uint32_t kPartialTail = 2 * kFp32PerBlock;
constexpr uint32_t kPartialMaxLane = 0;
constexpr uint32_t kPartialSumLane = kFp32PerBlock;
constexpr uint32_t kMinBurstBytes = 128;

constexpr uint32_t kTelemetryMaxLane = kPartialMaxLane;
constexpr uint32_t kTelemetryPosLane = 1;
constexpr uint32_t kTelemetryPageLane = 2;
constexpr float kTelemetryInvalid = -1.0f;
struct TelemetryPredicateLayout {
    static constexpr uint32_t kPosition = 392;
    static constexpr uint32_t kPage = 400;
    static constexpr uint32_t kArgmax = 536;
    static constexpr uint32_t kWidth = 8;
    static constexpr uint32_t kArenaFloats = kArgmax + kWidth;
};
static_assert(TelemetryPredicateLayout::kPosition + TelemetryPredicateLayout::kWidth <=
    TelemetryPredicateLayout::kPage);
static_assert(TelemetryPredicateLayout::kPage + TelemetryPredicateLayout::kWidth <=
    TelemetryPredicateLayout::kArgmax);
static_assert(kTelemetryPosLane > kPartialMaxLane && kTelemetryPageLane < kPartialSumLane,
              "the probe lanes must sit in the first tail block's dead padding");

#ifndef ASCEND_TQ_DECODE_TELEMETRY
#define ASCEND_TQ_DECODE_TELEMETRY 1
#endif

constexpr uint32_t kCubeTileRows = 64;
constexpr uint32_t kCubeUnpackRows = 8;
constexpr uint32_t kOperandC0 = 32;
constexpr uint32_t kCubeTileM = 16;
constexpr uint32_t kCubeSlots = 2;

constexpr uint32_t kCubePipelineSlots = 3;

constexpr uint32_t kCubeScoreWindows = 2;

constexpr uint32_t kBankSkewBytes = 32;

constexpr uint64_t NzTiledPackedByte(uint64_t slot, uint64_t kvHead, uint64_t column,
                                     uint64_t numKvHeads, uint64_t packedBytes)
{
    const uint64_t tileRow = slot % kCubeTileRows;
    return (slot - tileRow) * numKvHeads * packedBytes + kvHead * kCubeTileRows * packedBytes +
           ((column / kOperandC0) * kCubeTileRows + tileRow) * kOperandC0 + column % kOperandC0;
}

constexpr uint64_t NzTiledChannelByte(uint64_t slot, uint64_t kvHead, uint64_t channel,
                                      uint64_t numKvHeads, uint64_t headSize)
{
    const uint64_t packedBytes = headSize / kPackFactor;
    return NzTiledPackedByte(slot, kvHead, channel % packedBytes, numKvHeads, packedBytes);
}

constexpr uint32_t NzTiledChannelNibble(uint64_t channel, uint64_t headSize)
{
    return channel < headSize / kPackFactor ? 0u : 1u;
}

constexpr uint32_t NzScaleSlotFloats(uint32_t numKvHeads)
{
    return ((2u * numKvHeads + kFp32PerBlock - 1u) / kFp32PerBlock) * kFp32PerBlock;
}

constexpr uint64_t NzTiledScaleIndex(uint64_t slot, uint64_t kvHead, uint64_t kv,
                                     uint64_t numKvHeads)
{
    return slot * NzScaleSlotFloats(static_cast<uint32_t>(numKvHeads)) + kv * numKvHeads + kvHead;
}

constexpr uint16_t kFlagOperandsReady = 0;
constexpr uint16_t kFlagProductReady = 1;
constexpr uint16_t kFlagSlotReady = 0;
constexpr uint16_t kFlagSlotFree = 2;
constexpr uint16_t kFlagScoresReady = 4;
constexpr uint16_t kFlagContextReady = 5;
constexpr uint16_t kFlagProbsReady = 6;

constexpr uint16_t kOptFlagSlotReady = 0;
constexpr uint16_t kOptFlagSlotFree = kOptFlagSlotReady + static_cast<uint16_t>(kCubePipelineSlots);
constexpr uint16_t kOptFlagScoresReady =
    kOptFlagSlotFree + static_cast<uint16_t>(kCubePipelineSlots);
constexpr uint16_t kOptFlagContextReady =
    kOptFlagScoresReady + static_cast<uint16_t>(kCubeScoreWindows);
constexpr uint16_t kOptFlagProbsReady = kOptFlagContextReady + 1;

constexpr uint16_t kOptFlagReservedFirst = 11;

constexpr uint32_t kRotateQTile = 16;
constexpr uint32_t kRotateQMaxChunkElements = 4096;
constexpr uint32_t kRotateQMinCubeTokens = 2;
constexpr uint32_t kRotateQH16Elements = kRotateQTile * kRotateQTile;
constexpr uint16_t kRotateQHalfOne = 0x3C00;
constexpr uint16_t kRotateQHalfMinusOne = 0xBC00;

enum RotateQVariant : uint32_t {
    kHiLo = 0x1u,
    kDualDst = 0x2u,
};

enum class RotateQPrecision : uint32_t {
    kSinglePassRint = 0,
    kHiLoResidual = 1,
};

constexpr uint32_t kQueryBasisCubeVectors = 16;
constexpr uint32_t kQueryBasisMinCubeVectors = 16;

enum class TurboQuantOutputStage : uint32_t {
    kRotatedBasis = 0,
    kUnrotated = 1,
    kGated = 2,
};

constexpr float kGateLogitFloor = -80.0f;

}
}

#endif
