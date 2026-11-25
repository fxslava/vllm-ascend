#pragma once

#include <cstddef>
#include <cstdint>

namespace turboquant_splitk {
constexpr std::uint32_t kChannels = 256;
constexpr std::uint32_t kStateLanes = 8;
constexpr std::uint32_t kMaxRows = 8;
constexpr std::uint32_t kMaxSplits = 16;
constexpr std::uint32_t kRecordFloats = 272;

struct RowState {
    float m_global[kStateLanes];
    float l_global[kStateLanes];
    float acc_global[kChannels];
};

constexpr std::uint32_t kMaxStateRows = 16;
constexpr std::uint32_t kPartitionFloats = kMaxStateRows * kRecordFloats;

constexpr std::uint32_t kMaxCoResidentBlocks = 32;

#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr bool GridFitsOneWave(std::uint32_t requests, std::uint32_t heads,
                               std::uint32_t splits)
{
    return splits <= 1u ||
           static_cast<std::uint64_t>(requests) * heads * splits <= kMaxCoResidentBlocks;
}

constexpr std::size_t WorkspaceFloats(std::uint32_t requests, std::uint32_t heads,
                                      std::uint32_t splits)
{
    return static_cast<std::size_t>(requests) * heads * splits * kPartitionFloats;
}

#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr std::uint64_t PartitionOffset(std::uint32_t head_item, std::uint32_t split,
                                        std::uint32_t splits)
{
    return (static_cast<std::uint64_t>(head_item) * splits + split) * kPartitionFloats;
}

struct alignas(32) WorkerMetrics {

    std::uint32_t exported[8];

    std::uint32_t partition[8];

    std::uint32_t reduced[8];

    std::uint32_t pages[8];

    std::uint32_t tail[8];
    std::uint32_t finish_hi[8], finish_lo[8];

    std::uint32_t export_hi[8], export_lo[8];
};

constexpr std::uint32_t kMetricBlocks = 9;
constexpr std::uint32_t kMetricLanes = kMetricBlocks * 8;

constexpr std::uint32_t kNoPartition = 0xFFFFFFFFu;

struct PageRange { std::uint32_t begin, count; };
#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr PageRange Partition(std::uint32_t pages, std::uint32_t splits,
                              std::uint32_t split)
{
    const std::uint32_t width = (pages + splits - 1u) / splits;
    const std::uint32_t begin = split * width;
    const std::uint32_t remaining = begin < pages ? pages - begin : 0u;
    return {begin, remaining < width ? remaining : width};
}

static_assert(sizeof(RowState) == kRecordFloats * sizeof(float), "row ABI");
static_assert(kRecordFloats * sizeof(float) % 32 == 0,
              "a row record must be a whole number of 32-byte bursts");
static_assert(kPartitionFloats * sizeof(float) % 32 == 0,
              "a partition base must be 32-byte aligned");
static_assert(kMaxRows <= kMaxStateRows, "the shared stride must hold both paths");
static_assert(sizeof(WorkerMetrics) == kMetricLanes * sizeof(std::uint32_t),
              "metric ABI: nine replicated DataBlocks, no padding");
static_assert(sizeof(WorkerMetrics) % 32 == 0,
              "the census must be a whole number of replicated DataBlocks");
}
