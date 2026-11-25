#pragma once
#include <cstdint>
namespace e8_packet_abi {

constexpr std::uint32_t kRows = 64;
constexpr std::uint32_t kChannels = 256;
constexpr std::uint32_t kLabelsPerRow = 32;
constexpr std::uint32_t kLabelRowBytes = 64;
constexpr std::uint32_t kScaleOffset = 4096;
constexpr std::uint32_t kScaleBytes = 256;
constexpr std::uint32_t kPacketBytes = 4352;
static_assert(kRows * kLabelsPerRow * 2u == kScaleOffset);
static_assert(kScaleOffset + kRows * sizeof(float) == kPacketBytes);
}
