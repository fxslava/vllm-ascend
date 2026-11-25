#pragma once
#include <cstddef>
#include <cstdint>

namespace e8_decode {
constexpr std::uint32_t kRows = 64, kHalfRows = 32, kChannels = 256;
constexpr std::uint32_t kHeads = 6, kLocalHeads = 3, kPaddedHeads = 16;
constexpr std::uint32_t kRing = 2, kMaxTiles = 256, kPacketBytes = 4352;
constexpr std::uint32_t kPad = 256;
constexpr std::uint32_t Round32(std::uint32_t n) { return (n + 31u) & ~31u; }
constexpr std::uint32_t kLabelAllocation = 2048 + kPad;
constexpr std::uint32_t kScaleAllocation = 128 + kPad;
constexpr std::uint32_t kPackedAllocation = 32768 + kPad;
constexpr std::uint32_t kKeyGroupStrideBytes = 1056;
constexpr std::uint32_t kKeyPackedAllocation = 32*kKeyGroupStrideBytes + kPad;
constexpr std::uint32_t kKeyPlaneAllocation = kLabelAllocation + kScaleAllocation + kKeyPackedAllocation;
constexpr std::uint32_t kPlaneAllocation = kLabelAllocation + kScaleAllocation + kPackedAllocation;
constexpr std::uint32_t kSlotAllocation = kKeyPlaneAllocation + kPlaneAllocation + 32;
constexpr std::uint32_t kUbScores = 2 * kSlotAllocation;
constexpr std::uint32_t kUbContext = kUbScores + 768 + kPad;
constexpr std::uint32_t kUbAccumulator = kUbContext + 3072 + kPad;
constexpr std::uint32_t kUbProbabilities = kUbAccumulator + 3072 + kPad;
constexpr std::uint32_t kUbMaximum = kUbProbabilities + 768 + kPad;
constexpr std::uint32_t kUbSum = kUbMaximum + 96 + kPad;
constexpr std::uint32_t kUbOutput = kUbSum + 96 + kPad;
constexpr std::uint32_t kUbOffsets = kUbOutput + 1536 + kPad;
constexpr std::uint32_t kUbStorageBytes = kUbOffsets + kMaxTiles * 8 + kPad;
static_assert(kUbStorageBytes == 158464);
static_assert(kUbStorageBytes <= 192u * 1024u);
static_assert(256u*1024u - kUbStorageBytes - 8u*1024u >= 32u*1024u);
constexpr std::uint32_t kL1Plane = 65536 + 32;
constexpr std::uint32_t kL1Slot = 2 * kL1Plane;
constexpr std::uint32_t kL1Query = 2 * kL1Slot;
constexpr std::uint32_t kL1Probabilities = kL1Query + 16384 + 32;
constexpr std::uint32_t kL1Bytes = kL1Probabilities + 4096 + 32;
static_assert(kL1Bytes == 282816 && kL1Bytes <= 512u*1024u);
constexpr std::uint32_t PlaneBase(std::uint32_t slot, std::uint32_t plane) {
    return slot*kSlotAllocation + plane*kKeyPlaneAllocation;
}
constexpr std::uint32_t Labels(std::uint32_t s, std::uint32_t p) { return PlaneBase(s,p); }
constexpr std::uint32_t Scales(std::uint32_t s, std::uint32_t p) { return Labels(s,p)+kLabelAllocation; }
constexpr std::uint32_t Packed(std::uint32_t s, std::uint32_t p) { return Scales(s,p)+kScaleAllocation; }
constexpr std::uint32_t KeyPackedIndex(std::uint32_t row, std::uint32_t channel) {
    return (channel/8)*(kKeyGroupStrideBytes/4) + row*8 + channel%8;
}
constexpr std::uint32_t ValuePackedIndex(std::uint32_t row, std::uint32_t channel) {
    return (row/8)*256*8 + channel*8 + row%8;
}

// Query is padded to16x256 FP32, rows6..15 zero. Offsets are uint64 byte
// offsets into separate K/V arrays; each names a complete4352-byte packet.
// Initial entry supports one logical KV task, six query heads, even4..256tiles.
struct Launch {
    std::uint32_t tiles = 6;
    std::uint32_t last_rows = 32;
    float attention_scale = 0.0625f;
};
struct Buffers {
    void *key_packets, *value_packets, *packet_offsets, *query_padded;
    void *output, *key_witness, *value_witness, *score_witness;
};
}
extern "C" void launch_e8_fused_decode_fp32(void* stream,
    const e8_decode::Launch* launch, const e8_decode::Buffers* buffers);
