// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

#include "csrc/attention/turboquant/kernels/turboquant_wave_protocol.hpp"
#include "csrc/attention/turboquant/op_kernel/common/turboquant_layout.h"
#include "csrc/attention/turboquant/op_kernel/common/turboquant_strip_layout.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

// Replay the cross-core handshakes across several waves. An announcement may
// only fill an empty latch, and every block must leave all latches empty.
template <class Policy>
void CheckChannels() {
  using namespace turboquant_wave;
  constexpr uint32_t kOwners = 2;
  constexpr uint32_t kWaves = 4;
  std::array<bool, kOwners * kIntraNum> pending{};
  const auto event = [&pending](uint16_t id, bool set) {
    Check(LegalIntraId(id), "a channel overlaps reserved flag ids");
    Check(pending.at(id) != set, "an unmatched cross-core notification");
    pending.at(id) = set;
  };
  for (uint32_t wave = 0; wave < kWaves; ++wave) {
    for (uint32_t owner = 0; owner < kOwners; ++owner) {
      for (uint32_t slot = 0; slot < kReshapeLandDepth; ++slot) event(Policy::PanelFreePrime(owner, slot), true);
      for (uint32_t chunk = owner; chunk < kReshapeChunks; chunk += kOwners) {
        event(Policy::PanelFreeWait(owner, chunk), false);
        event(Policy::ProductReady(owner, chunk), true);
        event(Policy::ProductReady(owner, chunk), false);
        event(Policy::PanelFreeSet(owner, chunk), true);
      }
      for (uint32_t slot = 0; slot < kReshapeLandDepth; ++slot) event(Policy::EpilogueDrain(owner, slot), false);
      event(Policy::PageDrained(owner), true);
      event(Policy::PageDrained(owner), false);
    }
    for (bool value : pending) Check(!value, "a notification leaks into the next wave");
  }
}

void CheckWaves() {
  using namespace turboquant_wave;
  CheckChannels<ReshapeRecycled>();
  CheckChannels<ReshapeSingleShot>();
  Check(kReshapeDefaultReuse, "the recycled policy must remain the default");
  for (uint32_t strips : {0u, 1u, 32u, 33u, 48u, 64u, 99u, 128u}) {
    const uint32_t tokens = strips * kReshapeTokensPerStrip;
    Check(WholeStrips(tokens, kReshapeTokensPerStrip), "whole strips rejected");
    Check(TotalStrips(tokens, kReshapeTokensPerStrip) == strips, "incorrect grid size");
    Check(BlockOutOfRange(strips, tokens, kReshapeTokensPerStrip), "out-of-range block is active");
    if (strips != 0) Check(!BlockOutOfRange(strips - 1, tokens, kReshapeTokensPerStrip), "last block is inactive");
  }
  Check(!WholeStrips(17, kReshapeTokensPerStrip), "a trailing token would be lost");
  Check(!WholeStrips(1000, kReshapeTokensPerStrip), "a partial trailing strip would be lost");
  Check(!WholeStrips(16, 0), "zero strip width is valid");
  Check(BlockOutOfRange(0, 16, 0), "zero strip width does not guard every block");
}

void CheckDecodeLayout() {
  using namespace vllm_ascend::turboquant;
  constexpr uint32_t kHeads = 4;
  constexpr uint32_t kPages = 3;
  constexpr uint32_t kHeadSize = 256;
  constexpr uint32_t kNibbleValues = 16;
  for (uint32_t blockSize : {16u, 32u, 64u, 128u}) {
    const uint64_t planeBytes = kPages * kHeads * blockSize * kHeadSize / kPackFactor;
    std::array<std::vector<uint8_t>, 2> cache{std::vector<uint8_t>(planeBytes), std::vector<uint8_t>(planeBytes)};
    std::vector<float> scales(2 * kPages * kHeads * blockSize);
    // Independent writer oracle: fill each strip as StoreChunk does, with
    // page/head/row/channel-specific values that expose swaps and offsets.
    for (uint32_t page = 0; page < kPages; ++page)
      for (uint32_t head = 0; head < kHeads; ++head)
        for (uint32_t row = 0; row < blockSize; ++row)
          for (uint32_t channel = 0; channel < kHeadSize; ++channel) {
            const uint32_t code = (page + 3 * head + row + 5 * channel) % kNibbleValues;
            const uint64_t base = (page * kHeads + head) * blockSize * kHeadSize / kPackFactor;
            const uint32_t keyNibble = row * kHeadSize + channel;
            const uint32_t valueNibble = (row / kCacheStripRows) * kCacheStripRows * kHeadSize +
                                         channel * kCacheStripRows + row % kCacheStripRows;
            for (uint32_t plane = 0; plane < 2; ++plane) {
              const uint32_t nibble = plane == 0 ? keyNibble : valueNibble;
              cache[plane][base + nibble / kPackFactor] |= code << ((nibble % kPackFactor) * 4);
              scales[((plane * kPages + page) * kHeads + head) * blockSize + row] =
                  static_cast<float>(1 + plane + 2 * page + 3 * head + row);
            }
          }
    // Decode ingress: read each head's 16-token strip, expand signed
    // nibbles, and apply the exact planar scale. This covers every INT4
    // code, multiple physical pages, both planes and nonzero strip offsets.
    for (uint32_t plane = 0; plane < 2; ++plane)
      for (uint32_t page = 0; page < kPages; ++page)
        for (uint32_t head = 0; head < kHeads; ++head)
          for (uint32_t row = 0; row < blockSize; ++row)
            for (uint32_t channel = 0; channel < kHeadSize; ++channel) {
              const uint32_t strip = row / kCacheStripRows * kCacheStripRows;
              const uint32_t nibble = plane == 0 ? (row - strip) * kHeadSize + channel
                                                 : StripValueGatherByte(row - strip, channel) / sizeof(float);
              const uint64_t start = StripCacheHeadByte(page, head, strip, kHeads, kHeadSize, blockSize);
              const uint8_t byte = cache[plane].at(start + nibble / kPackFactor);
              const int code = (byte >> (nibble % kPackFactor * 4)) & (kNibbleValues - 1);
              const int value = code < 8 ? code : code - kNibbleValues;
              const int expectedCode = (page + 3 * head + row + 5 * channel) % kNibbleValues;
              const int expectedValue = expectedCode < 8 ? expectedCode : expectedCode - kNibbleValues;
              const float scale = scales.at(StripCacheScaleFloat(plane, page, head, row, kPages, kHeads, blockSize));
              Check(value == expectedValue, "decode permutes cache coordinates or changes signed nibbles");
              Check(scale == static_cast<float>(1 + plane + 2 * page + 3 * head + row), "decode mixes scale planes");
              Check(StripCacheNibble(row, channel, kHeadSize, plane != 0) == strip * kHeadSize + nibble,
                    "strip address disagrees with the writer layout");
            }
  }
  Check(kOptFlagProbsReady < kOptFlagReservedFirst, "decode flags overlap reserved ids");
  Check(kCubePipelineSlots >= 3 && kCubeScoreWindows >= 2, "decode pipeline cannot overlap");
}

}  // namespace

int main(int argc, char** argv) {
  Check(argc == 2, "expected wave or decode test selection");
  const std::string test = argv[1];
  if (test == "wave")
    CheckWaves();
  else if (test == "decode")
    CheckDecodeLayout();
  else
    throw std::runtime_error("unknown test selection");
}
