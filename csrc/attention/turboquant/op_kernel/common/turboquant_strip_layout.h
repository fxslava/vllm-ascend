// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
#pragma once

#include "../../kernels/turboquant_wave_protocol.hpp"

namespace vllm_ascend::turboquant {

constexpr uint32_t kCacheStripRows = turboquant_wave::kReshapeTokensPerStrip;

TQ_WAVE_FN constexpr uint64_t StripCacheHeadByte(uint32_t page, uint32_t head, uint32_t row, uint32_t numHeads,
                                                 uint32_t headSize, uint32_t blockSize) {
  return ((static_cast<uint64_t>(page) * numHeads + head) * blockSize + row) * headSize / 2;
}

TQ_WAVE_FN constexpr uint64_t StripCacheScaleFloat(uint32_t plane, uint32_t page, uint32_t head, uint32_t row,
                                                   uint32_t numPages, uint32_t numHeads, uint32_t blockSize) {
  return ((static_cast<uint64_t>(plane) * numPages + page) * numHeads + head) * blockSize + row;
}

// Nibble within one head's page; V is channel-major within each 16-token strip.
TQ_WAVE_FN constexpr uint32_t StripCacheNibble(uint32_t row, uint32_t channel, uint32_t headSize, bool value) {
  return value
             ? (row / kCacheStripRows) * headSize * kCacheStripRows + channel * kCacheStripRows + row % kCacheStripRows
             : row * headSize + channel;
}

TQ_WAVE_FN constexpr uint32_t StripValueGatherByte(uint32_t token, uint32_t channel) {
  return (channel * kCacheStripRows + token) * sizeof(float);
}

}  // namespace vllm_ascend::turboquant
