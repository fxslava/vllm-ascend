// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
#pragma once

// NOT PART OF THE AUDITED KERNEL TREE. Every other file in this directory is a
// verbatim mirror of kernels/ops/turboquant/vllm_integration_layer in the
// ascend-micro-harness; this one is the repo's own host-side restatement of the
// strip cache's addressing, kept because it is the only place that arithmetic
// is stated for a host reader and because tests/ut/ops/test_turboquant_kernel_contracts.cpp
// is a device-free gate that checks the restatement against the layout headers.
//
// The audited writers carry the same arithmetic inline:
// op_kernel/turboquant_reshape_batch.cpp's "Layout contracts" block and
// op_kernel/turboquant_reshape_scatter.cpp's nibble-index comment. If one of
// those moves, this file is wrong and that test is what says so.

#include "turboquant_wave_protocol.hpp"

namespace vllm_ascend::turboquant {

// The strip height, R: `kTokens` in op_kernel/turboquant_reshape_batch.cpp,
// where it is a compile-time constant of the four-stage schedule. The audited
// wave protocol does not publish it (the legacy one did, as
// turboquant_wave::kReshapeTokensPerStrip), so it is stated here.
constexpr uint32_t kCacheStripRows = 16u;

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
