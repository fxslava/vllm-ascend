// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dsv4_test_oracle.hpp"

namespace vllm_ascend::test::dsv4 {
constexpr int64_t kProductionHidden = 7168;
constexpr int64_t kProductionInter = 2048;
// The kernel's tiling struct, field for field with
// op_kernel/dsv4_moe_expert_tiling_data.h. The tests build it by hand rather
// than driving the host tiling path, so it has to match exactly:
// GET_TILING_DATA_WITH_STRUCT copies sizeof(struct) bytes out of the buffer,
// and a short one is an out-of-bounds read that segfaults the CPU interpreter.
struct TilingBuffer {
  int64_t hidden_size;
  int64_t inter_size;
  int64_t block_size;
  float swiglu_limit;
  float reserved;
};

// One generated problem, laid out exactly as the kernel's argument order.
struct Problem {
  int64_t hidden = 0;
  int64_t inter = 0;
  std::vector<uint16_t> x;  // [hidden] bf16 bits
  std::vector<uint8_t> w1;  // [inter, hidden/2]
  std::vector<uint8_t> w2;  // [hidden, inter/2]
  std::vector<uint8_t> w3;  // [inter, hidden/2]
  std::vector<uint8_t> w1_scale;
  std::vector<uint8_t> w2_scale;
  std::vector<uint8_t> w3_scale;

  ExpertInputs View() const {
    return ExpertInputs{x.data(), w1.data(), w2.data(), w3.data(), w1_scale.data(), w2_scale.data(), w3_scale.data()};
  }

  TilingBuffer Tiling() const { return TilingBuffer{hidden, inter, kFp4Block, kSwigluLimit, 0.0f}; }
};

// Integer-only generator: identical data across host standard libraries.
inline Problem MakeDeviceProblem(int64_t hidden, int64_t inter, uint32_t seed, uint8_t min_scale = 120,
                                 uint8_t max_scale = 124) {
  if (!GeometryIsAccepted(hidden, inter)) throw std::invalid_argument("unsupported DSV4 geometry");
  if (min_scale > max_scale) throw std::invalid_argument("invalid E8M0 scale range");
  uint32_t state = seed | 1u;
  auto next = [&state]() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  };
  Problem p;
  p.hidden = hidden;
  p.inter = inter;
  p.x.resize(static_cast<size_t>(hidden));
  for (auto& v : p.x) v = FloatToBf16Bits((static_cast<int>(next() % 513) - 256) / 128.0f);
  for (auto* w : {&p.w1, &p.w2, &p.w3}) {
    w->resize(static_cast<size_t>(hidden * inter / kFp4PerByte));
    for (auto& b : *w) b = static_cast<uint8_t>(next());
    for (size_t j = 0; j < w->size() && j < 256; ++j) (*w)[j] = static_cast<uint8_t>(j);
  }
  for (auto* s : {&p.w1_scale, &p.w2_scale, &p.w3_scale}) {
    s->resize(static_cast<size_t>(hidden * inter / kFp4Block));
    for (auto& b : *s)
      b = static_cast<uint8_t>(min_scale + next() % (static_cast<uint32_t>(max_scale) - min_scale + 1));
  }
  return p;
}

inline Problem MakeProblem(int64_t hidden, int64_t inter, uint32_t seed, int scale_span = 6) {
  if (scale_span < 0 || scale_span > 126) throw std::invalid_argument("invalid scale span");
  return MakeDeviceProblem(hidden, inter, seed, static_cast<uint8_t>(127 - scale_span),
                           static_cast<uint8_t>(127 + scale_span));
}

inline Problem MakeSaturatingProblem(int64_t hidden, int64_t inter, uint32_t seed, bool positive = false) {
  auto p = MakeDeviceProblem(hidden, inter, seed);
  std::fill(p.x.begin(), p.x.end(), FloatToBf16Bits(2.0f));
  std::fill(p.w1.begin(), p.w1.end(), positive ? 0x77 : 0xFF);
  std::fill(p.w1_scale.begin(), p.w1_scale.end(), 127);
  return p;
}

// Sweep all exponent bytes, including subnormal 0 and NaN 255, in the gate
// projection. Other legs stay healthy so codec classification is attributable.
inline Problem MakeScaleSweepProblem(int64_t hidden, int64_t inter, uint32_t seed) {
  auto p = MakeDeviceProblem(hidden, inter, seed);
  std::fill(p.x.begin(), p.x.end(), FloatToBf16Bits(1.0f));
  std::fill(p.w1.begin(), p.w1.end(), 0);
  std::fill(p.w1_scale.begin(), p.w1_scale.end(), 127);
  for (int64_t row = 0; row < inter; ++row) {
    p.w1_scale[static_cast<size_t>(row * hidden / kFp4Block)] = static_cast<uint8_t>(row % 256);
    p.w1[static_cast<size_t>(row * hidden / kFp4PerByte)] = 0x07;
  }
  return p;
}

inline DeviceOutputs Golden(const Problem& p) {
  auto g = ReferenceExpert(p.View(), p.hidden, p.inter);
  return {std::move(g.gate_out), std::move(g.up_out), std::move(g.activated), std::move(g.down_out)};
}
}  // namespace vllm_ascend::test::dsv4
