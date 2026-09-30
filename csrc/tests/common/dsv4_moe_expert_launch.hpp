/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Host-side view of the DSV4 expert kernel: the launcher the test-owned
// kernel TU defines, plus the problem-generation helpers the simulator and
// device tiers share. The numerics live in reference/dsv4_moe_expert_cpu.h.

#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "dsv4_moe_expert_cpu.h"

namespace vllm_ascend {

// Defined in common/dsv4_moe_expert_kernels.cpp, inside the ascendc_library.
void dsv4_moe_expert_impl(void *stream, uint32_t blockDim, void *x, void *w1, void *w2, void *w3, void *w1Scale,
                          void *w2Scale, void *w3Scale, void *gateOut, void *upOut, void *activatedOut,
                          void *downOut, void *workspace, void *tiling);

namespace test {
namespace dsv4 {

// The kernel's tiling struct, in the field order of
// op_kernel/dsv4_moe_expert_tiling_data.h. The tests build it by hand rather
// than driving the host tiling path.
struct TilingBuffer {
  int64_t hidden_size;
  int64_t inter_size;
  int64_t block_size;
};

// One generated problem, laid out exactly as the kernel's argument order.
struct Problem {
  int64_t hidden = 0;
  int64_t inter = 0;
  std::vector<uint16_t> x;   // [hidden] bf16 bits
  std::vector<uint8_t> w1;   // [inter, hidden/2]
  std::vector<uint8_t> w2;   // [hidden, inter/2]
  std::vector<uint8_t> w3;   // [inter, hidden/2]
  std::vector<uint8_t> w1_scale;
  std::vector<uint8_t> w2_scale;
  std::vector<uint8_t> w3_scale;

  ExpertInputs View() const {
    return ExpertInputs{x.data(),        w1.data(),       w2.data(),      w3.data(),
                        w1_scale.data(), w2_scale.data(), w3_scale.data()};
  }

  TilingBuffer Tiling() const { return TilingBuffer{hidden, inter, kFp4Block}; }
};

// `scale_span` bounds the E8M0 exponents to 127 +/- span. The specials (0x00
// subnormal, 0xFF NaN) are deliberately excluded: they are covered exhaustively
// by the host tier, and including them here would let a genuine arithmetic
// regression hide behind a NaN comparison.
inline Problem MakeProblem(int64_t hidden, int64_t inter, uint32_t seed, int scale_span = 6) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_int_distribution<int> scale_dist(127 - scale_span, 127 + scale_span);
  std::normal_distribution<float> x_dist(0.0f, 1.0f);

  Problem p;
  p.hidden = hidden;
  p.inter = inter;
  p.x.resize(static_cast<size_t>(hidden));
  for (auto& v : p.x) {
    v = FloatToBf16Bits(x_dist(rng));
  }

  auto fill = [&](std::vector<uint8_t>& v, size_t n, bool is_scale) {
    v.resize(n);
    for (auto& b : v) {
      b = static_cast<uint8_t>(is_scale ? scale_dist(rng) : byte_dist(rng));
    }
  };
  fill(p.w1, static_cast<size_t>(inter * hidden / kFp4PerByte), false);
  fill(p.w3, static_cast<size_t>(inter * hidden / kFp4PerByte), false);
  fill(p.w2, static_cast<size_t>(hidden * inter / kFp4PerByte), false);
  fill(p.w1_scale, static_cast<size_t>(inter * hidden / kFp4Block), true);
  fill(p.w3_scale, static_cast<size_t>(inter * hidden / kFp4Block), true);
  fill(p.w2_scale, static_cast<size_t>(hidden * inter / kFp4Block), true);
  return p;
}

// Drive the gate activations hard negative, so every element takes the
// exp() overflow path the camodel reports as `check_fp_status instr input
// data inf`. Built by giving w1 the most negative E2M1 code (0xF -> -6) and a
// large positive block scale, with x biased positive.
inline Problem MakeSaturatingProblem(int64_t hidden, int64_t inter, uint32_t seed) {
  Problem p = MakeProblem(hidden, inter, seed);
  for (auto& v : p.x) {
    v = FloatToBf16Bits(2.0f);
  }
  for (auto& b : p.w1) {
    b = 0xFF;  // both nibbles = code 15 = -6.0
  }
  for (auto& b : p.w1_scale) {
    b = 127 + 4;  // 2^4, so gate ~= -6 * 16 * 2 * hidden, far past the exp() range
  }
  return p;
}

}  // namespace dsv4
}  // namespace test
}  // namespace vllm_ascend
