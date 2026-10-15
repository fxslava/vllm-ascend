/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#pragma once

// The E2M1 (OCP FP4) rounding the kv3fp4 host models share, beside the fp16 and
// E4M3Fn converters next to it. Header-only, no ACL dependency, so the host tier
// uses them too.
//
// Why two steps: an fp4 operand reaches the Cube through two casts, fp32 to
// bfloat16 and bfloat16 to e2m1, both round-to-nearest-even
// (TurboQuantCodecMx::CastToOperand). A reference that wants the operand the
// hardware actually multiplies has to take the same two. Double rounding only
// changes the answer for a value within a bfloat16 step of an e2m1 midpoint, so
// it almost never shows -- but it costs one line to be exact rather than almost.

#include <cmath>
#include <cstdint>
#include <cstring>

namespace vllm_ascend {
namespace test {
namespace fp4 {

// The eight e2m1 magnitudes in bit order: the two subnormals, then one pair per exponent.
// The index IS the 3-bit magnitude field, so its low bit is the mantissa and a tie resolves
// to the even index.
constexpr int kE2m1Magnitudes = 8;
constexpr float kE2m1Grid[kE2m1Magnitudes] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
constexpr float kE2m1Max = 6.0f;

// Round-to-nearest-even onto bfloat16, which is an fp32 keeping only its top 16 bits.
inline float Bf16RoundNearest(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  if (((bits >> 23) & 0xFFu) != 0xFFu) {
    bits += 0x7FFFu + ((bits >> 16) & 1u);
  }
  bits &= 0xFFFF0000u;
  float rounded = 0.0f;
  std::memcpy(&rounded, &bits, sizeof(rounded));
  return rounded;
}

// Round-to-nearest-even onto the e2m1 grid, saturating at +-6. Returns the level rather than
// the bit pattern: nothing on the host multiplies fp4 bits, and every caller wants the value.
inline float E2m1RoundNearest(float value) {
  const float magnitude = std::fabs(value);
  int index = kE2m1Magnitudes - 1;
  for (int i = 0; i + 1 < kE2m1Magnitudes; ++i) {
    const float midpoint = 0.5f * (kE2m1Grid[i] + kE2m1Grid[i + 1]);
    if (magnitude < midpoint || (magnitude == midpoint && (i % 2) == 0)) {
      index = i;
      break;
    }
  }
  return std::signbit(value) ? -kE2m1Grid[index] : kE2m1Grid[index];
}

// The pair of casts TurboQuantCodecMx::CastToOperand issues for an fp4 mode, in its order.
inline float E2m1Operand(float value) { return E2m1RoundNearest(Bf16RoundNearest(value)); }

}  // namespace fp4
}  // namespace test
}  // namespace vllm_ascend
