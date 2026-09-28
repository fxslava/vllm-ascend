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

// The E4M3Fn (OCP FP8) bit conversions the kv4fp8 host models share, beside the
// fp16 converters in fp16.hpp. Header-only, no ACL dependency, so the host tier
// uses them too.

#include <cmath>
#include <cstdint>

namespace vllm_ascend {
namespace test {
namespace fp8 {

constexpr int kE4m3fnExponentBias = 7;
constexpr int kE4m3fnMantissaBits = 3;
constexpr uint32_t kE4m3fnExponentMask = 0x0F;
constexpr uint32_t kE4m3fnMantissaMask = 0x07;
constexpr uint32_t kE4m3fnSignBit = 0x80;

inline uint8_t E4m3fnBits(float value) {
  const uint32_t sign = std::signbit(value) ? kE4m3fnSignBit : 0u;
  int exponent = 0;
  const float fraction = std::frexp(std::fabs(value), &exponent);
  const uint32_t biased = static_cast<uint32_t>(exponent - 1 + kE4m3fnExponentBias);
  const uint32_t mantissa =
      static_cast<uint32_t>(std::lround((2.0f * fraction - 1.0f) * static_cast<float>(1 << kE4m3fnMantissaBits)));
  return static_cast<uint8_t>(sign | (biased << kE4m3fnMantissaBits) | mantissa);
}

inline float E4m3fnValue(uint8_t bits) {
  const int biased = static_cast<int>((static_cast<uint32_t>(bits) >> kE4m3fnMantissaBits) & kE4m3fnExponentMask);
  const float mantissa =
      1.0f + static_cast<float>(bits & kE4m3fnMantissaMask) / static_cast<float>(1 << kE4m3fnMantissaBits);
  const float magnitude = std::ldexp(mantissa, biased - kE4m3fnExponentBias);
  return (bits & kE4m3fnSignBit) != 0u ? -magnitude : magnitude;
}

}  // namespace fp8
}  // namespace test
}  // namespace vllm_ascend
