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

// Host-side IEEE-754 binary16 <-> binary32 conversion.
//
// The bare-metal test suite deliberately avoids <torch/types.h>, ATen and the
// CANN half type, so the conversion is implemented here with plain bit
// manipulation. Rounding is round-to-nearest-even, which is what both the
// DaVinci vector unit and PyTorch use, so a value that round-trips through this
// header has the exact same bit pattern the NPU would have produced.

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace vllm_ascend {
namespace test {

inline float HalfBitsToFloat(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1Fu;
  uint32_t mantissa = h & 0x3FFu;
  uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;  // +/- zero
    } else {
      // Subnormal binary16: renormalise into the binary32 normal range.
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3FFu;
      bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1Fu) {
    bits = sign | 0x7F800000u | (mantissa << 13);  // Inf / NaN
  } else {
    bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  }

  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline uint16_t FloatToHalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t raw_exponent = (bits >> 23) & 0xFFu;
  uint32_t mantissa = bits & 0x7FFFFFu;

  if (raw_exponent == 0xFFu) {
    // Inf stays Inf; NaN keeps a non-zero payload so it stays NaN.
    return static_cast<uint16_t>(sign | 0x7C00u | (mantissa != 0 ? 0x200u : 0u));
  }

  const int32_t exponent = static_cast<int32_t>(raw_exponent) - 127 + 15;
  if (exponent >= 0x1F) {
    return static_cast<uint16_t>(sign | 0x7C00u);  // overflow saturates to Inf
  }

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);  // underflows past the subnormal range
    }
    mantissa |= 0x800000u;  // restore the implicit leading one
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t result = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const uint32_t half_way = 1u << (shift - 1);
    if (remainder > half_way || (remainder == half_way && (result & 1u) != 0)) {
      ++result;
    }
    return static_cast<uint16_t>(sign | result);
  }

  // A carry out of the mantissa here correctly increments the exponent field,
  // and an exponent that reaches 0x1F lands on the Inf pattern.
  uint32_t result = (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
  const uint32_t remainder = mantissa & 0x1FFFu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1u) != 0)) {
    ++result;
  }
  return static_cast<uint16_t>(sign | result);
}

// Storage-only half type. It has no arithmetic operators on purpose: every
// computation in the reference implementations happens in float or double, and
// the conversion points are meant to be visible in the code.
struct Half {
  uint16_t bits = 0;

  Half() = default;
  explicit Half(float value) : bits(FloatToHalfBits(value)) {}
  float ToFloat() const { return HalfBitsToFloat(bits); }
};

static_assert(sizeof(Half) == 2, "Half must be exactly 2 bytes for device transfers");

inline std::vector<Half> FloatToHalf(const std::vector<float>& src) {
  std::vector<Half> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = Half(src[i]);
  }
  return out;
}

inline std::vector<float> HalfToFloat(const std::vector<Half>& src) {
  std::vector<float> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = src[i].ToFloat();
  }
  return out;
}

// Rounds every element to the nearest representable binary16 value while
// keeping float storage. Reference implementations use this to model the
// precision of an fp16 input without giving up float arithmetic.
inline std::vector<float> QuantizeToHalf(const std::vector<float>& src) {
  std::vector<float> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = HalfBitsToFloat(FloatToHalfBits(src[i]));
  }
  return out;
}

}  // namespace test
}  // namespace vllm_ascend
