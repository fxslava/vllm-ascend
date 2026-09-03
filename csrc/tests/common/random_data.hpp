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

// Deterministic test data.
//
// std::mt19937 is specified bit-exactly by the standard, but the distribution
// classes are not: libstdc++ and libc++ produce different sequences from the
// same engine. Box-Muller and the uniform mapping are therefore written out
// here so a failure reproduces identically on every host and in CI.

#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "fp16.hpp"

namespace vllm_ascend {
namespace test {

class DeterministicRandom {
 public:
  explicit DeterministicRandom(uint32_t seed) : engine_(seed) {}

  float Uniform(float low, float high) {
    // 24 bits of mantissa is the most a float can hold exactly.
    const uint32_t bits = engine_() >> 8;
    const float unit = static_cast<float>(bits) / static_cast<float>(1u << 24);
    return low + unit * (high - low);
  }

  float Normal(float mean, float stddev) {
    if (has_spare_) {
      has_spare_ = false;
      return mean + stddev * spare_;
    }
    // Box-Muller. u1 is clamped away from zero so the log stays finite.
    float u1 = Uniform(0.0f, 1.0f);
    if (u1 < 1e-7f) {
      u1 = 1e-7f;
    }
    const float u2 = Uniform(0.0f, 1.0f);
    const float magnitude = std::sqrt(-2.0f * std::log(u1));
    const float angle = 6.2831853071795864769f * u2;
    spare_ = magnitude * std::sin(angle);
    has_spare_ = true;
    return mean + stddev * magnitude * std::cos(angle);
  }

  int32_t IntInRange(int32_t low, int32_t high_inclusive) {
    const uint32_t span = static_cast<uint32_t>(high_inclusive - low) + 1u;
    return low + static_cast<int32_t>(engine_() % span);
  }

  // Values already rounded to fp16, so the reference and the device see exactly
  // the same inputs and the only difference measured is the arithmetic.
  std::vector<float> NormalHalfExact(size_t count, float mean, float stddev) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
      values[i] = HalfBitsToFloat(FloatToHalfBits(Normal(mean, stddev)));
    }
    return values;
  }

  std::vector<float> UniformHalfExact(size_t count, float low, float high) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
      values[i] = HalfBitsToFloat(FloatToHalfBits(Uniform(low, high)));
    }
    return values;
  }

  // Fisher-Yates shuffle over 0..count-1, used for block tables and slot maps
  // so paged accesses are genuinely scattered rather than sequential.
  std::vector<int32_t> Permutation(int32_t count) {
    std::vector<int32_t> values(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
      values[static_cast<size_t>(i)] = i;
    }
    for (int32_t i = count - 1; i > 0; --i) {
      const int32_t j = IntInRange(0, i);
      const int32_t tmp = values[static_cast<size_t>(i)];
      values[static_cast<size_t>(i)] = values[static_cast<size_t>(j)];
      values[static_cast<size_t>(j)] = tmp;
    }
    return values;
  }

 private:
  std::mt19937 engine_;
  float spare_ = 0.0f;
  bool has_spare_ = false;
};

}  // namespace test
}  // namespace vllm_ascend
