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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../attention/turboquant/op_kernel/common/turboquant_mode.h"
#include "random_data.hpp"

namespace vllm_ascend {
namespace test {
namespace turboquant_host {

using MirroredTraits = vllm_ascend::turboquant::TurboQuantModeTraits<vllm_ascend::turboquant::TurboQuantMode::KV4_FP8>;

constexpr int32_t kMirroredLevels = MirroredTraits::kLevels;
constexpr float kMirroredGain = MirroredTraits::kGain;
constexpr float kMirroredAffineBias = MirroredTraits::kAffineBias;
constexpr int32_t kMirroredNibbleSignShift = kMirroredLevels / 2;
constexpr uint32_t kMirroredNibbleMask = 0x0F;
constexpr int kMirroredNibbleBits = 4;
constexpr int64_t kMirroredScaleLanes = 8;

constexpr int kFp8E4m3fnExponentBias = 7;
constexpr int kFp8E4m3fnMantissaBits = 3;
constexpr uint32_t kFp8E4m3fnExponentMask = 0x0F;
constexpr uint32_t kFp8E4m3fnMantissaMask = 0x07;
constexpr uint32_t kFp8E4m3fnSignBit = 0x80;

inline uint8_t Fp8E4m3fnBits(float value) {
  const uint32_t sign = std::signbit(value) ? kFp8E4m3fnSignBit : 0u;
  int exponent = 0;
  const float fraction = std::frexp(std::fabs(value), &exponent);
  const uint32_t biased = static_cast<uint32_t>(exponent - 1 + kFp8E4m3fnExponentBias);
  const uint32_t mantissa = static_cast<uint32_t>(
      std::lround((2.0f * fraction - 1.0f) * static_cast<float>(1 << kFp8E4m3fnMantissaBits)));
  return static_cast<uint8_t>(sign | (biased << kFp8E4m3fnMantissaBits) | mantissa);
}

inline float Fp8E4m3fnValue(uint8_t bits) {
  const int biased =
      static_cast<int>((static_cast<uint32_t>(bits) >> kFp8E4m3fnMantissaBits) & kFp8E4m3fnExponentMask);
  const float mantissa =
      1.0f + static_cast<float>(bits & kFp8E4m3fnMantissaMask) / static_cast<float>(1 << kFp8E4m3fnMantissaBits);
  const float magnitude = std::ldexp(mantissa, biased - kFp8E4m3fnExponentBias);
  return (bits & kFp8E4m3fnSignBit) != 0u ? -magnitude : magnitude;
}

inline float MirroredLevel(int32_t index) { return static_cast<float>(index) - kMirroredAffineBias; }

inline int64_t MirroredScaleSlotFloats(int64_t num_kv_heads) {
  return (2 * num_kv_heads + kMirroredScaleLanes - 1) / kMirroredScaleLanes * kMirroredScaleLanes;
}

struct MirroredKvCache {
  std::vector<int8_t> key_packed;
  std::vector<int8_t> value_packed;
  std::vector<int8_t> key_operands;
  std::vector<int8_t> value_operands;
  std::vector<float> scales;
  std::vector<float> key;
  std::vector<float> value;
};

inline MirroredKvCache BuildMirroredKvCache(DeterministicRandom& rng, int64_t context_len, int64_t num_blocks,
                                            int64_t block_size, int64_t num_kv_heads, int64_t head_size,
                                            const std::vector<int32_t>& block_table) {
  const size_t half = static_cast<size_t>(head_size / 2);
  const size_t head = static_cast<size_t>(head_size);
  const size_t packed_bytes =
      static_cast<size_t>(head_size * MirroredTraits::kBytesPerGroup / MirroredTraits::kElemsPerGroup);
  const size_t slot_floats = static_cast<size_t>(MirroredScaleSlotFloats(num_kv_heads));
  const size_t kv_heads = static_cast<size_t>(num_kv_heads);
  const size_t slots = static_cast<size_t>(num_blocks * block_size);
  const int8_t zero_byte_operand = static_cast<int8_t>(Fp8E4m3fnBits(MirroredLevel(kMirroredNibbleSignShift)));

  MirroredKvCache cache;
  cache.key_packed.assign(slots * kv_heads * packed_bytes, 0);
  cache.value_packed.assign(cache.key_packed.size(), 0);
  cache.key_operands.assign(slots * kv_heads * head, zero_byte_operand);
  cache.value_operands.assign(cache.key_operands.size(), zero_byte_operand);
  cache.scales.assign(slots * slot_floats, 0.0f);
  cache.key.assign(static_cast<size_t>(context_len) * kv_heads * head, 0.0f);
  cache.value.assign(cache.key.size(), 0.0f);

  for (int64_t t = 0; t < context_len; ++t) {
    const size_t slot = static_cast<size_t>(block_table[static_cast<size_t>(t / block_size)]) *
                            static_cast<size_t>(block_size) +
                        static_cast<size_t>(t % block_size);
    for (size_t kv = 0; kv < kv_heads; ++kv) {
      for (int plane = 0; plane < 2; ++plane) {
        std::vector<int8_t>& packed = plane == 0 ? cache.key_packed : cache.value_packed;
        std::vector<int8_t>& operands = plane == 0 ? cache.key_operands : cache.value_operands;
        std::vector<float>& dense = plane == 0 ? cache.key : cache.value;

        const std::vector<float> source = rng.NormalHalfExact(half, 0.0f, 1.0f);
        double energy = 0.0;
        for (const float x : source) {
          energy += static_cast<double>(x) * static_cast<double>(x);
        }
        const float scale = static_cast<float>(std::sqrt(energy / static_cast<double>(half)));
        cache.scales[slot * slot_floats + (plane == 0 ? kv : kv_heads + kv)] = scale;

        const size_t row = slot * kv_heads + kv;
        const size_t dense_row = (static_cast<size_t>(t) * kv_heads + kv) * head;
        for (size_t j = 0; j < half; ++j) {
          const int32_t index = std::clamp(
              static_cast<int32_t>(std::floor(source[j] / scale * kMirroredGain + kMirroredAffineBias + 0.5f)), 0,
              kMirroredLevels - 1);
          const float level = MirroredLevel(index);
          const uint32_t nibble = static_cast<uint32_t>(index - kMirroredNibbleSignShift) & kMirroredNibbleMask;
          packed[row * packed_bytes + j] = static_cast<int8_t>(nibble | (nibble << kMirroredNibbleBits));
          const int8_t operand = static_cast<int8_t>(Fp8E4m3fnBits(level));
          operands[row * head + j] = operand;
          operands[row * head + half + j] = operand;
          const float dequantised = level * scale / kMirroredGain;
          dense[dense_row + j] = dequantised;
          dense[dense_row + half + j] = dequantised;
        }
      }
    }
  }
  return cache;
}

}
}
}
