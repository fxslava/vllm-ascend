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

#ifndef VLLM_ASCEND_TESTS_REFERENCE_TURBO_QUANT_CPU_H
#define VLLM_ASCEND_TESTS_REFERENCE_TURBO_QUANT_CPU_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace vllm_ascend {
namespace test {
namespace turboquant_ref {

constexpr int kBits = 4;
constexpr int kLevels = 1 << kBits;
constexpr int kPackFactor = 8 / kBits;
constexpr float kLevelMax = kLevels - 1;
constexpr int kThresholdCount = kLevels - 1;
constexpr float kPackHigh = kLevels;
constexpr float kInt8Bias = 128.0f;
constexpr float kEps = 1e-20f;

constexpr float kLloydMaxCentroids[kLevels] = {
    -2.7325895709951710f, -2.0690172265313920f, -1.6180463860218863f, -1.2562311973471796f,
    -0.9423404564869651f, -0.6567591185324659f, -0.3880482994902919f, -0.1283950298511473f,
    0.1283950298511473f,  0.3880482994902919f,  0.6567591185324659f,  0.9423404564869651f,
    1.2562311973471796f,  1.6180463860218863f,  2.0690172265313920f,  2.7325895709951710f};

constexpr float kLloydMaxThresholds[kThresholdCount] = {
    -2.4008033987632817f, -1.8435318062766393f, -1.4371387916845330f, -1.0992858269170722f,
    -0.7995497875097155f, -0.5224037090113789f, -0.2582216646707196f, 0.0f,
    0.2582216646707196f,  0.5224037090113789f,  0.7995497875097155f,  1.0992858269170722f,
    1.4371387916845330f,  1.8435318062766393f,  2.4008033987632817f};

constexpr uint32_t kPiSeed = 0x5F3759DFu;

inline std::vector<int8_t> cpu_pi_sign_vector(int d) {
  std::vector<int8_t> signs(static_cast<size_t>(d));
  uint32_t state = kPiSeed + static_cast<uint32_t>(d);
  for (int i = 0; i < d; ++i) {
    state = state * 1664525u + 1013904223u;
    signs[static_cast<size_t>(i)] = ((state >> 16) & 1u) ? 1 : -1;
  }
  return signs;
}

inline void cpu_fwht(float* vec, int d) {
  for (int stride = 1; stride < d; stride <<= 1) {
    for (int base = 0; base < d; base += 2 * stride) {
      for (int j = 0; j < stride; ++j) {
        const float top = vec[base + j];
        const float bottom = vec[base + j + stride];
        vec[base + j] = top + bottom;
        vec[base + j + stride] = top - bottom;
      }
    }
  }
  const float norm = 1.0f / std::sqrt(static_cast<float>(d));
  for (int i = 0; i < d; ++i) {
    vec[i] *= norm;
  }
}

inline void cpu_apply_pi(float* vec, int d, const int8_t* sign_vec) {
  for (int i = 0; i < d; ++i) {
    vec[i] *= static_cast<float>(sign_vec[i]);
  }
  cpu_fwht(vec, d);
  for (int i = 0; i < d; ++i) {
    vec[i] *= static_cast<float>(sign_vec[i]);
  }
}

inline float cpu_rms_scale(const float* vec, int d) {
  float sumsq = 0.0f;
  for (int i = 0; i < d; ++i) {
    sumsq += vec[i] * vec[i];
  }
  return std::sqrt(sumsq) * (1.0f / std::sqrt(static_cast<float>(d))) + kEps;
}

inline int cpu_lloyd_max_bin(float u) {
  int q = 0;
  for (int i = 0; i < kThresholdCount; ++i) {
    q += (u > kLloydMaxThresholds[i]) ? 1 : 0;
  }
  return q;
}

inline void cpu_quantize_4bit(const float* vec, int d, int8_t* packed, float* scale) {
  const float s = cpu_rms_scale(vec, d);
  const float inv_scale = 1.0f / s;

  for (int c = 0; c < d / kPackFactor; ++c) {
    float codes[kPackFactor];
    for (int k = 0; k < kPackFactor; ++k) {
      codes[k] = static_cast<float>(cpu_lloyd_max_bin(vec[c * kPackFactor + k] * inv_scale));
    }
    const float byte = codes[0] + kPackHigh * codes[1] - kInt8Bias;
    packed[c] = static_cast<int8_t>(std::lrint(byte));
  }
  *scale = s;
}

inline void cpu_dequantize_4bit(const int8_t* packed, int d, float scale, float* out) {
  for (int p = 0; p < d; ++p) {
    const float byte = static_cast<float>(packed[p / kPackFactor]) + kInt8Bias;
    const float high = std::floor(byte / kPackHigh);
    const float low = byte - kPackHigh * high;
    const float level = (p % kPackFactor == 0) ? low : high;
    out[p] = kLloydMaxCentroids[static_cast<int>(level)] * scale;
  }
}

inline void cpu_dequantize_4bit_batch(const int8_t* packed, int rows, int d, float scale, float* out) {
  const size_t packed_stride = static_cast<size_t>(d / kPackFactor);
  for (int row = 0; row < rows; ++row) {
    cpu_dequantize_4bit(packed + static_cast<size_t>(row) * packed_stride, d, scale,
                        out + static_cast<size_t>(row) * d);
  }
}

inline size_t cpu_scale_slot_floats(int num_kv_heads) {
  constexpr size_t kBurstFloats = 8;
  const size_t lanes = 2u * static_cast<size_t>(num_kv_heads);
  return ((lanes + kBurstFloats - 1) / kBurstFloats) * kBurstFloats;
}

inline void cpu_reshape_and_cache_one(const float* vec, int d, const int8_t* sign_vec, int slot, int num_kv_heads,
                                      int kv_head, int scale_lane, int8_t* cache, float* scale_plane) {
  std::vector<float> rotated(vec, vec + d);
  cpu_apply_pi(rotated.data(), d, sign_vec);

  const size_t packed_stride = static_cast<size_t>(d / kPackFactor);
  const size_t cache_off = (static_cast<size_t>(slot) * num_kv_heads + kv_head) * packed_stride;
  const size_t scale_off = static_cast<size_t>(slot) * cpu_scale_slot_floats(num_kv_heads) + scale_lane;

  cpu_quantize_4bit(rotated.data(), d, cache + cache_off, scale_plane + scale_off);
}

// `softmax_max` and `softmax_mass`, when given, receive one float per head: the running maximum over the whole
// context and the mass sum_i exp(s_i - max), which is what the decode's optional `lse` out-tensor carries
// (turboquant_layout.h). They are reported *before* the kEps floor below, because the kernels write the pair
// before NormalizeHeads adds theirs -- so a test that compares the two is comparing the same quantity.
//
// Trailing and defaulted, so adding them left every existing call site of this reference alone.
inline void cpu_paged_attention_turboquant(const float* query, const int8_t* key_cache, const int8_t* value_cache,
                                           const float* scale_plane, const int32_t* block_table, int context_len,
                                           int num_heads, int num_kv_heads, int d, int block_size, float scale,
                                           const int8_t* sign_vec, float* out, float* softmax_max = nullptr,
                                           float* softmax_mass = nullptr) {
  const int group = num_heads / num_kv_heads;
  const size_t packed_stride = static_cast<size_t>(d / kPackFactor);
  const size_t slot_floats = cpu_scale_slot_floats(num_kv_heads);
  std::vector<float> q_rot(static_cast<size_t>(d));
  std::vector<float> k_levels(static_cast<size_t>(d));
  std::vector<float> v_levels(static_cast<size_t>(d));
  std::vector<float> scores(static_cast<size_t>(std::max(context_len, 0)));
  std::vector<float> acc(static_cast<size_t>(d));

  for (int head = 0; head < num_heads; ++head) {
    const int kv_head = head / group;

    std::copy(query + static_cast<size_t>(head) * d, query + static_cast<size_t>(head + 1) * d, q_rot.begin());
    cpu_apply_pi(q_rot.data(), d, sign_vec);

    float running_max = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < context_len; ++i) {
      const int block = block_table[i / block_size];
      const int offset = i % block_size;
      const size_t row = static_cast<size_t>(block) * block_size + offset;
      const size_t cache_off = (row * num_kv_heads + kv_head) * packed_stride;
      const size_t scale_off = row * slot_floats + static_cast<size_t>(kv_head);

      cpu_dequantize_4bit(key_cache + cache_off, d, 1.0f, k_levels.data());
      float dot = 0.0f;
      for (int c = 0; c < d; ++c) {
        dot += q_rot[static_cast<size_t>(c)] * k_levels[static_cast<size_t>(c)];
      }
      scores[static_cast<size_t>(i)] = dot * scale_plane[scale_off] * scale;
      running_max = std::max(running_max, scores[static_cast<size_t>(i)]);
    }

    float denom = 0.0f;
    for (int i = 0; i < context_len; ++i) {
      scores[static_cast<size_t>(i)] = std::exp(scores[static_cast<size_t>(i)] - running_max);
      denom += scores[static_cast<size_t>(i)];
    }

    if (softmax_max != nullptr) {
      // An empty context leaves running_max at -inf here and the kernels leave it at their own
      // sentinel; neither is a logit, and the mass beside it is 0, which is what a reader branches on.
      softmax_max[static_cast<size_t>(head)] = context_len > 0 ? running_max : 0.0f;
    }
    if (softmax_mass != nullptr) {
      softmax_mass[static_cast<size_t>(head)] = denom;
    }

    std::fill(acc.begin(), acc.end(), 0.0f);
    for (int i = 0; i < context_len; ++i) {
      const int block = block_table[i / block_size];
      const int offset = i % block_size;
      const size_t row = static_cast<size_t>(block) * block_size + offset;
      const size_t cache_off = (row * num_kv_heads + kv_head) * packed_stride;
      const size_t scale_off = row * slot_floats + static_cast<size_t>(num_kv_heads + kv_head);

      cpu_dequantize_4bit(value_cache + cache_off, d, 1.0f, v_levels.data());
      const float weight = scores[static_cast<size_t>(i)] * scale_plane[scale_off];
      for (int c = 0; c < d; ++c) {
        acc[static_cast<size_t>(c)] += weight * v_levels[static_cast<size_t>(c)];
      }
    }

    const float inv_denom = 1.0f / (denom + kEps);
    for (int c = 0; c < d; ++c) {
      acc[static_cast<size_t>(c)] *= inv_denom;
    }

    cpu_apply_pi(acc.data(), d, sign_vec);
    std::copy(acc.begin(), acc.end(), out + static_cast<size_t>(head) * d);
  }
}

struct FidelityMetrics {
  double cosine_similarity = 0.0;
  double snr_db = 0.0;
  double relative_l2 = 0.0;
};

inline FidelityMetrics cpu_fidelity(const std::vector<float>& actual, const std::vector<float>& expected) {
  FidelityMetrics m;
  const size_t n = std::min(actual.size(), expected.size());
  double dot = 0.0;
  double norm_a = 0.0;
  double norm_e = 0.0;
  double noise = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double a = actual[i];
    const double e = expected[i];
    dot += a * e;
    norm_a += a * a;
    norm_e += e * e;
    noise += (a - e) * (a - e);
  }
  const double denom = std::sqrt(norm_a) * std::sqrt(norm_e);
  m.cosine_similarity = denom > 0.0 ? dot / denom : 0.0;
  m.relative_l2 = norm_e > 0.0 ? std::sqrt(noise / norm_e) : 0.0;
  m.snr_db = noise > 0.0 ? 10.0 * std::log10(norm_e / noise) : std::numeric_limits<double>::infinity();
  return m;
}

}
}
}

#endif
