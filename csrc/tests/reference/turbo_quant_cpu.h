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

// Host reference for the TurboQuant 4-bit KV cache, matching the layout and the
// arithmetic of csrc/attention/turboquant/turboquant_codec_950.h element for
// element. Nothing here touches ACL or allocates device memory.
//
// The transform is
//
//     Pi x = D (H (D x)),   D = diag(+-1),   H = normalised Walsh-Hadamard
//
// which is symmetric and an involution, so cpu_apply_pi is both the rotation
// and the un-rotation. Rotation is applied to activations only.
//
// Storage per vector of D channels:
//
//     packed[c] = int8((q[2c] + 16 * q[2c + 1]) - 128),   c in [0, D/2)
//     scale     = ||Pi x||_2 / sqrt(D)
//     q         = sum_{i=1}^{15} [Pi_x / scale > t_i]
//
// reconstructing as scale * c[q] against the 16-level Lloyd-Max table for
// N(0, 1). The -128 bias is undone numerically on the way back, so nothing here
// depends on the int8 bit pattern.

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

// Mirrors TurboQuantCodec<4>.
constexpr int kBits = 4;
constexpr int kLevels = 1 << kBits;                 // 16
constexpr int kPackFactor = 8 / kBits;              // 2 codes per byte
constexpr float kLevelMax = kLevels - 1;            // 15
constexpr int kThresholdCount = kLevels - 1;        // 15 decision boundaries
constexpr float kPackHigh = kLevels;                // 16
constexpr float kInt8Bias = 128.0f;
constexpr float kEps = 1e-20f;

// The 16-level Lloyd-Max quantiser for N(0, 1): the fixed point of
//
//     t_i = (c_{i-1} + c_i) / 2,     c_i = E[X | t_i < X < t_{i+1}],
//
// byte-identical to TurboQuantCodec<4>::Threshold(), to the centroid table the
// host writes into the codec's constant-table image, and to
// LLOYD_MAX_4BIT_{CENTROIDS, THRESHOLDS} in scripts/tq_kv_quant_reference.py.
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

// The +-1 diagonal the Python side derives from TURBOQUANT_PI_SEED. Reproduced
// here with an explicit LCG so the C++ and Python halves agree byte for byte
// without either depending on the other's RNG.
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

// In-place normalised fast Walsh-Hadamard transform of vec[0, d).
// H is symmetric and orthonormal, so applying it twice is the identity.
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

// In-place Pi x = D (H (D x)).  Its own inverse: call it again to undo it.
inline void cpu_apply_pi(float* vec, int d, const int8_t* sign_vec) {
  for (int i = 0; i < d; ++i) {
    vec[i] *= static_cast<float>(sign_vec[i]);
  }
  cpu_fwht(vec, d);
  for (int i = 0; i < d; ++i) {
    vec[i] *= static_cast<float>(sign_vec[i]);
  }
}

// The RMS scale the Lloyd-Max table is stated in: ||vec||_2 / sqrt(d). Written
// as sqrt(sumsq) * (1 / sqrt(d)) rather than as a division, because that is the
// order the kernel evaluates it in.
inline float cpu_rms_scale(const float* vec, int d) {
  float sumsq = 0.0f;
  for (int i = 0; i < d; ++i) {
    sumsq += vec[i] * vec[i];
  }
  return std::sqrt(sumsq) * (1.0f / std::sqrt(static_cast<float>(d))) + kEps;
}

// Bin index of a normalised coordinate: sum_{i=1}^{15} [u > t_i], the strict
// comparison the boundaries are defined with. The kernel computes exactly this
// from the sign bit of t_i - u, so a coordinate sitting on a boundary breaks
// the tie downwards on both sides.
inline int cpu_lloyd_max_bin(float u) {
  int q = 0;
  for (int i = 0; i < kThresholdCount; ++i) {
    q += (u > kLloydMaxThresholds[i]) ? 1 : 0;
  }
  return q;
}

// Quantise one already-rotated vector to 4 bits.
//   vec     [d]     input, rotated
//   packed  [d / 2] output, low nibble = channel 2c, high nibble = 2c + 1
//   scale           output, ||vec||_2 / sqrt(d)
//
// Reads exactly vec[0, d) and writes exactly packed[0, d/2) plus one float, so
// poisoned padding beyond d/2 must survive the call untouched.
//
// Two edges: the kEps floor keeps an all-zero vector from dividing by zero, but
// the table has no exact zero, so an all-zero input reconstructs to
// kEps * c[7], about -1.3e-21. And the byte is built so bins (15, 15) give
// 255 - 128 = +127 and (0, 0) give -128.
//
// The RMS scale depends on the order the squares are summed, so the kernel's
// tree reduction and this serial loop can differ in the last bits.
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

// Expand d/2 packed bytes back into d channels. Passing scale = 1 yields the
// bare centroids c[q], which is what the kernel keeps in UB: it folds the
// per-vector scale into the score row (K) or the softmax probabilities (V),
// which works because the reconstruction is linear in the scale.
//
// Reads exactly packed[0, d/2) and writes exactly out[0, d), so the caller can
// hand it a slice of a larger, poisoned buffer.
inline void cpu_dequantize_4bit(const int8_t* packed, int d, float scale, float* out) {
  for (int p = 0; p < d; ++p) {
    const float byte = static_cast<float>(packed[p / kPackFactor]) + kInt8Bias;
    const float high = std::floor(byte / kPackHigh);
    const float low = byte - kPackHigh * high;
    const float level = (p % kPackFactor == 0) ? low : high;
    out[p] = kLloydMaxCentroids[static_cast<int>(level)] * scale;
  }
}

// Batched expansion, mirroring TurboQuantCodec<4>::Dequantize4Bit(dst, src,
// rows, len). A call with rows < batchRows must confine itself to the live
// prefix: it reads packed[0, rows * d / 2) and writes out[0, rows * d), and
// everything past those bounds must be neither read nor written.
//
// `scale` is applied uniformly; pass 1.0f for the bare centroids.
inline void cpu_dequantize_4bit_batch(const int8_t* packed, int rows, int d, float scale, float* out) {
  const size_t packed_stride = static_cast<size_t>(d / kPackFactor);
  for (int row = 0; row < rows; ++row) {
    cpu_dequantize_4bit(packed + static_cast<size_t>(row) * packed_stride, d, scale,
                        out + static_cast<size_t>(row) * d);
  }
}

// fp32 lanes one token occupies in the scale plane: K then V for every kv head,
// padded to a whole 32-byte burst.  The padding is what lets the kernel scatter
// a token's scales with one aligned DataCopy instead of 2 * num_kv_heads
// four-byte writes.  Mirrors ScaleSlotFloats() in turboquant_kernels.cpp.
inline size_t cpu_scale_slot_floats(int num_kv_heads) {
  constexpr size_t kBurstFloats = 8;
  const size_t lanes = 2u * static_cast<size_t>(num_kv_heads);
  return ((lanes + kBurstFloats - 1) / kBurstFloats) * kBurstFloats;
}

// One (token, kv head, K-or-V) write into the paged cache: rotate, quantise,
// scatter.
//
//   vec         [d]      raw activation (post-RoPE K, or raw projection V)
//   cache       flat int8 [num_blocks, block_size, num_kv_heads, d / 2]
//   scale_plane flat f32  [num_blocks, block_size, scale_slot]
//   scale_lane  kv_head for a key, num_kv_heads + kv_head for a value
//
// `slot` is already the flat (block, offset) row index, which is exactly how the
// kernel treats it: the cache is contiguous over blocks and block offsets.
inline void cpu_reshape_and_cache_one(const float* vec, int d, const int8_t* sign_vec, int slot, int num_kv_heads,
                                      int kv_head, int scale_lane, int8_t* cache, float* scale_plane) {
  std::vector<float> rotated(vec, vec + d);
  cpu_apply_pi(rotated.data(), d, sign_vec);

  const size_t packed_stride = static_cast<size_t>(d / kPackFactor);
  const size_t cache_off = (static_cast<size_t>(slot) * num_kv_heads + kv_head) * packed_stride;
  const size_t scale_off = static_cast<size_t>(slot) * cpu_scale_slot_floats(num_kv_heads) + scale_lane;

  cpu_quantize_4bit(rotated.data(), d, cache + cache_off, scale_plane + scale_off);
}

// Reference paged decode for one query token, executed entirely in the rotated
// basis and un-rotated once at the end:
//
//     Out = Pi (sum_i alpha_i (Pi v_i))  ==  sum_i alpha_i v_i
//
//   query        [num_heads, d]        post-RoPE, unrotated
//   key/value    flat int8 caches as written by cpu_reshape_and_cache_one
//   scale_plane  flat f32 [num_blocks, block_size, scale_slot]
//   block_table  [max_blocks_per_seq]  physical block ids
//   out          [num_heads, d]
inline void cpu_paged_attention_turboquant(const float* query, const int8_t* key_cache, const int8_t* value_cache,
                                           const float* scale_plane, const int32_t* block_table, int context_len,
                                           int num_heads, int num_kv_heads, int d, int block_size, float scale,
                                           const int8_t* sign_vec, float* out) {
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

    // q~ = Pi q, once per head.
    std::copy(query + static_cast<size_t>(head) * d, query + static_cast<size_t>(head + 1) * d, q_rot.begin());
    cpu_apply_pi(q_rot.data(), d, sign_vec);

    // Scores in the rotated basis.  The per-vector step multiplies the whole
    // row, exactly as the kernel folds it.
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

    // Value accumulation, still rotated.
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

    // Out = Pi Out~.
    cpu_apply_pi(acc.data(), d, sign_vec);
    std::copy(acc.begin(), acc.end(), out + static_cast<size_t>(head) * d);
  }
}

// --- fidelity metrics -------------------------------------------------------

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
  // 10 * log10 of a power ratio; the signal and noise are already energies.
  m.snr_db = noise > 0.0 ? 10.0 * std::log10(norm_e / noise) : std::numeric_limits<double>::infinity();
  return m;
}

}  // namespace turboquant_ref
}  // namespace test
}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_TESTS_REFERENCE_TURBO_QUANT_CPU_H
