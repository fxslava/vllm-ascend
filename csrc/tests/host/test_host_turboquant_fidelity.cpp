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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "ascend950_shapes.hpp"
#include "cpu_reference.hpp"
#include "fp16.hpp"
#include "golden_layer3.hpp"
#include "random_data.hpp"
#include "turbo_quant_cpu.h"
#include "turboquant_mirrored_cache.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace s = shapes950;
namespace tq = turboquant_ref;

std::vector<float> RoundToHalf(const std::vector<float>& values) { return QuantizeToHalf(values); }

std::vector<float> ProjectOnCpu(const std::vector<float>& a, const std::vector<float>& w, int64_t m, int64_t k,
                                int64_t n) {
  std::vector<float> out;
  reference::MatmulTransposedB(a, w, m, k, n, &out);
  return RoundToHalf(out);
}

void PrintMetrics(const char* label, const tq::FidelityMetrics& m) {
  std::printf("  %-38s cos=%.6f  snr=%7.2f dB  relL2=%.6f\n", label, m.cosine_similarity, m.snr_db, m.relative_l2);
}

void ExactPagedAttention(const std::vector<float>& query, const std::vector<float>& key,
                         const std::vector<float>& value, int context_len, int num_heads, int num_kv_heads, int d,
                         float scale, std::vector<float>* out) {
  const int group = num_heads / num_kv_heads;
  out->assign(static_cast<size_t>(num_heads) * d, 0.0f);
  for (int head = 0; head < num_heads; ++head) {
    const int kv_head = head / group;
    std::vector<float> scores(static_cast<size_t>(context_len));
    float running_max = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < context_len; ++i) {
      double dot = 0.0;
      const size_t k_base = (static_cast<size_t>(i) * num_kv_heads + kv_head) * d;
      for (int c = 0; c < d; ++c) {
        dot += static_cast<double>(query[static_cast<size_t>(head) * d + c]) * key[k_base + c];
      }
      scores[static_cast<size_t>(i)] = static_cast<float>(dot) * scale;
      running_max = std::max(running_max, scores[static_cast<size_t>(i)]);
    }
    float denom = 0.0f;
    for (int i = 0; i < context_len; ++i) {
      scores[static_cast<size_t>(i)] = std::exp(scores[static_cast<size_t>(i)] - running_max);
      denom += scores[static_cast<size_t>(i)];
    }
    for (int i = 0; i < context_len; ++i) {
      const float weight = scores[static_cast<size_t>(i)] / denom;
      const size_t v_base = (static_cast<size_t>(i) * num_kv_heads + kv_head) * d;
      for (int c = 0; c < d; ++c) {
        (*out)[static_cast<size_t>(head) * d + c] += weight * value[v_base + c];
      }
    }
  }
}

void RunTurboQuantDecode(const std::vector<float>& query, const std::vector<float>& key,
                         const std::vector<float>& value, int context_len, int num_heads, int num_kv_heads, int d,
                         int block_size, float scale, std::vector<float>* out) {
  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(d);
  const int num_blocks = (context_len + block_size - 1) / block_size;
  const size_t packed_stride = static_cast<size_t>(d / tq::kPackFactor);
  const size_t slot_floats = tq::cpu_scale_slot_floats(num_kv_heads);

  std::vector<int8_t> key_cache(static_cast<size_t>(num_blocks) * block_size * num_kv_heads * packed_stride, 0);
  std::vector<int8_t> value_cache(key_cache.size(), 0);
  std::vector<float> scale_plane(static_cast<size_t>(num_blocks) * block_size * slot_floats, 0.0f);

  for (int pos = 0; pos < context_len; ++pos) {
    for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
      const size_t base = (static_cast<size_t>(pos) * num_kv_heads + kv_head) * d;
      tq::cpu_reshape_and_cache_one(key.data() + base, d, signs.data(), pos, num_kv_heads, kv_head, kv_head,
                                    key_cache.data(), scale_plane.data());
      tq::cpu_reshape_and_cache_one(value.data() + base, d, signs.data(), pos, num_kv_heads, kv_head,
                                    num_kv_heads + kv_head, value_cache.data(), scale_plane.data());
    }
  }

  std::vector<int32_t> block_table(static_cast<size_t>(num_blocks));
  for (int b = 0; b < num_blocks; ++b) {
    block_table[static_cast<size_t>(b)] = b;
  }

  out->assign(static_cast<size_t>(num_heads) * d, 0.0f);
  tq::cpu_paged_attention_turboquant(query.data(), key_cache.data(), value_cache.data(), scale_plane.data(),
                                     block_table.data(), context_len, num_heads, num_kv_heads, d, block_size, scale,
                                     signs.data(), out->data());
}

std::vector<float> LayerTailFromContext(const GoldenLayer3& golden, const std::vector<float>& attn_context,
                                        const std::vector<float>& attn_gate) {
  std::vector<float> gated(static_cast<size_t>(s::kTokens * s::kQDim));
  for (size_t i = 0; i < gated.size(); ++i) {
    const float sigmoid = 1.0f / (1.0f + std::exp(-attn_gate[i]));
    gated[i] = attn_context[i] * sigmoid;
  }
  gated = RoundToHalf(gated);

  const std::vector<float> attn_out = ProjectOnCpu(gated, golden.w_out, s::kTokens, s::kQDim, s::kHidden);

  std::vector<float> x(golden.input_x);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] += attn_out[i];
  }
  x = RoundToHalf(x);

  std::vector<float> norm2;
  std::vector<float> rstd;
  reference::RmsNorm(x, golden.post_attn_norm_gamma, s::kTokens, s::kHidden, s::kRmsNormEps, &norm2, &rstd);
  norm2 = RoundToHalf(norm2);

  const std::vector<float> mlp_gate =
      ProjectOnCpu(norm2, golden.w_gate, s::kTokens, s::kHidden, s::kIntermediate);
  const std::vector<float> mlp_up = ProjectOnCpu(norm2, golden.w_up, s::kTokens, s::kHidden, s::kIntermediate);

  std::vector<float> gate_up;
  gate_up.reserve(mlp_gate.size() + mlp_up.size());
  gate_up.insert(gate_up.end(), mlp_gate.begin(), mlp_gate.end());
  gate_up.insert(gate_up.end(), mlp_up.begin(), mlp_up.end());

  std::vector<float> swiglu;
  reference::SiluAndMul(gate_up, s::kTokens, s::kIntermediate, &swiglu);
  swiglu = RoundToHalf(swiglu);

  const std::vector<float> mlp_out =
      ProjectOnCpu(swiglu, golden.w_down, s::kTokens, s::kIntermediate, s::kHidden);

  for (size_t i = 0; i < x.size(); ++i) {
    x[i] += mlp_out[i];
  }
  return RoundToHalf(x);
}

TEST(TurboQuantCodecInvariants, PiIsAnInvolution) {
  constexpr int d = 256;
  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(d);
  std::vector<float> x(static_cast<size_t>(d));
  for (int i = 0; i < d; ++i) {
    x[static_cast<size_t>(i)] = std::sin(0.37f * static_cast<float>(i)) * 3.1f;
  }
  const std::vector<float> original = x;

  tq::cpu_apply_pi(x.data(), d, signs.data());
  tq::cpu_apply_pi(x.data(), d, signs.data());

  double worst = 0.0;
  for (int i = 0; i < d; ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(x[static_cast<size_t>(i)]) -
                                      original[static_cast<size_t>(i)]));
  }
  std::printf("[turboquant] Pi(Pi(x)) - x max abs error: %.3e\n", worst);
  EXPECT_LT(worst, 1e-4);
}

TEST(TurboQuantCodecInvariants, PiPreservesDotProducts) {
  constexpr int d = 256;
  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(d);
  std::vector<float> a(static_cast<size_t>(d));
  std::vector<float> b(static_cast<size_t>(d));
  for (int i = 0; i < d; ++i) {
    a[static_cast<size_t>(i)] = std::cos(0.11f * static_cast<float>(i));
    b[static_cast<size_t>(i)] = std::sin(0.23f * static_cast<float>(i)) * 2.0f;
  }
  double plain = 0.0;
  for (int i = 0; i < d; ++i) {
    plain += static_cast<double>(a[static_cast<size_t>(i)]) * b[static_cast<size_t>(i)];
  }
  tq::cpu_apply_pi(a.data(), d, signs.data());
  tq::cpu_apply_pi(b.data(), d, signs.data());
  double rotated = 0.0;
  for (int i = 0; i < d; ++i) {
    rotated += static_cast<double>(a[static_cast<size_t>(i)]) * b[static_cast<size_t>(i)];
  }
  std::printf("[turboquant] q.k unrotated %.6f vs rotated %.6f\n", plain, rotated);
  EXPECT_NEAR(rotated, plain, 1e-3 * std::max(1.0, std::fabs(plain)));
}

TEST(TurboQuantCodecInvariants, PiSignVectorMatchesThePythonReference) {
  const std::vector<int8_t> d256 = tq::cpu_pi_sign_vector(256);
  const std::vector<int8_t> d128 = tq::cpu_pi_sign_vector(128);
  ASSERT_EQ(d256.size(), 256u);
  ASSERT_EQ(d128.size(), 128u);

  const std::vector<int8_t> expected_256{-1, 1, -1, -1, -1, -1, 1, 1};
  const std::vector<int8_t> expected_128{1, -1, 1, -1, -1, -1, 1, 1};
  for (size_t i = 0; i < expected_256.size(); ++i) {
    EXPECT_EQ(d256[i], expected_256[i]) << "head_size=256 channel " << i;
    EXPECT_EQ(d128[i], expected_128[i]) << "head_size=128 channel " << i;
  }
  for (int8_t sign : d256) {
    ASSERT_TRUE(sign == 1 || sign == -1);
  }
}

TEST(TurboQuantCodecInvariants, NibblePackRoundTripsExactly) {
  constexpr int d = 64;
  std::vector<int8_t> packed(static_cast<size_t>(d / tq::kPackFactor));
  std::vector<float> levels(static_cast<size_t>(d));
  for (int i = 0; i < d; ++i) {
    levels[static_cast<size_t>(i)] = static_cast<float>(i % tq::kLevels);
  }
  for (int c = 0; c < d / tq::kPackFactor; ++c) {
    const float byte = levels[static_cast<size_t>(2 * c)] +
                       tq::kPackHigh * levels[static_cast<size_t>(2 * c + 1)] - tq::kInt8Bias;
    packed[static_cast<size_t>(c)] = static_cast<int8_t>(std::lrint(byte));
  }
  std::vector<float> out(static_cast<size_t>(d));
  tq::cpu_dequantize_4bit(packed.data(), d, 1.0f, out.data());
  int mismatches = 0;
  for (int i = 0; i < d; ++i) {
    const int level = static_cast<int>(levels[static_cast<size_t>(i)]);
    if (out[static_cast<size_t>(i)] != tq::kLloydMaxCentroids[level]) {
      ++mismatches;
    }
  }
  EXPECT_EQ(mismatches, 0);
}

constexpr uint32_t kPoisonBits[] = {0x7F800001u, 0x7FC00000u, 0xDEADBEEFu};
constexpr uint32_t kPoisonBitsAlt[] = {0x7FC00001u, 0x7F800002u, 0xBAADF00Du};
constexpr int8_t kPoisonInt8A = 127;
constexpr int8_t kPoisonInt8B = -128;

float BitsToFloat(uint32_t bits) {
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

uint32_t FloatToBits(float value) {
  uint32_t out = 0;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

void PoisonFloats(float* dst, size_t count, int variant = 0) {
  const uint32_t* table = (variant == 0) ? kPoisonBits : kPoisonBitsAlt;
  for (size_t i = 0; i < count; ++i) {
    dst[i] = BitsToFloat(table[i % 3]);
  }
}

void PoisonInt8(int8_t* dst, size_t count, int variant = 0) {
  for (size_t i = 0; i < count; ++i) {
    const bool even = (i % 2) == 0;
    dst[i] = (variant == 0) ? (even ? kPoisonInt8A : kPoisonInt8B) : (even ? kPoisonInt8B : kPoisonInt8A);
  }
}

bool IsPoisonFloat(float value) {
  const uint32_t bits = FloatToBits(value);
  for (uint32_t pattern : kPoisonBits) {
    if (bits == pattern) {
      return true;
    }
  }
  for (uint32_t pattern : kPoisonBitsAlt) {
    if (bits == pattern) {
      return true;
    }
  }
  return false;
}

template <typename T>
class GuardedBuffer {
 public:
  static constexpr uint8_t kGuardByte = 0xA5;
  static constexpr size_t kGuardBytes = 64;
  static constexpr size_t kGuardElems = kGuardBytes / sizeof(T);

  explicit GuardedBuffer(size_t count) : count_(count), storage_(count + 2 * kGuardElems) { ResetGuards(); }

  T* data() { return storage_.data() + kGuardElems; }
  const T* data() const { return storage_.data() + kGuardElems; }
  size_t size() const { return count_; }

  void ResetGuards() {
    std::memset(storage_.data(), kGuardByte, kGuardBytes);
    std::memset(storage_.data() + kGuardElems + count_, kGuardByte, kGuardBytes);
  }

  bool GuardsIntact() const {
    const uint8_t* low = reinterpret_cast<const uint8_t*>(storage_.data());
    const uint8_t* high = reinterpret_cast<const uint8_t*>(storage_.data() + kGuardElems + count_);
    for (size_t i = 0; i < kGuardBytes; ++i) {
      if (low[i] != kGuardByte || high[i] != kGuardByte) {
        return false;
      }
    }
    return true;
  }

 private:
  size_t count_;
  std::vector<T> storage_;
};

std::vector<float> MakeVector(int d, float amplitude = 1.0f, uint32_t seed = 7u) {
  std::vector<float> out(static_cast<size_t>(d));
  uint32_t state = seed;
  for (int i = 0; i < d; ++i) {
    state = state * 1664525u + 1013904223u;
    const float unit = static_cast<float>(static_cast<int32_t>(state >> 8) % 20001 - 10000) / 10000.0f;
    out[static_cast<size_t>(i)] = unit * amplitude;
  }
  return out;
}

TEST(TurboQuantPoison, QuantizeIgnoresPoisonedDestinationAndPadding) {
  constexpr int d = 128;
  constexpr int batch_rows = 8;
  const size_t packed_stride = static_cast<size_t>(d / tq::kPackFactor);
  const std::vector<float> vec = MakeVector(d, 2.5f);

  std::vector<int8_t> reference(packed_stride);
  float reference_scale = 0.0f;
  tq::cpu_quantize_4bit(vec.data(), d, reference.data(), &reference_scale);

  for (int variant = 0; variant < 2; ++variant) {
    std::vector<int8_t> packed(packed_stride * batch_rows);
    PoisonInt8(packed.data(), packed.size(), variant);
    std::vector<float> scale_slot(static_cast<size_t>(batch_rows));
    PoisonFloats(scale_slot.data(), scale_slot.size(), variant);

    tq::cpu_quantize_4bit(vec.data(), d, packed.data(), scale_slot.data());

    for (size_t i = 0; i < packed_stride; ++i) {
      ASSERT_EQ(packed[i], reference[i]) << "variant " << variant << " byte " << i;
    }
    EXPECT_EQ(FloatToBits(scale_slot[0]), FloatToBits(reference_scale));
    EXPECT_FALSE(IsPoisonFloat(scale_slot[0]));

    std::vector<int8_t> expected_tail(packed.size() - packed_stride);
    PoisonInt8(expected_tail.data(), expected_tail.size(), variant);
    for (size_t i = 0; i < expected_tail.size(); ++i) {
      const size_t absolute = packed_stride + i;
      const bool even = (absolute % 2) == 0;
      const int8_t want = (variant == 0) ? (even ? kPoisonInt8A : kPoisonInt8B)
                                         : (even ? kPoisonInt8B : kPoisonInt8A);
      ASSERT_EQ(packed[absolute], want) << "quantise wrote into padding at " << absolute;
    }
    for (size_t i = 1; i < scale_slot.size(); ++i) {
      EXPECT_TRUE(IsPoisonFloat(scale_slot[i])) << "quantise wrote a scale it does not own at " << i;
    }
  }
}

TEST(TurboQuantPoison, QuantizeWritesNoByteOutsideItsSlice) {
  constexpr int d = 256;
  const std::vector<float> vec = MakeVector(d, 4.0f);

  GuardedBuffer<int8_t> packed(static_cast<size_t>(d / tq::kPackFactor));
  GuardedBuffer<float> scale(1);
  PoisonInt8(packed.data(), packed.size());
  PoisonFloats(scale.data(), scale.size());

  tq::cpu_quantize_4bit(vec.data(), d, packed.data(), scale.data());

  EXPECT_TRUE(packed.GuardsIntact()) << "quantise wrote past the packed slice";
  EXPECT_TRUE(scale.GuardsIntact()) << "quantise wrote past the scale slot";
}

TEST(TurboQuantPoison, DequantizePartialBatchIgnoresPoisonedTail) {
  for (int d : {64, 128, 256}) {
    for (int batch_rows : {4, 8}) {
      for (int rows : {1, 3}) {
        if (rows > batch_rows) {
          continue;
        }
        const size_t packed_stride = static_cast<size_t>(d / tq::kPackFactor);
        const size_t live_packed = packed_stride * static_cast<size_t>(rows);
        const size_t live_out = static_cast<size_t>(d) * static_cast<size_t>(rows);
        const size_t batch_packed = packed_stride * static_cast<size_t>(batch_rows);
        const size_t batch_out = static_cast<size_t>(d) * static_cast<size_t>(batch_rows);

        std::vector<int8_t> live(live_packed);
        for (size_t i = 0; i < live.size(); ++i) {
          const int low = static_cast<int>(i % 16);
          const int high = static_cast<int>((i / 16) % 16);
          live[i] = static_cast<int8_t>(low + 16 * high - 128);
        }

        std::vector<float> results[2];
        for (int variant = 0; variant < 2; ++variant) {
          std::vector<int8_t> packed(batch_packed);
          PoisonInt8(packed.data(), packed.size(), variant);
          std::copy(live.begin(), live.end(), packed.begin());

          std::vector<float> out(batch_out);
          PoisonFloats(out.data(), out.size(), variant);

          tq::cpu_dequantize_4bit_batch(packed.data(), rows, d, 1.0f, out.data());

          for (size_t i = 0; i < live_out; ++i) {
            ASSERT_FALSE(IsPoisonFloat(out[i]))
                << "poison leaked into the payload: d=" << d << " rows=" << rows << " index=" << i;
            ASSERT_TRUE(std::isfinite(out[i])) << "non-finite payload at " << i;
          }
          for (size_t i = live_out; i < batch_out; ++i) {
            ASSERT_TRUE(IsPoisonFloat(out[i]))
                << "dequantise wrote past its rows: d=" << d << " rows=" << rows << " index=" << i;
          }
          results[variant].assign(out.begin(), out.begin() + static_cast<ptrdiff_t>(live_out));
        }

        for (size_t i = 0; i < live_out; ++i) {
          ASSERT_EQ(FloatToBits(results[0][i]), FloatToBits(results[1][i]))
              << "output depends on padding content: d=" << d << " rows=" << rows << " index=" << i;
        }
      }
    }
  }
}

TEST(TurboQuantPoison, DequantizeWritesNoByteOutsideItsSlice) {
  constexpr int d = 128;
  constexpr int rows = 3;
  const size_t packed_stride = static_cast<size_t>(d / tq::kPackFactor);

  GuardedBuffer<int8_t> packed(packed_stride * rows);
  GuardedBuffer<float> out(static_cast<size_t>(d) * rows);
  PoisonInt8(packed.data(), packed.size());
  PoisonFloats(out.data(), out.size());

  tq::cpu_dequantize_4bit_batch(packed.data(), rows, d, 1.0f, out.data());

  EXPECT_TRUE(packed.GuardsIntact()) << "dequantise wrote into its own source guard";
  EXPECT_TRUE(out.GuardsIntact()) << "dequantise wrote past the destination slice";
}

TEST(TurboQuantPoison, ApplyPiIgnoresSurroundingMemory) {
  for (int d : {64, 128, 256}) {
    const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(d);
    const std::vector<float> vec = MakeVector(d, 3.0f);

    std::vector<float> reference = vec;
    tq::cpu_apply_pi(reference.data(), d, signs.data());

    for (int variant = 0; variant < 2; ++variant) {
      GuardedBuffer<float> arena(static_cast<size_t>(d) * 4);
      PoisonFloats(arena.data(), arena.size(), variant);
      std::copy(vec.begin(), vec.end(), arena.data());

      tq::cpu_apply_pi(arena.data(), d, signs.data());

      ASSERT_TRUE(arena.GuardsIntact()) << "ApplyPi wrote past the arena, d=" << d;
      for (int i = 0; i < d; ++i) {
        ASSERT_EQ(FloatToBits(arena.data()[i]), FloatToBits(reference[static_cast<size_t>(i)]))
            << "ApplyPi result depends on neighbouring memory: d=" << d << " index=" << i;
      }
      for (size_t i = static_cast<size_t>(d); i < arena.size(); ++i) {
        ASSERT_TRUE(IsPoisonFloat(arena.data()[i])) << "ApplyPi wrote past its length at " << i;
      }
    }
  }
}

std::vector<int32_t> BuildExpandOffsets(size_t batch_len) {
  std::vector<int32_t> table(batch_len);
  for (size_t p = 0; p < batch_len; ++p) {
    const float halved = static_cast<float>(p) / static_cast<float>(tq::kPackFactor);
    table[p] = static_cast<int32_t>(std::floor(halved) * static_cast<float>(sizeof(float)));
  }
  return table;
}

std::vector<float> BuildOddSelect(size_t batch_len) {
  std::vector<float> table(batch_len);
  for (size_t p = 0; p < batch_len; ++p) {
    const float parity = std::floor(static_cast<float>(p)) - 2.0f * std::floor(static_cast<float>(p) * 0.5f);
    table[p] = parity;
  }
  return table;
}

std::vector<int32_t> BuildPackOffsets(size_t packed_len, bool odd) {
  std::vector<int32_t> table(packed_len);
  const float stride = static_cast<float>(tq::kPackFactor * sizeof(float));
  const float first = odd ? static_cast<float>(sizeof(float)) : 0.0f;
  for (size_t c = 0; c < packed_len; ++c) {
    table[c] = static_cast<int32_t>(first + stride * static_cast<float>(c));
  }
  return table;
}

void TableDrivenDequantize(const int8_t* packed, int rows, int len, int batch_rows, float* out, int variant) {
  const size_t batch_len = static_cast<size_t>(batch_rows) * static_cast<size_t>(len);
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(len);
  const size_t packed_count = n / tq::kPackFactor;

  const std::vector<int32_t> expand = BuildExpandOffsets(batch_len);
  const std::vector<float> odd_select = BuildOddSelect(batch_len);

  std::vector<float> swap(batch_len);
  std::vector<float> scratch(batch_len);
  PoisonFloats(swap.data(), swap.size(), variant);
  PoisonFloats(scratch.data(), scratch.size(), variant);

  for (size_t c = 0; c < packed_count; ++c) {
    scratch[c] = static_cast<float>(packed[c]) + tq::kInt8Bias;
  }

  for (size_t p = 0; p < n; ++p) {
    const size_t source = static_cast<size_t>(expand[p]) / sizeof(float);
    swap[p] = scratch[source];
  }

  for (size_t p = 0; p < n; ++p) {
    const float high = std::floor(swap[p] / tq::kPackHigh);
    const float low = swap[p] - tq::kPackHigh * high;
    const float bin = low + odd_select[p] * (high - low);
    const size_t offset = static_cast<size_t>(std::lrint(bin * static_cast<float>(sizeof(float))));
    out[p] = tq::kLloydMaxCentroids[offset / sizeof(float)];
  }
}

void TableDrivenQuantize(const float* vec, int len, int batch_rows, int8_t* packed, float* scale, int variant) {
  const size_t batch_len = static_cast<size_t>(batch_rows) * static_cast<size_t>(len);
  const size_t n = static_cast<size_t>(len);
  const size_t packed_count = n / tq::kPackFactor;

  const std::vector<int32_t> even = BuildPackOffsets(packed_count, false);
  const std::vector<int32_t> odd = BuildPackOffsets(packed_count, true);

  std::vector<float> swap(batch_len);
  std::vector<float> scratch(batch_len);
  PoisonFloats(swap.data(), swap.size(), variant);
  PoisonFloats(scratch.data(), scratch.size(), variant);

  float sumsq = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    sumsq += vec[i] * vec[i];
  }
  const float rms = std::sqrt(sumsq) * (1.0f / std::sqrt(static_cast<float>(len))) + tq::kEps;
  const float neg_inv_scale = -1.0f / rms;

  for (size_t i = 0; i < n; ++i) {
    const float neg_u = vec[i] * neg_inv_scale;
    int32_t bin = 0;
    for (int t = 0; t < tq::kThresholdCount; ++t) {
      const float diff = neg_u + tq::kLloydMaxThresholds[t];
      uint32_t bits = 0;
      std::memcpy(&bits, &diff, sizeof(bits));
      bin += static_cast<int32_t>(bits >> 31);
    }
    scratch[i] = static_cast<float>(bin);
  }

  for (size_t c = 0; c < packed_count; ++c) {
    swap[c] = scratch[static_cast<size_t>(even[c]) / sizeof(float)];
    swap[packed_count + c] = scratch[static_cast<size_t>(odd[c]) / sizeof(float)];
  }
  for (size_t c = 0; c < packed_count; ++c) {
    const float byte = swap[c] + tq::kPackHigh * swap[packed_count + c] - tq::kInt8Bias;
    packed[c] = static_cast<int8_t>(std::lrint(byte));
  }
  *scale = rms;
}

TEST(TurboQuantTableDriven, DequantizeMatchesTheReferenceOnPartialBatches) {
  for (int len : {64, 128, 256}) {
    for (int batch_rows : {4, 8}) {
      for (int rows : {1, 3, 4}) {
        if (rows > batch_rows) {
          continue;
        }
        const size_t packed_stride = static_cast<size_t>(len / tq::kPackFactor);
        std::vector<int8_t> packed(packed_stride * static_cast<size_t>(rows));
        for (size_t i = 0; i < packed.size(); ++i) {
          packed[i] = static_cast<int8_t>(static_cast<int>(i % 16) + 16 * static_cast<int>((i / 5) % 16) - 128);
        }

        std::vector<float> expected(static_cast<size_t>(rows) * len);
        tq::cpu_dequantize_4bit_batch(packed.data(), rows, len, 1.0f, expected.data());

        for (int variant = 0; variant < 2; ++variant) {
          GuardedBuffer<float> out(static_cast<size_t>(rows) * len);
          PoisonFloats(out.data(), out.size(), variant);
          TableDrivenDequantize(packed.data(), rows, len, batch_rows, out.data(), variant);

          ASSERT_TRUE(out.GuardsIntact()) << "len=" << len << " rows=" << rows;
          for (size_t i = 0; i < expected.size(); ++i) {
            ASSERT_FALSE(IsPoisonFloat(out.data()[i]))
                << "poisoned UB padding reached the output: len=" << len << " rows=" << rows << " index=" << i;
            ASSERT_EQ(FloatToBits(out.data()[i]), FloatToBits(expected[i]))
                << "table-driven path diverged: len=" << len << " rows=" << rows << " index=" << i;
          }
        }
      }
    }
  }
}

TEST(TurboQuantTableDriven, QuantizeMatchesTheReferenceWithPoisonedScratch) {
  for (int len : {64, 128, 256}) {
    for (int batch_rows : {4, 8}) {
      const std::vector<float> vec = MakeVector(len, 6.0f, 29u);
      const size_t packed_count = static_cast<size_t>(len / tq::kPackFactor);

      std::vector<int8_t> expected(packed_count);
      float expected_scale = 0.0f;
      tq::cpu_quantize_4bit(vec.data(), len, expected.data(), &expected_scale);

      for (int variant = 0; variant < 2; ++variant) {
        GuardedBuffer<int8_t> packed(packed_count);
        PoisonInt8(packed.data(), packed.size(), variant);
        float scale = BitsToFloat(kPoisonBits[0]);

        TableDrivenQuantize(vec.data(), len, batch_rows, packed.data(), &scale, variant);

        ASSERT_TRUE(packed.GuardsIntact()) << "len=" << len;
        EXPECT_EQ(FloatToBits(scale), FloatToBits(expected_scale));
        for (size_t i = 0; i < packed_count; ++i) {
          ASSERT_EQ(packed.data()[i], expected[i])
              << "table-driven pack diverged: len=" << len << " batch_rows=" << batch_rows << " byte=" << i;
        }
      }
    }
  }
}

TEST(TurboQuantEdgeCases, AllZeroVectorIsNumericallyStable) {
  for (int d : {64, 128, 256}) {
    const std::vector<float> zeros(static_cast<size_t>(d), 0.0f);
    std::vector<int8_t> packed(static_cast<size_t>(d / tq::kPackFactor));
    PoisonInt8(packed.data(), packed.size());
    float step = BitsToFloat(kPoisonBits[0]);

    tq::cpu_quantize_4bit(zeros.data(), d, packed.data(), &step);

    ASSERT_TRUE(std::isfinite(step)) << "scale is not finite for an all-zero vector, d=" << d;
    ASSERT_GT(step, 0.0f) << "scale collapsed to zero, d=" << d;
    EXPECT_FLOAT_EQ(step, tq::kEps);

    std::vector<float> levels(static_cast<size_t>(d));
    tq::cpu_dequantize_4bit(packed.data(), d, 1.0f, levels.data());
    for (int i = 0; i < d; ++i) {
      ASSERT_TRUE(std::isfinite(levels[static_cast<size_t>(i)]));
      EXPECT_FLOAT_EQ(levels[static_cast<size_t>(i)], tq::kLloydMaxCentroids[7]) << "channel " << i;
    }

    std::vector<float> recon(static_cast<size_t>(d));
    tq::cpu_dequantize_4bit(packed.data(), d, step, recon.data());
    for (int i = 0; i < d; ++i) {
      ASSERT_TRUE(std::isfinite(recon[static_cast<size_t>(i)]));
      EXPECT_LT(std::fabs(recon[static_cast<size_t>(i)]), 1e-18f) << "channel " << i;
    }
  }
  std::printf("[turboquant] all-zero vector: scale = kEps, every bin 7, |recon| < 1e-18\n");
}

TEST(TurboQuantEdgeCases, SingleDominantOutlier) {
  constexpr int d = 128;
  for (float magnitude : {1.0e4f, -1.0e4f}) {
    std::vector<float> vec(static_cast<size_t>(d), 0.0f);
    vec[0] = magnitude;

    std::vector<int8_t> packed(static_cast<size_t>(d / tq::kPackFactor));
    PoisonInt8(packed.data(), packed.size());
    float step = 0.0f;
    tq::cpu_quantize_4bit(vec.data(), d, packed.data(), &step);

    EXPECT_NEAR(step, std::fabs(magnitude) / std::sqrt(static_cast<float>(d)),
                std::fabs(magnitude) * 1e-6f);
    ASSERT_TRUE(std::isfinite(step));

    std::vector<float> levels(static_cast<size_t>(d));
    tq::cpu_dequantize_4bit(packed.data(), d, 1.0f, levels.data());
    const int extreme_bin = (magnitude > 0.0f) ? tq::kLevels - 1 : 0;
    EXPECT_FLOAT_EQ(levels[0], tq::kLloydMaxCentroids[extreme_bin])
        << "the outlier must saturate the table's outermost bin";
    for (int i = 1; i < d; ++i) {
      EXPECT_FLOAT_EQ(levels[static_cast<size_t>(i)], tq::kLloydMaxCentroids[7])
          << "zero channel " << i << " moved off bin 7";
    }

    for (size_t i = 0; i < packed.size(); ++i) {
      ASSERT_GE(static_cast<int>(packed[i]), -128);
      ASSERT_LE(static_cast<int>(packed[i]), 127);
    }

    std::vector<float> recon(static_cast<size_t>(d));
    tq::cpu_dequantize_4bit(packed.data(), d, step, recon.data());
    const float clipped =
        magnitude * tq::kLloydMaxCentroids[tq::kLevels - 1] / std::sqrt(static_cast<float>(d));
    EXPECT_NEAR(recon[0], clipped, std::fabs(clipped) * 1e-5f);
    EXPECT_LT(std::fabs(recon[0]), std::fabs(magnitude)) << "a clipped outlier cannot grow";
  }
}

TEST(TurboQuantEdgeCases, ClampingAndSaturationExtremes) {
  constexpr int d = 64;
  constexpr float amplitude = 3.0f;

  struct Case {
    const char* name;
    float first;
    float second;
    int expected_low;
    int expected_high;
  };
  const Case cases[] = {
      {"(+A, +A) -> (15, 15) -> +127", amplitude, amplitude, 15, 15},
      {"(-A, -A) -> ( 0,  0) -> -128", -amplitude, -amplitude, 0, 0},
      {"(+A, -A) -> (15,  0) -> -113", amplitude, -amplitude, 15, 0},
      {"(-A, +A) -> ( 0, 15) -> +112", -amplitude, amplitude, 0, 15},
  };

  const int quiet_byte = 7 + 16 * 7 - 128;

  for (const Case& c : cases) {
    std::vector<float> vec(static_cast<size_t>(d), 0.0f);
    vec[0] = c.first;
    vec[1] = c.second;

    std::vector<int8_t> packed(static_cast<size_t>(d / tq::kPackFactor));
    PoisonInt8(packed.data(), packed.size());
    float step = 0.0f;
    tq::cpu_quantize_4bit(vec.data(), d, packed.data(), &step);

    const float u = amplitude / step;
    ASSERT_GT(u, tq::kLloydMaxThresholds[tq::kThresholdCount - 1])
        << c.name << ": the sparse outlier no longer saturates, so the endpoints are untested";

    const int expected_byte = c.expected_low + 16 * c.expected_high - 128;
    ASSERT_EQ(static_cast<int>(packed[0]), expected_byte) << c.name;
    for (size_t i = 1; i < packed.size(); ++i) {
      ASSERT_EQ(static_cast<int>(packed[i]), quiet_byte) << c.name << " at quiet byte " << i;
    }

    std::vector<float> levels(static_cast<size_t>(d));
    tq::cpu_dequantize_4bit(packed.data(), d, 1.0f, levels.data());
    EXPECT_FLOAT_EQ(levels[0], tq::kLloydMaxCentroids[c.expected_low]) << c.name << " channel 0";
    EXPECT_FLOAT_EQ(levels[1], tq::kLloydMaxCentroids[c.expected_high]) << c.name << " channel 1";
    for (int i = 2; i < d; ++i) {
      EXPECT_FLOAT_EQ(levels[static_cast<size_t>(i)], tq::kLloydMaxCentroids[7]) << c.name << " channel " << i;
    }
    std::printf("[turboquant] %s  scale=%.6f  u=%.3f\n", c.name, step, u);
  }
}

TEST(TurboQuantEdgeCases, EveryNibblePairRoundTrips) {
  int mismatches = 0;
  for (int low = 0; low < 16; ++low) {
    for (int high = 0; high < 16; ++high) {
      const int byte = low + 16 * high - 128;
      ASSERT_GE(byte, -128);
      ASSERT_LE(byte, 127);
      const int8_t packed = static_cast<int8_t>(byte);
      float out[2] = {0.0f, 0.0f};
      tq::cpu_dequantize_4bit(&packed, 2, 1.0f, out);
      if (out[0] != tq::kLloydMaxCentroids[low] || out[1] != tq::kLloydMaxCentroids[high]) {
        ++mismatches;
      }
    }
  }
  EXPECT_EQ(mismatches, 0);
}

inline float WidestHalfBin() {
  float widest = 0.0f;
  for (int i = 0; i < tq::kThresholdCount; ++i) {
    widest = std::max(widest, tq::kLloydMaxThresholds[i] - tq::kLloydMaxCentroids[i]);
    widest = std::max(widest, tq::kLloydMaxCentroids[i + 1] - tq::kLloydMaxThresholds[i]);
  }
  return widest;
}

TEST(TurboQuantEdgeCases, PartialBatchesMatchPerRowQuantisation) {
  const float kWidestHalfBin = WidestHalfBin();
  for (int d : {64, 128, 256}) {
    for (int batch_rows : {4, 8}) {
      for (int rows : {1, 3}) {
        const size_t packed_stride = static_cast<size_t>(d / tq::kPackFactor);
        std::vector<int8_t> packed(packed_stride * static_cast<size_t>(batch_rows));
        PoisonInt8(packed.data(), packed.size());
        std::vector<float> scales(static_cast<size_t>(batch_rows));
        PoisonFloats(scales.data(), scales.size());

        std::vector<float> rows_data;
        for (int row = 0; row < rows; ++row) {
          const std::vector<float> vec =
              MakeVector(d, 1.0f + static_cast<float>(row) * 4.0f, 11u + static_cast<uint32_t>(row));
          rows_data.insert(rows_data.end(), vec.begin(), vec.end());
          tq::cpu_quantize_4bit(vec.data(), d, packed.data() + static_cast<size_t>(row) * packed_stride,
                                scales.data() + row);
        }

        std::vector<float> expanded(static_cast<size_t>(d) * static_cast<size_t>(batch_rows));
        PoisonFloats(expanded.data(), expanded.size());
        tq::cpu_dequantize_4bit_batch(packed.data(), rows, d, 1.0f, expanded.data());

        for (int row = 0; row < rows; ++row) {
          ASSERT_TRUE(std::isfinite(scales[static_cast<size_t>(row)]));
          ASSERT_GT(scales[static_cast<size_t>(row)], 0.0f);
          for (int i = 0; i < d; ++i) {
            const size_t idx = static_cast<size_t>(row) * d + i;
            const float recon = expanded[idx] * scales[static_cast<size_t>(row)];
            const float want = rows_data[idx];
            const float u = want / scales[static_cast<size_t>(row)];
            ASSERT_LE(std::fabs(u), tq::kLloydMaxThresholds[tq::kThresholdCount - 1])
                << "this fixture clipped, so the in-range bound below does not apply: d=" << d << " row=" << row
                << " channel=" << i;
            ASSERT_LE(std::fabs(recon - want), kWidestHalfBin * scales[static_cast<size_t>(row)] * 1.001f)
                << "d=" << d << " rows=" << rows << " row=" << row << " channel=" << i;
          }
        }
        for (int row = rows; row < batch_rows; ++row) {
          EXPECT_TRUE(IsPoisonFloat(scales[static_cast<size_t>(row)]))
              << "a scale outside the live rows was written";
        }
      }
    }
  }
}

// The kv4fp8 cache layout both kernels issue as DMA (turboquant_layout.h, NzTiledPackedByte): every cell of a
// block is written exactly once, the decode's one-burst (tile, kv head) window holds exactly that tile's cells,
// the plane read from the window unpacks in the Cube's NZ order, and the writer's strided burst lands on them.
TEST(TurboQuantNzTiledCache, TilesAreContiguousNzImagesAndTheWriterBurstHitsThem) {
  namespace tqt = vllm_ascend::turboquant;
  constexpr uint64_t kTile = tqt::kCubeTileRows;
  constexpr uint64_t kC0 = tqt::kOperandC0;
  constexpr uint64_t kBlocks = 2;
  // The arch35 NZ layout the Cube's LoadData2D consumes (NzOffset in turboquant_launch.hpp).
  const auto nz_offset = [](uint64_t row, uint64_t column, uint64_t rows) {
    return (column / kC0) * rows * kC0 + row * kC0 + column % kC0;
  };

  for (const uint64_t kv_heads : {1u, 2u, 8u}) {
    for (const uint64_t head_size : {64u, 128u, 256u}) {
      for (const uint64_t block_size : {64u, 128u}) {
        const uint64_t packed = head_size / 2;
        const uint64_t groups = packed / kC0;
        const uint64_t slots = kBlocks * block_size;
        std::vector<uint32_t> hits(slots * kv_heads * packed, 0);
        size_t window_mismatches = 0;
        size_t nz_mismatches = 0;
        size_t writer_mismatches = 0;
        for (uint64_t slot = 0; slot < slots; ++slot) {
          const uint64_t tile_row = slot % kTile;
          // What ReadPlane calls row: physical * block_size + rowBase, a whole number of tiles.
          const uint64_t row = slot - tile_row;
          // WriteNzTiled's base address; its burst i lands kTile blocks after burst i - 1.
          const uint64_t writer_base = row * kv_heads * packed + tile_row * kC0;
          for (uint64_t kv = 0; kv < kv_heads; ++kv) {
            const uint64_t window = (row * kv_heads + kv * kTile) * packed;
            for (uint64_t column = 0; column < packed; ++column) {
              const uint64_t at = tqt::NzTiledPackedByte(slot, kv, column, kv_heads, packed);
              ASSERT_LT(at, hits.size());
              ++hits[at];
              if (at < window || at - window >= kTile * packed) {
                ++window_mismatches;
                continue;
              }
              // The low nibble is coordinate `column`, the high nibble coordinate `packed + column`, and
              // UnpackPlaneToL1 writes the high plane one tile plane after the low one.
              const uint64_t in_window = at - window;
              nz_mismatches += (in_window != nz_offset(tile_row, column, kTile) ||
                                kTile * packed + in_window != nz_offset(tile_row, packed + column, kTile))
                                   ? 1u
                                   : 0u;
              const uint64_t burst = kv * groups + column / kC0;
              writer_mismatches += (writer_base + burst * kTile * kC0 + column % kC0 != at) ? 1u : 0u;
            }
          }
        }
        const size_t uncovered =
            static_cast<size_t>(std::count_if(hits.begin(), hits.end(), [](uint32_t n) { return n != 1; }));
        const std::string shape = "kv_heads=" + std::to_string(kv_heads) + " head_size=" + std::to_string(head_size) +
                                  " block_size=" + std::to_string(block_size);
        EXPECT_EQ(uncovered, 0u) << shape << ": the layout is not a bijection onto the block";
        EXPECT_EQ(window_mismatches, 0u) << shape << ": a cell lies outside its (tile, kv head) read window";
        EXPECT_EQ(nz_mismatches, 0u) << shape << ": the window does not unpack in NZ order";
        EXPECT_EQ(writer_mismatches, 0u) << shape << ": the writer's strided burst misses its cells";
      }
    }
  }
}

TEST(TurboQuantMirroredCache, PackedBytesAndFp8OperandsDescribeTheSameLevels) {
  namespace tqh = turboquant_host;
  // Two tiles per block and two column groups per head, with the context crossing a tile and a block.
  constexpr int64_t kContext = 150;
  constexpr int64_t kBlock = 128;
  constexpr int64_t kKvHeads = 2;
  constexpr int64_t kHead = 128;
  constexpr int64_t kBlocks = 4;
  constexpr int kSignedNibbleRange = 16;
  constexpr int kSignedNibbleMax = 7;
  const std::vector<int32_t> table = {2, 0};

  for (int32_t index = 0; index < tqh::kMirroredLevels; ++index) {
    const float level = tqh::MirroredLevel(index);
    ASSERT_EQ(tqh::Fp8E4m3fnValue(tqh::Fp8E4m3fnBits(level)), level);
  }

  DeterministicRandom rng(0x4D49u);
  const tqh::MirroredKvCache cache = tqh::BuildMirroredKvCache(rng, kContext, kBlocks, kBlock, kKvHeads, kHead, table);
  const size_t half = static_cast<size_t>(kHead / 2);
  const size_t head = static_cast<size_t>(kHead);
  const size_t kv_heads = static_cast<size_t>(kKvHeads);
  const size_t slot_floats = static_cast<size_t>(tqh::MirroredScaleSlotFloats(kKvHeads));
  const size_t slots = static_cast<size_t>(kBlocks * kBlock);
  ASSERT_EQ(cache.key_packed.size(), slots * kv_heads * half);
  ASSERT_EQ(cache.key_operands.size(), slots * kv_heads * head);
  ASSERT_EQ(cache.scales.size(), slots * slot_floats);

  std::vector<bool> written(slots, false);
  size_t mismatches = 0;
  for (int64_t t = 0; t < kContext; ++t) {
    const size_t slot = static_cast<size_t>(table[static_cast<size_t>(t / kBlock)] * kBlock + t % kBlock);
    written[slot] = true;
    for (size_t kv = 0; kv < kv_heads; ++kv) {
      for (int plane = 0; plane < 2; ++plane) {
        const std::vector<int8_t>& packed = plane == 0 ? cache.key_packed : cache.value_packed;
        const std::vector<int8_t>& operands = plane == 0 ? cache.key_operands : cache.value_operands;
        const std::vector<float>& dense = plane == 0 ? cache.key : cache.value;
        const float scale = cache.scales[slot * slot_floats + (plane == 0 ? kv : kv_heads + kv)];
        ASSERT_GT(scale, 0.0f);
        const size_t row = slot * kv_heads + kv;
        const size_t dense_row = (static_cast<size_t>(t) * kv_heads + kv) * head;
        for (size_t j = 0; j < half; ++j) {
          const uint32_t byte = static_cast<uint8_t>(packed[tqh::MirroredPackedByte(slot, kv, j, kv_heads, half)]);
          const int low = static_cast<int>(byte & tqh::kMirroredNibbleMask);
          const int high = static_cast<int>(byte >> tqh::kMirroredNibbleBits);
          const int signed_nibble = low > kSignedNibbleMax ? low - kSignedNibbleRange : low;
          const float level =
              static_cast<float>(signed_nibble) + (tqh::kMirroredNibbleSignShift - tqh::kMirroredAffineBias);
          const float first = tqh::Fp8E4m3fnValue(static_cast<uint8_t>(operands[row * head + j]));
          const float second = tqh::Fp8E4m3fnValue(static_cast<uint8_t>(operands[row * head + half + j]));
          const float expected = level * scale / tqh::kMirroredGain;
          mismatches += (low != high || first != level || second != level || dense[dense_row + j] != expected ||
                         dense[dense_row + half + j] != expected)
                            ? 1u
                            : 0u;
        }
      }
    }
  }
  EXPECT_EQ(mismatches, 0u);

  const float zero_byte_level = tqh::MirroredLevel(tqh::kMirroredNibbleSignShift);
  size_t pad_mismatches = 0;
  for (size_t slot = 0; slot < slots; ++slot) {
    if (written[slot]) {
      continue;
    }
    for (size_t kv = 0; kv < kv_heads; ++kv) {
      for (size_t j = 0; j < half; ++j) {
        const size_t at = tqh::MirroredPackedByte(slot, kv, j, kv_heads, half);
        pad_mismatches += (cache.key_packed[at] != 0 || cache.value_packed[at] != 0) ? 1u : 0u;
      }
    }
    for (size_t e = 0; e < kv_heads * head; ++e) {
      const size_t at = slot * kv_heads * head + e;
      pad_mismatches += (tqh::Fp8E4m3fnValue(static_cast<uint8_t>(cache.key_operands[at])) != zero_byte_level ||
                         tqh::Fp8E4m3fnValue(static_cast<uint8_t>(cache.value_operands[at])) != zero_byte_level)
                            ? 1u
                            : 0u;
    }
  }
  EXPECT_EQ(pad_mismatches, 0u) << "an unwritten slot's operands must be the expansion of a zero packed byte";
}

TEST(TurboQuantEdgeCases, InvolutionOnCanonicalBasisVectors) {
  for (int d : {64, 128, 256}) {
    const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(d);
    const int indices[] = {0, 1, 7, 8, d - 1};

    double worst_norm = 0.0;
    double worst_involution = 0.0;
    for (int index : indices) {
      std::vector<float> basis(static_cast<size_t>(d), 0.0f);
      basis[static_cast<size_t>(index)] = 1.0f;

      std::vector<float> rotated = basis;
      tq::cpu_apply_pi(rotated.data(), d, signs.data());

      double norm_sq = 0.0;
      for (int i = 0; i < d; ++i) {
        norm_sq += static_cast<double>(rotated[static_cast<size_t>(i)]) * rotated[static_cast<size_t>(i)];
      }
      worst_norm = std::max(worst_norm, std::fabs(std::sqrt(norm_sq) - 1.0));
      const float expected_magnitude = 1.0f / std::sqrt(static_cast<float>(d));
      for (int i = 0; i < d; ++i) {
        EXPECT_NEAR(std::fabs(rotated[static_cast<size_t>(i)]), expected_magnitude, 1e-6f)
            << "d=" << d << " e_" << index << " channel " << i;
      }

      tq::cpu_apply_pi(rotated.data(), d, signs.data());
      for (int i = 0; i < d; ++i) {
        worst_involution = std::max(
            worst_involution, std::fabs(static_cast<double>(rotated[static_cast<size_t>(i)]) -
                                        basis[static_cast<size_t>(i)]));
      }
    }
    EXPECT_LT(worst_norm, 1e-6) << "||Pi e_i|| drifted from 1, d=" << d;
    EXPECT_LT(worst_involution, 1e-6) << "Pi(Pi(e_i)) drifted from e_i, d=" << d;
    std::printf("[turboquant] d=%3d canonical basis: max |norm-1| = %.3e, max |Pi^2 e_i - e_i| = %.3e\n", d,
                worst_norm, worst_involution);
  }
}

class TurboQuantGoldenFidelity : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    loaded_ = LoadGoldenLayer3(&golden_, &error_);
  }

  static GoldenLayer3 golden_;
  static std::string error_;
  static bool loaded_;
};

GoldenLayer3 TurboQuantGoldenFidelity::golden_;
std::string TurboQuantGoldenFidelity::error_;
bool TurboQuantGoldenFidelity::loaded_ = false;

TEST_F(TurboQuantGoldenFidelity, Qwen35Layer3DecodeReport) {
  if (!loaded_) {
    GTEST_SKIP() << "golden dump unavailable: " << error_;
  }

  const std::vector<float>& rope_q = golden_.tap_rope_q;
  const std::vector<float>& rope_k = golden_.tap_rope_k;
  const std::vector<float> value(golden_.tap_qkv.begin() + static_cast<ptrdiff_t>(s::kQDim + s::kKvDim),
                                 golden_.tap_qkv.begin() + static_cast<ptrdiff_t>(s::kQDim + 2 * s::kKvDim));

  ASSERT_EQ(rope_q.size(), static_cast<size_t>(s::kTokens * s::kQDim));
  ASSERT_EQ(rope_k.size(), static_cast<size_t>(s::kTokens * s::kKvDim));
  ASSERT_EQ(value.size(), static_cast<size_t>(s::kKvDim));

  const int d = static_cast<int>(s::kHeadDim);
  const int num_heads = static_cast<int>(s::kNumHeads);
  const int num_kv_heads = static_cast<int>(s::kNumKvHeads);
  const int block_size = static_cast<int>(s::kBlockSize);
  const int context_len = static_cast<int>(s::kContextLen);

  std::vector<float> exact_context;
  ExactPagedAttention(rope_q, rope_k, value, context_len, num_heads, num_kv_heads, d, s::kAttentionScale,
                      &exact_context);

  std::vector<float> tq_context;
  RunTurboQuantDecode(rope_q, rope_k, value, context_len, num_heads, num_kv_heads, d, block_size,
                      s::kAttentionScale, &tq_context);

  std::vector<float> norm1;
  std::vector<float> rstd;
  reference::RmsNorm(golden_.input_x, golden_.input_norm_gamma, s::kTokens, s::kHidden, s::kRmsNormEps, &norm1,
                     &rstd);
  norm1 = RoundToHalf(norm1);
  const std::vector<float> attn_gate =
      ProjectOnCpu(norm1, golden_.w_gate_attn, s::kTokens, s::kHidden, s::kQDim);

  const std::vector<float> exact_layer = LayerTailFromContext(golden_, exact_context, attn_gate);
  const std::vector<float> tq_layer = LayerTailFromContext(golden_, tq_context, attn_gate);

  std::printf("\n[turboquant] Qwen3.5 layer-3 fidelity, 4-bit rotated KV cache\n");
  std::printf("  head_dim=%d heads=%d kv_heads=%d context_len=%d block_size=%d\n", d, num_heads, num_kv_heads,
              context_len, block_size);
  PrintMetrics("attention context vs exact fp32", tq::cpu_fidelity(tq_context, exact_context));
  PrintMetrics("layer output vs exact fp32 path", tq::cpu_fidelity(tq_layer, exact_layer));
  PrintMetrics("layer output vs golden fp16 dump", tq::cpu_fidelity(tq_layer, golden_.golden_output));
  PrintMetrics("exact path vs golden fp16 dump", tq::cpu_fidelity(exact_layer, golden_.golden_output));
  std::printf(
      "  note: this dump is a single decode step with context_len=1, so the\n"
      "        softmax is over one score and this measures codec fidelity on\n"
      "        real activations, not the online-softmax accumulation.\n");

  SUCCEED();
}

TEST(TurboQuantSyntheticFidelity, LongContextDecodeReport) {
  constexpr int d = 128;
  constexpr int num_heads = 8;
  constexpr int num_kv_heads = 2;
  constexpr int block_size = 128;
  constexpr int context_len = 300;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));

  uint32_t state = 12345u;
  auto next = [&state]() {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(static_cast<int32_t>(state >> 8) % 20000 - 10000) / 10000.0f;
  };

  std::vector<float> query(static_cast<size_t>(num_heads) * d);
  for (auto& value : query) {
    value = next() / std::sqrt(static_cast<float>(d));
  }
  std::vector<float> key(static_cast<size_t>(context_len) * num_kv_heads * d);
  std::vector<float> value_cache(key.size());
  for (auto& value : key) {
    value = next();
  }
  for (auto& value : value_cache) {
    value = next();
  }
  for (int pos = 0; pos < context_len; ++pos) {
    for (int h = 0; h < num_kv_heads; ++h) {
      key[(static_cast<size_t>(pos) * num_kv_heads + h) * d] *= 30.0f;
    }
  }

  std::vector<float> exact;
  ExactPagedAttention(query, key, value_cache, context_len, num_heads, num_kv_heads, d, scale, &exact);
  std::vector<float> approx;
  RunTurboQuantDecode(query, key, value_cache, context_len, num_heads, num_kv_heads, d, block_size, scale, &approx);

  std::printf("\n[turboquant] synthetic long-context fidelity\n");
  std::printf("  head_dim=%d heads=%d kv_heads=%d context_len=%d block_size=%d (outlier channel in K)\n", d,
              num_heads, num_kv_heads, context_len, block_size);
  PrintMetrics("attention output vs exact fp32", tq::cpu_fidelity(approx, exact));

  SUCCEED();
}

}
}
}
