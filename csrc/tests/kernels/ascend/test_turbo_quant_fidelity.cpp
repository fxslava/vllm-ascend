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

// End-to-end fidelity report for the TurboQuant 4-bit KV cache, run entirely on
// the host against the Qwen3.5 layer-3 golden dump.
//
// This binary links neither libascendcl.so nor anything else from the CANN
// runtime: the write path, the rotated-basis decode and the un-rotation are all
// the CPU reference in ../../reference/turbo_quant_cpu.h, which mirrors
// turboquant_codec_950.h instruction for instruction. That is the point - the
// codec's accuracy on real activations can be measured on a build machine, in a
// simulator, or in CI, long before a 950PR is available.
//
// Assertion policy: this is an *analytical reporter*. It prints cosine
// similarity, SNR and relative L2 error and never asserts on any of them, so a
// change in quantiser behaviour shows up as a number moving rather than as a
// red build. The only EXPECTs here are on structural invariants that are exact
// integer or involution identities and cannot drift with the data.
//
// What the golden dump can and cannot show: it is a single decode step with
// kContextLen == 1, so the softmax is over one score and the attention context
// is V. The golden case therefore measures the codec's fidelity on real Qwen
// activations, but not the online-softmax accumulation. The synthetic case
// below covers a full 128-position context for that reason, and says so.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "ascend950_shapes.hpp"
#include "cpu_reference.hpp"
#include "fp16.hpp"
#include "golden_layer3.hpp"
#include "turbo_quant_cpu.h"

namespace vllm_ascend {
namespace test {
namespace {

namespace s = shapes950;
namespace tq = turboquant_ref;

// The golden dump is fp16 on the wire; every stage boundary rounds back to it
// so the reference tracks what a kernel actually stores.
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

// Exact fp32 paged attention over the same cache geometry, with K and V held at
// full precision. Written in the same shape as
// cpu_paged_attention_turboquant so the two differ only in the codec, not in
// how the softmax is arranged.
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

// Writes every context position of K and V through the TurboQuant write path,
// then reads them back with the rotated-basis decode.
void RunTurboQuantDecode(const std::vector<float>& query, const std::vector<float>& key,
                         const std::vector<float>& value, int context_len, int num_heads, int num_kv_heads, int d,
                         int block_size, float scale, std::vector<float>* out) {
  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(d);
  const int num_blocks = (context_len + block_size - 1) / block_size;
  const size_t packed_stride = static_cast<size_t>(d / tq::kPackFactor);

  std::vector<int8_t> key_cache(static_cast<size_t>(num_blocks) * block_size * num_kv_heads * packed_stride, 0);
  std::vector<int8_t> value_cache(key_cache.size(), 0);
  std::vector<float> key_scale(static_cast<size_t>(num_blocks) * num_kv_heads * block_size, 0.0f);
  std::vector<float> value_scale(key_scale.size(), 0.0f);

  for (int pos = 0; pos < context_len; ++pos) {
    for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
      const size_t base = (static_cast<size_t>(pos) * num_kv_heads + kv_head) * d;
      tq::cpu_reshape_and_cache_one(key.data() + base, d, signs.data(), pos, block_size, num_kv_heads, kv_head,
                                    key_cache.data(), key_scale.data());
      tq::cpu_reshape_and_cache_one(value.data() + base, d, signs.data(), pos, block_size, num_kv_heads, kv_head,
                                    value_cache.data(), value_scale.data());
    }
  }

  std::vector<int32_t> block_table(static_cast<size_t>(num_blocks));
  for (int b = 0; b < num_blocks; ++b) {
    block_table[static_cast<size_t>(b)] = b;
  }

  out->assign(static_cast<size_t>(num_heads) * d, 0.0f);
  tq::cpu_paged_attention_turboquant(query.data(), key_cache.data(), value_cache.data(), key_scale.data(),
                                     value_scale.data(), block_table.data(), context_len, num_heads, num_kv_heads, d,
                                     block_size, scale, signs.data(), out->data());
}

// Stages 6 to 9 of the layer, taking an attention context and producing the
// layer output. Shared so the exact and quantised contexts travel the identical
// downstream and the metric isolates the codec.
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

// -----------------------------------------------------------------------------
// Structural invariants. Exact identities, independent of the activations, so
// these are safe to assert without making the suite flaky.
// -----------------------------------------------------------------------------

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

// The sign vector is a derived constant shared by three implementations: this
// header, vllm_ascend/attention/turboquant_v1.py, and whatever writes a cache
// that another one reads back. Pinning the first few values catches a drift in
// any of them, which would otherwise show up only as silently wrong dequantised
// activations.
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
  // Every one of the 16 codes appears in both nibble positions.
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
    if (out[static_cast<size_t>(i)] + tq::kZeroPoint != levels[static_cast<size_t>(i)]) {
      ++mismatches;
    }
  }
  EXPECT_EQ(mismatches, 0);
}

// -----------------------------------------------------------------------------
// Golden fidelity report
// -----------------------------------------------------------------------------

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

  // Golden activations, taken straight from the dump: post-RoPE Q and K, and V
  // sliced out of tap_qkv, which is laid out [Q | K | V].
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

  // The attention gate is not dumped on its own, so recompute it from norm1 the
  // way the layer does; it is shared by both downstream runs and so cancels out
  // of the comparison between them.
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

// -----------------------------------------------------------------------------
// Synthetic long-context report: covers what the ctx=1 golden dump cannot.
// -----------------------------------------------------------------------------

TEST(TurboQuantSyntheticFidelity, LongContextDecodeReport) {
  constexpr int d = 128;
  constexpr int num_heads = 8;
  constexpr int num_kv_heads = 2;
  constexpr int block_size = 128;
  constexpr int context_len = 300;  // spans three paged blocks, last one partial
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));

  // Deterministic pseudo-random activations; the same LCG the sign vector uses,
  // so the report is reproducible bit for bit on any host.
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
  // One channel per K vector carries an outlier: this is the regime the
  // rotation exists for, and the case plain absmax quantisation handles worst.
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

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
