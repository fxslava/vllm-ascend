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

// TIER 1 (host) - the DSV4 routed expert's arithmetic, with no CANN, no
// toolkit, no NPU.
//
// What this tier can settle, and the simulator and device tiers therefore need
// not re-litigate:
//   * the FP4 / E8M0 / bf16 codecs, exhaustively rather than by sampling;
//   * the reference expert against an independent fp64 oracle;
//   * the SwiGLU boundary behaviour the kernel relies on, including the inf
//     path that a clamp would remove;
//   * the geometry rules the host tiling function enforces.
//
// What it cannot settle, and must not be read as settling: anything about pipe
// synchronisation. See test_sim_950pr_dsv4_moe_expert.cpp.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "dsv4_moe_expert_cpu.h"

namespace {

using vllm_ascend::test::dsv4::Bf16BitsToFloat;
using vllm_ascend::test::dsv4::Bf16UlpDistance;
using vllm_ascend::test::dsv4::E2m1Table;
using vllm_ascend::test::dsv4::E8m0ToScale;
using vllm_ascend::test::dsv4::ExpertInputs;
using vllm_ascend::test::dsv4::FloatToBf16Bits;
using vllm_ascend::test::dsv4::GeometryIsAccepted;
using vllm_ascend::test::dsv4::kFp4Block;
using vllm_ascend::test::dsv4::kFp4PerByte;
using vllm_ascend::test::dsv4::ReferenceExpert;
using vllm_ascend::test::dsv4::SwiGluElement;
using vllm_ascend::test::dsv4::UnpackFp4;

constexpr int64_t kMaxUlp = 2;  // the Gate B criterion, in bf16 ULPs

// ---------------------------------------------------------------------------
// FP4 / E2M1 packing
// ---------------------------------------------------------------------------

TEST(Dsv4Codec, E2m1TableIsTheOcpGrid) {
  const float* t = E2m1Table();
  const float expected[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(t[i], expected[i]) << "code " << i;
    EXPECT_FLOAT_EQ(t[i + 8], -expected[i]) << "code " << (i + 8);
  }
  // Code 8 is negative zero, not a distinct magnitude.
  EXPECT_TRUE(std::signbit(t[8]));
  EXPECT_EQ(t[8], 0.0f);
}

TEST(Dsv4Codec, NibbleOrderIsLowThenHigh) {
  // Byte i holds logical element 2i in the LOW nibble, 2i+1 in the HIGH.
  // Exhaustive over all 256 byte values and both positions.
  for (int value = 0; value < 256; ++value) {
    const uint8_t byte = static_cast<uint8_t>(value);
    const float even = UnpackFp4(&byte, 0);
    const float odd = UnpackFp4(&byte, 1);
    EXPECT_FLOAT_EQ(even, E2m1Table()[byte & 0x0Fu]) << "byte " << value;
    EXPECT_FLOAT_EQ(odd, E2m1Table()[byte >> 4]) << "byte " << value;
  }
}

// ---------------------------------------------------------------------------
// E8M0 block scales
// ---------------------------------------------------------------------------

TEST(Dsv4Codec, E8m0IsAnExactPowerOfTwo) {
  // Every representable exponent except the two specials must be exactly
  // 2^(b-127) -- exactly, not approximately, because the kernel folds it into
  // a product and any error is multiplicative across the whole block.
  for (uint32_t b = 1; b <= 254; ++b) {
    const float s = E8m0ToScale(b);
    EXPECT_EQ(s, std::ldexp(1.0f, static_cast<int>(b) - 127)) << "byte " << b;
  }
}

TEST(Dsv4Codec, E8m0SpecialsFollowOcpMx) {
  // b = 0 is 2^-127, which is subnormal in fp32: the exponent field alone
  // cannot express it, so a naive `b << 23` yields zero instead.
  const float smallest = E8m0ToScale(0);
  EXPECT_GT(smallest, 0.0f);
  EXPECT_EQ(smallest, std::ldexp(1.0f, -127));
  EXPECT_FALSE(std::isnormal(smallest));

  // b = 0xFF is NaN.
  EXPECT_TRUE(std::isnan(E8m0ToScale(0xFF)));

  // b = 127 is unity, the identity scale.
  EXPECT_EQ(E8m0ToScale(127), 1.0f);
}

// ---------------------------------------------------------------------------
// bf16 conversion
// ---------------------------------------------------------------------------

TEST(Dsv4Codec, Bf16WideningIsExactAndRoundTrips) {
  // Every bf16 bit pattern widens exactly and narrows back to itself.
  for (uint32_t bits = 0; bits <= 0xFFFFu; ++bits) {
    const uint16_t b = static_cast<uint16_t>(bits);
    const float widened = Bf16BitsToFloat(b);
    if (std::isnan(widened)) continue;  // NaN payloads are not required to survive
    EXPECT_EQ(FloatToBf16Bits(widened), b) << "bits 0x" << std::hex << bits;
  }
}

TEST(Dsv4Codec, Bf16RoundsHalfToEven) {
  // Exactly halfway between two bf16 values: 1.0 + 2^-9 sits on the tie, and
  // round-to-nearest-even must pick the even mantissa (1.0), not round up.
  const float tie_down = 1.0f + std::ldexp(1.0f, -9);
  EXPECT_EQ(FloatToBf16Bits(tie_down), FloatToBf16Bits(1.0f));

  // The next representable bf16 up is 1 + 2^-8; a tie from there rounds up to
  // the even neighbour 1 + 2^-7.
  const float tie_up = (1.0f + std::ldexp(1.0f, -8)) + std::ldexp(1.0f, -9);
  EXPECT_EQ(FloatToBf16Bits(tie_up), FloatToBf16Bits(1.0f + std::ldexp(1.0f, -7)));
}

TEST(Dsv4Codec, Bf16SpecialsTruncate) {
  EXPECT_EQ(FloatToBf16Bits(std::numeric_limits<float>::infinity()), 0x7F80u);
  EXPECT_EQ(FloatToBf16Bits(-std::numeric_limits<float>::infinity()), 0xFF80u);
  EXPECT_TRUE((FloatToBf16Bits(std::numeric_limits<float>::quiet_NaN()) & 0x7F80u) == 0x7F80u);
  EXPECT_EQ(FloatToBf16Bits(0.0f), 0x0000u);
  EXPECT_EQ(FloatToBf16Bits(-0.0f), 0x8000u);
}

// ---------------------------------------------------------------------------
// SwiGLU boundary behaviour
// ---------------------------------------------------------------------------

TEST(Dsv4SwiGlu, LargeNegativeGateProducesZeroThroughInf) {
  // This is the path the camodel flags as `check_fp_status instr input data
  // inf`: exp(-gate) overflows, and the result is correct only because
  // 1/(1+inf) is exactly 0. The kernel currently relies on that.
  const float gate = -100.0f;  // exp(100) overflows fp32 (threshold ~88.7)
  EXPECT_TRUE(std::isinf(std::exp(-gate)));
  const float activated = SwiGluElement(gate, 2.0f);
  EXPECT_EQ(activated, 0.0f) << "silu saturates to zero on the negative tail";
  EXPECT_FALSE(std::isnan(activated));
}

TEST(Dsv4SwiGlu, ClampIsNotBitNeutralAndThatIsTheCostOfAdoptingIt) {
  // Measured, and it contradicts the intuition that a clamp is free: on the
  // far negative tail the unclamped path returns exactly zero (sigmoid
  // underflows to 0 through inf), while a clamp at 80 returns a tiny non-zero.
  // Both are negligible in magnitude, but they are DIFFERENT bf16 values.
  //
  // The consequence is concrete: adopting the clamp requires re-baselining
  // every golden that contains a saturated element. It is not a drop-in.
  bool observed_difference = false;
  for (float gate : {-100.0f, -120.0f, -1e6f}) {
    for (float up : {1.0f, -3.5f, 0.25f}) {
      const float unclamped = SwiGluElement(gate, up, 0.0f);
      const float clamped = SwiGluElement(gate, up, 80.0f);

      // The unclamped path goes through inf; the clamped one does not.
      EXPECT_TRUE(std::isinf(std::exp(-gate))) << "gate=" << gate;
      EXPECT_FALSE(std::isinf(std::exp(80.0f)));

      EXPECT_EQ(unclamped, 0.0f) << "gate=" << gate << " up=" << up;
      EXPECT_NEAR(clamped, 0.0f, 1e-30f) << "gate=" << gate << " up=" << up;
      if (FloatToBf16Bits(unclamped) != FloatToBf16Bits(clamped)) {
        observed_difference = true;
      }
    }
  }
  EXPECT_TRUE(observed_difference)
      << "if this ever stops being true the clamp became bit-neutral and the "
         "re-baselining caveat can be dropped";
}

TEST(Dsv4SwiGlu, LargePositiveGateApproachesGateTimesUp) {
  const float gate = 100.0f;
  const float up = 3.0f;
  EXPECT_FLOAT_EQ(SwiGluElement(gate, up), gate * up);
}

TEST(Dsv4SwiGlu, ZeroesPropagate) {
  EXPECT_EQ(SwiGluElement(0.0f, 5.0f), 0.0f);
  EXPECT_EQ(SwiGluElement(5.0f, 0.0f), 0.0f);
  // silu(0) == 0 exactly, so a zero gate kills the product regardless of up.
  EXPECT_EQ(SwiGluElement(0.0f, -5.0f), 0.0f);
}

TEST(Dsv4SwiGlu, NanPropagatesRatherThanBeingSwallowed) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(std::isnan(SwiGluElement(nan, 1.0f)));
  EXPECT_TRUE(std::isnan(SwiGluElement(1.0f, nan)));
  // A clamp built from min/max must not turn a NaN gate into a finite value.
  EXPECT_TRUE(std::isnan(SwiGluElement(nan, 1.0f, 80.0f)))
      << "the clamp swallowed a NaN; min/max operand order is wrong";
}

// ---------------------------------------------------------------------------
// The reference expert against an independent fp64 oracle
// ---------------------------------------------------------------------------

namespace {

struct Problem {
  int64_t hidden;
  int64_t inter;
  std::vector<uint16_t> x;
  std::vector<uint8_t> w1, w2, w3, w1s, w2s, w3s;

  ExpertInputs View() const {
    return ExpertInputs{x.data(), w1.data(), w2.data(), w3.data(),
                        w1s.data(), w2s.data(), w3s.data()};
  }
};

Problem MakeProblem(int64_t hidden, int64_t inter, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  // A healthy 2^-6 .. 2^+6 scale window: the specials are covered exhaustively
  // by the codec tests above, and including them here would let a genuine
  // arithmetic regression hide behind a NaN comparison.
  std::uniform_int_distribution<int> scale_dist(127 - 6, 127 + 6);
  std::normal_distribution<float> x_dist(0.0f, 1.0f);

  Problem p;
  p.hidden = hidden;
  p.inter = inter;
  p.x.resize(static_cast<size_t>(hidden));
  for (auto& v : p.x) v = FloatToBf16Bits(x_dist(rng));

  auto fill_bytes = [&](std::vector<uint8_t>& v, size_t n, bool is_scale) {
    v.resize(n);
    for (auto& b : v) {
      b = static_cast<uint8_t>(is_scale ? scale_dist(rng) : byte_dist(rng));
    }
  };
  fill_bytes(p.w1, static_cast<size_t>(inter * hidden / kFp4PerByte), false);
  fill_bytes(p.w3, static_cast<size_t>(inter * hidden / kFp4PerByte), false);
  fill_bytes(p.w2, static_cast<size_t>(hidden * inter / kFp4PerByte), false);
  fill_bytes(p.w1s, static_cast<size_t>(inter * hidden / kFp4Block), true);
  fill_bytes(p.w3s, static_cast<size_t>(inter * hidden / kFp4Block), true);
  fill_bytes(p.w2s, static_cast<size_t>(hidden * inter / kFp4Block), true);
  return p;
}

// Independent oracle: the same mathematics in fp64, written separately from
// the fp32 reference so a shared mistake cannot cancel out.
std::vector<uint16_t> Fp64DownOut(const Problem& p) {
  const int64_t hidden = p.hidden, inter = p.inter;
  std::vector<double> x(static_cast<size_t>(hidden));
  for (int64_t c = 0; c < hidden; ++c) x[static_cast<size_t>(c)] = Bf16BitsToFloat(p.x[c]);

  auto project = [&](const std::vector<double>& in, const std::vector<uint8_t>& w,
                     const std::vector<uint8_t>& s, int64_t rows, int64_t cols) {
    std::vector<double> out(static_cast<size_t>(rows), 0.0);
    for (int64_t r = 0; r < rows; ++r) {
      double acc = 0.0;
      for (int64_t c = 0; c < cols; ++c) {
        const uint8_t byte = w[static_cast<size_t>(r * (cols / 2) + (c >> 1))];
        const uint32_t nib = (c & 1) ? (byte >> 4) : (byte & 0x0Fu);
        const uint8_t sb = s[static_cast<size_t>(r * (cols / kFp4Block) + c / kFp4Block)];
        acc += x.empty() ? 0.0
                         : in[static_cast<size_t>(c)] *
                               (static_cast<double>(E2m1Table()[nib]) *
                                std::ldexp(1.0, static_cast<int>(sb) - 127));
      }
      out[static_cast<size_t>(r)] = acc;
    }
    return out;
  };

  const std::vector<double> gate = project(x, p.w1, p.w1s, inter, hidden);
  const std::vector<double> up = project(x, p.w3, p.w3s, inter, hidden);
  std::vector<double> activated(static_cast<size_t>(inter));
  for (int64_t j = 0; j < inter; ++j) {
    const double g = gate[static_cast<size_t>(j)];
    const double sig = 1.0 / (1.0 + std::exp(-g));
    activated[static_cast<size_t>(j)] = (g * sig) * up[static_cast<size_t>(j)];
  }
  const std::vector<double> down = project(activated, p.w2, p.w2s, hidden, inter);

  std::vector<uint16_t> out(static_cast<size_t>(hidden));
  for (int64_t r = 0; r < hidden; ++r) {
    out[static_cast<size_t>(r)] = FloatToBf16Bits(static_cast<float>(down[static_cast<size_t>(r)]));
  }
  return out;
}

int64_t MaxUlp(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  int64_t worst = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, Bf16UlpDistance(a[i], b[i]));
  }
  return worst;
}

}  // namespace

TEST(Dsv4Reference, MatchesFp64OracleWithinTwoUlp) {
  for (uint32_t seed = 0; seed < 4; ++seed) {
    const Problem p = MakeProblem(256, 128, seed);
    const auto got = ReferenceExpert(p.View(), p.hidden, p.inter);
    const auto oracle = Fp64DownOut(p);
    const int64_t ulp = MaxUlp(got.down_out, oracle);
    EXPECT_LE(ulp, kMaxUlp) << "seed " << seed << ": FpDiff " << ulp << " ULP";
  }
}

TEST(Dsv4Reference, IsDeterministic) {
  const Problem p = MakeProblem(256, 128, 7);
  const auto a = ReferenceExpert(p.View(), p.hidden, p.inter);
  const auto b = ReferenceExpert(p.View(), p.hidden, p.inter);
  EXPECT_EQ(a.gate_out, b.gate_out);
  EXPECT_EQ(a.up_out, b.up_out);
  EXPECT_EQ(a.activated, b.activated);
  EXPECT_EQ(a.down_out, b.down_out);
}

TEST(Dsv4Reference, ActivatedIsSwiGluOfTheFp32GateAndUp) {
  // The three [1, inter] outputs are not independent: activated is the SwiGLU
  // of the other two. The check has to run on the fp32 intermediates, not the
  // bf16 outputs -- see the next test for why.
  const Problem p = MakeProblem(128, 64, 3);
  const auto out = ReferenceExpert(p.View(), p.hidden, p.inter);
  for (int64_t j = 0; j < p.inter; ++j) {
    const size_t i = static_cast<size_t>(j);
    const float expected = SwiGluElement(out.gate_f32[i], out.up_f32[i]);
    EXPECT_EQ(out.activated_f32[i], expected) << "element " << j;
    EXPECT_EQ(out.activated[i], FloatToBf16Bits(expected)) << "element " << j;
  }
}

TEST(Dsv4Reference, ActivatedCannotBeReconstructedFromTheBf16Outputs) {
  // Measured: recomputing the activation from the ROUNDED gate/up diverges by
  // tens of bf16 ULPs, because silu's relative sensitivity to its gate input
  // grows like |gate| on the negative tail -- a 0.4% perturbation of gate
  // (one bf16 step) moves exp(-gate) by a double-digit percentage.
  //
  // This is why a consumer of these three outputs must not cross-validate them
  // against each other, and why the kernel writes `activated` rather than
  // leaving the caller to derive it.
  const Problem p = MakeProblem(128, 64, 3);
  const auto out = ReferenceExpert(p.View(), p.hidden, p.inter);
  int64_t worst = 0;
  for (int64_t j = 0; j < p.inter; ++j) {
    const size_t i = static_cast<size_t>(j);
    const float g = Bf16BitsToFloat(out.gate_out[i]);
    const float u = Bf16BitsToFloat(out.up_out[i]);
    worst = std::max(worst, Bf16UlpDistance(out.activated[i], FloatToBf16Bits(SwiGluElement(g, u))));
  }
  EXPECT_GT(worst, kMaxUlp)
      << "reconstruction from rounded inputs agreed to " << worst
      << " ULP; if this holds generally the caveat above can be relaxed";
}

// ---------------------------------------------------------------------------
// Tiling geometry rules
// ---------------------------------------------------------------------------

TEST(Dsv4Tiling, AcceptsTheBringUpAndRejectsProduction) {
  EXPECT_TRUE(GeometryIsAccepted(256, 128)) << "the reduced bring-up geometry";
  EXPECT_TRUE(GeometryIsAccepted(512, 256));
  EXPECT_TRUE(GeometryIsAccepted(64, 64));

  // Production DeepSeek-V4 needs 4 MiB of packed weight per projection against
  // a 96 KiB single-load budget. Tiling must refuse it rather than silently
  // overflow UB; this is the assertion that the milestone's scope is enforced.
  EXPECT_FALSE(GeometryIsAccepted(4096, 2048));
}

TEST(Dsv4Tiling, RequiresMultiplesOfSixtyFour) {
  // Row packing needs an even column count and the block scale needs 32, so
  // the reduction dimension must be a multiple of 64.
  EXPECT_FALSE(GeometryIsAccepted(32, 64));
  EXPECT_FALSE(GeometryIsAccepted(64, 32));
  EXPECT_FALSE(GeometryIsAccepted(96, 64));
  EXPECT_FALSE(GeometryIsAccepted(0, 64));
}

TEST(Dsv4Tiling, InterGreaterThanHiddenIsAcceptedAndMustNotOverflow) {
  // The tiling function does not compare inter against hidden, so this passes
  // every geometry check. The kernel aliases its fp32 scratch between a
  // hidden-sized and an inter-sized use, so the buffer has to be sized for
  // max(hidden, inter) -- this case is what makes that a requirement rather
  // than a nicety.
  ASSERT_TRUE(GeometryIsAccepted(64, 128));
  const Problem p = MakeProblem(64, 128, 11);
  const auto out = ReferenceExpert(p.View(), p.hidden, p.inter);
  EXPECT_EQ(out.activated.size(), static_cast<size_t>(128));
  EXPECT_EQ(out.down_out.size(), static_cast<size_t>(64));
}

}  // namespace
