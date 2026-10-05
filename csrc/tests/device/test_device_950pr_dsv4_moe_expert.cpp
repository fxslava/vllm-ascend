// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "dsv4_device_case.hpp"
#include "test_harness.hpp"

namespace vllm_ascend::test {
namespace {
namespace d = dsv4;

void ExpectParity(const d::DeviceOutputs& got, const d::DeviceOutputs& want) {
  constexpr const char* names[] = {"gate_out", "up_out", "activated_out", "down_out"};
  for (size_t t = 0; t < got.size(); ++t) {
    ASSERT_EQ(got[t].size(), want[t].size());
    int64_t worst = 0;
    size_t violations = 0;
    for (size_t i = 0; i < got[t].size(); ++i) {
      const float a = d::Bf16BitsToFloat(got[t][i]);
      const float b = d::Bf16BitsToFloat(want[t][i]);
      // Specials compare by classification/sign; NaN payloads are not numeric ULPs.
      if (!std::isfinite(a) || !std::isfinite(b)) {
        const bool matches =
            (std::isnan(a) && std::isnan(b)) || (std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b));
        EXPECT_TRUE(matches) << names[t] << " element " << i;
        if (!matches) ++violations;
      } else {
        const int64_t ulp = d::Bf16UlpDistance(got[t][i], want[t][i]);
        worst = std::max(worst, ulp);
        if (ulp > d::kDeviceMaxUlp) ++violations;
      }
    }
    const double rate = static_cast<double>(violations) / got[t].size();
    std::printf("%s FpDiff=%lld BF16_ULP RateDiff=%.8f\n", names[t], static_cast<long long>(worst), rate);
    EXPECT_LE(worst, d::kDeviceMaxUlp) << names[t];
    EXPECT_LE(rate, d::kDeviceMaxRate) << names[t];
  }
}

class Dsv4Production : public ::testing::TestWithParam<int64_t> {
 protected:
  void SetUp() override {
    REQUIRE_ASCEND_950PR();
#ifndef VLLM_ASCEND_DSV4_SIM_SUITE
    if (IsRunningOnSimulator()) GTEST_SKIP() << "physical device suite refuses camodel";
#endif
    ASSERT_EQ(AscendTestEnvironment::Instance().device().device_id(), 0);
  }
  d::Problem Problem(uint32_t seed) const { return d::MakeDeviceProblem(GetParam(), d::kProductionInter, seed); }
};

TEST_P(Dsv4Production, GoldenAndRepeatedBitParity) {
  const auto p = Problem(0xD540);
  const auto want = d::Golden(p);
  d::DeviceExpert expert(p);
  expert.Enqueue();
  const auto first = expert.Read();
  ExpectParity(first, want);
  for (int repeat = 0; repeat < 20; ++repeat) {
    expert.Enqueue();
    EXPECT_EQ(expert.Read(), first) << "repeat " << repeat;
  }
}

TEST_P(Dsv4Production, PositiveAndNegativeGateClamp) {
  for (const uint8_t code : {uint8_t{0x77}, uint8_t{0xFF}}) {
    auto p = Problem(0xD541);
    std::fill(p.x.begin(), p.x.end(), d::FloatToBf16Bits(2.0f));
    std::fill(p.w1.begin(), p.w1.end(), code);
    std::fill(p.w1_scale.begin(), p.w1_scale.end(), 127);
    const auto oracle = d::ReferenceExpert(p.View(), p.hidden, p.inter);
    for (float gate : oracle.gate_f32) ASSERT_GT(std::abs(gate), 100.0f);
    d::DeviceExpert expert(p);
    expert.Enqueue();
    const auto got = expert.Read();
    ExpectParity(got, d::Golden(p));
    for (const auto& tensor : got)
      for (uint16_t bits : tensor) EXPECT_TRUE(std::isfinite(d::Bf16BitsToFloat(bits)));
  }
}

TEST_P(Dsv4Production, SixConcurrentExpertsHavePrivateStorage) {
  std::vector<std::unique_ptr<d::DeviceExpert>> experts;
  std::vector<d::DeviceOutputs> goldens;
  for (uint32_t i = 0; i < 6; ++i) {
    const auto p = Problem(0xD550 + 2 * i);
    goldens.push_back(d::Golden(p));
    experts.push_back(std::make_unique<d::DeviceExpert>(p));
  }
  for (int repeat = 0; repeat < 10; ++repeat) {
    // Submit all streams before waiting on any: no host-side serialization.
    for (auto& expert : experts) expert->Enqueue();
    for (size_t i = 0; i < experts.size(); ++i) ExpectParity(experts[i]->Read(), goldens[i]);
  }
}

TEST_P(Dsv4Production, EveryScaleByteAndAllFp4Codes) {
  auto p = Problem(0xD560);
  std::fill(p.x.begin(), p.x.end(), d::FloatToBf16Bits(1.0f));
  std::fill(p.w1.begin(), p.w1.end(), 0);
  std::fill(p.w1_scale.begin(), p.w1_scale.end(), 127);
  // First block per row sweeps [0,254] and 255 (NaN). Code 7 at scale 254
  // produces arithmetic overflow; E8M0 itself has no dedicated infinity code.
  for (int64_t row = 0; row < p.inter; ++row) {
    p.w1_scale[static_cast<size_t>(row * p.hidden / d::kFp4Block)] = static_cast<uint8_t>(row % 256);
    p.w1[static_cast<size_t>(row * p.hidden / d::kFp4PerByte)] = 0x07;
  }
  d::DeviceExpert expert(p);
  expert.Enqueue();
  const auto got = expert.Read();
  const auto want = d::Golden(p);
  // NaN clamp semantics and overflow reassociation are not finite ULP tests.
  // Compare the projection codec classification separately from healthy data.
  for (size_t row = 0; row < got[0].size(); ++row) {
    const float a = d::Bf16BitsToFloat(got[0][row]);
    const float b = d::Bf16BitsToFloat(want[0][row]);
    if (std::isnan(b))
      EXPECT_TRUE(std::isnan(a)) << row;
    else if (std::isinf(b))
      EXPECT_EQ(a, b) << row;
    else
      EXPECT_LE(d::Bf16UlpDistance(got[0][row], want[0][row]), d::kDeviceMaxUlp) << row;
  }
}

INSTANTIATE_TEST_SUITE_P(DeepSeekV4, Dsv4Production, ::testing::Values(704, 4096, d::kProductionHidden));
}  // namespace
}  // namespace vllm_ascend::test
