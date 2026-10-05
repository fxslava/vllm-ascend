// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "ascend_concurrent_runner.hpp"
#include "dsv4_metrics.hpp"
#include "dsv4_synthetic_data.hpp"

namespace vllm_ascend::test {
namespace {
namespace d = dsv4;

TEST(Dsv4CommonMetrics, StrictToleranceCountsAndMaximum) {
  const std::vector<uint16_t> golden = {0x3F80, 0x3F80, 0x3F80, 0x3F80};
  const std::vector<uint16_t> actual = {0x3F80, 0x3F81, 0x3F82, 0x3F83};
  EXPECT_EQ(d::ComputeFpDiff(actual, golden), 3);
  const auto result = d::ComputeRateDiff(actual, golden);
  EXPECT_EQ(result.mismatch, 1u);
  EXPECT_DOUBLE_EQ(result.rate_diff, 0.25);
  EXPECT_EQ(d::ComputeRateDiff(actual, golden, 3).mismatch, 0u);
}

TEST(Dsv4CommonMetrics, SpecialsRequireClassificationAndInfinitySign) {
  const std::vector<uint16_t> golden = {0x7FC0, 0x7F80, 0xFF80};
  EXPECT_EQ(d::CompareBf16({0x7FFF, 0x7F80, 0xFF80}, golden).mismatch, 0u);
  const auto result = d::CompareBf16({0x3F80, 0xFF80, 0x7F80}, golden);
  EXPECT_EQ(result.special_mismatch, 3u);
  EXPECT_EQ(result.mismatch, 3u);
  EXPECT_DOUBLE_EQ(result.rate_diff, 1.0);
}

TEST(Dsv4CommonMetrics, EmptySignedZeroAndInvalidInputs) {
  EXPECT_EQ(d::CompareBf16({}, {}).mismatch, 0u);
  EXPECT_DOUBLE_EQ(d::CompareBf16({}, {}).rate_diff, 0.0);
  EXPECT_EQ(d::ComputeFpDiff({0x0000}, {0x8000}), 0);
  EXPECT_THROW(d::CompareBf16({0}, {}), std::invalid_argument);
  EXPECT_THROW(d::CompareBf16({0}, {0}, -1), std::invalid_argument);
  d::DeviceOutputs empty;
  EXPECT_THROW(d::RequireOutputParity(empty, empty), std::invalid_argument);
}

TEST(Dsv4CommonData, ReproducibleBitsAndExhaustivePackedBytes) {
  const auto a = d::MakeDeviceProblem(64, 64, 0xD540);
  const auto b = d::MakeDeviceProblem(64, 64, 0xD540);
  EXPECT_EQ(a.x, b.x);
  EXPECT_EQ(a.w1, b.w1);
  EXPECT_EQ(a.w2, b.w2);
  EXPECT_EQ(a.w3, b.w3);
  EXPECT_EQ(a.w1_scale, b.w1_scale);
  EXPECT_EQ(a.w2_scale, b.w2_scale);
  EXPECT_EQ(a.w3_scale, b.w3_scale);
  EXPECT_NE(a.x, d::MakeDeviceProblem(64, 64, 0xD542).x);
  for (const auto* weights : {&a.w1, &a.w2, &a.w3})
    for (size_t i = 0; i < 256; ++i) EXPECT_EQ((*weights)[i], i);
  for (const auto* scales : {&a.w1_scale, &a.w2_scale, &a.w3_scale})
    for (uint8_t scale : *scales) EXPECT_TRUE(scale >= 120 && scale <= 124);
}

TEST(Dsv4CommonData, ScaleSweepIncludesSubnormalAndNaN) {
  const auto p = d::MakeScaleSweepProblem(64, 256, 1);
  for (size_t row = 0; row < 256; ++row) EXPECT_EQ(p.w1_scale[row * 2], row);
  EXPECT_EQ(d::FloatToBf16Bits(d::E8m0ToScale(p.w1_scale[0])), 0x0040);
  EXPECT_TRUE(std::isnan(d::E8m0ToScale(p.w1_scale[255 * 2])));
}

TEST(Dsv4CommonData, SaturationIsSignedAndActivationFinite) {
  for (bool positive : {false, true}) {
    const auto p = d::MakeSaturatingProblem(64, 64, 3, positive);
    const auto oracle = d::ReferenceExpert(p.View(), p.hidden, p.inter);
    for (float gate : oracle.gate_f32) {
      if (positive)
        EXPECT_GT(gate, 100.0f);
      else
        EXPECT_LT(gate, -100.0f);
    }
    for (uint16_t bits : oracle.activated) EXPECT_TRUE(std::isfinite(d::Bf16BitsToFloat(bits)));
  }
}

TEST(Dsv4CommonData, InvalidGeometryAndScaleRangeFail) {
  EXPECT_THROW(d::MakeDeviceProblem(63, 64, 1), std::invalid_argument);
  EXPECT_THROW(d::MakeDeviceProblem(64, 64, 1, 125, 120), std::invalid_argument);
  EXPECT_THROW(d::MakeProblem(64, 64, 1, 127), std::invalid_argument);
}

TEST(Dsv4CommonData, ProductionFitsConservativeUbAndAliasUsesLargestDimension) {
  EXPECT_EQ(d::ExpertUbBytes(7168, 2048), 173568u);
  EXPECT_LT(d::ExpertUbBytes(7168, 2048), 192u * 1024);
  EXPECT_EQ(d::ExpertUbBytes(64, 128), d::kExpertStreamingScratchBytes + 128u + 512u + 2304u);
}

struct FakeExpert {
  int id;
  std::vector<int>& actions;
  int stream() const { return id; }
  void Enqueue() { actions.push_back(id); }
  int Read() {
    actions.push_back(-id);
    return id;
  }
};

TEST(AscendConcurrentRunner, SubmitsAllSixBeforeAnyRead) {
  std::vector<int> actions;
  std::vector<std::unique_ptr<FakeExpert>> experts;
  for (int i = 1; i <= 6; ++i) experts.push_back(std::make_unique<FakeExpert>(FakeExpert{i, actions}));
  EXPECT_EQ(RunConcurrentExperts(experts), (std::vector<int>{1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(actions, (std::vector<int>{1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6}));
}

TEST(AscendConcurrentRunner, RejectsSharedStreamsAndNullBeforeSubmission) {
  std::vector<int> actions;
  std::vector<std::unique_ptr<FakeExpert>> experts;
  experts.push_back(std::make_unique<FakeExpert>(FakeExpert{1, actions}));
  experts.push_back(std::make_unique<FakeExpert>(FakeExpert{1, actions}));
  EXPECT_THROW(RunConcurrentExperts(experts), std::invalid_argument);
  EXPECT_TRUE(actions.empty());
  experts.back().reset();
  EXPECT_THROW(RunConcurrentExperts(experts), std::invalid_argument);
  EXPECT_TRUE(actions.empty());
}
}  // namespace
}  // namespace vllm_ascend::test
