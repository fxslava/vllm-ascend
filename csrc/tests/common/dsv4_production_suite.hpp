// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "dsv4_device_case.hpp"
#include "dsv4_test_checks.hpp"

namespace vllm_ascend::test {
namespace {
namespace d = dsv4;

enum class ExpertHardware { k950PR, k910B };
struct ExpertCase {
  int64_t hidden;
  int64_t inter;
  ExpertHardware hardware;
  bool simulator = false;
};
class Dsv4Production : public ::testing::TestWithParam<ExpertCase> {
 protected:
  void SetUp() override {
    auto& env = AscendTestEnvironment::Instance();
    if (GetParam().hardware == ExpertHardware::k910B) {
      ASSERT_TRUE(env.available()) << env.unavailable_reason();
      const auto& soc = env.soc_name();
      ASSERT_TRUE(soc == "Ascend910B1" || soc == "Ascend910B2" || soc == "Ascend910B3" || soc == "Ascend910B4") << soc;
      constexpr size_t k910BUbBytes = 192 * 1024;
      // Same explicit buffers as the shipping 8x512 streaming kernel.
      ASSERT_LT(d::ExpertUbBytes(GetParam().hidden, GetParam().inter), k910BUbBytes);
    } else {
      REQUIRE_ASCEND_950PR();
    }
    ASSERT_EQ(IsRunningOnSimulator(), GetParam().simulator);
    ASSERT_EQ(env.device().device_id(), 0);
  }
  d::Problem Problem(uint32_t seed) const { return d::MakeDeviceProblem(GetParam().hidden, GetParam().inter, seed); }
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
  std::printf("determinism: 20 consecutive launches, four buffers bit-identical\n");
}

TEST_P(Dsv4Production, PositiveAndNegativeGateClamp) {
  for (const uint8_t code : {uint8_t{0x77}, uint8_t{0xFF}}) {
    const auto p = d::MakeSaturatingProblem(GetParam().hidden, GetParam().inter, 0xD541, code == 0x77);
    const auto oracle = d::ReferenceExpert(p.View(), p.hidden, p.inter);
    for (float gate : oracle.gate_f32) {
      if (code == 0x77)
        ASSERT_GT(gate, 100.0f);
      else
        ASSERT_LT(gate, -100.0f);
    }
    d::DeviceExpert expert(p);
    expert.Enqueue();
    const auto got = expert.Read();
    ExpectParity(got, d::Golden(p));
    for (const auto& tensor : got)
      for (uint16_t bits : tensor) EXPECT_TRUE(std::isfinite(d::Bf16BitsToFloat(bits)));
  }
}

// Random weights hide a row that decodes the wrong staged bytes: the result is
// just another plausible dot product. Under MakeSawtoothProblem every element
// is a small signed integer whose sign is fixed by its row's phase, so an
// aliased or short-read row flips a sign. That is the shape the arch220 byte
// widening failed in -- a Gather bounded by its source tensor's extent read
// undefined data past the first 512 staged bytes, so from the third row of
// every eight-row chunk onwards the decode was garbage while
// GoldenAndRepeatedBitParity still reported a plausible miss.
TEST_P(Dsv4Production, SawtoothRowPhaseParity) {
  const auto p = d::MakeSawtoothProblem(GetParam().hidden, GetParam().inter, 0xD543);
  d::DeviceExpert expert(p);
  expert.Enqueue();
  ExpectParity(expert.Read(), d::Golden(p));
}

TEST_P(Dsv4Production, SixConcurrentExpertsHavePrivateStorage) {
  std::vector<std::unique_ptr<d::DeviceExpert>> experts;
  std::vector<d::DeviceOutputs> goldens;
  for (uint32_t i = 0; i < 6; ++i) {
    const auto p = Problem(0xD550 + 2 * i);
    goldens.push_back(d::Golden(p));
    experts.push_back(std::make_unique<d::DeviceExpert>(p));
    for (size_t j = 0; j < i; ++j) ASSERT_NE(experts[i]->stream(), experts[j]->stream());
  }
  std::vector<d::DeviceOutputs> first(experts.size());
  for (int repeat = 0; repeat < 10; ++repeat) {
    // Submit all streams before waiting on any: no host-side serialization.
    const auto outputs = RunConcurrentExperts(experts);
    for (size_t i = 0; i < experts.size(); ++i) {
      const auto& got = outputs[i];
      ExpectParity(got, goldens[i]);
      if (repeat == 0)
        first[i] = got;
      else
        EXPECT_EQ(got, first[i]) << "stream " << i << " repeat " << repeat;
    }
  }
  std::printf("concurrency: six private-buffer streams, ten rounds, golden and bit parity checked\n");
}

TEST_P(Dsv4Production, ExhaustivePackedBytesInBothNibblePositions) {
  // Isolate either nibble with a one-hot x; cancellation cannot hide a decoder defect.
  for (int nibble = 0; nibble < 2; ++nibble) {
    auto p = Problem(0xD561);
    std::fill(p.x.begin(), p.x.end(), d::FloatToBf16Bits(0.0f));
    p.x[nibble] = d::FloatToBf16Bits(1.0f);
    for (auto* weights : {&p.w1, &p.w3}) {
      std::fill(weights->begin(), weights->end(), 0);
      for (int64_t row = 0; row < p.inter; ++row)
        (*weights)[static_cast<size_t>(row * p.hidden / d::kFp4PerByte)] = static_cast<uint8_t>(row % 256);
    }
    std::fill(p.w1_scale.begin(), p.w1_scale.end(), 127);
    std::fill(p.w3_scale.begin(), p.w3_scale.end(), 127);
    d::DeviceExpert expert(p);
    expert.Enqueue();
    const auto got = expert.Read();
    ExpectParity(got, d::Golden(p));
    for (size_t row = 0; row < got[0].size(); ++row) {
      const uint8_t byte = static_cast<uint8_t>(row % 256);
      const float value = d::E2m1Table()[nibble == 0 ? byte & 15 : byte >> 4];
      EXPECT_EQ(d::Bf16UlpDistance(got[0][row], d::FloatToBf16Bits(value)), 0);
      EXPECT_EQ(d::Bf16UlpDistance(got[1][row], d::FloatToBf16Bits(value)), 0);
    }
  }
}

TEST_P(Dsv4Production, EveryScaleByteAndAllFp4Codes) {
  const auto p = d::MakeScaleSweepProblem(GetParam().hidden, GetParam().inter, 0xD560);
  d::DeviceExpert expert(p);
  expert.Enqueue();
  const auto got = expert.Read();
  const auto want = d::Golden(p);
  // NaN clamp semantics and overflow reassociation are not finite ULP tests.
  // Compare the projection codec classification separately from healthy data.
  ExpectBf16Parity(got[0], want[0], "gate_out/all_scales");
}

TEST_P(Dsv4Production, FiniteScaleBoundariesAcrossAllOutputs) {
  auto p = Problem(0xD562);
  std::fill(p.x.begin(), p.x.end(), d::FloatToBf16Bits(1.0f));
  for (auto* weights : {&p.w1, &p.w3}) std::fill(weights->begin(), weights->end(), 0);
  for (auto* scales : {&p.w1_scale, &p.w3_scale}) std::fill(scales->begin(), scales->end(), 127);
  for (int64_t row = 0; row < p.inter; ++row) {
    // 0 exercises the fp32 subnormal; 1..127 includes all normal test scales.
    const uint8_t scale = static_cast<uint8_t>(row % 128);
    for (auto* weights : {&p.w1, &p.w3}) (*weights)[static_cast<size_t>(row * p.hidden / d::kFp4PerByte)] = 0x22;
    for (auto* scales : {&p.w1_scale, &p.w3_scale})
      (*scales)[static_cast<size_t>(row * p.hidden / d::kFp4Block)] = scale;
  }
  d::DeviceExpert expert(p);
  expert.Enqueue();
  ExpectParity(expert.Read(), d::Golden(p));
}

}  // namespace
}  // namespace vllm_ascend::test
