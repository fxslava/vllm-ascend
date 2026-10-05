// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include <gtest/gtest.h>

#include <cstdio>

#include "dsv4_metrics.hpp"

namespace vllm_ascend::test {
inline void ExpectBf16Parity(const std::vector<uint16_t>& actual, const std::vector<uint16_t>& golden,
                             const char* name) {
  ASSERT_EQ(actual.size(), golden.size()) << name;
  ASSERT_FALSE(actual.empty()) << name;
  const auto metrics = dsv4::CompareBf16(actual, golden);
  std::printf("%s FpDiff=%lld BF16_ULP RateDiff=%.8f Mismatch=%zu Specials=%zu\n", name,
              static_cast<long long>(metrics.fp_diff), metrics.rate_diff, metrics.mismatch, metrics.special_mismatch);
  EXPECT_EQ(metrics.special_mismatch, 0u) << name;
  EXPECT_LE(metrics.fp_diff, dsv4::kDeviceMaxUlp) << name;
  EXPECT_LE(metrics.rate_diff, dsv4::kDeviceMaxRate) << name;
}

inline void ExpectParity(const dsv4::DeviceOutputs& actual, const dsv4::DeviceOutputs& golden) {
  constexpr const char* kOutputNames[] = {"gate_out", "up_out", "activated_out", "down_out"};
  for (size_t t = 0; t < actual.size(); ++t) ExpectBf16Parity(actual[t], golden[t], kOutputNames[t]);
}
}  // namespace vllm_ascend::test
