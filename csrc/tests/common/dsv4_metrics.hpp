// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "dsv4_test_oracle.hpp"

namespace vllm_ascend::test::dsv4 {
constexpr int64_t kDeviceMaxUlp = 2;
constexpr double kDeviceMaxRate = 1e-2;

struct Bf16Metrics {
  int64_t fp_diff = 0;  // Finite values only; specials have no numeric ULP distance.
  double rate_diff = 0;
  size_t mismatch = 0;
  size_t special_mismatch = 0;
};

inline bool SpecialValuesMatch(float actual, float golden) {
  return (std::isnan(actual) && std::isnan(golden)) ||
         (std::isinf(actual) && std::isinf(golden) && std::signbit(actual) == std::signbit(golden));
}

inline Bf16Metrics CompareBf16(const std::vector<uint16_t>& actual, const std::vector<uint16_t>& golden,
                               int64_t max_ulp_tol = kDeviceMaxUlp) {
  if (actual.size() != golden.size()) throw std::invalid_argument("BF16 comparison size mismatch");
  if (max_ulp_tol < 0) throw std::invalid_argument("negative BF16 ULP tolerance");
  Bf16Metrics result;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float a = Bf16BitsToFloat(actual[i]);
    const float b = Bf16BitsToFloat(golden[i]);
    if (!std::isfinite(a) || !std::isfinite(b)) {
      if (!SpecialValuesMatch(a, b)) {
        ++result.special_mismatch;
        ++result.mismatch;
      }
    } else {
      const int64_t ulp = Bf16UlpDistance(actual[i], golden[i]);
      result.fp_diff = std::max(result.fp_diff, ulp);
      if (ulp > max_ulp_tol) ++result.mismatch;
    }
  }
  if (!actual.empty()) result.rate_diff = static_cast<double>(result.mismatch) / actual.size();
  return result;
}

inline int64_t ComputeFpDiff(const std::vector<uint16_t>& actual, const std::vector<uint16_t>& golden) {
  return CompareBf16(actual, golden).fp_diff;
}

// Returns the rate together with Mismatch and special classification counts.
inline Bf16Metrics ComputeRateDiff(const std::vector<uint16_t>& actual, const std::vector<uint16_t>& golden,
                                   int64_t max_ulp_tol = kDeviceMaxUlp) {
  return CompareBf16(actual, golden, max_ulp_tol);
}

inline void RequireOutputParity(const DeviceOutputs& actual, const DeviceOutputs& golden) {
  for (size_t t = 0; t < actual.size(); ++t) {
    if (actual[t].empty()) throw std::invalid_argument("empty expert output");
    const auto result = CompareBf16(actual[t], golden[t]);
    if (result.fp_diff > kDeviceMaxUlp || result.rate_diff > kDeviceMaxRate || result.special_mismatch != 0)
      throw std::runtime_error("expert output " + std::to_string(t) + " failed golden parity; refusing timings");
  }
}
}  // namespace vllm_ascend::test::dsv4
