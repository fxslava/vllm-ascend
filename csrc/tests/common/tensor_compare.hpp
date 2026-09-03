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

// Numerical comparison between an NPU result and the CPU reference.
//
// The predicate is the same one torch.allclose uses:
//     |actual - expected| <= atol + rtol * |expected|
// so a tolerance chosen here means the same thing as in the Python tests.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace vllm_ascend {
namespace test {

struct Tolerance {
  double atol;
  double rtol;
  // Why these numbers, so a future change to them is a deliberate decision.
  const char* rationale;
};

// Default from the plugin Python tests. One fp16 ULP at magnitude 1.0 is about
// 9.8e-4, so this is roughly "within one ULP plus a rounding step" and is the
// right bar for the elementwise and single-reduction kernels.
inline constexpr Tolerance kFp16DefaultTolerance{1e-3, 1e-3,
                                                 "atol=rtol=1e-3, matching tests/ut; ~1 fp16 ULP near unit scale"};

// Softmax over a long context accumulates over hundreds of terms and the
// hardware exp differs from libm in the last bits, so the decode attention
// output needs more room than a single elementwise op.
inline constexpr Tolerance kPagedAttentionTolerance{
    4e-3, 4e-3, "relaxed from 1e-3: fp16 softmax over multi-block contexts; re-tune on first hardware run"};

struct ComparisonReport {
  bool passed = true;
  size_t element_count = 0;
  size_t mismatch_count = 0;
  size_t non_finite_count = 0;
  double max_abs_error = 0.0;
  size_t max_abs_index = 0;
  double max_rel_error = 0.0;
  size_t max_rel_index = 0;
  double first_mismatch_actual = 0.0;
  double first_mismatch_expected = 0.0;
  size_t first_mismatch_index = 0;

  std::string Describe(const Tolerance& tolerance) const {
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(6);
    stream << "\n  elements        : " << element_count;
    stream << "\n  mismatches      : " << mismatch_count;
    stream << "\n  non-finite      : " << non_finite_count;
    stream << "\n  tolerance       : atol=" << tolerance.atol << " rtol=" << tolerance.rtol;
    stream << "\n                    (" << tolerance.rationale << ")";
    stream << "\n  max |abs error| : " << max_abs_error << " at index " << max_abs_index;
    stream << "\n  max |rel error| : " << max_rel_error << " at index " << max_rel_index;
    if (mismatch_count > 0) {
      stream << "\n  first mismatch  : index " << first_mismatch_index << " npu=" << first_mismatch_actual
             << " reference=" << first_mismatch_expected;
    }
    return stream.str();
  }
};

inline ComparisonReport CompareAllClose(const std::vector<float>& actual, const std::vector<float>& expected,
                                        const Tolerance& tolerance) {
  ComparisonReport report;
  report.element_count = expected.size();

  if (actual.size() != expected.size()) {
    report.passed = false;
    report.mismatch_count = expected.size();
    return report;
  }

  bool first_recorded = false;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double a = static_cast<double>(actual[i]);
    const double e = static_cast<double>(expected[i]);

    if (!std::isfinite(a) || !std::isfinite(e)) {
      ++report.non_finite_count;
      // Two NaNs or two identical infinities agree; anything else is a failure.
      const bool both_nan = std::isnan(a) && std::isnan(e);
      const bool same_inf = std::isinf(a) && std::isinf(e) && ((a > 0) == (e > 0));
      if (!both_nan && !same_inf) {
        ++report.mismatch_count;
        report.passed = false;
        if (!first_recorded) {
          first_recorded = true;
          report.first_mismatch_index = i;
          report.first_mismatch_actual = a;
          report.first_mismatch_expected = e;
        }
      }
      continue;
    }

    const double abs_error = std::fabs(a - e);
    const double rel_error = abs_error / std::max(std::fabs(e), 1e-12);

    if (abs_error > report.max_abs_error) {
      report.max_abs_error = abs_error;
      report.max_abs_index = i;
    }
    if (rel_error > report.max_rel_error) {
      report.max_rel_error = rel_error;
      report.max_rel_index = i;
    }

    if (abs_error > tolerance.atol + tolerance.rtol * std::fabs(e)) {
      ++report.mismatch_count;
      report.passed = false;
      if (!first_recorded) {
        first_recorded = true;
        report.first_mismatch_index = i;
        report.first_mismatch_actual = a;
        report.first_mismatch_expected = e;
      }
    }
  }
  return report;
}

#define EXPECT_TENSORS_ALLCLOSE(actual, expected, tolerance)                                    \
  do {                                                                                          \
    const ::vllm_ascend::test::ComparisonReport vllm_ascend_report =                            \
        ::vllm_ascend::test::CompareAllClose((actual), (expected), (tolerance));                \
    EXPECT_TRUE(vllm_ascend_report.passed)                                                      \
        << "NPU result does not match the CPU reference:"                                       \
        << vllm_ascend_report.Describe(tolerance);                                              \
  } while (false)

#define ASSERT_TENSORS_ALLCLOSE(actual, expected, tolerance)                                    \
  do {                                                                                          \
    const ::vllm_ascend::test::ComparisonReport vllm_ascend_report =                            \
        ::vllm_ascend::test::CompareAllClose((actual), (expected), (tolerance));                \
    ASSERT_TRUE(vllm_ascend_report.passed)                                                      \
        << "NPU result does not match the CPU reference:"                                       \
        << vllm_ascend_report.Describe(tolerance);                                              \
  } while (false)

}  // namespace test
}  // namespace vllm_ascend
