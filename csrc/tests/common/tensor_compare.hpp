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
#include <iostream>
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

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------
//
// allclose asks whether every element is individually close. Cosine similarity
// asks whether the two tensors point the same way, which is the question worth
// asking once a value is a whole layer's output rather than one operator's:
//
//   * It is scale-free. A residual stream that has grown to |x| ~ 3.5 makes a
//     fixed atol=1e-3 a far tighter bar than the same number meant at unit
//     magnitude, so a purely elementwise gate either passes trivially or fails
//     for reasons that have nothing to do with the kernel.
//   * It is one number over the whole tensor, so fp16 noise spread thinly
//     across 2048 channels does not read as a failure, while a genuine defect -
//     a transposed weight, a mis-split fused projection, a rotary table at the
//     wrong base - moves it immediately. 1 - cos is quadratic in the angle, so
//     a floor of 0.9999 is a bound of about 1.4e-2 on the relative size of the
//     component pointing the wrong way: tight, not generous.
//
// MAE and max |abs error| are reported alongside it because cosine similarity
// alone cannot see a uniform scale error: a result that is exactly 2x the
// reference has cosine similarity 1.

struct SimilarityReport {
  bool passed = true;
  size_t element_count = 0;
  size_t non_finite_count = 0;
  double cosine_similarity = 0.0;
  double mean_abs_error = 0.0;
  double max_abs_error = 0.0;
  size_t max_abs_index = 0;
  double actual_norm = 0.0;
  double expected_norm = 0.0;

  std::string Describe(const char* label, double floor) const {
    std::ostringstream stream;
    stream << "\n  stage           : " << label;
    stream << "\n  elements        : " << element_count;
    stream << "\n  non-finite      : " << non_finite_count;
    stream << std::fixed << std::setprecision(8);
    stream << "\n  cosine sim      : " << cosine_similarity << "  (floor " << floor << ")";
    stream << std::scientific << std::setprecision(6);
    stream << "\n  MAE             : " << mean_abs_error;
    stream << "\n  max |abs error| : " << max_abs_error << " at index " << max_abs_index;
    stream << "\n  norms           : device=" << actual_norm << " reference=" << expected_norm;
    return stream.str();
  }
};

// Accumulation is in double: an fp32 dot product over a few thousand terms
// loses enough of the tail that the eighth decimal of the result - exactly the
// digit the 0.9999 gate reads - would be accumulation noise.
//
// Two all-zero tensors are defined to have similarity 1, since they agree,
// while a zero against a non-zero is 0: there the angle really is undefined and
// the honest report is a failure rather than a division by a guard value.
inline SimilarityReport CompareCosineSimilarity(const std::vector<float>& actual,
                                                const std::vector<float>& expected) {
  SimilarityReport report;
  report.element_count = expected.size();

  if (actual.size() != expected.size()) {
    report.passed = false;
    return report;
  }

  double dot = 0.0;
  double actual_square = 0.0;
  double expected_square = 0.0;
  double absolute_error_sum = 0.0;

  for (size_t i = 0; i < expected.size(); ++i) {
    const double a = static_cast<double>(actual[i]);
    const double e = static_cast<double>(expected[i]);

    if (!std::isfinite(a) || !std::isfinite(e)) {
      ++report.non_finite_count;
      report.passed = false;
      continue;
    }

    dot += a * e;
    actual_square += a * a;
    expected_square += e * e;

    const double abs_error = std::fabs(a - e);
    absolute_error_sum += abs_error;
    if (abs_error > report.max_abs_error) {
      report.max_abs_error = abs_error;
      report.max_abs_index = i;
    }
  }

  report.actual_norm = std::sqrt(actual_square);
  report.expected_norm = std::sqrt(expected_square);
  report.mean_abs_error =
      expected.empty() ? 0.0 : absolute_error_sum / static_cast<double>(expected.size());

  const double denominator = report.actual_norm * report.expected_norm;
  if (denominator > 0.0) {
    report.cosine_similarity = dot / denominator;
  } else {
    report.cosine_similarity = (report.actual_norm == 0.0 && report.expected_norm == 0.0) ? 1.0 : 0.0;
  }
  return report;
}

// The quality gate a whole-layer parity test wants: same direction to within
// `floor`, with the metrics reported either way - so a pass is an observation
// and not just a green tick. A failure gets them from gtest's own message, so
// the [ METRIC ] line is printed only when there is no failure message for it
// to duplicate.
#define EXPECT_TENSORS_COSINE_SIMILAR(actual, expected, floor, label)                           \
  do {                                                                                          \
    const ::vllm_ascend::test::SimilarityReport vllm_ascend_similarity =                        \
        ::vllm_ascend::test::CompareCosineSimilarity((actual), (expected));                     \
    const bool vllm_ascend_within_floor = vllm_ascend_similarity.passed &&                      \
                                          vllm_ascend_similarity.cosine_similarity >= (floor);  \
    EXPECT_TRUE(vllm_ascend_within_floor)                                                       \
        << "device result diverges from the reference:"                                         \
        << vllm_ascend_similarity.Describe((label), (floor));                                   \
    if (vllm_ascend_within_floor) {                                                             \
      std::cout << "[  METRIC  ]" << vllm_ascend_similarity.Describe((label), (floor)) << "\n"; \
    }                                                                                           \
  } while (false)

}  // namespace test
}  // namespace vllm_ascend
