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

// RMSNorm on the native CUDA C backend, fp16 in / fp16 out.
//
// The device cases are the same three the Ascend suite runs in
// test_rmsnorm_310p.cpp - parity against reference::RmsNorm, invariance to row
// scaling, and a near-zero row - over the same Qwen3.5 shapes and with the same
// seeds and tolerances. What changes is only how the kernel is reached:
// LaunchRmsNormHalf instead of a dlsym'd aclnnRmsNorm.
//
// The host-only checks are repeated here rather than shared, because a CUDA
// build never compiles the Ascend test files and the reference deserves to be
// checked in whichever build is running.

#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "cpu_reference.hpp"
#include "cuda_device_tensor.hpp"
#include "cuda_kernels.hpp"
#include "cuda_runtime.hpp"
#include "qwen_shapes.hpp"
#include "random_data.hpp"
#include "tensor_compare.hpp"

namespace vllm_ascend {
namespace test {
namespace {

struct RmsNormResult {
  std::vector<float> y;
  std::vector<float> rstd;
};

RmsNormResult RunRmsNormOnDevice(const std::vector<float>& x, const std::vector<float>& gamma,
                                 int64_t num_tokens, int64_t hidden, float epsilon) {
  cudaStream_t stream = CUDATestEnvironment::Instance().stream();

  CudaDeviceTensor x_device = CudaDeviceTensor::Half({num_tokens, hidden}, x);
  CudaDeviceTensor gamma_device = CudaDeviceTensor::Half({hidden}, gamma);
  CudaDeviceTensor y_device = CudaDeviceTensor::HalfEmpty({num_tokens, hidden});
  // rstd keeps the reduction in fp32. The plugin discards it, but it is the
  // cheapest window into whether the reduction itself is right.
  CudaDeviceTensor rstd_device = CudaDeviceTensor::FloatEmpty({num_tokens, 1});

  cuda::LaunchRmsNormHalf(x_device.half_data(), gamma_device.half_data(), y_device.half_data(),
                          rstd_device.float_data(), num_tokens, hidden, epsilon, stream);
  CUDATestEnvironment::Instance().device().SynchronizeStream();

  RmsNormResult result;
  result.y = y_device.ToFloatFromHalf();
  result.rstd = rstd_device.ToFloat();
  return result;
}

// -----------------------------------------------------------------------------
// Host-only checks. These run on a machine with no GPU attached, so a mistake
// in the reference itself is caught before any hardware is involved.
// -----------------------------------------------------------------------------

TEST(RmsNormReference, NormalisesAConstantRowToOne) {
  // Every element equal to c gives mean(x^2) = c^2, so y = c / sqrt(c^2 + eps),
  // which is 1 to within eps for a unit gamma.
  const int64_t hidden = 2048;
  const std::vector<float> x(static_cast<size_t>(hidden), 2.0f);
  const std::vector<float> gamma(static_cast<size_t>(hidden), 1.0f);

  std::vector<float> y;
  std::vector<float> rstd;
  reference::RmsNorm(x, gamma, 1, hidden, shapes::kRmsNormEpsilon, &y, &rstd);

  for (int64_t i = 0; i < hidden; ++i) {
    EXPECT_NEAR(y[static_cast<size_t>(i)], 1.0f, 1e-5f) << "at index " << i;
  }
  EXPECT_NEAR(rstd[0], 0.5f, 1e-5f);
}

TEST(RmsNormReference, AppliesGammaPerChannel) {
  const int64_t hidden = 16;
  const std::vector<float> x(static_cast<size_t>(hidden), 1.0f);
  std::vector<float> gamma(static_cast<size_t>(hidden));
  for (int64_t i = 0; i < hidden; ++i) {
    gamma[static_cast<size_t>(i)] = static_cast<float>(i);
  }

  std::vector<float> y;
  std::vector<float> rstd;
  reference::RmsNorm(x, gamma, 1, hidden, shapes::kRmsNormEpsilon, &y, &rstd);

  // rstd is ~1 for a row of ones, so y should reproduce gamma.
  for (int64_t i = 0; i < hidden; ++i) {
    EXPECT_NEAR(y[static_cast<size_t>(i)], static_cast<float>(i), 1e-3f) << "at index " << i;
  }
}

// -----------------------------------------------------------------------------
// Device parity
// -----------------------------------------------------------------------------

class RmsNormCudaTest : public ::testing::TestWithParam<std::tuple<int64_t, int64_t>> {
 protected:
  int64_t num_tokens() const { return std::get<0>(GetParam()); }
  int64_t hidden() const { return std::get<1>(GetParam()); }
};

TEST_P(RmsNormCudaTest, MatchesCpuReference) {
  REQUIRE_CUDA_DEVICE();

  DeterministicRandom random(0x5157454eu);  // "QWEN"

  // Values are pre-rounded to fp16 so the device and the reference start from
  // bit-identical inputs and the comparison measures arithmetic only.
  const std::vector<float> x =
      random.NormalHalfExact(static_cast<size_t>(num_tokens() * hidden()), 0.0f, 1.0f);
  const std::vector<float> gamma = random.NormalHalfExact(static_cast<size_t>(hidden()), 1.0f, 0.1f);

  const RmsNormResult actual = RunRmsNormOnDevice(x, gamma, num_tokens(), hidden(), shapes::kRmsNormEpsilon);

  std::vector<float> expected_y;
  std::vector<float> expected_rstd;
  reference::RmsNorm(x, gamma, num_tokens(), hidden(), shapes::kRmsNormEpsilon, &expected_y, &expected_rstd);

  // The device writes fp16, so the reference is rounded the same way before
  // comparing; otherwise the fp16 output quantisation alone eats the tolerance.
  EXPECT_TENSORS_ALLCLOSE(actual.y, QuantizeToHalf(expected_y), kFp16DefaultTolerance);
  EXPECT_TENSORS_ALLCLOSE(actual.rstd, expected_rstd, kFp16DefaultTolerance);
}

TEST_P(RmsNormCudaTest, IsInvariantToRowScaling) {
  REQUIRE_CUDA_DEVICE();

  // RMSNorm divides by the row RMS, so scaling a row by k leaves the output
  // unchanged except for the epsilon term. This catches a kernel that folds the
  // scale in the wrong place without needing the CPU reference at all.
  DeterministicRandom random(0x524d534eu);  // "RMSN"

  const std::vector<float> x =
      random.NormalHalfExact(static_cast<size_t>(num_tokens() * hidden()), 0.0f, 1.0f);
  const std::vector<float> gamma(static_cast<size_t>(hidden()), 1.0f);

  std::vector<float> scaled(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    // 4 is exact in fp16, so scaling introduces no rounding of its own.
    scaled[i] = HalfBitsToFloat(FloatToHalfBits(x[i] * 4.0f));
  }

  const RmsNormResult base = RunRmsNormOnDevice(x, gamma, num_tokens(), hidden(), shapes::kRmsNormEpsilon);
  const RmsNormResult scaled_result =
      RunRmsNormOnDevice(scaled, gamma, num_tokens(), hidden(), shapes::kRmsNormEpsilon);

  EXPECT_TENSORS_ALLCLOSE(scaled_result.y, base.y, kFp16DefaultTolerance);
}

TEST_P(RmsNormCudaTest, HandlesNearZeroRowsWithoutBlowingUp) {
  REQUIRE_CUDA_DEVICE();

  // A row of exact zeros makes the epsilon the only term under the square root.
  // With eps=1e-6 the reciprocal is 1e3, large enough that a kernel computing
  // it in fp16 rather than fp32 would saturate, and large enough that the fast
  // rsqrtf approximation would be visible.
  const std::vector<float> x(static_cast<size_t>(num_tokens() * hidden()), 0.0f);
  const std::vector<float> gamma(static_cast<size_t>(hidden()), 1.0f);

  const RmsNormResult actual = RunRmsNormOnDevice(x, gamma, num_tokens(), hidden(), shapes::kRmsNormEpsilon);

  for (size_t i = 0; i < actual.y.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual.y[i])) << "non-finite output at index " << i;
    EXPECT_NEAR(actual.y[i], 0.0f, 1e-6f) << "at index " << i;
  }
  for (size_t i = 0; i < actual.rstd.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual.rstd[i])) << "non-finite rstd at index " << i;
    EXPECT_NEAR(actual.rstd[i], 1.0f / std::sqrt(shapes::kRmsNormEpsilon), 1.0f) << "at index " << i;
  }
}

std::string RmsNormTestName(const ::testing::TestParamInfo<std::tuple<int64_t, int64_t>>& info) {
  std::ostringstream name;
  name << "tokens" << std::get<0>(info.param) << "_hidden" << std::get<1>(info.param);
  return name.str();
}

INSTANTIATE_TEST_SUITE_P(Qwen35, RmsNormCudaTest,
                         ::testing::Combine(::testing::ValuesIn(shapes::TokenCounts()),
                                            ::testing::ValuesIn(shapes::RmsNormHiddenSizes())),
                         RmsNormTestName);

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
