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

// The RMSNorm sweep both silicon legs run: aclnnRmsNorm (y and rstd) against
// the float CPU reference over the shared Qwen3.5 hidden sizes, plus the
// CPU-reference checks that need no device. Compiled into
// test_device_950pr_rmsnorm and test_rmsnorm_310p alike; the SoC each binary
// targets surfaces only as the REQUIRE_OPERATOR_TARGET_SOC gate. The per-part
// tests live in the leg's own correctness/test_rmsnorm.cpp.

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "aclnn_ops.hpp"
#include "aclnn_runtime.hpp"
#include "cpu_reference.hpp"
#include "device_tensor.hpp"
#include "op_test_fixture.hpp"
#include "random_data.hpp"
#include "sweep_shapes.hpp"
#include "tensor_compare.hpp"
#include "test_harness.hpp"

namespace vllm_ascend {
namespace test {
namespace {

struct RmsNormResult {
  std::vector<float> y;
  std::vector<float> rstd;
};

const AclnnOp& RmsNormOp() {
  static const AclnnOp op(ops::kRmsNorm);
  return op;
}

RmsNormResult RunRmsNormOnDevice(const std::vector<float>& x, const std::vector<float>& gamma, int64_t num_tokens,
                                 int64_t hidden, float epsilon) {
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  DeviceTensor x_device = DeviceTensor::Half({num_tokens, hidden}, x);
  DeviceTensor gamma_device = DeviceTensor::Half({hidden}, gamma);
  DeviceTensor y_device = DeviceTensor::HalfEmpty({num_tokens, hidden});
  DeviceTensor rstd_device = DeviceTensor::FloatEmpty({num_tokens, 1});

  RunAclnn<ops::RmsNormWorkspaceFn>(RmsNormOp(), stream, x_device.get(), gamma_device.get(),
                                    static_cast<double>(epsilon), y_device.get(), rstd_device.get());

  RmsNormResult result;
  result.y = y_device.ToFloatFromHalf();
  result.rstd = rstd_device.ToFloat();
  return result;
}

// CPU-reference checks. No device, no SoC gate: the reference is shared code
// and must agree with itself on every part.
TEST(RmsNormReference, NormalisesAConstantRowToOne) {
  const int64_t hidden = 2048;
  const std::vector<float> x(static_cast<size_t>(hidden), 2.0f);
  const std::vector<float> gamma(static_cast<size_t>(hidden), 1.0f);

  std::vector<float> y;
  std::vector<float> rstd;
  reference::RmsNorm(x, gamma, 1, hidden, sweeps::kRmsNormEpsilon, &y, &rstd);

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
  reference::RmsNorm(x, gamma, 1, hidden, sweeps::kRmsNormEpsilon, &y, &rstd);

  for (int64_t i = 0; i < hidden; ++i) {
    EXPECT_NEAR(y[static_cast<size_t>(i)], static_cast<float>(i), 1e-3f) << "at index " << i;
  }
}

class RmsNormSweepTest : public ::testing::TestWithParam<std::tuple<int64_t, int64_t>> {
 protected:
  int64_t num_tokens() const { return std::get<0>(GetParam()); }
  int64_t hidden() const { return std::get<1>(GetParam()); }
};

TEST_P(RmsNormSweepTest, MatchesCpuReference) {
  REQUIRE_OPERATOR_TARGET_SOC();
  REQUIRE_ACLNN_OP(RmsNormOp());

  DeterministicRandom random(0x5157454eu);

  const std::vector<float> x = random.NormalHalfExact(static_cast<size_t>(num_tokens() * hidden()), 0.0f, 1.0f);
  const std::vector<float> gamma = random.NormalHalfExact(static_cast<size_t>(hidden()), 1.0f, 0.1f);

  const RmsNormResult actual = RunRmsNormOnDevice(x, gamma, num_tokens(), hidden(), sweeps::kRmsNormEpsilon);

  std::vector<float> expected_y;
  std::vector<float> expected_rstd;
  reference::RmsNorm(x, gamma, num_tokens(), hidden(), sweeps::kRmsNormEpsilon, &expected_y, &expected_rstd);

  EXPECT_HALF_TENSORS_ALLCLOSE(actual.y, expected_y);
  EXPECT_HALF_TENSORS_ALLCLOSE(actual.rstd, expected_rstd);
}

TEST_P(RmsNormSweepTest, IsInvariantToRowScaling) {
  REQUIRE_OPERATOR_TARGET_SOC();
  REQUIRE_ACLNN_OP(RmsNormOp());

  DeterministicRandom random(0x524d534eu);

  const std::vector<float> x = random.NormalHalfExact(static_cast<size_t>(num_tokens() * hidden()), 0.0f, 1.0f);
  const std::vector<float> gamma(static_cast<size_t>(hidden()), 1.0f);

  std::vector<float> scaled(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    scaled[i] = HalfBitsToFloat(FloatToHalfBits(x[i] * 4.0f));
  }

  const RmsNormResult base = RunRmsNormOnDevice(x, gamma, num_tokens(), hidden(), sweeps::kRmsNormEpsilon);
  const RmsNormResult scaled_result =
      RunRmsNormOnDevice(scaled, gamma, num_tokens(), hidden(), sweeps::kRmsNormEpsilon);

  EXPECT_TENSORS_ALLCLOSE(scaled_result.y, base.y, kFp16DefaultTolerance);
}

TEST_P(RmsNormSweepTest, HandlesNearZeroRowsWithoutBlowingUp) {
  REQUIRE_OPERATOR_TARGET_SOC();
  REQUIRE_ACLNN_OP(RmsNormOp());

  const std::vector<float> x(static_cast<size_t>(num_tokens() * hidden()), 0.0f);
  const std::vector<float> gamma(static_cast<size_t>(hidden()), 1.0f);

  const RmsNormResult actual = RunRmsNormOnDevice(x, gamma, num_tokens(), hidden(), sweeps::kRmsNormEpsilon);

  for (size_t i = 0; i < actual.y.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual.y[i])) << "non-finite output at index " << i;
    EXPECT_NEAR(actual.y[i], 0.0f, 1e-6f) << "at index " << i;
  }
  for (size_t i = 0; i < actual.rstd.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual.rstd[i])) << "non-finite rstd at index " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(Qwen35, RmsNormSweepTest,
                         ::testing::Combine(::testing::ValuesIn(sweeps::TokenCounts()),
                                            ::testing::ValuesIn(sweeps::RmsNormHiddenSizes())),
                         op_case::TupleName("tokens", "hidden"));

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
