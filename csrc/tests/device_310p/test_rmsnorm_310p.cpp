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

#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "aclnn_ops.hpp"
#include "aclnn_runtime.hpp"
#include "cpu_reference.hpp"
#include "device_tensor.hpp"
#include "qwen_shapes.hpp"
#include "random_data.hpp"
#include "tensor_compare.hpp"
#include "test_harness.hpp"

namespace vllm_ascend {
namespace test {
namespace {

struct RmsNormResult {
  std::vector<float> y;
  std::vector<float> rstd;
};

RmsNormResult RunRmsNormOnDevice(const std::vector<float>& x, const std::vector<float>& gamma, int64_t num_tokens,
                                 int64_t hidden, float epsilon, const AclnnOp& op) {
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  DeviceTensor x_device = DeviceTensor::Half({num_tokens, hidden}, x);
  DeviceTensor gamma_device = DeviceTensor::Half({hidden}, gamma);
  DeviceTensor y_device = DeviceTensor::HalfEmpty({num_tokens, hidden});
  DeviceTensor rstd_device = DeviceTensor::FloatEmpty({num_tokens, 1});

  RunAclnn<ops::RmsNormWorkspaceFn>(op, stream, x_device.get(), gamma_device.get(),
                                    static_cast<double>(epsilon), y_device.get(), rstd_device.get());

  RmsNormResult result;
  result.y = y_device.ToFloatFromHalf();
  result.rstd = rstd_device.ToFloat();
  return result;
}

const AclnnOp& RmsNormOp() {
  static const AclnnOp op(ops::kRmsNorm);
  return op;
}

TEST(RmsNormReference, NormalisesAConstantRowToOne) {
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

  for (int64_t i = 0; i < hidden; ++i) {
    EXPECT_NEAR(y[static_cast<size_t>(i)], static_cast<float>(i), 1e-3f) << "at index " << i;
  }
}

TEST(RmsNormShapes, QwenHiddenSizesAreBurstAligned) {
  for (int64_t hidden : shapes::RmsNormHiddenSizes()) {
    EXPECT_EQ(hidden % shapes::kFp16ElementsPerBurst, 0)
        << "hidden=" << hidden << " is not a multiple of " << shapes::kFp16ElementsPerBurst
        << " fp16 elements (32 bytes)";
  }
}

class RmsNorm310PTest : public ::testing::TestWithParam<std::tuple<int64_t, int64_t>> {
 protected:
  int64_t num_tokens() const { return std::get<0>(GetParam()); }
  int64_t hidden() const { return std::get<1>(GetParam()); }
};

TEST_P(RmsNorm310PTest, MatchesCpuReference) {
  REQUIRE_ASCEND_310P();
  REQUIRE_ACLNN_OP(RmsNormOp());

  DeterministicRandom random(0x5157454eu);

  const std::vector<float> x =
      random.NormalHalfExact(static_cast<size_t>(num_tokens() * hidden()), 0.0f, 1.0f);
  const std::vector<float> gamma = random.NormalHalfExact(static_cast<size_t>(hidden()), 1.0f, 0.1f);

  const RmsNormResult actual =
      RunRmsNormOnDevice(x, gamma, num_tokens(), hidden(), shapes::kRmsNormEpsilon, RmsNormOp());

  std::vector<float> expected_y;
  std::vector<float> expected_rstd;
  reference::RmsNorm(x, gamma, num_tokens(), hidden(), shapes::kRmsNormEpsilon, &expected_y, &expected_rstd);

  EXPECT_TENSORS_ALLCLOSE(actual.y, QuantizeToHalf(expected_y), kFp16DefaultTolerance);
  EXPECT_TENSORS_ALLCLOSE(actual.rstd, expected_rstd, kFp16DefaultTolerance);
}

TEST_P(RmsNorm310PTest, IsInvariantToRowScaling) {
  REQUIRE_ASCEND_310P();
  REQUIRE_ACLNN_OP(RmsNormOp());

  DeterministicRandom random(0x524d534eu);

  const std::vector<float> x =
      random.NormalHalfExact(static_cast<size_t>(num_tokens() * hidden()), 0.0f, 1.0f);
  const std::vector<float> gamma(static_cast<size_t>(hidden()), 1.0f);

  std::vector<float> scaled(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    scaled[i] = HalfBitsToFloat(FloatToHalfBits(x[i] * 4.0f));
  }

  const RmsNormResult base =
      RunRmsNormOnDevice(x, gamma, num_tokens(), hidden(), shapes::kRmsNormEpsilon, RmsNormOp());
  const RmsNormResult scaled_result =
      RunRmsNormOnDevice(scaled, gamma, num_tokens(), hidden(), shapes::kRmsNormEpsilon, RmsNormOp());

  EXPECT_TENSORS_ALLCLOSE(scaled_result.y, base.y, kFp16DefaultTolerance);
}

TEST_P(RmsNorm310PTest, HandlesNearZeroRowsWithoutBlowingUp) {
  REQUIRE_ASCEND_310P();
  REQUIRE_ACLNN_OP(RmsNormOp());

  const std::vector<float> x(static_cast<size_t>(num_tokens() * hidden()), 0.0f);
  const std::vector<float> gamma(static_cast<size_t>(hidden()), 1.0f);

  const RmsNormResult actual =
      RunRmsNormOnDevice(x, gamma, num_tokens(), hidden(), shapes::kRmsNormEpsilon, RmsNormOp());

  for (size_t i = 0; i < actual.y.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual.y[i])) << "non-finite output at index " << i;
    EXPECT_NEAR(actual.y[i], 0.0f, 1e-6f) << "at index " << i;
  }
  for (size_t i = 0; i < actual.rstd.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual.rstd[i])) << "non-finite rstd at index " << i;
  }
}

std::string RmsNormTestName(const ::testing::TestParamInfo<std::tuple<int64_t, int64_t>>& info) {
  std::ostringstream name;
  name << "tokens" << std::get<0>(info.param) << "_hidden" << std::get<1>(info.param);
  return name.str();
}

INSTANTIATE_TEST_SUITE_P(Qwen35, RmsNorm310PTest,
                         ::testing::Combine(::testing::ValuesIn(shapes::TokenCounts()),
                                            ::testing::ValuesIn(shapes::RmsNormHiddenSizes())),
                         RmsNormTestName);

}
}
}
