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
#include "ascend950_shapes.hpp"
#include "cpu_reference.hpp"
#include "device_tensor.hpp"
#include "random_data.hpp"
#include "tensor_compare.hpp"
#include "test_harness.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace s = shapes950;

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

TEST(RmsNorm950PrShapes, LayerWidthsAreBurstAligned) {
  EXPECT_EQ(s::kHidden % s::kFp16ElementsPerBurst, 0);
  EXPECT_EQ(s::kIntermediate % s::kFp16ElementsPerBurst, 0);
  for (int64_t hidden : s::RmsNormHiddenSizes()) {
    EXPECT_EQ(hidden % s::kFp16ElementsPerBurst, 0)
        << "hidden=" << hidden << " is not a multiple of " << s::kFp16ElementsPerBurst << " fp16 elements";
  }
}

TEST(RmsNorm950PrSocGate, AcceptsEvery950PrBinAndNothingElse) {
  EXPECT_TRUE(s::IsAscend950PrSocName("Ascend950PR_9599"));
  EXPECT_TRUE(s::IsAscend950PrSocName("Ascend950PR_957b"));
  EXPECT_TRUE(s::IsAscend950PrSocName("Ascend950PR_950z"));
  EXPECT_FALSE(s::IsAscend950PrSocName("Ascend950DT_9591"));
  EXPECT_FALSE(s::IsAscend950PrSocName("Ascend310P3"));
  EXPECT_FALSE(s::IsAscend950PrSocName("Ascend950"));
  EXPECT_FALSE(s::IsAscend950PrSocName(""));
}

class RmsNorm950PrTest : public ::testing::TestWithParam<std::tuple<int64_t, int64_t>> {
 protected:
  int64_t num_tokens() const { return std::get<0>(GetParam()); }
  int64_t hidden() const { return std::get<1>(GetParam()); }
};

TEST_P(RmsNorm950PrTest, MatchesCpuReference) {
  REQUIRE_ASCEND_950PR();
  REQUIRE_ACLNN_OP(RmsNormOp());

  DeterministicRandom random(0x5157454eu);

  const std::vector<float> x =
      random.NormalHalfExact(static_cast<size_t>(num_tokens() * hidden()), 0.0f, 1.0f);
  const std::vector<float> gamma = random.NormalHalfExact(static_cast<size_t>(hidden()), 1.0f, 0.1f);

  const RmsNormResult actual = RunRmsNormOnDevice(x, gamma, num_tokens(), hidden(), s::kRmsNormEps);

  std::vector<float> expected_y;
  std::vector<float> expected_rstd;
  reference::RmsNorm(x, gamma, num_tokens(), hidden(), s::kRmsNormEps, &expected_y, &expected_rstd);

  EXPECT_TENSORS_ALLCLOSE(actual.y, QuantizeToHalf(expected_y), kFp16DefaultTolerance);
  EXPECT_TENSORS_ALLCLOSE(actual.rstd, expected_rstd, kFp16DefaultTolerance);
}

TEST_P(RmsNorm950PrTest, IsInvariantToRowScaling) {
  REQUIRE_ASCEND_950PR();
  REQUIRE_ACLNN_OP(RmsNormOp());

  DeterministicRandom random(0x524d534eu);

  const std::vector<float> x =
      random.NormalHalfExact(static_cast<size_t>(num_tokens() * hidden()), 0.0f, 1.0f);
  const std::vector<float> gamma(static_cast<size_t>(hidden()), 1.0f);

  std::vector<float> scaled(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    scaled[i] = HalfBitsToFloat(FloatToHalfBits(x[i] * 4.0f));
  }

  const RmsNormResult base = RunRmsNormOnDevice(x, gamma, num_tokens(), hidden(), s::kRmsNormEps);
  const RmsNormResult scaled_result = RunRmsNormOnDevice(scaled, gamma, num_tokens(), hidden(), s::kRmsNormEps);

  EXPECT_TENSORS_ALLCLOSE(scaled_result.y, base.y, kFp16DefaultTolerance);
}

TEST_P(RmsNorm950PrTest, HandlesNearZeroRowsWithoutBlowingUp) {
  REQUIRE_ASCEND_950PR();
  REQUIRE_ACLNN_OP(RmsNormOp());

  const std::vector<float> x(static_cast<size_t>(num_tokens() * hidden()), 0.0f);
  const std::vector<float> gamma(static_cast<size_t>(hidden()), 1.0f);

  const RmsNormResult actual = RunRmsNormOnDevice(x, gamma, num_tokens(), hidden(), s::kRmsNormEps);

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

INSTANTIATE_TEST_SUITE_P(Qwen35, RmsNorm950PrTest,
                         ::testing::Combine(::testing::ValuesIn(s::TokenCounts()),
                                            ::testing::ValuesIn(s::RmsNormHiddenSizes())),
                         RmsNormTestName);

}
}
}
