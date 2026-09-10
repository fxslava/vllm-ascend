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

// RMSNorm on Ascend 950PR, fp16 in / fp16 out.
//
// Stage 1 and stage 7 of the Qwen3.5 decoder layer. The operator is the stock
// aclnnRmsNorm - the 950PR custom kernel set in csrc/ adds fused variants
// (AddRmsNormBias, RmsNormDynamicQuant) but nothing that replaces a plain
// fp16 RMSNorm, so there is no custom kernel to prefer here.
//
// The seeds, shapes and tolerances are the ones the CUDA suite uses
// (csrc/tests/kernels/cuda/test_rmsnorm.cpp), so a number that differs between
// the two backends is a backend difference and not a test difference.

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
  // rstd keeps the reduction in fp32; the operator requires the output even
  // though the plugin discards it.
  DeviceTensor rstd_device = DeviceTensor::FloatEmpty({num_tokens, 1});

  RunAclnn<ops::RmsNormWorkspaceFn>(RmsNormOp(), stream, x_device.get(), gamma_device.get(),
                                    static_cast<double>(epsilon), y_device.get(), rstd_device.get());

  RmsNormResult result;
  result.y = y_device.ToFloatFromHalf();
  result.rstd = rstd_device.ToFloat();
  return result;
}

// -----------------------------------------------------------------------------
// Host-only checks. These run on a build machine with no NPU attached.
// -----------------------------------------------------------------------------

TEST(RmsNorm950PrShapes, LayerWidthsAreBurstAligned) {
  // An fp16 row must be a whole number of 32-byte MTE bursts, otherwise the
  // tail burst reads past the row. Both widths the Qwen3.5 layer uses, and
  // every width the sweep covers, satisfy it.
  EXPECT_EQ(s::kHidden % s::kFp16ElementsPerBurst, 0);
  EXPECT_EQ(s::kIntermediate % s::kFp16ElementsPerBurst, 0);
  for (int64_t hidden : s::RmsNormHiddenSizes()) {
    EXPECT_EQ(hidden % s::kFp16ElementsPerBurst, 0)
        << "hidden=" << hidden << " is not a multiple of " << s::kFp16ElementsPerBurst << " fp16 elements";
  }
}

TEST(RmsNorm950PrSocGate, AcceptsEvery950PrBinAndNothingElse) {
  // The gate the device tests use. Every 950PR bin in the CANN platform_config
  // set shares the prefix; the DT part and the 310P part must not slip through,
  // and neither must an empty name from a runtime with no aclrtGetSocName.
  EXPECT_TRUE(s::IsAscend950PrSocName("Ascend950PR_9599"));
  EXPECT_TRUE(s::IsAscend950PrSocName("Ascend950PR_957b"));
  EXPECT_TRUE(s::IsAscend950PrSocName("Ascend950PR_950z"));
  EXPECT_FALSE(s::IsAscend950PrSocName("Ascend950DT_9591"));
  EXPECT_FALSE(s::IsAscend950PrSocName("Ascend310P3"));
  EXPECT_FALSE(s::IsAscend950PrSocName("Ascend950"));
  EXPECT_FALSE(s::IsAscend950PrSocName(""));
}

// -----------------------------------------------------------------------------
// Device parity
// -----------------------------------------------------------------------------

class RmsNorm950PrTest : public ::testing::TestWithParam<std::tuple<int64_t, int64_t>> {
 protected:
  int64_t num_tokens() const { return std::get<0>(GetParam()); }
  int64_t hidden() const { return std::get<1>(GetParam()); }
};

TEST_P(RmsNorm950PrTest, MatchesCpuReference) {
  REQUIRE_ASCEND_950PR();
  REQUIRE_ACLNN_OP(RmsNormOp());

  DeterministicRandom random(0x5157454eu);  // "QWEN"

  // Values are pre-rounded to fp16 so the device and the reference start from
  // bit-identical inputs and the comparison measures arithmetic only.
  const std::vector<float> x =
      random.NormalHalfExact(static_cast<size_t>(num_tokens() * hidden()), 0.0f, 1.0f);
  const std::vector<float> gamma = random.NormalHalfExact(static_cast<size_t>(hidden()), 1.0f, 0.1f);

  const RmsNormResult actual = RunRmsNormOnDevice(x, gamma, num_tokens(), hidden(), s::kRmsNormEps);

  std::vector<float> expected_y;
  std::vector<float> expected_rstd;
  reference::RmsNorm(x, gamma, num_tokens(), hidden(), s::kRmsNormEps, &expected_y, &expected_rstd);

  // The device writes fp16, so the reference is rounded the same way before
  // comparing; otherwise the fp16 output quantisation alone eats the tolerance.
  EXPECT_TENSORS_ALLCLOSE(actual.y, QuantizeToHalf(expected_y), kFp16DefaultTolerance);
  EXPECT_TENSORS_ALLCLOSE(actual.rstd, expected_rstd, kFp16DefaultTolerance);
}

TEST_P(RmsNorm950PrTest, IsInvariantToRowScaling) {
  REQUIRE_ASCEND_950PR();
  REQUIRE_ACLNN_OP(RmsNormOp());

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

  const RmsNormResult base = RunRmsNormOnDevice(x, gamma, num_tokens(), hidden(), s::kRmsNormEps);
  const RmsNormResult scaled_result = RunRmsNormOnDevice(scaled, gamma, num_tokens(), hidden(), s::kRmsNormEps);

  EXPECT_TENSORS_ALLCLOSE(scaled_result.y, base.y, kFp16DefaultTolerance);
}

TEST_P(RmsNorm950PrTest, HandlesNearZeroRowsWithoutBlowingUp) {
  REQUIRE_ASCEND_950PR();
  REQUIRE_ACLNN_OP(RmsNormOp());

  // A row of exact zeros makes the epsilon the only term under the square root.
  // With eps=1e-6 the reciprocal is 1e3, large enough that a kernel computing it
  // in fp16 rather than fp32 would saturate.
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

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
