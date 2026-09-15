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
#include "fp16.hpp"
#include "random_data.hpp"
#include "tensor_compare.hpp"
#include "test_harness.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace s = shapes950;

constexpr int64_t kSwiGluSplitDim = -1;

const AclnnOp& SwiGluOp() {
  static const AclnnOp op(ops::kSwiGlu);
  return op;
}

std::vector<float> RunSwiGluOnDevice(const std::vector<float>& x, int64_t num_tokens, int64_t intermediate) {
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  DeviceTensor x_device = DeviceTensor::Half({num_tokens, intermediate * 2}, x);
  DeviceTensor out_device = DeviceTensor::HalfEmpty({num_tokens, intermediate});

  RunAclnn<ops::SwiGluWorkspaceFn>(SwiGluOp(), stream, x_device.get(), kSwiGluSplitDim, out_device.get());

  return out_device.ToFloatFromHalf();
}

TEST(SwiGlu950PrReference, MatchesClosedFormAtKnownPoints) {
  const std::vector<float> x = {0.0f, 1.0f, -1.0f, 2.0f,
                                3.0f, 2.0f, 5.0f, 0.0f};
  std::vector<float> out;
  reference::SiluAndMul(x, 1, 4, &out);

  ASSERT_EQ(out.size(), 4u);
  EXPECT_NEAR(out[0], 0.0f, 1e-6f);
  EXPECT_NEAR(out[1], 0.7310586f * 2.0f, 1e-5f);
  EXPECT_NEAR(out[2], -0.2689414f * 5.0f, 1e-5f);
  EXPECT_NEAR(out[3], 0.0f, 1e-6f);
}

TEST(SwiGlu950PrShapes, LayerAndSweepWidthsAreBurstAligned) {
  EXPECT_EQ(s::kIntermediate % s::kFp16ElementsPerBurst, 0);
  EXPECT_EQ((s::kIntermediate * 2) % s::kFp16ElementsPerBurst, 0);
  for (int64_t intermediate : s::IntermediateSizes()) {
    EXPECT_EQ(intermediate % s::kFp16ElementsPerBurst, 0) << "intermediate=" << intermediate;
    EXPECT_EQ((intermediate * 2) % s::kFp16ElementsPerBurst, 0) << "intermediate=" << intermediate;
  }
}

class SwiGlu950PrTest : public ::testing::TestWithParam<std::tuple<int64_t, int64_t>> {
 protected:
  int64_t num_tokens() const { return std::get<0>(GetParam()); }
  int64_t intermediate() const { return std::get<1>(GetParam()); }
};

TEST_P(SwiGlu950PrTest, MatchesCpuReference) {
  REQUIRE_ASCEND_950PR();
  REQUIRE_ACLNN_OP(SwiGluOp());

  DeterministicRandom random(0x53574755u);

  const std::vector<float> x =
      random.NormalHalfExact(static_cast<size_t>(num_tokens() * intermediate() * 2), 0.0f, 2.0f);

  const std::vector<float> actual = RunSwiGluOnDevice(x, num_tokens(), intermediate());

  std::vector<float> expected;
  reference::SiluAndMul(x, num_tokens(), intermediate(), &expected);

  EXPECT_TENSORS_ALLCLOSE(actual, QuantizeToHalf(expected), kFp16DefaultTolerance);
}

TEST_P(SwiGlu950PrTest, SplitsTheInputAtTheHalfwayPoint) {
  REQUIRE_ASCEND_950PR();
  REQUIRE_ACLNN_OP(SwiGluOp());

  const int64_t width = intermediate();
  std::vector<float> x(static_cast<size_t>(num_tokens() * width * 2), 0.0f);
  DeterministicRandom random(0x53504c54u);

  for (int64_t token = 0; token < num_tokens(); ++token) {
    const size_t row = static_cast<size_t>(token * width * 2);
    for (int64_t i = 0; i < width; ++i) {
      x[row + static_cast<size_t>(i)] = HalfBitsToFloat(FloatToHalfBits(random.Normal(0.0f, 1.0f)));
      x[row + static_cast<size_t>(width + i)] = 0.0f;
    }
  }

  const std::vector<float> actual = RunSwiGluOnDevice(x, num_tokens(), width);
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i], 0.0f) << "up half was zero but output is non-zero at index " << i;
  }
}

TEST_P(SwiGlu950PrTest, SaturatesRatherThanOverflowsOnLargeGates) {
  REQUIRE_ASCEND_950PR();
  REQUIRE_ACLNN_OP(SwiGluOp());

  const int64_t width = intermediate();
  std::vector<float> x(static_cast<size_t>(num_tokens() * width * 2), 0.0f);
  for (int64_t token = 0; token < num_tokens(); ++token) {
    const size_t row = static_cast<size_t>(token * width * 2);
    for (int64_t i = 0; i < width; ++i) {
      x[row + static_cast<size_t>(i)] = (i % 2 == 0) ? 30.0f : -30.0f;
      x[row + static_cast<size_t>(width + i)] = 1.0f;
    }
  }

  const std::vector<float> actual = RunSwiGluOnDevice(x, num_tokens(), width);

  for (int64_t token = 0; token < num_tokens(); ++token) {
    const size_t row = static_cast<size_t>(token * width);
    for (int64_t i = 0; i < width; ++i) {
      const float value = actual[row + static_cast<size_t>(i)];
      ASSERT_TRUE(std::isfinite(value)) << "non-finite output at token " << token << " index " << i;
      if (i % 2 == 0) {
        EXPECT_NEAR(value, 30.0f, 5e-2f) << "positive tail at token " << token << " index " << i;
      } else {
        EXPECT_NEAR(value, 0.0f, 1e-3f) << "negative tail at token " << token << " index " << i;
      }
    }
  }
}

std::string SwiGluTestName(const ::testing::TestParamInfo<std::tuple<int64_t, int64_t>>& info) {
  std::ostringstream name;
  name << "tokens" << std::get<0>(info.param) << "_intermediate" << std::get<1>(info.param);
  return name.str();
}

INSTANTIATE_TEST_SUITE_P(Qwen35, SwiGlu950PrTest,
                         ::testing::Combine(::testing::ValuesIn(s::TokenCounts()),
                                            ::testing::ValuesIn(s::IntermediateSizes())),
                         SwiGluTestName);

}
}
}
