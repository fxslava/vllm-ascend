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
#include "fp16.hpp"
#include "qwen_shapes.hpp"
#include "random_data.hpp"
#include "tensor_compare.hpp"
#include "test_harness.hpp"

namespace vllm_ascend {
namespace test {
namespace {

const AclnnOp& MatmulOp() {
  static const AclnnOp op(ops::kMatmul);
  return op;
}

std::vector<float> RunMatmulOnDevice(const std::vector<float>& a, const std::vector<float>& b_t, int64_t m,
                                     int64_t k, int64_t n, const AclnnOp& op) {
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  DeviceTensor a_device = DeviceTensor::Half({m, k}, a);
  DeviceTensor b_device = DeviceTensor::HalfTransposed2D(n, k, b_t);
  DeviceTensor out_device = DeviceTensor::HalfEmpty({m, n});

  RunAclnn<ops::MatmulWorkspaceFn>(op, stream, a_device.get(), b_device.get(), out_device.get(),
                                   ops::kCubeMathTypeKeepDtype);

  return out_device.ToFloatFromHalf();
}

TEST(MatmulReference, IdentityWeightReproducesInput) {
  const int64_t k = 8;
  const int64_t n = 8;
  std::vector<float> a(static_cast<size_t>(k));
  for (int64_t i = 0; i < k; ++i) {
    a[static_cast<size_t>(i)] = static_cast<float>(i) - 3.0f;
  }

  std::vector<float> b_t(static_cast<size_t>(n * k), 0.0f);
  for (int64_t i = 0; i < n; ++i) {
    b_t[static_cast<size_t>(i * k + i)] = 1.0f;
  }

  std::vector<float> out;
  reference::MatmulTransposedB(a, b_t, 1, k, n, &out);

  ASSERT_EQ(out.size(), static_cast<size_t>(n));
  for (int64_t j = 0; j < n; ++j) {
    EXPECT_FLOAT_EQ(out[static_cast<size_t>(j)], a[static_cast<size_t>(j)]) << "at column " << j;
  }
}

TEST(MatmulReference, MatchesHandComputedCase) {
  const std::vector<float> a{1.0f, 2.0f, 3.0f};
  const std::vector<float> b_t{1.0f, 0.0f, -1.0f, 2.0f, 2.0f, 2.0f};

  std::vector<float> out;
  reference::MatmulTransposedB(a, b_t, 1, 3, 2, &out);

  ASSERT_EQ(out.size(), 2u);
  EXPECT_FLOAT_EQ(out[0], -2.0f);
  EXPECT_FLOAT_EQ(out[1], 12.0f);
}

TEST(MatmulReference, IsLinearInTheInput) {
  const int64_t k = 32;
  const int64_t n = 16;
  DeterministicRandom random(0x4c494e21u);

  const std::vector<float> a1 = random.NormalHalfExact(static_cast<size_t>(k), 0.0f, 1.0f);
  const std::vector<float> a2 = random.NormalHalfExact(static_cast<size_t>(k), 0.0f, 1.0f);
  const std::vector<float> b_t = random.NormalHalfExact(static_cast<size_t>(n * k), 0.0f, 0.25f);

  std::vector<float> sum_input(a1.size());
  for (size_t i = 0; i < a1.size(); ++i) {
    sum_input[i] = a1[i] + a2[i];
  }

  std::vector<float> out_sum;
  std::vector<float> out1;
  std::vector<float> out2;
  reference::MatmulTransposedB(sum_input, b_t, 1, k, n, &out_sum);
  reference::MatmulTransposedB(a1, b_t, 1, k, n, &out1);
  reference::MatmulTransposedB(a2, b_t, 1, k, n, &out2);

  for (int64_t j = 0; j < n; ++j) {
    const size_t idx = static_cast<size_t>(j);
    EXPECT_NEAR(out_sum[idx], out1[idx] + out2[idx], 1e-4f) << "at column " << j;
  }
}

TEST(MatmulShapes, QwenProjectionShapesAreBurstAligned) {
  for (int64_t k : shapes::LinearInputSizes()) {
    EXPECT_EQ(k % shapes::kFp16ElementsPerBurst, 0)
        << "K=" << k << " is not a multiple of " << shapes::kFp16ElementsPerBurst << " fp16 elements (32 bytes)";
  }
  for (int64_t n : shapes::LinearOutputSizes()) {
    EXPECT_EQ(n % shapes::kFp16ElementsPerBurst, 0)
        << "N=" << n << " is not a multiple of " << shapes::kFp16ElementsPerBurst << " fp16 elements (32 bytes)";
  }
}

class Matmul310PTest : public ::testing::TestWithParam<std::tuple<int64_t, int64_t>> {
 protected:
  int64_t k() const { return std::get<0>(GetParam()); }
  int64_t n() const { return std::get<1>(GetParam()); }
  int64_t m() const { return shapes::kDecodeTokenCount; }

  float weight_stddev() const { return 1.0f / std::sqrt(static_cast<float>(k())); }
};

TEST_P(Matmul310PTest, MatchesCpuReference) {
  REQUIRE_ASCEND_310P();
  REQUIRE_ACLNN_OP(MatmulOp());

  DeterministicRandom random(0x4d4d554cu);

  const std::vector<float> a = random.NormalHalfExact(static_cast<size_t>(m() * k()), 0.0f, 1.0f);
  const std::vector<float> b_t =
      random.NormalHalfExact(static_cast<size_t>(n() * k()), 0.0f, weight_stddev());

  const std::vector<float> actual = RunMatmulOnDevice(a, b_t, m(), k(), n(), MatmulOp());

  std::vector<float> expected;
  reference::MatmulTransposedB(a, b_t, m(), k(), n(), &expected);

  EXPECT_TENSORS_ALLCLOSE(actual, QuantizeToHalf(expected), kFp16DefaultTolerance);
}

TEST_P(Matmul310PTest, ZeroWeightsProduceZeroOutput) {
  REQUIRE_ASCEND_310P();
  REQUIRE_ACLNN_OP(MatmulOp());

  DeterministicRandom random(0x5a45524fu);

  const std::vector<float> a = random.NormalHalfExact(static_cast<size_t>(m() * k()), 0.0f, 1.0f);
  const std::vector<float> b_t(static_cast<size_t>(n() * k()), 0.0f);

  const std::vector<float> actual = RunMatmulOnDevice(a, b_t, m(), k(), n(), MatmulOp());

  ASSERT_EQ(actual.size(), static_cast<size_t>(m() * n()));
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual[i])) << "non-finite output at index " << i;
    EXPECT_EQ(actual[i], 0.0f) << "at index " << i;
  }
}

TEST_P(Matmul310PTest, IsLinearInTheInput) {
  REQUIRE_ASCEND_310P();
  REQUIRE_ACLNN_OP(MatmulOp());

  DeterministicRandom random(0x4c494e32u);

  const std::vector<float> a = random.NormalHalfExact(static_cast<size_t>(m() * k()), 0.0f, 1.0f);
  const std::vector<float> b_t =
      random.NormalHalfExact(static_cast<size_t>(n() * k()), 0.0f, weight_stddev());

  std::vector<float> doubled(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    doubled[i] = HalfBitsToFloat(FloatToHalfBits(a[i] * 2.0f));
  }

  const std::vector<float> base = RunMatmulOnDevice(a, b_t, m(), k(), n(), MatmulOp());
  const std::vector<float> scaled = RunMatmulOnDevice(doubled, b_t, m(), k(), n(), MatmulOp());

  std::vector<float> expected(base.size());
  for (size_t i = 0; i < base.size(); ++i) {
    expected[i] = base[i] * 2.0f;
  }

  EXPECT_TENSORS_ALLCLOSE(scaled, QuantizeToHalf(expected), kFp16DefaultTolerance);
}

std::string MatmulTestName(const ::testing::TestParamInfo<std::tuple<int64_t, int64_t>>& info) {
  std::ostringstream name;
  name << "k" << std::get<0>(info.param) << "_n" << std::get<1>(info.param);
  return name.str();
}

INSTANTIATE_TEST_SUITE_P(Qwen35, Matmul310PTest,
                         ::testing::Combine(::testing::ValuesIn(shapes::LinearInputSizes()),
                                            ::testing::ValuesIn(shapes::LinearOutputSizes())),
                         MatmulTestName);

}
}
}
