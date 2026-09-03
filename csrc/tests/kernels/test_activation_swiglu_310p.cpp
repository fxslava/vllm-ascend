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

// SiluAndMul / SwiGLU on Ascend 310P, fp16 in / fp16 out.
//
// Mirrors vllm_ascend/_310p/ops/activation.py :: AscendSiluAndMul310.forward:
//
//     if x.shape[-1] % 32 == 0:  out = torch_npu.npu_swiglu(x)
//     else:                      out = F.silu(x[..., :h]) * x[..., h:]
//
// The 32-element gate is a real 310P constraint rather than a style choice, so
// it gets its own test alongside the numerical parity checks.

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

// aclnnSwiGlu splits the last axis, matching npu_swiglu with its default dim.
constexpr int64_t kSwiGluSplitDim = -1;

const AclnnOp& SwiGluOp() {
  static const AclnnOp op(ops::kSwiGlu);
  return op;
}

std::vector<float> RunSwiGluOnDevice(const std::vector<float>& x, int64_t num_tokens, int64_t intermediate,
                                     const AclnnOp& op) {
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  DeviceTensor x_device = DeviceTensor::Half({num_tokens, intermediate * 2}, x);
  DeviceTensor out_device = DeviceTensor::HalfEmpty({num_tokens, intermediate});

  RunAclnn<ops::SwiGluWorkspaceFn>(op, stream, x_device.get(), kSwiGluSplitDim, out_device.get());

  return out_device.ToFloatFromHalf();
}

// -----------------------------------------------------------------------------
// Host-only checks
// -----------------------------------------------------------------------------

TEST(SwiGluReference, MatchesClosedFormAtKnownPoints) {
  // silu(0) = 0, silu(1) = 1/(1+e^-1) = 0.7310586, silu(-1) = -0.2689414.
  const std::vector<float> x = {0.0f, 1.0f, -1.0f, 2.0f,   // gate half
                                3.0f, 2.0f, 5.0f, 0.0f};   // up half
  std::vector<float> out;
  reference::SiluAndMul(x, /*num_tokens=*/1, /*intermediate=*/4, &out);

  ASSERT_EQ(out.size(), 4u);
  EXPECT_NEAR(out[0], 0.0f, 1e-6f);
  EXPECT_NEAR(out[1], 0.7310586f * 2.0f, 1e-5f);
  EXPECT_NEAR(out[2], -0.2689414f * 5.0f, 1e-5f);
  EXPECT_NEAR(out[3], 0.0f, 1e-6f);
}

TEST(SwiGluReference, GateOfOneLeavesTheUpProjectionScaledBySilu) {
  const int64_t intermediate = 64;
  std::vector<float> x(static_cast<size_t>(intermediate * 2), 0.0f);
  for (int64_t i = 0; i < intermediate; ++i) {
    x[static_cast<size_t>(i)] = 1.0f;                                 // gate
    x[static_cast<size_t>(intermediate + i)] = static_cast<float>(i);  // up
  }

  std::vector<float> out;
  reference::SiluAndMul(x, 1, intermediate, &out);

  const float silu_of_one = 1.0f / (1.0f + std::exp(-1.0f));
  for (int64_t i = 0; i < intermediate; ++i) {
    EXPECT_NEAR(out[static_cast<size_t>(i)], silu_of_one * static_cast<float>(i), 1e-4f) << "at index " << i;
  }
}

TEST(SwiGluShapes, QwenIntermediateSizesSatisfyThe310PGate) {
  // AscendSiluAndMul310 only reaches npu_swiglu when the *input* last dim,
  // which is 2 * intermediate_size, is a multiple of 32. Confirm every size the
  // suite exercises takes the kernel path rather than the eager fallback.
  for (int64_t intermediate : shapes::IntermediateSizes()) {
    const int64_t input_last_dim = intermediate * 2;
    EXPECT_EQ(input_last_dim % shapes::kSwiGluLastDimMultiple, 0)
        << "intermediate=" << intermediate << " gives last dim " << input_last_dim
        << ", which would take the eager fallback instead of npu_swiglu";
  }
}

TEST(SwiGluShapes, OddLastDimensionWouldTakeTheEagerFallback) {
  // Documents the other side of the gate: a width that is not a multiple of 32
  // must not be sent to the kernel. 2 * 20 = 40, which is 16-aligned but not
  // 32-aligned, so it is exactly the case the plugin guards against.
  const int64_t intermediate = 20;
  EXPECT_NE((intermediate * 2) % shapes::kSwiGluLastDimMultiple, 0);
}

// -----------------------------------------------------------------------------
// Device parity
// -----------------------------------------------------------------------------

class SwiGlu310PTest : public ::testing::TestWithParam<std::tuple<int64_t, int64_t>> {
 protected:
  int64_t num_tokens() const { return std::get<0>(GetParam()); }
  int64_t intermediate() const { return std::get<1>(GetParam()); }
};

TEST_P(SwiGlu310PTest, MatchesCpuReference) {
  REQUIRE_ASCEND_310P();
  REQUIRE_ACLNN_OP(SwiGluOp());

  DeterministicRandom random(0x53574755u);  // "SWGU"

  // A stddev of 2 pushes a useful fraction of the gate into the saturating
  // tails of silu, where a low-precision sigmoid approximation would show up.
  const std::vector<float> x =
      random.NormalHalfExact(static_cast<size_t>(num_tokens() * intermediate() * 2), 0.0f, 2.0f);

  const std::vector<float> actual = RunSwiGluOnDevice(x, num_tokens(), intermediate(), SwiGluOp());

  std::vector<float> expected;
  reference::SiluAndMul(x, num_tokens(), intermediate(), &expected);

  EXPECT_TENSORS_ALLCLOSE(actual, QuantizeToHalf(expected), kFp16DefaultTolerance);
}

TEST_P(SwiGlu310PTest, SplitsTheInputAtTheHalfwayPoint) {
  REQUIRE_ASCEND_310P();
  REQUIRE_ACLNN_OP(SwiGluOp());

  // Zeroing the up half must zero the whole output. A kernel that split the
  // input at the wrong offset, or interleaved the halves, fails here even
  // though a random-input comparison might stay inside the tolerance.
  const int64_t width = intermediate();
  std::vector<float> x(static_cast<size_t>(num_tokens() * width * 2), 0.0f);
  DeterministicRandom random(0x53504c54u);  // "SPLT"

  for (int64_t token = 0; token < num_tokens(); ++token) {
    const size_t row = static_cast<size_t>(token * width * 2);
    for (int64_t i = 0; i < width; ++i) {
      x[row + static_cast<size_t>(i)] = HalfBitsToFloat(FloatToHalfBits(random.Normal(0.0f, 1.0f)));
      x[row + static_cast<size_t>(width + i)] = 0.0f;
    }
  }

  const std::vector<float> actual = RunSwiGluOnDevice(x, num_tokens(), width, SwiGluOp());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i], 0.0f) << "up half was zero but output is non-zero at index " << i;
  }
}

TEST_P(SwiGlu310PTest, SaturatesRatherThanOverflowsOnLargeGates) {
  REQUIRE_ASCEND_310P();
  REQUIRE_ACLNN_OP(SwiGluOp());

  // silu saturates to the identity for large positive input and to zero for
  // large negative input. fp16 tops out near 65504, so a gate of +/-30 with a
  // unit up projection stays representable while still exercising the tails.
  const int64_t width = intermediate();
  std::vector<float> x(static_cast<size_t>(num_tokens() * width * 2), 0.0f);
  for (int64_t token = 0; token < num_tokens(); ++token) {
    const size_t row = static_cast<size_t>(token * width * 2);
    for (int64_t i = 0; i < width; ++i) {
      x[row + static_cast<size_t>(i)] = (i % 2 == 0) ? 30.0f : -30.0f;
      x[row + static_cast<size_t>(width + i)] = 1.0f;
    }
  }

  const std::vector<float> actual = RunSwiGluOnDevice(x, num_tokens(), width, SwiGluOp());

  for (int64_t token = 0; token < num_tokens(); ++token) {
    const size_t row = static_cast<size_t>(token * width);
    for (int64_t i = 0; i < width; ++i) {
      const float value = actual[row + static_cast<size_t>(i)];
      ASSERT_TRUE(std::isfinite(value)) << "non-finite output at token " << token << " index " << i;
      if (i % 2 == 0) {
        // One fp16 ULP at magnitude 30 is 0.03125, so the bound has to sit
        // above that or an exactly-correct rounded result would fail.
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

INSTANTIATE_TEST_SUITE_P(Qwen35, SwiGlu310PTest,
                         ::testing::Combine(::testing::ValuesIn(shapes::TokenCounts()),
                                            ::testing::ValuesIn(shapes::IntermediateSizes())),
                         SwiGluTestName);

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
