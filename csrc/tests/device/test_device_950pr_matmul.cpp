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

// Linear projections on Ascend 950PR, through the cube unit.
//
// Stages 2, 6 and 8 of the Qwen3.5 decoder layer. Every one is x @ W^T with W
// stored in the torch [out_features, in_features] layout, which is what
// DeviceTensor::HalfTransposed2D describes with strides rather than a host-side
// transpose.
//
// The operator is the stock aclnnMatmul with cubeMathType KEEP_DTYPE, so fp16
// operands stay fp16 through the cube unit. The 950PR custom matmuls are all
// quantised, grouped or fused forms.
//
// Seeds, shapes and tolerances match csrc/tests/kernels/cuda/test_matmul.cpp.
//
// NOT COVERED: M > 1. Decode is M=1, but a GEMV does not exercise the cube
// unit's M tiling, and arch35 splits cube and vector into separate cores
// (cube_vector_combine=split), so the M tiling and the cube/vector handover are
// 950PR-specific behaviour a single-row case cannot reach.

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

const AclnnOp& MatmulOp() {
  static const AclnnOp op(ops::kMatmul);
  return op;
}

// Runs aclnnMatmul with `b_t` supplied in Linear [N, K] layout and viewed as
// [K, N], and returns the fp16 [M, N] output widened back to float.
std::vector<float> RunMatmulOnDevice(const std::vector<float>& a, const std::vector<float>& b_t, int64_t m,
                                     int64_t k, int64_t n) {
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  DeviceTensor a_device = DeviceTensor::Half({m, k}, a);
  DeviceTensor b_device = DeviceTensor::HalfTransposed2D(n, k, b_t);
  DeviceTensor out_device = DeviceTensor::HalfEmpty({m, n});

  RunAclnn<ops::MatmulWorkspaceFn>(MatmulOp(), stream, a_device.get(), b_device.get(), out_device.get(),
                                   ops::kCubeMathTypeKeepDtype);

  return out_device.ToFloatFromHalf();
}

// -----------------------------------------------------------------------------
// Host-only checks
// -----------------------------------------------------------------------------

TEST(Matmul950PrShapes, LayerProjectionsAreCoveredByTheSweep) {
  // The five distinct (K, N) pairs the golden layer puts through the cube unit.
  // If one of them is not in the sweep, a projection that only the end-to-end
  // test touches has no unit-level coverage at all, and a failure there has to
  // be bisected by hand.
  struct Projection {
    const char* name;
    int64_t k;
    int64_t n;
  };
  const Projection projections[] = {
      {"q / attn_gate", s::kHidden, s::kQDim},        {"k / v", s::kHidden, s::kKvDim},
      {"o_proj", s::kQDim, s::kHidden},               {"gate / up", s::kHidden, s::kIntermediate},
      {"down", s::kIntermediate, s::kHidden},
  };

  for (const Projection& projection : projections) {
    // Both sides must be a whole number of fp16 bursts for the cube unit's
    // 16-element fractal to be exact.
    EXPECT_EQ(projection.k % s::kFp16ElementsPerBurst, 0) << projection.name << " K";
    EXPECT_EQ(projection.n % s::kFp16ElementsPerBurst, 0) << projection.name << " N";
  }
}

TEST(Matmul950PrReference, MatchesHandComputedCase) {
  // a = [1, 2, 3]                 (m=1, k=3)
  // b_t = [[1, 0, -1],            (n=2, k=3) -> output channel 0
  //        [2, 2,  2]]                          output channel 1
  // out = [1*1 + 2*0 + 3*(-1), 1*2 + 2*2 + 3*2] = [-2, 12]
  //
  // The reference is shared with the 310P and CUDA suites, so this is really a
  // check that the [N, K] weight layout is being read the same way here.
  const std::vector<float> a{1.0f, 2.0f, 3.0f};
  const std::vector<float> b_t{1.0f, 0.0f, -1.0f, 2.0f, 2.0f, 2.0f};

  std::vector<float> out;
  reference::MatmulTransposedB(a, b_t, 1, 3, 2, &out);

  ASSERT_EQ(out.size(), 2u);
  EXPECT_FLOAT_EQ(out[0], -2.0f);
  EXPECT_FLOAT_EQ(out[1], 12.0f);
}

// -----------------------------------------------------------------------------
// Device parity
// -----------------------------------------------------------------------------

class Matmul950PrTest : public ::testing::TestWithParam<std::tuple<int64_t, int64_t>> {
 protected:
  int64_t k() const { return std::get<0>(GetParam()); }
  int64_t n() const { return std::get<1>(GetParam()); }
  int64_t m() const { return s::kDecodeTokenCount; }

  // stddev 1/sqrt(K) for the weights, so the output sits near unit magnitude.
  // At magnitude ~1 an fp16 ULP is ~9.8e-4 and atol=rtol=1e-3 is a one-ULP bar;
  // with N(0,1) weights a 2048-deep dot product lands near magnitude 45, where
  // the tolerance would be measuring fp16 storage instead of the kernel.
  float weight_stddev() const { return 1.0f / std::sqrt(static_cast<float>(k())); }
};

TEST_P(Matmul950PrTest, MatchesCpuReference) {
  REQUIRE_ASCEND_950PR();
  REQUIRE_ACLNN_OP(MatmulOp());

  DeterministicRandom random(0x4d4d554cu);  // "MMUL"

  // Pre-rounded to fp16 so the device and the reference start from
  // bit-identical inputs and the comparison measures arithmetic only.
  const std::vector<float> a = random.NormalHalfExact(static_cast<size_t>(m() * k()), 0.0f, 1.0f);
  const std::vector<float> b_t = random.NormalHalfExact(static_cast<size_t>(n() * k()), 0.0f, weight_stddev());

  const std::vector<float> actual = RunMatmulOnDevice(a, b_t, m(), k(), n());

  std::vector<float> expected;
  reference::MatmulTransposedB(a, b_t, m(), k(), n(), &expected);

  EXPECT_TENSORS_ALLCLOSE(actual, QuantizeToHalf(expected), kFp16DefaultTolerance);
}

TEST_P(Matmul950PrTest, ZeroWeightsProduceZeroOutput) {
  REQUIRE_ASCEND_950PR();
  REQUIRE_ACLNN_OP(MatmulOp());

  // An all-zero weight must give an exactly zero output, with no NaN leaking in
  // from an uninitialised accumulator or a padded K tile. This needs no
  // reference at all, so it isolates the kernel from the host arithmetic.
  DeterministicRandom random(0x5a45524fu);  // "ZERO"

  const std::vector<float> a = random.NormalHalfExact(static_cast<size_t>(m() * k()), 0.0f, 1.0f);
  const std::vector<float> b_t(static_cast<size_t>(n() * k()), 0.0f);

  const std::vector<float> actual = RunMatmulOnDevice(a, b_t, m(), k(), n());

  ASSERT_EQ(actual.size(), static_cast<size_t>(m() * n()));
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual[i])) << "non-finite output at index " << i;
    EXPECT_EQ(actual[i], 0.0f) << "at index " << i;
  }
}

TEST_P(Matmul950PrTest, IsLinearInTheInput) {
  REQUIRE_ASCEND_950PR();
  REQUIRE_ACLNN_OP(MatmulOp());

  // Scaling A by 2 must scale the output by 2. 2 is exact in fp16, so this
  // catches a kernel that folds a scale or a tile offset in the wrong place
  // without depending on the CPU reference.
  DeterministicRandom random(0x4c494e32u);  // "LIN2"

  const std::vector<float> a = random.NormalHalfExact(static_cast<size_t>(m() * k()), 0.0f, 1.0f);
  const std::vector<float> b_t = random.NormalHalfExact(static_cast<size_t>(n() * k()), 0.0f, weight_stddev());

  std::vector<float> doubled(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    doubled[i] = HalfBitsToFloat(FloatToHalfBits(a[i] * 2.0f));
  }

  const std::vector<float> base = RunMatmulOnDevice(a, b_t, m(), k(), n());
  const std::vector<float> scaled = RunMatmulOnDevice(doubled, b_t, m(), k(), n());

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

INSTANTIATE_TEST_SUITE_P(Qwen35, Matmul950PrTest,
                         ::testing::Combine(::testing::ValuesIn(s::LinearInputSizes()),
                                            ::testing::ValuesIn(s::LinearOutputSizes())),
                         MatmulTestName);

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
