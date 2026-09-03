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

// MatMul on the native CUDA C backend, fp16 in / fp16 out, with a transposed B.
//
// The device cases are the same three the Ascend suite runs in
// test_matmul_310p.cpp - parity against reference::MatmulTransposedB, an
// all-zero weight, and linearity in the input - over the same Qwen3.5
// projection shapes and with the same seeds and tolerances. What changes is
// only how the operator is reached: CublasGemmFp16 instead of a dlsym'd
// aclnnMatmul.
//
// This is the one operator in the CUDA backend that is not a hand-written
// kernel. The reasoning is in cuda_runtime.hpp; the short version is that a
// naive GEMM would be checking the kernel this suite just wrote, whereas cuBLAS
// is what a CUDA deployment actually runs for a linear projection.
//
// The two choices the tolerance depends on carry over unchanged from the Ascend
// file:
//
// 1. B is presented transposed. A Linear layer stores its weight as
//    [out_features, in_features] = [N, K] and computes x @ W^T. The test
//    uploads that [N, K] buffer and passes transpose_b, which becomes
//    CUBLAS_OP_T rather than a materialised transpose - the same thing torch
//    does.
//
// 2. Inputs are scaled so the output lands near unit magnitude. A K=4096 dot
//    product of N(0,1) terms has magnitude ~sqrt(K)=64, where one fp16 ULP is
//    0.0625 -- the output rounding alone would then exceed atol=1e-3 and the
//    test would be measuring fp16 storage, not the arithmetic. Drawing B with
//    stddev 1/sqrt(K) (which is also how the weights are actually initialised)
//    keeps the result near 1.0, where one fp16 ULP is ~9.8e-4 and atol=1e-3 is
//    a meaningful bar.
//
// The host-only checks are repeated from the Ascend file rather than shared,
// because a CUDA build never compiles the Ascend test files and the reference
// deserves to be checked in whichever build is running.
//
// NOT COVERED: M > 1. Decode is M=1, which is what the requirement targets, but
// a GEMV does not exercise the tiling a batched prefill would. A prefill-shaped
// case (M = 32 / 128 against the same weights) is the obvious next addition.

#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "cpu_reference.hpp"
#include "cuda_device_tensor.hpp"
#include "cuda_runtime.hpp"
#include "fp16.hpp"
#include "qwen_shapes.hpp"
#include "random_data.hpp"
#include "tensor_compare.hpp"

namespace vllm_ascend {
namespace test {
namespace {

// Runs the GEMM with `b_t` supplied in Linear [N, K] layout, and returns the
// fp16 [M, N] output widened back to float.
std::vector<float> RunMatmulOnDevice(const std::vector<float>& a, const std::vector<float>& b_t, int64_t m,
                                     int64_t k, int64_t n) {
  CUDADevice& device = CUDATestEnvironment::Instance().device();

  CudaDeviceTensor a_device = CudaDeviceTensor::Half({m, k}, a);
  CudaDeviceTensor b_device = CudaDeviceTensor::Half({n, k}, b_t);
  CudaDeviceTensor out_device = CudaDeviceTensor::HalfEmpty({m, n});

  CublasGemmFp16(device.cublas_handle(), /*transpose_a=*/false, /*transpose_b=*/true, m, n, k,
                 a_device.half_data(), b_device.half_data(), out_device.half_data());
  device.SynchronizeStream();

  return out_device.ToFloatFromHalf();
}

// Row-major [rows, cols] -> row-major [cols, rows]. Only used to build the
// alternative operand layouts the transposition flags describe.
std::vector<float> TransposeRowMajor(const std::vector<float>& src, int64_t rows, int64_t cols) {
  std::vector<float> out(src.size());
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      out[static_cast<size_t>(c * rows + r)] = src[static_cast<size_t>(r * cols + c)];
    }
  }
  return out;
}

// -----------------------------------------------------------------------------
// Host-only checks. These run on a machine with no GPU attached, so a mistake
// in the reference itself is caught before any hardware is involved.
// -----------------------------------------------------------------------------

TEST(MatmulReference, IdentityWeightReproducesInput) {
  // b_t = I in [n, k] layout with n == k means out[0][j] = a[0][j].
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
  // a = [1, 2, 3]                 (m=1, k=3)
  // b_t = [[1, 0, -1],            (n=2, k=3) -> output channel 0
  //        [2, 2,  2]]                          output channel 1
  // out = [1*1 + 2*0 + 3*(-1), 1*2 + 2*2 + 3*2] = [-2, 12]
  const std::vector<float> a{1.0f, 2.0f, 3.0f};
  const std::vector<float> b_t{1.0f, 0.0f, -1.0f, 2.0f, 2.0f, 2.0f};

  std::vector<float> out;
  reference::MatmulTransposedB(a, b_t, 1, 3, 2, &out);

  ASSERT_EQ(out.size(), 2u);
  EXPECT_FLOAT_EQ(out[0], -2.0f);
  EXPECT_FLOAT_EQ(out[1], 12.0f);
}

TEST(MatmulReference, IsLinearInTheInput) {
  // matmul(a1 + a2, B) == matmul(a1, B) + matmul(a2, B). Catches an indexing
  // error that happens to be self-consistent on a single input.
  const int64_t k = 32;
  const int64_t n = 16;
  DeterministicRandom random(0x4c494e21u);  // "LIN!"

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

// -----------------------------------------------------------------------------
// The wrapper itself
// -----------------------------------------------------------------------------

TEST(CublasGemmFp16Wrapper, HonoursEveryTranspositionFlag) {
  REQUIRE_CUDA_DEVICE();

  // The row-major-to-column-major identity in CublasGemmFp16 is the one place a
  // sign or a leading dimension can be wrong without any Qwen shape noticing:
  // at M=1 the two A layouts are the same bytes, and a square-ish B hides a
  // swapped ld. So this runs all four flag combinations over a shape where m, k
  // and n are pairwise different and every layout is genuinely distinct.
  const int64_t m = 3;
  const int64_t k = 4;
  const int64_t n = 2;

  DeterministicRandom random(0x54524e53u);  // "TRNS"

  const std::vector<float> a = random.NormalHalfExact(static_cast<size_t>(m * k), 0.0f, 1.0f);
  const std::vector<float> b_t = random.NormalHalfExact(static_cast<size_t>(n * k), 0.0f, 0.5f);

  std::vector<float> expected;
  reference::MatmulTransposedB(a, b_t, m, k, n, &expected);
  const std::vector<float> expected_half = QuantizeToHalf(expected);

  // a is stored [m, k], or [k, m] transposed; b_t is stored [n, k], which is
  // the transposed form, or [k, n] when it is not.
  const std::vector<float> a_transposed = TransposeRowMajor(a, m, k);
  const std::vector<float> b = TransposeRowMajor(b_t, n, k);

  CUDADevice& device = CUDATestEnvironment::Instance().device();

  for (int flags = 0; flags < 4; ++flags) {
    const bool transpose_a = (flags & 1) != 0;
    const bool transpose_b = (flags & 2) != 0;
    SCOPED_TRACE(::testing::Message() << "transpose_a=" << transpose_a << " transpose_b=" << transpose_b);

    CudaDeviceTensor a_device = transpose_a ? CudaDeviceTensor::Half({k, m}, a_transposed)
                                            : CudaDeviceTensor::Half({m, k}, a);
    CudaDeviceTensor b_device =
        transpose_b ? CudaDeviceTensor::Half({n, k}, b_t) : CudaDeviceTensor::Half({k, n}, b);
    CudaDeviceTensor out_device = CudaDeviceTensor::HalfEmpty({m, n});

    CublasGemmFp16(device.cublas_handle(), transpose_a, transpose_b, m, n, k, a_device.half_data(),
                   b_device.half_data(), out_device.half_data());
    device.SynchronizeStream();

    EXPECT_TENSORS_ALLCLOSE(out_device.ToFloatFromHalf(), expected_half, kFp16DefaultTolerance);
  }
}

// -----------------------------------------------------------------------------
// Device parity
// -----------------------------------------------------------------------------

class MatmulCudaTest : public ::testing::TestWithParam<std::tuple<int64_t, int64_t>> {
 protected:
  int64_t k() const { return std::get<0>(GetParam()); }
  int64_t n() const { return std::get<1>(GetParam()); }
  int64_t m() const { return shapes::kDecodeTokenCount; }

  // stddev 1/sqrt(K) for the weights, so the output sits near unit magnitude.
  // See the tolerance discussion at the top of this file.
  float weight_stddev() const { return 1.0f / std::sqrt(static_cast<float>(k())); }
};

TEST_P(MatmulCudaTest, MatchesCpuReference) {
  REQUIRE_CUDA_DEVICE();

  DeterministicRandom random(0x4d4d554cu);  // "MMUL"

  // Pre-rounded to fp16 so the device and the reference start from
  // bit-identical inputs and the comparison measures arithmetic only.
  const std::vector<float> a = random.NormalHalfExact(static_cast<size_t>(m() * k()), 0.0f, 1.0f);
  const std::vector<float> b_t =
      random.NormalHalfExact(static_cast<size_t>(n() * k()), 0.0f, weight_stddev());

  const std::vector<float> actual = RunMatmulOnDevice(a, b_t, m(), k(), n());

  std::vector<float> expected;
  reference::MatmulTransposedB(a, b_t, m(), k(), n(), &expected);

  // The device writes fp16, so the reference is rounded the same way before
  // comparing; otherwise the output quantisation alone eats the tolerance.
  EXPECT_TENSORS_ALLCLOSE(actual, QuantizeToHalf(expected), kFp16DefaultTolerance);
}

TEST_P(MatmulCudaTest, ZeroWeightsProduceZeroOutput) {
  REQUIRE_CUDA_DEVICE();

  // An all-zero weight must give an exactly zero output, with no NaN leaking in
  // from an uninitialised accumulator or a padded K tile. This needs no
  // reference at all, so it isolates the operator from the host arithmetic.
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

TEST_P(MatmulCudaTest, IsLinearInTheInput) {
  REQUIRE_CUDA_DEVICE();

  // Scaling A by 2 must scale the output by 2. 2 is exact in fp16, so this
  // catches an operator that folds a scale or a tile offset in the wrong place
  // without depending on the CPU reference.
  DeterministicRandom random(0x4c494e32u);  // "LIN2"

  const std::vector<float> a = random.NormalHalfExact(static_cast<size_t>(m() * k()), 0.0f, 1.0f);
  const std::vector<float> b_t =
      random.NormalHalfExact(static_cast<size_t>(n() * k()), 0.0f, weight_stddev());

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

INSTANTIATE_TEST_SUITE_P(Qwen35, MatmulCudaTest,
                         ::testing::Combine(::testing::ValuesIn(shapes::LinearInputSizes()),
                                            ::testing::ValuesIn(shapes::LinearOutputSizes())),
                         MatmulTestName);

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
