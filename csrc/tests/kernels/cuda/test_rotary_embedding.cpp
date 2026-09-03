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

// Rotary position embedding on the native CUDA C backend, fp16 in / fp16 out.
//
// Scope note: the CUDA kernel implements the "half" (neox / rotate_half)
// layout only, which is the one Qwen3.5 uses and the one the task targets. The
// Ascend suite additionally drives the interleave layout through
// npu_apply_rotary_pos_emb; here the interleave layout is exercised only in the
// host-only reference checks, which is enough to keep the mode parameter of
// reference::ApplyRotaryPosEmb honest but is not a device claim.
//
// The device cases mirror test_rotary_embedding_310p.cpp: parity against the
// reference, position zero as the identity, and pair-norm preservation, over
// the same head dims (64 and 128), the same GQA splits and both rope thetas.

#include <gtest/gtest.h>

#include <cmath>
#include <ostream>
#include <sstream>
#include <string>
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

using reference::RotaryMode;

const char* RotaryModeLabel(RotaryMode mode) { return (mode == RotaryMode::kHalf) ? "half" : "interleave"; }

struct RotaryResult {
  std::vector<float> query;
  std::vector<float> key;
};

// query/key are [num_tokens, num_heads, head_dim] on entry and are rotated in
// place by the kernel, so the result is read back out of the same buffers.
// This matches what npu_apply_rotary_pos_emb does on the Ascend side.
RotaryResult RunApplyRotaryPosEmbOnDevice(const std::vector<float>& query, const std::vector<float>& key,
                                          const std::vector<float>& cos_full,
                                          const std::vector<float>& sin_full, int64_t num_tokens,
                                          int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim) {
  cudaStream_t stream = CUDATestEnvironment::Instance().stream();

  CudaDeviceTensor query_device = CudaDeviceTensor::Half({num_tokens, num_q_heads, head_dim}, query);
  CudaDeviceTensor key_device = CudaDeviceTensor::Half({num_tokens, num_kv_heads, head_dim}, key);
  CudaDeviceTensor cos_device = CudaDeviceTensor::Half({num_tokens, head_dim}, cos_full);
  CudaDeviceTensor sin_device = CudaDeviceTensor::Half({num_tokens, head_dim}, sin_full);

  cuda::LaunchApplyRotaryPosEmbHalfMode(query_device.half_data(), cos_device.half_data(),
                                        sin_device.half_data(), num_tokens, num_q_heads, head_dim,
                                        head_dim, stream);
  cuda::LaunchApplyRotaryPosEmbHalfMode(key_device.half_data(), cos_device.half_data(),
                                        sin_device.half_data(), num_tokens, num_kv_heads, head_dim,
                                        head_dim, stream);
  CUDATestEnvironment::Instance().device().SynchronizeStream();

  RotaryResult result;
  result.query = query_device.ToFloatFromHalf();
  result.key = key_device.ToFloatFromHalf();
  return result;
}

// L2 norm of each rotary pair. A rotation preserves it exactly, which is a
// property check that needs no reference implementation.
std::vector<float> PairNorms(const std::vector<float>& x, int64_t num_tokens, int64_t num_heads,
                             int64_t head_dim, RotaryMode mode) {
  const int64_t half = head_dim / 2;
  std::vector<float> norms;
  norms.reserve(static_cast<size_t>(num_tokens * num_heads * half));

  for (int64_t token = 0; token < num_tokens; ++token) {
    for (int64_t head = 0; head < num_heads; ++head) {
      const size_t base = static_cast<size_t>((token * num_heads + head) * head_dim);
      for (int64_t k = 0; k < half; ++k) {
        const int64_t first = (mode == RotaryMode::kHalf) ? k : (2 * k);
        const int64_t second = (mode == RotaryMode::kHalf) ? (k + half) : (2 * k + 1);
        const float a = x[base + static_cast<size_t>(first)];
        const float b = x[base + static_cast<size_t>(second)];
        norms.push_back(std::sqrt(a * a + b * b));
      }
    }
  }
  return norms;
}

// -----------------------------------------------------------------------------
// Host-only checks
// -----------------------------------------------------------------------------

TEST(RotaryReference, PositionZeroIsTheIdentity) {
  // cos(0) = 1 and sin(0) = 0, so position 0 must leave the vector untouched
  // regardless of the rotary layout.
  const int64_t head_dim = 128;
  const std::vector<float> cache = reference::BuildCosSinCache(8, head_dim, shapes::kRopeThetaDefault);
  const std::vector<int32_t> positions = {0};

  DeterministicRandom random(0x524f5045u);  // "ROPE"
  const std::vector<float> x = random.NormalHalfExact(static_cast<size_t>(head_dim), 0.0f, 1.0f);

  for (RotaryMode mode : {RotaryMode::kHalf, RotaryMode::kInterleave}) {
    std::vector<float> cos_full;
    std::vector<float> sin_full;
    reference::GatherFullCosSin(cache, positions, head_dim, mode, &cos_full, &sin_full);

    std::vector<float> out;
    reference::ApplyRotaryPosEmb(x, cos_full, sin_full, 1, 1, head_dim, head_dim, mode, &out);

    for (int64_t i = 0; i < head_dim; ++i) {
      EXPECT_NEAR(out[static_cast<size_t>(i)], x[static_cast<size_t>(i)], 1e-6f)
          << "mode=" << RotaryModeLabel(mode) << " index=" << i;
    }
  }
}

TEST(RotaryReference, HalfAndInterleaveDisagree) {
  // If the two layouts ever produced the same answer, targeting only the half
  // layout on the device would be an untested restriction rather than a
  // deliberate one.
  const int64_t head_dim = 64;
  const std::vector<float> cache = reference::BuildCosSinCache(64, head_dim, shapes::kRopeThetaDefault);
  const std::vector<int32_t> positions = {7};

  DeterministicRandom random(0x44494646u);  // "DIFF"
  const std::vector<float> x = random.NormalHalfExact(static_cast<size_t>(head_dim), 0.0f, 1.0f);

  std::vector<float> cos_half;
  std::vector<float> sin_half;
  reference::GatherFullCosSin(cache, positions, head_dim, RotaryMode::kHalf, &cos_half, &sin_half);
  std::vector<float> out_half;
  reference::ApplyRotaryPosEmb(x, cos_half, sin_half, 1, 1, head_dim, head_dim, RotaryMode::kHalf,
                               &out_half);

  std::vector<float> cos_interleave;
  std::vector<float> sin_interleave;
  reference::GatherFullCosSin(cache, positions, head_dim, RotaryMode::kInterleave, &cos_interleave,
                              &sin_interleave);
  std::vector<float> out_interleave;
  reference::ApplyRotaryPosEmb(x, cos_interleave, sin_interleave, 1, 1, head_dim, head_dim,
                               RotaryMode::kInterleave, &out_interleave);

  bool any_difference = false;
  for (size_t i = 0; i < out_half.size(); ++i) {
    if (std::fabs(out_half[i] - out_interleave[i]) > 1e-4f) {
      any_difference = true;
      break;
    }
  }
  EXPECT_TRUE(any_difference);
}

TEST(RotaryShapes, CoveredHeadDimsAreEven) {
  // The kernel splits the head in two, so an odd width has no meaning. 64 and
  // 128 are also exactly the dims the Ascend rotary op accepts, which keeps the
  // two backends comparable.
  for (int64_t head_dim : shapes::RotaryHeadDims()) {
    EXPECT_TRUE(head_dim == 64 || head_dim == 128) << "unexpected head dim " << head_dim;
    EXPECT_EQ(head_dim % 2, 0);
  }
}

// -----------------------------------------------------------------------------
// Device parity
// -----------------------------------------------------------------------------

struct RotaryCase {
  int64_t num_tokens;
  int64_t head_dim;
  int64_t num_q_heads;
  int64_t num_kv_heads;
  double theta;
};

// See the note in test_paged_attention.cpp: without this GTest names each case
// after a byte dump of the struct. The text matches what RotaryTestName below
// produces, so the name is the same whether it is read out of GTest or out of
// ctest, and stays usable as a -R filter.
void PrintTo(const RotaryCase& test_case, std::ostream* stream) {
  *stream << "tokens" << test_case.num_tokens << "_d" << test_case.head_dim << "_q"
          << test_case.num_q_heads << "_kv" << test_case.num_kv_heads << "_half_theta"
          << static_cast<int64_t>(test_case.theta);
}

class RotaryEmbeddingCudaTest : public ::testing::TestWithParam<RotaryCase> {};

TEST_P(RotaryEmbeddingCudaTest, MatchesCpuReference) {
  REQUIRE_CUDA_DEVICE();

  const RotaryCase& test_case = GetParam();
  DeterministicRandom random(0x51524f50u);  // "QROP"

  const std::vector<float> query = random.NormalHalfExact(
      static_cast<size_t>(test_case.num_tokens * test_case.num_q_heads * test_case.head_dim), 0.0f, 1.0f);
  const std::vector<float> key = random.NormalHalfExact(
      static_cast<size_t>(test_case.num_tokens * test_case.num_kv_heads * test_case.head_dim), 0.0f, 1.0f);

  // Positions are scattered rather than sequential so a kernel that ignores the
  // per-token cos/sin row cannot pass by accident.
  std::vector<int32_t> positions(static_cast<size_t>(test_case.num_tokens));
  for (int64_t i = 0; i < test_case.num_tokens; ++i) {
    positions[static_cast<size_t>(i)] = static_cast<int32_t>((i * 37 + 11) % shapes::kMaxPositionEmbeddings);
  }

  const std::vector<float> cache =
      reference::BuildCosSinCache(shapes::kMaxPositionEmbeddings, test_case.head_dim, test_case.theta);

  std::vector<float> cos_full;
  std::vector<float> sin_full;
  reference::GatherFullCosSin(cache, positions, test_case.head_dim, RotaryMode::kHalf, &cos_full,
                              &sin_full);

  // The device reads cos/sin as fp16, so the reference has to use the same
  // rounded angles or the comparison measures the cache precision instead of
  // the kernel.
  const std::vector<float> cos_half_exact = QuantizeToHalf(cos_full);
  const std::vector<float> sin_half_exact = QuantizeToHalf(sin_full);

  const RotaryResult actual =
      RunApplyRotaryPosEmbOnDevice(query, key, cos_half_exact, sin_half_exact, test_case.num_tokens,
                                   test_case.num_q_heads, test_case.num_kv_heads, test_case.head_dim);

  std::vector<float> expected_query;
  reference::ApplyRotaryPosEmb(query, cos_half_exact, sin_half_exact, test_case.num_tokens,
                               test_case.num_q_heads, test_case.head_dim, test_case.head_dim,
                               RotaryMode::kHalf, &expected_query);
  std::vector<float> expected_key;
  reference::ApplyRotaryPosEmb(key, cos_half_exact, sin_half_exact, test_case.num_tokens,
                               test_case.num_kv_heads, test_case.head_dim, test_case.head_dim,
                               RotaryMode::kHalf, &expected_key);

  EXPECT_TENSORS_ALLCLOSE(actual.query, QuantizeToHalf(expected_query), kFp16DefaultTolerance);
  EXPECT_TENSORS_ALLCLOSE(actual.key, QuantizeToHalf(expected_key), kFp16DefaultTolerance);
}

TEST_P(RotaryEmbeddingCudaTest, PositionZeroLeavesTheInputUnchanged) {
  REQUIRE_CUDA_DEVICE();

  const RotaryCase& test_case = GetParam();
  DeterministicRandom random(0x5a45524fu);  // "ZERO"

  const std::vector<float> query = random.NormalHalfExact(
      static_cast<size_t>(test_case.num_tokens * test_case.num_q_heads * test_case.head_dim), 0.0f, 1.0f);
  const std::vector<float> key = random.NormalHalfExact(
      static_cast<size_t>(test_case.num_tokens * test_case.num_kv_heads * test_case.head_dim), 0.0f, 1.0f);

  const std::vector<int32_t> positions(static_cast<size_t>(test_case.num_tokens), 0);
  const std::vector<float> cache =
      reference::BuildCosSinCache(shapes::kMaxPositionEmbeddings, test_case.head_dim, test_case.theta);

  std::vector<float> cos_full;
  std::vector<float> sin_full;
  reference::GatherFullCosSin(cache, positions, test_case.head_dim, RotaryMode::kHalf, &cos_full,
                              &sin_full);

  const RotaryResult actual =
      RunApplyRotaryPosEmbOnDevice(query, key, cos_full, sin_full, test_case.num_tokens,
                                   test_case.num_q_heads, test_case.num_kv_heads, test_case.head_dim);

  EXPECT_TENSORS_ALLCLOSE(actual.query, query, kFp16DefaultTolerance);
  EXPECT_TENSORS_ALLCLOSE(actual.key, key, kFp16DefaultTolerance);
}

TEST_P(RotaryEmbeddingCudaTest, PreservesPairNormsOnDevice) {
  REQUIRE_CUDA_DEVICE();

  const RotaryCase& test_case = GetParam();
  DeterministicRandom random(0x504e524du);  // "PNRM"

  const std::vector<float> query = random.NormalHalfExact(
      static_cast<size_t>(test_case.num_tokens * test_case.num_q_heads * test_case.head_dim), 0.0f, 1.0f);
  const std::vector<float> key = random.NormalHalfExact(
      static_cast<size_t>(test_case.num_tokens * test_case.num_kv_heads * test_case.head_dim), 0.0f, 1.0f);

  std::vector<int32_t> positions(static_cast<size_t>(test_case.num_tokens));
  for (int64_t i = 0; i < test_case.num_tokens; ++i) {
    positions[static_cast<size_t>(i)] = static_cast<int32_t>(i * 13 + 5);
  }

  const std::vector<float> cache =
      reference::BuildCosSinCache(shapes::kMaxPositionEmbeddings, test_case.head_dim, test_case.theta);
  std::vector<float> cos_full;
  std::vector<float> sin_full;
  reference::GatherFullCosSin(cache, positions, test_case.head_dim, RotaryMode::kHalf, &cos_full,
                              &sin_full);

  const RotaryResult actual = RunApplyRotaryPosEmbOnDevice(
      query, key, QuantizeToHalf(cos_full), QuantizeToHalf(sin_full), test_case.num_tokens,
      test_case.num_q_heads, test_case.num_kv_heads, test_case.head_dim);

  const std::vector<float> before =
      PairNorms(query, test_case.num_tokens, test_case.num_q_heads, test_case.head_dim, RotaryMode::kHalf);
  const std::vector<float> after = PairNorms(actual.query, test_case.num_tokens, test_case.num_q_heads,
                                             test_case.head_dim, RotaryMode::kHalf);

  ASSERT_EQ(before.size(), after.size());
  // Looser than the parity bound: the norm is a product of two fp16 roundings
  // plus the sqrt, so it accumulates a little more than a single element does.
  const Tolerance norm_tolerance{5e-3, 5e-3, "pair norm through two fp16 roundings plus a square root"};
  EXPECT_TENSORS_ALLCLOSE(after, before, norm_tolerance);
}

std::string RotaryTestName(const ::testing::TestParamInfo<RotaryCase>& info) {
  std::ostringstream name;
  name << "tokens" << info.param.num_tokens << "_d" << info.param.head_dim << "_q" << info.param.num_q_heads
       << "_kv" << info.param.num_kv_heads << "_half_theta" << static_cast<int64_t>(info.param.theta);
  return name.str();
}

INSTANTIATE_TEST_SUITE_P(
    Qwen35, RotaryEmbeddingCudaTest,
    ::testing::Values(
        // head_dim 128 with the two Qwen3.5 GQA splits.
        RotaryCase{1, 128, 28, 4, shapes::kRopeThetaDefault},
        RotaryCase{16, 128, 28, 4, shapes::kRopeThetaDefault},
        RotaryCase{128, 128, 28, 4, shapes::kRopeThetaExtended},
        RotaryCase{16, 128, 32, 8, shapes::kRopeThetaDefault},
        RotaryCase{128, 128, 32, 8, shapes::kRopeThetaExtended},
        // head_dim 64, the other size the Ascend rotary op accepts.
        RotaryCase{1, 64, 16, 2, shapes::kRopeThetaDefault},
        RotaryCase{32, 64, 16, 2, shapes::kRopeThetaDefault}),
    RotaryTestName);

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
