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
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "device_buffer.hpp"
#include "fp16.hpp"
#include "random_data.hpp"
#include "test_harness.hpp"
#include "turbo_quant_cpu.h"
#include "turboquant_launch.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace tq = turboquant_ref;
namespace tqh = turboquant_host;
namespace tqr = vllm_ascend::turboquant;

constexpr int64_t kHeadSize = 256;

constexpr int64_t kNumHeads = 8;

constexpr float kCubeAbsTolerance = 1e-5f;

constexpr int64_t kCubeCoreNum = 4;

struct Batch {
  int64_t batch = 0;
  int64_t num_vectors = 0;
  std::vector<float> query;
  std::vector<float> reference;
  std::vector<float> pi_signs;
};

Batch MakeBatch(int64_t batch, uint32_t seed) {
  Batch b;
  b.batch = batch;
  b.num_vectors = batch * kNumHeads;

  DeterministicRandom rng(seed);
  b.query = rng.NormalHalfExact(static_cast<size_t>(b.num_vectors * kHeadSize), 0.0f, 1.0f);

  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(static_cast<int>(kHeadSize));
  b.pi_signs.resize(signs.size());
  for (size_t i = 0; i < signs.size(); ++i) {
    b.pi_signs[i] = static_cast<float>(signs[i]);
  }

  b.reference = b.query;
  for (int64_t v = 0; v < b.num_vectors; ++v) {
    tq::cpu_apply_pi(&b.reference[static_cast<size_t>(v * kHeadSize)], static_cast<int>(kHeadSize), signs.data());
  }
  return b;
}

struct Deviation {
  double max_abs = 0.0;
  size_t worst_at = 0;
  size_t differing = 0;
};

Deviation Compare(const std::vector<float>& got, const std::vector<float>& want) {
  Deviation d;
  for (size_t i = 0; i < want.size(); ++i) {
    const double err = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    if (err != 0.0) {
      ++d.differing;
    }
    if (err > d.max_abs) {
      d.max_abs = err;
      d.worst_at = i;
    }
  }
  return d;
}

std::vector<float> RunRotation(const Batch& b, aclrtStream stream, bool force_aiv,
                               tqr::RotateQPlan* plan_out = nullptr, int64_t core_num_override = 0) {
  DeviceBuffer query = DeviceBuffer::FromHost(FloatToHalf(b.query));
  DeviceBuffer pi_signs = DeviceBuffer::FromHost(b.pi_signs);
  DeviceBuffer h16 = DeviceBuffer::FromHost(tqh::Hadamard16Half());
  DeviceBuffer rot_tables = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, 1));
  DeviceBuffer query_rot = DeviceBuffer::Empty<float>(static_cast<size_t>(b.num_vectors * kHeadSize));

  bool queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&queried);
  int64_t core_num = core_num_override > 0 ? core_num_override : aiv_num / 2;
  if (core_num < 1) {
    core_num = 1;
  }

  tqr::RotateQPlan plan = tqr::PlanRotateQ(b.batch, b.num_vectors, kHeadSize, core_num);
  if (force_aiv && plan.use_cube) {
    int64_t vectors_per_block = (b.num_vectors + core_num - 1) / core_num;
    if (vectors_per_block < 2) {
      vectors_per_block = 2;
    }
    plan.use_cube = false;
    plan.vectors_per_chunk = 1;
    plan.variant = 0;
    plan.vectors_per_block = static_cast<uint32_t>(vectors_per_block);
    plan.block_dim = static_cast<uint32_t>((b.num_vectors + vectors_per_block - 1) / vectors_per_block);
  }
  if (plan_out != nullptr) {
    *plan_out = plan;
  }

  const float inv_sqrt_len = 1.0f / std::sqrt(static_cast<float>(kHeadSize));
  turboquant_rotate_q_impl(AscendType::FP16, stream, plan.block_dim, plan.use_cube, query.get(), pi_signs.get(),
                           h16.get(), rot_tables.get(), query_rot.get(), static_cast<uint32_t>(b.num_vectors),
                           static_cast<uint32_t>(kHeadSize), plan.vectors_per_block, plan.vectors_per_chunk,
                           plan.variant, inv_sqrt_len);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  return query_rot.ToHost<float>();
}

TEST(RotateQPlan, Dispatch) {
  struct Expect {
    int64_t num_tokens;
    int64_t num_vectors;
    int64_t head_size;
    int64_t core_num;
    bool use_cube;
  };
  const Expect cases[] = {
      {1, 4, 256, 16, false},
      {1, 8, 256, 16, false},
      {3, 12, 256, 16, false},
      {2, 16, 256, 16, true},
      {4, 32, 256, 16, true},
      {8, 64, 256, 16, true},
      {32, 256, 256, 16, true},
      {1, 1, 256, 16, false},
      {2, 16, 128, 16, true},
      {2, 16, 64, 16, true},
      {1, 16, 256, 4, false},
      {1, 256, 256, 16, false},
      {2, 16, 256, 32, false},
      {4, 32, 256, 32, true},
      {0, 16, 256, 16, false},
  };

  for (const Expect& e : cases) {
    const tqr::RotateQPlan plan = tqr::PlanRotateQ(e.num_tokens, e.num_vectors, e.head_size, e.core_num);
    EXPECT_EQ(plan.use_cube, e.use_cube)
        << "B=" << e.num_tokens << " N=" << e.num_vectors << " D=" << e.head_size << " cores=" << e.core_num;
    if (e.num_tokens <= 0) {
      EXPECT_EQ(plan.block_dim, 0u) << "an empty batch must not launch";
      continue;
    }
    EXPECT_GT(plan.block_dim, 0u) << "N=" << e.num_vectors;

    if (plan.use_cube) {
      EXPECT_EQ(e.num_vectors % plan.vectors_per_block, 0)
          << "vectors_per_block " << plan.vectors_per_block << " must divide N " << e.num_vectors;
      EXPECT_EQ(plan.vectors_per_block % plan.vectors_per_chunk, 0u)
          << "vectors_per_chunk " << plan.vectors_per_chunk << " must divide the block share "
          << plan.vectors_per_block;
      EXPECT_GE(plan.vectors_per_block, tqr::kRotateQTile);
      EXPECT_EQ(plan.vectors_per_chunk % 2u, 0u);
      EXPECT_NE(plan.variant & static_cast<uint32_t>(tqr::kDualDst), 0u);
      EXPECT_EQ(plan.variant & static_cast<uint32_t>(tqr::kHiLo), 0u);
      EXPECT_LE(static_cast<int64_t>(plan.vectors_per_chunk) * e.head_size, tqr::kRotateQMaxChunkElements);
      EXPECT_EQ(static_cast<int64_t>(plan.block_dim) * plan.vectors_per_block, e.num_vectors);
    } else {
      EXPECT_EQ(plan.vectors_per_chunk, 1u);
      EXPECT_GE(static_cast<int64_t>(plan.block_dim) * plan.vectors_per_block, e.num_vectors);
    }
  }

  const tqr::RotateQPlan single = tqr::PlanRotateQ(8, 64, 256, 16);
  ASSERT_TRUE(single.use_cube);
  EXPECT_EQ(single.variant & static_cast<uint32_t>(tqr::kHiLo), 0u)
      << "the residual Mmad must be opted into, for fp16 and bf16 alike";

  const tqr::RotateQPlan hilo = tqr::PlanRotateQ(8, 64, 256, 16, tqr::RotateQPrecision::kHiLoResidual);
  ASSERT_TRUE(hilo.use_cube);
  EXPECT_NE(hilo.variant & static_cast<uint32_t>(tqr::kHiLo), 0u);

  const tqr::RotateQPlan decode = tqr::PlanRotateQ(1, 64, 256, 1, tqr::RotateQPrecision::kHiLoResidual);
  EXPECT_FALSE(decode.use_cube) << "a single-token decode rotates on the vector units at any head count";
  EXPECT_EQ(decode.variant, 0u);
}

TEST(RotateQVector, SparseBatchIsBitIdenticalToTheReference) {
  REQUIRE_ASCEND_950PR();
  const Batch b = MakeBatch(1, 0x51A7u);
  ASSERT_EQ(b.num_vectors, kNumHeads);

  tqr::RotateQPlan plan;
  const std::vector<float> got = RunRotation(b, AscendTestEnvironment::Instance().stream(), false,
                                             &plan);
  ASSERT_FALSE(plan.use_cube) << "N=" << b.num_vectors << " should not select the Cube path";

  const Deviation d = Compare(got, b.reference);
  std::printf("[ rotate_q ] aiv  N=%lld D=%lld: %zu of %zu elements differ, max |err| %.3e\n",
              static_cast<long long>(b.num_vectors), static_cast<long long>(kHeadSize), d.differing,
              b.reference.size(), d.max_abs);
  EXPECT_EQ(d.differing, 0u) << "first difference at element " << d.worst_at << ", max |err| " << d.max_abs;
}

class RotateQCube : public ::testing::TestWithParam<int64_t> {};

TEST_P(RotateQCube, MatchesTheReference) {
  REQUIRE_ASCEND_950PR();
  const int64_t batch = GetParam();
  const Batch b = MakeBatch(batch, 0x9E37u + static_cast<uint32_t>(batch));

  tqr::RotateQPlan plan;
  const std::vector<float> got = RunRotation(b, AscendTestEnvironment::Instance().stream(), false,
                                             &plan, kCubeCoreNum);
  ASSERT_TRUE(plan.use_cube) << "N=" << b.num_vectors << " should select the Cube path";

  const Deviation d = Compare(got, b.reference);
  std::printf("[ rotate_q ] cube N=%lld D=%lld: %u blocks x %u vectors, chunk %u, variant 0x%x; "
              "%zu of %zu elements differ, max |err| %.3e\n",
              static_cast<long long>(b.num_vectors), static_cast<long long>(kHeadSize), plan.block_dim,
              plan.vectors_per_block, plan.vectors_per_chunk, plan.variant, d.differing, b.reference.size(),
              d.max_abs);

  for (size_t i = 0; i < got.size(); ++i) {
    ASSERT_TRUE(std::isfinite(got[i])) << "non-finite output at element " << i;
  }
  EXPECT_LT(d.max_abs, kCubeAbsTolerance)
      << "worst element " << d.worst_at << " of " << b.reference.size();
}

INSTANTIATE_TEST_SUITE_P(DenseBatches, RotateQCube, ::testing::Values(int64_t{2}, int64_t{8}));

TEST(RotateQCube, AgreesWithTheVectorPathOnIdenticalInput) {
  REQUIRE_ASCEND_950PR();
  const Batch b = MakeBatch(8, 0x2C41u);
  ASSERT_EQ(b.num_vectors, 64);

  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  tqr::RotateQPlan cube_plan;
  const std::vector<float> cube = RunRotation(b, stream, false, &cube_plan, kCubeCoreNum);
  ASSERT_TRUE(cube_plan.use_cube);

  tqr::RotateQPlan aiv_plan;
  const std::vector<float> aiv = RunRotation(b, stream, true, &aiv_plan);
  ASSERT_FALSE(aiv_plan.use_cube);

  const Deviation d = Compare(cube, aiv);
  std::printf("[ rotate_q ] cube vs aiv at N=%lld: %zu of %zu elements differ, max |err| %.3e\n",
              static_cast<long long>(b.num_vectors), d.differing, cube.size(), d.max_abs);
  EXPECT_LT(d.max_abs, kCubeAbsTolerance) << "worst element " << d.worst_at;

  const Deviation ref = Compare(aiv, b.reference);
  EXPECT_EQ(ref.differing, 0u) << "the vector path drifted from the reference at element " << ref.worst_at;
}

}
}
}
