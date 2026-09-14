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

// npu_turboquant_rotate_q, both dispatch paths, against the CPU reference.
//
// This is the gate on the pre-rotated Q refactor. The decode kernels no longer
// contain a Walsh-Hadamard transform of any kind, so if this operator is wrong
// every downstream number is wrong in a way that looks like a codec fidelity
// failure -- a collapsed cosine with no fault and no exception dump. Checking
// the rotation here, in isolation and against turbo_quant_cpu.h, is what keeps
// that diagnosis one test away instead of a session away.
//
// WHAT IS CHECKED, AND WHY EACH BOUND IS WHAT IT IS
//
//   1. Dispatch. PlanRotateQ is a pure function of (N, D, cores); the cases pin
//      which path each shape selects and that the two divisibility invariants
//      the Cube kernel relies on actually hold. A plan that silently fell back
//      to the vector path would still pass every numerical check below, and the
//      Cube path would then be untested while appearing tested.
//
//   2. The vector path is BIT-IDENTICAL to the reference. It runs the same
//      TurboQuantCodec4::ApplyPi the decode used to call per head, on the same
//      fp32 data, in the same order -- so the only licensed difference against
//      cpu_apply_pi is fp32 rounding that both sides do identically. The bound
//      is exact equality, not a tolerance. This is the property that makes the
//      vector path the reference the Cube path is judged against.
//
//   3. The Cube path is exact for a half query. H_16 holds only +-1, both exact
//      in fp16; the sign multiply is by +-1; so a half input survives the
//      fp32 -> fp16 operand cast unchanged and the single Mmad introduces no
//      error at all. Its residual stages are the same fp32 Add/Sub the vector
//      path runs. What is left is summation ORDER: the Cube evaluates strides
//      1, 2, 4, 8 as a matrix product and the reference as four butterfly
//      passes, which reassociates the fp32 adds. kCubeAbsTolerance is sized for
//      that and nothing else.
//
//   4. Both paths agree with each other, at the same bound, on the same input.
//
// A pass here is seconds to a couple of minutes on the camodel -- there is no
// decode in it -- which is what makes it the first thing to run after any
// change to the rotation.

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

// The decode's head size. The sim tier runs 64 elsewhere for speed, but the
// rotation's cost and its factorisation both scale with D and 256 is the shape
// the Cube path exists for; a whole case here is still under a minute.
constexpr int64_t kHeadSize = 256;

// Query heads per token, so a batch of B gives N = B * kNumHeads vectors.
constexpr int64_t kNumHeads = 8;

/*
 * Reassociation only. The Cube evaluates the four lowest butterfly stages as
 * one 16x16 matrix product with an fp32 accumulator; the reference evaluates
 * them as four passes of pairwise adds. Both are exact in their own operand
 * grid -- H_16 is +-1 in fp16, the input is fp16-exact, the accumulate is fp32 --
 * so the whole difference is the order sixteen fp32 terms are summed in.
 *
 * At unit-scale data and D = 256 that is a handful of ulps carried through five
 * further exact stages. 1e-5 is roughly two orders above what a run actually
 * shows and two orders below the 1e-3 at which the fp16 operand grid itself
 * would be implicated -- so a failure here means a layout or a stride bug, not
 * rounding.
 */
constexpr float kCubeAbsTolerance = 1e-5f;

// One shape's worth of inputs, and the reference rotation of them.
struct Batch {
  int64_t batch = 0;
  int64_t num_vectors = 0;
  std::vector<float> query;      // [N, D], fp16-exact
  std::vector<float> reference;  // Pi applied per vector
  std::vector<float> pi_signs;
};

Batch MakeBatch(int64_t batch, uint32_t seed) {
  Batch b;
  b.batch = batch;
  b.num_vectors = batch * kNumHeads;

  // fp16-exact, so the host reference and the device see the same bits: the
  // query reaches the kernel as fp16 and is cast up, and a value that lost
  // precision in that round trip would show up as a rotation error.
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

// Largest absolute difference and where, plus how many elements differ at all.
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

/*
 * Run the operator over one batch and read the rotated query back.
 *
 * `force_aiv` drives the vector path at a shape the host would otherwise send
 * to the Cube, which is how the two are compared on identical input. It bypasses
 * PlanRotateQ rather than reimplementing it: the plan the production host
 * computes is checked separately, in RotateQPlanDispatch.
 */
std::vector<float> RunRotation(const Batch& b, aclrtStream stream, bool force_aiv,
                               tqr::RotateQPlan* plan_out = nullptr) {
  DeviceBuffer query = DeviceBuffer::FromHost(FloatToHalf(b.query));
  DeviceBuffer pi_signs = DeviceBuffer::FromHost(b.pi_signs);
  DeviceBuffer h16 = DeviceBuffer::FromHost(tqh::Hadamard16Half());
  DeviceBuffer rot_tables = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, 1));
  DeviceBuffer query_rot = DeviceBuffer::Empty<float>(static_cast<size_t>(b.num_vectors * kHeadSize));

  bool queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&queried);
  int64_t core_num = aiv_num / 2;
  if (core_num < 1) {
    core_num = 1;
  }

  tqr::RotateQPlan plan = tqr::PlanRotateQ(b.num_vectors, kHeadSize, core_num, /*input_exact_in_half=*/true);
  if (force_aiv && plan.use_cube) {
    // The shape the vector path would have been given had the gate rejected
    // this batch: the planner's own fallback arithmetic, restated rather than
    // called, because PlanRotateQ has no argument that means "not the Cube".
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

// --- 1. dispatch -------------------------------------------------------------

TEST(RotateQPlan, Dispatch) {
  // Host-side arithmetic only; no device needed, so it runs even where the
  // camodel does not.
  struct Expect {
    int64_t num_vectors;
    int64_t head_size;
    int64_t core_num;
    bool use_cube;
  };
  // 16 cube cores, i.e. a 32-AIV part -- the shape the benchmark device
  // reports. The gate is divisibility, not core count, so the expectations do
  // not move with it.
  const Expect cases[] = {
      {4, 256, 16, false},     // sparse: below the tile
      {8, 256, 16, false},     // B=1, H=8: not a whole tile
      {12, 256, 16, false},    // not a multiple of 16
      {16, 256, 16, true},     // MLA at B=1, H_Q=16
      {32, 256, 16, true},     // GLM/LLaMA at B=1, H_Q=32
      {64, 256, 16, true},     // B=4, H_Q=16
      {256, 256, 16, true},    // B=16, H_Q=16
      {1, 256, 16, false},     // degenerate
      {16, 128, 16, true},     // a head size whose chunk can hold more
      {16, 64, 16, true},
  };

  for (const Expect& e : cases) {
    const tqr::RotateQPlan plan = tqr::PlanRotateQ(e.num_vectors, e.head_size, e.core_num, true);
    EXPECT_EQ(plan.use_cube, e.use_cube)
        << "N=" << e.num_vectors << " D=" << e.head_size << " cores=" << e.core_num;
    EXPECT_GT(plan.block_dim, 0u) << "N=" << e.num_vectors;

    if (plan.use_cube) {
      // The two invariants the Cube kernel relies on and does not check.
      EXPECT_EQ(e.num_vectors % plan.vectors_per_block, 0)
          << "vectors_per_block " << plan.vectors_per_block << " must divide N " << e.num_vectors;
      EXPECT_EQ(plan.vectors_per_block % plan.vectors_per_chunk, 0u)
          << "vectors_per_chunk " << plan.vectors_per_chunk << " must divide the block share "
          << plan.vectors_per_block;
      // Dense enough for the factorisation to pay, and even so the
      // dual-destination Fixpipe splits on a vector boundary.
      EXPECT_GE(plan.vectors_per_block, tqr::kRotateQTile);
      EXPECT_EQ(plan.vectors_per_chunk % 2u, 0u);
      EXPECT_NE(plan.variant & static_cast<uint32_t>(tqr::kDualDst), 0u);
      // A half query is exact in the Cube's operand grid, so the second Mmad
      // is not issued.
      EXPECT_EQ(plan.variant & static_cast<uint32_t>(tqr::kHiLo), 0u);
      // The chunk has to fit the UB budget the kernel sizes against.
      EXPECT_LE(static_cast<int64_t>(plan.vectors_per_chunk) * e.head_size, tqr::kRotateQMaxChunkElements);
      // Every block's share is covered, and no block is launched with nothing.
      EXPECT_EQ(static_cast<int64_t>(plan.block_dim) * plan.vectors_per_block, e.num_vectors);
    } else {
      EXPECT_EQ(plan.vectors_per_chunk, 1u);
      EXPECT_GE(static_cast<int64_t>(plan.block_dim) * plan.vectors_per_block, e.num_vectors);
    }
  }

  // A bfloat16 query is not exactly representable in half, so the Cube path
  // must issue the second Mmad.
  const tqr::RotateQPlan bf16 = tqr::PlanRotateQ(64, 256, 16, /*input_exact_in_half=*/false);
  ASSERT_TRUE(bf16.use_cube);
  EXPECT_NE(bf16.variant & static_cast<uint32_t>(tqr::kHiLo), 0u);
}

// --- 2. the vector path, bit-identical ---------------------------------------

TEST(RotateQVector, SparseBatchIsBitIdenticalToTheReference) {
  REQUIRE_ASCEND_950PR();
  // N = 4: below the tile, so this is the shape the gate actually sends here.
  const Batch b = MakeBatch(/*batch=*/1, 0x51A7u);
  ASSERT_EQ(b.num_vectors, kNumHeads);

  tqr::RotateQPlan plan;
  const std::vector<float> got = RunRotation(b, AscendTestEnvironment::Instance().stream(), /*force_aiv=*/false,
                                             &plan);
  ASSERT_FALSE(plan.use_cube) << "N=" << b.num_vectors << " should not select the Cube path";

  const Deviation d = Compare(got, b.reference);
  std::printf("[ rotate_q ] aiv  N=%lld D=%lld: %zu of %zu elements differ, max |err| %.3e\n",
              static_cast<long long>(b.num_vectors), static_cast<long long>(kHeadSize), d.differing,
              b.reference.size(), d.max_abs);
  // Exact. The device runs the reference's own algorithm on the same fp32
  // values in the same order; anything non-zero here is a real defect.
  EXPECT_EQ(d.differing, 0u) << "first difference at element " << d.worst_at << ", max |err| " << d.max_abs;
}

// --- 3. the Cube path --------------------------------------------------------

class RotateQCube : public ::testing::TestWithParam<int64_t> {};

TEST_P(RotateQCube, MatchesTheReference) {
  REQUIRE_ASCEND_950PR();
  const int64_t batch = GetParam();
  const Batch b = MakeBatch(batch, 0x9E37u + static_cast<uint32_t>(batch));

  tqr::RotateQPlan plan;
  const std::vector<float> got = RunRotation(b, AscendTestEnvironment::Instance().stream(), /*force_aiv=*/false,
                                             &plan);
  ASSERT_TRUE(plan.use_cube) << "N=" << b.num_vectors << " should select the Cube path";

  const Deviation d = Compare(got, b.reference);
  std::printf("[ rotate_q ] cube N=%lld D=%lld: %u blocks x %u vectors, chunk %u, variant 0x%x; "
              "%zu of %zu elements differ, max |err| %.3e\n",
              static_cast<long long>(b.num_vectors), static_cast<long long>(kHeadSize), plan.block_dim,
              plan.vectors_per_block, plan.vectors_per_chunk, plan.variant, d.differing, b.reference.size(),
              d.max_abs);

  // Every element finite: an fp16 operand that overflowed, or a fractal pad left
  // undefined, reaches the Fixpipe as inf/nan rather than as a large number.
  for (size_t i = 0; i < got.size(); ++i) {
    ASSERT_TRUE(std::isfinite(got[i])) << "non-finite output at element " << i;
  }
  EXPECT_LT(d.max_abs, kCubeAbsTolerance)
      << "worst element " << d.worst_at << " of " << b.reference.size();
}

// N = 16 is MLA at B=1 with H_Q=16, and the smallest shape the gate admits;
// N = 64 is B=4 at the same head count, and the first that spreads over more
// than one block. Both are the regimes the refactor targets.
INSTANTIATE_TEST_SUITE_P(DenseBatches, RotateQCube, ::testing::Values(int64_t{2}, int64_t{8}));

// --- 4. the two paths against each other -------------------------------------

TEST(RotateQCube, AgreesWithTheVectorPathOnIdenticalInput) {
  REQUIRE_ASCEND_950PR();
  const Batch b = MakeBatch(/*batch=*/8, 0x2C41u);
  ASSERT_EQ(b.num_vectors, 64);

  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  tqr::RotateQPlan cube_plan;
  const std::vector<float> cube = RunRotation(b, stream, /*force_aiv=*/false, &cube_plan);
  ASSERT_TRUE(cube_plan.use_cube);

  tqr::RotateQPlan aiv_plan;
  const std::vector<float> aiv = RunRotation(b, stream, /*force_aiv=*/true, &aiv_plan);
  ASSERT_FALSE(aiv_plan.use_cube);

  const Deviation d = Compare(cube, aiv);
  std::printf("[ rotate_q ] cube vs aiv at N=%lld: %zu of %zu elements differ, max |err| %.3e\n",
              static_cast<long long>(b.num_vectors), d.differing, cube.size(), d.max_abs);
  EXPECT_LT(d.max_abs, kCubeAbsTolerance) << "worst element " << d.worst_at;

  // And the vector path is still exactly the reference at this shape, which is
  // what licenses using it as the comparison above.
  const Deviation ref = Compare(aiv, b.reference);
  EXPECT_EQ(ref.differing, 0u) << "the vector path drifted from the reference at element " << ref.worst_at;
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
