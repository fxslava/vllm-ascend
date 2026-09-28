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

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "device_buffer.hpp"
#include "hadamard_harness.hpp"
#include "hadamard_spike.hpp"
#include "test_harness.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace hh = hadamard_harness;
namespace hs = hadamard_spike;

constexpr int64_t kDim = 256;
constexpr int64_t kNumVectors = 16;

constexpr double kMaxAbsError = 1e-4;

constexpr double kHiLoFloor = 2.384185791015625e-7;

void Report(const char* name, const hs::Deviation& d, const std::vector<float>& got, const std::vector<float>& want) {
  hh::PrintDeviation(name, d, got, want, 20);
}

hh::Run RunHybridChunked(const char* name, uint32_t variant, int64_t vectors_per_chunk, aclrtStream stream) {
  hh::Run run = hh::RunHybrid(kDim, kNumVectors, vectors_per_chunk, variant, stream);
  Report(name, run.deviation, run.output, run.golden);
  return run;
}

hh::Run RunHybrid(const char* name, uint32_t variant, aclrtStream stream) {
  return RunHybridChunked(name, variant, hs::HadamardVectorsPerChunk(kDim, kNumVectors), stream);
}

TEST(CubeHadamard256, AivBaseline) {
  REQUIRE_ASCEND_950PR();
  const hh::Run run = hh::RunAiv(kDim, kNumVectors, AscendTestEnvironment::Instance().stream());
  Report("aiv fp32 (8 stages)", run.deviation, run.output, run.golden);

  EXPECT_LT(run.deviation.max_abs, kMaxAbsError) << "the AIV-only fp32 transform does not reproduce cpu_fwht, so "
                                                    "the golden reference and the device disagree before the Cube "
                                                    "is involved";
}

TEST(CubeHadamard256, HybridSingleMmad) {
  REQUIRE_ASCEND_950PR();
  const hs::Deviation d =
      RunHybrid("hybrid fp16 x1", hs::kHybridSingleMmad, AscendTestEnvironment::Instance().stream()).deviation;

  EXPECT_LT(d.max_abs, 1e-2) << "a single fp16 Mmad is off by far more than fp16 rounding can account for, so the "
                                "factorisation or the fractal layout is wrong, not the operand grid";
}

TEST(CubeHadamard256, HybridHiLo) {
  REQUIRE_ASCEND_950PR();
  const hs::Deviation d =
      RunHybrid("hybrid fp16 hi+lo", hs::kHybridHiLo, AscendTestEnvironment::Instance().stream()).deviation;
  EXPECT_LT(d.max_abs, kMaxAbsError)
      << "the two-Mmad hi/lo split does not reach 1e-4, so the Cube factorisation cannot replace the lower four "
         "butterfly stages at the fidelity TurboQuant's rotation needs";
}

TEST(CubeHadamard256, HybridHiLoDualDst) {
  REQUIRE_ASCEND_950PR();
  ASSERT_TRUE(hs::HadamardDualDstApplies(kDim, kNumVectors))
      << "this shape chunks to an odd number of vectors, so the kernel ignores the dual-destination bit and the "
         "case would silently measure HybridHiLo again";
  const hs::Deviation d =
      RunHybrid("hybrid dual-dst", hs::kHybridHiLo | hs::kHybridDualDst, AscendTestEnvironment::Instance().stream())
          .deviation;
  EXPECT_LT(d.max_abs, kMaxAbsError)
      << "Fixpipe's dual-destination mode did not put half the product in each subcore's UB; if the worst lane is at "
      << (kNumVectors / 2) * kDim << " or above, subcore 1 never received its half";
}

TEST(CubeHadamard256, PipelinedFourChunks) {
  REQUIRE_ASCEND_950PR();
  const hs::Deviation d = RunHybridChunked("pipelined 4 x 4", hs::kHybridHiLo | hs::kHybridDualDst,
                                           hs::kPipelineVectorsPerChunk, AscendTestEnvironment::Instance().stream())
                              .deviation;
  EXPECT_LT(d.max_abs, kMaxAbsError) << "the pipelined kernel does not reproduce the transform over "
                                     << (kNumVectors / hs::kPipelineVectorsPerChunk)
                                     << " chunks; an error at O(1) on whole chunks is a slot race, not arithmetic";
  EXPECT_LE(d.max_abs, kHiLoFloor) << "the pipelined kernel is above the hi/lo fp32 floor measured at one chunk";
}

TEST(CubeHadamard256, PipelinedFourChunksSingleDst) {
  REQUIRE_ASCEND_950PR();
  const hs::Deviation d = RunHybridChunked("pipelined 4 x 4 single-dst", hs::kHybridHiLo, hs::kPipelineVectorsPerChunk,
                                           AscendTestEnvironment::Instance().stream())
                              .deviation;
  EXPECT_LE(d.max_abs, kHiLoFloor)
      << "the pipelined kernel corrupts chunks when the Fixpipe writes only subcore 0's UB, so the AIV -> AIC flag is "
         "NOT waiting for both subcores and kFlagProductFree is released by the idle one";
}

TEST(CubeHadamard256, LockstepFourChunks) {
  REQUIRE_ASCEND_950PR();
  const hs::Deviation d = RunHybridChunked("lockstep 4 x 4", hs::kHybridHiLo | hs::kHybridDualDst | hs::kHybridLockstep,
                                           hs::kPipelineVectorsPerChunk, AscendTestEnvironment::Instance().stream())
                              .deviation;
  EXPECT_LE(d.max_abs, kHiLoFloor) << "the lockstep baseline itself does not reach the hi/lo floor at four chunks, so "
                                      "the comparison against the pipelined kernel has no baseline";
}

TEST(CubeHadamard256, PipelinedMatchesLockstepBitwise) {
  REQUIRE_ASCEND_950PR();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();
  const uint32_t variant = hs::kHybridHiLo | hs::kHybridDualDst;

  const std::vector<float> lockstep =
      RunHybridChunked("lockstep 4 x 4", variant | hs::kHybridLockstep, hs::kPipelineVectorsPerChunk, stream).output;
  const std::vector<float> pipelined =
      RunHybridChunked("pipelined 4 x 4", variant, hs::kPipelineVectorsPerChunk, stream).output;

  ASSERT_EQ(lockstep.size(), pipelined.size());
  size_t differing = 0;
  size_t first_at = 0;
  for (size_t i = 0; i < lockstep.size(); ++i) {
    if (lockstep[i] != pipelined[i]) {
      if (differing == 0) {
        first_at = i;
      }
      ++differing;
    }
  }
  EXPECT_EQ(differing, 0u) << differing << " lanes differ, first at " << first_at << " (chunk "
                           << first_at / static_cast<size_t>(hs::kPipelineVectorsPerChunk * kDim)
                           << "): the pipelined kernel and the lockstep kernel issue identical arithmetic, so a "
                              "difference is a slot hazard";
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
