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

/*
 * EXPLORATORY SPIKE. Nothing here is on the decode path.
 *
 * Does a Walsh-Hadamard transform get cheaper if its lower four butterfly
 * stages run on the Cube instead of the vector unit? Sylvester says
 * H_D = H_R (x) H_16 for D = 16 R, so reshaping the vector into an R x 16 tile
 * turns those four stages into one Mmad against a static +-1 matrix and leaves
 * the upper log2(R) as row butterflies on the AIV.
 *
 * The micro-kernel is csrc/tests/sim/sim_hadamard_hybrid_kernels.cpp; its
 * header carries the derivation and the layout facts that make the shape work.
 * This file is the CAModel measurement at ONE shape. The sweep over
 * D in {64, 128, 256, 512} and V in {1, 8, 16, 32} is
 * device/test_device_950pr_cube_hadamard.cpp, which needs silicon: a
 * cycle-level simulator is 40 seconds per case here and the sweep is 64 of
 * them.
 *
 * WHAT EACH CASE IS FOR
 *
 *   AivBaseline        all log2(D) stages on the AIV. Pins the golden reference
 *                      on the device and is the thing the hybrid has to beat.
 *   HybridSingleMmad   one Mmad, operands fp16. Reported, not asserted at the
 *                      1e-4 gate: the fp32 -> fp16 cast of the input alone is
 *                      worth about 1e-3 after a normalised H_256, so this
 *                      measures the operand grid rather than the factorisation.
 *   HybridHiLo         x = hi + lo with both halves fp16, two Mmads into one
 *                      L0C. This is the configuration that has to clear 1e-4.
 *   HybridHiLoDualDst  the same, with Fixpipe's dualDstCtl = 0b01 handing each
 *                      vector subcore half the product so both run the residual
 *                      stages. Same numbers as HybridHiLo or the dual-destination
 *                      mode is not doing what its assert says.
 *
 * Those four all run at ONE chunk, which is what the default chunk size comes
 * to at this shape, so none of them exercises the kernel's macro-pipeline. Two
 * more do:
 *
 *   PipelinedFourChunks           four chunks of four vectors, the smallest
 *                                 count that reaches every slot-reuse edge of
 *                                 the L1/UB ping-pong.
 *   LockstepFourChunks            the same four chunks through the loop the
 *                                 pipeline replaced. One launch per process, so
 *                                 the two cases' camodel instruction logs are
 *                                 what the overlap is read from.
 *   PipelinedMatchesLockstepBitwise
 *                                 both, in one process, compared bit for bit.
 *
 * A camodel pass here is well under a minute per case.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "device_buffer.hpp"
#include "hadamard_spike.hpp"
#include "test_harness.hpp"
#include "turbo_quant_cpu.h"

namespace vllm_ascend {
namespace test {
namespace {

namespace tq = turboquant_ref;
namespace hs = hadamard_spike;

// One shape, and the one the spike's numbers are quoted at. head_dim 256 is
// Qwen3.5-2B's; 16 vectors is one 4096-element chunk, so the Cube stage is a
// single Mmad and the residual is one pass per subcore.
constexpr int64_t kDim = 256;
constexpr int64_t kNumVectors = 16;

// The gate this spike has to clear on a normalised transform.
constexpr double kMaxAbsError = 1e-4;

/*
 * What the hi/lo split actually measures, against which the pipelined form has
 * to be no worse: 2^-22, one fp32 ulp at unit scale, which is the floor the
 * two-Mmad path reached at D = 256 on the camodel. The spike's notes quote it
 * rounded to 2.38e-7; the measurement is 2.384185791e-7 and a gate written at
 * the rounded figure fails on the exact one.
 */
constexpr double kHiLoFloor = 2.384185791015625e-7;

std::vector<float> GoldenBatch(const std::vector<float>& input, int64_t dim, int64_t num_vectors) {
  std::vector<float> golden = input;
  for (int64_t v = 0; v < num_vectors; ++v) {
    tq::cpu_fwht(&golden[static_cast<size_t>(v * dim)], static_cast<int>(dim));
  }
  return golden;
}

void Report(const char* name, const hs::Deviation& d, const std::vector<float>& got,
            const std::vector<float>& want) {
  std::printf("[ hadamard ] %-20s max|err| = %-12g rms = %-12g (worst lane %zu: got %g, want %g)\n", name, d.max_abs,
              d.rms, d.worst_at, static_cast<double>(got[d.worst_at]), static_cast<double>(want[d.worst_at]));
  std::fflush(stdout);
}

// A sentinel, not a zero fill: it separates "the kernel wrote nothing" from
// "the kernel wrote zeros", which on this part are different bugs with the
// same symptom.
void FillSentinel(DeviceBuffer& buffer, size_t elements) {
  const std::vector<float> sentinel(elements, -12345.0f);
  buffer.CopyFromHost(sentinel.data(), sentinel.size() * sizeof(float));
}

struct HybridRun {
  hs::Deviation deviation;
  std::vector<float> output;
};

HybridRun RunHybridChunked(const char* name, uint32_t variant, int64_t vectors_per_chunk, aclrtStream stream) {
  const std::vector<float> input = hs::SyntheticBatch(kDim, kNumVectors);
  const std::vector<float> golden = GoldenBatch(input, kDim, kNumVectors);
  const std::vector<uint16_t> h16 = hs::Hadamard16Half();

  DeviceBuffer in_dev = DeviceBuffer::FromHost(input);
  DeviceBuffer h_dev = DeviceBuffer::FromHost(h16);
  DeviceBuffer out_dev = DeviceBuffer::Empty<float>(input.size());
  FillSentinel(out_dev, input.size());

  sim_hadamard_hybrid_impl(stream, in_dev.get(), h_dev.get(), out_dev.get(), static_cast<uint32_t>(kDim),
                           static_cast<uint32_t>(kNumVectors), static_cast<uint32_t>(vectors_per_chunk), variant,
                           hs::InvSqrtDim(kDim));
  ACL_CHECK(aclrtSynchronizeStream(stream));

  HybridRun run;
  run.output = out_dev.ToHost<float>();
  run.deviation = hs::Compare(run.output, golden);
  Report(name, run.deviation, run.output, golden);
  return run;
}

hs::Deviation RunHybrid(const char* name, uint32_t variant, aclrtStream stream) {
  return RunHybridChunked(name, variant, hs::HadamardVectorsPerChunk(kDim, kNumVectors), stream).deviation;
}

// The AIV-only transform. Also the check that the golden reference and the
// device agree at all, independently of anything the Cube does.
TEST(CubeHadamard256, AivBaseline) {
  REQUIRE_ASCEND_950PR();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  const std::vector<float> input = hs::SyntheticBatch(kDim, kNumVectors);
  const std::vector<float> golden = GoldenBatch(input, kDim, kNumVectors);
  const std::vector<int32_t> tables = hs::EarlyStageTables(kDim);

  DeviceBuffer in_dev = DeviceBuffer::FromHost(input);
  DeviceBuffer tab_dev = DeviceBuffer::FromHost(tables);
  DeviceBuffer out_dev = DeviceBuffer::Empty<float>(input.size());
  FillSentinel(out_dev, input.size());

  sim_hadamard_aiv_impl(stream, in_dev.get(), tab_dev.get(), out_dev.get(), static_cast<uint32_t>(kDim),
                        static_cast<uint32_t>(kNumVectors), hs::InvSqrtDim(kDim));
  ACL_CHECK(aclrtSynchronizeStream(stream));

  const std::vector<float> got = out_dev.ToHost<float>();
  const hs::Deviation d = hs::Compare(got, golden);
  Report("aiv fp32 (8 stages)", d, got, golden);

  // fp32 throughout, so the only difference from the host is the order the
  // eight stages accumulate in. It measured exactly 0.
  EXPECT_LT(d.max_abs, kMaxAbsError) << "the AIV-only fp32 transform does not reproduce cpu_fwht, so the golden "
                                        "reference and the device disagree before the Cube is involved";
}

// One Mmad, fp16 operands. Data, not a gate: what it measures is what a single
// fp16 pass costs in accuracy, and the answer decides whether the hi/lo split
// below is necessary.
TEST(CubeHadamard256, HybridSingleMmad) {
  REQUIRE_ASCEND_950PR();
  const hs::Deviation d =
      RunHybrid("hybrid fp16 x1", hs::kHybridSingleMmad, AscendTestEnvironment::Instance().stream());

  // Loose, and deliberately so: the bound that matters is on the hi/lo case.
  // This one only has to show the factorisation is right and the error is
  // rounding rather than a wrong matrix - a layout defect lands at O(1), not at
  // O(1e-3). Measured 3.58e-4.
  EXPECT_LT(d.max_abs, 1e-2) << "a single fp16 Mmad is off by far more than fp16 rounding can account for, so the "
                                "factorisation or the fractal layout is wrong, not the operand grid";
}

// The configuration that has to clear the gate.
TEST(CubeHadamard256, HybridHiLo) {
  REQUIRE_ASCEND_950PR();
  const hs::Deviation d = RunHybrid("hybrid fp16 hi+lo", hs::kHybridHiLo, AscendTestEnvironment::Instance().stream());
  EXPECT_LT(d.max_abs, kMaxAbsError)
      << "the two-Mmad hi/lo split does not reach 1e-4, so the Cube factorisation cannot replace the lower four "
         "butterfly stages at the fidelity TurboQuant's rotation needs";
}

/*
 * The same numbers, with the product split across both vector subcores by the
 * Fixpipe itself.
 *
 * dualDstCtl = 0b01 is undocumented in the public headers; the assert in
 * FixpipeL0cToUB (dav_3510/kernel_operator_fixpipe_impl.h) is what says it
 * writes M/2 x N into each subcore's UB. If that reading is wrong the most
 * likely outcome is that subcore 1 finds its half of the buffer untouched and
 * writes the sentinel back out, which shows up as an O(1) error on exactly the
 * upper half of the batch - so the failure message names that shape.
 */
TEST(CubeHadamard256, HybridHiLoDualDst) {
  REQUIRE_ASCEND_950PR();
  ASSERT_TRUE(hs::HadamardDualDstApplies(kDim, kNumVectors))
      << "this shape chunks to an odd number of vectors, so the kernel ignores the dual-destination bit and the "
         "case would silently measure HybridHiLo again";
  const hs::Deviation d = RunHybrid("hybrid dual-dst", hs::kHybridHiLo | hs::kHybridDualDst,
                                    AscendTestEnvironment::Instance().stream());
  EXPECT_LT(d.max_abs, kMaxAbsError)
      << "Fixpipe's dual-destination mode did not put half the product in each subcore's UB; if the worst lane is at "
      << (kNumVectors / 2) * kDim << " or above, subcore 1 never received its half";
}

/*
 * THE PIPELINE SMOKE CASE.
 *
 * Everything above runs at one chunk, where the kernel is a prolog and an
 * epilogue and the two-slot ping-pong never turns over. Four vectors per chunk
 * gives four chunks, which is the smallest count that reaches every edge: the
 * AIV's wait on kFlagOperandsFree (chunks 2 and 3), the AIC's wait on
 * kFlagProductFree (chunks 2 and 3), and the conditioned sets that have to stay
 * balanced against them on the last two chunks.
 *
 * Two things are checked and they fail differently. A deadlock in the handshake
 * hangs - there is no assertion for it, the case simply never returns, which is
 * why it is run on its own. A race that does NOT deadlock shows up here as
 * numbers: a Fixpipe landing in a UB slot whose residual has not finished, or a
 * stage overwriting an L1 slot the Cube has not read, corrupts whole chunks at
 * O(1), nowhere near the fp32 floor the transform itself sits at.
 */
TEST(CubeHadamard256, PipelinedFourChunks) {
  REQUIRE_ASCEND_950PR();
  const hs::Deviation d = RunHybridChunked("pipelined 4 x 4", hs::kHybridHiLo | hs::kHybridDualDst,
                                           hs::kPipelineVectorsPerChunk, AscendTestEnvironment::Instance().stream())
                              .deviation;
  EXPECT_LT(d.max_abs, kMaxAbsError)
      << "the pipelined kernel does not reproduce the transform over " << (kNumVectors / hs::kPipelineVectorsPerChunk)
      << " chunks; an error at O(1) on whole chunks is a slot race, not arithmetic";
  // The chunking changes nothing arithmetically - each vector's transform is
  // independent of every other - so the hi/lo bound measured at one chunk has
  // to hold unchanged here. A pipelined kernel that is merely CLOSE has a
  // partially-overwritten buffer somewhere.
  EXPECT_LE(d.max_abs, kHiLoFloor) << "the pipelined kernel is above the hi/lo fp32 floor measured at one chunk";
}

/*
 * The pipeline with the product landing in ONE subcore's UB.
 *
 * Not a second data point on accuracy - the numbers are the same as
 * PipelinedFourChunks - but the only case that exercises the assumption the
 * kFlagProductFree edge rests on. Without dualDstCtl, subcore 1 owns no vectors
 * of the chunk: it returns from the residual immediately and signals that UB is
 * free while subcore 0 is still mid-butterfly. What holds the Cube back is that
 * an arch35 AIV -> AIC flag in mode 0x02 is released only once BOTH subcores
 * have set it. If that reading is wrong, the Fixpipe of chunk k overwrites the
 * product of chunk k - 2 underneath subcore 0's residual, and the damage is
 * O(1) on whole chunks rather than at the fp32 floor.
 *
 * The device sweep reaches this path - it drives kHybridHiLo without the
 * dual-destination bit - so it is not a hypothetical shape.
 */
TEST(CubeHadamard256, PipelinedFourChunksSingleDst) {
  REQUIRE_ASCEND_950PR();
  const hs::Deviation d = RunHybridChunked("pipelined 4 x 4 single-dst", hs::kHybridHiLo,
                                           hs::kPipelineVectorsPerChunk, AscendTestEnvironment::Instance().stream())
                              .deviation;
  EXPECT_LE(d.max_abs, kHiLoFloor)
      << "the pipelined kernel corrupts chunks when the Fixpipe writes only subcore 0's UB, so the AIV -> AIC flag is "
         "NOT waiting for both subcores and kFlagProductFree is released by the idle one";
}

/*
 * The same four chunks through the lockstep loop, on its own.
 *
 * Nothing here that PipelinedMatchesLockstepBitwise does not also check - what
 * this case is for is the camodel's per-core instruction logs, which accumulate
 * over every launch a process makes. Reading the lockstep cost out of a case
 * that launches twice means subtracting one run from another and hoping the
 * simulator's fixed start-up lands on the right side of the difference; one
 * launch per process is the measurement, and this is the launch.
 */
TEST(CubeHadamard256, LockstepFourChunks) {
  REQUIRE_ASCEND_950PR();
  const hs::Deviation d =
      RunHybridChunked("lockstep 4 x 4", hs::kHybridHiLo | hs::kHybridDualDst | hs::kHybridLockstep,
                       hs::kPipelineVectorsPerChunk, AscendTestEnvironment::Instance().stream())
          .deviation;
  EXPECT_LE(d.max_abs, kHiLoFloor) << "the lockstep baseline itself does not reach the hi/lo floor at four chunks, so "
                                      "the comparison against the pipelined kernel has no baseline";
}

/*
 * The pipeline against the loop it replaced, at the same chunking, same input.
 *
 * Bitwise, not approximately: the two paths issue the same Mmads over the same
 * operands in the same order and differ only in what overlaps what, so any
 * difference at all is a hazard the pipeline opened rather than a rounding
 * effect. This is also the case whose two camodel instruction logs are the
 * overlap measurement - the lockstep run is the AIV-idle-through-Cube baseline.
 */
TEST(CubeHadamard256, PipelinedMatchesLockstepBitwise) {
  REQUIRE_ASCEND_950PR();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();
  const uint32_t variant = hs::kHybridHiLo | hs::kHybridDualDst;

  const std::vector<float> lockstep =
      RunHybridChunked("lockstep 4 x 4", variant | hs::kHybridLockstep, hs::kPipelineVectorsPerChunk, stream).output;
  const std::vector<float> pipelined = RunHybridChunked("pipelined 4 x 4", variant, hs::kPipelineVectorsPerChunk,
                                                        stream)
                                           .output;

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
