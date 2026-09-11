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
 * EXPLORATORY SPIKE, on silicon: the Cube-factorised Walsh-Hadamard over the
 * whole shape sweep, against the CPU golden.
 *
 * sim/test_sim_950pr_cube_hadamard.cpp is the same check at D = 256, V = 16 on
 * the camodel, and that is all a cycle-level simulator can afford - one case is
 * around 40 seconds there and this file is 64 of them. The split is the tier
 * split, not a difference of intent: the shapes here are the ones that decide
 * whether the factorisation is worth integrating, and they need the part.
 *
 * WHAT THE SWEEP IS FOR, AXIS BY AXIS
 *
 *   D in {64, 128, 256, 512}   The Cube absorbs FOUR stages at every D - the
 *                              lower nibble of the index, always a 16 x 16
 *                              constant - so the fraction of the transform it
 *                              covers falls from 4/6 at D = 64 to 4/9 at
 *                              D = 512. Correctness must not care; the benefit
 *                              is expected to. 256 is Qwen3.5-2B's head_dim.
 *
 *   V in {1, 8, 16, 32}        The amortisation axis. One Mmad covers a whole
 *                              chunk of vectors, so V = 1 is the worst case for
 *                              the Cube - a 16-row fractal load for four rows
 *                              of real data at D = 64 - and the case most
 *                              likely to expose a padding bug. V = 32 at
 *                              D = 512 is the only shape in the sweep that
 *                              chunks more than once, so it is also the test of
 *                              the chunk loop's cross-iteration syncs.
 *
 *   three variants             single Mmad, hi/lo, hi/lo + dual destination.
 *                              The middle one is the one gated at 1e-4; see the
 *                              bound notes on each case.
 *
 * TWO SHAPES THAT DEGENERATE, BOTH DELIBERATE AND BOTH ASSERTED RATHER THAN
 * ASSUMED:
 *
 *   V = 1 chunks to one vector, so the dual-destination Fixpipe would split a
 *   single vector's tile between the two subcores and leave neither able to run
 *   a row butterfly. The kernel drops back to the single-destination path, and
 *   HadamardDualDstApplies is what says so; the dual-destination case skips
 *   there rather than quietly measuring hi/lo twice.
 *
 *   D = 64, V = 1 is four rows of a sixteen-row fractal. The other twelve are
 *   zeroed by the kernel before staging, because a LoadData of one fractal
 *   reads all sixteen whatever m says.
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

const int64_t kDims[] = {64, 128, 256, 512};
const int64_t kBatches[] = {1, 8, 16, 32};

// The gate the hi/lo configuration has to clear, at every shape in the sweep.
// It is an absolute bound on a transform of unit-scale data, and the hi/lo
// error is the fp32 floor rather than anything that grows with D - measured
// 2.4e-7 at D = 256 - so one number covers the sweep with three orders of
// margin.
constexpr double kMaxAbsError = 1e-4;

// A single fp16 pass is the fp32 -> fp16 cast of the input carried through a
// normalised transform: about 3.6e-4 at D = 256, and it grows slowly with D
// because the transform is orthogonal. This bound is not a fidelity claim, it
// is the line between "fp16 rounding" and "the Cube multiplied the wrong
// matrix" - a layout defect lands at O(1).
constexpr double kSingleMmadCeiling = 2e-3;

// fp32 from end to end; the only difference from the host is the order the
// stages accumulate in.
constexpr double kAivCeiling = 1e-5;

std::vector<float> GoldenBatch(const std::vector<float>& input, int64_t dim, int64_t num_vectors) {
  std::vector<float> golden = input;
  for (int64_t v = 0; v < num_vectors; ++v) {
    tq::cpu_fwht(&golden[static_cast<size_t>(v * dim)], static_cast<int>(dim));
  }
  return golden;
}

// A sentinel, not a zero fill: it separates "the kernel wrote nothing" from
// "the kernel wrote zeros", which on this part are different bugs with the same
// symptom. A skipped subcore leaves -12345 behind and the worst lane names it.
void FillSentinel(DeviceBuffer& buffer, size_t elements) {
  const std::vector<float> sentinel(elements, -12345.0f);
  buffer.CopyFromHost(sentinel.data(), sentinel.size() * sizeof(float));
}

void Report(const std::string& label, const hs::Deviation& d, const std::vector<float>& got,
            const std::vector<float>& want) {
  std::printf("[ hadamard ] %-26s max|err| = %-12g rms = %-12g (worst lane %zu: got %g, want %g)\n", label.c_str(),
              d.max_abs, d.rms, d.worst_at, static_cast<double>(got[d.worst_at]),
              static_cast<double>(want[d.worst_at]));
  std::fflush(stdout);
}

hs::Deviation RunHybrid(int64_t dim, int64_t num_vectors, uint32_t variant, const char* variant_label,
                        aclrtStream stream) {
  const std::vector<float> input = hs::SyntheticBatch(dim, num_vectors);
  const std::vector<float> golden = GoldenBatch(input, dim, num_vectors);
  const std::vector<uint16_t> h16 = hs::Hadamard16Half();

  DeviceBuffer in_dev = DeviceBuffer::FromHost(input);
  DeviceBuffer h_dev = DeviceBuffer::FromHost(h16);
  DeviceBuffer out_dev = DeviceBuffer::Empty<float>(input.size());
  FillSentinel(out_dev, input.size());

  sim_hadamard_hybrid_impl(stream, in_dev.get(), h_dev.get(), out_dev.get(), static_cast<uint32_t>(dim),
                           static_cast<uint32_t>(num_vectors),
                           static_cast<uint32_t>(hs::HadamardVectorsPerChunk(dim, num_vectors)), variant,
                           hs::InvSqrtDim(dim));
  ACL_CHECK(aclrtSynchronizeStream(stream));

  const std::vector<float> got = out_dev.ToHost<float>();
  const hs::Deviation d = hs::Compare(got, golden);
  Report(hs::CaseLabel(dim, num_vectors, variant_label), d, got, golden);
  return d;
}

// All log2(D) stages on the vector unit. Pins the golden on the device, so a
// failure here says the reference and the part disagree before the Cube is
// involved and every hybrid result below is unreadable.
TEST(CubeHadamardSweep, AivBaseline) {
  REQUIRE_PHYSICAL_ASCEND_950PR();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  for (int64_t dim : kDims) {
    const std::vector<int32_t> tables = hs::EarlyStageTables(dim);
    DeviceBuffer tab_dev = DeviceBuffer::FromHost(tables);
    for (int64_t num_vectors : kBatches) {
      const std::vector<float> input = hs::SyntheticBatch(dim, num_vectors);
      const std::vector<float> golden = GoldenBatch(input, dim, num_vectors);

      DeviceBuffer in_dev = DeviceBuffer::FromHost(input);
      DeviceBuffer out_dev = DeviceBuffer::Empty<float>(input.size());
      FillSentinel(out_dev, input.size());

      sim_hadamard_aiv_impl(stream, in_dev.get(), tab_dev.get(), out_dev.get(), static_cast<uint32_t>(dim),
                            static_cast<uint32_t>(num_vectors), hs::InvSqrtDim(dim));
      ACL_CHECK(aclrtSynchronizeStream(stream));

      const std::vector<float> got = out_dev.ToHost<float>();
      const hs::Deviation d = hs::Compare(got, golden);
      Report(hs::CaseLabel(dim, num_vectors, "aiv"), d, got, golden);
      EXPECT_LT(d.max_abs, kAivCeiling)
          << "the AIV-only fp32 transform does not reproduce cpu_fwht at D=" << dim << ", V=" << num_vectors;
    }
  }
}

// One Mmad, fp16 operands. Swept for the shape of the error against D, not
// against the 1e-4 gate.
TEST(CubeHadamardSweep, HybridSingleMmad) {
  REQUIRE_PHYSICAL_ASCEND_950PR();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  for (int64_t dim : kDims) {
    for (int64_t num_vectors : kBatches) {
      const hs::Deviation d = RunHybrid(dim, num_vectors, hs::kHybridSingleMmad, "single", stream);
      EXPECT_LT(d.max_abs, kSingleMmadCeiling)
          << "a single fp16 Mmad at D=" << dim << ", V=" << num_vectors
          << " is off by more than fp16 rounding can account for, so the factorisation or the fractal layout is "
             "wrong rather than the operand grid";
    }
  }
}

// The configuration that has to clear the gate, at every shape.
TEST(CubeHadamardSweep, HybridHiLo) {
  REQUIRE_PHYSICAL_ASCEND_950PR();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  for (int64_t dim : kDims) {
    for (int64_t num_vectors : kBatches) {
      const hs::Deviation d = RunHybrid(dim, num_vectors, hs::kHybridHiLo, "hilo", stream);
      EXPECT_LT(d.max_abs, kMaxAbsError)
          << "the two-Mmad hi/lo split does not reach 1e-4 at D=" << dim << ", V=" << num_vectors
          << ", so the Cube factorisation cannot replace the lower four butterfly stages at the fidelity "
             "TurboQuant's rotation needs";
    }
  }
}

/*
 * The same, with Fixpipe's dualDstCtl = 0b01 splitting the product across both
 * vector subcores.
 *
 * The mode is undocumented in the public headers; the assert in FixpipeL0cToUB
 * (dav_3510/kernel_operator_fixpipe_impl.h) is what says it writes M/2 x N into
 * each subcore's UB. If that reading is wrong at some shape, subcore 1 finds
 * its half of the buffer untouched and the sentinel survives into the output -
 * an O(1) error confined to the upper half of the batch, which is what the
 * failure message tells the reader to look for.
 */
TEST(CubeHadamardSweep, HybridHiLoDualDst) {
  REQUIRE_PHYSICAL_ASCEND_950PR();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  for (int64_t dim : kDims) {
    for (int64_t num_vectors : kBatches) {
      if (!hs::HadamardDualDstApplies(dim, num_vectors)) {
        // V = 1. The kernel takes the single-destination path, so running the
        // case would measure HybridHiLo a second time under another name.
        std::printf("[ hadamard ] %-26s skipped: chunk holds one vector, dual destination does not apply\n",
                    hs::CaseLabel(dim, num_vectors, "dualdst").c_str());
        continue;
      }
      const hs::Deviation d =
          RunHybrid(dim, num_vectors, hs::kHybridHiLo | hs::kHybridDualDst, "dualdst", stream);
      EXPECT_LT(d.max_abs, kMaxAbsError)
          << "Fixpipe's dual-destination mode did not put half the product in each subcore's UB at D=" << dim
          << ", V=" << num_vectors << "; if the worst lane is at " << (num_vectors / 2) * dim
          << " or above, subcore 1 never received its half";
    }
  }
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
