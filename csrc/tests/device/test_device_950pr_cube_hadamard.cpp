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

constexpr double kMaxAbsError = 1e-4;

constexpr double kSingleMmadCeiling = 2e-3;

constexpr double kAivCeiling = 1e-5;

void Report(const std::string& label, const hs::Deviation& d, const std::vector<float>& got,
            const std::vector<float>& want) {
  hh::PrintDeviation(label, d, got, want, 26);
}

TEST(CubeHadamardSweep, AivBaseline) {
  REQUIRE_PHYSICAL_ASCEND_950PR();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  for (int64_t dim : hh::kDefaultDims) {
    DeviceBuffer tab_dev = DeviceBuffer::FromHost(hs::EarlyStageTables(dim));
    for (int64_t num_vectors : hh::kDefaultBatches) {
      const hh::Run run = hh::RunAiv(dim, num_vectors, tab_dev, stream);
      Report(hs::CaseLabel(dim, num_vectors, "aiv"), run.deviation, run.output, run.golden);
      EXPECT_LT(run.deviation.max_abs, kAivCeiling)
          << "the AIV-only fp32 transform does not reproduce cpu_fwht at D=" << dim << ", V=" << num_vectors;
    }
  }
}

TEST(CubeHadamardSweep, HybridSingleMmad) {
  REQUIRE_PHYSICAL_ASCEND_950PR();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  for (int64_t dim : hh::kDefaultDims) {
    for (int64_t num_vectors : hh::kDefaultBatches) {
      const hh::Run run = hh::RunHybrid(dim, num_vectors, hs::kHybridSingleMmad, stream);
      Report(hs::CaseLabel(dim, num_vectors, "single"), run.deviation, run.output, run.golden);
      EXPECT_LT(run.deviation.max_abs, kSingleMmadCeiling)
          << "a single fp16 Mmad at D=" << dim << ", V=" << num_vectors
          << " is off by more than fp16 rounding can account for, so the factorisation or the fractal layout is "
             "wrong rather than the operand grid";
    }
  }
}

TEST(CubeHadamardSweep, HybridHiLo) {
  REQUIRE_PHYSICAL_ASCEND_950PR();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  for (int64_t dim : hh::kDefaultDims) {
    for (int64_t num_vectors : hh::kDefaultBatches) {
      const hh::Run run = hh::RunHybrid(dim, num_vectors, hs::kHybridHiLo, stream);
      Report(hs::CaseLabel(dim, num_vectors, "hilo"), run.deviation, run.output, run.golden);
      EXPECT_LT(run.deviation.max_abs, kMaxAbsError)
          << "the two-Mmad hi/lo split does not reach 1e-4 at D=" << dim << ", V=" << num_vectors
          << ", so the Cube factorisation cannot replace the lower four butterfly stages at the fidelity "
             "TurboQuant's rotation needs";
    }
  }
}

TEST(CubeHadamardSweep, HybridHiLoDualDst) {
  REQUIRE_PHYSICAL_ASCEND_950PR();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  for (int64_t dim : hh::kDefaultDims) {
    for (int64_t num_vectors : hh::kDefaultBatches) {
      if (!hs::HadamardDualDstApplies(dim, num_vectors)) {
        std::printf("[ hadamard ] %-26s skipped: chunk holds one vector, dual destination does not apply\n",
                    hs::CaseLabel(dim, num_vectors, "dualdst").c_str());
        continue;
      }
      const hh::Run run = hh::RunHybrid(dim, num_vectors, hs::kHybridHiLo | hs::kHybridDualDst, stream);
      Report(hs::CaseLabel(dim, num_vectors, "dualdst"), run.deviation, run.output, run.golden);
      EXPECT_LT(run.deviation.max_abs, kMaxAbsError)
          << "Fixpipe's dual-destination mode did not put half the product in each subcore's UB at D=" << dim
          << ", V=" << num_vectors << "; if the worst lane is at " << (num_vectors / 2) * dim
          << " or above, subcore 1 never received its half";
    }
  }
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
