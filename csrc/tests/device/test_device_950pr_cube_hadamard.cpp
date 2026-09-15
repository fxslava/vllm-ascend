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

constexpr double kMaxAbsError = 1e-4;

constexpr double kSingleMmadCeiling = 2e-3;

constexpr double kAivCeiling = 1e-5;

std::vector<float> GoldenBatch(const std::vector<float>& input, int64_t dim, int64_t num_vectors) {
  std::vector<float> golden = input;
  for (int64_t v = 0; v < num_vectors; ++v) {
    tq::cpu_fwht(&golden[static_cast<size_t>(v * dim)], static_cast<int>(dim));
  }
  return golden;
}

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

TEST(CubeHadamardSweep, HybridHiLoDualDst) {
  REQUIRE_PHYSICAL_ASCEND_950PR();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  for (int64_t dim : kDims) {
    for (int64_t num_vectors : kBatches) {
      if (!hs::HadamardDualDstApplies(dim, num_vectors)) {
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

}
}
}
