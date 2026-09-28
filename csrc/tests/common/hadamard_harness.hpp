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

#pragma once

// The shared driver of the Cube-Hadamard spike: the CPU golden, the
// upload -> launch -> synchronise -> readback -> compare sequence over the
// test-owned kernels in hadamard_spike_kernels.cpp, and the deviation printout.
// Used by the sim tier (test_sim_950pr_cube_hadamard.cpp, one shape), the device
// tier (test_device_950pr_cube_hadamard.cpp, the full sweep) and the benchmark
// (bench_950pr_cube_hadamard.cpp).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "device_buffer.hpp"
#include "hadamard_spike.hpp"
#include "turbo_quant_cpu.h"

namespace vllm_ascend {
namespace test {
namespace hadamard_harness {

namespace hs = hadamard_spike;

// The sweep the benchmark and the device test run: dim by vectors.
constexpr int64_t kDefaultDims[] = {64, 128, 256, 512};
constexpr int64_t kDefaultBatches[] = {1, 8, 16, 32};

// The fp32 golden: cpu_fwht per dim-vector, what the kernels must reproduce.
inline std::vector<float> GoldenBatch(const std::vector<float>& input, int64_t dim, int64_t num_vectors) {
  std::vector<float> golden = input;
  for (int64_t v = 0; v < num_vectors; ++v) {
    turboquant_ref::cpu_fwht(&golden[static_cast<size_t>(v * dim)], static_cast<int>(dim));
  }
  return golden;
}

// Poison the output so an untouched lane cannot pass as a result.
inline void FillSentinel(DeviceBuffer& buffer, size_t elements, float sentinel = -12345.0f) {
  const std::vector<float> poison(elements, sentinel);
  buffer.CopyFromHost(poison.data(), poison.size() * sizeof(float));
}

struct Run {
  hs::Deviation deviation;
  std::vector<float> output;
  std::vector<float> golden;
};

// One hybrid variant over one shape: builds the input and the H16 operand,
// launches, synchronises, reads back and compares against the CPU golden. The
// buffers are per-call, which is what the tests want; the benchmark keeps its
// own persistent Shape so the timed loop allocates nothing.
inline Run RunHybrid(int64_t dim, int64_t num_vectors, int64_t vectors_per_chunk, uint32_t variant,
                     aclrtStream stream) {
  const std::vector<float> input = hs::SyntheticBatch(dim, num_vectors);
  const std::vector<float> golden = GoldenBatch(input, dim, num_vectors);
  const std::vector<uint16_t> h16 = hs::Hadamard16Half();

  DeviceBuffer in_dev = DeviceBuffer::FromHost(input);
  DeviceBuffer h_dev = DeviceBuffer::FromHost(h16);
  DeviceBuffer out_dev = DeviceBuffer::Empty<float>(input.size());
  FillSentinel(out_dev, input.size());

  sim_hadamard_hybrid_impl(stream, in_dev.get(), h_dev.get(), out_dev.get(), static_cast<uint32_t>(dim),
                           static_cast<uint32_t>(num_vectors), static_cast<uint32_t>(vectors_per_chunk), variant,
                           hs::InvSqrtDim(dim));
  ACL_CHECK(aclrtSynchronizeStream(stream));

  Run run;
  run.output = out_dev.ToHost<float>();
  run.deviation = hs::Compare(run.output, golden);
  run.golden = std::move(golden);
  return run;
}

inline Run RunHybrid(int64_t dim, int64_t num_vectors, uint32_t variant, aclrtStream stream) {
  return RunHybrid(dim, num_vectors, hs::HadamardVectorsPerChunk(dim, num_vectors), variant, stream);
}

// The AIV-only fp32 baseline. `tables` may be handed in pre-uploaded (the
// device sweep builds one per dim); otherwise it is uploaded here.
inline Run RunAiv(int64_t dim, int64_t num_vectors, const DeviceBuffer& tables, aclrtStream stream) {
  const std::vector<float> input = hs::SyntheticBatch(dim, num_vectors);
  const std::vector<float> golden = GoldenBatch(input, dim, num_vectors);

  DeviceBuffer in_dev = DeviceBuffer::FromHost(input);
  DeviceBuffer out_dev = DeviceBuffer::Empty<float>(input.size());
  FillSentinel(out_dev, input.size());

  sim_hadamard_aiv_impl(stream, in_dev.get(), tables.get(), out_dev.get(), static_cast<uint32_t>(dim),
                        static_cast<uint32_t>(num_vectors), hs::InvSqrtDim(dim));
  ACL_CHECK(aclrtSynchronizeStream(stream));

  Run run;
  run.output = out_dev.ToHost<float>();
  run.deviation = hs::Compare(run.output, golden);
  run.golden = std::move(golden);
  return run;
}

inline Run RunAiv(int64_t dim, int64_t num_vectors, aclrtStream stream) {
  DeviceBuffer tables = DeviceBuffer::FromHost(hs::EarlyStageTables(dim));
  return RunAiv(dim, num_vectors, tables, stream);
}

// The shared "[ hadamard ]" line: worst-lane max error and RMS.
inline void PrintDeviation(const std::string& label, const hs::Deviation& d, const std::vector<float>& got,
                           const std::vector<float>& want, int label_width) {
  std::printf("[ hadamard ] %-*s max|err| = %-12g rms = %-12g (worst lane %zu: got %g, want %g)\n", label_width,
              label.c_str(), d.max_abs, d.rms, d.worst_at, static_cast<double>(got[d.worst_at]),
              static_cast<double>(want[d.worst_at]));
  std::fflush(stdout);
}

}  // namespace hadamard_harness
}  // namespace test
}  // namespace vllm_ascend
