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

// MatMul (aclnnMatmul) on the v200 cube unit, fp16 in / fp16 out, B presented
// transposed exactly as test_matmul_310p.cpp does.
//
// Two regimes are covered because they are bounded by different things:
//
//   M = 1   decode. A GEMV: every weight element is read once and used once, so
//           arithmetic intensity is ~1 FLOP per byte and the kernel is bounded
//           by how fast the weight can be streamed from HBM. The TFLOP/s column
//           will look poor and the GB/s column is the one that means something.
//   M > 1   prefill. The same weight serves M rows, so intensity rises with M
//           and the cube unit starts to be the limit. This is where TFLOP/s is
//           the number to read.
//
// Both columns are reported for every shape so the crossover is visible rather
// than assumed.
//
// Shape coverage goes beyond the parity suite in two directions: the M sweep
// above, and the fused-QKV and down_proj widths that the parity suite's
// cartesian product of hidden sizes cannot express. See kExtraProjectionShapes.

#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "aclnn_ops.hpp"
#include "aclnn_runtime.hpp"
#include "benchmark.hpp"
#include "device_tensor.hpp"
#include "qwen_shapes.hpp"
#include "random_data.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {

const char* kSuiteName = "matmul_310p (aclnnMatmul, cube unit, fp16)";

namespace {

const AclnnOp& MatmulOp() {
  static const AclnnOp op(ops::kMatmul);
  return op;
}

std::string CaseName(int64_t m, int64_t k, int64_t n) {
  std::ostringstream name;
  name << "m" << m << "_k" << k << "_n" << n;
  return name.str();
}

struct ProjectionCase {
  int64_t m;
  int64_t k;
  int64_t n;
};

struct ProjectionShape {
  int64_t k;
  int64_t n;
};

// Two Qwen3.5 projections that the cartesian sweep over LinearInputSizes x
// LinearOutputSizes cannot reach, because both of its axes are hidden sizes:
//
//   fused QKV   K = hidden, N = (num_q_heads + 2 * num_kv_heads) * head_dim,
//               which is neither a hidden size nor an MLP width. 6144 is the
//               32h/8kv/d128 split, 4608 is 28h/4kv/d128.
//   down_proj   K = the MLP intermediate width, so K is 11008 rather than 2048
//               or 4096. This is the deepest reduction in the whole forward
//               pass and the only projection whose K exceeds its N.
//
// The parity suite covers neither; see csrc/tests/COVERAGE.md.
const ProjectionShape kExtraProjectionShapes[] = {
    ProjectionShape{4096, 6144},   // fused QKV, 32 heads / 8 kv / d128
    ProjectionShape{2048, 4608},   // fused QKV, 28 heads / 4 kv / d128
    ProjectionShape{11008, 4096},  // down_proj
    ProjectionShape{11008, 2048},  // down_proj, narrower model
};

// The extra shapes are swept over one decode and one prefill M rather than all
// four, to keep the suite inside its ctest timeout: what they are there to show
// is the K and N dependence, which one point on each side of the crossover
// already gives.
const int64_t kExtraProjectionTokenCounts[] = {1, 128};

// M = 1 first so the decode numbers are at the top of the table, then the
// prefill batch sizes, then the two projections the sweep cannot express.
std::vector<ProjectionCase> BuildCaseList() {
  std::vector<int64_t> token_counts{shapes::kDecodeTokenCount};
  for (int64_t tokens : shapes::PrefillTokenCounts()) {
    token_counts.push_back(tokens);
  }

  std::vector<ProjectionCase> cases;
  for (int64_t m : token_counts) {
    for (int64_t k : shapes::LinearInputSizes()) {
      for (int64_t n : shapes::LinearOutputSizes()) {
        cases.push_back(ProjectionCase{m, k, n});
      }
    }
  }
  for (int64_t m : kExtraProjectionTokenCounts) {
    for (const ProjectionShape& shape : kExtraProjectionShapes) {
      cases.push_back(ProjectionCase{m, shape.k, shape.n});
    }
  }
  return cases;
}

}  // namespace

void BuildSuite(BenchmarkRunner& runner) {
  const AclnnOp& op = MatmulOp();
  if (!op.available()) {
    runner.Skip("all shapes", op.unavailable_reason());
    return;
  }

  DeterministicRandom random(0x424d554cu);  // "BMUL"

  for (const ProjectionCase& projection : BuildCaseList()) {
    const int64_t m = projection.m;
    const int64_t k = projection.k;
    const int64_t n = projection.n;
    const std::string name = CaseName(m, k, n);
    try {
      // Weights drawn with stddev 1/sqrt(K), which is both how they are
      // initialised and what keeps the output near unit magnitude. The
      // magnitude does not change the timing, but it keeps the checksum away
      // from the fp16 overflow that would make it useless as a guard.
      const float weight_stddev = 1.0f / std::sqrt(static_cast<float>(k));
      const std::vector<float> a = random.NormalHalfExact(static_cast<size_t>(m * k), 0.0f, 1.0f);
      const std::vector<float> b_t =
          random.NormalHalfExact(static_cast<size_t>(n * k), 0.0f, weight_stddev);

      // Setup phase: everything below is allocated once and reused by every
      // launch in the timed loop.
      DeviceTensor a_device = DeviceTensor::Half({m, k}, a, ACL_FORMAT_ND, kBenchmarkAlignBytes);
      DeviceTensor b_device = DeviceTensor::HalfTransposed2D(n, k, b_t, kBenchmarkAlignBytes);
      DeviceTensor out_device = DeviceTensor::HalfEmpty({m, n}, ACL_FORMAT_ND, kBenchmarkAlignBytes);

      PlannedOp planned = PlanAclnn<ops::MatmulWorkspaceFn>(
          op, a_device.get(), b_device.get(), out_device.get(), ops::kCubeMathTypeKeepDtype);

      BenchmarkCase benchmark_case;
      benchmark_case.name = name;
      // One multiply and one add per (m, n, k) triple.
      benchmark_case.flops_per_iteration =
          2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
      // A, B and the output, all fp16, each touched once. The cube unit re-reads
      // tiles from L1/L0, so this is traffic at the HBM boundary assuming
      // perfect reuse, i.e. a lower bound on what actually moves.
      benchmark_case.bytes_per_iteration =
          2.0 * (static_cast<double>(m) * static_cast<double>(k) +
                 static_cast<double>(k) * static_cast<double>(n) +
                 static_cast<double>(m) * static_cast<double>(n));
      benchmark_case.launch = [&planned](aclrtStream stream) { planned.Launch(stream); };
      // Same inputs every launch, so the output must be bit-identical.
      benchmark_case.checksum = [&out_device]() { return ChecksumSum(out_device.ToFloatFromHalf()); };

      runner.Run(benchmark_case);
    } catch (const std::exception& error) {
      runner.RecordFailure(name, error.what());
    }
  }
}

}  // namespace bench
}  // namespace test
}  // namespace vllm_ascend
