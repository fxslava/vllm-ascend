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

const ProjectionShape kExtraProjectionShapes[] = {
    ProjectionShape{4096, 6144},
    ProjectionShape{2048, 4608},
    ProjectionShape{11008, 4096},
    ProjectionShape{11008, 2048},
};

const int64_t kExtraProjectionTokenCounts[] = {1, 128};

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

}

void BuildSuite(BenchmarkRunner& runner) {
  const AclnnOp& op = MatmulOp();
  if (!op.available()) {
    runner.Skip("all shapes", op.unavailable_reason());
    return;
  }

  DeterministicRandom random(0x424d554cu);

  for (const ProjectionCase& projection : BuildCaseList()) {
    const int64_t m = projection.m;
    const int64_t k = projection.k;
    const int64_t n = projection.n;
    const std::string name = CaseName(m, k, n);
    try {
      const float weight_stddev = 1.0f / std::sqrt(static_cast<float>(k));
      const std::vector<float> a = random.NormalHalfExact(static_cast<size_t>(m * k), 0.0f, 1.0f);
      const std::vector<float> b_t =
          random.NormalHalfExact(static_cast<size_t>(n * k), 0.0f, weight_stddev);

      DeviceTensor a_device = DeviceTensor::Half({m, k}, a, ACL_FORMAT_ND, kBenchmarkAlignBytes);
      DeviceTensor b_device = DeviceTensor::HalfTransposed2D(n, k, b_t, kBenchmarkAlignBytes);
      DeviceTensor out_device = DeviceTensor::HalfEmpty({m, n}, ACL_FORMAT_ND, kBenchmarkAlignBytes);

      PlannedOp planned = PlanAclnn<ops::MatmulWorkspaceFn>(
          op, a_device.get(), b_device.get(), out_device.get(), ops::kCubeMathTypeKeepDtype);

      BenchmarkCase benchmark_case;
      benchmark_case.name = name;
      benchmark_case.flops_per_iteration =
          2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
      benchmark_case.bytes_per_iteration =
          2.0 * (static_cast<double>(m) * static_cast<double>(k) +
                 static_cast<double>(k) * static_cast<double>(n) +
                 static_cast<double>(m) * static_cast<double>(n));
      benchmark_case.launch = [&planned](aclrtStream stream) { planned.Launch(stream); };
      benchmark_case.checksum = [&out_device]() { return ChecksumSum(out_device.ToFloatFromHalf()); };

      runner.Run(benchmark_case);
    } catch (const std::exception& error) {
      runner.RecordFailure(name, error.what());
    }
  }
}

}
}
}
