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

const char* kSuiteName = "activation_swiglu_310p (aclnnSwiGlu, vector unit, fp16)";

namespace {

constexpr int64_t kSwiGluSplitDim = -1;

const AclnnOp& SwiGluOp() {
  static const AclnnOp op(ops::kSwiGlu);
  return op;
}

std::string CaseName(int64_t num_tokens, int64_t intermediate) {
  std::ostringstream name;
  name << "tokens" << num_tokens << "_intermediate" << intermediate;
  return name.str();
}

}

void BuildSuite(BenchmarkRunner& runner) {
  const AclnnOp& op = SwiGluOp();
  if (!op.available()) {
    runner.Skip("all shapes", op.unavailable_reason());
    return;
  }

  DeterministicRandom random(0x42535747u);

  for (int64_t num_tokens : shapes::BenchmarkTokenCounts()) {
    for (int64_t intermediate : shapes::IntermediateSizes()) {
      const std::string name = CaseName(num_tokens, intermediate);
      try {
        const std::vector<float> x =
            random.NormalHalfExact(static_cast<size_t>(num_tokens * intermediate * 2), 0.0f, 2.0f);

        DeviceTensor x_device = DeviceTensor::Half({num_tokens, intermediate * 2}, x, ACL_FORMAT_ND,
                                                   kBenchmarkAlignBytes);
        DeviceTensor out_device =
            DeviceTensor::HalfEmpty({num_tokens, intermediate}, ACL_FORMAT_ND, kBenchmarkAlignBytes);

        PlannedOp planned = PlanAclnn<ops::SwiGluWorkspaceFn>(op, x_device.get(), kSwiGluSplitDim,
                                                              out_device.get());

        BenchmarkCase benchmark_case;
        benchmark_case.name = name;
        benchmark_case.bytes_per_iteration =
            2.0 * static_cast<double>(num_tokens) * static_cast<double>(intermediate) * 3.0;
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
}
