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

// RMSNorm (aclnnRmsNorm) on the v200 vector unit, fp16 in / fp16 out.
//
// One pass over the row to accumulate the sum of squares and a second to scale
// it, so there is no reuse to exploit and the kernel is bandwidth-bound at
// every shape here. GB/s is therefore the number that means something; the FLOP
// count is a handful of operations per element and is not reported.
//
// The bytes counted are the ones that must cross the HBM boundary: x read once,
// y written once, gamma read once per row batch (it is small enough to stay
// resident, so counting it once rather than once per token is the honest lower
// bound), and the fp32 rstd written once per token. A kernel that fuses the two
// passes reaches this figure; one that spills the row to GM between them moves
// twice as much and will show up as roughly half the bandwidth.

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

const char* kSuiteName = "rmsnorm_310p (aclnnRmsNorm, vector unit, fp16)";

namespace {

const AclnnOp& RmsNormOp() {
  static const AclnnOp op(ops::kRmsNorm);
  return op;
}

std::string CaseName(int64_t num_tokens, int64_t hidden) {
  std::ostringstream name;
  name << "tokens" << num_tokens << "_hidden" << hidden;
  return name.str();
}

}  // namespace

void BuildSuite(BenchmarkRunner& runner) {
  const AclnnOp& op = RmsNormOp();
  if (!op.available()) {
    runner.Skip("all shapes", op.unavailable_reason());
    return;
  }

  DeterministicRandom random(0x42524d53u);  // "BRMS"

  for (int64_t num_tokens : shapes::BenchmarkTokenCounts()) {
    for (int64_t hidden : shapes::RmsNormHiddenSizes()) {
      const std::string name = CaseName(num_tokens, hidden);
      try {
        const std::vector<float> x =
            random.NormalHalfExact(static_cast<size_t>(num_tokens * hidden), 0.0f, 1.0f);
        const std::vector<float> gamma =
            random.NormalHalfExact(static_cast<size_t>(hidden), 1.0f, 0.1f);

        DeviceTensor x_device =
            DeviceTensor::Half({num_tokens, hidden}, x, ACL_FORMAT_ND, kBenchmarkAlignBytes);
        DeviceTensor gamma_device =
            DeviceTensor::Half({hidden}, gamma, ACL_FORMAT_ND, kBenchmarkAlignBytes);
        DeviceTensor y_device =
            DeviceTensor::HalfEmpty({num_tokens, hidden}, ACL_FORMAT_ND, kBenchmarkAlignBytes);
        // Required output even though the plugin discards it, and it is fp32.
        DeviceTensor rstd_device =
            DeviceTensor::FloatEmpty({num_tokens, 1}, ACL_FORMAT_ND, kBenchmarkAlignBytes);

        PlannedOp planned = PlanAclnn<ops::RmsNormWorkspaceFn>(
            op, x_device.get(), gamma_device.get(), static_cast<double>(shapes::kRmsNormEpsilon),
            y_device.get(), rstd_device.get());

        BenchmarkCase benchmark_case;
        benchmark_case.name = name;
        benchmark_case.bytes_per_iteration =
            2.0 * static_cast<double>(num_tokens) * static_cast<double>(hidden) * 2.0  // x in, y out
            + 2.0 * static_cast<double>(hidden)                                        // gamma
            + 4.0 * static_cast<double>(num_tokens);                                   // rstd, fp32
        benchmark_case.launch = [&planned](aclrtStream stream) { planned.Launch(stream); };
        benchmark_case.checksum = [&y_device]() { return ChecksumSum(y_device.ToFloatFromHalf()); };

        runner.Run(benchmark_case);
      } catch (const std::exception& error) {
        runner.RecordFailure(name, error.what());
      }
    }
  }
}

}  // namespace bench
}  // namespace test
}  // namespace vllm_ascend
