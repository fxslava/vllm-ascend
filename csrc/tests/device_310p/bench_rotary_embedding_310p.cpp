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
#include "cpu_reference.hpp"
#include "device_tensor.hpp"
#include "fp16.hpp"
#include "qwen_shapes.hpp"
#include "random_data.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {

const char* kSuiteName = "rotary_embedding_310p (aclnnApplyRotaryPosEmbV2, vector unit, fp16)";

namespace {

using reference::RotaryMode;

char kRotaryModeHalf[] = "half";
char kRotaryModeInterleave[] = "interleave";

char* RotaryModeString(RotaryMode mode) {
  return (mode == RotaryMode::kHalf) ? kRotaryModeHalf : kRotaryModeInterleave;
}

const char* RotaryModeLabel(RotaryMode mode) { return (mode == RotaryMode::kHalf) ? "half" : "interleave"; }

const AclnnOp& ApplyRotaryPosEmbOp() {
  static const AclnnOp op = ops::ResolveFirstAvailable({ops::kApplyRotaryPosEmbV2, ops::kApplyRotaryPosEmb});
  return op;
}

struct HeadConfiguration {
  int64_t head_dim;
  int64_t num_q_heads;
  int64_t num_kv_heads;
  RotaryMode mode;
};

const HeadConfiguration kHeadConfigurations[] = {
    HeadConfiguration{128, 28, 4, RotaryMode::kHalf},
    HeadConfiguration{128, 32, 8, RotaryMode::kHalf},
    HeadConfiguration{128, 32, 8, RotaryMode::kInterleave},
    HeadConfiguration{64, 16, 2, RotaryMode::kHalf},
    HeadConfiguration{64, 16, 2, RotaryMode::kInterleave},
};

std::string CaseName(int64_t num_tokens, const HeadConfiguration& configuration) {
  std::ostringstream name;
  name << "tokens" << num_tokens << "_d" << configuration.head_dim << "_q" << configuration.num_q_heads << "_kv"
       << configuration.num_kv_heads << "_" << RotaryModeLabel(configuration.mode);
  return name.str();
}

}

void BuildSuite(BenchmarkRunner& runner) {
  const AclnnOp& op = ApplyRotaryPosEmbOp();
  if (!op.available()) {
    runner.Skip("all shapes", op.unavailable_reason());
    return;
  }

  DeterministicRandom random(0x42524f50u);

  for (int64_t num_tokens : shapes::BenchmarkTokenCounts()) {
    for (const HeadConfiguration& configuration : kHeadConfigurations) {
      const std::string name = CaseName(num_tokens, configuration);
      try {
        const int64_t head_dim = configuration.head_dim;
        const std::vector<float> query = random.NormalHalfExact(
            static_cast<size_t>(num_tokens * configuration.num_q_heads * head_dim), 0.0f, 1.0f);
        const std::vector<float> key = random.NormalHalfExact(
            static_cast<size_t>(num_tokens * configuration.num_kv_heads * head_dim), 0.0f, 1.0f);

        std::vector<int32_t> positions(static_cast<size_t>(num_tokens));
        for (int64_t i = 0; i < num_tokens; ++i) {
          positions[static_cast<size_t>(i)] =
              static_cast<int32_t>((i * 37 + 11) % shapes::kMaxPositionEmbeddings);
        }

        const std::vector<float> cache = reference::BuildCosSinCache(shapes::kMaxPositionEmbeddings, head_dim,
                                                                    shapes::kRopeThetaDefault);
        std::vector<float> cos_full;
        std::vector<float> sin_full;
        reference::GatherFullCosSin(cache, positions, head_dim, configuration.mode, &cos_full, &sin_full);

        DeviceTensor query_device =
            DeviceTensor::Half({1, num_tokens, configuration.num_q_heads, head_dim}, query, ACL_FORMAT_ND,
                               kBenchmarkAlignBytes);
        DeviceTensor key_device =
            DeviceTensor::Half({1, num_tokens, configuration.num_kv_heads, head_dim}, key, ACL_FORMAT_ND,
                               kBenchmarkAlignBytes);
        DeviceTensor cos_device = DeviceTensor::Half({1, num_tokens, 1, head_dim}, QuantizeToHalf(cos_full),
                                                     ACL_FORMAT_ND, kBenchmarkAlignBytes);
        DeviceTensor sin_device = DeviceTensor::Half({1, num_tokens, 1, head_dim}, QuantizeToHalf(sin_full),
                                                     ACL_FORMAT_ND, kBenchmarkAlignBytes);

        PlannedOp planned = PlanAclnn<ops::ApplyRotaryPosEmbWorkspaceFn>(
            op, query_device.get(), key_device.get(), cos_device.get(), sin_device.get(),
            ops::kApplyRotaryPosEmbLayoutBsnd, RotaryModeString(configuration.mode));

        BenchmarkCase benchmark_case;
        benchmark_case.name = name;
        const double q_elements =
            static_cast<double>(num_tokens) * static_cast<double>(configuration.num_q_heads) *
            static_cast<double>(head_dim);
        const double k_elements =
            static_cast<double>(num_tokens) * static_cast<double>(configuration.num_kv_heads) *
            static_cast<double>(head_dim);
        const double cos_sin_elements = 2.0 * static_cast<double>(num_tokens) * static_cast<double>(head_dim);
        benchmark_case.bytes_per_iteration = 2.0 * (2.0 * (q_elements + k_elements) + cos_sin_elements);
        benchmark_case.launch = [&planned](aclrtStream stream) { planned.Launch(stream); };
        benchmark_case.checksum = [&query_device, &key_device]() {
          return ChecksumSumOfSquares(query_device.ToFloatFromHalf()) +
                 ChecksumSumOfSquares(key_device.ToFloatFromHalf());
        };
        benchmark_case.checksum_rtol = 5e-2;

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
