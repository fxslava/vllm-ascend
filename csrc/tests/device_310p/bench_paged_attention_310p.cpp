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

#include <algorithm>
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
#include "qwen_shapes.hpp"
#include "random_data.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {

const char* kSuiteName = "paged_attention_310p (aclnnScatterPaKvCache, 5-D NZ cache, fp16)";

namespace {

using reference::PagedKvLayout;

constexpr aclFormat kKvCacheFormat = ACL_FORMAT_FRACTAL_NZ;

char kCacheModeNorm[] = "Norm";

const AclnnOp& ScatterPaKvCacheOp() {
  static const AclnnOp op(ops::kScatterPaKvCache);
  return op;
}

struct KvConfiguration {
  const char* label;
  int64_t num_kv_heads;
  int64_t head_size;
};

const KvConfiguration kKvConfigurations[] = {
    KvConfiguration{"kv8_d128", 8, 128},
    KvConfiguration{"kv4_d128", 4, 128},
    KvConfiguration{"kv2_d64", 2, 64},
};

const int64_t kTokenCounts[] = {1, 32, 128, 512};

std::vector<int64_t> KvCacheDims(const PagedKvLayout& layout) {
  return {layout.num_blocks, layout.fractal_rows(), layout.block_size, shapes::kKvCacheFractalWidth};
}

std::string CaseName(const KvConfiguration& configuration, int64_t block_size, int64_t num_tokens) {
  std::ostringstream name;
  name << "tokens" << num_tokens << "_" << configuration.label << "_b" << block_size;
  return name.str();
}

}

void BuildSuite(BenchmarkRunner& runner) {
  runner.Skip("paged attention decode",
              "aclnnPagedAttention is not provided by CANN 9.1.0; torch_npu._npu_paged_attention is backed by "
              "ATB (libatb.so), which the aclnn launch path cannot drive. See csrc/tests/common/aclnn_ops.hpp "
              "for the verified aclnnIncreFlashAttentionV4 prototype and what wiring it would take.");

  const AclnnOp& op = ScatterPaKvCacheOp();
  if (!op.available()) {
    runner.Skip("all cache-write shapes", op.unavailable_reason());
    return;
  }

  DeterministicRandom random(0x42504143u);

  for (const KvConfiguration& configuration : kKvConfigurations) {
    for (int64_t block_size : shapes::Supported310PBlockSizes()) {
      if (!shapes::IsValid310PBlockSize(block_size, configuration.head_size)) {
        std::ostringstream reason;
        reason << "block_size * head_size = " << block_size * configuration.head_size << " exceeds the 310P "
               << "limit of " << shapes::kAttentionBlockSizeLimit;
        runner.Skip(CaseName(configuration, block_size, 0), reason.str());
        continue;
      }

      for (int64_t num_tokens : kTokenCounts) {
        const std::string name = CaseName(configuration, block_size, num_tokens);
        try {
          PagedKvLayout layout;
          layout.block_size = block_size;
          layout.num_kv_heads = configuration.num_kv_heads;
          layout.head_size = configuration.head_size;
          const int64_t minimum_blocks = (num_tokens + block_size - 1) / block_size;
          layout.num_blocks = std::max<int64_t>(4, minimum_blocks * 4);

          const size_t kv_elements = static_cast<size_t>(num_tokens) *
                                     static_cast<size_t>(configuration.num_kv_heads) *
                                     static_cast<size_t>(configuration.head_size);
          const std::vector<float> key = random.NormalHalfExact(kv_elements, 0.0f, 1.0f);
          const std::vector<float> value = random.NormalHalfExact(kv_elements, 0.0f, 1.0f);

          const std::vector<int32_t> pool =
              random.Permutation(static_cast<int32_t>(layout.num_blocks * block_size));
          const std::vector<int32_t> slot_mapping(pool.begin(),
                                                  pool.begin() + static_cast<size_t>(num_tokens));

          DeviceTensor key_device =
              DeviceTensor::Half({num_tokens, configuration.num_kv_heads, configuration.head_size}, key,
                                 ACL_FORMAT_ND, kBenchmarkAlignBytes);
          DeviceTensor value_device =
              DeviceTensor::Half({num_tokens, configuration.num_kv_heads, configuration.head_size}, value,
                                 ACL_FORMAT_ND, kBenchmarkAlignBytes);
          DeviceTensor slot_device =
              DeviceTensor::Int32({num_tokens}, slot_mapping, ACL_FORMAT_ND, kBenchmarkAlignBytes);

          const std::vector<int64_t> cache_dims = KvCacheDims(layout);
          DeviceTensor key_cache_device =
              DeviceTensor::HalfEmpty(cache_dims, kKvCacheFormat, kBenchmarkAlignBytes);
          DeviceTensor value_cache_device =
              DeviceTensor::HalfEmpty(cache_dims, kKvCacheFormat, kBenchmarkAlignBytes);

          PlannedOp planned = PlanAclnn<ops::ScatterPaKvCacheWorkspaceFn>(
              op, key_device.get(), key_cache_device.get(), slot_device.get(), value_device.get(),
              value_cache_device.get(), static_cast<const aclTensor*>(nullptr),
              static_cast<const aclTensor*>(nullptr), static_cast<const aclTensor*>(nullptr), kCacheModeNorm,
              static_cast<char*>(nullptr), static_cast<const aclIntArray*>(nullptr),
              static_cast<const aclIntArray*>(nullptr));

          BenchmarkCase benchmark_case;
          benchmark_case.name = name;
          benchmark_case.bytes_per_iteration =
              2.0 * 2.0 * 2.0 * static_cast<double>(kv_elements) + 4.0 * static_cast<double>(num_tokens);
          benchmark_case.launch = [&planned](aclrtStream stream) { planned.Launch(stream); };
          benchmark_case.checksum = [&key_cache_device]() {
            return ChecksumSum(key_cache_device.ToFloatFromHalf());
          };

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
}
