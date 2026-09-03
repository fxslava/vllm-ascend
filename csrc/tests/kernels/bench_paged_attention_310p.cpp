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

// The paged KV path on Ascend 310P: the cache write, benchmarked, and the
// decode attention, which cannot be.
//
// aclnnPagedAttention does not exist on CANN 9.1.0. torch_npu._npu_paged_attention
// is an ATB operator (atb::PagedAttentionOperation in libatb.so), a C++ object
// API that the aclnn two-phase launch path here cannot drive, and the aclnn
// alternative aclnnIncreFlashAttentionV4 expects a different paged KV layout
// from the 310P 5-D NZ cache. See common/aclnn_ops.hpp. The decode case is
// therefore registered as a skip with that reason rather than quietly omitted,
// so the report shows the hole.
//
// What is benchmarked is aclnnScatterPaKvCache, the aclnn route to
// torch_npu._npu_reshape_and_cache, which every decode step runs once per layer
// before attention. It is a pure scatter: each token's key and value are copied
// into the slot the block table assigned, with no arithmetic at all, so GB/s is
// the only meaningful figure.
//
// The slot mapping is a shuffle rather than a run of consecutive slots. That is
// the realistic case - a decode batch writes one token per sequence, and those
// sequences hold unrelated physical blocks - and it is also the expensive one,
// because consecutive slots would let the kernel coalesce writes that the real
// workload cannot.

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

// The runner allocates both halves of the KV cache with FRACTAL_NZ over the
// already-decomposed 4-D shape, so the descriptors carry the same tag.
constexpr aclFormat kKvCacheFormat = ACL_FORMAT_FRACTAL_NZ;

// cacheMode is a mutable char* in the verified prototype.
char kCacheModeNorm[] = "Norm";

const AclnnOp& ScatterPaKvCacheOp() {
  static const AclnnOp op(ops::kScatterPaKvCache);
  return op;
}

// The KV cache write depends only on num_kv_heads and head_size, not on the
// query head count, so the four GQA configurations in qwen_shapes.hpp collapse
// to these three distinct KV shapes.
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

// Tokens written per launch: one decode step of a batch this size, or a
// prefill chunk of this length.
const int64_t kTokenCounts[] = {1, 32, 128, 512};

std::vector<int64_t> KvCacheDims(const PagedKvLayout& layout) {
  return {layout.num_blocks, layout.fractal_rows(), layout.block_size, shapes::kKvCacheFractalWidth};
}

std::string CaseName(const KvConfiguration& configuration, int64_t block_size, int64_t num_tokens) {
  std::ostringstream name;
  name << "tokens" << num_tokens << "_" << configuration.label << "_b" << block_size;
  return name.str();
}

}  // namespace

void BuildSuite(BenchmarkRunner& runner) {
  // Register the hole first, so it is visible even if the scatter benchmarks
  // then fail for an unrelated reason.
  runner.Skip("paged attention decode",
              "aclnnPagedAttention is not provided by CANN 9.1.0; torch_npu._npu_paged_attention is backed by "
              "ATB (libatb.so), which the aclnn launch path cannot drive. See csrc/tests/common/aclnn_ops.hpp "
              "for the verified aclnnIncreFlashAttentionV4 prototype and what wiring it would take.");

  const AclnnOp& op = ScatterPaKvCacheOp();
  if (!op.available()) {
    runner.Skip("all cache-write shapes", op.unavailable_reason());
    return;
  }

  DeterministicRandom random(0x42504143u);  // "BPAC"

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
          // Four times as many blocks as the tokens strictly need, so the
          // shuffled slots land in genuinely unrelated blocks rather than in a
          // handful that would stay resident.
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

          // Argument order follows the header exactly: key, keyCache,
          // slotMapping, then value, valueCache, then the optional tensors.
          PlannedOp planned = PlanAclnn<ops::ScatterPaKvCacheWorkspaceFn>(
              op, key_device.get(), key_cache_device.get(), slot_device.get(), value_device.get(),
              value_cache_device.get(), static_cast<const aclTensor*>(nullptr),
              static_cast<const aclTensor*>(nullptr), static_cast<const aclTensor*>(nullptr), kCacheModeNorm,
              static_cast<char*>(nullptr), static_cast<const aclIntArray*>(nullptr),
              static_cast<const aclIntArray*>(nullptr));

          BenchmarkCase benchmark_case;
          benchmark_case.name = name;
          // key and value read, key cache and value cache written, plus the
          // int32 slot mapping. The cache is far larger than this, but only the
          // written slots are touched.
          benchmark_case.bytes_per_iteration =
              2.0 * 2.0 * 2.0 * static_cast<double>(kv_elements) + 4.0 * static_cast<double>(num_tokens);
          benchmark_case.launch = [&planned](aclrtStream stream) { planned.Launch(stream); };
          // The scatter is idempotent: the same values go to the same slots
          // every launch, so the cache must be bit-identical across the run.
          // The whole cache is summed rather than just the written slots -
          // DeviceBuffer zeroes on allocation, so the untouched slots contribute
          // nothing and a stray write outside the slot mapping is caught too.
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

}  // namespace bench
}  // namespace test
}  // namespace vllm_ascend
