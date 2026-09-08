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

#include <ATen/ATen.h>
#include <acl/acl.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <algorithm>
#include <cmath>

#include "../../kernels/types.h"

namespace vllm_ascend {

extern void turboquant_reshape_and_cache_impl(AscendType type, void *stream, uint32_t blockDim, void *key, void *value,
                                              void *keyCache, void *valueCache, void *keyScale, void *valueScale,
                                              void *slotMapping, void *piSigns, uint32_t numTokens,
                                              uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,
                                              uint32_t tokensPerCore, float invSqrtLen);

extern void turboquant_paged_attention_impl(AscendType type, void *stream, uint32_t blockDim, void *query,
                                            void *keyCache, void *valueCache, void *keyScale, void *valueScale,
                                            void *blockTables, void *contextLens, void *piSigns, void *workspace,
                                            void *output, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                            uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                            uint32_t numSplits, uint32_t splitTasksPerCore,
                                            uint32_t reduceTasksPerCore, float scale, float invSqrtLen);

namespace turboquant_adpt {

// Bounds the flash-decoding workspace; beyond this the reduce stage costs more
// than the extra parallelism buys.
constexpr int64_t kMaxSequenceSplits = 8;
// fp32 words appended to every partial accumulator: the running max, the
// running sum, and padding out to a 32B block.
constexpr int64_t kPartialTail = 8;

inline AscendType ToAscendType(at::ScalarType scalarType)
{
    TORCH_CHECK(scalarType == at::ScalarType::Half || scalarType == at::ScalarType::BFloat16,
                "TurboQuant KV cache supports float16 and bfloat16 activations, got ", scalarType);
    return scalarType == at::ScalarType::BFloat16 ? AscendType::BF16 : AscendType::FP16;
}

inline int64_t VectorCoreNum()
{
    int64_t aivNum = 0;
    TORCH_CHECK(aclGetDeviceCapability(0, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aivNum) == ACL_SUCCESS,
                "failed to query ACL_DEVICE_INFO_VECTOR_CORE_NUM");
    TORCH_CHECK(aivNum > 0, "device reported a non-positive vector core count");
    return aivNum;
}

inline void CheckHeadSize(int64_t headSize)
{
    TORCH_CHECK(headSize >= 64 && headSize <= 256, "TurboQuant requires 64 <= head_size <= 256, got ", headSize);
    TORCH_CHECK((headSize & (headSize - 1)) == 0, "TurboQuant needs a power-of-two head_size for the Walsh-Hadamard "
                                                  "transform, got ",
                headSize);
}

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

}  // namespace turboquant_adpt

/*
 * Rotate, quantise to 4 bits and scatter K/V into the ND paged cache.
 *
 *   key, value      [num_tokens, num_kv_heads, head_size]                 fp16 | bf16
 *   key/value cache [num_blocks, block_size, num_kv_heads, head_size / 2] int8
 *   key/value scale [num_blocks, num_kv_heads, block_size]                fp32
 *   slot_mapping    [num_tokens]                                          int32, -1 skips
 *   pi_signs        [head_size]                                           fp32, +-1
 *
 * `key` is the post-RoPE key and `value` the raw value projection, both exactly
 * as the model produced them: the kernel applies Pi in UB and no weight is ever
 * rewritten, so RoPE stays correct.
 */
inline void npu_turboquant_reshape_and_cache(at::Tensor &key, at::Tensor &value, at::Tensor &key_cache,
                                             at::Tensor &value_cache, at::Tensor &key_scale, at::Tensor &value_scale,
                                             at::Tensor &slot_mapping, at::Tensor &pi_signs)
{
    namespace adpt = turboquant_adpt;

    TORCH_CHECK(key.dim() == 3 && value.dim() == 3, "key/value must be [num_tokens, num_kv_heads, head_size]");
    TORCH_CHECK(key.sizes() == value.sizes(), "key and value must have the same shape");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
                "kv cache must be [num_blocks, block_size, num_kv_heads, head_size / 2]");
    TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "key and value caches must have the same shape");
    TORCH_CHECK(key_cache.scalar_type() == at::ScalarType::Char && value_cache.scalar_type() == at::ScalarType::Char,
                "the packed 4-bit kv cache must be int8");
    TORCH_CHECK(key_scale.scalar_type() == at::ScalarType::Float && value_scale.scalar_type() == at::ScalarType::Float,
                "kv cache scales must be float32");
    TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Int, "slot_mapping must be int32");
    TORCH_CHECK(pi_signs.scalar_type() == at::ScalarType::Float, "pi_signs must be float32");
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous(), "key/value must be contiguous");

    const int64_t num_tokens = key.size(0);
    const int64_t num_kv_heads = key.size(1);
    const int64_t head_size = key.size(2);
    const int64_t num_blocks = key_cache.size(0);
    const int64_t block_size = key_cache.size(1);

    adpt::CheckHeadSize(head_size);
    TORCH_CHECK(key_cache.size(2) == num_kv_heads, "cache num_kv_heads mismatch: ", key_cache.size(2), " vs ",
                num_kv_heads);
    TORCH_CHECK(key_cache.size(3) == head_size / 2, "packed cache head dim must be head_size / 2, got ",
                key_cache.size(3));
    TORCH_CHECK(block_size % 16 == 0, "TurboQuant requires block_size to be a multiple of 16, got ", block_size);
    TORCH_CHECK(slot_mapping.numel() == num_tokens, "slot_mapping must hold one slot per token");
    TORCH_CHECK(pi_signs.numel() == head_size, "pi_signs must hold one sign per channel");
    TORCH_CHECK(key_scale.numel() == num_blocks * num_kv_heads * block_size,
                "key_scale must be [num_blocks, num_kv_heads, block_size]");
    TORCH_CHECK(value_scale.numel() == key_scale.numel(), "value_scale must match key_scale");

    if (num_tokens == 0) {
        return;
    }

    const at::ScalarType scalar_type = key.scalar_type();
    const int64_t aiv_num = adpt::VectorCoreNum();
    const int64_t tokens_per_core = adpt::CeilDiv(num_tokens, aiv_num);
    const uint32_t block_dim = static_cast<uint32_t>(adpt::CeilDiv(num_tokens, tokens_per_core));
    const float inv_sqrt_len = 1.0f / std::sqrt(static_cast<float>(head_size));

    void *key_ptr = key.data_ptr();
    void *value_ptr = value.data_ptr();
    void *key_cache_ptr = key_cache.data_ptr();
    void *value_cache_ptr = value_cache.data_ptr();
    void *key_scale_ptr = key_scale.data_ptr();
    void *value_scale_ptr = value_scale.data_ptr();
    void *slot_ptr = slot_mapping.data_ptr();
    void *signs_ptr = pi_signs.data_ptr();
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    at_npu::native::OpCommand cmd;
    cmd.Name("npu_turboquant_reshape_and_cache");
    cmd.SetCustomHandler([scalar_type, stream, block_dim, key_ptr, value_ptr, key_cache_ptr, value_cache_ptr,
                          key_scale_ptr, value_scale_ptr, slot_ptr, signs_ptr, num_tokens, num_kv_heads, head_size,
                          block_size, tokens_per_core, inv_sqrt_len]() -> int {
        turboquant_reshape_and_cache_impl(
            turboquant_adpt::ToAscendType(scalar_type), stream, block_dim, key_ptr, value_ptr, key_cache_ptr,
            value_cache_ptr, key_scale_ptr, value_scale_ptr, slot_ptr, signs_ptr, static_cast<uint32_t>(num_tokens),
            static_cast<uint32_t>(num_kv_heads), static_cast<uint32_t>(head_size), static_cast<uint32_t>(block_size),
            static_cast<uint32_t>(tokens_per_core), inv_sqrt_len);
        return 0;
    });
    cmd.Run();
}

/*
 * Paged decode attention over the 4-bit cache.
 *
 *   query        [num_tokens, num_heads, head_size]  fp16 | bf16
 *   block_tables [num_tokens, max_blocks_per_seq]    int32
 *   context_lens [num_tokens]                        int32
 *   out          [num_tokens, num_heads, head_size]  same dtype as query
 *
 * `query` is the post-RoPE query.  The kernel rotates it once, runs the whole
 * softmax and value accumulation in the rotated basis, and applies Pi to the
 * accumulator again on the way out -- Pi being an involution, that is the
 * un-rotation.
 */
inline void npu_turboquant_paged_attention(at::Tensor &query, at::Tensor &key_cache, at::Tensor &value_cache,
                                           at::Tensor &key_scale, at::Tensor &value_scale, at::Tensor &block_tables,
                                           at::Tensor &context_lens, at::Tensor &pi_signs, int64_t num_kv_heads,
                                           int64_t num_heads, double scale_value, at::Tensor &out)
{
    namespace adpt = turboquant_adpt;

    TORCH_CHECK(query.dim() == 3, "query must be [num_tokens, num_heads, head_size]");
    TORCH_CHECK(out.sizes() == query.sizes(), "out must have the same shape as query");
    TORCH_CHECK(out.scalar_type() == query.scalar_type(), "out must have the same dtype as query");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
                "kv cache must be [num_blocks, block_size, num_kv_heads, head_size / 2]");
    TORCH_CHECK(key_cache.scalar_type() == at::ScalarType::Char && value_cache.scalar_type() == at::ScalarType::Char,
                "the packed 4-bit kv cache must be int8");
    TORCH_CHECK(block_tables.dim() == 2 && block_tables.scalar_type() == at::ScalarType::Int,
                "block_tables must be an int32 [num_tokens, max_blocks_per_seq] tensor");
    TORCH_CHECK(context_lens.scalar_type() == at::ScalarType::Int, "context_lens must be int32");
    TORCH_CHECK(pi_signs.scalar_type() == at::ScalarType::Float, "pi_signs must be float32");
    TORCH_CHECK(query.is_contiguous() && out.is_contiguous(), "query and out must be contiguous");

    const int64_t num_tokens = query.size(0);
    const int64_t head_size = query.size(2);
    const int64_t block_size = key_cache.size(1);
    const int64_t max_blocks_per_seq = block_tables.size(1);

    adpt::CheckHeadSize(head_size);
    TORCH_CHECK(query.size(1) == num_heads, "query head count ", query.size(1), " does not match num_heads ",
                num_heads);
    TORCH_CHECK(num_kv_heads > 0 && num_heads % num_kv_heads == 0, "num_heads ", num_heads,
                " must be a multiple of num_kv_heads ", num_kv_heads);
    TORCH_CHECK(key_cache.size(2) == num_kv_heads, "cache num_kv_heads mismatch: ", key_cache.size(2), " vs ",
                num_kv_heads);
    TORCH_CHECK(key_cache.size(3) == head_size / 2, "packed cache head dim must be head_size / 2, got ",
                key_cache.size(3));
    TORCH_CHECK(block_size % 16 == 0, "TurboQuant requires block_size to be a multiple of 16, got ", block_size);
    TORCH_CHECK(block_tables.size(0) == num_tokens, "block_tables must hold one row per query token");
    TORCH_CHECK(context_lens.numel() == num_tokens, "context_lens must hold one length per query token");
    TORCH_CHECK(pi_signs.numel() == head_size, "pi_signs must hold one sign per channel");

    if (num_tokens == 0) {
        return;
    }

    const int64_t aiv_num = adpt::VectorCoreNum();
    const int64_t base_tasks = num_tokens * num_heads;
    int64_t num_splits = adpt::CeilDiv(aiv_num, base_tasks);
    num_splits = std::min(num_splits, std::min<int64_t>(adpt::kMaxSequenceSplits, std::max<int64_t>(1,
                                                                                                    max_blocks_per_seq)));
    num_splits = std::max<int64_t>(num_splits, 1);

    const int64_t partial_stride = head_size + adpt::kPartialTail;
    at::Tensor workspace = at::empty({base_tasks * num_splits * partial_stride}, query.options().dtype(at::kFloat));

    const uint32_t block_dim = static_cast<uint32_t>(aiv_num);
    const int64_t split_tasks_per_core = adpt::CeilDiv(base_tasks * num_splits, aiv_num);
    const int64_t reduce_tasks_per_core = adpt::CeilDiv(base_tasks, aiv_num);
    const float inv_sqrt_len = 1.0f / std::sqrt(static_cast<float>(head_size));
    const float scale = static_cast<float>(scale_value);

    const at::ScalarType scalar_type = query.scalar_type();
    void *query_ptr = query.data_ptr();
    void *key_cache_ptr = key_cache.data_ptr();
    void *value_cache_ptr = value_cache.data_ptr();
    void *key_scale_ptr = key_scale.data_ptr();
    void *value_scale_ptr = value_scale.data_ptr();
    void *block_tables_ptr = block_tables.data_ptr();
    void *context_lens_ptr = context_lens.data_ptr();
    void *signs_ptr = pi_signs.data_ptr();
    void *workspace_ptr = workspace.data_ptr();
    void *out_ptr = out.data_ptr();
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    at_npu::native::OpCommand cmd;
    cmd.Name("npu_turboquant_paged_attention");
    cmd.SetCustomHandler([scalar_type, stream, block_dim, query_ptr, key_cache_ptr, value_cache_ptr, key_scale_ptr,
                          value_scale_ptr, block_tables_ptr, context_lens_ptr, signs_ptr, workspace_ptr, out_ptr,
                          num_tokens, num_heads, num_kv_heads, head_size, block_size, max_blocks_per_seq, num_splits,
                          split_tasks_per_core, reduce_tasks_per_core, scale, inv_sqrt_len]() -> int {
        turboquant_paged_attention_impl(
            turboquant_adpt::ToAscendType(scalar_type), stream, block_dim, query_ptr, key_cache_ptr, value_cache_ptr,
            key_scale_ptr, value_scale_ptr, block_tables_ptr, context_lens_ptr, signs_ptr, workspace_ptr, out_ptr,
            static_cast<uint32_t>(num_tokens), static_cast<uint32_t>(num_heads), static_cast<uint32_t>(num_kv_heads),
            static_cast<uint32_t>(head_size), static_cast<uint32_t>(block_size),
            static_cast<uint32_t>(max_blocks_per_seq), static_cast<uint32_t>(num_splits),
            static_cast<uint32_t>(split_tasks_per_core), static_cast<uint32_t>(reduce_tasks_per_core), scale,
            inv_sqrt_len);
        return 0;
    });
    cmd.Run();
}

}  // namespace vllm_ascend
