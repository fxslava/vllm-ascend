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

#include <algorithm>
#include <cmath>
#include <utility>

#include "../../../kernels/types.h"
#include "../../../npu_device_registry.h"
#include "../op_host/turboquant_tiling.h"
#include "../op_kernel/common/turboquant_mode.h"

namespace vllm_ascend {

extern void turboquant_reshape_and_cache_impl(AscendType type, void *stream, uint32_t blockDim, void *key,
                                              void *value, void *keyCache, void *valueCache, void *scaleCache,
                                              void *slotMapping, void *piSigns, void *tables, uint32_t numTokens,
                                              uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,
                                              uint32_t numBlocks, uint32_t tokensPerCore, float invSqrtLen);

extern void turboquant_paged_attention_impl(AscendType type, void *stream, uint32_t blockDim, void *queryRot,
                                            void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
                                            void *contextLens, void *tables, void *workspace, void *output,
                                            uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                            uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                            uint32_t numSplits, uint32_t splitTasksPerCore,
                                            uint32_t reduceTasksPerCore, uint32_t fusedContextLimit, float scale,
                                            float invSqrtLen, void *lse = nullptr);

extern void turboquant_rotate_q_impl(AscendType type, void *stream, uint32_t blockDim, bool useCube, void *query,
                                     void *piSigns, void *h16, void *rotTables, void *queryRot, uint32_t numVectors,
                                     uint32_t headSize, uint32_t vectorsPerBlock, uint32_t vectorsPerChunk,
                                     uint32_t variant, float invSqrtLen);

#ifdef VLLM_ENABLE_TURBOQUANT_CUBE
extern void turboquant_mm_reshape_and_cache_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim,
                                                 void *key, void *value, void *keyCache, void *valueCache,
                                                 void *scaleCache, void *slotMapping, void *piSigns, void *rotTables,
                                                 void *modeTables, uint32_t numTokens, uint32_t numKvHeads,
                                                 uint32_t headSize, uint32_t blockSize, uint32_t numBlocks,
                                                 uint32_t tokensPerCore, float invSqrtLen);

extern void turboquant_mm_fused_decode_raw_query_impl(
    int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *query, void *piSigns, void *rotTables,
    void *h16, void *gate, void *queryRot, void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
    void *contextLens, void *modeTables, void *workspace, void *output, uint32_t numTokens, uint32_t numHeads,
    uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,
    uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock, uint32_t prologueVectorsPerBlock,
    uint32_t prologueCubeChunkVectors, uint32_t outputStage, uint32_t fusedContextLimit, float scale,
    float invSqrtLen, void *lse = nullptr);
#endif

namespace turboquant_adpt {

namespace tq = vllm_ascend::turboquant;

inline AscendType ToAscendType(at::ScalarType scalarType)
{
    TORCH_CHECK(scalarType == at::ScalarType::Half || scalarType == at::ScalarType::BFloat16,
                "TurboQuant KV cache supports float16 and bfloat16 activations, got ", scalarType);
    return scalarType == at::ScalarType::BFloat16 ? AscendType::BF16 : AscendType::FP16;
}

inline int64_t VectorCoreNum()
{
    return device_registry::VectorCoreNum();
}

inline void CheckHeadSize(int64_t headSize)
{
    TORCH_CHECK(headSize >= 64 && headSize <= 256, "TurboQuant requires 64 <= head_size <= 256, got ", headSize);
    TORCH_CHECK((headSize & (headSize - 1)) == 0, "TurboQuant needs a power-of-two head_size for the Walsh-Hadamard "
                                                  "transform, got ",
                headSize);
}

// Every global-memory copy the kernels make moves whole 32-byte bursts, so an operand whose address
// is not a multiple of one starts the copy mid-burst. The AI core reports that as "the address for
// scalar to access GM is invalid" (error 264) from inside the launch, naming nothing; caught here it
// names the tensor. A contiguity check does not cover it: a contiguous view keeps its storage offset,
// so `.contiguous()` can hand the operator an address part-way through a burst.
constexpr size_t kGmBurstBytes = tq::kFp32PerBlock * sizeof(float);

inline void CheckGmBurstAligned(const at::Tensor &tensor, const char *name)
{
    const auto address = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    TORCH_CHECK(address % kGmBurstBytes == 0, name, " starts at ", address, ", which is not a whole ",
                kGmBurstBytes, "-byte burst; the kernel's global-memory copies would start mid-burst");
}

// The check above is about MTE bursts, so it belongs to the operands the kernels DataCopy. An operand the
// kernels only read one element at a time with GlobalTensor::GetValue -- slot_mapping, and the decodes'
// block_tables and context_lens, which have never been burst-checked -- is a scalar load, and a scalar load
// needs its own element's alignment, nothing wider. Holding those to a 32-byte burst rejects sub-slices a
// caller is entitled to pass: `slot_mapping[1:]` off an int32 tensor is four bytes in, contiguous, and
// perfectly legal for every access the kernel makes of it.
inline void CheckGmScalarAligned(const at::Tensor &tensor, const char *name)
{
    const auto address = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    const auto element = static_cast<uintptr_t>(tensor.element_size());
    TORCH_CHECK(address % element == 0, name, " starts at ", address, ", which is not a multiple of its ",
                element, "-byte element; the kernel's scalar global-memory reads would be misaligned");
}

// Every operand below is handed to a kernel as a `__gm__` pointer, so it has to live in the
// device's global memory. A host tensor is not a wrong answer, it is "the address for scalar
// to access GM is invalid" (264) raised from whichever later launch the runtime happened to
// be working on -- and nothing about the tensor says so, because dtype, shape and contiguity
// are all exactly what the kernel wants. It is worth naming at the boundary: the metadata
// keeps some index tensors (seq_lens) on the host on purpose, since the CANN paged attention
// the stock backend calls reads its lengths host-side to tile with. These kernels do not.
inline void CheckOnDevice(const at::Tensor &tensor, const char *name, const at::Tensor &reference,
                          const char *reference_name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name, " is on ", tensor.device(), " but ", reference_name,
                " is on ", reference.device(),
                "; the kernel reads it from global memory, so a host tensor reaches the AI core as an invalid "
                "address rather than as a wrong answer");
}

// The decodes' optional softmax-statistics out-tensor: a strict [num_tokens, num_heads, kLseStride]
// float32 block, checked to the element rather than to the byte count. The kernel addresses it by
// (token * num_heads + head) * kLseStride with no bound of its own, so a tensor one rung short of the
// batch is an out-of-range global write -- which reaches the host as whatever launch the runtime was
// working on when the page fault landed, naming nothing. A shape check here is what names it.
//
// Returns the pointer to hand the launch, or nullptr when the caller passed nothing, which is what
// leaves the writer disabled inside the kernel.
inline void *CheckLse(const c10::optional<at::Tensor> &lse, const at::Tensor &reference, int64_t numTokens,
                      int64_t numHeads)
{
    if (!lse.has_value() || !lse->defined()) {
        return nullptr;
    }
    const int64_t stride = static_cast<int64_t>(tq::kLseStride);
    TORCH_CHECK(lse->scalar_type() == at::ScalarType::Float, "lse must be float32, got ", lse->scalar_type());
    TORCH_CHECK(lse->is_contiguous(), "lse must be contiguous");
    TORCH_CHECK(lse->dim() == 3 && lse->size(0) == numTokens && lse->size(1) == numHeads &&
                    lse->size(2) == stride,
                "lse must be [", numTokens, ", ", numHeads, ", ", stride, "], got ", lse->sizes());
    CheckOnDevice(*lse, "lse", reference, "the decode's query");
    CheckGmBurstAligned(*lse, "lse");
    return lse->data_ptr();
}

inline void CheckCodecTables(const at::Tensor &tables, int64_t headSize, int64_t batchRows)
{
    const int64_t words = tq::CodecTableWords(headSize, batchRows);
    TORCH_CHECK(tables.scalar_type() == at::ScalarType::Int, "codec tables must be int32");
    TORCH_CHECK(tables.is_contiguous(), "codec tables must be contiguous");
    TORCH_CHECK(tables.numel() == words, "codec tables must hold ", words, " words for head_size ", headSize,
                " and batch_rows ", batchRows, ", got ", tables.numel());
}

}

inline int64_t npu_turboquant_vector_core_num()
{
    return turboquant_adpt::VectorCoreNum();
}

inline int64_t npu_turboquant_workspace_size(int64_t num_tokens, int64_t num_heads, int64_t head_size,
                                             int64_t max_blocks_per_seq, int64_t block_size)
{
    namespace adpt = turboquant_adpt;
    TORCH_CHECK(num_tokens >= 0 && num_heads > 0, "workspace sizing needs num_tokens >= 0 and num_heads > 0, got ",
                num_tokens, " and ", num_heads);
    TORCH_CHECK(block_size > 0, "workspace sizing needs block_size > 0, got ", block_size);
    adpt::CheckHeadSize(head_size);
    return static_cast<int64_t>(vllm_ascend::turboquant::PlanPagedAttention(num_tokens, num_heads, head_size,
                                                                            max_blocks_per_seq, block_size,
                                                                            adpt::VectorCoreNum())
                                    .workspace_floats);
}

inline void npu_turboquant_reshape_and_cache(at::Tensor &key, at::Tensor &value, at::Tensor &key_cache,
                                             at::Tensor &value_cache, at::Tensor &scale_cache,
                                             at::Tensor &slot_mapping, at::Tensor &pi_signs,
                                             at::Tensor &codec_tables)
{
    namespace adpt = turboquant_adpt;

    TORCH_CHECK(key.dim() == 3 && value.dim() == 3, "key/value must be [num_tokens, num_kv_heads, head_size]");
    TORCH_CHECK(key.sizes() == value.sizes(), "key and value must have the same shape");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
                "kv cache must be [num_blocks, block_size, num_kv_heads, head_size / 2]");
    TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "key and value caches must have the same shape");
    TORCH_CHECK(key_cache.scalar_type() == at::ScalarType::Char && value_cache.scalar_type() == at::ScalarType::Char,
                "the packed 4-bit kv cache must be int8");
    TORCH_CHECK(scale_cache.scalar_type() == at::ScalarType::Float, "the kv cache scale plane must be float32");
    TORCH_CHECK(scale_cache.dim() == 3, "scale cache must be [num_blocks, block_size, scale_slot]");
    TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Int, "slot_mapping must be int32");
    TORCH_CHECK(pi_signs.scalar_type() == at::ScalarType::Float, "pi_signs must be float32");
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous(), "key/value must be contiguous");
    TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous() && scale_cache.is_contiguous(),
                "the kv cache and its scale plane must be contiguous");
    adpt::CheckGmBurstAligned(key, "key");
    adpt::CheckGmBurstAligned(value, "value");
    adpt::CheckGmBurstAligned(key_cache, "key_cache");
    adpt::CheckGmBurstAligned(value_cache, "value_cache");
    adpt::CheckGmBurstAligned(scale_cache, "scale_cache");
    // slot_mapping is read one int32 at a time (slotGm_.GetValue), never copied.
    adpt::CheckGmScalarAligned(slot_mapping, "slot_mapping");
    for (const auto &operand : {std::make_pair("value", &value), std::make_pair("key_cache", &key_cache),
                                std::make_pair("value_cache", &value_cache),
                                std::make_pair("scale_cache", &scale_cache),
                                std::make_pair("slot_mapping", &slot_mapping), std::make_pair("pi_signs", &pi_signs),
                                std::make_pair("codec_tables", &codec_tables)}) {
        adpt::CheckOnDevice(*operand.second, operand.first, key, "key");
    }

    const int64_t num_tokens = key.size(0);
    const int64_t num_kv_heads = key.size(1);
    const int64_t head_size = key.size(2);
    const int64_t num_blocks = key_cache.size(0);
    const int64_t block_size = key_cache.size(1);

    adpt::CheckHeadSize(head_size);
    adpt::CheckCodecTables(codec_tables, head_size, 1);
    TORCH_CHECK(key_cache.size(2) == num_kv_heads, "cache num_kv_heads mismatch: ", key_cache.size(2), " vs ",
                num_kv_heads);
    TORCH_CHECK(key_cache.size(3) == head_size / 2, "packed cache head dim must be head_size / 2, got ",
                key_cache.size(3));
    TORCH_CHECK(block_size % 16 == 0, "TurboQuant requires block_size to be a multiple of 16, got ", block_size);
    TORCH_CHECK(slot_mapping.numel() == num_tokens, "slot_mapping must hold one slot per token");
    TORCH_CHECK(pi_signs.numel() == head_size, "pi_signs must hold one sign per channel");
    TORCH_CHECK(scale_cache.size(0) == num_blocks && scale_cache.size(1) == block_size,
                "the scale plane must be shaped to the same blocks as the cache");
    TORCH_CHECK(scale_cache.size(2) == vllm_ascend::turboquant::ScaleSlotFloats64(num_kv_heads),
                "scale slot must be round_up(2 * num_kv_heads, 8) = ",
                vllm_ascend::turboquant::ScaleSlotFloats64(num_kv_heads), ", got ", scale_cache.size(2));

    if (num_tokens == 0) {
        return;
    }

    const vllm_ascend::turboquant::ReshapeAndCacheGrid grid =
        vllm_ascend::turboquant::PlanReshapeAndCache(num_tokens, adpt::VectorCoreNum());
    const float inv_sqrt_len = 1.0f / std::sqrt(static_cast<float>(head_size));

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    turboquant_reshape_and_cache_impl(
        adpt::ToAscendType(key.scalar_type()), stream, grid.block_dim, key.data_ptr(), value.data_ptr(),
        key_cache.data_ptr(), value_cache.data_ptr(), scale_cache.data_ptr(), slot_mapping.data_ptr(),
        pi_signs.data_ptr(), codec_tables.data_ptr(), static_cast<uint32_t>(num_tokens),
        static_cast<uint32_t>(num_kv_heads), static_cast<uint32_t>(head_size), static_cast<uint32_t>(block_size),
        static_cast<uint32_t>(num_blocks), grid.tokens_per_core, inv_sqrt_len);
}

inline void npu_turboquant_rotate_q(at::Tensor &query, at::Tensor &pi_signs, at::Tensor &codec_tables,
                                    at::Tensor &hadamard16, at::Tensor &query_rot)
{
    namespace adpt = turboquant_adpt;
    namespace tq = vllm_ascend::turboquant;

    TORCH_CHECK(query.dim() == 3, "query must be [num_tokens, num_heads, head_size]");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(query.scalar_type() == at::ScalarType::Half || query.scalar_type() == at::ScalarType::BFloat16,
                "query must be float16 or bfloat16, got ", query.scalar_type());
    TORCH_CHECK(query_rot.sizes() == query.sizes(), "query_rot must have the same shape as query");
    TORCH_CHECK(query_rot.scalar_type() == at::ScalarType::Float, "query_rot must be float32, got ",
                query_rot.scalar_type());
    TORCH_CHECK(query_rot.is_contiguous(), "query_rot must be contiguous");
    TORCH_CHECK(pi_signs.scalar_type() == at::ScalarType::Float, "pi_signs must be float32");
    TORCH_CHECK(hadamard16.scalar_type() == at::ScalarType::Half, "hadamard16 must be float16");
    TORCH_CHECK(hadamard16.is_contiguous() && hadamard16.numel() == tq::kRotateQH16Elements,
                "hadamard16 must be a contiguous ", tq::kRotateQH16Elements, "-element tensor, got ",
                hadamard16.numel());
    TORCH_CHECK(codec_tables.scalar_type() == at::ScalarType::Int, "codec tables must be int32");
    TORCH_CHECK(codec_tables.is_contiguous(), "codec tables must be contiguous");

    const int64_t num_tokens = query.size(0);
    const int64_t num_heads = query.size(1);
    const int64_t head_size = query.size(2);
    adpt::CheckHeadSize(head_size);
    TORCH_CHECK(pi_signs.numel() == head_size, "pi_signs must hold one sign per channel");
    TORCH_CHECK(codec_tables.numel() >= vllm_ascend::turboquant::CodecTableWords(head_size, 1), "codec tables must hold at least ",
                vllm_ascend::turboquant::CodecTableWords(head_size, 1), " words for head_size ", head_size, ", got ",
                codec_tables.numel());

    const int64_t num_vectors = num_tokens * num_heads;
    if (num_vectors == 0) {
        return;
    }

    const tq::RotateQPlan plan =
        tq::PlanRotateQ(num_tokens, num_vectors, head_size, tq::RotateQCoreNum(adpt::VectorCoreNum()));

    if (plan.use_cube) {
        TORCH_CHECK(plan.vectors_per_block > 0 && num_vectors % plan.vectors_per_block == 0,
                    "rotate_q plan is inconsistent: vectors_per_block ", plan.vectors_per_block,
                    " does not divide num_vectors ", num_vectors);
        TORCH_CHECK(plan.vectors_per_chunk > 0 && plan.vectors_per_block % plan.vectors_per_chunk == 0,
                    "rotate_q plan is inconsistent: vectors_per_chunk ", plan.vectors_per_chunk,
                    " does not divide vectors_per_block ", plan.vectors_per_block);
    }

    const float inv_sqrt_len = 1.0f / std::sqrt(static_cast<float>(head_size));
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    turboquant_rotate_q_impl(adpt::ToAscendType(query.scalar_type()), stream, plan.block_dim, plan.use_cube,
                             query.data_ptr(), pi_signs.data_ptr(), hadamard16.data_ptr(), codec_tables.data_ptr(),
                             query_rot.data_ptr(), static_cast<uint32_t>(num_vectors),
                             static_cast<uint32_t>(head_size), plan.vectors_per_block, plan.vectors_per_chunk,
                             plan.variant, inv_sqrt_len);
}

inline void npu_turboquant_paged_attention(at::Tensor &query_rot, at::Tensor &key_cache, at::Tensor &value_cache,
                                           at::Tensor &scale_cache, at::Tensor &block_tables,
                                           at::Tensor &context_lens, at::Tensor &codec_tables, at::Tensor &workspace,
                                           int64_t num_kv_heads, int64_t num_heads, double scale_value,
                                           at::Tensor &out, const c10::optional<at::Tensor> &lse)
{
    namespace adpt = turboquant_adpt;

    TORCH_CHECK(query_rot.dim() == 3, "query_rot must be [num_tokens, num_heads, head_size]");
    TORCH_CHECK(query_rot.scalar_type() == at::ScalarType::Float, "query_rot must be float32, got ",
                query_rot.scalar_type());
    TORCH_CHECK(out.sizes() == query_rot.sizes(), "out must have the same shape as query_rot");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::Half || out.scalar_type() == at::ScalarType::BFloat16,
                "out must be float16 or bfloat16, got ", out.scalar_type());
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
                "kv cache must be [num_blocks, block_size, num_kv_heads, head_size / 2]");
    TORCH_CHECK(key_cache.scalar_type() == at::ScalarType::Char && value_cache.scalar_type() == at::ScalarType::Char,
                "the packed 4-bit kv cache must be int8");
    TORCH_CHECK(scale_cache.scalar_type() == at::ScalarType::Float && scale_cache.dim() == 3,
                "the scale plane must be a float32 [num_blocks, block_size, scale_slot] tensor");
    TORCH_CHECK(block_tables.dim() == 2 && block_tables.scalar_type() == at::ScalarType::Int,
                "block_tables must be an int32 [num_tokens, max_blocks_per_seq] tensor");
    TORCH_CHECK(context_lens.scalar_type() == at::ScalarType::Int && context_lens.is_contiguous(),
                "context_lens must be a contiguous int32 tensor");
    TORCH_CHECK(query_rot.is_contiguous() && out.is_contiguous(), "query_rot and out must be contiguous");
    for (const auto &operand : {std::make_pair("key_cache", &key_cache), std::make_pair("value_cache", &value_cache),
                                std::make_pair("scale_cache", &scale_cache),
                                std::make_pair("block_tables", &block_tables),
                                std::make_pair("context_lens", &context_lens),
                                std::make_pair("codec_tables", &codec_tables), std::make_pair("workspace", &workspace),
                                std::make_pair("out", &out)}) {
        adpt::CheckOnDevice(*operand.second, operand.first, query_rot, "query_rot");
    }
    TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous() && scale_cache.is_contiguous(),
                "the kv cache and its scale plane must be contiguous");

    const int64_t num_tokens = query_rot.size(0);
    const int64_t head_size = query_rot.size(2);
    const int64_t block_size = key_cache.size(1);
    const int64_t max_blocks_per_seq = block_tables.size(1);

    adpt::CheckHeadSize(head_size);
    adpt::CheckCodecTables(codec_tables, head_size, vllm_ascend::turboquant::kAivTileRows);
    TORCH_CHECK(query_rot.size(1) == num_heads, "query head count ", query_rot.size(1), " does not match num_heads ",
                num_heads);
    TORCH_CHECK(num_kv_heads > 0 && num_heads % num_kv_heads == 0, "num_heads ", num_heads,
                " must be a multiple of num_kv_heads ", num_kv_heads);
    TORCH_CHECK(key_cache.size(2) == num_kv_heads, "cache num_kv_heads mismatch: ", key_cache.size(2), " vs ",
                num_kv_heads);
    TORCH_CHECK(key_cache.size(3) == head_size / 2, "packed cache head dim must be head_size / 2, got ",
                key_cache.size(3));
    TORCH_CHECK(block_size % vllm_ascend::turboquant::kAivTileRows == 0,
                "TurboQuant requires block_size to be a multiple of ", vllm_ascend::turboquant::kAivTileRows,
                ", got ", block_size);
    TORCH_CHECK(scale_cache.size(2) == vllm_ascend::turboquant::ScaleSlotFloats64(num_kv_heads),
                "scale slot must be round_up(2 * num_kv_heads, 8) = ",
                vllm_ascend::turboquant::ScaleSlotFloats64(num_kv_heads), ", got ", scale_cache.size(2));
    TORCH_CHECK(block_tables.size(0) == num_tokens, "block_tables must hold one row per query token");
    TORCH_CHECK(context_lens.numel() == num_tokens, "context_lens must hold one length per query token");

    if (num_tokens == 0) {
        return;
    }

    const vllm_ascend::turboquant::PagedAttentionGrid plan = vllm_ascend::turboquant::PlanPagedAttention(
        num_tokens, num_heads, head_size, max_blocks_per_seq, block_size, adpt::VectorCoreNum());
    const int64_t workspace_floats = static_cast<int64_t>(plan.workspace_floats);

    TORCH_CHECK(workspace.scalar_type() == at::ScalarType::Float, "the decode workspace must be float32, got ",
                workspace.scalar_type());
    TORCH_CHECK(workspace.is_contiguous(), "the decode workspace must be contiguous");
    TORCH_CHECK(workspace.numel() >= workspace_floats, "the decode workspace holds ", workspace.numel(),
                " float32 words but this launch needs ", workspace_floats, " (num_tokens ", num_tokens,
                ", num_heads ", num_heads, ", head_size ", head_size, ", num_splits ", plan.num_splits,
                "); size it with npu_turboquant_workspace_size");

    void *lse_ptr = adpt::CheckLse(lse, query_rot, num_tokens, num_heads);

    const float inv_sqrt_len = 1.0f / std::sqrt(static_cast<float>(head_size));
    const float scale = static_cast<float>(scale_value);

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    turboquant_paged_attention_impl(
        adpt::ToAscendType(out.scalar_type()), stream, plan.block_dim, query_rot.data_ptr(), key_cache.data_ptr(),
        value_cache.data_ptr(), scale_cache.data_ptr(), block_tables.data_ptr(), context_lens.data_ptr(),
        codec_tables.data_ptr(), workspace_floats > 0 ? workspace.data_ptr() : nullptr, out.data_ptr(),
        static_cast<uint32_t>(num_tokens), static_cast<uint32_t>(num_heads), static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(head_size), static_cast<uint32_t>(block_size),
        static_cast<uint32_t>(max_blocks_per_seq), static_cast<uint32_t>(plan.num_splits), plan.split_tasks_per_core,
        plan.reduce_tasks_per_core, vllm_ascend::turboquant::kFusedContextLimit, scale, inv_sqrt_len, lse_ptr);
}

#ifdef VLLM_ENABLE_TURBOQUANT_CUBE
// The kv4fp8 Cube path (950 only): an NZ-tiled cache written by npu_turboquant_cube_reshape_and_cache and read
// by npu_turboquant_cube_decode, which rotates the raw query and runs the output stage in the same launch
// (csrc/tests/TURBOQUANT_TESTS.md 13.36). The packed planes have the AIV layout's shapes and scale plane but
// not its byte order, so a cache is written and read by one path only. float16 activations only.

namespace turboquant_adpt {

constexpr int32_t kCubeMode = static_cast<int32_t>(vllm_ascend::turboquant::TurboQuantMode::KV4_FP8);

inline void CheckCubeActivation(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Half, "the TurboQuant Cube path takes float16 ", name,
                ", got ", tensor.scalar_type());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// [num_blocks, block_size, num_kv_heads, head_size / 2] int8 planes, block_size a whole number of Cube tiles, and
// the [num_blocks, block_size, scale_slot] fp32 scale plane.
inline void CheckCubeCache(const at::Tensor &key_cache, const at::Tensor &value_cache, const at::Tensor &scale_cache,
                           int64_t num_kv_heads, int64_t head_size)
{
    namespace tq = vllm_ascend::turboquant;
    TORCH_CHECK(key_cache.dim() == 4 && key_cache.sizes() == value_cache.sizes(),
                "kv cache must be two [num_blocks, block_size, num_kv_heads, head_size / 2] tensors");
    TORCH_CHECK(key_cache.scalar_type() == at::ScalarType::Char && value_cache.scalar_type() == at::ScalarType::Char,
                "the packed 4-bit kv cache must be int8");
    TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous() && scale_cache.is_contiguous(),
                "the kv cache and its scale plane must be contiguous");
    TORCH_CHECK(key_cache.size(2) == num_kv_heads, "cache num_kv_heads mismatch: ", key_cache.size(2), " vs ",
                num_kv_heads);
    TORCH_CHECK(key_cache.size(3) == head_size / 2, "packed cache head dim must be head_size / 2, got ",
                key_cache.size(3));
    TORCH_CHECK(key_cache.size(1) % tq::kCubeTileRows == 0, "the TurboQuant Cube path requires block_size to be a ",
                "multiple of ", tq::kCubeTileRows, ", got ", key_cache.size(1));
    TORCH_CHECK(scale_cache.scalar_type() == at::ScalarType::Float && scale_cache.dim() == 3,
                "the scale plane must be a float32 [num_blocks, block_size, scale_slot] tensor");
    TORCH_CHECK(scale_cache.size(0) == key_cache.size(0) && scale_cache.size(1) == key_cache.size(1),
                "the scale plane must be shaped to the same blocks as the cache");
    TORCH_CHECK(scale_cache.size(2) == tq::ScaleSlotFloats64(num_kv_heads), "scale slot must be ",
                "round_up(2 * num_kv_heads, 8) = ", tq::ScaleSlotFloats64(num_kv_heads), ", got ", scale_cache.size(2));
}

inline void CheckRotationTables(const at::Tensor &pi_signs, const at::Tensor &codec_tables, int64_t head_size)
{
    TORCH_CHECK(pi_signs.scalar_type() == at::ScalarType::Float && pi_signs.numel() == head_size,
                "pi_signs must be float32 with one sign per channel");
    TORCH_CHECK(pi_signs.is_contiguous(), "pi_signs must be contiguous");
    CheckCodecTables(codec_tables, head_size, 1);
}

}

inline int64_t npu_turboquant_cube_workspace_size(int64_t num_tokens, int64_t num_heads, int64_t num_kv_heads,
                                                  int64_t head_size, int64_t max_blocks_per_seq, int64_t block_size)
{
    namespace adpt = turboquant_adpt;
    TORCH_CHECK(num_tokens >= 0 && num_heads > 0 && num_kv_heads > 0 && num_heads % num_kv_heads == 0,
                "workspace sizing needs num_tokens >= 0 and num_heads a positive multiple of num_kv_heads, got ",
                num_tokens, ", ", num_heads, " and ", num_kv_heads);
    TORCH_CHECK(block_size > 0, "workspace sizing needs block_size > 0, got ", block_size);
    adpt::CheckHeadSize(head_size);
    if (num_tokens == 0) {
        return 0;
    }
    return static_cast<int64_t>(vllm_ascend::turboquant::PlanFusedDecode(num_tokens, num_heads, num_kv_heads,
                                                                         head_size, max_blocks_per_seq, block_size,
                                                                         adpt::VectorCoreNum())
                                    .workspace_floats);
}

inline void npu_turboquant_cube_reshape_and_cache(at::Tensor &key, at::Tensor &value, at::Tensor &key_cache,
                                                  at::Tensor &value_cache, at::Tensor &scale_cache,
                                                  at::Tensor &slot_mapping, at::Tensor &pi_signs,
                                                  at::Tensor &codec_tables)
{
    namespace adpt = turboquant_adpt;
    namespace tq = vllm_ascend::turboquant;

    TORCH_CHECK(key.dim() == 3 && key.sizes() == value.sizes(),
                "key/value must be two [num_tokens, num_kv_heads, head_size] tensors");
    adpt::CheckCubeActivation(key, "key");
    adpt::CheckCubeActivation(value, "value");
    TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Int && slot_mapping.is_contiguous(),
                "slot_mapping must be a contiguous int32 tensor");

    const int64_t num_tokens = key.size(0);
    const int64_t num_kv_heads = key.size(1);
    const int64_t head_size = key.size(2);
    adpt::CheckHeadSize(head_size);
    adpt::CheckCubeCache(key_cache, value_cache, scale_cache, num_kv_heads, head_size);
    adpt::CheckRotationTables(pi_signs, codec_tables, head_size);
    TORCH_CHECK(slot_mapping.numel() == num_tokens, "slot_mapping must hold one slot per token");
    adpt::CheckGmBurstAligned(key, "key");
    adpt::CheckGmBurstAligned(value, "value");
    adpt::CheckGmBurstAligned(key_cache, "key_cache");
    adpt::CheckGmBurstAligned(value_cache, "value_cache");
    adpt::CheckGmBurstAligned(scale_cache, "scale_cache");
    // slot_mapping is read one int32 at a time (slotGm_.GetValue), never copied.
    adpt::CheckGmScalarAligned(slot_mapping, "slot_mapping");
    for (const auto &operand : {std::make_pair("value", &value), std::make_pair("key_cache", &key_cache),
                                std::make_pair("value_cache", &value_cache),
                                std::make_pair("scale_cache", &scale_cache),
                                std::make_pair("slot_mapping", &slot_mapping), std::make_pair("pi_signs", &pi_signs),
                                std::make_pair("codec_tables", &codec_tables)}) {
        adpt::CheckOnDevice(*operand.second, operand.first, key, "key");
    }

    if (num_tokens == 0) {
        return;
    }

    const tq::ReshapeAndCacheGrid grid = tq::PlanReshapeAndCache(num_tokens, adpt::VectorCoreNum());
    const float inv_sqrt_len = 1.0f / std::sqrt(static_cast<float>(head_size));
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    // kv4fp8's affine codec reads no mode table; the rotation tables stand in for the unused address.
    turboquant_mm_reshape_and_cache_impl(
        adpt::kCubeMode, AscendType::FP16, stream, grid.block_dim, key.data_ptr(), value.data_ptr(),
        key_cache.data_ptr(), value_cache.data_ptr(), scale_cache.data_ptr(), slot_mapping.data_ptr(),
        pi_signs.data_ptr(), codec_tables.data_ptr(), codec_tables.data_ptr(), static_cast<uint32_t>(num_tokens),
        static_cast<uint32_t>(num_kv_heads), static_cast<uint32_t>(head_size),
        static_cast<uint32_t>(key_cache.size(1)), static_cast<uint32_t>(key_cache.size(0)), grid.tokens_per_core,
        inv_sqrt_len);
}

// One decode launch over the raw query. output_stage is a TurboQuantOutputStage: 0 leaves the output rotated
// (a Pi-folded o_proj), 1 un-rotates it, 2 also multiplies it by sigmoid(gate). query_rot is the launch's fp32
// scratch for the rotated query; workspace is sized by npu_turboquant_cube_workspace_size.
inline void npu_turboquant_cube_decode(at::Tensor &query, const c10::optional<at::Tensor> &gate,
                                       at::Tensor &pi_signs, at::Tensor &codec_tables, at::Tensor &hadamard16,
                                       at::Tensor &key_cache, at::Tensor &value_cache, at::Tensor &scale_cache,
                                       at::Tensor &block_tables, at::Tensor &context_lens, at::Tensor &workspace,
                                       at::Tensor &query_rot, int64_t num_kv_heads, int64_t num_heads,
                                       double scale_value, int64_t output_stage, at::Tensor &out,
                                       const c10::optional<at::Tensor> &lse)
{
    namespace adpt = turboquant_adpt;
    namespace tq = vllm_ascend::turboquant;

    TORCH_CHECK(query.dim() == 3, "query must be [num_tokens, num_heads, head_size]");
    adpt::CheckCubeActivation(query, "query");
    adpt::CheckCubeActivation(out, "out");
    TORCH_CHECK(out.sizes() == query.sizes(), "out must have the same shape as query");
    TORCH_CHECK(query_rot.scalar_type() == at::ScalarType::Float && query_rot.is_contiguous(),
                "query_rot must be a contiguous float32 tensor");
    TORCH_CHECK(query_rot.numel() == query.numel(), "query_rot must hold one fp32 word per query element");
    TORCH_CHECK(output_stage >= static_cast<int64_t>(tq::TurboQuantOutputStage::kRotatedBasis) &&
                    output_stage <= static_cast<int64_t>(tq::TurboQuantOutputStage::kGated),
                "output_stage must be 0 (rotated), 1 (unrotated) or 2 (gated), got ", output_stage);
    const bool gated = output_stage == static_cast<int64_t>(tq::TurboQuantOutputStage::kGated);
    TORCH_CHECK(!gated || gate.has_value(), "a gated output stage needs the gate tensor");
    if (gated) {
        adpt::CheckCubeActivation(*gate, "gate");
        TORCH_CHECK(gate->numel() == query.numel(), "gate must hold one logit per query element");
    }
    TORCH_CHECK(hadamard16.scalar_type() == at::ScalarType::Half && hadamard16.is_contiguous() &&
                    hadamard16.numel() == tq::kRotateQH16Elements,
                "hadamard16 must be a contiguous ", tq::kRotateQH16Elements, "-element float16 tensor");
    TORCH_CHECK(block_tables.dim() == 2 && block_tables.scalar_type() == at::ScalarType::Int &&
                    block_tables.is_contiguous(),
                "block_tables must be a contiguous int32 [num_tokens, max_blocks_per_seq] tensor");
    TORCH_CHECK(context_lens.scalar_type() == at::ScalarType::Int && context_lens.is_contiguous(),
                "context_lens must be a contiguous int32 tensor");
    for (const auto &operand : {std::make_pair("key_cache", &key_cache), std::make_pair("value_cache", &value_cache),
                                std::make_pair("scale_cache", &scale_cache),
                                std::make_pair("block_tables", &block_tables),
                                std::make_pair("context_lens", &context_lens), std::make_pair("pi_signs", &pi_signs),
                                std::make_pair("codec_tables", &codec_tables),
                                std::make_pair("hadamard16", &hadamard16), std::make_pair("query_rot", &query_rot),
                                std::make_pair("workspace", &workspace), std::make_pair("out", &out)}) {
        adpt::CheckOnDevice(*operand.second, operand.first, query, "query");
    }
    if (gated) {
        adpt::CheckOnDevice(*gate, "gate", query, "query");
    }

    const int64_t num_tokens = query.size(0);
    const int64_t head_size = query.size(2);
    adpt::CheckHeadSize(head_size);
    TORCH_CHECK(query.size(1) == num_heads, "query head count ", query.size(1), " does not match num_heads ",
                num_heads);
    TORCH_CHECK(num_kv_heads > 0 && num_heads % num_kv_heads == 0, "num_heads ", num_heads,
                " must be a multiple of num_kv_heads ", num_kv_heads);
    adpt::CheckCubeCache(key_cache, value_cache, scale_cache, num_kv_heads, head_size);
    adpt::CheckRotationTables(pi_signs, codec_tables, head_size);
    TORCH_CHECK(block_tables.size(0) == num_tokens, "block_tables must hold one row per query token");
    TORCH_CHECK(context_lens.numel() == num_tokens, "context_lens must hold one length per query token");

    if (num_tokens == 0) {
        return;
    }

    const int64_t block_size = key_cache.size(1);
    const int64_t max_blocks_per_seq = block_tables.size(1);
    const tq::FusedDecodeGrid grid = tq::PlanFusedDecode(num_tokens, num_heads, num_kv_heads, head_size,
                                                         max_blocks_per_seq, block_size, adpt::VectorCoreNum());
    const int64_t workspace_floats = static_cast<int64_t>(grid.workspace_floats);
    TORCH_CHECK(workspace.scalar_type() == at::ScalarType::Float && workspace.is_contiguous(),
                "the decode workspace must be a contiguous float32 tensor");
    TORCH_CHECK(workspace.numel() >= workspace_floats, "the decode workspace holds ", workspace.numel(),
                " float32 words but this launch needs ", workspace_floats, " (num_tokens ", num_tokens,
                ", num_splits ", grid.num_splits, "); size it with npu_turboquant_cube_workspace_size");

    void *lse_ptr = adpt::CheckLse(lse, query, num_tokens, num_heads);

    const float inv_sqrt_len = 1.0f / std::sqrt(static_cast<float>(head_size));
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    turboquant_mm_fused_decode_raw_query_impl(
        adpt::kCubeMode, AscendType::FP16, stream, grid.block_dim, query.data_ptr(), pi_signs.data_ptr(),
        codec_tables.data_ptr(), hadamard16.data_ptr(), gated ? gate->data_ptr() : nullptr, query_rot.data_ptr(),
        key_cache.data_ptr(), value_cache.data_ptr(), scale_cache.data_ptr(), block_tables.data_ptr(),
        context_lens.data_ptr(), codec_tables.data_ptr(), workspace_floats > 0 ? workspace.data_ptr() : nullptr,
        out.data_ptr(), static_cast<uint32_t>(num_tokens), static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(num_kv_heads), static_cast<uint32_t>(head_size), static_cast<uint32_t>(block_size),
        static_cast<uint32_t>(max_blocks_per_seq), static_cast<uint32_t>(grid.num_splits), grid.heads_per_task,
        grid.tasks_per_block, grid.reduce_tasks_per_block, grid.prologue_vectors_per_block,
        grid.prologue_cube_chunk_vectors, static_cast<uint32_t>(output_stage), grid.fused_context_limit,
        static_cast<float>(scale_value), inv_sqrt_len, lse_ptr);
}
#endif

}
