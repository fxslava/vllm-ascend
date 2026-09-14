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

#include "../../kernels/types.h"
#include "../../npu_device_registry.h"
#include "turboquant_rotate_q.h"

namespace vllm_ascend {

extern void turboquant_reshape_and_cache_impl(AscendType type, void *stream, uint32_t blockDim, void *key,
                                              void *value, void *keyCache, void *valueCache, void *scaleCache,
                                              void *slotMapping, void *piSigns, void *tables, uint32_t numTokens,
                                              uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,
                                              uint32_t tokensPerCore, float invSqrtLen);

extern void turboquant_paged_attention_impl(AscendType type, void *stream, uint32_t splitBlockDim,
                                            uint32_t combineBlockDim, void *queryRot, void *keyCache,
                                            void *valueCache, void *scaleCache, void *blockTables,
                                            void *contextLens, void *piSigns,
                                            void *tables, void *workspace, void *output, uint32_t numTokens,
                                            uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize,
                                            uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,
                                            uint32_t splitTasksPerCore, uint32_t combineTasksPerCore, float scale,
                                            float invSqrtLen);

extern void turboquant_rotate_q_impl(AscendType type, void *stream, uint32_t blockDim, bool useCube, void *query,
                                     void *piSigns, void *h16, void *rotTables, void *queryRot, uint32_t numVectors,
                                     uint32_t headSize, uint32_t vectorsPerBlock, uint32_t vectorsPerChunk,
                                     uint32_t variant, float invSqrtLen);

namespace turboquant_adpt {

// Bounds the flash-decoding workspace; beyond this the reduce stage costs more
// than the extra parallelism buys.
constexpr int64_t kMaxSequenceSplits = 8;
// fp32 words appended to every partial accumulator: one 32B block holding the
// running max and a second holding the running sum. They are separate blocks
// because a vector operand's base must be 32-byte aligned, so the two values
// cannot share one; see kPartialTail in turboquant_kernels.cpp.
constexpr int64_t kPartialTail = 16;
// fp32 lanes in one 32B burst.
constexpr int64_t kFp32PerBlock = 8;
// Rows of a paged block the decode kernel processes per tile; must match
// kTileRows in turboquant_kernels.cpp, because the codec's shuffle tables are
// built for that batch size.
constexpr int64_t kTileRows = 16;
// Reconstruction levels of the codec, and so the length of the Lloyd-Max
// centroid table the constant-table image carries; must match
// TurboQuantCodec<4>::kLevels.
constexpr int64_t kCodecLevels = 16;

inline AscendType ToAscendType(at::ScalarType scalarType)
{
    TORCH_CHECK(scalarType == at::ScalarType::Half || scalarType == at::ScalarType::BFloat16,
                "TurboQuant KV cache supports float16 and bfloat16 activations, got ", scalarType);
    return scalarType == at::ScalarType::BFloat16 ? AscendType::BF16 : AscendType::FP16;
}

// AI Vector cores on the device this rank is bound to, cached per device; see
// csrc/npu_device_registry.h.  Resolving the device per call rather than
// assuming 0 is what makes this correct under tensor parallelism.
inline int64_t VectorCoreNum()
{
    return device_registry::VectorCoreNum();
}

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

// fp32 words one token occupies in the scale plane: K then V for every kv head,
// padded to a whole 32-byte burst so the scatter never touches an unaligned
// global address.  Mirrored by ScaleSlotFloats() in turboquant_kernels.cpp and
// by turboquant_scale_slot() on the Python side.
inline int64_t ScaleSlotFloats(int64_t numKvHeads)
{
    return CeilDiv(2 * numKvHeads, kFp32PerBlock) * kFp32PerBlock;
}

// Words in the codec's constant-table image; must equal
// TurboQuantCodec<4>::ConstTableWords(headSize, batchRows).  The trailing
// kCodecLevels words are the Lloyd-Max centroid table Dequantize4Bit gathers
// against.
inline int64_t CodecTableWords(int64_t headSize, int64_t batchRows)
{
    return 7 * headSize + 2 * headSize * batchRows + kCodecLevels;
}

inline void CheckHeadSize(int64_t headSize)
{
    TORCH_CHECK(headSize >= 64 && headSize <= 256, "TurboQuant requires 64 <= head_size <= 256, got ", headSize);
    TORCH_CHECK((headSize & (headSize - 1)) == 0, "TurboQuant needs a power-of-two head_size for the Walsh-Hadamard "
                                                  "transform, got ",
                headSize);
}

inline void CheckCodecTables(const at::Tensor &tables, int64_t headSize, int64_t batchRows)
{
    TORCH_CHECK(tables.scalar_type() == at::ScalarType::Int, "codec tables must be int32");
    TORCH_CHECK(tables.is_contiguous(), "codec tables must be contiguous");
    TORCH_CHECK(tables.numel() == CodecTableWords(headSize, batchRows), "codec tables must hold ",
                CodecTableWords(headSize, batchRows), " words for head_size ", headSize, " and batch_rows ",
                batchRows, ", got ", tables.numel());
}

// fp32 words one flash-decoding partial occupies: the accumulator itself
// followed by the running max and running sum blocks.
inline int64_t PartialStride(int64_t headSize)
{
    return headSize + kPartialTail;
}

// Everything about a paged-attention launch that is a pure function of the
// shapes and the core count: the split count, the workspace it needs, and the
// two grids.  Mirrored for the torch-free suite by PlanPagedAttention() in
// csrc/tests/common/turboquant_launch.cpp.
struct PagedAttentionPlan {
    int64_t num_splits = 1;
    // fp32 words the split stage writes and the combine stage reads:
    // base_tasks * num_splits * PartialStride(head_size).
    int64_t workspace_floats = 0;
    uint32_t split_block_dim = 0;
    uint32_t combine_block_dim = 0;
    uint32_t split_tasks_per_core = 0;
    uint32_t combine_tasks_per_core = 0;
};

inline PagedAttentionPlan PlanPagedAttention(int64_t numTokens, int64_t numHeads, int64_t headSize,
                                             int64_t maxBlocksPerSeq, int64_t aivNum)
{
    PagedAttentionPlan plan;

    const int64_t base_tasks = numTokens * numHeads;
    if (base_tasks <= 0) {
        return plan;
    }

    // Split the sequence only while there are idle cores to hand the pieces to,
    // and never past the number of blocks there are to split.
    int64_t num_splits = CeilDiv(aivNum, base_tasks);
    num_splits = std::min(num_splits, std::min<int64_t>(kMaxSequenceSplits, std::max<int64_t>(1, maxBlocksPerSeq)));
    num_splits = std::max<int64_t>(num_splits, 1);

    // Each stage gets its own grid: the split stage has num_splits times as many
    // tasks as the combine stage, and sizing them separately keeps the combine
    // launch from spawning cores with nothing to do.
    const int64_t split_tasks = base_tasks * num_splits;
    const int64_t split_tasks_per_core = CeilDiv(split_tasks, aivNum);
    const int64_t combine_tasks_per_core = CeilDiv(base_tasks, aivNum);

    plan.num_splits = num_splits;
    plan.workspace_floats = split_tasks * PartialStride(headSize);
    plan.split_block_dim = static_cast<uint32_t>(CeilDiv(split_tasks, split_tasks_per_core));
    plan.combine_block_dim = static_cast<uint32_t>(CeilDiv(base_tasks, combine_tasks_per_core));
    plan.split_tasks_per_core = static_cast<uint32_t>(split_tasks_per_core);
    plan.combine_tasks_per_core = static_cast<uint32_t>(combine_tasks_per_core);
    return plan;
}

}  // namespace turboquant_adpt

// AI Vector cores on this rank's device, from the cached registry.  Exposed so
// the host can prime the registry while the model is still loading and keep the
// first driver query off the decode path.
inline int64_t npu_turboquant_vector_core_num()
{
    return turboquant_adpt::VectorCoreNum();
}

// fp32 words of workspace npu_turboquant_paged_attention needs for a decode of
// this shape.  The caller owns the buffer; this exists so the host can allocate
// it once from the same arithmetic the operator uses.
inline int64_t npu_turboquant_workspace_size(int64_t num_tokens, int64_t num_heads, int64_t head_size,
                                             int64_t max_blocks_per_seq)
{
    namespace adpt = turboquant_adpt;
    TORCH_CHECK(num_tokens >= 0 && num_heads > 0, "workspace sizing needs num_tokens >= 0 and num_heads > 0, got ",
                num_tokens, " and ", num_heads);
    adpt::CheckHeadSize(head_size);
    return adpt::PlanPagedAttention(num_tokens, num_heads, head_size, max_blocks_per_seq, adpt::VectorCoreNum())
        .workspace_floats;
}

/*
 * Rotate, quantise to 4 bits and scatter K/V into the ND paged cache.
 *
 *   key, value      [num_tokens, num_kv_heads, head_size]                 fp16 | bf16
 *   key/value cache [num_blocks, block_size, num_kv_heads, head_size / 2] int8
 *   scale cache     [num_blocks, block_size, scale_slot]                  fp32
 *   slot_mapping    [num_tokens]                                          int32, -1 skips
 *   pi_signs        [head_size]                                           fp32, +-1
 *   codec_tables    [7 * head_size + 2 * head_size]                       int32
 *
 * `key` is the post-RoPE key and `value` the raw value projection.  The kernel
 * applies Pi in UB and rewrites no weight.  `codec_tables` is the host-built
 * shuffle/sign image the kernel pulls in with a single DataCopy.
 */
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
    TORCH_CHECK(scale_cache.size(2) == adpt::ScaleSlotFloats(num_kv_heads),
                "scale slot must be round_up(2 * num_kv_heads, 8) = ", adpt::ScaleSlotFloats(num_kv_heads), ", got ",
                scale_cache.size(2));

    if (num_tokens == 0) {
        return;
    }

    const int64_t aiv_num = adpt::VectorCoreNum();
    const int64_t tokens_per_core = adpt::CeilDiv(num_tokens, aiv_num);
    const uint32_t block_dim = static_cast<uint32_t>(adpt::CeilDiv(num_tokens, tokens_per_core));
    const float inv_sqrt_len = 1.0f / std::sqrt(static_cast<float>(head_size));

    // Pushed straight onto the current stream rather than through
    // OpCommand::SetCustomHandler, which would add host work per launch.
    // getCurrentNPUStream().stream() drains the pending queue before returning
    // the raw handle, so this still lands behind whatever the thread queued
    // earlier.  The launch itself is asynchronous.
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    turboquant_reshape_and_cache_impl(
        adpt::ToAscendType(key.scalar_type()), stream, block_dim, key.data_ptr(), value.data_ptr(),
        key_cache.data_ptr(), value_cache.data_ptr(), scale_cache.data_ptr(), slot_mapping.data_ptr(),
        pi_signs.data_ptr(), codec_tables.data_ptr(), static_cast<uint32_t>(num_tokens),
        static_cast<uint32_t>(num_kv_heads), static_cast<uint32_t>(head_size), static_cast<uint32_t>(block_size),
        static_cast<uint32_t>(tokens_per_core), inv_sqrt_len);
}

/*
 * Rotate every query vector of a decode step: q~ = Pi q = D (H (D q)).
 *
 *   query        [num_tokens, num_heads, head_size]   fp16 | bf16, post-RoPE
 *   pi_signs     [head_size]                          fp32, +-1
 *   codec_tables CodecTableWords(head_size, 1) or more int32
 *   hadamard16   [16, 16]                             fp16, FillHadamard16Half
 *   query_rot    [num_tokens, num_heads, head_size]   fp32, the output
 *
 * The output is fp32 rather than the query's own type because both decode
 * splits consume it in fp32: the Cube split scales it to its operand grid and
 * casts, and the vector split tiles it straight into its fp32 dot product.
 * Handing them anything narrower would put a cast back into the decode, which is
 * the loop this change exists to empty.
 *
 * `codec_tables` is checked for at least the batch-rows-1 prefix rather than an
 * exact length: the vector path's codec reads only the sign and shuffle
 * sections, which sit at batch-rows-independent offsets, so either of the two
 * table tensors the caller already owns will do.
 */
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
    TORCH_CHECK(codec_tables.numel() >= adpt::CodecTableWords(head_size, 1), "codec tables must hold at least ",
                adpt::CodecTableWords(head_size, 1), " words for head_size ", head_size, ", got ",
                codec_tables.numel());

    const int64_t num_vectors = num_tokens * num_heads;
    if (num_vectors == 0) {
        return;
    }

    // MIX blocks, not vector cores: arch35 pairs one cube core with two vector
    // subcores, and blockDim on a MIX launch counts the pairs.
    int64_t core_num = adpt::VectorCoreNum() / 2;
    if (core_num < 1) {
        core_num = 1;
    }
    // Only a half query is exactly representable in half after the sign
    // multiply, so only a half query may take the single-Mmad Cube path.
    const bool input_exact_in_half = query.scalar_type() == at::ScalarType::Half;
    const tq::RotateQPlan plan = tq::PlanRotateQ(num_vectors, head_size, core_num, input_exact_in_half);

    // The two divisibility invariants the Cube kernel relies on and does not
    // check itself. A violation would silently drop the remainder of a block's
    // share rather than fault, which is exactly the class of defect that reads
    // as a numerical bug.
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

/*
 * Paged decode attention over the 4-bit cache.
 *
 *   query_rot    [num_tokens, num_heads, head_size]         fp32
 *   block_tables [num_tokens, max_blocks_per_seq]           int32
 *   context_lens [num_tokens]                               int32
 *   workspace    [>= npu_turboquant_workspace_size(...)]    fp32
 *   out          [num_tokens, num_heads, head_size]         fp16 | bf16
 *
 * `query_rot` is the fp32 output of npu_turboquant_rotate_q, NOT the model's
 * query -- call that operator first.  The split kernel applies no rotation of
 * any kind; it runs the softmax and the value accumulation in the rotated basis
 * and the combine applies Pi again on the way out, which is why `pi_signs` is
 * still an argument.  A caller that passed the unrotated query here would get a
 * plausible-looking wrong answer rather than an error.
 *
 * The element type the kernels are instantiated for comes from `out`: it is the
 * only tensor in the call that still carries the model's dtype.
 *
 * This dispatches *two* kernels on the current stream: an in-kernel barrier
 * only orders co-resident blocks, so stream order between two launches is the
 * guarantee that always holds.
 *
 * `workspace` is caller-owned scratch, not an output.  It is an argument rather
 * than an at::empty because a decode step must not allocate.
 */
inline void npu_turboquant_paged_attention(at::Tensor &query_rot, at::Tensor &key_cache, at::Tensor &value_cache,
                                           at::Tensor &scale_cache, at::Tensor &block_tables,
                                           at::Tensor &context_lens, at::Tensor &pi_signs, at::Tensor &codec_tables,
                                           at::Tensor &workspace, int64_t num_kv_heads, int64_t num_heads,
                                           double scale_value, at::Tensor &out)
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
    TORCH_CHECK(context_lens.scalar_type() == at::ScalarType::Int, "context_lens must be int32");
    TORCH_CHECK(pi_signs.scalar_type() == at::ScalarType::Float, "pi_signs must be float32");
    TORCH_CHECK(query_rot.is_contiguous() && out.is_contiguous(), "query_rot and out must be contiguous");
    TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous() && scale_cache.is_contiguous(),
                "the kv cache and its scale plane must be contiguous");

    const int64_t num_tokens = query_rot.size(0);
    const int64_t head_size = query_rot.size(2);
    const int64_t block_size = key_cache.size(1);
    const int64_t max_blocks_per_seq = block_tables.size(1);

    adpt::CheckHeadSize(head_size);
    adpt::CheckCodecTables(codec_tables, head_size, adpt::kTileRows);
    TORCH_CHECK(query_rot.size(1) == num_heads, "query head count ", query_rot.size(1), " does not match num_heads ",
                num_heads);
    TORCH_CHECK(num_kv_heads > 0 && num_heads % num_kv_heads == 0, "num_heads ", num_heads,
                " must be a multiple of num_kv_heads ", num_kv_heads);
    TORCH_CHECK(key_cache.size(2) == num_kv_heads, "cache num_kv_heads mismatch: ", key_cache.size(2), " vs ",
                num_kv_heads);
    TORCH_CHECK(key_cache.size(3) == head_size / 2, "packed cache head dim must be head_size / 2, got ",
                key_cache.size(3));
    TORCH_CHECK(block_size % adpt::kTileRows == 0, "TurboQuant requires block_size to be a multiple of ",
                adpt::kTileRows, ", got ", block_size);
    TORCH_CHECK(scale_cache.size(2) == adpt::ScaleSlotFloats(num_kv_heads),
                "scale slot must be round_up(2 * num_kv_heads, 8) = ", adpt::ScaleSlotFloats(num_kv_heads), ", got ",
                scale_cache.size(2));
    TORCH_CHECK(block_tables.size(0) == num_tokens, "block_tables must hold one row per query token");
    TORCH_CHECK(context_lens.numel() == num_tokens, "context_lens must hold one length per query token");
    TORCH_CHECK(pi_signs.numel() == head_size, "pi_signs must hold one sign per channel");

    if (num_tokens == 0) {
        return;
    }

    const adpt::PagedAttentionPlan plan =
        adpt::PlanPagedAttention(num_tokens, num_heads, head_size, max_blocks_per_seq, adpt::VectorCoreNum());

    TORCH_CHECK(workspace.scalar_type() == at::ScalarType::Float, "the decode workspace must be float32, got ",
                workspace.scalar_type());
    TORCH_CHECK(workspace.is_contiguous(), "the decode workspace must be contiguous");
    // Larger than needed is fine and expected -- the host sizes one buffer for
    // the widest decode it will run and reuses it for narrower ones.  Too small
    // is not survivable, so it is caught here rather than as a stray
    // out-of-bounds write from the split stage.
    TORCH_CHECK(workspace.numel() >= plan.workspace_floats, "the decode workspace holds ", workspace.numel(),
                " float32 words but this launch needs ", plan.workspace_floats, " (num_tokens ", num_tokens,
                ", num_heads ", num_heads, ", head_size ", head_size, ", num_splits ", plan.num_splits,
                "); size it with npu_turboquant_workspace_size");

    const float inv_sqrt_len = 1.0f / std::sqrt(static_cast<float>(head_size));
    const float scale = static_cast<float>(scale_value);

    // See the reshape-and-cache launch above for why this goes straight onto the
    // stream. The split-before-combine ordering the two stages depend on is the
    // stream's, and two launches into the same stream keep it.
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    turboquant_paged_attention_impl(
        adpt::ToAscendType(out.scalar_type()), stream, plan.split_block_dim, plan.combine_block_dim,
        query_rot.data_ptr(), key_cache.data_ptr(), value_cache.data_ptr(), scale_cache.data_ptr(),
        block_tables.data_ptr(), context_lens.data_ptr(), pi_signs.data_ptr(), codec_tables.data_ptr(),
        workspace.data_ptr(), out.data_ptr(), static_cast<uint32_t>(num_tokens), static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(num_kv_heads), static_cast<uint32_t>(head_size), static_cast<uint32_t>(block_size),
        static_cast<uint32_t>(max_blocks_per_seq), static_cast<uint32_t>(plan.num_splits), plan.split_tasks_per_core,
        plan.combine_tasks_per_core, scale, inv_sqrt_len);
}

}  // namespace vllm_ascend
