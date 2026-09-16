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

#include "turboquant_tiling.h"

#include <algorithm>

namespace vllm_ascend {
namespace turboquant {

int64_t CeilDiv64(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

int64_t ScaleSlotFloats64(int64_t num_kv_heads)
{
    const int64_t lanes = kFp32PerBlock;
    return CeilDiv64(2 * num_kv_heads, lanes) * lanes;
}

size_t PackedCacheBytes(int64_t num_blocks, int64_t block_size, int64_t num_kv_heads, int64_t head_size)
{
    return static_cast<size_t>(num_blocks) * static_cast<size_t>(block_size) * static_cast<size_t>(num_kv_heads) *
           static_cast<size_t>(head_size / kPackFactor);
}

size_t ScalePlaneFloats(int64_t num_blocks, int64_t block_size, int64_t num_kv_heads)
{
    return static_cast<size_t>(num_blocks) * static_cast<size_t>(block_size) *
           static_cast<size_t>(ScaleSlotFloats64(num_kv_heads));
}

int64_t CodecTableWords(int64_t head_size, int64_t batch_rows)
{
    return 7 * head_size + 2 * head_size * batch_rows + kCodecLevels;
}

int64_t RotateQCoreNum(int64_t aiv_num)
{
    const int64_t cores = aiv_num / kVectorSubcoresPerBlock;
    return cores < 1 ? 1 : cores;
}

ReshapeAndCacheGrid PlanReshapeAndCache(int64_t num_tokens, int64_t aiv_num)
{
    ReshapeAndCacheGrid grid;
    if (num_tokens <= 0 || aiv_num <= 0) {
        return grid;
    }
    const int64_t tokens_per_core = CeilDiv64(num_tokens, aiv_num);
    grid.tokens_per_core = static_cast<uint32_t>(tokens_per_core);
    grid.block_dim = static_cast<uint32_t>(CeilDiv64(num_tokens, tokens_per_core));
    return grid;
}

PagedAttentionGrid PlanPagedAttention(int64_t num_tokens, int64_t num_heads, int64_t head_size,
                                      int64_t max_blocks_per_seq, int64_t block_size, int64_t aiv_num,
                                      int64_t fused_context_limit)
{
    PagedAttentionGrid grid;
    const int64_t base_tasks = num_tokens * num_heads;
    if (num_tokens <= 0 || num_heads <= 0 || aiv_num <= 0) {
        return grid;
    }

    const int64_t blocks = std::max<int64_t>(1, max_blocks_per_seq);
    const int64_t context_bound = blocks * block_size;
    const int64_t max_splits = kMaxSequenceSplits;

    int64_t num_splits = 1;
    if (context_bound > fused_context_limit) {
        const int64_t by_context =
            fused_context_limit > 0 ? CeilDiv64(context_bound, fused_context_limit) : max_splits;
        num_splits = std::max(CeilDiv64(aiv_num, base_tasks), by_context);
        num_splits = std::min(num_splits, std::min(max_splits, blocks));
    }
    grid.num_splits = num_splits;

    const int64_t split_tasks = base_tasks * num_splits;
    const int64_t split_tasks_per_core = CeilDiv64(split_tasks, aiv_num);
    const int64_t block_dim = CeilDiv64(split_tasks, split_tasks_per_core);

    grid.block_dim = static_cast<uint32_t>(block_dim);
    grid.split_tasks_per_core = static_cast<uint32_t>(split_tasks_per_core);
    if (num_splits > 1) {
        grid.reduce_tasks_per_core = static_cast<uint32_t>(CeilDiv64(base_tasks, block_dim));
        grid.workspace_floats = static_cast<size_t>(split_tasks * (head_size + kPartialTail));
    }
    return grid;
}

FusedDecodeGrid PlanFusedDecode(int64_t num_tokens, int64_t num_heads, int64_t num_kv_heads, int64_t head_size,
                                int64_t max_blocks_per_seq, int64_t block_size, int64_t aiv_num,
                                int64_t fused_context_limit, FusedSplitPolicy split_policy)
{
    FusedDecodeGrid grid;
    if (num_tokens <= 0 || num_heads <= 0 || num_kv_heads <= 0) {
        return grid;
    }

    const int64_t subcores = kVectorSubcoresPerBlock;
    const int64_t tile_m = kCubeTileM;
    const int64_t max_splits = kMaxSequenceSplits;
    const int64_t mix_blocks = std::max<int64_t>(1, aiv_num / subcores);
    const int64_t group = num_heads / num_kv_heads;
    const int64_t blocks = std::max<int64_t>(1, max_blocks_per_seq);
    const int64_t context_bound = blocks * block_size;

    // Head chunks per GQA group once `groups` (token, kv head, split) triples share the blocks.
    const auto chunks_for = [&](int64_t groups) -> int64_t {
        if (group <= 1) {
            return 1;
        }
        const int64_t fewest_chunks = CeilDiv64(group, tile_m);
        const int64_t most_chunks = CeilDiv64(group, subcores);
        return std::min(std::max(CeilDiv64(mix_blocks, groups), fewest_chunks), most_chunks);
    };

    int64_t num_splits = 1;
    int64_t launch_limit = fused_context_limit;
    if (context_bound > fused_context_limit) {
        const int64_t by_context =
            fused_context_limit > 0 ? CeilDiv64(context_bound, fused_context_limit) : max_splits;
        num_splits = std::max(CeilDiv64(mix_blocks, num_tokens * num_kv_heads), by_context);
        num_splits = std::min(num_splits, std::min(max_splits, blocks));
    } else if (split_policy == FusedSplitPolicy::kFillBlocks) {
        // Only as many splits as leave every task a block of its own: the chunking at one split is
        // what a split multiplies, and it does not change once the splits fit.
        const int64_t unsplit_tasks = num_tokens * num_kv_heads * chunks_for(num_tokens * num_kv_heads);
        const int64_t fill = std::min(std::min(max_splits, blocks), mix_blocks / unsplit_tasks);
        if (fill > 1) {
            num_splits = fill;
            launch_limit = 0;
        }
    }
    grid.num_splits = num_splits;
    grid.fused_context_limit = static_cast<uint32_t>(std::max<int64_t>(0, launch_limit));

    const int64_t groups = num_tokens * num_kv_heads * num_splits;
    const int64_t heads_per_task = group > 1 ? CeilDiv64(group, chunks_for(groups)) : 1;
    grid.heads_per_task = static_cast<uint32_t>(heads_per_task);

    const int64_t tasks = groups * CeilDiv64(group, heads_per_task);
    const int64_t tasks_per_block = CeilDiv64(tasks, mix_blocks);
    const int64_t block_dim = CeilDiv64(tasks, tasks_per_block);
    grid.num_tasks = tasks;
    grid.tasks_per_block = static_cast<uint32_t>(tasks_per_block);
    grid.block_dim = static_cast<uint32_t>(block_dim);
    if (num_splits > 1) {
        const int64_t reduce_tasks = num_tokens * num_heads;
        grid.reduce_tasks_per_block = static_cast<uint32_t>(CeilDiv64(reduce_tasks, block_dim));
        grid.workspace_floats = static_cast<size_t>(reduce_tasks * num_splits * (head_size + kPartialTail));
    }
    return grid;
}

RotateQPlan PlanRotateQ(int64_t num_tokens, int64_t num_vectors, int64_t head_size, int64_t core_num,
                        RotateQPrecision precision)
{
    RotateQPlan plan;
    if (num_tokens <= 0 || num_vectors <= 0 || head_size <= 0) {
        return plan;
    }
    if (core_num < 1) {
        core_num = 1;
    }

    const int64_t tile = kRotateQTile;
    const int64_t fits = static_cast<int64_t>(kRotateQMaxChunkElements) / head_size;
    const bool cube_worth_staging =
        num_tokens >= static_cast<int64_t>(kRotateQMinCubeTokens) && num_vectors >= core_num;

    if (cube_worth_staging && num_vectors >= tile && num_vectors % tile == 0 && fits >= 2) {
        int64_t vectors_per_block = num_vectors;
        for (int64_t candidate = tile; candidate <= num_vectors; candidate += tile) {
            if (num_vectors % candidate != 0) {
                continue;
            }
            if (num_vectors / candidate <= core_num) {
                vectors_per_block = candidate;
                break;
            }
        }

        int64_t vectors_per_chunk = fits < vectors_per_block ? fits : vectors_per_block;
        if (vectors_per_chunk % 2 != 0) {
            --vectors_per_chunk;
        }
        while (vectors_per_chunk >= 2 && vectors_per_block % vectors_per_chunk != 0) {
            vectors_per_chunk -= 2;
        }

        if (vectors_per_chunk >= 2) {
            plan.use_cube = true;
            plan.vectors_per_block = static_cast<uint32_t>(vectors_per_block);
            plan.vectors_per_chunk = static_cast<uint32_t>(vectors_per_chunk);
            plan.block_dim = static_cast<uint32_t>(num_vectors / vectors_per_block);
            plan.variant = static_cast<uint32_t>(kDualDst);
            if (precision == RotateQPrecision::kHiLoResidual) {
                plan.variant |= static_cast<uint32_t>(kHiLo);
            }
            return plan;
        }
    }

    int64_t vectors_per_block = CeilDiv64(num_vectors, core_num);
    if (vectors_per_block < 2) {
        vectors_per_block = 2;
    }
    plan.use_cube = false;
    plan.vectors_per_block = static_cast<uint32_t>(vectors_per_block);
    plan.vectors_per_chunk = 1;
    plan.block_dim = static_cast<uint32_t>(CeilDiv64(num_vectors, vectors_per_block));
    plan.variant = 0;
    return plan;
}

void FillHadamard16Half(uint16_t *out)
{
    const int64_t tile = kRotateQTile;
    for (int64_t i = 0; i < tile; ++i) {
        for (int64_t j = 0; j < tile; ++j) {
            unsigned bits = static_cast<unsigned>(i & j);
            int parity = 0;
            while (bits != 0) {
                parity ^= static_cast<int>(bits & 1u);
                bits >>= 1;
            }
            out[i * tile + j] = parity ? kRotateQHalfMinusOne : kRotateQHalfOne;
        }
    }
}

}
}
