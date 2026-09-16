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

// Host-side tiling of the TurboQuant operators: how a launch is cut into blocks and tasks, and how big
// every buffer the caller owns has to be. Torch-free and toolkit-free, so the adapter and the C++ test
// suite call the same arithmetic instead of each keeping a copy.

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_TILING_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_TILING_H

#include <cstddef>
#include <cstdint>

#include "../op_kernel/common/turboquant_layout.h"

namespace vllm_ascend {
namespace turboquant {

int64_t CeilDiv64(int64_t a, int64_t b);

int64_t ScaleSlotFloats64(int64_t num_kv_heads);

// Packed 4-bit cache bytes and scale-plane floats for the AIV decode's layout.
size_t PackedCacheBytes(int64_t num_blocks, int64_t block_size, int64_t num_kv_heads, int64_t head_size);
size_t ScalePlaneFloats(int64_t num_blocks, int64_t block_size, int64_t num_kv_heads);

// Words in the AIV codec's table image for batch_rows vectors per launch step.
int64_t CodecTableWords(int64_t head_size, int64_t batch_rows);

// A vector core count when the device query failed: every plan still has to be launchable.
int64_t RotateQCoreNum(int64_t aiv_num);

struct ReshapeAndCacheGrid {
    uint32_t block_dim = 0;
    uint32_t tokens_per_core = 0;
};

ReshapeAndCacheGrid PlanReshapeAndCache(int64_t num_tokens, int64_t aiv_num);

// The AIV-only decode. One launch; a context of at most fused_context_limit tokens is never split and
// needs no workspace. Above it, the launch splits the sequence and reduces the partials in place.
struct PagedAttentionGrid {
    uint32_t block_dim = 0;
    uint32_t split_tasks_per_core = 0;
    uint32_t reduce_tasks_per_core = 0;
    int64_t num_splits = 1;
    size_t workspace_floats = 0;
};

PagedAttentionGrid PlanPagedAttention(int64_t num_tokens, int64_t num_heads, int64_t head_size,
                                      int64_t max_blocks_per_seq, int64_t block_size, int64_t aiv_num,
                                      int64_t fused_context_limit = kFusedContextLimit);

// The Cube decode. Tasks are (token, kv head, split, head chunk) over the AIC blocks; each task's heads
// are shared between the block's two vector subcores.
struct FusedDecodeGrid {
    uint32_t block_dim = 0;
    uint32_t tasks_per_block = 0;
    uint32_t reduce_tasks_per_block = 0;
    uint32_t heads_per_task = 0;
    int64_t num_splits = 1;
    int64_t num_tasks = 0;
    size_t workspace_floats = 0;
};

FusedDecodeGrid PlanFusedDecode(int64_t num_tokens, int64_t num_heads, int64_t num_kv_heads, int64_t head_size,
                                int64_t max_blocks_per_seq, int64_t block_size, int64_t aiv_num,
                                int64_t fused_context_limit = kFusedContextLimit);

// The query rotation Pi = D H D: on the Cube (Mmad over a 16x16 fp16 Hadamard plus AIV butterflies)
// when there are enough vectors to be worth staging, on the AIV cores alone otherwise.
struct RotateQPlan {
    bool use_cube = false;
    uint32_t block_dim = 0;
    uint32_t vectors_per_block = 0;
    uint32_t vectors_per_chunk = 0;
    uint32_t variant = 0;
};

RotateQPlan PlanRotateQ(int64_t num_tokens, int64_t num_vectors, int64_t head_size, int64_t core_num,
                        RotateQPrecision precision = RotateQPrecision::kSinglePassRint);

void FillHadamard16Half(uint16_t *out);

}
}

#endif
