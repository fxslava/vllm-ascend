// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
#pragma once

#include <cstdint>

#include "../../../kernels/types.h"

namespace vllm_ascend {

// Low-level launchers for the BF16 writer's signed INT4 cache. numBlocks is
// the physical cache capacity, needed to locate scale plane V after plane K.
// queryRot is FP32; output is FP16. A pre-rotated launch leaves output in the
// rotated basis. See ../README.md for tiling and scratch-buffer requirements.
void turboquant_mm_fused_decode_opt_impl(int32_t mode, AscendType type, void* stream, uint32_t blockDim, void* queryRot,
                                         void* keyCache, void* valueCache, void* scaleCache, void* blockTables,
                                         void* contextLens, void* modeTables, void* workspace, void* output,
                                         uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize,
                                         uint32_t blockSize, uint32_t numBlocks, uint32_t maxBlocksPerSeq,
                                         uint32_t numSplits, uint32_t headsPerTask, uint32_t tasksPerBlock,
                                         uint32_t reduceTasksPerBlock, uint32_t fusedContextLimit, float scale,
                                         float invSqrtLen);

void turboquant_mm_fused_decode_opt_raw_query_impl(
    int32_t mode, AscendType type, void* stream, uint32_t blockDim, void* query, void* piSigns, void* rotTables,
    void* h16, void* gate, void* queryRot, void* keyCache, void* valueCache, void* scaleCache, void* blockTables,
    void* contextLens, void* modeTables, void* workspace, void* output, uint32_t numTokens, uint32_t numHeads,
    uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t numBlocks, uint32_t maxBlocksPerSeq,
    uint32_t numSplits, uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,
    uint32_t prologueVectorsPerBlock, uint32_t prologueCubeChunkVectors, uint32_t outputStage,
    uint32_t fusedContextLimit, float scale, float invSqrtLen);

}  // namespace vllm_ascend
