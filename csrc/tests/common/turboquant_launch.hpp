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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../attention/turboquant/turboquant_mode.h"
#include "../../attention/turboquant/turboquant_rotate_q.h"
#include "../../kernels/types.h"

namespace vllm_ascend {

void turboquant_reshape_and_cache_impl(AscendType type, void *stream, uint32_t blockDim, void *key, void *value,
                                       void *keyCache, void *valueCache, void *scaleCache, void *slotMapping,
                                       void *piSigns, void *tables, uint32_t numTokens, uint32_t numKvHeads,
                                       uint32_t headSize, uint32_t blockSize, uint32_t tokensPerCore,
                                       float invSqrtLen);

void turboquant_paged_attention_impl(AscendType type, void *stream, uint32_t splitBlockDim, uint32_t combineBlockDim,
                                     void *queryRot, void *keyCache, void *valueCache, void *scaleCache,
                                     void *blockTables, void *contextLens, void *tables, void *workspace,
                                     void *output, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                     uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                     uint32_t numSplits, uint32_t splitTasksPerCore, uint32_t combineTasksPerCore,
                                     float scale, float invSqrtLen);

void turboquant_rotate_q_impl(AscendType type, void *stream, uint32_t blockDim, bool useCube, void *query,
                              void *piSigns, void *h16, void *rotTables, void *queryRot, uint32_t numVectors,
                              uint32_t headSize, uint32_t vectorsPerBlock, uint32_t vectorsPerChunk,
                              uint32_t variant, float invSqrtLen);

void turboquant_paged_attention_combine_impl(AscendType type, void *stream, uint32_t blockDim, void *workspace,
                                             void *output, uint32_t numTokens, uint32_t numHeads, uint32_t headSize,
                                             uint32_t numSplits, uint32_t tasksPerCore);

void turboquant_mm_reshape_and_cache_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *key,
                                          void *value, void *keyCache, void *valueCache, void *scaleCache,
                                          void *slotMapping, void *piSigns, void *rotTables, void *modeTables,
                                          uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,
                                          uint32_t blockSize, uint32_t tokensPerCore, float invSqrtLen);

void turboquant_mm_decode_split_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *queryRot,
                                     void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
                                     void *contextLens, void *modeTables,
                                     void *workspace, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                     uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                     uint32_t numSplits, uint32_t tasksPerCore, float scale, float invSqrtLen);

void turboquant_mm_decode_ablation_impl(int32_t stage, AscendType type, void *stream, uint32_t blockDim,
                                        void *queryRot,
                                        void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
                                        void *contextLens, void *modeTables,
                                        void *workspace, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                        uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                        uint32_t numSplits, uint32_t tasksPerCore, float scale, float invSqrtLen);

void turboquant_mm_decode_bypass_unpack_impl(AscendType type, void *stream, uint32_t blockDim, void *queryRot,
                                             void *keyOperandCache, void *valueOperandCache, void *scaleCache,
                                             void *blockTables, void *contextLens, void *modeTables,
                                             void *workspace, uint32_t numTokens, uint32_t numHeads,
                                             uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,
                                             uint32_t maxBlocksPerSeq, uint32_t numSplits, uint32_t tasksPerCore,
                                             float scale, float invSqrtLen);

void turboquant_fp16_decode_impl(AscendType type, void *stream, uint32_t splitBlockDim, uint32_t combineBlockDim,
                                 void *query, void *keyCache, void *valueCache, void *blockTables,
                                 void *contextLens, void *workspace, void *output, uint32_t numTokens,
                                 uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,
                                 uint32_t maxBlocksPerSeq, uint32_t numSplits, uint32_t splitTasksPerCore,
                                 uint32_t combineTasksPerCore, float scale);

void turboquant_cube_gemm_probe_impl(void *stream, void *a, void *b, void *c, uint32_t m, uint32_t k, uint32_t n,
                                     uint32_t headSize, uint32_t tileRows, uint32_t aElems, uint32_t bElems,
                                     uint32_t cElems, uint32_t bIsNk, uint32_t variant);

namespace test {
namespace turboquant_host {

constexpr int64_t kMaxSequenceSplits = 8;
constexpr int64_t kPartialTail = 16;
constexpr int64_t kFp32PerBlock = 8;
constexpr int64_t kTileRows = 16;
constexpr int64_t kPackFactor = 2;

constexpr int64_t kFallbackVectorCoreNum = 8;

inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline int64_t ScaleSlotFloats(int64_t num_kv_heads) {
  return CeilDiv(2 * num_kv_heads, kFp32PerBlock) * kFp32PerBlock;
}

constexpr int64_t kCodecLevels = 16;

inline int64_t CodecTableWords(int64_t head_size, int64_t batch_rows) {
  return 7 * head_size + 2 * head_size * batch_rows + kCodecLevels;
}

inline size_t PackedCacheBytes(int64_t num_blocks, int64_t block_size, int64_t num_kv_heads, int64_t head_size) {
  return static_cast<size_t>(num_blocks) * static_cast<size_t>(block_size) * static_cast<size_t>(num_kv_heads) *
         static_cast<size_t>(head_size / kPackFactor);
}

inline size_t ScalePlaneFloats(int64_t num_blocks, int64_t block_size, int64_t num_kv_heads) {
  return static_cast<size_t>(num_blocks) * static_cast<size_t>(block_size) *
         static_cast<size_t>(ScaleSlotFloats(num_kv_heads));
}

std::vector<float> PiSigns(int64_t head_size);

std::vector<int32_t> CodecTables(int64_t head_size, int64_t batch_rows);

int64_t VectorCoreNum(bool *queried);

struct ReshapeAndCacheGrid {
  uint32_t block_dim = 0;
  uint32_t tokens_per_core = 0;
};

ReshapeAndCacheGrid PlanReshapeAndCache(int64_t num_tokens, int64_t aiv_num);

struct PagedAttentionGrid {
  uint32_t split_block_dim = 0;
  uint32_t combine_block_dim = 0;
  uint32_t split_tasks_per_core = 0;
  uint32_t combine_tasks_per_core = 0;
  int64_t num_splits = 1;
  size_t workspace_floats = 0;
};

PagedAttentionGrid PlanPagedAttention(int64_t num_tokens, int64_t num_heads, int64_t head_size,
                                      int64_t max_blocks_per_seq, int64_t aiv_num);

std::vector<float> UnrotateHeads(std::vector<float> rotated, int64_t head_size);

struct CombineUbFootprint {
  size_t out_queue = 0;
  size_t accumulators = 0;
  size_t state = 0;
  size_t partial = 0;
  size_t broadcast = 0;
  size_t unrotation_codec_tables = 0;
  size_t unrotation_codec_scratch = 0;
  size_t unrotation_signs = 0;
  size_t unrotation_ping_pong = 0;

  size_t Current() const { return out_queue + accumulators + state + partial + broadcast; }
  size_t Released() const {
    return unrotation_codec_tables + unrotation_codec_scratch + unrotation_signs + unrotation_ping_pong;
  }
  size_t Previous() const { return Current() + Released(); }
};

CombineUbFootprint PlanCombineUb(int64_t head_size, int64_t scalar_bytes);

inline int64_t CodecWorkBufferWords(int64_t head_size, int64_t batch_rows) {
  constexpr int64_t kBrcbDstLanes = kFp32PerBlock * kFp32PerBlock;
  constexpr int64_t kBinLanes = 8;
  return 2 * head_size * batch_rows + head_size + kBrcbDstLanes + kFp32PerBlock + kBinLanes * head_size;
}

constexpr int64_t kCubeTileRows = 64;
constexpr int64_t kUnpackRows = 8;
constexpr int64_t kOperandC0 = 32;
constexpr int64_t kCubeTileM = 16;

int64_t ModePackedBytes(vllm_ascend::turboquant::TurboQuantMode mode, int64_t head_size);

size_t ModePackedCacheBytes(vllm_ascend::turboquant::TurboQuantMode mode, int64_t num_blocks, int64_t block_size,
                            int64_t num_kv_heads, int64_t head_size);

int64_t ModeTableWords(vllm_ascend::turboquant::TurboQuantMode mode, int64_t head_size, int64_t batch_rows);

std::vector<int32_t> ModeTables(vllm_ascend::turboquant::TurboQuantMode mode, int64_t head_size, int64_t batch_rows,
                                int64_t nz_rows);

inline int64_t NzOffset(int64_t r, int64_t c, int64_t rows) {
  return (c / kOperandC0) * rows * kOperandC0 + r * kOperandC0 + (c % kOperandC0);
}

struct CubeDecodeGrid {
  uint32_t split_block_dim = 0;
  uint32_t combine_block_dim = 0;
  uint32_t split_tasks_per_core = 0;
  uint32_t combine_tasks_per_core = 0;
  int64_t num_splits = 1;
  size_t workspace_floats = 0;
};

CubeDecodeGrid PlanCubeDecode(int64_t num_tokens, int64_t num_heads, int64_t num_kv_heads, int64_t head_size,
                              int64_t max_blocks_per_seq, int64_t aiv_num);

std::vector<uint16_t> Hadamard16Half();

vllm_ascend::turboquant::RotateQPlan RotateQuery(void *stream, AscendType type, void *query, void *pi_signs,
                                                 void *h16, void *rot_tables, void *query_rot, int64_t num_tokens,
                                                 int64_t num_heads, int64_t head_size, int64_t aiv_num,
                                                 bool input_exact_in_half);

}
}
}
