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

#include "../../attention/turboquant/op_host/turboquant_tiling.h"
#include "../../attention/turboquant/op_kernel/common/turboquant_mode.h"
#include "../../kernels/types.h"

namespace vllm_ascend {

void turboquant_reshape_and_cache_impl(AscendType type, void *stream, uint32_t blockDim, void *key, void *value,
                                       void *keyCache, void *valueCache, void *scaleCache, void *slotMapping,
                                       void *piSigns, void *tables, uint32_t numTokens, uint32_t numKvHeads,
                                       uint32_t headSize, uint32_t blockSize, uint32_t numBlocks,
                                       uint32_t tokensPerCore, float invSqrtLen);

void turboquant_paged_attention_impl(AscendType type, void *stream, uint32_t blockDim, void *queryRot,
                                     void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
                                     void *contextLens, void *tables, void *workspace, void *output,
                                     uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize,
                                     uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,
                                     uint32_t splitTasksPerCore, uint32_t reduceTasksPerCore,
                                     uint32_t fusedContextLimit, float scale, float invSqrtLen);

void turboquant_rotate_q_impl(AscendType type, void *stream, uint32_t blockDim, bool useCube, void *query,
                              void *piSigns, void *h16, void *rotTables, void *queryRot, uint32_t numVectors,
                              uint32_t headSize, uint32_t vectorsPerBlock, uint32_t vectorsPerChunk,
                              uint32_t variant, float invSqrtLen);

void turboquant_mm_reshape_and_cache_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *key,
                                          void *value, void *keyCache, void *valueCache, void *scaleCache,
                                          void *slotMapping, void *piSigns, void *rotTables, void *modeTables,
                                          uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,
                                          uint32_t blockSize, uint32_t numBlocks, uint32_t tokensPerCore,
                                          float invSqrtLen);

void turboquant_mm_fused_decode_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *queryRot,
                                    void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
                                    void *contextLens, void *modeTables, void *workspace, void *output,
                                    uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize,
                                    uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,
                                    uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,
                                    uint32_t fusedContextLimit, float scale, float invSqrtLen);

// The fused decode handed the raw query: the launch rotates it into queryRot before the task loop (on the Cube
// in chunks of prologueCubeChunkVectors, 0 for the vector cores alone), and outputStage
// (turboquant::TurboQuantOutputStage) selects what its output holds. h16 is read only with a chunk size and
// gate only for kGated. kv4fp8 only.
void turboquant_mm_fused_decode_raw_query_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim,
                                              void *query, void *piSigns, void *rotTables, void *h16, void *gate,
                                              void *queryRot, void *keyCache, void *valueCache, void *scaleCache,
                                              void *blockTables, void *contextLens, void *modeTables,
                                              void *workspace, void *output, uint32_t numTokens, uint32_t numHeads,
                                              uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,
                                              uint32_t maxBlocksPerSeq, uint32_t numSplits, uint32_t headsPerTask,
                                              uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,
                                              uint32_t prologueVectorsPerBlock, uint32_t prologueCubeChunkVectors,
                                              uint32_t outputStage, uint32_t fusedContextLimit, float scale,
                                              float invSqrtLen);

void turboquant_cube_gemm_probe_impl(void *stream, void *a, void *b, void *c, uint32_t m, uint32_t k, uint32_t n,
                                     uint32_t headSize, uint32_t tileRows, uint32_t aElems, uint32_t bElems,
                                     uint32_t cElems, uint32_t bIsNk, uint32_t variant);

namespace test {
namespace turboquant_host {

namespace tqt = vllm_ascend::turboquant;

constexpr int64_t kMaxSequenceSplits = tqt::kMaxSequenceSplits;
constexpr int64_t kFusedContextLimit = tqt::kFusedContextLimit;
constexpr int64_t kPartialTail = tqt::kPartialTail;
constexpr int64_t kFp32PerBlock = tqt::kFp32PerBlock;
constexpr int64_t kTileRows = tqt::kAivTileRows;
constexpr int64_t kPackFactor = tqt::kPackFactor;
constexpr int64_t kCodecLevels = tqt::kCodecLevels;

constexpr int64_t kVectorSubcoresPerBlock = tqt::kVectorSubcoresPerBlock;

constexpr int64_t kFallbackVectorCoreNum = 8;

using tqt::CodecTableWords;
using tqt::DecodeNeedsReduction;
using tqt::FusedDecodeGrid;
using tqt::FusedSplitPolicy;
using tqt::PackedCacheBytes;
using tqt::PagedAttentionGrid;
using tqt::PlanFusedDecode;
using tqt::PlanPagedAttention;
using tqt::PlanReshapeAndCache;
using tqt::ReshapeAndCacheGrid;
using tqt::ScalePlaneFloats;

inline int64_t CeilDiv(int64_t a, int64_t b) { return tqt::CeilDiv64(a, b); }

inline int64_t ScaleSlotFloats(int64_t num_kv_heads) { return tqt::ScaleSlotFloats64(num_kv_heads); }

std::vector<float> PiSigns(int64_t head_size);

std::vector<int32_t> CodecTables(int64_t head_size, int64_t batch_rows);

int64_t VectorCoreNum(bool *queried);

std::vector<float> UnrotateHeads(std::vector<float> rotated, int64_t head_size);

constexpr int64_t kCubeTileRows = tqt::kCubeTileRows;
constexpr int64_t kUnpackRows = tqt::kCubeUnpackRows;
constexpr int64_t kOperandC0 = tqt::kOperandC0;
constexpr int64_t kCubeTileM = tqt::kCubeTileM;

int64_t ModePackedBytes(vllm_ascend::turboquant::TurboQuantMode mode, int64_t head_size);

size_t ModePackedCacheBytes(vllm_ascend::turboquant::TurboQuantMode mode, int64_t num_blocks, int64_t block_size,
                            int64_t num_kv_heads, int64_t head_size);

int64_t ModeTableWords(vllm_ascend::turboquant::TurboQuantMode mode, int64_t head_size, int64_t batch_rows);

std::vector<int32_t> ModeTables(vllm_ascend::turboquant::TurboQuantMode mode, int64_t head_size, int64_t batch_rows,
                                int64_t nz_rows);

inline int64_t NzOffset(int64_t r, int64_t c, int64_t rows) {
  return (c / kOperandC0) * rows * kOperandC0 + r * kOperandC0 + (c % kOperandC0);
}

std::vector<uint16_t> Hadamard16Half();

vllm_ascend::turboquant::RotateQPlan RotateQuery(void *stream, AscendType type, void *query, void *pi_signs,
                                                 void *h16, void *rot_tables, void *query_rot, int64_t num_tokens,
                                                 int64_t num_heads, int64_t head_size, int64_t aiv_num,
                                                 vllm_ascend::turboquant::RotateQPrecision precision =
                                                     vllm_ascend::turboquant::RotateQPrecision::kSinglePassRint);

}
}
}
