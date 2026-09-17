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

// The TurboQuant host tiling (op_host/turboquant_tiling.cpp) is what the adapter compiles into the wheel.
// These are its contracts, checked without a toolkit: a context inside the fused limit is one launch with
// no workspace unless the Cube planner splits it to fill idle MIX blocks, a split launch sizes its
// workspace for every partial, every task lands on a block, and a Cube task never carries more heads than
// the Cube's M fractal.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "op_host/turboquant_tiling.h"

namespace vllm_ascend {
namespace test {
namespace {

namespace tqt = vllm_ascend::turboquant;

constexpr int64_t kHeadSize = 256;
const int64_t kVectorCores[] = {8, 40, 64};

TEST(TurboQuantTiling, PagedAttentionIsOneLaunchWithoutWorkspaceInsideTheFusedLimit) {
  for (const int64_t aiv : kVectorCores) {
    for (const int64_t tokens : {1, 4, 64}) {
      for (const int64_t heads : {4, 16, 32}) {
        for (const int64_t block_size : {16, 128}) {
          for (const int64_t blocks : {0, 1, 3, 32, 64, 256}) {
            const tqt::PagedAttentionGrid grid =
                tqt::PlanPagedAttention(tokens, heads, kHeadSize, blocks, block_size, aiv);
            const std::string where = "aiv " + std::to_string(aiv) + " tokens " + std::to_string(tokens) +
                                      " heads " + std::to_string(heads) + " blocks " + std::to_string(blocks) +
                                      " block_size " + std::to_string(block_size);
            const int64_t bound = std::max<int64_t>(blocks, 1) * block_size;
            ASSERT_GE(grid.num_splits, 1) << where;
            ASSERT_LE(grid.num_splits, static_cast<int64_t>(tqt::kMaxSequenceSplits)) << where;
            ASSERT_GT(grid.block_dim, 0u) << where;
            ASSERT_LE(static_cast<int64_t>(grid.block_dim), aiv) << where;
            const int64_t tasks = tokens * heads * grid.num_splits;
            ASSERT_GE(static_cast<int64_t>(grid.block_dim) * grid.split_tasks_per_core, tasks) << where;
            if (bound <= static_cast<int64_t>(tqt::kFusedContextLimit)) {
              EXPECT_EQ(grid.num_splits, 1) << where;
              EXPECT_EQ(grid.workspace_floats, 0u) << where;
              EXPECT_EQ(grid.reduce_tasks_per_core, 0u) << where;
            } else if (grid.num_splits > 1) {
              EXPECT_EQ(grid.workspace_floats,
                        static_cast<size_t>(tasks * (kHeadSize + static_cast<int64_t>(tqt::kPartialTail))))
                  << where;
              EXPECT_GE(static_cast<int64_t>(grid.block_dim) * grid.reduce_tasks_per_core, tokens * heads) << where;
            }
          }
        }
      }
    }
  }
}

TEST(TurboQuantTiling, FusedDecodeTasksCoverEveryHeadOnTheCubeFractal) {
  for (const tqt::FusedSplitPolicy policy :
       {tqt::FusedSplitPolicy::kContextOnly, tqt::FusedSplitPolicy::kFillBlocks, tqt::FusedSplitPolicy::kAdaptive}) {
    for (const int64_t aiv : kVectorCores) {
      for (const int64_t tokens : {1, 8}) {
        for (const int64_t kv_heads : {1, 2, 4, 8}) {
          for (const int64_t group : {1, 2, 5, 16, 20}) {
            for (const int64_t blocks : {1, 2, 4, 64, 128}) {
              const int64_t heads = kv_heads * group;
              const int64_t mix_blocks = std::max<int64_t>(1, aiv / 2);
              const tqt::FusedDecodeGrid grid =
                  tqt::PlanFusedDecode(tokens, heads, kv_heads, kHeadSize, blocks,
                                       static_cast<int64_t>(tqt::kCubeTileRows), aiv, tqt::kFusedContextLimit, policy);
              const std::string where =
                  "policy " + std::to_string(static_cast<int>(policy)) + " aiv " + std::to_string(aiv) + " tokens " +
                  std::to_string(tokens) + " kv_heads " + std::to_string(kv_heads) + " group " +
                  std::to_string(group) + " blocks " + std::to_string(blocks);
              ASSERT_GE(grid.heads_per_task, 1u) << where;
              ASSERT_LE(grid.heads_per_task, tqt::kCubeTileM) << where;
              ASSERT_LE(static_cast<int64_t>(grid.heads_per_task), group) << where;
              const int64_t chunks = (group + grid.heads_per_task - 1) / grid.heads_per_task;
              ASSERT_EQ(grid.num_tasks, tokens * kv_heads * grid.num_splits * chunks) << where;
              ASSERT_GE(static_cast<int64_t>(grid.block_dim) * grid.tasks_per_block, grid.num_tasks) << where;
              ASSERT_LE(static_cast<int64_t>(grid.block_dim), mix_blocks) << where;
              // The raw-query prologue rotates [block * per_block, (block + 1) * per_block) on each block.
              ASSERT_GE(static_cast<int64_t>(grid.block_dim) * grid.prologue_vectors_per_block, tokens * heads)
                  << where;
              ASSERT_GE(grid.num_splits, 1) << where;
              ASSERT_LE(grid.num_splits, std::min<int64_t>(blocks, tqt::kMaxSequenceSplits)) << where;
              if (grid.num_splits > 1) {
                EXPECT_EQ(grid.workspace_floats,
                          static_cast<size_t>(tokens * heads * grid.num_splits *
                                              (kHeadSize + static_cast<int64_t>(tqt::kPartialTail))))
                    << where;
                EXPECT_GE(static_cast<int64_t>(grid.block_dim) * grid.reduce_tasks_per_block, tokens * heads)
                    << where;
              } else {
                EXPECT_EQ(grid.workspace_floats, 0u) << where;
                EXPECT_EQ(grid.reduce_tasks_per_block, 0u) << where;
              }
              if (policy == tqt::FusedSplitPolicy::kAdaptive) {
                // The unsplit tasks are what a one-block context plans to under any policy.
                const int64_t base_tasks =
                    tqt::PlanFusedDecode(tokens, heads, kv_heads, kHeadSize, 1,
                                         static_cast<int64_t>(tqt::kCubeTileRows), aiv, tqt::kFusedContextLimit,
                                         tqt::FusedSplitPolicy::kContextOnly)
                        .num_tasks;
                const int64_t want =
                    base_tasks >= mix_blocks
                        ? 1
                        : std::min((mix_blocks + base_tasks - 1) / base_tasks,
                                   std::min<int64_t>(blocks, tqt::kMaxSequenceSplits));
                EXPECT_EQ(grid.num_splits, want) << where << " base_tasks " << base_tasks;
                EXPECT_EQ(grid.fused_context_limit, grid.num_splits > 1 ? 0u : tqt::kFusedContextLimit) << where;
                continue;
              }
              if (blocks * static_cast<int64_t>(tqt::kCubeTileRows) > static_cast<int64_t>(tqt::kFusedContextLimit)) {
                EXPECT_EQ(grid.fused_context_limit, tqt::kFusedContextLimit) << where;
                continue;
              }
              if (policy == tqt::FusedSplitPolicy::kContextOnly) {
                EXPECT_EQ(grid.num_splits, 1) << where;
                EXPECT_EQ(grid.fused_context_limit, tqt::kFusedContextLimit) << where;
                continue;
              }
              // Filling splits a context inside the limit only when every task still gets a block of its
              // own, and then the launch must be told to reduce them.
              if (grid.num_splits > 1) {
                EXPECT_EQ(grid.fused_context_limit, 0u) << where;
                EXPECT_EQ(grid.tasks_per_block, 1u) << where;
                EXPECT_LE(grid.num_tasks, mix_blocks) << where;
              } else {
                EXPECT_EQ(grid.fused_context_limit, tqt::kFusedContextLimit) << where;
                EXPECT_TRUE(blocks == 1 || 2 * tokens * kv_heads * chunks > mix_blocks) << where;
              }
            }
          }
        }
      }
    }
  }
}

TEST(TurboQuantTiling, FusedDecodeFillsEveryMixBlockAtTheSingleTokenBaseline) {
  constexpr int64_t kAiv = 64;
  constexpr int64_t kMixBlocks = kAiv / 2;
  constexpr int64_t kBaselineHeads = 16;
  constexpr int64_t kBaselineKvHeads = 8;
  constexpr int64_t kContextBlocks = 32;
  constexpr int64_t kSplits = 4;
  const tqt::FusedDecodeGrid grid = tqt::PlanFusedDecode(
      1, kBaselineHeads, kBaselineKvHeads, kHeadSize, kContextBlocks, static_cast<int64_t>(tqt::kCubeTileRows), kAiv);
  EXPECT_EQ(grid.num_splits, kSplits);
  EXPECT_EQ(grid.heads_per_task, 2u);
  EXPECT_EQ(grid.num_tasks, kMixBlocks);
  EXPECT_EQ(grid.block_dim, static_cast<uint32_t>(kMixBlocks));
  EXPECT_EQ(grid.tasks_per_block, 1u);
  EXPECT_EQ(grid.fused_context_limit, 0u);
  EXPECT_EQ(grid.reduce_tasks_per_block, 1u);

  const tqt::FusedDecodeGrid unsplit =
      tqt::PlanFusedDecode(1, kBaselineHeads, kBaselineKvHeads, kHeadSize, kContextBlocks,
                           static_cast<int64_t>(tqt::kCubeTileRows), kAiv, tqt::kFusedContextLimit,
                           tqt::FusedSplitPolicy::kContextOnly);
  EXPECT_EQ(unsplit.num_splits, 1);
  EXPECT_EQ(unsplit.block_dim, static_cast<uint32_t>(kBaselineKvHeads));
  EXPECT_EQ(unsplit.fused_context_limit, tqt::kFusedContextLimit);

  // The msprof trace's Cube leg, DeepSeek-V4-Flash at S 2048 and block 128: eight 2-head chunks of the
  // 16:1 group, each split four times.
  constexpr int64_t kTraceHeads = 16;
  constexpr int64_t kTraceKvHeads = 1;
  constexpr int64_t kTraceBlockSize = 128;
  constexpr int64_t kTraceContext = 2048;
  const tqt::FusedDecodeGrid trace = tqt::PlanFusedDecode(1, kTraceHeads, kTraceKvHeads, kHeadSize,
                                                          kTraceContext / kTraceBlockSize, kTraceBlockSize, kAiv);
  EXPECT_EQ(trace.num_splits, kSplits);
  EXPECT_EQ(trace.heads_per_task, 2u);
  EXPECT_EQ(trace.block_dim, static_cast<uint32_t>(kMixBlocks));
  EXPECT_EQ(trace.fused_context_limit, 0u);
}

// The bench's decode rows (aiv 64, block 128) under the default adaptive policy, against the fill policy it
// replaced: a saturated grid runs unsplit at every context, an under-filled one splits at every context.
TEST(TurboQuantTiling, FusedDecodeAdaptiveSplitsOnlyAnUnderfilledGrid) {
  constexpr int64_t kAiv = 64;
  constexpr int64_t kBlock = 128;
  struct Row {
    int64_t heads;
    int64_t head_size;
    int64_t context;
    int64_t batch;
    int64_t adaptive_splits;
    int64_t fill_splits;
  };
  // H_KV 1 throughout: Qwen3.5 4:1 D 128, GLM-5.2 8:1 D 128, DeepSeek-V4-Flash 16:1 D 256.
  const Row rows[] = {
      {4, 128, 2048, 1, 8, 8},   {4, 128, 2048, 4, 4, 4},   {4, 128, 2048, 8, 2, 2},
      {4, 128, 32768, 1, 8, 8},  {4, 128, 32768, 4, 4, 8},  {4, 128, 32768, 8, 2, 8},
      {8, 128, 2048, 1, 8, 8},   {8, 128, 2048, 4, 2, 2},   {8, 128, 2048, 8, 1, 1},
      {8, 128, 32768, 1, 8, 8},  {8, 128, 32768, 4, 2, 8},  {8, 128, 32768, 8, 1, 8},
      {16, 256, 2048, 1, 4, 4},  {16, 256, 2048, 4, 1, 1},  {16, 256, 32768, 4, 1, 8},
  };
  for (const Row& row : rows) {
    const int64_t blocks = (row.context + kBlock - 1) / kBlock;
    const std::string where = "H_Q " + std::to_string(row.heads) + " S " + std::to_string(row.context) + " B " +
                              std::to_string(row.batch);
    const tqt::FusedDecodeGrid adaptive =
        tqt::PlanFusedDecode(row.batch, row.heads, 1, row.head_size, blocks, kBlock, kAiv);
    const tqt::FusedDecodeGrid fill = tqt::PlanFusedDecode(row.batch, row.heads, 1, row.head_size, blocks, kBlock,
                                                           kAiv, tqt::kFusedContextLimit,
                                                           tqt::FusedSplitPolicy::kFillBlocks);
    EXPECT_EQ(adaptive.num_splits, row.adaptive_splits) << where;
    EXPECT_EQ(fill.num_splits, row.fill_splits) << where;
    if (adaptive.num_splits == 1) {
      EXPECT_EQ(adaptive.workspace_floats, 0u) << where;
      EXPECT_EQ(adaptive.reduce_tasks_per_block, 0u) << where;
    } else {
      EXPECT_EQ(adaptive.fused_context_limit, 0u) << where;
    }
  }
}

// Every camodel fused case (test_sim_950pr_turboquant_fused, S <= 256) plans the same grid under the adaptive
// policy as under the one its golden was recorded with, so the default change moves no executed grid.
TEST(TurboQuantTiling, FusedDecodeAdaptiveKeepsTheCamodelCaseGrids) {
  constexpr int64_t kAiv = 64;
  constexpr int64_t kTwoBlockAiv = 4;
  constexpr int64_t kBlock = 64;
  struct Case {
    int64_t heads;
    int64_t kv_heads;
    int64_t head_size;
    int64_t context;
    int64_t aiv;
  };
  const Case cases[] = {
      {4, 2, 256, 64, kAiv}, {4, 2, 256, 256, kAiv}, {8, 2, 256, 120, kTwoBlockAiv}, {4, 1, 128, 64, kAiv}};
  for (const Case& c : cases) {
    const int64_t blocks = (c.context + kBlock - 1) / kBlock;
    const std::string where = "H_Q " + std::to_string(c.heads) + " S " + std::to_string(c.context);
    const tqt::FusedDecodeGrid adaptive =
        tqt::PlanFusedDecode(1, c.heads, c.kv_heads, c.head_size, blocks, kBlock, c.aiv);
    const tqt::FusedDecodeGrid fill = tqt::PlanFusedDecode(1, c.heads, c.kv_heads, c.head_size, blocks, kBlock, c.aiv,
                                                           tqt::kFusedContextLimit,
                                                           tqt::FusedSplitPolicy::kFillBlocks);
    EXPECT_EQ(adaptive.num_splits, fill.num_splits) << where;
    EXPECT_EQ(adaptive.heads_per_task, fill.heads_per_task) << where;
    EXPECT_EQ(adaptive.block_dim, fill.block_dim) << where;
    EXPECT_EQ(adaptive.tasks_per_block, fill.tasks_per_block) << where;
    EXPECT_EQ(adaptive.fused_context_limit, fill.fused_context_limit) << where;
  }
}

TEST(TurboQuantTiling, RotateQStagesTheCubeOnlyWithEvenDualDestinationChunks) {
  for (const int64_t aiv : kVectorCores) {
    const int64_t cores = tqt::RotateQCoreNum(aiv);
    for (const int64_t tokens : {1, 2, 16, 64}) {
      for (const int64_t heads : {4, 8, 32}) {
        for (const int64_t head_size : {64, 128, 256}) {
          const int64_t vectors = tokens * heads;
          const tqt::RotateQPlan plan = tqt::PlanRotateQ(tokens, vectors, head_size, cores);
          const std::string where = "aiv " + std::to_string(aiv) + " tokens " + std::to_string(tokens) +
                                    " heads " + std::to_string(heads) + " head_size " + std::to_string(head_size);
          ASSERT_GT(plan.block_dim, 0u) << where;
          ASSERT_GE(static_cast<int64_t>(plan.block_dim) * plan.vectors_per_block, vectors) << where;
          if (plan.use_cube) {
            EXPECT_EQ(vectors % plan.vectors_per_block, 0) << where;
            EXPECT_EQ(plan.vectors_per_chunk % 2, 0u) << where;
            EXPECT_EQ(plan.vectors_per_block % plan.vectors_per_chunk, 0u) << where;
            EXPECT_LE(static_cast<int64_t>(plan.vectors_per_chunk) * head_size,
                      static_cast<int64_t>(tqt::kRotateQMaxChunkElements))
                << where;
            EXPECT_EQ(plan.variant, static_cast<uint32_t>(tqt::kDualDst)) << where;
          } else {
            EXPECT_EQ(plan.vectors_per_chunk, 1u) << where;
            EXPECT_EQ(plan.variant, 0u) << where;
          }
          if (tokens < static_cast<int64_t>(tqt::kRotateQMinCubeTokens)) {
            EXPECT_FALSE(plan.use_cube) << where << ": a single decode token is never worth staging";
          }
        }
      }
    }
  }
}

TEST(TurboQuantTiling, Hadamard16IsSylvesterInHalfPrecision) {
  std::vector<uint16_t> h(tqt::kRotateQH16Elements);
  tqt::FillHadamard16Half(h.data());
  for (uint32_t i = 0; i < tqt::kRotateQTile; ++i) {
    EXPECT_EQ(h[i * tqt::kRotateQTile], tqt::kRotateQHalfOne) << "row " << i;
    EXPECT_EQ(h[i], tqt::kRotateQHalfOne) << "column " << i;
    for (uint32_t j = i + 1; j < tqt::kRotateQTile; ++j) {
      int dot = 0;
      for (uint32_t k = 0; k < tqt::kRotateQTile; ++k) {
        const int a = h[i * tqt::kRotateQTile + k] == tqt::kRotateQHalfOne ? 1 : -1;
        const int b = h[j * tqt::kRotateQTile + k] == tqt::kRotateQHalfOne ? 1 : -1;
        dot += a * b;
      }
      EXPECT_EQ(dot, 0) << "rows " << i << " and " << j << " are not orthogonal";
    }
  }
}

TEST(TurboQuantTiling, BufferSizesFollowTheLayout) {
  EXPECT_EQ(tqt::ScaleSlotFloats64(1), 8);
  EXPECT_EQ(tqt::ScaleSlotFloats64(4), 8);
  EXPECT_EQ(tqt::ScaleSlotFloats64(5), 16);
  EXPECT_EQ(tqt::PackedCacheBytes(3, 128, 2, 256), static_cast<size_t>(3 * 128 * 2 * 128));
  EXPECT_EQ(tqt::ScalePlaneFloats(3, 128, 5), static_cast<size_t>(3 * 128 * 16));
  EXPECT_EQ(tqt::CodecTableWords(256, 16), 7 * 256 + 2 * 256 * 16 + 16);
  EXPECT_EQ(tqt::RotateQCoreNum(0), 1);
  EXPECT_EQ(tqt::RotateQCoreNum(64), 32);
}

}
}
}
