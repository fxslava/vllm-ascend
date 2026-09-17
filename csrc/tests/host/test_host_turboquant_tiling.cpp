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
// no workspace unless the Cube planner splits it to fill idle MIX blocks, a bandwidth-bound Cube context
// always splits, a split launch sizes its workspace for every partial, every task lands on a block, and a
// Cube task never carries more heads than the Cube's M fractal.

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

int64_t CeilPow2(int64_t value) {
  int64_t pow2 = 1;
  while (pow2 < value) {
    pow2 *= 2;
  }
  return pow2;
}

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
            for (const int64_t blocks : {1, 2, 4, 64, 128, 130, 512}) {
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
              const int64_t bound = blocks * static_cast<int64_t>(tqt::kCubeTileRows);
              if (policy == tqt::FusedSplitPolicy::kAdaptive) {
                // The unsplit tasks are what a one-block context plans to under any policy.
                const int64_t base_tasks =
                    tqt::PlanFusedDecode(tokens, heads, kv_heads, kHeadSize, 1,
                                         static_cast<int64_t>(tqt::kCubeTileRows), aiv, tqt::kFusedContextLimit,
                                         tqt::FusedSplitPolicy::kContextOnly)
                        .num_tasks;
                int64_t wanted = base_tasks >= mix_blocks ? 1 : (mix_blocks + base_tasks - 1) / base_tasks;
                if (bound >= tqt::kBandwidthSplitContext) {
                  const int64_t stride_splits = (bound + tqt::kBandwidthSplitRows - 1) / tqt::kBandwidthSplitRows;
                  wanted = CeilPow2(std::max(wanted, stride_splits));
                }
                const int64_t cap = std::min<int64_t>(blocks, tqt::kMaxSequenceSplits);
                EXPECT_EQ(grid.num_splits, std::min(wanted, cap)) << where << " base_tasks " << base_tasks;
                EXPECT_EQ(grid.fused_context_limit, grid.num_splits > 1 ? 0u : tqt::kFusedContextLimit) << where;
                EXPECT_EQ(tqt::DecodeNeedsReduction(grid.num_splits, bound, grid.fused_context_limit),
                          grid.num_splits > 1)
                    << where;
                if (bound >= tqt::kBandwidthSplitContext) {
                  // A bandwidth-bound context splits on a saturated grid too.
                  EXPECT_GE(grid.num_splits, std::min(cap, bound / tqt::kBandwidthSplitRows)) << where;
                }
                continue;
              }
              if (bound > static_cast<int64_t>(tqt::kFusedContextLimit)) {
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

TEST(TurboQuantTiling, DecodeNeedsReductionMirrorsTheKernels) {
  EXPECT_FALSE(tqt::DecodeNeedsReduction(1, 32768, 0));
  EXPECT_FALSE(tqt::DecodeNeedsReduction(8, 4096, tqt::kFusedContextLimit));
  EXPECT_TRUE(tqt::DecodeNeedsReduction(8, 4097, tqt::kFusedContextLimit));
  EXPECT_TRUE(tqt::DecodeNeedsReduction(2, 64, 0));
}

// The bench's decode matrix (aiv 64 -> 32 MIX blocks, block 128, H_KV 1) under the default two-tier adaptive
// policy. The bandwidth tier restores the fill policy's K = 8 at every long context: the first adaptive draft
// planned 1 / 2 / 4 splits for DeepSeek / GLM / Qwen at S 32768 B 4, which silicon measured 4.0x / 2.9x / 1.7x
// slower. Short contexts keep saturation pruning (DeepSeek B >= 4, GLM B 8: K 1, no reduction).
TEST(TurboQuantTiling, FusedDecodeAdaptivePinsTheBenchSchedule) {
  constexpr int64_t kAiv = 64;
  constexpr int64_t kMixBlocks = kAiv / 2;
  constexpr int64_t kBlock = 128;
  struct Model {
    const char* label;
    int64_t heads;
    int64_t head_size;
  };
  const Model qwen = {"Qwen3.5 4:1 D128", 4, 128};
  const Model dsv4 = {"DeepSeek-V4-Flash 16:1 D256", 16, 256};
  const Model glm = {"GLM-5.2 8:1 D128", 8, 128};
  struct Row {
    const Model* model;
    int64_t context;
    int64_t batch;
    int64_t splits;
    int64_t tasks;
    uint32_t blocks;
  };
  const Row rows[] = {
      {&qwen, 2048, 1, 8, 16, 16},     {&qwen, 2048, 2, 8, 32, 32},     {&qwen, 2048, 4, 4, 32, 32},
      {&qwen, 2048, 8, 2, 32, 32},     {&qwen, 32768, 1, 8, 16, 16},    {&qwen, 32768, 2, 8, 32, 32},
      {&qwen, 32768, 4, 8, 32, 32},    {&qwen, 32768, 8, 8, 64, 32},    {&qwen, 262144, 1, 8, 16, 16},
      {&qwen, 262144, 2, 8, 32, 32},   {&qwen, 262144, 4, 8, 32, 32},   {&qwen, 262144, 8, 8, 64, 32},
      {&qwen, 1048576, 1, 8, 16, 16},  {&qwen, 1048576, 2, 8, 32, 32},  {&qwen, 1048576, 4, 8, 32, 32},
      {&qwen, 1048576, 8, 8, 64, 32},  {&dsv4, 2048, 1, 4, 32, 32},     {&dsv4, 2048, 2, 2, 32, 32},
      {&dsv4, 2048, 4, 1, 32, 32},     {&dsv4, 2048, 8, 1, 32, 32},     {&dsv4, 32768, 1, 8, 32, 32},
      {&dsv4, 32768, 2, 8, 32, 32},    {&dsv4, 32768, 4, 8, 32, 32},    {&dsv4, 32768, 8, 8, 64, 32},
      {&dsv4, 262144, 1, 8, 32, 32},   {&dsv4, 262144, 2, 8, 32, 32},   {&dsv4, 262144, 4, 8, 32, 32},
      {&dsv4, 262144, 8, 8, 64, 32},   {&dsv4, 1048576, 1, 8, 32, 32},  {&dsv4, 1048576, 2, 8, 32, 32},
      {&dsv4, 1048576, 4, 8, 32, 32},  {&dsv4, 1048576, 8, 8, 64, 32},  {&glm, 2048, 1, 8, 32, 32},
      {&glm, 2048, 2, 4, 32, 32},      {&glm, 2048, 4, 2, 32, 32},      {&glm, 2048, 8, 1, 32, 32},
      {&glm, 32768, 1, 8, 32, 32},     {&glm, 32768, 2, 8, 32, 32},     {&glm, 32768, 4, 8, 32, 32},
      {&glm, 32768, 8, 8, 64, 32},     {&glm, 262144, 1, 8, 32, 32},    {&glm, 262144, 2, 8, 32, 32},
      {&glm, 262144, 4, 8, 32, 32},    {&glm, 262144, 8, 8, 64, 32},    {&glm, 1048576, 1, 8, 32, 32},
      {&glm, 1048576, 2, 8, 32, 32},   {&glm, 1048576, 4, 8, 32, 32},   {&glm, 1048576, 8, 8, 64, 32},
  };
  for (const Row& row : rows) {
    const Model& model = *row.model;
    const int64_t blocks = (row.context + kBlock - 1) / kBlock;
    const std::string where = std::string(model.label) + " S " + std::to_string(row.context) + " B " +
                              std::to_string(row.batch);
    const tqt::FusedDecodeGrid adaptive =
        tqt::PlanFusedDecode(row.batch, model.heads, 1, model.head_size, blocks, kBlock, kAiv);
    EXPECT_EQ(adaptive.num_splits, row.splits) << where;
    EXPECT_EQ(adaptive.num_tasks, row.tasks) << where;
    EXPECT_EQ(adaptive.block_dim, row.blocks) << where;
    EXPECT_EQ(static_cast<int64_t>(adaptive.block_dim), std::min(adaptive.num_tasks, kMixBlocks))
        << where << ": the grid leaves MIX blocks idle";
    const bool reduces = tqt::DecodeNeedsReduction(adaptive.num_splits, row.context, adaptive.fused_context_limit);
    EXPECT_EQ(reduces, row.splits > 1) << where;
    if (row.context >= tqt::kBandwidthSplitContext) {
      EXPECT_EQ(adaptive.num_splits, static_cast<int64_t>(tqt::kMaxSequenceSplits)) << where;
    }
    if (!reduces) {
      EXPECT_EQ(adaptive.workspace_floats, 0u) << where;
      EXPECT_EQ(adaptive.reduce_tasks_per_block, 0u) << where;
    }

    // On every bench row the adaptive grid is the fill policy's; only the launch limit differs, and a uniform
    // batch beyond it reduces either way.
    const tqt::FusedDecodeGrid fill = tqt::PlanFusedDecode(row.batch, model.heads, 1, model.head_size, blocks, kBlock,
                                                           kAiv, tqt::kFusedContextLimit,
                                                           tqt::FusedSplitPolicy::kFillBlocks);
    EXPECT_EQ(adaptive.num_splits, fill.num_splits) << where;
    EXPECT_EQ(adaptive.num_tasks, fill.num_tasks) << where;
    EXPECT_EQ(adaptive.heads_per_task, fill.heads_per_task) << where;
    EXPECT_EQ(adaptive.tasks_per_block, fill.tasks_per_block) << where;
    EXPECT_EQ(adaptive.block_dim, fill.block_dim) << where;
    EXPECT_EQ(tqt::DecodeNeedsReduction(fill.num_splits, row.context, fill.fused_context_limit), reduces) << where;
  }
}

// Where the tiers meet (block 128): the latency tier up to 8064 tokens, 4 splits at 8192, 8 beyond. A literal
// ceil(S / 2048) would plan 6 splits at 12288, which puts 48 tasks on 24 of the 32 blocks.
TEST(TurboQuantTiling, FusedDecodeAdaptiveCrossesIntoTheBandwidthTier) {
  constexpr int64_t kAiv = 64;
  constexpr int64_t kBlock = 128;
  struct Row {
    int64_t heads;
    int64_t head_size;
    int64_t context;
    int64_t batch;
    int64_t splits;
    uint32_t blocks;
  };
  const Row rows[] = {
      {16, 256, 8064, 4, 1, 32}, {16, 256, 8192, 4, 4, 32},  {16, 256, 8320, 4, 8, 32},
      {16, 256, 8192, 1, 4, 32}, {8, 128, 8064, 8, 1, 32},   {8, 128, 8192, 8, 4, 32},
      {4, 128, 8064, 8, 2, 32},  {4, 128, 8192, 8, 4, 32},   {4, 128, 12288, 4, 8, 32},
  };
  for (const Row& row : rows) {
    const int64_t blocks = (row.context + kBlock - 1) / kBlock;
    const std::string where = "H_Q " + std::to_string(row.heads) + " S " + std::to_string(row.context) + " B " +
                              std::to_string(row.batch);
    const tqt::FusedDecodeGrid grid =
        tqt::PlanFusedDecode(row.batch, row.heads, 1, row.head_size, blocks, kBlock, kAiv);
    EXPECT_EQ(grid.num_splits, row.splits) << where;
    EXPECT_EQ(grid.block_dim, row.blocks) << where;
  }
}

// Over every context a sequence can have, the adaptive K never falls as the context grows, and a power-of-two
// batch of power-of-two groups never leaves a MIX block idle in either tier.
TEST(TurboQuantTiling, FusedDecodeAdaptiveNeverDipsOrStrandsBlocks) {
  constexpr int64_t kAiv = 64;
  constexpr int64_t kMixBlocks = kAiv / 2;
  constexpr int64_t kBlock = 128;
  constexpr int64_t kMaxBlocks = 512;
  for (const int64_t batch : {1, 2, 3, 4, 5, 6, 7, 8, 16}) {
    for (const int64_t kv_heads : {1, 2, 4, 8}) {
      for (const int64_t group : {1, 2, 4, 8, 16}) {
        int64_t previous = 1;
        for (int64_t blocks = 1; blocks <= kMaxBlocks; ++blocks) {
          const tqt::FusedDecodeGrid grid =
              tqt::PlanFusedDecode(batch, kv_heads * group, kv_heads, kHeadSize, blocks, kBlock, kAiv);
          const std::string where = "B " + std::to_string(batch) + " kv_heads " + std::to_string(kv_heads) +
                                    " group " + std::to_string(group) + " S " + std::to_string(blocks * kBlock);
          ASSERT_GE(grid.num_splits, previous) << where;
          previous = grid.num_splits;
          if (CeilPow2(batch) == batch) {
            ASSERT_EQ(static_cast<int64_t>(grid.block_dim), std::min(grid.num_tasks, kMixBlocks)) << where;
          }
        }
      }
    }
  }
}

// The grids test_sim_950pr_turboquant_fused executes, planned the way each case plans them (the policy its
// golden was recorded with, S <= 256, block 64), pinned. Every case sits in the latency tier, so the adaptive
// policy plans each fill-recorded grid unchanged too.
TEST(TurboQuantTiling, FusedDecodeKeepsTheCamodelCaseGrids) {
  constexpr int64_t kAiv = 64;
  constexpr int64_t kTwoBlockAiv = 4;
  constexpr int64_t kBlock = 64;
  constexpr uint32_t kLimit = tqt::kFusedContextLimit;
  constexpr tqt::FusedSplitPolicy kFill = tqt::FusedSplitPolicy::kFillBlocks;
  constexpr tqt::FusedSplitPolicy kContext = tqt::FusedSplitPolicy::kContextOnly;
  struct Case {
    const char* tag;
    int64_t heads;
    int64_t kv_heads;
    int64_t head_size;
    int64_t context;
    int64_t aiv;
    tqt::FusedSplitPolicy policy;
    int64_t splits;
    int64_t tasks;
    uint32_t heads_per_task;
    uint32_t tasks_per_block;
    uint32_t blocks;
    uint32_t reduce_tasks_per_block;
    uint32_t prologue_vectors_per_block;
    uint32_t limit;
  };
  const Case cases[] = {
      {"(a)", 4, 2, 256, 64, kAiv, kFill, 1, 2, 2, 1, 2, 0, 2, kLimit},
      {"(b)", 4, 2, 256, 256, kAiv, kContext, 1, 2, 2, 1, 2, 0, 2, kLimit},
      {"(c)", 4, 2, 256, 256, kAiv, kFill, 4, 8, 2, 1, 8, 1, 1, 0},
      {"(d)", 8, 2, 256, 120, kTwoBlockAiv, kFill, 1, 2, 4, 1, 2, 0, 4, kLimit},
      {"(e)", 4, 2, 256, 64, kAiv, kFill, 1, 2, 2, 1, 2, 0, 2, kLimit},
      {"(f)", 4, 1, 128, 64, kAiv, kFill, 1, 2, 2, 1, 2, 0, 2, kLimit},
      {"(g)", 4, 2, 256, 256, kAiv, kFill, 4, 8, 2, 1, 8, 1, 1, 0},
  };
  for (const Case& c : cases) {
    const int64_t blocks = (c.context + kBlock - 1) / kBlock;
    const std::string where = std::string("case ") + c.tag;
    const tqt::FusedDecodeGrid recorded = tqt::PlanFusedDecode(1, c.heads, c.kv_heads, c.head_size, blocks, kBlock,
                                                               c.aiv, tqt::kFusedContextLimit, c.policy);
    EXPECT_EQ(recorded.num_splits, c.splits) << where;
    EXPECT_EQ(recorded.num_tasks, c.tasks) << where;
    EXPECT_EQ(recorded.heads_per_task, c.heads_per_task) << where;
    EXPECT_EQ(recorded.tasks_per_block, c.tasks_per_block) << where;
    EXPECT_EQ(recorded.block_dim, c.blocks) << where;
    EXPECT_EQ(recorded.reduce_tasks_per_block, c.reduce_tasks_per_block) << where;
    EXPECT_EQ(recorded.prologue_vectors_per_block, c.prologue_vectors_per_block) << where;
    EXPECT_EQ(recorded.fused_context_limit, c.limit) << where;
    if (c.policy != kFill) {
      continue;
    }
    const tqt::FusedDecodeGrid adaptive =
        tqt::PlanFusedDecode(1, c.heads, c.kv_heads, c.head_size, blocks, kBlock, c.aiv);
    EXPECT_EQ(adaptive.num_splits, recorded.num_splits) << where;
    EXPECT_EQ(adaptive.num_tasks, recorded.num_tasks) << where;
    EXPECT_EQ(adaptive.heads_per_task, recorded.heads_per_task) << where;
    EXPECT_EQ(adaptive.block_dim, recorded.block_dim) << where;
    EXPECT_EQ(adaptive.tasks_per_block, recorded.tasks_per_block) << where;
    EXPECT_EQ(adaptive.fused_context_limit, recorded.fused_context_limit) << where;
  }
}

// The raw-query prologue (TURBOQUANT_TESTS.md 13.36). Below kQueryBasisMinCubeVectors query vectors every share
// rotates on the vector cores, so a B = 1, H_Q = 8 decode never pads an Mmad. From there each block's share is a
// whole number of Cube chunks, except the last one's remainder, and a chunk is one H16 staging that the two
// subcores split on whole 16-row fractals inside the kernel's scratch.
TEST(TurboQuantTiling, FusedDecodePrologueRoutesWholeChunksToTheCube) {
  constexpr int64_t kBlock = 64;
  constexpr int64_t kBlocks = 2;
  for (const int64_t aiv : kVectorCores) {
    for (const int64_t tokens : {1, 2, 3, 4, 8, 64}) {
      for (const int64_t heads : {4, 8, 16, 32}) {
        for (const int64_t head_size : {64, 128, 256}) {
          const int64_t kv_heads = std::max<int64_t>(1, heads / 4);
          const tqt::FusedDecodeGrid grid =
              tqt::PlanFusedDecode(tokens, heads, kv_heads, head_size, kBlocks, kBlock, aiv);
          const int64_t vectors = tokens * heads;
          const int64_t chunk = grid.prologue_cube_chunk_vectors;
          const std::string where = "aiv " + std::to_string(aiv) + " tokens " + std::to_string(tokens) +
                                    " heads " + std::to_string(heads) + " head_size " + std::to_string(head_size);
          ASSERT_GE(static_cast<int64_t>(grid.block_dim) * grid.prologue_vectors_per_block, vectors) << where;
          EXPECT_EQ(chunk, tqt::QueryBasisCubeChunk(vectors, head_size)) << where;
          if (vectors < static_cast<int64_t>(tqt::kQueryBasisMinCubeVectors)) {
            EXPECT_EQ(chunk, 0) << where << ": too few vectors fill a Cube chunk";
            EXPECT_EQ(grid.prologue_vectors_per_block, static_cast<uint32_t>((vectors + grid.block_dim - 1) /
                                                                             grid.block_dim))
                << where;
            continue;
          }
          ASSERT_GT(chunk, 0) << where;
          EXPECT_EQ(chunk % 2, 0) << where << ": the dual-destination Fixpipe splits a chunk in half";
          EXPECT_LE(chunk / 2, static_cast<int64_t>(tqt::kCubeTileM / 2)) << where << ": half a chunk is kernel scratch";
          EXPECT_LE(chunk * head_size, static_cast<int64_t>(tqt::kRotateQMaxChunkElements)) << where;
          const int64_t rows = chunk * head_size / tqt::kRotateQTile;
          EXPECT_EQ((rows / 2) % tqt::kRotateQTile, 0) << where << ": each half has to fill whole 16-row fractals";
          EXPECT_EQ(grid.prologue_vectors_per_block % chunk, 0) << where << ": a block share splits a chunk";
        }
      }
    }
  }

  // Fused case (h): three tokens of H_Q 8 / H_KV 2, D 256, S 128 on 64 vector cores, unsplit and filled.
  constexpr int64_t kAiv = 64;
  const tqt::FusedDecodeGrid unsplit = tqt::PlanFusedDecode(3, 8, 2, 256, kBlocks, kBlock, kAiv, tqt::kFusedContextLimit,
                                                            tqt::FusedSplitPolicy::kContextOnly);
  EXPECT_EQ(unsplit.num_splits, 1);
  EXPECT_EQ(unsplit.block_dim, 12u);
  EXPECT_EQ(unsplit.prologue_vectors_per_block, 16u);
  EXPECT_EQ(unsplit.prologue_cube_chunk_vectors, 16u);
  const tqt::FusedDecodeGrid filled = tqt::PlanFusedDecode(3, 8, 2, 256, kBlocks, kBlock, kAiv, tqt::kFusedContextLimit,
                                                           tqt::FusedSplitPolicy::kFillBlocks);
  EXPECT_EQ(filled.num_splits, 2);
  EXPECT_EQ(filled.block_dim, 24u);
  EXPECT_EQ(filled.fused_context_limit, 0u);
  EXPECT_EQ(filled.prologue_vectors_per_block, 16u);
  EXPECT_EQ(filled.prologue_cube_chunk_vectors, 16u);
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
