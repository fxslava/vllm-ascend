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

// IRoutedMoeBlock: the expert-execution seam of the decode graph (DIP).
//
// Dsv4Pipeline used to plan and launch the stock grouped-GEMM chain
// (GroupedMatmulSwigluQuantV2 / GroupedMatmulV5) directly, which welded the
// orchestrator to one operator family and one slot byte contract. Behind
// this interface the expert half is a swappable block:
//
//   StandardAclnnMoeBlock  the stock ACLNN V5 path -- GMM V5 with FP4/E8M0
//                          slots, fused or decomposed activation; today's
//                          behavior, byte for byte.
//   Lattice24MoeBlock      skeleton for 2-bit Leech-lattice (Lambda_24)
//                          packed weights, unpacked on the AIV decode
//                          stream by a custom kernel; ~50% of the slot.
//
// The contract, in three moves:
//
//   PlanStages       once, at build time: plan the expert stages into the
//                    pipeline's StaticOpSlotTable against the arena's
//                    descriptors and the six resident slot addresses. Same
//                    static-runtime rule as every other stage -- planned
//                    once, replayed 43 times with address swaps only.
//   ExecuteMoe       once per MoE layer: dispatch, weight binding, the
//                    expert GEMMs, the routing combine. Enqueues only; the
//                    block never synchronizes (the one forced D2H per layer
//                    belongs to MoeRouterEngine::ScoreAndSelect, before the
//                    block runs).
//   GetExpertSlotBytes  the slot byte contract this backend imposes on the
//                    residency planner and the exclusive hierarchy.
//
// What a block does NOT own: the router (scoring/selection stays upstream
// -- every backend consumes the same dispatched tokens), the shared expert
// (it is dense and never routed), the residuals, and the head. The pipeline
// keeps all of those.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "moe/core/stream_engine.hpp"
#include "moe/memory/exclusive_staging.hpp"
#include "moe/pipeline/static_arena_manager.hpp"
#include "moe/pipeline/static_op_slot_table.hpp"

// Forward declaration matching aclnn/acl_meta.h; the real type comes with
// StaticArenaManager's includes in translation units that need it.
typedef struct aclTensor aclTensor;

namespace ascend_moe {

// Everything the block needs for ONE MoE layer, handed in per call so a
// block instance stays stateless across layers (the 43 layers are
// identical; only addresses move).
struct MoeDispatchContext {
  // ---- which layer, how many tokens ----
  int32_t layer = -1;       // 0..num_hidden_layers-1
  int64_t token_count = 0;  // tokens dispatched this step (1 for this decode graph)
  int64_t top_k = 0;        // experts per token == weight-list length

  // ---- the dispatch outputs (written by MoeRouterEngine::Dispatch) ----
  aclTensor* expanded_x = nullptr;       // [top_k*token_count, hidden] FP8
  aclTensor* expanded_scale = nullptr;   // [top_k*token_count, hidden/32] E8M0
  aclTensor* group_list = nullptr;       // [top_k*token_count] INT64 group cumsum

  // ---- the expert compute buffers the block fills ----
  aclTensor* gemm1_out = nullptr;        // [top_k*token_count, intermediate] FP8
  aclTensor* gemm1_scale = nullptr;      // ... its E8M0 microscales
  aclTensor* gemm2_out = nullptr;        // [top_k*token_count, hidden] BF16

  // ---- the combine: weighted sum of the six expanded rows ----
  aclTensor* expanded_weights_row = nullptr;  // [token_count, top_k] FP32 routing weights
  aclTensor* routed_out = nullptr;            // [token_count, hidden] BF16 output buffer

  // ---- residency: which device slots hold this layer's experts ----
  const LayerSwapPlan* swap_plan = nullptr;

  // ---- ordering: the pipeline's compute stream, already ordered after the
  //      dispatch by the arena's stream contract ----
  DeviceStream compute_stream = nullptr;
};

class IRoutedMoeBlock {
 public:
  virtual ~IRoutedMoeBlock() = default;

  // Plans the expert stages into the pipeline's table, once, at build time.
  // `slot_addrs` holds the six resident experts' region addresses at
  // descriptor time (slots 0..5 of the exclusive hierarchy).
  virtual void PlanStages(StaticOpSlotTable& op_table, StaticArenaManager& arena,
                          const ExpertSlotAddresses& slot_addrs) = 0;

  // Executes one MoE layer: dispatch, weight binding, expert GEMMs, combine.
  // Enqueues only -- launches land on `ctx.compute_stream`.
  virtual void ExecuteMoe(const MoeDispatchContext& ctx, IStreamEngine& stream_engine) = 0;

  // The slot byte contract this backend imposes: one expert's packed weight
  // + scale footprint in the exclusive hierarchy. The slot planner subtracts
  // it from free HBM, the residency manager sizes every slot with it.
  virtual size_t GetExpertSlotBytes() const = 0;

  // Launches enqueued by ExecuteMoe so far. The pipeline folds the delta
  // into StepCounters::launches, so the zero-descriptor / zero-alloc step
  // accounting stays truthful no matter which backend ran.
  virtual uint64_t launches() const { return 0; }

  // One line for the reports: what runs, and at what slot cost.
  virtual std::string DescribeBackend() const = 0;
};

}  // namespace ascend_moe
