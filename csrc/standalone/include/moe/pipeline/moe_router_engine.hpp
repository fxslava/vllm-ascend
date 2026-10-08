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

// MoeRouterEngine: the DeepSeek-V4 Flash router, as one unit (SRP).
//
// The full scoring-and-dispatch chain lives here --
//
//   router matmul -> aclnnSoftplus -> aclnnSqrt -> aclnnMoeGatingTopKV2
//   -> (the one forced host readback per MoE layer)
//   -> aclnnMoeInitRoutingV4
//
// -- because the stock gating operator cannot express sqrt(softplus(logits)):
// its normType only enumerates softmax / sigmoid, so the scores are computed
// by the decomposed elementwise chain and the gating stage receives
// pre-normalized scores (normType -1, renorm 1, eps 1e-20, scaling 1.5 --
// `scoring_func: sqrtsoftplus`, `topk_method: noaux_tc` in the checkpoint's
// config.json). Mirrors SqrtSoftplusRouter in
// tools/dsv4_moe_runtime/hardware/v5_ops_moe.py.
//
// The engine owns its two pinned mailboxes and its readback stream; the
// compute stream and its boundary event belong to the pipeline, which passes
// them in per call (the event is shared with the swap engine and the next
// layer, so it cannot have a single owner here).

#pragma once

#include <cstdint>
#include <vector>

#include "moe/core/config.hpp"
#include "moe/core/device_allocator.hpp"
#include "moe/core/device_types.hpp"
#include "moe/core/op_table.hpp"
#include "moe/core/stream_engine.hpp"
#include "moe/pipeline/static_arena_manager.hpp"

namespace ascend_moe {

class MoeRouterEngine {
 public:
  MoeRouterEngine(IDeviceAllocator& allocator, IStreamEngine& streams);
  ~MoeRouterEngine();

  MoeRouterEngine(const MoeRouterEngine&) = delete;
  MoeRouterEngine& operator=(const MoeRouterEngine&) = delete;

  // Plans the five router stages against the arena's descriptors: the logits
  // matmul, the decomposed sqrtsoftplus scoring, the noaux_tc gating and the
  // dropless dispatch. Requires the arena committed.
  void PlanStages(const OpTable& ops, StaticArenaManager& arena_manager, const RuntimeConfig& config);

  // Launches logits -> softplus -> sqrt -> gating on `compute_stream`, then
  // performs THE ONE FORCED HOST ROUND TRIP per MoE layer (the 24-byte top-6
  // readback on a dedicated stream, ordered after the gating launch by the
  // compute event). Returns the global expert ids.
  const int32_t* ScoreAndSelect(int32_t layer, StaticArenaManager& arena_manager, const OpTable& ops,
                                const BackboneWeights::Layer& layer_weights, DeviceStream compute_stream,
                                DeviceEvent compute_done);

  // After the exclusive manager made the six chosen experts resident and the
  // GEMM weight lists were repointed: renumber the global ids to the local
  // 0..5 (the weight list has six entries, so the device cumsum needs six
  // groups) and launch the dropless dispatch.
  void Dispatch(StaticArenaManager& arena_manager, const OpTable& ops, DeviceStream compute_stream);

  // Forced host readbacks performed so far: one per ScoreAndSelect call.
  uint64_t readbacks() const { return readbacks_; }

  const std::vector<PipelineStage>& stages() const { return stages_; }
  uint64_t launches() const { return launches_; }

  PipelineStage& stage(const char* name);
  const PipelineStage& stage(const char* name) const;
  // Launches one planned router stage against the shared workspace.
  void Launch(PipelineStage& stage_entry, const OpTable& ops, const StaticArenaManager& arena_manager,
              DeviceStream compute_stream);

 private:
  static constexpr size_t kMaxRouterStages = 8;

  IDeviceAllocator& allocator_;
  IStreamEngine& streams_;
  // Born at full capacity: StaticOpSlot is not movable.
  std::vector<PipelineStage> stages_ = std::vector<PipelineStage>(kMaxRouterStages);
  size_t stage_count_ = 0;
  uint64_t launches_ = 0;
  uint64_t readbacks_ = 0;

  DeviceStream readback_stream_ = nullptr;
  int32_t* routing_mailbox_ = nullptr;      // [top_k] global expert ids, D2H
  int32_t* local_index_mailbox_ = nullptr;  // [top_k] local 0..top_k-1, H2D
};

}  // namespace ascend_moe
