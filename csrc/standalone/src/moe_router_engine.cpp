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

// MoeRouterEngine: scoring, selection and dropless dispatch. Extracted from
// Dsv4Pipeline; the pipeline calls ScoreAndSelect before the expert swap and
// Dispatch after the GEMM weights are bound.

#include "moe/pipeline/moe_router_engine.hpp"

#include <aclnn/acl_meta.h>

#include <cstring>

#include "moe/core/error.hpp"
#include "moe/pipeline/pipeline.hpp"  // slot:: IR index constants

namespace ascend_moe {

MoeRouterEngine::MoeRouterEngine(IDeviceAllocator& allocator, IStreamEngine& streams)
    : allocator_(allocator), streams_(streams) {
  readback_stream_ = streams_.CreateStream();
  routing_mailbox_ = static_cast<int32_t*>(allocator_.HostPinnedMalloc(Int32Bytes(kNumExpertsPerTok)));
  local_index_mailbox_ = static_cast<int32_t*>(allocator_.HostPinnedMalloc(Int32Bytes(kNumExpertsPerTok)));
  std::memset(routing_mailbox_, 0, Int32Bytes(kNumExpertsPerTok));
  std::memset(local_index_mailbox_, 0, Int32Bytes(kNumExpertsPerTok));
}

MoeRouterEngine::~MoeRouterEngine() {
  if (routing_mailbox_ != nullptr) {
    allocator_.HostPinnedFree(routing_mailbox_);
  }
  if (local_index_mailbox_ != nullptr) {
    allocator_.HostPinnedFree(local_index_mailbox_);
  }
  if (readback_stream_ != nullptr) {
    streams_.DestroyStream(readback_stream_);
  }
}

PipelineStage& MoeRouterEngine::stage(const char* name) {
  for (size_t i = 0; i < stage_count_; ++i) {
    PipelineStage& entry = stages_[i];
    if (std::strcmp(entry.name, name) == 0) {
      return entry;
    }
  }
  throw Dsv4Error(std::string("no router stage named ") + name);
}

const PipelineStage& MoeRouterEngine::stage(const char* name) const {
  for (size_t i = 0; i < stage_count_; ++i) {
    const PipelineStage& entry = stages_[i];
    if (std::strcmp(entry.name, name) == 0) {
      return entry;
    }
  }
  throw Dsv4Error(std::string("no router stage named ") + name);
}

void MoeRouterEngine::Launch(PipelineStage& entry, const OpTable& ops, const StaticArenaManager& arena_manager,
                             DeviceStream compute_stream) {
  entry.slot.Launch(ops, arena_manager.arena().workspace(), compute_stream);
  ++launches_;
}

void MoeRouterEngine::PlanStages(const OpTable& ops, StaticArenaManager& arena_manager,
                                 const RuntimeConfig& config) {
  ArenaTensors& t = arena_manager.tensors();
  StaticMemoryArena& arena = arena_manager.arena();

  // Every operator the router drives must exist before anything is planned.
  ops.RequireAll({OpId::kMatmul, OpId::kSoftplus, OpId::kSqrt, OpId::kMoeGatingTopKV2, OpId::kMoeInitRoutingV4});

  auto add = [&](const char* name, OpId op) -> PipelineStage& {
    DSV4_REQUIRE(stage_count_ < kMaxRouterStages, "router stage capacity exceeded at " << name);
    PipelineStage& entry = stages_[stage_count_++];
    entry.name = name;
    entry.op = op;
    return entry;
  };
  auto adopt = [&](PipelineStage& entry, uint64_t workspace, aclOpExecutor* executor) {
    arena.NoteWorkspace(workspace);
    entry.slot.Adopt(entry.op, entry.name, workspace, executor);
  };

  aclOpExecutor* executor = nullptr;

  // 1. router logits over the post-attention normed hidden state.
  {
    PipelineStage& entry = add("router", OpId::kMatmul);
    const uint64_t workspace = PlanAclnnOp<MatmulPlanFn>(ops, entry.op, &executor, t.normed, t.w_router,
                                                         t.router_logits, kCubeMathTypeKeepDtype);
    adopt(entry, workspace, executor);
  }
  // 2-3. the DSV4 sqrtsoftplus scoring, decomposed onto the same stream.
  //     The stock gating operator's normType only offers softmax / sigmoid
  //     (the older aclnnMoeGatingTopK's two modes), so scores =
  //     sqrt(softplus(logits)) is computed by two elementwise stages and the
  //     gating stage below receives pre-normalized scores.
  {
    PipelineStage& entry = add("router_softplus", OpId::kSoftplus);
    const uint64_t workspace =
        PlanAclnnOp<SoftplusPlanFn>(ops, entry.op, &executor, t.router_logits, t.softplus_beta,
                                    t.softplus_threshold, t.router_softplus);
    adopt(entry, workspace, executor);
  }
  {
    PipelineStage& entry = add("router_sqrt", OpId::kSqrt);
    const uint64_t workspace =
        PlanAclnnOp<UnaryPlanFn>(ops, entry.op, &executor, t.router_softplus, t.router_scores);
    adopt(entry, workspace, executor);
  }
  // 4. noaux_tc gating over the pre-normalized scores: bias shifts selection
  //    only, renorm=1 L1-renormalizes the top-6 scores to sum 1.0 (eps-
  //    guarded) and scaling 1.5 is applied here.
  {
    PipelineStage& entry = add("gating", OpId::kMoeGatingTopKV2);
    const uint64_t workspace = PlanAclnnOp<MoeGatingTopKV2PlanFn>(
        ops, entry.op, &executor, t.router_scores, t.w_router_bias, nullptr, nullptr, kNumExpertsPerTok,
        kGatingKGroup, kGatingGroupCount, kGatingGroupSelectMode, kGatingRenormL1, config.gating_norm_type, false,
        kRoutedScalingFactor, kGatingEps, t.gating_weights, t.gating_indices, nullptr);
    adopt(entry, workspace, executor);
  }
  // 5. dropless dispatch over the SIX locally renumbered experts. The expert
  //    ids fed in are 0..5, not the global 0..255: the weight list has six
  //    entries, so the device cumsum must have six groups to match it.
  {
    PipelineStage& entry = add("routing", OpId::kMoeInitRoutingV4);
    const uint64_t workspace = PlanAclnnOp<MoeInitRoutingV4PlanFn>(
        ops, entry.op, &executor, t.normed_fp8, t.local_indices, t.normed_mx_scale, nullptr, nullptr,
        t.gating_weights, 0, kNumExpertsPerTok, kRoutingDropless, kRoutingTokensNumCumsum, true,
        kRoutingQuantModeNone, nullptr, kRoutingRowIdxGather, t.expanded_x, t.expanded_row_idx, t.group_list,
        t.expanded_scale, t.expanded_weights);
    adopt(entry, workspace, executor);
  }
}

const int32_t* MoeRouterEngine::ScoreAndSelect(int32_t layer, StaticArenaManager& arena_manager,
                                               const OpTable& ops, const BackboneWeights::Layer& layer_weights,
                                               DeviceStream compute_stream, DeviceEvent compute_done) {
  (void)layer;
  const ArenaTensors& t = arena_manager.tensors();
  const StaticMemoryArena& arena = arena_manager.arena();

  // Repoint this layer's router weights. The descriptors and the executors are
  // the ones planned at init; only the addresses move.
  stage("router").slot.SetAddress(slot::kMatmulMat2, t.w_router, arena.Address(layer_weights.router_weight));
  stage("gating").slot.SetAddress(slot::kGatingBias, t.w_router_bias, arena.Address(layer_weights.router_bias));

  Launch(stage("router"), ops, arena_manager, compute_stream);
  Launch(stage("router_softplus"), ops, arena_manager, compute_stream);
  Launch(stage("router_sqrt"), ops, arena_manager, compute_stream);
  Launch(stage("gating"), ops, arena_manager, compute_stream);

  // THE ONE FORCED HOST ROUND TRIP per MoE layer: 24 bytes on a dedicated
  // stream, ordered after the gating launch by the compute event.
  streams_.RecordEvent(compute_done, compute_stream);
  streams_.StreamWaitEvent(readback_stream_, compute_done);
  streams_.MemcpyAsync(routing_mailbox_, Int32Bytes(kNumExpertsPerTok), arena.Address(t.h_gating_indices),
                       Int32Bytes(kNumExpertsPerTok), MemcpyKind::kDeviceToHost, readback_stream_);
  streams_.SynchronizeStream(readback_stream_);
  ++readbacks_;
  return routing_mailbox_;
}

void MoeRouterEngine::Dispatch(StaticArenaManager& arena_manager, const OpTable& ops, DeviceStream compute_stream) {
  const ArenaTensors& t = arena_manager.tensors();
  // Renumber the global expert ids to 0..5 so the dispatch's cumsum has exactly
  // as many groups as the weight list has entries. The renumbering is free --
  // the host already holds the global ids, because the exclusive swap engine
  // needed them to decide what to promote.
  for (int64_t index = 0; index < kNumExpertsPerTok; ++index) {
    local_index_mailbox_[index] = static_cast<int32_t>(index);
  }
  streams_.MemcpyAsync(arena_manager.arena().Address(t.h_local_indices), Int32Bytes(kNumExpertsPerTok),
                       local_index_mailbox_, Int32Bytes(kNumExpertsPerTok), MemcpyKind::kHostToDevice,
                       compute_stream);
  Launch(stage("routing"), ops, arena_manager, compute_stream);
}

}  // namespace ascend_moe
