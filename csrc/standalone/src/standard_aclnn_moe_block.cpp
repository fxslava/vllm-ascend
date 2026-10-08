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

// The stock ACLNN V5 expert path: the stages Dsv4Pipeline used to plan and
// launch inline, now behind IRoutedMoeBlock. Stage names, op selection and
// plan arguments are carried over unchanged, so the planned graph is
// byte-identical to the pre-abstraction build.

#include "moe/pipeline/standard_aclnn_moe_block.hpp"

#include <aclnn/acl_meta.h>

#include "moe/core/error.hpp"
#include "moe/pipeline/pipeline.hpp"  // slot:: IR index constants

namespace ascend_moe {
namespace {

// `aclnnSwigluMxQuant` round mode (decomposed path only). FP8 destinations
// support "rint" only (artifacts/cann92_audit/REPORT.md section 0).
// Non-const because the aclnn signature takes `char*`.
char kMxRoundModeRint[] = "rint";

}  // namespace

StandardAclnnMoeBlock::StandardAclnnMoeBlock(const OpTable& ops, MoeRouterEngine& router,
                                             ExclusiveExpertManager& experts, const RuntimeConfig& config)
    : ops_(ops), router_(router), experts_(experts), config_(config) {}

void StandardAclnnMoeBlock::PlanStages(StaticOpSlotTable& op_table, StaticArenaManager& arena,
                                       const ExpertSlotAddresses& slot_addrs) {
  (void)slot_addrs;  // the descriptors already carry these addresses; kept in
                     // the signature because a backend that builds its own
                     // descriptors from slot geometry needs them (lattice).
  arena_ = &arena;
  table_ = &op_table;
  ArenaTensors& t = arena.tensors();
  StaticMemoryArena& arena_memory = arena.arena();

  // Every operator this backend drives must exist before anything is
  // planned, so a toolkit gap is one clear message rather than a failure
  // part-way through the graph.
  if (config_.moe_path == MoePath::kFused) {
    ops_.RequireAll({OpId::kGroupedMatmulV5, OpId::kGroupedMatmulSwigluQuantV2});
  } else {
    ops_.RequireAll({OpId::kGroupedMatmulV5, OpId::kSwigluMxQuant});
  }

  aclOpExecutor* executor = nullptr;
  const auto adopt = [&](PipelineStage& entry, uint64_t workspace, aclOpExecutor* planned) {
    arena_memory.NoteWorkspace(workspace);
    entry.slot.Adopt(entry.op, entry.name, workspace, planned);
  };

  // Expert GEMM 1. Fused path: GEMM + clamped SwiGLU (limit 10.0) + MX
  // requant in one op.
  if (config_.moe_path == MoePath::kFused) {
    PipelineStage& entry = op_table.Add("expert_gemm1", OpId::kGroupedMatmulSwigluQuantV2);
    const uint64_t workspace = PlanAclnnOp<GroupedMatmulSwigluQuantV2PlanFn>(
        ops_, entry.op, &executor, t.expanded_x, t.expert_gate_up_list, t.expert_gate_up_scale_list, nullptr,
        nullptr, t.expanded_scale, nullptr, t.group_list, kGmmDequantModeMx, static_cast<int64_t>(kAclFloat8E4m3Fn),
        kGmmDequantModeMx, kGmmGroupListTypeCumsum, nullptr, t.gemm1_out, t.gemm1_scale);
    adopt(entry, workspace, executor);
  } else {
    PipelineStage& entry = op_table.Add("expert_gemm1", OpId::kGroupedMatmulV5);
    const uint64_t workspace = PlanAclnnOp<GroupedMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.gemm1_x_list, t.expert_gate_up_list, nullptr, t.expert_gate_up_scale_list,
        nullptr, nullptr, nullptr, t.gemm1_x_scale_list, t.group_list, nullptr, nullptr, nullptr,
        kGmmSplitItemSingleOut, kGmmGroupTypeM, kGmmGroupListTypeCumsum, kGmmActTypeNone, nullptr,
        t.gemm1_raw_out_list, nullptr, nullptr);
    adopt(entry, workspace, executor);

    PipelineStage& activation = op_table.Add("expert_swiglu", OpId::kSwigluMxQuant);
    const uint64_t act_workspace = PlanAclnnOp<SwigluMxQuantPlanFn>(
        ops_, activation.op, &executor, t.gemm1_raw, nullptr, kSwigluActivateDimLast, true, kSwigluModeDefault,
        kSwigluLimit, kSwigluGluAlpha, kSwigluGluBias, kSwigluGroupModeNone, kSwigluAxisLast,
        static_cast<int64_t>(kAclFloat8E4m3Fn), kMxRoundModeRint, kSwigluScaleAlgOcp, kSwigluMaxDtypeValue,
        t.gemm1_out, t.gemm1_scale);
    adopt(activation, act_workspace, executor);
  }
  // Expert GEMM 2. A tensor-LIST weight, which is what lets the six experts
  // sit in six scattered HBM slots. `aclnnGroupedMatmulFinalizeRoutingV3`
  // would fuse the combine, but its x2 is a single tensor, i.e. the experts
  // must be contiguous -- which an exclusive LRU slot pool cannot promise.
  {
    PipelineStage& entry = op_table.Add("expert_gemm2", OpId::kGroupedMatmulV5);
    const uint64_t workspace = PlanAclnnOp<GroupedMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.gemm2_x_list, t.expert_down_list, nullptr, t.expert_down_scale_list, nullptr,
        nullptr, nullptr, t.gemm2_x_scale_list, t.group_list, nullptr, nullptr, nullptr, kGmmSplitItemSingleOut,
        kGmmGroupTypeM, kGmmGroupListTypeCumsum, kGmmActTypeNone, nullptr, t.gemm2_out_list, nullptr, nullptr);
    adopt(entry, workspace, executor);
  }
  // Routing combine. For a single-token step the six expanded rows are the
  // same token, so the weighted sum IS a [1, 6] x [6, hidden] matmul over
  // the permuted routing weights the dispatch emitted. Exact, and both ops
  // have an ascend950 kernel. This is the one stage that does not generalize
  // past token_count == 1: a batched decode needs a scatter-add by
  // `expanded_row_idx`.
  {
    PipelineStage& entry = op_table.Add("expert_combine", OpId::kMatmul);
    const uint64_t workspace =
        PlanAclnnOp<MatmulPlanFn>(ops_, entry.op, &executor, t.expanded_weights_row, t.gemm2_out, t.routed_out,
                                  kCubeMathTypeKeepDtype);
    adopt(entry, workspace, executor);
  }
}

void StandardAclnnMoeBlock::ExecuteMoe(const MoeDispatchContext& ctx, IStreamEngine& stream_engine) {
  // Single-stream by construction: every launch lands on ctx.compute_stream,
  // already ordered after the dispatch and the slot exchange. A backend that
  // overlaps a multi-stream unpack (lattice) is the reason the engine is a
  // parameter at all.
  (void)stream_engine;
  DSV4_REQUIRE(arena_ != nullptr, "StandardAclnnMoeBlock::ExecuteMoe before PlanStages");
  DSV4_REQUIRE(ctx.swap_plan != nullptr, "a MoE dispatch context without the layer's residency plan");
  DSV4_REQUIRE(ctx.top_k == ctx.swap_plan->count,
               "the dispatch context names " << ctx.top_k << " experts but the residency plan moved "
                                             << ctx.swap_plan->count);
  DSV4_REQUIRE(ctx.token_count == 1,
               "the standard block's combine is shaped for single-token steps, got token_count "
                   << ctx.token_count);

  // The dropless dispatch over the SIX locally renumbered experts (the
  // weight list has six entries, so the device cumsum needs six groups).
  router_.Dispatch(*arena_, ops_, ctx.compute_stream);

  BindExpertWeights(*ctx.swap_plan);

  Launch(table_->stage("expert_gemm1"), ctx.compute_stream);
  if (config_.moe_path == MoePath::kDecomposed) {
    Launch(table_->stage("expert_swiglu"), ctx.compute_stream);
  }
  Launch(table_->stage("expert_gemm2"), ctx.compute_stream);
  Launch(table_->stage("expert_combine"), ctx.compute_stream);
}

void StandardAclnnMoeBlock::BindExpertWeights(const LayerSwapPlan& plan) {
  ArenaTensors& t = arena_->tensors();
  PipelineStage& gemm1 = table_->stage("expert_gemm1");
  PipelineStage& gemm2 = table_->stage("expert_gemm2");
  const size_t gemm1_scale_index =
      config_.moe_path == MoePath::kFused ? slot::kGmmSwigluScaleList : slot::kGmmV5ScaleList;
  for (int32_t index = 0; index < plan.count; ++index) {
    const int32_t slot_id = plan.device_slots[index];
    void* gate_up = experts_.RegionAddress(slot_id, ExpertRegionId::kGateUpWeight);
    void* gate_up_scale = experts_.RegionAddress(slot_id, ExpertRegionId::kGateUpScale);
    void* down = experts_.RegionAddress(slot_id, ExpertRegionId::kDownWeight);
    void* down_scale = experts_.RegionAddress(slot_id, ExpertRegionId::kDownScale);
    const size_t relative = static_cast<size_t>(index);
    gemm1.slot.SetTensorListAddress(slot::kGmmWeightList, relative, t.expert_gate_up_list, gate_up);
    gemm1.slot.SetTensorListAddress(gemm1_scale_index, relative, t.expert_gate_up_scale_list, gate_up_scale);
    gemm2.slot.SetTensorListAddress(slot::kGmmWeightList, relative, t.expert_down_list, down);
    gemm2.slot.SetTensorListAddress(slot::kGmmV5ScaleList, relative, t.expert_down_scale_list, down_scale);
  }
}

size_t StandardAclnnMoeBlock::GetExpertSlotBytes() const { return experts_.layout().slot_num_bytes(); }

void StandardAclnnMoeBlock::Launch(PipelineStage& entry, DeviceStream stream) {
  entry.slot.Launch(ops_, arena_->arena().workspace(), stream);
  ++launches_;
}

std::string StandardAclnnMoeBlock::DescribeBackend() const {
  return config_.moe_path == MoePath::kFused
             ? "standard aclnn: GroupedMatmulSwigluQuantV2 + GroupedMatmulV5 (fused)"
             : "standard aclnn: GroupedMatmulV5 -> SwigluMxQuant -> GroupedMatmulV5 (decomposed)";
}

}  // namespace ascend_moe
