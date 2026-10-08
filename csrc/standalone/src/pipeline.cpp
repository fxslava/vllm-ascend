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

// The orchestrator's half of the graph: attention, the expert GEMMs, the
// shared expert and the head. Memory lives in StaticArenaManager, the router
// chain in MoeRouterEngine; this file sequences them across the 43 layers.

#include "moe/pipeline/pipeline.hpp"

#include <aclnn/acl_meta.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "moe/core/error.hpp"
#include "moe/memory/expert_layout.hpp"
#include "moe/pipeline/standard_aclnn_moe_block.hpp"

namespace ascend_moe {
namespace {

// This runner decodes one token at a time (greedy, batch 1). Every buffer the
// manager reserves is sized for it, and the routing combine is only correct
// for it -- see the note on the `expert_combine` stage.
constexpr int64_t kTokensPerStep = 1;

// Scale dtype for the dense block-128 weight scales and for the MX block-32
// activation scales: both are OCP E8M0, one byte per block.
constexpr int32_t kScaleDtype = kAclFloat8E8m0;

// `aclnnRmsNormDynamicMxQuant` round mode. FP8 destinations support "rint"
// only (artifacts/cann92_audit/REPORT.md section 0). Non-const because the
// aclnn signatures take `char*`.
char kMxRoundModeRint[] = "rint";
char kFiaLayout[] = "TND";
char kRotaryMode[] = "half";
char kScatterCacheMode[] = "Norm";

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Dsv4Pipeline::Dsv4Pipeline(IDeviceAllocator& allocator, IStreamEngine& streams, const OpTable& ops,
                           ExclusiveExpertManager& experts, MoeRouterEngine& router, const RuntimeConfig& config,
                           std::unique_ptr<IRoutedMoeBlock> moe_block)
    : allocator_(allocator),
      streams_(streams),
      ops_(ops),
      experts_(experts),
      config_(config),
      arena_manager_(allocator, streams, config),
      router_(router),
      moe_block_(std::move(moe_block)) {
  // The default backend: the stock ACLNN V5 expert path. An explicit
  // injection (the lattice skeleton, a bring-up fork) is taken as given.
  if (moe_block_ == nullptr) {
    moe_block_ = std::make_unique<StandardAclnnMoeBlock>(ops_, router_, experts_, config_);
  }
  compute_stream_ = streams_.CreateStream();
  compute_done_ = streams_.CreateEvent();
  // Primed for the same reason the staging engine primes its events: the first
  // layer's ScoreAndSelect waits on it before anything has recorded it.
  streams_.RecordEvent(compute_done_, compute_stream_);
  streams_.SynchronizeStream(compute_stream_);

  token_mailbox_ = static_cast<int64_t*>(allocator_.HostPinnedMalloc(Int64Bytes(1)));
  slot_mailbox_ = static_cast<int32_t*>(allocator_.HostPinnedMalloc(Int32Bytes(1)));
  *token_mailbox_ = 0;
  *slot_mailbox_ = 0;
}

Dsv4Pipeline::~Dsv4Pipeline() {
  // The stages own repeatable executors. They are released by the member
  // destruction that follows this body: `stages_` is declared after
  // `arena_manager_`, so reverse-order destruction destroys the executors
  // BEFORE the arena frees the memory their descriptors point at.
  if (token_mailbox_ != nullptr) {
    allocator_.HostPinnedFree(token_mailbox_);
  }
  if (slot_mailbox_ != nullptr) {
    allocator_.HostPinnedFree(slot_mailbox_);
  }
  if (compute_done_ != nullptr) {
    streams_.DestroyEvent(compute_done_);
  }
  if (compute_stream_ != nullptr) {
    streams_.DestroyStream(compute_stream_);
  }
}

size_t Dsv4Pipeline::BackboneDeviceBytes(const MlaGeometry& mla, int64_t block_size, int64_t max_context_len) {
  return StaticArenaManager::BackboneDeviceBytes(mla, block_size, max_context_len);
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

ExpertSlotAddresses Dsv4Pipeline::CollectExpertSlotAddresses() {
  // The six experts start in slots 0..5 so the plan phase sees real, resident
  // addresses; every layer afterwards repoints them via
  // aclSetDynamicTensorAddr. Kept in `slot_addrs_`: the MoE block's planning
  // consumes the same map.
  slot_addrs_ = ExpertSlotAddresses{};
  for (int64_t index = 0; index < kNumExpertsPerTok; ++index) {
    const int32_t slot = static_cast<int32_t>(index);
    const size_t at = static_cast<size_t>(index);
    slot_addrs_.gate_up_weight[at] = experts_.RegionAddress(slot, ExpertRegionId::kGateUpWeight);
    slot_addrs_.gate_up_scale[at] = experts_.RegionAddress(slot, ExpertRegionId::kGateUpScale);
    slot_addrs_.down_weight[at] = experts_.RegionAddress(slot, ExpertRegionId::kDownWeight);
    slot_addrs_.down_scale[at] = experts_.RegionAddress(slot, ExpertRegionId::kDownScale);
  }
  return slot_addrs_;
}

void Dsv4Pipeline::Build(WeightByteSource& source) {
  arena_manager_.ReserveActivations();
  arena_manager_.ReserveBackbone();
  arena_manager_.Commit();
  arena_manager_.IngestBackbone(source);
  arena_manager_.CreateDescriptors(experts_.layout(), CollectExpertSlotAddresses());
  router_.PlanStages(ops_, arena_manager_, config_);
  PlanStages();
  arena_manager_.CommitWorkspace();
  arena_manager_.Seal();
}

// ---------------------------------------------------------------------------
// Planning (the orchestrator's stages; the router's five live in
// MoeRouterEngine::PlanStages, the expert GEMM stages in the injected
// IRoutedMoeBlock -- all into ONE StaticOpSlotTable)
// ---------------------------------------------------------------------------

void Dsv4Pipeline::PlanStages() {
  const MlaGeometry& mla = config_.mla;
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();

  // Every required op must exist before anything is planned, so a toolkit gap
  // is one clear message rather than a failure part-way through the graph.
  // The MoE block checks its own grouped-GEMM surface in its PlanStages.
  std::vector<OpId> required = {
      OpId::kRmsNorm,          OpId::kRmsNormDynamicMxQuant, OpId::kDynamicMxQuant,
      OpId::kQuantMatmulV5,    OpId::kApplyRotaryPosEmbV2,   OpId::kScatterPaKvCache,
      OpId::kFusedInferAttentionScoreV5, OpId::kInplaceAdd,  OpId::kSwiGlu,
      OpId::kArgMax,
  };
  ops_.RequireAll(required);

  // The stage table is born at full capacity (StaticOpSlot is not movable,
  // see stages_ in the header); planning only fills slots in place.
  auto add = [&](const char* name, OpId op) -> PipelineStage& { return stages_.Add(name, op); };
  auto adopt = [&](PipelineStage& entry, uint64_t workspace, aclOpExecutor* executor) {
    arena.NoteWorkspace(workspace);
    entry.slot.Adopt(entry.op, entry.name, workspace, executor);
  };

  aclOpExecutor* executor = nullptr;

  // 1. input RMSNorm fused with the FP8 activation quantization.
  {
    PipelineStage& entry = add("input_norm_quant", OpId::kRmsNormDynamicMxQuant);
    const uint64_t workspace = PlanAclnnOp<RmsNormDynamicMxQuantPlanFn>(
        ops_, entry.op, &executor, t.hidden, t.w_input_norm, nullptr, kRmsNormEpsilon, kSwigluScaleAlgOcp,
        kMxRoundModeRint, static_cast<int64_t>(kAclFloat8E4m3Fn), false, t.normed_fp8, t.normed_mx_scale, nullptr);
    adopt(entry, workspace, executor);
  }
  // 2. q down-projection.
  {
    PipelineStage& entry = add("q_a_proj", OpId::kQuantMatmulV5);
    const uint64_t workspace = PlanAclnnOp<QuantMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.normed_fp8, t.w_q_a, t.normed_mx_scale, t.w_q_a_scale, nullptr, nullptr,
        nullptr, nullptr, nullptr, false, true, config_.dense_group_size, t.q_a);
    adopt(entry, workspace, executor);
  }
  // 3. q LoRA norm, fused quantization again.
  {
    PipelineStage& entry = add("q_a_norm_quant", OpId::kRmsNormDynamicMxQuant);
    const uint64_t workspace = PlanAclnnOp<RmsNormDynamicMxQuantPlanFn>(
        ops_, entry.op, &executor, t.q_a, t.w_q_a_norm, nullptr, kRmsNormEpsilon, kSwigluScaleAlgOcp,
        kMxRoundModeRint, static_cast<int64_t>(kAclFloat8E4m3Fn), false, t.q_a_fp8, t.q_a_mx_scale, nullptr);
    adopt(entry, workspace, executor);
  }
  // 4. q up-projection with W_UK folded in: emits [heads, kv_lora | rope], so
  //    attention happens in the compressed latent space.
  {
    PipelineStage& entry = add("q_b_proj", OpId::kQuantMatmulV5);
    const uint64_t workspace = PlanAclnnOp<QuantMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.q_a_fp8, t.w_q_b, t.q_a_mx_scale, t.w_q_b_scale, nullptr, nullptr, nullptr,
        nullptr, nullptr, false, true, config_.dense_group_size, t.q_b);
    adopt(entry, workspace, executor);
  }
  // 5. kv down-projection with MQA rope: emits [kv_lora | rope].
  {
    PipelineStage& entry = add("kv_a_proj", OpId::kQuantMatmulV5);
    const uint64_t workspace = PlanAclnnOp<QuantMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.normed_fp8, t.w_kv_a, t.normed_mx_scale, t.w_kv_a_scale, nullptr, nullptr,
        nullptr, nullptr, nullptr, false, true, config_.dense_group_size, t.kv_a);
    adopt(entry, workspace, executor);
  }
  // 6. the compressed latent's own RMSNorm (no quantization: it is cached bf16).
  {
    PipelineStage& entry = add("kv_a_norm", OpId::kRmsNorm);
    const uint64_t workspace = PlanAclnnOp<RmsNormPlanFn>(ops_, entry.op, &executor, t.kv_latent, t.w_kv_a_norm,
                                                          kRmsNormEpsilon, t.kv_latent_normed, t.rstd);
    adopt(entry, workspace, executor);
  }
  // 7. partial RoPE over the q and k rope slices, in place.
  {
    PipelineStage& entry = add("rope", OpId::kApplyRotaryPosEmbV2);
    const uint64_t workspace = PlanAclnnOp<ApplyRotaryPosEmbV2PlanFn>(
        ops_, entry.op, &executor, t.q_rope, t.k_rope, t.rope_cos, t.rope_sin, kRotaryLayoutBsnd, kRotaryMode);
    adopt(entry, workspace, executor);
  }
  // 8. paged cache write. The two co-indexed caches carry the latent and the
  //    rope slice, so one scatter writes both halves of the MLA row.
  {
    PipelineStage& entry = add("kv_cache_write", OpId::kScatterPaKvCache);
    const uint64_t workspace = PlanAclnnOp<ScatterPaKvCachePlanFn>(
        ops_, entry.op, &executor, t.kv_latent_normed, t.kv_latent_cache, t.slot_mapping, t.k_rope,
        t.kv_rope_cache, nullptr, nullptr, nullptr, kScatterCacheMode, nullptr, nullptr, nullptr);
    adopt(entry, workspace, executor);
  }
  // 9. paged MLA decode attention.
  {
    PipelineStage& entry = add("attention", OpId::kFusedInferAttentionScoreV5);
    const double scale = 1.0 / std::sqrt(static_cast<double>(mla.q_head_dim()));
    const uint64_t workspace = PlanAclnnOp<FusedInferAttentionScoreV5PlanFn>(
        ops_, entry.op, &executor, t.q_latent, t.key_list, t.value_list, nullptr, nullptr, t.actual_seq_q,
        t.actual_seq_kv, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, t.block_table, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, t.q_rope, t.key_rope_cache_view,
        nullptr, nullptr, nullptr, nullptr, nullptr, heads_, scale, kFiaPreTokensAll, kFiaNextTokensCausal,
        kFiaLayout, 1, kFiaSparseModeBand, kFiaInnerPreciseHighPrecision, config_.block_size,
        kFiaAntiquantModeNone, false, kFiaAntiquantModeNone, kFiaAntiquantModeNone, kFiaQueryQuantModeNone,
        kFiaPseTypeNone, t.attn_out, t.softmax_lse);
    arena.NoteWorkspace(workspace);
    // FIA V5 also exports an upper bound over every shape it may be given. When
    // the toolkit has it, reserve that too: a re-plan can legitimately return
    // more than the first plan did as the context grows, and the
    // AssertWorkspaceFits guard would otherwise fire mid-run.
    if (ops_.available(OpId::kFiaV5GetMaxWorkspace)) {
      aclOpExecutor* bound_executor = nullptr;
      const uint64_t upper_bound = PlanAclnnOp<FusedInferAttentionScoreV5PlanFn>(
          ops_, OpId::kFiaV5GetMaxWorkspace, &bound_executor, t.q_latent, t.key_list, t.value_list, nullptr,
          nullptr, t.actual_seq_q, t.actual_seq_kv, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          t.block_table, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, t.q_rope,
          t.key_rope_cache_view, nullptr, nullptr, nullptr, nullptr, nullptr, heads_, scale, kFiaPreTokensAll,
          kFiaNextTokensCausal, kFiaLayout, 1, kFiaSparseModeBand, kFiaInnerPreciseHighPrecision,
          config_.block_size, kFiaAntiquantModeNone, false, kFiaAntiquantModeNone, kFiaAntiquantModeNone,
          kFiaQueryQuantModeNone, kFiaPseTypeNone, t.attn_out, t.softmax_lse);
      arena.NoteWorkspace(upper_bound);
      if (bound_executor != nullptr) {
        aclDestroyAclOpExecutor(bound_executor);
      }
    }
    entry.slot.Adopt(entry.op, entry.name, workspace, executor);
  }
  // 10. quantize the latent attention output for the folded o_proj.
  {
    PipelineStage& entry = add("attn_quant", OpId::kDynamicMxQuant);
    const uint64_t workspace = PlanAclnnOp<DynamicMxQuantPlanFn>(
        ops_, entry.op, &executor, t.attn_flat, kSwigluAxisLast, kMxRoundModeRint,
        static_cast<int64_t>(kAclFloat8E4m3Fn), kRoutedScaleBlock, kSwigluScaleAlgOcp, t.attn_fp8,
        t.attn_mx_scale);
    adopt(entry, workspace, executor);
  }
  // 11. output projection, with W_UV folded in.
  {
    PipelineStage& entry = add("o_proj", OpId::kQuantMatmulV5);
    const uint64_t workspace = PlanAclnnOp<QuantMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.attn_fp8, t.w_o, t.attn_mx_scale, t.w_o_scale, nullptr, nullptr, nullptr,
        nullptr, nullptr, false, true, config_.dense_group_size, t.proj_out);
    adopt(entry, workspace, executor);
  }
  // 12. attention residual.
  {
    PipelineStage& entry = add("residual_attn", OpId::kInplaceAdd);
    const uint64_t workspace =
        PlanAclnnOp<InplaceAddPlanFn>(ops_, entry.op, &executor, t.hidden, t.proj_out, nullptr);
    adopt(entry, workspace, executor);
  }
  // 13. post-attention RMSNorm, quantized, for the expert and shared GEMMs.
  {
    PipelineStage& entry = add("post_norm_quant", OpId::kRmsNormDynamicMxQuant);
    const uint64_t workspace = PlanAclnnOp<RmsNormDynamicMxQuantPlanFn>(
        ops_, entry.op, &executor, t.hidden, t.w_post_norm, nullptr, kRmsNormEpsilon, kSwigluScaleAlgOcp,
        kMxRoundModeRint, static_cast<int64_t>(kAclFloat8E4m3Fn), false, t.normed_fp8, t.normed_mx_scale, nullptr);
    adopt(entry, workspace, executor);
  }
  // 14. the same norm unquantized, because the router reads bf16.
  {
    PipelineStage& entry = add("post_norm", OpId::kRmsNorm);
    const uint64_t workspace = PlanAclnnOp<RmsNormPlanFn>(ops_, entry.op, &executor, t.hidden, t.w_post_norm,
                                                          kRmsNormEpsilon, t.normed, t.rstd);
    adopt(entry, workspace, executor);
  }
  // 15.-18. live in MoeRouterEngine (router matmul, sqrtsoftplus scoring,
  //     gating, dropless dispatch).
  //
  // 19.-21. the routed experts: planned by the injected IRoutedMoeBlock into
  //     this same table -- expert_gemm1 (+ expert_swiglu on the decomposed
  //     path), expert_gemm2 and expert_combine. The orchestrator no longer
  //     names a single grouped-GEMM entry point; the backend decides.
  moe_block_->PlanStages(stages_, arena_manager_, slot_addrs_);
  // 22-25. the shared expert: every token uses it, so it is never routed.
  {
    PipelineStage& entry = add("shared_gate_up", OpId::kQuantMatmulV5);
    const uint64_t workspace = PlanAclnnOp<QuantMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.normed_fp8, t.w_shared_gate_up, t.normed_mx_scale, t.w_shared_gate_up_scale,
        nullptr, nullptr, nullptr, nullptr, nullptr, false, true, config_.dense_group_size, t.shared_gate_up);
    adopt(entry, workspace, executor);
  }
  {
    PipelineStage& entry = add("shared_act", OpId::kSwiGlu);
    const uint64_t workspace =
        PlanAclnnOp<SwiGluPlanFn>(ops_, entry.op, &executor, t.shared_gate_up, -1, t.shared_act);
    adopt(entry, workspace, executor);
  }
  {
    PipelineStage& entry = add("shared_act_quant", OpId::kDynamicMxQuant);
    const uint64_t workspace = PlanAclnnOp<DynamicMxQuantPlanFn>(
        ops_, entry.op, &executor, t.shared_act, kSwigluAxisLast, kMxRoundModeRint,
        static_cast<int64_t>(kAclFloat8E4m3Fn), kRoutedScaleBlock, kSwigluScaleAlgOcp, t.shared_act_fp8,
        t.shared_act_scale);
    adopt(entry, workspace, executor);
  }
  {
    PipelineStage& entry = add("shared_down", OpId::kQuantMatmulV5);
    const uint64_t workspace = PlanAclnnOp<QuantMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.shared_act_fp8, t.w_shared_down, t.shared_act_scale, t.w_shared_down_scale,
        nullptr, nullptr, nullptr, nullptr, nullptr, false, true, config_.dense_group_size, t.shared_out);
    adopt(entry, workspace, executor);
  }
  // 26-27. routed + shared, then the MoE residual.
  {
    PipelineStage& entry = add("add_shared", OpId::kInplaceAdd);
    const uint64_t workspace =
        PlanAclnnOp<InplaceAddPlanFn>(ops_, entry.op, &executor, t.routed_out, t.shared_out, nullptr);
    adopt(entry, workspace, executor);
  }
  {
    PipelineStage& entry = add("residual_moe", OpId::kInplaceAdd);
    const uint64_t workspace =
        PlanAclnnOp<InplaceAddPlanFn>(ops_, entry.op, &executor, t.hidden, t.routed_out, nullptr);
    adopt(entry, workspace, executor);
  }
  // 28-30. the head.
  {
    PipelineStage& entry = add("final_norm", OpId::kRmsNorm);
    const uint64_t workspace = PlanAclnnOp<RmsNormPlanFn>(ops_, entry.op, &executor, t.hidden, t.w_final_norm,
                                                          kRmsNormEpsilon, t.final_normed, t.rstd);
    adopt(entry, workspace, executor);
  }
  {
    PipelineStage& entry = add("lm_head", OpId::kMatmul);
    const uint64_t workspace = PlanAclnnOp<MatmulPlanFn>(ops_, entry.op, &executor, t.final_normed, t.w_lm_head,
                                                         t.logits, kCubeMathTypeKeepDtype);
    adopt(entry, workspace, executor);
  }
  {
    PipelineStage& entry = add("argmax", OpId::kArgMax);
    const uint64_t workspace = PlanAclnnOp<ArgMaxPlanFn>(ops_, entry.op, &executor, t.logits, -1, false, t.argmax);
    adopt(entry, workspace, executor);
  }
}

// ---------------------------------------------------------------------------
// Decode step
// ---------------------------------------------------------------------------

void Dsv4Pipeline::Launch(PipelineStage& entry) {
  entry.slot.Launch(ops_, arena_manager_.arena().workspace(), compute_stream_);
  ++counters_.launches;
}

void Dsv4Pipeline::RunAttention(int32_t layer, int64_t position) {
  const MlaGeometry& mla = config_.mla;
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();
  const BackboneWeights& weights = arena_manager_.backbone();
  const BackboneWeights::Layer& layer_weights = weights.layers[static_cast<size_t>(layer)];

  // Repoint this layer's weights. The descriptors and the executors are the
  // ones planned at init; only the addresses move.
  stages_.stage("input_norm_quant")
      .slot.SetAddress(slot::kMxQuantGamma, t.w_input_norm, arena.Address(layer_weights.input_norm));
  stages_.stage("q_a_proj").slot.SetAddress(slot::kQuantMmX2, t.w_q_a, arena.Address(layer_weights.q_a_weight));
  stages_.stage("q_a_proj").slot.SetAddress(slot::kQuantMmX2Scale, t.w_q_a_scale, arena.Address(layer_weights.q_a_scale));
  stages_.stage("q_a_norm_quant").slot.SetAddress(slot::kMxQuantGamma, t.w_q_a_norm, arena.Address(layer_weights.q_a_norm));
  stages_.stage("q_b_proj").slot.SetAddress(slot::kQuantMmX2, t.w_q_b, arena.Address(layer_weights.q_b_weight));
  stages_.stage("q_b_proj").slot.SetAddress(slot::kQuantMmX2Scale, t.w_q_b_scale, arena.Address(layer_weights.q_b_scale));
  stages_.stage("kv_a_proj").slot.SetAddress(slot::kQuantMmX2, t.w_kv_a, arena.Address(layer_weights.kv_a_weight));
  stages_.stage("kv_a_proj").slot.SetAddress(slot::kQuantMmX2Scale, t.w_kv_a_scale, arena.Address(layer_weights.kv_a_scale));
  stages_.stage("kv_a_norm").slot.SetAddress(slot::kRmsNormGamma, t.w_kv_a_norm, arena.Address(layer_weights.kv_a_norm));
  stages_.stage("o_proj").slot.SetAddress(slot::kQuantMmX2, t.w_o, arena.Address(layer_weights.o_weight));
  stages_.stage("o_proj").slot.SetAddress(slot::kQuantMmX2Scale, t.w_o_scale, arena.Address(layer_weights.o_scale));

  // This layer's slice of the paged cache.
  const size_t latent_stride = Bf16Bytes(arena_manager_.num_blocks() * config_.block_size * mla.kv_lora_rank);
  const size_t rope_stride = Bf16Bytes(arena_manager_.num_blocks() * config_.block_size * mla.qk_rope_head_dim);
  uint8_t* latent =
      arena.AddressAs<uint8_t>(weights.kv_latent_cache) + latent_stride * static_cast<size_t>(layer);
  uint8_t* rope = arena.AddressAs<uint8_t>(weights.kv_rope_cache) + rope_stride * static_cast<size_t>(layer);
  stages_.stage("kv_cache_write").slot.SetAddress(slot::kScatterKeyCache, t.kv_latent_cache, latent);
  stages_.stage("kv_cache_write").slot.SetAddress(slot::kScatterValueCache, t.kv_rope_cache, rope);
  stages_.stage("attention").slot.SetTensorListAddress(slot::kFiaKeyList, 0, t.key_list, latent);
  stages_.stage("attention").slot.SetTensorListAddress(slot::kFiaValueList, 0, t.value_list, latent);
  stages_.stage("attention").slot.SetAddress(slot::kFiaKeyRope, t.key_rope_cache_view, rope);

  // The rope table row for this position.
  uint8_t* cos_row = arena.AddressAs<uint8_t>(weights.rope_cos) + Bf16Bytes(position * mla.qk_rope_head_dim);
  uint8_t* sin_row = arena.AddressAs<uint8_t>(weights.rope_sin) + Bf16Bytes(position * mla.qk_rope_head_dim);
  stages_.stage("rope").slot.SetAddress(slot::kRotaryCos, t.rope_cos, cos_row);
  stages_.stage("rope").slot.SetAddress(slot::kRotarySin, t.rope_sin, sin_row);

  Launch(stages_.stage("input_norm_quant"));
  Launch(stages_.stage("q_a_proj"));
  Launch(stages_.stage("q_a_norm_quant"));
  Launch(stages_.stage("q_b_proj"));
  Launch(stages_.stage("kv_a_proj"));
  Launch(stages_.stage("kv_a_norm"));
  Launch(stages_.stage("rope"));
  Launch(stages_.stage("kv_cache_write"));
  Launch(stages_.stage("attention"));
  Launch(stages_.stage("attn_quant"));
  Launch(stages_.stage("o_proj"));
  Launch(stages_.stage("residual_attn"));
}

void Dsv4Pipeline::RunSharedExpert(int32_t layer) {
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();
  const BackboneWeights::Layer& weights = arena_manager_.backbone().layers[static_cast<size_t>(layer)];
  stages_.stage("shared_gate_up")
      .slot.SetAddress(slot::kQuantMmX2, t.w_shared_gate_up, arena.Address(weights.shared_gate_up_weight));
  stages_.stage("shared_gate_up")
      .slot.SetAddress(slot::kQuantMmX2Scale, t.w_shared_gate_up_scale,
                       arena.Address(weights.shared_gate_up_scale));
  stages_.stage("shared_down")
      .slot.SetAddress(slot::kQuantMmX2, t.w_shared_down, arena.Address(weights.shared_down_weight));
  stages_.stage("shared_down")
      .slot.SetAddress(slot::kQuantMmX2Scale, t.w_shared_down_scale, arena.Address(weights.shared_down_scale));
  Launch(stages_.stage("shared_gate_up"));
  Launch(stages_.stage("shared_act"));
  Launch(stages_.stage("shared_act_quant"));
  Launch(stages_.stage("shared_down"));
}

void Dsv4Pipeline::RunMoe(int32_t layer) {
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();
  const BackboneWeights::Layer& weights = arena_manager_.backbone().layers[static_cast<size_t>(layer)];

  stages_.stage("post_norm_quant").slot.SetAddress(slot::kMxQuantGamma, t.w_post_norm, arena.Address(weights.post_norm));
  stages_.stage("post_norm").slot.SetAddress(slot::kRmsNormGamma, t.w_post_norm, arena.Address(weights.post_norm));

  Launch(stages_.stage("post_norm_quant"));
  Launch(stages_.stage("post_norm"));

  // Score, select, and read the top-6 back (deviation 1: the one forced host
  // round trip per MoE layer).
  const int32_t* expert_ids =
      router_.ScoreAndSelect(layer, arena_manager_, ops_, weights, compute_stream_, compute_done_);
  ++counters_.host_synchronizations;

  // The exclusive hierarchy makes the six chosen experts resident, batching
  // every miss into one event-ordered duplex exchange before the GEMM. The
  // residency plan travels into the dispatch context: binding the weight
  // lists is the backend's job now.
  const LayerSwapPlan plan = experts_.PrepareLayer(layer, expert_ids, static_cast<int32_t>(kNumExpertsPerTok),
                                                   compute_stream_, compute_done_);
  counters_.expert_slot_hits += static_cast<uint64_t>(plan.hit_count);
  counters_.expert_slot_misses += static_cast<uint64_t>(plan.miss_count);

  MoeDispatchContext dispatch;
  dispatch.layer = layer;
  dispatch.token_count = kTokensPerStep;
  dispatch.top_k = kNumExpertsPerTok;
  dispatch.expanded_x = t.expanded_x;
  dispatch.expanded_scale = t.expanded_scale;
  dispatch.group_list = t.group_list;
  dispatch.gemm1_out = t.gemm1_out;
  dispatch.gemm1_scale = t.gemm1_scale;
  dispatch.gemm2_out = t.gemm2_out;
  dispatch.expanded_weights_row = t.expanded_weights_row;
  dispatch.routed_out = t.routed_out;
  dispatch.swap_plan = &plan;
  dispatch.compute_stream = compute_stream_;

  const uint64_t block_launches_before = moe_block_->launches();
  moe_block_->ExecuteMoe(dispatch, streams_);
  counters_.launches += moe_block_->launches() - block_launches_before;

  RunSharedExpert(layer);
  Launch(stages_.stage("add_shared"));
  Launch(stages_.stage("residual_moe"));

  // Publish the compute boundary: the next layer's swap batch waits on it
  // before overwriting any slot this layer's GEMMs read.
  streams_.RecordEvent(compute_done_, compute_stream_);
  ++counters_.layers;
}

void Dsv4Pipeline::DecodeStep(int32_t token_id, int64_t position) {
  DSV4_REQUIRE(arena_manager_.arena().sealed(), "DecodeStep before Build()");
  DSV4_REQUIRE(token_id >= 0 && token_id < kVocabSize, "token id " << token_id << " outside the vocabulary");
  DSV4_REQUIRE(position >= 0 && position < config_.max_context_len,
               "position " << position << " outside the reserved context of " << config_.max_context_len);

  const uint64_t allocations_before = allocator_.DeviceAllocationCount();
  const size_t descriptors_before = arena_manager_.arena().descriptors().size();

  // Embedding lookup for a single greedy token is one row copy, not a gather
  // kernel: no descriptor, no workspace, no launch.
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();
  const BackboneWeights& weights = arena_manager_.backbone();
  uint8_t* embed_row =
      arena.AddressAs<uint8_t>(weights.embed_tokens) + Bf16Bytes(static_cast<int64_t>(token_id) * kHiddenSize);
  streams_.MemcpyAsync(arena.Address(t.h_hidden), Bf16Bytes(kHiddenSize), embed_row, Bf16Bytes(kHiddenSize),
                       MemcpyKind::kDeviceToDevice, compute_stream_);

  // The slot this token occupies in the paged cache. The pinned mailbox is the
  // DMA source, so the copy can be async without a stack lifetime problem.
  *slot_mailbox_ = static_cast<int32_t>(position);
  streams_.MemcpyAsync(arena.Address(t.h_slot_mapping), Int32Bytes(1), slot_mailbox_, Int32Bytes(1),
                       MemcpyKind::kHostToDevice, compute_stream_);

  for (int32_t layer = 0; layer < static_cast<int32_t>(kNumLayers); ++layer) {
    RunAttention(layer, position);
    RunMoe(layer);
  }
  Launch(stages_.stage("final_norm"));
  Launch(stages_.stage("lm_head"));
  Launch(stages_.stage("argmax"));

  ++counters_.steps;
  counters_.device_allocations_in_step += allocator_.DeviceAllocationCount() - allocations_before;
  counters_.descriptors_built_in_step += arena_manager_.arena().descriptors().size() - descriptors_before;
}

int32_t Dsv4Pipeline::ReadArgmaxToken() {
  ArenaTensors& t = arena_manager_.tensors();
  // The one synchronization that ends a step.
  streams_.MemcpyAsync(token_mailbox_, Int64Bytes(1), arena_manager_.arena().Address(t.h_argmax), Int64Bytes(1),
                       MemcpyKind::kDeviceToHost, compute_stream_);
  streams_.SynchronizeStream(compute_stream_);
  ++counters_.host_synchronizations;
  const int64_t token = *token_mailbox_;
  DSV4_REQUIRE(token >= 0 && token < kVocabSize,
               "the LM head returned token " << token << ", outside the vocabulary");
  return static_cast<int32_t>(token);
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

std::string Dsv4Pipeline::DescribeStages() const {
  std::ostringstream out;
  const size_t total_stages = stages_.size() + router_.stages().size();
  out << "pipeline: " << total_stages << " planned stages (" << stages_.size()
      << " in the orchestrator table, " << router_.stages().size()
      << " in the router engine), replayed " << kNumLayers << " times with address swaps\n";
  out << "  moe block:    " << moe_block_->DescribeBackend() << "\n";
  uint64_t high_water = 0;
  const auto describe_stage = [&out, &high_water](const PipelineStage& entry) {
    out << "  " << std::left << std::setw(20) << entry.name << std::right << std::setw(40) << OpName(entry.op)
        << "  workspace=" << std::setw(10) << entry.slot.workspace_size() << "\n";
    high_water = std::max(high_water, entry.slot.workspace_size());
  };
  for (size_t i = 0; i < stages_.size(); ++i) {
    describe_stage(stages_[i]);
  }
  for (const PipelineStage& entry : router_.stages()) {
    describe_stage(entry);
  }
  out << "  shared workspace high-water: " << high_water << " bytes\n";
  out << "  NOT APPLIED: the Lightning Indexer (index_topk=" << kIndexTopK
      << "). The specified attention op is the dense paged MLA decode; the sparse\n"
         "               selection belongs to aclnnSparseFlashMla, which the brief's mapping does not list.\n";
  out << "  DECOMPOSED SCORING: the router scoring is sqrt(softplus(logits)) -- aclnnSoftplus (beta="
      << kSoftplusBeta << ",\n               threshold=" << kSoftplusThreshold << ") then aclnnSqrt -- because the "
         "stock gating\n               operator's normType offers softmax/sigmoid only. aclnnMoeGatingTopKV2 receives "
         "the\n               pre-normalized scores (normType="
      << config_.gating_norm_type << ", renorm=" << kGatingRenormL1 << ", eps=" << kGatingEps
      << ").\n               The pass-through normType value still needs a device A/B against the host reference in\n"
         "               v5_ops_moe.py (sqrt_softplus_routing); --gating-norm-type overrides it.\n";
  out << "  UNVERIFIED: aclnnQuantMatmulV5 groupSize=" << config_.dense_group_size
      << " (the block-128 dense scale encoding).\n";
  if (!arena_manager_.backbone().rope_tables_populated) {
    out << "  WARNING: the rope cos/sin tables were NOT populated by the checkpoint and are zero. Attention\n"
           "           would compute without positional information. Supply model.rotary_emb.{cos,sin}_cached.\n";
  }
  return out.str();
}

std::string Dsv4Pipeline::DescribeSlotIndexMap() const {
  std::ostringstream out;
  out << "aclSetTensorAddr index map (derive-and-verify; the plan phase cannot check these)\n";
  out << "  RmsNorm                 x=" << slot::kRmsNormX << " gamma=" << slot::kRmsNormGamma
      << " y=" << slot::kRmsNormY << " rstd=" << slot::kRmsNormRstd << "\n";
  out << "  RmsNormDynamicMxQuant   x=" << slot::kMxQuantX << " gamma=" << slot::kMxQuantGamma
      << " beta=" << slot::kMxQuantBeta << " y=" << slot::kMxQuantY << " mxscale=" << slot::kMxQuantScale << "\n";
  out << "  Matmul                  self=" << slot::kMatmulSelf << " mat2=" << slot::kMatmulMat2
      << " out=" << slot::kMatmulOut << "\n";
  out << "  QuantMatmulV5           x1=" << slot::kQuantMmX1 << " x2=" << slot::kQuantMmX2
      << " x1Scale=" << slot::kQuantMmX1Scale << " x2Scale=" << slot::kQuantMmX2Scale
      << " out=" << slot::kQuantMmOut << "\n";
  out << "  Softplus                x=" << slot::kSoftplusX << " out=" << slot::kSoftplusOut
      << " (beta/threshold are scalars, skipped by the IR numbering)\n";
  out << "  Sqrt                    x=" << slot::kSqrtX << " out=" << slot::kSqrtOut << "\n";
  out << "  MoeGatingTopKV2         x=" << slot::kGatingX << " bias=" << slot::kGatingBias
      << " y=" << slot::kGatingY << " expertIdx=" << slot::kGatingExpertIdx << "\n";
  out << "  ApplyRotaryPosEmbV2     cos=" << slot::kRotaryCos << " sin=" << slot::kRotarySin << "\n";
  out << "  ScatterPaKvCache        keyCacheRef=" << slot::kScatterKeyCache
      << " valueCacheRef=" << slot::kScatterValueCache << "\n";
  out << "  FusedInferAttentionV5   key[0]=" << slot::kFiaKeyList << " value[0]=" << slot::kFiaValueList
      << " keyRope=" << slot::kFiaKeyRope << "\n";
  out << "  GroupedMatmulV5         weight[i]=" << slot::kGmmWeightList << " scale[i]=" << slot::kGmmV5ScaleList
      << "\n";
  out << "  GmmSwigluQuantV2        weight[i]=" << slot::kGmmWeightList
      << " weightScale[i]=" << slot::kGmmSwigluScaleList << "\n";
  return out.str();
}

}  // namespace ascend_moe
