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

#include "dsv4_pipeline.hpp"

#include <aclnn/acl_meta.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "dsv4_acl_check.hpp"
#include "dsv4_expert_layout.hpp"

namespace vllm_ascend {
namespace dsv4 {
namespace {

// This runner decodes one token at a time (greedy, batch 1). Every buffer below
// is sized for it, and the routing combine is only correct for it -- see the
// note on the `expert_combine` stage.
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

int64_t DivideUp(int64_t value, int64_t divisor) { return (value + divisor - 1) / divisor; }

size_t Fp8Bytes(int64_t elements) { return static_cast<size_t>(elements); }
size_t Bf16Bytes(int64_t elements) { return static_cast<size_t>(elements) * 2; }
size_t Fp32Bytes(int64_t elements) { return static_cast<size_t>(elements) * 4; }
size_t Int32Bytes(int64_t elements) { return static_cast<size_t>(elements) * 4; }
size_t Int64Bytes(int64_t elements) { return static_cast<size_t>(elements) * 8; }

// Number of block-128 scale columns for a weight whose reduction axis is `k`.
int64_t DenseScaleCols(int64_t k) { return DivideUp(k, kDenseScaleBlock); }
// Number of block-32 microscale columns.
int64_t MxScaleCols(int64_t k) { return DivideUp(k, kRoutedScaleBlock); }

// The folded MLA projections this pipeline consumes. A real checkpoint ships
// q_b_proj and o_proj in their unfolded form plus kv_b_proj; folding W_UK into
// q_b and W_UV into o_proj is a dequantize -> matmul -> requantize pass at full
// precision, which belongs to a checkpoint converter and not to a runtime that
// must not lose accuracy silently. The runner expects the folded tensors by
// these names and says so when they are missing.
const char* kFoldedQName = "model.layers.{L}.self_attn.q_b_proj_latent.weight";
const char* kFoldedQScaleName = "model.layers.{L}.self_attn.q_b_proj_latent.weight_scale_inv";
const char* kFoldedOName = "model.layers.{L}.self_attn.o_proj_folded.weight";
const char* kFoldedOScaleName = "model.layers.{L}.self_attn.o_proj_folded.weight_scale_inv";

std::string LayerTensorName(const char* pattern, int64_t layer) {
  std::string name(pattern);
  const std::string token = "{L}";
  const size_t position = name.find(token);
  if (position != std::string::npos) {
    name.replace(position, token.size(), std::to_string(layer));
  }
  return name;
}

}  // namespace

// ---------------------------------------------------------------------------
// Opaque state
// ---------------------------------------------------------------------------

// Per-layer backbone weight reservations. kv_b_proj never appears: its two
// up-projections are absorbed into q_b_proj and o_proj at checkpoint-conversion
// time, which is what makes the decode graph attend in the compressed latent
// space and the paged cache `kv_lora_rank + qk_rope_head_dim` wide.
struct Dsv4Pipeline::Backbone {
  struct Layer {
    ArenaHandle input_norm = kInvalidArenaHandle;
    ArenaHandle q_a_weight = kInvalidArenaHandle;
    ArenaHandle q_a_scale = kInvalidArenaHandle;
    ArenaHandle q_a_norm = kInvalidArenaHandle;
    ArenaHandle q_b_weight = kInvalidArenaHandle;  // folded: emits q_latent | q_rope
    ArenaHandle q_b_scale = kInvalidArenaHandle;
    ArenaHandle kv_a_weight = kInvalidArenaHandle;  // emits kv_latent | k_rope
    ArenaHandle kv_a_scale = kInvalidArenaHandle;
    ArenaHandle kv_a_norm = kInvalidArenaHandle;
    ArenaHandle o_weight = kInvalidArenaHandle;  // folded: consumes the latent attn out
    ArenaHandle o_scale = kInvalidArenaHandle;
    ArenaHandle post_norm = kInvalidArenaHandle;
    ArenaHandle router_weight = kInvalidArenaHandle;
    ArenaHandle router_bias = kInvalidArenaHandle;  // noaux_tc selection bias
    ArenaHandle shared_gate_up_weight = kInvalidArenaHandle;
    ArenaHandle shared_gate_up_scale = kInvalidArenaHandle;
    ArenaHandle shared_down_weight = kInvalidArenaHandle;
    ArenaHandle shared_down_scale = kInvalidArenaHandle;
  };
  std::vector<Layer> layers;
  ArenaHandle embed_tokens = kInvalidArenaHandle;
  ArenaHandle final_norm = kInvalidArenaHandle;
  ArenaHandle lm_head = kInvalidArenaHandle;
  ArenaHandle rope_cos = kInvalidArenaHandle;
  ArenaHandle rope_sin = kInvalidArenaHandle;
  ArenaHandle kv_latent_cache = kInvalidArenaHandle;
  ArenaHandle kv_rope_cache = kInvalidArenaHandle;
  ArenaHandle block_table = kInvalidArenaHandle;
  ArenaHandle slot_mapping = kInvalidArenaHandle;
  bool rope_tables_populated = false;
};

// Every descriptor the graph uses. Created once in CreateDescriptors.
struct Dsv4Pipeline::Tensors {
  // activations
  aclTensor* hidden = nullptr;            // [1, 4096] bf16
  aclTensor* normed = nullptr;            // [1, 4096] bf16
  aclTensor* normed_fp8 = nullptr;        // [1, 4096] fp8
  aclTensor* normed_mx_scale = nullptr;   // [1, 128] e8m0
  aclTensor* rstd = nullptr;              // [1, 1] fp32
  aclTensor* q_a = nullptr;               // [1, 1024] bf16
  aclTensor* q_a_fp8 = nullptr;           // [1, 1024] fp8
  aclTensor* q_a_mx_scale = nullptr;      // [1, 32] e8m0
  aclTensor* q_b = nullptr;               // [1, heads*(kv_lora+rope)] bf16
  aclTensor* q_latent = nullptr;          // [1, heads, kv_lora] strided view of q_b
  aclTensor* q_rope = nullptr;            // [1, heads, rope] strided view of q_b
  aclTensor* kv_a = nullptr;              // [1, kv_lora+rope] bf16
  aclTensor* kv_latent = nullptr;         // [1, kv_lora] view of kv_a
  aclTensor* k_rope = nullptr;            // [1, 1, rope] view of kv_a
  aclTensor* kv_latent_normed = nullptr;  // [1, kv_lora] bf16
  aclTensor* attn_out = nullptr;          // [1, heads, kv_lora] bf16
  aclTensor* attn_flat = nullptr;         // [1, heads*kv_lora] bf16
  aclTensor* attn_fp8 = nullptr;          // [1, heads*kv_lora] fp8
  aclTensor* attn_mx_scale = nullptr;     // [1, heads*kv_lora/32] e8m0
  aclTensor* proj_out = nullptr;          // [1, 4096] bf16
  aclTensor* softmax_lse = nullptr;       // [1, heads, 1] fp32

  // routing
  aclTensor* router_logits = nullptr;         // [1, 256] fp32
  aclTensor* gating_weights = nullptr;        // [1, 6] fp32
  aclTensor* gating_indices = nullptr;        // [1, 6] int32
  aclTensor* local_indices = nullptr;         // [1, 6] int32 (0..5, host-written)
  aclTensor* expanded_x = nullptr;            // [6, 4096] fp8
  aclTensor* expanded_row_idx = nullptr;      // [6] int32
  aclTensor* expanded_scale = nullptr;        // [6, 128] e8m0
  aclTensor* expanded_weights = nullptr;      // [6] fp32
  aclTensor* expanded_weights_row = nullptr;  // [1, 6] fp32 view, for the combine
  aclTensor* group_list = nullptr;            // [6] int64 cumsum
  aclTensor* gemm1_raw = nullptr;             // [6, 2*2048] bf16 (decomposed path)
  aclTensor* gemm1_out = nullptr;             // [6, 2048] fp8
  aclTensor* gemm1_scale = nullptr;           // [6, 64] e8m0
  aclTensor* gemm2_out = nullptr;             // [6, 4096] bf16
  aclTensor* routed_out = nullptr;            // [1, 4096] bf16
  aclTensorList* gemm1_x_list = nullptr;
  aclTensorList* gemm1_x_scale_list = nullptr;
  aclTensorList* gemm1_raw_out_list = nullptr;
  aclTensorList* gemm2_x_list = nullptr;
  aclTensorList* gemm2_x_scale_list = nullptr;
  aclTensorList* gemm2_out_list = nullptr;

  // shared expert
  aclTensor* shared_gate_up = nullptr;    // [1, 2*2048] bf16
  aclTensor* shared_act = nullptr;        // [1, 2048] bf16
  aclTensor* shared_act_fp8 = nullptr;    // [1, 2048] fp8
  aclTensor* shared_act_scale = nullptr;  // [1, 64] e8m0
  aclTensor* shared_out = nullptr;        // [1, 4096] bf16

  // head
  aclTensor* final_normed = nullptr;  // [1, 4096] bf16
  aclTensor* logits = nullptr;        // [1, vocab] bf16
  aclTensor* argmax = nullptr;        // [1] int64

  // paged KV
  aclTensor* kv_latent_cache = nullptr;
  aclTensor* kv_rope_cache = nullptr;
  aclTensorList* key_list = nullptr;
  aclTensorList* value_list = nullptr;
  aclTensor* key_rope_cache_view = nullptr;
  aclTensor* block_table = nullptr;
  aclTensor* slot_mapping = nullptr;
  aclIntArray* actual_seq_q = nullptr;
  aclIntArray* actual_seq_kv = nullptr;

  // rope tables
  aclTensor* rope_cos = nullptr;
  aclTensor* rope_sin = nullptr;

  // per-layer weights, repointed by aclSetTensorAddr
  aclTensor* w_input_norm = nullptr;
  aclTensor* w_q_a = nullptr;
  aclTensor* w_q_a_scale = nullptr;
  aclTensor* w_q_a_norm = nullptr;
  aclTensor* w_q_b = nullptr;
  aclTensor* w_q_b_scale = nullptr;
  aclTensor* w_kv_a = nullptr;
  aclTensor* w_kv_a_scale = nullptr;
  aclTensor* w_kv_a_norm = nullptr;
  aclTensor* w_o = nullptr;
  aclTensor* w_o_scale = nullptr;
  aclTensor* w_post_norm = nullptr;
  aclTensor* w_router = nullptr;
  aclTensor* w_router_bias = nullptr;
  aclTensor* w_shared_gate_up = nullptr;
  aclTensor* w_shared_gate_up_scale = nullptr;
  aclTensor* w_shared_down = nullptr;
  aclTensor* w_shared_down_scale = nullptr;
  aclTensor* w_final_norm = nullptr;
  aclTensor* w_lm_head = nullptr;

  // the six active experts, repointed per layer
  std::vector<aclTensor*> expert_gate_up;
  std::vector<aclTensor*> expert_gate_up_scale;
  std::vector<aclTensor*> expert_down;
  std::vector<aclTensor*> expert_down_scale;
  aclTensorList* expert_gate_up_list = nullptr;
  aclTensorList* expert_gate_up_scale_list = nullptr;
  aclTensorList* expert_down_list = nullptr;
  aclTensorList* expert_down_scale_list = nullptr;

  // arena handles the step needs addresses from
  ArenaHandle h_hidden = kInvalidArenaHandle;
  ArenaHandle h_gating_indices = kInvalidArenaHandle;
  ArenaHandle h_local_indices = kInvalidArenaHandle;
  ArenaHandle h_group_list = kInvalidArenaHandle;
  ArenaHandle h_argmax = kInvalidArenaHandle;
  ArenaHandle h_slot_mapping = kInvalidArenaHandle;
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Dsv4Pipeline::Dsv4Pipeline(DeviceOps& device, const OpTable& ops, ExclusiveExpertManager& experts,
                           const RuntimeConfig& config)
    : device_(device), ops_(ops), experts_(experts), config_(config), arena_(device) {
  DSV4_REQUIRE(config_.block_size > 0, "paged block size must be positive");
  DSV4_REQUIRE(config_.max_context_len >= config_.block_size,
               "max context " << config_.max_context_len << " is below one block of " << config_.block_size);
  num_blocks_ = DivideUp(config_.max_context_len, config_.block_size);
  tensors_ = new Tensors();
  backbone_ = new Backbone();
  backbone_->layers.resize(static_cast<size_t>(kNumLayers));

  compute_stream_ = device_.CreateStream();
  readback_stream_ = device_.CreateStream();
  compute_done_ = device_.CreateEvent();
  // Primed for the same reason the staging engine primes its events: the first
  // layer's ExecutePlan waits on it before anything has recorded it.
  device_.RecordEvent(compute_done_, compute_stream_);
  device_.SynchronizeStream(compute_stream_);

  routing_mailbox_ = static_cast<int32_t*>(device_.HostPinnedMalloc(Int32Bytes(kNumExpertsPerTok)));
  local_index_mailbox_ = static_cast<int32_t*>(device_.HostPinnedMalloc(Int32Bytes(kNumExpertsPerTok)));
  token_mailbox_ = static_cast<int64_t*>(device_.HostPinnedMalloc(Int64Bytes(1)));
  slot_mailbox_ = static_cast<int32_t*>(device_.HostPinnedMalloc(Int32Bytes(1)));
  std::memset(routing_mailbox_, 0, Int32Bytes(kNumExpertsPerTok));
  std::memset(local_index_mailbox_, 0, Int32Bytes(kNumExpertsPerTok));
  *token_mailbox_ = 0;
  *slot_mailbox_ = 0;
}

Dsv4Pipeline::~Dsv4Pipeline() {
  // The stages own repeatable executors; destroy them before the arena frees
  // the memory their descriptors point at.
  stages_.clear();
  if (routing_mailbox_ != nullptr) {
    device_.HostPinnedFree(routing_mailbox_);
  }
  if (local_index_mailbox_ != nullptr) {
    device_.HostPinnedFree(local_index_mailbox_);
  }
  if (token_mailbox_ != nullptr) {
    device_.HostPinnedFree(token_mailbox_);
  }
  if (slot_mailbox_ != nullptr) {
    device_.HostPinnedFree(slot_mailbox_);
  }
  if (compute_done_ != nullptr) {
    device_.DestroyEvent(compute_done_);
  }
  if (compute_stream_ != nullptr) {
    device_.DestroyStream(compute_stream_);
  }
  if (readback_stream_ != nullptr) {
    device_.DestroyStream(readback_stream_);
  }
  delete tensors_;
  delete backbone_;
}

size_t Dsv4Pipeline::BackboneDeviceBytes(const MlaGeometry& mla, int64_t block_size, int64_t max_context_len) {
  const int64_t kv_row = mla.kv_row_elements();
  const int64_t q_b_width = kNumAttentionHeads * kv_row;          // folded q_b output
  const int64_t o_input = kNumAttentionHeads * mla.kv_lora_rank;  // folded o_proj input
  const int64_t shared_inter = kMoeIntermediateSize * kNumSharedExperts;

  size_t per_layer = 0;
  per_layer += Bf16Bytes(kHiddenSize);                                   // input norm
  per_layer += Fp8Bytes(kQLoraRank * kHiddenSize);                       // q_a
  per_layer += Fp8Bytes(kQLoraRank * DenseScaleCols(kHiddenSize));       // q_a scale
  per_layer += Bf16Bytes(kQLoraRank);                                    // q_a norm
  per_layer += Fp8Bytes(q_b_width * kQLoraRank);                         // q_b (folded)
  per_layer += Fp8Bytes(q_b_width * DenseScaleCols(kQLoraRank));         // q_b scale
  per_layer += Fp8Bytes(kv_row * kHiddenSize);                           // kv_a
  per_layer += Fp8Bytes(kv_row * DenseScaleCols(kHiddenSize));           // kv_a scale
  per_layer += Bf16Bytes(mla.kv_lora_rank);                              // kv_a norm
  per_layer += Fp8Bytes(kHiddenSize * o_input);                          // o (folded)
  per_layer += Fp8Bytes(kHiddenSize * DenseScaleCols(o_input));          // o scale
  per_layer += Bf16Bytes(kHiddenSize);                                   // post norm
  per_layer += Bf16Bytes(kNumRoutedExperts * kHiddenSize);               // router
  per_layer += Fp32Bytes(kNumRoutedExperts);                             // router bias
  per_layer += Fp8Bytes(2 * shared_inter * kHiddenSize);                 // shared gate/up
  per_layer += Fp8Bytes(2 * shared_inter * DenseScaleCols(kHiddenSize)); // shared gate/up scale
  per_layer += Fp8Bytes(kHiddenSize * shared_inter);                     // shared down
  per_layer += Fp8Bytes(kHiddenSize * DenseScaleCols(shared_inter));     // shared down scale

  const int64_t blocks = DivideUp(max_context_len, block_size);
  size_t total = per_layer * static_cast<size_t>(kNumLayers);
  total += Bf16Bytes(kVocabSize * kHiddenSize) * 2;                // embed_tokens + lm_head
  total += Bf16Bytes(kHiddenSize);                                 // final norm
  total += Bf16Bytes(max_context_len * mla.qk_rope_head_dim) * 2;  // cos / sin tables
  // Paged MLA cache: one latent row and one rope row per token slot, per layer.
  total += Bf16Bytes(blocks * block_size * mla.kv_lora_rank) * kNumLayers;
  total += Bf16Bytes(blocks * block_size * mla.qk_rope_head_dim) * kNumLayers;
  return total;
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

void Dsv4Pipeline::Build(WeightByteSource& source) {
  ReserveBuffers();
  ReserveBackbone();
  arena_.Commit();
  IngestBackbone(source);
  CreateDescriptors();
  PlanStages();
  arena_.CommitWorkspace();
  arena_.Seal();
}

void Dsv4Pipeline::ReserveBuffers() {
  const MlaGeometry& mla = config_.mla;
  const int64_t kv_row = mla.kv_row_elements();
  const int64_t q_b_width = heads_ * kv_row;
  const int64_t o_input = heads_ * mla.kv_lora_rank;
  const int64_t expanded_rows = kTokensPerStep * kNumExpertsPerTok;
  const int64_t shared_inter = kMoeIntermediateSize * kNumSharedExperts;
  Tensors& t = *tensors_;

  t.h_hidden = arena_.Reserve("act.hidden", Bf16Bytes(kTokensPerStep * kHiddenSize));
  arena_.Reserve("act.normed", Bf16Bytes(kTokensPerStep * kHiddenSize));
  arena_.Reserve("act.normed_fp8", Fp8Bytes(kTokensPerStep * kHiddenSize));
  arena_.Reserve("act.normed_mx_scale", Fp8Bytes(kTokensPerStep * MxScaleCols(kHiddenSize)));
  arena_.Reserve("act.rstd", Fp32Bytes(kTokensPerStep));
  arena_.Reserve("act.q_a", Bf16Bytes(kTokensPerStep * kQLoraRank));
  arena_.Reserve("act.q_a_fp8", Fp8Bytes(kTokensPerStep * kQLoraRank));
  arena_.Reserve("act.q_a_mx_scale", Fp8Bytes(kTokensPerStep * MxScaleCols(kQLoraRank)));
  arena_.Reserve("act.q_b", Bf16Bytes(kTokensPerStep * q_b_width));
  arena_.Reserve("act.kv_a", Bf16Bytes(kTokensPerStep * kv_row));
  arena_.Reserve("act.kv_latent_normed", Bf16Bytes(kTokensPerStep * mla.kv_lora_rank));
  arena_.Reserve("act.attn_out", Bf16Bytes(kTokensPerStep * o_input));
  arena_.Reserve("act.attn_fp8", Fp8Bytes(kTokensPerStep * o_input));
  arena_.Reserve("act.attn_mx_scale", Fp8Bytes(kTokensPerStep * MxScaleCols(o_input)));
  arena_.Reserve("act.softmax_lse", Fp32Bytes(kTokensPerStep * heads_));
  arena_.Reserve("act.proj_out", Bf16Bytes(kTokensPerStep * kHiddenSize));

  arena_.Reserve("moe.router_logits", Fp32Bytes(kTokensPerStep * kNumRoutedExperts));
  arena_.Reserve("moe.gating_weights", Fp32Bytes(kTokensPerStep * kNumExpertsPerTok));
  t.h_gating_indices = arena_.Reserve("moe.gating_indices", Int32Bytes(kTokensPerStep * kNumExpertsPerTok));
  t.h_local_indices = arena_.Reserve("moe.local_indices", Int32Bytes(kTokensPerStep * kNumExpertsPerTok));
  arena_.Reserve("moe.expanded_x", Fp8Bytes(expanded_rows * kHiddenSize));
  arena_.Reserve("moe.expanded_row_idx", Int32Bytes(expanded_rows));
  arena_.Reserve("moe.expanded_scale", Fp8Bytes(expanded_rows * MxScaleCols(kHiddenSize)));
  arena_.Reserve("moe.expanded_weights", Fp32Bytes(expanded_rows));
  t.h_group_list = arena_.Reserve("moe.group_list", Int64Bytes(kNumExpertsPerTok));
  arena_.Reserve("moe.gemm1_raw", Bf16Bytes(expanded_rows * 2 * kMoeIntermediateSize));
  arena_.Reserve("moe.gemm1_out", Fp8Bytes(expanded_rows * kMoeIntermediateSize));
  arena_.Reserve("moe.gemm1_scale", Fp8Bytes(expanded_rows * MxScaleCols(kMoeIntermediateSize)));
  arena_.Reserve("moe.gemm2_out", Bf16Bytes(expanded_rows * kHiddenSize));
  arena_.Reserve("moe.routed_out", Bf16Bytes(kTokensPerStep * kHiddenSize));

  arena_.Reserve("shared.gate_up", Bf16Bytes(kTokensPerStep * 2 * shared_inter));
  arena_.Reserve("shared.act", Bf16Bytes(kTokensPerStep * shared_inter));
  arena_.Reserve("shared.act_fp8", Fp8Bytes(kTokensPerStep * shared_inter));
  arena_.Reserve("shared.act_mx_scale", Fp8Bytes(kTokensPerStep * MxScaleCols(shared_inter)));
  arena_.Reserve("shared.out", Bf16Bytes(kTokensPerStep * kHiddenSize));

  arena_.Reserve("head.final_normed", Bf16Bytes(kTokensPerStep * kHiddenSize));
  arena_.Reserve("head.logits", Bf16Bytes(kTokensPerStep * kVocabSize));
  t.h_argmax = arena_.Reserve("head.argmax", Int64Bytes(kTokensPerStep));

  backbone_->block_table = arena_.Reserve("kv.block_table", Int32Bytes(kTokensPerStep * num_blocks_));
  t.h_slot_mapping = arena_.Reserve("kv.slot_mapping", Int32Bytes(kTokensPerStep));
  backbone_->slot_mapping = t.h_slot_mapping;
}

void Dsv4Pipeline::ReserveBackbone() {
  const MlaGeometry& mla = config_.mla;
  const int64_t kv_row = mla.kv_row_elements();
  const int64_t q_b_width = heads_ * kv_row;
  const int64_t o_input = heads_ * mla.kv_lora_rank;
  const int64_t shared_inter = kMoeIntermediateSize * kNumSharedExperts;

  backbone_->embed_tokens = arena_.Reserve("w.embed_tokens", Bf16Bytes(kVocabSize * kHiddenSize));
  backbone_->lm_head = arena_.Reserve("w.lm_head", Bf16Bytes(kVocabSize * kHiddenSize));
  backbone_->final_norm = arena_.Reserve("w.final_norm", Bf16Bytes(kHiddenSize));
  backbone_->rope_cos = arena_.Reserve("w.rope_cos", Bf16Bytes(config_.max_context_len * mla.qk_rope_head_dim));
  backbone_->rope_sin = arena_.Reserve("w.rope_sin", Bf16Bytes(config_.max_context_len * mla.qk_rope_head_dim));
  backbone_->kv_latent_cache = arena_.Reserve(
      "kv.latent_cache", Bf16Bytes(kNumLayers * num_blocks_ * config_.block_size * mla.kv_lora_rank));
  backbone_->kv_rope_cache = arena_.Reserve(
      "kv.rope_cache", Bf16Bytes(kNumLayers * num_blocks_ * config_.block_size * mla.qk_rope_head_dim));

  for (int64_t index = 0; index < kNumLayers; ++index) {
    Backbone::Layer& layer = backbone_->layers[static_cast<size_t>(index)];
    layer.input_norm = arena_.Reserve("w.layer.input_norm", Bf16Bytes(kHiddenSize));
    layer.q_a_weight = arena_.Reserve("w.layer.q_a", Fp8Bytes(kQLoraRank * kHiddenSize));
    layer.q_a_scale = arena_.Reserve("w.layer.q_a_scale", Fp8Bytes(kQLoraRank * DenseScaleCols(kHiddenSize)));
    layer.q_a_norm = arena_.Reserve("w.layer.q_a_norm", Bf16Bytes(kQLoraRank));
    layer.q_b_weight = arena_.Reserve("w.layer.q_b_folded", Fp8Bytes(q_b_width * kQLoraRank));
    layer.q_b_scale = arena_.Reserve("w.layer.q_b_scale", Fp8Bytes(q_b_width * DenseScaleCols(kQLoraRank)));
    layer.kv_a_weight = arena_.Reserve("w.layer.kv_a", Fp8Bytes(kv_row * kHiddenSize));
    layer.kv_a_scale = arena_.Reserve("w.layer.kv_a_scale", Fp8Bytes(kv_row * DenseScaleCols(kHiddenSize)));
    layer.kv_a_norm = arena_.Reserve("w.layer.kv_a_norm", Bf16Bytes(mla.kv_lora_rank));
    layer.o_weight = arena_.Reserve("w.layer.o_folded", Fp8Bytes(kHiddenSize * o_input));
    layer.o_scale = arena_.Reserve("w.layer.o_scale", Fp8Bytes(kHiddenSize * DenseScaleCols(o_input)));
    layer.post_norm = arena_.Reserve("w.layer.post_norm", Bf16Bytes(kHiddenSize));
    layer.router_weight = arena_.Reserve("w.layer.router", Bf16Bytes(kNumRoutedExperts * kHiddenSize));
    layer.router_bias = arena_.Reserve("w.layer.router_bias", Fp32Bytes(kNumRoutedExperts));
    layer.shared_gate_up_weight =
        arena_.Reserve("w.layer.shared_gate_up", Fp8Bytes(2 * shared_inter * kHiddenSize));
    layer.shared_gate_up_scale =
        arena_.Reserve("w.layer.shared_gate_up_scale", Fp8Bytes(2 * shared_inter * DenseScaleCols(kHiddenSize)));
    layer.shared_down_weight = arena_.Reserve("w.layer.shared_down", Fp8Bytes(kHiddenSize * shared_inter));
    layer.shared_down_scale =
        arena_.Reserve("w.layer.shared_down_scale", Fp8Bytes(kHiddenSize * DenseScaleCols(shared_inter)));
  }
}

void Dsv4Pipeline::IngestBackbone(WeightByteSource& source) {
  struct Binding {
    ArenaHandle handle;
    const char* pattern;
  };

  // Streamed through a bounded host staging buffer. Coupling this to the expert
  // manager's pinned transit chunk would give that chunk two owners; this is
  // init-only work, so a plain heap buffer is the right trade.
  std::vector<uint8_t> staging(kTransferChunkBytes);

  auto ingest = [&](ArenaHandle handle, const std::string& name, bool required) -> bool {
    const size_t bytes = arena_.Bytes(handle);
    if (!source.HasNamed(name)) {
      DSV4_REQUIRE(!required,
                   "the checkpoint has no tensor '"
                       << name
                       << "'. If that is q_b_proj_latent or o_proj_folded, the checkpoint has not been through "
                          "the MLA absorption pass: this runner consumes q_b with W_UK folded in and o_proj with "
                          "W_UV folded in, because folding block-scaled FP8 at startup would requantize and lose "
                          "accuracy without saying so.");
      return false;
    }
    const size_t available = source.NamedByteSize(name);
    DSV4_REQUIRE(available == 0 || available == bytes,
                 "tensor '" << name << "' holds " << available << " bytes, the arena reserved " << bytes);
    uint8_t* destination = arena_.AddressAs<uint8_t>(handle);
    for (size_t offset = 0; offset < bytes; offset += kTransferChunkBytes) {
      const size_t count = std::min(kTransferChunkBytes, bytes - offset);
      source.ReadNamed(name, staging.data(), staging.size(), offset, count);
      device_.MemcpySync(destination + offset, bytes - offset, staging.data(), count, MemcpyKind::kHostToDevice);
    }
    return true;
  };

  ingest(backbone_->embed_tokens, "model.embed_tokens.weight", true);
  ingest(backbone_->lm_head, "lm_head.weight", true);
  ingest(backbone_->final_norm, "model.norm.weight", true);

  for (int64_t index = 0; index < kNumLayers; ++index) {
    const Backbone::Layer& layer = backbone_->layers[static_cast<size_t>(index)];
    const Binding bindings[] = {
        {layer.input_norm, "model.layers.{L}.input_layernorm.weight"},
        {layer.q_a_weight, "model.layers.{L}.self_attn.q_a_proj.weight"},
        {layer.q_a_scale, "model.layers.{L}.self_attn.q_a_proj.weight_scale_inv"},
        {layer.q_a_norm, "model.layers.{L}.self_attn.q_a_layernorm.weight"},
        {layer.q_b_weight, kFoldedQName},
        {layer.q_b_scale, kFoldedQScaleName},
        {layer.kv_a_weight, "model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight"},
        {layer.kv_a_scale, "model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight_scale_inv"},
        {layer.kv_a_norm, "model.layers.{L}.self_attn.kv_a_layernorm.weight"},
        {layer.o_weight, kFoldedOName},
        {layer.o_scale, kFoldedOScaleName},
        {layer.post_norm, "model.layers.{L}.post_attention_layernorm.weight"},
        {layer.router_weight, "model.layers.{L}.mlp.gate.weight"},
        {layer.router_bias, "model.layers.{L}.mlp.gate.e_score_correction_bias"},
        {layer.shared_gate_up_weight, "model.layers.{L}.mlp.shared_experts.gate_up_proj.weight"},
        {layer.shared_gate_up_scale, "model.layers.{L}.mlp.shared_experts.gate_up_proj.weight_scale_inv"},
        {layer.shared_down_weight, "model.layers.{L}.mlp.shared_experts.down_proj.weight"},
        {layer.shared_down_scale, "model.layers.{L}.mlp.shared_experts.down_proj.weight_scale_inv"},
    };
    for (const Binding& binding : bindings) {
      ingest(binding.handle, LayerTensorName(binding.pattern, index), true);
    }
  }

  // The rope tables are derived, not stored. A checkpoint that ships them is
  // honoured; otherwise they stay zeroed and the report says the rope tables
  // were not populated, because attending with cos=0 is a wrong answer with no
  // symptom.
  const bool cos = ingest(backbone_->rope_cos, "model.rotary_emb.cos_cached", false);
  const bool sin = ingest(backbone_->rope_sin, "model.rotary_emb.sin_cached", false);
  backbone_->rope_tables_populated = cos && sin;
}

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

void Dsv4Pipeline::CreateDescriptors() {
  const MlaGeometry& mla = config_.mla;
  const int64_t kv_lora = mla.kv_lora_rank;
  const int64_t rope = mla.qk_rope_head_dim;
  const int64_t kv_row = mla.kv_row_elements();
  const int64_t q_b_width = heads_ * kv_row;
  const int64_t o_input = heads_ * kv_lora;
  const int64_t shared_inter = kMoeIntermediateSize * kNumSharedExperts;
  const int64_t expanded_rows = kTokensPerStep * kNumExpertsPerTok;
  Tensors& t = *tensors_;
  Backbone& b = *backbone_;

  // Reservations are looked up by name here rather than threaded through as
  // handles: every buffer is created exactly once and the name is the same
  // string literal the reservation used.
  auto address = [&](const char* name) -> void* {
    for (size_t index = 0; index < arena_.reservations().size(); ++index) {
      if (std::strcmp(arena_.reservations()[index].name, name) == 0) {
        return arena_.Address(index);
      }
    }
    throw Dsv4Error(std::string("no arena reservation named ") + name);
  };

  uint8_t* q_b_base = static_cast<uint8_t*>(address("act.q_b"));
  uint8_t* kv_a_base = static_cast<uint8_t*>(address("act.kv_a"));

  // --- activations -------------------------------------------------------
  t.hidden = arena_.CreateTensor("hidden", {kTokensPerStep, kHiddenSize}, kAclBf16, address("act.hidden"));
  t.normed = arena_.CreateTensor("normed", {kTokensPerStep, kHiddenSize}, kAclBf16, address("act.normed"));
  t.normed_fp8 =
      arena_.CreateTensor("normed_fp8", {kTokensPerStep, kHiddenSize}, kAclFloat8E4m3Fn, address("act.normed_fp8"));
  t.normed_mx_scale = arena_.CreateTensor("normed_mx_scale", {kTokensPerStep, MxScaleCols(kHiddenSize)}, kScaleDtype,
                                          address("act.normed_mx_scale"));
  t.rstd = arena_.CreateTensor("rstd", {kTokensPerStep, 1}, kAclFloat32, address("act.rstd"));
  t.q_a = arena_.CreateTensor("q_a", {kTokensPerStep, kQLoraRank}, kAclBf16, address("act.q_a"));
  t.q_a_fp8 = arena_.CreateTensor("q_a_fp8", {kTokensPerStep, kQLoraRank}, kAclFloat8E4m3Fn, address("act.q_a_fp8"));
  t.q_a_mx_scale = arena_.CreateTensor("q_a_mx_scale", {kTokensPerStep, MxScaleCols(kQLoraRank)}, kScaleDtype,
                                       address("act.q_a_mx_scale"));
  t.q_b = arena_.CreateTensor("q_b", {kTokensPerStep, q_b_width}, kAclBf16, q_b_base);
  // q_b is laid out per head as [kv_lora | rope], so the latent and the rope
  // slice are views at a stride of kv_row, not contiguous halves.
  t.q_latent = arena_.CreateTensor("q_latent", {kTokensPerStep, heads_, kv_lora}, kAclBf16, q_b_base);
  t.q_rope =
      arena_.CreateTensor("q_rope", {kTokensPerStep, heads_, rope}, kAclBf16, q_b_base + Bf16Bytes(kv_lora));
  t.kv_a = arena_.CreateTensor("kv_a", {kTokensPerStep, kv_row}, kAclBf16, kv_a_base);
  t.kv_latent = arena_.CreateTensor("kv_latent", {kTokensPerStep, kv_lora}, kAclBf16, kv_a_base);
  t.k_rope = arena_.CreateTensor("k_rope", {kTokensPerStep, 1, rope}, kAclBf16, kv_a_base + Bf16Bytes(kv_lora));
  t.kv_latent_normed =
      arena_.CreateTensor("kv_latent_normed", {kTokensPerStep, kv_lora}, kAclBf16, address("act.kv_latent_normed"));
  t.attn_out = arena_.CreateTensor("attn_out", {kTokensPerStep, heads_, kv_lora}, kAclBf16, address("act.attn_out"));
  t.attn_flat = arena_.CreateTensor("attn_flat", {kTokensPerStep, o_input}, kAclBf16, address("act.attn_out"));
  t.attn_fp8 = arena_.CreateTensor("attn_fp8", {kTokensPerStep, o_input}, kAclFloat8E4m3Fn, address("act.attn_fp8"));
  t.attn_mx_scale = arena_.CreateTensor("attn_mx_scale", {kTokensPerStep, MxScaleCols(o_input)}, kScaleDtype,
                                        address("act.attn_mx_scale"));
  t.softmax_lse =
      arena_.CreateTensor("softmax_lse", {kTokensPerStep, heads_, 1}, kAclFloat32, address("act.softmax_lse"));
  t.proj_out = arena_.CreateTensor("proj_out", {kTokensPerStep, kHiddenSize}, kAclBf16, address("act.proj_out"));

  // --- routing -----------------------------------------------------------
  t.router_logits = arena_.CreateTensor("router_logits", {kTokensPerStep, kNumRoutedExperts}, kAclFloat32,
                                        address("moe.router_logits"));
  t.gating_weights = arena_.CreateTensor("gating_weights", {kTokensPerStep, kNumExpertsPerTok}, kAclFloat32,
                                         address("moe.gating_weights"));
  t.gating_indices = arena_.CreateTensor("gating_indices", {kTokensPerStep, kNumExpertsPerTok}, kAclInt32,
                                         address("moe.gating_indices"));
  t.local_indices = arena_.CreateTensor("local_indices", {kTokensPerStep, kNumExpertsPerTok}, kAclInt32,
                                        address("moe.local_indices"));
  t.expanded_x =
      arena_.CreateTensor("expanded_x", {expanded_rows, kHiddenSize}, kAclFloat8E4m3Fn, address("moe.expanded_x"));
  t.expanded_row_idx =
      arena_.CreateTensor("expanded_row_idx", {expanded_rows}, kAclInt32, address("moe.expanded_row_idx"));
  t.expanded_scale = arena_.CreateTensor("expanded_scale", {expanded_rows, MxScaleCols(kHiddenSize)}, kScaleDtype,
                                         address("moe.expanded_scale"));
  t.expanded_weights =
      arena_.CreateTensor("expanded_weights", {expanded_rows}, kAclFloat32, address("moe.expanded_weights"));
  t.expanded_weights_row = arena_.CreateTensor("expanded_weights_row", {kTokensPerStep, expanded_rows},
                                               kAclFloat32, address("moe.expanded_weights"));
  t.group_list = arena_.CreateTensor("group_list", {kNumExpertsPerTok}, kAclInt64, address("moe.group_list"));
  t.gemm1_raw = arena_.CreateTensor("gemm1_raw", {expanded_rows, 2 * kMoeIntermediateSize}, kAclBf16,
                                    address("moe.gemm1_raw"));
  t.gemm1_out = arena_.CreateTensor("gemm1_out", {expanded_rows, kMoeIntermediateSize}, kAclFloat8E4m3Fn,
                                    address("moe.gemm1_out"));
  t.gemm1_scale = arena_.CreateTensor("gemm1_scale", {expanded_rows, MxScaleCols(kMoeIntermediateSize)},
                                      kScaleDtype, address("moe.gemm1_scale"));
  t.gemm2_out = arena_.CreateTensor("gemm2_out", {expanded_rows, kHiddenSize}, kAclBf16, address("moe.gemm2_out"));
  t.routed_out =
      arena_.CreateTensor("routed_out", {kTokensPerStep, kHiddenSize}, kAclBf16, address("moe.routed_out"));
  t.gemm1_x_list = arena_.CreateTensorList("gemm1_x_list", {t.expanded_x});
  t.gemm1_x_scale_list = arena_.CreateTensorList("gemm1_x_scale_list", {t.expanded_scale});
  t.gemm1_raw_out_list = arena_.CreateTensorList("gemm1_raw_out_list", {t.gemm1_raw});
  t.gemm2_x_list = arena_.CreateTensorList("gemm2_x_list", {t.gemm1_out});
  t.gemm2_x_scale_list = arena_.CreateTensorList("gemm2_x_scale_list", {t.gemm1_scale});
  t.gemm2_out_list = arena_.CreateTensorList("gemm2_out_list", {t.gemm2_out});

  // --- shared expert -----------------------------------------------------
  t.shared_gate_up =
      arena_.CreateTensor("shared_gate_up", {kTokensPerStep, 2 * shared_inter}, kAclBf16, address("shared.gate_up"));
  t.shared_act = arena_.CreateTensor("shared_act", {kTokensPerStep, shared_inter}, kAclBf16, address("shared.act"));
  t.shared_act_fp8 = arena_.CreateTensor("shared_act_fp8", {kTokensPerStep, shared_inter}, kAclFloat8E4m3Fn,
                                         address("shared.act_fp8"));
  t.shared_act_scale = arena_.CreateTensor("shared_act_scale", {kTokensPerStep, MxScaleCols(shared_inter)},
                                           kScaleDtype, address("shared.act_mx_scale"));
  t.shared_out = arena_.CreateTensor("shared_out", {kTokensPerStep, kHiddenSize}, kAclBf16, address("shared.out"));

  // --- head --------------------------------------------------------------
  t.final_normed =
      arena_.CreateTensor("final_normed", {kTokensPerStep, kHiddenSize}, kAclBf16, address("head.final_normed"));
  t.logits = arena_.CreateTensor("logits", {kTokensPerStep, kVocabSize}, kAclBf16, address("head.logits"));
  t.argmax = arena_.CreateTensor("argmax", {kTokensPerStep}, kAclInt64, address("head.argmax"));

  // --- paged KV ----------------------------------------------------------
  // One cache per layer, carved out of one reservation. FIA V5 reads the
  // compressed latent through `key`/`value` and the rope slice through
  // `keyRope`, which is the native MLA input set.
  uint8_t* latent_cache_base = arena_.AddressAs<uint8_t>(b.kv_latent_cache);
  uint8_t* rope_cache_base = arena_.AddressAs<uint8_t>(b.kv_rope_cache);
  t.kv_latent_cache = arena_.CreateTensor("kv_latent_cache", {num_blocks_, config_.block_size, 1, kv_lora},
                                          kAclBf16, latent_cache_base);
  t.kv_rope_cache =
      arena_.CreateTensor("kv_rope_cache", {num_blocks_, config_.block_size, 1, rope}, kAclBf16, rope_cache_base);
  t.key_rope_cache_view = arena_.CreateTensor("key_rope_cache_view", {num_blocks_, config_.block_size, 1, rope},
                                              kAclBf16, rope_cache_base);
  t.key_list = arena_.CreateTensorList("fia_key_list", {t.kv_latent_cache});
  t.value_list = arena_.CreateTensorList("fia_value_list", {t.kv_latent_cache});
  t.block_table =
      arena_.CreateTensor("block_table", {kTokensPerStep, num_blocks_}, kAclInt32, arena_.Address(b.block_table));
  t.slot_mapping = arena_.CreateTensor("slot_mapping", {kTokensPerStep}, kAclInt32, arena_.Address(b.slot_mapping));
  // TND with one query token. The kv length is the reserved context: FIA
  // derives the live length from the block table, and `kv_padding_size` is left
  // null, so neither array changes per step and both are built once.
  t.actual_seq_q = arena_.CreateIntArray("actual_seq_q", {kTokensPerStep});
  t.actual_seq_kv = arena_.CreateIntArray("actual_seq_kv", {config_.max_context_len});

  t.rope_cos = arena_.CreateTensor("rope_cos", {kTokensPerStep, rope}, kAclBf16, arena_.Address(b.rope_cos));
  t.rope_sin = arena_.CreateTensor("rope_sin", {kTokensPerStep, rope}, kAclBf16, arena_.Address(b.rope_sin));

  // --- layer weights, pointed at layer 0 ---------------------------------
  const Backbone::Layer& first = b.layers[0];
  t.w_input_norm = arena_.CreateTensor("w_input_norm", {kHiddenSize}, kAclBf16, arena_.Address(first.input_norm));
  t.w_q_a =
      arena_.CreateTensor("w_q_a", {kQLoraRank, kHiddenSize}, kAclFloat8E4m3Fn, arena_.Address(first.q_a_weight));
  t.w_q_a_scale = arena_.CreateTensor("w_q_a_scale", {kQLoraRank, DenseScaleCols(kHiddenSize)}, kScaleDtype,
                                      arena_.Address(first.q_a_scale));
  t.w_q_a_norm = arena_.CreateTensor("w_q_a_norm", {kQLoraRank}, kAclBf16, arena_.Address(first.q_a_norm));
  t.w_q_b =
      arena_.CreateTensor("w_q_b", {q_b_width, kQLoraRank}, kAclFloat8E4m3Fn, arena_.Address(first.q_b_weight));
  t.w_q_b_scale = arena_.CreateTensor("w_q_b_scale", {q_b_width, DenseScaleCols(kQLoraRank)}, kScaleDtype,
                                      arena_.Address(first.q_b_scale));
  t.w_kv_a =
      arena_.CreateTensor("w_kv_a", {kv_row, kHiddenSize}, kAclFloat8E4m3Fn, arena_.Address(first.kv_a_weight));
  t.w_kv_a_scale = arena_.CreateTensor("w_kv_a_scale", {kv_row, DenseScaleCols(kHiddenSize)}, kScaleDtype,
                                       arena_.Address(first.kv_a_scale));
  t.w_kv_a_norm = arena_.CreateTensor("w_kv_a_norm", {kv_lora}, kAclBf16, arena_.Address(first.kv_a_norm));
  t.w_o = arena_.CreateTensor("w_o", {kHiddenSize, o_input}, kAclFloat8E4m3Fn, arena_.Address(first.o_weight));
  t.w_o_scale = arena_.CreateTensor("w_o_scale", {kHiddenSize, DenseScaleCols(o_input)}, kScaleDtype,
                                    arena_.Address(first.o_scale));
  t.w_post_norm = arena_.CreateTensor("w_post_norm", {kHiddenSize}, kAclBf16, arena_.Address(first.post_norm));
  t.w_router = arena_.CreateTensor("w_router", {kNumRoutedExperts, kHiddenSize}, kAclBf16,
                                   arena_.Address(first.router_weight));
  t.w_router_bias =
      arena_.CreateTensor("w_router_bias", {kNumRoutedExperts}, kAclFloat32, arena_.Address(first.router_bias));
  t.w_shared_gate_up = arena_.CreateTensor("w_shared_gate_up", {2 * shared_inter, kHiddenSize}, kAclFloat8E4m3Fn,
                                           arena_.Address(first.shared_gate_up_weight));
  t.w_shared_gate_up_scale =
      arena_.CreateTensor("w_shared_gate_up_scale", {2 * shared_inter, DenseScaleCols(kHiddenSize)}, kScaleDtype,
                          arena_.Address(first.shared_gate_up_scale));
  t.w_shared_down = arena_.CreateTensor("w_shared_down", {kHiddenSize, shared_inter}, kAclFloat8E4m3Fn,
                                        arena_.Address(first.shared_down_weight));
  t.w_shared_down_scale =
      arena_.CreateTensor("w_shared_down_scale", {kHiddenSize, DenseScaleCols(shared_inter)}, kScaleDtype,
                          arena_.Address(first.shared_down_scale));
  t.w_final_norm = arena_.CreateTensor("w_final_norm", {kHiddenSize}, kAclBf16, arena_.Address(b.final_norm));
  t.w_lm_head = arena_.CreateTensor("w_lm_head", {kVocabSize, kHiddenSize}, kAclBf16, arena_.Address(b.lm_head));

  // --- the six active experts -------------------------------------------
  // Views into the exclusive manager's HBM slot pool, repointed every layer by
  // aclSetDynamicTensorAddr. They start pointed at slots 0..5 so the plan phase
  // sees real, resident addresses.
  const ExpertSlotLayout& slots = experts_.layout();
  const ExpertRegionSpec& gate_up = slots.region(ExpertRegionId::kGateUpWeight);
  const ExpertRegionSpec& gate_up_scale = slots.region(ExpertRegionId::kGateUpScale);
  const ExpertRegionSpec& down = slots.region(ExpertRegionId::kDownWeight);
  const ExpertRegionSpec& down_scale = slots.region(ExpertRegionId::kDownScale);
  for (int64_t index = 0; index < kNumExpertsPerTok; ++index) {
    const int32_t slot = static_cast<int32_t>(index);
    t.expert_gate_up.push_back(arena_.CreateFp4Tensor("expert_gate_up", {gate_up.rows, gate_up.cols},
                                                      experts_.RegionAddress(slot, ExpertRegionId::kGateUpWeight)));
    t.expert_gate_up_scale.push_back(arena_.CreateTensor(
        "expert_gate_up_scale", {gate_up_scale.rows, static_cast<int64_t>(gate_up_scale.stored_cols())},
        kScaleDtype, experts_.RegionAddress(slot, ExpertRegionId::kGateUpScale)));
    t.expert_down.push_back(arena_.CreateFp4Tensor("expert_down", {down.rows, down.cols},
                                                   experts_.RegionAddress(slot, ExpertRegionId::kDownWeight)));
    t.expert_down_scale.push_back(arena_.CreateTensor(
        "expert_down_scale", {down_scale.rows, static_cast<int64_t>(down_scale.stored_cols())}, kScaleDtype,
        experts_.RegionAddress(slot, ExpertRegionId::kDownScale)));
  }
  t.expert_gate_up_list = arena_.CreateTensorList("expert_gate_up_list", t.expert_gate_up);
  t.expert_gate_up_scale_list = arena_.CreateTensorList("expert_gate_up_scale_list", t.expert_gate_up_scale);
  t.expert_down_list = arena_.CreateTensorList("expert_down_list", t.expert_down);
  t.expert_down_scale_list = arena_.CreateTensorList("expert_down_scale_list", t.expert_down_scale);
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

PipelineStage& Dsv4Pipeline::stage(const char* name) {
  for (size_t i = 0; i < stage_count_; ++i) {
    PipelineStage& entry = stages_[i];
    if (std::strcmp(entry.name, name) == 0) {
      return entry;
    }
  }
  throw Dsv4Error(std::string("no pipeline stage named ") + name);
}

const PipelineStage& Dsv4Pipeline::stage(const char* name) const {
  for (size_t i = 0; i < stage_count_; ++i) {
    const PipelineStage& entry = stages_[i];
    if (std::strcmp(entry.name, name) == 0) {
      return entry;
    }
  }
  throw Dsv4Error(std::string("no pipeline stage named ") + name);
}

void Dsv4Pipeline::PlanStages() {
  const MlaGeometry& mla = config_.mla;
  Tensors& t = *tensors_;

  // Every required op must exist before anything is planned, so a toolkit gap
  // is one clear message rather than a failure part-way through the graph.
  std::vector<OpId> required = {
      OpId::kRmsNorm,       OpId::kRmsNormDynamicMxQuant, OpId::kDynamicMxQuant,
      OpId::kMatmul,        OpId::kQuantMatmulV5,         OpId::kApplyRotaryPosEmbV2,
      OpId::kScatterPaKvCache, OpId::kFusedInferAttentionScoreV5, OpId::kInplaceAdd,
      OpId::kSwiGlu,        OpId::kArgMax,                OpId::kMoeGatingTopKV2,
      OpId::kMoeInitRoutingV4, OpId::kGroupedMatmulV5,
  };
  if (config_.moe_path == MoePath::kFused) {
    required.push_back(OpId::kGroupedMatmulSwigluQuantV2);
  } else {
    required.push_back(OpId::kSwigluMxQuant);
  }
  ops_.RequireAll(required);

  // The stage vector is born at full capacity (StaticOpSlot is not movable,
  // see stages_ in the header); planning only fills slots in place.
  auto add = [&](const char* name, OpId op) -> PipelineStage& {
    DSV4_REQUIRE(stage_count_ < kMaxPipelineStages, "pipeline stage capacity exceeded at " << name);
    PipelineStage& entry = stages_[stage_count_++];
    entry.name = name;
    entry.op = op;
    return entry;
  };
  auto adopt = [&](PipelineStage& entry, uint64_t workspace, aclOpExecutor* executor) {
    arena_.NoteWorkspace(workspace);
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
    arena_.NoteWorkspace(workspace);
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
      arena_.NoteWorkspace(upper_bound);
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
  // 15. router logits.
  {
    PipelineStage& entry = add("router", OpId::kMatmul);
    const uint64_t workspace = PlanAclnnOp<MatmulPlanFn>(ops_, entry.op, &executor, t.normed, t.w_router,
                                                         t.router_logits, kCubeMathTypeKeepDtype);
    adopt(entry, workspace, executor);
  }
  // 16. noaux_tc gating: bias shifts selection only, scaling 1.5 applied here.
  {
    PipelineStage& entry = add("gating", OpId::kMoeGatingTopKV2);
    const uint64_t workspace = PlanAclnnOp<MoeGatingTopKV2PlanFn>(
        ops_, entry.op, &executor, t.router_logits, t.w_router_bias, nullptr, nullptr, kNumExpertsPerTok,
        kGatingKGroup, kGatingGroupCount, kGatingGroupSelectMode, kGatingRenormOff, config_.gating_norm_type,
        false, kRoutedScalingFactor, kGatingEps, t.gating_weights, t.gating_indices, nullptr);
    adopt(entry, workspace, executor);
  }
  // 17. dropless dispatch over the SIX locally renumbered experts. The expert
  //     ids fed in are 0..5, not the global 0..255: the weight list has six
  //     entries, so the device cumsum must have six groups to match it. The
  //     renumbering is free -- the host already holds the global ids, because
  //     the exclusive swap engine needed them to decide what to promote.
  {
    PipelineStage& entry = add("routing", OpId::kMoeInitRoutingV4);
    const uint64_t workspace = PlanAclnnOp<MoeInitRoutingV4PlanFn>(
        ops_, entry.op, &executor, t.normed_fp8, t.local_indices, t.normed_mx_scale, nullptr, nullptr,
        t.gating_weights, 0, kNumExpertsPerTok, kRoutingDropless, kRoutingTokensNumCumsum, true,
        kRoutingQuantModeNone, nullptr, kRoutingRowIdxGather, t.expanded_x, t.expanded_row_idx, t.group_list,
        t.expanded_scale, t.expanded_weights);
    adopt(entry, workspace, executor);
  }
  // 18. expert GEMM 1. Fused path: GEMM + clamped SwiGLU (limit 10.0) + MX
  //     requant in one op.
  if (config_.moe_path == MoePath::kFused) {
    PipelineStage& entry = add("expert_gemm1", OpId::kGroupedMatmulSwigluQuantV2);
    const uint64_t workspace = PlanAclnnOp<GroupedMatmulSwigluQuantV2PlanFn>(
        ops_, entry.op, &executor, t.expanded_x, t.expert_gate_up_list, t.expert_gate_up_scale_list, nullptr,
        nullptr, t.expanded_scale, nullptr, t.group_list, kGmmDequantModeMx,
        static_cast<int64_t>(kAclFloat8E4m3Fn), kGmmDequantModeMx, kGmmGroupListTypeCumsum, nullptr, t.gemm1_out,
        t.gemm1_scale);
    adopt(entry, workspace, executor);
  } else {
    PipelineStage& entry = add("expert_gemm1", OpId::kGroupedMatmulV5);
    const uint64_t workspace = PlanAclnnOp<GroupedMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.gemm1_x_list, t.expert_gate_up_list, nullptr, t.expert_gate_up_scale_list,
        nullptr, nullptr, nullptr, t.gemm1_x_scale_list, t.group_list, nullptr, nullptr, nullptr,
        kGmmSplitItemSingleOut, kGmmGroupTypeM, kGmmGroupListTypeCumsum, kGmmActTypeNone, nullptr,
        t.gemm1_raw_out_list, nullptr, nullptr);
    adopt(entry, workspace, executor);

    PipelineStage& activation = add("expert_swiglu", OpId::kSwigluMxQuant);
    const uint64_t act_workspace = PlanAclnnOp<SwigluMxQuantPlanFn>(
        ops_, activation.op, &executor, t.gemm1_raw, nullptr, kSwigluActivateDimLast, true, kSwigluModeDefault,
        kSwigluLimit, kSwigluGluAlpha, kSwigluGluBias, kSwigluGroupModeNone, kSwigluAxisLast,
        static_cast<int64_t>(kAclFloat8E4m3Fn), kMxRoundModeRint, kSwigluScaleAlgOcp, kSwigluMaxDtypeValue,
        t.gemm1_out, t.gemm1_scale);
    adopt(activation, act_workspace, executor);
  }
  // 19. expert GEMM 2. A tensor-LIST weight, which is what lets the six experts
  //     sit in six scattered HBM slots. `aclnnGroupedMatmulFinalizeRoutingV3`
  //     would fuse the combine, but its x2 is a single tensor, i.e. the experts
  //     must be contiguous -- which an exclusive LRU slot pool cannot promise.
  {
    PipelineStage& entry = add("expert_gemm2", OpId::kGroupedMatmulV5);
    const uint64_t workspace = PlanAclnnOp<GroupedMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.gemm2_x_list, t.expert_down_list, nullptr, t.expert_down_scale_list, nullptr,
        nullptr, nullptr, t.gemm2_x_scale_list, t.group_list, nullptr, nullptr, nullptr, kGmmSplitItemSingleOut,
        kGmmGroupTypeM, kGmmGroupListTypeCumsum, kGmmActTypeNone, nullptr, t.gemm2_out_list, nullptr, nullptr);
    adopt(entry, workspace, executor);
  }
  // 20. routing combine. For a single-token step the six expanded rows are the
  //     same token, so the weighted sum IS a [1, 6] x [6, hidden] matmul over
  //     the permuted routing weights the dispatch emitted. Exact, and both ops
  //     have an ascend950 kernel. This is the one stage that does not
  //     generalize past kTokensPerStep == 1: a batched decode needs a
  //     scatter-add by `expanded_row_idx`.
  {
    PipelineStage& entry = add("expert_combine", OpId::kMatmul);
    const uint64_t workspace = PlanAclnnOp<MatmulPlanFn>(ops_, entry.op, &executor, t.expanded_weights_row,
                                                         t.gemm2_out, t.routed_out, kCubeMathTypeKeepDtype);
    adopt(entry, workspace, executor);
  }
  // 21-24. the shared expert: every token uses it, so it is never routed.
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
  // 25-26. routed + shared, then the MoE residual.
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
  // 27-29. the head.
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
  entry.slot.Launch(ops_, arena_.workspace(), compute_stream_);
  ++counters_.launches;
}

void Dsv4Pipeline::RunAttention(int32_t layer, int64_t position) {
  const MlaGeometry& mla = config_.mla;
  Tensors& t = *tensors_;
  const Backbone::Layer& weights = backbone_->layers[static_cast<size_t>(layer)];

  // Repoint this layer's weights. The descriptors and the executors are the
  // ones planned at init; only the addresses move.
  stage("input_norm_quant")
      .slot.SetAddress(slot::kMxQuantGamma, t.w_input_norm, arena_.Address(weights.input_norm));
  stage("q_a_proj").slot.SetAddress(slot::kQuantMmX2, t.w_q_a, arena_.Address(weights.q_a_weight));
  stage("q_a_proj").slot.SetAddress(slot::kQuantMmX2Scale, t.w_q_a_scale, arena_.Address(weights.q_a_scale));
  stage("q_a_norm_quant").slot.SetAddress(slot::kMxQuantGamma, t.w_q_a_norm, arena_.Address(weights.q_a_norm));
  stage("q_b_proj").slot.SetAddress(slot::kQuantMmX2, t.w_q_b, arena_.Address(weights.q_b_weight));
  stage("q_b_proj").slot.SetAddress(slot::kQuantMmX2Scale, t.w_q_b_scale, arena_.Address(weights.q_b_scale));
  stage("kv_a_proj").slot.SetAddress(slot::kQuantMmX2, t.w_kv_a, arena_.Address(weights.kv_a_weight));
  stage("kv_a_proj").slot.SetAddress(slot::kQuantMmX2Scale, t.w_kv_a_scale, arena_.Address(weights.kv_a_scale));
  stage("kv_a_norm").slot.SetAddress(slot::kRmsNormGamma, t.w_kv_a_norm, arena_.Address(weights.kv_a_norm));
  stage("o_proj").slot.SetAddress(slot::kQuantMmX2, t.w_o, arena_.Address(weights.o_weight));
  stage("o_proj").slot.SetAddress(slot::kQuantMmX2Scale, t.w_o_scale, arena_.Address(weights.o_scale));

  // This layer's slice of the paged cache.
  const size_t latent_stride = Bf16Bytes(num_blocks_ * config_.block_size * mla.kv_lora_rank);
  const size_t rope_stride = Bf16Bytes(num_blocks_ * config_.block_size * mla.qk_rope_head_dim);
  uint8_t* latent =
      arena_.AddressAs<uint8_t>(backbone_->kv_latent_cache) + latent_stride * static_cast<size_t>(layer);
  uint8_t* rope = arena_.AddressAs<uint8_t>(backbone_->kv_rope_cache) + rope_stride * static_cast<size_t>(layer);
  stage("kv_cache_write").slot.SetAddress(slot::kScatterKeyCache, t.kv_latent_cache, latent);
  stage("kv_cache_write").slot.SetAddress(slot::kScatterValueCache, t.kv_rope_cache, rope);
  stage("attention").slot.SetTensorListAddress(slot::kFiaKeyList, 0, t.key_list, latent);
  stage("attention").slot.SetTensorListAddress(slot::kFiaValueList, 0, t.value_list, latent);
  stage("attention").slot.SetAddress(slot::kFiaKeyRope, t.key_rope_cache_view, rope);

  // The rope table row for this position.
  uint8_t* cos_row = arena_.AddressAs<uint8_t>(backbone_->rope_cos) + Bf16Bytes(position * mla.qk_rope_head_dim);
  uint8_t* sin_row = arena_.AddressAs<uint8_t>(backbone_->rope_sin) + Bf16Bytes(position * mla.qk_rope_head_dim);
  stage("rope").slot.SetAddress(slot::kRotaryCos, t.rope_cos, cos_row);
  stage("rope").slot.SetAddress(slot::kRotarySin, t.rope_sin, sin_row);

  Launch(stage("input_norm_quant"));
  Launch(stage("q_a_proj"));
  Launch(stage("q_a_norm_quant"));
  Launch(stage("q_b_proj"));
  Launch(stage("kv_a_proj"));
  Launch(stage("kv_a_norm"));
  Launch(stage("rope"));
  Launch(stage("kv_cache_write"));
  Launch(stage("attention"));
  Launch(stage("attn_quant"));
  Launch(stage("o_proj"));
  Launch(stage("residual_attn"));
}

void Dsv4Pipeline::FetchRouting(int32_t layer) {
  (void)layer;
  Tensors& t = *tensors_;
  // THE ONE FORCED HOST ROUND TRIP (see the header's deviation 1). 24 bytes on
  // a dedicated stream, ordered after the gating launch by the compute event.
  device_.RecordEvent(compute_done_, compute_stream_);
  device_.StreamWaitEvent(readback_stream_, compute_done_);
  device_.MemcpyAsync(routing_mailbox_, Int32Bytes(kNumExpertsPerTok), arena_.Address(t.h_gating_indices),
                      Int32Bytes(kNumExpertsPerTok), MemcpyKind::kDeviceToHost, readback_stream_);
  device_.SynchronizeStream(readback_stream_);
  ++counters_.host_synchronizations;
}

void Dsv4Pipeline::BindExpertWeights(const LayerSwapPlan& plan) {
  Tensors& t = *tensors_;
  PipelineStage& gemm1 = stage("expert_gemm1");
  PipelineStage& gemm2 = stage("expert_gemm2");
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

void Dsv4Pipeline::RunSharedExpert(int32_t layer) {
  Tensors& t = *tensors_;
  const Backbone::Layer& weights = backbone_->layers[static_cast<size_t>(layer)];
  stage("shared_gate_up")
      .slot.SetAddress(slot::kQuantMmX2, t.w_shared_gate_up, arena_.Address(weights.shared_gate_up_weight));
  stage("shared_gate_up")
      .slot.SetAddress(slot::kQuantMmX2Scale, t.w_shared_gate_up_scale,
                       arena_.Address(weights.shared_gate_up_scale));
  stage("shared_down")
      .slot.SetAddress(slot::kQuantMmX2, t.w_shared_down, arena_.Address(weights.shared_down_weight));
  stage("shared_down")
      .slot.SetAddress(slot::kQuantMmX2Scale, t.w_shared_down_scale, arena_.Address(weights.shared_down_scale));
  Launch(stage("shared_gate_up"));
  Launch(stage("shared_act"));
  Launch(stage("shared_act_quant"));
  Launch(stage("shared_down"));
}

void Dsv4Pipeline::RunMoe(int32_t layer) {
  Tensors& t = *tensors_;
  const Backbone::Layer& weights = backbone_->layers[static_cast<size_t>(layer)];

  stage("post_norm_quant").slot.SetAddress(slot::kMxQuantGamma, t.w_post_norm, arena_.Address(weights.post_norm));
  stage("post_norm").slot.SetAddress(slot::kRmsNormGamma, t.w_post_norm, arena_.Address(weights.post_norm));
  stage("router").slot.SetAddress(slot::kMatmulMat2, t.w_router, arena_.Address(weights.router_weight));
  stage("gating").slot.SetAddress(slot::kGatingBias, t.w_router_bias, arena_.Address(weights.router_bias));

  Launch(stage("post_norm_quant"));
  Launch(stage("post_norm"));
  Launch(stage("router"));
  Launch(stage("gating"));

  FetchRouting(layer);

  // The exclusive hierarchy makes the six chosen experts resident, batching
  // every miss into one event-ordered duplex exchange before the GEMM.
  const LayerSwapPlan plan = experts_.PrepareLayer(layer, routing_mailbox_, static_cast<int32_t>(kNumExpertsPerTok),
                                                   compute_stream_, compute_done_);
  counters_.expert_slot_hits += static_cast<uint64_t>(plan.hit_count);
  counters_.expert_slot_misses += static_cast<uint64_t>(plan.miss_count);
  BindExpertWeights(plan);

  // Renumber the global expert ids to 0..5 so the dispatch's cumsum has exactly
  // as many groups as the weight list has entries.
  for (int64_t index = 0; index < kNumExpertsPerTok; ++index) {
    local_index_mailbox_[index] = static_cast<int32_t>(index);
  }
  device_.MemcpyAsync(arena_.Address(t.h_local_indices), Int32Bytes(kNumExpertsPerTok), local_index_mailbox_,
                      Int32Bytes(kNumExpertsPerTok), MemcpyKind::kHostToDevice, compute_stream_);

  Launch(stage("routing"));
  Launch(stage("expert_gemm1"));
  if (config_.moe_path == MoePath::kDecomposed) {
    Launch(stage("expert_swiglu"));
  }
  Launch(stage("expert_gemm2"));
  Launch(stage("expert_combine"));
  RunSharedExpert(layer);
  Launch(stage("add_shared"));
  Launch(stage("residual_moe"));

  // Publish the compute boundary: the next layer's swap batch waits on it
  // before overwriting any slot this layer's GEMMs read.
  device_.RecordEvent(compute_done_, compute_stream_);
  ++counters_.layers;
}

void Dsv4Pipeline::DecodeStep(int32_t token_id, int64_t position) {
  DSV4_REQUIRE(arena_.sealed(), "DecodeStep before Build()");
  DSV4_REQUIRE(token_id >= 0 && token_id < kVocabSize, "token id " << token_id << " outside the vocabulary");
  DSV4_REQUIRE(position >= 0 && position < config_.max_context_len,
               "position " << position << " outside the reserved context of " << config_.max_context_len);

  const uint64_t allocations_before = device_.counters().device_allocations;
  const size_t descriptors_before = arena_.descriptors().size();

  // Embedding lookup for a single greedy token is one row copy, not a gather
  // kernel: no descriptor, no workspace, no launch.
  Tensors& t = *tensors_;
  uint8_t* embed_row =
      arena_.AddressAs<uint8_t>(backbone_->embed_tokens) + Bf16Bytes(static_cast<int64_t>(token_id) * kHiddenSize);
  device_.MemcpyAsync(arena_.Address(t.h_hidden), Bf16Bytes(kHiddenSize), embed_row, Bf16Bytes(kHiddenSize),
                      MemcpyKind::kDeviceToDevice, compute_stream_);

  // The slot this token occupies in the paged cache. The pinned mailbox is the
  // DMA source, so the copy can be async without a stack lifetime problem.
  *slot_mailbox_ = static_cast<int32_t>(position);
  device_.MemcpyAsync(arena_.Address(t.h_slot_mapping), Int32Bytes(1), slot_mailbox_, Int32Bytes(1),
                      MemcpyKind::kHostToDevice, compute_stream_);

  for (int32_t layer = 0; layer < static_cast<int32_t>(kNumLayers); ++layer) {
    RunAttention(layer, position);
    RunMoe(layer);
  }
  Launch(stage("final_norm"));
  Launch(stage("lm_head"));
  Launch(stage("argmax"));

  ++counters_.steps;
  counters_.device_allocations_in_step += device_.counters().device_allocations - allocations_before;
  counters_.descriptors_built_in_step += arena_.descriptors().size() - descriptors_before;
}

int32_t Dsv4Pipeline::ReadArgmaxToken() {
  Tensors& t = *tensors_;
  // The one synchronization that ends a step.
  device_.MemcpyAsync(token_mailbox_, Int64Bytes(1), arena_.Address(t.h_argmax), Int64Bytes(1),
                      MemcpyKind::kDeviceToHost, compute_stream_);
  device_.SynchronizeStream(compute_stream_);
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
  out << "pipeline: " << stage_count_ << " planned stages, replayed " << kNumLayers
      << " times with address swaps (MoE path: "
      << (config_.moe_path == MoePath::kFused ? "fused" : "decomposed") << ")\n";
  uint64_t high_water = 0;
  for (size_t i = 0; i < stage_count_; ++i) {
    const PipelineStage& entry = stages_[i];
    out << "  " << std::left << std::setw(20) << entry.name << std::right << std::setw(40) << OpName(entry.op)
        << "  workspace=" << std::setw(10) << entry.slot.workspace_size() << "\n";
    high_water = std::max(high_water, entry.slot.workspace_size());
  }
  out << "  shared workspace high-water: " << high_water << " bytes\n";
  out << "  NOT APPLIED: the Lightning Indexer (index_topk=" << kIndexTopK
      << "). The specified attention op is the dense paged MLA decode; the sparse\n"
         "               selection belongs to aclnnSparseFlashMla, which the brief's mapping does not list.\n";
  out << "  UNVERIFIED: the router scoring function. aclnnMoeGatingTopKV2 normType=" << config_.gating_norm_type
      << ", and no header\n              documents which value is sqrtsoftplus. Needs a device A/B against a host "
         "reference.\n";
  out << "  UNVERIFIED: aclnnQuantMatmulV5 groupSize=" << config_.dense_group_size
      << " (the block-128 dense scale encoding).\n";
  if (!backbone_->rope_tables_populated) {
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

}  // namespace dsv4
}  // namespace vllm_ascend
