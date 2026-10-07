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

// StaticArenaManager: reservations, backbone ingestion and every descriptor
// the graph consumes. Extracted from Dsv4Pipeline so that memory shape and
// stage scheduling evolve independently (SRP); the pipeline orchestrates, it
// does not allocate.

#include "moe/pipeline/static_arena_manager.hpp"

#include <aclnn/acl_meta.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "moe/core/error.hpp"

namespace ascend_moe {
namespace {

// This runner decodes one token at a time (greedy, batch 1). Every buffer the
// manager reserves is sized for it.
constexpr int64_t kTokensPerStep = 1;

// Scale dtype for the dense block-128 weight scales and for the MX block-32
// activation scales: both are OCP E8M0, one byte per block.
constexpr int32_t kScaleDtype = kAclFloat8E8m0;

// The folded MLA projections this runner consumes. A real checkpoint ships
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

int64_t DivideUp(int64_t value, int64_t divisor) { return (value + divisor - 1) / divisor; }
size_t Fp8Bytes(int64_t elements) { return static_cast<size_t>(elements); }
size_t Bf16Bytes(int64_t elements) { return static_cast<size_t>(elements) * 2; }
size_t Fp32Bytes(int64_t elements) { return static_cast<size_t>(elements) * 4; }
size_t Int32Bytes(int64_t elements) { return static_cast<size_t>(elements) * 4; }
size_t Int64Bytes(int64_t elements) { return static_cast<size_t>(elements) * 8; }
int64_t DenseScaleCols(int64_t k) { return DivideUp(k, kDenseScaleBlock); }
int64_t MxScaleCols(int64_t k) { return DivideUp(k, kRoutedScaleBlock); }

StaticArenaManager::StaticArenaManager(IDeviceAllocator& allocator, IStreamEngine& streams, const RuntimeConfig& config)
    : allocator_(allocator), streams_(streams), config_(config), arena_(allocator) {
  DSV4_REQUIRE(config_.block_size > 0, "paged block size must be positive");
  DSV4_REQUIRE(config_.max_context_len >= config_.block_size,
               "max context " << config_.max_context_len << " is below one block of " << config_.block_size);
  num_blocks_ = DivideUp(config_.max_context_len, config_.block_size);
  tensors_ = new ArenaTensors();
  backbone_ = new BackboneWeights();
  backbone_->layers.resize(static_cast<size_t>(kNumLayers));
}

StaticArenaManager::~StaticArenaManager() {
  delete tensors_;
  delete backbone_;
}

size_t StaticArenaManager::BackboneDeviceBytes(const MlaGeometry& mla, int64_t block_size,
                                               int64_t max_context_len) {
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

void StaticArenaManager::ReserveActivations() {
  const MlaGeometry& mla = config_.mla;
  const int64_t kv_row = mla.kv_row_elements();
  const int64_t q_b_width = heads_ * kv_row;
  const int64_t o_input = heads_ * mla.kv_lora_rank;
  const int64_t expanded_rows = kTokensPerStep * kNumExpertsPerTok;
  const int64_t shared_inter = kMoeIntermediateSize * kNumSharedExperts;
  ArenaTensors& t = *tensors_;

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
  arena_.Reserve("moe.router_softplus", Fp32Bytes(kTokensPerStep * kNumRoutedExperts));
  arena_.Reserve("moe.router_scores", Fp32Bytes(kTokensPerStep * kNumRoutedExperts));
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

void StaticArenaManager::ReserveBackbone() {
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
    BackboneWeights::Layer& layer = backbone_->layers[static_cast<size_t>(index)];
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

void StaticArenaManager::Commit() { arena_.Commit(); }

void StaticArenaManager::IngestBackbone(WeightByteSource& source) {
  struct Binding {
    ArenaHandle handle;
    const char* pattern;
  };

  // Streamed through a bounded pinned staging buffer, not a plain heap vector:
  // it is the DMA source for every backbone transfer, and page-locked staging
  // is what the transfer path can validate end to end (the exclusive
  // hierarchy's transit scratch takes the same stance). Init-only work, freed
  // before this function returns.
  uint8_t* staging = static_cast<uint8_t*>(allocator_.HostPinnedMalloc(kTransferChunkBytes));
  DSV4_REQUIRE(staging != nullptr, "the backbone staging buffer could not be pinned");

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
      source.ReadNamed(name, staging, kTransferChunkBytes, offset, count);
      streams_.MemcpySync(destination + offset, bytes - offset, staging, count, MemcpyKind::kHostToDevice);
    }
    return true;
  };

  ingest(backbone_->embed_tokens, "model.embed_tokens.weight", true);
  ingest(backbone_->lm_head, "lm_head.weight", true);
  ingest(backbone_->final_norm, "model.norm.weight", true);

  for (int64_t index = 0; index < kNumLayers; ++index) {
    const BackboneWeights::Layer& layer = backbone_->layers[static_cast<size_t>(index)];
    const Binding bindings[] = {
        {layer.input_norm, "model.layers.{L}.input_layernorm.weight"},
        {layer.q_a_weight, "model.layers.{L}.self_attn.q_a_proj.weight"},
        {layer.q_a_scale, "model.layers.{L}.self_attn.q_a_proj.weight_scale_inv"},
        {layer.q_a_norm, "model.layers.{L}.self_attn.q_a_layernorm.weight"},
        {layer.q_b_weight, kFoldedQName},
        {layer.q_b_scale, kFoldedQScaleName},
        {layer.kv_a_weight, "model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight"},
        {layer.kv_a_scale, "model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight_scale_inv"},
        {layer.kv_a_norm, "model.layers.{L}.self_attn.q_a_layernorm.weight"},
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

  allocator_.HostPinnedFree(staging);
}

void StaticArenaManager::CreateDescriptors(const ExpertSlotLayout& slots, const ExpertSlotAddresses& experts) {
  const MlaGeometry& mla = config_.mla;
  const int64_t kv_lora = mla.kv_lora_rank;
  const int64_t rope = mla.qk_rope_head_dim;
  const int64_t kv_row = mla.kv_row_elements();
  const int64_t q_b_width = heads_ * kv_row;
  const int64_t o_input = heads_ * kv_lora;
  const int64_t shared_inter = kMoeIntermediateSize * kNumSharedExperts;
  const int64_t expanded_rows = kTokensPerStep * kNumExpertsPerTok;
  ArenaTensors& t = *tensors_;
  BackboneWeights& b = *backbone_;

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
  t.router_softplus = arena_.CreateTensor("router_softplus", {kTokensPerStep, kNumRoutedExperts}, kAclFloat32,
                                          address("moe.router_softplus"));
  t.router_scores = arena_.CreateTensor("router_scores", {kTokensPerStep, kNumRoutedExperts}, kAclFloat32,
                                        address("moe.router_scores"));
  {
    // The host scalars must carry fp32 bytes: the constants are double, and
    // passing a double's bit pattern as an ACL_FLOAT scalar would silently
    // change softplus's beta/threshold.
    const float beta = static_cast<float>(kSoftplusBeta);
    const float threshold = static_cast<float>(kSoftplusThreshold);
    t.softplus_beta = arena_.CreateScalar("softplus_beta", kAclFloat32, &beta);
    t.softplus_threshold = arena_.CreateScalar("softplus_threshold", kAclFloat32, &threshold);
  }
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
  const BackboneWeights::Layer& first = b.layers[0];
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
  const ExpertRegionSpec& gate_up = slots.region(ExpertRegionId::kGateUpWeight);
  const ExpertRegionSpec& gate_up_scale = slots.region(ExpertRegionId::kGateUpScale);
  const ExpertRegionSpec& down = slots.region(ExpertRegionId::kDownWeight);
  const ExpertRegionSpec& down_scale = slots.region(ExpertRegionId::kDownScale);
  for (int64_t index = 0; index < kNumExpertsPerTok; ++index) {
    const size_t slot = static_cast<size_t>(index);
    t.expert_gate_up.push_back(
        arena_.CreateFp4Tensor("expert_gate_up", {gate_up.rows, gate_up.cols}, experts.gate_up_weight[slot]));
    t.expert_gate_up_scale.push_back(arena_.CreateTensor(
        "expert_gate_up_scale", {gate_up_scale.rows, static_cast<int64_t>(gate_up_scale.stored_cols())},
        kScaleDtype, experts.gate_up_scale[slot]));
    t.expert_down.push_back(
        arena_.CreateFp4Tensor("expert_down", {down.rows, down.cols}, experts.down_weight[slot]));
    t.expert_down_scale.push_back(arena_.CreateTensor(
        "expert_down_scale", {down_scale.rows, static_cast<int64_t>(down_scale.stored_cols())}, kScaleDtype,
        experts.down_scale[slot]));
  }
  t.expert_gate_up_list = arena_.CreateTensorList("expert_gate_up_list", t.expert_gate_up);
  t.expert_gate_up_scale_list = arena_.CreateTensorList("expert_gate_up_scale_list", t.expert_gate_up_scale);
  t.expert_down_list = arena_.CreateTensorList("expert_down_list", t.expert_down);
  t.expert_down_scale_list = arena_.CreateTensorList("expert_down_scale_list", t.expert_down_scale);
}

void StaticArenaManager::CommitWorkspace() { arena_.CommitWorkspace(); }

void StaticArenaManager::Seal() { arena_.Seal(); }

}  // namespace ascend_moe
