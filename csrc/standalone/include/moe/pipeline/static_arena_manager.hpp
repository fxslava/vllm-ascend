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

// StaticArenaManager: the memory half of the decode graph (SRP).
//
// Everything about WHAT is allocated and WHICH descriptors describe it lives
// here; everything about WHEN it runs lives in Dsv4Pipeline and
// MoeRouterEngine. The manager owns the StaticMemoryArena, the reservation /
// ingestion / descriptor phases of its three-phase lifecycle, and the two
// descriptor sets (activations + backbone weights) the stages consume. It
// knows nothing about streams, operators or the 43-layer replay.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "moe/core/config.hpp"
#include "moe/core/device_allocator.hpp"
#include "moe/core/device_types.hpp"
#include "moe/core/op_table.hpp"
#include "moe/core/weight_source.hpp"
#include "moe/memory/expert_layout.hpp"
#include "moe/memory/static_arena.hpp"

namespace ascend_moe {

// Shared byte-arithmetic of the graph's shapes (reservations, descriptors and
// per-layer addressing all speak in these).
int64_t DivideUp(int64_t value, int64_t divisor);
size_t Fp8Bytes(int64_t elements);
size_t Bf16Bytes(int64_t elements);
size_t Fp32Bytes(int64_t elements);
size_t Int32Bytes(int64_t elements);
size_t Int64Bytes(int64_t elements);
// Block-128 scale columns for a dense weight whose reduction axis is `k`.
int64_t DenseScaleCols(int64_t k);
// Block-32 microscale columns.
int64_t MxScaleCols(int64_t k);

// Where the six active experts' regions sit inside their HBM slots at
// descriptor time. The pipeline resolves these from the exclusive manager
// (slots 0..5 are resident at init) and hands them to the manager, which is
// what keeps this class free of any staging dependency (DIP).
struct ExpertSlotAddresses {
  void* gate_up_weight[kNumExpertsPerTok] = {};
  void* gate_up_scale[kNumExpertsPerTok] = {};
  void* down_weight[kNumExpertsPerTok] = {};
  void* down_scale[kNumExpertsPerTok] = {};
};

// Every activation descriptor the graph uses. Created once, in
// CreateDescriptors; the stages only ever repoint addresses.
struct ArenaTensors {
  // activations
  aclTensor* hidden = nullptr;            // [1, 4096] bf16
  aclTensor* normed = nullptr;           // [1, 4096] bf16
  aclTensor* normed_fp8 = nullptr;       // [1, 4096] fp8
  aclTensor* normed_mx_scale = nullptr;  // [1, 128] e8m0
  aclTensor* rstd = nullptr;             // [1, 1] fp32
  aclTensor* q_a = nullptr;              // [1, 1024] bf16
  aclTensor* q_a_fp8 = nullptr;          // [1, 1024] fp8
  aclTensor* q_a_mx_scale = nullptr;     // [1, 32] e8m0
  aclTensor* q_b = nullptr;              // [1, heads*(kv_lora+rope)] bf16
  aclTensor* q_latent = nullptr;         // [1, heads, kv_lora] strided view of q_b
  aclTensor* q_rope = nullptr;           // [1, heads, rope] strided view of q_b
  aclTensor* kv_a = nullptr;             // [1, kv_lora+rope] bf16
  aclTensor* kv_latent = nullptr;        // [1, kv_lora] view of kv_a
  aclTensor* k_rope = nullptr;           // [1, 1, rope] view of kv_a
  aclTensor* kv_latent_normed = nullptr;  // [1, kv_lora] bf16
  aclTensor* attn_out = nullptr;          // [1, heads, kv_lora] bf16
  aclTensor* attn_flat = nullptr;         // [1, heads*kv_lora] bf16
  aclTensor* attn_fp8 = nullptr;          // [1, heads*kv_lora] fp8
  aclTensor* attn_mx_scale = nullptr;     // [1, heads*kv_lora/32] e8m0
  aclTensor* proj_out = nullptr;          // [1, 4096] bf16
  aclTensor* softmax_lse = nullptr;       // [1, heads, 1] fp32

  // routing
  aclTensor* router_logits = nullptr;      // [1, 256] fp32
  aclTensor* router_softplus = nullptr;    // [1, 256] fp32, softplus(logits)
  aclTensor* router_scores = nullptr;      // [1, 256] fp32, sqrt(softplus(logits))
  aclScalar* softplus_beta = nullptr;      // host scalar, kSoftplusBeta as fp32
  aclScalar* softplus_threshold = nullptr; // host scalar, kSoftplusThreshold as fp32
  aclTensor* gating_weights = nullptr;     // [1, 6] fp32
  aclTensor* gating_indices = nullptr;     // [1, 6] int32
  aclTensor* local_indices = nullptr;      // [1, 6] int32 (0..5, host-written)
  aclTensor* expanded_x = nullptr;         // [6, 4096] fp8
  aclTensor* expanded_row_idx = nullptr;   // [6] int32
  aclTensor* expanded_scale = nullptr;     // [6, 128] e8m0
  aclTensor* expanded_weights = nullptr;   // [6] fp32
  aclTensor* expanded_weights_row = nullptr;  // [1, 6] fp32 view, for the combine
  aclTensor* group_list = nullptr;         // [6] int64 cumsum
  aclTensor* gemm1_raw = nullptr;          // [6, 2*2048] bf16 (decomposed path)
  aclTensor* gemm1_out = nullptr;          // [6, 2048] fp8
  aclTensor* gemm1_scale = nullptr;        // [6, 64] e8m0
  aclTensor* gemm2_out = nullptr;          // [6, 4096] bf16
  aclTensor* routed_out = nullptr;         // [1, 4096] bf16
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

// Per-layer backbone weight reservations. kv_b_proj never appears: its two
// up-projections are absorbed into q_b_proj and o_proj at checkpoint-conversion
// time, which is what makes the decode graph attend in the compressed latent
// space and the paged cache `kv_lora_rank + qk_rope_head_dim` wide.
struct BackboneWeights {
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

class StaticArenaManager {
 public:
  // The stream engine is used exactly once per phase: the ordered H2D
  // transfers of backbone ingestion. Everything else is allocation.
  StaticArenaManager(IDeviceAllocator& allocator, IStreamEngine& streams, const RuntimeConfig& config);
  ~StaticArenaManager();

  StaticArenaManager(const StaticArenaManager&) = delete;
  StaticArenaManager& operator=(const StaticArenaManager&) = delete;

  // The memory the backbone needs, so the slot planner can subtract it from
  // free HBM before choosing K. Pure arithmetic; no allocation.
  static size_t BackboneDeviceBytes(const MlaGeometry& mla, int64_t block_size, int64_t max_context_len);

  // ---- phase 1: RESERVE -------------------------------------------------
  void ReserveActivations();
  void ReserveBackbone();

  // ---- phase 2: BUILD ---------------------------------------------------
  void Commit();
  void IngestBackbone(WeightByteSource& source);
  void CreateDescriptors(const ExpertSlotLayout& slots, const ExpertSlotAddresses& experts);
  void CommitWorkspace();

  // ---- phase 3: SEALED --------------------------------------------------
  void Seal();

  // ---- accessors ---------------------------------------------------------
  // The non-const overloads exist for the planning phase (workspace notes,
  // descriptor creation); the decode loop sees const only.
  const StaticMemoryArena& arena() const { return arena_; }
  StaticMemoryArena& arena() { return arena_; }
  const ArenaTensors& tensors() const { return *tensors_; }
  ArenaTensors& tensors() { return *tensors_; }
  const BackboneWeights& backbone() const { return *backbone_; }
  int64_t num_blocks() const { return num_blocks_; }

 private:
  IDeviceAllocator& allocator_;
  IStreamEngine& streams_;
  RuntimeConfig config_;
  StaticMemoryArena arena_;
  ArenaTensors* tensors_ = nullptr;
  BackboneWeights* backbone_ = nullptr;
  int64_t num_blocks_ = 0;
  int64_t heads_ = kNumAttentionHeads;
};

}  // namespace ascend_moe
