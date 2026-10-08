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

// ModelConfig: the DeepSeek-V4 Flash checkpoint contract, read at RUNTIME
// from the model directory's config.json instead of being taken from the
// compiled constants in config.hpp alone.
//
// TWO JOBS, ONE RULE
// ------------------
// 1. VALIDATE the checkpoint against what this binary can execute. The
//    router contract asserts are the pipeline's own decisions, spelled out:
//
//      topk_method  == "noaux_tc"       the gating stage runs bias-shifted
//                                       selection with no group constraint
//                                       (kGatingKGroup=1, group_select 0)
//      scoring_func == "sqrtsoftplus"   the scores arrive pre-normalized
//                                       from the decomposed
//                                       aclnnSoftplus -> aclnnSqrt chain
//                                       (normType -1 in MoeGatingTopKV2)
//      norm_topk_prob == true           the gating stage renorm=1
//                                       L1-renormalizes the selected six
//
//    A checkpoint with any other router must not reach the device.
//
// 2. FEED the parametric parts of the stack: the expert-slot planner
//    (ExpertSlotLayout::ForGeometry(hidden, moe_intermediate)), the
//    exclusive hierarchy's options (layers, experts, top-k), the weight
//    sources, and the MLA geometry that IS published by the checkpoint.
//    The graph descriptors, by contrast, are planned against the compiled
//    constants -- that is the static-runtime contract -- so
//    `AssertMatchesBinaryContract()` refuses any checkpoint whose topology
//    diverges from the binary, naming every diverging field. A divergent
//    model needs a rebuild, never a silent mis-plan.
//
// Default-constructing a ModelConfig yields exactly the compiled DSV4-Flash
// contract, so "no config.json anywhere" and "config.json agrees" are the
// same code path downstream.

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "moe/core/config.hpp"

namespace ascend_moe {

class ModelConfig {
 public:
  // ---- topology (config.json names, verbatim) ---------------------------
  int64_t num_hidden_layers = kNumLayers;
  int64_t n_routed_experts = kNumRoutedExperts;
  int64_t num_experts_per_tok = kNumExpertsPerTok;
  int64_t n_shared_experts = kNumSharedExperts;
  int64_t hidden_size = kHiddenSize;
  int64_t moe_intermediate_size = kMoeIntermediateSize;
  int64_t vocab_size = kVocabSize;
  int64_t num_attention_heads = kNumAttentionHeads;
  int64_t q_lora_rank = kQLoraRank;

  // ---- router contract (asserted, see the class comment) ----------------
  double routed_scaling_factor = kRoutedScalingFactor;
  std::string topk_method = "noaux_tc";
  std::string scoring_func = "sqrtsoftplus";
  bool norm_topk_prob = true;

  // ---- activation / norm -------------------------------------------------
  double swiglu_limit = kSwigluLimit;
  double rms_norm_eps = kRmsNormEpsilon;

  // ---- the compressed Lightning Indexer (reported, NOT applied: the
  //      pipeline's deviation note 2 says why) -----------------------------
  int64_t index_topk = kIndexTopK;
  int64_t index_head_dim = kIndexHeadDim;
  int64_t index_n_heads = kIndexNumHeads;

  // ---- MLA geometry: only what the checkpoint actually publishes is
  //      overwritten. This config.json carries qk_rope_head_dim but NOT
  //      kv_lora_rank / qk_nope_head_dim / v_head_dim, and a guessed
  //      reservation would corrupt the paged-KV budget (see config.hpp).
  MlaGeometry mla{};

  // ---- quantization ------------------------------------------------------
  std::string expert_dtype = "fp4";         // the slot layout packs E2M1 nibbles
  int64_t dense_weight_block = kDenseScaleBlock;  // quantization_config.weight_block_size[0]
  std::string scale_fmt = "ue8m0";          // OCP E8M0 microscales

  // Parses and Validate()s. Throws (Dsv4Error) on an unreadable file, a
  // malformed document, a wrongly-typed field, or a violated router
  // contract -- every message names the field and the file.
  static ModelConfig FromJsonFile(const std::string& config_path);
  static ModelConfig FromJsonText(const std::string& json_text, const std::string& source_name);

  // The router-contract asserts plus the structural invariants the slot
  // layout and the grouped GEMM depend on (positive dims, hidden and
  // intermediate multiples of the block-32 microscale, top_k <= experts).
  void Validate() const;

  // Throws naming EVERY field where the checkpoint topology diverges from
  // the compiled graph constants. Call after Validate(), before any
  // reservation.
  void AssertMatchesBinaryContract() const;

  // True when `field` was actually present in the parsed JSON (used for the
  // MLA provenance report: family default vs checkpoint value).
  bool contains(const char* field) const {
    return std::find(present_keys_.begin(), present_keys_.end(), field) != present_keys_.end();
  }

  // The human-readable startup block.
  std::string DescribeSummary(const std::string& source_path) const;

 private:
  // Which fields the document supplied (config.json names).
  std::vector<std::string> present_keys_;
};

}  // namespace ascend_moe
