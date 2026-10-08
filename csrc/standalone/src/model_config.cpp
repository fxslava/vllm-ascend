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

// ModelConfig implementation: config.json -> validated checkpoint contract.
// See model_config.hpp for the two-jobs-one-rule design note.

#include "moe/core/model_config.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

#include "moe/core/error.hpp"
#include "moe/core/json.hpp"

namespace ascend_moe {
namespace {

// Reads one field if present, recording its presence for the provenance
// report. A wrongly-typed field is a hard error, not a silent default: a
// config.json that says "num_hidden_layers": "43" (string) is a broken
// checkpoint, and the pipeline must never plan against a fallback silently.
class FieldReader {
 public:
  FieldReader(const moe_json::Value& root, const std::string& source_name, std::vector<std::string>* present_keys)
      : root_(root), source_name_(source_name), present_keys_(present_keys) {}

  void Integer(const char* key, int64_t* destination) {
    const moe_json::Value* value = root_.find(key);
    if (value == nullptr) {
      return;
    }
    Typed(value, key, "number");
    *destination = value->as_int();
    MarkPresent(key);
  }

  void Real(const char* key, double* destination) {
    const moe_json::Value* value = root_.find(key);
    if (value == nullptr) {
      return;
    }
    Typed(value, key, "number");
    *destination = value->as_number();
    MarkPresent(key);
  }

  void Boolean(const char* key, bool* destination) {
    const moe_json::Value* value = root_.find(key);
    if (value == nullptr) {
      return;
    }
    Typed(value, key, "boolean");
    *destination = value->as_bool();
    MarkPresent(key);
  }

  void String(const char* key, std::string* destination) {
    const moe_json::Value* value = root_.find(key);
    if (value == nullptr) {
      return;
    }
    Typed(value, key, "string");
    *destination = value->as_string();
    MarkPresent(key);
  }

 private:
  void Typed(const moe_json::Value* value, const char* key, const char* expected) {
    const bool matches = (std::string(expected) == "number" && value->is_number()) ||
                         (std::string(expected) == "boolean" && value->is_bool()) ||
                         (std::string(expected) == "string" && value->is_string());
    DSV4_REQUIRE(matches, source_name_ << ": field '" << key << "' must be a " << expected);
  }

  void MarkPresent(const char* key) { present_keys_->push_back(key); }

  const moe_json::Value& root_;
  const std::string& source_name_;
  std::vector<std::string>* present_keys_;
};

}  // namespace

ModelConfig ModelConfig::FromJsonText(const std::string& json_text, const std::string& source_name) {
  moe_json::Value root = moe_json::Value::MakeNull();
  try {
    root = moe_json::Parse(json_text);
  } catch (const moe_json::ParseError& error) {
    throw Dsv4Error(source_name + ": " + error.what());
  } catch (const std::exception& error) {
    throw Dsv4Error(source_name + ": malformed JSON: " + error.what());
  }
  DSV4_REQUIRE(root.is_object(), source_name << ": the document is not a JSON object");

  ModelConfig config;
  FieldReader reader(root, source_name, &config.present_keys_);

  reader.Integer("num_hidden_layers", &config.num_hidden_layers);
  reader.Integer("n_routed_experts", &config.n_routed_experts);
  reader.Integer("num_experts_per_tok", &config.num_experts_per_tok);
  reader.Integer("n_shared_experts", &config.n_shared_experts);
  reader.Integer("hidden_size", &config.hidden_size);
  reader.Integer("moe_intermediate_size", &config.moe_intermediate_size);
  reader.Integer("vocab_size", &config.vocab_size);
  reader.Integer("num_attention_heads", &config.num_attention_heads);
  reader.Integer("q_lora_rank", &config.q_lora_rank);

  reader.Real("routed_scaling_factor", &config.routed_scaling_factor);
  reader.String("topk_method", &config.topk_method);
  reader.String("scoring_func", &config.scoring_func);
  reader.Boolean("norm_topk_prob", &config.norm_topk_prob);

  reader.Real("swiglu_limit", &config.swiglu_limit);
  reader.Real("rms_norm_eps", &config.rms_norm_eps);

  reader.Integer("index_topk", &config.index_topk);
  reader.Integer("index_head_dim", &config.index_head_dim);
  reader.Integer("index_n_heads", &config.index_n_heads);

  // The MLA geometry: read, but only into the fields the checkpoint really
  // publishes. The provenance flips to kCheckpointConfig iff any of the four
  // appeared, exactly like the report's wording promises.
  reader.Integer("kv_lora_rank", &config.mla.kv_lora_rank);
  reader.Integer("qk_rope_head_dim", &config.mla.qk_rope_head_dim);
  reader.Integer("qk_nope_head_dim", &config.mla.qk_nope_head_dim);
  reader.Integer("v_head_dim", &config.mla.v_head_dim);
  const bool mla_published =
      config.contains("kv_lora_rank") || config.contains("qk_rope_head_dim") ||
      config.contains("qk_nope_head_dim") || config.contains("v_head_dim");
  if (mla_published) {
    config.mla.provenance = GeometryProvenance::kCheckpointConfig;
  }

  reader.String("expert_dtype", &config.expert_dtype);
  reader.String("scale_fmt", &config.scale_fmt);
  if (const moe_json::Value* quantization = root.find("quantization_config")) {
    DSV4_REQUIRE(quantization->is_object(), source_name << ": 'quantization_config' must be an object");
    if (const moe_json::Value* blocks = quantization->find("weight_block_size")) {
      DSV4_REQUIRE(blocks->is_array() && blocks->size() >= 1 && (*blocks)[0].is_number(),
                   source_name << ": 'quantization_config.weight_block_size' must be an array of numbers");
      config.dense_weight_block = (*blocks)[0].as_int();
      config.present_keys_.push_back("quantization_config.weight_block_size");
    }
  }

  // The fields a DeepSeek MoE checkpoint cannot be interpreted without. A
  // missing one must not silently fall back to the compiled default: the
  // whole point of reading config.json is that the checkpoint, not the
  // binary, is the source of truth.
  static const char* kRequiredFields[] = {
      "num_hidden_layers",   "n_routed_experts", "num_experts_per_tok", "hidden_size",
      "moe_intermediate_size", "routed_scaling_factor", "topk_method", "scoring_func", "norm_topk_prob",
  };
  for (const char* field : kRequiredFields) {
    DSV4_REQUIRE(config.contains(field), source_name << ": required field '" << field << "' is missing");
  }

  config.Validate();
  return config;
}

ModelConfig ModelConfig::FromJsonFile(const std::string& config_path) {
  std::ifstream file(config_path, std::ios::binary);
  DSV4_REQUIRE(file.is_open(), "cannot open the model configuration at " << config_path);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return FromJsonText(buffer.str(), config_path);
}

void ModelConfig::Validate() const {
  DSV4_REQUIRE(topk_method == "noaux_tc",
               "unsupported topk_method '" << topk_method << "': the gating stage plans aclnnMoeGatingTopKV2 with "
                                           << "no group constraint and a bias-shifted selection, which encodes "
                                           << "exactly noaux_tc");
  DSV4_REQUIRE(scoring_func == "sqrtsoftplus",
               "unsupported scoring_func '" << scoring_func << "': the router computes scores as "
                                            << "sqrt(softplus(logits)) on the decomposed aclnnSoftplus -> aclnnSqrt "
                                            << "chain, and the gating stage receives them pre-normalized");
  DSV4_REQUIRE(norm_topk_prob,
               "unsupported norm_topk_prob=false: the gating stage renorm=1 L1-renormalizes the selected "
               << "top-k scores before routedScalingFactor is applied");

  DSV4_REQUIRE(num_hidden_layers > 0 && n_routed_experts > 0 && num_experts_per_tok > 0 && hidden_size > 0 &&
                   moe_intermediate_size > 0 && vocab_size > 0 && num_attention_heads > 0,
               "model dimensions must be positive");
  DSV4_REQUIRE(num_experts_per_tok <= n_routed_experts,
               "num_experts_per_tok " << num_experts_per_tok << " exceeds n_routed_experts " << n_routed_experts);
  // The FP4 payload packs two E2M1 nibbles per byte and the E8M0 microscales
  // are one byte per 32 elements along the reduction axis: a reduction axis
  // that is not a block multiple cannot tile a slot (ExpertSlotLayout refuses
  // it too, with the same arithmetic).
  DSV4_REQUIRE(hidden_size % kRoutedScaleBlock == 0,
               "hidden_size " << hidden_size << " is not a multiple of the block-" << kRoutedScaleBlock
                              << " microscale");
  DSV4_REQUIRE(moe_intermediate_size % kRoutedScaleBlock == 0,
               "moe_intermediate_size " << moe_intermediate_size << " is not a multiple of the block-"
                                        << kRoutedScaleBlock << " microscale");
  DSV4_REQUIRE(std::isfinite(routed_scaling_factor) && routed_scaling_factor > 0.0,
               "routed_scaling_factor must be a positive finite number");
  DSV4_REQUIRE(std::isfinite(swiglu_limit) && swiglu_limit > 0.0, "swiglu_limit must be a positive finite number");
}

void ModelConfig::AssertMatchesBinaryContract() const {
  // Every entry: {config.json name, parsed value, compiled value}. The
  // compiled values are what StaticArenaManager / MoeRouterEngine /
  // Dsv4Pipeline planned their descriptors against; a divergent checkpoint
  // would silently mis-compute, so it is refused here, before any
  // reservation, with the full divergence list.
  struct ContractField {
    const char* name;
    int64_t parsed;
    int64_t compiled;
  };
  const std::vector<ContractField> integer_fields = {
      {"num_hidden_layers", num_hidden_layers, kNumLayers},
      {"n_routed_experts", n_routed_experts, kNumRoutedExperts},
      {"num_experts_per_tok", num_experts_per_tok, kNumExpertsPerTok},
      {"n_shared_experts", n_shared_experts, kNumSharedExperts},
      {"hidden_size", hidden_size, kHiddenSize},
      {"moe_intermediate_size", moe_intermediate_size, kMoeIntermediateSize},
      {"vocab_size", vocab_size, kVocabSize},
      {"num_attention_heads", num_attention_heads, kNumAttentionHeads},
      {"q_lora_rank", q_lora_rank, kQLoraRank},
      {"dense weight block", dense_weight_block, kDenseScaleBlock},
  };

  std::ostringstream divergences;
  for (const ContractField& field : integer_fields) {
    if (field.parsed != field.compiled) {
      divergences << "\n  " << field.name << ": config says " << field.parsed << ", binary is compiled for "
                  << field.compiled;
    }
  }
  if (routed_scaling_factor != kRoutedScalingFactor) {
    divergences << "\n  routed_scaling_factor: config says " << routed_scaling_factor << ", binary is compiled for "
                << kRoutedScalingFactor;
  }
  if (swiglu_limit != kSwigluLimit) {
    divergences << "\n  swiglu_limit: config says " << swiglu_limit << ", binary is compiled for " << kSwigluLimit;
  }
  if (!divergences.str().empty()) {
    throw Dsv4Error(
        "the checkpoint topology does not match this binary's compiled graph geometry:" + divergences.str() +
        "\nThe decode graph's descriptors are planned once against the compiled constants (the static-runtime "
        "contract); a divergent model must not run against them. Rebuild the engine for this geometry, or point "
        "--config at the matching checkpoint.");
  }
}

std::string ModelConfig::DescribeSummary(const std::string& source_path) const {
  std::ostringstream out;
  if (source_path.empty()) {
    // A default-constructed ModelConfig IS the compiled contract; say so and
    // print it like any other, so the report has one shape.
    out << "  model config  NOT READ (no config.json found); the compiled DeepSeek-V4 Flash contract:\n";
  } else {
    out << "  model config  parsed from " << source_path << " (all three router asserts passed)\n";
  }
  out << "  topology      " << num_hidden_layers << " layers, hidden " << hidden_size << ", moe_intermediate "
      << moe_intermediate_size << ", vocab " << vocab_size << "\n";
  out << "  router        topk_method " << topk_method << ", scoring_func " << scoring_func
      << ", norm_topk_prob " << (norm_topk_prob ? "true" : "false") << "\n";
  out << "  routed        top-" << num_experts_per_tok << " of " << n_routed_experts << " + " << n_shared_experts
      << " shared, scaling " << routed_scaling_factor << "\n";
  out << "  attention     " << num_attention_heads << " heads, q_lora " << q_lora_rank << ", indexer top-"
      << index_topk << " (" << index_n_heads << " x " << index_head_dim << ", NOT applied -- see stage report)\n";
  out << "  mla geometry  kv_lora " << mla.kv_lora_rank << " (" << (contains("kv_lora_rank") ? "checkpoint" : "family default; checkpoint does not publish it")
      << "), qk_rope " << mla.qk_rope_head_dim << " (" << (contains("qk_rope_head_dim") ? "checkpoint" : "family default")
      << ") -> cache row " << mla.kv_row_elements() << "\n";
  out << "  activation    swiglu_limit " << swiglu_limit << ", rms_norm_eps " << rms_norm_eps << "\n";
  out << "  quantization  expert_dtype " << expert_dtype << " (packed E2M1), dense block " << dense_weight_block
      << ", scale_fmt " << scale_fmt << "\n";
  return out.str();
}

}  // namespace ascend_moe
