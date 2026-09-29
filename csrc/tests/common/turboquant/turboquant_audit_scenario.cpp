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

#include "turboquant_audit_scenario.hpp"

#include <algorithm>

#include "turbo_quant_cpu.h"

namespace vllm_ascend {
namespace test {
namespace turboquant_audit {

namespace tqh = turboquant_host;

std::string RotationModesLabel() {
  std::string label;
  for (const RotationMode mode : kRotationModes) {
    if (RotationModeEnabled(mode)) {
      label += (label.empty() ? "" : ", ") + std::string(RotationModeKey(mode));
    }
  }
  return label;
}

const char* UnpackModeLabel() {
  const std::string raw = env::String(kUnpackEnv);
  for (const char* mode : {kUnpackOff, kUnpackBoth, kUnpackGather, kUnpackLloydMax, kUnpackAll}) {
    if (raw == mode) {
      return mode;
    }
  }
  return kUnpackOn;
}

const std::vector<const char*>& PrefillLegs() {
  static const std::vector<const char*> legs = {kLegPfIngest, kLegPfRotQ, kLegPfAttnCore, kLegPfRotO,
                                                kLegPfE2E,    kLegPfV5,   kLegPfV5Ingest};
  return legs;
}

const std::vector<const char*>& DecodeLegs() {
  static const std::vector<const char*> legs = {kLegDecRotQ,      kLegDecAttnCore, kLegDecRotO,
                                                kLegDecE2E,       kLegDecV5,       kLegDecFusedQAttnCore,
                                                kLegDecFusedQE2E, kLegDecNoUnpack, kLegDecGather,
                                                kLegDecLloydMax};
  return legs;
}

std::string CaseName(const char* leg, const Config& config) { return std::string(leg) + "_" + config.id(); }

bool LegEnabled(const char* leg) { return env::Selects("ASCEND_BENCH_TQ_AUDIT_LEGS", leg); }
bool PhaseEnabled(const char* phase) { return env::Selects("ASCEND_BENCH_TQ_AUDIT_PHASES", phase); }
bool NativeEnabled() { return env::On("ASCEND_BENCH_TQ_FIA"); }

Budget BudgetFor(int64_t seq_len) {
  if (seq_len >= kUltraContextThreshold) {
    return Budget{env::IntClamped("ASCEND_BENCH_TQ_AUDIT_ULTRA_WARMUP", kUltraWarmup, 0),
                  env::IntClamped("ASCEND_BENCH_TQ_AUDIT_ULTRA_ITERS", kUltraIterations, 1)};
  }
  return Budget{env::IntClamped("ASCEND_BENCH_TQ_AUDIT_WARMUP", kShortWarmup, 0),
                env::IntClamped("ASCEND_BENCH_TQ_AUDIT_ITERS", kShortIterations, 1)};
}

std::vector<Regime> Regimes() {
  return {
      Regime{2048, {1, 4, 8}, "short/interactive"},
      Regime{32768, {1, 4, 8}, "enterprise-long"},
      Regime{262144, {1, 2}, "ultra-long"},
      Regime{1048576, {1}, "extreme-needle/1M"},
  };
}

DecodeDispatch PlanDispatch(const Config& config, int64_t aiv_num) {
  const int64_t hq = config.model.num_heads;
  DecodeDispatch dispatch;
  if (config.path == PathMode::kCube) {
    const tqh::FusedDecodeGrid grid =
        tqh::PlanFusedDecode(config.batch, hq, config.model.num_kv_heads, config.model.head_size,
                             config.blocks_per_seq(), kBlockSize, aiv_num, tqh::kFusedContextLimit, SplitPolicy());
    dispatch.split_k = grid.num_splits;
    dispatch.total_tasks = grid.num_tasks;
    dispatch.active_blocks = grid.block_dim;
    dispatch.device_blocks = std::max<int64_t>(1, aiv_num / tqh::kVectorSubcoresPerBlock);
    dispatch.needs_reduction = tqh::DecodeNeedsReduction(grid.num_splits, config.seq_len, grid.fused_context_limit);
    return dispatch;
  }
  const tqh::PagedAttentionGrid grid =
      tqh::PlanPagedAttention(config.batch, hq, config.model.head_size, config.blocks_per_seq(), kBlockSize, aiv_num);
  dispatch.split_k = grid.num_splits;
  dispatch.total_tasks = config.batch * hq * grid.num_splits;
  dispatch.active_blocks = grid.block_dim;
  dispatch.device_blocks = aiv_num;
  dispatch.needs_reduction = tqh::DecodeNeedsReduction(grid.num_splits, config.seq_len, tqh::kFusedContextLimit);
  return dispatch;
}

std::vector<Config> BuildSweep() {
  std::vector<Config> sweep;
  for (const ModelSpec& model : Models()) {
    if (!env::Selects("ASCEND_BENCH_TQ_AUDIT_MODELS", model.key)) {
      continue;
    }
    for (const Regime& regime : Regimes()) {
      if (!env::SelectsInt("ASCEND_BENCH_TQ_AUDIT_S", regime.seq_len)) {
        continue;
      }
      for (const int64_t batch : regime.batches) {
        if (!env::SelectsInt("ASCEND_BENCH_TQ_AUDIT_B", batch)) {
          continue;
        }
        Config config;
        config.model = model;
        config.seq_len = regime.seq_len;
        config.batch = batch;
        config.chunk = PrefillChunk(regime.seq_len);
        config.path = SelectPath(model);
        config.budget = BudgetFor(regime.seq_len);
        config.regime = regime.label;
        sweep.push_back(config);
      }
    }
  }
  return sweep;
}

Traffic ModelTraffic(const Config& config, const DecodeDispatch& dispatch) {
  const double d = static_cast<double>(config.model.head_size);
  const double hq = static_cast<double>(config.model.num_heads);
  const double hkv = static_cast<double>(config.model.num_kv_heads);
  const double packed_slot = d / static_cast<double>(tqh::kPackFactor);
  const double scale_slot = static_cast<double>(tqh::ScaleSlotFloats(config.model.num_kv_heads)) * kFloatBytes;
  const double context = static_cast<double>(config.context_tokens());
  const double chunk = static_cast<double>(config.chunk_tokens());
  const double batch = static_cast<double>(config.batch);
  const double splits = static_cast<double>(dispatch.split_k);

  Traffic traffic;
  traffic.fp16_kv_bytes = 2.0 * context * hkv * d * kHalfBytes;
  traffic.tq_kv_bytes = 2.0 * context * hkv * packed_slot + context * scale_slot;

  const double chunk_kv_fp16 = 2.0 * chunk * hkv * d * kHalfBytes;
  const double chunk_kv_tq = 2.0 * chunk * hkv * packed_slot + chunk * scale_slot;
  const double chunk_query = chunk * hq * d * kHalfBytes;
  const double mask = static_cast<double>(kCausalMaskSide * kCausalMaskSide);

  traffic.pf_ingest = chunk_kv_fp16 + chunk_kv_tq + chunk * kFloatBytes;
  traffic.pf_rot_q = chunk * hq * d * (kHalfBytes + kFloatBytes);
  traffic.pf_attn_core = chunk_query + traffic.fp16_kv_bytes + chunk_query + mask;
  traffic.pf_rot_o = chunk * hq * d * (kHalfBytes + kFloatBytes);
  traffic.pf_e2e = traffic.pf_rot_q + traffic.pf_attn_core + (config.model.folds_output ? 0.0 : traffic.pf_rot_o);
  traffic.pf_v5 = traffic.pf_attn_core;
  traffic.pf_v5_ingest = 2.0 * chunk_kv_fp16 + chunk * kFloatBytes;
  {
    const double s = static_cast<double>(config.seq_len);
    const double c = static_cast<double>(config.chunk);
    traffic.pf_flops = 4.0 * batch * hq * d * c * (s - 0.5 * c + 0.5);
  }
  traffic.pf_untimed_cast =
      chunk * hq * d * (kFloatBytes + kHalfBytes) + context * hkv * d * (kFloatBytes + kHalfBytes);

  const double step_query_fp16 = batch * hq * d * kHalfBytes;
  const double step_query_fp32 = batch * hq * d * kFloatBytes;
  const double step_out_fp16 = step_query_fp16;
  const double partials =
      batch * hq * splits * static_cast<double>(config.model.head_size + tqh::kPartialTail) * kFloatBytes;

  traffic.dec_rot_q = step_query_fp16 + step_query_fp32;
  // One launch: the query, the packed cache and the output token; partials cross GM twice only when the
  // launch reduces its splits, both inside the same launch.
  traffic.dec_attn_core =
      step_query_fp32 + traffic.tq_kv_bytes + step_out_fp16 + (dispatch.needs_reduction ? 2.0 * partials : 0.0);
  traffic.dec_fused_q_attn_core = traffic.dec_rot_q + traffic.dec_attn_core;
  traffic.dec_fused_q_e2e = traffic.dec_fused_q_attn_core;
  traffic.dec_rot_o = step_out_fp16 + batch * hq * d * kFloatBytes;
  traffic.dec_e2e = traffic.dec_rot_q + traffic.dec_attn_core + (config.model.folds_output ? 0.0 : traffic.dec_rot_o);
  traffic.dec_v5 = step_query_fp16 + traffic.fp16_kv_bytes + step_out_fp16;
  traffic.dec_flops = 4.0 * batch * hq * d * static_cast<double>(config.seq_len);
  traffic.dispatch = dispatch;

  return traffic;
}

double ScenarioBytes(const Config& config) {
  const double d = static_cast<double>(config.model.head_size);
  const double hq = static_cast<double>(config.model.num_heads);
  const double hkv = static_cast<double>(config.model.num_kv_heads);
  const double context = static_cast<double>(config.context_tokens());
  const double chunk = static_cast<double>(config.chunk_tokens());
  const double pool_rows = static_cast<double>(config.pool_blocks() * kBlockSize);
  const double scale_slot = static_cast<double>(tqh::ScaleSlotFloats(config.model.num_kv_heads)) * kFloatBytes;

  const double contiguous_kv = 4.0 * context * hkv * d * kHalfBytes;
  const double chunk_kv = 2.0 * chunk * hkv * d * kHalfBytes;
  const double tq_cache = 2.0 * pool_rows * hkv * (d / static_cast<double>(tqh::kPackFactor)) + pool_rows * scale_slot;
  const double fp16_cache = 2.0 * pool_rows * hkv * d * kHalfBytes;
  const double prefill_activations = chunk * hq * d * (4.0 * kHalfBytes + 2.0 * kFloatBytes);
  const double decode_activations = static_cast<double>(config.batch) * hq * d * (4.0 * kHalfBytes + 2.0 * kFloatBytes);
  const double mask = static_cast<double>(kCausalMaskSide * kCausalMaskSide);
  const double workspace_allowance = 2.0 * chunk * hq * d * kHalfBytes;

  return contiguous_kv + chunk_kv + tq_cache + fp16_cache + prefill_activations + decode_activations + mask +
         workspace_allowance;
}

double FullModelSavedMib(const Config& config, const Traffic& traffic) {
  return traffic.saved_mib() * static_cast<double>(config.model.kv_layers * config.model.model_kv_heads) /
         static_cast<double>(config.model.num_kv_heads);
}

std::vector<float> RotatePiRows(std::vector<float> values, int64_t head_size) {
  RotatePiRowsInPlace(values, head_size);
  return values;
}

void RotatePiRowsInPlace(std::vector<float>& values, int64_t head_size) {
  const std::vector<int8_t> signs = turboquant_ref::cpu_pi_sign_vector(static_cast<int>(head_size));
  const size_t row = static_cast<size_t>(head_size);
  for (size_t base = 0; base + row <= values.size(); base += row) {
    turboquant_ref::cpu_apply_pi(values.data() + base, static_cast<int>(head_size), signs.data());
  }
}

}  // namespace turboquant_audit
}  // namespace test
}  // namespace vllm_ascend
