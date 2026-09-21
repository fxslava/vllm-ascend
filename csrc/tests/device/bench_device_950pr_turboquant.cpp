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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "acl_check.hpp"
#include "aclnn_ops.hpp"
#include "aclnn_ops_950pr.hpp"
#include "aclnn_runtime.hpp"
#include "ascend950_shapes.hpp"
#include "benchmark.hpp"
#include "device_buffer.hpp"
#include "fp16.hpp"
#include "random_data.hpp"
#include "turbo_quant_cpu.h"
#include "turboquant_audit_models.hpp"
#include "turboquant_launch.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {

const char* kSuiteName =
    "turboquant_950pr (end-to-end audit: TurboQuant against the native CANN V5 attention operator)";

namespace {

namespace tqh = turboquant_host;
namespace tqm = vllm_ascend::turboquant;
namespace s950 = shapes950;
namespace tqa = turboquant_audit;

using tqa::ModelSpec;
using tqa::Models;
using tqa::PathLabel;
using tqa::PathMode;
using tqa::PrefillChunk;
using tqa::PrefillChunkTokens;
using tqa::SelectPath;
using tqa::SplitPolicy;
using tqa::SplitPolicyLabel;

constexpr int64_t kBlockSize = s950::kDefaultBlockSize;

constexpr int64_t kCausalMaskSide = 2048;
constexpr int64_t kFiaSparseModeRightDownCausal = 3;
constexpr int64_t kFiaNoPaging = 0;

constexpr int64_t kPatternTokens = 1024;

constexpr double kHbmBudget = 0.85;

constexpr size_t kChecksumElements = 1u << 20;

constexpr size_t kExactContextElements = 1u << 22;

constexpr double kRotatedBasisMinCosine = 0.999;
constexpr double kDecodeTiePointMinCosine = 0.99;
// The two rotation modes decode the same cache; only rotate_q's Cube rounding (B >= 2) may separate them.
constexpr double kRotationModeMinCosine = 0.9999;
constexpr float kUnwrittenSentinel = -1234.5f;

constexpr double kHalfBytes = 2.0;
constexpr double kFloatBytes = 4.0;
constexpr double kMiB = 1024.0 * 1024.0;

constexpr int kShortWarmup = 5;
constexpr int kShortIterations = 50;
constexpr int kUltraWarmup = 1;
constexpr int kUltraIterations = 3;
constexpr int kPipelineBatch = 1;

constexpr int64_t kUltraContextThreshold = 262144;
constexpr int64_t kOversizedPoolContextLimit = 8192;
constexpr int64_t kPoolOversizeFactor = 4;

std::vector<std::string> SplitCsv(const char* raw) {
  std::vector<std::string> fields;
  if (raw == nullptr || *raw == '\0') {
    return fields;
  }
  std::istringstream stream(raw);
  std::string field;
  while (std::getline(stream, field, ',')) {
    if (!field.empty()) {
      fields.push_back(field);
    }
  }
  return fields;
}

int EnvInt(const char* name, int fallback, int minimum) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return fallback;
  }
  const long parsed = std::strtol(raw, nullptr, 10);
  return parsed < minimum ? minimum : static_cast<int>(parsed);
}

bool EnvOn(const char* name) {
  const char* raw = std::getenv(name);
  return raw == nullptr || *raw == '\0' || std::strcmp(raw, "0") != 0;
}

bool EnvSelects(const char* name, const std::string& value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return true;
  }
  for (const std::string& field : SplitCsv(raw)) {
    if (field == value) {
      return true;
    }
  }
  return false;
}

bool EnvSelectsInt(const char* name, int64_t value) {
  std::ostringstream text;
  text << value;
  return EnvSelects(name, text.str());
}

std::string EnvString(const char* name) {
  const char* raw = std::getenv(name);
  return raw == nullptr ? std::string() : std::string(raw);
}

// Where a decode step changes basis. kSeparate launches rotate_q ahead of the decode (PRE_ROTATED = true) and,
// for an unfolded W_o, rotate_q again over its output; kFusedPrologue hands the raw fp16 query to the decode,
// which rotates it and, unfolded, un-rotates its output inside the same launch (PRE_ROTATED = false,
// kUnrotated; fused cases (g) and (h); kv4fp8 on the Cube path only). A Qwen3.5 layer would also gate in that
// launch (kGated); the bench leaves the gate out on both sides, as it does for the native decode.
enum class RotationMode { kSeparate, kFusedPrologue };

const RotationMode kRotationModes[] = {RotationMode::kSeparate, RotationMode::kFusedPrologue};

constexpr const char* kRotationModeEnv = "ASCEND_BENCH_TQ_ROTATION_MODE";
constexpr const char* kRotationModeBoth = "both";

const char* RotationModeKey(RotationMode mode) {
  return mode == RotationMode::kSeparate ? "separate" : "fused_prologue";
}

bool RotationModeValid() {
  const std::string raw = EnvString(kRotationModeEnv);
  return raw.empty() || raw == kRotationModeBoth || raw == RotationModeKey(RotationMode::kSeparate) ||
         raw == RotationModeKey(RotationMode::kFusedPrologue);
}

// ASCEND_BENCH_TQ_ROTATION_MODE=separate|fused_prologue|both, both by default; the banner reports any other
// value, which is read as both.
bool RotationModeEnabled(RotationMode mode) {
  const std::string raw = EnvString(kRotationModeEnv);
  if (raw == RotationModeKey(RotationMode::kSeparate) || raw == RotationModeKey(RotationMode::kFusedPrologue)) {
    return raw == RotationModeKey(mode);
  }
  return true;
}

struct Regime {
  int64_t seq_len;
  std::vector<int64_t> batches;
  const char* label;
};

std::vector<Regime> Regimes() {
  return {
      Regime{2048, {1, 4, 8}, "short/interactive"},
      Regime{32768, {1, 4, 8}, "enterprise-long"},
      Regime{262144, {1, 2}, "ultra-long"},
      Regime{1048576, {1}, "extreme-needle/1M"},
  };
}

struct Budget {
  int warmup = 0;
  int iterations = 0;

  bool operator==(const Budget& other) const {
    return warmup == other.warmup && iterations == other.iterations;
  }
};

Budget BudgetFor(int64_t seq_len) {
  if (seq_len >= kUltraContextThreshold) {
    return Budget{EnvInt("ASCEND_BENCH_TQ_AUDIT_ULTRA_WARMUP", kUltraWarmup, 0),
                  EnvInt("ASCEND_BENCH_TQ_AUDIT_ULTRA_ITERS", kUltraIterations, 1)};
  }
  return Budget{EnvInt("ASCEND_BENCH_TQ_AUDIT_WARMUP", kShortWarmup, 0),
                EnvInt("ASCEND_BENCH_TQ_AUDIT_ITERS", kShortIterations, 1)};
}

struct Config {
  ModelSpec model;
  int64_t seq_len = 0;
  int64_t batch = 0;
  int64_t chunk = 0;
  PathMode path = PathMode::kAiv;
  Budget budget;
  const char* regime = "";

  int64_t chunk_tokens() const { return batch * chunk; }
  int64_t context_tokens() const { return batch * seq_len; }
  int64_t blocks_per_seq() const { return tqh::CeilDiv(seq_len, kBlockSize); }
  int64_t pool_blocks() const {
    const int64_t needed = blocks_per_seq() * batch;
    return seq_len <= kOversizedPoolContextLimit ? needed * kPoolOversizeFactor : needed;
  }
  float attention_scale() const { return 1.0f / std::sqrt(static_cast<float>(model.head_size)); }

  std::string id() const {
    std::ostringstream name;
    name << model.key << "_s" << seq_len << "_b" << batch;
    return name.str();
  }

  // The raw-query decode entry exists for the Cube decode alone.
  bool decodes_in_mode(RotationMode mode) const {
    return RotationModeEnabled(mode) && (mode == RotationMode::kSeparate || path == PathMode::kCube);
  }

  std::string path_tag(RotationMode mode) const {
    return std::string(PathLabel(path)) + (mode == RotationMode::kSeparate ? "-Sep" : "-FusedQ");
  }

  // Host dispatches of one decode step: separately, rotate-q, the decode and rotate-o unless W_o is folded;
  // in-launch, the decode alone.
  int decode_launches(RotationMode mode) const {
    return mode == RotationMode::kSeparate ? 2 + (model.folds_output ? 0 : 1) : 1;
  }

  // What the raw-query decode writes: the rotated basis for a folded W_o, the model basis otherwise.
  uint32_t fused_q_output_stage() const {
    return static_cast<uint32_t>(model.folds_output ? vllm_ascend::turboquant::TurboQuantOutputStage::kRotatedBasis
                                                    : vllm_ascend::turboquant::TurboQuantOutputStage::kUnrotated);
  }
};

std::vector<Config> BuildSweep() {
  std::vector<Config> sweep;
  for (const ModelSpec& model : Models()) {
    if (!EnvSelects("ASCEND_BENCH_TQ_AUDIT_MODELS", model.key)) {
      continue;
    }
    for (const Regime& regime : Regimes()) {
      if (!EnvSelectsInt("ASCEND_BENCH_TQ_AUDIT_S", regime.seq_len)) {
        continue;
      }
      for (const int64_t batch : regime.batches) {
        if (!EnvSelectsInt("ASCEND_BENCH_TQ_AUDIT_B", batch)) {
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

constexpr const char* kLegPfIngest = "pf_ingest";
constexpr const char* kLegPfRotQ = "pf_rot_q";
constexpr const char* kLegPfAttnCore = "pf_attn_core";
constexpr const char* kLegPfRotO = "pf_rot_o";
constexpr const char* kLegPfE2E = "pf_e2e";
constexpr const char* kLegPfV5 = "pf_v5";
constexpr const char* kLegPfV5Ingest = "pf_v5_ingest";

constexpr const char* kLegDecRotQ = "dec_rot_q";
constexpr const char* kLegDecAttnCore = "dec_attn_core";
constexpr const char* kLegDecRotO = "dec_rot_o";
constexpr const char* kLegDecE2E = "dec_e2e";
constexpr const char* kLegDecV5 = "dec_v5";
constexpr const char* kLegDecFusedQAttnCore = "dec_fq_attn_core";
constexpr const char* kLegDecFusedQE2E = "dec_fq_e2e";

const char* const kPrefillLegs[] = {kLegPfIngest, kLegPfRotQ, kLegPfAttnCore, kLegPfRotO,
                                    kLegPfE2E,    kLegPfV5,   kLegPfV5Ingest};
const char* const kDecodeLegs[] = {kLegDecRotQ, kLegDecAttnCore,       kLegDecRotO,     kLegDecE2E,
                                   kLegDecV5,   kLegDecFusedQAttnCore, kLegDecFusedQE2E};

std::string CaseName(const char* leg, const Config& config) {
  return std::string(leg) + "_" + config.id();
}

bool LegEnabled(const char* leg) { return EnvSelects("ASCEND_BENCH_TQ_AUDIT_LEGS", leg); }
bool PhaseEnabled(const char* phase) { return EnvSelects("ASCEND_BENCH_TQ_AUDIT_PHASES", phase); }
bool NativeEnabled() { return EnvOn("ASCEND_BENCH_TQ_FIA"); }

// One decode step's dispatch vector as the planner tiles it (Table B's Cfg [K/Tsk/Blk/Red]).
struct DecodeDispatch {
  int64_t split_k = 1;
  // Cube: B x H_KV x K x head chunks. AIV: B x H_Q x K.
  int64_t total_tasks = 0;
  // Blocks the launch spans, of device_blocks: MIX blocks on the Cube path, vector cores on the AIV path.
  int64_t active_blocks = 0;
  int64_t device_blocks = 0;
  // The launch reduces split partials after a SyncAll (the kernels' NeedsReduction()).
  bool needs_reduction = false;

  std::string label() const {
    std::ostringstream text;
    text << split_k << '/' << total_tasks << '/' << active_blocks << '/' << (needs_reduction ? "ON" : "OFF");
    return text.str();
  }
};

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
  const tqh::PagedAttentionGrid grid = tqh::PlanPagedAttention(config.batch, hq, config.model.head_size,
                                                               config.blocks_per_seq(), kBlockSize, aiv_num);
  dispatch.split_k = grid.num_splits;
  dispatch.total_tasks = config.batch * hq * grid.num_splits;
  dispatch.active_blocks = grid.block_dim;
  dispatch.device_blocks = aiv_num;
  dispatch.needs_reduction = tqh::DecodeNeedsReduction(grid.num_splits, config.seq_len, tqh::kFusedContextLimit);
  return dispatch;
}

struct Traffic {
  double fp16_kv_bytes = 0.0;
  double tq_kv_bytes = 0.0;

  double pf_ingest = 0.0;
  double pf_rot_q = 0.0;
  double pf_attn_core = 0.0;
  double pf_rot_o = 0.0;
  double pf_e2e = 0.0;
  double pf_v5 = 0.0;
  double pf_v5_ingest = 0.0;
  double pf_flops = 0.0;
  double pf_untimed_cast = 0.0;

  double dec_rot_q = 0.0;
  double dec_attn_core = 0.0;
  // The raw-query decode moves rotate_q's bytes too: the fp16 query in, the rotated fp32 query into GM. Its
  // output stage adds none: the un-rotated output is the one fp16 write the decode makes anyway.
  double dec_fused_q_attn_core = 0.0;
  double dec_fused_q_e2e = 0.0;
  double dec_rot_o = 0.0;
  double dec_e2e = 0.0;
  double dec_v5 = 0.0;
  double dec_flops = 0.0;
  DecodeDispatch dispatch;

  double compression_ratio() const { return tq_kv_bytes > 0.0 ? fp16_kv_bytes / tq_kv_bytes : 0.0; }
  double saved_mib() const { return (fp16_kv_bytes - tq_kv_bytes) / kMiB; }
};

// The benchmarked slice (config.model.num_kv_heads kv heads of one layer) scaled to the full model's residency.
double FullModelSavedMib(const Config& config, const Traffic& traffic) {
  return traffic.saved_mib() * static_cast<double>(config.model.kv_layers * config.model.model_kv_heads) /
         static_cast<double>(config.model.num_kv_heads);
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
  traffic.pf_e2e = traffic.pf_rot_q + traffic.pf_attn_core +
                   (config.model.folds_output ? 0.0 : traffic.pf_rot_o);
  traffic.pf_v5 = traffic.pf_attn_core;
  traffic.pf_v5_ingest = 2.0 * chunk_kv_fp16 + chunk * kFloatBytes;
  {
    const double s = static_cast<double>(config.seq_len);
    const double c = static_cast<double>(config.chunk);
    traffic.pf_flops = 4.0 * batch * hq * d * c * (s - 0.5 * c + 0.5);
  }
  traffic.pf_untimed_cast = chunk * hq * d * (kFloatBytes + kHalfBytes) +
                            context * hkv * d * (kFloatBytes + kHalfBytes);

  const double step_query_fp16 = batch * hq * d * kHalfBytes;
  const double step_query_fp32 = batch * hq * d * kFloatBytes;
  const double step_out_fp16 = step_query_fp16;
  const double partials = batch * hq * splits * static_cast<double>(config.model.head_size + tqh::kPartialTail) *
                          kFloatBytes;

  traffic.dec_rot_q = step_query_fp16 + step_query_fp32;
  // One launch: the query, the packed cache and the output token; partials cross GM twice only when the
  // launch reduces its splits, both inside the same launch.
  traffic.dec_attn_core =
      step_query_fp32 + traffic.tq_kv_bytes + step_out_fp16 + (dispatch.needs_reduction ? 2.0 * partials : 0.0);
  traffic.dec_fused_q_attn_core = traffic.dec_rot_q + traffic.dec_attn_core;
  traffic.dec_fused_q_e2e = traffic.dec_fused_q_attn_core;
  traffic.dec_rot_o = step_out_fp16 + batch * hq * d * kFloatBytes;
  traffic.dec_e2e = traffic.dec_rot_q + traffic.dec_attn_core +
                    (config.model.folds_output ? 0.0 : traffic.dec_rot_o);
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
  const double tq_cache = 2.0 * pool_rows * hkv * (d / static_cast<double>(tqh::kPackFactor)) +
                          pool_rows * scale_slot;
  const double fp16_cache = 2.0 * pool_rows * hkv * d * kHalfBytes;
  const double prefill_activations = chunk * hq * d * (4.0 * kHalfBytes + 2.0 * kFloatBytes);
  const double decode_activations =
      static_cast<double>(config.batch) * hq * d * (4.0 * kHalfBytes + 2.0 * kFloatBytes);
  const double mask = static_cast<double>(kCausalMaskSide * kCausalMaskSide);
  const double workspace_allowance = 2.0 * chunk * hq * d * kHalfBytes;

  return contiguous_kv + chunk_kv + tq_cache + fp16_cache + prefill_activations + decode_activations + mask +
         workspace_allowance;
}

template <typename T>
void TileToDevice(const DeviceBuffer& dst, const std::vector<T>& pattern) {
  const size_t total = dst.size_bytes();
  const size_t chunk = pattern.size() * sizeof(T);
  if (chunk == 0) {
    return;
  }
  for (size_t offset = 0; offset < total; offset += chunk) {
    const size_t bytes = std::min(chunk, total - offset);
    ACL_CHECK(aclrtMemcpy(static_cast<char*>(dst.get()) + offset, dst.capacity_bytes() - offset, pattern.data(),
                          bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  }
}

template <typename T>
std::vector<T> HostTiled(const std::vector<T>& pattern, size_t elements) {
  std::vector<T> tiled(elements);
  if (pattern.empty()) {
    return tiled;
  }
  for (size_t i = 0; i < elements; ++i) {
    tiled[i] = pattern[i % pattern.size()];
  }
  return tiled;
}

template <typename T>
std::vector<T> LeadingElements(const DeviceBuffer& buffer) {
  std::vector<T> host(std::min(buffer.size_bytes() / sizeof(T), kChecksumElements));
  if (!host.empty()) {
    buffer.CopyToHost(host.data(), host.size() * sizeof(T));
  }
  return host;
}

void RotateRowsInPlace(std::vector<float>& values, int64_t head_size) {
  const std::vector<int8_t> signs = turboquant_ref::cpu_pi_sign_vector(static_cast<int>(head_size));
  const size_t row = static_cast<size_t>(head_size);
  for (size_t base = 0; base + row <= values.size(); base += row) {
    turboquant_ref::cpu_apply_pi(values.data() + base, static_cast<int>(head_size), signs.data());
  }
}

const AclnnOp& FiaV5() {
  static const AclnnOp op(ops950::kFusedInferAttentionScoreV5);
  return op;
}
const AclnnOp& FiaV2() {
  static const AclnnOp op(ops950::kFusedInferAttentionScoreV2);
  return op;
}
const AclnnOp& ScatterPaKvCacheOp() {
  static const AclnnOp op(ops::kScatterPaKvCache);
  return op;
}

class Scenario {
 public:
  Scenario(const Config& config, int64_t aiv_num) : config_(config), aiv_num_(aiv_num) {
    const int64_t d = config.model.head_size;
    const int64_t hq = config.model.num_heads;
    const int64_t hkv = config.model.num_kv_heads;
    const int64_t context = config.context_tokens();
    const int64_t chunk = config.chunk_tokens();
    const int64_t pattern = std::min<int64_t>(context, kPatternTokens);
    const auto elems = [](int64_t a, int64_t b, int64_t c) {
      return static_cast<size_t>(a) * static_cast<size_t>(b) * static_cast<size_t>(c);
    };

    exact_context_ = elems(context, hkv, d) <= kExactContextElements;

    DeterministicRandom rng(0xA53Du + static_cast<uint32_t>(config.seq_len + 7 * config.batch + 31 * d +
                                                            127 * hq + 509 * hkv));

    const std::vector<float> key_pattern = rng.NormalHalfExact(elems(pattern, hkv, d), 0.0f, 1.0f);
    const std::vector<float> value_pattern = rng.NormalHalfExact(elems(pattern, hkv, d), 0.0f, 1.0f);
    std::vector<float> key_rot_pattern = key_pattern;
    std::vector<float> value_rot_pattern = value_pattern;
    RotateRowsInPlace(key_rot_pattern, d);
    RotateRowsInPlace(value_rot_pattern, d);

    key_ctx_ = DeviceBuffer::Empty<Half>(elems(context, hkv, d), kBenchmarkAlignBytes);
    value_ctx_ = DeviceBuffer::Empty<Half>(elems(context, hkv, d), kBenchmarkAlignBytes);
    key_ctx_rot_ = DeviceBuffer::Empty<Half>(elems(context, hkv, d), kBenchmarkAlignBytes);
    value_ctx_rot_ = DeviceBuffer::Empty<Half>(elems(context, hkv, d), kBenchmarkAlignBytes);
    TileToDevice(key_ctx_, FloatToHalf(key_pattern));
    TileToDevice(value_ctx_, FloatToHalf(value_pattern));
    TileToDevice(key_ctx_rot_, FloatToHalf(key_rot_pattern));
    TileToDevice(value_ctx_rot_, FloatToHalf(value_rot_pattern));

    const int64_t chunk_pattern = std::min<int64_t>(chunk, kPatternTokens);
    const std::vector<float> chunk_key = rng.NormalHalfExact(elems(chunk_pattern, hkv, d), 0.0f, 1.0f);
    const std::vector<float> chunk_value = rng.NormalHalfExact(elems(chunk_pattern, hkv, d), 0.0f, 1.0f);
    key_chunk_ = DeviceBuffer::Empty<Half>(elems(chunk, hkv, d), kBenchmarkAlignBytes);
    value_chunk_ = DeviceBuffer::Empty<Half>(elems(chunk, hkv, d), kBenchmarkAlignBytes);
    TileToDevice(key_chunk_, FloatToHalf(chunk_key));
    TileToDevice(value_chunk_, FloatToHalf(chunk_value));

    const int64_t query_pattern = std::min<int64_t>(chunk, kPatternTokens);
    std::vector<float> prefill_query = rng.NormalHalfExact(elems(query_pattern, hq, d), 0.0f, 1.0f);
    query_pf_ = DeviceBuffer::Empty<Half>(elems(chunk, hq, d), kBenchmarkAlignBytes);
    TileToDevice(query_pf_, FloatToHalf(prefill_query));
    std::vector<float> prefill_query_rot = prefill_query;
    RotateRowsInPlace(prefill_query_rot, d);
    query_pf_rot_half_ = DeviceBuffer::Empty<Half>(elems(chunk, hq, d), kBenchmarkAlignBytes);
    TileToDevice(query_pf_rot_half_, FloatToHalf(prefill_query_rot));
    query_pf_rot_fp32_ = DeviceBuffer::Empty<float>(elems(chunk, hq, d), kBenchmarkAlignBytes);

    out_pf_tq_ = DeviceBuffer::Empty<Half>(elems(chunk, hq, d), kBenchmarkAlignBytes);
    out_pf_v5_ = DeviceBuffer::Empty<Half>(elems(chunk, hq, d), kBenchmarkAlignBytes);
    out_pf_rot_ = DeviceBuffer::Empty<float>(elems(chunk, hq, d), kBenchmarkAlignBytes);
    lse_pf_tq_ = DeviceBuffer::Empty<Half>(1, kBenchmarkAlignBytes);
    lse_pf_v5_ = DeviceBuffer::Empty<Half>(1, kBenchmarkAlignBytes);

    const std::vector<float> decode_query =
        rng.NormalHalfExact(elems(config.batch, hq, d), 0.0f, 1.0f);
    query_dec_ = DeviceBuffer::FromHost(FloatToHalf(decode_query), kBenchmarkAlignBytes);
    query_dec_rot_ = DeviceBuffer::Empty<float>(elems(config.batch, hq, d), kBenchmarkAlignBytes);
    query_dec_prologue_rot_ = DeviceBuffer::Empty<float>(elems(config.batch, hq, d), kBenchmarkAlignBytes);
    out_dec_tq_ = DeviceBuffer::Empty<Half>(elems(config.batch, hq, d), kBenchmarkAlignBytes);
    out_dec_v5_ = DeviceBuffer::Empty<Half>(elems(config.batch, hq, d), kBenchmarkAlignBytes);
    out_dec_rot_ = DeviceBuffer::Empty<float>(elems(config.batch, hq, d), kBenchmarkAlignBytes);
    lse_dec_ = DeviceBuffer::Empty<Half>(1, kBenchmarkAlignBytes);

    const int64_t blocks_per_seq = config.blocks_per_seq();
    const std::vector<int32_t> pool = rng.Permutation(static_cast<int32_t>(config.pool_blocks()));
    std::vector<int32_t> block_table(static_cast<size_t>(config.batch * blocks_per_seq));
    for (int64_t seq = 0; seq < config.batch; ++seq) {
      for (int64_t block = 0; block < blocks_per_seq; ++block) {
        block_table[static_cast<size_t>(seq * blocks_per_seq + block)] =
            pool[static_cast<size_t>(seq * blocks_per_seq + block)];
      }
    }
    block_tables_ = DeviceBuffer::FromHost(block_table, kBenchmarkAlignBytes);

    std::vector<int32_t> slots_full(static_cast<size_t>(context));
    for (int64_t seq = 0; seq < config.batch; ++seq) {
      for (int64_t pos = 0; pos < config.seq_len; ++pos) {
        const int32_t block = block_table[static_cast<size_t>(seq * blocks_per_seq + pos / kBlockSize)];
        slots_full[static_cast<size_t>(seq * config.seq_len + pos)] =
            block * static_cast<int32_t>(kBlockSize) + static_cast<int32_t>(pos % kBlockSize);
      }
    }
    std::vector<int32_t> slots_chunk(static_cast<size_t>(chunk));
    for (int64_t seq = 0; seq < config.batch; ++seq) {
      for (int64_t j = 0; j < config.chunk; ++j) {
        const int64_t pos = config.seq_len - config.chunk + j;
        slots_chunk[static_cast<size_t>(seq * config.chunk + j)] =
            slots_full[static_cast<size_t>(seq * config.seq_len + pos)];
      }
    }
    slots_full_ = DeviceBuffer::FromHost(slots_full, kBenchmarkAlignBytes);
    slots_chunk_ = DeviceBuffer::FromHost(slots_chunk, kBenchmarkAlignBytes);
    context_lens_ = DeviceBuffer::FromHost(
        std::vector<int32_t>(static_cast<size_t>(config.batch), static_cast<int32_t>(config.seq_len)),
        kBenchmarkAlignBytes);

    pi_signs_ = DeviceBuffer::FromHost(tqh::PiSigns(d), kBenchmarkAlignBytes);
    h16_ = DeviceBuffer::FromHost(tqh::Hadamard16Half(), kBenchmarkAlignBytes);
    rot_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(d, 1), kBenchmarkAlignBytes);

    const int64_t pool_blocks = config.pool_blocks();
    if (config.path == PathMode::kCube) {
      key_cache_ = DeviceBuffer::Empty<int8_t>(
          tqh::ModePackedCacheBytes(kCubeMode, pool_blocks, kBlockSize, hkv, d), kBenchmarkAlignBytes);
      write_tables_ =
          DeviceBuffer::FromHost(tqh::ModeTables(kCubeMode, d, 1, 0), kBenchmarkAlignBytes);
      decode_tables_ = DeviceBuffer::FromHost(
          tqh::ModeTables(kCubeMode, d, tqh::kUnpackRows, tqh::kCubeTileRows), kBenchmarkAlignBytes);
      cube_grid_ = tqh::PlanFusedDecode(config.batch, hq, hkv, d, blocks_per_seq, kBlockSize, aiv_num,
                                        tqh::kFusedContextLimit, SplitPolicy());
      num_splits_ = cube_grid_.num_splits;
      workspace_floats_ = cube_grid_.workspace_floats;
    } else {
      key_cache_ =
          DeviceBuffer::Empty<int8_t>(tqh::PackedCacheBytes(pool_blocks, kBlockSize, hkv, d), kBenchmarkAlignBytes);
      write_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(d, 1), kBenchmarkAlignBytes);
      decode_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(d, tqh::kTileRows), kBenchmarkAlignBytes);
      aiv_grid_ = tqh::PlanPagedAttention(config.batch, hq, d, blocks_per_seq, kBlockSize, aiv_num);
      num_splits_ = aiv_grid_.num_splits;
      workspace_floats_ = aiv_grid_.workspace_floats;
    }
    value_cache_ = DeviceBuffer::Empty<int8_t>(key_cache_.size_bytes(), kBenchmarkAlignBytes);
    scale_plane_ = DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(pool_blocks, kBlockSize, hkv),
                                              kBenchmarkAlignBytes);
    workspace_ = DeviceBuffer::Empty<float>(workspace_floats_, kBenchmarkAlignBytes);

    context_write_grid_ = tqh::PlanReshapeAndCache(context, aiv_num);
    chunk_write_grid_ = tqh::PlanReshapeAndCache(chunk, aiv_num);

    const size_t cache_elements = elems(pool_blocks * kBlockSize, hkv, d);
    fp16_key_cache_ = DeviceBuffer::Empty<Half>(cache_elements, kBenchmarkAlignBytes);
    fp16_value_cache_ = DeviceBuffer::Empty<Half>(cache_elements, kBenchmarkAlignBytes);
    if (exact_context_) {
      const size_t row = static_cast<size_t>(hkv * d);
      const std::vector<float> key_host = HostTiled(key_pattern, elems(context, hkv, d));
      const std::vector<float> value_host = HostTiled(value_pattern, elems(context, hkv, d));
      std::vector<float> key_cache(cache_elements, 0.0f);
      std::vector<float> value_cache(cache_elements, 0.0f);
      for (int64_t pos = 0; pos < context; ++pos) {
        const size_t slot = static_cast<size_t>(slots_full[static_cast<size_t>(pos)]);
        const size_t src = static_cast<size_t>(pos) * row;
        const size_t dst = slot * row;
        std::copy(key_host.begin() + static_cast<std::ptrdiff_t>(src),
                  key_host.begin() + static_cast<std::ptrdiff_t>(src + row),
                  key_cache.begin() + static_cast<std::ptrdiff_t>(dst));
        std::copy(value_host.begin() + static_cast<std::ptrdiff_t>(src),
                  value_host.begin() + static_cast<std::ptrdiff_t>(src + row),
                  value_cache.begin() + static_cast<std::ptrdiff_t>(dst));
      }
      fp16_key_cache_.CopyFromHost(FloatToHalf(key_cache).data(), cache_elements * sizeof(Half));
      fp16_value_cache_.CopyFromHost(FloatToHalf(value_cache).data(), cache_elements * sizeof(Half));
    } else {
      TileToDevice(fp16_key_cache_, FloatToHalf(key_pattern));
      TileToDevice(fp16_value_cache_, FloatToHalf(value_pattern));
    }

    std::vector<int8_t> mask(static_cast<size_t>(kCausalMaskSide * kCausalMaskSide), 0);
    for (int64_t row = 0; row < kCausalMaskSide; ++row) {
      for (int64_t col = row + 1; col < kCausalMaskSide; ++col) {
        mask[static_cast<size_t>(row * kCausalMaskSide + col)] = 1;
      }
    }
    mask_ = DeviceBuffer::FromHost(mask, kBenchmarkAlignBytes);

    BuildDescriptors();
    PlanOperators();
  }

  void EnqueuePrefillIngest(aclrtStream stream) const {
    EnqueueCacheWrite(stream, key_chunk_.get(), value_chunk_.get(), slots_chunk_.get(), config_.chunk_tokens(),
                      chunk_write_grid_);
  }

  void EnqueuePrefillRotateQ(aclrtStream stream) const {
    tqh::RotateQuery(stream, AscendType::FP16, query_pf_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                     query_pf_rot_fp32_.get(), config_.chunk_tokens(), config_.model.num_heads,
                     config_.model.head_size, aiv_num_);
  }

  void EnqueuePrefillAttnCore(aclrtStream stream) const { fia_prefill_rotated_->Launch(stream); }

  void EnqueuePrefillRotateO(aclrtStream stream) const {
    tqh::RotateQuery(stream, AscendType::FP16, out_pf_tq_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                     out_pf_rot_.get(), config_.chunk_tokens(), config_.model.num_heads, config_.model.head_size,
                     aiv_num_);
  }

  void EnqueuePrefillE2E(aclrtStream stream) const {
    EnqueuePrefillRotateQ(stream);
    EnqueuePrefillAttnCore(stream);
    if (!config_.model.folds_output) {
      EnqueuePrefillRotateO(stream);
    }
  }

  void EnqueuePrefillNative(aclrtStream stream) const { fia_prefill_native_->Launch(stream); }
  void EnqueuePrefillNativeIngest(aclrtStream stream) const { native_write_->Launch(stream); }

  void EnqueueDecodeRotateQ(aclrtStream stream) const {
    tqh::RotateQuery(stream, AscendType::FP16, query_dec_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                     query_dec_rot_.get(), config_.batch, config_.model.num_heads, config_.model.head_size,
                     aiv_num_);
  }

  void EnqueueDecodeFusedCube(aclrtStream stream) const {
    turboquant_mm_fused_decode_impl(
        static_cast<int32_t>(kCubeMode), AscendType::FP16, stream, cube_grid_.block_dim, query_dec_rot_.get(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(),
        decode_tables_.get(), workspace_.get(), out_dec_tq_.get(), static_cast<uint32_t>(config_.batch),
        static_cast<uint32_t>(config_.model.num_heads), static_cast<uint32_t>(config_.model.num_kv_heads),
        static_cast<uint32_t>(config_.model.head_size), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(config_.blocks_per_seq()), static_cast<uint32_t>(cube_grid_.num_splits),
        cube_grid_.heads_per_task, cube_grid_.tasks_per_block, cube_grid_.reduce_tasks_per_block,
        cube_grid_.fused_context_limit, config_.attention_scale(), config_.attention_scale());
  }

  // The same grid handed the raw fp16 query: the launch rotates every (token, head) into
  // query_dec_prologue_rot_ ahead of its split tasks (on the Cube from 16 vectors), so no rotate_q launch
  // precedes it, and for an unfolded W_o it writes out_dec_tq_ un-rotated, so no rotate_o launch follows.
  void EnqueueDecodeFusedQ(aclrtStream stream) const {
    turboquant_mm_fused_decode_raw_query_impl(
        static_cast<int32_t>(kCubeMode), AscendType::FP16, stream, cube_grid_.block_dim, query_dec_.get(),
        pi_signs_.get(), rot_tables_.get(), h16_.get(), nullptr, query_dec_prologue_rot_.get(), key_cache_.get(),
        value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(), decode_tables_.get(),
        workspace_.get(), out_dec_tq_.get(), static_cast<uint32_t>(config_.batch),
        static_cast<uint32_t>(config_.model.num_heads), static_cast<uint32_t>(config_.model.num_kv_heads),
        static_cast<uint32_t>(config_.model.head_size), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(config_.blocks_per_seq()), static_cast<uint32_t>(cube_grid_.num_splits),
        cube_grid_.heads_per_task, cube_grid_.tasks_per_block, cube_grid_.reduce_tasks_per_block,
        cube_grid_.prologue_vectors_per_block, cube_grid_.prologue_cube_chunk_vectors,
        config_.fused_q_output_stage(), cube_grid_.fused_context_limit, config_.attention_scale(),
        config_.attention_scale());
  }

  // One launch: the in-launch output stage leaves nothing to follow it.
  void EnqueueDecodeFusedQE2E(aclrtStream stream) const { EnqueueDecodeFusedQ(stream); }

  // Both rotation modes over the same cache: the in-launch rotation against rotate_q's word for word, and the
  // two decode outputs in the basis the step hands on (after rotate_o, unfolded) against each other. The two
  // rotations stage the Cube under different rules, so only the outputs are held to a bound.
  struct RotationAgreement {
    size_t rotated_words = 0;
    size_t word_mismatches = 0;
    double output_cosine = 0.0;
  };

  RotationAgreement CompareRotationModes(aclrtStream stream) {
    EnqueueDecodeE2E(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    const std::vector<float> separate_rot = query_dec_rot_.ToHost<float>();
    const std::vector<float> separate_out =
        config_.model.folds_output ? DecodeTqOutput() : out_dec_rot_.ToHost<float>();

    const std::vector<float> unwritten(separate_rot.size(), kUnwrittenSentinel);
    query_dec_prologue_rot_.CopyFromHost(unwritten.data(), unwritten.size() * sizeof(float));
    EnqueueDecodeFusedQ(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    const std::vector<float> fused_rot = query_dec_prologue_rot_.ToHost<float>();

    RotationAgreement agreement;
    agreement.rotated_words = separate_rot.size();
    for (size_t i = 0; i < separate_rot.size(); ++i) {
      agreement.word_mismatches +=
          (i >= fused_rot.size() || std::memcmp(&separate_rot[i], &fused_rot[i], sizeof(float)) != 0) ? 1u : 0u;
    }
    agreement.output_cosine = turboquant_ref::cpu_fidelity(DecodeTqOutput(), separate_out).cosine_similarity;
    return agreement;
  }

  void EnqueueDecodeAttnCore(aclrtStream stream) const {
    if (config_.path == PathMode::kCube) {
      EnqueueDecodeFusedCube(stream);
      return;
    }
    turboquant_paged_attention_impl(
        AscendType::FP16, stream, aiv_grid_.block_dim, query_dec_rot_.get(), key_cache_.get(), value_cache_.get(),
        scale_plane_.get(), block_tables_.get(), context_lens_.get(), decode_tables_.get(), workspace_.get(),
        out_dec_tq_.get(), static_cast<uint32_t>(config_.batch), static_cast<uint32_t>(config_.model.num_heads),
        static_cast<uint32_t>(config_.model.num_kv_heads), static_cast<uint32_t>(config_.model.head_size),
        static_cast<uint32_t>(kBlockSize), static_cast<uint32_t>(config_.blocks_per_seq()),
        static_cast<uint32_t>(aiv_grid_.num_splits), aiv_grid_.split_tasks_per_core, aiv_grid_.reduce_tasks_per_core,
        static_cast<uint32_t>(tqh::kFusedContextLimit), config_.attention_scale(), config_.attention_scale());
  }

  void EnqueueDecodeRotateO(aclrtStream stream) const {
    tqh::RotateQuery(stream, AscendType::FP16, out_dec_tq_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                     out_dec_rot_.get(), config_.batch, config_.model.num_heads, config_.model.head_size, aiv_num_);
  }

  void EnqueueDecodeE2E(aclrtStream stream) const {
    EnqueueDecodeRotateQ(stream);
    EnqueueDecodeAttnCore(stream);
    if (!config_.model.folds_output) {
      EnqueueDecodeRotateO(stream);
    }
  }

  void EnqueueDecodeNative(aclrtStream stream) const { fia_decode_native_->Launch(stream); }

  void FillCache(aclrtStream stream) const {
    EnqueueCacheWrite(stream, key_ctx_.get(), value_ctx_.get(), slots_full_.get(), config_.context_tokens(),
                      context_write_grid_);
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  void Prime(aclrtStream stream) const {
    EnqueueDecodeRotateQ(stream);
    EnqueueDecodeAttnCore(stream);
    if (config_.decodes_in_mode(RotationMode::kFusedPrologue)) {
      EnqueueDecodeFusedQ(stream);
    }
    if (prefill_available()) {
      EnqueuePrefillAttnCore(stream);
      EnqueuePrefillNative(stream);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  bool prefill_available() const { return fia_prefill_rotated_ != nullptr && fia_prefill_native_ != nullptr; }
  bool decode_available() const { return fia_decode_native_ != nullptr; }
  bool native_write_available() const { return native_write_ != nullptr; }
  bool exact_context() const { return exact_context_; }
  const std::string& fia_operator() const { return fia_operator_; }
  const std::string& fia_note() const { return fia_note_; }
  const std::string& native_write_note() const { return native_write_note_; }
  int64_t num_splits() const { return num_splits_; }
  uint32_t block_dim() const { return config_.path == PathMode::kCube ? cube_grid_.block_dim : aiv_grid_.block_dim; }

  double ScaleChecksum() const { return ChecksumSum(LeadingElements<float>(scale_plane_)); }
  double PrefillRotatedQueryChecksum() const { return ChecksumSum(LeadingElements<float>(query_pf_rot_fp32_)); }
  double PrefillTqChecksum() const { return ChecksumSum(HalfToFloat(LeadingElements<Half>(out_pf_tq_))); }
  double PrefillNativeChecksum() const { return ChecksumSum(HalfToFloat(LeadingElements<Half>(out_pf_v5_))); }
  double PrefillRotatedOutChecksum() const { return ChecksumSum(LeadingElements<float>(out_pf_rot_)); }
  double NativeCacheChecksum() const {
    return ChecksumSum(HalfToFloat(LeadingElements<Half>(fp16_value_cache_)));
  }
  double DecodeRotatedQueryChecksum() const { return ChecksumSum(LeadingElements<float>(query_dec_rot_)); }
  double DecodeTqChecksum() const { return ChecksumSum(HalfToFloat(LeadingElements<Half>(out_dec_tq_))); }
  double DecodeNativeChecksum() const { return ChecksumSum(HalfToFloat(LeadingElements<Half>(out_dec_v5_))); }
  double DecodeRotatedOutChecksum() const { return ChecksumSum(LeadingElements<float>(out_dec_rot_)); }

  double PrefillPipelineChecksum() const {
    return config_.model.folds_output ? PrefillTqChecksum() : PrefillRotatedOutChecksum();
  }
  double DecodePipelineChecksum() const {
    return config_.model.folds_output ? DecodeTqChecksum() : DecodeRotatedOutChecksum();
  }

  std::vector<float> PrefillTqOutput() const { return HalfToFloat(out_pf_tq_.ToHost<Half>()); }
  std::vector<float> PrefillNativeOutput() const { return HalfToFloat(out_pf_v5_.ToHost<Half>()); }
  std::vector<float> DecodeTqOutput() const { return HalfToFloat(out_dec_tq_.ToHost<Half>()); }
  std::vector<float> DecodeNativeOutput() const { return HalfToFloat(out_dec_v5_.ToHost<Half>()); }

 private:
  static constexpr tqm::TurboQuantMode kCubeMode = tqm::TurboQuantMode::KV4_FP8;

  void EnqueueCacheWrite(aclrtStream stream, void* key, void* value, void* slots, int64_t tokens,
                         const tqh::ReshapeAndCacheGrid& grid) const {
    const uint32_t token_count = static_cast<uint32_t>(tokens);
    const uint32_t kv_heads = static_cast<uint32_t>(config_.model.num_kv_heads);
    const uint32_t head_size = static_cast<uint32_t>(config_.model.head_size);
    if (config_.path == PathMode::kCube) {
      turboquant_mm_reshape_and_cache_impl(static_cast<int32_t>(kCubeMode), AscendType::FP16, stream,
                                           grid.block_dim, key, value, key_cache_.get(), value_cache_.get(),
                                           scale_plane_.get(), slots, pi_signs_.get(), rot_tables_.get(),
                                           write_tables_.get(), token_count, kv_heads, head_size,
                                           static_cast<uint32_t>(kBlockSize),
                                           static_cast<uint32_t>(config_.pool_blocks()), grid.tokens_per_core,
                                           config_.attention_scale());
      return;
    }
    turboquant_reshape_and_cache_impl(AscendType::FP16, stream, grid.block_dim, key, value, key_cache_.get(),
                                      value_cache_.get(), scale_plane_.get(), slots, pi_signs_.get(),
                                      write_tables_.get(), token_count, kv_heads, head_size,
                                      static_cast<uint32_t>(kBlockSize),
                                      static_cast<uint32_t>(config_.pool_blocks()), grid.tokens_per_core,
                                      config_.attention_scale());
  }

  void BuildDescriptors() {
    const int64_t d = config_.model.head_size;
    const int64_t hq = config_.model.num_heads;
    const int64_t hkv = config_.model.num_kv_heads;
    const int64_t chunk = config_.chunk_tokens();
    const int64_t context = config_.context_tokens();
    const int64_t pool_blocks = config_.pool_blocks();

    query_pf_tnd_.reset(new AclnnTensor({chunk, hq, d}, ACL_FLOAT16, query_pf_.get()));
    query_pf_rot_tnd_.reset(new AclnnTensor({chunk, hq, d}, ACL_FLOAT16, query_pf_rot_half_.get()));
    key_ctx_tnd_.reset(new AclnnTensor({context, hkv, d}, ACL_FLOAT16, key_ctx_.get()));
    value_ctx_tnd_.reset(new AclnnTensor({context, hkv, d}, ACL_FLOAT16, value_ctx_.get()));
    key_ctx_rot_tnd_.reset(new AclnnTensor({context, hkv, d}, ACL_FLOAT16, key_ctx_rot_.get()));
    value_ctx_rot_tnd_.reset(new AclnnTensor({context, hkv, d}, ACL_FLOAT16, value_ctx_rot_.get()));
    mask_tensor_.reset(new AclnnTensor({kCausalMaskSide, kCausalMaskSide}, ACL_INT8, mask_.get()));
    out_pf_tq_tensor_.reset(new AclnnTensor({chunk, hq, d}, ACL_FLOAT16, out_pf_tq_.get()));
    out_pf_v5_tensor_.reset(new AclnnTensor({chunk, hq, d}, ACL_FLOAT16, out_pf_v5_.get()));
    lse_pf_tq_tensor_.reset(new AclnnTensor({1}, ACL_FLOAT16, lse_pf_tq_.get()));
    lse_pf_v5_tensor_.reset(new AclnnTensor({1}, ACL_FLOAT16, lse_pf_v5_.get()));

    key_ctx_list_.reset(new AclnnTensorList({key_ctx_tnd_->get()}));
    value_ctx_list_.reset(new AclnnTensorList({value_ctx_tnd_->get()}));
    key_ctx_rot_list_.reset(new AclnnTensorList({key_ctx_rot_tnd_->get()}));
    value_ctx_rot_list_.reset(new AclnnTensorList({value_ctx_rot_tnd_->get()}));

    std::vector<int64_t> cumulative_q(static_cast<size_t>(config_.batch));
    std::vector<int64_t> cumulative_kv(static_cast<size_t>(config_.batch));
    for (int64_t seq = 0; seq < config_.batch; ++seq) {
      cumulative_q[static_cast<size_t>(seq)] = (seq + 1) * config_.chunk;
      cumulative_kv[static_cast<size_t>(seq)] = (seq + 1) * config_.seq_len;
    }
    prefill_seq_q_.reset(new AclnnIntArray(cumulative_q));
    prefill_seq_kv_.reset(new AclnnIntArray(cumulative_kv));

    query_dec_tnd_.reset(new AclnnTensor({config_.batch, hq, d}, ACL_FLOAT16, query_dec_.get()));
    out_dec_v5_tensor_.reset(new AclnnTensor({config_.batch, hq, d}, ACL_FLOAT16, out_dec_v5_.get()));
    lse_dec_tensor_.reset(new AclnnTensor({1}, ACL_FLOAT16, lse_dec_.get()));
    fp16_key_flat_.reset(new AclnnTensor(s950::FiaKeyCacheView(pool_blocks, kBlockSize, hkv, d), ACL_FLOAT16,
                                         fp16_key_cache_.get()));
    fp16_value_flat_.reset(new AclnnTensor(s950::FiaKeyCacheView(pool_blocks, kBlockSize, hkv, d), ACL_FLOAT16,
                                           fp16_value_cache_.get()));
    fp16_key_list_.reset(new AclnnTensorList({fp16_key_flat_->get()}));
    fp16_value_list_.reset(new AclnnTensorList({fp16_value_flat_->get()}));
    block_table_tensor_.reset(
        new AclnnTensor({config_.batch, config_.blocks_per_seq()}, ACL_INT32, block_tables_.get()));
    {
      std::vector<int64_t> cumulative_q(static_cast<size_t>(config_.batch));
      for (int64_t seq = 0; seq < config_.batch; ++seq) {
        cumulative_q[static_cast<size_t>(seq)] = seq + 1;
      }
      decode_seq_q_.reset(new AclnnIntArray(cumulative_q));
      std::vector<int64_t> kv_lens(static_cast<size_t>(config_.batch), config_.seq_len);
      decode_seq_kv_.reset(new AclnnIntArray(kv_lens));
    }

    key_chunk_tensor_.reset(new AclnnTensor({chunk, hkv, d}, ACL_FLOAT16, key_chunk_.get()));
    value_chunk_tensor_.reset(new AclnnTensor({chunk, hkv, d}, ACL_FLOAT16, value_chunk_.get()));
    slots_chunk_tensor_.reset(new AclnnTensor({chunk}, ACL_INT32, slots_chunk_.get()));
    fp16_key_cache_tensor_.reset(
        new AclnnTensor({pool_blocks, kBlockSize, hkv, d}, ACL_FLOAT16, fp16_key_cache_.get()));
    fp16_value_cache_tensor_.reset(
        new AclnnTensor({pool_blocks, kBlockSize, hkv, d}, ACL_FLOAT16, fp16_value_cache_.get()));
  }

  void PlanOperators() {
    if (!NativeEnabled()) {
      fia_note_ = "the native V5 legs were dropped by ASCEND_BENCH_TQ_FIA=0";
      return;
    }
    fia_prefill_rotated_ = PlanPrefill(query_pf_rot_tnd_->get(), key_ctx_rot_list_->get(),
                                       value_ctx_rot_list_->get(), out_pf_tq_tensor_->get(),
                                       lse_pf_tq_tensor_->get());
    fia_prefill_native_ = PlanPrefill(query_pf_tnd_->get(), key_ctx_list_->get(), value_ctx_list_->get(),
                                      out_pf_v5_tensor_->get(), lse_pf_v5_tensor_->get());
    fia_decode_native_ = PlanDecode();
    try {
      native_write_.reset(new PlannedOp(PlanAclnn<ops::ScatterPaKvCacheWorkspaceFn>(
          ScatterPaKvCacheOp(), key_chunk_tensor_->get(), fp16_key_cache_tensor_->get(),
          slots_chunk_tensor_->get(), value_chunk_tensor_->get(), fp16_value_cache_tensor_->get(),
          nullptr, nullptr, nullptr,
          const_cast<char*>(ops::kScatterCacheModeNorm), nullptr, nullptr,
          nullptr)));
    } catch (const AclError& error) {
      native_write_note_ = std::string(ops::kScatterPaKvCache) + ": " + error.what();
    }
  }

  std::unique_ptr<PlannedOp> PlanPrefill(const aclTensor* query, const aclTensorList* key_list,
                                         const aclTensorList* value_list, const aclTensor* out,
                                         const aclTensor* lse) {
    const int64_t hq = config_.model.num_heads;
    const int64_t hkv = config_.model.num_kv_heads;
    const double scale = static_cast<double>(config_.attention_scale());
    try {
      std::unique_ptr<PlannedOp> planned(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV5WorkspaceFn>(
          FiaV5(), query,
          key_list, value_list, nullptr, mask_tensor_->get(), prefill_seq_q_->get(),
          prefill_seq_kv_->get(), nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, hq, scale,
          s950::kFiaUnboundedTokens, s950::kFiaUnboundedTokens, const_cast<char*>(ops950::kFiaLayoutTnd), hkv,
          kFiaSparseModeRightDownCausal, s950::kFiaInnerPreciseDefault, kFiaNoPaging, 0,
          false, 0, 0,
          s950::kFiaQueryQuantModeNone, s950::kFiaPseTypeDefault, out, lse)));
      RecordOperator(ops950::kFusedInferAttentionScoreV5);
      return planned;
    } catch (const AclError& error) {
      AppendNote(std::string(ops950::kFusedInferAttentionScoreV5) + " (prompt role): " + error.what());
    }
    try {
      std::unique_ptr<PlannedOp> planned(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV2WorkspaceFn>(
          FiaV2(), query, key_list,
          value_list, nullptr, mask_tensor_->get(), prefill_seq_q_->get(), prefill_seq_kv_->get(),
          nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, hq, scale, s950::kFiaUnboundedTokens, s950::kFiaUnboundedTokens,
          const_cast<char*>(ops950::kFiaLayoutTnd), hkv, kFiaSparseModeRightDownCausal,
          s950::kFiaInnerPreciseDefault, kFiaNoPaging, 0, false,
          0, 0, out, lse)));
      RecordOperator(ops950::kFusedInferAttentionScoreV2);
      return planned;
    } catch (const AclError& error) {
      AppendNote(std::string(ops950::kFusedInferAttentionScoreV2) + " (prompt role): " + error.what());
    }
    return nullptr;
  }

  std::unique_ptr<PlannedOp> PlanDecode() {
    const int64_t hq = config_.model.num_heads;
    const int64_t hkv = config_.model.num_kv_heads;
    const double scale = static_cast<double>(config_.attention_scale());
    try {
      std::unique_ptr<PlannedOp> planned(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV5WorkspaceFn>(
          FiaV5(), query_dec_tnd_->get(), fp16_key_list_->get(), fp16_value_list_->get(), nullptr,
          nullptr, decode_seq_q_->get(), decode_seq_kv_->get(), nullptr,
          nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, block_table_tensor_->get(),
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, hq, scale, s950::kFiaUnboundedTokens, s950::kFiaUnboundedTokens,
          const_cast<char*>(ops950::kFiaLayoutTnd), hkv, s950::kFiaSparseModeNone,
          s950::kFiaInnerPreciseDefault, kBlockSize, 0, false,
          0, 0, s950::kFiaQueryQuantModeNone,
          s950::kFiaPseTypeDefault, out_dec_v5_tensor_->get(), lse_dec_tensor_->get())));
      RecordOperator(ops950::kFusedInferAttentionScoreV5);
      return planned;
    } catch (const AclError& error) {
      AppendNote(std::string(ops950::kFusedInferAttentionScoreV5) + " (incremental role): " + error.what());
    }
    try {
      std::unique_ptr<PlannedOp> planned(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV2WorkspaceFn>(
          FiaV2(), query_dec_tnd_->get(), fp16_key_list_->get(), fp16_value_list_->get(), nullptr,
          nullptr, decode_seq_q_->get(), decode_seq_kv_->get(), nullptr,
          nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr, block_table_tensor_->get(),
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, hq,
          scale, s950::kFiaUnboundedTokens, s950::kFiaUnboundedTokens, const_cast<char*>(ops950::kFiaLayoutTnd),
          hkv, s950::kFiaSparseModeNone, s950::kFiaInnerPreciseDefault, kBlockSize, 0,
          false, 0, 0,
          out_dec_v5_tensor_->get(), lse_dec_tensor_->get())));
      RecordOperator(ops950::kFusedInferAttentionScoreV2);
      return planned;
    } catch (const AclError& error) {
      AppendNote(std::string(ops950::kFusedInferAttentionScoreV2) + " (incremental role): " + error.what());
    }
    return nullptr;
  }

  void RecordOperator(const char* name) {
    if (fia_operator_.empty()) {
      fia_operator_ = name;
    }
  }
  void AppendNote(const std::string& note) {
    if (!fia_note_.empty()) {
      fia_note_ += "; ";
    }
    fia_note_ += note;
  }

  Config config_;
  int64_t aiv_num_ = 0;
  bool exact_context_ = false;
  int64_t num_splits_ = 1;
  size_t workspace_floats_ = 0;

  tqh::ReshapeAndCacheGrid context_write_grid_;
  tqh::ReshapeAndCacheGrid chunk_write_grid_;
  tqh::PagedAttentionGrid aiv_grid_;
  tqh::FusedDecodeGrid cube_grid_;

  DeviceBuffer key_ctx_, value_ctx_, key_ctx_rot_, value_ctx_rot_;
  DeviceBuffer key_chunk_, value_chunk_;
  DeviceBuffer query_pf_, query_pf_rot_half_, query_pf_rot_fp32_;
  DeviceBuffer out_pf_tq_, out_pf_v5_, out_pf_rot_, lse_pf_tq_, lse_pf_v5_;
  DeviceBuffer query_dec_, query_dec_rot_, query_dec_prologue_rot_;
  DeviceBuffer out_dec_tq_, out_dec_v5_, out_dec_rot_, lse_dec_;
  DeviceBuffer slots_full_, slots_chunk_, block_tables_, context_lens_;
  DeviceBuffer pi_signs_, h16_, rot_tables_, write_tables_, decode_tables_;
  DeviceBuffer key_cache_, value_cache_, scale_plane_, workspace_;
  DeviceBuffer fp16_key_cache_, fp16_value_cache_;
  DeviceBuffer mask_;

  std::unique_ptr<AclnnTensor> query_pf_tnd_, query_pf_rot_tnd_;
  std::unique_ptr<AclnnTensor> key_ctx_tnd_, value_ctx_tnd_, key_ctx_rot_tnd_, value_ctx_rot_tnd_;
  std::unique_ptr<AclnnTensor> mask_tensor_, out_pf_tq_tensor_, out_pf_v5_tensor_;
  std::unique_ptr<AclnnTensor> lse_pf_tq_tensor_, lse_pf_v5_tensor_;
  std::unique_ptr<AclnnTensor> query_dec_tnd_, out_dec_v5_tensor_, lse_dec_tensor_;
  std::unique_ptr<AclnnTensor> fp16_key_flat_, fp16_value_flat_, block_table_tensor_;
  std::unique_ptr<AclnnTensor> key_chunk_tensor_, value_chunk_tensor_, slots_chunk_tensor_;
  std::unique_ptr<AclnnTensor> fp16_key_cache_tensor_, fp16_value_cache_tensor_;
  std::unique_ptr<AclnnTensorList> key_ctx_list_, value_ctx_list_, key_ctx_rot_list_, value_ctx_rot_list_;
  std::unique_ptr<AclnnTensorList> fp16_key_list_, fp16_value_list_;
  std::unique_ptr<AclnnIntArray> prefill_seq_q_, prefill_seq_kv_, decode_seq_q_, decode_seq_kv_;

  std::string fia_operator_;
  std::string fia_note_;
  std::string native_write_note_;
  std::unique_ptr<PlannedOp> fia_prefill_rotated_, fia_prefill_native_, fia_decode_native_, native_write_;
};

class RunnerSet {
 public:
  RunnerSet(BenchmarkRunner& primary, const std::vector<Budget>& budgets) : primary_(&primary) {
    if (budgets.empty()) {
      budgets_.push_back(Budget{kShortWarmup, kShortIterations});
    } else {
      budgets_ = budgets;
    }
    BenchmarkOptions options = Options(primary.options(), budgets_.front());
    primary.set_options(options);
    runners_.push_back(&primary);
    for (size_t index = 1; index < budgets_.size(); ++index) {
      std::ostringstream name;
      name << kSuiteName << " -- warmup " << budgets_[index].warmup << ", " << budgets_[index].iterations
           << " iterations";
      owned_.emplace_back(new BenchmarkRunner(name.str(), Options(primary.options(), budgets_[index]),
                                              primary.stream()));
      runners_.push_back(owned_.back().get());
    }
  }

  BenchmarkRunner& For(const Budget& budget) const {
    for (size_t index = 0; index < budgets_.size(); ++index) {
      if (budgets_[index] == budget) {
        return *runners_[index];
      }
    }
    return *primary_;
  }

  const std::vector<BenchmarkRunner*>& all() const { return runners_; }

  void Fail(const std::string& name, const std::string& why) const { primary_->RecordFailure(name, why); }

  void MirrorFail(const BenchmarkRunner& runner, const std::string& name, const std::string& why) const {
    if (&runner != primary_) {
      primary_->RecordFailure(name, why);
    }
  }

  void ReportExtras() const {
    for (const std::unique_ptr<BenchmarkRunner>& runner : owned_) {
      runner->Report();
    }
  }

 private:
  static BenchmarkOptions Options(const BenchmarkOptions& base, const Budget& budget) {
    BenchmarkOptions options = base;
    options.warmup_iterations = budget.warmup;
    options.timed_iterations = budget.iterations;
    options.pipeline_batch = kPipelineBatch;
    return options;
  }

  BenchmarkRunner* primary_;
  std::vector<Budget> budgets_;
  std::vector<std::unique_ptr<BenchmarkRunner>> owned_;
  std::vector<BenchmarkRunner*> runners_;
};

struct Sample {
  bool present = false;
  bool structural_zero = false;
  double median_us = 0.0;
  double p95_us = 0.0;
  double gigabytes_per_second = 0.0;
  double tflops = 0.0;
  const char* mode = "-";
};

const BenchmarkResult* FindResult(const std::vector<BenchmarkRunner*>& runners, const std::string& name,
                                  TimingMode mode) {
  for (const BenchmarkRunner* runner : runners) {
    for (const BenchmarkResult& result : runner->results()) {
      if (result.case_name == name && result.mode == mode) {
        return &result;
      }
    }
  }
  return nullptr;
}

Sample SampleFor(const std::vector<BenchmarkRunner*>& runners, const char* leg, const Config& config) {
  Sample sample;
  const std::string name = CaseName(leg, config);
  for (const TimingMode mode : {TimingMode::kDeviceEvents, TimingMode::kPipelined}) {
    const BenchmarkResult* result = FindResult(runners, name, mode);
    if (result != nullptr && result->latency.median_us > 0.0) {
      sample.present = true;
      sample.median_us = result->latency.median_us;
      sample.p95_us = result->latency.p95_us;
      sample.gigabytes_per_second = result->gigabytes_per_second();
      sample.tflops = result->tflops();
      sample.mode = TimingModeLabel(mode);
      return sample;
    }
  }
  return sample;
}

Sample StructuralZero() {
  Sample sample;
  sample.present = true;
  sample.structural_zero = true;
  sample.mode = "folded";
  return sample;
}

Sample RotateOutputSample(const std::vector<BenchmarkRunner*>& runners, const char* leg, const Config& config) {
  return config.model.folds_output ? StructuralZero() : SampleFor(runners, leg, config);
}

// A decode that rotates its raw query in its own launch has no rotate_q to time.
Sample InLaunchZero() {
  Sample sample = StructuralZero();
  sample.mode = "in-launch";
  return sample;
}

double Speedup(const Sample& baseline, const Sample& candidate) {
  if (!baseline.present || !candidate.present || candidate.median_us <= 0.0) {
    return 0.0;
  }
  return baseline.median_us / candidate.median_us;
}

void PrintUs(const Sample& sample, int width) {
  if (sample.present) {
    std::printf(" %*.2f", width, sample.median_us);
  } else {
    std::printf(" %*s", width, "-");
  }
}

void PrintRatio(double ratio, int width) {
  if (ratio > 0.0) {
    std::printf(" %*.3fx", width - 1, ratio);
  } else {
    std::printf(" %*s", width, "-");
  }
}

void PrintHeaderAndRule(const char* header, int width) {
  std::printf("%s\n", header);
  const size_t indent = 2;
  const size_t rule = width > static_cast<int>(indent) ? static_cast<size_t>(width) - indent : 0;
  std::printf("  %s\n", std::string(rule, '-').c_str());
}

void PrintDouble(double value, int width, int precision) {
  if (value > 0.0) {
    std::printf(" %*.*f", width, precision, value);
  } else {
    std::printf(" %*s", width, "-");
  }
}

struct PrefillRow {
  Sample ingest, rot_q, attn_core, rot_o, e2e, native, native_ingest;
  double sum_of_parts_us = 0.0;
  double speedup = 0.0;
  double gigabytes_per_second = 0.0;
};

struct DecodeRow {
  Sample rot_q, attn_core, rot_o, e2e, native;
  double sum_of_parts_us = 0.0;
  double speedup = 0.0;
  double gigabytes_per_second = 0.0;

  bool measured() const {
    return (rot_q.present && !rot_q.structural_zero) || attn_core.present || e2e.present || native.present;
  }
};

PrefillRow ReadPrefillRow(const std::vector<BenchmarkRunner*>& runners, const Config& config,
                          const Traffic& traffic) {
  PrefillRow row;
  row.ingest = SampleFor(runners, kLegPfIngest, config);
  row.rot_q = SampleFor(runners, kLegPfRotQ, config);
  row.attn_core = SampleFor(runners, kLegPfAttnCore, config);
  row.rot_o = RotateOutputSample(runners, kLegPfRotO, config);
  row.e2e = SampleFor(runners, kLegPfE2E, config);
  row.native = SampleFor(runners, kLegPfV5, config);
  row.native_ingest = SampleFor(runners, kLegPfV5Ingest, config);
  row.sum_of_parts_us = (row.rot_q.present ? row.rot_q.median_us : 0.0) +
                        (row.attn_core.present ? row.attn_core.median_us : 0.0) +
                        (row.rot_o.present ? row.rot_o.median_us : 0.0);
  row.speedup = Speedup(row.native, row.e2e);
  if (row.e2e.present && row.e2e.median_us > 0.0) {
    row.gigabytes_per_second = traffic.pf_e2e / (row.e2e.median_us * 1.0e3);
  }
  return row;
}

DecodeRow ReadDecodeRow(const std::vector<BenchmarkRunner*>& runners, const Config& config, const Traffic& traffic,
                        RotationMode mode) {
  const bool separate = mode == RotationMode::kSeparate;
  DecodeRow row;
  row.rot_q = separate ? SampleFor(runners, kLegDecRotQ, config) : InLaunchZero();
  row.attn_core = SampleFor(runners, separate ? kLegDecAttnCore : kLegDecFusedQAttnCore, config);
  row.rot_o = separate || config.model.folds_output ? RotateOutputSample(runners, kLegDecRotO, config)
                                                     : InLaunchZero();
  row.e2e = SampleFor(runners, separate ? kLegDecE2E : kLegDecFusedQE2E, config);
  row.native = SampleFor(runners, kLegDecV5, config);

  row.sum_of_parts_us = (row.rot_q.present ? row.rot_q.median_us : 0.0) +
                        (row.attn_core.present ? row.attn_core.median_us : 0.0) +
                        (row.rot_o.present ? row.rot_o.median_us : 0.0);
  row.speedup = Speedup(row.native, row.e2e);
  if (row.e2e.present && row.e2e.median_us > 0.0) {
    row.gigabytes_per_second = (separate ? traffic.dec_e2e : traffic.dec_fused_q_e2e) / (row.e2e.median_us * 1.0e3);
  }
  return row;
}

double MedianResidual(const std::vector<double>& residuals) {
  if (residuals.empty()) {
    return 0.0;
  }
  std::vector<double> sorted = residuals;
  std::sort(sorted.begin(), sorted.end());
  return sorted[sorted.size() / 2];
}

void PrintTableA(const std::vector<BenchmarkRunner*>& runners, const std::vector<Config>& sweep,
                 const std::vector<Traffic>& traffic, const std::string& fia_operator) {
  std::printf("\n");
  std::printf("[ascend-bench] ====================================================================\n");
  std::printf("[ascend-bench] TABLE A -- PREFILL PIPELINE PERFORMANCE\n");
  std::printf("[ascend-bench] ====================================================================\n");
  std::printf("[ascend-bench] device time from ACL events, median us per step; native operator: %s\n",
              fia_operator.c_str());
  std::printf("[ascend-bench] TQ_E2E = T_rot_q + T_attn_core + T_rot_o, measured as ONE composite region\n");
  std::printf("[ascend-bench] Context is the prefix S; every prefill step is chunked: min(S, %lld) query tokens "
              "against it (ASCEND_BENCH_TQ_AUDIT_CHUNK)\n\n",
              static_cast<long long>(PrefillChunkTokens()));

  char header[512];
  const int header_width =
      std::snprintf(header, sizeof(header), "  %-17s %13s %3s %9s %4s | %11s %10s %12s %10s %11s %11s | %8s %10s %12s",
                    "Model", "Context", "B", "H_Q/H_KV", "D", "TQ Ingest", "T_rot_q", "T_attn_core", "T_rot_o",
                    "TQ_E2E", "V5_Native", "Speedup", "HBM GB/s", "KV Saved MB");
  PrintHeaderAndRule(header, header_width);

  bool any = false;
  std::vector<double> residuals;
  for (size_t index = 0; index < sweep.size(); ++index) {
    const Config& config = sweep[index];
    const PrefillRow row = ReadPrefillRow(runners, config, traffic[index]);
    if (!row.ingest.present && !row.rot_q.present && !row.attn_core.present && !row.e2e.present &&
        !row.native.present) {
      continue;
    }
    any = true;
    std::ostringstream heads;
    heads << config.model.num_heads << '/' << config.model.num_kv_heads;
    std::printf("  %-17s %13lld %3lld %9s %4lld |", config.model.label, static_cast<long long>(config.seq_len),
                static_cast<long long>(config.batch), heads.str().c_str(),
                static_cast<long long>(config.model.head_size));
    PrintUs(row.ingest, 11);
    PrintUs(row.rot_q, 10);
    PrintUs(row.attn_core, 12);
    PrintUs(row.rot_o, 10);
    PrintUs(row.e2e, 11);
    PrintUs(row.native, 11);
    std::printf(" |");
    PrintRatio(row.speedup, 8);
    PrintDouble(row.gigabytes_per_second, 10, 1);
    PrintDouble(FullModelSavedMib(config, traffic[index]), 12, 1);
    std::printf("\n");

    if (row.e2e.present && row.sum_of_parts_us > 0.0) {
      residuals.push_back(std::fabs(row.e2e.median_us - row.sum_of_parts_us) / row.e2e.median_us);
    }
  }
  if (!any) {
    std::printf("  (no prefill configuration produced a sample)\n");
  }

  std::printf("\n[ascend-bench]   Speedup is V5_Native / TQ_E2E: above 1.000x is TurboQuant ahead. TQ Ingest is\n"
              "[ascend-bench]   reported beside the pipeline and is NOT inside TQ_E2E, matching the accounting\n"
              "[ascend-bench]   this audit was specified with; the native operator's own cache write is timed as\n"
              "[ascend-bench]   pf_v5_ingest and carried in the CSV, so neither side hides a write.\n");
  std::printf("[ascend-bench]   T_rot_o = 0.00 on a folded layer is exact, not missing: W_o' = W_o (I (x) Pi)\n"
              "[ascend-bench]   absorbs the de-rotation offline. Qwen3.5 cannot fold -- attn_output_gate sits\n"
              "[ascend-bench]   between attention and o_proj -- so its column is a measured kernel.\n");
  std::printf("[ascend-bench]   HBM GB/s is the pipeline's compulsory traffic over its measured composite time.\n"
              "[ascend-bench]   KV Saved MB is the whole batch's context over the whole model: fp16 residency minus\n"
              "[ascend-bench]   TurboQuant's for the benchmarked kv head, times KV layers x cached heads per layer\n"
              "[ascend-bench]   (config.json; hybrid linear-attention layers keep no KV, an MLA latent is one slot):\n");
  for (const ModelSpec& model : Models()) {
    std::printf("[ascend-bench]     %-17s %lld KV layers x %lld kv heads\n", model.label,
                static_cast<long long>(model.kv_layers), static_cast<long long>(model.model_kv_heads));
  }
  if (!residuals.empty()) {
    std::printf("[ascend-bench]   composite vs sum-of-parts: median gap %.1f%% over %zu rows. That gap is the\n"
                "[ascend-bench]   launch overhead between stages, which the per-component columns cannot show.\n",
                100.0 * MedianResidual(residuals), residuals.size());
  }
  std::printf("[ascend-bench]   NOT TIMED: the fp32 -> fp16 narrowing between npu_turboquant_rotate_q and FIA.\n"
              "[ascend-bench]   See WHAT IS STILL NOT PRICED in this file's header; its modelled byte cost is in\n"
              "[ascend-bench]   the CSV as untimed_cast_bytes.\n");
  std::fflush(stdout);
}

std::string RotationModesLabel() {
  std::string label;
  for (const RotationMode mode : kRotationModes) {
    if (RotationModeEnabled(mode)) {
      label += (label.empty() ? "" : ", ") + std::string(RotationModeKey(mode));
    }
  }
  return label;
}

void PrintTableB(const std::vector<BenchmarkRunner*>& runners, const std::vector<Config>& sweep,
                 const std::vector<Traffic>& traffic, const std::string& fia_operator, int64_t aiv_num) {
  std::printf("\n");
  std::printf("[ascend-bench] ====================================================================\n");
  std::printf("[ascend-bench] TABLE B -- DECODE LATENCY BREAKDOWN\n");
  std::printf("[ascend-bench] ====================================================================\n");
  std::printf("[ascend-bench] device time from ACL events, median us per decode step; native operator: %s\n",
              fia_operator.c_str());
  std::printf("[ascend-bench] one decode token per sequence; B is the batch of sequences; split policy %s\n"
              "[ascend-bench] (ASCEND_BENCH_TQ_AUDIT_SPLIT); rotation modes %s (%s)\n\n",
              SplitPolicyLabel(SplitPolicy()), RotationModesLabel().c_str(), kRotationModeEnv);

  char header[512];
  const int header_width =
      std::snprintf(header, sizeof(header), "  %-17s %9s %3s %-11s %-19s | %10s %14s %8s %10s %11s %11s | %8s %10s",
                    "Model", "Context", "B", "Path", "Cfg [K/Tsk/Blk/Red]", "T_rot_q", "T_FusedDecode", "Launches",
                    "T_rot_o", "TQ_E2E", "V5_Decode", "Speedup", "Eff GB/s");
  PrintHeaderAndRule(header, header_width);

  bool any = false;
  std::vector<double> residuals;
  std::vector<std::string> deltas;
  for (size_t index = 0; index < sweep.size(); ++index) {
    const Config& config = sweep[index];
    const Traffic& model = traffic[index];
    DecodeRow separate;
    DecodeRow fused_q;
    for (const RotationMode mode : kRotationModes) {
      if (!config.decodes_in_mode(mode)) {
        continue;
      }
      const DecodeRow row = ReadDecodeRow(runners, config, model, mode);
      if (mode == RotationMode::kSeparate) {
        separate = row;
      } else {
        fused_q = row;
      }
      if (!row.measured()) {
        continue;
      }
      any = true;
      std::printf("  %-17s %9lld %3lld %-11s %-19s |", config.model.label, static_cast<long long>(config.seq_len),
                  static_cast<long long>(config.batch), config.path_tag(mode).c_str(),
                  model.dispatch.label().c_str());
      PrintUs(row.rot_q, 10);
      PrintUs(row.attn_core, 14);
      std::printf(" %8d", config.decode_launches(mode));
      PrintUs(row.rot_o, 10);
      PrintUs(row.e2e, 11);
      PrintUs(row.native, 11);
      std::printf(" |");
      PrintRatio(row.speedup, 8);
      PrintDouble(row.gigabytes_per_second, 10, 1);
      std::printf("\n");

      if (row.e2e.present && row.sum_of_parts_us > 0.0) {
        residuals.push_back(std::fabs(row.e2e.median_us - row.sum_of_parts_us) / row.e2e.median_us);
      }
    }
    if (separate.e2e.present && fused_q.e2e.present && separate.e2e.median_us > 0.0) {
      char line[256];
      const double delta = fused_q.e2e.median_us - separate.e2e.median_us;
      std::snprintf(line, sizeof(line), "  %-17s %9lld %3lld | TQ_E2E %+9.2f us (%+6.1f%%)", config.model.label,
                    static_cast<long long>(config.seq_len), static_cast<long long>(config.batch), delta,
                    100.0 * delta / separate.e2e.median_us);
      std::string text(line);
      if (separate.attn_core.present && fused_q.attn_core.present && separate.rot_q.present) {
        std::snprintf(line, sizeof(line), ", T_FusedDecode %+9.2f us against a %.2f us rotate_q launch",
                      fused_q.attn_core.median_us - separate.attn_core.median_us, separate.rot_q.median_us);
        text += line;
      }
      deltas.push_back(text);
    }
  }
  if (!any) {
    std::printf("  (no decode configuration produced a sample)\n");
  }

  const int64_t mix_blocks = std::max<int64_t>(1, aiv_num / tqh::kVectorSubcoresPerBlock);
  std::printf("\n[ascend-bench]   Speedup is V5_Decode / TQ_E2E: above 1.000x is TurboQuant ahead. TQ_E2E is one\n"
              "[ascend-bench]   composite ACL-event region over the step's launches; T_FusedDecode is the\n"
              "[ascend-bench]   attention core, ONE launch on either path. Launches counts the host dispatches\n"
              "[ascend-bench]   of one decode step: rotate-q (-Sep only), the core, and rotate-o if unfolded.\n");
  std::printf("[ascend-bench]   Path -Sep launches rotate_q ahead of the decode and, unfolded, rotate_q over its\n"
              "[ascend-bench]   output. Path -FusedQ hands the decode the raw fp16 query, which it rotates in the\n"
              "[ascend-bench]   same launch ahead of its split tasks, and un-rotates the output before its fp16\n"
              "[ascend-bench]   write (PRE_ROTATED=false, kUnrotated, Cube kv4fp8 only): T_rot_q and T_rot_o are\n"
              "[ascend-bench]   0.00 in-launch and T_FusedDecode includes both. Both share the same Cfg and V5.\n");
  std::printf("[ascend-bench]   Cfg [K/Tsk/Blk/Red]: K context splits per sequence; Tsk tasks the launch\n"
              "[ascend-bench]   schedules (Cube B x H_KV x K x head chunks, AIV B x H_Q x K); Blk the blocks it\n"
              "[ascend-bench]   spans, of %lld MIX blocks on the Cube path and %lld vector cores on the AIV path;\n"
              "[ascend-bench]   Red ON when the launch reduces split partials after a SyncAll (NeedsReduction).\n",
              static_cast<long long>(mix_blocks), static_cast<long long>(aiv_num));
  std::printf("[ascend-bench]   Every model takes the Cube decode, whatever its GQA group or head size;\n"
              "[ascend-bench]   ASCEND_BENCH_TQ_AUDIT_PATH=aiv forces the vector-only path for an A/B.\n");
  std::printf("[ascend-bench]   Eff GB/s is the step's compulsory traffic over its measured composite time.\n"
              "[ascend-bench]   KV compression (fp16 over TurboQuant residency) is in the decode CSV.\n");
  if (!residuals.empty()) {
    std::printf("[ascend-bench]   composite vs sum-of-parts: median gap %.1f%% over %zu rows.\n",
                100.0 * MedianResidual(residuals), residuals.size());
  }
  if (!deltas.empty()) {
    std::printf("\n[ascend-bench]   in-launch basis change (-FusedQ) against separate rotate_q launches (-Sep), "
                "negative is -FusedQ faster:\n");
    for (const std::string& line : deltas) {
      std::printf("%s\n", line.c_str());
    }
  }
  std::fflush(stdout);
}

void WriteCell(std::ofstream& csv, const Sample& sample) {
  csv << ',';
  if (sample.present) {
    csv << sample.median_us;
  }
}

void WritePrefillCsv(const std::vector<BenchmarkRunner*>& runners, const std::vector<Config>& sweep,
                     const std::vector<Traffic>& traffic, const std::string& path,
                     const std::string& fia_operator) {
  std::ofstream csv(path, std::ios::trunc);
  if (!csv) {
    std::printf("[ascend-bench] could not open ASCEND_BENCH_TQ_AUDIT_PREFILL_CSV='%s' for writing\n",
                path.c_str());
    return;
  }
  csv << "model,regime,seq_len,chunk_tokens,batch,num_heads,num_kv_heads,head_dim,path,native_operator,"
      << "tq_ingest_us,t_rot_q_us,t_attn_core_us,t_rot_o_us,rot_o_is_structural_zero,"
      << "tq_e2e_us,tq_e2e_sum_of_parts_us,tq_e2e_with_ingest_us,"
      << "v5_native_us,v5_native_ingest_us,net_speedup,net_speedup_with_ingest,"
      << "hbm_gbps,attn_core_gbps,v5_gbps,attn_core_tflops,v5_tflops,"
      << "fp16_kv_bytes,tq_kv_bytes,kv_saved_mib,compression_ratio,"
      << "e2e_bytes,attention_flops,untimed_cast_bytes,warmup,iterations,timing_mode\n";

  for (size_t index = 0; index < sweep.size(); ++index) {
    const Config& config = sweep[index];
    const Traffic& model = traffic[index];
    const PrefillRow row = ReadPrefillRow(runners, config, model);

    csv << config.model.key << ',' << config.regime << ',' << config.seq_len << ',' << config.chunk << ','
        << config.batch << ',' << config.model.num_heads << ',' << config.model.num_kv_heads << ','
        << config.model.head_size << ',' << PathLabel(config.path) << ',' << fia_operator;
    WriteCell(csv, row.ingest);
    WriteCell(csv, row.rot_q);
    WriteCell(csv, row.attn_core);
    WriteCell(csv, row.rot_o);
    csv << ',' << (row.rot_o.structural_zero ? 1 : 0);
    WriteCell(csv, row.e2e);
    csv << ',';
    if (row.sum_of_parts_us > 0.0) {
      csv << row.sum_of_parts_us;
    }
    csv << ',';
    if (row.e2e.present && row.ingest.present) {
      csv << row.e2e.median_us + row.ingest.median_us;
    }
    WriteCell(csv, row.native);
    WriteCell(csv, row.native_ingest);
    csv << ',';
    if (row.speedup > 0.0) {
      csv << row.speedup;
    }
    csv << ',';
    if (row.e2e.present && row.ingest.present && row.native.present && row.native_ingest.present) {
      const double tq = row.e2e.median_us + row.ingest.median_us;
      if (tq > 0.0) {
        csv << (row.native.median_us + row.native_ingest.median_us) / tq;
      }
    }
    csv << ',';
    if (row.gigabytes_per_second > 0.0) {
      csv << row.gigabytes_per_second;
    }
    csv << ',';
    if (row.attn_core.present) {
      csv << row.attn_core.gigabytes_per_second;
    }
    csv << ',';
    if (row.native.present) {
      csv << row.native.gigabytes_per_second;
    }
    csv << ',';
    if (row.attn_core.present) {
      csv << row.attn_core.tflops;
    }
    csv << ',';
    if (row.native.present) {
      csv << row.native.tflops;
    }
    csv << ',' << model.fp16_kv_bytes << ',' << model.tq_kv_bytes << ',' << model.saved_mib() << ','
        << model.compression_ratio() << ',' << model.pf_e2e << ',' << model.pf_flops << ','
        << model.pf_untimed_cast << ',' << config.budget.warmup << ',' << config.budget.iterations << ','
        << row.e2e.mode << '\n';
  }
  std::printf("[ascend-bench] Table A written to %s\n", path.c_str());
}

// The timed columns of one decode CSV row, from t_rot_q_us on.
void WriteDecodeCsvTimings(std::ofstream& csv, const Config& config, const Traffic& model, const DecodeRow& row,
                           RotationMode mode) {
  WriteCell(csv, row.rot_q);
  WriteCell(csv, row.attn_core);
  csv << ',' << config.decode_launches(mode);
  WriteCell(csv, row.rot_o);
  csv << ',' << (row.rot_o.structural_zero ? 1 : 0);
  WriteCell(csv, row.e2e);
  csv << ',';
  if (row.sum_of_parts_us > 0.0) {
    csv << row.sum_of_parts_us;
  }
  WriteCell(csv, row.native);
  csv << ',';
  if (row.speedup > 0.0) {
    csv << row.speedup;
  }
  csv << ',';
  if (row.gigabytes_per_second > 0.0) {
    csv << row.gigabytes_per_second;
  }
  csv << ',';
  if (row.attn_core.present) {
    csv << row.attn_core.gigabytes_per_second;
  }
  csv << ',';
  if (row.native.present) {
    csv << row.native.gigabytes_per_second;
  }
  csv << ',';
  if (row.attn_core.present) {
    csv << row.attn_core.tflops;
  }
  csv << ',';
  if (row.native.present) {
    csv << row.native.tflops;
  }
  csv << ',' << model.fp16_kv_bytes << ',' << model.tq_kv_bytes << ',' << model.saved_mib() << ','
      << model.compression_ratio() << ','
      << (mode == RotationMode::kSeparate ? model.dec_e2e : model.dec_fused_q_e2e) << ',' << model.dec_flops << ',';
  if (row.e2e.present) {
    csv << row.e2e.p95_us;
  }
  csv << ',';
  if (row.native.present) {
    csv << row.native.p95_us;
  }
  csv << ',' << config.budget.warmup << ',' << config.budget.iterations << ',' << row.e2e.mode << ','
      << SplitPolicyLabel(SplitPolicy()) << '\n';
}

// One row per (configuration, rotation mode) the sweep decodes in.
void WriteDecodeCsv(const std::vector<BenchmarkRunner*>& runners, const std::vector<Config>& sweep,
                    const std::vector<Traffic>& traffic, const std::string& path,
                    const std::string& fia_operator) {
  std::ofstream csv(path, std::ios::trunc);
  if (!csv) {
    std::printf("[ascend-bench] could not open ASCEND_BENCH_TQ_AUDIT_DECODE_CSV='%s' for writing\n", path.c_str());
    return;
  }
  csv << "model,regime,seq_len,batch,num_heads,num_kv_heads,head_dim,path,rotation_mode,"
      << "split_k,total_tasks,active_blocks,device_blocks,needs_reduction,native_operator,"
      << "t_rot_q_us,t_fused_decode_us,decode_launches,t_rot_o_us,rot_o_is_structural_zero,"
      << "tq_e2e_us,tq_e2e_sum_of_parts_us,v5_decode_us,net_speedup,"
      << "effective_gbps,attn_core_gbps,v5_gbps,attn_core_tflops,v5_tflops,"
      << "fp16_kv_bytes,tq_kv_bytes,kv_saved_mib,compression_ratio,"
      << "e2e_bytes,attention_flops,tq_e2e_p95_us,v5_p95_us,warmup,iterations,timing_mode,split_policy\n";

  for (size_t index = 0; index < sweep.size(); ++index) {
    const Config& config = sweep[index];
    const Traffic& model = traffic[index];
    for (const RotationMode mode : kRotationModes) {
      if (!config.decodes_in_mode(mode)) {
        continue;
      }
      const DecodeRow row = ReadDecodeRow(runners, config, model, mode);
      const DecodeDispatch& dispatch = model.dispatch;

      csv << config.model.key << ',' << config.regime << ',' << config.seq_len << ',' << config.batch << ','
          << config.model.num_heads << ',' << config.model.num_kv_heads << ',' << config.model.head_size << ','
          << PathLabel(config.path) << ',' << RotationModeKey(mode) << ',' << dispatch.split_k << ','
          << dispatch.total_tasks << ',' << dispatch.active_blocks << ',' << dispatch.device_blocks << ','
          << (dispatch.needs_reduction ? 1 : 0) << ',' << fia_operator;
      WriteDecodeCsvTimings(csv, config, model, row, mode);
    }
  }
  std::printf("[ascend-bench] Table B written to %s\n", path.c_str());
}

double RotatedBasisCosine(const Scenario& scenario, const Config& config, aclrtStream stream) {
  scenario.EnqueuePrefillAttnCore(stream);
  scenario.EnqueuePrefillNative(stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  const std::vector<float> plain = scenario.PrefillNativeOutput();
  const std::vector<float> folded = tqh::UnrotateHeads(scenario.PrefillTqOutput(), config.model.head_size);
  return turboquant_ref::cpu_fidelity(folded, plain).cosine_similarity;
}

double DecodeTiePointCosine(const Scenario& scenario, const Config& config, aclrtStream stream) {
  scenario.EnqueueDecodeE2E(stream);
  scenario.EnqueueDecodeNative(stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  const std::vector<float> native = scenario.DecodeNativeOutput();
  const std::vector<float> folded = tqh::UnrotateHeads(scenario.DecodeTqOutput(), config.model.head_size);
  return turboquant_ref::cpu_fidelity(folded, native).cosine_similarity;
}

void PrintBanner(const std::vector<Config>& sweep, int64_t aiv_num, bool aiv_queried) {
  std::printf("[ascend-bench] TurboQuant end-to-end audit against the native CANN V5 attention operator\n");
  std::printf("[ascend-bench]   vector cores = %lld%s\n", static_cast<long long>(aiv_num),
              aiv_queried ? " (from aclGetDeviceCapability)" : " (assumed; the runtime declined to answer)");
  std::printf("[ascend-bench]\n");
  std::printf("[ascend-bench]   THE NATIVE OPERATOR IS aclnnFusedInferAttentionScoreV5, in both roles.\n"
              "[ascend-bench]   aclnnPromptFlashAttentionV5 and aclnnFusionIncrementalAttention DO NOT EXIST in\n"
              "[ascend-bench]   this toolkit -- the PFA family stops at V3 and the IFA family at V4, and the\n"
              "[ascend-bench]   second name appears in no header at all. FIA V5 is the unified interface that\n"
              "[ascend-bench]   replaces both on an Ascend950 (V1..V4 of it return 361001), and it covers the\n"
              "[ascend-bench]   prompt role and the incremental role through its argument list. V2 is the\n"
              "[ascend-bench]   fallback; every table says which one actually planned.\n");
  std::printf("[ascend-bench]\n");
  std::printf("[ascend-bench]   NO NUMBER BELOW HAS EVER BEEN TAKEN ON SILICON. No Ascend 950PR part has been\n"
              "[ascend-bench]   available to this project; the kernels are verified on the arch35 camodel.\n");
  std::printf("[ascend-bench]\n");
  std::printf("[ascend-bench]   decode: split policy %s (ASCEND_BENCH_TQ_AUDIT_SPLIT), rotation modes %s (%s)\n",
              SplitPolicyLabel(SplitPolicy()), RotationModesLabel().c_str(), kRotationModeEnv);
  if (!RotationModeValid()) {
    std::printf("[ascend-bench]   %s='%s' is not separate, fused_prologue or both; timing both\n", kRotationModeEnv,
                EnvString(kRotationModeEnv).c_str());
  }
  std::printf("[ascend-bench]\n");

  if (sweep.empty()) {
    std::printf("[ascend-bench]   the sweep is empty: every configuration was filtered out by an\n"
                "[ascend-bench]   ASCEND_BENCH_TQ_AUDIT_* selector.\n");
    std::fflush(stdout);
    return;
  }

  std::printf("[ascend-bench]   %zu configurations:\n", sweep.size());
  std::string previous_model;
  for (const Config& config : sweep) {
    if (previous_model != config.model.key) {
      previous_model = config.model.key;
      std::printf("[ascend-bench]     %-17s D=%-4lld H_Q=%-3lld H_KV=%-2lld  %-6s  %s\n", config.model.label,
                  static_cast<long long>(config.model.head_size),
                  static_cast<long long>(config.model.num_heads),
                  static_cast<long long>(config.model.num_kv_heads),
                  config.model.folds_output ? "folded" : "UNFOLD", PathLabel(config.path));
      std::printf("[ascend-bench]       %s\n", config.model.note);
    }
  }
  std::fflush(stdout);
}

}

void BuildSuite(BenchmarkRunner& primary) {
  bool aiv_queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&aiv_queried);

  const std::vector<Config> sweep = BuildSweep();
  PrintBanner(sweep, aiv_num, aiv_queried);
  if (sweep.empty()) {
    return;
  }

  std::vector<Budget> budgets;
  for (const Config& config : sweep) {
    if (std::find(budgets.begin(), budgets.end(), config.budget) == budgets.end()) {
      budgets.push_back(config.budget);
    }
  }
  RunnerSet runners(primary, budgets);
  for (const Budget& budget : budgets) {
    std::printf("[ascend-bench]   budget: warmup %d, %d timed iterations, pipeline_batch %d\n", budget.warmup,
                budget.iterations, kPipelineBatch);
  }
  std::fflush(stdout);

  const bool run_prefill = PhaseEnabled("prefill");
  const bool run_decode = PhaseEnabled("decode");

  size_t basis_check = sweep.size();
  size_t tie_point_check = sweep.size();
  for (size_t index = 0; index < sweep.size(); ++index) {
    const auto weight = [](const Config& config) {
      return config.chunk_tokens() * config.model.num_heads * config.model.head_size;
    };
    if (basis_check == sweep.size() || weight(sweep[index]) < weight(sweep[basis_check])) {
      basis_check = index;
    }
    const bool exact = static_cast<size_t>(sweep[index].context_tokens() * sweep[index].model.num_kv_heads *
                                           sweep[index].model.head_size) <= kExactContextElements;
    if (exact && (tie_point_check == sweep.size() ||
                  sweep[index].context_tokens() < sweep[tie_point_check].context_tokens())) {
      tie_point_check = index;
    }
  }

  std::vector<Traffic> traffic;
  traffic.reserve(sweep.size());
  std::string fia_operator = "none";

  for (size_t index = 0; index < sweep.size(); ++index) {
    const Config& config = sweep[index];
    BenchmarkRunner& runner = runners.For(config.budget);

    const DecodeDispatch dispatch = PlanDispatch(config, aiv_num);
    traffic.push_back(ModelTraffic(config, dispatch));

    size_t free_hbm = 0;
    size_t total_hbm = 0;
    const bool known = aclrtGetMemInfo(ACL_HBM_MEM, &free_hbm, &total_hbm) == ACL_SUCCESS && free_hbm > 0;
    const double needed = ScenarioBytes(config);
    if (known && needed > kHbmBudget * static_cast<double>(free_hbm)) {
      std::ostringstream why;
      why << "needs ~" << needed / (1024.0 * 1024.0 * 1024.0) << " GiB of HBM; the budget is "
          << kHbmBudget * 100.0 << "% of " << static_cast<double>(free_hbm) / (1024.0 * 1024.0 * 1024.0)
          << " GiB free";
      for (const char* leg : kPrefillLegs) {
        runner.Skip(CaseName(leg, config), why.str());
      }
      for (const char* leg : kDecodeLegs) {
        runner.Skip(CaseName(leg, config), why.str());
      }
      std::printf("[ascend-bench] %s: skipped -- %s\n", config.id().c_str(), why.str().c_str());
      std::fflush(stdout);
      continue;
    }

    std::unique_ptr<Scenario> scenario;
    try {
      scenario.reset(new Scenario(config, aiv_num));
    } catch (const std::exception& error) {
      runners.Fail(CaseName("setup", config), error.what());
      continue;
    }
    if (!scenario->fia_operator().empty()) {
      fia_operator = scenario->fia_operator();
    }
    std::printf("\n[ascend-bench] %-17s S=%lld B=%lld D=%lld H_Q=%lld H_KV=%lld | %s path, chunk %lld, "
                "blocks/seq %lld, pool %lld, Cfg [K/Tsk/Blk/Red] %s of %lld blocks | native: %s\n",
                config.model.label, static_cast<long long>(config.seq_len),
                static_cast<long long>(config.batch), static_cast<long long>(config.model.head_size),
                static_cast<long long>(config.model.num_heads),
                static_cast<long long>(config.model.num_kv_heads), PathLabel(config.path),
                static_cast<long long>(config.chunk), static_cast<long long>(config.blocks_per_seq()),
                static_cast<long long>(config.pool_blocks()), dispatch.label().c_str(),
                static_cast<long long>(dispatch.device_blocks),
                scenario->decode_available() ? scenario->fia_operator().c_str() : "unavailable");
    if (scenario->num_splits() != dispatch.split_k ||
        static_cast<int64_t>(scenario->block_dim()) != dispatch.active_blocks) {
      runners.Fail(CaseName("dispatch", config), "the scenario launches a different grid than Table B reports");
    }
    if (!scenario->prefill_available() || !scenario->decode_available()) {
      std::printf("[ascend-bench]   native operator note: %s\n", scenario->fia_note().c_str());
    }
    std::fflush(stdout);

    try {
      scenario->FillCache(runner.stream());
      scenario->Prime(runner.stream());
    } catch (const std::exception& error) {
      runners.Fail(CaseName("fill", config), error.what());
      continue;
    }

    if (index == basis_check && scenario->prefill_available()) {
      try {
        const double cosine = RotatedBasisCosine(*scenario, config, runner.stream());
        std::printf("[ascend-bench]   rotated-basis identity: FIA(Q~,K~,V~) un-rotated vs FIA(Q,K,V), "
                    "cos = %.9f (bound %.4f)\n",
                    cosine, kRotatedBasisMinCosine);
        if (!(cosine > kRotatedBasisMinCosine)) {
          runners.Fail(CaseName("rotated_basis_identity", config),
                       "attention in the rotated basis does not un-rotate to attention in the plain one");
        }
      } catch (const std::exception& error) {
        runners.Fail(CaseName("rotated_basis_identity", config), error.what());
      }
    }
    if (index == tie_point_check && scenario->decode_available() && scenario->exact_context()) {
      try {
        const double cosine = DecodeTiePointCosine(*scenario, config, runner.stream());
        std::printf("[ascend-bench]   decode tie-point: TurboQuant un-rotated vs native V5 over the same "
                    "context, cos = %.6f (bound %.2f)\n",
                    cosine, kDecodeTiePointMinCosine);
        if (!(cosine > kDecodeTiePointMinCosine)) {
          runners.Fail(CaseName("decode_tie_point", config),
                       "the TurboQuant decode and the native decode are not attending over the same context");
        }
      } catch (const std::exception& error) {
        runners.Fail(CaseName("decode_tie_point", config), error.what());
      }
    }
    // Every configuration that times the raw-query decode first checks it against the pre-rotated one: the
    // raw entry had executed on the camodel at B = 1, D = 256 only.
    if (run_decode && config.decodes_in_mode(RotationMode::kFusedPrologue)) {
      try {
        const Scenario::RotationAgreement agreement = scenario->CompareRotationModes(runner.stream());
        std::printf("[ascend-bench]   rotation modes: the in-launch rotation differs from rotate_q in %zu of %zu "
                    "fp32 words; decode outputs agree to cos = %.9f (bound %.4f)\n",
                    agreement.word_mismatches, agreement.rotated_words, agreement.output_cosine,
                    kRotationModeMinCosine);
        if (!(agreement.output_cosine > kRotationModeMinCosine)) {
          runners.Fail(CaseName("rotation_mode_agreement", config),
                       "the raw-query decode does not reproduce the pre-rotated decode");
        }
      } catch (const std::exception& error) {
        runners.Fail(CaseName("rotation_mode_agreement", config), error.what());
      }
    }
    std::fflush(stdout);

    const Traffic& model = traffic.back();
    const Scenario* sc = scenario.get();
    const auto run_leg = [&](const char* leg, double flops, double bytes, int tasks,
                             std::function<void(aclrtStream)> launch, std::function<double()> checksum) {
      const std::string name = CaseName(leg, config);
      if (!LegEnabled(leg)) {
        runner.Skip(name, "not in ASCEND_BENCH_TQ_AUDIT_LEGS");
        return;
      }
      try {
        BenchmarkCase benchmark_case;
        benchmark_case.name = name;
        benchmark_case.flops_per_iteration = flops;
        benchmark_case.bytes_per_iteration = bytes;
        benchmark_case.tasks_per_launch = tasks;
        benchmark_case.launch = std::move(launch);
        benchmark_case.checksum = std::move(checksum);
        runner.Run(benchmark_case);
      } catch (const std::exception& error) {
        runner.RecordFailure(name, error.what());
        runners.MirrorFail(runner, name, error.what());
      }
    };
    const int rot_o_tasks = config.model.folds_output ? 0 : 1;

    if (run_decode) {
      const bool separate = config.decodes_in_mode(RotationMode::kSeparate);
      const bool fused_q = config.decodes_in_mode(RotationMode::kFusedPrologue);
      const std::string separate_off =
          std::string("rotation mode separate is not selected by ") + kRotationModeEnv;
      const std::string fused_q_off =
          config.path != PathMode::kCube
              ? std::string("the raw-query decode exists on the Cube path only")
              : std::string("rotation mode fused_prologue is not selected by ") + kRotationModeEnv;

      if (separate) {
        run_leg(kLegDecRotQ, 0.0, model.dec_rot_q, 1, [sc](aclrtStream s) { sc->EnqueueDecodeRotateQ(s); },
                [sc]() { return sc->DecodeRotatedQueryChecksum(); });
        run_leg(kLegDecAttnCore, model.dec_flops, model.dec_attn_core, 1,
                [sc](aclrtStream s) { sc->EnqueueDecodeAttnCore(s); },
                [sc]() { return sc->DecodeTqChecksum(); });
      } else {
        runner.Skip(CaseName(kLegDecRotQ, config), separate_off);
        runner.Skip(CaseName(kLegDecAttnCore, config), separate_off);
      }

      if (fused_q) {
        run_leg(kLegDecFusedQAttnCore, model.dec_flops, model.dec_fused_q_attn_core, 1,
                [sc](aclrtStream s) { sc->EnqueueDecodeFusedQ(s); },
                [sc]() { return sc->DecodeTqChecksum(); });
      } else {
        runner.Skip(CaseName(kLegDecFusedQAttnCore, config), fused_q_off);
      }

      if (config.model.folds_output) {
        runner.Skip(CaseName(kLegDecRotO, config),
                    "W_o is folded: the de-rotation is an offline weight transform and costs 0.0 us at runtime");
      } else {
        run_leg(kLegDecRotO, 0.0, model.dec_rot_o, 1, [sc](aclrtStream s) { sc->EnqueueDecodeRotateO(s); },
                [sc]() { return sc->DecodeRotatedOutChecksum(); });
      }

      if (separate) {
        run_leg(kLegDecE2E, model.dec_flops, model.dec_e2e, config.decode_launches(RotationMode::kSeparate),
                [sc](aclrtStream s) { sc->EnqueueDecodeE2E(s); },
                [sc]() { return sc->DecodePipelineChecksum(); });
      } else {
        runner.Skip(CaseName(kLegDecE2E, config), separate_off);
      }

      if (fused_q) {
        run_leg(kLegDecFusedQE2E, model.dec_flops, model.dec_fused_q_e2e,
                config.decode_launches(RotationMode::kFusedPrologue),
                [sc](aclrtStream s) { sc->EnqueueDecodeFusedQE2E(s); },
                [sc]() { return sc->DecodeTqChecksum(); });
      } else {
        runner.Skip(CaseName(kLegDecFusedQE2E, config), fused_q_off);
      }

      if (!sc->decode_available()) {
        runner.Skip(CaseName(kLegDecV5, config), sc->fia_note());
      } else {
        run_leg(kLegDecV5, model.dec_flops, model.dec_v5, 1, [sc](aclrtStream s) { sc->EnqueueDecodeNative(s); },
                [sc]() { return sc->DecodeNativeChecksum(); });
      }
    }

    if (run_prefill) {
      run_leg(kLegPfIngest, 0.0, model.pf_ingest, 1, [sc](aclrtStream s) { sc->EnqueuePrefillIngest(s); },
              [sc]() { return sc->ScaleChecksum(); });
      run_leg(kLegPfRotQ, 0.0, model.pf_rot_q, 1, [sc](aclrtStream s) { sc->EnqueuePrefillRotateQ(s); },
              [sc]() { return sc->PrefillRotatedQueryChecksum(); });

      if (!sc->prefill_available()) {
        for (const char* leg : {kLegPfAttnCore, kLegPfRotO, kLegPfE2E, kLegPfV5}) {
          runner.Skip(CaseName(leg, config), sc->fia_note());
        }
      } else {
        run_leg(kLegPfAttnCore, model.pf_flops, model.pf_attn_core, 1,
                [sc](aclrtStream s) { sc->EnqueuePrefillAttnCore(s); },
                [sc]() { return sc->PrefillTqChecksum(); });

        if (config.model.folds_output) {
          runner.Skip(CaseName(kLegPfRotO, config),
                      "W_o is folded: the de-rotation is an offline weight transform and costs 0.0 us at runtime");
        } else {
          run_leg(kLegPfRotO, 0.0, model.pf_rot_o, 1, [sc](aclrtStream s) { sc->EnqueuePrefillRotateO(s); },
                  [sc]() { return sc->PrefillRotatedOutChecksum(); });
        }

        run_leg(kLegPfE2E, model.pf_flops, model.pf_e2e, 1 + 1 + rot_o_tasks,
                [sc](aclrtStream s) { sc->EnqueuePrefillE2E(s); },
                [sc]() { return sc->PrefillPipelineChecksum(); });
        run_leg(kLegPfV5, model.pf_flops, model.pf_v5, 1, [sc](aclrtStream s) { sc->EnqueuePrefillNative(s); },
                [sc]() { return sc->PrefillNativeChecksum(); });
      }

      if (!sc->native_write_available()) {
        runner.Skip(CaseName(kLegPfV5Ingest, config), sc->native_write_note().empty()
                                                          ? std::string("the native write was not planned")
                                                          : sc->native_write_note());
      } else {
        run_leg(kLegPfV5Ingest, 0.0, model.pf_v5_ingest, 1,
                [sc](aclrtStream s) { sc->EnqueuePrefillNativeIngest(s); },
                [sc]() { return sc->NativeCacheChecksum(); });
      }
    }
  }

  runners.ReportExtras();

  if (run_prefill) {
    PrintTableA(runners.all(), sweep, traffic, fia_operator);
    const std::string path = EnvString("ASCEND_BENCH_TQ_AUDIT_PREFILL_CSV");
    if (!path.empty()) {
      WritePrefillCsv(runners.all(), sweep, traffic, path, fia_operator);
    }
  }
  if (run_decode) {
    PrintTableB(runners.all(), sweep, traffic, fia_operator, aiv_num);
    const std::string path = EnvString("ASCEND_BENCH_TQ_AUDIT_DECODE_CSV");
    if (!path.empty()) {
      WriteDecodeCsv(runners.all(), sweep, traffic, path, fia_operator);
    }
  }
  std::printf("\n[ascend-bench] the raw per-case table for the first iteration budget follows below.\n");
  std::fflush(stdout);
}

}
}
}
