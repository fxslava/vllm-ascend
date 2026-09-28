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

#pragma once

// The data model of the TurboQuant end-to-end audit: the sweep (models, regimes,
// rotation modes, budgets), the per-configuration traffic accounting, the leg
// names and the environment selectors, plus the device-buffer toolkit the
// benchmark (bench_device_950pr_turboquant.cpp) and the msprof trace
// (prof_device_950pr_msprof_trace.cpp) both build their scenarios with. The
// tables and CSV exporters that print this model live in
// turboquant_audit_report.hpp.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "ascend950_shapes.hpp"
#include "device_buffer.hpp"
#include "env_utils.hpp"
#include "turboquant_audit_models.hpp"
#include "turboquant_launch.hpp"

namespace vllm_ascend {
namespace test {
namespace turboquant_audit {

namespace tqh = turboquant_host;

struct Config;

// -----------------------------------------------------------------------------
// Shared scenario constants
// -----------------------------------------------------------------------------

inline constexpr int64_t kBlockSize = shapes950::kDefaultBlockSize;

inline constexpr int64_t kCausalMaskSide = 2048;

inline constexpr int64_t kPatternTokens = 1024;

// The leading slice read back to host for a checksum, in elements.
inline constexpr size_t kLeadingReadbackElements = size_t{1} << 20;

inline constexpr double kHalfBytes = 2.0;
inline constexpr double kFloatBytes = 4.0;
inline constexpr double kMiB = 1024.0 * 1024.0;

// The per-configuration iteration budgets: the short one everywhere, the ultra
// one from kUltraContextThreshold up, and both overridable by environment.
inline constexpr int kShortWarmup = 5;
inline constexpr int kShortIterations = 50;
inline constexpr int kUltraWarmup = 1;
inline constexpr int kUltraIterations = 3;

inline constexpr int64_t kUltraContextThreshold = 262144;

// Configurations this small multiply their pool by kPoolOversizeFactor so the
// cache scatter has blocks to stray into.
inline constexpr int64_t kOversizedPoolContextLimit = 8192;
inline constexpr int64_t kPoolOversizeFactor = 4;

// -----------------------------------------------------------------------------
// Rotation modes: where a decode step changes basis. kSeparate launches rotate_q
// ahead of the decode (PRE_ROTATED = true) and, for an unfolded W_o, rotate_q
// again over its output; kFusedPrologue hands the raw fp16 query to the decode,
// which rotates it and, unfolded, un-rotates its output inside the same launch
// (PRE_ROTATED = false, kUnrotated; fused cases (g) and (h); kv4fp8 on the Cube
// path only). A Qwen3.5 layer would also gate in that launch (kGated); the bench
// leaves the gate out on both sides, as it does for the native decode.
// -----------------------------------------------------------------------------

enum class RotationMode { kSeparate, kFusedPrologue };

inline constexpr RotationMode kRotationModes[] = {RotationMode::kSeparate, RotationMode::kFusedPrologue};

inline constexpr const char* kRotationModeEnv = "ASCEND_BENCH_TQ_ROTATION_MODE";
inline constexpr const char* kRotationModeBoth = "both";

inline const char* RotationModeKey(RotationMode mode) {
  return mode == RotationMode::kSeparate ? "separate" : "fused_prologue";
}

inline bool RotationModeValid() {
  const std::string raw = env::String(kRotationModeEnv);
  return raw.empty() || raw == kRotationModeBoth || raw == RotationModeKey(RotationMode::kSeparate) ||
         raw == RotationModeKey(RotationMode::kFusedPrologue);
}

// ASCEND_BENCH_TQ_ROTATION_MODE=separate|fused_prologue|both, both by default; a
// banner reports any other value, which is read as both.
inline bool RotationModeEnabled(RotationMode mode) {
  const std::string raw = env::String(kRotationModeEnv);
  if (raw == RotationModeKey(RotationMode::kSeparate) || raw == RotationModeKey(RotationMode::kFusedPrologue)) {
    return raw == RotationModeKey(mode);
  }
  return true;
}

std::string RotationModesLabel();

// -----------------------------------------------------------------------------
// The unpack ablation, which exists to price one phase of the Cube decode on
// silicon: the vector-pipe expand of the packed KV plane into Cube operands. The
// ablated launch (turboquant_mm_fused_decode_nounpack_impl) is the shipping
// decode with that expand and nothing else removed, so
//
//     T(dec_nounpack) - T(dec_attn_core) = the unpack phase, at fixed traffic.
//
// ASCEND_BENCH_TQ_UNPACK=on|off|both|gather|all, on by default. Three expands of
// the same decode share one launch shape and one traffic model:
//
//   on      the shipping affine expand alone (kLegDecAttnCore).
//   off     the ablation in its place (kLegDecNoUnpack): no expand at all. Its
//           output is all-zero by construction and its checksum means nothing.
//   both    the pair above, which is the measurement above.
//   gather  the 16-entry UB Gather expand in place of the shipping expand
//           (kLegDecGather): a correct variant, whose checksum must match the
//           unablated decode's.
//   all     all three, which is what prices Option C against both.
// -----------------------------------------------------------------------------

inline constexpr const char* kUnpackEnv = "ASCEND_BENCH_TQ_UNPACK";
inline constexpr const char* kUnpackOn = "on";
inline constexpr const char* kUnpackOff = "off";
inline constexpr const char* kUnpackBoth = "both";
inline constexpr const char* kUnpackGather = "gather";
inline constexpr const char* kUnpackAll = "all";

inline bool UnpackModeValid() {
  const std::string raw = env::String(kUnpackEnv);
  return raw.empty() || raw == kUnpackOn || raw == kUnpackOff || raw == kUnpackBoth || raw == kUnpackGather ||
         raw == kUnpackAll;
}

// The unablated legs. Both replacements ("off" and "gather") stand in for them rather than joining them.
inline bool UnpackStandardEnabled() {
  const std::string raw = env::String(kUnpackEnv);
  return raw != kUnpackOff && raw != kUnpackGather;
}

// The ablated leg. Off unless asked for by name.
inline bool UnpackAblationEnabled() {
  const std::string raw = env::String(kUnpackEnv);
  return raw == kUnpackOff || raw == kUnpackBoth || raw == kUnpackAll;
}

// The gather-expand leg. Off unless asked for by name.
inline bool UnpackGatherEnabled() {
  const std::string raw = env::String(kUnpackEnv);
  return raw == kUnpackGather || raw == kUnpackAll;
}

// Anything the parser does not recognise reads back as "on", which is the mode such a run actually times.
const char* UnpackModeLabel();

// -----------------------------------------------------------------------------
// Legs
// -----------------------------------------------------------------------------

inline constexpr const char* kLegPfIngest = "pf_ingest";
inline constexpr const char* kLegPfRotQ = "pf_rot_q";
inline constexpr const char* kLegPfAttnCore = "pf_attn_core";
inline constexpr const char* kLegPfRotO = "pf_rot_o";
inline constexpr const char* kLegPfE2E = "pf_e2e";
inline constexpr const char* kLegPfV5 = "pf_v5";
inline constexpr const char* kLegPfV5Ingest = "pf_v5_ingest";

inline constexpr const char* kLegDecRotQ = "dec_rot_q";
inline constexpr const char* kLegDecAttnCore = "dec_attn_core";
inline constexpr const char* kLegDecRotO = "dec_rot_o";
inline constexpr const char* kLegDecE2E = "dec_e2e";
inline constexpr const char* kLegDecV5 = "dec_v5";
inline constexpr const char* kLegDecFusedQAttnCore = "dec_fq_attn_core";
inline constexpr const char* kLegDecFusedQE2E = "dec_fq_e2e";
// The unpack ablation's counterpart to kLegDecAttnCore; see kUnpackEnv.
inline constexpr const char* kLegDecNoUnpack = "dec_nounpack";
// Option C's counterpart to kLegDecAttnCore: the same decode expanding through a UB Gather.
inline constexpr const char* kLegDecGather = "dec_gather";

const std::vector<const char*>& PrefillLegs();
const std::vector<const char*>& DecodeLegs();

std::string CaseName(const char* leg, const Config& config);

bool LegEnabled(const char* leg);
bool PhaseEnabled(const char* phase);
bool NativeEnabled();

// -----------------------------------------------------------------------------
// Sweep model
// -----------------------------------------------------------------------------

struct Budget {
  int warmup = 0;
  int iterations = 0;

  bool operator==(const Budget& other) const { return warmup == other.warmup && iterations == other.iterations; }
};

Budget BudgetFor(int64_t seq_len);

struct Regime {
  int64_t seq_len;
  std::vector<int64_t> batches;
  const char* label;
};

std::vector<Regime> Regimes();

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

DecodeDispatch PlanDispatch(const Config& config, int64_t aiv_num);

std::vector<Config> BuildSweep();

// -----------------------------------------------------------------------------
// Traffic accounting
// -----------------------------------------------------------------------------

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

Traffic ModelTraffic(const Config& config, const DecodeDispatch& dispatch);

double ScenarioBytes(const Config& config);

// The benchmarked slice (config.model.num_kv_heads kv heads of one layer) scaled to the full model's residency.
double FullModelSavedMib(const Config& config, const Traffic& traffic);

// -----------------------------------------------------------------------------
// Scenario buffer toolkit, shared with the msprof trace
// -----------------------------------------------------------------------------

inline size_t Elements(int64_t a, int64_t b, int64_t c) {
  return static_cast<size_t>(a) * static_cast<size_t>(b) * static_cast<size_t>(c);
}

// Tile a host-side pattern across the whole device buffer.
template <typename T>
void TileToDevice(const DeviceBuffer& dst, const std::vector<T>& pattern) {
  const size_t total = dst.size_bytes();
  const size_t chunk = pattern.size() * sizeof(T);
  if (chunk == 0) {
    return;
  }
  for (size_t offset = 0; offset < total; offset += chunk) {
    const size_t bytes = std::min(chunk, total - offset);
    ACL_CHECK(aclrtMemcpy(static_cast<char*>(dst.get()) + offset, dst.capacity_bytes() - offset, pattern.data(), bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
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

// The leading `limit` elements of a device buffer, for a cheap checksum.
template <typename T>
std::vector<T> LeadingElements(const DeviceBuffer& buffer, size_t limit = kLeadingReadbackElements) {
  std::vector<T> host(std::min(buffer.size_bytes() / sizeof(T), limit));
  if (!host.empty()) {
    buffer.CopyToHost(host.data(), host.size() * sizeof(T));
  }
  return host;
}

// The Pi involution applied row-wise; the copy that rotate_q's fp16 path stages on the host.
std::vector<float> RotatePiRows(std::vector<float> values, int64_t head_size);

void RotatePiRowsInPlace(std::vector<float>& values, int64_t head_size);

}  // namespace turboquant_audit
}  // namespace test
}  // namespace vllm_ascend
