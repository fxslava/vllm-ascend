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

// ============================================================================
// TurboQuant end-to-end audit against the native CANN V5 attention operator
// ============================================================================
//
// One question, asked twice -- once for prefill and once for decode:
//
//     does a TurboQuant layer finish an attention step sooner than the same
//     layer built on the stock CANN operator, AFTER every transformation the
//     4-bit rotated cache imposes has been paid for?
//
// Everything in this binary exists to answer that and nothing else. There is no
// leg here that compares TurboQuant against a hypothetical fp16 memory layout,
// and no leg that compares one TurboQuant kernel against another: those are
// ablations, they live in bench_device_950pr_turboquant_ablation, and a
// benchmark that mixes them with a competitive comparison invites reading an
// internal ratio as a product claim. What is left is a component-by-component
// breakdown of the shipping path and one native baseline to divide it by.
//
// ---------------------------------------------------------------------------
// WHICH NATIVE OPERATOR, AND WHY IT IS NOT THE ONE THE BRIEF NAMED
// ---------------------------------------------------------------------------
//
// The brief for this file asked for `aclnnPromptFlashAttentionV5` in the
// prefill role and `aclnnFusionIncrementalAttention` in the decode role.
// NEITHER SYMBOL EXISTS. The CANN 9.2.0-beta.2 toolkit this tree builds against
// ships `aclnn_prompt_flash_attention{,_v2,_v3}.h` and
// `aclnn_incre_flash_attention{,_v2,_v3,_v4}.h` -- the PFA family stops at V3
// and the IFA family at V4 -- and there is no header, and no exported symbol,
// spelled `aclnnFusionIncrementalAttention` anywhere in the install.
//
// What DOES exist, and what "the native V5 suite" means on this part, is
// `aclnnFusedInferAttentionScore`, whose V5 interface is the unified successor
// that REPLACES both families: V1..V4 of it are withdrawn on an Ascend950
// (planning returns 361001, measured on CANN 9.1.0), and V5 is what a caller
// has left. One operator covers both roles, selected by its argument list:
//
//     prefill role   ("prompt")       sparse_mode 3, the 2048x2048 compressed
//                                     right-down causal mask, contiguous TND
//                                     K/V, block_size 0 -- no paging.
//     decode role    ("incremental")  sparse_mode 0, a block table, the paged
//                                     fp16 cache as a one-entry aclTensorList,
//                                     actualSeqLengthsKv per sequence.
//
// Both roles are planned and timed below, and both are labelled by the operator
// that actually planned. `aclnnFusedInferAttentionScoreV2` is kept as a fallback
// for a toolkit where V5 is absent; the row records which one answered.
//
// Guessing a prototype for the two symbols the brief named was considered and
// rejected. A hand-declared argument list behind `dlsym` that does not match the
// operator is undefined behaviour AT LAUNCH, not a planning refusal -- it takes
// the stream down and every case queued behind it -- and this suite's rule is
// that every aclnn prototype is transcribed from a header it can point at. See
// common/aclnn_ops_950pr.hpp, where the V5 list is quoted line for line.
//
// ---------------------------------------------------------------------------
// THE ACCOUNTING
// ---------------------------------------------------------------------------
//
//     T_TQ_E2E = T_rot_q + T_attn_core + T_rot_o
//
// measured on the device with ACL events, per component AND as one composite
// region over the whole graph. Both are reported: the components say where the
// time goes, the composite says what the caller waits for, and the difference
// between the composite and the sum of the parts is the launch overhead the
// component view cannot see. A row whose composite is far from its sum is
// launch-bound, and the footer prints that residual rather than leaving a
// reader to subtract.
//
//   T_rot_q       npu_turboquant_rotate_q over the step's query.
//   T_attn_core   decode:  the Cube split + the shared combine, or the AIV
//                          launcher that submits both.
//                 prefill: FIA V5 over the rotated basis (Q~, K~, V~).
//   T_rot_o       npu_turboquant_rotate_q over the attention output, before the
//                 sigmoid gate. EXACTLY 0.0 us for a folded layer -- W_o' = W_o
//                 (I (x) Pi) absorbs it offline -- and measured for a layer that
//                 cannot fold. The 0.0 is printed as a number and not as a dash,
//                 because it is a fact about the layer and not a missing sample.
//
// The native column is the V5 operator alone on standard uncompressed tensors,
// and neither side's KV-cache WRITE is inside its E2E figure. TurboQuant's write
// is reported in its own column (Table A, "TQ Ingest / Reshape"); the native
// write (`aclnnScatterPaKvCache`) is timed as `pf_v5_ingest` and carried in the
// prefill CSV. Symmetric, and both are visible.
//
// ---------------------------------------------------------------------------
// WHAT IS STILL NOT PRICED, SAID HERE RATHER THAN HIDDEN
// ---------------------------------------------------------------------------
//
// 1. The fp32 -> fp16 narrowing between the rotation kernel and FIA, in the
//    PREFILL core only. npu_turboquant_rotate_q writes fp32; FIA reads fp16.
//    Production does the cast with a torch op; this suite has no header-verified
//    aclnnCast prototype and will not guess one, so the rotated tensors FIA is
//    handed are prepared on the host in fp16 and the cast is not in any number
//    below. It is one elementwise pass over B*C*H_Q*D (query) and B*S*H_KV*D
//    (value) elements; the modelled byte cost of both is in the prefill CSV as
//    `untimed_cast_bytes` so a reader can bound it. The DECODE core has no such
//    gap: the split kernels consume the fp32 rotation directly.
// 2. The sigmoid gate itself, for Qwen3.5. T_rot_o stops where the brief says it
//    stops -- "prior to the sigmoid gate" -- and the gate is the same elementwise
//    cost on both sides of the comparison.
// 3. Nothing about accuracy. TurboQuant stores 4 bits per coordinate and the
//    native leg stores 16; the outputs are not the same numbers and are not
//    meant to be. The fidelity gates are test_device_950pr_turboquant,
//    test_host_turboquant_fidelity and the sim tier. What this file does check
//    is that both legs are attending over the same context at all -- see THE TWO
//    DEVICE CHECKS below.
//
// ---------------------------------------------------------------------------
// THE WORKLOAD
// ---------------------------------------------------------------------------
//
// Three flagship families at TP=8, one NPU's share of each. These are serving
// ranks, not a geometric sweep: a dense cartesian product of head counts
// produces hundreds of shapes nobody deploys and buries the nine that matter.
//
//   Qwen3.5-9B          D=128  H_Q=4   H_KV=1   GQA 4:1    UNFOLDED
//                       attn_output_gate multiplies the context by sigmoid(gate)
//                       before o_proj, so W_o cannot absorb Pi and T_rot_o is a
//                       real kernel on the critical path.
//   DeepSeek-V4-Flash   D=256  H_Q=16  H_KV=1   MLA        folded
//                       decoupled latent KV. The 16:1 group exactly fills the
//                       Cube's M=16 fractal, which is what puts this family on
//                       the Cube decode.
//   GLM-5.2-744B        D=128  H_Q=8   H_KV=1   GQA 8:1    folded
//
//   S = 2048      B in {1, 4, 8}      short / interactive
//   S = 32768     B in {1, 4, 8}      enterprise long
//   S = 262144    B in {1, 2}         ultra-long
//   S = 1048576   B in {1}            extreme needle / 1M
//
// 3 families x 9 (S, B) pairs = 27 configurations, each audited in both phases:
// 27 rows in Table A and 27 in Table B.
//
// PREFILL IS CHUNKED, and the table says so. A step processes C = min(S, 2048)
// query tokens against the S-token prefix, which is what vLLM's scheduler
// actually submits and what keeps a 1M-context row from costing 2.5 hours of
// O(S^2) attention per iteration. The S column prints `S/C` whenever C < S. At
// S = 2048 the chunk is the whole prefill and the column prints plain `2048`.
//
// ITERATIONS ARE ADAPTIVE. S <= 32K gets 5 warmup and 20 timed; S >= 262K gets
// 1 and 3. Two budgets means two BenchmarkRunners -- the generic table stamps
// one warmup/iterations line on the whole of itself, so a second budget gets a
// second table rather than rows its header misdescribes.
//
// ---------------------------------------------------------------------------
// THE DECODE PATH IS CHOSEN BY THE GQA GROUP, NOT BY A FLAG
// ---------------------------------------------------------------------------
//
// The Cube decode batches a kv head's query heads into the GEMM's M dimension,
// and that fractal is 16 rows wide (turboquant_host::kCubeTileM). A 16:1 group
// fills it; a 4:1 or 8:1 group would run it three-quarters or half empty, and
// the vector path is the right one there. So:
//
//     group = H_Q / H_KV >= 16   ->  Cube   (TurboQuantCubeDecodeSplit +
//                                            TurboQuantPagedAttentionCombine,
//                                            two launches, separately timed)
//     group < 16                 ->  AIV    (turboquant_paged_attention, one
//                                            launcher submitting both stages)
//
// which puts DeepSeek-V4-Flash on the Cube and Qwen3.5 / GLM-5.2 on the AIV
// path. `ASCEND_BENCH_TQ_AUDIT_PATH=cube|aiv` overrides it for both.
//
// THE AIV PATH'S SPLIT COLUMN IS DERIVED, AND IS MARKED WITH `~`. There is no
// exported entry point for the AIV split alone -- turboquant_paged_attention_impl
// submits split and combine together -- so T_DecodeSplit for that path is the
// measured core minus the separately measured combine. The Cube path's split is
// measured directly and carries no marker. The CSV carries the flag as
// `split_is_derived`.
//
// ---------------------------------------------------------------------------
// THE TWO DEVICE CHECKS, BEFORE ANYTHING IS TIMED
// ---------------------------------------------------------------------------
//
// 1. THE ROTATED-BASIS IDENTITY. Pi is orthogonal, so attention in the rotated
//    basis is attention: softmax(Q~ K~^T) V~ = Pi (softmax(Q K^T) V). The check
//    runs FIA over (Q~, K~, V~), un-rotates the result on the host, and demands
//    cos > 0.999 against FIA over (Q, K, V). That is the device-level statement
//    that the prefill core this file times is computing the layer's attention
//    and not some other bilinear form, and it is what licenses timing the folded
//    path with T_rot_o = 0.
// 2. THE DECODE TIE-POINT. On the smallest configuration whose context fits an
//    exact host image, the TurboQuant decode's output is un-rotated and compared
//    against the native V5 decode over the same context in the same slots:
//    cos > 0.90. That bound is deliberately loose -- 4 bits against 16 will not
//    do better, and the fidelity gates are elsewhere -- but it is tight enough to
//    catch the failure this binary is otherwise blind to, which is timing a
//    kernel that is reading an empty cache.
//
// Both are recorded as failures when they miss, so ctest sees them.
//
// ---------------------------------------------------------------------------
// TRAFFIC, AND WHAT THE GB/s COLUMNS MEAN
// ---------------------------------------------------------------------------
//
// Every byte figure is COMPULSORY traffic: each distinct byte the step must move
// across HBM, counted once, under one convention for every leg. A cached KV row
// is counted once per kv head and not once per query head, whichever path reads
// it, so the ratio between two legs' byte counts is the ratio between their
// storage formats and not between their task decompositions. Re-reads that L2
// may or may not absorb are not guessed at in either direction.
//
// ---------------------------------------------------------------------------
// OUTPUT
// ---------------------------------------------------------------------------
//
//   Table A   prefill pipeline, one row per configuration
//   Table B   decode latency breakdown, one row per configuration
//   plus the generic per-case table and CSV from the shared harness.
//
//   ASCEND_BENCH_TQ_AUDIT_PREFILL_CSV=<path>   Table A, plus every column the
//                                              terminal has no room for.
//   ASCEND_BENCH_TQ_AUDIT_DECODE_CSV=<path>    Table B, likewise.
//   ASCEND_BENCH_TQ_AUDIT_MODELS=qwen35,dsv4,glm52
//   ASCEND_BENCH_TQ_AUDIT_S=2048,32768         restrict the context regimes
//   ASCEND_BENCH_TQ_AUDIT_B=1,4                restrict the batches
//   ASCEND_BENCH_TQ_AUDIT_PHASES=prefill,decode
//   ASCEND_BENCH_TQ_AUDIT_LEGS=dec_e2e,dec_v5  restrict the legs
//   ASCEND_BENCH_TQ_AUDIT_CHUNK=2048           the prefill chunk C
//   ASCEND_BENCH_TQ_AUDIT_PATH=auto|cube|aiv
//   ASCEND_BENCH_TQ_AUDIT_GLM_D=128            GLM-5.2 is specified at 128 or 256
//   ASCEND_BENCH_TQ_AUDIT_WARMUP / _ITERS      the S <= 32K budget (5 / 20)
//   ASCEND_BENCH_TQ_AUDIT_ULTRA_WARMUP / _ULTRA_ITERS   the S >= 262K one (1 / 3)
//   ASCEND_BENCH_TQ_FIA=0                      drop the native V5 legs entirely
//
// and the shared ASCEND_BENCH_* set; see common/benchmark.hpp. Note that
// ASCEND_BENCH_MODES selects three timing modes by default and every one of them
// is a full warmup-plus-iterations run: on the ultra-long rows that is the
// difference between minutes and tens of minutes.
//
// NOTHING BELOW HAS EVER RUN ON SILICON. No Ascend 950PR part has been available
// to this project. The kernels are verified on the arch35 camodel and the
// numbers this file would print are, as of this writing, hypothetical.

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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Every paged cache here is block_size 128, as the plugin's own cache manager
// names it.
constexpr int64_t kBlockSize = s950::kDefaultBlockSize;

// FIA's compressed causal mask for sparse_mode 3 is always 2048 x 2048 whatever
// the sequence length; AttentionMaskBuilder.get_splitfuse_attn_mask builds the
// same int8 triu(ones, diagonal=1).
constexpr int64_t kCausalMaskSide = 2048;
constexpr int64_t kFiaSparseModeRightDownCausal = 3;
// FIA reads no paging in the prompt role, so its block-size argument is 0.
constexpr int64_t kFiaNoPaging = 0;

// Tokens of distinct random data a scenario draws; longer contexts tile it. The
// pattern is a whole number of tokens, so a tile boundary never falls inside a
// D-wide vector and the host-rotated copies stay exactly Pi applied to the
// unrotated ones, tile for tile. Attention latency does not depend on the
// values, and generating 5e8 fp16 elements on the host for a 1M context would
// cost minutes per configuration and risk a host OOM.
constexpr int64_t kPatternTokens = 1024;

// Query tokens one chunked-prefill step submits. vLLM's scheduler default
// scale; see THE WORKLOAD above for why prefill is chunked at all.
constexpr int64_t kPrefillChunkTokens = 2048;

// Share of free HBM a configuration may plan to occupy before it is skipped.
constexpr double kHbmBudget = 0.85;

// Elements a checksum reads back. Enough to catch a launch that has stopped
// writing without copying gigabytes to the host twice per case.
constexpr size_t kChecksumElements = 1u << 20;

// Context elements (B * S * H_KV * D) below which the fp16 paged cache is built
// slot-exact on the host, which is what the decode tie-point needs. Above it the
// cache is filled by tiling, because the host image would be gigabytes.
constexpr size_t kExactContextElements = 1u << 22;

constexpr double kRotatedBasisMinCosine = 0.999;
constexpr double kDecodeTiePointMinCosine = 0.90;

constexpr double kHalfBytes = 2.0;
constexpr double kFloatBytes = 4.0;
constexpr double kMiB = 1024.0 * 1024.0;

// The two iteration budgets, per the context regime.
constexpr int kShortWarmup = 5;
constexpr int kShortIterations = 20;
constexpr int kUltraWarmup = 1;
constexpr int kUltraIterations = 3;
// One launch per event pair. These shapes are orders of magnitude heavier than
// the microbenchmarks the shared default of 10 was chosen for.
constexpr int kPipelineBatch = 1;

// Contexts at or above this take the ultra budget.
constexpr int64_t kUltraContextThreshold = 262144;
// Below this the block pool is over-allocated so the block table is a genuine
// scatter rather than a run of consecutive blocks; above it nothing is resident
// anyway and a 4x pool is gigabytes of idle HBM.
constexpr int64_t kOversizedPoolContextLimit = 8192;
constexpr int64_t kPoolOversizeFactor = 4;

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

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

int64_t EnvInt64(const char* name, int64_t fallback, int64_t minimum) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return fallback;
  }
  const long long parsed = std::strtoll(raw, nullptr, 10);
  return parsed < minimum ? minimum : static_cast<int64_t>(parsed);
}

// True unless the variable is set to exactly "0". An unset variable means the
// default, which for every switch here is "on".
bool EnvOn(const char* name) {
  const char* raw = std::getenv(name);
  return raw == nullptr || *raw == '\0' || std::strcmp(raw, "0") != 0;
}

// True when `value` is in the comma list `name` names, or when the variable is
// unset -- an unset filter selects everything.
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

// ---------------------------------------------------------------------------
// The workload
// ---------------------------------------------------------------------------

struct ModelSpec {
  const char* key;     // selector for ASCEND_BENCH_TQ_AUDIT_MODELS and case names
  const char* label;   // the table column
  int64_t head_size;
  int64_t num_heads;     // H_Q on one NPU at TP=8
  int64_t num_kv_heads;  // H_KV on one NPU at TP=8
  // False when the output projection cannot absorb Pi, so the de-rotation is a
  // kernel on the critical path rather than an offline weight transform.
  bool folds_output;
  const char* note;
};

// GLM-5.2 is specified at D = 128 or 256; 128 is the default and the variable
// switches it, because the two differ in how much of the Cube's C0 a vector
// fills and a reader may want both.
int64_t GlmHeadSize() {
  const int64_t d = EnvInt64("ASCEND_BENCH_TQ_AUDIT_GLM_D", 128, 64);
  if (d != 128 && d != 256) {
    std::printf("[ascend-bench] ASCEND_BENCH_TQ_AUDIT_GLM_D=%lld is neither 128 nor 256; using 128\n",
                static_cast<long long>(d));
    return 128;
  }
  return d;
}

std::vector<ModelSpec> Models() {
  return {
      ModelSpec{"qwen35", "Qwen3.5-9B", 128, 4, 1, /*folds_output=*/false,
                "attn_output_gate: sigmoid(gate) * context sits between attention and o_proj, "
                "so W_o cannot absorb Pi and the O de-rotation is measured"},
      ModelSpec{"dsv4", "DeepSeek-V4-Flash", 256, 16, 1, /*folds_output=*/true,
                "MLA, decoupled latent KV; the 16:1 group exactly fills the Cube's M=16 fractal; "
                "W_o folded offline"},
      ModelSpec{"glm52", "GLM-5.2-744B", GlmHeadSize(), 8, 1, /*folds_output=*/true,
                "ultra-wide GQA; W_o folded offline"},
  };
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

// How TurboQuant's decode splits the work at this shape.
enum class PathMode { kCube, kAiv };

const char* PathLabel(PathMode path) { return path == PathMode::kCube ? "Cube" : "AIV"; }

// The GQA group the Cube's M dimension has to be filled from; see THE DECODE
// PATH IS CHOSEN BY THE GQA GROUP above.
PathMode SelectPath(const ModelSpec& model) {
  const std::string forced = EnvString("ASCEND_BENCH_TQ_AUDIT_PATH");
  if (forced == "cube") {
    return PathMode::kCube;
  }
  if (forced == "aiv") {
    return PathMode::kAiv;
  }
  const int64_t group = model.num_heads / model.num_kv_heads;
  return group >= tqh::kCubeTileM ? PathMode::kCube : PathMode::kAiv;
}

int64_t PrefillChunk(int64_t seq_len) {
  const int64_t chunk = EnvInt64("ASCEND_BENCH_TQ_AUDIT_CHUNK", kPrefillChunkTokens, 1);
  return std::min(seq_len, chunk);
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

  // Query tokens one prefill step submits, across the batch.
  int64_t chunk_tokens() const { return batch * chunk; }
  // Tokens of context the whole batch holds.
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
  // "2048" when the chunk is the whole prefill, "262144/2048" when it is not.
  std::string context_label() const {
    std::ostringstream text;
    text << seq_len;
    if (chunk < seq_len) {
      text << '/' << chunk;
    }
    return text.str();
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

// ---------------------------------------------------------------------------
// Legs
// ---------------------------------------------------------------------------

constexpr const char* kLegPfIngest = "pf_ingest";
constexpr const char* kLegPfRotQ = "pf_rot_q";
constexpr const char* kLegPfAttnCore = "pf_attn_core";
constexpr const char* kLegPfRotO = "pf_rot_o";
constexpr const char* kLegPfE2E = "pf_e2e";
constexpr const char* kLegPfV5 = "pf_v5";
constexpr const char* kLegPfV5Ingest = "pf_v5_ingest";

constexpr const char* kLegDecRotQ = "dec_rot_q";
constexpr const char* kLegDecSplit = "dec_split";
constexpr const char* kLegDecCombine = "dec_combine";
constexpr const char* kLegDecAttnCore = "dec_attn_core";
constexpr const char* kLegDecRotO = "dec_rot_o";
constexpr const char* kLegDecE2E = "dec_e2e";
constexpr const char* kLegDecV5 = "dec_v5";

const char* const kPrefillLegs[] = {kLegPfIngest, kLegPfRotQ, kLegPfAttnCore, kLegPfRotO,
                                    kLegPfE2E,    kLegPfV5,   kLegPfV5Ingest};
const char* const kDecodeLegs[] = {kLegDecRotQ, kLegDecSplit,  kLegDecCombine, kLegDecAttnCore,
                                   kLegDecRotO, kLegDecE2E,    kLegDecV5};

std::string CaseName(const char* leg, const Config& config) {
  return std::string(leg) + "_" + config.id();
}

bool LegEnabled(const char* leg) { return EnvSelects("ASCEND_BENCH_TQ_AUDIT_LEGS", leg); }
bool PhaseEnabled(const char* phase) { return EnvSelects("ASCEND_BENCH_TQ_AUDIT_PHASES", phase); }
bool NativeEnabled() { return EnvOn("ASCEND_BENCH_TQ_FIA"); }

// ---------------------------------------------------------------------------
// Traffic
// ---------------------------------------------------------------------------
//
// Compulsory bytes: each distinct byte the step moves across HBM, once, under
// one convention for every leg. See TRAFFIC above.
struct Traffic {
  // Residency, for the whole batch's context.
  double fp16_kv_bytes = 0.0;
  double tq_kv_bytes = 0.0;

  // Prefill, per step.
  double pf_ingest = 0.0;
  double pf_rot_q = 0.0;
  double pf_attn_core = 0.0;
  double pf_rot_o = 0.0;
  double pf_e2e = 0.0;
  double pf_v5 = 0.0;
  double pf_v5_ingest = 0.0;
  double pf_flops = 0.0;
  // The fp32 -> fp16 narrowing this suite does not time; see WHAT IS STILL NOT
  // PRICED. Reported so a reader can bound it, never added to a leg.
  double pf_untimed_cast = 0.0;

  // Decode, per step.
  double dec_rot_q = 0.0;
  double dec_split = 0.0;
  double dec_combine = 0.0;
  double dec_attn_core = 0.0;
  double dec_rot_o = 0.0;
  double dec_e2e = 0.0;
  double dec_v5 = 0.0;
  double dec_flops = 0.0;
  // Flash-decoding splits the grid planned for this shape, which is what the
  // partial term above was computed from. Carried so the CSV can state it
  // rather than leaving a reader to re-derive the planner's arithmetic.
  int64_t dec_split_count = 1;

  double compression_ratio() const { return tq_kv_bytes > 0.0 ? fp16_kv_bytes / tq_kv_bytes : 0.0; }
  double saved_mib() const { return (fp16_kv_bytes - tq_kv_bytes) / kMiB; }
};

Traffic ModelTraffic(const Config& config, int64_t num_splits) {
  const double d = static_cast<double>(config.model.head_size);
  const double hq = static_cast<double>(config.model.num_heads);
  const double hkv = static_cast<double>(config.model.num_kv_heads);
  const double packed_slot = d / static_cast<double>(tqh::kPackFactor);
  const double scale_slot = static_cast<double>(tqh::ScaleSlotFloats(config.model.num_kv_heads)) * kFloatBytes;
  const double context = static_cast<double>(config.context_tokens());
  const double chunk = static_cast<double>(config.chunk_tokens());
  const double batch = static_cast<double>(config.batch);
  const double splits = static_cast<double>(num_splits);

  Traffic traffic;
  traffic.fp16_kv_bytes = 2.0 * context * hkv * d * kHalfBytes;
  traffic.tq_kv_bytes = 2.0 * context * hkv * packed_slot + context * scale_slot;

  // --- prefill -------------------------------------------------------------
  const double chunk_kv_fp16 = 2.0 * chunk * hkv * d * kHalfBytes;
  const double chunk_kv_tq = 2.0 * chunk * hkv * packed_slot + chunk * scale_slot;
  const double chunk_query = chunk * hq * d * kHalfBytes;
  const double mask = static_cast<double>(kCausalMaskSide * kCausalMaskSide);

  traffic.pf_ingest = chunk_kv_fp16 + chunk_kv_tq + chunk * kFloatBytes;
  traffic.pf_rot_q = chunk * hq * d * (kHalfBytes + kFloatBytes);
  // Q in, the whole prefix's K and V, O out, and the compressed mask.
  traffic.pf_attn_core = chunk_query + traffic.fp16_kv_bytes + chunk_query + mask;
  traffic.pf_rot_o = chunk * hq * d * (kHalfBytes + kFloatBytes);
  traffic.pf_e2e = traffic.pf_rot_q + traffic.pf_attn_core +
                   (config.model.folds_output ? 0.0 : traffic.pf_rot_o);
  traffic.pf_v5 = traffic.pf_attn_core;
  // The native write: read the chunk's fp16 K/V, write them into the paged
  // cache, read the slot map.
  traffic.pf_v5_ingest = 2.0 * chunk_kv_fp16 + chunk * kFloatBytes;
  // Causal over a chunk at the tail of an S-token prefix: query i of the chunk
  // attends to S - C + i + 1 keys, so the count is C * (S - C/2 + 1/2) per
  // sequence per head. Two operations per element for QK^T and two for the value
  // accumulation.
  {
    const double s = static_cast<double>(config.seq_len);
    const double c = static_cast<double>(config.chunk);
    traffic.pf_flops = 4.0 * batch * hq * d * c * (s - 0.5 * c + 0.5);
  }
  // One elementwise narrowing of the rotated query and the rotated value.
  traffic.pf_untimed_cast = chunk * hq * d * (kFloatBytes + kHalfBytes) +
                            context * hkv * d * (kFloatBytes + kHalfBytes);

  // --- decode --------------------------------------------------------------
  const double step_query_fp16 = batch * hq * d * kHalfBytes;
  const double step_query_fp32 = batch * hq * d * kFloatBytes;
  const double step_out_fp16 = step_query_fp16;
  // Flash-decoding partials: the split writes one per (token, head, split) and
  // the combine reads it back, so each crosses HBM twice.
  const double partials = batch * hq * splits * static_cast<double>(config.model.head_size + tqh::kPartialTail) *
                          kFloatBytes;

  traffic.dec_rot_q = step_query_fp16 + step_query_fp32;
  traffic.dec_split = step_query_fp32 + traffic.tq_kv_bytes + partials;
  traffic.dec_combine = partials + step_out_fp16;
  traffic.dec_attn_core = traffic.dec_split + traffic.dec_combine;
  traffic.dec_rot_o = step_out_fp16 + batch * hq * d * kFloatBytes;
  traffic.dec_e2e = traffic.dec_rot_q + traffic.dec_attn_core +
                    (config.model.folds_output ? 0.0 : traffic.dec_rot_o);
  traffic.dec_v5 = step_query_fp16 + traffic.fp16_kv_bytes + step_out_fp16;
  traffic.dec_flops = 4.0 * batch * hq * d * static_cast<double>(config.seq_len);
  traffic.dec_split_count = num_splits;

  return traffic;
}

// HBM a configuration allocates, bar the operators' own workspaces, which are
// allowed one more attention output each.
double ScenarioBytes(const Config& config) {
  const double d = static_cast<double>(config.model.head_size);
  const double hq = static_cast<double>(config.model.num_heads);
  const double hkv = static_cast<double>(config.model.num_kv_heads);
  const double context = static_cast<double>(config.context_tokens());
  const double chunk = static_cast<double>(config.chunk_tokens());
  const double pool_rows = static_cast<double>(config.pool_blocks() * kBlockSize);
  const double scale_slot = static_cast<double>(tqh::ScaleSlotFloats(config.model.num_kv_heads)) * kFloatBytes;

  // K, V, K~ and V~ over the whole prefix.
  const double contiguous_kv = 4.0 * context * hkv * d * kHalfBytes;
  // The chunk's own K and V, which the ingest leg writes from.
  const double chunk_kv = 2.0 * chunk * hkv * d * kHalfBytes;
  const double tq_cache = 2.0 * pool_rows * hkv * (d / static_cast<double>(tqh::kPackFactor)) +
                          pool_rows * scale_slot;
  const double fp16_cache = 2.0 * pool_rows * hkv * d * kHalfBytes;
  // Q, Q~ (fp16), Q~ (fp32), O_tq, O_v5, O~ (fp32) for the chunk; the same six
  // for the decode step, which is negligible beside them.
  const double prefill_activations = chunk * hq * d * (4.0 * kHalfBytes + 2.0 * kFloatBytes);
  const double decode_activations =
      static_cast<double>(config.batch) * hq * d * (4.0 * kHalfBytes + 2.0 * kFloatBytes);
  const double mask = static_cast<double>(kCausalMaskSide * kCausalMaskSide);
  // Two attention outputs' worth of slack for the operators' workspaces.
  const double workspace_allowance = 2.0 * chunk * hq * d * kHalfBytes;

  return contiguous_kv + chunk_kv + tq_cache + fp16_cache + prefill_activations + decode_activations + mask +
         workspace_allowance;
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Fills `dst` by repeating `pattern` end to end, one whole-pattern copy at a
// time. Chunked by construction: a 1M-token context is filled from a
// 1024-token host image and never needs a host buffer of its own size.
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

// The host image of what TileToDevice would have written: `pattern` repeated to
// `elements`. Only ever called for a context small enough to hold exactly; see
// kExactContextElements.
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

// Checksum source: at most kChecksumElements leading elements of a buffer.
template <typename T>
std::vector<T> LeadingElements(const DeviceBuffer& buffer) {
  std::vector<T> host(std::min(buffer.size_bytes() / sizeof(T), kChecksumElements));
  if (!host.empty()) {
    buffer.CopyToHost(host.data(), host.size() * sizeof(T));
  }
  return host;
}

// Pi applied to every head_size-wide row, on the host, in place.
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

// ---------------------------------------------------------------------------
// One configuration on the device
// ---------------------------------------------------------------------------
//
// Built, audited and destroyed before the next one is built, so the sweep's peak
// HBM is its largest configuration's and not the sum of all of them.
//
// The aclnn descriptors are unique_ptrs created in dependency order and the
// planned operators are declared last, so they are destroyed first: a PlannedOp
// holds the descriptor handles it was planned with.
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

    // --- the context, tiled --------------------------------------------------
    //
    // K and V are one pattern; K~ and V~ are Pi applied to that same pattern and
    // then tiled the same way, so the four buffers stand in the exact Pi
    // relationship the rotated-basis identity asserts, at any context length.
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

    // --- the prefill chunk ---------------------------------------------------
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

    // --- the decode step -----------------------------------------------------
    const std::vector<float> decode_query =
        rng.NormalHalfExact(elems(config.batch, hq, d), 0.0f, 1.0f);
    query_dec_ = DeviceBuffer::FromHost(FloatToHalf(decode_query), kBenchmarkAlignBytes);
    query_dec_rot_ = DeviceBuffer::Empty<float>(elems(config.batch, hq, d), kBenchmarkAlignBytes);
    out_dec_tq_ = DeviceBuffer::Empty<Half>(elems(config.batch, hq, d), kBenchmarkAlignBytes);
    out_dec_v5_ = DeviceBuffer::Empty<Half>(elems(config.batch, hq, d), kBenchmarkAlignBytes);
    out_dec_rot_ = DeviceBuffer::Empty<float>(elems(config.batch, hq, d), kBenchmarkAlignBytes);
    lse_dec_ = DeviceBuffer::Empty<Half>(1, kBenchmarkAlignBytes);

    // --- paging --------------------------------------------------------------
    //
    // Each sequence gets its own run of blocks out of a shuffled pool, so the
    // cache scatters the way a serving pool does rather than reading as one
    // contiguous sweep.
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

    // Slot maps: every context position, for the setup fill; and the chunk's own
    // positions, which sit at the tail of each sequence's prefix the way a
    // chunked prefill's do.
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

    // --- TurboQuant constants and cache -------------------------------------
    pi_signs_ = DeviceBuffer::FromHost(tqh::PiSigns(d), kBenchmarkAlignBytes);
    h16_ = DeviceBuffer::FromHost(tqh::Hadamard16Half(), kBenchmarkAlignBytes);
    rot_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(d, 1), kBenchmarkAlignBytes);

    const int64_t pool_blocks = config.pool_blocks();
    if (config.path == PathMode::kCube) {
      key_cache_ = DeviceBuffer::Empty<int8_t>(
          tqh::ModePackedCacheBytes(kCubeMode, pool_blocks, kBlockSize, hkv, d), kBenchmarkAlignBytes);
      write_tables_ =
          DeviceBuffer::FromHost(tqh::ModeTables(kCubeMode, d, 1, /*nz_rows=*/0), kBenchmarkAlignBytes);
      decode_tables_ = DeviceBuffer::FromHost(
          tqh::ModeTables(kCubeMode, d, tqh::kUnpackRows, tqh::kCubeTileRows), kBenchmarkAlignBytes);
      cube_grid_ = tqh::PlanCubeDecode(config.batch, hq, hkv, d, blocks_per_seq, aiv_num);
      num_splits_ = cube_grid_.num_splits;
      workspace_floats_ = cube_grid_.workspace_floats;
    } else {
      key_cache_ =
          DeviceBuffer::Empty<int8_t>(tqh::PackedCacheBytes(pool_blocks, kBlockSize, hkv, d), kBenchmarkAlignBytes);
      write_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(d, 1), kBenchmarkAlignBytes);
      decode_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(d, tqh::kTileRows), kBenchmarkAlignBytes);
      aiv_grid_ = tqh::PlanPagedAttention(config.batch, hq, d, blocks_per_seq, aiv_num);
      num_splits_ = aiv_grid_.num_splits;
      workspace_floats_ = aiv_grid_.workspace_floats;
    }
    value_cache_ = DeviceBuffer::Empty<int8_t>(key_cache_.size_bytes(), kBenchmarkAlignBytes);
    scale_plane_ = DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(pool_blocks, kBlockSize, hkv),
                                              kBenchmarkAlignBytes);
    workspace_ = DeviceBuffer::Empty<float>(workspace_floats_, kBenchmarkAlignBytes);

    context_write_grid_ = tqh::PlanReshapeAndCache(context, aiv_num);
    chunk_write_grid_ = tqh::PlanReshapeAndCache(chunk, aiv_num);

    // --- the native operator's fp16 paged cache ------------------------------
    //
    // Slot-exact when the context fits a host image, because the decode
    // tie-point compares the two legs over the same context in the same slots.
    // Tiled otherwise: above kExactContextElements the host image would be
    // gigabytes, and attention latency does not depend on the values.
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

    // --- the causal mask -----------------------------------------------------
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

  // --- prefill ------------------------------------------------------------

  void EnqueuePrefillIngest(aclrtStream stream) const {
    EnqueueCacheWrite(stream, key_chunk_.get(), value_chunk_.get(), slots_chunk_.get(), config_.chunk_tokens(),
                      chunk_write_grid_);
  }

  void EnqueuePrefillRotateQ(aclrtStream stream) const {
    tqh::RotateQuery(stream, AscendType::FP16, query_pf_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                     query_pf_rot_fp32_.get(), config_.chunk_tokens(), config_.model.num_heads,
                     config_.model.head_size, aiv_num_, /*input_exact_in_half=*/true);
  }

  void EnqueuePrefillAttnCore(aclrtStream stream) const { fia_prefill_rotated_->Launch(stream); }

  void EnqueuePrefillRotateO(aclrtStream stream) const {
    tqh::RotateQuery(stream, AscendType::FP16, out_pf_tq_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                     out_pf_rot_.get(), config_.chunk_tokens(), config_.model.num_heads, config_.model.head_size,
                     aiv_num_, /*input_exact_in_half=*/true);
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

  // --- decode -------------------------------------------------------------

  void EnqueueDecodeRotateQ(aclrtStream stream) const {
    tqh::RotateQuery(stream, AscendType::FP16, query_dec_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                     query_dec_rot_.get(), config_.batch, config_.model.num_heads, config_.model.head_size,
                     aiv_num_, /*input_exact_in_half=*/true);
  }

  // The Cube split alone. Only meaningful on the Cube path; the AIV launcher
  // does not expose its split stage separately, which is why that path's split
  // column is derived. See THE DECODE PATH above.
  void EnqueueDecodeSplit(aclrtStream stream) const {
    turboquant_mm_decode_split_impl(
        static_cast<int32_t>(kCubeMode), AscendType::FP16, stream, cube_grid_.split_block_dim,
        query_dec_rot_.get(), key_cache_.get(), value_cache_.get(), scale_plane_.get(), block_tables_.get(),
        context_lens_.get(), decode_tables_.get(), workspace_.get(), static_cast<uint32_t>(config_.batch),
        static_cast<uint32_t>(config_.model.num_heads), static_cast<uint32_t>(config_.model.num_kv_heads),
        static_cast<uint32_t>(config_.model.head_size), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(config_.blocks_per_seq()), static_cast<uint32_t>(cube_grid_.num_splits),
        cube_grid_.split_tasks_per_core, config_.attention_scale(), config_.attention_scale());
  }

  // The shared reduction, on both paths. Reads the workspace and nothing else.
  void EnqueueDecodeCombine(aclrtStream stream) const {
    const uint32_t block_dim =
        config_.path == PathMode::kCube ? cube_grid_.combine_block_dim : aiv_grid_.combine_block_dim;
    const uint32_t tasks =
        config_.path == PathMode::kCube ? cube_grid_.combine_tasks_per_core : aiv_grid_.combine_tasks_per_core;
    turboquant_paged_attention_combine_impl(
        AscendType::FP16, stream, block_dim, workspace_.get(), out_dec_tq_.get(),
        static_cast<uint32_t>(config_.batch), static_cast<uint32_t>(config_.model.num_heads),
        static_cast<uint32_t>(config_.model.head_size), static_cast<uint32_t>(num_splits_), tasks);
  }

  void EnqueueDecodeAttnCore(aclrtStream stream) const {
    if (config_.path == PathMode::kCube) {
      EnqueueDecodeSplit(stream);
      EnqueueDecodeCombine(stream);
      return;
    }
    turboquant_paged_attention_impl(
        AscendType::FP16, stream, aiv_grid_.split_block_dim, aiv_grid_.combine_block_dim, query_dec_rot_.get(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(),
        decode_tables_.get(), workspace_.get(), out_dec_tq_.get(), static_cast<uint32_t>(config_.batch),
        static_cast<uint32_t>(config_.model.num_heads), static_cast<uint32_t>(config_.model.num_kv_heads),
        static_cast<uint32_t>(config_.model.head_size), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(config_.blocks_per_seq()), static_cast<uint32_t>(aiv_grid_.num_splits),
        aiv_grid_.split_tasks_per_core, aiv_grid_.combine_tasks_per_core, config_.attention_scale(),
        config_.attention_scale());
  }

  void EnqueueDecodeRotateO(aclrtStream stream) const {
    tqh::RotateQuery(stream, AscendType::FP16, out_dec_tq_.get(), pi_signs_.get(), h16_.get(), rot_tables_.get(),
                     out_dec_rot_.get(), config_.batch, config_.model.num_heads, config_.model.head_size, aiv_num_,
                     /*input_exact_in_half=*/true);
  }

  void EnqueueDecodeE2E(aclrtStream stream) const {
    EnqueueDecodeRotateQ(stream);
    EnqueueDecodeAttnCore(stream);
    if (!config_.model.folds_output) {
      EnqueueDecodeRotateO(stream);
    }
  }

  void EnqueueDecodeNative(aclrtStream stream) const { fia_decode_native_->Launch(stream); }

  // --- setup, outside every timed region ----------------------------------

  // Quantises the whole batch's context into the TurboQuant cache, so the decode
  // reads a populated cache rather than the allocation's zeros.
  void FillCache(aclrtStream stream) const {
    EnqueueCacheWrite(stream, key_ctx_.get(), value_ctx_.get(), slots_full_.get(), config_.context_tokens(),
                      context_write_grid_);
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  // Leaves the flash-decoding workspace and both attention outputs holding real
  // values. Without it `dec_combine` timed on its own would reduce a workspace
  // of zeros -- a zero softmax denominator, so a non-finite checksum and a
  // spurious failure -- and `dec_rot_o` / `pf_rot_o` would rotate zeros.
  void Prime(aclrtStream stream) const {
    EnqueueDecodeRotateQ(stream);
    EnqueueDecodeAttnCore(stream);
    if (prefill_available()) {
      EnqueuePrefillAttnCore(stream);
      EnqueuePrefillNative(stream);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  // --- availability -------------------------------------------------------

  bool prefill_available() const { return fia_prefill_rotated_ != nullptr && fia_prefill_native_ != nullptr; }
  bool decode_available() const { return fia_decode_native_ != nullptr; }
  bool native_write_available() const { return native_write_ != nullptr; }
  bool exact_context() const { return exact_context_; }
  const std::string& fia_operator() const { return fia_operator_; }
  const std::string& fia_note() const { return fia_note_; }
  const std::string& native_write_note() const { return native_write_note_; }
  int64_t num_splits() const { return num_splits_; }
  uint32_t split_block_dim() const {
    return config_.path == PathMode::kCube ? cube_grid_.split_block_dim : aiv_grid_.split_block_dim;
  }
  uint32_t combine_block_dim() const {
    return config_.path == PathMode::kCube ? cube_grid_.combine_block_dim : aiv_grid_.combine_block_dim;
  }

  // --- checksums and readback ---------------------------------------------

  double ScaleChecksum() const { return ChecksumSum(LeadingElements<float>(scale_plane_)); }
  double WorkspaceChecksum() const { return ChecksumSum(LeadingElements<float>(workspace_)); }
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

  // The prefill E2E's product: the rotated output for an unfolded layer, the
  // rotated-basis attention output for a folded one.
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
  // The Cube-native rate this audit stores at. kv3fp4 and kv5fp8 are built and
  // covered by the sim tier; neither is the shipping 4-bit path.
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
                                           static_cast<uint32_t>(kBlockSize), grid.tokens_per_core,
                                           config_.attention_scale());
      return;
    }
    turboquant_reshape_and_cache_impl(AscendType::FP16, stream, grid.block_dim, key, value, key_cache_.get(),
                                      value_cache_.get(), scale_plane_.get(), slots, pi_signs_.get(),
                                      write_tables_.get(), token_count, kv_heads, head_size,
                                      static_cast<uint32_t>(kBlockSize), grid.tokens_per_core,
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

    // TND wants the cumulative token count at the end of each sequence, for the
    // query and the key independently: a chunked prefill has C query tokens
    // against S keys, which is exactly what sparse_mode 3 anchors bottom-right.
    std::vector<int64_t> cumulative_q(static_cast<size_t>(config_.batch));
    std::vector<int64_t> cumulative_kv(static_cast<size_t>(config_.batch));
    for (int64_t seq = 0; seq < config_.batch; ++seq) {
      cumulative_q[static_cast<size_t>(seq)] = (seq + 1) * config_.chunk;
      cumulative_kv[static_cast<size_t>(seq)] = (seq + 1) * config_.seq_len;
    }
    prefill_seq_q_.reset(new AclnnIntArray(cumulative_q));
    prefill_seq_kv_.reset(new AclnnIntArray(cumulative_kv));

    // The decode role: one query token per sequence, the paged cache as a
    // one-entry list, and the context length per sequence.
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
    // The plugin's own decode convention, which the golden test reproduces
    // argument for argument: actualSeqLengths is the CUMULATIVE query token
    // count -- one token per sequence, so 1, 2, ... B -- and actualSeqLengthsKv
    // is the context length PER SEQUENCE, not cumulative. The two readings
    // coincide at B=1, which is why nothing in this tree had to distinguish them
    // before; at B > 1 they do not, and this follows
    // AscendAttentionBackendImpl._get_fia_params rather than guessing.
    {
      std::vector<int64_t> cumulative_q(static_cast<size_t>(config_.batch));
      for (int64_t seq = 0; seq < config_.batch; ++seq) {
        cumulative_q[static_cast<size_t>(seq)] = seq + 1;
      }
      decode_seq_q_.reset(new AclnnIntArray(cumulative_q));
      std::vector<int64_t> kv_lens(static_cast<size_t>(config_.batch), config_.seq_len);
      decode_seq_kv_.reset(new AclnnIntArray(kv_lens));
    }

    // The native write's descriptors, over the chunk.
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
    // Two plans over the same shape: TurboQuant's core reads the rotated basis
    // (Q~, K~, V~) and produces O~; the native leg reads the plain one. Pi is
    // orthogonal, so the two compute the same attention up to that rotation --
    // which is what the rotated-basis identity check asserts on the device.
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
          /*compress_lens=*/nullptr, /*compress_seq_offset=*/nullptr, /*seq_lens=*/nullptr,
          const_cast<char*>(ops::kScatterCacheModeNorm), /*scatter_mode=*/nullptr, /*strides=*/nullptr,
          /*offsets=*/nullptr)));
    } catch (const AclError& error) {
      native_write_note_ = std::string(ops::kScatterPaKvCache) + ": " + error.what();
    }
  }

  // The prompt role: contiguous TND K/V, no paging, sparse_mode 3 with the
  // compressed causal mask. V5 first, V2 as a fallback.
  std::unique_ptr<PlannedOp> PlanPrefill(const aclTensor* query, const aclTensorList* key_list,
                                         const aclTensorList* value_list, const aclTensor* out,
                                         const aclTensor* lse) {
    const int64_t hq = config_.model.num_heads;
    const int64_t hkv = config_.model.num_kv_heads;
    const double scale = static_cast<double>(config_.attention_scale());
    try {
      std::unique_ptr<PlannedOp> planned(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV5WorkspaceFn>(
          FiaV5(), query,
          key_list, value_list, /*pse_shift=*/nullptr, mask_tensor_->get(), prefill_seq_q_->get(),
          prefill_seq_kv_->get(), /*deq_scale1=*/nullptr, /*quant_scale1=*/nullptr, /*deq_scale2=*/nullptr,
          /*quant_scale2=*/nullptr, /*quant_offset2=*/nullptr, /*antiquant_scale=*/nullptr,
          /*antiquant_offset=*/nullptr, /*block_table=*/nullptr, /*query_padding_size=*/nullptr,
          /*kv_padding_size=*/nullptr, /*key_antiquant_scale=*/nullptr, /*key_antiquant_offset=*/nullptr,
          /*value_antiquant_scale=*/nullptr, /*value_antiquant_offset=*/nullptr, /*key_shared_prefix=*/nullptr,
          /*value_shared_prefix=*/nullptr, /*actual_shared_prefix_len=*/nullptr, /*query_rope=*/nullptr,
          /*key_rope=*/nullptr, /*key_rope_antiquant_scale=*/nullptr, /*dequant_scale_query=*/nullptr,
          /*learnable_sink=*/nullptr, /*q_start_idx=*/nullptr, /*kv_start_idx=*/nullptr, hq, scale,
          s950::kFiaUnboundedTokens, s950::kFiaUnboundedTokens, const_cast<char*>(ops950::kFiaLayoutTnd), hkv,
          kFiaSparseModeRightDownCausal, s950::kFiaInnerPreciseDefault, kFiaNoPaging, /*antiquant_mode=*/0,
          /*softmax_lse_flag=*/false, /*key_antiquant_mode=*/0, /*value_antiquant_mode=*/0,
          s950::kFiaQueryQuantModeNone, s950::kFiaPseTypeDefault, out, lse)));
      RecordOperator(ops950::kFusedInferAttentionScoreV5);
      return planned;
    } catch (const AclError& error) {
      AppendNote(std::string(ops950::kFusedInferAttentionScoreV5) + " (prompt role): " + error.what());
    }
    try {
      std::unique_ptr<PlannedOp> planned(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV2WorkspaceFn>(
          FiaV2(), query, key_list,
          value_list, /*pse_shift=*/nullptr, mask_tensor_->get(), prefill_seq_q_->get(), prefill_seq_kv_->get(),
          /*deq_scale1=*/nullptr, /*quant_scale1=*/nullptr, /*deq_scale2=*/nullptr, /*quant_scale2=*/nullptr,
          /*quant_offset2=*/nullptr, /*antiquant_scale=*/nullptr, /*antiquant_offset=*/nullptr,
          /*block_table=*/nullptr, /*query_padding_size=*/nullptr, /*kv_padding_size=*/nullptr,
          /*key_antiquant_scale=*/nullptr, /*key_antiquant_offset=*/nullptr, /*value_antiquant_scale=*/nullptr,
          /*value_antiquant_offset=*/nullptr, /*key_shared_prefix=*/nullptr, /*value_shared_prefix=*/nullptr,
          /*actual_shared_prefix_len=*/nullptr, hq, scale, s950::kFiaUnboundedTokens, s950::kFiaUnboundedTokens,
          const_cast<char*>(ops950::kFiaLayoutTnd), hkv, kFiaSparseModeRightDownCausal,
          s950::kFiaInnerPreciseDefault, kFiaNoPaging, /*antiquant_mode=*/0, /*softmax_lse_flag=*/false,
          /*key_antiquant_mode=*/0, /*value_antiquant_mode=*/0, out, lse)));
      RecordOperator(ops950::kFusedInferAttentionScoreV2);
      return planned;
    } catch (const AclError& error) {
      AppendNote(std::string(ops950::kFusedInferAttentionScoreV2) + " (prompt role): " + error.what());
    }
    return nullptr;
  }

  // The incremental role: one query token per sequence over the paged fp16
  // cache, sparse_mode 0, the block table as a tensor.
  std::unique_ptr<PlannedOp> PlanDecode() {
    const int64_t hq = config_.model.num_heads;
    const int64_t hkv = config_.model.num_kv_heads;
    const double scale = static_cast<double>(config_.attention_scale());
    try {
      std::unique_ptr<PlannedOp> planned(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV5WorkspaceFn>(
          FiaV5(), query_dec_tnd_->get(), fp16_key_list_->get(), fp16_value_list_->get(), /*pse_shift=*/nullptr,
          /*atten_mask=*/nullptr, decode_seq_q_->get(), decode_seq_kv_->get(), /*deq_scale1=*/nullptr,
          /*quant_scale1=*/nullptr, /*deq_scale2=*/nullptr, /*quant_scale2=*/nullptr, /*quant_offset2=*/nullptr,
          /*antiquant_scale=*/nullptr, /*antiquant_offset=*/nullptr, block_table_tensor_->get(),
          /*query_padding_size=*/nullptr, /*kv_padding_size=*/nullptr, /*key_antiquant_scale=*/nullptr,
          /*key_antiquant_offset=*/nullptr, /*value_antiquant_scale=*/nullptr, /*value_antiquant_offset=*/nullptr,
          /*key_shared_prefix=*/nullptr, /*value_shared_prefix=*/nullptr, /*actual_shared_prefix_len=*/nullptr,
          /*query_rope=*/nullptr, /*key_rope=*/nullptr, /*key_rope_antiquant_scale=*/nullptr,
          /*dequant_scale_query=*/nullptr, /*learnable_sink=*/nullptr, /*q_start_idx=*/nullptr,
          /*kv_start_idx=*/nullptr, hq, scale, s950::kFiaUnboundedTokens, s950::kFiaUnboundedTokens,
          const_cast<char*>(ops950::kFiaLayoutTnd), hkv, s950::kFiaSparseModeNone,
          s950::kFiaInnerPreciseDefault, kBlockSize, /*antiquant_mode=*/0, /*softmax_lse_flag=*/false,
          /*key_antiquant_mode=*/0, /*value_antiquant_mode=*/0, s950::kFiaQueryQuantModeNone,
          s950::kFiaPseTypeDefault, out_dec_v5_tensor_->get(), lse_dec_tensor_->get())));
      RecordOperator(ops950::kFusedInferAttentionScoreV5);
      return planned;
    } catch (const AclError& error) {
      AppendNote(std::string(ops950::kFusedInferAttentionScoreV5) + " (incremental role): " + error.what());
    }
    try {
      std::unique_ptr<PlannedOp> planned(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV2WorkspaceFn>(
          FiaV2(), query_dec_tnd_->get(), fp16_key_list_->get(), fp16_value_list_->get(), /*pse_shift=*/nullptr,
          /*atten_mask=*/nullptr, decode_seq_q_->get(), decode_seq_kv_->get(), /*deq_scale1=*/nullptr,
          /*quant_scale1=*/nullptr, /*deq_scale2=*/nullptr, /*quant_scale2=*/nullptr, /*quant_offset2=*/nullptr,
          /*antiquant_scale=*/nullptr, /*antiquant_offset=*/nullptr, block_table_tensor_->get(),
          /*query_padding_size=*/nullptr, /*kv_padding_size=*/nullptr, /*key_antiquant_scale=*/nullptr,
          /*key_antiquant_offset=*/nullptr, /*value_antiquant_scale=*/nullptr, /*value_antiquant_offset=*/nullptr,
          /*key_shared_prefix=*/nullptr, /*value_shared_prefix=*/nullptr, /*actual_shared_prefix_len=*/nullptr, hq,
          scale, s950::kFiaUnboundedTokens, s950::kFiaUnboundedTokens, const_cast<char*>(ops950::kFiaLayoutTnd),
          hkv, s950::kFiaSparseModeNone, s950::kFiaInnerPreciseDefault, kBlockSize, /*antiquant_mode=*/0,
          /*softmax_lse_flag=*/false, /*key_antiquant_mode=*/0, /*value_antiquant_mode=*/0,
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
  tqh::CubeDecodeGrid cube_grid_;

  DeviceBuffer key_ctx_, value_ctx_, key_ctx_rot_, value_ctx_rot_;
  DeviceBuffer key_chunk_, value_chunk_;
  DeviceBuffer query_pf_, query_pf_rot_half_, query_pf_rot_fp32_;
  DeviceBuffer out_pf_tq_, out_pf_v5_, out_pf_rot_, lse_pf_tq_, lse_pf_v5_;
  DeviceBuffer query_dec_, query_dec_rot_;
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
  // Last, so destroyed first: a PlannedOp holds the descriptor handles above.
  std::unique_ptr<PlannedOp> fia_prefill_rotated_, fia_prefill_native_, fia_decode_native_, native_write_;
};

// ---------------------------------------------------------------------------
// Runners, one per iteration budget
// ---------------------------------------------------------------------------

class RunnerSet {
 public:
  RunnerSet(BenchmarkRunner& primary, const std::vector<Budget>& budgets) : primary_(&primary) {
    if (budgets.empty()) {
      budgets_.push_back(Budget{kShortWarmup, kShortIterations});
    } else {
      budgets_ = budgets;
    }
    // The first budget goes on the runner the harness owns and will report; the
    // rest each get one of their own. Setting options before any case has run is
    // what keeps every table header describing the rows under it.
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

  // A failure recorded anywhere has to reach the runner the harness reads the
  // exit code from.
  void Fail(const std::string& name, const std::string& why) const { primary_->RecordFailure(name, why); }

  // The same, for a failure already recorded against `runner` so that it shows
  // up in that runner's own report. Mirrored into the primary only when `runner`
  // is not the primary, so a failure is never counted twice.
  void MirrorFail(const BenchmarkRunner& runner, const std::string& name, const std::string& why) const {
    if (&runner != primary_) {
      primary_->RecordFailure(name, why);
    }
  }

  // Everything but the primary, which the harness reports for us after
  // BuildSuite returns.
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

// ---------------------------------------------------------------------------
// Reading the results back
// ---------------------------------------------------------------------------

struct Sample {
  bool present = false;
  // True for a figure that is true by construction rather than measured: a
  // folded layer's T_rot_o. Printed as a number, flagged in the CSV.
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

// Device time: the device-events mode when it ran, the pipelined one otherwise.
// Both are ACL event pairs; the host wall clock is never used for a reported
// figure, only as a cross-check in the raw table.
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

// T_rot_o: exactly zero for a folded layer, measured otherwise.
Sample RotateOutputSample(const std::vector<BenchmarkRunner*>& runners, const char* leg, const Config& config) {
  return config.model.folds_output ? StructuralZero() : SampleFor(runners, leg, config);
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

// The AIV split figure is the measured core minus the measured combine, and is
// marked so no reader takes it for a direct measurement.
void PrintUsDerived(const Sample& sample, int width, bool derived) {
  if (!sample.present) {
    std::printf(" %*s ", width, "-");
    return;
  }
  std::printf(" %*.2f%c", width, sample.median_us, derived ? '~' : ' ');
}

void PrintRatio(double ratio, int width) {
  if (ratio > 0.0) {
    std::printf(" %*.3fx", width - 1, ratio);
  } else {
    std::printf(" %*s", width, "-");
  }
}

// Prints a column header and a rule of exactly its own width underneath. The
// width comes from the snprintf that built the header rather than from a sum of
// the field widths restated here, so the two cannot drift as columns move.
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

// One configuration's worth of derived figures, computed once so the table and
// the CSV cannot disagree about them.
struct PrefillRow {
  Sample ingest, rot_q, attn_core, rot_o, e2e, native, native_ingest;
  double sum_of_parts_us = 0.0;
  double speedup = 0.0;
  double gigabytes_per_second = 0.0;
};

struct DecodeRow {
  Sample rot_q, split, combine, attn_core, rot_o, e2e, native;
  bool split_is_derived = false;
  double sum_of_parts_us = 0.0;
  double speedup = 0.0;
  double gigabytes_per_second = 0.0;
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

DecodeRow ReadDecodeRow(const std::vector<BenchmarkRunner*>& runners, const Config& config,
                        const Traffic& traffic) {
  DecodeRow row;
  row.rot_q = SampleFor(runners, kLegDecRotQ, config);
  row.combine = SampleFor(runners, kLegDecCombine, config);
  row.attn_core = SampleFor(runners, kLegDecAttnCore, config);
  row.rot_o = RotateOutputSample(runners, kLegDecRotO, config);
  row.e2e = SampleFor(runners, kLegDecE2E, config);
  row.native = SampleFor(runners, kLegDecV5, config);

  if (config.path == PathMode::kCube) {
    row.split = SampleFor(runners, kLegDecSplit, config);
  } else if (row.attn_core.present && row.combine.present) {
    // No exported entry point for the AIV split alone; see THE DECODE PATH.
    row.split.present = true;
    row.split.median_us = std::max(0.0, row.attn_core.median_us - row.combine.median_us);
    row.split.mode = row.attn_core.mode;
    row.split_is_derived = true;
  }

  row.sum_of_parts_us = (row.rot_q.present ? row.rot_q.median_us : 0.0) +
                        (row.attn_core.present ? row.attn_core.median_us : 0.0) +
                        (row.rot_o.present ? row.rot_o.median_us : 0.0);
  row.speedup = Speedup(row.native, row.e2e);
  if (row.e2e.present && row.e2e.median_us > 0.0) {
    row.gigabytes_per_second = traffic.dec_e2e / (row.e2e.median_us * 1.0e3);
  }
  return row;
}

// Median relative gap between the composite E2E measurement and the sum of its
// separately measured parts, over the rows that produced both. Reported rather
// than hidden: it is the launch overhead the component view cannot see.
double MedianResidual(const std::vector<double>& residuals) {
  if (residuals.empty()) {
    return 0.0;
  }
  std::vector<double> sorted = residuals;
  std::sort(sorted.begin(), sorted.end());
  return sorted[sorted.size() / 2];
}

// ---------------------------------------------------------------------------
// Table A -- prefill
// ---------------------------------------------------------------------------

void PrintTableA(const std::vector<BenchmarkRunner*>& runners, const std::vector<Config>& sweep,
                 const std::vector<Traffic>& traffic, const std::string& fia_operator) {
  std::printf("\n");
  std::printf("[ascend-bench] ====================================================================\n");
  std::printf("[ascend-bench] TABLE A -- PREFILL PIPELINE PERFORMANCE\n");
  std::printf("[ascend-bench] ====================================================================\n");
  std::printf("[ascend-bench] device time from ACL events, median us per step; native operator: %s\n",
              fia_operator.c_str());
  std::printf("[ascend-bench] TQ_E2E = T_rot_q + T_attn_core + T_rot_o, measured as ONE composite region\n");
  std::printf("[ascend-bench] Context S/C means a chunked step: C query tokens against an S-token prefix\n\n");

  char header[512];
  const int header_width =
      std::snprintf(header, sizeof(header), "  %-17s %13s %3s %9s %4s | %11s %10s %12s %10s %11s %11s | %8s %10s %12s",
                    "Model", "Context (S)", "B", "H_Q/H_KV", "D", "TQ Ingest", "T_rot_q", "T_attn_core", "T_rot_o",
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
    std::printf("  %-17s %13s %3lld %9s %4lld |", config.model.label, config.context_label().c_str(),
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
    PrintDouble(traffic[index].saved_mib(), 12, 1);
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
              "[ascend-bench]   KV Saved MB is the whole batch's context: fp16 residency minus TurboQuant's.\n");
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

// ---------------------------------------------------------------------------
// Table B -- decode
// ---------------------------------------------------------------------------

void PrintTableB(const std::vector<BenchmarkRunner*>& runners, const std::vector<Config>& sweep,
                 const std::vector<Traffic>& traffic, const std::string& fia_operator) {
  std::printf("\n");
  std::printf("[ascend-bench] ====================================================================\n");
  std::printf("[ascend-bench] TABLE B -- DECODE LATENCY BREAKDOWN\n");
  std::printf("[ascend-bench] ====================================================================\n");
  std::printf("[ascend-bench] device time from ACL events, median us per decode step; native operator: %s\n",
              fia_operator.c_str());
  std::printf("[ascend-bench] one decode token per sequence; B is the batch of sequences\n\n");

  char header[512];
  const int header_width =
      std::snprintf(header, sizeof(header), "  %-17s %9s %3s %5s | %10s %14s %11s %10s %11s %11s | %8s %10s %12s",
                    "Model", "Context", "B", "Path", "T_rot_q", "T_DecodeSplit", "T_Combine", "T_rot_o", "TQ_E2E",
                    "V5_Decode", "Speedup", "Eff GB/s", "Compression");
  PrintHeaderAndRule(header, header_width);

  bool any = false;
  bool any_derived = false;
  std::vector<double> residuals;
  for (size_t index = 0; index < sweep.size(); ++index) {
    const Config& config = sweep[index];
    const DecodeRow row = ReadDecodeRow(runners, config, traffic[index]);
    if (!row.rot_q.present && !row.attn_core.present && !row.e2e.present && !row.native.present) {
      continue;
    }
    any = true;
    any_derived = any_derived || row.split_is_derived;
    std::printf("  %-17s %9lld %3lld %5s |", config.model.label, static_cast<long long>(config.seq_len),
                static_cast<long long>(config.batch), PathLabel(config.path));
    PrintUs(row.rot_q, 10);
    PrintUsDerived(row.split, 13, row.split_is_derived);
    PrintUs(row.combine, 11);
    PrintUs(row.rot_o, 10);
    PrintUs(row.e2e, 11);
    PrintUs(row.native, 11);
    std::printf(" |");
    PrintRatio(row.speedup, 8);
    PrintDouble(row.gigabytes_per_second, 10, 1);
    PrintDouble(traffic[index].compression_ratio(), 12, 3);
    std::printf("\n");

    if (row.e2e.present && row.sum_of_parts_us > 0.0) {
      residuals.push_back(std::fabs(row.e2e.median_us - row.sum_of_parts_us) / row.e2e.median_us);
    }
  }
  if (!any) {
    std::printf("  (no decode configuration produced a sample)\n");
  }

  std::printf("\n[ascend-bench]   Speedup is V5_Decode / TQ_E2E: above 1.000x is TurboQuant ahead. TQ_E2E is one\n"
              "[ascend-bench]   composite ACL-event region over rotate-q, the attention core and (unfolded only)\n"
              "[ascend-bench]   rotate-o; T_DecodeSplit + T_Combine are the core's two stages.\n");
  if (any_derived) {
    std::printf("[ascend-bench]   `~` marks a DERIVED split: the AIV path submits both stages through one\n"
                "[ascend-bench]   launcher and exports no split-only entry point, so that column is the measured\n"
                "[ascend-bench]   core minus the measured combine. The Cube path's split is measured directly.\n");
  }
  std::printf("[ascend-bench]   Path is chosen by the GQA group and not by a flag: H_Q/H_KV >= %lld fills the\n"
              "[ascend-bench]   Cube's M fractal and takes the Cube split, anything narrower takes the vector\n"
              "[ascend-bench]   path. ASCEND_BENCH_TQ_AUDIT_PATH overrides it.\n",
              static_cast<long long>(tqh::kCubeTileM));
  std::printf("[ascend-bench]   Eff GB/s is the step's compulsory traffic over its measured composite time.\n"
              "[ascend-bench]   Compression is fp16 KV residency over TurboQuant's, scale plane included -- at\n"
              "[ascend-bench]   H_KV=1 the scale plane's 32-byte burst floor is most of the gap from 4.00x.\n");
  if (!residuals.empty()) {
    std::printf("[ascend-bench]   composite vs sum-of-parts: median gap %.1f%% over %zu rows.\n",
                100.0 * MedianResidual(residuals), residuals.size());
  }
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// CSVs
// ---------------------------------------------------------------------------

// An absent leg leaves its fields EMPTY rather than writing 0: a zero latency in
// a spreadsheet is indistinguishable from a measurement, and a CSV is the
// artifact most likely to be read without its banner. A structural zero -- a
// folded layer's T_rot_o -- is written as 0 and flagged in its own column.
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

void WriteDecodeCsv(const std::vector<BenchmarkRunner*>& runners, const std::vector<Config>& sweep,
                    const std::vector<Traffic>& traffic, const std::string& path,
                    const std::string& fia_operator) {
  std::ofstream csv(path, std::ios::trunc);
  if (!csv) {
    std::printf("[ascend-bench] could not open ASCEND_BENCH_TQ_AUDIT_DECODE_CSV='%s' for writing\n", path.c_str());
    return;
  }
  csv << "model,regime,seq_len,batch,num_heads,num_kv_heads,head_dim,path,num_splits,native_operator,"
      << "t_rot_q_us,t_decode_split_us,split_is_derived,t_combine_us,t_rot_o_us,rot_o_is_structural_zero,"
      << "tq_e2e_us,tq_e2e_sum_of_parts_us,v5_decode_us,net_speedup,"
      << "effective_gbps,attn_core_gbps,v5_gbps,attn_core_tflops,v5_tflops,"
      << "fp16_kv_bytes,tq_kv_bytes,kv_saved_mib,compression_ratio,"
      << "e2e_bytes,attention_flops,tq_e2e_p95_us,v5_p95_us,warmup,iterations,timing_mode\n";

  for (size_t index = 0; index < sweep.size(); ++index) {
    const Config& config = sweep[index];
    const Traffic& model = traffic[index];
    const DecodeRow row = ReadDecodeRow(runners, config, model);

    csv << config.model.key << ',' << config.regime << ',' << config.seq_len << ',' << config.batch << ','
        << config.model.num_heads << ',' << config.model.num_kv_heads << ',' << config.model.head_size << ','
        << PathLabel(config.path) << ',' << model.dec_split_count << ',' << fia_operator;
    WriteCell(csv, row.rot_q);
    WriteCell(csv, row.split);
    csv << ',' << (row.split_is_derived ? 1 : 0);
    WriteCell(csv, row.combine);
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
        << model.compression_ratio() << ',' << model.dec_e2e << ',' << model.dec_flops << ',';
    if (row.e2e.present) {
      csv << row.e2e.p95_us;
    }
    csv << ',';
    if (row.native.present) {
      csv << row.native.p95_us;
    }
    csv << ',' << config.budget.warmup << ',' << config.budget.iterations << ',' << row.e2e.mode << '\n';
  }
  std::printf("[ascend-bench] Table B written to %s\n", path.c_str());
}

// ---------------------------------------------------------------------------
// The two device checks
// ---------------------------------------------------------------------------

// Pi is orthogonal, so FIA over (Q~, K~, V~) un-rotated per head must be FIA
// over (Q, K, V). The device-level statement that the prefill core below is the
// layer's attention, and what licenses timing the folded path with T_rot_o = 0.
double RotatedBasisCosine(const Scenario& scenario, const Config& config, aclrtStream stream) {
  scenario.EnqueuePrefillAttnCore(stream);
  scenario.EnqueuePrefillNative(stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  const std::vector<float> plain = scenario.PrefillNativeOutput();
  const std::vector<float> folded = tqh::UnrotateHeads(scenario.PrefillTqOutput(), config.model.head_size);
  return turboquant_ref::cpu_fidelity(folded, plain).cosine_similarity;
}

// The TurboQuant decode's output, un-rotated, against the native decode over the
// same context in the same slots. A loose bound on purpose: 4 bits against 16
// will not do better, and what this catches is a leg reading an empty cache.
double DecodeTiePointCosine(const Scenario& scenario, const Config& config, aclrtStream stream) {
  scenario.EnqueueDecodeE2E(stream);
  scenario.EnqueueDecodeNative(stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  const std::vector<float> native = scenario.DecodeNativeOutput();
  const std::vector<float> folded = tqh::UnrotateHeads(scenario.DecodeTqOutput(), config.model.head_size);
  return turboquant_ref::cpu_fidelity(folded, native).cosine_similarity;
}

// ---------------------------------------------------------------------------
// Banner
// ---------------------------------------------------------------------------

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

}  // namespace

// ---------------------------------------------------------------------------
// The suite
// ---------------------------------------------------------------------------

void BuildSuite(BenchmarkRunner& primary) {
  bool aiv_queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&aiv_queried);

  const std::vector<Config> sweep = BuildSweep();
  PrintBanner(sweep, aiv_num, aiv_queried);
  if (sweep.empty()) {
    return;
  }

  // One runner per distinct iteration budget, in the order the sweep first asks
  // for them; see RunnerSet.
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

  // The rotated-basis identity runs once, on the configuration whose attention
  // output is smallest, where reading two of them back costs megabytes rather
  // than gigabytes. The decode tie-point runs once, on the smallest
  // configuration whose context also fits an exact host image.
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

    // The split count follows from the grid, which follows from the shape, and
    // the traffic model's partial term needs it. Planned here rather than read
    // off the scenario so that a configuration skipped for HBM still gets a row
    // in the model.
    const int64_t planned_splits =
        config.path == PathMode::kCube
            ? tqh::PlanCubeDecode(config.batch, config.model.num_heads, config.model.num_kv_heads,
                                  config.model.head_size, config.blocks_per_seq(), aiv_num)
                  .num_splits
            : tqh::PlanPagedAttention(config.batch, config.model.num_heads, config.model.head_size,
                                      config.blocks_per_seq(), aiv_num)
                  .num_splits;
    traffic.push_back(ModelTraffic(config, planned_splits));

    // --- the HBM guard ------------------------------------------------------
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

    // --- build --------------------------------------------------------------
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
                "blocks/seq %lld, pool %lld, splits %lld, grid %u/%u | native: %s\n",
                config.model.label, static_cast<long long>(config.seq_len),
                static_cast<long long>(config.batch), static_cast<long long>(config.model.head_size),
                static_cast<long long>(config.model.num_heads),
                static_cast<long long>(config.model.num_kv_heads), PathLabel(config.path),
                static_cast<long long>(config.chunk), static_cast<long long>(config.blocks_per_seq()),
                static_cast<long long>(config.pool_blocks()), static_cast<long long>(scenario->num_splits()),
                scenario->split_block_dim(), scenario->combine_block_dim(),
                scenario->decode_available() ? scenario->fia_operator().c_str() : "unavailable");
    if (!scenario->prefill_available() || !scenario->decode_available()) {
      std::printf("[ascend-bench]   native operator note: %s\n", scenario->fia_note().c_str());
    }
    std::fflush(stdout);

    // --- setup, outside every timed region ----------------------------------
    //
    // The cache is filled from the whole context, then one untimed pass leaves
    // the flash-decoding workspace and both attention outputs holding real
    // values -- see Scenario::Prime for why a timed combine cannot start from
    // zeros.
    try {
      scenario->FillCache(runner.stream());
      scenario->Prime(runner.stream());
    } catch (const std::exception& error) {
      runners.Fail(CaseName("fill", config), error.what());
      continue;
    }

    // --- the two device checks, before anything is timed --------------------
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
    std::fflush(stdout);

    // --- registration -------------------------------------------------------
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

    // DECODE FIRST, then prefill. The prefill ingest leg rewrites the tail of
    // the TurboQuant cache from the chunk's own K/V and the native ingest leg
    // rewrites part of the fp16 paged cache, so running them first would leave
    // the decode legs reading a cache the setup phase did not build. Neither
    // rewrite changes any latency -- both write the same bytes to the same slots
    // every launch -- but the order keeps the decode measuring a cache whose
    // provenance is the one the tie-point checked.
    if (run_decode) {
      run_leg(kLegDecRotQ, 0.0, model.dec_rot_q, 1, [sc](aclrtStream s) { sc->EnqueueDecodeRotateQ(s); },
              [sc]() { return sc->DecodeRotatedQueryChecksum(); });

      if (config.path == PathMode::kCube) {
        run_leg(kLegDecSplit, model.dec_flops, model.dec_split, 1,
                [sc](aclrtStream s) { sc->EnqueueDecodeSplit(s); },
                [sc]() { return sc->WorkspaceChecksum(); });
      } else {
        runner.Skip(CaseName(kLegDecSplit, config),
                    "the AIV path submits split and combine through one launcher and exports no split-only "
                    "entry point; Table B derives that column");
      }

      run_leg(kLegDecCombine, 0.0, model.dec_combine, 1, [sc](aclrtStream s) { sc->EnqueueDecodeCombine(s); },
              [sc]() { return sc->DecodeTqChecksum(); });
      run_leg(kLegDecAttnCore, model.dec_flops, model.dec_attn_core, 2,
              [sc](aclrtStream s) { sc->EnqueueDecodeAttnCore(s); },
              [sc]() { return sc->DecodeTqChecksum(); });

      if (config.model.folds_output) {
        runner.Skip(CaseName(kLegDecRotO, config),
                    "W_o is folded: the de-rotation is an offline weight transform and costs 0.0 us at runtime");
      } else {
        run_leg(kLegDecRotO, 0.0, model.dec_rot_o, 1, [sc](aclrtStream s) { sc->EnqueueDecodeRotateO(s); },
                [sc]() { return sc->DecodeRotatedOutChecksum(); });
      }

      // rotate-q, the core's two stages, and the de-rotation when the layer
      // cannot fold. The count has to be right: the harness uses it to decide
      // when the stream is close to full, and a case that undercounts starts
      // measuring the runtime's back-pressure rather than the kernels.
      run_leg(kLegDecE2E, model.dec_flops, model.dec_e2e, 3 + rot_o_tasks,
              [sc](aclrtStream s) { sc->EnqueueDecodeE2E(s); },
              [sc]() { return sc->DecodePipelineChecksum(); });

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

  // The extra budgets' raw tables. The primary's is printed by the harness after
  // this function returns, which is why the audit tables come last: they are the
  // artifact, and the raw per-case table is the record behind them.
  runners.ReportExtras();

  if (run_prefill) {
    PrintTableA(runners.all(), sweep, traffic, fia_operator);
    const std::string path = EnvString("ASCEND_BENCH_TQ_AUDIT_PREFILL_CSV");
    if (!path.empty()) {
      WritePrefillCsv(runners.all(), sweep, traffic, path, fia_operator);
    }
  }
  if (run_decode) {
    PrintTableB(runners.all(), sweep, traffic, fia_operator);
    const std::string path = EnvString("ASCEND_BENCH_TQ_AUDIT_DECODE_CSV");
    if (!path.empty()) {
      WriteDecodeCsv(runners.all(), sweep, traffic, path, fia_operator);
    }
  }
  std::printf("\n[ascend-bench] the raw per-case table for the first iteration budget follows below.\n");
  std::fflush(stdout);
}

}  // namespace bench
}  // namespace test
}  // namespace vllm_ascend
