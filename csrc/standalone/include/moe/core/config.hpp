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

// DeepSeek-V4 Flash geometry and precision contract for the Ascend 950PR
// standalone runner.
//
// Everything here is a named constant: no shape or dtype code is written at a
// call site. The routed-expert half mirrors
// `tools/dsv4_moe_runtime/core/config.py` (DeepSeekV4MoEConfig) and
// `csrc/tests/common/aclnn_ops_950pr.hpp`, the two places in this repository
// that already pin it.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ascend_moe {

// ---------------------------------------------------------------------------
// Topology (DeepSeek-V4 Flash)
// ---------------------------------------------------------------------------

inline constexpr int64_t kNumLayers = 43;
inline constexpr int64_t kHiddenSize = 4096;
inline constexpr int64_t kMoeIntermediateSize = 2048;
inline constexpr int64_t kNumRoutedExperts = 256;
inline constexpr int64_t kNumSharedExperts = 1;
inline constexpr int64_t kNumExpertsPerTok = 6;
inline constexpr int64_t kVocabSize = 129280;

// Attention: MLA with the compressed Lightning Indexer.
inline constexpr int64_t kNumAttentionHeads = 64;
inline constexpr int64_t kQLoraRank = 1024;
inline constexpr int64_t kIndexTopK = 512;
inline constexpr int64_t kIndexHeadDim = 128;
inline constexpr int64_t kIndexNumHeads = 64;

// Router / activation.
inline constexpr double kRoutedScalingFactor = 1.5;
inline constexpr double kSwigluLimit = 10.0;
inline constexpr double kRmsNormEpsilon = 1e-6;

// The whole routed working set: |Set_Device| + |Set_Host| must equal this.
inline constexpr int64_t kTotalRoutedExperts = kNumLayers * kNumRoutedExperts;  // 11008

// ---------------------------------------------------------------------------
// MLA KV geometry -- the one piece of the brief that is NOT public
// ---------------------------------------------------------------------------
//
// `tools/dsv4_moe_runtime/core/profiles.py` deliberately models the Flash
// backbone as `backbone=None`, with the comment "its config is not public, and
// a guessed reservation would corrupt the memory budget". The same caution
// applies here and harder: these four numbers size the paged KV cache and the
// FIA V5 descriptors, so a silent guess would produce a runner that allocates
// and computes the wrong thing without ever failing.
//
// They are therefore *defaults carried over from the published DeepSeek MLA
// family* (V2 / V2-Lite / V3 all use 512 / 64 / 128 / 128), overridable on the
// CLI (`--kv-lora-rank`, `--qk-rope-head-dim`, `--qk-nope-head-dim`,
// `--v-head-dim`) and read from a `config.json` beside the weights when one is
// present. `MlaGeometry::provenance` records which of the three supplied them,
// and every report prints it.
inline constexpr int64_t kDefaultKvLoraRank = 512;
inline constexpr int64_t kDefaultQkRopeHeadDim = 64;
inline constexpr int64_t kDefaultQkNopeHeadDim = 128;
inline constexpr int64_t kDefaultVHeadDim = 128;

enum class GeometryProvenance { kFamilyDefault, kCheckpointConfig, kCommandLine };

struct MlaGeometry {
  int64_t kv_lora_rank = kDefaultKvLoraRank;
  int64_t qk_rope_head_dim = kDefaultQkRopeHeadDim;
  int64_t qk_nope_head_dim = kDefaultQkNopeHeadDim;
  int64_t v_head_dim = kDefaultVHeadDim;
  GeometryProvenance provenance = GeometryProvenance::kFamilyDefault;

  // Width of one cached token: the compressed latent plus the rope slice. This
  // is what FIA V5 reads through `key` + `keyRope`, and what the paged cache
  // rows are sized for.
  int64_t kv_row_elements() const { return kv_lora_rank + qk_rope_head_dim; }

  int64_t q_head_dim() const { return qk_nope_head_dim + qk_rope_head_dim; }

  const char* provenance_name() const {
    switch (provenance) {
      case GeometryProvenance::kCheckpointConfig:
        return "checkpoint config.json";
      case GeometryProvenance::kCommandLine:
        return "command line";
      default:
        return "DeepSeek MLA family default (NOT read from a DSV4-Flash config)";
    }
  }
};

// ---------------------------------------------------------------------------
// Precision contract (aclDataType codes, acl/acl_base_rt.h)
// ---------------------------------------------------------------------------

inline constexpr int32_t kAclFloat32 = 0;
inline constexpr int32_t kAclFloat16 = 1;
inline constexpr int32_t kAclInt32 = 3;
inline constexpr int32_t kAclUint8 = 4;
inline constexpr int32_t kAclInt64 = 9;
inline constexpr int32_t kAclBf16 = 27;
inline constexpr int32_t kAclFloat8E4m3Fn = 36;  // backbone / dense GEMMs, MLA
inline constexpr int32_t kAclFloat8E8m0 = 37;    // OCP microscales
inline constexpr int32_t kAclFloat4E2m1 = 40;    // routed expert weights

// ACL_FORMAT_ND
inline constexpr int32_t kAclFormatNd = 2;

// Block scaling: dense / backbone FP8 uses block-128 scales, the routed FP4
// expert GEMM uses OCP block-32 microscales.
inline constexpr int64_t kDenseScaleBlock = 128;
inline constexpr int64_t kRoutedScaleBlock = 32;
inline constexpr int64_t kFp4ElementsPerByte = 2;

// ---------------------------------------------------------------------------
// aclnn scalar selectors used by the pipeline
// ---------------------------------------------------------------------------

// aclnnGroupedMatmulV5 / aclnnGroupedMatmulSwigluQuantV2
inline constexpr int64_t kGmmSplitItemSingleOut = 3;
inline constexpr int64_t kGmmGroupTypeM = 0;
inline constexpr int64_t kGmmGroupListTypeCumsum = 0;
inline constexpr int64_t kGmmActTypeNone = 0;
inline constexpr int64_t kGmmDequantModeMx = 1;

// aclnnMoeInitRoutingV4
inline constexpr int64_t kRoutingDropless = 0;         // dropPadMode
inline constexpr int64_t kRoutingTokensNumCumsum = 0;  // expertTokensNumType
inline constexpr int64_t kRoutingQuantModeNone = 0;
inline constexpr int64_t kRoutingRowIdxGather = 0;

// aclnnSoftplus -- stage 1 of the decomposed sqrtsoftplus router scoring:
// softplus(x) = (1/beta) log(1 + exp(beta * x)) below `threshold`, x above it.
// Matches SoftplusConfig in tools/dsv4_moe_runtime/hardware/v5_ops_moe.py.
inline constexpr double kSoftplusBeta = 1.0;
inline constexpr double kSoftplusThreshold = 20.0;

// aclnnMoeGatingTopKV2 -- noaux_tc over PRE-NORMALIZED sqrtsoftplus scores:
// the bias shifts *selection only*, routing weights keep the raw scores, and
// selection is not group-constrained.
//
// The stock operator cannot express DSV4 Flash's `scoring_func =
// "sqrtsoftplus"` itself: its normType only enumerates the softmax(0) /
// sigmoid(1) of the older aclnnMoeGatingTopK (checked against CANN
// 9.2.0-beta.2: aclnn_moe_gating_top_k_v2.h documents neither the attribute
// nor its range). The scoring therefore runs as the decomposed
// aclnnSoftplus -> aclnnSqrt chain on the same stream (dsv4_pipeline.cpp
// stages `router_softplus` / `router_sqrt`), and the gating stage receives
// scores that are already computed: normType -1 is the pre-normalized /
// pass-through mode, renorm 1 L1-renormalizes the selected top-6 scores to
// sum 1.0 (denominator guarded by eps) before routedScalingFactor is applied.
//
// The exact pass-through value of normType remains a CLI knob
// (`--gating-norm-type`) because no header enumerates it; the device A/B
// against the host reference in v5_ops_moe.py is the bring-up item that
// settles it, and the pipeline's report says so.
inline constexpr int64_t kGatingNormTypePreNormalized = -1;
inline constexpr int64_t kGatingKGroup = 1;
inline constexpr int64_t kGatingGroupCount = 1;
inline constexpr int64_t kGatingGroupSelectMode = 0;
inline constexpr int64_t kGatingRenormL1 = 1;
inline constexpr double kGatingEps = 1e-20;

// aclnnSwigluMxQuant
inline constexpr int64_t kSwigluActivateDimLast = -1;
inline constexpr int64_t kSwigluModeDefault = 0;
inline constexpr int64_t kSwigluGroupModeNone = 0;
inline constexpr int64_t kSwigluAxisLast = -1;
inline constexpr int64_t kSwigluScaleAlgOcp = 0;
inline constexpr double kSwigluGluAlpha = 1.0;
inline constexpr double kSwigluGluBias = 0.0;
inline constexpr double kSwigluMaxDtypeValue = 448.0;  // FP8 E4M3FN maximum

// aclnnFusedInferAttentionScoreV5
inline constexpr const char* kFiaLayoutTnd = "TND";
inline constexpr int64_t kFiaSparseModeBand = 3;
inline constexpr int64_t kFiaInnerPreciseHighPrecision = 0;
inline constexpr int64_t kFiaAntiquantModeNone = 0;
inline constexpr int64_t kFiaQueryQuantModeNone = 0;
inline constexpr int64_t kFiaPseTypeNone = 0;
inline constexpr int64_t kFiaPreTokensAll = 65536;
inline constexpr int64_t kFiaNextTokensCausal = 0;

// aclnnMatmul / aclnnApplyRotaryPosEmbV2 / aclnnScatterPaKvCache
inline constexpr int8_t kCubeMathTypeKeepDtype = 0;
inline constexpr int64_t kRotaryLayoutBsnd = 1;
inline constexpr const char* kRotaryModeHalf = "half";
inline constexpr const char* kScatterCacheModeNorm = "Norm";

// ---------------------------------------------------------------------------
// Memory hierarchy bounds (ported from hardware/exclusive_partition.py)
// ---------------------------------------------------------------------------

// One bounded transit chunk, on the host and on the device. The slot exchange
// is chunked at this size so an in-place swap never needs a resident-sized
// mirror.
inline constexpr size_t kTransferChunkBytes = 4ull * 1024 * 1024;

// Pinned host memory is requested in fixed blocks rather than one monolithic
// reservation: a single ~100 GiB pin request can exceed a driver allocation
// limit or round up pathologically. Each block still holds a whole number of
// expert slots.
inline constexpr size_t kHostBlockBytes = 1ull * 1024 * 1024 * 1024;

// Headroom left to the operating system after the pinned partition.
inline constexpr size_t kHostReserveBytes = 2ull * 1024 * 1024 * 1024;

// Device-side headroom left for the CANN runtime's own allocations beyond the
// activation arena and the slot pool.
inline constexpr size_t kDeviceReserveBytes = 512ull * 1024 * 1024;

// Every region inside an expert slot starts at this alignment, and the slot
// total is rounded up to it.
inline constexpr size_t kSlotRegionAlignBytes = 128;

// Activation-arena alignment. 512 covers the 32-byte UB requirement, the
// 64-byte on-chip D2D copy requirement and the FP4 / E8M0 block strides.
inline constexpr size_t kArenaAlignBytes = 512;

// Alignment the simulated device allocation hands out. aclrtMalloc with
// ACL_MEM_MALLOC_HUGE_FIRST serves page-aligned (in practice huge-page-aligned)
// memory, and the arena contract is stated on ABSOLUTE descriptor addresses, so
// the simulator must guarantee at least the finest alignment any reservation
// asks for -- the smoke test checks a 4096-byte reservation on its absolute
// address, not just its offset.
inline constexpr size_t kSimDeviceAllocAlignBytes = 4096;

// ---------------------------------------------------------------------------
// MoE execution path
// ---------------------------------------------------------------------------
//
// The brief's primary mapping is the decomposed stack (GroupedMatmulV5 ->
// SwigluMxQuant -> GroupedMatmulV5 -> MoeTokenUnpermute) and offers the fused
// ops as alternatives. On a 950PR the choice is not stylistic:
// `opp/.../kernel/ascend950` ships `grouped_matmul_swiglu_quant_v2_apt` and
// `grouped_matmul_finalize_routing_apt` but has **no `moe_token_unpermute`
// kernel directory at all**, so the decomposed combine cannot execute on the
// part. `kFused` is therefore the default; `kDecomposed` stays selectable for a
// toolkit that does ship it, and `OpTable` reports which ops resolved.
enum class MoePath { kFused, kDecomposed };

// ---------------------------------------------------------------------------
// Runtime configuration (populated by the CLI)
// ---------------------------------------------------------------------------

struct RuntimeConfig {
  std::string weights_path;
  std::string prompt;
  int64_t max_new_tokens = 512;

  // K: resident routed-expert slots in device HBM. -1 asks the planner for the
  // largest K the device can hold.
  int64_t vram_slots = -1;

  int32_t device_id = 0;

  // Paged KV cache.
  int64_t block_size = 128;
  int64_t max_context_len = 8192;

  MlaGeometry mla;
  MoePath moe_path = MoePath::kFused;
  int64_t gating_norm_type = kGatingNormTypePreNormalized;

  // aclnnQuantMatmulV5's `groupSize` packs the per-axis block sizes of the two
  // scale tensors. CANN 9.2.0-beta.2's header declares the argument and
  // documents neither its encoding nor its range, so the dense block-128 weight
  // scale cannot be expressed here from the headers alone. 0 asks the operator
  // for its default; `--dense-group-size` sets it once a device A/B has
  // established the encoding. The scale tensors themselves are always laid out
  // block-128 along K, per the brief, so only this one scalar is open.
  int64_t dense_group_size = 0;

  // Bring-up switches. `routed_coverage` shrinks |Set_Device| + |Set_Host|
  // below the full 11008 so the hierarchy can be exercised on a host that
  // cannot pin ~137 GiB; production runs leave it at kTotalRoutedExperts.
  int64_t routed_coverage = kTotalRoutedExperts;
  bool synthetic_weights = false;
  bool dry_run = false;
  bool verbose = false;
  std::string report_path;
};

}  // namespace ascend_moe
