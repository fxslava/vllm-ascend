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

// The 43-layer DeepSeek-V4 Flash decode graph on ACLNN V5, as an ORCHESTRATOR.
//
// Dsv4Pipeline no longer does everything itself; it wires the pieces together
// and owns the per-layer replay:
//
//   StaticArenaManager       what is allocated and which descriptors cover it
//   MoeRouterEngine          scoring -> selection -> dropless dispatch
//   ExclusiveExpertManager   routed-expert residency (passed in)
//   IDeviceAllocator / IStreamEngine   the backend contract (passed in)
//
// What remains here is WHEN things run: the attention half of every layer, the
// expert GEMMs and the shared expert, the head, and the step boundaries.
//
// ONE EXECUTOR PER STAGE, NOT PER LAYER
// -------------------------------------
// All 43 layers have identical shapes, so each stage is planned exactly once
// and replayed 43 times with `aclSetTensorAddr` / `aclSetDynamicTensorAddr`
// swapping that layer's weights and the step's activations into the retained,
// repeatable executor. That is ~26 planned executors for the whole model
// instead of 26 x 43, and it is what the brief's "all descriptors created ONCE
// during initialization, dynamic addresses swapped via aclSetTensorAddr"
// describes.
//
// THREE DEVIATIONS FROM THE BRIEF, ALL FORCED, ALL REPORTED
// ---------------------------------------------------------
// 1. ONE HOST SYNCHRONIZATION PER MoE LAYER, not one per step.
//
//    The brief asks for "zero CPU-device synchronization calls inside the
//    43-layer loop" (3.4) *and* for a host-managed exclusive swap engine whose
//    residency decisions are made in C++ tables (2). Those cannot both hold:
//    the engine has to know which six experts the router chose before it can
//    decide what to promote and what to evict, and the router runs on the
//    device. So each MoE layer copies the 24-byte top-6 index vector D2H and
//    synchronizes (MoeRouterEngine::ScoreAndSelect). `StepCounters` counts
//    them and the report prints the expected total, so the cost is visible
//    rather than hidden. Removing it needs either a device-side residency
//    table the GEMM indexes itself -- which means a custom Ascend C kernel,
//    excluded by 3 -- or routing the next layer one layer early. Both are
//    named in README.md.
//
// 2. THE LIGHTNING INDEXER IS NOT APPLIED.
//
//    DSV4 Flash selects `index_topk = 512` keys per query with a compressed
//    indexer. The brief's operator mapping (3.1) specifies
//    `aclnnFusedInferAttentionScoreV5` for the attention stage, which is the
//    *dense* paged MLA decode; the sparse selection belongs to
//    `aclnnSparseFlashMla` (+ `...Metadata`), which the mapping does not list.
//    This pipeline therefore attends the full context, as specified, and says
//    so in its report. It is a functional difference from the model, not a
//    performance one.
//
// 3. THE COMBINE IS A [1, 6] x [6, hidden] MATMUL, NOT MoeTokenUnpermute.
//
//    `aclnnMoeTokenUnpermute` has no `ascend950` kernel directory in CANN
//    9.2.0-beta.2, and `aclnnGroupedMatmulFinalizeRoutingV3` -- which does --
//    takes its expert weights as ONE tensor, i.e. contiguous, which an
//    exclusive LRU slot pool cannot promise. For a single-token step the six
//    expanded rows are the same token, so the weighted sum IS a small matmul
//    over the permuted routing weights, which is exact and runs on the part.
//    See the comment on the `expert_combine` stage.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "moe/core/config.hpp"
#include "moe/core/device_allocator.hpp"
#include "moe/core/device_types.hpp"
#include "moe/core/op_table.hpp"
#include "moe/core/stream_engine.hpp"
#include "moe/core/weight_source.hpp"
#include "moe/memory/exclusive_staging.hpp"
#include "moe/pipeline/moe_router_engine.hpp"
#include "moe/pipeline/static_arena_manager.hpp"

namespace ascend_moe {

struct StepCounters {
  uint64_t steps = 0;
  uint64_t layers = 0;
  uint64_t launches = 0;
  // Must be exactly `kNumLayers` per step (deviation 1) plus one for the
  // argmax readback. Anything above that is a defect.
  uint64_t host_synchronizations = 0;
  uint64_t device_allocations_in_step = 0;  // must stay 0
  uint64_t descriptors_built_in_step = 0;   // must stay 0
  uint64_t expert_slot_hits = 0;
  uint64_t expert_slot_misses = 0;
};

// Which tensor index inside an op's retained executor each swappable address
// occupies. ACLNN numbers an executor's tensors in IR order -- every input in
// declaration order, then every output -- counting only tensor and tensor-list
// arguments and skipping the scalars. Each constant below is derived that way
// from the signature in op_table.hpp, with the argument it refers to
// named.
//
// This derivation is the one part of the design the plan phase cannot confirm:
// `GetWorkspaceSize` succeeds whatever indices a later `aclSetTensorAddr`
// uses, and a wrong index silently repoints the wrong tensor.
// `DescribeSlotIndexMap()` prints the whole map for exactly that reason, so a
// device bring-up can check it against each op's IR definition before trusting
// a number.
namespace slot {

// aclnnRmsNorm(x, gamma, eps, yOut, rstdOut)
inline constexpr size_t kRmsNormX = 0;
inline constexpr size_t kRmsNormGamma = 1;
inline constexpr size_t kRmsNormY = 2;
inline constexpr size_t kRmsNormRstd = 3;

// aclnnRmsNormDynamicMxQuant(x, gamma, beta, ..., yOut, mxscaleOut, rstdOut)
inline constexpr size_t kMxQuantX = 0;
inline constexpr size_t kMxQuantGamma = 1;
inline constexpr size_t kMxQuantBeta = 2;
inline constexpr size_t kMxQuantY = 3;
inline constexpr size_t kMxQuantScale = 4;
inline constexpr size_t kMxQuantRstd = 5;

// aclnnMatmul(self, mat2, out, cubeMathType)
inline constexpr size_t kMatmulSelf = 0;
inline constexpr size_t kMatmulMat2 = 1;
inline constexpr size_t kMatmulOut = 2;

// aclnnQuantMatmulV5(x1, x2, x1Scale, x2Scale, yScale, x1Offset, x2Offset,
//                    yOffset, bias, ..., out)
inline constexpr size_t kQuantMmX1 = 0;
inline constexpr size_t kQuantMmX2 = 1;
inline constexpr size_t kQuantMmX1Scale = 2;
inline constexpr size_t kQuantMmX2Scale = 3;
inline constexpr size_t kQuantMmOut = 9;

// aclnnSoftplus(x, beta, threshold, out) and aclnnSqrt(x, out): the two host
// scalars are skipped by the IR numbering, so both unary stages have out at
// tensor index 1. Neither address is ever swapped -- the scoring buffers are
// step-static activations -- but the indices are recorded so a bring-up can
// check them like every other stage.
inline constexpr size_t kSoftplusX = 0;
inline constexpr size_t kSoftplusOut = 1;
inline constexpr size_t kSqrtX = 0;
inline constexpr size_t kSqrtOut = 1;

// aclnnMoeGatingTopKV2(x, bias, inputIds, tid2eid, ..., yOut, expertIdxOut,
//                      outOut)
inline constexpr size_t kGatingX = 0;
inline constexpr size_t kGatingBias = 1;
inline constexpr size_t kGatingY = 4;
inline constexpr size_t kGatingExpertIdx = 5;

// aclnnApplyRotaryPosEmbV2(queryRef, keyRef, cos, sin, ...)
inline constexpr size_t kRotaryCos = 2;
inline constexpr size_t kRotarySin = 3;

// aclnnScatterPaKvCache(key, keyCacheRef, slotMapping, value, valueCacheRef, ..)
inline constexpr size_t kScatterKeyCache = 1;
inline constexpr size_t kScatterValueCache = 4;

// aclnnFusedInferAttentionScoreV5: query, key[], value[] then 22 more tensor
// arguments before queryRope (24) and keyRope (25).
inline constexpr size_t kFiaKeyList = 1;
inline constexpr size_t kFiaValueList = 2;
inline constexpr size_t kFiaKeyRope = 25;

// GroupedMatmulV5(x[], weight[], bias[], scale[], ...) and
// GroupedMatmulSwigluQuantV2(x, weight[], weightScale[], ...)
inline constexpr size_t kGmmWeightList = 1;
inline constexpr size_t kGmmV5ScaleList = 3;
inline constexpr size_t kGmmSwigluScaleList = 2;

}  // namespace slot

// The orchestrator. Allocation lives in StaticArenaManager, router scoring and
// dispatch in MoeRouterEngine, expert residency in the ExclusiveExpertManager
// passed in; this class sequences the 43-layer replay over all of them.
class Dsv4Pipeline {
 public:
  Dsv4Pipeline(IDeviceAllocator& allocator, IStreamEngine& streams, const OpTable& ops,
               ExclusiveExpertManager& experts, const RuntimeConfig& config);
  ~Dsv4Pipeline();

  Dsv4Pipeline(const Dsv4Pipeline&) = delete;
  Dsv4Pipeline& operator=(const Dsv4Pipeline&) = delete;

  // Reserves the arena, ingests the backbone weights from `source`, creates
  // every descriptor, plans every stage and seals. `source` is NOT closed here:
  // the expert manager closes it, after it has taken its own bytes, so there is
  // exactly one place that seals the only copy.
  void Build(WeightByteSource& source);

  // Runs the 43 layers for one token at `position`. Enqueues work; reads
  // nothing back except the per-layer routing vector (deviation 1).
  void DecodeStep(int32_t token_id, int64_t position);

  // The single `aclrtSynchronizeStream` that ends a step, plus the 8-byte
  // readback of the greedy token.
  int32_t ReadArgmaxToken();

  const StepCounters& counters() const { return counters_; }
  const StaticArenaManager& arena_manager() const { return arena_manager_; }
  const MoeRouterEngine& router() const { return router_; }
  std::string DescribeStages() const;
  std::string DescribeSlotIndexMap() const;

  // The memory the backbone needs, so the slot planner can subtract it from
  // free HBM before choosing K. Pure arithmetic; no allocation.
  static size_t BackboneDeviceBytes(const MlaGeometry& mla, int64_t block_size, int64_t max_context_len);

 private:
  // ---- build helpers ----
  void PlanStages();
  ExpertSlotAddresses CollectExpertSlotAddresses();

  // ---- per-layer helpers ----
  void RunAttention(int32_t layer, int64_t position);
  void RunMoe(int32_t layer);
  void RunSharedExpert(int32_t layer);
  void BindExpertWeights(const LayerSwapPlan& plan);
  void Launch(PipelineStage& stage_entry);

  PipelineStage& stage(const char* name);
  const PipelineStage& stage(const char* name) const;

  IDeviceAllocator& allocator_;
  IStreamEngine& streams_;
  const OpTable& ops_;
  ExclusiveExpertManager& experts_;
  RuntimeConfig config_;

  StaticArenaManager arena_manager_;
  MoeRouterEngine router_;

  DeviceStream compute_stream_ = nullptr;
  DeviceEvent compute_done_ = nullptr;

  // `stages_` holds StaticOpSlot, which is move-disabled -- not even
  // std::vector::reserve compiles for it, because libstdc++'s reallocation path
  // move-empties a zero-size vector. The vector is therefore born at full
  // capacity and only the first `stage_count_` entries are live; it is never
  // resized or copied again.
  static constexpr size_t kMaxPipelineStages = 40;
  std::vector<PipelineStage> stages_ = std::vector<PipelineStage>(kMaxPipelineStages);
  size_t stage_count_ = 0;
  StepCounters counters_;

  // Shapes, fixed at construction from the config.
  int64_t heads_ = kNumAttentionHeads;

  // The pinned host mailboxes for the per-step embedding and cache-slot writes.
  int64_t* token_mailbox_ = nullptr;  // [1] greedy token, D2H
  int32_t* slot_mailbox_ = nullptr;   // [1] paged cache slot, H2D
};

}  // namespace ascend_moe
