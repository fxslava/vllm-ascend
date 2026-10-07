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

// The Exclusive RAM/VRAM swap engine in native C++.
//
// PORTED FROM: `tools/dsv4_moe_runtime/hardware/exclusive_partition.py`
// (`ExclusiveExpertPartition`) together with the physical-residency half of
// `core/slot_pool.py` (`StaticExpertSlotPool`).
//
// A NOTE ON THE BRIEF'S FILE REFERENCE
// ------------------------------------
// The task names `hardware/exclusive_staging.py` as the source. That file is a
// *different* design: `ExclusiveStagingProvider` holds no permanent host copy
// at all ("staged means the bytes currently sit somewhere inside the bounded
// window"), and a window miss **streams the expert in from disk during
// execution**. It cannot satisfy the two invariants the brief then states --
// |Set_Host| = 11008 - K resident host slots, and no disk reads during
// execution -- because it has neither. `exclusive_partition.py` has exactly
// those invariants, by name: `HOST_BLOCK_BYTES = 1 GiB`,
// `TRANSFER_CHUNK_BYTES = 4 MiB`, `h2d_stream` / `d2h_stream`,
// `validate_residency` enforcing `gpu_keys & ram_keys == {}` and
// `gpu_keys | ram_keys == self.keys`, `read_param` raising "runtime disk reads
// are forbidden in the exclusive hierarchy", and `disk_source.close()` after
// ingestion. This port follows `exclusive_partition.py`, which is what the
// brief describes.
//
// ONE DELIBERATE IMPROVEMENT OVER THE PYTHON
// ------------------------------------------
// The Python `_swap` stages the incoming bytes through a *host* scratch with a
// CPU `ctypes.memmove`, then does H2D and D2H in parallel and
// `synchronize_stream`s both -- a host synchronization per 4 MiB chunk. That
// cannot coexist with the brief's "zero CPU-device synchronization calls inside
// the 43-layer loop".
//
// This port parks the victim on the *device* instead, so the exchange is three
// device-side stages with no host participation at all:
//
//   1. D2D   transit_scratch[half] <= device_slot            (park the victim)
//   2. H2D   device_slot           <= host_slot              (promote)
//   3. D2H   host_slot             <= transit_scratch[half]  (land the victim)
//
// Stage 2 must finish before stage 3 (both touch `host_slot`; 2 reads it, 3
// writes it) and stage 1 before stage 2 (both touch `device_slot`). Each
// dependency is an `aclrtRecordEvent` / `aclrtStreamWaitEvent` pair, and the
// scratch is double-buffered so chunk c+2 can start parking while chunk c is
// still landing. The host transit scratch is still allocated, and is still
// pinned, but its job is startup ingestion -- where a staged H2D genuinely
// needs page-locked source memory -- not the steady-state swap.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "dsv4_config.hpp"
#include "dsv4_device_ops.hpp"
#include "dsv4_expert_layout.hpp"
#include "dsv4_weight_source.hpp"

namespace vllm_ascend {
namespace dsv4 {

inline constexpr int32_t kNoSlot = -1;
inline constexpr int32_t kTransitHalves = 2;

// Flat routed-expert identity: `layer * num_experts + expert`.
inline int32_t FlatExpertKey(int32_t layer, int32_t expert, int32_t num_experts) {
  return layer * num_experts + expert;
}

struct ExclusiveStagingStats {
  uint64_t layer_requests = 0;
  uint64_t slot_hits = 0;
  uint64_t slot_misses = 0;
  uint64_t swaps = 0;
  uint64_t swap_chunks = 0;
  uint64_t host_to_device_bytes = 0;
  uint64_t device_to_host_bytes = 0;
  uint64_t device_to_device_bytes = 0;
  uint64_t startup_bytes = 0;
  // Must stay at zero for the lifetime of a run: it is incremented only by the
  // guard that refuses a read from a closed source.
  uint64_t runtime_disk_read_attempts = 0;
};

// What `PlanLayer` decided, before any byte moved. Fixed-capacity so planning
// allocates nothing.
struct LayerSwapPlan {
  static constexpr int32_t kMaxTopK = 16;
  int32_t layer = -1;
  int32_t count = 0;
  int32_t hit_count = 0;
  int32_t miss_count = 0;
  int32_t expert_ids[kMaxTopK] = {};
  int32_t incoming_keys[kMaxTopK] = {};  // flat key of each requested expert
  int32_t device_slots[kMaxTopK] = {};   // slot each requested expert ends up in
  bool was_hit[kMaxTopK] = {};
  int32_t victim_keys[kMaxTopK] = {};    // flat key evicted to make room (misses only)
};

class ExclusiveExpertManager {
 public:
  struct Options {
    int64_t num_layers = kNumLayers;
    int64_t num_experts = kNumRoutedExperts;
    // |Set_Device| + |Set_Host|. The production value is
    // num_layers * num_experts; a smaller coverage is a bring-up affordance and
    // is reported as such.
    int64_t routed_coverage = kTotalRoutedExperts;
    int64_t device_slots = 0;  // K
    size_t transfer_chunk_bytes = kTransferChunkBytes;
    size_t host_block_bytes = kHostBlockBytes;
    int64_t top_k = kNumExpertsPerTok;
    // 0 = consult /proc/meminfo before pinning the host half (production). A
    // positive value overrides that check, for hosts whose pinned memory is
    // symbolic rather than physical -- the mock runtime's interval registry,
    // which carries the full 137 GiB routed set as spans, not bytes.
    int64_t host_available_bytes = 0;
  };

  ExclusiveExpertManager(DeviceOps& device, const ExpertSlotLayout& layout, const Options& options);
  ~ExclusiveExpertManager();

  ExclusiveExpertManager(const ExclusiveExpertManager&) = delete;
  ExclusiveExpertManager& operator=(const ExclusiveExpertManager&) = delete;

  // ---- planning ---------------------------------------------------------

  // Largest K that fits, given the free HBM and everything else already
  // reserved. Mirrors `exclusive_partition.plan_vram_slots`.
  static int64_t PlanDeviceSlots(size_t free_device_bytes, size_t fixed_device_bytes, size_t slot_bytes,
                                 int64_t routed_coverage, size_t device_reserve_bytes, size_t transfer_chunk_bytes,
                                 int64_t minimum_slots, int64_t requested);

  // MemAvailable from /proc/meminfo, or 0 when it cannot be read.
  static size_t QueryHostAvailableBytes();

  // ---- ingestion --------------------------------------------------------

  // Fills every slot once from `source` and then closes it. `hot_keys` names
  // the experts that start resident in HBM; the rest go to pinned host RAM.
  // When `hot_keys` is empty a layer-major prefix is used.
  //
  // On return: the disjointness invariant holds, `source.closed()` is true, and
  // no descriptor under the source's root is open.
  void Ingest(WeightByteSource& source, const std::vector<int32_t>& hot_keys);

  // ---- decode-step path -------------------------------------------------

  // Plans the layer's top-k without mutating anything. Throws before any state
  // changes if an admission has no authoritative host slot.
  LayerSwapPlan PlanLayer(int32_t layer, const int32_t* expert_ids, int32_t count) const;

  // Executes the plan: the batched, event-ordered duplex exchange, then the
  // residency bookkeeping. `compute_stream` is made to wait on the batch, and
  // `compute_done` is waited on first so the previous layer's GEMM cannot still
  // be reading a slot this batch is about to overwrite.
  //
  // `compute_done` is REQUIRED whenever the plan has a miss and must be an
  // event the caller has already recorded on `compute_stream` after the
  // previous layer's expert GEMMs. That event is the only thing standing
  // between this batch's D2D/H2D and an in-flight read of the same slot --
  // victim selection deliberately does not reserve the previous layer's slots,
  // because a K equal to top_k would then have nothing left to evict.
  void ExecutePlan(const LayerSwapPlan& plan, DeviceStream compute_stream, DeviceEvent compute_done);

  // PlanLayer + ExecutePlan.
  LayerSwapPlan PrepareLayer(int32_t layer, const int32_t* expert_ids, int32_t count, DeviceStream compute_stream,
                             DeviceEvent compute_done);

  // ---- addresses --------------------------------------------------------

  void* DeviceSlotAddress(int32_t slot) const;
  int32_t DeviceSlotOf(int32_t layer, int32_t expert) const;
  void* RegionAddress(int32_t slot, ExpertRegionId region) const;

  // ---- invariants -------------------------------------------------------

  // Set_Device and Set_Host are disjoint, together cover exactly
  // `routed_coverage` experts, occupancies are K and coverage-K, and no two
  // experts alias one slot. Throws with the arithmetic on failure. O(coverage),
  // so it is an ingestion / shutdown / test check, not a per-layer one.
  void ValidateResidency() const;

  bool poisoned() const { return poisoned_; }
  int64_t device_slot_count() const { return options_.device_slots; }
  int64_t host_slot_count() const { return host_slot_count_; }
  size_t device_bytes() const { return device_arena_bytes_; }
  size_t host_bytes() const { return host_arena_bytes_; }
  size_t slot_num_bytes() const { return layout_.slot_num_bytes(); }
  const ExclusiveStagingStats& stats() const { return stats_; }
  const ExpertSlotLayout& layout() const { return layout_; }

  // Addresses that must never move after construction: the device arena, every
  // pinned host block, and the transit scratches. The runner fingerprints them
  // at init and re-checks at shutdown.
  std::vector<const void*> Fingerprint() const;

  std::string DescribeHierarchy() const;

  // Drain the transfer streams. Called at shutdown and by the contract test
  // between phases; never from inside the 43-layer loop.
  void Synchronize();

 private:
  struct TransitHalf {
    void* device_scratch = nullptr;
    DeviceEvent parked = nullptr;    // D2D into the scratch has completed
    DeviceEvent promoted = nullptr;  // H2D out of the host slot has completed
    DeviceEvent landed = nullptr;    // D2H into the host slot has completed
  };

  void AllocateDeviceArena();
  void AllocateHostArena();
  void AllocateTransit();
  void PrimeEvents();
  uint8_t* HostSlotAddress(int32_t host_slot) const;
  void ExchangeSlot(int32_t device_slot, int32_t host_slot);
  void RefuseIfPoisoned() const;

  DeviceOps& device_;
  ExpertSlotLayout layout_;
  Options options_;

  // Device side: one arena, K slots at fixed offsets.
  void* device_arena_ = nullptr;
  size_t device_arena_bytes_ = 0;

  // Host side: fixed ~1 GiB pinned blocks, each holding a whole number of
  // slots, so a single huge pin request is never made.
  std::vector<void*> host_blocks_;
  size_t host_slots_per_block_ = 0;
  int64_t host_slot_count_ = 0;
  size_t host_arena_bytes_ = 0;

  // Bounded transit: two device halves for the steady-state exchange, one
  // pinned host chunk for ingestion.
  TransitHalf transit_[kTransitHalves];
  void* host_transit_scratch_ = nullptr;
  size_t transit_chunk_bytes_ = 0;

  DeviceStream h2d_stream_ = nullptr;
  DeviceStream d2h_stream_ = nullptr;
  DeviceStream stage_stream_ = nullptr;
  DeviceEvent h2d_batch_done_ = nullptr;
  DeviceEvent d2h_batch_done_ = nullptr;
  bool first_batch_ = true;

  // Residency, zero-alloc after construction.
  std::vector<int32_t> device_slot_of_;     // flat key -> slot, or kNoSlot
  std::vector<int32_t> host_slot_of_;       // flat key -> host slot, or kNoSlot
  std::vector<int32_t> slot_owner_;         // slot -> flat key, or kNoSlot
  std::vector<uint64_t> slot_last_touch_;   // LRU ordering
  std::vector<uint32_t> access_count_;      // frequency, the LRU tie-break
  // Victim-selection scratch over K, sized once so PlanLayer allocates nothing.
  mutable std::vector<uint8_t> claim_scratch_;
  uint64_t touch_clock_ = 0;
  uint64_t layer_epoch_ = 0;

  bool ingested_ = false;
  bool poisoned_ = false;
  ExclusiveStagingStats stats_;
};

}  // namespace dsv4
}  // namespace vllm_ascend
