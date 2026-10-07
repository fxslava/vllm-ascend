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

#include "dsv4_exclusive_staging.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "dsv4_acl_check.hpp"

namespace vllm_ascend {
namespace dsv4 {

ExclusiveExpertManager::ExclusiveExpertManager(DeviceOps& device, const ExpertSlotLayout& layout,
                                               const Options& options)
    : device_(device), layout_(layout), options_(options) {
  DSV4_REQUIRE(options_.num_layers > 0 && options_.num_experts > 0,
               "routed geometry needs positive layers/experts, got " << options_.num_layers << "/"
                                                                     << options_.num_experts);
  DSV4_REQUIRE(options_.routed_coverage > 0 && options_.routed_coverage <= options_.num_layers * options_.num_experts,
               "routed coverage " << options_.routed_coverage << " must be in (0, "
                                  << (options_.num_layers * options_.num_experts) << "]");
  DSV4_REQUIRE(options_.top_k > 0 && options_.top_k <= LayerSwapPlan::kMaxTopK,
               "top_k " << options_.top_k << " must be in (0, " << LayerSwapPlan::kMaxTopK << "]");
  DSV4_REQUIRE(options_.device_slots >= options_.top_k,
               "K=" << options_.device_slots << " device slots cannot hold a full top-k of " << options_.top_k);
  DSV4_REQUIRE(options_.device_slots <= options_.routed_coverage,
               "K=" << options_.device_slots << " exceeds the routed coverage " << options_.routed_coverage);
  DSV4_REQUIRE(options_.transfer_chunk_bytes > 0, "transit chunk size must be positive");

  transit_chunk_bytes_ = std::min(options_.transfer_chunk_bytes, layout_.slot_num_bytes());
  // The chunk is the DMA granularity of a slot exchange; keeping it a multiple
  // of the region alignment means a chunk boundary never splits a scale byte
  // from the block it describes in a way the layout cannot express.
  transit_chunk_bytes_ = (transit_chunk_bytes_ / kSlotRegionAlignBytes) * kSlotRegionAlignBytes;
  DSV4_REQUIRE(transit_chunk_bytes_ > 0, "transit chunk collapsed below the region alignment");

  host_slot_count_ = options_.routed_coverage - options_.device_slots;

  AllocateDeviceArena();
  AllocateHostArena();
  AllocateTransit();

  device_slot_of_.assign(static_cast<size_t>(options_.routed_coverage), kNoSlot);
  host_slot_of_.assign(static_cast<size_t>(options_.routed_coverage), kNoSlot);
  access_count_.assign(static_cast<size_t>(options_.routed_coverage), 0u);
  slot_owner_.assign(static_cast<size_t>(options_.device_slots), kNoSlot);
  slot_last_touch_.assign(static_cast<size_t>(options_.device_slots), 0ull);
  claim_scratch_.assign(static_cast<size_t>(options_.device_slots), 0u);

  h2d_stream_ = device_.CreateStream();
  d2h_stream_ = device_.CreateStream();
  stage_stream_ = device_.CreateStream();
  h2d_batch_done_ = device_.CreateEvent();
  d2h_batch_done_ = device_.CreateEvent();
  PrimeEvents();
}

ExclusiveExpertManager::~ExclusiveExpertManager() {
  // Best effort: a throwing destructor would mask the original failure.
  try {
    Synchronize();
  } catch (...) {
  }
  for (TransitHalf& half : transit_) {
    if (half.device_scratch != nullptr) {
      device_.DeviceFree(half.device_scratch);
    }
    if (half.parked != nullptr) {
      device_.DestroyEvent(half.parked);
    }
    if (half.promoted != nullptr) {
      device_.DestroyEvent(half.promoted);
    }
    if (half.landed != nullptr) {
      device_.DestroyEvent(half.landed);
    }
  }
  if (host_transit_scratch_ != nullptr) {
    device_.HostPinnedFree(host_transit_scratch_);
  }
  if (h2d_batch_done_ != nullptr) {
    device_.DestroyEvent(h2d_batch_done_);
  }
  if (d2h_batch_done_ != nullptr) {
    device_.DestroyEvent(d2h_batch_done_);
  }
  if (h2d_stream_ != nullptr) {
    device_.DestroyStream(h2d_stream_);
  }
  if (d2h_stream_ != nullptr) {
    device_.DestroyStream(d2h_stream_);
  }
  if (stage_stream_ != nullptr) {
    device_.DestroyStream(stage_stream_);
  }
  for (void* block : host_blocks_) {
    device_.HostPinnedFree(block);
  }
  if (device_arena_ != nullptr) {
    device_.DeviceFree(device_arena_);
  }
}

// ---------------------------------------------------------------------------
// Planning helpers
// ---------------------------------------------------------------------------

int64_t ExclusiveExpertManager::PlanDeviceSlots(size_t free_device_bytes, size_t fixed_device_bytes,
                                                size_t slot_bytes, int64_t routed_coverage,
                                                size_t device_reserve_bytes, size_t transfer_chunk_bytes,
                                                int64_t minimum_slots, int64_t requested) {
  DSV4_REQUIRE(slot_bytes > 0, "expert slot size must be positive");
  // Two transit halves plus the committed reservations come off the top before
  // any slot is counted.
  const size_t overhead = fixed_device_bytes + device_reserve_bytes + kTransitHalves * transfer_chunk_bytes;
  const size_t usable = free_device_bytes > overhead ? free_device_bytes - overhead : 0;
  const int64_t maximum = std::min<int64_t>(routed_coverage, static_cast<int64_t>(usable / slot_bytes));
  const int64_t selected = requested < 0 ? maximum : requested;
  DSV4_REQUIRE(selected >= minimum_slots && selected <= maximum,
               "VRAM plan permits " << minimum_slots << ".." << maximum << " routed slots (" << free_device_bytes
                                    << " free, " << fixed_device_bytes << " already reserved, " << slot_bytes
                                    << " per slot); requested " << selected);
  return selected;
}

size_t ExclusiveExpertManager::QueryHostAvailableBytes() {
  std::FILE* file = std::fopen("/proc/meminfo", "re");
  if (file == nullptr) {
    return 0;
  }
  char line[256];
  size_t available = 0;
  while (std::fgets(line, sizeof(line), file) != nullptr) {
    unsigned long long kilobytes = 0;
    if (std::sscanf(line, "MemAvailable: %llu kB", &kilobytes) == 1) {
      available = static_cast<size_t>(kilobytes) * 1024;
      break;
    }
  }
  std::fclose(file);
  return available;
}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

void ExclusiveExpertManager::AllocateDeviceArena() {
  device_arena_bytes_ = static_cast<size_t>(options_.device_slots) * layout_.slot_num_bytes();
  device_arena_ = device_.DeviceMalloc(device_arena_bytes_);
  // Zeroed so a slot nothing has written reads as zeros rather than as whatever
  // the previous tenant of that HBM page left, which would be a silent wrong
  // answer instead of an obvious one.
  device_.DeviceMemset(device_arena_, device_arena_bytes_, 0, device_arena_bytes_);
}

void ExclusiveExpertManager::AllocateHostArena() {
  if (host_slot_count_ == 0) {
    return;
  }
  const size_t slot_bytes = layout_.slot_num_bytes();
  host_arena_bytes_ = static_cast<size_t>(host_slot_count_) * slot_bytes;

  const size_t available = options_.host_available_bytes > 0
                               ? static_cast<size_t>(options_.host_available_bytes)
                               : QueryHostAvailableBytes();
  if (available != 0) {
    DSV4_REQUIRE(host_arena_bytes_ + options_.transfer_chunk_bytes + kHostReserveBytes <= available,
                 "exclusive partition needs " << host_arena_bytes_ << " pinned bytes for " << host_slot_count_
                                              << " host slots plus a " << kHostReserveBytes
                                              << "-byte host reserve, but MemAvailable is only " << available
                                              << "; lowering --vram-slots is not the fix (that grows the host "
                                                 "half) -- raise K, or lower --routed-coverage for a bring-up run");
  }

  // A single ~100 GiB pin request can exceed a driver allocation limit or round
  // up pathologically in the allocator. Fixed blocks still contain exactly
  // host_slot_count_ slots in total, and every block is page-locked.
  host_slots_per_block_ = std::max<size_t>(1, options_.host_block_bytes / slot_bytes);
  host_blocks_.reserve(static_cast<size_t>((host_slot_count_ + static_cast<int64_t>(host_slots_per_block_) - 1) /
                                           static_cast<int64_t>(host_slots_per_block_)));
  for (int64_t first = 0; first < host_slot_count_; first += static_cast<int64_t>(host_slots_per_block_)) {
    const size_t slots_here = std::min<size_t>(host_slots_per_block_, static_cast<size_t>(host_slot_count_ - first));
    host_blocks_.push_back(device_.HostPinnedMalloc(slots_here * slot_bytes));
  }
}

void ExclusiveExpertManager::AllocateTransit() {
  for (TransitHalf& half : transit_) {
    half.device_scratch = device_.DeviceMalloc(transit_chunk_bytes_);
    device_.DeviceMemset(half.device_scratch, transit_chunk_bytes_, 0, transit_chunk_bytes_);
    half.parked = device_.CreateEvent();
    half.promoted = device_.CreateEvent();
    half.landed = device_.CreateEvent();
  }
  host_transit_scratch_ = device_.HostPinnedMalloc(transit_chunk_bytes_);
  std::memset(host_transit_scratch_, 0, transit_chunk_bytes_);
}

void ExclusiveExpertManager::PrimeEvents() {
  // `StreamWaitEvent` on an event that was never recorded has no defined
  // meaning, and the exchange's first chunk would otherwise wait on exactly
  // such an event. Record each one once on the stream that will own it and
  // drain, so every wait in the steady state refers to a real recording.
  for (TransitHalf& half : transit_) {
    device_.RecordEvent(half.parked, stage_stream_);
    device_.RecordEvent(half.promoted, h2d_stream_);
    device_.RecordEvent(half.landed, d2h_stream_);
  }
  device_.RecordEvent(h2d_batch_done_, h2d_stream_);
  device_.RecordEvent(d2h_batch_done_, d2h_stream_);
  Synchronize();
}

void ExclusiveExpertManager::Synchronize() {
  device_.SynchronizeStream(stage_stream_);
  device_.SynchronizeStream(h2d_stream_);
  device_.SynchronizeStream(d2h_stream_);
}

// ---------------------------------------------------------------------------
// Addresses
// ---------------------------------------------------------------------------

void* ExclusiveExpertManager::DeviceSlotAddress(int32_t slot) const {
  DSV4_REQUIRE(slot >= 0 && slot < options_.device_slots,
               "device slot " << slot << " outside [0, " << options_.device_slots << ")");
  return static_cast<uint8_t*>(device_arena_) + static_cast<size_t>(slot) * layout_.slot_num_bytes();
}

void* ExclusiveExpertManager::RegionAddress(int32_t slot, ExpertRegionId region) const {
  return static_cast<uint8_t*>(DeviceSlotAddress(slot)) + layout_.region(region).offset_bytes;
}

uint8_t* ExclusiveExpertManager::HostSlotAddress(int32_t host_slot) const {
  DSV4_REQUIRE(host_slot >= 0 && host_slot < host_slot_count_,
               "host slot " << host_slot << " outside [0, " << host_slot_count_ << ")");
  const size_t block = static_cast<size_t>(host_slot) / host_slots_per_block_;
  const size_t within = static_cast<size_t>(host_slot) % host_slots_per_block_;
  return static_cast<uint8_t*>(host_blocks_[block]) + within * layout_.slot_num_bytes();
}

int32_t ExclusiveExpertManager::DeviceSlotOf(int32_t layer, int32_t expert) const {
  const int32_t key = FlatExpertKey(layer, expert, static_cast<int32_t>(options_.num_experts));
  DSV4_REQUIRE(key >= 0 && key < options_.routed_coverage,
               "expert (layer=" << layer << ", id=" << expert << ") is outside the routed coverage of "
                                << options_.routed_coverage);
  return device_slot_of_[static_cast<size_t>(key)];
}

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------

void ExclusiveExpertManager::Ingest(WeightByteSource& source, const std::vector<int32_t>& hot_keys) {
  DSV4_REQUIRE(!ingested_, "the exclusive partition has already been ingested");
  RefuseIfPoisoned();
  const size_t slot_bytes = layout_.slot_num_bytes();

  // 1. Decide the placement before any byte moves: `placement[key]` is the
  //    device slot a key starts in, or kNoSlot for a host resident.
  std::vector<int32_t> placement(static_cast<size_t>(options_.routed_coverage), kNoSlot);
  if (hot_keys.empty()) {
    for (int64_t slot = 0; slot < options_.device_slots; ++slot) {
      placement[static_cast<size_t>(slot)] = static_cast<int32_t>(slot);
    }
  } else {
    DSV4_REQUIRE(static_cast<int64_t>(hot_keys.size()) == options_.device_slots,
                 "the hot set names " << hot_keys.size() << " experts but there are " << options_.device_slots
                                      << " device slots; the startup residency must fill every slot exactly once");
    int32_t slot = 0;
    for (int32_t key : hot_keys) {
      DSV4_REQUIRE(key >= 0 && key < options_.routed_coverage,
                   "hot-set key " << key << " is outside the routed coverage");
      DSV4_REQUIRE(placement[static_cast<size_t>(key)] == kNoSlot, "hot-set key " << key << " is named twice");
      placement[static_cast<size_t>(key)] = slot++;
    }
  }

  // 2. Fill. Device slots are staged through the bounded pinned host chunk --
  //    which is what that chunk is for -- and host slots are written directly.
  int32_t next_host_slot = 0;
  for (int64_t key = 0; key < options_.routed_coverage; ++key) {
    const int32_t layer = static_cast<int32_t>(key / options_.num_experts);
    const int32_t expert = static_cast<int32_t>(key % options_.num_experts);
    DSV4_REQUIRE(source.Contains(layer, expert),
                 "expert (layer=" << layer << ", id=" << expert
                                  << ") is outside the byte source coverage; the exclusive hierarchy has no "
                                     "second place to get it from");
    const int32_t slot = placement[static_cast<size_t>(key)];
    if (slot != kNoSlot) {
      uint8_t* staging = static_cast<uint8_t*>(host_transit_scratch_);
      uint8_t* destination = static_cast<uint8_t*>(DeviceSlotAddress(slot));
      for (size_t offset = 0; offset < slot_bytes; offset += transit_chunk_bytes_) {
        const size_t count = std::min(transit_chunk_bytes_, slot_bytes - offset);
        source.ReadExpertSlotRange(staging, transit_chunk_bytes_, offset, count, layer, expert);
        device_.MemcpyAsync(destination + offset, slot_bytes - offset, staging, count, MemcpyKind::kHostToDevice,
                            h2d_stream_);
        // Ingestion is the one place a host synchronization is correct: the
        // next read overwrites the staging chunk the DMA is sourcing from.
        device_.SynchronizeStream(h2d_stream_);
      }
      device_slot_of_[static_cast<size_t>(key)] = slot;
      slot_owner_[static_cast<size_t>(slot)] = static_cast<int32_t>(key);
      slot_last_touch_[static_cast<size_t>(slot)] = ++touch_clock_;
    } else {
      DSV4_REQUIRE(next_host_slot < host_slot_count_,
                   "host slots exhausted at key " << key << ": the partition is mis-sized");
      source.FillExpertSlot(HostSlotAddress(next_host_slot), slot_bytes, layer, expert);
      host_slot_of_[static_cast<size_t>(key)] = next_host_slot;
      ++next_host_slot;
    }
    stats_.startup_bytes += slot_bytes;
  }
  DSV4_REQUIRE(next_host_slot == host_slot_count_,
               "ingestion filled " << next_host_slot << " of " << host_slot_count_ << " host slots");

  // 3. Seal the source. From here on the hierarchy is the only copy, which is
  //    what makes "no disk reads during execution" structural rather than a
  //    convention: there is nothing left to read from.
  source.Close();

  Synchronize();
  ValidateResidency();
  ingested_ = true;
}

// ---------------------------------------------------------------------------
// Decode-step path
// ---------------------------------------------------------------------------

void ExclusiveExpertManager::RefuseIfPoisoned() const {
  DSV4_REQUIRE(!poisoned_,
               "the exclusive partition is unusable after a failed transfer: an in-place exchange cannot be "
               "retried once part of a slot has been overwritten");
}

LayerSwapPlan ExclusiveExpertManager::PlanLayer(int32_t layer, const int32_t* expert_ids, int32_t count) const {
  RefuseIfPoisoned();
  DSV4_REQUIRE(layer >= 0 && layer < options_.num_layers,
               "layer " << layer << " outside [0, " << options_.num_layers << ")");
  DSV4_REQUIRE(count > 0 && count <= options_.top_k,
               "layer request of " << count << " experts, top_k is " << options_.top_k);

  LayerSwapPlan plan;
  plan.layer = layer;
  plan.count = count;

  // Pass 1: classify. Duplicate expert ids would make two plan entries claim
  // one slot, so they are rejected rather than deduplicated silently.
  for (int32_t index = 0; index < count; ++index) {
    const int32_t expert = expert_ids[index];
    DSV4_REQUIRE(expert >= 0 && expert < options_.num_experts,
                 "expert id " << expert << " outside [0, " << options_.num_experts << ")");
    for (int32_t earlier = 0; earlier < index; ++earlier) {
      DSV4_REQUIRE(expert_ids[earlier] != expert, "layer " << layer << " requests expert " << expert << " twice");
    }
    plan.expert_ids[index] = expert;
    plan.device_slots[index] = kNoSlot;
    plan.victim_keys[index] = kNoSlot;
    const int32_t key = FlatExpertKey(layer, expert, static_cast<int32_t>(options_.num_experts));
    DSV4_REQUIRE(key < options_.routed_coverage,
                 "expert (layer=" << layer << ", id=" << expert << ") is outside the routed coverage of "
                                  << options_.routed_coverage);
    plan.incoming_keys[index] = key;
    const int32_t resident = device_slot_of_[static_cast<size_t>(key)];
    plan.was_hit[index] = resident != kNoSlot;
    if (plan.was_hit[index]) {
      plan.device_slots[index] = resident;
      ++plan.hit_count;
    } else {
      DSV4_REQUIRE(host_slot_of_[static_cast<size_t>(key)] != kNoSlot,
                   "requested missing expert (layer=" << layer << ", id=" << expert
                                                      << ") has no authoritative RAM slot");
      ++plan.miss_count;
    }
  }

  // Pass 2: choose a victim per miss -- the least-recently-touched slot that
  // neither this request's hits nor an earlier miss of the same request has
  // already claimed, with the access-frequency counter as the tie-break. The
  // claim bitmap is a member sized at construction, so this allocates nothing.
  const size_t slot_count = static_cast<size_t>(options_.device_slots);
  std::fill_n(claim_scratch_.begin(), slot_count, static_cast<uint8_t>(0));
  for (int32_t index = 0; index < count; ++index) {
    if (plan.was_hit[index]) {
      claim_scratch_[static_cast<size_t>(plan.device_slots[index])] = 1u;
    }
  }
  for (int32_t index = 0; index < count; ++index) {
    if (plan.was_hit[index]) {
      continue;
    }
    int32_t victim_slot = kNoSlot;
    uint64_t oldest_touch = 0;
    uint32_t coldest_count = 0;
    for (size_t slot = 0; slot < slot_count; ++slot) {
      if (claim_scratch_[slot] != 0u) {
        continue;
      }
      const int32_t owner = slot_owner_[slot];
      const uint32_t frequency = owner == kNoSlot ? 0u : access_count_[static_cast<size_t>(owner)];
      const bool better = victim_slot == kNoSlot || slot_last_touch_[slot] < oldest_touch ||
                          (slot_last_touch_[slot] == oldest_touch && frequency < coldest_count);
      if (better) {
        victim_slot = static_cast<int32_t>(slot);
        oldest_touch = slot_last_touch_[slot];
        coldest_count = frequency;
      }
    }
    DSV4_REQUIRE(victim_slot != kNoSlot,
                 "no evictable device slot for layer " << layer << " expert " << plan.expert_ids[index]
                                                       << ": K=" << options_.device_slots
                                                       << " cannot hold a top-k of " << count);
    const int32_t victim_key = slot_owner_[static_cast<size_t>(victim_slot)];
    DSV4_REQUIRE(victim_key != kNoSlot,
                 "device slot " << victim_slot
                                << " has no owner; the exclusive invariant forbids an empty slot after ingestion");
    claim_scratch_[static_cast<size_t>(victim_slot)] = 1u;
    plan.device_slots[index] = victim_slot;
    plan.victim_keys[index] = victim_key;
  }
  return plan;
}

void ExclusiveExpertManager::ExchangeSlot(int32_t device_slot, int32_t host_slot) {
  const size_t slot_bytes = layout_.slot_num_bytes();
  uint8_t* device_base = static_cast<uint8_t*>(DeviceSlotAddress(device_slot));
  uint8_t* host_base = HostSlotAddress(host_slot);

  int32_t half_index = 0;
  for (size_t offset = 0; offset < slot_bytes; offset += transit_chunk_bytes_) {
    const size_t count = std::min(transit_chunk_bytes_, slot_bytes - offset);
    TransitHalf& half = transit_[half_index];

    // 1. Park the victim's chunk on the device. Must not start until the
    //    previous user of this half has finished reading it out (`landed`).
    device_.StreamWaitEvent(stage_stream_, half.landed);
    device_.MemcpyAsync(half.device_scratch, transit_chunk_bytes_, device_base + offset, count,
                        MemcpyKind::kDeviceToDevice, stage_stream_);
    device_.RecordEvent(half.parked, stage_stream_);

    // 2. Promote the incoming chunk into the now-backed-up device slot.
    device_.StreamWaitEvent(h2d_stream_, half.parked);
    device_.MemcpyAsync(device_base + offset, slot_bytes - offset, host_base + offset, count,
                        MemcpyKind::kHostToDevice, h2d_stream_);
    device_.RecordEvent(half.promoted, h2d_stream_);

    // 3. Land the victim in the host slot the promotee just vacated. Reading
    //    and writing the same host bytes is why this waits on `promoted`.
    device_.StreamWaitEvent(d2h_stream_, half.promoted);
    device_.MemcpyAsync(host_base + offset, slot_bytes - offset, half.device_scratch, count,
                        MemcpyKind::kDeviceToHost, d2h_stream_);
    device_.RecordEvent(half.landed, d2h_stream_);

    half_index = (half_index + 1) % kTransitHalves;
    ++stats_.swap_chunks;
  }
  stats_.host_to_device_bytes += slot_bytes;
  stats_.device_to_host_bytes += slot_bytes;
  stats_.device_to_device_bytes += slot_bytes;
  ++stats_.swaps;
}

void ExclusiveExpertManager::ExecutePlan(const LayerSwapPlan& plan, DeviceStream compute_stream,
                                         DeviceEvent compute_done) {
  RefuseIfPoisoned();
  DSV4_REQUIRE(ingested_, "the exclusive partition must be ingested before a layer is prepared");
  ++layer_epoch_;
  ++stats_.layer_requests;
  stats_.slot_hits += static_cast<uint64_t>(plan.hit_count);
  stats_.slot_misses += static_cast<uint64_t>(plan.miss_count);

  // Touch every slot this layer reads, hits included, so the LRU order
  // reflects use rather than admission.
  for (int32_t index = 0; index < plan.count; ++index) {
    slot_last_touch_[static_cast<size_t>(plan.device_slots[index])] = ++touch_clock_;
    ++access_count_[static_cast<size_t>(plan.incoming_keys[index])];
  }

  if (plan.miss_count == 0) {
    return;  // nothing to transfer: no events, no gate, no stream touched
  }

  DSV4_REQUIRE(compute_done != nullptr,
               "ExecutePlan needs the compute-boundary event when the layer has a miss ("
                   << plan.miss_count
                   << " here): without it nothing orders this batch's overwrite after the previous layer's GEMM");

  // Two cross-resource hazards, both closed with events rather than a host
  // synchronization: the previous layer's expert GEMM may still be reading a
  // device slot this batch overwrites, and the previous batch's D2H may still
  // be writing a host slot this batch reads.
  device_.StreamWaitEvent(stage_stream_, compute_done);
  device_.StreamWaitEvent(h2d_stream_, compute_done);
  if (!first_batch_) {
    device_.StreamWaitEvent(h2d_stream_, d2h_batch_done_);
  }

  // Enqueue every exchange of the batch, publishing each expert's residency
  // only once its transfer is on the streams. A throw mid-batch poisons the
  // partition: an in-place exchange whose first chunks already landed cannot be
  // retried or rolled back.
  try {
    for (int32_t index = 0; index < plan.count; ++index) {
      if (plan.was_hit[index]) {
        continue;
      }
      const int32_t slot = plan.device_slots[index];
      const int32_t incoming_key = plan.incoming_keys[index];
      const int32_t victim_key = plan.victim_keys[index];
      const int32_t host_slot = host_slot_of_[static_cast<size_t>(incoming_key)];
      DSV4_REQUIRE(host_slot != kNoSlot,
                   "expert key " << incoming_key << " lost its host slot between plan and execute");

      ExchangeSlot(slot, host_slot);

      // The promotee takes the device slot; the victim takes the host slot the
      // promotee just vacated. Exactly one location per expert, always.
      device_slot_of_[static_cast<size_t>(incoming_key)] = slot;
      host_slot_of_[static_cast<size_t>(incoming_key)] = kNoSlot;
      device_slot_of_[static_cast<size_t>(victim_key)] = kNoSlot;
      host_slot_of_[static_cast<size_t>(victim_key)] = host_slot;
      slot_owner_[static_cast<size_t>(slot)] = incoming_key;
    }
  } catch (...) {
    poisoned_ = true;
    throw;
  }

  // One gate for the whole batch: the compute stream waits on the promotions,
  // and the next batch's H2D waits on this batch's landings.
  device_.RecordEvent(h2d_batch_done_, h2d_stream_);
  device_.RecordEvent(d2h_batch_done_, d2h_stream_);
  if (compute_stream != nullptr) {
    device_.StreamWaitEvent(compute_stream, h2d_batch_done_);
  }
  first_batch_ = false;
}

LayerSwapPlan ExclusiveExpertManager::PrepareLayer(int32_t layer, const int32_t* expert_ids, int32_t count,
                                                   DeviceStream compute_stream, DeviceEvent compute_done) {
  LayerSwapPlan plan = PlanLayer(layer, expert_ids, count);
  ExecutePlan(plan, compute_stream, compute_done);
  return plan;
}

// ---------------------------------------------------------------------------
// Invariants and reporting
// ---------------------------------------------------------------------------

void ExclusiveExpertManager::ValidateResidency() const {
  int64_t device_residents = 0;
  int64_t host_residents = 0;
  for (int64_t key = 0; key < options_.routed_coverage; ++key) {
    const int32_t in_device = device_slot_of_[static_cast<size_t>(key)];
    const int32_t in_host = host_slot_of_[static_cast<size_t>(key)];
    DSV4_REQUIRE(!(in_device != kNoSlot && in_host != kNoSlot),
                 "exclusive overlap: expert key " << key << " is resident in device slot " << in_device
                                                  << " and host slot " << in_host << " at the same time");
    DSV4_REQUIRE(in_device != kNoSlot || in_host != kNoSlot,
                 "exclusive coverage hole: expert key " << key << " is resident nowhere");
    if (in_device != kNoSlot) {
      ++device_residents;
    } else {
      ++host_residents;
    }
  }
  DSV4_REQUIRE(device_residents == options_.device_slots,
               "device occupancy changed: " << device_residents << " residents for " << options_.device_slots
                                            << " slots");
  DSV4_REQUIRE(host_residents == host_slot_count_,
               "host occupancy changed: " << host_residents << " residents for " << host_slot_count_ << " slots");

  // No two experts may alias one slot, on either side.
  std::vector<uint8_t> seen_device(static_cast<size_t>(options_.device_slots), 0u);
  std::vector<uint8_t> seen_host(static_cast<size_t>(std::max<int64_t>(host_slot_count_, 1)), 0u);
  for (int64_t key = 0; key < options_.routed_coverage; ++key) {
    const int32_t in_device = device_slot_of_[static_cast<size_t>(key)];
    if (in_device != kNoSlot) {
      DSV4_REQUIRE(seen_device[static_cast<size_t>(in_device)] == 0u,
                   "multiple experts alias device slot " << in_device);
      seen_device[static_cast<size_t>(in_device)] = 1u;
      DSV4_REQUIRE(slot_owner_[static_cast<size_t>(in_device)] == static_cast<int32_t>(key),
                   "device slot " << in_device << " claims owner " << slot_owner_[static_cast<size_t>(in_device)]
                                  << " but key " << key << " points at it");
      continue;
    }
    const int32_t in_host = host_slot_of_[static_cast<size_t>(key)];
    DSV4_REQUIRE(seen_host[static_cast<size_t>(in_host)] == 0u, "multiple experts alias host slot " << in_host);
    seen_host[static_cast<size_t>(in_host)] = 1u;
  }
  DSV4_REQUIRE(stats_.runtime_disk_read_attempts == 0,
               stats_.runtime_disk_read_attempts << " runtime disk reads were attempted");
}

std::vector<const void*> ExclusiveExpertManager::Fingerprint() const {
  std::vector<const void*> pointers;
  pointers.reserve(host_blocks_.size() + kTransitHalves + 2);
  pointers.push_back(device_arena_);
  for (void* block : host_blocks_) {
    pointers.push_back(block);
  }
  for (const TransitHalf& half : transit_) {
    pointers.push_back(half.device_scratch);
  }
  pointers.push_back(host_transit_scratch_);
  return pointers;
}

std::string ExclusiveExpertManager::DescribeHierarchy() const {
  const double mib = 1024.0 * 1024.0;
  const double gib = mib * 1024.0;
  std::ostringstream out;
  out << std::fixed << std::setprecision(2);
  out << "exclusive routed-expert hierarchy (" << device_.backend_name() << " backend)\n";
  out << "  coverage      " << options_.routed_coverage << " experts = " << options_.num_layers << " layers x "
      << options_.num_experts << " routed";
  if (options_.routed_coverage != options_.num_layers * options_.num_experts) {
    out << "  [BRING-UP SUBSET of " << (options_.num_layers * options_.num_experts) << "]";
  }
  out << "\n";
  out << "  slot          " << layout_.slot_num_bytes() << " bytes (" << layout_.slot_num_bytes() / mib
      << " MiB)\n";
  out << "  device  (K)   " << options_.device_slots << " slots = " << device_arena_bytes_ / gib << " GiB HBM\n";
  out << "  host  (N-K)   " << host_slot_count_ << " slots = " << host_arena_bytes_ / gib << " GiB pinned, in "
      << host_blocks_.size() << " block(s) of up to " << host_slots_per_block_ << " slots\n";
  out << "  transit       " << kTransitHalves << " x " << transit_chunk_bytes_ / mib << " MiB device + 1 x "
      << transit_chunk_bytes_ / mib << " MiB pinned host (ingestion)\n";
  out << "  streams       h2d, d2h, stage (3) + " << (kTransitHalves * 3 + 2) << " events\n";
  return out.str();
}

}  // namespace dsv4
}  // namespace vllm_ascend
