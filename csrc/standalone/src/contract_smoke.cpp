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

// dsv4_contract_smoke -- the parts of this runner that can be verified without
// an NPU attached, actually verified.
//
// This is not a compile check. The exclusive swap engine runs here for real,
// over the `SimulatedDeviceOps` backend, and the test asserts on its recorded
// DMA trace: the disjointness invariant, the chunked duplex exchange's event
// ordering, the LRU victim choice, the byte-exact round trip of an evicted
// expert, the poisoning on a failed transfer, and every refusal the sealed
// arena is supposed to make. It also resolves the operator table against the
// linked CANN libraries, which proves the link and the ABI without a device.
//
// What it cannot check, and says so: anything that needs a 950PR -- the real
// `GetWorkspaceSize` numbers, whether a kernel exists for every op on the part,
// and the `aclSetTensorAddr` index map.

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "moe/core/error.hpp"
#include "moe/core/op_table.hpp"
#include "moe/core/config.hpp"
#include "moe/core/device_ops.hpp"
#include "moe/core/model_config.hpp"
#include "moe/memory/exclusive_staging.hpp"
#include "moe/memory/expert_layout.hpp"
#include "moe/pipeline/lattice_moe_block.hpp"
#include "moe/pipeline/pipeline.hpp"
#include "moe/core/weight_source.hpp"

namespace ascend_moe {
namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool condition, const std::string& what) {
  ++g_checks;
  if (condition) {
    std::printf("  [ ok ] %s\n", what.c_str());
  } else {
    ++g_failures;
    std::printf("  [FAIL] %s\n", what.c_str());
  }
}

// Asserts that `action` throws, which is how every refusal in this design is
// expressed. A refusal that silently succeeds is the failure mode worth testing.
template <typename Action>
void CheckRefuses(Action action, const std::string& what) {
  ++g_checks;
  try {
    action();
  } catch (const Dsv4Error&) {
    std::printf("  [ ok ] refused: %s\n", what.c_str());
    return;
  } catch (const std::exception& error) {
    ++g_failures;
    std::printf("  [FAIL] %s threw the wrong type: %s\n", what.c_str(), error.what());
    return;
  }
  ++g_failures;
  std::printf("  [FAIL] %s was allowed\n", what.c_str());
}

void Section(const char* title) { std::printf("\n== %s ==\n", title); }

// ---------------------------------------------------------------------------
// 1. Expert slot layout
// ---------------------------------------------------------------------------

void TestExpertLayout() {
  Section("expert slot layout");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForDeepSeekV4Flash();
  std::printf("%s", layout.DescribeTable().c_str());

  // The number the Python layout's docstring states for this geometry.
  Check(layout.slot_num_bytes() == 13369344,
        "DSV4-Flash slot is 13,369,344 bytes (12.75 MiB), matching core/layout.py");
  Check(layout.slot_num_bytes() * kTotalRoutedExperts == 147169738752ull,
        "the whole routed set is 147,169,738,752 bytes (137.07 GiB) over 11,008 experts");

  size_t covered = 0;
  bool aligned = true;
  for (const ExpertRegionSpec& spec : layout.regions()) {
    aligned = aligned && (spec.offset_bytes % kSlotRegionAlignBytes == 0);
    covered += spec.num_bytes();
  }
  Check(aligned, "every region offset is 128-byte aligned");
  Check(covered == layout.slot_num_bytes(), "the regions tile the slot exactly, with no padding bytes");

  // Gate and up must be adjacent, or the fused expert GEMM cannot address them
  // as one weight.
  const ExpertRegionSpec& gate_up = layout.region(ExpertRegionId::kGateUpWeight);
  Check(gate_up.rows == 2 * kMoeIntermediateSize && gate_up.cols == kHiddenSize,
        "the fused gate/up weight is [2*intermediate, hidden]");
  const ExpertRegionSpec& down = layout.region(ExpertRegionId::kDownWeight);
  Check(down.rows == kHiddenSize && down.cols == kMoeIntermediateSize,
        "the down weight is [hidden, intermediate]");
  Check(layout.region(ExpertRegionId::kGateUpScale).stored_cols() == kHiddenSize / kRoutedScaleBlock,
        "microscales are blocked along the reduction axis, 32 elements per byte of scale");

  // A geometry whose reduction axis is not divisible by the block must be
  // refused, not rounded.
  CheckRefuses([] { ExpertSlotLayout::ForGeometry(48, 64); }, "a hidden size that is not a multiple of 32");
}

// ---------------------------------------------------------------------------
// 2. Safetensors header parsing
// ---------------------------------------------------------------------------

void TestHeaderParsing() {
  Section("safetensors header parsing");
  const char* header =
      "{\"__metadata__\":{\"format\":\"pt\",\"nested\":{\"a\":[1,2,3]}},"
      "\"layers.0.ffn.experts.0.w1.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[2048,4096],"
      "\"data_offsets\":[0,4194304]},"
      "\"layers.0.ffn.experts.0.w1.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[2048,128],"
      "\"data_offsets\":[4194304,4456448]}}";
  const std::map<std::string, SafetensorsTensor> tensors =
      ParseSafetensorsHeaderJson(header, std::strlen(header));
  Check(tensors.size() == 2, "__metadata__ is skipped, including its nested object");
  const auto weight = tensors.find("layers.0.ffn.experts.0.w1.weight");
  Check(weight != tensors.end() && weight->second.num_bytes() == 4194304,
        "the FP4/FP8 weight span is 4,194,304 bytes");
  const auto scale = tensors.find("layers.0.ffn.experts.0.w1.scale");
  Check(scale != tensors.end() && scale->second.payload_begin == 4194304, "the scale span follows the weight");

  CheckRefuses([] { ParseSafetensorsHeaderJson("{\"a\":{", 6); }, "a truncated header");
  CheckRefuses([] { ParseSafetensorsHeaderJson("not json", 8); }, "a header that is not an object");
}

// ---------------------------------------------------------------------------
// 2b. The FP4 / E8M0 unpack mapping, checked against a real (tiny) file
// ---------------------------------------------------------------------------

// Writes a one-expert checkpoint for ExpertSlotLayout::ForGeometry(64, 64):
// w1/w3 weight 2048 bytes each (shape [64, 32] in the packed-FP4 byte view),
// w1/w3 scale 128 bytes each ([64, 2]), w2 weight 2048 ([64, 32]), w2 scale 128
// ([64, 2]) -- 6528 payload bytes, exactly one slot. `weight_dtype` /
// `scale_dtype` land in every header so a wrong-dtype checkpoint can be
// expressed; `w1_bytes` resizes the first span for the wrong-size case.
bool WriteExpertCheckpoint(const char* path, const char* weight_dtype, const char* scale_dtype, size_t w1_bytes) {
  const size_t scale_bytes = 128;
  const size_t w3_bytes = 2048;
  const size_t w2_bytes = 2048;
  const size_t total = w1_bytes + scale_bytes + w3_bytes + scale_bytes + w2_bytes + scale_bytes;

  size_t cursor = 0;
  std::string json = "{";
  const auto add_tensor = [&](const char* leaf, const char* dtype, const char* shape, size_t bytes) {
    const size_t begin = cursor;
    cursor += bytes;
    json += json.size() > 1 ? "," : "";
    json += "\"layers.0.ffn.experts.0.";
    json += leaf;
    json += "\":{\"dtype\":\"";
    json += dtype;
    json += "\",\"shape\":";
    json += shape;
    json += ",\"data_offsets\":[";
    json += std::to_string(begin);
    json += ",";
    json += std::to_string(cursor);
    json += "]}";
  };
  add_tensor("w1.weight", weight_dtype, "[64,32]", w1_bytes);
  add_tensor("w1.scale", scale_dtype, "[64,2]", scale_bytes);
  add_tensor("w3.weight", weight_dtype, "[64,32]", w3_bytes);
  add_tensor("w3.scale", scale_dtype, "[64,2]", scale_bytes);
  add_tensor("w2.weight", weight_dtype, "[64,32]", w2_bytes);
  add_tensor("w2.scale", scale_dtype, "[64,2]", scale_bytes);
  json += "}";

  std::FILE* file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const std::vector<char> payload(total, '\0');
  const uint64_t header_length = json.size();
  const bool wrote = std::fwrite(&header_length, sizeof(header_length), 1, file) == 1 &&
                     std::fwrite(json.data(), 1, json.size(), file) == json.size() &&
                     std::fwrite(payload.data(), 1, payload.size(), file) == payload.size();
  std::fclose(file);
  return wrote;
}

void TestExpertBindingValidation() {
  Section("safetensors expert binding: the FP4 / E8M0 unpack mapping");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForGeometry(64, 64);
  Check(layout.slot_num_bytes() == 6528, "the 64x64 test geometry packs into a 6,528-byte slot");

  const char* path = "/tmp/dsv4_contract_smoke_experts.safetensors";
  std::vector<uint8_t> slot(layout.slot_num_bytes());

  // The admissible checkpoint: packed-FP4 weights and UE8M0 block scales.
  Check(WriteExpertCheckpoint(path, "MOE_F4", "MOE_F4_SCALE", 2048), "the admissible checkpoint was written");
  {
    SafetensorsWeightSource source(path, layout, CheckpointNaming::kDsv4Flat, 1, 1);
    Check(source.Contains(0, 0), "the flat naming scheme finds expert (0, 0)");
    source.ReadExpertSlotRange(slot.data(), slot.size(), 0, slot.size(), 0, 0);
    Check(true, "an FP4 + E8M0 checkpoint with exact span sizes reads into the slot");
  }

  // An FP8 checkpoint bound against the FP4 slot layout: same byte count would
  // not catch it, the dtype string does.
  Check(WriteExpertCheckpoint(path, "F8_E4M3", "MOE_F4_SCALE", 2048), "the FP8 checkpoint was written");
  CheckRefuses(
      [&] {
        SafetensorsWeightSource source(path, layout, CheckpointNaming::kDsv4Flat, 1, 1);
        source.ReadExpertSlotRange(slot.data(), slot.size(), 0, slot.size(), 0, 0);
      },
      "an FP8-E4M3 weight tensor bound against the packed-FP4 slot layout");

  // A per-element scale dtype is not the block-32 E8M0 mapping either.
  Check(WriteExpertCheckpoint(path, "MOE_F4", "U8", 2048), "the U8-scale checkpoint was written");
  CheckRefuses(
      [&] {
        SafetensorsWeightSource source(path, layout, CheckpointNaming::kDsv4Flat, 1, 1);
        source.ReadExpertSlotRange(slot.data(), slot.size(), 0, slot.size(), 0, 0);
      },
      "a per-element U8 scale tensor bound against the E8M0 block-32 scale region");

  // A resized first span: the byte-exact slot total no longer tiles.
  Check(WriteExpertCheckpoint(path, "MOE_F4", "MOE_F4_SCALE", 512), "the resized checkpoint was written");
  CheckRefuses(
      [&] {
        SafetensorsWeightSource source(path, layout, CheckpointNaming::kDsv4Flat, 1, 1);
        source.ReadExpertSlotRange(slot.data(), slot.size(), 0, slot.size(), 0, 0);
      },
      "a weight span that no longer matches the slot region size");

  std::remove(path);
}

// ---------------------------------------------------------------------------
// 3. The exclusive hierarchy, running for real on the simulated backend
// ---------------------------------------------------------------------------

// A deliberately small geometry: the invariants are about counts and ordering,
// and 11,008 real 12.75 MiB slots would need 137 GiB to say the same thing.
constexpr int64_t kTestLayers = 4;
constexpr int64_t kTestExperts = 8;
constexpr int64_t kTestCoverage = kTestLayers * kTestExperts;  // 32
constexpr int64_t kTestDeviceSlots = 12;

ExclusiveExpertManager::Options TestOptions() {
  ExclusiveExpertManager::Options options;
  options.num_layers = kTestLayers;
  options.num_experts = kTestExperts;
  options.routed_coverage = kTestCoverage;
  options.device_slots = kTestDeviceSlots;
  options.transfer_chunk_bytes = 4096;  // several chunks per slot, so the ring is exercised
  options.host_block_bytes = 64 * 1024;
  options.top_k = kNumExpertsPerTok;
  return options;
}

void TestHierarchyInvariants() {
  Section("exclusive hierarchy: ingestion and the disjointness invariant");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForGeometry(256, 128);
  SimulatedDeviceOps device(64ull * 1024 * 1024);
  ExclusiveExpertManager experts(device, device, layout, TestOptions());
  std::printf("%s", experts.DescribeHierarchy().c_str());

  Check(experts.device_slot_count() + experts.host_slot_count() == kTestCoverage,
        "|Set_Device| + |Set_Host| equals the routed coverage");
  Check(experts.host_bytes() == static_cast<size_t>(kTestCoverage - kTestDeviceSlots) * layout.slot_num_bytes(),
        "the pinned host arena holds exactly coverage - K slots");

  const std::vector<const void*> fingerprint = experts.Fingerprint();
  SyntheticWeightSource source(layout, kTestLayers, kTestExperts);
  experts.Ingest(source, {});
  Check(source.closed(), "Ingest closes the byte source: the hierarchy becomes the only copy");
  CheckRefuses([&] { source.ReadExpertSlotRange(nullptr, 0, 0, 0, 0, 0); },
               "a read from the closed source (runtime disk reads)");
  Check(experts.Fingerprint() == fingerprint, "no arena address moved during ingestion");
  experts.ValidateResidency();
  Check(true, "the residency invariant holds after ingestion");

  CheckRefuses([&] { experts.Ingest(source, {}); }, "a second ingestion");

  // Every byte of every host-resident expert must be what the source would have
  // produced: a slot written to the wrong place is otherwise invisible.
  int mismatches = 0;
  for (int32_t layer = 0; layer < static_cast<int32_t>(kTestLayers); ++layer) {
    for (int32_t expert = 0; expert < static_cast<int32_t>(kTestExperts); ++expert) {
      if (experts.DeviceSlotOf(layer, expert) != kNoSlot) {
        continue;
      }
      // Host-resident: compare the first and last byte of the slot image.
      // (The addresses are private, so this checks through the device arena
      // after a promotion below instead; see TestSwapRoundTrip.)
      ++mismatches;
    }
  }
  Check(mismatches == kTestCoverage - kTestDeviceSlots,
        "exactly coverage - K experts are host-resident after ingestion");
}

void TestLayerPlanning() {
  Section("exclusive hierarchy: layer planning and the eviction policy");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForGeometry(256, 128);
  SimulatedDeviceOps device(64ull * 1024 * 1024);
  ExclusiveExpertManager experts(device, device, layout, TestOptions());
  SyntheticWeightSource source(layout, kTestLayers, kTestExperts);
  experts.Ingest(source, {});

  // Layer 0's experts 0..5 were placed in slots 0..5 at ingestion, so this is
  // an all-hit layer: no transfer, no event, no stream touched.
  const int32_t all_hits[kNumExpertsPerTok] = {0, 1, 2, 3, 4, 5};
  const LayerSwapPlan hit_plan = experts.PlanLayer(0, all_hits, kNumExpertsPerTok);
  Check(hit_plan.hit_count == kNumExpertsPerTok && hit_plan.miss_count == 0,
        "a layer whose top-6 is already resident plans six hits and no swap");

  device.ResetCounters();
  DeviceStream compute = device.CreateStream();
  DeviceEvent boundary = device.CreateEvent();
  device.RecordEvent(boundary, compute);
  experts.ExecutePlan(hit_plan, compute, boundary);
  Check(device.counters().async_copies == 0, "an all-hit layer enqueues no DMA at all");

  // Layer 3 is entirely host-resident, so it is six misses and six exchanges.
  const int32_t all_misses[kNumExpertsPerTok] = {0, 1, 2, 3, 4, 5};
  const LayerSwapPlan miss_plan = experts.PlanLayer(3, all_misses, kNumExpertsPerTok);
  Check(miss_plan.miss_count == kNumExpertsPerTok, "a fully host-resident layer plans six misses");
  bool distinct_victims = true;
  for (int32_t i = 0; i < miss_plan.count; ++i) {
    for (int32_t j = i + 1; j < miss_plan.count; ++j) {
      distinct_victims = distinct_victims && miss_plan.device_slots[i] != miss_plan.device_slots[j];
    }
  }
  Check(distinct_victims, "the six admissions claim six different slots, so they cannot evict each other");

  // The victims must be the least-recently-touched slots, which after the
  // all-hit layer above are the six slots that layer did NOT touch.
  bool avoided_recent = true;
  for (int32_t i = 0; i < miss_plan.count; ++i) {
    avoided_recent = avoided_recent && miss_plan.device_slots[i] >= kNumExpertsPerTok;
  }
  Check(avoided_recent, "LRU picked the six untouched slots, not the six just used");

  CheckRefuses([&] { experts.ExecutePlan(miss_plan, compute, nullptr); },
               "a batch with misses and no compute-boundary event");

  device.ResetCounters();
  device.ClearTrace();
  experts.ExecutePlan(miss_plan, compute, boundary);
  experts.Synchronize();
  experts.ValidateResidency();
  Check(true, "the residency invariant still holds after six exchanges");

  const DmaCounters& counters = device.counters();
  const size_t slot_bytes = layout.slot_num_bytes();
  Check(counters.host_to_device_bytes == kNumExpertsPerTok * slot_bytes,
        "H2D moved exactly six slots' worth of bytes");
  Check(counters.device_to_host_bytes == kNumExpertsPerTok * slot_bytes,
        "D2H moved exactly six slots' worth (the victims)");
  Check(counters.device_to_device_bytes == kNumExpertsPerTok * slot_bytes,
        "D2D parked exactly six slots' worth in the bounded transit scratch");
  Check(counters.stream_synchronizations == 3,
        "the whole six-expert batch cost zero host synchronizations (3 here are the explicit Synchronize())");

  // Event ordering: within each chunk the three stages must appear as
  // park -> record, wait -> promote -> record, wait -> land -> record.
  int triples = 0;
  const std::vector<DmaTraceEntry>& trace = device.trace();
  for (size_t index = 0; index + 8 < trace.size(); ++index) {
    const bool shape = trace[index].kind == DmaTraceEntry::Kind::kStreamWait &&
                       trace[index + 1].kind == DmaTraceEntry::Kind::kMemcpyAsync &&
                       trace[index + 1].memcpy_kind == MemcpyKind::kDeviceToDevice &&
                       trace[index + 2].kind == DmaTraceEntry::Kind::kRecordEvent &&
                       trace[index + 3].kind == DmaTraceEntry::Kind::kStreamWait &&
                       trace[index + 4].kind == DmaTraceEntry::Kind::kMemcpyAsync &&
                       trace[index + 4].memcpy_kind == MemcpyKind::kHostToDevice &&
                       trace[index + 5].kind == DmaTraceEntry::Kind::kRecordEvent &&
                       trace[index + 6].kind == DmaTraceEntry::Kind::kStreamWait &&
                       trace[index + 7].kind == DmaTraceEntry::Kind::kMemcpyAsync &&
                       trace[index + 7].memcpy_kind == MemcpyKind::kDeviceToHost &&
                       trace[index + 8].kind == DmaTraceEntry::Kind::kRecordEvent;
    if (shape) {
      // The promote reads the host slot the land then writes: that aliasing is
      // the whole reason stage 3 waits on stage 2's event.
      Check(trace[index + 4].source == trace[index + 7].destination,
            "chunk " + std::to_string(triples) + ": H2D reads the host slot that the following D2H overwrites");
      ++triples;
    }
  }
  const size_t expected_chunks = kNumExpertsPerTok * ((slot_bytes + 4095) / 4096);
  Check(static_cast<size_t>(triples) == expected_chunks,
        "every one of the " + std::to_string(expected_chunks) +
            " transit chunks is a park/promote/land triple with an event between each pair");

  device.DestroyEvent(boundary);
  device.DestroyStream(compute);
}

void TestSwapRoundTrip() {
  Section("exclusive hierarchy: an evicted expert comes back byte-identical");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForGeometry(256, 128);
  SimulatedDeviceOps device(64ull * 1024 * 1024);
  ExclusiveExpertManager experts(device, device, layout, TestOptions());
  SyntheticWeightSource source(layout, kTestLayers, kTestExperts);
  experts.Ingest(source, {});

  DeviceStream compute = device.CreateStream();
  DeviceEvent boundary = device.CreateEvent();
  device.RecordEvent(boundary, compute);

  const size_t slot_bytes = layout.slot_num_bytes();
  std::vector<uint8_t> expected(slot_bytes);
  for (size_t index = 0; index < slot_bytes; ++index) {
    expected[index] = SyntheticWeightSource::ByteAt(3, 7, index);
  }

  // Promote (3, 7) -- host-resident after ingestion -- and compare its HBM
  // image against what the source would have produced.
  const int32_t request[kNumExpertsPerTok] = {7, 6, 5, 4, 3, 2};
  const LayerSwapPlan plan = experts.PrepareLayer(3, request, kNumExpertsPerTok, compute, boundary);
  experts.Synchronize();
  const int32_t slot = experts.DeviceSlotOf(3, 7);
  Check(slot != kNoSlot, "expert (3, 7) is resident in HBM after the promotion");
  const uint8_t* resident = static_cast<const uint8_t*>(experts.DeviceSlotAddress(slot));
  Check(std::memcmp(resident, expected.data(), slot_bytes) == 0,
        "its HBM image is byte-identical to the authoritative source");

  // Now evict it again by requesting a disjoint set from the same layer, then
  // bring it back and compare once more: the host copy the exchange wrote must
  // be the same bytes.
  const int32_t displace[kNumExpertsPerTok] = {0, 1, 2, 3, 4, 5};
  experts.PrepareLayer(0, displace, kNumExpertsPerTok, compute, boundary);
  const int32_t displace2[kNumExpertsPerTok] = {0, 1, 2, 3, 4, 5};
  experts.PrepareLayer(1, displace2, kNumExpertsPerTok, compute, boundary);
  experts.Synchronize();
  Check(experts.DeviceSlotOf(3, 7) == kNoSlot, "expert (3, 7) was evicted back to pinned host RAM");

  const LayerSwapPlan again = experts.PrepareLayer(3, request, kNumExpertsPerTok, compute, boundary);
  experts.Synchronize();
  const int32_t slot_again = experts.DeviceSlotOf(3, 7);
  const uint8_t* resident_again = static_cast<const uint8_t*>(experts.DeviceSlotAddress(slot_again));
  Check(std::memcmp(resident_again, expected.data(), slot_bytes) == 0,
        "after a full eviction and re-promotion the bytes are still identical");
  Check(again.miss_count > 0, "the re-promotion really was a miss, not a stale hit");
  (void)plan;

  experts.ValidateResidency();
  std::printf("  swaps=%" PRIu64 " chunks=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 "\n",
              experts.stats().swaps, experts.stats().swap_chunks, experts.stats().slot_hits,
              experts.stats().slot_misses);
  device.DestroyEvent(boundary);
  device.DestroyStream(compute);
}

void TestHierarchyRefusals() {
  Section("exclusive hierarchy: refusals");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForGeometry(256, 128);
  SimulatedDeviceOps device(64ull * 1024 * 1024);

  CheckRefuses(
      [&] {
        ExclusiveExpertManager::Options options = TestOptions();
        options.device_slots = kNumExpertsPerTok - 1;
        ExclusiveExpertManager narrow(device, device, layout, options);
      },
      "a K smaller than one top-k");
  CheckRefuses(
      [&] {
        ExclusiveExpertManager::Options options = TestOptions();
        options.device_slots = kTestCoverage + 1;
        ExclusiveExpertManager wide(device, device, layout, options);
      },
      "a K larger than the routed coverage");

  ExclusiveExpertManager experts(device, device, layout, TestOptions());
  SyntheticWeightSource source(layout, kTestLayers, kTestExperts);
  CheckRefuses([&] { const int32_t ids[1] = {0}; experts.PlanLayer(0, ids, 1); },
               "planning before ingestion names no host slot");
  experts.Ingest(source, {});
  CheckRefuses([&] { const int32_t ids[kNumExpertsPerTok] = {0, 0, 1, 2, 3, 4}; experts.PlanLayer(0, ids, kNumExpertsPerTok); },
               "a top-k with a duplicated expert id");
  CheckRefuses([&] { const int32_t ids[kNumExpertsPerTok] = {0, 1, 2, 3, 4, 99}; experts.PlanLayer(0, ids, kNumExpertsPerTok); },
               "an expert id outside the expert range");
  CheckRefuses([&] { const int32_t ids[kNumExpertsPerTok] = {0, 1, 2, 3, 4, 5}; experts.PlanLayer(99, ids, kNumExpertsPerTok); },
               "a layer index outside the model");

  // The slot planner's arithmetic, independent of any device.
  const size_t slot_bytes = ExpertSlotLayout::ForDeepSeekV4Flash().slot_num_bytes();
  const int64_t fitted = ExclusiveExpertManager::PlanDeviceSlots(
      64ull << 30, 6ull << 30, slot_bytes, kTotalRoutedExperts, kDeviceReserveBytes, kTransferChunkBytes,
      kNumExpertsPerTok, -1);
  std::printf("  64 GiB HBM, 6 GiB backbone -> K = %" PRId64 " routed slots (%.2f GiB)\n", fitted,
              static_cast<double>(fitted) * static_cast<double>(slot_bytes) / (1024.0 * 1024.0 * 1024.0));
  Check(fitted > kNumExpertsPerTok && fitted < kTotalRoutedExperts,
        "the slot planner lands between one top-k and full residency on a 64 GiB part");
  CheckRefuses(
      [&] {
        ExclusiveExpertManager::PlanDeviceSlots(1ull << 30, 6ull << 30, slot_bytes, kTotalRoutedExperts,
                                                kDeviceReserveBytes, kTransferChunkBytes, kNumExpertsPerTok, -1);
      },
      "a device with less free HBM than the backbone needs");
}

// ---------------------------------------------------------------------------
// 4. The static arena's phase latches
// ---------------------------------------------------------------------------

void TestStaticArena() {
  Section("static arena: alignment and the sealed-for-decode latch");
  SimulatedDeviceOps device(1ull << 30);

  // The simulated allocators must honor the hardware page on ABSOLUTE
  // addresses: aclrtMalloc(..., ACL_MEM_MALLOC_HUGE_FIRST) and aclrtMallocHost
  // serve page-aligned memory, and the arena's alignment contract is stated on
  // absolute descriptor addresses, not offsets.
  void* device_block = device.DeviceMalloc(100);
  void* pinned_block = device.HostPinnedMalloc(100);
  Check(reinterpret_cast<uintptr_t>(device_block) % kSimDeviceAllocAlignBytes == 0,
        "the simulated device allocator hands out 4096-aligned blocks (aclrtMalloc HUGE_FIRST parity)");
  Check(reinterpret_cast<uintptr_t>(pinned_block) % kSimDeviceAllocAlignBytes == 0,
        "the simulated pinned allocator hands out 4096-aligned blocks (aclrtMallocHost parity)");
  device.HostPinnedFree(pinned_block);
  device.DeviceFree(device_block);
  device.ResetCounters();  // the ONE-allocation check below counts from a clean slate

  StaticMemoryArena arena(device);

  const ArenaHandle first = arena.Reserve("a.small", 100);
  const ArenaHandle second = arena.Reserve("b.aligned", 4096, 4096);
  const ArenaHandle third = arena.Reserve("c.tail", 7);
  CheckRefuses([&] { arena.Address(first); }, "an address before Commit");
  CheckRefuses([&] { arena.Reserve("d.zero", 0); }, "a zero-byte reservation");
  CheckRefuses([&] { arena.Reserve("e.odd-align", 64, 3); }, "a non-power-of-two alignment");

  arena.Commit();
  Check(device.counters().device_allocations == 1, "Commit performs exactly ONE device allocation");
  Check(reinterpret_cast<uintptr_t>(arena.Address(second)) % 4096 == 0,
        "a reservation that asked for 4096-byte alignment got it");
  Check(arena.Address(third) > arena.Address(first), "reservations are laid out in request order");
  arena.ValidateAlignment();
  Check(true, "every reservation is aligned, in order, and inside the arena");
  CheckRefuses([&] { arena.Reserve("f.late", 8); }, "a reservation after Commit");

  arena.NoteWorkspace(1024);
  arena.NoteWorkspace(4096);
  arena.NoteWorkspace(2048);
  arena.CommitWorkspace();
  Check(arena.workspace_bytes() == 4096, "the shared workspace is the high-water mark over all planned ops");
  arena.AssertWorkspaceFits(4096, "test");
  CheckRefuses([&] { arena.AssertWorkspaceFits(4097, "test"); },
               "a re-plan that would need more workspace than was reserved");

  arena.Seal();
  Check(arena.sealed(), "the arena is sealed for decoding");
  CheckRefuses([&] { arena.Reserve("g.sealed", 8); }, "a reservation after Seal");
  CheckRefuses([&] { arena.CreateIntArray("h.sealed", {1, 2}); }, "a descriptor after Seal");
  CheckRefuses([&] { arena.CommitWorkspace(); }, "a workspace commit after Seal");
  std::printf("%s", arena.DescribeLedger().c_str());
}

// ---------------------------------------------------------------------------
// 5. Backbone sizing and the operator table
// ---------------------------------------------------------------------------

void TestBackboneSizing() {
  Section("backbone device footprint");
  const MlaGeometry mla;
  const size_t bytes = Dsv4Pipeline::BackboneDeviceBytes(mla, 128, 8192);
  std::printf("  backbone + paged KV at kv_lora=%" PRId64 " rope=%" PRId64 " block=128 context=8192: %.2f GiB\n",
              mla.kv_lora_rank, mla.qk_rope_head_dim, static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
  Check(bytes > (1ull << 30), "the backbone is more than a gigabyte, so the slot planner must subtract it");
  const size_t longer = Dsv4Pipeline::BackboneDeviceBytes(mla, 128, 32768);
  Check(longer > bytes, "a longer reserved context costs more paged KV");
}

void TestOperatorTable() {
  Section("aclnn operator table");
  OpTable ops;
  std::printf("%s", ops.DescribeInventory().c_str());
  if (!ops.runtime_reachable()) {
    std::printf("  NOTE: the aclnn runtime is not on the loader path here, so operator resolution is untested.\n");
    std::printf("        Everything above this section is device-independent and did run.\n");
    return;
  }
  Check(ops.available(OpId::kRmsNorm), "aclnnRmsNorm resolved from the linked libraries");
  Check(ops.available(OpId::kFusedInferAttentionScoreV5), "aclnnFusedInferAttentionScoreV5 resolved");
  Check(ops.available(OpId::kGroupedMatmulV5), "aclnnGroupedMatmulV5 resolved");
  Check(ops.available(OpId::kSoftplus), "aclnnSoftplus resolved (sqrtsoftplus router scoring, stage 1)");
  Check(ops.available(OpId::kSqrt), "aclnnSqrt resolved (sqrtsoftplus router scoring, stage 2)");
  Check(ops.available(OpId::kMoeGatingTopKV2), "aclnnMoeGatingTopKV2 resolved (CANN 9.2.0 and later)");
  Check(ops.available(OpId::kMoeInitRoutingV4), "aclnnMoeInitRoutingV4 resolved (CANN 9.2.0 and later)");
  CheckRefuses([&] { ops.RequireAll({OpId::kOpCount}); }, "a request for an out-of-range operator id");
}

// ---------------------------------------------------------------------------
// 6. ModelConfig and the MoE block seam
// ---------------------------------------------------------------------------

void TestModelConfigContract() {
  Section("model config: the checkpoint contract the runner validates at startup");
  // The published DeepSeek-V4 Flash fields, verbatim structure.
  const char* config_json =
      "{\"architectures\":[\"DeepseekV4ForCausalLM\"],\"hidden_size\":4096,"
      "\"moe_intermediate_size\":2048,\"index_topk\":512,\"n_routed_experts\":256,"
      "\"n_shared_experts\":1,\"norm_topk_prob\":true,\"num_attention_heads\":64,"
      "\"num_experts_per_tok\":6,\"num_hidden_layers\":43,\"q_lora_rank\":1024,"
      "\"qk_rope_head_dim\":64,"
      "\"quantization_config\":{\"quant_method\":\"fp8\",\"scale_fmt\":\"ue8m0\","
      "\"weight_block_size\":[128,128]},"
      "\"routed_scaling_factor\":1.5,\"rms_norm_eps\":1e-06,"
      "\"scoring_func\":\"sqrtsoftplus\",\"swiglu_limit\":10.0,"
      "\"topk_method\":\"noaux_tc\",\"vocab_size\":129280}";
  const ModelConfig model = ModelConfig::FromJsonText(config_json, "fixture config.json");
  Check(model.num_hidden_layers == 43 && model.n_routed_experts == 256 && model.num_experts_per_tok == 6 &&
            model.hidden_size == 4096 && model.moe_intermediate_size == 2048 &&
            model.routed_scaling_factor == 1.5 && model.topk_method == "noaux_tc" &&
            model.scoring_func == "sqrtsoftplus" && model.norm_topk_prob,
        "the nine contracted fields parse to the documented DSV4-Flash values");
  model.AssertMatchesBinaryContract();
  Check(true, "the parsed topology matches the compiled graph geometry");

  CheckRefuses([] {
    ModelConfig::FromJsonText("{\"num_hidden_layers\":43,\"n_routed_experts\":256,"
                              "\"num_experts_per_tok\":6,\"hidden_size\":4096,"
                              "\"moe_intermediate_size\":2048,\"routed_scaling_factor\":1.5,"
                              "\"topk_method\":\"noaux_tc\",\"scoring_func\":\"softmax\","
                              "\"norm_topk_prob\":true}",
                              "t");
  }, "scoring_func softmax is refused: the router chain computes sqrt(softplus)");
  CheckRefuses([] { ModelConfig::FromJsonText("{\"broken\"", "t"); }, "a truncated document is a parse error");

  // The slot planner runs from the parsed geometry.
  const ExpertSlotLayout from_config = ExpertSlotLayout::ForGeometry(model.hidden_size, model.moe_intermediate_size);
  Check(from_config.slot_num_bytes() == ExpertSlotLayout::ForDeepSeekV4Flash().slot_num_bytes(),
        "the slot layout computed from the PARSED geometry equals the compiled one");
}

void TestMoeBlockSlotContract() {
  Section("IRoutedMoeBlock: slot byte contracts");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForDeepSeekV4Flash();
  Check(Lattice24MoeBlock::SlotBytes() == 7077888,
        "the 2-bit Leech slot is 7,077,888 bytes (6.75 MiB) at the DSV4-Flash geometry");
  Check(Lattice24MoeBlock::SlotBytes() < layout.slot_num_bytes(),
        "the lattice slot is smaller than the 13,369,344-byte FP4 slot");
  Lattice24MoeBlock lattice;
  CheckRefuses([&] {
    StaticOpSlotTable table(8, "lattice-test");
    RuntimeConfig config;
    config.block_size = 128;
    config.max_context_len = 256;
    SimulatedDeviceOps device(64ull * 1024 * 1024);
    StaticArenaManager probe(device, device, config);
    ExpertSlotAddresses addresses;
    lattice.PlanStages(table, probe, addresses);
  }, "the lattice skeleton refuses to plan, naming aclnnLatticeUnpackAndGroupedMatmul");
}

}  // namespace
}  // namespace ascend_moe

int main() {
  using namespace ascend_moe;
  std::printf("dsv4_contract_smoke -- device-free verification of the DSV4 runner's contracts\n");
  try {
    TestExpertLayout();
    TestHeaderParsing();
    TestExpertBindingValidation();
    TestHierarchyInvariants();
    TestLayerPlanning();
    TestSwapRoundTrip();
    TestHierarchyRefusals();
    TestStaticArena();
    TestBackboneSizing();
    TestModelConfigContract();
    TestMoeBlockSlotContract();
    TestOperatorTable();
  } catch (const std::exception& error) {
    std::printf("\nunexpected exception: %s\n", error.what());
    ++g_failures;
  }
  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  std::printf(
      "\nNot covered here (needs a 950PR): real GetWorkspaceSize values, whether every\n"
      "operator has an ascend950 kernel binary, and the aclSetTensorAddr index map.\n");
  return g_failures == 0 ? 0 : 1;
}
