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

// The two backend implementations of the IDeviceAllocator + IStreamEngine
// contract.
//
// The swap engine's correctness is a question about *ordering* -- which copy
// may overwrite which slot, and which event has to gate it -- and that
// question is answerable without an NPU. The simulated backend runs the
// identical ExclusiveExpertManager code over plain host memory and records
// the DMA trace, so the contract test verifies the disjointness invariant,
// the chunked duplex exchange and the eviction policy on a build machine
// with no device attached.
//
// Cost: one virtual call per DMA *request*, i.e. a few dozen per 43-layer
// decode step, against transfers measured in MiB. It is not on any per-byte
// path.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "moe/core/device_allocator.hpp"
#include "moe/core/device_types.hpp"
#include "moe/core/stream_engine.hpp"

namespace ascend_moe {

// ---------------------------------------------------------------------------
// Physical backend: aclrtMalloc / aclrtMallocHost / aclrtMemcpyAsync
// ---------------------------------------------------------------------------

class AclDeviceOps final : public IDeviceAllocator, public IStreamEngine {
 public:
  // Calls aclInit + aclrtSetDevice + aclrtCreateContext. Throws AclError when
  // no device is attached, which makes "device unavailable" an explicit outcome
  // rather than a silent fallback onto host memory.
  explicit AclDeviceOps(int32_t device_id);
  ~AclDeviceOps() override;

  const char* backend_name() const override { return "acl"; }

  void* DeviceMalloc(size_t bytes) override;
  void DeviceFree(void* pointer) override;
  void* HostPinnedMalloc(size_t bytes) override;
  void HostPinnedFree(void* pointer) override;
  void DeviceMemset(void* pointer, size_t capacity, int value, size_t count) override;
  uint64_t DeviceAllocationCount() const override { return counters_.device_allocations; }
  bool QueryDeviceMemory(size_t* free_bytes, size_t* total_bytes) override;

  DeviceStream CreateStream() override;
  void DestroyStream(DeviceStream stream) override;
  DeviceEvent CreateEvent() override;
  void DestroyEvent(DeviceEvent event) override;
  void RecordEvent(DeviceEvent event, DeviceStream stream) override;
  void StreamWaitEvent(DeviceStream stream, DeviceEvent event) override;
  void SynchronizeStream(DeviceStream stream) override;

  void MemcpyAsync(void* destination, size_t destination_capacity, const void* source, size_t count,
                   MemcpyKind kind, DeviceStream stream) override;
  void MemcpySync(void* destination, size_t destination_capacity, const void* source, size_t count,
                  MemcpyKind kind) override;

  const DmaCounters& counters() const { return counters_; }
  void ResetCounters() { counters_.Reset(); }
  const std::string& soc_name() const { return soc_name_; }

 private:
  void AccountCopy(MemcpyKind kind, size_t count);

  int32_t device_id_ = 0;
  void* context_ = nullptr;
  std::string soc_name_;
  DmaCounters counters_;
};

// ---------------------------------------------------------------------------
// Simulated backend: host memory, ordered trace, no attached device needed
// ---------------------------------------------------------------------------

struct DmaTraceEntry {
  enum class Kind { kMemcpyAsync, kMemcpySync, kRecordEvent, kStreamWait, kStreamSync };
  Kind kind = Kind::kMemcpyAsync;
  int32_t stream_id = -1;
  int32_t event_id = -1;
  MemcpyKind memcpy_kind = MemcpyKind::kHostToHost;
  const void* destination = nullptr;
  const void* source = nullptr;
  size_t count = 0;
};

class SimulatedDeviceOps final : public IDeviceAllocator, public IStreamEngine {
 public:
  // `trace_capacity` bounds the recorded DMA trace. It is reserved up front so
  // recording never allocates; once full, recording stops and `trace_overflow`
  // is set rather than the buffer growing.
  explicit SimulatedDeviceOps(size_t device_memory_bytes, size_t trace_capacity = 1u << 16);
  ~SimulatedDeviceOps() override;

  const char* backend_name() const override { return "simulated"; }

  void* DeviceMalloc(size_t bytes) override;
  void DeviceFree(void* pointer) override;
  void* HostPinnedMalloc(size_t bytes) override;
  void HostPinnedFree(void* pointer) override;
  void DeviceMemset(void* pointer, size_t capacity, int value, size_t count) override;
  uint64_t DeviceAllocationCount() const override { return counters_.device_allocations; }
  bool QueryDeviceMemory(size_t* free_bytes, size_t* total_bytes) override;

  DeviceStream CreateStream() override;
  void DestroyStream(DeviceStream stream) override;
  DeviceEvent CreateEvent() override;
  void DestroyEvent(DeviceEvent event) override;
  void RecordEvent(DeviceEvent event, DeviceStream stream) override;
  void StreamWaitEvent(DeviceStream stream, DeviceEvent event) override;
  void SynchronizeStream(DeviceStream stream) override;

  void MemcpyAsync(void* destination, size_t destination_capacity, const void* source, size_t count,
                   MemcpyKind kind, DeviceStream stream) override;
  void MemcpySync(void* destination, size_t destination_capacity, const void* source, size_t count,
                  MemcpyKind kind) override;

  const DmaCounters& counters() const { return counters_; }
  void ResetCounters() { counters_.Reset(); }

  const std::vector<DmaTraceEntry>& trace() const { return trace_; }
  bool trace_overflow() const { return trace_overflow_; }
  void ClearTrace() {
    trace_.clear();
    trace_overflow_ = false;
  }
  // Pointer identity, so a test can assert that a copy's source really was the
  // device slot / the host slot / the transit scratch.
  bool IsDeviceAddress(const void* pointer) const;

 private:
  void Record(const DmaTraceEntry& entry);
  void AccountCopy(MemcpyKind kind, size_t count);

  size_t device_memory_bytes_ = 0;
  size_t device_memory_used_ = 0;
  std::vector<std::pair<const char*, size_t>> device_blocks_;  // base, size
  std::vector<DmaTraceEntry> trace_;
  bool trace_overflow_ = false;
  int32_t next_stream_id_ = 1;
  int32_t next_event_id_ = 1;
  DmaCounters counters_;
};

}  // namespace ascend_moe
