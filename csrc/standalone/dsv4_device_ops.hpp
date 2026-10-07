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

// The memory / stream primitives the exclusive hierarchy is built on, behind
// one interface.
//
// This mirrors `tools/dsv4_moe_runtime/hardware/runtime.py` (DeviceRuntime) and
// exists for the same reason: the swap engine's correctness is a question about
// *ordering* -- which copy may overwrite which slot, and which event has to
// gate it -- and that question is answerable without an NPU. The simulated
// backend runs the identical ExclusiveExpertManager code over plain host memory
// and records the DMA trace, so `dsv4_contract_smoke` verifies the disjointness
// invariant, the chunked duplex exchange and the eviction policy on a build
// machine with no device attached.
//
// Cost: one virtual call per DMA *request*, i.e. a few dozen per 43-layer
// decode step, against transfers measured in MiB. It is not on any per-byte
// path.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vllm_ascend {
namespace dsv4 {

using DeviceStream = void*;
using DeviceEvent = void*;

enum class MemcpyKind { kHostToHost, kHostToDevice, kDeviceToHost, kDeviceToDevice };

const char* MemcpyKindName(MemcpyKind kind);

struct DmaCounters {
  uint64_t host_to_device_bytes = 0;
  uint64_t device_to_host_bytes = 0;
  uint64_t device_to_device_bytes = 0;
  uint64_t async_copies = 0;
  uint64_t sync_copies = 0;
  uint64_t events_recorded = 0;
  uint64_t stream_waits = 0;
  uint64_t stream_synchronizations = 0;
  uint64_t device_allocations = 0;
  uint64_t host_pinned_allocations = 0;

  void Reset() { *this = DmaCounters(); }
};

class DeviceOps {
 public:
  virtual ~DeviceOps() = default;

  virtual const char* backend_name() const = 0;
  // True when the backend talks to a real device; false for the simulator.
  virtual bool is_physical() const = 0;

  virtual void* DeviceMalloc(size_t bytes) = 0;
  virtual void DeviceFree(void* pointer) = 0;
  virtual void* HostPinnedMalloc(size_t bytes) = 0;
  virtual void HostPinnedFree(void* pointer) = 0;
  virtual void DeviceMemset(void* pointer, size_t capacity, int value, size_t count) = 0;

  virtual DeviceStream CreateStream() = 0;
  virtual void DestroyStream(DeviceStream stream) = 0;
  virtual DeviceEvent CreateEvent() = 0;
  virtual void DestroyEvent(DeviceEvent event) = 0;
  virtual void RecordEvent(DeviceEvent event, DeviceStream stream) = 0;
  virtual void StreamWaitEvent(DeviceStream stream, DeviceEvent event) = 0;
  virtual void SynchronizeStream(DeviceStream stream) = 0;

  virtual void MemcpyAsync(void* destination, size_t destination_capacity, const void* source, size_t count,
                           MemcpyKind kind, DeviceStream stream) = 0;
  virtual void MemcpySync(void* destination, size_t destination_capacity, const void* source, size_t count,
                          MemcpyKind kind) = 0;

  // Free / total HBM. False when the backend cannot report it, in which case
  // the slot planner needs an explicit K.
  virtual bool QueryDeviceMemory(size_t* free_bytes, size_t* total_bytes) = 0;

  const DmaCounters& counters() const { return counters_; }
  void ResetCounters() { counters_.Reset(); }

 protected:
  void AccountCopy(MemcpyKind kind, size_t count);

  DmaCounters counters_;
};

// ---------------------------------------------------------------------------
// Physical backend: aclrtMalloc / aclrtMallocHost / aclrtMemcpyAsync
// ---------------------------------------------------------------------------

class AclDeviceOps : public DeviceOps {
 public:
  // Calls aclInit + aclrtSetDevice + aclrtCreateContext. Throws AclError when
  // no device is attached, which makes "device unavailable" an explicit outcome
  // rather than a silent fallback onto host memory.
  explicit AclDeviceOps(int32_t device_id);
  ~AclDeviceOps() override;

  const char* backend_name() const override { return "acl"; }
  bool is_physical() const override { return true; }

  void* DeviceMalloc(size_t bytes) override;
  void DeviceFree(void* pointer) override;
  void* HostPinnedMalloc(size_t bytes) override;
  void HostPinnedFree(void* pointer) override;
  void DeviceMemset(void* pointer, size_t capacity, int value, size_t count) override;

  DeviceStream CreateStream() override;
  void DestroyStream(DeviceStream stream) override;
  DeviceEvent CreateEvent() override;
  void DestroyEvent(DeviceEvent event) override;
  void RecordEvent(DeviceEvent event, DeviceStream stream) override;
  void StreamWaitEvent(DeviceStream stream, DeviceEvent event) override;
  void SynchronizeStream(DeviceStream stream) override;

  void MemcpyAsync(void* destination, size_t destination_capacity, const void* source, size_t count, MemcpyKind kind,
                   DeviceStream stream) override;
  void MemcpySync(void* destination, size_t destination_capacity, const void* source, size_t count,
                  MemcpyKind kind) override;

  bool QueryDeviceMemory(size_t* free_bytes, size_t* total_bytes) override;

  const std::string& soc_name() const { return soc_name_; }

 private:
  int32_t device_id_ = 0;
  void* context_ = nullptr;
  std::string soc_name_;
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

class SimulatedDeviceOps : public DeviceOps {
 public:
  // `trace_capacity` bounds the recorded DMA trace. It is reserved up front so
  // recording never allocates; once full, recording stops and `trace_overflow`
  // is set rather than the buffer growing.
  explicit SimulatedDeviceOps(size_t device_memory_bytes, size_t trace_capacity = 1u << 16);
  ~SimulatedDeviceOps() override;

  const char* backend_name() const override { return "simulated"; }
  bool is_physical() const override { return false; }

  void* DeviceMalloc(size_t bytes) override;
  void DeviceFree(void* pointer) override;
  void* HostPinnedMalloc(size_t bytes) override;
  void HostPinnedFree(void* pointer) override;
  void DeviceMemset(void* pointer, size_t capacity, int value, size_t count) override;

  DeviceStream CreateStream() override;
  void DestroyStream(DeviceStream stream) override;
  DeviceEvent CreateEvent() override;
  void DestroyEvent(DeviceEvent event) override;
  void RecordEvent(DeviceEvent event, DeviceStream stream) override;
  void StreamWaitEvent(DeviceStream stream, DeviceEvent event) override;
  void SynchronizeStream(DeviceStream stream) override;

  void MemcpyAsync(void* destination, size_t destination_capacity, const void* source, size_t count, MemcpyKind kind,
                   DeviceStream stream) override;
  void MemcpySync(void* destination, size_t destination_capacity, const void* source, size_t count,
                  MemcpyKind kind) override;

  bool QueryDeviceMemory(size_t* free_bytes, size_t* total_bytes) override;

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

  size_t device_memory_bytes_ = 0;
  size_t device_memory_used_ = 0;
  std::vector<std::pair<const char*, size_t>> device_blocks_;  // base, size
  std::vector<DmaTraceEntry> trace_;
  bool trace_overflow_ = false;
  int32_t next_stream_id_ = 1;
  int32_t next_event_id_ = 1;
};

}  // namespace dsv4
}  // namespace vllm_ascend
