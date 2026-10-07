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

#include "dsv4_device_ops.hpp"

#include <acl/acl.h>

#include <cstdlib>
#include <cstring>

#include "dsv4_acl_check.hpp"
#include "dsv4_config.hpp"

namespace vllm_ascend {
namespace dsv4 {
namespace {

aclrtMemcpyKind ToAclKind(MemcpyKind kind) {
  switch (kind) {
    case MemcpyKind::kHostToHost:
      return ACL_MEMCPY_HOST_TO_HOST;
    case MemcpyKind::kHostToDevice:
      return ACL_MEMCPY_HOST_TO_DEVICE;
    case MemcpyKind::kDeviceToHost:
      return ACL_MEMCPY_DEVICE_TO_HOST;
    default:
      return ACL_MEMCPY_DEVICE_TO_DEVICE;
  }
}

// std::aligned_alloc demands a size that is a multiple of the alignment.
void* AlignedHostAlloc(size_t bytes, size_t alignment = kArenaAlignBytes) {
  const size_t rounded = ((bytes + alignment - 1) / alignment) * alignment;
  return std::aligned_alloc(alignment, rounded);
}

}  // namespace

const char* MemcpyKindName(MemcpyKind kind) {
  switch (kind) {
    case MemcpyKind::kHostToHost:
      return "H2H";
    case MemcpyKind::kHostToDevice:
      return "H2D";
    case MemcpyKind::kDeviceToHost:
      return "D2H";
    default:
      return "D2D";
  }
}

void DeviceOps::AccountCopy(MemcpyKind kind, size_t count) {
  switch (kind) {
    case MemcpyKind::kHostToDevice:
      counters_.host_to_device_bytes += count;
      break;
    case MemcpyKind::kDeviceToHost:
      counters_.device_to_host_bytes += count;
      break;
    case MemcpyKind::kDeviceToDevice:
      counters_.device_to_device_bytes += count;
      break;
    default:
      break;
  }
}

// ---------------------------------------------------------------------------
// AclDeviceOps
// ---------------------------------------------------------------------------

AclDeviceOps::AclDeviceOps(int32_t device_id) : device_id_(device_id) {
  // aclInit is per process; a second call answers ACL_ERROR_REPEAT_INITIALIZE,
  // which is not a failure for this constructor.
  const aclError init_status = aclInit(nullptr);
  if (init_status != ACL_SUCCESS && init_status != ACL_ERROR_REPEAT_INITIALIZE) {
    throw AclError("aclInit", __FILE__, __LINE__, init_status);
  }
  DSV4_ACL_CHECK(aclrtSetDevice(device_id_));
  aclrtContext context = nullptr;
  DSV4_ACL_CHECK(aclrtCreateContext(&context, device_id_));
  context_ = context;
  DSV4_ACL_CHECK(aclrtSetCurrentContext(context));
  const char* soc = aclrtGetSocName();
  soc_name_ = soc != nullptr ? soc : "unknown";
}

AclDeviceOps::~AclDeviceOps() {
  if (context_ != nullptr) {
    aclrtDestroyContext(static_cast<aclrtContext>(context_));
    context_ = nullptr;
  }
  aclrtResetDevice(device_id_);
}

void* AclDeviceOps::DeviceMalloc(size_t bytes) {
  void* pointer = nullptr;
  DSV4_ACL_CHECK(aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ++counters_.device_allocations;
  return pointer;
}

void AclDeviceOps::DeviceFree(void* pointer) {
  if (pointer != nullptr) {
    aclrtFree(pointer);
  }
}

void* AclDeviceOps::HostPinnedMalloc(size_t bytes) {
  void* pointer = nullptr;
  DSV4_ACL_CHECK(aclrtMallocHost(&pointer, bytes));
  ++counters_.host_pinned_allocations;
  return pointer;
}

void AclDeviceOps::HostPinnedFree(void* pointer) {
  if (pointer != nullptr) {
    aclrtFreeHost(pointer);
  }
}

void AclDeviceOps::DeviceMemset(void* pointer, size_t capacity, int value, size_t count) {
  DSV4_ACL_CHECK(aclrtMemset(pointer, capacity, value, count));
}

DeviceStream AclDeviceOps::CreateStream() {
  aclrtStream stream = nullptr;
  DSV4_ACL_CHECK(aclrtCreateStream(&stream));
  return static_cast<DeviceStream>(stream);
}

void AclDeviceOps::DestroyStream(DeviceStream stream) {
  if (stream != nullptr) {
    aclrtDestroyStream(static_cast<aclrtStream>(stream));
  }
}

DeviceEvent AclDeviceOps::CreateEvent() {
  aclrtEvent event = nullptr;
  // ACL_EVENT_SYNC is the cross-stream dependency flavour: no timestamp, which
  // is what makes it cheap enough to gate every transit chunk.
  DSV4_ACL_CHECK(aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC));
  return static_cast<DeviceEvent>(event);
}

void AclDeviceOps::DestroyEvent(DeviceEvent event) {
  if (event != nullptr) {
    aclrtDestroyEvent(static_cast<aclrtEvent>(event));
  }
}

void AclDeviceOps::RecordEvent(DeviceEvent event, DeviceStream stream) {
  DSV4_ACL_CHECK(aclrtRecordEvent(static_cast<aclrtEvent>(event), static_cast<aclrtStream>(stream)));
  ++counters_.events_recorded;
}

void AclDeviceOps::StreamWaitEvent(DeviceStream stream, DeviceEvent event) {
  DSV4_ACL_CHECK(aclrtStreamWaitEvent(static_cast<aclrtStream>(stream), static_cast<aclrtEvent>(event)));
  ++counters_.stream_waits;
}

void AclDeviceOps::SynchronizeStream(DeviceStream stream) {
  DSV4_ACL_CHECK(aclrtSynchronizeStream(static_cast<aclrtStream>(stream)));
  ++counters_.stream_synchronizations;
}

void AclDeviceOps::MemcpyAsync(void* destination, size_t destination_capacity, const void* source, size_t count,
                               MemcpyKind kind, DeviceStream stream) {
  DSV4_ACL_CHECK(aclrtMemcpyAsync(destination, destination_capacity, source, count, ToAclKind(kind),
                                  static_cast<aclrtStream>(stream)));
  ++counters_.async_copies;
  AccountCopy(kind, count);
}

void AclDeviceOps::MemcpySync(void* destination, size_t destination_capacity, const void* source, size_t count,
                              MemcpyKind kind) {
  DSV4_ACL_CHECK(aclrtMemcpy(destination, destination_capacity, source, count, ToAclKind(kind)));
  ++counters_.sync_copies;
  AccountCopy(kind, count);
}

bool AclDeviceOps::QueryDeviceMemory(size_t* free_bytes, size_t* total_bytes) {
  size_t free_value = 0;
  size_t total_value = 0;
  if (aclrtGetMemInfo(ACL_HBM_MEM, &free_value, &total_value) != ACL_SUCCESS) {
    return false;
  }
  if (free_bytes != nullptr) {
    *free_bytes = free_value;
  }
  if (total_bytes != nullptr) {
    *total_bytes = total_value;
  }
  return true;
}

// ---------------------------------------------------------------------------
// SimulatedDeviceOps
// ---------------------------------------------------------------------------

SimulatedDeviceOps::SimulatedDeviceOps(size_t device_memory_bytes, size_t trace_capacity)
    : device_memory_bytes_(device_memory_bytes) {
  trace_.reserve(trace_capacity);
}

SimulatedDeviceOps::~SimulatedDeviceOps() = default;

void* SimulatedDeviceOps::DeviceMalloc(size_t bytes) {
  DSV4_REQUIRE(device_memory_used_ + bytes <= device_memory_bytes_,
               "simulated device memory exhausted: requested "
                   << bytes << " with " << (device_memory_bytes_ - device_memory_used_) << " free");
  void* pointer = AlignedHostAlloc(bytes, kSimDeviceAllocAlignBytes);
  DSV4_REQUIRE(pointer != nullptr, "simulated device allocation of " << bytes << " bytes failed");
  DSV4_REQUIRE(reinterpret_cast<uintptr_t>(pointer) % kSimDeviceAllocAlignBytes == 0,
               "simulated device allocation is not page-aligned");
  device_memory_used_ += bytes;
  device_blocks_.emplace_back(static_cast<const char*>(pointer), bytes);
  ++counters_.device_allocations;
  return pointer;
}

void SimulatedDeviceOps::DeviceFree(void* pointer) {
  if (pointer == nullptr) {
    return;
  }
  const char* base = static_cast<const char*>(pointer);
  for (size_t index = 0; index < device_blocks_.size(); ++index) {
    if (device_blocks_[index].first == base) {
      device_memory_used_ -= device_blocks_[index].second;
      device_blocks_.erase(device_blocks_.begin() + static_cast<std::ptrdiff_t>(index));
      break;
    }
  }
  std::free(pointer);
}

void* SimulatedDeviceOps::HostPinnedMalloc(size_t bytes) {
  void* pointer = AlignedHostAlloc(bytes);
  DSV4_REQUIRE(pointer != nullptr, "simulated pinned host allocation of " << bytes << " bytes failed");
  ++counters_.host_pinned_allocations;
  return pointer;
}

void SimulatedDeviceOps::HostPinnedFree(void* pointer) { std::free(pointer); }

void SimulatedDeviceOps::DeviceMemset(void* pointer, size_t capacity, int value, size_t count) {
  DSV4_REQUIRE(count <= capacity, "memset count " << count << " exceeds capacity " << capacity);
  std::memset(pointer, value, count);
}

DeviceStream SimulatedDeviceOps::CreateStream() {
  return reinterpret_cast<DeviceStream>(static_cast<std::intptr_t>(next_stream_id_++));
}

void SimulatedDeviceOps::DestroyStream(DeviceStream) {}

DeviceEvent SimulatedDeviceOps::CreateEvent() {
  return reinterpret_cast<DeviceEvent>(static_cast<std::intptr_t>(next_event_id_++));
}

void SimulatedDeviceOps::DestroyEvent(DeviceEvent) {}

void SimulatedDeviceOps::RecordEvent(DeviceEvent event, DeviceStream stream) {
  DmaTraceEntry entry;
  entry.kind = DmaTraceEntry::Kind::kRecordEvent;
  entry.stream_id = static_cast<int32_t>(reinterpret_cast<std::intptr_t>(stream));
  entry.event_id = static_cast<int32_t>(reinterpret_cast<std::intptr_t>(event));
  Record(entry);
  ++counters_.events_recorded;
}

void SimulatedDeviceOps::StreamWaitEvent(DeviceStream stream, DeviceEvent event) {
  DmaTraceEntry entry;
  entry.kind = DmaTraceEntry::Kind::kStreamWait;
  entry.stream_id = static_cast<int32_t>(reinterpret_cast<std::intptr_t>(stream));
  entry.event_id = static_cast<int32_t>(reinterpret_cast<std::intptr_t>(event));
  Record(entry);
  ++counters_.stream_waits;
}

void SimulatedDeviceOps::SynchronizeStream(DeviceStream stream) {
  DmaTraceEntry entry;
  entry.kind = DmaTraceEntry::Kind::kStreamSync;
  entry.stream_id = static_cast<int32_t>(reinterpret_cast<std::intptr_t>(stream));
  Record(entry);
  ++counters_.stream_synchronizations;
}

void SimulatedDeviceOps::MemcpyAsync(void* destination, size_t destination_capacity, const void* source, size_t count,
                                     MemcpyKind kind, DeviceStream stream) {
  DSV4_REQUIRE(count <= destination_capacity, "simulated " << MemcpyKindName(kind) << " copy of " << count
                                                           << " bytes exceeds destination capacity "
                                                           << destination_capacity);
  // Copies execute immediately and in issue order. The simulator's value is the
  // recorded *order* of requests and gates: a program whose correctness needs
  // two of its own copies to overlap in time is already wrong.
  std::memcpy(destination, source, count);
  DmaTraceEntry entry;
  entry.kind = DmaTraceEntry::Kind::kMemcpyAsync;
  entry.stream_id = static_cast<int32_t>(reinterpret_cast<std::intptr_t>(stream));
  entry.memcpy_kind = kind;
  entry.destination = destination;
  entry.source = source;
  entry.count = count;
  Record(entry);
  ++counters_.async_copies;
  AccountCopy(kind, count);
}

void SimulatedDeviceOps::MemcpySync(void* destination, size_t destination_capacity, const void* source, size_t count,
                                    MemcpyKind kind) {
  DSV4_REQUIRE(count <= destination_capacity,
               "simulated sync copy of " << count << " bytes exceeds capacity " << destination_capacity);
  std::memcpy(destination, source, count);
  DmaTraceEntry entry;
  entry.kind = DmaTraceEntry::Kind::kMemcpySync;
  entry.memcpy_kind = kind;
  entry.destination = destination;
  entry.source = source;
  entry.count = count;
  Record(entry);
  ++counters_.sync_copies;
  AccountCopy(kind, count);
}

bool SimulatedDeviceOps::QueryDeviceMemory(size_t* free_bytes, size_t* total_bytes) {
  if (free_bytes != nullptr) {
    *free_bytes = device_memory_bytes_ - device_memory_used_;
  }
  if (total_bytes != nullptr) {
    *total_bytes = device_memory_bytes_;
  }
  return true;
}

bool SimulatedDeviceOps::IsDeviceAddress(const void* pointer) const {
  const char* address = static_cast<const char*>(pointer);
  for (const auto& block : device_blocks_) {
    if (address >= block.first && address < block.first + block.second) {
      return true;
    }
  }
  return false;
}

void SimulatedDeviceOps::Record(const DmaTraceEntry& entry) {
  if (trace_.size() == trace_.capacity()) {
    trace_overflow_ = true;  // bounded: recording stops, the buffer never grows
    return;
  }
  trace_.push_back(entry);
}

}  // namespace dsv4
}  // namespace vllm_ascend
