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

// IDeviceAllocator: the memory half of the backend contract (ISP).
//
// Consumers that only allocate -- the activation arena, the backbone staging
// buffer, the exclusive hierarchy's slot pools -- depend on this interface
// alone and never see streams, events or transfer ordering. The allocation
// counter is part of the contract because the decode loop's zero-allocation
// invariant is enforced by watching it across a step.

#pragma once

#include <cstddef>
#include <cstdint>

namespace ascend_moe {

class IDeviceAllocator {
 public:
  virtual ~IDeviceAllocator() = default;

  // Device (HBM) memory. Alignment follows the backend's hardware page
  // (aclrtMalloc(..., ACL_MEM_MALLOC_HUGE_FIRST) serves 4096-aligned memory
  // on the physical backend; every backend must match that).
  virtual void* DeviceMalloc(size_t bytes) = 0;
  virtual void DeviceFree(void* pointer) = 0;

  // Page-locked host memory, the only kind a DMA may source or land in.
  virtual void* HostPinnedMalloc(size_t bytes) = 0;
  virtual void HostPinnedFree(void* pointer) = 0;

  // Fills `count` bytes of an allocation. Synchronous; no stream ordering.
  virtual void DeviceMemset(void* pointer, size_t capacity, int value, size_t count) = 0;

  // Allocations served so far. DecodeStep asserts the delta over a step is
  // zero -- that is how "no dynamic allocation inside the 43-layer loop"
  // stays enforced rather than aspirational.
  virtual uint64_t DeviceAllocationCount() const = 0;

  // Free / total device memory. False when the backend cannot report it, in
  // which case the slot planner needs an explicit K.
  virtual bool QueryDeviceMemory(size_t* free_bytes, size_t* total_bytes) = 0;

  // Diagnostics identity for reports ("acl", "simulated", ...).
  virtual const char* backend_name() const = 0;
};

}  // namespace ascend_moe
