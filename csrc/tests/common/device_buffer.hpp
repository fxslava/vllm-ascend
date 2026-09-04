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

// RAII wrapper around a device allocation, with the alignment rules the
// DaVinci v200 (Ascend 310P) data-movement units impose.
//
// MTE2 (GM -> UB) and MTE3 (UB -> GM) move data in 32-byte bursts. A kernel
// whose tail block is not a whole number of bursts still issues a full burst,
// so the allocation is padded up to a 32-byte multiple. Without the padding the
// tail burst reads or writes past the end of the allocation, which shows up as
// an intermittent EXCEPTION rather than a deterministic failure.
//
// Allocate() takes an optional stronger alignment. The correctness tests use
// the 32-byte default; the benchmarks ask for 512 so that a measurement is not
// silently penalised by a buffer that straddles a cache line or an HBM burst.
// aclrtMalloc already returns pointers aligned far past 32 bytes in practice,
// but the documented guarantee stops there, so every request over-allocates by
// one alignment and offsets into the block rather than trusting the allocator.
//
// The slack has to be unconditional. Making it conditional on a
// stronger-than-default alignment was a latent overrun: with no slack, a base
// pointer that needed advancing left fewer than capacity_bytes_ bytes
// addressable from data_, and the memset, CopyFromHost and Zero below all
// write the padded capacity rather than the requested size.

#pragma once

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "acl_check.hpp"

namespace vllm_ascend {
namespace test {

// Burst size of the MTE2/MTE3 units on DaVinci v200.
constexpr size_t kDeviceAlignBytes = 32;

// Alignment the benchmarks request. 512 bytes is a whole L2 line on v200 and a
// whole HBM burst, so a timed buffer never starts mid-line; see the note at the
// top of this file for why it is requested rather than assumed.
constexpr size_t kBenchmarkAlignBytes = 512;

constexpr size_t AlignUp(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t size_bytes, size_t alignment = kDeviceAlignBytes) {
    Allocate(size_bytes, alignment);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : base_(other.base_),
        data_(other.data_),
        size_bytes_(other.size_bytes_),
        capacity_bytes_(other.capacity_bytes_),
        alignment_(other.alignment_) {
    other.base_ = nullptr;
    other.data_ = nullptr;
    other.size_bytes_ = 0;
    other.capacity_bytes_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Release();
      base_ = other.base_;
      data_ = other.data_;
      size_bytes_ = other.size_bytes_;
      capacity_bytes_ = other.capacity_bytes_;
      alignment_ = other.alignment_;
      other.base_ = nullptr;
      other.data_ = nullptr;
      other.size_bytes_ = 0;
      other.capacity_bytes_ = 0;
    }
    return *this;
  }

  ~DeviceBuffer() { Release(); }

  void Allocate(size_t size_bytes, size_t alignment = kDeviceAlignBytes) {
    Release();
    if (size_bytes == 0) {
      return;
    }
    alignment_ = (alignment < kDeviceAlignBytes) ? kDeviceAlignBytes : alignment;
    // AlignUp and the `% alignment_` check below are a correct pair only for a
    // power of two, and every burst and line size on this part is one.
    if ((alignment_ & (alignment_ - 1)) != 0) {
      throw AclError("device allocation alignment must be a power of two", __FILE__, __LINE__, -1);
    }
    // AlignUp wraps rather than saturating, so without this a size near the top
    // of the address space would quietly produce a tiny allocation.
    if (size_bytes > SIZE_MAX - 2 * alignment_) {
      throw AclError("device allocation size overflows when padded to alignment", __FILE__, __LINE__, -1);
    }
    size_bytes_ = size_bytes;
    capacity_bytes_ = AlignUp(size_bytes, alignment_);

    // One alignment of slack, always; see the note at the top of this file for
    // why making it conditional was wrong.
    const size_t request = capacity_bytes_ + alignment_;
    ACL_CHECK(aclrtMalloc(&base_, request, ACL_MEM_MALLOC_HUGE_FIRST));

    const size_t base_address = reinterpret_cast<size_t>(base_);
    const size_t data_address = AlignUp(base_address, alignment_);
    data_ = reinterpret_cast<void*>(data_address);

    // The two invariants everything below relies on: data_ is aligned as asked,
    // and capacity_bytes_ bytes really are addressable from it.
    if ((data_address % alignment_) != 0 ||
        (data_address - base_address) + capacity_bytes_ > request) {
      throw AclError("device allocation could not be aligned as requested", __FILE__, __LINE__, -1);
    }
    // Zero the padding so a tail burst never reads uninitialised device memory.
    ACL_CHECK(aclrtMemset(data_, capacity_bytes_, 0, capacity_bytes_));
  }

  void Release() {
    if (base_ != nullptr) {
      ACL_CHECK_NOTHROW(aclrtFree(base_));
      base_ = nullptr;
    }
    data_ = nullptr;
    size_bytes_ = 0;
    capacity_bytes_ = 0;
  }

  void CopyFromHost(const void* host, size_t size_bytes) {
    ACL_CHECK(aclrtMemcpy(data_, capacity_bytes_, host, size_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  }

  void CopyToHost(void* host, size_t size_bytes) const {
    ACL_CHECK(aclrtMemcpy(host, size_bytes, data_, size_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
  }

  void Zero() {
    if (data_ != nullptr) {
      ACL_CHECK(aclrtMemset(data_, capacity_bytes_, 0, capacity_bytes_));
    }
  }

  template <typename T>
  static DeviceBuffer FromHost(const std::vector<T>& host, size_t alignment = kDeviceAlignBytes) {
    DeviceBuffer buffer(host.size() * sizeof(T), alignment);
    if (!host.empty()) {
      buffer.CopyFromHost(host.data(), host.size() * sizeof(T));
    }
    return buffer;
  }

  template <typename T>
  static DeviceBuffer Empty(size_t element_count, size_t alignment = kDeviceAlignBytes) {
    return DeviceBuffer(element_count * sizeof(T), alignment);
  }

  template <typename T>
  std::vector<T> ToHost() const {
    std::vector<T> host(size_bytes_ / sizeof(T));
    if (!host.empty()) {
      CopyToHost(host.data(), host.size() * sizeof(T));
    }
    return host;
  }

  void* get() const { return data_; }
  size_t size_bytes() const { return size_bytes_; }
  size_t capacity_bytes() const { return capacity_bytes_; }
  size_t alignment() const { return alignment_; }
  bool empty() const { return data_ == nullptr; }

 private:
  void* base_ = nullptr;       // pointer aclrtFree must be given
  void* data_ = nullptr;       // base_, advanced to the requested alignment
  size_t size_bytes_ = 0;      // logical size requested by the caller
  size_t capacity_bytes_ = 0;  // usable size from data_, padded to alignment_
  size_t alignment_ = kDeviceAlignBytes;
};

}  // namespace test
}  // namespace vllm_ascend
