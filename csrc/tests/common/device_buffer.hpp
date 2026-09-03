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

constexpr size_t AlignUp(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t size_bytes) { Allocate(size_bytes); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(other.data_), size_bytes_(other.size_bytes_), capacity_bytes_(other.capacity_bytes_) {
    other.data_ = nullptr;
    other.size_bytes_ = 0;
    other.capacity_bytes_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Release();
      data_ = other.data_;
      size_bytes_ = other.size_bytes_;
      capacity_bytes_ = other.capacity_bytes_;
      other.data_ = nullptr;
      other.size_bytes_ = 0;
      other.capacity_bytes_ = 0;
    }
    return *this;
  }

  ~DeviceBuffer() { Release(); }

  void Allocate(size_t size_bytes) {
    Release();
    if (size_bytes == 0) {
      return;
    }
    size_bytes_ = size_bytes;
    capacity_bytes_ = AlignUp(size_bytes, kDeviceAlignBytes);
    ACL_CHECK(aclrtMalloc(&data_, capacity_bytes_, ACL_MEM_MALLOC_HUGE_FIRST));
    // aclrtMalloc aligns well past 32 bytes in practice; the check documents
    // the contract the kernels rely on and fails loudly if it ever changes.
    if ((reinterpret_cast<uintptr_t>(data_) % kDeviceAlignBytes) != 0) {
      throw AclError("aclrtMalloc returned a pointer that is not 32-byte aligned", __FILE__, __LINE__, -1);
    }
    // Zero the padding so a tail burst never reads uninitialised device memory.
    ACL_CHECK(aclrtMemset(data_, capacity_bytes_, 0, capacity_bytes_));
  }

  void Release() {
    if (data_ != nullptr) {
      ACL_CHECK_NOTHROW(aclrtFree(data_));
      data_ = nullptr;
    }
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
  static DeviceBuffer FromHost(const std::vector<T>& host) {
    DeviceBuffer buffer(host.size() * sizeof(T));
    if (!host.empty()) {
      buffer.CopyFromHost(host.data(), host.size() * sizeof(T));
    }
    return buffer;
  }

  template <typename T>
  static DeviceBuffer Empty(size_t element_count) {
    return DeviceBuffer(element_count * sizeof(T));
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
  bool empty() const { return data_ == nullptr; }

 private:
  void* data_ = nullptr;
  size_t size_bytes_ = 0;      // logical size requested by the caller
  size_t capacity_bytes_ = 0;  // size actually allocated, padded to 32 bytes
};

}  // namespace test
}  // namespace vllm_ascend
