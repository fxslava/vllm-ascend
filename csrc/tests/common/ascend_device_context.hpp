// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ascend_concurrent_runner.hpp"
#include "device_buffer.hpp"
#include "test_harness.hpp"

namespace vllm_ascend::test {
// Keep the existing exception-safe aclInit/SetDevice/ResetDevice owner and
// aligned HBM allocator as the single implementation used by every tier.
using AscendDeviceContext = AscendDevice;
using ManagedHbmBuffer = DeviceBuffer;

class AscendStream {
 public:
  explicit AscendStream(bool create = true) {
    if (create) ACL_CHECK(aclrtCreateStream(&stream_));
  }
  ~AscendStream() {
    if (stream_ == nullptr) return;
    ACL_CHECK_NOTHROW(aclrtSynchronizeStream(stream_));
    ACL_CHECK_NOTHROW(aclrtDestroyStream(stream_));
  }
  AscendStream(const AscendStream&) = delete;
  AscendStream& operator=(const AscendStream&) = delete;
  aclrtStream get() const { return stream_; }

 private:
  aclrtStream stream_ = nullptr;
};

class PinnedHostBuffer {
 public:
  explicit PinnedHostBuffer(size_t bytes) { ACL_CHECK(aclrtMallocHost(&data_, bytes)); }
  ~PinnedHostBuffer() { ACL_CHECK_NOTHROW(aclrtFreeHost(data_)); }
  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
  void* get() const { return data_; }

 private:
  void* data_ = nullptr;
};

template <typename T>
std::vector<T> CopyToHostAsync(const DeviceBuffer& buffer, aclrtStream stream) {
  const size_t bytes = buffer.size_bytes();
  if (bytes % sizeof(T) != 0) throw std::invalid_argument("D2H element size mismatch");
  std::vector<T> result(bytes / sizeof(T));
  if (bytes == 0) return result;
  PinnedHostBuffer pinned(bytes);
  ACL_CHECK(aclrtMemcpyAsync(pinned.get(), bytes, buffer.get(), bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream));
  // Keep pinned storage alive even if synchronization throws.
  const aclError status = aclrtSynchronizeStream(stream);
  if (status != ACL_SUCCESS) ACL_CHECK_NOTHROW(aclrtSynchronizeStream(stream));
  ACL_CHECK(status);
  std::memcpy(result.data(), pinned.get(), bytes);
  return result;
}

template <typename T>
void CopyFromHostAsync(DeviceBuffer& buffer, const std::vector<T>& values, aclrtStream stream) {
  const size_t bytes = values.size() * sizeof(T);
  if (bytes > buffer.size_bytes()) throw std::invalid_argument("H2D exceeds buffer size");
  if (bytes == 0) return;
  PinnedHostBuffer pinned(bytes);
  std::memcpy(pinned.get(), values.data(), bytes);
  ACL_CHECK(
      aclrtMemcpyAsync(buffer.get(), buffer.capacity_bytes(), pinned.get(), bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream));
  const aclError status = aclrtSynchronizeStream(stream);
  if (status != ACL_SUCCESS) ACL_CHECK_NOTHROW(aclrtSynchronizeStream(stream));
  ACL_CHECK(status);
}

}  // namespace vllm_ascend::test
