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

// Device runtime for the CUDA backend of the bare-metal kernel suite.
//
// This is the NVIDIA-side counterpart of device_buffer.hpp plus
// test_harness.hpp: an RAII allocation type, an RAII owner of the device and
// its stream, and the GTest environment that constructs the latter once per
// process. The names and the method set deliberately mirror the Ascend ones so
// a test written against either backend reads the same way.
//
// Nothing here depends on PyTorch, torch_npu, Cutlass, cuBLAS or any Python
// binding: it is the CUDA Runtime API and the C++ standard library only. The
// header uses <cuda_runtime_api.h> rather than <cuda_runtime.h> so that a .cpp
// translation unit compiled by the host compiler can include it without nvcc
// being involved; the kernels are the only thing that needs nvcc.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cuda_check.hpp"

namespace vllm_ascend {
namespace test {

// The Ascend buffer pads to 32 bytes because the MTE2/MTE3 units move data in
// 32-byte bursts. CUDA has no equivalent hard requirement - cudaMalloc already
// returns at least 256-byte aligned memory - but the same padding is applied
// here so that a kernel ported between the two backends sees an allocation with
// identical tail behaviour, and so the vectorised half2 paths can assume a
// 4-byte aligned base without a per-call check.
constexpr size_t kCudaDeviceAlignBytes = 32;

// What the benchmark paths ask for: 512 bytes is a whole sector group on every
// architecture this suite targets, so a timed buffer never starts mid-line.
constexpr size_t kCudaBenchmarkAlignBytes = 512;

constexpr size_t CudaAlignUp(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

// RAII device allocation. Interface-compatible with the Ascend DeviceBuffer for
// everything the tests use: Allocate/Release, host transfers, Zero, the
// FromHost/Empty/ToHost helpers, and the accessors.
class CUDADeviceBuffer {
 public:
  CUDADeviceBuffer() = default;

  explicit CUDADeviceBuffer(size_t size_bytes, size_t alignment = kCudaDeviceAlignBytes) {
    Allocate(size_bytes, alignment);
  }

  CUDADeviceBuffer(const CUDADeviceBuffer&) = delete;
  CUDADeviceBuffer& operator=(const CUDADeviceBuffer&) = delete;

  CUDADeviceBuffer(CUDADeviceBuffer&& other) noexcept
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

  CUDADeviceBuffer& operator=(CUDADeviceBuffer&& other) noexcept {
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

  ~CUDADeviceBuffer() { Release(); }

  void Allocate(size_t size_bytes, size_t alignment = kCudaDeviceAlignBytes);
  void Release();

  // Synchronous transfers, matching aclrtMemcpy semantics: the copy has
  // completed when the call returns.
  void CopyFromHost(const void* host, size_t size_bytes);
  void CopyToHost(void* host, size_t size_bytes) const;
  void Zero();

  // Stream-ordered variants. The caller synchronises before touching the host
  // side of a device-to-host copy.
  void CopyFromHostAsync(const void* host, size_t size_bytes, cudaStream_t stream);
  void CopyToHostAsync(void* host, size_t size_bytes, cudaStream_t stream) const;
  void ZeroAsync(cudaStream_t stream);

  template <typename T>
  static CUDADeviceBuffer FromHost(const std::vector<T>& host, size_t alignment = kCudaDeviceAlignBytes) {
    CUDADeviceBuffer buffer(host.size() * sizeof(T), alignment);
    if (!host.empty()) {
      buffer.CopyFromHost(host.data(), host.size() * sizeof(T));
    }
    return buffer;
  }

  template <typename T>
  static CUDADeviceBuffer Empty(size_t element_count, size_t alignment = kCudaDeviceAlignBytes) {
    return CUDADeviceBuffer(element_count * sizeof(T), alignment);
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
  void* base_ = nullptr;       // pointer cudaFree must be given
  void* data_ = nullptr;       // base_, advanced to the requested alignment
  size_t size_bytes_ = 0;      // logical size requested by the caller
  size_t capacity_bytes_ = 0;  // usable size from data_, padded to alignment_
  size_t alignment_ = kCudaDeviceAlignBytes;
};

// Device ordinal used by every test. Override with CUDA_TEST_DEVICE_ID.
int ResolveCudaDeviceId();

// RAII owner of the CUDA runtime state. Construction performs cudaSetDevice ->
// cudaGetDeviceProperties -> cudaStreamCreate; destruction drains and destroys
// the stream and then resets the device, and is safe to run after a partially
// failed setup.
//
// There is no cudaInit() to match aclInit(): the runtime API brings up the
// primary context lazily on first use, so cudaSetDevice followed by a call that
// touches the context is the equivalent handshake.
class CUDADevice {
 public:
  CUDADevice();
  ~CUDADevice();

  CUDADevice(const CUDADevice&) = delete;
  CUDADevice& operator=(const CUDADevice&) = delete;

  cudaStream_t stream() const { return stream_; }
  int device_id() const { return device_id_; }

  const std::string& name() const { return name_; }
  int compute_capability_major() const { return compute_capability_major_; }
  int compute_capability_minor() const { return compute_capability_minor_; }

  // Upper bound the paged-attention decode launcher checks its per-block score
  // buffer against before launching.
  size_t max_shared_memory_per_block() const { return max_shared_memory_per_block_; }

  // "12.0", for log lines and skip reasons.
  std::string compute_capability_string() const;

  void SynchronizeStream() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

 private:
  int device_id_ = 0;
  bool device_set_ = false;
  cudaStream_t stream_ = nullptr;
  std::string name_;
  int compute_capability_major_ = 0;
  int compute_capability_minor_ = 0;
  size_t max_shared_memory_per_block_ = 0;
};

// GTest global environment. SetUp() never fails the run when no GPU is present:
// it records why the device is unavailable and every device test skips with
// that reason, so the suite stays runnable on a machine with no NVIDIA card.
class CUDATestEnvironment {
 public:
  static CUDATestEnvironment& Instance();

  void SetUp();
  void TearDown();

  bool available() const { return device_ != nullptr; }
  const std::string& unavailable_reason() const { return unavailable_reason_; }

  CUDADevice& device();
  cudaStream_t stream();

  const std::string& device_name() const { return device_name_; }

 private:
  CUDATestEnvironment() = default;

  std::unique_ptr<CUDADevice> device_;
  std::string unavailable_reason_;
  std::string device_name_;
};

// Registers the environment with GTest. Called from cuda_main.cpp.
void RegisterCudaTestEnvironment();

#define REQUIRE_CUDA_DEVICE()                                                  \
  do {                                                                         \
    if (!::vllm_ascend::test::CUDATestEnvironment::Instance().available()) {    \
      GTEST_SKIP() << "No usable CUDA device: "                                \
                   << ::vllm_ascend::test::CUDATestEnvironment::Instance()     \
                          .unavailable_reason();                               \
    }                                                                          \
  } while (false)

}  // namespace test
}  // namespace vllm_ascend
