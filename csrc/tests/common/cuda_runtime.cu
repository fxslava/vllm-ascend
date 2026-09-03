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

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "cuda_runtime.hpp"

namespace vllm_ascend {
namespace test {

void ReportIgnoredCudaFailure(const char* expression, const char* file, int line, cudaError_t status) {
  std::fprintf(stderr, "[cuda-test] ignoring failure during cleanup: %s\n",
               FormatCudaError(expression, file, line, status).c_str());
}

// ---------------------------------------------------------------------------
// CUDADeviceBuffer
// ---------------------------------------------------------------------------

void CUDADeviceBuffer::Allocate(size_t size_bytes, size_t alignment) {
  Release();
  if (size_bytes == 0) {
    return;
  }
  alignment_ = (alignment < kCudaDeviceAlignBytes) ? kCudaDeviceAlignBytes : alignment;
  size_bytes_ = size_bytes;
  capacity_bytes_ = CudaAlignUp(size_bytes, alignment_);

  // cudaMalloc is documented to return memory aligned to at least 256 bytes, so
  // the extra slack is only ever paid for a request stronger than that. The
  // over-allocate-and-offset shape is kept identical to the Ascend buffer so
  // both backends have the same failure mode if the guarantee ever changes.
  const size_t request =
      capacity_bytes_ + ((alignment_ > kCudaDeviceAlignBytes) ? alignment_ : 0);
  CUDA_CHECK(cudaMalloc(&base_, request));
  data_ = reinterpret_cast<void*>(CudaAlignUp(reinterpret_cast<uintptr_t>(base_), alignment_));

  if ((reinterpret_cast<uintptr_t>(data_) % alignment_) != 0) {
    throw CudaError("device allocation could not be aligned as requested", __FILE__, __LINE__,
                    cudaErrorMemoryAllocation);
  }
  // Zero the padding so a vectorised tail access never reads uninitialised
  // device memory, matching what the Ascend buffer does for a tail burst.
  CUDA_CHECK(cudaMemset(data_, 0, capacity_bytes_));
}

void CUDADeviceBuffer::Release() {
  if (base_ != nullptr) {
    CUDA_CHECK_NOTHROW(cudaFree(base_));
    base_ = nullptr;
  }
  data_ = nullptr;
  size_bytes_ = 0;
  capacity_bytes_ = 0;
}

void CUDADeviceBuffer::CopyFromHost(const void* host, size_t size_bytes) {
  CUDA_CHECK(cudaMemcpy(data_, host, size_bytes, cudaMemcpyHostToDevice));
}

void CUDADeviceBuffer::CopyToHost(void* host, size_t size_bytes) const {
  CUDA_CHECK(cudaMemcpy(host, data_, size_bytes, cudaMemcpyDeviceToHost));
}

void CUDADeviceBuffer::Zero() {
  if (data_ != nullptr) {
    CUDA_CHECK(cudaMemset(data_, 0, capacity_bytes_));
  }
}

void CUDADeviceBuffer::CopyFromHostAsync(const void* host, size_t size_bytes, cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(data_, host, size_bytes, cudaMemcpyHostToDevice, stream));
}

void CUDADeviceBuffer::CopyToHostAsync(void* host, size_t size_bytes, cudaStream_t stream) const {
  CUDA_CHECK(cudaMemcpyAsync(host, data_, size_bytes, cudaMemcpyDeviceToHost, stream));
}

void CUDADeviceBuffer::ZeroAsync(cudaStream_t stream) {
  if (data_ != nullptr) {
    CUDA_CHECK(cudaMemsetAsync(data_, 0, capacity_bytes_, stream));
  }
}

// ---------------------------------------------------------------------------
// CUDADevice
// ---------------------------------------------------------------------------

int ResolveCudaDeviceId() {
  const char* raw = std::getenv("CUDA_TEST_DEVICE_ID");
  if (raw == nullptr || raw[0] == '\0') {
    return 0;
  }
  return std::atoi(raw);
}

CUDADevice::CUDADevice() {
  device_id_ = ResolveCudaDeviceId();
  try {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
      throw CudaError("cudaGetDeviceCount reported no devices", __FILE__, __LINE__, cudaErrorNoDevice);
    }
    if (device_id_ < 0 || device_id_ >= device_count) {
      throw CudaError("CUDA_TEST_DEVICE_ID is out of range for the installed devices", __FILE__, __LINE__,
                      cudaErrorInvalidDevice);
    }

    CUDA_CHECK(cudaSetDevice(device_id_));
    device_set_ = true;

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device_id_));
    name_ = properties.name;
    compute_capability_major_ = properties.major;
    compute_capability_minor_ = properties.minor;
    max_shared_memory_per_block_ = properties.sharedMemPerBlock;

    // cudaSetDevice on its own is lazy. Creating the stream is the first call
    // that actually brings up the primary context, so a driver or
    // architecture mismatch is reported here rather than from inside a test.
    CUDA_CHECK(cudaStreamCreate(&stream_));
  } catch (...) {
    // Undo whatever succeeded before rethrowing, so a failed construction does
    // not leave the runtime half-initialised for the next attempt.
    if (stream_ != nullptr) {
      CUDA_CHECK_NOTHROW(cudaStreamDestroy(stream_));
      stream_ = nullptr;
    }
    if (device_set_) {
      CUDA_CHECK_NOTHROW(cudaDeviceReset());
      device_set_ = false;
    }
    throw;
  }
}

CUDADevice::~CUDADevice() {
  if (stream_ != nullptr) {
    // Drain before destroying: an in-flight kernel holding a reference to a
    // destroyed stream is the usual source of teardown errors.
    CUDA_CHECK_NOTHROW(cudaStreamSynchronize(stream_));
    CUDA_CHECK_NOTHROW(cudaStreamDestroy(stream_));
    stream_ = nullptr;
  }
  if (device_set_) {
    CUDA_CHECK_NOTHROW(cudaDeviceReset());
    device_set_ = false;
  }
}

std::string CUDADevice::compute_capability_string() const {
  std::ostringstream stream;
  stream << compute_capability_major_ << "." << compute_capability_minor_;
  return stream.str();
}

// ---------------------------------------------------------------------------
// CUDATestEnvironment
// ---------------------------------------------------------------------------

CUDATestEnvironment& CUDATestEnvironment::Instance() {
  static CUDATestEnvironment instance;
  return instance;
}

void CUDATestEnvironment::SetUp() {
  try {
    device_ = std::unique_ptr<CUDADevice>(new CUDADevice());
    device_name_ = device_->name();
    std::fprintf(stdout, "[cuda-test] device %d ready, name='%s', compute capability %s\n",
                 device_->device_id(), device_name_.c_str(),
                 device_->compute_capability_string().c_str());
  } catch (const std::exception& error) {
    device_.reset();
    unavailable_reason_ = error.what();
    std::fprintf(stdout, "[cuda-test] device unavailable, all device tests will skip:\n%s\n",
                 unavailable_reason_.c_str());
  }
}

void CUDATestEnvironment::TearDown() { device_.reset(); }

CUDADevice& CUDATestEnvironment::device() {
  if (device_ == nullptr) {
    throw CudaError("CUDATestEnvironment::device() called without a device", __FILE__, __LINE__,
                    cudaErrorNoDevice);
  }
  return *device_;
}

cudaStream_t CUDATestEnvironment::stream() { return device().stream(); }

// RegisterCudaTestEnvironment is deliberately not defined here: it needs
// <gtest/gtest.h>, and this translation unit is the one nvcc compiles. Keeping
// GTest out of it means a googletest release that trips over the device
// compiler cannot break the runtime layer. See cuda_main.cpp for the
// definition.

}  // namespace test
}  // namespace vllm_ascend
