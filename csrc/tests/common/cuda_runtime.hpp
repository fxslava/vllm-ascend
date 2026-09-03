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
// Nothing here depends on PyTorch, torch_npu, Cutlass or any Python binding.
// The one library beyond the CUDA Runtime API is cuBLAS, and only for the
// linear projections - see the cuBLAS section below for why that one is not
// hand-written like the rest. The header uses <cuda_runtime_api.h> rather than
// <cuda_runtime.h>, and <cublas_v2.h> is a plain C header as well, so a .cpp
// translation unit compiled by the host compiler can include this without nvcc
// being involved; the kernels are the only thing that needs nvcc.

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <sstream>
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

// ---------------------------------------------------------------------------
// cuBLAS
// ---------------------------------------------------------------------------
//
// The one library this backend does not write out by hand. A GEMM worth
// checking a reference against is a project of its own, and a naive one would
// only be measuring the kernel this suite just wrote; cuBLAS is what a CUDA
// deployment actually calls for a linear projection, so it is what the
// projections are checked against here. Everything in kernels/cuda stays
// hand-written.
//
// cublasStatus_t is a separate error domain from cudaError_t, so it gets its
// own formatter and exception rather than being squeezed into CudaError.
// cublasGetStatusName only exists from CUDA 11.4.2, and README.md puts the
// toolkit floor at 11.0, so the names are spelled out instead.

inline const char* CublasStatusName(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  }
}

inline std::string FormatCublasError(const char* expression, const char* file, int line,
                                     cublasStatus_t status) {
  std::ostringstream stream;
  stream << file << ":" << line << ": " << expression << " failed with status " << static_cast<int>(status)
         << "\n  cuBLAS: " << CublasStatusName(status);
  return stream.str();
}

class CublasError : public std::exception {
 public:
  CublasError(const char* expression, const char* file, int line, cublasStatus_t status)
      : message_(FormatCublasError(expression, file, line, status)), status_(status) {}

  const char* what() const noexcept override { return message_.c_str(); }
  cublasStatus_t status() const { return status_; }

 private:
  std::string message_;
  cublasStatus_t status_;
};

void ReportIgnoredCublasFailure(const char* expression, const char* file, int line, cublasStatus_t status);

#define CUBLAS_CHECK(expression)                                                         \
  do {                                                                                   \
    const cublasStatus_t vllm_ascend_cublas_status = (expression);                       \
    if (vllm_ascend_cublas_status != CUBLAS_STATUS_SUCCESS) {                            \
      throw ::vllm_ascend::test::CublasError(#expression, __FILE__, __LINE__,            \
                                             vllm_ascend_cublas_status);                 \
    }                                                                                    \
  } while (false)

// Non-throwing variant for destructors and teardown paths, matching
// CUDA_CHECK_NOTHROW.
#define CUBLAS_CHECK_NOTHROW(expression)                                                 \
  do {                                                                                   \
    const cublasStatus_t vllm_ascend_cublas_status = (expression);                       \
    if (vllm_ascend_cublas_status != CUBLAS_STATUS_SUCCESS) {                            \
      ::vllm_ascend::test::ReportIgnoredCublasFailure(#expression, __FILE__, __LINE__,   \
                                                      vllm_ascend_cublas_status);        \
    }                                                                                    \
  } while (false)

// RAII cublasHandle_t, bound to `stream` at construction so every GEMM is
// ordered against the same stream the rest of the suite uses and a single
// cudaStreamSynchronize is enough to see the result.
class CUDABlasHandle {
 public:
  explicit CUDABlasHandle(cudaStream_t stream);
  ~CUDABlasHandle();

  CUDABlasHandle(const CUDABlasHandle&) = delete;
  CUDABlasHandle& operator=(const CUDABlasHandle&) = delete;

  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

// Row-major fp16 GEMM:
//     C[m, n] = alpha * op_a(A) * op_b(B) + beta * C
//
// Storage is fp16 and the accumulation is fp32, which is both what the CPU
// reference does and what the v200 cube unit does. The three therefore agree to
// within the fp16 rounding of the output rather than to within the accumulator
// width, which is what makes kFp16DefaultTolerance the right bar.
//
// The flags describe how an operand is *stored*; the helper never materialises
// a transpose:
//   * transpose_b = true  - B is stored [n, k], one output channel per row.
//     That is the Linear weight layout, [out_features, in_features], and the
//     case every projection in the model hits. It becomes CUBLAS_OP_T, which is
//     what torch does rather than transposing the weight.
//   * transpose_b = false - B is stored [k, n].
//   * transpose_a = true  - A is stored [k, m]; false - A is stored [m, k].
//
// cuBLAS is column-major, so none of these map onto its flags directly. See the
// definition in cuda_runtime.cu for the identity that gets from one to the
// other.
void CublasGemmFp16(cublasHandle_t handle, bool transpose_a, bool transpose_b, int64_t m, int64_t n,
                    int64_t k, const uint16_t* a, const uint16_t* b, uint16_t* c, float alpha = 1.0f,
                    float beta = 0.0f);

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

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

  // Shared cuBLAS handle, bound to stream(). Created on first call and
  // destroyed before the stream it is bound to, so a binary that never runs a
  // GEMM never pays for it - cuBLAS reserves workspace and loads its kernel
  // images when the handle is created, not when it is first used.
  cublasHandle_t cublas_handle();

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
  std::unique_ptr<CUDABlasHandle> blas_;
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
