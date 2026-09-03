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

// Error-checking helpers for the CUDA runtime, mirroring acl_check.hpp one for
// one so a reader who knows the Ascend path recognises the shape of this one.
//
//   * CUDA_CHECK          - throws CudaError. Use it inside RAII types and
//                           helpers where a GTest fatal assertion cannot unwind.
//   * CUDA_CHECK_NOTHROW  - reports and continues. For destructors and teardown.
//   * ASSERT_CUDA_OK      - GTest fatal assertion, for use in test bodies.
//
// Only <cuda_runtime_api.h> is included: it is a plain C header, so this file
// compiles under the host compiler in a .cpp translation unit that nvcc never
// sees. Nothing here needs the device-side language extensions.

#pragma once

#include <cuda_runtime_api.h>

#include <exception>
#include <sstream>
#include <string>

namespace vllm_ascend {
namespace test {

// cudaGetErrorString is the CUDA equivalent of aclGetRecentErrMsg: the numeric
// status on its own is rarely enough to identify which call went wrong.
inline std::string FormatCudaError(const char* expression, const char* file, int line, cudaError_t status) {
  std::ostringstream stream;
  stream << file << ":" << line << ": " << expression << " failed with status " << static_cast<int>(status)
         << "\n  CUDA: " << cudaGetErrorName(status) << ": " << cudaGetErrorString(status);
  return stream.str();
}

class CudaError : public std::exception {
 public:
  CudaError(const char* expression, const char* file, int line, cudaError_t status)
      : message_(FormatCudaError(expression, file, line, status)), status_(status) {}

  const char* what() const noexcept override { return message_.c_str(); }
  cudaError_t status() const { return status_; }

 private:
  std::string message_;
  cudaError_t status_;
};

void ReportIgnoredCudaFailure(const char* expression, const char* file, int line, cudaError_t status);

#define CUDA_CHECK(expression)                                                           \
  do {                                                                                   \
    const cudaError_t vllm_ascend_cuda_status = (expression);                            \
    if (vllm_ascend_cuda_status != cudaSuccess) {                                        \
      throw ::vllm_ascend::test::CudaError(#expression, __FILE__, __LINE__,              \
                                           vllm_ascend_cuda_status);                     \
    }                                                                                    \
  } while (false)

// Non-throwing variant for destructors and teardown paths, where an exception
// would terminate the process and hide the original failure.
#define CUDA_CHECK_NOTHROW(expression)                                                   \
  do {                                                                                   \
    const cudaError_t vllm_ascend_cuda_status = (expression);                            \
    if (vllm_ascend_cuda_status != cudaSuccess) {                                        \
      ::vllm_ascend::test::ReportIgnoredCudaFailure(#expression, __FILE__, __LINE__,     \
                                                    vllm_ascend_cuda_status);            \
    }                                                                                    \
  } while (false)

#define ASSERT_CUDA_OK(expression)                                                       \
  do {                                                                                   \
    const cudaError_t vllm_ascend_cuda_status = (expression);                            \
    ASSERT_EQ(vllm_ascend_cuda_status, cudaSuccess)                                      \
        << ::vllm_ascend::test::FormatCudaError(#expression, __FILE__, __LINE__,         \
                                                vllm_ascend_cuda_status);                \
  } while (false)

#define EXPECT_CUDA_OK(expression)                                                       \
  do {                                                                                   \
    const cudaError_t vllm_ascend_cuda_status = (expression);                            \
    EXPECT_EQ(vllm_ascend_cuda_status, cudaSuccess)                                      \
        << ::vllm_ascend::test::FormatCudaError(#expression, __FILE__, __LINE__,         \
                                                vllm_ascend_cuda_status);                \
  } while (false)

}  // namespace test
}  // namespace vllm_ascend
