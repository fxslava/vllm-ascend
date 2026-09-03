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

// Typed device buffers for the CUDA backend, the counterpart of
// device_tensor.hpp.
//
// The Ascend DeviceTensor exists to keep an aclTensor descriptor alive
// alongside its allocation. CUDA kernels take raw pointers, so there is no
// descriptor to own here and this type is only about the host conversions: a
// shape, a typed pointer, and the fp16 round trip in one place rather than
// spelled out at every call site.
//
// fp16 storage is uint16_t throughout, the same convention fp16.hpp uses. The
// host never needs a half type, and the kernels reinterpret the pointer as
// __half on the device side.

#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "cuda_runtime.hpp"
#include "fp16.hpp"

namespace vllm_ascend {
namespace test {

inline size_t CudaElementCount(const std::vector<int64_t>& dims) {
  size_t count = 1;
  for (int64_t dim : dims) {
    count *= static_cast<size_t>(dim);
  }
  return dims.empty() ? 0 : count;
}

class CudaDeviceTensor {
 public:
  CudaDeviceTensor() = default;

  CudaDeviceTensor(CudaDeviceTensor&&) = default;
  CudaDeviceTensor& operator=(CudaDeviceTensor&&) = default;
  CudaDeviceTensor(const CudaDeviceTensor&) = delete;
  CudaDeviceTensor& operator=(const CudaDeviceTensor&) = delete;

  const std::vector<int64_t>& dims() const { return dims_; }
  size_t element_count() const { return element_count_; }
  void* data() const { return buffer_.get(); }

  // Typed views of the same allocation, for passing straight to a launcher.
  uint16_t* half_data() const { return static_cast<uint16_t*>(buffer_.get()); }
  float* float_data() const { return static_cast<float*>(buffer_.get()); }
  int32_t* int32_data() const { return static_cast<int32_t*>(buffer_.get()); }

  // --- fp16 -----------------------------------------------------------------

  // Uploads float values, rounding each to fp16 on the way. Pass values that
  // are already exactly representable in fp16 when the test needs the device
  // and the reference to start from identical bits.
  static CudaDeviceTensor Half(const std::vector<int64_t>& dims, const std::vector<float>& values,
                               size_t alignment = kCudaDeviceAlignBytes) {
    std::vector<uint16_t> bits(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      bits[i] = FloatToHalfBits(values[i]);
    }
    return CudaDeviceTensor(dims, sizeof(uint16_t), bits.data(), alignment);
  }

  static CudaDeviceTensor HalfEmpty(const std::vector<int64_t>& dims,
                                    size_t alignment = kCudaDeviceAlignBytes) {
    return CudaDeviceTensor(dims, sizeof(uint16_t), nullptr, alignment);
  }

  std::vector<float> ToFloatFromHalf() const {
    std::vector<uint16_t> bits(element_count_);
    if (!bits.empty()) {
      buffer_.CopyToHost(bits.data(), bits.size() * sizeof(uint16_t));
    }
    std::vector<float> values(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) {
      values[i] = HalfBitsToFloat(bits[i]);
    }
    return values;
  }

  // --- fp32 -----------------------------------------------------------------

  static CudaDeviceTensor Float(const std::vector<int64_t>& dims, const std::vector<float>& values,
                                size_t alignment = kCudaDeviceAlignBytes) {
    return CudaDeviceTensor(dims, sizeof(float), values.data(), alignment);
  }

  static CudaDeviceTensor FloatEmpty(const std::vector<int64_t>& dims,
                                     size_t alignment = kCudaDeviceAlignBytes) {
    return CudaDeviceTensor(dims, sizeof(float), nullptr, alignment);
  }

  std::vector<float> ToFloat() const {
    std::vector<float> values(element_count_);
    if (!values.empty()) {
      buffer_.CopyToHost(values.data(), values.size() * sizeof(float));
    }
    return values;
  }

  // --- int32 ----------------------------------------------------------------

  static CudaDeviceTensor Int32(const std::vector<int64_t>& dims, const std::vector<int32_t>& values,
                                size_t alignment = kCudaDeviceAlignBytes) {
    return CudaDeviceTensor(dims, sizeof(int32_t), values.data(), alignment);
  }

  std::vector<int32_t> ToInt32() const {
    std::vector<int32_t> values(element_count_);
    if (!values.empty()) {
      buffer_.CopyToHost(values.data(), values.size() * sizeof(int32_t));
    }
    return values;
  }

 private:
  CudaDeviceTensor(std::vector<int64_t> dims, size_t element_size, const void* host_data, size_t alignment)
      : dims_(std::move(dims)) {
    element_count_ = CudaElementCount(dims_);
    buffer_.Allocate(element_count_ * element_size, alignment);
    if (host_data != nullptr && element_count_ != 0) {
      buffer_.CopyFromHost(host_data, element_count_ * element_size);
    }
  }

  CUDADeviceBuffer buffer_;
  std::vector<int64_t> dims_;
  size_t element_count_ = 0;
};

}  // namespace test
}  // namespace vllm_ascend
