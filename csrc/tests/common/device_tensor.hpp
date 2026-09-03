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

// Pairs a device allocation with the aclTensor descriptor that points at it.
//
// The two have to live and die together: an aclTensor whose backing buffer has
// been freed is the classic way to turn a test failure into an unrelated
// EXCEPTION further down the run. Bundling them makes the lifetime obvious and
// keeps the host conversions in one place.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "aclnn_runtime.hpp"
#include "device_buffer.hpp"
#include "fp16.hpp"

namespace vllm_ascend {
namespace test {

class DeviceTensor {
 public:
  DeviceTensor() = default;

  DeviceTensor(DeviceTensor&&) = default;
  DeviceTensor& operator=(DeviceTensor&&) = default;
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;

  aclTensor* get() const { return tensor_.get(); }
  operator aclTensor*() const { return tensor_.get(); }
  const std::vector<int64_t>& dims() const { return dims_; }
  size_t element_count() const { return element_count_; }
  void* data() const { return buffer_.get(); }

  // --- fp16 -----------------------------------------------------------------

  // Uploads float values, rounding each to fp16 on the way. Pass values that
  // are already exactly representable in fp16 when the test needs the device
  // and the reference to start from identical bits.
  static DeviceTensor Half(const std::vector<int64_t>& dims, const std::vector<float>& values,
                           aclFormat format = ACL_FORMAT_ND);
  static DeviceTensor HalfEmpty(const std::vector<int64_t>& dims, aclFormat format = ACL_FORMAT_ND);
  std::vector<float> ToFloatFromHalf() const;

  // A [k, n] tensor whose storage is [n, k] row-major, described with strides
  // {1, k} instead of being copied. This is the weight layout a Linear layer
  // already holds ([out_features, in_features]), so a matmul test can pass a
  // transposed B the same way the plugin does, without a host-side transpose.
  //
  // `values` is the [n, k] buffer, n * k elements.
  static DeviceTensor HalfTransposed2D(int64_t n, int64_t k, const std::vector<float>& values);

  // --- fp32 -----------------------------------------------------------------

  static DeviceTensor Float(const std::vector<int64_t>& dims, const std::vector<float>& values,
                            aclFormat format = ACL_FORMAT_ND);
  static DeviceTensor FloatEmpty(const std::vector<int64_t>& dims, aclFormat format = ACL_FORMAT_ND);
  std::vector<float> ToFloat() const;

  // --- int32 ----------------------------------------------------------------

  static DeviceTensor Int32(const std::vector<int64_t>& dims, const std::vector<int32_t>& values,
                            aclFormat format = ACL_FORMAT_ND);
  std::vector<int32_t> ToInt32() const;

 private:
  DeviceTensor(std::vector<int64_t> dims, aclDataType dtype, aclFormat format, size_t element_size,
               const void* host_data);

  DeviceBuffer buffer_;
  AclnnTensor tensor_;
  std::vector<int64_t> dims_;
  size_t element_count_ = 0;
};

inline DeviceTensor::DeviceTensor(std::vector<int64_t> dims, aclDataType dtype, aclFormat format,
                                  size_t element_size, const void* host_data)
    : dims_(std::move(dims)) {
  element_count_ = ElementCount(dims_);
  buffer_.Allocate(element_count_ * element_size);
  if (host_data != nullptr) {
    buffer_.CopyFromHost(host_data, element_count_ * element_size);
  }
  tensor_ = AclnnTensor(dims_, dtype, buffer_.get(), format);
}

inline DeviceTensor DeviceTensor::Half(const std::vector<int64_t>& dims, const std::vector<float>& values,
                                       aclFormat format) {
  const std::vector<uint16_t> bits = [&values]() {
    std::vector<uint16_t> converted(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      converted[i] = FloatToHalfBits(values[i]);
    }
    return converted;
  }();
  return DeviceTensor(dims, ACL_FLOAT16, format, sizeof(uint16_t), bits.data());
}

inline DeviceTensor DeviceTensor::HalfEmpty(const std::vector<int64_t>& dims, aclFormat format) {
  return DeviceTensor(dims, ACL_FLOAT16, format, sizeof(uint16_t), nullptr);
}

inline DeviceTensor DeviceTensor::HalfTransposed2D(int64_t n, int64_t k, const std::vector<float>& values) {
  const size_t count = static_cast<size_t>(n) * static_cast<size_t>(k);
  assert(values.size() == count);

  std::vector<uint16_t> bits(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    bits[i] = FloatToHalfBits(values[i]);
  }

  DeviceTensor tensor;
  // dims_ is the logical view the operator sees; the allocation is the same
  // n * k elements either way.
  tensor.dims_ = {k, n};
  tensor.element_count_ = count;
  tensor.buffer_.Allocate(count * sizeof(uint16_t));
  tensor.buffer_.CopyFromHost(bits.data(), bits.size() * sizeof(uint16_t));
  tensor.tensor_ = AclnnTensor(/*dims=*/{k, n}, /*strides=*/{1, k}, /*offset=*/0, ACL_FLOAT16, ACL_FORMAT_ND,
                               /*storage_dims=*/{n, k}, tensor.buffer_.get());
  return tensor;
}

inline std::vector<float> DeviceTensor::ToFloatFromHalf() const {
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

inline DeviceTensor DeviceTensor::Float(const std::vector<int64_t>& dims, const std::vector<float>& values,
                                        aclFormat format) {
  return DeviceTensor(dims, ACL_FLOAT, format, sizeof(float), values.data());
}

inline DeviceTensor DeviceTensor::FloatEmpty(const std::vector<int64_t>& dims, aclFormat format) {
  return DeviceTensor(dims, ACL_FLOAT, format, sizeof(float), nullptr);
}

inline std::vector<float> DeviceTensor::ToFloat() const {
  std::vector<float> values(element_count_);
  if (!values.empty()) {
    buffer_.CopyToHost(values.data(), values.size() * sizeof(float));
  }
  return values;
}

inline DeviceTensor DeviceTensor::Int32(const std::vector<int64_t>& dims, const std::vector<int32_t>& values,
                                        aclFormat format) {
  return DeviceTensor(dims, ACL_INT32, format, sizeof(int32_t), values.data());
}

inline std::vector<int32_t> DeviceTensor::ToInt32() const {
  std::vector<int32_t> values(element_count_);
  if (!values.empty()) {
    buffer_.CopyToHost(values.data(), values.size() * sizeof(int32_t));
  }
  return values;
}

}  // namespace test
}  // namespace vllm_ascend
