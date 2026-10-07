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

// The shadow descriptor engine: aclCreateTensor and friends over the open
// MockAclTensor metadata, plus the executor address-repointing entry points
// with interval-registry validation.

#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "mock_acl_tensor.hpp"
#include "mock_allocator.hpp"

namespace ascend_moe {
namespace mock {
namespace {

std::mutex g_mutex;  // guards nothing shared today beyond create/destroy order
std::string g_last_failure;

}  // namespace

aclnnStatus MockContractFailure(const std::string& what) {
  const std::lock_guard<std::mutex> lock(g_mutex);
  g_last_failure = what;
  return 100001;  // the mock's invalid-parameter status
}

std::string MockLastContractFailure() {
  const std::lock_guard<std::mutex> lock(g_mutex);
  return g_last_failure;
}

size_t MockDataTypeBytes(aclDataType dtype) {
  switch (dtype) {
    case ACL_BOOL:
      return 1;
    case ACL_FP4X2_E2M1:
      return 1;  // two elements per byte; byte size is halved by the caller
    case ACL_UINT8:
    case ACL_INT8:
    case ACL_FLOAT8_E5M2:
    case ACL_FLOAT8_E4M3FN:
    case ACL_FLOAT8_E8M0:
      return 1;
    case ACL_FLOAT16:
    case ACL_BF16:
    case ACL_INT16:
      return 2;
    case ACL_FLOAT32:
    case ACL_INT32:
      return 4;
    case ACL_INT64:
    case ACL_FLOAT64:
      return 8;
    default:
      return 0;
  }
}

MockAclTensor* AsMockTensor(const aclTensor* tensor) {
  MockAclTensor* mock = reinterpret_cast<MockAclTensor*>(const_cast<aclTensor*>(tensor));
  if (mock == nullptr || mock->magic != kMockAclTensorMagic) {
    MockContractFailure("an aclTensor handle is not a MockAclTensor (bad magic)");
    return nullptr;
  }
  return mock;
}

MockAclScalar* AsMockScalar(const aclScalar* scalar) {
  return reinterpret_cast<MockAclScalar*>(const_cast<aclScalar*>(scalar));
}

MockAclIntArray* AsMockIntArray(const aclIntArray* array) {
  return reinterpret_cast<MockAclIntArray*>(const_cast<aclIntArray*>(array));
}

MockAclTensorList* AsMockTensorList(const aclTensorList* list) {
  return reinterpret_cast<MockAclTensorList*>(const_cast<aclTensorList*>(list));
}

MockAclOpExecutor* AsMockExecutor(aclOpExecutor* executor) { return reinterpret_cast<MockAclOpExecutor*>(executor); }

double MockAclScalar::as_f64() const {
  if (dtype == ACL_FLOAT32) {
    float value = 0.0f;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
  }
  if (dtype == ACL_FLOAT64) {
    double value = 0.0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
  }
  return static_cast<double>(as_i64());
}

int64_t MockAclScalar::as_i64() const {
  int64_t value = 0;
  std::memcpy(&value, bytes, std::min(width, sizeof(value)));
  return value;
}

namespace {

size_t TensorStorageBytes(const std::vector<int64_t>& dims, aclDataType dtype) {
  int64_t elements = 1;
  for (int64_t dim : dims) {
    if (dim < 0) {
      return 0;
    }
    elements *= dim;
  }
  const size_t unit = MockDataTypeBytes(dtype);
  if (unit == 0) {
    return 0;
  }
  if (dtype == ACL_FP4X2_E2M1) {
    // Two E2M1 values share a byte; the layout guarantees even element
    // counts, and an odd tail is a contract violation caught by the caller.
    return static_cast<size_t>((elements + 1) / 2);
  }
  return static_cast<size_t>(elements) * unit;
}

}  // namespace

}  // namespace mock
}  // namespace ascend_moe

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

using namespace ascend_moe::mock;

aclTensor* aclCreateTensor(const int64_t* view_dims, uint64_t view_dims_num, aclDataType dtype,
                           const int64_t* view_strides, int64_t storage_offset, aclFormat format,
                           const int64_t* storage_dims, uint64_t storage_dims_num, void* device_data) {
  (void)view_strides;
  (void)storage_offset;
  (void)storage_dims;
  (void)storage_dims_num;
  if (view_dims == nullptr || view_dims_num == 0) {
    return nullptr;
  }
  auto* tensor = new MockAclTensor();
  tensor->shape.assign(view_dims, view_dims + view_dims_num);
  tensor->dtype = dtype;
  tensor->format = format;
  tensor->total_bytes = TensorStorageBytes(tensor->shape, dtype);
  tensor->device_addr = device_data;
  return reinterpret_cast<aclTensor*>(tensor);
}

aclnnStatus aclDestroyTensor(const aclTensor* tensor) {
  MockAclTensor* mock = reinterpret_cast<MockAclTensor*>(const_cast<aclTensor*>(tensor));
  if (mock == nullptr || mock->magic != kMockAclTensorMagic) {
    return 100001;
  }
  delete mock;
  return 0;
}

aclTensorList* aclCreateTensorList(const aclTensor* const* tensors, uint64_t size) {
  if (tensors == nullptr || size == 0) {
    return nullptr;
  }
  auto* list = new MockAclTensorList();
  list->items.resize(static_cast<size_t>(size));
  for (uint64_t index = 0; index < size; ++index) {
    list->items[static_cast<size_t>(index)] = const_cast<aclTensor*>(tensors[index]);
  }
  return reinterpret_cast<aclTensorList*>(list);
}

aclnnStatus aclDestroyTensorList(const aclTensorList* tensor_list) {
  auto* list = reinterpret_cast<MockAclTensorList*>(const_cast<aclTensorList*>(tensor_list));
  if (list == nullptr) {
    return 100001;
  }
  delete list;
  return 0;
}

aclIntArray* aclCreateIntArray(const int64_t* value, uint64_t size) {
  if (value == nullptr || size == 0) {
    return nullptr;
  }
  auto* array = new MockAclIntArray();
  array->values.assign(value, value + size);
  return reinterpret_cast<aclIntArray*>(array);
}

aclnnStatus aclDestroyIntArray(const aclIntArray* array) {
  auto* mock = reinterpret_cast<MockAclIntArray*>(const_cast<aclIntArray*>(array));
  if (mock == nullptr) {
    return 100001;
  }
  delete mock;
  return 0;
}

aclScalar* aclCreateScalar(void* value, aclDataType dtype) {
  if (value == nullptr) {
    return nullptr;
  }
  auto* scalar = new MockAclScalar();
  scalar->dtype = dtype;
  scalar->width = MockDataTypeBytes(dtype);
  if (scalar->width == 0 || scalar->width > sizeof(scalar->bytes)) {
    scalar->width = std::min<size_t>(scalar->width, sizeof(scalar->bytes));
  }
  std::memcpy(scalar->bytes, value, scalar->width);
  return reinterpret_cast<aclScalar*>(scalar);
}

aclnnStatus aclDestroyScalar(const aclScalar* scalar) {
  auto* mock = reinterpret_cast<MockAclScalar*>(const_cast<aclScalar*>(scalar));
  if (mock == nullptr) {
    return 100001;
  }
  delete mock;
  return 0;
}

aclnnStatus aclSetTensorAddr(aclOpExecutor* executor, uint64_t index, const aclTensor* tensor, void* address) {
  MockAclOpExecutor* exec = AsMockExecutor(executor);
  MockAclTensor* mock = AsMockTensor(tensor);
  if (exec == nullptr || mock == nullptr) {
    return 100001;
  }
  MockNoteSetAddrCheck();
  // Index-vs-handle agreement is the one part of the repeatable-executor
  // protocol no header pins down (the product's slot map is documented as
  // derive-and-verify, and the FIA rope indices are its known open item). A
  // mismatch is COUNTED, not fatal: the address is still validated against
  // the registry and applied to the tensor the caller passed, and the test
  // prints the tally.
  if (index >= exec->tensors.size() || exec->tensors[index] != reinterpret_cast<aclTensor*>(mock)) {
    MockNoteSlotMapMismatch();
  }
  const uintptr_t base = reinterpret_cast<uintptr_t>(address);
  if (!MockSpanContains(base, mock->total_bytes)) {
    std::ostringstream text;
    text << exec->op_name << ": aclSetTensorAddr wants [" << address << ", "
         << reinterpret_cast<void*>(base + mock->total_bytes) << ") for a " << mock->total_bytes << "-byte tensor, "
         << "which is not strictly inside any registered span";
    return MockContractFailure(text.str());
  }
  mock->device_addr = address;
  return 0;
}

aclnnStatus aclSetDynamicTensorAddr(aclOpExecutor* executor, uint64_t ir_index, uint64_t relative_index,
                                    const aclTensorList* tensor_list, void* address) {
  MockAclOpExecutor* exec = AsMockExecutor(executor);
  MockAclTensorList* list = AsMockTensorList(tensor_list);
  if (exec == nullptr || list == nullptr) {
    return 100001;
  }
  if (relative_index >= list->items.size()) {
    std::ostringstream text;
    text << exec->op_name << ": aclSetDynamicTensorAddr relative index " << relative_index
         << " is outside the " << list->items.size() << "-entry list";
    return MockContractFailure(text.str());
  }
  MockAclTensor* mock = AsMockTensor(list->items[relative_index]);
  if (mock == nullptr) {
    return 100001;
  }
  MockNoteSetAddrCheck();
  const uintptr_t base = reinterpret_cast<uintptr_t>(address);
  if (!MockSpanContains(base, mock->total_bytes)) {
    std::ostringstream text;
    text << exec->op_name << ": aclSetDynamicTensorAddr(ir " << ir_index << ", rel " << relative_index
         << ") wants a " << mock->total_bytes << "-byte tensor at " << address
         << ", which is not strictly inside any registered span";
    return MockContractFailure(text.str());
  }
  mock->device_addr = address;
  return 0;
}

aclnnStatus aclSetAclOpExecutorRepeatable(aclOpExecutor*) { return 0; }

aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor* executor) {
  MockAclOpExecutor* exec = AsMockExecutor(executor);
  if (exec == nullptr) {
    return 100001;
  }
  delete exec;
  return 0;
}

}  // extern "C"
