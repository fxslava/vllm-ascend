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

// The open metadata behind the opaque aclTensor pointer.
//
// In the real runtime an aclTensor is an opaque handle whose layout the ACLNN
// libraries own. Here the handle IS this struct: every GetWorkspaceSize stub
// in mock_ops.cpp reads shape / strides / dtype straight off it, which is what
// makes the contract assertions possible without a device, and what lets
// aclSetTensorAddr validate [addr, addr + total_bytes) against the interval
// registry before accepting a new address.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "aclnn/acl_meta.h"

namespace vllm_ascend {
namespace dsv4 {
namespace mock {

inline constexpr uint32_t kMockAclTensorMagic = 0xAC17E450;

// The one allocation the mock allows itself: descriptor metadata. Weights,
// slots and arenas never get physical memory (see mock_allocator.hpp).
struct MockAclTensor {
  uint32_t magic = kMockAclTensorMagic;
  void* device_addr = nullptr;  // symbolic until aclSetTensorAddr validates it
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  aclDataType dtype = ACL_FLOAT32;
  aclFormat format = ACL_FORMAT_ND;
  size_t total_bytes = 0;  // product(shape) * element size (FP4: 2 per byte)

  int64_t elements() const {
    int64_t count = 1;
    for (int64_t dim : shape) {
      count *= dim;
    }
    return count;
  }
  int64_t dim(size_t index) const { return shape.at(index); }
  bool same_shape_as(const MockAclTensor& other) const { return shape == other.shape; }
};

struct MockAclScalar {
  aclDataType dtype = ACL_FLOAT32;
  uint8_t bytes[8] = {};
  size_t width = 0;

  double as_f64() const;
  int64_t as_i64() const;
};

struct MockAclIntArray {
  std::vector<int64_t> values;
};

struct MockAclTensorList {
  std::vector<aclTensor*> items;
};

// What one planned operator invocation captured at GetWorkspaceSize time.
// Execution stubs are no-ops, so this exists only so aclSetTensorAddr /
// aclSetDynamicTensorAddr can find the tensor a slot index refers to and
// repoint it -- the same handle-indirection the real repeatable-executor
// protocol uses.
struct MockAclOpExecutor {
  const char* op_name = "<none>";
  // Tensor arguments in IR order. List arguments are flattened, with
  // `list_spans` remembering [ir_index, begin, end) so (ir_index,
  // relative_index) resolves back to one entry.
  std::vector<aclTensor*> tensors;
  struct ListSpan {
    size_t ir_index;
    size_t begin;
    size_t end;
  };
  std::vector<ListSpan> list_spans;
  uint64_t workspace_size = 0;
};

// Cast helpers with the magic check: a wrong pointer is a loud failure, never
// a silent reinterpretation.
MockAclTensor* AsMockTensor(const aclTensor* tensor);
MockAclScalar* AsMockScalar(const aclScalar* scalar);
MockAclIntArray* AsMockIntArray(const aclIntArray* array);
MockAclTensorList* AsMockTensorList(const aclTensorList* list);
MockAclOpExecutor* AsMockExecutor(aclOpExecutor* executor);

// Bytes one element of `dtype` occupies in storage (FP4 packs two per byte).
size_t MockDataTypeBytes(aclDataType dtype);

// Shared error reporting for the contract validators: records the reason and
// returns the mock's invalid-parameter status. Every stub funnels through
// this so a refusal names the violated contract.
aclnnStatus MockContractFailure(const std::string& what);

// The most recent contract-failure text (for tests that assert refusals).
std::string MockLastContractFailure();

}  // namespace mock
}  // namespace dsv4
}  // namespace vllm_ascend
