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

// Torch-free access to the CANN single-operator API (aclnn).
//
// Why dlsym instead of #include <aclnnop/...> plus -lopapi:
//
//   * The operator set in libopapi.so moves between CANN releases. Linking
//     directly turns a missing operator into a link error for the whole binary;
//     here it becomes a GTEST_SKIP naming the exact symbol, and the other three
//     kernel suites still run.
//   * It mirrors what torch_npu itself does. csrc/aclnn_torch_adapter/op_api_common.h
//     resolves every aclnn entry point through dlopen of libopapi.so plus dlsym;
//     this file is the same mechanism with the ATen dependencies removed.
//
// The cost is that the argument list of each GetWorkspaceSize function is
// declared by hand in aclnn_ops.hpp rather than checked by the compiler. Every
// declaration there names the CANN header to verify it against, and a mismatch
// surfaces as a non-zero status plus the aclGetRecentErrMsg text rather than as
// silent corruption.

#pragma once

#include <acl/acl.h>
#include <acl/acl_base.h>

#include <cstdint>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "device_buffer.hpp"

// Opaque aclnn handle types, matching the forward declarations in
// csrc/aclnn_torch_adapter/op_api_common.h.
typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;

namespace vllm_ascend {
namespace test {

// Process-wide handle to libopapi.so and the constructor/destructor entry
// points it re-exports from libnnopbase.so.
class OpApiLibrary {
 public:
  static OpApiLibrary& Instance();

  bool loaded() const { return handle_ != nullptr; }
  const std::string& load_error() const { return load_error_; }

  // Returns nullptr when the symbol is absent; callers decide whether that is
  // a skip or a failure.
  void* Resolve(const char* symbol) const;

  aclTensor* CreateTensor(const std::vector<int64_t>& view_dims, const std::vector<int64_t>& strides,
                          int64_t offset, aclDataType dtype, aclFormat format,
                          const std::vector<int64_t>& storage_dims, void* data) const;
  void DestroyTensor(const aclTensor* tensor) const;

  aclIntArray* CreateIntArray(const int64_t* values, uint64_t size) const;
  void DestroyIntArray(const aclIntArray* array) const;

  aclTensorList* CreateTensorList(const aclTensor* const* tensors, uint64_t size) const;
  void DestroyTensorList(const aclTensorList* list) const;

 private:
  OpApiLibrary();

  void* handle_ = nullptr;
  std::string load_error_;
  std::string library_path_;
};

// Row-major contiguous strides, in elements, for the given shape.
std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& dims);

size_t ElementCount(const std::vector<int64_t>& dims);

// RAII aclTensor. Owns the descriptor only; the device memory belongs to the
// DeviceBuffer that was passed in.
class AclnnTensor {
 public:
  AclnnTensor() = default;

  // Contiguous ND tensor over the given device pointer.
  AclnnTensor(std::vector<int64_t> dims, aclDataType dtype, void* data, aclFormat format = ACL_FORMAT_ND);

  // Explicit strides, for views such as one half of a SwiGLU input.
  AclnnTensor(std::vector<int64_t> dims, std::vector<int64_t> strides, int64_t offset, aclDataType dtype,
              aclFormat format, std::vector<int64_t> storage_dims, void* data);

  ~AclnnTensor();

  AclnnTensor(const AclnnTensor&) = delete;
  AclnnTensor& operator=(const AclnnTensor&) = delete;
  AclnnTensor(AclnnTensor&& other) noexcept;
  AclnnTensor& operator=(AclnnTensor&& other) noexcept;

  aclTensor* get() const { return tensor_; }
  operator aclTensor*() const { return tensor_; }
  const std::vector<int64_t>& dims() const { return dims_; }

 private:
  void Release();

  aclTensor* tensor_ = nullptr;
  std::vector<int64_t> dims_;
};

class AclnnIntArray {
 public:
  explicit AclnnIntArray(std::vector<int64_t> values);
  ~AclnnIntArray();

  AclnnIntArray(const AclnnIntArray&) = delete;
  AclnnIntArray& operator=(const AclnnIntArray&) = delete;

  aclIntArray* get() const { return array_; }
  operator aclIntArray*() const { return array_; }

 private:
  std::vector<int64_t> values_;
  aclIntArray* array_ = nullptr;
};

class AclnnTensorList {
 public:
  explicit AclnnTensorList(const std::vector<const aclTensor*>& tensors);
  ~AclnnTensorList();

  AclnnTensorList(const AclnnTensorList&) = delete;
  AclnnTensorList& operator=(const AclnnTensorList&) = delete;

  aclTensorList* get() const { return list_; }
  operator aclTensorList*() const { return list_; }

 private:
  aclTensorList* list_ = nullptr;
};

// A resolved aclnn operator: the two-phase GetWorkspaceSize / launch pair.
class AclnnOp {
 public:
  explicit AclnnOp(const char* name);

  bool available() const { return get_workspace_size_ != nullptr && launch_ != nullptr; }
  const std::string& name() const { return name_; }

  // Human-readable explanation of why available() is false.
  std::string unavailable_reason() const;

  void* get_workspace_size_fn() const { return get_workspace_size_; }
  void* launch_fn() const { return launch_; }

 private:
  std::string name_;
  void* get_workspace_size_ = nullptr;
  void* launch_ = nullptr;
};

// Launch phase, shared by every aclnn operator.
using AclnnLaunchFn = int (*)(void* workspace, uint64_t workspace_size, aclOpExecutor* executor,
                              aclrtStream stream);

// Runs the two-phase call: resolve workspace size, allocate it, launch, then
// synchronise. WorkspaceSizeFn is the operator-specific function-pointer type
// declared in aclnn_ops.hpp; args are the operator arguments up to but not
// including the trailing workspaceSize and executor out-parameters.
//
// Throws AclError on any non-zero status, with the CANN diagnostic attached.
template <typename WorkspaceSizeFn, typename... Args>
void RunAclnn(const AclnnOp& op, aclrtStream stream, Args... args) {
  if (!op.available()) {
    throw AclError(op.unavailable_reason().c_str(), __FILE__, __LINE__, -1);
  }

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  auto* workspace_size_fn = reinterpret_cast<WorkspaceSizeFn>(op.get_workspace_size_fn());
  const int plan_status = workspace_size_fn(args..., &workspace_size, &executor);
  if (plan_status != 0) {
    const std::string label = op.name() + "GetWorkspaceSize";
    throw AclError(label.c_str(), __FILE__, __LINE__, plan_status);
  }

  DeviceBuffer workspace;
  if (workspace_size > 0) {
    workspace.Allocate(static_cast<size_t>(workspace_size));
  }

  auto* launch_fn = reinterpret_cast<AclnnLaunchFn>(op.launch_fn());
  const int launch_status = launch_fn(workspace.get(), workspace_size, executor, stream);
  if (launch_status != 0) {
    throw AclError(op.name().c_str(), __FILE__, __LINE__, launch_status);
  }

  ACL_CHECK(aclrtSynchronizeStream(stream));
}

// Skips the current test when the operator could not be resolved. Kept as a
// macro so that GTEST_SKIP lands in the test body.
#define REQUIRE_ACLNN_OP(op)                     \
  do {                                           \
    if (!(op).available()) {                     \
      GTEST_SKIP() << (op).unavailable_reason(); \
    }                                            \
  } while (false)

}  // namespace test
}  // namespace vllm_ascend
