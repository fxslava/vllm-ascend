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

#include "aclnn_runtime.hpp"

#include <dlfcn.h>

#include <cstdlib>
#include <sstream>

namespace vllm_ascend {
namespace test {

namespace {

using CreateTensorFn = aclTensor* (*)(const int64_t* view_dims, uint64_t view_dims_num, aclDataType dtype,
                                      const int64_t* stride, int64_t offset, aclFormat format,
                                      const int64_t* storage_dims, uint64_t storage_dims_num, void* data);
using DestroyTensorFn = int (*)(const aclTensor* tensor);
using CreateIntArrayFn = aclIntArray* (*)(const int64_t* values, uint64_t size);
using DestroyIntArrayFn = int (*)(const aclIntArray* array);
using CreateTensorListFn = aclTensorList* (*)(const aclTensor* const* tensors, uint64_t size);
using DestroyTensorListFn = int (*)(const aclTensorList* list);

// libopapi.so lives in the CANN op-api lib directory, which the standard
// set_env.sh puts on LD_LIBRARY_PATH. The absolute candidates cover the case
// where a test binary is launched without sourcing it.
std::vector<std::string> OpApiCandidatePaths() {
  std::vector<std::string> candidates;
  candidates.emplace_back("libopapi.so");

  const char* ascend_home = std::getenv("ASCEND_HOME_PATH");
  if (ascend_home == nullptr || ascend_home[0] == '\0') {
    ascend_home = std::getenv("ASCEND_TOOLKIT_HOME");
  }
  if (ascend_home != nullptr && ascend_home[0] != '\0') {
    const std::string home(ascend_home);
    candidates.push_back(home + "/lib64/libopapi.so");
    candidates.push_back(home + "/lib64/stub/libopapi.so");
  }
  return candidates;
}

}  // namespace

OpApiLibrary::OpApiLibrary() {
  std::ostringstream failures;
  for (const std::string& candidate : OpApiCandidatePaths()) {
    // RTLD_LAZY matches op_api_common.h; RTLD_GLOBAL lets dlsym on this handle
    // reach the aclCreateTensor family that libnnopbase.so provides.
    handle_ = dlopen(candidate.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (handle_ != nullptr) {
      library_path_ = candidate;
      return;
    }
    const char* error = dlerror();
    failures << "\n  " << candidate << ": " << (error != nullptr ? error : "unknown dlopen error");
  }
  load_error_ = "failed to load libopapi.so; source the CANN set_env.sh first. Tried:" + failures.str();
}

OpApiLibrary& OpApiLibrary::Instance() {
  static OpApiLibrary instance;
  return instance;
}

void* OpApiLibrary::Resolve(const char* symbol) const {
  if (handle_ == nullptr) {
    return nullptr;
  }
  dlerror();  // clear any stale error before the lookup
  return dlsym(handle_, symbol);
}

aclTensor* OpApiLibrary::CreateTensor(const std::vector<int64_t>& view_dims, const std::vector<int64_t>& strides,
                                      int64_t offset, aclDataType dtype, aclFormat format,
                                      const std::vector<int64_t>& storage_dims, void* data) const {
  auto* create = reinterpret_cast<CreateTensorFn>(Resolve("aclCreateTensor"));
  if (create == nullptr) {
    throw AclError("aclCreateTensor not found in libopapi.so / libnnopbase.so", __FILE__, __LINE__, -1);
  }
  aclTensor* tensor =
      create(view_dims.data(), static_cast<uint64_t>(view_dims.size()), dtype, strides.data(), offset, format,
             storage_dims.data(), static_cast<uint64_t>(storage_dims.size()), data);
  if (tensor == nullptr) {
    throw AclError("aclCreateTensor returned nullptr", __FILE__, __LINE__, -1);
  }
  return tensor;
}

void OpApiLibrary::DestroyTensor(const aclTensor* tensor) const {
  if (tensor == nullptr) {
    return;
  }
  auto* destroy = reinterpret_cast<DestroyTensorFn>(Resolve("aclDestroyTensor"));
  if (destroy != nullptr) {
    destroy(tensor);
  }
}

aclIntArray* OpApiLibrary::CreateIntArray(const int64_t* values, uint64_t size) const {
  auto* create = reinterpret_cast<CreateIntArrayFn>(Resolve("aclCreateIntArray"));
  if (create == nullptr) {
    throw AclError("aclCreateIntArray not found in libopapi.so / libnnopbase.so", __FILE__, __LINE__, -1);
  }
  aclIntArray* array = create(values, size);
  if (array == nullptr) {
    throw AclError("aclCreateIntArray returned nullptr", __FILE__, __LINE__, -1);
  }
  return array;
}

void OpApiLibrary::DestroyIntArray(const aclIntArray* array) const {
  if (array == nullptr) {
    return;
  }
  auto* destroy = reinterpret_cast<DestroyIntArrayFn>(Resolve("aclDestroyIntArray"));
  if (destroy != nullptr) {
    destroy(array);
  }
}

aclTensorList* OpApiLibrary::CreateTensorList(const aclTensor* const* tensors, uint64_t size) const {
  auto* create = reinterpret_cast<CreateTensorListFn>(Resolve("aclCreateTensorList"));
  if (create == nullptr) {
    throw AclError("aclCreateTensorList not found in libopapi.so / libnnopbase.so", __FILE__, __LINE__, -1);
  }
  aclTensorList* list = create(tensors, size);
  if (list == nullptr) {
    throw AclError("aclCreateTensorList returned nullptr", __FILE__, __LINE__, -1);
  }
  return list;
}

void OpApiLibrary::DestroyTensorList(const aclTensorList* list) const {
  if (list == nullptr) {
    return;
  }
  auto* destroy = reinterpret_cast<DestroyTensorListFn>(Resolve("aclDestroyTensorList"));
  if (destroy != nullptr) {
    destroy(list);
  }
}

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& dims) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (size_t i = dims.size(); i-- > 1;) {
    strides[i - 1] = strides[i] * dims[i];
  }
  return strides;
}

size_t ElementCount(const std::vector<int64_t>& dims) {
  size_t count = 1;
  for (int64_t dim : dims) {
    count *= static_cast<size_t>(dim);
  }
  return count;
}

AclnnTensor::AclnnTensor(std::vector<int64_t> dims, aclDataType dtype, void* data, aclFormat format)
    : dims_(std::move(dims)) {
  const std::vector<int64_t> strides = ContiguousStrides(dims_);
  tensor_ = OpApiLibrary::Instance().CreateTensor(dims_, strides, 0, dtype, format, dims_, data);
}

AclnnTensor::AclnnTensor(std::vector<int64_t> dims, std::vector<int64_t> strides, int64_t offset, aclDataType dtype,
                         aclFormat format, std::vector<int64_t> storage_dims, void* data)
    : dims_(std::move(dims)) {
  tensor_ = OpApiLibrary::Instance().CreateTensor(dims_, strides, offset, dtype, format, storage_dims, data);
}

AclnnTensor::~AclnnTensor() { Release(); }

AclnnTensor::AclnnTensor(AclnnTensor&& other) noexcept : tensor_(other.tensor_), dims_(std::move(other.dims_)) {
  other.tensor_ = nullptr;
}

AclnnTensor& AclnnTensor::operator=(AclnnTensor&& other) noexcept {
  if (this != &other) {
    Release();
    tensor_ = other.tensor_;
    dims_ = std::move(other.dims_);
    other.tensor_ = nullptr;
  }
  return *this;
}

void AclnnTensor::Release() {
  if (tensor_ != nullptr) {
    OpApiLibrary::Instance().DestroyTensor(tensor_);
    tensor_ = nullptr;
  }
}

AclnnIntArray::AclnnIntArray(std::vector<int64_t> values) : values_(std::move(values)) {
  array_ = OpApiLibrary::Instance().CreateIntArray(values_.data(), static_cast<uint64_t>(values_.size()));
}

AclnnIntArray::~AclnnIntArray() { OpApiLibrary::Instance().DestroyIntArray(array_); }

AclnnTensorList::AclnnTensorList(const std::vector<const aclTensor*>& tensors) {
  list_ = OpApiLibrary::Instance().CreateTensorList(tensors.data(), static_cast<uint64_t>(tensors.size()));
}

AclnnTensorList::~AclnnTensorList() { OpApiLibrary::Instance().DestroyTensorList(list_); }

AclnnOp::AclnnOp(const char* name) : name_(name) {
  const OpApiLibrary& library = OpApiLibrary::Instance();
  const std::string workspace_symbol = name_ + "GetWorkspaceSize";
  get_workspace_size_ = library.Resolve(workspace_symbol.c_str());
  launch_ = library.Resolve(name_.c_str());
}

std::string AclnnOp::unavailable_reason() const {
  if (available()) {
    return std::string();
  }
  const OpApiLibrary& library = OpApiLibrary::Instance();
  if (!library.loaded()) {
    return "aclnn operator " + name_ + " unavailable: " + library.load_error();
  }
  std::ostringstream reason;
  reason << "aclnn operator " << name_ << " is not exported by this CANN install (";
  reason << name_ << "GetWorkspaceSize=" << (get_workspace_size_ != nullptr ? "found" : "missing");
  reason << ", " << name_ << "=" << (launch_ != nullptr ? "found" : "missing") << "). ";
  reason << "Check the operator name against " << "$ASCEND_HOME_PATH/include/aclnnop/.";
  return reason.str();
}

}  // namespace test
}  // namespace vllm_ascend
