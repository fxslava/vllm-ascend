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
#include <fstream>
#include <sstream>
#include <string>

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
using CreateScalarFn = aclScalar* (*)(void* value, aclDataType dtype);
using DestroyScalarFn = int (*)(const aclScalar* scalar);

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
    candidates.push_back(home + "/aarch64-linux/lib64/libopapi.so");
    candidates.push_back(home + "/x86_64-linux/lib64/libopapi.so");
  }
  return candidates;
}

std::string EnvOrEmpty(const char* name) {
  const char* value = std::getenv(name);
  return (value != nullptr) ? std::string(value) : std::string();
}

std::vector<std::string> SplitOn(const std::string& text, char separator) {
  std::vector<std::string> parts;
  std::string current;
  std::istringstream stream(text);
  while (std::getline(stream, current, separator)) {
    const size_t first = current.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
      continue;
    }
    const size_t last = current.find_last_not_of(" \t\r\n");
    parts.push_back(current.substr(first, last - first + 1));
  }
  return parts;
}

std::vector<std::string> CustomOpApiCandidatePaths() {
  std::vector<std::string> candidates;

  const std::string custom_opp_path = EnvOrEmpty("ASCEND_CUSTOM_OPP_PATH");
  for (const std::string& entry : SplitOn(custom_opp_path, ':')) {
    candidates.push_back(entry + "/op_api/lib/libcust_opapi.so");
  }

  const std::string opp_path = EnvOrEmpty("ASCEND_OPP_PATH");
  if (opp_path.empty()) {
    return candidates;
  }

  const std::string vendors_path = opp_path + "/vendors";
  std::ifstream config((vendors_path + "/config.ini").c_str());
  if (!config) {
    return candidates;
  }

  std::string line;
  const std::string key = "load_priority=";
  while (std::getline(config, line)) {
    if (line.rfind(key, 0) != 0) {
      continue;
    }
    for (const std::string& vendor : SplitOn(line.substr(key.size()), ',')) {
      candidates.push_back(vendors_path + "/" + vendor + "/op_api/lib/libcust_opapi.so");
    }
    break;
  }
  return candidates;
}

}

OpApiLibrary::OpApiLibrary() {
  for (const std::string& candidate : CustomOpApiCandidatePaths()) {
    void* custom = dlopen(candidate.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (custom != nullptr) {
      custom_handles_.push_back(custom);
      custom_paths_.push_back(candidate);
    } else {
      dlerror();
    }
  }

  std::ostringstream failures;
  for (const std::string& candidate : OpApiCandidatePaths()) {
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

void* OpApiLibrary::Resolve(const char* symbol, std::string* source) const {
  dlerror();

  for (size_t i = 0; i < custom_handles_.size(); ++i) {
    void* address = dlsym(custom_handles_[i], symbol);
    if (address != nullptr) {
      if (source != nullptr) {
        *source = custom_paths_[i];
      }
      return address;
    }
    dlerror();
  }

  if (handle_ == nullptr) {
    return nullptr;
  }
  void* address = dlsym(handle_, symbol);
  if (address != nullptr && source != nullptr) {
    *source = library_path_;
  }
  return address;
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

aclScalar* OpApiLibrary::CreateScalar(void* value, aclDataType dtype) const {
  auto* create = reinterpret_cast<CreateScalarFn>(Resolve("aclCreateScalar"));
  if (create == nullptr) {
    throw AclError("aclCreateScalar not found in libopapi.so / libnnopbase.so", __FILE__, __LINE__, -1);
  }
  aclScalar* scalar = create(value, dtype);
  if (scalar == nullptr) {
    throw AclError("aclCreateScalar returned nullptr", __FILE__, __LINE__, -1);
  }
  return scalar;
}

void OpApiLibrary::DestroyScalar(const aclScalar* scalar) const {
  if (scalar == nullptr) {
    return;
  }
  auto* destroy = reinterpret_cast<DestroyScalarFn>(Resolve("aclDestroyScalar"));
  if (destroy != nullptr) {
    destroy(scalar);
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

AclnnScalar::AclnnScalar(float value) {
  scalar_ = OpApiLibrary::Instance().CreateScalar(&value, ACL_FLOAT);
}

AclnnScalar::~AclnnScalar() { OpApiLibrary::Instance().DestroyScalar(scalar_); }

AclnnTensorList::AclnnTensorList(const std::vector<const aclTensor*>& tensors) {
  list_ = OpApiLibrary::Instance().CreateTensorList(tensors.data(), static_cast<uint64_t>(tensors.size()));
}

AclnnTensorList::~AclnnTensorList() { OpApiLibrary::Instance().DestroyTensorList(list_); }

AclnnOp::AclnnOp(const char* name) : name_(name) {
  const OpApiLibrary& library = OpApiLibrary::Instance();
  const std::string workspace_symbol = name_ + "GetWorkspaceSize";
  get_workspace_size_ = library.Resolve(workspace_symbol.c_str(), &source_);
  launch_ = library.Resolve(name_.c_str());
  if (!available()) {
    source_.clear();
  }
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
  const std::vector<std::string>& custom = library.custom_library_paths();
  if (custom.empty()) {
    reason << " No custom operator package was found: set ASCEND_CUSTOM_OPP_PATH,"
              " or install one into $ASCEND_OPP_PATH/vendors, if this operator comes from csrc/.";
  } else {
    reason << " Custom operator packages searched:";
    for (const std::string& path : custom) {
      reason << " " << path;
    }
  }
  return reason.str();
}

}
}
