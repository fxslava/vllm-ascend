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

#include "test_harness.hpp"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vllm_ascend {
namespace test {

void ReportIgnoredAclFailure(const char* expression, const char* file, int line, int status) {
  std::fprintf(stderr, "[ascend-test] ignoring failure during cleanup: %s\n",
               FormatAclError(expression, file, line, status).c_str());
}

int ResolveDeviceId() {
  const char* raw = std::getenv("ASCEND_TEST_DEVICE_ID");
  if (raw == nullptr || raw[0] == '\0') {
    return 0;
  }
  return std::atoi(raw);
}

namespace {

// aclrtGetSocName is not present in every CANN release the plugin supports, so
// it is resolved at runtime instead of being a link-time dependency. RTLD_DEFAULT
// finds it in libascendcl.so, which the test binary already links.
std::string QuerySocName() {
  using GetSocNameFn = const char* (*)();
  auto* symbol = reinterpret_cast<GetSocNameFn>(dlsym(RTLD_DEFAULT, "aclrtGetSocName"));
  if (symbol == nullptr) {
    return std::string();
  }
  const char* name = symbol();
  return (name != nullptr) ? std::string(name) : std::string();
}

}  // namespace

AscendDevice::AscendDevice() {
  device_id_ = ResolveDeviceId();
  try {
    // A null config path means "use the built-in defaults"; the tests do not
    // need a dump or profiling json here, msprof attaches externally.
    ACL_CHECK(aclInit(nullptr));
    acl_initialised_ = true;

    ACL_CHECK(aclrtSetDevice(device_id_));
    device_set_ = true;

    ACL_CHECK(aclrtCreateContext(&context_, device_id_));
    ACL_CHECK(aclrtSetCurrentContext(context_));
    ACL_CHECK(aclrtCreateStream(&stream_));

    soc_name_ = QuerySocName();
  } catch (...) {
    // Undo whatever succeeded before rethrowing, so a failed construction does
    // not leave the runtime half-initialised for the next attempt.
    if (stream_ != nullptr) {
      ACL_CHECK_NOTHROW(aclrtDestroyStream(stream_));
      stream_ = nullptr;
    }
    if (context_ != nullptr) {
      ACL_CHECK_NOTHROW(aclrtDestroyContext(context_));
      context_ = nullptr;
    }
    if (device_set_) {
      ACL_CHECK_NOTHROW(aclrtResetDevice(device_id_));
      device_set_ = false;
    }
    if (acl_initialised_) {
      ACL_CHECK_NOTHROW(aclFinalize());
      acl_initialised_ = false;
    }
    throw;
  }
}

AscendDevice::~AscendDevice() {
  if (stream_ != nullptr) {
    // Drain before destroying: an in-flight task holding a reference to a
    // freed stream is the usual source of teardown EXCEPTIONs.
    ACL_CHECK_NOTHROW(aclrtSynchronizeStream(stream_));
    ACL_CHECK_NOTHROW(aclrtDestroyStream(stream_));
    stream_ = nullptr;
  }
  if (context_ != nullptr) {
    ACL_CHECK_NOTHROW(aclrtDestroyContext(context_));
    context_ = nullptr;
  }
  if (device_set_) {
    ACL_CHECK_NOTHROW(aclrtResetDevice(device_id_));
    device_set_ = false;
  }
  if (acl_initialised_) {
    ACL_CHECK_NOTHROW(aclFinalize());
    acl_initialised_ = false;
  }
}

AscendTestEnvironment& AscendTestEnvironment::Instance() {
  static AscendTestEnvironment instance;
  return instance;
}

void AscendTestEnvironment::SetUp() {
  try {
    device_ = std::unique_ptr<AscendDevice>(new AscendDevice());
    soc_name_ = device_->soc_name();
    std::fprintf(stdout, "[ascend-test] device %d ready, soc='%s'\n", device_->device_id(),
                 soc_name_.empty() ? "<unknown>" : soc_name_.c_str());
  } catch (const std::exception& error) {
    device_.reset();
    unavailable_reason_ = error.what();
    std::fprintf(stdout, "[ascend-test] device unavailable, all device tests will skip:\n%s\n",
                 unavailable_reason_.c_str());
  }
}

void AscendTestEnvironment::TearDown() { device_.reset(); }

AscendDevice& AscendTestEnvironment::device() {
  if (device_ == nullptr) {
    throw AclError("AscendTestEnvironment::device() called without a device", __FILE__, __LINE__, -1);
  }
  return *device_;
}

aclrtStream AscendTestEnvironment::stream() { return device().stream(); }

bool AscendTestEnvironment::is_310p() const {
  // Covers Ascend310P1/P3/P5 and the "Ascend310P" short form. An empty SoC name
  // means the runtime could not tell us, in which case we do not block the test.
  if (soc_name_.empty()) {
    return true;
  }
  return soc_name_.find("310P") != std::string::npos || soc_name_.find("310p") != std::string::npos;
}

void RegisterAscendTestEnvironment() {
  // GTest takes ownership of registered environments and deletes them, so the
  // singleton is wrapped in a non-owning shim.
  class EnvironmentShim : public ::testing::Environment {
   public:
    void SetUp() override { AscendTestEnvironment::Instance().SetUp(); }
    void TearDown() override { AscendTestEnvironment::Instance().TearDown(); }
  };
  ::testing::AddGlobalTestEnvironment(new EnvironmentShim());
}

}  // namespace test
}  // namespace vllm_ascend
