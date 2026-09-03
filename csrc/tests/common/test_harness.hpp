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

// Process-wide Ascend runtime lifecycle for the bare-metal kernel tests.
//
// Ownership model: a single AscendDevice object owns aclInit, the device, the
// context and the default stream, and tears them down in the reverse order in
// its destructor. AscendTestEnvironment is the GTest adapter that constructs it
// once per process. Nothing here touches Python, PyTorch or torch_npu.

#pragma once

#include <acl/acl.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "acl_check.hpp"

namespace vllm_ascend {
namespace test {

// Device ordinal used by every test. Override with ASCEND_TEST_DEVICE_ID.
int ResolveDeviceId();

// RAII owner of the ACL runtime. Construction performs aclInit ->
// aclrtSetDevice -> aclrtCreateContext -> aclrtCreateStream; destruction undoes
// all four in reverse, and is safe to run after a partially failed setup.
class AscendDevice {
 public:
  AscendDevice();
  ~AscendDevice();

  AscendDevice(const AscendDevice&) = delete;
  AscendDevice& operator=(const AscendDevice&) = delete;

  aclrtStream stream() const { return stream_; }
  aclrtContext context() const { return context_; }
  int32_t device_id() const { return device_id_; }

  // Value of aclrtGetSocName(), e.g. "Ascend310P3". Empty when the running
  // CANN build does not export the symbol.
  const std::string& soc_name() const { return soc_name_; }

  void SynchronizeStream() const { ACL_CHECK(aclrtSynchronizeStream(stream_)); }

 private:
  int32_t device_id_ = 0;
  bool acl_initialised_ = false;
  bool device_set_ = false;
  aclrtContext context_ = nullptr;
  aclrtStream stream_ = nullptr;
  std::string soc_name_;
};

// GTest global environment. SetUp() never fails the run when no NPU is present:
// it records why the device is unavailable and every test skips with that
// reason, so the suite stays runnable on a build machine.
class AscendTestEnvironment : public ::testing::Environment {
 public:
  static AscendTestEnvironment& Instance();

  void SetUp() override;
  void TearDown() override;

  bool available() const { return device_ != nullptr; }
  const std::string& unavailable_reason() const { return unavailable_reason_; }

  AscendDevice& device();
  aclrtStream stream();

  const std::string& soc_name() const { return soc_name_; }

  // True when the attached device reports an Ascend 310P part. Tests that
  // encode v200-specific layouts (the 5-D NZ KV cache, the 32-element SwiGLU
  // constraint) gate on this rather than on a build-time define.
  bool is_310p() const;

 private:
  AscendTestEnvironment() = default;

  std::unique_ptr<AscendDevice> device_;
  std::string unavailable_reason_;
  std::string soc_name_;
};

// Registers the environment with GTest. Called from main().
void RegisterAscendTestEnvironment();

#define REQUIRE_ASCEND_DEVICE()                                                            \
  do {                                                                                     \
    if (!::vllm_ascend::test::AscendTestEnvironment::Instance().available()) {             \
      GTEST_SKIP() << "No usable Ascend device: "                                          \
                   << ::vllm_ascend::test::AscendTestEnvironment::Instance()               \
                          .unavailable_reason();                                           \
    }                                                                                      \
  } while (false)

#define REQUIRE_ASCEND_310P()                                                              \
  do {                                                                                     \
    REQUIRE_ASCEND_DEVICE();                                                               \
    if (!::vllm_ascend::test::AscendTestEnvironment::Instance().is_310p()) {               \
      GTEST_SKIP() << "Test targets Ascend 310P; attached device reports '"                \
                   << ::vllm_ascend::test::AscendTestEnvironment::Instance().soc_name()    \
                   << "'";                                                                 \
    }                                                                                      \
  } while (false)

}  // namespace test
}  // namespace vllm_ascend
