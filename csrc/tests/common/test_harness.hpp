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

#pragma once

#include <acl/acl.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string>

#include "acl_check.hpp"

namespace vllm_ascend {
namespace test {

int ResolveDeviceId();

class AscendDevice {
 public:
  AscendDevice();
  ~AscendDevice();

  AscendDevice(const AscendDevice&) = delete;
  AscendDevice& operator=(const AscendDevice&) = delete;

  aclrtStream stream() const { return stream_; }
  aclrtContext context() const { return context_; }
  int32_t device_id() const { return device_id_; }

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

  bool is_310p() const;

  bool is_950pr() const;

 private:
  AscendTestEnvironment() = default;

  std::unique_ptr<AscendDevice> device_;
  std::string unavailable_reason_;
  std::string soc_name_;
};

void RegisterAscendTestEnvironment();

bool IsRunningOnSimulator();
const std::string& SimulatorEvidence();

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

#define REQUIRE_ASCEND_950PR()                                                             \
  do {                                                                                     \
    REQUIRE_ASCEND_DEVICE();                                                               \
    if (!::vllm_ascend::test::AscendTestEnvironment::Instance().is_950pr()) {              \
      GTEST_SKIP() << "Test targets Ascend 950PR; attached device reports '"               \
                   << ::vllm_ascend::test::AscendTestEnvironment::Instance().soc_name()    \
                   << "'";                                                                 \
    }                                                                                      \
  } while (false)

#define REQUIRE_PHYSICAL_ASCEND_950PR()                                                    \
  do {                                                                                     \
    REQUIRE_ASCEND_950PR();                                                                \
    const char* allow_simulator = std::getenv("ASCEND_TEST_ALLOW_SIMULATOR");               \
    const bool simulator_allowed = allow_simulator != nullptr && allow_simulator[0] == '1'; \
    if (::vllm_ascend::test::IsRunningOnSimulator() && !simulator_allowed) {               \
      GTEST_SKIP() << "Test targets a physical Ascend 950PR and this process has the CANN " \
                      "camodel loaded ("                                                   \
                   << ::vllm_ascend::test::SimulatorEvidence()                              \
                   << "). Rebuild with -DRUN_MODE=npu and run on the part, or set "         \
                      "ASCEND_TEST_ALLOW_SIMULATOR=1 to run it here anyway (hours).";       \
    }                                                                                      \
  } while (false)

#if defined(VLLM_ASCEND_TQ_CUBE_WIP_DEFAULT_ON)
constexpr bool kCubeWipDefaultOn = true;
#else
constexpr bool kCubeWipDefaultOn = false;
#endif

inline bool CubeWipOptedIn() {
  const char* wip = std::getenv("VLLM_ASCEND_TQ_CUBE_WIP");
  if (wip == nullptr || wip[0] == '\0') {
    return kCubeWipDefaultOn;
  }
  return wip[0] != '0';
}

#define REQUIRE_CUBE_WIP_OPT_IN(what, why)                                                 \
  do {                                                                                     \
    if (!::vllm_ascend::test::CubeWipOptedIn()) {                                          \
      GTEST_SKIP() << (what) << " is work in progress and does not pass: " << (why)        \
                   << " Set VLLM_ASCEND_TQ_CUBE_WIP=1 to run it anyway. See "              \
                      "csrc/tests/TURBOQUANT_TESTS.md section 13.8.";                      \
    }                                                                                      \
  } while (false)

}
}
