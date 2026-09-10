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

#include <cstdlib>
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

  // True when the attached device reports an Ascend 950PR part, i.e. a SoC name
  // beginning "Ascend950PR" - the platform_config files name one bin each
  // (Ascend950PR_9599 and friends) and the tests must accept all of them.
  //
  // Unlike is_310p(), an unknown SoC name is NOT accepted here. is_310p()
  // returns true for an empty name because that suite predates any second
  // part and refusing would have made a CANN build without aclrtGetSocName
  // skip everything; this one runs alongside the 310P binaries in the same
  // build tree, so "we could not tell" has to mean "not this part" or a 310P
  // host would silently run 950PR-shaped work.
  bool is_950pr() const;

 private:
  AscendTestEnvironment() = default;

  std::unique_ptr<AscendDevice> device_;
  std::string unavailable_reason_;
  std::string soc_name_;
};

// Registers the environment with GTest. Called from main().
void RegisterAscendTestEnvironment();

// --- silicon versus the camodel ---------------------------------------------
//
// The CANN camodel reports a real SoC name, so is_950pr() cannot tell it from a
// 950PR: that is the point of the simulator and the reason the TurboQuant tests
// can run at all without the part. A suite that only means something on
// hardware - because it measures behaviour the simulator models rather than
// reproduces, or because its shapes would take days cycle-by-cycle - has to ask
// a different question.
//
// SimulatorEvidence() returns the mapped object that gives the camodel away
// (libruntime_camodel.so, or anything loaded out of tools/simulator/), or an
// empty string when nothing does. Silicon is the default assumption: a host
// whose /proc cannot be read reports no evidence rather than skipping.
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

// A 950PR, and not the camodel standing in for one. For suites whose shapes are
// production-sized: the simulator is cycle-level, so a case that is milliseconds
// on the part is hours there, and a run that started anyway would look like a
// hang rather than a skip.
//
// ASCEND_TEST_ALLOW_SIMULATOR=1 runs them under the camodel regardless. That is
// for smoke-checking the binary itself, not for producing results; expect the
// whole suite to take hours, and shrink the sweep with the suite's own shape
// override first.
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


// A case that exercises a path known not to work yet.
//
// The distinction this draws is between "the machine cannot run this" -- which
// is what every gate above is for -- and "this code is wrong and we know it".
// Both should skip in a default run, but for opposite reasons, and conflating
// them hides the second: a suite where the broken cases are simply absent looks
// green, and a suite where they fail looks broken in a way that says nothing
// about the change under test.
//
// So a case gated on this skips with the reason and a pointer, and
// VLLM_ASCEND_TQ_CUBE_WIP=1 runs it. Delete the gate, do not weaken it, when the
// path works.
//
// -DVLLM_ASCEND_TESTS_ENABLE_WIP_CUBE=ON flips the default for a tree dedicated
// to fixing that path; the environment variable still decides in both
// directions, so such a tree can be asked for the stable baseline with
// VLLM_ASCEND_TQ_CUBE_WIP=0 rather than a reconfigure. The same pair gates the
// Cube legs of bench_device_950pr_turboquant, deliberately: one switch, wherever
// the work-in-progress path appears.
#if defined(VLLM_ASCEND_TQ_CUBE_WIP_DEFAULT_ON)
constexpr bool kCubeWipDefaultOn = true;
#else
constexpr bool kCubeWipDefaultOn = false;
#endif

// True when the work-in-progress Cube cases should run.
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

}  // namespace test
}  // namespace vllm_ascend
