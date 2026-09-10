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

// Entry point shared by the Ascend 950PR test binaries of both device-side
// tiers, sim/ and device/.
//
// Same shape as common/main.cpp, but it prints the 950PR pipeline inventory
// instead of the 310P one. The two differ in which operators they list and in
// the note about the custom op package, and a binary that printed the wrong one
// would send a reader looking for the wrong missing symbol.
//
// It also states which tier the binary belongs to and whether a camodel is
// loaded. Those two lines together are the answer to "can I trust a timing from
// this run" - a sim-tier binary, or any binary with libruntime_camodel.so in
// its address space, measures the simulator - and printing them costs nothing
// next to having to work it out from the build directory name afterwards.

#include <gtest/gtest.h>

#include <cstdio>

#include "aclnn_ops_950pr.hpp"
#include "test_harness.hpp"

// Set per target by the tier's CMakeLists.txt. The fallback keeps the file
// buildable on its own.
#ifndef VLLM_ASCEND_TEST_TIER
#define VLLM_ASCEND_TEST_TIER "unspecified"
#endif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  std::printf("[ascend-test] vllm-ascend bare-metal kernel tests, Ascend 950PR (no Python, no torch)\n");
  std::printf("[ascend-test] tier: %s\n", VLLM_ASCEND_TEST_TIER);
  if (::vllm_ascend::test::IsRunningOnSimulator()) {
    std::printf("[ascend-test] CAModel loaded: %s\n"
                "[ascend-test]   this process is running against the simulator. Functional results are\n"
                "[ascend-test]   meaningful; every wall clock in it measures the simulator, not the part.\n",
                ::vllm_ascend::test::SimulatorEvidence().c_str());
  } else {
    std::printf("[ascend-test] CAModel loaded: no (physical runtime)\n");
  }
  ::vllm_ascend::test::ops950::PrintAscend950OperatorInventory();
  std::fflush(stdout);

  ::vllm_ascend::test::RegisterAscendTestEnvironment();
  return RUN_ALL_TESTS();
}
