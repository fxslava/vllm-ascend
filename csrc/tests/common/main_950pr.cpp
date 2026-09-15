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

#include <gtest/gtest.h>

#include <cstdio>

#include "aclnn_ops_950pr.hpp"
#include "test_harness.hpp"

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
