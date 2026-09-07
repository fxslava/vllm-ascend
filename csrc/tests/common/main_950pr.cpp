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

// Entry point for the Ascend 950PR test binaries.
//
// Same shape as common/main.cpp, but it prints the 950PR pipeline inventory
// instead of the 310P one. The two differ in which operators they list and in
// the note about the custom op package, and a binary that printed the wrong one
// would send a reader looking for the wrong missing symbol.

#include <gtest/gtest.h>

#include <cstdio>

#include "aclnn_ops_950pr.hpp"
#include "test_harness.hpp"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  std::printf("[ascend-test] vllm-ascend bare-metal kernel tests, Ascend 950PR (no Python, no torch)\n");
  ::vllm_ascend::test::ops950::PrintAscend950OperatorInventory();
  std::fflush(stdout);

  ::vllm_ascend::test::RegisterAscendTestEnvironment();
  return RUN_ALL_TESTS();
}
