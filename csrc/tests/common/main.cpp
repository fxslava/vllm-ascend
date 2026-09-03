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

// Entry point shared by every kernel test binary.
//
// The environment inventory is printed before the first test so that a run that
// ends in skips says why on its first few lines, rather than requiring the
// reader to correlate skip messages afterwards.

#include <gtest/gtest.h>

#include <cstdio>

#include "aclnn_ops.hpp"
#include "test_harness.hpp"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  std::printf("[ascend-test] vllm-ascend bare-metal kernel tests (no Python, no torch)\n");
  ::vllm_ascend::test::ops::PrintOperatorInventory();
  std::fflush(stdout);

  ::vllm_ascend::test::RegisterAscendTestEnvironment();
  return RUN_ALL_TESTS();
}
