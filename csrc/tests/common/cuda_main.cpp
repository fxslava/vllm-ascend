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

// Entry point shared by every CUDA kernel test binary, the counterpart of
// common/main.cpp.
//
// The device inventory is printed before the first test so that a run which
// ends in skips says why on its first few lines, rather than requiring the
// reader to correlate skip messages afterwards. There is no operator inventory
// to print: the CUDA backend compiles its kernels into the binary instead of
// resolving them out of a vendor library at runtime, so a kernel is either
// there or the link failed.

#include <gtest/gtest.h>

#include <cstdio>

#include "cuda_runtime.hpp"

namespace vllm_ascend {
namespace test {

void RegisterCudaTestEnvironment() {
  // GTest takes ownership of registered environments and deletes them, so the
  // singleton is wrapped in a non-owning shim, exactly as the Ascend harness
  // does.
  class EnvironmentShim : public ::testing::Environment {
   public:
    void SetUp() override { CUDATestEnvironment::Instance().SetUp(); }
    void TearDown() override { CUDATestEnvironment::Instance().TearDown(); }
  };
  ::testing::AddGlobalTestEnvironment(new EnvironmentShim());
}

}  // namespace test
}  // namespace vllm_ascend

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  std::printf("[cuda-test] vllm-ascend bare-metal kernel tests, native CUDA C backend\n");
  std::printf("[cuda-test] no PyTorch, no torch_npu, no Cutlass: CUDA Runtime API only\n");
  std::fflush(stdout);

  ::vllm_ascend::test::RegisterCudaTestEnvironment();
  return RUN_ALL_TESTS();
}
