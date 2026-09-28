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

// The 310P-specific RMSNorm checks. The parameterised aclnnRmsNorm sweep and
// the CPU-reference checks are shared with the 950PR leg - this binary also
// compiles common/testing/test_rmsnorm_sweep.cpp (see CMakeLists.txt); this
// file keeps only what is specific to this part: that the shared sweep widths
// satisfy the v200 burst-alignment rule.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "qwen_shapes.hpp"

namespace vllm_ascend {
namespace test {
namespace {

TEST(RmsNormShapes, QwenHiddenSizesAreBurstAligned) {
  for (int64_t hidden : shapes::RmsNormHiddenSizes()) {
    EXPECT_EQ(hidden % shapes::kFp16ElementsPerBurst, 0)
        << "hidden=" << hidden << " is not a multiple of " << shapes::kFp16ElementsPerBurst
        << " fp16 elements (32 bytes)";
  }
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
