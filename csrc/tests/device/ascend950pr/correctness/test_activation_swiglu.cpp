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

// The 950PR-specific SwiGLU checks. The parameterised aclnnSwiGlu sweep and
// the CPU-reference checks are shared with the 310P leg - this binary also
// compiles common/testing/test_swiglu_sweep.cpp (see CMakeLists.txt); this
// file keeps only what is specific to this part: that the Qwen3.5-2B
// intermediate widths, and every sweep width, are burst aligned.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "ascend950_shapes.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace s = shapes950;

TEST(SwiGlu950PrShapes, LayerAndSweepWidthsAreBurstAligned) {
  EXPECT_EQ(s::kIntermediate % s::kFp16ElementsPerBurst, 0);
  EXPECT_EQ((s::kIntermediate * 2) % s::kFp16ElementsPerBurst, 0);
  for (int64_t intermediate : s::IntermediateSizes()) {
    EXPECT_EQ(intermediate % s::kFp16ElementsPerBurst, 0) << "intermediate=" << intermediate;
    EXPECT_EQ((intermediate * 2) % s::kFp16ElementsPerBurst, 0) << "intermediate=" << intermediate;
  }
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
