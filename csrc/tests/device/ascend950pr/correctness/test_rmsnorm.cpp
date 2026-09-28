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

// The 950PR-specific RMSNorm checks. The parameterised aclnnRmsNorm sweep and
// the CPU-reference checks are shared with the 310P leg - this binary also
// compiles common/testing/test_rmsnorm_sweep.cpp (see CMakeLists.txt); this
// file keeps only what is specific to this part: the layer widths this
// pipeline launches, and the SoC-name gate the harness skips on.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "ascend950_shapes.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace s = shapes950;

TEST(RmsNorm950PrShapes, LayerWidthsAreBurstAligned) {
  EXPECT_EQ(s::kHidden % s::kFp16ElementsPerBurst, 0);
  EXPECT_EQ(s::kIntermediate % s::kFp16ElementsPerBurst, 0);
  for (int64_t hidden : s::RmsNormHiddenSizes()) {
    EXPECT_EQ(hidden % s::kFp16ElementsPerBurst, 0)
        << "hidden=" << hidden << " is not a multiple of " << s::kFp16ElementsPerBurst << " fp16 elements";
  }
}

TEST(RmsNorm950PrSocGate, AcceptsEvery950PrBinAndNothingElse) {
  EXPECT_TRUE(s::IsAscend950PrSocName("Ascend950PR_9599"));
  EXPECT_TRUE(s::IsAscend950PrSocName("Ascend950PR_957b"));
  EXPECT_TRUE(s::IsAscend950PrSocName("Ascend950PR_950z"));
  EXPECT_FALSE(s::IsAscend950PrSocName("Ascend950DT_9591"));
  EXPECT_FALSE(s::IsAscend950PrSocName("Ascend310P3"));
  EXPECT_FALSE(s::IsAscend950PrSocName("Ascend950"));
  EXPECT_FALSE(s::IsAscend950PrSocName(""));
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
