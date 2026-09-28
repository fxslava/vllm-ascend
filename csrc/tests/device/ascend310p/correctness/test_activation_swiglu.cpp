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

// The 310P-specific SwiGLU checks. The parameterised aclnnSwiGlu sweep and the
// CPU-reference checks are shared with the 950PR leg - this binary also
// compiles common/testing/test_swiglu_sweep.cpp (see CMakeLists.txt); this
// file keeps only what is specific to this part: the v200 rule that the fused
// npu_swiglu kernel requires the input's last dim to be a multiple of 32,
// everything else taking the eager fallback.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "qwen_shapes.hpp"

namespace vllm_ascend {
namespace test {
namespace {

TEST(SwiGluShapes, QwenIntermediateSizesSatisfyThe310PGate) {
  for (int64_t intermediate : shapes::IntermediateSizes()) {
    const int64_t input_last_dim = intermediate * 2;
    EXPECT_EQ(input_last_dim % shapes::kSwiGluLastDimMultiple, 0)
        << "intermediate=" << intermediate << " gives last dim " << input_last_dim
        << ", which would take the eager fallback instead of npu_swiglu";
  }
}

TEST(SwiGluShapes, OddLastDimensionWouldTakeTheEagerFallback) {
  const int64_t intermediate = 20;
  EXPECT_NE((intermediate * 2) % shapes::kSwiGluLastDimMultiple, 0);
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
