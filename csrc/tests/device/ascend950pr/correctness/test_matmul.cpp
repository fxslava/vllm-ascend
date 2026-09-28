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

// The 950PR-specific matmul checks. The parameterised aclnnMatmul sweep and the
// CPU-reference checks are shared with the 310P leg - this binary also compiles
// common/testing/test_matmul_sweep.cpp (see CMakeLists.txt); this file keeps
// only what is specific to this part: that every Qwen3.5-2B projection this
// pipeline launches is covered by, and aligned for, the shared sweep.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "ascend950_shapes.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace s = shapes950;

TEST(Matmul950PrShapes, LayerProjectionsAreCoveredByTheSweep) {
  struct Projection {
    const char* name;
    int64_t k;
    int64_t n;
  };
  const Projection projections[] = {
      {"q / attn_gate", s::kHidden, s::kQDim}, {"k / v", s::kHidden, s::kKvDim},
      {"o_proj", s::kQDim, s::kHidden},        {"gate / up", s::kHidden, s::kIntermediate},
      {"down", s::kIntermediate, s::kHidden},
  };

  for (const Projection& projection : projections) {
    EXPECT_EQ(projection.k % s::kFp16ElementsPerBurst, 0) << projection.name << " K";
    EXPECT_EQ(projection.n % s::kFp16ElementsPerBurst, 0) << projection.name << " N";
  }
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
