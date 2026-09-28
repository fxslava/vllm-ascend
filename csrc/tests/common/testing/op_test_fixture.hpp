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

#pragma once

// The boilerplate behind the stock-operator tests: the fp16-quantised allclose
// against a float CPU reference, the parameter-name generator for the
// (int64, int64) shape sweeps both tiers run, and the SoC gate for the shared
// sweep sources.

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <tuple>

#include "fp16.hpp"
#include "tensor_compare.hpp"
#include "test_harness.hpp"

// The parameterised sweeps in test_matmul_sweep.cpp and friends are compiled
// into both silicon legs - the 950PR binaries and their *_310p twins - from one
// source. The legs are mutually exclusive in any one build tree, and the 310P
// leg defines ASCEND_PLATFORM_310P for everything it builds (see its
// CMakeLists.txt), so that macro alone picks the matching REQUIRE_ASCEND_*
// gate. GTEST_SKIP must fire from the test body itself, which is why this is a
// macro rather than a helper function.
#if defined(ASCEND_PLATFORM_310P)
#define REQUIRE_OPERATOR_TARGET_SOC() REQUIRE_ASCEND_310P()
#else
#define REQUIRE_OPERATOR_TARGET_SOC() REQUIRE_ASCEND_950PR()
#endif

namespace vllm_ascend {
namespace test {
namespace op_case {

// EXPECT_TENSORS_ALLCLOSE(actual, QuantizeToHalf(expected), kFp16DefaultTolerance),
// the comparison every fp16 operator test makes against its float CPU reference.
#define EXPECT_HALF_TENSORS_ALLCLOSE(actual, expected_float)                             \
  do {                                                                                   \
    EXPECT_TENSORS_ALLCLOSE(actual, ::vllm_ascend::test::QuantizeToHalf(expected_float), \
                            ::vllm_ascend::test::kFp16DefaultTolerance);                 \
  } while (false)

// The gtest name generator for TestWithParam<std::tuple<int64_t, int64_t>>
// sweeps: TupleName("tokens", "hidden") names a case "tokens32_hidden2048".
class TupleName {
 public:
  TupleName(const char* first, const char* second) : first_(first), second_(second) {}

  std::string operator()(const ::testing::TestParamInfo<std::tuple<int64_t, int64_t>>& info) const {
    std::ostringstream name;
    name << first_ << std::get<0>(info.param) << '_' << second_ << std::get<1>(info.param);
    return name.str();
  }

 private:
  const char* first_;
  const char* second_;
};

}  // namespace op_case
}  // namespace test
}  // namespace vllm_ascend
