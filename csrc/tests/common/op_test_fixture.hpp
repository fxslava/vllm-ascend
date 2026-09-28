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
// against a float CPU reference, and the parameter-name generator for the
// (int64, int64) shape sweeps both tiers run.

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <tuple>

#include "fp16.hpp"
#include "tensor_compare.hpp"

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
