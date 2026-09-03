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

// Entry point shared by every benchmark binary. Each kernels/bench_*.cpp
// provides the two symbols declared here and nothing else: no GTest, no test
// registration, and no global constructors that could run before the runtime
// is up.

#include "benchmark.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {

// Defined by the bench_*.cpp this binary links.
extern const char* kSuiteName;
void BuildSuite(BenchmarkRunner& runner);

}  // namespace bench
}  // namespace test
}  // namespace vllm_ascend

int main() {
  return ::vllm_ascend::test::bench::RunBenchmarkSuite(::vllm_ascend::test::bench::kSuiteName,
                                                       ::vllm_ascend::test::bench::BuildSuite);
}
