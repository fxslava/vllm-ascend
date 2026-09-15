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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "benchmark.hpp"
#include "test_harness.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {

extern const char* kSuiteName;
void BuildSuite(BenchmarkRunner& runner);

}
}
}

namespace {

constexpr const char* kStageFlag = "--stage=";
constexpr const char* kTimeoutFlag = "--sync-timeout-ms=";
constexpr char kLastStageDigit = '5';
constexpr size_t kMaxTimeoutDigits = 9;
constexpr int kUsageExitCode = 2;

void PrintUsage(const char* argv0) {
  std::printf("usage: %s [--stage=<list>] [--sync-timeout-ms=<ms>] [--help]\n"
              "\n"
              "  --stage=<list>          comma-separated stages 0..5, run in that order.\n"
              "                          Default: all six. One stage per process isolates it.\n"
              "  --sync-timeout-ms=<ms>  deadline for each stage's first launch. Default 30000;\n"
              "                          0 waits forever. A stage that misses it exits 3.\n"
              "\n"
              "Restrict the shapes with ASCEND_BENCH_TQ_ABLATION_DIMS and\n"
              "ASCEND_BENCH_TQ_ABLATION_CONTEXTS, and shape the timing with the shared\n"
              "ASCEND_BENCH_* variables.\n",
              argv0);
}

bool IsStageList(const std::string& value) {
  if (value.empty()) {
    return false;
  }
  for (size_t i = 0; i < value.size(); ++i) {
    const bool digit_slot = i % 2 == 0;
    const char c = value[i];
    if (digit_slot ? (c < '0' || c > kLastStageDigit) : c != ',') {
      return false;
    }
  }
  return value.size() % 2 == 1;
}

bool IsTimeout(const std::string& value) {
  return !value.empty() && value.size() <= kMaxTimeoutDigits &&
         value.find_first_not_of("0123456789") == std::string::npos;
}

}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strncmp(arg, kStageFlag, std::strlen(kStageFlag)) == 0) {
      const std::string value = arg + std::strlen(kStageFlag);
      if (!IsStageList(value)) {
        std::fprintf(stderr, "[ascend-bench] --stage='%s' is not a comma-separated list of stages 0..5\n",
                     value.c_str());
        return kUsageExitCode;
      }
      ::setenv("ASCEND_BENCH_TQ_ABLATION_STAGES", value.c_str(), 1);
      continue;
    }
    if (std::strncmp(arg, kTimeoutFlag, std::strlen(kTimeoutFlag)) == 0) {
      const std::string value = arg + std::strlen(kTimeoutFlag);
      if (!IsTimeout(value)) {
        std::fprintf(stderr, "[ascend-bench] --sync-timeout-ms='%s' is not a non-negative integer\n", value.c_str());
        return kUsageExitCode;
      }
      ::setenv("ASCEND_BENCH_TQ_ABLATION_SYNC_TIMEOUT_MS", value.c_str(), 1);
      continue;
    }
    if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      PrintUsage(argv[0]);
      return 0;
    }
    std::fprintf(stderr, "[ascend-bench] unrecognised argument '%s'\n", arg);
    PrintUsage(argv[0]);
    return kUsageExitCode;
  }

  if (::vllm_ascend::test::IsRunningOnSimulator()) {
    std::printf("[ascend-bench] CAModel loaded (%s).\n"
                "[ascend-bench] This is a device-tier benchmark and the simulator's wall clock is not a\n"
                "[ascend-bench] measurement of the part. Refusing to produce numbers; skipping.\n"
                "[ascend-bench] sim/test_sim_950pr_turboquant_ablation runs the same stages there.\n",
                ::vllm_ascend::test::SimulatorEvidence().c_str());
    return ::vllm_ascend::test::bench::kBenchmarkSkipExitCode;
  }
  return ::vllm_ascend::test::bench::RunBenchmarkSuite(::vllm_ascend::test::bench::kSuiteName,
                                                       ::vllm_ascend::test::bench::BenchmarkTargetPart::kAscend950PR,
                                                       ::vllm_ascend::test::bench::BuildSuite);
}
