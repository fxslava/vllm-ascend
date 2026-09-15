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

constexpr const char* kCsvFlag = "--csv=";
constexpr const char* kDefaultCsvPath = "hadamard_benchmark_results.csv";

void PrintUsage(const char* argv0) {
  std::printf(
      "usage: %s [--csv=<path>] [--help]\n"
      "\n"
      "  --csv=<path>  write the per-case results to <path>. Defaults to %s;\n"
      "                pass --csv= with an empty value to write no CSV.\n"
      "\n"
      "The sweep is D in {64,128,256,512} by V in {1,8,16,32}, four legs each.\n"
      "Restrict it with ASCEND_BENCH_HADAMARD_DIMS and\n"
      "ASCEND_BENCH_HADAMARD_BATCHES (comma-separated), and shape the timing\n"
      "with the shared ASCEND_BENCH_* variables.\n",
      argv0, kDefaultCsvPath);
}

}

int main(int argc, char** argv) {
  bool csv_from_flag = false;
  std::string csv_path;

  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strncmp(arg, kCsvFlag, std::strlen(kCsvFlag)) == 0) {
      csv_from_flag = true;
      csv_path = arg + std::strlen(kCsvFlag);
      continue;
    }
    if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      PrintUsage(argv[0]);
      return 0;
    }
    std::fprintf(stderr, "[ascend-bench] unrecognised argument '%s'\n", arg);
    PrintUsage(argv[0]);
    return 2;
  }

  if (csv_from_flag) {
    ::setenv("ASCEND_BENCH_CSV", csv_path.c_str(), 1);
  } else if (std::getenv("ASCEND_BENCH_CSV") == nullptr) {
    ::setenv("ASCEND_BENCH_CSV", kDefaultCsvPath, 1);
  }

  if (::vllm_ascend::test::IsRunningOnSimulator()) {
    std::printf("[ascend-bench] CAModel loaded (%s).\n"
                "[ascend-bench] This is a device-tier benchmark and the simulator's wall clock is not a\n"
                "[ascend-bench] measurement of the part. Refusing to produce numbers; skipping.\n",
                ::vllm_ascend::test::SimulatorEvidence().c_str());
    return ::vllm_ascend::test::bench::kBenchmarkSkipExitCode;
  }
  return ::vllm_ascend::test::bench::RunBenchmarkSuite(::vllm_ascend::test::bench::kSuiteName,
                                                       ::vllm_ascend::test::bench::BenchmarkTargetPart::kAscend950PR,
                                                       ::vllm_ascend::test::bench::BuildSuite);
}
