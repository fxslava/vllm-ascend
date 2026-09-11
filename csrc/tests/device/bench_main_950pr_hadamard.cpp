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

// Entry point for bench_950pr_cube_hadamard, and the only reason it is not
// common/bench_main_950pr.cpp: this sweep is meant to be plotted, so it takes
// --csv=<path> on the command line and writes a CSV by default rather than only
// when ASCEND_BENCH_CSV is set.
//
// The flag is translated into that same environment variable before the suite
// runs, because BenchmarkOptions::FromEnvironment is what the shared harness
// reads and adding an argv path to it would change every other benchmark's
// interface for one binary's convenience.
//
// Precedence, most specific first:
//
//   --csv=<path>     writes there
//   --csv=           writes nothing; the escape hatch for a run that only wants
//                    the table on stdout
//   ASCEND_BENCH_CSV honoured when no flag is given, as everywhere else
//   neither          hadamard_benchmark_results.csv in the working directory
//
// Everything else - the banner, the SoC check and the camodel refusal - is the
// shared entry point's behaviour, restated rather than shared because there is
// no argv-taking overload of it.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "benchmark.hpp"
#include "test_harness.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {

// Defined by bench_950pr_cube_hadamard.cpp.
extern const char* kSuiteName;
void BuildSuite(BenchmarkRunner& runner);

}  // namespace bench
}  // namespace test
}  // namespace vllm_ascend

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

}  // namespace

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
    // An empty --csv= clears whatever the environment asked for, which is what
    // makes it an off switch rather than a no-op.
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
