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

#include <acl/acl.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "aclnn_runtime.hpp"
#include "device_buffer.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {

enum class TimingMode {
  kPipelined,
  kDeviceEvents,
  kHostWallClock,
};

const char* TimingModeLabel(TimingMode mode);

struct BenchmarkOptions {
  int warmup_iterations = 20;
  int timed_iterations = 100;
  int pipeline_batch = 10;
  std::vector<TimingMode> modes;
  std::string csv_path;
  bool allow_repeatable_executor = true;

  static BenchmarkOptions FromEnvironment();
};

struct LatencySamples {
  std::vector<double> microseconds;
  size_t rejected = 0;
};

struct LatencyStatistics {
  size_t sample_count = 0;
  size_t discarded_count = 0;
  double min_us = 0.0;
  double median_us = 0.0;
  double mean_us = 0.0;
  double p95_us = 0.0;
  double p99_us = 0.0;
  double max_us = 0.0;
  double stddev_us = 0.0;

  static LatencyStatistics From(std::vector<double> samples_us);
};

const char* EventTimingSourceLabel();

class PlannedOp {
 public:
  using Planner = std::function<int(uint64_t*, aclOpExecutor**)>;

  PlannedOp(std::string op_name, void* launch_fn, Planner planner);
  ~PlannedOp();

  PlannedOp(const PlannedOp&) = delete;
  PlannedOp& operator=(const PlannedOp&) = delete;
  PlannedOp(PlannedOp&& other) noexcept;

  void Launch(aclrtStream stream);

  bool repeatable() const { return repeatable_; }
  uint64_t workspace_size() const { return workspace_size_; }
  const std::string& op_name() const { return op_name_; }

  static const char* LaunchPathLabel();

  static void SetRepeatableExecutorAllowed(bool allowed);

 private:
  void DestroyExecutor();
  static void DestroyOrphanedExecutor(aclOpExecutor* executor);

  std::string op_name_;
  void* launch_fn_ = nullptr;
  Planner planner_;
  DeviceBuffer workspace_;
  uint64_t workspace_size_ = 0;
  aclOpExecutor* executor_ = nullptr;
  bool repeatable_ = false;
};

template <typename WorkspaceSizeFn, typename... Args>
PlannedOp PlanAclnn(const AclnnOp& op, Args... args) {
  if (!op.available()) {
    throw AclError(op.unavailable_reason().c_str(), __FILE__, __LINE__, -1);
  }
  void* workspace_size_fn = op.get_workspace_size_fn();
  PlannedOp::Planner planner = [workspace_size_fn, args...](uint64_t* workspace_size,
                                                            aclOpExecutor** executor) -> int {
    return reinterpret_cast<WorkspaceSizeFn>(workspace_size_fn)(args..., workspace_size, executor);
  };
  return PlannedOp(op.name(), op.launch_fn(), std::move(planner));
}

struct BenchmarkCase {
  std::string name;

  double flops_per_iteration = 0.0;
  double bytes_per_iteration = 0.0;

  std::function<void(aclrtStream)> launch;

  int tasks_per_launch = 1;

  std::function<double()> checksum;

  double checksum_rtol = 0.0;
};

double ChecksumSum(const std::vector<float>& values);
double ChecksumSumOfSquares(const std::vector<float>& values);

struct BenchmarkResult {
  std::string case_name;
  TimingMode mode;
  LatencyStatistics latency;
  double flops_per_iteration = 0.0;
  double bytes_per_iteration = 0.0;

  double tflops() const;
  double gigabytes_per_second() const;
};

class BenchmarkRunner {
 public:
  BenchmarkRunner(std::string suite_name, const BenchmarkOptions& options, aclrtStream stream);

  aclrtStream stream() const { return stream_; }
  const BenchmarkOptions& options() const { return options_; }

  void set_options(const BenchmarkOptions& options);

  void Run(const BenchmarkCase& benchmark_case);

  void Skip(const std::string& case_name, const std::string& reason);

  void Report() const;

  bool has_results() const { return !results_.empty(); }
  size_t failure_count() const { return failures_.size(); }

  const std::vector<BenchmarkResult>& results() const { return results_; }

  void RecordFailure(const std::string& case_name, const std::string& reason);

 private:
  LatencySamples TimePipelined(const BenchmarkCase& benchmark_case);
  LatencySamples TimeDeviceEvents(const BenchmarkCase& benchmark_case);
  LatencySamples TimeHostWallClock(const BenchmarkCase& benchmark_case);

  void WarmUp(const BenchmarkCase& benchmark_case);

  void DrainIfQueueIsDeep(int* enqueued, int about_to_enqueue);

  void RunOrThrow(const BenchmarkCase& benchmark_case);

  void PrintTable() const;
  void WriteCsv() const;

  struct SkippedCase {
    std::string name;
    std::string reason;
  };

  std::string suite_name_;
  BenchmarkOptions options_;
  aclrtStream stream_ = nullptr;
  std::vector<BenchmarkResult> results_;
  std::vector<SkippedCase> skipped_;
  std::vector<SkippedCase> failures_;
};

enum class BenchmarkTargetPart {
  kAscend310P,
  kAscend950PR,
};

int RunBenchmarkSuite(const char* suite_name, BenchmarkTargetPart part,
                      const std::function<void(BenchmarkRunner&)>& build);

int RunBenchmarkSuite(const char* suite_name, const std::function<void(BenchmarkRunner&)>& build);

constexpr int kBenchmarkSkipExitCode = 77;

}
}
}
