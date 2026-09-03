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

#include "benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ios>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "aclnn_ops.hpp"
#include "test_harness.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {

namespace {

// ---------------------------------------------------------------------------
// Environment plumbing
// ---------------------------------------------------------------------------

std::string EnvironmentString(const char* name) {
  const char* raw = std::getenv(name);
  return (raw != nullptr) ? std::string(raw) : std::string();
}

int EnvironmentInt(const char* name, int fallback, int minimum) {
  const std::string raw = EnvironmentString(name);
  if (raw.empty()) {
    return fallback;
  }
  const int parsed = std::atoi(raw.c_str());
  if (parsed < minimum) {
    std::fprintf(stderr, "[ascend-bench] %s=%s is below the minimum of %d, using %d\n", name, raw.c_str(),
                 minimum, fallback);
    return fallback;
  }
  return parsed;
}

std::vector<std::string> SplitOnCommas(const std::string& value) {
  std::vector<std::string> parts;
  std::string current;
  std::istringstream stream(value);
  while (std::getline(stream, current, ',')) {
    // Trim, so "device, host" parses the same as "device,host".
    const size_t first = current.find_first_not_of(" \t");
    const size_t last = current.find_last_not_of(" \t");
    if (first != std::string::npos) {
      parts.push_back(current.substr(first, last - first + 1));
    }
  }
  return parts;
}

// ---------------------------------------------------------------------------
// Repeatable-executor API, resolved the same way the operators are
// ---------------------------------------------------------------------------

using SetExecutorRepeatableFn = int (*)(aclOpExecutor*);
using DestroyExecutorFn = int (*)(aclOpExecutor*);

// VERIFIED against CANN 9.1.0
// $ASCEND_HOME_PATH/include/aclnn/acl_meta.h:76
//   aclnnStatus aclSetAclOpExecutorRepeatable(aclOpExecutor *executor);
//   aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor *executor);
// aclnnStatus is int32_t, so the typedefs above match. Both are exported by
// libnnopbase.so and reachable through the RTLD_GLOBAL handle on libopapi.so,
// the same way aclCreateTensor is. Older CANN lines do not have them, which is
// why PlannedOp has a re-plan fallback at all.
struct ExecutorApi {
  SetExecutorRepeatableFn set_repeatable = nullptr;
  DestroyExecutorFn destroy = nullptr;

  static const ExecutorApi& Instance() {
    static const ExecutorApi instance = [] {
      ExecutorApi api;
      const OpApiLibrary& library = OpApiLibrary::Instance();
      api.set_repeatable =
          reinterpret_cast<SetExecutorRepeatableFn>(library.Resolve("aclSetAclOpExecutorRepeatable"));
      api.destroy = reinterpret_cast<DestroyExecutorFn>(library.Resolve("aclDestroyAclOpExecutor"));
      return api;
    }();
    return instance;
  }
};

// Why a repeatable executor was not used, when it was not. Recorded globally
// because the answer is a property of the CANN build, not of one operator.
enum class LaunchPath {
  kRepeatable,
  kSymbolAbsent,
  kDisabledByEnvironment,
  kRejectedByRuntime,
};

LaunchPath g_launch_path = LaunchPath::kSymbolAbsent;
bool g_repeatable_allowed = true;

// libopapi.so caches planning scratch per thread when these are called around
// the work, which is what torch_npu's EXEC_NPU_CMD does. Without them the
// re-plan fallback pays an allocation per iteration that the plugin would not.
using InitHugeMemThreadLocalFn = int (*)(void*, bool);
using UnInitHugeMemThreadLocalFn = void (*)(void*, bool);

struct HugeMemScope {
  UnInitHugeMemThreadLocalFn uninitialise = nullptr;

  HugeMemScope() {
    const OpApiLibrary& library = OpApiLibrary::Instance();
    auto* initialise =
        reinterpret_cast<InitHugeMemThreadLocalFn>(library.Resolve("InitHugeMemThreadLocal"));
    uninitialise =
        reinterpret_cast<UnInitHugeMemThreadLocalFn>(library.Resolve("UnInitHugeMemThreadLocal"));
    if (initialise != nullptr) {
      initialise(nullptr, false);
    }
  }

  ~HugeMemScope() {
    if (uninitialise != nullptr) {
      uninitialise(nullptr, false);
    }
  }

  HugeMemScope(const HugeMemScope&) = delete;
  HugeMemScope& operator=(const HugeMemScope&) = delete;
};

// ---------------------------------------------------------------------------
// Timing events
// ---------------------------------------------------------------------------

// A pool of start/stop event pairs, created before the timed loop and destroyed
// after it. Creating a fresh pair per iteration keeps the loop free of the
// re-record semantics aclrtResetEvent exists for, which differ across CANN
// releases, and event creation is a setup-phase cost either way.
class EventPool {
 public:
  explicit EventPool(size_t pair_count) : events_(pair_count * 2, nullptr) {
    for (size_t i = 0; i < events_.size(); ++i) {
#ifdef ACL_EVENT_TIME_LINE
      // Only an event created with the timestamp flag can be handed to
      // aclrtEventElapsedTime on the releases that distinguish the two.
      ACL_CHECK(aclrtCreateEventWithFlag(&events_[i], ACL_EVENT_TIME_LINE));
#else
      ACL_CHECK(aclrtCreateEvent(&events_[i]));
#endif
    }
  }

  ~EventPool() {
    for (aclrtEvent event : events_) {
      if (event != nullptr) {
        ACL_CHECK_NOTHROW(aclrtDestroyEvent(event));
      }
    }
  }

  EventPool(const EventPool&) = delete;
  EventPool& operator=(const EventPool&) = delete;

  aclrtEvent start(size_t index) const { return events_[index * 2]; }
  aclrtEvent stop(size_t index) const { return events_[index * 2 + 1]; }

 private:
  std::vector<aclrtEvent> events_;
};

// An Ascend stream holds a bounded number of submitted tasks; past that the
// runtime either blocks the host or refuses the submission, depending on the
// CANN release. At the defaults a pipelined run enqueues 100 * (10 + 2) = 1200
// tasks, which is over the 1024 that releases documenting a limit give, so the
// timing loops drain at sample boundaries once they get close. Two drains per
// run at most, and never inside a batch.
constexpr int kMaxInFlightTasks = 768;

double ElapsedMicroseconds(aclrtEvent start, aclrtEvent stop) {
  float milliseconds = 0.0f;
  ACL_CHECK(aclrtEventElapsedTime(&milliseconds, start, stop));
  return static_cast<double>(milliseconds) * 1000.0;
}

// ---------------------------------------------------------------------------
// Output verification
// ---------------------------------------------------------------------------

void CheckChecksum(const BenchmarkCase& benchmark_case, double after_warmup, double after_timing) {
  if (!std::isfinite(after_timing)) {
    throw std::runtime_error(benchmark_case.name + ": output checksum is not finite after timing (" +
                             std::to_string(after_timing) + "); the operator wrote NaN or Inf");
  }
  const double difference = std::fabs(after_timing - after_warmup);
  const double allowed = benchmark_case.checksum_rtol * std::fabs(after_warmup);
  if (difference > allowed) {
    std::ostringstream message;
    message << benchmark_case.name << ": output checksum moved by more than the case allows across the timed "
            << "loop (" << after_warmup << " -> " << after_timing << ", rtol="
            << benchmark_case.checksum_rtol
            << "). The launches are not all computing the same thing, which makes the timings meaningless.";
    throw std::runtime_error(message.str());
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// BenchmarkOptions
// ---------------------------------------------------------------------------

const char* TimingModeLabel(TimingMode mode) {
  switch (mode) {
    case TimingMode::kPipelined:
      return "pipelined";
    case TimingMode::kDeviceEvents:
      return "device";
    case TimingMode::kHostWallClock:
      return "host";
  }
  return "unknown";
}

BenchmarkOptions BenchmarkOptions::FromEnvironment() {
  BenchmarkOptions options;
  options.warmup_iterations = EnvironmentInt("ASCEND_BENCH_WARMUP", options.warmup_iterations, 0);
  options.timed_iterations = EnvironmentInt("ASCEND_BENCH_ITERS", options.timed_iterations, 1);
  options.pipeline_batch = EnvironmentInt("ASCEND_BENCH_BATCH", options.pipeline_batch, 1);
  options.csv_path = EnvironmentString("ASCEND_BENCH_CSV");

  const std::string repeatable = EnvironmentString("ASCEND_BENCH_REPEATABLE");
  options.allow_repeatable_executor = !(repeatable == "0" || repeatable == "off" || repeatable == "false");

  const std::string requested_modes = EnvironmentString("ASCEND_BENCH_MODES");
  if (requested_modes.empty()) {
    options.modes = {TimingMode::kPipelined, TimingMode::kDeviceEvents, TimingMode::kHostWallClock};
  } else {
    for (const std::string& name : SplitOnCommas(requested_modes)) {
      if (name == "pipelined") {
        options.modes.push_back(TimingMode::kPipelined);
      } else if (name == "device") {
        options.modes.push_back(TimingMode::kDeviceEvents);
      } else if (name == "host") {
        options.modes.push_back(TimingMode::kHostWallClock);
      } else {
        std::fprintf(stderr, "[ascend-bench] unknown mode '%s' in ASCEND_BENCH_MODES, expected one of "
                             "pipelined,device,host\n",
                     name.c_str());
      }
    }
    if (options.modes.empty()) {
      options.modes = {TimingMode::kPipelined};
    }
  }
  return options;
}

// ---------------------------------------------------------------------------
// LatencyStatistics
// ---------------------------------------------------------------------------

LatencyStatistics LatencyStatistics::From(std::vector<double> samples_us) {
  LatencyStatistics statistics;
  if (samples_us.empty()) {
    return statistics;
  }
  std::sort(samples_us.begin(), samples_us.end());
  const size_t count = samples_us.size();
  statistics.sample_count = count;
  statistics.min_us = samples_us.front();
  statistics.max_us = samples_us.back();

  // Even counts take the mean of the two middle samples, matching numpy's
  // default so a figure here can be compared with one computed from the CSV.
  statistics.median_us = (count % 2 == 1) ? samples_us[count / 2]
                                          : 0.5 * (samples_us[count / 2 - 1] + samples_us[count / 2]);

  double sum = 0.0;
  for (double sample : samples_us) {
    sum += sample;
  }
  statistics.mean_us = sum / static_cast<double>(count);

  double sum_of_squares = 0.0;
  for (double sample : samples_us) {
    const double deviation = sample - statistics.mean_us;
    sum_of_squares += deviation * deviation;
  }
  // Population standard deviation: these are all the samples taken, not a
  // sample drawn from a larger set.
  statistics.stddev_us = std::sqrt(sum_of_squares / static_cast<double>(count));

  // Nearest-rank percentiles, so every reported figure is an observed sample.
  const auto percentile = [&samples_us, count](double fraction) {
    size_t rank = static_cast<size_t>(std::ceil(fraction * static_cast<double>(count)));
    if (rank == 0) {
      rank = 1;
    }
    if (rank > count) {
      rank = count;
    }
    return samples_us[rank - 1];
  };
  statistics.p95_us = percentile(0.95);
  statistics.p99_us = percentile(0.99);
  return statistics;
}

// ---------------------------------------------------------------------------
// Checksums
// ---------------------------------------------------------------------------

double ChecksumSum(const std::vector<float>& values) {
  double total = 0.0;
  for (float value : values) {
    total += static_cast<double>(value);
  }
  return total;
}

double ChecksumSumOfSquares(const std::vector<float>& values) {
  double total = 0.0;
  for (float value : values) {
    const double widened = static_cast<double>(value);
    total += widened * widened;
  }
  return total;
}

// ---------------------------------------------------------------------------
// BenchmarkResult
// ---------------------------------------------------------------------------

double BenchmarkResult::tflops() const {
  if (flops_per_iteration <= 0.0 || latency.median_us <= 0.0) {
    return 0.0;
  }
  // FLOP / us -> FLOP / s is 1e6; FLOP/s -> TFLOP/s is 1e-12.
  return flops_per_iteration / latency.median_us * 1e-6;
}

double BenchmarkResult::gigabytes_per_second() const {
  if (bytes_per_iteration <= 0.0 || latency.median_us <= 0.0) {
    return 0.0;
  }
  // byte / us -> byte / s is 1e6; byte/s -> GB/s is 1e-9. GB is 1e9 bytes here,
  // not 2^30, which is how HBM bandwidth is specified.
  return bytes_per_iteration / latency.median_us * 1e-3;
}

// ---------------------------------------------------------------------------
// PlannedOp
// ---------------------------------------------------------------------------

void PlannedOp::SetRepeatableExecutorAllowed(bool allowed) { g_repeatable_allowed = allowed; }

const char* PlannedOp::LaunchPathLabel() {
  switch (g_launch_path) {
    case LaunchPath::kRepeatable:
      return "repeatable-executor (plan once, launch only in the timed loop)";
    case LaunchPath::kDisabledByEnvironment:
      return "replan-per-launch (ASCEND_BENCH_REPEATABLE=0)";
    case LaunchPath::kRejectedByRuntime:
      return "replan-per-launch (aclSetAclOpExecutorRepeatable rejected the executor)";
    case LaunchPath::kSymbolAbsent:
      break;
  }
  return "replan-per-launch (aclSetAclOpExecutorRepeatable not exported by this CANN build)";
}

PlannedOp::PlannedOp(std::string op_name, void* launch_fn, Planner planner)
    : op_name_(std::move(op_name)), launch_fn_(launch_fn), planner_(std::move(planner)) {
  if (launch_fn_ == nullptr) {
    throw AclError((op_name_ + ": launch entry point not resolved").c_str(), __FILE__, __LINE__, -1);
  }

  const int status = planner_(&workspace_size_, &executor_);
  if (status != 0) {
    const std::string label = op_name_ + "GetWorkspaceSize";
    throw AclError(label.c_str(), __FILE__, __LINE__, status);
  }
  if (workspace_size_ > 0) {
    workspace_.Allocate(static_cast<size_t>(workspace_size_), kBenchmarkAlignBytes);
  }

  const ExecutorApi& api = ExecutorApi::Instance();
  if (!g_repeatable_allowed) {
    g_launch_path = LaunchPath::kDisabledByEnvironment;
  } else if (api.set_repeatable == nullptr) {
    g_launch_path = LaunchPath::kSymbolAbsent;
  } else if (api.set_repeatable(executor_) != 0) {
    // The executor is still valid and will be consumed by the first Launch.
    g_launch_path = LaunchPath::kRejectedByRuntime;
  } else {
    repeatable_ = true;
    g_launch_path = LaunchPath::kRepeatable;
  }
}

PlannedOp::PlannedOp(PlannedOp&& other) noexcept
    : op_name_(std::move(other.op_name_)),
      launch_fn_(other.launch_fn_),
      planner_(std::move(other.planner_)),
      workspace_(std::move(other.workspace_)),
      workspace_size_(other.workspace_size_),
      executor_(other.executor_),
      repeatable_(other.repeatable_) {
  other.launch_fn_ = nullptr;
  other.executor_ = nullptr;
  other.repeatable_ = false;
  other.workspace_size_ = 0;
}

PlannedOp::~PlannedOp() { DestroyExecutor(); }

void PlannedOp::DestroyExecutor() {
  // executor_ is non-null exactly when an executor exists that no launch has
  // consumed: a repeatable one, which opted out of being consumed and is ours
  // to free, or the plan from the constructor on a case that threw before its
  // first launch. Both need destroying; a consumed one has already been zeroed.
  if (executor_ == nullptr) {
    return;
  }
  const ExecutorApi& api = ExecutorApi::Instance();
  if (api.destroy != nullptr) {
    api.destroy(executor_);
  }
  executor_ = nullptr;
}

void PlannedOp::Launch(aclrtStream stream) {
  aclOpExecutor* executor = executor_;
  if (executor == nullptr) {
    // Fallback path: the previous launch consumed the executor, so plan again.
    // The workspace is not reallocated - the arguments are identical, so the
    // size cannot change, and a runtime that disagrees is a bug worth failing on.
    uint64_t workspace_size = 0;
    const int status = planner_(&workspace_size, &executor);
    if (status != 0) {
      const std::string label = op_name_ + "GetWorkspaceSize";
      throw AclError(label.c_str(), __FILE__, __LINE__, status);
    }
    if (workspace_size > workspace_size_) {
      throw AclError((op_name_ + ": workspace grew between identical plans").c_str(), __FILE__, __LINE__, -1);
    }
  }

  using LaunchFn = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
  const int status =
      reinterpret_cast<LaunchFn>(launch_fn_)(workspace_.get(), workspace_size_, executor, stream);
  if (status != 0) {
    throw AclError(op_name_.c_str(), __FILE__, __LINE__, status);
  }
  if (!repeatable_) {
    executor_ = nullptr;  // consumed by the launch above
  }
}

// ---------------------------------------------------------------------------
// BenchmarkRunner
// ---------------------------------------------------------------------------

BenchmarkRunner::BenchmarkRunner(std::string suite_name, const BenchmarkOptions& options, aclrtStream stream)
    : suite_name_(std::move(suite_name)), options_(options), stream_(stream) {}

void BenchmarkRunner::Skip(const std::string& case_name, const std::string& reason) {
  skipped_.push_back(SkippedCase{case_name, reason});
}

void BenchmarkRunner::RecordFailure(const std::string& case_name, const std::string& reason) {
  failures_.push_back(SkippedCase{case_name, reason});
}

void BenchmarkRunner::DrainIfQueueIsDeep(int* enqueued) {
  if (*enqueued < kMaxInFlightTasks) {
    return;
  }
  ACL_CHECK(aclrtSynchronizeStream(stream_));
  *enqueued = 0;
}

std::vector<double> BenchmarkRunner::TimePipelined(const BenchmarkCase& benchmark_case) {
  const size_t samples = static_cast<size_t>(options_.timed_iterations);
  const int batch = options_.pipeline_batch;
  EventPool events(samples);

  // Nothing between the launches: the whole point of this mode is to let the
  // runtime keep the pipeline full.
  int enqueued = 0;
  for (size_t sample = 0; sample < samples; ++sample) {
    DrainIfQueueIsDeep(&enqueued);
    ACL_CHECK(aclrtRecordEvent(events.start(sample), stream_));
    for (int i = 0; i < batch; ++i) {
      benchmark_case.launch(stream_);
    }
    ACL_CHECK(aclrtRecordEvent(events.stop(sample), stream_));
    enqueued += batch + 2;
  }
  ACL_CHECK(aclrtSynchronizeStream(stream_));

  std::vector<double> per_iteration_us(samples, 0.0);
  for (size_t sample = 0; sample < samples; ++sample) {
    per_iteration_us[sample] =
        ElapsedMicroseconds(events.start(sample), events.stop(sample)) / static_cast<double>(batch);
  }
  return per_iteration_us;
}

std::vector<double> BenchmarkRunner::TimeDeviceEvents(const BenchmarkCase& benchmark_case) {
  const size_t samples = static_cast<size_t>(options_.timed_iterations);
  EventPool events(samples);

  // Every event is recorded before anything is read back, so the loop never
  // waits on the host.
  int enqueued = 0;
  for (size_t sample = 0; sample < samples; ++sample) {
    DrainIfQueueIsDeep(&enqueued);
    ACL_CHECK(aclrtRecordEvent(events.start(sample), stream_));
    benchmark_case.launch(stream_);
    ACL_CHECK(aclrtRecordEvent(events.stop(sample), stream_));
    enqueued += 3;
  }
  ACL_CHECK(aclrtSynchronizeStream(stream_));

  std::vector<double> per_iteration_us(samples, 0.0);
  for (size_t sample = 0; sample < samples; ++sample) {
    per_iteration_us[sample] = ElapsedMicroseconds(events.start(sample), events.stop(sample));
  }
  return per_iteration_us;
}

std::vector<double> BenchmarkRunner::TimeHostWallClock(const BenchmarkCase& benchmark_case) {
  const size_t samples = static_cast<size_t>(options_.timed_iterations);
  std::vector<double> per_iteration_us(samples, 0.0);

  // steady_clock rather than high_resolution_clock: on libstdc++ the latter is
  // an alias for system_clock, which is not monotonic and can step under NTP
  // mid-run. Both have nanosecond resolution here.
  for (size_t sample = 0; sample < samples; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    benchmark_case.launch(stream_);
    ACL_CHECK(aclrtSynchronizeStream(stream_));
    const auto finished = std::chrono::steady_clock::now();
    per_iteration_us[sample] =
        std::chrono::duration<double, std::micro>(finished - started).count();
  }
  return per_iteration_us;
}

void BenchmarkRunner::Run(const BenchmarkCase& benchmark_case) {
  try {
    RunOrThrow(benchmark_case);
  } catch (...) {
    // The caller's handler is about to destroy the tensors and the workspace
    // this case launched against, and a failed launch can leave earlier work
    // still in flight. Drain first, or the next case reports a fault that has
    // nothing to do with it.
    ACL_CHECK_NOTHROW(aclrtSynchronizeStream(stream_));
    throw;
  }
}

void BenchmarkRunner::RunOrThrow(const BenchmarkCase& benchmark_case) {
  // Warmup: kernel compilation and caching, driver-side first-touch of the
  // workspace, and AI Core clock ramp all happen here rather than in sample 0.
  for (int i = 0; i < options_.warmup_iterations; ++i) {
    benchmark_case.launch(stream_);
  }
  ACL_CHECK(aclrtSynchronizeStream(stream_));

  double checksum_after_warmup = 0.0;
  if (benchmark_case.checksum) {
    checksum_after_warmup = benchmark_case.checksum();
    if (!std::isfinite(checksum_after_warmup)) {
      throw std::runtime_error(benchmark_case.name +
                               ": output checksum is not finite after warmup; the operator did not produce a "
                               "usable result, so there is nothing worth timing");
    }
  }

  for (TimingMode mode : options_.modes) {
    std::vector<double> samples_us;
    switch (mode) {
      case TimingMode::kPipelined:
        samples_us = TimePipelined(benchmark_case);
        break;
      case TimingMode::kDeviceEvents:
        samples_us = TimeDeviceEvents(benchmark_case);
        break;
      case TimingMode::kHostWallClock:
        samples_us = TimeHostWallClock(benchmark_case);
        break;
    }

    BenchmarkResult result;
    result.case_name = benchmark_case.name;
    result.mode = mode;
    result.latency = LatencyStatistics::From(std::move(samples_us));
    result.flops_per_iteration = benchmark_case.flops_per_iteration;
    result.bytes_per_iteration = benchmark_case.bytes_per_iteration;
    results_.push_back(result);
  }

  if (benchmark_case.checksum) {
    CheckChecksum(benchmark_case, checksum_after_warmup, benchmark_case.checksum());
  }
}

void BenchmarkRunner::PrintTable() const {
  std::ostringstream table;
  table << std::fixed;

  table << "\n[ascend-bench] " << suite_name_ << "\n";
  table << "[ascend-bench]   warmup=" << options_.warmup_iterations
        << " iterations=" << options_.timed_iterations << " pipeline_batch=" << options_.pipeline_batch << "\n";
  table << "[ascend-bench]   launch path: " << PlannedOp::LaunchPathLabel() << "\n";
  table << "[ascend-bench]   times are microseconds per operator launch; percentiles are nearest-rank\n\n";

  // Column widths chosen so the longest shape label in any of the five suites
  // fits without the numbers moving.
  const int name_width = 34;
  table << "  " << std::left << std::setw(name_width) << "case" << std::setw(11) << "mode" << std::right
        << std::setw(10) << "min" << std::setw(10) << "median" << std::setw(10) << "mean" << std::setw(10)
        << "p95" << std::setw(10) << "p99" << std::setw(10) << "stddev" << std::setw(12) << "TFLOP/s"
        << std::setw(10) << "GB/s" << "\n";
  table << "  " << std::string(name_width + 11 + 10 * 6 + 12, '-') << "\n";

  std::string previous_case;
  for (const BenchmarkResult& result : results_) {
    // Blank line between cases keeps the three mode rows visually grouped.
    if (!previous_case.empty() && result.case_name != previous_case) {
      table << "\n";
    }
    previous_case = result.case_name;

    table << "  " << std::left << std::setw(name_width) << result.case_name << std::setw(11)
          << TimingModeLabel(result.mode) << std::right << std::setprecision(2) << std::setw(10)
          << result.latency.min_us << std::setw(10) << result.latency.median_us << std::setw(10)
          << result.latency.mean_us << std::setw(10) << result.latency.p95_us << std::setw(10)
          << result.latency.p99_us << std::setw(10) << result.latency.stddev_us;

    if (result.flops_per_iteration > 0.0) {
      table << std::setw(12) << std::setprecision(3) << result.tflops();
    } else {
      table << std::setw(12) << "-";
    }
    if (result.bytes_per_iteration > 0.0) {
      table << std::setw(10) << std::setprecision(1) << result.gigabytes_per_second();
    } else {
      table << std::setw(10) << "-";
    }
    table << "\n";
  }

  if (!skipped_.empty()) {
    table << "\n[ascend-bench] skipped:\n";
    for (const SkippedCase& entry : skipped_) {
      table << "  " << entry.name << ": " << entry.reason << "\n";
    }
  }
  if (!failures_.empty()) {
    table << "\n[ascend-bench] FAILED:\n";
    for (const SkippedCase& entry : failures_) {
      table << "  " << entry.name << ": " << entry.reason << "\n";
    }
  }

  std::fputs(table.str().c_str(), stdout);
  std::fflush(stdout);
}

void BenchmarkRunner::WriteCsv() const {
  if (options_.csv_path.empty()) {
    return;
  }
  std::ofstream csv(options_.csv_path);
  if (!csv) {
    std::fprintf(stderr, "[ascend-bench] could not open ASCEND_BENCH_CSV=%s for writing\n",
                 options_.csv_path.c_str());
    return;
  }
  csv << "suite,case,mode,samples,min_us,median_us,mean_us,p95_us,p99_us,max_us,stddev_us,tflops,gbps,"
         "flops_per_iter,bytes_per_iter,launch_path\n";
  csv << std::setprecision(9);
  for (const BenchmarkResult& result : results_) {
    csv << suite_name_ << "," << result.case_name << "," << TimingModeLabel(result.mode) << ","
        << result.latency.sample_count << "," << result.latency.min_us << "," << result.latency.median_us << ","
        << result.latency.mean_us << "," << result.latency.p95_us << "," << result.latency.p99_us << ","
        << result.latency.max_us << "," << result.latency.stddev_us << "," << result.tflops() << ","
        << result.gigabytes_per_second() << "," << result.flops_per_iteration << ","
        << result.bytes_per_iteration << ",\"" << PlannedOp::LaunchPathLabel() << "\"\n";
  }
  std::fprintf(stdout, "[ascend-bench] wrote %s\n", options_.csv_path.c_str());
}

void BenchmarkRunner::Report() const {
  PrintTable();
  WriteCsv();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int RunBenchmarkSuite(const char* suite_name, const std::function<void(BenchmarkRunner&)>& build) {
  std::printf("[ascend-bench] vllm-ascend 310P kernel microbenchmarks (no Python, no torch)\n");
  ops::PrintOperatorInventory();
  std::fflush(stdout);

  std::unique_ptr<AscendDevice> device;
  try {
    device.reset(new AscendDevice());
  } catch (const std::exception& error) {
    std::printf("[ascend-bench] no usable Ascend device, skipping the suite:\n%s\n", error.what());
    return kBenchmarkSkipExitCode;
  }

  const std::string& soc = device->soc_name();
  // Empty means the runtime would not tell us; the correctness suite treats
  // that the same way rather than refusing to run.
  if (!soc.empty() && soc.find("310P") == std::string::npos && soc.find("310p") == std::string::npos) {
    std::printf("[ascend-bench] attached device reports soc='%s', these benchmarks target Ascend 310P; "
                "skipping\n",
                soc.c_str());
    return kBenchmarkSkipExitCode;
  }
  std::printf("[ascend-bench] device %d ready, soc='%s'\n", device->device_id(),
              soc.empty() ? "<unknown>" : soc.c_str());

  const BenchmarkOptions options = BenchmarkOptions::FromEnvironment();
  PlannedOp::SetRepeatableExecutorAllowed(options.allow_repeatable_executor);

  // A stream of its own, so nothing the setup phase or the runtime left on the
  // device's default stream lands inside a timed region.
  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  int exit_code = 0;
  {
    HugeMemScope huge_mem;
    BenchmarkRunner runner(suite_name, options, stream);
    try {
      build(runner);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "[ascend-bench] suite aborted: %s\n", error.what());
      runner.RecordFailure("<suite>", error.what());
    }
    runner.Report();
    if (runner.failure_count() > 0) {
      exit_code = 1;
    } else if (!runner.has_results()) {
      // Everything skipped: nothing ran, so this is not a pass.
      exit_code = kBenchmarkSkipExitCode;
    }
  }

  ACL_CHECK_NOTHROW(aclrtSynchronizeStream(stream));
  ACL_CHECK_NOTHROW(aclrtDestroyStream(stream));
  return exit_code;
}

}  // namespace bench
}  // namespace test
}  // namespace vllm_ascend
