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
#include <cerrno>
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
#include "ascend950_shapes.hpp"
#include "test_harness.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {

namespace {

std::string EnvironmentString(const char* name) {
  const char* raw = std::getenv(name);
  return (raw != nullptr) ? std::string(raw) : std::string();
}

int EnvironmentInt(const char* name, int fallback, int minimum, int maximum) {
  const std::string raw = EnvironmentString(name);
  if (raw.empty()) {
    return fallback;
  }
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(raw.c_str(), &end, 10);
  const bool parsed_cleanly = (end != nullptr) && (end != raw.c_str()) && (*end == '\0') && (errno == 0);
  if (!parsed_cleanly || parsed < static_cast<long>(minimum) || parsed > static_cast<long>(maximum)) {
    std::fprintf(stderr, "[ascend-bench] %s=%s is not an integer in [%d, %d], using %d\n", name,
                 raw.c_str(), minimum, maximum, fallback);
    return fallback;
  }
  return static_cast<int>(parsed);
}

std::vector<std::string> SplitOnCommas(const std::string& value) {
  std::vector<std::string> parts;
  std::string current;
  std::istringstream stream(value);
  while (std::getline(stream, current, ',')) {
    const size_t first = current.find_first_not_of(" \t");
    const size_t last = current.find_last_not_of(" \t");
    if (first != std::string::npos) {
      parts.push_back(current.substr(first, last - first + 1));
    }
  }
  return parts;
}

using SetExecutorRepeatableFn = int (*)(aclOpExecutor*);
using DestroyExecutorFn = int (*)(aclOpExecutor*);

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

enum class LaunchPath {
  kRepeatable,
  kSymbolAbsent,
  kDisabledByEnvironment,
  kRejectedByRuntime,
};

LaunchPath g_launch_path = LaunchPath::kSymbolAbsent;
bool g_repeatable_allowed = true;

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

enum class EventTimingSource {
  kTimelineAndSync,
  kTimelineOnly,
  kRuntimeDefault,
};

EventTimingSource ResolveEventTimingSource() {
  static const EventTimingSource resolved = [] {
    aclrtEvent probe = nullptr;
    (void)probe;
#ifdef ACL_EVENT_TIME_LINE
    if (aclrtCreateEventWithFlag(&probe, ACL_EVENT_TIME_LINE | ACL_EVENT_SYNC) == ACL_SUCCESS) {
      ACL_CHECK_NOTHROW(aclrtDestroyEvent(probe));
      return EventTimingSource::kTimelineAndSync;
    }
    probe = nullptr;
    if (aclrtCreateEventWithFlag(&probe, ACL_EVENT_TIME_LINE) == ACL_SUCCESS) {
      ACL_CHECK_NOTHROW(aclrtDestroyEvent(probe));
      return EventTimingSource::kTimelineOnly;
    }
#endif
    return EventTimingSource::kRuntimeDefault;
  }();
  return resolved;
}

aclrtEvent CreateTimingEvent() {
  aclrtEvent event = nullptr;
#ifdef ACL_EVENT_TIME_LINE
  const EventTimingSource source = ResolveEventTimingSource();
  if (source == EventTimingSource::kTimelineAndSync) {
    ACL_CHECK(aclrtCreateEventWithFlag(&event, ACL_EVENT_TIME_LINE | ACL_EVENT_SYNC));
    return event;
  }
  if (source == EventTimingSource::kTimelineOnly) {
    ACL_CHECK(aclrtCreateEventWithFlag(&event, ACL_EVENT_TIME_LINE));
    return event;
  }
#endif
  ACL_CHECK(aclrtCreateEvent(&event));
  return event;
}

class EventPool {
 public:
  explicit EventPool(size_t pair_count) : events_(pair_count * 2, nullptr) {
    try {
      for (size_t i = 0; i < events_.size(); ++i) {
        events_[i] = CreateTimingEvent();
      }
    } catch (...) {
      Destroy();
      throw;
    }
  }

  ~EventPool() { Destroy(); }

  EventPool(const EventPool&) = delete;
  EventPool& operator=(const EventPool&) = delete;

  aclrtEvent start(size_t index) const { return events_[index * 2]; }
  aclrtEvent stop(size_t index) const { return events_[index * 2 + 1]; }

 private:
  void Destroy() {
    for (aclrtEvent& event : events_) {
      if (event != nullptr) {
        ACL_CHECK_NOTHROW(aclrtDestroyEvent(event));
        event = nullptr;
      }
    }
  }

  std::vector<aclrtEvent> events_;
};

constexpr int kMaxInFlightTasks = 768;

constexpr int kMaxPipelineBatch = kMaxInFlightTasks - 2;

bool ReadElapsedMicroseconds(aclrtEvent start, aclrtEvent stop, double* microseconds) {
  if (ResolveEventTimingSource() != EventTimingSource::kTimelineOnly) {
    ACL_CHECK(aclrtSynchronizeEvent(start));
    ACL_CHECK(aclrtSynchronizeEvent(stop));
  }

  float milliseconds = 0.0f;
  ACL_CHECK(aclrtEventElapsedTime(&milliseconds, start, stop));

  const double elapsed_us = static_cast<double>(milliseconds) * 1000.0;
  if (!std::isfinite(elapsed_us) || elapsed_us < 0.0) {
    return false;
  }
  *microseconds = elapsed_us;
  return true;
}

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

}

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

const char* EventTimingSourceLabel() {
  switch (ResolveEventTimingSource()) {
    case EventTimingSource::kTimelineAndSync:
      return "ACL_EVENT_TIME_LINE|ACL_EVENT_SYNC (timestamped, host-waitable)";
    case EventTimingSource::kTimelineOnly:
      return "ACL_EVENT_TIME_LINE (timestamped; not host-waitable, stream barrier only)";
    case EventTimingSource::kRuntimeDefault:
      break;
  }
  return "aclrtCreateEvent default (NO timestamp guarantee - treat the device modes as suspect)";
}

BenchmarkOptions BenchmarkOptions::FromEnvironment() {
  BenchmarkOptions options;
  constexpr int kMaxIterationCount = 1000000;
  options.warmup_iterations =
      EnvironmentInt("ASCEND_BENCH_WARMUP", options.warmup_iterations, 0, kMaxIterationCount);
  options.timed_iterations =
      EnvironmentInt("ASCEND_BENCH_ITERS", options.timed_iterations, 1, kMaxIterationCount);
  options.pipeline_batch =
      EnvironmentInt("ASCEND_BENCH_BATCH", options.pipeline_batch, 1, kMaxPipelineBatch);
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

LatencyStatistics LatencyStatistics::From(std::vector<double> samples_us) {
  LatencyStatistics statistics;

  const size_t requested = samples_us.size();
  samples_us.erase(std::remove_if(samples_us.begin(), samples_us.end(),
                                  [](double sample) { return !std::isfinite(sample) || sample <= 0.0; }),
                   samples_us.end());
  statistics.discarded_count = requested - samples_us.size();

  if (samples_us.empty()) {
    return statistics;
  }
  std::sort(samples_us.begin(), samples_us.end());
  const size_t count = samples_us.size();
  statistics.sample_count = count;
  statistics.min_us = samples_us.front();
  statistics.max_us = samples_us.back();

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
  statistics.stddev_us = std::sqrt(sum_of_squares / static_cast<double>(count));

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

double BenchmarkResult::tflops() const {
  if (flops_per_iteration <= 0.0 || latency.median_us <= 0.0) {
    return 0.0;
  }
  return flops_per_iteration / latency.median_us * 1e-6;
}

double BenchmarkResult::gigabytes_per_second() const {
  if (bytes_per_iteration <= 0.0 || latency.median_us <= 0.0) {
    return 0.0;
  }
  return bytes_per_iteration / latency.median_us * 1e-3;
}

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
  return "replan-per-launch (aclSetAclOpExecutorRepeatable / aclDestroyAclOpExecutor not both exported by "
         "this CANN build)";
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
  } else if (api.set_repeatable == nullptr || api.destroy == nullptr) {
    g_launch_path = LaunchPath::kSymbolAbsent;
  } else if (api.set_repeatable(executor_) != 0) {
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

void PlannedOp::DestroyOrphanedExecutor(aclOpExecutor* executor) {
  if (executor == nullptr) {
    return;
  }
  const ExecutorApi& api = ExecutorApi::Instance();
  if (api.destroy != nullptr) {
    api.destroy(executor);
  }
}

void PlannedOp::DestroyExecutor() {
  DestroyOrphanedExecutor(executor_);
  executor_ = nullptr;
}

void PlannedOp::Launch(aclrtStream stream) {
  aclOpExecutor* executor = executor_;
  const bool replanned = (executor == nullptr);
  if (replanned) {
    uint64_t workspace_size = 0;
    const int status = planner_(&workspace_size, &executor);
    if (status != 0) {
      const std::string label = op_name_ + "GetWorkspaceSize";
      throw AclError(label.c_str(), __FILE__, __LINE__, status);
    }
    if (workspace_size > workspace_size_) {
      DestroyOrphanedExecutor(executor);
      throw AclError((op_name_ + ": workspace grew between identical plans").c_str(), __FILE__, __LINE__, -1);
    }
  }

  using LaunchFn = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
  const int status =
      reinterpret_cast<LaunchFn>(launch_fn_)(workspace_.get(), workspace_size_, executor, stream);
  if (status != 0) {
    if (replanned) {
      DestroyOrphanedExecutor(executor);
    }
    throw AclError(op_name_.c_str(), __FILE__, __LINE__, status);
  }
  if (!repeatable_) {
    executor_ = nullptr;
  }
}

BenchmarkRunner::BenchmarkRunner(std::string suite_name, const BenchmarkOptions& options, aclrtStream stream)
    : suite_name_(std::move(suite_name)), options_(options), stream_(stream) {}

void BenchmarkRunner::set_options(const BenchmarkOptions& options) {
  if (!results_.empty()) {
    throw AclError("BenchmarkRunner::set_options after a case has run: the report header states one option "
                   "set for the whole table",
                   __FILE__, __LINE__, -1);
  }
  options_ = options;
}

void BenchmarkRunner::Skip(const std::string& case_name, const std::string& reason) {
  skipped_.push_back(SkippedCase{case_name, reason});
}

void BenchmarkRunner::RecordFailure(const std::string& case_name, const std::string& reason) {
  failures_.push_back(SkippedCase{case_name, reason});
}

void BenchmarkRunner::DrainIfQueueIsDeep(int* enqueued, int about_to_enqueue) {
  if (*enqueued + about_to_enqueue <= kMaxInFlightTasks) {
    return;
  }
  ACL_CHECK(aclrtSynchronizeStream(stream_));
  *enqueued = 0;
}

void BenchmarkRunner::WarmUp(const BenchmarkCase& benchmark_case) {
  const int tasks_per_launch = std::max(1, benchmark_case.tasks_per_launch);
  const int warmup_iterations =
      benchmark_case.checksum ? std::max(1, options_.warmup_iterations) : options_.warmup_iterations;
  int enqueued = 0;
  for (int i = 0; i < warmup_iterations; ++i) {
    DrainIfQueueIsDeep(&enqueued, tasks_per_launch);
    benchmark_case.launch(stream_);
    enqueued += tasks_per_launch;
  }
  ACL_CHECK(aclrtSynchronizeStream(stream_));
}

LatencySamples BenchmarkRunner::TimePipelined(const BenchmarkCase& benchmark_case) {
  const size_t samples = static_cast<size_t>(options_.timed_iterations);
  const int batch = options_.pipeline_batch;
  EventPool events(samples);

  const int tasks_per_sample = batch * std::max(1, benchmark_case.tasks_per_launch) + 2;
  int enqueued = 0;
  for (size_t sample = 0; sample < samples; ++sample) {
    DrainIfQueueIsDeep(&enqueued, tasks_per_sample);
    ACL_CHECK(aclrtRecordEvent(events.start(sample), stream_));
    for (int i = 0; i < batch; ++i) {
      benchmark_case.launch(stream_);
    }
    ACL_CHECK(aclrtRecordEvent(events.stop(sample), stream_));
    enqueued += tasks_per_sample;
  }
  ACL_CHECK(aclrtSynchronizeStream(stream_));

  LatencySamples result;
  result.microseconds.reserve(samples);
  for (size_t sample = 0; sample < samples; ++sample) {
    double elapsed_us = 0.0;
    if (!ReadElapsedMicroseconds(events.start(sample), events.stop(sample), &elapsed_us)) {
      ++result.rejected;
      continue;
    }
    result.microseconds.push_back(elapsed_us / static_cast<double>(batch));
  }
  return result;
}

LatencySamples BenchmarkRunner::TimeDeviceEvents(const BenchmarkCase& benchmark_case) {
  const size_t samples = static_cast<size_t>(options_.timed_iterations);
  EventPool events(samples);

  const int tasks_per_sample = std::max(1, benchmark_case.tasks_per_launch) + 2;
  int enqueued = 0;
  for (size_t sample = 0; sample < samples; ++sample) {
    DrainIfQueueIsDeep(&enqueued, tasks_per_sample);
    ACL_CHECK(aclrtRecordEvent(events.start(sample), stream_));
    benchmark_case.launch(stream_);
    ACL_CHECK(aclrtRecordEvent(events.stop(sample), stream_));
    enqueued += tasks_per_sample;
  }
  ACL_CHECK(aclrtSynchronizeStream(stream_));

  LatencySamples result;
  result.microseconds.reserve(samples);
  for (size_t sample = 0; sample < samples; ++sample) {
    double elapsed_us = 0.0;
    if (!ReadElapsedMicroseconds(events.start(sample), events.stop(sample), &elapsed_us)) {
      ++result.rejected;
      continue;
    }
    result.microseconds.push_back(elapsed_us);
  }
  return result;
}

LatencySamples BenchmarkRunner::TimeHostWallClock(const BenchmarkCase& benchmark_case) {
  const size_t samples = static_cast<size_t>(options_.timed_iterations);
  LatencySamples result;
  result.microseconds.reserve(samples);

  for (size_t sample = 0; sample < samples; ++sample) {
    const auto started = std::chrono::steady_clock::now();
    benchmark_case.launch(stream_);
    ACL_CHECK(aclrtSynchronizeStream(stream_));
    const auto finished = std::chrono::steady_clock::now();
    const int64_t elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count();
    result.microseconds.push_back(static_cast<double>(elapsed_ns) / 1000.0);
  }
  return result;
}

void BenchmarkRunner::Run(const BenchmarkCase& benchmark_case) {
  try {
    RunOrThrow(benchmark_case);
  } catch (...) {
    ACL_CHECK_NOTHROW(aclrtSynchronizeStream(stream_));
    throw;
  }
}

void BenchmarkRunner::RunOrThrow(const BenchmarkCase& benchmark_case) {
  WarmUp(benchmark_case);

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
    LatencySamples samples;
    switch (mode) {
      case TimingMode::kPipelined:
        samples = TimePipelined(benchmark_case);
        break;
      case TimingMode::kDeviceEvents:
        samples = TimeDeviceEvents(benchmark_case);
        break;
      case TimingMode::kHostWallClock:
        samples = TimeHostWallClock(benchmark_case);
        break;
    }

    LatencyStatistics latency = LatencyStatistics::From(std::move(samples.microseconds));
    latency.discarded_count += samples.rejected;
    const size_t attempted = latency.discarded_count + latency.sample_count;

    if (latency.sample_count == 0) {
      std::ostringstream message;
      message << benchmark_case.name << ": all " << attempted << " " << TimingModeLabel(mode)
              << " samples were rejected as not a usable duration. Device timings come from "
              << EventTimingSourceLabel()
              << ". A row of zeros here would read as a measurement, so the case fails instead.";
      throw std::runtime_error(message.str());
    }
    if (latency.discarded_count > 0) {
      std::fprintf(stderr,
                   "[ascend-bench] %s/%s: discarded %zu of %zu samples that were not a usable duration\n",
                   benchmark_case.name.c_str(), TimingModeLabel(mode), latency.discarded_count, attempted);
    }

    BenchmarkResult result;
    result.case_name = benchmark_case.name;
    result.mode = mode;
    result.latency = latency;
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
  table << "[ascend-bench]   device timing events: " << EventTimingSourceLabel() << "\n";
  table << "[ascend-bench]   times are microseconds per operator launch; percentiles are nearest-rank\n\n";

  const int name_width = 34;
  table << "  " << std::left << std::setw(name_width) << "case" << std::setw(11) << "mode" << std::right
        << std::setw(10) << "min" << std::setw(10) << "median" << std::setw(10) << "mean" << std::setw(10)
        << "p95" << std::setw(10) << "p99" << std::setw(10) << "stddev" << std::setw(12) << "TFLOP/s"
        << std::setw(10) << "GB/s" << "\n";
  table << "  " << std::string(name_width + 11 + 10 * 6 + 12, '-') << "\n";

  std::string previous_case;
  for (const BenchmarkResult& result : results_) {
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

  size_t total_discarded = 0;
  for (const BenchmarkResult& result : results_) {
    total_discarded += result.latency.discarded_count;
  }
  if (total_discarded > 0) {
    table << "\n[ascend-bench] discarded samples (not a usable duration; excluded from every figure "
             "above):\n";
    for (const BenchmarkResult& result : results_) {
      if (result.latency.discarded_count == 0) {
        continue;
      }
      table << "  " << result.case_name << " / " << TimingModeLabel(result.mode) << ": "
            << result.latency.discarded_count << " of "
            << (result.latency.discarded_count + result.latency.sample_count) << "\n";
    }
    table << "  device timing events: " << EventTimingSourceLabel() << "\n";
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
  csv << "suite,case,mode,samples,discarded,min_us,median_us,mean_us,p95_us,p99_us,max_us,stddev_us,tflops,"
         "gbps,flops_per_iter,bytes_per_iter,launch_path,event_timing\n";
  csv << std::setprecision(9);
  for (const BenchmarkResult& result : results_) {
    csv << suite_name_ << "," << result.case_name << "," << TimingModeLabel(result.mode) << ","
        << result.latency.sample_count << "," << result.latency.discarded_count << ","
        << result.latency.min_us << "," << result.latency.median_us << ","
        << result.latency.mean_us << "," << result.latency.p95_us << "," << result.latency.p99_us << ","
        << result.latency.max_us << "," << result.latency.stddev_us << "," << result.tflops() << ","
        << result.gigabytes_per_second() << "," << result.flops_per_iteration << ","
        << result.bytes_per_iteration << ",\"" << PlannedOp::LaunchPathLabel() << "\",\""
        << EventTimingSourceLabel() << "\"\n";
  }
  std::fprintf(stdout, "[ascend-bench] wrote %s\n", options_.csv_path.c_str());
}

void BenchmarkRunner::Report() const {
  PrintTable();
  WriteCsv();
}

namespace {

bool SocMatchesTargetPart(const std::string& soc, BenchmarkTargetPart part) {
  switch (part) {
    case BenchmarkTargetPart::kAscend310P:
      return soc.empty() || soc.find("310P") != std::string::npos || soc.find("310p") != std::string::npos;
    case BenchmarkTargetPart::kAscend950PR:
      return shapes950::IsAscend950PrSocName(soc);
  }
  return false;
}

const char* TargetPartLabel(BenchmarkTargetPart part) {
  return part == BenchmarkTargetPart::kAscend310P ? "Ascend 310P" : "Ascend 950PR";
}

}

int RunBenchmarkSuite(const char* suite_name, const std::function<void(BenchmarkRunner&)>& build) {
  return RunBenchmarkSuite(suite_name, BenchmarkTargetPart::kAscend310P, build);
}

int RunBenchmarkSuite(const char* suite_name, BenchmarkTargetPart part,
                      const std::function<void(BenchmarkRunner&)>& build) {
  std::printf("[ascend-bench] vllm-ascend %s kernel microbenchmarks (no Python, no torch)\n",
              TargetPartLabel(part));
  if (part == BenchmarkTargetPart::kAscend310P) {
    ops::PrintOperatorInventory();
  }
  std::fflush(stdout);

  std::unique_ptr<AscendDevice> device;
  try {
    device.reset(new AscendDevice());
  } catch (const std::exception& error) {
    std::printf("[ascend-bench] no usable Ascend device, skipping the suite:\n%s\n", error.what());
    return kBenchmarkSkipExitCode;
  }

  const std::string& soc = device->soc_name();
  if (!SocMatchesTargetPart(soc, part)) {
    std::printf("[ascend-bench] attached device reports soc='%s', these benchmarks target %s; skipping\n",
                soc.empty() ? "<unknown>" : soc.c_str(), TargetPartLabel(part));
    return kBenchmarkSkipExitCode;
  }
  std::printf("[ascend-bench] device %d ready, soc='%s'\n", device->device_id(),
              soc.empty() ? "<unknown>" : soc.c_str());

  const BenchmarkOptions options = BenchmarkOptions::FromEnvironment();
  PlannedOp::SetRepeatableExecutorAllowed(options.allow_repeatable_executor);

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
      exit_code = kBenchmarkSkipExitCode;
    }
  }

  ACL_CHECK_NOTHROW(aclrtSynchronizeStream(stream));
  ACL_CHECK_NOTHROW(aclrtDestroyStream(stream));
  return exit_code;
}

}
}
}
