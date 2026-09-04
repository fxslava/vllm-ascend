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

// strtol rather than atoi: atoi cannot tell "0" from "banana", and it is
// undefined on a value too large for int rather than reporting it. Both used to
// land silently on a plausible-looking configuration - ASCEND_BENCH_WARMUP with
// a typo in it became zero warmup, which is exactly the setting whose absence
// the first sample of every case then pays for.
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

// How the events the device modes time with were actually created.
//
// ACL_EVENT_TIME_LINE is what makes an event carry a device timestamp, and
// therefore what makes aclrtEventElapsedTime mean anything at all; ACL_EVENT_SYNC
// additionally makes it host-waitable, which is what lets the readback resolve
// each event before querying it. Plain aclrtCreateEvent guarantees neither, and
// asking for the elapsed time between two events that carry no timestamp is
// undefined - in practice a large negative number, which is the shape the bad
// latencies in the reports had.
enum class EventTimingSource {
  kTimelineAndSync,
  kTimelineOnly,
  kRuntimeDefault,
};

// Resolved once per process with a single probe event, so the per-event path is
// one call that either works or fails loudly. Probing per event would also
// leave a stale aclGetRecentErrMsg behind every unsupported flag combination,
// which is exactly the diagnostic a real failure needs to carry.
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

// A pool of start/stop event pairs, created before the timed loop and destroyed
// after it. Creating a fresh pair per iteration keeps the loop free of the
// re-record semantics aclrtResetEvent exists for, which differ across CANN
// releases, and event creation is a setup-phase cost either way.
class EventPool {
 public:
  explicit EventPool(size_t pair_count) : events_(pair_count * 2, nullptr) {
    try {
      for (size_t i = 0; i < events_.size(); ++i) {
        events_[i] = CreateTimingEvent();
      }
    } catch (...) {
      // A constructor that throws gets no destructor, so the events created
      // before the failure have to go back here. Without this, a run that hits
      // the device's event limit part-way through leaks a pool's worth of
      // events per attempt and every later case fails for the same reason.
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

// An Ascend stream holds a bounded number of submitted tasks; past that the
// runtime either blocks the host or refuses the submission, depending on the
// CANN release. At the defaults a pipelined run enqueues 100 * (10 + 2) = 1200
// tasks, which is over the 1024 that releases documenting a limit give, so the
// timing loops drain at sample boundaries once they get close. Two drains per
// run at most, and never inside a batch.
constexpr int kMaxInFlightTasks = 768;

// Two events bracket a pipelined batch, so this is the largest batch that can
// still be submitted without the runtime blocking the host part-way through a
// sample. A batch past this point does not measure a fuller pipeline, it
// measures the runtime's back-pressure, so ASCEND_BENCH_BATCH is clamped to it
// with a warning rather than quietly producing a serialised "pipelined" number.
constexpr int kMaxPipelineBatch = kMaxInFlightTasks - 2;

// Reads the time between two recorded events, in microseconds.
//
// Returns false, without throwing, when the runtime hands back something that
// cannot be a duration: negative, NaN or infinite. That happens when an event
// carries no timestamp, when the device counter behind it wrapped, and when the
// device has not finished with the event yet. One such sample in a set drags
// the mean below zero and takes the TFLOP/s and GB/s derived from the median
// with it, so the sample is dropped and counted instead of being aggregated.
//
// Both events are resolved first. The timing loops already drain the whole
// stream before reading anything back, so these return immediately; they are
// here so that a caller which loses that barrier waits rather than reading a
// timestamp the device has not written. A TIME_LINE-only event is not
// guaranteed to be host-waitable, so for that one case the stream barrier is
// the only guarantee available and is left to do the job.
bool ReadElapsedMicroseconds(aclrtEvent start, aclrtEvent stop, double* microseconds) {
  if (ResolveEventTimingSource() != EventTimingSource::kTimelineOnly) {
    ACL_CHECK(aclrtSynchronizeEvent(start));
    ACL_CHECK(aclrtSynchronizeEvent(stop));
  }

  float milliseconds = 0.0f;
  ACL_CHECK(aclrtEventElapsedTime(&milliseconds, start, stop));

  // Widen before the scale, and stay in double from here to the report: the
  // conversion never passes through a narrower integer, which is the other way
  // a long sample comes out negative.
  const double elapsed_us = static_cast<double>(milliseconds) * 1000.0;
  if (!std::isfinite(elapsed_us) || elapsed_us < 0.0) {
    return false;
  }
  *microseconds = elapsed_us;
  return true;
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
  // An upper bound on the loop counts as well as a lower one. The event pool a
  // timed loop builds is two events per iteration, so an ASCEND_BENCH_ITERS
  // with an extra digit in it used to be reported as a device-side event
  // allocation failure half an hour into a sweep rather than as the typo it is.
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

// ---------------------------------------------------------------------------
// LatencyStatistics
// ---------------------------------------------------------------------------

LatencyStatistics LatencyStatistics::From(std::vector<double> samples_us) {
  LatencyStatistics statistics;

  // The bounds check, before anything is accumulated. A sample that is not a
  // positive finite duration is not a duration: it is a wrapped or unresolved
  // device counter, an event with no timestamp, or a host clock that went
  // backwards. Dropping it here keeps one bad reading from pulling the mean -
  // and the TFLOP/s and GB/s the report derives from the median - somewhere
  // impossible, and the count is carried out so the drop is never silent.
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
    // Both halves or neither. A repeatable executor opts out of being consumed
    // by the launch, so a build that can mark one but cannot destroy one would
    // strand an executor per case for the length of the sweep; re-planning is
    // slower but bounded.
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
  // executor_ is non-null exactly when an executor exists that no launch has
  // consumed: a repeatable one, which opted out of being consumed and is ours
  // to free, or the plan from the constructor on a case that threw before its
  // first launch. Both need destroying; a consumed one has already been zeroed.
  DestroyOrphanedExecutor(executor_);
  executor_ = nullptr;
}

void PlannedOp::Launch(aclrtStream stream) {
  aclOpExecutor* executor = executor_;
  const bool replanned = (executor == nullptr);
  if (replanned) {
    // Fallback path: the previous launch consumed the executor, so plan again.
    // The workspace is not reallocated - the arguments are identical, so the
    // size cannot change, and a runtime that disagrees is a bug worth failing
    // on. It is the buffer PlannedOp has owned since the constructor either
    // way, so the pointer the launch below is handed stays valid and stays put
    // for every iteration of every loop in the run.
    uint64_t workspace_size = 0;
    const int status = planner_(&workspace_size, &executor);
    if (status != 0) {
      const std::string label = op_name_ + "GetWorkspaceSize";
      throw AclError(label.c_str(), __FILE__, __LINE__, status);
    }
    if (workspace_size > workspace_size_) {
      // This executor is local to the call and nothing else will free it.
      DestroyOrphanedExecutor(executor);
      throw AclError((op_name_ + ": workspace grew between identical plans").c_str(), __FILE__, __LINE__, -1);
    }
  }

  using LaunchFn = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
  const int status =
      reinterpret_cast<LaunchFn>(launch_fn_)(workspace_.get(), workspace_size_, executor, stream);
  if (status != 0) {
    // A failed launch does not consume the executor. The one from the
    // constructor is still in executor_ for the destructor to deal with; a
    // re-planned one exists only in this frame and has to go back here, or a
    // suite that fails one case per shape strands one executor per failure.
    if (replanned) {
      DestroyOrphanedExecutor(executor);
    }
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

void BenchmarkRunner::DrainIfQueueIsDeep(int* enqueued, int about_to_enqueue) {
  if (*enqueued + about_to_enqueue <= kMaxInFlightTasks) {
    return;
  }
  ACL_CHECK(aclrtSynchronizeStream(stream_));
  *enqueued = 0;
}

void BenchmarkRunner::WarmUp(const BenchmarkCase& benchmark_case) {
  // Kernel compilation and caching, the driver-side first touch of the
  // workspace, and the AI Core clock ramp all happen here rather than in sample
  // 0. A long warmup is drained on the way so it cannot overrun the queue.
  int enqueued = 0;
  for (int i = 0; i < options_.warmup_iterations; ++i) {
    DrainIfQueueIsDeep(&enqueued, 1);
    benchmark_case.launch(stream_);
    ++enqueued;
  }
  // The hard barrier between warmup and timing. Every warmup launch, and every
  // MTE transfer it queued, has retired before the first sample is recorded, so
  // sample 0 is never charged for the backlog behind it.
  ACL_CHECK(aclrtSynchronizeStream(stream_));
}

LatencySamples BenchmarkRunner::TimePipelined(const BenchmarkCase& benchmark_case) {
  const size_t samples = static_cast<size_t>(options_.timed_iterations);
  const int batch = options_.pipeline_batch;
  EventPool events(samples);

  // Nothing between the launches: the whole point of this mode is to let the
  // runtime keep the pipeline full. The batch plus both events is what one
  // sample puts on the stream, and the drain is told about all of it up front
  // so a deep batch cannot be split by the runtime's own back-pressure.
  const int tasks_per_sample = batch + 2;
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
  // Every launch and every event has retired before a timestamp is read.
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

  // Every event is recorded before anything is read back, so the loop never
  // waits on the host.
  constexpr int kTasksPerSample = 3;  // the launch, plus both events
  int enqueued = 0;
  for (size_t sample = 0; sample < samples; ++sample) {
    DrainIfQueueIsDeep(&enqueued, kTasksPerSample);
    ACL_CHECK(aclrtRecordEvent(events.start(sample), stream_));
    benchmark_case.launch(stream_);
    ACL_CHECK(aclrtRecordEvent(events.stop(sample), stream_));
    enqueued += kTasksPerSample;
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

  // steady_clock rather than high_resolution_clock: on libstdc++ the latter is
  // an alias for system_clock, which is not monotonic and can step under NTP
  // mid-run. Both have nanosecond resolution here.
  //
  // The difference is taken in the clock's own 64-bit nanosecond representation
  // and converted once, in double. Nothing on this path narrows: microseconds
  // in a 32-bit signed integer wrap after 2.14 seconds, which is well inside
  // the range a slow prefill matmul reaches, and a wrapped sample comes out
  // negative rather than large.
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
    // The caller's handler is about to destroy the tensors and the workspace
    // this case launched against, and a failed launch can leave earlier work
    // still in flight. Drain first, or the next case reports a fault that has
    // nothing to do with it.
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

    // Two ways a sample can be lost: the runtime would not report a duration
    // for it at all, and it reported one that is not a duration. Both are
    // discards and both are reported as one number.
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

  // Every figure above is drawn only from samples that passed validation, so a
  // run that dropped some has to say which and how many: a median over 40 of
  // 100 samples is a different claim from a median over all 100.
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
  // `samples` is what the statistics were computed from and `discarded` is what
  // was thrown away, so a consumer can tell a clean run from a salvaged one
  // without going back to the log.
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
