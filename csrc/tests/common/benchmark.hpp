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

// Microbenchmark harness for the Ascend 310P kernels the correctness suite
// covers. Shares the device, tensor and operator plumbing with the tests; the
// only thing that differs is what happens around the launch.
//
// What the timed region contains, and what it deliberately does not:
//
//   * No allocation. Every device buffer, every aclTensor descriptor and the
//     operator workspace are created in the setup phase. DeviceBuffer is asked
//     for kBenchmarkAlignBytes (512) rather than the 32-byte test default so a
//     measurement is never charged for a buffer that starts mid-line.
//   * No host synchronisation, except in kHostWallClock, which exists precisely
//     to measure host-visible single-iteration latency.
//   * No operator planning, when the CANN build exports
//     aclSetAclOpExecutorRepeatable. See PlannedOp: the plan is done once in
//     setup and the timed loop calls only the launch entry point. When the
//     symbol is absent the fallback re-plans inside the loop, which is what
//     torch_npu does per call, and every result is labelled with which path ran
//     so the two are never compared as if they measured the same thing.
//
// Three timing modes, all reported:
//
//   pipelined  - one event pair around a batch of `pipeline_batch` launches,
//                divided by the batch size. Nothing serialises the launches, so
//                this is the throughput number and the one TFLOP/s and GB/s are
//                derived from.
//   device     - one event pair around each launch, all recorded back to back
//                with a single stream synchronisation afterwards. Pure device
//                time per iteration, including the in-stream event overhead.
//   host       - std::chrono around launch + aclrtSynchronizeStream. The
//                host-visible latency of one call, which is what a decode step
//                actually pays.
//
// A run reports min / median / mean / P95 / P99 over `timed_iterations`
// samples. Percentiles are nearest-rank, so at the default of 100 iterations
// P99 is the second-largest sample and is dominated by whatever the OS did that
// millisecond; read it as a tail indicator, not as a number to tune against.
//
// Every figure is a double from end to end. The device modes get theirs from
// aclrtEventElapsedTime, which reports milliseconds as a float and has three
// ways of handing back something that is not a duration at all: an event the
// device has not finished with, an event created without a timestamp, and a
// wrapped device counter. Any of those produce a negative or non-finite value,
// and one of them in a sample set drags the mean under zero and turns the
// derived TFLOP/s and GB/s into nonsense. So a sample is validated before it is
// aggregated, a rejected one is counted and reported rather than silently
// replaced, and a mode whose samples are all rejected fails its case instead of
// printing zeros that look like measurements.

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

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

enum class TimingMode {
  kPipelined,
  kDeviceEvents,
  kHostWallClock,
};

const char* TimingModeLabel(TimingMode mode);

struct BenchmarkOptions {
  // Enough to have the runtime compile and cache the kernel, settle the AI Core
  // clock and touch every page of the workspace at least once.
  int warmup_iterations = 20;
  int timed_iterations = 100;
  // Launches per event pair in kPipelined. Large enough that the event pair
  // costs a small fraction of the sample, small enough that a sample is still
  // one point in a distribution rather than the whole run.
  int pipeline_batch = 10;
  std::vector<TimingMode> modes;
  // Written when ASCEND_BENCH_CSV names a path. One row per (case, mode).
  std::string csv_path;
  // ASCEND_BENCH_REPEATABLE=0 forces the re-plan-per-iteration path even when
  // the runtime supports repeatable executors, so the two can be compared.
  bool allow_repeatable_executor = true;

  // warmup / iterations / batch / csv / repeatable from ASCEND_BENCH_WARMUP,
  // ASCEND_BENCH_ITERS, ASCEND_BENCH_BATCH, ASCEND_BENCH_CSV,
  // ASCEND_BENCH_REPEATABLE; modes from ASCEND_BENCH_MODES as a comma-separated
  // subset of "pipelined,device,host".
  static BenchmarkOptions FromEnvironment();
};

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

// One mode's raw samples, plus the number the timing loop could not obtain a
// usable duration for at all. Those never reach the vector, so they have to be
// carried alongside it to be reported.
struct LatencySamples {
  std::vector<double> microseconds;
  size_t rejected = 0;
};

struct LatencyStatistics {
  // Samples the statistics below were computed from: the ones that survived
  // validation, not the ones that were requested.
  size_t sample_count = 0;
  // Samples thrown away because they were not a usable duration - non-finite,
  // negative, or zero - plus the ones the runtime refused to report at all.
  // Non-zero means the numbers next to it are drawn from a short run, and a
  // run with a lot of these is measuring the driver rather than the kernel.
  size_t discarded_count = 0;
  double min_us = 0.0;
  double median_us = 0.0;
  double mean_us = 0.0;
  double p95_us = 0.0;
  double p99_us = 0.0;
  double max_us = 0.0;
  double stddev_us = 0.0;

  // Takes the samples by value, drops the ones that are not a usable duration
  // into discarded_count, and sorts the rest in place. Everything is
  // accumulated in double; there is no narrowing anywhere on this path.
  //
  // sample_count == 0 on return means nothing usable was measured. Callers must
  // treat that as a failure rather than as a row of zeros: see BenchmarkRunner.
  static LatencyStatistics From(std::vector<double> samples_us);
};

// How the device-event timings were obtained, for the report header. An event
// without a timestamp cannot be handed to aclrtEventElapsedTime and is the
// first thing to suspect when the device modes disagree with the host mode.
const char* EventTimingSourceLabel();

// ---------------------------------------------------------------------------
// Plan-once operator launch
// ---------------------------------------------------------------------------

// An aclnn operator whose GetWorkspaceSize call and workspace allocation have
// already happened, so Launch() is as close to just the kernel as the runtime
// allows.
//
// aclnn is a two-phase API: GetWorkspaceSize builds an aclOpExecutor and the
// launch entry point consumes it. aclSetAclOpExecutorRepeatable, where the CANN
// build exports it, opts out of that consumption so one executor can be
// launched repeatedly; the executor then has to be destroyed by hand with
// aclDestroyAclOpExecutor. Both symbols are resolved with dlsym for the same
// reason the operators themselves are (see aclnn_runtime.hpp), and their
// absence degrades to re-planning rather than to a failure.
class PlannedOp {
 public:
  // Calls the operator's GetWorkspaceSize with the captured arguments.
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

  // "repeatable-executor" or "replan-per-launch". Printed with the results.
  static const char* LaunchPathLabel();

  // Set once by BenchmarkOptions::FromEnvironment; consulted at plan time.
  static void SetRepeatableExecutorAllowed(bool allowed);

 private:
  void DestroyExecutor();
  // Frees an executor that no launch consumed and that nothing else owns: the
  // one a re-plan built for a launch that then failed. A no-op when the CANN
  // build does not export aclDestroyAclOpExecutor.
  static void DestroyOrphanedExecutor(aclOpExecutor* executor);

  std::string op_name_;
  void* launch_fn_ = nullptr;
  Planner planner_;
  DeviceBuffer workspace_;
  uint64_t workspace_size_ = 0;
  aclOpExecutor* executor_ = nullptr;
  bool repeatable_ = false;
};

// Plans `op` with the given arguments, which are everything up to but not
// including the trailing workspaceSize and executor out-parameters, exactly as
// RunAclnn takes them. WorkspaceSizeFn is the hand-declared function-pointer
// type from aclnn_ops.hpp.
//
// The arguments are captured by value: they are pointers into buffers the
// caller keeps alive for the whole benchmark, plus scalars.
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

// ---------------------------------------------------------------------------
// Cases and results
// ---------------------------------------------------------------------------

struct BenchmarkCase {
  // Shape label, e.g. "m1_k4096_n11008". Appears in the table and the CSV.
  std::string name;

  // Analytic work per launch. Zero means "do not report this column": a
  // GEMV is bandwidth-bound and an elementwise kernel does no useful FLOPs, so
  // most cases fill in one of the two.
  double flops_per_iteration = 0.0;
  double bytes_per_iteration = 0.0;

  // Enqueues exactly one operator launch. Must not allocate, must not
  // synchronise, and must not touch host memory.
  std::function<void(aclrtStream)> launch;

  // Optional. Reads the output back and reduces it to one number. Called once
  // after warmup and once after the last timed iteration; a change between the
  // two, or a non-finite value, fails the case. This is what stops a benchmark
  // from happily timing an operator that stopped writing its output - the
  // classic failure of a repeatable-executor path.
  std::function<double()> checksum;

  // Relative tolerance for the two checksums. Zero, the default, demands they
  // be bit-identical, which is the right bar for an operator that recomputes
  // the same output from the same inputs every launch. An in-place operator
  // such as rotary embedding consumes its own previous output, so its invariant
  // (the sum of squares, which a rotation preserves) drifts by the fp16
  // rounding of every launch and needs room.
  double checksum_rtol = 0.0;
};

// Reductions for BenchmarkCase::checksum. Both accumulate in double and in a
// fixed order, so two calls over an unchanged buffer agree bit for bit.
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

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

class BenchmarkRunner {
 public:
  BenchmarkRunner(std::string suite_name, const BenchmarkOptions& options, aclrtStream stream);

  // The dedicated stream every case is enqueued on. Separate from the device's
  // default stream so nothing the setup phase did is still in flight.
  aclrtStream stream() const { return stream_; }
  const BenchmarkOptions& options() const { return options_; }

  // Warms up, times every enabled mode, and records the results. Throws
  // AclError on a launch failure; the caller decides whether one bad case ends
  // the suite.
  void Run(const BenchmarkCase& benchmark_case);

  // Records a case that could not run, with the reason, so the report shows a
  // hole rather than silently omitting a shape.
  void Skip(const std::string& case_name, const std::string& reason);

  // Prints the table, and writes the CSV when one was configured.
  void Report() const;

  bool has_results() const { return !results_.empty(); }
  size_t failure_count() const { return failures_.size(); }

  // Records a case that failed to run. The suite keeps going and the report
  // lists it, so one unsupported shape does not cost the whole run.
  void RecordFailure(const std::string& case_name, const std::string& reason);

 private:
  LatencySamples TimePipelined(const BenchmarkCase& benchmark_case);
  LatencySamples TimeDeviceEvents(const BenchmarkCase& benchmark_case);
  LatencySamples TimeHostWallClock(const BenchmarkCase& benchmark_case);

  // Warms the case up and puts a hard barrier between the warmup and the first
  // timed sample, so nothing the warmup queued - launches included, but MTE
  // transfers especially - is still retiring when timing starts.
  void WarmUp(const BenchmarkCase& benchmark_case);

  // Drains the stream when the `about_to_enqueue` tasks the caller is next
  // going to submit would take it past what one stream should hold, and resets
  // the count. Called only at sample boundaries, never inside a batch, so no
  // sample is ever split by a synchronisation. Taking the pending count rather
  // than only the running one is what keeps a large ASCEND_BENCH_BATCH from
  // overrunning the queue in a single sample.
  void DrainIfQueueIsDeep(int* enqueued, int about_to_enqueue);

  // Run() wraps this so that a failure drains the stream before unwinding.
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

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

// Owns the process: initialises the runtime, creates the dedicated benchmark
// stream, calls `build` to register and run the cases, then reports.
//
// Returns 0 on success, kBenchmarkSkipExitCode when no usable 310P device is
// attached (wired to ctest's SKIP_RETURN_CODE so a build host reports a skip
// rather than a failure), and 1 when a case failed.
int RunBenchmarkSuite(const char* suite_name, const std::function<void(BenchmarkRunner&)>& build);

// ctest SKIP_RETURN_CODE; see csrc/tests/CMakeLists.txt.
constexpr int kBenchmarkSkipExitCode = 77;

}  // namespace bench
}  // namespace test
}  // namespace vllm_ascend
