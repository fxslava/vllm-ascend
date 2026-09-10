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

// Regression tests for the parts of the benchmark harness that decide what a
// timing report says, none of which need a device.
//
// This is the only test binary in the suite with no REQUIRE_ASCEND_DEVICE in
// it, and that is the point: the bugs it covers - a negative event time
// reaching the statistics, a malformed ASCEND_BENCH_* value silently becoming a
// plausible one, throughput computed from a latency that cannot be one - all
// showed up as wrong numbers in a report from a healthy 310P, so they have to
// be catchable without one.
//
// The device-side halves of the same fixes are not reachable from here:
// aclrtEventElapsedTime validation needs recorded events, and DeviceBuffer's
// alignment slack needs aclrtMalloc. What is covered below is the arithmetic
// those two paths hand their results to, plus the pure alignment arithmetic the
// allocator is built on.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "benchmark.hpp"
#include "device_buffer.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {
namespace {

// ---------------------------------------------------------------------------
// LatencyStatistics: nothing that is not a duration reaches a reported figure
// ---------------------------------------------------------------------------

TEST(LatencyStatistics, DropsNonPositiveSamples) {
  // -5 is what aclrtEventElapsedTime returns for an event pair the device never
  // resolved, and 0 is what a host clock too coarse for the kernel returns.
  // Neither is a duration, and before the fix both were averaged in: the mean
  // of this set was 11 us and its minimum was -5 us.
  const LatencyStatistics statistics = LatencyStatistics::From({10.0, -5.0, 20.0, 0.0, 30.0});

  EXPECT_EQ(statistics.sample_count, 3u);
  EXPECT_EQ(statistics.discarded_count, 2u);
  EXPECT_DOUBLE_EQ(statistics.min_us, 10.0);
  EXPECT_DOUBLE_EQ(statistics.median_us, 20.0);
  EXPECT_DOUBLE_EQ(statistics.mean_us, 20.0);
  EXPECT_DOUBLE_EQ(statistics.max_us, 30.0);
}

TEST(LatencyStatistics, DropsNonFiniteSamples) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const LatencyStatistics statistics = LatencyStatistics::From({4.0, nan, 8.0, inf, -inf});

  EXPECT_EQ(statistics.sample_count, 2u);
  EXPECT_EQ(statistics.discarded_count, 3u);
  EXPECT_DOUBLE_EQ(statistics.mean_us, 6.0);
  EXPECT_TRUE(std::isfinite(statistics.stddev_us));
}

TEST(LatencyStatistics, ReportsNoSamplesWhenNothingIsUsable) {
  // BenchmarkRunner keys the "this mode measured nothing" failure off
  // sample_count, so a set with nothing usable in it has to come back empty
  // rather than as a row of zeros that reads like a fast kernel.
  const LatencyStatistics statistics = LatencyStatistics::From({-1.0, -2.0, 0.0});

  EXPECT_EQ(statistics.sample_count, 0u);
  EXPECT_EQ(statistics.discarded_count, 3u);
  EXPECT_DOUBLE_EQ(statistics.min_us, 0.0);
  EXPECT_DOUBLE_EQ(statistics.median_us, 0.0);
  EXPECT_DOUBLE_EQ(statistics.mean_us, 0.0);
  EXPECT_DOUBLE_EQ(statistics.p95_us, 0.0);
  EXPECT_DOUBLE_EQ(statistics.p99_us, 0.0);
}

TEST(LatencyStatistics, EmptyInputIsNotADiscard) {
  const LatencyStatistics statistics = LatencyStatistics::From({});

  EXPECT_EQ(statistics.sample_count, 0u);
  EXPECT_EQ(statistics.discarded_count, 0u);
}

TEST(LatencyStatistics, PercentilesAreNearestRankOverTheSurvivingSamples) {
  std::vector<double> samples;
  samples.reserve(100);
  for (int i = 1; i <= 100; ++i) {
    samples.push_back(static_cast<double>(i));
  }
  const LatencyStatistics statistics = LatencyStatistics::From(samples);

  EXPECT_EQ(statistics.sample_count, 100u);
  EXPECT_EQ(statistics.discarded_count, 0u);
  // Even count, so the median is the mean of the two middle samples.
  EXPECT_DOUBLE_EQ(statistics.median_us, 50.5);
  EXPECT_DOUBLE_EQ(statistics.p95_us, 95.0);
  EXPECT_DOUBLE_EQ(statistics.p99_us, 99.0);
}

TEST(LatencyStatistics, PercentilesRankAgainstTheSurvivorsNotTheRequest) {
  // Ranks are taken over what is left, so a run that lost half its samples
  // still reports an observed sample for P95 rather than indexing off the end.
  std::vector<double> samples;
  for (int i = 1; i <= 10; ++i) {
    samples.push_back(static_cast<double>(i));
    samples.push_back(-1.0);
  }
  const LatencyStatistics statistics = LatencyStatistics::From(samples);

  EXPECT_EQ(statistics.sample_count, 10u);
  EXPECT_EQ(statistics.discarded_count, 10u);
  EXPECT_DOUBLE_EQ(statistics.p95_us, 10.0);
  EXPECT_DOUBLE_EQ(statistics.p99_us, 10.0);
}

// ---------------------------------------------------------------------------
// Derived throughput
// ---------------------------------------------------------------------------

TEST(BenchmarkResult, ThroughputIsComputedFromTheMedian) {
  BenchmarkResult result;
  result.flops_per_iteration = 2.0e9;
  result.bytes_per_iteration = 1.0e6;
  result.latency = LatencyStatistics::From({100.0});

  // 2e9 FLOP in 100 us is 2e13 FLOP/s, i.e. 20 TFLOP/s.
  EXPECT_DOUBLE_EQ(result.tflops(), 20.0);
  // 1e6 bytes in 100 us is 1e10 byte/s, i.e. 10 GB/s.
  EXPECT_DOUBLE_EQ(result.gigabytes_per_second(), 10.0);
}

TEST(BenchmarkResult, ThroughputIsZeroWhenNothingWasMeasured) {
  // The median of an all-discarded set is 0, and dividing by it would give an
  // infinite TFLOP/s. Reporting zero is what makes the empty row obvious.
  BenchmarkResult result;
  result.flops_per_iteration = 2.0e9;
  result.bytes_per_iteration = 1.0e6;
  result.latency = LatencyStatistics::From({-1.0, -2.0});

  EXPECT_EQ(result.latency.sample_count, 0u);
  EXPECT_DOUBLE_EQ(result.tflops(), 0.0);
  EXPECT_DOUBLE_EQ(result.gigabytes_per_second(), 0.0);
}

// ---------------------------------------------------------------------------
// BenchmarkOptions::FromEnvironment
// ---------------------------------------------------------------------------

class BenchmarkEnvironment : public ::testing::Test {
 protected:
  void SetUp() override { ClearAll(); }
  void TearDown() override { ClearAll(); }

  static void Set(const char* name, const char* value) { ::setenv(name, value, 1); }

 private:
  static void ClearAll() {
    for (const char* name : {"ASCEND_BENCH_WARMUP", "ASCEND_BENCH_ITERS", "ASCEND_BENCH_BATCH",
                             "ASCEND_BENCH_MODES", "ASCEND_BENCH_CSV", "ASCEND_BENCH_REPEATABLE"}) {
      ::unsetenv(name);
    }
  }
};

TEST_F(BenchmarkEnvironment, DefaultsApplyWhenNothingIsSet) {
  const BenchmarkOptions options = BenchmarkOptions::FromEnvironment();

  EXPECT_EQ(options.warmup_iterations, 20);
  EXPECT_EQ(options.timed_iterations, 100);
  EXPECT_EQ(options.pipeline_batch, 10);
  EXPECT_EQ(options.modes.size(), 3u);
  EXPECT_TRUE(options.allow_repeatable_executor);
}

TEST_F(BenchmarkEnvironment, WellFormedValuesAreTaken) {
  Set("ASCEND_BENCH_WARMUP", "5");
  Set("ASCEND_BENCH_ITERS", "12");
  Set("ASCEND_BENCH_BATCH", "4");
  const BenchmarkOptions options = BenchmarkOptions::FromEnvironment();

  EXPECT_EQ(options.warmup_iterations, 5);
  EXPECT_EQ(options.timed_iterations, 12);
  EXPECT_EQ(options.pipeline_batch, 4);
}

TEST_F(BenchmarkEnvironment, NonNumericValuesFallBackInsteadOfBecomingZero) {
  // atoi returned 0 for all three of these. Zero warmup is a valid setting, so
  // the typo did not announce itself: it just moved the kernel compile into
  // sample 0 of every case.
  Set("ASCEND_BENCH_WARMUP", "twenty");
  Set("ASCEND_BENCH_ITERS", "100x");
  Set("ASCEND_BENCH_BATCH", "");
  const BenchmarkOptions options = BenchmarkOptions::FromEnvironment();

  EXPECT_EQ(options.warmup_iterations, 20);
  EXPECT_EQ(options.timed_iterations, 100);
  EXPECT_EQ(options.pipeline_batch, 10);
}

TEST_F(BenchmarkEnvironment, OutOfRangeValuesFallBack) {
  // Below the minimum, and past what a stream can hold in one sample or an int
  // can hold at all.
  Set("ASCEND_BENCH_ITERS", "0");
  Set("ASCEND_BENCH_BATCH", "100000");
  Set("ASCEND_BENCH_WARMUP", "99999999999999");
  const BenchmarkOptions options = BenchmarkOptions::FromEnvironment();

  EXPECT_EQ(options.timed_iterations, 100);
  EXPECT_EQ(options.pipeline_batch, 10);
  EXPECT_EQ(options.warmup_iterations, 20);
}

TEST_F(BenchmarkEnvironment, ModesAreParsedAndTrimmed) {
  Set("ASCEND_BENCH_MODES", "device, host");
  const BenchmarkOptions options = BenchmarkOptions::FromEnvironment();

  ASSERT_EQ(options.modes.size(), 2u);
  EXPECT_STREQ(TimingModeLabel(options.modes[0]), "device");
  EXPECT_STREQ(TimingModeLabel(options.modes[1]), "host");
}

TEST_F(BenchmarkEnvironment, AnAllUnknownModeListFallsBackToPipelined) {
  Set("ASCEND_BENCH_MODES", "wallclock,cuda");
  const BenchmarkOptions options = BenchmarkOptions::FromEnvironment();

  ASSERT_EQ(options.modes.size(), 1u);
  EXPECT_STREQ(TimingModeLabel(options.modes[0]), "pipelined");
}

// ---------------------------------------------------------------------------
// Allocation arithmetic
// ---------------------------------------------------------------------------

TEST(DeviceBufferAlignment, PaddedCapacityFitsInsideTheRequestForAnyBaseAddress) {
  // The invariant DeviceBuffer::Allocate now checks at runtime, stated over
  // every base-address residue rather than the one aclrtMalloc happens to
  // return. The slack used to be added only for a stronger-than-default
  // alignment, which made this false for the 32-byte case whenever the
  // allocator handed back a pointer that needed advancing.
  for (size_t alignment : {kDeviceAlignBytes, kBenchmarkAlignBytes}) {
    for (size_t size_bytes : {size_t{1}, size_t{31}, size_t{512}, size_t{1536}, size_t{11008 * 2}}) {
      const size_t capacity = AlignUp(size_bytes, alignment);
      const size_t request = capacity + alignment;
      for (size_t base = 0; base < alignment; ++base) {
        const size_t offset = AlignUp(base, alignment) - base;
        EXPECT_LE(offset + capacity, request)
            << "alignment=" << alignment << " size=" << size_bytes << " base residue=" << base;
      }
    }
  }
}

TEST(DeviceBufferAlignment, AlignUpNeverShrinksAndLandsOnTheBoundary) {
  for (size_t alignment : {kDeviceAlignBytes, kBenchmarkAlignBytes}) {
    for (size_t value = 0; value < 4 * alignment; ++value) {
      const size_t aligned = AlignUp(value, alignment);
      EXPECT_GE(aligned, value);
      EXPECT_EQ(aligned % alignment, 0u);
      EXPECT_LT(aligned - value, alignment);
    }
  }
}

// ---------------------------------------------------------------------------
// Checksum reductions
// ---------------------------------------------------------------------------

TEST(Checksum, AccumulateInDoubleAndAreOrderStable) {
  // The guard that catches an operator which stopped writing its output is only
  // as good as its reproducibility: two calls over an unchanged buffer have to
  // agree bit for bit, which is why both reductions accumulate in double in a
  // fixed order rather than in the element type.
  std::vector<float> values;
  for (int i = 0; i < 1000; ++i) {
    values.push_back(static_cast<float>(i % 7) - 3.0f);
  }

  EXPECT_DOUBLE_EQ(ChecksumSum(values), ChecksumSum(values));
  EXPECT_DOUBLE_EQ(ChecksumSumOfSquares(values), ChecksumSumOfSquares(values));
  EXPECT_GE(ChecksumSumOfSquares(values), 0.0);
}

}  // namespace
}  // namespace bench
}  // namespace test
}  // namespace vllm_ascend
