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

TEST(LatencyStatistics, DropsNonPositiveSamples) {
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
  EXPECT_DOUBLE_EQ(statistics.median_us, 50.5);
  EXPECT_DOUBLE_EQ(statistics.p95_us, 95.0);
  EXPECT_DOUBLE_EQ(statistics.p99_us, 99.0);
}

TEST(LatencyStatistics, PercentilesRankAgainstTheSurvivorsNotTheRequest) {
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

TEST(BenchmarkResult, ThroughputIsComputedFromTheMedian) {
  BenchmarkResult result;
  result.flops_per_iteration = 2.0e9;
  result.bytes_per_iteration = 1.0e6;
  result.latency = LatencyStatistics::From({100.0});

  EXPECT_DOUBLE_EQ(result.tflops(), 20.0);
  EXPECT_DOUBLE_EQ(result.gigabytes_per_second(), 10.0);
}

TEST(BenchmarkResult, ThroughputIsZeroWhenNothingWasMeasured) {
  BenchmarkResult result;
  result.flops_per_iteration = 2.0e9;
  result.bytes_per_iteration = 1.0e6;
  result.latency = LatencyStatistics::From({-1.0, -2.0});

  EXPECT_EQ(result.latency.sample_count, 0u);
  EXPECT_DOUBLE_EQ(result.tflops(), 0.0);
  EXPECT_DOUBLE_EQ(result.gigabytes_per_second(), 0.0);
}

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
  Set("ASCEND_BENCH_WARMUP", "twenty");
  Set("ASCEND_BENCH_ITERS", "100x");
  Set("ASCEND_BENCH_BATCH", "");
  const BenchmarkOptions options = BenchmarkOptions::FromEnvironment();

  EXPECT_EQ(options.warmup_iterations, 20);
  EXPECT_EQ(options.timed_iterations, 100);
  EXPECT_EQ(options.pipeline_batch, 10);
}

TEST_F(BenchmarkEnvironment, OutOfRangeValuesFallBack) {
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

TEST(DeviceBufferAlignment, PaddedCapacityFitsInsideTheRequestForAnyBaseAddress) {
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

TEST(Checksum, AccumulateInDoubleAndAreOrderStable) {
  std::vector<float> values;
  for (int i = 0; i < 1000; ++i) {
    values.push_back(static_cast<float>(i % 7) - 3.0f);
  }

  EXPECT_DOUBLE_EQ(ChecksumSum(values), ChecksumSum(values));
  EXPECT_DOUBLE_EQ(ChecksumSumOfSquares(values), ChecksumSumOfSquares(values));
  EXPECT_GE(ChecksumSumOfSquares(values), 0.0);
}

}
}
}
}
