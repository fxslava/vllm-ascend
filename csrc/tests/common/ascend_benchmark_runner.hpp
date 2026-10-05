// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include "benchmark.hpp"

namespace vllm_ascend::test {
constexpr int kExpertWarmupIterations = 50;
constexpr int kExpertMeasuredIterations = 200;

// Use the existing RAII ACL event pool, duration validation and statistics.
// Prepare runs outside the timed interval; this supports one-shot ACLNN plans.
template <typename Prepare, typename Launch>
bench::BenchmarkResult MeasureExpert(aclrtStream stream, Prepare prepare, Launch launch, int64_t hidden, int64_t inter,
                                     double logical_bytes) {
  bench::BenchmarkOptions options;
  options.warmup_iterations = kExpertWarmupIterations;
  options.timed_iterations = kExpertMeasuredIterations;
  options.modes = {bench::TimingMode::kDeviceEvents};
  bench::BenchmarkRunner runner("DSV4 expert", options, stream);
  bench::BenchmarkCase sample;
  sample.name = "expert";
  sample.flops_per_iteration = 6.0 * hidden * inter;
  sample.bytes_per_iteration = logical_bytes;
  sample.prepare = prepare;
  sample.launch = [launch](aclrtStream) { launch(); };
  runner.Run(sample);
  return runner.results().front();
}
}  // namespace vllm_ascend::test
