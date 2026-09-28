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

// The TurboQuant audit's reporting: the banner, the two ASCII tables and the
// two CSV exporters of bench_device_950pr_turboquant.cpp. Everything here reads
// finished BenchmarkRunner results plus the sweep's traffic model
// (turboquant_audit_scenario.hpp) and changes no output byte.

#include <cstdint>
#include <string>
#include <vector>

#include "benchmark.hpp"
#include "turboquant_audit_scenario.hpp"

namespace vllm_ascend {
namespace test {
namespace turboquant_audit {

namespace bench = vllm_ascend::test::bench;

// One timed column, as the tables and CSVs print it: a median/p95 pair, the
// derived rates, and whether the value is a structural zero (a launch the
// configuration does not dispatch at all, e.g. rotate_o on a folded layer).
struct Sample {
  bool present = false;
  bool structural_zero = false;
  double median_us = 0.0;
  double p95_us = 0.0;
  double gigabytes_per_second = 0.0;
  double tflops = 0.0;
  const char* mode = "-";
};

struct PrefillRow {
  Sample ingest, rot_q, attn_core, rot_o, e2e, native, native_ingest;
  double sum_of_parts_us = 0.0;
  double speedup = 0.0;
  double gigabytes_per_second = 0.0;
};

struct DecodeRow {
  Sample rot_q, attn_core, rot_o, e2e, native;
  double sum_of_parts_us = 0.0;
  double speedup = 0.0;
  double gigabytes_per_second = 0.0;

  bool measured() const {
    return (rot_q.present && !rot_q.structural_zero) || attn_core.present || e2e.present || native.present;
  }
};

void PrintBanner(const std::vector<Config>& sweep, int64_t aiv_num, bool aiv_queried);

void PrintTableA(const std::vector<bench::BenchmarkRunner*>& runners, const std::vector<Config>& sweep,
                 const std::vector<Traffic>& traffic, const std::string& fia_operator);

void PrintTableB(const std::vector<bench::BenchmarkRunner*>& runners, const std::vector<Config>& sweep,
                 const std::vector<Traffic>& traffic, const std::string& fia_operator, int64_t aiv_num);

void WritePrefillCsv(const std::vector<bench::BenchmarkRunner*>& runners, const std::vector<Config>& sweep,
                     const std::vector<Traffic>& traffic, const std::string& path, const std::string& fia_operator);

void WriteDecodeCsv(const std::vector<bench::BenchmarkRunner*>& runners, const std::vector<Config>& sweep,
                    const std::vector<Traffic>& traffic, const std::string& path, const std::string& fia_operator);

}  // namespace turboquant_audit
}  // namespace test
}  // namespace vllm_ascend
