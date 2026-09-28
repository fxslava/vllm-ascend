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

// The one home for the ad-hoc environment/CSV parsing every benchmark and trace
// harness used to carry its own copy of (EnvironmentInt / EnvInt / EnvInt64 /
// EnvSelects / EnvString / EnvOr / SplitCsv / SplitOnCommas / ParseList). No
// ACL or gtest dependency, so every tier including host/ can use it.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace vllm_ascend {
namespace test {
namespace env {

// The variable's value, or "" when unset.
inline std::string String(const char* name) {
  const char* raw = std::getenv(name);
  return (raw != nullptr) ? std::string(raw) : std::string();
}

// The variable's value, or `fallback` when unset or empty.
inline std::string Or(const char* name, const char* fallback) {
  const std::string raw = String(name);
  return raw.empty() ? std::string(fallback) : raw;
}

// The variable's integer value with a floor: unparsable text yields the
// fallback, a smaller value is clamped up to `minimum`.
inline int64_t Int64Clamped(const char* name, int64_t fallback, int64_t minimum) {
  const std::string raw = String(name);
  if (raw.empty()) {
    return fallback;
  }
  const long long parsed = std::strtoll(raw.c_str(), nullptr, 10);
  return parsed < minimum ? minimum : static_cast<int64_t>(parsed);
}

inline int IntClamped(const char* name, int fallback, int minimum) {
  return static_cast<int>(Int64Clamped(name, fallback, minimum));
}

// The variable's integer value, which must parse cleanly and land in
// [minimum, maximum]. Anything else reports to stderr and yields the fallback.
// This is the ASCEND_BENCH_WARMUP / _ITERS / _BATCH contract, which
// test_device_950pr_benchmark_harness pins.
inline int Int(const char* name, int fallback, int minimum, int maximum) {
  const std::string raw = String(name);
  if (raw.empty()) {
    return fallback;
  }
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(raw.c_str(), &end, 10);
  const bool parsed_cleanly = (end != nullptr) && (end != raw.c_str()) && (*end == '\0') && (errno == 0);
  if (!parsed_cleanly || parsed < static_cast<long>(minimum) || parsed > static_cast<long>(maximum)) {
    std::fprintf(stderr, "[ascend-bench] %s=%s is not an integer in [%d, %d], using %d\n", name, raw.c_str(), minimum,
                 maximum, fallback);
    return fallback;
  }
  return static_cast<int>(parsed);
}

// The comma-separated fields of `raw`, trimmed, empty fields dropped.
inline std::vector<std::string> SplitCsv(const std::string& raw) {
  std::vector<std::string> fields;
  std::istringstream stream(raw);
  std::string field;
  while (std::getline(stream, field, ',')) {
    const size_t first = field.find_first_not_of(" \t");
    const size_t last = field.find_last_not_of(" \t");
    if (first != std::string::npos) {
      fields.push_back(field.substr(first, last - first + 1));
    }
  }
  return fields;
}

inline std::vector<std::string> SplitCsv(const char* raw) {
  return (raw != nullptr) ? SplitCsv(std::string(raw)) : std::vector<std::string>();
}

// On unless the variable holds exactly "0"; unset and empty count as on.
inline bool On(const char* name) {
  const std::string raw = String(name);
  return raw.empty() || raw != "0";
}

// Whether a CSV variable selects `value`. An unset or empty variable selects
// everything, which is what makes the ASCEND_BENCH_TQ_AUDIT_* filters optional.
inline bool Selects(const char* name, const std::string& value) {
  const std::string raw = String(name);
  if (raw.empty()) {
    return true;
  }
  for (const std::string& field : SplitCsv(raw)) {
    if (field == value) {
      return true;
    }
  }
  return false;
}

inline bool SelectsInt(const char* name, int64_t value) { return Selects(name, std::to_string(value)); }

// An int64 list from a CSV variable, falling back to `defaults` when the
// variable is unset, empty or holds nothing parsable.
inline std::vector<int64_t> IntList(const char* name, const int64_t* defaults, size_t default_count) {
  const std::string raw = String(name);
  if (raw.empty()) {
    return std::vector<int64_t>(defaults, defaults + default_count);
  }
  std::vector<int64_t> out;
  for (const std::string& token : SplitCsv(raw)) {
    out.push_back(std::strtoll(token.c_str(), nullptr, 10));
  }
  if (out.empty()) {
    return std::vector<int64_t>(defaults, defaults + default_count);
  }
  return out;
}

}  // namespace env
}  // namespace test
}  // namespace vllm_ascend
