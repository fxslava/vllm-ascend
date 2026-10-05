// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#include <cstdio>
#include <stdexcept>

#include "dsv4_benchmark_suite.hpp"
#include "test_harness.hpp"

int main() {
  using namespace vllm_ascend::test;
  if (IsRunningOnSimulator()) {
    std::puts("Refusing simulator timings");
    return 77;
  }
  auto& env = AscendTestEnvironment::Instance();
  env.SetUp();
  int status = 0;
  try {
    if (!env.available() || !env.is_950pr()) {
      std::printf("No physical Ascend950PR: %s\n", env.unavailable_reason().c_str());
      status = 77;
    } else {
      if (env.device().device_id() != 0) throw std::runtime_error("benchmark requires npu:0");
      dsv4_benchmark::Run();
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    status = 1;
  }
  env.TearDown();
  return status;
}
