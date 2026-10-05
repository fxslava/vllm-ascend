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
    const auto& soc = env.soc_name();
    if (!env.available() ||
        !(soc == "Ascend910B1" || soc == "Ascend910B2" || soc == "Ascend910B3" || soc == "Ascend910B4")) {
      std::printf("No physical Ascend910B: %s (SoC=%s)\n", env.unavailable_reason().c_str(), soc.c_str());
      status = 77;
    } else {
      if (env.device().device_id() != 0) throw std::runtime_error("benchmark requires npu:0");
      dsv4_benchmark::Run();
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "910B benchmark failed operator planning, parity, or execution: %s\n", e.what());
    status = 1;
  }
  env.TearDown();
  return status;
}
