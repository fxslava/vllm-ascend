# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
set(VLLM_ASCEND_TEST_TIER_LABEL "device;910b")

if(VLLM_ASCEND_TESTS_HAVE_DSV4)
  vllm_ascend_add_device_executable(test_device_910b_dsv4_moe_expert
    test_device_910b_dsv4_moe_expert.cpp)
  target_link_libraries(test_device_910b_dsv4_moe_expert PRIVATE vllm_ascend_test_dsv4)
  target_compile_options(test_device_910b_dsv4_moe_expert PRIVATE -Werror)
  vllm_ascend_register_correctness_test(test_device_910b_dsv4_moe_expert 3600 TRUE)
  add_custom_target(dsv4_device_targets DEPENDS test_device_910b_dsv4_moe_expert)
  if(VLLM_ASCEND_TESTS_BUILD_BENCHMARKS)
    vllm_ascend_add_device_executable(bench_device_910b_dsv4_moe_expert
      bench_device_910b_dsv4_moe_expert.cpp ../common/dsv4_benchmark_suite.cpp)
    target_link_libraries(bench_device_910b_dsv4_moe_expert PRIVATE vllm_ascend_test_dsv4)
    add_dependencies(dsv4_device_targets bench_device_910b_dsv4_moe_expert)
    add_test(NAME bench_device_910b_dsv4_moe_expert COMMAND bench_device_910b_dsv4_moe_expert)
    set_tests_properties(bench_device_910b_dsv4_moe_expert PROPERTIES
      LABELS "benchmark;device;910b" SKIP_RETURN_CODE 77 TIMEOUT 3600
      ENVIRONMENT "LD_LIBRARY_PATH=${VLLM_ASCEND_TEST_LIBRARY_PATH};ASCEND_HOME_PATH=${ASCEND_HOME_PATH}")
  endif()
endif()

# Shared benchmark requires operator planning and golden parity before timing.
