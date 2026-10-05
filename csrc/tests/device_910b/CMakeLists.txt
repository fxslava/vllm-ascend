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
endif()

# A 910B comparative benchmark can use common/dsv4_benchmark_suite.cpp after
# GroupedMatmulV5 BF16->FP32 tuples are validated on that hardware. Do not
# publish a target or silently change the baseline's numerical contract yet.
