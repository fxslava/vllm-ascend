# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
cmake_minimum_required(VERSION 3.16)
project(dsv4_910b_offline LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(SOC_VERSION Ascend910B1 CACHE STRING "910B SoC variant")
set(RUN_MODE sim CACHE STRING "sim or npu")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/8.0.0/aarch64-linux"
    CACHE PATH "Target CANN 8.0.0 aarch64 toolkit")
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    message(FATAL_ERROR "This offline project requires an aarch64 host cross compiler")
endif()
if(NOT SOC_VERSION MATCHES "^Ascend910B[1-4]$")
    message(FATAL_ERROR "Use a concrete 910B SoC variant, e.g. Ascend910B1")
endif()
include("${ASCEND_CANN_PACKAGE_PATH}/tikcpp/ascendc_kernel_cmake/ascendc.cmake")

ascendc_library(dsv4_expert SHARED ../common/dsv4_moe_expert_kernels.cpp)
ascendc_library(dsv4_indices SHARED index_kernel.cpp)
# CANN_VERSION_MAJOR has to reach both compilations ascendc_library does, not
# just the device precompile: the host stub parses the same headers and picks
# ReduceRepeat or WholeReduceSum on it. ascendc_compile_definitions feeds the
# device side only, so the host side needs the forwarded -D as well -- and both
# must come from ONE ascendc_compile_options call, because a second call
# overwrites rather than appends the host half. See the note in
# ../ascendc/CMakeLists.txt.
foreach(kernel dsv4_expert dsv4_indices)
    ascendc_compile_options(${kernel} PRIVATE --cce-auto-sync=off -Werror
        -forward-options-to-host-compiler -DCANN_VERSION_MAJOR=8)
    ascendc_compile_definitions(${kernel} PRIVATE CANN_VERSION_MAJOR=8)
endforeach()

add_executable(test_dsv4_expert expert_main.cpp)
target_link_libraries(test_dsv4_expert PRIVATE dsv4_expert)
add_executable(test_dsv4_indices index_main.cpp)
target_link_libraries(test_dsv4_indices PRIVATE dsv4_indices)
foreach(test test_dsv4_expert test_dsv4_indices)
    target_include_directories(${test} PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include" ../common)
    target_compile_options(${test} PRIVATE -Wall -Wextra -Werror)
    target_link_directories(${test} PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
    target_link_options(${test} PRIVATE
        "-Wl,-rpath-link,${ASCEND_CANN_PACKAGE_PATH}/lib64"
        "-Wl,-rpath-link,${ASCEND_CANN_PACKAGE_PATH}/devlib/linux/aarch64")
    if(RUN_MODE STREQUAL "sim")
        target_link_directories(${test} PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/simulator/${SOC_VERSION}/lib")
        target_link_options(${test} PRIVATE "-Wl,-rpath-link,${ASCEND_CANN_PACKAGE_PATH}/simulator/${SOC_VERSION}/lib")
        target_link_libraries(${test} PRIVATE runtime_camodel npu_drv ascendcl)
    elseif(RUN_MODE STREQUAL "npu")
        target_link_libraries(${test} PRIVATE runtime ascendcl)
    else()
        message(FATAL_ERROR "RUN_MODE must be sim or npu")
    endif()
endforeach()

enable_testing()
add_test(NAME dsv4_indices COMMAND test_dsv4_indices)
add_test(NAME dsv4_expert_micro COMMAND test_dsv4_expert)
set_tests_properties(dsv4_indices dsv4_expert_micro PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)
