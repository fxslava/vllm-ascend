# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone CPU-interpreter harness for the dsv4_moe_expert kernel (Gate B).
#
# This is deliberately NOT part of csrc/tests: that tree's simulator tier is
# scoped to the TurboQuant kernels and pulls in the whole ascendc_library build.
# Here the kernel source is compiled directly into a host binary with
# ASCENDC_CPU_DEBUG=1 and linked against tikicpulib, which takes seconds rather
# than minutes and needs no op package, no driver and no NPU.
#
# Configure with:
#   cmake -S . -B build \
#     -DCMAKE_PREFIX_PATH=$ASCEND_HOME_PATH/tools/tikicpulib/lib64/cmake \
#     -DSOC_VERSION=Ascend950PR_9579
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(dsv4_moe_expert_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cmake's product_map() folds every C310 part onto the ascend950pr_9599 series,
# so Ascend950PR_9579 -- which has no simulator directory of its own -- resolves
# to a target that does exist.
set(SOC_VERSION "Ascend950PR_9579" CACHE STRING "tikicpulib product type")

find_package(tikicpulib REQUIRED)

set(KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../csrc/moe/dsv4_moe_expert")

add_executable(dsv4_moe_expert_sim
    sim_main.cpp
    "${KERNEL_DIR}/op_kernel/dsv4_moe_expert.cpp"
)

target_include_directories(dsv4_moe_expert_sim PRIVATE
    "${KERNEL_DIR}/op_kernel"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

# GET_TILING_DATA_WITH_STRUCT is synthesised per-op by the TBE pipeline and does
# not exist outside `build.sh --pkg`. Force-include the shim so the kernel source
# compiles here unmodified -- the shipping op package never sees this flag.
target_compile_options(dsv4_moe_expert_sim PRIVATE
    -include "${CMAKE_CURRENT_SOURCE_DIR}/cpu_tiling_shim.h"
)

# The kernel ships with -Werror in its op_host/CMakeLists.txt; hold the harness
# build to the same bar so a warning cannot creep in through this path.
target_compile_options(dsv4_moe_expert_sim PRIVATE -Wall -Wextra -Werror -Wno-unused-parameter)

target_link_libraries(dsv4_moe_expert_sim PRIVATE tikicpulib::${SOC_VERSION})
