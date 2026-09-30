# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Gate C: the dsv4_moe_expert kernel compiled to device object code and run on
# the CANN camodel, so the simulator can record a cycle-level trace.
#
# This is the only tier that exercises the kernel's pipe synchronisation. The
# CPU interpreter used for Gate B executes the kernel serially on the host and
# treats SetFlag/WaitFlag as no-ops, so it can neither prove nor disprove the
# MTE2/MTE3 discipline; the camodel models the pipes.
#
# Configure with:
#   cmake -S . -B build -G "Unix Makefiles" \
#         -DSOC_VERSION=Ascend950PR_9599 -DRUN_MODE=sim
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(dsv4_moe_expert_gate_c CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The simulator ships no Ascend950PR_9579 directory; CANN itself folds every
# C310 part onto the 9599 series (see tikicpulib-config.cmake product_map()),
# so 9599 is the documented stand-in and tick numbers are labelled accordingly.
set(SOC_VERSION "Ascend950PR_9599" CACHE STRING "Simulator SoC")
set(RUN_MODE "sim" CACHE STRING "ascendc_library run mode: sim (camodel) or npu")

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set; source set_env.sh first")
endif()
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")

# ascendc_library() drives four nested ExternalProjects, each configured with
# the OUTER project's generator. Under Ninja, CMake writes out-of-tree object
# paths with a "/./" segment that CANN's extract_host_stub.py does not normalise
# when it looks them up in compile_commands.json; the build dies with a bare
# KeyError after the kernel has already compiled. Makefiles emits the normalised
# path. This is a CANN 9.1.0 toolchain bug, not something the source can avoid.
set(CMAKE_GENERATOR "Unix Makefiles")

if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found under ${ASCEND_HOME_PATH}")
endif()
message(STATUS "gate_c: ascendc cmake at ${ASCENDC_CMAKE_DIR}, SOC=${SOC_VERSION}, RUN_MODE=${RUN_MODE}")

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

set(KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../csrc/moe/dsv4_moe_expert")
if(NOT EXISTS "${KERNEL_DIR}/op_kernel/dsv4_moe_expert.cpp")
    message(FATAL_ERROR "kernel source not found at ${KERNEL_DIR}/op_kernel/dsv4_moe_expert.cpp")
endif()

# One source only. dsv4_launch.cpp includes the shipping kernel verbatim and
# defines the <<<>>> launcher beside it; listing the kernel separately as well
# would make the launcher generator emit two stubs for one kernel name and the
# merge link would fail on an undefined `dsv4_moe_expert_origin`. See the header
# comment in dsv4_launch.cpp.
ascendc_library(dsv4_gatec_kernels SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/dsv4_launch.cpp")

# No ascendc_compile_options: the tiling shim is included by dsv4_launch.cpp
# itself rather than force-included on the command line, because a forwarded
# `-include` pair does not survive the bisheng host-stub stage. See that file.

add_executable(dsv4_gate_c gate_c_main.cpp)
target_include_directories(dsv4_gate_c PRIVATE "${ASCEND_HOME_PATH}/include")

set(ASCEND_LIB_DIRS
    "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/runtime/lib64"
    "${ASCEND_HOME_PATH}/acllib/lib64")
find_library(ASCEND_CL_LIBRARY NAMES ascendcl PATHS ${ASCEND_LIB_DIRS} NO_DEFAULT_PATH)
if(NOT ASCEND_CL_LIBRARY)
    message(FATAL_ERROR "libascendcl.so not found under ${ASCEND_LIB_DIRS}")
endif()

target_link_libraries(dsv4_gate_c PRIVATE dsv4_gatec_kernels ${ASCEND_CL_LIBRARY})

# The camodel replaces libruntime.so, so its directory must precede the
# toolkit's on the loader path at run time as well as at link time.
set(SIMULATOR_LIB_DIR "${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib")
if(NOT EXISTS "${SIMULATOR_LIB_DIR}/libruntime_camodel.so")
    message(FATAL_ERROR
        "Gate C needs the camodel for ${SOC_VERSION}, and "
        "${SIMULATOR_LIB_DIR}/libruntime_camodel.so does not exist.\n"
        "Check SOC_VERSION against the directory names under ${ASCEND_HOME_PATH}/tools/simulator.")
endif()

# ascendc_library puts the simulator directory on the link line, so the linker
# pulls in libruntime_camodel.so -- but its own DT_NEEDED (libnpu_drv_camodel.so)
# and the register symbols libnpu_drv.so takes from libstars.so are resolved
# only at load time, out of that same directory. -rpath-link lets the linker
# find the transitive libraries without linking them, and
# --allow-shlib-undefined stops it from rejecting the cross-library symbols the
# loader will satisfy. Both are properties of how the camodel is packaged, not
# of this kernel.
target_link_options(dsv4_gate_c PRIVATE
    "-Wl,-rpath-link,${SIMULATOR_LIB_DIR}"
    "-Wl,--allow-shlib-undefined")

set_target_properties(dsv4_gate_c PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN;${CMAKE_BINARY_DIR}/lib;${SIMULATOR_LIB_DIR};${ASCEND_LIB_DIRS}")
