# -----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
#
# Standalone GTest suite for the kernels a Qwen3.5 forward pass needs on an
# Ascend 310P3, driven straight through the CANN runtime.
#
# This project is deliberately NOT included from the repository root
# CMakeLists.txt: that one requires Torch 2.10.0 and pybind11, and the whole
# point here is to build without Python, PyTorch or torch_npu. Configure it on
# its own:
#
#   cmake -S csrc/tests -B build/csrc-tests -DCMAKE_BUILD_TYPE=Release
#
# -----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

project(vllm_ascend_kernel_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# The suite targets one part. The value is not passed to a kernel compiler here
# (no Ascend C sources are built), but it is checked at runtime against
# aclrtGetSocName so a run on the wrong part skips rather than reporting
# meaningless numbers.
set(SOC_VERSION "Ascend310P3" CACHE STRING "Ascend SoC the tests target")

option(VLLM_ASCEND_TESTS_FETCH_GTEST "Download googletest with FetchContent" ON)
option(VLLM_ASCEND_TESTS_WERROR "Treat compiler warnings as errors" OFF)
option(VLLM_ASCEND_TESTS_BUILD_BENCHMARKS "Build the bench_* microbenchmark binaries" ON)

# -----------------------------------------------------------------------------
# CANN toolkit
# -----------------------------------------------------------------------------

if(NOT DEFINED ASCEND_HOME_PATH OR ASCEND_HOME_PATH STREQUAL "")
  if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
  elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_HOME_PATH $ENV{ASCEND_TOOLKIT_HOME})
  else()
    set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
endif()
set(ASCEND_HOME_PATH ${ASCEND_HOME_PATH} CACHE PATH "CANN toolkit install root" FORCE)

if(NOT EXISTS "${ASCEND_HOME_PATH}/include/acl/acl.h")
  message(FATAL_ERROR
    "CANN not found at ASCEND_HOME_PATH=${ASCEND_HOME_PATH}\n"
    "Source the toolkit environment first:\n"
    "  source /usr/local/Ascend/ascend-toolkit/set_env.sh\n"
    "or configure with -DASCEND_HOME_PATH=/path/to/ascend-toolkit/latest")
endif()

message(STATUS "vllm-ascend tests: CANN at ${ASCEND_HOME_PATH}")
message(STATUS "vllm-ascend tests: target SoC ${SOC_VERSION}")

find_package(Threads REQUIRED)

set(ASCEND_LIB_DIRS
  "${ASCEND_HOME_PATH}/lib64"
  "${ASCEND_HOME_PATH}/runtime/lib64"
  "${ASCEND_HOME_PATH}/acllib/lib64")

# ascendcl is the only hard link-time dependency: aclInit, aclrtMalloc, the
# stream API. Everything in the aclnn op-api is resolved with dlopen/dlsym at
# runtime (see common/aclnn_runtime.cpp), which keeps a CANN release that
# renamed an operator from breaking the whole binary.
find_library(ASCEND_CL_LIBRARY NAMES ascendcl PATHS ${ASCEND_LIB_DIRS} NO_DEFAULT_PATH)
if(NOT ASCEND_CL_LIBRARY)
  message(FATAL_ERROR "libascendcl.so not found under ${ASCEND_LIB_DIRS}")
endif()

# Probed but NOT linked. These are still looked up so that a broken or partial
# CANN install is reported at configure time rather than from inside a test, but
# nothing here calls into them directly: libopapi.so (and the constructors it
# re-exports from libnnopbase.so) is opened with dlopen/dlsym at runtime by
# common/aclnn_runtime.cpp, and libacl_op_compiler.so is only needed by the
# graph-compile path this suite does not use.
#
# Keeping them off the link line means the binaries carry one CANN DT_NEEDED
# entry instead of four, so a CANN release that drops or renames one of these
# libraries no longer stops the suite from loading at all.
find_library(ASCEND_OP_COMPILER_LIBRARY NAMES acl_op_compiler PATHS ${ASCEND_LIB_DIRS} NO_DEFAULT_PATH)
find_library(ASCEND_NNOPBASE_LIBRARY NAMES nnopbase PATHS ${ASCEND_LIB_DIRS} NO_DEFAULT_PATH)
find_library(ASCEND_OPAPI_LIBRARY NAMES opapi PATHS ${ASCEND_LIB_DIRS} NO_DEFAULT_PATH)

# ascendcl is the one hard link-time dependency; see the comment above it.
set(VLLM_ASCEND_TEST_CANN_LIBS ${ASCEND_CL_LIBRARY})
foreach(runtime_lib
        ASCEND_OP_COMPILER_LIBRARY
        ASCEND_NNOPBASE_LIBRARY
        ASCEND_OPAPI_LIBRARY)
  if(${runtime_lib})
    message(STATUS "vllm-ascend tests: found ${${runtime_lib}} (resolved at runtime, not linked)")
  else()
    message(STATUS "vllm-ascend tests: ${runtime_lib} not found, resolving those symbols at runtime")
  endif()
endforeach()

# The repository root CMakeLists.txt skips vllm_ascend_kernels entirely on
# ascend310p ("Hardware ${SOC_VERSION} detected: skip vllm_ascend_kernels
# compile"), and the four kernels under test come from the CANN op-api rather
# than from csrc. The lookup stays so a 910-series build, or a future 310P
# kernel library, is picked up without editing this file.
find_library(VLLM_ASCEND_KERNELS_LIBRARY
  NAMES vllm_ascend_kernels
  HINTS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../build"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../out"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../vllm_ascend"
  NO_DEFAULT_PATH)
if(VLLM_ASCEND_KERNELS_LIBRARY)
  list(APPEND VLLM_ASCEND_TEST_CANN_LIBS ${VLLM_ASCEND_KERNELS_LIBRARY})
  message(STATUS "vllm-ascend tests: linking ${VLLM_ASCEND_KERNELS_LIBRARY}")
else()
  message(STATUS "vllm-ascend tests: vllm_ascend_kernels not found (expected on ascend310p)")
endif()

# -----------------------------------------------------------------------------
# googletest
# -----------------------------------------------------------------------------

if(VLLM_ASCEND_TESTS_FETCH_GTEST)
  include(FetchContent)
  # An offline 310P host can point FETCHCONTENT_SOURCE_DIR_GOOGLETEST at an
  # unpacked tarball, or configure with -DVLLM_ASCEND_TESTS_FETCH_GTEST=OFF to
  # use a system install instead.
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  set(VLLM_ASCEND_TEST_GTEST_TARGET gtest)
else()
  find_package(GTest REQUIRED)
  set(VLLM_ASCEND_TEST_GTEST_TARGET GTest::gtest)
endif()

# Provides gtest_discover_tests.
include(GoogleTest)

# -----------------------------------------------------------------------------
# Shared test infrastructure
# -----------------------------------------------------------------------------

add_library(vllm_ascend_test_common STATIC
  common/test_harness.cpp
  common/aclnn_runtime.cpp
  common/aclnn_ops.cpp
  common/cpu_reference.cpp
  common/benchmark.cpp)

target_include_directories(vllm_ascend_test_common
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/common"
    "${ASCEND_HOME_PATH}/include")

target_link_directories(vllm_ascend_test_common PUBLIC ${ASCEND_LIB_DIRS})

target_link_libraries(vllm_ascend_test_common
  PUBLIC
    ${VLLM_ASCEND_TEST_CANN_LIBS}
    ${VLLM_ASCEND_TEST_GTEST_TARGET}
    ${CMAKE_DL_LIBS}
    Threads::Threads)

target_compile_definitions(vllm_ascend_test_common PUBLIC ASCEND_PLATFORM_310P)

target_compile_options(vllm_ascend_test_common PUBLIC -Wall -Wextra)
if(VLLM_ASCEND_TESTS_WERROR)
  target_compile_options(vllm_ascend_test_common PUBLIC -Werror)
endif()

# Size: emit every function and variable into its own section so the linker can
# drop the ones no test reaches. The suite statically links googletest, which is
# the bulk of each binary and most of which any single test file never calls.
# --as-needed additionally drops DT_NEEDED entries for libraries that end up
# contributing no referenced symbol, which matters here because the aclnn
# operators are resolved with dlopen rather than linked.
target_compile_options(vllm_ascend_test_common PUBLIC -ffunction-sections -fdata-sections)
target_link_options(vllm_ascend_test_common PUBLIC -Wl,--gc-sections -Wl,--as-needed)

# -----------------------------------------------------------------------------
# Test binaries
# -----------------------------------------------------------------------------

enable_testing()

set(VLLM_ASCEND_KERNEL_TESTS
  test_matmul_310p
  test_rmsnorm_310p
  test_rotary_embedding_310p
  test_activation_swiglu_310p
  test_paged_attention_310p)

# Tests over the shared harness rather than over a kernel, built from common/
# for that reason. These are the only ones in the suite that pass with no device
# attached: what they cover is the arithmetic a benchmark run does around the
# launches - sample validation, the ASCEND_BENCH_* parses, the alignment rules -
# all of which produced wrong numbers on healthy hardware and so have to be
# testable without any.
set(VLLM_ASCEND_HARNESS_TESTS
  test_benchmark_harness)

# Discovery runs the binary, which needs libopapi.so and friends on the loader
# path even though they are opened lazily.
set(VLLM_ASCEND_TEST_LIBRARY_PATH "${ASCEND_HOME_PATH}/lib64:$ENV{LD_LIBRARY_PATH}")

# Every executable this project produces, collected so the strip step in
# .vscode-build-aarch64.sh can be driven from a generated manifest rather than a
# hand-maintained list. A stale name in that list is handed to `strip` as a path
# that either does not exist or is not an ELF file, which is where the "file
# format not recognized" warnings came from.
set(VLLM_ASCEND_TEST_BINARY_FILES "")

# Shared configuration for both the test and the benchmark binaries: they differ
# only in which main() they link and how ctest is told to run them.
function(vllm_ascend_add_device_executable target_name source_file entry_point)
  add_executable(${target_name} "${source_file}" "${entry_point}")
  target_link_libraries(${target_name} PRIVATE vllm_ascend_test_common)

  # $ORIGIN keeps a copied-out build tree working; the CANN lib dir covers the
  # normal in-place run.
  set_target_properties(${target_name} PROPERTIES
    BUILD_RPATH "${ASCEND_HOME_PATH}/lib64"
    INSTALL_RPATH "$ORIGIN:${ASCEND_HOME_PATH}/lib64")

  set(VLLM_ASCEND_TEST_BINARY_FILES
    ${VLLM_ASCEND_TEST_BINARY_FILES} "$<TARGET_FILE:${target_name}>" PARENT_SCOPE)
endfunction()

# Cross builds get one ctest entry per binary instead of one per GTest case.
# Even PRE_TEST discovery has to execute the binary, and an aarch64 binary on
# the x86 build host cannot run - `ctest -N` there aborts on the first discovery
# attempt and lists nothing at all, benchmarks included. Running ctest on the
# 310P itself, where the binaries are native, is unaffected.
function(vllm_ascend_register_correctness_test target_name)
  if(COMMAND gtest_discover_tests AND NOT CMAKE_VERSION VERSION_LESS "3.18" AND NOT CMAKE_CROSSCOMPILING)
    # PRE_TEST defers enumeration to `ctest` time. The default POST_BUILD mode
    # runs the binary during the build, which fails on a build host that has the
    # CANN headers but no driver.
    gtest_discover_tests(${target_name}
      DISCOVERY_MODE PRE_TEST
      DISCOVERY_TIMEOUT 120
      PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${VLLM_ASCEND_TEST_LIBRARY_PATH};ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
        LABELS "correctness"
        TIMEOUT 900)
  else()
    add_test(NAME ${target_name} COMMAND ${target_name})
    set_tests_properties(${target_name} PROPERTIES
      ENVIRONMENT "LD_LIBRARY_PATH=${VLLM_ASCEND_TEST_LIBRARY_PATH};ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
      LABELS "correctness"
      TIMEOUT 900)
  endif()
endfunction()

foreach(test_name ${VLLM_ASCEND_KERNEL_TESTS})
  vllm_ascend_add_device_executable(${test_name} "kernels/${test_name}.cpp" common/main.cpp)
  vllm_ascend_register_correctness_test(${test_name})
endforeach()

foreach(test_name ${VLLM_ASCEND_HARNESS_TESTS})
  vllm_ascend_add_device_executable(${test_name} "common/${test_name}.cpp" common/main.cpp)
  vllm_ascend_register_correctness_test(${test_name})
endforeach()

message(STATUS "vllm-ascend tests: configured ${VLLM_ASCEND_KERNEL_TESTS}")
message(STATUS "vllm-ascend tests: configured ${VLLM_ASCEND_HARNESS_TESTS}")

# -----------------------------------------------------------------------------
# Benchmark binaries
# -----------------------------------------------------------------------------
#
# Each bench_* binary links common/bench_main.cpp instead of common/main.cpp: no
# GTest registration, no test environment, just the runtime setup and the
# suite's own BuildSuite(). They are registered with ctest under the "benchmark"
# label so the default `ctest` run stays a correctness run:
#
#   ctest --test-dir build/csrc-tests -LE benchmark   # tests only
#   ctest --test-dir build/csrc-tests -L  benchmark   # benchmarks only
#
# A benchmark exits 77 when no usable 310P is attached, which SKIP_RETURN_CODE
# turns into a ctest skip rather than a failure, matching how the correctness
# suite behaves on a build host.

if(VLLM_ASCEND_TESTS_BUILD_BENCHMARKS)
  set(VLLM_ASCEND_KERNEL_BENCHMARKS
    bench_matmul_310p
    bench_rmsnorm_310p
    bench_rotary_embedding_310p
    bench_activation_swiglu_310p
    bench_paged_attention_310p)

  add_custom_target(benchmarks COMMENT "Building the vllm-ascend 310P microbenchmarks")

  foreach(bench_name ${VLLM_ASCEND_KERNEL_BENCHMARKS})
    vllm_ascend_add_device_executable(${bench_name} "kernels/${bench_name}.cpp" common/bench_main.cpp)
    add_dependencies(benchmarks ${bench_name})

    add_test(NAME ${bench_name} COMMAND ${bench_name})
    set_tests_properties(${bench_name} PROPERTIES
      ENVIRONMENT "LD_LIBRARY_PATH=${VLLM_ASCEND_TEST_LIBRARY_PATH};ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
      LABELS "benchmark"
      SKIP_RETURN_CODE 77
      # Wall clock, not device time: 1300+ launches per case across every shape
      # in a suite, and the largest matmul sweep is the slow one.
      TIMEOUT 1800)
  endforeach()

  message(STATUS "vllm-ascend tests: configured ${VLLM_ASCEND_KERNEL_BENCHMARKS}")
endif()

# -----------------------------------------------------------------------------
# Binary manifest
# -----------------------------------------------------------------------------
#
# One absolute path per line, written at generate time so the genexes above are
# resolved. Consumed by .vscode-build-aarch64.sh:
#
#   xargs -r -a build/csrc-tests-aarch64/test_binaries.txt aarch64-linux-gnu-strip --strip-unneeded

string(ASCII 10 VLLM_ASCEND_NEWLINE)
file(GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test_binaries.txt"
  CONTENT "$<JOIN:${VLLM_ASCEND_TEST_BINARY_FILES},${VLLM_ASCEND_NEWLINE}>${VLLM_ASCEND_NEWLINE}")
