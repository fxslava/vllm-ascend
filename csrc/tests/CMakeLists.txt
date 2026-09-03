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
# Standalone GTest suite for the kernels a Qwen3.5 forward pass needs, with two
# interchangeable device backends.
#
# This project is deliberately NOT included from the repository root
# CMakeLists.txt: that one requires Torch 2.10.0 and pybind11, and the whole
# point here is to build without Python, PyTorch or torch_npu.
#
#   Ascend 310P, through the CANN runtime (default):
#
#     cmake -S csrc/tests -B build/csrc-tests -DCMAKE_BUILD_TYPE=Release
#
#   NVIDIA GPU, through native CUDA C kernels:
#
#     cmake -S csrc/tests -B build/csrc-tests-cuda -DENABLE_CUDA=ON \
#           -DCMAKE_BUILD_TYPE=Release
#
# The CUDA backend exists so the suite can be developed and run on a host PC
# with no Ascend hardware attached. It shares the CPU references, the fp16
# conversion, the deterministic data generator, the tolerance machinery and the
# Qwen3.5 shape tables with the Ascend path; only the device runtime and the
# kernels differ. Neither backend uses PyTorch, Cutlass or any Python binding.
# -----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

option(ENABLE_CUDA "Build bare-metal tests with CUDA C backend" OFF)

if(ENABLE_CUDA)
  # CMAKE_CUDA_ARCHITECTURES and first-class CUDA language support need 3.18.
  # The Ascend path keeps the older floor.
  cmake_minimum_required(VERSION 3.18)

  # Whether the caller chose the architectures, checked before project() because
  # enabling the CUDA language seeds CMAKE_CUDA_ARCHITECTURES in the cache with
  # the compiler's own default. After that point the variable is always set and
  # a plain set(... CACHE ...) would silently do nothing, so the default list
  # below has to know whether it is allowed to overwrite.
  if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(VLLM_ASCEND_CUDA_ARCHITECTURES_EXPLICIT TRUE)
  else()
    set(VLLM_ASCEND_CUDA_ARCHITECTURES_EXPLICIT FALSE)
  endif()

  project(vllm_ascend_kernel_tests LANGUAGES CXX CUDA)
else()
  project(vllm_ascend_kernel_tests LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

option(VLLM_ASCEND_TESTS_FETCH_GTEST "Download googletest with FetchContent" ON)
option(VLLM_ASCEND_TESTS_WERROR "Treat compiler warnings as errors" OFF)
option(VLLM_ASCEND_TESTS_BUILD_BENCHMARKS "Build the bench_* microbenchmark binaries" ON)

# -----------------------------------------------------------------------------
# Warning flags
# -----------------------------------------------------------------------------
#
# The suite is built warning-clean under -Wall -Wextra. MSVC spells that /W4
# (its own -Wall also reports every warning inside the system headers, which is
# not the same request), and nvcc forwards host flags with -Xcompiler, so the
# mapping is done once here rather than at every target.

function(vllm_ascend_tests_apply_warning_flags target scope)
  set(cuda_only_flags "")
  if(MSVC)
    set(host_flags /W4)
    # C4211 "redefined extern to static" is reported against the .stub.c file
    # nvcc generates to register kernels, not against anything in this tree, and
    # there is no source edit that removes it. Suppressed for CUDA compilations
    # only, so the same warning in hand-written code would still be reported.
    set(cuda_only_flags /wd4211)
    if(VLLM_ASCEND_TESTS_WERROR)
      list(APPEND host_flags /WX)
    endif()
  else()
    set(host_flags -Wall -Wextra)
    if(VLLM_ASCEND_TESTS_WERROR)
      list(APPEND host_flags -Werror)
    endif()
  endif()

  # One genex per flag. Putting a ;-separated list inside a single genex works
  # by accident of how CMake re-splits the expanded string, and stops working
  # the moment a flag contains a character that needs quoting.
  foreach(flag IN LISTS host_flags)
    target_compile_options(${target} ${scope} "$<$<COMPILE_LANGUAGE:CXX>:${flag}>")
  endforeach()

  # nvcc forwards host flags with -Xcompiler, which takes one comma-separated
  # argument rather than repeated options.
  set(cuda_host_flags ${host_flags} ${cuda_only_flags})
  list(JOIN cuda_host_flags "," cuda_host_flags_joined)
  target_compile_options(${target} ${scope}
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${cuda_host_flags_joined}>")
endfunction()

find_package(Threads REQUIRED)

if(NOT ENABLE_CUDA)
  # ---------------------------------------------------------------------------
  # CANN toolkit
  # ---------------------------------------------------------------------------

  # The suite targets one part. The value is not passed to a kernel compiler
  # here (no Ascend C sources are built), but it is checked at runtime against
  # aclrtGetSocName so a run on the wrong part skips rather than reporting
  # meaningless numbers.
  set(SOC_VERSION "Ascend310P3" CACHE STRING "Ascend SoC the tests target")

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
      "or configure with -DASCEND_HOME_PATH=/path/to/ascend-toolkit/latest\n"
      "To build the CUDA backend on a host with no Ascend hardware instead, "
      "configure with -DENABLE_CUDA=ON.")
  endif()

  message(STATUS "vllm-ascend tests: CANN at ${ASCEND_HOME_PATH}")
  message(STATUS "vllm-ascend tests: target SoC ${SOC_VERSION}")

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

else()
  # ---------------------------------------------------------------------------
  # CUDA toolkit
  # ---------------------------------------------------------------------------

  find_package(CUDAToolkit REQUIRED)

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  # Turing through Hopper, plus Blackwell when the toolkit knows about it
  # (sm_120 landed in CUDA 12.8). Override with -DCMAKE_CUDA_ARCHITECTURES=...
  # for a faster single-target build; the kernels use nothing newer than
  # __shfl_xor_sync and half2, so every entry compiles.
  if(NOT VLLM_ASCEND_CUDA_ARCHITECTURES_EXPLICIT)
    set(VLLM_ASCEND_CUDA_ARCHITECTURES_DEFAULT "75;80;86;89;90")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
      list(APPEND VLLM_ASCEND_CUDA_ARCHITECTURES_DEFAULT "120")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${VLLM_ASCEND_CUDA_ARCHITECTURES_DEFAULT}" CACHE STRING
        "CUDA architectures the kernels are compiled for" FORCE)
  endif()

  message(STATUS "vllm-ascend tests: CUDA backend enabled")
  message(STATUS "vllm-ascend tests: CUDA toolkit ${CUDAToolkit_VERSION} at ${CUDAToolkit_LIBRARY_ROOT}")
  message(STATUS "vllm-ascend tests: CUDA architectures ${CMAKE_CUDA_ARCHITECTURES}")

  # cudart is the whole link-time dependency. No cuBLAS, no cuDNN, no Cutlass:
  # every kernel in kernels/cuda is written out by hand.
  set(VLLM_ASCEND_TEST_CUDA_LIBS CUDA::cudart)
endif()

# -----------------------------------------------------------------------------
# googletest
# -----------------------------------------------------------------------------
#
# Shared by both backends: the same FetchContent declaration, so a build tree
# that already has _deps/googletest-src reuses it rather than downloading a
# second copy, and an offline host can point FETCHCONTENT_SOURCE_DIR_GOOGLETEST
# at an unpacked tarball or configure with -DVLLM_ASCEND_TESTS_FETCH_GTEST=OFF
# to use a system install instead.

if(VLLM_ASCEND_TESTS_FETCH_GTEST)
  include(FetchContent)

  # Reuse a googletest already unpacked by a sibling build directory before
  # reaching for the network. This is what makes an ENABLE_CUDA build next to an
  # existing Ascend build (or the reverse) cost no download at all.
  if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_GOOGLETEST)
    file(GLOB VLLM_ASCEND_EXISTING_GTEST_DIRS
      "${CMAKE_CURRENT_SOURCE_DIR}/../../build/*/_deps/googletest-src")
    foreach(candidate ${VLLM_ASCEND_EXISTING_GTEST_DIRS})
      if(EXISTS "${candidate}/CMakeLists.txt")
        set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST "${candidate}" CACHE PATH
            "Pre-unpacked googletest source tree")
        message(STATUS "vllm-ascend tests: reusing googletest from ${candidate}")
        break()
      endif()
    endforeach()
  endif()

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  # MSVC links the C runtime dynamically by default; googletest defaults to the
  # static one, and mixing the two is a link error rather than a warning.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  set(VLLM_ASCEND_TEST_GTEST_TARGET gtest)
else()
  find_package(GTest REQUIRED)
  set(VLLM_ASCEND_TEST_GTEST_TARGET GTest::gtest)
endif()

# Provides gtest_discover_tests.
include(GoogleTest)

enable_testing()

# Every executable this project produces, collected so a strip step can be
# driven from a generated manifest rather than a hand-maintained list.
set(VLLM_ASCEND_TEST_BINARY_FILES "")

if(NOT ENABLE_CUDA)
  # ===========================================================================
  # Ascend 310P backend
  # ===========================================================================

  # ---------------------------------------------------------------------------
  # Shared test infrastructure
  # ---------------------------------------------------------------------------

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

  vllm_ascend_tests_apply_warning_flags(vllm_ascend_test_common PUBLIC)

  # Size: emit every function and variable into its own section so the linker can
  # drop the ones no test reaches. The suite statically links googletest, which is
  # the bulk of each binary and most of which any single test file never calls.
  # --as-needed additionally drops DT_NEEDED entries for libraries that end up
  # contributing no referenced symbol, which matters here because the aclnn
  # operators are resolved with dlopen rather than linked.
  target_compile_options(vllm_ascend_test_common PUBLIC -ffunction-sections -fdata-sections)
  target_link_options(vllm_ascend_test_common PUBLIC -Wl,--gc-sections -Wl,--as-needed)

  # ---------------------------------------------------------------------------
  # Test binaries
  # ---------------------------------------------------------------------------

  set(VLLM_ASCEND_KERNEL_TESTS
    test_matmul_310p
    test_rmsnorm_310p
    test_rotary_embedding_310p
    test_activation_swiglu_310p
    test_paged_attention_310p)

  # Discovery runs the binary, which needs libopapi.so and friends on the loader
  # path even though they are opened lazily.
  set(VLLM_ASCEND_TEST_LIBRARY_PATH "${ASCEND_HOME_PATH}/lib64:$ENV{LD_LIBRARY_PATH}")

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

  foreach(test_name ${VLLM_ASCEND_KERNEL_TESTS})
    vllm_ascend_add_device_executable(${test_name} "kernels/${test_name}.cpp" common/main.cpp)

    # Cross builds get one ctest entry per binary instead of one per GTest case.
    # Even PRE_TEST discovery has to execute the binary, and an aarch64 binary on
    # the x86 build host cannot run - `ctest -N` there aborts on the first
    # discovery attempt and lists nothing at all, benchmarks included. Running
    # ctest on the 310P itself, where the binaries are native, is unaffected.
    if(COMMAND gtest_discover_tests AND NOT CMAKE_VERSION VERSION_LESS "3.18" AND NOT CMAKE_CROSSCOMPILING)
      # PRE_TEST defers enumeration to `ctest` time. The default POST_BUILD mode
      # runs the binary during the build, which fails on a build host that has the
      # CANN headers but no driver.
      gtest_discover_tests(${test_name}
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 120
        PROPERTIES
          ENVIRONMENT "LD_LIBRARY_PATH=${VLLM_ASCEND_TEST_LIBRARY_PATH};ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
          LABELS "correctness"
          TIMEOUT 900)
    else()
      add_test(NAME ${test_name} COMMAND ${test_name})
      set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${VLLM_ASCEND_TEST_LIBRARY_PATH};ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
        LABELS "correctness"
        TIMEOUT 900)
    endif()
  endforeach()

  message(STATUS "vllm-ascend tests: configured ${VLLM_ASCEND_KERNEL_TESTS}")

  # ---------------------------------------------------------------------------
  # Benchmark binaries
  # ---------------------------------------------------------------------------
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

else()
  # ===========================================================================
  # CUDA backend
  # ===========================================================================

  # ---------------------------------------------------------------------------
  # Kernels
  # ---------------------------------------------------------------------------
  #
  # Kept in their own library so the boundary is visible: everything in here is
  # compiled by nvcc and nothing in here includes GTest.

  add_library(vllm_ascend_cuda_kernels STATIC
    kernels/cuda/rmsnorm_kernel.cu
    kernels/cuda/rotary_kernel.cu
    kernels/cuda/swiglu_kernel.cu
    kernels/cuda/paged_attention_kernel.cu)

  target_include_directories(vllm_ascend_cuda_kernels
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/common"
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda")

  target_link_libraries(vllm_ascend_cuda_kernels PUBLIC ${VLLM_ASCEND_TEST_CUDA_LIBS})

  vllm_ascend_tests_apply_warning_flags(vllm_ascend_cuda_kernels PRIVATE)

  # ---------------------------------------------------------------------------
  # Shared test infrastructure
  # ---------------------------------------------------------------------------
  #
  # cpu_reference.cpp is the same file the Ascend build compiles. That is the
  # point of the backend split: the ground truth does not fork.

  add_library(vllm_ascend_test_common STATIC
    common/cpu_reference.cpp
    common/cuda_runtime.cu)

  target_include_directories(vllm_ascend_test_common
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/common")

  target_link_libraries(vllm_ascend_test_common
    PUBLIC
      vllm_ascend_cuda_kernels
      ${VLLM_ASCEND_TEST_CUDA_LIBS}
      ${VLLM_ASCEND_TEST_GTEST_TARGET}
      ${CMAKE_DL_LIBS}
      Threads::Threads)

  target_compile_definitions(vllm_ascend_test_common PUBLIC USE_CUDA_BACKEND)

  vllm_ascend_tests_apply_warning_flags(vllm_ascend_test_common PUBLIC)

  # ---------------------------------------------------------------------------
  # Test binaries
  # ---------------------------------------------------------------------------

  set(VLLM_ASCEND_CUDA_KERNEL_TESTS
    test_rmsnorm
    test_rotary_embedding
    test_activation_swiglu
    test_paged_attention)

  foreach(test_name ${VLLM_ASCEND_CUDA_KERNEL_TESTS})
    add_executable(${test_name} "kernels/cuda/${test_name}.cpp" common/cuda_main.cpp)
    target_link_libraries(${test_name} PRIVATE vllm_ascend_test_common)

    list(APPEND VLLM_ASCEND_TEST_BINARY_FILES "$<TARGET_FILE:${test_name}>")

    if(NOT CMAKE_CROSSCOMPILING)
      # PRE_TEST defers enumeration to `ctest` time. The default POST_BUILD mode
      # runs the binary during the build, which fails on a build host that has
      # the CUDA headers but no driver.
      gtest_discover_tests(${test_name}
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 120
        PROPERTIES
          LABELS "correctness"
          TIMEOUT 900)
    else()
      add_test(NAME ${test_name} COMMAND ${test_name})
      set_tests_properties(${test_name} PROPERTIES LABELS "correctness" TIMEOUT 900)
    endif()
  endforeach()

  message(STATUS "vllm-ascend tests: configured ${VLLM_ASCEND_CUDA_KERNEL_TESTS}")

  if(VLLM_ASCEND_TESTS_BUILD_BENCHMARKS)
    # The bench_* binaries drive aclnn operators through common/benchmark.cpp and
    # have no CUDA counterpart yet. Say so once rather than failing the build.
    message(STATUS "vllm-ascend tests: benchmarks are Ascend-only, skipping them for the CUDA backend")
  endif()

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
