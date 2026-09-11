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
# Standalone GTest suite for the kernels a Qwen3.5 forward pass needs, driven
# straight through the CANN runtime.
#
# This project is deliberately NOT included from the repository root
# CMakeLists.txt: that one requires Torch 2.10.0 and pybind11, and the whole
# point here is to build without Python, PyTorch or torch_npu. Configure it on
# its own:
#
#   cmake -S csrc/tests -B build/csrc-tests -DCMAKE_BUILD_TYPE=Release
#
# -----------------------------------------------------------------------------
# THREE EXECUTION TIERS, MUTUALLY EXCLUSIVE BY CONSTRUCTION
# -----------------------------------------------------------------------------
#
# A test is defined by where it can run, and the three answers need different
# link lines, not just different runtime gates. Each tier is a subdirectory with
# its own CMakeLists.txt, and no binary belongs to two of them:
#
#   host/          test_host_*             Pure CPU. Algorithmic verification of
#                                          the codec - the Lloyd-Max grid, the Pi
#                                          involution, the packing, the golden
#                                          reference dumps. Links neither
#                                          libascendcl.so nor any NPU runtime,
#                                          and is built in EVERY configuration,
#                                          including -DVLLM_ASCEND_TESTS_HOST_ONLY=ON
#                                          on a machine with no toolkit at all.
#
#   sim/           test_sim_950pr_*        CAModel only. Functional validation of
#                                          the Ascend C kernels against the CPU
#                                          reference, at shapes small enough for
#                                          a cycle-level simulator to finish.
#                                          Links libruntime_camodel.so out of
#                                          $ASCEND_HOME_PATH/tools/simulator, and
#                                          therefore CANNOT run on silicon.
#
#   device/        test_device_950pr_*     Physical 950PR silicon. Production
#                  bench_device_950pr_*    shapes and every timing measurement.
#                                          Links the real libruntime.so and
#                                          libascendcl.so, with no simulator on
#                                          the link line and none on the RUNPATH,
#                                          so there is no CAModel fallback to
#                                          accidentally take.
#
# sim/ and device/ are selected by RUN_MODE and are never configured together:
# ascendc_library() builds one kernel library per RUN_MODE, so a tree cannot hold
# both a camodel and a silicon build of it. That is what makes the exclusivity
# structural rather than a naming convention.
#
#   -DRUN_MODE=npu   (default)  host/ + device/
#   -DRUN_MODE=sim              host/ + sim/
#   -DVLLM_ASCEND_TESTS_HOST_ONLY=ON   host/ alone
#
# -----------------------------------------------------------------------------
# ARCHITECTURE ISOLATION
# -----------------------------------------------------------------------------
#
# SOC_VERSION selects the part, and the two legs are exclusive:
#
#   SOC_VERSION=Ascend950PR_*   the three tiers above. device_310p/ is not
#                               configured, not compiled, and contributes no
#                               target - a `*310*` binary cannot exist in a
#                               950PR build tree.
#   anything else               device_310p/ alone, plus host/. The 950PR tiers
#                               are not configured.
#
# device_310p/ keeps its historical binary names (test_matmul_310p and friends)
# on purpose: .agents/skills/ascend-310p-csrc-tests/SKILL.md and the shipped
# dist/ascend310p3-aarch64-tests/run_tests.sh both drive them by name, they
# already carry their part in the name, and they are unreachable from a 950PR
# build anyway. The isolation the tiers needed was directory- and
# configure-level, not a rename.
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

# Host-only mode: build just the targets that need neither the CANN headers
# nor libascendcl.so. Today that is test_turbo_quant_fidelity, whose whole
# pipeline is the CPU reference in reference/turbo_quant_cpu.h, so the 4-bit
# KV cache can be measured on a build machine, a simulator or in CI with no
# NPU and no toolkit installed at all.
#
#   cmake -S csrc/tests -B build/csrc-tests-host -DVLLM_ASCEND_TESTS_HOST_ONLY=ON
#
# The host-only targets are built in the normal configuration too; this
# option only removes everything else.
option(VLLM_ASCEND_TESTS_HOST_ONLY "Build only the targets that need no CANN toolkit" OFF)

# The TurboQuant 4-bit KV cache: the Ascend C kernels themselves, plus the tests
# and the benchmark that drive them. Unlike everything else in this file, this
# compiles device code - it needs the Ascend C kernel toolchain (ccec / bisheng
# and tools/tikcpp/ascendc_kernel_cmake), not just the CANN headers and
# libascendcl.so. Turn it off to build the rest of a 950PR tier on a machine
# that has the toolkit but not the kernel compiler; with it OFF the sim tier is
# empty, because every binary in it drives those kernels.
#
# Reachable only on a 950PR SoC: the kernels carry an AICore config for arch32
# and arch35 and nothing else.
option(VLLM_ASCEND_TESTS_BUILD_TURBOQUANT_KERNELS
       "Compile the TurboQuant Ascend C kernels and build the tests that drive them" ON)

# The Cube-native multi-mode decode (turboquant_mm_kernels.cpp) is work in
# progress and does not pass: the score GEMM's [n, k] B form does not reproduce
# the host product, and TurboQuantCubeMm::Init allocates L0 on both halves of
# the MIX kernel, which stalls the split kernel on the camodel.
# TURBOQUANT_TESTS.md section 13.8 records both in full.
#
# It is still COMPILED in every configuration - it has to keep building clean
# under -Werror while it is being fixed - so what this option changes is only
# whether the cases that LAUNCH it run by default:
#
#   OFF (default)  bench_device_950pr_turboquant runs the verified 4-bit AIV
#                  path and records a skip for each Cube case; the two sim
#                  tests that drive the Cube skip through
#                  REQUIRE_CUBE_WIP_OPT_IN.
#   ON             those cases run without any environment variable, for a tree
#                  dedicated to fixing the path.
#
# VLLM_ASCEND_TQ_CUBE_WIP overrides this at run time in BOTH directions, so a
# tree configured with it ON can still be asked for the stable baseline.
#
# SCOPE: the sim tier only. bench_device_950pr_turboquant runs its Cube legs by
# default and is not affected by this option; VLLM_ASCEND_TQ_CUBE_WIP=0 is what
# drops them there. The L0 allocation defect that made those legs fault is
# fixed; what remains open is the score GEMM's operand form, which is a wrong
# number rather than a dead stream, and the benchmark says so on every run.
#
# Delete the option, and the gates it feeds, when the Cube path works.
option(VLLM_ASCEND_TESTS_ENABLE_WIP_CUBE
       "Run the work-in-progress Cube sim cases by default (they are compiled either way)" OFF)

if(VLLM_ASCEND_TESTS_ENABLE_WIP_CUBE)
  add_compile_definitions(VLLM_ASCEND_TQ_CUBE_WIP_DEFAULT_ON)
  message(WARNING
    "vllm-ascend tests: VLLM_ASCEND_TESTS_ENABLE_WIP_CUBE=ON - the Cube-native sim cases will run "
    "by default. That path does not yet reproduce the host product (TURBOQUANT_TESTS.md section "
    "13.8); treat what it computes as provisional.")
endif()

# How the Ascend C kernels are built and where they run.
#
#   npu   real silicon, linked against libruntime.so                (default)
#   sim   the CANN camodel, linked against libruntime_camodel.so out of
#         $ASCEND_HOME_PATH/tools/simulator/${SOC_VERSION}/lib
#
# The simulator is the only way to execute an arch35 kernel without a 950PR
# attached, and it is slow enough - cycle-level - that the tests written for it
# run a single decode pass rather than a sweep. See
# kernels/ascend/test_turboquant_npu_simulator.cpp.
#
# The name and the values are CANN's, not this project's: ascendc_library()
# reads ${RUN_MODE} directly.
set(RUN_MODE "npu" CACHE STRING "Ascend C run mode: npu (silicon) or sim (CANN camodel)")
set_property(CACHE RUN_MODE PROPERTY STRINGS npu sim)

if(NOT RUN_MODE STREQUAL "npu" AND NOT RUN_MODE STREQUAL "sim")
  message(FATAL_ERROR "RUN_MODE must be 'npu' or 'sim', not '${RUN_MODE}'")
endif()

# -----------------------------------------------------------------------------
# Tier resolution
# -----------------------------------------------------------------------------
#
# Decided once, here, so every message below and every subdirectory reads the
# same answer. See the tier and isolation notes at the top of this file.

string(TOLOWER "${SOC_VERSION}" VLLM_ASCEND_SOC_LOWER)
if(VLLM_ASCEND_SOC_LOWER MATCHES "^ascend950")
  set(VLLM_ASCEND_TESTS_SOC_IS_950PR TRUE)
else()
  set(VLLM_ASCEND_TESTS_SOC_IS_950PR FALSE)
endif()

# ENABLE_ASCEND_950PR used to be the way in. SOC_VERSION is now the only one, so
# a script that still passes the old flag has to be told rather than silently
# handed a 310P tree it did not ask for.
if(DEFINED ENABLE_ASCEND_950PR AND ENABLE_ASCEND_950PR AND NOT VLLM_ASCEND_TESTS_SOC_IS_950PR)
  message(FATAL_ERROR
    "ENABLE_ASCEND_950PR is set but SOC_VERSION is '${SOC_VERSION}'.\n"
    "That option is gone: the 950PR tiers are selected by the SoC alone. Configure with\n"
    "  -DSOC_VERSION=Ascend950PR_9599\n"
    "and drop -DENABLE_ASCEND_950PR.")
endif()

# host/ is unconditional. The other three are exclusive.
set(VLLM_ASCEND_TESTS_TIER_HOST      TRUE)
set(VLLM_ASCEND_TESTS_TIER_SIM       FALSE)
set(VLLM_ASCEND_TESTS_TIER_DEVICE    FALSE)
set(VLLM_ASCEND_TESTS_TIER_DEVICE310 FALSE)

if(NOT VLLM_ASCEND_TESTS_HOST_ONLY)
  if(VLLM_ASCEND_TESTS_SOC_IS_950PR)
    if(RUN_MODE STREQUAL "sim")
      set(VLLM_ASCEND_TESTS_TIER_SIM TRUE)
    else()
      set(VLLM_ASCEND_TESTS_TIER_DEVICE TRUE)
    endif()
  else()
    set(VLLM_ASCEND_TESTS_TIER_DEVICE310 TRUE)
    if(RUN_MODE STREQUAL "sim")
      message(WARNING
        "RUN_MODE=sim has no effect on SOC_VERSION '${SOC_VERSION}': the camodel tier exists only "
        "for the 950PR, whose Ascend C kernels this suite builds itself. The 310P tests call "
        "operators CANN already shipped and need real hardware.")
    endif()
  endif()
endif()

message(STATUS "vllm-ascend tests: SoC ${SOC_VERSION}, RUN_MODE ${RUN_MODE}")
message(STATUS "vllm-ascend tests: tiers -> host=${VLLM_ASCEND_TESTS_TIER_HOST} "
               "sim=${VLLM_ASCEND_TESTS_TIER_SIM} device=${VLLM_ASCEND_TESTS_TIER_DEVICE} "
               "device_310p=${VLLM_ASCEND_TESTS_TIER_DEVICE310}")
if(VLLM_ASCEND_TESTS_SOC_IS_950PR)
  message(STATUS "vllm-ascend tests: 310P targets are excluded from this build entirely")
endif()

# -----------------------------------------------------------------------------
# CANN toolkit
# -----------------------------------------------------------------------------

find_package(Threads REQUIRED)

if(VLLM_ASCEND_TESTS_HOST_ONLY)
  message(STATUS "vllm-ascend tests: host-only build, skipping CANN discovery")
endif()

if(NOT VLLM_ASCEND_TESTS_HOST_ONLY)

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

endif()  # NOT VLLM_ASCEND_TESTS_HOST_ONLY

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

if(VLLM_ASCEND_TESTS_FETCH_GTEST)
  set(VLLM_ASCEND_TEST_GTEST_MAIN_TARGET gtest_main)
else()
  set(VLLM_ASCEND_TEST_GTEST_MAIN_TARGET GTest::gtest_main)
endif()

enable_testing()

# -----------------------------------------------------------------------------
# Host-only targets
# -----------------------------------------------------------------------------
#
# Sources that touch neither acl.h nor any CANN library, collected into their own
# library so the binaries built from them carry no DT_NEEDED entry for
# libascendcl.so and link on a machine with no toolkit. common/cpu_reference.cpp
# and common/golden_layer3.cpp are already pure host code; keeping them out of
# vllm_ascend_test_common here rather than reusing that target is what makes the
# independence real instead of accidental.

# The dump the golden tests read. Defined before the 950PR block because the
# host-only fidelity report needs it too; QWEN_GOLDEN_LAYER3_DIR overrides it at
# run time. The files are tracked with Git LFS and the loader recognises an
# unfetched pointer and skips - see common/golden_layer3.cpp.
set(VLLM_ASCEND_GOLDEN_LAYER3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/golden_layer3")

add_library(vllm_ascend_test_host STATIC
  common/cpu_reference.cpp
  common/golden_layer3.cpp)

target_include_directories(vllm_ascend_test_host
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/common"
    "${CMAKE_CURRENT_SOURCE_DIR}/reference")

target_compile_definitions(vllm_ascend_test_host
  PUBLIC VLLM_ASCEND_GOLDEN_LAYER3_DIR="${VLLM_ASCEND_GOLDEN_LAYER3_DIR}")

target_link_libraries(vllm_ascend_test_host
  PUBLIC
    ${VLLM_ASCEND_TEST_GTEST_TARGET}
    Threads::Threads)

target_compile_options(vllm_ascend_test_host PUBLIC -Wall -Wextra)
if(VLLM_ASCEND_TESTS_WERROR)
  target_compile_options(vllm_ascend_test_host PUBLIC -Werror)
endif()

if(NOT EXISTS "${VLLM_ASCEND_GOLDEN_LAYER3_DIR}/golden_output.bin")
  message(WARNING
    "golden dump not found at ${VLLM_ASCEND_GOLDEN_LAYER3_DIR}.\n"
    "The golden cases of the host tier and of test_device_950pr_qwen_layer_golden will skip. "
    "Run `git lfs pull`, or regenerate with `python scripts/dump_qwen35_layer3.py`.")
endif()

# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
#
# Defined before any tier subdirectory, because all four call them.

# Every executable this project produces, collected so the strip step in
# .vscode-build-aarch64.sh can be driven from a generated manifest rather than a
# hand-maintained list. A stale name in that list is handed to `strip` as a path
# that either does not exist or is not an ELF file, which is where the "file
# format not recognized" warnings came from.
#
# A GLOBAL property rather than a variable: the tiers are subdirectories now, and
# a variable set with PARENT_SCOPE from inside one of them would land in that
# subdirectory's scope and never reach this file.
set_property(GLOBAL PROPERTY VLLM_ASCEND_TEST_BINARIES "")

function(vllm_ascend_record_binary target_name)
  set_property(GLOBAL APPEND PROPERTY VLLM_ASCEND_TEST_BINARIES "$<TARGET_FILE:${target_name}>")
endfunction()

# Discovery runs the binary, which needs libopapi.so and friends on the loader
# path even though they are opened lazily.
if(NOT VLLM_ASCEND_TESTS_HOST_ONLY)
  set(VLLM_ASCEND_TEST_LIBRARY_PATH "${ASCEND_HOME_PATH}/lib64:$ENV{LD_LIBRARY_PATH}")
endif()

# Shared configuration for both the test and the benchmark binaries: they differ
# only in which main() they link and how ctest is told to run them.
function(vllm_ascend_add_device_executable target_name source_file entry_point)
  add_executable(${target_name} "${source_file}" "${entry_point}")
  target_link_libraries(${target_name} PRIVATE vllm_ascend_test_common)

  # $ORIGIN keeps a copied-out build tree working; the CANN lib dirs cover the
  # normal in-place run, and ${CMAKE_BINARY_DIR}/lib is where ascendc_library()
  # leaves libvllm_ascend_turboquant.so.
  #
  # BUILD_WITH_INSTALL_RPATH links with the list below verbatim and skips the
  # build-rpath computation entirely. That is the point of it, not a shortcut:
  # CANN's tools/tikcpp/ascendc_kernel_cmake adds
  #
  #   ${ASCEND_CANN_PACKAGE_PATH}/tools/simulator/${SOC_VERSION}/lib
  #
  # as a PUBLIC link directory on every ascendc_library() target, *without ever
  # looking at RUN_MODE*, and CMake then appends it to the consumer's RUNPATH.
  # A device-tier binary would come out carrying a camodel directory on its own
  # loader path - harmless on a machine where it does not exist, and a silent
  # swap of the simulator runtime for the real one on a machine where it does.
  # DT_NEEDED never names a camodel library, so nothing but this stops it, and
  # the tier's "no CAModel fallback" guarantee rests on it. Measured:
  # BUILD_RPATH_USE_LINK_PATH FALSE does *not* suppress it on CMake 3.22, and
  # this does.
  #
  # sim/CMakeLists.txt deliberately puts that same directory back, first,
  # because there it is the point.
  set_target_properties(${target_name} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN;${CMAKE_BINARY_DIR}/lib;${ASCEND_LIB_DIRS}")

  vllm_ascend_record_binary(${target_name})
endfunction()

# Cross builds get one ctest entry per binary instead of one per GTest case.
# Even PRE_TEST discovery has to execute the binary, and an aarch64 binary on
# the x86 build host cannot run - `ctest -N` there aborts on the first discovery
# attempt and lists nothing at all, benchmarks included. Running ctest on the
# part itself, where the binaries are native, is unaffected.
#
# The optional second argument is the per-test timeout in seconds; it defaults to
# 900, which is generous for a kernel launch on silicon and nowhere near enough
# for one under the camodel simulator, where a single decode pass is minutes.
# Both registration paths below need it, which is why it is a parameter rather
# than a set_tests_properties() call afterwards: gtest_discover_tests names its
# tests at ctest time, so there is nothing to set properties on here.
#
# The optional third argument suppresses GTest discovery and registers the whole
# binary as one ctest entry, the way a cross build already does. It exists for
# the binaries that link libvllm_ascend_turboquant.so, and the reason is not a
# preference:
#
#   PRE_TEST discovery has to *execute* the binary to enumerate its cases. Built
#   for silicon, that library's static initialisation aborts on a host with no
#   NPU driver - measured on CANN 9.2.0-beta.2:
#
#     terminate called after throwing an instance of 'std::logic_error'
#       what(): basic_string::_S_construct null not valid
#
#   before main() reaches RUN_ALL_TESTS, so even `--gtest_list_tests` aborts.
#   ctest reports "Error running test executable / Subprocess aborted" and then
#   lists *nothing at all* - not the failing binary, not the other forty tests,
#   because one failed discovery ends the whole enumeration. `ctest -N` on a
#   build host becomes useless.
#
#   It is not this suite's bug: test_device_950pr_matmul, which links no kernel
#   library, enumerates fine in the same tree, and the same binaries enumerate
#   fine in the sim tier where the camodel runtime is present. It is what a
#   silicon-mode kernel library does when the driver is absent, and the cost of
#   tripping over it is out of all proportion to the per-case ctest granularity
#   it buys.
function(vllm_ascend_register_correctness_test target_name)
  if(ARGC GREATER 1)
    set(test_timeout ${ARGV1})
  else()
    set(test_timeout 900)
  endif()
  set(discover_cases TRUE)
  if(ARGC GREATER 2 AND ARGV2)
    set(discover_cases FALSE)
  endif()
  if(discover_cases AND COMMAND gtest_discover_tests AND NOT CMAKE_VERSION VERSION_LESS "3.18" AND NOT CMAKE_CROSSCOMPILING)
    # PRE_TEST defers enumeration to `ctest` time. The default POST_BUILD mode
    # runs the binary during the build, which fails on a build host that has the
    # CANN headers but no driver.
    gtest_discover_tests(${target_name}
      DISCOVERY_MODE PRE_TEST
      DISCOVERY_TIMEOUT 120
      PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${VLLM_ASCEND_TEST_LIBRARY_PATH};ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
        LABELS "correctness;${VLLM_ASCEND_TEST_TIER_LABEL}"
        TIMEOUT ${test_timeout})
  else()
    add_test(NAME ${target_name} COMMAND ${target_name})
    set_tests_properties(${target_name} PROPERTIES
      ENVIRONMENT "LD_LIBRARY_PATH=${VLLM_ASCEND_TEST_LIBRARY_PATH};ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
      LABELS "correctness;${VLLM_ASCEND_TEST_TIER_LABEL}"
      TIMEOUT ${test_timeout})
  endif()
endfunction()

# -----------------------------------------------------------------------------
# Tier 1: host
# -----------------------------------------------------------------------------
#
# Unconditional. Nothing in it links a CANN library, so it is the only tier that
# survives -DVLLM_ASCEND_TESTS_HOST_ONLY=ON, and the only one that builds on a
# machine with no toolkit installed.

add_subdirectory(host)

if(NOT VLLM_ASCEND_TESTS_HOST_ONLY)

# -----------------------------------------------------------------------------
# Shared device infrastructure
# -----------------------------------------------------------------------------
#
# The device-side plumbing every non-host tier needs: the runtime lifecycle, the
# dlopen/dlsym operator loader, the hand-declared aclnn prototypes and the
# benchmark harness.
#
# common/aclnn_ops.{hpp,cpp} is NOT a 310P file despite its history. It is the
# suite's whole operator prototype table, and the 950PR tests read it too -
# ops::kMatmul, ops::kRmsNorm, ops::kSwiGlu, ops::kScatterPaKvCache and
# ops::kApplyRotaryPosEmbV2 are all resolved through it. It stays shared; what
# the architecture isolation excludes is the 310P *targets*, in device_310p/.

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
# Shared 950PR infrastructure
# -----------------------------------------------------------------------------
#
# Built for the sim and device tiers only, so a 310P tree carries neither the
# 950PR operator audit nor the golden-dump path define.

if(VLLM_ASCEND_TESTS_TIER_SIM OR VLLM_ASCEND_TESTS_TIER_DEVICE)
  add_library(vllm_ascend_test_common_950pr STATIC
    common/aclnn_ops_950pr.cpp
    common/golden_layer3.cpp
    common/partial_rotary_950pr.cpp)

  target_link_libraries(vllm_ascend_test_common_950pr PUBLIC vllm_ascend_test_common)

  target_compile_definitions(vllm_ascend_test_common_950pr
    PUBLIC VLLM_ASCEND_GOLDEN_LAYER3_DIR="${VLLM_ASCEND_GOLDEN_LAYER3_DIR}")

  # libcust_opapi.so, where the kernels built out of csrc/ live, is opened with
  # dlopen at run time by common/aclnn_runtime.cpp - it is found through
  # ASCEND_CUSTOM_OPP_PATH or the vendors config, not through a link line, so
  # there is deliberately nothing to link here. The lookup below only reports
  # whether one is visible at configure time, so a run that skips the custom
  # rotary path says so before the build rather than after it.
  if(DEFINED ENV{ASCEND_CUSTOM_OPP_PATH} AND NOT "$ENV{ASCEND_CUSTOM_OPP_PATH}" STREQUAL "")
    message(STATUS "vllm-ascend tests: ASCEND_CUSTOM_OPP_PATH=$ENV{ASCEND_CUSTOM_OPP_PATH}")
  elseif(EXISTS "${ASCEND_HOME_PATH}/opp/vendors/config.ini")
    message(STATUS "vllm-ascend tests: custom op vendors at ${ASCEND_HOME_PATH}/opp/vendors")
  else()
    message(STATUS
      "vllm-ascend tests: no custom operator package visible; aclnnInplacePartialRotaryMul "
      "will not resolve and the rotary stages take the packed aclnnApplyRotaryPosEmbV2 fallback")
  endif()

  # ---------------------------------------------------------------------------
  # The TurboQuant Ascend C kernels
  # ---------------------------------------------------------------------------
  #
  # The only part of this suite that compiles device code. Everything else calls
  # operators CANN already built; these are built here, from
  # csrc/attention/turboquant/turboquant_kernels.cpp - the same source the wheel
  # compiles, not a copy.
  #
  # One library per tree, built for whichever RUN_MODE this configuration
  # selected. That is precisely why sim/ and device/ cannot both be configured:
  # they would need two different builds of this one target.

  set(VLLM_ASCEND_TESTS_HAVE_TURBOQUANT FALSE)

  if(VLLM_ASCEND_TESTS_BUILD_TURBOQUANT_KERNELS)
    # Two sources, one library. turboquant_kernels.cpp is the AIV-only 4-bit
    # path; turboquant_mm_kernels.cpp is the multi-mode Cube-native one. They
    # share a library rather than having one each because the Cube decode reuses
    # the AIV path's combine kernel, and a <<<>>> call site can only reach a
    # kernel whose generated launcher is in the same ascendc_library.
    set(VLLM_ASCEND_TURBOQUANT_KERNEL_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/../attention/turboquant/turboquant_kernels.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../attention/turboquant/turboquant_mm_kernels.cpp"
        # Test-owned device code, and the one source in this list the wheel does
        # not build - the root CMakeLists.txt names only the two above. It is an
        # exploratory spike (a Cube-factorised Walsh-Hadamard, see its header)
        # and it rides in this library rather than one of its own because a
        # <<<>>> call site can only reach a kernel whose generated launcher is
        # in the same ascendc_library. Nothing in csrc/attention/ includes it.
        # Driven by sim/test_sim_950pr_cube_hadamard.cpp at one shape and by
        # device/test_device_950pr_cube_hadamard.cpp over the whole sweep, so it
        # has to be in the library both tiers link.
        "${CMAKE_CURRENT_SOURCE_DIR}/sim/sim_hadamard_hybrid_kernels.cpp")

    foreach(kernel_source ${VLLM_ASCEND_TURBOQUANT_KERNEL_SOURCE})
      if(NOT EXISTS "${kernel_source}")
        message(FATAL_ERROR "TurboQuant kernel source not found at ${kernel_source}")
      endif()
    endforeach()

    set(VLLM_ASCEND_SIMULATOR_LIB_DIR "${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib")
    if(VLLM_ASCEND_TESTS_TIER_SIM)
      if(NOT EXISTS "${VLLM_ASCEND_SIMULATOR_LIB_DIR}/libruntime_camodel.so")
        message(FATAL_ERROR
          "The sim tier needs the camodel for ${SOC_VERSION}, and "
          "${VLLM_ASCEND_SIMULATOR_LIB_DIR}/libruntime_camodel.so does not exist.\n"
          "Check SOC_VERSION against the directory names under ${ASCEND_HOME_PATH}/tools/simulator, "
          "or configure with -DRUN_MODE=npu to build the device tier instead.")
      endif()
      message(STATUS "vllm-ascend tests: camodel libs at ${VLLM_ASCEND_SIMULATOR_LIB_DIR}")
    endif()

    # Defines the vllm_ascend_turboquant target. Kept in a subdirectory so
    # ascendc.cmake's side effects stay out of this scope; see the note there.
    add_subdirectory(turboquant)

    # The Torch-free binding: the two _impl prototypes, the codec table builder
    # and the grid arithmetic that turboquant_torch_adpt.h would have done with
    # ATen. Carries the reference/ include directory because the Pi diagonal
    # comes from the CPU reference rather than from a second generator.
    add_library(vllm_ascend_test_turboquant STATIC common/turboquant_launch.cpp)

    target_include_directories(vllm_ascend_test_turboquant
      PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/reference")

    target_link_libraries(vllm_ascend_test_turboquant
      PUBLIC vllm_ascend_test_common_950pr vllm_ascend_turboquant)

    set(VLLM_ASCEND_TESTS_HAVE_TURBOQUANT TRUE)
  else()
    message(STATUS
      "vllm-ascend tests: VLLM_ASCEND_TESTS_BUILD_TURBOQUANT_KERNELS=OFF; no Ascend C kernels, "
      "and every binary that drives them is skipped")
  endif()
endif()

# -----------------------------------------------------------------------------
# Tiers 2 and 3, and the isolated 310P leg
# -----------------------------------------------------------------------------

if(VLLM_ASCEND_TESTS_TIER_SIM)
  add_subdirectory(sim)
endif()

if(VLLM_ASCEND_TESTS_TIER_DEVICE)
  add_subdirectory(device)
endif()

if(VLLM_ASCEND_TESTS_TIER_DEVICE310)
  add_subdirectory(device_310p)
endif()

endif()  # NOT VLLM_ASCEND_TESTS_HOST_ONLY

# -----------------------------------------------------------------------------
# Binary manifest
# -----------------------------------------------------------------------------
#
# One absolute path per line, written at generate time so the genexes above are
# resolved. Consumed by .vscode-build-aarch64.sh:
#
#   xargs -r -a build/csrc-tests-aarch64/test_binaries.txt aarch64-linux-gnu-strip --strip-unneeded

get_property(VLLM_ASCEND_TEST_BINARY_FILES GLOBAL PROPERTY VLLM_ASCEND_TEST_BINARIES)
string(ASCII 10 VLLM_ASCEND_NEWLINE)
file(GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test_binaries.txt"
  CONTENT "$<JOIN:${VLLM_ASCEND_TEST_BINARY_FILES},${VLLM_ASCEND_NEWLINE}>${VLLM_ASCEND_NEWLINE}")
