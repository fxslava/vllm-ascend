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
# TIER 2 - sim. The CANN camodel, and nothing else.
#
# Reached with -DRUN_MODE=sim on a 950PR SoC. These binaries link
# libruntime_camodel.so in place of libruntime.so and carry the simulator
# directory first on their RUNPATH, so they run on a machine with no NPU at all
# and CANNOT run on silicon. That is the tier's definition, not a limitation of
# it: the device tier exists for silicon.
#
# What it is for: functional validation of the Ascend C kernels against the CPU
# reference. The camodel is a cycle-level functional model, so what it proves is
# that the kernel computes the right thing - and what it cannot say anything
# about is how long that takes. Nothing in this tier is timed, and nothing in it
# may ever be presented as a performance number.
#
# Shapes are deliberately the smallest that still exercise the feature under
# test (head_size 64, block_size 16, context 32), because a single decode pass
# here is two to four minutes. ctest gets a 5400 s timeout for that reason.
#
# Exit criterion: every case passes, or skips naming an operator the camodel has
# no binary kernel for. A timeout is a failure of the machine, not of the
# kernel; re-run before believing it.
# -----------------------------------------------------------------------------

set(VLLM_ASCEND_TEST_TIER_LABEL "simulator")

if(NOT VLLM_ASCEND_TESTS_HAVE_TURBOQUANT)
  message(STATUS
    "vllm-ascend tests: [simulator] empty - every binary in this tier drives the TurboQuant "
    "Ascend C kernels and VLLM_ASCEND_TESTS_BUILD_TURBOQUANT_KERNELS is OFF")
  return()
endif()

set(VLLM_ASCEND_SIM_TESTS
  # The fp8 Cube fractal contract: NZ layout, LoadData step units, Mmad
  # argument order, Fixpipe output layout. Seconds, not minutes, and it is what
  # every other Cube case here depends on being right.
  test_sim_950pr_cube_gemm
  # Exploratory spike, isolated from the decode: a D = 256 Walsh-Hadamard whose
  # lower four butterfly stages are one Cube Mmad. Drives its own kernel source
  # (sim_hadamard_hybrid_kernels.cpp in this directory), not the decode's.
  test_sim_950pr_cube_hadamard
  # The standalone query rotation, both dispatch paths, against the CPU
  # reference. No decode in it, so it is seconds to a minute -- and it is the
  # gate on everything below, because the decode kernels no longer contain a
  # Walsh-Hadamard transform to be wrong.
  test_sim_950pr_turboquant_rotate_q
  test_sim_950pr_turboquant_kernels
  test_sim_950pr_turboquant_decode
  # The fused single-launch Cube decode, kv4fp8 at D = 256, five shapes: goldens,
  # cos vs fp32, the kernel writer's cache bytes, no sentinel left, no exception dump.
  test_sim_950pr_turboquant_fused
  # Single-shot dispatch + fidelity smoke of the Cube-native multi-mode decode.
  # One pass per mode at S = 16; see the file header for why it must stay one.
  test_sim_950pr_turboquant_multimode)

foreach(test_name ${VLLM_ASCEND_SIM_TESTS})
  vllm_ascend_add_device_executable(${test_name} "${test_name}.cpp" ../common/main_950pr.cpp)
  target_link_libraries(${test_name} PRIVATE vllm_ascend_test_turboquant)
  target_compile_definitions(${test_name} PRIVATE VLLM_ASCEND_TEST_TIER="simulator")

  # The camodel replaces libruntime.so, so its directory has to be ahead of the
  # toolkit's on the loader path at run time as well as at link time. This is
  # the one tier that wants it there; see the note in
  # vllm_ascend_add_device_executable for why every other one must not inherit
  # it from CANN's unconditional link directory.
  set_target_properties(${test_name} PROPERTIES
    INSTALL_RPATH
      "$ORIGIN;${CMAKE_BINARY_DIR}/lib;${VLLM_ASCEND_SIMULATOR_LIB_DIR};${ASCEND_LIB_DIRS}")

  # A camodel decode pass is minutes, not milliseconds. The third argument
  # registers the binary as one ctest entry rather than one per case; see the
  # note at the function definition for the static-initialisation abort that
  # makes GTest discovery unusable for anything linking the kernel library.
  vllm_ascend_register_correctness_test(${test_name} 5400 TRUE)

  if(TEST ${test_name})
    set_tests_properties(${test_name} PROPERTIES
      ENVIRONMENT
        "LD_LIBRARY_PATH=${VLLM_ASCEND_SIMULATOR_LIB_DIR}:${VLLM_ASCEND_TEST_LIBRARY_PATH};ASCEND_HOME_PATH=${ASCEND_HOME_PATH}")
  endif()
endforeach()

message(STATUS "vllm-ascend tests: [simulator] ${VLLM_ASCEND_SIM_TESTS}")
