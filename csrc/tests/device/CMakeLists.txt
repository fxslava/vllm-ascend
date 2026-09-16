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
# TIER 3 - device. Physical Ascend 950PR silicon.
#
# Reached with -DRUN_MODE=npu (the default) on a 950PR SoC. Every binary here
# links the real libruntime.so and libascendcl.so. There is no camodel on the
# link line, none on the RUNPATH, and no fallback to one: a machine without the
# part cannot run these, and is not supposed to be able to.
#
# What it is for, and the split is deliberate:
#
#   correctness   the same properties the sim tier checks, at the shapes
#                 Qwen3.5-2B actually decodes at - head_dim 256, block_size 128,
#                 context 512/1024/2048 - which a cycle-level simulator cannot
#                 reach in usable time. Plus the four stock-operator tests and
#                 the end-to-end layer parity test, which have no camodel path
#                 at all because the camodel ships no binary kernels for them.
#
#   performance   every timing number this project quotes. The sim tier cannot
#                 produce one, so this is the only place they may come from.
#
# Exit criterion: every correctness case passes; the benchmarks exit 0 having
# produced a table, or exit 77 when no 950PR is attached, which ctest records as
# a skip. A benchmark that fails its checksum is a correctness failure wearing a
# benchmark's clothes - the launches stopped agreeing with each other - and must
# be treated as one.
# -----------------------------------------------------------------------------

set(VLLM_ASCEND_TEST_TIER_LABEL "device")

# --- correctness --------------------------------------------------------------

# Stock-operator and end-to-end tests. These call operators CANN already built,
# so they link the 950PR operator audit but not the Ascend C kernels.
set(VLLM_ASCEND_DEVICE_OPERATOR_TESTS
  test_device_950pr_matmul
  test_device_950pr_rmsnorm
  test_device_950pr_rotary_embedding
  test_device_950pr_activation_swiglu
  test_device_950pr_qwen_layer_golden)

foreach(test_name ${VLLM_ASCEND_DEVICE_OPERATOR_TESTS})
  vllm_ascend_add_device_executable(${test_name} "${test_name}.cpp" ../common/main_950pr.cpp)
  target_link_libraries(${test_name} PRIVATE vllm_ascend_test_common_950pr)
  target_compile_definitions(${test_name} PRIVATE VLLM_ASCEND_TEST_TIER="device")
  vllm_ascend_register_correctness_test(${test_name})
endforeach()

# Tests over the shared benchmark harness rather than over a kernel. This is the
# one binary in the tier that needs no device attached: what it covers is the
# arithmetic a benchmark run does *around* the launches - sample validation, the
# ASCEND_BENCH_* parses, the alignment rules, the checksum comparison - all of
# which produced wrong numbers on healthy hardware and so have to be testable
# without any. It lives here rather than in host/ because it includes acl.h and
# links libascendcl.so, which the host tier's entry criterion forbids.
vllm_ascend_add_device_executable(test_device_950pr_benchmark_harness
                                  test_device_950pr_benchmark_harness.cpp
                                  ../common/main_950pr.cpp)
target_link_libraries(test_device_950pr_benchmark_harness PRIVATE vllm_ascend_test_common_950pr)
target_compile_definitions(test_device_950pr_benchmark_harness
  PRIVATE VLLM_ASCEND_TEST_TIER="device_harness")
vllm_ascend_register_correctness_test(test_device_950pr_benchmark_harness)

set(VLLM_ASCEND_DEVICE_TESTS
  ${VLLM_ASCEND_DEVICE_OPERATOR_TESTS}
  test_device_950pr_benchmark_harness)

# The TurboQuant bare-metal suite: production shapes, and the only binary in the
# project that refuses to run under a camodel at run time as well as by
# construction (REQUIRE_PHYSICAL_ASCEND_950PR reads /proc/self/maps).
if(VLLM_ASCEND_TESTS_HAVE_TURBOQUANT)
  vllm_ascend_add_device_executable(test_device_950pr_turboquant
                                    test_device_950pr_turboquant.cpp
                                    ../common/main_950pr.cpp)
  target_link_libraries(test_device_950pr_turboquant PRIVATE vllm_ascend_test_turboquant)
  target_compile_definitions(test_device_950pr_turboquant
    PRIVATE VLLM_ASCEND_TEST_TIER="device")
  # One ctest entry per binary; see the note at the function definition.
  vllm_ascend_register_correctness_test(test_device_950pr_turboquant 900 TRUE)
  list(APPEND VLLM_ASCEND_DEVICE_TESTS test_device_950pr_turboquant)

  # The Cube-native decode's fidelity check, on silicon.
  #
  # It compiles the SIM tier's source deliberately rather than copying it: the
  # file drives the multi-rate decode through turboquant_launch and touches
  # nothing camodel-specific, so the only thing that differs between the two
  # tiers is which libruntime the link picks up. Keeping one source means the
  # cos bound the simulator enforces and the one silicon enforces cannot drift
  # apart, which is the whole value of running it here.
  #
  # Shape and rate come from the environment -- ASCEND_TQ_SIM_CONTEXT and
  # ASCEND_TQ_SIM_MODES -- so one invocation is one context length and one
  # rate. On the camodel that matters because a pass is minutes; on silicon it
  # is what lets a single binary cover S=64 and S=512 in separate runs.
  vllm_ascend_add_device_executable(test_device_950pr_turboquant_multimode
                                    ../sim/test_sim_950pr_turboquant_multimode.cpp
                                    ../common/main_950pr.cpp)
  target_link_libraries(test_device_950pr_turboquant_multimode PRIVATE vllm_ascend_test_turboquant)
  # VLLM_ASCEND_TEST_TIER_DEVICE, on top of the string label every device target
  # gets, is what the shared source switches its DEFAULT SWEEP on: one batch and
  # one context length under the camodel, and batch {1,8} x context
  # {64,512,1024,2048} at the production head count here. A string literal
  # cannot be tested with #ifdef, hence the second definition.
  target_compile_definitions(test_device_950pr_turboquant_multimode
    PRIVATE VLLM_ASCEND_TEST_TIER="device" VLLM_ASCEND_TEST_TIER_DEVICE=1)
  vllm_ascend_register_correctness_test(test_device_950pr_turboquant_multimode 900 TRUE)
  list(APPEND VLLM_ASCEND_DEVICE_TESTS test_device_950pr_turboquant_multimode)

  # The Cube-Hadamard spike's shape sweep: D in {64, 128, 256, 512} by V in
  # {1, 8, 16, 32}, three hybrid variants and an AIV-only baseline, all against
  # the CPU golden. It drives a test-owned kernel (sim/sim_hadamard_hybrid_kernels.cpp,
  # carried by the same ascendc_library) and touches nothing on the decode path.
  #
  # It lives in the device tier because of arithmetic, not preference: the sim
  # tier's copy of this check runs ONE shape and takes around 40 seconds on the
  # camodel, and 64 of them is most of a day. See the header of the sim file.
  vllm_ascend_add_device_executable(test_device_950pr_cube_hadamard
                                    test_device_950pr_cube_hadamard.cpp
                                    ../common/main_950pr.cpp)
  target_link_libraries(test_device_950pr_cube_hadamard PRIVATE vllm_ascend_test_turboquant)
  target_compile_definitions(test_device_950pr_cube_hadamard PRIVATE VLLM_ASCEND_TEST_TIER="device")
  vllm_ascend_register_correctness_test(test_device_950pr_cube_hadamard 900 TRUE)
  list(APPEND VLLM_ASCEND_DEVICE_TESTS test_device_950pr_cube_hadamard)
endif()

message(STATUS "vllm-ascend tests: [device] ${VLLM_ASCEND_DEVICE_TESTS}")

# --- performance --------------------------------------------------------------
#
# Registered with ctest under the "benchmark" label so the default run stays a
# correctness run:
#
#   ctest -LE benchmark    correctness only
#   ctest -L  benchmark    benchmarks only
#
# A benchmark exits 77 when no usable 950PR is attached, which SKIP_RETURN_CODE
# turns into a ctest skip rather than a failure.

if(VLLM_ASCEND_TESTS_BUILD_BENCHMARKS AND VLLM_ASCEND_TESTS_HAVE_TURBOQUANT)
  set(VLLM_ASCEND_DEVICE_BENCHMARKS
    bench_device_950pr_turboquant)

  add_custom_target(benchmarks COMMENT "Building the vllm-ascend 950PR device benchmarks")

  foreach(bench_name ${VLLM_ASCEND_DEVICE_BENCHMARKS})
    vllm_ascend_add_device_executable(${bench_name} "${bench_name}.cpp" ../common/bench_main_950pr.cpp)
    target_link_libraries(${bench_name} PRIVATE vllm_ascend_test_turboquant)
    target_compile_definitions(${bench_name} PRIVATE VLLM_ASCEND_TEST_TIER="device")
    add_dependencies(benchmarks ${bench_name})

    add_test(NAME ${bench_name} COMMAND ${bench_name})
    set_tests_properties(${bench_name} PROPERTIES
      ENVIRONMENT "LD_LIBRARY_PATH=${VLLM_ASCEND_TEST_LIBRARY_PATH};ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
      LABELS "benchmark;${VLLM_ASCEND_TEST_TIER_LABEL}"
      SKIP_RETURN_CODE 77
      # Wall clock, not device time: 1300+ launches per case across every shape.
      TIMEOUT 1800)
  endforeach()

  # The Cube-Hadamard sweep. It takes its own entry point rather than
  # ../common/bench_main_950pr.cpp because it accepts --csv=<path> on the
  # command line and writes a CSV by default: the sweep is 16 shapes by 4 legs
  # and is meant to be plotted, not read off a terminal. Everything else about
  # the run - the SoC check, the camodel refusal, the ASCEND_BENCH_* parses - is
  # the shared harness.
  vllm_ascend_add_device_executable(bench_950pr_cube_hadamard
                                    bench_950pr_cube_hadamard.cpp
                                    bench_main_950pr_hadamard.cpp)
  target_link_libraries(bench_950pr_cube_hadamard PRIVATE vllm_ascend_test_turboquant)
  target_compile_definitions(bench_950pr_cube_hadamard PRIVATE VLLM_ASCEND_TEST_TIER="device")
  add_dependencies(benchmarks bench_950pr_cube_hadamard)
  list(APPEND VLLM_ASCEND_DEVICE_BENCHMARKS bench_950pr_cube_hadamard)

  add_test(NAME bench_950pr_cube_hadamard COMMAND bench_950pr_cube_hadamard)
  set_tests_properties(bench_950pr_cube_hadamard PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${VLLM_ASCEND_TEST_LIBRARY_PATH};ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
    LABELS "benchmark;${VLLM_ASCEND_TEST_TIER_LABEL}"
    SKIP_RETURN_CODE 77
    # 16 shapes x 4 legs x (warmup + 100 timed iterations), all tiny kernels.
    TIMEOUT 1800)

  message(STATUS "vllm-ascend tests: [device] ${VLLM_ASCEND_DEVICE_BENCHMARKS}")
endif()

# --- profiling ----------------------------------------------------------------
#
# A trace harness, not a benchmark: per shape it launches the TurboQuant attention
# chain once and aclnnFusedInferAttentionScoreV5 once, each bracketed by mstx
# ranges, so an msprof timeline shows the launch gaps a median hides. It defines
# its own main() for --models= / --contexts= / --batches= / --mode=.
#
# Deliberately not registered with ctest. Without msprof attached it records
# nothing, so neither the correctness run nor the benchmark run has a use for
# it; it prints the msprof command that collects its trace. It does not depend
# on VLLM_ASCEND_TESTS_BUILD_BENCHMARKS for the same reason. TURBOQUANT_TESTS.md
# section 7.7.
if(VLLM_ASCEND_TESTS_HAVE_TURBOQUANT)
  vllm_ascend_add_device_executable(prof_device_950pr_msprof_trace prof_device_950pr_msprof_trace.cpp)
  target_link_libraries(prof_device_950pr_msprof_trace PRIVATE vllm_ascend_test_turboquant)
  message(STATUS "vllm-ascend tests: [device] prof_device_950pr_msprof_trace (not a ctest entry)")
endif()
