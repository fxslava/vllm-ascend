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
# The Ascend 310P (v200) leg of tier 3. A different part, not a tier.
#
# NOT CONFIGURED AT ALL when SOC_VERSION names a 950PR: neither the root
# CMakeLists.txt nor device/CMakeLists.txt calls add_subdirectory() on this
# file, so none of these targets exists, none of these sources is compiled, and
# `ctest -N` in a 950PR tree cannot list a `*310*` binary because there is not
# one to list.
#
# The binary names are unchanged from before the restructure - test_matmul_310p,
# bench_matmul_310p and so on - and that is deliberate rather than an oversight:
#
#   * .agents/skills/ascend-310p-csrc-tests/SKILL.md drives them by name,
#   * the shipped dist/ascend310p3-aarch64-tests/run_tests.sh iterates over them
#     by name,
#   * they already carry their part in the name, so nothing is ambiguous, and
#   * they are unreachable from a 950PR build, so nothing can collide.
#
# What the isolation needed was a directory of their own and a configure-time
# gate. A rename would have bought nothing and broken two external consumers.
# The source FILES, unlike the binaries, drop the redundant _310p suffix
# (correctness/test_matmul.cpp); correctness/ and bench/ split the leg by
# purpose the same way the 950PR leg does.
#
# Everything here encodes v200 rules that are simply false on a 950PR - the 5-D
# FRACTAL_NZ KV cache, the 32-element SwiGLU gate, rotary head dims limited to
# 64 and 128 - which is the substantive reason the two legs cannot share a
# directory.
# -----------------------------------------------------------------------------

set(VLLM_ASCEND_TEST_TIER_LABEL "device_310p")

# Mirrors the root build's convention for v200 sources. Nothing in this suite
# reads it today - csrc/torch_binding.cpp does, and that is not built here - but
# it is scoped to this leg rather than to the shared library so a 950PR tree
# never carries a define that claims it is a 310P.
add_compile_definitions(ASCEND_PLATFORM_310P)

# <binary name>:<source file under this directory, without the extension>.
set(VLLM_ASCEND_310P_TESTS
  test_matmul_310p:correctness/test_matmul
  test_rmsnorm_310p:correctness/test_rmsnorm
  test_rotary_embedding_310p:correctness/test_rotary_embedding
  test_activation_swiglu_310p:correctness/test_activation_swiglu
  test_paged_attention_310p:correctness/test_paged_attention)

foreach(entry ${VLLM_ASCEND_310P_TESTS})
  string(REPLACE ":" ";" name_and_source "${entry}")
  list(GET name_and_source 0 test_name)
  list(GET name_and_source 1 source_name)
  vllm_ascend_add_device_executable(${test_name} "${source_name}.cpp" ../../common/main.cpp)
  vllm_ascend_register_correctness_test(${test_name})
endforeach()

message(STATUS "vllm-ascend tests: [device_310p] ${VLLM_ASCEND_310P_TESTS}")

if(VLLM_ASCEND_TESTS_BUILD_BENCHMARKS)
  set(VLLM_ASCEND_310P_BENCHMARKS
    bench_matmul_310p:bench/bench_matmul
    bench_rmsnorm_310p:bench/bench_rmsnorm
    bench_rotary_embedding_310p:bench/bench_rotary_embedding
    bench_activation_swiglu_310p:bench/bench_activation_swiglu
    bench_paged_attention_310p:bench/bench_paged_attention)

  add_custom_target(benchmarks COMMENT "Building the vllm-ascend 310P microbenchmarks")

  foreach(entry ${VLLM_ASCEND_310P_BENCHMARKS})
    string(REPLACE ":" ";" name_and_source "${entry}")
    list(GET name_and_source 0 bench_name)
    list(GET name_and_source 1 source_name)
    vllm_ascend_add_device_executable(${bench_name} "${source_name}.cpp" ../../common/bench_main.cpp)
    add_dependencies(benchmarks ${bench_name})

    add_test(NAME ${bench_name} COMMAND ${bench_name})
    set_tests_properties(${bench_name} PROPERTIES
      ENVIRONMENT "LD_LIBRARY_PATH=${VLLM_ASCEND_TEST_LIBRARY_PATH};ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
      LABELS "benchmark;${VLLM_ASCEND_TEST_TIER_LABEL}"
      SKIP_RETURN_CODE 77
      # Wall clock, not device time: 1300+ launches per case across every shape
      # in a suite, and the largest matmul sweep is the slow one.
      TIMEOUT 1800)
  endforeach()

  message(STATUS "vllm-ascend tests: [device_310p] ${VLLM_ASCEND_310P_BENCHMARKS}")
endif()
