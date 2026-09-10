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
# TIER 1 - host. Pure CPU.
#
# The entry criterion is absolute and is what the tier is for: a binary here
# links neither libascendcl.so nor any NPU runtime, needs no toolkit, no driver
# and no device, and runs on a laptop and in CI. Everything it exercises is the
# algorithm - the codec's arithmetic, the layout arithmetic, the golden dumps -
# rather than a kernel.
#
# Exit criterion: every case passes. There is no "skips because no hardware"
# state here, which is exactly why this tier is the one CI can gate on. The one
# conditional skip is a golden dump that was never fetched from Git LFS, and it
# names `git lfs pull` when it happens.
#
# These targets are built in EVERY configuration, including
# -DVLLM_ASCEND_TESTS_HOST_ONLY=ON, and are the only ones that survive it.
# -----------------------------------------------------------------------------

set(VLLM_ASCEND_TEST_TIER_LABEL "host")

# Analytical fidelity report for the TurboQuant 4-bit KV cache: the Lloyd-Max
# grid and its decision boundaries, the Pi involution and orthogonality, the
# nibble packing, the poison/bounds discipline of every codec entry point, and
# the codec's measured fidelity on the real Qwen3.5 layer-3 activations.
#
# It links gtest_main rather than one of the suite's main()s: those initialise
# the ACL runtime, which is exactly the dependency this tier must not have.
add_executable(test_host_turboquant_fidelity test_host_turboquant_fidelity.cpp)

target_link_libraries(test_host_turboquant_fidelity
  PRIVATE vllm_ascend_test_host ${VLLM_ASCEND_TEST_GTEST_MAIN_TARGET})

add_test(NAME test_host_turboquant_fidelity COMMAND test_host_turboquant_fidelity)
set_tests_properties(test_host_turboquant_fidelity PROPERTIES
  LABELS "correctness;${VLLM_ASCEND_TEST_TIER_LABEL}"
  TIMEOUT 300)

vllm_ascend_record_binary(test_host_turboquant_fidelity)

message(STATUS "vllm-ascend tests: [host] test_host_turboquant_fidelity")
