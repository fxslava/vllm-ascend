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
# The TurboQuant Ascend C kernels, compiled for the tests out of exactly the
# source the wheel builds: csrc/attention/turboquant/turboquant_kernels.cpp.
# There is no second copy and no test-only variant - a kernel the suite passes
# on and the kernel that ships have to be the same object code, or the suite is
# testing something nobody runs.
#
# This lives in its own directory rather than inline in ../../../CMakeLists.txt
# for two reasons, both of which are about containment:
#
#   * ascendc.cmake is not a well-behaved include. It sets CMAKE_SKIP_RPATH,
#     pulls in ExternalProject, and defines a dozen ASCENDC_* variables in
#     whatever scope includes it. Keeping it in a leaf directory keeps all of
#     that off the 310P and host-only targets, which must stay buildable with no
#     kernel toolchain at all.
#
#   * the generator override below has to be scoped, and a subdirectory is the
#     only scope CMake offers.
#
# -----------------------------------------------------------------------------

# ascendc_library() drives its compilation through four nested ExternalProjects,
# each configured with `-G ${CMAKE_GENERATOR}` - the *outer* project's generator.
# With Ninja, CMake writes the object path for an out-of-tree source as
#
#   CMakeFiles/preprocess_obj.dir/./work/csrc/attention/turboquant/...cpp.o
#
# and CANN's extract_host_stub.py looks that path up in compile_commands.json,
# where the same object is recorded without the "/./" - it normalises one and
# not the other. The build dies with a bare Python KeyError naming the object,
# after the kernel has already compiled cleanly, which is about as unhelpful as
# a build failure gets. The Makefiles generator emits the normalised path and
# the lookup succeeds.
#
# This is a CANN 9.1.0 toolchain bug, not something the kernel source can avoid:
# the source is out-of-tree for those sub-projects no matter where it lives,
# because their source directory is inside the CANN installation. Override it
# if a later CANN fixes the path handling.
set(VLLM_ASCEND_TESTS_ASCENDC_GENERATOR "Unix Makefiles"
    CACHE STRING "Generator for the ascendc_library sub-builds; see the note in this file before changing it")
set(CMAKE_GENERATOR "${VLLM_ASCEND_TESTS_ASCENDC_GENERATOR}")

set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})

# The same three-way lookup the repository root CMakeLists.txt does; which one
# exists depends on how the toolkit was laid out, not on its version.
if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
  set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
  message(FATAL_ERROR
    "ascendc_kernel_cmake not found under ${ASCEND_HOME_PATH}.\n"
    "The TurboQuant kernels need the Ascend C kernel toolchain. Install it, or configure with "
    "-DVLLM_ASCEND_TESTS_BUILD_TURBOQUANT_KERNELS=OFF to build the rest of the 950PR leg without them.")
endif()

message(STATUS "vllm-ascend tests: ascendc kernel cmake at ${ASCENDC_CMAKE_DIR}")
message(STATUS "vllm-ascend tests: TurboQuant kernels RUN_MODE=${RUN_MODE}, SOC_VERSION=${SOC_VERSION}")

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# Unquoted on purpose: VLLM_ASCEND_TURBOQUANT_KERNEL_SOURCE is a list of two
# sources, and quoting it would hand ascendc_library one semicolon-joined path.
ascendc_library(vllm_ascend_turboquant SHARED
  ${VLLM_ASCEND_TURBOQUANT_KERNEL_SOURCE})

# The decode ablation ladder's entry points, turboquant_mm_decode_ablation_*.
# Test-only: the root CMakeLists.txt does not set this, so the wheel's library
# carries the shipping kernels and nothing else. The shipping split is compiled
# identically either way -- it is the ladder's top rung, not a copy of it. See
# DecodeAblationStage in attention/turboquant/turboquant_mode.h.
#
# It takes two calls because ascendc_library compiles the source twice.
# ascendc_compile_definitions reaches the device precompile only -- alone, it
# builds the five kernels and their launchers but not the host-side
# turboquant_mm_decode_ablation_impl that calls them, and the benchmark fails to
# link. The host stub's flags come from host_intf_pub, which only
# ascendc_compile_options feeds, through -forward-options-to-host-compiler.
ascendc_compile_definitions(vllm_ascend_turboquant PRIVATE VLLM_ASCEND_TQ_DECODE_ABLATION)
ascendc_compile_options(vllm_ascend_turboquant PRIVATE
  -forward-options-to-host-compiler -DVLLM_ASCEND_TQ_DECODE_ABLATION)
