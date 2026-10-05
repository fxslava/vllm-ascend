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
# EVERY Ascend C kernel library the suite builds, compiled out of exactly the
# sources the wheel builds - csrc/attention/turboquant/op_kernel/ for
# TurboQuant, csrc/moe/dsv4_moe_expert/op_kernel/ for the DSV4 expert. There is
# no second copy and no test-only variant of either: a kernel the suite passes
# on and the kernel that ships have to be the same object code, or the suite is
# testing something nobody runs.
#
# Everything that compiles device code lives in this one directory, for three
# reasons:
#
#   * ascendc.cmake CANNOT be included twice in one project. Its
#     legacy_modules/host_intf.cmake creates `host_intf_pub` and
#     bisheng_intf.cmake creates `device_intf_pub`, `c310_aiv_intf_pub` and
#     others, all with bare unguarded add_library() calls. A second
#     subdirectory that included it would fail at configure time with "another
#     target with the same name already exists". ascendc_library() itself is
#     safe to call repeatedly: every target it creates derives from the target
#     name it is given.
#
#   * ascendc.cmake is not a well-behaved include in general. It sets
#     CMAKE_SKIP_RPATH, pulls in ExternalProject, and defines a dozen ASCENDC_*
#     variables in whatever scope includes it. Keeping it in a leaf directory
#     keeps all of that off the 310P and host-only targets, which must stay
#     buildable with no kernel toolchain at all.
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
find_program(VLLM_ASCEND_DEVICE_COMPILER NAMES bisheng ccec
  PATHS "${ASCEND_HOME_PATH}/tools/ccec_compiler/bin"
        "${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin"
        "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin" NO_DEFAULT_PATH)
if(NOT VLLM_ASCEND_DEVICE_COMPILER)
  message(FATAL_ERROR "Ascend C compiler (bisheng/ccec) not found under ${ASCEND_HOME_PATH}/tools or compiler. Install the CANN compiler package.")
endif()
get_filename_component(ascend_compiler_bin "${VLLM_ASCEND_DEVICE_COMPILER}" DIRECTORY)
set(ENV{PATH} "${ascend_compiler_bin}:$ENV{PATH}")
message(STATUS "vllm-ascend tests: device compiler ${VLLM_ASCEND_DEVICE_COMPILER}")

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
    "DSV4 and TurboQuant kernels need the Ascend C compiler package. Install it, or disable both kernel build options.")
endif()

message(STATUS "vllm-ascend tests: ascendc kernel cmake at ${ASCENDC_CMAKE_DIR}")
message(STATUS "vllm-ascend tests: Ascend C RUN_MODE=${RUN_MODE}, SOC_VERSION=${SOC_VERSION}")

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# Unquoted on purpose: VLLM_ASCEND_TURBOQUANT_KERNEL_SOURCE is a list of
# sources, and quoting it would hand ascendc_library one semicolon-joined path.
if(VLLM_ASCEND_TESTS_BUILD_TURBOQUANT_KERNELS)
ascendc_library(vllm_ascend_turboquant SHARED
  ${VLLM_ASCEND_TURBOQUANT_KERNEL_SOURCE})

# The Cube kernels only the suite launches: the kv3fp4 / kv5fp8 fused decode and cache write, and the fp8
# GEMM probe (op_kernel/turboquant_fused_decode.cpp). The root CMakeLists.txt does not set this, so the
# wheel ships the kv4fp8 kernels alone; those compile identically either way.
#
# It takes two calls because ascendc_library compiles the source twice. ascendc_compile_definitions reaches
# the device precompile only -- alone, it builds the kernels and their launchers but not the host-side
# *_impl functions that call them, and the tests fail to link. The host stub's flags come from
# host_intf_pub, which only ascendc_compile_options feeds, through -forward-options-to-host-compiler.
ascendc_compile_definitions(vllm_ascend_turboquant PRIVATE VLLM_ASCEND_TQ_TEST_KERNELS)
ascendc_compile_options(vllm_ascend_turboquant PRIVATE
  --cce-auto-sync=off -Werror
  -forward-options-to-host-compiler -DVLLM_ASCEND_TQ_TEST_KERNELS)
endif()

# ---------------------------------------------------------------------------
# The DSV4 MoE expert kernel
# ---------------------------------------------------------------------------
#
# One source, not two. common/dsv4_moe_expert_kernels.cpp includes the shipping
# kernel verbatim and defines the <<<>>> launcher beside it; listing the kernel
# separately as well would make the launcher generator emit two stubs for one
# kernel name and the merge link would fail on an undefined
# `dsv4_moe_expert_origin`. The header comment in that file has the detail.
if(VLLM_ASCEND_TESTS_BUILD_DSV4_KERNELS)
  ascendc_library(vllm_ascend_dsv4_moe_expert SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/dsv4_moe_expert_kernels.cpp")
  ascendc_compile_options(vllm_ascend_dsv4_moe_expert PRIVATE --cce-auto-sync=off -Werror)
  message(STATUS "vllm-ascend tests: DSV4 MoE expert kernel RUN_MODE=${RUN_MODE}, SOC_VERSION=${SOC_VERSION}")
endif()

# Test-owned E2M1/E8M0 hybrid probe; never enters a production/device build.
if(VLLM_ASCEND_TESTS_TIER_SIM AND SOC_VERSION STREQUAL "Ascend950PR_9599")
  ascendc_library(vllm_ascend_moe_unpack_probe SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/../sim/moe_unpack_probe_kernels.cpp")
  # CANN 9.2 recognizes __simt_vf__ in Ascend C hybrid language mode.
  # -fsimt is not a supported flag in this installed toolchain.
  # The legacy preprocess linker overwrites its compiler object with raw text.
  # An empty generated include refreshed once per build forces that object to
  # be recreated before relinking, without touching any source file.
  set(probe_epoch "${CMAKE_CURRENT_BINARY_DIR}/moe_unpack_probe_epoch.h")
  file(WRITE "${probe_epoch}" "// Generated build dependency.\n")
  add_custom_target(moe_unpack_probe_epoch
    COMMAND "${CMAKE_COMMAND}" -E touch "${probe_epoch}")
  add_dependencies(vllm_ascend_moe_unpack_probe_precompile moe_unpack_probe_epoch)
  ascendc_compile_options(vllm_ascend_moe_unpack_probe PRIVATE
    --cce-auto-sync=off -Werror -include "${probe_epoch}")
endif()

# Ascend C's ExternalProject compilations do not inherit the parent directory's
# definitions. Feed both device precompilation and the host launcher compiler.
foreach(kernel_target vllm_ascend_dsv4_moe_expert vllm_ascend_turboquant vllm_ascend_moe_unpack_probe)
  if(TARGET ${kernel_target})
    ascendc_compile_definitions(${kernel_target} PRIVATE CANN_VERSION_MAJOR=${CANN_VERSION_MAJOR})
    ascendc_compile_options(${kernel_target} PRIVATE
      -forward-options-to-host-compiler -DCANN_VERSION_MAJOR=${CANN_VERSION_MAJOR})
  endif()
endforeach()
