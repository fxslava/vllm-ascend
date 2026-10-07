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
# dsv4_runner -- standalone DeepSeek-V4 Flash decode for Ascend 950PR.
#
# Like csrc/tests, this project is deliberately NOT included from the repository
# root CMakeLists.txt: the root needs Torch and pybind11, and the entire point
# of this tree is a binary with neither. Configure it on its own:
#
#   cmake -S csrc/standalone -B build/dsv4-runner -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/dsv4-runner -j
#
# TARGETS
#   dsv4_runner          the inference binary
#   dsv4_contract_smoke  the device-free contract test
#   dsv4_verify_linkage  a custom target that asserts zero undefined symbols and
#                        no libpython / libtorch in either binary (see below)
#
# AARCH64
# The 950PR host CPU is aarch64, and so is the production toolkit. Two ways in:
#
#   native on the device host:
#     cmake -S csrc/standalone -B build/dsv4-runner
#
#   cross from an x86_64 build container, against <repo>/cann_aarch64:
#     cmake -S csrc/standalone -B build/dsv4-runner-aarch64 \
#       -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/aarch64-toolchain.cmake
#
# The toolchain file in cmake/ already points ASCEND_HOME_PATH at the aarch64
# toolkit and adds its devlib stubs to -rpath-link, so nothing here is
# arch-specific beyond picking the right lib64 directory.
#
# BISHENG
# `-DDSV4_USE_BISHENG=ON` selects $ASCEND_HOME_PATH/bin/bisheng (CANN's
# Clang-based compiler) for the host C++ too. It is OFF by default because this
# tree contains no device code -- there is not a single Ascend C kernel here, by
# design (see README) -- so the host toolchain is a free choice and GCC is what
# every CI image has. The option exists so a BiSheng-only environment can build
# the same sources without editing this file.
# -----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

option(DSV4_USE_BISHENG "Compile the host C++ with CANN's BiSheng (clang) instead of the default compiler" OFF)
# ENABLE_MOCK_RUNTIME: build the zero-NPU symbolic mock runtime (libopapi_mock)
# plus the dsv4_mock_runner / dsv4_mock_test targets -- no CANN toolkit, no
# Docker, and no physical memory behind any weight, slot or arena.
option(ENABLE_MOCK_RUNTIME "Build the zero-NPU symbolic mock runtime and mock test targets" OFF)

# The compiler has to be chosen before project(), so this block runs first.
if(DSV4_USE_BISHENG AND NOT CMAKE_CXX_COMPILER)
  set(_dsv4_toolkit "$ENV{ASCEND_HOME_PATH}")
  if(NOT _dsv4_toolkit)
    set(_dsv4_toolkit "$ENV{ASCEND_TOOLKIT_HOME}")
  endif()
  if(EXISTS "${_dsv4_toolkit}/bin/bisheng")
    set(CMAKE_CXX_COMPILER "${_dsv4_toolkit}/bin/bisheng" CACHE FILEPATH "BiSheng C++ compiler" FORCE)
    set(CMAKE_C_COMPILER "${_dsv4_toolkit}/bin/bisheng" CACHE FILEPATH "BiSheng C compiler" FORCE)
  else()
    message(FATAL_ERROR
      "DSV4_USE_BISHENG=ON but ${_dsv4_toolkit}/bin/bisheng does not exist. "
      "Source set_env.sh so ASCEND_HOME_PATH is set, or configure without this option.")
  endif()
endif()

project(dsv4_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

option(DSV4_WERROR "Treat compiler warnings as errors" OFF)

# -----------------------------------------------------------------------------
# The CANN toolkit
# -----------------------------------------------------------------------------
# Everything from here to the linkage-verification block is the physical
# (device) build; the mock build skips all of it and jumps to the
# ENABLE_MOCK_RUNTIME section at the bottom.
if(NOT ENABLE_MOCK_RUNTIME)
# ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME is set by the toolkit's set_env.sh, and
# the aarch64 toolchain file forces it at the aarch64 copy.
if(NOT ASCEND_HOME_PATH)
  if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
  elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_HOME_PATH "$ENV{ASCEND_TOOLKIT_HOME}")
  else()
    set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
endif()
set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit install root")

if(NOT EXISTS "${ASCEND_HOME_PATH}/include/acl/acl.h")
  message(FATAL_ERROR
    "No CANN toolkit at ASCEND_HOME_PATH=${ASCEND_HOME_PATH} "
    "(expected ${ASCEND_HOME_PATH}/include/acl/acl.h). Source set_env.sh, or pass "
    "-DASCEND_HOME_PATH=<toolkit root>.")
endif()

# A toolkit root carries both an arch-qualified tree and a flat lib64. Prefer the
# one that matches the target, so an x86_64 build host cannot accidentally link
# the aarch64 copy or the reverse.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(_dsv4_arch_dir "aarch64-linux")
else()
  set(_dsv4_arch_dir "x86_64-linux")
endif()

set(DSV4_CANN_LIB_DIRS "")
foreach(_candidate
    "${ASCEND_HOME_PATH}/${_dsv4_arch_dir}/lib64"
    "${ASCEND_HOME_PATH}/lib64")
  if(EXISTS "${_candidate}/libascendcl.so")
    list(APPEND DSV4_CANN_LIB_DIRS "${_candidate}")
  endif()
endforeach()
# Link-time driver stubs. libascendcl pulls in toolkit libraries whose own
# DT_NEEDED reaches driver symbols (drvHdc*, hal*); the driver lives on the
# device, and CANN ships stubs for it under devlib. Without these on the link
# path the link fails with "undefined reference to drvHdcGetCapacity".
foreach(_candidate
    "${ASCEND_HOME_PATH}/devlib/linux/${_dsv4_arch_dir}"
    "${ASCEND_HOME_PATH}/devlib")
  if(EXISTS "${_candidate}")
    list(APPEND DSV4_CANN_DEVLIB_DIRS "${_candidate}")
  endif()
endforeach()

if(NOT DSV4_CANN_LIB_DIRS)
  message(FATAL_ERROR
    "Found no libascendcl.so under ${ASCEND_HOME_PATH}/${_dsv4_arch_dir}/lib64 or "
    "${ASCEND_HOME_PATH}/lib64. This toolkit does not carry libraries for "
    "${CMAKE_SYSTEM_PROCESSOR}.")
endif()
list(GET DSV4_CANN_LIB_DIRS 0 DSV4_CANN_LIB_DIR)
message(STATUS "dsv4_runner: CANN ${ASCEND_HOME_PATH}, libraries ${DSV4_CANN_LIB_DIR}")

# The libraries the brief names, plus libnnopbase (aclCreateTensor,
# aclSetTensorAddr, aclSetAclOpExecutorRepeatable -- the static-arena
# primitives) and libopapi_transformer, which is where the attention and
# grouped-GEMM entry points actually live in the 9.2.0 split.
set(_dsv4_required_libs ascendcl nnopbase opapi)
set(_dsv4_optional_libs opapi_nn opapi_math opapi_transformer)

set(DSV4_CANN_LIBRARIES "")
foreach(_name IN LISTS _dsv4_required_libs)
  find_library(_dsv4_lib_${_name} NAMES ${_name} PATHS ${DSV4_CANN_LIB_DIRS} NO_DEFAULT_PATH)
  if(NOT _dsv4_lib_${_name})
    message(FATAL_ERROR "dsv4_runner: lib${_name}.so not found in ${DSV4_CANN_LIB_DIRS}")
  endif()
  list(APPEND DSV4_CANN_LIBRARIES "${_dsv4_lib_${_name}}")
endforeach()
foreach(_name IN LISTS _dsv4_optional_libs)
  find_library(_dsv4_lib_${_name} NAMES ${_name} PATHS ${DSV4_CANN_LIB_DIRS} NO_DEFAULT_PATH)
  if(_dsv4_lib_${_name})
    list(APPEND DSV4_CANN_LIBRARIES "${_dsv4_lib_${_name}}")
  else()
    message(STATUS "dsv4_runner: lib${_name}.so absent; its operators resolve from libopapi.so instead")
  endif()
endforeach()

# -----------------------------------------------------------------------------
# The library every target shares
# -----------------------------------------------------------------------------

add_library(dsv4_core STATIC
  dsv4_device_ops.cpp
  dsv4_expert_layout.cpp
  dsv4_weight_source.cpp
  dsv4_exclusive_staging.cpp
  dsv4_aclnn_v5.cpp
  dsv4_aclnn_signature_check.cpp
  dsv4_static_arena.cpp
  dsv4_pipeline.cpp
)

target_include_directories(dsv4_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${ASCEND_HOME_PATH}/include"
)

target_link_libraries(dsv4_core PUBLIC ${DSV4_CANN_LIBRARIES} ${CMAKE_DL_LIBS})

# --no-as-needed on the CANN libraries: the operator entry points are reached
# with dlsym(RTLD_DEFAULT, ...) rather than by a direct call (see
# dsv4_aclnn_v5.hpp for why), so the linker would otherwise drop libopapi from
# DT_NEEDED and RTLD_DEFAULT would find nothing at run time.
target_link_options(dsv4_core PUBLIC "LINKER:--no-as-needed")

target_compile_options(dsv4_core PRIVATE -Wall -Wextra)
if(DSV4_WERROR)
  target_compile_options(dsv4_core PRIVATE -Werror)
endif()

# The toolkit lib64 dirs go on -rpath-link too: the linker must walk the full
# DT_NEEDED closure (libnnopbase -> libgraph_base -> libge_common_base ->
# libascend_protobuf, libruntime -> liberror_manager, ...) to verify that every
# symbol the CANN shared libraries reference is defined somewhere. Without it
# the link fails with hundreds of undefined ge::/protobuf symbols that all live
# in sibling lib64 .so files.
foreach(_dir IN LISTS DSV4_CANN_LIB_DIRS DSV4_CANN_DEVLIB_DIRS)
  target_link_options(dsv4_core PUBLIC "LINKER:-rpath-link,${_dir}")
endforeach()

# -----------------------------------------------------------------------------
# Binaries
# -----------------------------------------------------------------------------

add_executable(dsv4_runner main.cpp)
target_link_libraries(dsv4_runner PRIVATE dsv4_core)

add_executable(dsv4_contract_smoke contract_smoke.cpp)
target_link_libraries(dsv4_contract_smoke PRIVATE dsv4_core)

foreach(_target dsv4_runner dsv4_contract_smoke)
  set_target_properties(${_target} PROPERTIES
    BUILD_RPATH "${DSV4_CANN_LIB_DIR}"
    INSTALL_RPATH "${DSV4_CANN_LIB_DIR}"
  )
  target_compile_options(${_target} PRIVATE -Wall -Wextra)
  if(DSV4_WERROR)
    target_compile_options(${_target} PRIVATE -Werror)
  endif()
endforeach()

# -----------------------------------------------------------------------------
# Linkage verification
# -----------------------------------------------------------------------------
# Three properties the brief asks to be verified, checked by the build rather
# than by hand:
#
#   1. zero undefined symbols  -- `ldd -r` reports none, so every symbol this
#      binary calls directly is satisfied by its DT_NEEDED set. (The dlsym'd
#      operator entry points are deliberately NOT in that set; they are resolved
#      at run time out of the same libraries, and the operator table reports any
#      that a given toolkit does not export.)
#   2. no libpython, no libtorch anywhere in the dependency closure.
#   3. libopapi survived --as-needed, so RTLD_DEFAULT can see the operators.
configure_file(verify_linkage.sh.in "${CMAKE_CURRENT_BINARY_DIR}/verify_linkage.sh" @ONLY)

add_custom_target(dsv4_verify_linkage
  COMMAND ${CMAKE_COMMAND} -E env bash "${CMAKE_CURRENT_BINARY_DIR}/verify_linkage.sh"
          "$<TARGET_FILE:dsv4_runner>" "$<TARGET_FILE:dsv4_contract_smoke>"
  DEPENDS dsv4_runner dsv4_contract_smoke
  COMMENT "Verifying zero undefined symbols and no Python/Torch linkage"
  VERBATIM
)

enable_testing()
add_test(NAME dsv4_contract_smoke COMMAND dsv4_contract_smoke)

endif()  # NOT ENABLE_MOCK_RUNTIME (physical build)

# -----------------------------------------------------------------------------
# The symbolic mock runtime (ENABLE_MOCK_RUNTIME=ON)
# -----------------------------------------------------------------------------
# A second consumer of the same dsv4 sources that links libopapi_mock instead
# of the CANN toolkit: no device, no Docker, and no physical memory behind any
# weight, slot or arena -- the mock allocator hands out fake 4096-aligned
# addresses tracked in an interval registry, every DMA and aclSetTensorAddr is
# validated against it, and the aclnn GetWorkspaceSize stubs assert the
# DeepSeek-V4 Flash shape/dtype/flag contracts. Compile and run natively on an
# x86 WSL host.
if(ENABLE_MOCK_RUNTIME)
  set(DSV4_MOCK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mock_runtime")

  # The mock "toolkit" headers come FIRST, so <acl/acl.h> and <aclnn/acl_meta.h>
  # resolve to the mock declarations for every dsv4 translation unit.
  add_library(opapi_mock SHARED
    "${DSV4_MOCK_DIR}/mock_allocator.cpp"
    "${DSV4_MOCK_DIR}/mock_descriptor.cpp"
    "${DSV4_MOCK_DIR}/mock_ops.cpp"
  )
  target_include_directories(opapi_mock PUBLIC "${DSV4_MOCK_DIR}" "${DSV4_MOCK_DIR}/include")
  target_compile_options(opapi_mock PRIVATE -Wall -Wextra)
  find_package(Threads REQUIRED)
  target_link_libraries(opapi_mock PUBLIC Threads::Threads)

  # The dsv4 core against the mock headers. dsv4_aclnn_signature_check.cpp is
  # excluded: it static_asserts the hand-written prototypes against the REAL
  # CANN headers, which is meaningless against a mock and needs the toolkit.
  add_library(dsv4_core_mock STATIC
    dsv4_device_ops.cpp
    dsv4_expert_layout.cpp
    dsv4_weight_source.cpp
    dsv4_exclusive_staging.cpp
    dsv4_aclnn_v5.cpp
    dsv4_static_arena.cpp
    dsv4_pipeline.cpp
  )
  target_include_directories(dsv4_core_mock PUBLIC
    "${DSV4_MOCK_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(dsv4_core_mock PUBLIC opapi_mock ${CMAKE_DL_LIBS})
  target_compile_options(dsv4_core_mock PRIVATE -Wall -Wextra)

  add_executable(dsv4_mock_runner main.cpp)
  target_link_libraries(dsv4_mock_runner PRIVATE dsv4_core_mock)
  target_compile_options(dsv4_mock_runner PRIVATE -Wall -Wextra)

  add_executable(dsv4_mock_test "${DSV4_MOCK_DIR}/dsv4_mock_test.cpp")
  target_link_libraries(dsv4_mock_test PRIVATE dsv4_core_mock)
  target_compile_options(dsv4_mock_test PRIVATE -Wall -Wextra)

  # $ORIGIN rpath so the test finds libopapi_mock.so next to itself; the
  # dlsym(RTLD_DEFAULT, ...) operator table needs the mock loaded and its
  # symbols exported.
  foreach(_target dsv4_mock_runner dsv4_mock_test)
    set_target_properties(${_target} PROPERTIES
      BUILD_RPATH "$ORIGIN"
      INSTALL_RPATH "$ORIGIN"
    )
  endforeach()

  add_test(NAME dsv4_mock_test COMMAND dsv4_mock_test)
endif()
