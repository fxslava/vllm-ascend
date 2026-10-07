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
# libascend_moe_engine -- the standalone DeepSeek-V4 Flash decode engine.
#
# This project is fully self-contained: it never includes the repository
# root's CMakeLists.txt and references no Python paths. Configure it on its
# own, from this directory:
#
#   native against a CANN toolkit:
#     cmake -S csrc/standalone -B build/engine -DCMAKE_BUILD_TYPE=Release
#     cmake --build build/engine -j
#
#   host-only symbolic verification (no CANN, no Docker, no NPU):
#     cmake -S csrc/standalone -B build/mock -DENABLE_MOCK_RUNTIME=ON
#     cmake --build build/mock -j && ./build/mock/dsv4_mock_test
#
# TARGETS
#   ascend_moe_engine    libascend_moe_engine.so -- the engine itself
#   moe_runner           the inference binary
#   moe_contract_smoke   the device-free contract test
#   dsv4_mock_test       (mock mode) the zero-allocation symbolic suite
#   dsv4_verify_linkage  (CANN mode) zero undefined symbols, no Python/Torch
#
# AARCH64
# The 950PR host CPU is aarch64, and so is the production toolkit. Two ways in:
#
#   native on the device host:
#     cmake -S csrc/standalone -B build/engine
#
#   cross from an x86_64 build container, against <repo>/cann_aarch64:
#     cmake -S csrc/standalone -B build/engine-aarch64 \
#       -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/aarch64-toolchain.cmake
#
# BISHENG
# `-DDSV4_USE_BISHENG=ON` selects $ASCEND_HOME_PATH/bin/bisheng (CANN's
# Clang-based compiler) for the host C++ too. It is OFF by default because this
# tree contains no device code -- there is not a single Ascend C kernel here, by
# design -- so the host toolchain is a free choice and GCC is what every CI
# image has. The option exists so a BiSheng-only environment can build the same
# sources without editing this file.
# -----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

option(DSV4_USE_BISHENG "Compile the host C++ with CANN's BiSheng (clang) instead of the default compiler" OFF)
# ENABLE_MOCK_RUNTIME: build against the symbolic libopapi_mock instead of the
# CANN toolkit -- no device, no Docker, and no physical memory behind any
# weight, slot or arena.
option(ENABLE_MOCK_RUNTIME "Build against the zero-allocation symbolic mock runtime" OFF)

# The compiler has to be chosen before project(), so this block runs first.
if(DSV4_USE_BISHENG AND NOT CMAKE_CXX_COMPILER)
  set(_moe_toolkit "$ENV{ASCEND_HOME_PATH}")
  if(NOT _moe_toolkit)
    set(_moe_toolkit "$ENV{ASCEND_TOOLKIT_HOME}")
  endif()
  if(EXISTS "${_moe_toolkit}/bin/bisheng")
    set(CMAKE_CXX_COMPILER "${_moe_toolkit}/bin/bisheng" CACHE FILEPATH "BiSheng C++ compiler" FORCE)
    set(CMAKE_C_COMPILER "${_moe_toolkit}/bin/bisheng" CACHE FILEPATH "BiSheng C compiler" FORCE)
  else()
    message(FATAL_ERROR
      "DSV4_USE_BISHENG=ON but ${_moe_toolkit}/bin/bisheng does not exist. "
      "Source set_env.sh so ASCEND_HOME_PATH is set, or configure without this option.")
  endif()
endif()

project(ascend_moe_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

option(DSV4_WERROR "Treat compiler warnings as errors" OFF)

# -----------------------------------------------------------------------------
# The engine library
# -----------------------------------------------------------------------------
set(MOE_ENGINE_SOURCES
  src/device_ops.cpp
  src/expert_layout.cpp
  src/weight_source.cpp
  src/exclusive_staging.cpp
  src/op_table.cpp
  src/static_arena.cpp
  src/static_arena_manager.cpp
  src/moe_router_engine.cpp
  src/pipeline.cpp
)

add_library(ascend_moe_engine SHARED ${MOE_ENGINE_SOURCES})

target_include_directories(ascend_moe_engine PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_options(ascend_moe_engine PRIVATE -Wall -Wextra)
if(DSV4_WERROR)
  target_compile_options(ascend_moe_engine PRIVATE -Werror)
endif()

find_package(Threads REQUIRED)
target_link_libraries(ascend_moe_engine PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

# -----------------------------------------------------------------------------
# Backend selection
# -----------------------------------------------------------------------------
if(ENABLE_MOCK_RUNTIME)
  # --- the symbolic mock backend -------------------------------------------
  set(MOE_MOCK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mock")

  # The mock "toolkit" headers come FIRST, so <acl/acl.h> and <aclnn/acl_meta.h>
  # resolve to the mock declarations for every engine translation unit.
  target_include_directories(ascend_moe_engine PUBLIC "${MOE_MOCK_DIR}/include")

  add_library(opapi_mock SHARED
    "${MOE_MOCK_DIR}/mock_allocator.cpp"
    "${MOE_MOCK_DIR}/mock_descriptor.cpp"
    "${MOE_MOCK_DIR}/mock_ops.cpp"
  )
  target_include_directories(opapi_mock PUBLIC "${MOE_MOCK_DIR}" "${MOE_MOCK_DIR}/include")
  target_compile_options(opapi_mock PRIVATE -Wall -Wextra)
  target_link_libraries(opapi_mock PUBLIC Threads::Threads)
  target_link_libraries(ascend_moe_engine PUBLIC opapi_mock)

  # aclnn_signature_check.cpp static_asserts the hand-written prototypes
  # against the REAL CANN headers, which is meaningless against a mock and
  # needs the toolkit; it is the one source the mock build omits.
else()
  # --- the physical CANN backend --------------------------------------------
  # ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME is set by the toolkit's set_env.sh,
  # and the aarch64 toolchain file forces it at the aarch64 copy.
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
      "(expected ${ASCEND_HOME_PATH}/include/acl/acl.h). Source set_env.sh, pass "
      "-DASCEND_HOME_PATH=<toolkit root>, or configure with -DENABLE_MOCK_RUNTIME=ON.")
  endif()

  target_include_directories(ascend_moe_engine PUBLIC "${ASCEND_HOME_PATH}/include")

  # A toolkit root carries both an arch-qualified tree and a flat lib64. Prefer
  # the one that matches the target, so an x86_64 build host cannot accidentally
  # link the aarch64 copy or the reverse.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_moe_arch_dir "aarch64-linux")
  else()
    set(_moe_arch_dir "x86_64-linux")
  endif()

  set(MOE_CANN_LIB_DIRS "")
  foreach(_candidate
      "${ASCEND_HOME_PATH}/${_moe_arch_dir}/lib64"
      "${ASCEND_HOME_PATH}/lib64")
    if(EXISTS "${_candidate}/libascendcl.so")
      list(APPEND MOE_CANN_LIB_DIRS "${_candidate}")
    endif()
  endforeach()
  # Link-time driver stubs. libascendcl pulls in toolkit libraries whose own
  # DT_NEEDED reaches driver symbols (drvHdc*, hal*); the driver lives on the
  # device, and CANN ships stubs for it under devlib. Without these on the link
  # path the link fails with "undefined reference to drvHdcGetCapacity".
  foreach(_candidate
      "${ASCEND_HOME_PATH}/devlib/linux/${_moe_arch_dir}"
      "${ASCEND_HOME_PATH}/devlib")
    if(EXISTS "${_candidate}")
      list(APPEND MOE_CANN_DEVLIB_DIRS "${_candidate}")
    endif()
  endforeach()

  if(NOT MOE_CANN_LIB_DIRS)
    message(FATAL_ERROR
      "Found no libascendcl.so under ${ASCEND_HOME_PATH}/${_moe_arch_dir}/lib64 or "
      "${ASCEND_HOME_PATH}/lib64. This toolkit does not carry libraries for "
      "${CMAKE_SYSTEM_PROCESSOR}.")
  endif()
  list(GET MOE_CANN_LIB_DIRS 0 MOE_CANN_LIB_DIR)
  message(STATUS "ascend_moe_engine: CANN ${ASCEND_HOME_PATH}, libraries ${MOE_CANN_LIB_DIR}")

  # The libraries the brief names, plus libnnopbase (aclCreateTensor,
  # aclSetTensorAddr, aclSetAclOpExecutorRepeatable -- the static-arena
  # primitives) and libopapi_transformer, which is where the attention and
  # grouped-GEMM entry points actually live in the 9.2.0 split.
  set(_moe_required_libs ascendcl nnopbase opapi)
  set(_moe_optional_libs opapi_nn opapi_math opapi_transformer)

  set(MOE_CANN_LIBRARIES "")
  foreach(_name IN LISTS _moe_required_libs)
    find_library(_moe_lib_${_name} NAMES ${_name} PATHS ${MOE_CANN_LIB_DIRS} NO_DEFAULT_PATH)
    if(NOT _moe_lib_${_name})
      message(FATAL_ERROR "ascend_moe_engine: lib${_name}.so not found in ${MOE_CANN_LIB_DIRS}")
    endif()
    list(APPEND MOE_CANN_LIBRARIES "${_moe_lib_${_name}}")
  endforeach()
  foreach(_name IN LISTS _moe_optional_libs)
    find_library(_moe_lib_${_name} NAMES ${_name} PATHS ${MOE_CANN_LIB_DIRS} NO_DEFAULT_PATH)
    if(_moe_lib_${_name})
      list(APPEND MOE_CANN_LIBRARIES "${_moe_lib_${_name}}")
    else()
      message(STATUS "ascend_moe_engine: lib${_name}.so absent; its operators resolve from libopapi.so instead")
    endif()
  endforeach()
  target_link_libraries(ascend_moe_engine PUBLIC ${MOE_CANN_LIBRARIES})

  # --no-as-needed on the CANN libraries: the operator entry points are reached
  # with dlsym(RTLD_DEFAULT, ...) rather than by a direct call (see
  # op_table.hpp for why), so the linker would otherwise drop libopapi from
  # DT_NEEDED and RTLD_DEFAULT would find nothing at run time.
  target_link_options(ascend_moe_engine PUBLIC "LINKER:--no-as-needed")

  # The toolkit lib64 dirs go on -rpath-link too: the linker must walk the full
  # DT_NEEDED closure (libnnopbase -> libgraph_base -> libge_common_base ->
  # libascend_protobuf, libruntime -> liberror_manager, ...) to verify that every
  # symbol the CANN shared libraries reference is defined somewhere. Without it
  # the link fails with hundreds of undefined ge::/protobuf symbols that all live
  # in sibling lib64 .so files.
  foreach(_dir IN LISTS MOE_CANN_LIB_DIRS MOE_CANN_DEVLIB_DIRS)
    target_link_options(ascend_moe_engine PUBLIC "LINKER:-rpath-link,${_dir}")
  endforeach()

  target_sources(ascend_moe_engine PRIVATE src/aclnn_signature_check.cpp)

  set_target_properties(ascend_moe_engine PROPERTIES
    BUILD_RPATH "${MOE_CANN_LIB_DIR}"
    INSTALL_RPATH "${MOE_CANN_LIB_DIR}"
  )
endif()

# -----------------------------------------------------------------------------
# Executables and tests
# -----------------------------------------------------------------------------
add_executable(moe_runner src/main.cpp)
target_link_libraries(moe_runner PRIVATE ascend_moe_engine)
target_compile_options(moe_runner PRIVATE -Wall -Wextra)
if(DSV4_WERROR)
  target_compile_options(moe_runner PRIVATE -Werror)
endif()

add_executable(moe_contract_smoke src/contract_smoke.cpp)
target_link_libraries(moe_contract_smoke PRIVATE ascend_moe_engine)
target_compile_options(moe_contract_smoke PRIVATE -Wall -Wextra)
if(DSV4_WERROR)
  target_compile_options(moe_contract_smoke PRIVATE -Werror)
endif()

# $ORIGIN rpath so the executables find libascend_moe_engine.so (and, in mock
# mode, libopapi_mock.so) next to themselves; the dlsym(RTLD_DEFAULT, ...)
# operator table needs the libraries loaded and their symbols exported.
foreach(_target moe_runner moe_contract_smoke)
  set_target_properties(${_target} PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
  )
endforeach()

enable_testing()
add_test(NAME moe_contract_smoke COMMAND moe_contract_smoke)

if(ENABLE_MOCK_RUNTIME)
  add_executable(dsv4_mock_test "${MOE_MOCK_DIR}/dsv4_mock_test.cpp")
  target_link_libraries(dsv4_mock_test PRIVATE ascend_moe_engine)
  target_compile_options(dsv4_mock_test PRIVATE -Wall -Wextra)
  set_target_properties(dsv4_mock_test PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
  )
  add_test(NAME dsv4_mock_test COMMAND dsv4_mock_test)
else()
  # -----------------------------------------------------------------------------
  # Linkage verification (physical build)
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
            "$<TARGET_FILE:moe_runner>" "$<TARGET_FILE:moe_contract_smoke>"
    DEPENDS moe_runner moe_contract_smoke
    COMMENT "Verifying zero undefined symbols and no Python/Torch linkage"
    VERBATIM
  )
endif()
