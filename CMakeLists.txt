cmake_minimum_required(VERSION 3.16)
project(vllm_ascend_C)

# include(CheckCXXcompilerFlag)
# check_cxx_compiler_flag("-std=c++17", COMPILER_SUPPORTS_CXX17)
set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12")

find_package(pybind11 REQUIRED)

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
set(VLLM_ASCEND_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")

find_package(Torch REQUIRED)

run_python(TORCH_VERSION
  "import torch; print(torch.__version__)" "Failed to locate torch path")
# check torch version is 2.10.0
if(NOT ${TORCH_VERSION} VERSION_EQUAL "2.10.0")
  message(FATAL_ERROR "Expected PyTorch version 2.10.0, but found ${TORCH_VERSION}")
endif()

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION ${SOC_VERSION})
message(STATUS "Detected SOC version: ${SOC_VERSION}")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Build type Release/Debug (default Release)" FORCE)
endif()

if (CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRINGS "path to install()")
endif()

set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})
if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

# Ascend 950PR (A5) drives its AscendC kernels through the CANN 9.2 ASC language
# toolchain: asc-config.cmake registers ASC as a CMake language whose compiler is
# bisheng, and CMAKE_ASC_ARCHITECTURES becomes --npu-arch=. Every other part keeps
# the legacy ascendc.cmake path, which wraps each kernel in an ExternalProject and
# a generated host stub. The two headers define ascendc_library() differently, so
# exactly one of them is ever loaded.
if(SOC_VERSION MATCHES "ascend950")
    if(NOT EXISTS ${ASCENDC_CMAKE_DIR}/asc-config.cmake)
        message(FATAL_ERROR
            "ASC language toolchain not found at ${ASCENDC_CMAKE_DIR}/asc-config.cmake. "
            "An ascend950 build needs CANN 9.2 or newer.")
    endif()
    # CMakeDetermineASCCompiler.cmake sets ASCEND_CANN_PACKAGE_LINUX_PATH on the
    # first configure only, and FindASC.cmake reads it on every one, so pin it here.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
        set(ASCEND_CANN_PACKAGE_LINUX_PATH ${ASCEND_HOME_PATH}/aarch64-linux)
    else()
        set(ASCEND_CANN_PACKAGE_LINUX_PATH ${ASCEND_HOME_PATH}/x86_64-linux)
    endif()
    set(CMAKE_ASC_ARCHITECTURES "dav-3510")
    set(CUSTOM_ASC_CMAKE_PATH ${ASCENDC_CMAKE_DIR})
    include(${ASCENDC_CMAKE_DIR}/asc-config.cmake)
    enable_language(ASC)
    message(STATUS "ASC language toolchain enabled: --npu-arch=${CMAKE_ASC_ARCHITECTURES}")
else()
    include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
endif()

file(GLOB KERNEL_FILES
${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/*.cpp)

set(VLLM_ASCEND_CUSTOM_OP
    ${KERNEL_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp
)

set(VLLM_ASCEND_CUSTOM_OP_EXCLUDE_ASCEND950
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp
)

if(SOC_VERSION MATCHES "ascend950")
    message(STATUS "A5 hardware detected: disabling MLAPO operators")
    message(STATUS "A5 hardware detected: excluding batch_matmul_transpose operators")
    list(REMOVE_ITEM VLLM_ASCEND_CUSTOM_OP ${VLLM_ASCEND_CUSTOM_OP_EXCLUDE_ASCEND950})
endif()

if(SOC_VERSION MATCHES "ascend310p.*|ascend950")
    message(STATUS "Hardware ${SOC_VERSION} detected: skip vllm_ascend_kernels compile")
else()
    ascendc_library(vllm_ascend_kernels SHARED
        ${VLLM_ASCEND_CUSTOM_OP}
    )
endif()

# The TurboQuant 4-bit KV cache kernels. They live in their own ASC library
# because vllm_ascend_kernels is skipped on 950, and they are 950-only: the
# kv4fp8 Cube decode and its NZ-tiled cache writer need the arch35 Cube (fp8
# Mmad, dual-destination Fixpipe) that only dav-3510 has.
#
# One library for every source. A <<<>>> launch resolves straight to the
# kernel's own __global__ definition under the ASC toolchain, so each launcher
# has to be compiled in the same translation unit as the kernel it launches --
# which is how they are written -- and the adapter reaches all of them through
# this one .so.
if(SOC_VERSION MATCHES "ascend950")
    set(VLLM_ASCEND_BUILD_TURBOQUANT ON)
    set(VLLM_ASCEND_BUILD_TURBOQUANT_CUBE ON)
    message(STATUS "Hardware ${SOC_VERSION} detected: building TurboQuant KV cache kernels")
    set(VLLM_ASCEND_TURBOQUANT_KERNEL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/turboquant/op_kernel/turboquant_paged_attention.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/turboquant/op_kernel/turboquant_rotate_q.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/turboquant/op_kernel/turboquant_fused_decode.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/turboquant/op_kernel/turboquant_fused_decode_opt.cpp
    )
    # ascendc_library() installs the target itself, into CMAKE_INSTALL_LIBDIR.
    # Point that at the directory the extension already installs into so the
    # wheel carries one copy of the .so rather than a second one under lib/.
    set(CMAKE_INSTALL_LIBDIR ".")
    ascendc_library(vllm_ascend_turboquant SHARED ${VLLM_ASCEND_TURBOQUANT_KERNEL_SOURCES})
else()
    set(VLLM_ASCEND_BUILD_TURBOQUANT OFF)
    set(VLLM_ASCEND_BUILD_TURBOQUANT_CUBE OFF)
    message(STATUS "Hardware ${SOC_VERSION} detected: skip TurboQuant KV cache kernels")
endif()

message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")

if(SOC_VERSION MATCHES "ascend310p.*")
    file(GLOB VLLM_ASCEND_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_torch_adapter/*.cpp)
else()
    file(GLOB VLLM_ASCEND_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_torch_adapter/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/batch_matmul_transpose/op_host/tiling/tiling_data.cpp)
endif()

# The TurboQuant adapter's host tiling (op_host/): plain C++, no device code, so
# it is compiled by the host compiler into the extension rather than by bisheng.
if(VLLM_ASCEND_BUILD_TURBOQUANT)
    list(APPEND VLLM_ASCEND_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/turboquant/op_host/turboquant_tiling.cpp)
endif()

include_directories(
  ${pybind11_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ASCEND_HOME_PATH}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/batch_matmul_transpose/op_host
)

set(
  INCLUDES
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_INCLUDE_DIRS}
  ${ASCEND_HOME_PATH}/include
)

pybind11_add_module(vllm_ascend_C ${VLLM_ASCEND_SRC})

# Detect aclrtMemcpyBatchAsync availability (CANN 8.5+)
# Can be overridden via VLLM_ASCEND_ENABLE_BATCH_MEMCPY env var (registered
# in vllm_ascend/envs.py, forwarded by setup.py as a CMake variable):
#   VLLM_ASCEND_ENABLE_BATCH_MEMCPY=1  -> force enable
#   VLLM_ASCEND_ENABLE_BATCH_MEMCPY=0  -> force disable
#   unset                               -> auto-detect from CANN headers
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${ASCEND_HOME_PATH}/include)
set(CMAKE_REQUIRED_LIBRARIES ascendcl)
set(CMAKE_REQUIRED_LINK_OPTIONS "-L${ASCEND_HOME_PATH}/lib64")

if(DEFINED VLLM_ASCEND_ENABLE_BATCH_MEMCPY)
  if("${VLLM_ASCEND_ENABLE_BATCH_MEMCPY}" STREQUAL "1")
    message(STATUS "aclrtMemcpyBatchAsync: force enabled via VLLM_ASCEND_ENABLE_BATCH_MEMCPY=1")
    target_compile_definitions(vllm_ascend_C PRIVATE CANN_MEMCPY_BATCH_ASYNC)
  else()
    message(STATUS "aclrtMemcpyBatchAsync: force disabled via VLLM_ASCEND_ENABLE_BATCH_MEMCPY=0")
  endif()
else()
  # Test the full code pattern we actually use, including struct member access.
  # This ensures the macro is only defined when the API is fully compatible.
  check_cxx_source_compiles("
    #include <acl/acl_rt.h>
    int main() {
      aclrtMemLocation loc = {};
      loc.type = ACL_MEM_LOCATION_TYPE_HOST;
      loc.id = 0;
      aclrtMemcpyBatchAttr attr = {};
      attr.srcLoc = loc;
      attr.dstLoc = loc;
      (void)aclrtMemcpyBatchAsync;
      return 0;
    }
  " HAVE_ACLRT_MEMCPY_BATCH_ASYNC)
  if(HAVE_ACLRT_MEMCPY_BATCH_ASYNC)
    message(STATUS "aclrtMemcpyBatchAsync: detected in CANN headers, enabling batch memcpy path")
    target_compile_definitions(vllm_ascend_C PRIVATE CANN_MEMCPY_BATCH_ASYNC)
  else()
    message(STATUS "aclrtMemcpyBatchAsync: not found in CANN headers, using fallback aclrtMemcpyAsync loop")
  endif()
endif()

if(SOC_VERSION MATCHES "ascend310p.*")
    target_compile_definitions(vllm_ascend_C PRIVATE -DASCEND_PLATFORM_310P)
endif()

if(VLLM_ASCEND_BUILD_TURBOQUANT)
    target_compile_definitions(vllm_ascend_C PRIVATE -DVLLM_ENABLE_TURBOQUANT)
endif()

if(VLLM_ASCEND_BUILD_TURBOQUANT_CUBE)
    target_compile_definitions(vllm_ascend_C PRIVATE -DVLLM_ENABLE_TURBOQUANT_CUBE)
endif()

if(NOT (SOC_VERSION MATCHES "ascend310p.*|ascend950"))
    target_compile_definitions(vllm_ascend_C PRIVATE -DVLLM_ENABLE_ATB_AND_DIRECT_KERNELS)
endif()

target_link_directories(
  vllm_ascend_C
  PRIVATE
  ${TORCH_LIBRARY_DIRS}
  ${TORCH_NPU_PATH}/lib/
  ${ASCEND_HOME_PATH}/lib64
)

set(VLLM_ASCEND_C_COMMON_LIBS
  ${TORCH_LIBRARIES}
  torch_npu
  ascendcl
  tiling_api
  register
  platform
  ascendalog
  dl
  opapi
)

if(SOC_VERSION MATCHES "ascend310p.*|ascend950")
  target_link_libraries(
    vllm_ascend_C
    PUBLIC
    ${VLLM_ASCEND_C_COMMON_LIBS}
  )
else()
  target_link_libraries(
    vllm_ascend_C
    PUBLIC
    vllm_ascend_kernels
    ${VLLM_ASCEND_C_COMMON_LIBS}
  )
endif()

if(VLLM_ASCEND_BUILD_TURBOQUANT)
  target_link_libraries(vllm_ascend_C PUBLIC vllm_ascend_turboquant)
endif()

target_link_options(vllm_ascend_C PRIVATE "-Wl,-rpath,$ORIGIN:$ORIGIN/lib:$ORIGIN/_cann_ops_custom/vendors/custom_transformer/op_api/lib")

if(SOC_VERSION MATCHES "ascend310p.*|ascend950")
  install(TARGETS vllm_ascend_C DESTINATION ${VLLM_ASCEND_INSTALL_PATH})
else()
  install(TARGETS vllm_ascend_C vllm_ascend_kernels DESTINATION ${VLLM_ASCEND_INSTALL_PATH})
endif()

if(VLLM_ASCEND_BUILD_TURBOQUANT)
  install(TARGETS vllm_ascend_turboquant DESTINATION ${VLLM_ASCEND_INSTALL_PATH})
endif()
