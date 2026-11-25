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

find_package(Torch REQUIRED)

run_python(TORCH_VERSION
  "import torch; print(torch.__version__)" "Failed to locate torch path")
# check torch version is 2.10.0
if(NOT ${TORCH_VERSION} VERSION_EQUAL "2.10.0")
  message(FATAL_ERROR "Expected PyTorch version 2.10.0, but found ${TORCH_VERSION}")
endif()

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
string(TOLOWER "${SOC_VERSION}" VLLM_ASCEND_SOC_FAMILY)
if(VLLM_ASCEND_SOC_FAMILY STREQUAL "ascend950pr_9599" OR VLLM_ASCEND_SOC_FAMILY STREQUAL "dav-3510")
    set(SOC_VERSION "Ascend950PR_9599")
    set(VLLM_ASCEND_SOC_FAMILY "ascend950pr_9599")
endif()
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

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# TurboQuant uses the dav-3510 ASC toolchain independently of the ACLNN package.
set(VLLM_ASCEND_BUILD_TURBOQUANT OFF)
set(VLLM_ASCEND_TURBOQUANT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/turboquant")
if(VLLM_ASCEND_SOC_FAMILY MATCHES "^ascend950")
    if(NOT EXISTS ${ASCENDC_CMAKE_DIR}/asc-config.cmake)
        message(FATAL_ERROR
            "ASC language toolchain not found at ${ASCENDC_CMAKE_DIR}/asc-config.cmake. "
            "An ascend950 build of the TurboQuant kernel needs CANN 9.2 or newer.")
    endif()
    set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "ASC --npu-arch target" FORCE)
    include(${ASCENDC_CMAKE_DIR}/asc-config.cmake)
    enable_language(ASC)
    message(STATUS "ASC language toolchain enabled: --npu-arch=${CMAKE_ASC_ARCHITECTURES}")

    set(VLLM_ASCEND_BUILD_TURBOQUANT ON)
    ascendc_library(vllm_ascend_turboquant SHARED
        ${VLLM_ASCEND_TURBOQUANT_SOURCE_DIR}/ingress/native/turboquant_reshape_batch.cpp
        ${VLLM_ASCEND_TURBOQUANT_SOURCE_DIR}/ingress/native/turboquant_reshape_scatter.cpp
        ${VLLM_ASCEND_TURBOQUANT_SOURCE_DIR}/decode/native/turboquant_fused_decode.cpp)
    ascendc_compile_definitions(vllm_ascend_turboquant PRIVATE
        ASCEND_MICRO_CANN_LAUNCH ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1 TURBOQUANT_INGRESS_NZ=1)
    ascendc_compile_options(vllm_ascend_turboquant PRIVATE
        -forward-options-to-host-compiler
        -DASCEND_MICRO_CANN_LAUNCH
        -DASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1
        -DTURBOQUANT_INGRESS_NZ=1)
    # E8 has its own packet ABI and cache-basis query contract. Keep its exact
    # floating-point flags independent of the native signed-nibble kernels.
    ascendc_library(vllm_ascend_turboquant_e8 SHARED
        ${VLLM_ASCEND_TURBOQUANT_SOURCE_DIR}/bindings/asc_e8_reader.cpp
        ${VLLM_ASCEND_TURBOQUANT_SOURCE_DIR}/bindings/asc_e8_component.cpp)
    ascendc_compile_definitions(vllm_ascend_turboquant_e8 PRIVATE
        ASCEND_MICRO_CANN_LAUNCH ENABLE_CV_COMM_VIA_SSBUF=1)
    ascendc_compile_options(vllm_ascend_turboquant_e8 PRIVATE
        -std=c++17 -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off
        -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH
        -DENABLE_CV_COMM_VIA_SSBUF=1)
endif()

file(GLOB KERNEL_FILES
${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/*.cpp)

set(VLLM_ASCEND_CUSTOM_OP
    ${KERNEL_FILES}
)

if(VLLM_ASCEND_SOC_FAMILY MATCHES "^ascend310p|^ascend950")
    message(STATUS "Hardware ${SOC_VERSION} detected: skip vllm_ascend_kernels compile")
else()
    ascendc_library(vllm_ascend_kernels SHARED
        ${VLLM_ASCEND_CUSTOM_OP}
    )
endif()

message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")

if(VLLM_ASCEND_SOC_FAMILY MATCHES "^ascend310p")
    file(GLOB VLLM_ASCEND_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_torch_adapter/*.cpp)
else()
    file(GLOB VLLM_ASCEND_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_torch_adapter/*.cpp)
endif()

if(VLLM_ASCEND_BUILD_TURBOQUANT)
    list(APPEND VLLM_ASCEND_SRC
        ${VLLM_ASCEND_TURBOQUANT_SOURCE_DIR}/bindings/torch_bindings.cpp
        ${VLLM_ASCEND_TURBOQUANT_SOURCE_DIR}/bindings/torch_ingress.cpp
        ${VLLM_ASCEND_TURBOQUANT_SOURCE_DIR}/bindings/torch_decode.cpp
        ${VLLM_ASCEND_TURBOQUANT_SOURCE_DIR}/common/turboquant_tiling.cpp)
endif()

# Sparse KV cache offloading is currently supported on Atlas A3 only.
if(VLLM_ASCEND_SOC_FAMILY MATCHES "^ascend910_93")
  find_package(OpenMP REQUIRED)
endif()

include_directories(
  ${pybind11_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${TORCH_INCLUDE_DIRS}
  # Prefer the selected CANN toolkit headers over the ACL snapshot bundled in
  # some torch_npu wheels. Feature checks below use ASCEND_HOME_PATH as well.
  ${ASCEND_HOME_PATH}/include
  ${TORCH_NPU_PATH}/include
)

set(
  INCLUDES
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_INCLUDE_DIRS}
  ${ASCEND_HOME_PATH}/include
)

pybind11_add_module(vllm_ascend_C ${VLLM_ASCEND_SRC})
if(VLLM_ASCEND_SOC_FAMILY MATCHES "^ascend910_93")
  target_compile_definitions(vllm_ascend_C PRIVATE VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD)
  target_link_libraries(vllm_ascend_C PRIVATE OpenMP::OpenMP_CXX)
endif()

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

# Keep builds on older CANN versions working; endpoint selection must fail
# explicitly if the installed CANN lacks the user-to-physical conversion.
check_cxx_source_compiles("
  #include <acl/acl_rt.h>
  int main() {
    int32_t physical_id = 0;
    return aclrtGetPhyDevIdByUserDevId(0, &physical_id);
  }
" HAVE_ACLRT_GET_PHY_DEV_ID_BY_USER_DEV_ID)
if(HAVE_ACLRT_GET_PHY_DEV_ID_BY_USER_DEV_ID)
  target_compile_definitions(vllm_ascend_C PRIVATE CANN_DEVICE_ID_MAPPING)
endif()

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

if(VLLM_ASCEND_SOC_FAMILY MATCHES "^ascend310p")
    target_compile_definitions(vllm_ascend_C PRIVATE -DASCEND_PLATFORM_310P)
endif()

if(NOT (VLLM_ASCEND_SOC_FAMILY MATCHES "^ascend310p|^ascend950"))
    target_compile_definitions(vllm_ascend_C PRIVATE -DVLLM_ENABLE_ATB_AND_DIRECT_KERNELS)
endif()

# The TurboQuant ops are now a translation unit of their own, added to
# VLLM_ASCEND_SRC above, so nothing in csrc/torch_binding.cpp is #ifdef'd on
# this any more. It stays as the flag Python-visible build introspection and any
# future host-side gate read.
if(VLLM_ASCEND_BUILD_TURBOQUANT)
    target_compile_definitions(vllm_ascend_C PRIVATE -DVLLM_ENABLE_TURBOQUANT)
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

if(VLLM_ASCEND_SOC_FAMILY MATCHES "^ascend310p|^ascend950")
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
  target_link_libraries(vllm_ascend_C PUBLIC vllm_ascend_turboquant vllm_ascend_turboquant_e8)
endif()

target_link_options(vllm_ascend_C PRIVATE "-Wl,-rpath,$ORIGIN:$ORIGIN/lib:$ORIGIN/_cann_ops_custom/vendors/custom_transformer/op_api/lib")

set(VLLM_ASCEND_NATIVE_TARGETS vllm_ascend_C)
if(TARGET vllm_ascend_kernels)
  list(APPEND VLLM_ASCEND_NATIVE_TARGETS vllm_ascend_kernels)
endif()
if(VLLM_ASCEND_BUILD_TURBOQUANT)
  list(APPEND VLLM_ASCEND_NATIVE_TARGETS vllm_ascend_turboquant vllm_ascend_turboquant_e8)
endif()
install(TARGETS ${VLLM_ASCEND_NATIVE_TARGETS}
        LIBRARY DESTINATION . COMPONENT vllm_ascend_native)
