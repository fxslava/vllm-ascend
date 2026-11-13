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

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# -----------------------------------------------------------------------------
# TurboQuant BF16 ingestion and fused decode: Ascend 950PR / arch35 / dav-3510 only.
#
# The kernel is a MIX_AIC_1_2 launch that puts one Cube core and BOTH Vector
# cores of a block in one four-stage pipeline, and it depends on facilities only
# dav-3510 has: a Fixpipe that picks a sub-block's Unified Buffer as its
# destination, mode-4 intra-block flags between the Cube and each Vector core,
# and a 16-bit Cube fractal with C0 = 16. There is no 910B form of it, so the
# library is simply absent from every other build, and so is
# csrc/attention/turboquant/torch_bindings.cpp -- so torch.ops.turboquant_ops
# does not exist there rather than existing with nothing behind it. That absence
# is the Python side's fallback signal: the adapter in
# vllm_ascend/attention/turboquant_dry_run.py warns once and runs its golden
# model when a stage is switched on and its op is unregistered.
#
# CANN 9.2 drives these through the ASC language toolchain: asc-config.cmake
# registers ASC as a CMake language whose compiler is bisheng, and
# CMAKE_ASC_ARCHITECTURES becomes --npu-arch=. FindASC.cmake re-reads that
# variable on every configure while CMake would otherwise set it on the first
# only, so it is pinned in the cache here.
set(VLLM_ASCEND_BUILD_TURBOQUANT OFF)
if(SOC_VERSION MATCHES "ascend950")
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
    # The three audited kernel translation units, mirrored from
    # kernels/ops/turboquant/vllm_integration_layer of the ascend-micro-harness.
    # The ASC toolchain compiles each source's device kernels and its host
    # launcher together -- one bisheng invocation per source runs a host pass and
    # two device passes over the same text -- so this one library carries both
    # halves of all three paths: the prefill batch writer, the decode scatter
    # writer and the fused raw-query decode. torch_bindings.cpp below links
    # against these launchers.
    #
    # Single-source is also why the sources keep their `__global__` entry-point
    # *definitions* outside `#if defined(__CCE_AICORE__)`, with only the bodies
    # inside it: the host-side launch stub of a `__global__` function is emitted
    # by the host pass from the definition, and the legacy ascendc_library's
    # generated host_stub.cpp -- which is what supplies those stubs in the
    # harness's own build -- does not exist on this path. With the definitions
    # hidden from the host pass this library still links, with every launch_*
    # entry present and every kernel it calls undefined. See
    # csrc/attention/turboquant/README.md, "Why the entry points are outside the
    # device guard".
    #
    # One library, not three: the fused decode's Split-K reducer and the two
    # writers share op_kernel/common, and ascendc_library has no incremental
    # rebuild (its preprocess step feeds ld.lld its own previous output), so a
    # source change means one clean rebuild either way.
    ascendc_library(vllm_ascend_turboquant SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/turboquant/op_kernel/turboquant_reshape_batch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/turboquant/op_kernel/turboquant_reshape_scatter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/turboquant/op_kernel/turboquant_fused_decode.cpp
    )
    # ASCEND_MICRO_CANN_LAUNCH is NOT optional. Each of the three sources is
    #
    #     #if defined(__CCE_AICORE__)      ... the device kernels ...
    #     #elif defined(ASCEND_MICRO_CANN_LAUNCH)   ... the host launchers ...
    #     #endif
    #
    # so the host compilation of the TU, which is the one that defines
    # launch_turboquant_reshape_and_cache_bf16,
    # launch_turboquant_reshape_scatter_aiv and
    # turboquant_mm_fused_decode_opt_raw_query_impl, compiles to NOTHING past the
    # entry points without it. The library then still looks complete -- every
    # device kernel is there, and under the legacy module set so is every
    # generated aclrtlaunch_* stub, whose names contain the launcher names as
    # substrings -- and torch_bindings.cpp fails to link with an undefined
    # reference. Taken from the harness's own standalone/CMakeLists.txt, which is
    # the authority on these flags.
    #
    # ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE drives the decode's reduction with the
    # Cube GEMM reducer (op_kernel/cube/turboquant_gemm_reduce_service.h) rather
    # than TurboQuantPartialReducer. ON in the harness build and in its
    # standalone project alike, so a wheel build compiles the same operator the
    # harness verifies under that file name. The other two sources never
    # reference it.
    #
    # TWO calls, and only ONE of them may be ascendc_compile_options: in CANN's
    # legacy module set the device half APPENDS its options while the host half
    # ASSIGNS them, so a second ascendc_compile_options on this target would
    # silently discard the first call's -forward-options-to-host-compiler
    # defines. Both defines therefore go in this one call.
    #
    # On the ASC path the second call is redundant rather than required:
    # ascendc_compile_definitions is a plain target_compile_definitions there, so
    # its -D reaches every pass including the host one. It stays because it costs
    # nothing and it is what makes this block work unchanged if a standalone gate
    # builds the same sources through the legacy ascendc_library, where the
    # host-forwarded copy IS the only one the host pass sees.
    # TURBOQUANT_INGRESS_NZ=1 turns the prefill batch writer into the NZ-tiled
    # ingress: it emits the [64-row tile][kv head][32-byte column group][tile
    # row][byte] cache the fused decode reads (NzTiledPackedByte in
    # op_kernel/common/turboquant_layout.h) instead of the RFC 3.1 strip cache,
    # with the kv4fp8 nibble pairing and the slot-major scale plane. This is the
    # fix for the writer/reader cache-format mismatch: the prefill writer and
    # the decode now name the same byte. turboquant-core builds the same source
    # twice (tq_prefill_npu / tq_prefill_nz_npu) so both variants ship; this
    # repo wants only the NZ one, so the define is target-wide here. It is
    # inert for the other two sources: turboquant_reshape_scatter.cpp and
    # turboquant_fused_decode.cpp never reference it. The scatter writer has NO
    # NZ variant in turboquant-core and still writes the strip cache -- see
    # csrc/attention/turboquant/README.md, "The scatter writer still does not
    # meet the decode".
    ascendc_compile_definitions(vllm_ascend_turboquant PRIVATE
        ASCEND_MICRO_CANN_LAUNCH ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1 TURBOQUANT_INGRESS_NZ=1)
    ascendc_compile_options(vllm_ascend_turboquant PRIVATE
        -forward-options-to-host-compiler
        -DASCEND_MICRO_CANN_LAUNCH
        -DASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1
        -DTURBOQUANT_INGRESS_NZ=1)
endif()

file(GLOB KERNEL_FILES
${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/*.cpp)

set(VLLM_ASCEND_CUSTOM_OP
    ${KERNEL_FILES}
)

if(SOC_VERSION MATCHES "ascend310p.*|ascend950")
    message(STATUS "Hardware ${SOC_VERSION} detected: skip vllm_ascend_kernels compile")
else()
    ascendc_library(vllm_ascend_kernels SHARED
        ${VLLM_ASCEND_CUSTOM_OP}
    )
endif()

message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")

if(SOC_VERSION MATCHES "ascend310p.*")
    file(GLOB VLLM_ASCEND_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_torch_adapter/*.cpp)
else()
    file(GLOB VLLM_ASCEND_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_torch_adapter/*.cpp)
endif()

# The TurboQuant Torch bindings are a translation unit of their own and the glob
# above only reaches csrc/*.cpp, so they are named here. They must NOT be folded
# into csrc/torch_binding.cpp: the audited
# op_kernel/common/types.h and csrc/kernels/types.h both declare
# vllm_ascend::AscendType behind their own #pragma once, so a TU that reaches
# both fails to compile. Adding the file only on the ascend950 branch is also
# what keeps torch.ops.turboquant_ops absent -- rather than registered with
# nothing behind it -- on every other SoC.
#
# op_host/turboquant_tiling.cpp joins it. The decode's host tiling used to be an
# `inline` function in a device-side header, which meant every translation unit
# that wanted the geometry also pulled in the kernel tree; it is an ordinary
# host C++ definition now, so the binding, the kernel launcher and the
# device-free gate in tests/ut/ops all link the SAME object instead of each
# re-instantiating it. It compiles with the plain host compiler -- no CANN, no
# ASC -- which is exactly why the gate can build it on a host with no toolkit.
if(VLLM_ASCEND_BUILD_TURBOQUANT)
    list(APPEND VLLM_ASCEND_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/turboquant/torch_bindings.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/turboquant/op_host/turboquant_tiling.cpp)
endif()

# Sparse KV cache offloading is currently supported on Atlas A3 only.
if(SOC_VERSION MATCHES "ascend910_93.*")
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
if(SOC_VERSION MATCHES "ascend910_93.*")
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

if(SOC_VERSION MATCHES "ascend310p.*")
    target_compile_definitions(vllm_ascend_C PRIVATE -DASCEND_PLATFORM_310P)
endif()

if(NOT (SOC_VERSION MATCHES "ascend310p.*|ascend950"))
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
