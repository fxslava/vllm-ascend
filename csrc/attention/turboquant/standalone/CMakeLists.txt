# Standalone build of the TurboQuant operators for Ascend 950: the three kernel sources
# vllm_ascend_C's TurboQuant library is made of, the host tiling, and a binding that
# registers the same torch.ops._C_ascend schemas -- without configuring, compiling or
# linking anything else of the package.
#
# Produces, directly under CMAKE_INSTALL_PREFIX (cmake --install --component vllm_turboquant):
#   libvllm_turboquant_cube.so          load this: torch.ops.load_library(...)
#   libvllm_turboquant_cube_kernels.so  the Ascend C kernels it launches (found via $ORIGIN)
#
# Driven by tools/tq_longbench/build_turboquant_ops.py, which fills in the variables below.
# Two ascendc_library() constraints it also enforces:
#   * the "Unix Makefiles" generator: under Ninja, extract_host_stub.py dies with a KeyError;
#   * a clean build directory every time: the preprocess step links foo.o onto itself, so
#     a second build over the same tree fails with "unknown file type".

cmake_minimum_required(VERSION 3.16)
project(vllm_turboquant_standalone LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

foreach(required SOC_VERSION ASCEND_HOME_PATH TORCH_NPU_PATH)
  if(NOT ${required})
    message(FATAL_ERROR "${required} is not set; configure through tools/tq_longbench/build_turboquant_ops.py")
  endif()
endforeach()

string(TOLOWER "${SOC_VERSION}" VLLM_TQ_SOC_LOWER)
if(NOT VLLM_TQ_SOC_LOWER MATCHES "^ascend950")
  message(FATAL_ERROR "the TurboQuant Cube decode is Ascend 950 only (arch35 fp8 Mmad), got SOC_VERSION=${SOC_VERSION}")
endif()

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})
if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
  set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
  message(FATAL_ERROR "ascendc_kernel_cmake is missing under ${ASCEND_HOME_PATH}; is the CANN toolkit installed?")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

get_filename_component(VLLM_TQ_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
get_filename_component(VLLM_CSRC_DIR "${VLLM_TQ_DIR}/../.." ABSOLUTE)

# The same sources as the root CMakeLists.txt's vllm_ascend_turboquant on 950. One
# library: a <<<>>> call site only reaches kernels whose launchers share it.
ascendc_library(vllm_turboquant_cube_kernels SHARED
  ${VLLM_TQ_DIR}/op_kernel/turboquant_paged_attention.cpp
  ${VLLM_TQ_DIR}/op_kernel/turboquant_rotate_q.cpp
  ${VLLM_TQ_DIR}/op_kernel/turboquant_fused_decode.cpp
)

find_package(Torch REQUIRED)

add_library(vllm_turboquant_cube SHARED
  ${CMAKE_CURRENT_LIST_DIR}/turboquant_standalone_binding.cpp
  ${VLLM_TQ_DIR}/op_host/turboquant_tiling.cpp
)
target_compile_definitions(vllm_turboquant_cube PRIVATE VLLM_ENABLE_TURBOQUANT VLLM_ENABLE_TURBOQUANT_CUBE=1)
target_compile_options(vllm_turboquant_cube PRIVATE -O3)
target_include_directories(vllm_turboquant_cube PRIVATE
  ${VLLM_CSRC_DIR}
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ASCEND_HOME_PATH}/include
)
target_link_directories(vllm_turboquant_cube PRIVATE
  ${TORCH_NPU_PATH}/lib
  ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(vllm_turboquant_cube PRIVATE
  vllm_turboquant_cube_kernels
  ${TORCH_LIBRARIES}
  torch_npu
  ascendcl
)

# $ORIGIN finds the kernel library installed beside it; the absolute paths let a
# bare torch.ops.load_library() resolve torch and torch_npu on the host that built it.
#
# Through the RPATH properties, not a -Wl,-rpath link option. Under the Makefiles
# generator CMake 3.22 writes a "$ORIGIN" link option into the binary as "$$ORIGIN"
# -- a literal directory that never matches, so the kernel library "cannot be found"
# beside the binding -- while CMake 4.x keeps it; quoting it ('$ORIGIN') leaves the
# quotes in DT_RUNPATH on both. The properties come out as $ORIGIN on both
# (readelf, 3.22.1 and 4.4.3). They only apply once CMAKE_SKIP_RPATH is off again:
# CANN's host_config.cmake and bisheng_config.cmake set it TRUE for the whole
# directory, which silently discards every RPATH property.
set(CMAKE_SKIP_RPATH FALSE)
get_filename_component(VLLM_TQ_TORCH_LIB_DIR "${TORCH_LIBRARY}" DIRECTORY)
set(VLLM_TQ_RPATH "$ORIGIN" "${TORCH_NPU_PATH}/lib" "${VLLM_TQ_TORCH_LIB_DIR}" "${ASCEND_HOME_PATH}/lib64")
set_target_properties(vllm_turboquant_cube PROPERTIES
  BUILD_RPATH "${VLLM_TQ_RPATH}"
  INSTALL_RPATH "${VLLM_TQ_RPATH}"
)

# Straight into CMAKE_INSTALL_PREFIX, under a component of its own: ascendc_library()
# adds install rules too (COMPONENT asc-devkit: the kernel library into lib/, its
# aclrtlaunch_*.h into include/), which would nest a lib/lib in tools/tq_longbench/lib.
# The build script installs --component vllm_turboquant alone.
install(TARGETS vllm_turboquant_cube vllm_turboquant_cube_kernels
  LIBRARY DESTINATION . COMPONENT vllm_turboquant
)
