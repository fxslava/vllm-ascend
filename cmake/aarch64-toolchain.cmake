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
# CMake toolchain file for cross-compiling to aarch64 (Ascend 310P3 host CPU)
# from an x86_64 build container.
#
# The CANN toolkit shipped in the x86_64 development image only carries
# x86_64-linux libraries, so a cross build needs a second, aarch64 copy of the
# toolkit. That copy is expected at `cann_aarch64/` in the repository root,
# laid out as a normal toolkit root:
#
#   cann_aarch64/
#     include/acl/acl.h ...
#     lib64/libascendcl.so ...
#
# Usage:
#
#   cmake -S csrc/tests -B build/csrc-tests-aarch64 -G Ninja \
#     -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/aarch64-toolchain.cmake \
#     -DSOC_VERSION=ascend310p3
#
# `CANN_AARCH64_ROOT` may be overridden to point at a toolkit copy kept outside
# the source tree.
# -----------------------------------------------------------------------------

include_guard(GLOBAL)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# -----------------------------------------------------------------------------
# Cross compilers
# -----------------------------------------------------------------------------
# From the Debian/Ubuntu cross toolchain packages:
#   apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
#                      libc6-dev-arm64-cross
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Nothing in this tree is built for the build host, so the usual "compile and
# run a probe" check cannot work. Link-only is enough to validate the toolchain.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# -----------------------------------------------------------------------------
# aarch64 CANN toolkit
# -----------------------------------------------------------------------------
# Defaults to <repo>/cann_aarch64, which inside the development container is
# /vllm-workspace/vllm-ascend/cann_aarch64.
if(NOT DEFINED CANN_AARCH64_ROOT OR CANN_AARCH64_ROOT STREQUAL "")
  get_filename_component(CANN_AARCH64_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../cann_aarch64" ABSOLUTE)
endif()
set(CANN_AARCH64_ROOT "${CANN_AARCH64_ROOT}" CACHE PATH
  "aarch64 CANN toolkit root used for cross compilation")

set(CANN_AARCH64_INCLUDE_DIR "${CANN_AARCH64_ROOT}/include")
set(CANN_AARCH64_LIB_DIR     "${CANN_AARCH64_ROOT}/lib64")

if(NOT EXISTS "${CANN_AARCH64_INCLUDE_DIR}/acl/acl.h")
  message(FATAL_ERROR
    "aarch64 CANN toolkit not found at CANN_AARCH64_ROOT=${CANN_AARCH64_ROOT}\n"
    "Expected ${CANN_AARCH64_INCLUDE_DIR}/acl/acl.h\n"
    "Populate it by extracting aarch64-linux from the arm64 CANN image, or "
    "point -DCANN_AARCH64_ROOT at an existing copy.")
endif()

# Consumers of this toolchain (e.g. csrc/tests) look the toolkit up through
# ASCEND_HOME_PATH; point it at the aarch64 copy so they do not fall back to
# the x86_64 toolkit installed under /usr/local/Ascend.
set(ASCEND_HOME_PATH "${CANN_AARCH64_ROOT}" CACHE PATH
  "CANN toolkit install root" FORCE)

include_directories(SYSTEM "${CANN_AARCH64_INCLUDE_DIR}")
link_directories("${CANN_AARCH64_LIB_DIR}")

# -----------------------------------------------------------------------------
# Search path policy
# -----------------------------------------------------------------------------
# Look for headers and libraries in the cross sysroot and the aarch64 toolkit
# only, so an x86_64 library from the host never satisfies a find_library().
# Programs are still taken from the host: those run on the build machine.
set(CMAKE_FIND_ROOT_PATH
  "/usr/aarch64-linux-gnu"
  "${CANN_AARCH64_ROOT}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# libascendcl.so pulls in further toolkit libraries through DT_NEEDED, and those
# in turn need driver symbols (drvHdc*, hal*). The driver itself is not part of
# the toolkit -- it lives on the device -- but CANN ships link-time stubs for it
# under devlib/. Both directories must be searchable or the link fails with
# "undefined reference to drvHdcGetCapacity" and friends.
set(CANN_AARCH64_DEVLIB_DIR "${CANN_AARCH64_ROOT}/devlib/linux/aarch64")
set(_cann_rpath_link "${CANN_AARCH64_LIB_DIR}:${CANN_AARCH64_DEVLIB_DIR}")

# -rpath-link only, deliberately: it steers link-time resolution and is not
# itself recorded in the output ELF, so the devlib stubs can never shadow the
# real driver at runtime. On the 310P3 the loader finds the real driver and
# toolkit through set_env.sh / LD_LIBRARY_PATH.
#
# Note that CMake still records a RUNPATH of its own: linking a library by
# absolute path makes it add that library's directory to the build-tree RPATH.
# The result is a target binary carrying a build-host path
# (<repo>/cann_aarch64/lib64) that does not exist on the device. It is harmless
# there -- the loader falls through to LD_LIBRARY_PATH -- but to strip it,
# configure with -DCMAKE_SKIP_BUILD_RPATH=ON, or set INSTALL_RPATH on the target
# and build with CMAKE_BUILD_WITH_INSTALL_RPATH. It is left alone here so this
# file does not silently override the RPATH policy a consuming project sets
# (csrc/tests, for one, sets BUILD_RPATH and INSTALL_RPATH explicitly).
set(CMAKE_EXE_LINKER_FLAGS_INIT    "-Wl,-rpath-link,${_cann_rpath_link}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-Wl,-rpath-link,${_cann_rpath_link}")
