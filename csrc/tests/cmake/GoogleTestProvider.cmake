# SPDX-License-Identifier: Apache-2.0
include(FetchContent)
set(gtest_source "")
if(FETCHCONTENT_SOURCE_DIR_GOOGLETEST)
  if(NOT EXISTS "${FETCHCONTENT_SOURCE_DIR_GOOGLETEST}/googletest/CMakeLists.txt")
    message(FATAL_ERROR "Invalid offline GoogleTest source: ${FETCHCONTENT_SOURCE_DIR_GOOGLETEST}. "
      "Point FETCHCONTENT_SOURCE_DIR_GOOGLETEST at the unpacked repository root.")
  endif()
  set(gtest_source "${FETCHCONTENT_SOURCE_DIR_GOOGLETEST}")
else()
  foreach(candidate "$ENV{HOME}/googletest" "/tmp/googletest"
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest" "${CMAKE_BINARY_DIR}/_deps/googletest-src"
      "${CMAKE_BINARY_DIR}/_deps/googletest-download")
    if(EXISTS "${candidate}/googletest/CMakeLists.txt")
      set(gtest_source "${candidate}")
      break()
    endif()
  endforeach()
endif()
if(NOT gtest_source)
  find_package(GTest QUIET)
  if(TARGET GTest::gtest AND TARGET GTest::gtest_main)
    set(VLLM_ASCEND_TEST_GTEST_TARGET GTest::gtest)
    set(VLLM_ASCEND_TEST_GTEST_MAIN_TARGET GTest::gtest_main)
    return()
  endif()
endif()
if(NOT gtest_source)
  if(NOT VLLM_ASCEND_TESTS_FETCH_GTEST OR FETCHCONTENT_FULLY_DISCONNECTED)
    message(FATAL_ERROR "GoogleTest is unavailable offline. Install GTest or pass "
      "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest; also checked ~/googletest, "
      "/tmp/googletest and third_party/googletest. Network fetching is disabled.")
  endif()
  find_package(Git QUIET)
  if(NOT Git_FOUND)
    message(FATAL_ERROR "GoogleTest download needs git. Supply FETCHCONTENT_SOURCE_DIR_GOOGLETEST for offline builds.")
  endif()
  set(gtest_source "${CMAKE_BINARY_DIR}/_deps/googletest-download")
  execute_process(COMMAND "${GIT_EXECUTABLE}" clone --depth 1 --branch v1.14.0
    https://github.com/google/googletest.git "${gtest_source}"
    RESULT_VARIABLE download_result ERROR_VARIABLE download_error TIMEOUT 120)
  if(NOT download_result STREQUAL "0")
    message(FATAL_ERROR "GoogleTest download failed: ${download_error}\n"
      "For air-gapped builds pass -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest "
      "and -DVLLM_ASCEND_TESTS_FETCH_GTEST=OFF.")
  endif()
endif()
message(STATUS "vllm-ascend tests: local GoogleTest at ${gtest_source}")
FetchContent_Declare(googletest SOURCE_DIR "${gtest_source}")
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
set(VLLM_ASCEND_TEST_GTEST_TARGET gtest)
set(VLLM_ASCEND_TEST_GTEST_MAIN_TARGET gtest_main)
