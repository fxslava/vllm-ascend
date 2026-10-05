# SPDX-License-Identifier: Apache-2.0
# An explicit cache path is authoritative; otherwise try environment roots in order.
set(cann_candidates "${ASCEND_HOME_PATH}" "$ENV{ASCEND_TOOLKIT_HOME}"
    "$ENV{ASCEND_HOME_PATH}" "/usr/local/Ascend/ascend-toolkit/latest")
set(cann_checked "")
set(cann_root "")
foreach(candidate IN LISTS cann_candidates)
  if(candidate STREQUAL "")
    continue()
  endif()
  list(APPEND cann_checked "${candidate}/include/acl/acl.h")
  if(EXISTS "${candidate}/include/acl/acl.h")
    set(cann_root "${candidate}")
    break()
  endif()
  if(DEFINED ASCEND_HOME_PATH AND NOT ASCEND_HOME_PATH STREQUAL "")
    message(FATAL_ERROR "Explicit ASCEND_HOME_PATH is invalid. Checked: ${cann_checked}")
  endif()
endforeach()
if(cann_root STREQUAL "")
  message(FATAL_ERROR "CANN toolkit not found. Checked (in precedence order): ${cann_checked}\n"
    "Set -DASCEND_HOME_PATH=/path/to/toolkit or source the CANN set_env.sh.")
endif()
set(ASCEND_HOME_PATH "${cann_root}" CACHE PATH "CANN toolkit install root" FORCE)
if(DEFINED VLLM_ASCEND_RESOLVED_TOOLKIT AND NOT VLLM_ASCEND_RESOLVED_TOOLKIT STREQUAL cann_root)
  foreach(library ASCEND_CL_LIBRARY ASCEND_OP_COMPILER_LIBRARY ASCEND_NNOPBASE_LIBRARY
      ASCEND_OPAPI_LIBRARY ASCEND_RUNTIME_LIBRARY)
    unset(${library} CACHE)
  endforeach()
endif()
set(VLLM_ASCEND_RESOLVED_TOOLKIT "${cann_root}" CACHE INTERNAL "Previously resolved toolkit")

# Link-time stubs never enter the runtime RPATH. Prefer the installed driver.
set(VLLM_ASCEND_DRIVER_LINK_DIRS "")
foreach(candidate "/usr/local/Ascend/driver/lib64/driver" "/usr/local/Ascend/driver/lib64/common"
    "${ASCEND_HOME_PATH}/lib64/stub" "${ASCEND_HOME_PATH}/lib64/stub/linux/${CMAKE_SYSTEM_PROCESSOR}"
    "${ASCEND_HOME_PATH}/lib64/stub/${CMAKE_SYSTEM_PROCESSOR}"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/linux/${CMAKE_SYSTEM_PROCESSOR}"
    "${ASCEND_HOME_PATH}/devlib/linux/${CMAKE_SYSTEM_PROCESSOR}")
  if(IS_DIRECTORY "${candidate}")
    list(APPEND VLLM_ASCEND_DRIVER_LINK_DIRS "${candidate}")
  endif()
endforeach()
