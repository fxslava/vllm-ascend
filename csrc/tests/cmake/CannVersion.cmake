# SPDX-License-Identifier: Apache-2.0
# Keep automatic detection out of the cache so switching toolkits cannot reuse
# a previously detected major. A user-provided cache value is authoritative.
if(DEFINED CANN_VERSION_MAJOR AND NOT CANN_VERSION_MAJOR STREQUAL "")
  if(NOT CANN_VERSION_MAJOR MATCHES "^[1-9][0-9]*$" OR CANN_VERSION_MAJOR LESS 8)
    message(FATAL_ERROR "CANN_VERSION_MAJOR must be an integer >= 8, got '${CANN_VERSION_MAJOR}'")
  endif()
  message(STATUS "vllm-ascend tests: CANN major ${CANN_VERSION_MAJOR} (manual override)")
  return()
endif()

set(cann_version_checked "")
file(GLOB cann_install_info "${ASCEND_HOME_PATH}/*-linux/ascend_toolkit_install.info")
foreach(version_file "${ASCEND_HOME_PATH}/version.info"
    "${ASCEND_HOME_PATH}/compiler/version.info"
    "${ASCEND_HOME_PATH}/ascend_toolkit_install.info"
    "${ASCEND_HOME_PATH}/share/info/ascend_toolkit_install.info" ${cann_install_info})
  list(APPEND cann_version_checked "${version_file}")
  if(EXISTS "${version_file}")
    # CANN 8.0 publishes component Version=7.6.0.1.220 alongside
    # version_dir=8.0.0. The product version_dir takes precedence.
    file(STRINGS "${version_file}" product_lines REGEX "^[ \t]*version_dir[ \t]*=")
    file(STRINGS "${version_file}" component_lines REGEX "^[ \t]*[Vv]ersion[ \t]*=")
    set(version_lines ${product_lines} ${component_lines})
    foreach(version_line IN LISTS version_lines)
      if(version_line MATCHES "=[ \t]*\"?([0-9]+)\\.")
        if(CMAKE_MATCH_1 GREATER_EQUAL 8)
          set(CANN_VERSION_MAJOR "${CMAKE_MATCH_1}")
          message(STATUS "vllm-ascend tests: CANN major ${CANN_VERSION_MAJOR} from ${version_file}")
          return()
        endif()
      endif()
    endforeach()
  endif()
endforeach()
get_filename_component(cann_real_root "${ASCEND_HOME_PATH}" REALPATH)
foreach(version_path "${cann_real_root}" "${ASCEND_HOME_PATH}")
  if(version_path MATCHES "(^|/)(cann-)?([0-9]+)\\.[0-9]+(\\.[0-9]+)?([^/]*)(/|$)")
    if(CMAKE_MATCH_3 GREATER_EQUAL 8)
      set(CANN_VERSION_MAJOR "${CMAKE_MATCH_3}")
      message(STATUS "vllm-ascend tests: CANN major ${CANN_VERSION_MAJOR} from path ${version_path}")
      return()
    endif()
  endif()
endforeach()
message(FATAL_ERROR "Cannot detect CANN major version. Checked: ${cann_version_checked}; "
  "paths ${ASCEND_HOME_PATH} and ${cann_real_root}. Pass -DCANN_VERSION_MAJOR=8 or 9.")
