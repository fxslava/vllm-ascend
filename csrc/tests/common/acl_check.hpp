/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Error-checking helpers for the raw ACL runtime.
//
// Two flavours are provided on purpose:
//   * ACL_CHECK      - throws AclError. Use it inside RAII types and helpers
//                      where a GTest fatal assertion cannot unwind cleanly.
//   * ASSERT_ACL_OK  - GTest fatal assertion. Use it directly in test bodies.

#pragma once

#include <acl/acl.h>
#include <acl/acl_base.h>

#include <exception>
#include <sstream>
#include <string>

namespace vllm_ascend {
namespace test {

// aclGetRecentErrMsg() carries the CANN-side diagnostic and is far more useful
// than the numeric status on its own, so it is always folded into the message.
inline std::string AclRecentErrorMessage() {
  const char* message = aclGetRecentErrMsg();
  return (message != nullptr) ? std::string(message) : std::string("<no CANN error message>");
}

inline std::string FormatAclError(const char* expression, const char* file, int line, int status) {
  std::ostringstream stream;
  stream << file << ":" << line << ": " << expression << " failed with status " << status << "\n  CANN: "
         << AclRecentErrorMessage();
  return stream.str();
}

class AclError : public std::exception {
 public:
  AclError(const char* expression, const char* file, int line, int status)
      : message_(FormatAclError(expression, file, line, status)), status_(status) {}

  const char* what() const noexcept override { return message_.c_str(); }
  int status() const { return status_; }

 private:
  std::string message_;
  int status_;
};

#define ACL_CHECK(expression)                                                     \
  do {                                                                            \
    const int vllm_ascend_acl_status = static_cast<int>(expression);               \
    if (vllm_ascend_acl_status != ACL_SUCCESS) {                                  \
      throw ::vllm_ascend::test::AclError(#expression, __FILE__, __LINE__,        \
                                          vllm_ascend_acl_status);                \
    }                                                                             \
  } while (false)

// Non-throwing variant for destructors and teardown paths, where an exception
// would terminate the process and hide the original failure.
#define ACL_CHECK_NOTHROW(expression)                                             \
  do {                                                                            \
    const int vllm_ascend_acl_status = static_cast<int>(expression);               \
    if (vllm_ascend_acl_status != ACL_SUCCESS) {                                  \
      ::vllm_ascend::test::ReportIgnoredAclFailure(                               \
          #expression, __FILE__, __LINE__, vllm_ascend_acl_status);               \
    }                                                                             \
  } while (false)

void ReportIgnoredAclFailure(const char* expression, const char* file, int line, int status);

#define ASSERT_ACL_OK(expression)                                                 \
  do {                                                                            \
    const int vllm_ascend_acl_status = static_cast<int>(expression);               \
    ASSERT_EQ(vllm_ascend_acl_status, ACL_SUCCESS)                                \
        << ::vllm_ascend::test::FormatAclError(#expression, __FILE__, __LINE__,   \
                                               vllm_ascend_acl_status);           \
  } while (false)

#define EXPECT_ACL_OK(expression)                                                 \
  do {                                                                            \
    const int vllm_ascend_acl_status = static_cast<int>(expression);               \
    EXPECT_EQ(vllm_ascend_acl_status, ACL_SUCCESS)                                \
        << ::vllm_ascend::test::FormatAclError(#expression, __FILE__, __LINE__,   \
                                               vllm_ascend_acl_status);           \
  } while (false)

}  // namespace test
}  // namespace vllm_ascend
