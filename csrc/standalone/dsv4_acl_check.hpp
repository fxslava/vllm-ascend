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

// Status checking for the standalone runner.
//
// Every ACL / ACLNN status is checked at its call site and turned into an
// exception carrying the call, the file and the numeric status. There is no
// "log and continue" path: a failed DMA inside the exclusive hierarchy leaves
// the partition in a state no retry can repair (see ExclusiveExpertManager's
// poisoning), so the only safe response is to unwind.

#pragma once

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vllm_ascend {
namespace dsv4 {

class Dsv4Error : public std::runtime_error {
 public:
  explicit Dsv4Error(const std::string& message) : std::runtime_error(message) {}
};

class AclError : public Dsv4Error {
 public:
  AclError(const char* call, const char* file, int line, int64_t status)
      : Dsv4Error(Format(call, file, line, status)), status_(status) {}

  int64_t status() const { return status_; }

 private:
  static std::string Format(const char* call, const char* file, int line, int64_t status) {
    std::ostringstream out;
    out << call << " failed with status " << status << " at " << file << ":" << line;
    return out.str();
  }

  int64_t status_ = 0;
};

#define DSV4_ACL_CHECK(expr)                                                   \
  do {                                                                         \
    const auto dsv4_status_ = (expr);                                          \
    if (dsv4_status_ != 0) {                                                   \
      throw ::vllm_ascend::dsv4::AclError(#expr, __FILE__, __LINE__,           \
                                          static_cast<int64_t>(dsv4_status_)); \
    }                                                                          \
  } while (false)

#define DSV4_REQUIRE(condition, message)                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::ostringstream dsv4_msg_;                                            \
      dsv4_msg_ << message << " [" << #condition << " at " << __FILE__ << ":"   \
                << __LINE__ << "]";                                            \
      throw ::vllm_ascend::dsv4::Dsv4Error(dsv4_msg_.str());                   \
    }                                                                          \
  } while (false)

}  // namespace dsv4
}  // namespace vllm_ascend
