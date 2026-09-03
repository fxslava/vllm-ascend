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

#include "aclnn_ops.hpp"

#include <cstdio>
#include <sstream>

namespace vllm_ascend {
namespace test {
namespace ops {

AclnnOp ResolveFirstAvailable(std::initializer_list<const char*> candidate_names) {
  const char* first = nullptr;
  for (const char* name : candidate_names) {
    if (first == nullptr) {
      first = name;
    }
    AclnnOp candidate(name);
    if (candidate.available()) {
      return candidate;
    }
  }
  // Nothing resolved. Return the first candidate so the skip message names a
  // concrete operator, and let the caller print the full list.
  return AclnnOp(first != nullptr ? first : "<no candidates>");
}

std::vector<OpAvailability> ProbeAllOperators() {
  // Order: the operators the suite uses, then the three that CANN 9.1.0 does not
  // provide as aclnn (kept so the inventory documents their absence).
  const char* names[] = {
      kRmsNorm,          kSwiGlu,          kApplyRotaryPosEmbV2, kApplyRotaryPosEmb,
      kScatterPaKvCache, kIncreFlashAttentionV4,
      // ATB-backed / GE-backed, expected MISSING on CANN 9.1.0:
      kRotaryMul,        kReshapeAndCache, kPagedAttention,
  };

  std::vector<OpAvailability> results;
  results.reserve(sizeof(names) / sizeof(names[0]));
  for (const char* name : names) {
    AclnnOp op(name);
    OpAvailability entry;
    entry.name = name;
    entry.available = op.available();
    if (!entry.available) {
      entry.detail = op.unavailable_reason();
    }
    results.push_back(entry);
  }
  return results;
}

void PrintOperatorInventory() {
  const OpApiLibrary& library = OpApiLibrary::Instance();
  std::printf("[ascend-test] aclnn operator inventory\n");
  if (!library.loaded()) {
    std::printf("[ascend-test]   libopapi.so NOT loaded: %s\n", library.load_error().c_str());
    return;
  }
  for (const OpAvailability& entry : ProbeAllOperators()) {
    std::printf("[ascend-test]   %-28s %s\n", entry.name.c_str(), entry.available ? "found" : "MISSING");
  }
}

}  // namespace ops
}  // namespace test
}  // namespace vllm_ascend
