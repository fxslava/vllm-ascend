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

#include "aclnn_ops_950pr.hpp"

#include <cstdio>

namespace vllm_ascend {
namespace test {
namespace ops950 {

std::vector<ops::OpAvailability> ProbeAscend950Operators() {
  // Pipeline order, so the inventory reads like the layer: norm, projections,
  // rotary, KV write, decode, activation, elementwise.
  struct Entry {
    const char* name;
    const char* note;
  };

  const Entry entries[] = {
      {ops::kRmsNorm, "stage 1 and 7 - input and post-attention RMSNorm"},
      {ops::kMatmul, "stage 2, 6, 8 - every linear projection, cube unit"},
      {kInplacePartialRotaryMul, "stage 3/4 - partial RoPE, vllm-ascend custom op"},
      {ops::kApplyRotaryPosEmbV2, "stage 3/4 - partial RoPE fallback over a packed slice"},
      {ops::kScatterPaKvCache, "stage 5 - paged KV cache write"},
      {kFusedInferAttentionScoreV2, "stage 5 - paged decode attention"},
      {kSigmoid, "stage 6 - attention output gate"},
      {kMul, "stage 6 - attention output gate"},
      {ops::kSwiGlu, "stage 8 - SwiGLU activation"},
      {kInplaceAdd, "stage 6 and 9 - residual adds"},
  };

  std::vector<ops::OpAvailability> results;
  results.reserve(sizeof(entries) / sizeof(entries[0]));
  for (const Entry& entry : entries) {
    AclnnOp op(entry.name);
    ops::OpAvailability record;
    record.name = entry.name;
    record.available = op.available();
    record.detail = entry.note;
    // Which library it came from is as interesting as whether it resolved: a
    // "found" that came out of libcust_opapi.so is a kernel from csrc/, and a
    // "found" from libopapi.so is stock CANN.
    if (op.available()) {
      record.detail += op.is_custom() ? "  [custom op package]" : "  [CANN]";
    }
    results.push_back(record);
  }
  return results;
}

void PrintAscend950OperatorInventory() {
  const OpApiLibrary& library = OpApiLibrary::Instance();
  std::printf("[ascend-test] aclnn operator inventory (Ascend 950PR pipeline)\n");
  if (!library.loaded()) {
    std::printf("[ascend-test]   libopapi.so NOT loaded: %s\n", library.load_error().c_str());
    return;
  }
  if (library.custom_library_paths().empty()) {
    std::printf("[ascend-test]   custom op packages: none found\n");
  } else {
    for (const std::string& path : library.custom_library_paths()) {
      std::printf("[ascend-test]   custom op package: %s\n", path.c_str());
    }
  }
  for (const ops::OpAvailability& entry : ProbeAscend950Operators()) {
    std::printf("[ascend-test]   %-34s %-7s %s\n", entry.name.c_str(), entry.available ? "found" : "MISSING",
                entry.detail.c_str());
  }
  // The one expected MISSING on a tree with stock CANN, called out so a reader
  // does not treat it as a broken install. See the header for why.
  if (!AclnnOp(kInplacePartialRotaryMul).available()) {
    std::printf(
        "[ascend-test]   note: %s comes from the vllm-ascend custom op package\n"
        "[ascend-test]         (csrc/attention/inplace_partial_rotary_mul), not from CANN.\n"
        "[ascend-test]         Rotary stages fall back to %s over a packed slice.\n",
        kInplacePartialRotaryMul, ops::kApplyRotaryPosEmbV2);
  }
}

}  // namespace ops950
}  // namespace test
}  // namespace vllm_ascend
