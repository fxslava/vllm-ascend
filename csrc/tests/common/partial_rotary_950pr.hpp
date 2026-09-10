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

// Partial rotary position embedding on Ascend 950PR.
//
// Qwen3.5 sets partial_rotary_factor to 0.25 with head_dim 256, so a decode
// step must rotate channels [0, 64) of every head and leave [64, 256) exactly
// as they were. aclnnApplyRotaryPosEmbV2 cannot express that: it rotates the
// whole trailing dimension, and its cos/sin must be that same width.
//
// Two ways out, both implemented here and chosen between at run time:
//
//   1. aclnnInplacePartialRotaryMul, the vllm-ascend custom operator built from
//      csrc/attention/inplace_partial_rotary_mul. It takes the full head plus a
//      partial_slice attribute and has an ascend950 AICore config. It is NOT
//      part of CANN: it only resolves once the vllm-ascend custom op package is
//      installed into the OPP.
//   2. Pack the rotary slice of every head into a contiguous
//      [1, tokens, heads, rotary_dim] buffer, rotate that with the stock
//      aclnnApplyRotaryPosEmbV2, and unpack it back.
//
// A strided aclTensor view over the full head is deliberately not used: queryRef
// is an in-place output, and an aclnn operator handed a non-contiguous input may
// materialise a contiguous copy first, leaving the caller's buffer unchanged.
//
// The path actually taken is reported back to the caller.

#pragma once

#include <acl/acl.h>

#include <cstdint>
#include <string>
#include <vector>

#include "aclnn_ops.hpp"
#include "aclnn_ops_950pr.hpp"
#include "aclnn_runtime.hpp"

namespace vllm_ascend {
namespace test {

enum class PartialRotaryPath {
  // aclnnInplacePartialRotaryMul, one call per tensor.
  kCustomOp,
  // aclnnApplyRotaryPosEmbV2 over a packed copy of the rotary slice.
  kPackedApplyRotary,
  // rotary_dim == head_dim, so no slicing is needed and the stock operator runs
  // directly over the caller's buffers.
  kFullApplyRotary,
};

const char* PartialRotaryPathName(PartialRotaryPath path);

// Process-wide resolved operators, so a skip message can name what is missing
// without a test having to resolve them itself.
const AclnnOp& PartialRotaryCustomOp();
const AclnnOp& PartialRotaryStockOp();

// Which path ApplyPartialRotaryQK would take for this rotary_dim / head_dim
// pair, without running anything. `reason` is filled in and false returned when
// neither operator is available.
bool SelectPartialRotaryPath(int64_t head_dim, int64_t rotary_dim, PartialRotaryPath* path, std::string* reason);

// Rotates query and key in place.
//
//   q_data   fp16 device memory, [tokens, q_heads, head_dim]
//   k_data   fp16 device memory, [tokens, kv_heads, head_dim]
//   cos_full host values, [tokens, rotary_dim], already widened to the full
//            rotary width the way reference::GatherFullCosSin produces and
//            scripts/dump_qwen35_layer3.py writes
//
// Returns the path taken. Throws AclError if neither operator resolves; call
// SelectPartialRotaryPath first to turn that into a skip.
PartialRotaryPath ApplyPartialRotaryQK(void* q_data, void* k_data, const std::vector<float>& cos_full,
                                       const std::vector<float>& sin_full, int64_t tokens, int64_t q_heads,
                                       int64_t kv_heads, int64_t head_dim, int64_t rotary_dim,
                                       aclrtStream stream);

}  // namespace test
}  // namespace vllm_ascend
