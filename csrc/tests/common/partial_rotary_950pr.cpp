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

#include "partial_rotary_950pr.hpp"

#include <cstddef>

#include "acl_check.hpp"
#include "device_buffer.hpp"
#include "device_tensor.hpp"

namespace vllm_ascend {
namespace test {
namespace {

constexpr size_t kHalfBytes = sizeof(uint16_t);

// "half" is the neox / rotate_half pairing: element k with element
// k + rotary_dim/2. That is what Qwen3.5 uses, what
// scripts/dump_qwen35_layer3.py's apply_partial_rope implements, and what the
// cos/sin tables in the dump are laid out for (each is concat(v, v), so the two
// paired elements read the same angle by construction).
//
// Non-const because the operator takes char*, not const char*.
char kRotaryModeHalfString[] = "half";

// Copies the [0, rotary_dim) slice of every head out of a
// [tokens, heads, head_dim] buffer into a contiguous
// [tokens, heads, rotary_dim] one, or back again.
//
// One copy per head. tokens * heads is 8 for the golden layer and at most a few
// thousand for the unit test sweeps, which is small enough that the launch cost
// does not matter next to the operator itself - and this is a correctness
// suite, not a benchmark.
void CopyRotarySlices(void* strided, void* packed, int64_t tokens, int64_t heads, int64_t head_dim,
                      int64_t rotary_dim, bool pack, aclrtStream stream) {
  auto* strided_bytes = static_cast<uint8_t*>(strided);
  auto* packed_bytes = static_cast<uint8_t*>(packed);
  const size_t slice_bytes = static_cast<size_t>(rotary_dim) * kHalfBytes;

  for (int64_t index = 0; index < tokens * heads; ++index) {
    uint8_t* strided_head = strided_bytes + static_cast<size_t>(index * head_dim) * kHalfBytes;
    uint8_t* packed_head = packed_bytes + static_cast<size_t>(index * rotary_dim) * kHalfBytes;
    uint8_t* src = pack ? strided_head : packed_head;
    uint8_t* dst = pack ? packed_head : strided_head;
    ACL_CHECK(aclrtMemcpyAsync(dst, slice_bytes, src, slice_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
  }
}

// Runs the stock operator over two BSND tensors that are already exactly
// rotary_dim wide.
void RunStockRotary(void* q_data, void* k_data, const DeviceTensor& cos, const DeviceTensor& sin, int64_t tokens,
                    int64_t q_heads, int64_t kv_heads, int64_t width, aclrtStream stream) {
  AclnnTensor query({1, tokens, q_heads, width}, ACL_FLOAT16, q_data);
  AclnnTensor key({1, tokens, kv_heads, width}, ACL_FLOAT16, k_data);

  RunAclnn<ops::ApplyRotaryPosEmbWorkspaceFn>(PartialRotaryStockOp(), stream, query.get(), key.get(), cos.get(),
                                              sin.get(), ops::kApplyRotaryPosEmbLayoutBsnd,
                                              kRotaryModeHalfString);
}

// Runs the custom operator once per tensor over the full head width.
void RunCustomRotary(void* q_data, void* k_data, const DeviceTensor& cos, const DeviceTensor& sin, int64_t tokens,
                     int64_t q_heads, int64_t kv_heads, int64_t head_dim, int64_t rotary_dim,
                     aclrtStream stream) {
  // partial_slice as {start, length}: rotate [0, rotary_dim) of every head. See
  // the UNVERIFIED note on kInplacePartialRotaryMul - the default {0, 0} means
  // "no slice", which is what makes {start, length} the reading this assumes.
  AclnnIntArray partial_slice(std::vector<int64_t>{0, rotary_dim});

  AclnnTensor query({1, tokens, q_heads, head_dim}, ACL_FLOAT16, q_data);
  RunAclnn<ops950::InplacePartialRotaryMulWorkspaceFn>(PartialRotaryCustomOp(), stream, query.get(), cos.get(),
                                                       sin.get(), ops950::kRotaryModeHalf, partial_slice.get());

  AclnnTensor key({1, tokens, kv_heads, head_dim}, ACL_FLOAT16, k_data);
  RunAclnn<ops950::InplacePartialRotaryMulWorkspaceFn>(PartialRotaryCustomOp(), stream, key.get(), cos.get(),
                                                       sin.get(), ops950::kRotaryModeHalf, partial_slice.get());
}

}  // namespace

const char* PartialRotaryPathName(PartialRotaryPath path) {
  switch (path) {
    case PartialRotaryPath::kCustomOp:
      return "aclnnInplacePartialRotaryMul (vllm-ascend custom op)";
    case PartialRotaryPath::kPackedApplyRotary:
      return "aclnnApplyRotaryPosEmbV2 over a packed rotary slice";
    case PartialRotaryPath::kFullApplyRotary:
      return "aclnnApplyRotaryPosEmbV2 (rotary_dim == head_dim, no slicing)";
  }
  return "<unknown>";
}

const AclnnOp& PartialRotaryCustomOp() {
  static const AclnnOp op(ops950::kInplacePartialRotaryMul);
  return op;
}

const AclnnOp& PartialRotaryStockOp() {
  // The operator gained a V2 suffix partway through the CANN 8.x line; try the
  // newer name first and fall back to the original.
  static const AclnnOp op = ops::ResolveFirstAvailable({ops::kApplyRotaryPosEmbV2, ops::kApplyRotaryPosEmb});
  return op;
}

bool SelectPartialRotaryPath(int64_t head_dim, int64_t rotary_dim, PartialRotaryPath* path, std::string* reason) {
  if (rotary_dim == head_dim) {
    if (!PartialRotaryStockOp().available()) {
      *reason = PartialRotaryStockOp().unavailable_reason();
      return false;
    }
    *path = PartialRotaryPath::kFullApplyRotary;
    return true;
  }

  // Partial rotation. Prefer the operator that was written for it.
  if (PartialRotaryCustomOp().available()) {
    *path = PartialRotaryPath::kCustomOp;
    return true;
  }
  if (PartialRotaryStockOp().available()) {
    *path = PartialRotaryPath::kPackedApplyRotary;
    return true;
  }

  *reason = "no partial rotary operator available: " + PartialRotaryCustomOp().unavailable_reason() +
            " (this one ships with the vllm-ascend custom op package, not with CANN) and " +
            PartialRotaryStockOp().unavailable_reason();
  return false;
}

PartialRotaryPath ApplyPartialRotaryQK(void* q_data, void* k_data, const std::vector<float>& cos_full,
                                       const std::vector<float>& sin_full, int64_t tokens, int64_t q_heads,
                                       int64_t kv_heads, int64_t head_dim, int64_t rotary_dim,
                                       aclrtStream stream) {
  PartialRotaryPath path = PartialRotaryPath::kPackedApplyRotary;
  std::string reason;
  if (!SelectPartialRotaryPath(head_dim, rotary_dim, &path, &reason)) {
    throw AclError(reason.c_str(), __FILE__, __LINE__, -1);
  }

  // BSND with the head axis broadcast: cos and sin are shared across heads, so
  // the N axis is 1. This is the shape _rope_forward_oot builds and the shape
  // the golden dump's [1, rotary_dim] tables widen to for tokens == 1.
  DeviceTensor cos = DeviceTensor::Half({1, tokens, 1, rotary_dim}, cos_full);
  DeviceTensor sin = DeviceTensor::Half({1, tokens, 1, rotary_dim}, sin_full);

  switch (path) {
    case PartialRotaryPath::kFullApplyRotary:
      RunStockRotary(q_data, k_data, cos, sin, tokens, q_heads, kv_heads, head_dim, stream);
      return path;

    case PartialRotaryPath::kCustomOp:
      RunCustomRotary(q_data, k_data, cos, sin, tokens, q_heads, kv_heads, head_dim, rotary_dim, stream);
      return path;

    case PartialRotaryPath::kPackedApplyRotary: {
      DeviceBuffer packed_q(static_cast<size_t>(tokens * q_heads * rotary_dim) * kHalfBytes);
      DeviceBuffer packed_k(static_cast<size_t>(tokens * kv_heads * rotary_dim) * kHalfBytes);

      CopyRotarySlices(q_data, packed_q.get(), tokens, q_heads, head_dim, rotary_dim, /*pack=*/true, stream);
      CopyRotarySlices(k_data, packed_k.get(), tokens, kv_heads, head_dim, rotary_dim, /*pack=*/true, stream);
      // RunAclnn synchronises after the launch, but the copies above are async
      // on the same stream and the operator has to see them completed. Same
      // stream means same order, so this is ordering, not a race - the
      // synchronise is here so a failure in a copy is reported against the copy.
      ACL_CHECK(aclrtSynchronizeStream(stream));

      RunStockRotary(packed_q.get(), packed_k.get(), cos, sin, tokens, q_heads, kv_heads, rotary_dim, stream);

      CopyRotarySlices(q_data, packed_q.get(), tokens, q_heads, head_dim, rotary_dim, /*pack=*/false, stream);
      CopyRotarySlices(k_data, packed_k.get(), tokens, kv_heads, head_dim, rotary_dim, /*pack=*/false, stream);
      // packed_q and packed_k are freed when this scope ends, so the unpack has
      // to have finished before that happens.
      ACL_CHECK(aclrtSynchronizeStream(stream));
      return path;
    }
  }

  // Unreachable: every enumerator returns above. Kept so the function has a
  // definite return value under compilers that do not see that.
  return path;
}

}  // namespace test
}  // namespace vllm_ascend
