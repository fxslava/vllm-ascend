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

// Signatures of the CANN aclnn operators exercised by this suite.
//
// This is the single version-sensitive file in the subproject. Every prototype
// below is declared by hand because the suite resolves operators with dlsym
// (see aclnn_runtime.hpp) and therefore gets no compiler check on the argument
// list. Before trusting a first run on a new CANN release, verify each one:
//
//   grep -A20 'GetWorkspaceSize' $ASCEND_HOME_PATH/include/aclnnop/aclnn_rms_norm.h
//   grep -A20 'GetWorkspaceSize' $ASCEND_HOME_PATH/include/aclnnop/aclnn_swi_glu.h
//   grep -rA20 'ApplyRotaryPosEmb.*GetWorkspaceSize' $ASCEND_HOME_PATH/include/aclnnop/
//   grep -rA20 'ReshapeAndCacheGetWorkspaceSize'     $ASCEND_HOME_PATH/include/aclnnop/
//   grep -rA24 'PagedAttentionGetWorkspaceSize'      $ASCEND_HOME_PATH/include/aclnnop/
//
// A prototype that does not match produces a non-zero status from the planning
// call, and RunAclnn attaches aclGetRecentErrMsg to the failure, so a mismatch
// is reported rather than silently miscomputed.
//
// The "mirrors" line on each operator records the torch_npu entry point and the
// vllm-ascend call site the test is standing in for, so the C++ coverage can be
// traced back to the Python path it replaces.

#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "aclnn_runtime.hpp"

namespace vllm_ascend {
namespace test {
namespace ops {

// Returns the first operator name that resolves. Operator names gained V2/V3
// suffixes across CANN releases; the candidate list keeps that churn out of the
// test bodies. When none resolve, the returned op reports every name tried.
AclnnOp ResolveFirstAvailable(std::initializer_list<const char*> candidate_names);

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------
// mirrors: torch_npu.npu_rms_norm
//          vllm_ascend/_310p/ops/layernorm.py :: AscendRMSNorm310.forward_oot
//
// y     = x / sqrt(mean(x^2) + epsilon) * gamma
// rstd  = 1 / sqrt(mean(x^2) + epsilon), shape = x.shape with the last dim = 1
//
// Reduction runs over the last axis. `rstd` is a required output even though
// the plugin discards it (the Python side unpacks it as `_`).
using RmsNormWorkspaceFn = int (*)(const aclTensor* x, const aclTensor* gamma, double epsilon, const aclTensor* y_out,
                                   const aclTensor* rstd_out, uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kRmsNorm = "aclnnRmsNorm";

// ---------------------------------------------------------------------------
// SwiGLU (SiluAndMul)
// ---------------------------------------------------------------------------
// mirrors: torch_npu.npu_swiglu
//          vllm_ascend/_310p/ops/activation.py :: AscendSiluAndMul310.forward
//
// Splits x in half along `dim` and returns silu(first_half) * second_half, so
// out.shape[dim] == x.shape[dim] / 2.
//
// The 310P path only calls this when x.shape[-1] % 32 == 0 and otherwise falls
// back to eager torch, which the activation suite covers explicitly.
using SwiGluWorkspaceFn = int (*)(const aclTensor* x, int64_t dim, const aclTensor* out, uint64_t* workspace_size,
                                  aclOpExecutor** executor);
inline const char* kSwiGlu = "aclnnSwiGlu";

// ---------------------------------------------------------------------------
// Rotary position embedding
// ---------------------------------------------------------------------------
// mirrors: torch_npu.npu_apply_rotary_pos_emb
//          vllm_ascend/_310p/ops/rotary_embedding.py :: _rope_forward_oot
//
// In-place on query and key, both in BSND layout [1, num_tokens, num_heads,
// head_dim]. cos/sin are [1, num_tokens, 1, rotary_dim] and already carry the
// full rotary dim (the plugin concatenates the half-dim cache with itself; see
// set_mrope_apply_rotary_slices).
//
//   out = x * cos + rotate(x) * sin
//
// rotary_mode selects rotate():
//   "half"       - neox / rotate_half: (-x[d/2:], x[:d/2])
//   "interleave" - GPT-J pairs:        (-x[1::2], x[0::2]) interleaved back
//
// Only head_dim 64 and 128 are supported by this operator on 310P, which is
// why AscendMRotaryEmbedding310 gates on `self.rotary_dim in (64, 128)`.
using ApplyRotaryPosEmbWorkspaceFn = int (*)(aclTensor* query_ref, aclTensor* key_ref, const aclTensor* cos,
                                             const aclTensor* sin, char* layout_optional, char* rotary_mode_optional,
                                             uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kApplyRotaryPosEmbV2 = "aclnnApplyRotaryPosEmbV2";
inline const char* kApplyRotaryPosEmb = "aclnnApplyRotaryPosEmb";

// mirrors: torch_npu.npu_rotary_mul
//          vllm_ascend/ops/rotary_embedding.py :: AscendRotaryEmbedding.forward_oot (partial-rotary path)
//
// Out-of-place variant used when the rotary dim is a strict prefix of head_dim:
// out = x * r1 + rotate_half(x) * r2, with r1/r2 broadcast over [B, S, N, D].
using RotaryMulWorkspaceFn = int (*)(const aclTensor* x, const aclTensor* r1, const aclTensor* r2,
                                     const aclTensor* out, uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kRotaryMul = "aclnnRotaryMul";

// ---------------------------------------------------------------------------
// KV cache write
// ---------------------------------------------------------------------------
// mirrors: torch_npu._npu_reshape_and_cache
//          vllm_ascend/device/device_op.py :: Ascend310PDeviceAdaptor.reshape_and_cache
//
// Scatters key/value [num_tokens, num_kv_heads, head_size] into the paged cache
// at the flat slots given by slot_mapping (slot = block_id * block_size + offset).
// keyCacheRef/valueCacheRef are mutated in place.
using ReshapeAndCacheWorkspaceFn = int (*)(const aclTensor* key, const aclTensor* value, aclTensor* key_cache_ref,
                                           aclTensor* value_cache_ref, const aclTensor* slot_mapping,
                                           uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kReshapeAndCache = "aclnnReshapeAndCache";

// ---------------------------------------------------------------------------
// Paged attention (decode)
// ---------------------------------------------------------------------------
// mirrors: torch_npu._npu_paged_attention
//          vllm_ascend/_310p/attention/attention_v1.py :: AscendAttentionBackendImpl310.forward_paged_attention
//
// query      [num_tokens, num_heads, head_size]
// key/value  310P 5-D NZ cache, [num_blocks, num_kv_heads*head_size/16, block_size, 16]
// blockTable [num_seqs, max_blocks_per_seq] int32
// contextLens[num_seqs] int32
// attnOut    [num_tokens, num_heads, head_size]
//
// The parameter order below follows the torch_npu binding, which
// attention_v1.py calls as
//
//   torch_npu._npu_paged_attention(query, key_cache, value_cache, num_kv_heads,
//                                  num_heads, scale_value, block_table,
//                                  context_lens, out)
//
// so the scalars sit between the caches and the paging tensors rather than
// after them. This operator carries the largest number of optional arguments in
// CANN and is the most likely of the five to differ between releases. If the
// planning call returns a non-zero status on a new toolkit, check the local
// aclnnop header first: a tensors-then-scalars ordering
// (query, keyCache, valueCache, blockTable, contextLens, numKvHeads, numHeads,
// scaleValue, attnOut) is the other shape this operator has shipped with.
using PagedAttentionWorkspaceFn = int (*)(const aclTensor* query, const aclTensor* key_cache,
                                          const aclTensor* value_cache, int64_t num_kv_heads, int64_t num_heads,
                                          double scale_value, const aclTensor* block_table,
                                          const aclTensor* context_lens, const aclTensor* attn_out,
                                          uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kPagedAttention = "aclnnPagedAttention";

// Reports which of the operators above the local CANN install exports. Printed
// once by main() so a failing run starts with the environment inventory.
struct OpAvailability {
  std::string name;
  bool available = false;
  std::string detail;
};

std::vector<OpAvailability> ProbeAllOperators();

void PrintOperatorInventory();

}  // namespace ops
}  // namespace test
}  // namespace vllm_ascend
