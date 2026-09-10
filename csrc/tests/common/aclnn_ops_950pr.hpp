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

// Operators the Ascend 950PR (A5) leg of the suite calls, on top of the four
// already declared in aclnn_ops.hpp.
//
//   stage        operator                            source
//   ---------------------------------------------------------------------------
//   RMSNorm      aclnnRmsNorm                        stock CANN
//   projections  aclnnMatmul                         stock CANN (cube)
//   RoPE         aclnnInplacePartialRotaryMul        vllm-ascend custom op
//                 -> aclnnApplyRotaryPosEmbV2        stock CANN, packed fallback
//   KV write     aclnnScatterPaKvCache               stock CANN
//   decode       aclnnFusedInferAttentionScoreV2     stock CANN
//   SwiGLU       aclnnSwiGlu                         stock CANN
//   gate / adds  aclnnSigmoid, aclnnMul, aclnnInplaceAdd   stock CANN
//
// Only the rotary stage needs a custom kernel: Qwen3.5 rotates channels [0, 64)
// of a 256-wide head and aclnnApplyRotaryPosEmbV2 rotates the whole trailing
// dim. Every name above resolves against stock CANN 9.1.0 EXCEPT
// aclnnInplacePartialRotaryMul, which only appears once the vllm-ascend custom
// op package is installed into the OPP.
//
// Like aclnn_ops.hpp, this file is version-sensitive: the operators are
// resolved with dlsym, so the compiler cannot check an argument list. Each
// declaration names the header it was read from.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "aclnn_ops.hpp"
#include "aclnn_runtime.hpp"

namespace vllm_ascend {
namespace test {
namespace ops950 {

// ---------------------------------------------------------------------------
// Partial rotary position embedding (vllm-ascend custom op)
// ---------------------------------------------------------------------------
// mirrors: torch.ops._C_ascend.inplace_partial_rotary_mul
//          csrc/torch_binding.cpp :: inplace_partial_rotary_mul_npu
//
// Rotates a slice of every head in place and leaves the rest untouched:
//
//   x[..., s : s + n] = rotate(x[..., s : s + n], cos, sin)
//   x[..., elsewhere] unchanged
//
// x is BSND [batch, seq, num_heads, head_dim]; the torch binding rejects any
// other rank. cos and sin carry the rotary width, not the head width.
//
// From
// csrc/attention/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_def.cpp:
//
//   Input("x")   fp16 | fp32 | bf16          AutoContiguous
//   Input("cos") fp16 | fp32 | bf16          AutoContiguous
//   Input("sin") fp16 | fp32 | bf16          AutoContiguous
//   Output("x")  fp16 | fp32 | bf16
//   Attr("mode")          Int,     default 0
//   Attr("partial_slice") ListInt, default {0, 0}
//   AICore config for ascend910b, ascend910_93 and ascend950
//
// The torch binding calls
// EXEC_NPU_CMD(aclnnInplacePartialRotaryMul, x, r1, r2, mode, partial_slice),
// which lays arguments out in declaration order and aliases the single output
// onto the input, so the aclnn argument list is x, cos, sin, mode,
// partialSlice. mode is from {half:0, interleave:1, quarter:2,
// interleave-half:3}.
//
// UNVERIFIED: derived from the operator definition and the torch call site, not
// read out of a generated header, which only exists once the custom op package
// is built. partial_slice is assumed to be {start, length}.
using InplacePartialRotaryMulWorkspaceFn = int (*)(aclTensor* x_ref, const aclTensor* cos, const aclTensor* sin,
                                                   int64_t mode, const aclIntArray* partial_slice,
                                                   uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kInplacePartialRotaryMul = "aclnnInplacePartialRotaryMul";

// The mode enum, from the mode_map in csrc/torch_binding.cpp.
inline constexpr int64_t kRotaryModeHalf = 0;
inline constexpr int64_t kRotaryModeInterleave = 1;
inline constexpr int64_t kRotaryModeQuarter = 2;
inline constexpr int64_t kRotaryModeInterleaveHalf = 3;

// ---------------------------------------------------------------------------
// Fused infer attention score (paged decode)
// ---------------------------------------------------------------------------
// mirrors: torch_npu.npu_fused_infer_attention_score
//          vllm_ascend/attention/attention_v1.py, the DecodeOnly branch of
//          AscendAttentionBackendImpl._get_fia_params and the call after it.
//
// The plugin's decode recipe, which the golden test reproduces argument for
// argument:
//
//   key   = key_cache.view(num_blocks, block_size, num_kv_heads * head_size)
//   value = value_cache.view(same)
//   input_layout          = "TND"
//   query                 [total_tokens, num_heads, head_size]
//   block_table           [batch, max_blocks_per_seq], int32
//   actual_seq_lengths    cumulative query lengths
//   actual_seq_lengths_kv context length per sequence
//   sparse_mode           0
//
// key and value arrive as aclTensorList with one entry.
//
// This operator does not run on an Ascend950: the planning call returns 361001,
// "Interface aclnnFusedInferAttentionScore versions V1 to V4 are no longer
// supported on Ascend950". Tests that call it tolerate its absence rather than
// requiring it.
//
// VERIFIED against CANN 9.1.0
// $ASCEND_HOME_PATH/include/aclnnop/aclnn_fused_infer_attention_score_v2.h
//   aclnnStatus aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
//       const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
//       const aclTensor *pseShiftOptional, const aclTensor *attenMaskOptional,
//       const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional,
//       const aclTensor *deqScale1Optional, const aclTensor *quantScale1Optional,
//       const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional,
//       const aclTensor *quantOffset2Optional, const aclTensor *antiquantScaleOptional,
//       const aclTensor *antiquantOffsetOptional, const aclTensor *blockTableOptional,
//       const aclTensor *queryPaddingSizeOptional, const aclTensor *kvPaddingSizeOptional,
//       const aclTensor *keyAntiquantScaleOptional, const aclTensor *keyAntiquantOffsetOptional,
//       const aclTensor *valueAntiquantScaleOptional, const aclTensor *valueAntiquantOffsetOptional,
//       const aclTensor *keySharedPrefixOptional, const aclTensor *valueSharedPrefixOptional,
//       const aclIntArray *actualSharedPrefixLenOptional, int64_t numHeads, double scaleValue,
//       int64_t preTokens, int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads,
//       int64_t sparseMode, int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode,
//       bool softmaxLseFlag, int64_t keyAntiquantMode, int64_t valueAntiquantMode,
//       const aclTensor *attentionOut, const aclTensor *softmaxLse,
//       uint64_t *workspaceSize, aclOpExecutor **executor);
//
// softmaxLse is a required output tensor even when softmaxLseFlag is false; the
// plugin allocates a one-element tensor for it and discards it.
using FusedInferAttentionScoreV2WorkspaceFn = int (*)(
    const aclTensor* query, const aclTensorList* key, const aclTensorList* value, const aclTensor* pse_shift,
    const aclTensor* atten_mask, const aclIntArray* actual_seq_lengths, const aclIntArray* actual_seq_lengths_kv,
    const aclTensor* deq_scale1, const aclTensor* quant_scale1, const aclTensor* deq_scale2,
    const aclTensor* quant_scale2, const aclTensor* quant_offset2, const aclTensor* antiquant_scale,
    const aclTensor* antiquant_offset, const aclTensor* block_table, const aclTensor* query_padding_size,
    const aclTensor* kv_padding_size, const aclTensor* key_antiquant_scale, const aclTensor* key_antiquant_offset,
    const aclTensor* value_antiquant_scale, const aclTensor* value_antiquant_offset,
    const aclTensor* key_shared_prefix, const aclTensor* value_shared_prefix,
    const aclIntArray* actual_shared_prefix_len, int64_t num_heads, double scale_value, int64_t pre_tokens,
    int64_t next_tokens, char* input_layout, int64_t num_key_value_heads, int64_t sparse_mode,
    int64_t inner_precise, int64_t block_size, int64_t antiquant_mode, bool softmax_lse_flag,
    int64_t key_antiquant_mode, int64_t value_antiquant_mode, const aclTensor* attention_out,
    const aclTensor* softmax_lse, uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kFusedInferAttentionScoreV2 = "aclnnFusedInferAttentionScoreV2";

// V5 - the interface that replaces the withdrawn V1..V4 family on an Ascend950.
// Callers should prefer it and fall back to V2; see
// device/bench_device_950pr_turboquant.cpp.
//
// It is V2 plus nine parameters. Seven optional inputs slot in after
// actualSharedPrefixLenOptional:
//
//   queryRopeOptional, keyRopeOptional        MLA's split RoPE path
//   keyRopeAntiquantScaleOptional
//   dequantScaleQueryOptional                 per-query dequant
//   learnableSinkOptional                     learned attention-sink logit
//   qStartIdxOptional, kvStartIdxOptional     sliding-window offsets
//
// and two int64 scalars after valueAntiquantMode: queryQuantMode and pseType. A
// plain fp16 paged decode wants none of the seven, queryQuantMode 0 and the
// pseType default; see shapes950::kFiaQueryQuantModeNone and
// kFiaPseTypeDefault.
//
// VERIFIED against CANN 9.2.0-beta.2
// $ASCEND_HOME_PATH/include/aclnnop/aclnn_fused_infer_attention_score_v5.h
//   aclnnStatus aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
//       const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
//       const aclTensor *pseShiftOptional, const aclTensor *attenMaskOptional,
//       const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional,
//       const aclTensor *deqScale1Optional, const aclTensor *quantScale1Optional,
//       const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional,
//       const aclTensor *quantOffset2Optional, const aclTensor *antiquantScaleOptional,
//       const aclTensor *antiquantOffsetOptional, const aclTensor *blockTableOptional,
//       const aclTensor *queryPaddingSizeOptional, const aclTensor *kvPaddingSizeOptional,
//       const aclTensor *keyAntiquantScaleOptional, const aclTensor *keyAntiquantOffsetOptional,
//       const aclTensor *valueAntiquantScaleOptional, const aclTensor *valueAntiquantOffsetOptional,
//       const aclTensor *keySharedPrefixOptional, const aclTensor *valueSharedPrefixOptional,
//       const aclIntArray *actualSharedPrefixLenOptional, const aclTensor *queryRopeOptional,
//       const aclTensor *keyRopeOptional, const aclTensor *keyRopeAntiquantScaleOptional,
//       const aclTensor *dequantScaleQueryOptional, const aclTensor *learnableSinkOptional,
//       const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
//       int64_t numHeads, double scaleValue, int64_t preTokens, int64_t nextTokens,
//       char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise,
//       int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode,
//       int64_t valueAntiquantMode, int64_t queryQuantMode, int64_t pseType,
//       const aclTensor *attentionOut, const aclTensor *softmaxLse,
//       uint64_t *workspaceSize, aclOpExecutor **executor);
//
// NOT YET EXECUTED on a part. A mismatch shows up as a non-zero planning status
// with the CANN diagnostic attached.
using FusedInferAttentionScoreV5WorkspaceFn = int (*)(
    const aclTensor* query, const aclTensorList* key, const aclTensorList* value, const aclTensor* pse_shift,
    const aclTensor* atten_mask, const aclIntArray* actual_seq_lengths, const aclIntArray* actual_seq_lengths_kv,
    const aclTensor* deq_scale1, const aclTensor* quant_scale1, const aclTensor* deq_scale2,
    const aclTensor* quant_scale2, const aclTensor* quant_offset2, const aclTensor* antiquant_scale,
    const aclTensor* antiquant_offset, const aclTensor* block_table, const aclTensor* query_padding_size,
    const aclTensor* kv_padding_size, const aclTensor* key_antiquant_scale, const aclTensor* key_antiquant_offset,
    const aclTensor* value_antiquant_scale, const aclTensor* value_antiquant_offset,
    const aclTensor* key_shared_prefix, const aclTensor* value_shared_prefix,
    const aclIntArray* actual_shared_prefix_len, const aclTensor* query_rope, const aclTensor* key_rope,
    const aclTensor* key_rope_antiquant_scale, const aclTensor* dequant_scale_query,
    const aclTensor* learnable_sink, const aclIntArray* q_start_idx, const aclIntArray* kv_start_idx,
    int64_t num_heads, double scale_value, int64_t pre_tokens, int64_t next_tokens, char* input_layout,
    int64_t num_key_value_heads, int64_t sparse_mode, int64_t inner_precise, int64_t block_size,
    int64_t antiquant_mode, bool softmax_lse_flag, int64_t key_antiquant_mode, int64_t value_antiquant_mode,
    int64_t query_quant_mode, int64_t pse_type, const aclTensor* attention_out, const aclTensor* softmax_lse,
    uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kFusedInferAttentionScoreV5 = "aclnnFusedInferAttentionScoreV5";

// The layout string the plugin's decode path passes. "TND" means the query is
// [total_tokens, num_heads, head_size] with the per-sequence split carried by
// actualSeqLengths rather than by a batch axis, which is what lets a ragged
// decode batch go through one call.
inline const char* kFiaLayoutTnd = "TND";

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------
// The attention output gate is sigmoid(gate) * context and the two residual
// connections are x += y. Neither is fused on this part.

// VERIFIED against CANN 9.1.0 $ASCEND_HOME_PATH/include/aclnnop/aclnn_sigmoid.h
//   aclnnStatus aclnnSigmoidGetWorkspaceSize(const aclTensor* self, aclTensor* out,
//                                            uint64_t* workspaceSize, aclOpExecutor** executor);
using SigmoidWorkspaceFn = int (*)(const aclTensor* self, aclTensor* out, uint64_t* workspace_size,
                                   aclOpExecutor** executor);
inline const char* kSigmoid = "aclnnSigmoid";

// VERIFIED against CANN 9.1.0 $ASCEND_HOME_PATH/include/aclnnop/aclnn_mul.h
//   aclnnStatus aclnnMulGetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
//                                        uint64_t* workspaceSize, aclOpExecutor** executor);
using MulWorkspaceFn = int (*)(const aclTensor* self, const aclTensor* other, aclTensor* out,
                               uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kMul = "aclnnMul";

// VERIFIED against CANN 9.1.0 $ASCEND_HOME_PATH/include/aclnnop/aclnn_add.h
//   aclnnStatus aclnnInplaceAddGetWorkspaceSize(const aclTensor* selfRef, const aclTensor* other,
//                                               const aclScalar* alpha, uint64_t* workspaceSize,
//                                               aclOpExecutor** executor);
//
// selfRef += alpha * other. A residual add wants alpha = 1.
using InplaceAddWorkspaceFn = int (*)(const aclTensor* self_ref, const aclTensor* other, const aclScalar* alpha,
                                      uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kInplaceAdd = "aclnnInplaceAdd";

// ---------------------------------------------------------------------------
// Inventory
// ---------------------------------------------------------------------------

// Every operator the 950PR leg can call, stock and custom, in pipeline order.
// Printed once by main_950pr.cpp so a run that ends in skips explains itself on
// its first few lines.
std::vector<ops::OpAvailability> ProbeAscend950Operators();

void PrintAscend950OperatorInventory();

}  // namespace ops950
}  // namespace test
}  // namespace vllm_ascend
