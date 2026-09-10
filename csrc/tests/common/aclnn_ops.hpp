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
// Version-sensitive: the suite resolves operators with dlsym (see
// aclnn_runtime.hpp), so the compiler checks none of these argument lists.
// Every prototype marked VERIFIED was read out of the CANN 9.1.0 headers named
// beside it. A mismatch produces a non-zero status from the planning call, with
// aclGetRecentErrMsg attached by RunAclnn.
//
// Three operators the plugin uses are not aclnn at all on CANN 9.1.0 and are
// declared here only so the inventory reports their absence: PagedAttention and
// ReshapeAndCache are ATB operators, RotaryMul is a GE graph op.
//
// The "mirrors" line on each operator records the torch_npu entry point and the
// vllm-ascend call site the test stands in for.

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
// MatMul (Cube core)
// ---------------------------------------------------------------------------
// mirrors: torch.nn.functional.linear, i.e. every QKV / o_proj / gate_up / down
//          projection in the Qwen3.5 forward pass. On 310P these run on the
//          v200 cube unit, which none of the other four suites touch: RMSNorm,
//          SwiGLU and RoPE are all vector-unit work.
//
//   out = self @ mat2      self [M, K], mat2 [K, N], out [M, N]
//
// A Linear layer stores its weight as [out_features, in_features] = [N, K] and
// computes x @ W^T, so the test hands `mat2` a [K, N] *view* with strides
// {1, K} over [N, K] storage rather than materialising a transpose. See
// DeviceTensor::HalfTransposed2D.
//
// VERIFIED against CANN 9.1.0
// $ASCEND_HOME_PATH/include/aclnnop/aclnn_matmul.h
//   aclnnStatus aclnnMatmulGetWorkspaceSize(
//       const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType,
//       uint64_t* workspaceSize, aclOpExecutor** executor);
using MatmulWorkspaceFn = int (*)(const aclTensor* self, const aclTensor* mat2, aclTensor* out,
                                  int8_t cube_math_type, uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kMatmul = "aclnnMatmul";

// cubeMathType is an int8 enum. Documented in the aclnn_mv.h comment block:
//   0 KEEP_DTYPE, 1 ALLOW_FP32_DOWN_PRECISION, 2 USE_FP16, 3 USE_HF32
// The inputs here are already fp16, so KEEP_DTYPE leaves the cube unit in fp16
// and the comparison measures the kernel rather than a precision policy.
inline constexpr int8_t kCubeMathTypeKeepDtype = 0;

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
// VERIFIED against CANN 9.1.0
// $ASCEND_HOME_PATH/include/aclnnop/aclnn_apply_rotary_pos_emb_v2.h:24
//   aclnnStatus aclnnApplyRotaryPosEmbV2GetWorkspaceSize(
//       aclTensor* queryRef, aclTensor* keyRef, const aclTensor* cos, const aclTensor* sin,
//       int64_t layout, char* rotaryMode, uint64_t* workspaceSize, aclOpExecutor** executor);
using ApplyRotaryPosEmbWorkspaceFn = int (*)(aclTensor* query_ref, aclTensor* key_ref, const aclTensor* cos,
                                             const aclTensor* sin, int64_t layout, char* rotary_mode,
                                             uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kApplyRotaryPosEmbV2 = "aclnnApplyRotaryPosEmbV2";
inline const char* kApplyRotaryPosEmb = "aclnnApplyRotaryPosEmb";

// `layout` is an int64 enum, not a string. torch_npu calls
// npu_apply_rotary_pos_emb(..., layout=1, rotary_mode='half') and the 310P
// module hands it BSND-shaped tensors [1, num_tokens, num_heads, head_dim], so
// 1 is BSND. Confirm on hardware before relying on any other value.
inline constexpr int64_t kApplyRotaryPosEmbLayoutBsnd = 1;

// NOT AVAILABLE as aclnn on CANN 9.1.0.
// torch_npu.npu_rotary_mul is backed by the GE graph op `RotaryMul`
// (ge::op::RotaryMul in liboponnx_plugin_legacy.so / liboptf_plugin_legacy.so),
// not by an aclnn entry point. There is no aclnnRotaryMul in libopapi.so.
// Kept only so the inventory reports it; no test uses it.
inline const char* kRotaryMul = "aclnnRotaryMul";

// ---------------------------------------------------------------------------
// KV cache write
// ---------------------------------------------------------------------------
// NOT AVAILABLE as aclnn on CANN 9.1.0. torch_npu._npu_reshape_and_cache is an
// ATB operator (atb::ReshapeAndCacheOperation in libatb.so), a C++ object API
// that RunAclnn's two-phase aclnn path cannot drive.
//
// Kept so the inventory reports it and the test skip message stays accurate.
inline const char* kReshapeAndCache = "aclnnReshapeAndCache";

// The aclnn route to the same result.
//
// mirrors: torch_npu.npu_scatter_pa_kv_cache
//          vllm_ascend/device/device_op.py :: BaseDeviceAdaptor.reshape_and_cache
//          which calls it with cache_mode="Norm".
//
// VERIFIED against CANN 9.1.0
// $ASCEND_HOME_PATH/include/aclnnop/level2/aclnn_scatter_pa_kv_cache.h
//   aclnnStatus aclnnScatterPaKvCacheGetWorkspaceSize(
//       const aclTensor *key, aclTensor *keyCacheRef, const aclTensor *slotMapping,
//       const aclTensor *value, aclTensor *valueCacheRef,
//       const aclTensor *compressLensOptional, const aclTensor *compressSeqOffsetOptional,
//       const aclTensor *seqLensOptional, char *cacheModeOptional, char *scatterModeOptional,
//       const aclIntArray *stridesOptional, const aclIntArray *offsetsOptional,
//       uint64_t *workspaceSize, aclOpExecutor **executor);
//
// Note the interleaved argument order: key, keyCache, slotMapping, then value
// and valueCache. The optional tensors take nullptr.
using ScatterPaKvCacheWorkspaceFn = int (*)(const aclTensor* key, aclTensor* key_cache_ref,
                                            const aclTensor* slot_mapping, const aclTensor* value,
                                            aclTensor* value_cache_ref, const aclTensor* compress_lens_optional,
                                            const aclTensor* compress_seq_offset_optional,
                                            const aclTensor* seq_lens_optional, char* cache_mode_optional,
                                            char* scatter_mode_optional, const aclIntArray* strides_optional,
                                            const aclIntArray* offsets_optional, uint64_t* workspace_size,
                                            aclOpExecutor** executor);
inline const char* kScatterPaKvCache = "aclnnScatterPaKvCache";

// cacheMode as passed by BaseDeviceAdaptor.reshape_and_cache. The header does
// not enumerate the accepted values, and whether the 310P 5-D NZ cache needs a
// different mode is UNCONFIRMED - verify on hardware before trusting a result.
inline const char* kScatterCacheModeNorm = "Norm";

// ---------------------------------------------------------------------------
// Paged attention (decode)
// ---------------------------------------------------------------------------
// NOT AVAILABLE as aclnn on CANN 9.1.0. torch_npu._npu_paged_attention is an
// ATB operator (atb::PagedAttentionOperation in libatb.so), a C++ object API
// that RunAclnn's two-phase aclnn path cannot drive. There is no
// aclnnPagedAttention in libopapi.so.
//
// Kept so the inventory reports it and the test skip message stays accurate.
inline const char* kPagedAttention = "aclnnPagedAttention";

// The aclnn route to paged decode attention.
//
// mirrors: torch_npu.npu_fused_infer_attention_score / npu_incre_flash_attention,
//          used by the generic backend in vllm_ascend/attention/attention_v1.py.
//
// VERIFIED against CANN 9.1.0
// $ASCEND_HOME_PATH/include/aclnnop/aclnn_incre_flash_attention_v4.h:54
//   aclnnStatus aclnnIncreFlashAttentionV4GetWorkspaceSize(
//       const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
//       const aclTensor *pseShift, const aclTensor *attenMask,
//       const aclIntArray *actualSeqLengths, const aclTensor *dequantScale1,
//       const aclTensor *quantScale1, const aclTensor *dequantScale2,
//       const aclTensor *quantScale2, const aclTensor *quantOffset2,
//       const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
//       const aclTensor *blocktable, const aclTensor *kvPaddingSize,
//       int64_t numHeads, double scaleValue, char *inputLayout,
//       int64_t numKeyValueHeads, int64_t blockSize, int64_t innerPrecise,
//       const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor);
//
// key/value arrive as aclTensorList, one entry per layer, not as a single cache
// tensor, and the paged KV layout IFA expects alongside blocktable/blockSize is
// not the 310P 5-D NZ shape the plugin allocates -- so this is not a drop-in
// swap for the 310P test and the paged-attention suite still skips.
using IncreFlashAttentionV4WorkspaceFn = int (*)(
    const aclTensor* query, const aclTensorList* key, const aclTensorList* value, const aclTensor* pse_shift,
    const aclTensor* atten_mask, const aclIntArray* actual_seq_lengths, const aclTensor* dequant_scale1,
    const aclTensor* quant_scale1, const aclTensor* dequant_scale2, const aclTensor* quant_scale2,
    const aclTensor* quant_offset2, const aclTensor* antiquant_scale, const aclTensor* antiquant_offset,
    const aclTensor* blocktable, const aclTensor* kv_padding_size, int64_t num_heads, double scale_value,
    char* input_layout, int64_t num_key_value_heads, int64_t block_size, int64_t inner_precise,
    const aclTensor* attention_out, uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kIncreFlashAttentionV4 = "aclnnIncreFlashAttentionV4";

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
