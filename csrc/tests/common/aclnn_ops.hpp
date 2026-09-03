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
// list.
//
// Every prototype marked VERIFIED was read out of the CANN 9.1.0 headers in
// swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-310p-ubuntu22.04-py3.10.
// To re-verify on another toolkit:
//
//   INC=$ASCEND_HOME_PATH/include/aclnnop
//   grep -rA8 'aclnnRmsNormGetWorkspaceSize('              $INC/aclnn_rms_norm.h
//   grep -rA8 'aclnnSwiGluGetWorkspaceSize('               $INC/level2/aclnn_swi_glu.h
//   grep -rA6 'aclnnApplyRotaryPosEmbV2GetWorkspaceSize('  $INC/aclnn_apply_rotary_pos_emb_v2.h
//   grep -rA16 'aclnnScatterPaKvCacheGetWorkspaceSize('    $INC/level2/aclnn_scatter_pa_kv_cache.h
//   grep -rA10 'aclnnIncreFlashAttentionV4GetWorkspaceSize(' $INC/aclnn_incre_flash_attention_v4.h
//
// To confirm which library exports a symbol at all:
//
//   nm -D --defined-only $ASCEND_HOME_PATH/lib64/libopapi.so | grep -o 'aclnn[A-Za-z0-9_]*'
//
// Three operators the plugin uses are NOT aclnn at all on CANN 9.1.0 and are
// documented as such below: PagedAttention and ReshapeAndCache are ATB
// operators in /usr/local/Ascend/nnal/atb/.../libatb.so, and RotaryMul is a GE
// graph op. They are declared here only so the inventory reports their absence
// and the test skip messages stay accurate.
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
// NOT AVAILABLE as aclnn on CANN 9.1.0.
//
// torch_npu._npu_reshape_and_cache (which Ascend310PDeviceAdaptor.reshape_and_cache
// calls) is an ATB operator, not an aclnn one. A full-depth nm -D sweep of
// /usr/local/Ascend found it only as C++ symbols in
//   /usr/local/Ascend/nnal/atb/9.1.0/atb/cxx_abi_{0,1}/lib/libatb.so
//     atb::ReshapeAndCacheOperation
//     atb::CreateOperation<atb::infer::ReshapeAndCacheParam>
// plus kernel-side tiling in libatb_mixops.so (ReshapeAndCacheTilingNd /
// ReshapeAndCacheTilingNz) and infershape stubs in libopsproto.so.
//
// ATB is a C++ object API (Operation + VariantPack + Context), not the aclnn
// two-phase C API, so it cannot be driven through RunAclnn. Reaching it would
// mean linking libatb.so and writing a second execution path.
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
// NOT AVAILABLE as aclnn on CANN 9.1.0.
//
// torch_npu._npu_paged_attention (AscendAttentionBackendImpl310.forward_paged_attention)
// is an ATB operator. A full-depth nm -D sweep of /usr/local/Ascend found it
// only as C++ symbols in
//   /usr/local/Ascend/nnal/atb/9.1.0/atb/cxx_abi_{0,1}/lib/libatb.so
//     atb::PagedAttentionOperation
//     atb::CreateOperation<atb::infer::PagedAttentionParam>
// with tiling in libatb_mixops.so (AtbOps::PagedAttentionTilingParams).
// There is no aclnnPagedAttention in libopapi.so, and no matching header in
// $ASCEND_HOME_PATH/include/aclnnop/.
//
// ATB is a C++ object API (Operation + VariantPack + Context), not the aclnn
// two-phase C API, so it cannot be driven through RunAclnn.
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
// Two differences from the ATB operator that make this NOT a drop-in swap for
// the 310P test, and why the paged-attention suite still skips:
//   * key/value arrive as aclTensorList, one entry per layer, not as a single
//     cache tensor.
//   * the paged KV layout IFA expects alongside `blocktable`/`blockSize` is not
//     the 310P 5-D NZ shape the plugin allocates. Wiring the existing test to
//     it would mean guessing that layout, which is exactly what this file is
//     meant to avoid.
// The declaration is here, verified, so that work starts from a checked
// prototype rather than a guess.
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
