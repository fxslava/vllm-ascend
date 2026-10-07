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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "aclnn_ops.hpp"
#include "aclnn_runtime.hpp"

namespace vllm_ascend {
namespace test {
namespace ops950 {

using InplacePartialRotaryMulWorkspaceFn = int (*)(aclTensor* x_ref, const aclTensor* cos, const aclTensor* sin,
                                                   int64_t mode, const aclIntArray* partial_slice,
                                                   uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kInplacePartialRotaryMul = "aclnnInplacePartialRotaryMul";

inline constexpr int64_t kRotaryModeHalf = 0;
inline constexpr int64_t kRotaryModeInterleave = 1;
inline constexpr int64_t kRotaryModeQuarter = 2;
inline constexpr int64_t kRotaryModeInterleaveHalf = 3;

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

inline const char* kFiaLayoutTnd = "TND";

// ---------------------------------------------------------------------------
// DeepSeek-V4 Flash execution mapping (Ascend 950PR, aclnn V5): 43 layers,
// hidden 4096, moe_intermediate 2048, 256 routed + 1 shared expert, top-6,
// sqrtsoftplus/noaux_tc routing with routed scaling 1.5, swiglu limit 10.0.
// Precision contract: FP8 E4M3 (aclDataType 36) dense/MLA with UE8M0 block-128
// scales; routed expert GEMM weights FP4 E2M1 (40) with UE8M0 block-32 (37).
// ---------------------------------------------------------------------------
inline constexpr int64_t kDsv4HiddenSize = 4096;
inline constexpr int64_t kDsv4MoeIntermediateSize = 2048;
inline constexpr int64_t kDsv4NumLayers = 43;
inline constexpr int64_t kDsv4NumRoutedExperts = 256;
inline constexpr int64_t kDsv4NumSharedExperts = 1;
inline constexpr int64_t kDsv4NumExpertsPerTok = 6;
inline constexpr int64_t kDsv4NumAttentionHeads = 64;
inline constexpr int64_t kDsv4QLoraRank = 1024;
inline constexpr int64_t kDsv4IndexHeadDim = 128;
inline constexpr double kDsv4RoutedScalingFactor = 1.5;
inline constexpr double kDsv4SwigluLimit = 10.0;

// GMMActType::SILU on the grouped GEMM; the DSV4 path keeps actType NONE on
// the GEMM and applies the clamped SwiGLU (limit 10.0) via aclnnSwigluMxQuant.
inline constexpr int64_t kGmmActTypeSilu = 5;
inline constexpr int64_t kGmmActTypeNone = 0;

// aclnnGroupedMatmulV5: M-grouped FP4 expert GEMM. groupListOptional is a
// device INT64 tensor (cumsum / sizes / sparse-kv via groupListType), so the
// ragged per-expert token counts never round-trip through the host.
using GroupedMatmulV5WorkspaceFn =
    int (*)(const aclTensorList* x, const aclTensorList* weight, const aclTensorList* bias_optional,
            const aclTensorList* scale_optional, const aclTensorList* offset_optional,
            const aclTensorList* antiquant_scale_optional, const aclTensorList* antiquant_offset_optional,
            const aclTensorList* per_token_scale_optional, const aclTensor* group_list_optional,
            const aclTensorList* activation_input_optional, const aclTensorList* activation_quant_scale_optional,
            const aclTensorList* activation_quant_offset_optional, int64_t split_item, int64_t group_type,
            int64_t group_list_type, int64_t act_type, const aclIntArray* tuning_config_optional, aclTensorList* out,
            aclTensorList* activation_feature_out_optional, aclTensorList* dyn_quant_scale_out_optional,
            uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kGroupedMatmulV5 = "aclnnGroupedMatmulV5";
inline const char* kGroupedMatmulV5GetWorkspaceSize = "aclnnGroupedMatmulV5GetWorkspaceSize";

// aclnnMoeGatingTopKV2: noaux_tc router (bias shifts selection only, weights
// keep their raw scores); takes the optional tid2eid table the DSV4 pool builds.
using MoeGatingTopKV2WorkspaceFn =
    int (*)(const aclTensor* x, const aclTensor* bias_optional, const aclTensor* input_ids_optional,
            const aclTensor* tid2eid_optional, int64_t k, int64_t k_group, int64_t group_count,
            int64_t group_select_mode, int64_t renorm, int64_t norm_type, bool out_flag,
            double routed_scaling_factor, double eps, const aclTensor* y_out, const aclTensor* expert_idx_out,
            const aclTensor* out_out, uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kMoeGatingTopKV2 = "aclnnMoeGatingTopKV2";
inline const char* kMoeGatingTopKV2GetWorkspaceSize = "aclnnMoeGatingTopKV2GetWorkspaceSize";

// aclnnMoeInitRoutingV4: dropless dispatch; expertTokensCountOrCumsumOut is
// the device cumsum that feeds GroupedMatmulV5's groupList directly.
using MoeInitRoutingV4WorkspaceFn =
    int (*)(const aclTensor* x, const aclTensor* expert_idx, const aclTensor* scale_optional,
            const aclTensor* offset_optional, const aclTensor* active_num_optional,
            const aclTensor* topk_weight_optional, int64_t expert_capacity, int64_t expert_num,
            int64_t drop_pad_mode, int64_t expert_tokens_num_type, bool expert_tokens_num_flag,
            int64_t quant_mode, const aclIntArray* active_expert_range_optional, int64_t row_idx_type,
            const aclTensor* expanded_x_out, const aclTensor* expanded_row_idx_out,
            const aclTensor* expert_tokens_count_or_cumsum_out, const aclTensor* expanded_scale_out,
            const aclTensor* expanded_topk_weight_out, uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kMoeInitRoutingV4 = "aclnnMoeInitRoutingV4";
inline const char* kMoeInitRoutingV4GetWorkspaceSize = "aclnnMoeInitRoutingV4GetWorkspaceSize";

using SigmoidWorkspaceFn = int (*)(const aclTensor* self, aclTensor* out, uint64_t* workspace_size,
                                   aclOpExecutor** executor);
inline const char* kSigmoid = "aclnnSigmoid";

using MulWorkspaceFn = int (*)(const aclTensor* self, const aclTensor* other, aclTensor* out,
                               uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kMul = "aclnnMul";

using InplaceAddWorkspaceFn = int (*)(const aclTensor* self_ref, const aclTensor* other, const aclScalar* alpha,
                                      uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kInplaceAdd = "aclnnInplaceAdd";

std::vector<ops::OpAvailability> ProbeAscend950Operators();

void PrintAscend950OperatorInventory();

}
}
}
