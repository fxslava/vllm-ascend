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
