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
#include <initializer_list>
#include <string>
#include <vector>

#include "aclnn_runtime.hpp"

namespace vllm_ascend {
namespace test {
namespace ops {

AclnnOp ResolveFirstAvailable(std::initializer_list<const char*> candidate_names);

using MatmulWorkspaceFn = int (*)(const aclTensor* self, const aclTensor* mat2, aclTensor* out,
                                  int8_t cube_math_type, uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kMatmul = "aclnnMatmul";

inline constexpr int8_t kCubeMathTypeKeepDtype = 0;

using RmsNormWorkspaceFn = int (*)(const aclTensor* x, const aclTensor* gamma, double epsilon, const aclTensor* y_out,
                                   const aclTensor* rstd_out, uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kRmsNorm = "aclnnRmsNorm";

using SwiGluWorkspaceFn = int (*)(const aclTensor* x, int64_t dim, const aclTensor* out, uint64_t* workspace_size,
                                  aclOpExecutor** executor);
inline const char* kSwiGlu = "aclnnSwiGlu";

using ApplyRotaryPosEmbWorkspaceFn = int (*)(aclTensor* query_ref, aclTensor* key_ref, const aclTensor* cos,
                                             const aclTensor* sin, int64_t layout, char* rotary_mode,
                                             uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kApplyRotaryPosEmbV2 = "aclnnApplyRotaryPosEmbV2";
inline const char* kApplyRotaryPosEmb = "aclnnApplyRotaryPosEmb";

inline constexpr int64_t kApplyRotaryPosEmbLayoutBsnd = 1;

inline const char* kRotaryMul = "aclnnRotaryMul";

inline const char* kReshapeAndCache = "aclnnReshapeAndCache";

using ScatterPaKvCacheWorkspaceFn = int (*)(const aclTensor* key, aclTensor* key_cache_ref,
                                            const aclTensor* slot_mapping, const aclTensor* value,
                                            aclTensor* value_cache_ref, const aclTensor* compress_lens_optional,
                                            const aclTensor* compress_seq_offset_optional,
                                            const aclTensor* seq_lens_optional, char* cache_mode_optional,
                                            char* scatter_mode_optional, const aclIntArray* strides_optional,
                                            const aclIntArray* offsets_optional, uint64_t* workspace_size,
                                            aclOpExecutor** executor);
inline const char* kScatterPaKvCache = "aclnnScatterPaKvCache";

inline const char* kScatterCacheModeNorm = "Norm";

inline const char* kPagedAttention = "aclnnPagedAttention";

using IncreFlashAttentionV4WorkspaceFn = int (*)(
    const aclTensor* query, const aclTensorList* key, const aclTensorList* value, const aclTensor* pse_shift,
    const aclTensor* atten_mask, const aclIntArray* actual_seq_lengths, const aclTensor* dequant_scale1,
    const aclTensor* quant_scale1, const aclTensor* dequant_scale2, const aclTensor* quant_scale2,
    const aclTensor* quant_offset2, const aclTensor* antiquant_scale, const aclTensor* antiquant_offset,
    const aclTensor* blocktable, const aclTensor* kv_padding_size, int64_t num_heads, double scale_value,
    char* input_layout, int64_t num_key_value_heads, int64_t block_size, int64_t inner_precise,
    const aclTensor* attention_out, uint64_t* workspace_size, aclOpExecutor** executor);
inline const char* kIncreFlashAttentionV4 = "aclnnIncreFlashAttentionV4";

struct OpAvailability {
  std::string name;
  bool available = false;
  std::string detail;
};

std::vector<OpAvailability> ProbeAllOperators();

void PrintOperatorInventory();

}
}
}
