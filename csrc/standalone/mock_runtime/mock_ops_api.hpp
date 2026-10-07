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

// The test-facing surface of libopapi_mock: direct declarations of the
// operator entry points dsv4_mock_test calls (the dsv4 product reaches them
// through dlsym, so no product header declares them) plus the validator
// entry points that skip executor plumbing for negative cases.

#pragma once

#include <cstdint>

#include "aclnn/acl_meta.h"

extern "C" {
aclnnStatus aclnnSoftplusGetWorkspaceSize(const aclTensor* self, const aclScalar* beta, const aclScalar* threshold,
                                          aclTensor* out, uint64_t* workspace_size, aclOpExecutor** executor);
aclnnStatus aclnnSqrtGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspace_size,
                                      aclOpExecutor** executor);
aclnnStatus aclnnMoeGatingTopKV2GetWorkspaceSize(const aclTensor* x, const aclTensor* bias_optional,
                                                 const aclTensor* input_ids_optional,
                                                 const aclTensor* tid2eid_optional, int64_t k, int64_t k_group,
                                                 int64_t group_count, int64_t group_select_mode, int64_t renorm,
                                                 int64_t norm_type, bool out_flag, double routed_scaling_factor,
                                                 double eps, const aclTensor* y_out, const aclTensor* expert_idx_out,
                                                 const aclTensor* out_out, uint64_t* workspace_size,
                                                 aclOpExecutor** executor);
aclnnStatus aclnnMoeInitRoutingV4GetWorkspaceSize(
    const aclTensor* x, const aclTensor* expert_idx, const aclTensor* scale_optional,
    const aclTensor* offset_optional, const aclTensor* active_num_optional, const aclTensor* topk_weight_optional,
    int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode, int64_t expert_tokens_num_type,
    bool expert_tokens_num_flag, int64_t quant_mode, const aclIntArray* active_expert_range_optional,
    int64_t row_idx_type, const aclTensor* expanded_x_out, const aclTensor* expanded_row_idx_out,
    const aclTensor* expert_tokens_count_or_cumsum_out, const aclTensor* expanded_scale_out,
    const aclTensor* expanded_topk_weight_out, uint64_t* workspace_size, aclOpExecutor** executor);
aclnnStatus aclnnGroupedMatmulV5GetWorkspaceSize(
    const aclTensorList* x, const aclTensorList* weight, const aclTensorList* bias_optional,
    const aclTensorList* scale_optional, const aclTensorList* offset_optional,
    const aclTensorList* antiquant_scale_optional, const aclTensorList* antiquant_offset_optional,
    const aclTensorList* per_token_scale_optional, const aclTensor* group_list_optional,
    const aclTensorList* activation_input_optional, const aclTensorList* activation_quant_scale_optional,
    const aclTensorList* activation_quant_offset_optional, int64_t split_item, int64_t group_type,
    int64_t group_list_type, int64_t act_type, aclIntArray* tuning_config_optional, aclTensorList* out,
    aclTensorList* activation_feature_out_optional, aclTensorList* dyn_quant_scale_out_optional,
    uint64_t* workspace_size, aclOpExecutor** executor);
}

namespace vllm_ascend {
namespace dsv4 {
namespace mock {

aclnnStatus MockValidateGatingForTest(const aclTensor* x, const aclTensor* bias_optional, int64_t k, int64_t k_group,
                                      int64_t group_count, int64_t group_select_mode, int64_t renorm,
                                      int64_t norm_type, double routed_scaling_factor, const aclTensor* y_out,
                                      const aclTensor* expert_idx_out);
aclnnStatus MockValidateRoutingForTest(const aclTensor* expert_idx, int64_t expert_num,
                                       const aclTensor* group_list_out);
aclnnStatus MockValidateGmmForTest(const aclTensorList* weight, const aclTensorList* scale_optional,
                                   int64_t split_item, int64_t group_type);

}  // namespace mock
}  // namespace dsv4
}  // namespace vllm_ascend
