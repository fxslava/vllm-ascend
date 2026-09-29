/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef DSV4_MOE_EXPERT_TORCH_ADPT_H
#define DSV4_MOE_EXPERT_TORCH_ADPT_H

namespace vllm_ascend {

// One expert's fused projection: gate/up FP4 GEMM + SwiGLU + down FP4 GEMM.
// Every tensor is caller-allocated (write-only outputs), mirroring the
// zero-allocation ExpertKernelRunner seam of tools/dsv4_moe_runtime:
//   x          [1, hidden]      bf16
//   w1, w3     [inter, hidden/2] uint8 (packed FP4)   + [inter, hidden/32] uint8 E8M0 scales
//   w2         [hidden, inter/2] uint8 (packed FP4)   + [hidden, inter/32] uint8 E8M0 scales
//   gate_out/up_out/activated [1, inter] bf16, down_out [1, hidden] bf16
void dsv4_moe_expert(const at::Tensor& x,
                     const at::Tensor& w1,
                     const at::Tensor& w2,
                     const at::Tensor& w3,
                     const at::Tensor& w1_scale,
                     const at::Tensor& w2_scale,
                     const at::Tensor& w3_scale,
                     at::Tensor& gate_out,
                     at::Tensor& up_out,
                     at::Tensor& activated,
                     at::Tensor& down_out)
{
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1,
                "dsv4_moe_expert: x must be [1, hidden], got ", x.sizes());
    const int64_t hidden = x.size(1);
    const int64_t inter = w1.size(0);
    TORCH_CHECK(x.scalar_type() == at::kBFloat16,
                "dsv4_moe_expert: x must be bfloat16, got ", x.scalar_type());
    TORCH_CHECK(w1.dim() == 2 && w1.size(1) == hidden / 2,
                "dsv4_moe_expert: w1 must be [inter, hidden/2], got ", w1.sizes());
    TORCH_CHECK(w3.sizes() == w1.sizes(),
                "dsv4_moe_expert: w3 must match w1, got ", w3.sizes());
    TORCH_CHECK(w2.dim() == 2 && w2.size(0) == hidden && w2.size(1) == inter / 2,
                "dsv4_moe_expert: w2 must be [hidden, inter/2], got ", w2.sizes());
    for (const auto* t : {&w1, &w2, &w3, &w1_scale, &w2_scale, &w3_scale}) {
        TORCH_CHECK(t->scalar_type() == torch::kUInt8 && t->is_contiguous(),
                    "dsv4_moe_expert: weights and scales must be contiguous uint8");
    }
    TORCH_CHECK(gate_out.size(1) == inter && up_out.size(1) == inter && activated.size(1) == inter &&
                    down_out.size(1) == hidden,
                "dsv4_moe_expert: output rows must match [1, inter]/[1, hidden]");

    EXEC_NPU_CMD(aclnnDsv4MoeExpert,
                 x,
                 w1,
                 w2,
                 w3,
                 w1_scale,
                 w2_scale,
                 w3_scale,
                 gate_out,
                 up_out,
                 activated,
                 down_out);
}

}  // namespace vllm_ascend

#endif  // DSV4_MOE_EXPERT_TORCH_ADPT_H
