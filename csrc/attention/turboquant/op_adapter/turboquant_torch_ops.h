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

// The TurboQuant operators' torch.ops._C_ascend schemas and NPU implementations.
//
// Shared by the two libraries that register them: vllm_ascend_C (csrc/torch_binding.cpp)
// and the standalone libvllm_turboquant_cube.so (csrc/attention/turboquant/standalone/),
// so the schema a caller sees does not depend on which one was loaded. Both are
// fragments, so either library can register them next to other _C_ascend ops -- but
// not both into one process: the second would redefine the same schemas.
//
// These are static registrations: include this from exactly one translation unit
// per library, after turboquant_torch_adpt.h. VLLM_ENABLE_TURBOQUANT_CUBE adds the
// kv4fp8 Cube operators, which exist only in an Ascend 950 build.

#pragma once

#include <torch/library.h>

#include "turboquant_torch_adpt.h"

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    ops.def(
        "npu_turboquant_reshape_and_cache(Tensor key, "
        "                                 Tensor value, "
        "                                 Tensor! key_cache, "
        "                                 Tensor! value_cache, "
        "                                 Tensor! scale_cache, "
        "                                 Tensor slot_mapping, "
        "                                 Tensor pi_signs, "
        "                                 Tensor codec_tables) -> ()");
    ops.impl("npu_turboquant_reshape_and_cache", c10::kPrivateUse1,
             &vllm_ascend::npu_turboquant_reshape_and_cache);

    ops.def(
        "npu_turboquant_rotate_q(Tensor query, "
        "                        Tensor pi_signs, "
        "                        Tensor codec_tables, "
        "                        Tensor hadamard16, "
        "                        Tensor! query_rot) -> ()");
    ops.impl("npu_turboquant_rotate_q", c10::kPrivateUse1,
             &vllm_ascend::npu_turboquant_rotate_q);

    ops.def(
        "npu_turboquant_paged_attention(Tensor query_rot, "
        "                               Tensor key_cache, "
        "                               Tensor value_cache, "
        "                               Tensor scale_cache, "
        "                               Tensor block_tables, "
        "                               Tensor context_lens, "
        "                               Tensor codec_tables, "
        "                               Tensor! workspace, "
        "                               int num_kv_heads, "
        "                               int num_heads, "
        "                               float scale_value, "
        "                               Tensor! out) -> ()");
    ops.impl("npu_turboquant_paged_attention", c10::kPrivateUse1,
             &vllm_ascend::npu_turboquant_paged_attention);

    ops.def("npu_turboquant_vector_core_num() -> int");
    ops.impl("npu_turboquant_vector_core_num",
             &vllm_ascend::npu_turboquant_vector_core_num);

    ops.def(
        "npu_turboquant_workspace_size(int num_tokens, "
        "                              int num_heads, "
        "                              int head_size, "
        "                              int max_blocks_per_seq, "
        "                              int block_size) -> int");
    ops.impl("npu_turboquant_workspace_size",
             &vllm_ascend::npu_turboquant_workspace_size);
}

#ifdef VLLM_ENABLE_TURBOQUANT_CUBE
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    ops.def(
        "npu_turboquant_cube_reshape_and_cache(Tensor key, "
        "                                      Tensor value, "
        "                                      Tensor! key_cache, "
        "                                      Tensor! value_cache, "
        "                                      Tensor! scale_cache, "
        "                                      Tensor slot_mapping, "
        "                                      Tensor pi_signs, "
        "                                      Tensor codec_tables) -> ()");
    ops.impl("npu_turboquant_cube_reshape_and_cache", c10::kPrivateUse1,
             &vllm_ascend::npu_turboquant_cube_reshape_and_cache);

    ops.def(
        "npu_turboquant_cube_decode(Tensor query, "
        "                           Tensor? gate, "
        "                           Tensor pi_signs, "
        "                           Tensor codec_tables, "
        "                           Tensor hadamard16, "
        "                           Tensor key_cache, "
        "                           Tensor value_cache, "
        "                           Tensor scale_cache, "
        "                           Tensor block_tables, "
        "                           Tensor context_lens, "
        "                           Tensor! workspace, "
        "                           Tensor! query_rot, "
        "                           int num_kv_heads, "
        "                           int num_heads, "
        "                           float scale_value, "
        "                           int output_stage, "
        "                           Tensor! out) -> ()");
    ops.impl("npu_turboquant_cube_decode", c10::kPrivateUse1,
             &vllm_ascend::npu_turboquant_cube_decode);

    ops.def(
        "npu_turboquant_cube_workspace_size(int num_tokens, "
        "                                   int num_heads, "
        "                                   int num_kv_heads, "
        "                                   int head_size, "
        "                                   int max_blocks_per_seq, "
        "                                   int block_size) -> int");
    ops.impl("npu_turboquant_cube_workspace_size",
             &vllm_ascend::npu_turboquant_cube_workspace_size);
}
#endif
