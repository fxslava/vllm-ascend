/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

// The TurboQuant BF16 ingestion schema, as a TORCH_LIBRARY_FRAGMENT so it
// registers beside the other _C_ascend ops rather than redefining the library.
// Included exactly once, from csrc/torch_binding.cpp.
//
// One operator. The decode side of TurboQuant is not part of this branch: the
// cache this writes is read by nothing in-tree yet, so adding rotate_q, paged
// attention or a Cube decode schema here would register entry points with no
// implementation behind them.

TORCH_LIBRARY_FRAGMENT_EXPAND(CONCAT(_C, _ascend), ops)
{
    // key/value   [num_tokens, num_kv_heads, head_size] bfloat16, contiguous
    // key_cache   [num_blocks, block_size, num_kv_heads, head_size / 2] int8
    // value_cache the same, strip-blocked channel-major within a page
    // scale_cache [num_blocks, block_size, scale_slot] fp32, burst-aligned
    // slot_mapping [num_tokens] int32, 16-aligned and consecutive per strip
    // pi_signs    kept in the signature; folded into `tables`, never read
    // tables      16 row-major 32x32 bfloat16 blocks (16384 elements)
    ops.def(
        "npu_turboquant_reshape_and_cache(Tensor key, Tensor value, Tensor! key_cache, "
        "Tensor! value_cache, Tensor! scale_cache, Tensor slot_mapping, Tensor pi_signs, "
        "Tensor tables) -> ()");
    ops.impl("npu_turboquant_reshape_and_cache", c10::kPrivateUse1,
             &vllm_ascend::npu_turboquant_reshape_and_cache);
}
