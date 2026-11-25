#include "torch_bindings_internal.h"

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(turboquant_ops, ops) {
    ops.def(
        "npu_turboquant_reshape_and_cache_a8(Tensor key, Tensor value, Tensor! key_cache, "
        "Tensor! value_cache, Tensor! scale_cache, Tensor slot_mapping, Tensor pi_signs, "
        "Tensor tables, float inv_sqrt_len, int quant_mode) -> ()");
    ops.impl("npu_turboquant_reshape_and_cache_a8", c10::kPrivateUse1,
             &vllm_ascend::turboquant_bindings::npu_turboquant_reshape_and_cache_a8);
    ops.impl("npu_turboquant_reshape_and_cache_a8", c10::kMeta,
             &vllm_ascend::turboquant_bindings::meta::npu_turboquant_reshape_and_cache_a8_meta);

    ops.def(
        "npu_turboquant_reshape_scatter_aiv(Tensor key, Tensor value, Tensor! key_cache, "
        "Tensor! value_cache, Tensor! scale_cache, Tensor slot_mapping, Tensor pi_signs, "
        "float inv_sqrt_len, int quant_mode) -> ()");
    ops.impl("npu_turboquant_reshape_scatter_aiv", c10::kPrivateUse1,
             &vllm_ascend::turboquant_bindings::npu_turboquant_reshape_scatter_aiv);
    ops.impl("npu_turboquant_reshape_scatter_aiv", c10::kMeta,
             &vllm_ascend::turboquant_bindings::meta::npu_turboquant_reshape_scatter_aiv_meta);

    ops.def(
        "npu_turboquant_fused_decode(Tensor query, Tensor pi_signs, Tensor rot_tables, Tensor h16, "
        "Tensor? gate, Tensor key_cache, Tensor value_cache, Tensor scale_cache, Tensor block_tables, "
        "Tensor context_lens, Tensor tiling, Tensor! workspace, Tensor! query_rot, Tensor! output, "
        "int num_heads, int num_kv_heads, int num_splits, "
        "int heads_per_task, int output_stage, float scale, float inv_sqrt_len, "
        "int decode_unpack, int fused_context_limit=4096) -> ()");
    ops.impl("npu_turboquant_fused_decode", c10::kPrivateUse1,
             &vllm_ascend::turboquant_bindings::npu_turboquant_fused_decode);
    ops.impl("npu_turboquant_fused_decode", c10::kMeta,
             &vllm_ascend::turboquant_bindings::meta::npu_turboquant_fused_decode_meta);

    ops.def(
        "npu_turboquant_build_decode_tiling(Tensor context_lens, Tensor! tiling, int num_heads, "
        "int num_kv_heads, int head_size, int block_size, int max_blocks, int num_splits, "
        "int heads_per_task, int fused_context_limit=4096) -> ()");
    ops.impl("npu_turboquant_build_decode_tiling", c10::kPrivateUse1,
             &vllm_ascend::turboquant_bindings::npu_turboquant_build_decode_tiling);
    ops.impl("npu_turboquant_build_decode_tiling", c10::kCPU,
             &vllm_ascend::turboquant_bindings::npu_turboquant_build_decode_tiling);
    ops.impl("npu_turboquant_build_decode_tiling", c10::kMeta,
             &vllm_ascend::turboquant_bindings::meta::npu_turboquant_build_decode_tiling_meta);
}

TORCH_LIBRARY_FRAGMENT(turboquant_e8_ops, ops) {
    ops.def(
        "reader(Tensor query, Tensor key_packets, Tensor value_packets, Tensor packet_offsets, "
        "Tensor! output, Tensor! key_witness, Tensor! value_witness, Tensor! score_witness, "
        "int tiles, int last_rows, float scale) -> ()");
    ops.impl("reader", c10::kPrivateUse1, &vllm_ascend::turboquant_e8::reader);
    ops.impl("reader", c10::kMeta, &vllm_ascend::turboquant_e8::reader_meta);
    ops.def("dequant(Tensor labels, Tensor scales, Tensor! bf16, Tensor! int8) -> ()");
    ops.impl("dequant", c10::kPrivateUse1, &vllm_ascend::turboquant_e8::dequant);
    ops.impl("dequant", c10::kMeta, &vllm_ascend::turboquant_e8::dequant_meta);
}
