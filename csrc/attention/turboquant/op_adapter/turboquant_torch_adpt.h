#pragma once

// Phase 1 declaration surface. The standalone NPU tests call the C launchers
// directly and do not include ATen or require PyTorch on the build host.
#include <cstdint>

#include "../op_kernel/common/types.h"
#include "../include/turboquant_fused_decode_opt.hpp"

namespace vllm_ascend {

void turboquant_mm_fused_decode_opt_raw_query_impl(
    int32_t mode, AscendType type, void* stream, uint32_t block_dim,
    void* query, void* pi_signs, void* rot_tables, void* h16, void* gate,
    void* query_rot, void* key_cache, void* value_cache, void* scale_cache,
    void* block_tables, void* context_lens, void* mode_tables, void* workspace,
    void* output, void* tiling, uint32_t num_tokens, uint32_t num_heads, uint32_t num_kv_heads,
    uint32_t head_size, uint32_t block_size, uint32_t max_blocks_per_seq,
    uint32_t num_splits, uint32_t heads_per_task, uint32_t tasks_per_block,
    uint32_t reduce_tasks_per_block, uint32_t prologue_vectors_per_block,
    uint32_t prologue_cube_chunk_vectors, uint32_t output_stage,
    uint32_t fused_context_limit, float scale, float inv_sqrt_len);

}  // namespace vllm_ascend

extern "C" {
void launch_fused_decode_opt_raw_query(
    void* stream, const ascend_micro::fused_decode_opt::Launch* config,
    const ascend_micro::fused_decode_opt::RawQueryBuffers* buffers);
void launch_fused_decode_opt(
    void* stream, const ascend_micro::fused_decode_opt::Launch* config,
    const ascend_micro::fused_decode_opt::Buffers* buffers);
void launch_turboquant_reshape_and_cache_bf16(
    void* stream, const uint16_t* key, const uint16_t* value, int8_t* key_cache,
    int8_t* value_cache, float* scale_cache, const int32_t* slot_mapping,
    const float* pi_signs, const int32_t* tables, uint32_t num_tokens,
    uint32_t num_kv_heads, uint32_t head_size, uint32_t block_size,
    uint32_t num_blocks, uint32_t tokens_per_core, float inv_sqrt_len,
    bool reuse_mode4_flags);
void launch_turboquant_reshape_scatter_aiv(
    void* stream, uint32_t block_dim, uint16_t* key, uint16_t* value,
    int8_t* key_cache, int8_t* value_cache, float* scale_cache,
    int32_t* slot_mapping, float* pi_signs, int32_t* cu_seqlens,
    int32_t* launch_probe, uint32_t num_requests, uint32_t num_kv_heads,
    uint32_t head_size, uint32_t block_size, uint32_t num_blocks,
    uint32_t item_begin, uint32_t item_count, float inv_sqrt_len);
}
