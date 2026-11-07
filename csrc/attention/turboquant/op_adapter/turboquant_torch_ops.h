#pragma once

// Phase 1 schema inventory. Registrations are intentionally deferred: this
// staging build validates the existing kernel C ABIs without a Torch runtime.
namespace vllm_integration_layer {
inline constexpr char kExistingDecodeSymbol[] =
    "turboquant_mm_fused_decode_opt_raw_query_kv4fp8_half";
inline constexpr char kExistingScatterSymbol[] = "turboquant_reshape_scatter_aiv";
inline constexpr char kExistingBatchSymbols[] =
    "turboquant_reshape_and_cache_bf16";
}  // namespace vllm_integration_layer
