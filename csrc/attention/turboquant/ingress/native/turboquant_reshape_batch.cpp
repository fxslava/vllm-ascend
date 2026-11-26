#include "routine_reshape_batch_pipeline.hpp"

#if defined(__CCE_AICORE__) || \
    (defined(ASCEND_MICRO_CANN_LAUNCH) && !defined(TURBOQUANT_LEGACY_CANN_LAUNCH))
#if !defined(__CCE_AICORE__)

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_bf16(

    __gm__ bfloat16_t* __restrict__ key, __gm__ bfloat16_t* __restrict__ value,
    __gm__ std::int8_t* __restrict__ keyCache, __gm__ std::int8_t* __restrict__ valueCache,
    __gm__ float* __restrict__ scaleCache, __gm__ std::int32_t* __restrict__ slotMapping,
    __gm__ float* __restrict__ piSigns, __gm__ std::int32_t* __restrict__ tables,
    uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,
    uint32_t blockSize, uint32_t numBlocks, uint32_t tokensPerCore, float invSqrtLen,
    uint32_t quantMode) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    turboquant_bf16::KernelBody<std::uint32_t>(
        key, value, keyCache, valueCache, scaleCache, slotMapping, piSigns, tables, numTokens,
        numKvHeads, headSize, blockSize, numBlocks, tokensPerCore, invSqrtLen, quantMode);
#else
    (void)key; (void)value; (void)keyCache; (void)valueCache; (void)scaleCache;
    (void)slotMapping; (void)piSigns; (void)tables; (void)numTokens; (void)numKvHeads;
    (void)headSize; (void)blockSize; (void)numBlocks; (void)tokensPerCore; (void)invSqrtLen;
    (void)quantMode;
#endif
}

#if !defined(__CCE_AICORE__)
#pragma clang diagnostic pop
#endif
#endif

#if defined(ASCEND_MICRO_CANN_LAUNCH) && !defined(__CCE_AICORE__)
#include "../../common/turboquant_launch.h"

#include <stdexcept>

#if defined(TURBOQUANT_LEGACY_CANN_LAUNCH)
using BatchLaunchElement = std::uint16_t;
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_bf16(
    __gm__ std::uint16_t*, __gm__ std::uint16_t*, __gm__ std::int8_t*, __gm__ std::int8_t*,
    __gm__ float*, __gm__ std::int32_t*, __gm__ float*, __gm__ std::int32_t*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t);
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_bf16_wide(
    __gm__ std::uint16_t*, __gm__ std::uint16_t*, __gm__ std::int8_t*, __gm__ std::int8_t*,
    __gm__ float*, __gm__ std::int32_t*, __gm__ float*, __gm__ std::int32_t*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t);
#else
using BatchLaunchElement = bfloat16_t;
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_bf16_wide(
    __gm__ bfloat16_t*, __gm__ bfloat16_t*, __gm__ std::int8_t*, __gm__ std::int8_t*,
    __gm__ float*, __gm__ std::int32_t*, __gm__ float*, __gm__ std::int32_t*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t);
#endif

extern "C" void launch_turboquant_reshape_and_cache_bf16(
    void* stream, const std::uint16_t* key, const std::uint16_t* value,
    std::int8_t* key_cache, std::int8_t* value_cache, float* scale_cache,
    const std::int32_t* slot_mapping, const float* pi_signs, const std::int32_t* tables,
    std::uint32_t num_tokens, std::uint32_t num_kv_heads, std::uint32_t head_size,
    std::uint32_t block_size, std::uint32_t num_blocks, std::uint32_t tokens_per_core,
    float inv_sqrt_len, std::uint32_t quant_mode) {
    if (num_tokens == 0u) return;
    if (!turboquant_wave::WholeStripShare(tokens_per_core))
        throw std::invalid_argument("reshape_and_cache_bf16: tokensPerCore error");
    if (!turboquant_wave::WholeStrips(num_tokens, tokens_per_core))
        throw std::invalid_argument("reshape_and_cache_bf16: numTokens error");
#if TURBOQUANT_INGRESS_NZ

    if (block_size % turboquant_bf16::kNzTileRows != 0u)
        throw std::invalid_argument("reshape_and_cache_bf16: blockSize must be a multiple of 64 "
                                    "for the NZ-tiled ingress");
    if (num_kv_heads != turboquant_bf16::kHeads || head_size != turboquant_bf16::kHeadSize)
        throw std::invalid_argument("reshape_and_cache_bf16: the NZ-tiled ingress is compiled "
                                    "for 4 kv heads of 256 channels");
#endif
    if (block_size == 0u || num_blocks == 0u)
        throw std::invalid_argument("reshape_and_cache_bf16: cache geometry must be nonzero");
    if (quant_mode > 1u)
        throw std::invalid_argument("reshape_and_cache_bf16: quant_mode is 0 (uniform) or 1 "
                                    "(Lloyd-Max)");
    const std::uint32_t grid = turboquant_wave::TotalStrips(num_tokens, tokens_per_core);
    auto* key_gm = reinterpret_cast<BatchLaunchElement*>(const_cast<std::uint16_t*>(key));
    auto* value_gm = reinterpret_cast<BatchLaunchElement*>(const_cast<std::uint16_t*>(value));
    auto* slots_gm = const_cast<std::int32_t*>(slot_mapping);
    auto* signs_gm = const_cast<float*>(pi_signs);
    auto* tables_gm = const_cast<std::int32_t*>(tables);
    if (turboquant_bf16::FrozenBatchGeometry(num_tokens, num_kv_heads, head_size, block_size,
                                          num_blocks, tokens_per_core)) {
        turboquant_reshape_and_cache_bf16<<<grid, nullptr, stream>>>(
            key_gm, value_gm, key_cache, value_cache, scale_cache, slots_gm, signs_gm, tables_gm,
            num_tokens, num_kv_heads, head_size, block_size, num_blocks, tokens_per_core,
            inv_sqrt_len, quant_mode);
    } else {
        turboquant_reshape_and_cache_bf16_wide<<<grid, nullptr, stream>>>(
            key_gm, value_gm, key_cache, value_cache, scale_cache, slots_gm, signs_gm, tables_gm,
            num_tokens, num_kv_heads, head_size, block_size, num_blocks, tokens_per_core,
            inv_sqrt_len, quant_mode);
    }
}
#endif
