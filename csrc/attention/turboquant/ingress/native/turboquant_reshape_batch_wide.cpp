#include "routine_reshape_batch_pipeline.hpp"

#if defined(__CCE_AICORE__) || \
    (defined(ASCEND_MICRO_CANN_LAUNCH) && !defined(TURBOQUANT_LEGACY_CANN_LAUNCH))
#if !defined(__CCE_AICORE__)

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_bf16_wide(

    __gm__ bfloat16_t* __restrict__ key, __gm__ bfloat16_t* __restrict__ value,
    __gm__ std::int8_t* __restrict__ keyCache, __gm__ std::int8_t* __restrict__ valueCache,
    __gm__ float* __restrict__ scaleCache, __gm__ std::int32_t* __restrict__ slotMapping,
    __gm__ float* __restrict__ piSigns, __gm__ std::int32_t* __restrict__ tables,
    uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,
    uint32_t blockSize, uint32_t numBlocks, uint32_t tokensPerCore, float invSqrtLen,
    uint32_t quantMode) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    turboquant_bf16::KernelBody<std::uint64_t>(
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
