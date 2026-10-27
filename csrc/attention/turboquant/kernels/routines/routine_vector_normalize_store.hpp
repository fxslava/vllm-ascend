#pragma once

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "kernel_operator.h"
#endif

namespace ascend_micro {

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#define ASCEND_MICRO_NORMALIZE_INLINE __aicore__ inline
#else
#define ASCEND_MICRO_NORMALIZE_INLINE inline
#endif

// Separate asynchronous leaves permit the caller to enforce V->MTE3 readiness.
// count must be a multiple of 16 so FP16/BF16 DMA length is 32-byte aligned.
template <class Engine>
ASCEND_MICRO_NORMALIZE_INLINE void vector_normalize(Engine& engine, std::uint32_t count)
{
    engine.broadcast_denominator(count);
    engine.divide(count);
    engine.cast_output(count);
}

template <class Engine>
ASCEND_MICRO_NORMALIZE_INLINE void vector_store(Engine& engine, std::uint32_t count)
{
    engine.mte3_store(count);
}

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)

constexpr AscendC::DivConfig kPreciseNormalizeDiv = {
    AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};

template <class Output>
struct NativeNormalizeStoreEngine {
    AscendC::LocalTensor<float> acc;
    AscendC::LocalTensor<float> l;
    AscendC::LocalTensor<float> l_broadcast;
    AscendC::LocalTensor<float> normalized;
    AscendC::LocalTensor<Output> ub_output;
    AscendC::GlobalTensor<Output> gm_output;

    // Duplicate's tensor-source form broadcasts l[0] to count lanes. Unlike
    // the online-softmax stage, l is not guaranteed replicated here and count
    // may exceed one Brcb repeat (64 FP32), so Duplicate is the right API --
    // but on dav-3510 it lowers to an *unmasked* Reg::LoadAlign of a whole
    // vector register (VECTOR_REG_WIDTH == 256 bytes) from the source base,
    // independent of count. It performs no scalar read and needs no TPipe, so
    // it is not a scalar hazard; it is a footprint hazard. The caller must
    // therefore keep kVecReg bytes of addressable UB behind l. See the
    // static_assert on kLDenominator in kernels/camodel/normalize_store.cpp.
    __aicore__ inline void broadcast_denominator(std::uint32_t count)
    {
        AscendC::Duplicate(l_broadcast, l, count);
    }
    __aicore__ inline void divide(std::uint32_t count)
    {
        AscendC::Div<float, kPreciseNormalizeDiv>(
            normalized, acc, l_broadcast, count);
    }
    __aicore__ inline void cast_output(std::uint32_t count)
    {
        AscendC::Cast(ub_output, normalized,
                      AscendC::RoundMode::CAST_RINT, count);
    }
    __aicore__ inline void mte3_store(std::uint32_t count)
    {
        AscendC::DataCopy(gm_output, ub_output, count);
    }
};

#endif

#undef ASCEND_MICRO_NORMALIZE_INLINE
}  // namespace ascend_micro
