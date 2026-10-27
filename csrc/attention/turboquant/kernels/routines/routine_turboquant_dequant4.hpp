#pragma once

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "kernel_operator.h"
#endif

namespace ascend_micro {

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#define ASCEND_MICRO_DEQUANT_INLINE __aicore__ inline
#else
#define ASCEND_MICRO_DEQUANT_INLINE inline
#endif

// One packed row at a time. The caller supplies a scalar scale for this row,
// ready UB operands, and completion ordering outside this asynchronous leaf.
// count is a multiple of 64: 32 input bytes and 64 output nibbles per block.
constexpr std::uint32_t kDequant4MinElements = 64;
constexpr std::uint32_t kDequant4NibbleMask = 15;
constexpr std::uint32_t kBankSkewBytes = 32;

template <class Engine>
ASCEND_MICRO_DEQUANT_INLINE void turboquant_dequant4(Engine& engine, std::uint32_t count)
{
    engine.cast_packed_nibbles(count);
    engine.mask_unsigned_nibbles(count);
    engine.cast_and_scale(count);
}

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)

template <class Output>
struct NativeDequant4Engine {
    AscendC::LocalTensor<std::int8_t> packed;
    AscendC::LocalTensor<std::int16_t> signed_nibbles;
    AscendC::LocalTensor<std::int16_t> nibble_mask;
    AscendC::LocalTensor<std::int16_t> unsigned_nibbles;
    AscendC::LocalTensor<Output> converted;
    AscendC::LocalTensor<Output> output;
    Output row_scale;

    __aicore__ inline void cast_packed_nibbles(std::uint32_t count)
    {
        auto nibbles = packed.ReinterpretCast<AscendC::int4b_t>();
        AscendC::Cast(signed_nibbles, nibbles,
                      AscendC::RoundMode::CAST_NONE, count);
    }

    __aicore__ inline void mask_unsigned_nibbles(std::uint32_t count)
    {
        AscendC::Duplicate(nibble_mask, std::int16_t{kDequant4NibbleMask}, count);
        AscendC::And(unsigned_nibbles, signed_nibbles, nibble_mask, count);
    }

    __aicore__ inline void cast_and_scale(std::uint32_t count)
    {
        if constexpr (sizeof(Output) == sizeof(float)) {
            AscendC::Cast(converted, unsigned_nibbles,
                          AscendC::RoundMode::CAST_NONE, count);
        } else {
            AscendC::Cast(converted, unsigned_nibbles,
                          AscendC::RoundMode::CAST_RINT, count);
        }
        AscendC::Muls(output, converted, row_scale, count);
    }
};

template <class Output>
__aicore__ inline void turboquant_dequant4_native(
    NativeDequant4Engine<Output>& engine, std::uint32_t count)
{
    turboquant_dequant4(engine, count);
}

#endif

#undef ASCEND_MICRO_DEQUANT_INLINE
}  // namespace ascend_micro
