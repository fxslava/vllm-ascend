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

// SIGNED int4 straight to the output type, with no row scale, no mask and no
// intermediate plane the caller has to allocate.
//
// For a caller that stores symmetric int4 (-8..+7, which is what `int4b_t`
// means) and applies the quantisation scale somewhere cheaper -- the decode
// operator folds the key scale into the 16 score columns and the value scale
// into the 16 probabilities rather than multiplying a 4096-entry page -- this
// is one instruction for a 16-bit output and two for FP32.
//
// Which conversions exist is not a matter of taste here
// -----------------------------------------------------
// The 950PR `Cast` table (type_conversion/Cast.md) lists, for a bfloat16_t
// destination, exactly four sources: int4b_t, the two fp4 types, and half.
// **`int16_t -> bfloat16_t` is not among them**, and on dav-3510 that pair
// compiles without a diagnostic and emits NO instructions at all. That cost a
// full debugging cycle: the paged decode operator's score panel came back
// 256/256 zero on CAModel while every address, every flag and every other
// instruction in the trace was correct, and what finally located it was a
// full opcode census of the staging core's dump -- three vector stores and one
// conversion where the four-stage unpack should have left four and two. The
// missing store was the key page itself.
//
// So this path uses only listed pairs:
//
//   int4b_t -> bfloat16_t   CAST_NONE   direct, and likewise int4b_t -> half
//   int4b_t -> int16_t      CAST_NONE   then
//   int16_t -> float        CAST_NONE   because int4b_t -> float is not listed
//
// The unsigned-nibble path above is unaffected and keeps its own semantics:
// it masks to 0..15 and scales, which is what its callers were verified with.
template <class Engine>
ASCEND_MICRO_DEQUANT_INLINE void turboquant_dequant4_signed(Engine& engine,
                                                            std::uint32_t count)
{
    engine.cast_signed(count);
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

    // The signed nibble value in the output type, over listed Cast pairs only.
    // A 16-bit float output is one instruction; FP32 goes through int16_t,
    // because int4b_t -> float is not a listed pair. `signed_nibbles` is used
    // only on the FP32 path, so a 16-bit caller need not allocate it.
    __aicore__ inline void cast_signed(std::uint32_t count)
    {
        auto nibbles = packed.template ReinterpretCast<AscendC::int4b_t>();
        if constexpr (sizeof(Output) == sizeof(float)) {
            AscendC::Cast(signed_nibbles, nibbles,
                          AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(output, signed_nibbles,
                          AscendC::RoundMode::CAST_NONE, count);
        } else {
            AscendC::Cast(output, nibbles, AscendC::RoundMode::CAST_NONE, count);
        }
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
