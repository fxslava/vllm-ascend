// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include "kernel_operator.h"

// ---------------------------------------------------------------------------
// Target classification, shared by this header and the kernel beside it.
//
// Ascend C predefines __CCE_AICORE__ from --cce-aicore-arch: 220 for the
// Ascend910B1..B4 / 910_93 vector core (dav-c220) and 310 for the Ascend950
// regbase core (dav-c310). Those are the only two core generations this
// operator is registered for, and they do not expose the same vector
// primitives, so the choice is named once here. Every arch-dependent branch
// keys off DSV4_ARCH_C220; its complement is the regbase path.
//
// A third generation must not silently inherit the regbase path, so an
// unexpected __CCE_AICORE__ is a build error rather than a wrong code path.
// Two builds of this source are not device builds and are exempt: the
// host-stub pass, which sees no __CCE_AICORE__ at all, and tikicpulib's CPU
// interpreter under tools/dsv4_moe_runtime/kernel_bringup. Neither emits
// vector code, so both take the regbase branch.
// ---------------------------------------------------------------------------
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#define DSV4_ARCH_C220 1
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ != 310 && \
    !(defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1)
#error "dsv4_moe_expert supports __CCE_AICORE__ 220 (Ascend910B) and 310 (Ascend950) only"
#endif
#ifndef DSV4_ARCH_C220
#define DSV4_ARCH_C220 0
#endif

namespace Dsv4MoeExpertOp {

// Generate 0..count-1 in integer lanes. The callers use 8..256 lanes in
// multiples of eight. CANN 8's arch220 CreateVecIndex seeds the first block
// through scalar float SetValue; the CANN 8.0.0 device build leaves that block
// stale in CAModel. Subsequent blocks then propagate the stale seed. Keep the
// whole construction on PIPE_V, including the first eight lanes.
__aicore__ inline void CreateGatherIndices(const AscendC::LocalTensor<int32_t>& offsets, uint32_t count)
{
    using namespace AscendC;
#if DSV4_ARCH_C220
    constexpr uint32_t LANES_PER_REPEAT = 64;
    constexpr uint32_t INDEX_BITS = 6;
    constexpr uint64_t BIT_MASKS[INDEX_BITS] = {
        0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL,
        0xF0F0F0F0F0F0F0F0ULL, 0xFF00FF00FF00FF00ULL,
        0xFFFF0000FFFF0000ULL, 0xFFFFFFFF00000000ULL
    };
    const uint8_t repeats = static_cast<uint8_t>(count / LANES_PER_REPEAT);
    const uint32_t tail = count % LANES_PER_REPEAT;
    Duplicate(offsets, static_cast<int32_t>(0), count);
    PipeBarrier<PIPE_V>();
    for (uint32_t bit = 0; bit < INDEX_BITS; ++bit) {
        // A scalar uint64_t mask selects a contiguous lane count. The array
        // overload selects individual bits; 32-bit lanes use its first word.
        uint64_t fullMask[2] = {BIT_MASKS[bit], 0};
        uint64_t tailMask[2] = {BIT_MASKS[bit] & ((uint64_t{1} << tail) - 1), 0};
        const int32_t value = static_cast<int32_t>(1U << bit);
        if (repeats != 0) {
            Adds(offsets, offsets, value, fullMask, repeats, {1, 1, 8, 8});
        }
        if (tailMask[0] != 0) {
            const uint32_t base = repeats * LANES_PER_REPEAT;
            Adds(offsets[base], offsets[base], value, tailMask, 1, {1, 1, 8, 8});
        }
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t base = LANES_PER_REPEAT; base < count; base += LANES_PER_REPEAT) {
        const uint32_t remaining = count - base;
        const uint32_t active = remaining < LANES_PER_REPEAT ? remaining : LANES_PER_REPEAT;
        Adds(offsets[base], offsets[base], static_cast<int32_t>(base), active);
    }
#else
    CreateVecIndex(offsets, static_cast<int32_t>(0), count);
#endif
}

// CANN 8 arch220 calcount And/Or issue 16-bit instructions even for int32
// tensors. Cover BOTH halves of every word; no floating-point conversion.
__aicore__ inline void BitAnd32(const AscendC::LocalTensor<int32_t>& dst,
                              const AscendC::LocalTensor<int32_t>& lhs,
                              const AscendC::LocalTensor<int32_t>& rhs, uint32_t count)
{
#if DSV4_ARCH_C220
    AscendC::And(dst.ReinterpretCast<uint16_t>(), lhs.ReinterpretCast<uint16_t>(),
                 rhs.ReinterpretCast<uint16_t>(), count * 2);
#else
    AscendC::And(dst, lhs, rhs, count);
#endif
}

__aicore__ inline void BitOr32(const AscendC::LocalTensor<int32_t>& dst,
                             const AscendC::LocalTensor<int32_t>& lhs,
                             const AscendC::LocalTensor<int32_t>& rhs, uint32_t count)
{
#if DSV4_ARCH_C220
    AscendC::Or(dst.ReinterpretCast<uint16_t>(), lhs.ReinterpretCast<uint16_t>(),
                rhs.ReinterpretCast<uint16_t>(), count * 2);
#else
    AscendC::Or(dst, lhs, rhs, count);
#endif
}

#if DSV4_ARCH_C220
// Expand bytes using only integer lanes. Gather duplicates each packed word
// four times, then masked shifts align its four bytes with their output lanes.
// Callers provide separate dst/scratch tensors, count divisible by eight,
// and a 32-byte-aligned source containing at least count bytes.
__aicore__ inline void WidenBytes32(const AscendC::LocalTensor<int32_t>& dst,
                                  const AscendC::LocalTensor<uint8_t>& src,
                                  const AscendC::LocalTensor<int32_t>& scratch, uint32_t count)
{
    using namespace AscendC;
    constexpr uint32_t LANES_PER_REPEAT = 64;
    constexpr uint32_t BYTE_BITS = 8;
    constexpr uint32_t WORD_BYTES = sizeof(int32_t);
    constexpr uint64_t BYTE_LANE_MASKS[WORD_BYTES - 1] = {
        0x2222222222222222ULL, 0x4444444444444444ULL, 0x8888888888888888ULL
    };
    CreateGatherIndices(scratch, count);
    ShiftRight(scratch, scratch, 2, count);
    PipeBarrier<PIPE_V>();
    ShiftLeft(scratch, scratch, 2, count);
    PipeBarrier<PIPE_V>();
    Gather(dst, src.ReinterpretCast<int32_t>(), scratch.ReinterpretCast<uint32_t>(), 0, count);
    PipeBarrier<PIPE_V>();
    const uint8_t repeats = static_cast<uint8_t>(count / LANES_PER_REPEAT);
    const uint32_t tail = count % LANES_PER_REPEAT;
    for (uint32_t byte = 1; byte < WORD_BYTES; ++byte) {
        uint64_t mask[2] = {BYTE_LANE_MASKS[byte - 1], 0};
        uint64_t tailMask[2] = {mask[0] & ((uint64_t{1} << tail) - 1), 0};
        const int32_t shift = static_cast<int32_t>(byte * BYTE_BITS);
        if (repeats != 0) {
            ShiftRight(dst, dst, shift, mask, repeats, {1, 1, 8, 8});
        }
        if (tailMask[0] != 0) {
            const uint32_t base = repeats * LANES_PER_REPEAT;
            ShiftRight(dst[base], dst[base], shift, tailMask, 1, {1, 1, 8, 8});
        }
        PipeBarrier<PIPE_V>();
    }
    Duplicate(scratch, int32_t{0xFF}, count);
    PipeBarrier<PIPE_V>();
    BitAnd32(dst, dst, scratch, count);
    PipeBarrier<PIPE_V>();
}
#endif

} // namespace Dsv4MoeExpertOp
