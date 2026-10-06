// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include "kernel_operator.h"

namespace Dsv4MoeExpertOp {

// Generate 0..count-1 in integer lanes. The callers use 8..256 lanes in
// multiples of eight. CANN 8's arch220 CreateVecIndex seeds the first block
// through scalar float SetValue; the CANN 8.0.0 device build leaves that block
// stale in CAModel. Subsequent blocks then propagate the stale seed. Keep the
// whole construction on PIPE_V, including the first eight lanes.
__aicore__ inline void CreateGatherIndices(const AscendC::LocalTensor<int32_t>& offsets, uint32_t count)
{
    using namespace AscendC;
#if __CCE_AICORE__ == 220
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

} // namespace Dsv4MoeExpertOp
