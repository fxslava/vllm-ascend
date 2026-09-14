/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * The contract between npu_turboquant_rotate_q's host planner and its kernels.
 *
 * Everything here is shared by three callers that must not disagree: the torch
 * operator in turboquant_torch_adpt.h, the torch-free harness in
 * csrc/tests/common/turboquant_launch.cpp, and the kernels themselves in
 * turboquant_rotate_q.cpp. A disagreement about the chunk rule would show up as
 * a fidelity failure and be read as a kernel bug.
 *
 * No STL and no Ascend C here on purpose: the device compile of
 * turboquant_rotate_q.cpp includes this file too.
 */

#ifndef VLLM_ASCEND_TURBOQUANT_ROTATE_Q_H
#define VLLM_ASCEND_TURBOQUANT_ROTATE_Q_H

#include <cstdint>

namespace vllm_ascend {
namespace turboquant {

// The Cube's fractal quantum, the order of the constant matrix the Mmad
// multiplies by, and the vector count the Cube path is gated on. Sylvester makes
// any power-of-two head size a multiple of it, so the tile never needs column
// padding.
constexpr int64_t kRotateQTile = 16;

// Elements one chunk of the Cube path may hold. Its UB working set is 24 bytes
// per element -- scalar_t in, fp32 in, fp16 cast, and a two-slot fp32 product
// and ping-pong -- so this is 96 KB of the subcore's 248 KB. It is a constant
// rather than a computed fit because a wrong fit shows up as InitBuffer handing
// back a base of 0 and stores decoding as DDR, which does not fault.
constexpr int64_t kRotateQMaxChunkElements = 4096;

// fp16 bit patterns for the only two values H_16 contains. Both are exact, so
// the Cube stage introduces no error of its own.
constexpr uint16_t kRotateQHalfOne = 0x3C00;
constexpr uint16_t kRotateQHalfMinusOne = 0xBC00;

// Variant bits, mirrored by the kernel.
enum RotateQVariant : uint32_t {
    // Two Mmads accumulating into one L0C, x = hi + lo, both halves fp16. The
    // host sets it whenever the query's element type is not exactly
    // representable in half after the sign multiply -- which is every type but
    // half itself. See the precision note in turboquant_rotate_q.cpp.
    kHiLo = 0x1u,
    // Fixpipe dualDstCtl = 0b01: half the product into each subcore's UB. The
    // kernel ignores it when a chunk holds an odd number of vectors, because the
    // M split would then land inside a vector's tile.
    kDualDst = 0x2u,
};

/*
 * How one rotate_q launch is shaped. Every field is a pure function of the
 * batch, the head size and the core count.
 *
 * The two divisibility guarantees the Cube kernel relies on and does not check:
 *
 *   vectors_per_block divides num_vectors      -- every block's share is full
 *   vectors_per_chunk divides vectors_per_block -- no chunk is short
 *
 * Together they are what keeps the Mmad's m constant across chunks, which is
 * what lets the L1 and L0 buffers be sized once in Init.
 */
struct RotateQPlan {
    bool use_cube = false;
    uint32_t block_dim = 0;
    uint32_t vectors_per_block = 0;
    uint32_t vectors_per_chunk = 0;
    uint32_t variant = 0;
};

/*
 * num_vectors        B * H_Q, i.e. num_tokens * num_heads
 * head_size          a power of two; the Cube path wants it >= 64 so that
 *                    R = head_size / 16 is at least 4
 * core_num           MIX blocks available, i.e. AI cube cores. arch35 is
 *                    1 AIC : 2 AIV, so this is the vector core count halved.
 * input_exact_in_half  true when the query's element type survives a cast to
 *                    half exactly, which decides kHiLo
 *
 * The Cube gate is `num_vectors >= 16 and num_vectors % 16 == 0`. Exact
 * divisibility rather than a threshold: a block share that is not a whole number
 * of tiles would need a short final Mmad, and a short Mmad is the one piece of
 * the spike that was never run.
 */
inline RotateQPlan PlanRotateQ(int64_t num_vectors, int64_t head_size, int64_t core_num, bool input_exact_in_half)
{
    RotateQPlan plan;
    if (num_vectors <= 0 || head_size <= 0) {
        return plan;
    }
    if (core_num < 1) {
        core_num = 1;
    }

    // Vectors one chunk can hold at this head size: 16 at D = 256, 32 at 128,
    // 8 at 512.
    const int64_t fits = kRotateQMaxChunkElements / head_size;

    if (num_vectors >= kRotateQTile && num_vectors % kRotateQTile == 0 && fits >= 2) {
        // The smallest whole number of tiles that divides the batch and still
        // keeps the grid inside the core count. Smallest, not largest: spreading
        // the batch over more blocks is what shortens the launch, and the tile
        // floor is what keeps each block dense enough for the Cube form to beat
        // the vector one.
        int64_t vectors_per_block = num_vectors;
        for (int64_t tile = kRotateQTile; tile <= num_vectors; tile += kRotateQTile) {
            if (num_vectors % tile != 0) {
                continue;
            }
            if (num_vectors / tile <= core_num) {
                vectors_per_block = tile;
                break;
            }
        }

        // The chunk: as large as UB allows, even so the dual-destination Fixpipe
        // applies, and an exact divisor of the block's share so no chunk is
        // short.
        int64_t vectors_per_chunk = fits < vectors_per_block ? fits : vectors_per_block;
        if (vectors_per_chunk % 2 != 0) {
            --vectors_per_chunk;
        }
        while (vectors_per_chunk >= 2 && vectors_per_block % vectors_per_chunk != 0) {
            vectors_per_chunk -= 2;
        }

        if (vectors_per_chunk >= 2) {
            plan.use_cube = true;
            plan.vectors_per_block = static_cast<uint32_t>(vectors_per_block);
            plan.vectors_per_chunk = static_cast<uint32_t>(vectors_per_chunk);
            plan.block_dim = static_cast<uint32_t>(num_vectors / vectors_per_block);
            plan.variant = static_cast<uint32_t>(kDualDst);
            if (!input_exact_in_half) {
                plan.variant |= static_cast<uint32_t>(kHiLo);
            }
            return plan;
        }
        // No even chunk divides the share. vectors_per_block is a multiple of 16
        // and 2 divides 16, so this is unreachable at fits >= 2; it is here so a
        // future tile size cannot make the fall-through silently wrong.
    }

    // The vector path. Two vectors per block so both subcores have work, spread
    // over as many blocks as the batch and the core count allow.
    int64_t vectors_per_block = (num_vectors + core_num - 1) / core_num;
    if (vectors_per_block < 2) {
        vectors_per_block = 2;
    }
    plan.use_cube = false;
    plan.vectors_per_block = static_cast<uint32_t>(vectors_per_block);
    plan.vectors_per_chunk = 1;
    plan.block_dim = static_cast<uint32_t>((num_vectors + vectors_per_block - 1) / vectors_per_block);
    plan.variant = 0;
    return plan;
}

/*
 * H_16[i][j] = (-1)^popcount(i & j): Sylvester's construction, symmetric, so the
 * [n, k] image the Cube loads and the [k, n] image the product needs are the
 * same bytes. Row major, which for a 16-column half operand is also NZ.
 *
 * Fills kRotateQTile * kRotateQTile uint16_t. A raw pointer rather than a
 * container so the one definition serves the torch operator, the torch-free
 * harness and any host stub the device build generates.
 */
inline void FillHadamard16Half(uint16_t *out)
{
    for (int64_t i = 0; i < kRotateQTile; ++i) {
        for (int64_t j = 0; j < kRotateQTile; ++j) {
            unsigned bits = static_cast<unsigned>(i & j);
            int parity = 0;
            while (bits != 0) {
                parity ^= static_cast<int>(bits & 1u);
                bits >>= 1;
            }
            out[i * kRotateQTile + j] = parity ? kRotateQHalfMinusOne : kRotateQHalfOne;
        }
    }
}

// Elements of the H_16 constant tensor the operator hands the kernel.
constexpr int64_t kRotateQH16Elements = kRotateQTile * kRotateQTile;

}  // namespace turboquant
}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_TURBOQUANT_ROTATE_Q_H
