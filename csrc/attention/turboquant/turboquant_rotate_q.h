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

#ifndef VLLM_ASCEND_TURBOQUANT_ROTATE_Q_H
#define VLLM_ASCEND_TURBOQUANT_ROTATE_Q_H

#include <cstdint>

namespace vllm_ascend {
namespace turboquant {

constexpr int64_t kRotateQTile = 16;

constexpr int64_t kRotateQMaxChunkElements = 4096;

constexpr uint16_t kRotateQHalfOne = 0x3C00;
constexpr uint16_t kRotateQHalfMinusOne = 0xBC00;

enum RotateQVariant : uint32_t {
    kHiLo = 0x1u,
    kDualDst = 0x2u,
};

struct RotateQPlan {
    bool use_cube = false;
    uint32_t block_dim = 0;
    uint32_t vectors_per_block = 0;
    uint32_t vectors_per_chunk = 0;
    uint32_t variant = 0;
};

inline RotateQPlan PlanRotateQ(int64_t num_vectors, int64_t head_size, int64_t core_num, bool input_exact_in_half)
{
    RotateQPlan plan;
    if (num_vectors <= 0 || head_size <= 0) {
        return plan;
    }
    if (core_num < 1) {
        core_num = 1;
    }

    const int64_t fits = kRotateQMaxChunkElements / head_size;

    if (num_vectors >= kRotateQTile && num_vectors % kRotateQTile == 0 && fits >= 2) {
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
    }

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

constexpr int64_t kRotateQH16Elements = kRotateQTile * kRotateQTile;

}
}

#endif
