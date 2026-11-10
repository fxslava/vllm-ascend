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

#pragma once
// The golden image of one turboquant_reshape_scatter_aiv launch: what every nibble and every
// scale float of the physical cache must hold afterwards, and which lanes the launch must have
// left untouched.
//
// THE READ-MODIFY-WRITE ALWAYS PRESERVES. A work item owns 1..R tokens of one (block, kv head)
// strip; every other lane of that 2048-byte strip goes back unchanged. That is not defensive --
// in decode a strip accumulates over R successive steps, so zeroing the unowned lanes would
// destroy the previous tokens -- and it is what makes the canary protocol strong, because an
// untouched lane must still read back as canary.
//
// BIT EQUALITY, NOT A TOLERANCE. The fixture is integer-valued in BF16, so every intermediate of
// the rotation is an exact integer in FP32, the scale is a power of two extracted by integer ops
// on the FP32 bit pattern, and the single round-to-nearest-even is the only rounding in the
// path. See tests/common/turboquant_golden_oracle.hpp for why that holds.

#include "rotate_butterfly_oracle.hpp"
#include "turboquant_golden_oracle.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace tq_scatter {

// The strip layout, the geometry and the bit helpers are single-sourced in
// tests/common/turboquant_test_common.hpp, which takes R and the nibble packing from the
// production contract in turboquant_layout.h. This header adds only what is specific to the
// scatter writer: the fixture, the per-row reference and the launch plan.
using tq_test::kNibbleMax;
using tq_test::kNibbleMin;
using tq_test::kPlanes;
using tq_test::kStripHeight;
using tq_test::Shape;
using tq_test::bf16_is_exact;
using tq_test::bf16_to_float;
using tq_test::float_to_bf16_rne;
using tq_test::key_nibble;
using tq_test::read_nibble;
using tq_test::scale_index;
using tq_test::strip_byte_offset;
using tq_test::value_nibble;

inline void write_nibble(std::vector<std::int8_t>& cache, std::size_t nibble, int value)
{
    tq_test::require(nibble / 2 < cache.size(), "scatter reference wrote past the cache image");
    tq_test::write_nibble(cache.data(), nibble, value);
}

// -------------------------------------------------------------- the fixture
// Integer-valued BF16, distinct per (plane, head, token, channel) and per
// request, so a work item that read another item's rows cannot produce the
// right answer. `zero_token` makes one token's K and V rows identically zero,
// which is the only way to reach the absmax == 0 branch -- the case that never
// arises with an integer fixture and arises immediately in production.
inline float activation(int request, int plane, int head, int token, int channel,
                        int zero_token)
{
    if (token == zero_token) return 0.0f;
    const int rank = plane * 8 + head;
    const std::uint32_t h = tq_golden::mix(
        0x51ED2701u ^ (static_cast<std::uint32_t>(request) * 0x9E3779B9u) ^
        (static_cast<std::uint32_t>(rank) * 0x85EBCA6Bu) ^
        (static_cast<std::uint32_t>(token) * 0xC2B2AE35u) ^
        (static_cast<std::uint32_t>(channel) * 0x27D4EB2Du));
    const int n = static_cast<int>(h % 25u) - 12;
    return static_cast<float>(n * (1 + rank % 3));
}

// ------------------------------------------------------- the per-row reference
struct RowRef {
    std::vector<float> rotated;  // head_size, Pi . x with no 1/16
    std::vector<int> q;          // head_size, the INT4 codes
    float absmax = 0.0f;
    int exponent = 0;
    float scale = 0.0f;
    bool zero_row = false;
    bool clamped = false;
};

// One (token, head, plane) row: the rotation and the quantiser.
//
// The quantiser, restated so the device's bit arithmetic has something to be
// compared against rather than described by. For 2^e <= absmax < 2^(e+1):
//   exp_bits = bits(absmax) & 0x7F800000  -> 2^e
//   pow      = 2^(e-2)                    -> the stored scale, times invSqrtLen
//   inv      = 2^(2-e)                    -> the exact reciprocal
//   q        = min(rint_even(y * inv), 7)      clamped from above only
// and absmax == 0 takes the explicit branch scale = 1.0f, q = 0 (RFC I15):
// the exponent field of FP32 zero is 0, so 2^(e-2) would be built from a
// borrowed exponent and the stored scale would be garbage.
inline RowRef quantise_row(const Shape& shape, const std::vector<float>& x,
                           const std::vector<float>& signs)
{
    RowRef out;
    out.rotated = rot_oracle::staged(rot_oracle::Direction::Ingress, x.data(), signs.data());
    out.q.assign(static_cast<std::size_t>(shape.head_size), 0);

    float absmax = 0.0f;
    for (float value : out.rotated) absmax = std::fmax(absmax, std::fabs(value));
    out.absmax = absmax;
    if (absmax == 0.0f) {
        out.zero_row = true;
        out.scale = 1.0f;
        out.exponent = 0;
        return out;
    }
    const int e = std::ilogb(absmax);
    out.exponent = e;
    out.scale = std::ldexp(1.0f, e - 2) * shape.inv_sqrt_len();
    for (int c = 0; c < shape.head_size; ++c) {
        // y * 2^(2-e) is exact: y is an integer of magnitude below 2^17 and
        // the factor is a power of two, so nearbyint (FE_TONEAREST, ties to
        // even) is the single rounding the whole path performs.
        const double scaled = std::ldexp(static_cast<double>(out.rotated[c]), 2 - e);
        long long v = static_cast<long long>(std::nearbyint(scaled));
        if (v > kNibbleMax) {
            v = kNibbleMax;
            out.clamped = true;
        }
        out.q[static_cast<std::size_t>(c)] = static_cast<int>(v);
    }
    return out;
}

// ---------------------------------------------------------- packing helpers


// -------------------------------------------------------------- the launch
// One token of one request: where it goes and what it carries.
struct TokenPlan {
    int request = 0;
    int token_index = 0;  // index into the flat token list / slotMapping
    int slot = 0;         // slotMapping value; < 0 means a padded token
    int block = 0;
    int token_in_block = 0;
    int strip = 0;
};

inline TokenPlan plan_token(const Shape& shape, int request, int token_index, int slot)
{
    TokenPlan plan;
    plan.request = request;
    plan.token_index = token_index;
    plan.slot = slot;
    if (slot < 0) return plan;
    plan.block = slot / shape.block_size;
    plan.token_in_block = slot % shape.block_size;
    plan.strip = plan.token_in_block / kStripHeight;
    return plan;
}

// The whole reference. `prior_key`/`prior_value`/`prior_scale` are the cache
// images BEFORE the launch, so the value plane's read-modify-write can be
// checked for what it PRESERVED as well as for what it wrote -- which is the
// property a zero-fill reference could not express. They are copied, mutated
// where the launch owns a lane, and returned.
struct Reference {
    Shape shape;
    std::vector<std::int8_t> key_cache;
    std::vector<std::int8_t> value_cache;
    std::vector<float> scale_cache;
    // One entry per (token_index, head, plane), in that order.
    std::vector<RowRef> rows;
    // true for every nibble index the launch owns, per plane.
    std::vector<char> key_owned;
    std::vector<char> value_owned;
    std::vector<char> scale_owned;
    int zero_rows = 0;
    int clamped_rows = 0;

    std::size_t row_index(int token_index, int head, int plane) const
    {
        return (static_cast<std::size_t>(token_index) * shape.num_kv_heads + head) * kPlanes +
               plane;
    }
};

// `head_filter` is the one KV head the launch owns, or -1 for all of them. A
// head the launch did not run must come back looking exactly like a head
// nobody wrote, so the single-head canary protocol needs the reference to
// agree about that rather than to be corrected afterwards.
inline Reference compute_reference(const Shape& shape, const std::vector<TokenPlan>& tokens,
                                   const std::vector<float>& signs, int zero_token,
                                   const std::vector<std::int8_t>& prior_key,
                                   const std::vector<std::int8_t>& prior_value,
                                   const std::vector<float>& prior_scale,
                                   int head_filter = -1)
{
    Reference ref;
    ref.shape = shape;
    ref.key_cache = prior_key;
    ref.value_cache = prior_value;
    ref.scale_cache = prior_scale;
    ref.key_owned.assign(shape.cache_bytes() * 2, 0);
    ref.value_owned.assign(shape.cache_bytes() * 2, 0);
    ref.scale_owned.assign(shape.scale_floats(), 0);
    ref.rows.resize(tokens.size() * static_cast<std::size_t>(shape.num_kv_heads) * kPlanes);

    std::vector<float> x(static_cast<std::size_t>(shape.head_size));
    for (const TokenPlan& plan : tokens) {
        if (plan.slot < 0) continue;
        for (int head = 0; head < shape.num_kv_heads; ++head) {
            if (head_filter >= 0 && head != head_filter) continue;
            for (int plane = 0; plane < kPlanes; ++plane) {
                for (int c = 0; c < shape.head_size; ++c) {
                    const float raw = activation(plan.request, plane, head, plan.token_index, c,
                                                 zero_token);
                    // Through BF16 and back, because that is the operand the
                    // device reads.
                    x[static_cast<std::size_t>(c)] = bf16_to_float(float_to_bf16_rne(raw));
                }
                RowRef row = quantise_row(shape, x, signs);
                if (row.zero_row) ++ref.zero_rows;
                if (row.clamped) ++ref.clamped_rows;
                for (int c = 0; c < shape.head_size; ++c) {
                    const std::size_t nibble =
                        plane == 0 ? key_nibble(shape, plan.block, head, plan.token_in_block, c)
                                   : value_nibble(shape, plan.block, head, plan.token_in_block, c);
                    auto& cache = plane == 0 ? ref.key_cache : ref.value_cache;
                    auto& owned = plane == 0 ? ref.key_owned : ref.value_owned;
                    write_nibble(cache, nibble, row.q[static_cast<std::size_t>(c)]);
                    owned[nibble] = 1;
                }
                const std::size_t si =
                    scale_index(shape, plane, plan.block, head, plan.token_in_block);
                ref.scale_cache[si] = row.scale;
                ref.scale_owned[si] = 1;
                ref.rows[ref.row_index(plan.token_index, head, plane)] = std::move(row);
            }
        }
    }
    return ref;
}

}  // namespace tq_scatter
