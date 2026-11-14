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

// The golden oracle: the reference integer quantiser, the power-of-two scale extractor and the
// FP64 attention reference, in one place so that no runner carries its own copy.
//
// WHY THIS CAN BE BIT-EXACT. The fixture's activations are integers. A BF16 significand is 8
// bits, so integers up to 256 are exactly representable; a product of two is at most 2^16 and
// exact in FP32; and a sum of 256 such products is an integer below 2^24, hence exact in FP32
// WHATEVER ORDER the Cube accumulates in. Downstream, the quantiser is power-of-two: with
// 2^e <= absmax < 2^(e+1) the scale is 2^(e-2) and the code is min(rint(y * 2^(2-e)), 7), and
// 2^(2-e) is a power of two, so the product is exact and round-to-nearest-even on it is exact.
// Every nibble and every stored scale is therefore reproducible here in integer arithmetic,
// which is what lets a runner compare bytes rather than a cosine.
//
// With general, non-integer BF16 activations none of that holds: the Cube's FP32 accumulation
// order is unspecified and no integer oracle can reproduce it. Such a test has to degrade to a
// cosine plus a relative-L2 bound, which no longer detects a wrong-strip read the way nibble
// comparison does. It belongs in a separate, clearly labelled runner and must not replace this.
//
// THE ATTENTION REFERENCE is the other half: given a packed cache and a query it computes
// softmax(q k^T * scale) v in double, dequantising each nibble by its stored scale, so the
// fused decode's fp32 accumulation is compared against a reference whose own rounding is orders
// of magnitude finer.

#ifndef TURBOQUANT_GOLDEN_ORACLE_HPP
#define TURBOQUANT_GOLDEN_ORACLE_HPP

#include "turboquant_test_common.hpp"
#include "csrc/attention/turboquant/op_kernel/common/turboquant_mode.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace tq_golden {

using tq_test::kNibbleMax;
using tq_test::kNibbleMin;
using tq_test::kNibblesPerByte;
using tq_test::kPlanes;
using tq_test::Shape;
using tq_test::float_bits;
using tq_test::require;

// The width of the algorithm's Hadamard group. A resident rotation block is kGroup x kGroup.
constexpr int kGroup = static_cast<int>(vllm_ascend::turboquant::kOperandC0);

// ------------------------------------------------------------- primitives
inline int parity(unsigned value)
{
    int p = 0;
    while (value != 0u) {
        p ^= static_cast<int>(value & 1u);
        value >>= 1;
    }
    return p;
}

// H[i][j] = (-1)^popcount(i & j), the Sylvester Hadamard entry.
inline int hadamard(int i, int j)
{
    return parity(static_cast<unsigned>(i & j)) ? -1 : 1;
}

// The sign diagonal that sits between the two Hadamard factors. Balanced
// enough that a dropped sign multiply cannot be mistaken for a pass, and not a
// simple alternation, which the butterfly's own structure could mask.
inline int pi_sign(int channel)
{
    return ((channel * channel + 3 * channel + 1) % 7 < 3) ? -1 : 1;
}

inline std::uint32_t mix(std::uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

// Round half to even of y / 2^shift, in integers. The kernel reaches the same
// value through one FP32 CAST_RINT of an exactly representable product.
inline long long rint_even_shift(long long y, int shift)
{
    if (shift <= 0) return y << (-shift);
    const long long den = 1LL << shift;
    long long q = y / den;
    const long long rem = y % den;
    if (rem == 0) return q;
    const long long twice = 2 * (rem < 0 ? -rem : rem);
    const bool away = (twice > den) || (twice == den && (q % 2 != 0));
    if (away) q += (y >= 0 ? 1 : -1);
    return q;
}

// ---------------------------------------------------------------- geometry
// The physical geometry is tq_test::Shape. Two derived counts the prefill reference needs:
inline int group_count(const Shape& shape) { return shape.head_size / kGroup; }

// Strips of the launch. The prefill reference quantises one strip per physical page, so a page
// is exactly one strip high and strip s lives on page slot_mapping[s * block_size] / block_size.
inline int strip_count(const Shape& shape) { return shape.num_tokens / shape.block_size; }

// --------------------------------------------------- resident rotation blocks
// `tables` carries the Cube's resident operand: one row-major 32x32 INT8 block
// per (sign, source group) pair, block s * groups + g holding
//
//   T[s][g][c0][i] = (s ? -1 : +1) * piSigns[g*32 + c0] * H32[c0][i].
//
// The kernel multiplies the page's group-g fractal by block
// parity(c1 & g) * groups + g to produce output group c1, so the sign diagonal
// and the outer Hadamard factor both live inside these blocks. Composed, they
// must reproduce the dense matrix this oracle applies:
//
//   M[c1*32 + c0][g*32 + i] = H8[c1][g] * piSigns[g*32 + c0] * H32[c0][i].
inline std::vector<std::int8_t> build_tables(const Shape& shape)
{
    const int groups = group_count(shape);
    std::vector<std::int8_t> tables(
        static_cast<std::size_t>(kPlanes) * groups * kGroup * kGroup);
    for (int s = 0; s < kPlanes; ++s)
        for (int g = 0; g < groups; ++g)
            for (int c0 = 0; c0 < kGroup; ++c0)
                for (int i = 0; i < kGroup; ++i)
                    tables[((static_cast<std::size_t>(s) * groups + g) * kGroup + c0) *
                               kGroup + i] =
                        static_cast<std::int8_t>((s ? -1 : 1) * pi_sign(g * kGroup + c0) *
                                                 hadamard(c0, i));
    return tables;
}

// The per-entry check that ties the device's operand to this oracle's matrix.
// Returns the number of composed entries, or -1 on the first disagreement: a
// wrong sign index or a transposed block still yields *some* orthogonal map,
// and the nibble comparison downstream would then fail without saying why.
inline long long audit_tables(const Shape& shape, const std::int8_t* tables)
{
    const int groups = group_count(shape);
    long long composed = 0;
    for (int c1 = 0; c1 < groups; ++c1)
        for (int g = 0; g < groups; ++g) {
            const int block = parity(static_cast<unsigned>(c1 & g)) * groups + g;
            const std::int8_t* entries =
                tables + static_cast<std::size_t>(block) * kGroup * kGroup;
            for (int c0 = 0; c0 < kGroup; ++c0)
                for (int i = 0; i < kGroup; ++i) {
                    const int want =
                        hadamard(c1, g) * pi_sign(g * kGroup + c0) * hadamard(c0, i);
                    if (entries[c0 * kGroup + i] != want) return -1;
                    ++composed;
                }
        }
    return composed;
}

// ------------------------------------------------------------ the fixture
// One activation generator for both harnesses, so the fixture properties the
// CAModel runner asserts also hold for the device runner.
//
// A structured input is the wrong fixture here: a low-order polynomial in (token, channel)
// makes the rotated panel sparse, so the per-token maxima repeat, the nibbles cluster on 0 and
// +-1 and the INT4 upper clamp is never taken. This is a hashed value in [-12, 12]
// scaled by a per-chunk amplitude ramp, which gives a roughly Gaussian rotated
// panel: all sixteen nibbles are exercised, the per-token maxima are distinct
// and the upper clamp is reached. The strip index enters the hash, so the
// pages differ from each other as well -- a strip that read its neighbour's
// data cannot produce the right answer. The two injected corners put INT8
// -128 and +127 through the path.
inline std::int8_t activation(int strip, int plane, int head, int token, int channel,
                              int num_kv_heads)
{
    const int rank = plane * num_kv_heads + head;
    if (token == 0 && channel == 0) return plane == 0 ? -128 : 127;
    if (token == 1 && channel == 1) return plane == 0 ? 127 : -128;
    const std::uint32_t h =
        mix(0x1234567u ^ (static_cast<std::uint32_t>(strip) * 0x9E3779B9u) ^
            (static_cast<std::uint32_t>(rank) * 0x85EBCA6Bu) ^
            (static_cast<std::uint32_t>(token) * 0xC2B2AE35u) ^
            (static_cast<std::uint32_t>(channel) * 0x27D4EB2Du));
    const int n = static_cast<int>(h % 25u) - 12;
    return static_cast<std::int8_t>(n * (4 + 2 * rank) / 4);
}

// Activation blocks in the kernel's input layout: token-major rows of
// numKvHeads * headSize channels.
inline void build_activations(const Shape& shape, std::vector<std::int8_t>& key,
                              std::vector<std::int8_t>& value)
{
    const std::size_t channels =
        static_cast<std::size_t>(shape.num_kv_heads) * shape.head_size;
    key.assign(static_cast<std::size_t>(shape.num_tokens) * channels, 0);
    value.assign(key.size(), 0);
    for (int t = 0; t < shape.num_tokens; ++t) {
        const int strip = t / shape.block_size;
        const int token = t % shape.block_size;
        for (int head = 0; head < shape.num_kv_heads; ++head)
            for (int c = 0; c < shape.head_size; ++c) {
                const std::size_t offset = static_cast<std::size_t>(t) * channels +
                                           static_cast<std::size_t>(head) * shape.head_size + c;
                key[offset] = activation(strip, 0, head, token, c, shape.num_kv_heads);
                value[offset] = activation(strip, 1, head, token, c, shape.num_kv_heads);
            }
    }
}

// --------------------------------------------------------- the slot mapping
// Strip s occupies physical page 2s + 1, with numBlocks = 2 * strips + 2.
//
// Deliberately NOT the identity mapping. Under slot[t] = t the page index equals the grid block
// index, so a kernel that derived its page from GetBlockIdx() rather than from its own
// slotMapping entry would pass, and a cache sized to exactly the written pages turns an
// out-of-range page into a heap overrun instead of a canary failure. Odd pages only, so every
// even page must still hold its canary afterwards, and adjacent strips land on non-adjacent
// pages.
inline int page_of_strip(int strip) { return 2 * strip + 1; }

inline int fragmented_num_blocks(int strips) { return 2 * strips + 2; }

inline std::vector<std::int32_t> build_slot_mapping(const Shape& shape)
{
    std::vector<std::int32_t> slots(static_cast<std::size_t>(shape.num_tokens));
    for (int t = 0; t < shape.num_tokens; ++t) {
        const int strip = t / shape.block_size;
        const int token = t % shape.block_size;
        slots[static_cast<std::size_t>(t)] =
            page_of_strip(strip) * shape.block_size + token;
    }
    return slots;
}

// The kernel's slot mapping contract: a prefill strip fills one whole physical
// page, so slotMapping[i] == slotMapping[0] + i and slotMapping[0] is a
// multiple of blockSize. Returns the page index, or -1 when the strip violates
// the contract or leaves the cache.
inline int page_of_slots(const Shape& shape, const std::int32_t* slots, int strip)
{
    const std::int32_t* strip_slots = slots + static_cast<std::size_t>(strip) * shape.block_size;
    const std::int32_t first = strip_slots[0];
    if (first < 0 || first % shape.block_size != 0) return -1;
    for (int i = 1; i < shape.block_size; ++i)
        if (strip_slots[i] != first + i) return -1;
    const int page = first / shape.block_size;
    return page < shape.num_blocks ? page : -1;
}

// ------------------------------------------------------- the reference itself
// One (strip, plane, head) panel: the rotation, the quantiser's state and the
// packed page in both layouts. `packed_alt` is the same nibbles in the *other*
// cache layout; nothing compares against it, it exists so a failing page can
// be told apart from a page that is right but packed the wrong way round --
// the one failure mode a byte comparison reports most misleadingly.
struct ChunkRef {
    int strip = 0;
    int plane = 0;
    int head = 0;
    int page = 0;
    std::vector<long long> y;        // blockSize * headSize, the rotated panel
    std::vector<long long> abs_max;  // blockSize
    std::vector<int> exponent;       // blockSize
    std::vector<int> q;              // blockSize * headSize, the INT4 values
    std::vector<float> scale;        // blockSize
    std::vector<std::uint8_t> packed;      // headBytes, this plane's layout
    std::vector<std::uint8_t> packed_alt;  // headBytes, the other layout
    long long chunk_max = 0;
    bool clamped = false;
    int min_q = 0;
    int max_q = 0;
};

struct Reference {
    Shape shape;
    // Strip-major, then plane, then head.
    std::vector<ChunkRef> chunks;
    // Full physical cache images. Only pages with page_written[page] != 0 are
    // comparable: the rest are whatever the harness put there before launch.
    std::vector<std::int8_t> key_cache;
    std::vector<std::int8_t> value_cache;
    std::vector<float> scale_cache;
    std::vector<char> page_written;  // numBlocks
    // -1 when no strip violated the slot mapping contract, else that strip.
    int bad_slot_strip = -1;

    const ChunkRef& chunk(int strip, int plane, int head) const
    {
        return chunks[((static_cast<std::size_t>(strip) * kPlanes) + plane) *
                          shape.num_kv_heads + head];
    }
    // The kernel's chunk numbering -- value heads first, then key heads -- is
    // a property of its four-stage schedule, not of the golden model, so each
    // harness maps it to (plane, head) itself and this header stays free of it.
    std::size_t page_offset(int page) const
    {
        return static_cast<std::size_t>(page) * shape.page_bytes();
    }
    std::size_t head_offset(int page, int head) const
    {
        return page_offset(page) + static_cast<std::size_t>(head) * shape.head_bytes();
    }
    std::size_t scale_offset(int plane, int page, int head) const
    {
        return ((static_cast<std::size_t>(plane) * shape.num_blocks + page) *
                    shape.num_kv_heads + head) * shape.block_size;
    }
};

// The nibble index of (token, channel) inside a (page, head) packed page. The value plane's
// form is the RFC 3.1 strip map specialised to a page exactly one strip high, which is the
// geometry this reference is defined for; compute_oracle_reference refuses anything else.
inline int nibble_index(const Shape& shape, int plane, int token, int channel)
{
    return plane == 0 ? token * shape.head_size + channel
                      : channel * shape.block_size + token;
}

// One panel of the rotation and the quantiser, from the activation block.
inline void build_chunk(const Shape& shape, const std::int8_t* input, int strip, int plane,
                        int head, int page, ChunkRef& out)
{
    const int groups = group_count(shape);
    const int head_size = shape.head_size;
    const int block_size = shape.block_size;
    const std::size_t channels =
        static_cast<std::size_t>(shape.num_kv_heads) * head_size;

    out.strip = strip;
    out.plane = plane;
    out.head = head;
    out.page = page;
    out.y.assign(static_cast<std::size_t>(block_size) * head_size, 0);
    out.abs_max.assign(static_cast<std::size_t>(block_size), 0);
    out.exponent.assign(static_cast<std::size_t>(block_size), 0);
    out.q.assign(out.y.size(), 0);
    out.scale.assign(static_cast<std::size_t>(block_size), 0.0f);
    out.packed.assign(static_cast<std::size_t>(shape.head_bytes()), 0);
    out.packed_alt.assign(out.packed.size(), 0);
    out.chunk_max = 0;
    out.clamped = false;
    out.min_q = kNibbleMax;
    out.max_q = kNibbleMin;

    // Inner factor: z[t][g*32 + o] = sum_i H32[o][i] * x[t][g*32 + i].
    std::vector<long long> z(out.y.size(), 0);
    for (int t = 0; t < block_size; ++t) {
        const std::size_t row =
            (static_cast<std::size_t>(strip) * block_size + t) * channels +
            static_cast<std::size_t>(head) * head_size;
        for (int g = 0; g < groups; ++g)
            for (int o = 0; o < kGroup; ++o) {
                long long acc = 0;
                for (int i = 0; i < kGroup; ++i)
                    acc += static_cast<long long>(hadamard(o, i)) *
                           static_cast<int>(input[row + static_cast<std::size_t>(g) * kGroup + i]);
                z[static_cast<std::size_t>(t) * head_size + g * kGroup + o] = acc;
            }
    }

    // Sign diagonal, then the outer factor over the channel group index.
    for (int t = 0; t < block_size; ++t)
        for (int c1 = 0; c1 < groups; ++c1)
            for (int c0 = 0; c0 < kGroup; ++c0) {
                long long acc = 0;
                for (int g = 0; g < groups; ++g) {
                    const int ch = g * kGroup + c0;
                    acc += static_cast<long long>(hadamard(c1, g)) *
                           static_cast<long long>(pi_sign(ch)) *
                           z[static_cast<std::size_t>(t) * head_size + ch];
                }
                out.y[static_cast<std::size_t>(t) * head_size + c1 * kGroup + c0] = acc;
            }

    for (int t = 0; t < block_size; ++t) {
        long long m = 1;  // the kernel floors absmax at 1.0
        for (int ch = 0; ch < head_size; ++ch) {
            const long long v = out.y[static_cast<std::size_t>(t) * head_size + ch];
            const long long a = v < 0 ? -v : v;
            if (a > m) m = a;
        }
        out.abs_max[static_cast<std::size_t>(t)] = m;
        if (m > out.chunk_max) out.chunk_max = m;
        int e = 0;
        while ((1LL << (e + 1)) <= m) ++e;
        out.exponent[static_cast<std::size_t>(t)] = e;
        out.scale[static_cast<std::size_t>(t)] =
            std::ldexp(1.0f, e - 2) * shape.inv_sqrt_len();
        for (int ch = 0; ch < head_size; ++ch) {
            // |y * 2^(2-e)| < 8 and round-to-nearest-even can reach +8, which
            // INT4 cannot hold, so the kernel clamps from above only.
            long long v = rint_even_shift(out.y[static_cast<std::size_t>(t) * head_size + ch],
                                          e - 2);
            if (v > kNibbleMax) {
                v = kNibbleMax;
                out.clamped = true;
            }
            const int qv = static_cast<int>(v);
            out.q[static_cast<std::size_t>(t) * head_size + ch] = qv;
            out.min_q = std::min(out.min_q, qv);
            out.max_q = std::max(out.max_q, qv);

            const int index = nibble_index(shape, plane, t, ch);
            const int other = nibble_index(shape, 1 - plane, t, ch);
            const auto nib = static_cast<std::uint8_t>(qv & 0xF);
            out.packed[static_cast<std::size_t>(index) / 2] |= static_cast<std::uint8_t>(
                (index % 2 == 0) ? nib : static_cast<std::uint8_t>(nib << 4));
            out.packed_alt[static_cast<std::size_t>(other) / 2] |= static_cast<std::uint8_t>(
                (other % 2 == 0) ? nib : static_cast<std::uint8_t>(nib << 4));
        }
    }
}

// Decode a packed page back into q[token][channel] under the given layout, so
// a mismatch can be reported in quantised values rather than in bytes.
inline void unpack_page(const Shape& shape, int plane, const std::uint8_t* page,
                        std::vector<int>& q)
{
    q.assign(static_cast<std::size_t>(shape.block_size) * shape.head_size, 0);
    for (int t = 0; t < shape.block_size; ++t)
        for (int ch = 0; ch < shape.head_size; ++ch) {
            const int index = nibble_index(shape, plane, t, ch);
            const std::uint8_t byte = page[static_cast<std::size_t>(index) / 2];
            const int raw = (index % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
            q[static_cast<std::size_t>(t) * shape.head_size + ch] = raw >= 8 ? raw - 16 : raw;
        }
}

// The whole reference: every strip's panels, and the physical cache images the
// device must produce. `slot_mapping` is the caller's, so a harness can feed a
// deliberately fragmented or out-of-range mapping and see it reflected here.
//
// `strip_runs`, when given, is one flag per strip saying whether that strip's grid block
// actually executes. A strip that does not run contributes no panels and leaves `page_written`
// zero for its page, so every downstream check treats that page the way it treats a page no
// strip owns: it must still hold its canary. nullptr means every strip runs.
inline Reference compute_oracle_reference(const Shape& shape, const std::int8_t* key,
                                          const std::int8_t* value,
                                          const std::int32_t* slot_mapping,
                                          const char* strip_runs = nullptr)
{
    require(shape.valid(), "the prefill reference needs a valid cache geometry");
    require(shape.block_size == tq_test::kStripHeight,
            "the prefill reference quantises one strip per page, so block_size must be R");
    Reference ref;
    ref.shape = shape;
    ref.key_cache.assign(shape.cache_bytes(), 0);
    ref.value_cache.assign(shape.cache_bytes(), 0);
    ref.scale_cache.assign(shape.scale_floats(), 0.0f);
    ref.page_written.assign(static_cast<std::size_t>(shape.num_blocks), 0);
    ref.chunks.resize(static_cast<std::size_t>(strip_count(shape)) * kPlanes * shape.num_kv_heads);

    for (int strip = 0; strip < strip_count(shape); ++strip) {
        const int page = page_of_slots(shape, slot_mapping, strip);
        if (page < 0) {
            if (ref.bad_slot_strip < 0) ref.bad_slot_strip = strip;
            continue;
        }
        // The slot mapping contract is checked for every strip, sampled or
        // not: a mapping that is only legal for the blocks that happen to run
        // is not a legal mapping.
        if (strip_runs != nullptr && strip_runs[strip] == 0) continue;
        ref.page_written[static_cast<std::size_t>(page)] = 1;
        for (int plane = 0; plane < kPlanes; ++plane)
            for (int head = 0; head < shape.num_kv_heads; ++head) {
                ChunkRef& out = ref.chunks[((static_cast<std::size_t>(strip) * kPlanes) +
                                            plane) * shape.num_kv_heads + head];
                build_chunk(shape, plane == 0 ? key : value, strip, plane, head, page, out);
                auto& cache = plane == 0 ? ref.key_cache : ref.value_cache;
                std::memcpy(cache.data() + ref.head_offset(page, head), out.packed.data(),
                            out.packed.size());
                std::memcpy(ref.scale_cache.data() + ref.scale_offset(plane, page, head),
                            out.scale.data(), out.scale.size() * sizeof(float));
            }
    }
    return ref;
}

// The six-scalar form the harnesses call, so the shape never has to be
// assembled twice.
inline Reference compute_oracle_reference(int num_tokens, int num_kv_heads, int head_size,
                                          int block_size, int num_blocks,
                                          const std::int8_t* key, const std::int8_t* value,
                                          const std::int32_t* slot_mapping,
                                          const char* strip_runs = nullptr)
{
    Shape shape;
    shape.num_tokens = num_tokens;
    shape.num_kv_heads = num_kv_heads;
    shape.head_size = head_size;
    shape.block_size = block_size;
    shape.num_blocks = num_blocks;
    return compute_oracle_reference(shape, key, value, slot_mapping, strip_runs);
}

// --------------------------------------------------------- oracle self-audit
// Phase one of the two-phase oracle: the properties that make this fixture
// able to detect a wrong read at all. A fixture whose chunks or tokens are
// interchangeable cannot tell a wrong-slot, wrong-core or wrong-token read
// from a correct one, which is the entire point of running several strips over
// rings of depth 4 and 2.
struct Audit {
    int nibbles_covered = 0;        // of 16
    int clamped_chunks = 0;
    int mixed_exponent_chunks = 0;  // chunks carrying two or more exponents
    int distinct_exponents = 0;
    int min_q = 0;
    int max_q = 0;
    int sign_negatives = 0;
    bool pages_distinct = true;          // pairwise, inside every strip
    bool tokens_distinct = true;         // pairwise, inside every chunk
    bool strips_distinct = true;         // same (plane, head) across strips
    bool nibbles_in_range = true;
    // Informational, NOT a requirement. Two chunks of one strip sharing a max |y| is harmless:
    // the quantised nibbles are scale invariant, so two chunks can share a maximum and still
    // pack to completely different pages, and it is `pages_distinct` that makes a wrong-chunk or
    // wrong-core read detectable. Requiring distinct maxima would be a birthday-paradox trap --
    // 8 maxima a strip drawn from a range of a couple of thousand collide about once in 70
    // strips. Counted so that a genuinely degenerate fixture is still visible.
    int chunk_max_collisions = 0;
    int chunk_pairs = 0;

    bool ok() const
    {
        return nibbles_covered == 16 && clamped_chunks > 0 &&
               mixed_exponent_chunks > 0 && distinct_exponents >= 3 &&
               nibbles_in_range && pages_distinct && tokens_distinct &&
               strips_distinct;
    }
};

inline Audit audit_reference(const Reference& ref)
{
    const Shape& shape = ref.shape;
    const int strips = strip_count(shape);
    const int heads = shape.num_kv_heads;
    Audit audit;
    audit.min_q = kNibbleMax;
    audit.max_q = kNibbleMin;
    for (int ch = 0; ch < shape.head_size; ++ch) audit.sign_negatives += (pi_sign(ch) < 0);

    bool seen_nibble[16] = {};
    std::vector<int> seen_exponent;
    for (int strip = 0; strip < strips; ++strip)
        for (int plane = 0; plane < kPlanes; ++plane)
            for (int head = 0; head < heads; ++head) {
                const ChunkRef& c = ref.chunk(strip, plane, head);
                if (c.q.empty()) continue;
                for (int q : c.q) seen_nibble[q & 0xF] = true;
                audit.min_q = std::min(audit.min_q, c.min_q);
                audit.max_q = std::max(audit.max_q, c.max_q);
                audit.clamped_chunks += c.clamped;
                int lo = c.exponent[0], hi = c.exponent[0];
                for (int e : c.exponent) {
                    lo = std::min(lo, e);
                    hi = std::max(hi, e);
                    if (std::find(seen_exponent.begin(), seen_exponent.end(), e) ==
                        seen_exponent.end())
                        seen_exponent.push_back(e);
                }
                audit.mixed_exponent_chunks += (hi > lo);
                // Tokens inside one chunk must quantise differently, or a
                // cross-token read inside a page is invisible.
                for (int a = 0; a < shape.block_size && audit.tokens_distinct; ++a)
                    for (int b = a + 1; b < shape.block_size; ++b)
                        if (std::memcmp(&c.q[static_cast<std::size_t>(a) * shape.head_size],
                                        &c.q[static_cast<std::size_t>(b) * shape.head_size],
                                        static_cast<std::size_t>(shape.head_size) * sizeof(int)) == 0) {
                            audit.tokens_distinct = false;
                            break;
                        }
            }
    for (bool seen : seen_nibble) audit.nibbles_covered += seen;
    audit.distinct_exponents = static_cast<int>(seen_exponent.size());
    audit.nibbles_in_range = audit.min_q >= kNibbleMin && audit.max_q <= kNibbleMax;

    // Pairwise distinctness inside a strip: a wrong-chunk or wrong-core read
    // must not be able to produce the right page.
    const int chunks_per_strip = kPlanes * heads;
    for (int strip = 0; strip < strips; ++strip)
        for (int a = 0; a < chunks_per_strip; ++a)
            for (int b = a + 1; b < chunks_per_strip; ++b) {
                const ChunkRef& ca = ref.chunk(strip, a / heads, a % heads);
                const ChunkRef& cb = ref.chunk(strip, b / heads, b % heads);
                if (ca.packed.empty() || cb.packed.empty()) continue;
                ++audit.chunk_pairs;
                audit.chunk_max_collisions += (ca.chunk_max == cb.chunk_max);
                if (ca.packed.size() == cb.packed.size() &&
                    std::memcmp(ca.packed.data(), cb.packed.data(), ca.packed.size()) == 0)
                    audit.pages_distinct = false;
            }

    // And across strips: a strip that read its neighbour's activations must
    // not be able to produce the right page either.
    for (int plane = 0; plane < kPlanes; ++plane)
        for (int head = 0; head < heads; ++head)
            for (int a = 0; a < strips; ++a)
                for (int b = a + 1; b < strips; ++b) {
                    const ChunkRef& ca = ref.chunk(a, plane, head);
                    const ChunkRef& cb = ref.chunk(b, plane, head);
                    if (ca.packed.empty() || cb.packed.empty()) continue;
                    if (std::memcmp(ca.packed.data(), cb.packed.data(), ca.packed.size()) == 0)
                        audit.strips_distinct = false;
                }
    return audit;
}



// ==================================================== the FP64 attention reference
// softmax(q k^T * score_scale) v over an NZ-tiled packed cache, in double.
//
// THE RECONSTRUCTION RULE, and where it comes from. The decode reads a byte as two SIGNED
// nibbles (turboquant_dequant4_signed), DeInterleaves them into a low and a high plane, and casts
// each straight to the Cube operand (ComputeNibbleOperands in src/common/turboquant_codec_mx.h):
// the writers' convention, which the PyTorch golden shares, so a stored signed nibble n
// reconstructs as
//
//   n * stored_scale
//
// with the stored scale the whole per-row scale -- no centroid offset and no codebook gain. This
// is the one place that rule is restated outside the codec.
//
// The byte and the nibble of a logical (slot, kv head, channel) come from the production address
// functions in turboquant_layout.h, not from a copy: an oracle that restated the addressing
// would be wrong together with a wrong contract.
template <vllm_ascend::turboquant::TurboQuantMode Mode>
constexpr double affine_offset()
{
    using Traits = vllm_ascend::turboquant::TurboQuantModeTraits<Mode>;
    static_assert(Traits::kIsAffine, "the affine reconstruction is only defined for a uniform mode");
    return 0.0;
}

struct DecodeGeometry {
    int requests = 1;
    int kv_heads = 1;
    int group_heads = 16;
    int head_size = 256;
    int page_tokens = 64;   // the cache block size, 64 or 128
    int max_pages = 8;      // columns of the block table
    float score_scale = 0.0625f;
    int query_heads() const { return kv_heads * group_heads; }
    int packed_bytes() const { return head_size / kNibblesPerByte; }
    int scale_slot() const
    {
        return static_cast<int>(vllm_ascend::turboquant::NzScaleSlotFloats(
            static_cast<std::uint32_t>(kv_heads)));
    }
};

// One dequantised cache element. `plane` is 0 for the key tensor and 1 for the value tensor;
// `slot` is the global row, physical * page_tokens + token.
inline double dequantise_nz(const DecodeGeometry& g, const std::uint8_t* packed,
                            const float* scales, int plane, int slot, int kv_head, int channel,
                            double offset)
{
    const auto byte = vllm_ascend::turboquant::NzTiledChannelByte(
        static_cast<std::uint64_t>(slot), static_cast<std::uint64_t>(kv_head),
        static_cast<std::uint64_t>(channel), static_cast<std::uint64_t>(g.kv_heads),
        static_cast<std::uint64_t>(g.head_size));
    const auto which = vllm_ascend::turboquant::NzTiledChannelNibble(
        static_cast<std::uint64_t>(channel), static_cast<std::uint64_t>(g.head_size));
    const unsigned raw = which == 0u ? (packed[byte] & 0x0Fu) : (packed[byte] >> 4);
    const int level = raw >= 8u ? static_cast<int>(raw) - 16 : static_cast<int>(raw);
    const auto slot_index = vllm_ascend::turboquant::NzTiledScaleIndex(
        static_cast<std::uint64_t>(slot), static_cast<std::uint64_t>(kv_head),
        static_cast<std::uint64_t>(plane), static_cast<std::uint64_t>(g.kv_heads));
    return (static_cast<double>(level) + offset) * static_cast<double>(scales[slot_index]);
}

// The reference itself. Row-major [request][query head][channel], in double.
//
// The softmax is computed with the running-maximum subtraction the device uses, so the two agree
// on which exponentials are representable; the accumulation order differs on purpose, because an
// oracle that reproduced the device's order would not detect a wrong order.
template <vllm_ascend::turboquant::TurboQuantMode Mode =
              vllm_ascend::turboquant::TurboQuantMode::KV4_FP8>
inline std::vector<double> attention_reference(const DecodeGeometry& g, const float* query,
                                               const std::uint8_t* key, const std::uint8_t* value,
                                               const float* scales, const std::int32_t* table,
                                               const std::int32_t* lengths)
{
    require(g.requests > 0 && g.kv_heads > 0 && g.group_heads > 0 && g.head_size > 0,
            "the attention reference needs a non-empty geometry");
    require(g.page_tokens > 0 &&
                g.page_tokens % static_cast<int>(vllm_ascend::turboquant::kCubeTileRows) == 0,
            "a cache page must be a whole number of 64-row Cube tiles");
    constexpr double offset = affine_offset<Mode>();
    const int heads = g.query_heads();
    std::vector<double> out(static_cast<std::size_t>(g.requests) * heads * g.head_size, 0.0);

    for (int request = 0; request < g.requests; ++request) {
        const int context = lengths[request] > 0 ? lengths[request] : 0;
        const int pages = (context + g.page_tokens - 1) / g.page_tokens;
        require(pages <= g.max_pages, "a context longer than the block table can address");
        for (int qh = 0; qh < heads; ++qh) {
            const int kv_head = qh / g.group_heads;
            const float* q = query + (static_cast<std::size_t>(request) * heads + qh) * g.head_size;
            const std::size_t base =
                (static_cast<std::size_t>(request) * heads + qh) * g.head_size;

            // Pass one: the scores and their maximum.
            std::vector<double> scores(static_cast<std::size_t>(context), 0.0);
            double max_score = -std::numeric_limits<double>::infinity();
            for (int row = 0; row < context; ++row) {
                const int page = row / g.page_tokens;
                const std::int32_t physical = table[static_cast<std::size_t>(request) * g.max_pages + page];
                require(physical >= 0, "a live logical page is unmapped in the block table");
                const int slot = physical * g.page_tokens + row % g.page_tokens;
                double dot = 0.0;
                for (int c = 0; c < g.head_size; ++c)
                    dot += static_cast<double>(q[c]) *
                           dequantise_nz(g, key, scales, 0, slot, kv_head, c, offset);
                const double s = dot * static_cast<double>(g.score_scale);
                scores[static_cast<std::size_t>(row)] = s;
                if (s > max_score) max_score = s;
            }
            if (context == 0) continue;

            // Pass two: the normalised weights and the context vector.
            double denominator = 0.0;
            for (int row = 0; row < context; ++row) {
                const double w = std::exp(scores[static_cast<std::size_t>(row)] - max_score);
                scores[static_cast<std::size_t>(row)] = w;
                denominator += w;
            }
            require(denominator > 0.0, "a degenerate softmax denominator");
            for (int row = 0; row < context; ++row) {
                const double w = scores[static_cast<std::size_t>(row)];
                if (w == 0.0) continue;
                const int page = row / g.page_tokens;
                const std::int32_t physical = table[static_cast<std::size_t>(request) * g.max_pages + page];
                const int slot = physical * g.page_tokens + row % g.page_tokens;
                for (int c = 0; c < g.head_size; ++c)
                    out[base + static_cast<std::size_t>(c)] +=
                        w * dequantise_nz(g, value, scales, 1, slot, kv_head, c, offset);
            }
            for (int c = 0; c < g.head_size; ++c) out[base + static_cast<std::size_t>(c)] /= denominator;
        }
    }
    return out;
}

}  // namespace tq_golden

#endif  // TURBOQUANT_GOLDEN_ORACLE_HPP
