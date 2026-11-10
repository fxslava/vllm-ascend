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

// Can an ingress store be read back by the kv4fp8 fused decode?
//
// There are two 4-bit KV cache layouts in this repository, and this is the gate that says which
// one a writer is speaking.
//
//   (A) STRIP -- RFC 3.1, what turboquant_reshape_batch.cpp and turboquant_reshape_scatter.cpp
//       write. Normative in turboquant_layout.h, mirrored as index maps in
//       tests/common/turboquant_test_common.hpp, R = kStripHeight:
//         key   nibble = ((block*H + head)*S + token)*d + channel
//         value nibble = ((block*H + head)*S)*d + (token/R)*(R*d) + channel*R + (token % R)
//         scale index  = ((kv*P + block)*H + head)*S + token
//
//   (B) NZ-TILED -- what the kv4fp8 fused decode reads: kCubeTileRows-row tiles,
//       kOperandC0-byte column groups, and a nibble pairing that puts channels (c, c + d/2) of
//       ONE token in a byte. Its ingress side is NzTiledChannelByte / NzTiledChannelNibble /
//       NzTiledScaleIndex.
//
// CHECK 1 IS THE HEADLINE: the layout a writer targets under (B) names THE SAME BYTE AND THE
// SAME NIBBLE as the fused decode reads, for every logical element. That is the condition under
// which a real write -> read round trip exists.
//
// The comparison is not vacuous. The writer's address function is the production
// NzTiledChannelByte; the reader's is re-derived here, independently, by walking what the decode
// actually does to get from a GM byte to a logical (token, channel):
//
//   1. ReadPlane issues ONE tilePlaneBytes burst from
//      ((slot - tileRow) * numKvHeads + kvHead * kCubeTileRows) * packedBytes, so
//      packed[i] = GM[tileBase + i].
//   2. UnpackAffine DeInterleaves it: nzLow[i] = low nibble of packed[i], nzHigh[i] = high.
//   3. StageNzTileToL1 stages nzLow at L1 offset 0 and nzHigh at + tilePlaneBytes, in ONE
//      contiguous burst, so L1[i] = nzLow[i] below tilePlaneBytes and nzHigh[i - tilePlaneBytes]
//      above it.
//   4. L1 is the kCubeTileRows x d NZ image, where element (row n, col e) sits at
//      (e / kOperandC0) * kCubeTileRows * kOperandC0 + n * kOperandC0 + e % kOperandC0.
//
// Inverting 1..4 gives DecodeReadByte below. If the writer's arithmetic and that walk diverge,
// this test fails and the round trip is off.

#include "common/turboquant_test_common.hpp"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

namespace tq = vllm_ascend::turboquant;
using tq_test::Shape;

tq_test::Failures g_fail;

struct Addr {
    std::size_t byte;
    int nibble;  // 0 = low, 1 = high
};

bool same(const Addr& a, const Addr& b) { return a.byte == b.byte && a.nibble == b.nibble; }

// --------------------------------------------- (B) the ingress writer's address function
// The production contract, as the kernel issues it. K and V are symmetric, so one function
// serves both planes -- which is the whole reason the NZ ingress is simpler than the strip one.
Addr IngressNzByte(const Shape& s, int block, int head, int token, int channel)
{
    const std::uint64_t slot = static_cast<std::uint64_t>(block) * s.block_size + token;
    return Addr{tq::NzTiledChannelByte(slot, static_cast<std::uint64_t>(head),
                                       static_cast<std::uint64_t>(channel),
                                       static_cast<std::uint64_t>(s.num_kv_heads),
                                       static_cast<std::uint64_t>(s.head_size)),
                static_cast<int>(tq::NzTiledChannelNibble(
                    static_cast<std::uint64_t>(channel),
                    static_cast<std::uint64_t>(s.head_size)))};
}

// ----------------------------------------------- (B) the fused decode's reader, walked
// Re-derived from ReadPlane -> UnpackAffine -> StageNzTileToL1 -> the L1 NZ image, per the file
// header. Deliberately written in the decode's own terms and NOT by calling the writer's helper,
// so that check 1 compares two independent derivations.
Addr DecodeReadByte(const Shape& s, int block, int head, int token, int channel)
{
    const std::uint64_t packed_bytes = static_cast<std::uint64_t>(s.head_size) / tq::kPackFactor;
    const std::uint64_t slot = static_cast<std::uint64_t>(block) * s.block_size + token;
    const std::uint64_t tile_row = slot % tq::kCubeTileRows;

    // Step 1: the tile's GM base, exactly as ReadPlane computes it.
    const std::uint64_t tile_base =
        ((slot - tile_row) * s.num_kv_heads +
         static_cast<std::uint64_t>(head) * tq::kCubeTileRows) * packed_bytes;

    // Step 4: where (tile_row, channel) sits in the L1 NZ image of the staged tile.
    const std::uint64_t group = static_cast<std::uint64_t>(channel) / tq::kOperandC0;
    const std::uint64_t within = static_cast<std::uint64_t>(channel) % tq::kOperandC0;
    const std::uint64_t l1 =
        group * tq::kCubeTileRows * tq::kOperandC0 + tile_row * tq::kOperandC0 + within;

    // Steps 2 and 3, inverted: below the plane boundary the element came from a low nibble of
    // the burst, above it from a high nibble of the byte that many positions into the burst.
    const std::uint64_t plane_bytes = tq::kCubeTileRows * packed_bytes;
    if (l1 < plane_bytes) return Addr{tile_base + l1, 0};
    return Addr{tile_base + l1 - plane_bytes, 1};
}

std::size_t DecodeReadScale(const Shape& s, int plane, int block, int head, int token)
{
    const std::uint64_t slot = static_cast<std::uint64_t>(block) * s.block_size + token;
    return tq::NzTiledScaleIndex(slot, static_cast<std::uint64_t>(head),
                                 static_cast<std::uint64_t>(plane),
                                 static_cast<std::uint64_t>(s.num_kv_heads));
}

// ---------------------------------------------------- (A) the legacy strip contract
Addr StripByte(const Shape& s, int plane, int block, int head, int token, int channel)
{
    const std::size_t nibble = tq_test::strip_nibble(s, plane, block, head, token, channel);
    return Addr{nibble / 2, static_cast<int>(nibble % 2)};
}

// ------------------------------------------------------------------------------- the checks

// CHECK 1. The writer and the reader name the same byte and the same nibble, everywhere.
void check_agreement(const Shape& s)
{
    std::size_t compared = 0;
    bool agree = true;
    for (int block = 0; block < s.num_blocks && agree; ++block)
        for (int head = 0; head < s.num_kv_heads && agree; ++head)
            for (int token = 0; token < s.block_size && agree; ++token)
                for (int channel = 0; channel < s.head_size; ++channel) {
                    ++compared;
                    if (!same(IngressNzByte(s, block, head, token, channel),
                              DecodeReadByte(s, block, head, token, channel))) {
                        std::printf("  first disagreement at block %d head %d token %d channel %d\n",
                                    block, head, token, channel);
                        agree = false;
                        break;
                    }
                }
    g_fail.check(agree, "the NZ ingress writer and the fused decode name the same nibble");
    std::printf("  check 1: %zu logical elements, writer == reader\n", compared);
}

// CHECK 2. The NZ map is a bijection onto the (byte, nibble) pairs of the allocation. A map that
// agreed with the reader but collided would lose data without ever disagreeing.
void check_bijective(const Shape& s)
{
    const std::size_t nibbles = s.cache_bytes() * 2;
    std::vector<char> seen(nibbles, 0);
    bool in_range = true, distinct = true;
    for (int block = 0; block < s.num_blocks; ++block)
        for (int head = 0; head < s.num_kv_heads; ++head)
            for (int token = 0; token < s.block_size; ++token)
                for (int channel = 0; channel < s.head_size; ++channel) {
                    const Addr a = IngressNzByte(s, block, head, token, channel);
                    const std::size_t index = a.byte * 2 + static_cast<std::size_t>(a.nibble);
                    if (index >= nibbles) {
                        in_range = false;
                        continue;
                    }
                    if (seen[index]) distinct = false;
                    seen[index] = 1;
                }
    bool onto = true;
    for (char bit : seen)
        if (!bit) onto = false;
    g_fail.check(in_range, "every NZ address lands inside the cache allocation");
    g_fail.check(distinct, "no two logical elements share an NZ nibble");
    g_fail.check(onto, "the NZ map covers every nibble of the allocation");
    std::printf("  check 2: NZ map is a bijection over %zu nibbles\n", nibbles);
}

// CHECK 3. The same for the strip contract, and the CO-BYTE structure that distinguishes the
// two layouts. In the strip layout a byte holds two ADJACENT channels of one token on the key
// plane and two adjacent TOKENS of one channel on the value plane; in the NZ layout a byte holds
// channels (c, c + d/2) of one token on BOTH planes. Those are the pairings a packer has to get
// right, and they are what no address permutation can reconcile.
void check_strip_bijective_and_cobytes(const Shape& s)
{
    for (int plane = 0; plane < tq_test::kPlanes; ++plane) {
        const std::size_t nibbles = s.cache_bytes() * 2;
        std::vector<char> seen(nibbles, 0);
        bool distinct = true, in_range = true;
        for (int block = 0; block < s.num_blocks; ++block)
            for (int head = 0; head < s.num_kv_heads; ++head)
                for (int token = 0; token < s.block_size; ++token)
                    for (int channel = 0; channel < s.head_size; ++channel) {
                        const Addr a = StripByte(s, plane, block, head, token, channel);
                        const std::size_t index = a.byte * 2 + static_cast<std::size_t>(a.nibble);
                        if (index >= nibbles) {
                            in_range = false;
                            continue;
                        }
                        if (seen[index]) distinct = false;
                        seen[index] = 1;
                    }
        bool onto = true;
        for (char bit : seen)
            if (!bit) onto = false;
        g_fail.check(in_range && distinct && onto,
                     plane == 0 ? "the strip key map is a bijection"
                                : "the strip value map is a bijection");
    }

    // The co-byte partner of (token, channel): the logical element sharing its byte.
    bool key_cobyte = true, value_cobyte = true, nz_cobyte = true;
    const int half = s.head_size / 2;
    for (int token = 0; token < s.block_size; ++token)
        for (int channel = 0; channel < s.head_size; ++channel) {
            // Key plane: channel ^ 1 of the same token.
            const Addr k0 = StripByte(s, 0, 1, 0, token, channel);
            const Addr k1 = StripByte(s, 0, 1, 0, token, channel ^ 1);
            if (k0.byte != k1.byte || k0.nibble == k1.nibble) key_cobyte = false;
            // Value plane: token ^ 1 inside the strip, same channel.
            const int partner = (token / tq_test::kStripHeight) * tq_test::kStripHeight +
                                ((token % tq_test::kStripHeight) ^ 1);
            const Addr v0 = StripByte(s, 1, 1, 0, token, channel);
            const Addr v1 = StripByte(s, 1, 1, 0, partner, channel);
            if (v0.byte != v1.byte || v0.nibble == v1.nibble) value_cobyte = false;
            // NZ, both planes: channel +- d/2 of the same token.
            const int nz_partner = channel < half ? channel + half : channel - half;
            const Addr n0 = IngressNzByte(s, 1, 0, token, channel);
            const Addr n1 = IngressNzByte(s, 1, 0, token, nz_partner);
            if (n0.byte != n1.byte || n0.nibble == n1.nibble) nz_cobyte = false;
        }
    g_fail.check(key_cobyte, "a strip key byte holds channels (c, c^1) of one token");
    g_fail.check(value_cobyte, "a strip value byte holds tokens (t, t^1) of one channel");
    g_fail.check(nz_cobyte, "an NZ byte holds channels (c, c + d/2) of one token");
    std::printf("  check 3: strip maps bijective; co-byte pairings as contracted\n");
}

// CHECK 4. The two contracts DISAGREE, and not by a permutation of addresses. The co-byte
// partner is an invariant of the byte grouping, not of the numbering, so two layouts whose
// partners differ cannot be related by any relabelling of bytes -- which is why an ingress
// writer has to choose, and why check 1 exists.
void check_layouts_are_distinct(const Shape& s)
{
    std::size_t differing = 0, total = 0;
    for (int plane = 0; plane < tq_test::kPlanes; ++plane)
        for (int token = 0; token < s.block_size; ++token)
            for (int channel = 0; channel < s.head_size; ++channel) {
                ++total;
                if (!same(StripByte(s, plane, 1, 0, token, channel),
                          IngressNzByte(s, 1, 0, token, channel)))
                    ++differing;
            }
    g_fail.check(differing > 0, "the strip and NZ contracts are not the same map");

    // The structural statement: no relabelling bridges them, because the partner relation does
    // not match. d/2 != 1 and d/2 != 0, so an NZ byte never pairs (c, c^1).
    const int half = s.head_size / 2;
    g_fail.check(half != 1, "a byte's channel pairing distinguishes the two contracts");
    std::printf("  check 4: %zu of %zu addresses differ; byte groupings are incompatible\n",
                differing, total);
}

// CHECK 5. The scale plane. The NZ cache keeps 2*H lanes per cache ROW, rounded up to a whole
// fp32 block, so a tile's scales are one contiguous burst beside the tile; the strip contract
// keeps a contiguous run of S floats per (kv, block, head). Both formulas are asserted, and the
// NZ one is checked for bijectivity over the rows of the allocation.
void check_scale_planes(const Shape& s)
{
    for (int h = 1; h <= 4; ++h) {
        const std::uint32_t lanes = tq::NzScaleSlotFloats(static_cast<std::uint32_t>(h));
        g_fail.check(lanes % tq::kFp32PerBlock == 0, "an NZ scale slot is a whole fp32 block");
        g_fail.check(lanes >= 2u * static_cast<std::uint32_t>(h),
                     "an NZ scale slot holds a key and a value lane per head");
    }

    const std::size_t rows = static_cast<std::size_t>(s.num_blocks) * s.block_size;
    const std::size_t floats = rows * tq::NzScaleSlotFloats(static_cast<std::uint32_t>(s.num_kv_heads));
    std::vector<char> seen(floats, 0);
    bool distinct = true, in_range = true;
    for (int block = 0; block < s.num_blocks; ++block)
        for (int head = 0; head < s.num_kv_heads; ++head)
            for (int token = 0; token < s.block_size; ++token)
                for (int plane = 0; plane < tq_test::kPlanes; ++plane) {
                    const std::size_t index = DecodeReadScale(s, plane, block, head, token);
                    if (index >= floats) {
                        in_range = false;
                        continue;
                    }
                    if (seen[index]) distinct = false;
                    seen[index] = 1;
                }
    g_fail.check(in_range, "every NZ scale index lands inside the scale plane");
    g_fail.check(distinct, "no two (plane, head, row) share an NZ scale slot");

    // The strip contract's scale plane, for the same reason.
    const std::size_t strip_floats = s.scale_floats();
    std::vector<char> strip_seen(strip_floats, 0);
    bool strip_ok = true;
    for (int plane = 0; plane < tq_test::kPlanes; ++plane)
        for (int block = 0; block < s.num_blocks; ++block)
            for (int head = 0; head < s.num_kv_heads; ++head)
                for (int token = 0; token < s.block_size; ++token) {
                    const std::size_t index = tq_test::scale_index(s, plane, block, head, token);
                    if (index >= strip_floats || strip_seen[index]) strip_ok = false;
                    else strip_seen[index] = 1;
                }
    g_fail.check(strip_ok, "the strip scale index is a bijection over its plane");
    std::printf("  check 5: scale planes bijective (NZ %zu floats, strip %zu floats)\n", floats,
                strip_floats);
}

void run(int num_blocks, int num_kv_heads, int block_size)
{
    Shape s;
    s.num_kv_heads = num_kv_heads;
    s.head_size = 256;
    s.block_size = block_size;
    s.num_blocks = num_blocks;
    s.num_tokens = block_size;
    tq_test::require(s.valid(), "the bridge geometry must be a legal cache shape");
    tq_test::require(block_size % static_cast<int>(tq::kCubeTileRows) == 0,
                     "an NZ page must be a whole number of Cube tiles");
    std::printf("geometry: %d pages x %d kv heads x %d tokens x %d channels\n", num_blocks,
                num_kv_heads, block_size, s.head_size);
    check_agreement(s);
    check_bijective(s);
    check_strip_bijective_and_cobytes(s);
    check_layouts_are_distinct(s);
    check_scale_planes(s);
}

}  // namespace

int main()
{
    try {
        // One page and several, one kv head and four, and both page heights valid_launch admits.
        run(1, 1, 64);
        run(2, 4, 64);
        run(3, 2, 128);
        run(1, 4, 128);
    } catch (const std::exception& error) {
        std::printf("FAIL test_layout_bridge: %s\n", error.what());
        return 1;
    }
    if (!g_fail.clean()) {
        std::printf("FAIL test_layout_bridge: %d check(s) failed\n", g_fail.count);
        return 1;
    }
    std::printf("PASS test_layout_bridge: NZ writer == decode reader, both contracts bijective, "
                "co-byte pairings incompatible, scale planes bijective\n");
    return 0;
}
