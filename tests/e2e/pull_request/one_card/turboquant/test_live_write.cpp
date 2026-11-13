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

// LIVE WRITE: both ingress operators against the golden oracle, on CAModel or on silicon.
//
//   --op prefill   turboquant_reshape_and_cache_bf16, the MIX batch writer. N aligned 16-token
//                  strips in ONE monolithic grid, S strips a block as one chunk stream.
//   --op scatter   turboquant_reshape_scatter_aiv, the AIV single-token / sub-strip writer, whose
//                  value plane is a read-modify-write of one 2048-byte strip.
//   --op both      (default) each in turn.
//
// STRIP-ERA GATE. This runner pins kBlockSize = 16 and the RFC 3.1 strip cache,
// which the kernels refuse under the TURBOQUANT_INGRESS_NZ=1 build this repo
// now produces: both launchers throw on a block size that is not a multiple of
// 64. The NZ counterparts live in turboquant-core's own runner
// (tests/npu/test_live_write.cpp, built twice -- strip and NZ, see its
// scatter_nz_rig) and the Python-side device gate is
// tests/ut/attention/test_turboquant_live_write_npu.py. This file stays as the
// reference for the strip build until the NZ rig is ported over; do not read a
// PASS from it as evidence about the NZ ingress, or a failure as a regression
// in it.
//
// WHAT MAKES THIS A GATE AND NOT A SMOKE TEST
//
// 1. BIT EQUALITY, NOT A TOLERANCE. The fixtures are integer-valued in BF16, so every
//    intermediate of the rotation is an exact integer in FP32, the scale is a power of two
//    extracted by integer ops on the FP32 bit pattern, and the single round-to-nearest-even is
//    the only rounding in the path. Nibbles and scale floats are compared bit for bit, which is
//    what detects a wrong-strip or wrong-core read; a cosine would not.
//
// 2. CANARIES ON EVERY BUFFER, AND ON EVERY PAGE NOBODY OWNS. Each Global Memory allocation
//    carries a guard band on both sides, and the fixture maps strip s to physical page 2s + 1,
//    so every even page must still read back as 0xDEADBEEF afterwards. A burst that ran off the
//    end of a page, or a kernel that derived its page from the block index rather than from its
//    slotMapping entry, fails here by name.
//
// 3. THE SCATTER WRITE PRESERVES. Lanes a work item does not own go back unchanged, because in
//    decode a strip accumulates over sixteen successive steps.
//
// WHAT THIS RUNNER CANNOT SEE: whether a cross-core announcement overtook the pipeline work it
// reports. Both earlier revisions of the batch writer produced bit-exact output while their
// traces showed a wait retiring before its set. Audit the trace; do not read the PASS line
// alone.

#include "tests/ut/ops/turboquant/common/camodel_session.hpp"
#include "tests/ut/ops/turboquant/common/turboquant_golden_oracle.hpp"
#include "tests/ut/ops/turboquant/common/turboquant_scatter_oracle.hpp"
#include "csrc/attention/turboquant/op_kernel/common/turboquant_wave_protocol.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

extern "C" void launch_turboquant_reshape_and_cache_bf16(
    void* stream, const std::uint16_t* key, const std::uint16_t* value,
    std::int8_t* key_cache, std::int8_t* value_cache, float* scale_cache,
    const std::int32_t* slot_mapping, const float* pi_signs,
    const std::int32_t* tables, std::uint32_t num_tokens,
    std::uint32_t num_kv_heads, std::uint32_t head_size, std::uint32_t block_size,
    std::uint32_t num_blocks, std::uint32_t tokens_per_core, float inv_sqrt_len);

namespace prefill_rig {

constexpr int kHeads = 4;
constexpr int kHeadSize = 256;
constexpr int kBlockSize = 16;
constexpr int kMaxSlices = 64;

using tq_golden::ChunkRef;
using tq_golden::Shape;

void require(bool ok, const char* what)
{
    if (!ok) {
        std::fprintf(stderr, "FIXTURE/CHECK FAILED: %s\n", what);
        throw std::runtime_error(what);
    }
}

// An integer in [-256, 256] as a BF16 bit pattern. BF16 is the top 16 bits of
// the FP32 pattern, and for these values the discarded low 16 bits are all
// zero, so the narrowing is exact rather than rounded.
std::uint16_t to_bf16(int value)
{
    const float f = static_cast<float>(value);
    const std::uint32_t bits = tq_golden::float_bits(f);
    if ((bits & 0xFFFFu) != 0u) throw std::runtime_error("activation is not exact in BF16");
    return static_cast<std::uint16_t>(bits >> 16);
}

int chunk_plane(int chunk) { return chunk >= kHeads ? 0 : 1; }
int chunk_head(int chunk) { return chunk % kHeads; }

void usage(std::FILE* output, const char* argv0)
{
    std::fprintf(output,
                 "Usage: %s [--slices N] [--strips-per-block S]\n"
                 "  N aligned 16-token strips in ONE monolithic grid. There is no launch\n"
                 "  window: a grid wider than the AI Core count spans several physical waves\n"
                 "  inside the single dispatch, which is the case the inter-wave protocol\n"
                 "  exists for. Default 1, 1..%d.\n"
                 "  S strips per grid block (default 1). S > 1 exercises the seamless chunk\n"
                 "  pipeline: one block runs S * 8 chunks as one stream, so MTE2 never drains\n"
                 "  at a strip boundary. N must be a multiple of S.\n"
                 "  Do not read the PASS line alone -- audit the trace. A bit-exact result says\n"
                 "  nothing about whether an announcement overtook the pipeline work it\n"
                 "  reports; only the fence and census audit does.\n",
                 argv0, kMaxSlices);
}

int run(int slices, int strips_per_block)
{
    Shape shape;
    shape.num_tokens = slices * kBlockSize;
    shape.num_kv_heads = kHeads;
    shape.head_size = kHeadSize;
    shape.block_size = kBlockSize;
    shape.num_blocks = tq_golden::fragmented_num_blocks(slices);
    require(shape.valid(), "unsupported shape");
    const int head_bytes = shape.head_bytes();
    const int channels = kHeads * kHeadSize;

    // ---- fixture and oracle, shared with the INT8 instance ----------------
    std::vector<std::int8_t> key_in, value_in;
    tq_golden::build_activations(shape, key_in, value_in);
    const std::vector<std::int32_t> slot_map = tq_golden::build_slot_mapping(shape);
    const std::vector<std::int8_t> table_blob = tq_golden::build_tables(shape);
    const tq_golden::Reference ref = tq_golden::compute_oracle_reference(
        shape, key_in.data(), value_in.data(), slot_map.data());
    require(ref.bad_slot_strip < 0,
            "the fixture's slot mapping breaks the one-strip-one-page contract");

    const tq_golden::Audit audit = tq_golden::audit_reference(ref);
    require(audit.nibbles_in_range, "the oracle produced a value outside INT4");
    require(audit.nibbles_covered == 16,
            "the fixture does not exercise all sixteen nibbles");
    require(audit.clamped_chunks > 0, "the INT4 upper clamp is never taken");
    require(audit.mixed_exponent_chunks > 0,
            "no chunk carries two different scale exponents");
    require(audit.distinct_exponents >= 3, "the scale exponent barely varies");
    require(audit.pages_distinct, "two chunks of one strip pack to identical pages");
    require(audit.tokens_distinct, "two tokens of one chunk quantise identically");
    require(audit.strips_distinct, "two strips pack to identical pages");

    const long long composed = tq_golden::audit_tables(shape, table_blob.data());
    require(composed == static_cast<long long>(kHeadSize) * kHeadSize,
            "the resident rotation blocks do not compose to the oracle's matrix");

    // ---- device buffers ---------------------------------------------------
    camodel::Session session;
    camodel::GuardedDeviceBuffer key(key_in.size() * sizeof(std::uint16_t));
    camodel::GuardedDeviceBuffer value(value_in.size() * sizeof(std::uint16_t));
    camodel::GuardedDeviceBuffer key_cache(shape.cache_bytes());
    camodel::GuardedDeviceBuffer value_cache(shape.cache_bytes());
    camodel::GuardedDeviceBuffer scale_cache(shape.scale_floats() * sizeof(float));
    camodel::GuardedDeviceBuffer slot_mapping(slot_map.size() * sizeof(std::int32_t));
    camodel::GuardedDeviceBuffer pi_signs(kHeadSize * sizeof(float));
    camodel::GuardedDeviceBuffer tables(table_blob.size() * sizeof(std::uint16_t));

    // The activations and the resident blocks go up as BF16; everything else is
    // byte for byte what the INT8 instance uploads.
    auto* kb = static_cast<std::uint16_t*>(key.host_payload());
    auto* vb = static_cast<std::uint16_t*>(value.host_payload());
    for (std::size_t i = 0; i < key_in.size(); ++i) {
        kb[i] = to_bf16(key_in[i]);
        vb[i] = to_bf16(value_in[i]);
    }
    auto* tb = static_cast<std::uint16_t*>(tables.host_payload());
    for (std::size_t i = 0; i < table_blob.size(); ++i) tb[i] = to_bf16(table_blob[i]);
    std::memcpy(slot_mapping.host_payload(), slot_map.data(),
                slot_map.size() * sizeof(std::int32_t));
    auto* ps = static_cast<float*>(pi_signs.host_payload());
    for (int ch = 0; ch < kHeadSize; ++ch) ps[ch] = static_cast<float>(tq_golden::pi_sign(ch));

    key.upload();
    value.upload();
    slot_mapping.upload();
    pi_signs.upload();
    tables.upload();

    // ONE launch of however many grid blocks the share count gives, spanning however many
    // physical waves. Not a loop over capped windows: the kernel carries the inter-wave protocol
    // itself -- the boundary guard before InitSocState, the bounded mode-2 channels, and the
    // epilogue drain of every queue of every role -- so a multi-wave grid is a single dispatch
    // and that is what has to be tested. A capped window would re-arm everything at each
    // boundary and hide exactly the thing under test.
    launch_turboquant_reshape_and_cache_bf16(
        session.stream,
        static_cast<const std::uint16_t*>(key.payload()),
        static_cast<const std::uint16_t*>(value.payload()),
        static_cast<std::int8_t*>(key_cache.payload()),
        static_cast<std::int8_t*>(value_cache.payload()),
        static_cast<float*>(scale_cache.payload()),
        static_cast<const std::int32_t*>(slot_mapping.payload()),
        static_cast<const float*>(pi_signs.payload()),
        static_cast<const std::int32_t*>(tables.payload()),
        static_cast<std::uint32_t>(shape.num_tokens),
        static_cast<std::uint32_t>(shape.num_kv_heads),
        static_cast<std::uint32_t>(shape.head_size),
        static_cast<std::uint32_t>(shape.block_size),
        static_cast<std::uint32_t>(shape.num_blocks),
        static_cast<std::uint32_t>(shape.block_size * strips_per_block), shape.inv_sqrt_len());
    camodel::check(aclrtSynchronizeStream(session.stream), "aclrtSynchronizeStream");

    key_cache.download();
    value_cache.download();
    scale_cache.download();
    key.download();
    value.download();
    slot_mapping.download();
    pi_signs.download();
    tables.download();

    key.assert_guards();
    value.assert_guards();
    key_cache.assert_guards();
    value_cache.assert_guards();
    scale_cache.assert_guards();
    slot_mapping.assert_guards();
    pi_signs.assert_guards();
    tables.assert_guards();

    const auto* kc = static_cast<const std::uint8_t*>(key_cache.host_payload());
    const auto* vc = static_cast<const std::uint8_t*>(value_cache.host_payload());
    const auto* sc = static_cast<const float*>(scale_cache.host_payload());

    // ---- check 2: nothing outside a written page moved --------------------
    std::size_t untouched = 0;
    for (int page = 0; page < shape.num_blocks; ++page) {
        if (ref.page_written[static_cast<std::size_t>(page)] != 0) continue;
        const std::size_t base = ref.page_offset(page);
        for (int i = 0; i < shape.page_bytes(); ++i) {
            const std::size_t at = base + static_cast<std::size_t>(i);
            const auto lane =
                static_cast<std::uint8_t>((camodel::kCanary >> (8 * (at % 4))) & 0xFFu);
            require(kc[at] == lane, "keyCache byte on an unwritten page was overwritten");
            require(vc[at] == lane, "valueCache byte on an unwritten page was overwritten");
            ++untouched;
        }
        for (int plane = 0; plane < tq_golden::kPlanes; ++plane)
            for (int head = 0; head < kHeads; ++head) {
                const std::size_t at = ref.scale_offset(plane, page, head);
                for (int t = 0; t < shape.block_size; ++t)
                    require(tq_golden::float_bits(sc[at + static_cast<std::size_t>(t)]) ==
                                camodel::kCanary,
                            "scaleCache float on an unwritten page moved");
            }
    }

    // ---- checks 3-5: bit-exact payloads ----------------------------------
    std::size_t key_bytes = 0, value_bytes = 0, scales = 0;
    int failing = 0;
    for (int strip = 0; strip < tq_golden::strip_count(shape) && failing < 3; ++strip)
        for (int chunk = 0; chunk < 2 * kHeads && failing < 3; ++chunk) {
            const int plane = chunk_plane(chunk);
            const int head = chunk_head(chunk);
            const ChunkRef& want = ref.chunk(strip, plane, head);
            const std::uint8_t* got =
                (plane == 0 ? kc : vc) + ref.head_offset(want.page, head);
            const float* got_scales = sc + ref.scale_offset(plane, want.page, head);

            int bad_bytes = 0, first_bad = -1;
            for (int i = 0; i < head_bytes; ++i)
                if (got[i] != want.packed[static_cast<std::size_t>(i)]) {
                    if (first_bad < 0) first_bad = i;
                    ++bad_bytes;
                }
            int bad_scales = 0;
            for (int t = 0; t < shape.block_size; ++t)
                if (tq_golden::float_bits(got_scales[t]) !=
                    tq_golden::float_bits(want.scale[static_cast<std::size_t>(t)]))
                    ++bad_scales;

            if (bad_bytes == 0 && bad_scales == 0) {
                if (plane == 0)
                    key_bytes += static_cast<std::size_t>(head_bytes);
                else
                    value_bytes += static_cast<std::size_t>(head_bytes);
                scales += static_cast<std::size_t>(shape.block_size);
                continue;
            }
            ++failing;
            std::fprintf(stderr,
                         "\nstrip %d page %d chunk %d (%s head %d): %d/%d bytes differ "
                         "(first byte %d: device 0x%02x oracle 0x%02x), %d/%d scales "
                         "differ\n",
                         strip, want.page, chunk, plane == 0 ? "key" : "value", head,
                         bad_bytes, head_bytes, first_bad,
                         first_bad < 0 ? 0u : static_cast<unsigned>(got[first_bad]),
                         first_bad < 0
                             ? 0u
                             : static_cast<unsigned>(
                                   want.packed[static_cast<std::size_t>(first_bad)]),
                         bad_scales, shape.block_size);
        }
    if (failing != 0) throw std::runtime_error("packed cache page mismatch");

    const int chunks = tq_golden::strip_count(shape) * 2 * kHeads;
    std::printf("PASS turboquant_reshape_and_cache_bf16: %d strips x 16 tokens x %d "
                "channels, %d chunks, BF16 activations\n",
                tq_golden::strip_count(shape), channels, chunks);
    std::printf("  1 launch of %d grid blocks spanning %d physical wave(s) of 32, %d strip(s) "
                "a block as one chunk stream, strip s -> physical page 2s+1 of %d\n",
                tq_golden::strip_count(shape) / strips_per_block,
                (tq_golden::strip_count(shape) / strips_per_block + 31) / 32, strips_per_block,
                shape.num_blocks);
    std::printf("  keyCache   %zu/%d bytes bit-exact, key-major\n", key_bytes,
                tq_golden::strip_count(shape) * kHeads * head_bytes);
    std::printf("  valueCache %zu/%d bytes bit-exact, channel-major\n", value_bytes,
                tq_golden::strip_count(shape) * kHeads * head_bytes);
    std::printf("  scaleCache %zu/%d FP32 bit-exact\n", scales, chunks * shape.block_size);
    std::printf("  %zu cache bytes on the %d unwritten pages untouched, %d/%d guard "
                "bytes intact\n",
                untouched, shape.num_blocks - tq_golden::strip_count(shape),
                8 * 2 * static_cast<int>(camodel::kGuardBytes),
                8 * 2 * static_cast<int>(camodel::kGuardBytes));
    std::printf("  every nibble reproduced by the integer oracle: no BF16-specific reference "
                "and no tolerance\n");
    std::printf("  K0 = %d (BF16 fractal 16x16), Hadamard group 32 = 2 fractals, "
                "%d resident blocks of 32x32 BF16 (%zu B), %lld composed entries\n",
                32 / 2, 2 * tq_golden::group_count(shape), table_blob.size() * sizeof(std::uint16_t),
                composed);
    return 0;
}

int run_main(int argc, char** argv)
{
    int slices = 1;
    int strips_per_block = 1;
    bool saw_slices = false, saw_share = false;
    const auto parse = [](std::string_view text, int& out, int lo, int hi) {
        int value = 0;
        const auto got = std::from_chars(text.data(), text.data() + text.size(), value);
        if (got.ec != std::errc() || got.ptr != text.data() + text.size() || value < lo ||
            value > hi)
            return false;
        out = value;
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argc == 2 ? stdout : stderr, argv[0]);
            return argc == 2 ? 0 : 2;
        }
        if (arg == "--slices" && !saw_slices && i + 1 < argc &&
            parse(argv[++i], slices, 1, kMaxSlices)) {
            saw_slices = true;
            continue;
        }
        if (arg == "--strips-per-block" && !saw_share && i + 1 < argc &&
            parse(argv[++i], strips_per_block, 1, kMaxSlices)) {
            saw_share = true;
            continue;
        }
        usage(stderr, argv[0]);
        return 2;
    }
    if (slices % strips_per_block != 0) {
        std::fprintf(stderr,
                     "--slices must be a multiple of --strips-per-block: the chunk pipeline has "
                     "no ragged tail\n");
        return 2;
    }
    try {
        return run(slices, strips_per_block);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL turboquant_reshape_and_cache_bf16 (%d strips, %d a block): "
                     "%s\n", slices, strips_per_block, error.what());
        return 1;
    }
}

}  // namespace prefill_rig

extern "C" void launch_turboquant_reshape_scatter_aiv(
    void*, std::uint32_t, std::uint16_t*, std::uint16_t*, std::int8_t*, std::int8_t*, float*,
    std::int32_t*, float*, std::int32_t*, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, float);

namespace scatter_rig {

using namespace tq_scatter;

constexpr std::uint32_t kPoison = 0x7FC0BEEFu;

int failures = 0;

void expect(bool condition, const char* what)
{
    if (condition) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

std::uint32_t bits_of(float value)
{
    std::uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
}

void fill_payload(camodel::GuardedDeviceBuffer& buffer, std::uint32_t word)
{
    auto* payload = static_cast<std::uint32_t*>(buffer.host_payload());
    for (std::size_t i = 0; i < buffer.payload_bytes / 4; ++i) payload[i] = word;
}

int parse_u32(std::string_view text, std::uint32_t& out)
{
    const auto result = std::from_chars(text.data(), text.data() + text.size(), out);
    return result.ec == std::errc() ? 0 : -1;
}

// The fixture geometry. block_size == kStripHeight puts one strip in a page,
// which is the shape the layout contract calls byte-identical to plain
// channel-major and the one the 2048-byte read-modify-write is defined on.
Shape make_shape()
{
    Shape shape;
    shape.num_requests = 1;
    shape.num_kv_heads = 4;
    shape.head_size = rot_oracle::kChannels;
    shape.block_size = kStripHeight;
    shape.num_blocks = 8;
    return shape;
}

// Request 0 lands on page 3: an ODD page in the middle of the pool, so every
// other page must still hold its poison afterwards and a kernel that derived
// its page from the block index rather than from slotMapping fails.
constexpr int kActivePage = 3;

struct Outcome {
    std::size_t key_bytes_wrong = 0;
    std::size_t value_bytes_wrong = 0;
    std::size_t scale_words_wrong = 0;
    std::size_t owned_nibbles_wrong = 0;
    std::size_t preserved_nibbles_moved = 0;  // unowned, inside a written strip
    std::size_t foreign_nibbles_moved = 0;    // unowned, outside every written strip
    std::size_t owned_still_poison = 0;
};

// Which (block, head, strip) windows this launch wrote, so an unowned nibble
// can be classified as "inside the strip the kernel legitimately rewrote" or
// "somewhere it had no business touching".
struct StripKey {
    int block;
    int head;
    int strip;
    bool operator==(const StripKey& other) const
    {
        return block == other.block && head == other.head && strip == other.strip;
    }
};

bool in_written_strip(const Shape& shape, const std::vector<StripKey>& strips,
                      std::size_t nibble)
{
    for (const StripKey& key : strips) {
        const std::size_t begin = strip_byte_offset(shape, key.block, key.head, key.strip) * 2;
        if (nibble >= begin && nibble < begin + static_cast<std::size_t>(kStripHeight) *
                                                    shape.head_size)
            return true;
    }
    return false;
}

// One launch, start to finish: build the fixture, poison the caches, run the
// kernel, and compare the whole of Global Memory against the oracle's image.
Outcome run_launch(camodel::Session& session, const Shape& shape, std::uint32_t head,
                   const std::vector<int>& token_slots, int zero_token,
                   std::vector<std::int8_t>& key_image, std::vector<std::int8_t>& value_image,
                   std::vector<float>& scale_image, bool first_launch, const char* tag)
{
    const int tokens = static_cast<int>(token_slots.size());
    const std::vector<float> signs = rot_oracle::canonical_signs();

    // ---- host side: the activations and the plan ------------------------
    std::vector<TokenPlan> plan;
    plan.reserve(token_slots.size());
    for (int t = 0; t < tokens; ++t)
        plan.push_back(plan_token(shape, 0, t, token_slots[static_cast<std::size_t>(t)]));

    const std::size_t channels =
        static_cast<std::size_t>(shape.num_kv_heads) * shape.head_size;
    std::vector<std::uint16_t> key_rows(static_cast<std::size_t>(tokens) * channels, 0);
    std::vector<std::uint16_t> value_rows(key_rows.size(), 0);
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < shape.num_kv_heads; ++h)
            for (int c = 0; c < shape.head_size; ++c) {
                const std::size_t off = static_cast<std::size_t>(t) * channels +
                                        static_cast<std::size_t>(h) * shape.head_size + c;
                key_rows[off] = float_to_bf16_rne(activation(0, 0, h, t, c, zero_token));
                value_rows[off] = float_to_bf16_rne(activation(0, 1, h, t, c, zero_token));
            }

    // The launch owns exactly one head, so the reference is taken for that
    // head alone: a head the launch did not run must come back looking
    // exactly like a head nobody wrote, and that is part of what the byte
    // comparison below checks rather than something it is corrected for.
    const Reference ref =
        compute_reference(shape, plan, signs, zero_token, key_image, value_image, scale_image,
                          static_cast<int>(head));

    // ---- device side ----------------------------------------------------
    camodel::GuardedDeviceBuffer d_key(key_rows.size() * 2);
    camodel::GuardedDeviceBuffer d_value(value_rows.size() * 2);
    camodel::GuardedDeviceBuffer d_key_cache(shape.cache_bytes());
    camodel::GuardedDeviceBuffer d_value_cache(shape.cache_bytes());
    camodel::GuardedDeviceBuffer d_scale(shape.scale_floats() * 4);
    camodel::GuardedDeviceBuffer d_slots(((token_slots.size() * 4 + 31) / 32) * 32);
    camodel::GuardedDeviceBuffer d_signs(signs.size() * 4);
    camodel::GuardedDeviceBuffer d_cu(64);

    std::memcpy(d_key.host_payload(), key_rows.data(), key_rows.size() * 2);
    std::memcpy(d_value.host_payload(), value_rows.data(), value_rows.size() * 2);
    std::memcpy(d_signs.host_payload(), signs.data(), signs.size() * 4);
    std::memcpy(d_key_cache.host_payload(), key_image.data(), key_image.size());
    std::memcpy(d_value_cache.host_payload(), value_image.data(), value_image.size());
    std::memcpy(d_scale.host_payload(), scale_image.data(), scale_image.size() * 4);
    {
        auto* slots = static_cast<std::int32_t*>(d_slots.host_payload());
        for (std::size_t i = 0; i < d_slots.payload_bytes / 4; ++i) slots[i] = -1;
        for (std::size_t i = 0; i < token_slots.size(); ++i)
            slots[i] = token_slots[i];
        auto* cu = static_cast<std::int32_t*>(d_cu.host_payload());
        cu[0] = 0;
        cu[1] = tokens;
    }
    d_key.upload();
    d_value.upload();
    d_key_cache.upload();
    d_value_cache.upload();
    d_scale.upload();
    d_slots.upload();
    d_signs.upload();
    d_cu.upload();

    // One work item, one block: head `head` of request 0 and nothing else.
    const std::uint32_t item = 0u * static_cast<std::uint32_t>(shape.num_kv_heads) + head;
    std::printf("  [%s] launch: head %u, %d token(s), page %d, %u work item, grid 1\n", tag,
                head, tokens, kActivePage, 1u);
    launch_turboquant_reshape_scatter_aiv(
        session.stream, 1u, static_cast<std::uint16_t*>(d_key.payload()),
        static_cast<std::uint16_t*>(d_value.payload()),
        static_cast<std::int8_t*>(d_key_cache.payload()),
        static_cast<std::int8_t*>(d_value_cache.payload()),
        static_cast<float*>(d_scale.payload()), static_cast<std::int32_t*>(d_slots.payload()),
        static_cast<float*>(d_signs.payload()), static_cast<std::int32_t*>(d_cu.payload()),
        1u,
        static_cast<std::uint32_t>(shape.num_kv_heads),
        static_cast<std::uint32_t>(shape.head_size),
        static_cast<std::uint32_t>(shape.block_size),
        static_cast<std::uint32_t>(shape.num_blocks), item, 1u, shape.inv_sqrt_len());
    camodel::check(aclrtSynchronizeStream(session.stream), "aclrtSynchronizeStream");

    d_key_cache.download();
    d_value_cache.download();
    d_scale.download();
    d_key.download();
    d_value.download();
    d_signs.download();
    d_key.assert_guards();
    d_value.assert_guards();
    d_key_cache.assert_guards();
    d_value_cache.assert_guards();
    d_scale.assert_guards();
    d_slots.download();
    d_slots.assert_guards();
    d_signs.assert_guards();

    if (first_launch) {
    }

    // ---- the comparison --------------------------------------------------
    const auto* dev_key = static_cast<const std::int8_t*>(d_key_cache.host_payload());
    const auto* dev_value = static_cast<const std::int8_t*>(d_value_cache.host_payload());
    const auto* dev_scale = static_cast<const float*>(d_scale.host_payload());

    std::vector<StripKey> strips;
    for (const TokenPlan& p : plan) {
        if (p.slot < 0) continue;
        StripKey key{p.block, static_cast<int>(head), p.strip};
        if (std::find(strips.begin(), strips.end(), key) == strips.end()) strips.push_back(key);
    }

    Outcome out;
    for (std::size_t i = 0; i < ref.key_cache.size(); ++i)
        if (dev_key[i] != ref.key_cache[i]) ++out.key_bytes_wrong;
    for (std::size_t i = 0; i < ref.value_cache.size(); ++i)
        if (dev_value[i] != ref.value_cache[i]) ++out.value_bytes_wrong;
    for (std::size_t i = 0; i < ref.scale_cache.size(); ++i)
        if (bits_of(dev_scale[i]) != bits_of(ref.scale_cache[i])) ++out.scale_words_wrong;

    const std::size_t nibbles = shape.cache_bytes() * 2;
    bool said = false;
    for (std::size_t n = 0; n < nibbles; ++n) {
        for (int pl = 0; pl < kPlanes; ++pl) {
            const std::int8_t* dev = pl == 0 ? dev_key : dev_value;
            const auto& want = pl == 0 ? ref.key_cache : ref.value_cache;
            const auto& own = pl == 0 ? ref.key_owned : ref.value_owned;
            const int got = read_nibble(dev, n);
            const int exp = read_nibble(want.data(), n);
            if (got == exp) continue;
            if (own[n]) {
                ++out.owned_nibbles_wrong;
                if (!said) {
                    std::fprintf(stderr,
                                 "  [%s] first owned divergence: plane %d nibble %zu, "
                                 "device %d oracle %d\n",
                                 tag, pl, n, got, exp);
                    said = true;
                }
            } else if (in_written_strip(shape, strips, n)) {
                ++out.preserved_nibbles_moved;
            } else {
                ++out.foreign_nibbles_moved;
            }
        }
    }
    // Owned lanes that still read as poison were never written at all; the
    // poison pattern 0x7FC0BEEF is 0xEF 0xBE 0xC0 0x7F, whose nibbles are a
    // fixed sequence the reference almost never reproduces, so this is a
    // separate count rather than a separate comparison.
    for (std::size_t n = 0; n < nibbles; ++n) {
        if (ref.key_owned[n] && read_nibble(dev_key, n) == read_nibble(key_image.data(), n) &&
            read_nibble(ref.key_cache.data(), n) != read_nibble(key_image.data(), n))
            ++out.owned_still_poison;
        if (ref.value_owned[n] &&
            read_nibble(dev_value, n) == read_nibble(value_image.data(), n) &&
            read_nibble(ref.value_cache.data(), n) != read_nibble(value_image.data(), n))
            ++out.owned_still_poison;
    }

    std::printf("  [%s] zero rows %d, clamped rows %d; key bytes wrong %zu, value bytes "
                "wrong %zu, scale words wrong %zu\n",
                tag, ref.zero_rows, ref.clamped_rows, out.key_bytes_wrong,
                out.value_bytes_wrong, out.scale_words_wrong);
    std::printf("  [%s] nibble census: owned wrong %zu, preserved-in-strip moved %zu, "
                "foreign moved %zu, owned unwritten %zu\n",
                tag, out.owned_nibbles_wrong, out.preserved_nibbles_moved,
                out.foreign_nibbles_moved, out.owned_still_poison);

    // Carry the device images forward so a following launch sees what the
    // previous one actually left, not what it was supposed to leave. That is
    // what makes the sequential ladder a test of the device's own
    // read-modify-write rather than of the oracle's.
    key_image.assign(dev_key, dev_key + ref.key_cache.size());
    value_image.assign(dev_value, dev_value + ref.value_cache.size());
    scale_image.assign(dev_scale, dev_scale + ref.scale_cache.size());
    return out;
}

void check(const Outcome& out, const char* tag)
{
    char what[192];
    std::snprintf(what, sizeof(what), "%s: owned nibbles bit-exact", tag);
    expect(out.owned_nibbles_wrong == 0, what);
    std::snprintf(what, sizeof(what), "%s: the RMW preserved every unowned lane of the strip",
                  tag);
    expect(out.preserved_nibbles_moved == 0, what);
    std::snprintf(what, sizeof(what), "%s: no byte outside a written strip moved", tag);
    expect(out.foreign_nibbles_moved == 0, what);
    std::snprintf(what, sizeof(what), "%s: every owned lane was written", tag);
    expect(out.owned_still_poison == 0, what);
    std::snprintf(what, sizeof(what), "%s: the scale plane is bit-exact", tag);
    expect(out.scale_words_wrong == 0, what);
    std::snprintf(what, sizeof(what), "%s: both cache planes are byte-identical", tag);
    expect(out.key_bytes_wrong == 0 && out.value_bytes_wrong == 0, what);
}

void poison_caches(const Shape& shape, std::vector<std::int8_t>& key_image,
                   std::vector<std::int8_t>& value_image, std::vector<float>& scale_image)
{
    key_image.assign(shape.cache_bytes(), 0);
    value_image.assign(shape.cache_bytes(), 0);
    scale_image.assign(shape.scale_floats(), 0.0f);
    for (std::size_t i = 0; i < key_image.size(); i += 4) {
        std::memcpy(key_image.data() + i, &kPoison, 4);
        std::memcpy(value_image.data() + i, &kPoison, 4);
    }
    for (float& value : scale_image) std::memcpy(&value, &kPoison, 4);
}

int run_main(int argc, char** argv)
{
    std::string_view mode = "canary";
    std::uint32_t head = 0;
    std::uint32_t tokens = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--head" && i + 1 < argc) {
            if (parse_u32(argv[++i], head) != 0 || head > 3) {
                std::fprintf(stderr, "--head must be 0..3\n");
                return 2;
            }
        } else if (arg == "--tokens" && i + 1 < argc) {
            if (parse_u32(argv[++i], tokens) != 0 || tokens == 0 ||
                tokens > static_cast<std::uint32_t>(kStripHeight)) {
                std::fprintf(stderr, "--tokens must be 1..%d\n", kStripHeight);
                return 2;
            }
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    const Shape shape = make_shape();
    if (!shape.valid()) {
        std::fprintf(stderr, "fixture shape is not valid\n");
        return 2;
    }
    std::printf("turboquant_reshape_scatter_aiv CAModel: mode %.*s, head %u, "
                "d %d, S %d, P %d, strip %zu B\n",
                static_cast<int>(mode.size()), mode.data(), head, shape.head_size,
                shape.block_size, shape.num_blocks, shape.strip_bytes());
    std::printf("  head %u isolated\n", head);

    // The fixture's own precondition: the rotated panel must stay exactly
    // representable, or none of the bit-equality gates below means anything.
    {
        const std::vector<float> signs = rot_oracle::canonical_signs();
        std::vector<float> x(static_cast<std::size_t>(shape.head_size));
        double peak = 0.0;
        for (int c = 0; c < shape.head_size; ++c) {
            const float raw = activation(0, 0, static_cast<int>(head), 0, c, -1);
            expect(bf16_is_exact(raw), "the fixture activation is exact in BF16");
            x[static_cast<std::size_t>(c)] = raw;
        }
        const RowRef row = quantise_row(shape, x, signs);
        for (float value : row.rotated) peak = std::fmax(peak, std::fabs(value));
        expect(peak < 16777216.0, "the rotated panel is exactly representable in FP32");
        std::printf("  fixture: peak |Pi . x| %.0f, absmax %.0f, exponent %d, scale %.9g\n",
                    peak, static_cast<double>(row.absmax), row.exponent,
                    static_cast<double>(row.scale));
    }

    camodel::Session session;
    std::vector<std::int8_t> key_image, value_image;
    std::vector<float> scale_image;

    if (mode == "canary" || mode == "zero") {
        poison_caches(shape, key_image, value_image, scale_image);
        std::vector<int> slots;
        for (std::uint32_t t = 0; t < tokens; ++t)
            slots.push_back(kActivePage * shape.block_size + static_cast<int>(t));
        const int zero_token = mode == "zero" ? 0 : -1;
        const Outcome out = run_launch(session, shape, head, slots, zero_token, key_image,
                                       value_image, scale_image, true, "run");
        check(out, "run");
        if (mode == "zero") {
            // The branch under test: an all-zero row must store scale 1.0f and
            // all-zero codes, and the oracle must have produced that case at
            // all -- a gate that passes because the fixture never reached the
            // branch is worse than no gate.
            const std::vector<float> signs = rot_oracle::canonical_signs();
            std::vector<float> zeros(static_cast<std::size_t>(shape.head_size), 0.0f);
            const RowRef zero_row = quantise_row(shape, zeros, signs);
            expect(zero_row.zero_row, "the zero fixture reaches the absmax == 0 branch");
            expect(zero_row.scale == 1.0f, "the absmax == 0 branch stores scale 1.0f");
            bool all_zero = true;
            for (int code : zero_row.q)
                if (code != 0) all_zero = false;
            expect(all_zero, "the absmax == 0 branch stores all-zero codes");
        }
    } else if (mode == "sweep") {
        for (std::uint32_t n = 1; n <= static_cast<std::uint32_t>(kStripHeight); ++n) {
            poison_caches(shape, key_image, value_image, scale_image);
            std::vector<int> slots;
            for (std::uint32_t t = 0; t < n; ++t)
                slots.push_back(kActivePage * shape.block_size + static_cast<int>(t));
            char tag[32];
            std::snprintf(tag, sizeof(tag), "n=%u", n);
            const Outcome out = run_launch(session, shape, head, slots, -1, key_image,
                                           value_image, scale_image, n == 1, tag);
            check(out, tag);
        }
    } else if (mode == "sequential") {
        // The read-modify-write ladder. Three launches into ONE strip: token 0,
        // then token 1, then tokens 2..4. Each launch's reference starts from
        // what the PREVIOUS launch actually left in Global Memory, so a kernel
        // that zeroed the lanes it does not own fails at step two and a kernel
        // that wrote the right nibbles into the wrong token slot fails at
        // step two as well.
        poison_caches(shape, key_image, value_image, scale_image);
        const int base = kActivePage * shape.block_size;
        const std::vector<std::vector<int>> steps = {
            {base + 0}, {base + 1}, {base + 2, base + 3, base + 4}};
        int step = 0;
        for (const std::vector<int>& slots : steps) {
            char tag[32];
            std::snprintf(tag, sizeof(tag), "step%d", step);
            const Outcome out = run_launch(session, shape, head, slots, -1, key_image,
                                           value_image, scale_image, step == 0, tag);
            check(out, tag);
            ++step;
        }
    } else {
        std::fprintf(stderr, "--mode must be canary, sweep, sequential or zero\n");
        return 2;
    }

    if (failures == 0) {
        std::printf("turboquant_reshape_scatter_aiv CAModel (%.*s, head %u): PASS\n",
                    static_cast<int>(mode.size()), mode.data(), head);
        return 0;
    }
    std::fprintf(stderr, "turboquant_reshape_scatter_aiv CAModel (%.*s, head %u): %d FAILURES\n",
                 static_cast<int>(mode.size()), mode.data(), head, failures);
    return 1;
}

}  // namespace scatter_rig

int main(int argc, char** argv)
{
    std::string_view op = "both";
    std::vector<char*> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--op" && i + 1 < argc) {
            op = argv[++i];
            continue;
        }
        rest.push_back(argv[i]);
    }
    const int n = static_cast<int>(rest.size());
    if (op == "prefill") return prefill_rig::run_main(n, rest.data());
    if (op == "scatter") return scatter_rig::run_main(n, rest.data());
    if (op != "both") {
        std::fprintf(stderr, "--op must be prefill, scatter or both\n");
        return 2;
    }
    // Both rigs take their own flags; `both` runs each with its defaults so one invocation
    // covers the batch writer and the scatter writer.
    char* bare[1] = {argv[0]};
    const int prefill = prefill_rig::run_main(1, bare);
    const int scatter = scatter_rig::run_main(1, bare);
    return prefill != 0 ? prefill : scatter;
}
