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
// The fused decode's gold model: the cache image, the query, and `expected` -- the attention
// output computed in DOUBLE.
//
// The cache is structured rather than random, and that is what makes the reference exact: every
// row of a 64-row tile carries the same key level, so softmax(q k^T * scale) v collapses to a
// closed form over the tiles, which this evaluates in double with std::exp. The device computes
// the same quantity in fp32 with an online softmax over the same tiles, so the comparison has a
// reference whose own rounding is orders of magnitude finer than the thing being measured --
// and a wrong tile count, a wrong live-row mask or a skipped page all move the closed form.
//
// `element` selects the format every 16-bit input is encoded in and every 16-bit output is
// compared in, and Launch::element_type with it, so one fixture cannot be built for one
// instance and launched at the other. It changes NO fp32 value here: the device's arithmetic up
// to the output cast is identical for fp16 and bf16.

#include "csrc/attention/turboquant/include/turboquant_fused_decode_opt.hpp"
#include "turboquant_test_common.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace fused_decode_fixture {
namespace fd = ascend_micro::fused_decode_opt;
namespace tq = vllm_ascend::turboquant;
inline std::size_t aligned(std::size_t n) { return (n + 31) / 32 * 32; }

// ------------------------------------------------------------- the element format, in the gold model
//
// The operator is instantiated at fp16 and at bf16, and `scalar_t` is the QUERY/OUTPUT element type,
// not the cache format. For the oracle below to be a comparison rather than a coincidence, EVERY
// value the device holds in 16 bits has to be held in 16 bits here too -- and in the SAME 16 bits.
// Rounding an fp32 oracle at the very end is only the right model where the device's own rounding is
// also at the very end, and on this operator that is true of exactly one value (the output) and
// false of two others:
//
//   output      the device accumulates the context in fp32 and casts ONCE, at the final GM burst
//               (WriteHeads). So the gold model computes in fp64, narrows to fp32, and rounds to
//               the element format here -- one rounding, in the same place.
//
//   raw query   the device READS bf16 (or fp16) from GM and widens it to fp32. So the gold model
//               must build the raw query in the element format and then feed the WIDENED value to
//               the rest of the oracle -- not the fp32 ideal it rounded from. `raw_query` below
//               therefore refuses a value the format cannot hold exactly instead of rounding it,
//               because the whole reason this fixture can hold the raw-query entry to the
//               pre-rotated entry's oracle is that Pi(raw) reproduces `query` BIT FOR BIT.
//
//   gate        same: the device reads the gate in the element format and widens it, so
//               `expected_row`'s kGated stage divides by 1 + exp(-widened), never by
//               1 + exp(-gate_value), and `gate_logits` refuses an inexact logit.
//
// The intervening arithmetic -- scores, the online softmax, the probabilities, the accumulator, the
// Split-K partials and their merge, and the Pi basis itself -- is fp32 on the device for both
// instances, so it stays fp64-then-fp32 here for both. Carrying it in 16 bits would model a kernel
// this is not.
enum class Element : std::uint32_t { kFp16 = 0, kBf16 = 1 };

inline const char* element_name(Element e) { return e == Element::kBf16 ? "bf16" : "fp16"; }
inline unsigned element_mantissa_bits(Element e) { return e == Element::kBf16 ? 7u : 10u; }

// Positive normal results are in [1,15]. Round the FP32 significand before
// rebiasing, so a mantissa carry correctly increments the FP16 exponent.
inline std::uint16_t reference_half(float value)
{
    if (!(value >= 1 && value <= 15)) throw std::runtime_error("oracle result outside fixture range");
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    return static_cast<std::uint16_t>(((bits + 0xfffu + ((bits >> 13) & 1u)) >> 13) - (112u << 10));
}

// The same for bf16, and deliberately NOT by the add-and-shift that
// tq_test::float_to_bf16_rne uses: this one splits the word, decides the round
// half-to-even explicitly and reassembles, so the two implementations agree only if both are right.
// tests/cpu/test_fused_decode_opt_fixture.cpp pins that agreement over the fixture's own values and
// over the tie cases. A carry out of the 7-bit significand lands in the exponent field by
// construction, exactly as it does for fp16 above, because the fields are adjacent.
inline std::uint16_t reference_bfloat16(float value)
{
    if (!(value >= 1 && value <= 15)) throw std::runtime_error("oracle result outside fixture range");
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    const std::uint32_t keep = bits >> 16;             // sign | exponent | 7 significand bits
    const std::uint32_t dropped = bits & 0xffffu;      // the 16 bits bf16 does not store
    const bool round_up = dropped > 0x8000u || (dropped == 0x8000u && (keep & 1u));
    return static_cast<std::uint16_t>(keep + (round_up ? 1u : 0u));
}

// The correctly-rounded 16-bit word of an fp32 oracle value, in the instance's own format.
inline std::uint16_t reference_bits(Element e, float value)
{
    return e == Element::kBf16 ? reference_bfloat16(value) : reference_half(value);
}

// The value the DEVICE sees after widening a stored word: exact for both formats, since fp32 holds
// every fp16 and every bf16 value. This is what an oracle downstream of a 16-bit input must use.
inline float widen_bits(Element e, std::uint16_t bits)
{
    return e == Element::kBf16 ? tq_test::bf16_to_float(bits)
                               : tq_test::half_to_float(bits);
}

// Encode an INPUT the device will read in the element format, refusing anything the format cannot
// hold exactly. An input the fixture had to round would make `expected` an approximation of a
// different kernel's answer, and the exact-boundary rungs would then be measuring the fixture.
inline std::uint16_t exact_input_bits(Element e, float value, const char* what)
{
    const std::uint16_t bits = e == Element::kBf16 ? tq_test::float_to_bf16_rne(value)
                                                  : tq_test::float_to_half_rne(value);
    if (widen_bits(e, bits) != value) {
        char message[160];
        std::snprintf(message, sizeof message,
                      "fused-opt fixture: %s is not exactly representable in %s", what,
                      element_name(e));
        throw std::runtime_error(message);
    }
    return bits;
}

// ---------------------------------------------------------------- the raw-query Pi basis
//
// The pre-rotated entry takes `queryRot` already in the rotated basis. The RAW-query entry
// takes the query in the original basis plus the basis tables, rotates it once in a prologue
// (TurboQuantQueryBasis::Rotate) and can rotate the context back on egress
// (FinishHeads). This builds every input that path needs, and the host reference for it.
//
// THE BASIS. TurboQuantCodec4::ApplyPi is `x *= piSigns; FWHT(x); x *= piSigns`, and its FWHT
// is the unnormalised Walsh-Hadamard butterfly followed by one `Muls(invSqrtLen)`. So
//
//     Pi = (1 / sqrt(D)) . diag(s) . H . diag(s),   H[i][j] = (-1)^popcount(i & j)
//
// with H symmetric and H^2 = D.I, which makes Pi symmetric and Pi^2 = I EXACTLY -- not
// approximately. That is what lets the raw-query fixture reuse the pre-rotated oracle: the raw
// query whose rotation is a given target IS Pi(target).
//
// THE TABLE POOL. `rotTables` is the codec's whole constant pool, which TurboQuantCodec<4>
// ::Init copies from GM and carves up in this order (vecLen = 256, batchRows = 1, so
// ConstTableWords = 7*256 + 2*256 + 16 = 2320 words):
//
//     stage 0 sign_ (256 float) | stage 0 xorOffset_ (256 uint32) | stage 1 ... | stage 2 ...
//     evenOffset_ (128 uint32) | oddOffset_ (128) | expandOffset_ (256) | oddSelect_ (256
//     float) | centroid_ (16 float)
//
// ApplyPi reads ONLY the three early stages: `Gather(swap_, x, xorOffset_[stage], 0, n)` then
// `Mul(tmp, x, sign_[stage], n)` then `Add(x, swap_, tmp, n)`, which is
// x'[i] = x[i ^ stride] + sign[i] * x[i] -- the butterfly, with sign[i] = +1 when
// (i & stride) == 0 and -1 otherwise, and the gather index a BYTE offset (kGatherSrcBase is 0,
// and the vector service's own gather indices are byte offsets too). The rest of the pool
// belongs to Quantize4Bit/Dequantize4Bit, which the basis never calls, so it is left zero --
// zero is also an in-bounds byte offset, so a future caller that did gather with it would read
// element 0 rather than out of bounds.
struct RawQueryBasis {
    static constexpr int kEarlyStages = 3;  // TurboQuantCodec<4>::kEarlyStages
    static constexpr int kLevels = 16;      // 1 << BITS
    static constexpr int kPackFactor = 2;   // 8 / BITS
    static constexpr int kH16 = 16;

    int dim;
    std::vector<float> signs;

    explicit RawQueryBasis(int d = static_cast<int>(fd::kHeadSize))
        : dim(d), signs(static_cast<std::size_t>(d), 0.0f)
    {
        if (d <= 0 || (d & (d - 1)) != 0) throw std::invalid_argument("Pi needs a power-of-two D");
        for (int i = 0; i < d; ++i) signs[std::size_t(i)] = sign_at(i);
    }

    // Deterministic, so the fixture, the oracle and the uploaded table are one function.
    static float sign_at(int channel)
    {
        const unsigned x = static_cast<unsigned>(channel);
        return ((x * 2654435761u) >> 13) & 1u ? -1.0f : 1.0f;
    }
    static float inv_sqrt_len(int d)
    {
        float r = 1.0f;
        for (int n = d; n > 1; n >>= 2) r *= 0.5f;  // exact: D is a power of four here
        return r;
    }

    // Pi applied to one vector, in double, stage for stage as ApplyPi would.
    std::vector<float> apply(const float* x) const
    {
        std::vector<double> v(static_cast<std::size_t>(dim), 0.0);
        for (int i = 0; i < dim; ++i) v[std::size_t(i)] = double(x[i]) * signs[std::size_t(i)];
        for (int stride = 1; stride < dim; stride <<= 1)
            for (int i = 0; i < dim; ++i)
                if ((i & stride) == 0) {
                    const double a = v[std::size_t(i)], b = v[std::size_t(i ^ stride)];
                    v[std::size_t(i)] = a + b;
                    v[std::size_t(i ^ stride)] = a - b;
                }
        const double scale = double(inv_sqrt_len(dim));
        std::vector<float> out(static_cast<std::size_t>(dim), 0.0f);
        for (int i = 0; i < dim; ++i)
            out[std::size_t(i)] = static_cast<float>(v[std::size_t(i)] * signs[std::size_t(i)] * scale);
        return out;
    }

    // The codec's GM constant pool, exactly as Init unpacks it.
    std::vector<std::int32_t> tables() const
    {
        const std::size_t words = 9u * static_cast<std::size_t>(dim) + kLevels;
        std::vector<std::int32_t> pool(words, 0);
        std::size_t offset = 0;
        for (int stage = 0; stage < kEarlyStages; ++stage) {
            const int stride = 1 << stage;
            for (int i = 0; i < dim; ++i) {
                const float sign = (i & stride) == 0 ? 1.0f : -1.0f;
                std::int32_t bits;
                std::memcpy(&bits, &sign, 4);
                pool[offset + std::size_t(i)] = bits;
            }
            offset += std::size_t(dim);
            for (int i = 0; i < dim; ++i)
                pool[offset + std::size_t(i)] =
                    static_cast<std::int32_t>(std::uint32_t(i ^ stride) * sizeof(float));
            offset += std::size_t(dim);
        }
        if (offset != 2u * static_cast<std::size_t>(kEarlyStages) * static_cast<std::size_t>(dim))
            throw std::runtime_error("the early-stage pool layout drifted");
        return pool;
    }

    // The 16x16 Hadamard in fp16, which only the Cube prologue chunk reads.
    std::vector<std::uint16_t> h16() const
    {
        std::vector<std::uint16_t> out(static_cast<std::size_t>(kH16) * kH16, 0);
        for (int i = 0; i < kH16; ++i)
            for (int j = 0; j < kH16; ++j) {
                unsigned bits = static_cast<unsigned>(i & j), parity = 0;
                while (bits) { parity ^= bits & 1u; bits >>= 1; }
                out[std::size_t(i) * kH16 + j] = parity ? 0xbc00 : 0x3c00;  // -1.0h, +1.0h
            }
        return out;
    }
};

// FP32 pre-rotated GQA queries, source-format KV4 NZ cache, FP16 context.
// Tile scores vary by request, KV head and query head. Values vary by tile
// and channel. A scalar FP64 attention sum supplies an independent oracle.
//
// RAGGED CONTEXTS. `context` need not be a multiple of `page_tokens` or of the 64-row Cube
// tile: `pages` is the ceiling, the last page holds `context - (pages - 1) * page_tokens`
// live tokens, and the last live tile holds `context % 64` of them when that is nonzero. The
// cache is still filled over EVERY row of every mapped page, tail rows included, which is what
// makes the kernel's masking observable -- a tail row it fails to mask carries the same key
// level and the same value levels as the live rows of its own tile, so including it weights
// that tile by 64 rows instead of by its true count and the oracle below disagrees.
//
// The oracle weights each tile by its live row count, which is the same thing the CAModel
// runner's own oracle does (`numerator += valid * weight * ...` in
// tests/sim/kernel/test_camodel_fused_decode_opt.cpp). All 64 rows of a tile share one key
// level, so a tile of v live rows contributes v * exp(score) to both sums; at v = 64 that is
// an exact power-of-two scaling of both, so an ALIGNED context's expected values are
// bit-identical to the unweighted form this replaced. tests/cpu/test_fused_decode_opt_fixture
// pins that.
struct Fixture {
    int context, requests, kv_heads, group_heads, page_tokens, pages, physical_pages;
    // The instance this fixture is the gold model for. It changes NO fp32 value below -- the query,
    // the cache, the scales and `expected` are identical for both instances, because the device's
    // arithmetic up to the output cast is identical -- and it selects (a) the format every 16-bit
    // input is encoded in and every 16-bit output is compared in, and (b) Launch::element_type, so
    // one fixture cannot be built for one instance and launched at the other.
    Element element;
    std::vector<float> query, scales, expected;
    std::vector<std::uint8_t> key, value;
    std::vector<std::int32_t> table, lengths;
    Fixture(int n, int b, int h, int g, int block, Element e = Element::kFp16)
        : context(n), requests(b), kv_heads(h), group_heads(g), page_tokens(block),
          pages(block > 0 && n > 0 ? (n + block - 1) / block : 0), physical_pages(2 * b * pages + 2),
          element(e)
    {
        // n is no longer required to divide the page: see RAGGED CONTEXTS above. The page size
        // still has to be 64 or 128, because that is what valid_launch admits and what the
        // NZ-tiled ingest is built on.
        if (n <= 0 || n > 16384 || (block != 64 && block != 128) || b < 1 || b > 16 ||
            h < 1 || h > 4 || g < 1 || g > 16)
            throw std::invalid_argument("invalid fused-opt fixture geometry");
        query.resize(std::size_t(b) * h * g * 256);
        expected.resize(query.size());
        key.resize(std::size_t(physical_pages) * block * h * 128);
        value.resize(key.size());
        for (std::size_t i = 0; i < key.size(); ++i)
            key[i] = value[i] = static_cast<std::uint8_t>(0x7FC0BEEFu >> ((i % 4) * 8));
        float poison; const std::uint32_t word = 0x7FC0BEEF;
        std::memcpy(&poison, &word, 4);
        scales.assign(std::size_t(physical_pages) * block * 8, poison);
        table.assign(aligned(std::size_t(b) * pages * 4) / 4, -1);
        lengths.assign(aligned(b * 4) / 4, 0);
        const float scale = tq::TurboQuantModeTraits<tq::TurboQuantMode::KV4_FP8>::kGain * 2;
        for (int request = 0; request < b; ++request) {
            lengths[request] = n;
            for (int page = 0; page < pages; ++page) {
                const int physical = 2 * (request * pages + page) + 1;
                table[request * pages + page] = physical;
                for (int row = 0; row < block; ++row) for (int head = 0; head < h; ++head) {
                    const int tile = (page * block + row) / 64;
                    const unsigned kn = key_level(request, head, tile);
                    const auto si = (std::size_t(physical) * block + row) * 8;
                    scales[si + head] = scales[si + h + head] = scale;
                    for (int col = 0; col < 128; ++col) {
                        const auto off = std::size_t(physical) * block * h * 128 +
                            tq::NzTiledPackedByte(row, head, col, h, 128);
                        key[off] = static_cast<std::uint8_t>(kn | (kn << 4));
                        value[off] = static_cast<std::uint8_t>(value_level(request, head, tile, col) |
                            (value_level(request, head, tile, col + 128) << 4));
                    }
                }
            }
            for (int qh = 0; qh < h * g; ++qh) {
                const int head = qh / g, coefficient = 1 + qh % 3;
                const auto base = (std::size_t(request) * h * g + qh) * 256;
                std::fill(query.begin() + base, query.begin() + base + 256, coefficient / 256.0f);
                for (int col = 0; col < 256; ++col) {
                    double numerator = 0, denominator = 0;
                    for (int tile = 0; tile < tiles(); ++tile) {
                        const int live = valid_rows(tile);
                        const double score = (2 * key_level(request, head, tile) + 1) * coefficient / 16.0;
                        const double weight = live * std::exp(score);
                        numerator += weight * (2 * value_level(request, head, tile, col) + 1);
                        denominator += weight;
                    }
                    expected[base + col] = static_cast<float>(numerator / denominator);
                }
            }
        }
    }
    // 64-row Cube tiles over the live context, and the live rows of one of them. The last tile
    // is short exactly when context % 64 != 0; every other tile is full, including the ones in
    // a page that is itself partially live.
    int tiles() const { return (context + 63) / 64; }
    int valid_rows(int tile) const
    {
        const int rest = context - 64 * tile;
        return rest < 64 ? rest : 64;
    }
    // The live tokens of one logical page, which is page_tokens for every page but the last.
    int page_rows(int page) const
    {
        const int rest = context - page_tokens * page;
        return rest < page_tokens ? rest : page_tokens;
    }
    static unsigned key_level(int request, int head, int tile)
    { return (tile + 2 * request + 3 * head) % 8; }
    static unsigned value_level(int request, int head, int tile, int channel)
    {
        const int col = channel < 128 ? channel : (channel - 128 + 3) % 128;
        return (tile + 3 * request + 5 * head + col) % 8;
    }
    // The raw-query inputs. `raw_query` is the ORIGINAL-basis query whose rotation is the
    // pre-rotated `query` above: Pi is an involution, so that is simply Pi(query). It is encoded in
    // THIS FIXTURE'S element format and this throws rather than rounding, because the whole point
    // of building it this way is that the device's prologue reproduces `query` bit for bit and the
    // attention oracle therefore still applies at output stage 0.
    //
    // IT IS EXACT IN BOTH FORMATS, and that is a measured property of this basis rather than a
    // lucky margin: `query` is the constant vector (coefficient / 256) * 1 with coefficient in
    // {1,2,3}, and Pi of a constant vector is +-coefficient * k / 2048 for integer k = (H s)_i / 2,
    // so the entries are small integers over a power of two. Over the fixture's own sign vector the
    // widest significand that actually occurs is 7 bits -- which bf16's 7 stored bits hold exactly,
    // as fp16's 10 already did. 7 is the MEASURED width, not a bound on it: a different sign
    // function or a non-constant query could push it to 9 and break bf16 while leaving fp16 intact,
    // which is exactly why this check is an assertion in the gold model and not a comment.
    std::vector<std::uint16_t> raw_query(const RawQueryBasis& basis) const
    {
        if (basis.dim != static_cast<int>(fd::kHeadSize))
            throw std::invalid_argument("the raw-query basis must match the head size");
        std::vector<std::uint16_t> out(query.size(), 0);
        const std::size_t rows = query.size() / fd::kHeadSize;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::vector<float> image = basis.apply(query.data() + row * fd::kHeadSize);
            for (std::size_t c = 0; c < fd::kHeadSize; ++c)
                out[row * fd::kHeadSize + c] = exact_input_bits(element, image[c], "the raw query");
        }
        return out;
    }

    // Gate logits for TurboQuantOutputStage::kGated, exactly representable in both formats. Spread
    // over [-2, 2] in quarters so the sigmoid is neither saturated nor constant, and nowhere
    // near kGateLogitFloor, which is a clamp this fixture is not trying to exercise. A quarter-step
    // in [-2, 2] needs at most 3 significand bits, so both formats hold every logit exactly and
    // `gate_widened` below is the same number as `gate_value` -- asserted, not assumed.
    static float gate_value(std::size_t index)
    { return static_cast<float>(static_cast<int>(index % 17) - 8) * 0.25f; }
    std::vector<std::uint16_t> gate_logits() const
    {
        std::vector<std::uint16_t> out(query.size(), 0);
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = exact_input_bits(element, gate_value(i), "a gate logit");
        return out;
    }
    // The gate logit AS THE DEVICE SEES IT: the stored word of this fixture's format, widened. The
    // kGated oracle divides by 1 + exp(-this), not by 1 + exp(-gate_value), so that a format which
    // could not hold a logit exactly would produce a wrong oracle rather than a silently different
    // one. (With the quarter-step logits above the two coincide, and exact_input_bits enforces it.)
    float gate_widened(std::size_t index) const
    { return widen_bits(element, exact_input_bits(element, gate_value(index), "a gate logit")); }

    // The expected output for one (token, query head) row under a given output stage.
    //   0 kRotatedBasis: the context as the pre-rotated entry emits it
    //   1 kUnrotated:    Pi applied once more
    //   2 kGated:        that, divided by 1 + exp(-gate)
    std::vector<float> expected_row(const RawQueryBasis& basis, std::size_t row,
                                    unsigned output_stage) const
    {
        const float* source = expected.data() + row * fd::kHeadSize;
        if (output_stage == 0) return std::vector<float>(source, source + fd::kHeadSize);
        std::vector<float> out = basis.apply(source);
        if (output_stage == 1) return out;
        for (std::size_t c = 0; c < fd::kHeadSize; ++c) {
            const float g = gate_widened(row * fd::kHeadSize + c);
            out[c] = static_cast<float>(static_cast<double>(out[c]) /
                                        (1.0 + std::exp(-static_cast<double>(g))));
        }
        return out;
    }

    // The raw-query launch. Same geometry as launch(), plus the prologue share and the stage.
    fd::Launch launch_raw(int splits, unsigned output_stage, unsigned cube_chunk_vectors) const
    {
        if (output_stage > 2) throw std::invalid_argument("output stage must be 0, 1 or 2");
        if (cube_chunk_vectors != 0 && cube_chunk_vectors != 16)
            throw std::invalid_argument("the prologue Cube chunk is 0 or 16 vectors");
        fd::Launch c = launch(splits);
        const unsigned vectors = static_cast<unsigned>(requests) * c.heads;
        c.prologue_vectors_per_block = (vectors + c.grid - 1) / c.grid;
        c.prologue_cube_chunk_vectors = cube_chunk_vectors;
        c.output_stage = output_stage;
        return c;
    }

    fd::Launch launch(int splits) const
    {
        if (splits < 1 || splits > 8) throw std::invalid_argument("fused-opt splits must be 1..8");
        fd::Launch c;
        c.element_type = static_cast<std::uint32_t>(element);
        c.tokens = requests; c.kv_heads = kv_heads; c.heads = kv_heads * group_heads;
        c.block_size = page_tokens; c.max_blocks = pages; c.splits = splits;
        c.heads_per_task = group_heads; c.fused_context_limit = 0;
        const unsigned tasks = requests * kv_heads * splits;
        c.tasks_per_block = (tasks + 31) / 32;
        c.grid = (tasks + c.tasks_per_block - 1) / c.tasks_per_block;
        c.reduce_tasks_per_block = (requests * c.heads + c.grid - 1) / c.grid;
        if (!fd::valid_launch(c.grid, c.tokens, c.heads, c.kv_heads, c.head_size,
                c.block_size, c.max_blocks, c.splits, c.heads_per_task,
                c.tasks_per_block, c.reduce_tasks_per_block))
            throw std::invalid_argument("unsupported fused-opt launch");
        return c;
    }
    double cache_bytes() const
    { return double(requests) * context * (kv_heads * 256 + 8 * 4); }
};
}  // namespace fused_decode_fixture
