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

// THE FUSED DECODE'S HOST TILING DESCRIPTORS.
//
// WHAT THIS FILE IS FOR. Every quantity the modulo-scheduled decode used to DERIVE on the
// DaVinci scalar unit -- the tile count of a Split-K worker, the worker's logical page span,
// the live row count of its final tile, and the (token, kv head, split, chunk) a grid block's
// task range starts at -- is computed on the CPU and handed to the device as plain words.
// The device then does no integer division, walks no block-table span to count tiles, and
// takes no data-dependent branch to advance a tile cursor.
//
// WHY, ON dav-3510 SPECIFICALLY. The SPU issues in order for the whole core. A scalar integer
// divide, or a branch whose condition depends on a scalar load from GM, stalls the issue queue;
// everything behind it in program order -- including the MTE2 descriptor writes that prefetch
// the next ring slot and the Mmad issue that keeps the Cube systolic array fed -- waits. The
// cursor arithmetic this file replaces sat between a tile's softmax and the next tile's stage,
// which is exactly the dependence chain the three-zone schedule exists to keep full.
//
// DEPENDENCY-FREE ON PURPOSE. <cstdint> only, no CANN and no CCE. The device kernel, the host
// launcher, the host gate and the Torch adapter all include this one declaration rather than
// each carrying a copy of the layout, because the device reads these words by OFFSET and
// nothing in the toolchain would diagnose two sides disagreeing about which word is which.

#pragma once
#include <cstddef>
#include <cstdint>

namespace ascend_micro::fused_decode {

// The pipeline's tile is 64 KV rows. It is kCubeTileRows in turboquant_layout.h; restated here
// because this header must not depend on the device tree, and pinned to it by a static_assert
// in the kernel translation unit that includes both.
constexpr std::uint32_t kDecodeTileRows = 64;
constexpr std::uint32_t kDecodeTileRowsShift = 6;
static_assert(kDecodeTileRows == (1u << kDecodeTileRowsShift));

// ONE DESCRIPTOR PER (token, split). The first eight words are the mandated tiling contract;
// the next three are what the device's closed-form cursor needs on top of it, and they live
// inside the same 64 bytes rather than in a second array. The padding is NAMED and the host
// zeroes it: this structure is uploaded to GM verbatim, so implicit padding would be
// uninitialised bytes crossing into device memory.
//
// Every field describes ONE Split-K worker of ONE token. A worker that has no work -- a split
// past the end of the sequence, or the duplicate splits of a token short enough to run fused --
// is the all-zero descriptor, and `num_tiles == 0` is the single predicate both the Cube and
// the Vector role branch on to skip the pipeline. That predicate is now a host-computed word
// instead of the return value of a loop over the block table, which is what made the two roles'
// agreement on it a thing to argue about rather than a thing to read.
struct alignas(64) TurboQuantDecodeTilingData {
    std::uint32_t num_tiles;        // exact number of 64-token tiles for this split task
    std::uint32_t regular_tiles;    // number of full 64-row tiles (unmasked)
    std::uint32_t tail_rows;        // length of live tail in final tile (0 if aligned)
    std::uint32_t block_start_idx;  // logical page offset in blockTable for this split
    std::uint32_t blocks_in_split;  // number of pages in this split
    std::uint32_t head_dim;         // D = 256
    std::uint32_t group_heads;      // GQA ratio (e.g. 6 or 16)
    std::uint32_t split_k;          // split count S
    // The tile index INSIDE block_start_idx at which this worker's first tile begins. Zero for
    // every production worker. It is what lets an isolated single-tile canary name "page 7,
    // second 64-row tile" as a descriptor instead of as a probe branch inside the device's
    // cursor: the diagnostic ABI that used to be compared against UINT32_MAX once per tile on
    // every production launch is now this word, and the comparison is gone.
    std::uint32_t tile_origin;
    // log2(block_size / 64). The closed-form cursor turns a tile index into (page, row base)
    // with one shift and one mask, so the page size has to be a power-of-two multiple of 64;
    // build_decode_tiling rejects anything else rather than let the device divide.
    std::uint32_t tiles_per_page_shift;
    std::uint32_t flags;            // kDecodeSplitLive, kDecodeSplitCarriesTail
    // ceil(group_heads / heads_per_task): how many head chunks one (token, kv head, split)
    // splits into. Init used to take it as a CeilDiv of a division, which were the last two
    // runtime integer divides left anywhere on the device side of this operator.
    std::uint32_t chunks_per_group;
    std::uint32_t reserved[4];
};
static_assert(sizeof(TurboQuantDecodeTilingData) == 64, "the descriptor is one 64-byte line");
static_assert(alignof(TurboQuantDecodeTilingData) == 64, "64-byte aligned, as the contract says");

constexpr std::uint32_t kDecodeSplitLive = 1u << 0;
constexpr std::uint32_t kDecodeSplitCarriesTail = 1u << 1;

// Word offsets of the fields the DEVICE reads, for the scalar loads in the task prologue. The
// device reads words, not a struct: a __gm__ struct load would be several scalar loads with no
// guarantee about their order, and these five are one cache line.
enum DecodeTilingWord : std::uint32_t {
    kTilingWordNumTiles = 0,
    kTilingWordRegularTiles = 1,
    kTilingWordTailRows = 2,
    kTilingWordBlockStart = 3,
    kTilingWordBlocksInSplit = 4,
    kTilingWordHeadDim = 5,
    kTilingWordGroupHeads = 6,
    kTilingWordSplitK = 7,
    kTilingWordTileOrigin = 8,
    kTilingWordTilesPerPageShift = 9,
    kTilingWordFlags = 10,
    kTilingWordChunksPerGroup = 11,
};
constexpr std::uint32_t kDecodeTilingWords = sizeof(TurboQuantDecodeTilingData) / sizeof(std::uint32_t);
static_assert(kDecodeTilingWords == 16);

// ONE RECORD PER GRID BLOCK: where that block's task range starts, so the device never divides
// to find out. Process() used to decode its first task id with six integer div/mods -- chunk,
// splitTask, split, tokenKvHead, kvHead, token -- and then decode EVERY later task the same
// way. It now reads this record once and advances an odometer.
//
// The two reduce cursors are the two vector subcores' halves of the block's reduction share.
// Reduce() split that share with `task / numHeads_` and `task % numHeads_` per task; the halving
// itself is a shift, but the (token, head) each half starts at was a divide, so both starts are
// here.
struct alignas(32) TurboQuantDecodeDispatch {
    std::uint32_t token;        // attention task cursor at block entry
    std::uint32_t kv_head;
    std::uint32_t split;
    std::uint32_t chunk;
    std::uint32_t reduce_token;     // subcore 0's first reduction task
    std::uint32_t reduce_head;
    std::uint32_t reduce_token_hi;  // subcore 1's first reduction task
    std::uint32_t reduce_head_hi;
};
static_assert(sizeof(TurboQuantDecodeDispatch) == 32);
constexpr std::uint32_t kDecodeDispatchWords =
    sizeof(TurboQuantDecodeDispatch) / sizeof(std::uint32_t);
static_assert(kDecodeDispatchWords == 8);

// THE BUFFER. Descriptors first, so the array the device indexes most often starts at offset
// zero and every element is 64-byte aligned; the per-block dispatch records follow. Both arrays
// are read by scalar loads on both roles and nothing writes them, so they can share one
// read-only GM allocation.
constexpr std::size_t decode_tiling_words(const std::uint32_t tokens, const std::uint32_t splits,
                                          const std::uint32_t grid)
{
    return static_cast<std::size_t>(tokens) * splits * kDecodeTilingWords +
           static_cast<std::size_t>(grid) * kDecodeDispatchWords;
}
constexpr std::size_t decode_tiling_bytes(const std::uint32_t tokens, const std::uint32_t splits,
                                          const std::uint32_t grid)
{
    return decode_tiling_words(tokens, splits, grid) * sizeof(std::uint32_t);
}
// The word at which the dispatch records begin, which the device computes the same way.
constexpr std::uint32_t decode_dispatch_word_base(const std::uint32_t tokens,
                                                  const std::uint32_t splits)
{
    return tokens * splits * kDecodeTilingWords;
}

// What build_decode_tiling refused to describe. Every one of these is a geometry the device's
// closed-form cursor cannot express, so it is rejected on the host rather than approximated on
// the device.
enum class DecodeTilingStatus : std::uint32_t {
    kOk = 0,
    kBadGeometry,     // zero/overflowing extent, or head_dim/group_heads inconsistent
    kPageSizeNotPow2, // block_size is not a power-of-two multiple of 64
    kUnmappedPage,    // an interior logical page of a live split carries a negative entry
    kProbeOutOfRange, // the isolated-canary descriptor names a page/tile outside the table
};

inline const char *decode_tiling_status_text(const DecodeTilingStatus status)
{
    switch (status) {
        case DecodeTilingStatus::kOk: return "ok";
        case DecodeTilingStatus::kBadGeometry: return "unsupported decode tiling geometry";
        case DecodeTilingStatus::kPageSizeNotPow2:
            return "block_size must be a power-of-two multiple of 64 for the closed-form cursor";
        case DecodeTilingStatus::kUnmappedPage:
            return "a live split holds an unmapped logical page; the closed-form tile cursor "
                   "cannot skip pages";
        case DecodeTilingStatus::kProbeOutOfRange:
            return "isolated-canary page/tile outside the block table";
    }
    return "unknown decode tiling status";
}

// The geometry build_decode_tiling needs. It is the subset of the launch descriptor that
// decides tiling, named separately so this header stays free of the launch ABI (and so the
// host gate can drive the builder without the device contract header).
struct DecodeTilingRequest {
    std::uint32_t tokens = 1;
    std::uint32_t num_heads = 16;
    std::uint32_t num_kv_heads = 1;
    std::uint32_t head_dim = 256;
    std::uint32_t block_size = 128;
    std::uint32_t max_blocks = 8;
    std::uint32_t splits = 1;
    std::uint32_t heads_per_task = 16;
    std::uint32_t tasks_per_block = 1;
    std::uint32_t reduce_tasks_per_block = 16;
    std::uint32_t grid = 1;
    std::uint32_t fused_context_limit = 4096;
    // The isolated single-tile canary, as a descriptor rather than a device branch.
    // UINT32_MAX leaves every worker on its production span.
    std::uint32_t probe_page = UINT32_MAX;
    std::uint32_t probe_tile = 0;
};

namespace detail {
constexpr std::uint32_t ceil_div(const std::uint32_t a, const std::uint32_t b)
{
    return b ? (a + b - 1) / b : 0;
}
// The one place log2 is taken. On the host, where it costs nothing.
constexpr std::uint32_t exact_log2(std::uint32_t value)
{
    std::uint32_t shift = 0;
    while (value > 1) { value >>= 1; ++shift; }
    return shift;
}
constexpr bool is_pow2(const std::uint32_t value)
{
    return value && (value & (value - 1)) == 0;
}
}  // namespace detail

// Fill `out` with decode_tiling_bytes(tokens, splits, grid) bytes of descriptors.
//
//   context_lengths  host-visible int32[tokens]. Required: the tile count of a worker is a
//                    function of its token's context length, and that is the one input the
//                    device used to fetch from GM and divide by.
//   block_table      host-visible int32[tokens][max_blocks], or nullptr. When given, every
//                    live worker's logical page span is checked for unmapped (negative)
//                    entries, which the closed-form cursor cannot skip. When null the table is
//                    assumed dense over each live span; the device still refuses to stage a
//                    span it finds a negative entry in, so a wrong assumption here is a
//                    skipped worker rather than an out-of-bounds read.
inline DecodeTilingStatus build_decode_tiling(const DecodeTilingRequest &r,
                                              const std::int32_t *context_lengths,
                                              const std::int32_t *block_table, void *out)
{
    if (!out || !context_lengths) return DecodeTilingStatus::kBadGeometry;
    if (!r.tokens || !r.splits || !r.grid || !r.num_heads || !r.num_kv_heads ||
        r.num_heads % r.num_kv_heads || !r.head_dim || !r.max_blocks || !r.heads_per_task ||
        !r.tasks_per_block || !r.reduce_tasks_per_block)
        return DecodeTilingStatus::kBadGeometry;
    if (!detail::is_pow2(r.block_size) || r.block_size % kDecodeTileRows)
        return DecodeTilingStatus::kPageSizeNotPow2;

    const std::uint32_t group_heads = r.num_heads / r.num_kv_heads;
    std::uint32_t heads_per_task = r.heads_per_task > group_heads ? group_heads : r.heads_per_task;
    if (!heads_per_task) heads_per_task = 1;
    const std::uint32_t chunks_per_group = detail::ceil_div(group_heads, heads_per_task);
    const std::uint32_t shift = detail::exact_log2(r.block_size) - kDecodeTileRowsShift;
    const bool isolated = r.probe_page != UINT32_MAX;
    if (isolated && (r.probe_page >= r.max_blocks || r.probe_tile >= (1u << shift)))
        return DecodeTilingStatus::kProbeOutOfRange;

    auto *words = static_cast<std::uint32_t *>(out);
    const std::size_t total = decode_tiling_words(r.tokens, r.splits, r.grid);
    for (std::size_t i = 0; i < total; ++i) words[i] = 0;

    for (std::uint32_t token = 0; token < r.tokens; ++token) {
        const std::int32_t raw = context_lengths[token];
        const std::uint32_t context = raw > 0 ? static_cast<std::uint32_t>(raw) : 0;
        const bool fused = r.splits <= 1 || raw <= static_cast<std::int32_t>(r.fused_context_limit);
        const std::uint32_t seq_blocks = detail::ceil_div(context, r.block_size);
        const std::uint32_t blocks_per_split =
            fused ? seq_blocks : detail::ceil_div(seq_blocks, r.splits);
        for (std::uint32_t split = 0; split < r.splits; ++split) {
            auto *d = reinterpret_cast<TurboQuantDecodeTilingData *>(words) + token * r.splits + split;
            d->head_dim = r.head_dim;
            d->group_heads = group_heads;
            d->split_k = r.splits;
            d->tiles_per_page_shift = shift;
            d->chunks_per_group = chunks_per_group;
            if (isolated) {
                // The isolated canary is one 64-row tile of one logical page, on split zero of
                // token zero, with the full cache and table allocations and a nonzero logical
                // page index left exactly as the fixture built them.
                if (token || split) continue;
                d->num_tiles = 1;
                d->regular_tiles = 1;
                d->block_start_idx = r.probe_page;
                d->blocks_in_split = 1;
                d->tile_origin = r.probe_tile;
                d->flags = kDecodeSplitLive;
                continue;
            }
            if (fused && split) continue;  // the duplicate workers of a fused token: no work
            const std::uint32_t block_start = split * blocks_per_split;
            if (block_start >= seq_blocks) continue;
            std::uint32_t block_end = block_start + blocks_per_split;
            if (block_end > seq_blocks) block_end = seq_blocks;
            if (block_end > r.max_blocks) return DecodeTilingStatus::kBadGeometry;
            if (block_table) {
                const std::int32_t *row = block_table + static_cast<std::size_t>(token) * r.max_blocks;
                for (std::uint32_t b = block_start; b < block_end; ++b)
                    if (row[b] < 0) return DecodeTilingStatus::kUnmappedPage;
            }
            const std::uint32_t rows_start = block_start * r.block_size;
            std::uint32_t rows_end = block_end * r.block_size;
            if (rows_end > context) rows_end = context;
            if (rows_end <= rows_start) continue;
            const std::uint32_t live = rows_end - rows_start;
            d->num_tiles = detail::ceil_div(live, kDecodeTileRows);
            d->tail_rows = live % kDecodeTileRows;
            d->regular_tiles = d->num_tiles - (d->tail_rows ? 1u : 0u);
            d->block_start_idx = block_start;
            d->blocks_in_split = block_end - block_start;
            d->flags = kDecodeSplitLive | (d->tail_rows ? kDecodeSplitCarriesTail : 0u);
        }
    }

    // The per-block dispatch records. The odometer below is the host-side twin of the device's
    // AdvanceTask: fastest-varying first, chunk, split, kv head, token, which is the order
    // Process()'s six divisions used to impose.
    const std::uint32_t tasks = r.tokens * r.num_kv_heads * r.splits * chunks_per_group;
    const std::uint32_t reduce_total = r.tokens * r.num_heads;
    auto *dispatch = reinterpret_cast<TurboQuantDecodeDispatch *>(
        words + decode_dispatch_word_base(r.tokens, r.splits));
    for (std::uint32_t block = 0; block < r.grid; ++block) {
        const std::uint32_t start = block * r.tasks_per_block;
        auto &record = dispatch[block];
        std::uint32_t id = start < tasks ? start : tasks;
        record.chunk = chunks_per_group ? id % chunks_per_group : 0;
        id = chunks_per_group ? id / chunks_per_group : 0;
        record.split = id % r.splits;
        id /= r.splits;
        record.kv_head = id % r.num_kv_heads;
        record.token = id / r.num_kv_heads;
        // Reduce()'s two halves. `end` is the share's clamp and the halving is the Fixpipe's
        // ceil/floor, both exactly as the device computes them.
        const std::uint32_t r_start = block * r.reduce_tasks_per_block;
        std::uint32_t r_end = r_start + r.reduce_tasks_per_block;
        if (r_end > reduce_total) r_end = reduce_total;
        const std::uint32_t lo = r_start < r_end ? r_start : r_end;
        const std::uint32_t half = r_end > r_start ? detail::ceil_div(r_end - r_start, 2) : 0;
        const std::uint32_t hi = lo + half;
        record.reduce_token = lo / r.num_heads;
        record.reduce_head = lo % r.num_heads;
        record.reduce_token_hi = hi / r.num_heads;
        record.reduce_head_hi = hi % r.num_heads;
    }
    return DecodeTilingStatus::kOk;
}

// The launch descriptor's tiling subset, taken from whatever launch structure the caller has.
// Templated rather than overloaded on `Launch` so this header stays free of the launch ABI --
// it is included by the device kernel, the host gate and the Torch adapter, and only one of
// those three has a reason to know what a Launch is.
template <typename LaunchT>
inline DecodeTilingRequest decode_tiling_request(const LaunchT &c)
{
    DecodeTilingRequest r;
    r.tokens = c.tokens;
    r.num_heads = c.heads;
    r.num_kv_heads = c.kv_heads;
    r.head_dim = c.head_size;
    r.block_size = c.block_size;
    r.max_blocks = c.max_blocks;
    r.splits = c.splits;
    r.heads_per_task = c.heads_per_task;
    r.tasks_per_block = c.tasks_per_block;
    r.reduce_tasks_per_block = c.reduce_tasks_per_block;
    r.grid = c.grid;
    r.fused_context_limit = c.fused_context_limit;
    r.probe_page = c.probe_page;
    r.probe_tile = c.probe_tile;
    return r;
}

// decode_tiling_bytes for a launch, so a caller sizes its tiling buffer from the same
// descriptor it will hand the builder.
template <typename LaunchT>
inline std::size_t decode_tiling_bytes_for(const LaunchT &c)
{
    return decode_tiling_bytes(c.tokens, c.splits, c.grid);
}

}  // namespace ascend_micro::fused_decode
