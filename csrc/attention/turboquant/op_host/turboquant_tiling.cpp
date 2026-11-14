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

#include "turboquant_tiling.h"

namespace ascend_micro::fused_decode {

namespace {

constexpr std::uint32_t CeilDiv(const std::uint32_t a, const std::uint32_t b)
{
    return b ? (a + b - 1) / b : 0;
}

// The one place a log2 is taken, on the host, where it costs nothing.
constexpr std::uint32_t ExactLog2(std::uint32_t value)
{
    std::uint32_t shift = 0;
    while (value > 1) {
        value >>= 1;
        ++shift;
    }
    return shift;
}

constexpr bool IsPow2(const std::uint32_t value)
{
    return value && (value & (value - 1)) == 0;
}

}  // namespace

const char *decode_tiling_status_text(const DecodeTilingStatus status)
{
    switch (status) {
        case DecodeTilingStatus::kOk:
            return "ok";
        case DecodeTilingStatus::kBadGeometry:
            return "unsupported decode tiling geometry";
        case DecodeTilingStatus::kPageSizeNotPow2:
            return "block_size must be a power-of-two multiple of 64 for the closed-form cursor";
        case DecodeTilingStatus::kUnmappedPage:
            return "a live split holds an unmapped logical page; the closed-form tile cursor "
                   "cannot skip pages";
        case DecodeTilingStatus::kProbeOutOfRange:
            return "isolated-canary page/tile outside the block table, or naming a tile the "
                   "context does not reach";
    }
    return "unknown decode tiling status";
}

DecodeTilingStatus build_decode_tiling(const DecodeTilingRequest &r,
                                       const std::int32_t *context_lengths,
                                       const std::int32_t *block_table, void *out)
{
    if (!out || !context_lengths) return DecodeTilingStatus::kBadGeometry;
    if (!r.tokens || !r.splits || !r.grid || !r.num_heads || !r.num_kv_heads ||
        r.num_heads % r.num_kv_heads || !r.head_dim || !r.max_blocks || !r.heads_per_task ||
        !r.tasks_per_block || !r.reduce_tasks_per_block)
        return DecodeTilingStatus::kBadGeometry;
    if (!IsPow2(r.block_size) || r.block_size % kDecodeTileRows)
        return DecodeTilingStatus::kPageSizeNotPow2;

    const std::uint32_t group_heads = r.num_heads / r.num_kv_heads;
    std::uint32_t heads_per_task = r.heads_per_task > group_heads ? group_heads : r.heads_per_task;
    if (!heads_per_task) heads_per_task = 1;
    const std::uint32_t chunks_per_group = CeilDiv(group_heads, heads_per_task);
    const std::uint32_t shift = ExactLog2(r.block_size) - kDecodeTileRowsShift;
    const bool isolated = r.probe_page != UINT32_MAX;
    if (isolated && (r.probe_page >= r.max_blocks || r.probe_tile >= (1u << shift)))
        return DecodeTilingStatus::kProbeOutOfRange;

    auto *words = static_cast<std::uint32_t *>(out);
    const std::size_t total = decode_tiling_words(r.tokens, r.splits, r.grid);
    for (std::size_t i = 0; i < total; ++i) words[i] = 0;

    for (std::uint32_t token = 0; token < r.tokens; ++token) {
        const std::int32_t raw = context_lengths[token];
        const std::uint32_t context = raw > 0 ? static_cast<std::uint32_t>(raw) : 0;
        const bool fused =
            r.splits <= 1 || raw <= static_cast<std::int32_t>(r.fused_context_limit);
        const std::uint32_t seq_blocks = CeilDiv(context, r.block_size);
        const std::uint32_t blocks_per_split = fused ? seq_blocks : CeilDiv(seq_blocks, r.splits);
        for (std::uint32_t split = 0; split < r.splits; ++split) {
            auto *d =
                reinterpret_cast<TurboQuantDecodeTilingData *>(words) + token * r.splits + split;
            d->head_dim = r.head_dim;
            d->group_heads = group_heads;
            d->split_k = r.splits;
            d->tiles_per_page_shift = shift;
            d->chunks_per_group = chunks_per_group;
            if (isolated) {
                // One 64-row tile of one logical page, on split zero of token zero, with the
                // full cache and table allocations left exactly as the fixture built them.
                if (token || split) continue;
                // THE PROBED TILE'S OWN LIVE ROW COUNT, so an edge tile is probed AS the ragged
                // tile it is in the real grid. Without this a probe always claimed 64 live rows
                // and the one tile worth isolating -- the short last one, where the mask and
                // `lastValid` come into it -- could not be reached through this ABI at all.
                const std::uint32_t tile_first =
                    r.probe_page * r.block_size + r.probe_tile * kDecodeTileRows;
                const std::uint32_t live = tile_first < context ? context - tile_first : 0u;
                // A probe naming a tile the sequence does not reach is a caller error. It used
                // to be served as a full live tile, which is a launch over rows the context
                // never had: refused instead, so a sweep that walks off the end says so.
                if (live == 0) return DecodeTilingStatus::kProbeOutOfRange;
                d->num_tiles = 1;
                d->tail_rows = live < kDecodeTileRows ? live : 0u;
                d->regular_tiles = d->tail_rows ? 0u : 1u;
                d->block_start_idx = r.probe_page;
                d->blocks_in_split = 1;
                d->tile_origin = r.probe_tile;
                d->flags = kDecodeSplitLive | (d->tail_rows ? kDecodeSplitCarriesTail : 0u);
                continue;
            }
            if (fused && split) continue;  // the duplicate workers of a fused token: no work
            const std::uint32_t block_start = split * blocks_per_split;
            if (block_start >= seq_blocks) continue;
            std::uint32_t block_end = block_start + blocks_per_split;
            if (block_end > seq_blocks) block_end = seq_blocks;
            if (block_end > r.max_blocks) return DecodeTilingStatus::kBadGeometry;
            if (block_table) {
                const std::int32_t *row =
                    block_table + static_cast<std::size_t>(token) * r.max_blocks;
                for (std::uint32_t b = block_start; b < block_end; ++b)
                    if (row[b] < 0) return DecodeTilingStatus::kUnmappedPage;
            }
            const std::uint32_t rows_start = block_start * r.block_size;
            std::uint32_t rows_end = block_end * r.block_size;
            if (rows_end > context) rows_end = context;
            if (rows_end <= rows_start) continue;
            const std::uint32_t live = rows_end - rows_start;
            d->num_tiles = CeilDiv(live, kDecodeTileRows);
            d->tail_rows = live % kDecodeTileRows;
            d->regular_tiles = d->num_tiles - (d->tail_rows ? 1u : 0u);
            d->block_start_idx = block_start;
            d->blocks_in_split = block_end - block_start;
            d->flags = kDecodeSplitLive | (d->tail_rows ? kDecodeSplitCarriesTail : 0u);
        }
    }

    // The per-block dispatch records. The odometer is the host-side twin of the device's
    // AdvanceTask: fastest-varying first -- chunk, split, kv head, token.
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
        // The two reduction halves: `end` is the share's clamp and the halving is the Fixpipe's
        // ceil/floor, both exactly as the device computes them.
        const std::uint32_t r_start = block * r.reduce_tasks_per_block;
        std::uint32_t r_end = r_start + r.reduce_tasks_per_block;
        if (r_end > reduce_total) r_end = reduce_total;
        const std::uint32_t lo = r_start < r_end ? r_start : r_end;
        const std::uint32_t half = r_end > r_start ? CeilDiv(r_end - r_start, 2) : 0;
        const std::uint32_t hi = lo + half;
        record.reduce_token = lo / r.num_heads;
        record.reduce_head = lo % r.num_heads;
        record.reduce_token_hi = hi / r.num_heads;
        record.reduce_head_hi = hi % r.num_heads;
    }
    return DecodeTilingStatus::kOk;
}

}  // namespace ascend_micro::fused_decode
