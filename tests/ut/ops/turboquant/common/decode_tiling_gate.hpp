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

// THE HOST TILING GATE for the fused decode.
//
// Everything the device would otherwise derive on the scalar unit is built on the host, so this
// is where it is checked. Three things, and the first two are equivalences rather than spot
// values:
//
//   1. the dispatch odometer reproduces, for every task id of every grid block, exactly the
//      (token, kv head, split, chunk) the six integer div/mods in the device cursor would have
//      computed -- which is the one way to know the task STREAM is unchanged;
//   2. the same for the reduction cursors and the two vector subcores' halves;
//   3. the descriptors' tile counts, live tails and page spans, against the closed form a page
//      walk would produce -- per token, over RAGGED length vectors, including zero-length rows.
//      That last part matters more than it looks. A decode step's tokens do not share a context
//      length, and three of build_decode_tiling's decisions are per token rather than per
//      request: whether the token is fused or split (`raw <= fused_context_limit`), whether a
//      given split is live at all (`block_start >= seq_blocks`), and where its live rows stop
//      (`rows_end > context`). A gate that gives every token the same length cannot see any of
//      them disagree. The zero-length row is the one the builder's padded-row contract depends
//      on: it has to produce the all-zero descriptor, because `num_tiles == 0` is the single
//      predicate both kernel roles branch on to skip a worker.
//
// And the refusals: the geometries the closed-form cursor cannot express are rejected on the
// host rather than approximated on the device.
//
// Device-free, CANN-free, runs in milliseconds, and it is NOT kernel evidence: it checks that
// the HOST computes what the device reads, not that the device reads it correctly. A bounded
// CAModel ladder is what checks that.

#pragma once
#include "csrc/attention/turboquant/op_host/turboquant_tiling.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tiling_gate {
namespace fdt = ascend_micro::fused_decode;

inline void require(bool ok)
{
    if (!ok) throw std::runtime_error("fused decode host tiling gate");
}


struct Geometry {
    unsigned tokens, heads, kv_heads, splits, heads_per_task, block_size, max_blocks, context;
    unsigned fused_limit;
};

inline void check_dispatch(const Geometry& g, const std::vector<std::int32_t>& lengths)
{
    const unsigned group = g.heads / g.kv_heads;
    const unsigned per_task = g.heads_per_task > group ? group : g.heads_per_task;
    const unsigned chunks = (group + per_task - 1) / per_task;
    const unsigned tasks = g.tokens * g.kv_heads * g.splits * chunks;
    const unsigned reduce_total = g.tokens * g.heads;
    for (unsigned tasks_per_block = 1; tasks_per_block <= 8; ++tasks_per_block) {
        const unsigned grid = (tasks + tasks_per_block - 1) / tasks_per_block;
        if (!grid || grid > 32) continue;
        const unsigned reduce_per_block = (reduce_total + grid - 1) / grid;
        fdt::DecodeTilingRequest r;
        r.tokens = g.tokens; r.num_heads = g.heads; r.num_kv_heads = g.kv_heads;
        r.block_size = g.block_size; r.max_blocks = g.max_blocks; r.splits = g.splits;
        r.heads_per_task = g.heads_per_task; r.tasks_per_block = tasks_per_block;
        r.reduce_tasks_per_block = reduce_per_block; r.grid = grid;
        r.fused_context_limit = g.fused_limit;
        std::vector<std::int32_t> table(std::size_t(g.tokens) * g.max_blocks);
        for (unsigned t = 0; t < g.tokens; ++t)
            for (unsigned b = 0; b < g.max_blocks; ++b)
                table[std::size_t(t) * g.max_blocks + b] =
                    static_cast<std::int32_t>(2 * (t * g.max_blocks + b) + 1);
        std::vector<std::uint32_t> image(
            fdt::decode_tiling_words(r.tokens, r.splits, r.grid), 0xCDu);
        require(fdt::build_decode_tiling(r, lengths.data(), table.data(), image.data()) ==
                fdt::DecodeTilingStatus::kOk);

        const auto* dispatch = reinterpret_cast<const fdt::TurboQuantDecodeDispatch*>(
            image.data() + fdt::decode_dispatch_word_base(r.tokens, r.splits));
        for (unsigned block = 0; block < grid; ++block) {
            // 1. The odometer, walked from the record, against the divisions it stands for.
            unsigned start = block * tasks_per_block;
            if (start > tasks) start = tasks;
            fdt::TurboQuantDecodeDispatch cursor = dispatch[block];
            unsigned end = block * tasks_per_block + tasks_per_block;
            if (end > tasks) end = tasks;
            for (unsigned id = start; id < end; ++id) {
                const unsigned chunk = id % chunks;
                const unsigned split_task = id / chunks;
                const unsigned split = split_task % g.splits;
                const unsigned token_kv = split_task / g.splits;
                require(cursor.chunk == chunk && cursor.split == split &&
                        cursor.kv_head == token_kv % g.kv_heads &&
                        cursor.token == token_kv / g.kv_heads);
                if (++cursor.chunk == chunks) {
                    cursor.chunk = 0;
                    if (++cursor.split == g.splits) {
                        cursor.split = 0;
                        if (++cursor.kv_head == g.kv_heads) { cursor.kv_head = 0; ++cursor.token; }
                    }
                }
            }
            // 2. Both reduction halves, against `task / numHeads` and `task % numHeads`.
            unsigned r_start = block * reduce_per_block;
            unsigned r_end = r_start + reduce_per_block;
            if (r_end > reduce_total) r_end = reduce_total;
            const unsigned lo = r_start < r_end ? r_start : r_end;
            const unsigned half = r_end > r_start ? (r_end - r_start + 1) / 2 : 0;
            require(dispatch[block].reduce_token == lo / g.heads &&
                    dispatch[block].reduce_head == lo % g.heads &&
                    dispatch[block].reduce_token_hi == (lo + half) / g.heads &&
                    dispatch[block].reduce_head_hi == (lo + half) % g.heads);
        }
    }
}

// The closed form, against an explicit page walk.
inline unsigned walk_tiles(unsigned block_start, unsigned block_end, unsigned block_size, unsigned context)
{
    unsigned tiles = 0;
    for (unsigned b = block_start; b < block_end; ++b) {
        unsigned rows = block_size;
        const unsigned consumed = b * block_size;
        if (consumed >= context) continue;
        if (consumed + rows > context) rows = context - consumed;
        tiles += (rows + 63) / 64;
    }
    return tiles;
}

inline void check_descriptors(const Geometry& g, const std::vector<std::int32_t>& lengths)
{
    fdt::DecodeTilingRequest r;
    r.tokens = g.tokens; r.num_heads = g.heads; r.num_kv_heads = g.kv_heads;
    r.block_size = g.block_size; r.max_blocks = g.max_blocks; r.splits = g.splits;
    r.heads_per_task = g.heads_per_task; r.tasks_per_block = 1;
    r.reduce_tasks_per_block = g.tokens * g.heads; r.grid = 1;
    r.fused_context_limit = g.fused_limit;
    require(lengths.size() == g.tokens);
    std::vector<std::uint32_t> image(fdt::decode_tiling_words(r.tokens, r.splits, r.grid), 0);
    require(fdt::build_decode_tiling(r, lengths.data(), nullptr, image.data()) ==
            fdt::DecodeTilingStatus::kOk);
    const auto* d = reinterpret_cast<const fdt::TurboQuantDecodeTilingData*>(image.data());
    unsigned total = 0, live_rows = 0;
    for (unsigned token = 0; token < g.tokens; ++token) {
        // Per token, because every one of these three is a per-token decision.
        const std::int32_t raw = lengths[token];
        const unsigned context = raw > 0 ? static_cast<unsigned>(raw) : 0u;
        const unsigned seq_blocks = (context + g.block_size - 1) / g.block_size;
        const bool fused = g.splits <= 1 || raw <= static_cast<std::int32_t>(g.fused_limit);
        const unsigned per_split = fused ? seq_blocks : (seq_blocks + g.splits - 1) / g.splits;
        live_rows += (context + 63) / 64;
        for (unsigned split = 0; split < g.splits; ++split) {
            const auto& e = d[token * g.splits + split];
            unsigned expect = 0, tail = 0, span = 0, start = split * per_split;
            if (!(fused && split) && start < seq_blocks) {
                unsigned block_end = start + per_split;
                if (block_end > seq_blocks) block_end = seq_blocks;
                expect = walk_tiles(start, block_end, g.block_size, context);
                unsigned rows_end = block_end * g.block_size;
                if (rows_end > context) rows_end = context;
                tail = rows_end > start * g.block_size ? (rows_end - start * g.block_size) % 64 : 0;
                span = expect ? block_end - start : 0;
            }
            require(e.num_tiles == expect);
            require(e.tail_rows == tail);
            require(e.regular_tiles == expect - (tail ? 1u : 0u));
            require(e.blocks_in_split == span);
            require(!expect || e.block_start_idx == start);
            require(e.tile_origin == 0);
            require(e.head_dim == 256 && e.split_k == g.splits);
            require(e.group_heads == g.heads / g.kv_heads);
            require((1u << e.tiles_per_page_shift) * 64u == g.block_size);
            require(((e.flags & fdt::kDecodeSplitLive) != 0) == (expect != 0));
            require(((e.flags & fdt::kDecodeSplitCarriesTail) != 0) == (tail != 0));
            total += e.num_tiles;
        }
    }
    // Every live row of every token is covered exactly once, whatever the split count -- and a
    // zero-length token contributes nothing, which is what makes a padded row safe to launch.
    require(total == live_rows);
}

inline void run()
{
    // The five bounded rungs' own geometries, plus ragged contexts, 4 KV heads, chunked GQA
    // groups, non-power-of-two split counts and the fused/split crossover.
    const Geometry cases[] = {
        {1, 16, 1, 1, 16, 128, 8, 64, 4096},      // rung 1: one tile
        {1, 16, 1, 1, 16, 128, 8, 79, 4096},      // rung 2: two tiles, tail 15
        {1, 16, 1, 1, 16, 128, 8, 257, 4096},     // rung 3: five tiles, tail 1
        {1, 16, 1, 1, 16, 128, 8, 192, 4096},     // rung 4: three tiles = kSlots
        {1, 16, 1, 1, 16, 128, 8, 384, 4096},     // rung 5: six tiles = 2 * kSlots
        {1, 16, 1, 4, 16, 128, 8, 1024, 0},       // four real Split-K workers
        {1, 16, 1, 3, 16, 128, 8, 1000, 0},       // a non-power-of-two split count, ragged
        {1, 16, 1, 5, 16, 128, 8, 641, 0},        // five splits, one-row tail, empty workers
        {4, 16, 1, 1, 16, 128, 8, 300, 4096},     // a batch, ragged
        {2, 64, 4, 2, 16, 128, 16, 1000, 0},      // 4 KV heads, 2 splits
        {2, 64, 4, 2, 6, 128, 16, 1000, 0},       // chunked GQA group: 16 / 6 -> 3 chunks
        {3, 16, 1, 4, 16, 64, 32, 700, 128},      // 64-row pages, fused limit in the middle
        {1, 16, 1, 1, 16, 128, 8, 1, 4096},       // one live row in the whole sequence
    };
    for (const auto& g : cases) {
        const std::vector<std::int32_t> uniform(g.tokens, static_cast<std::int32_t>(g.context));
        check_dispatch(g, uniform);
        check_descriptors(g, uniform);
    }

    // THE RAGGED GRID. `context` is unused for these; the lengths are explicit. Each one is a
    // length vector the uniform grid above cannot express.
    struct Ragged {
        Geometry g;
        std::vector<std::int32_t> lengths;
    };
    const std::vector<Ragged> ragged = {
        // A 64-exact row, a ragged row, a DEAD row and a one-row sequence in one step. The dead
        // row is the FIA padding request the builder zeroes; under the convention it replaces
        // it carried a 1 and came out live over logical page 0.
        {{4, 16, 1, 1, 16, 128, 8, 0, 4096}, {64, 300, 0, 1}},
        // The fuse decision straddled INSIDE one step: token 0 is under the limit and fuses,
        // token 1 is over it and splits four ways, token 2 is dead. No uniform case can show
        // these three descriptors disagreeing about `split_k`'s meaning.
        {{3, 16, 1, 4, 16, 128, 16, 0, 128}, {64, 2048, 0}},
        // A fully padded step: every row dead, every descriptor all-zero, nothing launched.
        {{2, 16, 1, 1, 16, 128, 8, 0, 4096}, {0, 0}},
        // Four KV heads and two splits, one live row and one dead.
        {{2, 64, 4, 2, 16, 128, 16, 0, 0}, {1000, 0}},
        // A dead row FIRST, so the descriptor array's live entries do not start at index zero.
        {{3, 16, 1, 2, 16, 128, 16, 0, 0}, {0, 1024, 65}},
    };
    for (const auto& r : ragged) {
        check_dispatch(r.g, r.lengths);
        check_descriptors(r.g, r.lengths);
    }

    // THE REFUSALS. Each of these is a geometry the device's closed-form cursor cannot express,
    // and each is refused here rather than approximated there.
    std::int32_t length = 1024;
    std::vector<std::uint32_t> image(fdt::decode_tiling_words(1, 1, 1) + 64, 0);
    fdt::DecodeTilingRequest r;
    r.tokens = 1; r.num_heads = 16; r.num_kv_heads = 1; r.splits = 1; r.grid = 1;
    r.max_blocks = 16; r.block_size = 128;
    require(fdt::build_decode_tiling(r, &length, nullptr, image.data()) ==
            fdt::DecodeTilingStatus::kOk);
    // A 192-row page would need an integer division per tile on the SPU.
    r.block_size = 192;
    require(fdt::build_decode_tiling(r, &length, nullptr, image.data()) ==
            fdt::DecodeTilingStatus::kPageSizeNotPow2);
    r.block_size = 128;
    // An unmapped logical page inside a live span cannot be skipped by an affine cursor.
    std::vector<std::int32_t> table(16);
    for (int i = 0; i < 16; ++i) table[std::size_t(i)] = 2 * i + 1;
    require(fdt::build_decode_tiling(r, &length, table.data(), image.data()) ==
            fdt::DecodeTilingStatus::kOk);
    table[3] = -1;  // logical page 3 of 8 live pages
    require(fdt::build_decode_tiling(r, &length, table.data(), image.data()) ==
            fdt::DecodeTilingStatus::kUnmappedPage);
    // Padding PAST the live span is not a hole and must stay admissible.
    table[3] = 7;
    table[9] = -1;
    require(fdt::build_decode_tiling(r, &length, table.data(), image.data()) ==
            fdt::DecodeTilingStatus::kOk);
    // The isolated canary has to name a page and a tile the table and the page size hold.
    r.probe_page = 16;
    require(fdt::build_decode_tiling(r, &length, nullptr, image.data()) ==
            fdt::DecodeTilingStatus::kProbeOutOfRange);
    r.probe_page = 7; r.probe_tile = 2;  // a 128-row page holds tiles 0 and 1
    require(fdt::build_decode_tiling(r, &length, nullptr, image.data()) ==
            fdt::DecodeTilingStatus::kProbeOutOfRange);
    r.probe_tile = 1;
    require(fdt::build_decode_tiling(r, &length, nullptr, image.data()) ==
            fdt::DecodeTilingStatus::kOk);
    {
        const auto* d = reinterpret_cast<const fdt::TurboQuantDecodeTilingData*>(image.data());
        require(d->num_tiles == 1 && d->block_start_idx == 7 && d->tile_origin == 1 &&
                d->blocks_in_split == 1 && d->tail_rows == 0);
    }
    // Nothing may be written past the descriptors and the dispatch records.
    for (std::size_t i = fdt::decode_tiling_words(1, 1, 1); i < image.size(); ++i)
        require(image[i] == 0);
}
}  // namespace tiling_gate
