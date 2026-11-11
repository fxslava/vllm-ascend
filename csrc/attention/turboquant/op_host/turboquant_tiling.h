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

// Host tiling for the TurboQuant operators.
//
// Device-free and toolkit-free. Every quantity the modulo-scheduled decode would otherwise
// derive on the DaVinci scalar unit -- the tile count of a Split-K worker, its logical page
// span, the live row count of its final tile, and the (token, kv head, split, chunk) a grid
// block's task range starts at -- is computed here and handed to the device as plain words, so
// the device issues no integer divide and takes no data-dependent branch inside the three-zone
// schedule.
//
// WHY ON dav-3510 SPECIFICALLY. The SPU issues in order for the whole core. A scalar integer
// divide, or a branch whose condition depends on a scalar load from GM, stalls the issue queue;
// everything behind it in program order -- including the MTE2 descriptor writes that prefetch
// the next ring slot and the Mmad issue that keeps the Cube systolic array fed -- waits. The
// cursor arithmetic this replaces sat between a tile's softmax and the next tile's stage, which
// is exactly the dependence chain the three-zone schedule exists to keep full.
//
// The device reads the descriptor words by OFFSET, and nothing in the toolchain diagnoses two
// sides disagreeing about which word is which, so this is the single declaration the kernel,
// the launcher, the host gate and any adapter all include.

#ifndef TURBOQUANT_HOST_TILING_H
#define TURBOQUANT_HOST_TILING_H

#include <cstddef>
#include <cstdint>

namespace ascend_micro::fused_decode {

// The pipeline's tile is 64 KV rows. It is kCubeTileRows in turboquant_layout.h; restated here
// so this layer stays independent of the device tree, and pinned to it by a static_assert in
// the kernel translation unit that includes both.
constexpr std::uint32_t kDecodeTileRows = 64;
constexpr std::uint32_t kDecodeTileRowsShift = 6;
static_assert(kDecodeTileRows == (1u << kDecodeTileRowsShift));

// One descriptor per (token, split), describing one Split-K worker. The first eight words are
// the tiling contract; the next four are what the closed-form device cursor needs on top of it,
// inside the same 64 bytes. The padding is named and the host zeroes it: this structure is
// uploaded to GM verbatim, so implicit padding would be uninitialised bytes crossing into
// device memory.
//
// A worker with no work -- a split past the end of the sequence, or a duplicate split of a
// token short enough to run fused -- is the all-zero descriptor, and `num_tiles == 0` is the
// single predicate both the Cube and the Vector role branch on to skip the pipeline.
struct alignas(64) TurboQuantDecodeTilingData {
    std::uint32_t num_tiles;        // exact number of 64-token tiles for this split task
    std::uint32_t regular_tiles;    // number of full 64-row tiles (unmasked)
    std::uint32_t tail_rows;        // live rows in the final tile (0 if aligned)
    std::uint32_t block_start_idx;  // logical page offset in blockTable for this split
    std::uint32_t blocks_in_split;  // number of pages in this split
    std::uint32_t head_dim;         // D = 256
    std::uint32_t group_heads;      // GQA ratio
    std::uint32_t split_k;          // split count S
    // The tile index INSIDE block_start_idx at which this worker's first tile begins. Zero for
    // every production worker; it is what lets an isolated single-tile canary name "page 7,
    // second 64-row tile" as a descriptor rather than as a device-side probe branch.
    std::uint32_t tile_origin;
    // log2(block_size / 64). The closed-form cursor turns a tile index into (page, row base)
    // with one shift and one mask, so the page size has to be a power-of-two multiple of 64;
    // build_decode_tiling rejects anything else rather than let the device divide.
    std::uint32_t tiles_per_page_shift;
    std::uint32_t flags;            // kDecodeSplitLive, kDecodeSplitCarriesTail
    // ceil(group_heads / heads_per_task): how many head chunks one (token, kv head, split)
    // splits into.
    std::uint32_t chunks_per_group;
    std::uint32_t reserved[4];
};
static_assert(sizeof(TurboQuantDecodeTilingData) == 64, "the descriptor is one 64-byte line");
static_assert(alignof(TurboQuantDecodeTilingData) == 64, "64-byte aligned, as the contract says");

constexpr std::uint32_t kDecodeSplitLive = 1u << 0;
constexpr std::uint32_t kDecodeSplitCarriesTail = 1u << 1;

// Word offsets of the fields the device reads, for the scalar loads in the task prologue. The
// device reads words, not a struct: a __gm__ struct load would be several scalar loads with no
// guarantee about their order, and these are one cache line.
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
constexpr std::uint32_t kDecodeTilingWords =
    sizeof(TurboQuantDecodeTilingData) / sizeof(std::uint32_t);
static_assert(kDecodeTilingWords == 16);

// One record per grid block: where that block's task range starts, so the device advances an
// odometer instead of decoding every task id with six integer div/mods. The two reduce cursors
// are the two vector subcores' halves of the block's reduction share.
struct alignas(32) TurboQuantDecodeDispatch {
    std::uint32_t token;            // attention task cursor at block entry
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

// The buffer. Descriptors first, so the array the device indexes most often starts at offset
// zero and every element is 64-byte aligned; the per-block dispatch records follow. Both arrays
// are read by scalar loads on both roles and nothing writes them, so they share one read-only
// GM allocation.
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
    kBadGeometry,      // zero/overflowing extent, or head_dim/group_heads inconsistent
    kPageSizeNotPow2,  // block_size is not a power-of-two multiple of 64
    kUnmappedPage,     // an interior logical page of a live split carries a negative entry
    kProbeOutOfRange,  // the isolated-canary descriptor names a page/tile outside the table
};

const char *decode_tiling_status_text(DecodeTilingStatus status);

// The geometry build_decode_tiling needs: the subset of the launch descriptor that decides
// tiling, named separately so this layer stays free of the launch ABI.
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
    // The isolated single-tile canary, as a descriptor rather than a device branch. UINT32_MAX
    // leaves every worker on its production span.
    std::uint32_t probe_page = UINT32_MAX;
    std::uint32_t probe_tile = 0;
};

// The host work split of one decode launch: how many grid blocks, and how many attention,
// reduction and prologue tasks each one owns.
//
// WHY THIS IS HERE AND NOT IN THE BINDING. Two callers need the same answer and neither can be
// the authority. The operator derives it to size its own loops, and the caller derives it to
// size the tiling buffer -- `decode_tiling_words` takes `grid`, so a caller that guessed `grid`
// differently would size the dispatch array differently from the array the device indexes. One
// inline definition both include is what makes that impossible. It is deliberately NOT a
// member of DecodeTilingRequest's own arithmetic: the request carries these three as inputs
// because a probe or a fixture may pin them, and this is the production derivation of them.
//
// Free of the device tree, like the rest of this header: `max_grid` is a parameter rather than
// kDecodeMaxGrid, and the Cube-prologue chunk decision -- which needs
// layout::kQueryBasisMinCubeVectors -- stays with the caller that has those constants.
struct DecodeLaunchSplit {
    std::uint32_t grid;                        // launched blocks
    std::uint32_t tasks_per_block;             // attention tasks per block
    std::uint32_t reduce_tasks_per_block;      // Split-K reduction tasks per block
    std::uint32_t prologue_vectors_per_block;  // query vectors the basis prologue rotates
    std::uint32_t chunks_per_group;            // ceil(group_heads / heads_per_task)
    std::uint32_t tasks;                       // total attention tasks
};

inline DecodeLaunchSplit decode_launch_split(const DecodeTilingRequest &r, const std::uint32_t max_grid)
{
    const auto ceil_div = [](const std::uint32_t a, const std::uint32_t b) -> std::uint32_t {
        return b ? (a + b - 1) / b : 0;
    };
    const std::uint32_t group_heads = r.num_kv_heads ? r.num_heads / r.num_kv_heads : r.num_heads;
    // Clamped exactly as build_decode_tiling clamps it, including the zero guard, so the
    // chunks_per_group the device reads out of the descriptor is the one the loops assume.
    std::uint32_t heads_per_task = r.heads_per_task > group_heads ? group_heads : r.heads_per_task;
    if (!heads_per_task) heads_per_task = 1;

    DecodeLaunchSplit s{};
    s.chunks_per_group = ceil_div(group_heads, heads_per_task);
    s.tasks = r.tokens * r.num_kv_heads * r.splits * s.chunks_per_group;
    const std::uint32_t reduce_total = r.tokens * r.num_heads;
    // A split launch reaches AscendC::SyncAll<false>(), which on dav-3510 only completes if
    // every launched block is resident at the same time, so the grid is capped either way.
    const std::uint32_t live_tasks = s.tasks ? s.tasks : 1u;
    s.grid = live_tasks < max_grid ? live_tasks : max_grid;
    s.tasks_per_block = ceil_div(s.tasks, s.grid);
    s.reduce_tasks_per_block = ceil_div(reduce_total, s.grid);
    s.prologue_vectors_per_block = s.reduce_tasks_per_block;
    return s;
}

// Fill `out` with decode_tiling_bytes(tokens, splits, grid) bytes of descriptors.
//
//   context_lengths  host-visible int32[tokens]. Required: a worker's tile count is a function
//                    of its token's context length.
//   block_table      host-visible int32[tokens][max_blocks], or nullptr. When given, every live
//                    worker's logical page span is checked for unmapped (negative) entries,
//                    which the closed-form cursor cannot skip. When null the table is assumed
//                    dense over each live span; the device still refuses to stage a span it
//                    finds a negative entry in, so a wrong assumption here is a skipped worker
//                    rather than an out-of-bounds read.
DecodeTilingStatus build_decode_tiling(const DecodeTilingRequest &r,
                                       const std::int32_t *context_lengths,
                                       const std::int32_t *block_table, void *out);

// The launch descriptor's tiling subset, taken from whatever launch structure the caller has.
// Templated rather than overloaded on `Launch` so this header stays free of the launch ABI.
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

#endif  // TURBOQUANT_HOST_TILING_H
