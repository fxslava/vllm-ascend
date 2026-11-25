#ifndef TURBOQUANT_HOST_TILING_H
#define TURBOQUANT_HOST_TILING_H

#include <cstddef>
#include <cstdint>

namespace ascend_micro::fused_decode {

constexpr std::uint32_t kDecodeTileRows = 64;
constexpr std::uint32_t kDecodeTileRowsShift = 6;
static_assert(kDecodeTileRows == (1u << kDecodeTileRowsShift));

struct alignas(64) TurboQuantDecodeTilingData {
    std::uint32_t num_tiles;
    std::uint32_t regular_tiles;
    std::uint32_t tail_rows;
    std::uint32_t block_start_idx;
    std::uint32_t blocks_in_split;
    std::uint32_t head_dim;
    std::uint32_t group_heads;
    std::uint32_t split_k;

    std::uint32_t tile_origin;

    std::uint32_t tiles_per_page_shift;
    std::uint32_t flags;

    std::uint32_t chunks_per_group;
    std::uint32_t reserved[4];
};
static_assert(sizeof(TurboQuantDecodeTilingData) == 64, "the descriptor is one 64-byte line");
static_assert(alignof(TurboQuantDecodeTilingData) == 64, "64-byte aligned, as the contract says");

constexpr std::uint32_t kDecodeSplitLive = 1u << 0;
constexpr std::uint32_t kDecodeSplitCarriesTail = 1u << 1;

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

struct alignas(32) TurboQuantDecodeDispatch {
    std::uint32_t token;
    std::uint32_t kv_head;
    std::uint32_t split;
    std::uint32_t chunk;
    std::uint32_t reduce_token;
    std::uint32_t reduce_head;
    std::uint32_t reduce_token_hi;
    std::uint32_t reduce_head_hi;
};
static_assert(sizeof(TurboQuantDecodeDispatch) == 32);
constexpr std::uint32_t kDecodeDispatchWords =
    sizeof(TurboQuantDecodeDispatch) / sizeof(std::uint32_t);
static_assert(kDecodeDispatchWords == 8);

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

constexpr std::uint32_t decode_dispatch_word_base(const std::uint32_t tokens,
                                                  const std::uint32_t splits)
{
    return tokens * splits * kDecodeTilingWords;
}

enum class DecodeTilingStatus : std::uint32_t {
    kOk = 0,
    kBadGeometry,
    kPageSizeNotPow2,
    kUnmappedPage,

    kContextOutOfRange,
    kProbeOutOfRange,
};

const char *decode_tiling_status_text(DecodeTilingStatus status);

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

    std::uint32_t probe_page = UINT32_MAX;
    std::uint32_t probe_tile = 0;
};

DecodeTilingStatus build_decode_tiling(const DecodeTilingRequest &r,
                                       const std::int32_t *context_lengths,
                                       const std::int32_t *block_table, void *out);

#if !defined(__CCE_AICORE__)
struct DecodeLaunchSplit {
    std::uint32_t grid;
    std::uint32_t tasks_per_block;
    std::uint32_t reduce_tasks_per_block;
    std::uint32_t prologue_vectors_per_block;
    std::uint32_t chunks_per_group;
    std::uint32_t tasks;
};

inline DecodeLaunchSplit decode_launch_split(const DecodeTilingRequest &r, const std::uint32_t max_grid)
{
    const auto ceil_div = [](const std::uint32_t a, const std::uint32_t b) -> std::uint32_t {
        return b ? (a + b - 1) / b : 0;
    };
    const std::uint32_t group_heads = r.num_kv_heads ? r.num_heads / r.num_kv_heads : r.num_heads;

    std::uint32_t heads_per_task = r.heads_per_task > group_heads ? group_heads : r.heads_per_task;
    if (!heads_per_task) heads_per_task = 1;

    DecodeLaunchSplit s{};
    s.chunks_per_group = ceil_div(group_heads, heads_per_task);
    s.tasks = r.tokens * r.num_kv_heads * r.splits * s.chunks_per_group;
    const std::uint32_t reduce_total = r.tokens * r.num_heads;

    const std::uint32_t live_tasks = s.tasks ? s.tasks : 1u;
    s.grid = live_tasks < max_grid ? live_tasks : max_grid;
    s.tasks_per_block = ceil_div(s.tasks, s.grid);
    s.reduce_tasks_per_block = ceil_div(reduce_total, s.grid);
    s.prologue_vectors_per_block = s.reduce_tasks_per_block;
    return s;
}

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

template <typename LaunchT>
inline std::size_t decode_tiling_bytes_for(const LaunchT &c)
{
    return decode_tiling_bytes(c.tokens, c.splits, c.grid);
}

#endif

}

#endif
