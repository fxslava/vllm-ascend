#pragma once
#include <cstdint>
#include <stdexcept>
#include "turboquant_tiling.h"
#include "turboquant_types.h"
#include "turboquant_layout.h"
#include "turboquant_mode.h"

namespace ascend_micro::fused_decode_opt {
constexpr std::uint32_t kHeadSize = 256;
constexpr std::uint32_t kTileRows = vllm_ascend::turboquant::kCubeTileRows;
constexpr std::uint32_t kSlots = vllm_ascend::turboquant::kCubePipelineSlots;
constexpr std::uint32_t kWindows = vllm_ascend::turboquant::kCubeScoreWindows;
constexpr std::uint32_t kTaskJoinFlag = 10;
static_assert(kSlots == 3 && kWindows == 2 && kTileRows == 64);
static_assert(kTileRows == ascend_micro::fused_decode::kDecodeTileRows,
    "the host tiling contract and the device layout disagree about the tile");
static_assert(vllm_ascend::turboquant::kOptFlagProbsReady < kTaskJoinFlag);
static_assert(kTaskJoinFlag < vllm_ascend::turboquant::kOptFlagReservedFirst);

constexpr std::uint32_t kUbSizes[] = {
    2048, 2048, 8192, 8192, 32, 8192, 32, 8192, 32,
    2048, 32, 2048, 32, 2048, 32, 16384, 8192, 8192, 8192,
    2048, 512, 3072, 2048, 512, 1024, 4096, 512,
    (256 * 32 + 3 * 256 + 64 + 8 + 8 * 256) * 4};
constexpr std::uint32_t ub_end()
{
    std::uint32_t bytes = 0;
    for (auto size : kUbSizes) bytes += size;
    return bytes;
}
constexpr bool aligned_allocations()
{
    for (auto size : kUbSizes) if (size % 32) return false;
    return true;
}
static_assert(aligned_allocations());
static_assert((8192 + 32) / 32 % 16 != 0, "packed slots must use different banks");
static_assert((2048 + 32) / 32 % 16 != 0, "scale slots must use different banks");
constexpr std::uint32_t kUbEnd = ub_end() + 32;

constexpr std::uint32_t kTransientPoolBytes = kUbSizes[15] + kUbSizes[16] + kUbSizes[17];
static_assert(kTransientPoolBytes == 32768, "the transient pool, pinned on both sides");

constexpr std::uint32_t kBasisUbBytes = 8 * 256 * (2 + 4 + 2 + 4) + 256 * 4 +
    (7 * 256 + 2 * 256 + 16) * 4 + (2 * 256 + 256 + 64 + 8 + 8 * 256) * 4;
constexpr std::uint32_t kReducerUbBytes = (2 * 256 + 16) * 4 + 5 * 8 * 4 +
    (2 * 64 + 2 * 8) * 4 + 256 * 2;
static_assert(kUbEnd + kBasisUbBytes + kReducerUbBytes + 256 <= 192 * 1024,
    "reserve a final full-register read pad even with raw basis and reduction");

static_assert(kUbEnd == 142336 && kBasisUbBytes == 46432 && kReducerUbBytes == 3360,
    "turboquant_gemm_reduce.hpp restates these; update kOperatorUbEndBytes, "
    "kOperatorBasisUbBytes and kVectorReducerUbBytes there to match");
static_assert(16 * 256 + 16 * 64 + 2 * kSlots * 64 * 256 <= 512 * 1024);

constexpr bool valid_launch(std::uint32_t grid, std::uint32_t tokens,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim,
    std::uint32_t block_size, std::uint32_t max_blocks, std::uint32_t splits,
    std::uint32_t heads_per_task, std::uint32_t tasks_per_block,
    std::uint32_t reduce_tasks_per_block)
{
    if (!grid || grid > 32 || !tokens || !kv_heads || kv_heads > 4 || !heads ||
        heads % kv_heads || dim != kHeadSize || !block_size || block_size % kTileRows ||
        !max_blocks || !splits || splits > 8 || !heads_per_task || heads_per_task > 16 ||
        !tasks_per_block || !reduce_tasks_per_block) return false;

    if (block_size & (block_size - 1)) return false;
    const auto group = heads / kv_heads;
    const auto chunk = heads_per_task < group ? heads_per_task : group;
    const std::uint64_t tasks = std::uint64_t{tokens} * kv_heads * splits *
        ((std::uint64_t{group} + chunk - 1) / chunk);
    return tasks <= UINT32_MAX && std::uint64_t{grid} * tasks_per_block >= tasks &&
        std::uint64_t{tokens} * heads <= UINT32_MAX &&
        std::uint64_t{grid} * tasks_per_block <= UINT32_MAX &&
        std::uint64_t{grid} * reduce_tasks_per_block <= UINT32_MAX &&
        std::uint64_t{grid} * reduce_tasks_per_block >= std::uint64_t{tokens} * heads;
}

struct Launch {
    std::uint32_t grid = 1, tokens = 1, heads = 16, kv_heads = 1;
    std::uint32_t head_size = kHeadSize, block_size = 128, max_blocks = 8;
    std::uint32_t splits = 1, heads_per_task = 16, tasks_per_block = 1;
    std::uint32_t reduce_tasks_per_block = 16, fused_context_limit = 4096;
    float scale = 0.0625f, inv_sqrt_len = 0.0625f;

    std::uint32_t prologue_vectors_per_block = 0, prologue_cube_chunk_vectors = 0;
    std::uint32_t output_stage = 0;

    std::uint32_t element_type = 0;

    std::uint32_t decode_unpack = 0;

    std::uint32_t num_blocks = 0;

    std::uint32_t probe_page = UINT32_MAX, probe_tile = 0;
};

inline vllm_ascend::AscendType ascend_type_of([[maybe_unused]] std::uint32_t element_type)
{
    switch (element_type) {
    case 0: return vllm_ascend::AscendType::FP16;
    case 1: return vllm_ascend::AscendType::BF16;
    default: throw std::invalid_argument("fused decode launch with no kernel instance for its element type");
    }
}

constexpr std::uint32_t element_bytes(std::uint32_t) { return 2; }
struct Buffers {
    void *query, *key, *value, *scales, *block_table, *context_lengths;
    void *mode_tables, *workspace, *output;

    void *tiling = nullptr;
};

struct RawQueryBuffers {
    void *raw_query, *pi_signs, *rot_tables, *h16, *gate;
    Buffers decode;
};
}

extern "C" void launch_fused_decode_opt(void* stream,
    const ascend_micro::fused_decode_opt::Launch* launch,
    const ascend_micro::fused_decode_opt::Buffers* buffers);

extern "C" void launch_fused_decode_opt_raw_query(void* stream,
    const ascend_micro::fused_decode_opt::Launch* launch,
    const ascend_micro::fused_decode_opt::RawQueryBuffers* buffers);

namespace vllm_ascend {
void turboquant_mm_fused_decode_opt_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim,
                                         void *queryRot, void *keyCache, void *valueCache, void *scaleCache,
                                         void *blockTables, void *contextLens, void *modeTables, void *workspace,
                                         void *output, void *tiling, uint32_t numTokens, uint32_t numHeads,
                                         uint32_t numKvHeads,
                                         uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                         uint32_t numSplits, uint32_t headsPerTask, uint32_t tasksPerBlock,
                                         uint32_t reduceTasksPerBlock, uint32_t fusedContextLimit, float scale,
                                         float invSqrtLen, uint32_t numBlocks, uint32_t decodeUnpack = 0);
void turboquant_mm_fused_decode_opt_raw_query_impl(
    int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *query, void *piSigns, void *rotTables,
    void *h16, void *gate, void *queryRot, void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
    void *contextLens, void *modeTables, void *workspace, void *output, void *tiling, uint32_t numTokens,
    uint32_t numHeads,
    uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,
    uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock, uint32_t prologueVectorsPerBlock,
    uint32_t prologueCubeChunkVectors, uint32_t outputStage, uint32_t fusedContextLimit, float scale,
    float invSqrtLen, uint32_t numBlocks, uint32_t decodeUnpack = 0);
}
