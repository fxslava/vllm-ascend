#pragma once
#include <cstdint>
#include <stdexcept>
#include "turboquant_fused_decode.hpp"
#include "../op_kernel/common/types.h"
#include "../op_kernel/common/turboquant_layout.h"
#include "../op_kernel/common/turboquant_mode.h"

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

// UB allocation sizes, in production InitBuffers order, for D=256 and <=4 KV
// heads (scaleSlot=8). Includes codec workspace; all payloads are disjoint.
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
constexpr std::uint32_t kUbEnd = ub_end();

// The attention phase's transient pool: kUbSizes[15..17], which are the tile ingest's NZ
// staging operand, the per-task online-softmax accumulator and the query staging. The vector
// service allocates those three from a TBufPool of exactly this length so the Cube GEMM
// reduction can alias its own pool onto the same bytes once SyncAll<false>() has made them
// dead (turboquant_gemm_reduce.hpp carries that budget and restates this number).
//
// Exactly their sum, and that is load-bearing: a longer pool would move kUbEnd and every
// buffer allocated after it, and the Cube's dual-destination Fixpipe windows at UB offsets 0
// and kScoreBytes are addressed by offset on three cores with nothing in the toolchain checking
// that they agree.
constexpr std::uint32_t kTransientPoolBytes = kUbSizes[15] + kUbSizes[16] + kUbSizes[17];
static_assert(kTransientPoolBytes == 32768, "the transient pool, pinned on both sides");
// NONE OF THESE MOVE WITH THE ELEMENT TYPE, and that is the whole reason the bf16 instance of the
// operator is a type substitution rather than a second UB map. Every `2` below is a 16-bit extent:
// in kBasisUbBytes, the basis service's raw-query/gate staging buffer and its Cube chunk operand
// (`8 * 256 * (2 + 4 + 2 + 4)` is inBuf_, floatBuf_, halfBuf_, pairBuf_ over kScratchRows * headSize);
// in kReducerUbBytes, the reducer's `256 * 2` output row. sizeof(half) == sizeof(bfloat16_t) == 2, so
// both instances allocate the identical list at the identical offsets -- which matters more here than
// it looks, because the Cube's dual-destination Fixpipe names the vector cores' landings by OFFSET on
// three cores and nothing in the toolchain checks that the three agree. Both services carry a
// `sizeof(scalar_t) == 2` / `sizeof(OperandT) == 2` static_assert so a third element type cannot
// quietly reach this map; it would have to come here first.
constexpr std::uint32_t kBasisUbBytes = 8 * 256 * (2 + 4 + 2 + 4) + 256 * 4 +
    (7 * 256 + 2 * 256 + 16) * 4 + (2 * 256 + 256 + 64 + 8 + 8 * 256) * 4;
constexpr std::uint32_t kReducerUbBytes = (2 * 256 + 16) * 4 + 5 * 8 * 4 +
    (2 * 64 + 2 * 8) * 4 + 256 * 2;
static_assert(kUbEnd + kBasisUbBytes + kReducerUbBytes + 256 <= 192 * 1024,
    "reserve a final full-register read pad even with raw basis and reduction");
// turboquant_gemm_reduce.hpp restates these three terms so it can compute the Cube GEMM
// reduction's UB deficit without including this header -- it is dependency-free on purpose, and
// the host gate that compiles it must not see production's -Wextra warnings. Pin the
// restatement here so a change on either side is a compile error rather than a silent drift in
// the number that decides whether that reduction can be enabled.
static_assert(kUbEnd == 142304 && kBasisUbBytes == 46432 && kReducerUbBytes == 3360,
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
    // The device turns a tile index into (logical page, row base) with one shift and one
    // mask, so a page has to be a power-of-two multiple of kTileRows. A page of 192 rows
    // would need an integer division on the SPU once per tile, which is the whole thing
    // the host tiling exists to remove; it is refused here rather than divided for.
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
    // The RAW-QUERY entry only; the pre-rotated entry ignores all three.
    //   prologue_vectors_per_block  grid * this must cover tokens * heads query vectors
    //   prologue_cube_chunk_vectors 0 = the vector cores rotate the whole share with ApplyPi;
    //                               16 = whole chunks of the share rotate on the Cube
    //   output_stage                TurboQuantOutputStage: 0 leaves the context in the rotated
    //                               basis, 1 rotates it back (Pi^2 = I), 2 also divides by
    //                               1 + exp(-gate). Only 1 and 2 read the gate buffer.
    std::uint32_t prologue_vectors_per_block = 0, prologue_cube_chunk_vectors = 0;
    std::uint32_t output_stage = 0;
    // The QUERY and OUTPUT element type, as vllm_ascend::AscendType: 0 = FP16, 1 = BF16. It does NOT
    // name the cache format, which is the launch's MODE and stays KV4/FP8 either way, and it does not
    // change the tiling: the descriptors `build_decode_tiling` produces are counts, page indices and
    // row bases, so one tiling buffer serves both instances of the operator. What it selects is which
    // kernel symbol the launcher dispatches to, and with it the type of `Buffers::output`, and of
    // `RawQueryBuffers::raw_query` and `::gate` on the raw-query entry. Default 0, so a caller that
    // predates this field keeps the fp16 behaviour it had.
    std::uint32_t element_type = 0;
    // Diagnostic ABI only: production launcher never receives these fields.
    std::uint32_t probe_page = UINT32_MAX, probe_tile = 0;
};

// The element type as the operator's own enum, refusing anything the kernel has no instance for
// rather than returning a value the `_impl` dispatch would silently drop on the floor. A launch that
// quietly computes nothing is the exact failure this operator's audit trail exists to prevent.
inline vllm_ascend::AscendType ascend_type_of(std::uint32_t element_type)
{
    switch (element_type) {
    case 0: return vllm_ascend::AscendType::FP16;
    case 1: return vllm_ascend::AscendType::BF16;
    default: throw std::invalid_argument("fused decode launch with no kernel instance for its element type");
    }
}
// How wide one output element is, for a caller sizing its output or gate buffer. Both instances are
// 16-bit; see the kBasisUbBytes note above for why that is load-bearing rather than incidental.
constexpr std::uint32_t element_bytes(std::uint32_t) { return 2; }
struct Buffers {
    void *query, *key, *value, *scales, *block_table, *context_lengths;
    void *mode_tables, *workspace, *output;
    // The host tiling descriptors, decode_tiling_bytes(tokens, splits, grid) of them,
    // built by ascend_micro::fused_decode::build_decode_tiling and uploaded like any
    // other input. Read-only on the device and read by BOTH roles. There is no device
    // fallback if it is null: the launcher refuses the launch, because a decode that
    // derived this itself is the operator this change removed.
    void *tiling = nullptr;
};

// The raw-query entry's inputs: the pre-rotated entry's buffers plus the original-basis query
// and the basis tables. `raw_query` and `gate` are scalar_t -- fp16 or bf16, as Launch::element_type
// says -- while `decode.query` is the fp32 rotated image the prologue WRITES and the decode then
// reads, so a raw-query launch needs both and they are not interchangeable. `h16` is fp16 for BOTH
// element types: its entries are +-1, it is a shared input, and the kernel converts it on ingest
// where the Cube chunk operand is not fp16 (op_kernel/cube/turboquant_query_basis.h).
struct RawQueryBuffers {
    void *raw_query, *pi_signs, *rot_tables, *h16, *gate;
    Buffers decode;
};
} // namespace ascend_micro::fused_decode_opt

extern "C" void launch_fused_decode_opt(void* stream,
    const ascend_micro::fused_decode_opt::Launch* launch,
    const ascend_micro::fused_decode_opt::Buffers* buffers);

// The raw-query counterpart. Same geometry, same validation, one extra guard set: the prologue
// share has to cover every query vector and the output stage has to name a real stage.
extern "C" void launch_fused_decode_opt_raw_query(void* stream,
    const ascend_micro::fused_decode_opt::Launch* launch,
    const ascend_micro::fused_decode_opt::RawQueryBuffers* buffers);

// Original launch APIs retained for callers that already have upstream tiling.
namespace vllm_ascend {
void turboquant_mm_fused_decode_opt_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim,
                                         void *queryRot, void *keyCache, void *valueCache, void *scaleCache,
                                         void *blockTables, void *contextLens, void *modeTables, void *workspace,
                                         void *output, void *tiling, uint32_t numTokens, uint32_t numHeads,
                                         uint32_t numKvHeads,
                                         uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                         uint32_t numSplits, uint32_t headsPerTask, uint32_t tasksPerBlock,
                                         uint32_t reduceTasksPerBlock, uint32_t fusedContextLimit, float scale,
                                         float invSqrtLen);
void turboquant_mm_fused_decode_opt_raw_query_impl(
    int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *query, void *piSigns, void *rotTables,
    void *h16, void *gate, void *queryRot, void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
    void *contextLens, void *modeTables, void *workspace, void *output, void *tiling, uint32_t numTokens,
    uint32_t numHeads,
    uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,
    uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock, uint32_t prologueVectorsPerBlock,
    uint32_t prologueCubeChunkVectors, uint32_t outputStage, uint32_t fusedContextLimit, float scale,
    float invSqrtLen);
} // namespace vllm_ascend
