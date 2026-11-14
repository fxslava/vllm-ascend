#pragma once

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "kernel_operator.h"
#endif

namespace ascend_micro {

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#define ASCEND_MICRO_ONLINE_INLINE __aicore__ inline
#else
#define ASCEND_MICRO_ONLINE_INLINE inline
#endif

constexpr std::uint32_t kOnlineTile = 16;

// One UB DataBlock holds 8 FP32. Every logically scalar element of the online
// recurrence (m, l, alpha, tile max, row sum) is carried as a whole DataBlock
// with the value replicated across all 8 lanes.
constexpr std::uint32_t kStateLanes = 8;

// Brcb consumes 8 source elements per repeat and writes 8 DataBlocks, so one
// repeat produces exactly 64 FP32 of broadcast output.
constexpr std::uint32_t kBroadcastLanes = 64;

// All state and scratch operands are UB resident. The four phases only issue
// vector instructions; the caller owns input readiness and output visibility.
template <class Engine>
ASCEND_MICRO_ONLINE_INLINE void vector_online_softmax_probabilities(Engine& engine)
{
    engine.update_max_and_alpha();
    engine.exponentiate_scores();
    engine.update_denominator();
}

// Cube attention stages these probabilities into L1 before the context GEMM.
// Keep the same recurrence phases for Vector-only and Cube-backed attention.
template <class Engine>
ASCEND_MICRO_ONLINE_INLINE void vector_online_softmax(Engine& engine)
{
    vector_online_softmax_probabilities(engine);
    engine.update_accumulator();
}

// Split-K merge, phase for phase like the recurrence above. A token whose context exceeds the
// fused limit leaves one [accumulator, run_max, run_sum] partial per split; merging two of them
// is one online-softmax step with both sides weighted.
template <class Engine>
ASCEND_MICRO_ONLINE_INLINE void vector_merge_partials(Engine& engine)
{
    engine.update_max_and_weights();
    engine.update_denominator();
    engine.update_accumulator();
    engine.commit_max();
}

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)

// Replicated-state invariant
// --------------------------
// m_prev/l_prev/m_new/l_new/alpha/alpha_delta/l_scaled each occupy one full
// 32-byte DataBlock whose 8 FP32 lanes all hold the same value. Two properties
// follow, and both are load-bearing:
//
//   1. Every state update runs at count == kStateLanes, that is exactly one
//      whole DataBlock from a 32-byte aligned base. No sub-block operand and
//      no partially covered block appears anywhere in the recurrence.
//   2. Brcb reads 8 source elements and splats each into its own DataBlock.
//      Fed a replicated block, one Brcb repeat fills all 64 destination lanes
//      with the same value, so a single instruction broadcasts the state to
//      any tile width up to kBroadcastLanes with no scalar trip.
//
// Reduction destinations (tile_max_raw, p_sum_raw) are the one exception: a
// reduction writes lane 0 only, so those blocks are not replicated. They are
// passed straight to Brcb, which re-establishes replication in its first
// output block.
//
// Why Brcb and not Duplicate
// --------------------------
// Duplicate(dst, srcTensor, count) is a documented 950PR prototype, but on
// dav-3510 it lowers to an unmasked Reg::LoadAlign of a full vector register
// (VECTOR_REG_WIDTH == 256 bytes) from the source base, independent of count.
// Pointed at a 32-byte state block that is a 224-byte over-read, and it faults
// outright when the block lies within 256 bytes of the end of the UB window.
// The load-distribute mode Brcb uses (DIST_E2B_B32) reads exactly one 32-byte
// block, so its source footprint matches its operand.
//
// Why ReduceRepeat and not ReduceMax/ReduceSum
// --------------------------------------------
// ReduceSum<float> on dav-3510 ends with an unconditional, library-internal
// scalar readback: it allocates a HardEvent::V_S event from GetTPipePtr(),
// drains the vector pipe through SetFlag/WaitFlag, then executes
// accValFloat = dstLocal[0] so that GetAccVal() can observe the result. In a
// kernel that places its own LocalTensors and never instantiates a TPipe,
// GetTPipePtr() is null and the event-bitmap read/write lands at GM address
// 0x40, which is the "invalid ldst addr" trap. ReduceRepeat has no accumulator
// side channel, needs no workspace, and never leaves the vector pipe.
//
// PipeBarrier<PIPE_V> placement
// -----------------------------
// A PipeBarrier<PIPE_V> is intra-pipe ordering, not a cross-engine handshake:
// it consumes no event ID and creates no token the caller must balance, so it
// does not breach leaf-routine purity the way SetFlag/WaitFlag would. It is
// issued only at the two reduction to broadcast boundaries, where a 4-byte
// lane-0 write is consumed as a full 32-byte block read.
struct NativeOnlineSoftmaxEngine {
    // Tile inputs.
    AscendC::LocalTensor<float> scores;                    // kOnlineTile
    AscendC::LocalTensor<float> v_by_channel;              // kOnlineTile^2
    AscendC::LocalTensor<std::uint32_t> p_repeat_offsets;  // kOnlineTile^2

    // Carried state, replicated 32-byte blocks.
    AscendC::LocalTensor<float> m_prev;       // kStateLanes
    AscendC::LocalTensor<float> l_prev;       // kStateLanes
    AscendC::LocalTensor<float> m_new;        // kStateLanes
    AscendC::LocalTensor<float> l_new;        // kStateLanes
    AscendC::LocalTensor<float> alpha_delta;  // kStateLanes
    AscendC::LocalTensor<float> alpha;        // kStateLanes
    AscendC::LocalTensor<float> l_scaled;     // kStateLanes

    // Carried accumulator, one lane per channel.
    AscendC::LocalTensor<float> acc_prev;    // kOnlineTile
    AscendC::LocalTensor<float> acc_new;     // kOnlineTile
    AscendC::LocalTensor<float> acc_scaled;  // kOnlineTile

    // Reduction destinations: lane 0 only, deliberately not replicated.
    AscendC::LocalTensor<float> tile_max_raw;  // kStateLanes
    AscendC::LocalTensor<float> p_sum_raw;     // kStateLanes

    // Brcb destinations: one repeat writes kBroadcastLanes FP32 (256 bytes).
    AscendC::LocalTensor<float> tile_max_bc;  // kBroadcastLanes
    AscendC::LocalTensor<float> m_bc;         // kBroadcastLanes
    AscendC::LocalTensor<float> p_sum_bc;     // kBroadcastLanes
    AscendC::LocalTensor<float> alpha_bc;     // kBroadcastLanes

    // Tile scratch.
    AscendC::LocalTensor<float> centered;      // kOnlineTile
    AscendC::LocalTensor<float> p;             // kOnlineTile
    AscendC::LocalTensor<float> p_repeated;    // kOnlineTile^2
    AscendC::LocalTensor<float> weighted;      // kOnlineTile^2
    AscendC::LocalTensor<float> weighted_sum;  // kOnlineTile

    // kOnlineTile FP32 == 64 bytes == 2 DataBlocks: one reduction repeat.
    static constexpr std::int32_t kTileLanes =
        static_cast<std::int32_t>(kOnlineTile);
    static constexpr std::int32_t kTileBlocks =
        static_cast<std::int32_t>(kOnlineTile * sizeof(float) / 32);

    __aicore__ inline void update_max_and_alpha()
    {
        AscendC::ReduceRepeat<AscendC::ReduceType::MAX, float>(
            tile_max_raw, scores, kTileLanes, 1, 1, 1, kTileBlocks,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(tile_max_bc, tile_max_raw, 1, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max(m_new, m_prev, tile_max_bc, kStateLanes);
        AscendC::Sub(alpha_delta, m_prev, m_new, kStateLanes);
        AscendC::Exp(alpha, alpha_delta, kStateLanes);
    }
    __aicore__ inline void exponentiate_scores()
    {
        // m_new is replicated, so all kBroadcastLanes of m_bc receive m_new.
        AscendC::Brcb(m_bc, m_new, 1, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(centered, scores, m_bc, kOnlineTile);
        AscendC::Exp(p, centered, kOnlineTile);
    }
    __aicore__ inline void update_denominator()
    {
        AscendC::ReduceRepeat<AscendC::ReduceType::SUM, float>(
            p_sum_raw, p, kTileLanes, 1, 1, 1, kTileBlocks,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(p_sum_bc, p_sum_raw, 1, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(l_scaled, l_prev, alpha, kStateLanes);
        AscendC::Add(l_new, l_scaled, p_sum_bc, kStateLanes);
    }
    __aicore__ inline void update_accumulator()
    {
        // Caller supplies kOnlineTile columns of V contiguously and byte
        // offsets repeating 0,4,...,60 per column. No scalar index generation.
        constexpr std::uint32_t panel = kOnlineTile * kOnlineTile;
        AscendC::Gather(p_repeated, p, p_repeat_offsets, 0, panel);
        AscendC::Mul(weighted, p_repeated, v_by_channel, panel);
        AscendC::ReduceRepeat<AscendC::ReduceType::SUM, float>(
            weighted_sum, weighted, kTileLanes, kTileLanes, 1, 1, kTileBlocks,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        // alpha is replicated, so one repeat covers all kOnlineTile channels.
        AscendC::Brcb(alpha_bc, alpha, 1, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(acc_scaled, acc_prev, alpha_bc, kOnlineTile);
        AscendC::Add(acc_new, acc_scaled, weighted_sum, kOnlineTile);
    }
};

// A replicated 32-byte block out of one scalar lane. The load-distribute mode Brcb uses reads
// exactly one block, so the source footprint matches the operand -- see the Brcb rationale above
// for why this is never Duplicate.
__aicore__ inline void vector_broadcast_block(
    const AscendC::LocalTensor<float>& destination,
    const AscendC::LocalTensor<float>& source)
{
    AscendC::Brcb(destination, source, 1, {1, static_cast<std::uint16_t>(kStateLanes)});
}

// destination = source * a replicated block, one whole repeat at a time with the ragged
// remainder as a single-repeat tail. The block operand advances by nothing, so one 32-byte
// block scales every lane it covers.
__aicore__ inline void vector_broadcast_mul(
    const AscendC::LocalTensor<float>& destination,
    const AscendC::LocalTensor<float>& source,
    const AscendC::LocalTensor<float>& block, std::uint32_t count)
{
    constexpr std::uint8_t whole_blocks = static_cast<std::uint8_t>(kBroadcastLanes / kStateLanes);
    const std::uint32_t    repeats      = count / kBroadcastLanes;
    const std::uint32_t    covered      = repeats * kBroadcastLanes;
    const std::uint32_t    tail         = count - covered;
    if (repeats > 0)
        AscendC::Mul(destination, source, block, std::uint64_t{kBroadcastLanes}, static_cast<std::uint8_t>(repeats), {1, 1, 0, whole_blocks, whole_blocks, 0});
    if (tail > 0)
        AscendC::Mul(destination[covered], source[covered], block, std::uint64_t{tail}, 1, {1, 1, 0, 0, 0, 0});
}

// Unlike the tile recurrence, every state operand here is a SINGLE LANE and not a replicated
// block: a merge advances one head at a time, and the two broadcast destinations are the only
// wide operands. That is why the accumulator scaling goes through vector_broadcast_mul rather
// than a block-strided repeat -- the weights arrive as lanes and are splatted here.
struct NativeMergeEngine {
    AscendC::LocalTensor<float> accumulator, partial_accumulator;
    AscendC::LocalTensor<float> run_max, run_sum, partial_max, partial_sum;
    AscendC::LocalTensor<float> new_max, alpha, beta;
    AscendC::LocalTensor<float> alpha_block, beta_block;
    std::uint32_t               channels;

    __aicore__ inline void update_max_and_weights()
    {
        AscendC::Max(new_max, run_max, partial_max, 1);
        AscendC::Sub(alpha, run_max, new_max, 1);
        AscendC::Sub(beta, partial_max, new_max, 1);
        AscendC::Exp(alpha, alpha, 1);
        AscendC::Exp(beta, beta, 1);
    }
    __aicore__ inline void update_denominator()
    {
        AscendC::Mul(run_sum, run_sum, alpha, 1);
        AscendC::Mul(partial_sum, partial_sum, beta, 1);
        AscendC::Add(run_sum, run_sum, partial_sum, 1);
    }
    __aicore__ inline void update_accumulator()
    {
        vector_broadcast_block(alpha_block, alpha);
        vector_broadcast_block(beta_block, beta);
        vector_broadcast_mul(accumulator, accumulator, alpha_block, channels);
        vector_broadcast_mul(partial_accumulator, partial_accumulator, beta_block, channels);
        AscendC::Add(accumulator, accumulator, partial_accumulator, channels);
    }
    __aicore__ inline void commit_max() { AscendC::Adds(run_max, new_max, 0.0f, 1); }
};

#endif

#undef ASCEND_MICRO_ONLINE_INLINE
}  // namespace ascend_micro
