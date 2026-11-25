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

constexpr std::uint32_t kStateLanes = 8;

constexpr std::uint32_t kBroadcastLanes = 64;

template <class Engine>
ASCEND_MICRO_ONLINE_INLINE void vector_online_softmax_probabilities(Engine& engine)
{
    engine.update_max_and_alpha();
    engine.exponentiate_scores();
    engine.update_denominator();
}

template <class Engine>
ASCEND_MICRO_ONLINE_INLINE void vector_online_softmax(Engine& engine)
{
    vector_online_softmax_probabilities(engine);
    engine.update_accumulator();
}

template <class Engine>
ASCEND_MICRO_ONLINE_INLINE void vector_online_softmax_probabilities_probed(Engine& engine)
{
    engine.update_max_and_alpha();
    engine.argmax_probe();
    engine.exponentiate_scores();
    engine.update_denominator();
}

template <class Engine>
ASCEND_MICRO_ONLINE_INLINE void vector_merge_partials(Engine& engine)
{
    engine.update_max_and_weights();
    engine.update_denominator();
    engine.update_accumulator();
    engine.commit_max();
}

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)

struct NativeOnlineSoftmaxEngine {

    AscendC::LocalTensor<float> scores;
    AscendC::LocalTensor<float> v_by_channel;
    AscendC::LocalTensor<std::uint32_t> p_repeat_offsets;

    AscendC::LocalTensor<float> m_prev;
    AscendC::LocalTensor<float> l_prev;
    AscendC::LocalTensor<float> m_new;
    AscendC::LocalTensor<float> l_new;
    AscendC::LocalTensor<float> alpha_delta;
    AscendC::LocalTensor<float> alpha;
    AscendC::LocalTensor<float> l_scaled;

    AscendC::LocalTensor<float> acc_prev;
    AscendC::LocalTensor<float> acc_new;
    AscendC::LocalTensor<float> acc_scaled;

    AscendC::LocalTensor<float> tile_max_raw;
    AscendC::LocalTensor<float> p_sum_raw;

    AscendC::LocalTensor<float> tile_max_bc;
    AscendC::LocalTensor<float> m_bc;
    AscendC::LocalTensor<float> p_sum_bc;
    AscendC::LocalTensor<float> alpha_bc;

    AscendC::LocalTensor<float> centered;
    AscendC::LocalTensor<float> p;
    AscendC::LocalTensor<float> p_repeated;
    AscendC::LocalTensor<float> weighted;
    AscendC::LocalTensor<float> weighted_sum;

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

        constexpr std::uint32_t panel = kOnlineTile * kOnlineTile;
        AscendC::Gather(p_repeated, p, p_repeat_offsets, 0, panel);
        AscendC::Mul(weighted, p_repeated, v_by_channel, panel);
        AscendC::ReduceRepeat<AscendC::ReduceType::SUM, float>(
            weighted_sum, weighted, kTileLanes, kTileLanes, 1, 1, kTileBlocks,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);

        AscendC::Brcb(alpha_bc, alpha, 1, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(acc_scaled, acc_prev, alpha_bc, kOnlineTile);
        AscendC::Add(acc_new, acc_scaled, weighted_sum, kOnlineTile);
    }
};

__aicore__ inline void vector_broadcast_block(
    const AscendC::LocalTensor<float>& destination,
    const AscendC::LocalTensor<float>& source)
{
    AscendC::Brcb(destination, source, 1, {1, static_cast<std::uint16_t>(kStateLanes)});
}

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
}
