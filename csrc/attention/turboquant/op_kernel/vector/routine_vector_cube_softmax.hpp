#pragma once

// KV4/FP8 Cube decode: 64 keys per tile, up to eight query heads per AIV.
// The caller owns DMA, Fixpipe visibility, and inter-core synchronization.
// All arithmetic stays on Vector; state fields are replicated 32-byte blocks.
#include "routine_vector_online_softmax.hpp"

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
namespace ascend_micro {

template <class Codec, class Operand>
struct NativeCubeSoftmaxEngine {
    AscendC::LocalTensor<float> scores, key_scale, value_scale, q_inv, mask;
    AscendC::LocalTensor<float> run_max, run_sum, tile_max, new_max, alpha;
    AscendC::LocalTensor<float> prob_scale, part, row_reduce;
    AscendC::LocalTensor<Operand> probabilities;
    Codec& codec;
    std::uint32_t heads, valid;
    float operand_max, epsilon;
    AscendC::BinaryRepeatParams by_vector, by_block;

    template <AscendC::ReduceType Kind>
    __aicore__ inline void reduce(const AscendC::LocalTensor<float>& dst)
    {
        if constexpr (Kind == AscendC::ReduceType::SUM) {
            AscendC::ReduceRepeat<Kind>(row_reduce, scores, 64, heads, 1, 1, 8);
        } else {
            AscendC::ReduceRepeat<Kind>(row_reduce, scores, 64, heads, 1, 1, 8,
                                       AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(dst, row_reduce, 1, {1, 8});
    }

    __aicore__ inline void update_max_and_alpha()
    {
        AscendC::Mul(scores, scores, key_scale, uint64_t{64}, heads, by_vector);
        AscendC::Mul(scores, scores, q_inv, uint64_t{64}, heads, by_block);
        if (valid < 64)
            AscendC::Min(scores, scores, mask, uint64_t{64}, heads, by_vector);
        reduce<AscendC::ReduceType::MAX>(tile_max);
        AscendC::Max(new_max, run_max, tile_max, heads * 8);
        AscendC::Sub(alpha, run_max, new_max, heads * 8);
    }

    __aicore__ inline void exponentiate_scores()
    {
        AscendC::Sub(scores, scores, new_max, uint64_t{64}, heads, by_block);
        AscendC::Exp(alpha, alpha, heads * 8);
        AscendC::Exp(scores, scores, heads * 64);
    }

    __aicore__ inline void update_denominator()
    {
        reduce<AscendC::ReduceType::SUM>(part);
        AscendC::Mul(run_sum, run_sum, alpha, heads * 8);
        AscendC::Add(run_sum, run_sum, part, heads * 8);
    }

    __aicore__ inline void quantize_probabilities()
    {
        // Denominator uses unscaled probabilities. Value scale belongs only
        // in the numerator, followed by the source's FP8 dynamic range map.
        AscendC::Mul(scores, scores, value_scale, uint64_t{64}, heads, by_vector);
        reduce<AscendC::ReduceType::MAX>(part);
        AscendC::Duplicate(prob_scale, operand_max, heads * 8);
        AscendC::Adds(part, part, epsilon, heads * 8);
        AscendC::Div(prob_scale, prob_scale, part, heads * 8);
        AscendC::Mul(scores, scores, prob_scale, uint64_t{64}, heads, by_block);
        codec.template CastToOperand<Operand>(probabilities, scores, heads * 64);
    }
};

// A replicated head block broadcasts over each 64-channel chunk. No scalar
// extraction and no reduction side channel, including for ragged head counts.
__aicore__ inline void vector_cube_scale_rows(
    const AscendC::LocalTensor<float>& rows,
    const AscendC::LocalTensor<float>& scale, std::uint32_t heads,
    std::uint32_t channels, const AscendC::BinaryRepeatParams& repeat)
{
    for (std::uint32_t col = 0; col < channels; col += 64)
        AscendC::Mul(rows[col], rows[col], scale, uint64_t{64}, heads, repeat);
}

__aicore__ inline void vector_cube_accumulate(
    const AscendC::LocalTensor<float>& acc,
    const AscendC::LocalTensor<float>& context,
    const AscendC::LocalTensor<float>& probability_scale,
    const AscendC::LocalTensor<float>& inverse, std::uint32_t heads,
    std::uint32_t channels, const AscendC::BinaryRepeatParams& repeat)
{
    AscendC::Duplicate(inverse, 1.0f, heads * 8);
    AscendC::Div(inverse, inverse, probability_scale, heads * 8);
    vector_cube_scale_rows(context, inverse, heads, channels, repeat);
    AscendC::Add(acc, acc, context, heads * channels);
}
} // namespace ascend_micro
#endif
