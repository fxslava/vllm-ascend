#pragma once

// KV4/FP8 Cube decode: 64 keys per tile, up to eight query heads per AIV.
// The caller owns DMA, Fixpipe visibility, and inter-core synchronization.
// All arithmetic stays on Vector; state fields are replicated 32-byte blocks.
#include "routine_vector_online_softmax.hpp"

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
namespace ascend_micro {

// Two distinct sixty-fours. kCubeTileKeys is the tile's key count, which sets every row
// operand's element count and a reduction's mask. kCubeRepeatLanes is how many FP32 lanes one
// vector repeat covers, which sets the stride of a sweep across a head's channels. They are
// equal on this part and neither is the other.
constexpr std::uint32_t kCubeTileKeys    = 64;
constexpr std::uint32_t kCubeRepeatLanes = 64;

// The 32-byte blocks one tile row spans, which is a reduction's source repeat stride.
constexpr std::uint32_t kCubeTileBlocks = kCubeTileKeys * sizeof(float) / 32;

template <class Codec, class Operand>
struct NativeCubeSoftmaxEngine {
    AscendC::LocalTensor<float>   scores, key_scale, value_scale, q_inv, mask;
    AscendC::LocalTensor<float>   run_max, run_sum, tile_max, new_max, alpha;
    AscendC::LocalTensor<float>   prob_scale, part, row_reduce;
    AscendC::LocalTensor<Operand> probabilities;
    Codec&                        codec;
    std::uint32_t                 heads, valid;
    float                         operand_max, epsilon;
    AscendC::BinaryRepeatParams   by_vector, by_block;

    template <AscendC::ReduceType Kind>
    __aicore__ inline void reduce(const AscendC::LocalTensor<float>& dst)
    {
        constexpr std::int32_t mask       = static_cast<std::int32_t>(kCubeTileKeys);
        constexpr std::int32_t rep_blocks = static_cast<std::int32_t>(kCubeTileBlocks);
        if constexpr (Kind == AscendC::ReduceType::SUM) {
            AscendC::ReduceRepeat<Kind>(row_reduce, scores, mask, heads, 1, 1, rep_blocks);
        } else {
            AscendC::ReduceRepeat<Kind>(row_reduce, scores, mask, heads, 1, 1, rep_blocks,
                                       AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        }
        AscendC::PipeBarrier<PIPE_V>();
        vector_broadcast_block(dst, row_reduce);
    }

    __aicore__ inline void update_max_and_alpha()
    {
        AscendC::Mul(scores, scores, key_scale, std::uint64_t{kCubeTileKeys}, heads, by_vector);
        AscendC::Mul(scores, scores, q_inv, std::uint64_t{kCubeTileKeys}, heads, by_block);
        if (valid < kCubeTileKeys)
            AscendC::Min(scores, scores, mask, std::uint64_t{kCubeTileKeys}, heads, by_vector);
        reduce<AscendC::ReduceType::MAX>(tile_max);
        AscendC::Max(new_max, run_max, tile_max, heads * kStateLanes);
        AscendC::Sub(alpha, run_max, new_max, heads * kStateLanes);
    }

    __aicore__ inline void exponentiate_scores()
    {
        AscendC::Sub(scores, scores, new_max, std::uint64_t{kCubeTileKeys}, heads, by_block);
        AscendC::Exp(alpha, alpha, heads * kStateLanes);
        AscendC::Exp(scores, scores, heads * kCubeTileKeys);
    }

    __aicore__ inline void update_denominator()
    {
        reduce<AscendC::ReduceType::SUM>(part);
        AscendC::Mul(run_sum, run_sum, alpha, heads * kStateLanes);
        AscendC::Add(run_sum, run_sum, part, heads * kStateLanes);
    }

    // Denominator uses unscaled probabilities. Value scale belongs only in the numerator,
    // followed by the source's FP8 dynamic range map.
    __aicore__ inline void quantize_probabilities()
    {
        AscendC::Mul(scores, scores, value_scale, std::uint64_t{kCubeTileKeys}, heads, by_vector);
        reduce<AscendC::ReduceType::MAX>(part);
        AscendC::Duplicate(prob_scale, operand_max, heads * kStateLanes);
        AscendC::Adds(part, part, epsilon, heads * kStateLanes);
        AscendC::Div(prob_scale, prob_scale, part, heads * kStateLanes);
        AscendC::Mul(scores, scores, prob_scale, std::uint64_t{kCubeTileKeys}, heads, by_block);
        codec.template CastToOperand<Operand>(probabilities, scores, heads * kCubeTileKeys);
    }
};

// A replicated head block broadcasts over each 64-channel chunk. No scalar
// extraction and no reduction side channel, including for ragged head counts.
__aicore__ inline void vector_cube_scale_rows(
    const AscendC::LocalTensor<float>& rows,
    const AscendC::LocalTensor<float>& scale, std::uint32_t heads,
    std::uint32_t channels, const AscendC::BinaryRepeatParams& repeat)
{
    for (std::uint32_t col = 0; col < channels; col += kCubeRepeatLanes)
        AscendC::Mul(rows[col], rows[col], scale, std::uint64_t{kCubeRepeatLanes}, heads, repeat);
}

__aicore__ inline void vector_cube_accumulate(
    const AscendC::LocalTensor<float>& acc,
    const AscendC::LocalTensor<float>& context,
    const AscendC::LocalTensor<float>& probability_scale,
    const AscendC::LocalTensor<float>& inverse, std::uint32_t heads,
    std::uint32_t channels, const AscendC::BinaryRepeatParams& repeat)
{
    AscendC::Duplicate(inverse, 1.0f, heads * kStateLanes);
    AscendC::Div(inverse, inverse, probability_scale, heads * kStateLanes);
    vector_cube_scale_rows(context, inverse, heads, channels, repeat);
    AscendC::Add(acc, acc, context, heads * channels);
}

// alpha = exp(run_max - new_max) decays what is already accumulated, before this tile's context
// lands. The new maximum becomes the running one in the same call, so the caller sees one stage
// and cannot leave the two halves of the step apart.
__aicore__ inline void vector_cube_decay_accumulator(
    const AscendC::LocalTensor<float>& acc,
    const AscendC::LocalTensor<float>& alpha,
    const AscendC::LocalTensor<float>& run_max,
    const AscendC::LocalTensor<float>& new_max, std::uint32_t heads,
    std::uint32_t channels, const AscendC::BinaryRepeatParams& repeat)
{
    vector_cube_scale_rows(acc, alpha, heads, channels, repeat);
    AscendC::Adds(run_max, new_max, 0.0f, heads * kStateLanes);
}

// Scores of a partial last tile are Min-ed with this mask: a ramp crossing zero at `valid`,
// huge and positive before it and huge and negative after, so one Min leaves the live keys
// untouched and drives the dead ones below any running maximum. `lanes` aliases the mask
// buffer's own tail as int32 scratch for the progression.
__aicore__ inline void vector_cube_tail_mask(
    const AscendC::LocalTensor<float>&        mask,
    const AscendC::LocalTensor<std::int32_t>& lanes,
    std::uint32_t valid, float midpoint, float gain)
{
    AscendC::ArithProgression(lanes, 0, 1, static_cast<std::int32_t>(kCubeTileKeys));
    AscendC::Cast(mask, lanes, AscendC::RoundMode::CAST_NONE, kCubeTileKeys);
    AscendC::Adds(mask, mask, midpoint - static_cast<float>(valid), kCubeTileKeys);
    AscendC::Muls(mask, mask, gain, kCubeTileKeys);
}

// The task's kv head picked out of every tile row's scale slot, K lanes first and V lanes
// behind, from byte offsets the caller built once for the task. The stored scale is the whole
// per-row scale (the writers' convention, turboquant_codec_mx.h): the score scale carries the
// bare attention scale and the inverse gain is 1 -- a non-uniform codebook would reconstruct
// inside the operand and leave nothing to divide out either.
__aicore__ inline void vector_cube_tile_scales(
    const AscendC::LocalTensor<float>&         key_scale,
    const AscendC::LocalTensor<float>&         value_scale,
    const AscendC::LocalTensor<float>&         scale_tile,
    const AscendC::LocalTensor<std::uint32_t>& offsets,
    float score_scale, float inverse_gain)
{
    AscendC::Gather(key_scale, scale_tile, offsets, 0, kCubeTileKeys);
    AscendC::Gather(value_scale, scale_tile, offsets[kCubeTileKeys], 0, kCubeTileKeys);
    AscendC::Muls(key_scale, key_scale, score_scale, kCubeTileKeys);
    AscendC::Muls(value_scale, value_scale, inverse_gain, kCubeTileKeys);
}

// The online softmax's close-out: 1 / (run_sum + eps) per head, then one scaling sweep over the
// accumulator's channels.
//
// UB FOOTPRINT IS A CONTRACT HERE. `sums` is a Brcb destination, and Brcb writes 64 floats from
// sums[head * kStateLanes] for every head, so it needs (heads - 1) * kStateLanes + 64 floats and
// NOT the heads * kStateLanes a lane count suggests. Callers static_assert that `sums` cannot
// reach `inverses` and that `inverses` cannot reach whatever follows it; neither operand may be
// narrowed to its lane count.
__aicore__ inline void vector_cube_normalize_heads(
    const AscendC::LocalTensor<float>& acc,
    const AscendC::LocalTensor<float>& run_sum,
    const AscendC::LocalTensor<float>& sums,
    const AscendC::LocalTensor<float>& inverses, std::uint32_t heads,
    std::uint32_t channels, float epsilon)
{
    const std::uint8_t                row_blocks = static_cast<std::uint8_t>(channels / kStateLanes);
    const AscendC::BinaryRepeatParams by_block{1, 1, 0, row_blocks, row_blocks, 1};
    for (std::uint32_t head = 0; head < heads; ++head)
        AscendC::Adds(run_sum[head * kStateLanes], run_sum[head * kStateLanes], epsilon, 1);
    for (std::uint32_t head = 0; head < heads; ++head)
        vector_broadcast_block(sums[head * kStateLanes], run_sum[head * kStateLanes]);
    AscendC::Duplicate(inverses, 1.0f, heads * kStateLanes);
    AscendC::Div(inverses, inverses, sums, heads * kStateLanes);
    vector_cube_scale_rows(acc, inverses, heads, channels, by_block);
}

// One split's workspace record per head is [accumulator (channels), run_max lane, run_sum lane]
// and `stride` is the tail's whole width, so a record stays 32-byte aligned however few lanes
// carry state. The zeroing and the two lane writes are one vector chain; ccec orders the
// overlap within the pipe.
__aicore__ inline void vector_cube_pack_partial_tails(
    const AscendC::LocalTensor<float>& tails,
    const AscendC::LocalTensor<float>& run_max,
    const AscendC::LocalTensor<float>& run_sum, std::uint32_t heads,
    std::uint32_t stride, std::uint32_t max_lane, std::uint32_t sum_lane)
{
    AscendC::Duplicate(tails, 0.0f, heads * stride);
    for (std::uint32_t head = 0; head < heads; ++head) {
        AscendC::Adds(tails[head * stride + max_lane], run_max[head * kStateLanes], 0.0f, 1);
        AscendC::Adds(tails[head * stride + sum_lane], run_sum[head * kStateLanes], 0.0f, 1);
    }
}

// Query quantization onto the operand grid, all of a subcore's rows at once. The per-head
// reciprocals stay in UB as replicated blocks, so nothing here waits on the vector unit and the
// softmax reads them as a block operand rather than through a scalar.
//
// ReduceMax AND NOT ReduceRepeat, deliberately: a query row is wider than one vector register,
// so the reduction cannot be a single repeat, and ReduceMax's library-internal scalar readback
// is harmless because the decode runs under a TPipe. The leaf recurrence in
// routine_vector_online_softmax.hpp may not use it -- a kernel that places its own LocalTensors
// has a null GetTPipePtr() and traps at GM 0x40. Do not lift this engine into such a kernel.
template <class Codec, class Operand>
struct NativeCubeQueryEngine {
    AscendC::LocalTensor<float>   query, magnitude, scratch;
    AscendC::LocalTensor<float>   row_max, amax, scale, inverse;
    AscendC::LocalTensor<Operand> operand;
    Codec&                        codec;
    std::uint32_t                 heads, channels;
    float                         operand_max, epsilon;
    AscendC::BinaryRepeatParams   by_block;

    __aicore__ inline void measure_magnitude()
    {
        AscendC::Abs(magnitude, query, heads * channels);
        for (std::uint32_t head = 0; head < heads; ++head)
            AscendC::ReduceMax<float>(row_max[head], magnitude[head * channels], scratch, channels, false);
    }

    __aicore__ inline void derive_scale()
    {
        vector_broadcast_block(amax, row_max);
        AscendC::Adds(amax, amax, epsilon, heads * kStateLanes);
        AscendC::Duplicate(scale, operand_max, heads * kStateLanes);
        AscendC::Duplicate(inverse, 1.0f, heads * kStateLanes);
        AscendC::Div(scale, scale, amax, heads * kStateLanes);
        AscendC::Div(inverse, inverse, scale, heads * kStateLanes);
    }

    __aicore__ inline void quantize()
    {
        vector_cube_scale_rows(query, scale, heads, channels, by_block);
        codec.template CastToOperand<Operand>(operand, query, heads * channels);
    }
};

template <class Engine>
__aicore__ inline void vector_cube_quantize_query(Engine& engine)
{
    engine.measure_magnitude();
    engine.derive_scale();
    engine.quantize();
}
} // namespace ascend_micro
#endif
