#pragma once

#include "routine_vector_online_softmax.hpp"

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
namespace ascend_micro {

constexpr std::uint32_t kCubeTileKeys    = 64;
constexpr std::uint32_t kCubeRepeatLanes = 64;

constexpr std::uint32_t kCubeTileBlocks = kCubeTileKeys * sizeof(float) / 32;

constexpr float kTailDeadLogit = -3.4028235e38f;

constexpr std::uint32_t kRegisterSoftmaxHeads = 3;
static_assert(kCubeTileKeys * sizeof(float) == 256);
static_assert(kStateLanes * sizeof(float) == 32);

__simd_callee__ inline void quantize_native_probability_row(
    AscendC::Reg::RegTensor<float>& weighted_row, __ubuf__ float* scale_out,
    __ubuf__ fp8_e4m3fn_t* operand_out, AscendC::Reg::MaskReg& live_mask,
    float operand_max, float epsilon)
{
    namespace reg = AscendC::Reg;
    auto row_mask = reg::CreateMask<float, reg::MaskPattern::ALL>();
    auto state_mask = reg::CreateMask<float, reg::MaskPattern::VL8>();
    reg::RegTensor<float> zero, reduced, denominator, numerator, scale, broadcast;

    reg::Duplicate<float>(zero, 0.0f);
    reg::Select<float>(weighted_row, weighted_row, zero, live_mask);
    reg::Reduce<reg::ReduceType::MAX>(reduced, weighted_row, row_mask);
    reg::Duplicate(denominator, reduced, state_mask);
    reg::Adds(denominator, denominator, epsilon, state_mask);
    reg::Duplicate<float>(numerator, operand_max);

    reg::Div<float, reg::MaskMergeMode::ZEROING>(scale, numerator, denominator, state_mask);
    reg::StoreAlign(scale_out, scale, state_mask);
    reg::Duplicate(broadcast, scale, row_mask);
    reg::Mul(weighted_row, weighted_row, broadcast, row_mask);

    static constexpr reg::CastTrait cast_trait = {
        reg::RegLayout::ZERO, reg::SatMode::SAT, reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    reg::RegTensor<fp8_e4m3fn_t> packed;
    reg::Cast<fp8_e4m3fn_t, float, cast_trait>(packed, weighted_row, row_mask);
    reg::StoreAlign<fp8_e4m3fn_t, reg::StoreDist::DIST_PACK4_B32>(operand_out, packed, row_mask);
}

template <bool PACK_NOW>
__simd_callee__ inline void online_softmax_loaded_row(
    AscendC::Reg::RegTensor<float>& row, __ubuf__ float* decay_out, __ubuf__ float* maximum_out,
    __ubuf__ float* ramp, std::uint32_t context_base, std::uint32_t logical_page,
    AscendC::Reg::RegTensor<float>& run_max, AscendC::Reg::RegTensor<float>& run_sum,
    AscendC::Reg::RegTensor<float>& position, AscendC::Reg::RegTensor<float>& page,
    AscendC::Reg::RegTensor<float>& value_scale,
    AscendC::Reg::RegTensor<float>& key_scale, AscendC::Reg::RegTensor<float>& q_inv,
    AscendC::Reg::RegTensor<float>& dead, AscendC::Reg::MaskReg& live_mask,
    __ubuf__ float* probability_scale = nullptr, __ubuf__ fp8_e4m3fn_t* probability = nullptr,
    float operand_max = 0.0f, float epsilon = 0.0f)
{
    namespace reg = AscendC::Reg;
    auto row_mask = reg::CreateMask<float, reg::MaskPattern::ALL>();
    auto state_mask = reg::CreateMask<float, reg::MaskPattern::VL8>();
    reg::RegTensor<float> reduced, tile_max, next_max, decay, scaled;

    reg::Mul(row, row, key_scale, row_mask);
    reg::Mul(row, row, q_inv, row_mask);
    reg::Select<float>(row, row, dead, live_mask);
    reg::Reduce<reg::ReduceType::MAX>(reduced, row, row_mask);
    reg::Duplicate(tile_max, reduced, row_mask);
    reg::Max(next_max, run_max, tile_max, row_mask);
#if defined(ASCEND_TQ_DECODE_TELEMETRY) && ASCEND_TQ_DECODE_TELEMETRY

    reg::RegTensor<float> diff, zero, ramp_row, sentinel, candidate;
    reg::MaskReg equal, improved;
    reg::Duplicate<float>(zero, 0.0f);
    reg::Duplicate<float>(sentinel, static_cast<float>(kCubeTileKeys));
    reg::Sub(diff, row, tile_max, row_mask);
    reg::Compare<float, AscendC::CMPMODE::EQ>(equal, diff, zero, row_mask);
    reg::LoadAlign(ramp_row, ramp);
    reg::Select<float>(candidate, ramp_row, sentinel, equal);
    reg::Reduce<reg::ReduceType::MIN>(reduced, candidate, row_mask);
    reg::Duplicate(candidate, reduced, state_mask);
    reg::Adds(candidate, candidate, static_cast<float>(context_base), state_mask);
    reg::Sub(diff, tile_max, run_max, state_mask);
    reg::Compare<float, AscendC::CMPMODE::GT>(improved, diff, zero, state_mask);
    reg::Select<float>(position, candidate, position, improved);
    reg::Duplicate<float>(candidate, static_cast<float>(logical_page));
    reg::Select<float>(page, candidate, page, improved);
#else
    static_cast<void>(position);
    static_cast<void>(page);
    static_cast<void>(ramp);
    static_cast<void>(context_base);
    static_cast<void>(logical_page);
#endif
    reg::Sub(decay, run_max, next_max, state_mask);
    reg::Sub(row, row, next_max, row_mask);
    reg::Exp(decay, decay, state_mask);
    reg::Exp(row, row, row_mask);
    reg::Reduce<reg::ReduceType::SUM>(reduced, row, row_mask);
    reg::Duplicate(tile_max, reduced, state_mask);

    reg::Mul(scaled, run_sum, decay, state_mask);
    reg::Add(run_sum, scaled, tile_max, state_mask);
    reg::Mul(row, row, value_scale, row_mask);
    reg::StoreAlign(decay_out, decay, state_mask);
    reg::StoreAlign(maximum_out, next_max, state_mask);
    reg::Move(run_max, next_max);
    if constexpr (PACK_NOW) {
        quantize_native_probability_row(row, probability_scale, probability, live_mask, operand_max, epsilon);
    }
}

__simd_callee__ inline void load_native_softmax_pair_head(
    AscendC::Reg::RegTensor<float>& first, AscendC::Reg::RegTensor<float>& second,
    AscendC::Reg::RegTensor<float>& m, AscendC::Reg::RegTensor<float>& l,
    AscendC::Reg::RegTensor<float>& pos, AscendC::Reg::RegTensor<float>& page_id,
    AscendC::Reg::RegTensor<float>& q_scale,
    __ubuf__ float* first_scores, __ubuf__ float* second_scores, __ubuf__ float* run_max,
    __ubuf__ float* run_sum, __ubuf__ float* position, __ubuf__ float* page,
    __ubuf__ float* q_inv, std::uint32_t head)
{
    namespace reg = AscendC::Reg;
    const std::uint32_t row = head * kCubeTileKeys;
    const std::uint32_t state = head * kStateLanes;
    reg::LoadAlign(first, first_scores + row);
    reg::LoadAlign(second, second_scores + row);
    reg::LoadAlign<float, reg::LoadDist::DIST_BRC_B32>(q_scale, q_inv + state);
    reg::LoadAlign<float, reg::LoadDist::DIST_BRC_B32>(m, run_max + state);
    reg::LoadAlign<float, reg::LoadDist::DIST_BRC_B32>(l, run_sum + state);
#if defined(ASCEND_TQ_DECODE_TELEMETRY) && ASCEND_TQ_DECODE_TELEMETRY
    reg::LoadAlign<float, reg::LoadDist::DIST_BRC_B32>(pos, position + state);
    reg::LoadAlign<float, reg::LoadDist::DIST_BRC_B32>(page_id, page + state);
#else
    static_cast<void>(pos);
    static_cast<void>(page_id);
    static_cast<void>(position);
    static_cast<void>(page);
#endif
}

__simd_vf__ inline void native_cube_softmax_pair_vf(
    __ubuf__ float* first_scores, __ubuf__ float* second_scores,
    __ubuf__ float* first_value_scale, __ubuf__ float* second_value_scale,
    __ubuf__ float* run_max, __ubuf__ float* run_sum,
    __ubuf__ float* first_max, __ubuf__ float* first_alpha,
    __ubuf__ float* second_max, __ubuf__ float* second_alpha,
    __ubuf__ float* position, __ubuf__ float* page, __ubuf__ float* ramp,
    std::uint32_t first_context, std::uint32_t first_page,
    std::uint32_t second_context, std::uint32_t second_page,
    __ubuf__ float* first_k, __ubuf__ float* second_k, __ubuf__ float* q_inv,
    std::uint32_t second_valid, __ubuf__ float* first_probability_scale, __ubuf__ float* second_probability_scale,
    __ubuf__ fp8_e4m3fn_t* first_probability, __ubuf__ fp8_e4m3fn_t* second_probability,
    float operand_max, float epsilon)
{
    namespace reg = AscendC::Reg;
    auto state_mask = reg::CreateMask<float, reg::MaskPattern::VL8>();
    reg::RegTensor<float> first_values, second_values;
    reg::LoadAlign(first_values, first_value_scale);
    reg::LoadAlign(second_values, second_value_scale);
    auto first_live = reg::CreateMask<float, reg::MaskPattern::ALL>();
    auto second_live = reg::UpdateMask<float>(second_valid);
    reg::RegTensor<float> first_keys, second_keys, dead;
    reg::LoadAlign(first_keys, first_k);
    reg::LoadAlign(second_keys, second_k);
    reg::Duplicate<float>(dead, kTailDeadLogit);
    reg::RegTensor<float> first_row, second_row, next_second_row;
    reg::RegTensor<float> m, l, pos, page_id, q_scale;
    load_native_softmax_pair_head(first_row, second_row, m, l, pos, page_id, q_scale, first_scores,
                           second_scores, run_max, run_sum, position, page, q_inv, 0);

    #pragma clang loop unroll(disable)
    for (std::uint32_t head = 0; head < kRegisterSoftmaxHeads; ++head) {
        const std::uint32_t state = head * kStateLanes;
        const std::uint32_t row = head * kCubeTileKeys;
        online_softmax_loaded_row<true>(
            first_row, first_alpha + state, first_max + state, ramp, first_context, first_page,
            m, l, pos, page_id, first_values, first_keys, q_scale, dead, first_live,
            first_probability_scale + state, first_probability + row, operand_max, epsilon);
        online_softmax_loaded_row<false>(
            second_row, second_alpha + state, second_max + state, ramp, second_context, second_page,
            m, l, pos, page_id, second_values, second_keys, q_scale, dead, second_live,
            second_probability_scale + state, second_probability + row, operand_max, epsilon);
        reg::StoreAlign(run_sum + state, l, state_mask);
#if defined(ASCEND_TQ_DECODE_TELEMETRY) && ASCEND_TQ_DECODE_TELEMETRY
        reg::StoreAlign(position + state, pos, state_mask);
        reg::StoreAlign(page + state, page_id, state_mask);
#endif
        const std::uint32_t next_head = head + 1;
        if (next_head < kRegisterSoftmaxHeads) {
            load_native_softmax_pair_head(first_row, next_second_row, m, l, pos, page_id, q_scale,
                                   first_scores, second_scores, run_max, run_sum, position, page,
                                   q_inv, next_head);
        }
        quantize_native_probability_row(second_row, second_probability_scale + state,
                                         second_probability + row, second_live, operand_max, epsilon);
        if (next_head < kRegisterSoftmaxHeads) {
            reg::Move(second_row, next_second_row);
        }
    }
}

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

    AscendC::LocalTensor<float>   probe_pos, probe_page, probe_ramp;
    AscendC::LocalTensor<float>   probe_bc, probe_diff, probe_sel;
    AscendC::LocalTensor<float>   probe_stage, probe_cand_pos, probe_cand_page;
    AscendC::LocalTensor<float>   probe_pred;
    std::uint32_t                 probe_context_base = 0, probe_page_id = 0;

    __aicore__ inline void argmax_probe()
    {
#if defined(ASCEND_TQ_DECODE_TELEMETRY) && ASCEND_TQ_DECODE_TELEMETRY
        const std::uint32_t lanes = heads * kStateLanes;
        const AscendC::LocalTensor<std::uint8_t> predicate =
            probe_pred.template ReinterpretCast<std::uint8_t>();
        for (std::uint32_t head = 0; head < heads; ++head) {
            AscendC::Brcb(probe_bc, tile_max[head * kStateLanes], 1, {1, 8});
            AscendC::Sub(probe_diff, scores[head * kCubeTileKeys], probe_bc,
                         std::uint64_t{kCubeTileKeys});
            AscendC::Compares(predicate, probe_diff, 0.0f, AscendC::CMPMODE::EQ,
                              std::uint32_t{kCubeTileKeys});
            AscendC::Select(probe_sel, predicate, probe_ramp,
                            static_cast<float>(kCubeTileKeys),
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                            std::uint32_t{kCubeTileKeys});
            AscendC::ReduceRepeat<AscendC::ReduceType::MIN>(
                probe_stage, probe_sel, static_cast<std::int32_t>(kCubeTileKeys), 1, 1, 1,
                static_cast<std::int32_t>(kCubeTileBlocks), AscendC::ReduceOrder::ORDER_ONLY_VALUE);
            AscendC::Brcb(probe_bc, probe_stage, 1, {1, 8});
            AscendC::Adds(probe_cand_pos[head * kStateLanes], probe_bc, 0.0f,
                          std::uint64_t{kStateLanes});
        }
        AscendC::Adds(probe_cand_pos, probe_cand_pos, static_cast<float>(probe_context_base),
                      lanes);
        AscendC::Sub(probe_diff, tile_max, run_max, lanes);
        AscendC::Compares(predicate, probe_diff, 0.0f, AscendC::CMPMODE::GT, lanes);
        AscendC::Select(probe_pos, predicate, probe_cand_pos, probe_pos,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, lanes);
        AscendC::Duplicate(probe_cand_page, static_cast<float>(probe_page_id), lanes);
        AscendC::Select(probe_page, predicate, probe_cand_page, probe_page,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, lanes);
#endif
    }

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

        if (valid < kCubeTileKeys) {
            const AscendC::LocalTensor<std::uint8_t> predicate =
                mask[kCubeTileKeys].template ReinterpretCast<std::uint8_t>();
            for (std::uint32_t head = 0; head < heads; ++head) {
                const std::uint32_t at = head * kCubeTileKeys;
                AscendC::Select(scores[at], predicate, scores[at], kTailDeadLogit,
                                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                                std::uint32_t{kCubeTileKeys});
            }
        }
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

    __aicore__ inline void quantize_probabilities()
    {
        AscendC::Mul(scores, scores, value_scale, std::uint64_t{kCubeTileKeys}, heads, by_vector);

        if (valid < kCubeTileKeys) {
            const AscendC::LocalTensor<std::uint8_t> predicate =
                mask[kCubeTileKeys].template ReinterpretCast<std::uint8_t>();
            for (std::uint32_t head = 0; head < heads; ++head) {
                const std::uint32_t at = head * kCubeTileKeys;
                AscendC::Select(scores[at], predicate, scores[at], 0.0f,
                                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                                std::uint32_t{kCubeTileKeys});
            }
        }
        reduce<AscendC::ReduceType::MAX>(part);
        AscendC::Duplicate(prob_scale, operand_max, heads * kStateLanes);
        AscendC::Adds(part, part, epsilon, heads * kStateLanes);
        AscendC::Div(prob_scale, prob_scale, part, heads * kStateLanes);
        AscendC::Mul(scores, scores, prob_scale, std::uint64_t{kCubeTileKeys}, heads, by_block);
        codec.template CastToOperand<Operand>(probabilities, scores, heads * kCubeTileKeys);
    }
};

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
}
#endif
