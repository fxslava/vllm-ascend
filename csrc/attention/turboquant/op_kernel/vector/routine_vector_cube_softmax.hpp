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

// DIRTY-DATA (D2): the finite sentinel written over a dead tail lane. It is FLT_MAX negative,
// so exp(sentinel - max) underflows to zero, and -- unlike the old Min against a ramp -- a
// SELECT writes it even when the lane's score is NaN, because it does not touch the score's
// value at all.
constexpr float kTailDeadLogit = -3.4028235e38f;

// The paired register VF owns exactly three rows per AIV.
constexpr std::uint32_t kRegisterSoftmaxHeads = 3;
static_assert(kCubeTileKeys * sizeof(float) == 256);
static_assert(kStateLanes * sizeof(float) == 32);

// Preserve the prepared FP32 -> FP8 numerator quantizer's operation order.
__simd_callee__ inline void quantize_native_probability_row(
    AscendC::Reg::RegTensor<float>& weighted_row, __ubuf__ float* scale_out,
    __ubuf__ fp8_e4m3fn_t* operand_out, AscendC::Reg::MaskReg& live_mask,
    float operand_max, float epsilon)
{
    namespace reg = AscendC::Reg;
    auto row_mask = reg::CreateMask<float, reg::MaskPattern::ALL>();
    auto state_mask = reg::CreateMask<float, reg::MaskPattern::VL8>();
    reg::RegTensor<float> zero, reduced, denominator, numerator, scale, broadcast;
    // Invalid values may already be NaN after value scaling. Select exact +0.
    reg::Duplicate<float>(zero, 0.0f);
    reg::Select<float>(weighted_row, weighted_row, zero, live_mask);
    reg::Reduce<reg::ReduceType::MAX>(reduced, weighted_row, row_mask);
    reg::Duplicate(denominator, reduced, state_mask);
    reg::Adds(denominator, denominator, epsilon, state_mask);
    reg::Duplicate<float>(numerator, operand_max);
    // The tensor Div default uses DivAlgo::INTRINSIC with ZEROING.
    reg::Div<float, reg::MaskMergeMode::ZEROING>(scale, numerator, denominator, state_mask);
    reg::StoreAlign(scale_out, scale, state_mask);
    reg::Duplicate(broadcast, scale, row_mask);
    reg::Mul(weighted_row, weighted_row, broadcast, row_mask);
    // Same SAT + RINT and FP32-lane packed store as AscendC::Cast.
    static constexpr reg::CastTrait cast_trait = {
        reg::RegLayout::ZERO, reg::SatMode::SAT, reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    reg::RegTensor<fp8_e4m3fn_t> packed;
    reg::Cast<fp8_e4m3fn_t, float, cast_trait>(packed, weighted_row, row_mask);
    reg::StoreAlign<fp8_e4m3fn_t, reg::StoreDist::DIST_PACK4_B32>(operand_out, packed, row_mask);
}

// Runs one online recurrence over two consecutive, independently staged
// score windows. The two PVs still consume distinct probability generations.
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
    // Match the two original FP32 multiply rounds and select away dirty tail NaNs.
    reg::Mul(row, row, key_scale, row_mask);
    reg::Mul(row, row, q_inv, row_mask);
    reg::Select<float>(row, row, dead, live_mask);
    reg::Reduce<reg::ReduceType::MAX>(reduced, row, row_mask);
    reg::Duplicate(tile_max, reduced, row_mask);
    reg::Max(next_max, run_max, tile_max, row_mask);
#if defined(ASCEND_TQ_DECODE_TELEMETRY) && ASCEND_TQ_DECODE_TELEMETRY
    // Match the earliest-lane and earlier-tile tie rules of argmax_probe().
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
    // Keep the separate FP32 multiply and add of the approved recurrence.
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


// The two Cube score windows are disjoint and already ready before the paired VF.
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
    // Keep one head's successor in flight. No next-head load is permitted at drain.
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

    // The attention telemetry probe (ASCEND_TQ_DECODE_TELEMETRY, turboquant_layout.h). The
    // caller carves this arena out of a buffer that is dead for the whole tile loop and fills
    // probe_pos/probe_page with the sentinel and probe_ramp with 0..63 before the first tile;
    // probe_context_base is the context index of this tile's first row and probe_page_id the
    // physical cache page it was read from. argmax_probe() runs between update_max_and_alpha
    // and exponentiate_scores (vector_online_softmax_probabilities_probed), the only point
    // where the finished logits, their per-head maxima and the PRE-commit running maxima are
    // all live. With the macro off the phase compiles to nothing and the operands are unread.
    AscendC::LocalTensor<float>   probe_pos, probe_page, probe_ramp;
    AscendC::LocalTensor<float>   probe_bc, probe_diff, probe_sel;
    AscendC::LocalTensor<float>   probe_stage, probe_cand_pos, probe_cand_page;
    AscendC::LocalTensor<float>   probe_pred;  // 32 bytes, reinterpreted as the predicate
    std::uint32_t                 probe_context_base = 0, probe_page_id = 0;

    // Per-head running argmax fold. A head "improved" this tile when tile_max > run_max
    // (strictly: ties keep the EARLIER position, within a tile by the MIN reduce over the
    // ascending index ramp and across tiles by the strict comparison). The candidate position
    // is the tile's context base plus the first lane whose logit equals the tile maximum --
    // score == tile_max exactly, because the maximum is one of the lanes and x - x is +0.
    // Everything is vector instructions over replicated head blocks; no scalar readback, no
    // events, no GM. probe_bc is reused for both per-head broadcasts; probe_diff holds the
    // row difference first and the wide improvement test after the loop.
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
        // DIRTY-DATA (D2): rows past `valid` in a partial last tile were read from the cache
        // page (the tail of the page, not the live context). Their key codes and scales are
        // dirty: 0xDEADBEEF bytes are finite garbage, but a poisoned fp32 scale is a NaN, and
        // Min's NaN behaviour is not specified. Overwrite every dead lane with a finite sentinel
        // through the lane predicate instead, so the MAX below and the online recurrence never
        // see the neighbor slots' data. Live lanes are copied through unchanged. `scores` holds
        // `heads` contiguous 64-lane rows and the 64-bit predicate is the same for each, so the
        // Select is per row: the first-n form only consumes one row's mask bits.
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

    // Denominator uses unscaled probabilities. Value scale belongs only in the numerator,
    // followed by the source's FP8 dynamic range map.
    __aicore__ inline void quantize_probabilities()
    {
        AscendC::Mul(scores, scores, value_scale, std::uint64_t{kCubeTileKeys}, heads, by_vector);
        // DIRTY-DATA (D2): the value scale of a dead tail row can be NaN (the scale plane past
        // `context_lens` is allocator garbage). exp(dead) is zero, but 0 * NaN is NaN, and the
        // MAX reduction below would then hand a NaN probability scale to every LIVE row of the
        // tile. SELECT the dead lanes to exactly zero after the multiply, so the reduction and
        // the operand cast only ever see the live rows. Per row, for the same reason as above.
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

// Scores of a partial last tile are gated by this ramp: a ramp crossing zero at `valid`, huge
// and positive before it and huge and negative after. The caller turns its sign into the bit
// predicate the softmax SELECTs against; the ramp itself is kept because a lane index ramp is
// what names the live rows, not the scores (a dirty score can be NaN). `lanes` aliases the mask
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
