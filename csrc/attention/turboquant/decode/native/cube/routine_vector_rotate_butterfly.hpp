#pragma once

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "kernel_operator.h"
#endif

namespace ascend_micro {

constexpr std::uint32_t kRotChannels = 256;
constexpr std::uint32_t kRotVecLanes = 64;
constexpr std::uint32_t kRotRegisters = kRotChannels / kRotVecLanes;
constexpr std::uint32_t kRotLanesPerBlock = 8;
constexpr std::uint32_t kRotStages = 8;
constexpr std::uint32_t kRotLaneStages = 6;
constexpr std::uint32_t kRotRegStages = 2;

constexpr float kRotOrthoScale = 1.0f / 16.0f;

enum class RotateDirection : unsigned {
    Egress = 0,
    Ingress = 1,
};

enum class RotateImpl : unsigned {
    Blocked = 0,
    RegGather = 1,
    RegShuffle = 2,
};

constexpr std::uint32_t kRotStrideOf[2][kRotStages] = {
    {32, 64, 128, 1, 2, 4, 8, 16},
    {1, 2, 4, 8, 16, 32, 64, 128},
};

constexpr std::uint32_t kRotSignAfterStage[2] = {3, 5};

constexpr std::uint32_t kRotGatherStages = 3;
constexpr std::uint32_t kRotGatherStrideOf[kRotGatherStages] = {1, 2, 4};

constexpr std::uint32_t kRotBlockedTableBytes =
    kRotGatherStages * kRotChannels * 4u * 2u;

constexpr unsigned RotDirIndex(RotateDirection d) { return static_cast<unsigned>(d); }

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)

template <std::uint32_t S>
constexpr std::uint32_t kRotLanesPerRepeat = S < kRotVecLanes ? S : kRotVecLanes;
template <std::uint32_t S>
constexpr std::uint32_t kRotRepeats = (kRotChannels / 2u) / kRotLanesPerRepeat<S>;
template <std::uint32_t S>
constexpr std::uint32_t kRotRepStrideBlocks =
    S <= kRotVecLanes ? (2u * S / kRotLanesPerBlock) : (kRotVecLanes / kRotLanesPerBlock);

struct NativeRotateBlockedEngine {

    AscendC::LocalTensor<float> pi_sign;

    AscendC::LocalTensor<std::uint32_t> gather_offsets;
    AscendC::LocalTensor<float> gather_signs;

    AscendC::LocalTensor<float> buffer_a;
    AscendC::LocalTensor<float> buffer_b;
    AscendC::LocalTensor<float> permuted;
    AscendC::LocalTensor<float> signed_tmp;

    static constexpr std::int32_t kChannelLanes = static_cast<std::int32_t>(kRotChannels);

    template <std::uint32_t S>
    __aicore__ inline void block_stage(const AscendC::LocalTensor<float>& dst,
                                       const AscendC::LocalTensor<float>& src) const
    {
        constexpr std::int32_t mask = static_cast<std::int32_t>(kRotLanesPerRepeat<S>);
        constexpr std::uint8_t repeats = static_cast<std::uint8_t>(kRotRepeats<S>);
        constexpr std::uint8_t stride = static_cast<std::uint8_t>(kRotRepStrideBlocks<S>);
        const AscendC::BinaryRepeatParams params(1, 1, 1, stride, stride, stride);
        AscendC::Add(dst, src, src[S], mask, repeats, params);
        AscendC::Sub(dst[S], src, src[S], mask, repeats, params);
    }

    template <std::uint32_t Index>
    __aicore__ inline void gather_stage(const AscendC::LocalTensor<float>& dst,
                                        const AscendC::LocalTensor<float>& src) const
    {
        constexpr std::uint32_t table = Index * kRotChannels;
        AscendC::Gather(permuted, src, gather_offsets[table], 0u, kRotChannels);
        AscendC::Mul(signed_tmp, src, gather_signs[table], kChannelLanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(dst, permuted, signed_tmp, kChannelLanes);
    }

    __aicore__ inline void sign_stage(const AscendC::LocalTensor<float>& dst,
                                      const AscendC::LocalTensor<float>& src) const
    {
        AscendC::Mul(dst, src, pi_sign, kChannelLanes);
    }

    template <RotateDirection D>
    __aicore__ inline void rotate(const AscendC::LocalTensor<float>& dst,
                                  const AscendC::LocalTensor<float>& src)
    {
        if constexpr (D == RotateDirection::Egress) {
            block_stage<32>(buffer_b, src);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<64>(buffer_a, buffer_b);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<128>(buffer_b, buffer_a);
            AscendC::PipeBarrier<PIPE_V>();
            sign_stage(buffer_a, buffer_b);
            AscendC::PipeBarrier<PIPE_V>();
            gather_stage<0>(buffer_b, buffer_a);
            AscendC::PipeBarrier<PIPE_V>();
            gather_stage<1>(buffer_a, buffer_b);
            AscendC::PipeBarrier<PIPE_V>();
            gather_stage<2>(buffer_b, buffer_a);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<8>(buffer_a, buffer_b);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<16>(dst, buffer_a);
        } else {
            gather_stage<0>(buffer_b, src);
            AscendC::PipeBarrier<PIPE_V>();
            gather_stage<1>(buffer_a, buffer_b);
            AscendC::PipeBarrier<PIPE_V>();
            gather_stage<2>(buffer_b, buffer_a);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<8>(buffer_a, buffer_b);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<16>(buffer_b, buffer_a);
            AscendC::PipeBarrier<PIPE_V>();
            sign_stage(buffer_a, buffer_b);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<32>(buffer_b, buffer_a);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<64>(buffer_a, buffer_b);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<128>(dst, buffer_a);
        }
    }

    __aicore__ inline void scale(const AscendC::LocalTensor<float>& dst, float factor)
    {
        AscendC::Muls(dst, dst, factor, kChannelLanes);
    }
};

struct RotRegSet {
    AscendC::Reg::RegTensor<float> v0;
    AscendC::Reg::RegTensor<float> v1;
    AscendC::Reg::RegTensor<float> v2;
    AscendC::Reg::RegTensor<float> v3;
};

__simd_callee__ inline void rot_reg_stage64(RotRegSet& v, AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<float> s0, d0, s1, d1;
    AscendC::Reg::Add<float>(s0, v.v0, v.v1, mask);
    AscendC::Reg::Sub<float>(d0, v.v0, v.v1, mask);
    AscendC::Reg::Add<float>(s1, v.v2, v.v3, mask);
    AscendC::Reg::Sub<float>(d1, v.v2, v.v3, mask);
    v.v0 = s0;
    v.v1 = d0;
    v.v2 = s1;
    v.v3 = d1;
}

__simd_callee__ inline void rot_reg_stage128(RotRegSet& v, AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<float> s0, d0, s1, d1;
    AscendC::Reg::Add<float>(s0, v.v0, v.v2, mask);
    AscendC::Reg::Sub<float>(d0, v.v0, v.v2, mask);
    AscendC::Reg::Add<float>(s1, v.v1, v.v3, mask);
    AscendC::Reg::Sub<float>(d1, v.v1, v.v3, mask);
    v.v0 = s0;
    v.v2 = d0;
    v.v1 = s1;
    v.v3 = d1;
}

struct RotGatherTables {
    AscendC::Reg::RegTensor<std::uint32_t> i0, i1, i2, i3, i4, i5;

    AscendC::Reg::RegTensor<std::uint32_t> s0, s1, s2, s3, s4, s5;
};

__simd_callee__ inline void rot_one_table(AscendC::Reg::RegTensor<std::uint32_t>& idx,
                                          AscendC::Reg::RegTensor<std::uint32_t>& sgn_bits,
                                          AscendC::Reg::RegTensor<std::uint32_t>& lane,
                                          AscendC::Reg::RegTensor<std::uint32_t>& one_bits,
                                          std::uint32_t k, AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<std::uint32_t> stride, bit, shifted;
    AscendC::Reg::Duplicate<std::uint32_t>(stride, 1u << k);
    AscendC::Reg::Xor<std::uint32_t>(idx, lane, stride, mask);
    AscendC::Reg::And<std::uint32_t>(bit, lane, stride, mask);

    AscendC::Reg::ShiftLefts<std::uint32_t>(shifted, bit,
                                            static_cast<std::int16_t>(31u - k), mask);
    AscendC::Reg::Or<std::uint32_t>(sgn_bits, shifted, one_bits, mask);
}

__simd_callee__ inline void rot_build_gather_tables(RotGatherTables& t,
                                                    AscendC::Reg::MaskReg& mask)
{

    AscendC::Reg::RegTensor<std::int32_t> ramp;
    AscendC::Reg::Arange<std::int32_t>(ramp, 0);
    auto& lane = reinterpret_cast<AscendC::Reg::RegTensor<std::uint32_t>&>(ramp);
    AscendC::Reg::RegTensor<std::uint32_t> one_bits;
    AscendC::Reg::Duplicate<std::uint32_t>(one_bits, 0x3F800000u);
    rot_one_table(t.i0, t.s0, lane, one_bits, 0u, mask);
    rot_one_table(t.i1, t.s1, lane, one_bits, 1u, mask);
    rot_one_table(t.i2, t.s2, lane, one_bits, 2u, mask);
    rot_one_table(t.i3, t.s3, lane, one_bits, 3u, mask);
    rot_one_table(t.i4, t.s4, lane, one_bits, 4u, mask);
    rot_one_table(t.i5, t.s5, lane, one_bits, 5u, mask);
}

__simd_callee__ inline void rot_gather_one(AscendC::Reg::RegTensor<float>& v,
                                           AscendC::Reg::RegTensor<std::uint32_t>& idx,
                                           AscendC::Reg::RegTensor<std::uint32_t>& sgn_bits,
                                           AscendC::Reg::MaskReg& mask)
{
    auto& sgn = reinterpret_cast<AscendC::Reg::RegTensor<float>&>(sgn_bits);
    AscendC::Reg::RegTensor<float> partner;
    AscendC::Reg::Gather<float, std::uint32_t>(partner, v, idx);
    AscendC::Reg::MulAddDst<float>(partner, v, sgn, mask);
    v = partner;
}

__simd_callee__ inline void rot_gather_stage(RotRegSet& v,
                                             AscendC::Reg::RegTensor<std::uint32_t>& idx,
                                             AscendC::Reg::RegTensor<std::uint32_t>& sgn,
                                             AscendC::Reg::MaskReg& mask)
{
    rot_gather_one(v.v0, idx, sgn, mask);
    rot_gather_one(v.v1, idx, sgn, mask);
    rot_gather_one(v.v2, idx, sgn, mask);
    rot_gather_one(v.v3, idx, sgn, mask);
}

__simd_callee__ inline void rot_shuffle_one(AscendC::Reg::RegTensor<float>& v,
                                            AscendC::Reg::MaskReg& mask,
                                            AscendC::Reg::MaskReg& half)
{
    AscendC::Reg::RegTensor<float> even, odd, sum, diff;
    AscendC::Reg::DeInterleave<float>(even, odd, v, v);
    AscendC::Reg::Add<float>(sum, even, odd, mask);
    AscendC::Reg::Sub<float>(diff, even, odd, mask);
    AscendC::Reg::Select<float>(v, sum, diff, half);
}

__simd_callee__ inline void rot_shuffle_stage(RotRegSet& v, AscendC::Reg::MaskReg& mask,
                                              AscendC::Reg::MaskReg& half)
{
    rot_shuffle_one(v.v0, mask, half);
    rot_shuffle_one(v.v1, mask, half);
    rot_shuffle_one(v.v2, mask, half);
    rot_shuffle_one(v.v3, mask, half);
}

__simd_callee__ inline void rot_shuffle_permute(AscendC::Reg::RegTensor<float>& v,
                                                AscendC::Reg::MaskReg& half)
{
    AscendC::Reg::RegTensor<float> even, odd;
    AscendC::Reg::DeInterleave<float>(even, odd, v, v);
    AscendC::Reg::Select<float>(v, even, odd, half);
}

__simd_callee__ inline void rot_load_row(RotRegSet& v, __ubuf__ float* in)
{
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(v.v0, in);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(v.v1, in + kRotVecLanes);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(v.v2,
                                                                      in + 2u * kRotVecLanes);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(v.v3,
                                                                      in + 3u * kRotVecLanes);
}

__simd_callee__ inline void rot_store_row(__ubuf__ float* out, RotRegSet& v,
                                          AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(out, v.v0, mask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(out + kRotVecLanes,
                                                                            v.v1, mask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
        out + 2u * kRotVecLanes, v.v2, mask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
        out + 3u * kRotVecLanes, v.v3, mask);
}

__simd_callee__ inline void rot_scale_row(RotRegSet& v, float factor,
                                          AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::Muls<float, float>(v.v0, v.v0, factor, mask);
    AscendC::Reg::Muls<float, float>(v.v1, v.v1, factor, mask);
    AscendC::Reg::Muls<float, float>(v.v2, v.v2, factor, mask);
    AscendC::Reg::Muls<float, float>(v.v3, v.v3, factor, mask);
}

struct RotSignRegs {
    AscendC::Reg::RegTensor<float> p0, p1, p2, p3;
};

__simd_callee__ inline void rot_load_signs(RotSignRegs& p, __ubuf__ float* sign)
{
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(p.p0, sign);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(p.p1,
                                                                      sign + kRotVecLanes);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
        p.p2, sign + 2u * kRotVecLanes);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
        p.p3, sign + 3u * kRotVecLanes);
}

__simd_callee__ inline void rot_apply_signs(RotRegSet& v, RotSignRegs& p,
                                            AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::Mul<float>(v.v0, v.v0, p.p0, mask);
    AscendC::Reg::Mul<float>(v.v1, v.v1, p.p1, mask);
    AscendC::Reg::Mul<float>(v.v2, v.v2, p.p2, mask);
    AscendC::Reg::Mul<float>(v.v3, v.v3, p.p3, mask);
}

template <RotateDirection D>
__simd_callee__ inline void rot_row_gather(RotRegSet& v, RotGatherTables& t, RotSignRegs& p,
                                           AscendC::Reg::MaskReg& mask)
{
    if constexpr (D == RotateDirection::Egress) {
        rot_gather_stage(v, t.i5, t.s5, mask);
        rot_reg_stage64(v, mask);
        rot_reg_stage128(v, mask);
        rot_apply_signs(v, p, mask);
        rot_gather_stage(v, t.i0, t.s0, mask);
        rot_gather_stage(v, t.i1, t.s1, mask);
        rot_gather_stage(v, t.i2, t.s2, mask);
        rot_gather_stage(v, t.i3, t.s3, mask);
        rot_gather_stage(v, t.i4, t.s4, mask);
    } else {
        rot_gather_stage(v, t.i0, t.s0, mask);
        rot_gather_stage(v, t.i1, t.s1, mask);
        rot_gather_stage(v, t.i2, t.s2, mask);
        rot_gather_stage(v, t.i3, t.s3, mask);
        rot_gather_stage(v, t.i4, t.s4, mask);
        rot_apply_signs(v, p, mask);
        rot_gather_stage(v, t.i5, t.s5, mask);
        rot_reg_stage64(v, mask);
        rot_reg_stage128(v, mask);
    }
}

__simd_callee__ inline void rot_row_shuffle(RotRegSet& v, RotSignRegs& p,
                                            AscendC::Reg::MaskReg& mask,
                                            AscendC::Reg::MaskReg& half)
{

    rot_shuffle_stage(v, mask, half);
    rot_shuffle_stage(v, mask, half);
    rot_shuffle_stage(v, mask, half);
    rot_shuffle_stage(v, mask, half);
    rot_shuffle_stage(v, mask, half);
    rot_apply_signs(v, p, mask);

    rot_shuffle_stage(v, mask, half);
    rot_reg_stage64(v, mask);
    rot_reg_stage128(v, mask);
}

__simd_callee__ inline void rot_permute_signs_p5(RotSignRegs& p, AscendC::Reg::MaskReg& half)
{
    for (std::uint32_t k = 0; k < 5u; ++k) {
        rot_shuffle_permute(p.p0, half);
        rot_shuffle_permute(p.p1, half);
        rot_shuffle_permute(p.p2, half);
        rot_shuffle_permute(p.p3, half);
    }
}

template <RotateDirection D>
__simd_vf__ inline void rotate_butterfly_gather_vf(__ubuf__ float* dst, __ubuf__ float* src,
                                                   __ubuf__ float* sign, float scale,
                                                   std::uint16_t rows)
{
    AscendC::Reg::MaskReg mask =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    RotGatherTables t;
    rot_build_gather_tables(t, mask);
    RotSignRegs p;
    rot_load_signs(p, sign);

    for (std::uint16_t r = 0; r < rows; ++r) {
        __ubuf__ float* in = src + static_cast<std::uint32_t>(r) * kRotChannels;
        __ubuf__ float* out = dst + static_cast<std::uint32_t>(r) * kRotChannels;
        RotRegSet v;
        rot_load_row(v, in);
        rot_row_gather<D>(v, t, p, mask);
        rot_scale_row(v, scale, mask);
        rot_store_row(out, v, mask);
    }
}

__simd_vf__ inline void rotate_butterfly_shuffle_vf(__ubuf__ float* dst, __ubuf__ float* src,
                                                    __ubuf__ float* sign, float scale,
                                                    std::uint16_t rows)
{
    AscendC::Reg::MaskReg mask =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg half = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::H>();
    RotSignRegs p;
    rot_load_signs(p, sign);
    rot_permute_signs_p5(p, half);

    for (std::uint16_t r = 0; r < rows; ++r) {
        __ubuf__ float* in = src + static_cast<std::uint32_t>(r) * kRotChannels;
        __ubuf__ float* out = dst + static_cast<std::uint32_t>(r) * kRotChannels;
        RotRegSet v;
        rot_load_row(v, in);
        rot_row_shuffle(v, p, mask, half);
        rot_scale_row(v, scale, mask);
        rot_store_row(out, v, mask);
    }
}

#endif

}
