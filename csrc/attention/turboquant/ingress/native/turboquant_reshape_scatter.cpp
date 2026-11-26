#include <cstdint>

#include "../../common/turboquant_layout.h"
#include "../../common/turboquant_mode.h"

#ifndef TURBOQUANT_INGRESS_NZ
#define TURBOQUANT_INGRESS_NZ 0
#endif

#if defined(__CCE_AICORE__)
#define TQ_SCATTER_AICORE __aicore__
#else
#define TQ_SCATTER_AICORE
#endif

namespace turboquant_scatter {

#if TURBOQUANT_INGRESS_NZ

constexpr std::uint32_t kNzTileRows = 64u;
constexpr std::uint32_t kNzC0 = 32u;
constexpr std::uint32_t kNzChannels = 256u;
constexpr std::uint32_t kNzRowBytes = kNzChannels / 2u;
constexpr std::uint32_t kNzGroups = kNzRowBytes / kNzC0;
constexpr std::uint32_t kNzGroupBytes = kNzTileRows * kNzC0;
constexpr std::uint32_t kNzHeadBytes = kNzTileRows * kNzRowBytes;

namespace tq_contract = vllm_ascend::turboquant;

TQ_SCATTER_AICORE constexpr std::uint32_t NzScaleSlot(std::uint32_t heads)
{
    return ((2u * heads + 7u) / 8u) * 8u;
}

TQ_SCATTER_AICORE constexpr std::uint64_t NzByte(std::uint32_t slot, std::uint32_t head,
                                                 std::uint32_t group, std::uint32_t row,
                                                 std::uint32_t byte, std::uint32_t heads)
{
    return static_cast<std::uint64_t>(slot - slot % kNzTileRows) * heads * kNzRowBytes +
           head * kNzHeadBytes + group * kNzGroupBytes + row * kNzC0 + byte;
}

TQ_SCATTER_AICORE constexpr bool NzAgrees(std::uint32_t heads, std::uint32_t slot)
{
    for (std::uint32_t head = 0; head < heads; ++head)
        for (std::uint32_t group = 0; group < kNzGroups; ++group)
            for (std::uint32_t byte = 0; byte < kNzC0; ++byte)
                if (NzByte(slot, head, group, slot % kNzTileRows, byte, heads) !=
                    tq_contract::NzTiledPackedByte(slot, head, group * kNzC0 + byte, heads,
                                                   kNzRowBytes))
                    return false;
    for (std::uint32_t kv = 0; kv < 2u; ++kv)
        for (std::uint32_t head = 0; head < heads; ++head)
            if (static_cast<std::uint64_t>(slot) * NzScaleSlot(heads) + kv * heads + head !=
                tq_contract::NzTiledScaleIndex(slot, head, kv, heads))
                return false;
    return true;
}

static_assert(NzAgrees(4u, 0u) && NzAgrees(4u, 31u) && NzAgrees(4u, 63u) && NzAgrees(4u, 64u) &&
                  NzAgrees(4u, 3u * 64u + 17u) && NzAgrees(1u, 7u) && NzAgrees(8u, 100u),
              "the scatter writer's NZ addressing disagrees with turboquant_layout.h");
static_assert(kNzTileRows == tq_contract::kCubeTileRows && kNzC0 == tq_contract::kOperandC0,
              "the tile geometry is the contract's");
#endif

}

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "routine_vector_rotate_butterfly.hpp"

namespace turboquant_scatter {

using ascend_micro::kRotChannels;

using LloydMaxKv4 =
    vllm_ascend::turboquant::TurboQuantLloydMaxTraits<vllm_ascend::turboquant::TurboQuantMode::KV4_FP8>;

constexpr std::uint32_t kHeadSize = kRotChannels;
constexpr std::uint32_t kStripHeight = 16;
constexpr std::uint32_t kMaxRun = kStripHeight;
constexpr std::uint32_t kStripBytes = kStripHeight * kHeadSize / 2u;
#if !TURBOQUANT_INGRESS_NZ
constexpr std::uint32_t kStripWords = kStripBytes / 4u;
#endif
constexpr std::uint32_t kKeyTokenBytes = kHeadSize / 2u;
constexpr std::uint32_t kVecLanes = ascend_micro::kRotVecLanes;

constexpr std::uint32_t kBlock = 32;
constexpr std::uint32_t kBankSkew = 32;
constexpr std::uint32_t kVecReg = 256;
constexpr std::uint32_t kUbWindow = 192u * 1024u;

constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}
constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkew;
}

constexpr std::uint32_t kPanelElems = kMaxRun * kHeadSize;
constexpr std::uint32_t kUbRaw = 0;
constexpr std::uint32_t kUbAct = NextSlot(kUbRaw, kPanelElems * 2u);
constexpr std::uint32_t kUbQ = NextSlot(kUbAct, kPanelElems * 4u);
constexpr std::uint32_t kUbQuant = NextSlot(kUbQ, kPanelElems * 4u);
constexpr std::uint32_t kUbHalf = NextSlot(kUbQuant, kPanelElems * 4u);
constexpr std::uint32_t kUbPacked = NextSlot(kUbHalf, kPanelElems * 2u);
#if TURBOQUANT_INGRESS_NZ

constexpr std::uint32_t kUbSign = NextSlot(kUbPacked, kMaxRun * kKeyTokenBytes);
#else
constexpr std::uint32_t kUbStrip = NextSlot(kUbPacked, kMaxRun * kKeyTokenBytes);
constexpr std::uint32_t kUbSign = NextSlot(kUbStrip, kStripBytes);
#endif

constexpr std::uint32_t kUbScale = NextSlot(kUbSign, kHeadSize * 4u);
constexpr std::uint32_t kArenaEnd = kUbScale + kMaxRun * kVecLanes * 4u;

static_assert(kUbRaw % kBlock == 0u && kUbAct % kBlock == 0u && kUbQ % kBlock == 0u &&
                  kUbQuant % kBlock == 0u && kUbHalf % kBlock == 0u &&
                  kUbPacked % kBlock == 0u && kUbSign % kBlock == 0u && kUbScale % kBlock == 0u,
              "every UB operand base must be 32-byte aligned");
#if !TURBOQUANT_INGRESS_NZ
static_assert(kUbStrip % kBlock == 0u, "the strip image must be 32-byte aligned");
#endif

static_assert(kArenaEnd + kVecReg <= kUbWindow, "UB arena leaves no read pad");

static_assert(kHeadSize * 4u % kVecReg == 0u, "a 256-channel FP32 row is whole registers");
static_assert(kVecLanes * 4u == kVecReg, "64 FP32 lanes is one vector register");

static_assert(kStripBytes % kBlock == 0u && kKeyTokenBytes % kBlock == 0u &&
                  kHeadSize * 2u % kBlock == 0u,
              "DMA lengths must be whole 32-byte bursts");

static_assert(kStripBytes % 4u == 0u, "the strip must be addressable as uint32");

constexpr std::int32_t kEvtRawReady = 0;
#if !TURBOQUANT_INGRESS_NZ
constexpr std::int32_t kEvtStripReady = 1;
#endif
constexpr std::int32_t kEvtRawFree = 0;
constexpr std::int32_t kEvtPackReady = 0;
#if !TURBOQUANT_INGRESS_NZ
constexpr std::int32_t kEvtStripDone = 1;
#endif
constexpr std::int32_t kEvtPackFree = 0;
#if !TURBOQUANT_INGRESS_NZ
constexpr std::int32_t kEvtStripFree = 0;
#endif
constexpr std::int32_t kEvtDrain = 0;

constexpr auto kUb = AscendC::TPosition::VECCALC;

template <class T>
__aicore__ inline AscendC::LocalTensor<T> View(std::uint32_t base, std::uint32_t elems)
{
    return AscendC::LocalTensor<T>{kUb, base, elems};
}

__simd_callee__ inline void absmax_broadcast(AscendC::Reg::RegTensor<float>& m,
                                             ascend_micro::RotRegSet& v,
                                             ascend_micro::RotGatherTables& t,
                                             AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<float> a0, a1, a2, a3, partner;
    AscendC::Reg::Abs<float>(a0, v.v0, mask);
    AscendC::Reg::Abs<float>(a1, v.v1, mask);
    AscendC::Reg::Abs<float>(a2, v.v2, mask);
    AscendC::Reg::Abs<float>(a3, v.v3, mask);
    AscendC::Reg::Max<float>(m, a0, a1, mask);
    AscendC::Reg::Max<float>(m, m, a2, mask);
    AscendC::Reg::Max<float>(m, m, a3, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i0);
    AscendC::Reg::Max<float>(m, m, partner, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i1);
    AscendC::Reg::Max<float>(m, m, partner, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i2);
    AscendC::Reg::Max<float>(m, m, partner, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i3);
    AscendC::Reg::Max<float>(m, m, partner, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i4);
    AscendC::Reg::Max<float>(m, m, partner, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i5);
    AscendC::Reg::Max<float>(m, m, partner, mask);
}

__simd_callee__ inline void sumsq_broadcast(AscendC::Reg::RegTensor<float>& m,
                                            ascend_micro::RotRegSet& v,
                                            ascend_micro::RotGatherTables& t,
                                            AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<float> sq, partner;
    AscendC::Reg::Mul<float>(m, v.v0, v.v0, mask);
    AscendC::Reg::Mul<float>(sq, v.v1, v.v1, mask);
    AscendC::Reg::Add<float>(m, m, sq, mask);
    AscendC::Reg::Mul<float>(sq, v.v2, v.v2, mask);
    AscendC::Reg::Add<float>(m, m, sq, mask);
    AscendC::Reg::Mul<float>(sq, v.v3, v.v3, mask);
    AscendC::Reg::Add<float>(m, m, sq, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i0);
    AscendC::Reg::Add<float>(m, m, partner, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i1);
    AscendC::Reg::Add<float>(m, m, partner, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i2);
    AscendC::Reg::Add<float>(m, m, partner, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i3);
    AscendC::Reg::Add<float>(m, m, partner, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i4);
    AscendC::Reg::Add<float>(m, m, partner, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i5);
    AscendC::Reg::Add<float>(m, m, partner, mask);
}

template <bool LLOYD_MAX>
__simd_vf__ inline void scatter_rotate_quantise_vf(__ubuf__ float* qout,
                                                   __ubuf__ float* scaleOut,
                                                   __ubuf__ float* src,
                                                   __ubuf__ float* sign, float inv_sqrt_len,
                                                   std::uint16_t rows)
{
    AscendC::Reg::MaskReg mask =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg lane0 =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
    ascend_micro::RotGatherTables t;
    ascend_micro::rot_build_gather_tables(t, mask);
    ascend_micro::RotSignRegs p;
    ascend_micro::rot_load_signs(p, sign);

    AscendC::Reg::RegTensor<std::uint32_t> exp_mask, two_pow, inv_seed;
    AscendC::Reg::RegTensor<float> one, inv_sqrt, zero;
    if constexpr (!LLOYD_MAX) {
        AscendC::Reg::Duplicate<std::uint32_t>(exp_mask, 0x7F800000u);
        AscendC::Reg::Duplicate<std::uint32_t>(two_pow, 0x01000000u);
        AscendC::Reg::Duplicate<std::uint32_t>(inv_seed, 0x7F000000u);
        AscendC::Reg::Duplicate<float>(one, 1.0f);
        AscendC::Reg::Duplicate<float>(inv_sqrt, inv_sqrt_len);
        AscendC::Reg::Duplicate<float>(zero, 0.0f);
    } else {

        static_cast<void>(exp_mask);
        static_cast<void>(two_pow);
        static_cast<void>(inv_seed);
        static_cast<void>(one);
        static_cast<void>(inv_sqrt);
        static_cast<void>(zero);
        static_cast<void>(inv_sqrt_len);
    }

    for (std::uint16_t r = 0; r < rows; ++r) {
        const std::uint32_t row = static_cast<std::uint32_t>(r);
        ascend_micro::RotRegSet v;
        ascend_micro::rot_load_row(v, src + row * kHeadSize);
        ascend_micro::rot_row_gather<ascend_micro::RotateDirection::Ingress>(v, t, p, mask);

        AscendC::Reg::RegTensor<float> absmax;
        AscendC::Reg::RegTensor<std::uint32_t> exp_bits, pow_bits, inv_bits;
        if constexpr (LLOYD_MAX) {

            sumsq_broadcast(absmax, v, t, mask);
        } else {
            absmax_broadcast(absmax, v, t, mask);

            auto& abs_bits = reinterpret_cast<AscendC::Reg::RegTensor<std::uint32_t>&>(absmax);
            AscendC::Reg::And<std::uint32_t>(exp_bits, abs_bits, exp_mask, mask);
            AscendC::Reg::Sub<std::uint32_t>(pow_bits, exp_bits, two_pow, mask);
            AscendC::Reg::Sub<std::uint32_t>(inv_bits, inv_seed, pow_bits, mask);
            auto& inv = reinterpret_cast<AscendC::Reg::RegTensor<float>&>(inv_bits);

            AscendC::Reg::Mul<float>(v.v0, v.v0, inv, mask);
            AscendC::Reg::Mul<float>(v.v1, v.v1, inv, mask);
            AscendC::Reg::Mul<float>(v.v2, v.v2, inv, mask);
            AscendC::Reg::Mul<float>(v.v3, v.v3, inv, mask);
            AscendC::Reg::Mins<float, float>(v.v0, v.v0, 7.0f, mask);
            AscendC::Reg::Mins<float, float>(v.v1, v.v1, 7.0f, mask);
            AscendC::Reg::Mins<float, float>(v.v2, v.v2, 7.0f, mask);
            AscendC::Reg::Mins<float, float>(v.v3, v.v3, 7.0f, mask);
        }
#if TURBOQUANT_INGRESS_NZ

        AscendC::Reg::RegTensor<float> w0, w1, w2, w3;
        AscendC::Reg::Interleave<float>(w0, w1, v.v0, v.v2);
        AscendC::Reg::Interleave<float>(w2, w3, v.v1, v.v3);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            qout + row * kHeadSize, w0, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            qout + row * kHeadSize + kVecLanes, w1, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            qout + row * kHeadSize + 2u * kVecLanes, w2, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            qout + row * kHeadSize + 3u * kVecLanes, w3, mask);
#else
        ascend_micro::rot_store_row(qout + row * kHeadSize, v, mask);
#endif

        if constexpr (LLOYD_MAX) {
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                scaleOut + row * kVecLanes, absmax, lane0);
        } else {
            auto& pow_two = reinterpret_cast<AscendC::Reg::RegTensor<float>&>(pow_bits);
            AscendC::Reg::RegTensor<float> scaled, scale;
            AscendC::Reg::Mul<float>(scaled, pow_two, inv_sqrt, mask);
            AscendC::Reg::MaskReg zero_row;
            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(zero_row, absmax, zero, mask);
            AscendC::Reg::Select<float>(scale, one, scaled, zero_row);
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                scaleOut + row * kVecLanes, scale, lane0);
        }
    }
}

#if !TURBOQUANT_INGRESS_NZ

template <std::uint32_t Half>
__simd_vf__ inline void scatter_merge_token_vf(__ubuf__ std::uint32_t* strip,
                                               __ubuf__ std::int32_t* codes,
                                               std::int16_t shift)
{
    AscendC::Reg::MaskReg mask =
        AscendC::Reg::CreateMask<std::uint32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<std::uint32_t> zero, fifteen, ones, nib;
    AscendC::Reg::Duplicate<std::uint32_t>(zero, 0u);
    AscendC::Reg::Duplicate<std::uint32_t>(fifteen, 0xFu);
    AscendC::Reg::Duplicate<std::uint32_t>(ones, 0xFFFFFFFFu);
    AscendC::Reg::Duplicate<std::uint32_t>(
        nib, 0xFu << static_cast<std::uint32_t>(shift));

    AscendC::Reg::RegTensor<std::uint32_t> keep, keep_hi, drop;
    if constexpr (Half == 0u) {
        AscendC::Reg::Interleave<std::uint32_t>(keep, keep_hi, nib, zero);
    } else {
        AscendC::Reg::Interleave<std::uint32_t>(keep, keep_hi, zero, nib);
    }

    AscendC::Reg::Xor<std::uint32_t>(drop, keep, ones, mask);

    for (std::uint32_t r = 0; r < ascend_micro::kRotRegisters; ++r) {
        AscendC::Reg::RegTensor<std::int32_t> ci;
        AscendC::Reg::LoadAlign<std::int32_t, AscendC::Reg::LoadDist::DIST_NORM>(
            ci, codes + r * kVecLanes);
        auto& cu = reinterpret_cast<AscendC::Reg::RegTensor<std::uint32_t>&>(ci);
        AscendC::Reg::RegTensor<std::uint32_t> nibs, placed, e0, e1, s0, s1;
        AscendC::Reg::And<std::uint32_t>(nibs, cu, fifteen, mask);
        AscendC::Reg::ShiftLefts<std::uint32_t>(placed, nibs, shift, mask);
        AscendC::Reg::Interleave<std::uint32_t>(e0, e1, placed, placed);
        AscendC::Reg::And<std::uint32_t>(e0, e0, keep, mask);
        AscendC::Reg::And<std::uint32_t>(e1, e1, keep, mask);
        __ubuf__ std::uint32_t* w0 = strip + (2u * r) * kVecLanes;
        __ubuf__ std::uint32_t* w1 = strip + (2u * r + 1u) * kVecLanes;
        AscendC::Reg::LoadAlign<std::uint32_t, AscendC::Reg::LoadDist::DIST_NORM>(s0, w0);
        AscendC::Reg::LoadAlign<std::uint32_t, AscendC::Reg::LoadDist::DIST_NORM>(s1, w1);
        AscendC::Reg::And<std::uint32_t>(s0, s0, drop, mask);
        AscendC::Reg::And<std::uint32_t>(s1, s1, drop, mask);
        AscendC::Reg::Or<std::uint32_t>(s0, s0, e0, mask);
        AscendC::Reg::Or<std::uint32_t>(s1, s1, e1, mask);
        AscendC::Reg::StoreAlign<std::uint32_t, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            w0, s0, mask);
        AscendC::Reg::StoreAlign<std::uint32_t, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            w1, s1, mask);
    }
}
#endif

struct Geometry {
    std::uint32_t num_kv_heads;
    std::uint32_t head_size;
    std::uint32_t block_size;
    std::uint32_t num_blocks;

    std::uint32_t quant_mode;
    float inv_sqrt_len;
};

struct Run {
    std::uint32_t token_begin;
    std::uint32_t run;
    std::uint32_t head;
    std::uint32_t block;
    std::uint32_t token_in_block;
#if !TURBOQUANT_INGRESS_NZ
    std::uint32_t strip;
#endif
};

#if TURBOQUANT_INGRESS_NZ

__aicore__ inline std::uint64_t NzPlaneBase(const Run& item, const Geometry& geom)
{
    const std::uint64_t slot = static_cast<std::uint64_t>(item.block) * geom.block_size + item.token_in_block;
    return (slot - slot % kNzTileRows) * geom.num_kv_heads * kNzRowBytes +
           static_cast<std::uint64_t>(item.head) * kNzHeadBytes + (slot % kNzTileRows) * kNzC0;
}
#else

__aicore__ inline std::uint64_t StripWordOffset(const Run& item, const Geometry& geom)
{
    const std::uint64_t nibble =
        (static_cast<std::uint64_t>(item.block) * geom.num_kv_heads + item.head) *
            geom.block_size * geom.head_size +
        static_cast<std::uint64_t>(item.strip) * kStripHeight * geom.head_size;
    return nibble / 8u;
}
#endif

__aicore__ inline void StageRows(const AscendC::GlobalTensor<bfloat16_t>& plane,
                                 const Run& item, const Geometry& geom)
{
    AscendC::DataCopyParams rows;
    rows.blockCount = static_cast<std::uint16_t>(item.run);
    rows.blockLen = static_cast<std::uint16_t>(geom.head_size * 2u / kBlock);
    rows.srcStride =
        static_cast<std::uint16_t>((geom.num_kv_heads - 1u) * geom.head_size * 2u / kBlock);
    rows.dstStride = 0;
    const std::uint64_t base =
        (static_cast<std::uint64_t>(item.token_begin) * geom.num_kv_heads + item.head) *
        geom.head_size;
    AscendC::DataCopy(View<bfloat16_t>(kUbRaw, kMaxRun * kHeadSize), plane[base], rows);
}

__aicore__ inline void LloydMaxQuantise(const Run& item, const Geometry& geom)
{
    constexpr float kLmEps = 1e-20f;
    static constexpr AscendC::SqrtConfig kSqrt0 = {AscendC::SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};
    static constexpr AscendC::DivConfig kDiv0 = {AscendC::DivAlgo::DIFF_COMPENSATION};
    const std::uint32_t elems = item.run * kHeadSize;

    const auto scaleSlots = View<float>(kUbScale, kMaxRun * kVecLanes);
    const auto sigma      = View<float>(kUbScale + 768u * 4u, 16u);
    const auto invn       = View<float>(kUbScale + 784u * 4u, 16u);
    const auto bcast      = View<float>(kUbScale + 800u * 4u, 128u);
    const auto invRow     = View<float>(kUbScale + 928u * 4u, 64u);
    const auto offsets    = View<std::int32_t>(kUbScale + 992u * 4u, 16u);
    const auto tmp        = View<float>(kUbHalf, kPanelElems / 2u);
    const auto codes      = View<float>(kUbQ, kPanelElems);
    const auto bins       = View<std::int32_t>(kUbQuant, kPanelElems);

    AscendC::ArithProgression(offsets, 0, static_cast<std::int32_t>(kVecLanes * 4u), 16);
    AscendC::Gather(sigma, scaleSlots, offsets.ReinterpretCast<std::uint32_t>(), 0u, 16);
    AscendC::Adds(sigma, sigma, kLmEps, 16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sqrt<float, kSqrt0>(sigma, sigma, 16);
    AscendC::Muls(sigma, sigma, geom.inv_sqrt_len, 16);
    AscendC::Duplicate(invn, -1.0f, 16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div<float, kDiv0>(invn, invn, sigma, 16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Brcb(bcast, invn, 2, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::BinaryRepeatParams rowMul;
    rowMul.dstBlkStride = 1; rowMul.src0BlkStride = 1; rowMul.src1BlkStride = 1;
    rowMul.dstRepStride = 8; rowMul.src0RepStride = 8; rowMul.src1RepStride = 0;
    for (std::uint32_t r = 0; r < item.run; ++r) {
        AscendC::Brcb(invRow, bcast[r * 8u], 1, {1, 8});
        AscendC::Mul(codes[r * kHeadSize], codes[r * kHeadSize], invRow, 64, 4, rowMul);
    }
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Duplicate(bins, -8, elems);
    AscendC::PipeBarrier<PIPE_V>();
    for (std::uint32_t base = 0; base < elems; base += kPanelElems / 2u) {
        const std::uint32_t count =
            (elems - base) < (kPanelElems / 2u) ? (elems - base) : (kPanelElems / 2u);
        for (int i = 0; i < LloydMaxKv4::kThresholdCount; ++i) {
            AscendC::Adds(tmp, codes[base], LloydMaxKv4::kThresholds[i], count);

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftRight(tmp.ReinterpretCast<std::uint32_t>(),
                                tmp.ReinterpretCast<std::uint32_t>(), 31u,
                                static_cast<std::int32_t>(count));

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(bins[base], bins[base], tmp.ReinterpretCast<std::int32_t>(), count);

            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    AscendC::Muls(sigma, sigma, geom.inv_sqrt_len, 16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Brcb(scaleSlots, sigma, 2, {8, 64});
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void RotateQuantise(const Run& item, const Geometry& geom)
{
    const std::uint32_t elems = item.run * geom.head_size;
    AscendC::Cast(View<float>(kUbAct, kPanelElems), View<bfloat16_t>(kUbRaw, kPanelElems),
                  AscendC::RoundMode::CAST_NONE, static_cast<std::int32_t>(elems));
    AscendC::PipeBarrier<PIPE_V>();
    if (geom.quant_mode == 0u) {
        scatter_rotate_quantise_vf<false>(
            reinterpret_cast<__ubuf__ float*>(View<float>(kUbQ, kPanelElems).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(
                View<float>(kUbScale, kMaxRun * kVecLanes).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(View<float>(kUbAct, kPanelElems).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(View<float>(kUbSign, kHeadSize).GetPhyAddr()),
            geom.inv_sqrt_len, static_cast<std::uint16_t>(item.run));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Cast(View<std::int32_t>(kUbQuant, kPanelElems), View<float>(kUbQ, kPanelElems),
                      AscendC::RoundMode::CAST_RINT, static_cast<std::int32_t>(elems));
    } else {
        scatter_rotate_quantise_vf<true>(
            reinterpret_cast<__ubuf__ float*>(View<float>(kUbQ, kPanelElems).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(
                View<float>(kUbScale, kMaxRun * kVecLanes).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(View<float>(kUbAct, kPanelElems).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(View<float>(kUbSign, kHeadSize).GetPhyAddr()),
            geom.inv_sqrt_len, static_cast<std::uint16_t>(item.run));
        AscendC::PipeBarrier<PIPE_V>();

        LloydMaxQuantise(item, geom);
    }
}

#if TURBOQUANT_INGRESS_NZ

__aicore__ inline void WriteNzPlane(const AscendC::GlobalTensor<std::int8_t>& cache,
                                    const AscendC::GlobalTensor<float>& scale_cache,
                                    const Run& item, const Geometry& geom, std::uint32_t kv)
{
    const std::uint32_t elems = item.run * geom.head_size;

    AscendC::Cast(View<half>(kUbHalf, kPanelElems), View<std::int32_t>(kUbQuant, kPanelElems),
                  AscendC::RoundMode::CAST_RINT, static_cast<std::int32_t>(elems));
    AscendC::PipeBarrier<PIPE_V>();
    auto packed = View<std::int8_t>(kUbPacked, kMaxRun * kKeyTokenBytes);
    auto nibbles = packed.ReinterpretCast<AscendC::int4b_t>();
    AscendC::Cast(nibbles, View<half>(kUbHalf, kPanelElems), AscendC::RoundMode::CAST_RINT,
                  static_cast<std::int32_t>(elems));

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtPackReady);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtPackReady);

    const std::uint64_t page = NzPlaneBase(item, geom);
    AscendC::DataCopyExtParams groups;
    groups.blockCount = static_cast<std::uint16_t>(item.run);
    groups.blockLen = kNzC0;
    groups.srcStride = (kNzRowBytes - kNzC0) / kBlock;
    groups.dstStride = 0;
    groups.rsv = 0;
    for (std::uint32_t g = 0; g < kNzGroups; ++g)
        AscendC::DataCopyPad(cache[page + g * kNzGroupBytes], packed[g * kNzC0], groups);

    const std::uint32_t pitch = NzScaleSlot(geom.num_kv_heads);
    const std::uint64_t slot = static_cast<std::uint64_t>(item.block) * geom.block_size + item.token_in_block;
    const std::uint64_t scale_base =
        slot * pitch + static_cast<std::uint64_t>(kv) * geom.num_kv_heads + item.head;
    AscendC::DataCopyExtParams lanes;
    lanes.blockCount = static_cast<std::uint16_t>(item.run);
    lanes.blockLen = 4u;
    lanes.srcStride = kVecLanes * 4u / kBlock - 1u;
    lanes.dstStride = (pitch - 1u) * 4u;
    lanes.rsv = 0;
    AscendC::DataCopyPad(scale_cache[scale_base], View<float>(kUbScale, 1u), lanes);
}
#else
__aicore__ inline void WriteKeyPlane(const AscendC::GlobalTensor<std::int8_t>& key_cache,
                                     const AscendC::GlobalTensor<float>& scale_cache,
                                     const Run& item, const Geometry& geom)
{
    const std::uint32_t elems = item.run * geom.head_size;

    AscendC::Cast(View<half>(kUbHalf, kPanelElems), View<std::int32_t>(kUbQuant, kPanelElems),
                  AscendC::RoundMode::CAST_RINT, static_cast<std::int32_t>(elems));
    AscendC::PipeBarrier<PIPE_V>();
    auto packed = View<std::int8_t>(kUbPacked, kMaxRun * kKeyTokenBytes);
    auto nibbles = packed.ReinterpretCast<AscendC::int4b_t>();
    AscendC::Cast(nibbles, View<half>(kUbHalf, kPanelElems), AscendC::RoundMode::CAST_RINT,
                  static_cast<std::int32_t>(elems));

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtPackReady);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtPackReady);

    const std::uint64_t key_base =
        ((static_cast<std::uint64_t>(item.block) * geom.num_kv_heads + item.head) *
             geom.block_size +
         item.token_in_block) *
        (geom.head_size / 2u);
    AscendC::DataCopy(key_cache[key_base], packed, item.run * kKeyTokenBytes);

    const std::uint64_t scale_base =
        ((static_cast<std::uint64_t>(0u) * geom.num_blocks + item.block) * geom.num_kv_heads +
         item.head) *
            geom.block_size +
        item.token_in_block;
    AscendC::DataCopyExtParams one_float;
    one_float.blockCount = 1;
    one_float.blockLen = 4u;
    one_float.srcStride = 0;
    one_float.dstStride = 0;
    one_float.rsv = 0;
    for (std::uint32_t r = 0; r < item.run; ++r)
        AscendC::DataCopyPad(scale_cache[scale_base + r],
                             View<float>(kUbScale + r * kVecLanes * 4u, 1u), one_float);
}

__aicore__ inline void WriteValuePlane(
    const AscendC::GlobalTensor<std::uint32_t>& value_words,
    const AscendC::GlobalTensor<float>& scale_cache, const Run& item, const Geometry& geom)
{

    for (std::uint32_t r = 0; r < item.run; ++r) {
        const std::uint32_t token = (item.token_in_block + r) % kStripHeight;
        const auto shift = static_cast<std::int16_t>(4u * (token % 8u));
        __ubuf__ std::uint32_t* strip = reinterpret_cast<__ubuf__ std::uint32_t*>(
            View<std::uint32_t>(kUbStrip, kStripWords).GetPhyAddr());
        __ubuf__ std::int32_t* codes = reinterpret_cast<__ubuf__ std::int32_t*>(
            View<std::int32_t>(kUbQuant + r * kHeadSize * 4u, kHeadSize).GetPhyAddr());
        if (token < 8u)
            scatter_merge_token_vf<0u>(strip, codes, shift);
        else
            scatter_merge_token_vf<1u>(strip, codes, shift);
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtStripDone);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtStripDone);

    AscendC::DataCopy(value_words[StripWordOffset(item, geom)],
                      View<std::uint32_t>(kUbStrip, kStripWords), kStripWords);

    const std::uint64_t scale_base =
        ((static_cast<std::uint64_t>(1u) * geom.num_blocks + item.block) * geom.num_kv_heads +
         item.head) *
            geom.block_size +
        item.token_in_block;
    AscendC::DataCopyExtParams one_float;
    one_float.blockCount = 1;
    one_float.blockLen = 4u;
    one_float.srcStride = 0;
    one_float.dstStride = 0;
    one_float.rsv = 0;
    for (std::uint32_t r = 0; r < item.run; ++r)
        AscendC::DataCopyPad(scale_cache[scale_base + r],
                             View<float>(kUbScale + r * kVecLanes * 4u, 1u), one_float);
}
#endif

#if TURBOQUANT_INGRESS_NZ

__aicore__ inline void ProcessNzPlane(const AscendC::GlobalTensor<bfloat16_t>& plane,
                                      const AscendC::GlobalTensor<std::int8_t>& cache,
                                      const AscendC::GlobalTensor<float>& scale_cache,
                                      const Run& item, const Geometry& geom, std::uint32_t kv)
{
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtRawFree);
    StageRows(plane, item, geom);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtRawReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtRawReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtPackFree);
    RotateQuantise(item, geom);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtRawFree);
    AscendC::PipeBarrier<PIPE_V>();
    WriteNzPlane(cache, scale_cache, item, geom, kv);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtPackFree);
}

__aicore__ inline void ProcessRun(const AscendC::GlobalTensor<bfloat16_t>& key,
                                  const AscendC::GlobalTensor<bfloat16_t>& value,
                                  const AscendC::GlobalTensor<std::int8_t>& key_cache,
                                  const AscendC::GlobalTensor<std::int8_t>& value_cache,
                                  const AscendC::GlobalTensor<float>& scale_cache,
                                  const Run& item, const Geometry& geom)
{
    ProcessNzPlane(key, key_cache, scale_cache, item, geom, 0u);
    ProcessNzPlane(value, value_cache, scale_cache, item, geom, 1u);
}
#else
__aicore__ inline void ProcessRun(const AscendC::GlobalTensor<bfloat16_t>& key,
                                  const AscendC::GlobalTensor<bfloat16_t>& value,
                                  const AscendC::GlobalTensor<std::int8_t>& key_cache,
                                  const AscendC::GlobalTensor<std::uint32_t>& value_words,
                                  const AscendC::GlobalTensor<float>& scale_cache,
                                  const Run& item, const Geometry& geom)
{

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtRawFree);
    StageRows(key, item, geom);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtRawReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtRawReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtPackFree);
    RotateQuantise(item, geom);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtRawFree);
    AscendC::PipeBarrier<PIPE_V>();
    WriteKeyPlane(key_cache, scale_cache, item, geom);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtPackFree);

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtRawFree);
    StageRows(value, item, geom);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtRawReady);

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kEvtStripFree);
    AscendC::DataCopy(View<std::uint32_t>(kUbStrip, kStripWords),
                      value_words[StripWordOffset(item, geom)], kStripWords);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtStripReady);

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtRawReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtStripReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtPackFree);
    RotateQuantise(item, geom);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtRawFree);
    AscendC::PipeBarrier<PIPE_V>();
    WriteValuePlane(value_words, scale_cache, item, geom);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kEvtStripFree);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtPackFree);
}
#endif

__aicore__ inline void Run_(__gm__ bfloat16_t* key_gm, __gm__ bfloat16_t* value_gm,
                            __gm__ std::int8_t* key_cache_gm,
                            __gm__ std::int8_t* value_cache_gm, __gm__ float* scale_cache_gm,
                            __gm__ std::int32_t* slot_gm, __gm__ float* pi_signs_gm,
                            __gm__ std::int32_t* cu_seqlens_gm, std::uint32_t num_requests,
                            std::uint32_t num_kv_heads, std::uint32_t head_size,
                            std::uint32_t block_size,
                            std::uint32_t num_blocks, std::uint32_t item_begin,
                            std::uint32_t item_count, float inv_sqrt_len, std::uint32_t quant_mode)
{

    if (head_size != kHeadSize) return;
#if TURBOQUANT_INGRESS_NZ

    if (block_size % kNzTileRows != 0u) return;
#else
    if (block_size % kStripHeight != 0u) return;
#endif

    if (quant_mode > 1u) return;

    Geometry geom;
    geom.num_kv_heads = num_kv_heads;
    geom.head_size = head_size;
    geom.block_size = block_size;
    geom.num_blocks = num_blocks;
    geom.quant_mode = quant_mode;
    geom.inv_sqrt_len = inv_sqrt_len;

    AscendC::GlobalTensor<bfloat16_t> key, value;
    AscendC::GlobalTensor<std::int8_t> key_cache;
#if TURBOQUANT_INGRESS_NZ
    AscendC::GlobalTensor<std::int8_t> value_cache;
#else
    AscendC::GlobalTensor<std::uint32_t> value_words;
#endif
    AscendC::GlobalTensor<float> scale_cache, pi_signs;
    AscendC::GlobalTensor<std::int32_t> slots, cu_seqlens;
    key.SetGlobalBuffer(key_gm);
    value.SetGlobalBuffer(value_gm);
    key_cache.SetGlobalBuffer(key_cache_gm);
#if TURBOQUANT_INGRESS_NZ
    value_cache.SetGlobalBuffer(value_cache_gm);
#else
    value_words.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint32_t*>(value_cache_gm));
#endif
    scale_cache.SetGlobalBuffer(scale_cache_gm);
    pi_signs.SetGlobalBuffer(pi_signs_gm);
    slots.SetGlobalBuffer(slot_gm);
    if (cu_seqlens_gm != nullptr) cu_seqlens.SetGlobalBuffer(cu_seqlens_gm);

    const std::uint32_t block_idx = static_cast<std::uint32_t>(AscendC::GetBlockIdx());
    const std::uint32_t block_num = static_cast<std::uint32_t>(AscendC::GetBlockNum());

    AscendC::DataCopy(View<float>(kUbSign, head_size), pi_signs[0], head_size);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtRawReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtRawReady);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtRawFree);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtPackFree);
#if !TURBOQUANT_INGRESS_NZ
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kEvtStripFree);
#endif

    const std::uint64_t item_end = static_cast<std::uint64_t>(item_begin) + item_count;
    for (std::uint64_t item = static_cast<std::uint64_t>(item_begin) + block_idx;
         item < item_end; item += block_num) {
        const auto index = static_cast<std::uint32_t>(item);
        const std::uint32_t request = index / num_kv_heads;
        const std::uint32_t head = index % num_kv_heads;
        if (request >= num_requests) continue;
        std::uint32_t token = cu_seqlens_gm != nullptr
                                  ? static_cast<std::uint32_t>(cu_seqlens.GetValue(request))
                                  : request;
        const std::uint32_t token_end =
            cu_seqlens_gm != nullptr
                ? static_cast<std::uint32_t>(cu_seqlens.GetValue(request + 1u))
                : request + 1u;

        while (token < token_end) {
            const std::int32_t slot0 = slots.GetValue(token);
            if (slot0 < 0) {
                ++token;
                continue;
            }
            Run active;
            active.token_begin = token;
            active.head = head;
            active.block = static_cast<std::uint32_t>(slot0) / block_size;
            active.token_in_block = static_cast<std::uint32_t>(slot0) % block_size;
#if !TURBOQUANT_INGRESS_NZ
            active.strip = active.token_in_block / kStripHeight;
#endif
            active.run = 1u;

            while (active.token_begin + active.run < token_end && active.run < kMaxRun) {
                const std::int32_t next = slots.GetValue(active.token_begin + active.run);
                if (next != static_cast<std::int64_t>(slot0) + active.run) break;
                const std::uint32_t tib = static_cast<std::uint32_t>(next) % block_size;
                if (static_cast<std::uint32_t>(next) / block_size != active.block) break;
#if TURBOQUANT_INGRESS_NZ
                if (tib / kNzTileRows != active.token_in_block / kNzTileRows) break;
#else
                if (tib / kStripHeight != active.strip) break;
#endif
                ++active.run;
            }
#if TURBOQUANT_INGRESS_NZ
            ProcessRun(key, value, key_cache, value_cache, scale_cache, active, geom);
#else
            ProcessRun(key, value, key_cache, value_words, scale_cache, active, geom);
#endif
            token = active.token_begin + active.run;
        }
    }

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtRawFree);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtPackFree);
#if !TURBOQUANT_INGRESS_NZ
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kEvtStripFree);
#endif
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
}

}

#endif

#if defined(__CCE_AICORE__) || \
    (defined(ASCEND_MICRO_CANN_LAUNCH) && !defined(TURBOQUANT_LEGACY_CANN_LAUNCH))
#if !defined(__CCE_AICORE__)

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif
extern "C" __global__ __aicore__ void turboquant_reshape_scatter_aiv(
    __gm__ bfloat16_t* __restrict__ key, __gm__ bfloat16_t* __restrict__ value,
    __gm__ std::int8_t* __restrict__ keyCache, __gm__ std::int8_t* __restrict__ valueCache,
    __gm__ float* __restrict__ scaleCache, __gm__ std::int32_t* __restrict__ slotMapping,
    __gm__ float* __restrict__ piSigns, __gm__ std::int32_t* __restrict__ cuSeqlens,
    uint32_t numRequests, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,
    uint32_t numBlocks, uint32_t itemBegin, uint32_t itemCount, float invSqrtLen,
    uint32_t quantMode)
{
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    turboquant_scatter::Run_(key, value, keyCache, valueCache, scaleCache, slotMapping,
                             piSigns, cuSeqlens, numRequests, numKvHeads, headSize, blockSize,
                             numBlocks, itemBegin, itemCount, invSqrtLen, quantMode);
#else
    (void)key; (void)value; (void)keyCache; (void)valueCache; (void)scaleCache;
    (void)slotMapping; (void)piSigns; (void)cuSeqlens; (void)numRequests; (void)numKvHeads;
    (void)headSize; (void)blockSize; (void)numBlocks; (void)itemBegin; (void)itemCount;
    (void)invSqrtLen; (void)quantMode;
#endif
}

#if !defined(__CCE_AICORE__)
#pragma clang diagnostic pop
#endif
#endif

#if defined(ASCEND_MICRO_CANN_LAUNCH) && !defined(__CCE_AICORE__)
#include "../../common/turboquant_launch.h"

#include <stdexcept>
#if defined(TURBOQUANT_LEGACY_CANN_LAUNCH)
extern "C" __global__ __aicore__ void turboquant_reshape_scatter_aiv(
    __gm__ bfloat16_t*, __gm__ bfloat16_t*, __gm__ std::int8_t*, __gm__ std::int8_t*,
    __gm__ float*, __gm__ std::int32_t*, __gm__ float*, __gm__ std::int32_t*, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t);
#endif

extern "C" void launch_turboquant_reshape_scatter_aiv(
    void* stream, std::uint32_t block_dim, std::uint16_t* key, std::uint16_t* value,
    std::int8_t* key_cache, std::int8_t* value_cache, float* scale_cache,
    std::int32_t* slot_mapping, float* pi_signs, std::int32_t* cu_seqlens,
    std::uint32_t num_requests, std::uint32_t num_kv_heads,
    std::uint32_t head_size, std::uint32_t block_size, std::uint32_t num_blocks,
    std::uint32_t item_begin, std::uint32_t item_count, float inv_sqrt_len,
    std::uint32_t quant_mode)
{
#if TURBOQUANT_INGRESS_NZ

    if (block_size % turboquant_scatter::kNzTileRows != 0u)
        throw std::invalid_argument("reshape_scatter_aiv: blockSize must be a multiple of 64 "
                                    "for the NZ-tiled ingress");
#endif
    if (static_cast<std::uint64_t>(item_begin) + item_count > UINT32_MAX)
        throw std::invalid_argument("reshape_scatter_aiv: item range exceeds uint32 launch ABI");
    if (quant_mode > 1u)
        throw std::invalid_argument("reshape_scatter_aiv: quant_mode is 0 (uniform) or 1 "
                                    "(Lloyd-Max)");
    turboquant_reshape_scatter_aiv<<<block_dim, nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(key), reinterpret_cast<bfloat16_t*>(value), key_cache,
        value_cache, scale_cache, slot_mapping, pi_signs, cu_seqlens, num_requests,
        num_kv_heads, head_size, block_size, num_blocks, item_begin, item_count,
        inv_sqrt_len, quant_mode);
}

#endif
