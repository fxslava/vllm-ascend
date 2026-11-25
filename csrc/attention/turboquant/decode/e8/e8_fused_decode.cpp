#include "../../common/e8_decode.hpp"
#include <cstdint>
#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_simt.h"
#include "e8_parallel_decode.hpp"

namespace e8_decode {

constexpr AscendC::FixpipeConfig kFixUb{AscendC::CO2Layout::ROW_MAJOR, true};
template <AscendC::HardEvent Event>
__aicore__ inline void Sync() {
    AscendC::SetFlag<Event>(0);
    AscendC::WaitFlag<Event>(0);
}
template <class T>
__aicore__ inline AscendC::LocalTensor<T> L1(std::uint32_t bytes, std::uint32_t n) {
    return AscendC::LocalTensor<T>(AscendC::TPosition::A1, bytes, n);
}
template <class T>
__aicore__ inline AscendC::LocalTensor<T> Ub(std::uint32_t bytes, std::uint32_t n) {
    return AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC, bytes, n);
}

template <bool Value, unsigned Coordinate>
__simt_callee__ inline void StoreCoordinate(unsigned block, int doubled, float scale, bool live,
                                            __ubuf__ float* packed, __gm__ bfloat16_t* witness) {
    const unsigned row = block / 32, channel = (block % 32) * 8 + Coordinate;
    const bfloat16_t rounded = __float2bfloat16_rn(live ? (float(doubled) * 0.5f) * scale : 0.0f);
    const unsigned dst = Value ? (row / 8) * 256 * 8 + channel * 8 + row % 8
                               : (channel / 8) * (kKeyGroupStrideBytes / 4) + row * 8 + channel % 8;
    packed[dst] = __bfloat162float(rounded);
    witness[row * 256 + channel] = rounded;
}

template <bool Value>
__simt_callee__ inline void DecodeBlock(unsigned block, __ubuf__ const std::uint16_t* labels,
                                        __ubuf__ const float* scales, __ubuf__ float* packed,
                                        __gm__ bfloat16_t* witness, unsigned live_rows) {
    const unsigned row = block / 32;
    const bool live = row < live_rows;
    const unsigned label = live ? labels[block] : 0;
    const float scale = live ? scales[row] : 0.0f;
    const auto q = e8_parallel::Decode(label);
    StoreCoordinate<Value, 0>(block, q.a, scale, live, packed, witness);
    StoreCoordinate<Value, 1>(block, q.b, scale, live, packed, witness);
    StoreCoordinate<Value, 2>(block, q.c, scale, live, packed, witness);
    StoreCoordinate<Value, 3>(block, q.d, scale, live, packed, witness);
    StoreCoordinate<Value, 4>(block, q.e, scale, live, packed, witness);
    StoreCoordinate<Value, 5>(block, q.f, scale, live, packed, witness);
    StoreCoordinate<Value, 6>(block, q.g, scale, live, packed, witness);
    StoreCoordinate<Value, 7>(block, q.h, scale, live, packed, witness);
}

template <bool Value>
__simt_vf__ __launch_bounds__(256) inline void DecodePacked(__ubuf__ std::uint16_t* labels,
                                                            __ubuf__ float* scales, __ubuf__ float* packed,
                                                            __gm__ bfloat16_t* witness, unsigned live_rows) {
    DecodeBlock<Value>(threadIdx.x, labels, scales, packed, witness, live_rows);
    DecodeBlock<Value>(threadIdx.x + 256, labels, scales, packed, witness, live_rows);
    DecodeBlock<Value>(threadIdx.x + 512, labels, scales, packed, witness, live_rows);
    DecodeBlock<Value>(threadIdx.x + 768, labels, scales, packed, witness, live_rows);
}
template <unsigned Step>
__simt_callee__ inline void PackQuerySlice(__ubuf__ float* raw, __ubuf__ float* packed) {
    const unsigned i = threadIdx.x + Step * 256, row = i / 256, c = i % 256;
    packed[(c / 8) * 8 * 8 + row * 8 + c % 8] = raw[i];
}
__simt_vf__ __launch_bounds__(256) inline void PackQuery(__ubuf__ float* raw, __ubuf__ float* packed) {
    PackQuerySlice<0>(raw, packed);
    PackQuerySlice<1>(raw, packed);
    PackQuerySlice<2>(raw, packed);
    PackQuerySlice<3>(raw, packed);
    PackQuerySlice<4>(raw, packed);
    PackQuerySlice<5>(raw, packed);
    PackQuerySlice<6>(raw, packed);
    PackQuerySlice<7>(raw, packed);
}

template <unsigned Head>
__simd_callee__ inline void SoftmaxHead(__ubuf__ float* scores, __ubuf__ float* maximum, __ubuf__ float* sum,
                                        __ubuf__ float* accumulator, float scale, unsigned valid) {
    namespace r = AscendC::Reg;
    auto all = r::CreateMask<float, r::MaskPattern::ALL>();
    auto state = r::CreateMask<float, r::MaskPattern::VL8>();
    auto live = r::UpdateMask<float>(valid);
    r::RegTensor<float> row, m, l, dead, raw, tile, next, decay, reduced, scaled, acc;
    r::LoadAlign(row, scores + Head * 64);
    r::LoadAlign<float, r::LoadDist::DIST_BRC_B32>(m, maximum + Head * 8);
    r::LoadAlign<float, r::LoadDist::DIST_BRC_B32>(l, sum + Head * 8);
    r::Muls(row, row, scale, all);
    r::Duplicate<float>(dead, -3.4028235e38f);
    r::Select<float>(row, row, dead, live);
    r::Reduce<r::ReduceType::MAX>(raw, row, all);
    r::Duplicate(tile, raw, all);
    r::Max(next, m, tile, all);
    r::Sub(decay, m, next, all);
    r::Exp(decay, decay, all);
    r::Sub(row, row, next, all);
    r::Exp(row, row, all);
    r::Reduce<r::ReduceType::SUM>(raw, row, all);
    r::Duplicate(reduced, raw, all);
    r::Mul(scaled, l, decay, all);
    r::Add(l, scaled, reduced, all);
    r::StoreAlign(maximum + Head * 8, next, state);
    r::StoreAlign(sum + Head * 8, l, state);
    r::StoreAlign(scores + Head * 64, row, all);
#define E8_DECAY(C)                                  \
    r::LoadAlign(acc, accumulator + Head * 256 + C); \
    r::Mul(acc, acc, decay, all);                    \
    r::StoreAlign(accumulator + Head * 256 + C, acc, all)
    E8_DECAY(0);
    E8_DECAY(64);
    E8_DECAY(128);
    E8_DECAY(192);
#undef E8_DECAY
}
[[maybe_unused]] __simd_vf__ inline void Softmax(__ubuf__ float* scores, __ubuf__ float* maximum,
                                                 __ubuf__ float* sum, __ubuf__ float* accumulator,
                                                 float scale, unsigned valid) {
    SoftmaxHead<0>(scores, maximum, sum, accumulator, scale, valid);
    SoftmaxHead<1>(scores, maximum, sum, accumulator, scale, valid);
    SoftmaxHead<2>(scores, maximum, sum, accumulator, scale, valid);
}
template <unsigned Head>
__simd_callee__ inline void NormalizeHead(__ubuf__ float* accumulator, __ubuf__ float* sum) {
    namespace r = AscendC::Reg;
    auto all = r::CreateMask<float, r::MaskPattern::ALL>();
    r::RegTensor<float> l, one, inverse, acc;
    r::LoadAlign<float, r::LoadDist::DIST_BRC_B32>(l, sum + Head * 8);
    r::Duplicate<float>(one, 1.0f);
    r::Div<float, r::MaskMergeMode::ZEROING>(inverse, one, l, all);
#define E8_NORMALIZE(C)                              \
    r::LoadAlign(acc, accumulator + Head * 256 + C); \
    r::Mul(acc, acc, inverse, all);                  \
    r::StoreAlign(accumulator + Head * 256 + C, acc, all)
    E8_NORMALIZE(0);
    E8_NORMALIZE(64);
    E8_NORMALIZE(128);
    E8_NORMALIZE(192);
#undef E8_NORMALIZE
}
[[maybe_unused]] __simd_vf__ inline void Normalize(__ubuf__ float* accumulator, __ubuf__ float* sum) {
    NormalizeHead<0>(accumulator, sum);
    NormalizeHead<1>(accumulator, sum);
    NormalizeHead<2>(accumulator, sum);
}

struct VectorBuffers {
    AscendC::LocalTensor<std::uint16_t> labels[2][2];
    AscendC::LocalTensor<float> scales[2][2], packed[2][2];
    AscendC::LocalTensor<float> scores, context, accumulator, probabilities, maximum, sum;
    AscendC::LocalTensor<bfloat16_t> output;
    AscendC::LocalTensor<std::uint64_t> offsets;
};
template <unsigned Slot, unsigned Plane>
__aicore__ inline void BindPlane(std::uint32_t base, VectorBuffers& b) {
    constexpr std::uint32_t label_offset = Slot * kSlotAllocation + Plane * kKeyPlaneAllocation;
    b.labels[Slot][Plane] = Ub<std::uint16_t>(base + label_offset, kLabelAllocation / 2);
    b.scales[Slot][Plane] = Ub<float>(base + label_offset + kLabelAllocation, kScaleAllocation / 4);
    b.packed[Slot][Plane] = Ub<float>(base + label_offset + kLabelAllocation + kScaleAllocation,
                                      (Plane == 0 ? kKeyPackedAllocation : kPackedAllocation) / 4);
}
__aicore__ inline void BindBuffers(std::uint32_t base, VectorBuffers& b) {
    BindPlane<0, 0>(base, b);
    BindPlane<0, 1>(base, b);
    BindPlane<1, 0>(base, b);
    BindPlane<1, 1>(base, b);
    b.scores = Ub<float>(base + kUbScores, (768 + kPad) / 4);
    b.context = Ub<float>(base + kUbContext, (3072 + kPad) / 4);
    b.accumulator = Ub<float>(base + kUbAccumulator, (3072 + kPad) / 4);
    b.probabilities = Ub<float>(base + kUbProbabilities, (768 + kPad) / 4);
    b.maximum = Ub<float>(base + kUbMaximum, (96 + kPad) / 4);
    b.sum = Ub<float>(base + kUbSum, (96 + kPad) / 4);
    b.output = Ub<bfloat16_t>(base + kUbOutput, (1536 + kPad) / 2);
    b.offsets = Ub<std::uint64_t>(base + kUbOffsets, (kMaxTiles * 8 + kPad) / 8);
}
template <unsigned Head>
__aicore__ inline void PackProbabilities(VectorBuffers& b) {
    const AscendC::UnaryRepeatParams rp{1, 1, 3, 1};
    AscendC::Muls(b.probabilities[Head * 8], b.scores[Head * 64], 1.0f, std::uint64_t{8}, std::uint8_t{8},
                  rp);
}
template <unsigned Slot, unsigned Plane>
__aicore__ inline void StagePlane(VectorBuffers& b, __gm__ std::uint8_t* packets, std::uint64_t packet_offset,
                                  unsigned owner, unsigned valid, __gm__ bfloat16_t* witness) {
    AscendC::GlobalTensor<std::uint16_t> gm_labels;
    AscendC::GlobalTensor<float> gm_scales;
    gm_labels.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::uint16_t*>(packets + packet_offset + owner * 2048));
    gm_scales.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(packets + packet_offset + 4096 + owner * 128));
    AscendC::DataCopy(b.labels[Slot][Plane], gm_labels, 1024);
    AscendC::DataCopy(b.scales[Slot][Plane], gm_scales, 32);
    Sync<AscendC::HardEvent::MTE2_V>();
    const unsigned begin = owner * 32;
    const unsigned local_valid = valid > begin ? (valid - begin < 32 ? valid - begin : 32) : 0;
    asc_vf_call<DecodePacked<(Plane == 1)>>(
        dim3(256), reinterpret_cast<__ubuf__ std::uint16_t*>(b.labels[Slot][Plane].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(b.scales[Slot][Plane].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(b.packed[Slot][Plane].GetPhyAddr()), witness + begin * 256,
        local_valid);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
    Sync<AscendC::HardEvent::V_MTE3>();
    if constexpr (Plane == 0) {
        auto dst = L1<float>(Slot * kL1Slot + owner * 1024, 64 * 256);
        AscendC::DataCopy(dst, b.packed[Slot][Plane], AscendC::DataCopyParams{32, 32, 1, 32});
    } else {
        auto dst = L1<float>(Slot * kL1Slot + kL1Plane + owner * 32768, 32 * 256);
        AscendC::DataCopy(dst, b.packed[Slot][Plane], 32 * 256);
    }
    Sync<AscendC::HardEvent::MTE3_S>();
}
template <unsigned Slot>
__aicore__ inline void StageTile(VectorBuffers& b, __gm__ std::uint8_t* keys, __gm__ std::uint8_t* values,
                                 unsigned tile, unsigned tiles, unsigned last, unsigned owner,
                                 __gm__ bfloat16_t* key_witness, __gm__ bfloat16_t* value_witness) {
    const std::uint64_t offset = b.offsets.GetValue(tile);
    const unsigned valid = tile + 1 == tiles ? last : 64;
    StagePlane<Slot, 0>(b, keys, offset, owner, valid, key_witness + tile * 64 * 256);
    StagePlane<Slot, 1>(b, values, offset, owner, valid, value_witness + tile * 64 * 256);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(Slot);
}
__aicore__ inline void StageQuery(VectorBuffers& b, __gm__ float* query, unsigned owner) {
    auto raw = b.packed[0][1];
    auto packed = b.packed[0][0];
    AscendC::GlobalTensor<float> input;
    input.SetGlobalBuffer(query + owner * 8 * 256);
    AscendC::DataCopy(raw, input, 8 * 256);
    Sync<AscendC::HardEvent::MTE2_V>();
    asc_vf_call<PackQuery>(dim3(256), reinterpret_cast<__ubuf__ float*>(raw.GetPhyAddr()),
                           reinterpret_cast<__ubuf__ float*>(packed.GetPhyAddr()));
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
    Sync<AscendC::HardEvent::V_MTE3>();
    AscendC::DataCopy(L1<float>(kL1Query + owner * 256, 16 * 256), packed,
                      AscendC::DataCopyParams{32, 8, 0, 8});
    Sync<AscendC::HardEvent::MTE3_S>();
}

template <unsigned Slot, bool Refill>
__aicore__ inline void VectorTile(VectorBuffers& b, __gm__ std::uint8_t* keys, __gm__ std::uint8_t* values,
                                  unsigned tile, unsigned tiles, unsigned last, unsigned owner, float scale,
                                  __gm__ bfloat16_t* key_witness, __gm__ bfloat16_t* value_witness,
                                  __gm__ float* score_witness) {
    AscendC::CrossCoreWaitFlag<2, PIPE_V>(4);

    Sync<AscendC::HardEvent::V_MTE3>();
    AscendC::GlobalTensor<float> score_output;
    score_output.SetGlobalBuffer(score_witness + (tile * 6 + owner * 3) * 64);
    AscendC::DataCopy(score_output, b.scores, 3 * 64);
    Sync<AscendC::HardEvent::MTE3_V>();
    const unsigned valid = tile + 1 == tiles ? last : 64;
    Softmax(reinterpret_cast<__ubuf__ float*>(b.scores.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(b.maximum.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(b.sum.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(b.accumulator.GetPhyAddr()), scale, valid);
    AscendC::PipeBarrier<PIPE_V>();
    PackProbabilities<0>(b);
    PackProbabilities<1>(b);
    PackProbabilities<2>(b);
    Sync<AscendC::HardEvent::V_MTE3>();
    AscendC::DataCopy(L1<float>(kL1Probabilities + owner * 3 * 8 * 4, 16 * 64), b.probabilities,
                      AscendC::DataCopyParams{8, 3, 0, 13});
    Sync<AscendC::HardEvent::MTE3_S>();
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(5);
    AscendC::CrossCoreWaitFlag<2, PIPE_V>(6);
    AscendC::Add(b.accumulator, b.accumulator, b.context, 3 * 256);
    if constexpr (Refill) {
        AscendC::CrossCoreWaitFlag<2, PIPE_MTE3>(2 + Slot);

        Sync<AscendC::HardEvent::MTE3_S>();
        StageTile<Slot>(b, keys, values, tile + 2, tiles, last, owner, key_witness, value_witness);
    }
}

template <bool Scores>
__aicore__ inline void Product(__ca__ float* l0a, __cb__ float* l0b, __cc__ float* l0c, unsigned slot,
                               std::uint32_t ub_base) {
    using namespace asc::te;
    constexpr unsigned k = Scores ? 256 : 64, n = Scores ? 64 : 256;
    const unsigned left = Scores ? kL1Query : kL1Probabilities;
    const unsigned right = slot * kL1Slot + (Scores ? 0 : kL1Plane);
    auto sa = make_tensor(make_mem_ptr(reinterpret_cast<__cbuf__ float*>(left)),
                          make_frame_layout<nz_layout_ptn, float>(16, k));
    auto da = make_tensor(make_mem_ptr(l0a), make_frame_layout<nz_layout_ptn, float>(16, k));
    copy(make_copy(copy_l1_to_l0a{}), da, sa);
    auto sb = make_tensor(make_mem_ptr(reinterpret_cast<__cbuf__ float*>(right)),
                          make_frame_layout<zn_layout_ptn, float>(k, n));
    auto db = make_tensor(make_mem_ptr(l0b), make_frame_layout<zn_layout_ptn, float>(k, n));
    copy(make_copy(copy_l1_to_l0b{}), db, sb);
    Sync<AscendC::HardEvent::MTE1_M>();
    const auto a_base = static_cast<std::uint32_t>(reinterpret_cast<std::uint64_t>(l0a));
    const auto b_base = static_cast<std::uint32_t>(reinterpret_cast<std::uint64_t>(l0b));
    const auto c_base = static_cast<std::uint32_t>(reinterpret_cast<std::uint64_t>(l0c));
    const AscendC::LocalTensor<float> at(AscendC::TPosition::A2, a_base, 16 * 256);
    const AscendC::LocalTensor<float> bt(AscendC::TPosition::B2, b_base, 64 * 256);
    const AscendC::LocalTensor<float> ct(AscendC::TPosition::CO1, c_base, 16 * 256);
    AscendC::Mmad(ct, at, bt, AscendC::MmadParams(6, n, k, 0, false, true));
    Sync<AscendC::HardEvent::M_FIX>();
    AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fp(n, 6, 16, n);
    fp.dualDstCtl = 1;
    fp.subBlockId = false;
    AscendC::Fixpipe<float, float, kFixUb>(Ub<float>(ub_base + (Scores ? kUbScores : kUbContext), 3 * n), ct,
                                           fp);
    Sync<AscendC::HardEvent::FIX_S>();
}
template <unsigned Slot, bool Release>
__aicore__ inline void CubeTile(__ca__ float* l0a, __cb__ float* l0b, __cc__ float* l0c,
                                std::uint32_t ub_base) {
    AscendC::CrossCoreWaitFlag<2, PIPE_MTE1>(Slot);
    Product<true>(l0a, l0b, l0c, Slot, ub_base);
    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(4);
    AscendC::CrossCoreWaitFlag<2, PIPE_MTE1>(5);
    Product<false>(l0a, l0b, l0c, Slot, ub_base);
    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(6);
    if constexpr (Release) AscendC::CrossCoreSetFlag<2, PIPE_FIX>(2 + Slot);
}
}

extern "C" __global__ __aicore__ void e8_fused_decode_fp32(GM_ADDR gm_keys, GM_ADDR gm_values,
                                                           GM_ADDR gm_offsets, GM_ADDR gm_query,
                                                           GM_ADDR gm_output, GM_ADDR gm_key_witness,
                                                           GM_ADDR gm_value_witness, GM_ADDR gm_score_witness,
                                                           uint32_t tiles, uint32_t last_rows, float scale) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using namespace e8_decode;

    __ubuf__ std::uint8_t arena[kUbStorageBytes] __attribute__((aligned(32)));
    const std::uint32_t ub_base = static_cast<std::uint32_t>(reinterpret_cast<std::uint64_t>(arena));
    if ASCEND_IS_AIC {
        AscendC::SetMMRowMajor();
        AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
        __ca__ float l0a[16 * 256] __attribute__((aligned(512)));
        __cb__ float l0b[64 * 256] __attribute__((aligned(512)));
        __cc__ float l0c[16 * 256] __attribute__((aligned(1024)));

        AscendC::GlobalTensor<float> zeros;
        zeros.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gm_query) + 6 * 256);
        AscendC::DataCopy(e8_decode::L1<float>(kL1Probabilities, 16 * 64), zeros, 16 * 64);
        Sync<AscendC::HardEvent::MTE2_MTE1>();
        for (unsigned tile = 0; tile + 2 < tiles; tile += 2) {
            CubeTile<0, true>(l0a, l0b, l0c, ub_base);
            CubeTile<1, true>(l0a, l0b, l0c, ub_base);
        }
        CubeTile<0, false>(l0a, l0b, l0c, ub_base);
        CubeTile<1, false>(l0a, l0b, l0c, ub_base);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(7);
        AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(8);
        Sync<AscendC::HardEvent::FIX_S>();
    }
    if ASCEND_IS_AIV {
        VectorBuffers b;
        BindBuffers(ub_base, b);
        const unsigned owner = AscendC::GetSubBlockIdx();
        AscendC::GlobalTensor<std::uint64_t> map;
        map.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint64_t*>(gm_offsets));
        AscendC::DataCopy(b.offsets, map, ((tiles * 8 + 31u) & ~31u) / 8);
        Sync<AscendC::HardEvent::MTE2_S>();
        AscendC::Duplicate(b.maximum, -3.4028235e38f, 24);
        AscendC::Duplicate(b.sum, 0.0f, 24);
        AscendC::Duplicate(b.accumulator, 0.0f, 3 * 256);
        StageQuery(b, reinterpret_cast<__gm__ float*>(gm_query), owner);
        auto* kw = reinterpret_cast<__gm__ bfloat16_t*>(gm_key_witness);
        auto* vw = reinterpret_cast<__gm__ bfloat16_t*>(gm_value_witness);
        StageTile<0>(b, gm_keys, gm_values, 0, tiles, last_rows, owner, kw, vw);
        StageTile<1>(b, gm_keys, gm_values, 1, tiles, last_rows, owner, kw, vw);
        for (unsigned tile = 0; tile + 2 < tiles; tile += 2) {
            VectorTile<0, true>(b, gm_keys, gm_values, tile, tiles, last_rows, owner, scale, kw, vw,
                                reinterpret_cast<__gm__ float*>(gm_score_witness));
            VectorTile<1, true>(b, gm_keys, gm_values, tile + 1, tiles, last_rows, owner, scale, kw, vw,
                                reinterpret_cast<__gm__ float*>(gm_score_witness));
        }
        VectorTile<0, false>(b, gm_keys, gm_values, tiles - 2, tiles, last_rows, owner, scale, kw, vw,
                             reinterpret_cast<__gm__ float*>(gm_score_witness));
        VectorTile<1, false>(b, gm_keys, gm_values, tiles - 1, tiles, last_rows, owner, scale, kw, vw,
                             reinterpret_cast<__gm__ float*>(gm_score_witness));
        AscendC::CrossCoreWaitFlag<2, PIPE_V>(7);
        Normalize(reinterpret_cast<__ubuf__ float*>(b.accumulator.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ float*>(b.sum.GetPhyAddr()));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(b.output, b.accumulator, AscendC::RoundMode::CAST_RINT, 3 * 256);
        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::GlobalTensor<bfloat16_t> destination;
        destination.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gm_output) + owner * 3 * 256);
        AscendC::DataCopy(destination, b.output, 3 * 256);
        Sync<AscendC::HardEvent::MTE3_S>();
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(8);
    }
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)
#include "kernel_operator.h"
#include <stdexcept>
extern "C" __global__ __aicore__ void e8_fused_decode_fp32(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                                                           GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t,
                                                           float);
extern "C" void launch_e8_fused_decode_fp32(void* stream, const e8_decode::Launch* c,
                                            const e8_decode::Buffers* b) {
    if (!c || !b || c->tiles < 4 || c->tiles > e8_decode::kMaxTiles || c->tiles % 2 || !c->last_rows ||
        c->last_rows > 64 || !b->key_packets || !b->value_packets || !b->packet_offsets || !b->query_padded ||
        !b->output || !b->key_witness || !b->value_witness || !b->score_witness)
        throw std::invalid_argument("invalid E8 D256 six-head even-tile launch");
    e8_fused_decode_fp32<<<1, nullptr, stream>>>(
        static_cast<std::uint8_t*>(b->key_packets), static_cast<std::uint8_t*>(b->value_packets),
        static_cast<std::uint8_t*>(b->packet_offsets), static_cast<std::uint8_t*>(b->query_padded),
        static_cast<std::uint8_t*>(b->output), static_cast<std::uint8_t*>(b->key_witness),
        static_cast<std::uint8_t*>(b->value_witness), static_cast<std::uint8_t*>(b->score_witness), c->tiles,
        c->last_rows, c->attention_scale);
}
#endif
