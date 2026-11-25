#pragma once
#include <cstdint>
#include <type_traits>
#ifndef E8_CPU_TEST
#include "kernel_operator.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_simt.h"
#define E8_THREAD __simt_callee__ inline __attribute__((always_inline))
#define E8_UB __ubuf__
#else
#define E8_THREAD inline
#define E8_UB
#endif
#ifdef E8_CPU_TEST
#ifndef E8_PARALLEL_CPU
#define E8_PARALLEL_CPU
#define E8_PARALLEL_UNDEFINE_CPU
#endif
#endif
#include "../e8_parallel_decode.hpp"
#ifdef E8_PARALLEL_UNDEFINE_CPU
#undef E8_PARALLEL_CPU
#undef E8_PARALLEL_UNDEFINE_CPU
#endif
namespace ascend_micro::e8 {
enum class Backend { SIMD, SIMT };
enum class Representation { Arithmetic, CompactInt8, CompactBf16 };
enum class TableLayout { AoS, SoA, PaddedSoA, Blocked };
enum class Shaping { LegacyNearest, ResidualBits };
template <TableLayout Layout>
struct PartialTableFootprint {
    static constexpr unsigned PartialBytes =
        Layout == TableLayout::AoS || Layout == TableLayout::SoA ? 2048 : 2304;
    static constexpr unsigned TotalBytes = 2 * PartialBytes;
};
template <TableLayout Layout>
E8_THREAD unsigned PartialOffset(unsigned labelByte, unsigned coordinate) {
    if constexpr (Layout == TableLayout::AoS)
        return labelByte * 8 + coordinate;
    else if constexpr (Layout == TableLayout::SoA)
        return coordinate * 256 + labelByte;
    else if constexpr (Layout == TableLayout::PaddedSoA)
        return coordinate * 288 + labelByte;
    else
        return (labelByte / 32) * 288 + coordinate * 32 + labelByte % 32;
}
template <TableLayout Layout>
E8_THREAD void CompactProject(unsigned label, E8_UB const int8_t* partialTable, int (&projectedDoubled)[8]) {
    projectedDoubled[0] =
        int(partialTable[PartialOffset<Layout>(label & 255, 0)]) +
        int(partialTable[PartialTableFootprint<Layout>::PartialBytes + PartialOffset<Layout>(label >> 8, 0)]);
    projectedDoubled[1] =
        int(partialTable[PartialOffset<Layout>(label & 255, 1)]) +
        int(partialTable[PartialTableFootprint<Layout>::PartialBytes + PartialOffset<Layout>(label >> 8, 1)]);
    projectedDoubled[2] =
        int(partialTable[PartialOffset<Layout>(label & 255, 2)]) +
        int(partialTable[PartialTableFootprint<Layout>::PartialBytes + PartialOffset<Layout>(label >> 8, 2)]);
    projectedDoubled[3] =
        int(partialTable[PartialOffset<Layout>(label & 255, 3)]) +
        int(partialTable[PartialTableFootprint<Layout>::PartialBytes + PartialOffset<Layout>(label >> 8, 3)]);
    projectedDoubled[4] =
        int(partialTable[PartialOffset<Layout>(label & 255, 4)]) +
        int(partialTable[PartialTableFootprint<Layout>::PartialBytes + PartialOffset<Layout>(label >> 8, 4)]);
    projectedDoubled[5] =
        int(partialTable[PartialOffset<Layout>(label & 255, 5)]) +
        int(partialTable[PartialTableFootprint<Layout>::PartialBytes + PartialOffset<Layout>(label >> 8, 5)]);
    projectedDoubled[6] =
        int(partialTable[PartialOffset<Layout>(label & 255, 6)]) +
        int(partialTable[PartialTableFootprint<Layout>::PartialBytes + PartialOffset<Layout>(label >> 8, 6)]);
    projectedDoubled[7] =
        int(partialTable[PartialOffset<Layout>(label & 255, 7)]) +
        int(partialTable[PartialTableFootprint<Layout>::PartialBytes + PartialOffset<Layout>(label >> 8, 7)]);
}
#ifndef E8_SIMT_THREAD_COUNT
#define E8_SIMT_THREAD_COUNT 128
#endif
constexpr unsigned SimtThreadCount = E8_SIMT_THREAD_COUNT;
struct SimtDequantTraits {
    static constexpr unsigned ScratchBytes = 0;
    static constexpr unsigned Int8TableBytes = 4096;
    static constexpr unsigned BaseAlignmentBytes = 32;
    static constexpr unsigned FullRegisterReadPadBytes = 256;
};
template <unsigned Elements, Representation R, TableLayout Layout = TableLayout::AoS>
struct DequantFootprint {
    static_assert(Elements > 0 && Elements % 256 == 0);
    static_assert(R == Representation::Arithmetic || R == Representation::CompactInt8);
    static constexpr unsigned IndexCount = Elements / 8;
    static constexpr unsigned ScaleCount = Elements / 256;
    static constexpr unsigned OutputCount = Elements;
    static constexpr unsigned IndexBytes = IndexCount * sizeof(uint16_t);
    static constexpr unsigned ScaleBytes = ScaleCount * sizeof(float);
    static constexpr unsigned OutputBytes = OutputCount * sizeof(uint16_t);
    static constexpr unsigned ScratchBytes = 0;
    static constexpr unsigned TableBytes =
        R == Representation::Arithmetic ? 0 : PartialTableFootprint<Layout>::TotalBytes;
    static constexpr unsigned BaseAlignmentBytes = 32;
    static constexpr unsigned FullRegisterReadPadBytes = 256;
    static constexpr unsigned IndexAllocationBytes = ((IndexBytes + 31) & ~31u) + FullRegisterReadPadBytes;
    static constexpr unsigned ScaleAllocationBytes = ((ScaleBytes + 31) & ~31u) + FullRegisterReadPadBytes;
    static constexpr unsigned OutputAllocationBytes = ((OutputBytes + 31) & ~31u) + FullRegisterReadPadBytes;
    static constexpr unsigned ScratchAllocationBytes = 0;
    static constexpr unsigned TableAllocationBytes =
        TableBytes == 0 ? 0 : ((TableBytes + 31) & ~31u) + FullRegisterReadPadBytes;
};
E8_THREAD e8_parallel::I8 ToRegisters(const int (&p)[8]) {
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]};
}
E8_THREAD void FromRegisters(e8_parallel::I8 q, int (&p)[8]) {
    p[0] = q.a;
    p[1] = q.b;
    p[2] = q.c;
    p[3] = q.d;
    p[4] = q.e;
    p[5] = q.f;
    p[6] = q.g;
    p[7] = q.h;
}
E8_THREAD int Floor8(int x) { return x >= 0 ? x / 8 : -((-x + 7) / 8); }
E8_THREAD void Project(unsigned label, int (&p)[8]) { FromRegisters(e8_parallel::Project(label), p); }
E8_THREAD void RepairResidual(int (&p)[8], unsigned parity) {
    FromRegisters(e8_parallel::Repair(ToRegisters(p), parity), p);
}
E8_THREAD void ShapeResidualBits(int (&p)[8]) {
    FromRegisters(e8_parallel::ShapeResidualBits(ToRegisters(p)), p);
}

E8_THREAD void Shape(int (&p)[8]) { ShapeResidualBits(p); }
#ifndef E8_CPU_TEST
template <unsigned Elements, Representation R, unsigned Threads, TableLayout Layout = TableLayout::AoS,
          Shaping ShapeAlgorithm = Shaping::LegacyNearest>
__simt_vf__ __launch_bounds__(Threads) inline void DequantVf(__ubuf__ uint16_t* indices,
                                                             __ubuf__ float* scales,
                                                             __ubuf__ bfloat16_t* output,
                                                             __ubuf__ uint8_t* table) {
    for (unsigned block = threadIdx.x; block < Elements / 8; block += Threads) {
        unsigned label = indices[block];
        int projectedDoubled[8];
        if constexpr (R == Representation::Arithmetic)
            Project(label, projectedDoubled);
        else
            CompactProject<Layout>(label, reinterpret_cast<__ubuf__ const int8_t*>(table), projectedDoubled);
        if constexpr (ShapeAlgorithm == Shaping::LegacyNearest)
            Shape(projectedDoubled);
        else
            ShapeResidualBits(projectedDoubled);
        float scale = scales[block / 32];
        output[block * 8 + 0] = __float2bfloat16_rn((float(projectedDoubled[0]) * 0.5f) * scale);
        output[block * 8 + 1] = __float2bfloat16_rn((float(projectedDoubled[1]) * 0.5f) * scale);
        output[block * 8 + 2] = __float2bfloat16_rn((float(projectedDoubled[2]) * 0.5f) * scale);
        output[block * 8 + 3] = __float2bfloat16_rn((float(projectedDoubled[3]) * 0.5f) * scale);
        output[block * 8 + 4] = __float2bfloat16_rn((float(projectedDoubled[4]) * 0.5f) * scale);
        output[block * 8 + 5] = __float2bfloat16_rn((float(projectedDoubled[5]) * 0.5f) * scale);
        output[block * 8 + 6] = __float2bfloat16_rn((float(projectedDoubled[6]) * 0.5f) * scale);
        output[block * 8 + 7] = __float2bfloat16_rn((float(projectedDoubled[7]) * 0.5f) * scale);
    }
}
template <class T, unsigned Elements, Backend B, Representation R, unsigned Threads = SimtThreadCount,
          TableLayout Layout = TableLayout::AoS, Shaping ShapeAlgorithm = Shaping::LegacyNearest>
__aicore__ inline void Dequant(const AscendC::LocalTensor<uint16_t>& indices,
                               const AscendC::LocalTensor<float>& scales,
                               const AscendC::LocalTensor<T>& output,
                               const AscendC::LocalTensor<uint8_t>& scratch,
                               const AscendC::LocalTensor<uint8_t>& table) {
    static_assert(Elements > 0 && Elements % 256 == 0);
    static_assert(R != Representation::CompactBf16);
    static_assert(B == Backend::SIMT);
    static_assert(Threads >= 32 && Threads <= 2048 && Threads % 32 == 0);
    static_assert(std::is_same<T, bfloat16_t>::value, "BF16 SIMT leaf requires bfloat16_t");
    (void)scratch;
    asc_vf_call<DequantVf<Elements, R, Threads, Layout, ShapeAlgorithm>>(
        dim3(Threads), reinterpret_cast<__ubuf__ uint16_t*>(indices.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(scales.GetPhyAddr()),
        reinterpret_cast<__ubuf__ bfloat16_t*>(output.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint8_t*>(table.GetPhyAddr()));
}
#endif
}
#undef E8_THREAD
#undef E8_UB
