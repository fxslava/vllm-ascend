#pragma once
#include <cstdint>
#include <type_traits>
#ifndef E8_CPU_TEST
#include "kernel_operator.h"
#include "simt_api/device_functions.h"
#include "simt_api/asc_simt.h"
#define E8_THREAD __simt_callee__ inline __attribute__((always_inline))
#define E8_UB __ubuf__
#else
#define E8_THREAD inline
#define E8_UB
#endif
namespace ascend_micro::e8_int8_simt {
enum class Backend { SIMD, SIMT };
enum class Representation { Arithmetic, CompactInt8, CompactBf16 };
enum class TableLayout { AoS, SoA, PaddedSoA, Blocked };
enum class Shaping { LegacyNearest, ResidualBits };
template <TableLayout Layout> struct PartialTableFootprint {
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
E8_THREAD void CompactProject(unsigned label, E8_UB const int8_t* partialTable,
                              int (&projectedDoubled)[8]) {
#pragma unroll
  for (unsigned coordinate = 0; coordinate < 8; ++coordinate) {
    projectedDoubled[coordinate] =
        int(partialTable[PartialOffset<Layout>(label & 255, coordinate)]) +
        int(partialTable[PartialTableFootprint<Layout>::PartialBytes +
                         PartialOffset<Layout>(label >> 8, coordinate)]);
  }
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
template <unsigned Elements, Representation R,
          TableLayout Layout = TableLayout::AoS>
struct DequantFootprint {
  static_assert(Elements > 0 && Elements % 256 == 0);
  static_assert(R == Representation::Arithmetic ||
                R == Representation::CompactInt8);
  static constexpr unsigned IndexCount = Elements / 8;
  static constexpr unsigned ScaleCount = Elements / 256;
  static constexpr unsigned OutputCount = Elements;
  static constexpr unsigned IndexBytes = IndexCount * sizeof(uint16_t);
  static constexpr unsigned ScaleBytes = ScaleCount * sizeof(float);
  static constexpr unsigned OutputBytes = OutputCount * sizeof(int8_t);
  static constexpr unsigned ScratchBytes = 0;
  static constexpr unsigned TableBytes =
      R == Representation::Arithmetic
          ? 0
          : PartialTableFootprint<Layout>::TotalBytes;
  static constexpr unsigned BaseAlignmentBytes = 32;
  static constexpr unsigned FullRegisterReadPadBytes = 256;
  static constexpr unsigned IndexAllocationBytes =
      ((IndexBytes + 31) & ~31u) + FullRegisterReadPadBytes;
  static constexpr unsigned ScaleAllocationBytes =
      ((ScaleBytes + 31) & ~31u) + FullRegisterReadPadBytes;
  static constexpr unsigned OutputAllocationBytes =
      ((OutputBytes + 31) & ~31u) + FullRegisterReadPadBytes;
  static constexpr unsigned ScratchAllocationBytes = 0;
  static constexpr unsigned TableAllocationBytes =
      TableBytes == 0 ? 0
                      : ((TableBytes + 31) & ~31u) + FullRegisterReadPadBytes;
};
namespace detail {
E8_THREAD int Floor8(int x) { return x >= 0 ? x / 8 : -((-x + 7) / 8); }
E8_THREAD void Project(unsigned label, int (&projectedDoubled)[8]) {
  int coefficient0 = 2 * int((label >> 14) & 1) + int((label >> 15) & 1) - 2;
  int coefficient1 = int(label & 3) - 2,
      coefficient2 = int((label >> 2) & 3) - 2,
      coefficient3 = int((label >> 4) & 3) - 2;
  int coefficient4 = int((label >> 6) & 3) - 2,
      coefficient5 = int((label >> 8) & 3) - 2,
      coefficient6 = int((label >> 10) & 3) - 2,
      coefficient7 = int((label >> 12) & 3) - 2;
  projectedDoubled[0] = coefficient0;
  projectedDoubled[1] = coefficient0 + 2 * coefficient1;
  projectedDoubled[2] = coefficient0 - 2 * coefficient1 + 2 * coefficient2;
  projectedDoubled[3] = coefficient0 - 2 * coefficient2 + 2 * coefficient3;
  projectedDoubled[4] = coefficient0 - 2 * coefficient3 + 2 * coefficient4;
  projectedDoubled[5] = coefficient0 - 2 * coefficient4 + 2 * coefficient5;
  projectedDoubled[6] =
      coefficient0 - 2 * coefficient5 + 2 * coefficient6 + 2 * coefficient7;
  projectedDoubled[7] = coefficient0 - 2 * coefficient6 + 2 * coefficient7;
}
// Exact rational nearest E8: candidates in D8 and D8+h, doubled coordinates.
// Equal distance chooses largest nearest point, hence smallest residual lex.
E8_THREAD void Shape(int (&projectedDoubled)[8]) {
  int bestNearestDoubled[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int bestDistance = 0x7fffffff;
#pragma unroll
  for (int halfCoset = 0; halfCoset < 2; ++halfCoset) {
    int nearestDoubled[8], residualDoubled[8];
    int parity = 0;
#pragma unroll
    for (int coordinate = 0; coordinate < 8; ++coordinate) {
      int integerCoordinate =
          Floor8(projectedDoubled[coordinate] - 4 * halfCoset + 4);
      nearestDoubled[coordinate] = 2 * integerCoordinate + halfCoset;
      residualDoubled[coordinate] =
          projectedDoubled[coordinate] - 4 * nearestDoubled[coordinate];
      parity += integerCoordinate;
    }
    if (parity & 1) {
      int maximumAbsoluteResidual = -1, repairCoordinate = 0;
      bool repairPositive = false;
#pragma unroll
      for (int coordinate = 0; coordinate < 8; ++coordinate) {
        int absoluteResidual = residualDoubled[coordinate] < 0
                                   ? -residualDoubled[coordinate]
                                   : residualDoubled[coordinate];
        if (absoluteResidual > maximumAbsoluteResidual ||
            (absoluteResidual == maximumAbsoluteResidual &&
             ((residualDoubled[coordinate] >= 0 && !repairPositive) ||
              (residualDoubled[coordinate] < 0 && !repairPositive)))) {
          maximumAbsoluteResidual = absoluteResidual;
          repairCoordinate = coordinate;
          repairPositive = residualDoubled[coordinate] >= 0;
        }
      }
#pragma unroll
      for (int coordinate = 0; coordinate < 8; ++coordinate)
        if (coordinate == repairCoordinate)
          nearestDoubled[coordinate] += repairPositive ? 2 : -2;
    }
    int distance = 0;
    bool nearestLexGreater = false, lexDecided = false;
#pragma unroll
    for (int coordinate = 0; coordinate < 8; ++coordinate) {
      int residual =
          projectedDoubled[coordinate] - 4 * nearestDoubled[coordinate];
      distance += residual * residual;
      if (!lexDecided &&
          nearestDoubled[coordinate] != bestNearestDoubled[coordinate]) {
        nearestLexGreater =
            nearestDoubled[coordinate] > bestNearestDoubled[coordinate];
        lexDecided = true;
      }
    }
    if (distance < bestDistance ||
        (distance == bestDistance && nearestLexGreater)) {
      bestDistance = distance;
#pragma unroll
      for (int coordinate = 0; coordinate < 8; ++coordinate)
        bestNearestDoubled[coordinate] = nearestDoubled[coordinate];
    }
  }
#pragma unroll
  for (int coordinate = 0; coordinate < 8; ++coordinate)
    projectedDoubled[coordinate] -= 4 * bestNearestDoubled[coordinate];
}
// Repair the nearest D8 parity directly in doubled residual coordinates.
E8_THREAD void RepairResidual(int (&residuals)[8], unsigned parity) {
  if (parity == 0)
    return;
  int maximumAbsoluteResidual = -1;
  unsigned repairCoordinate = 0;
  bool repairPositive = false;
#pragma unroll
  for (unsigned coordinate = 0; coordinate < 8; ++coordinate) {
    const int residual = residuals[coordinate];
    const int absoluteResidual = residual < 0 ? -residual : residual;
    if (absoluteResidual > maximumAbsoluteResidual ||
        (absoluteResidual == maximumAbsoluteResidual && !repairPositive)) {
      maximumAbsoluteResidual = absoluteResidual;
      repairCoordinate = coordinate;
      repairPositive = residual >= 0;
    }
  }
#pragma unroll
  for (unsigned coordinate = 0; coordinate < 8; ++coordinate) {
    if (coordinate == repairCoordinate)
      residuals[coordinate] += repairPositive ? -8 : 8;
  }
}
E8_THREAD void ShapeResidualBits(int (&doubledCoordinates)[8]) {
  int halfCosetResiduals[8];
  unsigned integerParity = 0;
  unsigned negativeParity = 0;
#pragma unroll
  for (unsigned coordinate = 0; coordinate < 8; ++coordinate) {
    const uint32_t shiftedCoordinate =
        static_cast<uint32_t>(doubledCoordinates[coordinate]) + 4u;
    const int integerResidual = static_cast<int>(shiftedCoordinate & 7u) - 4;
    integerParity ^= (shiftedCoordinate >> 3u) & 1u;
    negativeParity ^= static_cast<unsigned>(integerResidual < 0);
    doubledCoordinates[coordinate] = integerResidual;
    halfCosetResiduals[coordinate] =
        integerResidual < 0 ? integerResidual + 4 : integerResidual - 4;
  }
  const unsigned halfCosetParity = integerParity ^ negativeParity;
  RepairResidual(doubledCoordinates, integerParity);
  RepairResidual(halfCosetResiduals, halfCosetParity);
  int integerDistance = 0;
  int halfCosetDistance = 0;
#pragma unroll
  for (unsigned coordinate = 0; coordinate < 8; ++coordinate) {
    integerDistance +=
        doubledCoordinates[coordinate] * doubledCoordinates[coordinate];
    halfCosetDistance +=
        halfCosetResiduals[coordinate] * halfCosetResiduals[coordinate];
  }
  const bool chooseHalfCoset = halfCosetDistance < integerDistance ||
                               (halfCosetDistance == integerDistance &&
                                halfCosetResiduals[0] < doubledCoordinates[0]);
#pragma unroll
  for (unsigned coordinate = 0; coordinate < 8; ++coordinate) {
    if (chooseHalfCoset)
      doubledCoordinates[coordinate] = halfCosetResiduals[coordinate];
  }
}
E8_THREAD int8_t PhysicalInt8(float physicalValue) {
  float lowerClamped = physicalValue < -128.0f ? -128.0f : physicalValue;
  float clampedValue = lowerClamped > 127.0f ? 127.0f : lowerClamped;
#ifndef E8_CPU_TEST
  int roundedValue = __float2int_rn(clampedValue);
#else
  int roundedValue = int(__builtin_nearbyintf(clampedValue));
#endif
  return static_cast<int8_t>(roundedValue);
}
#ifndef E8_CPU_TEST
template <unsigned Elements, Representation R, unsigned Threads,
          TableLayout Layout = TableLayout::AoS,
          Shaping ShapeAlgorithm = Shaping::LegacyNearest>
__simt_vf__ __launch_bounds__(Threads) inline void DequantVf(
    __ubuf__ uint16_t* indices, __ubuf__ float* scales,
    __ubuf__ int8_t* output, __ubuf__ uint8_t* table) {
  for (unsigned block = threadIdx.x; block < Elements / 8; block += Threads) {
    unsigned label = indices[block];
    int projectedDoubled[8];
    if constexpr (R == Representation::Arithmetic)
      Project(label, projectedDoubled);
    else
      CompactProject<Layout>(label,
                             reinterpret_cast<__ubuf__ const int8_t*>(table),
                             projectedDoubled);
    if constexpr (ShapeAlgorithm == Shaping::LegacyNearest)
      Shape(projectedDoubled);
    else
      ShapeResidualBits(projectedDoubled);
    float scale = scales[block / 32];
#pragma unroll
    for (int coordinate = 0; coordinate < 8; ++coordinate)
      output[block * 8 + coordinate] = PhysicalInt8(
          (float(projectedDoubled[coordinate]) * 0.5f) * scale);
  }
}
#endif
} // namespace detail
#ifndef E8_CPU_TEST
template <class T, unsigned Elements, Backend B, Representation R,
          unsigned Threads = SimtThreadCount,
          TableLayout Layout = TableLayout::AoS,
          Shaping ShapeAlgorithm = Shaping::LegacyNearest>
__aicore__ inline void Dequant(const AscendC::LocalTensor<uint16_t>& indices,
                               const AscendC::LocalTensor<float>& scales,
                               const AscendC::LocalTensor<T>& output,
                               const AscendC::LocalTensor<uint8_t>& scratch,
                               const AscendC::LocalTensor<uint8_t>& table) {
  static_assert(Elements > 0 && Elements % 256 == 0);
  static_assert(R != Representation::CompactBf16);
  static_assert(B == Backend::SIMT);
  static_assert(Threads >= 32 && Threads <= 2048 && Threads % 32 == 0);
  static_assert(std::is_same<T, int8_t>::value,
                "INT8 SIMT leaf requires int8_t");
  (void)scratch;
  asc_vf_call<detail::DequantVf<Elements, R, Threads, Layout, ShapeAlgorithm>>(
      dim3(Threads), reinterpret_cast<__ubuf__ uint16_t*>(indices.GetPhyAddr()),
      reinterpret_cast<__ubuf__ float*>(scales.GetPhyAddr()),
      reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr()),
      reinterpret_cast<__ubuf__ uint8_t*>(table.GetPhyAddr()));
}
#endif
} // namespace ascend_micro::e8_int8_simt
#undef E8_THREAD
#undef E8_UB
