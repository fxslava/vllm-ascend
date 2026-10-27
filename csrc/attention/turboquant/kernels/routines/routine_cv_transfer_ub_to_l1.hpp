#pragma once

#include <cstdint>
#include <utility>

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "kernel_operator.h"
#endif

namespace ascend_micro {

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#define ASCEND_MICRO_CV_INLINE __aicore__ inline
#else
#define ASCEND_MICRO_CV_INLINE inline
#endif

// Cube-Vector direct transfer leaf: Vector UB -> Cube L1, on chip.
// ================================================================
//
// What the hardware provides
// --------------------------
// dav-3510 has a UB->L1 channel driven by the AIV MTE3 queue. In CANN 9.2 the
// public entry point is
//
//     DataCopy(LocalTensor<T> dst /*A1|B1*/, LocalTensor<T> src /*VEC*/,
//              DataCopyParams)
//
// which dispatches to DataCopyUB2L1Impl
// (impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h:619). That
// function is declared __in_pipe__(MTE3) __out_pipe__(MTE3) and, whenever the
// AIC:AIV ratio is not 1:1, its body is a single CopyUbufToCbuf, that is the
// copy_ubuf_to_cbuf hardware move
// (impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_copy_ub2l1_impl.h).
// No TPipe, no stack buffer, no Global Memory. On a 1:1 ratio the same call
// falls back to a KFC relay through GM, so a kernel that depends on the direct
// path must declare KERNEL_TYPE_MIX_AIC_1_2 and assert the ratio.
//
// Why not DataCopy(dst, src, Nd2NzParams)
// ---------------------------------------
// The Nd2NzParams overload does exist for UB->L1, but on dav-3510 it is
// DataCopyUB2L1ND2NZImpl: it calls PopStackBuffer<T, TPosition::LCM>() and
// GetTPipePtr()->FetchEventID(), rearranges the cube inside UB with a vector
// TransND2NZ, and only then issues the same CopyUbufToCbuf. A kernel that
// places its own LocalTensors has no TPipe, so GetTPipePtr() is null and both
// accesses trap exactly as described in MICROROUTINE_DESIGN_GUIDELINES.md
// section 0. The arch-3510 variant that is free of all of that,
// DataCopyUB2L1ND2NZImplV2, is library-internal and unreachable from the
// public interface, so this leaf reimplements its burst schedule on top of the
// DataCopyParams overload. The result is the same instruction stream with none
// of the TPipe dependencies.
//
// The burst schedule
// ------------------
// ND->NZ at 32-byte block granularity is a strided placement, not an element
// permutation: a source row is cut into C0 blocks (C0 = 32 / sizeof(T)
// elements) and block column c of every row is written to one contiguous
// column strip in L1.
//
//   NValue        ND rows
//   DValue        ND columns, elements; the leaf requires 32-byte alignment
//   SrcDValue     UB row pitch, elements; 32-byte aligned
//   DstNzC0Stride L1 stride between adjacent C0 column strips, in blocks
//   DstNzNStride  L1 stride between adjacent rows inside a strip, in blocks
//
// One DataCopy per C0 column strip, each a (NValue x 1 block) strided burst.
// The strip count is a template parameter, so the emitter is a fold over a
// compile-time index sequence: no loop, no branch, no scalar arithmetic on
// device. For a panel exactly one C0 wide (DValue == C0, the 16x16 FP16 case)
// the schedule degenerates to a single contiguous NValue-block burst, which is
// one MTE3 instruction.
//
// Fractal note for the single-fractal FP16 case
// ---------------------------------------------
// For one 16x16 half fractal, Nz is row major and Zn is its transpose
// (cube_compute_fractal_intro/key_fractal_format_detailed_explanation.md gives
// Nz as (M1,N1,M0,N0) and Zn as (K1,N1,N0,K0)). Nz(X) and Zn(X transposed)
// are therefore the same bytes. A caller that needs the Zn image of K^T in L1
// hands this leaf the ND key page K[key][d] and asks for Nz placement.
//
// Ownership
// ---------
// Asynchronous leaf, like every other routine here. It issues only DataCopy.
// The caller owns the V->MTE3 edge that makes the dequantised page readable,
// the MTE3->V edge that frees the UB slot, and the cross-core flag that tells
// the Cube its L1 slot is live. The leaf issues no HardEvent and no barrier.
constexpr std::uint32_t kCvBlockBytes = 32;

// Variable templates, not constexpr functions: in the CCE dialect a plain
// constexpr function is __host__, so calling one from an __aicore__ function is
// rejected. A variable template is a compile-time value and has no such colour.
template <class T>
constexpr std::uint32_t kCvElemsPerBlock = kCvBlockBytes / sizeof(T);

// Number of C0 column strips a DValue-wide ND row is cut into.
template <class T, std::uint32_t DValue>
constexpr std::uint32_t kCvStripCount =
    (DValue + kCvElemsPerBlock<T> - 1u) / kCvElemsPerBlock<T>;

template <std::uint32_t NValue, std::uint32_t DValue, std::uint32_t SrcDValue,
          std::uint32_t DstNzC0Stride, std::uint32_t DstNzNStride, class Engine>
ASCEND_MICRO_CV_INLINE void cv_transfer_nd2nz(Engine& engine)
{
    engine.template place_nz<NValue, DValue, SrcDValue, DstNzC0Stride,
                             DstNzNStride>();
}

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)

template <class T>
struct NativeCvTransferEngine {
    // Byte offsets rather than LocalTensors, because each strip operand is
    // derived at compile time and LocalTensor subscripting is a scalar UB
    // access, which rule Z1 of the guidelines forbids inside a leaf.
    AscendC::TPosition ub_position = AscendC::TPosition::VECCALC;
    AscendC::TPosition l1_position = AscendC::TPosition::B1;
    std::uint32_t ub_base_bytes = 0;
    std::uint32_t l1_base_bytes = 0;

    static constexpr std::uint32_t kElems = kCvElemsPerBlock<T>;
    static constexpr std::uint32_t kBytes = sizeof(T);

    template <std::uint32_t NValue, std::uint32_t SrcDValue,
              std::uint32_t DstNzC0Stride, std::uint32_t DstNzNStride,
              std::uint32_t Strip>
    __aicore__ inline void emit_strip() const
    {
        AscendC::LocalTensor<T> src{
            ub_position,
            ub_base_bytes + Strip * kElems * kBytes,
            NValue * SrcDValue - Strip * kElems};
        AscendC::LocalTensor<T> dst{
            l1_position,
            l1_base_bytes + Strip * DstNzC0Stride * kCvBlockBytes,
            NValue * kElems};
        AscendC::DataCopyParams burst;
        burst.blockCount = static_cast<std::uint16_t>(NValue);
        burst.blockLen = 1;
        burst.srcStride = static_cast<std::uint16_t>(SrcDValue / kElems - 1u);
        burst.dstStride = static_cast<std::uint16_t>(DstNzNStride - 1u);
        AscendC::DataCopy(dst, src, burst);
    }

    template <std::uint32_t NValue, std::uint32_t SrcDValue,
              std::uint32_t DstNzC0Stride, std::uint32_t DstNzNStride,
              std::uint32_t... Strips>
    __aicore__ inline void emit_strips(
        std::integer_sequence<std::uint32_t, Strips...>) const
    {
        (emit_strip<NValue, SrcDValue, DstNzC0Stride, DstNzNStride, Strips>(),
         ...);
    }

    template <std::uint32_t NValue, std::uint32_t DValue,
              std::uint32_t SrcDValue, std::uint32_t DstNzC0Stride,
              std::uint32_t DstNzNStride>
    __aicore__ inline void place_nz() const
    {
        static_assert(NValue > 0 && DValue > 0 && SrcDValue > 0,
                      "ND extents must be non-zero");
        static_assert(DValue * sizeof(T) % kCvBlockBytes == 0,
                      "DValue must be a whole number of 32-byte blocks");
        static_assert(SrcDValue * sizeof(T) % kCvBlockBytes == 0,
                      "the UB row pitch must be 32-byte aligned");
        static_assert(DstNzC0Stride >= 1 && DstNzNStride >= 1,
                      "NZ strides are measured in whole blocks");
        static_assert(NValue <= 4095, "blockCount is a uint16 burst count");
        emit_strips<NValue, SrcDValue, DstNzC0Stride, DstNzNStride>(
            std::make_integer_sequence<std::uint32_t,
                                       kCvStripCount<T, DValue>>{});
    }
};

#endif

#undef ASCEND_MICRO_CV_INLINE
}  // namespace ascend_micro
