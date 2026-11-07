#pragma once

// Approach 2 -- MTE2 ingress-level layout formatting for 16-bit Cube operands
// on Ascend 950PR / arch35 / dav-3510, CANN 9.2.
//
// Why an ingress routine is needed at 16 bits and was not at 8
// ------------------------------------------------------------
// The arch-3510 fractal is 16 x (32B / sizeof(T)), so C0 = 32 for INT8 and
// C0 = 16 for bfloat16_t
// (cube_compute_fractal_intro/key_fractal_format_detailed_explanation.md).
// The two formats this kernel family uses are, element for element,
//
//   Nz(M, N):  (M, N) -> (N1, M1, M0, N0),  offset = n1*M1*M0*N0 + m1*M0*N0
//                                                    + m0*N0 + n0
//   Zn(K, N):  (K, N) -> (K1, N1, N0, K0),  offset = k1*N1*N0*K0 + n1*N0*K0
//                                                    + n0*K0 + k0
//
// with M0 = N0 = 16 and N0(Nz) = K0(Zn) = 32B/sizeof(T).
//
// A 32-wide axis is ONE fractal at 8 bits and TWO at 16. That is the whole
// difference, and it splits the INT8 kernel's two "identity" claims apart:
//
//  * The 16 x 256 activation page SURVIVES. Its Nz image is
//    offset = n1*256 + t*16 + n0 with channel = n1*16 + n0, and reading the
//    group-G slice as Zn(32, 16) asks for offset = k1*256 + t*16 + k0 with
//    channel = G*32 + k1*16 + k0. Those are the same bytes, because the page
//    has M1 = 1 (16 tokens) and the two sub-fractals of a group are adjacent
//    along the channel axis in both readings. Nothing has to change for the
//    page, at 8 bits or at 16.
//
//  * The 32 x 32 resident rotation block DOES NOT. At 8 bits a row-major
//    block is byte for byte its own Nz(32, 32) image, because K0 = 32 makes
//    the fractal a whole 32-wide row and the grid one fractal along K. At 16
//    bits the required image is
//
//      offset = (c/16)*512 + (r/16)*256 + (r%16)*16 + (c%16)        [W[r][c]]
//
//    -- four 16 x 16 sub-blocks, each row-major, ordered column-block major --
//    against row-major's r*32 + c. The two disagree at every entry outside the
//    first 16 x 16 sub-block's first row, so a flat ND->ND ingress feeds the
//    Cube a scrambled operand.
//
// The same image serves both operand roles, exactly as at 8 bits: the A-role
// Nz(32, 32) of W and the B-role Zn(32, 32) of W-as-B (B[k][n] = W[n][k])
// differ only by the name of the second index, so one ingress conversion and
// two byte-identity L1 -> L0 loads still cover the key and value chunks.
//
// `stage_blocks_nz` is therefore the fix: convert on the way in, once, in the
// prologue, and leave every downstream layout descriptor alone.
// `stage_blocks_flat` is the 8-bit behaviour, kept so a probe can run both and
// show the difference rather than assert it.

#include <cstdint>
#include <utility>

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#endif

namespace ascend_micro {

// C0 in elements for a 16-bit type: 32 bytes / 2.
constexpr std::uint32_t kBf16C0 = 16u;
constexpr std::uint32_t kBf16ElemBytes = 2u;

// The element offset of row-major (r, c) inside the Nz(Dim, Dim) image of a
// 16-bit square block. A host runner that pre-formats the blob and a host
// oracle that predicts the device's operand need exactly this formula, so it
// is defined outside the device-only section.
constexpr std::uint32_t Bf16NzBlockIndex(std::uint32_t dim, std::uint32_t r,
                                         std::uint32_t c)
{
    return (c / kBf16C0) * (dim * kBf16C0) + (r / kBf16C0) * (kBf16C0 * kBf16C0) +
           (r % kBf16C0) * kBf16C0 + (c % kBf16C0);
}

#if defined(__CCE_AICORE__)

// ------------------------------------------------------------------ the page
// GM -> L1, ND -> Nz, one strided burst. `row_pitch` is the activation block's
// full channel count, so the copy reads the headSize-wide window of one head
// out of a token-major row; that window is not the whole row, which is why the
// source is a pattern layout and not MakeFrameLayout's NDExt, whose row pitch
// is pinned to the column count.
//
// Unchanged from the 8-bit kernel on purpose: the page's Nz image is already
// the Zn image of its transpose at 16 bits, so there is nothing here for an
// ingress routine to repair. It is factored out so a probe can stage the page
// exactly the way production does.
template <std::uint32_t Tokens, std::uint32_t Channels>
__aicore__ inline void stage_page_nz(__gm__ const bfloat16_t* src,
                                     std::uint32_t l1_byte_base,
                                     std::uint32_t row_pitch)
{
    using namespace AscendC::Te;
    static_assert(Tokens % kBf16C0 == 0u, "a page must be a whole number of Nz rows");
    static_assert(Channels % kBf16C0 == 0u, "a page must be a whole number of fractals");

    auto* dst_ptr = reinterpret_cast<__cbuf__ bfloat16_t*>(l1_byte_base);
    auto shape = MakeShape(MakeShape(_1{}, Tokens), MakeShape(_1{}, Channels));
    auto stride = MakeStride(MakeStride(_0{}, row_pitch), MakeStride(_0{}, _1{}));
    auto s = MakeTensor(
        MakeMemPtr(const_cast<__gm__ bfloat16_t*>(src)),
        MakePatternLayout<NDExtLayoutPtn,
                          LayoutTrait<AscendC::Std::ignore_t, _1>>(shape, stride));
    auto d = MakeTensor(MakeMemPtr(dst_ptr),
                        MakeFrameLayout<NZLayoutPtn, bfloat16_t>(Tokens, Channels));
    Copy(MakeCopy(CopyGM2L1{}), d, s);
}

// -------------------------------------------------- the resident block blob
// GM -> L1, ND -> Nz, one conversion per Dim x Dim block. Block b lands at
// l1_byte_base + b * Dim * Dim * 2, which is where a flat blob would have put
// it, so only the bytes inside a block move and every downstream base address
// is unchanged.
//
// One Copy per block rather than one batched Copy over all of them. The
// batched form describes the blob as (B, (Dim, Dim)) and derives the
// destination spacing as "one cube after alignment", which happens to equal
// Dim*Dim*2 here but is a derived quantity rather than a stated one. The
// blocks are staged once in the prologue, so the extra MTE2 descriptors cost
// nothing worth that coupling.
template <std::uint32_t Dim, std::uint32_t B>
__aicore__ inline void StageBlockNz(__gm__ const bfloat16_t* src,
                                    std::uint32_t l1_byte_base)
{
    using namespace AscendC::Te;
    constexpr std::uint32_t elems = Dim * Dim;
    auto* d_ptr = reinterpret_cast<__cbuf__ bfloat16_t*>(
        l1_byte_base + B * elems * kBf16ElemBytes);
    auto s = MakeTensor(MakeMemPtr(const_cast<__gm__ bfloat16_t*>(src) + B * elems),
                        MakeFrameLayout<NDExtLayoutPtn>(Dim, Dim));
    auto d = MakeTensor(MakeMemPtr(d_ptr),
                        MakeFrameLayout<NZLayoutPtn, bfloat16_t>(Dim, Dim));
    Copy(MakeCopy(CopyGM2L1{}), d, s);
}

template <std::uint32_t Dim, std::uint32_t... Bs>
__aicore__ inline void StageBlocksNzEmit(__gm__ const bfloat16_t* src,
                                         std::uint32_t l1_byte_base,
                                         std::integer_sequence<std::uint32_t, Bs...>)
{
    (StageBlockNz<Dim, Bs>(src, l1_byte_base), ...);
}

template <std::uint32_t Blocks, std::uint32_t Dim>
__aicore__ inline void stage_blocks_nz(__gm__ const bfloat16_t* src,
                                       std::uint32_t l1_byte_base)
{
    static_assert(Dim % kBf16C0 == 0u, "a resident block must tile the 16-wide fractal");
    static_assert((Dim * Dim * kBf16ElemBytes) % 512u == 0u,
                  "a resident block must stay 512-byte aligned in L0");
    StageBlocksNzEmit<Dim>(src, l1_byte_base,
                           std::make_integer_sequence<std::uint32_t, Blocks>{});
}

// The 8-bit ingress: one flat ND -> ND copy of the whole blob. Correct while
// C0 == 32, wrong at 16 bits, and kept so that the difference is something a
// probe measures rather than something a comment claims.
template <std::uint32_t Blocks, std::uint32_t Dim>
__aicore__ inline void stage_blocks_flat(__gm__ const bfloat16_t* src,
                                         std::uint32_t l1_byte_base)
{
    using namespace AscendC::Te;
    constexpr std::uint32_t rows = Blocks * Dim;
    auto* d_ptr = reinterpret_cast<__cbuf__ bfloat16_t*>(l1_byte_base);
    auto s = MakeTensor(MakeMemPtr(const_cast<__gm__ bfloat16_t*>(src)),
                        MakeFrameLayout<NDExtLayoutPtn>(rows, Dim));
    auto d = MakeTensor(MakeMemPtr(d_ptr), MakeFrameLayout<NDExtLayoutPtn>(rows, Dim));
    Copy(MakeCopy(CopyGM2L1{}), d, s);
}

#endif  // __CCE_AICORE__

}  // namespace ascend_micro
