#pragma once

#include <cstdint>
#include <utility>

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#endif

namespace ascend_micro {

constexpr std::uint32_t kBf16C0 = 16u;
constexpr std::uint32_t kBf16ElemBytes = 2u;

constexpr std::uint32_t Bf16NzBlockIndex(std::uint32_t dim, std::uint32_t r,
                                         std::uint32_t c)
{
    return (c / kBf16C0) * (dim * kBf16C0) + (r / kBf16C0) * (kBf16C0 * kBf16C0) +
           (r % kBf16C0) * kBf16C0 + (c % kBf16C0);
}

#if defined(__CCE_AICORE__)

template <std::uint32_t Tokens, std::uint32_t Channels>
__aicore__ inline void stage_page_nz(__gm__ const bfloat16_t* src,
                                     std::uint32_t l1_byte_base,
                                     std::uint32_t row_pitch)
{
    using namespace asc::te;
    static_assert(Tokens % kBf16C0 == 0u, "a page must be a whole number of Nz rows");
    static_assert(Channels % kBf16C0 == 0u, "a page must be a whole number of fractals");

    auto* dst_ptr = reinterpret_cast<__cbuf__ bfloat16_t*>(l1_byte_base);
    auto shape = make_shape(make_shape(_1{}, Tokens), make_shape(_1{}, Channels));
    auto stride = make_stride(make_stride(_0{}, row_pitch), make_stride(_0{}, _1{}));
    auto s = make_tensor(
        make_mem_ptr(const_cast<__gm__ bfloat16_t*>(src)),
        make_pattern_layout<nd_ext_layout_ptn,
                          layout_trait<AscendC::Std::ignore_t, _1>>(shape, stride));
    auto d = make_tensor(make_mem_ptr(dst_ptr),
                        make_frame_layout<nz_layout_ptn, bfloat16_t>(Tokens, Channels));
    copy(make_copy(copy_gm_to_l1{}), d, s);
}

template <std::uint32_t Dim, std::uint32_t B>
__aicore__ inline void StageBlockNz(__gm__ const bfloat16_t* src,
                                    std::uint32_t l1_byte_base)
{
    using namespace asc::te;
    constexpr std::uint32_t elems = Dim * Dim;
    auto* d_ptr = reinterpret_cast<__cbuf__ bfloat16_t*>(
        l1_byte_base + B * elems * kBf16ElemBytes);
    auto s = make_tensor(make_mem_ptr(const_cast<__gm__ bfloat16_t*>(src) + B * elems),
                        make_frame_layout<nd_ext_layout_ptn>(Dim, Dim));
    auto d = make_tensor(make_mem_ptr(d_ptr),
                        make_frame_layout<nz_layout_ptn, bfloat16_t>(Dim, Dim));
    copy(make_copy(copy_gm_to_l1{}), d, s);
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

template <std::uint32_t Blocks, std::uint32_t Dim>
__aicore__ inline void stage_blocks_flat(__gm__ const bfloat16_t* src,
                                         std::uint32_t l1_byte_base)
{
    using namespace asc::te;
    constexpr std::uint32_t rows = Blocks * Dim;
    auto* d_ptr = reinterpret_cast<__cbuf__ bfloat16_t*>(l1_byte_base);
    auto s = make_tensor(make_mem_ptr(const_cast<__gm__ bfloat16_t*>(src)),
                        make_frame_layout<nd_ext_layout_ptn>(rows, Dim));
    auto d = make_tensor(make_mem_ptr(d_ptr), make_frame_layout<nd_ext_layout_ptn>(rows, Dim));
    copy(make_copy(copy_gm_to_l1{}), d, s);
}

#endif

}
