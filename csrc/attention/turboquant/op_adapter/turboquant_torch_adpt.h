/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */
#ifndef TURBOQUANT_TORCH_ADPT_H
#define TURBOQUANT_TORCH_ADPT_H

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vllm_ascend {

// The BF16 Cube ingestion kernel, built into libvllm_ascend_turboquant by the
// ASC toolchain for dav-3510. One launch of `block_dim` strips; the window loop
// below owns the grid.
extern "C" void turboquant_reshape_and_cache_bf16_impl(
    void *stream, std::uint32_t block_dim, const std::uint16_t *key, const std::uint16_t *value,
    std::int8_t *key_cache, std::int8_t *value_cache, float *scale_cache,
    const std::int32_t *slot_mapping, const float *pi_signs, const std::int32_t *tables,
    std::uint32_t num_tokens, std::uint32_t num_kv_heads, std::uint32_t head_size,
    std::uint32_t block_size, std::uint32_t num_blocks, std::uint32_t tokens_per_core,
    float inv_sqrt_len);

namespace turboquant {

// The kernel's strip height, and the only shape constant that is compile time
// *inside* the kernel rather than a launch argument: the DaVinci fractal is
// M0 = 16, so a strip is 16 tokens and the whole four-stage schedule is
// specialised to it. Everything else below is a runtime contract.
constexpr std::int64_t kTokensPerStrip = 16;

// Strips per launch. A strip's Cube core and its two Vector cores hand each
// other ten mode-4 intra-block flag ids, and those ids are a per-sub-block
// resource (0..15 for AIV0, 16..31 for AIV1, with 11..14 library reserved), not
// a queue that drains: a launch that puts more strips in flight than the id
// space holds has two strips announcing on one id, where the second
// announcement is released by the first wait and the scheduler then spins on a
// token nobody will set. 32 strips is the measured ceiling the harness ran
// clean (reports: verified on CAModel at 2 and 32 strips), so a wide context is
// driven as several launches of at most this many rather than one wide grid.
constexpr std::int64_t kStripsPerLaunch = 32;

// Every TurboQuant GM operand base must sit on a 32-byte burst boundary. A
// violation surfaces from CANN as error 264 rather than as an alignment report,
// so it is worth naming here.
constexpr std::int64_t kGmBurstBytes = 32;

inline void check_burst_aligned(const at::Tensor &t, const char *name)
{
    const auto address = reinterpret_cast<std::uintptr_t>(t.data_ptr());
    TORCH_CHECK(address % static_cast<std::uintptr_t>(kGmBurstBytes) == 0,
                "npu_turboquant_reshape_and_cache: ", name, " base address ", address,
                " is not ", kGmBurstBytes, "-byte aligned");
}

} // namespace turboquant

// TurboQuant BF16 prefill ingestion: rotate K and V by the randomised Hadamard
// Pi = (H8 (x) I32) . diag(piSigns) . (I8 (x) H32) on the Cube, quantise to
// INT4 with a power-of-two per-token scale on both Vector cores, and scatter
// the packed planes and their fp32 scale slice into the paged cache.
//
// The three cache planes are the ones AscendTurboQuantAttentionSpec budgets out
// of one page: packed keys, packed values, and the burst-aligned fp32 scales.
inline void npu_turboquant_reshape_and_cache(at::Tensor &key, at::Tensor &value,
                                             at::Tensor &key_cache, at::Tensor &value_cache,
                                             at::Tensor &scale_cache, at::Tensor &slot_mapping,
                                             at::Tensor &pi_signs, at::Tensor &tables)
{
    using namespace vllm_ascend::turboquant;

    // ---- activation contract -------------------------------------------------
    TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
                "npu_turboquant_reshape_and_cache: key and value must be "
                "[num_tokens, num_kv_heads, head_size]");
    TORCH_CHECK(key.sizes() == value.sizes(),
                "npu_turboquant_reshape_and_cache: key and value must have the same shape");
    TORCH_CHECK(key.scalar_type() == at::kBFloat16 && value.scalar_type() == at::kBFloat16,
                "npu_turboquant_reshape_and_cache: this is the BF16 ingestion kernel; key and "
                "value must be bfloat16, got ",
                key.scalar_type(), " and ", value.scalar_type());
    // A fused QKV projection hands over column slices, so the writer would
    // otherwise read a strided view as if it were packed.
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous(),
                "npu_turboquant_reshape_and_cache: key and value must be contiguous");

    const std::int64_t num_tokens = key.size(0);
    const std::int64_t num_kv_heads = key.size(1);
    const std::int64_t head_size = key.size(2);

    // The strip geometry the kernel is specialised to. These are the kernel's
    // own compile-time constants (kTokens, kHeads, kHeadSize), so they are
    // checked rather than passed through as free parameters.
    TORCH_CHECK(num_tokens % kTokensPerStrip == 0,
                "npu_turboquant_reshape_and_cache: num_tokens (", num_tokens,
                ") must be a multiple of the ", kTokensPerStrip, "-token strip");
    TORCH_CHECK(num_kv_heads == 4,
                "npu_turboquant_reshape_and_cache: the kernel is specialised to 4 KV heads, got ",
                num_kv_heads);
    TORCH_CHECK(head_size == 256,
                "npu_turboquant_reshape_and_cache: the kernel is specialised to head_size 256, "
                "got ",
                head_size);

    // ---- cache contract ------------------------------------------------------
    TORCH_CHECK(key_cache.scalar_type() == at::kChar && value_cache.scalar_type() == at::kChar,
                "npu_turboquant_reshape_and_cache: the packed planes are int8 nibble pairs");
    TORCH_CHECK(scale_cache.scalar_type() == at::kFloat,
                "npu_turboquant_reshape_and_cache: the scale plane is fp32");
    TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous() &&
                    scale_cache.is_contiguous(),
                "npu_turboquant_reshape_and_cache: every cache plane must be contiguous");

    // [num_blocks, block_size, num_kv_heads, head_size / 2]
    TORCH_CHECK(key_cache.dim() == 4 && key_cache.sizes() == value_cache.sizes(),
                "npu_turboquant_reshape_and_cache: the packed planes must both be "
                "[num_blocks, block_size, num_kv_heads, head_size / 2]");
    const std::int64_t num_blocks = key_cache.size(0);
    const std::int64_t block_size = key_cache.size(1);
    TORCH_CHECK(key_cache.size(2) == num_kv_heads && key_cache.size(3) == head_size / 2,
                "npu_turboquant_reshape_and_cache: packed plane is [", num_blocks, ", ",
                block_size, ", ", key_cache.size(2), ", ", key_cache.size(3),
                "], expected [..., ", num_kv_heads, ", ", head_size / 2, "]");

    // The one contract that replaces "a strip fills a whole page". A page may
    // hold any whole number of strips -- 8 at the vLLM default block_size 128 --
    // and run_vector_core places the strip at its in-page offset. What a page
    // may not be is a non-multiple of the strip, which would put a strip across
    // two pages.
    TORCH_CHECK(block_size % kTokensPerStrip == 0,
                "npu_turboquant_reshape_and_cache: block_size (", block_size,
                ") must be a multiple of the ", kTokensPerStrip, "-token strip");

    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt,
                "npu_turboquant_reshape_and_cache: slot_mapping must be int32");
    TORCH_CHECK(slot_mapping.is_contiguous() && slot_mapping.dim() == 1 &&
                    slot_mapping.size(0) == num_tokens,
                "npu_turboquant_reshape_and_cache: slot_mapping must be a contiguous int32 "
                "[num_tokens]");

    // The scale plane is [num_blocks, block_size, scale_slot], scale_slot being
    // 2 * num_kv_heads rounded up to one 32-byte burst of fp32 lanes. The
    // kernel indexes it as ((kv * num_blocks + block) * num_kv_heads + head) *
    // block_size + key, so what it needs from the host is that the plane is at
    // least that large.
    const std::int64_t scale_floats_needed = 2 * num_blocks * num_kv_heads * block_size;
    TORCH_CHECK(scale_cache.numel() >= scale_floats_needed,
                "npu_turboquant_reshape_and_cache: scale plane holds ", scale_cache.numel(),
                " floats, needs at least ", scale_floats_needed);

    // ---- rotation tables -----------------------------------------------------
    // 16 row-major 32 x 32 blocks: T[s][g] of the kernel's equation (2), the
    // sign diagonal already folded in. piSigns stays in the published signature
    // and is not read by the device -- whoever builds `tables` consumed it.
    constexpr std::int64_t kTableElems = 16 * 32 * 32;
    TORCH_CHECK(tables.scalar_type() == at::kBFloat16,
                "npu_turboquant_reshape_and_cache: tables must be bfloat16 (+-1 entries)");
    TORCH_CHECK(tables.is_contiguous() && tables.numel() == kTableElems,
                "npu_turboquant_reshape_and_cache: tables must be ", kTableElems,
                " contiguous bfloat16 elements (16 row-major 32x32 blocks), got ",
                tables.numel());

    check_burst_aligned(key, "key");
    check_burst_aligned(value, "value");
    check_burst_aligned(key_cache, "key_cache");
    check_burst_aligned(value_cache, "value_cache");
    check_burst_aligned(scale_cache, "scale_cache");
    check_burst_aligned(tables, "tables");

    // invSqrtLen folds into the stored scale, never into the nibble: scaling Y
    // and absmax together leaves every quantised value unchanged, which is what
    // keeps the device bit-exact against an integer host oracle.
    const float inv_sqrt_len = 1.0f / std::sqrt(static_cast<float>(head_size));

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    const auto *key_base = static_cast<const std::uint16_t *>(key.data_ptr());
    const auto *value_base = static_cast<const std::uint16_t *>(value.data_ptr());
    const auto *slot_base = static_cast<const std::int32_t *>(slot_mapping.data_ptr());
    // piSigns is folded into `tables` and the device never reads it, but it
    // stays in the published signature. An empty tensor is the honest way for a
    // caller to say so, and must not become a bogus pointer.
    const float *pi_signs_ptr =
        pi_signs.numel() > 0 ? static_cast<const float *>(pi_signs.data_ptr()) : nullptr;

    // ---- the window-32 loop --------------------------------------------------
    // Strips per launch is capped at kStripsPerLaunch so the mode-4 flag ids
    // cannot be exhausted. Between launches the three per-token bases advance
    // together: key and value by whole strips of num_kv_heads * head_size
    // elements (the pointers are typed, so this is element arithmetic), and
    // slot_mapping by the token count. The cache planes and the tables do not
    // move -- the cache is addressed from the strip's own first slot, and the
    // tables are page invariant.
    const std::int64_t total_strips = num_tokens / kTokensPerStrip;
    const std::int64_t tokens_per_strip_row = num_kv_heads * head_size;

    for (std::int64_t strip = 0; strip < total_strips; strip += kStripsPerLaunch) {
        const std::int64_t strips_this_launch = std::min(kStripsPerLaunch, total_strips - strip);
        const std::int64_t tokens_this_launch = strips_this_launch * kTokensPerStrip;
        const std::int64_t token_offset = strip * kTokensPerStrip;

        turboquant_reshape_and_cache_bf16_impl(
            stream, static_cast<std::uint32_t>(strips_this_launch),
            key_base + token_offset * tokens_per_strip_row,
            value_base + token_offset * tokens_per_strip_row,
            static_cast<std::int8_t *>(key_cache.data_ptr()),
            static_cast<std::int8_t *>(value_cache.data_ptr()),
            static_cast<float *>(scale_cache.data_ptr()), slot_base + token_offset,
            pi_signs_ptr, static_cast<const std::int32_t *>(tables.data_ptr()),
            static_cast<std::uint32_t>(tokens_this_launch),
            static_cast<std::uint32_t>(num_kv_heads), static_cast<std::uint32_t>(head_size),
            static_cast<std::uint32_t>(block_size), static_cast<std::uint32_t>(num_blocks),
            static_cast<std::uint32_t>(kTokensPerStrip), inv_sqrt_len);
    }
}

} // namespace vllm_ascend

#endif
