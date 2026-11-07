/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

// The TurboQuant Torch bindings: the three audited kernel paths of
// kernels/ops/turboquant/vllm_integration_layer, registered into
// torch.ops.turboquant_ops so vllm_ascend/attention/turboquant_dry_run.py can
// dispatch a stage to the NPU instead of its PyTorch golden model.
//
// WHY A TRANSLATION UNIT OF ITS OWN, and not another include of
// csrc/torch_binding.cpp. Two reasons, and the first one is a hard build
// error. op_kernel/common/types.h declares `vllm_ascend::AscendType` and so
// does csrc/kernels/types.h; both are `#pragma once` at different paths, so a
// TU that reaches both gets a redefinition. csrc/torch_binding.cpp reaches the
// second through csrc/ops.h, so the TurboQuant adapter cannot live there any
// more now that it pulls the audited `types.h` in. The second reason is that
// the three ops here are gated on one SoC and one CANN version: keeping them
// in one file means the gate is the file's presence in the source list, not a
// set of #ifdef regions threaded through a 3700-line binding.
//
// THE NAMESPACE IS `turboquant_ops`, which is TURBOQUANT_DEFAULT_NAMESPACE in
// vllm_ascend/attention/turboquant_dry_run.py. The adapter there reads
// `getattr(torch.ops, config.ops_namespace)` and then `getattr(ns, op_name)`,
// so a missing registration is an AttributeError the adapter turns into a
// warn-once and a golden fallback. That is the whole fallback contract: the
// Python side never needs to know whether this file was compiled.
//
// WHAT THESE THREE KERNELS DO NOT SHARE: A CACHE FORMAT. This is the single
// most important thing to know before enabling a stage, and it is the harness's
// Phase 1 position, not a defect introduced here (vllm_integration_layer's
// README: "The batch writer and scatter writer use the harness's BF16
// signed-int4 strip cache ... Phase 1 does not merge these formats or alter
// their interfaces").
//
// The difference is NOT a byte ordering that a reordering could fix. It is a
// different PAIRING OF NIBBLES INTO BYTES, which no permutation of bytes can
// undo:
//
//   * the two WRITERS: the key plane's nibble index inside a (block, head) page
//     is `token * head_size + channel`; the value plane's is
//     `(token / 16) * (16 * head_size) + channel * 16 + (token % 16)`
//     (op_kernel/turboquant_reshape_scatter.cpp's StripWordOffset plus its
//     within-strip index; the batch writer's "Layout contracts" block states
//     the same thing as `ch * blockSize + key` for its one-strip page).
//     Consecutive nibbles are the SAME CHANNEL at TWO TOKENS, so a byte holds
//     one channel of tokens 2p and 2p+1.
//   * the FUSED DECODE reads the NZ-tiled plane of
//     op_kernel/common/turboquant_layout.h's NzTiledPackedByte(), laid out as
//     [64-row tile][kv head][32-byte column group][tile row][byte]. A byte
//     holds ONE TOKEN at channels `col` and `col + 128` -- the two halves of
//     the channel axis. The harness's own reference writer for this plane is
//     tests/common/fused_decode_opt_fixture.hpp, which fills it as
//     `value[NzTiledPackedByte(row, head, col, h, 128)] =
//      level(col) | (level(col + 128) << 4)`.
//
// At block_size 64 both layouts occupy the same 8192 bytes per (tile, kv head)
// and ZERO of those 8192 bytes hold the same pair of nibbles; the multisets of
// pairs differ, so a pure byte permutation between them does not exist. At
// block_size 16 the NZ plane does not even fit: its tile row stride is 64, so a
// 16-token page would address 6656 bytes to store 2048.
//
// Why that is fatal rather than cosmetic: StageNzTileToL1 nibble-expands with
// DeInterleave (op_kernel/common/turboquant_codec_mx.h, UnpackAffine), which
// sends every byte's low nibble to one contiguous L1 plane and every high
// nibble to another. Under the NZ plane the low plane is channels 0..127 of all
// 64 tokens -- a fractal-ordered K x N tile, which is what the context GEMM's
// L0B load then transposes (turboquant_cube_service.h: "the context GEMM's
// value tile is K x N and the load transposes it", ifTranspose = true). Under
// the strip plane the low plane is instead ALL THE EVEN TOKENS. The ZN
// orientation the Cube needs is imposed at that L1 -> L0B load, so GM does not
// have to be "already ZN" -- but it does have to deposit the right K x N tile
// in L1, and a channel-major strip deposits a token-parity permutation of it.
// The decode would contract probabilities of token t against the value of a
// different token: a wrong answer with every shape check satisfied.
//
// So no single vLLM KV cache can be written by these writers and read by this
// decode, and the obstacle is numerical rather than dimensional. Each op
// therefore TORCH_CHECKs its own hardware contract -- not a cross-format
// refusal, just what its own addressing admits -- and those contracts happen to
// have no common block_size. Merging the formats is kernel work in the harness.
// The evidence above is reproducible: see the tabulation in the hand-over note.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>

#include <torch/library.h>
#include <torch/torch.h>

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "include/turboquant_fused_decode.hpp"
#include "op_adapter/turboquant_torch_adpt.h"
#include "op_kernel/common/turboquant_layout.h"

namespace vllm_ascend {
namespace turboquant_bindings {

namespace layout = ::vllm_ascend::turboquant;
namespace tiling = ::ascend_micro::fused_decode;

// The batch writer's strip height: kTokens in op_kernel/turboquant_reshape_batch.cpp,
// where it is a compile-time constant of the four-stage schedule and of the
// 16 x 256 L1 page the Cube contracts. The host cannot choose it, which is why
// `tokens_per_core` is derived here and is not in any schema below.
constexpr std::int64_t kTokensPerStrip = 16;

// The fused decode's hard geometry, from valid_launch() in
// include/turboquant_fused_decode_opt.hpp. Restated rather than #included as
// expressions because these are the numbers a TORCH_CHECK message has to name.
constexpr std::int64_t kDecodeHeadSize = 256;
constexpr std::int64_t kDecodeMaxKvHeads = 4;
constexpr std::int64_t kDecodeMaxGrid = 32;
constexpr std::int64_t kDecodeMaxSplits = 8;
constexpr std::int64_t kDecodeMaxHeadsPerTask = 16;

// Every TurboQuant GM operand base sits on a 32-byte burst boundary. CANN
// reports a violation as error 264 rather than as an alignment report, so it is
// worth naming at the seam.
constexpr std::uintptr_t kGmBurstBytes = 32;

// 2 * (head_size / 32) row-major 32x32 INT8 blocks: T[s][g] of the rotation,
// with the sign diagonal pre-composed. 16384 bytes at head_size 256, which is
// what `_codec_tables` in turboquant_dry_run.py builds.
constexpr std::int64_t kTableGroup = 32;
constexpr std::int64_t kTablePlanes = 2;

inline std::int64_t ceil_div(const std::int64_t a, const std::int64_t b)
{
    return b > 0 ? (a + b - 1) / b : 0;
}

inline void check_burst_aligned(const at::Tensor &t, const char *op, const char *name)
{
    const auto address = reinterpret_cast<std::uintptr_t>(t.data_ptr());
    TORCH_CHECK(address % kGmBurstBytes == 0, op, ": ", name, " base address ", address, " is not ",
                kGmBurstBytes, "-byte aligned");
}

// The activation pair every ingestion op takes: [num_tokens, num_kv_heads, head_size] bf16.
struct Activations {
    std::int64_t num_tokens;
    std::int64_t num_kv_heads;
    std::int64_t head_size;
};

inline Activations check_activations(const at::Tensor &key, const at::Tensor &value, const char *op)
{
    TORCH_CHECK(key.dim() == 3 && value.dim() == 3, op,
                ": key and value must be [num_tokens, num_kv_heads, head_size]");
    TORCH_CHECK(key.sizes() == value.sizes(), op, ": key and value must have the same shape");
    TORCH_CHECK(key.scalar_type() == at::kBFloat16 && value.scalar_type() == at::kBFloat16, op,
                ": both writers are the BF16 ingestion path; key and value must be bfloat16, got ",
                key.scalar_type(), " and ", value.scalar_type());
    // A fused QKV projection hands over column slices, so the writer would
    // otherwise read a strided view as if it were packed.
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous(), op, ": key and value must be contiguous");
    return Activations{key.size(0), key.size(1), key.size(2)};
}

// The three cache planes, as AscendTurboQuantAttentionSpec carves them:
// two INT8 packed planes [num_blocks, block_size, num_kv_heads, head_size / 2]
// and one fp32 scale plane [2, num_blocks, num_kv_heads, block_size].
struct CachePlanes {
    std::int64_t num_blocks;
    std::int64_t block_size;
};

inline CachePlanes check_cache(const at::Tensor &key_cache, const at::Tensor &value_cache,
                               const at::Tensor &scale_cache, const Activations &a, const char *op)
{
    TORCH_CHECK(key_cache.scalar_type() == at::kChar && value_cache.scalar_type() == at::kChar, op,
                ": the packed planes are int8 nibble pairs, got ", key_cache.scalar_type(), " and ",
                value_cache.scalar_type());
    TORCH_CHECK(scale_cache.scalar_type() == at::kFloat, op, ": the scale plane is fp32, got ",
                scale_cache.scalar_type());
    TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous() && scale_cache.is_contiguous(), op,
                ": every cache plane must be contiguous");
    TORCH_CHECK(key_cache.dim() == 4 && key_cache.sizes() == value_cache.sizes(), op,
                ": the packed planes must both be [num_blocks, block_size, num_kv_heads, head_size / 2]");

    const std::int64_t num_blocks = key_cache.size(0);
    const std::int64_t block_size = key_cache.size(1);
    TORCH_CHECK(key_cache.size(2) == a.num_kv_heads && key_cache.size(3) == a.head_size / 2, op,
                ": packed plane is [", num_blocks, ", ", block_size, ", ", key_cache.size(2), ", ",
                key_cache.size(3), "], expected [..., ", a.num_kv_heads, ", ", a.head_size / 2, "]");

    // The kernels index the scale plane as
    //   ((kv * num_blocks + block) * num_kv_heads + head) * block_size + token
    // with kv 0 for keys and 1 for values, so the two halves are planes over
    // the whole cache rather than two fields of a page. The highest lane
    // written is this bound minus one, so the check is exact.
    const std::int64_t scale_floats = kTablePlanes * num_blocks * a.num_kv_heads * block_size;
    TORCH_CHECK(scale_cache.numel() >= scale_floats, op, ": scale plane holds ", scale_cache.numel(),
                " floats, needs at least ", scale_floats);
    return CachePlanes{num_blocks, block_size};
}

inline void check_slot_mapping(const at::Tensor &slot_mapping, const std::int64_t expected, const char *op)
{
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, op, ": slot_mapping must be int32, got ",
                slot_mapping.scalar_type());
    TORCH_CHECK(slot_mapping.is_contiguous() && slot_mapping.dim() == 1 && slot_mapping.size(0) == expected, op,
                ": slot_mapping must be a contiguous int32 [", expected, "], got ", slot_mapping.sizes());
}

inline void check_tables(const at::Tensor &tables, const std::int64_t head_size, const char *op)
{
    const std::int64_t groups = head_size / kTableGroup;
    const std::int64_t expected = kTablePlanes * groups * kTableGroup * kTableGroup;
    TORCH_CHECK(tables.scalar_type() == at::kChar, op,
                ": tables must be int8 (the Cube's resident INT8 operand), got ", tables.scalar_type());
    TORCH_CHECK(tables.is_contiguous() && tables.numel() == expected, op, ": tables must be ", expected,
                " contiguous int8 elements (", kTablePlanes * groups, " row-major 32x32 blocks), got ",
                tables.numel());
}

// `pi_signs` stays in every published signature and the device does not read it
// on the ingestion paths: whoever built `tables` consumed it. An empty tensor is
// the honest way for a caller to say so, and must not become a bogus pointer.
inline const float *optional_pi_signs(const at::Tensor &pi_signs)
{
    return pi_signs.numel() > 0 ? static_cast<const float *>(pi_signs.data_ptr()) : nullptr;
}

// VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS, read once per process and registered in
// vllm_ascend/envs.py. The slot values live on the device, so validating them
// costs a device-to-host copy and a synchronisation on every cache write of
// every layer -- diagnostic only, never a serving default, which is why it is
// opt-in. A host-side read of the device pointer would be a bus error, not a
// check.
inline bool validate_slots()
{
    static const bool enabled = [] {
        const char *value = std::getenv("VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS");
        return value != nullptr && value[0] == '1' && value[1] == 0;
    }();
    return enabled;
}

// The batch writer's slot contract, checked only under the env above.
//
// Each strip's first slot is read on the scalar unit and both the page and the
// strip inside it are derived from it, so a strip whose slots are not
// strip-aligned and consecutive silently writes the wrong place rather than
// failing. Alignment is also what keeps a strip inside one page: with
// `first % kTokensPerStrip == 0` and `block_size % kTokensPerStrip == 0` the
// strip's 16 tokens cannot straddle a block boundary, so no separate crossing
// check is needed.
inline void check_strip_slots(const at::Tensor &slot_mapping, const std::int64_t num_tokens, const char *op)
{
    if (!validate_slots()) {
        return;
    }
    const at::Tensor host_slots = slot_mapping.to(at::kCPU).contiguous();
    const auto *slots = static_cast<const std::int32_t *>(host_slots.data_ptr());
    for (std::int64_t strip = 0; strip < num_tokens / kTokensPerStrip; ++strip) {
        const std::int64_t base = strip * kTokensPerStrip;
        const std::int32_t first = slots[base];
        TORCH_CHECK(first >= 0, op, ": slot_mapping[", base, "] is ", first,
                    "; the batch writer cannot skip a padded strip");
        TORCH_CHECK(first % kTokensPerStrip == 0, op, ": strip ", strip, " starts at slot ", first,
                    ", which is not ", kTokensPerStrip, "-aligned");
        for (std::int64_t i = 1; i < kTokensPerStrip; ++i) {
            const std::int32_t want = first + static_cast<std::int32_t>(i);
            TORCH_CHECK(slots[base + i] == want, op, ": strip ", strip, " is not consecutive: slot_mapping[",
                        base + i, "] is ", slots[base + i], ", expected ", want);
        }
    }
}

// ---------------------------------------------------------------------------
// 1. Prefill batch ingestion
// ---------------------------------------------------------------------------

// Rotate, quantise, pack and write whole physical pages, over
// launch_turboquant_reshape_and_cache_bf16 (op_kernel/turboquant_reshape_batch.cpp).
//
// The launcher computes its own grid as num_tokens / tokens_per_core and
// refuses a partial trailing strip, so the only host derivation left here is
// tokens_per_core itself -- pinned to the kernel's compile-time 16-token strip,
// which is why it is not a schema argument. The grid is therefore
// num_tokens / 16 and spans as many execution waves as it needs; the kernel's
// boundary guard and inter-wave epilogue drain are what make one monolithic
// dispatch safe.
void npu_turboquant_reshape_and_cache_a8(at::Tensor &key, at::Tensor &value, at::Tensor &key_cache,
                                         at::Tensor &value_cache, at::Tensor &scale_cache,
                                         at::Tensor &slot_mapping, at::Tensor &pi_signs, at::Tensor &tables,
                                         double inv_sqrt_len, bool reuse_mode4_flags)
{
    constexpr const char *op = "npu_turboquant_reshape_and_cache_a8";

    const Activations a = check_activations(key, value, op);
    const CachePlanes c = check_cache(key_cache, value_cache, scale_cache, a, op);
    check_slot_mapping(slot_mapping, a.num_tokens, op);
    check_tables(tables, a.head_size, op);

    // The kernel's compile-time specialisation (kTokens, kHeads, kHeadSize in
    // op_kernel/turboquant_reshape_batch.cpp). The scalar arguments drive the
    // physical cache addressing only, so a geometry outside these is not a slow
    // path -- it is a wrong answer with every canary intact.
    TORCH_CHECK(a.num_tokens % kTokensPerStrip == 0, op, ": num_tokens (", a.num_tokens,
                ") must be a whole number of ", kTokensPerStrip,
                "-token strips; the launcher drops a partial trailing strip rather than writing it");
    TORCH_CHECK(a.num_kv_heads == 4, op, ": the batch writer is specialised to 4 KV heads, got ",
                a.num_kv_heads);
    TORCH_CHECK(a.head_size == kDecodeHeadSize, op, ": the batch writer is specialised to head_size ",
                kDecodeHeadSize, ", got ", a.head_size);
    // kTokens == blockSize in the kernel: a prefill strip fills one whole
    // physical page, and the value plane's channel-major nibble index
    // `channel * block_size + token` is what makes that a 2048-byte burst.
    // Any whole number of strips per page, same as the scatter writer. This one
    // used to need block_size == 16: its only address was a page base, so at
    // block_size 64 all four strips of a page computed the same base and
    // overwrote each other. `run_vector_core` derives `strip_in_block =
    // (slot0 % block_size) / kTokens` now and offsets both the packed base and
    // the two scale bases by it, so a page may hold as many strips as it likes
    // (op_kernel/turboquant_reshape_batch.cpp, the VectorEngine setup).
    TORCH_CHECK(c.block_size % kTokensPerStrip == 0, op, ": the strip cache is written ", kTokensPerStrip,
                " tokens at a time, so block_size must be a multiple of ", kTokensPerStrip, ", got ",
                c.block_size);

    check_burst_aligned(key, op, "key");
    check_burst_aligned(value, op, "value");
    check_burst_aligned(key_cache, op, "key_cache");
    check_burst_aligned(value_cache, op, "value_cache");
    check_burst_aligned(scale_cache, op, "scale_cache");
    check_burst_aligned(tables, op, "tables");
    check_strip_slots(slot_mapping, a.num_tokens, op);

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    launch_turboquant_reshape_and_cache_bf16(
        stream, static_cast<const std::uint16_t *>(key.data_ptr()),
        static_cast<const std::uint16_t *>(value.data_ptr()),
        static_cast<std::int8_t *>(key_cache.data_ptr()),
        static_cast<std::int8_t *>(value_cache.data_ptr()), static_cast<float *>(scale_cache.data_ptr()),
        static_cast<const std::int32_t *>(slot_mapping.data_ptr()), optional_pi_signs(pi_signs),
        // The launcher publishes `tables` as int32* and reinterprets it device
        // side; the bytes are the INT8 blocks checked above.
        static_cast<const std::int32_t *>(tables.data_ptr()), static_cast<std::uint32_t>(a.num_tokens),
        static_cast<std::uint32_t>(a.num_kv_heads), static_cast<std::uint32_t>(a.head_size),
        static_cast<std::uint32_t>(c.block_size), static_cast<std::uint32_t>(c.num_blocks),
        static_cast<std::uint32_t>(kTokensPerStrip), static_cast<float>(inv_sqrt_len), reuse_mode4_flags);
}

// ---------------------------------------------------------------------------
// 2. Decode scatter ingestion
// ---------------------------------------------------------------------------

// One token per request, rotated and packed in UB and spliced into the page,
// over launch_turboquant_reshape_scatter_aiv (op_kernel/turboquant_reshape_scatter.cpp).
//
// The value plane is a read-modify-write: channel-major inside the strip pairs
// tokens 2t and 2t+1 in one byte, so a per-token write has to read the strip,
// splice its own nibbles in by register masks and write the strip back. That is
// why this kernel exists. The PyTorch golden model does the same thing a strip
// at a time (_write_strip_planes in turboquant_dry_run.py), so the two agree on
// what lands; only this one does it without reading the strip back to host.
//
// Host derivation, as the brief pins it: total_items = num_requests *
// num_kv_heads, item_begin = 0, item_count = total_items, block_dim =
// min(total_items, aiv_core_num). `cu_seqlens` and `launch_probe` are null on a
// standard decode step -- the first is the prefix sum a multi-token-per-request
// (MTP) step needs to find a request's own token, the second the canary runner's
// int32 probe.
void npu_turboquant_reshape_scatter_aiv(at::Tensor &key, at::Tensor &value, at::Tensor &key_cache,
                                        at::Tensor &value_cache, at::Tensor &scale_cache,
                                        at::Tensor &slot_mapping, at::Tensor &pi_signs, double inv_sqrt_len)
{
    constexpr const char *op = "npu_turboquant_reshape_scatter_aiv";

    const Activations a = check_activations(key, value, op);
    const CachePlanes c = check_cache(key_cache, value_cache, scale_cache, a, op);
    check_slot_mapping(slot_mapping, a.num_tokens, op);

    TORCH_CHECK(a.head_size == kDecodeHeadSize, op, ": the scatter writer is specialised to head_size ",
                kDecodeHeadSize, ", got ", a.head_size);
    // Any whole number of strips per page. Unlike the batch writer this one
    // carries an in-page strip index -- `item.strip` in `StripWordOffset` and
    // `item.token_in_block` in the key burst
    // (op_kernel/turboquant_reshape_scatter.cpp) -- so a page may hold as many
    // 16-token strips as it likes.
    TORCH_CHECK(c.block_size % kTokensPerStrip == 0, op, ": the strip cache is written ", kTokensPerStrip,
                " tokens at a time, so block_size must be a multiple of ", kTokensPerStrip, ", got ",
                c.block_size);

    check_burst_aligned(key, op, "key");
    check_burst_aligned(value, op, "value");
    check_burst_aligned(key_cache, op, "key_cache");
    check_burst_aligned(value_cache, op, "value_cache");
    check_burst_aligned(scale_cache, op, "scale_cache");

    // A decode step ingests exactly one token per request, so the token count
    // IS the request count here. A multi-token-per-request (MTP) step would
    // break that, and the writer's way of finding a request's own token is the
    // `cu_seqlens` prefix sum this op passes as null -- so MTP needs a caller
    // that builds it, not a different token count.
    const std::int64_t num_requests = a.num_tokens;
    // One work item per (request, kv head): the writer folds the two into one
    // index so that two heads of one request cannot collide on a strip.
    const std::int64_t total_items = num_requests * a.num_kv_heads;
    TORCH_CHECK(total_items > 0, op, ": nothing to scatter");

    std::int64_t aiv_num = 0;
    TORCH_CHECK(aclGetDeviceCapability(0, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS, op,
                ": could not read the device's vector core count");
    TORCH_CHECK(aiv_num > 0, op, ": the device reports ", aiv_num, " vector cores");
    const std::int64_t block_dim = std::min(total_items, aiv_num);

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    launch_turboquant_reshape_scatter_aiv(
        stream, static_cast<std::uint32_t>(block_dim), static_cast<std::uint16_t *>(key.data_ptr()),
        static_cast<std::uint16_t *>(value.data_ptr()), static_cast<std::int8_t *>(key_cache.data_ptr()),
        static_cast<std::int8_t *>(value_cache.data_ptr()), static_cast<float *>(scale_cache.data_ptr()),
        static_cast<std::int32_t *>(slot_mapping.data_ptr()),
        // Cast away const: the launcher's parameter list is non-const here.
        const_cast<float *>(optional_pi_signs(pi_signs)),
        /*cu_seqlens=*/nullptr, /*launch_probe=*/nullptr, static_cast<std::uint32_t>(num_requests),
        static_cast<std::uint32_t>(a.num_kv_heads), static_cast<std::uint32_t>(a.head_size),
        static_cast<std::uint32_t>(c.block_size), static_cast<std::uint32_t>(c.num_blocks),
        /*item_begin=*/0u, static_cast<std::uint32_t>(total_items), static_cast<float>(inv_sqrt_len));
}

// ---------------------------------------------------------------------------
// 3. Fused attention decode
// ---------------------------------------------------------------------------

// The raw-query one-launch decode, over
// vllm_ascend::turboquant_mm_fused_decode_opt_raw_query_impl.
//
// Three buffers the kernel needs are the caller's to build and are built here
// rather than exposed in the schema, because each one is derived from the
// launch geometry and a caller that got one wrong would get a hang or a wrong
// answer, not an error:
//
//   tiling     decode_tiling_bytes(tokens, splits, grid) of host-built
//              descriptors, uploaded like any other input. The launcher throws
//              without it: a decode that derived its own tiling on the SPU is
//              the operator the harness removed.
//   workspace  the Split-K partials, tokens * num_heads * splits *
//              (head_size + kPartialTail) fp32, which TurboQuantPartialReducer
//              addresses as ((token * heads + head) * splits + split) * stride.
//   query_rot  the fp32 rotated image the prologue WRITES and the decode then
//              reads. It is not `query` and the two are not interchangeable.
//
// build_decode_tiling needs the context lengths HOST-visible, because a
// worker's tile count is a function of its token's context length and that is
// the one input the device used to fetch from GM and divide by. The metadata
// keeps seq_lens on the host already; if a device tensor arrives, it is copied
// down, which is a device-to-host copy and therefore illegal on a capturing
// stream. That is one of the reasons TurboQuant is served PIECEWISE-only, where
// attention runs eagerly between captured pieces.
//
// The block table is passed to the builder as nullptr: the alternative is a
// second device-to-host copy per layer per step, and the header's own contract
// says a null table is assumed dense over each live span while the device still
// refuses to stage a span holding a negative entry -- so a wrong assumption is a
// skipped worker, never an out-of-bounds read.
void npu_turboquant_fused_decode(at::Tensor &query, at::Tensor &pi_signs, at::Tensor &rot_tables,
                                 at::Tensor &h16, const std::optional<at::Tensor> &gate,
                                 at::Tensor &key_cache, at::Tensor &value_cache, at::Tensor &scale_cache,
                                 at::Tensor &block_tables, at::Tensor &context_lens, at::Tensor &output,
                                 int64_t num_heads, int64_t num_kv_heads, int64_t num_splits,
                                 int64_t heads_per_task, int64_t output_stage, double scale,
                                 double inv_sqrt_len)
{
    constexpr const char *op = "npu_turboquant_fused_decode";

    // ---- query and output ---------------------------------------------------
    TORCH_CHECK(query.dim() == 3 && output.dim() == 3, op,
                ": query and output must be [num_tokens, num_heads, head_size]");
    TORCH_CHECK(query.sizes() == output.sizes(), op, ": query and output must have the same shape");
    TORCH_CHECK(query.is_contiguous() && output.is_contiguous(), op, ": query and output must be contiguous");
    TORCH_CHECK(query.scalar_type() == output.scalar_type(), op,
                ": query and output must share an element type, got ", query.scalar_type(), " and ",
                output.scalar_type());
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16, op,
                ": the operator has kernel instances for fp16 and bf16 only, got ", query.scalar_type());

    const std::int64_t num_tokens = query.size(0);
    const std::int64_t head_size = query.size(2);
    TORCH_CHECK(query.size(1) == num_heads, op, ": query has ", query.size(1), " heads, num_heads says ",
                num_heads);

    // ---- geometry valid_launch() will refuse, named here instead -----------
    TORCH_CHECK(head_size == kDecodeHeadSize, op, ": the fused decode is specialised to head_size ",
                kDecodeHeadSize, ", got ", head_size);
    TORCH_CHECK(num_kv_heads > 0 && num_kv_heads <= kDecodeMaxKvHeads, op, ": num_kv_heads must be 1..",
                kDecodeMaxKvHeads, ", got ", num_kv_heads);
    TORCH_CHECK(num_heads > 0 && num_heads % num_kv_heads == 0, op, ": num_heads (", num_heads,
                ") must be a positive multiple of num_kv_heads (", num_kv_heads, ")");
    TORCH_CHECK(num_splits > 0 && num_splits <= kDecodeMaxSplits, op, ": num_splits must be 1..",
                kDecodeMaxSplits, ", got ", num_splits);
    TORCH_CHECK(heads_per_task > 0 && heads_per_task <= kDecodeMaxHeadsPerTask, op,
                ": heads_per_task must be 1..", kDecodeMaxHeadsPerTask, ", got ", heads_per_task);
    TORCH_CHECK(output_stage >= 0 && output_stage <= 2, op,
                ": output_stage names a TurboQuantOutputStage (0 rotated basis, 1 unrotated, 2 gated), got ",
                output_stage);
    // Only kGated reads the gate buffer. The launch header's field comment says
    // "Only 1 and 2 read the gate buffer", but op_kernel/cube/turboquant_query_basis.h
    // returns before touching gateGm_ unless outputStage_ == kGated, so stage 1
    // with a null gate is safe by the code. Checked against the code, not the
    // comment; if a device run ever faults at stage 1 with no gate, that comment
    // is the thing to go back to.
    TORCH_CHECK(output_stage != static_cast<int64_t>(layout::TurboQuantOutputStage::kGated) || gate.has_value(),
                op, ": output_stage 2 divides by 1 + exp(-gate) and needs the gate buffer");

    // ---- cache ---------------------------------------------------------------
    TORCH_CHECK(key_cache.dim() == 4 && key_cache.sizes() == value_cache.sizes(), op,
                ": the packed planes must both be 4-D and identically shaped");
    TORCH_CHECK(key_cache.scalar_type() == at::kChar && value_cache.scalar_type() == at::kChar, op,
                ": the packed planes are int8 nibble pairs");
    TORCH_CHECK(scale_cache.scalar_type() == at::kFloat, op, ": the scale plane is fp32");
    const std::int64_t block_size = key_cache.size(1);
    TORCH_CHECK(block_size > 0 && block_size % layout::kCubeTileRows == 0 &&
                    (block_size & (block_size - 1)) == 0,
                op, ": the closed-form tile cursor needs block_size to be a power-of-two multiple of ",
                layout::kCubeTileRows, ", got ", block_size,
                ". The NZ-tiled plane this operator reads strides its tile rows by ", layout::kCubeTileRows,
                ", so a smaller page cannot address it at all.");

    TORCH_CHECK(block_tables.dim() == 2 && block_tables.scalar_type() == at::kInt, op,
                ": block_tables must be an int32 [num_tokens, max_blocks_per_seq]");
    TORCH_CHECK(block_tables.size(0) >= num_tokens, op, ": block_tables has ", block_tables.size(0),
                " rows for ", num_tokens, " tokens");
    const std::int64_t max_blocks_per_seq = block_tables.size(1);
    TORCH_CHECK(context_lens.scalar_type() == at::kInt && context_lens.dim() == 1 &&
                    context_lens.size(0) >= num_tokens,
                op, ": context_lens must be an int32 [num_tokens], got ", context_lens.scalar_type(), " ",
                context_lens.sizes());

    // ---- the basis operands --------------------------------------------------
    TORCH_CHECK(pi_signs.scalar_type() == at::kFloat && pi_signs.numel() == head_size, op,
                ": pi_signs must be ", head_size, " fp32 entries, got ", pi_signs.numel());
    check_tables(rot_tables, head_size, op);
    TORCH_CHECK(h16.scalar_type() == at::kHalf &&
                    h16.numel() == static_cast<std::int64_t>(layout::kRotateQH16Elements),
                op, ": h16 is the Cube's resident ", layout::kRotateQH16Elements,
                "-element fp16 sign matrix (16x16, +-1), got ", h16.scalar_type(), " with ", h16.numel(),
                " elements");

    // ---- the host work split -------------------------------------------------
    // The odometer build_decode_tiling walks: chunk, split, kv head, token.
    const std::int64_t group_heads = num_heads / num_kv_heads;
    const std::int64_t chunk = std::min<std::int64_t>(heads_per_task, group_heads);
    const std::int64_t chunks_per_group = ceil_div(group_heads, chunk);
    const std::int64_t tasks = num_tokens * num_kv_heads * num_splits * chunks_per_group;
    const std::int64_t reduce_total = num_tokens * num_heads;
    TORCH_CHECK(tasks > 0, op, ": nothing to decode");

    // A split launch reaches AscendC::SyncAll<false>(), which on dav-3510 only
    // completes if every launched block is resident at the same time, so the
    // grid is capped at the physical block count either way.
    const std::int64_t grid = std::min<std::int64_t>(kDecodeMaxGrid, std::max<std::int64_t>(tasks, 1));
    const std::int64_t tasks_per_block = ceil_div(tasks, grid);
    const std::int64_t reduce_tasks_per_block = ceil_div(reduce_total, grid);
    const std::int64_t prologue_vectors_per_block = ceil_div(reduce_total, grid);
    // 0 rotates the whole share with ApplyPi on the vector cores; 16 rotates
    // whole chunks on the Cube, which is only worth a launch when the block's
    // share actually holds that many query vectors.
    const std::int64_t prologue_cube_chunk_vectors =
        prologue_vectors_per_block >= static_cast<std::int64_t>(layout::kQueryBasisMinCubeVectors)
            ? static_cast<std::int64_t>(layout::kQueryBasisCubeVectors)
            : 0;

    // ---- the host tiling descriptors ----------------------------------------
    tiling::DecodeTilingRequest request;
    request.tokens = static_cast<std::uint32_t>(num_tokens);
    request.num_heads = static_cast<std::uint32_t>(num_heads);
    request.num_kv_heads = static_cast<std::uint32_t>(num_kv_heads);
    request.head_dim = static_cast<std::uint32_t>(head_size);
    request.block_size = static_cast<std::uint32_t>(block_size);
    request.max_blocks = static_cast<std::uint32_t>(max_blocks_per_seq);
    request.splits = static_cast<std::uint32_t>(num_splits);
    request.heads_per_task = static_cast<std::uint32_t>(heads_per_task);
    request.tasks_per_block = static_cast<std::uint32_t>(tasks_per_block);
    request.reduce_tasks_per_block = static_cast<std::uint32_t>(reduce_tasks_per_block);
    request.grid = static_cast<std::uint32_t>(grid);
    request.fused_context_limit = layout::kFusedContextLimit;

    // The context lengths are needed on BOTH sides, and which copy exists
    // depends on where the caller keeps them. vLLM's metadata keeps seq_lens on
    // the host on purpose, so the common case is a host tensor here and the
    // device copy below is the one that has to be made; a caller that hands
    // over a device tensor pays the device-to-host copy instead. Either copy is
    // illegal on a capturing stream, which is one of the reasons TurboQuant is
    // served PIECEWISE-only.
    const at::Tensor context_lens_host =
        context_lens.device().is_cpu() ? context_lens.contiguous() : context_lens.to(at::kCPU).contiguous();
    const at::Tensor context_lens_device = context_lens.device() == query.device()
                                               ? context_lens.contiguous()
                                               : context_lens.to(query.device()).contiguous();

    const std::size_t tiling_bytes = tiling::decode_tiling_bytes(
        request.tokens, request.splits, request.grid);
    const at::Tensor tiling_host =
        at::empty({static_cast<std::int64_t>(tiling_bytes)}, query.options().dtype(at::kByte).device(at::kCPU));
    // build_decode_tiling reinterprets this buffer as 64-byte-aligned
    // descriptors and writes them by offset, so a misaligned base would be
    // undefined behaviour rather than a slow path. Torch's CPU allocator gives
    // 64 bytes; the check is here because nothing else would say so.
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tiling_host.data_ptr()) % alignof(tiling::TurboQuantDecodeTilingData) ==
                    0,
                op, ": the tiling buffer is not ", alignof(tiling::TurboQuantDecodeTilingData), "-byte aligned");
    const tiling::DecodeTilingStatus status = tiling::build_decode_tiling(
        request, static_cast<const std::int32_t *>(context_lens_host.data_ptr()),
        /*block_table=*/nullptr, tiling_host.data_ptr());
    TORCH_CHECK(status == tiling::DecodeTilingStatus::kOk, op, ": ",
                tiling::decode_tiling_status_text(status));
    const at::Tensor tiling_device = tiling_host.to(query.device(), /*non_blocking=*/false);

    // ---- the Split-K workspace and the rotated query image -------------------
    const std::int64_t workspace_floats =
        num_tokens * num_heads * num_splits * (head_size + static_cast<std::int64_t>(layout::kPartialTail));
    const at::Tensor workspace =
        at::zeros({workspace_floats}, query.options().dtype(at::kFloat));
    const at::Tensor query_rot =
        at::empty({num_tokens, num_heads, head_size}, query.options().dtype(at::kFloat));

    const AscendType element_type =
        query.scalar_type() == at::kBFloat16 ? AscendType::BF16 : AscendType::FP16;

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    turboquant_mm_fused_decode_opt_raw_query_impl(
        static_cast<std::int32_t>(layout::TurboQuantMode::KV4_FP8), element_type, stream,
        static_cast<std::uint32_t>(grid), query.data_ptr(), pi_signs.data_ptr(), rot_tables.data_ptr(),
        h16.data_ptr(), gate.has_value() ? gate->data_ptr() : nullptr, query_rot.data_ptr(),
        key_cache.data_ptr(), value_cache.data_ptr(), scale_cache.data_ptr(), block_tables.data_ptr(),
        context_lens_device.data_ptr(), /*mode_tables=*/nullptr, workspace.data_ptr(), output.data_ptr(),
        tiling_device.data_ptr(), static_cast<std::uint32_t>(num_tokens),
        static_cast<std::uint32_t>(num_heads), static_cast<std::uint32_t>(num_kv_heads),
        static_cast<std::uint32_t>(head_size), static_cast<std::uint32_t>(block_size),
        static_cast<std::uint32_t>(max_blocks_per_seq), static_cast<std::uint32_t>(num_splits),
        static_cast<std::uint32_t>(heads_per_task), static_cast<std::uint32_t>(tasks_per_block),
        static_cast<std::uint32_t>(reduce_tasks_per_block),
        static_cast<std::uint32_t>(prologue_vectors_per_block),
        static_cast<std::uint32_t>(prologue_cube_chunk_vectors), static_cast<std::uint32_t>(output_stage),
        layout::kFusedContextLimit, static_cast<float>(scale), static_cast<float>(inv_sqrt_len));
}

// ---------------------------------------------------------------------------
// Meta kernels
// ---------------------------------------------------------------------------

// All three ops return nothing and write through their `Tensor!` arguments, so
// a meta kernel has no shape to compute: it exists so torch.export and a
// FakeTensor trace walk the op instead of trying to run it. The served path
// does not need them -- both TurboQuant entry points are graph splitting ops
// under PIECEWISE, so the kernels launch eagerly between captured pieces -- but
// a trace that reaches an op with no Meta key dies, and that failure points at
// the tracer rather than at the missing registration.
//
// The fused decode's meta kernel must NOT run build_decode_tiling: the builder
// reads the context lengths, and under FakeTensor there is no storage to read.
namespace meta {

void npu_turboquant_reshape_and_cache_a8_meta(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                                              at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, double,
                                              bool)
{
}

void npu_turboquant_reshape_scatter_aiv_meta(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                                             at::Tensor &, at::Tensor &, at::Tensor &, double)
{
}

void npu_turboquant_fused_decode_meta(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                                      const std::optional<at::Tensor> &, at::Tensor &, at::Tensor &,
                                      at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, int64_t,
                                      int64_t, int64_t, int64_t, int64_t, double, double)
{
}

} // namespace meta
} // namespace turboquant_bindings
} // namespace vllm_ascend

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

// A FRAGMENT, not a TORCH_LIBRARY: nothing else defines `turboquant_ops`
// today, but a fragment lets a second translation unit add to the namespace
// later without either file owning it.
//
// The scalars every schema does NOT take are derived from the tensors, which is
// why the Python stubs in turboquant_dry_run.py pass fewer arguments than the C
// launcher's parameter list: num_tokens, num_kv_heads, head_size, block_size
// and num_blocks all come off the activation and cache shapes, `tokens_per_core`
// is the kernel's own compile-time strip height, and the scatter writer's
// block_dim / item_begin / item_count are the host work split above. A schema
// that took them would let a caller contradict the tensors it also passed.
TORCH_LIBRARY_FRAGMENT(turboquant_ops, ops)
{
    // key/value    [num_tokens, num_kv_heads, head_size] bf16, contiguous
    // key_cache    [num_blocks, 16, num_kv_heads, head_size / 2] int8, key-major
    //              within a (block, head) page: nibble = token * head_size + channel
    // value_cache  the same extent, channel-major: nibble = channel * block_size + token
    // scale_cache  [2, num_blocks, num_kv_heads, 16] fp32, kv 0 keys / 1 values
    // slot_mapping [num_tokens] int32; strip i must be slot_mapping[0] + i with
    //              slot_mapping[0] a multiple of block_size
    // pi_signs     folded into `tables`; an empty tensor says so
    // tables       2 * (head_size / 32) row-major 32x32 int8 blocks
    ops.def(
        "npu_turboquant_reshape_and_cache_a8(Tensor key, Tensor value, Tensor! key_cache, "
        "Tensor! value_cache, Tensor! scale_cache, Tensor slot_mapping, Tensor pi_signs, "
        "Tensor tables, float inv_sqrt_len, bool reuse_mode4_flags) -> ()");
    ops.impl("npu_turboquant_reshape_and_cache_a8", c10::kPrivateUse1,
             &vllm_ascend::turboquant_bindings::npu_turboquant_reshape_and_cache_a8);
    ops.impl("npu_turboquant_reshape_and_cache_a8", c10::kMeta,
             &vllm_ascend::turboquant_bindings::meta::npu_turboquant_reshape_and_cache_a8_meta);

    // One token per request, arbitrary slots. The value plane's write is a
    // read-modify-write of one 2048-byte strip, which is the reason this op
    // exists: the PyTorch golden model cannot do it safely.
    ops.def(
        "npu_turboquant_reshape_scatter_aiv(Tensor key, Tensor value, Tensor! key_cache, "
        "Tensor! value_cache, Tensor! scale_cache, Tensor slot_mapping, Tensor pi_signs, "
        "float inv_sqrt_len) -> ()");
    ops.impl("npu_turboquant_reshape_scatter_aiv", c10::kPrivateUse1,
             &vllm_ascend::turboquant_bindings::npu_turboquant_reshape_scatter_aiv);
    ops.impl("npu_turboquant_reshape_scatter_aiv", c10::kMeta,
             &vllm_ascend::turboquant_bindings::meta::npu_turboquant_reshape_scatter_aiv_meta);

    // query/output [num_tokens, num_heads, 256] fp16 or bf16
    // rot_tables   the int8 blocks above; h16 the 16x16 fp16 +-1 matrix
    // gate         read only at output_stage 2
    // output_stage TurboQuantOutputStage: 0 rotated basis, 1 unrotated, 2 gated
    // The tiling descriptors, the Split-K workspace and the fp32 rotated query
    // image are built inside the op, not passed: each is derived from this
    // geometry and a caller that got one wrong would hang rather than fail.
    ops.def(
        "npu_turboquant_fused_decode(Tensor query, Tensor pi_signs, Tensor rot_tables, Tensor h16, "
        "Tensor? gate, Tensor key_cache, Tensor value_cache, Tensor scale_cache, Tensor block_tables, "
        "Tensor context_lens, Tensor! output, int num_heads, int num_kv_heads, int num_splits, "
        "int heads_per_task, int output_stage, float scale, float inv_sqrt_len) -> ()");
    ops.impl("npu_turboquant_fused_decode", c10::kPrivateUse1,
             &vllm_ascend::turboquant_bindings::npu_turboquant_fused_decode);
    ops.impl("npu_turboquant_fused_decode", c10::kMeta,
             &vllm_ascend::turboquant_bindings::meta::npu_turboquant_fused_decode_meta);
}
