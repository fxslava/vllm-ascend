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
// THE BATCH WRITER AND THE FUSED DECODE NOW SHARE A CACHE FORMAT; THE SCATTER
// WRITER STILL DOES NOT. This was the harness's Phase 1 position -- both
// writers produced the RFC 3.1 strip cache while the decode reads the NZ-tiled
// plane of op_kernel/common/turboquant_layout.h's NzTiledPackedByte(), and the
// two formats disagree on the PAIRING OF NIBBLES INTO BYTES, which no
// permutation of bytes can undo. The mismatch and its evidence are kept at the
// bottom of this comment block; what changed is the writer side:
//
//   * the BATCH WRITER (op_kernel/turboquant_reshape_batch.cpp) is now built
//     with TURBOQUANT_INGRESS_NZ=1 (see the root CMakeLists). It emits the
//     NZ-tiled cache the decode reads -- same address function, same nibble
//     pairing, same slot-major scale records -- and its UB-side NZ stage is
//     tied back to turboquant_layout.h by static_assert, so a writer that
//     drifted from the reader cannot compile. tests/ut/ops/turboquant/
//     test_layout_bridge.cpp is the same gate from the reader's side.
//   * the SCATTER WRITER (op_kernel/turboquant_reshape_scatter.cpp) has no NZ
//     variant in turboquant-core and still writes the strip cache. It must not
//     touch a cache the NZ pair shares: every byte it writes lands in the wrong
//     place for the decode. The Python adapter
//     (vllm_ascend/attention/turboquant_dry_run.py) keeps it off an NZ cache
//     and covers per-token writes with its golden NZ writer until an NZ
//     scatter kernel exists in the harness.
//
// --- the original mismatch, kept as the record of why the ingress exists ---
//
// The difference was NOT a byte ordering that a reordering could fix. It was a
// different PAIRING OF NIBBLES INTO BYTES, which no permutation of bytes can
// undo:
//
//   * the two WRITERS: the key plane's nibble index inside a (block, head) page
//     is `token * head_size + channel`; the value plane's is
//     `(token / 16) * (16 * head_size) + channel * 16 + (token % 16)`
//     (op_kernel/turboquant_reshape_scatter.cpp's StripWordOffset plus its
//     within-strip index). Consecutive nibbles are the SAME CHANNEL at TWO
//     TOKENS, so a byte holds one channel of tokens 2p and 2p+1.
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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#include <utility>

#include <torch/library.h>
#include <torch/torch.h>

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "op_host/turboquant_tiling.h"
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

// 2 * (head_size / 32) row-major 32x32 BF16 blocks: T[s][g] of the rotation,
// with the sign diagonal pre-composed. 16384 ELEMENTS and therefore 32768 BYTES
// at head_size 256, which is what `_codec_tables` in turboquant_dry_run.py
// builds.
//
// BF16, NOT INT8, and this was wrong here until it was checked against the
// kernel. The batch writer stages the blob as the Cube's BF16 L1 operand:
//
//   // op_kernel/turboquant_reshape_batch.cpp
//   ascend_micro::stage_blocks_nz<kTableBlocks, kGroup>(
//       reinterpret_cast<__gm__ const bfloat16_t*>(tables), kL1Table);
//
// and `stage_blocks_nz<Blocks, Dim>` reads `Blocks * Dim * Dim` *bfloat16_t*
// elements, which is the `kTableBytes = kTableElems * kElemBytes` it reserves in
// L1 with `kElemBytes == 2`. An int8 tensor of the same element count is half
// those bytes, so accepting one fed the Cube +-1 bytes reinterpreted as BF16 --
// a wrong rotation, not a slightly wrong one -- and read 16 KB past the end of
// the operand.
//
// Nothing in the signature could have caught it: the launcher's parameter is
// declared `const std::int32_t*` and the kernel reinterpret_casts it, so the
// declared pointer type says nothing about the element type and neither side of
// the cast is checkable. The authority is the operator's own runner, which
// converts the oracle's int8 blob on the way up -- tests/npu/test_live_write.cpp,
// `tb[i] = to_bf16(table_blob[i])` into a buffer sized
// `table_blob.size() * sizeof(std::uint16_t)`.
//
// Every entry is +-1, exact in BF16, so the conversion the Python side does is a
// representation change rather than a rounding: the operand the Cube contracts
// against is bit-identical to the oracle's, element for element.
constexpr std::int64_t kTableGroup = 32;
constexpr std::int64_t kTablePlanes = 2;

// The QUERY BASIS codec's GM constant pool, which is a different operand from both of the
// above. TurboQuantCodec4::ConstTableWords (op_kernel/common/turboquant_codec_950.h) is
// `7 * vecLen + 2 * vecLen * batchRows + kLevels`, and TurboQuantQueryBasis::Init calls it
// with batchRows 1, so the pool is 9 * head_size + kLevels words. kEarlyStages of those are
// the butterfly's (sign, XOR byte offset) pairs that ApplyPi's first three stages gather
// through; see the pool's construction in npu_turboquant_fused_decode for why it is built
// here and what reading the Cube blob into it did.
constexpr std::int64_t kCodecEarlyStages = 3;
constexpr std::int64_t kCodecLevels = 16;
constexpr std::int64_t kCodecPoolWordsPerChannel = 9;

inline std::int64_t ceil_div(const std::int64_t a, const std::int64_t b)
{
    return b > 0 ? (a + b - 1) / b : 0;
}

// The tiling request, from the geometry both the decode and the tiling builder
// are given. One definition because the two must describe the same launch: the
// builder writes descriptors the decode reads by offset, and a request that
// differed in any field would produce a buffer the decode walks wrongly without
// any check being able to say so.
//
// `tasks_per_block`, `reduce_tasks_per_block` and `grid` are left at their
// defaults here and filled by tiling::decode_launch_split, which is the
// production derivation of them; `probe_page` / `probe_tile` stay at the
// defaults that mean "every worker on its production span" -- the isolated
// canary is a C++ fixture's, never a served launch's.
inline ::ascend_micro::fused_decode::DecodeTilingRequest decode_request(
    const std::int64_t num_tokens, const std::int64_t num_heads, const std::int64_t num_kv_heads,
    const std::int64_t head_size, const std::int64_t block_size, const std::int64_t max_blocks,
    const std::int64_t num_splits, const std::int64_t heads_per_task)
{
    ::ascend_micro::fused_decode::DecodeTilingRequest r;
    r.tokens = static_cast<std::uint32_t>(num_tokens);
    r.num_heads = static_cast<std::uint32_t>(num_heads);
    r.num_kv_heads = static_cast<std::uint32_t>(num_kv_heads);
    r.head_dim = static_cast<std::uint32_t>(head_size);
    r.block_size = static_cast<std::uint32_t>(block_size);
    r.max_blocks = static_cast<std::uint32_t>(max_blocks);
    r.splits = static_cast<std::uint32_t>(num_splits);
    r.heads_per_task = static_cast<std::uint32_t>(heads_per_task);
    r.fused_context_limit = ::vllm_ascend::turboquant::kFusedContextLimit;
    return r;
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

    // The two writers index the scale plane differently -- the scatter writer
    // plane-major, `((kv * num_blocks + block) * num_kv_heads + head) *
    // block_size + token`; the NZ batch writer slot-major, `slot *
    // NzScaleSlotFloats(num_kv_heads) + kv * num_kv_heads + head` -- but at
    // num_kv_heads 4 both address exactly 8 floats per slot and the highest
    // lane written is this bound minus one either way, so the check is exact
    // for both. NzScaleSlotFloats pads the record past 2 * num_kv_heads, so a
    // writer compiled for a head count with 2 * num_kv_heads % 8 != 0 would
    // need a larger extent than this; the NZ ingress is compiled for 4 heads
    // and refuses any other in its launcher.
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

// The Cube's resident rotation operand. See the kTableGroup block above for why
// this is BF16 and what accepting int8 did.
//
// `name` is the schema's own spelling -- `tables` on the batch writer,
// `rot_tables` on the fused decode -- so a message names the argument the caller
// passed rather than a shared alias. The two are the same blob and must agree on
// dtype, and the decode's copy is checked even though the decode does not hand it
// to the device at all: it builds the basis codec's own pool instead, for the
// reason written out in npu_turboquant_fused_decode.
//
// THE `ConstTableWords` THAT RETURNS 0 IS THE MODE CODEC'S, not the basis codec's.
// `TurboQuantModeCodec::ConstTableWords`
// (op_kernel/common/turboquant_codec_mx.h) returns `kGatherLut ? kLevels : 0` for
// an affine mode, so `mode_tables` really is unread under KV4_FP8. Reading that
// onto this operand -- which goes to a `TurboQuantCodec4`, whose own
// ConstTableWords is an unconditional 9 * head_size + kLevels -- is what put the
// Cube's blob into the basis's gather-offset pool and AIV error 341 on the device.
// A decode that runs is no evidence that either operand is right.
inline void check_tables(const at::Tensor &tables, const std::int64_t head_size, const char *op,
                         const char *name)
{
    const std::int64_t groups = head_size / kTableGroup;
    const std::int64_t expected = kTablePlanes * groups * kTableGroup * kTableGroup;
    TORCH_CHECK(tables.scalar_type() == at::kBFloat16, op, ": ", name,
                " is the Cube's resident BF16 operand (stage_blocks_nz reads it as bfloat16_t), got ",
                tables.scalar_type(),
                ". An int8 blob is half the bytes the kernel reads and the wrong arithmetic.");
    TORCH_CHECK(tables.is_contiguous() && tables.numel() == expected, op, ": ", name, " must be ", expected,
                " contiguous bfloat16 elements (", kTablePlanes * groups, " row-major ", kTableGroup, "x",
                kTableGroup, " blocks of +-1, ", expected * 2, " bytes), got ", tables.numel());
}

// ---------------------------------------------------------------------------
// Operands that are constants, cached for the process
// ---------------------------------------------------------------------------
//
// WHY THESE ARE STATICS NOW. Both of the buffers below are pure functions of
// `head_size`, and both used to be built and uploaded inside every launch of
// every layer: a 1536-word host loop plus a ~9 KB host-to-device copy, sixty
// times per decode step on a sixty-layer model. The copy is also illegal on a
// capturing stream (107030, "Stream during the capture stage is not
// supported"), so caching them is both the performance fix and one of the four
// things that stood between this operator and an ACL graph.
//
// THE ONE CAVEAT, written down because nothing enforces it: the first call for
// a given (head_size, device) is what builds and uploads the entry, and that
// call must not itself be inside a capture. vLLM guarantees it --
// `cudagraph_num_of_warmups` is 1 and `_dummy_run` executes the model eagerly
// before `capture_model` captures anything -- but if that ever stopped being
// true, the symptom would be a capture failure on the first graph rather than a
// wrong answer.
//
// Which constant, so that two of them cannot collide on one key. The basis
// pool depends on head_size and mode_tables does not, so mode_tables would
// otherwise have to pick a head_size to stand in for "none" -- and every value
// it could pick is one the basis pool may legitimately use.
enum class CachedConstant : int {
    kBasisPool = 0,
    kModeTables = 1,
};

// Keyed by device index as well, because a hybrid model has two head sizes and
// a multi-device process one entry per device.
using ConstantKey = std::tuple<CachedConstant, std::int64_t, c10::DeviceIndex>;

inline at::Tensor cached_constant(const CachedConstant which, const std::int64_t head_size,
                                  const at::Device &device, const std::function<at::Tensor()> &build)
{
    static std::mutex guard;
    static std::map<ConstantKey, at::Tensor> cache;
    const ConstantKey key{which, head_size, device.index()};
    const std::lock_guard<std::mutex> held(guard);
    auto found = cache.find(key);
    // The device is in the key by index only, so confirm the whole device
    // matches before reusing an entry.
    if (found != cache.end() && found->second.device() == device) {
        return found->second;
    }
    at::Tensor built = build();
    cache[key] = built;
    return built;
}

// The device's vector core count, read once. aclGetDeviceCapability does not
// touch a stream, so it was never a capture hazard -- it is simply a runtime
// call that does not belong in a launch.
inline std::int64_t aiv_core_num(const char *op)
{
    static std::int64_t cores = [] {
        std::int64_t value = 0;
        if (aclGetDeviceCapability(0, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &value) != ACL_SUCCESS) {
            return std::int64_t{0};
        }
        return value;
    }();
    TORCH_CHECK(cores > 0, op, ": the device reports ", cores,
                " vector cores (aclGetDeviceCapability failed or answered zero)");
    return cores;
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
// refuses both a partial trailing share and a share that is not a whole number
// of strips, so the only host derivation left here is tokens_per_core itself --
// one strip per block, which is why it is not a schema argument. The grid is
// therefore num_tokens / 16 and spans as many execution waves as it needs; the
// kernel's boundary guard and its epilogue drain are what make one monolithic
// dispatch safe.
//
// The kernel's own pipeline is seamless across strip boundaries, so a share
// wider than one strip is now a legal geometry rather than a refusal: a block
// that owns `tokens_per_core / 16` strips runs them as one uninterrupted chunk
// stream, with no extra prologue, drain or rendezvous per strip. Raising
// kTokensPerStrip here is the knob for that, and it is deliberately not raised
// yet -- nothing has measured which share width is fastest on the part, and a
// one-strip share is the geometry the goldens were taken at.
void npu_turboquant_reshape_and_cache_a8(at::Tensor &key, at::Tensor &value, at::Tensor &key_cache,
                                         at::Tensor &value_cache, at::Tensor &scale_cache,
                                         at::Tensor &slot_mapping, at::Tensor &pi_signs, at::Tensor &tables,
                                         double inv_sqrt_len)
{
    constexpr const char *op = "npu_turboquant_reshape_and_cache_a8";

    const Activations a = check_activations(key, value, op);
    const CachePlanes c = check_cache(key_cache, value_cache, scale_cache, a, op);
    check_slot_mapping(slot_mapping, a.num_tokens, op);
    check_tables(tables, a.head_size, op, "tables");

    // The kernel's compile-time specialisation (kTokens, kHeads, kHeadSize in
    // op_kernel/turboquant_reshape_batch.cpp). The scalar arguments drive the
    // physical cache addressing only, so a geometry outside these is not a slow
    // path -- it is a wrong answer with every canary intact.
    //
    // A scheduler step is not a whole number of strips -- a 220-token prompt is
    // 13 of them plus 12 tokens, and a batch of several requests has boundaries
    // inside a strip -- so the caller plans the step across both writers rather
    // than handing an arbitrary count over here:
    // `turboquant_plan_strip_ingestion` in
    // vllm_ascend/attention/turboquant_dry_run.py. This check stays a hard
    // refusal because what it refuses has no safe behaviour, only a silent one.
    TORCH_CHECK(a.num_tokens % kTokensPerStrip == 0, op, ": num_tokens (", a.num_tokens,
                ") must be a whole number of ", kTokensPerStrip,
                "-token strips; the launcher drops a partial trailing strip rather than writing it");
    TORCH_CHECK(a.num_kv_heads == 4, op, ": the batch writer is specialised to 4 KV heads, got ",
                a.num_kv_heads);
    TORCH_CHECK(a.head_size == kDecodeHeadSize, op, ": the batch writer is specialised to head_size ",
                kDecodeHeadSize, ", got ", a.head_size);
    // TURBOQUANT_INGRESS_NZ=1: this writer now emits the NZ-tiled cache the
    // fused decode reads, laid out as whole 64-row tiles
    // (op_kernel/turboquant_reshape_batch.cpp, the NzStripBase block, tied to
    // op_kernel/common/turboquant_layout.h by static_assert). A page must
    // therefore hold a whole number of tiles, and the kernel's own launcher
    // refuses anything else. The decode's closed-form tile cursor is stricter
    // still -- a power-of-two multiple of 64 -- so 64, 128, 256, ... serve both
    // and 192 would pass here and fail there.
    TORCH_CHECK(c.block_size % layout::kCubeTileRows == 0, op,
                ": the NZ-tiled ingress writes ", layout::kCubeTileRows,
                "-row tiles, so block_size must be a multiple of ", layout::kCubeTileRows, ", got ",
                c.block_size, " (the fused decode additionally needs a power-of-two multiple)");

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
        // The launcher publishes `tables` as int32* and the kernel reinterprets
        // it device side as bfloat16_t. The pointer type is meaningless here;
        // the bytes are the BF16 blocks check_tables verified above, and that
        // check is the only thing standing between a caller and a 16 KB
        // over-read of a half-sized operand.
        static_cast<const std::int32_t *>(tables.data_ptr()), static_cast<std::uint32_t>(a.num_tokens),
        static_cast<std::uint32_t>(a.num_kv_heads), static_cast<std::uint32_t>(a.head_size),
        static_cast<std::uint32_t>(c.block_size), static_cast<std::uint32_t>(c.num_blocks),
        static_cast<std::uint32_t>(kTokensPerStrip), static_cast<float>(inv_sqrt_len));
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
// min(total_items, aiv_core_num). `cu_seqlens` is null on a standard decode
// step; it is the prefix sum a multi-token-per-request (MTP) step needs to find
// a request's own token. The canary runner's int32 `launch_probe` is gone from
// the kernel and from this call: it was a synthetic buffer that published
// GetBlockIdx / GetBlockNum / GetSubBlockNum back to a host runner, and nothing
// in this repo ever read it.
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

    const std::int64_t block_dim = std::min(total_items, aiv_core_num(op));

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    launch_turboquant_reshape_scatter_aiv(
        stream, static_cast<std::uint32_t>(block_dim), static_cast<std::uint16_t *>(key.data_ptr()),
        static_cast<std::uint16_t *>(value.data_ptr()), static_cast<std::int8_t *>(key_cache.data_ptr()),
        static_cast<std::int8_t *>(value_cache.data_ptr()), static_cast<float *>(scale_cache.data_ptr()),
        static_cast<std::int32_t *>(slot_mapping.data_ptr()),
        // Cast away const: the launcher's parameter list is non-const here.
        const_cast<float *>(optional_pi_signs(pi_signs)),
        /*cu_seqlens=*/nullptr, static_cast<std::uint32_t>(num_requests),
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
// Four buffers the kernel needs are the caller's to build and are built here
// rather than exposed in the schema, because each one is derived from the
// launch geometry and a caller that got one wrong would get a hang or a wrong
// answer, not an error:
//
//   tiling      decode_tiling_bytes(tokens, splits, grid) of host-built
//               descriptors, uploaded like any other input. The launcher throws
//               without it: a decode that derived its own tiling on the SPU is
//               the operator the harness removed.
//   workspace   the Split-K partials, tokens * num_heads * splits *
//               (head_size + kPartialTail) fp32, which TurboQuantPartialReducer
//               addresses as ((token * heads + head) * splits + split) * stride.
//   query_rot   the fp32 rotated image the prologue WRITES and the decode then
//               reads. It is not `query` and the two are not interchangeable.
//   mode_tables the decode codec's constant pool. KV4_FP8 reads nothing from
//               it, so this used to be a null pointer; see the allocation below
//               for why "nothing reads it" is a property of the mode rather
//               than of the ABI and so not something to pass null on.
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
                                 at::Tensor &block_tables, at::Tensor &context_lens, at::Tensor &tiling,
                                 at::Tensor &workspace, at::Tensor &query_rot, at::Tensor &output,
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
    check_tables(rot_tables, head_size, op, "rot_tables");
    TORCH_CHECK(h16.scalar_type() == at::kHalf &&
                    h16.numel() == static_cast<std::int64_t>(layout::kRotateQH16Elements),
                op, ": h16 is the Cube's resident ", layout::kRotateQH16Elements,
                "-element fp16 sign matrix (16x16, +-1), got ", h16.scalar_type(), " with ", h16.numel(),
                " elements");

    // ---- the host work split -------------------------------------------------
    //
    // Derived by tiling::decode_launch_split, which the tiling builder calls
    // with the same request. It used to be open-coded here, and the caller now
    // needs `grid` too -- decode_tiling_words takes it -- so three places would
    // otherwise have to agree about a number none of them publishes. The
    // odometer build_decode_tiling walks is unchanged: chunk, split, kv head,
    // token.
    const tiling::DecodeTilingRequest request =
        decode_request(num_tokens, num_heads, num_kv_heads, head_size, block_size, max_blocks_per_seq,
                       num_splits, heads_per_task);
    const tiling::DecodeLaunchSplit split =
        tiling::decode_launch_split(request, static_cast<std::uint32_t>(kDecodeMaxGrid));
    TORCH_CHECK(split.tasks > 0, op, ": nothing to decode");
    const std::int64_t grid = split.grid;
    const std::int64_t tasks_per_block = split.tasks_per_block;
    const std::int64_t reduce_tasks_per_block = split.reduce_tasks_per_block;
    const std::int64_t prologue_vectors_per_block = split.prologue_vectors_per_block;
    // 0 rotates the whole share with ApplyPi on the vector cores; 16 rotates
    // whole chunks on the Cube, which is only worth a launch when the block's
    // share actually holds that many query vectors.
    const std::int64_t prologue_cube_chunk_vectors =
        prologue_vectors_per_block >= static_cast<std::int64_t>(layout::kQueryBasisMinCubeVectors)
            ? static_cast<std::int64_t>(layout::kQueryBasisCubeVectors)
            : 0;

    // ---- the caller's three launch buffers -----------------------------------
    //
    // `tiling`, `workspace` and `query_rot` were allocated here, inside the
    // launch, which gave each of them a fresh address on every step. A captured
    // ACL graph replays the address it recorded, so they are the caller's now:
    // allocated once by AscendTurboQuantMetadataBuilder and refreshed in place.
    // Building the tiling descriptors here also meant reading the context
    // lengths on the HOST, and `aclrtMemcpy` on a capturing stream fails with
    // 107030 -- so this operator no longer has a host-visible input at all.
    //
    // The sizes are still derived here rather than trusted, because the caller
    // has to reproduce them and a wrong one is an out-of-bounds device read
    // rather than a wrong answer. `tiling` is checked with >= because its exact
    // extent depends on `grid`, which is the derivation just above: a caller
    // that sized it from an upper bound is correct and a caller that guessed
    // `grid` low is not.
    TORCH_CHECK(context_lens.device() == query.device(), op, ": context_lens is on ",
                context_lens.device(), " but the query is on ", query.device(),
                ". The operator does not copy it: a host-to-device copy here would run once per "
                "layer per step and is illegal on a capturing stream. Pass the metadata's "
                "turboquant_context_lens.");
    TORCH_CHECK(context_lens.is_contiguous(), op,
                ": context_lens must be contiguous; the device reads it as a flat int32 run");

    const std::int64_t tiling_words =
        static_cast<std::int64_t>(tiling::decode_tiling_words(request.tokens, request.splits, split.grid));
    TORCH_CHECK(tiling.scalar_type() == at::kInt && tiling.is_contiguous() && tiling.dim() == 1, op,
                ": tiling must be a contiguous 1-D int32 buffer, got ", tiling.scalar_type(), " ",
                tiling.sizes());
    TORCH_CHECK(tiling.numel() >= tiling_words, op, ": tiling holds ", tiling.numel(),
                " int32 words; this launch needs ", tiling_words, " (", request.tokens, " tokens x ",
                request.splits, " splits of descriptors, then ", split.grid, " dispatch records)");
    TORCH_CHECK(tiling.device() == query.device(), op, ": tiling is on ", tiling.device(),
                " but the query is on ", query.device());
    // The device reads the descriptors as 64-byte lines indexed from word zero.
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tiling.data_ptr()) %
                        alignof(tiling::TurboQuantDecodeTilingData) ==
                    0,
                op, ": the tiling buffer is not ", alignof(tiling::TurboQuantDecodeTilingData),
                "-byte aligned");

    const std::int64_t workspace_floats =
        num_tokens * num_heads * num_splits * (head_size + static_cast<std::int64_t>(layout::kPartialTail));
    TORCH_CHECK(workspace.scalar_type() == at::kFloat && workspace.is_contiguous(), op,
                ": workspace must be contiguous fp32, got ", workspace.scalar_type());
    TORCH_CHECK(workspace.numel() == workspace_floats, op, ": workspace holds ", workspace.numel(),
                " floats, this launch needs exactly ", workspace_floats, " (", num_tokens, " x ",
                num_heads, " x ", num_splits, " x (", head_size, " + ", layout::kPartialTail, "))");
    TORCH_CHECK(workspace.device() == query.device(), op, ": workspace is on ", workspace.device(),
                " but the query is on ", query.device());

    const std::int64_t query_rot_floats = num_tokens * num_heads * head_size;
    TORCH_CHECK(query_rot.scalar_type() == at::kFloat && query_rot.is_contiguous(), op,
                ": query_rot must be contiguous fp32, got ", query_rot.scalar_type());
    TORCH_CHECK(query_rot.numel() == query_rot_floats, op, ": query_rot holds ", query_rot.numel(),
                " floats, this launch needs exactly ", query_rot_floats);
    TORCH_CHECK(query_rot.device() == query.device(), op, ": query_rot is on ", query_rot.device(),
                " but the query is on ", query.device());

    check_burst_aligned(tiling, op, "tiling");
    check_burst_aligned(workspace, op, "workspace");
    check_burst_aligned(query_rot, op, "query_rot");

    // ---- the query basis codec's constant pool -------------------------------
    //
    // NOT `rot_tables`, and this is the ABI break that put AIV error 341 on the device from
    // Python while every CAModel and NPU run of the C++ fixture passed on the identical
    // geometry.
    //
    // TurboQuantQueryBasis::Init does `codec_.Init(pipe, headSize_, 1, invSqrtLen,
    // rotTablesGm_)` on a TurboQuantCodec4, and THAT codec's ConstTableWords
    // (op_kernel/common/turboquant_codec_950.h) is an unconditional
    // `7 * vecLen + 2 * vecLen * batchRows + kLevels` -- 2320 words at head_size 256 -- which
    // Init DataCopies out of whatever pointer this argument carries, before any vector op
    // reads it. Words 0..1535 are three (sign vector, XOR byte-offset vector) pairs, and
    // `ShuffleStage` hands the offsets straight to
    // `AscendC::Gather(swap_, x, xorOffset_[stage], kGatherSrcBase, n)`: the first three
    // butterfly stages of ApplyPi, which is rotate_q.
    //
    // THE CONSTTABLEWORDS THAT RETURNS 0 IS A DIFFERENT CODEC'S. The affine-mode-with-the-
    // gather-LUT-off case belongs to the mode codec in
    // op_kernel/common/turboquant_codec_mx.h, and it governs `mode_tables`, which really is
    // unread under KV4_FP8. Reading that fact onto `rot_tables` is what shipped the Cube's
    // +-1 BF16 rotation blocks into this slot, and a pair of packed bf16 words is a nonsense
    // byte offset: +1.0f is 0x3F80, so one int32 of the blob is 0x3F803F80 == 1065369984, and
    // the prologue's first Gather addressed a gigabyte past the 1 KB row it was shuffling.
    // Every AIV of the launch raised 341 at one pc -- the offset is a constant of the DATA, so
    // all of them faulted on the same instruction -- and the AIC halves were then left in
    // CrossCoreWaitFlag until the watchdog reported them as "timeout or trap error", which is
    // why the device log names the Cube first and the fault second. It is geometry
    // independent, which is why single-tile probes and the full five-tile grid were both clean
    // over this very geometry.
    //
    // Built here rather than asked of the caller because it is a pure function of head_size
    // and because a wrong pool is not something any shape check can catch -- this one was the
    // right ELEMENT COUNT and the right dtype. The tail past word 1536 belongs to
    // Quantize4Bit / Dequantize4Bit, which the basis never calls, and stays zero: exactly what
    // RawQueryBasis::tables() in the operator's own fixture uploads.
    //
    // CACHED, as of the capture work: being a pure function of head_size is exactly what makes
    // it cacheable, and rebuilding it per launch meant a 1536-word host loop and a 9 KB
    // host-to-device copy once per layer per step -- the copy being illegal on a capturing
    // stream. The bytes are identical to what the loop below produced; nothing about the
    // operand changed. See cached_constant for the first-call caveat.
    const std::int64_t basis_pool_words = kCodecPoolWordsPerChannel * head_size + kCodecLevels;
    // The device reads the pool as one DataCopy of whole 32-byte bursts.
    TORCH_CHECK(basis_pool_words * 4 % static_cast<std::int64_t>(kGmBurstBytes) == 0, op,
                ": the basis codec pool is ",
                basis_pool_words * 4, " bytes, not a whole number of ", kGmBurstBytes,
                "-byte bursts");
    const at::Tensor basis_pool = cached_constant(
        CachedConstant::kBasisPool, head_size, query.device(), [&] {
            at::Tensor host = at::zeros({basis_pool_words}, at::TensorOptions().dtype(at::kInt));
            std::int32_t *words = host.data_ptr<std::int32_t>();
            std::int64_t at_word = 0;
            for (std::int64_t stage = 0; stage < kCodecEarlyStages; ++stage) {
                const std::int64_t stride = std::int64_t{1} << stage;
                // The butterfly's sign diagonal for this stage: +1 on the low index of each
                // pair, -1 on the high one, so `x[i] = x[i ^ stride] + sign * x[i]` is a + b
                // and a - b.
                for (std::int64_t i = 0; i < head_size; ++i) {
                    const float sign = (i & stride) == 0 ? 1.0f : -1.0f;
                    std::memcpy(&words[at_word + i], &sign, sizeof(float));
                }
                at_word += head_size;
                // The partner index as a BYTE offset, the unit Gather's srcOffset takes.
                for (std::int64_t i = 0; i < head_size; ++i)
                    words[at_word + i] = static_cast<std::int32_t>((i ^ stride) * sizeof(float));
                at_word += head_size;
            }
            TORCH_CHECK(at_word == 2 * kCodecEarlyStages * head_size, op,
                        ": the basis pool's early-stage span is ", at_word, " words, expected ",
                        2 * kCodecEarlyStages * head_size);
            return host.to(query.device(), /*non_blocking=*/false);
        });
    check_burst_aligned(basis_pool, op, "basis codec pool");

    // ---- the decode codec's constant pool ------------------------------------
    // A required positional argument of the launcher that this mode never
    // dereferences: TurboQuantModeCodec::ConstTableWords returns 0 for an affine
    // mode with the gather LUT off, so Init reads no words from it under
    // KV4_FP8 (op_kernel/common/turboquant_codec_mx.h). A null pointer was
    // therefore safe, and was what this passed.
    //
    // It is a real allocation now because "safe" there is a property of the MODE
    // and not of the ABI. SetGlobalBuffer(nullptr) is only harmless while
    // nothing reads it, so switching the mode -- or turning GATHER_LUT on, which
    // is a template argument away -- would turn this into a null dereference on
    // the device, which surfaces as an AIV fault with no line attached to it.
    // kMinBurstBytes of zeros removes that edge; the operator's own fixture does
    // the same thing (`modes.Upload(std::vector<std::uint32_t>(8, 0))` in
    // tests/npu/test_fused_decode_e2e.cpp).
    //
    // Cached rather than allocated per launch, for the same reason the basis
    // pool is: it is a constant, and a zero-fill allocation inside a region that
    // has to be capturable is an allocation the graph's pool has to serve. Its
    // head_size slot is 0 because the buffer does not depend on one; the enum
    // tag is what keeps that from colliding with a basis pool.
    const at::Tensor mode_tables =
        cached_constant(CachedConstant::kModeTables, /*head_size=*/0, query.device(), [&] {
            return at::zeros({static_cast<std::int64_t>(layout::kMinBurstBytes) / 4},
                             query.options().dtype(at::kInt));
        });

    // BF16 ONLY, checked rather than mapped. The audited fused decode carries a
    // single pair of template instantiations -- `..._kv4fp8_bf16` and
    // `..._raw_query_kv4fp8_bf16` -- and its dispatch RETURNS without launching
    // anything for any other AscendType. Mapping a non-bf16 query to
    // AscendType::FP16, as this did, would therefore leave `output` whatever
    // at::empty gave it and report success: the worst available failure mode.
    // The fp16 chain was already dead in the wheel, so this is a refusal rather
    // than a regression.
    TORCH_CHECK(query.scalar_type() == at::kBFloat16, op,
                ": the fused decode is bf16 only, got ", query.scalar_type(),
                "; there is no fp16 kernel instance to dispatch to");
    const AscendType element_type = AscendType::BF16;

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    turboquant_mm_fused_decode_opt_raw_query_impl(
        static_cast<std::int32_t>(layout::TurboQuantMode::KV4_FP8), element_type, stream,
        static_cast<std::uint32_t>(grid), query.data_ptr(), pi_signs.data_ptr(), basis_pool.data_ptr(),
        h16.data_ptr(), gate.has_value() ? gate->data_ptr() : nullptr, query_rot.data_ptr(),
        key_cache.data_ptr(), value_cache.data_ptr(), scale_cache.data_ptr(), block_tables.data_ptr(),
        context_lens.data_ptr(), mode_tables.data_ptr(), workspace.data_ptr(), output.data_ptr(),
        tiling.data_ptr(), static_cast<std::uint32_t>(num_tokens),
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
// 4. The decode tiling descriptors
// ---------------------------------------------------------------------------

// Fill the fused decode's tiling buffer for one step, for every layer at once.
//
// WHY THIS IS AN OPERATOR. The descriptors used to be built inside
// npu_turboquant_fused_decode, which meant two things that had to end. They
// were built once per LAYER although they are a function of the batch alone, so
// a sixty-layer model did the same host loop and the same host-to-device copy
// sixty times per decode step. And building them there required reading the
// context lengths on the host from inside the launch, which is illegal on a
// capturing stream (107030) -- the single hardest of the four blockers in
// TURBOQUANT_GRAPH_CAPTURE_RFC.md §1.2.
//
// Making it an op rather than reimplementing build_decode_tiling in Python is
// the whole point: `build_decode_tiling` stays the one authority on what a
// descriptor word means. Python supplies the geometry and a buffer, and derives
// no descriptor. The sizes it does derive are checked here.
//
// `context_lens` is HOST-VISIBLE on purpose and is the one host tensor left in
// the TurboQuant path. It is read by `build_decode_tiling`, which is host C++,
// and the metadata builder already has these lengths on the host -- vLLM's
// metadata keeps the CPU mirror deliberately. This op runs once per step from
// the metadata builder, OUTSIDE any capture, and the graph replays over the
// buffer it fills; a captured launch never calls it. That is why a host tensor
// here is not the thing the RFC removed: what it removed was the host read
// *inside* the launch.
//
// A device-side builder kernel -- which would remove this host work and this
// copy as well -- is rung 5 of the ladder and deliberately not here: it needs a
// new kernel translation unit and a CAModel gate against this function as its
// oracle.
void npu_turboquant_build_decode_tiling(at::Tensor &context_lens, at::Tensor &tiling, int64_t num_heads,
                                        int64_t num_kv_heads, int64_t head_size, int64_t block_size,
                                        int64_t max_blocks, int64_t num_splits, int64_t heads_per_task)
{
    constexpr const char *op = "npu_turboquant_build_decode_tiling";

    TORCH_CHECK(context_lens.scalar_type() == at::kInt && context_lens.dim() == 1 &&
                    context_lens.is_contiguous(),
                op, ": context_lens must be a contiguous 1-D int32 tensor, got ",
                context_lens.scalar_type(), " ", context_lens.sizes());
    TORCH_CHECK(context_lens.device().is_cpu(), op, ": context_lens must be on the host, got ",
                context_lens.device(),
                ". build_decode_tiling is host C++ and reads these values directly; a device "
                "tensor here would mean a device-to-host copy, which is what this op exists to "
                "stop happening inside the launch.");
    TORCH_CHECK(tiling.scalar_type() == at::kInt && tiling.dim() == 1 && tiling.is_contiguous(), op,
                ": tiling must be a contiguous 1-D int32 buffer, got ", tiling.scalar_type(), " ",
                tiling.sizes());

    const std::int64_t num_tokens = context_lens.size(0);
    TORCH_CHECK(num_tokens > 0, op, ": nothing to describe");
    TORCH_CHECK(num_kv_heads > 0 && num_heads > 0 && num_heads % num_kv_heads == 0, op,
                ": num_heads (", num_heads, ") must be a positive multiple of num_kv_heads (",
                num_kv_heads, ")");
    TORCH_CHECK(num_splits > 0 && num_splits <= kDecodeMaxSplits, op, ": num_splits must be 1..",
                kDecodeMaxSplits, ", got ", num_splits);

    tiling::DecodeTilingRequest request = decode_request(num_tokens, num_heads, num_kv_heads, head_size,
                                                         block_size, max_blocks, num_splits, heads_per_task);
    const tiling::DecodeLaunchSplit split =
        tiling::decode_launch_split(request, static_cast<std::uint32_t>(kDecodeMaxGrid));
    TORCH_CHECK(split.tasks > 0, op, ": nothing to describe");
    request.grid = split.grid;
    request.tasks_per_block = split.tasks_per_block;
    request.reduce_tasks_per_block = split.reduce_tasks_per_block;

    const std::int64_t words =
        static_cast<std::int64_t>(tiling::decode_tiling_words(request.tokens, request.splits, request.grid));
    TORCH_CHECK(tiling.numel() >= words, op, ": tiling holds ", tiling.numel(),
                " int32 words; ", num_tokens, " tokens x ", num_splits, " splits with grid ",
                request.grid, " needs ", words);

    // Staged on the host and copied once. `build_decode_tiling` writes 64-byte
    // descriptors by offset and zeroes the whole buffer first, so it needs a
    // 64-byte-aligned host base -- which is why this is not a write straight
    // into a pinned view of the caller's buffer. Torch's CPU allocator gives
    // 64 bytes; the check is here because nothing else would say so.
    const at::Tensor host = at::empty({words}, at::TensorOptions().dtype(at::kInt));
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(host.data_ptr()) %
                        alignof(tiling::TurboQuantDecodeTilingData) ==
                    0,
                op, ": the host staging buffer is not ",
                alignof(tiling::TurboQuantDecodeTilingData), "-byte aligned");

    const tiling::DecodeTilingStatus status = tiling::build_decode_tiling(
        request, static_cast<const std::int32_t *>(context_lens.data_ptr()),
        /*block_table=*/nullptr, host.data_ptr());
    TORCH_CHECK(status == tiling::DecodeTilingStatus::kOk, op, ": ",
                tiling::decode_tiling_status_text(status));

    // Only the prefix this launch describes. The tail of an upper-bound-sized
    // buffer is never read: the device computes the dispatch records' base from
    // `tokens * splits`, exactly as decode_dispatch_word_base does.
    tiling.narrow(0, 0, words).copy_(host, /*non_blocking=*/false);
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
                                              at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, double)
{
}

void npu_turboquant_reshape_scatter_aiv_meta(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                                             at::Tensor &, at::Tensor &, at::Tensor &, double)
{
}

void npu_turboquant_fused_decode_meta(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                                      const std::optional<at::Tensor> &, at::Tensor &, at::Tensor &,
                                      at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                                      at::Tensor &, at::Tensor &, int64_t, int64_t, int64_t, int64_t,
                                      int64_t, double, double)
{
}

// The tiling builder's meta kernel must NOT call build_decode_tiling either:
// under FakeTensor there is no storage behind `context_lens` to read, and the
// buffer it would write has no storage to write to.
void npu_turboquant_build_decode_tiling_meta(at::Tensor &, at::Tensor &, int64_t, int64_t, int64_t,
                                             int64_t, int64_t, int64_t, int64_t)
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
    // key_cache    [num_blocks, block_size, num_kv_heads, head_size / 2] int8,
    //              holding the NZ-tiled plane the fused decode reads:
    //              [64-row tile][kv head][32-byte column group][tile row][byte]
    //              (NzTiledPackedByte), one byte carrying channels (c, c + 128)
    //              of ONE token. The shape is the allocation; the fractal
    //              inside it is the contract.
    // value_cache  the same extent and the same NZ addressing as the key plane
    // scale_cache  fp32, slot-major records:
    //              index = slot * NzScaleSlotFloats(num_kv_heads)
    //                    + kv * num_kv_heads + head      (kv 0 keys / 1 values)
    //              At num_kv_heads 4 a record is exactly 8 floats, so this is
    //              the same extent as the old [2, num_blocks, num_kv_heads,
    //              block_size] plane, addressed differently.
    // slot_mapping [num_tokens] int32. Each 16-token strip must be consecutive
    //              from a 16-aligned first slot -- NOT page-aligned: a strip
    //              writes its own 16 tile rows of each column group, so strips
    //              may jump between pages freely. block_size is any multiple of
    //              64 (the decode wants a power-of-two multiple).
    // pi_signs     folded into `tables`; an empty tensor says so
    // tables       2 * (head_size / 32) row-major 32x32 BF16 blocks of +-1
    ops.def(
        "npu_turboquant_reshape_and_cache_a8(Tensor key, Tensor value, Tensor! key_cache, "
        "Tensor! value_cache, Tensor! scale_cache, Tensor slot_mapping, Tensor pi_signs, "
        "Tensor tables, float inv_sqrt_len) -> ()");
    ops.impl("npu_turboquant_reshape_and_cache_a8", c10::kPrivateUse1,
             &vllm_ascend::turboquant_bindings::npu_turboquant_reshape_and_cache_a8);
    ops.impl("npu_turboquant_reshape_and_cache_a8", c10::kMeta,
             &vllm_ascend::turboquant_bindings::meta::npu_turboquant_reshape_and_cache_a8_meta);

    // One token per request, arbitrary slots. The value plane's write is a
    // read-modify-write of one 2048-byte strip, which is the reason this op
    // exists: the PyTorch golden model cannot do it safely.
    //
    // STRIP CACHE ONLY. This writer has no NZ variant in turboquant-core: it
    // produces the RFC 3.1 strip layout the fused decode cannot read, so it
    // must not write a cache the NZ batch writer and the decode share. The
    // Python adapter keeps it off an NZ cache; an NZ scatter kernel is harness
    // work that has not landed.
    ops.def(
        "npu_turboquant_reshape_scatter_aiv(Tensor key, Tensor value, Tensor! key_cache, "
        "Tensor! value_cache, Tensor! scale_cache, Tensor slot_mapping, Tensor pi_signs, "
        "float inv_sqrt_len) -> ()");
    ops.impl("npu_turboquant_reshape_scatter_aiv", c10::kPrivateUse1,
             &vllm_ascend::turboquant_bindings::npu_turboquant_reshape_scatter_aiv);
    ops.impl("npu_turboquant_reshape_scatter_aiv", c10::kMeta,
             &vllm_ascend::turboquant_bindings::meta::npu_turboquant_reshape_scatter_aiv_meta);

    // query/output [num_tokens, num_heads, 256] bf16. The schema admits fp16 so
    //              the error names the dtype rather than the schema, but the op
    //              refuses it: there is no fp16 kernel instance and the
    //              dispatch would return without launching.
    // rot_tables   the BF16 blocks above, checked identically but NOT forwarded:
    //              the device slot it used to fill is the basis codec's constant
    //              pool, which this op builds itself; h16 the 16x16 fp16 +-1 matrix
    // gate         read only at output_stage 2
    // context_lens int32 ON THE QUERY'S DEVICE. Refused on the host: the op used
    //              to copy whichever way the caller's tensor pointed, once per
    //              layer per step, and that copy is illegal on a capturing
    //              stream. Pass AscendTurboQuantMetadata.turboquant_context_lens.
    // tiling       int32, >= decode_tiling_words(tokens, splits, grid) where grid
    //              is tiling::decode_launch_split's. Filled by
    //              npu_turboquant_build_decode_tiling once per step, for every
    //              layer; read-only here.
    // workspace    fp32[tokens * heads * splits * (head_size + kPartialTail)],
    //              the Split-K partials. query_rot fp32[tokens * heads *
    //              head_size], the rotated query image the prologue writes and
    //              the decode reads -- it is not `query`.
    // output_stage TurboQuantOutputStage: 0 rotated basis, 1 unrotated, 2 gated
    //
    // THE THREE BUFFERS ARE THE CALLER'S NOW. They were allocated inside the
    // launch, which gave each a fresh address every step; a captured ACL graph
    // replays the address it recorded. The op still derives every size and
    // checks what it was handed, so a caller that got one wrong gets a named
    // refusal rather than an out-of-bounds device read. What remains the op's
    // own are the two codec constant pools, which are pure functions of
    // head_size and are cached for the process.
    ops.def(
        "npu_turboquant_fused_decode(Tensor query, Tensor pi_signs, Tensor rot_tables, Tensor h16, "
        "Tensor? gate, Tensor key_cache, Tensor value_cache, Tensor scale_cache, Tensor block_tables, "
        "Tensor context_lens, Tensor tiling, Tensor! workspace, Tensor! query_rot, Tensor! output, "
        "int num_heads, int num_kv_heads, int num_splits, "
        "int heads_per_task, int output_stage, float scale, float inv_sqrt_len) -> ()");
    ops.impl("npu_turboquant_fused_decode", c10::kPrivateUse1,
             &vllm_ascend::turboquant_bindings::npu_turboquant_fused_decode);
    ops.impl("npu_turboquant_fused_decode", c10::kMeta,
             &vllm_ascend::turboquant_bindings::meta::npu_turboquant_fused_decode_meta);

    // context_lens int32 ON THE HOST, [tokens]. The one host tensor left in the
    //              TurboQuant path, and it is read by host C++
    //              (build_decode_tiling) from outside any capture -- this op
    //              runs once per step in the metadata builder, never inside a
    //              captured launch.
    // tiling       int32, >= decode_tiling_words(...). Sized by the caller from
    //              turboquant_tiling_upper_bound_words, which is why the check
    //              is >= and not ==.
    //
    // Registered for both device keys and with one implementation, because the
    // dispatch key is not a property of the work: it is host arithmetic plus one
    // copy, and the only device thing about it is where `tiling` lives. In
    // service that is the NPU, so the key computed from the argument list is
    // PrivateUse1; in a device-free test both tensors are on the host and it is
    // CPU. Same function either way.
    ops.def(
        "npu_turboquant_build_decode_tiling(Tensor context_lens, Tensor! tiling, int num_heads, "
        "int num_kv_heads, int head_size, int block_size, int max_blocks, int num_splits, "
        "int heads_per_task) -> ()");
    ops.impl("npu_turboquant_build_decode_tiling", c10::kPrivateUse1,
             &vllm_ascend::turboquant_bindings::npu_turboquant_build_decode_tiling);
    ops.impl("npu_turboquant_build_decode_tiling", c10::kCPU,
             &vllm_ascend::turboquant_bindings::npu_turboquant_build_decode_tiling);
    ops.impl("npu_turboquant_build_decode_tiling", c10::kMeta,
             &vllm_ascend::turboquant_bindings::meta::npu_turboquant_build_decode_tiling_meta);
}
