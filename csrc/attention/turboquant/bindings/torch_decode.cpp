#include "torch_bindings_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "../common/e8_component.hpp"
#include "../common/e8_decode.hpp"
#include "../common/turboquant_launch.h"
#include "../common/turboquant_tiling.h"

namespace vllm_ascend::turboquant_bindings {
namespace {
namespace tiling = ::ascend_micro::fused_decode;

constexpr std::int64_t kDecodeMaxKvHeads = 4;
constexpr std::int64_t kDecodeMaxGrid = 32;
constexpr std::int64_t kDecodeMaxSplits = 8;
constexpr std::int64_t kDecodeMaxHeadsPerTask = 16;

constexpr std::int64_t kCodecEarlyStages = 3;
constexpr std::int64_t kCodecLevels = 16;
constexpr std::int64_t kCodecPoolWordsPerChannel = 9;

inline ::ascend_micro::fused_decode::DecodeTilingRequest decode_request(
    const std::int64_t num_tokens, const std::int64_t num_heads, const std::int64_t num_kv_heads,
    const std::int64_t head_size, const std::int64_t block_size, const std::int64_t max_blocks,
    const std::int64_t num_splits, const std::int64_t heads_per_task) {
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

enum class CachedConstant : int {
    kBasisPool = 0,
    kModeTables = 1,
};

using ConstantKey = std::tuple<CachedConstant, std::int64_t, c10::DeviceIndex>;

inline at::Tensor cached_constant(const CachedConstant which, const std::int64_t head_size,
                                  const at::Device& device, const std::function<at::Tensor()>& build) {
    static std::mutex guard;
    static std::map<ConstantKey, at::Tensor> cache;
    const ConstantKey key{which, head_size, device.index()};
    const std::lock_guard<std::mutex> held(guard);
    auto found = cache.find(key);
    if (found != cache.end() && found->second.device() == device) {
        return found->second;
    }
    at::Tensor built = build();
    cache[key] = built;
    return built;
}
}

void npu_turboquant_fused_decode(at::Tensor& query, at::Tensor& pi_signs, at::Tensor& rot_tables,
                                 at::Tensor& h16, const std::optional<at::Tensor>& gate,
                                 at::Tensor& key_cache, at::Tensor& value_cache, at::Tensor& scale_cache,
                                 at::Tensor& block_tables, at::Tensor& context_lens, at::Tensor& tiling,
                                 at::Tensor& workspace, at::Tensor& query_rot, at::Tensor& output,
                                 int64_t num_heads, int64_t num_kv_heads, int64_t num_splits,
                                 int64_t heads_per_task, int64_t output_stage, double scale,
                                 double inv_sqrt_len, int64_t decode_unpack, int64_t fused_context_limit) {
    constexpr const char* op = "npu_turboquant_fused_decode";
    TORCH_CHECK(fused_context_limit >= 0 && fused_context_limit <= INT32_MAX, op,
                ": fused_context_limit must be in [0, INT32_MAX], got ", fused_context_limit);

    TORCH_CHECK(decode_unpack == static_cast<int64_t>(layout::DecodeUnpack::kNative) ||
                    decode_unpack == static_cast<int64_t>(layout::DecodeUnpack::kLloydMaxLut),
                op, ": decode_unpack is 0 (kNative) or 3 (kLloydMaxLut), got ", decode_unpack);

    TORCH_CHECK(query.dim() == 3 && output.dim() == 3, op,
                ": query and output must be [num_tokens, num_heads, head_size]");
    TORCH_CHECK(query.sizes() == output.sizes(), op, ": query and output must have the same shape");
    TORCH_CHECK(query.is_contiguous() && output.is_contiguous(), op, ": query and output must be contiguous");
    TORCH_CHECK(query.scalar_type() == output.scalar_type(), op,
                ": query and output must share an element type, got ", query.scalar_type(), " and ",
                output.scalar_type());
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16, op,
                ": the operator has kernel instances for fp16 and bf16 only, got ", query.scalar_type());
    TORCH_CHECK(query.data_ptr() != output.data_ptr(), op, ": query and output alias at ", query.data_ptr(),
                "; the decode reads the query while it writes output, so an in-place call would "
                "overwrite operands still in flight");

    const std::int64_t num_tokens = query.size(0);
    const std::int64_t head_size = query.size(2);
    TORCH_CHECK(query.size(1) == num_heads, op, ": query has ", query.size(1), " heads, num_heads says ",
                num_heads);

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
    TORCH_CHECK(
        output_stage != static_cast<int64_t>(layout::TurboQuantOutputStage::kGated) || gate.has_value(), op,
        ": output_stage 2 divides by 1 + exp(-gate) and needs the gate buffer");

    TORCH_CHECK(key_cache.dim() == 4 && key_cache.sizes() == value_cache.sizes(), op,
                ": the packed planes must both be 4-D and identically shaped");
    TORCH_CHECK(key_cache.scalar_type() == at::kChar && value_cache.scalar_type() == at::kChar, op,
                ": the packed planes are int8 nibble pairs");
    TORCH_CHECK(scale_cache.scalar_type() == at::kFloat, op, ": the scale plane is fp32");
    TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous() && scale_cache.is_contiguous(), op,
                ": every cache plane must be contiguous; the device addresses them as flat runs");
    const std::int64_t num_blocks = key_cache.size(0);
    const std::int64_t block_size = key_cache.size(1);
    TORCH_CHECK(key_cache.size(2) == num_kv_heads && key_cache.size(3) == head_size / 2, op,
                ": packed plane is [", num_blocks, ", ", block_size, ", ", key_cache.size(2), ", ",
                key_cache.size(3), "], expected [..., ", num_kv_heads, ", ", head_size / 2,
                "] from the scalar geometry");
    TORCH_CHECK(num_blocks > 0 && num_blocks <= static_cast<std::int64_t>(UINT32_MAX), op,
                ": the cache plane holds ", num_blocks,
                " pages, which the device's uint32 page "
                "index cannot name");
    TORCH_CHECK((2 * num_kv_heads) % 8 == 0, op,
                ": the slot-major scale plane needs 2 * num_kv_heads to be a multiple of 8, got ",
                num_kv_heads, " KV heads, whose NzScaleSlotFloats record is ",
                layout::NzScaleSlotFloats(static_cast<std::uint32_t>(num_kv_heads)),
                " floats while the physical plane holds only 2 * num_kv_heads");
    const std::int64_t scale_floats =
        static_cast<std::int64_t>(layout::NzScaleSlotFloats(static_cast<std::uint32_t>(num_kv_heads))) *
        num_blocks * block_size;
    TORCH_CHECK(scale_cache.numel() >= scale_floats, op, ": the scale plane holds ", scale_cache.numel(),
                " floats, needs at least ", scale_floats, " (", num_blocks, " blocks x ", block_size,
                " tokens x ", layout::NzScaleSlotFloats(static_cast<std::uint32_t>(num_kv_heads)),
                " floats a slot)");
    TORCH_CHECK(
        block_size > 0 && block_size % layout::kCubeTileRows == 0 && (block_size & (block_size - 1)) == 0, op,
        ": the closed-form tile cursor needs block_size to be a power-of-two multiple of ",
        layout::kCubeTileRows, ", got ", block_size,
        ". The NZ-tiled plane this operator reads strides its tile rows by ", layout::kCubeTileRows,
        ", so a smaller page cannot address it at all.");

    TORCH_CHECK(block_tables.dim() == 2 && block_tables.scalar_type() == at::kInt, op,
                ": block_tables must be an int32 [num_tokens, max_blocks_per_seq]");
    TORCH_CHECK(block_tables.is_contiguous(), op,
                ": block_tables must be contiguous; the device indexes it as token * max_blocks");
    TORCH_CHECK(block_tables.size(0) >= num_tokens, op, ": block_tables has ", block_tables.size(0),
                " rows for ", num_tokens, " tokens");
    const std::int64_t max_blocks_per_seq = block_tables.size(1);
    TORCH_CHECK(max_blocks_per_seq > 0, op, ": block_tables has no columns");
    TORCH_CHECK(context_lens.scalar_type() == at::kInt && context_lens.dim() == 1 &&
                    context_lens.size(0) >= num_tokens,
                op, ": context_lens must be an int32 [num_tokens], got ", context_lens.scalar_type(), " ",
                context_lens.sizes());

    TORCH_CHECK(pi_signs.scalar_type() == at::kFloat && pi_signs.numel() == head_size, op,
                ": pi_signs must be ", head_size, " fp32 entries, got ", pi_signs.numel());
    check_tables(rot_tables, head_size, op, "rot_tables");
    TORCH_CHECK(h16.scalar_type() == at::kHalf &&
                    h16.numel() == static_cast<std::int64_t>(layout::kRotateQH16Elements),
                op, ": h16 is the Cube's resident ", layout::kRotateQH16Elements,
                "-element fp16 sign matrix (16x16, +-1), got ", h16.scalar_type(), " with ", h16.numel(),
                " elements");

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
    const std::int64_t prologue_cube_chunk_vectors =
        prologue_vectors_per_block >= static_cast<std::int64_t>(layout::kQueryBasisMinCubeVectors)
            ? static_cast<std::int64_t>(layout::kQueryBasisCubeVectors)
            : 0;

    TORCH_CHECK(context_lens.device() == query.device(), op, ": context_lens is on ", context_lens.device(),
                " but the query is on ", query.device(),
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
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(tiling.data_ptr()) % alignof(tiling::TurboQuantDecodeTilingData) ==
            0,
        op, ": the tiling buffer is not ", alignof(tiling::TurboQuantDecodeTilingData), "-byte aligned");

    const std::int64_t workspace_floats =
        num_tokens * num_heads * num_splits * (head_size + static_cast<std::int64_t>(layout::kPartialTail));
    TORCH_CHECK(workspace.scalar_type() == at::kFloat && workspace.is_contiguous(), op,
                ": workspace must be contiguous fp32, got ", workspace.scalar_type());
    TORCH_CHECK(workspace.numel() == workspace_floats, op, ": workspace holds ", workspace.numel(),
                " floats, this launch needs exactly ", workspace_floats, " (", num_tokens, " x ", num_heads,
                " x ", num_splits, " x (", head_size, " + ", layout::kPartialTail, "))");
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

    const std::int64_t basis_pool_words = kCodecPoolWordsPerChannel * head_size + kCodecLevels;
    TORCH_CHECK(basis_pool_words * 4 % static_cast<std::int64_t>(kGmBurstBytes) == 0, op,
                ": the basis codec pool is ", basis_pool_words * 4, " bytes, not a whole number of ",
                kGmBurstBytes, "-byte bursts");
    const at::Tensor basis_pool = cached_constant(CachedConstant::kBasisPool, head_size, query.device(), [&] {
        at::Tensor host = at::zeros({basis_pool_words}, at::TensorOptions().dtype(at::kInt));
        std::int32_t* words = host.data_ptr<std::int32_t>();
        std::int64_t at_word = 0;
        for (std::int64_t stage = 0; stage < kCodecEarlyStages; ++stage) {
            const std::int64_t stride = std::int64_t{1} << stage;
            for (std::int64_t i = 0; i < head_size; ++i) {
                const float sign = (i & stride) == 0 ? 1.0f : -1.0f;
                std::memcpy(&words[at_word + i], &sign, sizeof(float));
            }
            at_word += head_size;
            for (std::int64_t i = 0; i < head_size; ++i)
                words[at_word + i] = static_cast<std::int32_t>((i ^ stride) * sizeof(float));
            at_word += head_size;
        }
        TORCH_CHECK(at_word == 2 * kCodecEarlyStages * head_size, op,
                    ": the basis pool's early-stage span is ", at_word, " words, expected ",
                    2 * kCodecEarlyStages * head_size);
        return host.to(query.device(), false);
    });
    check_burst_aligned(basis_pool, op, "basis codec pool");

    const at::Tensor mode_tables = cached_constant(CachedConstant::kModeTables, 0, query.device(), [&] {
        at::Tensor host = at::zeros({static_cast<std::int64_t>(layout::kMinBurstBytes) / 4},
                                    at::TensorOptions().dtype(at::kInt));
        std::int32_t* words = host.data_ptr<std::int32_t>();
        using LloydMax = layout::TurboQuantLloydMaxTraits<layout::TurboQuantMode::KV4_FP8>;
        for (std::int64_t i = 0; i < LloydMax::kLevels; ++i)
            std::memcpy(&words[i], &LloydMax::kCentroids[i], sizeof(float));
        return host.to(query.device(), false);
    });

    TORCH_CHECK(query.scalar_type() == at::kBFloat16, op, ": the fused decode is bf16 only, got ",
                query.scalar_type(), "; there is no fp16 kernel instance to dispatch to");
    const AscendType element_type = AscendType::BF16;

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    turboquant_mm_fused_decode_opt_raw_query_impl(
        static_cast<std::int32_t>(layout::TurboQuantMode::KV4_FP8), element_type, stream,
        static_cast<std::uint32_t>(grid), query.data_ptr(), pi_signs.data_ptr(), basis_pool.data_ptr(),
        h16.data_ptr(), gate.has_value() ? gate->data_ptr() : nullptr, query_rot.data_ptr(),
        key_cache.data_ptr(), value_cache.data_ptr(), scale_cache.data_ptr(), block_tables.data_ptr(),
        context_lens.data_ptr(), mode_tables.data_ptr(), workspace.data_ptr(), output.data_ptr(),
        tiling.data_ptr(), static_cast<std::uint32_t>(num_tokens), static_cast<std::uint32_t>(num_heads),
        static_cast<std::uint32_t>(num_kv_heads), static_cast<std::uint32_t>(head_size),
        static_cast<std::uint32_t>(block_size), static_cast<std::uint32_t>(max_blocks_per_seq),
        static_cast<std::uint32_t>(num_splits), static_cast<std::uint32_t>(heads_per_task),
        static_cast<std::uint32_t>(tasks_per_block), static_cast<std::uint32_t>(reduce_tasks_per_block),
        static_cast<std::uint32_t>(prologue_vectors_per_block),
        static_cast<std::uint32_t>(prologue_cube_chunk_vectors), static_cast<std::uint32_t>(output_stage),
        static_cast<std::uint32_t>(fused_context_limit), static_cast<float>(scale),
        static_cast<float>(inv_sqrt_len), static_cast<std::uint32_t>(num_blocks),
        static_cast<std::uint32_t>(decode_unpack));
}

void npu_turboquant_build_decode_tiling(at::Tensor& context_lens, at::Tensor& tiling, int64_t num_heads,
                                        int64_t num_kv_heads, int64_t head_size, int64_t block_size,
                                        int64_t max_blocks, int64_t num_splits, int64_t heads_per_task,
                                        int64_t fused_context_limit) {
    constexpr const char* op = "npu_turboquant_build_decode_tiling";
    TORCH_CHECK(fused_context_limit >= 0 && fused_context_limit <= INT32_MAX, op,
                ": fused_context_limit must be in [0, INT32_MAX], got ", fused_context_limit);

    TORCH_CHECK(
        context_lens.scalar_type() == at::kInt && context_lens.dim() == 1 && context_lens.is_contiguous(), op,
        ": context_lens must be a contiguous 1-D int32 tensor, got ", context_lens.scalar_type(), " ",
        context_lens.sizes());
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
    TORCH_CHECK(num_kv_heads > 0 && num_heads > 0 && num_heads % num_kv_heads == 0, op, ": num_heads (",
                num_heads, ") must be a positive multiple of num_kv_heads (", num_kv_heads, ")");
    TORCH_CHECK(num_splits > 0 && num_splits <= kDecodeMaxSplits, op, ": num_splits must be 1..",
                kDecodeMaxSplits, ", got ", num_splits);
    TORCH_CHECK(block_size > 0 && max_blocks > 0, op, ": block_size and max_blocks must be positive, got ",
                block_size, " and ", max_blocks);
    TORCH_CHECK(max_blocks <= INT64_MAX / block_size, op,
                ": max_blocks * block_size overflows int64: ", max_blocks, " x ", block_size);
    const std::int64_t max_context = max_blocks * block_size;
    const auto* host_lengths = static_cast<const std::int32_t*>(context_lens.data_ptr());
    for (std::int64_t token = 0; token < num_tokens; ++token) {
        TORCH_CHECK(host_lengths[token] >= 0 && host_lengths[token] <= max_context, op, ": context_lens[",
                    token, "] is ", host_lengths[token],
                    "; a decode length must be in [0, max_blocks * block_size] = [0, ", max_context, "]");
    }

    tiling::DecodeTilingRequest request = decode_request(num_tokens, num_heads, num_kv_heads, head_size,
                                                         block_size, max_blocks, num_splits, heads_per_task);
    request.fused_context_limit = static_cast<std::uint32_t>(fused_context_limit);
    const tiling::DecodeLaunchSplit split =
        tiling::decode_launch_split(request, static_cast<std::uint32_t>(kDecodeMaxGrid));
    TORCH_CHECK(split.tasks > 0, op, ": nothing to describe");
    request.grid = split.grid;
    request.tasks_per_block = split.tasks_per_block;
    request.reduce_tasks_per_block = split.reduce_tasks_per_block;

    const std::int64_t words =
        static_cast<std::int64_t>(tiling::decode_tiling_words(request.tokens, request.splits, request.grid));
    TORCH_CHECK(tiling.numel() >= words, op, ": tiling holds ", tiling.numel(), " int32 words; ", num_tokens,
                " tokens x ", num_splits, " splits with grid ", request.grid, " needs ", words);

    const bool upload_to_device = !tiling.device().is_cpu();
    const at::Tensor host = at::empty(
        {words}, at::TensorOptions().device(at::kCPU).dtype(at::kInt).pinned_memory(upload_to_device));
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(host.data_ptr()) % alignof(tiling::TurboQuantDecodeTilingData) == 0,
        op, ": the host staging buffer is not ", alignof(tiling::TurboQuantDecodeTilingData),
        "-byte aligned");

    const tiling::DecodeTilingStatus status = tiling::build_decode_tiling(
        request, static_cast<const std::int32_t*>(context_lens.data_ptr()), nullptr, host.data_ptr());
    TORCH_CHECK(status == tiling::DecodeTilingStatus::kOk, op, ": ",
                tiling::decode_tiling_status_text(status));

    tiling.narrow(0, 0, words).copy_(host, upload_to_device);
}

namespace meta {
void npu_turboquant_fused_decode_meta(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                                      const std::optional<at::Tensor>&, at::Tensor&, at::Tensor&, at::Tensor&,
                                      at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                                      at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, double,
                                      double, int64_t, int64_t) {}

void npu_turboquant_build_decode_tiling_meta(at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t,
                                             int64_t, int64_t, int64_t, int64_t) {}
}
}

namespace vllm_ascend::turboquant_e8 {

constexpr std::uint64_t kComponentRows = 16;
static_assert(e8_packet_abi::kRows % kComponentRows == 0);

template <std::size_t N>
static void disjoint_outputs(const at::Tensor* const (&tensors)[N], std::size_t first_output) {
    if (tensors[0]->is_meta()) return;
    for (std::size_t i = first_output; i < N; ++i) {
        const auto begin = reinterpret_cast<std::uintptr_t>(tensors[i]->data_ptr());
        const auto end = begin + tensors[i]->nbytes();
        for (std::size_t j = 0; j < i; ++j) {
            const auto other = reinterpret_cast<std::uintptr_t>(tensors[j]->data_ptr());
            TORCH_CHECK(end <= other || other + tensors[j]->nbytes() <= begin,
                        "E8 outputs/witnesses must not overlap another argument");
        }
    }
}

static void tensor(const at::Tensor& t, at::ScalarType type, std::int64_t elements, const at::Tensor& anchor,
                   const char* name) {
    TORCH_CHECK(t.scalar_type() == type && t.is_contiguous() && t.numel() == elements, name,
                ": unexpected dtype, size or strides");
    TORCH_CHECK(t.device() == anchor.device(), name, ": device differs from query/labels");
    if (!t.is_meta()) {
        TORCH_CHECK(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 32 == 0, name,
                    ": expected a 32-byte aligned buffer");
    }
}

static void validate_reader(const at::Tensor& query, const at::Tensor& key_packets,
                            const at::Tensor& value_packets, const at::Tensor& packet_offsets,
                            const at::Tensor& output, const at::Tensor& key_witness,
                            const at::Tensor& value_witness, const at::Tensor& score_witness,
                            std::int64_t tiles, std::int64_t last_rows, double scale) {
    TORCH_CHECK(tiles >= 4 && tiles <= e8_decode::kMaxTiles && tiles % 2 == 0,
                "E8 reader requires an even tile count in [4, 256]");
    TORCH_CHECK(last_rows > 0 && last_rows <= e8_decode::kRows && std::isfinite(static_cast<float>(scale)),
                "invalid E8 live rows or attention scale");
    TORCH_CHECK(query.dim() == 2 && query.size(0) == 16 && query.size(1) == 256,
                "E8 query must be cache-basis FP32 [16, 256], rows 6..15 zero");
    tensor(query, at::kFloat, 16 * 256, query, "query");
    TORCH_CHECK(key_packets.dim() == 2 && value_packets.sizes() == key_packets.sizes() &&
                    key_packets.size(0) > 0 && key_packets.size(1) == e8_packet_abi::kPacketBytes,
                "E8 planes must be UINT8 [packets, 4352], with embedded FP32 scales");
    tensor(key_packets, at::kByte, key_packets.numel(), query, "key_packets");
    tensor(value_packets, at::kByte, key_packets.numel(), query, "value_packets");

    TORCH_CHECK(packet_offsets.dim() == 1, "E8 offsets must be a flat INT64 byte-offset table");

    tensor(packet_offsets, at::kLong, ((tiles + 3) / 4) * 4, query, "packet_offsets");
    tensor(output, at::kBFloat16, 6 * 256, query, "output");
    tensor(key_witness, at::kBFloat16, tiles * 64 * 256, query, "key_witness");
    tensor(value_witness, at::kBFloat16, tiles * 64 * 256, query, "value_witness");
    tensor(score_witness, at::kFloat, tiles * 6 * 64, query, "score_witness");
    const at::Tensor* const buffers[] = {&query,  &key_packets, &value_packets, &packet_offsets,
                                         &output, &key_witness, &value_witness, &score_witness};
    disjoint_outputs(buffers, 4);
}

void reader(at::Tensor& query, at::Tensor& key_packets, at::Tensor& value_packets, at::Tensor& packet_offsets,
            at::Tensor& output, at::Tensor& key_witness, at::Tensor& value_witness, at::Tensor& score_witness,
            std::int64_t tiles, std::int64_t last_rows, double scale) {
    validate_reader(query, key_packets, value_packets, packet_offsets, output, key_witness, value_witness,
                    score_witness, tiles, last_rows, scale);
    const e8_decode::Launch launch{static_cast<std::uint32_t>(tiles), static_cast<std::uint32_t>(last_rows),
                                   static_cast<float>(scale)};
    const e8_decode::Buffers buffers{key_packets.data_ptr(),    value_packets.data_ptr(),
                                     packet_offsets.data_ptr(), query.data_ptr(),
                                     output.data_ptr(),         key_witness.data_ptr(),
                                     value_witness.data_ptr(),  score_witness.data_ptr()};
    launch_e8_fused_decode_fp32(c10_npu::getCurrentNPUStream().stream(), &launch, &buffers);
}

void reader_meta(at::Tensor& query, at::Tensor& key_packets, at::Tensor& value_packets,
                 at::Tensor& packet_offsets, at::Tensor& output, at::Tensor& key_witness,
                 at::Tensor& value_witness, at::Tensor& score_witness, std::int64_t tiles,
                 std::int64_t last_rows, double scale) {
    validate_reader(query, key_packets, value_packets, packet_offsets, output, key_witness, value_witness,
                    score_witness, tiles, last_rows, scale);
}

static void validate_dequant(const at::Tensor& labels, const at::Tensor& scales, const at::Tensor& bf16,
                             const at::Tensor& int8) {
    tensor(labels, at::kUInt16, 512, labels, "labels");
    tensor(scales, at::kFloat, kComponentRows, labels, "scales");
    tensor(bf16, at::kBFloat16, 4096, labels, "bf16");
    tensor(int8, at::kChar, 4096, labels, "int8");
    const at::Tensor* const buffers[] = {&labels, &scales, &bf16, &int8};
    disjoint_outputs(buffers, 2);
}

void dequant(at::Tensor& labels, at::Tensor& scales, at::Tensor& bf16, at::Tensor& int8) {
    validate_dequant(labels, scales, bf16, int8);
    launch_e8_parallel_component(
        c10_npu::getCurrentNPUStream().stream(), static_cast<std::uint16_t*>(labels.data_ptr()),
        static_cast<float*>(scales.data_ptr()), static_cast<std::uint16_t*>(bf16.data_ptr()),
        static_cast<std::int8_t*>(int8.data_ptr()));
}

void dequant_meta(at::Tensor& labels, at::Tensor& scales, at::Tensor& bf16, at::Tensor& int8) {
    validate_dequant(labels, scales, bf16, int8);
}

static void validate_dequant_packet(const at::Tensor& packets, const at::Tensor& bf16, const at::Tensor& int8) {
    TORCH_CHECK(packets.dim() == 2 && packets.size(0) > 0 && packets.size(1) == e8_packet_abi::kPacketBytes,
                "E8 packets must be UINT8 [packets, 4352], with embedded FP32 scales");
    TORCH_CHECK(bf16.dim() == 3 && bf16.size(0) == packets.size(0) && bf16.size(1) == e8_packet_abi::kRows &&
                    bf16.size(2) == e8_packet_abi::kChannels && int8.sizes() == bf16.sizes(),
                "E8 packet dequantization outputs must both have shape [packets, 64, 256]");
    tensor(packets, at::kByte, packets.numel(), packets, "packets");
    tensor(bf16, at::kBFloat16, bf16.numel(), packets, "bf16");
    tensor(int8, at::kChar, bf16.numel(), packets, "int8");
    const at::Tensor* const buffers[] = {&packets, &bf16, &int8};
    disjoint_outputs(buffers, 1);
}

void dequant_packet(at::Tensor& packets, at::Tensor& bf16, at::Tensor& int8) {
    validate_dequant_packet(packets, bf16, int8);
    auto* stream = c10_npu::getCurrentNPUStream().stream();
    auto* packet_bytes = static_cast<std::uint8_t*>(packets.data_ptr());
    auto* bf16_rows = static_cast<std::uint16_t*>(bf16.data_ptr());
    auto* int8_rows = static_cast<std::int8_t*>(int8.data_ptr());
    const auto packet_count = static_cast<std::uint64_t>(packets.size(0));
    for (std::uint64_t i = 0; i < packet_count; ++i) {
        auto* packet = packet_bytes + i * e8_packet_abi::kPacketBytes;
        for (std::uint64_t row = 0; row < e8_packet_abi::kRows; row += kComponentRows) {
            auto* labels = reinterpret_cast<std::uint16_t*>(packet + row * e8_packet_abi::kLabelRowBytes);
            auto* scales = reinterpret_cast<float*>(packet + e8_packet_abi::kScaleOffset + row * sizeof(float));
            const std::uint64_t output = (i * e8_packet_abi::kRows + row) * e8_packet_abi::kChannels;
            launch_e8_parallel_component(stream, labels, scales, bf16_rows + output, int8_rows + output);
        }
    }
}

void dequant_packet_meta(at::Tensor& packets, at::Tensor& bf16, at::Tensor& int8) {
    validate_dequant_packet(packets, bf16, int8);
}
}
