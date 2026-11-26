#include "torch_bindings_internal.h"

#include <algorithm>
#include <cstdlib>

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "../common/turboquant_launch.h"
#include "../common/turboquant_wave_protocol.hpp"

namespace vllm_ascend::turboquant_bindings {
namespace {
constexpr std::int64_t kTokensPerStrip = 16;
constexpr std::int64_t kChampionTokensPerCore = 6 * kTokensPerStrip;
constexpr std::int64_t kE8QuantMode = 2;

void check_ingress_quant_mode(std::int64_t quant_mode, const char* op) {
    TORCH_CHECK(quant_mode != kE8QuantMode, op,
                ": device E8 encoding is gated by the r3/revision2 canary. Supply packed 4352-byte E8 K/V planes.");
    TORCH_CHECK(quant_mode == 0 || quant_mode == 1, op, ": quant_mode is 0 (uniform) or 1 (Lloyd-Max), got ",
                quant_mode);
}

void check_launch_u32(std::int64_t value, const char* op, const char* name) {
    TORCH_CHECK(turboquant_wave::FitsLaunchU32(value), op, ": ", name, " exceeds the uint32 launch ABI");
}

struct Activations {
    std::int64_t num_tokens;
    std::int64_t num_kv_heads;
    std::int64_t head_size;
};

inline Activations check_activations(const at::Tensor& key, const at::Tensor& value, const char* op) {
    TORCH_CHECK(key.dim() == 3 && value.dim() == 3, op,
                ": key and value must be [num_tokens, num_kv_heads, head_size]");
    TORCH_CHECK(key.sizes() == value.sizes(), op, ": key and value must have the same shape");
    TORCH_CHECK(key.scalar_type() == at::kBFloat16 && value.scalar_type() == at::kBFloat16, op,
                ": both writers are the BF16 ingestion path; key and value must be bfloat16, got ",
                key.scalar_type(), " and ", value.scalar_type());
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous(), op, ": key and value must be contiguous");
    const Activations activations{key.size(0), key.size(1), key.size(2)};
    check_launch_u32(activations.num_tokens, op, "num_tokens");
    check_launch_u32(activations.num_kv_heads, op, "num_kv_heads");
    check_launch_u32(activations.head_size, op, "head_size");
    return activations;
}

struct CachePlanes {
    std::int64_t num_blocks;
    std::int64_t block_size;
};

inline CachePlanes check_cache(const at::Tensor& key_cache, const at::Tensor& value_cache,
                               const at::Tensor& scale_cache, const Activations& a, const char* op) {
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
    check_launch_u32(num_blocks, op, "num_blocks");
    check_launch_u32(block_size, op, "block_size");
    TORCH_CHECK(num_blocks > 0 && block_size > 0, op, ": cache dimensions must be nonzero");
    TORCH_CHECK(key_cache.size(2) == a.num_kv_heads && key_cache.size(3) == a.head_size / 2, op,
                ": packed plane is [", num_blocks, ", ", block_size, ", ", key_cache.size(2), ", ",
                key_cache.size(3), "], expected [..., ", a.num_kv_heads, ", ", a.head_size / 2, "]");

    TORCH_CHECK(a.num_kv_heads > 0 && a.head_size == kDecodeHeadSize, op,
                ": expected at least one KV head and head_size ", kDecodeHeadSize);
    TORCH_CHECK((2 * a.num_kv_heads) % 8 == 0, op, ": the slot-major scale plane needs ",
                "2 * num_kv_heads to be a multiple of 8, got ", a.num_kv_heads,
                " KV heads, whose NzScaleSlotFloats record is ",
                layout::NzScaleSlotFloats(static_cast<std::uint32_t>(a.num_kv_heads)),
                " floats while the physical plane holds only 2 * num_kv_heads");
    const std::int64_t scale_floats =
        static_cast<std::int64_t>(layout::NzScaleSlotFloats(static_cast<std::uint32_t>(a.num_kv_heads))) *
        num_blocks * block_size;
    TORCH_CHECK(scale_cache.numel() >= scale_floats, op, ": scale plane holds ", scale_cache.numel(),
                " floats, needs at least ", scale_floats);
    return CachePlanes{num_blocks, block_size};
}

inline void check_slot_mapping(const at::Tensor& slot_mapping, const std::int64_t expected, const char* op) {
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, op, ": slot_mapping must be int32, got ",
                slot_mapping.scalar_type());
    TORCH_CHECK(slot_mapping.is_contiguous() && slot_mapping.dim() == 1 && slot_mapping.size(0) == expected,
                op, ": slot_mapping must be a contiguous int32 [", expected, "], got ", slot_mapping.sizes());
}

inline std::int64_t aiv_core_num(const char* op) {
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

inline const float* optional_pi_signs(const at::Tensor& pi_signs) {
    return pi_signs.numel() > 0 ? static_cast<const float*>(pi_signs.data_ptr()) : nullptr;
}

inline bool validate_slots() {
    static const bool enabled = [] {
        const char* value = std::getenv("VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS");
        return value != nullptr && value[0] == '1' && value[1] == 0;
    }();
    return enabled;
}

inline void check_strip_slots(const at::Tensor& slot_mapping, const std::int64_t num_tokens, const char* op) {
    if (!validate_slots()) {
        return;
    }
    const at::Tensor host_slots = slot_mapping.to(at::kCPU).contiguous();
    const auto* slots = static_cast<const std::int32_t*>(host_slots.data_ptr());
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
}

void npu_turboquant_reshape_and_cache_a8(at::Tensor& key, at::Tensor& value, at::Tensor& key_cache,
                                         at::Tensor& value_cache, at::Tensor& scale_cache,
                                         at::Tensor& slot_mapping, at::Tensor& pi_signs, at::Tensor& tables,
                                         double inv_sqrt_len, int64_t quant_mode) {
    constexpr const char* op = "npu_turboquant_reshape_and_cache_a8";

    check_ingress_quant_mode(quant_mode, op);
    const Activations a = check_activations(key, value, op);
    const CachePlanes c = check_cache(key_cache, value_cache, scale_cache, a, op);
    check_slot_mapping(slot_mapping, a.num_tokens, op);
    check_tables(tables, a.head_size, op, "tables");
    TORCH_CHECK(a.num_tokens % kTokensPerStrip == 0, op, ": num_tokens (", a.num_tokens,
                ") must be a whole number of ", kTokensPerStrip,
                "-token strips; the launcher drops a partial trailing strip rather than writing it");
    TORCH_CHECK(a.num_kv_heads == 4, op, ": the batch writer is specialised to 4 KV heads, got ",
                a.num_kv_heads);
    TORCH_CHECK(a.head_size == kDecodeHeadSize, op, ": the batch writer is specialised to head_size ",
                kDecodeHeadSize, ", got ", a.head_size);
    TORCH_CHECK(c.block_size % layout::kCubeTileRows == 0, op, ": the NZ-tiled ingress writes ",
                layout::kCubeTileRows, "-row tiles, so block_size must be a multiple of ",
                layout::kCubeTileRows, ", got ", c.block_size,
                " (the fused decode additionally needs a power-of-two multiple)");

    if (a.num_tokens == 0) {
        return;
    }

    check_burst_aligned(key, op, "key");
    check_burst_aligned(value, op, "value");
    check_burst_aligned(key_cache, op, "key_cache");
    check_burst_aligned(value_cache, op, "value_cache");
    check_burst_aligned(scale_cache, op, "scale_cache");
    check_burst_aligned(tables, op, "tables");
    check_strip_slots(slot_mapping, a.num_tokens, op);

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    const auto tokens_per_core =
        a.num_tokens % kChampionTokensPerCore == 0 ? kChampionTokensPerCore : kTokensPerStrip;
    launch_turboquant_reshape_and_cache_bf16(
        stream, static_cast<const std::uint16_t*>(key.data_ptr()),
        static_cast<const std::uint16_t*>(value.data_ptr()), static_cast<std::int8_t*>(key_cache.data_ptr()),
        static_cast<std::int8_t*>(value_cache.data_ptr()), static_cast<float*>(scale_cache.data_ptr()),
        static_cast<const std::int32_t*>(slot_mapping.data_ptr()), optional_pi_signs(pi_signs),
        static_cast<const std::int32_t*>(tables.data_ptr()), static_cast<std::uint32_t>(a.num_tokens),
        static_cast<std::uint32_t>(a.num_kv_heads), static_cast<std::uint32_t>(a.head_size),
        static_cast<std::uint32_t>(c.block_size), static_cast<std::uint32_t>(c.num_blocks),
        static_cast<std::uint32_t>(tokens_per_core), static_cast<float>(inv_sqrt_len),
        static_cast<std::uint32_t>(quant_mode));
}

void npu_turboquant_reshape_scatter_aiv(at::Tensor& key, at::Tensor& value, at::Tensor& key_cache,
                                        at::Tensor& value_cache, at::Tensor& scale_cache,
                                        at::Tensor& slot_mapping, at::Tensor& pi_signs, double inv_sqrt_len,
                                        int64_t quant_mode) {
    constexpr const char* op = "npu_turboquant_reshape_scatter_aiv";

    check_ingress_quant_mode(quant_mode, op);
    const Activations a = check_activations(key, value, op);
    const CachePlanes c = check_cache(key_cache, value_cache, scale_cache, a, op);
    check_slot_mapping(slot_mapping, a.num_tokens, op);
    TORCH_CHECK(a.head_size == kDecodeHeadSize, op, ": the scatter writer is specialised to head_size ",
                kDecodeHeadSize, ", got ", a.head_size);
    TORCH_CHECK(c.block_size % layout::kCubeTileRows == 0, op, ": the NZ-tiled ingress writes ",
                layout::kCubeTileRows, "-row tiles, so block_size must be a multiple of ",
                layout::kCubeTileRows, ", got ", c.block_size,
                " (the fused decode additionally needs a power-of-two multiple)");
    TORCH_CHECK(scale_cache.numel() >= static_cast<std::int64_t>(layout::NzScaleSlotFloats(a.num_kv_heads)) *
                                           c.num_blocks * c.block_size,
                op, ": the NZ scale plane needs ", layout::NzScaleSlotFloats(a.num_kv_heads),
                " floats per slot (NzScaleSlotFloats(", a.num_kv_heads, ")), got ", scale_cache.numel());

    check_burst_aligned(key, op, "key");
    check_burst_aligned(value, op, "value");
    check_burst_aligned(key_cache, op, "key_cache");
    check_burst_aligned(value_cache, op, "value_cache");
    check_burst_aligned(scale_cache, op, "scale_cache");

    const std::int64_t num_requests = a.num_tokens;
    const std::uint64_t total_items = static_cast<std::uint64_t>(num_requests) * a.num_kv_heads;
    TORCH_CHECK(total_items > 0 && total_items <= UINT32_MAX, op,
                ": scatter item count must be in [1, UINT32_MAX], got ", total_items);

    const std::int64_t block_dim = std::min(static_cast<std::int64_t>(total_items), aiv_core_num(op));

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    launch_turboquant_reshape_scatter_aiv(
        stream, static_cast<std::uint32_t>(block_dim), static_cast<std::uint16_t*>(key.data_ptr()),
        static_cast<std::uint16_t*>(value.data_ptr()), static_cast<std::int8_t*>(key_cache.data_ptr()),
        static_cast<std::int8_t*>(value_cache.data_ptr()), static_cast<float*>(scale_cache.data_ptr()),
        static_cast<std::int32_t*>(slot_mapping.data_ptr()), const_cast<float*>(optional_pi_signs(pi_signs)),
        nullptr, static_cast<std::uint32_t>(num_requests), static_cast<std::uint32_t>(a.num_kv_heads),
        static_cast<std::uint32_t>(a.head_size), static_cast<std::uint32_t>(c.block_size),
        static_cast<std::uint32_t>(c.num_blocks), 0u, static_cast<std::uint32_t>(total_items),
        static_cast<float>(inv_sqrt_len), static_cast<std::uint32_t>(quant_mode));
}

namespace meta {
void npu_turboquant_reshape_and_cache_a8_meta(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                                              at::Tensor&, at::Tensor&, at::Tensor&, double, int64_t quant_mode) {
    check_ingress_quant_mode(quant_mode, "npu_turboquant_reshape_and_cache_a8");
}

void npu_turboquant_reshape_scatter_aiv_meta(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                                             at::Tensor&, at::Tensor&, double, int64_t quant_mode) {
    check_ingress_quant_mode(quant_mode, "npu_turboquant_reshape_scatter_aiv");
}
}
}
