#pragma once

#include <cstdint>
#include <optional>

#include <ATen/ATen.h>

#include "../common/turboquant_layout.h"

namespace vllm_ascend::turboquant_bindings {
namespace layout = ::vllm_ascend::turboquant;

constexpr std::int64_t kDecodeHeadSize = 256;
constexpr std::uintptr_t kGmBurstBytes = 32;

constexpr std::int64_t kTableGroup = 32;
constexpr std::int64_t kTablePlanes = 2;

inline void check_burst_aligned(const at::Tensor& t, const char* op, const char* name) {
    const auto address = reinterpret_cast<std::uintptr_t>(t.data_ptr());
    TORCH_CHECK(address % kGmBurstBytes == 0, op, ": ", name, " base address ", address, " is not ",
                kGmBurstBytes, "-byte aligned");
}

inline void check_tables(const at::Tensor& tables, const std::int64_t head_size, const char* op,
                         const char* name) {
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

void npu_turboquant_reshape_and_cache_a8(at::Tensor& key, at::Tensor& value, at::Tensor& key_cache,
                                         at::Tensor& value_cache, at::Tensor& scale_cache,
                                         at::Tensor& slot_mapping, at::Tensor& pi_signs, at::Tensor& tables,
                                         double inv_sqrt_len, int64_t quant_mode);
void npu_turboquant_reshape_scatter_aiv(at::Tensor& key, at::Tensor& value, at::Tensor& key_cache,
                                        at::Tensor& value_cache, at::Tensor& scale_cache,
                                        at::Tensor& slot_mapping, at::Tensor& pi_signs, double inv_sqrt_len,
                                        int64_t quant_mode);
void npu_turboquant_fused_decode(at::Tensor& query, at::Tensor& pi_signs, at::Tensor& rot_tables,
                                 at::Tensor& h16, const std::optional<at::Tensor>& gate,
                                 at::Tensor& key_cache, at::Tensor& value_cache, at::Tensor& scale_cache,
                                 at::Tensor& block_tables, at::Tensor& context_lens, at::Tensor& tiling,
                                 at::Tensor& workspace, at::Tensor& query_rot, at::Tensor& output,
                                 int64_t num_heads, int64_t num_kv_heads, int64_t num_splits,
                                 int64_t heads_per_task, int64_t output_stage, double scale,
                                 double inv_sqrt_len, int64_t decode_unpack, int64_t fused_context_limit);
void npu_turboquant_build_decode_tiling(at::Tensor& context_lens, at::Tensor& tiling, int64_t num_heads,
                                        int64_t num_kv_heads, int64_t head_size, int64_t block_size,
                                        int64_t max_blocks, int64_t num_splits, int64_t heads_per_task,
                                        int64_t fused_context_limit);

namespace meta {
void npu_turboquant_reshape_and_cache_a8_meta(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                                              at::Tensor&, at::Tensor&, at::Tensor&, double, int64_t);
void npu_turboquant_reshape_scatter_aiv_meta(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                                             at::Tensor&, at::Tensor&, double, int64_t);
void npu_turboquant_fused_decode_meta(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                                      const std::optional<at::Tensor>&, at::Tensor&, at::Tensor&, at::Tensor&,
                                      at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                                      at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, double,
                                      double, int64_t, int64_t);
void npu_turboquant_build_decode_tiling_meta(at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t,
                                             int64_t, int64_t, int64_t, int64_t);
}
}

namespace vllm_ascend::turboquant_e8 {
void reader(at::Tensor& query, at::Tensor& key_packets, at::Tensor& value_packets, at::Tensor& packet_offsets,
            at::Tensor& output, at::Tensor& key_witness, at::Tensor& value_witness, at::Tensor& score_witness,
            std::int64_t tiles, std::int64_t last_rows, double scale);
void reader_meta(at::Tensor& query, at::Tensor& key_packets, at::Tensor& value_packets,
                 at::Tensor& packet_offsets, at::Tensor& output, at::Tensor& key_witness,
                 at::Tensor& value_witness, at::Tensor& score_witness, std::int64_t tiles,
                 std::int64_t last_rows, double scale);
void dequant(at::Tensor& labels, at::Tensor& scales, at::Tensor& bf16, at::Tensor& int8);
void dequant_meta(at::Tensor& labels, at::Tensor& scales, at::Tensor& bf16, at::Tensor& int8);
}
