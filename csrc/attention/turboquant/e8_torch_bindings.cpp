// SPDX-License-Identifier: Apache-2.0
// Prepared E8 cache-basis operations. No encoder, raw-query rotation or
// inverse-output transform is implied by these bounded verified entries.
#include <cmath>
#include <cstdint>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "e8/e8_decode.hpp"
#include "e8/e8_packet_layout.hpp"
#include "e8/e8_component.hpp"

namespace vllm_ascend::turboquant_e8 {

template <std::size_t N>
void disjoint_outputs(const at::Tensor* const (&tensors)[N], std::size_t first_output) {
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

void tensor(const at::Tensor& t, at::ScalarType type, std::int64_t elements,
            const at::Tensor& anchor, const char* name) {
    TORCH_CHECK(t.scalar_type() == type && t.is_contiguous() && t.numel() == elements,
                name, ": unexpected dtype, size or strides");
    TORCH_CHECK(t.device() == anchor.device(), name, ": device differs from query/labels");
    if (!t.is_meta()) {
        TORCH_CHECK(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 32 == 0,
                    name, ": expected a 32-byte aligned buffer");
    }
}

void validate_reader(const at::Tensor& query, const at::Tensor& key_packets,
    const at::Tensor& value_packets, const at::Tensor& packet_offsets,
    const at::Tensor& output, const at::Tensor& key_witness,
    const at::Tensor& value_witness, const at::Tensor& score_witness,
    std::int64_t tiles, std::int64_t last_rows, double scale) {
    TORCH_CHECK(tiles >= 4 && tiles <= e8_decode::kMaxTiles && tiles % 2 == 0,
                "E8 reader requires an even tile count in [4, 256]");
    TORCH_CHECK(last_rows > 0 && last_rows <= e8_decode::kRows &&
                std::isfinite(static_cast<float>(scale)),
                "invalid E8 live rows or attention scale");
    TORCH_CHECK(query.dim() == 2 && query.size(0) == 16 && query.size(1) == 256,
                "E8 query must be cache-basis FP32 [16, 256], rows 6..15 zero");
    tensor(query, at::kFloat, 16 * 256, query, "query");
    TORCH_CHECK(key_packets.dim() == 2 && value_packets.sizes() == key_packets.sizes() &&
                key_packets.size(0) > 0 && key_packets.size(1) == e8_packet_abi::kPacketBytes,
                "E8 planes must be UINT8 [packets, 4352], with embedded FP32 scales");
    tensor(key_packets, at::kByte, key_packets.numel(), query, "key_packets");
    tensor(value_packets, at::kByte, key_packets.numel(), query, "value_packets");
    // Entries are prevalidated nonnegative byte offsets, uploaded once by the
    // caller. Do not read GM metadata back or allocate a temporary on this path.
    TORCH_CHECK(packet_offsets.dim() == 1, "E8 offsets must be a flat INT64 byte-offset table");
    // MTE2 rounds this read up to a 32-byte burst (four UINT64 offsets).
    tensor(packet_offsets, at::kLong, ((tiles + 3) / 4) * 4, query, "packet_offsets");
    tensor(output, at::kBFloat16, 6 * 256, query, "output");
    tensor(key_witness, at::kBFloat16, tiles * 64 * 256, query, "key_witness");
    tensor(value_witness, at::kBFloat16, tiles * 64 * 256, query, "value_witness");
    tensor(score_witness, at::kFloat, tiles * 6 * 64, query, "score_witness");
    const at::Tensor* const buffers[] = {&query, &key_packets, &value_packets, &packet_offsets,
        &output, &key_witness, &value_witness, &score_witness};
    disjoint_outputs(buffers, 4);
}

void reader(at::Tensor& query, at::Tensor& key_packets, at::Tensor& value_packets,
    at::Tensor& packet_offsets, at::Tensor& output, at::Tensor& key_witness,
    at::Tensor& value_witness, at::Tensor& score_witness,
    std::int64_t tiles, std::int64_t last_rows, double scale) {
    validate_reader(query, key_packets, value_packets, packet_offsets, output,
                    key_witness, value_witness, score_witness, tiles, last_rows, scale);
    const e8_decode::Launch launch{static_cast<std::uint32_t>(tiles),
        static_cast<std::uint32_t>(last_rows), static_cast<float>(scale)};
    const e8_decode::Buffers buffers{key_packets.data_ptr(), value_packets.data_ptr(),
        packet_offsets.data_ptr(), query.data_ptr(), output.data_ptr(),
        key_witness.data_ptr(), value_witness.data_ptr(), score_witness.data_ptr()};
    launch_e8_fused_decode_fp32(c10_npu::getCurrentNPUStream().stream(), &launch, &buffers);
}

void reader_meta(at::Tensor& query, at::Tensor& key_packets, at::Tensor& value_packets,
    at::Tensor& packet_offsets, at::Tensor& output, at::Tensor& key_witness,
    at::Tensor& value_witness, at::Tensor& score_witness,
    std::int64_t tiles, std::int64_t last_rows, double scale) {
    validate_reader(query, key_packets, value_packets, packet_offsets, output,
                    key_witness, value_witness, score_witness, tiles, last_rows, scale);
}

void validate_dequant(const at::Tensor& labels, const at::Tensor& scales,
    const at::Tensor& bf16, const at::Tensor& int8) {
    tensor(labels, at::kUInt16, 512, labels, "labels");
    tensor(scales, at::kFloat, 16, labels, "scales");
    tensor(bf16, at::kBFloat16, 4096, labels, "bf16");
    tensor(int8, at::kChar, 4096, labels, "int8");
    const at::Tensor* const buffers[] = {&labels, &scales, &bf16, &int8};
    disjoint_outputs(buffers, 2);
}

void dequant(at::Tensor& labels, at::Tensor& scales, at::Tensor& bf16, at::Tensor& int8) {
    validate_dequant(labels, scales, bf16, int8);
    launch_e8_parallel_component(c10_npu::getCurrentNPUStream().stream(),
        static_cast<std::uint16_t*>(labels.data_ptr()), static_cast<float*>(scales.data_ptr()),
        static_cast<std::uint16_t*>(bf16.data_ptr()), static_cast<std::int8_t*>(int8.data_ptr()));
}

void dequant_meta(at::Tensor& labels, at::Tensor& scales, at::Tensor& bf16, at::Tensor& int8) {
    validate_dequant(labels, scales, bf16, int8);
}
}  // namespace vllm_ascend::turboquant_e8

TORCH_LIBRARY_FRAGMENT(turboquant_e8_ops, ops) {
    ops.def("reader(Tensor query, Tensor key_packets, Tensor value_packets, Tensor packet_offsets, "
            "Tensor! output, Tensor! key_witness, Tensor! value_witness, Tensor! score_witness, "
            "int tiles, int last_rows, float scale) -> ()");
    ops.impl("reader", c10::kPrivateUse1, &vllm_ascend::turboquant_e8::reader);
    ops.impl("reader", c10::kMeta, &vllm_ascend::turboquant_e8::reader_meta);
    ops.def("dequant(Tensor labels, Tensor scales, Tensor! bf16, Tensor! int8) -> ()");
    ops.impl("dequant", c10::kPrivateUse1, &vllm_ascend::turboquant_e8::dequant);
    ops.impl("dequant", c10::kMeta, &vllm_ascend::turboquant_e8::dequant_meta);
}
