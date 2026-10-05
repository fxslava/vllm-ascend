/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// CPU oracle for the DeepSeek-V4 routed expert kernel
// (csrc/moe/dsv4_moe_expert). Every bit-level helper here mirrors one in
// op_kernel/dsv4_moe_expert.cpp and one in
// tools/dsv4_moe_runtime/kernel_bringup/gen_golden.py; the three must agree
// exactly, and the host tier is where that is asserted.
//
// Contract:
//   * packed FP4: byte i holds logical element 2i in the LOW nibble and
//     element 2i+1 in the HIGH nibble;
//   * E8M0 scale byte b decodes to 2^(b-127); b=0 is the fp32 subnormal
//     2^-127 (bit pattern 0x00400000), b=0xFF is NaN, per OCP MX;
//   * accumulation is fp32, one multiply-add per element, reduction dimension
//     walked in ascending order;
//   * outputs are bf16, round-to-nearest-even, specials truncated.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

namespace vllm_ascend {
namespace test {
namespace dsv4 {

using DeviceOutputs = std::array<std::vector<uint16_t>, 4>;

// Explicit TBuf payload of the shipping 8x512 streaming kernel. TPipe stack
// overhead is additional; no allocation or offset here assumes 256 KiB UB.
constexpr size_t kExpertStreamingScratchBytes = 93696;
constexpr size_t ExpertUbBytes(int64_t hidden, int64_t inter) {
  return kExpertStreamingScratchBytes + static_cast<size_t>(2 * hidden + 4 * std::max(hidden, inter) + 18 * inter);
}

constexpr int64_t kFp4Block = 32;   // E8M0 scale span, in logical elements
constexpr int64_t kFp4PerByte = 2;  // two E2M1 nibbles per storage byte

// E2M1: bit3 sign, bits2..1 exponent (bias 1), bit0 mantissa.
inline const float* E2m1Table() {
  static const float kTable[16] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
  };
  return kTable;
}

// E8M0 exponent byte -> exact fp32 power of two. 0xFF is NaN; 0 is 2^-127,
// which the exponent field alone cannot express.
inline float E8m0ToScale(uint32_t exponent_byte) {
  uint32_t bits;
  if (exponent_byte == 0xFFu) {
    bits = 0x7FC00000u;  // quiet NaN, per OCP MX
  } else if (exponent_byte == 0u) {
    bits = 0x00400000u;  // 2^-127, subnormal
  } else {
    bits = exponent_byte << 23;
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// bf16 bits -> fp32. Exact widening.
inline float Bf16BitsToFloat(uint16_t bits) {
  const uint32_t widened = static_cast<uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &widened, sizeof(out));
  return out;
}

// fp32 -> bf16 bits, round-to-nearest-even. inf/NaN truncate without rounding.
inline uint16_t FloatToBf16Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7F800000u) == 0x7F800000u) {
    return static_cast<uint16_t>(bits >> 16);
  }
  const uint32_t rounding = 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

// Distance between two bf16 values in ULPs, via the monotone
// sign-magnitude -> ordinal map.
inline int64_t Bf16UlpDistance(uint16_t a, uint16_t b) {
  auto ordinal = [](uint16_t bits) -> int64_t {
    const int64_t v = static_cast<int64_t>(bits);
    return (v & 0x8000) ? (0x8000 - (v & 0x7FFF)) : (v + 0x8000);
  };
  const int64_t d = ordinal(a) - ordinal(b);
  return d < 0 ? -d : d;
}

// One nibble of a packed row. `col` is the logical element index.
inline float UnpackFp4(const uint8_t* packed_row, int64_t col) {
  const uint8_t byte = packed_row[col >> 1];
  const uint32_t nibble = (col & 1) ? (byte >> 4) : (byte & 0x0Fu);
  return E2m1Table()[nibble];
}

// out[r] = sum_c x[c] * (E2M1(w[r,c]) * E8M0(scale[r, c/32])).
//
// Ascending column order and a single fp32 accumulator. The device uses
// block reductions and column tiles; its reassociation must pass the ULP gate.
inline void Project(const float* x, const uint8_t* packed, const uint8_t* scales, int64_t rows, int64_t cols,
                    float* out) {
  const int64_t packed_per_row = cols / kFp4PerByte;
  const int64_t scales_per_row = cols / kFp4Block;
  for (int64_t row = 0; row < rows; ++row) {
    const uint8_t* packed_row = packed + row * packed_per_row;
    const uint8_t* scale_row = scales + row * scales_per_row;
    float acc = 0.0f;
    for (int64_t col = 0; col < cols; ++col) {
      const float scale = E8m0ToScale(scale_row[col / kFp4Block]);
      acc += x[col] * (UnpackFp4(packed_row, col) * scale);
    }
    out[row] = acc;
  }
}

// DeepSeek-V4 architectural constant: the gate is clamped symmetrically to
// +/- 10.0 before the activation. Part of the model definition, so the kernel,
// this reference and the goldens must all apply it. `clamp <= 0` disables it,
// which is only useful for characterising what the clamp changes.
constexpr float kSwigluLimit = 10.0f;

// Canonical operands: src0=up, src1=gate; src0 * swish(src1).
// silu(clamp(g)) * u, evaluated as (g * sigmoid(g)) * u to match the kernel's
// expression.
inline float SwiGluElement(float gate, float up, float clamp = kSwigluLimit) {
  float g = gate;
  if (clamp > 0.0f) {
    g = std::clamp(g, -clamp, clamp);
  }
  const float neg_exp = std::exp(-g);
  const float sigmoid = 1.0f / (1.0f + neg_exp);
  return (g * sigmoid) * up;
}

struct ExpertOutputs {
  std::vector<uint16_t> gate_out;   // [inter] bf16 bits
  std::vector<uint16_t> up_out;     // [inter] bf16 bits
  std::vector<uint16_t> activated;  // [inter] bf16 bits
  std::vector<uint16_t> down_out;   // [hidden] bf16 bits

  // The fp32 intermediates, before the final rounding. Exposed because the
  // bf16 outputs cannot be used to reconstruct one another: SwiGLU is
  // exponentially sensitive to its gate input in the negative tail, so
  // recomputing `activated` from the rounded `gate_out`/`up_out` diverges by
  // tens of ULPs. A cross-check of the three outputs has to be done here.
  std::vector<float> gate_f32;
  std::vector<float> up_f32;
  std::vector<float> activated_f32;
};

struct ExpertInputs {
  const uint16_t* x;        // [hidden] bf16 bits
  const uint8_t* w1;        // [inter, hidden/2]
  const uint8_t* w2;        // [hidden, inter/2]
  const uint8_t* w3;        // [inter, hidden/2]
  const uint8_t* w1_scale;  // [inter, hidden/32]
  const uint8_t* w2_scale;  // [hidden, inter/32]
  const uint8_t* w3_scale;  // [inter, hidden/32]
};

// The full expert pipeline: gate/up FP4 GEMM, SwiGLU, down FP4 GEMM.
inline ExpertOutputs ReferenceExpert(const ExpertInputs& in, int64_t hidden, int64_t inter,
                                     float clamp = kSwigluLimit) {
  std::vector<float> x_f32(static_cast<size_t>(hidden));
  for (int64_t c = 0; c < hidden; ++c) {
    x_f32[static_cast<size_t>(c)] = Bf16BitsToFloat(in.x[c]);
  }

  std::vector<float> gate(static_cast<size_t>(inter));
  std::vector<float> up(static_cast<size_t>(inter));
  Project(x_f32.data(), in.w1, in.w1_scale, inter, hidden, gate.data());
  Project(x_f32.data(), in.w3, in.w3_scale, inter, hidden, up.data());

  std::vector<float> activated(static_cast<size_t>(inter));
  for (int64_t j = 0; j < inter; ++j) {
    activated[static_cast<size_t>(j)] = SwiGluElement(gate[static_cast<size_t>(j)], up[static_cast<size_t>(j)], clamp);
  }

  std::vector<float> down(static_cast<size_t>(hidden));
  std::vector<float> down_input(activated.size());
  for (size_t j = 0; j < activated.size(); ++j) {
    down_input[j] = Bf16BitsToFloat(FloatToBf16Bits(activated[j]));
  }
  Project(down_input.data(), in.w2, in.w2_scale, hidden, inter, down.data());

  ExpertOutputs out;
  out.gate_out.resize(static_cast<size_t>(inter));
  out.up_out.resize(static_cast<size_t>(inter));
  out.activated.resize(static_cast<size_t>(inter));
  out.down_out.resize(static_cast<size_t>(hidden));
  for (int64_t j = 0; j < inter; ++j) {
    out.gate_out[static_cast<size_t>(j)] = FloatToBf16Bits(gate[static_cast<size_t>(j)]);
    out.up_out[static_cast<size_t>(j)] = FloatToBf16Bits(up[static_cast<size_t>(j)]);
    out.activated[static_cast<size_t>(j)] = FloatToBf16Bits(activated[static_cast<size_t>(j)]);
  }
  for (int64_t r = 0; r < hidden; ++r) {
    out.down_out[static_cast<size_t>(r)] = FloatToBf16Bits(down[static_cast<size_t>(r)]);
  }
  out.gate_f32 = std::move(gate);
  out.up_f32 = std::move(up);
  out.activated_f32 = std::move(activated);
  return out;
}

// The geometry rules the host tiling function enforces
// (op_host/dsv4_moe_expert_tiling.cpp). Kept here so the host tier can assert
// them without a CANN toolkit.
constexpr int64_t kMaxHiddenSize = 7168;
constexpr int64_t kMaxInterSize = 2048;

inline bool GeometryIsAccepted(int64_t hidden, int64_t inter) {
  if (hidden % (kFp4PerByte * kFp4Block) != 0) return false;
  if (inter % (kFp4PerByte * kFp4Block) != 0) return false;
  if (hidden <= 0 || inter <= 0) return false;
  return hidden <= kMaxHiddenSize && inter <= kMaxInterSize;
}

}  // namespace dsv4
}  // namespace test
}  // namespace vllm_ascend
