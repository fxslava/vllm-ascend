/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

// Shapes the Qwen3.5 forward pass puts through the four kernels under test,
// plus the Ascend 310P alignment rules that constrain them.

#pragma once

#include <cstdint>
#include <vector>

namespace vllm_ascend {
namespace test {
namespace shapes {

// --- Ascend 310P (DaVinci v200) alignment rules -----------------------------

// MTE2/MTE3 move 32 bytes per burst, so an fp16 row wants a multiple of 16
// elements. Every hidden and intermediate size below satisfies this.
constexpr int64_t kFp16ElementsPerBurst = 16;

// AscendSiluAndMul310.forward calls npu_swiglu only when x.shape[-1] % 32 == 0
// and otherwise falls back to eager torch. See vllm_ascend/_310p/ops/activation.py.
constexpr int64_t kSwiGluLastDimMultiple = 32;

// The trailing axis of the 310P 5-D KV cache. See
// AscendAttentionBackend310.get_kv_cache_shape in
// vllm_ascend/_310p/attention/attention_v1.py.
constexpr int64_t kKvCacheFractalWidth = 16;

// AscendAttentionBackend310.get_supported_kernel_block_sizes() returns
// [128, 64]. These are the only paging granularities the 310P attention kernel
// accepts, which is why the tests do not use the 16 and 32 common on GPU.
inline std::vector<int64_t> Supported310PBlockSizes() { return {64, 128}; }

// The 310P paged attention kernel additionally requires
// block_size * head_size <= 128 * 128. See _ATTENTION_BLOCK_SIZE_LIMIT in
// vllm_ascend/_310p/model_runner_310p.py, which uses it to pick the largest
// usable block size for a given head size.
constexpr int64_t kAttentionBlockSizeLimit = 128 * 128;

inline bool IsValid310PBlockSize(int64_t block_size, int64_t head_size) {
  return block_size * head_size <= kAttentionBlockSizeLimit;
}

// --- RMSNorm ----------------------------------------------------------------

// Hidden sizes spanning the Qwen3.5 range, from the small dense models up to
// the wide MoE variants. Applied before every attention and MLP block.
inline std::vector<int64_t> RmsNormHiddenSizes() { return {1536, 2048, 4096, 8192}; }

// Token counts covering a single decode step, a ragged prefill chunk, and a
// full aligned batch.
inline std::vector<int64_t> TokenCounts() { return {1, 7, 32, 128}; }

// Qwen uses 1e-6 for rms_norm_eps; the value matters because it is added under
// the square root and dominates when a row is near zero.
constexpr float kRmsNormEpsilon = 1e-6f;

// --- SiluAndMul / SwiGLU ----------------------------------------------------

// MLP intermediate sizes. The gate_up projection produces 2 * intermediate,
// which is what the kernel splits.
inline std::vector<int64_t> IntermediateSizes() { return {4096, 8960, 11008, 14336}; }

// --- Rotary embedding -------------------------------------------------------

// npu_apply_rotary_pos_emb accepts head dims 64 and 128 only; see the
// `self.rotary_dim in (64, 128)` gate in
// vllm_ascend/_310p/ops/rotary_embedding.py.
inline std::vector<int64_t> RotaryHeadDims() { return {64, 128}; }

// Qwen3 rope_theta. Larger contexts use 1e6; both are exercised because the
// cos/sin cache construction is where a theta mismatch shows up.
constexpr double kRopeThetaDefault = 10000.0;
constexpr double kRopeThetaExtended = 1000000.0;

constexpr int64_t kMaxPositionEmbeddings = 4096;

// --- Attention head configurations ------------------------------------------

struct AttentionHeads {
  const char* label;
  int64_t num_heads;
  int64_t num_kv_heads;
  int64_t head_size;
};

// Grouped-query configurations across the Qwen3.5 family. num_heads is always a
// multiple of num_kv_heads, and num_kv_heads * head_size is always a multiple
// of 16 so the 310P fractal KV cache shape is exact.
inline std::vector<AttentionHeads> GqaConfigurations() {
  return {
      AttentionHeads{"mha_8h_8kv_d128", 8, 8, 128},   // no grouping, control case
      AttentionHeads{"gqa_28h_4kv_d128", 28, 4, 128},
      AttentionHeads{"gqa_32h_8kv_d128", 32, 8, 128},
      AttentionHeads{"gqa_16h_2kv_d64", 16, 2, 64},
  };
}

// Context lengths chosen so that at least one spans several blocks at both
// supported block sizes, and one lands exactly on a block boundary.
inline std::vector<int64_t> ContextLengths() { return {1, 63, 64, 200, 512}; }

}  // namespace shapes
}  // namespace test
}  // namespace vllm_ascend
