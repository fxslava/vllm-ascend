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

// Naive CPU references for the Qwen3.5 forward-pass kernels.
//
// Conventions used throughout:
//   * Inputs and outputs are std::vector<float>. Callers convert to and from
//     fp16 at the device boundary, so a reference never sees a half type.
//   * Arithmetic is float, not double. The DaVinci vector unit accumulates
//     these reductions in fp32, so matching that keeps the tolerance honest
//     rather than flattering the kernel with a more accurate reference.
//   * Nothing here allocates device memory or touches ACL: every function is
//     callable on a build machine with no NPU attached, and the suite has
//     self-tests that do exactly that.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vllm_ascend {
namespace test {
namespace reference {

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------
// x     [num_tokens, hidden]
// gamma [hidden]
// y     [num_tokens, hidden]  = x * rstd * gamma
// rstd  [num_tokens]          = 1 / sqrt(mean(x^2) + epsilon)
//
// Matches torch_npu.npu_rms_norm, which is what AscendRMSNorm310 calls.
void RmsNorm(const std::vector<float>& x, const std::vector<float>& gamma, int64_t num_tokens, int64_t hidden,
             float epsilon, std::vector<float>* y, std::vector<float>* rstd);

// ---------------------------------------------------------------------------
// SiluAndMul / SwiGLU
// ---------------------------------------------------------------------------
// x   [num_tokens, 2 * intermediate]
// out [num_tokens, intermediate] = silu(x[..., :intermediate]) * x[..., intermediate:]
//
// silu(v) = v * sigmoid(v). Matches torch_npu.npu_swiglu with dim = -1.
void SiluAndMul(const std::vector<float>& x, int64_t num_tokens, int64_t intermediate, std::vector<float>* out);

// ---------------------------------------------------------------------------
// Rotary position embedding
// ---------------------------------------------------------------------------
enum class RotaryMode {
  // neox / rotate_half. Pairs element i with i + rotary_dim/2.
  kHalf,
  // GPT-J / interleaved. Pairs element 2k with 2k + 1.
  kInterleave,
};

// Builds the vLLM RotaryEmbedding cos/sin cache:
//   inv_freq[i] = 1 / base^(2i / rotary_dim),  i in [0, rotary_dim/2)
//   cache[p]    = concat(cos(p * inv_freq), sin(p * inv_freq))
// Shape [max_position, rotary_dim]; the two halves are cos then sin.
std::vector<float> BuildCosSinCache(int64_t max_position, int64_t rotary_dim, double base);

// Expands the half-width cache entries for `positions` into the full rotary dim
// the operator expects. kHalf concatenates (cos, cos); kInterleave repeats each
// element twice, which is what vLLM ApplyRotaryEmb does for the GPT-J layout.
//
// cos_full / sin_full come back as [num_tokens, rotary_dim].
void GatherFullCosSin(const std::vector<float>& cos_sin_cache, const std::vector<int32_t>& positions,
                      int64_t rotary_dim, RotaryMode mode, std::vector<float>* cos_full,
                      std::vector<float>* sin_full);

// x   [num_tokens, num_heads, head_dim]
// out [num_tokens, num_heads, head_dim]
//
//   out[..., :rotary_dim] = x_rot * cos + rotate(x_rot) * sin
//   out[..., rotary_dim:] = x[..., rotary_dim:]      (pass-through)
//
// cos_full / sin_full are [num_tokens, rotary_dim] and shared across heads.
void ApplyRotaryPosEmb(const std::vector<float>& x, const std::vector<float>& cos_full,
                       const std::vector<float>& sin_full, int64_t num_tokens, int64_t num_heads, int64_t head_dim,
                       int64_t rotary_dim, RotaryMode mode, std::vector<float>* out);

// ---------------------------------------------------------------------------
// Paged KV cache, Ascend 310P layout
// ---------------------------------------------------------------------------
// AscendAttentionBackend310.get_kv_cache_shape returns
//     (2, num_blocks, num_kv_heads * head_size / 16, block_size, 16)
// so each of the key and value caches is a 4-D block of
//     [num_blocks, hidden / 16, block_size, 16]
// where hidden = num_kv_heads * head_size. The trailing 16 is the fp16
// fractal width the v200 cube unit consumes, which is why head_size * num_kv_heads
// must be a multiple of 16 on this part.
struct PagedKvLayout {
  int64_t num_blocks = 0;
  int64_t block_size = 0;
  int64_t num_kv_heads = 0;
  int64_t head_size = 0;

  int64_t hidden() const { return num_kv_heads * head_size; }
  int64_t fractal_rows() const { return hidden() / 16; }
  size_t ElementCount() const {
    return static_cast<size_t>(num_blocks) * static_cast<size_t>(fractal_rows()) *
           static_cast<size_t>(block_size) * 16u;
  }
};

// Flat element offset of (block, offset-in-block, kv head, head dim) in the
// 5-D NZ cache described above.
size_t NzCacheOffset(const PagedKvLayout& layout, int64_t block_id, int64_t block_offset, int64_t kv_head,
                     int64_t dim);

// Host-side model of torch_npu._npu_reshape_and_cache.
// key / value    [num_tokens, num_kv_heads, head_size]
// slot_mapping   [num_tokens], slot = block_id * block_size + offset_in_block
// key_cache / value_cache are updated in place.
void ReshapeAndCache(const std::vector<float>& key, const std::vector<float>& value,
                     const std::vector<int32_t>& slot_mapping, const PagedKvLayout& layout,
                     std::vector<float>* key_cache, std::vector<float>* value_cache);

// ---------------------------------------------------------------------------
// Paged attention (decode: exactly one query token per sequence)
// ---------------------------------------------------------------------------
struct PagedAttentionShape {
  int64_t num_seqs = 0;
  int64_t num_heads = 0;      // query heads
  int64_t num_kv_heads = 0;   // key/value heads; num_heads must be a multiple
  int64_t head_size = 0;
  int64_t block_size = 0;
  int64_t max_blocks_per_seq = 0;
  float scale = 1.0f;
};

// query        [num_seqs, num_heads, head_size]
// key_cache    NZ 5-D, see PagedKvLayout
// value_cache  NZ 5-D
// block_table  [num_seqs, max_blocks_per_seq], logical block -> physical block
// context_lens [num_seqs]
// out          [num_seqs, num_heads, head_size]
//
// GQA mapping is the vLLM one: query head h reads kv head h / (num_heads / num_kv_heads).
void PagedAttentionDecode(const std::vector<float>& query, const std::vector<float>& key_cache,
                          const std::vector<float>& value_cache, const std::vector<int32_t>& block_table,
                          const std::vector<int32_t>& context_lens, const PagedKvLayout& layout,
                          const PagedAttentionShape& shape, std::vector<float>* out);

}  // namespace reference
}  // namespace test
}  // namespace vllm_ascend
