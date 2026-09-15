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

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vllm_ascend {
namespace test {
namespace reference {

void MatmulTransposedB(const std::vector<float>& a, const std::vector<float>& b_t, int64_t m, int64_t k,
                       int64_t n, std::vector<float>* out);

void RmsNorm(const std::vector<float>& x, const std::vector<float>& gamma, int64_t num_tokens, int64_t hidden,
             float epsilon, std::vector<float>* y, std::vector<float>* rstd);

void SiluAndMul(const std::vector<float>& x, int64_t num_tokens, int64_t intermediate, std::vector<float>* out);

enum class RotaryMode {
  kHalf,
  kInterleave,
};

std::vector<float> BuildCosSinCache(int64_t max_position, int64_t rotary_dim, double base);

void GatherFullCosSin(const std::vector<float>& cos_sin_cache, const std::vector<int32_t>& positions,
                      int64_t rotary_dim, RotaryMode mode, std::vector<float>* cos_full,
                      std::vector<float>* sin_full);

void ApplyRotaryPosEmb(const std::vector<float>& x, const std::vector<float>& cos_full,
                       const std::vector<float>& sin_full, int64_t num_tokens, int64_t num_heads, int64_t head_dim,
                       int64_t rotary_dim, RotaryMode mode, std::vector<float>* out);

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

size_t NzCacheOffset(const PagedKvLayout& layout, int64_t block_id, int64_t block_offset, int64_t kv_head,
                     int64_t dim);

void ReshapeAndCache(const std::vector<float>& key, const std::vector<float>& value,
                     const std::vector<int32_t>& slot_mapping, const PagedKvLayout& layout,
                     std::vector<float>* key_cache, std::vector<float>* value_cache);

struct PagedAttentionShape {
  int64_t num_seqs = 0;
  int64_t num_heads = 0;
  int64_t num_kv_heads = 0;
  int64_t head_size = 0;
  int64_t block_size = 0;
  int64_t max_blocks_per_seq = 0;
  float scale = 1.0f;
};

void PagedAttentionDecode(const std::vector<float>& query, const std::vector<float>& key_cache,
                          const std::vector<float>& value_cache, const std::vector<int32_t>& block_table,
                          const std::vector<int32_t>& context_lens, const PagedKvLayout& layout,
                          const PagedAttentionShape& shape, std::vector<float>* out);

}
}
}
