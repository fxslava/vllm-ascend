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

#include "cpu_reference.hpp"

#include <cassert>
#include <cmath>
#include <limits>

namespace vllm_ascend {
namespace test {
namespace reference {

void MatmulTransposedB(const std::vector<float>& a, const std::vector<float>& b_t, int64_t m, int64_t k,
                       int64_t n, std::vector<float>* out) {
  assert(a.size() == static_cast<size_t>(m * k));
  assert(b_t.size() == static_cast<size_t>(n * k));

  out->assign(static_cast<size_t>(m * n), 0.0f);

  for (int64_t i = 0; i < m; ++i) {
    const size_t a_row = static_cast<size_t>(i * k);
    for (int64_t j = 0; j < n; ++j) {
      const size_t b_row = static_cast<size_t>(j * k);
      float sum = 0.0f;
      for (int64_t p = 0; p < k; ++p) {
        sum += a[a_row + static_cast<size_t>(p)] * b_t[b_row + static_cast<size_t>(p)];
      }
      (*out)[static_cast<size_t>(i * n + j)] = sum;
    }
  }
}

void RmsNorm(const std::vector<float>& x, const std::vector<float>& gamma, int64_t num_tokens, int64_t hidden,
             float epsilon, std::vector<float>* y, std::vector<float>* rstd) {
  assert(x.size() == static_cast<size_t>(num_tokens * hidden));
  assert(gamma.size() == static_cast<size_t>(hidden));

  y->assign(static_cast<size_t>(num_tokens * hidden), 0.0f);
  rstd->assign(static_cast<size_t>(num_tokens), 0.0f);

  for (int64_t token = 0; token < num_tokens; ++token) {
    const size_t row = static_cast<size_t>(token * hidden);

    float sum_of_squares = 0.0f;
    for (int64_t i = 0; i < hidden; ++i) {
      const float value = x[row + static_cast<size_t>(i)];
      sum_of_squares += value * value;
    }

    const float mean_square = sum_of_squares / static_cast<float>(hidden);
    const float inverse_rms = 1.0f / std::sqrt(mean_square + epsilon);
    (*rstd)[static_cast<size_t>(token)] = inverse_rms;

    for (int64_t i = 0; i < hidden; ++i) {
      (*y)[row + static_cast<size_t>(i)] =
          x[row + static_cast<size_t>(i)] * inverse_rms * gamma[static_cast<size_t>(i)];
    }
  }
}

void SiluAndMul(const std::vector<float>& x, int64_t num_tokens, int64_t intermediate, std::vector<float>* out) {
  assert(x.size() == static_cast<size_t>(num_tokens * intermediate * 2));

  out->assign(static_cast<size_t>(num_tokens * intermediate), 0.0f);

  for (int64_t token = 0; token < num_tokens; ++token) {
    const size_t in_row = static_cast<size_t>(token * intermediate * 2);
    const size_t out_row = static_cast<size_t>(token * intermediate);
    for (int64_t i = 0; i < intermediate; ++i) {
      const float gate = x[in_row + static_cast<size_t>(i)];
      const float up = x[in_row + static_cast<size_t>(intermediate + i)];
      const float silu = gate / (1.0f + std::exp(-gate));
      (*out)[out_row + static_cast<size_t>(i)] = silu * up;
    }
  }
}

std::vector<float> BuildCosSinCache(int64_t max_position, int64_t rotary_dim, double base) {
  const int64_t half = rotary_dim / 2;
  std::vector<float> cache(static_cast<size_t>(max_position * rotary_dim), 0.0f);

  for (int64_t position = 0; position < max_position; ++position) {
    const size_t row = static_cast<size_t>(position * rotary_dim);
    for (int64_t i = 0; i < half; ++i) {
      const double exponent = static_cast<double>(2 * i) / static_cast<double>(rotary_dim);
      const double inverse_frequency = 1.0 / std::pow(base, exponent);
      const double angle = static_cast<double>(position) * inverse_frequency;
      cache[row + static_cast<size_t>(i)] = static_cast<float>(std::cos(angle));
      cache[row + static_cast<size_t>(half + i)] = static_cast<float>(std::sin(angle));
    }
  }
  return cache;
}

void GatherFullCosSin(const std::vector<float>& cos_sin_cache, const std::vector<int32_t>& positions,
                      int64_t rotary_dim, RotaryMode mode, std::vector<float>* cos_full,
                      std::vector<float>* sin_full) {
  const int64_t half = rotary_dim / 2;
  const size_t num_tokens = positions.size();

  cos_full->assign(num_tokens * static_cast<size_t>(rotary_dim), 0.0f);
  sin_full->assign(num_tokens * static_cast<size_t>(rotary_dim), 0.0f);

  for (size_t token = 0; token < num_tokens; ++token) {
    const size_t cache_row = static_cast<size_t>(positions[token]) * static_cast<size_t>(rotary_dim);
    const size_t out_row = token * static_cast<size_t>(rotary_dim);

    for (int64_t i = 0; i < half; ++i) {
      const float cos_value = cos_sin_cache[cache_row + static_cast<size_t>(i)];
      const float sin_value = cos_sin_cache[cache_row + static_cast<size_t>(half + i)];

      if (mode == RotaryMode::kHalf) {
        // concat(cos, cos): index i and index i + half share one value.
        (*cos_full)[out_row + static_cast<size_t>(i)] = cos_value;
        (*cos_full)[out_row + static_cast<size_t>(half + i)] = cos_value;
        (*sin_full)[out_row + static_cast<size_t>(i)] = sin_value;
        (*sin_full)[out_row + static_cast<size_t>(half + i)] = sin_value;
      } else {
        // repeat_interleave(2): index 2i and index 2i + 1 share one value.
        (*cos_full)[out_row + static_cast<size_t>(2 * i)] = cos_value;
        (*cos_full)[out_row + static_cast<size_t>(2 * i + 1)] = cos_value;
        (*sin_full)[out_row + static_cast<size_t>(2 * i)] = sin_value;
        (*sin_full)[out_row + static_cast<size_t>(2 * i + 1)] = sin_value;
      }
    }
  }
}

void ApplyRotaryPosEmb(const std::vector<float>& x, const std::vector<float>& cos_full,
                       const std::vector<float>& sin_full, int64_t num_tokens, int64_t num_heads, int64_t head_dim,
                       int64_t rotary_dim, RotaryMode mode, std::vector<float>* out) {
  assert(x.size() == static_cast<size_t>(num_tokens * num_heads * head_dim));
  assert(rotary_dim <= head_dim);
  assert(rotary_dim % 2 == 0);

  out->assign(x.size(), 0.0f);

  const int64_t half = rotary_dim / 2;

  for (int64_t token = 0; token < num_tokens; ++token) {
    const size_t angle_row = static_cast<size_t>(token * rotary_dim);
    for (int64_t head = 0; head < num_heads; ++head) {
      const size_t base = static_cast<size_t>((token * num_heads + head) * head_dim);

      for (int64_t i = 0; i < rotary_dim; ++i) {
        // rotate(x)[i] is the partner element, negated on the first element of
        // each pair and kept positive on the second.
        int64_t partner = 0;
        float sign = 0.0f;
        if (mode == RotaryMode::kHalf) {
          partner = (i < half) ? (i + half) : (i - half);
          sign = (i < half) ? -1.0f : 1.0f;
        } else {
          partner = (i % 2 == 0) ? (i + 1) : (i - 1);
          sign = (i % 2 == 0) ? -1.0f : 1.0f;
        }

        const float value = x[base + static_cast<size_t>(i)];
        const float partner_value = x[base + static_cast<size_t>(partner)];
        (*out)[base + static_cast<size_t>(i)] =
            value * cos_full[angle_row + static_cast<size_t>(i)] +
            sign * partner_value * sin_full[angle_row + static_cast<size_t>(i)];
      }

      // Dimensions past rotary_dim pass through unchanged.
      for (int64_t i = rotary_dim; i < head_dim; ++i) {
        (*out)[base + static_cast<size_t>(i)] = x[base + static_cast<size_t>(i)];
      }
    }
  }
}

size_t NzCacheOffset(const PagedKvLayout& layout, int64_t block_id, int64_t block_offset, int64_t kv_head,
                     int64_t dim) {
  const int64_t hidden_index = kv_head * layout.head_size + dim;
  const int64_t fractal_row = hidden_index / 16;
  const int64_t fractal_column = hidden_index % 16;

  return ((static_cast<size_t>(block_id) * static_cast<size_t>(layout.fractal_rows()) +
           static_cast<size_t>(fractal_row)) *
              static_cast<size_t>(layout.block_size) +
          static_cast<size_t>(block_offset)) *
             16u +
         static_cast<size_t>(fractal_column);
}

size_t Dense4DCacheOffset(const PagedKvLayout& layout, int64_t block_id, int64_t block_offset,
                          int64_t kv_head, int64_t dim) {
  return ((static_cast<size_t>(block_id) * static_cast<size_t>(layout.num_kv_heads) +
           static_cast<size_t>(kv_head)) *
              static_cast<size_t>(layout.block_size) +
          static_cast<size_t>(block_offset)) *
             static_cast<size_t>(layout.head_size) +
         static_cast<size_t>(dim);
}

size_t CacheOffset(const PagedKvLayout& layout, int64_t block_id, int64_t block_offset, int64_t kv_head,
                   int64_t dim) {
  if (layout.kind == PagedKvCacheLayout::kDense4D) {
    return Dense4DCacheOffset(layout, block_id, block_offset, kv_head, dim);
  }
  return NzCacheOffset(layout, block_id, block_offset, kv_head, dim);
}

void ReshapeAndCache(const std::vector<float>& key, const std::vector<float>& value,
                     const std::vector<int32_t>& slot_mapping, const PagedKvLayout& layout,
                     std::vector<float>* key_cache, std::vector<float>* value_cache) {
  assert(key.size() == value.size());
  assert(key_cache->size() == layout.ElementCount());
  assert(value_cache->size() == layout.ElementCount());

  const int64_t num_tokens = static_cast<int64_t>(slot_mapping.size());
  for (int64_t token = 0; token < num_tokens; ++token) {
    const int32_t slot = slot_mapping[static_cast<size_t>(token)];
    if (slot < 0) {
      continue;  // vLLM marks padded tokens with a negative slot
    }
    const int64_t block_id = slot / layout.block_size;
    const int64_t block_offset = slot % layout.block_size;

    for (int64_t head = 0; head < layout.num_kv_heads; ++head) {
      for (int64_t dim = 0; dim < layout.head_size; ++dim) {
        const size_t source = static_cast<size_t>((token * layout.num_kv_heads + head) * layout.head_size + dim);
        const size_t destination = CacheOffset(layout, block_id, block_offset, head, dim);
        (*key_cache)[destination] = key[source];
        (*value_cache)[destination] = value[source];
      }
    }
  }
}

void PagedAttentionDecode(const std::vector<float>& query, const std::vector<float>& key_cache,
                          const std::vector<float>& value_cache, const std::vector<int32_t>& block_table,
                          const std::vector<int32_t>& context_lens, const PagedKvLayout& layout,
                          const PagedAttentionShape& shape, std::vector<float>* out) {
  assert(shape.num_kv_heads > 0);
  assert(shape.num_heads % shape.num_kv_heads == 0);

  const int64_t group_size = shape.num_heads / shape.num_kv_heads;
  out->assign(static_cast<size_t>(shape.num_seqs * shape.num_heads * shape.head_size), 0.0f);

  std::vector<float> scores;

  for (int64_t seq = 0; seq < shape.num_seqs; ++seq) {
    const int64_t context_len = context_lens[static_cast<size_t>(seq)];
    if (context_len <= 0) {
      continue;
    }
    const size_t table_row = static_cast<size_t>(seq * shape.max_blocks_per_seq);

    for (int64_t head = 0; head < shape.num_heads; ++head) {
      const int64_t kv_head = head / group_size;
      const size_t query_base = static_cast<size_t>((seq * shape.num_heads + head) * shape.head_size);

      scores.assign(static_cast<size_t>(context_len), 0.0f);

      float max_score = -std::numeric_limits<float>::infinity();
      for (int64_t position = 0; position < context_len; ++position) {
        const int64_t logical_block = position / shape.block_size;
        const int64_t block_offset = position % shape.block_size;
        const int64_t physical_block = block_table[table_row + static_cast<size_t>(logical_block)];

        float dot = 0.0f;
        for (int64_t dim = 0; dim < shape.head_size; ++dim) {
          const size_t key_index = CacheOffset(layout, physical_block, block_offset, kv_head, dim);
          dot += query[query_base + static_cast<size_t>(dim)] * key_cache[key_index];
        }
        const float score = dot * shape.scale;
        scores[static_cast<size_t>(position)] = score;
        if (score > max_score) {
          max_score = score;
        }
      }

      // Softmax with the standard max subtraction, matching the kernel.
      float denominator = 0.0f;
      for (int64_t position = 0; position < context_len; ++position) {
        const float weight = std::exp(scores[static_cast<size_t>(position)] - max_score);
        scores[static_cast<size_t>(position)] = weight;
        denominator += weight;
      }
      const float inverse_denominator = 1.0f / denominator;

      for (int64_t position = 0; position < context_len; ++position) {
        const float weight = scores[static_cast<size_t>(position)] * inverse_denominator;
        const int64_t logical_block = position / shape.block_size;
        const int64_t block_offset = position % shape.block_size;
        const int64_t physical_block = block_table[table_row + static_cast<size_t>(logical_block)];

        for (int64_t dim = 0; dim < shape.head_size; ++dim) {
          const size_t value_index = CacheOffset(layout, physical_block, block_offset, kv_head, dim);
          (*out)[query_base + static_cast<size_t>(dim)] += weight * value_cache[value_index];
        }
      }
    }
  }
}

}  // namespace reference
}  // namespace test
}  // namespace vllm_ascend
