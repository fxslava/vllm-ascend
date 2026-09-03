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

// Paged KV cache scatter and decode attention over the dense 4-D layout
//
//     [num_blocks, num_kv_heads, block_size, head_dim]
//
// which is the standard GPU shape, not the 5-D fractal-NZ one the Ascend 310P
// attention kernel consumes. reference::PagedKvCacheLayout::kDense4D selects
// the matching offset in the host reference, so both backends are checked
// against the same ReshapeAndCache and PagedAttentionDecode.
//
// The decode kernel is the v1 shape: one block per (sequence, head), the whole
// score row held in shared memory, and three phases - dot products, softmax,
// weighted value sum - with a block-wide reduction between each. It follows
// reference::PagedAttentionDecode operation for operation, including the
// max subtraction and the `weight = score * (1/denominator)` factorisation, so
// the only difference is the order the reductions associate in.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_block_reduce.cuh"
#include "cuda_check.hpp"
#include "cuda_kernels.hpp"

namespace vllm_ascend {
namespace test {
namespace cuda {
namespace {

// One warp per 32 context positions in the scoring phase, and enough threads
// to cover a 128-wide head one element each in the output phase.
constexpr int kDecodeBlockThreads = 128;

// Flat element offset into [num_blocks, num_kv_heads, block_size, head_dim].
__device__ __forceinline__ long long DenseCacheOffset(int block_id, int kv_head, int block_offset,
                                                      int num_kv_heads, int block_size, int head_dim) {
  return ((static_cast<long long>(block_id) * num_kv_heads + kv_head) * block_size + block_offset) *
         head_dim;
}

// grid = (num_tokens, num_kv_heads), one thread per head dimension.
__global__ void PagedCacheScatterHalfKernel(const __half* __restrict__ key,
                                            const __half* __restrict__ value,
                                            const int* __restrict__ slot_mapping,
                                            __half* __restrict__ key_cache,
                                            __half* __restrict__ value_cache, int num_kv_heads,
                                            int head_dim, int block_size) {
  const int token = static_cast<int>(blockIdx.x);
  const int kv_head = static_cast<int>(blockIdx.y);

  const int slot = slot_mapping[token];
  if (slot < 0) {
    return;  // vLLM marks padded tokens with a negative slot
  }
  const int block_id = slot / block_size;
  const int block_offset = slot % block_size;

  const long long source = (static_cast<long long>(token) * num_kv_heads + kv_head) * head_dim;
  const long long destination =
      DenseCacheOffset(block_id, kv_head, block_offset, num_kv_heads, block_size, head_dim);

  // A straight fp16 move: no conversion, so the cache holds the input bits.
  for (int dim = static_cast<int>(threadIdx.x); dim < head_dim; dim += static_cast<int>(blockDim.x)) {
    key_cache[destination + dim] = key[source + dim];
    value_cache[destination + dim] = value[source + dim];
  }
}

// grid = (num_seqs, num_heads).
//
// Dynamic shared memory holds the query row in fp32 followed by one float per
// context position; `reduce_scratch` is a separate static allocation so the two
// block reductions cannot alias the score row.
__global__ void PagedAttentionDecodeV1HalfKernel(
    const __half* __restrict__ query, const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache, const int* __restrict__ block_table,
    const int* __restrict__ context_lens, __half* __restrict__ out, int num_heads, int num_kv_heads,
    int head_size, int block_size, int max_blocks_per_seq, float scale) {
  extern __shared__ float shared_storage[];
  __shared__ float reduce_scratch[kMaxWarpsPerBlock];

  const int sequence = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);

  // vLLM GQA mapping: query head h reads kv head h / (num_heads / num_kv_heads).
  const int group_size = num_heads / num_kv_heads;
  const int kv_head = head / group_size;

  const long long query_base = (static_cast<long long>(sequence) * num_heads + head) * head_size;
  const int context_len = context_lens[sequence];

  // An empty context contributes nothing; the reference leaves the row at zero.
  if (context_len <= 0) {
    for (int dim = static_cast<int>(threadIdx.x); dim < head_size;
         dim += static_cast<int>(blockDim.x)) {
      out[query_base + dim] = __float2half(0.0f);
    }
    return;
  }

  float* query_row = shared_storage;
  float* scores = shared_storage + head_size;

  for (int dim = static_cast<int>(threadIdx.x); dim < head_size; dim += static_cast<int>(blockDim.x)) {
    query_row[dim] = __half2float(query[query_base + dim]);
  }
  __syncthreads();

  const int* table_row = block_table + static_cast<long long>(sequence) * max_blocks_per_seq;

  // Phase 1: scaled dot product for every cached position.
  float thread_max = VLLM_ASCEND_NEGATIVE_INFINITY;
  for (int position = static_cast<int>(threadIdx.x); position < context_len;
       position += static_cast<int>(blockDim.x)) {
    const int logical_block = position / block_size;
    const int block_offset = position % block_size;
    const int physical_block = table_row[logical_block];
    const long long key_base =
        DenseCacheOffset(physical_block, kv_head, block_offset, num_kv_heads, block_size, head_size);

    float dot = 0.0f;
    for (int dim = 0; dim < head_size; ++dim) {
      dot += query_row[dim] * __half2float(key_cache[key_base + dim]);
    }

    const float score = dot * scale;
    scores[position] = score;
    thread_max = fmaxf(thread_max, score);
  }

  const float max_score = BlockReduceMax(thread_max, reduce_scratch);

  // Phase 2: softmax with the standard max subtraction, in place over `scores`.
  float thread_sum = 0.0f;
  for (int position = static_cast<int>(threadIdx.x); position < context_len;
       position += static_cast<int>(blockDim.x)) {
    const float weight = expf(scores[position] - max_score);
    scores[position] = weight;
    thread_sum += weight;
  }

  const float denominator = BlockReduceSum(thread_sum, reduce_scratch);
  const float inverse_denominator = 1.0f / denominator;

  // Phase 3: weighted sum over the value cache, one output element per thread.
  // Positions are visited in ascending order for a fixed dim, which is the
  // accumulation order reference::PagedAttentionDecode uses.
  for (int dim = static_cast<int>(threadIdx.x); dim < head_size; dim += static_cast<int>(blockDim.x)) {
    float accumulator = 0.0f;
    for (int position = 0; position < context_len; ++position) {
      const int logical_block = position / block_size;
      const int block_offset = position % block_size;
      const int physical_block = table_row[logical_block];
      const long long value_base = DenseCacheOffset(physical_block, kv_head, block_offset,
                                                    num_kv_heads, block_size, head_size);

      const float weight = scores[position] * inverse_denominator;
      accumulator += weight * __half2float(value_cache[value_base + dim]);
    }
    out[query_base + dim] = __float2half(accumulator);
  }
}

}  // namespace

void LaunchPagedCacheScatterHalf(const uint16_t* key, const uint16_t* value, const int32_t* slot_mapping,
                                 uint16_t* key_cache, uint16_t* value_cache, int64_t num_tokens,
                                 int64_t num_kv_heads, int64_t head_dim, int64_t block_size,
                                 cudaStream_t stream) {
  if (num_tokens <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
    return;
  }
  if (block_size <= 0) {
    throw CudaError("block_size must be positive", __FILE__, __LINE__, cudaErrorInvalidValue);
  }

  const int threads = (head_dim < 256) ? static_cast<int>(head_dim) : 256;
  const dim3 grid(static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(num_kv_heads));
  const dim3 block(static_cast<unsigned int>(threads));

  PagedCacheScatterHalfKernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(key), reinterpret_cast<const __half*>(value), slot_mapping,
      reinterpret_cast<__half*>(key_cache), reinterpret_cast<__half*>(value_cache),
      static_cast<int>(num_kv_heads), static_cast<int>(head_dim), static_cast<int>(block_size));

  CUDA_CHECK(cudaGetLastError());
}

size_t PagedAttentionDecodeV1SharedBytes(int64_t head_size, int64_t max_context_len) {
  const int64_t floats = head_size + ((max_context_len > 0) ? max_context_len : 0);
  return static_cast<size_t>(floats) * sizeof(float);
}

void LaunchPagedAttentionDecodeV1Half(const uint16_t* query, const uint16_t* key_cache,
                                      const uint16_t* value_cache, const int32_t* block_table,
                                      const int32_t* context_lens, uint16_t* out, int64_t num_seqs,
                                      int64_t num_heads, int64_t num_kv_heads, int64_t head_size,
                                      int64_t block_size, int64_t max_blocks_per_seq,
                                      int64_t max_context_len, float scale, cudaStream_t stream) {
  if (num_seqs <= 0 || num_heads <= 0) {
    return;
  }
  if (num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
    throw CudaError("num_heads must be a positive multiple of num_kv_heads", __FILE__, __LINE__,
                    cudaErrorInvalidValue);
  }

  const size_t shared_bytes = PagedAttentionDecodeV1SharedBytes(head_size, max_context_len);

  // The default per-block limit is 48 KB on every architecture this suite
  // targets. Report it as an error rather than letting the launch fail with a
  // bare cudaErrorInvalidValue that says nothing about which shape caused it.
  int shared_limit = 0;
  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));
  CUDA_CHECK(cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlock, device_id));
  if (shared_bytes > static_cast<size_t>(shared_limit)) {
    throw CudaError(
        "paged_attention_decode_v1 needs more shared memory than the device allows for this context "
        "length; a v2-style split-K decode would be required",
        __FILE__, __LINE__, cudaErrorInvalidValue);
  }

  const dim3 grid(static_cast<unsigned int>(num_seqs), static_cast<unsigned int>(num_heads));
  const dim3 block(kDecodeBlockThreads);

  PagedAttentionDecodeV1HalfKernel<<<grid, block, shared_bytes, stream>>>(
      reinterpret_cast<const __half*>(query), reinterpret_cast<const __half*>(key_cache),
      reinterpret_cast<const __half*>(value_cache), block_table, context_lens,
      reinterpret_cast<__half*>(out), static_cast<int>(num_heads), static_cast<int>(num_kv_heads),
      static_cast<int>(head_size), static_cast<int>(block_size),
      static_cast<int>(max_blocks_per_seq), scale);

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace test
}  // namespace vllm_ascend
