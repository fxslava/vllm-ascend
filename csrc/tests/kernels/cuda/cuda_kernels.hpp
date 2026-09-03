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

// Host-callable launchers for the native CUDA C kernels.
//
// This header is the seam between the test binaries, which the host compiler
// builds, and the kernels, which nvcc builds. It therefore names no CUDA
// language extension and no __half: fp16 buffers are passed as uint16_t*,
// which is the same storage-only convention fp16.hpp uses on the host, and the
// .cu side reinterprets them as __half. The only CUDA type that crosses the
// seam is cudaStream_t, out of the plain C runtime header.
//
// Every launcher is synchronous in the sense that matters for a test: it
// enqueues on `stream` and then checks cudaGetLastError, throwing CudaError on
// a launch failure. Waiting for completion is the caller's job, which for the
// tests happens when a result is copied back.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace vllm_ascend {
namespace test {
namespace cuda {

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------
// x     [num_tokens, hidden]      fp16
// gamma [hidden]                  fp16
// y     [num_tokens, hidden]      fp16   = x * rstd * gamma
// rstd  [num_tokens]              fp32   = 1 / sqrt(mean(x^2) + epsilon)
//
// The reduction runs in fp32 and `rstd` may be null when the caller does not
// want it, matching how the plugin discards the second output of
// torch_npu.npu_rms_norm.
void LaunchRmsNormHalf(const uint16_t* x, const uint16_t* gamma, uint16_t* y, float* rstd,
                       int64_t num_tokens, int64_t hidden, float epsilon, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Rotary position embedding, "half" (neox / rotate_half) layout
// ---------------------------------------------------------------------------
// x         [num_tokens, num_heads, head_dim]  fp16, rotated in place
// cos_full  [num_tokens, rotary_dim]           fp16, shared across heads
// sin_full  [num_tokens, rotary_dim]           fp16, shared across heads
//
//   v1' = v1 * cos - v2 * sin
//   v2' = v2 * cos + v1 * sin
//
// where v1 is element i and v2 is element i + rotary_dim/2. Dimensions past
// rotary_dim pass through untouched, which for an in-place kernel means they
// are simply not written. head_dim is 64 or 128 for the Qwen3.5 configurations
// this suite covers; nothing here requires it, but see kRotarySupportedHeadDims.
void LaunchApplyRotaryPosEmbHalfMode(uint16_t* x, const uint16_t* cos_full, const uint16_t* sin_full,
                                     int64_t num_tokens, int64_t num_heads, int64_t head_dim,
                                     int64_t rotary_dim, cudaStream_t stream);

// ---------------------------------------------------------------------------
// SiluAndMul / SwiGLU
// ---------------------------------------------------------------------------
// x   [num_tokens, 2 * intermediate]  fp16
// out [num_tokens, intermediate]      fp16 = silu(x[..., :i]) * x[..., i:]
//
// silu(v) = v / (1 + exp(-v)), evaluated in fp32. An even `intermediate` takes
// the vectorised half2 path; the scalar path exists so an odd width is a
// slower answer rather than a wrong one.
void LaunchSwiGluHalf(const uint16_t* x, uint16_t* out, int64_t num_tokens, int64_t intermediate,
                      cudaStream_t stream);

// ---------------------------------------------------------------------------
// Attention output gate
// ---------------------------------------------------------------------------
// x    [count]  fp16   attention context, before the out projection
// gate [count]  fp16   the gate projection's output, same shape as x
// out  [count]  fp16   = x * sigmoid(gate)
//
// This is the `attn_output_gate` a Qwen3-Next-style block applies to the
// attention context before o_proj: a separate projection off the same
// normalised hidden state produces `gate`, and the context is modulated by its
// sigmoid. sigmoid is evaluated in fp32; `out` may alias `x`.
void LaunchSigmoidGateHalf(const uint16_t* x, const uint16_t* gate, uint16_t* out, int64_t count,
                           cudaStream_t stream);

// ---------------------------------------------------------------------------
// Residual add
// ---------------------------------------------------------------------------
// x     [count]  fp16, updated in place: x += delta
// delta [count]  fp16
//
// The add is done in fp32 and rounded once, so a residual stream that has
// grown past the fp16 range of its increment still accumulates the way the
// reference does.
void LaunchResidualAddHalf(uint16_t* x, const uint16_t* delta, int64_t count, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Paged KV cache, dense 4-D layout
// ---------------------------------------------------------------------------
// The CUDA backend stores the cache as
//     [num_blocks, num_kv_heads, block_size, head_dim]
// rather than the 5-D fractal-NZ shape the Ascend 310P attention kernel wants.
// reference::PagedKvCacheLayout::kDense4D selects the matching host reference,
// so both backends are still checked against the same two functions.
//
// key / value  [num_tokens, num_kv_heads, head_dim]  fp16
// slot_mapping [num_tokens]                          int32, slot =
//                                                    block_id * block_size + offset
//
// A negative slot marks a padded token and is skipped, as in vLLM. The copy is
// a straight fp16 move with no conversion, so the cache is bit-identical to
// the input.
void LaunchPagedCacheScatterHalf(const uint16_t* key, const uint16_t* value, const int32_t* slot_mapping,
                                 uint16_t* key_cache, uint16_t* value_cache, int64_t num_tokens,
                                 int64_t num_kv_heads, int64_t head_dim, int64_t block_size,
                                 cudaStream_t stream);

// ---------------------------------------------------------------------------
// Paged attention, decode (exactly one query token per sequence)
// ---------------------------------------------------------------------------
// query        [num_seqs, num_heads, head_size]            fp16
// key_cache    [num_blocks, num_kv_heads, block_size, hd]  fp16
// value_cache  same shape                                  fp16
// block_table  [num_seqs, max_blocks_per_seq]              int32
// context_lens [num_seqs]                                  int32
// out          [num_seqs, num_heads, head_size]            fp16
//
// GQA follows vLLM: query head h reads kv head h / (num_heads / num_kv_heads).
// Scores, softmax and the value accumulation all run in fp32.
//
// Like vLLM's own paged_attention_v1, this variant keeps the whole score row
// in shared memory, which caps the context length it can serve. Ask
// PagedAttentionDecodeV1SharedBytes first and compare it against the device
// limit; the launcher throws rather than silently truncating.
void LaunchPagedAttentionDecodeV1Half(const uint16_t* query, const uint16_t* key_cache,
                                      const uint16_t* value_cache, const int32_t* block_table,
                                      const int32_t* context_lens, uint16_t* out, int64_t num_seqs,
                                      int64_t num_heads, int64_t num_kv_heads, int64_t head_size,
                                      int64_t block_size, int64_t max_blocks_per_seq,
                                      int64_t max_context_len, float scale, cudaStream_t stream);

// Dynamic shared memory the decode kernel needs: the query row plus one float
// per context position.
size_t PagedAttentionDecodeV1SharedBytes(int64_t head_size, int64_t max_context_len);

}  // namespace cuda
}  // namespace test
}  // namespace vllm_ascend
