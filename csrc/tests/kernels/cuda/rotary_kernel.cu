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

// Rotary position embedding in the "half" (neox / rotate_half) layout, which is
// the one Qwen3.5 uses:
//
//     v1' = v1 * cos - v2 * sin
//     v2' = v2 * cos + v1 * sin
//
// with v1 = x[i] and v2 = x[i + rotary_dim/2].
//
// One block per (token, head), one thread per rotary pair. The update is in
// place, which is safe because a thread owns both halves of its pair: it reads
// v1 and v2 before writing either.
//
// cos_full / sin_full are [num_tokens, rotary_dim], the widened cache
// reference::GatherFullCosSin produces, and are shared across heads. Both
// halves are read separately even though GatherFullCosSin makes cos[i] and
// cos[i + half] equal by construction: the kernel then reads exactly the
// element reference::ApplyRotaryPosEmb reads, so a future cache layout that
// stopped duplicating would show up here rather than pass silently.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "cuda_kernels.hpp"

namespace vllm_ascend {
namespace test {
namespace cuda {
namespace {

// rotary_dim is 64 or 128 on the paths this suite covers, so half_dim is 32 or
// 64 and one warp or two covers a pair set exactly. The cap keeps the launch
// legal if a caller ever passes a wider head.
constexpr int kRotaryMaxBlockThreads = 256;

__global__ void ApplyRotaryPosEmbHalfModeKernel(__half* __restrict__ x, const __half* __restrict__ cos_full,
                                                const __half* __restrict__ sin_full, int num_heads,
                                                int head_dim, int rotary_dim) {
  const int half_dim = rotary_dim / 2;

  const long long index = blockIdx.x;  // token * num_heads + head
  const long long token = index / num_heads;
  const long long base = index * head_dim;
  const long long angle_row = token * rotary_dim;

  for (int k = static_cast<int>(threadIdx.x); k < half_dim; k += static_cast<int>(blockDim.x)) {
    const float first = __half2float(x[base + k]);
    const float second = __half2float(x[base + half_dim + k]);

    const float cos_first = __half2float(cos_full[angle_row + k]);
    const float sin_first = __half2float(sin_full[angle_row + k]);
    const float cos_second = __half2float(cos_full[angle_row + half_dim + k]);
    const float sin_second = __half2float(sin_full[angle_row + half_dim + k]);

    x[base + k] = __float2half(first * cos_first - second * sin_first);
    x[base + half_dim + k] = __float2half(second * cos_second + first * sin_second);
  }

  // Dimensions at or past rotary_dim pass through unchanged, which for an
  // in-place kernel means leaving them alone.
}

}  // namespace

void LaunchApplyRotaryPosEmbHalfMode(uint16_t* x, const uint16_t* cos_full, const uint16_t* sin_full,
                                     int64_t num_tokens, int64_t num_heads, int64_t head_dim,
                                     int64_t rotary_dim, cudaStream_t stream) {
  if (num_tokens <= 0 || num_heads <= 0 || rotary_dim <= 0) {
    return;
  }
  if (rotary_dim % 2 != 0 || rotary_dim > head_dim) {
    throw CudaError("rotary_dim must be even and no larger than head_dim", __FILE__, __LINE__,
                    cudaErrorInvalidValue);
  }

  const int half_dim = static_cast<int>(rotary_dim / 2);
  const int threads = (half_dim < kRotaryMaxBlockThreads) ? half_dim : kRotaryMaxBlockThreads;

  const dim3 grid(static_cast<unsigned int>(num_tokens * num_heads));
  const dim3 block(static_cast<unsigned int>(threads));

  ApplyRotaryPosEmbHalfModeKernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<__half*>(x), reinterpret_cast<const __half*>(cos_full),
      reinterpret_cast<const __half*>(sin_full), static_cast<int>(num_heads),
      static_cast<int>(head_dim), static_cast<int>(rotary_dim));

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace test
}  // namespace vllm_ascend
