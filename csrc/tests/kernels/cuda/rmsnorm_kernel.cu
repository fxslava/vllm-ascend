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

// RMSNorm, fp16 in / fp16 out with an fp32 reduction:
//
//     rstd = 1 / sqrt(mean(x^2) + epsilon)
//     y    = x * rstd * gamma
//
// One block per token. The multiply order in the second loop is
// (x * rstd) * gamma, matching reference::RmsNorm element for element so the
// two differ only by the order the squares are summed in - a tree here, a
// sequential loop there - which is fp32 reassociation noise far below the fp16
// rounding of the output.
//
// sqrtf and the division are used rather than rsqrtf: rsqrtf is the fast
// approximate reciprocal square root, and the near-zero-row test drives rstd
// up to 1e3 where its error is easiest to see.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_block_reduce.cuh"
#include "cuda_check.hpp"
#include "cuda_kernels.hpp"

namespace vllm_ascend {
namespace test {
namespace cuda {
namespace {

// 256 threads covers every hidden size the suite uses (1536 to 8192) with a
// short strided loop, and keeps eight warps in the block reduction.
constexpr int kRmsNormBlockThreads = 256;

__global__ void RmsNormHalfKernel(const __half* __restrict__ x, const __half* __restrict__ gamma,
                                  __half* __restrict__ y, float* __restrict__ rstd, int hidden,
                                  float epsilon) {
  __shared__ float reduce_scratch[kMaxWarpsPerBlock];

  const long long token = blockIdx.x;
  const long long row = token * hidden;

  float sum_of_squares = 0.0f;
  for (int i = static_cast<int>(threadIdx.x); i < hidden; i += static_cast<int>(blockDim.x)) {
    const float value = __half2float(x[row + i]);
    sum_of_squares += value * value;
  }
  sum_of_squares = BlockReduceSum(sum_of_squares, reduce_scratch);

  const float mean_square = sum_of_squares / static_cast<float>(hidden);
  const float inverse_rms = 1.0f / sqrtf(mean_square + epsilon);

  if (threadIdx.x == 0 && rstd != nullptr) {
    rstd[token] = inverse_rms;
  }

  for (int i = static_cast<int>(threadIdx.x); i < hidden; i += static_cast<int>(blockDim.x)) {
    const float value = __half2float(x[row + i]);
    y[row + i] = __float2half(value * inverse_rms * __half2float(gamma[i]));
  }
}

}  // namespace

void LaunchRmsNormHalf(const uint16_t* x, const uint16_t* gamma, uint16_t* y, float* rstd,
                       int64_t num_tokens, int64_t hidden, float epsilon, cudaStream_t stream) {
  if (num_tokens <= 0 || hidden <= 0) {
    return;
  }

  const dim3 grid(static_cast<unsigned int>(num_tokens));
  const dim3 block(kRmsNormBlockThreads);

  RmsNormHalfKernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(gamma),
      reinterpret_cast<__half*>(y), rstd, static_cast<int>(hidden), epsilon);

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace test
}  // namespace vllm_ascend
