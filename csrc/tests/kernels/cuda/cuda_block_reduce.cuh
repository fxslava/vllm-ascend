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

// Warp and block reductions shared by the RMSNorm and paged-attention kernels.
//
// The warp stage uses __shfl_xor_sync, which leaves every lane holding the
// result rather than only lane 0, so the second stage needs no extra broadcast
// inside the warp.
//
// Both block reductions end with
//
//     __syncthreads(); read scratch[0]; __syncthreads();
//
// The trailing barrier is what makes them safe to call twice on the same
// scratch array: without it a fast warp entering the next reduction could
// overwrite scratch[0] while a slow warp had not yet read the previous result.
// The paged-attention kernel does exactly that, once for the max and once for
// the softmax denominator.

#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

namespace vllm_ascend {
namespace test {
namespace cuda {

// Every architecture this suite targets has 32-lane warps. Spelled out rather
// than read from warpSize, which is not a compile-time constant.
constexpr int kWarpSize = 32;
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFu;

// Upper bound on warps per block: 1024 threads / 32 lanes.
constexpr int kMaxWarpsPerBlock = 32;

// The identity for a max reduction. CUDART_INF_F is __int_as_float(0x7f800000),
// an exact bit pattern; the <cmath> INFINITY macro expands to a double literal
// that MSVC narrows to float, which nvcc reports as a lossy conversion.
#define VLLM_ASCEND_NEGATIVE_INFINITY (-CUDART_INF_F)

__device__ __forceinline__ float WarpReduceSum(float value) {
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(kFullWarpMask, value, mask);
  }
  return value;
}

__device__ __forceinline__ float WarpReduceMax(float value) {
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(kFullWarpMask, value, mask));
  }
  return value;
}

// `scratch` must hold at least kMaxWarpsPerBlock floats and be the same array
// for every thread in the block.
__device__ __forceinline__ float BlockReduceSum(float value, float* scratch) {
  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int warp_count = static_cast<int>((blockDim.x + kWarpSize - 1) / kWarpSize);

  value = WarpReduceSum(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  float total = (static_cast<int>(threadIdx.x) < warp_count) ? scratch[threadIdx.x] : 0.0f;
  __syncthreads();  // every read of scratch is retired before it is rewritten

  if (warp == 0) {
    total = WarpReduceSum(total);
    if (lane == 0) {
      scratch[0] = total;
    }
  }
  __syncthreads();
  const float result = scratch[0];
  __syncthreads();
  return result;
}

__device__ __forceinline__ float BlockReduceMax(float value, float* scratch) {
  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int warp_count = static_cast<int>((blockDim.x + kWarpSize - 1) / kWarpSize);

  value = WarpReduceMax(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  float total =
      (static_cast<int>(threadIdx.x) < warp_count) ? scratch[threadIdx.x] : VLLM_ASCEND_NEGATIVE_INFINITY;
  __syncthreads();

  if (warp == 0) {
    total = WarpReduceMax(total);
    if (lane == 0) {
      scratch[0] = total;
    }
  }
  __syncthreads();
  const float result = scratch[0];
  __syncthreads();
  return result;
}

}  // namespace cuda
}  // namespace test
}  // namespace vllm_ascend
