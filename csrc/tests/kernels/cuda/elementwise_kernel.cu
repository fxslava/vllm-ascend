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

// The two elementwise steps a decoder layer needs between the operators that
// already have kernels of their own:
//
//     out[i] = x[i] * sigmoid(g[i])     attention output gate
//     x[i]  += delta[i]                 residual add, in place
//
// Both are memory-bound and trivially parallel, so they are written the same
// way as swiglu_kernel.cu: a grid-stride loop, a vectorised half2 path for an
// even element count, and a scalar path so an odd count is slower rather than
// wrong. Arithmetic is fp32 with a single rounding to fp16 at the store, which
// is what every other kernel in this directory does and what the CPU reference
// models.
//
// sigmoid is written 1/(1+expf(-v)) with the accurate libdevice expf rather
// than __expf, matching the SiluFloat next door: the gate saturates for a
// large |v|, which is exactly where the fast intrinsic loses bits.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "cuda_kernels.hpp"

namespace vllm_ascend {
namespace test {
namespace cuda {
namespace {

constexpr int kElementwiseBlockThreads = 256;
constexpr long long kElementwiseMaxBlocks = 65535;

__device__ __forceinline__ float SigmoidFloat(float value) { return 1.0f / (1.0f + expf(-value)); }

__device__ __forceinline__ long long ElementwiseGridStride() {
  return static_cast<long long>(gridDim.x) * static_cast<long long>(blockDim.x);
}

__device__ __forceinline__ long long ElementwiseGlobalIndex() {
  return static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x) +
         static_cast<long long>(threadIdx.x);
}

unsigned int BlockCount(long long work_items) {
  const long long blocks = (work_items + kElementwiseBlockThreads - 1) / kElementwiseBlockThreads;
  return static_cast<unsigned int>((blocks < kElementwiseMaxBlocks) ? blocks : kElementwiseMaxBlocks);
}

__global__ void SigmoidGateHalf2Kernel(const __half* __restrict__ x, const __half* __restrict__ gate,
                                       __half* __restrict__ out, long long total_pairs) {
  for (long long pair = ElementwiseGlobalIndex(); pair < total_pairs; pair += ElementwiseGridStride()) {
    const long long element = pair * 2;

    const float2 value = __half22float2(*reinterpret_cast<const __half2*>(x + element));
    const float2 g = __half22float2(*reinterpret_cast<const __half2*>(gate + element));

    const float2 result =
        make_float2(value.x * SigmoidFloat(g.x), value.y * SigmoidFloat(g.y));
    *reinterpret_cast<__half2*>(out + element) = __float22half2_rn(result);
  }
}

__global__ void SigmoidGateScalarKernel(const __half* __restrict__ x, const __half* __restrict__ gate,
                                        __half* __restrict__ out, long long count) {
  for (long long element = ElementwiseGlobalIndex(); element < count;
       element += ElementwiseGridStride()) {
    const float value = __half2float(x[element]);
    const float g = __half2float(gate[element]);
    out[element] = __float2half(value * SigmoidFloat(g));
  }
}

__global__ void ResidualAddHalf2Kernel(__half* __restrict__ x, const __half* __restrict__ delta,
                                       long long total_pairs) {
  for (long long pair = ElementwiseGlobalIndex(); pair < total_pairs; pair += ElementwiseGridStride()) {
    const long long element = pair * 2;

    const float2 value = __half22float2(*reinterpret_cast<const __half2*>(x + element));
    const float2 d = __half22float2(*reinterpret_cast<const __half2*>(delta + element));

    const float2 result = make_float2(value.x + d.x, value.y + d.y);
    *reinterpret_cast<__half2*>(x + element) = __float22half2_rn(result);
  }
}

__global__ void ResidualAddScalarKernel(__half* __restrict__ x, const __half* __restrict__ delta,
                                        long long count) {
  for (long long element = ElementwiseGlobalIndex(); element < count;
       element += ElementwiseGridStride()) {
    x[element] = __float2half(__half2float(x[element]) + __half2float(delta[element]));
  }
}

}  // namespace

void LaunchSigmoidGateHalf(const uint16_t* x, const uint16_t* gate, uint16_t* out, int64_t count,
                           cudaStream_t stream) {
  if (count <= 0) {
    return;
  }

  const dim3 block(kElementwiseBlockThreads);

  if (count % 2 == 0) {
    const long long total_pairs = count / 2;
    const dim3 grid(BlockCount(total_pairs));
    SigmoidGateHalf2Kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(gate),
        reinterpret_cast<__half*>(out), total_pairs);
  } else {
    const dim3 grid(BlockCount(count));
    SigmoidGateScalarKernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(gate),
        reinterpret_cast<__half*>(out), static_cast<long long>(count));
  }

  CUDA_CHECK(cudaGetLastError());
}

void LaunchResidualAddHalf(uint16_t* x, const uint16_t* delta, int64_t count, cudaStream_t stream) {
  if (count <= 0) {
    return;
  }

  const dim3 block(kElementwiseBlockThreads);

  if (count % 2 == 0) {
    const long long total_pairs = count / 2;
    const dim3 grid(BlockCount(total_pairs));
    ResidualAddHalf2Kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(x), reinterpret_cast<const __half*>(delta), total_pairs);
  } else {
    const dim3 grid(BlockCount(count));
    ResidualAddScalarKernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(x), reinterpret_cast<const __half*>(delta),
        static_cast<long long>(count));
  }

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace test
}  // namespace vllm_ascend
