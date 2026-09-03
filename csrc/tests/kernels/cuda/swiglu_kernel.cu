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

// SiluAndMul / SwiGLU, fp16 in / fp16 out:
//
//     out[t][i] = silu(x[t][i]) * x[t][intermediate + i]
//     silu(v)   = v / (1 + exp(-v))
//
// The division form is used rather than v * sigmoid(v) because that is what
// reference::SiluAndMul evaluates, and the two differ in the last fp32 bit.
// exp is the accurate libdevice expf, not __expf: the saturation test drives
// the gate to +/-30, where the fast intrinsic has visibly more error.
//
// The vectorised path loads a half2 from each half of the row. Both loads are
// 4-byte aligned whenever `intermediate` is even: the row base offset
// token * 2 * intermediate is even, the element index i is even by
// construction, and intermediate + i is then even too. Every Qwen3.5 MLP width
// is even, so the scalar kernel is only there to keep an odd width slow rather
// than wrong.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "cuda_kernels.hpp"

namespace vllm_ascend {
namespace test {
namespace cuda {
namespace {

constexpr int kSwiGluBlockThreads = 256;

// Cap on the grid so a very wide MLP still launches; the kernels are
// grid-stride loops, so a short grid costs iterations rather than correctness.
constexpr long long kSwiGluMaxBlocks = 65535;

__device__ __forceinline__ float SiluFloat(float value) { return value / (1.0f + expf(-value)); }

__device__ __forceinline__ long long SwiGluGridStride() {
  return static_cast<long long>(gridDim.x) * static_cast<long long>(blockDim.x);
}

__device__ __forceinline__ long long SwiGluGlobalIndex() {
  return static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x) +
         static_cast<long long>(threadIdx.x);
}

// Two output elements per thread.
__global__ void SwiGluHalf2Kernel(const __half* __restrict__ x, __half* __restrict__ out,
                                  long long pairs_per_row, long long intermediate,
                                  long long total_pairs) {
  for (long long pair = SwiGluGlobalIndex(); pair < total_pairs; pair += SwiGluGridStride()) {
    const long long token = pair / pairs_per_row;
    const long long column = (pair - token * pairs_per_row) * 2;

    const long long in_row = token * 2 * intermediate;
    const long long out_row = token * intermediate;

    const __half2 gate = *reinterpret_cast<const __half2*>(x + in_row + column);
    const __half2 up = *reinterpret_cast<const __half2*>(x + in_row + intermediate + column);

    const float2 gate_value = __half22float2(gate);
    const float2 up_value = __half22float2(up);

    const float2 result = make_float2(SiluFloat(gate_value.x) * up_value.x,
                                      SiluFloat(gate_value.y) * up_value.y);
    *reinterpret_cast<__half2*>(out + out_row + column) = __float22half2_rn(result);
  }
}

__global__ void SwiGluScalarKernel(const __half* __restrict__ x, __half* __restrict__ out,
                                   long long intermediate, long long total_elements) {
  for (long long element = SwiGluGlobalIndex(); element < total_elements;
       element += SwiGluGridStride()) {
    const long long token = element / intermediate;
    const long long column = element - token * intermediate;

    const long long in_row = token * 2 * intermediate;

    const float gate = __half2float(x[in_row + column]);
    const float up = __half2float(x[in_row + intermediate + column]);
    out[element] = __float2half(SiluFloat(gate) * up);
  }
}

unsigned int BlockCount(long long work_items) {
  const long long blocks = (work_items + kSwiGluBlockThreads - 1) / kSwiGluBlockThreads;
  return static_cast<unsigned int>((blocks < kSwiGluMaxBlocks) ? blocks : kSwiGluMaxBlocks);
}

}  // namespace

void LaunchSwiGluHalf(const uint16_t* x, uint16_t* out, int64_t num_tokens, int64_t intermediate,
                      cudaStream_t stream) {
  if (num_tokens <= 0 || intermediate <= 0) {
    return;
  }

  const dim3 block(kSwiGluBlockThreads);

  if (intermediate % 2 == 0) {
    const long long pairs_per_row = intermediate / 2;
    const long long total_pairs = static_cast<long long>(num_tokens) * pairs_per_row;
    const dim3 grid(BlockCount(total_pairs));

    SwiGluHalf2Kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const __half*>(x),
                                                  reinterpret_cast<__half*>(out), pairs_per_row,
                                                  intermediate, total_pairs);
  } else {
    const long long total_elements = static_cast<long long>(num_tokens) * intermediate;
    const dim3 grid(BlockCount(total_elements));

    SwiGluScalarKernel<<<grid, block, 0, stream>>>(reinterpret_cast<const __half*>(x),
                                                   reinterpret_cast<__half*>(out), intermediate,
                                                   total_elements);
  }

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace test
}  // namespace vllm_ascend
