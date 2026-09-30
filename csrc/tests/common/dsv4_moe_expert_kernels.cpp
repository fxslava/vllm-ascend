/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

// The one translation unit ascendc_library compiles for the DSV4 expert.
// Test-owned, like common/hadamard_spike_kernels.cpp -- but unlike that file it
// contains no kernel of its own: it includes the SHIPPING kernel source
// verbatim, so the suite and the wheel cannot drift.
//
// Two ascendc_library constraints force this shape, and both are the tool's:
//
//   1. A <<<>>> call site can only reach a kernel whose generated launcher is
//      in the same ascendc_library.
//   2. The launcher generator emits a stub calling `<name>_origin` for every
//      `__global__ __aicore__` signature it finds -- including a bare
//      *declaration*. Listing the kernel source and a separate launcher TU
//      that forward-declares it in one library therefore yields two stubs for
//      one name, and the merge link dies with
//      `undefined symbol: dsv4_moe_expert_origin`.
//
// So the kernel and its launcher are one TU, and that TU is the only source
// the library is given.

#include "dsv4_tiling_shim.h"  // must precede the kernel: defines GET_TILING_DATA_WITH_STRUCT

// Relative on purpose: ascendc_library compiles through four nested
// ExternalProjects and -I does not reach every stage, but a quoted include is
// always resolved against this file's own directory.
#include "../../moe/dsv4_moe_expert/op_kernel/dsv4_moe_expert.cpp"  // NOLINT(bugprone-suspicious-include)

namespace vllm_ascend {

void dsv4_moe_expert_impl(void *stream, uint32_t blockDim, void *x, void *w1, void *w2, void *w3, void *w1Scale,
                          void *w2Scale, void *w3Scale, void *gateOut, void *upOut, void *activatedOut,
                          void *downOut, void *workspace, void *tiling)
{
    dsv4_moe_expert<<<blockDim, nullptr, stream>>>(x, w1, w2, w3, w1Scale, w2Scale, w3Scale, gateOut, upOut,
                                                   activatedOut, downOut, workspace, tiling);
}

}  // namespace vllm_ascend
