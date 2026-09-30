/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file dsv4_launch.cpp
 * \brief The one translation unit ascendc_library compiles for Gate C.
 *
 * Two constraints force this shape, and both are ascendc_library's:
 *
 *   1. A <<<>>> call site can only reach a kernel whose generated launcher is
 *      in the same ascendc_library.
 *   2. The launcher generator scans each source for `__global__ __aicore__`
 *      signatures and emits a stub calling `<name>_origin` for every one it
 *      finds -- including a bare *declaration*. Putting the kernel source and a
 *      separate launcher TU that forward-declares the kernel into one library
 *      therefore yields two stubs for one name and the merge link dies with
 *      `undefined symbol: dsv4_moe_expert_origin`.
 *
 * So the shipping kernel source is included verbatim here and the launcher sits
 * beside it, exactly as csrc/tests/common/hadamard_spike_kernels.cpp keeps its
 * kernels and their _impl launchers in one file. Including a .cpp is deliberate:
 * it guarantees Gate C measures the same source the wheel compiles, with no
 * second copy to drift.
 */

// Both includes are relative on purpose: ascendc_library compiles this source
// through four nested ExternalProjects, -I does not reach every stage, and a
// forwarded `-include` pair reaches the bisheng host-stub stage as two separate
// tokens ("cannot specify -o when generating multiple output files"). A quoted
// include is always resolved against this file's own directory, in every stage.
//
// The shim must precede the kernel: it defines GET_TILING_DATA_WITH_STRUCT,
// which the TBE op-package pipeline would otherwise have synthesised.
#include "../cpu_tiling_shim.h"

#include "../../../../csrc/moe/dsv4_moe_expert/op_kernel/dsv4_moe_expert.cpp" // NOLINT(bugprone-suspicious-include)

namespace dsv4_gate_c {

void LaunchDsv4MoeExpert(uint32_t blockDim, void *stream, void *x, void *w1, void *w2, void *w3, void *w1Scale,
                         void *w2Scale, void *w3Scale, void *gateOut, void *upOut, void *activatedOut,
                         void *downOut, void *workspace, void *tiling)
{
    dsv4_moe_expert<<<blockDim, nullptr, stream>>>(x, w1, w2, w3, w1Scale, w2Scale, w3Scale, gateOut, upOut,
                                                   activatedOut, downOut, workspace, tiling);
}

} // namespace dsv4_gate_c
