/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file cpu_tiling_shim.h
 * \brief GET_TILING_DATA_WITH_STRUCT for out-of-pipeline (CPU-interpreter) builds.
 *
 * GET_TILING_DATA_WITH_STRUCT is NOT a header macro. The TBE op-compile
 * pipeline synthesises it per operator (see
 * $ASCEND_HOME_PATH/python/site-packages/tbe/tikcpp/get_op_tiling.py), baking
 * the host tiling function's output into the kernel binary as a constexpr byte
 * array. That only happens inside `build.sh --pkg`, so a kernel source compiled
 * anywhere else -- such as this CPU-interpreter harness -- has no definition
 * for it and will not compile.
 *
 * This header supplies the equivalent: deserialise the struct from the raw
 * tiling buffer the caller passes. The values are the same ones the host tiling
 * function emits, so the kernel sees an identical tiling struct either way.
 *
 * It is force-included (-include) ONLY by the harness target. The shipping op
 * package never sees it, and the kernel source needs no conditional compilation.
 */

#ifndef DSV4_MOE_EXPERT_CPU_TILING_SHIM_H
#define DSV4_MOE_EXPERT_CPU_TILING_SHIM_H

#include <cstdint>

// Deliberately a macro with no helper function: the body is only expanded at
// the use site inside the kernel, where __gm__ is already defined. A template
// taking a plain pointer would drop the address-space qualifier, which the
// device compiler does not tolerate; the force-include happens before any
// AscendC header, so nothing here may reference __gm__ at file scope.
//
// The copy is byte-wise because the tiling buffer is a raw serialisation of the
// host struct and nothing guarantees its alignment.
#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg)                        \
    tiling_struct tiling_data;                                                                     \
    do {                                                                                           \
        __gm__ const uint8_t *dsv4ShimSrc = reinterpret_cast<__gm__ const uint8_t *>(tiling_arg);   \
        uint8_t *dsv4ShimDst = reinterpret_cast<uint8_t *>(&tiling_data);                          \
        for (unsigned dsv4ShimIdx = 0; dsv4ShimIdx < sizeof(tiling_struct); ++dsv4ShimIdx) {       \
            dsv4ShimDst[dsv4ShimIdx] = dsv4ShimSrc[dsv4ShimIdx];                                   \
        }                                                                                          \
    } while (0)
#endif

#endif // DSV4_MOE_EXPERT_CPU_TILING_SHIM_H
