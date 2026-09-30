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

// GET_TILING_DATA_WITH_STRUCT for builds outside the TBE op-compile pipeline.
//
// The macro is NOT a header macro. The op-compile pipeline synthesises it per
// operator (see $ASCEND_HOME_PATH/python/site-packages/tbe/tikcpp/
// get_op_tiling.py), baking the host tiling function's output into the kernel
// binary as a constexpr byte array. That only happens inside `build.sh --pkg`,
// so a kernel source compiled through ascendc_library -- as this suite does --
// has no definition for it and will not compile.
//
// This supplies the equivalent: deserialise the struct from the raw tiling
// buffer the caller passes. The values are the ones the host tiling function
// emits, so the kernel sees an identical struct either way.
//
// It is included by the test-owned kernel TU before the shipping kernel
// source, never force-included on a command line: a forwarded `-include` pair
// does not survive the bisheng host-stub stage ("cannot specify -o when
// generating multiple output files").

#ifndef VLLM_ASCEND_TESTS_DSV4_TILING_SHIM_H
#define VLLM_ASCEND_TESTS_DSV4_TILING_SHIM_H

#include <cstdint>

// Deliberately a macro with no helper function: the body is expanded at the
// use site inside the kernel, where __gm__ is already defined. A template
// taking a plain pointer would drop the address-space qualifier, which the
// device compiler does not tolerate, and this header is included before any
// AscendC header, so nothing here may reference __gm__ at file scope.
//
// The copy is byte-wise because the tiling buffer is a raw serialisation of
// the host struct and nothing guarantees its alignment.
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

#endif  // VLLM_ASCEND_TESTS_DSV4_TILING_SHIM_H
