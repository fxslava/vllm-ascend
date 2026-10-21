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

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_LAUNCH_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_LAUNCH_H

#include <cstdint>

namespace vllm_ascend {
namespace turboquant {

// How a host pointer reaches a kernel's GM_ADDR parameter.
//
// Under the ASC language toolchain both halves of a kernel translation unit go
// through bisheng, so a <<<>>> launch is type-checked against the kernel's own
// GM_ADDR -- that is, __gm__ uint8_t * -- parameters. The legacy
// ascendc_library path interposed a generated host stub taking void * and never
// checked this, so the launchers below were free to forward void * directly.
//
// bisheng accepts exactly one spelling here. An unqualified uint8_t * is
// promoted into the __gm__ address space at the launch, while a void * is
// rejected outright and a reinterpret_cast into __gm__ is rejected as an
// illegal address-space cast. So the launchers keep the plain void * signatures
// that the Torch adapter -- built by the host compiler, which does not know
// __gm__ -- has to declare, and narrow to uint8_t * here at the call.
inline uint8_t *AsBytes(void *ptr)
{
    return static_cast<uint8_t *>(ptr);
}

}
}

#endif
