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

// libvllm_turboquant_cube.so: the TurboQuant operators and nothing else of vllm_ascend_C.
//
// Loaded with torch.ops.load_library, it registers the same torch.ops._C_ascend schemas
// as the full extension (the registrations are shared, turboquant_torch_ops.h), so code
// that calls torch.ops._C_ascend.npu_turboquant_cube_decode cannot tell which one it got.
// There is no Python module here and no pybind11: registration is static, at dlopen.

#if !defined(VLLM_ENABLE_TURBOQUANT) || !defined(VLLM_ENABLE_TURBOQUANT_CUBE)
#error "the standalone TurboQuant library is the Ascend 950 build: define VLLM_ENABLE_TURBOQUANT and VLLM_ENABLE_TURBOQUANT_CUBE"
#endif

#include "attention/turboquant/op_adapter/turboquant_torch_adpt.h"
#include "attention/turboquant/op_adapter/turboquant_torch_ops.h"
