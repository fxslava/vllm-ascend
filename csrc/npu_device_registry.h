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

#pragma once

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"

#include <array>
#include <atomic>
#include <cstdint>

namespace vllm_ascend {
namespace device_registry {

constexpr int32_t kMaxDevices = 64;

inline int32_t CurrentDevice()
{
    int32_t device_id = 0;
    if (c10_npu::GetDevice(&device_id) != ACL_SUCCESS) {
        device_id = static_cast<int32_t>(c10_npu::current_device());
    }
    return device_id;
}

inline int64_t QueryVectorCoreNum(int32_t device_id)
{
    int64_t aiv = 0;
    aclError ret = aclrtGetDeviceInfo(static_cast<uint32_t>(device_id), ACL_DEV_ATTR_VECTOR_CORE_NUM, &aiv);
    if (ret != ACL_SUCCESS || aiv <= 0) {
        ret = aclGetDeviceCapability(static_cast<uint32_t>(device_id), ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv);
    }
    TORCH_CHECK(ret == ACL_SUCCESS && aiv > 0, "failed to query the vector core count of device ", device_id,
                ": ret=", static_cast<int>(ret), " aiv=", aiv);
    return aiv;
}

inline int64_t VectorCoreNum(int32_t device_id)
{
    static std::array<std::atomic<int64_t>, kMaxDevices> cache{};

    TORCH_CHECK(device_id >= 0 && device_id < kMaxDevices, "device id ", device_id,
                " is outside the registry, which holds ", kMaxDevices, " devices");
    std::atomic<int64_t> &slot = cache[static_cast<size_t>(device_id)];
    const int64_t cached = slot.load(std::memory_order_relaxed);
    if (cached > 0) {
        return cached;
    }
    const int64_t aiv = QueryVectorCoreNum(device_id);
    slot.store(aiv, std::memory_order_relaxed);
    return aiv;
}

inline int64_t VectorCoreNum()
{
    return VectorCoreNum(CurrentDevice());
}

}
}
