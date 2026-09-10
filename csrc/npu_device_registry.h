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

// A per-device cache of the hardware properties a kernel launch needs to size
// its grid.  The properties are fixed for the life of the process, and
// aclGetDeviceCapability enters the CANN driver, so querying once per device
// and reading a cached word keeps a context switch off the decode path.
//
// The cache is a flat array of atomics indexed by device id: a hit is one
// relaxed load, and two threads racing on a cold slot both store the same
// value, so no stronger ordering is needed.
namespace vllm_ascend {
namespace device_registry {

// Slots in the registry.  ACL identifies devices by a small dense index, so a
// flat array indexed by id is both the fastest lookup and the whole data
// structure; 64 covers any single host this runs on.
constexpr int32_t kMaxDevices = 64;

// The device this thread is currently bound to.
//
// c10_npu::GetDevice answers from torch's own bookkeeping and lazily
// initialises the device when the thread has not touched the NPU yet, which a
// bare aclrtGetDevice would report as an error; current_device() is the
// fallback.  Resolving per thread is what makes the registry correct under
// tensor parallelism.
inline int32_t CurrentDevice()
{
    int32_t device_id = 0;
    if (c10_npu::GetDevice(&device_id) != ACL_SUCCESS) {
        device_id = static_cast<int32_t>(c10_npu::current_device());
    }
    return device_id;
}

// Ask the driver for a device's AI Vector core count.  Which of
// aclrtGetDeviceInfo and aclGetDeviceCapability answers depends on the CANN
// version and the SOC; mirrors the fallback chain in
// csrc/attention/k2q_csr/k2q_csr_torch_adpt.h.
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

// Cached AI Vector core count of `device_id`.  The driver is asked at most once
// per device per process; every later call is a relaxed atomic load.
inline int64_t VectorCoreNum(int32_t device_id)
{
    // Zero means "not yet queried"; a real count is always positive, so the
    // sentinel cannot collide with a valid answer.
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

// Cached AI Vector core count of the calling thread's device.
inline int64_t VectorCoreNum()
{
    return VectorCoreNum(CurrentDevice());
}

}  // namespace device_registry
}  // namespace vllm_ascend
