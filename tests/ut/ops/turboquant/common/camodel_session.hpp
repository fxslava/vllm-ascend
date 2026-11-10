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
// The CAModel / device session and a Global Memory allocation with canary bands on both sides.
//
// Every buffer a runner hands the kernel is guarded: kGuardBytes of 0xDEADBEEF in front of the
// payload and the same behind it, uploaded with the payload and checked after the launch. A
// burst that ran off the end of a page is then a named failure rather than a wrong number
// somewhere else, and a page no work item owns must still read back as canary.


#include <acl/acl.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace camodel {
constexpr std::size_t kGuardBytes = 64;
constexpr std::uint32_t kCanary = 0xDEADBEEF;

inline void check(aclError result, const char* call)
{
    if (result != ACL_SUCCESS) {
        std::fprintf(stderr, "%s failed: %d\n", call, static_cast<int>(result));
        throw std::runtime_error(call);
    }
}

struct Session {
    aclrtStream stream = nullptr;
    Session()
    {
        check(aclInit(nullptr), "aclInit");
        check(aclrtSetDevice(0), "aclrtSetDevice");
        check(aclrtCreateStream(&stream), "aclrtCreateStream");
    }
    ~Session()
    {
        if (stream) aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }
};

struct GuardedDeviceBuffer {
    std::vector<std::uint32_t> host;
    void* device = nullptr;
    std::size_t payload_bytes;
    explicit GuardedDeviceBuffer(std::size_t bytes)
        : host((bytes + 2 * kGuardBytes + 3) / 4, kCanary),
          payload_bytes(bytes)
    {
        if (bytes % 32 != 0) throw std::runtime_error("unaligned GM payload");
        check(aclrtMalloc(&device, host.size() * 4, ACL_MEM_MALLOC_NORMAL_ONLY),
              "aclrtMalloc");
        upload();
    }
    ~GuardedDeviceBuffer() { if (device) aclrtFree(device); }
    void* payload() const
    {
        return static_cast<std::uint8_t*>(device) + kGuardBytes;
    }
    void* host_payload() { return reinterpret_cast<std::uint8_t*>(host.data()) + kGuardBytes; }
    void upload()
    {
        check(aclrtMemcpy(device, host.size() * 4, host.data(), host.size() * 4,
                          ACL_MEMCPY_HOST_TO_DEVICE), "H2D guarded buffer");
    }
    void download()
    {
        check(aclrtMemcpy(host.data(), host.size() * 4, device, host.size() * 4,
                          ACL_MEMCPY_DEVICE_TO_HOST), "D2H guarded buffer");
    }
    void assert_guards() const
    {
        const std::size_t start = kGuardBytes / 4;
        const std::size_t end = (kGuardBytes + payload_bytes) / 4;
        for (std::size_t i = 0; i < start; ++i)
            if (host[i] != kCanary) throw std::runtime_error("head GM canary corrupted");
        for (std::size_t i = end; i < host.size(); ++i)
            if (host[i] != kCanary) throw std::runtime_error("tail GM canary corrupted");
    }
};
inline void require(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "PRECONDITION FAILED: %s\n", message);
        throw std::runtime_error(message);
    }
}

// A plain device allocation, for a runner that lays its own guard bands inside the payload
// rather than taking this header's. Every length is rounded to 32 bytes, because an MTE burst's
// granularity is a DataBlock and an unaligned tail would make an over-length burst land in
// whatever the allocator put next rather than in a canary.
struct Buffer {
    void* ptr = nullptr;
    std::size_t bytes = 0;

    explicit Buffer(std::size_t length) : bytes((length + 31) / 32 * 32)
    {
        require(bytes > 0, "a zero-length device allocation");
        check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
    }
    ~Buffer()
    {
        if (ptr) aclrtFree(ptr);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    template <class T>
    void upload(const std::vector<T>& host) const
    {
        const std::size_t n = host.size() * sizeof(T);
        require(n <= bytes, "device upload exceeds its allocation");
        check(aclrtMemcpy(ptr, bytes, host.data(), n, ACL_MEMCPY_HOST_TO_DEVICE), "H2D buffer");
    }
    template <class T>
    void download(std::vector<T>& host) const
    {
        const std::size_t n = host.size() * sizeof(T);
        require(n <= bytes, "device download exceeds its allocation");
        check(aclrtMemcpy(host.data(), n, ptr, n, ACL_MEMCPY_DEVICE_TO_HOST), "D2H buffer");
    }
};
}  // namespace camodel
