/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file gate_c_main.cpp
 * \brief Host driver for the camodel (Gate C) run of dsv4_moe_expert.
 *
 * Unlike the CPU-interpreter harness, this launches real device object code
 * through the ACL runtime. Linked against libruntime_camodel.so it runs on a
 * machine with no NPU, and the simulator records a cycle-level trace of the
 * actual pipe behaviour -- which is what makes it, and not Gate B, the check
 * that the kernel's SetFlag/WaitFlag discipline is correct.
 *
 * Usage: dsv4_gate_c <io-dir> [hidden] [inter]
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "acl/acl.h"

namespace dsv4_gate_c {
void LaunchDsv4MoeExpert(uint32_t blockDim, void *stream, void *x, void *w1, void *w2, void *w3, void *w1Scale,
                         void *w2Scale, void *w3Scale, void *gateOut, void *upOut, void *activatedOut,
                         void *downOut, void *workspace, void *tiling);
} // namespace dsv4_gate_c

namespace {

constexpr int64_t FP4_BLOCK = 32;
constexpr int64_t FP4_PER_BYTE = 2;
constexpr float SWIGLU_LIMIT = 10.0f; // DeepSeek-V4 architectural constant

#define ACL_OK(expr)                                                                       \
    do {                                                                                   \
        const aclError dsv4Err = (expr);                                                   \
        if (dsv4Err != ACL_SUCCESS) {                                                      \
            std::fprintf(stderr, "%s:%d: %s failed with %d\n", __FILE__, __LINE__, #expr,  \
                         static_cast<int>(dsv4Err));                                       \
            return 1;                                                                      \
        }                                                                                  \
    } while (0)

bool ReadFile(const std::string &path, std::vector<uint8_t> &out)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return false;
    }
    const std::streamsize size = in.tellg();
    in.seekg(0);
    out.resize(static_cast<size_t>(size));
    return static_cast<bool>(in.read(reinterpret_cast<char *>(out.data()), size));
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <io-dir> [hidden] [inter]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const int64_t hidden = (argc > 2) ? std::stoll(argv[2]) : 256;
    const int64_t inter = (argc > 3) ? std::stoll(argv[3]) : 128;

    const size_t xBytes = static_cast<size_t>(hidden) * sizeof(uint16_t);
    const size_t gateWeightBytes = static_cast<size_t>(inter * hidden / FP4_PER_BYTE);
    const size_t downWeightBytes = static_cast<size_t>(hidden * inter / FP4_PER_BYTE);
    const size_t gateScaleBytes = static_cast<size_t>(inter * hidden / FP4_BLOCK);
    const size_t downScaleBytes = static_cast<size_t>(hidden * inter / FP4_BLOCK);
    const size_t interOutBytes = static_cast<size_t>(inter) * sizeof(uint16_t);
    const size_t hiddenOutBytes = static_cast<size_t>(hidden) * sizeof(uint16_t);

    std::vector<uint8_t> input;
    if (!ReadFile(dir + "/input.bin", input)) {
        return 1;
    }

    ACL_OK(aclInit(nullptr));
    ACL_OK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_OK(aclrtCreateStream(&stream));

    std::vector<void *> allocated;
    auto devAlloc = [&](size_t bytes, const void *host) -> void * {
        void *ptr = nullptr;
        if (aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            return nullptr;
        }
        allocated.push_back(ptr);
        if (host != nullptr) {
            if (aclrtMemcpy(ptr, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                return nullptr;
            }
        } else {
            std::vector<uint8_t> zeros(bytes, 0);
            if (aclrtMemcpy(ptr, bytes, zeros.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                return nullptr;
            }
        }
        return ptr;
    };

    size_t cursor = 0;
    auto take = [&](size_t bytes) -> const uint8_t * {
        const uint8_t *p = input.data() + cursor;
        cursor += bytes;
        return p;
    };

    void *xDev = devAlloc(xBytes, take(xBytes));
    void *w1Dev = devAlloc(gateWeightBytes, take(gateWeightBytes));
    void *w2Dev = devAlloc(downWeightBytes, take(downWeightBytes));
    void *w3Dev = devAlloc(gateWeightBytes, take(gateWeightBytes));
    void *w1sDev = devAlloc(gateScaleBytes, take(gateScaleBytes));
    void *w2sDev = devAlloc(downScaleBytes, take(downScaleBytes));
    void *w3sDev = devAlloc(gateScaleBytes, take(gateScaleBytes));

    void *gateDev = devAlloc(interOutBytes, nullptr);
    void *upDev = devAlloc(interOutBytes, nullptr);
    void *activatedDev = devAlloc(interOutBytes, nullptr);
    void *downDev = devAlloc(hiddenOutBytes, nullptr);
    void *workspaceDev = devAlloc(static_cast<size_t>(FP4_BLOCK), nullptr);

    // Mirrors op_kernel/dsv4_moe_expert_tiling_data.h field for field:
    // GET_TILING_DATA_WITH_STRUCT copies sizeof(struct) bytes out of this
    // buffer, so a short one is an out-of-bounds read.
    struct TilingLayout {
        int64_t hiddenSize;
        int64_t interSize;
        int64_t blockSize;
        float swigluLimit;
        float tilingReserved;
    } tiling{hidden, inter, FP4_BLOCK, SWIGLU_LIMIT, 0.0f};
    void *tilingDev = devAlloc(sizeof(tiling), &tiling);

    if (tilingDev == nullptr || downDev == nullptr) {
        std::fprintf(stderr, "device allocation failed\n");
        return 1;
    }

    std::printf("dsv4_moe_expert camodel run: hidden=%lld inter=%lld blockDim=1\n",
                static_cast<long long>(hidden), static_cast<long long>(inter));
    std::fflush(stdout);

    dsv4_gate_c::LaunchDsv4MoeExpert(1, stream, xDev, w1Dev, w2Dev, w3Dev, w1sDev, w2sDev, w3sDev, gateDev, upDev,
                                     activatedDev, downDev, workspaceDev, tilingDev);
    ACL_OK(aclrtSynchronizeStream(stream));

    std::vector<uint8_t> actual(interOutBytes * 3 + hiddenOutBytes);
    size_t at = 0;
    auto fetch = [&](void *dev, size_t bytes) -> bool {
        const bool ok = aclrtMemcpy(actual.data() + at, bytes, dev, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
        at += bytes;
        return ok;
    };
    bool ok = fetch(gateDev, interOutBytes);
    ok = fetch(upDev, interOutBytes) && ok;
    ok = fetch(activatedDev, interOutBytes) && ok;
    ok = fetch(downDev, hiddenOutBytes) && ok;
    if (!ok) {
        std::fprintf(stderr, "device-to-host copy failed\n");
        return 1;
    }

    std::ofstream out(dir + "/actual.bin", std::ios::binary);
    out.write(reinterpret_cast<const char *>(actual.data()), static_cast<std::streamsize>(actual.size()));
    out.close();
    std::printf("wrote %s/actual.bin (%zu bytes)\n", dir.c_str(), actual.size());

    for (void *ptr : allocated) {
        (void)aclrtFree(ptr);
    }
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
    return 0;
}
