/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file sim_main.cpp
 * \brief Standalone CPU-interpreter driver for the dsv4_moe_expert kernel.
 *
 * Builds against tikicpulib (tikicpulib::Ascend950PR_9579, which cmake maps to
 * the ascend950pr_9599 C310 series) and launches the kernel through
 * AscendC::RunKernelFunctionOnCpu. There is no NPU and no aclnn op package in
 * the loop: the kernel source is compiled straight into this binary with
 * ASCENDC_CPU_DEBUG=1.
 *
 * Input layout is the concatenation written by
 * kernel_bringup/gen_golden.py gen, in kernel argument order:
 *   x, w1, w2, w3, w1_scale, w2_scale, w3_scale
 * Output layout, written to actual.bin, is likewise in kernel argument order:
 *   gate_out, up_out, activated, down_out
 *
 * Usage: sim_main <io-dir> [hidden] [inter]
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void dsv4_moe_expert(GM_ADDR x, GM_ADDR w1, GM_ADDR w2, GM_ADDR w3,
                                                      GM_ADDR w1Scale, GM_ADDR w2Scale, GM_ADDR w3Scale,
                                                      GM_ADDR gateOut, GM_ADDR upOut, GM_ADDR activatedOut,
                                                      GM_ADDR downOut, GM_ADDR workspace, GM_ADDR tiling);

namespace {

constexpr int64_t FP4_BLOCK = 32;
constexpr int64_t FP4_PER_BYTE = 2;
// DeepSeek-V4 architectural constant; must match the host tiling function and
// gen_golden.py's SWIGLU_LIMIT.
constexpr float SWIGLU_LIMIT = 10.0f;

// Mirrors op_kernel/dsv4_moe_expert_tiling_data.h field for field. The host
// tiling path is not driven here, so the buffer is built by hand -- and it has
// to match the kernel's struct exactly, because GET_TILING_DATA_WITH_STRUCT
// copies sizeof(struct) bytes out of it.
struct TilingLayout {
    int64_t hiddenSize;
    int64_t interSize;
    int64_t blockSize;
    float swigluLimit;
    float tilingReserved;
};

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

bool WriteFile(const std::string &path, const std::vector<uint8_t> &data)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        return false;
    }
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(out);
}

// GmAlloc + copy-in helper. Every device buffer is shared memory, which matters
// because RunKernelFunctionOnCpu forks a process per core.
uint8_t *DeviceBuffer(size_t bytes, const uint8_t *source = nullptr)
{
    uint8_t *ptr = static_cast<uint8_t *>(AscendC::GmAlloc(bytes));
    if (ptr == nullptr) {
        return nullptr;
    }
    if (source != nullptr) {
        std::memcpy(ptr, source, bytes);
    } else {
        std::memset(ptr, 0, bytes);
    }
    return ptr;
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

    if (hidden % (FP4_PER_BYTE * FP4_BLOCK) != 0 || inter % (FP4_PER_BYTE * FP4_BLOCK) != 0) {
        std::fprintf(stderr, "hidden/inter must be multiples of 64\n");
        return 2;
    }

    // Byte sizes, derived exactly as gen_golden.py derives them.
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
    const size_t expectedInput =
        xBytes + gateWeightBytes + downWeightBytes + gateWeightBytes + gateScaleBytes + downScaleBytes + gateScaleBytes;
    if (input.size() != expectedInput) {
        std::fprintf(stderr, "input.bin is %zu bytes, expected %zu for hidden=%lld inter=%lld\n", input.size(),
                     expectedInput, static_cast<long long>(hidden), static_cast<long long>(inter));
        return 1;
    }

    size_t cursor = 0;
    auto take = [&](size_t bytes) -> const uint8_t * {
        const uint8_t *p = input.data() + cursor;
        cursor += bytes;
        return p;
    };

    uint8_t *xGm = DeviceBuffer(xBytes, take(xBytes));
    uint8_t *w1Gm = DeviceBuffer(gateWeightBytes, take(gateWeightBytes));
    uint8_t *w2Gm = DeviceBuffer(downWeightBytes, take(downWeightBytes));
    uint8_t *w3Gm = DeviceBuffer(gateWeightBytes, take(gateWeightBytes));
    uint8_t *w1sGm = DeviceBuffer(gateScaleBytes, take(gateScaleBytes));
    uint8_t *w2sGm = DeviceBuffer(downScaleBytes, take(downScaleBytes));
    uint8_t *w3sGm = DeviceBuffer(gateScaleBytes, take(gateScaleBytes));

    uint8_t *gateGm = DeviceBuffer(interOutBytes);
    uint8_t *upGm = DeviceBuffer(interOutBytes);
    uint8_t *activatedGm = DeviceBuffer(interOutBytes);
    uint8_t *downGm = DeviceBuffer(hiddenOutBytes);
    uint8_t *workspaceGm = DeviceBuffer(FP4_BLOCK); // tiling reports 0; keep a valid pointer

    TilingLayout tiling{};
    tiling.hiddenSize = hidden;
    tiling.interSize = inter;
    tiling.blockSize = FP4_BLOCK;
    tiling.swigluLimit = SWIGLU_LIMIT;
    tiling.tilingReserved = 0.0f;
    uint8_t *tilingGm = DeviceBuffer(sizeof(tiling), reinterpret_cast<const uint8_t *>(&tiling));

    if (tilingGm == nullptr || downGm == nullptr) {
        std::fprintf(stderr, "GmAlloc failed\n");
        return 1;
    }

    std::printf("dsv4_moe_expert CPU-interpreter run: hidden=%lld inter=%lld block=%lld blockDim=1\n",
                static_cast<long long>(hidden), static_cast<long long>(inter),
                static_cast<long long>(FP4_BLOCK));
    std::fflush(stdout);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dsv4_moe_expert, 1, xGm, w1Gm, w2Gm, w3Gm, w1sGm, w2sGm, w3sGm, gateGm, upGm, activatedGm, downGm,
                workspaceGm, tilingGm);

    std::vector<uint8_t> actual;
    actual.reserve(interOutBytes * 3 + hiddenOutBytes);
    auto append = [&](const uint8_t *src, size_t bytes) {
        actual.insert(actual.end(), src, src + bytes);
    };
    append(gateGm, interOutBytes);
    append(upGm, interOutBytes);
    append(activatedGm, interOutBytes);
    append(downGm, hiddenOutBytes);

    const bool ok = WriteFile(dir + "/actual.bin", actual);
    std::printf("wrote %s/actual.bin (%zu bytes)\n", dir.c_str(), actual.size());

    for (uint8_t *p : {xGm, w1Gm, w2Gm, w3Gm, w1sGm, w2sGm, w3sGm, gateGm, upGm, activatedGm, downGm, workspaceGm,
                       tilingGm}) {
        AscendC::GmFree(p);
    }
    return ok ? 0 : 1;
}
