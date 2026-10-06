// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#include <acl/acl.h>
#include <array>
#include <cstdio>
#include <vector>

#include "dsv4_moe_expert_launch.hpp"
#include "dsv4_metrics.hpp"

#define ACL_CHECK(call) do { \
    const auto error = (call); \
    if (error != ACL_SUCCESS) { \
        std::fprintf(stderr, "%s failed: %d\n", #call, static_cast<int>(error)); \
        return 2; \
    } \
} while (0)

int main()
{
    namespace dsv4 = vllm_ascend::test::dsv4;
    constexpr int64_t HIDDEN = 64;
    constexpr int64_t INTER = 64;
    constexpr uint32_t SEED = 704;
    const auto problem = dsv4::MakeDeviceProblem(HIDDEN, INTER, SEED);
    const auto tiling = problem.Tiling();
    const auto golden = dsv4::Golden(problem);
    const std::array<const void*, 7> inputs = {
        problem.x.data(), problem.w1.data(), problem.w2.data(), problem.w3.data(),
        problem.w1_scale.data(), problem.w2_scale.data(), problem.w3_scale.data()
    };
    const std::array<size_t, 13> sizes = {
        problem.x.size() * sizeof(uint16_t), problem.w1.size(), problem.w2.size(), problem.w3.size(),
        problem.w1_scale.size(), problem.w2_scale.size(), problem.w3_scale.size(),
        INTER * sizeof(uint16_t), INTER * sizeof(uint16_t), INTER * sizeof(uint16_t),
        HIDDEN * sizeof(uint16_t), 32, sizeof(tiling)
    };
    std::array<void*, 13> buffers{};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    for (size_t i = 0; i < buffers.size(); ++i) {
        ACL_CHECK(aclrtMalloc(&buffers[i], sizes[i], ACL_MEM_MALLOC_NORMAL_ONLY));
        ACL_CHECK(aclrtMemset(buffers[i], sizes[i], 0xff, sizes[i]));
        if (i < inputs.size()) {
            ACL_CHECK(aclrtMemcpy(buffers[i], sizes[i], inputs[i], sizes[i], ACL_MEMCPY_HOST_TO_DEVICE));
        }
    }
    ACL_CHECK(aclrtMemcpy(buffers[12], sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE));
    vllm_ascend::dsv4_moe_expert_impl(stream, 1, buffers[0], buffers[1], buffers[2], buffers[3],
        buffers[4], buffers[5], buffers[6], buffers[7], buffers[8], buffers[9], buffers[10],
        buffers[11], buffers[12]);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    constexpr const char* NAMES[] = {"gate", "up", "activated", "down"};
    unsigned failures = 0;
    for (size_t k = 0; k < golden.size(); ++k) {
        std::vector<uint16_t> actual(golden[k].size());
        ACL_CHECK(aclrtMemcpy(actual.data(), sizes[7 + k], buffers[7 + k], sizes[7 + k],
                             ACL_MEMCPY_DEVICE_TO_HOST));
        const auto metrics = dsv4::CompareBf16(actual, golden[k]);
        size_t specials = 0;
        for (auto bits : actual) {
            specials += !std::isfinite(dsv4::Bf16BitsToFloat(bits));
        }
        std::printf("%s FpDiff=%lld BF16_ULP Mismatch=%zu Specials=%zu SpecialMismatch=%zu\n",
            NAMES[k], static_cast<long long>(metrics.fp_diff), metrics.mismatch, specials,
            metrics.special_mismatch);
        failures += metrics.fp_diff > dsv4::kDeviceMaxUlp || metrics.mismatch != 0 || specials != 0;
    }
    for (auto buffer : buffers) {
        ACL_CHECK(aclrtFree(buffer));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return failures == 0 ? 0 : 1;
}
