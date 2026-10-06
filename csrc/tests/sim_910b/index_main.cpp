// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#include <acl/acl.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>

extern "C" void dsv4_gather_index_probe_launch(void* stream, void* input, void* output);

#define ACL_CHECK(call) do { \
    const auto error = (call); \
    if (error != ACL_SUCCESS) { \
        std::fprintf(stderr, "%s failed: %d\n", #call, static_cast<int>(error)); \
        return 2; \
    } \
} while (0)

int main()
{
    constexpr size_t SOURCE_COUNT = 2048;
    constexpr size_t OUTPUT_COUNT = 1520;
    std::array<float, SOURCE_COUNT> input{};
    std::array<uint32_t, OUTPUT_COUNT> output{};
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) + 0.25f;
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    void* deviceInput = nullptr;
    void* deviceOutput = nullptr;
    ACL_CHECK(aclrtMalloc(&deviceInput, sizeof(input), ACL_MEM_MALLOC_NORMAL_ONLY));
    ACL_CHECK(aclrtMalloc(&deviceOutput, sizeof(output), ACL_MEM_MALLOC_NORMAL_ONLY));
    ACL_CHECK(aclrtMemcpy(deviceInput, sizeof(input), input.data(), sizeof(input), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemset(deviceOutput, sizeof(output), 0xff, sizeof(output)));
    dsv4_gather_index_probe_launch(stream, deviceInput, deviceOutput);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(output.data(), sizeof(output), deviceOutput, sizeof(output), ACL_MEMCPY_DEVICE_TO_HOST));
    size_t base = 0;
    unsigned mismatches = 0;
    unsigned specials = 0;
    const auto check = [&](uint32_t count, uint32_t strideBytes, uint32_t oddBytes) {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t expectedOffset = strideBytes * i + oddBytes;
            const float expected = input[expectedOffset / sizeof(float)];
            uint32_t expectedBits;
            std::memcpy(&expectedBits, &expected, sizeof(expectedBits));
            float actual;
            std::memcpy(&actual, &output[base + count + i], sizeof(actual));
            mismatches += output[base + i] != expectedOffset;
            mismatches += output[base + count + i] != expectedBits;
            specials += !std::isfinite(actual);
        }
        base += 2 * count;
    };
    for (uint32_t count = 8; count <= 256; count *= 2) {
        check(count, 32, 0);
    }
    check(256, 2 * sizeof(float), sizeof(float));
    std::printf("Gather indices (8..256): Mismatch=%u Specials=%u\n", mismatches, specials);
    ACL_CHECK(aclrtFree(deviceInput));
    ACL_CHECK(aclrtFree(deviceOutput));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return mismatches == 0 && specials == 0 ? 0 : 1;
}
