// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#include <gtest/gtest.h>
#include <cmath>
#include "moe_unpack_case.hpp"
#include "dsv4_test_checks.hpp"
using namespace vllm_ascend::test;
TEST(MoeSimtUnpack, AllNibblesAndE8m0Specials) {
  // All 16 nibbles in both positions at every E8M0 exponent.
  std::vector<uint8_t> packed(256 * 16), scales(packed.size() / 16);
  for (size_t i = 0; i < packed.size(); ++i) {
    packed[i] = static_cast<uint8_t>((i % 16) | ((15 - i % 16) << 4));
    scales[i / 16] = static_cast<uint8_t>(i / 16);
  }
  for (uint32_t path : {0U, 1U, 2U, 3U}) {
    auto got = dsv4::RunUnpack(packed, scales, 256, path, false, AscendTestEnvironment::Instance().stream());
    for (size_t i = 0; i < got.decoded.size(); ++i) {
      const float ref = dsv4::UnpackFp4(packed.data(), i) * dsv4::E8m0ToScale(scales[i / 32]);
      if (std::isnan(ref))
        EXPECT_TRUE(std::isnan(got.decoded[i])) << path << ':' << i;
      else {
        EXPECT_EQ(dsv4::FloatToBf16Bits(ref), dsv4::FloatToBf16Bits(got.decoded[i])) << path << ':' << i;
      }
    }
  }
}
TEST(MoeSimtUnpack, ContractionAndCanonicalSwiGlu) {
  constexpr uint32_t n = 4096;
  constexpr uint32_t kRows = n / 256;
  std::vector<uint8_t> packed(n / 2), scales(n / 32, 120);
  dsv4::FillProjectionInput(packed, scales, 256);
  for (uint32_t path : {0U, 1U, 2U, 3U}) {
    auto got = dsv4::RunUnpack(packed, scales, 256, path, true, AscendTestEnvironment::Instance().stream());
    size_t failed = 0;
    for (uint32_t row = 0; row < kRows; ++row) {
      float sum = 0;
      for (uint32_t col = 0; col < 256; ++col)
        sum += dsv4::UnpackFp4(packed.data(), row * 256 + col) * dsv4::E8m0ToScale(scales[(row * 256 + col) / 32]);
      const auto ref = dsv4::FloatToBf16Bits(dsv4::SwiGluElement(sum, sum));
      const auto ulp = dsv4::Bf16UlpDistance(got.activated[row], ref);
      EXPECT_LE(ulp, 2) << path << ':' << row;
      failed += ulp > 2;
    }
    EXPECT_LE(static_cast<double>(failed) / kRows, 1e-2);
  }
}

TEST(MoeSimtUnpack, RejectsUnsafeGeometryBeforeLaunch) {
  std::vector<uint8_t> packed(512), scales(32, 127);
  EXPECT_THROW(dsv4::RunUnpack(packed, scales, 0, 1, false, nullptr), std::invalid_argument);
  EXPECT_THROW(dsv4::RunUnpack(packed, scales, 128, 4, false, nullptr), std::invalid_argument);
  scales.pop_back();
  EXPECT_THROW(dsv4::RunUnpack(packed, scales, 128, 1, false, nullptr), std::invalid_argument);
}

TEST(MoeSimtUnpack, SwitchControlsPreserveData) {
  constexpr uint32_t kControlElements = 32;
  const auto stream = AscendTestEnvironment::Instance().stream();
  DeviceBuffer output(kControlElements * sizeof(float));
  for (uint32_t simt : {0U, 1U}) {
    LaunchWatchdog watchdog(3600, "switch_control");
    watchdog.Arm("control launch");
    ACL_CHECK(dsv4_switch_control_impl(stream, output.get(), 8, simt));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    watchdog.Disarm();
    for (float value : output.ToHost<float>()) EXPECT_EQ(value, 1.0f);
  }
}
