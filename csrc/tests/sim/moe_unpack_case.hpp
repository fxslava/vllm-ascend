// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>
#include "ascend_device_context.hpp"
#include "camodel_guard.hpp"
#include "dsv4_test_oracle.hpp"
#include "moe_unpack_probe.hpp"
namespace vllm_ascend::test::dsv4 {
inline void FillProjectionInput(std::vector<uint8_t>& packed, std::vector<uint8_t>& scales, uint32_t columns) {
  for (size_t i = 0; i < packed.size(); ++i) {
    const size_t row = 2 * i / columns;
    const uint8_t sign = row % 3 == 1 ? 0x88 : (row % 3 == 2 ? 0x80 : 0);
    packed[i] = static_cast<uint8_t>(((i * 37) & 7) | (((i * 11 + 3) & 7) << 4) | sign);
    scales[i / 16] = static_cast<uint8_t>(120 + row % 5);
  }
}
struct UnpackResult {
  std::vector<float> decoded;
  std::vector<uint16_t> activated;
};
inline UnpackResult RunUnpack(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& scales, uint32_t columns,
                              uint32_t path, bool math, aclrtStream stream) {
  const uint32_t n = static_cast<uint32_t>(packed.size() * 2);
  if (n == 0 || n % 1024 || columns < 64 || columns > 512 || columns % 64 || 4096 % columns ||
      scales.size() != n / 32 || path > 3)
    throw std::invalid_argument("unpack probe requires aligned tiles and 64..512 columns");
  auto p = DeviceBuffer::FromHost(packed);
  auto s = DeviceBuffer::FromHost(scales);
  DeviceBuffer dense(n * sizeof(float)), act(n / columns * sizeof(uint16_t));
  ACL_CHECK(aclrtMemset(dense.get(), dense.size_bytes(), 0xa5, dense.size_bytes()));
  ACL_CHECK(aclrtMemset(act.get(), act.size_bytes(), 0xa5, act.size_bytes()));
  LaunchWatchdog watchdog(3600, "moe_unpack_probe");
  watchdog.Arm("launch and synchronization");
  ACL_CHECK(dsv4_unpack_probe_impl(stream, p.get(), s.get(), dense.get(), act.get(), n, columns, path, math));
  UnpackResult result;
  ACL_CHECK(aclrtSynchronizeStream(stream));
  watchdog.Disarm();
  result.decoded.resize(n);
  dense.CopyToHost(result.decoded.data(), dense.size_bytes());
  if (math) {
    result.activated.resize(n / columns);
    act.CopyToHost(result.activated.data(), act.size_bytes());
  }
  return result;
}
}  // namespace vllm_ascend::test::dsv4
