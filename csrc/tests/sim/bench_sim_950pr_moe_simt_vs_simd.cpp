// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#include <iostream>
#include <string>
#include "moe_unpack_case.hpp"
int main(int argc, char** argv) {
  using namespace vllm_ascend::test;
  try {
    if (argc == 3 && (std::string(argv[1]) == "switch-simd" || std::string(argv[1]) == "switch-simt")) {
      const unsigned long count = std::stoul(argv[2]);
      if (count == 0 || count > 64) throw std::invalid_argument("control iterations must be 1..64");
      AscendDeviceContext device;
      AscendStream stream;
      DeviceBuffer output(32 * sizeof(float));
      LaunchWatchdog watchdog(3600, "switch_control");
      watchdog.Arm("control launch");
      ACL_CHECK(dsv4_switch_control_impl(stream.get(), output.get(), static_cast<uint32_t>(count),
                                         std::string(argv[1]) == "switch-simt"));
      ACL_CHECK(aclrtSynchronizeStream(stream.get()));
      watchdog.Disarm();
      for (float value : output.ToHost<float>())
        if (value != 1.0f) throw std::runtime_error("switch control mismatch");
      std::cout << "PASS control=" << argv[1] << " iterations=" << count << "\n";
      return 0;
    }
    if (argc != 4 && argc != 5)
      throw std::invalid_argument("usage: bench path(simd|simt|simt-cg|simt-lut) hidden(256|512) inter [unpack]");
    const std::string name(argv[1]);
    const bool math = argc == 4;
    if (argc == 5 && std::string(argv[4]) != "unpack") throw std::invalid_argument("unknown stage");
    if (name != "simd" && name != "simt" && name != "simt-cg" && name != "simt-lut")
      throw std::invalid_argument("unknown path");
    const auto hidden = static_cast<uint32_t>(std::stoul(argv[2]));
    const auto inter = static_cast<uint32_t>(std::stoul(argv[3]));
    if (inter == 0 || inter > 2048 || (hidden != 256 && hidden != 512))
      throw std::invalid_argument("unsupported geometry");
    AscendDeviceContext device;
    AscendStream stream;
    std::vector<uint8_t> packed(hidden * inter / 2), scales(hidden * inter / 32, 120);
    dsv4::FillProjectionInput(packed, scales, hidden);
    auto got =
        dsv4::RunUnpack(packed, scales, hidden, name == "simd" ? 0 : (name == "simt" ? 1 : (name == "simt-cg" ? 2 : 3)),
                        math, stream.get());
    for (size_t i = 0; i < got.decoded.size(); ++i) {
      float ref = dsv4::UnpackFp4(packed.data(), i) * dsv4::E8m0ToScale(scales[i / 32]);
      if (got.decoded[i] != ref) throw std::runtime_error("decoded mismatch");
    }
    for (uint32_t row = 0; math && row < inter; ++row) {
      float sum = 0;
      for (uint32_t col = 0; col < hidden; ++col) sum += got.decoded[row * hidden + col];
      uint16_t ref = dsv4::FloatToBf16Bits(dsv4::SwiGluElement(sum, sum));
      // Ordered BF16 distance for signed values, including signed zero.
      auto order = [](uint16_t x) { return x & 0x8000 ? 0x8000 - (x & 0x7fff) : 0x8000 + x; };
      if (std::abs(order(ref) - order(got.activated[row])) > 2) throw std::runtime_error("activation mismatch");
    }
    std::cout << "PASS path=" << name << " elements=" << hidden * inter
              << " modeled_cycles=read_npusim_summary (wall_time_is_not_device_time)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
