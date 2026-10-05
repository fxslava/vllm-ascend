// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "ascend_device_context.hpp"
#include "dsv4_metrics.hpp"
#include "dsv4_moe_expert_launch.hpp"

namespace vllm_ascend::test::dsv4 {
using ExpertStream = AscendStream;
using PinnedOutput = PinnedHostBuffer;

class DeviceExpert {
 public:
  explicit DeviceExpert(const Problem& p, aclrtStream external_stream = nullptr)
      : stream_(external_stream == nullptr),
        external_stream_(external_stream),
        hidden_(p.hidden),
        inter_(p.inter),
        x_(DeviceBuffer::FromHost(p.x)),
        w1_(DeviceBuffer::FromHost(p.w1)),
        w2_(DeviceBuffer::FromHost(p.w2)),
        w3_(DeviceBuffer::FromHost(p.w3)),
        s1_(DeviceBuffer::FromHost(p.w1_scale)),
        s2_(DeviceBuffer::FromHost(p.w2_scale)),
        s3_(DeviceBuffer::FromHost(p.w3_scale)),
        workspace_(32) {
    if (!GeometryIsAccepted(hidden_, inter_)) throw std::invalid_argument("unsafe DSV4 launch");
    const TilingBuffer tiling = p.Tiling();
    tiling_.Allocate(sizeof(tiling));
    tiling_.CopyFromHost(&tiling, sizeof(tiling));
    for (size_t i = 0; i < output_.size(); ++i) {
      output_[i].Allocate(static_cast<size_t>(i == 3 ? hidden_ : inter_) * sizeof(uint16_t));
      ACL_CHECK(aclrtMemset(output_[i].get(), output_[i].size_bytes(), 0xA5, output_[i].size_bytes()));
    }
  }
  ~DeviceExpert() { ACL_CHECK_NOTHROW(aclrtSynchronizeStream(stream())); }
  DeviceExpert(const DeviceExpert&) = delete;
  DeviceExpert& operator=(const DeviceExpert&) = delete;
  aclrtStream stream() const { return external_stream_ == nullptr ? stream_.get() : external_stream_; }
  void Enqueue() {
    dsv4_moe_expert_impl(stream(), 1, x_.get(), w1_.get(), w2_.get(), w3_.get(), s1_.get(), s2_.get(), s3_.get(),
                         output_[0].get(), output_[1].get(), output_[2].get(), output_[3].get(), workspace_.get(),
                         tiling_.get());
  }
  DeviceOutputs Read() {
    DeviceOutputs result;
    for (size_t i = 0; i < result.size(); ++i) result[i] = CopyToHostAsync<uint16_t>(output_[i], stream());
    return result;
  }
  size_t AllocatedBytes() const {
    size_t bytes = 0;
    for (const auto* b : {&x_, &w1_, &w2_, &w3_, &s1_, &s2_, &s3_, &workspace_, &tiling_})
      bytes += b->capacity_bytes() + b->alignment();
    for (const auto& b : output_) bytes += b.capacity_bytes() + b.alignment();
    return bytes;
  }

 private:
  ExpertStream stream_;                    // destroyed after buffers; destructor drains before their release
  aclrtStream external_stream_ = nullptr;  // borrowed; caller keeps it alive through expert destruction
  int64_t hidden_, inter_;
  DeviceBuffer x_, w1_, w2_, w3_, s1_, s2_, s3_, workspace_, tiling_;
  std::array<DeviceBuffer, 4> output_;
};
}  // namespace vllm_ascend::test::dsv4
