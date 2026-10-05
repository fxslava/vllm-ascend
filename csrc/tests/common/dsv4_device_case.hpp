// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "device_buffer.hpp"
#include "dsv4_moe_expert_launch.hpp"

namespace vllm_ascend::test::dsv4 {
constexpr int64_t kProductionHidden = 7168;
constexpr int64_t kProductionInter = 2048;
constexpr int64_t kDeviceMaxUlp = 2;
constexpr double kDeviceMaxRate = 1e-2;
using DeviceOutputs = std::array<std::vector<uint16_t>, 4>;

// Integer-only generator: identical data across host standard libraries.
inline Problem MakeDeviceProblem(int64_t hidden, int64_t inter, uint32_t seed) {
  if (!GeometryIsAccepted(hidden, inter)) throw std::invalid_argument("unsupported DSV4 geometry");
  uint32_t state = seed | 1u;
  auto next = [&state]() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  };
  Problem p;
  p.hidden = hidden;
  p.inter = inter;
  p.x.resize(static_cast<size_t>(hidden));
  for (auto& v : p.x) v = FloatToBf16Bits((static_cast<int>(next() % 513) - 256) / 128.0f);
  for (auto* w : {&p.w1, &p.w2, &p.w3}) {
    w->resize(static_cast<size_t>(hidden * inter / kFp4PerByte));
    for (auto& b : *w) b = static_cast<uint8_t>(next());
    for (size_t j = 0; j < w->size() && j < 256; ++j) (*w)[j] = static_cast<uint8_t>(j);
  }
  for (auto* s : {&p.w1_scale, &p.w2_scale, &p.w3_scale}) {
    s->resize(static_cast<size_t>(hidden * inter / kFp4Block));
    for (auto& b : *s) b = static_cast<uint8_t>(120 + next() % 5);
  }
  return p;
}

inline DeviceOutputs Golden(const Problem& p) {
  auto g = ReferenceExpert(p.View(), p.hidden, p.inter);
  return {std::move(g.gate_out), std::move(g.up_out), std::move(g.activated), std::move(g.down_out)};
}

class ExpertStream {
 public:
  ExpertStream() { ACL_CHECK(aclrtCreateStream(&stream_)); }
  ~ExpertStream() {
    ACL_CHECK_NOTHROW(aclrtSynchronizeStream(stream_));
    ACL_CHECK_NOTHROW(aclrtDestroyStream(stream_));
  }
  ExpertStream(const ExpertStream&) = delete;
  ExpertStream& operator=(const ExpertStream&) = delete;
  aclrtStream get() const { return stream_; }

 private:
  aclrtStream stream_ = nullptr;
};

class PinnedOutput {
 public:
  explicit PinnedOutput(size_t bytes) { ACL_CHECK(aclrtMallocHost(&data_, bytes)); }
  ~PinnedOutput() { ACL_CHECK_NOTHROW(aclrtFreeHost(data_)); }
  PinnedOutput(const PinnedOutput&) = delete;
  PinnedOutput& operator=(const PinnedOutput&) = delete;
  void* get() const { return data_; }

 private:
  void* data_ = nullptr;
};

class DeviceExpert {
 public:
  explicit DeviceExpert(const Problem& p)
      : hidden_(p.hidden),
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
  ~DeviceExpert() { ACL_CHECK_NOTHROW(aclrtSynchronizeStream(stream_.get())); }
  DeviceExpert(const DeviceExpert&) = delete;
  DeviceExpert& operator=(const DeviceExpert&) = delete;
  aclrtStream stream() const { return stream_.get(); }
  void Enqueue() {
    dsv4_moe_expert_impl(stream(), 1, x_.get(), w1_.get(), w2_.get(), w3_.get(), s1_.get(), s2_.get(), s3_.get(),
                         output_[0].get(), output_[1].get(), output_[2].get(), output_[3].get(), workspace_.get(),
                         tiling_.get());
  }
  DeviceOutputs Read() {
    DeviceOutputs result;
    for (size_t i = 0; i < result.size(); ++i) {
      const size_t bytes = output_[i].size_bytes();
      PinnedOutput pinned(bytes);
      ACL_CHECK(aclrtMemcpyAsync(pinned.get(), bytes, output_[i].get(), bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream()));
      ACL_CHECK(aclrtSynchronizeStream(stream()));
      result[i].resize(bytes / sizeof(uint16_t));
      std::memcpy(result[i].data(), pinned.get(), bytes);
    }
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
  ExpertStream stream_;  // destroyed after buffers; destructor drains before their release
  int64_t hidden_, inter_;
  DeviceBuffer x_, w1_, w2_, w3_, s1_, s2_, s3_, workspace_, tiling_;
  std::array<DeviceBuffer, 4> output_;
};
}  // namespace vllm_ascend::test::dsv4
