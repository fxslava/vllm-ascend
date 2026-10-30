// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
// Standalone CANN/NPU or CAModel regression; build instructions are in the
// TurboQuant README. This runs the real writer followed by the real decoder.
#include "csrc/attention/turboquant/op_adapter/turboquant_fused_decode.h"

#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void turboquant_reshape_and_cache_bf16_impl(void*, uint32_t, const uint16_t*, const uint16_t*, int8_t*,
                                                       int8_t*, float*, const int32_t*, const float*, const int32_t*,
                                                       uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                                       float);

namespace {
constexpr uint32_t kHeads = 4;
constexpr uint32_t kChannels = 256;
constexpr uint32_t kStrip = 16;
constexpr uint32_t kGroup = 32;
constexpr uint32_t kGroups = kChannels / kGroup;
constexpr uint32_t kPartialTail = 16;

void Check(aclError error) {
  if (error != ACL_SUCCESS) throw std::runtime_error("CANN call failed: " + std::to_string(error));
}

struct DeviceBuffer {
  void* data = nullptr;
  size_t size;
  explicit DeviceBuffer(size_t bytes) : size(std::max<size_t>(32, (bytes + 31) / 32 * 32)) {
    Check(aclrtMalloc(&data, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    Check(aclrtMemset(data, size, 0, size));
  }
  ~DeviceBuffer() {
    if (data) aclrtFree(data);
  }
  void Upload(const void* host, size_t bytes) {
    Check(aclrtMemcpy(data, size, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  }
  void Download(void* host, size_t bytes) { Check(aclrtMemcpy(host, bytes, data, bytes, ACL_MEMCPY_DEVICE_TO_HOST)); }
};

uint16_t Bf16(int value) {
  const float input = static_cast<float>(value);
  uint32_t bits;
  std::memcpy(&bits, &input, sizeof(bits));
  return bits >> 16;
}

int Sign(uint32_t index) { return (index * 13 + index / 7) % 3 == 0 ? -1 : 1; }
int Hadamard(uint32_t a, uint32_t b) { return __builtin_parity(a & b) ? -1 : 1; }

float Fp16(uint16_t bits) {
  const uint32_t exponent = (bits >> 10) & 31;
  const float mantissa = static_cast<float>(bits & 1023);
  const float value = exponent == 0 ? std::ldexp(mantissa, -24) : std::ldexp(1 + mantissa / 1024, exponent - 15);
  return bits & 0x8000 ? -value : value;
}

void Run(aclrtStream stream, uint32_t blockSize, uint32_t strips, uint32_t context, uint32_t splits, int outputStage) {
  const uint32_t tokens = strips * kStrip;
  if (blockSize == 0 || blockSize % kStrip || context > tokens || splits == 0 || splits > 8)
    throw std::runtime_error("invalid block size, context or split count");
  const uint32_t pages = (tokens + blockSize - 1) / blockSize + 2;
  const uint32_t logicalPages = (tokens + blockSize - 1) / blockSize;
  const size_t elements = static_cast<size_t>(tokens) * kHeads * kChannels;
  std::vector<uint16_t> keys(elements), values(elements), tables(2 * kGroups * kGroup * kGroup);
  std::vector<int32_t> slots(tokens), blocks(logicalPages);
  for (size_t i = 0; i < elements; ++i) {
    keys[i] = Bf16(static_cast<int>((i * 7 + i / kChannels) % 11) - 5);
    values[i] = Bf16(static_cast<int>((i * 3 + i / kChannels) % 13) - 6);
  }
  for (uint32_t s = 0; s < 2; ++s)
    for (uint32_t group = 0; group < kGroups; ++group)
      for (uint32_t output = 0; output < kGroup; ++output)
        for (uint32_t input = 0; input < kGroup; ++input)
          tables[((s * kGroups + group) * kGroup + output) * kGroup + input] =
              Bf16((s ? -1 : 1) * Sign(group * kGroup + output) * Hadamard(output, input));
  for (uint32_t token = 0; token < tokens; ++token)
    slots[token] = (pages - 1 - token / blockSize) * blockSize + token % blockSize;
  for (uint32_t page = 0; page < logicalPages; ++page) blocks[page] = pages - 1 - page;
  std::vector<float> query(kHeads * kChannels);
  for (uint32_t i = 0; i < query.size(); ++i) query[i] = (static_cast<int>(i % 17) - 8) / 512.0f;
  std::vector<uint16_t> rawQuery(query.size());
  for (size_t i = 0; i < query.size(); ++i) {
    uint32_t bits;
    std::memcpy(&bits, &query[i], sizeof(bits));
    rawQuery[i] =
        query[i] == 0 ? 0 : ((bits >> 16) & 0x8000) | ((((bits >> 23) & 255) - 127 + 15) << 10) | ((bits >> 13) & 1023);
  }
  const auto transform = [](const std::vector<double>& input, bool inverse) {
    std::vector<double> result(kChannels);
    for (uint32_t out = 0; out < kChannels; ++out)
      for (uint32_t in = 0; in < kChannels; ++in) {
        const uint32_t a = inverse ? in : out, b = inverse ? out : in;
        result[out] += input[in] * Hadamard(a / kGroup, b / kGroup) * Sign((b / kGroup) * kGroup + a % kGroup) *
                       Hadamard(a % kGroup, b % kGroup) / 16;
      }
    return result;
  };
  if (outputStage >= 0) {
    for (uint32_t head = 0; head < kHeads; ++head) {
      const std::vector<double> raw(query.begin() + head * kChannels, query.begin() + (head + 1) * kChannels);
      const auto rotated = transform(raw, false);
      std::copy(rotated.begin(), rotated.end(), query.begin() + head * kChannels);
    }
  }
  const size_t cacheBytes = static_cast<size_t>(pages) * kHeads * blockSize * kChannels / 2;
  std::vector<uint8_t> keyCache(cacheBytes), valueCache(cacheBytes);
  std::vector<float> scales(2 * pages * kHeads * blockSize);
  DeviceBuffer key(keys.size() * 2), value(values.size() * 2), table(tables.size() * 2), slot(slots.size() * 4);
  DeviceBuffer kc(cacheBytes), vc(cacheBytes), sc(scales.size() * 4), bt(blocks.size() * 4), lens(32);
  DeviceBuffer q(query.size() * 4), output(query.size() * 2);
  DeviceBuffer workspace(kHeads * splits * (kChannels + kPartialTail) * sizeof(float));
  key.Upload(keys.data(), keys.size() * 2);
  value.Upload(values.data(), values.size() * 2);
  table.Upload(tables.data(), tables.size() * 2);
  slot.Upload(slots.data(), slots.size() * 4);
  bt.Upload(blocks.data(), blocks.size() * 4);
  lens.Upload(&context, sizeof(context));
  q.Upload(query.data(), query.size() * 4);
  turboquant_reshape_and_cache_bf16_impl(stream, strips, static_cast<uint16_t*>(key.data),
                                         static_cast<uint16_t*>(value.data), static_cast<int8_t*>(kc.data),
                                         static_cast<int8_t*>(vc.data), static_cast<float*>(sc.data),
                                         static_cast<int32_t*>(slot.data), nullptr, static_cast<int32_t*>(table.data),
                                         tokens, kHeads, kChannels, blockSize, pages, kStrip, 1.0f / 16);
  Check(aclrtSynchronizeStream(stream));
  kc.Download(keyCache.data(), cacheBytes);
  vc.Download(valueCache.data(), cacheBytes);
  sc.Download(scales.data(), scales.size() * 4);
  const auto begin = std::chrono::steady_clock::now();
  if (outputStage < 0) {
    vllm_ascend::turboquant_mm_fused_decode_opt_impl(4, vllm_ascend::AscendType::FP16, stream, kHeads, q.data, kc.data,
                                                     vc.data, sc.data, bt.data, lens.data, nullptr, workspace.data,
                                                     output.data, 1, kHeads, kHeads, kChannels, blockSize, pages,
                                                     logicalPages, splits, 1, splits, 1, 0, 1.0f / 16, 1.0f / 16);
    Check(aclrtSynchronizeStream(stream));
  } else {
    // Only the first three FWHT stages' shuffle tables are read by this
    // raw-query basis; the remaining codec table words stay zero.
    std::vector<int32_t> rotationTables(9 * kChannels + 16);
    std::vector<float> signs(kChannels);
    for (uint32_t i = 0; i < kChannels; ++i) signs[i] = Sign(i);
    for (uint32_t stage = 0; stage < 3; ++stage)
      for (uint32_t i = 0; i < kChannels; ++i) {
        const float sign = (i & (1u << stage)) ? -1.0f : 1.0f;
        std::memcpy(&rotationTables[2 * stage * kChannels + i], &sign, sizeof(sign));
        rotationTables[(2 * stage + 1) * kChannels + i] = (i ^ (1u << stage)) * sizeof(float);
      }
    DeviceBuffer raw(rawQuery.size() * 2), rot(rotationTables.size() * 4), pi(signs.size() * 4),
        gate(rawQuery.size() * 2);
    raw.Upload(rawQuery.data(), rawQuery.size() * 2);
    rot.Upload(rotationTables.data(), rotationTables.size() * 4);
    pi.Upload(signs.data(), signs.size() * 4);
    vllm_ascend::turboquant_mm_fused_decode_opt_raw_query_impl(
        4, vllm_ascend::AscendType::FP16, stream, kHeads, raw.data, pi.data, rot.data, nullptr, gate.data, q.data,
        kc.data, vc.data, sc.data, bt.data, lens.data, nullptr, workspace.data, output.data, 1, kHeads, kHeads,
        kChannels, blockSize, pages, logicalPages, splits, 1, splits, 1, 1, 0, static_cast<uint32_t>(outputStage), 0,
        1.0f / 16, 1.0f / 16);
    Check(aclrtSynchronizeStream(stream));
  }
  const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
  std::vector<uint16_t> actual(query.size());
  output.Download(actual.data(), actual.size() * 2);
  double error = 0, norm = 0;
  for (uint32_t head = 0; head < kHeads; ++head) {
    std::vector<double> logits(context), reference(kChannels);
    const auto reconstructed = [&](uint32_t token, uint32_t channel, bool isValue) {
      const uint32_t page = blocks[token / blockSize], row = token % blockSize;
      const uint32_t nibble =
          isValue ? (row / kStrip) * kChannels * kStrip + channel * kStrip + row % kStrip : row * kChannels + channel;
      const size_t base = (page * kHeads + head) * blockSize * kChannels / 2;
      const auto& cache = isValue ? valueCache : keyCache;
      const int code = (cache[base + nibble / 2] >> (nibble % 2 * 4)) & 15;
      const float scale = scales[((static_cast<uint32_t>(isValue) * pages + page) * kHeads + head) * blockSize + row];
      return (code < 8 ? code : code - 16) * scale;
    };
    for (uint32_t token = 0; token < context; ++token)
      for (uint32_t channel = 0; channel < kChannels; ++channel)
        logits[token] += query[head * kChannels + channel] * reconstructed(token, channel, false) / 16;
    double sum = 0;
    const double maximum = context ? *std::max_element(logits.begin(), logits.end()) : 0;
    for (double& logit : logits) {
      logit = std::exp(logit - maximum);
      sum += logit;
    }
    for (uint32_t token = 0; token < context; ++token)
      for (uint32_t channel = 0; channel < kChannels; ++channel)
        reference[channel] += logits[token] / sum * reconstructed(token, channel, true);
    if (outputStage > 0) reference = transform(reference, true);
    if (outputStage == 2)
      for (double& value : reference) value *= 0.5;
    for (uint32_t channel = 0; channel < kChannels; ++channel) {
      const double value = Fp16(actual[head * kChannels + channel]);
      if (!std::isfinite(value)) throw std::runtime_error("nonfinite decode output");
      error += std::pow(value - reference[channel], 2);
      norm += std::pow(reference[channel], 2);
    }
  }
  const double relative = std::sqrt(error / std::max(norm, 1e-12));
  std::printf("block=%u strips=%u context=%u splits=%u output_stage=%d relative_l2=%g decode_ms=%g\n", blockSize,
              strips, context, splits, outputStage, relative, elapsed);
  if (relative > 0.05) throw std::runtime_error("decode disagrees with the writer-cache reference");
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "contracts") {
    // These cases must fail before dereferencing any pointer or launching a
    // kernel. The empty case must also work without initializing the NPU.
    for (const auto geometry : {std::array<uint32_t, 3>{1, 17, 16}, {1, 16, 0}, {2, 16, 16}, {0, 16, 16}, {1, 16, 8}}) {
      bool rejected = false;
      try {
        turboquant_reshape_and_cache_bf16_impl(nullptr, geometry[0], nullptr, nullptr, nullptr, nullptr, nullptr,
                                               nullptr, nullptr, nullptr, geometry[1], kHeads, kChannels, kStrip, 1,
                                               geometry[2], 1.0f / 16);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      if (!rejected) throw std::runtime_error("invalid ingestion geometry was accepted");
    }
    turboquant_reshape_and_cache_bf16_impl(nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                           nullptr, 0, kHeads, kChannels, kStrip, 1, kStrip, 1.0f / 16);
    std::puts("PASS ingestion launch geometry and empty context");
    return 0;
  }
  if (argc != 5 && argc != 6) return 2;
  Check(aclInit(nullptr));
  Check(aclrtSetDevice(0));
  aclrtStream stream = nullptr;
  Check(aclrtCreateStream(&stream));
  Run(stream, std::stoul(argv[1]), std::stoul(argv[2]), std::stoul(argv[3]), std::stoul(argv[4]),
      argc == 6 ? std::stoi(argv[5]) : -1);
  Check(aclrtDestroyStream(stream));
  Check(aclrtResetDevice(0));
  Check(aclFinalize());
}
