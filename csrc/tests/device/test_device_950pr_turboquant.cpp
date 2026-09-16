/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "ascend950_shapes.hpp"
#include "device_buffer.hpp"
#include "fp16.hpp"
#include "random_data.hpp"
#include "test_harness.hpp"
#include "turbo_quant_cpu.h"
#include "turboquant_launch.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace tq = turboquant_ref;
namespace tqh = turboquant_host;
namespace s950 = shapes950;

constexpr int kHeadSize = static_cast<int>(s950::kHeadDim);
constexpr int kNumHeads = static_cast<int>(s950::kNumHeads);
constexpr int kNumKvHeads = static_cast<int>(s950::kNumKvHeads);
constexpr int kBlockSize = static_cast<int>(s950::kBlockSize);
constexpr int kQueryTokens = 1;
constexpr float kAttentionScale = s950::kAttentionScale;
constexpr float kInvSqrtHeadSize = s950::kAttentionScale;

const int kDefaultContextLens[] = {64, 512, 1024, 2048};

constexpr int kMaxLevelDrift = 1;
constexpr double kMaxDifferingChannelFraction = 0.02;
constexpr double kScaleRelativeTolerance = 1e-5;

constexpr double kMinDecodeCosine = 0.999;
constexpr double kMaxDecodeRelativeL2 = 5e-3;

std::vector<int> ContextLens() {
  const char* override_value = std::getenv("ASCEND_TQ_BARE_METAL_CONTEXTS");
  if (override_value == nullptr || *override_value == '\0') {
    return std::vector<int>(std::begin(kDefaultContextLens), std::end(kDefaultContextLens));
  }
  std::vector<int> lens;
  std::istringstream stream(override_value);
  std::string field;
  while (std::getline(stream, field, ',')) {
    const long parsed = std::strtol(field.c_str(), nullptr, 10);
    if (parsed > 0) {
      lens.push_back(static_cast<int>(parsed));
    }
  }
  return lens.empty() ? std::vector<int>(std::begin(kDefaultContextLens), std::end(kDefaultContextLens)) : lens;
}

struct Scenario {
  int context_len = 0;
  int blocks_per_seq = 0;
  int num_blocks = 0;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> query;
  std::vector<int32_t> slots;
  std::vector<int32_t> table;
};

Scenario MakeScenario(int context_len, uint32_t seed, const std::vector<float>& key_override = {},
                      const std::vector<float>& value_override = {}) {
  DeterministicRandom rng(seed);
  Scenario s;
  s.context_len = context_len;
  s.blocks_per_seq = (context_len + kBlockSize - 1) / kBlockSize;
  s.num_blocks = std::max(4, s.blocks_per_seq * 4);

  const size_t kv_elems = static_cast<size_t>(context_len) * kNumKvHeads * kHeadSize;
  s.key = key_override.empty() ? rng.NormalHalfExact(kv_elems, 0.0f, 1.0f) : key_override;
  s.value = value_override.empty() ? rng.NormalHalfExact(kv_elems, 0.0f, 1.0f) : value_override;
  s.query = rng.NormalHalfExact(static_cast<size_t>(kQueryTokens) * kNumHeads * kHeadSize, 0.0f, 1.0f);

  const std::vector<int32_t> permutation = rng.Permutation(s.num_blocks);
  s.table.assign(permutation.begin(), permutation.begin() + s.blocks_per_seq);

  s.slots.resize(static_cast<size_t>(context_len));
  for (int i = 0; i < context_len; ++i) {
    const int block = s.table[static_cast<size_t>(i / kBlockSize)];
    s.slots[static_cast<size_t>(i)] = block * kBlockSize + (i % kBlockSize);
  }
  return s;
}

void ReferenceWritePath(const Scenario& s, int8_t fill, std::vector<int8_t>* key_cache,
                        std::vector<int8_t>* value_cache, std::vector<float>* scale_plane) {
  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(kHeadSize);
  key_cache->assign(tqh::PackedCacheBytes(s.num_blocks, kBlockSize, kNumKvHeads, kHeadSize), fill);
  value_cache->assign(key_cache->size(), fill);
  scale_plane->assign(tqh::ScalePlaneFloats(s.num_blocks, kBlockSize, kNumKvHeads), 0.0f);

  for (int pos = 0; pos < s.context_len; ++pos) {
    const int slot = s.slots[static_cast<size_t>(pos)];
    if (slot < 0) {
      continue;
    }
    for (int kv_head = 0; kv_head < kNumKvHeads; ++kv_head) {
      const size_t base = (static_cast<size_t>(pos) * kNumKvHeads + kv_head) * kHeadSize;
      tq::cpu_reshape_and_cache_one(s.key.data() + base, kHeadSize, signs.data(), slot, kNumKvHeads, kv_head,
                                    kv_head, key_cache->data(), scale_plane->data());
      tq::cpu_reshape_and_cache_one(s.value.data() + base, kHeadSize, signs.data(), slot, kNumKvHeads, kv_head,
                                    kNumKvHeads + kv_head, value_cache->data(), scale_plane->data());
    }
  }
}

int BinOf(const int8_t* vec, int c) {
  const int byte = static_cast<int>(vec[c / tq::kPackFactor]) + static_cast<int>(tq::kInt8Bias);
  return (c % tq::kPackFactor == 0) ? (byte % tq::kLevels) : (byte / tq::kLevels);
}

struct BinAgreement {
  int max_drift = 0;
  size_t differing = 0;
  size_t examined = 0;
  double differing_fraction() const {
    return examined > 0 ? static_cast<double>(differing) / static_cast<double>(examined) : 0.0;
  }
};

BinAgreement ComparePackedCaches(const std::vector<int8_t>& actual, const std::vector<int8_t>& expected,
                                 const std::vector<int32_t>& slots) {
  BinAgreement agreement;
  const size_t packed_stride = static_cast<size_t>(kHeadSize / tq::kPackFactor);
  for (const int32_t slot : slots) {
    if (slot < 0) {
      continue;
    }
    for (int kv_head = 0; kv_head < kNumKvHeads; ++kv_head) {
      const size_t off = (static_cast<size_t>(slot) * kNumKvHeads + kv_head) * packed_stride;
      for (int c = 0; c < kHeadSize; ++c) {
        const int drift = std::abs(BinOf(actual.data() + off, c) - BinOf(expected.data() + off, c));
        agreement.max_drift = std::max(agreement.max_drift, drift);
        if (drift > 0) {
          ++agreement.differing;
        }
        ++agreement.examined;
      }
    }
  }
  return agreement;
}

class DeviceScenario {
 public:
  DeviceScenario(const Scenario& s, aclrtStream stream, int8_t cache_fill = 0) : scenario_(s), stream_(stream) {
    key_ = DeviceBuffer::FromHost(FloatToHalf(s.key));
    value_ = DeviceBuffer::FromHost(FloatToHalf(s.value));
    query_ = DeviceBuffer::FromHost(FloatToHalf(s.query));
    slots_ = DeviceBuffer::FromHost(s.slots);
    pi_signs_ = DeviceBuffer::FromHost(tqh::PiSigns(kHeadSize));

    write_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, 1));
    decode_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, tqh::kTileRows));

    const size_t cache_bytes = tqh::PackedCacheBytes(s.num_blocks, kBlockSize, kNumKvHeads, kHeadSize);
    key_cache_ = DeviceBuffer::FromHost(std::vector<int8_t>(cache_bytes, cache_fill));
    value_cache_ = DeviceBuffer::FromHost(std::vector<int8_t>(cache_bytes, cache_fill));
    scale_plane_ = DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(s.num_blocks, kBlockSize, kNumKvHeads));

    std::vector<int32_t> block_tables;
    for (int t = 0; t < kQueryTokens; ++t) {
      block_tables.insert(block_tables.end(), s.table.begin(), s.table.end());
    }
    block_tables_ = DeviceBuffer::FromHost(block_tables);
    context_lens_ = DeviceBuffer::FromHost(std::vector<int32_t>(kQueryTokens, s.context_len));
    out_ = DeviceBuffer::Empty<Half>(static_cast<size_t>(kQueryTokens) * kNumHeads * kHeadSize);
    h16_ = DeviceBuffer::FromHost(tqh::Hadamard16Half());
    query_rot_ = DeviceBuffer::Empty<float>(static_cast<size_t>(kQueryTokens) * kNumHeads * kHeadSize);

    aiv_num_ = tqh::VectorCoreNum(&aiv_queried_);
  }

  void RunWritePath() {
    const tqh::ReshapeAndCacheGrid grid = tqh::PlanReshapeAndCache(scenario_.context_len, aiv_num_);
    turboquant_reshape_and_cache_impl(
        AscendType::FP16, stream_, grid.block_dim, key_.get(), value_.get(), key_cache_.get(), value_cache_.get(),
        scale_plane_.get(), slots_.get(), pi_signs_.get(), write_tables_.get(),
        static_cast<uint32_t>(scenario_.context_len), static_cast<uint32_t>(kNumKvHeads),
        static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize), grid.tokens_per_core,
        kInvSqrtHeadSize);
    ACL_CHECK(aclrtSynchronizeStream(stream_));
  }

  void RunDecode() {
    const tqh::PagedAttentionGrid grid =
        tqh::PlanPagedAttention(kQueryTokens, kNumHeads, kHeadSize, scenario_.blocks_per_seq, kBlockSize, aiv_num_);
    workspace_ = DeviceBuffer::Empty<float>(grid.workspace_floats);

    rotate_plan_ =
        tqh::RotateQuery(stream_, AscendType::FP16, query_.get(), pi_signs_.get(), h16_.get(), write_tables_.get(),
                         query_rot_.get(), kQueryTokens, kNumHeads, kHeadSize, aiv_num_);

    turboquant_paged_attention_impl(
        AscendType::FP16, stream_, grid.block_dim, query_rot_.get(), key_cache_.get(), value_cache_.get(),
        scale_plane_.get(), block_tables_.get(), context_lens_.get(), decode_tables_.get(), workspace_.get(),
        out_.get(), static_cast<uint32_t>(kQueryTokens), static_cast<uint32_t>(kNumHeads),
        static_cast<uint32_t>(kNumKvHeads), static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(scenario_.blocks_per_seq), static_cast<uint32_t>(grid.num_splits),
        grid.split_tasks_per_core, grid.reduce_tasks_per_core, static_cast<uint32_t>(tqh::kFusedContextLimit),
        kAttentionScale, kInvSqrtHeadSize);
    ACL_CHECK(aclrtSynchronizeStream(stream_));
  }

  std::vector<int8_t> KeyCache() const { return key_cache_.ToHost<int8_t>(); }
  std::vector<int8_t> ValueCache() const { return value_cache_.ToHost<int8_t>(); }
  std::vector<float> ScalePlane() const { return scale_plane_.ToHost<float>(); }
  std::vector<Half> RawOutput() const { return out_.ToHost<Half>(); }
  std::vector<float> Output() const { return tqh::UnrotateHeads(HalfToFloat(out_.ToHost<Half>()), kHeadSize); }

  int64_t aiv_num() const { return aiv_num_; }
  bool aiv_queried() const { return aiv_queried_; }

  std::vector<const void*> DeviceBases() const {
    return {key_.get(),        value_.get(),       query_.get(),      slots_.get(),      pi_signs_.get(),
            write_tables_.get(), decode_tables_.get(), key_cache_.get(), value_cache_.get(),
            scale_plane_.get(), block_tables_.get(), context_lens_.get(), out_.get()};
  }

 private:
  Scenario scenario_;
  aclrtStream stream_;
  DeviceBuffer key_, value_, query_, slots_, pi_signs_;
  DeviceBuffer h16_, query_rot_;
  vllm_ascend::turboquant::RotateQPlan rotate_plan_;
  DeviceBuffer write_tables_, decode_tables_;
  DeviceBuffer key_cache_, value_cache_, scale_plane_;
  DeviceBuffer block_tables_, context_lens_, workspace_, out_;
  int64_t aiv_num_ = 0;
  bool aiv_queried_ = false;
};

std::vector<float> PiPreimageOfSignVector(uint32_t seed) {
  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(kHeadSize);
  DeterministicRandom rng(seed);
  std::vector<float> vec(static_cast<size_t>(kHeadSize));
  for (int c = 0; c < kHeadSize; ++c) {
    vec[static_cast<size_t>(c)] = rng.IntInRange(0, 1) == 0 ? -1.0f : 1.0f;
  }
  tq::cpu_apply_pi(vec.data(), kHeadSize, signs.data());
  return vec;
}

class TurboQuantBareMetal : public ::testing::Test {
 protected:
  static aclrtStream Stream() { return AscendTestEnvironment::Instance().stream(); }

  static void PrintHeader(const char* label, int context_len, const DeviceScenario& device) {
    std::printf("\n[turboquant/bare-metal] %s: S=%d head=%d heads=%d kv=%d block=%d aiv=%lld%s\n", label,
                context_len, kHeadSize, kNumHeads, kNumKvHeads, kBlockSize,
                static_cast<long long>(device.aiv_num()),
                device.aiv_queried() ? "" : " (assumed, runtime declined)");
  }
};

}

TEST_F(TurboQuantBareMetal, DeviceIsPhysicalSiliconAndReportsItsTopology) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  const std::string& soc = AscendTestEnvironment::Instance().soc_name();

  bool queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&queried);

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  const bool mem_known = aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes) == ACL_SUCCESS;

  std::printf("\n[turboquant/bare-metal] soc='%s' vector_cores=%lld%s hbm=%.1f/%.1f GiB\n", soc.c_str(),
              static_cast<long long>(aiv_num), queried ? "" : " (assumed)",
              mem_known ? static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0) : 0.0,
              mem_known ? static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0) : 0.0);
  std::fflush(stdout);

  EXPECT_TRUE(queried) << "aclGetDeviceCapability(ACL_DEVICE_INFO_VECTOR_CORE_NUM) declined on a physical part; "
                          "every grid below is planned against the assumed count of "
                       << tqh::kFallbackVectorCoreNum;
  EXPECT_GT(aiv_num, 0);

  if (mem_known) {
    EXPECT_GT(free_bytes, 0u);
  }
}

TEST_F(TurboQuantBareMetal, WritePathMatchesTheCpuReferenceAcrossContexts) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  for (const int context_len : ContextLens()) {
    const Scenario scenario = MakeScenario(context_len, 0xB4C0u + static_cast<uint32_t>(context_len));
    DeviceScenario device(scenario, Stream());
    PrintHeader("write path", context_len, device);

    device.RunWritePath();

    std::vector<int8_t> expected_key;
    std::vector<int8_t> expected_value;
    std::vector<float> expected_scales;
    ReferenceWritePath(scenario, 0, &expected_key, &expected_value, &expected_scales);

    const std::vector<int8_t> actual_key = device.KeyCache();
    const std::vector<int8_t> actual_value = device.ValueCache();
    const std::vector<float> actual_scales = device.ScalePlane();

    ASSERT_EQ(actual_key.size(), expected_key.size());
    ASSERT_EQ(actual_value.size(), expected_value.size());
    ASSERT_EQ(actual_scales.size(), expected_scales.size());

    const char* plane_names[2] = {"key", "value"};
    const std::vector<int8_t>* actual_planes[2] = {&actual_key, &actual_value};
    const std::vector<int8_t>* expected_planes[2] = {&expected_key, &expected_value};
    for (int plane = 0; plane < 2; ++plane) {
      const BinAgreement agreement =
          ComparePackedCaches(*actual_planes[plane], *expected_planes[plane], scenario.slots);
      ASSERT_GT(agreement.examined, 0u) << plane_names[plane] << ": the scenario wrote no live rows";
      std::printf("  %-6s S=%4d  max bin drift %d, %zu/%zu channels differ (%.4f%%)\n", plane_names[plane],
                  context_len, agreement.max_drift, agreement.differing, agreement.examined,
                  100.0 * agreement.differing_fraction());
      EXPECT_LE(agreement.max_drift, kMaxLevelDrift)
          << plane_names[plane] << " at S=" << context_len
          << ": a channel is off by more than one 4-bit bin, which a coordinate landing either side of a "
             "decision boundary cannot explain";
      EXPECT_LE(agreement.differing_fraction(), kMaxDifferingChannelFraction)
          << plane_names[plane] << " at S=" << context_len << ": " << agreement.differing << " of "
          << agreement.examined << " channels landed on a different level";
    }

    const size_t slot_floats = tq::cpu_scale_slot_floats(kNumKvHeads);
    double worst_scale_error = 0.0;
    for (const int32_t slot : scenario.slots) {
      if (slot < 0) {
        continue;
      }
      for (int lane = 0; lane < 2 * kNumKvHeads; ++lane) {
        const size_t off = static_cast<size_t>(slot) * slot_floats + static_cast<size_t>(lane);
        const double want = expected_scales[off];
        const double denom = std::max(std::fabs(want), 1e-30);
        worst_scale_error = std::max(worst_scale_error, std::fabs(actual_scales[off] - want) / denom);
      }
    }
    std::printf("  scales S=%4d  worst relative error %.3e\n", context_len, worst_scale_error);
    EXPECT_LE(worst_scale_error, kScaleRelativeTolerance)
        << "scale plane at S=" << context_len << ": the sum-of-squares reduction disagrees with the host by more "
                                                 "than fp32 rounding allows";
    std::fflush(stdout);
  }
}

TEST_F(TurboQuantBareMetal, PackedCacheIsByteIdenticalOnRotationExactInputs) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  const int context_len = ContextLens().front();

  const std::vector<float> vector_k = PiPreimageOfSignVector(0x1111u);
  const std::vector<float> vector_v = PiPreimageOfSignVector(0x2222u);

  for (int c = 0; c < kHeadSize; ++c) {
    ASSERT_FLOAT_EQ(HalfBitsToFloat(FloatToHalfBits(vector_k[static_cast<size_t>(c)])),
                    vector_k[static_cast<size_t>(c)])
        << "channel " << c << " of the Pi-preimage is not exactly representable in fp16; the construction in "
                              "PiPreimageOfSignVector assumes 1/sqrt(head_size) is a power of two";
  }

  const size_t kv_elems = static_cast<size_t>(context_len) * kNumKvHeads * kHeadSize;
  std::vector<float> key(kv_elems);
  std::vector<float> value(kv_elems);
  for (size_t base = 0; base < kv_elems; base += kHeadSize) {
    std::copy(vector_k.begin(), vector_k.end(), key.begin() + static_cast<std::ptrdiff_t>(base));
    std::copy(vector_v.begin(), vector_v.end(), value.begin() + static_cast<std::ptrdiff_t>(base));
  }

  const Scenario scenario = MakeScenario(context_len, 0xBE17u, key, value);
  DeviceScenario device(scenario, Stream());
  PrintHeader("bit-exact write", context_len, device);

  device.RunWritePath();

  std::vector<int8_t> expected_key;
  std::vector<int8_t> expected_value;
  std::vector<float> expected_scales;
  ReferenceWritePath(scenario, 0, &expected_key, &expected_value, &expected_scales);

  const std::vector<int8_t> actual_key = device.KeyCache();
  const std::vector<int8_t> actual_value = device.ValueCache();
  const std::vector<float> actual_scales = device.ScalePlane();

  const size_t packed_stride = static_cast<size_t>(kHeadSize / tq::kPackFactor);
  const size_t slot_floats = tq::cpu_scale_slot_floats(kNumKvHeads);
  size_t compared_bytes = 0;
  for (const int32_t slot : scenario.slots) {
    ASSERT_GE(slot, 0);
    for (int kv_head = 0; kv_head < kNumKvHeads; ++kv_head) {
      const size_t off = (static_cast<size_t>(slot) * kNumKvHeads + kv_head) * packed_stride;
      ASSERT_EQ(0, std::memcmp(actual_key.data() + off, expected_key.data() + off, packed_stride))
          << "key slot " << slot << " kv_head " << kv_head << " is not byte-identical to the CPU reference";
      ASSERT_EQ(0, std::memcmp(actual_value.data() + off, expected_value.data() + off, packed_stride))
          << "value slot " << slot << " kv_head " << kv_head << " is not byte-identical to the CPU reference";
      compared_bytes += 2 * packed_stride;

      for (const int lane : {kv_head, kNumKvHeads + kv_head}) {
        const size_t scale_off = static_cast<size_t>(slot) * slot_floats + static_cast<size_t>(lane);
        EXPECT_FLOAT_EQ(actual_scales[scale_off], expected_scales[scale_off])
            << "scale lane " << lane << " of slot " << slot;
        EXPECT_NEAR(actual_scales[scale_off], 1.0f, 1e-6f)
            << "the +-1 construction predicts a scale of exactly 1.0; got " << actual_scales[scale_off];
      }
    }
  }

  std::printf("  %zu packed bytes byte-identical to the CPU reference, scale exactly 1.0\n", compared_bytes);
  std::fflush(stdout);
}

TEST_F(TurboQuantBareMetal, DecodeMatchesTheCpuReferenceAcrossContexts) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(kHeadSize);

  for (const int context_len : ContextLens()) {
    const Scenario scenario = MakeScenario(context_len, 0xD3C0u + static_cast<uint32_t>(context_len));
    DeviceScenario device(scenario, Stream());
    PrintHeader("decode", context_len, device);

    device.RunWritePath();
    device.RunDecode();

    const std::vector<int8_t> key_cache = device.KeyCache();
    const std::vector<int8_t> value_cache = device.ValueCache();
    const std::vector<float> scale_plane = device.ScalePlane();

    std::vector<float> reference(static_cast<size_t>(kNumHeads) * kHeadSize, 0.0f);
    tq::cpu_paged_attention_turboquant(scenario.query.data(), key_cache.data(), value_cache.data(),
                                       scale_plane.data(), scenario.table.data(), context_len, kNumHeads,
                                       kNumKvHeads, kHeadSize, kBlockSize, kAttentionScale, signs.data(),
                                       reference.data());

    const tq::FidelityMetrics metrics = tq::cpu_fidelity(device.Output(), reference);
    std::printf("  decode S=%4d  cos=%.6f  snr=%7.2f dB  relL2=%.6f\n", context_len, metrics.cosine_similarity,
                metrics.snr_db, metrics.relative_l2);
    std::fflush(stdout);

    EXPECT_GT(metrics.cosine_similarity, kMinDecodeCosine)
        << "S=" << context_len
        << ": the decode kernel and the CPU reference run the same arithmetic; a disagreement in direction is a "
           "kernel bug, not quantisation";
    EXPECT_LT(metrics.relative_l2, kMaxDecodeRelativeL2)
        << "S=" << context_len
        << ": the decode kernel and the CPU reference disagree in magnitude by more than fp16 output rounding "
           "allows";
  }
}

TEST_F(TurboQuantBareMetal, CacheGeometryAndAlignmentMatchTheDocumentedLayout) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  constexpr size_t kBurstBytes = 32;
  const size_t scale_slot = tq::cpu_scale_slot_floats(kNumKvHeads);
  EXPECT_EQ(scale_slot % (kBurstBytes / sizeof(float)), 0u)
      << "a token's scale slot is not a whole 32-byte burst";
  EXPECT_EQ((static_cast<size_t>(kNumKvHeads) * (kHeadSize / tq::kPackFactor)) % kBurstBytes, 0u)
      << "a token's packed bytes across all kv heads are not a whole number of 32-byte bursts, so the scatter "
         "cannot be one aligned DataCopy";
  EXPECT_EQ(static_cast<size_t>(tqh::ScaleSlotFloats(kNumKvHeads)), scale_slot)
      << "the launch shim and the CPU reference disagree about the scale slot";

  const int context_len = ContextLens().front();
  const Scenario scenario = MakeScenario(context_len, 0x1A70u);
  DeviceScenario device(scenario, Stream());
  PrintHeader("layout", context_len, device);

  const size_t packed_bytes = tqh::PackedCacheBytes(scenario.num_blocks, kBlockSize, kNumKvHeads, kHeadSize);
  const size_t scale_floats = tqh::ScalePlaneFloats(scenario.num_blocks, kBlockSize, kNumKvHeads);
  EXPECT_EQ(packed_bytes, static_cast<size_t>(scenario.num_blocks) * kBlockSize * kNumKvHeads *
                              (kHeadSize / tq::kPackFactor));
  EXPECT_EQ(scale_floats, static_cast<size_t>(scenario.num_blocks) * kBlockSize * scale_slot);
  EXPECT_EQ(device.KeyCache().size(), packed_bytes);
  EXPECT_EQ(device.ValueCache().size(), packed_bytes);
  EXPECT_EQ(device.ScalePlane().size(), scale_floats);

  for (const void* base : device.DeviceBases()) {
    EXPECT_EQ(reinterpret_cast<uintptr_t>(base) % kBurstBytes, 0u)
        << "device allocation at " << base << " is not 32-byte aligned";
  }

  const double fp16_bytes = 2.0 * static_cast<double>(context_len) * kNumKvHeads * kHeadSize * sizeof(uint16_t);
  const double tq4_bytes = 2.0 * static_cast<double>(context_len) * kNumKvHeads * (kHeadSize / tq::kPackFactor) +
                           static_cast<double>(context_len) * static_cast<double>(scale_slot) * sizeof(float);
  std::printf("  S=%d: fp16 KV %.1f KiB, 4-bit KV %.1f KiB (%.2fx smaller, scale plane is %.1f%% of it)\n",
              context_len, fp16_bytes / 1024.0, tq4_bytes / 1024.0, fp16_bytes / tq4_bytes,
              100.0 * (static_cast<double>(context_len) * static_cast<double>(scale_slot) * sizeof(float)) /
                  tq4_bytes);
  std::fflush(stdout);
  EXPECT_GT(fp16_bytes / tq4_bytes, 3.0) << "the 4-bit cache is not saving what its layout says it should";
}

TEST_F(TurboQuantBareMetal, WritePathTouchesNoByteOutsideItsSlotMapping) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  constexpr int8_t kPoison = static_cast<int8_t>(0x5A);
  const int context_len = ContextLens().front();
  Scenario scenario = MakeScenario(context_len, 0x9E11u);

  for (int i = 0; i < context_len; i += 8) {
    scenario.slots[static_cast<size_t>(i)] = -1;
  }

  DeviceScenario device(scenario, Stream(), kPoison);
  PrintHeader("bounds", context_len, device);
  device.RunWritePath();

  const std::vector<int8_t> key_cache = device.KeyCache();
  const std::vector<int8_t> value_cache = device.ValueCache();

  std::vector<bool> live_slot(static_cast<size_t>(scenario.num_blocks) * kBlockSize, false);
  for (const int32_t slot : scenario.slots) {
    if (slot >= 0) {
      live_slot[static_cast<size_t>(slot)] = true;
    }
  }

  const size_t packed_stride = static_cast<size_t>(kHeadSize / tq::kPackFactor);
  size_t poisoned_rows = 0;
  size_t written_rows = 0;
  for (size_t slot = 0; slot < live_slot.size(); ++slot) {
    for (int kv_head = 0; kv_head < kNumKvHeads; ++kv_head) {
      const size_t off = (slot * kNumKvHeads + static_cast<size_t>(kv_head)) * packed_stride;
      if (live_slot[slot]) {
        ++written_rows;
        continue;
      }
      ++poisoned_rows;
      for (size_t byte = 0; byte < packed_stride; ++byte) {
        ASSERT_EQ(key_cache[off + byte], kPoison)
            << "the key cache was written at slot " << slot << " kv_head " << kv_head << " byte " << byte
            << ", which no slot in the mapping names";
        ASSERT_EQ(value_cache[off + byte], kPoison)
            << "the value cache was written at slot " << slot << " kv_head " << kv_head << " byte " << byte
            << ", which no slot in the mapping names";
      }
    }
  }

  size_t untouched_live_rows = 0;
  for (size_t slot = 0; slot < live_slot.size(); ++slot) {
    if (!live_slot[slot]) {
      continue;
    }
    for (int kv_head = 0; kv_head < kNumKvHeads; ++kv_head) {
      const size_t off = (slot * kNumKvHeads + static_cast<size_t>(kv_head)) * packed_stride;
      bool all_poison = true;
      for (size_t byte = 0; byte < packed_stride && all_poison; ++byte) {
        all_poison = key_cache[off + byte] == kPoison;
      }
      if (all_poison) {
        ++untouched_live_rows;
      }
    }
  }

  std::printf("  %zu poisoned rows survived, %zu live rows written, %zu live rows still all-poison\n",
              poisoned_rows, written_rows, untouched_live_rows);
  std::fflush(stdout);
  EXPECT_EQ(untouched_live_rows, 0u)
      << "rows the slot mapping names came back unwritten; the scatter did not reach them";
}

TEST_F(TurboQuantBareMetal, RepeatedDecodeLaunchesAreBitIdentical) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  const std::vector<int> contexts = ContextLens();
  const int context_len = *std::max_element(contexts.begin(), contexts.end());

  const Scenario scenario = MakeScenario(context_len, 0x5EEDu);
  DeviceScenario device(scenario, Stream());
  PrintHeader("determinism", context_len, device);

  device.RunWritePath();
  device.RunDecode();
  const std::vector<Half> first = device.RawOutput();

  constexpr int kRepeats = 8;
  for (int repeat = 1; repeat < kRepeats; ++repeat) {
    device.RunDecode();
    const std::vector<Half> again = device.RawOutput();
    ASSERT_EQ(again.size(), first.size());
    for (size_t i = 0; i < first.size(); ++i) {
      ASSERT_EQ(again[i].bits, first[i].bits)
          << "decode launch " << repeat << " differs from launch 0 at element " << i
          << "; the same cache and the same query produced a different answer, which is a race or an "
             "uninitialised workspace, not rounding";
    }
  }

  std::printf("  %d decode launches bit-identical over %zu fp16 outputs\n", kRepeats, first.size());
  std::fflush(stdout);
}

}
}
