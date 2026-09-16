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
#include <cstring>
#include <string>
#include <vector>

#include "acl_check.hpp"
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

constexpr int kHeadSize = 64;
constexpr int kNumKvHeads = 2;
constexpr int kNumHeads = 4;
constexpr int kBlockSize = 16;
constexpr int kNumBlocks = 4;
constexpr int kContextLen = 32;
constexpr int kQueryTokens = 1;
constexpr float kAttentionScale = 0.125f;

constexpr int kMaxLevelDrift = 1;
constexpr double kMaxDifferingChannelFraction = 0.02;
constexpr double kScaleRelativeTolerance = 1e-5;

struct Scenario {
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> query;
  std::vector<int32_t> slots;
  std::vector<int32_t> table;
  int blocks_per_seq = 0;
};

Scenario MakeScenario(uint32_t seed) {
  DeterministicRandom rng(seed);
  Scenario s;
  const size_t kv_elems = static_cast<size_t>(kContextLen) * kNumKvHeads * kHeadSize;

  s.key = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
  s.value = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
  s.query = rng.NormalHalfExact(static_cast<size_t>(kQueryTokens) * kNumHeads * kHeadSize, 0.0f, 1.0f);

  s.blocks_per_seq = (kContextLen + kBlockSize - 1) / kBlockSize;
  const std::vector<int32_t> permutation = rng.Permutation(kNumBlocks);
  s.table.assign(permutation.begin(), permutation.begin() + s.blocks_per_seq);

  s.slots.resize(static_cast<size_t>(kContextLen));
  for (int i = 0; i < kContextLen; ++i) {
    const int block = s.table[static_cast<size_t>(i / kBlockSize)];
    s.slots[static_cast<size_t>(i)] = block * kBlockSize + (i % kBlockSize);
  }
  return s;
}

void ReferenceWritePath(const Scenario& s, std::vector<int8_t>* key_cache, std::vector<int8_t>* value_cache,
                        std::vector<float>* scale_plane) {
  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(kHeadSize);
  key_cache->assign(tqh::PackedCacheBytes(kNumBlocks, kBlockSize, kNumKvHeads, kHeadSize), 0);
  value_cache->assign(key_cache->size(), 0);
  scale_plane->assign(tqh::ScalePlaneFloats(kNumBlocks, kBlockSize, kNumKvHeads), 0.0f);

  for (int pos = 0; pos < kContextLen; ++pos) {
    const int slot = s.slots[static_cast<size_t>(pos)];
    if (slot < 0) {
      continue;
    }
    for (int kv_head = 0; kv_head < kNumKvHeads; ++kv_head) {
      const size_t base = (static_cast<size_t>(pos) * kNumKvHeads + kv_head) * kHeadSize;
      tq::cpu_reshape_and_cache_one(s.key.data() + base, kHeadSize, signs.data(), slot, kNumKvHeads, kv_head, kv_head,
                                    key_cache->data(), scale_plane->data());
      tq::cpu_reshape_and_cache_one(s.value.data() + base, kHeadSize, signs.data(), slot, kNumKvHeads, kv_head,
                                    kNumKvHeads + kv_head, value_cache->data(), scale_plane->data());
    }
  }
}

void ExpectPackedCachesAgree(const char* label, const std::vector<int8_t>& actual, const std::vector<int8_t>& expected,
                             const std::vector<int32_t>& slots) {
  ASSERT_EQ(actual.size(), expected.size()) << label << ": cache sizes differ";
  const size_t packed_stride = static_cast<size_t>(kHeadSize / tq::kPackFactor);

  int max_drift = 0;
  size_t differing = 0;
  size_t examined = 0;

  const auto bin_of = [](const int8_t* vec, int c) {
    const int byte = static_cast<int>(vec[c / tq::kPackFactor]) + static_cast<int>(tq::kInt8Bias);
    return (c % tq::kPackFactor == 0) ? (byte % tq::kLevels) : (byte / tq::kLevels);
  };

  for (const int32_t slot : slots) {
    if (slot < 0) {
      continue;
    }
    for (int kv_head = 0; kv_head < kNumKvHeads; ++kv_head) {
      const size_t off = (static_cast<size_t>(slot) * kNumKvHeads + kv_head) * packed_stride;
      for (int c = 0; c < kHeadSize; ++c) {
        const int drift = std::abs(bin_of(actual.data() + off, c) - bin_of(expected.data() + off, c));
        max_drift = std::max(max_drift, drift);
        if (drift > 0) {
          ++differing;
        }
        ++examined;
      }
    }
  }

  ASSERT_GT(examined, 0u) << label << ": the scenario wrote no live rows";
  const double differing_fraction = static_cast<double>(differing) / static_cast<double>(examined);
  std::printf("  %-12s max bin drift %d, %zu/%zu channels differ (%.3f%%)\n", label, max_drift, differing,
              examined, 100.0 * differing_fraction);

  EXPECT_LE(max_drift, kMaxLevelDrift) << label << ": a channel is off by more than one 4-bit bin, which a "
                                                  "coordinate landing either side of a decision boundary cannot "
                                                  "explain";
  EXPECT_LE(differing_fraction, kMaxDifferingChannelFraction)
      << label << ": " << differing << " of " << examined << " channels landed on a different level";
}

void ExpectScalePlanesAgree(const std::vector<float>& actual, const std::vector<float>& expected,
                            const std::vector<int32_t>& slots) {
  ASSERT_EQ(actual.size(), expected.size()) << "scale plane sizes differ";
  const size_t slot_floats = tq::cpu_scale_slot_floats(kNumKvHeads);

  for (const int32_t slot : slots) {
    if (slot < 0) {
      continue;
    }
    for (int lane = 0; lane < 2 * kNumKvHeads; ++lane) {
      const size_t off = static_cast<size_t>(slot) * slot_floats + static_cast<size_t>(lane);
      const double got = actual[off];
      const double want = expected[off];
      const double denom = std::max(std::fabs(want), 1e-30);
      EXPECT_LE(std::fabs(got - want) / denom, kScaleRelativeTolerance)
          << "scale plane slot " << slot << " lane " << lane << ": got " << got << ", want " << want;
    }
  }
}

class DeviceScenario {
 public:
  DeviceScenario(const Scenario& s, aclrtStream stream) : stream_(stream) {
    const std::vector<float> pi_signs = tqh::PiSigns(kHeadSize);

    key_ = DeviceBuffer::FromHost(FloatToHalf(s.key));
    value_ = DeviceBuffer::FromHost(FloatToHalf(s.value));
    query_ = DeviceBuffer::FromHost(FloatToHalf(s.query));
    slots_ = DeviceBuffer::FromHost(s.slots);
    pi_signs_ = DeviceBuffer::FromHost(pi_signs);

    write_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, 1));
    decode_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, tqh::kTileRows));

    key_cache_ = DeviceBuffer::Empty<int8_t>(tqh::PackedCacheBytes(kNumBlocks, kBlockSize, kNumKvHeads, kHeadSize));
    value_cache_ = DeviceBuffer::Empty<int8_t>(key_cache_.size_bytes());
    scale_plane_ = DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(kNumBlocks, kBlockSize, kNumKvHeads));

    std::vector<int32_t> block_tables;
    for (int t = 0; t < kQueryTokens; ++t) {
      block_tables.insert(block_tables.end(), s.table.begin(), s.table.end());
    }
    block_tables_ = DeviceBuffer::FromHost(block_tables);
    context_lens_ = DeviceBuffer::FromHost(std::vector<int32_t>(kQueryTokens, kContextLen));
    out_ = DeviceBuffer::Empty<Half>(static_cast<size_t>(kQueryTokens) * kNumHeads * kHeadSize);
    h16_ = DeviceBuffer::FromHost(tqh::Hadamard16Half());
    query_rot_ = DeviceBuffer::Empty<float>(static_cast<size_t>(kQueryTokens) * kNumHeads * kHeadSize);

    aiv_num_ = tqh::VectorCoreNum(&aiv_queried_);
  }

  void RunWritePath() {
    const tqh::ReshapeAndCacheGrid grid = tqh::PlanReshapeAndCache(kContextLen, aiv_num_);
    turboquant_reshape_and_cache_impl(AscendType::FP16, stream_, grid.block_dim, key_.get(), value_.get(),
                                      key_cache_.get(), value_cache_.get(), scale_plane_.get(), slots_.get(),
                                      pi_signs_.get(), write_tables_.get(), static_cast<uint32_t>(kContextLen),
                                      static_cast<uint32_t>(kNumKvHeads), static_cast<uint32_t>(kHeadSize),
                                      static_cast<uint32_t>(kBlockSize), grid.tokens_per_core, kInvSqrtHeadSize);
    ACL_CHECK(aclrtSynchronizeStream(stream_));
  }

  tqh::PagedAttentionGrid RunDecode(int blocks_per_seq, int64_t fused_context_limit = tqh::kFusedContextLimit) {
    const tqh::PagedAttentionGrid grid = tqh::PlanPagedAttention(kQueryTokens, kNumHeads, kHeadSize, blocks_per_seq,
                                                                 kBlockSize, aiv_num_, fused_context_limit);
    workspace_ = DeviceBuffer::Empty<float>(grid.workspace_floats);

    tqh::RotateQuery(stream_, AscendType::FP16, query_.get(), pi_signs_.get(), h16_.get(), write_tables_.get(),
                     query_rot_.get(), kQueryTokens, kNumHeads, kHeadSize, aiv_num_);

    turboquant_paged_attention_impl(
        AscendType::FP16, stream_, grid.block_dim, query_rot_.get(), key_cache_.get(), value_cache_.get(),
        scale_plane_.get(), block_tables_.get(), context_lens_.get(), decode_tables_.get(), workspace_.get(),
        out_.get(), static_cast<uint32_t>(kQueryTokens), static_cast<uint32_t>(kNumHeads),
        static_cast<uint32_t>(kNumKvHeads), static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(blocks_per_seq), static_cast<uint32_t>(grid.num_splits), grid.split_tasks_per_core,
        grid.reduce_tasks_per_core, static_cast<uint32_t>(fused_context_limit), kAttentionScale, kInvSqrtHeadSize);
    ACL_CHECK(aclrtSynchronizeStream(stream_));
    return grid;
  }

  std::vector<int8_t> KeyCache() const { return key_cache_.ToHost<int8_t>(); }
  std::vector<int8_t> ValueCache() const { return value_cache_.ToHost<int8_t>(); }
  std::vector<float> ScalePlane() const { return scale_plane_.ToHost<float>(); }
  std::vector<float> Output() const { return tqh::UnrotateHeads(HalfToFloat(out_.ToHost<Half>()), kHeadSize); }
  std::vector<float> RotatedOutput() const { return HalfToFloat(out_.ToHost<Half>()); }
  int64_t aiv_num() const { return aiv_num_; }
  bool aiv_queried() const { return aiv_queried_; }

 private:
  static constexpr float kInvSqrtHeadSize = 0.125f;

  aclrtStream stream_;
  DeviceBuffer key_, value_, query_, slots_, pi_signs_;
  DeviceBuffer h16_, query_rot_;
  DeviceBuffer write_tables_, decode_tables_;
  DeviceBuffer key_cache_, value_cache_, scale_plane_;
  DeviceBuffer block_tables_, context_lens_, workspace_, out_;
  int64_t aiv_num_ = 0;
  bool aiv_queried_ = false;
};

}

TEST(TurboQuantLaunchContract, ScaleSlotMatchesTheCpuReference) {
  for (int heads = 1; heads <= 16; ++heads) {
    EXPECT_EQ(static_cast<size_t>(tqh::ScaleSlotFloats(heads)), tq::cpu_scale_slot_floats(heads))
        << "num_kv_heads " << heads;
    EXPECT_EQ(tqh::ScaleSlotFloats(heads) % tqh::kFp32PerBlock, 0)
        << "num_kv_heads " << heads << ": a token's scale slot must be a whole 32-byte burst";
    EXPECT_GE(tqh::ScaleSlotFloats(heads), 2 * heads) << "num_kv_heads " << heads << ": the slot must fit K and V";
  }
}

TEST(TurboQuantLaunchContract, CodecTableWordsMatchTheLayoutContract) {
  for (int head_size : {64, 128, 256}) {
    for (int batch_rows : {1, static_cast<int>(tqh::kTileRows)}) {
      const int64_t words = tqh::CodecTableWords(head_size, batch_rows);
      EXPECT_EQ(words, 7 * head_size + 2 * head_size * batch_rows + tq::kLevels);
      EXPECT_EQ(static_cast<int64_t>(tqh::CodecTables(head_size, batch_rows).size()), words)
          << "head_size " << head_size << ", batch_rows " << batch_rows;
      EXPECT_EQ(words % tqh::kFp32PerBlock, 0)
          << "head_size " << head_size << ", batch_rows " << batch_rows;
    }
  }
}

TEST(TurboQuantLaunchContract, CodecTablesHaveTheDocumentedLayout) {
  constexpr int kD = 128;
  constexpr int kRows = static_cast<int>(tqh::kTileRows);
  const std::vector<int32_t> tables = tqh::CodecTables(kD, kRows);

  auto as_float = [](int32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };

  for (int stage = 0; stage < 3; ++stage) {
    const int stride = 1 << stage;
    const size_t sign_base = static_cast<size_t>(2 * stage) * kD;
    const size_t xor_base = sign_base + kD;
    for (int c = 0; c < kD; ++c) {
      ASSERT_FLOAT_EQ(as_float(tables[sign_base + static_cast<size_t>(c)]),
                      static_cast<float>(1 - 2 * ((c / stride) & 1)))
          << "stage " << stage << " channel " << c;
      ASSERT_EQ(tables[xor_base + static_cast<size_t>(c)], 4 * (c ^ stride)) << "stage " << stage << " channel " << c;
    }
  }

  const size_t even_base = 6 * static_cast<size_t>(kD);
  const size_t odd_base = even_base + static_cast<size_t>(kD / 2);
  for (int p = 0; p < kD / 2; ++p) {
    ASSERT_EQ(tables[even_base + static_cast<size_t>(p)], 8 * p) << "pair " << p;
    ASSERT_EQ(tables[odd_base + static_cast<size_t>(p)], 8 * p + 4) << "pair " << p;
  }

  const size_t expand_base = 7 * static_cast<size_t>(kD);
  const size_t select_base = expand_base + static_cast<size_t>(kD) * kRows;
  for (int b = 0; b < kD * kRows; ++b) {
    ASSERT_EQ(tables[expand_base + static_cast<size_t>(b)], 4 * (b / 2)) << "batch lane " << b;
    ASSERT_FLOAT_EQ(as_float(tables[select_base + static_cast<size_t>(b)]), static_cast<float>(b % 2))
        << "batch lane " << b;
  }

  const size_t centroid_base = select_base + static_cast<size_t>(kD) * kRows;
  ASSERT_EQ(tables.size(), centroid_base + static_cast<size_t>(tq::kLevels));
  for (int level = 0; level < tq::kLevels; ++level) {
    ASSERT_FLOAT_EQ(as_float(tables[centroid_base + static_cast<size_t>(level)]), tq::kLloydMaxCentroids[level])
        << "centroid " << level;
  }
  for (int level = 1; level < tq::kLevels; ++level) {
    ASSERT_LT(tq::kLloydMaxCentroids[level - 1], tq::kLloydMaxCentroids[level]) << "centroid " << level;
  }

  for (int i = 0; i < tq::kThresholdCount; ++i) {
    const float midpoint = 0.5f * (tq::kLloydMaxCentroids[i] + tq::kLloydMaxCentroids[i + 1]);
    ASSERT_NEAR(tq::kLloydMaxThresholds[i], midpoint, 1e-6f) << "threshold " << i;
  }
}

TEST(TurboQuantLaunchContract, UnrotateHeadsIsPiPerHead) {
  constexpr int64_t kHeads = 3;
  for (const int64_t head_size : {64, 128, 256}) {
    DeterministicRandom rng(0xF01Du + static_cast<uint32_t>(head_size));
    const std::vector<float> rotated = rng.NormalHalfExact(static_cast<size_t>(kHeads * head_size), 0.0f, 1.0f);
    const std::vector<float> unrotated = tqh::UnrotateHeads(rotated, head_size);

    const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(static_cast<int>(head_size));
    for (int64_t head = 0; head < kHeads; ++head) {
      std::vector<float> expected(rotated.begin() + head * head_size, rotated.begin() + (head + 1) * head_size);
      tq::cpu_apply_pi(expected.data(), static_cast<int>(head_size), signs.data());
      for (int64_t c = 0; c < head_size; ++c) {
        ASSERT_EQ(unrotated[static_cast<size_t>(head * head_size + c)], expected[static_cast<size_t>(c)])
            << "head_size " << head_size << " head " << head << " channel " << c;
      }
    }

    const std::vector<float> round_trip = tqh::UnrotateHeads(unrotated, head_size);
    for (size_t i = 0; i < rotated.size(); ++i) {
      ASSERT_NEAR(round_trip[i], rotated[i], 1e-5f) << "head_size " << head_size << " element " << i;
    }
  }
}

TEST(TurboQuantLaunchContract, PiSignsMatchTheCpuReference) {
  for (int head_size : {64, 128, 256}) {
    const std::vector<float> host = tqh::PiSigns(head_size);
    const std::vector<int8_t> reference = tq::cpu_pi_sign_vector(head_size);
    ASSERT_EQ(host.size(), reference.size()) << "head_size " << head_size;
    for (size_t i = 0; i < host.size(); ++i) {
      ASSERT_FLOAT_EQ(host[i], static_cast<float>(reference[i])) << "head_size " << head_size << " channel " << i;
      ASSERT_TRUE(host[i] == 1.0f || host[i] == -1.0f) << "head_size " << head_size << " channel " << i;
    }
  }

  const std::vector<float> pinned = tqh::PiSigns(128);
  uint32_t state = tq::kPiSeed + 128u;
  for (int i = 0; i < 8; ++i) {
    state = state * 1664525u + 1013904223u;
    const float from_lcg = ((state >> 16) & 1u) ? 1.0f : -1.0f;
    EXPECT_FLOAT_EQ(pinned[static_cast<size_t>(i)], from_lcg) << "channel " << i;
  }
}

TEST(TurboQuantLaunchContract, TilingPlansHonourTheFusedLimit) {
  constexpr int64_t kAiv = 40;

  for (int64_t tokens : {1, 7, 40, 41, 1024}) {
    const tqh::ReshapeAndCacheGrid grid = tqh::PlanReshapeAndCache(tokens, kAiv);
    EXPECT_GT(grid.tokens_per_core, 0u) << "tokens " << tokens;
    EXPECT_GE(static_cast<int64_t>(grid.block_dim) * grid.tokens_per_core, tokens) << "tokens " << tokens;
    EXPECT_LE(static_cast<int64_t>(grid.block_dim - 1) * grid.tokens_per_core, tokens - 1) << "tokens " << tokens;
    EXPECT_LE(static_cast<int64_t>(grid.block_dim), kAiv) << "tokens " << tokens;
  }

  for (int64_t block_size : {16, 128}) {
    for (int64_t tokens : {1, 4, 64}) {
      for (int64_t blocks : {0, 1, 3, 32, 64}) {
        const tqh::PagedAttentionGrid grid =
            tqh::PlanPagedAttention(tokens, kNumHeads, kHeadSize, blocks, block_size, kAiv);
        const std::string where = "tokens " + std::to_string(tokens) + " blocks " + std::to_string(blocks) +
                                  " block_size " + std::to_string(block_size);
        const bool fits = std::max<int64_t>(blocks, 1) * block_size <= tqh::kFusedContextLimit;
        EXPECT_GE(grid.num_splits, 1) << where;
        EXPECT_LE(grid.num_splits, tqh::kMaxSequenceSplits) << where;
        if (blocks > 0) {
          EXPECT_LE(grid.num_splits, blocks) << where;
        }
        if (fits) {
          EXPECT_EQ(grid.num_splits, 1) << where << ": a context that fits the fused limit is never split";
          EXPECT_EQ(grid.workspace_floats, 0u) << where << ": a fused decode needs no workspace";
          EXPECT_EQ(grid.reduce_tasks_per_core, 0u) << where;
        }
        if (grid.num_splits > 1) {
          EXPECT_EQ(grid.workspace_floats,
                    static_cast<size_t>(tokens * kNumHeads * grid.num_splits * (kHeadSize + tqh::kPartialTail)))
              << where;
          EXPECT_GE(static_cast<int64_t>(grid.reduce_tasks_per_core) * grid.block_dim, tokens * kNumHeads)
              << where << ": the in-launch reduction must reach every (token, head)";
        }
        EXPECT_GT(grid.block_dim, 0u) << where;
        EXPECT_GE(static_cast<int64_t>(grid.split_tasks_per_core) * grid.block_dim,
                  tokens * kNumHeads * grid.num_splits)
            << where;
        EXPECT_LE(static_cast<int64_t>(grid.block_dim), kAiv) << where;
      }
    }
  }

  const tqh::PagedAttentionGrid long_decode = tqh::PlanPagedAttention(1, kNumHeads, kHeadSize, 64, 128, kAiv);
  EXPECT_GT(long_decode.num_splits, 1) << "a 8192-token bound must split";
  EXPECT_GT(long_decode.workspace_floats, 0u);

  const tqh::PagedAttentionGrid forced = tqh::PlanPagedAttention(1, kNumHeads, kHeadSize, 2, 16, kAiv, 0);
  EXPECT_EQ(forced.num_splits, 2) << "a zero fused limit splits even a short sequence, capped by its blocks";

  EXPECT_EQ(tqh::PlanReshapeAndCache(0, kAiv).block_dim, 0u);
  EXPECT_EQ(tqh::PlanPagedAttention(0, kNumHeads, kHeadSize, 4, kBlockSize, kAiv).block_dim, 0u);
}

TEST(TurboQuantKernels, ReshapeAndCacheMatchesTheCpuReference) {
  REQUIRE_ASCEND_950PR();

  const Scenario scenario = MakeScenario(0x7C0FFEE1u);
  DeviceScenario device(scenario, AscendTestEnvironment::Instance().stream());
  device.RunWritePath();

  std::vector<int8_t> reference_key, reference_value;
  std::vector<float> reference_scales;
  ReferenceWritePath(scenario, &reference_key, &reference_value, &reference_scales);

  std::printf("[turboquant] write path, %d tokens x %d kv heads x %d channels\n", kContextLen, kNumKvHeads,
              kHeadSize);
  ExpectPackedCachesAgree("key cache", device.KeyCache(), reference_key, scenario.slots);
  ExpectPackedCachesAgree("value cache", device.ValueCache(), reference_value, scenario.slots);
  ExpectScalePlanesAgree(device.ScalePlane(), reference_scales, scenario.slots);
}

TEST(TurboQuantKernels, ReshapeAndCacheLeavesNegativeSlotsUntouched) {
  REQUIRE_ASCEND_950PR();

  Scenario scenario = MakeScenario(0x13572468u);
  const int kSkipped = 5;
  const int32_t skipped_slot = scenario.slots[kSkipped];
  scenario.slots[kSkipped] = -1;

  DeviceScenario device(scenario, AscendTestEnvironment::Instance().stream());
  device.RunWritePath();

  std::vector<int8_t> reference_key, reference_value;
  std::vector<float> reference_scales;
  ReferenceWritePath(scenario, &reference_key, &reference_value, &reference_scales);

  const std::vector<int8_t> device_key = device.KeyCache();
  const size_t packed_stride = static_cast<size_t>(kHeadSize / tq::kPackFactor);
  for (int kv_head = 0; kv_head < kNumKvHeads; ++kv_head) {
    const size_t off = (static_cast<size_t>(skipped_slot) * kNumKvHeads + kv_head) * packed_stride;
    for (size_t b = 0; b < packed_stride; ++b) {
      ASSERT_EQ(device_key[off + b], 0) << "slot " << skipped_slot << " kv_head " << kv_head << " byte " << b
                                        << ": a -1 slot mapping still wrote to the cache";
    }
  }

  ExpectPackedCachesAgree("key cache", device_key, reference_key, scenario.slots);
  ExpectPackedCachesAgree("value cache", device.ValueCache(), reference_value, scenario.slots);
}

TEST(TurboQuantKernels, PagedAttentionMatchesTheCpuReference) {
  REQUIRE_ASCEND_950PR();

  const Scenario scenario = MakeScenario(0x0BADC0DEu);
  DeviceScenario device(scenario, AscendTestEnvironment::Instance().stream());

  device.RunWritePath();
  device.RunDecode(scenario.blocks_per_seq);

  const std::vector<int8_t> key_cache = device.KeyCache();
  const std::vector<int8_t> value_cache = device.ValueCache();
  const std::vector<float> scale_plane = device.ScalePlane();
  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(kHeadSize);

  std::vector<float> reference(static_cast<size_t>(kQueryTokens) * kNumHeads * kHeadSize, 0.0f);
  for (int token = 0; token < kQueryTokens; ++token) {
    tq::cpu_paged_attention_turboquant(
        scenario.query.data() + static_cast<size_t>(token) * kNumHeads * kHeadSize, key_cache.data(),
        value_cache.data(), scale_plane.data(), scenario.table.data(), kContextLen, kNumHeads, kNumKvHeads,
        kHeadSize, kBlockSize, kAttentionScale, signs.data(),
        reference.data() + static_cast<size_t>(token) * kNumHeads * kHeadSize);
  }

  const std::vector<float> actual = device.Output();
  ASSERT_EQ(actual.size(), reference.size());

  const tq::FidelityMetrics metrics = tq::cpu_fidelity(actual, reference);
  std::printf("[turboquant] decode vs CPU reference: cos=%.6f snr=%.2f dB relL2=%.6f (aiv=%lld%s)\n",
              metrics.cosine_similarity, metrics.snr_db, metrics.relative_l2,
              static_cast<long long>(device.aiv_num()), device.aiv_queried() ? "" : ", assumed");

  EXPECT_GT(metrics.cosine_similarity, 0.9995) << "the decode kernel disagrees with the reference in direction, "
                                                  "which fp16 rounding of the output cannot cause";
  EXPECT_LT(metrics.relative_l2, 5e-3) << "the decode kernel disagrees with the reference in magnitude";
}

}
}
