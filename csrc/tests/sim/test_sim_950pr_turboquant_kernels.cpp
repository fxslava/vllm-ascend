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

// The TurboQuant 4-bit KV-cache kernels, driven straight through the CANN
// runtime and checked against the CPU reference in reference/turbo_quant_cpu.h.
//
// Two things are under test and they are quite different in kind:
//
//   * the launch contract - the scale-plane geometry, the codec table image and
//     the grid arithmetic that turboquant_torch_adpt.h computes on the host.
//     These are pure functions of the shapes, they are duplicated in five
//     places (the Ascend C kernel, the torch adapter, the Python backend, the
//     CPU reference and common/turboquant_launch.hpp), and every one of those
//     copies can drift independently. The cases below run with no device at all.
//
//   * the kernels themselves - the split/combine decode pipeline and the write
//     path, run on hardware and compared element by element against the CPU
//     reference. These gate on REQUIRE_ASCEND_950PR.
//
// WHY THE CPU REFERENCE IS THE ORACLE. turbo_quant_cpu.h mirrors
// turboquant_codec_950.h instruction for instruction - the same LCG for the Pi
// diagonal, the same mid-rise grid, the same -128 byte bias, the same
// rotated-basis decode. It is not an independent reimplementation and does not
// pretend to be. What it establishes is that the kernel does on the device what
// the reference does on the host, which is the property that lets the host-only
// fidelity report in test_host_turboquant_fidelity.cpp stand in for a hardware run.
// The fidelity of the codec *itself* against exact attention is measured
// separately, in test_sim_950pr_turboquant_decode.cpp.
//
// ON EXACTNESS. The reconstruction check is exact-to-one-level rather than
// byte-identical, even though the measurement is in fact byte-identical: on the
// Ascend950PR camodel all 4096 channels land on the same 4-bit code as the
// reference, and the printed drift is 0.000. The butterfly pairing is the same
// on both sides, so that is what should happen. The tolerance is there because
// a channel whose rotated value sits exactly on a rounding boundary can be
// pushed either way by a single ulp of difference in the fp32 accumulation
// order, and asserting bit-identical packing would make the suite fail on a
// value nobody chose - most likely on silicon rather than on the simulator. The
// assertions are therefore: no channel off by more than one level, and the
// fraction of channels that differ at all is small. Both are real bounds; a
// genuine bug does not fit through either, as the two defects this file caught
// on its first run showed - both took the drift straight to 15.000, the full
// range.

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

// Shapes for the device cases. Deliberately small: this file runs on the
// camodel simulator as well as on silicon, and every launch there is minutes
// rather than microseconds. head_size 64 is the bottom of the supported range
// and the cheapest Walsh-Hadamard; block_size 16 is one kTileRows tile, so the
// decode kernel's tiling is exercised without a long context.
constexpr int kHeadSize = 64;
constexpr int kNumKvHeads = 2;
constexpr int kNumHeads = 4;  // group of 2, so the GQA head->kv_head mapping is live
constexpr int kBlockSize = 16;
constexpr int kNumBlocks = 4;
constexpr int kContextLen = 32;  // two blocks
constexpr int kQueryTokens = 1;  // decode
constexpr float kAttentionScale = 0.125f;  // 1 / sqrt(64)

// Bounds for the reconstruction comparison; see "ON EXACTNESS" above.
// Bins, not reconstructed values. The device sums the squares for its RMS
// scale in a tree and the host sums them serially, so the two scales differ in
// the last bits and a coordinate sitting on a decision boundary can fall either
// side of it. One bin is what that can cost; two is a real disagreement.
constexpr int kMaxLevelDrift = 1;
constexpr double kMaxDifferingChannelFraction = 0.02;
// The scale is one fp32 division of a reduced absmax. A relative difference
// larger than this means the reduction itself disagreed, not that rounding did.
constexpr double kScaleRelativeTolerance = 1e-5;

// One synthetic decode step: the context that gets written into the cache, the
// query that reads it back, and the paging that connects them.
struct Scenario {
  std::vector<float> key;         // [context_len, num_kv_heads, head_size]
  std::vector<float> value;       // same shape as key
  std::vector<float> query;       // [query_tokens, num_heads, head_size]
  std::vector<int32_t> slots;     // [context_len], flat (block, offset) row index
  std::vector<int32_t> table;     // [max_blocks_per_seq] physical block ids
  int blocks_per_seq = 0;
};

Scenario MakeScenario(uint32_t seed) {
  DeterministicRandom rng(seed);
  Scenario s;
  const size_t kv_elems = static_cast<size_t>(kContextLen) * kNumKvHeads * kHeadSize;

  // fp16-exact inputs: the device is handed these bit patterns, so the only
  // difference the comparison can see is the arithmetic, never the input.
  s.key = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
  s.value = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
  s.query = rng.NormalHalfExact(static_cast<size_t>(kQueryTokens) * kNumHeads * kHeadSize, 0.0f, 1.0f);

  // A scattered block table, so a bug that assumes the cache is sequential in
  // the sequence rather than paged shows up.
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

// The CPU reference's version of the write path over a whole scenario.
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

// Compares two packed caches by the bins they select rather than by their
// bytes. `label` names the plane in the failure message.
//
// The comparison is in bin indices, not in reconstructed values: the Lloyd-Max
// levels are unevenly spaced, so one bin is worth anywhere between 0.257 and
// 0.664, and a value-space bound tight enough to catch a two-bin slip at the
// centre of the table would reject a legitimate one-bin tie-break at its edge.
// The indices are exact, so the bound can be too.
//
// Only the rows the scenario actually wrote are examined: everything else in
// both caches is the zero fill, which quantises to a single bin and would
// drown a real difference in noise.
void ExpectPackedCachesAgree(const char* label, const std::vector<int8_t>& actual, const std::vector<int8_t>& expected,
                             const std::vector<int32_t>& slots) {
  ASSERT_EQ(actual.size(), expected.size()) << label << ": cache sizes differ";
  const size_t packed_stride = static_cast<size_t>(kHeadSize / tq::kPackFactor);

  int max_drift = 0;
  size_t differing = 0;
  size_t examined = 0;

  // Bin index of channel c of a packed vector: the low nibble for an even
  // channel, the high nibble for an odd one, with the -128 store bias undone.
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

// --- device driver ----------------------------------------------------------

// Everything the two kernels need on the device for one scenario, allocated
// once and reused by both launches. The cache and the scale plane are the
// pieces the write path fills and the decode path reads, so they deliberately
// outlive a single call.
class DeviceScenario {
 public:
  DeviceScenario(const Scenario& s, aclrtStream stream) : stream_(stream) {
    const std::vector<float> pi_signs = tqh::PiSigns(kHeadSize);

    key_ = DeviceBuffer::FromHost(FloatToHalf(s.key));
    value_ = DeviceBuffer::FromHost(FloatToHalf(s.value));
    query_ = DeviceBuffer::FromHost(FloatToHalf(s.query));
    slots_ = DeviceBuffer::FromHost(s.slots);
    pi_signs_ = DeviceBuffer::FromHost(pi_signs);

    // The write path expands one vector at a time; the decode path expands a
    // kTileRows tile. Their table images differ in the trailing expandOffset_ /
    // oddSelect_ sections and are not interchangeable.
    write_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, 1));
    decode_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, tqh::kTileRows));

    key_cache_ = DeviceBuffer::Empty<int8_t>(tqh::PackedCacheBytes(kNumBlocks, kBlockSize, kNumKvHeads, kHeadSize));
    value_cache_ = DeviceBuffer::Empty<int8_t>(key_cache_.size_bytes());
    scale_plane_ = DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(kNumBlocks, kBlockSize, kNumKvHeads));

    // One block-table row per query token, all reading the same context.
    std::vector<int32_t> block_tables;
    for (int t = 0; t < kQueryTokens; ++t) {
      block_tables.insert(block_tables.end(), s.table.begin(), s.table.end());
    }
    block_tables_ = DeviceBuffer::FromHost(block_tables);
    context_lens_ = DeviceBuffer::FromHost(std::vector<int32_t>(kQueryTokens, kContextLen));
    out_ = DeviceBuffer::Empty<Half>(static_cast<size_t>(kQueryTokens) * kNumHeads * kHeadSize);

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

  void RunDecode(int blocks_per_seq) {
    const tqh::PagedAttentionGrid grid =
        tqh::PlanPagedAttention(kQueryTokens, kNumHeads, kHeadSize, blocks_per_seq, aiv_num_);
    // Sized here rather than in the constructor: the split count, and so the
    // workspace, is a function of the block table this decode reads.
    workspace_ = DeviceBuffer::Empty<float>(grid.workspace_floats);

    turboquant_paged_attention_impl(
        AscendType::FP16, stream_, grid.split_block_dim, grid.combine_block_dim, query_.get(), key_cache_.get(),
        value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(), pi_signs_.get(),
        decode_tables_.get(), workspace_.get(), out_.get(), static_cast<uint32_t>(kQueryTokens),
        static_cast<uint32_t>(kNumHeads), static_cast<uint32_t>(kNumKvHeads), static_cast<uint32_t>(kHeadSize),
        static_cast<uint32_t>(kBlockSize), static_cast<uint32_t>(blocks_per_seq),
        static_cast<uint32_t>(grid.num_splits), grid.split_tasks_per_core, grid.combine_tasks_per_core,
        kAttentionScale, kInvSqrtHeadSize);
    ACL_CHECK(aclrtSynchronizeStream(stream_));
  }

  std::vector<int8_t> KeyCache() const { return key_cache_.ToHost<int8_t>(); }
  std::vector<int8_t> ValueCache() const { return value_cache_.ToHost<int8_t>(); }
  std::vector<float> ScalePlane() const { return scale_plane_.ToHost<float>(); }
  std::vector<float> Output() const { return HalfToFloat(out_.ToHost<Half>()); }
  int64_t aiv_num() const { return aiv_num_; }
  bool aiv_queried() const { return aiv_queried_; }

 private:
  static constexpr float kInvSqrtHeadSize = 0.125f;  // 1 / sqrt(64)

  aclrtStream stream_;
  DeviceBuffer key_, value_, query_, slots_, pi_signs_;
  DeviceBuffer write_tables_, decode_tables_;
  DeviceBuffer key_cache_, value_cache_, scale_plane_;
  DeviceBuffer block_tables_, context_lens_, workspace_, out_;
  int64_t aiv_num_ = 0;
  bool aiv_queried_ = false;
};

}  // namespace

// ---------------------------------------------------------------------------
// Launch contract - no device needed
// ---------------------------------------------------------------------------

TEST(TurboQuantLaunchContract, ScaleSlotMatchesTheCpuReference) {
  // Five copies of this number exist; two of them are here.
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
      // Init() moves the image with one DataCopy, so it must be a whole burst.
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

  // [0, 6D): three (sign, xorOffset) pairs, for strides 1, 2 and 4.
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

  // [6D, 7D): the nibble split, D/2 even offsets then D/2 odd offsets.
  const size_t even_base = 6 * static_cast<size_t>(kD);
  const size_t odd_base = even_base + static_cast<size_t>(kD / 2);
  for (int p = 0; p < kD / 2; ++p) {
    ASSERT_EQ(tables[even_base + static_cast<size_t>(p)], 8 * p) << "pair " << p;
    ASSERT_EQ(tables[odd_base + static_cast<size_t>(p)], 8 * p + 4) << "pair " << p;
  }

  // [7D, 7D + B) and [7D + B, 7D + 2B): batched expansion.
  const size_t expand_base = 7 * static_cast<size_t>(kD);
  const size_t select_base = expand_base + static_cast<size_t>(kD) * kRows;
  for (int b = 0; b < kD * kRows; ++b) {
    ASSERT_EQ(tables[expand_base + static_cast<size_t>(b)], 4 * (b / 2)) << "batch lane " << b;
    ASSERT_FLOAT_EQ(as_float(tables[select_base + static_cast<size_t>(b)]), static_cast<float>(b % 2))
        << "batch lane " << b;
  }

  // [7D + 2B, +16): the Lloyd-Max reconstruction levels, which Dequantize4Bit
  // gathers against with a byte offset of 4 * bin. The section is checked for
  // the two properties the codec actually relies on -- that it is the table the
  // host reference quantises with, and that it is strictly increasing, without
  // which the threshold scan's bin index would not select the nearest centroid.
  const size_t centroid_base = select_base + static_cast<size_t>(kD) * kRows;
  ASSERT_EQ(tables.size(), centroid_base + static_cast<size_t>(tq::kLevels));
  for (int level = 0; level < tq::kLevels; ++level) {
    ASSERT_FLOAT_EQ(as_float(tables[centroid_base + static_cast<size_t>(level)]), tq::kLloydMaxCentroids[level])
        << "centroid " << level;
  }
  for (int level = 1; level < tq::kLevels; ++level) {
    ASSERT_LT(tq::kLloydMaxCentroids[level - 1], tq::kLloydMaxCentroids[level]) << "centroid " << level;
  }

  // The table is the Lloyd-Max fixed point, so every threshold must sit exactly
  // midway between the centroids it separates. A table edited on one side only
  // would still be monotone and would still decode; it would just quantise to
  // something other than the nearest centroid.
  for (int i = 0; i < tq::kThresholdCount; ++i) {
    const float midpoint = 0.5f * (tq::kLloydMaxCentroids[i] + tq::kLloydMaxCentroids[i + 1]);
    ASSERT_NEAR(tq::kLloydMaxThresholds[i], midpoint, 1e-6f) << "threshold " << i;
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

  // The first eight values at head_size 128, pinned on the Python side too
  // (tests/ut/attention/test_turboquant_v1.py). A cache written by one half of
  // the project has to be readable by the other, and this is the cheapest place
  // a seed change would be caught.
  const std::vector<float> pinned = tqh::PiSigns(128);
  uint32_t state = tq::kPiSeed + 128u;
  for (int i = 0; i < 8; ++i) {
    // Recomputed from the documented LCG rather than transcribed, so this stays
    // a check on the generator and not on a copied literal.
    state = state * 1664525u + 1013904223u;
    const float from_lcg = ((state >> 16) & 1u) ? 1.0f : -1.0f;
    EXPECT_FLOAT_EQ(pinned[static_cast<size_t>(i)], from_lcg) << "channel " << i;
  }
}

TEST(TurboQuantLaunchContract, GridPlansMatchTheAdapterArithmetic) {
  constexpr int64_t kAiv = 40;

  // Write path: every token is covered exactly once, and no core is launched
  // with nothing to do.
  for (int64_t tokens : {1, 7, 40, 41, 1024}) {
    const tqh::ReshapeAndCacheGrid grid = tqh::PlanReshapeAndCache(tokens, kAiv);
    EXPECT_GT(grid.tokens_per_core, 0u) << "tokens " << tokens;
    EXPECT_GE(static_cast<int64_t>(grid.block_dim) * grid.tokens_per_core, tokens) << "tokens " << tokens;
    EXPECT_LE(static_cast<int64_t>(grid.block_dim - 1) * grid.tokens_per_core, tokens - 1) << "tokens " << tokens;
    EXPECT_LE(static_cast<int64_t>(grid.block_dim), kAiv) << "tokens " << tokens;
  }

  // Decode: the split count never exceeds the cap, never exceeds the blocks a
  // sequence has, and is at least one even for an empty block table.
  for (int64_t tokens : {1, 4, 64}) {
    for (int64_t blocks : {0, 1, 3, 64}) {
      const tqh::PagedAttentionGrid grid = tqh::PlanPagedAttention(tokens, kNumHeads, kHeadSize, blocks, kAiv);
      EXPECT_GE(grid.num_splits, 1) << "tokens " << tokens << " blocks " << blocks;
      EXPECT_LE(grid.num_splits, tqh::kMaxSequenceSplits) << "tokens " << tokens << " blocks " << blocks;
      if (blocks > 0) {
        EXPECT_LE(grid.num_splits, blocks) << "tokens " << tokens << " blocks " << blocks;
      }
      EXPECT_EQ(grid.workspace_floats,
                static_cast<size_t>(tokens * kNumHeads * grid.num_splits * (kHeadSize + tqh::kPartialTail)))
          << "tokens " << tokens << " blocks " << blocks;
      EXPECT_GT(grid.split_block_dim, 0u) << "tokens " << tokens << " blocks " << blocks;
      EXPECT_GT(grid.combine_block_dim, 0u) << "tokens " << tokens << " blocks " << blocks;
      // The combine stage has exactly base_tasks work items, so it must never
      // be given a larger grid than the split stage.
      EXPECT_LE(grid.combine_block_dim, grid.split_block_dim) << "tokens " << tokens << " blocks " << blocks;
    }
  }

  // Zero tokens is a no-op on both paths rather than a zero-sized grid, which
  // the runtime rejects.
  EXPECT_EQ(tqh::PlanReshapeAndCache(0, kAiv).block_dim, 0u);
  EXPECT_EQ(tqh::PlanPagedAttention(0, kNumHeads, kHeadSize, 4, kAiv).split_block_dim, 0u);
}

// ---------------------------------------------------------------------------
// The kernels themselves
// ---------------------------------------------------------------------------

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

  // A -1 slot is how the scheduler marks a padded token. The kernel must skip
  // it entirely - not write zeros, not write the quantised value somewhere
  // else - or a padded step would corrupt whatever block id -1 aliases to.
  Scenario scenario = MakeScenario(0x13572468u);
  const int kSkipped = 5;
  const int32_t skipped_slot = scenario.slots[kSkipped];
  scenario.slots[kSkipped] = -1;

  DeviceScenario device(scenario, AscendTestEnvironment::Instance().stream());
  device.RunWritePath();

  std::vector<int8_t> reference_key, reference_value;
  std::vector<float> reference_scales;
  ReferenceWritePath(scenario, &reference_key, &reference_value, &reference_scales);

  // Both caches start zeroed and the reference skips the same token, so the
  // slot that -1 replaced must still be all zero on the device.
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

  // The decode has to read what the write path wrote, not a separately built
  // cache: that is what makes this a test of the pipeline rather than of two
  // kernels that happen to agree with the same reference.
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

  // The kernel keeps the accumulator in fp32 and rounds once, at the store; the
  // reference does the same, so the two differ only by the fp16 output
  // quantisation and by the order the online softmax visits blocks. Both are
  // bounded, and neither is data dependent, so this is a tight bound rather
  // than a tuned one.
  EXPECT_GT(metrics.cosine_similarity, 0.9995) << "the decode kernel disagrees with the reference in direction, "
                                                  "which fp16 rounding of the output cannot cause";
  EXPECT_LT(metrics.relative_l2, 5e-3) << "the decode kernel disagrees with the reference in magnitude";
}

}  // namespace test
}  // namespace vllm_ascend
