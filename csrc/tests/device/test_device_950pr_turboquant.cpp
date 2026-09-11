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

// The TurboQuant 4-bit KV cache on a PHYSICAL Ascend 950PR: production shapes,
// driven straight through AscendCL, with the camodel explicitly excluded.
//
// This file runs head_size 256, block_size 128 and context 64 / 512 / 1024 /
// 2048 -- the shapes Qwen3.5-2B actually decodes at, plus the one the camodel
// verified the Cube path at -- so every case is a claim about silicon.
//
// S=64 is here so the silicon sweep starts at the shape simulation covered.
// It is also the only entry that is SMALLER than block_size, which makes it the
// one case in this file where the sequence occupies a partial paged block and
// the tail masking in the decode is live.
//
// Under RUN_MODE=sim aclrtGetSocName() reports a genuine 950PR bin, so
// REQUIRE_ASCEND_950PR would pass and the S=2048 cases would run for days.
// REQUIRE_PHYSICAL_ASCEND_950PR looks at what is mapped into the process
// instead. ASCEND_TEST_ALLOW_SIMULATOR=1 overrides it and
// ASCEND_TQ_BARE_METAL_CONTEXTS shrinks the sweep.
//
// WHAT IT VALIDATES:
//
//   1. topology       the part answers aclGetDeviceCapability with a vector core
//                     count and has the HBM the shapes assume.
//   2. write path     the packed cache and the scale plane against
//                     reference/turbo_quant_cpu.h, compared in bin indices with
//                     a one-bin tolerance (the device sums its RMS scale in a
//                     tree and the host serially).
//   3. bit-exactness  byte-for-byte equality on inputs constructed so tie-break
//                     cannot happen; see MakePiPreimageContext.
//   4. decode         the split/combine pipeline against the same CPU
//                     reference, reading back the cache the device wrote.
//   5. memory layout  the geometry all five copies of it must agree on, plus
//                     the 32-byte alignment claims the kernel source makes.
//   6. bounds         a poisoned cache and a slot mapping with -1 entries: every
//                     byte outside the mapping must survive untouched.
//   7. determinism    eight decode launches from one filled cache, required to
//                     be bit-identical.

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

// --- the shape ---------------------------------------------------------------
//
// Qwen3.5-2B's full-attention layer, as common/ascend950_shapes.hpp records it.

constexpr int kHeadSize = static_cast<int>(s950::kHeadDim);       // 256
constexpr int kNumHeads = static_cast<int>(s950::kNumHeads);      // 8
constexpr int kNumKvHeads = static_cast<int>(s950::kNumKvHeads);  // 2, so a GQA group of 4
constexpr int kBlockSize = static_cast<int>(s950::kBlockSize);    // 128
constexpr int kQueryTokens = 1;                                   // decode
constexpr float kAttentionScale = s950::kAttentionScale;          // 1 / sqrt(256)
constexpr float kInvSqrtHeadSize = s950::kAttentionScale;

// The brief's sweep, with S=64 prepended: that is the context the Cube-native
// decode's fidelity was measured at on the camodel (TURBOQUANT_TESTS.md section
// 13.9), so having it here means silicon and simulation overlap at one shape
// instead of meeting nowhere. ASCEND_TQ_BARE_METAL_CONTEXTS overrides the list.
const int kDefaultContextLens[] = {64, 512, 1024, 2048};

// --- bounds ------------------------------------------------------------------
//
// The same two the kernels test uses at the small shape, unrelaxed: nothing
// about a longer context makes a bin slip more likely, since the quantisation
// is per vector.
constexpr int kMaxLevelDrift = 1;
constexpr double kMaxDifferingChannelFraction = 0.02;
constexpr double kScaleRelativeTolerance = 1e-5;

// Decode against the CPU reference. Both sides run the identical algorithm, so
// the only licensed difference is the fp16 rounding at the store.
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

// --- one scenario ------------------------------------------------------------

struct Scenario {
  int context_len = 0;
  int blocks_per_seq = 0;
  int num_blocks = 0;
  std::vector<float> key;    // [context_len, num_kv_heads, head_size], fp16-exact
  std::vector<float> value;  // same
  std::vector<float> query;  // [query_tokens, num_heads, head_size], fp16-exact
  std::vector<int32_t> slots;
  std::vector<int32_t> table;
};

// `key_override` and `value_override`, when non-empty, replace the random
// context: that is how the bit-exact case feeds the write path a vector whose
// rotation it can predict exactly.
Scenario MakeScenario(int context_len, uint32_t seed, const std::vector<float>& key_override = {},
                      const std::vector<float>& value_override = {}) {
  DeterministicRandom rng(seed);
  Scenario s;
  s.context_len = context_len;
  s.blocks_per_seq = (context_len + kBlockSize - 1) / kBlockSize;
  // Four times the blocks the context needs, so the block table is a scatter
  // through a pool rather than a consecutive run - a decode that assumed the
  // cache was sequential in the sequence would pass on the latter.
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

// The CPU reference's write path over a whole scenario, into freshly zeroed
// planes. `fill` seeds both planes, so a poisoned run can be mirrored exactly.
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

// Bin index of channel c of a packed vector: the low nibble for an even channel,
// the high nibble for an odd one, with the -128 store bias undone.
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

// Compares two packed caches over the rows the scenario wrote, in bin indices
// rather than in reconstructed values: the Lloyd-Max levels are unevenly spaced,
// so a value-space bound tight enough to catch a two-bin slip at the centre of
// the table would reject a legitimate one-bin tie-break at its edge.
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

// --- device driver -----------------------------------------------------------

// Everything one scenario needs on the device. Unlike the small-shape driver in
// test_sim_950pr_turboquant_kernels.cpp this one takes the fill byte, because the
// bounds case needs a poisoned cache rather than a zeroed one.
class DeviceScenario {
 public:
  DeviceScenario(const Scenario& s, aclrtStream stream, int8_t cache_fill = 0) : scenario_(s), stream_(stream) {
    key_ = DeviceBuffer::FromHost(FloatToHalf(s.key));
    value_ = DeviceBuffer::FromHost(FloatToHalf(s.value));
    query_ = DeviceBuffer::FromHost(FloatToHalf(s.query));
    slots_ = DeviceBuffer::FromHost(s.slots);
    pi_signs_ = DeviceBuffer::FromHost(tqh::PiSigns(kHeadSize));

    // The write path expands one vector per call and the decode path expands a
    // kTileRows tile; the two table images differ in their trailing sections and
    // are not interchangeable.
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
        tqh::PlanPagedAttention(kQueryTokens, kNumHeads, kHeadSize, scenario_.blocks_per_seq, aiv_num_);
    // Sized here rather than in the constructor: the split count, and with it
    // the workspace, is a function of the block table this decode reads.
    workspace_ = DeviceBuffer::Empty<float>(grid.workspace_floats);

    turboquant_paged_attention_impl(
        AscendType::FP16, stream_, grid.split_block_dim, grid.combine_block_dim, query_.get(), key_cache_.get(),
        value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(), pi_signs_.get(),
        decode_tables_.get(), workspace_.get(), out_.get(), static_cast<uint32_t>(kQueryTokens),
        static_cast<uint32_t>(kNumHeads), static_cast<uint32_t>(kNumKvHeads), static_cast<uint32_t>(kHeadSize),
        static_cast<uint32_t>(kBlockSize), static_cast<uint32_t>(scenario_.blocks_per_seq),
        static_cast<uint32_t>(grid.num_splits), grid.split_tasks_per_core, grid.combine_tasks_per_core,
        kAttentionScale, kInvSqrtHeadSize);
    ACL_CHECK(aclrtSynchronizeStream(stream_));
  }

  std::vector<int8_t> KeyCache() const { return key_cache_.ToHost<int8_t>(); }
  std::vector<int8_t> ValueCache() const { return value_cache_.ToHost<int8_t>(); }
  std::vector<float> ScalePlane() const { return scale_plane_.ToHost<float>(); }
  std::vector<Half> RawOutput() const { return out_.ToHost<Half>(); }
  std::vector<float> Output() const { return HalfToFloat(out_.ToHost<Half>()); }

  int64_t aiv_num() const { return aiv_num_; }
  bool aiv_queried() const { return aiv_queried_; }

  // Every global address the kernels touch has to be 32-byte aligned; see the
  // layout note at the top of turboquant_kernels.cpp. These are the bases.
  std::vector<const void*> DeviceBases() const {
    return {key_.get(),        value_.get(),       query_.get(),      slots_.get(),      pi_signs_.get(),
            write_tables_.get(), decode_tables_.get(), key_cache_.get(), value_cache_.get(),
            scale_plane_.get(), block_tables_.get(), context_lens_.get(), out_.get()};
  }

 private:
  Scenario scenario_;
  aclrtStream stream_;
  DeviceBuffer key_, value_, query_, slots_, pi_signs_;
  DeviceBuffer write_tables_, decode_tables_;
  DeviceBuffer key_cache_, value_cache_, scale_plane_;
  DeviceBuffer block_tables_, context_lens_, workspace_, out_;
  int64_t aiv_num_ = 0;
  bool aiv_queried_ = false;
};

// --- the bit-exact fixture ---------------------------------------------------

// A context whose rotated coordinates are exactly +-1.
//
// Pi is an involution, so feeding the write path Pi u hands the codec back u.
// Choosing u in {-1, +1}^D makes every step exact on both sides:
//
//   * sum of squares is exactly D in any summation order, so the scale is
//     exactly 1.0 on both;
//   * every normalised coordinate is exactly +-1.0, 0.2 away from the nearest
//     Lloyd-Max decision boundary;
//   * Pi u is a multiple of 1/8 bounded by sqrt(D), so it survives the fp16
//     store without rounding. The test asserts that rather than trusting it.
//
// Only valid where 1/sqrt(D) is a power of two -- D = 64 and D = 256.
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

}  // namespace

// ---------------------------------------------------------------------------
// 1. Topology
// ---------------------------------------------------------------------------

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

  // On silicon the runtime answers this. It is an EXPECT rather than an ASSERT
  // because a grid is only a work split - a wrong core count changes how many
  // blocks are launched, not what the kernels compute - so a release that
  // declines to answer should say so loudly and still run the rest.
  EXPECT_TRUE(queried) << "aclGetDeviceCapability(ACL_DEVICE_INFO_VECTOR_CORE_NUM) declined on a physical part; "
                          "every grid below is planned against the assumed count of "
                       << tqh::kFallbackVectorCoreNum;
  EXPECT_GT(aiv_num, 0);

  // The largest scenario allocates two packed caches, a scale plane, the fp16
  // context and the workspace. Well under a gigabyte, but a part reporting
  // nothing free is worth knowing about before the allocation fails.
  if (mem_known) {
    EXPECT_GT(free_bytes, 0u);
  }
}

// ---------------------------------------------------------------------------
// 2. Write path
// ---------------------------------------------------------------------------

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

    // The scales themselves: one fp32 division of a reduced sum of squares. A
    // relative difference larger than the tolerance means the reduction
    // disagreed, not that rounding did.
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

// ---------------------------------------------------------------------------
// 3. Bit-exactness
// ---------------------------------------------------------------------------

TEST_F(TurboQuantBareMetal, PackedCacheIsByteIdenticalOnRotationExactInputs) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  // One context length is enough: what is under test is the arithmetic of a
  // single vector, repeated. The shortest one keeps the case quick.
  const int context_len = ContextLens().front();

  const std::vector<float> vector_k = PiPreimageOfSignVector(0x1111u);
  const std::vector<float> vector_v = PiPreimageOfSignVector(0x2222u);

  // The premise, asserted rather than assumed: Pi u has to survive the fp16
  // store the device reads it through, or the device and the host are not
  // starting from the same numbers and everything below is meaningless.
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

  // The rotated coordinates are +-1 and the scale is exactly 1, so every byte
  // of the packed cache is decided by arithmetic that cannot round differently
  // on the two sides. No tolerance, no drift count: the bytes are equal or the
  // kernel is wrong.
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

      // And the scale is the exact 1.0 the construction predicts, on both
      // sides. A device that reduced the squares differently would land beside
      // it, not on it.
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

// ---------------------------------------------------------------------------
// 4. Decode
// ---------------------------------------------------------------------------

TEST_F(TurboQuantBareMetal, DecodeMatchesTheCpuReferenceAcrossContexts) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(kHeadSize);

  for (const int context_len : ContextLens()) {
    const Scenario scenario = MakeScenario(context_len, 0xD3C0u + static_cast<uint32_t>(context_len));
    DeviceScenario device(scenario, Stream());
    PrintHeader("decode", context_len, device);

    device.RunWritePath();
    device.RunDecode();

    // The reference reads back the cache the device itself wrote, so a
    // disagreement here is the decode kernel and not the write path.
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

// ---------------------------------------------------------------------------
// 5. Memory layout
// ---------------------------------------------------------------------------

TEST_F(TurboQuantBareMetal, CacheGeometryAndAlignmentMatchTheDocumentedLayout) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  // The alignment claims turboquant_kernels.cpp makes about this layout, which
  // are what let every transfer be a DataCopy rather than a DataCopyPad. They
  // are properties of the shape, so they are checked before anything is
  // allocated.
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

  // The allocation sizes, against the same arithmetic the production host does.
  const size_t packed_bytes = tqh::PackedCacheBytes(scenario.num_blocks, kBlockSize, kNumKvHeads, kHeadSize);
  const size_t scale_floats = tqh::ScalePlaneFloats(scenario.num_blocks, kBlockSize, kNumKvHeads);
  EXPECT_EQ(packed_bytes, static_cast<size_t>(scenario.num_blocks) * kBlockSize * kNumKvHeads *
                              (kHeadSize / tq::kPackFactor));
  EXPECT_EQ(scale_floats, static_cast<size_t>(scenario.num_blocks) * kBlockSize * scale_slot);
  EXPECT_EQ(device.KeyCache().size(), packed_bytes);
  EXPECT_EQ(device.ValueCache().size(), packed_bytes);
  EXPECT_EQ(device.ScalePlane().size(), scale_floats);

  // Every base the kernels are handed has to sit on a 32-byte boundary; an
  // unaligned one is what DataCopyPad exists for and this path never calls it.
  for (const void* base : device.DeviceBases()) {
    EXPECT_EQ(reinterpret_cast<uintptr_t>(base) % kBurstBytes, 0u)
        << "device allocation at " << base << " is not 32-byte aligned";
  }

  // The 4-bit cache against the fp16 one it replaces: the capacity claim, stated
  // in bytes rather than in a ratio someone has to trust.
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

// ---------------------------------------------------------------------------
// 6. Bounds
// ---------------------------------------------------------------------------

TEST_F(TurboQuantBareMetal, WritePathTouchesNoByteOutsideItsSlotMapping) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  constexpr int8_t kPoison = static_cast<int8_t>(0x5A);
  const int context_len = ContextLens().front();
  Scenario scenario = MakeScenario(context_len, 0x9E11u);

  // Every eighth token is dropped, the way a padded batch drops one. The kernel
  // must skip it entirely: not write it, and not write a zero over it either.
  for (int i = 0; i < context_len; i += 8) {
    scenario.slots[static_cast<size_t>(i)] = -1;
  }

  DeviceScenario device(scenario, Stream(), kPoison);
  PrintHeader("bounds", context_len, device);
  device.RunWritePath();

  const std::vector<int8_t> key_cache = device.KeyCache();
  const std::vector<int8_t> value_cache = device.ValueCache();

  // Which rows of the cache the mapping licensed the kernel to write.
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

  // The complement of that check: the licensed rows really were written, so a
  // kernel that wrote nothing at all does not pass by leaving the poison intact
  // everywhere.
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

// ---------------------------------------------------------------------------
// 7. Determinism
// ---------------------------------------------------------------------------

TEST_F(TurboQuantBareMetal, RepeatedDecodeLaunchesAreBitIdentical) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  // The longest context in the sweep: the more sequence splits the grid uses,
  // the more partials the combine stage has to reduce, and a race between them
  // is what this case is for.
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

}  // namespace test
}  // namespace vllm_ascend
