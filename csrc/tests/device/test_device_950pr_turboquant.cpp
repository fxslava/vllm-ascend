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
#include <limits>
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

constexpr int kBlockSize = static_cast<int>(s950::kBlockSize);
constexpr int kQueryTokens = 1;

// A decode topology: everything the write path and the decode launch size
// their buffers from. The member defaults are the shapes950 decode this suite
// has always swept; ModelTopology() builds the target-model variants.
struct Topology {
  int head_size = static_cast<int>(s950::kHeadDim);
  int num_heads = static_cast<int>(s950::kNumHeads);
  int num_kv_heads = static_cast<int>(s950::kNumKvHeads);
  float attention_scale = s950::kAttentionScale;
};

Topology ModelTopology(int head_dim, int num_heads, int num_kv_heads) {
  Topology topo;
  topo.head_size = head_dim;
  topo.num_heads = num_heads;
  topo.num_kv_heads = num_kv_heads;
  // The shapes950 constant is 1/sqrt(head_dim) (exactly 0.0625 at 256); the
  // topology sweep asserts that identity where it re-derives the scale.
  topo.attention_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  return topo;
}

// GLM-4 / Qwen-2.5-class GQA: 32 query heads over 2 or 4 KV heads, at both
// head dims the codec compresses (D = 128 packs a head into 64 bytes).
constexpr int kGqaQueryHeads = 32;
const int kGqaKvHeadChoices[] = {2, 4};
const int kTopologyHeadDims[] = {128, 256};
constexpr int kTopologyContextLen = 320;

// Contexts that end mid-block (block_size = 128): the last block carries 127,
// 63 and 127 unwritten tail slots respectively.
const int kUnalignedContextLens[] = {1, 65, 129};

// What the poisoned-tail test writes past the context length: packed planes at
// the extreme Lloyd-Max bins (level 15 -> +2.7326, level 0 -> -2.7326), with
// scale lanes at +-1e4, so a slot that reaches the softmax exponent dominates
// every live score by orders of magnitude instead of merely perturbing them.
// PoisonInactiveTail lays them out; kTailTeethScoreMargin is the logit margin
// over the live context it then has to demonstrate, per head, on the host.
constexpr int8_t kTailKeySentinel = static_cast<int8_t>(0x7F);
constexpr int8_t kTailValueSentinel = static_cast<int8_t>(0x80);
constexpr float kTailScaleSentinel = 1.0e4f;
constexpr float kTailTeethScoreMargin = 100.0f;

// What the poisoned-tail test tells the decode its fused limit is. A context
// above the limit is split, and a split launch is the only one that writes its
// per-split (acc, m, L) partials through the workspace, so it is the only one
// whose running statistics the host can read. PlanPagedAttention caps the splits
// at the block count, so a context inside one block stays unsplit whatever the
// limit says.
constexpr int64_t kSplitForcingFusedLimit = 64;

// The target-model decode topologies: 32 query heads over 2 and 4 kv heads at
// both head dims the codec packs.
std::vector<Topology> ModelTopologies() {
  std::vector<Topology> topologies;
  for (const int head_dim : kTopologyHeadDims) {
    for (const int num_kv_heads : kGqaKvHeadChoices) {
      topologies.push_back(ModelTopology(head_dim, kGqaQueryHeads, num_kv_heads));
    }
  }
  return topologies;
}

// Topologies the tail-mask guard runs at: the shapes950 decode this suite has
// always swept, and the D = 128 GQA head, so a mask that turned out to depend on
// the head dim or on the group size would show up.
std::vector<Topology> TailMaskTopologies() {
  return {Topology{}, ModelTopology(kTopologyHeadDims[0], kGqaQueryHeads, kGqaKvHeadChoices[1])};
}

std::string TopologyLabel(const Topology& topo) {
  std::ostringstream label;
  label << "D" << topo.head_size << "/H" << topo.num_heads << "/KV" << topo.num_kv_heads;
  return label.str();
}

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
  Topology topo;
  int context_len = 0;
  int blocks_per_seq = 0;
  int num_blocks = 0;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> query;
  std::vector<int32_t> slots;
  std::vector<int32_t> table;
};

Scenario MakeScenario(int context_len, uint32_t seed, const Topology& topo = Topology{},
                      const std::vector<float>& key_override = {}, const std::vector<float>& value_override = {}) {
  DeterministicRandom rng(seed);
  Scenario s;
  s.topo = topo;
  s.context_len = context_len;
  s.blocks_per_seq = (context_len + kBlockSize - 1) / kBlockSize;
  s.num_blocks = std::max(4, s.blocks_per_seq * 4);

  const size_t kv_elems = static_cast<size_t>(context_len) * topo.num_kv_heads * topo.head_size;
  s.key = key_override.empty() ? rng.NormalHalfExact(kv_elems, 0.0f, 1.0f) : key_override;
  s.value = value_override.empty() ? rng.NormalHalfExact(kv_elems, 0.0f, 1.0f) : value_override;
  s.query = rng.NormalHalfExact(static_cast<size_t>(kQueryTokens) * topo.num_heads * topo.head_size, 0.0f, 1.0f);

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
  const Topology& topo = s.topo;
  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(topo.head_size);
  key_cache->assign(tqh::PackedCacheBytes(s.num_blocks, kBlockSize, topo.num_kv_heads, topo.head_size), fill);
  value_cache->assign(key_cache->size(), fill);
  scale_plane->assign(tqh::ScalePlaneFloats(s.num_blocks, kBlockSize, topo.num_kv_heads), 0.0f);

  for (int pos = 0; pos < s.context_len; ++pos) {
    const int slot = s.slots[static_cast<size_t>(pos)];
    if (slot < 0) {
      continue;
    }
    for (int kv_head = 0; kv_head < topo.num_kv_heads; ++kv_head) {
      const size_t base = (static_cast<size_t>(pos) * topo.num_kv_heads + kv_head) * topo.head_size;
      tq::cpu_reshape_and_cache_one(s.key.data() + base, topo.head_size, signs.data(), slot, topo.num_kv_heads, kv_head,
                                    kv_head, key_cache->data(), scale_plane->data());
      tq::cpu_reshape_and_cache_one(s.value.data() + base, topo.head_size, signs.data(), slot, topo.num_kv_heads,
                                    kv_head, topo.num_kv_heads + kv_head, value_cache->data(), scale_plane->data());
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
                                 const std::vector<int32_t>& slots, const Topology& topo) {
  BinAgreement agreement;
  const size_t packed_stride = static_cast<size_t>(topo.head_size / tq::kPackFactor);
  for (const int32_t slot : slots) {
    if (slot < 0) {
      continue;
    }
    for (int kv_head = 0; kv_head < topo.num_kv_heads; ++kv_head) {
      const size_t off = (static_cast<size_t>(slot) * topo.num_kv_heads + kv_head) * packed_stride;
      for (int c = 0; c < topo.head_size; ++c) {
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
    const Topology& topo = s.topo;
    key_ = DeviceBuffer::FromHost(FloatToHalf(s.key));
    value_ = DeviceBuffer::FromHost(FloatToHalf(s.value));
    query_ = DeviceBuffer::FromHost(FloatToHalf(s.query));
    slots_ = DeviceBuffer::FromHost(s.slots);
    pi_signs_ = DeviceBuffer::FromHost(tqh::PiSigns(topo.head_size));

    write_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(topo.head_size, 1));
    decode_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(topo.head_size, tqh::kTileRows));

    const size_t cache_bytes = tqh::PackedCacheBytes(s.num_blocks, kBlockSize, topo.num_kv_heads, topo.head_size);
    key_cache_ = DeviceBuffer::FromHost(std::vector<int8_t>(cache_bytes, cache_fill));
    value_cache_ = DeviceBuffer::FromHost(std::vector<int8_t>(cache_bytes, cache_fill));
    scale_plane_ = DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(s.num_blocks, kBlockSize, topo.num_kv_heads));

    std::vector<int32_t> block_tables;
    for (int t = 0; t < kQueryTokens; ++t) {
      block_tables.insert(block_tables.end(), s.table.begin(), s.table.end());
    }
    block_tables_ = DeviceBuffer::FromHost(block_tables);
    context_lens_ = DeviceBuffer::FromHost(std::vector<int32_t>(kQueryTokens, s.context_len));
    out_ = DeviceBuffer::Empty<Half>(static_cast<size_t>(kQueryTokens) * topo.num_heads * topo.head_size);
    h16_ = DeviceBuffer::FromHost(tqh::Hadamard16Half());
    query_rot_ = DeviceBuffer::Empty<float>(static_cast<size_t>(kQueryTokens) * topo.num_heads * topo.head_size);

    aiv_num_ = tqh::VectorCoreNum(&aiv_queried_);
  }

  void RunWritePath() {
    const Topology& topo = scenario_.topo;
    const tqh::ReshapeAndCacheGrid grid = tqh::PlanReshapeAndCache(scenario_.context_len, aiv_num_);
    turboquant_reshape_and_cache_impl(
        AscendType::FP16, stream_, grid.block_dim, key_.get(), value_.get(), key_cache_.get(), value_cache_.get(),
        scale_plane_.get(), slots_.get(), pi_signs_.get(), write_tables_.get(),
        static_cast<uint32_t>(scenario_.context_len), static_cast<uint32_t>(topo.num_kv_heads),
        static_cast<uint32_t>(topo.head_size), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(scenario_.num_blocks), grid.tokens_per_core, topo.attention_scale);
    ACL_CHECK(aclrtSynchronizeStream(stream_));
  }

  // fused_context_limit is what the launch is told, not a property of the shape: a limit
  // below the context is what makes the launch split the sequence, stage its per-split
  // partials through the workspace and reduce them there.
  void RunDecode(int64_t fused_context_limit = tqh::kFusedContextLimit) {
    const Topology& topo = scenario_.topo;
    const tqh::PagedAttentionGrid grid =
        tqh::PlanPagedAttention(kQueryTokens, topo.num_heads, topo.head_size, scenario_.blocks_per_seq, kBlockSize,
                                aiv_num_, fused_context_limit);
    workspace_ = DeviceBuffer::Empty<float>(grid.workspace_floats);
    num_splits_ = grid.num_splits;

    rotate_plan_ =
        tqh::RotateQuery(stream_, AscendType::FP16, query_.get(), pi_signs_.get(), h16_.get(), write_tables_.get(),
                         query_rot_.get(), kQueryTokens, topo.num_heads, topo.head_size, aiv_num_);

    turboquant_paged_attention_impl(
        AscendType::FP16, stream_, grid.block_dim, query_rot_.get(), key_cache_.get(), value_cache_.get(),
        scale_plane_.get(), block_tables_.get(), context_lens_.get(), decode_tables_.get(), workspace_.get(),
        out_.get(), static_cast<uint32_t>(kQueryTokens), static_cast<uint32_t>(topo.num_heads),
        static_cast<uint32_t>(topo.num_kv_heads), static_cast<uint32_t>(topo.head_size),
        static_cast<uint32_t>(kBlockSize), static_cast<uint32_t>(scenario_.blocks_per_seq),
        static_cast<uint32_t>(grid.num_splits), grid.split_tasks_per_core, grid.reduce_tasks_per_core,
        static_cast<uint32_t>(fused_context_limit), topo.attention_scale, topo.attention_scale);
    ACL_CHECK(aclrtSynchronizeStream(stream_));
  }

  // Replaces the device-side caches with host-modified copies of the images a
  // previous RunWritePath produced. Used by the poisoned-tail test to sabotage
  // exactly the rows past the context length and decode them again.
  void InstallCaches(const std::vector<int8_t>& key, const std::vector<int8_t>& value,
                     const std::vector<float>& scales) {
    key_cache_ = DeviceBuffer::FromHost(key);
    value_cache_ = DeviceBuffer::FromHost(value);
    scale_plane_ = DeviceBuffer::FromHost(scales);
  }

  std::vector<int8_t> KeyCache() const { return key_cache_.ToHost<int8_t>(); }
  std::vector<int8_t> ValueCache() const { return value_cache_.ToHost<int8_t>(); }
  std::vector<float> ScalePlane() const { return scale_plane_.ToHost<float>(); }
  std::vector<Half> RawOutput() const { return out_.ToHost<Half>(); }
  // The per-split partials of the last RunDecode: [acc (head_size), runMax lane, runSum
  // lane] per (token, head, split). Empty when the launch kept every context fused.
  std::vector<float> Partials() const { return workspace_.ToHost<float>(); }
  int64_t num_splits() const { return num_splits_; }
  std::vector<float> Output() const {
    return tqh::UnrotateHeads(HalfToFloat(out_.ToHost<Half>()), scenario_.topo.head_size);
  }

  const Topology& topo() const { return scenario_.topo; }
  int64_t aiv_num() const { return aiv_num_; }
  bool aiv_queried() const { return aiv_queried_; }

  std::vector<const void*> DeviceBases() const {
    return {key_.get(),          value_.get(),         query_.get(),     slots_.get(),       pi_signs_.get(),
            write_tables_.get(), decode_tables_.get(), key_cache_.get(), value_cache_.get(), scale_plane_.get(),
            block_tables_.get(), context_lens_.get(),  out_.get()};
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
  int64_t num_splits_ = 0;
  bool aiv_queried_ = false;
};

// Byte offset of one (slot, kv head) packed row. The packed cache is
// [slot][kv head][head_size / kPackFactor bytes], so a row of D = 128 is 64 bytes and a
// token's rows are adjacent: this is the indexing every check below re-derives.
size_t PackedRowOffset(const Topology& topo, int64_t slot, int kv_head) {
  const size_t packed_stride = static_cast<size_t>(topo.head_size / tq::kPackFactor);
  return (static_cast<size_t>(slot) * static_cast<size_t>(topo.num_kv_heads) + static_cast<size_t>(kv_head)) *
         packed_stride;
}

// The physical slots of the last, partially filled block that the context does not
// reach: offsets [context_len % kBlockSize, kBlockSize) of the block the table's last
// entry names. They are live memory inside a mapped block, so a decode that reads a
// whole tile reads them whatever the context length says.
std::vector<int32_t> InactiveTailSlots(const Scenario& s) {
  std::vector<int32_t> tail;
  const int filled = s.context_len % kBlockSize;
  if (filled == 0 || s.table.empty()) {
    return tail;
  }
  const int32_t block = s.table.back();
  for (int offset = filled; offset < kBlockSize; ++offset) {
    tail.push_back(block * kBlockSize + offset);
  }
  return tail;
}

// Every head's rotated query: Pi q, which is what the decode dots the cached levels
// against, and what the poison below is aimed at.
std::vector<std::vector<float>> RotatedQueryHeads(const Scenario& s, const std::vector<int8_t>& signs) {
  const Topology& topo = s.topo;
  std::vector<std::vector<float>> heads;
  for (int head = 0; head < topo.num_heads; ++head) {
    const auto base = static_cast<std::ptrdiff_t>(static_cast<size_t>(head) * topo.head_size);
    std::vector<float> rotated(s.query.begin() + base, s.query.begin() + base + topo.head_size);
    tq::cpu_apply_pi(rotated.data(), topo.head_size, signs.data());
    heads.push_back(std::move(rotated));
  }
  return heads;
}

// The tail row poisoned against query head `head`. One row each, so no two heads of a kv
// group share one: that needs at least as many tail rows as query heads, which the
// shortest context in kUnalignedContextLens satisfies and the test asserts.
int32_t AdversarialTailSlot(const std::vector<int32_t>& tail_slots, int head) {
  return tail_slots[static_cast<size_t>(head) % tail_slots.size()];
}

// Poisons the inactive tail in place, in two passes.
//
// Every row first goes to an extreme codec level (0x7F is bins (15, 15) = +2.7326, 0x80
// is bins (0, 0) = -2.7326) with both scale lanes at +-kTailScaleSentinel, the sign
// alternating so neither direction is left untried.
//
// Then one row per query head is overwritten with the pattern aimed at that head: level
// 15 where its rotated query is positive, level 0 where it is negative, which is the
// largest dot a 4-bit row can present to it, with a positive key scale. That is what
// makes the threat independent of the draw -- a uniform row's dot is
// 2.7326 * sum(Pi q), which a head whose query happens to sum to nearly zero shrugs off,
// while an aligned row's is 2.7326 * sum |Pi q|, about 280 at D = 128, and at a scale of
// 1e4 that is a logit near 2e5 for every head.
void PoisonInactiveTail(const Scenario& s, const std::vector<int32_t>& tail_slots,
                        const std::vector<std::vector<float>>& query_rot, std::vector<int8_t>* key_cache,
                        std::vector<int8_t>* value_cache, std::vector<float>* scale_plane) {
  const Topology& topo = s.topo;
  const size_t packed_stride = static_cast<size_t>(topo.head_size / tq::kPackFactor);
  const size_t slot_floats = tq::cpu_scale_slot_floats(topo.num_kv_heads);

  for (size_t i = 0; i < tail_slots.size(); ++i) {
    const int32_t slot = tail_slots[i];
    const float sign = (i % 2 == 0) ? 1.0f : -1.0f;
    for (int kv_head = 0; kv_head < topo.num_kv_heads; ++kv_head) {
      const auto off = static_cast<std::ptrdiff_t>(PackedRowOffset(topo, slot, kv_head));
      std::fill_n(key_cache->begin() + off, packed_stride, kTailKeySentinel);
      std::fill_n(value_cache->begin() + off, packed_stride, kTailValueSentinel);
      const size_t scale_off = static_cast<size_t>(slot) * slot_floats;
      (*scale_plane)[scale_off + static_cast<size_t>(kv_head)] = sign * kTailScaleSentinel;
      (*scale_plane)[scale_off + static_cast<size_t>(topo.num_kv_heads + kv_head)] = sign * kTailScaleSentinel;
    }
  }

  const int group = topo.num_heads / topo.num_kv_heads;
  for (int head = 0; head < topo.num_heads; ++head) {
    const int32_t slot = AdversarialTailSlot(tail_slots, head);
    const int kv_head = head / group;
    const std::vector<float>& rotated = query_rot[static_cast<size_t>(head)];
    int8_t* row = key_cache->data() + PackedRowOffset(topo, slot, kv_head);
    for (int c = 0; c < topo.head_size; c += tq::kPackFactor) {
      const int low = rotated[static_cast<size_t>(c)] > 0.0f ? tq::kLevels - 1 : 0;
      const int high = rotated[static_cast<size_t>(c + 1)] > 0.0f ? tq::kLevels - 1 : 0;
      row[c / tq::kPackFactor] = static_cast<int8_t>(low + tq::kLevels * high - static_cast<int>(tq::kInt8Bias));
    }
    (*scale_plane)[static_cast<size_t>(slot) * slot_floats + static_cast<size_t>(kv_head)] = kTailScaleSentinel;
  }
}

// The logit one cached key row contributes to one rotated query head, computed the way
// cpu_paged_attention_turboquant computes it: (Pi q . c[q]) * key scale lane * scale.
float CachedLogit(const std::vector<int8_t>& key_cache, const std::vector<float>& scale_plane, const Topology& topo,
                  const std::vector<float>& query_rot, int32_t slot, int kv_head) {
  std::vector<float> levels(static_cast<size_t>(topo.head_size));
  tq::cpu_dequantize_4bit(key_cache.data() + PackedRowOffset(topo, slot, kv_head), topo.head_size, 1.0f, levels.data());
  float dot = 0.0f;
  for (int c = 0; c < topo.head_size; ++c) {
    dot += query_rot[static_cast<size_t>(c)] * levels[static_cast<size_t>(c)];
  }
  const size_t scale_off =
      static_cast<size_t>(slot) * tq::cpu_scale_slot_floats(topo.num_kv_heads) + static_cast<size_t>(kv_head);
  return dot * scale_plane[scale_off] * topo.attention_scale;
}

// Whether a fp16 bit pattern is a NaN or an infinity: exponent all ones.
bool HalfIsNonFinite(uint16_t bits) { return (bits & 0x7C00u) == 0x7C00u; }

size_t CountNonFiniteHalves(const std::vector<Half>& values) {
  size_t count = 0;
  for (const Half value : values) {
    count += HalfIsNonFinite(value.bits) ? 1u : 0u;
  }
  return count;
}

size_t CountNonFiniteFloats(const std::vector<float>& values) {
  size_t count = 0;
  for (const float value : values) {
    count += std::isfinite(value) ? 0u : 1u;
  }
  return count;
}

// Bit comparison, not a tolerance: both sides are raw device reads of the same launch
// over inputs that must produce the same arithmetic.
size_t CountDifferingFloats(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return lhs.size() + rhs.size();
  }
  size_t differing = 0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    differing += (std::memcmp(&lhs[i], &rhs[i], sizeof(float)) != 0) ? 1u : 0u;
  }
  return differing;
}

// The decode of one scenario against the CPU reference over the cache image the device
// itself wrote, which is what makes this a kernel check rather than a codec one: both
// sides read the same 4-bit levels and the same scales.
tq::FidelityMetrics DecodeAgainstReference(const Scenario& scenario, const DeviceScenario& device) {
  const Topology& topo = scenario.topo;
  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(topo.head_size);
  const std::vector<int8_t> key_cache = device.KeyCache();
  const std::vector<int8_t> value_cache = device.ValueCache();
  const std::vector<float> scale_plane = device.ScalePlane();

  std::vector<float> reference(static_cast<size_t>(topo.num_heads) * topo.head_size, 0.0f);
  tq::cpu_paged_attention_turboquant(scenario.query.data(), key_cache.data(), value_cache.data(), scale_plane.data(),
                                     scenario.table.data(), scenario.context_len, topo.num_heads, topo.num_kv_heads,
                                     topo.head_size, kBlockSize, topo.attention_scale, signs.data(), reference.data());
  return tq::cpu_fidelity(device.Output(), reference);
}

std::vector<float> PiPreimageOfSignVector(uint32_t seed, int head_size) {
  const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(head_size);
  DeterministicRandom rng(seed);
  std::vector<float> vec(static_cast<size_t>(head_size));
  for (int c = 0; c < head_size; ++c) {
    vec[static_cast<size_t>(c)] = rng.IntInRange(0, 1) == 0 ? -1.0f : 1.0f;
  }
  tq::cpu_apply_pi(vec.data(), head_size, signs.data());
  return vec;
}

class TurboQuantBareMetal : public ::testing::Test {
 protected:
  static aclrtStream Stream() { return AscendTestEnvironment::Instance().stream(); }

  static void PrintHeader(const char* label, int context_len, const DeviceScenario& device) {
    const Topology& topo = device.topo();
    std::printf("\n[turboquant/bare-metal] %s: S=%d head=%d heads=%d kv=%d block=%d aiv=%lld%s\n", label, context_len,
                topo.head_size, topo.num_heads, topo.num_kv_heads, kBlockSize, static_cast<long long>(device.aiv_num()),
                device.aiv_queried() ? "" : " (assumed, runtime declined)");
  }
};

}  // namespace

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
          ComparePackedCaches(*actual_planes[plane], *expected_planes[plane], scenario.slots, scenario.topo);
      ASSERT_GT(agreement.examined, 0u) << plane_names[plane] << ": the scenario wrote no live rows";
      std::printf("  %-6s S=%4d  max bin drift %d, %zu/%zu channels differ (%.4f%%)\n", plane_names[plane], context_len,
                  agreement.max_drift, agreement.differing, agreement.examined, 100.0 * agreement.differing_fraction());
      EXPECT_LE(agreement.max_drift, kMaxLevelDrift)
          << plane_names[plane] << " at S=" << context_len
          << ": a channel is off by more than one 4-bit bin, which a coordinate landing either side of a "
             "decision boundary cannot explain";
      EXPECT_LE(agreement.differing_fraction(), kMaxDifferingChannelFraction)
          << plane_names[plane] << " at S=" << context_len << ": " << agreement.differing << " of "
          << agreement.examined << " channels landed on a different level";
    }

    const size_t slot_floats = tq::cpu_scale_slot_floats(scenario.topo.num_kv_heads);
    double worst_scale_error = 0.0;
    for (const int32_t slot : scenario.slots) {
      if (slot < 0) {
        continue;
      }
      for (int lane = 0; lane < 2 * scenario.topo.num_kv_heads; ++lane) {
        const size_t off = static_cast<size_t>(slot) * slot_floats + static_cast<size_t>(lane);
        const double want = expected_scales[off];
        const double denom = std::max(std::fabs(want), 1e-30);
        worst_scale_error = std::max(worst_scale_error, std::fabs(actual_scales[off] - want) / denom);
      }
    }
    std::printf("  scales S=%4d  worst relative error %.3e\n", context_len, worst_scale_error);
    EXPECT_LE(worst_scale_error, kScaleRelativeTolerance)
        << "scale plane at S=" << context_len
        << ": the sum-of-squares reduction disagrees with the host by more "
           "than fp32 rounding allows";
    std::fflush(stdout);
  }
}

TEST_F(TurboQuantBareMetal, PackedCacheIsByteIdenticalOnRotationExactInputs) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  const Topology topo;
  const int context_len = ContextLens().front();

  const std::vector<float> vector_k = PiPreimageOfSignVector(0x1111u, topo.head_size);
  const std::vector<float> vector_v = PiPreimageOfSignVector(0x2222u, topo.head_size);

  for (int c = 0; c < topo.head_size; ++c) {
    ASSERT_FLOAT_EQ(HalfBitsToFloat(FloatToHalfBits(vector_k[static_cast<size_t>(c)])),
                    vector_k[static_cast<size_t>(c)])
        << "channel " << c
        << " of the Pi-preimage is not exactly representable in fp16; the construction in "
           "PiPreimageOfSignVector assumes 1/sqrt(head_size) is a power of two";
  }

  const size_t kv_elems = static_cast<size_t>(context_len) * topo.num_kv_heads * topo.head_size;
  std::vector<float> key(kv_elems);
  std::vector<float> value(kv_elems);
  for (size_t base = 0; base < kv_elems; base += topo.head_size) {
    std::copy(vector_k.begin(), vector_k.end(), key.begin() + static_cast<std::ptrdiff_t>(base));
    std::copy(vector_v.begin(), vector_v.end(), value.begin() + static_cast<std::ptrdiff_t>(base));
  }

  const Scenario scenario = MakeScenario(context_len, 0xBE17u, Topology{}, key, value);
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

  const size_t packed_stride = static_cast<size_t>(topo.head_size / tq::kPackFactor);
  const size_t slot_floats = tq::cpu_scale_slot_floats(topo.num_kv_heads);
  size_t compared_bytes = 0;
  for (const int32_t slot : scenario.slots) {
    ASSERT_GE(slot, 0);
    for (int kv_head = 0; kv_head < topo.num_kv_heads; ++kv_head) {
      const size_t off = (static_cast<size_t>(slot) * topo.num_kv_heads + kv_head) * packed_stride;
      ASSERT_EQ(0, std::memcmp(actual_key.data() + off, expected_key.data() + off, packed_stride))
          << "key slot " << slot << " kv_head " << kv_head << " is not byte-identical to the CPU reference";
      ASSERT_EQ(0, std::memcmp(actual_value.data() + off, expected_value.data() + off, packed_stride))
          << "value slot " << slot << " kv_head " << kv_head << " is not byte-identical to the CPU reference";
      compared_bytes += 2 * packed_stride;

      for (const int lane : {kv_head, topo.num_kv_heads + kv_head}) {
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

  for (const int context_len : ContextLens()) {
    const Scenario scenario = MakeScenario(context_len, 0xD3C0u + static_cast<uint32_t>(context_len));
    DeviceScenario device(scenario, Stream());
    PrintHeader("decode", context_len, device);

    device.RunWritePath();
    device.RunDecode();

    const tq::FidelityMetrics metrics = DecodeAgainstReference(scenario, device);
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

// The target-model topologies: 32 query heads over 2 and 4 kv heads at D = 128 (GLM-4 /
// Qwen2.5, 64 packed bytes per token per kv head) and at D = 256. Every buffer in the
// launch is sized from the topology, so a head dim the tiling rounded differently or a
// group size the decode indexed wrongly lands off the reference here.
TEST_F(TurboQuantBareMetal, WriteAndDecodeMatchTheCpuReferenceOnTargetModelTopologies) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  // The one topology shapes950 states directly: the sweep's re-derived scale must be the
  // same number, or every case below is running a different decode from the rest of the
  // suite.
  EXPECT_FLOAT_EQ(ModelTopology(static_cast<int>(s950::kHeadDim), kGqaQueryHeads, kGqaKvHeadChoices[0]).attention_scale,
                  s950::kAttentionScale);

  for (const Topology& topo : ModelTopologies()) {
    const std::string label = TopologyLabel(topo);
    const Scenario scenario =
        MakeScenario(kTopologyContextLen, 0x7090u + static_cast<uint32_t>(topo.head_size + topo.num_kv_heads), topo);
    DeviceScenario device(scenario, Stream());
    PrintHeader(label.c_str(), kTopologyContextLen, device);

    device.RunWritePath();

    std::vector<int8_t> expected_key;
    std::vector<int8_t> expected_value;
    std::vector<float> expected_scales;
    ReferenceWritePath(scenario, 0, &expected_key, &expected_value, &expected_scales);
    ASSERT_EQ(device.KeyCache().size(), expected_key.size()) << label;

    const BinAgreement key_agreement = ComparePackedCaches(device.KeyCache(), expected_key, scenario.slots, topo);
    const BinAgreement value_agreement = ComparePackedCaches(device.ValueCache(), expected_value, scenario.slots, topo);
    ASSERT_GT(key_agreement.examined, 0u) << label << ": the scenario wrote no live rows";
    std::printf("  %s write  key %zu/%zu channels differ (drift %d), value %zu/%zu (drift %d)\n", label.c_str(),
                key_agreement.differing, key_agreement.examined, key_agreement.max_drift, value_agreement.differing,
                value_agreement.examined, value_agreement.max_drift);
    EXPECT_LE(key_agreement.max_drift, kMaxLevelDrift) << label << ": a key channel is off by more than one bin";
    EXPECT_LE(value_agreement.max_drift, kMaxLevelDrift) << label << ": a value channel is off by more than one bin";
    EXPECT_LE(key_agreement.differing_fraction(), kMaxDifferingChannelFraction) << label << " key";
    EXPECT_LE(value_agreement.differing_fraction(), kMaxDifferingChannelFraction) << label << " value";

    device.RunDecode();
    const tq::FidelityMetrics metrics = DecodeAgainstReference(scenario, device);
    std::printf("  %s decode S=%d  cos=%.6f  snr=%7.2f dB  relL2=%.6f\n", label.c_str(), kTopologyContextLen,
                metrics.cosine_similarity, metrics.snr_db, metrics.relative_l2);
    std::fflush(stdout);

    EXPECT_EQ(CountNonFiniteHalves(device.RawOutput()), 0u) << label << ": the decode wrote a NaN or an infinity";
    EXPECT_GT(metrics.cosine_similarity, kMinDecodeCosine)
        << label << " at S=" << kTopologyContextLen << ": the decode disagrees in direction with the reference";
    EXPECT_LT(metrics.relative_l2, kMaxDecodeRelativeL2)
        << label << " at S=" << kTopologyContextLen << ": the decode disagrees in magnitude with the reference";
  }
}

TEST_F(TurboQuantBareMetal, CacheGeometryAndAlignmentMatchTheDocumentedLayout) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  constexpr size_t kBurstBytes = 32;
  const int context_len = ContextLens().front();

  std::vector<Topology> topologies = ModelTopologies();
  topologies.insert(topologies.begin(), Topology{});

  for (const Topology& topo : topologies) {
    const std::string label = TopologyLabel(topo);
    const size_t packed_stride = static_cast<size_t>(topo.head_size / tq::kPackFactor);
    const size_t scale_slot = tq::cpu_scale_slot_floats(topo.num_kv_heads);

    EXPECT_EQ(scale_slot % (kBurstBytes / sizeof(float)), 0u)
        << label << ": a token's scale slot is not a whole 32-byte burst";
    EXPECT_EQ((static_cast<size_t>(topo.num_kv_heads) * packed_stride) % kBurstBytes, 0u)
        << label
        << ": a token's packed bytes across all kv heads are not a whole number of 32-byte bursts, so the "
           "scatter cannot be one aligned DataCopy";
    EXPECT_EQ(static_cast<size_t>(tqh::ScaleSlotFloats(topo.num_kv_heads)), scale_slot)
        << label << ": the launch shim and the CPU reference disagree about the scale slot";
    EXPECT_EQ(packed_stride * tq::kPackFactor, static_cast<size_t>(topo.head_size))
        << label << ": a packed row does not hold the head's channels two to a byte";

    const Scenario scenario = MakeScenario(context_len, 0x1A70u, topo);
    DeviceScenario device(scenario, Stream());
    PrintHeader(label.c_str(), context_len, device);

    const size_t packed_bytes =
        tqh::PackedCacheBytes(scenario.num_blocks, kBlockSize, topo.num_kv_heads, topo.head_size);
    const size_t scale_floats = tqh::ScalePlaneFloats(scenario.num_blocks, kBlockSize, topo.num_kv_heads);
    EXPECT_EQ(packed_bytes, static_cast<size_t>(scenario.num_blocks) * kBlockSize *
                                static_cast<size_t>(topo.num_kv_heads) * packed_stride)
        << label;
    EXPECT_EQ(scale_floats, static_cast<size_t>(scenario.num_blocks) * kBlockSize * scale_slot) << label;
    EXPECT_EQ(device.KeyCache().size(), packed_bytes) << label;
    EXPECT_EQ(device.ValueCache().size(), packed_bytes) << label;
    EXPECT_EQ(device.ScalePlane().size(), scale_floats) << label;

    // Packed row indexing: the kv heads of one slot are adjacent rows, the next slot
    // starts one row after the last of them, and the last row of the last slot ends
    // exactly on the allocation. At D = 128 that row is 64 bytes.
    EXPECT_EQ(PackedRowOffset(topo, 0, 1) - PackedRowOffset(topo, 0, 0), packed_stride) << label;
    EXPECT_EQ(PackedRowOffset(topo, 1, 0) - PackedRowOffset(topo, 0, topo.num_kv_heads - 1), packed_stride) << label;
    EXPECT_EQ(PackedRowOffset(topo, static_cast<int64_t>(scenario.num_blocks) * kBlockSize - 1, topo.num_kv_heads - 1) +
                  packed_stride,
              packed_bytes)
        << label << ": the last packed row does not end on the allocation";

    for (const void* base : device.DeviceBases()) {
      EXPECT_EQ(reinterpret_cast<uintptr_t>(base) % kBurstBytes, 0u)
          << label << ": device allocation at " << base << " is not 32-byte aligned";
    }

    const double fp16_bytes =
        2.0 * static_cast<double>(context_len) * topo.num_kv_heads * topo.head_size * sizeof(uint16_t);
    const double scale_bytes = static_cast<double>(context_len) * static_cast<double>(scale_slot) * sizeof(float);
    const double tq4_bytes =
        2.0 * static_cast<double>(context_len) * topo.num_kv_heads * static_cast<double>(packed_stride) + scale_bytes;
    std::printf(
        "  %s S=%d: row %zu B, fp16 KV %.1f KiB, 4-bit KV %.1f KiB (%.2fx smaller, scale plane is %.1f%% "
        "of it)\n",
        label.c_str(), context_len, packed_stride, fp16_bytes / 1024.0, tq4_bytes / 1024.0, fp16_bytes / tq4_bytes,
        100.0 * scale_bytes / tq4_bytes);
    std::fflush(stdout);
    EXPECT_GT(fp16_bytes / tq4_bytes, 3.0) << label << ": the 4-bit cache is not saving what its layout says it should";
  }
}

TEST_F(TurboQuantBareMetal, WritePathTouchesNoByteOutsideItsSlotMapping) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  constexpr int8_t kPoison = static_cast<int8_t>(0x5A);
  constexpr int kUnmappedStride = 8;
  const int context_len = ContextLens().front();

  std::vector<Topology> topologies = ModelTopologies();
  topologies.insert(topologies.begin(), Topology{});

  for (const Topology& topo : topologies) {
    const std::string label = TopologyLabel(topo);
    Scenario scenario = MakeScenario(context_len, 0x9E11u, topo);

    for (int i = 0; i < context_len; i += kUnmappedStride) {
      scenario.slots[static_cast<size_t>(i)] = -1;
    }

    DeviceScenario device(scenario, Stream(), kPoison);
    PrintHeader(label.c_str(), context_len, device);
    device.RunWritePath();

    const std::vector<int8_t> key_cache = device.KeyCache();
    const std::vector<int8_t> value_cache = device.ValueCache();

    std::vector<bool> live_slot(static_cast<size_t>(scenario.num_blocks) * kBlockSize, false);
    for (const int32_t slot : scenario.slots) {
      if (slot >= 0) {
        live_slot[static_cast<size_t>(slot)] = true;
      }
    }

    const size_t packed_stride = static_cast<size_t>(topo.head_size / tq::kPackFactor);
    size_t poisoned_rows = 0;
    size_t written_rows = 0;
    for (size_t slot = 0; slot < live_slot.size(); ++slot) {
      for (int kv_head = 0; kv_head < topo.num_kv_heads; ++kv_head) {
        const size_t off = PackedRowOffset(topo, static_cast<int64_t>(slot), kv_head);
        if (live_slot[slot]) {
          ++written_rows;
          continue;
        }
        ++poisoned_rows;
        for (size_t byte = 0; byte < packed_stride; ++byte) {
          ASSERT_EQ(key_cache[off + byte], kPoison)
              << label << ": the key cache was written at slot " << slot << " kv_head " << kv_head << " byte " << byte
              << ", which no slot in the mapping names";
          ASSERT_EQ(value_cache[off + byte], kPoison)
              << label << ": the value cache was written at slot " << slot << " kv_head " << kv_head << " byte " << byte
              << ", which no slot in the mapping names";
        }
      }
    }

    size_t untouched_live_rows = 0;
    for (size_t slot = 0; slot < live_slot.size(); ++slot) {
      if (!live_slot[slot]) {
        continue;
      }
      for (int kv_head = 0; kv_head < topo.num_kv_heads; ++kv_head) {
        const size_t off = PackedRowOffset(topo, static_cast<int64_t>(slot), kv_head);
        bool all_poison = true;
        for (size_t byte = 0; byte < packed_stride && all_poison; ++byte) {
          all_poison = key_cache[off + byte] == kPoison;
        }
        if (all_poison) {
          ++untouched_live_rows;
        }
      }
    }

    std::printf("  %s %zu poisoned rows survived, %zu live rows written, %zu live rows still all-poison\n",
                label.c_str(), poisoned_rows, written_rows, untouched_live_rows);
    std::fflush(stdout);
    EXPECT_GT(poisoned_rows, 0u) << label << ": the scenario left no unmapped slot to check";
    EXPECT_EQ(untouched_live_rows, 0u)
        << label << ": rows the slot mapping names came back unwritten; the scatter did not reach them";
  }
}

// Softmax logit bleed. A context that ends mid-block leaves the rest of that block's
// slots unwritten, and the decode reads whole kTileRows tiles: those slots are read, and
// only the tail mask keeps them out of the reduction. This poisons them with the codec's
// extreme levels at a scale of 1e4 -- logits around 2e4, whose exp overflows fp32, so an
// unmasked lane does not perturb the answer, it destroys it -- and requires the decode to
// come back bit-identical to the same launch over the zero-padded cache.
//
// The teeth check is the other half: before the comparison it computes, on the host, the
// logit each poisoned row would carry if it reached the softmax, and refuses to accept a
// pass unless that logit beats every live logit by kTailTeethScoreMargin for every head.
// Without it a mask that dropped the tail into the reduction at a harmless value would
// pass this test silently.
TEST_F(TurboQuantBareMetal, PoisonedInactiveTailSlotsCannotReachTheSoftmax) {
  REQUIRE_PHYSICAL_ASCEND_950PR();

  for (const Topology& topo : TailMaskTopologies()) {
    const std::string label = TopologyLabel(topo);
    const std::vector<int8_t> signs = tq::cpu_pi_sign_vector(topo.head_size);

    for (const int context_len : kUnalignedContextLens) {
      ASSERT_NE(context_len % kBlockSize, 0) << "S=" << context_len << " fills its last block, so it has no tail";

      const Scenario scenario =
          MakeScenario(context_len, 0x7A11u + static_cast<uint32_t>(context_len + topo.head_size), topo);
      const std::vector<int32_t> tail_slots = InactiveTailSlots(scenario);
      ASSERT_EQ(tail_slots.size(), static_cast<size_t>(kBlockSize - context_len % kBlockSize));
      ASSERT_GE(tail_slots.size(), static_cast<size_t>(topo.num_heads))
          << "the tail is shorter than the head count, so two heads would share one poisoned row";

      DeviceScenario device(scenario, Stream());
      PrintHeader(label.c_str(), context_len, device);
      device.RunWritePath();

      const std::vector<int8_t> clean_key = device.KeyCache();
      const std::vector<int8_t> clean_value = device.ValueCache();
      const std::vector<float> clean_scales = device.ScalePlane();

      device.RunDecode(kSplitForcingFusedLimit);
      const std::vector<Half> clean_out = device.RawOutput();
      const std::vector<float> clean_partials = device.Partials();
      const tq::FidelityMetrics clean_metrics = DecodeAgainstReference(scenario, device);
      const int64_t num_splits = device.num_splits();

      const std::vector<std::vector<float>> query_rot = RotatedQueryHeads(scenario, signs);
      std::vector<int8_t> poisoned_key = clean_key;
      std::vector<int8_t> poisoned_value = clean_value;
      std::vector<float> poisoned_scales = clean_scales;
      PoisonInactiveTail(scenario, tail_slots, query_rot, &poisoned_key, &poisoned_value, &poisoned_scales);

      // The teeth: the margin the least-threatened head sees over the live logits the
      // poison would have to beat. Both sides are host arithmetic over the same two cache
      // images the device decodes below, so this measures the poison as written, not as
      // intended.
      const int group = topo.num_heads / topo.num_kv_heads;
      double worst_margin = std::numeric_limits<double>::infinity();
      for (int head = 0; head < topo.num_heads; ++head) {
        const std::vector<float>& rotated = query_rot[static_cast<size_t>(head)];
        const int kv_head = head / group;

        double live_max = -std::numeric_limits<double>::infinity();
        for (const int32_t slot : scenario.slots) {
          live_max = std::max<double>(live_max, CachedLogit(clean_key, clean_scales, topo, rotated, slot, kv_head));
        }
        double poisoned_max = -std::numeric_limits<double>::infinity();
        for (const int32_t slot : tail_slots) {
          poisoned_max =
              std::max<double>(poisoned_max, CachedLogit(poisoned_key, poisoned_scales, topo, rotated, slot, kv_head));
        }
        worst_margin = std::min(worst_margin, poisoned_max - live_max);
      }

      device.InstallCaches(poisoned_key, poisoned_value, poisoned_scales);
      device.RunDecode(kSplitForcingFusedLimit);
      const std::vector<Half> poisoned_out = device.RawOutput();
      const std::vector<float> poisoned_partials = device.Partials();

      std::printf("  %s S=%3d  %zu poisoned tail slots, splits=%lld, %zu partial floats, teeth margin %.3e\n",
                  label.c_str(), context_len, tail_slots.size(), static_cast<long long>(num_splits),
                  clean_partials.size(), worst_margin);
      std::printf("  %s S=%3d  zero-padded decode cos=%.6f relL2=%.6f\n", label.c_str(), context_len,
                  clean_metrics.cosine_similarity, clean_metrics.relative_l2);
      std::fflush(stdout);

      ASSERT_GE(worst_margin, kTailTeethScoreMargin)
          << label << " at S=" << context_len
          << ": the poisoned tail does not outscore the live context, so this case would pass whether the "
             "decode masked its tail or not";

      // The zero-padded baseline has to be right in its own terms first: the reference
      // sums the live context only, so a decode that let a tail slot through at a
      // harmless logit -- exp(0) = 1 in the denominator -- would already miss it here.
      EXPECT_GT(clean_metrics.cosine_similarity, kMinDecodeCosine)
          << label << " at S=" << context_len << ": the zero-padded decode already disagrees with the reference";
      EXPECT_LT(clean_metrics.relative_l2, kMaxDecodeRelativeL2) << label << " at S=" << context_len;

      ASSERT_EQ(poisoned_out.size(), clean_out.size()) << label;
      size_t differing = 0;
      size_t first_difference = poisoned_out.size();
      for (size_t i = 0; i < clean_out.size(); ++i) {
        if (poisoned_out[i].bits != clean_out[i].bits) {
          first_difference = std::min(first_difference, i);
          ++differing;
        }
      }
      EXPECT_EQ(differing, 0u) << label << " at S=" << context_len << ": " << differing << " of " << clean_out.size()
                               << " output elements moved when the inactive tail was poisoned, first at "
                               << first_difference << "; logits from past the context length reached the softmax";

      EXPECT_EQ(CountNonFiniteHalves(poisoned_out), 0u)
          << label << " at S=" << context_len << ": the output holds a NaN or an infinity";
      EXPECT_EQ(CountNonFiniteFloats(poisoned_partials), 0u)
          << label << " at S=" << context_len
          << ": a partial accumulator or a running statistic is a NaN or an infinity";

      // The running statistics themselves, where the launch splits: kPartialMaxLane is m
      // and kPartialSumLane is L, per (token, head, split). A single-block context cannot
      // be split, so only the contexts past one block reach this.
      ASSERT_EQ(poisoned_partials.size(), clean_partials.size()) << label;
      if (num_splits > 1) {
        ASSERT_FALSE(clean_partials.empty())
            << label << ": a split launch wrote no partials, so m and L were never staged";
        const size_t stride = static_cast<size_t>(topo.head_size) + static_cast<size_t>(tqh::kPartialTail);
        size_t differing_stats = 0;
        for (size_t partial = 0; partial * stride < clean_partials.size(); ++partial) {
          for (const size_t lane : {static_cast<size_t>(vllm_ascend::turboquant::kPartialMaxLane),
                                    static_cast<size_t>(vllm_ascend::turboquant::kPartialSumLane)}) {
            const size_t off = partial * stride + static_cast<size_t>(topo.head_size) + lane;
            differing_stats +=
                (std::memcmp(&clean_partials[off], &poisoned_partials[off], sizeof(float)) != 0) ? 1u : 0u;
          }
        }
        EXPECT_EQ(differing_stats, 0u) << label << " at S=" << context_len << ": " << differing_stats
                                       << " running statistics (m, L) moved when the inactive tail was poisoned";
        EXPECT_EQ(CountDifferingFloats(clean_partials, poisoned_partials), 0u)
            << label << " at S=" << context_len << ": a split partial moved when the inactive tail was poisoned";
      } else {
        std::printf("  %s S=%3d  unsplit: m and L stay in UB, only the output is comparable\n", label.c_str(),
                    context_len);
      }
    }
  }
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

}  // namespace test
}  // namespace vllm_ascend
