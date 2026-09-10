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

// TurboQuant KV cache: the AIV-only 4-bit decode and the Cube-native kv5fp8
// decode, timed, against a physical fp16 decode of the same shape.
//
// Nothing in csrc/attention/turboquant/turboquant_kernels.cpp touches the Cube,
// so every number the 4-bit legs report is what the vector cores alone can do.
//
// What is timed, per context length S in {512, 1024, 2048}:
//
//   tq4_write_s<S>      the cache write path: S tokens rotated, quantised to
//                       4 bits and scattered. One launch.
//   tq4_decode_s<S>     the decode: split then combine, two launches on one
//                       stream. The cache is written once during setup, so the
//                       timed region is the read path only.
//   kv5fp8_write_s<S>   the 5-bit cache write: rotate, encode, scatter.
//   kv5fp8_decode_s<S>  the Cube-native decode: 5-bit storage unpacked to
//                       fp8_e4m3fn on the AIV and multiplied on the Cube.
//   fp16_decode_s<S>    the baseline: the same Cube decode over an unquantised
//                       fp16 paged KV cache.
//
// The last three are WORK IN PROGRESS and are OFF by default; see THE WIP CUBE
// GATE below and TURBOQUANT_TESTS.md section 13.8. A default run is the
// verified 4-bit AIV path and nothing else.
//
// The fp16 baseline is turboquant_fp16_decode_split in
// csrc/attention/turboquant/turboquant_mm_kernels.cpp -- the same Cube decode
// with the codec removed -- rather than aclnnFusedInferAttentionScore, which
// does not exist on this part (planning returns 361001). Same task
// decomposition, GQA batching, tile size, online softmax and split count, so
// the ratio between the two measures the codec.
//
// The traffic model is also printed. It is exact arithmetic over the two cache
// layouts, not a measurement.
//
// SHAPES. Qwen3.5-2B's full-attention layer as common/ascend950_shapes.hpp
// records it: head_dim 256, 8 query heads over 2 kv heads, block_size 128.
//
//   ASCEND_BENCH_TQ_CONTEXTS=512,1024   restricts the sweep. Everything else is
//                                       the shared ASCEND_BENCH_* set; see
//                                       common/benchmark.hpp.
//   VLLM_ASCEND_TQ_CUBE_WIP=1           adds the work-in-progress Cube legs.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "ascend950_shapes.hpp"
#include "benchmark.hpp"
#include "device_buffer.hpp"
#include "fp16.hpp"
#include "random_data.hpp"
#include "turbo_quant_cpu.h"
#include "turboquant_launch.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {

const char* kSuiteName = "turboquant_950pr (4-bit rotated KV cache, AIV only, no Cube)";

namespace {

namespace tqh = turboquant_host;
namespace tqm = vllm_ascend::turboquant;
namespace s950 = shapes950;

// --- the shape ---------------------------------------------------------------

constexpr int64_t kHeadSize = s950::kHeadDim;        // 256
constexpr int64_t kNumHeads = s950::kNumHeads;       // 8
constexpr int64_t kNumKvHeads = s950::kNumKvHeads;   // 2
constexpr int64_t kBlockSize = s950::kBlockSize;     // 128
constexpr int64_t kQueryTokens = 1;                  // decode
// 1 / sqrt(256), exact in fp16 and in fp32, so the scale contributes no
// rounding of its own. The kernels take the reciprocal square root rather than
// computing one.
constexpr float kAttentionScale = s950::kAttentionScale;
constexpr float kInvSqrtHeadSize = s950::kAttentionScale;

// Context lengths, in tokens. The brief's sweep.
const int64_t kDefaultContextLens[] = {512, 1024, 2048};

// fp16 bytes per element, and the fp32 words in a partial's tail. Named so the
// traffic arithmetic below reads as arithmetic rather than as constants.
constexpr double kHalfBytes = 2.0;
constexpr double kFloatBytes = 4.0;

std::vector<int64_t> ContextLens() {
  const char* override_value = std::getenv("ASCEND_BENCH_TQ_CONTEXTS");
  if (override_value == nullptr || *override_value == '\0') {
    return std::vector<int64_t>(std::begin(kDefaultContextLens), std::end(kDefaultContextLens));
  }
  std::vector<int64_t> lens;
  std::istringstream stream(override_value);
  std::string field;
  while (std::getline(stream, field, ',')) {
    const long long parsed = std::strtoll(field.c_str(), nullptr, 10);
    if (parsed > 0) {
      lens.push_back(static_cast<int64_t>(parsed));
    }
  }
  if (lens.empty()) {
    std::printf("[ascend-bench] ASCEND_BENCH_TQ_CONTEXTS='%s' parsed to nothing; using the default sweep\n",
                override_value);
    return std::vector<int64_t>(std::begin(kDefaultContextLens), std::end(kDefaultContextLens));
  }
  return lens;
}

// --- THE WIP CUBE GATE -------------------------------------------------------
//
// The Cube-native legs all drive
// csrc/attention/turboquant/turboquant_mm_kernels.cpp, and that path does not
// work yet. Two defects are open, both recorded in TURBOQUANT_TESTS.md 13.8:
// the score GEMM's [n, k] B operand does not reproduce the host product, and
// TurboQuantCubeMm::Init allocates L0A/L0B/L0C on both halves of the MIX
// kernel, which stalls the AIV half on "pem_lsu: unrecognize ldst addr".
//
// A launch into a faulting kernel takes the stream down and every case queued
// behind it, so the gate has to be before the launch rather than a check after
// it. VLLM_ASCEND_TQ_CUBE_WIP=1 adds the Cube legs back; that is the same
// variable REQUIRE_CUBE_WIP_OPT_IN uses for the two sim tests, restated here
// because that header includes gtest and this binary is not a gtest binary.
// -DVLLM_ASCEND_TESTS_ENABLE_WIP_CUBE=ON flips the default.
//
// Delete this gate when the path works. Do not weaken it.
#if defined(VLLM_ASCEND_TQ_CUBE_WIP_DEFAULT_ON)
constexpr bool kCubeWipDefault = true;
#else
constexpr bool kCubeWipDefault = false;
#endif

bool CubeWipEnabled() {
  const char* raw = std::getenv("VLLM_ASCEND_TQ_CUBE_WIP");
  if (raw == nullptr || *raw == '\0') {
    return kCubeWipDefault;
  }
  return raw[0] != '0';
}

void PrintCubeWipBanner(bool enabled) {
  if (enabled) {
    std::printf("[ascend-bench] Cube-native legs are ENABLED (VLLM_ASCEND_TQ_CUBE_WIP). They are work in\n"
                "[ascend-bench]   progress: the score GEMM's [n, k] B form does not reproduce the host\n"
                "[ascend-bench]   product, and TurboQuantCubeMm::Init allocates L0 on the vector core, which\n"
                "[ascend-bench]   stalls the split kernel on the camodel. Expect a stream fault rather than a\n"
                "[ascend-bench]   wrong number, and treat any timing they do produce as provisional.\n"
                "[ascend-bench]   See TURBOQUANT_TESTS.md section 13.8.\n");
  } else {
    std::printf("[ascend-bench] Cube-native legs (kv5fp8_write, kv5fp8_decode, fp16_decode) are SKIPPED:\n"
                "[ascend-bench]   turboquant_mm_kernels.cpp is work in progress and faults rather than\n"
                "[ascend-bench]   returning a wrong number. This run is the verified AIV-only 4-bit path.\n"
                "[ascend-bench]   Set VLLM_ASCEND_TQ_CUBE_WIP=1, or configure with\n"
                "[ascend-bench]   -DVLLM_ASCEND_TESTS_ENABLE_WIP_CUBE=ON, to include them.\n"
                "[ascend-bench]   See TURBOQUANT_TESTS.md section 13.8.\n");
  }
  std::fflush(stdout);
}

// --- the traffic model -------------------------------------------------------
//
// Exact byte counts, not estimates. Two different questions:
//
//   footprint   what the KV cache for S tokens occupies in HBM -- the capacity
//               win, independent of how attention reads it.
//   decode read what one decode step streams -- the bandwidth win. Larger per
//               byte of cache than the footprint suggests, because every query
//               head re-reads its kv head's rows. Whether L2 absorbs some of
//               those re-reads is not guessed at: the model counts what the
//               kernel asks the memory system for, under the same convention
//               for both legs.
struct TrafficModel {
  int64_t context_len = 0;
  double fp16_footprint_bytes = 0.0;
  double tq4_packed_bytes = 0.0;
  double tq4_scale_bytes = 0.0;
  double tq4_footprint_bytes = 0.0;
  double kv5_packed_bytes = 0.0;
  double kv5_footprint_bytes = 0.0;
  double fp16_decode_read_bytes = 0.0;
  double tq4_decode_read_bytes = 0.0;
  double kv5_decode_read_bytes = 0.0;

  double footprint_ratio() const {
    return tq4_footprint_bytes > 0.0 ? fp16_footprint_bytes / tq4_footprint_bytes : 0.0;
  }
  double decode_read_ratio() const {
    return tq4_decode_read_bytes > 0.0 ? fp16_decode_read_bytes / tq4_decode_read_bytes : 0.0;
  }
  double kv5_footprint_ratio() const {
    return kv5_footprint_bytes > 0.0 ? fp16_footprint_bytes / kv5_footprint_bytes : 0.0;
  }
  double kv5_decode_read_ratio() const {
    return kv5_decode_read_bytes > 0.0 ? fp16_decode_read_bytes / kv5_decode_read_bytes : 0.0;
  }
};

TrafficModel ModelTrafficFor(int64_t context_len, const tqh::PagedAttentionGrid& decode_grid) {
  const double s = static_cast<double>(context_len);
  const double d = static_cast<double>(kHeadSize);
  const double kv = static_cast<double>(kNumKvHeads);
  const double heads = static_cast<double>(kNumHeads);
  const double scale_slot = static_cast<double>(tqh::ScaleSlotFloats(kNumKvHeads));

  TrafficModel model;
  model.context_len = context_len;

  // Residency: K and V, one row per token per kv head.
  model.fp16_footprint_bytes = 2.0 * s * kv * d * kHalfBytes;
  model.tq4_packed_bytes = 2.0 * s * kv * (d / tqh::kPackFactor);
  model.tq4_scale_bytes = s * scale_slot * kFloatBytes;
  model.tq4_footprint_bytes = model.tq4_packed_bytes + model.tq4_scale_bytes;

  // The 5-bit slot is not d/2: it is a low plane of d nibbles plus an MSB plane
  // of d bits, and ModePackedBytes derives it from the mode config rather than
  // from a constant restated here. The scale plane is the 4-bit path's.
  const double kv5_slot = static_cast<double>(tqh::ModePackedBytes(tqm::TurboQuantMode::KV5_FP8, kHeadSize));
  model.kv5_packed_bytes = 2.0 * s * kv * kv5_slot;
  model.kv5_footprint_bytes = model.kv5_packed_bytes + model.tq4_scale_bytes;

  // One decode step. Query and output rows are two fp16 vectors per head and
  // are kept in the count so the legs are counted the same way, not because
  // they matter next to the context.
  const double query_and_output = 2.0 * heads * d * kHalfBytes;

  model.fp16_decode_read_bytes = heads * s * 2.0 * d * kHalfBytes + query_and_output;

  // The packed rows and the scale-plane tile that is copied in with them, plus
  // the flash-decoding partials: the split stage writes them and the combine
  // stage reads them back, so they cross HBM twice.
  const double partials = 2.0 * static_cast<double>(kQueryTokens * kNumHeads) *
                          static_cast<double>(decode_grid.num_splits) *
                          static_cast<double>(kHeadSize + tqh::kPartialTail) * kFloatBytes;
  model.tq4_decode_read_bytes =
      heads * s * 2.0 * (d / tqh::kPackFactor) + heads * s * scale_slot * kFloatBytes + query_and_output + partials;

  // The Cube decode reads each cached row once per *kv* head rather than once
  // per query head, because the query heads of a group are batched into the
  // GEMM's M dimension. Partial traffic is unchanged: partials are indexed per
  // head either way.
  model.kv5_decode_read_bytes =
      kv * s * 2.0 * kv5_slot + kv * s * scale_slot * kFloatBytes + query_and_output + partials;

  return model;
}

// Attention FLOPs for one decode step: QK^T and the value accumulation, two
// operations per element of each. The rotation and the codec are not counted,
// so this is the same number the fp16 leg does and TFLOP/s is comparable
// between them.
double AttentionFlops(int64_t context_len) {
  return 4.0 * static_cast<double>(kNumHeads) * static_cast<double>(context_len) * static_cast<double>(kHeadSize);
}

void PrintTrafficModel(const std::vector<TrafficModel>& models) {
  std::printf("\n[ascend-bench] KV traffic model (exact byte counts, head_dim=%lld heads=%lld kv_heads=%lld "
              "block=%lld)\n",
              static_cast<long long>(kHeadSize), static_cast<long long>(kNumHeads),
              static_cast<long long>(kNumKvHeads), static_cast<long long>(kBlockSize));
  std::printf("[ascend-bench]   cache footprint for S tokens, and the bytes one decode step streams\n\n");
  std::printf("  %6s  %14s %14s %8s   %16s %16s %8s\n", "S", "fp16 cache", "tq4 cache", "ratio",
              "fp16 decode read", "tq4 decode read", "ratio");
  std::printf("  %s\n", std::string(6 + 2 + 14 + 1 + 14 + 1 + 8 + 3 + 16 + 1 + 16 + 1 + 8, '-').c_str());
  for (const TrafficModel& model : models) {
    std::printf("  %6lld  %11.1f KiB %11.1f KiB %7.2fx   %13.1f KiB %13.1f KiB %7.2fx\n",
                static_cast<long long>(model.context_len), model.fp16_footprint_bytes / 1024.0,
                model.tq4_footprint_bytes / 1024.0, model.footprint_ratio(),
                model.fp16_decode_read_bytes / 1024.0, model.tq4_decode_read_bytes / 1024.0,
                model.decode_read_ratio());
  }
  // The 5-bit rate gets its own table rather than three more columns on the
  // first: it is a different cache layout read by a different task
  // decomposition, and putting it beside the 4-bit one invites reading the two
  // decode-read columns as if they counted the same thing.
  std::printf("\n  %6s  %14s %8s   %16s %8s\n", "S", "kv5 cache", "ratio", "kv5 decode read", "ratio");
  std::printf("  %s\n", std::string(6 + 2 + 14 + 1 + 8 + 3 + 16 + 1 + 8, '-').c_str());
  for (const TrafficModel& model : models) {
    std::printf("  %6lld  %11.1f KiB %7.2fx   %13.1f KiB %7.2fx\n", static_cast<long long>(model.context_len),
                model.kv5_footprint_bytes / 1024.0, model.kv5_footprint_ratio(),
                model.kv5_decode_read_bytes / 1024.0, model.kv5_decode_read_ratio());
  }
  // Derived from the same ScaleSlotFloats the kernel uses rather than quoted: a
  // token carries 2 * num_kv_heads live scale lanes in a slot rounded up to a
  // whole 32-byte burst.
  const double live_lanes = 2.0 * static_cast<double>(kNumKvHeads);
  const double slot_lanes = static_cast<double>(tqh::ScaleSlotFloats(kNumKvHeads));
  std::printf("\n[ascend-bench]   the tq4 cache column is %.1f%% packed codes and %.1f%% scale plane; at "
              "kv_heads=%lld a token's\n"
              "[ascend-bench]   slot is %.0f fp32 lanes for %.0f live ones, so %.0f%% of that scale plane is "
              "burst padding -\n"
              "[ascend-bench]   %.1f%% of the whole 4-bit cache, and the price of every DMA being aligned\n",
              100.0 * models.front().tq4_packed_bytes / models.front().tq4_footprint_bytes,
              100.0 * models.front().tq4_scale_bytes / models.front().tq4_footprint_bytes,
              static_cast<long long>(kNumKvHeads), slot_lanes, live_lanes,
              100.0 * (slot_lanes - live_lanes) / slot_lanes,
              100.0 * models.front().tq4_scale_bytes * (slot_lanes - live_lanes) / slot_lanes /
                  models.front().tq4_footprint_bytes);
  std::fflush(stdout);
}

// --- device-side scenario ----------------------------------------------------

std::string CaseName(const char* prefix, int64_t context_len) {
  std::ostringstream name;
  name << prefix << "_s" << context_len;
  return name.str();
}

// Everything one context length needs on the device, allocated once in
// BuildSuite and alive for the whole of that shape's cases. Buffers are asked
// for kBenchmarkAlignBytes rather than the 32-byte test default.
struct DecodeScenario {
  explicit DecodeScenario(int64_t context_len) : context_len_(context_len) {
    blocks_per_seq_ = (context_len + kBlockSize - 1) / kBlockSize;
    // Four times the blocks the context needs, so the block table is a genuine
    // scatter through a pool rather than a run of consecutive blocks that would
    // stay resident and flatter the measurement.
    num_blocks_ = std::max<int64_t>(4, blocks_per_seq_ * 4);

    DeterministicRandom rng(0x7451u);
    const size_t kv_elems = static_cast<size_t>(context_len * kNumKvHeads * kHeadSize);
    // Kept on the host as well as on the device, so the kv5 and fp16 legs
    // quantise, cache and read exactly the same context.
    key_host_ = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
    value_host_ = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
    const std::vector<float> query =
        rng.NormalHalfExact(static_cast<size_t>(kQueryTokens * kNumHeads * kHeadSize), 0.0f, 1.0f);
    const std::vector<float>& key = key_host_;
    const std::vector<float>& value = value_host_;

    const std::vector<int32_t> permutation = rng.Permutation(static_cast<int32_t>(num_blocks_));
    block_table_.assign(permutation.begin(), permutation.begin() + static_cast<size_t>(blocks_per_seq_));

    std::vector<int32_t> slots(static_cast<size_t>(context_len));
    for (int64_t i = 0; i < context_len; ++i) {
      const int32_t block = block_table_[static_cast<size_t>(i / kBlockSize)];
      slots[static_cast<size_t>(i)] = block * static_cast<int32_t>(kBlockSize) + static_cast<int32_t>(i % kBlockSize);
    }
    slots_host_ = slots;

    key_ = DeviceBuffer::FromHost(FloatToHalf(key), kBenchmarkAlignBytes);
    value_ = DeviceBuffer::FromHost(FloatToHalf(value), kBenchmarkAlignBytes);
    query_ = DeviceBuffer::FromHost(FloatToHalf(query), kBenchmarkAlignBytes);
    slots_ = DeviceBuffer::FromHost(slots, kBenchmarkAlignBytes);
    block_tables_ = DeviceBuffer::FromHost(block_table_, kBenchmarkAlignBytes);
    context_lens_ = DeviceBuffer::FromHost(std::vector<int32_t>(static_cast<size_t>(kQueryTokens),
                                                               static_cast<int32_t>(context_len)),
                                           kBenchmarkAlignBytes);
    pi_signs_ = DeviceBuffer::FromHost(tqh::PiSigns(kHeadSize), kBenchmarkAlignBytes);

    // The write path expands one vector per call and the decode path expands a
    // kTileRows tile; the two table images differ and are not interchangeable.
    write_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, 1), kBenchmarkAlignBytes);
    decode_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, tqh::kTileRows), kBenchmarkAlignBytes);

    key_cache_ = DeviceBuffer::Empty<int8_t>(
        tqh::PackedCacheBytes(num_blocks_, kBlockSize, kNumKvHeads, kHeadSize), kBenchmarkAlignBytes);
    value_cache_ = DeviceBuffer::Empty<int8_t>(key_cache_.size_bytes(), kBenchmarkAlignBytes);
    scale_plane_ = DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(num_blocks_, kBlockSize, kNumKvHeads),
                                              kBenchmarkAlignBytes);
    out_ = DeviceBuffer::Empty<Half>(static_cast<size_t>(kQueryTokens * kNumHeads * kHeadSize),
                                     kBenchmarkAlignBytes);

    aiv_num_ = tqh::VectorCoreNum(&aiv_queried_);
    write_grid_ = tqh::PlanReshapeAndCache(context_len, aiv_num_);
    decode_grid_ = tqh::PlanPagedAttention(kQueryTokens, kNumHeads, kHeadSize, blocks_per_seq_, aiv_num_);
    workspace_ = DeviceBuffer::Empty<float>(decode_grid_.workspace_floats, kBenchmarkAlignBytes);
  }

  void EnqueueWrite(aclrtStream stream) const {
    turboquant_reshape_and_cache_impl(
        AscendType::FP16, stream, write_grid_.block_dim, key_.get(), value_.get(), key_cache_.get(),
        value_cache_.get(), scale_plane_.get(), slots_.get(), pi_signs_.get(), write_tables_.get(),
        static_cast<uint32_t>(context_len_), static_cast<uint32_t>(kNumKvHeads), static_cast<uint32_t>(kHeadSize),
        static_cast<uint32_t>(kBlockSize), write_grid_.tokens_per_core, kInvSqrtHeadSize);
  }

  // Two launches, ordered by the stream. The split stage writes one partial per
  // (token, head, sequence split) and the combine stage reduces them; an
  // in-kernel barrier cannot order those across an arbitrary grid, so stream
  // order is the barrier.
  void EnqueueDecode(aclrtStream stream) const {
    turboquant_paged_attention_impl(
        AscendType::FP16, stream, decode_grid_.split_block_dim, decode_grid_.combine_block_dim, query_.get(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(),
        pi_signs_.get(), decode_tables_.get(), workspace_.get(), out_.get(), static_cast<uint32_t>(kQueryTokens),
        static_cast<uint32_t>(kNumHeads), static_cast<uint32_t>(kNumKvHeads), static_cast<uint32_t>(kHeadSize),
        static_cast<uint32_t>(kBlockSize), static_cast<uint32_t>(blocks_per_seq_),
        static_cast<uint32_t>(decode_grid_.num_splits), decode_grid_.split_tasks_per_core,
        decode_grid_.combine_tasks_per_core, kAttentionScale, kInvSqrtHeadSize);
  }

  // Fills the cache once, outside any timed region, so the decode case reads a
  // cache that holds the scenario's data rather than the allocation's zeros.
  void FillCache(aclrtStream stream) const {
    EnqueueWrite(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  int64_t context_len() const { return context_len_; }
  int64_t blocks_per_seq() const { return blocks_per_seq_; }
  int64_t num_blocks() const { return num_blocks_; }
  int64_t aiv_num() const { return aiv_num_; }
  bool aiv_queried() const { return aiv_queried_; }
  const tqh::PagedAttentionGrid& decode_grid() const { return decode_grid_; }
  const tqh::ReshapeAndCacheGrid& write_grid() const { return write_grid_; }
  const std::vector<int32_t>& block_table() const { return block_table_; }
  const std::vector<int32_t>& slots() const { return slots_host_; }
  const std::vector<float>& key_host() const { return key_host_; }
  const std::vector<float>& value_host() const { return value_host_; }

  std::vector<float> Output() const { return HalfToFloat(out_.ToHost<Half>()); }
  std::vector<float> ScalePlane() const { return scale_plane_.ToHost<float>(); }

  // The device-side inputs the other two legs reuse verbatim. Shared rather
  // than re-uploaded so that every leg reads one set of bytes: a second upload
  // would be a second chance for the legs to disagree.
  void* key() const { return key_.get(); }
  void* value() const { return value_.get(); }
  void* query() const { return query_.get(); }
  void* slots_dev() const { return slots_.get(); }
  void* block_tables() const { return block_tables_.get(); }
  void* context_lens() const { return context_lens_.get(); }
  void* pi_signs() const { return pi_signs_.get(); }

 private:
  int64_t context_len_ = 0;
  int64_t blocks_per_seq_ = 0;
  int64_t num_blocks_ = 0;
  int64_t aiv_num_ = 0;
  bool aiv_queried_ = false;

  std::vector<int32_t> block_table_;
  std::vector<int32_t> slots_host_;
  std::vector<float> key_host_, value_host_;

  DeviceBuffer key_, value_, query_, slots_, block_tables_, context_lens_, pi_signs_;
  DeviceBuffer write_tables_, decode_tables_;
  DeviceBuffer key_cache_, value_cache_, scale_plane_, workspace_, out_;
  tqh::ReshapeAndCacheGrid write_grid_;
  tqh::PagedAttentionGrid decode_grid_;
};

// --- the Cube-native kv5fp8 leg ---------------------------------------------
//
// WORK IN PROGRESS: nothing constructs this unless CubeWipEnabled(); see THE
// WIP CUBE GATE above.
struct Kv5Scenario {
  static constexpr tqm::TurboQuantMode kMode = tqm::TurboQuantMode::KV5_FP8;

  Kv5Scenario(const DecodeScenario& shared, int64_t context_len)
      : context_len_(context_len),
        blocks_per_seq_(shared.blocks_per_seq()),
        num_blocks_(shared.num_blocks()),
        aiv_num_(shared.aiv_num()) {
    key_cache_ = DeviceBuffer::Empty<int8_t>(
        tqh::ModePackedCacheBytes(kMode, num_blocks_, kBlockSize, kNumKvHeads, kHeadSize), kBenchmarkAlignBytes);
    value_cache_ = DeviceBuffer::Empty<int8_t>(key_cache_.size_bytes(), kBenchmarkAlignBytes);
    // The scale plane's layout does not depend on the rate: one fp32 per
    // (token, kv head, K/V), padded to a whole burst, is what both codecs
    // write, which is why the traffic model reuses the 4-bit figure for it.
    scale_plane_ = DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(num_blocks_, kBlockSize, kNumKvHeads),
                                              kBenchmarkAlignBytes);
    out_ = DeviceBuffer::Empty<Half>(static_cast<size_t>(kQueryTokens * kNumHeads * kHeadSize),
                                     kBenchmarkAlignBytes);

    // Three table images, and they are not interchangeable. The rotation image
    // is the shipping 4-bit codec's, because both kernels drive Pi through a
    // TurboQuantCodec4; the mode ones are this rate's. The decode image carries
    // the NZ permutation (nz_rows = kCubeTileRows); the write image does not.
    rot_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, 1), kBenchmarkAlignBytes);
    write_tables_ =
        DeviceBuffer::FromHost(tqh::ModeTables(kMode, kHeadSize, 1, /*nz_rows=*/0), kBenchmarkAlignBytes);
    decode_tables_ = DeviceBuffer::FromHost(
        tqh::ModeTables(kMode, kHeadSize, tqh::kUnpackRows, tqh::kCubeTileRows), kBenchmarkAlignBytes);

    write_grid_ = tqh::PlanReshapeAndCache(context_len, aiv_num_);
    decode_grid_ =
        tqh::PlanCubeDecode(kQueryTokens, kNumHeads, kNumKvHeads, kHeadSize, blocks_per_seq_, aiv_num_);
    workspace_ = DeviceBuffer::Empty<float>(decode_grid_.workspace_floats, kBenchmarkAlignBytes);
  }

  void EnqueueWrite(const DecodeScenario& shared, aclrtStream stream) const {
    turboquant_mm_reshape_and_cache_impl(
        static_cast<int32_t>(kMode), AscendType::FP16, stream, write_grid_.block_dim, shared.key(), shared.value(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), shared.slots_dev(), shared.pi_signs(),
        rot_tables_.get(), write_tables_.get(), static_cast<uint32_t>(context_len_),
        static_cast<uint32_t>(kNumKvHeads), static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize),
        write_grid_.tokens_per_core, kInvSqrtHeadSize);
  }

  // Split on the Cube, then combine. The combine stage is the shipping
  // AIV path's, unchanged: the partials the Cube split writes are in exactly
  // its layout, so there is no second reduction to keep in step. Stream order
  // is the barrier between the two, for the same reason as in the 4-bit path.
  void EnqueueDecode(const DecodeScenario& shared, aclrtStream stream) const {
    turboquant_mm_decode_split_impl(
        static_cast<int32_t>(kMode), AscendType::FP16, stream, decode_grid_.split_block_dim, shared.query(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), shared.block_tables(), shared.context_lens(),
        shared.pi_signs(), rot_tables_.get(), decode_tables_.get(), workspace_.get(),
        static_cast<uint32_t>(kQueryTokens), static_cast<uint32_t>(kNumHeads), static_cast<uint32_t>(kNumKvHeads),
        static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(blocks_per_seq_), static_cast<uint32_t>(decode_grid_.num_splits),
        decode_grid_.split_tasks_per_core, kAttentionScale, kInvSqrtHeadSize);

    turboquant_paged_attention_combine_impl(
        AscendType::FP16, stream, decode_grid_.combine_block_dim, workspace_.get(), shared.pi_signs(),
        rot_tables_.get(), out_.get(), static_cast<uint32_t>(kQueryTokens), static_cast<uint32_t>(kNumHeads),
        static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(decode_grid_.num_splits),
        decode_grid_.combine_tasks_per_core, kInvSqrtHeadSize);
  }

  void FillCache(const DecodeScenario& shared, aclrtStream stream) const {
    EnqueueWrite(shared, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  std::vector<float> Output() const { return HalfToFloat(out_.ToHost<Half>()); }
  std::vector<float> ScalePlane() const { return scale_plane_.ToHost<float>(); }
  const tqh::CubeDecodeGrid& decode_grid() const { return decode_grid_; }

 private:
  int64_t context_len_ = 0;
  int64_t blocks_per_seq_ = 0;
  int64_t num_blocks_ = 0;
  int64_t aiv_num_ = 0;

  DeviceBuffer key_cache_, value_cache_, scale_plane_, workspace_, out_;
  DeviceBuffer rot_tables_, write_tables_, decode_tables_;
  tqh::ReshapeAndCacheGrid write_grid_;
  tqh::CubeDecodeGrid decode_grid_;
};

// --- the physical fp16 baseline ----------------------------------------------
//
// The same Cube decode over an unquantised fp16 paged cache: no codec, no
// tables, no scale plane, and the same shape, paging and context as the
// quantised legs.
//
// WORK IN PROGRESS behind the same gate -- it is the same kernel with one stage
// removed, so it is not an independent control.
struct Fp16Scenario {
  Fp16Scenario(const DecodeScenario& shared, int64_t context_len)
      : context_len_(context_len), blocks_per_seq_(shared.blocks_per_seq()) {
    const size_t cache_elems =
        static_cast<size_t>(shared.num_blocks() * kBlockSize * kNumKvHeads * kHeadSize);
    std::vector<float> key_cache(cache_elems, 0.0f);
    std::vector<float> value_cache(cache_elems, 0.0f);

    // The same context, written into the same slots the quantised legs used, so
    // every leg reads identical data through identical paging and the
    // comparison is of the codec rather than of the access pattern.
    const std::vector<float>& key = shared.key_host();
    const std::vector<float>& value = shared.value_host();
    for (int64_t pos = 0; pos < context_len; ++pos) {
      const size_t slot = static_cast<size_t>(shared.slots()[static_cast<size_t>(pos)]);
      const size_t src = static_cast<size_t>(pos * kNumKvHeads * kHeadSize);
      const size_t dst = slot * static_cast<size_t>(kNumKvHeads * kHeadSize);
      std::copy(key.begin() + static_cast<std::ptrdiff_t>(src),
                key.begin() + static_cast<std::ptrdiff_t>(src + kNumKvHeads * kHeadSize),
                key_cache.begin() + static_cast<std::ptrdiff_t>(dst));
      std::copy(value.begin() + static_cast<std::ptrdiff_t>(src),
                value.begin() + static_cast<std::ptrdiff_t>(src + kNumKvHeads * kHeadSize),
                value_cache.begin() + static_cast<std::ptrdiff_t>(dst));
    }

    key_cache_ = DeviceBuffer::FromHost(FloatToHalf(key_cache), kBenchmarkAlignBytes);
    value_cache_ = DeviceBuffer::FromHost(FloatToHalf(value_cache), kBenchmarkAlignBytes);
    out_ = DeviceBuffer::Empty<Half>(static_cast<size_t>(kQueryTokens * kNumHeads * kHeadSize),
                                     kBenchmarkAlignBytes);

    // The same split count the quantised leg gets, from the same planner. The
    // partial traffic is the one term the two legs' models share exactly, and
    // they only share it if the grids agree.
    decode_grid_ = tqh::PlanCubeDecode(kQueryTokens, kNumHeads, kNumKvHeads, kHeadSize, blocks_per_seq_,
                                       shared.aiv_num());
    workspace_ = DeviceBuffer::Empty<float>(decode_grid_.workspace_floats, kBenchmarkAlignBytes);
  }

  // One call: unlike the quantised leg this kernel owns both stages, because
  // there is no codec between them that would want its own launch.
  void EnqueueDecode(const DecodeScenario& shared, aclrtStream stream) const {
    turboquant_fp16_decode_impl(
        AscendType::FP16, stream, decode_grid_.split_block_dim, decode_grid_.combine_block_dim, shared.query(),
        key_cache_.get(), value_cache_.get(), shared.block_tables(), shared.context_lens(), workspace_.get(),
        out_.get(), static_cast<uint32_t>(kQueryTokens), static_cast<uint32_t>(kNumHeads),
        static_cast<uint32_t>(kNumKvHeads), static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(blocks_per_seq_), static_cast<uint32_t>(decode_grid_.num_splits),
        decode_grid_.split_tasks_per_core, decode_grid_.combine_tasks_per_core, kAttentionScale);
  }

  std::vector<float> Output() const { return HalfToFloat(out_.ToHost<Half>()); }
  const tqh::CubeDecodeGrid& decode_grid() const { return decode_grid_; }

 private:
  int64_t context_len_ = 0;
  int64_t blocks_per_seq_ = 0;
  DeviceBuffer key_cache_, value_cache_, workspace_, out_;
  tqh::CubeDecodeGrid decode_grid_;
};

// --- reporting ---------------------------------------------------------------

const BenchmarkResult* FindResult(const BenchmarkRunner& runner, const std::string& case_name, TimingMode mode) {
  for (const BenchmarkResult& result : runner.results()) {
    if (result.case_name == case_name && result.mode == mode) {
      return &result;
    }
  }
  return nullptr;
}

// The suite's own summary, printed after every case has run and before the
// generic table. What it adds is the three things the generic table cannot
// know: decode steps per second, effective KV bandwidth, and the ratio to the
// fp16 baseline.
void PrintDecodeSummary(const BenchmarkRunner& runner, const std::vector<TrafficModel>& models,
                        bool cube_enabled) {
  std::printf("\n[ascend-bench] TurboQuant decode summary (median of the pipelined mode)\n\n");
  std::printf("  %6s  %10s %10s %10s   %10s %10s %10s   %9s %9s\n", "S", "tq4 us", "kv5 us", "fp16 us",
              "kv5 step/s", "kv5 GB/s", "kv5 TF/s", "kv5:fp16", "tq4:fp16");
  std::printf("  %s\n", std::string(6 + 2 + 10 * 6 + 5 + 9 * 2 + 5, '-').c_str());

  bool any = false;
  for (const TrafficModel& model : models) {
    const BenchmarkResult* tq4 =
        FindResult(runner, CaseName("tq4_decode", model.context_len), TimingMode::kPipelined);
    const BenchmarkResult* kv5 =
        FindResult(runner, CaseName("kv5fp8_decode", model.context_len), TimingMode::kPipelined);
    const BenchmarkResult* fp16 =
        FindResult(runner, CaseName("fp16_decode", model.context_len), TimingMode::kPipelined);
    if (tq4 == nullptr && kv5 == nullptr && fp16 == nullptr) {
      continue;
    }
    any = true;
    const double tq4_us = tq4 != nullptr ? tq4->latency.median_us : 0.0;
    const double kv5_us = kv5 != nullptr ? kv5->latency.median_us : 0.0;
    const double fp16_us = fp16 != nullptr ? fp16->latency.median_us : 0.0;
    std::printf("  %6lld  %10.2f %10.2f %10.2f   ", static_cast<long long>(model.context_len), tq4_us, kv5_us,
                fp16_us);
    if (kv5 != nullptr && kv5_us > 0.0) {
      std::printf("%10.0f %10.1f %10.3f   ", 1.0e6 / kv5_us, kv5->gigabytes_per_second(), kv5->tflops());
    } else {
      std::printf("%10s %10s %10s   ", "-", "-", "-");
    }
    if (fp16_us > 0.0 && kv5_us > 0.0) {
      std::printf("%8.2fx ", fp16_us / kv5_us);
    } else {
      std::printf("%9s ", "-");
    }
    if (fp16_us > 0.0 && tq4_us > 0.0) {
      std::printf("%8.2fx\n", fp16_us / tq4_us);
    } else {
      std::printf("%9s\n", "-");
    }
  }
  if (!any) {
    std::printf("  (no decode case produced a pipelined sample)\n");
  }
  std::printf("\n[ascend-bench]   steps/s is one sequence's decode step, not a batch. The GB/s column is the\n"
              "[ascend-bench]   traffic model's kv5 decode-read bytes over the measured time, so it is what the\n"
              "[ascend-bench]   kernel asked the memory system for and not a fill-rate ceiling. The speedup\n"
              "[ascend-bench]   columns are against the physical fp16 Cube decode in the same binary -- the\n"
              "[ascend-bench]   same kernel with the codec removed -- and not against an analytic model.\n");
  if (!cube_enabled) {
    // Repeated here rather than only in the banner: the summary is the part
    // that gets pasted into a report, and six dashes in a table invite the
    // reading that the hardware tried and could not.
    std::printf("[ascend-bench]   the kv5 and fp16 columns are empty because the Cube legs were gated off,\n"
                "[ascend-bench]   not because they ran and produced nothing. This is a 4-bit AIV run, and the\n"
                "[ascend-bench]   only comparison available in it is the traffic model above.\n");
  }
  std::fflush(stdout);
}

}  // namespace

void BuildSuite(BenchmarkRunner& runner) {
  const std::vector<int64_t> context_lens = ContextLens();
  const bool cube = CubeWipEnabled();

  bool aiv_queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&aiv_queried);
  std::printf("[ascend-bench] TurboQuant decode: vector cores = %lld%s\n", static_cast<long long>(aiv_num),
              aiv_queried ? " (from aclGetDeviceCapability)" : " (assumed; the runtime declined to answer)");
  PrintCubeWipBanner(cube);
  if (cube) {
    std::printf("[ascend-bench] fp16 baseline: turboquant_fp16_decode, the Cube decode with the codec removed.\n"
                "[ascend-bench]   aclnnFusedInferAttentionScore is not used: V1..V4 are withdrawn on an\n"
                "[ascend-bench]   Ascend950 (status 361001) and V5 has never been planned on hardware, so a\n"
                "[ascend-bench]   baseline resting on either would print an empty column. See this file's header.\n");
  }
  std::fflush(stdout);

  // Built alongside the scenarios so the model's partial-traffic term uses the
  // split count the decode is actually going to launch with.
  std::vector<TrafficModel> models;
  std::vector<std::unique_ptr<DecodeScenario>> scenarios;
  std::vector<std::unique_ptr<Kv5Scenario>> kv5_scenarios;
  std::vector<std::unique_ptr<Fp16Scenario>> fp16_scenarios;
  scenarios.reserve(context_lens.size());
  for (const int64_t context_len : context_lens) {
    scenarios.emplace_back(new DecodeScenario(context_len));
    // Gated at construction and not only at launch: these allocate a second and
    // a third KV cache of the same shape, which over the sweep is real HBM
    // spent on a path that is not going to run.
    if (cube) {
      kv5_scenarios.emplace_back(new Kv5Scenario(*scenarios.back(), context_len));
      fp16_scenarios.emplace_back(new Fp16Scenario(*scenarios.back(), context_len));
    }
    models.push_back(ModelTrafficFor(context_len, scenarios.back()->decode_grid()));
  }
  PrintTrafficModel(models);

  for (size_t index = 0; index < context_lens.size(); ++index) {
    const int64_t context_len = context_lens[index];
    DecodeScenario& scenario = *scenarios[index];
    const TrafficModel& model = models[index];

    if (cube) {
      const Kv5Scenario& kv5_grid = *kv5_scenarios[index];
      std::printf("\n[ascend-bench] S=%lld: blocks_per_seq=%lld pool=%lld "
                  "tq4 splits=%lld grid(split)=%u | kv5 splits=%lld grid(split)=%u grid(combine)=%u\n",
                  static_cast<long long>(context_len), static_cast<long long>(scenario.blocks_per_seq()),
                  static_cast<long long>(scenario.num_blocks()),
                  static_cast<long long>(scenario.decode_grid().num_splits),
                  scenario.decode_grid().split_block_dim,
                  static_cast<long long>(kv5_grid.decode_grid().num_splits),
                  kv5_grid.decode_grid().split_block_dim, kv5_grid.decode_grid().combine_block_dim);
    } else {
      std::printf("\n[ascend-bench] S=%lld: blocks_per_seq=%lld pool=%lld tq4 splits=%lld "
                  "grid(write)=%u grid(split)=%u grid(combine)=%u\n",
                  static_cast<long long>(context_len), static_cast<long long>(scenario.blocks_per_seq()),
                  static_cast<long long>(scenario.num_blocks()),
                  static_cast<long long>(scenario.decode_grid().num_splits), scenario.write_grid().block_dim,
                  scenario.decode_grid().split_block_dim, scenario.decode_grid().combine_block_dim);
    }
    std::fflush(stdout);

    // --- the 4-bit cache write ----------------------------------------------
    const std::string write_name = CaseName("tq4_write", context_len);
    try {
      BenchmarkCase write_case;
      write_case.name = write_name;
      // Read: S tokens of fp16 K and V. Written: the packed codes and the scale
      // plane for those tokens, plus the int32 slot mapping that is read.
      write_case.bytes_per_iteration = model.fp16_footprint_bytes + model.tq4_footprint_bytes +
                                       kFloatBytes * static_cast<double>(context_len);
      write_case.launch = [&scenario](aclrtStream stream) { scenario.EnqueueWrite(stream); };
      // The write is idempotent: the same tokens go to the same slots every
      // launch, so the scale plane must come back bit-identical. It is summed
      // whole rather than over the written slots alone, so a scale written
      // outside the slot mapping is caught too.
      write_case.checksum = [&scenario]() { return ChecksumSum(scenario.ScalePlane()); };
      runner.Run(write_case);
    } catch (const std::exception& error) {
      runner.RecordFailure(write_name, error.what());
    }

    // --- the 4-bit AIV decode -----------------------------------------------
    const std::string decode_name = CaseName("tq4_decode", context_len);
    try {
      scenario.FillCache(runner.stream());

      BenchmarkCase decode_case;
      decode_case.name = decode_name;
      decode_case.flops_per_iteration = AttentionFlops(context_len);
      decode_case.bytes_per_iteration = model.tq4_decode_read_bytes;
      decode_case.tasks_per_launch = 2;  // split, then combine
      decode_case.launch = [&scenario](aclrtStream stream) { scenario.EnqueueDecode(stream); };
      decode_case.checksum = [&scenario]() { return ChecksumSum(scenario.Output()); };
      runner.Run(decode_case);
    } catch (const std::exception& error) {
      runner.RecordFailure(decode_name, error.what());
    }

    // --- the work-in-progress Cube legs -------------------------------------
    //
    // A skip is recorded for each gated case so the absence shows up in the
    // result table.
    if (!cube) {
      runner.Skip(CaseName("kv5fp8_write", context_len),
                  "Cube path is work in progress; set VLLM_ASCEND_TQ_CUBE_WIP=1 to run it");
      runner.Skip(CaseName("kv5fp8_decode", context_len),
                  "Cube path is work in progress; set VLLM_ASCEND_TQ_CUBE_WIP=1 to run it");
      runner.Skip(CaseName("fp16_decode", context_len),
                  "the fp16 baseline is the same Cube kernel with the codec removed, and is gated with it");
      continue;
    }

    Kv5Scenario& kv5 = *kv5_scenarios[index];
    Fp16Scenario& fp16 = *fp16_scenarios[index];

    // --- the 5-bit cache write ----------------------------------------------
    const std::string kv5_write_name = CaseName("kv5fp8_write", context_len);
    try {
      BenchmarkCase write_case;
      write_case.name = kv5_write_name;
      write_case.bytes_per_iteration = model.fp16_footprint_bytes + model.kv5_footprint_bytes +
                                       kFloatBytes * static_cast<double>(context_len);
      write_case.launch = [&kv5, &scenario](aclrtStream stream) { kv5.EnqueueWrite(scenario, stream); };
      write_case.checksum = [&kv5]() { return ChecksumSum(kv5.ScalePlane()); };
      runner.Run(write_case);
    } catch (const std::exception& error) {
      runner.RecordFailure(kv5_write_name, error.what());
    }

    // --- the Cube-native 5-bit decode ---------------------------------------
    const std::string kv5_decode_name = CaseName("kv5fp8_decode", context_len);
    try {
      kv5.FillCache(scenario, runner.stream());

      BenchmarkCase decode_case;
      decode_case.name = kv5_decode_name;
      decode_case.flops_per_iteration = AttentionFlops(context_len);
      decode_case.bytes_per_iteration = model.kv5_decode_read_bytes;
      decode_case.tasks_per_launch = 2;  // split, then combine
      decode_case.launch = [&kv5, &scenario](aclrtStream stream) { kv5.EnqueueDecode(scenario, stream); };
      decode_case.checksum = [&kv5]() { return ChecksumSum(kv5.Output()); };
      runner.Run(decode_case);
    } catch (const std::exception& error) {
      runner.RecordFailure(kv5_decode_name, error.what());
    }

    // --- the physical fp16 baseline -----------------------------------------
    //
    // A failure here is a failure and not a skip: this leg is a kernel in this
    // binary, so if it does not run something is broken rather than missing.
    const std::string fp16_name = CaseName("fp16_decode", context_len);
    try {
      BenchmarkCase fp16_case;
      fp16_case.name = fp16_name;
      fp16_case.flops_per_iteration = AttentionFlops(context_len);
      fp16_case.bytes_per_iteration = model.fp16_decode_read_bytes;
      fp16_case.tasks_per_launch = 2;  // split, then combine
      fp16_case.launch = [&fp16, &scenario](aclrtStream stream) { fp16.EnqueueDecode(scenario, stream); };
      fp16_case.checksum = [&fp16]() { return ChecksumSum(fp16.Output()); };
      runner.Run(fp16_case);
    } catch (const std::exception& error) {
      runner.RecordFailure(fp16_name, error.what());
    }
  }

  PrintDecodeSummary(runner, models, cube);
}

}  // namespace bench
}  // namespace test
}  // namespace vllm_ascend
