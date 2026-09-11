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

// TurboQuant KV cache: the AIV-only 4-bit decode and the Cube-native kv4fp8
// decode, timed, against a physical fp16 decode of the same shape.
//
// Nothing in csrc/attention/turboquant/turboquant_kernels.cpp touches the Cube,
// so every number the tq4 legs report is what the vector cores alone can do.
//
// What is timed, per context length S in {512, 1024, 2048}:
//
//   tq4_write_s<S>      the cache write path: S tokens rotated, quantised to
//                       4 bits and scattered. One launch.
//   tq4_decode_s<S>     the decode: split then combine, two launches on one
//                       stream. The cache is written once during setup, so the
//                       timed region is the read path only.
//   kv4fp8_write_s<S>   the 4-bit cache write through the multi-rate codec.
//   kv4fp8_decode_s<S>  the Cube-native decode: the same 128-byte slot tq4
//                       stores, unpacked to fp8_e4m3fn on the AIV and
//                       multiplied on the Cube.
//   fp16_decode_s<S>    the baseline: the same Cube decode over an unquantised
//                       fp16 paged KV cache.
//
// All five run by default. The comparison the binary exists to make is the
// three-way one at a fixed context length: tq4 against kv4fp8 is the Cube
// against the vector cores with the codec held fixed -- the two store the same
// nibbles against the same Lloyd-Max thresholds -- and kv4fp8 against fp16 is
// what the 4-bit cache buys over an unquantised one.
//
// kv3fp4 and kv5fp8 are still built and still dispatched by the sim tier; they
// have no leg here. Neither is on the 4-bit path this binary measures, and a
// column for each was three more cache allocations of the same shape for a
// comparison nothing was asking of this binary.
//
// ONE DEFECT IS STILL OPEN, and it is not the L0 allocation split that used to
// stall these legs (that is fixed, in TurboQuantCubeMm::Init and in
// TurboQuantFp16DecodeSplit::Init). The score GEMM stages its B operand [n, k],
// and test_sim_950pr_cube_gemm's ScoreGemmBStagedNk does not reproduce the host
// product, so every Cube leg here -- kv4fp8 and fp16 alike -- computes a wrong
// score row. The timings are still the right shape of work in the right order,
// but they are provisional until that is closed. See TURBOQUANT_TESTS.md
// section 13.8, and the banner PrintCubeBanner prints at the top of every run.
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
//   VLLM_ASCEND_TQ_CUBE_WIP=0           drops the Cube legs, leaving the
//                                       AIV-only 4-bit path.

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

const char* kSuiteName = "turboquant_950pr (4-bit rotated KV cache: AIV, Cube, and an fp16 Cube baseline)";

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

// --- THE CUBE LEGS -----------------------------------------------------------
//
// kv4fp8_write, kv4fp8_decode and fp16_decode drive
// csrc/attention/turboquant/turboquant_mm_kernels.cpp and are ON by default.
// The defect that used to make that unsafe is closed: L0A/L0B/L0C were being
// allocated on both halves of the MIX kernel, and the AIV half stalled on
// "pem_lsu: unrecognize ldst addr" with ldst_addr: 2 the first time anything
// touched one. Both Cube kernels now allocate L1 on both cores and L0 on the
// AIC alone.
//
// What is still open is numerical rather than fatal, and the banner says so on
// every run: the score GEMM stages B as [n, k], and that form does not
// reproduce the host product (TURBOQUANT_TESTS.md 13.8). The legs launch, the
// stream completes and the checksums are stable -- what they are not yet is
// right.
//
// VLLM_ASCEND_TQ_CUBE_WIP=0 drops them, which is what to reach for if a future
// defect does take the stream down: a launch into a faulting kernel takes every
// case queued behind it with it, so the switch is before the launch rather than
// a check after it.
#if defined(VLLM_ASCEND_TQ_CUBE_OFF_BY_DEFAULT)
constexpr bool kCubeDefault = false;
#else
constexpr bool kCubeDefault = true;
#endif

bool CubeEnabled() {
  const char* raw = std::getenv("VLLM_ASCEND_TQ_CUBE_WIP");
  if (raw == nullptr || *raw == '\0') {
    return kCubeDefault;
  }
  return raw[0] != '0';
}

void PrintCubeBanner(bool enabled) {
  if (enabled) {
    std::printf("[ascend-bench] Cube-native legs (kv4fp8_write, kv4fp8_decode, fp16_decode) are ON.\n"
                "[ascend-bench]   PROVISIONAL: the score GEMM stages its B operand [n, k], and that form does\n"
                "[ascend-bench]   not reproduce the host product -- see TURBOQUANT_TESTS.md section 13.8. The\n"
                "[ascend-bench]   kernels launch and complete, and the work they do is the right work in the\n"
                "[ascend-bench]   right order, so the timings below are meaningful as timings. The values\n"
                "[ascend-bench]   those kernels compute are not yet correct. Do not quote these as verified.\n"
                "[ascend-bench]   VLLM_ASCEND_TQ_CUBE_WIP=0 drops them.\n");
  } else {
    std::printf("[ascend-bench] Cube-native legs (kv4fp8_write, kv4fp8_decode, fp16_decode) are SKIPPED\n"
                "[ascend-bench]   (VLLM_ASCEND_TQ_CUBE_WIP=0). This run is the AIV-only 4-bit path, and the\n"
                "[ascend-bench]   only comparison in it is the traffic model.\n");
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
  double kv4_packed_bytes = 0.0;
  double kv4_footprint_bytes = 0.0;
  double fp16_decode_read_bytes = 0.0;
  double tq4_decode_read_bytes = 0.0;
  double kv4_decode_read_bytes = 0.0;

  double footprint_ratio() const {
    return tq4_footprint_bytes > 0.0 ? fp16_footprint_bytes / tq4_footprint_bytes : 0.0;
  }
  double decode_read_ratio() const {
    return tq4_decode_read_bytes > 0.0 ? fp16_decode_read_bytes / tq4_decode_read_bytes : 0.0;
  }
  double kv4_footprint_ratio() const {
    return kv4_footprint_bytes > 0.0 ? fp16_footprint_bytes / kv4_footprint_bytes : 0.0;
  }
  double kv4_decode_read_ratio() const {
    return kv4_decode_read_bytes > 0.0 ? fp16_decode_read_bytes / kv4_decode_read_bytes : 0.0;
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

  // The Cube slot, derived from the mode config rather than from a constant
  // restated here. It is d/2 -- the same 128 bytes at d=256 the AIV path
  // stores, because the two share the nibble layout -- and the scale plane is
  // the 4-bit path's, whose geometry does not depend on the rate. So the kv4
  // footprint equals the tq4 one exactly, and this is computed rather than
  // aliased so that a change to either mode's packing shows up as a difference.
  const double kv4_slot = static_cast<double>(tqh::ModePackedBytes(tqm::TurboQuantMode::KV4_FP8, kHeadSize));
  model.kv4_packed_bytes = 2.0 * s * kv * kv4_slot;
  model.kv4_footprint_bytes = model.kv4_packed_bytes + model.tq4_scale_bytes;

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
  // head either way. That is why the kv4 decode-read figure is well below the
  // tq4 one at the same slot width -- the storage is identical and only the
  // task decomposition differs.
  model.kv4_decode_read_bytes =
      kv * s * 2.0 * kv4_slot + kv * s * scale_slot * kFloatBytes + query_and_output + partials;

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
  // kv4fp8 gets its own table rather than two more columns on the first: it is
  // read by a different task decomposition -- once per kv head rather than once
  // per query head -- and putting it beside the AIV path's invites reading the
  // two decode-read columns as if they counted the same thing. Its cache column
  // is identical to tq4's by construction, the two storing the same nibbles, so
  // the decode-read column is the only one that separates them and that
  // difference is the decomposition alone.
  std::printf("\n  %6s  %14s %8s %16s %8s\n", "S", "kv4 cache", "ratio", "kv4 decode read", "ratio");
  std::printf("  %s\n", std::string(6 + 2 + 14 + 1 + 8 + 1 + 16 + 1 + 8, '-').c_str());
  for (const TrafficModel& model : models) {
    std::printf("  %6lld  %11.1f KiB %7.2fx %13.1f KiB %7.2fx\n", static_cast<long long>(model.context_len),
                model.kv4_footprint_bytes / 1024.0, model.kv4_footprint_ratio(),
                model.kv4_decode_read_bytes / 1024.0, model.kv4_decode_read_ratio());
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
    // Kept on the host as well as on the device, so the kv4 and fp16 legs
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

// --- the Cube-native quantised legs -----------------------------------------
//
// One scenario for every Cube-native storage rate. The mode is a template
// parameter and not a constructor argument for the same reason it is one on the
// device: the cache slot width, the table image and the codebook all follow
// from it, and a rate that only agreed at run time could size a buffer for one
// rate and write it with another.
//
// Nothing constructs this unless CubeEnabled(); see THE CUBE LEGS above.
template <tqm::TurboQuantMode MODE>
struct ModeScenario {
  static constexpr tqm::TurboQuantMode kMode = MODE;

  ModeScenario(const DecodeScenario& shared, int64_t context_len)
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

// The 4-bit Cube rate, and the only mode with a leg here. kv3fp4 and kv5fp8
// are scaffolded on the device and covered by the sim tier; instantiating
// ModeScenario for either would allocate another KV cache of the same shape per
// context length for a comparison this binary is not making.
using Kv4Scenario = ModeScenario<tqm::TurboQuantMode::KV4_FP8>;

// --- the physical fp16 baseline ----------------------------------------------
//
// The same Cube decode over an unquantised fp16 paged cache: no codec, no
// tables, no scale plane, and the same shape, paging and context as the
// quantised legs.
//
// Behind the same switch as the quantised Cube leg, and it carries the same
// provisional score GEMM -- it is that kernel with one stage removed, so it is
// not an independent control.
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
//
// Two tables. The first is the comparison this binary exists to make -- the AIV
// 4-bit path, the Cube 4-bit path and the unquantised fp16 Cube path, side by
// side at the same context length, with the two ratios that read off it. The
// second is the derived rates for the Cube 4-bit leg, which have no meaning for
// tq4 (it reads a different number of bytes) and none for fp16 (its bytes are
// the baseline).
void PrintDecodeSummary(const BenchmarkRunner& runner, const std::vector<TrafficModel>& models,
                        bool cube_enabled) {
  std::printf("\n[ascend-bench] TurboQuant decode summary (median of the pipelined mode)\n\n");
  std::printf("  %6s  %10s %10s %10s   %9s %9s\n", "S", "tq4 us", "kv4 us", "fp16 us", "kv4:fp16", "kv4:tq4");
  std::printf("  %s\n", std::string(6 + 2 + 3 * 11 + 2 + 10 + 9, '-').c_str());

  bool any = false;
  for (const TrafficModel& model : models) {
    const BenchmarkResult* tq4 =
        FindResult(runner, CaseName("tq4_decode", model.context_len), TimingMode::kPipelined);
    const BenchmarkResult* kv4 =
        FindResult(runner, CaseName("kv4fp8_decode", model.context_len), TimingMode::kPipelined);
    const BenchmarkResult* fp16 =
        FindResult(runner, CaseName("fp16_decode", model.context_len), TimingMode::kPipelined);
    if (tq4 == nullptr && kv4 == nullptr && fp16 == nullptr) {
      continue;
    }
    any = true;
    const double tq4_us = tq4 != nullptr ? tq4->latency.median_us : 0.0;
    const double kv4_us = kv4 != nullptr ? kv4->latency.median_us : 0.0;
    const double fp16_us = fp16 != nullptr ? fp16->latency.median_us : 0.0;
    // A leg with no sample prints a dash. 0.00 us would read as a measurement,
    // which is the misreading the note at the foot of this table exists to
    // stop, and a run with the Cube legs dropped leaves two of these three
    // columns empty.
    const double latencies[] = {tq4_us, kv4_us, fp16_us};
    std::printf("  %6lld  ", static_cast<long long>(model.context_len));
    for (const double us : latencies) {
      if (us > 0.0) {
        std::printf("%10.2f ", us);
      } else {
        std::printf("%10s ", "-");
      }
    }
    std::printf("  ");
    // Both ratios are kv4 against a denominator: fp16 for what the 4-bit cache
    // buys, tq4 for what the Cube buys at a fixed rate. A ratio needs both legs
    // to have produced a sample; a missing one prints a dash rather than an
    // infinity or a zero that would read as a measurement.
    const double denominators[] = {fp16_us, tq4_us};
    for (int column = 0; column < 2; ++column) {
      const char* tail = column == 1 ? "\n" : " ";
      if (denominators[column] > 0.0 && kv4_us > 0.0) {
        std::printf("%8.2fx%s", denominators[column] / kv4_us, tail);
      } else {
        std::printf("%9s%s", "-", tail);
      }
    }
  }
  if (!any) {
    std::printf("  (no decode case produced a pipelined sample)\n");
  }

  // The derived rates for the Cube 4-bit leg. Its GB/s is its own decode-read
  // model over its own measured time, so it is comparable to nothing else in
  // the file. The table is suppressed when the leg did not run: a header over
  // no rows says less than its absence does.
  bool any_cube = false;
  for (const TrafficModel& model : models) {
    any_cube = any_cube ||
               FindResult(runner, CaseName("kv4fp8_decode", model.context_len), TimingMode::kPipelined) != nullptr;
  }
  if (any_cube) {
    std::printf("\n  %6s  %10s %10s %10s\n", "S", "kv4 step/s", "kv4 GB/s", "kv4 TF/s");
    std::printf("  %s\n", std::string(6 + 2 + 3 * 10 + 2, '-').c_str());
    for (const TrafficModel& model : models) {
      const BenchmarkResult* result =
          FindResult(runner, CaseName("kv4fp8_decode", model.context_len), TimingMode::kPipelined);
      if (result == nullptr || result->latency.median_us <= 0.0) {
        continue;
      }
      std::printf("  %6lld  %10.0f %10.1f %10.3f\n", static_cast<long long>(model.context_len),
                  1.0e6 / result->latency.median_us, result->gigabytes_per_second(), result->tflops());
    }
  }

  std::printf("\n[ascend-bench]   steps/s is one sequence's decode step, not a batch. The GB/s column is the\n"
              "[ascend-bench]   kv4 traffic-model decode-read bytes over the measured time, so it is what the\n"
              "[ascend-bench]   kernel asked the memory system for and not a fill-rate ceiling. kv4:fp16 is\n"
              "[ascend-bench]   against the physical fp16 Cube decode in the same binary -- the same kernel\n"
              "[ascend-bench]   with the codec removed -- and not against an analytic model. kv4:tq4 is the\n"
              "[ascend-bench]   Cube against the vector cores at one rate: the two store identical nibbles\n"
              "[ascend-bench]   against identical thresholds, so the codec is held fixed across that ratio.\n");
  if (cube_enabled) {
    // Repeated here rather than only in the banner, because the summary is the
    // part that gets pasted into a report and the banner is not.
    std::printf("[ascend-bench]   PROVISIONAL: the score GEMM's [n, k] B operand does not reproduce the host\n"
                "[ascend-bench]   product, so the kv4 and fp16 legs compute a wrong score row. These are\n"
                "[ascend-bench]   timings of the right work, not of a verified kernel. See TURBOQUANT_TESTS.md\n"
                "[ascend-bench]   section 13.8.\n");
  } else {
    // A row of dashes in a table invites the reading that the hardware tried
    // and could not.
    std::printf("[ascend-bench]   the kv4 and fp16 columns are empty because the Cube legs were dropped\n"
                "[ascend-bench]   (VLLM_ASCEND_TQ_CUBE_WIP=0), not because they ran and produced nothing. This\n"
                "[ascend-bench]   is a 4-bit AIV run, and the only comparison in it is the traffic model.\n");
  }
  std::fflush(stdout);
}

}  // namespace

void BuildSuite(BenchmarkRunner& runner) {
  const std::vector<int64_t> context_lens = ContextLens();
  const bool cube = CubeEnabled();

  bool aiv_queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&aiv_queried);
  std::printf("[ascend-bench] TurboQuant decode: vector cores = %lld%s\n", static_cast<long long>(aiv_num),
              aiv_queried ? " (from aclGetDeviceCapability)" : " (assumed; the runtime declined to answer)");
  PrintCubeBanner(cube);
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
  std::vector<std::unique_ptr<Kv4Scenario>> kv4_scenarios;
  std::vector<std::unique_ptr<Fp16Scenario>> fp16_scenarios;
  scenarios.reserve(context_lens.size());
  for (const int64_t context_len : context_lens) {
    scenarios.emplace_back(new DecodeScenario(context_len));
    // Gated at construction and not only at launch: these allocate two more KV
    // caches of the same shape, which over the sweep is real HBM spent on a
    // path that is not going to run.
    if (cube) {
      kv4_scenarios.emplace_back(new Kv4Scenario(*scenarios.back(), context_len));
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
      // One Cube grid line, not one per leg: PlanCubeDecode does not take the
      // mode, so kv4fp8 and the fp16 baseline launch the same grid. Printing it
      // twice would suggest they could differ.
      const Kv4Scenario& cube_grid = *kv4_scenarios[index];
      std::printf("\n[ascend-bench] S=%lld: blocks_per_seq=%lld pool=%lld "
                  "tq4 splits=%lld grid(split)=%u | cube splits=%lld grid(split)=%u grid(combine)=%u\n",
                  static_cast<long long>(context_len), static_cast<long long>(scenario.blocks_per_seq()),
                  static_cast<long long>(scenario.num_blocks()),
                  static_cast<long long>(scenario.decode_grid().num_splits),
                  scenario.decode_grid().split_block_dim,
                  static_cast<long long>(cube_grid.decode_grid().num_splits),
                  cube_grid.decode_grid().split_block_dim, cube_grid.decode_grid().combine_block_dim);
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

    // --- the Cube legs ------------------------------------------------------
    //
    // On by default. A skip is recorded for each case when they are dropped, so
    // the absence shows up in the result table.
    if (!cube) {
      for (const char* dropped : {"kv4fp8_write", "kv4fp8_decode"}) {
        runner.Skip(CaseName(dropped, context_len), "Cube legs dropped by VLLM_ASCEND_TQ_CUBE_WIP=0");
      }
      runner.Skip(CaseName("fp16_decode", context_len),
                  "the fp16 baseline is the same Cube kernel with the codec removed, and is dropped with it");
      continue;
    }

    Kv4Scenario& kv4 = *kv4_scenarios[index];
    Fp16Scenario& fp16 = *fp16_scenarios[index];

    // --- the Cube-native 4-bit cache write ----------------------------------
    //
    // Same nibbles as tq4_write and a different kernel: this one is
    // turboquant_mm_reshape_and_cache at MODE = KV4_FP8, which encodes through
    // the multi-rate codec rather than through TurboQuantCodec<4>. The two
    // codebooks agree by construction -- TurboQuantModeTraits<KV4_FP8>'s
    // thresholds are TurboQuantCodec<4>::Threshold -- so the difference the
    // pair measures is the packer, not the quantiser.
    const std::string kv4_write_name = CaseName("kv4fp8_write", context_len);
    try {
      BenchmarkCase write_case;
      write_case.name = kv4_write_name;
      write_case.bytes_per_iteration = model.fp16_footprint_bytes + model.kv4_footprint_bytes +
                                       kFloatBytes * static_cast<double>(context_len);
      write_case.launch = [&kv4, &scenario](aclrtStream stream) { kv4.EnqueueWrite(scenario, stream); };
      write_case.checksum = [&kv4]() { return ChecksumSum(kv4.ScalePlane()); };
      runner.Run(write_case);
    } catch (const std::exception& error) {
      runner.RecordFailure(kv4_write_name, error.what());
    }

    // --- the Cube-native 4-bit decode ---------------------------------------
    const std::string kv4_decode_name = CaseName("kv4fp8_decode", context_len);
    try {
      kv4.FillCache(scenario, runner.stream());

      BenchmarkCase decode_case;
      decode_case.name = kv4_decode_name;
      decode_case.flops_per_iteration = AttentionFlops(context_len);
      decode_case.bytes_per_iteration = model.kv4_decode_read_bytes;
      decode_case.tasks_per_launch = 2;  // split, then combine
      decode_case.launch = [&kv4, &scenario](aclrtStream stream) { kv4.EnqueueDecode(scenario, stream); };
      decode_case.checksum = [&kv4]() { return ChecksumSum(kv4.Output()); };
      runner.Run(decode_case);
    } catch (const std::exception& error) {
      runner.RecordFailure(kv4_decode_name, error.what());
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
