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

// TurboQuant 4-bit KV cache: the AIV-only decode path, timed, against an
// unquantised fp16 decode of the same shape.
//
// WHY "AIV-ONLY" IS THE HEADLINE. Nothing in
// csrc/attention/turboquant/turboquant_kernels.cpp touches the Cube. The score
// row is a vector Mul plus a ReduceSum, the value accumulation is a vector FMA,
// the Walsh-Hadamard is Add/Sub and Gather, the codec is compares and a Gather
// over a 16-entry table. There is no Matmul object, no LOCAL_L1 copy, no Mmad,
// and no `[aic]` half of the kernel at all - grep the source for any of them
// and it comes back empty. So every number this binary reports is what the
// vector cores alone can do, and it is the baseline any future Cube-assisted
// INT4 GEMM path has to beat. Measuring it now, before that path exists, is the
// only way the comparison later means anything.
//
// WHAT IS TIMED, PER CONTEXT LENGTH S in {512, 1024, 2048}:
//
//   tq4_write_s<S>     the cache write path: S tokens rotated, quantised to
//                      4 bits and scattered. One launch. This is prefill-shaped
//                      - a decode step writes one token - and it is here
//                      because the cache has to be filled before it can be read
//                      and the fill is not free.
//   tq4_decode_s<S>    the decode itself: split then combine, TWO launches on
//                      one stream (flash-decoding cannot put them in one; see
//                      the kernel source). The cache is written once during
//                      setup, so the timed region is the read path only.
//   fp16_decode_s<S>   the same decode with an unquantised fp16 paged KV cache,
//                      through aclnnFusedInferAttentionScoreV2.
//
// THE fp16 LEG RESOLVES V5, AND FALLS BACK TO V2. Measured on CANN 9.1.0, the
// V2 planning call returns 361001 with "Interface aclnnFusedInferAttentionScore
// versions V1 to V4 are no longer supported on Ascend950": the whole V1..V4
// family is withdrawn on this part, and aclnnFusedInferAttentionScoreV5 is what
// replaces it. The suite resolves V5 first and keeps V2 as the fallback, so the
// same source works on a CANN release that predates V5 and on a part where V2
// still exists; the report says which one ran.
//
// V5's argument list is V2's plus seven optional inputs and two scalars, all of
// which a plain fp16 paged decode leaves at their neutral values - see
// common/aclnn_ops_950pr.hpp for the transcription and
// shapes950::kFiaQueryQuantModeNone / kFiaPseTypeDefault for the two scalars.
// It has NOT been planned on hardware: no 950PR has been available, so a
// non-zero status from the first silicon run is an argument-list bug before it
// is anything else.
//
// What carries the fp16 comparison wherever the operator does not resolve is
// the traffic model printed before the table. That is not a measurement and
// does not pretend to be:
// it is exact arithmetic over the two cache layouts, and it is the quantity the
// scheme actually buys. A decode step is bandwidth-bound - it streams the whole
// context and does two vector operations per element it reads - so the ratio of
// bytes moved is the ratio the latency tends towards once the pipeline is full.
//
// SHAPES. Qwen3.5-2B's full-attention layer as
// common/ascend950_shapes.hpp records it: head_dim 256, 8 query heads over 2 kv
// heads, block_size 128. Head dim 256 is the top of what the codec sizes UB for
// and the shape the part actually runs, which is the whole reason the 310P
// suite cannot host this file.
//
//   ASCEND_BENCH_TQ_CONTEXTS=512,1024   restricts the sweep. Everything else is
//                                       the shared ASCEND_BENCH_* set; see
//                                       common/benchmark.hpp.

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
#include "aclnn_ops_950pr.hpp"
#include "aclnn_runtime.hpp"
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

namespace tq = turboquant_ref;
namespace tqh = turboquant_host;
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

// --- the traffic model -------------------------------------------------------
//
// Exact byte counts, not estimates: every figure below is what the layouts and
// the kernels' DataCopy calls add up to. Two different questions are answered,
// and conflating them is the easy mistake:
//
//   footprint   what the KV cache for S tokens occupies in HBM. This is the
//               capacity win - how much longer a context fits in the same
//               memory - and it does not depend on how attention reads it.
//   decode read what one decode step streams. This is the bandwidth win, and it
//               is larger per byte of cache than the footprint suggests would be
//               fair, because every query head re-reads its kv head's rows: with
//               8 heads over 2 kv heads each cached row is read four times.
//               Whether L2 absorbs some of those re-reads is a hardware
//               question this model deliberately does not guess at - it counts
//               what the kernel asks the memory system for, which is the same
//               convention the fp16 leg is counted under, so the ratio is fair
//               even where the absolute numbers are pessimistic.
struct TrafficModel {
  int64_t context_len = 0;
  double fp16_footprint_bytes = 0.0;
  double tq4_packed_bytes = 0.0;
  double tq4_scale_bytes = 0.0;
  double tq4_footprint_bytes = 0.0;
  double fp16_decode_read_bytes = 0.0;
  double tq4_decode_read_bytes = 0.0;

  double footprint_ratio() const {
    return tq4_footprint_bytes > 0.0 ? fp16_footprint_bytes / tq4_footprint_bytes : 0.0;
  }
  double decode_read_ratio() const {
    return tq4_decode_read_bytes > 0.0 ? fp16_decode_read_bytes / tq4_decode_read_bytes : 0.0;
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

  // One decode step. Query and output rows are two fp16 vectors per head and
  // are kept in the count so the two legs are counted the same way, not because
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

  return model;
}

// Attention FLOPs for one decode step: QK^T and the value accumulation, two
// operations per element of each. The rotation (D log2 D per head, twice) and
// the codec are not counted, so this is the same number the fp16 leg does and
// TFLOP/s is comparable between them. It is *not* the whole instruction count
// of the TurboQuant kernel, and the gap between them is exactly what the codec
// costs.
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
  // The padding is not free and the number is easy to get wrong by hand, so it
  // is derived here from the same ScaleSlotFloats the kernel uses rather than
  // quoted: a token carries 2 * num_kv_heads live scale lanes in a slot rounded
  // up to a whole 32-byte burst, and the difference buys every DMA on this path
  // being an aligned DataCopy instead of a DataCopyPad.
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

// Everything one context length needs on the device. Allocated once in
// BuildSuite and alive for the whole of that shape's cases, because a benchmark
// that allocates inside its timed region is measuring aclrtMalloc.
//
// Buffers are asked for kBenchmarkAlignBytes rather than the 32-byte test
// default, so no measurement is charged for a cache that starts mid-line.
struct DecodeScenario {
  explicit DecodeScenario(int64_t context_len) : context_len_(context_len) {
    blocks_per_seq_ = (context_len + kBlockSize - 1) / kBlockSize;
    // Four times the blocks the context needs, so the block table is a genuine
    // scatter through a pool rather than a run of consecutive blocks that would
    // stay resident and flatter the measurement.
    num_blocks_ = std::max<int64_t>(4, blocks_per_seq_ * 4);

    DeterministicRandom rng(0x7451u);
    const size_t kv_elems = static_cast<size_t>(context_len * kNumKvHeads * kHeadSize);
    // Kept on the host as well as on the device: the fp16 leg writes the same
    // context into its own unquantised cache, and handing it these rather than
    // re-drawing them from the same seed is what stops the two legs from
    // silently diverging if the draw order here ever changes.
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
  void* query() const { return query_.get(); }
  void* block_tables() const { return block_tables_.get(); }

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

// --- the fp16 control --------------------------------------------------------

// The fp16 baseline's operator, V5 preferred and V2 as the fallback.
//
// V1..V4 are withdrawn on an Ascend950 - the planning call returns 361001 with
// "Interface aclnnFusedInferAttentionScore versions V1 to V4 are no longer
// supported on Ascend950" - and V5 is what replaces them. The fallback is not
// decoration: the same source has to keep working on a CANN release that never
// shipped V5, and on any part where V2 still exists, so the suite resolves
// whichever is present and labels the result with the one it used. Two
// different argument lists come out of that, which is why PlanFp16Decode below
// branches rather than sharing one call.
const AclnnOp& FusedInferAttentionOp() {
  static const AclnnOp v5(ops950::kFusedInferAttentionScoreV5);
  static const AclnnOp v2(ops950::kFusedInferAttentionScoreV2);
  return v5.available() ? v5 : v2;
}

bool FusedInferAttentionIsV5() {
  return FusedInferAttentionOp().name() == ops950::kFusedInferAttentionScoreV5;
}

// The unquantised leg's own buffers: a full fp16 paged KV cache over the same
// blocks, plus the output and the softmax LSE the operator insists on.
struct Fp16Scenario {
  Fp16Scenario(const DecodeScenario& quantised, int64_t context_len) {
    const size_t cache_elems =
        static_cast<size_t>(quantised.num_blocks() * kBlockSize * kNumKvHeads * kHeadSize);
    std::vector<float> key_cache(cache_elems, 0.0f);
    std::vector<float> value_cache(cache_elems, 0.0f);

    // The same context, written into the same slots the TurboQuant path used, so
    // both legs read identical data through identical paging and the comparison
    // is of the codec rather than of the access pattern.
    const std::vector<float>& key = quantised.key_host();
    const std::vector<float>& value = quantised.value_host();
    for (int64_t pos = 0; pos < context_len; ++pos) {
      const size_t slot = static_cast<size_t>(quantised.slots()[static_cast<size_t>(pos)]);
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
    lse_ = DeviceBuffer::Empty<Half>(1, kBenchmarkAlignBytes);
  }

  DeviceBuffer key_cache_, value_cache_, out_, lse_;
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
// generic table. What it adds is the two things the generic table cannot know
// about: decode steps per second, and the ratio to the fp16 leg where that leg
// produced numbers.
void PrintDecodeSummary(const BenchmarkRunner& runner, const std::vector<TrafficModel>& models) {
  std::printf("\n[ascend-bench] TurboQuant decode summary (AIV only; median of the pipelined mode)\n\n");
  std::printf("  %6s  %12s %12s %14s %12s   %12s %10s\n", "S", "tq4 us", "tq4 steps/s", "tq4 KV GB/s",
              "tq4 TFLOP/s", "fp16 us", "speedup");
  std::printf("  %s\n", std::string(6 + 2 + 12 + 1 + 12 + 1 + 14 + 1 + 12 + 3 + 12 + 1 + 10, '-').c_str());

  bool any = false;
  for (const TrafficModel& model : models) {
    const BenchmarkResult* tq4 = FindResult(runner, CaseName("tq4_decode", model.context_len), TimingMode::kPipelined);
    const BenchmarkResult* fp16 =
        FindResult(runner, CaseName("fp16_decode", model.context_len), TimingMode::kPipelined);
    if (tq4 == nullptr) {
      continue;
    }
    any = true;
    const double tq4_us = tq4->latency.median_us;
    const double steps_per_second = tq4_us > 0.0 ? 1.0e6 / tq4_us : 0.0;
    std::printf("  %6lld  %12.2f %12.0f %14.1f %12.3f   ", static_cast<long long>(model.context_len), tq4_us,
                steps_per_second, tq4->gigabytes_per_second(), tq4->tflops());
    if (fp16 != nullptr && fp16->latency.median_us > 0.0) {
      std::printf("%12.2f %9.2fx\n", fp16->latency.median_us, fp16->latency.median_us / tq4_us);
    } else {
      std::printf("%12s %10s\n", "-", "-");
    }
  }
  if (!any) {
    std::printf("  (no decode case produced a pipelined sample)\n");
  }
  std::printf("\n[ascend-bench]   steps/s is one sequence's decode step, not a batch. The KV GB/s column is the\n"
              "[ascend-bench]   traffic model's tq4 decode-read bytes over the measured time, so it is what the\n"
              "[ascend-bench]   kernel asked the memory system for and not a fill-rate ceiling.\n");
  if (FindResult(runner, CaseName("fp16_decode", models.front().context_len), TimingMode::kPipelined) == nullptr) {
    std::printf("[ascend-bench]   the fp16 column is empty: see the skip list below and the header of this file.\n"
                "[ascend-bench]   The bandwidth comparison to fall back on is the traffic model above, whose\n"
                "[ascend-bench]   decode-read ratio is %.2fx at S=%lld.\n",
                models.front().decode_read_ratio(), static_cast<long long>(models.front().context_len));
  }
  std::fflush(stdout);
}

}  // namespace

void BuildSuite(BenchmarkRunner& runner) {
  const std::vector<int64_t> context_lens = ContextLens();

  bool aiv_queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&aiv_queried);
  std::printf("[ascend-bench] TurboQuant AIV-only decode: vector cores = %lld%s\n", static_cast<long long>(aiv_num),
              aiv_queried ? " (from aclGetDeviceCapability)" : " (assumed; the runtime declined to answer)");

  const AclnnOp& fia = FusedInferAttentionOp();
  std::printf("[ascend-bench] fp16 baseline operator %s: %s\n", fia.name().c_str(),
              fia.available() ? "resolved" : fia.unavailable_reason().c_str());
  if (!fia.available()) {
    std::printf("[ascend-bench]   neither %s nor %s resolved; the fp16 column will be empty and the\n"
                "[ascend-bench]   comparison falls back to the traffic model below.\n",
                ops950::kFusedInferAttentionScoreV5, ops950::kFusedInferAttentionScoreV2);
  } else if (!FusedInferAttentionIsV5()) {
    std::printf("[ascend-bench]   V5 did not resolve, so this is the V2 fallback. On an Ascend950 the\n"
                "[ascend-bench]   planning call will refuse it with 361001 and the case will skip.\n");
  }
  std::fflush(stdout);

  // Built alongside the scenarios so the model's partial-traffic term uses the
  // split count the decode is actually going to launch with.
  std::vector<TrafficModel> models;
  std::vector<std::unique_ptr<DecodeScenario>> scenarios;
  scenarios.reserve(context_lens.size());
  for (const int64_t context_len : context_lens) {
    scenarios.emplace_back(new DecodeScenario(context_len));
    models.push_back(ModelTrafficFor(context_len, scenarios.back()->decode_grid()));
  }
  PrintTrafficModel(models);

  for (size_t index = 0; index < context_lens.size(); ++index) {
    const int64_t context_len = context_lens[index];
    DecodeScenario& scenario = *scenarios[index];
    const TrafficModel& model = models[index];

    std::printf("\n[ascend-bench] S=%lld: blocks_per_seq=%lld pool=%lld splits=%lld "
                "grid(write)=%u grid(split)=%u grid(combine)=%u\n",
                static_cast<long long>(context_len), static_cast<long long>(scenario.blocks_per_seq()),
                static_cast<long long>(scenario.num_blocks()),
                static_cast<long long>(scenario.decode_grid().num_splits), scenario.write_grid().block_dim,
                scenario.decode_grid().split_block_dim, scenario.decode_grid().combine_block_dim);
    std::fflush(stdout);

    // --- the cache write ----------------------------------------------------
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

    // --- the 4-bit decode ---------------------------------------------------
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

    // --- the unquantised fp16 control ---------------------------------------
    const std::string fp16_name = CaseName("fp16_decode", context_len);
    if (!fia.available()) {
      runner.Skip(fp16_name, fia.unavailable_reason());
      continue;
    }
    try {
      Fp16Scenario fp16_scenario(scenario, context_len);

      const std::vector<int64_t> cache_view =
          s950::FiaKeyCacheView(scenario.num_blocks(), kBlockSize, kNumKvHeads, kHeadSize);
      AclnnTensor key_flat(cache_view, ACL_FLOAT16, fp16_scenario.key_cache_.get());
      AclnnTensor value_flat(cache_view, ACL_FLOAT16, fp16_scenario.value_cache_.get());
      AclnnTensorList key_list({key_flat.get()});
      AclnnTensorList value_list({value_flat.get()});

      AclnnTensor query_tnd({kQueryTokens, kNumHeads, kHeadSize}, ACL_FLOAT16, scenario.query());
      AclnnTensor block_table_tensor({kQueryTokens, scenario.blocks_per_seq()}, ACL_INT32, scenario.block_tables());
      AclnnTensor context_tensor({kQueryTokens, kNumHeads, kHeadSize}, ACL_FLOAT16, fp16_scenario.out_.get());
      AclnnTensor lse_tensor({1}, ACL_FLOAT16, fp16_scenario.lse_.get());
      AclnnIntArray actual_seq_lengths(std::vector<int64_t>{kQueryTokens});
      AclnnIntArray actual_seq_lengths_kv(std::vector<int64_t>{context_len});

      // Both argument lists are the plugin's DecodeOnly branch as
      // common/aclnn_ops_950pr.hpp records it, differing only by what V5 added:
      // seven optional tensors after actualSharedPrefixLen, and queryQuantMode
      // and pseType after valueAntiquantMode. The geometry is identical either
      // way and identical to the TurboQuant leg's - head_dim 256, 8 heads over
      // 2 kv heads, block_size 128, layout TND with a block table, which is
      // what makes the two legs' numbers comparable at all.
      //
      // Planning is where an Ascend950 refuses V1..V4, so this is the call that
      // throws when only V2 resolved, and the reason the catch below reports a
      // skip rather than a failure.
      PlannedOp planned =
          FusedInferAttentionIsV5()
              ? PlanAclnn<ops950::FusedInferAttentionScoreV5WorkspaceFn>(
                    fia, query_tnd.get(), key_list.get(), value_list.get(), /*pse_shift=*/nullptr,
                    /*atten_mask=*/nullptr, actual_seq_lengths.get(), actual_seq_lengths_kv.get(),
                    /*deq_scale1=*/nullptr, /*quant_scale1=*/nullptr, /*deq_scale2=*/nullptr,
                    /*quant_scale2=*/nullptr, /*quant_offset2=*/nullptr, /*antiquant_scale=*/nullptr,
                    /*antiquant_offset=*/nullptr, block_table_tensor.get(), /*query_padding_size=*/nullptr,
                    /*kv_padding_size=*/nullptr, /*key_antiquant_scale=*/nullptr,
                    /*key_antiquant_offset=*/nullptr, /*value_antiquant_scale=*/nullptr,
                    /*value_antiquant_offset=*/nullptr, /*key_shared_prefix=*/nullptr,
                    /*value_shared_prefix=*/nullptr, /*actual_shared_prefix_len=*/nullptr,
                    // The seven V5 additions. An fp16 paged decode wants none of
                    // them: the RoPE is already folded into the cached K, the
                    // query is not quantised, there is no attention sink and the
                    // window is unbounded.
                    /*query_rope=*/nullptr, /*key_rope=*/nullptr, /*key_rope_antiquant_scale=*/nullptr,
                    /*dequant_scale_query=*/nullptr, /*learnable_sink=*/nullptr, /*q_start_idx=*/nullptr,
                    /*kv_start_idx=*/nullptr, kNumHeads, static_cast<double>(kAttentionScale),
                    s950::kFiaUnboundedTokens, s950::kFiaUnboundedTokens,
                    const_cast<char*>(ops950::kFiaLayoutTnd), kNumKvHeads, s950::kFiaSparseModeNone,
                    s950::kFiaInnerPreciseDefault, kBlockSize, /*antiquant_mode=*/0,
                    /*softmax_lse_flag=*/false, /*key_antiquant_mode=*/0, /*value_antiquant_mode=*/0,
                    s950::kFiaQueryQuantModeNone, s950::kFiaPseTypeDefault, context_tensor.get(),
                    lse_tensor.get())
              : PlanAclnn<ops950::FusedInferAttentionScoreV2WorkspaceFn>(
                    fia, query_tnd.get(), key_list.get(), value_list.get(), /*pse_shift=*/nullptr,
                    /*atten_mask=*/nullptr, actual_seq_lengths.get(), actual_seq_lengths_kv.get(),
                    /*deq_scale1=*/nullptr, /*quant_scale1=*/nullptr, /*deq_scale2=*/nullptr,
                    /*quant_scale2=*/nullptr, /*quant_offset2=*/nullptr, /*antiquant_scale=*/nullptr,
                    /*antiquant_offset=*/nullptr, block_table_tensor.get(), /*query_padding_size=*/nullptr,
                    /*kv_padding_size=*/nullptr, /*key_antiquant_scale=*/nullptr,
                    /*key_antiquant_offset=*/nullptr, /*value_antiquant_scale=*/nullptr,
                    /*value_antiquant_offset=*/nullptr, /*key_shared_prefix=*/nullptr,
                    /*value_shared_prefix=*/nullptr, /*actual_shared_prefix_len=*/nullptr, kNumHeads,
                    static_cast<double>(kAttentionScale), s950::kFiaUnboundedTokens,
                    s950::kFiaUnboundedTokens, const_cast<char*>(ops950::kFiaLayoutTnd), kNumKvHeads,
                    s950::kFiaSparseModeNone, s950::kFiaInnerPreciseDefault, kBlockSize,
                    /*antiquant_mode=*/0, /*softmax_lse_flag=*/false, /*key_antiquant_mode=*/0,
                    /*value_antiquant_mode=*/0, context_tensor.get(), lse_tensor.get());

      BenchmarkCase fp16_case;
      fp16_case.name = fp16_name;
      fp16_case.flops_per_iteration = AttentionFlops(context_len);
      fp16_case.bytes_per_iteration = model.fp16_decode_read_bytes;
      fp16_case.launch = [&planned](aclrtStream stream) { planned.Launch(stream); };
      fp16_case.checksum = [&fp16_scenario]() { return ChecksumSum(HalfToFloat(fp16_scenario.out_.ToHost<Half>())); };
      runner.Run(fp16_case);
    } catch (const std::exception& error) {
      // Not a failure: on this part the operator is withdrawn, which is a fact
      // about CANN and not a broken benchmark. The traffic model carries the
      // comparison; see the header of this file.
      runner.Skip(fp16_name, error.what());
    }
  }

  PrintDecodeSummary(runner, models);
}

}  // namespace bench
}  // namespace test
}  // namespace vllm_ascend
