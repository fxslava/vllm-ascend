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
//   tq_rotate_s<S>      the query rotation, q~ = Pi q, on its own. ONE launch,
//                       and the only single-launch decode-side case here. Both
//                       decode legs enqueue this same rotation ahead of their
//                       own kernels, so subtracting it is what prices the extra
//                       operator invocation the pre-rotated Q contract costs.
//                       Flat in S: it reads the query and nothing paged.
//   tq4_decode_s<S>     the decode: rotate, split, combine -- three launches on
//                       one stream. The cache is written once during setup, so
//                       the timed region is the read path only.
//   kv4fp8_write_s<S>   the 4-bit cache write through the multi-rate codec.
//   kv4fp8_decode_s<S>  the Cube-native decode: the same 128-byte slot tq4
//                       stores, unpacked to fp8_e4m3fn on the AIV and
//                       multiplied on the Cube.
//   fp16_decode_s<S>    the baseline: the same Cube decode over an unquantised
//                       fp16 paged KV cache.
//
// All five run by default. The comparison the binary exists to make is the
// three-way one at a fixed context length: tq4 against kv4fp8 is the Cube
// against the vector cores, and kv4fp8 against fp16 is what the 4-bit cache buys
// over an unquantised one.
//
// THE CODEC IS NO LONGER HELD FIXED ACROSS tq4 AND kv4fp8, and the first column
// has to be read with that in mind. Both still store 4 bits per coordinate in a
// 128-byte slot, so the cache geometry and every DMA volume match -- but tq4
// stores a Lloyd-Max index and expands it with two UB Gathers, while kv4fp8 now
// stores a symmetric INT4 level and expands it with integer shifts and an Adds
// (TURBOQUANT_TESTS.md section 13.9). So the ratio is the Cube path with the new
// expand against the vector path with the old one, which conflates two changes.
// Separating them would mean keeping a Lloyd-Max Cube leg alive purely as a
// benchmark control; that has not been done.
//
// kv3fp4 and kv5fp8 are still built and still dispatched by the sim tier; they
// have no leg here. Neither is on the 4-bit path this binary measures, and a
// column for each was three more cache allocations of the same shape for a
// comparison nothing was asking of this binary.
//
// WHAT IS AND IS NOT VERIFIED. The Cube-native decode is correct: kv4fp8
// reaches cos 0.982351 against an fp32 host reference on the arch35 camodel at
// S=64, with no device error of any kind, and test_sim_950pr_cube_gemm pins
// both B operand forms at max|err| = 0. The defects that made it wrong -- the
// V -> MTE3 staging hazard above all -- are closed; see TURBOQUANT_TESTS.md
// section 13.8.
//
// What is NOT verified is anything this file measures. No Ascend 950PR silicon
// has been available to this project, so every number below is the first of its
// kind, and the fidelity evidence above covers S=64 on a simulator rather than
// the S in {512, 1024, 2048} swept here. The fp16 baseline shares the corrected
// staging but has no fidelity check of its own: it is a comparator, and nothing
// asserts that it computes the right thing.
//
// TWO BASELINES, and they answer different questions.
//
//   fp16_decode   turboquant_fp16_decode_split in
//                 csrc/attention/turboquant/turboquant_mm_kernels.cpp: the same
//                 Cube decode with the codec removed. Same task decomposition,
//                 GQA batching, tile size, online softmax and split count, so
//                 the ratio to it measures the CODEC and nothing else.
//   fia_decode    the stock CANN operator, aclnnFusedInferAttentionScoreV5 with
//                 aclnnFusedInferAttentionScoreV2 as a fallback, over the same
//                 fp16 paged cache. The ratio to it is what a caller would
//                 actually gain by switching, since this is what they have
//                 today.
//
// The operator leg is BEST-EFFORT and expected to be absent on some builds: V1
// to V4 are withdrawn on an Ascend950 (planning returns 361001, measured on CANN
// 9.1.0) and V5's argument list has never been planned on hardware. Whichever
// way it goes, the outcome is recorded -- a latency if it planned, the operator
// name and the planning status if it did not -- so the comparison table says why
// a column is empty instead of omitting it.
//
// The traffic model is also printed. It is exact arithmetic over the two cache
// layouts, not a measurement.
//
// OUTPUT. Three things: the generic per-case table and its CSV
// (ASCEND_BENCH_CSV), this suite's own summary, and a shape-keyed comparison CSV
// written when ASCEND_BENCH_TQ_COMPARE_CSV names a path -- one row per shape with
// every leg's latency, the speedups, and the derived rates. That last one is the
// artifact meant for pasting into a report.
//
// BATCH. One decode step per launch, so the B column is 1 throughout. The batch
// axis is validated rather than timed, by
// test_device_950pr_turboquant_multimode, which sweeps batch {1, 8}; sweeping it
// here would multiply four context lengths of KV cache by the batch and is a
// separate change.
//
// SHAPES. Qwen3.5-2B's full-attention layer as common/ascend950_shapes.hpp
// records it: head_dim 256, 8 query heads over 2 kv heads, block_size 128.
//
//   ASCEND_BENCH_TQ_CONTEXTS=512,1024   restricts the sweep. Everything else is
//                                       the shared ASCEND_BENCH_* set; see
//                                       common/benchmark.hpp.
//   ASCEND_BENCH_TQ_COMPARE_CSV=<path>  writes the shape-keyed comparison CSV.
//   ASCEND_BENCH_TQ_FIA=0               drops the stock-operator leg, which is
//                                       the one leg whose argument list has never
//                                       been launched on a part.
//   VLLM_ASCEND_TQ_CUBE_WIP=0           drops the Cube legs, leaving the
//                                       AIV-only 4-bit path.
//
// THE OUTPUT BASIS. Every quantised decode leg's combine now writes the rotated
// output O~ -- the inverse rotation is folded into W_o in production -- so the
// tq4 and kv4fp8 checksums are over rotated bits. Neither leg times an
// un-rotation, which is what a folded layer runs; a layer that cannot fold pays
// one more rotate launch, the size of tq_rotate_s<S>.
//
// PREFILL. After the decode summary, a second suite times prefill across a
// dense sweep of shapes against aclnnFusedInferAttentionScoreV5 over an fp16
// cache: the TurboQuant write, the rotation of V a folded layer needs, the
// stock attention and the pipelines they make up, with device time from ACL
// events, GB/s and the KV-cache compression ratios. See THE PREFILL SUITE below
// for what it times, what it does not, and its own ASCEND_BENCH_TQ_PREFILL_*
// variables; ASCEND_BENCH_TQ_PREFILL=0 drops it.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

const char* kSuiteName =
    "turboquant_950pr (4-bit rotated KV cache: AIV, Cube, an fp16 Cube baseline, and prefill against FIA V5)";

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

// Context lengths, in tokens. The brief's sweep, with S=64 prepended so the
// timings start at the shape the Cube decode's fidelity was measured at on the
// camodel (TURBOQUANT_TESTS.md section 13.9) -- otherwise every timed shape is
// one whose numerics have never been checked anywhere.
const int64_t kDefaultContextLens[] = {64, 512, 1024, 2048};

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

/*
 * The stock-operator leg, on by default and droppable.
 *
 * It exists because the aclnn argument list for a paged fp16 decode has never
 * been planned on this part, let alone launched: V1..V4 are withdrawn on an
 * Ascend950 and V5's list is transcribed from a header. A refusal at plan time is
 * harmless -- the leg records a skip and the sweep continues -- but a plan that
 * succeeds and a launch that faults takes the stream down and every case queued
 * behind it, which is the same hazard the Cube switch exists for. So it gets the
 * same treatment: ASCEND_BENCH_TQ_FIA=0 drops it before anything is constructed.
 */
bool FiaEnabled() {
  const char* raw = std::getenv("ASCEND_BENCH_TQ_FIA");
  return raw == nullptr || *raw == 0 || std::string(raw) != "0";
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
// They are no longer provisional: the decode is measured correct on the
// camodel and the rates here are what the device verification runs.
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
                "[ascend-bench]   kv4fp8 is verified: cos 0.982351 against an fp32 host reference on the\n"
                "[ascend-bench]   arch35 camodel at S=64, no device errors. What that does NOT cover is the\n"
                "[ascend-bench]   context lengths swept here, nor the fp16 leg, which is a comparator with no\n"
                "[ascend-bench]   fidelity check of its own. Timings from this binary have never been taken on\n"
                "[ascend-bench]   silicon. VLLM_ASCEND_TQ_CUBE_WIP=0 drops the Cube legs.\n");
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
    // The query rotation's constant and its output. One of each, shared by every
    // quantised leg for the same reason the query itself is: a second copy is a
    // second chance for the legs to disagree.
    h16_ = DeviceBuffer::FromHost(tqh::Hadamard16Half(), kBenchmarkAlignBytes);
    query_rot_ = DeviceBuffer::Empty<float>(static_cast<size_t>(kQueryTokens * kNumHeads * kHeadSize),
                                            kBenchmarkAlignBytes);

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
    EnqueueRotate(stream);
    turboquant_paged_attention_impl(
        AscendType::FP16, stream, decode_grid_.split_block_dim, decode_grid_.combine_block_dim, query_rot_.get(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(),
        decode_tables_.get(), workspace_.get(), out_.get(), static_cast<uint32_t>(kQueryTokens),
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
  void* h16() const { return h16_.get(); }
  void* query_rot() const { return query_rot_.get(); }
  void* write_tables() const { return write_tables_.get(); }
  std::vector<float> RotatedQuery() const { return query_rot_.ToHost<float>(); }

  /*
   * Enqueue Pi q for this step. Every quantised leg calls it at the head of its
   * own EnqueueDecode, so the rotation is inside the timed region exactly where
   * it used to be when the split kernel did it -- the accounting across the
   * refactor is unchanged, and `vs fp16` still compares a rotated decode with an
   * unrotated one.
   */
  vllm_ascend::turboquant::RotateQPlan EnqueueRotate(aclrtStream stream) const {
    return tqh::RotateQuery(stream, AscendType::FP16, query_.get(), pi_signs_.get(), h16_.get(),
                            write_tables_.get(), query_rot_.get(), kQueryTokens, kNumHeads, kHeadSize, aiv_num_,
                            /*input_exact_in_half=*/true);
  }

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
  DeviceBuffer h16_, query_rot_;
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
    // TurboQuantCodec4; the mode ones are this rate's. For a codebook rate the
    // decode image carries the NZ permutation (nz_rows = kCubeTileRows) and the
    // write image does not; for an affine rate both are a single zero block,
    // because that codec reads no table and both arguments are ignored (see
    // turboquant_host::ModeTables).
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
    shared.EnqueueRotate(stream);
    turboquant_mm_decode_split_impl(
        static_cast<int32_t>(kMode), AscendType::FP16, stream, decode_grid_.split_block_dim, shared.query_rot(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), shared.block_tables(), shared.context_lens(),
        decode_tables_.get(), workspace_.get(),
        static_cast<uint32_t>(kQueryTokens), static_cast<uint32_t>(kNumHeads), static_cast<uint32_t>(kNumKvHeads),
        static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize),
        static_cast<uint32_t>(blocks_per_seq_), static_cast<uint32_t>(decode_grid_.num_splits),
        decode_grid_.split_tasks_per_core, kAttentionScale, kInvSqrtHeadSize);

    turboquant_paged_attention_combine_impl(AscendType::FP16, stream, decode_grid_.combine_block_dim,
                                            workspace_.get(), out_.get(), static_cast<uint32_t>(kQueryTokens),
                                            static_cast<uint32_t>(kNumHeads), static_cast<uint32_t>(kHeadSize),
                                            static_cast<uint32_t>(decode_grid_.num_splits),
                                            decode_grid_.combine_tasks_per_core);
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
// Behind the same switch as the quantised Cube leg, and it shares that leg's
// staging -- it is the same kernel with the codec removed, so it is not an
// independent control, and nothing asserts it computes the right thing.
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

  // For the stock-operator leg, which reads the same two caches rather than
  // allocating a third pair: the comparison is of the decode, so both sides have
  // to see the identical bytes through the identical paging.
  void* key_cache() const { return key_cache_.get(); }
  void* value_cache() const { return value_cache_.get(); }

 private:
  int64_t context_len_ = 0;
  int64_t blocks_per_seq_ = 0;
  DeviceBuffer key_cache_, value_cache_, workspace_, out_;
  tqh::CubeDecodeGrid decode_grid_;
};

/*
 * --- the stock-operator baseline --------------------------------------------
 *
 * aclnnFusedInferAttentionScore over the fp16 paged cache Fp16Scenario built,
 * with the plugin's own decode argument list (the DecodeOnly branch of
 * AscendAttentionBackendImpl._get_fia_params): layout TND, one entry in the key
 * and value tensor lists, the block table as a tensor, and the context length as
 * actualSeqLengthsKv.
 *
 * V5 FIRST, V2 AS A FALLBACK. V1 to V4 are withdrawn on an Ascend950 -- planning
 * returns 361001, measured -- and V5 is what replaces them, but V5's argument
 * list has never been planned on a part. Trying both and recording what happened
 * is the only honest arrangement: a build where neither plans still gets a row
 * in the table, with the operator name and the status in it.
 *
 * Planned once at construction rather than per launch. PlanAclnn does the
 * GetWorkspaceSize call and the workspace allocation up front, so Launch() is as
 * close to the kernel as the runtime allows -- which is the same treatment every
 * other leg gets, and without it this column would be measuring the aclnn
 * planner.
 *
 * The member order is load-bearing: the tensor wrappers must outlive the
 * PlannedOp that captured their handles, and the lists and the output tensor
 * must be constructed after the tensors and the buffers they point at.
 */
struct FiaScenario {
  FiaScenario(const Fp16Scenario& fp16, const DecodeScenario& shared, int64_t context_len)
      : out_(DeviceBuffer::Empty<Half>(static_cast<size_t>(kQueryTokens * kNumHeads * kHeadSize),
                                       kBenchmarkAlignBytes)),
        // softmaxLse is a required output even with softmaxLseFlag false; the
        // plugin allocates one element for it and discards it.
        lse_(DeviceBuffer::Empty<Half>(1, kBenchmarkAlignBytes)),
        key_flat_(s950::FiaKeyCacheView(shared.num_blocks(), kBlockSize, kNumKvHeads, kHeadSize), ACL_FLOAT16,
                  fp16.key_cache()),
        value_flat_(s950::FiaKeyCacheView(shared.num_blocks(), kBlockSize, kNumKvHeads, kHeadSize), ACL_FLOAT16,
                    fp16.value_cache()),
        key_list_({key_flat_.get()}),
        value_list_({value_flat_.get()}),
        query_tnd_({kQueryTokens, kNumHeads, kHeadSize}, ACL_FLOAT16, shared.query()),
        block_table_({kQueryTokens, shared.blocks_per_seq()}, ACL_INT32, shared.block_tables()),
        context_tensor_({kQueryTokens, kNumHeads, kHeadSize}, ACL_FLOAT16, out_.get()),
        lse_tensor_({1}, ACL_FLOAT16, lse_.get()),
        actual_seq_lengths_(std::vector<int64_t>{kQueryTokens}),
        actual_seq_lengths_kv_(std::vector<int64_t>{context_len}) {
    try {
      planned_.reset(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV5WorkspaceFn>(
          V5(), query_tnd_.get(), key_list_.get(), value_list_.get(), /*pse_shift=*/nullptr,
          /*atten_mask=*/nullptr, actual_seq_lengths_.get(), actual_seq_lengths_kv_.get(), /*deq_scale1=*/nullptr,
          /*quant_scale1=*/nullptr, /*deq_scale2=*/nullptr, /*quant_scale2=*/nullptr, /*quant_offset2=*/nullptr,
          /*antiquant_scale=*/nullptr, /*antiquant_offset=*/nullptr, block_table_.get(),
          /*query_padding_size=*/nullptr, /*kv_padding_size=*/nullptr, /*key_antiquant_scale=*/nullptr,
          /*key_antiquant_offset=*/nullptr, /*value_antiquant_scale=*/nullptr, /*value_antiquant_offset=*/nullptr,
          /*key_shared_prefix=*/nullptr, /*value_shared_prefix=*/nullptr, /*actual_shared_prefix_len=*/nullptr,
          /*query_rope=*/nullptr, /*key_rope=*/nullptr, /*key_rope_antiquant_scale=*/nullptr,
          /*dequant_scale_query=*/nullptr, /*learnable_sink=*/nullptr, /*q_start_idx=*/nullptr,
          /*kv_start_idx=*/nullptr, kNumHeads, static_cast<double>(kAttentionScale), s950::kFiaUnboundedTokens,
          s950::kFiaUnboundedTokens, const_cast<char*>(ops950::kFiaLayoutTnd), kNumKvHeads,
          s950::kFiaSparseModeNone, s950::kFiaInnerPreciseDefault, kBlockSize, /*antiquant_mode=*/0,
          /*softmax_lse_flag=*/false, /*key_antiquant_mode=*/0, /*value_antiquant_mode=*/0,
          s950::kFiaQueryQuantModeNone, s950::kFiaPseTypeDefault, context_tensor_.get(), lse_tensor_.get())));
      op_name_ = ops950::kFusedInferAttentionScoreV5;
      return;
    } catch (const AclError& error) {
      note_ = std::string(ops950::kFusedInferAttentionScoreV5) + ": " + error.what();
    }
    try {
      planned_.reset(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV2WorkspaceFn>(
          V2(), query_tnd_.get(), key_list_.get(), value_list_.get(), /*pse_shift=*/nullptr,
          /*atten_mask=*/nullptr, actual_seq_lengths_.get(), actual_seq_lengths_kv_.get(), /*deq_scale1=*/nullptr,
          /*quant_scale1=*/nullptr, /*deq_scale2=*/nullptr, /*quant_scale2=*/nullptr, /*quant_offset2=*/nullptr,
          /*antiquant_scale=*/nullptr, /*antiquant_offset=*/nullptr, block_table_.get(),
          /*query_padding_size=*/nullptr, /*kv_padding_size=*/nullptr, /*key_antiquant_scale=*/nullptr,
          /*key_antiquant_offset=*/nullptr, /*value_antiquant_scale=*/nullptr, /*value_antiquant_offset=*/nullptr,
          /*key_shared_prefix=*/nullptr, /*value_shared_prefix=*/nullptr, /*actual_shared_prefix_len=*/nullptr,
          kNumHeads, static_cast<double>(kAttentionScale), s950::kFiaUnboundedTokens, s950::kFiaUnboundedTokens,
          const_cast<char*>(ops950::kFiaLayoutTnd), kNumKvHeads, s950::kFiaSparseModeNone,
          s950::kFiaInnerPreciseDefault, kBlockSize, /*antiquant_mode=*/0, /*softmax_lse_flag=*/false,
          /*key_antiquant_mode=*/0, /*value_antiquant_mode=*/0, context_tensor_.get(), lse_tensor_.get())));
      op_name_ = ops950::kFusedInferAttentionScoreV2;
    } catch (const AclError& error) {
      note_ += std::string("; ") + ops950::kFusedInferAttentionScoreV2 + ": " + error.what();
    }
  }

  bool available() const { return planned_ != nullptr; }
  const std::string& note() const { return note_; }
  const std::string& op_name() const { return op_name_; }

  void EnqueueDecode(aclrtStream stream) const { planned_->Launch(stream); }
  std::vector<float> Output() const { return HalfToFloat(out_.ToHost<Half>()); }

 private:
  static const AclnnOp& V5() {
    static const AclnnOp op(ops950::kFusedInferAttentionScoreV5);
    return op;
  }
  static const AclnnOp& V2() {
    static const AclnnOp op(ops950::kFusedInferAttentionScoreV2);
    return op;
  }

  DeviceBuffer out_, lse_;
  AclnnTensor key_flat_, value_flat_;
  AclnnTensorList key_list_, value_list_;
  AclnnTensor query_tnd_, block_table_, context_tensor_, lse_tensor_;
  AclnnIntArray actual_seq_lengths_, actual_seq_lengths_kv_;
  std::unique_ptr<PlannedOp> planned_;
  std::string note_;
  std::string op_name_;
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

// One leg's numbers at one shape, or "did not run". Pulling the four figures
// out of the result table once, into a struct, is what keeps the printing and the
// CSV from disagreeing about which median they quoted.
struct LegSample {
  bool present = false;
  double median_us = 0.0;
  double p95_us = 0.0;
  double gigabytes_per_second = 0.0;
  double tflops = 0.0;
};

LegSample SampleFor(const BenchmarkRunner& runner, const char* leg, int64_t context_len) {
  LegSample sample;
  const BenchmarkResult* result = FindResult(runner, CaseName(leg, context_len), TimingMode::kPipelined);
  if (result == nullptr || result->latency.median_us <= 0.0) {
    return sample;
  }
  sample.present = true;
  sample.median_us = result->latency.median_us;
  sample.p95_us = result->latency.p95_us;
  sample.gigabytes_per_second = result->gigabytes_per_second();
  sample.tflops = result->tflops();
  return sample;
}

// numerator / denominator, or 0 when either leg produced no sample. Callers print
// a dash for 0 rather than a ratio, because a 0.00x would read as a measurement.
double Speedup(const LegSample& baseline, const LegSample& candidate) {
  if (!baseline.present || !candidate.present || candidate.median_us <= 0.0) {
    return 0.0;
  }
  return baseline.median_us / candidate.median_us;
}

void PrintMicroseconds(const LegSample& sample) {
  if (sample.present) {
    std::printf("%10.2f ", sample.median_us);
  } else {
    std::printf("%10s ", "-");
  }
}

void PrintRatio(double ratio, const char* tail) {
  if (ratio > 0.0) {
    std::printf("%9.2fx%s", ratio, tail);
  } else {
    std::printf("%10s%s", "-", tail);
  }
}

/*
 * The shape-keyed comparison CSV, written when ASCEND_BENCH_TQ_COMPARE_CSV names
 * a path. One row per shape, every leg in it, so a spreadsheet can plot the
 * sweep without reshaping anything.
 *
 * Separate from the generic ASCEND_BENCH_CSV, which is one row per (case, mode)
 * and knows nothing about which cases are baselines for which. This one is the
 * comparison; that one is the raw record.
 *
 * An absent leg leaves its fields EMPTY rather than writing 0 -- a zero latency
 * or a zero speedup in a spreadsheet is indistinguishable from a measurement,
 * and this is the artifact most likely to be read without its banner.
 */
void WriteComparisonCsv(const BenchmarkRunner& runner, const std::vector<TrafficModel>& models,
                        const std::string& path, const std::string& fia_operator) {
  std::ofstream csv(path, std::ios::trunc);
  if (!csv) {
    std::printf("[ascend-bench] could not open ASCEND_BENCH_TQ_COMPARE_CSV='%s' for writing\n", path.c_str());
    return;
  }
  csv << "batch,context_len,num_heads,head_dim,num_kv_heads,block_size,"
      << "kv4fp8_us,tq4_us,fp16_us,fia_us,fia_operator,"
      << "speedup_vs_fp16,speedup_vs_tq4,speedup_vs_fia,"
      << "kv4fp8_gbps,kv4fp8_tflops,fp16_gbps,fp16_tflops,fia_gbps,fia_tflops,"
      << "kv4fp8_p95_us,fp16_p95_us,fia_p95_us,kv4fp8_bytes_per_step,fp16_bytes_per_step\n";

  for (const TrafficModel& model : models) {
    const LegSample kv4 = SampleFor(runner, "kv4fp8_decode", model.context_len);
    const LegSample tq4 = SampleFor(runner, "tq4_decode", model.context_len);
    const LegSample fp16 = SampleFor(runner, "fp16_decode", model.context_len);
    const LegSample fia = SampleFor(runner, "fia_decode", model.context_len);

    csv << kQueryTokens << ',' << model.context_len << ',' << kNumHeads << ',' << kHeadSize << ','
        << kNumKvHeads << ',' << kBlockSize << ',';
    const LegSample legs[] = {kv4, tq4, fp16, fia};
    for (const LegSample& leg : legs) {
      if (leg.present) {
        csv << leg.median_us;
      }
      csv << ',';
    }
    csv << fia_operator << ',';
    const double ratios[] = {Speedup(fp16, kv4), Speedup(tq4, kv4), Speedup(fia, kv4)};
    for (const double ratio : ratios) {
      if (ratio > 0.0) {
        csv << ratio;
      }
      csv << ',';
    }
    const LegSample rate_legs[] = {kv4, fp16, fia};
    for (const LegSample& leg : rate_legs) {
      if (leg.present) {
        csv << leg.gigabytes_per_second << ',' << leg.tflops;
      } else {
        csv << ',';
      }
      csv << ',';
    }
    for (const LegSample& leg : rate_legs) {
      if (leg.present) {
        csv << leg.p95_us;
      }
      csv << ',';
    }
    csv << model.kv4_decode_read_bytes << ',' << model.fp16_decode_read_bytes << '\n';
  }
  std::printf("[ascend-bench] comparison CSV written to %s\n", path.c_str());
}

/*
 * The suite's own summary, printed after every case has run and before the
 * generic table. What it adds is the three things the generic table cannot know:
 * which legs are baselines for which, the speedups that follow, and the derived
 * rates side by side.
 *
 * Three tables. The first is the latency comparison the binary exists to make --
 * the AIV 4-bit path, the Cube 4-bit path, the unquantised fp16 Cube path and the
 * stock operator, at one shape per row. The second is the speedups read off it.
 * The third is the derived rates per leg.
 */
void PrintDecodeSummary(const BenchmarkRunner& runner, const std::vector<TrafficModel>& models,
                        bool cube_enabled, const std::string& fia_operator) {
  std::printf("\n[ascend-bench] TurboQuant decode summary (median of the pipelined mode)\n");
  std::printf("[ascend-bench] shape: B=%lld H=%lld D=%lld kv_heads=%lld block=%lld\n",
              static_cast<long long>(kQueryTokens), static_cast<long long>(kNumHeads),
              static_cast<long long>(kHeadSize), static_cast<long long>(kNumKvHeads),
              static_cast<long long>(kBlockSize));
  std::printf("\n  %6s  %10s %10s %10s %10s\n", "S", "kv4 us", "tq4 us", "fp16 us", "fia us");
  std::printf("  %s\n", std::string(6 + 2 + 4 * 11, '-').c_str());

  bool any = false;
  for (const TrafficModel& model : models) {
    const LegSample kv4 = SampleFor(runner, "kv4fp8_decode", model.context_len);
    const LegSample tq4 = SampleFor(runner, "tq4_decode", model.context_len);
    const LegSample fp16 = SampleFor(runner, "fp16_decode", model.context_len);
    const LegSample fia = SampleFor(runner, "fia_decode", model.context_len);
    if (!kv4.present && !tq4.present && !fp16.present && !fia.present) {
      continue;
    }
    any = true;
    std::printf("  %6lld  ", static_cast<long long>(model.context_len));
    const LegSample legs[] = {kv4, tq4, fp16, fia};
    for (const LegSample& leg : legs) {
      PrintMicroseconds(leg);
    }
    std::printf("\n");
  }
  if (!any) {
    std::printf("  (no decode case produced a pipelined sample)\n");
    std::fflush(stdout);
    return;
  }

  // The speedups. Every one is kv4 against a denominator, because kv4 is the
  // thing being proposed and the other three are what it would replace.
  std::printf("\n  %6s  %10s %10s %10s\n", "S", "vs fp16", "vs tq4", "vs fia");
  std::printf("  %s\n", std::string(6 + 2 + 3 * 11, '-').c_str());
  for (const TrafficModel& model : models) {
    const LegSample kv4 = SampleFor(runner, "kv4fp8_decode", model.context_len);
    const LegSample tq4 = SampleFor(runner, "tq4_decode", model.context_len);
    const LegSample fp16 = SampleFor(runner, "fp16_decode", model.context_len);
    const LegSample fia = SampleFor(runner, "fia_decode", model.context_len);
    if (!kv4.present) {
      continue;
    }
    std::printf("  %6lld  ", static_cast<long long>(model.context_len));
    PrintRatio(Speedup(fp16, kv4), " ");
    PrintRatio(Speedup(tq4, kv4), " ");
    PrintRatio(Speedup(fia, kv4), "\n");
  }

  // The derived rates, now for every leg that produced a sample rather than for
  // kv4 alone. Each leg's GB/s is its OWN traffic model over its own measured
  // time, so the column compares a leg to its own memory demand and not to
  // another leg's -- a 4-bit cache and an fp16 one do not read the same bytes,
  // which is the entire point of the change.
  std::printf("\n  %6s  %-8s %11s %10s %10s\n", "S", "leg", "steps/s", "GB/s", "TFLOP/s");
  std::printf("  %s\n", std::string(6 + 2 + 9 + 11 + 10 + 10 + 3, '-').c_str());
  const char* rate_legs[] = {"kv4fp8_decode", "tq4_decode", "fp16_decode", "fia_decode"};
  const char* rate_labels[] = {"kv4fp8", "tq4", "fp16", "fia"};
  for (const TrafficModel& model : models) {
    for (size_t leg = 0; leg < sizeof(rate_legs) / sizeof(rate_legs[0]); ++leg) {
      const LegSample sample = SampleFor(runner, rate_legs[leg], model.context_len);
      if (!sample.present) {
        continue;
      }
      std::printf("  %6lld  %-8s %11.0f %10.1f %10.3f\n", static_cast<long long>(model.context_len),
                  rate_labels[leg], 1.0e6 / sample.median_us, sample.gigabytes_per_second, sample.tflops);
    }
  }

  std::printf("\n[ascend-bench]   steps/s is one sequence's decode step, not a batch -- every row is B=1; the\n"
              "[ascend-bench]   batch axis is validated by test_device_950pr_turboquant_multimode and not timed\n"
              "[ascend-bench]   here. A GB/s column is that leg's traffic-model decode-read bytes over its own\n"
              "[ascend-bench]   measured time, so it is what the kernel asked the memory system for and not a\n"
              "[ascend-bench]   fill-rate ceiling.\n");
  std::printf("[ascend-bench]   vs fp16 is against the physical fp16 Cube decode in this binary -- the same\n"
              "[ascend-bench]   kernel with the codec removed -- so it isolates the codec. vs tq4 is the Cube\n"
              "[ascend-bench]   against the vector cores at 4 bits, and since kv4fp8 became affine INT4 while\n"
              "[ascend-bench]   tq4 stayed Lloyd-Max that ratio now moves two things at once; see the header.\n"
              "[ascend-bench]   vs fia is against the stock CANN operator, which is what a caller has today.\n");
  if (cube_enabled) {
    // Repeated here rather than only in the banner, because the summary is the
    // part that gets pasted into a report and the banner is not.
    std::printf("[ascend-bench]   the kv4 decode is verified correct at S=64 on the camodel; these are the\n"
                "[ascend-bench]   first timings ever taken of it, and at S in {512,1024,2048} its fidelity has\n"
                "[ascend-bench]   not been measured anywhere. The fp16 column is a comparator with no fidelity\n"
                "[ascend-bench]   check of its own. See TURBOQUANT_TESTS.md sections 13.8 and 13.9.\n");
  } else {
    // A row of dashes in a table invites the reading that the hardware tried
    // and could not.
    std::printf("[ascend-bench]   the kv4, fp16 and fia columns are empty because the Cube legs were dropped\n"
                "[ascend-bench]   (VLLM_ASCEND_TQ_CUBE_WIP=0), not because they ran and produced nothing. This\n"
                "[ascend-bench]   is a 4-bit AIV run, and the only comparison in it is the traffic model.\n");
  }
  std::fflush(stdout);

  const char* csv_path = std::getenv("ASCEND_BENCH_TQ_COMPARE_CSV");
  if (csv_path != nullptr && *csv_path != 0) {
    WriteComparisonCsv(runner, models, csv_path, fia_operator);
  }
}

// =============================================================================
// THE PREFILL SUITE
// =============================================================================
//
// What a prefill step costs over a TurboQuant KV cache, against the stock CANN
// operator over an unquantised one, across a dense sweep of shapes. It runs in
// its own BenchmarkRunner on the decode suite's stream: the shapes are orders of
// magnitude heavier than a decode step, so it has its own iteration budget,
// its own result table and its own CSVs. Failures are mirrored into the decode
// runner, so the process exit code still reflects them.
//
// A prefill over a TurboQuant cache is PrefillNoCache in
// vllm_ascend/attention/turboquant_v1.py: the cache write, then full-precision
// attention over the batch's own K/V -- nothing is read back from the 4-bit
// cache. So the quantised pipeline is the stock attention plus a different
// write, and for a layer whose o_proj is Pi-folded, plus a rotation of V:
//
//   pf_tq4_write            turboquant_reshape_and_cache over B*S tokens.
//   pf_rotate_v             npu_turboquant_rotate_q on V, [B*S, H_KV, D] -> fp32.
//   pf_fp16_write           aclnnScatterPaKvCache into an fp16 paged cache, the
//                           baseline's own write.
//   pf_fia_v5               aclnnFusedInferAttentionScoreV5 (V2 as a fallback),
//                           TND, sparse_mode 3 with the 2048x2048 causal mask --
//                           the argument list the plugin's prefill passes. fp16.
//   pf_fp16_pipeline        pf_fp16_write + pf_fia_v5.
//   pf_tq4_pipeline         pf_tq4_write + pf_fia_v5: a layer with an unfolded
//                           o_proj, whose prefill is unchanged.
//   pf_tq4_folded_pipeline  pf_tq4_write + pf_rotate_v + FIA V5 over V~ = Pi V:
//                           a folded layer, whose prefill has to hand o_proj the
//                           rotated basis the decode does.
//
// NOT TIMED, and said so rather than hidden: the folded pipeline's fp32 -> fp16
// cast of V~. The backend does it with a torch cast; this suite has no
// header-verified aclnnCast prototype, so V~ is also prepared on the host in
// fp16 and FIA reads that copy. The cast is one elementwise pass over
// B*S*H_KV*D values and is not in any number below.
//
// npu_fusion_attention (aclnnFlashAttentionScore) is NOT a leg. Its argument
// list could not be read out of a CANN header in this environment -- the 9.1.0
// toolkit image carries no aclnnop operator headers -- and a guessed prototype
// behind dlsym is undefined behaviour at launch, not a planning refusal. The
// FIA V5 prototype this suite uses was transcribed from
// aclnn_fused_infer_attention_score_v5.h; see common/aclnn_ops_950pr.hpp.
//
// THE SWEEP. The cartesian product of
//
//   S in {512, 1024, 2048, 4096, 8192}    ASCEND_BENCH_TQ_PREFILL_S
//   B in {1, 2, 4}                        ASCEND_BENCH_TQ_PREFILL_B
//   D in {128, 256}                       ASCEND_BENCH_TQ_PREFILL_D
//   H_Q in {32, 64, 128}                  ASCEND_BENCH_TQ_PREFILL_HQ
//   H_KV in {1, 2, 8}                     ASCEND_BENCH_TQ_PREFILL_HKV
//
// 270 shapes by default, each list overridable as a comma list. H_KV = 1 is the
// MQA form DeepSeek MLA's single latent cache takes when it is served through
// this operator -- a shape proxy, not the MLA kernel, which splits the rotary
// half of Q/K into query_rope/key_rope. H_KV in {2, 8} are the GQA shapes.
//
// A shape whose estimated footprint exceeds 85% of free HBM is skipped with the
// two numbers in the reason; the estimate does not know FIA's workspace, and
// allows one more attention output for it.
//
// BUDGET. ASCEND_BENCH_TQ_PREFILL_WARMUP (3), _ITERS (10) and _BATCH (1), NOT the
// shared ASCEND_BENCH_WARMUP / _ITERS / _BATCH: the largest shape attends over
// 32,768 tokens and the decode suite's 20 + 3 * 100 launches would take hours on
// it alone. Every timing mode ASCEND_BENCH_MODES selects still runs.
//
//   ASCEND_BENCH_TQ_PREFILL=0                 drops the suite.
//   ASCEND_BENCH_TQ_PREFILL_LEGS=a,b          runs only the named legs.
//   ASCEND_BENCH_TQ_PREFILL_CSV=<path>        the raw (case, mode) table.
//   ASCEND_BENCH_TQ_PREFILL_COMPARE_CSV=<p>   one row per shape, every leg.
//
// REPORTED per shape: device time from ACL events (the device-events mode's
// median, pipelined when that mode is not selected), GB/s over each leg's own
// traffic model, TFLOP/s for the attention legs, and the KV-cache footprint of
// TurboQuant 4-bit and of an 8-bit layout against fp16. The 8-bit figure is
// LAYOUT ARITHMETIC ONLY: one byte per coordinate plus the same scale plane.
// There is no 8-bit TurboQuant kernel -- TurboQuantCodec is instantiated for
// b = 4 alone -- so there is no 8-bit leg to time.
//
// One correctness check runs once, on the smallest shape, before any timing:
// FIA over V~, un-rotated on the host, must match FIA over V (cos > 0.9999).
// That is the device-level statement that the folded prefill hands o_proj the
// basis the fold expects.

namespace prefill {

// Every paged cache here is block_size 128, as the plugin's PrefillNoCache
// branch names it.
constexpr int64_t kBlockSize = s950::kDefaultBlockSize;
// FIA's compressed causal mask for sparse_mode 3 is always 2048 x 2048, whatever
// the sequence length; AttentionMaskBuilder.get_splitfuse_attn_mask builds the
// same int8 triu(ones, diagonal=1).
constexpr int64_t kCausalMaskSide = 2048;
constexpr int64_t kFiaSparseModeRightDownCausal = 3;
// FIA reads no paging in prefill, so its block size argument is torch_npu's 0.
constexpr int64_t kFiaNoPaging = 0;
// Tokens of distinct random data a scenario draws; longer contexts tile it.
// Attention latency does not depend on the values, and generating 1e9 fp16
// elements on the host for the largest query would cost minutes per shape.
constexpr int64_t kPatternTokens = 1024;
// Share of free HBM a shape may plan to occupy before it is skipped.
constexpr double kHbmBudget = 0.85;
// Elements a checksum reads back: enough to catch a nondeterministic launch
// without copying gigabytes to the host twice per case.
constexpr size_t kChecksumElements = 1u << 20;
constexpr double kFoldIdentityMinCosine = 0.9999;
constexpr int kDefaultWarmup = 3;
constexpr int kDefaultIterations = 10;
constexpr int kDefaultPipelineBatch = 1;

constexpr const char* kLegTq4Write = "pf_tq4_write";
constexpr const char* kLegRotateValue = "pf_rotate_v";
constexpr const char* kLegFp16Write = "pf_fp16_write";
constexpr const char* kLegFia = "pf_fia_v5";
constexpr const char* kLegFp16Pipeline = "pf_fp16_pipeline";
constexpr const char* kLegTq4Pipeline = "pf_tq4_pipeline";
constexpr const char* kLegTq4FoldedPipeline = "pf_tq4_folded_pipeline";
const char* const kAllLegs[] = {kLegTq4Write,     kLegRotateValue, kLegFp16Write,        kLegFia,
                                kLegFp16Pipeline, kLegTq4Pipeline, kLegTq4FoldedPipeline};

struct Shape {
  int64_t batch = 0;
  int64_t seq_len = 0;
  int64_t head_size = 0;
  int64_t num_heads = 0;
  int64_t num_kv_heads = 0;

  int64_t tokens() const { return batch * seq_len; }
  int64_t blocks_per_seq() const { return tqh::CeilDiv(seq_len, kBlockSize); }
  int64_t num_blocks() const { return batch * blocks_per_seq(); }
};

const char* Family(const Shape& shape) {
  if (shape.num_kv_heads == 1) {
    return "mqa/mla";
  }
  return shape.num_kv_heads == shape.num_heads ? "mha" : "gqa";
}

std::string CaseName(const char* leg, const Shape& shape) {
  std::ostringstream name;
  name << leg << "_b" << shape.batch << "_s" << shape.seq_len << "_d" << shape.head_size << "_hq" << shape.num_heads
       << "_kv" << shape.num_kv_heads;
  return name.str();
}

std::vector<int64_t> EnvList(const char* name, const std::vector<int64_t>& defaults) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return defaults;
  }
  std::vector<int64_t> values;
  std::istringstream stream(raw);
  std::string field;
  while (std::getline(stream, field, ',')) {
    const long long parsed = std::strtoll(field.c_str(), nullptr, 10);
    if (parsed > 0) {
      values.push_back(static_cast<int64_t>(parsed));
    }
  }
  if (values.empty()) {
    std::printf("[ascend-bench] %s='%s' parsed to nothing; using the default list\n", name, raw);
    return defaults;
  }
  return values;
}

int EnvInt(const char* name, int fallback, int minimum) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return fallback;
  }
  const long parsed = std::strtol(raw, nullptr, 10);
  return parsed < minimum ? minimum : static_cast<int>(parsed);
}

bool Enabled() {
  const char* raw = std::getenv("ASCEND_BENCH_TQ_PREFILL");
  return raw == nullptr || *raw == '\0' || std::string(raw) != "0";
}

bool LegEnabled(const char* leg) {
  const char* raw = std::getenv("ASCEND_BENCH_TQ_PREFILL_LEGS");
  if (raw == nullptr || *raw == '\0') {
    return true;
  }
  std::istringstream stream(raw);
  std::string field;
  while (std::getline(stream, field, ',')) {
    if (field == leg) {
      return true;
    }
  }
  return false;
}

BenchmarkOptions Options(const BenchmarkOptions& base) {
  BenchmarkOptions options = base;
  options.warmup_iterations = EnvInt("ASCEND_BENCH_TQ_PREFILL_WARMUP", kDefaultWarmup, 0);
  options.timed_iterations = EnvInt("ASCEND_BENCH_TQ_PREFILL_ITERS", kDefaultIterations, 1);
  options.pipeline_batch = EnvInt("ASCEND_BENCH_TQ_PREFILL_BATCH", kDefaultPipelineBatch, 1);
  const char* csv = std::getenv("ASCEND_BENCH_TQ_PREFILL_CSV");
  options.csv_path = csv == nullptr ? std::string() : std::string(csv);
  return options;
}

// Ordered by D, H_Q, H_KV, B, S, so consecutive shapes allocate similar sizes and
// the table reads as nested sweeps. Shapes the kernels reject are dropped here,
// with a line saying so, rather than failing one by one.
std::vector<Shape> Sweep() {
  const std::vector<int64_t> seq_lens = EnvList("ASCEND_BENCH_TQ_PREFILL_S", {512, 1024, 2048, 4096, 8192});
  const std::vector<int64_t> batches = EnvList("ASCEND_BENCH_TQ_PREFILL_B", {1, 2, 4});
  const std::vector<int64_t> head_sizes = EnvList("ASCEND_BENCH_TQ_PREFILL_D", {128, 256});
  const std::vector<int64_t> query_heads = EnvList("ASCEND_BENCH_TQ_PREFILL_HQ", {32, 64, 128});
  const std::vector<int64_t> kv_heads = EnvList("ASCEND_BENCH_TQ_PREFILL_HKV", {1, 2, 8});

  constexpr int64_t kMinHeadSize = 64;
  constexpr int64_t kMaxHeadSize = 256;
  std::vector<Shape> shapes;
  for (const int64_t d : head_sizes) {
    if (d < kMinHeadSize || d > kMaxHeadSize || (d & (d - 1)) != 0) {
      std::printf("[ascend-bench] prefill: dropping D=%lld; the TurboQuant kernels take a power of two in [64, 256]\n",
                  static_cast<long long>(d));
      continue;
    }
    for (const int64_t hq : query_heads) {
      for (const int64_t hkv : kv_heads) {
        if (hkv > hq || hq % hkv != 0) {
          std::printf("[ascend-bench] prefill: dropping H_Q=%lld H_KV=%lld; H_Q must be a multiple of H_KV\n",
                      static_cast<long long>(hq), static_cast<long long>(hkv));
          continue;
        }
        for (const int64_t b : batches) {
          for (const int64_t s : seq_lens) {
            Shape shape;
            shape.batch = b;
            shape.seq_len = s;
            shape.head_size = d;
            shape.num_heads = hq;
            shape.num_kv_heads = hkv;
            shapes.push_back(shape);
          }
        }
      }
    }
  }
  return shapes;
}

// --- traffic -----------------------------------------------------------------
//
// Bytes each leg is handed in and writes out, under the decode suite's
// convention: what the kernel asks the memory system for, not an internal
// working set nobody can see. FIA's own workspace is therefore not in it.
struct Traffic {
  double fp16_kv = 0.0;
  double tq4_kv = 0.0;
  double kv8_kv = 0.0;
  double tq4_write = 0.0;
  double rotate_value = 0.0;
  double fp16_write = 0.0;
  double fia = 0.0;
  double fp16_pipeline = 0.0;
  double tq4_pipeline = 0.0;
  double tq4_folded_pipeline = 0.0;
  // Causal attention, QK^T and the value accumulation at two operations per
  // element each: 4 * B * H_Q * D * S (S + 1) / 2. The decode suite's
  // AttentionFlops is the same count at one query token.
  double attention_flops = 0.0;

  double tq4_ratio() const { return tq4_kv > 0.0 ? fp16_kv / tq4_kv : 0.0; }
  double kv8_ratio() const { return kv8_kv > 0.0 ? fp16_kv / kv8_kv : 0.0; }
};

Traffic ModelTraffic(const Shape& shape) {
  const double t = static_cast<double>(shape.tokens());
  const double s = static_cast<double>(shape.seq_len);
  const double d = static_cast<double>(shape.head_size);
  const double hq = static_cast<double>(shape.num_heads);
  const double kv = static_cast<double>(shape.num_kv_heads);
  const double scale_plane = t * static_cast<double>(tqh::ScaleSlotFloats(shape.num_kv_heads)) * kFloatBytes;

  Traffic traffic;
  traffic.fp16_kv = 2.0 * t * kv * d * kHalfBytes;
  traffic.tq4_kv = 2.0 * t * kv * (d / static_cast<double>(tqh::kPackFactor)) + scale_plane;
  traffic.kv8_kv = 2.0 * t * kv * d + scale_plane;

  const double slots = t * kFloatBytes;
  const double query = t * hq * d * kHalfBytes;
  const double output = query;
  const double mask = static_cast<double>(kCausalMaskSide * kCausalMaskSide);
  traffic.tq4_write = traffic.fp16_kv + traffic.tq4_kv + slots;
  traffic.rotate_value = t * kv * d * (kHalfBytes + kFloatBytes);
  traffic.fp16_write = 2.0 * traffic.fp16_kv + slots;
  traffic.fia = query + traffic.fp16_kv + output + mask;
  traffic.fp16_pipeline = traffic.fp16_write + traffic.fia;
  traffic.tq4_pipeline = traffic.tq4_write + traffic.fia;
  traffic.tq4_folded_pipeline = traffic.tq4_write + traffic.rotate_value + traffic.fia;
  traffic.attention_flops = 4.0 * static_cast<double>(shape.batch) * hq * d * s * (s + 1.0) / 2.0;
  return traffic;
}

// HBM a scenario allocates, bar FIA's workspace, which is allowed one more
// attention output.
double ScenarioBytes(const Shape& shape) {
  const double t = static_cast<double>(shape.tokens());
  const double d = static_cast<double>(shape.head_size);
  const double hq = static_cast<double>(shape.num_heads);
  const double kv = static_cast<double>(shape.num_kv_heads);
  const double pool_rows = static_cast<double>(shape.num_blocks() * kBlockSize);
  const double attention_output = t * hq * d * kHalfBytes;
  const double activations = attention_output + 3.0 * t * kv * d * kHalfBytes + t * kv * d * kFloatBytes;
  const double tq4_cache = 2.0 * pool_rows * kv * (d / static_cast<double>(tqh::kPackFactor)) +
                           pool_rows * static_cast<double>(tqh::ScaleSlotFloats(shape.num_kv_heads)) * kFloatBytes;
  const double fp16_cache = 2.0 * pool_rows * kv * d * kHalfBytes;
  return activations + 3.0 * attention_output + tq4_cache + fp16_cache +
         static_cast<double>(kCausalMaskSide * kCausalMaskSide);
}

// --- device data -------------------------------------------------------------

// Fills `dst` by repeating `pattern` end to end. Whole-pattern copies, so the
// tiled rows stay aligned to token boundaries when the pattern is whole tokens.
template <typename T>
void TileToDevice(const DeviceBuffer& dst, const std::vector<T>& pattern) {
  const size_t total = dst.size_bytes();
  const size_t chunk = pattern.size() * sizeof(T);
  for (size_t offset = 0; offset < total && chunk > 0; offset += chunk) {
    const size_t bytes = std::min(chunk, total - offset);
    ACL_CHECK(aclrtMemcpy(static_cast<char*>(dst.get()) + offset, dst.capacity_bytes() - offset, pattern.data(),
                          bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  }
}

// Checksum over at most kChecksumElements leading elements of a buffer.
template <typename T>
std::vector<T> LeadingElements(const DeviceBuffer& buffer) {
  std::vector<T> host(std::min(buffer.size_bytes() / sizeof(T), kChecksumElements));
  if (!host.empty()) {
    buffer.CopyToHost(host.data(), host.size() * sizeof(T));
  }
  return host;
}

const AclnnOp& FiaV5() {
  static const AclnnOp op(ops950::kFusedInferAttentionScoreV5);
  return op;
}
const AclnnOp& FiaV2() {
  static const AclnnOp op(ops950::kFusedInferAttentionScoreV2);
  return op;
}
const AclnnOp& ScatterPaKvCache() {
  static const AclnnOp op(ops::kScatterPaKvCache);
  return op;
}

/*
 * One prefill shape on the device. Built, timed and destroyed before the next
 * shape is built, so the sweep's peak HBM is its largest shape's and not the
 * sum of all of them.
 *
 * The aclnn descriptors are unique_ptrs created in dependency order, and the
 * planned operators are declared last so they are destroyed first: a
 * PlannedOp holds the descriptor handles it was planned with.
 */
class Scenario {
 public:
  Scenario(const Shape& shape, int64_t aiv_num) : shape_(shape), aiv_num_(aiv_num) {
    const int64_t t = shape.tokens();
    const int64_t d = shape.head_size;
    const int64_t hq = shape.num_heads;
    const int64_t hkv = shape.num_kv_heads;
    const int64_t pattern_tokens = std::min<int64_t>(t, kPatternTokens);
    const auto elems = [](int64_t a, int64_t b, int64_t c) { return static_cast<size_t>(a * b * c); };

    DeterministicRandom rng(0x9F11u + static_cast<uint32_t>(shape.seq_len + 7 * shape.batch + 31 * d +
                                                              127 * hq + 509 * hkv));
    const std::vector<float> query = rng.NormalHalfExact(elems(pattern_tokens, hq, d), 0.0f, 1.0f);
    const std::vector<float> key = rng.NormalHalfExact(elems(pattern_tokens, hkv, d), 0.0f, 1.0f);
    std::vector<float> value = rng.NormalHalfExact(elems(pattern_tokens, hkv, d), 0.0f, 1.0f);

    query_ = DeviceBuffer::Empty<Half>(elems(t, hq, d), kBenchmarkAlignBytes);
    key_ = DeviceBuffer::Empty<Half>(elems(t, hkv, d), kBenchmarkAlignBytes);
    value_ = DeviceBuffer::Empty<Half>(elems(t, hkv, d), kBenchmarkAlignBytes);
    value_rot_half_ = DeviceBuffer::Empty<Half>(elems(t, hkv, d), kBenchmarkAlignBytes);
    TileToDevice(query_, FloatToHalf(query));
    TileToDevice(key_, FloatToHalf(key));
    TileToDevice(value_, FloatToHalf(value));
    // V~ = Pi V, per head, for the folded pipeline's attention; see the note on
    // the untimed cast at the top of this section.
    const std::vector<int8_t> signs = turboquant_ref::cpu_pi_sign_vector(static_cast<int>(d));
    for (size_t base = 0; base + static_cast<size_t>(d) <= value.size(); base += static_cast<size_t>(d)) {
      turboquant_ref::cpu_apply_pi(value.data() + base, static_cast<int>(d), signs.data());
    }
    TileToDevice(value_rot_half_, FloatToHalf(value));
    value_rot_fp32_ = DeviceBuffer::Empty<float>(elems(t, hkv, d), kBenchmarkAlignBytes);

    // Each sequence gets its own run of blocks out of a shuffled pool, so the
    // cache writes scatter the way a serving pool does.
    const int64_t blocks_per_seq = shape.blocks_per_seq();
    const std::vector<int32_t> pool = rng.Permutation(static_cast<int32_t>(shape.num_blocks()));
    std::vector<int32_t> slots(static_cast<size_t>(t));
    for (int64_t seq = 0; seq < shape.batch; ++seq) {
      for (int64_t pos = 0; pos < shape.seq_len; ++pos) {
        const int32_t block = pool[static_cast<size_t>(seq * blocks_per_seq + pos / kBlockSize)];
        slots[static_cast<size_t>(seq * shape.seq_len + pos)] =
            block * static_cast<int32_t>(kBlockSize) + static_cast<int32_t>(pos % kBlockSize);
      }
    }
    slots_ = DeviceBuffer::FromHost(slots, kBenchmarkAlignBytes);
    pi_signs_ = DeviceBuffer::FromHost(tqh::PiSigns(d), kBenchmarkAlignBytes);
    h16_ = DeviceBuffer::FromHost(tqh::Hadamard16Half(), kBenchmarkAlignBytes);
    write_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(d, 1), kBenchmarkAlignBytes);

    tq4_key_cache_ = DeviceBuffer::Empty<int8_t>(tqh::PackedCacheBytes(shape.num_blocks(), kBlockSize, hkv, d),
                                                 kBenchmarkAlignBytes);
    tq4_value_cache_ = DeviceBuffer::Empty<int8_t>(tq4_key_cache_.size_bytes(), kBenchmarkAlignBytes);
    scale_plane_ = DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(shape.num_blocks(), kBlockSize, hkv),
                                              kBenchmarkAlignBytes);
    fp16_key_cache_ = DeviceBuffer::Empty<Half>(elems(shape.num_blocks() * kBlockSize, hkv, d), kBenchmarkAlignBytes);
    fp16_value_cache_ = DeviceBuffer::Empty<Half>(fp16_key_cache_.size_bytes() / sizeof(Half), kBenchmarkAlignBytes);

    std::vector<int8_t> mask(static_cast<size_t>(kCausalMaskSide * kCausalMaskSide), 0);
    for (int64_t row = 0; row < kCausalMaskSide; ++row) {
      for (int64_t col = row + 1; col < kCausalMaskSide; ++col) {
        mask[static_cast<size_t>(row * kCausalMaskSide + col)] = 1;
      }
    }
    mask_ = DeviceBuffer::FromHost(mask, kBenchmarkAlignBytes);
    out_ = DeviceBuffer::Empty<Half>(elems(t, hq, d), kBenchmarkAlignBytes);
    out_rot_ = DeviceBuffer::Empty<Half>(elems(t, hq, d), kBenchmarkAlignBytes);
    lse_ = DeviceBuffer::Empty<Half>(1, kBenchmarkAlignBytes);
    lse_rot_ = DeviceBuffer::Empty<Half>(1, kBenchmarkAlignBytes);

    write_grid_ = tqh::PlanReshapeAndCache(t, aiv_num);

    // --- descriptors ---------------------------------------------------------
    query_tnd_.reset(new AclnnTensor({t, hq, d}, ACL_FLOAT16, query_.get()));
    key_tnd_.reset(new AclnnTensor({t, hkv, d}, ACL_FLOAT16, key_.get()));
    value_tnd_.reset(new AclnnTensor({t, hkv, d}, ACL_FLOAT16, value_.get()));
    value_rot_tnd_.reset(new AclnnTensor({t, hkv, d}, ACL_FLOAT16, value_rot_half_.get()));
    mask_tensor_.reset(new AclnnTensor({kCausalMaskSide, kCausalMaskSide}, ACL_INT8, mask_.get()));
    out_tensor_.reset(new AclnnTensor({t, hq, d}, ACL_FLOAT16, out_.get()));
    out_rot_tensor_.reset(new AclnnTensor({t, hq, d}, ACL_FLOAT16, out_rot_.get()));
    lse_tensor_.reset(new AclnnTensor({1}, ACL_FLOAT16, lse_.get()));
    lse_rot_tensor_.reset(new AclnnTensor({1}, ACL_FLOAT16, lse_rot_.get()));
    slots_tensor_.reset(new AclnnTensor({t}, ACL_INT32, slots_.get()));
    fp16_key_cache_tensor_.reset(
        new AclnnTensor({shape.num_blocks(), kBlockSize, hkv, d}, ACL_FLOAT16, fp16_key_cache_.get()));
    fp16_value_cache_tensor_.reset(
        new AclnnTensor({shape.num_blocks(), kBlockSize, hkv, d}, ACL_FLOAT16, fp16_value_cache_.get()));
    key_list_.reset(new AclnnTensorList({key_tnd_->get()}));
    value_list_.reset(new AclnnTensorList({value_tnd_->get()}));
    value_rot_list_.reset(new AclnnTensorList({value_rot_tnd_->get()}));
    // TND wants the cumulative token count at the end of each sequence, the
    // actual_seq_lengths_q the plugin passes for both Q and KV.
    std::vector<int64_t> cumulative(static_cast<size_t>(shape.batch));
    for (int64_t seq = 0; seq < shape.batch; ++seq) {
      cumulative[static_cast<size_t>(seq)] = (seq + 1) * shape.seq_len;
    }
    seq_lens_.reset(new AclnnIntArray(cumulative));

    // --- planning ------------------------------------------------------------
    fia_ = PlanFia(value_list_->get(), out_tensor_->get(), lse_tensor_->get(), &fia_op_, &fia_note_);
    fia_rot_ = PlanFia(value_rot_list_->get(), out_rot_tensor_->get(), lse_rot_tensor_->get(), &fia_op_, &fia_note_);
    try {
      fp16_write_.reset(new PlannedOp(PlanAclnn<ops::ScatterPaKvCacheWorkspaceFn>(
          ScatterPaKvCache(), key_tnd_->get(), fp16_key_cache_tensor_->get(), slots_tensor_->get(),
          value_tnd_->get(), fp16_value_cache_tensor_->get(), /*compress_lens=*/nullptr,
          /*compress_seq_offset=*/nullptr, /*seq_lens=*/nullptr, const_cast<char*>(ops::kScatterCacheModeNorm),
          /*scatter_mode=*/nullptr, /*strides=*/nullptr, /*offsets=*/nullptr)));
    } catch (const AclError& error) {
      fp16_write_note_ = std::string(ops::kScatterPaKvCache) + ": " + error.what();
    }
  }

  void EnqueueTq4Write(aclrtStream stream) const {
    turboquant_reshape_and_cache_impl(
        AscendType::FP16, stream, write_grid_.block_dim, key_.get(), value_.get(), tq4_key_cache_.get(),
        tq4_value_cache_.get(), scale_plane_.get(), slots_.get(), pi_signs_.get(), write_tables_.get(),
        static_cast<uint32_t>(shape_.tokens()), static_cast<uint32_t>(shape_.num_kv_heads),
        static_cast<uint32_t>(shape_.head_size), static_cast<uint32_t>(kBlockSize), write_grid_.tokens_per_core,
        1.0f / std::sqrt(static_cast<float>(shape_.head_size)));
  }

  // The same operator the backend launches on V for a folded layer. It is the
  // query rotation's kernel; Pi does not care which activation it is handed.
  void EnqueueRotateValue(aclrtStream stream) const {
    tqh::RotateQuery(stream, AscendType::FP16, value_.get(), pi_signs_.get(), h16_.get(), write_tables_.get(),
                     value_rot_fp32_.get(), shape_.tokens(), shape_.num_kv_heads, shape_.head_size, aiv_num_,
                     /*input_exact_in_half=*/true);
  }

  void EnqueueFp16Write(aclrtStream stream) const { fp16_write_->Launch(stream); }
  void EnqueueFia(aclrtStream stream) const { fia_->Launch(stream); }
  void EnqueueFiaRotated(aclrtStream stream) const { fia_rot_->Launch(stream); }

  bool fia_available() const { return fia_ != nullptr && fia_rot_ != nullptr; }
  bool fp16_write_available() const { return fp16_write_ != nullptr; }
  const std::string& fia_op() const { return fia_op_; }
  const std::string& fia_note() const { return fia_note_; }
  const std::string& fp16_write_note() const { return fp16_write_note_; }

  double ScaleChecksum() const { return ChecksumSum(LeadingElements<float>(scale_plane_)); }
  double RotatedValueChecksum() const { return ChecksumSum(LeadingElements<float>(value_rot_fp32_)); }
  double Fp16CacheChecksum() const { return ChecksumSum(HalfToFloat(LeadingElements<Half>(fp16_value_cache_))); }
  double OutputChecksum() const { return ChecksumSum(HalfToFloat(LeadingElements<Half>(out_))); }
  double RotatedOutputChecksum() const { return ChecksumSum(HalfToFloat(LeadingElements<Half>(out_rot_))); }

  std::vector<float> Output() const { return HalfToFloat(out_.ToHost<Half>()); }
  std::vector<float> RotatedOutput() const { return HalfToFloat(out_rot_.ToHost<Half>()); }

 private:
  std::unique_ptr<PlannedOp> PlanFia(const aclTensorList* value_list, const aclTensor* out, const aclTensor* lse,
                                     std::string* op_name, std::string* note) {
    const int64_t hq = shape_.num_heads;
    const int64_t hkv = shape_.num_kv_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(shape_.head_size));
    try {
      std::unique_ptr<PlannedOp> planned(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV5WorkspaceFn>(
          FiaV5(), query_tnd_->get(), key_list_->get(), value_list, /*pse_shift=*/nullptr, mask_tensor_->get(),
          seq_lens_->get(), seq_lens_->get(), /*deq_scale1=*/nullptr, /*quant_scale1=*/nullptr,
          /*deq_scale2=*/nullptr, /*quant_scale2=*/nullptr, /*quant_offset2=*/nullptr, /*antiquant_scale=*/nullptr,
          /*antiquant_offset=*/nullptr, /*block_table=*/nullptr, /*query_padding_size=*/nullptr,
          /*kv_padding_size=*/nullptr, /*key_antiquant_scale=*/nullptr, /*key_antiquant_offset=*/nullptr,
          /*value_antiquant_scale=*/nullptr, /*value_antiquant_offset=*/nullptr, /*key_shared_prefix=*/nullptr,
          /*value_shared_prefix=*/nullptr, /*actual_shared_prefix_len=*/nullptr, /*query_rope=*/nullptr,
          /*key_rope=*/nullptr, /*key_rope_antiquant_scale=*/nullptr, /*dequant_scale_query=*/nullptr,
          /*learnable_sink=*/nullptr, /*q_start_idx=*/nullptr, /*kv_start_idx=*/nullptr, hq, scale,
          s950::kFiaUnboundedTokens, s950::kFiaUnboundedTokens, const_cast<char*>(ops950::kFiaLayoutTnd), hkv,
          kFiaSparseModeRightDownCausal, s950::kFiaInnerPreciseDefault, kFiaNoPaging, /*antiquant_mode=*/0,
          /*softmax_lse_flag=*/false, /*key_antiquant_mode=*/0, /*value_antiquant_mode=*/0,
          s950::kFiaQueryQuantModeNone, s950::kFiaPseTypeDefault, out, lse)));
      *op_name = ops950::kFusedInferAttentionScoreV5;
      return planned;
    } catch (const AclError& error) {
      *note = std::string(ops950::kFusedInferAttentionScoreV5) + ": " + error.what();
    }
    try {
      std::unique_ptr<PlannedOp> planned(new PlannedOp(PlanAclnn<ops950::FusedInferAttentionScoreV2WorkspaceFn>(
          FiaV2(), query_tnd_->get(), key_list_->get(), value_list, /*pse_shift=*/nullptr, mask_tensor_->get(),
          seq_lens_->get(), seq_lens_->get(), /*deq_scale1=*/nullptr, /*quant_scale1=*/nullptr,
          /*deq_scale2=*/nullptr, /*quant_scale2=*/nullptr, /*quant_offset2=*/nullptr, /*antiquant_scale=*/nullptr,
          /*antiquant_offset=*/nullptr, /*block_table=*/nullptr, /*query_padding_size=*/nullptr,
          /*kv_padding_size=*/nullptr, /*key_antiquant_scale=*/nullptr, /*key_antiquant_offset=*/nullptr,
          /*value_antiquant_scale=*/nullptr, /*value_antiquant_offset=*/nullptr, /*key_shared_prefix=*/nullptr,
          /*value_shared_prefix=*/nullptr, /*actual_shared_prefix_len=*/nullptr, hq, scale,
          s950::kFiaUnboundedTokens, s950::kFiaUnboundedTokens, const_cast<char*>(ops950::kFiaLayoutTnd), hkv,
          kFiaSparseModeRightDownCausal, s950::kFiaInnerPreciseDefault, kFiaNoPaging, /*antiquant_mode=*/0,
          /*softmax_lse_flag=*/false, /*key_antiquant_mode=*/0, /*value_antiquant_mode=*/0, out, lse)));
      *op_name = ops950::kFusedInferAttentionScoreV2;
      return planned;
    } catch (const AclError& error) {
      *note += std::string("; ") + ops950::kFusedInferAttentionScoreV2 + ": " + error.what();
    }
    return nullptr;
  }

  Shape shape_;
  int64_t aiv_num_ = 0;
  tqh::ReshapeAndCacheGrid write_grid_;

  DeviceBuffer query_, key_, value_, value_rot_half_, value_rot_fp32_;
  DeviceBuffer slots_, pi_signs_, h16_, write_tables_;
  DeviceBuffer tq4_key_cache_, tq4_value_cache_, scale_plane_;
  DeviceBuffer fp16_key_cache_, fp16_value_cache_;
  DeviceBuffer mask_, out_, out_rot_, lse_, lse_rot_;

  std::unique_ptr<AclnnTensor> query_tnd_, key_tnd_, value_tnd_, value_rot_tnd_, mask_tensor_;
  std::unique_ptr<AclnnTensor> out_tensor_, out_rot_tensor_, lse_tensor_, lse_rot_tensor_;
  std::unique_ptr<AclnnTensor> slots_tensor_, fp16_key_cache_tensor_, fp16_value_cache_tensor_;
  std::unique_ptr<AclnnTensorList> key_list_, value_list_, value_rot_list_;
  std::unique_ptr<AclnnIntArray> seq_lens_;

  std::string fia_op_, fia_note_, fp16_write_note_;
  // Last, so destroyed first.
  std::unique_ptr<PlannedOp> fia_, fia_rot_, fp16_write_;
};

// --- reporting ---------------------------------------------------------------

struct Sample {
  bool present = false;
  double median_us = 0.0;
  double gigabytes_per_second = 0.0;
  double tflops = 0.0;
  const char* mode = "-";
};

// Device time: the device-events mode when it ran, pipelined otherwise.
Sample SampleFor(const BenchmarkRunner& runner, const char* leg, const Shape& shape) {
  Sample sample;
  const std::string name = CaseName(leg, shape);
  for (const TimingMode mode : {TimingMode::kDeviceEvents, TimingMode::kPipelined}) {
    const BenchmarkResult* result = FindResult(runner, name, mode);
    if (result != nullptr && result->latency.median_us > 0.0) {
      sample.present = true;
      sample.median_us = result->latency.median_us;
      sample.gigabytes_per_second = result->gigabytes_per_second();
      sample.tflops = result->tflops();
      sample.mode = TimingModeLabel(mode);
      return sample;
    }
  }
  return sample;
}

double Ratio(const Sample& numerator, const Sample& denominator) {
  return numerator.present && denominator.present && denominator.median_us > 0.0
             ? numerator.median_us / denominator.median_us
             : 0.0;
}

void PrintCompression(const std::vector<Shape>& sweep) {
  std::printf("\n[ascend-bench] prefill KV-cache footprint per token (layout arithmetic, both K and V)\n");
  std::printf("  %4s %4s  %12s %12s %12s  %9s %9s\n", "D", "H_KV", "fp16 B", "tq4 B", "kv8 B", "fp16/tq4",
              "fp16/kv8");
  std::vector<std::pair<int64_t, int64_t>> printed;
  for (const Shape& shape : sweep) {
    const std::pair<int64_t, int64_t> key(shape.head_size, shape.num_kv_heads);
    if (std::find(printed.begin(), printed.end(), key) != printed.end()) {
      continue;
    }
    printed.push_back(key);
    Shape one = shape;
    one.batch = 1;
    one.seq_len = 1;
    const Traffic traffic = ModelTraffic(one);
    std::printf("  %4lld %4lld  %12.0f %12.0f %12.0f  %8.2fx %8.2fx\n", static_cast<long long>(shape.head_size),
                static_cast<long long>(shape.num_kv_heads), traffic.fp16_kv, traffic.tq4_kv, traffic.kv8_kv,
                traffic.tq4_ratio(), traffic.kv8_ratio());
  }
  std::printf("[ascend-bench]   tq4 is the shipping 4-bit layout: D/2 packed bytes per vector plus a scale plane of\n"
              "[ascend-bench]   round_up(2 * H_KV, 8) fp32 per token, whose burst padding is why H_KV = 1 and 2 pay\n"
              "[ascend-bench]   more per head. kv8 is the same scale plane with a byte per coordinate. There is NO\n"
              "[ascend-bench]   8-bit TurboQuant kernel; that column is arithmetic, not a measurement.\n");
}

void WriteCompareCsv(const BenchmarkRunner& runner, const std::vector<Shape>& sweep, const std::string& path,
                     const std::string& fia_operator) {
  std::ofstream csv(path, std::ios::trunc);
  if (!csv) {
    std::printf("[ascend-bench] could not open ASCEND_BENCH_TQ_PREFILL_COMPARE_CSV='%s' for writing\n",
                path.c_str());
    return;
  }
  csv << "batch,seq_len,head_dim,num_heads,num_kv_heads,family,fia_operator";
  for (const char* leg : kAllLegs) {
    csv << ',' << leg << "_us," << leg << "_gbps";
  }
  csv << ",fia_tflops,tq4_pipeline_over_fp16,tq4_folded_over_fp16,fp16_kv_bytes,tq4_kv_bytes,kv8_kv_bytes,"
      << "fp16_over_tq4,fp16_over_kv8\n";
  for (const Shape& shape : sweep) {
    const Traffic traffic = ModelTraffic(shape);
    csv << shape.batch << ',' << shape.seq_len << ',' << shape.head_size << ',' << shape.num_heads << ','
        << shape.num_kv_heads << ',' << Family(shape) << ',' << fia_operator;
    for (const char* leg : kAllLegs) {
      const Sample sample = SampleFor(runner, leg, shape);
      csv << ',';
      if (sample.present) {
        csv << sample.median_us;
      }
      csv << ',';
      if (sample.present) {
        csv << sample.gigabytes_per_second;
      }
    }
    const Sample fia = SampleFor(runner, kLegFia, shape);
    const Sample fp16 = SampleFor(runner, kLegFp16Pipeline, shape);
    const double ratios[] = {Ratio(SampleFor(runner, kLegTq4Pipeline, shape), fp16),
                             Ratio(SampleFor(runner, kLegTq4FoldedPipeline, shape), fp16)};
    csv << ',';
    if (fia.present) {
      csv << fia.tflops;
    }
    for (const double ratio : ratios) {
      csv << ',';
      if (ratio > 0.0) {
        csv << ratio;
      }
    }
    csv << ',' << traffic.fp16_kv << ',' << traffic.tq4_kv << ',' << traffic.kv8_kv << ',' << traffic.tq4_ratio()
        << ',' << traffic.kv8_ratio() << '\n';
  }
  std::printf("[ascend-bench] prefill comparison CSV written to %s\n", path.c_str());
}

void PrintSummary(const BenchmarkRunner& runner, const std::vector<Shape>& sweep, const std::string& fia_operator) {
  std::printf("\n[ascend-bench] TurboQuant prefill summary: device time from ACL events, median us (stock operator: "
              "%s)\n",
              fia_operator.c_str());
  std::printf("  %2s %5s %4s %4s %3s %-7s | %10s %10s %10s %11s | %11s %11s %11s | %7s %7s | %8s %8s %8s\n", "B",
              "S", "D", "H_Q", "KV", "family", "tq4 write", "rotate V", "fp16 write", "fia v5", "fp16 pipe",
              "tq4 pipe", "tq4 folded", "tq4/fp", "fold/fp", "fia GB/s", "tq4 GB/s", "fia TF/s");
  bool any = false;
  for (const Shape& shape : sweep) {
    const Sample samples[] = {SampleFor(runner, kLegTq4Write, shape),     SampleFor(runner, kLegRotateValue, shape),
                              SampleFor(runner, kLegFp16Write, shape),    SampleFor(runner, kLegFia, shape),
                              SampleFor(runner, kLegFp16Pipeline, shape), SampleFor(runner, kLegTq4Pipeline, shape),
                              SampleFor(runner, kLegTq4FoldedPipeline, shape)};
    bool row = false;
    for (const Sample& sample : samples) {
      row = row || sample.present;
    }
    if (!row) {
      continue;
    }
    any = true;
    std::printf("  %2lld %5lld %4lld %4lld %3lld %-7s |", static_cast<long long>(shape.batch),
                static_cast<long long>(shape.seq_len), static_cast<long long>(shape.head_size),
                static_cast<long long>(shape.num_heads), static_cast<long long>(shape.num_kv_heads), Family(shape));
    const int widths[] = {10, 10, 10, 11, 11, 11, 11};
    for (size_t leg = 0; leg < sizeof(samples) / sizeof(samples[0]); ++leg) {
      if (samples[leg].present) {
        std::printf(" %*.1f", widths[leg], samples[leg].median_us);
      } else {
        std::printf(" %*s", widths[leg], "-");
      }
      if (leg == 3) {
        std::printf(" |");
      }
    }
    std::printf(" |");
    for (const double ratio : {Ratio(samples[5], samples[4]), Ratio(samples[6], samples[4])}) {
      if (ratio > 0.0) {
        std::printf(" %6.3fx", ratio);
      } else {
        std::printf(" %7s", "-");
      }
    }
    std::printf(" |");
    for (const double value : {samples[3].gigabytes_per_second, samples[5].gigabytes_per_second, samples[3].tflops}) {
      if (value > 0.0) {
        std::printf(" %8.2f", value);
      } else {
        std::printf(" %8s", "-");
      }
    }
    std::printf("\n");
  }
  if (!any) {
    std::printf("  (no prefill case produced a sample)\n");
  }
  std::printf("[ascend-bench]   tq4/fp and fold/fp are pipeline latency over the fp16 pipeline: above 1 is what the\n"
              "[ascend-bench]   4-bit cache costs a prefill, since both pipelines run the same fp16 attention and differ\n"
              "[ascend-bench]   in the write (and, folded, the rotation of V). The saving is in the cache footprint\n"
              "[ascend-bench]   below and in the decode suite, not here. GB/s is each leg's traffic model over its own\n"
              "[ascend-bench]   time; the folded pipeline's fp32 -> fp16 cast of V~ is not in any column.\n");
  PrintCompression(sweep);
  std::fflush(stdout);

  const char* path = std::getenv("ASCEND_BENCH_TQ_PREFILL_COMPARE_CSV");
  if (path != nullptr && *path != '\0') {
    WriteCompareCsv(runner, sweep, path, fia_operator);
  }
}

// FIA over V~ un-rotated on the host must be FIA over V: the fold identity at
// the attention output, on the device, before anything is timed.
double FoldedBasisCosine(const Scenario& scenario, const Shape& shape, aclrtStream stream) {
  scenario.EnqueueFia(stream);
  scenario.EnqueueFiaRotated(stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  const std::vector<float> plain = scenario.Output();
  const std::vector<float> folded = tqh::UnrotateHeads(scenario.RotatedOutput(), shape.head_size);
  return turboquant_ref::cpu_fidelity(folded, plain).cosine_similarity;
}

void Run(BenchmarkRunner& decode_runner, int64_t aiv_num) {
  if (!Enabled()) {
    std::printf("\n[ascend-bench] prefill suite dropped by ASCEND_BENCH_TQ_PREFILL=0\n");
    return;
  }
  const std::vector<Shape> sweep = Sweep();
  BenchmarkRunner runner(std::string(kSuiteName) + " -- prefill", Options(decode_runner.options()),
                         decode_runner.stream());
  std::printf("\n[ascend-bench] TurboQuant prefill suite: %zu shapes x %zu legs, warmup=%d iterations=%d "
              "pipeline_batch=%d per case\n",
              sweep.size(), sizeof(kAllLegs) / sizeof(kAllLegs[0]), runner.options().warmup_iterations,
              runner.options().timed_iterations, runner.options().pipeline_batch);
  std::fflush(stdout);

  const auto fail = [&](const std::string& name, const std::string& why) {
    runner.RecordFailure(name, why);
    decode_runner.RecordFailure(name, why);
  };
  const auto run_leg = [&](const char* leg, const Shape& shape, double flops, double bytes, int tasks,
                           std::function<void(aclrtStream)> launch, std::function<double()> checksum) {
    const std::string name = CaseName(leg, shape);
    if (!LegEnabled(leg)) {
      runner.Skip(name, "not in ASCEND_BENCH_TQ_PREFILL_LEGS");
      return;
    }
    try {
      BenchmarkCase benchmark_case;
      benchmark_case.name = name;
      benchmark_case.flops_per_iteration = flops;
      benchmark_case.bytes_per_iteration = bytes;
      benchmark_case.tasks_per_launch = tasks;
      benchmark_case.launch = std::move(launch);
      benchmark_case.checksum = std::move(checksum);
      runner.Run(benchmark_case);
    } catch (const std::exception& error) {
      fail(name, error.what());
    }
  };

  // The identity check runs on the smallest shape, where reading two attention
  // outputs back costs megabytes rather than gigabytes.
  size_t check_index = sweep.size();
  for (size_t index = 0; index < sweep.size(); ++index) {
    if (check_index == sweep.size() || sweep[index].tokens() * sweep[index].num_heads * sweep[index].head_size <
                                           sweep[check_index].tokens() * sweep[check_index].num_heads *
                                               sweep[check_index].head_size) {
      check_index = index;
    }
  }

  std::string fia_operator = "none";
  for (size_t index = 0; index < sweep.size(); ++index) {
    const Shape& shape = sweep[index];
    const Traffic traffic = ModelTraffic(shape);

    size_t free_hbm = 0;
    size_t total_hbm = 0;
    const bool known = aclrtGetMemInfo(ACL_HBM_MEM, &free_hbm, &total_hbm) == ACL_SUCCESS && free_hbm > 0;
    const double needed = ScenarioBytes(shape);
    if (known && needed > kHbmBudget * static_cast<double>(free_hbm)) {
      std::ostringstream why;
      why << "needs ~" << needed / (1024.0 * 1024.0 * 1024.0) << " GiB of HBM, " << kHbmBudget * 100.0 << "% of "
          << static_cast<double>(free_hbm) / (1024.0 * 1024.0 * 1024.0) << " GiB free is the budget";
      for (const char* leg : kAllLegs) {
        runner.Skip(CaseName(leg, shape), why.str());
      }
      continue;
    }

    std::unique_ptr<Scenario> scenario;
    try {
      scenario.reset(new Scenario(shape, aiv_num));
    } catch (const std::exception& error) {
      fail(CaseName("pf_setup", shape), error.what());
      continue;
    }
    std::printf("[ascend-bench] prefill B=%lld S=%lld D=%lld H_Q=%lld H_KV=%lld (%s): %s%s\n",
                static_cast<long long>(shape.batch), static_cast<long long>(shape.seq_len),
                static_cast<long long>(shape.head_size), static_cast<long long>(shape.num_heads),
                static_cast<long long>(shape.num_kv_heads), Family(shape),
                scenario->fia_available() ? scenario->fia_op().c_str() : "no stock operator -- ",
                scenario->fia_available() ? "" : scenario->fia_note().c_str());
    std::fflush(stdout);
    if (scenario->fia_available()) {
      fia_operator = scenario->fia_op();
    }

    if (index == check_index && scenario->fia_available()) {
      try {
        const double cosine = FoldedBasisCosine(*scenario, shape, runner.stream());
        std::printf("[ascend-bench]   folded-basis identity: FIA(V~) un-rotated vs FIA(V), cos = %.9f (bound %.4f)\n",
                    cosine, kFoldIdentityMinCosine);
        if (!(cosine > kFoldIdentityMinCosine)) {
          fail(CaseName("pf_fold_identity", shape), "FIA over Pi V does not un-rotate to FIA over V");
        }
      } catch (const std::exception& error) {
        fail(CaseName("pf_fold_identity", shape), error.what());
      }
    }

    const Scenario* sc = scenario.get();
    run_leg(kLegTq4Write, shape, 0.0, traffic.tq4_write, 1, [sc](aclrtStream s) { sc->EnqueueTq4Write(s); },
            [sc]() { return sc->ScaleChecksum(); });
    run_leg(kLegRotateValue, shape, 0.0, traffic.rotate_value, 1, [sc](aclrtStream s) { sc->EnqueueRotateValue(s); },
            [sc]() { return sc->RotatedValueChecksum(); });

    if (!sc->fp16_write_available()) {
      runner.Skip(CaseName(kLegFp16Write, shape), sc->fp16_write_note());
      runner.Skip(CaseName(kLegFp16Pipeline, shape), sc->fp16_write_note());
    } else {
      run_leg(kLegFp16Write, shape, 0.0, traffic.fp16_write, 1, [sc](aclrtStream s) { sc->EnqueueFp16Write(s); },
              [sc]() { return sc->Fp16CacheChecksum(); });
    }

    if (!sc->fia_available()) {
      for (const char* leg : {kLegFia, kLegFp16Pipeline, kLegTq4Pipeline, kLegTq4FoldedPipeline}) {
        runner.Skip(CaseName(leg, shape), sc->fia_note());
      }
      continue;
    }
    run_leg(kLegFia, shape, traffic.attention_flops, traffic.fia, 1, [sc](aclrtStream s) { sc->EnqueueFia(s); },
            [sc]() { return sc->OutputChecksum(); });
    if (sc->fp16_write_available()) {
      run_leg(kLegFp16Pipeline, shape, traffic.attention_flops, traffic.fp16_pipeline, 2,
              [sc](aclrtStream s) {
                sc->EnqueueFp16Write(s);
                sc->EnqueueFia(s);
              },
              [sc]() { return sc->OutputChecksum(); });
    }
    run_leg(kLegTq4Pipeline, shape, traffic.attention_flops, traffic.tq4_pipeline, 2,
            [sc](aclrtStream s) {
              sc->EnqueueTq4Write(s);
              sc->EnqueueFia(s);
            },
            [sc]() { return sc->OutputChecksum(); });
    run_leg(kLegTq4FoldedPipeline, shape, traffic.attention_flops, traffic.tq4_folded_pipeline, 3,
            [sc](aclrtStream s) {
              sc->EnqueueTq4Write(s);
              sc->EnqueueRotateValue(s);
              sc->EnqueueFiaRotated(s);
            },
            [sc]() { return sc->RotatedOutputChecksum(); });
  }

  runner.Report();
  PrintSummary(runner, sweep, fia_operator);
}

}  // namespace prefill

}  // namespace

void BuildSuite(BenchmarkRunner& runner) {
  const std::vector<int64_t> context_lens = ContextLens();
  const bool cube = CubeEnabled();
  const bool fia = cube && FiaEnabled();

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
  std::vector<std::unique_ptr<FiaScenario>> fia_scenarios;
  scenarios.reserve(context_lens.size());
  for (const int64_t context_len : context_lens) {
    scenarios.emplace_back(new DecodeScenario(context_len));
    // Gated at construction and not only at launch: these allocate two more KV
    // caches of the same shape, which over the sweep is real HBM spent on a
    // path that is not going to run.
    if (cube) {
      kv4_scenarios.emplace_back(new Kv4Scenario(*scenarios.back(), context_len));
      fp16_scenarios.emplace_back(new Fp16Scenario(*scenarios.back(), context_len));
      // No allocation of its own beyond one output: it reads the fp16 caches the
      // line above built. Constructing it plans the operator, which is where a
      // 361001 refusal surfaces, so the reason is available before any timing.
      //
      // An `if`, not an early `continue`: the traffic model is appended after
      // this block and every later loop indexes `models` by shape, so skipping
      // the rest of an iteration would leave that vector short and the
      // registration loop would read past its end.
      if (fia) {
        fia_scenarios.emplace_back(new FiaScenario(*fp16_scenarios.back(), *scenarios.back(), context_len));
        if (fia_scenarios.back()->available()) {
          std::printf("[ascend-bench] S=%lld: stock operator baseline is %s\n",
                      static_cast<long long>(context_len), fia_scenarios.back()->op_name().c_str());
        } else {
          std::printf("[ascend-bench] S=%lld: no stock operator baseline -- %s\n",
                      static_cast<long long>(context_len), fia_scenarios.back()->note().c_str());
        }
      }
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

    // --- the query rotation, on its own -------------------------------------
    //
    // ONE launch, and the only case in this file that is one. Both decode legs
    // below now enqueue this same rotation ahead of their own kernels, so
    //
    //     tq_rotate_sS  is  launch + the rotation itself
    //     tq4_decode_sS  minus  tq_rotate_sS  is what the decode costs without
    //
    // and the difference between this and an empty launch is what pricing the
    // extra operator invocation actually needs. That is the measurement the
    // decision to make the rotation a separate operator rests on, and nothing
    // in this repository had produced it before: the standalone Hadamard sweep
    // in hadamard_benchmark_results.csv records its launch path as
    // `replan-per-launch`, which is a harness artefact and not the graph-replay
    // figure a decode step actually pays.
    //
    // Flat in S by construction -- the rotation reads the query and nothing
    // paged -- so a row that grows with the context is a bug in this case and
    // not a property of the kernel.
    const std::string rotate_name = CaseName("tq_rotate", context_len);
    try {
      BenchmarkCase rotate_case;
      rotate_case.name = rotate_name;
      // Read: the fp16 query. Written: its fp32 rotation.
      rotate_case.bytes_per_iteration =
          static_cast<double>(kQueryTokens * kNumHeads * kHeadSize) * (sizeof(Half) + kFloatBytes);
      rotate_case.launch = [&scenario](aclrtStream stream) { scenario.EnqueueRotate(stream); };
      // Deterministic: the same query through the same transform every launch.
      rotate_case.checksum = [&scenario]() { return ChecksumSum(scenario.RotatedQuery()); };
      runner.Run(rotate_case);
    } catch (const std::exception& error) {
      runner.RecordFailure(rotate_name, error.what());
    }

    // --- the 4-bit AIV decode -----------------------------------------------
    const std::string decode_name = CaseName("tq4_decode", context_len);
    try {
      scenario.FillCache(runner.stream());

      BenchmarkCase decode_case;
      decode_case.name = decode_name;
      decode_case.flops_per_iteration = AttentionFlops(context_len);
      decode_case.bytes_per_iteration = model.tq4_decode_read_bytes;
      decode_case.tasks_per_launch = 3;  // rotate, split, then combine
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
      runner.Skip(CaseName("fia_decode", context_len),
                  "the stock-operator baseline reads the fp16 cache the Cube legs build, and is dropped with them");
      continue;
    }

    Kv4Scenario& kv4 = *kv4_scenarios[index];
    Fp16Scenario& fp16 = *fp16_scenarios[index];

    // --- the Cube-native 4-bit cache write ----------------------------------
    //
    // Four bits per coordinate into a 128-byte slot, as tq4_write, and a
    // different kernel AND a different quantiser: this one is
    // turboquant_mm_reshape_and_cache at MODE = KV4_FP8, which bins against
    // uniform thresholds and packs the two nibbles of a byte as coordinates b
    // and b + d/2. tq4_write bins against Lloyd-Max thresholds and interleaves
    // 2b with 2b + 1. The cache geometry is identical and the contents are not,
    // so the pair measures the packer and the quantiser together; see the note
    // on the codec at the head of this file.
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
      decode_case.tasks_per_launch = 3;  // rotate, split, then combine
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

    // --- the stock-operator baseline ----------------------------------------
    //
    // A SKIP and not a failure when the operator did not plan: an Ascend950 with
    // V1..V4 withdrawn and no V5 is a configuration, not a defect in this
    // binary, and the reason carries into the report.
    const std::string fia_name = CaseName("fia_decode", context_len);
    if (!fia) {
      runner.Skip(fia_name, "the stock-operator leg was dropped by ASCEND_BENCH_TQ_FIA=0");
    } else if (!fia_scenarios[index]->available()) {
      runner.Skip(fia_name, fia_scenarios[index]->note());
    } else {
      const FiaScenario& operator_leg = *fia_scenarios[index];
      try {
        BenchmarkCase fia_case;
        fia_case.name = fia_name;
        fia_case.flops_per_iteration = AttentionFlops(context_len);
        // The same bytes the fp16 leg reads, because it is the same cache. That
        // makes the two GB/s figures comparable and both of them a statement
        // about the traffic model rather than about the operator's internals.
        fia_case.bytes_per_iteration = model.fp16_decode_read_bytes;
        fia_case.tasks_per_launch = 1;  // one operator, one task
        fia_case.launch = [&operator_leg](aclrtStream stream) { operator_leg.EnqueueDecode(stream); };
        fia_case.checksum = [&operator_leg]() { return ChecksumSum(operator_leg.Output()); };
        runner.Run(fia_case);
      } catch (const std::exception& error) {
        runner.RecordFailure(fia_name, error.what());
      }
    }
  }

  // The operator name the CSV records. Every shape plans the same operator, so
  // the first scenario that resolved one names it; "none" when none did.
  std::string fia_operator = "none";
  for (const std::unique_ptr<FiaScenario>& operator_leg : fia_scenarios) {
    if (operator_leg->available()) {
      fia_operator = operator_leg->op_name();
      break;
    }
  }
  PrintDecodeSummary(runner, models, cube, fia_operator);

  // The decode scenarios hold four caches per context length; the prefill sweep
  // wants that HBM back. The operator legs read the fp16 legs' caches, so they
  // go first.
  fia_scenarios.clear();
  fp16_scenarios.clear();
  kv4_scenarios.clear();
  scenarios.clear();
  prefill::Run(runner, aiv_num);
}

}  // namespace bench
}  // namespace test
}  // namespace vllm_ascend
