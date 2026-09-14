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

// The Cube-native kv4fp8 decode split, cut at each stage of its tile pipeline
// and timed, so the kernel's latency reads as a sum of parts.
//
// The ladder is DecodeAblationStage in attention/turboquant/turboquant_mode.h.
// Every rung runs everything the rung below it does plus one more piece; the
// cut is a template parameter on TurboQuantCubeDecodeSplit, not a runtime
// branch; and the top rung is the shipping split kernel itself. A rung's
// latency minus its neighbour's is therefore what that piece costs in situ,
// including what it does to the overlap between the vector cores, the DMA pipes
// and the Cube -- which a per-operator microbenchmark cannot show.
//
//   stage0_mte2         packed K/V and scale tiles, GM -> UB (CopyInTile)
//   stage1_unpack       + UnpackAffine onto the fp8 grid, in UB
//   stage2_query_prep   + the query: pre-rotated GM read, amax, operand cast
//   stage3_l1_staging   + every V -> MTE3 edge and UB -> L1 copy
//   stage4_score_gemm   + MTE1 Load2D, Q . K^T Mmad, Fixpipe to UB, handshake
//   stage5_full         + online softmax, P . V GEMM, accumulator, GM writeback
//
// Per head size D in {256, 512} and context S in {64, 512, 1024, 2048}, at
// Qwen3.5-2B's 8 query and 2 kv heads and block size 128. One launch per
// iteration: the split alone.
//
// READ THESE BEFORE THE NUMBERS.
//
//   * Stage 2 is per TASK, not per tile. The cache is written already rotated,
//     so the split's only Walsh-Hadamard is the query's -- once per query head
//     of each kv head -- and its delta should not grow with S. There is no
//     output inverse rotation on the device at all any more: it is folded into
//     W_o, and the combine the ladder does not time only reduces. The delta
//     also carries the query's amax, scale and fp8 cast.
//   * Stages 0 to 2 end each tile read or unpack pass with PipeBarrier<PIPE_ALL>
//     where production has a HardEvent edge, because at those cuts the edge's
//     wait would have no consumer on its pipe. Stage 1 swaps stage 0's barrier
//     for the MTE2 -> V edge its unpack consumes; stage 3 swaps stage 2's for
//     production's V -> MTE3 per chunk plus MTE3 -> V. So stage 3's delta is
//     the copies and the difference in synchronisation, not the copies alone.
//   * Stages 0 to 4 write nothing to GM. Each runs against a freshly zeroed
//     workspace, and a case fails if the workspace is not still zero after it:
//     a leaked writeback would mean the rung is not the cut it claims to be.
//   * D = 512 is outside what the production adapter accepts (CheckHeadSize in
//     turboquant_torch_adpt.h caps head_size at 256) and has never run
//     anywhere. By static count the split's UB buffers come to 234,624 B of the
//     part's 253,952. So straight after each shape's stage 5 the combine runs
//     once, untimed, and the output is held to a cosine against an fp32 host
//     attention; below kFidelityCos the shape is marked UNTRUSTED and the suite
//     fails. The timings of a kernel that is aliasing UB are not timings of
//     this kernel.
//
// HANG GUARD. The shared harness synchronises without a deadline, so a cut
// kernel that deadlocks the part would hang the process with nothing printed.
// Each stage's first launch is therefore issued here and synchronised with
// aclrtSynchronizeStreamWithTimeout before the harness sees the case. A stage
// that misses the deadline is named, the waterfall so far is printed, and the
// process ends with kHangExitCode without the shared report or any teardown:
// a deadlocked stream does not come back, and destroying it blocks as well.
//
// Command line, from bench_main_950pr_ablation.cpp:
//
//   --stage=4                 run only these stages, in this order; one stage
//                             per process is the isolation for a suspect stage
//   --sync-timeout-ms=30000   deadline for each stage's first launch; 0 waits
//                             forever
//
// Environment, on top of the shared ASCEND_BENCH_* set in common/benchmark.hpp.
// A command-line flag sets the matching variable, so the two cannot disagree:
//
//   ASCEND_BENCH_TQ_ABLATION_DIMS=256            head sizes; powers of two in [64, 512]
//   ASCEND_BENCH_TQ_ABLATION_CONTEXTS=64,512     contexts; positive multiples of 8,
//                                                see TURBOQUANT_TESTS.md 13.20
//   ASCEND_BENCH_TQ_ABLATION_STAGES=0,1,2        stages (default 0..5)
//   ASCEND_BENCH_TQ_ABLATION_SYNC_TIMEOUT_MS=0   first-launch deadline (default 30000)
//
// A variable that fails validation falls back to its default, loudly.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "ascend950_shapes.hpp"
#include "benchmark.hpp"
#include "device_buffer.hpp"
#include "fp16.hpp"
#include "random_data.hpp"
#include "turboquant_launch.hpp"

namespace vllm_ascend {
namespace test {
namespace bench {

const char* kSuiteName = "turboquant_950pr_ablation (kv4fp8 Cube decode split, cut stage by stage)";

namespace {

namespace tqh = turboquant_host;
namespace tqm = vllm_ascend::turboquant;
namespace s950 = shapes950;

constexpr tqm::TurboQuantMode kMode = tqm::TurboQuantMode::KV4_FP8;

constexpr int64_t kNumHeads = s950::kNumHeads;      // 8
constexpr int64_t kNumKvHeads = s950::kNumKvHeads;  // 2
constexpr int64_t kBlockSize = s950::kBlockSize;    // 128
constexpr int64_t kQueryTokens = 1;                 // decode

const int64_t kDefaultHeadSizes[] = {256, 512};
const int64_t kDefaultContextLens[] = {64, 512, 1024, 2048};

// Head sizes the sweep accepts: the Walsh-Hadamard needs a power of two, and
// nothing below 64 or above 512 is a shape anyone has asked about.
constexpr int64_t kMinHeadSize = 64;
constexpr int64_t kMaxHeadSize = 512;
// What turboquant_torch_adpt.h's CheckHeadSize lets a model reach.
constexpr int64_t kAdapterMaxHeadSize = 256;
// The softmax's tail mask faults on a partial tile whose valid row count is not
// a multiple of this. See TURBOQUANT_TESTS.md 13.20.
constexpr int64_t kContextMultiple = 8;

// The bound a shape's stage 5 output has to clear for its rows to be trusted.
// The multimode test's smoke bound: kv4fp8 measures 0.986 at S=64, and a decode
// that aliases UB or mis-stages an operand lands far below it.
constexpr double kFidelityCos = 0.90;

// A decode split is milliseconds on the part at every shape here, so thirty
// seconds is a deadlock and not a slow launch.
constexpr int64_t kDefaultSyncTimeoutMs = 30000;
// What aclrtSynchronizeStreamWithTimeout takes for "no deadline".
constexpr int32_t kWaitForever = -1;
// Distinct from the harness's 0, 1 and 77, so a caller can tell a deadlock from
// a failed case.
constexpr int kHangExitCode = 3;

constexpr int64_t kFloatBytes = 4;
constexpr double kPercent = 100.0;

uint32_t U32(int64_t value) { return static_cast<uint32_t>(value); }

// --- the sweep ----------------------------------------------------------------

// A comma-separated list of decimal integers no smaller than `min_value`, or
// false with `values` untouched. A bare strtoll turns a typo into a 0 that some
// later division trips over; this refuses it instead.
bool ParseDecimalList(const char* raw, int64_t min_value, std::vector<int64_t>* values) {
  constexpr size_t kMaxDigits = 9;
  std::vector<int64_t> parsed;
  std::istringstream stream(raw);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty() || token.size() > kMaxDigits || token.find_first_not_of("0123456789") != std::string::npos) {
      return false;
    }
    const int64_t value = std::strtoll(token.c_str(), nullptr, 10);
    if (value < min_value) {
      return false;
    }
    parsed.push_back(value);
  }
  if (parsed.empty()) {
    return false;
  }
  *values = parsed;
  return true;
}

bool IsPowerOfTwo(int64_t value) { return value > 0 && (value & (value - 1)) == 0; }

std::vector<int64_t> HeadSizes() {
  const std::vector<int64_t> defaults(std::begin(kDefaultHeadSizes), std::end(kDefaultHeadSizes));
  const char* raw = std::getenv("ASCEND_BENCH_TQ_ABLATION_DIMS");
  if (raw == nullptr || *raw == '\0') {
    return defaults;
  }
  std::vector<int64_t> parsed;
  bool valid = ParseDecimalList(raw, 1, &parsed);
  for (const int64_t head_size : parsed) {
    valid = valid && IsPowerOfTwo(head_size) && head_size >= kMinHeadSize && head_size <= kMaxHeadSize;
  }
  if (!valid) {
    std::printf("[ascend-bench] ASCEND_BENCH_TQ_ABLATION_DIMS='%s' is not a list of powers of two in [%lld, %lld]; "
                "using the default sweep\n",
                raw, static_cast<long long>(kMinHeadSize), static_cast<long long>(kMaxHeadSize));
    return defaults;
  }
  return parsed;
}

std::vector<int64_t> ContextLens() {
  const std::vector<int64_t> defaults(std::begin(kDefaultContextLens), std::end(kDefaultContextLens));
  const char* raw = std::getenv("ASCEND_BENCH_TQ_ABLATION_CONTEXTS");
  if (raw == nullptr || *raw == '\0') {
    return defaults;
  }
  std::vector<int64_t> parsed;
  bool valid = ParseDecimalList(raw, 1, &parsed);
  for (const int64_t context_len : parsed) {
    valid = valid && context_len % kContextMultiple == 0;
  }
  if (!valid) {
    std::printf("[ascend-bench] ASCEND_BENCH_TQ_ABLATION_CONTEXTS='%s' is not a list of positive multiples of %lld; "
                "using the default sweep\n",
                raw, static_cast<long long>(kContextMultiple));
    return defaults;
  }
  return parsed;
}

tqm::DecodeAblationStage StageAt(int32_t index) { return static_cast<tqm::DecodeAblationStage>(index); }

std::vector<tqm::DecodeAblationStage> Stages() {
  std::vector<tqm::DecodeAblationStage> stages;
  const char* raw = std::getenv("ASCEND_BENCH_TQ_ABLATION_STAGES");
  if (raw != nullptr && *raw != '\0') {
    std::vector<int64_t> parsed;
    bool valid = ParseDecimalList(raw, 0, &parsed);
    for (const int64_t index : parsed) {
      valid = valid && tqm::DecodeAblationStageIsValid(static_cast<int32_t>(index));
    }
    if (valid) {
      for (const int64_t index : parsed) {
        stages.push_back(StageAt(static_cast<int32_t>(index)));
      }
      return stages;
    }
    std::printf("[ascend-bench] ASCEND_BENCH_TQ_ABLATION_STAGES='%s' is not a list of stages 0..%d; running all\n",
                raw, tqm::kDecodeAblationStageCount - 1);
  }
  for (int32_t index = 0; index < tqm::kDecodeAblationStageCount; ++index) {
    stages.push_back(StageAt(index));
  }
  return stages;
}

// The first-launch deadline in milliseconds, as aclrtSynchronizeStreamWithTimeout
// takes it: kWaitForever for 0.
int32_t SyncTimeoutMs() {
  const char* raw = std::getenv("ASCEND_BENCH_TQ_ABLATION_SYNC_TIMEOUT_MS");
  int64_t timeout_ms = kDefaultSyncTimeoutMs;
  if (raw != nullptr && *raw != '\0') {
    std::vector<int64_t> parsed;
    if (ParseDecimalList(raw, 0, &parsed) && parsed.size() == 1) {
      timeout_ms = parsed[0];
    } else {
      std::printf("[ascend-bench] ASCEND_BENCH_TQ_ABLATION_SYNC_TIMEOUT_MS='%s' is not one non-negative integer; "
                  "using %lld\n",
                  raw, static_cast<long long>(kDefaultSyncTimeoutMs));
    }
  }
  return timeout_ms == 0 ? kWaitForever : static_cast<int32_t>(timeout_ms);
}

std::string ShapeName(int64_t head_size, int64_t context_len) {
  std::ostringstream name;
  name << "d" << head_size << "_s" << context_len;
  return name.str();
}

std::string CaseName(const std::string& shape, tqm::DecodeAblationStage stage) {
  return shape + "_" + tqm::DecodeAblationStageName(stage);
}

// --- the host reference ---------------------------------------------------------

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return 0.0;
  }
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  if (na <= 0.0 || nb <= 0.0) {
    return 0.0;
  }
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

// fp32 attention over the unquantised inputs, one decode token: no rotation and
// no codec, so agreeing with it is evidence about the whole pipeline rather than
// one stage of it. `query` is [heads, D]; `key` and `value` are [S, kv heads, D].
// The same computation as HostAttention in sim/test_sim_950pr_turboquant_multimode.cpp,
// with the head size and scale as arguments.
std::vector<float> HostAttention(int64_t head_size, int64_t context_len, double scale,
                                 const std::vector<float>& query, const std::vector<float>& key,
                                 const std::vector<float>& value) {
  std::vector<float> out(static_cast<size_t>(kNumHeads * head_size), 0.0f);
  const int64_t heads_per_kv = kNumHeads / kNumKvHeads;
  std::vector<double> logits(static_cast<size_t>(context_len), 0.0);
  for (int64_t h = 0; h < kNumHeads; ++h) {
    const int64_t kv = h / heads_per_kv;
    double max_logit = -1e30;
    for (int64_t t = 0; t < context_len; ++t) {
      double dot = 0.0;
      for (int64_t d = 0; d < head_size; ++d) {
        dot += static_cast<double>(query[static_cast<size_t>(h * head_size + d)]) *
               static_cast<double>(key[static_cast<size_t>((t * kNumKvHeads + kv) * head_size + d)]);
      }
      logits[static_cast<size_t>(t)] = dot * scale;
      max_logit = std::max(max_logit, logits[static_cast<size_t>(t)]);
    }
    double denom = 0.0;
    for (int64_t t = 0; t < context_len; ++t) {
      logits[static_cast<size_t>(t)] = std::exp(logits[static_cast<size_t>(t)] - max_logit);
      denom += logits[static_cast<size_t>(t)];
    }
    for (int64_t t = 0; t < context_len; ++t) {
      const double weight = logits[static_cast<size_t>(t)] / denom;
      for (int64_t d = 0; d < head_size; ++d) {
        out[static_cast<size_t>(h * head_size + d)] += static_cast<float>(
            weight * static_cast<double>(value[static_cast<size_t>((t * kNumKvHeads + kv) * head_size + d)]));
      }
    }
  }
  return out;
}

// --- one shape ------------------------------------------------------------------

// Everything one (D, S) shape needs, allocated once and alive for its cases and
// its fidelity check. The same data layout, table images and grids as
// ModeScenario in bench_device_950pr_turboquant.cpp, with the head size free.
class AblationScenario {
 public:
  AblationScenario(int64_t head_size, int64_t context_len, int64_t aiv_num)
      : head_size_(head_size),
        context_len_(context_len),
        attention_scale_(static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_size)))) {
    blocks_per_seq_ = (context_len + kBlockSize - 1) / kBlockSize;
    // Four times the blocks the context needs, so the block table scatters
    // through a pool instead of reading a resident run of consecutive blocks.
    num_blocks_ = std::max<int64_t>(4, blocks_per_seq_ * 4);

    DeterministicRandom rng(0x7451u);
    const size_t kv_elems = static_cast<size_t>(context_len * kNumKvHeads * head_size);
    key_host_ = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
    value_host_ = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
    query_host_ = rng.NormalHalfExact(static_cast<size_t>(kQueryTokens * kNumHeads * head_size), 0.0f, 1.0f);

    const std::vector<int32_t> permutation = rng.Permutation(static_cast<int32_t>(num_blocks_));
    const std::vector<int32_t> block_table(permutation.begin(),
                                           permutation.begin() + static_cast<std::ptrdiff_t>(blocks_per_seq_));
    std::vector<int32_t> slots(static_cast<size_t>(context_len));
    for (int64_t i = 0; i < context_len; ++i) {
      slots[static_cast<size_t>(i)] = block_table[static_cast<size_t>(i / kBlockSize)] *
                                          static_cast<int32_t>(kBlockSize) +
                                      static_cast<int32_t>(i % kBlockSize);
    }

    key_ = DeviceBuffer::FromHost(FloatToHalf(key_host_), kBenchmarkAlignBytes);
    value_ = DeviceBuffer::FromHost(FloatToHalf(value_host_), kBenchmarkAlignBytes);
    query_ = DeviceBuffer::FromHost(FloatToHalf(query_host_), kBenchmarkAlignBytes);
    slots_ = DeviceBuffer::FromHost(slots, kBenchmarkAlignBytes);
    block_tables_ = DeviceBuffer::FromHost(block_table, kBenchmarkAlignBytes);
    context_lens_ = DeviceBuffer::FromHost(
        std::vector<int32_t>(static_cast<size_t>(kQueryTokens), static_cast<int32_t>(context_len)),
        kBenchmarkAlignBytes);
    pi_signs_ = DeviceBuffer::FromHost(tqh::PiSigns(head_size), kBenchmarkAlignBytes);
    h16_ = DeviceBuffer::FromHost(tqh::Hadamard16Half(), kBenchmarkAlignBytes);
    query_rot_ = DeviceBuffer::Empty<float>(static_cast<size_t>(kQueryTokens * kNumHeads * head_size),
                                            kBenchmarkAlignBytes);
    aiv_num_ = aiv_num;

    // The rotation image is the shipping 4-bit codec's; the two mode images are
    // a single zero block each, because the affine codec reads no table.
    rot_tables_ = DeviceBuffer::FromHost(tqh::CodecTables(head_size, 1), kBenchmarkAlignBytes);
    write_tables_ = DeviceBuffer::FromHost(tqh::ModeTables(kMode, head_size, 1, /*nz_rows=*/0), kBenchmarkAlignBytes);
    decode_tables_ = DeviceBuffer::FromHost(
        tqh::ModeTables(kMode, head_size, tqh::kUnpackRows, tqh::kCubeTileRows), kBenchmarkAlignBytes);

    key_cache_ = DeviceBuffer::Empty<int8_t>(
        tqh::ModePackedCacheBytes(kMode, num_blocks_, kBlockSize, kNumKvHeads, head_size), kBenchmarkAlignBytes);
    value_cache_ = DeviceBuffer::Empty<int8_t>(key_cache_.size_bytes(), kBenchmarkAlignBytes);
    scale_plane_ = DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(num_blocks_, kBlockSize, kNumKvHeads),
                                              kBenchmarkAlignBytes);
    out_ = DeviceBuffer::Empty<Half>(static_cast<size_t>(kQueryTokens * kNumHeads * head_size), kBenchmarkAlignBytes);

    write_grid_ = tqh::PlanReshapeAndCache(context_len, aiv_num);
    decode_grid_ = tqh::PlanCubeDecode(kQueryTokens, kNumHeads, kNumKvHeads, head_size, blocks_per_seq_, aiv_num);
    ResetWorkspace();
  }

  // Writes the whole context once, outside any timed region.
  void FillCache(aclrtStream stream) const {
    turboquant_mm_reshape_and_cache_impl(
        static_cast<int32_t>(kMode), AscendType::FP16, stream, write_grid_.block_dim, key_.get(), value_.get(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), slots_.get(), pi_signs_.get(), rot_tables_.get(),
        write_tables_.get(), U32(context_len_), U32(kNumKvHeads), U32(head_size_), U32(kBlockSize),
        write_grid_.tokens_per_core, attention_scale_);
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  // Zeros uploaded rather than left to the allocator, before every stage, so a
  // cut stage's untouched check does not depend on which stages ran before it.
  void ResetWorkspace() {
    workspace_ = DeviceBuffer::FromHost(std::vector<float>(decode_grid_.workspace_floats, 0.0f), kBenchmarkAlignBytes);
  }

  /*
   * The query rotation, once, outside every timed rung.
   *
   * It is deliberately NOT part of EnqueueSplit: every rung consumes the same
   * pre-rotated query, so no rung's delta carries any of it and the waterfall
   * measures only what the split kernel itself does. What used to be stage 2's
   * bulk is now this call, and pricing it is the rotate benchmark's job.
   */
  void RotateQueryOnce(aclrtStream stream) {
    rotate_plan_ = tqh::RotateQuery(stream, AscendType::FP16, query_.get(), pi_signs_.get(), h16_.get(),
                                    rot_tables_.get(), query_rot_.get(), kQueryTokens, kNumHeads, head_size_,
                                    aiv_num_, /*input_exact_in_half=*/true);
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  const vllm_ascend::turboquant::RotateQPlan& rotate_plan() const { return rotate_plan_; }

  void EnqueueSplit(tqm::DecodeAblationStage stage, aclrtStream stream) const {
    turboquant_mm_decode_ablation_impl(
        static_cast<int32_t>(stage), AscendType::FP16, stream, decode_grid_.split_block_dim, query_rot_.get(),
        key_cache_.get(), value_cache_.get(), scale_plane_.get(), block_tables_.get(), context_lens_.get(),
        decode_tables_.get(), workspace_.get(), U32(kQueryTokens),
        U32(kNumHeads), U32(kNumKvHeads), U32(head_size_), U32(kBlockSize), U32(blocks_per_seq_),
        U32(decode_grid_.num_splits), decode_grid_.split_tasks_per_core, attention_scale_, attention_scale_);
  }

  // The combine, which the ladder does not time. Reads the partials stage 5
  // left in the workspace, so it is only meaningful straight after that case.
  // Returned un-rotated, as the folded W_o would see it, so it compares against
  // Reference() directly.
  std::vector<float> CombineAndReadBack(aclrtStream stream) const {
    turboquant_paged_attention_combine_impl(AscendType::FP16, stream, decode_grid_.combine_block_dim,
                                            workspace_.get(), out_.get(), U32(kQueryTokens), U32(kNumHeads),
                                            U32(head_size_), U32(decode_grid_.num_splits),
                                            decode_grid_.combine_tasks_per_core);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    return tqh::UnrotateHeads(HalfToFloat(out_.ToHost<Half>()), head_size_);
  }

  std::vector<float> Reference() const {
    return HostAttention(head_size_, context_len_, static_cast<double>(attention_scale_), query_host_, key_host_,
                         value_host_);
  }

  std::vector<float> Workspace() const { return workspace_.ToHost<float>(); }

  // Bytes CopyInTile moves per launch. Every tile of every kv head reads a full
  // kCubeTileRows rows of K, of V and of the scale plane whatever its valid row
  // count, so this counts tiles the way CountTiles does. The same at every rung:
  // GB/s is comparable up the ladder, and at stage 0 it is the MTE2 figure.
  double TileReadBytes() const {
    int64_t tiles = 0;
    for (int64_t block = 0; block < blocks_per_seq_; ++block) {
      const int64_t rows = std::min<int64_t>(kBlockSize, context_len_ - block * kBlockSize);
      tiles += (rows + tqh::kCubeTileRows - 1) / tqh::kCubeTileRows;
    }
    const int64_t row_bytes =
        2 * tqh::ModePackedBytes(kMode, head_size_) + tqh::ScaleSlotFloats(kNumKvHeads) * kFloatBytes;
    return static_cast<double>(kNumKvHeads * tiles * tqh::kCubeTileRows * row_bytes);
  }

  const tqh::CubeDecodeGrid& decode_grid() const { return decode_grid_; }
  int64_t blocks_per_seq() const { return blocks_per_seq_; }

 private:
  int64_t head_size_ = 0;
  int64_t context_len_ = 0;
  float attention_scale_ = 1.0f;
  int64_t blocks_per_seq_ = 0;
  int64_t num_blocks_ = 0;

  std::vector<float> key_host_, value_host_, query_host_;

  DeviceBuffer key_, value_, query_, slots_, block_tables_, context_lens_, pi_signs_;
  DeviceBuffer h16_, query_rot_;
  DeviceBuffer rot_tables_, write_tables_, decode_tables_;
  DeviceBuffer key_cache_, value_cache_, scale_plane_, workspace_, out_;
  tqh::ReshapeAndCacheGrid write_grid_;
  tqh::CubeDecodeGrid decode_grid_;
  vllm_ascend::turboquant::RotateQPlan rotate_plan_;
  int64_t aiv_num_ = 1;
};

// What the waterfall needs to say about a shape beyond its latencies.
struct ShapeVerdict {
  int64_t head_size = 0;
  int64_t context_len = 0;
  bool fidelity_checked = false;
  double cos = 0.0;
  // Empty when the shape's rows can be read at face value.
  std::string distrust;
};

enum class ShapeOutcome {
  kCompleted,
  // A stage missed its first-launch deadline. Nothing more may be launched on
  // the stream, and nothing may tear it down.
  kStreamWedged,
};

// Straight after stage 5: the combine, and the cosine against the host.
void CheckFidelity(BenchmarkRunner& runner, const AblationScenario& ready, const std::string& shape,
                   ShapeVerdict* verdict) {
  const std::string fidelity_name = shape + "_fidelity";
  try {
    const std::vector<float> output = ready.CombineAndReadBack(runner.stream());
    verdict->cos = Cosine(ready.Reference(), output);
    verdict->fidelity_checked = true;
    // Negated so a NaN cosine fails too.
    if (!(verdict->cos >= kFidelityCos)) {
      std::ostringstream reason;
      reason << "stage 5 plus combine gives cos " << verdict->cos << " against the fp32 host attention, below "
             << kFidelityCos;
      runner.RecordFailure(fidelity_name, reason.str());
      verdict->distrust = reason.str();
    }
  } catch (const std::exception& error) {
    runner.RecordFailure(fidelity_name, error.what());
    verdict->distrust = std::string("fidelity check failed: ") + error.what();
  }
}

ShapeOutcome RunShape(BenchmarkRunner& runner, int64_t aiv_num, const std::vector<tqm::DecodeAblationStage>& stages,
                      int32_t sync_timeout_ms, ShapeVerdict* verdict) {
  const std::string shape = ShapeName(verdict->head_size, verdict->context_len);

  std::unique_ptr<AblationScenario> scenario;
  try {
    scenario.reset(new AblationScenario(verdict->head_size, verdict->context_len, aiv_num));
    scenario->FillCache(runner.stream());
    // Once, before any rung: every stage reads the same rotated query, so the
    // waterfall's deltas are the split kernel's alone.
    scenario->RotateQueryOnce(runner.stream());
  } catch (const std::exception& error) {
    for (const tqm::DecodeAblationStage stage : stages) {
      runner.RecordFailure(CaseName(shape, stage), std::string("setup failed: ") + error.what());
    }
    verdict->distrust = "setup failed";
    return ShapeOutcome::kCompleted;
  }

  const tqh::CubeDecodeGrid& grid = scenario->decode_grid();
  std::printf("\n[ascend-bench] %s: blocks_per_seq=%lld splits=%lld grid(split)=%u tasks/core=%u "
              "tile read=%.0f B/launch\n",
              shape.c_str(), static_cast<long long>(scenario->blocks_per_seq()),
              static_cast<long long>(grid.num_splits), grid.split_block_dim, grid.split_tasks_per_core,
              scenario->TileReadBytes());
  std::fflush(stdout);

  AblationScenario& mutable_scenario = *scenario;
  const AblationScenario& ready = mutable_scenario;
  for (const tqm::DecodeAblationStage stage : stages) {
    const bool writes_workspace = stage == tqm::DecodeAblationStage::STAGE_5_FULL_PIPELINE;
    const std::string name = CaseName(shape, stage);
    mutable_scenario.ResetWorkspace();

    // The hang guard; see the file header. One launch, synchronised against a
    // deadline, before the harness -- whose synchronisations have none -- is
    // handed the case.
    std::printf("[ascend-bench] %s: first launch, deadline %d ms\n", name.c_str(), sync_timeout_ms);
    std::fflush(stdout);
    ready.EnqueueSplit(stage, runner.stream());
    const aclError synced = aclrtSynchronizeStreamWithTimeout(runner.stream(), sync_timeout_ms);
    if (synced != ACL_SUCCESS) {
      std::ostringstream reason;
      reason << "the first launch did not synchronise within " << sync_timeout_ms << " ms (aclError " << synced
             << "); treating the stage as deadlocked";
      std::printf("\n[ascend-bench] HANG: %s: %s\n", name.c_str(), reason.str().c_str());
      runner.RecordFailure(name, reason.str());
      verdict->distrust = name + " deadlocked";
      return ShapeOutcome::kStreamWedged;
    }

    try {
      BenchmarkCase ablation_case;
      ablation_case.name = name;
      ablation_case.bytes_per_iteration = ready.TileReadBytes();
      ablation_case.tasks_per_launch = 1;  // the split alone
      ablation_case.launch = [&ready, stage](aclrtStream stream) { ready.EnqueueSplit(stage, stream); };
      if (writes_workspace) {
        // Same partials every launch, so bit-identical.
        ablation_case.checksum = [&ready]() { return ChecksumSum(ready.Workspace()); };
      } else {
        ablation_case.checksum = [&ready]() { return ChecksumSumOfSquares(ready.Workspace()); };
      }
      runner.Run(ablation_case);

      if (writes_workspace) {
        CheckFidelity(runner, ready, shape, verdict);
      } else if (ChecksumSumOfSquares(ready.Workspace()) != 0.0) {
        // The checksum above only proves the workspace did not change between
        // two reads; a cut that wrote the same partials every launch passes it.
        // Zero is the stronger statement, and only a cut below stage 5 owes it.
        runner.RecordFailure(name, "wrote the workspace: a cut below stage 5 must not reach the partial writeback");
        verdict->distrust = "an ablation gate leaked the writeback";
      }
    } catch (const std::exception& error) {
      runner.RecordFailure(name, error.what());
    }
  }
  return ShapeOutcome::kCompleted;
}

// --- the waterfall --------------------------------------------------------------

// The timing mode the waterfall quotes: pipelined when it ran, else the first
// mode that did, so a narrowed ASCEND_BENCH_MODES still gets a table.
TimingMode WaterfallMode(const BenchmarkRunner& runner) {
  const std::vector<TimingMode>& modes = runner.options().modes;
  if (std::find(modes.begin(), modes.end(), TimingMode::kPipelined) != modes.end() || modes.empty()) {
    return TimingMode::kPipelined;
  }
  return modes.front();
}

const BenchmarkResult* FindResult(const BenchmarkRunner& runner, const std::string& case_name, TimingMode mode) {
  for (const BenchmarkResult& result : runner.results()) {
    if (result.case_name == case_name && result.mode == mode && result.latency.sample_count > 0) {
      return &result;
    }
  }
  return nullptr;
}

struct Rung {
  bool present = false;
  double median_us = 0.0;
  double p95_us = 0.0;
  // Against the rung below; valid only when both are present.
  bool has_delta = false;
  double delta_us = 0.0;
};

std::vector<Rung> Ladder(const BenchmarkRunner& runner, TimingMode mode, const std::string& shape) {
  std::vector<Rung> rungs(static_cast<size_t>(tqm::kDecodeAblationStageCount));
  for (int32_t index = 0; index < tqm::kDecodeAblationStageCount; ++index) {
    Rung& rung = rungs[static_cast<size_t>(index)];
    const BenchmarkResult* result = FindResult(runner, CaseName(shape, StageAt(index)), mode);
    if (result == nullptr) {
      continue;
    }
    rung.present = true;
    rung.median_us = result->latency.median_us;
    rung.p95_us = result->latency.p95_us;
    if (index == 0) {
      rung.has_delta = true;
      rung.delta_us = rung.median_us;
    } else if (rungs[static_cast<size_t>(index - 1)].present) {
      rung.has_delta = true;
      rung.delta_us = rung.median_us - rungs[static_cast<size_t>(index - 1)].median_us;
    }
  }
  return rungs;
}

void PrintShapeWaterfall(const BenchmarkRunner& runner, TimingMode mode, const ShapeVerdict& verdict) {
  const std::string shape = ShapeName(verdict.head_size, verdict.context_len);
  const std::vector<Rung> rungs = Ladder(runner, mode, shape);
  const Rung& full = rungs.back();

  std::printf("\n[ascend-bench] %s", shape.c_str());
  if (verdict.fidelity_checked) {
    std::printf("  cos %.6f vs fp32 host", verdict.cos);
  }
  if (verdict.head_size > kAdapterMaxHeadSize) {
    std::printf("  (head_size above the adapter's %lld)", static_cast<long long>(kAdapterMaxHeadSize));
  }
  std::printf("\n");
  if (!verdict.distrust.empty()) {
    std::printf("[ascend-bench]   UNTRUSTED: %s\n", verdict.distrust.c_str());
  }
  std::printf("  %-20s %12s %12s %8s %12s\n", "stage", "latency_us", "delta_us", "share", "p95_us");
  for (int32_t index = 0; index < tqm::kDecodeAblationStageCount; ++index) {
    const Rung& rung = rungs[static_cast<size_t>(index)];
    std::printf("  %-20s ", tqm::DecodeAblationStageName(StageAt(index)));
    if (!rung.present) {
      std::printf("%12s %12s %8s %12s\n", "-", "-", "-", "-");
      continue;
    }
    std::printf("%12.2f ", rung.median_us);
    if (rung.has_delta) {
      std::printf("%+12.2f ", rung.delta_us);
    } else {
      std::printf("%12s ", "-");
    }
    if (rung.has_delta && full.present && full.median_us > 0.0) {
      std::printf("%7.1f%% ", kPercent * rung.delta_us / full.median_us);
    } else {
      std::printf("%8s ", "-");
    }
    std::printf("%12.2f\n", rung.p95_us);
  }
}

// One table per head size, a column per context: each stage's delta, so how a
// piece scales with S can be read along a row. Stage 2 should be flat.
void PrintDeltaByContext(const BenchmarkRunner& runner, TimingMode mode, const std::vector<ShapeVerdict>& verdicts,
                         int64_t head_size) {
  std::vector<const ShapeVerdict*> columns;
  for (const ShapeVerdict& verdict : verdicts) {
    if (verdict.head_size == head_size) {
      columns.push_back(&verdict);
    }
  }
  if (columns.empty()) {
    return;
  }
  std::vector<std::vector<Rung>> ladders;
  for (const ShapeVerdict* column : columns) {
    ladders.push_back(Ladder(runner, mode, ShapeName(column->head_size, column->context_len)));
  }

  std::printf("\n[ascend-bench] D=%lld: delta_us by context\n", static_cast<long long>(head_size));
  std::printf("  %-20s", "stage");
  for (const ShapeVerdict* column : columns) {
    const std::string label =
        "S=" + std::to_string(column->context_len) + (column->distrust.empty() ? "" : "!");
    std::printf(" %15s", label.c_str());
  }
  std::printf("\n");
  for (int32_t index = 0; index < tqm::kDecodeAblationStageCount; ++index) {
    std::printf("  %-20s", tqm::DecodeAblationStageName(StageAt(index)));
    for (const std::vector<Rung>& ladder : ladders) {
      const Rung& rung = ladder[static_cast<size_t>(index)];
      if (rung.has_delta) {
        std::printf(" %+15.2f", rung.delta_us);
      } else {
        std::printf(" %15s", "-");
      }
    }
    std::printf("\n");
  }
  std::printf("  %-20s", "total (stage5)");
  for (const std::vector<Rung>& ladder : ladders) {
    if (ladder.back().present) {
      std::printf(" %15.2f", ladder.back().median_us);
    } else {
      std::printf(" %15s", "-");
    }
  }
  std::printf("\n");
}

void PrintWaterfall(const BenchmarkRunner& runner, const std::vector<int64_t>& head_sizes,
                    const std::vector<ShapeVerdict>& verdicts) {
  const TimingMode mode = WaterfallMode(runner);
  std::printf("\n[ascend-bench] ==== decode ablation waterfall (%s median) ====\n", TimingModeLabel(mode));
  std::printf("[ascend-bench] delta = this stage's median minus the one below it; share = delta over stage 5.\n"
              "[ascend-bench] A negative delta is noise or a change in pipe overlap, not a free stage.\n");
  for (const ShapeVerdict& verdict : verdicts) {
    PrintShapeWaterfall(runner, mode, verdict);
  }
  for (const int64_t head_size : head_sizes) {
    PrintDeltaByContext(runner, mode, verdicts, head_size);
  }
  std::printf("  ('!' marks a context whose rows are UNTRUSTED; see its waterfall above.)\n");
  std::fflush(stdout);
}

void PrintBanner(const std::vector<int64_t>& head_sizes, const std::vector<int64_t>& context_lens,
                 const std::vector<tqm::DecodeAblationStage>& stages, int32_t sync_timeout_ms, int64_t aiv_num,
                 bool aiv_queried) {
  std::printf("[ascend-bench] kv4fp8 decode ablation: vector cores = %lld%s\n", static_cast<long long>(aiv_num),
              aiv_queried ? " (from aclGetDeviceCapability)" : " (assumed; the runtime declined to answer)");
  std::printf("[ascend-bench] stages, each adding one piece of the split kernel to the one before:\n");
  std::printf("[ascend-bench]   %-18s packed K/V and scale tiles, GM -> UB\n",
              tqm::DecodeAblationStageName(tqm::DecodeAblationStage::STAGE_0_MTE2_ONLY));
  std::printf("[ascend-bench]   %-18s + affine unpack onto the fp8 grid\n",
              tqm::DecodeAblationStageName(tqm::DecodeAblationStage::STAGE_1_UNPACK));
  std::printf("[ascend-bench]   %-18s + query read, Pi rotation, operand cast (per task, not per tile)\n",
              tqm::DecodeAblationStageName(tqm::DecodeAblationStage::STAGE_2_QUERY_PREP));
  std::printf("[ascend-bench]   %-18s + V -> MTE3 edges and UB -> L1 copies\n",
              tqm::DecodeAblationStageName(tqm::DecodeAblationStage::STAGE_3_L1_STAGING));
  std::printf("[ascend-bench]   %-18s + Load2D, Q.K^T Mmad, Fixpipe to UB, AIV/AIC handshake\n",
              tqm::DecodeAblationStageName(tqm::DecodeAblationStage::STAGE_4_SCORE_GEMM));
  std::printf("[ascend-bench]   %-18s + online softmax, P.V GEMM, accumulator, GM writeback (the shipping kernel)\n",
              tqm::DecodeAblationStageName(tqm::DecodeAblationStage::STAGE_5_FULL_PIPELINE));
  std::printf("[ascend-bench] running:");
  for (const tqm::DecodeAblationStage stage : stages) {
    std::printf(" %s", tqm::DecodeAblationStageName(stage));
  }
  if (sync_timeout_ms == kWaitForever) {
    std::printf("   first-launch deadline: none\n");
  } else {
    std::printf("   first-launch deadline: %d ms\n", sync_timeout_ms);
  }
  std::printf("[ascend-bench] head sizes:");
  for (const int64_t head_size : head_sizes) {
    std::printf(" %lld%s", static_cast<long long>(head_size), head_size > kAdapterMaxHeadSize ? "*" : "");
  }
  std::printf("   contexts:");
  for (const int64_t context_len : context_lens) {
    std::printf(" %lld", static_cast<long long>(context_len));
  }
  std::printf("\n");
  for (const int64_t head_size : head_sizes) {
    if (head_size > kAdapterMaxHeadSize) {
      std::printf("[ascend-bench]   * above the production adapter's head_size ceiling of %lld and never run before;\n"
                  "[ascend-bench]     its rows count only where the fidelity gate (cos >= %.2f) passes.\n",
                  static_cast<long long>(kAdapterMaxHeadSize), kFidelityCos);
      break;
    }
  }
  std::fflush(stdout);
}

}  // namespace

void BuildSuite(BenchmarkRunner& runner) {
  const std::vector<int64_t> head_sizes = HeadSizes();
  const std::vector<int64_t> context_lens = ContextLens();
  const std::vector<tqm::DecodeAblationStage> stages = Stages();
  const int32_t sync_timeout_ms = SyncTimeoutMs();

  bool aiv_queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&aiv_queried);
  PrintBanner(head_sizes, context_lens, stages, sync_timeout_ms, aiv_num, aiv_queried);

  // Head size is the outer loop so every shape the production adapter accepts
  // is measured before one it does not: a D = 512 launch that faults the part
  // cannot take the D = 256 numbers with it.
  std::vector<ShapeVerdict> verdicts;
  for (const int64_t head_size : head_sizes) {
    for (const int64_t context_len : context_lens) {
      ShapeVerdict verdict;
      verdict.head_size = head_size;
      verdict.context_len = context_len;
      const ShapeOutcome outcome = RunShape(runner, aiv_num, stages, sync_timeout_ms, &verdict);
      verdicts.push_back(verdict);
      if (outcome == ShapeOutcome::kStreamWedged) {
        PrintWaterfall(runner, head_sizes, verdicts);
        std::printf("[ascend-bench] the stream is wedged: exiting %d without the shared report or teardown, "
                    "both of which would block on it. Rerun the other stages with --stage=.\n",
                    kHangExitCode);
        std::fflush(stdout);
        std::_Exit(kHangExitCode);
      }
    }
  }
  PrintWaterfall(runner, head_sizes, verdicts);
}

}  // namespace bench
}  // namespace test
}  // namespace vllm_ascend
