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

// Functional check of the Cube-native multi-mode decode. ONE SOURCE, TWO TIERS:
// compiled as test_sim_950pr_turboquant_multimode against the arch35 camodel and
// as test_device_950pr_turboquant_multimode against silicon, so the cos bound
// the simulator enforces and the one hardware enforces cannot drift apart.
//
// It asserts nothing about time on either tier. Every timing this project quotes
// comes from bench_device_950pr_turboquant on physical silicon.
//
// THE SWEEP IS THE ONE THING THAT DIFFERS, and the asymmetry is deliberate: a
// camodel pass is cycle-level, so there the default is a SINGLE shape and the
// sweep loop runs exactly once. On silicon the default is the whole validation
// matrix. See kDefaultContextLens / kDefaultBatchSizes for the lists and
// ASCEND_TQ_SIM_CONTEXT / ASCEND_TQ_SIM_BATCH for overriding either on either
// tier.
//
// THE SHAPE:
//
//   head_size 256     the production head dim, and the one the Cube path is
//                     sized for. It cannot drop to 64: the score GEMM's
//                     reduction length is head_size and the Cube's K step is
//                     64, so a smaller head would stop exercising the
//                     multi-step LoadData.
//   block_size 128    the production block size, and required: see kBlockSize.
//   heads / kv        4 over 2 on the camodel -- a GQA group of two is enough to
//                     prove the batching happens -- and the production 8 over 2
//                     on silicon.
//   batch             1 on the camodel; 1 and 8 on silicon, so the per-sequence
//                     block tables and context lengths are exercised. A wrong
//                     batch axis shows up as a collapsed cosine on every
//                     sequence but the first, which is why the fidelity check
//                     reports the worst sequence and not only the batch figure.
//   context           16 on the camodel (a partial 64-row tile, so the tail
//                     masking in the softmax is live); 64 / 512 / 1024 / 2048 on
//                     silicon, where 64 is the shape the camodel verified and
//                     the rest are the model's own and exercise the tile loop.
//
// WHAT IS ASSERTED, in increasing strength:
//
//   dispatch    every mode's write and decode launch returns ACL_SUCCESS and the
//               stream synchronises. This is the only thing kv3fp4 is held to.
//   finite      every mode's output is entirely finite and not identically zero.
//   fidelity    kv4fp8's and kv5fp8's output against an fp32 host reference.
//               Both expand onto the same fp8 e4m3fn operand grid, so a
//               structural defect that is common to them -- a transposed K
//               operand, a probability row staged at the wrong stride -- shows
//               up in both. They no longer share a packing or an expand: kv4fp8
//               is affine INT4 with a plane-split packing and a flat L1 stage,
//               kv5fp8 a Lloyd-Max index with an interleaved one and a strided
//               band stage (TURBOQUANT_TESTS.md section 13.9), so a defect in
//               either one's own layout shows up only there. The bound is
//               deliberately loose (see kSmokeCos) and is NOT the cos > 0.995
//               gate, whose measurement lives in
//               scripts/tq_multimode_calibration.py; there kv4fp8 reaches
//               0.98752 and kv5fp8 0.99613 at S = 512.
//
//               kv3fp4 is exempt because its operand grid is fp4 e2m1, which
//               costs it more than its codebook does: the same script measures
//               0.92813, and a bound loose enough to admit that would not catch
//               anything.

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
#include "device_buffer.hpp"
#include "fp16.hpp"
#include "random_data.hpp"
#include "test_harness.hpp"
#include "turbo_quant_cpu.h"
#include "turboquant_launch.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace tqh = turboquant_host;
namespace tq = turboquant_ref;
namespace tqm = vllm_ascend::turboquant;

// THE SHAPE DIFFERS BY TIER, and only in the query-head count.
//
// The simulator runs 4 query heads over 2 kv heads because a camodel pass is
// minutes and a GQA group of two is already enough to prove the heads are being
// batched into the GEMM's M dimension at all. Silicon runs the production 8
// over 2 -- shapes950::kNumHeads -- because there the point is not "does the
// batching happen" but "does it hold at the group width the model decodes at".
//
// head_dim 256 and block_size 128 are the production values on BOTH tiers; see
// kBlockSize below for why the block size is not negotiable here.
#ifdef VLLM_ASCEND_TEST_TIER_DEVICE
constexpr int64_t kNumHeads = 8;
#else
constexpr int64_t kNumHeads = 4;
#endif
constexpr int64_t kHeadSize = 256;
constexpr int64_t kNumKvHeads = 2;
// 128, the production block size, and NOT the 16 the rest of this tier uses to
// keep camodel passes short. The Cube decode tiles the packed cache in
// kCubeTileRows = 64 row blocks and requires that to divide block_size:
// CopyInTile issues one 64-row DataCopy per tile unconditionally, so at
// block_size 16 it reads 48 rows past the end of the block it was given -- into
// the next blocks of the pool, and past the end of the cache and scale-plane
// allocations entirely when the block is the last one. The garbage that pulls
// into the per-row scale lanes is multiplied into the score row before the
// invalid columns are masked, which is where the vec_err_idata_inf_nan flood
// came from. It is a shape this kernel does not support, not a kernel defect.
//
// It costs nothing here: at S <= 128 the tile loop still runs one 64-row tile,
// because rows is clamped to the context length.
constexpr int64_t kBlockSize = 128;
constexpr int64_t kMinBlocks = 4;

/*
 * THE SWEEP, and why it has one entry on the simulator and a matrix on silicon.
 *
 * A camodel pass is cycle-level: the single-tile shape below takes about seven
 * minutes and S=512 takes most of an hour. So the simulator default is ONE
 * batch and ONE context length, and the loop in the test body runs exactly once.
 * That is the constraint the old "do not add a loop to this file" note was
 * protecting, and it still holds -- what changed is that the loop is now over a
 * list whose simulator default has length one.
 *
 * On silicon a decode step is microseconds, so the device default is the whole
 * validation matrix:
 *
 *   batch    1, 8            one sequence, and a real decode batch
 *   context  64              the shape the camodel verified, so the two tiers
 *                            overlap at one point instead of meeting nowhere
 *            512, 1024, 2048 the shapes Qwen3.5-2B decodes at, and the ones
 *                            that exercise the tile loop and real paging
 *
 * ASCEND_TQ_SIM_CONTEXT and ASCEND_TQ_SIM_BATCH override either list as a
 * comma-separated set of positive integers, on both tiers -- that is how a
 * camodel run covers a second shape (one process per shape) and how a silicon
 * run narrows to the one that regressed.
 */
#ifdef VLLM_ASCEND_TEST_TIER_DEVICE
const int64_t kDefaultContextLens[] = {64, 512, 1024, 2048};
const int64_t kDefaultBatchSizes[] = {1, 8};
#else
const int64_t kDefaultContextLens[] = {16};
const int64_t kDefaultBatchSizes[] = {1};
#endif

// A comma-separated list of positive integers from `name`, or `fallback`
// unchanged when the variable is unset, empty, or parses to nothing. A partly
// parseable list keeps the entries that parsed: substituting the default because
// one field was a typo would silently run a different sweep than the one asked
// for, and so would dropping the list.
std::vector<int64_t> Int64ListFromEnv(const char* name, const int64_t* fallback, size_t fallback_count) {
  const std::vector<int64_t> defaults(fallback, fallback + fallback_count);
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return defaults;
  }
  std::vector<int64_t> parsed;
  std::stringstream stream(raw);
  std::string field;
  while (std::getline(stream, field, ',')) {
    const long long value = std::strtoll(field.c_str(), nullptr, 10);
    if (value > 0) {
      parsed.push_back(static_cast<int64_t>(value));
    }
  }
  if (parsed.empty()) {
    std::printf("[ multimode ] %s='%s' parsed to nothing; using the default sweep\n", name, raw);
    return defaults;
  }
  return parsed;
}

// Parsed once each: the sweep is fixed for the life of the process.
const std::vector<int64_t>& ContextLens() {
  static const std::vector<int64_t> value =
      Int64ListFromEnv("ASCEND_TQ_SIM_CONTEXT", kDefaultContextLens,
                       sizeof(kDefaultContextLens) / sizeof(kDefaultContextLens[0]));
  return value;
}

const std::vector<int64_t>& BatchSizes() {
  static const std::vector<int64_t> value = Int64ListFromEnv(
      "ASCEND_TQ_SIM_BATCH", kDefaultBatchSizes, sizeof(kDefaultBatchSizes) / sizeof(kDefaultBatchSizes[0]));
  return value;
}

int64_t BlocksPerSeq(int64_t context_len) { return (context_len + kBlockSize - 1) / kBlockSize; }

// The block pool, for `batch` sequences of `context_len` tokens each.
// Deliberately larger than the sequences need so the block table is a genuine
// scatter rather than a run of consecutive blocks, which is what makes the paged
// addressing real -- at the smallest shape that is a pool of 4 for a sequence
// needing 1.
//
// Sized off the whole batch and not off one sequence: the sequences must not
// share blocks, or a decode that read the wrong sequence's cache would still
// find plausible data there and the batch axis would go unchecked.
int64_t NumBlocks(int64_t batch, int64_t context_len) {
  return std::max(kMinBlocks, batch * BlocksPerSeq(context_len) * 4);
}
constexpr float kInvSqrtHeadSize = 0.0625f;  // 1 / sqrt(256), exact in fp32
constexpr float kAttentionScale = kInvSqrtHeadSize;

// The loose bound described in the file header: roughly four times the distance
// from 1.0 that the calibration script's S = 512 measurement shows, which
// leaves room for the short-context effect but not for a structurally wrong
// decode -- a transposed K operand lands near zero cosine.
constexpr double kSmokeCos = 0.90;

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
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

// fp32 attention over the unquantised inputs, on the host: the reference the
// fidelity bound is quoted against. Deliberately not a copy of the device's
// arithmetic -- no rotation, no codebook -- so that agreeing with it is
// evidence about the pipeline as a whole and not about one stage.
//
// Shapes: `query` is [batch, num_heads, head_size], `key` and `value` are
// [batch, context_len, num_kv_heads, head_size], and the result is
// [batch, num_heads, head_size]. One decode step per sequence, each over its own
// context -- which is what the device does, so a batch axis dropped on either
// side shows up as a cosine collapse on every sequence but the first.
std::vector<float> HostAttention(int64_t batch, int64_t context_len, const std::vector<float>& query,
                                 const std::vector<float>& key, const std::vector<float>& value) {
  std::vector<float> out(static_cast<size_t>(batch * kNumHeads * kHeadSize), 0.0f);
  const int64_t heads_per_kv = kNumHeads / kNumKvHeads;
  for (int64_t b = 0; b < batch; ++b) {
    const int64_t query_base = b * kNumHeads * kHeadSize;
    const int64_t kv_base = b * context_len * kNumKvHeads * kHeadSize;
    for (int64_t h = 0; h < kNumHeads; ++h) {
      const int64_t kv = h / heads_per_kv;
      std::vector<double> logits(static_cast<size_t>(context_len), 0.0);
      double max_logit = -1e30;
      for (int64_t t = 0; t < context_len; ++t) {
        double dot = 0.0;
        for (int64_t d = 0; d < kHeadSize; ++d) {
          dot += static_cast<double>(query[static_cast<size_t>(query_base + h * kHeadSize + d)]) *
                 static_cast<double>(key[static_cast<size_t>(kv_base + (t * kNumKvHeads + kv) * kHeadSize + d)]);
        }
        logits[static_cast<size_t>(t)] = dot * static_cast<double>(kAttentionScale);
        max_logit = std::max(max_logit, logits[static_cast<size_t>(t)]);
      }
      double denom = 0.0;
      for (int64_t t = 0; t < context_len; ++t) {
        logits[static_cast<size_t>(t)] = std::exp(logits[static_cast<size_t>(t)] - max_logit);
        denom += logits[static_cast<size_t>(t)];
      }
      for (int64_t t = 0; t < context_len; ++t) {
        const double w = logits[static_cast<size_t>(t)] / denom;
        for (int64_t d = 0; d < kHeadSize; ++d) {
          out[static_cast<size_t>(query_base + h * kHeadSize + d)] += static_cast<float>(
              w * static_cast<double>(value[static_cast<size_t>(kv_base + (t * kNumKvHeads + kv) * kHeadSize + d)]));
        }
      }
    }
  }
  return out;
}

// One shape: `batch` sequences of `context_len` tokens, with the block pool and
// the block table that go with them. Built once per shape and shared by every
// rate, so the rates are compared on identical data.
struct Shape {
  int64_t batch = 1;
  int64_t context_len = 0;
  int64_t blocks_per_seq = 0;
  int64_t num_blocks = 0;
  std::vector<float> key;    // [batch, context_len, num_kv_heads, head_size]
  std::vector<float> value;  // same
  std::vector<float> query;  // [batch, num_heads, head_size]
  std::vector<int32_t> slots;        // [batch * context_len], the write path's scatter
  std::vector<int32_t> block_table;  // [batch, blocks_per_seq]
  std::vector<int32_t> context_lens;  // [batch]
};

// One mode's whole pipeline: write the cache, decode it, read the output back.
struct ModeRun {
  std::vector<float> output;
  std::vector<float> scale_plane;
  int64_t num_splits = 0;
  int64_t packed_bytes = 0;
  // Which query-rotation path this shape selected, and how it was tiled.
  vllm_ascend::turboquant::RotateQPlan rotate_plan;
};

ModeRun RunMode(tqm::TurboQuantMode mode, const Shape& shape, aclrtStream stream) {
  ModeRun run;
  run.packed_bytes = tqh::ModePackedBytes(mode, kHeadSize);
  const int64_t batch = shape.batch;
  const int64_t context_len = shape.context_len;

  DeviceBuffer key_dev = DeviceBuffer::FromHost(FloatToHalf(shape.key));
  DeviceBuffer value_dev = DeviceBuffer::FromHost(FloatToHalf(shape.value));
  DeviceBuffer query_dev = DeviceBuffer::FromHost(FloatToHalf(shape.query));
  DeviceBuffer slots_dev = DeviceBuffer::FromHost(shape.slots);
  DeviceBuffer block_table_dev = DeviceBuffer::FromHost(shape.block_table);
  DeviceBuffer context_dev = DeviceBuffer::FromHost(shape.context_lens);
  DeviceBuffer pi_signs = DeviceBuffer::FromHost(tqh::PiSigns(kHeadSize));

  // Two table images. The rotation one is the shipping 4-bit codec's, because
  // both kernels drive Pi through a TurboQuantCodec4; the mode one is this
  // rate's. The decode's mode image carries the NZ permutation
  // (nz_rows = kCubeTileRows); the write path's does not.
  DeviceBuffer rot_tables = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, 1));
  DeviceBuffer write_tables = DeviceBuffer::FromHost(tqh::ModeTables(mode, kHeadSize, 1, /*nz_rows=*/0));
  DeviceBuffer decode_tables =
      DeviceBuffer::FromHost(tqh::ModeTables(mode, kHeadSize, tqh::kUnpackRows, tqh::kCubeTileRows));

  DeviceBuffer key_cache = DeviceBuffer::Empty<int8_t>(
      tqh::ModePackedCacheBytes(mode, shape.num_blocks, kBlockSize, kNumKvHeads, kHeadSize));
  DeviceBuffer value_cache = DeviceBuffer::Empty<int8_t>(key_cache.size_bytes());
  DeviceBuffer scale_plane =
      DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(shape.num_blocks, kBlockSize, kNumKvHeads));
  DeviceBuffer out = DeviceBuffer::Empty<Half>(static_cast<size_t>(batch * kNumHeads * kHeadSize));
  DeviceBuffer h16 = DeviceBuffer::FromHost(tqh::Hadamard16Half());
  DeviceBuffer query_rot = DeviceBuffer::Empty<float>(static_cast<size_t>(batch * kNumHeads * kHeadSize));

  bool queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&queried);
  // The write path's token count is the whole batch's, because every sequence's
  // context has to be encoded into the pool; the decode's is one step per
  // sequence, which is what `batch` counts.
  const tqh::ReshapeAndCacheGrid write_grid = tqh::PlanReshapeAndCache(batch * context_len, aiv_num);
  const tqh::CubeDecodeGrid decode_grid =
      tqh::PlanCubeDecode(batch, kNumHeads, kNumKvHeads, kHeadSize, shape.blocks_per_seq, aiv_num);
  run.num_splits = decode_grid.num_splits;

  DeviceBuffer workspace = DeviceBuffer::Empty<float>(decode_grid.workspace_floats);

  turboquant_mm_reshape_and_cache_impl(
      static_cast<int32_t>(mode), AscendType::FP16, stream, write_grid.block_dim, key_dev.get(), value_dev.get(),
      key_cache.get(), value_cache.get(), scale_plane.get(), slots_dev.get(), pi_signs.get(), rot_tables.get(),
      write_tables.get(), static_cast<uint32_t>(batch * context_len), static_cast<uint32_t>(kNumKvHeads),
      static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize), write_grid.tokens_per_core,
      kInvSqrtHeadSize);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  // Pi q for the whole step, once, before the split reads it. `run.rotate_plan`
  // records which of the two paths the shape selected so the case can assert on
  // it rather than infer it.
  run.rotate_plan = tqh::RotateQuery(stream, AscendType::FP16, query_dev.get(), pi_signs.get(), h16.get(),
                                     rot_tables.get(), query_rot.get(), batch, kNumHeads, kHeadSize, aiv_num,
                                     /*input_exact_in_half=*/true);

  turboquant_mm_decode_split_impl(
      static_cast<int32_t>(mode), AscendType::FP16, stream, decode_grid.split_block_dim, query_rot.get(),
      key_cache.get(), value_cache.get(), scale_plane.get(), block_table_dev.get(), context_dev.get(),
      decode_tables.get(), workspace.get(), static_cast<uint32_t>(batch),
      static_cast<uint32_t>(kNumHeads), static_cast<uint32_t>(kNumKvHeads), static_cast<uint32_t>(kHeadSize),
      static_cast<uint32_t>(kBlockSize), static_cast<uint32_t>(shape.blocks_per_seq),
      static_cast<uint32_t>(decode_grid.num_splits), decode_grid.split_tasks_per_core, kAttentionScale,
      kInvSqrtHeadSize);

  // The combine is the AIV path's, reused verbatim: the partials above are in
  // its layout. Stream order is the barrier -- an in-kernel one would order
  // only co-resident blocks.
  turboquant_paged_attention_combine_impl(AscendType::FP16, stream, decode_grid.combine_block_dim, workspace.get(),
                                          out.get(), static_cast<uint32_t>(batch), static_cast<uint32_t>(kNumHeads),
                                          static_cast<uint32_t>(kHeadSize),
                                          static_cast<uint32_t>(decode_grid.num_splits),
                                          decode_grid.combine_tasks_per_core);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  // Rotated on the device; un-rotated here as the folded W_o would.
  run.output = tqh::UnrotateHeads(HalfToFloat(out.ToHost<Half>()), kHeadSize);
  run.scale_plane = scale_plane.ToHost<float>();
  return run;
}

// The modes this process runs, from ASCEND_TQ_SIM_MODES as a comma-separated
// list of the names below; empty or unset runs all three.
//
// One mode per process is the useful form when what is being asked is "does
// this rate fault or hang at this shape", because a camodel pass is minutes and
// a fault in the first mode costs the others. gtest's own --gtest_filter cannot
// express it: the three modes are a loop inside one test, not three tests.
// One shape's inputs. Every sequence gets its own run of blocks out of one
// shuffled pool, so no two sequences share a block and the block table is a
// scatter in both axes.
Shape MakeShape(int64_t batch, int64_t context_len, DeterministicRandom* rng) {
  Shape shape;
  shape.batch = batch;
  shape.context_len = context_len;
  shape.blocks_per_seq = BlocksPerSeq(context_len);
  shape.num_blocks = NumBlocks(batch, context_len);

  const size_t kv_elems = static_cast<size_t>(batch * context_len * kNumKvHeads * kHeadSize);
  shape.key = rng->NormalHalfExact(kv_elems, 0.0f, 1.0f);
  shape.value = rng->NormalHalfExact(kv_elems, 0.0f, 1.0f);
  shape.query = rng->NormalHalfExact(static_cast<size_t>(batch * kNumHeads * kHeadSize), 0.0f, 1.0f);

  const std::vector<int32_t> permutation = rng->Permutation(static_cast<int32_t>(shape.num_blocks));
  shape.block_table.resize(static_cast<size_t>(batch * shape.blocks_per_seq));
  shape.context_lens.assign(static_cast<size_t>(batch), static_cast<int32_t>(context_len));
  shape.slots.resize(static_cast<size_t>(batch * context_len));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t i = 0; i < shape.blocks_per_seq; ++i) {
      shape.block_table[static_cast<size_t>(b * shape.blocks_per_seq + i)] =
          permutation[static_cast<size_t>(b * shape.blocks_per_seq + i)];
    }
    for (int64_t i = 0; i < context_len; ++i) {
      const int32_t block = shape.block_table[static_cast<size_t>(b * shape.blocks_per_seq + i / kBlockSize)];
      shape.slots[static_cast<size_t>(b * context_len + i)] =
          block * static_cast<int32_t>(kBlockSize) + static_cast<int32_t>(i % kBlockSize);
    }
  }
  return shape;
}

bool ModeSelected(const char* name) {
  const char* raw = std::getenv("ASCEND_TQ_SIM_MODES");
  if (raw == nullptr || *raw == '\0') {
    return true;
  }
  const std::string wanted(raw);
  const std::string needle(name);
  size_t at = wanted.find(needle);
  while (at != std::string::npos) {
    const bool left = at == 0 || wanted[at - 1] == ',';
    const size_t after = at + needle.size();
    const bool right = after == wanted.size() || wanted[after] == ',';
    if (left && right) {
      return true;
    }
    at = wanted.find(needle, at + 1);
  }
  return false;
}

TEST(TurboQuantMultiMode, DispatchAndFidelityAcrossShapes) {
  REQUIRE_ASCEND_950PR();
  // No test-level opt-in any more: kv4fp8 runs by default and asserts its
  // fidelity bound, because the Cube-native decode is corrected and measured
  // (see TURBOQUANT_TESTS.md sections 13.8 and 13.9). The gate moved onto the
  // two rates that are still scaffolded -- see `provisional` in the table
  // below -- so promoting one rate does not promote the others.

  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  struct ModeCase {
    tqm::TurboQuantMode mode;
    const char* name;
    bool assert_fidelity;
    // Still behind VLLM_ASCEND_TQ_CUBE_WIP. kv4fp8 is not: it is the rate the
    // device verification runs, and it is measured rather than scaffolded.
    // Neither of the others has been re-measured through the corrected
    // staging, which is the whole reason they stay gated.
    bool provisional;
  };
  // kv4fp8 first: it is the one that runs by default and carries the bound, so
  // a failure there is reported before a multi-minute pass has been spent on a
  // rate nobody is gating a release on.
  const ModeCase cases[] = {
      {tqm::TurboQuantMode::KV4_FP8, "kv4fp8", true, false},
      {tqm::TurboQuantMode::KV5_FP8, "kv5fp8", true, true},
      {tqm::TurboQuantMode::KV3_FP4, "kv3fp4", false, true},
  };

  std::printf("[ multimode ] tier=%s head_size=%lld heads=%lld kv_heads=%lld block=%lld\n",
              VLLM_ASCEND_TEST_TIER, static_cast<long long>(kHeadSize), static_cast<long long>(kNumHeads),
              static_cast<long long>(kNumKvHeads), static_cast<long long>(kBlockSize));
  std::printf("[ multimode ] sweep: batch x context = %zu x %zu = %zu shape(s)\n", BatchSizes().size(),
              ContextLens().size(), BatchSizes().size() * ContextLens().size());

  // Every shape gets its own generator seeded the same way, so a shape's inputs
  // do not depend on which other shapes the sweep happened to include -- a
  // narrowed rerun reproduces the case it narrowed to.
  for (const int64_t batch : BatchSizes()) {
    for (const int64_t context_len : ContextLens()) {
      DeterministicRandom rng(0x5A17u);
      const Shape shape = MakeShape(batch, context_len, &rng);
      const std::vector<float> reference = HostAttention(batch, context_len, shape.query, shape.key, shape.value);

      std::printf("\n[ multimode ] shape: B=%lld S=%lld pool=%lld blocks/seq=%lld\n",
                  static_cast<long long>(batch), static_cast<long long>(context_len),
                  static_cast<long long>(shape.num_blocks), static_cast<long long>(shape.blocks_per_seq));

      for (const ModeCase& mode_case : cases) {
        if (!ModeSelected(mode_case.name)) {
          std::printf("[ multimode ] %s: not selected by ASCEND_TQ_SIM_MODES; skipped\n", mode_case.name);
          continue;
        }
        if (mode_case.provisional && !CubeWipOptedIn()) {
          std::printf("[ multimode ] %s: still provisional, not re-measured through the corrected staging; "
                      "set VLLM_ASCEND_TQ_CUBE_WIP=1 to run it\n",
                      mode_case.name);
          continue;
        }
        const tqm::TurboQuantModeConfig cfg = tqm::TurboQuantModeConfigOf(mode_case.mode);
        std::printf("[ multimode ] %s: %d bits, %d levels, %lld packed bytes/vector, %s, %s\n", mode_case.name,
                    cfg.bits, cfg.levels, static_cast<long long>(cfg.PackedBytes(kHeadSize)),
                    cfg.operand == tqm::TurboQuantOperand::kFp4E2m1 ? "Cube mad_mx (fp4 e2m1)"
                                                                    : "Cube mad (fp8 e4m3fn)",
                    cfg.is_affine ? "affine INT4" : "Lloyd-Max index");
        ASSERT_TRUE(cfg.IsBurstAligned(kHeadSize))
            << mode_case.name << " slot is not a whole 32-byte burst; every DMA on this path assumes it is";
        ASSERT_EQ(kBlockSize % 64, 0) << "block_size must be a multiple of kCubeTileRows (64): CopyInTile issues "
                                         "one unconditional 64-row DataCopy per tile";

        const ModeRun run = RunMode(mode_case.mode, shape, stream);

        // Dispatch: reaching here at all means every launch returned ACL_SUCCESS
        // and the stream drained. On the camodel a pipeline fault does not return
        // quietly, so this is the instruction-level check.
        std::printf("[ multimode ] %s: dispatched, splits=%lld\n", mode_case.name,
                    static_cast<long long>(run.num_splits));

        size_t finite = 0;
        double abs_sum = 0.0;
        for (const float v : run.output) {
          if (std::isfinite(v)) {
            ++finite;
          }
          abs_sum += std::fabs(static_cast<double>(v));
        }
        EXPECT_EQ(finite, run.output.size()) << mode_case.name << " produced non-finite output";
        EXPECT_GT(abs_sum, 0.0) << mode_case.name << " produced an identically zero output";

        // The scale plane is written by the encoder and read by the decoder; an
        // all-zero one means the write path never ran, which would make the decode
        // a decode of zeros and still finite.
        double scale_sum = 0.0;
        for (const float v : run.scale_plane) {
          scale_sum += std::fabs(static_cast<double>(v));
        }
        EXPECT_GT(scale_sum, 0.0) << mode_case.name << " left the scale plane empty";

        // One cosine over the whole batch, and then the per-sequence worst. The
        // batched figure alone would hide a single wrong sequence in seven right
        // ones, which is exactly the failure a batch axis introduces.
        const double cos = Cosine(reference, run.output);
        double worst_cos = cos;
        int64_t worst_seq = 0;
        if (batch > 1) {
          const size_t stride = static_cast<size_t>(kNumHeads * kHeadSize);
          for (int64_t b = 0; b < batch; ++b) {
            const size_t base = static_cast<size_t>(b) * stride;
            const std::vector<float> ref_seq(reference.begin() + static_cast<std::ptrdiff_t>(base),
                                             reference.begin() + static_cast<std::ptrdiff_t>(base + stride));
            const std::vector<float> got_seq(run.output.begin() + static_cast<std::ptrdiff_t>(base),
                                             run.output.begin() + static_cast<std::ptrdiff_t>(base + stride));
            const double seq_cos = Cosine(ref_seq, got_seq);
            if (seq_cos < worst_cos) {
              worst_cos = seq_cos;
              worst_seq = b;
            }
          }
          std::printf("[ multimode ] %s: cos vs fp32 host reference = %.6f (batch), worst sequence %lld = %.6f\n",
                      mode_case.name, cos, static_cast<long long>(worst_seq), worst_cos);
        } else {
          std::printf("[ multimode ] %s: cos vs fp32 host reference = %.6f\n", mode_case.name, cos);
        }

        if (mode_case.assert_fidelity) {
          EXPECT_GT(worst_cos, kSmokeCos)
              << mode_case.name << " decode does not track the host reference at B=" << batch << " S=" << context_len
              << "; this bound is a structural check, not the cos > 0.995 gate -- see the file header";
        }
      }
    }
  }
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
