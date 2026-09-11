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

// Single-shot functional smoke of the Cube-native multi-mode decode on the
// arch35 camodel.
//
// The camodel simulates the pipeline cycle by cycle, so this file runs each
// mode EXACTLY ONCE and asserts nothing about time. Every timing this project
// quotes comes from bench_device_950pr_turboquant on physical silicon. Do not
// add a loop to this file.
//
// THE SHAPE:
//
//   head_size 256     the production head dim, and the one the Cube path is
//                     sized for. It cannot drop to 64: the score GEMM's
//                     reduction length is head_size and the Cube's K step is
//                     64, so a smaller head would stop exercising the
//                     multi-step LoadData.
//   context 16        one partial tile of the 64-row Cube tile, so the tail
//                     masking in the softmax is live.
//   block_size 16     one paged block, read through a non-identity block table.
//   4 heads / 2 kv    a GQA group of two, so the query heads really are batched
//                     into the GEMM's M dimension.
//
// WHAT IS ASSERTED, in increasing strength:
//
//   dispatch    every mode's write and decode launch returns ACL_SUCCESS and the
//               stream synchronises. This is the only thing kv3fp4 is held to.
//   finite      every mode's output is entirely finite and not identically zero.
//   fidelity    kv4fp8's and kv5fp8's output against an fp32 host reference.
//               Both expand onto the same fp8 e4m3fn operand grid and differ
//               only in the codebook, so a structural defect in one -- a
//               transposed K operand, a nibble read from the wrong plane, an NZ
//               band placed at the wrong stride -- shows up in both. The bound
//               is deliberately loose (see kSmokeCos) and is NOT the cos > 0.995
//               gate, whose measurement lives in
//               scripts/tq_multimode_calibration.py; there kv4fp8 reaches
//               0.98785 and kv5fp8 0.99613 at S = 512.
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

constexpr int64_t kHeadSize = 256;
constexpr int64_t kNumHeads = 4;
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
constexpr int64_t kDefaultContextLen = 16;
constexpr int64_t kQueryTokens = 1;
constexpr int64_t kMinBlocks = 4;

// The context length this process decodes at, from ASCEND_TQ_SIM_CONTEXT.
//
// It is a run-time value and not a constant because the thing this binary is
// used for varies with it: at the default 16 the decode is a single Cube tile
// and what is being checked is that the pipeline runs at all, while at 512 or
// 2048 it is many tiles and what is being checked is multi-tile addressing and
// deep-context stability. One process, one context length -- the camodel is
// cycle-level and a sweep inside one process is a sweep of multi-minute passes.
//
// Parsed once. Anything unparseable or non-positive falls back to the default
// rather than running a shape nobody asked for.
int64_t ContextLen() {
  static const int64_t value = [] {
    const char* raw = std::getenv("ASCEND_TQ_SIM_CONTEXT");
    if (raw == nullptr || *raw == '\0') {
      return kDefaultContextLen;
    }
    const long long parsed = std::strtoll(raw, nullptr, 10);
    if (parsed <= 0) {
      std::printf("[ multimode ] ASCEND_TQ_SIM_CONTEXT='%s' did not parse; using %lld\n", raw,
                  static_cast<long long>(kDefaultContextLen));
      return kDefaultContextLen;
    }
    return static_cast<int64_t>(parsed);
  }();
  return value;
}

int64_t BlocksPerSeq() { return (ContextLen() + kBlockSize - 1) / kBlockSize; }

// The block pool. Deliberately larger than the sequence needs so the block
// table is a genuine scatter rather than a run of consecutive blocks, which is
// what makes the paged addressing real -- at the default context that is the
// original pool of 4 for a 1-block sequence.
int64_t NumBlocks() { return std::max(kMinBlocks, BlocksPerSeq() * 4); }
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
std::vector<float> HostAttention(const std::vector<float>& query, const std::vector<float>& key,
                                 const std::vector<float>& value) {
  const int64_t context_len = ContextLen();
  std::vector<float> out(static_cast<size_t>(kNumHeads * kHeadSize), 0.0f);
  const int64_t heads_per_kv = kNumHeads / kNumKvHeads;
  for (int64_t h = 0; h < kNumHeads; ++h) {
    const int64_t kv = h / heads_per_kv;
    std::vector<double> logits(static_cast<size_t>(context_len), 0.0);
    double max_logit = -1e30;
    for (int64_t t = 0; t < context_len; ++t) {
      double dot = 0.0;
      for (int64_t d = 0; d < kHeadSize; ++d) {
        dot += static_cast<double>(query[static_cast<size_t>(h * kHeadSize + d)]) *
               static_cast<double>(key[static_cast<size_t>((t * kNumKvHeads + kv) * kHeadSize + d)]);
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
        out[static_cast<size_t>(h * kHeadSize + d)] += static_cast<float>(
            w * static_cast<double>(value[static_cast<size_t>((t * kNumKvHeads + kv) * kHeadSize + d)]));
      }
    }
  }
  return out;
}

// One mode's whole pipeline: write the cache, decode it, read the output back.
struct ModeRun {
  std::vector<float> output;
  std::vector<float> scale_plane;
  int64_t num_splits = 0;
  int64_t packed_bytes = 0;
};

ModeRun RunMode(tqm::TurboQuantMode mode, const std::vector<float>& key, const std::vector<float>& value,
                const std::vector<float>& query, const std::vector<int32_t>& slots,
                const std::vector<int32_t>& block_table, aclrtStream stream) {
  ModeRun run;
  run.packed_bytes = tqh::ModePackedBytes(mode, kHeadSize);

  DeviceBuffer key_dev = DeviceBuffer::FromHost(FloatToHalf(key));
  DeviceBuffer value_dev = DeviceBuffer::FromHost(FloatToHalf(value));
  DeviceBuffer query_dev = DeviceBuffer::FromHost(FloatToHalf(query));
  DeviceBuffer slots_dev = DeviceBuffer::FromHost(slots);
  DeviceBuffer block_table_dev = DeviceBuffer::FromHost(block_table);
  DeviceBuffer context_dev =
      DeviceBuffer::FromHost(std::vector<int32_t>{static_cast<int32_t>(ContextLen())});
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
      tqh::ModePackedCacheBytes(mode, NumBlocks(), kBlockSize, kNumKvHeads, kHeadSize));
  DeviceBuffer value_cache = DeviceBuffer::Empty<int8_t>(key_cache.size_bytes());
  DeviceBuffer scale_plane =
      DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(NumBlocks(), kBlockSize, kNumKvHeads));
  DeviceBuffer out = DeviceBuffer::Empty<Half>(static_cast<size_t>(kNumHeads * kHeadSize));

  bool queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&queried);
  const tqh::ReshapeAndCacheGrid write_grid = tqh::PlanReshapeAndCache(ContextLen(), aiv_num);
  const int64_t blocks_per_seq = (ContextLen() + kBlockSize - 1) / kBlockSize;
  const tqh::CubeDecodeGrid decode_grid =
      tqh::PlanCubeDecode(kQueryTokens, kNumHeads, kNumKvHeads, kHeadSize, blocks_per_seq, aiv_num);
  run.num_splits = decode_grid.num_splits;

  DeviceBuffer workspace = DeviceBuffer::Empty<float>(decode_grid.workspace_floats);

  turboquant_mm_reshape_and_cache_impl(
      static_cast<int32_t>(mode), AscendType::FP16, stream, write_grid.block_dim, key_dev.get(), value_dev.get(),
      key_cache.get(), value_cache.get(), scale_plane.get(), slots_dev.get(), pi_signs.get(), rot_tables.get(),
      write_tables.get(), static_cast<uint32_t>(ContextLen()), static_cast<uint32_t>(kNumKvHeads),
      static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(kBlockSize), write_grid.tokens_per_core,
      kInvSqrtHeadSize);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  turboquant_mm_decode_split_impl(
      static_cast<int32_t>(mode), AscendType::FP16, stream, decode_grid.split_block_dim, query_dev.get(),
      key_cache.get(), value_cache.get(), scale_plane.get(), block_table_dev.get(), context_dev.get(),
      pi_signs.get(), rot_tables.get(), decode_tables.get(), workspace.get(), static_cast<uint32_t>(kQueryTokens),
      static_cast<uint32_t>(kNumHeads), static_cast<uint32_t>(kNumKvHeads), static_cast<uint32_t>(kHeadSize),
      static_cast<uint32_t>(kBlockSize), static_cast<uint32_t>(blocks_per_seq),
      static_cast<uint32_t>(decode_grid.num_splits), decode_grid.split_tasks_per_core, kAttentionScale,
      kInvSqrtHeadSize);

  // The combine is the AIV path's, reused verbatim: the partials above are in
  // its layout. Stream order is the barrier -- an in-kernel one would order
  // only co-resident blocks.
  turboquant_paged_attention_combine_impl(
      AscendType::FP16, stream, decode_grid.combine_block_dim, workspace.get(), pi_signs.get(), rot_tables.get(),
      out.get(), static_cast<uint32_t>(kQueryTokens), static_cast<uint32_t>(kNumHeads),
      static_cast<uint32_t>(kHeadSize), static_cast<uint32_t>(decode_grid.num_splits),
      decode_grid.combine_tasks_per_core, kInvSqrtHeadSize);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  run.output = HalfToFloat(out.ToHost<Half>());
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

TEST(TurboQuantMultiMode, SingleShotDispatchAndFidelity) {
  REQUIRE_ASCEND_950PR();
  // No test-level opt-in any more: kv4fp8 runs by default and asserts its
  // fidelity bound, because the Cube-native decode is corrected and measured
  // (cos 0.982351 at S=64; see TURBOQUANT_TESTS.md section 13.8). The gate
  // moved onto the two rates that are still scaffolded -- see `provisional`
  // in the table below -- so promoting one rate does not promote the others.

  DeterministicRandom rng(0x5A17u);
  const size_t kv_elems = static_cast<size_t>(ContextLen() * kNumKvHeads * kHeadSize);
  const std::vector<float> key = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
  const std::vector<float> value = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
  const std::vector<float> query =
      rng.NormalHalfExact(static_cast<size_t>(kNumHeads * kHeadSize), 0.0f, 1.0f);

  // A non-identity block table, so paging is real rather than an offset of zero.
  const std::vector<int32_t> permutation = rng.Permutation(static_cast<int32_t>(NumBlocks()));
  const int64_t blocks_per_seq = (ContextLen() + kBlockSize - 1) / kBlockSize;
  const std::vector<int32_t> block_table(permutation.begin(),
                                         permutation.begin() + static_cast<size_t>(blocks_per_seq));
  std::vector<int32_t> slots(static_cast<size_t>(ContextLen()));
  for (int64_t i = 0; i < ContextLen(); ++i) {
    const int32_t block = block_table[static_cast<size_t>(i / kBlockSize)];
    slots[static_cast<size_t>(i)] = block * static_cast<int32_t>(kBlockSize) + static_cast<int32_t>(i % kBlockSize);
  }

  const std::vector<float> reference = HostAttention(query, key, value);

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

  std::printf("[ multimode ] shape: S=%lld head_size=%lld heads=%lld kv_heads=%lld block=%lld pool=%lld\n",
              static_cast<long long>(ContextLen()), static_cast<long long>(kHeadSize),
              static_cast<long long>(kNumHeads), static_cast<long long>(kNumKvHeads),
              static_cast<long long>(kBlockSize), static_cast<long long>(NumBlocks()));

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
    std::printf("[ multimode ] %s: %d bits, %d levels, %lld packed bytes/vector, %s\n", mode_case.name, cfg.bits,
                cfg.levels, static_cast<long long>(cfg.PackedBytes(kHeadSize)),
                cfg.operand == tqm::TurboQuantOperand::kFp4E2m1 ? "Cube mad_mx (fp4 e2m1)"
                                                                : "Cube mad (fp8 e4m3fn)");
    ASSERT_TRUE(cfg.IsBurstAligned(kHeadSize))
        << mode_case.name << " slot is not a whole 32-byte burst; every DMA on this path assumes it is";

    const ModeRun run = RunMode(mode_case.mode, key, value, query, slots, block_table, stream);

    // Dispatch: reaching here at all means both launches returned ACL_SUCCESS
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

    const double cos = Cosine(reference, run.output);
    std::printf("[ multimode ] %s: cos vs fp32 host reference = %.6f\n", mode_case.name, cos);

    if (mode_case.assert_fidelity) {
      EXPECT_GT(cos, kSmokeCos)
          << mode_case.name << " decode does not track the host reference at S=" << ContextLen()
          << "; this bound is a structural check, not the cos > 0.995 gate -- see the file header";
    }
  }
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
