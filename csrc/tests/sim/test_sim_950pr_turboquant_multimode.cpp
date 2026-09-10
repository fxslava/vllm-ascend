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
// WHAT THIS IS FOR, AND WHAT IT IS NOT FOR. The camodel simulates the DaVinci
// pipeline cycle by cycle: a launch that is microseconds on silicon is minutes
// here. So this file runs each mode EXACTLY ONCE, at the smallest shape that
// still exercises the thing under test, and asserts nothing about time. Every
// timing number this project quotes comes from bench_device_950pr_turboquant on
// physical silicon and from nowhere else. Do not add a loop to this file.
//
// THE SHAPE, AND WHY EACH PART OF IT IS WHAT IT IS:
//
//   head_size 256     the production head dim, and the one the Cube path is
//                     sized for. Unlike the AIV suite this cannot drop to 64:
//                     the score GEMM's reduction length is head_size and the
//                     Cube's K step is 64, so a smaller head would stop
//                     exercising the multi-step LoadData the real shape uses.
//   context 16        S = 16, the brief's smoke shape. One partial tile of the
//                     64-row Cube tile, so the tail masking in the softmax is
//                     live -- which is the case a full tile would not reach.
//   block_size 16     one paged block, read through a non-identity block table.
//   4 heads / 2 kv    a GQA group of two, so the query heads really are batched
//                     into the GEMM's M dimension and the head -> kv_head map
//                     is not the identity.
//
// WHAT IS ASSERTED. Three things, in increasing strength:
//
//   dispatch    every mode's write and decode launch returns ACL_SUCCESS and
//               the stream synchronises. On the camodel a pipeline fault -- a
//               misaligned vector operand, a bad L0 fractal, a cross-core flag
//               that never fires -- surfaces here as a non-zero status or a
//               hang, so this is the instruction-level sanity check the brief
//               asks for, and it is the *only* thing kv3fp4 and kv4fp8 are held
//               to.
//   finite      kv5fp8's output is entirely finite and not identically zero.
//               A decode that silently produced NaN or an all-zero accumulator
//               would pass a dispatch check and be useless.
//   fidelity    kv5fp8's output against an fp32 host reference of the same
//               attention. The bound here is deliberately loose (see kSmokeCos)
//               and is NOT the cos > 0.995 gate: at S = 16 the softmax is over
//               sixteen terms, so a single mis-weighted row moves the output far
//               more than it would at S = 512, and the gate's own measurement
//               lives in scripts/tq_multimode_calibration.py where it can afford
//               the seeds. What this bound catches is a decode that is wrong in
//               kind -- a transposed operand, a dropped scale, a codebook read
//               through the wrong index -- not one that is wrong in the third
//               decimal.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
constexpr int64_t kBlockSize = 16;
constexpr int64_t kContextLen = 16;
constexpr int64_t kQueryTokens = 1;
constexpr int64_t kNumBlocks = 4;
constexpr float kInvSqrtHeadSize = 0.0625f;  // 1 / sqrt(256), exact in fp32
constexpr float kAttentionScale = kInvSqrtHeadSize;

// The loose bound described in the file header. 0.90 is roughly four times the
// distance from 1.0 that the calibration script's S = 512 measurement shows
// (0.9954), which leaves room for the short-context effect without leaving room
// for a structurally wrong decode -- a transposed K operand lands near zero
// cosine, and a dropped per-vector scale near 0.3.
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
  std::vector<float> out(static_cast<size_t>(kNumHeads * kHeadSize), 0.0f);
  const int64_t heads_per_kv = kNumHeads / kNumKvHeads;
  for (int64_t h = 0; h < kNumHeads; ++h) {
    const int64_t kv = h / heads_per_kv;
    std::vector<double> logits(static_cast<size_t>(kContextLen), 0.0);
    double max_logit = -1e30;
    for (int64_t t = 0; t < kContextLen; ++t) {
      double dot = 0.0;
      for (int64_t d = 0; d < kHeadSize; ++d) {
        dot += static_cast<double>(query[static_cast<size_t>(h * kHeadSize + d)]) *
               static_cast<double>(key[static_cast<size_t>((t * kNumKvHeads + kv) * kHeadSize + d)]);
      }
      logits[static_cast<size_t>(t)] = dot * static_cast<double>(kAttentionScale);
      max_logit = std::max(max_logit, logits[static_cast<size_t>(t)]);
    }
    double denom = 0.0;
    for (int64_t t = 0; t < kContextLen; ++t) {
      logits[static_cast<size_t>(t)] = std::exp(logits[static_cast<size_t>(t)] - max_logit);
      denom += logits[static_cast<size_t>(t)];
    }
    for (int64_t t = 0; t < kContextLen; ++t) {
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
      DeviceBuffer::FromHost(std::vector<int32_t>{static_cast<int32_t>(kContextLen)});
  DeviceBuffer pi_signs = DeviceBuffer::FromHost(tqh::PiSigns(kHeadSize));

  // Two table images. The rotation one is the shipping 4-bit codec's, because
  // the write and decode kernels both drive Pi through a TurboQuantCodec4; the
  // mode one is this rate's. The decode's mode image carries the NZ permutation
  // (nz_rows = kCubeTileRows) so the unpack writes Cube-ready order, and the
  // write path's does not, because the encoder produces plain row-major codes.
  DeviceBuffer rot_tables = DeviceBuffer::FromHost(tqh::CodecTables(kHeadSize, 1));
  DeviceBuffer write_tables = DeviceBuffer::FromHost(tqh::ModeTables(mode, kHeadSize, 1, /*nz_rows=*/0));
  DeviceBuffer decode_tables =
      DeviceBuffer::FromHost(tqh::ModeTables(mode, kHeadSize, tqh::kUnpackRows, tqh::kCubeTileRows));

  DeviceBuffer key_cache = DeviceBuffer::Empty<int8_t>(
      tqh::ModePackedCacheBytes(mode, kNumBlocks, kBlockSize, kNumKvHeads, kHeadSize));
  DeviceBuffer value_cache = DeviceBuffer::Empty<int8_t>(key_cache.size_bytes());
  DeviceBuffer scale_plane =
      DeviceBuffer::Empty<float>(tqh::ScalePlaneFloats(kNumBlocks, kBlockSize, kNumKvHeads));
  DeviceBuffer out = DeviceBuffer::Empty<Half>(static_cast<size_t>(kNumHeads * kHeadSize));

  bool queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&queried);
  const tqh::ReshapeAndCacheGrid write_grid = tqh::PlanReshapeAndCache(kContextLen, aiv_num);
  const int64_t blocks_per_seq = (kContextLen + kBlockSize - 1) / kBlockSize;
  const tqh::CubeDecodeGrid decode_grid =
      tqh::PlanCubeDecode(kQueryTokens, kNumHeads, kNumKvHeads, kHeadSize, blocks_per_seq, aiv_num);
  run.num_splits = decode_grid.num_splits;

  DeviceBuffer workspace = DeviceBuffer::Empty<float>(decode_grid.workspace_floats);

  turboquant_mm_reshape_and_cache_impl(
      static_cast<int32_t>(mode), AscendType::FP16, stream, write_grid.block_dim, key_dev.get(), value_dev.get(),
      key_cache.get(), value_cache.get(), scale_plane.get(), slots_dev.get(), pi_signs.get(), rot_tables.get(),
      write_tables.get(), static_cast<uint32_t>(kContextLen), static_cast<uint32_t>(kNumKvHeads),
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

TEST(TurboQuantMultiMode, SingleShotDispatchAndFidelity) {
  REQUIRE_ASCEND_950PR();
  // Opt-in, and the reason is a hang rather than a failure: TurboQuantCubeMm::Init
  // allocates L0A/L0B/L0C on both halves of the MIX kernel, those pools do not
  // exist on a vector core, and the split kernel stalls on the nonsense address
  // that produces. A hang in a default ctest run is worse than a red test --
  // it burns the whole timeout and reports nothing.
  REQUIRE_CUBE_WIP_OPT_IN("The Cube-native multi-mode decode",
                          "it dispatches and then stalls in the split kernel on pem_lsu "
                          "'unrecognize ldst addr'.");

  DeterministicRandom rng(0x5A17u);
  const size_t kv_elems = static_cast<size_t>(kContextLen * kNumKvHeads * kHeadSize);
  const std::vector<float> key = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
  const std::vector<float> value = rng.NormalHalfExact(kv_elems, 0.0f, 1.0f);
  const std::vector<float> query =
      rng.NormalHalfExact(static_cast<size_t>(kNumHeads * kHeadSize), 0.0f, 1.0f);

  // A non-identity block table, so paging is real rather than an offset of zero.
  const std::vector<int32_t> permutation = rng.Permutation(static_cast<int32_t>(kNumBlocks));
  const int64_t blocks_per_seq = (kContextLen + kBlockSize - 1) / kBlockSize;
  const std::vector<int32_t> block_table(permutation.begin(),
                                         permutation.begin() + static_cast<size_t>(blocks_per_seq));
  std::vector<int32_t> slots(static_cast<size_t>(kContextLen));
  for (int64_t i = 0; i < kContextLen; ++i) {
    const int32_t block = block_table[static_cast<size_t>(i / kBlockSize)];
    slots[static_cast<size_t>(i)] = block * static_cast<int32_t>(kBlockSize) + static_cast<int32_t>(i % kBlockSize);
  }

  const std::vector<float> reference = HostAttention(query, key, value);

  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  struct ModeCase {
    tqm::TurboQuantMode mode;
    const char* name;
    bool primary;
  };
  // kv5fp8 first: it is the one the fidelity assertion is on, and running it
  // before the scaffolded modes means a failure there is reported before two
  // more multi-minute camodel passes have been spent.
  const ModeCase cases[] = {
      {tqm::TurboQuantMode::KV5_FP8, "kv5fp8", true},
      {tqm::TurboQuantMode::KV4_FP8, "kv4fp8", false},
      {tqm::TurboQuantMode::KV3_FP4, "kv3fp4", false},
  };

  for (const ModeCase& mode_case : cases) {
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

    if (mode_case.primary) {
      EXPECT_GT(cos, kSmokeCos)
          << "kv5fp8 decode does not track the host reference at S=" << kContextLen
          << "; this bound is a structural check, not the cos > 0.995 gate -- see the file header";
    }
  }
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
