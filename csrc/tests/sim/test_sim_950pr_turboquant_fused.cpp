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

// The fused single-launch Cube decode (TURBOQUANT_TESTS.md 13.25), kv4fp8, on six shapes (D 256 unless noted):
//
//   (a) H_Q 4,  H_KV 2, S 64,  block 64   one tile
//   (b) H_Q 4,  H_KV 2, S 256, block 64   four tiles in one task: the L1 slot ring and its free edge
//   (c) (b) planned to fill the MIX blocks: parallel splits and the in-launch reduction
//   (d) H_Q 8,  H_KV 2, S 120, block 64   a masked tail tile, two heads on each vector subcore
//   (e) (a) with its cache written by the kv4fp8 kernel writer (13.28) instead of uploaded
//   (f) H_Q 4,  H_KV 1, D 128, S 64,  block 64   Qwen3.5's 4:1 group at D = 128: a smoke of the D = 128 Cube
//       fractals (4 C0 groups per row) ahead of the silicon bench (13.31)
//
// Each fused output is hashed (FNV-1a over its half bit patterns) against a golden, and its cosine against
// exact fp32 attention must not regress. The goldens are the bit-exact reference: the barriered A/B instance
// they were once checked against was retired in TURBOQUANT_TESTS.md 13.30, when no switchable vector barrier
// was left for it to differ by. (a) to (d) decode a host-built cache and their goldens date from b48ed2951;
// (c) must agree with (b) to rounding. (e) draws independent vector halves, so a swapped nibble lane would show, and
// checks the written GM cache against the host encoder before decoding it.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "camodel_guard.hpp"
#include "test_harness.hpp"
#include "turboquant_fused_harness.hpp"
#include "turboquant_launch.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace tqh = turboquant_host;

constexpr int64_t kLaunchBudgetSeconds = 900;

// Recorded values are printed to six places; anything below one by more than that rounding regressed.
constexpr double kCosineRounding = 5e-7;
// A split launch reduces with the same online-softmax recurrence as the unsplit one, so the two may
// differ by float rounding only.
constexpr double kSplitAgreementCosine = 0.999999;
constexpr double kSplitFidelityShift = 2 * kCosineRounding;
constexpr uint64_t kGoldenUnrecorded = 0;

constexpr int64_t kSingleTileContext = 64;
constexpr int64_t kRingContext = 256;
constexpr int64_t kTailContext = 120;
constexpr int64_t kWideGroupHeads = 8;
constexpr int64_t kNarrowGroupHeads = 4;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kHeadSize = 256;
constexpr int64_t kQwenHeadSize = 128;
constexpr int64_t kQwenKvHeads = 1;
// A D = 128 decode whose fractal strides or C0 grouping were wrong would not reach this against exact fp32
// attention; the kv4fp8 D = 256 cases sit near 0.9994.
constexpr double kHeadSize128CosineFloor = 0.99;
constexpr int64_t kBlockSize = 64;
// Two MIX blocks for (d): the planner then keeps each GQA group of four in one task, two heads per subcore.
constexpr int64_t kTwoBlockAiv = 4;
constexpr int64_t kWideTaskHeads = 4;
constexpr tqh::FusedSplitPolicy kFillBlocks = tqh::FusedSplitPolicy::kFillBlocks;
constexpr tqh::FusedSplitPolicy kContextOnly = tqh::FusedSplitPolicy::kContextOnly;
// The writer's scales are float RMS values of vectors that went through fp16 and back through the rotation;
// the host's are double RMS values of the vectors before either.
constexpr double kWrittenScaleTolerance = 2e-3;

struct FusedCase {
  const char* tag;
  tqh::FusedShape shape;
  int64_t plan_aiv;
  tqh::FusedSplitPolicy split_policy;
  double recorded_cosine;
  uint64_t golden;
};

tqh::FusedShape Shape(int64_t num_heads, int64_t context_len, bool kernel_writer = false,
                      int64_t num_kv_heads = kKvHeads, int64_t head_size = kHeadSize) {
  tqh::FusedShape shape;
  shape.num_heads = num_heads;
  shape.num_kv_heads = num_kv_heads;
  shape.head_size = head_size;
  shape.block_size = kBlockSize;
  shape.context_len = context_len;
  shape.kernel_writer = kernel_writer;
  return shape;
}

// Goldens and cosines recorded 2026-09-16 on the Ascend950PR_9589 camodel (aiv 64) from b48ed2951, whose
// (a) is bit-identical to the retired split + combine. (c) was then planned with a fused limit of 0, the
// same grid the fill policy produces: 4 splits, 8 blocks, 2 heads per task.
const FusedCase kCaseA = {"(a)", Shape(kNarrowGroupHeads, kSingleTileContext), 0, kFillBlocks, 0.999407,
                          0x6176461416358ec1ull};
const FusedCase kCaseB = {"(b)", Shape(kNarrowGroupHeads, kRingContext), 0, kContextOnly, 0.999400,
                          0x470dad36e6708da5ull};
const FusedCase kCaseC = {"(c)", Shape(kNarrowGroupHeads, kRingContext), 0, kFillBlocks, 0.999400,
                          0xda6a2c77e20ff4a1ull};
const FusedCase kCaseD = {"(d)", Shape(kWideGroupHeads, kTailContext), kTwoBlockAiv, kFillBlocks, 0.999517,
                          0xcc1fcdfed39b48e5ull};
// Recorded 2026-09-17 on the same camodel from the NZ-tiled kernel writer (TURBOQUANT_TESTS.md 13.28).
const FusedCase kCaseE = {"(e)", Shape(kNarrowGroupHeads, kSingleTileContext, true), 0, kFillBlocks, 0.999576,
                          0x9ffe02efde1506cfull};
// Recorded 2026-09-17 on the same camodel from 539d4ab63 plus this case (TURBOQUANT_TESTS.md 13.32).
const FusedCase kCaseF = {"(f)", Shape(kNarrowGroupHeads, kSingleTileContext, false, kQwenKvHeads, kQwenHeadSize), 0,
                          kFillBlocks, 0.999543, 0xedc60ea1714295f1ull};

void PrintShape(const FusedCase& fused_case, int64_t aiv_num, bool queried) {
  const tqh::FusedShape& shape = fused_case.shape;
  std::printf("[ fused ] %s kv4fp8 H_Q=%lld H_KV=%lld D=%lld block=%lld S=%lld aiv=%lld%s plan_aiv=%lld %s%s\n",
              fused_case.tag, static_cast<long long>(shape.num_heads), static_cast<long long>(shape.num_kv_heads),
              static_cast<long long>(shape.head_size), static_cast<long long>(shape.block_size),
              static_cast<long long>(shape.context_len), static_cast<long long>(aiv_num),
              queried ? "" : " (fallback core count)",
              static_cast<long long>(fused_case.plan_aiv > 0 ? fused_case.plan_aiv : aiv_num),
              fused_case.split_policy == kFillBlocks ? "fill-blocks" : "context-only",
              shape.kernel_writer ? " kernel-writer" : "");
}

void PrintRun(const char* tag, const char* label, const tqh::FusedRun& run, const std::vector<float>& reference) {
  std::printf("  %s %-16s launches=%lld splits=%lld block_dim=%u heads/task=%u limit=%u untouched=%zu "
              "fnv1a=0x%016llx cos_vs_fp32=%.6f %.1f s\n",
              tag, label, static_cast<long long>(run.launches), static_cast<long long>(run.num_splits),
              run.block_dim, run.heads_per_task, run.fused_context_limit, run.untouched,
              static_cast<unsigned long long>(run.fnv1a), tqh::FusedCosine(run.output, reference), run.host_s);
}

// Runs a case through the fused kernel and checks what every case shares. Returns the fused run.
tqh::FusedRun RunCase(tqh::FusedCubeScenario* scenario, const FusedCase& fused_case, LaunchWatchdog* watchdog,
                      const std::vector<float>& reference) {
  const tqh::FusedDecodeGrid grid = scenario->Plan(fused_case.plan_aiv, fused_case.split_policy);
  const std::string tag(fused_case.tag);

  watchdog->Arm(tag + " turboquant_mm_fused_decode_impl");
  const tqh::FusedRun fused = scenario->RunFused(grid);
  watchdog->Disarm();
  PrintRun(fused_case.tag, "fused", fused, reference);

  const double cosine = tqh::FusedCosine(fused.output, reference);
  if (fused_case.golden == kGoldenUnrecorded) {
    std::printf("[ fused ] %s golden not recorded; this build's is 0x%016llx\n", fused_case.tag,
                static_cast<unsigned long long>(fused.fnv1a));
  }
  if (fused_case.recorded_cosine > 0.0) {
    std::printf("[ fused ] %s cos vs fp32 %.6f (recorded %.6f)\n", fused_case.tag, cosine,
                fused_case.recorded_cosine);
  }

  EXPECT_EQ(fused.launches, 1) << tag;
  EXPECT_EQ(fused.untouched, 0u) << tag << ": the fused launch left sentinel values in the output";
  if (fused_case.golden != kGoldenUnrecorded) {
    EXPECT_EQ(fused.fnv1a, fused_case.golden) << tag << ": the output moved off its recorded golden";
  }
  if (fused_case.recorded_cosine > 0.0) {
    EXPECT_GE(cosine, fused_case.recorded_cosine - kCosineRounding) << tag << ": cos vs exact fp32 regressed";
  }
  return fused;
}

void ExpectNoExceptionDumps(const char* tag) {
  const DumpCensus dumps = ExceptionDumps();
  std::printf("[ fused ] %s excp dumps in cwd: %zu files, %zu non-empty, %llu B\n", tag, dumps.files,
              dumps.non_empty, static_cast<unsigned long long>(dumps.bytes));
  EXPECT_EQ(dumps.non_empty, 0u) << tag << ": the camodel wrote an exception dump";
}

class TurboQuantFusedDecode : public ::testing::Test {
 protected:
  void SetUp() override {
    REQUIRE_ASCEND_950PR();
    stream_ = AscendTestEnvironment::Instance().stream();
    aiv_num_ = tqh::VectorCoreNum(&queried_);
  }

  aclrtStream stream_ = nullptr;
  int64_t aiv_num_ = 0;
  bool queried_ = false;
  LaunchWatchdog watchdog_{kLaunchBudgetSeconds, "[ fused ]"};
};

TEST_F(TurboQuantFusedDecode, DecodesASingleTileToItsGolden) {
  PrintShape(kCaseA, aiv_num_, queried_);
  watchdog_.Arm("(a) scenario setup and query rotation");
  tqh::FusedCubeScenario scenario(kCaseA.shape, stream_, aiv_num_);
  watchdog_.Disarm();

  const tqh::FusedRun fused = RunCase(&scenario, kCaseA, &watchdog_, scenario.Reference());
  EXPECT_EQ(fused.num_splits, 1) << "a context inside the fused limit is never split";
  ExpectNoExceptionDumps(kCaseA.tag);
}

TEST_F(TurboQuantFusedDecode, SlotRingAndItsSplitReductionAgree) {
  PrintShape(kCaseB, aiv_num_, queried_);
  PrintShape(kCaseC, aiv_num_, queried_);
  watchdog_.Arm("(b) scenario setup and query rotation");
  tqh::FusedCubeScenario scenario(kCaseB.shape, stream_, aiv_num_);
  watchdog_.Disarm();
  const std::vector<float> reference = scenario.Reference();

  const tqh::FusedRun unsplit = RunCase(&scenario, kCaseB, &watchdog_, reference);
  const tqh::FusedRun split = RunCase(&scenario, kCaseC, &watchdog_, reference);

  const double agreement = tqh::FusedCosine(split.output, unsplit.output);
  const double unsplit_cos = tqh::FusedCosine(unsplit.output, reference);
  const double split_cos = tqh::FusedCosine(split.output, reference);
  std::printf("[ fused ] (c) vs (b): cos %.9f; cos vs fp32 %.9f split, %.9f unsplit\n", agreement, split_cos,
              unsplit_cos);

  // Four tiles in one task is what reaches the slot-free edge (tile + kCubeSlots < numTiles).
  EXPECT_EQ(scenario.blocks_per_seq(), kRingContext / kBlockSize);
  EXPECT_EQ(unsplit.num_splits, 1) << "(b) must keep every tile in one task";
  EXPECT_EQ(unsplit.fused_context_limit, static_cast<uint32_t>(tqh::kFusedContextLimit));
  EXPECT_GT(split.num_splits, 1) << "(c) must exercise the in-launch reduction";
  EXPECT_EQ(split.fused_context_limit, 0u) << "a filled grid must tell the launch to reduce its splits";
  EXPECT_GE(agreement, kSplitAgreementCosine) << "the split reduction does not reproduce the unsplit decode";
  EXPECT_LE(std::fabs(split_cos - unsplit_cos), kSplitFidelityShift) << "the split moved cos vs exact fp32";
  ExpectNoExceptionDumps(kCaseC.tag);
}

TEST_F(TurboQuantFusedDecode, DecodesTheCacheTheKernelWriterWrote) {
  PrintShape(kCaseE, aiv_num_, queried_);
  watchdog_.Arm("(e) kernel cache write and query rotation");
  tqh::FusedCubeScenario scenario(kCaseE.shape, stream_, aiv_num_);
  watchdog_.Disarm();

  const tqh::WrittenCacheAgreement written = scenario.CompareWrittenCache();
  std::printf("[ fused ] (e) written cache: %zu of %zu packed bytes differ from the host encoder (%zu with the "
              "nibble lanes swapped); %zu scale lanes, %zu pad mismatches, max rel err %.3e\n",
              written.packed_mismatches, written.packed_bytes, written.swapped_lane_mismatches, written.scale_lanes,
              written.scale_pad_mismatches, written.max_scale_rel_err);
  EXPECT_EQ(written.packed_mismatches, 0u) << "the kernel writer's NZ-tiled bytes differ from the host encoder";
  EXPECT_EQ(written.scale_pad_mismatches, 0u) << "the kernel writer touched a pad lane or an unwritten slot";
  EXPECT_LE(written.max_scale_rel_err, kWrittenScaleTolerance) << "a written scale is not the RMS it encodes";

  const tqh::FusedRun fused = RunCase(&scenario, kCaseE, &watchdog_, scenario.Reference());
  EXPECT_EQ(fused.num_splits, 1) << "a context inside the fused limit is never split";
  ExpectNoExceptionDumps(kCaseE.tag);
}

TEST_F(TurboQuantFusedDecode, MasksATailTileWithTwoHeadsPerSubcore) {
  PrintShape(kCaseD, aiv_num_, queried_);
  watchdog_.Arm("(d) scenario setup and query rotation");
  tqh::FusedCubeScenario scenario(kCaseD.shape, stream_, aiv_num_);
  watchdog_.Disarm();

  const tqh::FusedRun fused = RunCase(&scenario, kCaseD, &watchdog_, scenario.Reference());
  EXPECT_NE(kTailContext % kBlockSize, 0) << "(d) must end on a partial tile";
  EXPECT_EQ(fused.num_splits, 1);
  EXPECT_EQ(fused.heads_per_task, static_cast<uint32_t>(kWideTaskHeads))
      << "(d) must put two heads on each vector subcore";
  ExpectNoExceptionDumps(kCaseD.tag);
}

TEST_F(TurboQuantFusedDecode, DecodesHeadSize128OnAQwenGroup) {
  PrintShape(kCaseF, aiv_num_, queried_);
  watchdog_.Arm("(f) scenario setup and query rotation");
  tqh::FusedCubeScenario scenario(kCaseF.shape, stream_, aiv_num_);
  watchdog_.Disarm();
  const std::vector<float> reference = scenario.Reference();

  const tqh::FusedRun fused = RunCase(&scenario, kCaseF, &watchdog_, reference);
  EXPECT_EQ(fused.num_splits, 1) << "a context inside the fused limit is never split";
  EXPECT_GE(tqh::FusedCosine(fused.output, reference), kHeadSize128CosineFloor)
      << "(f) the D = 128 Cube decode does not reproduce exact attention";
  ExpectNoExceptionDumps(kCaseF.tag);
}

}
}
}
