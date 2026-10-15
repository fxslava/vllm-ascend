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

// The sparse boundary grid of the modulo-scheduled decode
// (op_kernel/turboquant_fused_decode_opt.cpp), kv4fp8, on the camodel.
//
// WHAT IS BEING ASKED
//
// The opt kernel is a pure RESCHEDULE of turboquant_fused_decode.cpp's kv4fp8 instance: the same
// query quantisation, the same online softmax in the same order, the same two GEMMs, the same
// accumulate. Only WHEN each step runs changed -- a three-slot ingest ring instead of two, two
// Fixpipe score windows instead of one, and a loop unrolled by six so the ring indices are
// compile-time. Nothing in that list is allowed to move a bit of the answer.
//
// So the oracle here is not a recorded golden and not a cosine floor. It is the lock-step decode
// itself, run over the SAME scenario in the SAME process: every case below launches both kernels
// and requires their fp16 outputs to be bit-identical. A cosine against exact fp32 attention is
// also checked, but only as a backstop -- it would catch two kernels that are wrong together,
// which bit-identity cannot.
//
// WHY THESE CONTEXT LENGTHS
//
// Tiles per task is ceil(context_len / 64), and the schedule's state is (ring slot, score window),
// whose period is lcm(3, 2) = 6. The grid walks the tile count through every structurally distinct
// value and then one past the period:
//
//   1 tile    no steady phase at all -- the prologue stages slot 0, the Cube's prologue runs the
//             only score GEMM, and the single phase takes every "is there a next tile" branch false
//   2 tiles   the prologue's SECOND L1 stage exists (tile 1) but nothing is ever staged from inside
//             the loop; the ring is primed two deep and never wraps
//   3 tiles   the ring is primed to its full depth and the first in-loop stage happens; still no
//             slot is ever reused
//   4 tiles   the first slot REUSE (tile 3 lands in slot 0) and therefore the first kOptFlagSlotFree
//             wait and the first posted SlotFree token
//   5 tiles   two reuses; the MTE2 run-ahead has retired a full ring
//   6 tiles   exactly one unroll period: the loop runs its six-phase body once and the tail is empty
//   7 tiles   one phase past the period -- the loop body runs once and the TAIL runs phase 0, which
//             is the only arrangement where a phase's compile-time indices have to line up with a
//             tile index that did not come from the same unrolled copy
//
// and the ragged remainders (63, 65, 127, 129, 383, 385) put a partial last tile on each side of a
// tile boundary, which is what ComputeTailMask and the kCubeTileRows masking have to get right.
//
// WHAT THIS DOES NOT PROVE. Per the sparse-verification protocol: error growth over tile counts
// larger than the grid, resource high-water marks at full context, and anything about silicon. A
// pass here means "executes on the cycle-accurate model over this grid and is bit-identical to the
// kernel it reschedules", and nothing more.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "camodel_guard.hpp"
#include "test_harness.hpp"
#include "turboquant/turboquant_fused_harness.hpp"
#include "turboquant/turboquant_launch.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace tqh = turboquant_host;

constexpr int64_t kLaunchBudgetSeconds = 1800;
constexpr int64_t kBlockSize = 64;
constexpr int64_t kHeadSize = 256;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kGroupHeads = 4;
// The floor the request names. The kv4fp8 D = 256 shapes land near 0.9994 against exact fp32
// attention, so this is loose on purpose: it is a backstop against both kernels being wrong
// together, not the test's real oracle, which is bit-identity with the lock-step decode.
constexpr double kCosineFloor = 0.99;
constexpr tqh::FusedSplitPolicy kContextOnly = tqh::FusedSplitPolicy::kContextOnly;

struct BoundaryCase {
  int64_t context_len;
  int64_t tiles;
  const char* role;
};

// ceil(context_len / 64) is the tile count; `tiles` is recorded here so a mis-stated case fails
// its own ASSERT rather than quietly testing a different rung than it claims.
const BoundaryCase kBoundaryGrid[] = {
    {63, 1, "ragged single tile, 1 row short"},
    {64, 1, "exact single tile: no steady phase"},
    {65, 2, "ragged: a 1-row second tile"},
    {127, 2, "ragged: a 63-row second tile"},
    {128, 2, "exact two tiles: prologue primes both L1 slots, loop never stages"},
    {129, 3, "ragged: the ring reaches full depth"},
    {192, 3, "exact three tiles: first in-loop L1 stage"},
    {256, 4, "exact four tiles: first slot reuse and first SlotFree token"},
    {320, 5, "exact five tiles"},
    {383, 6, "ragged: one unroll period, last tile 63 rows"},
    {384, 6, "exact six tiles: one whole unroll period, empty tail"},
    {385, 7, "ragged: one period plus a one-row tail phase"},
    {448, 7, "exact seven tiles: one period plus a full tail phase"},
};

tqh::FusedShape BoundaryShape(int64_t context_len) {
  tqh::FusedShape shape;
  shape.num_heads = kGroupHeads;
  shape.num_kv_heads = kKvHeads;
  shape.head_size = kHeadSize;
  shape.block_size = kBlockSize;
  shape.context_len = context_len;
  return shape;
}

size_t CountNonFinite(const std::vector<float>& values) {
  size_t bad = 0;
  for (const float v : values) {
    bad += std::isfinite(v) ? 0u : 1u;
  }
  return bad;
}

size_t CountRawDiffs(const std::vector<Half>& got, const std::vector<Half>& want) {
  if (got.size() != want.size()) {
    return got.size() + want.size();
  }
  size_t diffs = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    diffs += (got[i].bits != want[i].bits) ? 1u : 0u;
  }
  return diffs;
}

void ExpectNoExceptionDumps(const char* tag) {
  const DumpCensus dumps = ExceptionDumps();
  std::printf("[ opt ] %s excp dumps in cwd: %zu files, %zu non-empty, %llu B\n", tag, dumps.files, dumps.non_empty,
              static_cast<unsigned long long>(dumps.bytes));
  EXPECT_EQ(dumps.non_empty, 0u) << tag << ": the camodel wrote an exception dump";
}

class TurboQuantFusedDecodeOptFixture {
 protected:
  void Prepare() {
    stream_ = AscendTestEnvironment::Instance().stream();
    aiv_num_ = tqh::VectorCoreNum(&queried_);
  }

  aclrtStream stream_ = nullptr;
  int64_t aiv_num_ = 0;
  bool queried_ = false;
  LaunchWatchdog watchdog_{kLaunchBudgetSeconds, "[ opt ]"};
};

class TurboQuantFusedDecodeOptBoundary : public TurboQuantFusedDecodeOptFixture,
                                         public ::testing::TestWithParam<BoundaryCase> {
 protected:
  void SetUp() override {
    REQUIRE_ASCEND_950PR();
    Prepare();
  }
};

TEST_P(TurboQuantFusedDecodeOptBoundary, MatchesTheLockStepDecodeBitForBit) {
  const BoundaryCase& bc = GetParam();
  const int64_t expected_tiles = (bc.context_len + kBlockSize - 1) / kBlockSize;
  ASSERT_EQ(expected_tiles, bc.tiles) << "the grid entry mis-states its tile count";

  std::printf("[ opt ] S=%lld tiles=%lld  %s\n", static_cast<long long>(bc.context_len),
              static_cast<long long>(bc.tiles), bc.role);

  watchdog_.Arm("boundary scenario setup");
  tqh::FusedCubeScenario scenario(BoundaryShape(bc.context_len), stream_, aiv_num_);
  watchdog_.Disarm();

  // kContextOnly keeps the whole context in one task, so one launch runs the pipeline end to end
  // and numTiles is exactly bc.tiles. A split grid would cut the tile count per task and test a
  // shorter rung than the row claims.
  const tqh::FusedDecodeGrid grid = scenario.Plan(0, kContextOnly);
  ASSERT_EQ(grid.num_splits, 1) << "the boundary grid has to run unsplit, or the tile count is not the rung";

  const std::vector<float> reference = scenario.Reference();

  watchdog_.Arm("lock-step turboquant_mm_fused_decode_impl");
  const tqh::FusedRun lock = scenario.RunFused(grid);
  watchdog_.Disarm();

  watchdog_.Arm("modulo turboquant_mm_fused_decode_opt_impl");
  const tqh::FusedRun opt = scenario.RunFusedOpt(grid);
  watchdog_.Disarm();

  const double lock_cos = tqh::FusedCosine(lock.output, reference);
  const double opt_cos = tqh::FusedCosine(opt.output, reference);
  const size_t diffs = CountRawDiffs(opt.raw, lock.raw);
  std::printf(
      "[ opt ] S=%lld block_dim=%u heads/task=%u untouched lock=%zu opt=%zu fnv1a lock=0x%016llx "
      "opt=0x%016llx bitdiffs=%zu cos lock=%.6f opt=%.6f  %.1f s / %.1f s\n",
      static_cast<long long>(bc.context_len), opt.block_dim, opt.heads_per_task, lock.untouched, opt.untouched,
      static_cast<unsigned long long>(lock.fnv1a), static_cast<unsigned long long>(opt.fnv1a), diffs, lock_cos,
      opt_cos, lock.host_s, opt.host_s);

  ASSERT_EQ(lock.launches, 1) << "the lock-step decode did not launch";
  ASSERT_EQ(opt.launches, 1) << "the modulo decode did not launch";
  EXPECT_EQ(lock.untouched, 0u) << "the lock-step launch left sentinel values in the output";
  EXPECT_EQ(opt.untouched, 0u) << "the modulo launch left sentinel values in the output";
  EXPECT_EQ(CountNonFinite(opt.output), 0u) << "the modulo launch wrote a NaN or an infinity";
  EXPECT_EQ(diffs, 0u) << "the modulo schedule changed the answer: it is supposed to reorder the work, "
                          "not the arithmetic";
  EXPECT_EQ(opt.fnv1a, lock.fnv1a) << "output hashes differ";
  EXPECT_GE(opt_cos, kCosineFloor) << "the modulo decode does not match exact fp32 attention";
  ExpectNoExceptionDumps("boundary");
}

std::string BoundaryName(const ::testing::TestParamInfo<BoundaryCase>& info) {
  char name[32];
  std::snprintf(name, sizeof(name), "S%04lld_T%lld", static_cast<long long>(info.param.context_len),
                static_cast<long long>(info.param.tiles));
  return name;
}

INSTANTIATE_TEST_SUITE_P(SparseGrid, TurboQuantFusedDecodeOptBoundary, ::testing::ValuesIn(kBoundaryGrid),
                         BoundaryName);

// The long rungs, and why they are DISABLED_ rather than absent.
//
// Nothing in them is structurally new: past seven tiles the schedule repeats its six-phase period and
// every index, flag and branch has already been exercised by the grid above. What a long context does
// test is the one thing the sparse protocol openly cannot -- ERROR ACCRUAL. The online softmax rescales
// the fp32 accumulator once per tile, so the distance from exact attention grows with the tile count in
// a way no sample of tile indices can show.
//
// They are off by default because they are priced, not because they are unimportant. Measured on this
// workstation's camodel a launch costs about 28 + 15 * tiles seconds, so one rung's pair of launches is
//
//   context 1024   16 tiles    ~9 minutes
//   context 4096   64 tiles    ~34 minutes
//
// against ~110 seconds for a rung in the grid above. Run them with --gtest_also_run_disabled_tests.
// Contexts past kFusedContextLimit are deliberately absent: they stop being one task and start being the
// split-and-reduce path, which is a different kernel path and belongs in its own case, and at 512 tiles a
// single launch extrapolates to over two hours.
const BoundaryCase kLongRungs[] = {
    {1023, 16, "ragged: sixteen tiles, last one 63 rows"},
    {1024, 16, "exact sixteen tiles"},
    {4096, 64, "exact sixty-four tiles, at kFusedContextLimit and still one task"},
};

INSTANTIATE_TEST_SUITE_P(DISABLED_LongRungs, TurboQuantFusedDecodeOptBoundary, ::testing::ValuesIn(kLongRungs),
                         BoundaryName);

class TurboQuantFusedDecodeOpt : public TurboQuantFusedDecodeOptFixture, public ::testing::Test {
 protected:
  void SetUp() override {
    REQUIRE_ASCEND_950PR();
    Prepare();
  }
};

// The token-balance check, as far as a camodel can make one.
//
// There is no counter to read: cross-core flags are hardware events and the simulator is not to be
// probed for its internals. What an imbalance DOES produce is observable, and it is one of two
// things. A flag waited more often than it is set strands a core and the launch never returns --
// the watchdog below is armed for that, and a hang is a failure, not a timeout to be raised. A flag
// SET more often than it is waited leaves a residual token in the event queue, and the next launch
// in the same process consumes it early: a core walks through a wait that should have blocked,
// reads an L1 slot or a Fixpipe window its producer has not filled yet, and the answer changes.
//
// So: run the kernel three times back to back over one scenario, at the shape that posts the most
// tokens of any rung in the grid, and require all three to be bit-identical. The first launch is
// the control, the second would see anything the first leaked, and the third anything that
// accumulates. Nothing cleans the event state between them -- that is the point.
TEST_F(TurboQuantFusedDecodeOpt, LeavesNoResidualTokenAcrossBackToBackLaunches) {
  constexpr int64_t kContext = 448;  // seven tiles: every flag class posted at least twice
  constexpr int kLaunches = 3;

  watchdog_.Arm("re-entrancy scenario setup");
  tqh::FusedCubeScenario scenario(BoundaryShape(kContext), stream_, aiv_num_);
  watchdog_.Disarm();
  const tqh::FusedDecodeGrid grid = scenario.Plan(0, kContextOnly);
  const std::vector<float> reference = scenario.Reference();

  std::vector<tqh::FusedRun> runs;
  for (int i = 0; i < kLaunches; ++i) {
    watchdog_.Arm("re-entrancy launch " + std::to_string(i));
    runs.push_back(scenario.RunFusedOpt(grid));
    watchdog_.Disarm();
    std::printf("[ opt ] re-entrancy launch %d untouched=%zu fnv1a=0x%016llx cos=%.6f %.1f s\n", i,
                runs.back().untouched, static_cast<unsigned long long>(runs.back().fnv1a),
                tqh::FusedCosine(runs.back().output, reference), runs.back().host_s);
    ASSERT_EQ(runs.back().launches, 1) << "launch " << i << " did not run";
  }

  for (int i = 1; i < kLaunches; ++i) {
    EXPECT_EQ(CountRawDiffs(runs[i].raw, runs[0].raw), 0u)
        << "launch " << i << " differs from the first: a cross-core token outlived its launch";
    EXPECT_EQ(runs[i].untouched, 0u) << "launch " << i << " left sentinel values in the output";
  }
  ExpectNoExceptionDumps("re-entrancy");
}

// Guard bands. The scenario's sparse_boundary mode fills every block-table row the decode must not
// read with -1 and poisons the GM behind the tiles it must not touch; SparseCanaryMismatches()
// reports any byte of that poison that moved, which is the out-of-bounds MTE check. The ragged
// context adds a partial last tile on top of it, so a tail mask that masked the wrong lanes would
// also show up as a changed answer when the inactive tail is re-poisoned between two launches.
TEST_F(TurboQuantFusedDecodeOpt, HoldsItsGuardBandsOnASparseRaggedContext) {
  tqh::FusedShape shape = BoundaryShape(4 * kBlockSize + 1);  // five logical tiles, last one 1 row
  shape.sparse_boundary = true;

  watchdog_.Arm("sparse guard-band setup");
  tqh::FusedCubeScenario scenario(shape, stream_, aiv_num_);
  watchdog_.Disarm();
  ASSERT_EQ(scenario.sparse_tile_count(), 4u) << "the sparse plan did not skip a tile";

  const std::vector<float> reference = scenario.Reference();
  const tqh::FusedDecodeGrid grid = scenario.Plan(0, kContextOnly);

  watchdog_.Arm("sparse guard-band decode");
  const tqh::FusedRun first = scenario.RunFusedOpt(grid);
  watchdog_.Disarm();
  ASSERT_EQ(first.launches, 1);
  EXPECT_EQ(first.untouched, 0u);
  EXPECT_EQ(CountNonFinite(first.output), 0u);
  const double cosine = tqh::FusedCosine(first.output, reference);
  std::printf("[ opt ] sparse guard band cos=%.6f canaries=%zu\n", cosine, scenario.SparseCanaryMismatches());
  EXPECT_GE(cosine, kCosineFloor) << "the sparse decode does not match the tokens it is supposed to read";
  EXPECT_EQ(scenario.SparseCanaryMismatches(), 0u) << "an MTE burst wrote outside the tiles the plan maps";

  // Re-poison what the decode must not read and run again: the answer may not move by one bit.
  scenario.PoisonInactiveTail();
  watchdog_.Arm("sparse guard-band poisoned decode");
  const tqh::FusedRun poisoned = scenario.RunFusedOpt(grid);
  watchdog_.Disarm();
  ASSERT_EQ(poisoned.launches, 1);
  EXPECT_EQ(CountRawDiffs(poisoned.raw, first.raw), 0u) << "poison in the inactive tail reached the output";
  EXPECT_EQ(scenario.SparseCanaryMismatches(), 0u);
  ExpectNoExceptionDumps("sparse guard band");
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
