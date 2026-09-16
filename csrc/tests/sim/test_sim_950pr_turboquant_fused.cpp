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

// Smoke of the fused single-launch Cube decode (TURBOQUANT_TESTS.md 13.25) against the legacy
// split + combine chain it replaces, on one shape: kv4fp8, H_Q 4, H_KV 2, D 256, one 64-row
// tile. The legacy chain keeps every vector barrier and the subcore-0 gate, so agreement
// between the two is the A/B for the barrier-free, dual-destination kernel.

#include <gtest/gtest.h>

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
constexpr double kMinReferenceCosine = 0.97;
constexpr double kMaxCosineShortfall = 1e-4;

void PrintRun(const char* label, const tqh::FusedRun& run, const std::vector<float>& reference) {
  std::printf("  %-16s launches=%lld splits=%lld block_dim=%u heads/task=%u untouched=%zu cos_vs_fp32=%.6f %.1f s\n",
              label, static_cast<long long>(run.launches), static_cast<long long>(run.num_splits), run.block_dim,
              run.heads_per_task, run.untouched, tqh::FusedCosine(run.output, reference), run.host_s);
}

TEST(TurboQuantFusedDecode, MatchesTheLegacySplitAndCombine) {
  REQUIRE_ASCEND_950PR();
  aclrtStream stream = AscendTestEnvironment::Instance().stream();
  bool queried = false;
  const int64_t aiv_num = tqh::VectorCoreNum(&queried);
  const tqh::FusedShape shape;
  LaunchWatchdog watchdog(kLaunchBudgetSeconds, "[ fused ]");

  std::printf("[ fused ] kv4fp8 H_Q=%lld H_KV=%lld D=%lld block=%lld S=%lld aiv=%lld%s\n",
              static_cast<long long>(shape.num_heads), static_cast<long long>(shape.num_kv_heads),
              static_cast<long long>(shape.head_size), static_cast<long long>(shape.block_size),
              static_cast<long long>(shape.context_len), static_cast<long long>(aiv_num),
              queried ? "" : " (fallback core count)");

  watchdog.Arm("scenario setup and query rotation");
  tqh::FusedCubeScenario scenario(shape, stream, aiv_num);
  watchdog.Disarm();
  const std::vector<float> reference = scenario.Reference();

  watchdog.Arm("turboquant_mm_fused_decode_impl");
  const tqh::FusedRun fused = scenario.RunFused();
  watchdog.Disarm();
  PrintRun("fused", fused, reference);

  watchdog.Arm("legacy turboquant_mm_decode_split_impl + combine");
  const tqh::FusedRun legacy = scenario.RunSplitAndCombine(1);
  watchdog.Disarm();
  PrintRun("legacy split", legacy, reference);

  const tqh::FusedAgreement agreement = tqh::CompareValues(fused.output, legacy.output);
  const double fused_cos = tqh::FusedCosine(fused.output, reference);
  const double legacy_cos = tqh::FusedCosine(legacy.output, reference);
  std::printf("[ fused ] fused vs legacy: %zu of %zu elements differ, max |err| %.3e\n", agreement.differing,
              agreement.compared, agreement.max_abs);

  const DumpCensus dumps = ExceptionDumps();
  std::printf("[ fused ] excp dumps in cwd: %zu files, %zu non-empty, %llu B\n", dumps.files, dumps.non_empty,
              static_cast<unsigned long long>(dumps.bytes));

  EXPECT_EQ(fused.num_splits, 1) << "a context inside the fused limit is never split";
  EXPECT_EQ(fused.untouched, 0u) << "the fused launch left sentinel values in the output";
  EXPECT_EQ(legacy.untouched, 0u) << "the legacy chain left sentinel values in the output";
  EXPECT_EQ(agreement.differing, 0u) << "the fused decode is not bit-identical to the legacy split + combine";
  EXPECT_GE(fused_cos, kMinReferenceCosine) << "fused decode vs exact fp32 attention";
  EXPECT_GE(fused_cos, legacy_cos - kMaxCosineShortfall) << "the fused decode is less faithful than the legacy chain";
  EXPECT_EQ(dumps.non_empty, 0u) << "the camodel wrote an exception dump";
}

}
}
}
