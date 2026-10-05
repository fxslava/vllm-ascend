/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

// TIER 2 (simulator) - the DSV4 routed expert kernel on the CANN camodel.
//
// This tier exists for one reason the host tier cannot serve: the camodel
// models the pipes. The kernel compiles with --cce-auto-sync=off and stages
// everything in TBuf rather than TQue, so every cross-pipe dependency in it is
// hand-written, and a missing flag is invisible to any serial execution. A CPU
// interpreter run of this kernel with every SetFlag/WaitFlag deleted passes
// byte-identically; that is measured, not supposed.
//
// Geometry note: the camodel is a cycle-level model and this kernel is a
// scalar reduction, so cost scales with the MAC count and nothing hides it.
// hidden=64 inter=64 is 12,288 MACs and completes; hidden=256 inter=128 is
// 98,304 and has been measured at over 19 minutes without finishing. Keep the
// default small and treat anything larger as opt-in.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "camodel_guard.hpp"
#include "dsv4_device_case.hpp"
#include "dsv4_test_checks.hpp"
#include "dsv4_moe_expert_launch.hpp"
#include "test_harness.hpp"

namespace vllm_ascend {
namespace test {
namespace {

using dsv4::Bf16UlpDistance;
using dsv4::MakeProblem;
using dsv4::MakeSaturatingProblem;
using dsv4::Problem;
using dsv4::ReferenceExpert;
using dsv4::TilingBuffer;

// A single camodel launch of this kernel is minutes. The watchdog turns a hang
// into a named failure instead of a ctest timeout with no attribution.
constexpr int64_t kLaunchBudgetSeconds = 1200;

constexpr int64_t kHidden = 64;
constexpr int64_t kInter = 64;

struct Outputs {
  std::vector<uint16_t> gate, up, activated, down;

  bool operator==(const Outputs& o) const {
    return gate == o.gate && up == o.up && activated == o.activated && down == o.down;
  }
};

// A fresh expert owner per launch keeps every output poisoned and drains before
// release, exactly as in the physical suite. The watchdog remains sim-specific.
Outputs Launch(const Problem& p, aclrtStream stream, const char* tag) {
  LaunchWatchdog watchdog(kLaunchBudgetSeconds, tag);
  dsv4::DeviceExpert expert(p, stream);
  expert.Enqueue();
  auto values = expert.Read();
  return Outputs{std::move(values[0]), std::move(values[1]), std::move(values[2]), std::move(values[3])};
}

void ExpectMatchesReference(const Problem& p, const Outputs& got) {
  ExpectParity(dsv4::DeviceOutputs{got.gate, got.up, got.activated, got.down}, dsv4::Golden(p));
}

// ---------------------------------------------------------------------------
// Numerical verification
// ---------------------------------------------------------------------------

TEST(Dsv4MoeExpertSim, MatchesCpuReference) {
  aclrtStream stream = AscendTestEnvironment::Instance().stream();
  const Problem p = MakeProblem(kHidden, kInter, /*seed=*/0);
  const Outputs got = Launch(p, stream, "dsv4_moe_expert/reference");
  ExpectMatchesReference(p, got);
}

TEST(Dsv4MoeExpertSim, WritesEveryOutputBuffer) {
  // The outputs are poisoned to 0xA5 before the launch. 0xA5A5 as bf16 is a
  // normal negative number, so a buffer the kernel never touched would sail
  // through a NaN check but cannot survive this one.
  aclrtStream stream = AscendTestEnvironment::Instance().stream();
  const Problem p = MakeProblem(kHidden, kInter, /*seed=*/1);
  const Outputs got = Launch(p, stream, "dsv4_moe_expert/writes");

  auto untouched = [](const std::vector<uint16_t>& v) {
    return std::all_of(v.begin(), v.end(), [](uint16_t b) { return b == 0xA5A5u; });
  };
  EXPECT_FALSE(untouched(got.gate)) << "gate_out was never written";
  EXPECT_FALSE(untouched(got.up)) << "up_out was never written";
  EXPECT_FALSE(untouched(got.activated)) << "activated was never written";
  EXPECT_FALSE(untouched(got.down)) << "down_out was never written";
}

// ---------------------------------------------------------------------------
// Pipeline synchronisation
// ---------------------------------------------------------------------------

TEST(Dsv4MoeExpertSim, RepeatedLaunchesAreBitIdentical) {
  // THE sync test. A missing cross-pipe flag does not usually corrupt a result
  // outright; it makes the result depend on how the pipes happened to
  // interleave. Correctness against a reference cannot see that -- only
  // repetition can.
  //
  // Concretely, this kernel reuses one staging buffer for all three projection
  // legs. Without the S_MTE2 write-after-read flags, the up leg's DataCopy can
  // overwrite the buffer while the gate leg's scalar reads are still retiring,
  // and the damage depends on timing.
  aclrtStream stream = AscendTestEnvironment::Instance().stream();
  const Problem p = MakeProblem(kHidden, kInter, /*seed=*/2);

  const int repeats = std::getenv("ASCEND_DSV4_SYNC_REPEATS") ? std::atoi(std::getenv("ASCEND_DSV4_SYNC_REPEATS")) : 3;
  ASSERT_GE(repeats, 2) << "a single launch proves nothing about synchronisation";

  const Outputs first = Launch(p, stream, "dsv4_moe_expert/sync[0]");
  for (int i = 1; i < repeats; ++i) {
    const Outputs again = Launch(p, stream, ("dsv4_moe_expert/sync[" + std::to_string(i) + "]").c_str());
    EXPECT_TRUE(first == again) << "launch " << i << " differs from launch 0 on identical input: a "
                                << "cross-pipe flag is missing (the kernel compiles with "
                                << "--cce-auto-sync=off, so nothing inserts them implicitly)";
  }
  // The runs agreeing is necessary but not sufficient, so still check the value.
  ExpectMatchesReference(p, first);
}

TEST(Dsv4MoeExpertSim, BackToBackLaunchesOnOneStreamDoNotLeakState) {
  // Two different problems submitted back to back. If the kernel leaves a
  // hardware event set -- a SetFlag without its WaitFlag on some path -- the
  // second launch inherits it and either hangs or reads stale UB.
  aclrtStream stream = AscendTestEnvironment::Instance().stream();
  const Problem a = MakeProblem(kHidden, kInter, /*seed=*/3);
  const Problem b = MakeProblem(kHidden, kInter, /*seed=*/4);

  const Outputs got_a = Launch(a, stream, "dsv4_moe_expert/leak[a]");
  const Outputs got_b = Launch(b, stream, "dsv4_moe_expert/leak[b]");

  ExpectMatchesReference(a, got_a);
  ExpectMatchesReference(b, got_b);
  EXPECT_FALSE(got_a == got_b) << "two different problems produced identical output";
}

// ---------------------------------------------------------------------------
// SwiGLU boundary
// ---------------------------------------------------------------------------

TEST(Dsv4MoeExpertSim, SaturatedGateLandsOnTheClampNotOnZero) {
  // Every gate element is driven far past the DeepSeek-V4 limit. Two things
  // have to hold, and together they are the clamp's whole observable effect:
  //
  //   * the kernel matches the reference, which clamps to -10 and evaluates
  //     silu(-10) * up -- a small non-zero, NOT zero;
  //   * nothing in the path reaches inf. An unclamped kernel would return
  //     exactly zero here, via 1/(1+inf), and the camodel would log
  //     `check_fp_status instr input data inf` for every element.
  aclrtStream stream = AscendTestEnvironment::Instance().stream();
  const Problem p = MakeSaturatingProblem(kHidden, kInter, /*seed=*/5);
  const Outputs got = Launch(p, stream, "dsv4_moe_expert/saturated");

  ExpectMatchesReference(p, got);

  const auto want = ReferenceExpert(p.View(), p.hidden, p.inter);
  ASSERT_FALSE(want.gate_f32.empty());
  EXPECT_TRUE(std::all_of(want.gate_f32.begin(), want.gate_f32.end(), [](float v) { return v < -dsv4::kSwigluLimit; }))
      << "the saturating problem no longer saturates; re-tune MakeSaturatingProblem";
  EXPECT_TRUE(std::all_of(want.activated_f32.begin(), want.activated_f32.end(), [](float v) {
    return std::isfinite(v) && v != 0.0f;
  })) << "a clamped gate must give a finite non-zero activation";
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
