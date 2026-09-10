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

// Partial rotary position embedding on Ascend 950PR, fp16 in / fp16 out.
//
// Stages 3 and 4 of the Qwen3.5 decoder layer. partial_rotary_factor 0.25
// against head_dim 256, so channels [0, 64) of every head rotate and [64, 256)
// must come out bit-identical. Two consequences:
//
//   * The 310P cannot run this shape: AscendMRotaryEmbedding310 gates on
//     rotary_dim in (64, 128).
//   * The stock aclnnApplyRotaryPosEmbV2 cannot express a partial rotation;
//     common/partial_rotary_950pr.hpp documents the two ways round that and
//     picks between them at run time.
//
// Every device test below prints which path ran.
//
// Seeds, shapes and tolerances follow
// csrc/tests/kernels/cuda/test_rotary_embedding.cpp; the 256/64 cases are new
// here.

#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "aclnn_ops.hpp"
#include "aclnn_runtime.hpp"
#include "ascend950_shapes.hpp"
#include "cpu_reference.hpp"
#include "device_tensor.hpp"
#include "fp16.hpp"
#include "partial_rotary_950pr.hpp"
#include "random_data.hpp"
#include "tensor_compare.hpp"
#include "test_harness.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace s = shapes950;
using reference::RotaryMode;

// The dump, the plugin and this suite all use neox / rotate_half. The
// interleave layout is a different model family's convention and there is no
// Qwen3.5 shape that reaches it, so it is not swept here - unlike on the 310P,
// where both were covered because the plugin picks between them per model.
constexpr RotaryMode kMode = RotaryMode::kHalf;

struct RotaryCase {
  int64_t num_tokens;
  int64_t head_dim;
  int64_t rotary_dim;
  int64_t num_q_heads;
  int64_t num_kv_heads;
  double theta;
};

struct RotaryResult {
  std::vector<float> query;
  std::vector<float> key;
  PartialRotaryPath path;
};

RotaryResult RunPartialRotaryOnDevice(const std::vector<float>& query, const std::vector<float>& key,
                                      const std::vector<float>& cos_full, const std::vector<float>& sin_full,
                                      const RotaryCase& test_case) {
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  // Flat [tokens * heads * head_dim] allocations; the helper builds whatever
  // BSND views the operator it picks needs over the same memory.
  DeviceTensor query_device =
      DeviceTensor::Half({test_case.num_tokens, test_case.num_q_heads, test_case.head_dim}, query);
  DeviceTensor key_device =
      DeviceTensor::Half({test_case.num_tokens, test_case.num_kv_heads, test_case.head_dim}, key);

  RotaryResult result;
  result.path = ApplyPartialRotaryQK(query_device.data(), key_device.data(), cos_full, sin_full,
                                     test_case.num_tokens, test_case.num_q_heads, test_case.num_kv_heads,
                                     test_case.head_dim, test_case.rotary_dim, stream);

  result.query = query_device.ToFloatFromHalf();
  result.key = key_device.ToFloatFromHalf();
  return result;
}

// Scattered rather than sequential, so a kernel that ignores the per-token
// cos/sin row cannot pass by accident.
std::vector<int32_t> ScatteredPositions(int64_t num_tokens) {
  std::vector<int32_t> positions(static_cast<size_t>(num_tokens));
  for (int64_t i = 0; i < num_tokens; ++i) {
    positions[static_cast<size_t>(i)] = static_cast<int32_t>((i * 37 + 11) % s::kMaxPositionEmbeddings);
  }
  return positions;
}

// cos/sin as the device will see them: built at the rotary width, gathered for
// these positions and rounded to fp16, so the reference uses the same angles
// and the comparison measures the kernel rather than the cache precision.
void BuildDeviceAngles(const RotaryCase& test_case, const std::vector<int32_t>& positions,
                       std::vector<float>* cos_full, std::vector<float>* sin_full) {
  const std::vector<float> cache =
      reference::BuildCosSinCache(s::kMaxPositionEmbeddings, test_case.rotary_dim, test_case.theta);
  reference::GatherFullCosSin(cache, positions, test_case.rotary_dim, kMode, cos_full, sin_full);
  *cos_full = QuantizeToHalf(*cos_full);
  *sin_full = QuantizeToHalf(*sin_full);
}

// L2 norm of each rotary pair. A rotation preserves it exactly, which is a
// property check that needs no reference implementation. Only pairs inside the
// rotary slice are collected: the pass-through channels are not paired at all.
std::vector<float> RotaryPairNorms(const std::vector<float>& x, int64_t num_tokens, int64_t num_heads,
                                   int64_t head_dim, int64_t rotary_dim) {
  const int64_t half = rotary_dim / 2;
  std::vector<float> norms;
  norms.reserve(static_cast<size_t>(num_tokens * num_heads * half));

  for (int64_t token = 0; token < num_tokens; ++token) {
    for (int64_t head = 0; head < num_heads; ++head) {
      const size_t base = static_cast<size_t>((token * num_heads + head) * head_dim);
      for (int64_t k = 0; k < half; ++k) {
        const float a = x[base + static_cast<size_t>(k)];
        const float b = x[base + static_cast<size_t>(k + half)];
        norms.push_back(std::sqrt(a * a + b * b));
      }
    }
  }
  return norms;
}

// -----------------------------------------------------------------------------
// Host-only checks
// -----------------------------------------------------------------------------

TEST(Rotary950PrReference, PartialRotationLeavesTheTailAlone) {
  // The property the whole file exists for, checked against the reference first
  // so a device failure cannot be blamed on the reference.
  const int64_t head_dim = s::kHeadDim;
  const int64_t rotary_dim = s::kRotaryDim;

  DeterministicRandom random(0x50525450u);  // "PRTP"
  const std::vector<float> x = random.NormalHalfExact(static_cast<size_t>(head_dim), 0.0f, 1.0f);

  const std::vector<float> cache = reference::BuildCosSinCache(64, rotary_dim, s::kRopeThetaExtended);
  std::vector<float> cos_full;
  std::vector<float> sin_full;
  // Position 17 is arbitrary but non-zero: at position 0 the rotation is the
  // identity and the tail would be preserved even by a kernel that rotated
  // everything.
  reference::GatherFullCosSin(cache, {17}, rotary_dim, kMode, &cos_full, &sin_full);

  std::vector<float> out;
  reference::ApplyRotaryPosEmb(x, cos_full, sin_full, 1, 1, head_dim, rotary_dim, kMode, &out);

  ASSERT_EQ(out.size(), x.size());
  for (int64_t i = rotary_dim; i < head_dim; ++i) {
    EXPECT_FLOAT_EQ(out[static_cast<size_t>(i)], x[static_cast<size_t>(i)]) << "channel " << i << " was modified";
  }

  bool rotated_something = false;
  for (int64_t i = 0; i < rotary_dim; ++i) {
    if (std::fabs(out[static_cast<size_t>(i)] - x[static_cast<size_t>(i)]) > 1e-4f) {
      rotated_something = true;
      break;
    }
  }
  EXPECT_TRUE(rotated_something) << "the rotary slice was not rotated at all";
}

TEST(Rotary950PrShapes, QwenHeadDimIsWithinTheOperatorLimit) {
  // The 950 rotary operator accepts head dims up to 1024 in half mode, and the
  // dim must be even. head_dim 256 with rotary_dim 64 satisfies both; the same
  // shape is out of range on the 310P, which is the reason this file is not a
  // copy of the 310P one.
  EXPECT_LE(s::kHeadDim, s::kMaxRotaryHeadDim);
  EXPECT_EQ(s::kHeadDim % s::kRotaryHalfModeDimMultiple, 0);
  EXPECT_EQ(s::kRotaryDim % s::kRotaryHalfModeDimMultiple, 0);
  EXPECT_LT(s::kRotaryDim, s::kHeadDim) << "this suite is about the partial case";
  // partial_rotary_factor 0.25.
  EXPECT_EQ(s::kRotaryDim * 4, s::kHeadDim);
}

TEST(Rotary950PrPathSelection, FullRotationDoesNotNeedTheCustomOperator) {
  // A shape with rotary_dim == head_dim needs no slicing, so it must never be
  // routed through the custom operator or the packed fallback even when the
  // custom operator is present. This runs with no device: it only inspects
  // which operators resolved.
  PartialRotaryPath path = PartialRotaryPath::kPackedApplyRotary;
  std::string reason;
  if (!SelectPartialRotaryPath(/*head_dim=*/128, /*rotary_dim=*/128, &path, &reason)) {
    GTEST_SKIP() << reason;
  }
  EXPECT_EQ(path, PartialRotaryPath::kFullApplyRotary);
}

// -----------------------------------------------------------------------------
// Device parity
// -----------------------------------------------------------------------------

class Rotary950PrTest : public ::testing::TestWithParam<RotaryCase> {
 protected:
  // Skips when no rotary operator resolved, and otherwise records the path so
  // every test in the suite reports it.
  void RequireRotaryOperator() {
    PartialRotaryPath path = PartialRotaryPath::kPackedApplyRotary;
    std::string reason;
    if (!SelectPartialRotaryPath(GetParam().head_dim, GetParam().rotary_dim, &path, &reason)) {
      GTEST_SKIP() << reason;
    }
    RecordProperty("rotary_path", PartialRotaryPathName(path));
  }
};

TEST_P(Rotary950PrTest, MatchesCpuReference) {
  REQUIRE_ASCEND_950PR();
  RequireRotaryOperator();

  const RotaryCase& test_case = GetParam();
  DeterministicRandom random(0x51524f50u);  // "QROP"

  const std::vector<float> query = random.NormalHalfExact(
      static_cast<size_t>(test_case.num_tokens * test_case.num_q_heads * test_case.head_dim), 0.0f, 1.0f);
  const std::vector<float> key = random.NormalHalfExact(
      static_cast<size_t>(test_case.num_tokens * test_case.num_kv_heads * test_case.head_dim), 0.0f, 1.0f);

  const std::vector<int32_t> positions = ScatteredPositions(test_case.num_tokens);
  std::vector<float> cos_full;
  std::vector<float> sin_full;
  BuildDeviceAngles(test_case, positions, &cos_full, &sin_full);

  const RotaryResult actual = RunPartialRotaryOnDevice(query, key, cos_full, sin_full, test_case);
  SCOPED_TRACE(PartialRotaryPathName(actual.path));

  std::vector<float> expected_query;
  reference::ApplyRotaryPosEmb(query, cos_full, sin_full, test_case.num_tokens, test_case.num_q_heads,
                               test_case.head_dim, test_case.rotary_dim, kMode, &expected_query);
  std::vector<float> expected_key;
  reference::ApplyRotaryPosEmb(key, cos_full, sin_full, test_case.num_tokens, test_case.num_kv_heads,
                               test_case.head_dim, test_case.rotary_dim, kMode, &expected_key);

  EXPECT_TENSORS_ALLCLOSE(actual.query, QuantizeToHalf(expected_query), kFp16DefaultTolerance);
  EXPECT_TENSORS_ALLCLOSE(actual.key, QuantizeToHalf(expected_key), kFp16DefaultTolerance);
}

TEST_P(Rotary950PrTest, LeavesChannelsPastRotaryDimBitIdentical) {
  REQUIRE_ASCEND_950PR();
  RequireRotaryOperator();

  const RotaryCase& test_case = GetParam();
  if (test_case.rotary_dim == test_case.head_dim) {
    GTEST_SKIP() << "full rotation: there are no pass-through channels to check";
  }

  // The assertion the partial path exists for, and the one an "improvement"
  // that rotated the whole head would fail. It is bit-exact deliberately: the
  // pass-through channels are copied, not computed, so any difference at all is
  // a bug rather than a rounding question.
  DeterministicRandom random(0x50415353u);  // "PASS"

  const std::vector<float> query = random.NormalHalfExact(
      static_cast<size_t>(test_case.num_tokens * test_case.num_q_heads * test_case.head_dim), 0.0f, 1.0f);
  const std::vector<float> key = random.NormalHalfExact(
      static_cast<size_t>(test_case.num_tokens * test_case.num_kv_heads * test_case.head_dim), 0.0f, 1.0f);

  const std::vector<int32_t> positions = ScatteredPositions(test_case.num_tokens);
  std::vector<float> cos_full;
  std::vector<float> sin_full;
  BuildDeviceAngles(test_case, positions, &cos_full, &sin_full);

  const RotaryResult actual = RunPartialRotaryOnDevice(query, key, cos_full, sin_full, test_case);
  SCOPED_TRACE(PartialRotaryPathName(actual.path));

  struct Tensor {
    const char* label;
    const std::vector<float>* before;
    const std::vector<float>* after;
    int64_t heads;
  };
  const Tensor tensors[] = {
      {"query", &query, &actual.query, test_case.num_q_heads},
      {"key", &key, &actual.key, test_case.num_kv_heads},
  };

  for (const Tensor& tensor : tensors) {
    ASSERT_EQ(tensor.after->size(), tensor.before->size()) << tensor.label;
    for (int64_t token = 0; token < test_case.num_tokens; ++token) {
      for (int64_t head = 0; head < tensor.heads; ++head) {
        const size_t base = static_cast<size_t>((token * tensor.heads + head) * test_case.head_dim);
        for (int64_t dim = test_case.rotary_dim; dim < test_case.head_dim; ++dim) {
          const size_t index = base + static_cast<size_t>(dim);
          ASSERT_EQ((*tensor.after)[index], (*tensor.before)[index])
              << tensor.label << " token " << token << " head " << head << " channel " << dim
              << " was modified past rotary_dim";
        }
      }
    }
  }
}

TEST_P(Rotary950PrTest, PositionZeroLeavesTheInputUnchanged) {
  REQUIRE_ASCEND_950PR();
  RequireRotaryOperator();

  const RotaryCase& test_case = GetParam();
  DeterministicRandom random(0x5a45524fu);  // "ZERO"

  const std::vector<float> query = random.NormalHalfExact(
      static_cast<size_t>(test_case.num_tokens * test_case.num_q_heads * test_case.head_dim), 0.0f, 1.0f);
  const std::vector<float> key = random.NormalHalfExact(
      static_cast<size_t>(test_case.num_tokens * test_case.num_kv_heads * test_case.head_dim), 0.0f, 1.0f);

  // cos(0) = 1, sin(0) = 0, so every channel - rotated or passed through - must
  // come back exactly as it went in.
  const std::vector<int32_t> positions(static_cast<size_t>(test_case.num_tokens), 0);
  std::vector<float> cos_full;
  std::vector<float> sin_full;
  BuildDeviceAngles(test_case, positions, &cos_full, &sin_full);

  const RotaryResult actual = RunPartialRotaryOnDevice(query, key, cos_full, sin_full, test_case);
  SCOPED_TRACE(PartialRotaryPathName(actual.path));

  EXPECT_TENSORS_ALLCLOSE(actual.query, query, kFp16DefaultTolerance);
  EXPECT_TENSORS_ALLCLOSE(actual.key, key, kFp16DefaultTolerance);
}

TEST_P(Rotary950PrTest, PreservesRotaryPairNorms) {
  REQUIRE_ASCEND_950PR();
  RequireRotaryOperator();

  // A rotation is orthogonal within each (k, k + rotary_dim/2) pair, so the
  // pair norm is invariant. This needs no reference at all and would catch a
  // kernel that applied cos and sin from the wrong row.
  const RotaryCase& test_case = GetParam();
  DeterministicRandom random(0x504e524du);  // "PNRM"

  const std::vector<float> query = random.NormalHalfExact(
      static_cast<size_t>(test_case.num_tokens * test_case.num_q_heads * test_case.head_dim), 0.0f, 1.0f);
  const std::vector<float> key = random.NormalHalfExact(
      static_cast<size_t>(test_case.num_tokens * test_case.num_kv_heads * test_case.head_dim), 0.0f, 1.0f);

  const std::vector<int32_t> positions = ScatteredPositions(test_case.num_tokens);
  std::vector<float> cos_full;
  std::vector<float> sin_full;
  BuildDeviceAngles(test_case, positions, &cos_full, &sin_full);

  const RotaryResult actual = RunPartialRotaryOnDevice(query, key, cos_full, sin_full, test_case);
  SCOPED_TRACE(PartialRotaryPathName(actual.path));

  // Two fp16 roundings and a square root, so this is looser than the parity
  // tolerance by design.
  const Tolerance norm_tolerance{5e-3, 5e-3, "pair norm through two fp16 roundings plus a square root"};

  EXPECT_TENSORS_ALLCLOSE(RotaryPairNorms(actual.query, test_case.num_tokens, test_case.num_q_heads,
                                          test_case.head_dim, test_case.rotary_dim),
                          RotaryPairNorms(query, test_case.num_tokens, test_case.num_q_heads,
                                          test_case.head_dim, test_case.rotary_dim),
                          norm_tolerance);
  EXPECT_TENSORS_ALLCLOSE(RotaryPairNorms(actual.key, test_case.num_tokens, test_case.num_kv_heads,
                                          test_case.head_dim, test_case.rotary_dim),
                          RotaryPairNorms(key, test_case.num_tokens, test_case.num_kv_heads,
                                          test_case.head_dim, test_case.rotary_dim),
                          norm_tolerance);
}

std::string RotaryTestName(const ::testing::TestParamInfo<RotaryCase>& info) {
  std::ostringstream name;
  name << "tokens" << info.param.num_tokens << "_d" << info.param.head_dim << "_rot" << info.param.rotary_dim
       << "_h" << info.param.num_q_heads << "_kv" << info.param.num_kv_heads << "_theta"
       << static_cast<int64_t>(info.param.theta);
  return name.str();
}

INSTANTIATE_TEST_SUITE_P(
    Qwen35, Rotary950PrTest,
    ::testing::Values(
        // The layer-3 decode shape the golden test runs, and a prefill-sized
        // batch of it. rope_theta 1e6 is what scripts/dump_qwen35_layer3.py
        // defaults to.
        RotaryCase{1, s::kHeadDim, s::kRotaryDim, s::kNumHeads, s::kNumKvHeads, s::kRopeThetaExtended},
        RotaryCase{32, s::kHeadDim, s::kRotaryDim, s::kNumHeads, s::kNumKvHeads, s::kRopeThetaExtended},
        RotaryCase{128, s::kHeadDim, s::kRotaryDim, s::kNumHeads, s::kNumKvHeads, s::kRopeThetaDefault},
        // A second partial ratio, so a kernel that hard-coded 0.25 fails.
        RotaryCase{16, s::kHeadDim, 128, s::kNumHeads, s::kNumKvHeads, s::kRopeThetaDefault},
        // Full-rotation cases, identical to the ones the 310P suite sweeps, so
        // the two backends can be compared directly on the same shapes.
        RotaryCase{1, 128, 128, 28, 4, s::kRopeThetaDefault},
        RotaryCase{16, 128, 128, 32, 8, s::kRopeThetaExtended},
        RotaryCase{32, 64, 64, 16, 2, s::kRopeThetaDefault}),
    RotaryTestName);

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
