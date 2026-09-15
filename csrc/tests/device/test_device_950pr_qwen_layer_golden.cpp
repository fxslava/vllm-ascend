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

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "acl_check.hpp"
#include "aclnn_ops.hpp"
#include "aclnn_ops_950pr.hpp"
#include "aclnn_runtime.hpp"
#include "ascend950_shapes.hpp"
#include "cpu_reference.hpp"
#include "device_buffer.hpp"
#include "device_tensor.hpp"
#include "fp16.hpp"
#include "golden_layer3.hpp"
#include "partial_rotary_950pr.hpp"
#include "tensor_compare.hpp"
#include "test_harness.hpp"

namespace vllm_ascend {
namespace test {
namespace {

namespace s = shapes950;

constexpr size_t kHalfBytes = sizeof(uint16_t);

constexpr int64_t kSwiGluSplitDim = -1;

char kFiaLayout[] = "TND";
char kScatterCacheMode[] = "Norm";

const AclnnOp& RmsNormOp() {
  static const AclnnOp op(ops::kRmsNorm);
  return op;
}
const AclnnOp& MatmulOp() {
  static const AclnnOp op(ops::kMatmul);
  return op;
}
const AclnnOp& SwiGluOp() {
  static const AclnnOp op(ops::kSwiGlu);
  return op;
}
const AclnnOp& ScatterPaKvCacheOp() {
  static const AclnnOp op(ops::kScatterPaKvCache);
  return op;
}
const AclnnOp& FusedInferAttentionOp() {
  static const AclnnOp op(ops950::kFusedInferAttentionScoreV2);
  return op;
}
const AclnnOp& SigmoidOp() {
  static const AclnnOp op(ops950::kSigmoid);
  return op;
}
const AclnnOp& MulOp() {
  static const AclnnOp op(ops950::kMul);
  return op;
}
const AclnnOp& InplaceAddOp() {
  static const AclnnOp op(ops950::kInplaceAdd);
  return op;
}

void Project(const DeviceTensor& a, const DeviceTensor& w_view, const DeviceTensor& out, aclrtStream stream) {
  RunAclnn<ops::MatmulWorkspaceFn>(MatmulOp(), stream, a.get(), w_view.get(), out.get(),
                                   ops::kCubeMathTypeKeepDtype);
}

void ResidualAdd(const DeviceTensor& x_ref, const DeviceTensor& y, aclrtStream stream) {
  AclnnScalar alpha(1.0f);
  RunAclnn<ops950::InplaceAddWorkspaceFn>(InplaceAddOp(), stream, x_ref.get(), y.get(), alpha.get());
}

struct Stages {
  std::vector<float> norm1;
  std::vector<float> qkv;
  std::vector<float> rope_q;
  std::vector<float> rope_k;
  std::vector<float> attn_context;
  std::vector<float> attn_out;
  std::vector<float> norm2;
  std::vector<float> swiglu;
  std::vector<float> output;
  PartialRotaryPath rotary_path = PartialRotaryPath::kPackedApplyRotary;
};

bool AllOperatorsAvailable(std::string* reason) {
  const AclnnOp* required[] = {&RmsNormOp(), &MatmulOp(),  &SwiGluOp(),      &ScatterPaKvCacheOp(),
                               &SigmoidOp(), &MulOp(),     &InplaceAddOp(),  &FusedInferAttentionOp()};
  for (const AclnnOp* op : required) {
    if (!op->available()) {
      *reason = op->unavailable_reason();
      return false;
    }
  }

  PartialRotaryPath path = PartialRotaryPath::kPackedApplyRotary;
  return SelectPartialRotaryPath(s::kHeadDim, s::kRotaryDim, &path, reason);
}

Stages RunLayerOnDevice(const GoldenLayer3& golden) {
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  DeviceTensor x = DeviceTensor::Half({s::kTokens, s::kHidden}, golden.input_x);
  DeviceTensor gamma1 = DeviceTensor::Half({s::kHidden}, golden.input_norm_gamma);
  DeviceTensor gamma2 = DeviceTensor::Half({s::kHidden}, golden.post_attn_norm_gamma);

  DeviceTensor w_q = DeviceTensor::HalfTransposed2D(s::kQDim, s::kHidden, golden.w_q);
  DeviceTensor w_k = DeviceTensor::HalfTransposed2D(s::kKvDim, s::kHidden, golden.w_k);
  DeviceTensor w_v = DeviceTensor::HalfTransposed2D(s::kKvDim, s::kHidden, golden.w_v);
  DeviceTensor w_gate_attn = DeviceTensor::HalfTransposed2D(s::kQDim, s::kHidden, golden.w_gate_attn);
  DeviceTensor w_out = DeviceTensor::HalfTransposed2D(s::kHidden, s::kQDim, golden.w_out);
  DeviceTensor w_gate = DeviceTensor::HalfTransposed2D(s::kIntermediate, s::kHidden, golden.w_gate);
  DeviceTensor w_up = DeviceTensor::HalfTransposed2D(s::kIntermediate, s::kHidden, golden.w_up);
  DeviceTensor w_down = DeviceTensor::HalfTransposed2D(s::kHidden, s::kIntermediate, golden.w_down);

  DeviceTensor norm1 = DeviceTensor::HalfEmpty({s::kTokens, s::kHidden});
  DeviceTensor rstd = DeviceTensor::FloatEmpty({s::kTokens, 1});
  DeviceTensor q = DeviceTensor::HalfEmpty({s::kTokens, s::kQDim});
  DeviceTensor k = DeviceTensor::HalfEmpty({s::kTokens, s::kKvDim});
  DeviceTensor v = DeviceTensor::HalfEmpty({s::kTokens, s::kKvDim});
  DeviceTensor attn_gate = DeviceTensor::HalfEmpty({s::kTokens, s::kQDim});
  DeviceTensor q_pre_rope = DeviceTensor::HalfEmpty({s::kTokens, s::kQDim});
  DeviceTensor k_pre_rope = DeviceTensor::HalfEmpty({s::kTokens, s::kKvDim});
  DeviceTensor gate_sigmoid = DeviceTensor::HalfEmpty({s::kTokens, s::kQDim});
  DeviceTensor gated = DeviceTensor::HalfEmpty({s::kTokens, s::kQDim});
  DeviceTensor attn_out = DeviceTensor::HalfEmpty({s::kTokens, s::kHidden});
  DeviceTensor norm2 = DeviceTensor::HalfEmpty({s::kTokens, s::kHidden});
  DeviceTensor swiglu = DeviceTensor::HalfEmpty({s::kTokens, s::kIntermediate});
  DeviceTensor mlp_out = DeviceTensor::HalfEmpty({s::kTokens, s::kHidden});

  DeviceTensor context = DeviceTensor::HalfEmpty({s::kTokens, s::kNumHeads, s::kHeadDim});
  DeviceTensor softmax_lse = DeviceTensor::HalfEmpty({1});

  RunAclnn<ops::RmsNormWorkspaceFn>(RmsNormOp(), stream, x.get(), gamma1.get(),
                                    static_cast<double>(s::kRmsNormEps), norm1.get(), rstd.get());

  Project(norm1, w_q, q, stream);
  Project(norm1, w_k, k, stream);
  Project(norm1, w_v, v, stream);
  Project(norm1, w_gate_attn, attn_gate, stream);

  ACL_CHECK(aclrtMemcpyAsync(q_pre_rope.data(), static_cast<size_t>(s::kTokens * s::kQDim) * kHalfBytes,
                             q.data(), static_cast<size_t>(s::kTokens * s::kQDim) * kHalfBytes,
                             ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
  ACL_CHECK(aclrtMemcpyAsync(k_pre_rope.data(), static_cast<size_t>(s::kTokens * s::kKvDim) * kHalfBytes,
                             k.data(), static_cast<size_t>(s::kTokens * s::kKvDim) * kHalfBytes,
                             ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
  ACL_CHECK(aclrtSynchronizeStream(stream));

  Stages stages;
  stages.rotary_path = ApplyPartialRotaryQK(q.data(), k.data(), golden.cos_tab, golden.sin_tab, s::kTokens,
                                            s::kNumHeads, s::kNumKvHeads, s::kHeadDim, s::kRotaryDim, stream);

  DeviceTensor key_cache =
      DeviceTensor::HalfEmpty({s::kNumBlocks, s::kBlockSize, s::kNumKvHeads, s::kHeadDim});
  DeviceTensor value_cache =
      DeviceTensor::HalfEmpty({s::kNumBlocks, s::kBlockSize, s::kNumKvHeads, s::kHeadDim});

  const std::vector<int32_t> slot_host{s::kPhysicalBlock * static_cast<int32_t>(s::kBlockSize)};
  const std::vector<int32_t> block_table_host{s::kPhysicalBlock};
  DeviceTensor slot_mapping = DeviceTensor::Int32({s::kTokens}, slot_host);
  DeviceTensor block_table = DeviceTensor::Int32({s::kTokens, s::kMaxBlocksPerSeq}, block_table_host);

  AclnnTensor k_bnd({s::kTokens, s::kNumKvHeads, s::kHeadDim}, ACL_FLOAT16, k.data());
  AclnnTensor v_bnd({s::kTokens, s::kNumKvHeads, s::kHeadDim}, ACL_FLOAT16, v.data());

  RunAclnn<ops::ScatterPaKvCacheWorkspaceFn>(
      ScatterPaKvCacheOp(), stream, k_bnd.get(), key_cache.get(), slot_mapping.get(), v_bnd.get(),
      value_cache.get(), nullptr, nullptr, nullptr,
      kScatterCacheMode, nullptr, nullptr, nullptr);

  const std::vector<int64_t> cache_view =
      s::FiaKeyCacheView(s::kNumBlocks, s::kBlockSize, s::kNumKvHeads, s::kHeadDim);
  AclnnTensor key_cache_flat(cache_view, ACL_FLOAT16, key_cache.data());
  AclnnTensor value_cache_flat(cache_view, ACL_FLOAT16, value_cache.data());
  AclnnTensorList key_list({key_cache_flat.get()});
  AclnnTensorList value_list({value_cache_flat.get()});

  AclnnTensor query_tnd({s::kTokens, s::kNumHeads, s::kHeadDim}, ACL_FLOAT16, q.data());
  AclnnIntArray actual_seq_lengths(std::vector<int64_t>{s::kTokens});
  AclnnIntArray actual_seq_lengths_kv(std::vector<int64_t>{s::kContextLen});

  RunAclnn<ops950::FusedInferAttentionScoreV2WorkspaceFn>(
      FusedInferAttentionOp(), stream, query_tnd.get(), key_list.get(), value_list.get(),
      nullptr, nullptr, actual_seq_lengths.get(), actual_seq_lengths_kv.get(),
      nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, block_table.get(),
      nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr,
      s::kNumHeads, static_cast<double>(s::kAttentionScale), s::kFiaUnboundedTokens, s::kFiaUnboundedTokens,
      kFiaLayout, s::kNumKvHeads, s::kFiaSparseModeNone, s::kFiaInnerPreciseDefault, s::kBlockSize,
      0, false, 0, 0,
      context.get(), softmax_lse.get());

  RunAclnn<ops950::SigmoidWorkspaceFn>(SigmoidOp(), stream, attn_gate.get(), gate_sigmoid.get());

  AclnnTensor context_flat({s::kTokens, s::kQDim}, ACL_FLOAT16, context.data());
  RunAclnn<ops950::MulWorkspaceFn>(MulOp(), stream, context_flat.get(), gate_sigmoid.get(), gated.get());

  Project(gated, w_out, attn_out, stream);
  ResidualAdd(x, attn_out, stream);

  RunAclnn<ops::RmsNormWorkspaceFn>(RmsNormOp(), stream, x.get(), gamma2.get(),
                                    static_cast<double>(s::kRmsNormEps), norm2.get(), rstd.get());

  static_assert(s::kTokens == 1, "gate/up share one buffer, which only works for a single row");
  static_assert((s::kIntermediate * 2 * static_cast<int64_t>(kHalfBytes)) % 32 == 0,
                "the up-projection view must land on a 32-byte boundary");

  DeviceBuffer gate_up(static_cast<size_t>(s::kTokens * 2 * s::kIntermediate) * kHalfBytes);
  auto* gate_up_bytes = static_cast<uint8_t*>(gate_up.get());

  AclnnTensor gate_view({s::kTokens, s::kIntermediate}, ACL_FLOAT16, gate_up_bytes);
  AclnnTensor up_view({s::kTokens, s::kIntermediate}, ACL_FLOAT16,
                      gate_up_bytes + static_cast<size_t>(s::kIntermediate) * kHalfBytes);
  AclnnTensor gate_up_view({s::kTokens, 2 * s::kIntermediate}, ACL_FLOAT16, gate_up_bytes);

  RunAclnn<ops::MatmulWorkspaceFn>(MatmulOp(), stream, norm2.get(), w_gate.get(), gate_view.get(),
                                   ops::kCubeMathTypeKeepDtype);
  RunAclnn<ops::MatmulWorkspaceFn>(MatmulOp(), stream, norm2.get(), w_up.get(), up_view.get(),
                                   ops::kCubeMathTypeKeepDtype);
  RunAclnn<ops::SwiGluWorkspaceFn>(SwiGluOp(), stream, gate_up_view.get(), kSwiGluSplitDim, swiglu.get());
  Project(swiglu, w_down, mlp_out, stream);

  ResidualAdd(x, mlp_out, stream);

  ACL_CHECK(aclrtSynchronizeStream(stream));

  stages.norm1 = norm1.ToFloatFromHalf();
  stages.rope_q = q.ToFloatFromHalf();
  stages.rope_k = k.ToFloatFromHalf();
  stages.attn_context = context.ToFloatFromHalf();
  stages.attn_out = attn_out.ToFloatFromHalf();
  stages.norm2 = norm2.ToFloatFromHalf();
  stages.swiglu = swiglu.ToFloatFromHalf();
  stages.output = x.ToFloatFromHalf();

  const std::vector<float> q_host = q_pre_rope.ToFloatFromHalf();
  const std::vector<float> k_host = k_pre_rope.ToFloatFromHalf();
  const std::vector<float> v_host = v.ToFloatFromHalf();
  stages.qkv.reserve(q_host.size() + k_host.size() + v_host.size());
  stages.qkv.insert(stages.qkv.end(), q_host.begin(), q_host.end());
  stages.qkv.insert(stages.qkv.end(), k_host.begin(), k_host.end());
  stages.qkv.insert(stages.qkv.end(), v_host.begin(), v_host.end());

  return stages;
}

std::vector<float> RoundToHalf(const std::vector<float>& values) { return QuantizeToHalf(values); }

std::vector<float> ProjectOnCpu(const std::vector<float>& a, const std::vector<float>& w, int64_t m, int64_t k,
                                int64_t n) {
  std::vector<float> out;
  reference::MatmulTransposedB(a, w, m, k, n, &out);
  return RoundToHalf(out);
}

Stages RunLayerOnCpu(const GoldenLayer3& golden) {
  Stages cpu;

  std::vector<float> norm1;
  std::vector<float> rstd;
  reference::RmsNorm(golden.input_x, golden.input_norm_gamma, s::kTokens, s::kHidden, s::kRmsNormEps, &norm1,
                     &rstd);
  cpu.norm1 = RoundToHalf(norm1);

  const std::vector<float> q = ProjectOnCpu(cpu.norm1, golden.w_q, s::kTokens, s::kHidden, s::kQDim);
  const std::vector<float> k = ProjectOnCpu(cpu.norm1, golden.w_k, s::kTokens, s::kHidden, s::kKvDim);
  const std::vector<float> v = ProjectOnCpu(cpu.norm1, golden.w_v, s::kTokens, s::kHidden, s::kKvDim);
  const std::vector<float> attn_gate =
      ProjectOnCpu(cpu.norm1, golden.w_gate_attn, s::kTokens, s::kHidden, s::kQDim);

  cpu.qkv.reserve(q.size() + k.size() + v.size());
  cpu.qkv.insert(cpu.qkv.end(), q.begin(), q.end());
  cpu.qkv.insert(cpu.qkv.end(), k.begin(), k.end());
  cpu.qkv.insert(cpu.qkv.end(), v.begin(), v.end());

  std::vector<float> rope_q;
  reference::ApplyRotaryPosEmb(q, golden.cos_tab, golden.sin_tab, s::kTokens, s::kNumHeads, s::kHeadDim,
                               s::kRotaryDim, reference::RotaryMode::kHalf, &rope_q);
  cpu.rope_q = RoundToHalf(rope_q);

  std::vector<float> rope_k;
  reference::ApplyRotaryPosEmb(k, golden.cos_tab, golden.sin_tab, s::kTokens, s::kNumKvHeads, s::kHeadDim,
                               s::kRotaryDim, reference::RotaryMode::kHalf, &rope_k);
  cpu.rope_k = RoundToHalf(rope_k);

  static_assert(s::kContextLen == 1, "the CPU stage-5 model assumes a single context position");
  const int64_t group = s::kNumHeads / s::kNumKvHeads;
  cpu.attn_context.assign(static_cast<size_t>(s::kNumHeads * s::kHeadDim), 0.0f);
  for (int64_t head = 0; head < s::kNumHeads; ++head) {
    const int64_t kv_head = head / group;
    for (int64_t dim = 0; dim < s::kHeadDim; ++dim) {
      cpu.attn_context[static_cast<size_t>(head * s::kHeadDim + dim)] =
          v[static_cast<size_t>(kv_head * s::kHeadDim + dim)];
    }
  }

  std::vector<float> gated(static_cast<size_t>(s::kTokens * s::kQDim));
  for (size_t i = 0; i < gated.size(); ++i) {
    const float sigmoid = 1.0f / (1.0f + std::exp(-attn_gate[i]));
    gated[i] = cpu.attn_context[i] * sigmoid;
  }
  gated = RoundToHalf(gated);

  cpu.attn_out = ProjectOnCpu(gated, golden.w_out, s::kTokens, s::kQDim, s::kHidden);

  std::vector<float> x(golden.input_x);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] += cpu.attn_out[i];
  }
  x = RoundToHalf(x);

  std::vector<float> norm2;
  reference::RmsNorm(x, golden.post_attn_norm_gamma, s::kTokens, s::kHidden, s::kRmsNormEps, &norm2, &rstd);
  cpu.norm2 = RoundToHalf(norm2);

  const std::vector<float> mlp_gate =
      ProjectOnCpu(cpu.norm2, golden.w_gate, s::kTokens, s::kHidden, s::kIntermediate);
  const std::vector<float> mlp_up =
      ProjectOnCpu(cpu.norm2, golden.w_up, s::kTokens, s::kHidden, s::kIntermediate);

  std::vector<float> gate_up;
  gate_up.reserve(mlp_gate.size() + mlp_up.size());
  gate_up.insert(gate_up.end(), mlp_gate.begin(), mlp_gate.end());
  gate_up.insert(gate_up.end(), mlp_up.begin(), mlp_up.end());

  std::vector<float> swiglu;
  reference::SiluAndMul(gate_up, s::kTokens, s::kIntermediate, &swiglu);
  cpu.swiglu = RoundToHalf(swiglu);

  const std::vector<float> mlp_out =
      ProjectOnCpu(cpu.swiglu, golden.w_down, s::kTokens, s::kIntermediate, s::kHidden);

  for (size_t i = 0; i < x.size(); ++i) {
    x[i] += mlp_out[i];
  }
  cpu.output = RoundToHalf(x);

  return cpu;
}

class QwenLayer3DumpTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!LoadGoldenLayer3(&golden_, &error_)) {
      return;
    }
    loaded_ = true;
    cpu_ = RunLayerOnCpu(golden_);
  }

  static void TearDownTestSuite() {
    golden_ = GoldenLayer3();
    cpu_ = Stages();
    loaded_ = false;
    error_.clear();
  }

  void SetUp() override {
    if (!loaded_) {
      GTEST_SKIP() << "golden dump unavailable: " << error_;
    }
  }

  static GoldenLayer3 golden_;
  static Stages cpu_;
  static bool loaded_;
  static std::string error_;
};

GoldenLayer3 QwenLayer3DumpTest::golden_;
Stages QwenLayer3DumpTest::cpu_;
bool QwenLayer3DumpTest::loaded_ = false;
std::string QwenLayer3DumpTest::error_;

TEST_F(QwenLayer3DumpTest, EveryTensorHasTheExpectedLength) {
  EXPECT_EQ(golden_.input_x.size(), static_cast<size_t>(s::kTokens * s::kHidden));
  EXPECT_EQ(golden_.cos_tab.size(), static_cast<size_t>(s::kTokens * s::kRotaryDim));
  EXPECT_EQ(golden_.sin_tab.size(), static_cast<size_t>(s::kTokens * s::kRotaryDim));
  EXPECT_EQ(golden_.tap_qkv.size(), static_cast<size_t>(s::kTokens * (s::kQDim + 2 * s::kKvDim)));
  EXPECT_EQ(golden_.tap_swiglu.size(), static_cast<size_t>(s::kTokens * s::kIntermediate));
  EXPECT_EQ(golden_.w_down.size(), static_cast<size_t>(s::kHidden * s::kIntermediate));
  EXPECT_EQ(golden_.golden_output.size(), static_cast<size_t>(s::kTokens * s::kHidden));
}

TEST_F(QwenLayer3DumpTest, Stage1InputRmsNorm) {
  EXPECT_TENSORS_ALLCLOSE(cpu_.norm1, golden_.tap_norm1, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3DumpTest, Stage2QkvAndAttentionGateProjections) {
  EXPECT_TENSORS_ALLCLOSE(cpu_.qkv, golden_.tap_qkv, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3DumpTest, Stage3PartialRotaryQ) {
  EXPECT_TENSORS_ALLCLOSE(cpu_.rope_q, golden_.tap_rope_q, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3DumpTest, Stage4PartialRotaryK) {
  EXPECT_TENSORS_ALLCLOSE(cpu_.rope_k, golden_.tap_rope_k, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3DumpTest, Stage6AttentionGateOutProjectionAndResidual) {
  EXPECT_TENSORS_ALLCLOSE(cpu_.attn_out, golden_.tap_attn_out, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3DumpTest, Stage7PostAttentionRmsNorm) {
  EXPECT_TENSORS_ALLCLOSE(cpu_.norm2, golden_.tap_norm2, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3DumpTest, Stage8SwiGlu) {
  EXPECT_TENSORS_ALLCLOSE(cpu_.swiglu, golden_.tap_swiglu, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3DumpTest, Stage9LayerOutputMatchesGolden) {
  EXPECT_TENSORS_ALLCLOSE(cpu_.output, golden_.golden_output, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3DumpTest, RotaryIsTheIdentityAtThisPosition) {
  for (size_t i = 0; i < golden_.cos_tab.size(); ++i) {
    EXPECT_FLOAT_EQ(golden_.cos_tab[i], 1.0f) << "cos[" << i << "]: dump is not at position 0";
    EXPECT_FLOAT_EQ(golden_.sin_tab[i], 0.0f) << "sin[" << i << "]: dump is not at position 0";
  }
}

TEST(QwenLayer3Loader, ReportsAnUnfetchedLfsPointer) {
  const std::string path = std::string(::testing::TempDir()) + "/vllm_ascend_lfs_pointer.bin";
  {
    std::ofstream file(path.c_str(), std::ios::binary);
    ASSERT_TRUE(file.is_open()) << "cannot write " << path;
    file << "version https://git-lfs.github.com/spec/v1\n"
         << "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
         << "size 4096\n";
  }

  std::vector<float> values;
  std::string error;
  EXPECT_FALSE(ReadHalfFile(path, 2048, &values, &error));
  EXPECT_NE(error.find("git lfs pull"), std::string::npos) << "actual message: " << error;
  std::remove(path.c_str());
}

TEST(QwenLayer3Loader, ReportsASizeMismatchInBytes) {
  const std::string path = std::string(::testing::TempDir()) + "/vllm_ascend_short_dump.bin";
  {
    std::ofstream file(path.c_str(), std::ios::binary);
    ASSERT_TRUE(file.is_open()) << "cannot write " << path;
    const std::vector<uint16_t> bits(8, 0);
    file.write(reinterpret_cast<const char*>(bits.data()),
               static_cast<std::streamsize>(bits.size() * sizeof(uint16_t)));
  }

  std::vector<float> values;
  std::string error;
  EXPECT_FALSE(ReadHalfFile(path, 2048, &values, &error));
  EXPECT_NE(error.find("16 bytes"), std::string::npos) << "actual message: " << error;
  EXPECT_NE(error.find("expected 4096"), std::string::npos) << "actual message: " << error;
  std::remove(path.c_str());
}

TEST(QwenLayer3Loader, ReportsAMissingFile) {
  std::vector<float> values;
  std::string error;
  EXPECT_FALSE(ReadHalfFile("/vllm-ascend-no-such-directory/no_such_dump.bin", 8, &values, &error));
  EXPECT_NE(error.find("cannot open"), std::string::npos) << "actual message: " << error;
}

class QwenLayer3Golden950PrTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    AscendTestEnvironment& environment = AscendTestEnvironment::Instance();
    if (!environment.available()) {
      unavailable_ = "No usable Ascend device: " + environment.unavailable_reason();
      return;
    }
    if (!environment.is_950pr()) {
      unavailable_ = "Test targets Ascend 950PR; attached device reports '" + environment.soc_name() + "'";
      return;
    }
    std::string reason;
    if (!AllOperatorsAvailable(&reason)) {
      unavailable_ = "pipeline operator unavailable: " + reason;
      return;
    }
    if (!LoadGoldenLayer3(&golden_, &reason)) {
      unavailable_ = "golden dump unavailable: " + reason;
      return;
    }
    stages_ = RunLayerOnDevice(golden_);
  }

  static void TearDownTestSuite() {
    golden_ = GoldenLayer3();
    stages_ = Stages();
    unavailable_.clear();
  }

  void SetUp() override {
    if (!unavailable_.empty()) {
      GTEST_SKIP() << unavailable_;
    }
    RecordProperty("rotary_path", PartialRotaryPathName(stages_.rotary_path));
  }

  static GoldenLayer3 golden_;
  static Stages stages_;
  static std::string unavailable_;
};

GoldenLayer3 QwenLayer3Golden950PrTest::golden_;
Stages QwenLayer3Golden950PrTest::stages_;
std::string QwenLayer3Golden950PrTest::unavailable_;

TEST_F(QwenLayer3Golden950PrTest, Stage1InputRmsNorm) {
  EXPECT_TENSORS_ALLCLOSE(stages_.norm1, golden_.tap_norm1, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3Golden950PrTest, Stage2QkvAndAttentionGateProjections) {
  EXPECT_TENSORS_ALLCLOSE(stages_.qkv, golden_.tap_qkv, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3Golden950PrTest, Stage3PartialRotaryQ) {
  EXPECT_TENSORS_ALLCLOSE(stages_.rope_q, golden_.tap_rope_q, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3Golden950PrTest, Stage4PartialRotaryK) {
  EXPECT_TENSORS_ALLCLOSE(stages_.rope_k, golden_.tap_rope_k, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3Golden950PrTest, Stage3And4LeaveChannelsPastRotaryDimUntouched) {
  ASSERT_EQ(stages_.rope_q.size(), stages_.qkv.size() - 2 * static_cast<size_t>(s::kKvDim));
  for (int64_t head = 0; head < s::kNumHeads; ++head) {
    for (int64_t dim = s::kRotaryDim; dim < s::kHeadDim; ++dim) {
      const size_t index = static_cast<size_t>(head * s::kHeadDim + dim);
      ASSERT_EQ(stages_.rope_q[index], stages_.qkv[index])
          << "query head " << head << " channel " << dim << " was modified past rotary_dim";
    }
  }
  const size_t k_base = static_cast<size_t>(s::kQDim);
  for (int64_t head = 0; head < s::kNumKvHeads; ++head) {
    for (int64_t dim = s::kRotaryDim; dim < s::kHeadDim; ++dim) {
      const size_t offset = static_cast<size_t>(head * s::kHeadDim + dim);
      ASSERT_EQ(stages_.rope_k[offset], stages_.qkv[k_base + offset])
          << "key head " << head << " channel " << dim << " was modified past rotary_dim";
    }
  }
}

TEST_F(QwenLayer3Golden950PrTest, Stage5DecodeAttentionContextIsTheCachedValue) {
  static_assert(s::kContextLen == 1, "this identity only holds for a single context position");
  static_assert(s::kNumHeads % s::kNumKvHeads == 0, "GQA needs a whole number of query heads per kv head");

  ASSERT_EQ(stages_.attn_context.size(), static_cast<size_t>(s::kNumHeads * s::kHeadDim));

  const size_t v_base = static_cast<size_t>(s::kQDim + s::kKvDim);
  ASSERT_EQ(golden_.tap_qkv.size(), v_base + static_cast<size_t>(s::kKvDim));

  const int64_t group = s::kNumHeads / s::kNumKvHeads;
  std::vector<float> expected(static_cast<size_t>(s::kNumHeads * s::kHeadDim));
  for (int64_t head = 0; head < s::kNumHeads; ++head) {
    const int64_t kv_head = head / group;
    for (int64_t dim = 0; dim < s::kHeadDim; ++dim) {
      expected[static_cast<size_t>(head * s::kHeadDim + dim)] =
          golden_.tap_qkv[v_base + static_cast<size_t>(kv_head * s::kHeadDim + dim)];
    }
  }

  EXPECT_TENSORS_ALLCLOSE(stages_.attn_context, expected, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3Golden950PrTest, Stage6AttentionGateOutProjectionAndResidual) {
  EXPECT_TENSORS_ALLCLOSE(stages_.attn_out, golden_.tap_attn_out, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3Golden950PrTest, Stage7PostAttentionRmsNorm) {
  EXPECT_TENSORS_ALLCLOSE(stages_.norm2, golden_.tap_norm2, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3Golden950PrTest, Stage8SwiGlu) {
  EXPECT_TENSORS_ALLCLOSE(stages_.swiglu, golden_.tap_swiglu, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3Golden950PrTest, Stage9LayerOutputMatchesGolden) {
  EXPECT_TENSORS_ALLCLOSE(stages_.output, golden_.golden_output, kFp16DefaultTolerance);
}

}
}
}
