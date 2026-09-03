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

// End-to-end parity for one Qwen3.5 decoder layer (layer 3, the first
// full_attention block) against a PyTorch dump.
//
// Every other test in this directory checks one operator against a CPU
// reference. This one checks that the operators compose: it runs a whole layer
// on the device - RMSNorm, four projections, partial RoPE, a paged KV write and
// decode, the attention output gate, the out projection, two residual adds,
// SwiGLU and the MLP - and compares seven intermediate taps plus the final
// output against scripts/dump_qwen35_layer3.py.
//
// Why taps rather than only the final output: a single end-to-end comparison
// tells you a layer is wrong, not where. The taps localise it to a stage, which
// is the entire reason for dumping them. They only work because the dumper
// rounds to fp16 at every stage boundary exactly as the kernels do - see the
// numerics note at the top of that script.
//
// The whole layer is enqueued on one stream with no host synchronisation until
// the end, which is how it would run in a real decode step. The taps are read
// back afterwards from the buffers that still hold them; the only values that
// do not survive are pre-RoPE Q and K, since RoPE is in place, so those are
// copied device-to-device before the rotation.
//
// The data lives in csrc/tests/data/golden_layer3 and is tracked with Git LFS.
// A tree where LFS was never fetched has pointer files there instead of dumps,
// so the loader recognises that case and skips with the command to fix it
// rather than failing on a size mismatch.
//
// WHAT THE SHIPPED DUMP DOES NOT TEST. It is pos=0, ctx_len=1, the decode step
// the requirement names, and that configuration has two blind spots - the
// second of them sharp enough to be worth stating plainly:
//
// 1. At position 0 the rotary tables are cos=1 / sin=0, so RoPE is the identity
//    and tap_rope_q equals the pre-RoPE Q.
//
// 2. With a single context position the softmax runs over one element and is
//    therefore exactly 1.0 whatever the score is. The attention context is then
//    just V, which means THE LAYER OUTPUT IS INDEPENDENT OF Q, K AND RoPE. This
//    was confirmed by regenerating the dump at pos=1024: cos/sin and
//    tap_rope_q change, golden_output.bin does not, byte for byte.
//
// So at ctx_len=1 a bug in the Q projection, the K projection, RoPE or the KV
// cache write cannot reach golden_output.bin - only tap_qkv, tap_rope_q and
// tap_rope_k can catch it. That is precisely why the taps exist, and it is the
// reason to prefer a regenerated `--pos N --ctx-len M` dump for any change to
// those stages. Nothing in this file depends on the position or the context
// length; both come from the dump.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

#include "cuda_device_tensor.hpp"
#include "cuda_kernels.hpp"
#include "cuda_runtime.hpp"
#include "fp16.hpp"
#include "tensor_compare.hpp"

#ifndef VLLM_ASCEND_GOLDEN_LAYER3_DIR
#define VLLM_ASCEND_GOLDEN_LAYER3_DIR ""
#endif

namespace vllm_ascend {
namespace test {
namespace {

// --- layer 3 configuration, mirroring scripts/dump_qwen35_layer3.py ----------

constexpr int64_t kTokens = 1;
constexpr int64_t kHidden = 2048;
constexpr int64_t kIntermediate = 6144;
constexpr int64_t kNumHeads = 8;
constexpr int64_t kNumKvHeads = 2;
constexpr int64_t kHeadDim = 256;
constexpr int64_t kQDim = kNumHeads * kHeadDim;     // 2048
constexpr int64_t kKvDim = kNumKvHeads * kHeadDim;  // 512
constexpr int64_t kRotaryDim = 64;                  // partial_rotary_factor 0.25
constexpr float kRmsNormEps = 1e-6f;
constexpr float kAttentionScale = 0.0625f;          // 1/sqrt(256), exact in fp16

// Paging. One token needs one block; a non-zero physical block is used so a
// block table that is ignored reads zeros rather than passing by luck.
constexpr int64_t kBlockSize = 64;
constexpr int64_t kNumBlocks = 4;
constexpr int32_t kPhysicalBlock = 2;
constexpr int64_t kContextLen = 1;
constexpr int64_t kMaxBlocksPerSeq = 1;

// std::getenv trips MSVC's C4996 and this suite builds warnings as errors, so
// the sanctioned _dupenv_s is used there and plain getenv everywhere else.
std::string EnvOrEmpty(const char* name) {
#ifdef _MSC_VER
  char* value = nullptr;
  size_t length = 0;
  if (_dupenv_s(&value, &length, name) != 0 || value == nullptr) {
    return std::string();
  }
  const std::string result(value);
  std::free(value);
  return result;
#else
  const char* value = std::getenv(name);
  return (value != nullptr) ? std::string(value) : std::string();
#endif
}

std::string GoldenDir() {
  const std::string env = EnvOrEmpty("QWEN_GOLDEN_LAYER3_DIR");
  if (!env.empty()) {
    return env;
  }
  return VLLM_ASCEND_GOLDEN_LAYER3_DIR;
}

// Raw little-endian fp16, no header, widened to float on the way in. Returns
// false and sets `error` rather than throwing, so a missing dump becomes a skip
// with an actionable message instead of a failure.
bool ReadHalfFile(const std::string& path, size_t expected_elements, std::vector<float>* out,
                  std::string* error) {
  std::ifstream file(path.c_str(), std::ios::binary | std::ios::ate);
  if (!file) {
    *error = "cannot open " + path;
    return false;
  }

  const std::streamsize size_bytes = file.tellg();
  file.seekg(0, std::ios::beg);

  const std::streamsize expected_bytes = static_cast<std::streamsize>(expected_elements * sizeof(uint16_t));
  if (size_bytes != expected_bytes) {
    // An unfetched LFS pointer is a ~130 byte text file whose first line is
    // "version https://git-lfs.github.com/spec/v1".
    char head[8] = {0};
    file.read(head, static_cast<std::streamsize>(sizeof(head) - 1));
    if (file.gcount() >= 7 && std::string(head) == "version") {
      *error = path + " is an unfetched Git LFS pointer; run `git lfs pull`";
    } else {
      *error = path + " is " + std::to_string(static_cast<long long>(size_bytes)) +
               " bytes, expected " + std::to_string(static_cast<long long>(expected_bytes));
    }
    return false;
  }

  std::vector<uint16_t> bits(expected_elements);
  file.read(reinterpret_cast<char*>(bits.data()), expected_bytes);
  if (file.gcount() != expected_bytes) {
    *error = "short read from " + path;
    return false;
  }

  out->resize(expected_elements);
  for (size_t i = 0; i < expected_elements; ++i) {
    (*out)[i] = HalfBitsToFloat(bits[i]);
  }
  return true;
}

// Everything the dump provides, widened to float.
struct Golden {
  std::vector<float> input_x;
  std::vector<float> input_norm_gamma;
  std::vector<float> post_attn_norm_gamma;
  std::vector<float> w_q, w_k, w_v, w_gate_attn, w_out;
  std::vector<float> w_gate, w_up, w_down;
  std::vector<float> cos_tab, sin_tab;
  std::vector<float> tap_norm1, tap_qkv, tap_rope_q, tap_rope_k;
  std::vector<float> tap_attn_out, tap_norm2, tap_swiglu;
  std::vector<float> golden_output;
};

// What the device produced, read back after the layer has run.
struct Stages {
  std::vector<float> norm1;
  std::vector<float> qkv;  // pre-RoPE q | k | v, concatenated to match the dump
  std::vector<float> rope_q;
  std::vector<float> rope_k;
  std::vector<float> attn_out;
  std::vector<float> norm2;
  std::vector<float> swiglu;
  std::vector<float> output;
};

bool LoadGolden(Golden* golden, std::string* error) {
  const std::string dir = GoldenDir();
  if (dir.empty()) {
    *error = "VLLM_ASCEND_GOLDEN_LAYER3_DIR was not defined at compile time";
    return false;
  }

  struct Entry {
    const char* file;
    size_t elements;
    std::vector<float>* target;
  };

  const Entry entries[] = {
      {"input_x.bin", static_cast<size_t>(kTokens * kHidden), &golden->input_x},
      {"input_norm_gamma.bin", static_cast<size_t>(kHidden), &golden->input_norm_gamma},
      {"post_attn_norm_gamma.bin", static_cast<size_t>(kHidden), &golden->post_attn_norm_gamma},
      {"w_q.bin", static_cast<size_t>(kQDim * kHidden), &golden->w_q},
      {"w_k.bin", static_cast<size_t>(kKvDim * kHidden), &golden->w_k},
      {"w_v.bin", static_cast<size_t>(kKvDim * kHidden), &golden->w_v},
      {"w_gate_attn.bin", static_cast<size_t>(kQDim * kHidden), &golden->w_gate_attn},
      {"w_out.bin", static_cast<size_t>(kHidden * kQDim), &golden->w_out},
      {"w_gate.bin", static_cast<size_t>(kIntermediate * kHidden), &golden->w_gate},
      {"w_up.bin", static_cast<size_t>(kIntermediate * kHidden), &golden->w_up},
      {"w_down.bin", static_cast<size_t>(kHidden * kIntermediate), &golden->w_down},
      {"cos_tab_d64.bin", static_cast<size_t>(kTokens * kRotaryDim), &golden->cos_tab},
      {"sin_tab_d64.bin", static_cast<size_t>(kTokens * kRotaryDim), &golden->sin_tab},
      {"tap_norm1.bin", static_cast<size_t>(kTokens * kHidden), &golden->tap_norm1},
      {"tap_qkv.bin", static_cast<size_t>(kTokens * (kQDim + 2 * kKvDim)), &golden->tap_qkv},
      {"tap_rope_q.bin", static_cast<size_t>(kTokens * kQDim), &golden->tap_rope_q},
      {"tap_rope_k.bin", static_cast<size_t>(kTokens * kKvDim), &golden->tap_rope_k},
      {"tap_attn_out.bin", static_cast<size_t>(kTokens * kHidden), &golden->tap_attn_out},
      {"tap_norm2.bin", static_cast<size_t>(kTokens * kHidden), &golden->tap_norm2},
      {"tap_swiglu.bin", static_cast<size_t>(kTokens * kIntermediate), &golden->tap_swiglu},
      {"golden_output.bin", static_cast<size_t>(kTokens * kHidden), &golden->golden_output},
  };

  for (const Entry& entry : entries) {
    if (!ReadHalfFile(dir + "/" + entry.file, entry.elements, entry.target, error)) {
      return false;
    }
  }
  return true;
}

// Runs the whole layer on one stream, then reads the taps back.
Stages RunLayerOnDevice(const Golden& golden) {
  CUDADevice& device = CUDATestEnvironment::Instance().device();
  cudaStream_t stream = device.stream();
  cublasHandle_t blas = device.cublas_handle();

  // --- resident state -------------------------------------------------------
  // x is the residual stream and is updated in place twice.
  CudaDeviceTensor x = CudaDeviceTensor::Half({kTokens, kHidden}, golden.input_x);
  CudaDeviceTensor gamma1 = CudaDeviceTensor::Half({kHidden}, golden.input_norm_gamma);
  CudaDeviceTensor gamma2 = CudaDeviceTensor::Half({kHidden}, golden.post_attn_norm_gamma);

  // Linear weights keep the torch [out_features, in_features] layout, which is
  // exactly the transpose_b=true case of CublasGemmFp16.
  CudaDeviceTensor w_q = CudaDeviceTensor::Half({kQDim, kHidden}, golden.w_q);
  CudaDeviceTensor w_k = CudaDeviceTensor::Half({kKvDim, kHidden}, golden.w_k);
  CudaDeviceTensor w_v = CudaDeviceTensor::Half({kKvDim, kHidden}, golden.w_v);
  CudaDeviceTensor w_gate_attn = CudaDeviceTensor::Half({kQDim, kHidden}, golden.w_gate_attn);
  CudaDeviceTensor w_out = CudaDeviceTensor::Half({kHidden, kQDim}, golden.w_out);
  CudaDeviceTensor w_gate = CudaDeviceTensor::Half({kIntermediate, kHidden}, golden.w_gate);
  CudaDeviceTensor w_up = CudaDeviceTensor::Half({kIntermediate, kHidden}, golden.w_up);
  CudaDeviceTensor w_down = CudaDeviceTensor::Half({kHidden, kIntermediate}, golden.w_down);

  CudaDeviceTensor cos_tab = CudaDeviceTensor::Half({kTokens, kRotaryDim}, golden.cos_tab);
  CudaDeviceTensor sin_tab = CudaDeviceTensor::Half({kTokens, kRotaryDim}, golden.sin_tab);

  // --- scratch --------------------------------------------------------------
  CudaDeviceTensor norm1 = CudaDeviceTensor::HalfEmpty({kTokens, kHidden});
  CudaDeviceTensor q = CudaDeviceTensor::HalfEmpty({kTokens, kQDim});
  CudaDeviceTensor k = CudaDeviceTensor::HalfEmpty({kTokens, kKvDim});
  CudaDeviceTensor v = CudaDeviceTensor::HalfEmpty({kTokens, kKvDim});
  CudaDeviceTensor attn_gate = CudaDeviceTensor::HalfEmpty({kTokens, kQDim});
  CudaDeviceTensor q_pre_rope = CudaDeviceTensor::HalfEmpty({kTokens, kQDim});
  CudaDeviceTensor k_pre_rope = CudaDeviceTensor::HalfEmpty({kTokens, kKvDim});
  CudaDeviceTensor context = CudaDeviceTensor::HalfEmpty({kTokens, kQDim});
  CudaDeviceTensor gated = CudaDeviceTensor::HalfEmpty({kTokens, kQDim});
  CudaDeviceTensor attn_out = CudaDeviceTensor::HalfEmpty({kTokens, kHidden});
  CudaDeviceTensor norm2 = CudaDeviceTensor::HalfEmpty({kTokens, kHidden});
  CudaDeviceTensor gate_up = CudaDeviceTensor::HalfEmpty({kTokens, 2 * kIntermediate});
  CudaDeviceTensor swiglu = CudaDeviceTensor::HalfEmpty({kTokens, kIntermediate});
  CudaDeviceTensor mlp_out = CudaDeviceTensor::HalfEmpty({kTokens, kHidden});

  // Allocation zeroes the caches, so every position this decode does not write
  // reads as zero rather than as whatever the allocator handed back.
  CudaDeviceTensor key_cache =
      CudaDeviceTensor::HalfEmpty({kNumBlocks, kNumKvHeads, kBlockSize, kHeadDim});
  CudaDeviceTensor value_cache =
      CudaDeviceTensor::HalfEmpty({kNumBlocks, kNumKvHeads, kBlockSize, kHeadDim});

  const std::vector<int32_t> slot_host{kPhysicalBlock * static_cast<int32_t>(kBlockSize)};
  const std::vector<int32_t> block_table_host{kPhysicalBlock};
  const std::vector<int32_t> context_lens_host{static_cast<int32_t>(kContextLen)};
  CudaDeviceTensor slot_mapping = CudaDeviceTensor::Int32({kTokens}, slot_host);
  CudaDeviceTensor block_table = CudaDeviceTensor::Int32({kTokens, kMaxBlocksPerSeq}, block_table_host);
  CudaDeviceTensor context_lens = CudaDeviceTensor::Int32({kTokens}, context_lens_host);

  // --- 1. input RMSNorm -----------------------------------------------------
  cuda::LaunchRmsNormHalf(x.half_data(), gamma1.half_data(), norm1.half_data(), nullptr, kTokens,
                          kHidden, kRmsNormEps, stream);

  // --- 2. Q / K / V and the attention gate ----------------------------------
  CublasGemmFp16(blas, false, true, kTokens, kQDim, kHidden, norm1.half_data(), w_q.half_data(),
                 q.half_data());
  CublasGemmFp16(blas, false, true, kTokens, kKvDim, kHidden, norm1.half_data(), w_k.half_data(),
                 k.half_data());
  CublasGemmFp16(blas, false, true, kTokens, kKvDim, kHidden, norm1.half_data(), w_v.half_data(),
                 v.half_data());
  CublasGemmFp16(blas, false, true, kTokens, kQDim, kHidden, norm1.half_data(),
                 w_gate_attn.half_data(), attn_gate.half_data());

  // RoPE is in place, so keep a copy of what tap_qkv is supposed to hold.
  CUDA_CHECK(cudaMemcpyAsync(q_pre_rope.data(), q.data(), static_cast<size_t>(kTokens * kQDim) * 2,
                             cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(k_pre_rope.data(), k.data(), static_cast<size_t>(kTokens * kKvDim) * 2,
                             cudaMemcpyDeviceToDevice, stream));

  // --- 3. partial RoPE ------------------------------------------------------
  // rotary_dim < head_dim, so the launcher rotates channels [0, 64) of every
  // head and leaves [64, 256) untouched, which is what partial_rotary_factor
  // 0.25 means.
  cuda::LaunchApplyRotaryPosEmbHalfMode(q.half_data(), cos_tab.half_data(), sin_tab.half_data(),
                                        kTokens, kNumHeads, kHeadDim, kRotaryDim, stream);
  cuda::LaunchApplyRotaryPosEmbHalfMode(k.half_data(), cos_tab.half_data(), sin_tab.half_data(),
                                        kTokens, kNumKvHeads, kHeadDim, kRotaryDim, stream);

  // --- 4. paged KV write and decode -----------------------------------------
  cuda::LaunchPagedCacheScatterHalf(k.half_data(), v.half_data(), slot_mapping.int32_data(),
                                    key_cache.half_data(), value_cache.half_data(), kTokens,
                                    kNumKvHeads, kHeadDim, kBlockSize, stream);
  cuda::LaunchPagedAttentionDecodeV1Half(
      q.half_data(), key_cache.half_data(), value_cache.half_data(), block_table.int32_data(),
      context_lens.int32_data(), context.half_data(), kTokens, kNumHeads, kNumKvHeads, kHeadDim,
      kBlockSize, kMaxBlocksPerSeq, kContextLen, kAttentionScale, stream);

  // --- 5. output gate, out projection, residual -----------------------------
  cuda::LaunchSigmoidGateHalf(context.half_data(), attn_gate.half_data(), gated.half_data(), kQDim,
                              stream);
  CublasGemmFp16(blas, false, true, kTokens, kHidden, kQDim, gated.half_data(), w_out.half_data(),
                 attn_out.half_data());
  cuda::LaunchResidualAddHalf(x.half_data(), attn_out.half_data(), kHidden, stream);

  // --- 6. post-attention RMSNorm --------------------------------------------
  cuda::LaunchRmsNormHalf(x.half_data(), gamma2.half_data(), norm2.half_data(), nullptr, kTokens,
                          kHidden, kRmsNormEps, stream);

  // --- 7. SwiGLU MLP --------------------------------------------------------
  // LaunchSwiGluHalf wants gate and up contiguous as [tokens, 2*intermediate],
  // so the two projections write the two halves of one buffer. That is only
  // valid because kTokens == 1: with more rows the halves would interleave and
  // each projection would need its own buffer.
  static_assert(kTokens == 1, "gate/up share one buffer, which only works for a single row");
  CublasGemmFp16(blas, false, true, kTokens, kIntermediate, kHidden, norm2.half_data(),
                 w_gate.half_data(), gate_up.half_data());
  CublasGemmFp16(blas, false, true, kTokens, kIntermediate, kHidden, norm2.half_data(),
                 w_up.half_data(), gate_up.half_data() + kIntermediate);
  cuda::LaunchSwiGluHalf(gate_up.half_data(), swiglu.half_data(), kTokens, kIntermediate, stream);
  CublasGemmFp16(blas, false, true, kTokens, kHidden, kIntermediate, swiglu.half_data(),
                 w_down.half_data(), mlp_out.half_data());

  // --- 8. final residual ----------------------------------------------------
  cuda::LaunchResidualAddHalf(x.half_data(), mlp_out.half_data(), kHidden, stream);

  device.SynchronizeStream();

  // --- read the taps back ---------------------------------------------------
  Stages stages;
  stages.norm1 = norm1.ToFloatFromHalf();
  stages.rope_q = q.ToFloatFromHalf();
  stages.rope_k = k.ToFloatFromHalf();
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

// The layer runs once for the whole suite; each stage then gets its own test so
// a failure names the stage rather than the whole pipeline.
class QwenLayer3GoldenTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!CUDATestEnvironment::Instance().available()) {
      unavailable_ = "No usable CUDA device: " + CUDATestEnvironment::Instance().unavailable_reason();
      return;
    }
    std::string error;
    if (!LoadGolden(&golden_, &error)) {
      unavailable_ = "golden dump unavailable: " + error;
      return;
    }
    stages_ = RunLayerOnDevice(golden_);
  }

  static void TearDownTestSuite() {
    golden_ = Golden();
    stages_ = Stages();
    unavailable_.clear();
  }

  void SetUp() override {
    if (!unavailable_.empty()) {
      GTEST_SKIP() << unavailable_;
    }
  }

  static Golden golden_;
  static Stages stages_;
  static std::string unavailable_;
};

Golden QwenLayer3GoldenTest::golden_;
Stages QwenLayer3GoldenTest::stages_;
std::string QwenLayer3GoldenTest::unavailable_;

TEST_F(QwenLayer3GoldenTest, Stage1InputRmsNorm) {
  EXPECT_TENSORS_ALLCLOSE(stages_.norm1, golden_.tap_norm1, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3GoldenTest, Stage2QkvProjections) {
  EXPECT_TENSORS_ALLCLOSE(stages_.qkv, golden_.tap_qkv, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3GoldenTest, Stage3PartialRotaryQ) {
  EXPECT_TENSORS_ALLCLOSE(stages_.rope_q, golden_.tap_rope_q, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3GoldenTest, Stage3PartialRotaryK) {
  EXPECT_TENSORS_ALLCLOSE(stages_.rope_k, golden_.tap_rope_k, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3GoldenTest, Stage3LeavesChannelsPastRotaryDimUntouched) {
  // The pass-through half of partial rotary: channels [rotary_dim, head_dim) of
  // every head must survive bit-exactly. At pos=0 the rotated channels are
  // unchanged too, so this is the one rotary assertion that still means
  // something there - it would catch a kernel that rotated the whole head.
  ASSERT_EQ(stages_.rope_q.size(), stages_.qkv.size() - 2 * static_cast<size_t>(kKvDim));
  for (int64_t head = 0; head < kNumHeads; ++head) {
    for (int64_t dim = kRotaryDim; dim < kHeadDim; ++dim) {
      const size_t index = static_cast<size_t>(head * kHeadDim + dim);
      ASSERT_EQ(stages_.rope_q[index], stages_.qkv[index])
          << "head " << head << " channel " << dim << " was modified past rotary_dim";
    }
  }
}

TEST_F(QwenLayer3GoldenTest, Stage5AttentionGateAndOutProjection) {
  EXPECT_TENSORS_ALLCLOSE(stages_.attn_out, golden_.tap_attn_out, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3GoldenTest, Stage6PostAttentionRmsNorm) {
  EXPECT_TENSORS_ALLCLOSE(stages_.norm2, golden_.tap_norm2, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3GoldenTest, Stage7SwiGlu) {
  EXPECT_TENSORS_ALLCLOSE(stages_.swiglu, golden_.tap_swiglu, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3GoldenTest, Stage8LayerOutputMatchesGolden) {
  EXPECT_TENSORS_ALLCLOSE(stages_.output, golden_.golden_output, kFp16DefaultTolerance);
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
