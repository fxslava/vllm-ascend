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

// Real-weight parity for one Qwen3.5 decoder layer (layer 3, the first
// full_attention block) against scripts/dump_qwen35_real_inference.py.
//
// test_qwen_layer_golden.cpp checks the same layer with random weights at
// pos=0, ctx_len=1. That configuration cannot fail for the right reasons: at
// position 0 the rotary tables are cos=1 / sin=0 so RoPE is the identity, and
// over a one-element context the softmax is exactly 1.0 whatever the score is,
// which makes the layer output independent of Q, K and RoPE entirely.
//
// This test uses the real checkpoint and a real decode step:
//
//   * Parameters come from F:\AI\Qwen3.5-2B, layer 3.
//   * The layer input is the hidden state the real model produces at layer 3
//     for a tokenised prompt - layers 0..2 are gated-DeltaNet blocks, so the
//     dump script runs them rather than guessing what their output looks like.
//   * The decode runs at position 64 over a 65-entry KV cache built from the
//     same prompt. cos[0] is 0.392 and sin[0] is 0.920, and the softmax spreads
//     over 65 competing positions at about 3.0 nats of entropy, so both stages
//     carry weight.
//
// Three structural facts about the checkpoint that the synthetic layer does not
// have, all of them handled in the dump and mirrored here:
//
//   1. q_proj is fused: [2 * num_heads * head_dim, hidden], with each head's
//      query rows followed by its output-gate rows. The dump ships the two
//      de-interleaved halves as w_q.bin and w_gate_attn.bin.
//   2. q_norm / k_norm are per-head RMSNorms over head_dim, between the
//      projection and RoPE. LaunchRmsNormHalf serves them by treating the head
//      axis as the token axis.
//   3. Qwen3_5RMSNorm scales by (1 + weight). Every gamma in the dump is
//      already shifted, so the kernel's plain x * rstd * gamma is correct.
//
// The metric
// ----------
// Every stage is gated on cosine similarity >= 0.9999 and reports MAE and
// max |abs error| alongside it. See the note above CompareCosineSimilarity in
// tensor_compare.hpp for why a scale-free metric is the right one once the
// values being compared are a whole layer's activations rather than one
// operator's output, and why the two elementwise numbers are still printed.
//
// The whole layer is enqueued on one stream with no host synchronisation until
// the end, which is how it would run in a real decode step. The taps are read
// back afterwards from the buffers that still hold them; the values that do not
// survive - pre-norm Q and K, since q_norm writes elsewhere but RoPE is in
// place - are copied device-to-device before they are overwritten.
//
// The data lives in csrc/tests/data/real_qwen_layer3 and is tracked with Git
// LFS. A tree where LFS was never fetched has pointer files there instead of
// dumps, so the loader recognises that case and skips with the command to fix
// it rather than failing on a size mismatch.

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

#ifndef VLLM_ASCEND_REAL_QWEN_LAYER3_DIR
#define VLLM_ASCEND_REAL_QWEN_LAYER3_DIR ""
#endif

namespace vllm_ascend {
namespace test {
namespace {

// --- layer 3 configuration, mirroring the dump script's constants -----------

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

// The decode position and the context it reads. Both come from meta.json in the
// dump directory; changing --pos there means changing these two lines.
constexpr int64_t kDecodePosition = 64;
constexpr int64_t kContextLen = kDecodePosition + 1;  // 65
constexpr int64_t kPastLen = kDecodePosition;         // 64 cached positions

// Paging. 65 positions at block_size 64 need two blocks, and the two physical
// blocks are neither zero nor adjacent so a block table that is ignored, or one
// that is read as an identity mapping, produces zeros rather than passing by
// luck.
constexpr int64_t kBlockSize = 64;
constexpr int64_t kNumBlocks = 8;
constexpr int64_t kMaxBlocksPerSeq = 2;
constexpr int32_t kPhysicalBlocks[kMaxBlocksPerSeq] = {5, 2};

// The quality gate. 1 - cos is quadratic in the angle between the tensors, so
// this bounds the component pointing the wrong way at roughly 1.4e-2 relative.
constexpr double kCosineSimilarityFloor = 0.9999;

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

std::string DumpDir() {
  const std::string env = EnvOrEmpty("QWEN_REAL_LAYER3_DIR");
  if (!env.empty()) {
    return env;
  }
  return VLLM_ASCEND_REAL_QWEN_LAYER3_DIR;
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
struct Dump {
  std::vector<float> input_x;
  std::vector<float> input_norm_gamma, post_attn_norm_gamma;
  std::vector<float> q_norm_gamma, k_norm_gamma;
  std::vector<float> w_q, w_gate_attn, w_k, w_v, w_out;
  std::vector<float> w_gate, w_up, w_down;
  std::vector<float> cos_tab, sin_tab;
  std::vector<float> k_cache, v_cache;
  std::vector<float> tap_norm1, tap_qkv, tap_qk_norm, tap_attn_gate;
  std::vector<float> tap_rope_q, tap_rope_k, tap_attn_ctx, tap_attn_out;
  std::vector<float> tap_norm2, tap_swiglu;
  std::vector<float> golden_output;
};

// What the device produced, read back after the layer has run.
struct Stages {
  std::vector<float> norm1;
  std::vector<float> qkv;      // pre-norm q | k | v, concatenated to match the dump
  std::vector<float> qk_norm;  // post q_norm q | post k_norm k
  std::vector<float> attn_gate;
  std::vector<float> rope_q;
  std::vector<float> rope_k;
  std::vector<float> attn_ctx;
  std::vector<float> attn_out;
  std::vector<float> norm2;
  std::vector<float> swiglu;
  std::vector<float> output;
};

bool LoadDump(Dump* dump, std::string* error) {
  const std::string dir = DumpDir();
  if (dir.empty()) {
    *error = "VLLM_ASCEND_REAL_QWEN_LAYER3_DIR was not defined at compile time";
    return false;
  }

  struct Entry {
    const char* file;
    size_t elements;
    std::vector<float>* target;
  };

  const Entry entries[] = {
      {"input_x.bin", static_cast<size_t>(kTokens * kHidden), &dump->input_x},
      {"input_norm_gamma.bin", static_cast<size_t>(kHidden), &dump->input_norm_gamma},
      {"post_attn_norm_gamma.bin", static_cast<size_t>(kHidden), &dump->post_attn_norm_gamma},
      {"q_norm_gamma.bin", static_cast<size_t>(kHeadDim), &dump->q_norm_gamma},
      {"k_norm_gamma.bin", static_cast<size_t>(kHeadDim), &dump->k_norm_gamma},
      {"w_q.bin", static_cast<size_t>(kQDim * kHidden), &dump->w_q},
      {"w_gate_attn.bin", static_cast<size_t>(kQDim * kHidden), &dump->w_gate_attn},
      {"w_k.bin", static_cast<size_t>(kKvDim * kHidden), &dump->w_k},
      {"w_v.bin", static_cast<size_t>(kKvDim * kHidden), &dump->w_v},
      {"w_out.bin", static_cast<size_t>(kHidden * kQDim), &dump->w_out},
      {"w_gate.bin", static_cast<size_t>(kIntermediate * kHidden), &dump->w_gate},
      {"w_up.bin", static_cast<size_t>(kIntermediate * kHidden), &dump->w_up},
      {"w_down.bin", static_cast<size_t>(kHidden * kIntermediate), &dump->w_down},
      {"cos_tab_d64.bin", static_cast<size_t>(kTokens * kRotaryDim), &dump->cos_tab},
      {"sin_tab_d64.bin", static_cast<size_t>(kTokens * kRotaryDim), &dump->sin_tab},
      {"k_cache.bin", static_cast<size_t>(kPastLen * kKvDim), &dump->k_cache},
      {"v_cache.bin", static_cast<size_t>(kPastLen * kKvDim), &dump->v_cache},
      {"tap_norm1.bin", static_cast<size_t>(kTokens * kHidden), &dump->tap_norm1},
      {"tap_qkv.bin", static_cast<size_t>(kTokens * (kQDim + 2 * kKvDim)), &dump->tap_qkv},
      {"tap_qk_norm.bin", static_cast<size_t>(kTokens * (kQDim + kKvDim)), &dump->tap_qk_norm},
      {"tap_attn_gate.bin", static_cast<size_t>(kTokens * kQDim), &dump->tap_attn_gate},
      {"tap_rope_q.bin", static_cast<size_t>(kTokens * kQDim), &dump->tap_rope_q},
      {"tap_rope_k.bin", static_cast<size_t>(kTokens * kKvDim), &dump->tap_rope_k},
      {"tap_attn_ctx.bin", static_cast<size_t>(kTokens * kQDim), &dump->tap_attn_ctx},
      {"tap_attn_out.bin", static_cast<size_t>(kTokens * kHidden), &dump->tap_attn_out},
      {"tap_norm2.bin", static_cast<size_t>(kTokens * kHidden), &dump->tap_norm2},
      {"tap_swiglu.bin", static_cast<size_t>(kTokens * kIntermediate), &dump->tap_swiglu},
      {"golden_output.bin", static_cast<size_t>(kTokens * kHidden), &dump->golden_output},
  };

  for (const Entry& entry : entries) {
    if (!ReadHalfFile(dir + "/" + entry.file, entry.elements, entry.target, error)) {
      return false;
    }
  }
  return true;
}

// Logical position p lives at block_table[p / block_size], offset p % block_size.
int32_t SlotForPosition(int64_t position) {
  const int32_t block = kPhysicalBlocks[position / kBlockSize];
  return block * static_cast<int32_t>(kBlockSize) + static_cast<int32_t>(position % kBlockSize);
}

// Runs the whole layer on one stream, then reads the taps back.
Stages RunLayerOnDevice(const Dump& dump) {
  CUDADevice& device = CUDATestEnvironment::Instance().device();
  cudaStream_t stream = device.stream();
  cublasHandle_t blas = device.cublas_handle();

  // --- resident state -------------------------------------------------------
  // x is the residual stream and is updated in place twice.
  CudaDeviceTensor x = CudaDeviceTensor::Half({kTokens, kHidden}, dump.input_x);
  CudaDeviceTensor gamma1 = CudaDeviceTensor::Half({kHidden}, dump.input_norm_gamma);
  CudaDeviceTensor gamma2 = CudaDeviceTensor::Half({kHidden}, dump.post_attn_norm_gamma);
  CudaDeviceTensor q_norm_gamma = CudaDeviceTensor::Half({kHeadDim}, dump.q_norm_gamma);
  CudaDeviceTensor k_norm_gamma = CudaDeviceTensor::Half({kHeadDim}, dump.k_norm_gamma);

  // Linear weights keep the torch [out_features, in_features] layout, which is
  // exactly the transpose_b=true case of CublasGemmFp16.
  CudaDeviceTensor w_q = CudaDeviceTensor::Half({kQDim, kHidden}, dump.w_q);
  CudaDeviceTensor w_gate_attn = CudaDeviceTensor::Half({kQDim, kHidden}, dump.w_gate_attn);
  CudaDeviceTensor w_k = CudaDeviceTensor::Half({kKvDim, kHidden}, dump.w_k);
  CudaDeviceTensor w_v = CudaDeviceTensor::Half({kKvDim, kHidden}, dump.w_v);
  CudaDeviceTensor w_out = CudaDeviceTensor::Half({kHidden, kQDim}, dump.w_out);
  CudaDeviceTensor w_gate = CudaDeviceTensor::Half({kIntermediate, kHidden}, dump.w_gate);
  CudaDeviceTensor w_up = CudaDeviceTensor::Half({kIntermediate, kHidden}, dump.w_up);
  CudaDeviceTensor w_down = CudaDeviceTensor::Half({kHidden, kIntermediate}, dump.w_down);

  CudaDeviceTensor cos_tab = CudaDeviceTensor::Half({kTokens, kRotaryDim}, dump.cos_tab);
  CudaDeviceTensor sin_tab = CudaDeviceTensor::Half({kTokens, kRotaryDim}, dump.sin_tab);

  // --- scratch --------------------------------------------------------------
  CudaDeviceTensor norm1 = CudaDeviceTensor::HalfEmpty({kTokens, kHidden});
  CudaDeviceTensor q = CudaDeviceTensor::HalfEmpty({kTokens, kQDim});
  CudaDeviceTensor k = CudaDeviceTensor::HalfEmpty({kTokens, kKvDim});
  CudaDeviceTensor v = CudaDeviceTensor::HalfEmpty({kTokens, kKvDim});
  CudaDeviceTensor attn_gate = CudaDeviceTensor::HalfEmpty({kTokens, kQDim});
  CudaDeviceTensor q_normed = CudaDeviceTensor::HalfEmpty({kTokens, kQDim});
  CudaDeviceTensor k_normed = CudaDeviceTensor::HalfEmpty({kTokens, kKvDim});
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

  // The prefill's keys and values, staged in the dense [tokens, kv_heads,
  // head_dim] layout the scatter launcher reads. Writing them through the same
  // kernel the decode step uses, rather than memcpy-ing the paged layout by
  // hand, means the block mapping under test is the one the history went in
  // through - a slot arithmetic bug cannot cancel itself out.
  CudaDeviceTensor past_k = CudaDeviceTensor::Half({kPastLen, kNumKvHeads, kHeadDim}, dump.k_cache);
  CudaDeviceTensor past_v = CudaDeviceTensor::Half({kPastLen, kNumKvHeads, kHeadDim}, dump.v_cache);

  std::vector<int32_t> past_slots_host(static_cast<size_t>(kPastLen));
  for (int64_t position = 0; position < kPastLen; ++position) {
    past_slots_host[static_cast<size_t>(position)] = SlotForPosition(position);
  }
  CudaDeviceTensor past_slots = CudaDeviceTensor::Int32({kPastLen}, past_slots_host);

  const std::vector<int32_t> slot_host{SlotForPosition(kDecodePosition)};
  const std::vector<int32_t> block_table_host(kPhysicalBlocks, kPhysicalBlocks + kMaxBlocksPerSeq);
  const std::vector<int32_t> context_lens_host{static_cast<int32_t>(kContextLen)};
  CudaDeviceTensor slot_mapping = CudaDeviceTensor::Int32({kTokens}, slot_host);
  CudaDeviceTensor block_table = CudaDeviceTensor::Int32({kTokens, kMaxBlocksPerSeq}, block_table_host);
  CudaDeviceTensor context_lens = CudaDeviceTensor::Int32({kTokens}, context_lens_host);

  // --- 0. the prefill's KV cache --------------------------------------------
  cuda::LaunchPagedCacheScatterHalf(past_k.half_data(), past_v.half_data(), past_slots.int32_data(),
                                    key_cache.half_data(), value_cache.half_data(), kPastLen,
                                    kNumKvHeads, kHeadDim, kBlockSize, stream);

  // --- 1. input RMSNorm -----------------------------------------------------
  cuda::LaunchRmsNormHalf(x.half_data(), gamma1.half_data(), norm1.half_data(), nullptr, kTokens,
                          kHidden, kRmsNormEps, stream);

  // --- 2. Q / K / V and the attention gate ----------------------------------
  // w_q and w_gate_attn are the two halves of the checkpoint's fused q_proj,
  // de-interleaved by the dump script; here they are just two projections.
  CublasGemmFp16(blas, false, true, kTokens, kQDim, kHidden, norm1.half_data(), w_q.half_data(),
                 q.half_data());
  CublasGemmFp16(blas, false, true, kTokens, kKvDim, kHidden, norm1.half_data(), w_k.half_data(),
                 k.half_data());
  CublasGemmFp16(blas, false, true, kTokens, kKvDim, kHidden, norm1.half_data(), w_v.half_data(),
                 v.half_data());
  CublasGemmFp16(blas, false, true, kTokens, kQDim, kHidden, norm1.half_data(),
                 w_gate_attn.half_data(), attn_gate.half_data());

  // --- 3. per-head Q/K RMSNorm ----------------------------------------------
  // q is [tokens, heads, head_dim] contiguous, so treating the head axis as the
  // token axis gives the kernel exactly the rows q_norm normalises over. gamma
  // is head_dim wide and shared by every head, which is what the checkpoint
  // stores.
  cuda::LaunchRmsNormHalf(q.half_data(), q_norm_gamma.half_data(), q_normed.half_data(), nullptr,
                          kTokens * kNumHeads, kHeadDim, kRmsNormEps, stream);
  cuda::LaunchRmsNormHalf(k.half_data(), k_norm_gamma.half_data(), k_normed.half_data(), nullptr,
                          kTokens * kNumKvHeads, kHeadDim, kRmsNormEps, stream);

  // RoPE is in place, so keep a copy of what tap_qk_norm is supposed to hold.
  CUDA_CHECK(cudaMemcpyAsync(q_pre_rope.data(), q_normed.data(),
                             static_cast<size_t>(kTokens * kQDim) * 2, cudaMemcpyDeviceToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(k_pre_rope.data(), k_normed.data(),
                             static_cast<size_t>(kTokens * kKvDim) * 2, cudaMemcpyDeviceToDevice,
                             stream));

  // --- 4. partial RoPE ------------------------------------------------------
  // rotary_dim < head_dim, so the launcher rotates channels [0, 64) of every
  // head and leaves [64, 256) untouched, which is what partial_rotary_factor
  // 0.25 means. The tables carry the angle for position 64 only: the cached
  // keys were rotated at their own positions before they were dumped, exactly
  // as a real prefill leaves them.
  cuda::LaunchApplyRotaryPosEmbHalfMode(q_normed.half_data(), cos_tab.half_data(),
                                        sin_tab.half_data(), kTokens, kNumHeads, kHeadDim,
                                        kRotaryDim, stream);
  cuda::LaunchApplyRotaryPosEmbHalfMode(k_normed.half_data(), cos_tab.half_data(),
                                        sin_tab.half_data(), kTokens, kNumKvHeads, kHeadDim,
                                        kRotaryDim, stream);

  // --- 5. paged KV write and decode -----------------------------------------
  cuda::LaunchPagedCacheScatterHalf(k_normed.half_data(), v.half_data(), slot_mapping.int32_data(),
                                    key_cache.half_data(), value_cache.half_data(), kTokens,
                                    kNumKvHeads, kHeadDim, kBlockSize, stream);
  cuda::LaunchPagedAttentionDecodeV1Half(
      q_normed.half_data(), key_cache.half_data(), value_cache.half_data(), block_table.int32_data(),
      context_lens.int32_data(), context.half_data(), kTokens, kNumHeads, kNumKvHeads, kHeadDim,
      kBlockSize, kMaxBlocksPerSeq, kContextLen, kAttentionScale, stream);

  // --- 6. output gate, out projection, residual -----------------------------
  cuda::LaunchSigmoidGateHalf(context.half_data(), attn_gate.half_data(), gated.half_data(), kQDim,
                              stream);
  CublasGemmFp16(blas, false, true, kTokens, kHidden, kQDim, gated.half_data(), w_out.half_data(),
                 attn_out.half_data());
  cuda::LaunchResidualAddHalf(x.half_data(), attn_out.half_data(), kHidden, stream);

  // --- 7. post-attention RMSNorm --------------------------------------------
  cuda::LaunchRmsNormHalf(x.half_data(), gamma2.half_data(), norm2.half_data(), nullptr, kTokens,
                          kHidden, kRmsNormEps, stream);

  // --- 8. SwiGLU MLP --------------------------------------------------------
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

  // --- 9. final residual ----------------------------------------------------
  cuda::LaunchResidualAddHalf(x.half_data(), mlp_out.half_data(), kHidden, stream);

  device.SynchronizeStream();

  // --- read the taps back ---------------------------------------------------
  Stages stages;
  stages.norm1 = norm1.ToFloatFromHalf();
  stages.attn_gate = attn_gate.ToFloatFromHalf();
  stages.rope_q = q_normed.ToFloatFromHalf();
  stages.rope_k = k_normed.ToFloatFromHalf();
  stages.attn_ctx = context.ToFloatFromHalf();
  stages.attn_out = attn_out.ToFloatFromHalf();
  stages.norm2 = norm2.ToFloatFromHalf();
  stages.swiglu = swiglu.ToFloatFromHalf();
  stages.output = x.ToFloatFromHalf();

  const std::vector<float> q_host = q.ToFloatFromHalf();
  const std::vector<float> k_host = k.ToFloatFromHalf();
  const std::vector<float> v_host = v.ToFloatFromHalf();
  stages.qkv.reserve(q_host.size() + k_host.size() + v_host.size());
  stages.qkv.insert(stages.qkv.end(), q_host.begin(), q_host.end());
  stages.qkv.insert(stages.qkv.end(), k_host.begin(), k_host.end());
  stages.qkv.insert(stages.qkv.end(), v_host.begin(), v_host.end());

  const std::vector<float> q_normed_host = q_pre_rope.ToFloatFromHalf();
  const std::vector<float> k_normed_host = k_pre_rope.ToFloatFromHalf();
  stages.qk_norm.reserve(q_normed_host.size() + k_normed_host.size());
  stages.qk_norm.insert(stages.qk_norm.end(), q_normed_host.begin(), q_normed_host.end());
  stages.qk_norm.insert(stages.qk_norm.end(), k_normed_host.begin(), k_normed_host.end());

  return stages;
}

// The layer runs once for the whole suite; each stage then gets its own test so
// a failure names the stage rather than the whole pipeline.
class QwenRealInferenceTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!CUDATestEnvironment::Instance().available()) {
      unavailable_ = "No usable CUDA device: " + CUDATestEnvironment::Instance().unavailable_reason();
      return;
    }
    std::string error;
    if (!LoadDump(&dump_, &error)) {
      unavailable_ = "real-weight dump unavailable: " + error;
      return;
    }
    stages_ = RunLayerOnDevice(dump_);
  }

  static void TearDownTestSuite() {
    dump_ = Dump();
    stages_ = Stages();
    unavailable_.clear();
  }

  void SetUp() override {
    if (!unavailable_.empty()) {
      GTEST_SKIP() << unavailable_;
    }
  }

  static Dump dump_;
  static Stages stages_;
  static std::string unavailable_;
};

Dump QwenRealInferenceTest::dump_;
Stages QwenRealInferenceTest::stages_;
std::string QwenRealInferenceTest::unavailable_;

TEST_F(QwenRealInferenceTest, Stage1InputRmsNorm) {
  EXPECT_TENSORS_COSINE_SIMILAR(stages_.norm1, dump_.tap_norm1, kCosineSimilarityFloor,
                                "1. input RMSNorm");
}

TEST_F(QwenRealInferenceTest, Stage2QkvProjections) {
  EXPECT_TENSORS_COSINE_SIMILAR(stages_.qkv, dump_.tap_qkv, kCosineSimilarityFloor,
                                "2. Q/K/V projections");
}

TEST_F(QwenRealInferenceTest, Stage2AttentionGateProjection) {
  // The other half of the fused q_proj. Checked on its own because a wrong
  // de-interleave produces a gate that is a valid-looking projection of the
  // same input, and the sigmoid downstream squashes the difference into
  // something the final output alone would report only as a small drift.
  EXPECT_TENSORS_COSINE_SIMILAR(stages_.attn_gate, dump_.tap_attn_gate, kCosineSimilarityFloor,
                                "2. attention output gate projection");
}

TEST_F(QwenRealInferenceTest, Stage3PerHeadQkNorm) {
  EXPECT_TENSORS_COSINE_SIMILAR(stages_.qk_norm, dump_.tap_qk_norm, kCosineSimilarityFloor,
                                "3. per-head q_norm / k_norm");
}

TEST_F(QwenRealInferenceTest, Stage4PartialRotaryQ) {
  EXPECT_TENSORS_COSINE_SIMILAR(stages_.rope_q, dump_.tap_rope_q, kCosineSimilarityFloor,
                                "4. partial RoPE, Q");
}

TEST_F(QwenRealInferenceTest, Stage4PartialRotaryK) {
  EXPECT_TENSORS_COSINE_SIMILAR(stages_.rope_k, dump_.tap_rope_k, kCosineSimilarityFloor,
                                "4. partial RoPE, K");
}

TEST_F(QwenRealInferenceTest, Stage4RotatesTheRotarySliceAtThisPosition) {
  // At pos=64 the angle is real - cos[0] ~ 0.392, sin[0] ~ 0.920 - so the
  // rotary slice must change and the pass-through slice must not. Both halves
  // of that statement are asserted: the first is what pos=0 could never check,
  // the second is what catches a kernel that rotated the whole head.
  ASSERT_EQ(stages_.rope_q.size(), stages_.qk_norm.size() - static_cast<size_t>(kKvDim));

  size_t rotated_channels = 0;
  for (int64_t head = 0; head < kNumHeads; ++head) {
    for (int64_t dim = 0; dim < kHeadDim; ++dim) {
      const size_t index = static_cast<size_t>(head * kHeadDim + dim);
      if (dim < kRotaryDim) {
        if (stages_.rope_q[index] != stages_.qk_norm[index]) {
          ++rotated_channels;
        }
      } else {
        ASSERT_EQ(stages_.rope_q[index], stages_.qk_norm[index])
            << "head " << head << " channel " << dim << " was modified past rotary_dim";
      }
    }
  }
  EXPECT_GT(rotated_channels, static_cast<size_t>(kNumHeads * kRotaryDim / 2))
      << "RoPE left most of the rotary slice unchanged at position " << kDecodePosition
      << "; the tables are probably not the ones the dump used";
}

TEST_F(QwenRealInferenceTest, Stage5PagedDecodeAttention) {
  // The stage the synthetic pos=0 / ctx_len=1 dump cannot exercise: 65 context
  // positions, so the softmax is a real distribution and the block table is
  // walked across two non-adjacent physical blocks.
  EXPECT_TENSORS_COSINE_SIMILAR(stages_.attn_ctx, dump_.tap_attn_ctx, kCosineSimilarityFloor,
                                "5. paged decode attention over 65 positions");
}

TEST_F(QwenRealInferenceTest, Stage6AttentionGateAndOutProjection) {
  EXPECT_TENSORS_COSINE_SIMILAR(stages_.attn_out, dump_.tap_attn_out, kCosineSimilarityFloor,
                                "6. sigmoid gate and out projection");
}

TEST_F(QwenRealInferenceTest, Stage7PostAttentionRmsNorm) {
  EXPECT_TENSORS_COSINE_SIMILAR(stages_.norm2, dump_.tap_norm2, kCosineSimilarityFloor,
                                "7. post-attention RMSNorm");
}

TEST_F(QwenRealInferenceTest, Stage8SwiGlu) {
  EXPECT_TENSORS_COSINE_SIMILAR(stages_.swiglu, dump_.tap_swiglu, kCosineSimilarityFloor,
                                "8. SwiGLU MLP");
}

TEST_F(QwenRealInferenceTest, Stage9LayerOutputMatchesGolden) {
  EXPECT_TENSORS_COSINE_SIMILAR(stages_.output, dump_.golden_output, kCosineSimilarityFloor,
                                "9. layer output");
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
