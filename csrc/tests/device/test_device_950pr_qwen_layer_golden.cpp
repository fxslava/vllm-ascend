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
// full_attention block) on Ascend 950PR, against the PyTorch dump in
// csrc/tests/data/golden_layer3.
//
// Every other test in this directory checks one operator against a CPU
// reference. This one checks that the operators compose: it runs a whole layer
// on the NPU - RMSNorm, four projections, partial RoPE, a paged KV write and
// decode, the attention output gate, the out projection, two residual adds,
// SwiGLU and the MLP - and compares seven intermediate taps plus the final
// output against scripts/dump_qwen35_layer3.py.
//
// It is the Ascend port of csrc/tests/kernels/cuda/test_qwen_layer_golden.cpp
// and consumes the same dump, unchanged, so a divergence between the two is a
// backend difference rather than a difference in what is being asked.
//
// Why taps rather than only the final output: a single end-to-end comparison
// tells you a layer is wrong, not where. The taps localise it to a stage, which
// is the entire reason for dumping them. They only work because the dumper
// rounds to fp16 at every stage boundary exactly as these operators do - see
// the numerics note at the top of that script.
//
// -----------------------------------------------------------------------------
// WHAT THE SHIPPED DUMP DOES AND DOES NOT TEST
// -----------------------------------------------------------------------------
//
// It is pos=0, ctx_len=1 - the decode step the requirement names - and that
// configuration has two blind spots, the second sharp enough to state plainly:
//
//   1. At position 0 the rotary tables are cos=1 / sin=0, so RoPE is the
//      identity and tap_rope_q equals the pre-RoPE Q. Stage3LeavesChannels...
//      below is the one rotary assertion that still bites here, and the real
//      rotary coverage lives in test_rotary_embedding_950pr.cpp.
//
//   2. With a single context position the softmax runs over one element and is
//      therefore exactly 1.0 whatever the score is. The attention context is
//      then just V, which means THE LAYER OUTPUT IS INDEPENDENT OF Q, K AND
//      RoPE. The CUDA port confirmed this by regenerating at pos=1024:
//      cos/sin and tap_rope_q change, golden_output.bin does not, byte for
//      byte.
//
// So at ctx_len=1 a bug in the Q projection, the K projection, RoPE or the KV
// cache write cannot reach golden_output.bin. This port turns that from a
// weakness into an assertion: Stage5DecodeAttentionContextIsTheCachedValue
// checks the attention context against V read out of tap_qkv, which is a
// property that holds *because* the softmax is 1.0 and which the CUDA version
// does not check at all. It is the only direct check on the paged KV write and
// the decode call anywhere in the suite.
//
// A CAVEAT ON REGENERATING THE DUMP. `--pos N` works: nothing here depends on
// the position. `--ctx-len M` for M > 1 does NOT, and the CUDA header's
// suggestion that it does is wrong: scripts/dump_qwen35_layer3.py generates
// k_past / v_past for the prior context but never writes them out, so a dump
// with M > 1 describes an attention over context the test has no way to
// reconstruct. Anyone extending this to a multi-position context has to teach
// the dumper to write those two tensors first.
//
// -----------------------------------------------------------------------------
// WHAT HAS AND HAS NOT BEEN RUN
// -----------------------------------------------------------------------------
//
// As committed, this file has been compiled but never executed: no Ascend 950PR
// device was available. The operator argument lists come from the CANN 9.1.0
// headers (see aclnn_ops.hpp and aclnn_ops_950pr.hpp, which name the header per
// operator) and the paged-decode configuration is copied argument for argument
// from AscendAttentionBackendImpl in vllm_ascend/attention/attention_v1.py, but
// neither has been confirmed against silicon. Stage 5 is the part to distrust
// first: it is the only stage whose layout is not a plain 2-D shape, and
// Stage5DecodeAttentionContextIsTheCachedValue is the assertion that will say
// so.

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

// aclnnSwiGlu splits the last axis, matching npu_swiglu with its default dim.
constexpr int64_t kSwiGluSplitDim = -1;

// Non-const because the operator takes char*, not const char*.
char kFiaLayout[] = "TND";
char kScatterCacheMode[] = "Norm";

// --- resolved operators -------------------------------------------------------

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

// --- small helpers ------------------------------------------------------------

// out[m, n] = a[m, k] @ w[n, k]^T, the shape every Linear in the layer has.
// `w` is uploaded in the torch [out_features, in_features] layout and described
// with strides as a [k, n] view, which is what the cube unit wants and what
// DeviceTensor::HalfTransposed2D exists for.
void Project(const DeviceTensor& a, const DeviceTensor& w_view, const DeviceTensor& out, aclrtStream stream) {
  RunAclnn<ops::MatmulWorkspaceFn>(MatmulOp(), stream, a.get(), w_view.get(), out.get(),
                                   ops::kCubeMathTypeKeepDtype);
}

// x += y, elementwise, in place. alpha is the scalar aclnnInplaceAdd multiplies
// `other` by; a residual add wants 1.
void ResidualAdd(const DeviceTensor& x_ref, const DeviceTensor& y, aclrtStream stream) {
  AclnnScalar alpha(1.0f);
  RunAclnn<ops950::InplaceAddWorkspaceFn>(InplaceAddOp(), stream, x_ref.get(), y.get(), alpha.get());
}

// --- what the device produced, read back after the layer has run --------------

struct Stages {
  std::vector<float> norm1;
  std::vector<float> qkv;  // pre-RoPE q | k | v, concatenated to match the dump
  std::vector<float> rope_q;
  std::vector<float> rope_k;
  std::vector<float> attn_context;  // the decode output, before the gate
  std::vector<float> attn_out;
  std::vector<float> norm2;
  std::vector<float> swiglu;
  std::vector<float> output;
  PartialRotaryPath rotary_path = PartialRotaryPath::kPackedApplyRotary;
};

// Every operator the pipeline needs, so a missing one becomes one skip naming
// it rather than a failure part-way through the layer.
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

// Runs the whole layer and reads the taps back.
Stages RunLayerOnDevice(const GoldenLayer3& golden) {
  aclrtStream stream = AscendTestEnvironment::Instance().stream();

  // --- resident state ---------------------------------------------------------
  // x is the residual stream and is updated in place twice.
  DeviceTensor x = DeviceTensor::Half({s::kTokens, s::kHidden}, golden.input_x);
  DeviceTensor gamma1 = DeviceTensor::Half({s::kHidden}, golden.input_norm_gamma);
  DeviceTensor gamma2 = DeviceTensor::Half({s::kHidden}, golden.post_attn_norm_gamma);

  // Linear weights keep the torch [out_features, in_features] layout.
  DeviceTensor w_q = DeviceTensor::HalfTransposed2D(s::kQDim, s::kHidden, golden.w_q);
  DeviceTensor w_k = DeviceTensor::HalfTransposed2D(s::kKvDim, s::kHidden, golden.w_k);
  DeviceTensor w_v = DeviceTensor::HalfTransposed2D(s::kKvDim, s::kHidden, golden.w_v);
  DeviceTensor w_gate_attn = DeviceTensor::HalfTransposed2D(s::kQDim, s::kHidden, golden.w_gate_attn);
  DeviceTensor w_out = DeviceTensor::HalfTransposed2D(s::kHidden, s::kQDim, golden.w_out);
  DeviceTensor w_gate = DeviceTensor::HalfTransposed2D(s::kIntermediate, s::kHidden, golden.w_gate);
  DeviceTensor w_up = DeviceTensor::HalfTransposed2D(s::kIntermediate, s::kHidden, golden.w_up);
  DeviceTensor w_down = DeviceTensor::HalfTransposed2D(s::kHidden, s::kIntermediate, golden.w_down);

  // --- scratch ----------------------------------------------------------------
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

  // The decode output in TND layout, [tokens, num_heads, head_dim].
  DeviceTensor context = DeviceTensor::HalfEmpty({s::kTokens, s::kNumHeads, s::kHeadDim});
  // Required output even with softmaxLseFlag false; the plugin allocates one
  // element for it and throws it away.
  DeviceTensor softmax_lse = DeviceTensor::HalfEmpty({1});

  // --- 1. input RMSNorm -------------------------------------------------------
  RunAclnn<ops::RmsNormWorkspaceFn>(RmsNormOp(), stream, x.get(), gamma1.get(),
                                    static_cast<double>(s::kRmsNormEps), norm1.get(), rstd.get());

  // --- 2. Q / K / V and the attention gate ------------------------------------
  Project(norm1, w_q, q, stream);
  Project(norm1, w_k, k, stream);
  Project(norm1, w_v, v, stream);
  Project(norm1, w_gate_attn, attn_gate, stream);

  // RoPE is in place, so keep a copy of what tap_qkv is supposed to hold.
  ACL_CHECK(aclrtMemcpyAsync(q_pre_rope.data(), static_cast<size_t>(s::kTokens * s::kQDim) * kHalfBytes,
                             q.data(), static_cast<size_t>(s::kTokens * s::kQDim) * kHalfBytes,
                             ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
  ACL_CHECK(aclrtMemcpyAsync(k_pre_rope.data(), static_cast<size_t>(s::kTokens * s::kKvDim) * kHalfBytes,
                             k.data(), static_cast<size_t>(s::kTokens * s::kKvDim) * kHalfBytes,
                             ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
  ACL_CHECK(aclrtSynchronizeStream(stream));

  // --- 3 and 4. partial RoPE on Q and K ---------------------------------------
  // rotary_dim 64 < head_dim 256, so channels [0, 64) of every head rotate and
  // [64, 256) pass through. See partial_rotary_950pr.hpp for which operator
  // ends up doing that and why there are two candidates.
  Stages stages;
  stages.rotary_path = ApplyPartialRotaryQK(q.data(), k.data(), golden.cos_tab, golden.sin_tab, s::kTokens,
                                            s::kNumHeads, s::kNumKvHeads, s::kHeadDim, s::kRotaryDim, stream);

  // --- 5. paged KV write and decode -------------------------------------------
  // Cache layout is AscendAttentionBackend.get_kv_cache_shape() split into its
  // two halves: [num_blocks, block_size, num_kv_heads, head_size], plain ND.
  // DeviceBuffer zeroes on allocation, so every position this decode does not
  // write reads as zero rather than as whatever the allocator handed back.
  DeviceTensor key_cache =
      DeviceTensor::HalfEmpty({s::kNumBlocks, s::kBlockSize, s::kNumKvHeads, s::kHeadDim});
  DeviceTensor value_cache =
      DeviceTensor::HalfEmpty({s::kNumBlocks, s::kBlockSize, s::kNumKvHeads, s::kHeadDim});

  // slot = block_id * block_size + offset_in_block. A non-zero physical block
  // is used so a block table that is ignored reads zeros rather than passing by
  // luck.
  const std::vector<int32_t> slot_host{s::kPhysicalBlock * static_cast<int32_t>(s::kBlockSize)};
  const std::vector<int32_t> block_table_host{s::kPhysicalBlock};
  DeviceTensor slot_mapping = DeviceTensor::Int32({s::kTokens}, slot_host);
  DeviceTensor block_table = DeviceTensor::Int32({s::kTokens, s::kMaxBlocksPerSeq}, block_table_host);

  // ScatterPaKvCache takes key and value as [num_tokens, num_kv_heads, head_size],
  // which is the same memory as the [tokens, kv_dim] buffers the projections
  // wrote, viewed with the head axis split out.
  AclnnTensor k_bnd({s::kTokens, s::kNumKvHeads, s::kHeadDim}, ACL_FLOAT16, k.data());
  AclnnTensor v_bnd({s::kTokens, s::kNumKvHeads, s::kHeadDim}, ACL_FLOAT16, v.data());

  RunAclnn<ops::ScatterPaKvCacheWorkspaceFn>(
      ScatterPaKvCacheOp(), stream, k_bnd.get(), key_cache.get(), slot_mapping.get(), v_bnd.get(),
      value_cache.get(), /*compress_lens=*/nullptr, /*compress_seq_offset=*/nullptr, /*seq_lens=*/nullptr,
      kScatterCacheMode, /*scatter_mode=*/nullptr, /*strides=*/nullptr, /*offsets=*/nullptr);

  // _get_fia_params flattens the head axes of the cache before the call.
  const std::vector<int64_t> cache_view =
      s::FiaKeyCacheView(s::kNumBlocks, s::kBlockSize, s::kNumKvHeads, s::kHeadDim);
  AclnnTensor key_cache_flat(cache_view, ACL_FLOAT16, key_cache.data());
  AclnnTensor value_cache_flat(cache_view, ACL_FLOAT16, value_cache.data());
  AclnnTensorList key_list({key_cache_flat.get()});
  AclnnTensorList value_list({value_cache_flat.get()});

  // TND: the query is [total_tokens, num_heads, head_size] and the per-sequence
  // split is carried by the cumulative query lengths rather than a batch axis.
  AclnnTensor query_tnd({s::kTokens, s::kNumHeads, s::kHeadDim}, ACL_FLOAT16, q.data());
  AclnnIntArray actual_seq_lengths(std::vector<int64_t>{s::kTokens});
  AclnnIntArray actual_seq_lengths_kv(std::vector<int64_t>{s::kContextLen});

  RunAclnn<ops950::FusedInferAttentionScoreV2WorkspaceFn>(
      FusedInferAttentionOp(), stream, query_tnd.get(), key_list.get(), value_list.get(),
      /*pse_shift=*/nullptr, /*atten_mask=*/nullptr, actual_seq_lengths.get(), actual_seq_lengths_kv.get(),
      /*deq_scale1=*/nullptr, /*quant_scale1=*/nullptr, /*deq_scale2=*/nullptr, /*quant_scale2=*/nullptr,
      /*quant_offset2=*/nullptr, /*antiquant_scale=*/nullptr, /*antiquant_offset=*/nullptr, block_table.get(),
      /*query_padding_size=*/nullptr, /*kv_padding_size=*/nullptr, /*key_antiquant_scale=*/nullptr,
      /*key_antiquant_offset=*/nullptr, /*value_antiquant_scale=*/nullptr, /*value_antiquant_offset=*/nullptr,
      /*key_shared_prefix=*/nullptr, /*value_shared_prefix=*/nullptr, /*actual_shared_prefix_len=*/nullptr,
      s::kNumHeads, static_cast<double>(s::kAttentionScale), s::kFiaUnboundedTokens, s::kFiaUnboundedTokens,
      kFiaLayout, s::kNumKvHeads, s::kFiaSparseModeNone, s::kFiaInnerPreciseDefault, s::kBlockSize,
      /*antiquant_mode=*/0, /*softmax_lse_flag=*/false, /*key_antiquant_mode=*/0, /*value_antiquant_mode=*/0,
      context.get(), softmax_lse.get());

  // --- 6. output gate, out projection, residual --------------------------------
  // gated = context * sigmoid(attn_gate), which is Qwen3.5's attn_output_gate.
  RunAclnn<ops950::SigmoidWorkspaceFn>(SigmoidOp(), stream, attn_gate.get(), gate_sigmoid.get());

  // context is [tokens, heads, head_dim] and the gate is [tokens, q_dim]; the
  // two are the same elements in the same order, so the multiply runs over a
  // flat view rather than a reshape.
  AclnnTensor context_flat({s::kTokens, s::kQDim}, ACL_FLOAT16, context.data());
  RunAclnn<ops950::MulWorkspaceFn>(MulOp(), stream, context_flat.get(), gate_sigmoid.get(), gated.get());

  Project(gated, w_out, attn_out, stream);
  ResidualAdd(x, attn_out, stream);

  // --- 7. post-attention RMSNorm ------------------------------------------------
  RunAclnn<ops::RmsNormWorkspaceFn>(RmsNormOp(), stream, x.get(), gamma2.get(),
                                    static_cast<double>(s::kRmsNormEps), norm2.get(), rstd.get());

  // --- 8. SwiGLU MLP ------------------------------------------------------------
  // aclnnSwiGlu splits one [tokens, 2 * intermediate] tensor, so the gate and up
  // projections write the two halves of a single buffer. That is only valid
  // because kTokens == 1: with more rows the halves would interleave and each
  // projection would need its own buffer plus a copy.
  static_assert(s::kTokens == 1, "gate/up share one buffer, which only works for a single row");
  // The up half starts at intermediate * 2 bytes into the buffer. 6144 * 2 is a
  // whole number of 32-byte bursts, so the second view is aligned exactly as
  // DeviceBuffer would have allocated it.
  static_assert((s::kIntermediate * 2 * static_cast<int64_t>(kHalfBytes)) % 32 == 0,
                "the up-projection view must land on a 32-byte boundary");

  DeviceBuffer gate_up(static_cast<size_t>(s::kTokens * 2 * s::kIntermediate) * kHalfBytes);
  auto* gate_up_bytes = static_cast<uint8_t*>(gate_up.get());

  // Two [1, intermediate] output views over the one buffer, so the matmuls can
  // write straight into the halves the activation expects.
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

  // --- 9. final residual --------------------------------------------------------
  ResidualAdd(x, mlp_out, stream);

  ACL_CHECK(aclrtSynchronizeStream(stream));

  // --- read the taps back -------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Host-only: the dump checked against the CPU reference
// -----------------------------------------------------------------------------
//
// The device suite below cannot run anywhere but on a 950PR, which leaves the
// three things it depends on - the loader, the dump itself and the CPU
// references the operator tests compare against - completely unexercised
// everywhere else. That is the wrong way round: those are the parts a build
// machine *can* check, and a fault in any of them would be diagnosed as a
// kernel bug on the one machine that can run the rest.
//
// So the same nine stages are run again here in float on the host, rounding to
// fp16 at every stage boundary exactly as scripts/dump_qwen35_layer3.py does,
// and compared against the same taps. A pass means the dump is internally
// consistent, the loader reads it correctly, and the CPU references agree with
// PyTorch - and it means all of that on a machine with no NPU attached.
//
// It also fixes the one thing the device suite structurally cannot: when the
// NPU comparison fails, this tells you whether to suspect the kernel or the
// reference you are comparing it against.

std::vector<float> RoundToHalf(const std::vector<float>& values) { return QuantizeToHalf(values); }

// out = a @ w^T, rounded to fp16 the way a kernel's store rounds.
std::vector<float> ProjectOnCpu(const std::vector<float>& a, const std::vector<float>& w, int64_t m, int64_t k,
                                int64_t n) {
  std::vector<float> out;
  reference::MatmulTransposedB(a, w, m, k, n, &out);
  return RoundToHalf(out);
}

// Runs the nine stages in float, rounding to fp16 at every stage boundary.
Stages RunLayerOnCpu(const GoldenLayer3& golden) {
  Stages cpu;

  // --- 1. input RMSNorm --------------------------------------------------------
  std::vector<float> norm1;
  std::vector<float> rstd;
  reference::RmsNorm(golden.input_x, golden.input_norm_gamma, s::kTokens, s::kHidden, s::kRmsNormEps, &norm1,
                     &rstd);
  cpu.norm1 = RoundToHalf(norm1);

  // --- 2. Q / K / V and the attention gate --------------------------------------
  const std::vector<float> q = ProjectOnCpu(cpu.norm1, golden.w_q, s::kTokens, s::kHidden, s::kQDim);
  const std::vector<float> k = ProjectOnCpu(cpu.norm1, golden.w_k, s::kTokens, s::kHidden, s::kKvDim);
  const std::vector<float> v = ProjectOnCpu(cpu.norm1, golden.w_v, s::kTokens, s::kHidden, s::kKvDim);
  const std::vector<float> attn_gate =
      ProjectOnCpu(cpu.norm1, golden.w_gate_attn, s::kTokens, s::kHidden, s::kQDim);

  cpu.qkv.reserve(q.size() + k.size() + v.size());
  cpu.qkv.insert(cpu.qkv.end(), q.begin(), q.end());
  cpu.qkv.insert(cpu.qkv.end(), k.begin(), k.end());
  cpu.qkv.insert(cpu.qkv.end(), v.begin(), v.end());

  // --- 3 and 4. partial RoPE ------------------------------------------------------
  std::vector<float> rope_q;
  reference::ApplyRotaryPosEmb(q, golden.cos_tab, golden.sin_tab, s::kTokens, s::kNumHeads, s::kHeadDim,
                               s::kRotaryDim, reference::RotaryMode::kHalf, &rope_q);
  cpu.rope_q = RoundToHalf(rope_q);

  std::vector<float> rope_k;
  reference::ApplyRotaryPosEmb(k, golden.cos_tab, golden.sin_tab, s::kTokens, s::kNumKvHeads, s::kHeadDim,
                               s::kRotaryDim, reference::RotaryMode::kHalf, &rope_k);
  cpu.rope_k = RoundToHalf(rope_k);

  // --- 5. decode attention ---------------------------------------------------------
  // kContextLen == 1, so the softmax is over a single score and is exactly 1.0:
  // the context is V, fetched per query head through the GQA mapping. Writing
  // it out rather than calling reference::PagedAttentionDecode is deliberate -
  // that helper models the 310P 5-D NZ cache, which is not the layout this part
  // uses, and going through it would be testing the wrong thing.
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

  // --- 6. output gate, out projection, residual --------------------------------------
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

  // --- 7. post-attention RMSNorm ------------------------------------------------------
  std::vector<float> norm2;
  reference::RmsNorm(x, golden.post_attn_norm_gamma, s::kTokens, s::kHidden, s::kRmsNormEps, &norm2, &rstd);
  cpu.norm2 = RoundToHalf(norm2);

  // --- 8. SwiGLU MLP --------------------------------------------------------------------
  const std::vector<float> mlp_gate =
      ProjectOnCpu(cpu.norm2, golden.w_gate, s::kTokens, s::kHidden, s::kIntermediate);
  const std::vector<float> mlp_up =
      ProjectOnCpu(cpu.norm2, golden.w_up, s::kTokens, s::kHidden, s::kIntermediate);

  // SiluAndMul reads gate and up as one [tokens, 2 * intermediate] row, which is
  // also the layout aclnnSwiGlu takes.
  std::vector<float> gate_up;
  gate_up.reserve(mlp_gate.size() + mlp_up.size());
  gate_up.insert(gate_up.end(), mlp_gate.begin(), mlp_gate.end());
  gate_up.insert(gate_up.end(), mlp_up.begin(), mlp_up.end());

  std::vector<float> swiglu;
  reference::SiluAndMul(gate_up, s::kTokens, s::kIntermediate, &swiglu);
  cpu.swiglu = RoundToHalf(swiglu);

  const std::vector<float> mlp_out =
      ProjectOnCpu(cpu.swiglu, golden.w_down, s::kTokens, s::kIntermediate, s::kHidden);

  // --- 9. final residual -----------------------------------------------------------------
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
  // The loader checks these too, but only as a precondition; asserting them here
  // means a dump regenerated at a different head count is reported as a shape
  // change rather than as a numerical failure nine stages later.
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
  // The dump ships at pos=0, where cos=1 and sin=0. Asserting it rather than
  // assuming it: if someone regenerates with --pos N, this test fails and says
  // why the rotary taps stopped being a copy of the projections, instead of the
  // reader wondering whether the rotary stage broke.
  for (size_t i = 0; i < golden_.cos_tab.size(); ++i) {
    EXPECT_FLOAT_EQ(golden_.cos_tab[i], 1.0f) << "cos[" << i << "]: dump is not at position 0";
    EXPECT_FLOAT_EQ(golden_.sin_tab[i], 0.0f) << "sin[" << i << "]: dump is not at position 0";
  }
}

TEST(QwenLayer3Loader, ReportsAnUnfetchedLfsPointer) {
  // The failure a fresh clone hits. The loader has to recognise it and say
  // `git lfs pull`, because the alternative message - a size mismatch in bytes -
  // sends the reader looking for a shape bug that is not there.
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
  // A file that is the wrong size but is not an LFS pointer: a dump regenerated
  // at a different width, most likely. The message has to carry both numbers or
  // it says nothing useful.
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

// -----------------------------------------------------------------------------
// Device parity
// -----------------------------------------------------------------------------

// The layer runs once for the whole suite; each stage then gets its own test so
// a failure names the stage rather than the whole pipeline.
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
  // The pass-through half of partial rotary: channels [rotary_dim, head_dim) of
  // every head must survive bit-exactly. At pos=0 the rotated channels are
  // unchanged too, so this is the one rotary assertion in this file that still
  // means something there - it would catch an operator that rotated the whole
  // 256-wide head, which is exactly what the stock rotary operator does if it
  // is handed the head instead of the slice.
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
  // The dump has no tap between the decode and the output gate, so without this
  // the paged KV write and the FIA call would only be covered through
  // tap_attn_out - and at this configuration they would barely be covered at
  // all, since a decode that returned garbage in the right shape would still
  // have to pass through the gate and o_proj before anything noticed.
  //
  // At kContextLen == 1 the softmax runs over a single score and is therefore
  // exactly 1.0 whatever that score is, so the context is just V - fetched from
  // the paged cache, through the block table, for the kv head each query head
  // maps to. That makes this an end-to-end check of the cache write, the block
  // table indexing and the GQA head mapping, none of which anything else here
  // touches.
  static_assert(s::kContextLen == 1, "this identity only holds for a single context position");
  static_assert(s::kNumHeads % s::kNumKvHeads == 0, "GQA needs a whole number of query heads per kv head");

  ASSERT_EQ(stages_.attn_context.size(), static_cast<size_t>(s::kNumHeads * s::kHeadDim));

  // V as the dump recorded it, i.e. the last kKvDim values of tap_qkv.
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
  // tap_norm2 is taken after the first residual add, so this covers stage 6's
  // add as well: a residual that landed in the wrong buffer would show up here
  // before it reached the output.
  EXPECT_TENSORS_ALLCLOSE(stages_.norm2, golden_.tap_norm2, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3Golden950PrTest, Stage8SwiGlu) {
  EXPECT_TENSORS_ALLCLOSE(stages_.swiglu, golden_.tap_swiglu, kFp16DefaultTolerance);
}

TEST_F(QwenLayer3Golden950PrTest, Stage9LayerOutputMatchesGolden) {
  EXPECT_TENSORS_ALLCLOSE(stages_.output, golden_.golden_output, kFp16DefaultTolerance);
}

}  // namespace
}  // namespace test
}  // namespace vllm_ascend
