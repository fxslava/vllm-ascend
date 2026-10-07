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

// The aclnn operator surface of libopapi_mock.
//
// Every GetWorkspaceSize stub is a CONTRACT VALIDATOR: it asserts the shapes,
// dtypes and scalar flags the DeepSeek-V4 Flash decode graph must produce,
// against the constants pinned by the checkpoint's config.json
// (F:\AI\models\DeepSeek-V4-Flash) and the operator semantics in
// F:\ops-transformer:
//
//   * moe_gating_top_k: y = [rows, k], expertIdx = [rows, k] (infershape)
//   * moe_init_routing_v4: the expert-token cumsum/count output is DT_INT64
//   * grouped_matmul: splitItem=3 (single output), groupType=0 (M groups)
//
// Execution stubs are NO-OPS returning ACL_SUCCESS: no buffer is touched, so
// the mock validates the graph's form, not its numerics. Numerical oracles
// live in tools/dsv4_moe_runtime (host references) and on the device.

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"  // aclrtStream for the launch stubs

#include "mock_acl_tensor.hpp"
#include "mock_allocator.hpp"

// The dsv4 headers are NOT included here: this library is standalone, so the
// contract constants below are the mock's own transcription of
// DeepSeek-V4-Flash config.json. A constant that drifts from the checkpoint
// is a bug in this file, and the test suite cross-checks the C++ side
// separately.
namespace vllm_ascend {
namespace dsv4 {
namespace mock {
namespace {

// DeepSeek-V4 Flash topology (config.json).
constexpr int64_t kExpertNum = 256;         // n_routed_experts
constexpr int64_t kTopK = 6;                // num_experts_per_tok
constexpr int64_t kHidden = 4096;           // hidden_size
constexpr int64_t kIntermediate = 2048;     // moe_intermediate_size
constexpr double kRoutedScaling = 1.5;      // routed_scaling_factor
constexpr int64_t kNormTypePreNormalized = -1;  // sqrtsoftplus runs decomposed
constexpr int64_t kRenormL1 = 1;                // norm_topk_prob = true
constexpr int64_t kNumHeads = 64;           // num_attention_heads
constexpr int64_t kRoutedScaleBlock = 32;   // OCP microscale block

constexpr uint64_t kWorkspaceElementwise = 32u << 10;
constexpr uint64_t kWorkspaceNorm = 16u << 10;
constexpr uint64_t kWorkspaceGating = 64u << 10;
constexpr uint64_t kWorkspaceRouting = 64u << 10;
constexpr uint64_t kWorkspaceAttention = 1u << 20;
constexpr uint64_t kWorkspaceAttentionMax = 8u << 20;
constexpr uint64_t kWorkspaceGmmBase = 256u << 10;

uint64_t Align4k(uint64_t bytes) { return (bytes + 4095) & ~4095ull; }

// Captures tensors into a fresh executor in IR order. Null tensor arguments
// still occupy their slot (ACLNN numbers optional tensors that were bound as
// null out of the IR; a later aclSetTensorAddr on such a slot is a product
// bug the strict path reports). List arguments pass their list handle: the
// slot holds it, and aclSetDynamicTensorAddr works through the caller's own
// list handle, so only the plain-index tally reads these.
aclOpExecutor* NewExecutor(const char* name, std::initializer_list<const void*> tensors) {
  auto* executor = new MockAclOpExecutor();
  executor->op_name = name;
  executor->tensors.reserve(tensors.size());
  for (const void* tensor : tensors) {
    executor->tensors.push_back(reinterpret_cast<aclTensor*>(const_cast<void*>(tensor)));
  }
  return reinterpret_cast<aclOpExecutor*>(executor);
}

// -- validator helpers -------------------------------------------------------

#define MOCK_REQUIRE(condition, message)      \
  do {                                        \
    if (!(condition)) {                       \
      return MockContractFailure(message);    \
    }                                         \
  } while (false)

bool Is2D(const MockAclTensor* tensor) { return tensor->shape.size() == 2; }

bool IsFloat(const MockAclTensor* tensor) {
  return tensor->dtype == ACL_FLOAT32 || tensor->dtype == ACL_BF16 || tensor->dtype == ACL_FLOAT16;
}

std::string ShapeOf(const MockAclTensor* tensor) {
  std::ostringstream text;
  text << "[";
  for (size_t index = 0; index < tensor->shape.size(); ++index) {
    if (index != 0) {
      text << ", ";
    }
    text << tensor->shape[index];
  }
  text << "]";
  return text.str();
}

// The five DeepSeek-V4 Flash contract validators -----------------------------

// aclnnSoftplus / aclnnSqrt: scores over [tokens, 256], float dtype.
aclnnStatus ValidateScoringStage(const char* name, const aclTensor* self, const aclTensor* out,
                                 uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* x = AsMockTensor(self);
  const MockAclTensor* y = AsMockTensor(out);
  MOCK_REQUIRE(x != nullptr && y != nullptr, std::string(name) + ": bad tensor handle");
  MOCK_REQUIRE(Is2D(x) && x->dim(1) == kExpertNum,
               std::string(name) + ": x must be [tokens, 256], got " + ShapeOf(x));
  MOCK_REQUIRE(IsFloat(x), std::string(name) + ": x must be FP32/BF16/FP16");
  MOCK_REQUIRE(y->same_shape_as(*x), std::string(name) + ": out must match x shape " + ShapeOf(x) +
                                          ", got " + ShapeOf(y));
  MOCK_REQUIRE(y->dtype == x->dtype, std::string(name) + ": out dtype must equal x dtype");
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor(name, {self, out});
  return 0;
}

}  // namespace

// exposed for the test
aclnnStatus MockValidateGatingForTest(const aclTensor* x, const aclTensor* bias_optional, int64_t k, int64_t k_group,
                                      int64_t group_count, int64_t group_select_mode, int64_t renorm,
                                      int64_t norm_type, double routed_scaling_factor, const aclTensor* y_out,
                                      const aclTensor* expert_idx_out);
aclnnStatus MockValidateRoutingForTest(const aclTensor* expert_idx, int64_t expert_num,
                                       const aclTensor* group_list_out);
aclnnStatus MockValidateGmmForTest(const aclTensorList* weight, const aclTensorList* scale_optional,
                                   int64_t split_item, int64_t group_type);

namespace {

// aclnnMoeGatingTopKV2 over pre-normalized sqrtsoftplus scores.
aclnnStatus ValidateGating(const aclTensor* x, const aclTensor* bias_optional, int64_t k, int64_t k_group,
                           int64_t group_count, int64_t group_select_mode, int64_t renorm, int64_t norm_type,
                           double routed_scaling_factor, const aclTensor* y_out, const aclTensor* expert_idx_out) {
  const MockAclTensor* scores = AsMockTensor(x);
  const MockAclTensor* bias = bias_optional == nullptr ? nullptr : AsMockTensor(bias_optional);
  const MockAclTensor* y = AsMockTensor(y_out);
  const MockAclTensor* idx = AsMockTensor(expert_idx_out);
  MOCK_REQUIRE(scores != nullptr && y != nullptr && idx != nullptr, "MoeGatingTopKV2: bad tensor handle");
  MOCK_REQUIRE(Is2D(scores) && scores->dim(1) == kExpertNum,
               "MoeGatingTopKV2: x must be [tokens, 256] (pre-normalized scores), got " + ShapeOf(scores));
  MOCK_REQUIRE(IsFloat(scores), "MoeGatingTopKV2: x must be FP32/BF16");
  MOCK_REQUIRE(k == kTopK, "MoeGatingTopKV2: k must be 6 (num_experts_per_tok), got " + std::to_string(k));
  MOCK_REQUIRE(group_count == 1,
               "MoeGatingTopKV2: groupCount must be 1 (noaux_tc is not group-constrained), got " +
                   std::to_string(group_count));
  MOCK_REQUIRE(norm_type == kNormTypePreNormalized,
               "MoeGatingTopKV2: normType must be -1 (scores arrive pre-normalized from the decomposed "
               "sqrtsoftplus chain), got " +
                   std::to_string(norm_type));
  MOCK_REQUIRE(renorm == kRenormL1,
               "MoeGatingTopKV2: renorm must be 1 (norm_topk_prob: top-k weights sum 1.0 before scaling), got " +
                   std::to_string(renorm));
  MOCK_REQUIRE(routed_scaling_factor == kRoutedScaling, "MoeGatingTopKV2: routedScalingFactor must be 1.5, got " +
                                                            std::to_string(routed_scaling_factor));
  const int64_t tokens = scores->dim(0);
  MOCK_REQUIRE(Is2D(y) && y->dim(0) == tokens && y->dim(1) == k,
               "MoeGatingTopKV2: yOut must be [tokens, 6], got " + ShapeOf(y));
  MOCK_REQUIRE(Is2D(idx) && idx->dim(0) == tokens && idx->dim(1) == k,
               "MoeGatingTopKV2: expertIdxOut must be [tokens, 6], got " + ShapeOf(idx));
  MOCK_REQUIRE(idx->dtype == ACL_INT32 || idx->dtype == ACL_INT64,
               "MoeGatingTopKV2: expertIdxOut must be INT32 or INT64");
  MOCK_REQUIRE(bias == nullptr || (Is2D(bias) && bias->dim(0) == kExpertNum && bias->dim(1) == 1) ||
                   (bias->shape.size() == 1 && bias->dim(0) == kExpertNum),
               "MoeGatingTopKV2: bias must cover 256 experts, got " + ShapeOf(bias));
  (void)k_group;
  (void)group_select_mode;
  return 0;
}

// aclnnMoeInitRoutingV4: the device cumsum groupList and its non-aliasing.
aclnnStatus ValidateRouting(const aclTensor* expert_idx, int64_t expert_num,
                            const aclTensor* group_list_out) {
  const MockAclTensor* idx = AsMockTensor(expert_idx);
  const MockAclTensor* cumsum = AsMockTensor(group_list_out);
  MOCK_REQUIRE(idx != nullptr && cumsum != nullptr, "MoeInitRoutingV4: bad tensor handle");
  MOCK_REQUIRE(cumsum->shape.size() == 1 && cumsum->dim(0) == expert_num,
               "MoeInitRoutingV4: groupListOut must be [expert_num] in cumsum mode (expertTokensNumType 0), got " +
                   ShapeOf(cumsum) + " for expert_num " + std::to_string(expert_num));
  MOCK_REQUIRE(cumsum->dtype == ACL_INT64, "MoeInitRoutingV4: groupListOut must be INT64 (the op infershape forces "
                                           "DT_INT64), got dtype code " +
                                               std::to_string(static_cast<int>(cumsum->dtype)));
  // Non-aliasing: the expert ids the router produced and the cumsum the
  // dispatcher writes must not share a single byte.
  if (idx->device_addr != nullptr && cumsum->device_addr != nullptr) {
    const uintptr_t idx_base = reinterpret_cast<uintptr_t>(idx->device_addr);
    const uintptr_t sum_base = reinterpret_cast<uintptr_t>(cumsum->device_addr);
    MOCK_REQUIRE(!MockPartiallyOverlaps(idx_base, idx->total_bytes, sum_base, cumsum->total_bytes),
                 "MoeInitRoutingV4: expertIdx and groupListOut alias the same bytes");
  }
  return 0;
}

// aclnnGroupedMatmulV5: the FP4/UE8M0 expert GEMM contract.
aclnnStatus ValidateGmm(const aclTensorList* weight, const aclTensorList* scale_optional, int64_t split_item,
                        int64_t group_type) {
  const MockAclTensorList* weights = AsMockTensorList(weight);
  const MockAclTensorList* scales = scale_optional == nullptr ? nullptr : AsMockTensorList(scale_optional);
  MOCK_REQUIRE(weights != nullptr, "GroupedMatmulV5: bad weight list handle");
  MOCK_REQUIRE(split_item == 3, "GroupedMatmulV5: splitItem must be 3 (one output tensor spanning all groups), got " +
                                    std::to_string(split_item));
  MOCK_REQUIRE(group_type == 0, "GroupedMatmulV5: groupType must be 0 (groups on the M/token axis), got " +
                                    std::to_string(group_type));
  MOCK_REQUIRE(!weights->items.empty(), "GroupedMatmulV5: the weight list is empty");
  for (size_t index = 0; index < weights->items.size(); ++index) {
    const MockAclTensor* w = AsMockTensor(weights->items[index]);
    MOCK_REQUIRE(w != nullptr, "GroupedMatmulV5: bad weight handle at list index " + std::to_string(index));
    MOCK_REQUIRE(w->dtype == ACL_FP4X2_E2M1,
                 "GroupedMatmulV5: weights must be ACL_FLOAT4_E2M1 (40), got dtype code " +
                     std::to_string(static_cast<int>(w->dtype)));
    MOCK_REQUIRE(w->total_bytes == static_cast<size_t>(w->dim(0)) * static_cast<size_t>((w->dim(1) + 1) / 2),
                 "GroupedMatmulV5: weight storage must be rows x cols/2 packed nibbles");
    if (scales != nullptr) {
      const MockAclTensor* s = AsMockTensor(scales->items[index]);
      MOCK_REQUIRE(s != nullptr, "GroupedMatmulV5: bad scale handle at list index " + std::to_string(index));
      MOCK_REQUIRE(s->dtype == ACL_FLOAT8_E8M0,
                   "GroupedMatmulV5: scales must be ACL_FLOAT8_E8M0 (37), got dtype code " +
                       std::to_string(static_cast<int>(s->dtype)));
      MOCK_REQUIRE(s->dim(1) == (w->dim(1) + kRoutedScaleBlock - 1) / kRoutedScaleBlock,
                   "GroupedMatmulV5: scale columns must be one E8M0 byte per block-32 of the reduction axis");
    }
  }
  return 0;
}

}  // namespace

}  // namespace mock
}  // namespace dsv4
}  // namespace vllm_ascend

// ---------------------------------------------------------------------------
// C entry points. Signatures are transcribed verbatim from
// csrc/standalone/dsv4_aclnn_v5.hpp (which transcribes the CANN 9.2.0-beta.2
// headers), because dsv4_aclnn_v5.cpp dlsym-casts these symbols.
// ---------------------------------------------------------------------------

extern "C" {

using namespace vllm_ascend::dsv4::mock;

// -- dense / backbone --------------------------------------------------------

aclnnStatus aclnnRmsNormGetWorkspaceSize(const aclTensor* x, const aclTensor* gamma, double epsilon,
                                         const aclTensor* y_out, const aclTensor* rstd_out,
                                         uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* mg = AsMockTensor(gamma);
  const MockAclTensor* my = AsMockTensor(y_out);
  MOCK_REQUIRE(mx != nullptr && mg != nullptr && my != nullptr, "RmsNorm: bad tensor handle");
  MOCK_REQUIRE(my->same_shape_as(*mx), "RmsNorm: y must match x shape");
  MOCK_REQUIRE(mg->elements() == mx->dim(mx->shape.size() - 1), "RmsNorm: gamma covers the last dimension");
  MOCK_REQUIRE(epsilon > 0.0, "RmsNorm: epsilon must be positive");
  *workspace_size = kWorkspaceNorm;
  *executor = NewExecutor("aclnnRmsNorm", {x, gamma, y_out, rstd_out});
  return 0;
}

aclnnStatus aclnnRmsNormDynamicMxQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* gamma,
                                                       const aclTensor* beta, double epsilon, int64_t scale_alg,
                                                       char* round_mode, int64_t dst_type, bool output_rstd,
                                                       aclTensor* y_out, aclTensor* mxscale_out, aclTensor* rstd_out,
                                                       uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* my = AsMockTensor(y_out);
  const MockAclTensor* ms = AsMockTensor(mxscale_out);
  MOCK_REQUIRE(mx != nullptr && my != nullptr && ms != nullptr, "RmsNormDynamicMxQuant: bad tensor handle");
  MOCK_REQUIRE(my->same_shape_as(*mx), "RmsNormDynamicMxQuant: y must match x shape");
  MOCK_REQUIRE(my->dtype == static_cast<aclDataType>(dst_type),
               "RmsNormDynamicMxQuant: y dtype must equal dstType");
  MOCK_REQUIRE(ms->shape.size() == mx->shape.size(), "RmsNormDynamicMxQuant: mxscale rank matches x");
  MOCK_REQUIRE(ms->dtype == ACL_FLOAT8_E8M0, "RmsNormDynamicMxQuant: mxscale must be E8M0");
  (void)gamma;
  (void)beta;
  (void)epsilon;
  (void)scale_alg;
  (void)round_mode;
  (void)output_rstd;
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor("aclnnRmsNormDynamicMxQuant", {x, gamma, beta, y_out, mxscale_out, rstd_out});
  return 0;
}


aclnnStatus aclnnDynamicMxQuantGetWorkspaceSize(const aclTensor* x, int64_t axis, char* round_mode_optional,
                                                int64_t dst_type, int64_t blocksize, int64_t scale_alg,
                                                const aclTensor* y_out, const aclTensor* mxscale_out,
                                                uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* my = AsMockTensor(y_out);
  const MockAclTensor* ms = AsMockTensor(mxscale_out);
  MOCK_REQUIRE(mx != nullptr && my != nullptr && ms != nullptr, "DynamicMxQuant: bad tensor handle");
  MOCK_REQUIRE(my->same_shape_as(*mx), "DynamicMxQuant: y must match x shape");
  MOCK_REQUIRE(blocksize == 32, "DynamicMxQuant: the DSV4 microscale block is 32");
  (void)axis;
  (void)round_mode_optional;
  (void)dst_type;
  (void)scale_alg;
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor("aclnnDynamicMxQuant", {x, y_out, mxscale_out});
  return 0;
}

aclnnStatus aclnnMatmulGetWorkspaceSize(const aclTensor* self, const aclTensor* mat2, aclTensor* out,
                                        int8_t cube_math_type, uint64_t* workspace_size,
                                        aclOpExecutor** executor) {
  const MockAclTensor* a = AsMockTensor(self);
  const MockAclTensor* b = AsMockTensor(mat2);
  const MockAclTensor* o = AsMockTensor(out);
  MOCK_REQUIRE(a != nullptr && b != nullptr && o != nullptr, "Matmul: bad tensor handle");
  MOCK_REQUIRE(a->shape.size() == 2 && b->shape.size() == 2 && o->shape.size() == 2,
               "Matmul: 2-D operands expected");
  // Two accepted orientations. `k_match` is torch semantics ([m,k] x [k,n]);
  // `n_match` is the [N,K] row-major weight of a linear layer (as checkpoints
  // store router / lm_head weights) consumed with an implicit transpose.
  // The DSV4 graph uses the latter for its projection matmuls and the former
  // for the routing combine; which orientation aclnnMatmul itself accepts is
  // a device bring-up item, recorded rather than guessed here.
  const bool k_match = a->dim(1) == b->dim(0);
  const bool n_match = a->dim(1) == b->dim(1);
  MOCK_REQUIRE(k_match || n_match, "Matmul: the reduction dimension must agree, got " +
                                       std::to_string(a->dim(1)) + " vs [" + std::to_string(b->dim(0)) + ", " +
                                       std::to_string(b->dim(1)) + "]");
  MOCK_REQUIRE(o->dim(0) == a->dim(0) && o->dim(1) == (k_match ? b->dim(1) : b->dim(0)),
               "Matmul: out must be [m, n]");
  (void)cube_math_type;
  *workspace_size = Align4k(static_cast<uint64_t>(a->dim(0)) * static_cast<uint64_t>(o->dim(1)) * 2) + (64u << 10);
  *executor = NewExecutor("aclnnMatmul", {self, mat2, out});
  return 0;
}

aclnnStatus aclnnQuantMatmulV5GetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1_scale,
                                               const aclTensor* x2_scale, const aclTensor* y_scale,
                                               const aclTensor* x1_offset, const aclTensor* x2_offset,
                                               const aclTensor* y_offset, const aclTensor* bias, bool transpose_x1,
                                               bool transpose_x2, int64_t group_size, aclTensor* out,
                                               uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* a = AsMockTensor(x1);
  const MockAclTensor* b = AsMockTensor(x2);
  const MockAclTensor* o = AsMockTensor(out);
  const MockAclTensor* as = AsMockTensor(x1_scale);
  const MockAclTensor* bs = AsMockTensor(x2_scale);
  MOCK_REQUIRE(a != nullptr && b != nullptr && o != nullptr, "QuantMatmulV5: bad tensor handle");
  MOCK_REQUIRE(a->shape.size() == 2 && b->shape.size() == 2 && o->shape.size() == 2,
               "QuantMatmulV5: 2-D operands expected");
  MOCK_REQUIRE(transpose_x2, "QuantMatmulV5: the DSV4 weights are [N, K] with transposeX2");
  MOCK_REQUIRE(a->dim(1) == b->dim(1), "QuantMatmulV5: reduction (K) dimensions must agree, got " +
                                           std::to_string(a->dim(1)) + " vs " + std::to_string(b->dim(1)));
  MOCK_REQUIRE(o->dim(0) == a->dim(0) && o->dim(1) == b->dim(0), "QuantMatmulV5: out must be [m, n]");
  if (as != nullptr) {
    MOCK_REQUIRE(as->dtype == ACL_FLOAT8_E8M0, "QuantMatmulV5: activation scales are E8M0");
  }
  if (bs != nullptr) {
    MOCK_REQUIRE(bs->dtype == ACL_FLOAT8_E8M0, "QuantMatmulV5: weight scales are E8M0");
  }
  (void)y_scale;
  (void)x1_offset;
  (void)x2_offset;
  (void)y_offset;
  (void)bias;
  (void)transpose_x1;
  (void)group_size;
  *workspace_size = Align4k(static_cast<uint64_t>(a->dim(0)) * static_cast<uint64_t>(b->dim(0)) * 2) + (128u << 10);
  *executor = NewExecutor("aclnnQuantMatmulV5", {x1, x2, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset,
                                                 bias, out});
  return 0;
}

aclnnStatus aclnnApplyRotaryPosEmbV2GetWorkspaceSize(aclTensor* query_ref, aclTensor* key_ref, const aclTensor* cos,
                                                     const aclTensor* sin, int64_t layout, char* rotary_mode,
                                                     uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* q = AsMockTensor(query_ref);
  const MockAclTensor* c = AsMockTensor(cos);
  const MockAclTensor* s = AsMockTensor(sin);
  MOCK_REQUIRE(q != nullptr && c != nullptr && s != nullptr, "ApplyRotaryPosEmbV2: bad tensor handle");
  MOCK_REQUIRE(c->same_shape_as(*s), "ApplyRotaryPosEmbV2: cos and sin must match");
  (void)key_ref;
  (void)layout;
  (void)rotary_mode;
  *workspace_size = kWorkspaceNorm;
  *executor = NewExecutor("aclnnApplyRotaryPosEmbV2", {query_ref, key_ref, cos, sin});
  return 0;
}

aclnnStatus aclnnScatterPaKvCacheGetWorkspaceSize(
    const aclTensor* key, aclTensor* key_cache_ref, const aclTensor* slot_mapping, const aclTensor* value,
    aclTensor* value_cache_ref, const aclTensor* compress_lens_optional, const aclTensor* compress_seq_offset_optional,
    const aclTensor* seq_lens_optional, char* cache_mode_optional, char* scatter_mode_optional,
    const aclIntArray* strides_optional, const aclIntArray* offsets_optional, uint64_t* workspace_size,
    aclOpExecutor** executor) {
  const MockAclTensor* k = AsMockTensor(key);
  const MockAclTensor* cache = AsMockTensor(key_cache_ref);
  const MockAclTensor* slots = AsMockTensor(slot_mapping);
  MOCK_REQUIRE(k != nullptr && cache != nullptr && slots != nullptr, "ScatterPaKvCache: bad tensor handle");
  MOCK_REQUIRE(slots->dtype == ACL_INT32 || slots->dtype == ACL_INT64, "ScatterPaKvCache: slot mapping is integral");
  MOCK_REQUIRE(cache->shape.size() >= 2, "ScatterPaKvCache: the cache is paged [blocks, block, ...]");
  (void)value;
  (void)value_cache_ref;
  (void)compress_lens_optional;
  (void)compress_seq_offset_optional;
  (void)seq_lens_optional;
  (void)cache_mode_optional;
  (void)scatter_mode_optional;
  (void)strides_optional;
  (void)offsets_optional;
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor("aclnnScatterPaKvCache",
                          {key, key_cache_ref, slot_mapping, value, value_cache_ref});
  return 0;
}

aclnnStatus aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
    const aclTensor* query, const aclTensorList* key, const aclTensorList* value, const aclTensor* pse_shift,
    const aclTensor* atten_mask, const aclIntArray* actual_seq_lengths, const aclIntArray* actual_seq_lengths_kv,
    const aclTensor* deq_scale1, const aclTensor* quant_scale1, const aclTensor* deq_scale2,
    const aclTensor* quant_scale2, const aclTensor* quant_offset2, const aclTensor* antiquant_scale,
    const aclTensor* antiquant_offset, const aclTensor* block_table, const aclTensor* query_padding_size,
    const aclTensor* kv_padding_size, const aclTensor* key_antiquant_scale, const aclTensor* key_antiquant_offset,
    const aclTensor* value_antiquant_scale, const aclTensor* value_antiquant_offset,
    const aclTensor* key_shared_prefix, const aclTensor* value_shared_prefix,
    const aclIntArray* actual_shared_prefix_len, const aclTensor* query_rope, const aclTensor* key_rope,
    const aclTensor* key_rope_antiquant_scale, const aclTensor* dequant_scale_query, const aclTensor* learnable_sink,
    const aclIntArray* q_start_idx, const aclIntArray* kv_start_idx, int64_t num_heads, double scale_value,
    int64_t pre_tokens, int64_t next_tokens, char* input_layout, int64_t num_key_value_heads, int64_t sparse_mode,
    int64_t inner_precise, int64_t block_size, int64_t antiquant_mode, bool softmax_lse_flag,
    int64_t key_antiquant_mode, int64_t value_antiquant_mode, int64_t query_quant_mode, int64_t pse_type,
    const aclTensor* attention_out, const aclTensor* softmax_lse, uint64_t* workspace_size,
    aclOpExecutor** executor) {
  const MockAclTensor* q = AsMockTensor(query);
  const MockAclTensorList* keys = AsMockTensorList(key);
  const MockAclTensorList* values = AsMockTensorList(value);
  const MockAclTensor* out = AsMockTensor(attention_out);
  MOCK_REQUIRE(q != nullptr && keys != nullptr && values != nullptr && out != nullptr,
               "FusedInferAttentionScoreV5: bad tensor handle");
  MOCK_REQUIRE(!keys->items.empty() && !values->items.empty(), "FusedInferAttentionScoreV5: empty key/value lists");
  MOCK_REQUIRE(num_heads == kNumHeads, "FusedInferAttentionScoreV5: 64 attention heads expected");
  MOCK_REQUIRE(scale_value > 0.0, "FusedInferAttentionScoreV5: a positive softmax scale is required");
  (void)pse_shift;
  (void)atten_mask;
  (void)actual_seq_lengths;
  (void)actual_seq_lengths_kv;
  (void)deq_scale1;
  (void)quant_scale1;
  (void)deq_scale2;
  (void)quant_scale2;
  (void)quant_offset2;
  (void)antiquant_scale;
  (void)antiquant_offset;
  (void)block_table;
  (void)query_padding_size;
  (void)kv_padding_size;
  (void)key_antiquant_scale;
  (void)key_antiquant_offset;
  (void)value_antiquant_scale;
  (void)value_antiquant_offset;
  (void)key_shared_prefix;
  (void)value_shared_prefix;
  (void)actual_shared_prefix_len;
  (void)query_rope;
  (void)key_rope;
  (void)key_rope_antiquant_scale;
  (void)dequant_scale_query;
  (void)learnable_sink;
  (void)q_start_idx;
  (void)kv_start_idx;
  (void)pre_tokens;
  (void)next_tokens;
  (void)input_layout;
  (void)num_key_value_heads;
  (void)sparse_mode;
  (void)inner_precise;
  (void)block_size;
  (void)antiquant_mode;
  (void)softmax_lse_flag;
  (void)key_antiquant_mode;
  (void)value_antiquant_mode;
  (void)query_quant_mode;
  (void)pse_type;
  (void)softmax_lse;
  *workspace_size = kWorkspaceAttention;
  // Every tensor argument is captured, nulls included, so the IR indices the
  // pipeline derives for aclSetTensorAddr (query=0, key list=1, value list=2,
  // ... keyRope late in the list) all land inside the captured range. The
  // rope index convention is a documented derive-and-verify item in the
  // product; the mock counts mismatches instead of failing on them.
  *executor = NewExecutor("aclnnFusedInferAttentionScoreV5",
                          {query, key, value, pse_shift, atten_mask, deq_scale1, quant_scale1, deq_scale2,
                           quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, block_table,
                           query_padding_size, kv_padding_size, key_antiquant_scale, key_antiquant_offset,
                           value_antiquant_scale, value_antiquant_offset, key_shared_prefix, value_shared_prefix,
                           query_rope, key_rope, key_rope_antiquant_scale, dequant_scale_query, learnable_sink,
                           attention_out, softmax_lse});
  return 0;
}

aclnnStatus aclnnFusedInferAttentionScoreV5GetMaxWorkspaceSize(const aclTensor* query, const aclTensorList* key,
                                                                const aclTensorList* value, const aclTensor* pse_shift,
                                                                const aclTensor* atten_mask,
                                                                const aclIntArray* actual_seq_lengths,
                                                                const aclIntArray* actual_seq_lengths_kv,
                                                                const aclTensor* deq_scale1,
                                                                const aclTensor* quant_scale1,
                                                                const aclTensor* deq_scale2,
                                                                const aclTensor* quant_scale2,
                                                                const aclTensor* quant_offset2,
                                                                const aclTensor* antiquant_scale,
                                                                const aclTensor* antiquant_offset,
                                                                const aclTensor* block_table,
                                                                const aclTensor* query_padding_size,
                                                                const aclTensor* kv_padding_size,
                                                                const aclTensor* key_antiquant_scale,
                                                                const aclTensor* key_antiquant_offset,
                                                                const aclTensor* value_antiquant_scale,
                                                                const aclTensor* value_antiquant_offset,
                                                                const aclTensor* key_shared_prefix,
                                                                const aclTensor* value_shared_prefix,
                                                                const aclIntArray* actual_shared_prefix_len,
                                                                const aclTensor* query_rope,
                                                                const aclTensor* key_rope,
                                                                const aclTensor* key_rope_antiquant_scale,
                                                                const aclTensor* dequant_scale_query,
                                                                const aclTensor* learnable_sink,
                                                                const aclIntArray* q_start_idx,
                                                                const aclIntArray* kv_start_idx, int64_t num_heads,
                                                                double scale_value, int64_t pre_tokens,
                                                                int64_t next_tokens, char* input_layout,
                                                                int64_t num_key_value_heads, int64_t sparse_mode,
                                                                int64_t inner_precise, int64_t block_size,
                                                                int64_t antiquant_mode, bool softmax_lse_flag,
                                                                int64_t key_antiquant_mode,
                                                                int64_t value_antiquant_mode,
                                                                int64_t query_quant_mode, int64_t pse_type,
                                                                const aclTensor* attention_out,
                                                                const aclTensor* softmax_lse,
                                                                uint64_t* workspace_size,
                                                                aclOpExecutor** executor) {
  // The workspace upper-bound helper: same contract, bigger number. The
  // pipeline only reserves the max and destroys the executor, so the capture
  // is minimal and no parameter is inspected.
  (void)key;
  (void)value;
  (void)pse_shift;
  (void)atten_mask;
  (void)actual_seq_lengths;
  (void)actual_seq_lengths_kv;
  (void)deq_scale1;
  (void)quant_scale1;
  (void)deq_scale2;
  (void)quant_scale2;
  (void)quant_offset2;
  (void)antiquant_scale;
  (void)antiquant_offset;
  (void)block_table;
  (void)query_padding_size;
  (void)kv_padding_size;
  (void)key_antiquant_scale;
  (void)key_antiquant_offset;
  (void)value_antiquant_scale;
  (void)value_antiquant_offset;
  (void)key_shared_prefix;
  (void)value_shared_prefix;
  (void)actual_shared_prefix_len;
  (void)query_rope;
  (void)key_rope;
  (void)key_rope_antiquant_scale;
  (void)dequant_scale_query;
  (void)learnable_sink;
  (void)q_start_idx;
  (void)kv_start_idx;
  (void)num_heads;
  (void)scale_value;
  (void)pre_tokens;
  (void)next_tokens;
  (void)input_layout;
  (void)num_key_value_heads;
  (void)sparse_mode;
  (void)inner_precise;
  (void)block_size;
  (void)antiquant_mode;
  (void)softmax_lse_flag;
  (void)key_antiquant_mode;
  (void)value_antiquant_mode;
  (void)query_quant_mode;
  (void)pse_type;
  (void)attention_out;
  (void)softmax_lse;
  *workspace_size = kWorkspaceAttentionMax;
  *executor = NewExecutor("aclnnFusedInferAttentionScoreV5GetMaxWorkspaceSize", {query, key, value});
  return 0;
}

// -- elementwise --------------------------------------------------------------

aclnnStatus aclnnSigmoidGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspace_size,
                                         aclOpExecutor** executor) {
  const MockAclTensor* x = AsMockTensor(self);
  const MockAclTensor* y = AsMockTensor(out);
  MOCK_REQUIRE(x != nullptr && y != nullptr && y->same_shape_as(*x), "Sigmoid: out must match x");
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor("aclnnSigmoid", {self, out});
  return 0;
}

aclnnStatus aclnnSqrtGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspace_size,
                                      aclOpExecutor** executor) {
  return ValidateScoringStage("aclnnSqrt", self, out, workspace_size, executor);
}

aclnnStatus aclnnSoftplusGetWorkspaceSize(const aclTensor* self, const aclScalar* beta, const aclScalar* threshold,
                                          aclTensor* out, uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclScalar* b = AsMockScalar(beta);
  const MockAclScalar* t = AsMockScalar(threshold);
  MOCK_REQUIRE(b != nullptr && t != nullptr, "aclnnSoftplus: beta/threshold scalars are required");
  MOCK_REQUIRE(b->as_f64() == 1.0, "aclnnSoftplus: the DSV4 sqrtsoftplus uses beta 1.0");
  MOCK_REQUIRE(t->as_f64() == 20.0, "aclnnSoftplus: the DSV4 sqrtsoftplus uses threshold 20.0");
  return ValidateScoringStage("aclnnSoftplus", self, out, workspace_size, executor);
}

aclnnStatus aclnnMulGetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                     uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* a = AsMockTensor(self);
  const MockAclTensor* b = AsMockTensor(other);
  const MockAclTensor* o = AsMockTensor(out);
  MOCK_REQUIRE(a != nullptr && b != nullptr && o != nullptr, "Mul: bad tensor handle");
  MOCK_REQUIRE(o->same_shape_as(*a) && o->same_shape_as(*b), "Mul: out must match both operands");
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor("aclnnMul", {self, other, out});
  return 0;
}

aclnnStatus aclnnInplaceAddGetWorkspaceSize(const aclTensor* self_ref, const aclTensor* other,
                                            const aclScalar* alpha, uint64_t* workspace_size,
                                            aclOpExecutor** executor) {
  const MockAclTensor* a = AsMockTensor(self_ref);
  const MockAclTensor* b = AsMockTensor(other);
  MOCK_REQUIRE(a != nullptr && b != nullptr, "InplaceAdd: bad tensor handle");
  MOCK_REQUIRE(a->same_shape_as(*b), "InplaceAdd: operands must agree in shape");
  (void)alpha;
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor("aclnnInplaceAdd", {self_ref, other});
  return 0;
}

aclnnStatus aclnnSwiGluGetWorkspaceSize(const aclTensor* x, int64_t dim, const aclTensor* out,
                                        uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* my = AsMockTensor(out);
  MOCK_REQUIRE(mx != nullptr && my != nullptr, "SwiGlu: bad tensor handle");
  MOCK_REQUIRE(mx->dim(dim < 0 ? static_cast<size_t>(dim + static_cast<int64_t>(mx->shape.size()))
                               : static_cast<size_t>(dim)) % 2 == 0,
               "SwiGlu: the fused gate/up axis must be even");
  (void)my;
  *workspace_size = kWorkspaceNorm;
  *executor = NewExecutor("aclnnSwiGlu", {x, out});
  return 0;
}

aclnnStatus aclnnArgMaxGetWorkspaceSize(const aclTensor* self, int64_t dim, bool keepdim, aclTensor* out,
                                        uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* x = AsMockTensor(self);
  const MockAclTensor* o = AsMockTensor(out);
  MOCK_REQUIRE(x != nullptr && o != nullptr, "ArgMax: bad tensor handle");
  MOCK_REQUIRE(o->elements() == 1, "ArgMax: one greedy index out");
  (void)dim;
  (void)keepdim;
  *workspace_size = kWorkspaceNorm;
  *executor = NewExecutor("aclnnArgMax", {self, out});
  return 0;
}

// -- MoE ----------------------------------------------------------------------

aclnnStatus aclnnMoeGatingTopKV2GetWorkspaceSize(const aclTensor* x, const aclTensor* bias_optional,
                                                 const aclTensor* input_ids_optional,
                                                 const aclTensor* tid2eid_optional, int64_t k, int64_t k_group,
                                                 int64_t group_count, int64_t group_select_mode, int64_t renorm,
                                                 int64_t norm_type, bool out_flag, double routed_scaling_factor,
                                                 double eps, const aclTensor* y_out, const aclTensor* expert_idx_out,
                                                 const aclTensor* out_out, uint64_t* workspace_size,
                                                 aclOpExecutor** executor) {
  const aclnnStatus status =
      ValidateGating(x, bias_optional, k, k_group, group_count, group_select_mode, renorm, norm_type,
                     routed_scaling_factor, y_out, expert_idx_out);
  if (status != 0) {
    return status;
  }
  MOCK_REQUIRE(eps > 0.0, "MoeGatingTopKV2: the renorm denominator needs a positive eps guard");
  MOCK_REQUIRE(!out_flag, "MoeGatingTopKV2: the full softmax output is not requested by the DSV4 graph");
  (void)input_ids_optional;
  (void)tid2eid_optional;
  (void)out_out;
  *workspace_size = kWorkspaceGating;
  *executor = NewExecutor("aclnnMoeGatingTopKV2", {x, bias_optional, input_ids_optional, tid2eid_optional, y_out,
                                                   expert_idx_out, out_out});
  return 0;
}

aclnnStatus aclnnMoeInitRoutingV4GetWorkspaceSize(
    const aclTensor* x, const aclTensor* expert_idx, const aclTensor* scale_optional,
    const aclTensor* offset_optional, const aclTensor* active_num_optional, const aclTensor* topk_weight_optional,
    int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode, int64_t expert_tokens_num_type,
    bool expert_tokens_num_flag, int64_t quant_mode, const aclIntArray* active_expert_range_optional,
    int64_t row_idx_type, const aclTensor* expanded_x_out, const aclTensor* expanded_row_idx_out,
    const aclTensor* expert_tokens_count_or_cumsum_out, const aclTensor* expanded_scale_out,
    const aclTensor* expanded_topk_weight_out, uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* idx = AsMockTensor(expert_idx);
  const MockAclTensor* ex = AsMockTensor(expanded_x_out);
  const MockAclTensor* cumsum = AsMockTensor(expert_tokens_count_or_cumsum_out);
  MOCK_REQUIRE(mx != nullptr && idx != nullptr && ex != nullptr && cumsum != nullptr,
               "MoeInitRoutingV4: bad tensor handle");
  const aclnnStatus contract = ValidateRouting(expert_idx, expert_num, expert_tokens_count_or_cumsum_out);
  if (contract != 0) {
    return contract;
  }
  MOCK_REQUIRE(expert_capacity == 0, "MoeInitRoutingV4: dropless dispatch (expertCapacity 0)");
  MOCK_REQUIRE(drop_pad_mode == 0, "MoeInitRoutingV4: dropPadMode 0");
  MOCK_REQUIRE(expert_tokens_num_type == 0, "MoeInitRoutingV4: cumsum token counts (type 0) feed the GMM groupList");
  MOCK_REQUIRE(expert_tokens_num_flag, "MoeInitRoutingV4: the token-count output must be requested");
  const int64_t tokens = mx->dim(0);
  const int64_t top_k = idx->shape.size() == 2 ? idx->dim(1) : 1;
  MOCK_REQUIRE(ex->shape.size() == 2 && ex->dim(0) == tokens * top_k && ex->dim(1) == mx->dim(1),
               "MoeInitRoutingV4: expandedX must be [tokens*k, hidden]");
  (void)offset_optional;
  (void)active_num_optional;
  (void)topk_weight_optional;
  (void)quant_mode;
  (void)active_expert_range_optional;
  (void)row_idx_type;
  (void)expanded_row_idx_out;
  (void)expanded_scale_out;
  (void)expanded_topk_weight_out;
  *workspace_size = kWorkspaceRouting + static_cast<uint64_t>(expert_num) * sizeof(int64_t);
  *executor = NewExecutor("aclnnMoeInitRoutingV4",
                          {x, expert_idx, scale_optional, offset_optional, active_num_optional,
                           topk_weight_optional, expanded_x_out, expanded_row_idx_out,
                           expert_tokens_count_or_cumsum_out, expanded_scale_out, expanded_topk_weight_out});
  return 0;
}

aclnnStatus aclnnGroupedMatmulV5GetWorkspaceSize(
    const aclTensorList* x, const aclTensorList* weight, const aclTensorList* bias_optional,
    const aclTensorList* scale_optional, const aclTensorList* offset_optional,
    const aclTensorList* antiquant_scale_optional, const aclTensorList* antiquant_offset_optional,
    const aclTensorList* per_token_scale_optional, const aclTensor* group_list_optional,
    const aclTensorList* activation_input_optional, const aclTensorList* activation_quant_scale_optional,
    const aclTensorList* activation_quant_offset_optional, int64_t split_item, int64_t group_type,
    int64_t group_list_type, int64_t act_type, aclIntArray* tuning_config_optional, aclTensorList* out,
    aclTensorList* activation_feature_out_optional, aclTensorList* dyn_quant_scale_out_optional,
    uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensorList* xs = AsMockTensorList(x);
  const MockAclTensor* groups = group_list_optional == nullptr ? nullptr : AsMockTensor(group_list_optional);
  MOCK_REQUIRE(xs != nullptr && !xs->items.empty(), "GroupedMatmulV5: empty x list");
  const aclnnStatus contract = ValidateGmm(weight, scale_optional, split_item, group_type);
  if (contract != 0) {
    return contract;
  }
  if (groups != nullptr) {
    MOCK_REQUIRE(groups->dtype == ACL_INT64, "GroupedMatmulV5: the groupList cumsum is INT64");
  }
  MOCK_REQUIRE(group_list_type == 0, "GroupedMatmulV5: groupListType 0 (per-expert token cumsum)");
  MOCK_REQUIRE(act_type == 0, "GroupedMatmulV5: actType NONE (the clamped SwiGLU is a separate op)");
  (void)bias_optional;
  (void)offset_optional;
  (void)antiquant_scale_optional;
  (void)antiquant_offset_optional;
  (void)per_token_scale_optional;
  (void)activation_input_optional;
  (void)activation_quant_scale_optional;
  (void)activation_quant_offset_optional;
  (void)tuning_config_optional;
  (void)activation_feature_out_optional;
  (void)dyn_quant_scale_out_optional;
  uint64_t rows = 0;
  uint64_t cols = 0;
  const MockAclTensor* x0 = AsMockTensor(xs->items[0]);
  const MockAclTensorList* outs = AsMockTensorList(out);
  const MockAclTensor* out0 = (outs != nullptr && !outs->items.empty()) ? AsMockTensor(outs->items[0]) : nullptr;
  if (x0 != nullptr && out0 != nullptr) {
    rows = static_cast<uint64_t>(x0->dim(0)) * xs->items.size();
    cols = static_cast<uint64_t>(out0->dim(1));
    // The tiling-formula stand-in: one aligned tile buffer per output matrix
    // plus the base, mirroring the shape of gmm tiling workspace formulas.
    *workspace_size = Align4k(rows * cols * 2) + kWorkspaceGmmBase;
  } else {
    *workspace_size = kWorkspaceGmmBase;
  }
  *executor = NewExecutor("aclnnGroupedMatmulV5", {x, weight, bias_optional, scale_optional, offset_optional,
                                                   antiquant_scale_optional, antiquant_offset_optional,
                                                   per_token_scale_optional, group_list_optional,
                                                   activation_input_optional, activation_quant_scale_optional,
                                                   activation_quant_offset_optional, out,
                                                   activation_feature_out_optional, dyn_quant_scale_out_optional});
  return 0;
}

aclnnStatus aclnnSwigluMxQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* group_index_optional,
                                               int64_t activate_dim, bool activate_left, int64_t swiglu_mode,
                                               double clamp_limit, double glu_alpha, double glu_bias,
                                               int64_t group_mode, int64_t axis, int64_t dst_type,
                                               char* round_mode_optional, int64_t scale_alg,
                                               double max_dtype_value, const aclTensor* y_out,
                                               const aclTensor* mxscale_out, uint64_t* workspace_size,
                                               aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* my = AsMockTensor(y_out);
  const MockAclTensor* ms = AsMockTensor(mxscale_out);
  MOCK_REQUIRE(mx != nullptr && my != nullptr && ms != nullptr, "SwigluMxQuant: bad tensor handle");
  MOCK_REQUIRE(mx->dim(1) == 2 * my->dim(1), "SwigluMxQuant: the gate/up input is twice the activation width");
  MOCK_REQUIRE(clamp_limit == 10.0, "SwigluMxQuant: the DSV4 swiglu clamp is 10.0");
  MOCK_REQUIRE(max_dtype_value == 448.0, "SwigluMxQuant: the FP8 E4M3 maximum is 448");
  (void)group_index_optional;
  (void)activate_dim;
  (void)activate_left;
  (void)swiglu_mode;
  (void)glu_alpha;
  (void)glu_bias;
  (void)group_mode;
  (void)axis;
  (void)dst_type;
  (void)round_mode_optional;
  (void)scale_alg;
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor("aclnnSwigluMxQuant", {x, group_index_optional, y_out, mxscale_out});
  return 0;
}

aclnnStatus aclnnMoeTokenUnpermuteGetWorkspaceSize(const aclTensor* permuted_tokens,
                                                   const aclTensor* sorted_indices, const aclTensor* probs_optional,
                                                   bool padded_mode, const aclIntArray* restore_shape_optional,
                                                   aclTensor* out, uint64_t* workspace_size,
                                                   aclOpExecutor** executor) {
  const MockAclTensor* t = AsMockTensor(permuted_tokens);
  const MockAclTensor* o = AsMockTensor(out);
  MOCK_REQUIRE(t != nullptr && o != nullptr, "MoeTokenUnpermute: bad tensor handle");
  (void)sorted_indices;
  (void)probs_optional;
  (void)padded_mode;
  (void)restore_shape_optional;
  *workspace_size = kWorkspaceRouting;
  *executor = NewExecutor("aclnnMoeTokenUnpermute", {permuted_tokens, sorted_indices, probs_optional, out});
  return 0;
}

aclnnStatus aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight, const aclTensorList* weight_scale,
    const aclTensorList* weight_assist_matrix, const aclTensor* bias, const aclTensor* x_scale,
    const aclTensor* smooth_scale, const aclTensor* group_list, int64_t dequant_mode, int64_t dequant_dtype,
    int64_t quant_mode, int64_t group_list_type, const aclIntArray* tuning_config_optional, aclTensor* output,
    aclTensor* output_scale, uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensorList* ws = AsMockTensorList(weight_scale);
  const MockAclTensor* ms = AsMockTensor(x_scale);
  const MockAclTensor* my = AsMockTensor(output);
  const MockAclTensor* mos = AsMockTensor(output_scale);
  const MockAclTensor* groups = group_list == nullptr ? nullptr : AsMockTensor(group_list);
  MOCK_REQUIRE(mx != nullptr && ws != nullptr && ms != nullptr && my != nullptr && mos != nullptr,
               "GroupedMatmulSwigluQuantV2: bad tensor handle");
  MOCK_REQUIRE(mx->shape.size() == 2 && my->shape.size() == 2 && mx->dim(1) == 2 * my->dim(1),
               "GroupedMatmulSwigluQuantV2: input is [tokens, hidden], output [tokens, intermediate]");
  for (size_t index = 0; index < ws->items.size(); ++index) {
    const MockAclTensor* w = AsMockTensor(ws->items[index]);
    MOCK_REQUIRE(w != nullptr && w->dtype == ACL_FLOAT8_E8M0,
                 "GroupedMatmulSwigluQuantV2: weight scales must be E8M0");
  }
  MOCK_REQUIRE(ms->dtype == ACL_FLOAT8_E8M0, "GroupedMatmulSwigluQuantV2: activation scales are E8M0");
  if (groups != nullptr) {
    MOCK_REQUIRE(groups->dtype == ACL_INT64, "GroupedMatmulSwigluQuantV2: the groupList cumsum is INT64");
  }
  MOCK_REQUIRE(dequant_mode == 1, "GroupedMatmulSwigluQuantV2: dequantMode 1 (MX block scales)");
  MOCK_REQUIRE(dequant_dtype == ACL_FLOAT8_E4M3FN, "GroupedMatmulSwigluQuantV2: dequant to FP8 E4M3");
  MOCK_REQUIRE(group_list_type == 0, "GroupedMatmulSwigluQuantV2: groupListType 0 (cumsum)");
  (void)weight;
  (void)weight_assist_matrix;
  (void)bias;
  (void)smooth_scale;
  (void)quant_mode;
  (void)tuning_config_optional;
  *workspace_size = Align4k(static_cast<uint64_t>(mx->dim(0)) * static_cast<uint64_t>(my->dim(1)) * 2) +
                    kWorkspaceGmmBase + (64u << 10);
  *executor = NewExecutor("aclnnGroupedMatmulSwigluQuantV2",
                          {x, weight, weight_scale, weight_assist_matrix, bias, x_scale, smooth_scale, group_list,
                           output, output_scale});
  return 0;
}

aclnnStatus aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize(
    const aclTensor* x1, aclTensor* x2, const aclTensor* scale_optional, const aclTensor* bias_optional,
    const aclTensor* offset_optional, const aclTensor* antiquant_scale_optional,
    const aclTensor* antiquant_offset_optional, const aclTensor* pertoken_scale_optional,
    const aclTensor* group_list_optional, const aclTensor* shared_input_optional, const aclTensor* logit_optional,
    const aclTensor* row_index_optional, int64_t dtype, float shared_input_weight, int64_t shared_input_offset,
    bool transpose_x1, bool transpose_x2, int64_t group_list_type, const aclIntArray* tuning_config_optional,
    aclTensor* out, uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* a = AsMockTensor(x1);
  const MockAclTensor* o = AsMockTensor(out);
  MOCK_REQUIRE(a != nullptr && o != nullptr, "GroupedMatmulFinalizeRoutingV3: bad tensor handle");
  (void)x2;
  (void)scale_optional;
  (void)bias_optional;
  (void)offset_optional;
  (void)antiquant_scale_optional;
  (void)antiquant_offset_optional;
  (void)pertoken_scale_optional;
  (void)group_list_optional;
  (void)shared_input_optional;
  (void)logit_optional;
  (void)row_index_optional;
  (void)dtype;
  (void)shared_input_weight;
  (void)shared_input_offset;
  (void)transpose_x1;
  (void)transpose_x2;
  (void)group_list_type;
  (void)tuning_config_optional;
  *workspace_size = kWorkspaceRouting;
  *executor = NewExecutor("aclnnGroupedMatmulFinalizeRoutingV3", {x1, x2, out});
  return 0;
}

// -- execution: every launch is a validated no-op -----------------------------

#define MOCK_NOOP_LAUNCH(name)                                                             \
  aclnnStatus name(void* workspace, uint64_t workspace_size, aclOpExecutor* executor,      \
                   aclrtStream stream) {                                                   \
    (void)workspace;                                                                       \
    (void)workspace_size;                                                                  \
    (void)executor;                                                                        \
    (void)stream;                                                                          \
    return 0;                                                                              \
  }

MOCK_NOOP_LAUNCH(aclnnRmsNorm)
MOCK_NOOP_LAUNCH(aclnnRmsNormDynamicMxQuant)
MOCK_NOOP_LAUNCH(aclnnDynamicMxQuant)
MOCK_NOOP_LAUNCH(aclnnMatmul)
MOCK_NOOP_LAUNCH(aclnnQuantMatmulV5)
MOCK_NOOP_LAUNCH(aclnnApplyRotaryPosEmbV2)
MOCK_NOOP_LAUNCH(aclnnScatterPaKvCache)
MOCK_NOOP_LAUNCH(aclnnFusedInferAttentionScoreV5)
MOCK_NOOP_LAUNCH(aclnnSigmoid)
MOCK_NOOP_LAUNCH(aclnnSqrt)
MOCK_NOOP_LAUNCH(aclnnSoftplus)
MOCK_NOOP_LAUNCH(aclnnMul)
MOCK_NOOP_LAUNCH(aclnnInplaceAdd)
MOCK_NOOP_LAUNCH(aclnnSwiGlu)
MOCK_NOOP_LAUNCH(aclnnArgMax)
MOCK_NOOP_LAUNCH(aclnnMoeGatingTopKV2)
MOCK_NOOP_LAUNCH(aclnnMoeInitRoutingV4)
MOCK_NOOP_LAUNCH(aclnnGroupedMatmulV5)
MOCK_NOOP_LAUNCH(aclnnSwigluMxQuant)
MOCK_NOOP_LAUNCH(aclnnMoeTokenUnpermute)
MOCK_NOOP_LAUNCH(aclnnGroupedMatmulSwigluQuantV2)
MOCK_NOOP_LAUNCH(aclnnGroupedMatmulFinalizeRoutingV3)

#undef MOCK_NOOP_LAUNCH

}  // extern "C"

namespace vllm_ascend {
namespace dsv4 {
namespace mock {

// Direct entry points for the test's negative cases (no executor plumbing).
aclnnStatus MockValidateGatingForTest(const aclTensor* x, const aclTensor* bias_optional, int64_t k, int64_t k_group,
                                      int64_t group_count, int64_t group_select_mode, int64_t renorm,
                                      int64_t norm_type, double routed_scaling_factor, const aclTensor* y_out,
                                      const aclTensor* expert_idx_out) {
  return ValidateGating(x, bias_optional, k, k_group, group_count, group_select_mode, renorm, norm_type,
                        routed_scaling_factor, y_out, expert_idx_out);
}

aclnnStatus MockValidateRoutingForTest(const aclTensor* expert_idx, int64_t expert_num,
                                       const aclTensor* group_list_out) {
  return ValidateRouting(expert_idx, expert_num, group_list_out);
}

aclnnStatus MockValidateGmmForTest(const aclTensorList* weight, const aclTensorList* scale_optional,
                                   int64_t split_item, int64_t group_type) {
  return ValidateGmm(weight, scale_optional, split_item, group_type);
}

}  // namespace mock
}  // namespace dsv4
}  // namespace vllm_ascend
