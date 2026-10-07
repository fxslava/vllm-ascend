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

// The ACLNN V5 operator surface the DSV4 pipeline drives.
//
// Every `GetWorkspaceSize` prototype below is transcribed from the CANN
// 9.2.0-beta.2 header of the same name, and the audited subset matches
// `csrc/tests/common/aclnn_ops_950pr.hpp` argument for argument.
// `dsv4_aclnn_signature_check.cpp` includes the real headers and static_asserts
// each typedef against the declared function, so a toolkit that changes one
// breaks the build instead of the dlsym'd call reading the wrong registers.
//
// WHY THE ENTRY POINTS ARE RESOLVED, NOT LINK-BOUND
// -------------------------------------------------
// The CANN libraries are on the link line (see CMakeLists.txt), which is what
// makes `ldd -r`'s "zero undefined symbols" check meaningful for the ACL
// runtime, `aclCreateTensor`, `aclSetTensorAddr` and friends. The *operator*
// entry points are then taken out of those same already-loaded libraries with
// `dlsym(RTLD_DEFAULT, ...)` -- no `dlopen`, no second copy -- for one concrete
// reason: the op set is not the same across the toolkits this binary must
// survive. Checked here:
//
//   CANN 9.2.0-beta.2 x86_64   exports all the ops below.
//   CANN 9.1.0 aarch64         does not export `aclnnMoeGatingTopKV2` or
//                              `aclnnMoeInitRoutingV4` at all.
//
// A link-time dependency on those two would make the binary refuse to start on
// 9.1.0 with a loader error naming a symbol, instead of starting and reporting
// which operator its toolkit is missing. `aclnnFusedInferAttentionScoreV5-
// GetMaxWorkspaceSize` is a third case: it is *exported* by both toolkits and
// declared by neither header, so it can only be reached this way.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dsv4_acl_check.hpp"
#include "dsv4_device_ops.hpp"

// Forward declarations matching acl/aclnn/acl_meta.h, so this header does not
// drag the whole toolkit into every translation unit.
typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclTensorList aclTensorList;

namespace vllm_ascend {
namespace dsv4 {

enum class OpId {
  // Dense / backbone
  kRmsNorm,
  kRmsNormDynamicMxQuant,
  kDynamicMxQuant,
  kMatmul,
  kQuantMatmulV5,
  kApplyRotaryPosEmbV2,
  kScatterPaKvCache,
  kFusedInferAttentionScoreV5,
  kFiaV5GetMaxWorkspace,
  kSigmoid,
  kMul,
  kInplaceAdd,
  kSwiGlu,
  kArgMax,
  // MoE
  kMoeGatingTopKV2,
  kMoeInitRoutingV4,
  kGroupedMatmulV5,
  kSwigluMxQuant,
  kMoeTokenUnpermute,
  kGroupedMatmulSwigluQuantV2,
  kGroupedMatmulFinalizeRoutingV3,
  kOpCount,
};

struct ResolvedOp {
  const char* name = nullptr;
  const char* role = nullptr;
  void* plan = nullptr;    // aclnn<Op>GetWorkspaceSize
  void* launch = nullptr;  // aclnn<Op>
  std::string provider;    // the shared object dlsym found it in
  bool required = true;    // an optional op may legitimately be absent

  bool available() const { return plan != nullptr && launch != nullptr; }
};

// ---------------------------------------------------------------------------
// Plan prototypes (CANN 9.2.0-beta.2 headers, verbatim)
// ---------------------------------------------------------------------------

using RmsNormPlanFn = int (*)(const aclTensor* x, const aclTensor* gamma, double epsilon, const aclTensor* y_out,
                              const aclTensor* rstd_out, uint64_t* workspace_size, aclOpExecutor** executor);

// RMSNorm fused with OCP block-32 MX quantization: the activation quantizer
// for both the dense FP8 GEMMs and the routed FP4 expert GEMM.
using RmsNormDynamicMxQuantPlanFn = int (*)(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta,
                                            double epsilon, int64_t scale_alg, char* round_mode, int64_t dst_type,
                                            bool output_rstd, aclTensor* y_out, aclTensor* mxscale_out,
                                            aclTensor* rstd_out, uint64_t* workspace_size,
                                            aclOpExecutor** executor);

// Standalone OCP block-32 MX quantization, for the two places an activation has
// to become FP8 without a fused RMSNorm in front of it: the MLA attention
// output and the shared expert's SwiGLU output.
using DynamicMxQuantPlanFn = int (*)(const aclTensor* x, int64_t axis, char* round_mode_optional, int64_t dst_type,
                                     int64_t blocksize, int64_t scale_alg, const aclTensor* y_out,
                                     const aclTensor* mxscale_out, uint64_t* workspace_size,
                                     aclOpExecutor** executor);

using MatmulPlanFn = int (*)(const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cube_math_type,
                             uint64_t* workspace_size, aclOpExecutor** executor);

using QuantMatmulV5PlanFn = int (*)(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1_scale,
                                    const aclTensor* x2_scale, const aclTensor* y_scale, const aclTensor* x1_offset,
                                    const aclTensor* x2_offset, const aclTensor* y_offset, const aclTensor* bias,
                                    bool transpose_x1, bool transpose_x2, int64_t group_size, aclTensor* out,
                                    uint64_t* workspace_size, aclOpExecutor** executor);

using ApplyRotaryPosEmbV2PlanFn = int (*)(aclTensor* query_ref, aclTensor* key_ref, const aclTensor* cos,
                                          const aclTensor* sin, int64_t layout, char* rotary_mode,
                                          uint64_t* workspace_size, aclOpExecutor** executor);

using ScatterPaKvCachePlanFn = int (*)(const aclTensor* key, aclTensor* key_cache_ref, const aclTensor* slot_mapping,
                                       const aclTensor* value, aclTensor* value_cache_ref,
                                       const aclTensor* compress_lens_optional,
                                       const aclTensor* compress_seq_offset_optional,
                                       const aclTensor* seq_lens_optional, char* cache_mode_optional,
                                       char* scatter_mode_optional, const aclIntArray* strides_optional,
                                       const aclIntArray* offsets_optional, uint64_t* workspace_size,
                                       aclOpExecutor** executor);

// 46 arguments. Identical to FusedInferAttentionScoreV5WorkspaceFn in
// csrc/tests/common/aclnn_ops_950pr.hpp.
using FusedInferAttentionScoreV5PlanFn = int (*)(
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
    aclOpExecutor** executor);

using UnaryPlanFn = int (*)(const aclTensor* self, aclTensor* out, uint64_t* workspace_size,
                            aclOpExecutor** executor);

using BinaryPlanFn = int (*)(const aclTensor* self, const aclTensor* other, aclTensor* out,
                             uint64_t* workspace_size, aclOpExecutor** executor);

using InplaceAddPlanFn = int (*)(const aclTensor* self_ref, const aclTensor* other, const aclScalar* alpha,
                                 uint64_t* workspace_size, aclOpExecutor** executor);

using SwiGluPlanFn = int (*)(const aclTensor* x, int64_t dim, const aclTensor* out, uint64_t* workspace_size,
                             aclOpExecutor** executor);

using ArgMaxPlanFn = int (*)(const aclTensor* self, int64_t dim, bool keepdim, aclTensor* out,
                             uint64_t* workspace_size, aclOpExecutor** executor);

using MoeGatingTopKV2PlanFn = int (*)(const aclTensor* x, const aclTensor* bias_optional,
                                      const aclTensor* input_ids_optional, const aclTensor* tid2eid_optional,
                                      int64_t k, int64_t k_group, int64_t group_count, int64_t group_select_mode,
                                      int64_t renorm, int64_t norm_type, bool out_flag,
                                      double routed_scaling_factor, double eps, const aclTensor* y_out,
                                      const aclTensor* expert_idx_out, const aclTensor* out_out,
                                      uint64_t* workspace_size, aclOpExecutor** executor);

using MoeInitRoutingV4PlanFn = int (*)(const aclTensor* x, const aclTensor* expert_idx,
                                       const aclTensor* scale_optional, const aclTensor* offset_optional,
                                       const aclTensor* active_num_optional, const aclTensor* topk_weight_optional,
                                       int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode,
                                       int64_t expert_tokens_num_type, bool expert_tokens_num_flag,
                                       int64_t quant_mode, const aclIntArray* active_expert_range_optional,
                                       int64_t row_idx_type, const aclTensor* expanded_x_out,
                                       const aclTensor* expanded_row_idx_out,
                                       const aclTensor* expert_tokens_count_or_cumsum_out,
                                       const aclTensor* expanded_scale_out,
                                       const aclTensor* expanded_topk_weight_out, uint64_t* workspace_size,
                                       aclOpExecutor** executor);

// 22 arguments.
using GroupedMatmulV5PlanFn = int (*)(const aclTensorList* x, const aclTensorList* weight,
                                      const aclTensorList* bias_optional, const aclTensorList* scale_optional,
                                      const aclTensorList* offset_optional,
                                      const aclTensorList* antiquant_scale_optional,
                                      const aclTensorList* antiquant_offset_optional,
                                      const aclTensorList* per_token_scale_optional,
                                      const aclTensor* group_list_optional,
                                      const aclTensorList* activation_input_optional,
                                      const aclTensorList* activation_quant_scale_optional,
                                      const aclTensorList* activation_quant_offset_optional, int64_t split_item,
                                      int64_t group_type, int64_t group_list_type, int64_t act_type,
                                      aclIntArray* tuning_config_optional, aclTensorList* out,
                                      aclTensorList* activation_feature_out_optional,
                                      aclTensorList* dyn_quant_scale_out_optional, uint64_t* workspace_size,
                                      aclOpExecutor** executor);

using SwigluMxQuantPlanFn = int (*)(const aclTensor* x, const aclTensor* group_index_optional, int64_t activate_dim,
                                    bool activate_left, int64_t swiglu_mode, double clamp_limit, double glu_alpha,
                                    double glu_bias, int64_t group_mode, int64_t axis, int64_t dst_type,
                                    char* round_mode_optional, int64_t scale_alg, double max_dtype_value,
                                    const aclTensor* y_out, const aclTensor* mxscale_out, uint64_t* workspace_size,
                                    aclOpExecutor** executor);

using MoeTokenUnpermutePlanFn = int (*)(const aclTensor* permuted_tokens, const aclTensor* sorted_indices,
                                        const aclTensor* probs_optional, bool padded_mode,
                                        const aclIntArray* restore_shape_optional, aclTensor* out,
                                        uint64_t* workspace_size, aclOpExecutor** executor);

using GroupedMatmulSwigluQuantV2PlanFn =
    int (*)(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weight_scale,
            const aclTensorList* weight_assist_matrix, const aclTensor* bias, const aclTensor* x_scale,
            const aclTensor* smooth_scale, const aclTensor* group_list, int64_t dequant_mode, int64_t dequant_dtype,
            int64_t quant_mode, int64_t group_list_type, const aclIntArray* tuning_config_optional,
            aclTensor* output, aclTensor* output_scale, uint64_t* workspace_size, aclOpExecutor** executor);

using GroupedMatmulFinalizeRoutingV3PlanFn =
    int (*)(const aclTensor* x1, aclTensor* x2, const aclTensor* scale_optional, const aclTensor* bias_optional,
            const aclTensor* offset_optional, const aclTensor* antiquant_scale_optional,
            const aclTensor* antiquant_offset_optional, const aclTensor* pertoken_scale_optional,
            const aclTensor* group_list_optional, const aclTensor* shared_input_optional,
            const aclTensor* logit_optional, const aclTensor* row_index_optional, int64_t dtype,
            float shared_input_weight, int64_t shared_input_offset, bool transpose_x1, bool transpose_x2,
            int64_t group_list_type, const aclIntArray* tuning_config_optional, aclTensor* out,
            uint64_t* workspace_size, aclOpExecutor** executor);

// Every two-phase launch entry has the same shape.
using AclnnLaunchFn = int (*)(void* workspace, uint64_t workspace_size, aclOpExecutor* executor, void* stream);

// ---------------------------------------------------------------------------
// OpTable
// ---------------------------------------------------------------------------

class OpTable {
 public:
  // Resolves every entry once. Does not throw when an op is missing; the
  // pipeline decides which absences are fatal for the path it selected.
  OpTable();

  const ResolvedOp& op(OpId id) const;
  bool available(OpId id) const { return op(id).available(); }

  // Throws naming every required op this toolkit does not export.
  void RequireAll(const std::vector<OpId>& ids) const;

  template <typename PlanFn>
  PlanFn plan_fn(OpId id) const {
    RequireAvailable(id);
    return reinterpret_cast<PlanFn>(op(id).plan);
  }

  AclnnLaunchFn launch_fn(OpId id) const;

  std::string DescribeInventory() const;

  // True when the aclnn libraries could be reached at all. False means the
  // binary is running without the CANN runtime on its loader path, which is a
  // different failure from a missing operator.
  bool runtime_reachable() const { return runtime_reachable_; }

 private:
  void Resolve(OpId id, const char* name, const char* role, bool required);
  void ResolveBare(OpId id, const char* name, const char* role, bool required);
  void RequireAvailable(OpId id) const;

  ResolvedOp ops_[static_cast<size_t>(OpId::kOpCount)];
  bool runtime_reachable_ = false;
};

// ---------------------------------------------------------------------------
// One pre-planned, address-swappable operator invocation
// ---------------------------------------------------------------------------
//
// The two-call ACLNN protocol normally consumes its executor on launch. With
// `aclSetAclOpExecutorRepeatable` the executor survives, and the only thing a
// later launch needs is the new addresses -- which is what makes a decode loop
// with zero descriptor creations and zero re-planning possible.
class StaticOpSlot {
 public:
  StaticOpSlot() = default;
  ~StaticOpSlot();

  StaticOpSlot(const StaticOpSlot&) = delete;
  StaticOpSlot& operator=(const StaticOpSlot&) = delete;

  // `PlanAclnnOp` has already run; this records the result and makes the
  // executor repeatable.
  void Adopt(OpId id, const char* label, uint64_t workspace_size, aclOpExecutor* executor);

  // Repoint input/output slot `index` of the retained executor at `address`.
  void SetAddress(size_t index, aclTensor* tensor, void* address) const;
  void SetTensorListAddress(size_t ir_index, size_t relative_index, aclTensorList* tensors, void* address) const;

  void Launch(const OpTable& table, void* workspace, void* stream) const;

  uint64_t workspace_size() const { return workspace_size_; }
  aclOpExecutor* executor() const { return executor_; }
  bool planned() const { return executor_ != nullptr; }
  const char* label() const { return label_; }
  OpId id() const { return id_; }

 private:
  OpId id_ = OpId::kOpCount;
  const char* label_ = "<unplanned>";
  uint64_t workspace_size_ = 0;
  aclOpExecutor* executor_ = nullptr;
};

const char* OpName(OpId id);

// Runs a plan function and returns the workspace size, throwing with the op's
// name on a non-zero status. Variadic so every op keeps its real signature; the
// two protocol arguments are appended here, never at the call site.
template <typename PlanFn, typename... Args>
uint64_t PlanAclnnOp(const OpTable& table, OpId id, aclOpExecutor** executor, Args... args) {
  PlanFn plan = table.plan_fn<PlanFn>(id);
  uint64_t workspace_size = 0;
  const int status = plan(args..., &workspace_size, executor);
  if (status != 0) {
    throw AclError(OpName(id), __FILE__, __LINE__, status);
  }
  if (*executor == nullptr) {
    throw Dsv4Error(std::string(OpName(id)) + "GetWorkspaceSize returned no executor");
  }
  return workspace_size;
}

}  // namespace dsv4
}  // namespace vllm_ascend
