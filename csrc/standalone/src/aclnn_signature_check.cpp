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

// Compile-time proof that the prototypes in dsv4_aclnn_v5.hpp are the toolkit's
// own.
//
// The operator entry points are reached with `dlsym`, which means a signature
// that drifted would be an ABI mismatch at run time -- arguments silently read
// from the wrong registers, no error, wrong numbers. This translation unit
// includes the real CANN headers and asserts that each hand-written typedef is
// exactly the type of the declared function. It contains no code: if it
// compiles, the prototypes are right for the toolkit being built against, and
// if the toolkit changes one, the build fails here instead of the kernel
// reading garbage.
//
// Ops that exist in CANN 9.2.0 but not 9.1.0 are guarded by `__has_include`, so
// a 9.1.0 cross build still checks everything 9.1.0 ships.

#include <type_traits>

#include "moe/core/op_table.hpp"

#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_apply_rotary_pos_emb_v2.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_dynamic_mx_quant.h>
#include <aclnnop/aclnn_fused_infer_attention_score_v5.h>
#include <aclnnop/aclnn_grouped_matmul_finalize_routing_v3.h>
#include <aclnnop/aclnn_grouped_matmul_swiglu_quant_v2.h>
#include <aclnnop/aclnn_grouped_matmul_v5.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_moe_token_unpermute.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_quant_matmul_v5.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <aclnnop/aclnn_rms_norm_dynamic_mx_quant.h>
#include <aclnnop/aclnn_scatter_pa_kv_cache.h>
#include <aclnnop/aclnn_sigmoid.h>
#include <aclnnop/aclnn_swi_glu.h>
#include <aclnnop/aclnn_swiglu_mx_quant.h>

#if __has_include(<aclnnop/aclnn_softplus.h>)
#include <aclnnop/aclnn_softplus.h>
#define DSV4_HAS_SOFTPLUS 1
#endif

#if __has_include(<aclnnop/aclnn_sqrt.h>)
#include <aclnnop/aclnn_sqrt.h>
#define DSV4_HAS_SQRT 1
#endif

#if __has_include(<aclnnop/aclnn_moe_gating_top_k_v2.h>)
#include <aclnnop/aclnn_moe_gating_top_k_v2.h>
#define DSV4_HAS_MOE_GATING_TOP_K_V2 1
#endif

#if __has_include(<aclnnop/aclnn_moe_init_routing_v4.h>)
#include <aclnnop/aclnn_moe_init_routing_v4.h>
#define DSV4_HAS_MOE_INIT_ROUTING_V4 1
#endif

namespace ascend_moe {
namespace {

template <typename Declared, typename Transcribed>
constexpr bool SameSignature() {
  return std::is_same<Declared, Transcribed>::value;
}

#define DSV4_ASSERT_SIGNATURE(symbol, typedef_name)                                         \
  static_assert(SameSignature<decltype(&symbol), typedef_name>(),                           \
                #symbol " does not match the transcribed " #typedef_name                    \
                        " in dsv4_aclnn_v5.hpp; the toolkit changed the signature and the " \
                        "dlsym'd call would read the wrong arguments")

DSV4_ASSERT_SIGNATURE(aclnnRmsNormGetWorkspaceSize, RmsNormPlanFn);
DSV4_ASSERT_SIGNATURE(aclnnRmsNormDynamicMxQuantGetWorkspaceSize, RmsNormDynamicMxQuantPlanFn);
DSV4_ASSERT_SIGNATURE(aclnnDynamicMxQuantGetWorkspaceSize, DynamicMxQuantPlanFn);
DSV4_ASSERT_SIGNATURE(aclnnMatmulGetWorkspaceSize, MatmulPlanFn);
DSV4_ASSERT_SIGNATURE(aclnnQuantMatmulV5GetWorkspaceSize, QuantMatmulV5PlanFn);
DSV4_ASSERT_SIGNATURE(aclnnApplyRotaryPosEmbV2GetWorkspaceSize, ApplyRotaryPosEmbV2PlanFn);
DSV4_ASSERT_SIGNATURE(aclnnScatterPaKvCacheGetWorkspaceSize, ScatterPaKvCachePlanFn);
DSV4_ASSERT_SIGNATURE(aclnnFusedInferAttentionScoreV5GetWorkspaceSize, FusedInferAttentionScoreV5PlanFn);
DSV4_ASSERT_SIGNATURE(aclnnSigmoidGetWorkspaceSize, UnaryPlanFn);
DSV4_ASSERT_SIGNATURE(aclnnMulGetWorkspaceSize, BinaryPlanFn);
DSV4_ASSERT_SIGNATURE(aclnnInplaceAddGetWorkspaceSize, InplaceAddPlanFn);
DSV4_ASSERT_SIGNATURE(aclnnSwiGluGetWorkspaceSize, SwiGluPlanFn);
DSV4_ASSERT_SIGNATURE(aclnnArgMaxGetWorkspaceSize, ArgMaxPlanFn);
DSV4_ASSERT_SIGNATURE(aclnnGroupedMatmulV5GetWorkspaceSize, GroupedMatmulV5PlanFn);
DSV4_ASSERT_SIGNATURE(aclnnSwigluMxQuantGetWorkspaceSize, SwigluMxQuantPlanFn);
DSV4_ASSERT_SIGNATURE(aclnnMoeTokenUnpermuteGetWorkspaceSize, MoeTokenUnpermutePlanFn);
DSV4_ASSERT_SIGNATURE(aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize, GroupedMatmulSwigluQuantV2PlanFn);
DSV4_ASSERT_SIGNATURE(aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize, GroupedMatmulFinalizeRoutingV3PlanFn);

#ifdef DSV4_HAS_SOFTPLUS
DSV4_ASSERT_SIGNATURE(aclnnSoftplusGetWorkspaceSize, SoftplusPlanFn);
#endif
#ifdef DSV4_HAS_SQRT
DSV4_ASSERT_SIGNATURE(aclnnSqrtGetWorkspaceSize, UnaryPlanFn);
#endif

#ifdef DSV4_HAS_MOE_GATING_TOP_K_V2
DSV4_ASSERT_SIGNATURE(aclnnMoeGatingTopKV2GetWorkspaceSize, MoeGatingTopKV2PlanFn);
#endif
#ifdef DSV4_HAS_MOE_INIT_ROUTING_V4
DSV4_ASSERT_SIGNATURE(aclnnMoeInitRoutingV4GetWorkspaceSize, MoeInitRoutingV4PlanFn);
#endif

// Every two-phase launch entry has this shape, and `StaticOpSlot::Launch`
// reinterpret_casts to it, so it is worth pinning too. aclrtStream is void*, so
// the transcribed `void* stream` is the same type.
DSV4_ASSERT_SIGNATURE(aclnnRmsNorm, AclnnLaunchFn);
DSV4_ASSERT_SIGNATURE(aclnnGroupedMatmulV5, AclnnLaunchFn);
DSV4_ASSERT_SIGNATURE(aclnnFusedInferAttentionScoreV5, AclnnLaunchFn);

#undef DSV4_ASSERT_SIGNATURE

}  // namespace
}  // namespace ascend_moe
