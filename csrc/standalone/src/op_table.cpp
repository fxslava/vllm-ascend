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

#include "moe/core/op_table.hpp"

#include <dlfcn.h>

#include <iomanip>
#include <sstream>

// acl_meta.h declares aclSetAclOpExecutorRepeatable, aclSetTensorAddr,
// aclSetDynamicTensorAddr and aclDestroyAclOpExecutor -- the four entry points
// the static-arena design rests on. They are link-bound, not dlsym'd: they
// exist in every toolkit checked (9.1.0 aarch64 and 9.2.0 x86_64), and a
// toolkit without them cannot run this design at all.
#include <aclnn/acl_meta.h>

namespace ascend_moe {
namespace {

struct OpDeclaration {
  OpId id;
  const char* name;
  const char* role;
  bool required;
  bool bare;  // exported without the GetWorkspaceSize/launch pair
};

// The entry points the pipeline can drive. `required=false` marks the ops that
// one of the two MoE paths uses and the other does not, the elementwise ops the
// Qwen3.5 stage map in aclnn_ops_950pr.hpp lists but DeepSeek MLA has no gate
// for, and the undeclared FIA upper-bound helper.
const OpDeclaration kDeclarations[] = {
    {OpId::kRmsNorm, "aclnnRmsNorm", "input, latent, post-attention and final RMSNorm", true, false},
    {OpId::kRmsNormDynamicMxQuant, "aclnnRmsNormDynamicMxQuant",
     "RMSNorm fused with FP8 + E8M0 block-32 activation quant", true, false},
    {OpId::kDynamicMxQuant, "aclnnDynamicMxQuant",
     "FP8 + E8M0 block-32 activation quant without a fused norm", true, false},
    {OpId::kMatmul, "aclnnMatmul", "router logits, the routing combine and the LM head", true, false},
    {OpId::kQuantMatmulV5, "aclnnQuantMatmulV5", "FP8 dense projections (block-128 weight scales)", true, false},
    {OpId::kApplyRotaryPosEmbV2, "aclnnApplyRotaryPosEmbV2", "partial RoPE on the q/k rope slices", true, false},
    {OpId::kScatterPaKvCache, "aclnnScatterPaKvCache", "paged MLA KV cache write", true, false},
    {OpId::kFusedInferAttentionScoreV5, "aclnnFusedInferAttentionScoreV5", "paged MLA decode attention", true,
     false},
    {OpId::kFiaV5GetMaxWorkspace, "aclnnFusedInferAttentionScoreV5GetMaxWorkspaceSize",
     "FIA workspace upper bound (exported, undeclared by any header)", false, true},
    {OpId::kSigmoid, "aclnnSigmoid", "attention output gate (Qwen3.5 stage map; unused by DSV4)", false, false},
    {OpId::kMul, "aclnnMul", "attention output gate (Qwen3.5 stage map; unused by DSV4)", false, false},
    {OpId::kInplaceAdd, "aclnnInplaceAdd", "attention and MoE residual adds", true, false},
    {OpId::kSwiGlu, "aclnnSwiGlu", "shared-expert activation", true, false},
    {OpId::kArgMax, "aclnnArgMax", "greedy token selection on the LM head", true, false},
    {OpId::kSoftplus, "aclnnSoftplus", "sqrtsoftplus router scoring, stage 1 (host-scalar beta/threshold)", true,
     false},
    {OpId::kSqrt, "aclnnSqrt", "sqrtsoftplus router scoring, stage 2", true, false},
    {OpId::kMoeGatingTopKV2, "aclnnMoeGatingTopKV2",
     "noaux_tc router over pre-normalized scores, top-6 of 256, scaling 1.5", true, false},
    {OpId::kMoeInitRoutingV4, "aclnnMoeInitRoutingV4", "dropless dispatch, device cumsum groupList", true, false},
    {OpId::kGroupedMatmulV5, "aclnnGroupedMatmulV5", "FP8xFP4 M-grouped expert GEMM", true, false},
    {OpId::kSwigluMxQuant, "aclnnSwigluMxQuant", "clamped SwiGLU + MX requant (decomposed path)", false, false},
    {OpId::kMoeTokenUnpermute, "aclnnMoeTokenUnpermute", "routing combine (decomposed path)", false, false},
    {OpId::kGroupedMatmulSwigluQuantV2, "aclnnGroupedMatmulSwigluQuantV2",
     "fused expert GEMM1 + clamped SwiGLU + MX requant (fused path)", false, false},
    {OpId::kGroupedMatmulFinalizeRoutingV3, "aclnnGroupedMatmulFinalizeRoutingV3",
     "fused GEMM2 + combine; unusable with scattered expert slots, see dsv4_pipeline.cpp", false, false},
};

// Resolve a symbol out of the libraries already on the loader path. The CANN
// libraries are DT_NEEDED of this binary (see CMakeLists.txt), so RTLD_DEFAULT
// finds them without a dlopen; the fallback covers a build where a split
// libopapi_* was dropped from the link line.
void* ResolveSymbol(const char* symbol, std::string* provider) {
  ::dlerror();
  void* address = ::dlsym(RTLD_DEFAULT, symbol);
  if (address == nullptr) {
    static const char* const kCandidates[] = {"libopapi.so", "libopapi_nn.so", "libopapi_transformer.so",
                                              "libopapi_math.so", "libcust_opapi.so"};
    for (const char* candidate : kCandidates) {
      void* handle = ::dlopen(candidate, RTLD_LAZY | RTLD_GLOBAL);
      if (handle == nullptr) {
        continue;
      }
      address = ::dlsym(handle, symbol);
      if (address != nullptr) {
        break;
      }
    }
  }
  if (address != nullptr && provider != nullptr) {
    Dl_info info{};
    if (::dladdr(address, &info) != 0 && info.dli_fname != nullptr) {
      *provider = info.dli_fname;
    } else {
      *provider = "<unknown>";
    }
  }
  return address;
}

}  // namespace

const char* OpName(OpId id) {
  for (const OpDeclaration& declaration : kDeclarations) {
    if (declaration.id == id) {
      return declaration.name;
    }
  }
  return "<unknown aclnn op>";
}

// ---------------------------------------------------------------------------
// OpTable
// ---------------------------------------------------------------------------

OpTable::OpTable() {
  for (const OpDeclaration& declaration : kDeclarations) {
    if (declaration.bare) {
      ResolveBare(declaration.id, declaration.name, declaration.role, declaration.required);
    } else {
      Resolve(declaration.id, declaration.name, declaration.role, declaration.required);
    }
  }
  // One unconditional probe of the runtime itself: if even aclnnRmsNorm is
  // unreachable the problem is the loader path, not the operator set.
  runtime_reachable_ = ops_[static_cast<size_t>(OpId::kRmsNorm)].plan != nullptr;
}

void OpTable::Resolve(OpId id, const char* name, const char* role, bool required) {
  ResolvedOp& entry = ops_[static_cast<size_t>(id)];
  entry.name = name;
  entry.role = role;
  entry.required = required;
  const std::string plan_symbol = std::string(name) + "GetWorkspaceSize";
  entry.plan = ResolveSymbol(plan_symbol.c_str(), &entry.provider);
  entry.launch = ResolveSymbol(name, entry.plan == nullptr ? &entry.provider : nullptr);
}

void OpTable::ResolveBare(OpId id, const char* name, const char* role, bool required) {
  // `aclnnFusedInferAttentionScoreV5GetMaxWorkspaceSize` is a planning helper,
  // not a two-phase operator: it has a plan signature and no launch. Pointing
  // `launch` at the same address keeps `available()` honest without inventing a
  // launch entry that does not exist.
  ResolvedOp& entry = ops_[static_cast<size_t>(id)];
  entry.name = name;
  entry.role = role;
  entry.required = required;
  entry.plan = ResolveSymbol(name, &entry.provider);
  entry.launch = entry.plan;
}

const ResolvedOp& OpTable::op(OpId id) const {
  const size_t index = static_cast<size_t>(id);
  DSV4_REQUIRE(index < static_cast<size_t>(OpId::kOpCount), "operator id " << index << " is out of range");
  return ops_[index];
}

void OpTable::RequireAvailable(OpId id) const {
  const ResolvedOp& entry = op(id);
  DSV4_REQUIRE(entry.available(), "this CANN installation does not export "
                                      << entry.name << " (" << entry.role
                                      << "); run with --verbose to print the full inventory");
}

void OpTable::RequireAll(const std::vector<OpId>& ids) const {
  std::ostringstream missing;
  int absent = 0;
  for (OpId id : ids) {
    if (!op(id).available()) {
      missing << (absent++ ? ", " : "") << op(id).name;
    }
  }
  DSV4_REQUIRE(absent == 0, "the selected execution path needs "
                                << absent << " operator(s) this CANN installation does not export: "
                                << missing.str());
}

AclnnLaunchFn OpTable::launch_fn(OpId id) const {
  RequireAvailable(id);
  return reinterpret_cast<AclnnLaunchFn>(op(id).launch);
}

std::string OpTable::DescribeInventory() const {
  std::ostringstream out;
  out << "aclnn V5 operator inventory\n";
  if (!runtime_reachable_) {
    out << "  the aclnn runtime is NOT on this process's loader path: no operator resolved.\n";
  }
  for (const OpDeclaration& declaration : kDeclarations) {
    const ResolvedOp& entry = op(declaration.id);
    out << "  " << std::left << std::setw(48) << entry.name << std::setw(9)
        << (entry.available() ? "found" : (entry.required ? "MISSING" : "absent")) << entry.role;
    if (entry.available() && !entry.provider.empty()) {
      out << "\n      from " << entry.provider;
    }
    out << "\n";
  }
  return out.str();
}

// ---------------------------------------------------------------------------
// StaticOpSlot
// ---------------------------------------------------------------------------

StaticOpSlot::~StaticOpSlot() {
  if (executor_ != nullptr) {
    aclDestroyAclOpExecutor(executor_);
    executor_ = nullptr;
  }
}

void StaticOpSlot::Adopt(OpId id, const char* label, uint64_t workspace_size, aclOpExecutor* executor) {
  DSV4_REQUIRE(executor != nullptr, label << ": cannot adopt a null executor");
  DSV4_REQUIRE(executor_ == nullptr, label << ": this slot already holds a planned executor");
  id_ = id;
  label_ = label;
  workspace_size_ = workspace_size;
  executor_ = executor;
  // Without this, the launch consumes the executor and the next step would have
  // to re-plan -- which is exactly the per-step host work the brief forbids.
  DSV4_ACL_CHECK(aclSetAclOpExecutorRepeatable(executor_));
}

void StaticOpSlot::SetAddress(size_t index, aclTensor* tensor, void* address) const {
  DSV4_REQUIRE(executor_ != nullptr, label_ << ": SetAddress before the stage was planned");
  DSV4_ACL_CHECK(aclSetTensorAddr(executor_, index, tensor, address));
}

void StaticOpSlot::SetTensorListAddress(size_t ir_index, size_t relative_index, aclTensorList* tensors,
                                        void* address) const {
  DSV4_REQUIRE(executor_ != nullptr, label_ << ": SetTensorListAddress before the stage was planned");
  DSV4_ACL_CHECK(aclSetDynamicTensorAddr(executor_, ir_index, relative_index, tensors, address));
}

void StaticOpSlot::Launch(const OpTable& table, void* workspace, void* stream) const {
  DSV4_REQUIRE(executor_ != nullptr, label_ << ": Launch before the stage was planned");
  DSV4_REQUIRE(workspace != nullptr || workspace_size_ == 0,
               label_ << ": needs " << workspace_size_ << " workspace bytes but was given none");
  AclnnLaunchFn launch = table.launch_fn(id_);
  const int status = launch(workspace_size_ == 0 ? nullptr : workspace, workspace_size_, executor_, stream);
  if (status != 0) {
    throw AclError(label_, __FILE__, __LINE__, status);
  }
}

}  // namespace ascend_moe
