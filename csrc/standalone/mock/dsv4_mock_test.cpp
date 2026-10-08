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

// dsv4_mock_test -- the zero-NPU, zero-allocation contract suite.
//
// Runs natively on an x86 host (WSL) against libopapi_mock: the symbolic
// allocator's interval registry, the shadow descriptor engine, the DeepSeek-V4
// Flash operator contracts in the GetWorkspaceSize stubs, and the FULL
// 43-layer decode graph -- arena, paged KV, exclusive expert hierarchy at the
// complete 11,008-expert coverage -- all as pure address bookkeeping. Physical
// footprint stays at the few pinned mailboxes; the final section reads
// /proc/self/status to prove it.

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "mock_acl_tensor.hpp"
#include "mock_allocator.hpp"
#include "mock_ops_api.hpp"

#include "moe/core/error.hpp"
#include "moe/core/model_config.hpp"
#include "moe/core/op_table.hpp"
#include "moe/core/config.hpp"
#include "moe/core/device_ops.hpp"
#include "moe/memory/exclusive_staging.hpp"
#include "moe/memory/expert_layout.hpp"
#include "moe/pipeline/lattice_moe_block.hpp"
#include "moe/pipeline/pipeline.hpp"
#include "moe/core/weight_source.hpp"

// The dlsym'd mock entry points are ABI-compatible with the product's
// transcribed plan typedefs -- pinned here so a drift in either side is a
// compile error, not a silent wrong-register read.
static_assert(std::is_same<decltype(&aclnnSoftplusGetWorkspaceSize),
                           ascend_moe::SoftplusPlanFn>::value,
              "mock aclnnSoftplus must match SoftplusPlanFn");
static_assert(std::is_same<decltype(&aclnnSqrtGetWorkspaceSize),
                           ascend_moe::UnaryPlanFn>::value,
              "mock aclnnSqrt must match UnaryPlanFn");
static_assert(std::is_same<decltype(&aclnnMoeGatingTopKV2GetWorkspaceSize),
                           ascend_moe::MoeGatingTopKV2PlanFn>::value,
              "mock aclnnMoeGatingTopKV2 must match MoeGatingTopKV2PlanFn");
static_assert(std::is_same<decltype(&aclnnMoeInitRoutingV4GetWorkspaceSize),
                           ascend_moe::MoeInitRoutingV4PlanFn>::value,
              "mock aclnnMoeInitRoutingV4 must match MoeInitRoutingV4PlanFn");
static_assert(std::is_same<decltype(&aclnnGroupedMatmulV5GetWorkspaceSize),
                           ascend_moe::GroupedMatmulV5PlanFn>::value,
              "mock aclnnGroupedMatmulV5 must match GroupedMatmulV5PlanFn");

namespace ascend_moe {
namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool condition, const std::string& what) {
  ++g_checks;
  if (condition) {
    std::printf("  [ ok ] %s\n", what.c_str());
  } else {
    ++g_failures;
    std::printf("  [FAIL] %s\n", what.c_str());
  }
}

template <typename Action>
void CheckRefuses(Action action, const std::string& what) {
  ++g_checks;
  try {
    action();
  } catch (const Dsv4Error&) {
    std::printf("  [ ok ] refused: %s\n", what.c_str());
    return;
  } catch (const std::exception& error) {
    ++g_failures;
    std::printf("  [FAIL] %s threw the wrong type: %s\n", what.c_str(), error.what());
    return;
  }
  ++g_failures;
  std::printf("  [FAIL] %s was allowed\n", what.c_str());
}

void Section(const char* title) { std::printf("\n== %s ==\n", title); }

using namespace mock;  // NOLINT(build/namespaces) -- test-local brevity

// ---------------------------------------------------------------------------
// 1. The interval registry
// ---------------------------------------------------------------------------

void TestIntervalRegistry() {
  Section("symbolic allocator: fake addresses and the interval registry");
  MockResetAllocatorForTest();

  const uintptr_t first = MockDeviceMalloc(100);
  const uintptr_t second = MockDeviceMalloc(13369344);
  const uintptr_t host = MockHostMalloc(24);  // a mailbox: real
  Check(first % kMockAllocAlignBytes == 0 && second % kMockAllocAlignBytes == 0 &&
            host % kMockAllocAlignBytes == 0,
        "every base address is 4096-byte aligned (aclrtMalloc HUGE_FIRST parity)");
  Check(first < second, "fake addresses are monotonically increasing");
  Check(MockSpanContains(first, 100), "a registered span contains its own bytes");
  Check(MockSpanContains(first + 4096 - 100, 100) || !MockSpanContains(first + 1, 100),
        "containment is strict: [first, first + size) does not spill past the span end");

  Check(!MockSpanContains(first + 100, 1), "containment is exact: one byte past the 100-byte span is outside");
  Check(!MockSpanContains(second + 13369344, 1), "a byte past the expert slot's span end is outside");
  Check(MockSpanContains(second, 13369344), "a full 13,369,344-byte expert slot fits its span");

  std::string reason;
  Check(MockCheckTransfer(second, first, 100, &reason), "a fully contained transfer passes");
  Check(!MockCheckTransfer(second, first, 200, &reason),
        "an out-of-bounds source interval is refused");
  Check(!MockCheckTransfer(second + 13369344 - 50, first, 100, &reason),
        "an out-of-bounds destination interval is refused");
  const bool overlap_refused = !MockCheckTransfer(second + 50, second, 100, &reason);
  Check(overlap_refused && reason.find("overlap") != std::string::npos,
        "partially overlapping source/destination is refused as non-inplace aliasing: " + reason);
  Check(MockCheckTransfer(first, first, 100, &reason), "an identical in-place interval is legal");

  Check(MockUnregisterSpan(first) && !MockSpanContains(first, 100), "aclrtFree deregisters the span");
  Check(!MockUnregisterSpan(first), "a double free is refused");
  MockUnregisterSpan(second);
  MockUnregisterSpan(host);
  Check(MockMemoryStatistics().rejected_operations > 0, "refusals were counted");
}

// ---------------------------------------------------------------------------
// 2. The shadow descriptor engine
// ---------------------------------------------------------------------------

aclTensor* MakeTensor(const std::vector<int64_t>& shape, aclDataType dtype, void* data) {
  return aclCreateTensor(shape.data(), shape.size(), dtype,
                         nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(), data);
}

void TestDescriptorEngine() {
  Section("shadow descriptor engine: MockAclTensor and aclSetTensorAddr");
  MockResetAllocatorForTest();

  const uintptr_t arena = MockDeviceMalloc(1ull << 20);
  aclTensor* tensor = MakeTensor({1, kNumRoutedExperts}, ACL_FLOAT32, reinterpret_cast<void*>(arena));
  MockAclTensor* meta = AsMockTensor(tensor);
  Check(meta != nullptr && meta->magic == kMockAclTensorMagic, "aclCreateTensor yields a MockAclTensor");
  Check(meta->total_bytes == 1024, "metadata carries the storage size (256 fp32 = 1024 bytes)");
  Check(meta->shape[1] == kNumRoutedExperts, "metadata carries the shape");

  float one = 1.0f;
  float twenty = 20.0f;
  aclScalar* beta = aclCreateScalar(&one, ACL_FLOAT32);
  aclScalar* threshold = aclCreateScalar(&twenty, ACL_FLOAT32);
  Check(AsMockScalar(beta)->as_f64() == 1.0f, "aclCreateScalar keeps fp32 bytes (1.0f)");

  // A small executor to repoint.
  uint64_t workspace = 0;
  aclOpExecutor* executor = nullptr;
  const bool planned = aclnnSoftplusGetWorkspaceSize(tensor, beta, threshold, tensor, &workspace, &executor) == 0 &&
                       workspace > 0 && executor != nullptr;
  Check(planned, "a softplus plan over [1, 256] fp32 succeeds and sizes a workspace");

  Check(aclSetTensorAddr(executor, 0, tensor, reinterpret_cast<void*>(arena + 512)) == 0,
        "aclSetTensorAddr accepts an in-arena address");
  Check(AsMockTensor(tensor)->device_addr == reinterpret_cast<void*>(arena + 512),
        "the repoint is visible through the same handle");
  const bool overruns = aclSetTensorAddr(executor, 0, tensor, reinterpret_cast<void*>(arena + (1 << 20) - 4)) != 0;
  Check(overruns, "aclSetTensorAddr refuses an address whose tensor would overrun the span");
  Check(AsMockTensor(MakeTensor({3}, ACL_FP4X2_E2M1, nullptr))->total_bytes == 2,
        "FP4 packs two E2M1 values per byte");

  aclDestroyTensor(tensor);
  aclDestroyScalar(beta);
  aclDestroyScalar(threshold);
  aclDestroyAclOpExecutor(executor);
  MockUnregisterSpan(arena);
}

// ---------------------------------------------------------------------------
// 3. The DeepSeek-V4 Flash operator contracts (positive and negative)
// ---------------------------------------------------------------------------

void TestOperatorContracts() {
  Section("operator contracts: sqrtsoftplus scoring, gating, routing, grouped GEMM");
  MockResetAllocatorForTest();
  const uintptr_t arena = MockDeviceMalloc(6ull * 13369344 + (1ull << 22));  // six full expert slots + slack
  uint64_t workspace = 0;
  aclOpExecutor* executor = nullptr;

  // -- softplus / sqrt over [tokens, 256] ------------------------------------
  aclTensor* scores = MakeTensor({4, 256}, ACL_FLOAT32, reinterpret_cast<void*>(arena));
  aclTensor* scores_out = MakeTensor({4, 256}, ACL_FLOAT32, reinterpret_cast<void*>(arena + 4096));
  float one = 1.0f;
  float twenty = 20.0f;
  aclScalar* beta = aclCreateScalar(&one, ACL_FLOAT32);
  aclScalar* threshold = aclCreateScalar(&twenty, ACL_FLOAT32);
  Check(aclnnSoftplusGetWorkspaceSize(scores, beta, threshold, scores_out, &workspace, &executor) == 0,
        "softplus accepts [4, 256] fp32 with beta 1.0 / threshold 20.0");
  aclDestroyAclOpExecutor(executor);
  executor = nullptr;
  Check(aclnnSqrtGetWorkspaceSize(scores, scores_out, &workspace, &executor) == 0, "sqrt accepts the same contract");
  aclDestroyAclOpExecutor(executor);
  executor = nullptr;
  bool refused = aclnnSoftplusGetWorkspaceSize(MakeTensor({4, 128}, ACL_FLOAT32, nullptr), beta, threshold,
                                               scores_out, &workspace, &executor) != 0;
  std::string why = MockLastContractFailure();
  Check(refused && why.find("[tokens, 256]") != std::string::npos, "softplus refuses [4, 128]: " + why);
  executor = nullptr;
  refused = aclnnSqrtGetWorkspaceSize(scores, MakeTensor({4, 255}, ACL_FLOAT32, nullptr), &workspace, &executor) != 0;
  Check(refused, "sqrt refuses a shape mismatch");

  float bad_beta = 2.0f;
  aclScalar* wrong_beta = aclCreateScalar(&bad_beta, ACL_FLOAT32);
  refused = aclnnSoftplusGetWorkspaceSize(scores, wrong_beta, threshold, scores_out, &workspace, &executor) != 0;
  Check(refused, "softplus refuses beta != 1.0");
  executor = nullptr;
  aclDestroyScalar(wrong_beta);

  // -- gating over pre-normalized scores --------------------------------------
  aclTensor* bias = MakeTensor({256}, ACL_FLOAT32, reinterpret_cast<void*>(arena + 8192));
  aclTensor* y = MakeTensor({4, 6}, ACL_FLOAT32, reinterpret_cast<void*>(arena + 12288));
  aclTensor* idx = MakeTensor({4, 6}, ACL_INT32, reinterpret_cast<void*>(arena + 13500));
  Check(MockValidateGatingForTest(scores, bias, 6, 1, 1, 0, 1, -1, 1.5, y, idx) == 0,
        "gating accepts k=6, groupCount=1, normType=-1, renorm=1, scaling=1.5 over [4, 256]");
  Check(MockValidateGatingForTest(scores, bias, 8, 1, 1, 0, 1, -1, 1.5, y, idx) != 0,
        "gating refuses k=8: " + MockLastContractFailure());
  Check(MockValidateGatingForTest(scores, bias, 6, 1, 2, 0, 1, -1, 1.5, y, idx) != 0,
        "gating refuses groupCount=2");
  Check(MockValidateGatingForTest(scores, bias, 6, 1, 1, 0, 0, -1, 1.5, y, idx) != 0,
        "gating refuses renorm=0 (norm_topk_prob is true)");
  Check(MockValidateGatingForTest(scores, bias, 6, 1, 1, 0, 1, 0, 1.5, y, idx) != 0,
        "gating refuses normType=0 (softmax): the scores arrive pre-normalized");
  Check(MockValidateGatingForTest(scores, bias, 6, 1, 1, 0, 1, -1, 2.0, y, idx) != 0,
        "gating refuses routedScalingFactor=2.0");
  Check(MockValidateGatingForTest(MakeTensor({4, 128}, ACL_FLOAT32, nullptr), bias, 6, 1, 1, 0, 1, -1, 1.5, y,
                                  idx) != 0,
        "gating refuses [4, 128] scores");
  Check(MockValidateGatingForTest(scores, bias, 6, 1, 1, 0, 1, -1, 1.5, MakeTensor({4, 8}, ACL_FLOAT32, nullptr),
                                  idx) != 0,
        "gating refuses yOut [4, 8]");
  Check(MockValidateGatingForTest(scores, bias, 6, 1, 1, 0, 1, -1, 1.5, y,
                                  MakeTensor({4, 6}, ACL_FLOAT32, nullptr)) != 0,
        "gating refuses a float expertIdxOut");

  // -- init routing: the device cumsum groupList --------------------------------
  aclTensor* expert_idx = MakeTensor({4, 6}, ACL_INT32, reinterpret_cast<void*>(arena + 13500));
  aclTensor* group_list = MakeTensor({256}, ACL_INT64, reinterpret_cast<void*>(arena + 20480));
  Check(MockValidateRoutingForTest(expert_idx, 256, group_list) == 0,
        "routing accepts a [256] INT64 cumsum groupList");
  refused = MockValidateRoutingForTest(expert_idx, 256, MakeTensor({255}, ACL_INT64, nullptr)) != 0;
  why = MockLastContractFailure();
  Check(refused && why.find("[expert_num]") != std::string::npos, "routing refuses a [255] groupList: " + why);
  Check(MockValidateRoutingForTest(expert_idx, 256, MakeTensor({256}, ACL_FLOAT32, nullptr)) != 0,
        "routing refuses a non-INT64 groupList");
  // Aliasing: expertIdx bytes and groupList bytes sharing storage.
  aclTensor* aliased = MakeTensor({256}, ACL_INT64, reinterpret_cast<void*>(arena + 13500 + 48));
  Check(MockValidateRoutingForTest(expert_idx, 256, aliased) != 0,
        "routing refuses expertIdx/groupList aliasing");

  // -- grouped matmul: FP4 weights, E8M0 block-32 scales ------------------------
  const ExpertSlotLayout layout = ExpertSlotLayout::ForDeepSeekV4Flash();
  Check(layout.slot_num_bytes() == 13369344,
        "the DSV4-Flash expert slot is exactly 13,369,344 bytes (12.75 MiB)");
  const ExpertRegionSpec& gate_up = layout.region(ExpertRegionId::kGateUpWeight);
  const ExpertRegionSpec& gate_up_scale = layout.region(ExpertRegionId::kGateUpScale);
  std::vector<aclTensor*> weights;
  std::vector<aclTensor*> scales;
  for (int slot = 0; slot < 6; ++slot) {
    weights.push_back(MakeTensor({gate_up.rows, gate_up.cols}, ACL_FP4X2_E2M1,
                                 reinterpret_cast<void*>(arena + layout.slot_num_bytes() * slot)));
    scales.push_back(MakeTensor({gate_up_scale.rows, static_cast<int64_t>(gate_up_scale.stored_cols())},
                                ACL_FLOAT8_E8M0,
                                reinterpret_cast<void*>(arena + layout.slot_num_bytes() * slot +
                                                        gate_up_scale.offset_bytes)));
  }
  aclTensorList* weight_list = aclCreateTensorList(weights.data(), weights.size());
  aclTensorList* scale_list = aclCreateTensorList(scales.data(), scales.size());
  Check(MockValidateGmmForTest(weight_list, scale_list, 3, 0) == 0,
        "grouped matmul accepts FP4 weights + E8M0 scales with splitItem=3, groupType=0");
  Check(MockValidateGmmForTest(weight_list, scale_list, 2, 0) != 0, "grouped matmul refuses splitItem=2");
  Check(MockValidateGmmForTest(weight_list, scale_list, 3, 1) != 0, "grouped matmul refuses groupType=1");

  aclTensor* fp8_weights_raw[] = {MakeTensor({gate_up.rows, gate_up.cols}, ACL_FLOAT8_E4M3FN, nullptr),
                                  MakeTensor({gate_up.rows, gate_up.cols}, ACL_FLOAT8_E4M3FN, nullptr)};
  aclTensorList* fp8_weights = aclCreateTensorList(fp8_weights_raw, 2);
  refused = MockValidateGmmForTest(fp8_weights, scale_list, 3, 0) != 0;
  why = MockLastContractFailure();
  Check(refused && why.find("ACL_FLOAT4_E2M1") != std::string::npos,
        "grouped matmul refuses FP8 weights bound to the FP4 slot layout: " + why);
  aclTensor* fp32_scale_raw[] = {
      MakeTensor({gate_up_scale.rows, static_cast<int64_t>(gate_up_scale.stored_cols())}, ACL_FLOAT32, nullptr),
      MakeTensor({gate_up_scale.rows, static_cast<int64_t>(gate_up_scale.stored_cols())}, ACL_FLOAT32, nullptr)};
  aclTensorList* fp32_scales = aclCreateTensorList(fp32_scale_raw, 2);
  Check(MockValidateGmmForTest(weight_list, fp32_scales, 3, 0) != 0, "grouped matmul refuses FP32 scales");

  // The slot-region offsets the expert pool hands the GEMM must tile the slot.
  for (int slot = 0; slot < 6; ++slot) {
    const uintptr_t slot_base = arena + layout.slot_num_bytes() * static_cast<uintptr_t>(slot);
    for (const ExpertRegionSpec& spec : layout.regions()) {
      Check(MockSpanContains(slot_base + spec.offset_bytes, spec.num_bytes()),
            std::string("slot ") + std::to_string(slot) + " region " + ExpertRegionName(spec.id) +
                " sits inside its registered span at the documented offset");
    }
  }

  for (aclTensor* tensor : weights) {
    aclDestroyTensor(tensor);
  }
  for (aclTensor* tensor : scales) {
    aclDestroyTensor(tensor);
  }
  aclDestroyTensorList(weight_list);
  aclDestroyTensorList(scale_list);
  aclDestroyTensorList(fp8_weights);
  for (aclTensor* tensor : fp8_weights_raw) {
    aclDestroyTensor(tensor);
  }
  aclDestroyTensorList(fp32_scales);
  for (aclTensor* tensor : fp32_scale_raw) {
    aclDestroyTensor(tensor);
  }
  aclDestroyTensor(scores);
  aclDestroyTensor(scores_out);
  aclDestroyTensor(bias);
  aclDestroyTensor(y);
  aclDestroyTensor(idx);
  aclDestroyTensor(expert_idx);
  aclDestroyTensor(group_list);
  aclDestroyTensor(aliased);
  aclDestroyScalar(beta);
  aclDestroyScalar(threshold);
  MockUnregisterSpan(arena);
}

// ---------------------------------------------------------------------------
// 4. ModelConfig: config.json parsing and the DSV4 contract asserts
// ---------------------------------------------------------------------------

// The published DeepSeek-V4 Flash configuration, reduced to the fields the
// runner consumes (structure and values preserved, including the nested
// quantization object and the array inside it).
const char* kDsv4FlashConfigJson =
    "{\n"
    "  \"architectures\": [\"DeepseekV4ForCausalLM\"],\n"
    "  \"expert_dtype\": \"fp4\",\n"
    "  \"hidden_size\": 4096,\n"
    "  \"index_n_heads\": 64,\n"
    "  \"index_topk\": 512,\n"
    "  \"moe_intermediate_size\": 2048,\n"
    "  \"n_routed_experts\": 256,\n"
    "  \"n_shared_experts\": 1,\n"
    "  \"norm_topk_prob\": true,\n"
    "  \"num_attention_heads\": 64,\n"
    "  \"num_experts_per_tok\": 6,\n"
    "  \"num_hidden_layers\": 43,\n"
    "  \"q_lora_rank\": 1024,\n"
    "  \"qk_rope_head_dim\": 64,\n"
    "  \"quantization_config\": {\n"
    "    \"activation_scheme\": \"dynamic\",\n"
    "    \"quant_method\": \"fp8\",\n"
    "    \"scale_fmt\": \"ue8m0\",\n"
    "    \"weight_block_size\": [128, 128]\n"
    "  },\n"
    "  \"routed_scaling_factor\": 1.5,\n"
    "  \"rms_norm_eps\": 1e-06,\n"
    "  \"rope_theta\": 10000,\n"
    "  \"scoring_func\": \"sqrtsoftplus\",\n"
    "  \"swiglu_limit\": 10.0,\n"
    "  \"topk_method\": \"noaux_tc\",\n"
    "  \"vocab_size\": 129280\n"
    "}\n";

std::string WithReplaced(const std::string& text, const std::string& from, const std::string& to) {
  const size_t at = text.find(from);
  DSV4_REQUIRE(at != std::string::npos, "test fixture lost its anchor: " << from);
  return text.substr(0, at) + to + text.substr(at + from.size());
}

void TestModelConfig() {
  Section("model config: config.json parsing, router asserts, binary-contract match");
  const ModelConfig model = ModelConfig::FromJsonText(kDsv4FlashConfigJson, "fixture config.json");
  Check(model.num_hidden_layers == 43 && model.n_routed_experts == 256 && model.num_experts_per_tok == 6 &&
            model.hidden_size == 4096 && model.moe_intermediate_size == 2048 &&
            model.routed_scaling_factor == 1.5,
        "the DSV4-Flash topology parses to the documented values");
  Check(model.topk_method == "noaux_tc" && model.scoring_func == "sqrtsoftplus" && model.norm_topk_prob,
        "the three router contract fields parse");
  Check(model.mla.provenance == GeometryProvenance::kCheckpointConfig && model.mla.qk_rope_head_dim == 64 &&
            model.mla.kv_lora_rank == kDefaultKvLoraRank,
        "qk_rope_head_dim comes from the checkpoint; kv_lora_rank keeps the family default "
        "(the checkpoint does not publish it)");
  Check(model.dense_weight_block == 128 && model.scale_fmt == "ue8m0",
        "the nested quantization_config object is read (block 128, ue8m0 scales)");
  model.AssertMatchesBinaryContract();
  Check(true, "the parsed topology matches the compiled graph geometry exactly");

  // Escape and number grammar, on fields that must survive to the assert.
  const std::string escaped = WithReplaced(std::string(kDsv4FlashConfigJson), "\"noaux_tc\"",
                                           "\"noaux_\\u0074c\"");  // \u0074 = 't'
  Check(ModelConfig::FromJsonText(escaped, "escaped").contains("topk_method"),
        "\\uXXXX escapes decode before the contract asserts see them");
  const std::string exponent =
      WithReplaced(std::string(kDsv4FlashConfigJson), "\"routed_scaling_factor\": 1.5",
                   "\"routed_scaling_factor\": 15e-1");
  Check(ModelConfig::FromJsonText(exponent, "exponent").routed_scaling_factor == 1.5,
        "exponent-notation numbers parse");

  // Every refusal names its field; a silently-defaulted wrong router would
  // reach the device.
  CheckRefuses([] { ModelConfig::FromJsonText(WithReplaced(std::string(kDsv4FlashConfigJson),
                                                            "\"scoring_func\": \"sqrtsoftplus\"",
                                                            "\"scoring_func\": \"sigmoid\""), "t"); },
               "scoring_func sigmoid: the decomposed chain computes sqrt(softplus)");
  CheckRefuses([] { ModelConfig::FromJsonText(WithReplaced(std::string(kDsv4FlashConfigJson),
                                                            "\"topk_method\": \"noaux_tc\"",
                                                            "\"topk_method\": \"group_limited_greedy\""), "t"); },
               "topk_method group_limited_greedy: the gating stage plans no group constraint");
  CheckRefuses([] { ModelConfig::FromJsonText(WithReplaced(std::string(kDsv4FlashConfigJson),
                                                            "\"norm_topk_prob\": true",
                                                            "\"norm_topk_prob\": false"), "t"); },
               "norm_topk_prob false: the gating stage renorm=1");
  CheckRefuses([] { ModelConfig::FromJsonText(WithReplaced(std::string(kDsv4FlashConfigJson),
                                                            "\"num_hidden_layers\": 43", "\"vocab_size\": 129280"),
                                              "t"); },
               "a missing required field (num_hidden_layers) does not silently default");
  CheckRefuses([] { ModelConfig::FromJsonText(WithReplaced(std::string(kDsv4FlashConfigJson),
                                                            "\"num_hidden_layers\": 43",
                                                            "\"num_hidden_layers\": \"43\""), "t"); },
               "a string where a number belongs is a hard type error");
  CheckRefuses([] { ModelConfig::FromJsonText(WithReplaced(std::string(kDsv4FlashConfigJson),
                                                            "\"hidden_size\": 4096",
                                                            "\"hidden_size\": 4097"), "t"); },
               "hidden_size 4097 does not tile the block-32 microscale");
  CheckRefuses([] { ModelConfig::FromJsonText("{not json", "t"); }, "a malformed document is a parse error");

  // Topology that is structurally valid but not what this binary compiled
  // for: refused with the divergence named, before any reservation.
  CheckRefuses([] {
    const ModelConfig other = ModelConfig::FromJsonText(WithReplaced(std::string(kDsv4FlashConfigJson),
                                                                     "\"hidden_size\": 4096",
                                                                     "\"hidden_size\": 2048"), "t");
    other.AssertMatchesBinaryContract();
  }, "a structurally valid but divergent topology is refused by the binary-contract check");

  // A default-constructed ModelConfig IS the compiled contract.
  ModelConfig{}.AssertMatchesBinaryContract();
  Check(true, "the default-constructed contract (no config.json anywhere) matches the binary");
}

// ---------------------------------------------------------------------------
// 5. The MoE block seam: slot contracts and the lattice skeleton
// ---------------------------------------------------------------------------

void TestMoeBlockSeam(IDeviceAllocator& allocator, IStreamEngine& streams, const RuntimeConfig& config) {
  Section("IRoutedMoeBlock: the slot byte contract and the lattice skeleton");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForDeepSeekV4Flash();
  Check(Lattice24MoeBlock::SlotBytes() == 7077888,
        "the 2-bit Leech slot is exactly 7,077,888 bytes (6.75 MiB): weights halve, E8M0 scales do not");
  Check(Lattice24MoeBlock::SlotBytes() * 2 > layout.slot_num_bytes() &&
            static_cast<double>(Lattice24MoeBlock::SlotBytes()) / static_cast<double>(layout.slot_num_bytes()) < 0.55,
        "the lattice slot is ~50% of the FP4 slot (weights halve, the 0.75 MiB of scales do not)");

  Lattice24MoeBlock lattice;
  CheckRefuses([&] {
    StaticOpSlotTable table(8, "lattice-test");
    StaticArenaManager probe(allocator, streams, config);
    ExpertSlotAddresses addresses;
    lattice.PlanStages(table, probe, addresses);
  }, "planning the lattice skeleton names the missing aclnnLatticeUnpackAndGroupedMatmul");
  Check(lattice.DescribeBackend().find("SKELETON") != std::string::npos,
        "the lattice backend describes itself as a skeleton");
}

// ---------------------------------------------------------------------------
// 6. The 43-layer decode graph, end to end, at full coverage
// ---------------------------------------------------------------------------

// A weight source that moves no byte: the mock's transfers are interval
// checks, so the graph runs over pure address bookkeeping.
class SymbolicWeightSource : public WeightByteSource {
 public:
  SymbolicWeightSource(int64_t num_layers, int64_t num_experts)
      : num_layers_(num_layers), num_experts_(num_experts) {}

  const char* source_name() const override { return "symbolic"; }
  bool Contains(int32_t layer, int32_t expert) const override {
    return layer >= 0 && layer < num_layers_ && expert >= 0 && expert < num_experts_;
  }
  void ReadExpertSlotRange(uint8_t*, size_t destination_capacity, size_t slot_offset, size_t count, int32_t layer,
                           int32_t expert) override {
    RefuseIfClosed("ReadExpertSlotRange");
    DSV4_REQUIRE(Contains(layer, expert),
                 "expert (layer=" << layer << ", id=" << expert << ") is outside the symbolic source coverage");
    DSV4_REQUIRE(slot_offset + count <= layout_.slot_num_bytes(),
                 "symbolic read leaves the " << layout_.slot_num_bytes() << "-byte slot");
    DSV4_REQUIRE(count <= destination_capacity, "symbolic read exceeds the destination capacity");
    bytes_read_ += count;
    ++read_requests_;
  }
  bool HasNamed(const std::string&) const override { return true; }
  void ReadNamed(const std::string& name, uint8_t*, size_t destination_capacity, size_t byte_offset,
                 size_t count) override {
    RefuseIfClosed("ReadNamed");
    (void)byte_offset;
    DSV4_REQUIRE(count <= destination_capacity, "named read '" << name << "' exceeds the destination capacity");
    bytes_read_ += count;
    ++read_requests_;
  }
  size_t NamedByteSize(const std::string&) const override { return 0; }  // 0 = do not size-check
  void Close() override { closed_ = true; }

 private:
  int64_t num_layers_;
  int64_t num_experts_;
};

// The symbolic oracle for the two forced device-to-host reads: the per-layer
// top-6 expert ids and the greedy token. Without operators that compute, the
// graph's own outputs are fabricated here -- deterministically, in range.
void SeedDeviceToHost(void* destination, size_t count) {
  if (count == sizeof(int32_t) * static_cast<size_t>(kNumExpertsPerTok)) {
    int32_t* ids = static_cast<int32_t*>(destination);
    for (int64_t index = 0; index < kNumExpertsPerTok; ++index) {
      ids[index] = static_cast<int32_t>(index);
    }
    return;
  }
  if (count == sizeof(int64_t)) {
    *static_cast<int64_t*>(destination) = 0;  // greedy token id 0
  }
}

size_t ReadResidentKb() {
  std::FILE* file = std::fopen("/proc/self/status", "r");
  if (file == nullptr) {
    return 0;
  }
  char line[256];
  size_t kb = 0;
  while (std::fgets(line, sizeof(line), file) != nullptr) {
    unsigned long long value = 0;
    if (std::sscanf(line, "VmRSS: %llu kB", &value) == 1) {
      kb = static_cast<size_t>(value);
      break;
    }
  }
  std::fclose(file);
  return kb;
}

void TestFullPipeline() {
  Section("the 43-layer decode graph: build, plan, decode, residency (full 11,008 coverage)");
  MockResetAllocatorForTest();
  MockSetReportedHbm(64ull << 30);

  OpTable ops;
  Check(ops.runtime_reachable(), "the operator table resolves from libopapi_mock via dlsym");
  for (OpId id : {OpId::kSoftplus, OpId::kSqrt, OpId::kMoeGatingTopKV2, OpId::kMoeInitRoutingV4,
                  OpId::kGroupedMatmulV5, OpId::kFusedInferAttentionScoreV5, OpId::kQuantMatmulV5,
                  OpId::kGroupedMatmulSwigluQuantV2}) {
    Check(ops.available(id), std::string("resolved: ") + OpName(id));
  }

  AclDeviceOps device(0);  // every aclrt* call lands in the symbolic mock
  Check(std::string(device.soc_name()).find("Mock") != std::string::npos,
        "the device backend reports the mock SoC");

  RuntimeConfig seam_config;
  seam_config.block_size = 128;
  seam_config.max_context_len = 256;
  TestMoeBlockSeam(device, device, seam_config);

  const ExpertSlotLayout layout = ExpertSlotLayout::ForDeepSeekV4Flash();
  const size_t backbone_bytes = Dsv4Pipeline::BackboneDeviceBytes(MlaGeometry(), 128, 256);
  const int64_t slots = ExclusiveExpertManager::PlanDeviceSlots(
      64ull << 30, backbone_bytes, layout.slot_num_bytes(), kTotalRoutedExperts, kDeviceReserveBytes,
      kTransferChunkBytes, kNumExpertsPerTok, -1);
  Check(slots > kNumExpertsPerTok && slots < kTotalRoutedExperts,
        "the slot planner fits K between one top-k and full residency on the reported 64 GiB");

  ExclusiveExpertManager::Options options;
  options.routed_coverage = kTotalRoutedExperts;  // FULL: 11,008 experts as spans, not bytes
  options.device_slots = slots;
  options.transfer_chunk_bytes = 512 * 1024;  // keeps the pinned transit scratch in the real tier
  options.host_available_bytes = 256ull << 30;  // the host half is symbolic; skip the meminfo guard
  ExclusiveExpertManager experts(device, device, layout, options);
  std::printf("%s", experts.DescribeHierarchy().c_str());
  Check(experts.host_slot_count() + experts.device_slot_count() == kTotalRoutedExperts,
        "|Set_Device| + |Set_Host| = 11,008 with zero physical bytes behind them");

  SymbolicWeightSource source(kNumLayers, kNumRoutedExperts);
  RuntimeConfig config;
  config.synthetic_weights = true;
  config.block_size = 128;
  config.max_context_len = 256;
  // The router is the runner's shared dependency now: the pipeline and the
  // injected MoE block both consume it.
  MoeRouterEngine router(device, device);
  Dsv4Pipeline pipeline(device, device, ops, experts, router, config);
  pipeline.Build(source);  // reserves, ingests, descriptors, PLANS EVERY STAGE
  experts.Ingest(source, {});

  Check(pipeline.arena_manager().arena().sealed(), "the arena is sealed after Build");
  Check(pipeline.arena_manager().arena().workspace_bytes() > 0, "the shared workspace holds the plan high-water mark");
  Check(pipeline.moe_block().GetExpertSlotBytes() == layout.slot_num_bytes(),
        "the injected standard block carries the exact 13,369,344-byte FP4/E8M0 slot contract");
  const std::string stage_report = pipeline.DescribeStages();
  Check(stage_report.find("expert_gemm1") != std::string::npos && stage_report.find("expert_combine") != std::string::npos,
        "the MoE block planned its expert stages into the pipeline's shared stage table");
  Check(stage_report.find("GroupedMatmulSwigluQuantV2") != std::string::npos,
        "the fused grouped-GEMM entry point appears in the planned stage list (fused default path)");
  const size_t planned_stages = [&] {
    const std::string report = pipeline.DescribeStages();
    return report.size();  // non-empty iff the stages planned
  }();
  Check(planned_stages > 0, "every stage of the 43-layer graph planned against the mock contracts");
  std::printf("%s", pipeline.DescribeStages().c_str());

  MockSetD2HSeed(&SeedDeviceToHost);
  pipeline.DecodeStep(7, 0);   // prompt token 7 at position 0
  const int32_t token = pipeline.ReadArgmaxToken();
  Check(token >= 0 && token < kVocabSize, "a seeded readback returns a valid greedy token id");
  pipeline.DecodeStep(token, 1);  // one decode step at position 1
  const int32_t next = pipeline.ReadArgmaxToken();
  Check(next >= 0 && next < kVocabSize, "the second readback is valid too");
  MockSetD2HSeed(nullptr);

  experts.ValidateResidency();
  Check(true, "the exclusive residency invariant holds after two 43-layer steps");

  const StepCounters& counters = pipeline.counters();
  Check(counters.steps == 2 && counters.layers == 2 * kNumLayers,
        "two decode steps covered all 43 layers each (86 layer executions)");
  Check(counters.device_allocations_in_step == 0, "zero device allocations inside a step");
  Check(counters.descriptors_built_in_step == 0, "zero descriptors built inside a step");
  Check(counters.host_synchronizations == 2 * (kNumLayers + 1),
        "exactly the forced one-per-MoE-layer readback plus one per step");
  Check(counters.expert_slot_misses > 0 && counters.expert_slot_hits > 0,
        "the seeded routing produced both hits and misses across the swap engine");

  const MockMemoryStats& stats = MockMemoryStatistics();
  Check(stats.rejected_operations == 0,
        "ZERO rejected operations: every DMA, memset and SetTensorAddr stayed inside a registered span");
  Check(stats.memcpy_checks > 0 && stats.setaddr_checks > 0,
        "the interval registry actually exercised transfers (" + std::to_string(stats.memcpy_checks) +
            " memcpy checks, " + std::to_string(stats.setaddr_checks) + " SetTensorAddr checks)");
  Check(stats.symbolic_device_bytes > 45ull << 30,
        "the symbolic device tier alone carries the K-slot pool and the arena (" +
            std::to_string(stats.symbolic_device_bytes >> 30) + " GiB of spans; the host half adds " +
            std::to_string(((static_cast<uint64_t>(experts.host_slot_count()) * layout.slot_num_bytes()) >> 30)) +
            " GiB more, all at zero physical cost)");
  std::printf("  slot-map index mismatches (derive-and-verify tally): %" PRIu64 "\n", stats.slot_map_mismatches);
  const size_t rss_kb = ReadResidentKb();
  Check(rss_kb < 512 * 1024, "physical RSS stayed under 512 MiB (read " + std::to_string(rss_kb) + " kB)");
}

}  // namespace
}  // namespace ascend_moe

int main() {
  using namespace ascend_moe;
  std::printf("dsv4_mock_test -- zero-NPU, zero-allocation contract suite over libopapi_mock\n");
  try {
    TestIntervalRegistry();
    TestDescriptorEngine();
    TestOperatorContracts();
    TestModelConfig();
    TestFullPipeline();
  } catch (const std::exception& error) {
    std::printf("\nunexpected exception: %s\n", error.what());
    ++g_failures;
  }
  const ascend_moe::mock::MockMemoryStats& stats = ascend_moe::mock::MockMemoryStatistics();
  std::printf("\nmock runtime tally: %" PRIu64 " device spans (%" PRIu64 " GiB symbolic), %" PRIu64
              " real host bytes,\n  %" PRIu64 " memcpy checks, %" PRIu64 " setaddr checks, %" PRIu64
              " refusals, %" PRIu64 " slot-map notes\n",
              stats.device_allocations, stats.symbolic_device_bytes >> 30, stats.real_host_bytes,
              stats.memcpy_checks, stats.setaddr_checks, stats.rejected_operations, stats.slot_map_mismatches);
  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
