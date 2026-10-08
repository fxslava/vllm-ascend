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

// StandardAclnnMoeBlock: the stock ACLNN V5 expert path behind
// IRoutedMoeBlock (LSP with today's behavior -- the stages, their order and
// their byte contract are the ones Dsv4Pipeline used to plan inline).
//
//   fused:       aclnnGroupedMatmulSwigluQuantV2  (GEMM1 + clamped SwiGLU
//                limit 10.0 + MX requant, one op)
//   decomposed:  aclnnGroupedMatmulV5 -> aclnnSwigluMxQuant
//   both:        aclnnGroupedMatmulV5  (GEMM2)
//                aclnnMatmul           (the [1,6]x[6,hidden] combine)
//
// The 13,369,344-byte FP4/E8M0 slot contract is inherited from the
// ExpertSlotLayout the exclusive hierarchy was constructed with.

#pragma once

#include <cstdint>
#include <string>

#include "moe/core/config.hpp"
#include "moe/core/op_table.hpp"
#include "moe/memory/exclusive_staging.hpp"
#include "moe/pipeline/moe_router_engine.hpp"
#include "moe/pipeline/routed_moe_block.hpp"

namespace ascend_moe {

class StandardAclnnMoeBlock : public IRoutedMoeBlock {
 public:
  // Services, fixed for the pipeline lifetime. `experts` supplies both the
  // slot layout (byte contract, region addresses) and the residency plan at
  // execution time; `router` performs the dropless dispatch that produces
  // the context's expanded activations.
  StandardAclnnMoeBlock(const OpTable& ops, MoeRouterEngine& router, ExclusiveExpertManager& experts,
                        const RuntimeConfig& config);

  // ---- IRoutedMoeBlock ----
  void PlanStages(StaticOpSlotTable& op_table, StaticArenaManager& arena,
                  const ExpertSlotAddresses& slot_addrs) override;
  void ExecuteMoe(const MoeDispatchContext& ctx, IStreamEngine& stream_engine) override;
  size_t GetExpertSlotBytes() const override;
  uint64_t launches() const override { return launches_; }
  std::string DescribeBackend() const override;

 private:
  // Repoints the planned GEMM weight/scale tensor lists at this layer's six
  // resident slots.
  void BindExpertWeights(const LayerSwapPlan& plan);
  void Launch(PipelineStage& entry, DeviceStream stream);

  const OpTable& ops_;
  MoeRouterEngine& router_;
  ExclusiveExpertManager& experts_;
  RuntimeConfig config_;

  // Set at PlanStages; ExecuteMoe replays against them.
  StaticArenaManager* arena_ = nullptr;
  StaticOpSlotTable* table_ = nullptr;

  uint64_t launches_ = 0;
};

}  // namespace ascend_moe
