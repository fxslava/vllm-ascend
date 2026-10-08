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

// Lattice24MoeBlock: SKELETON for the 2-bit Leech-lattice (Lambda_24)
// quantized expert backend.
//
// THE IDEA
// --------
// Leech-lattice quantization stores each expert weight block as an index
// into a 2-bit codebook derived from the 24-dimensional Leech lattice's
// shell structure, replacing the FP4 (E2M1) payload's 4 bits per element
// with 2. Decompression ("unpack") is a small table fold that an Ascend C
// kernel can run on the AIV (vector) decode stream, in front of the cube:
//
//   aclnnLatticeUnpackAndGroupedMatmul -- the custom op this backend exists
//   to dispatch. It is NOT in any CANN toolkit today; there is no OpTable
//   entry, no kernel directory, no GetWorkspaceSize prototype. When the
//   Ascend C kernel lands, register it in op_table.cpp and implement
//   PlanStages / ExecuteMoe here.
//
// SLOT ARITHMETIC (hidden 4096, intermediate 2048)
// ------------------------------------------------
//   gate_up weight  [2*2048, 4096] @ 2 bit  =  4,194,304 B  (FP4: 8 MiB)
//   down weight     [4096, 2048]   @ 2 bit  =  2,097,152 B  (FP4: 4 MiB)
//   E8M0 block-32 microscales, per element block:   =    786,432 B (unchanged)
//   slot total, 128-byte aligned                    =  7,077,888 B = 6.75 MiB
//
// That is ~50.6% of the FP4 slot (13,369,344 B): the weight payload halves,
// the microscales do not (they are per element block along the reduction
// axis, independent of the storage width). At the full 11,008-expert
// coverage the routed set shrinks from 137.07 GiB to ~71.7 GiB -- which is
// the point: K, the resident slot count, roughly doubles on the same HBM.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "moe/core/config.hpp"
#include "moe/core/error.hpp"
#include "moe/pipeline/routed_moe_block.hpp"

namespace ascend_moe {

class Lattice24MoeBlock : public IRoutedMoeBlock {
 public:
  // The custom operator this backend dispatches instead of the grouped GEMM
  // family. A name today; an OpTable entry when the kernel exists.
  static constexpr const char* kLatticeUnpackGemmOp = "aclnnLatticeUnpackAndGroupedMatmul";

  static size_t SlotBytes() {
    // 2-bit payload: four elements per stored byte.
    constexpr int64_t kElementsPerByte = 4;
    const int64_t gate_up_elements = 2 * kMoeIntermediateSize * kHiddenSize;
    const int64_t down_elements = kHiddenSize * kMoeIntermediateSize;
    const size_t packed =
        static_cast<size_t>((gate_up_elements + down_elements) / kElementsPerByte);
    // Microscales: one E8M0 byte per block-32 elements along the reduction
    // axis of each region, exactly the FP4 layout's scale regions.
    const size_t scales = static_cast<size_t>((gate_up_elements + down_elements) / kRoutedScaleBlock);
    const size_t total = packed + scales;
    // kSlotRegionAlignBytes is a power of two, so one round-up covers both
    // the region offsets and the slot total.
    return (total + kSlotRegionAlignBytes - 1) & ~(kSlotRegionAlignBytes - 1);
  }

  // Not executable: refusing at plan time is the honest state of a skeleton,
  // and it fires before any reservation so a mis-selected backend costs
  // nothing. The failure names the hook the kernel must register under.
  void PlanStages(StaticOpSlotTable&, StaticArenaManager&, const ExpertSlotAddresses&) override {
    throw Dsv4Error(std::string("Lattice24MoeBlock is a skeleton: ") + kLatticeUnpackGemmOp +
                    " is not registered in this toolkit's OpTable. The 2-bit Leech-lattice kernel (AIV decode "
                    "stream) does not exist yet; build with the standard aclnn backend.");
  }

  void ExecuteMoe(const MoeDispatchContext&, IStreamEngine&) override {
    throw Dsv4Error("Lattice24MoeBlock cannot execute: its unpack-and-grouped-GEMM kernel is not implemented");
  }

  size_t GetExpertSlotBytes() const override { return SlotBytes(); }

  std::string DescribeBackend() const override {
    return "lattice Lambda-24 2-bit (SKELETON, " + std::to_string(SlotBytes()) +
           " B/expert vs standard 13,369,344 B -- kernel pending)";
  }
};

}  // namespace ascend_moe
