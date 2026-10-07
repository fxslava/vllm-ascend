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

// Byte-level layout of one isomorphic routed-expert slot.
//
// This is the C++ port of `tools/dsv4_moe_runtime/core/layout.py`
// (ExpertTensorLayout, PACKED_FP4_KIND + E8M0_SCALE_KIND) with one deliberate
// difference, documented below. Both produce a **13,369,344-byte (12.75 MiB)**
// slot at hidden 4096 / intermediate 2048, and both keep every region at a
// 128-byte aligned offset.
//
// THE DIFFERENCE: gate and up are stored as one fused `w13` region
// --------------------------------------------------------------------------
// The Python layout emits w1, w2, w3 in that order, three separate weight
// regions. This layout emits `w13` (gate stacked above up, 2*intermediate rows)
// and then `w2`. The reason is the expert GEMM:
//
//   * `aclnnGroupedMatmulSwigluQuantV2` takes gate and up as one weight whose
//     output axis is 2*intermediate. Interleaving w2 between them -- the Python
//     order -- makes that single tensor unexpressible without a runtime
//     relayout.
//   * Block-32 microscales stay blocked along `cols`, which is the *reduction*
//     axis in both regions, so the MX contract is unchanged by the regrouping.
//
// The fusion is an ingestion-time concatenation (`WeightByteSource` writes
// gate_proj into rows [0, inter) and up_proj into rows [inter, 2*inter)), so it
// costs nothing at decode time and the slot total is byte-identical to the
// Python layout's.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ascend_moe {

enum class ExpertRegionKind {
  kPackedFp4,  // 2 E2M1 nibbles per stored byte
  kE8m0Scale,  // 1 E8M0 byte per `kRoutedScaleBlock` elements
};

// Stable identity of a region inside a slot. The ingestion path and the GEMM
// descriptors both address regions by this, never by index.
enum class ExpertRegionId {
  kGateUpWeight,  // w13: [2 * intermediate, hidden] packed FP4
  kGateUpScale,   // w13 microscales:  [2 * intermediate, hidden / 32] E8M0
  kDownWeight,    // w2:  [hidden, intermediate] packed FP4
  kDownScale,     // w2 microscales:   [hidden, intermediate / 32] E8M0
  kRegionCount,
};

const char* ExpertRegionName(ExpertRegionId id);

struct ExpertRegionSpec {
  ExpertRegionId id = ExpertRegionId::kGateUpWeight;
  ExpertRegionKind kind = ExpertRegionKind::kPackedFp4;
  int64_t rows = 0;         // output axis (N)
  int64_t cols = 0;         // logical element count along the reduction axis (K)
  size_t offset_bytes = 0;  // 128-byte aligned offset inside the slot

  // Bytes per row of the region; the second dimension of the uint8 view.
  size_t stored_cols() const;
  size_t num_bytes() const { return static_cast<size_t>(rows) * stored_cols(); }
  bool is_scale() const { return kind == ExpertRegionKind::kE8m0Scale; }
};

class ExpertSlotLayout {
 public:
  // DSV4 Flash geometry by default; parametric so a small geometry can be
  // exercised by the contract test.
  static ExpertSlotLayout ForGeometry(int64_t hidden_size, int64_t moe_intermediate_size);
  static ExpertSlotLayout ForDeepSeekV4Flash();

  size_t slot_num_bytes() const { return slot_num_bytes_; }
  int64_t hidden_size() const { return hidden_size_; }
  int64_t moe_intermediate_size() const { return moe_intermediate_size_; }

  const std::vector<ExpertRegionSpec>& regions() const { return regions_; }
  const ExpertRegionSpec& region(ExpertRegionId id) const;

  // Base address of one region inside a slot whose first byte is `slot_base`.
  uint8_t* RegionPointer(uint8_t* slot_base, ExpertRegionId id) const;
  const uint8_t* RegionPointer(const uint8_t* slot_base, ExpertRegionId id) const;

  // Throws unless every region offset is kSlotRegionAlignBytes-aligned, the
  // regions tile the slot without overlap, and the total is aligned. Called by
  // the factory and re-checked by the contract test.
  void Validate() const;

  std::string DescribeTable() const;

 private:
  ExpertSlotLayout() = default;

  int64_t hidden_size_ = 0;
  int64_t moe_intermediate_size_ = 0;
  size_t slot_num_bytes_ = 0;
  std::vector<ExpertRegionSpec> regions_;
};

// Round `value` up to `alignment`, which must be a power of two.
size_t AlignUp(size_t value, size_t alignment);

}  // namespace ascend_moe
