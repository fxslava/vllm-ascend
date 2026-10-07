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

#include "moe/memory/expert_layout.hpp"

#include <iomanip>
#include <sstream>

#include "moe/core/error.hpp"
#include "moe/core/config.hpp"

namespace ascend_moe {

size_t AlignUp(size_t value, size_t alignment) { return (value + alignment - 1) & ~(alignment - 1); }

const char* ExpertRegionName(ExpertRegionId id) {
  switch (id) {
    case ExpertRegionId::kGateUpWeight:
      return "w13_weight";
    case ExpertRegionId::kGateUpScale:
      return "w13_scale";
    case ExpertRegionId::kDownWeight:
      return "w2_weight";
    case ExpertRegionId::kDownScale:
      return "w2_scale";
    default:
      return "<invalid>";
  }
}

size_t ExpertRegionSpec::stored_cols() const {
  if (kind == ExpertRegionKind::kPackedFp4) {
    DSV4_REQUIRE(cols % kFp4ElementsPerByte == 0,
                 ExpertRegionName(id) << ": logical cols " << cols << " is not divisible by "
                                      << kFp4ElementsPerByte);
    return static_cast<size_t>(cols / kFp4ElementsPerByte);
  }
  DSV4_REQUIRE(cols % kRoutedScaleBlock == 0, ExpertRegionName(id)
                                                  << ": logical cols " << cols
                                                  << " is not divisible by the microscale block "
                                                  << kRoutedScaleBlock);
  return static_cast<size_t>(cols / kRoutedScaleBlock);
}

ExpertSlotLayout ExpertSlotLayout::ForGeometry(int64_t hidden_size, int64_t moe_intermediate_size) {
  DSV4_REQUIRE(hidden_size > 0 && moe_intermediate_size > 0,
               "expert geometry needs a positive hidden/intermediate, got " << hidden_size << "/"
                                                                            << moe_intermediate_size);

  ExpertSlotLayout layout;
  layout.hidden_size_ = hidden_size;
  layout.moe_intermediate_size_ = moe_intermediate_size;

  // Gate and up fused on the output axis; down is the transpose geometry.
  const int64_t gate_up_rows = 2 * moe_intermediate_size;
  struct Declaration {
    ExpertRegionId id;
    ExpertRegionKind kind;
    int64_t rows;
    int64_t cols;
  };
  const Declaration declarations[] = {
      {ExpertRegionId::kGateUpWeight, ExpertRegionKind::kPackedFp4, gate_up_rows, hidden_size},
      {ExpertRegionId::kGateUpScale, ExpertRegionKind::kE8m0Scale, gate_up_rows, hidden_size},
      {ExpertRegionId::kDownWeight, ExpertRegionKind::kPackedFp4, hidden_size, moe_intermediate_size},
      {ExpertRegionId::kDownScale, ExpertRegionKind::kE8m0Scale, hidden_size, moe_intermediate_size},
  };

  size_t cursor = 0;
  layout.regions_.reserve(static_cast<size_t>(ExpertRegionId::kRegionCount));
  for (const Declaration& declaration : declarations) {
    ExpertRegionSpec spec;
    spec.id = declaration.id;
    spec.kind = declaration.kind;
    spec.rows = declaration.rows;
    spec.cols = declaration.cols;
    spec.offset_bytes = AlignUp(cursor, kSlotRegionAlignBytes);
    cursor = spec.offset_bytes + spec.num_bytes();
    layout.regions_.push_back(spec);
  }
  layout.slot_num_bytes_ = AlignUp(cursor, kSlotRegionAlignBytes);
  layout.Validate();
  return layout;
}

ExpertSlotLayout ExpertSlotLayout::ForDeepSeekV4Flash() {
  return ForGeometry(kHiddenSize, kMoeIntermediateSize);
}

const ExpertRegionSpec& ExpertSlotLayout::region(ExpertRegionId id) const {
  for (const ExpertRegionSpec& spec : regions_) {
    if (spec.id == id) {
      return spec;
    }
  }
  throw Dsv4Error(std::string("expert slot layout has no region ") + ExpertRegionName(id));
}

uint8_t* ExpertSlotLayout::RegionPointer(uint8_t* slot_base, ExpertRegionId id) const {
  return slot_base + region(id).offset_bytes;
}

const uint8_t* ExpertSlotLayout::RegionPointer(const uint8_t* slot_base, ExpertRegionId id) const {
  return slot_base + region(id).offset_bytes;
}

void ExpertSlotLayout::Validate() const {
  DSV4_REQUIRE(regions_.size() == static_cast<size_t>(ExpertRegionId::kRegionCount),
               "expert slot layout declares " << regions_.size() << " regions, expected "
                                              << static_cast<int>(ExpertRegionId::kRegionCount));
  size_t previous_end = 0;
  for (const ExpertRegionSpec& spec : regions_) {
    DSV4_REQUIRE(spec.offset_bytes % kSlotRegionAlignBytes == 0,
                 ExpertRegionName(spec.id) << " starts at " << spec.offset_bytes << ", which is not "
                                           << kSlotRegionAlignBytes << "-byte aligned");
    DSV4_REQUIRE(spec.offset_bytes >= previous_end,
                 ExpertRegionName(spec.id) << " at " << spec.offset_bytes << " overlaps the region ending at "
                                           << previous_end);
    DSV4_REQUIRE(spec.num_bytes() > 0, ExpertRegionName(spec.id) << " is empty");
    previous_end = spec.offset_bytes + spec.num_bytes();
  }
  DSV4_REQUIRE(previous_end <= slot_num_bytes_,
               "expert slot regions end at " << previous_end << " beyond the " << slot_num_bytes_ << "-byte slot");
  DSV4_REQUIRE(slot_num_bytes_ % kSlotRegionAlignBytes == 0,
               "expert slot size " << slot_num_bytes_ << " is not " << kSlotRegionAlignBytes << "-byte aligned");
  // Every FP4 / E8M0 region is itself a multiple of the alignment at these
  // geometries, so the slot must tile exactly. A gap would mean bytes belonging
  // to no region, reachable from both the free list and a parameter view.
  DSV4_REQUIRE(previous_end == slot_num_bytes_,
               "expert slot has " << (slot_num_bytes_ - previous_end) << " padding bytes belonging to no region");
}

std::string ExpertSlotLayout::DescribeTable() const {
  std::ostringstream out;
  out << "expert slot: hidden=" << hidden_size_ << " intermediate=" << moe_intermediate_size_
      << " total=" << slot_num_bytes_ << " bytes (" << std::fixed << std::setprecision(2)
      << static_cast<double>(slot_num_bytes_) / (1024.0 * 1024.0) << " MiB)\n";
  for (const ExpertRegionSpec& spec : regions_) {
    out << "  " << std::left << std::setw(12) << ExpertRegionName(spec.id) << std::right
        << " kind=" << (spec.kind == ExpertRegionKind::kPackedFp4 ? "fp4_e2m1" : "e8m0    ") << " shape=["
        << std::setw(5) << spec.rows << "," << std::setw(5) << spec.cols << "]" << " stored=[" << std::setw(5)
        << spec.rows << "," << std::setw(5) << spec.stored_cols() << "]" << " offset=" << std::setw(9)
        << spec.offset_bytes << " bytes=" << std::setw(9) << spec.num_bytes() << "\n";
  }
  return out.str();
}

}  // namespace ascend_moe
