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

// StaticOpSlotTable: the planned stage collection of the decode graph (SRP).
//
// Extracted from Dsv4Pipeline so that more than one planner can contribute
// stages to ONE address-swappable table: the orchestrator plans its stages,
// and the injected IRoutedMoeBlock plans the expert GEMM stages into the
// same table, which is what keeps "which ops run" decoupled from "what
// sequence runs them" without a second bookkeeping structure.
//
// The capacity is fixed at construction and the vector is born at full
// size, because PipelineStage holds StaticOpSlot, which is move-disabled --
// not even std::vector::reserve compiles for it (libstdc++'s reallocation
// path move-empties a zero-size vector). Planning only fills slots in
// place; the table is never resized or copied after construction.

#pragma once

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "moe/core/error.hpp"
#include "moe/core/op_table.hpp"

namespace ascend_moe {

class StaticOpSlotTable {
 public:
  explicit StaticOpSlotTable(size_t capacity, const char* owner)
      : stages_(capacity), owner_(owner != nullptr ? owner : "stage table") {}

  // Records a new stage to be planned. Throws past the capacity, naming the
  // owner, so an over-full graph is one clear message.
  PipelineStage& Add(const char* name, OpId op) {
    DSV4_REQUIRE(count_ < stages_.size(), owner_ << " stage capacity exceeded at " << name);
    PipelineStage& entry = stages_[count_++];
    entry.name = name;
    entry.op = op;
    return entry;
  }

  const PipelineStage* Find(const char* name) const {
    for (size_t i = 0; i < count_; ++i) {
      if (std::strcmp(stages_[i].name, name) == 0) {
        return &stages_[i];
      }
    }
    return nullptr;
  }

  // The only stage named `name`; throws when the graph never planned it.
  // The non-const overload serves the decode loop (address swaps, launches).
  PipelineStage& stage(const char* name) {
    for (size_t i = 0; i < count_; ++i) {
      if (std::strcmp(stages_[i].name, name) == 0) {
        return stages_[i];
      }
    }
    throw Dsv4Error(std::string(owner_) + " has no stage named " + name);
  }

  const PipelineStage& stage(const char* name) const {
    const PipelineStage* found = Find(name);
    DSV4_REQUIRE(found != nullptr, owner_ << " has no stage named " << name);
    return *found;
  }

  size_t size() const { return count_; }
  size_t capacity() const { return stages_.size(); }
  const PipelineStage& operator[](size_t index) const { return stages_[index]; }

 private:
  // Born at full capacity; only the first `count_` entries are live.
  std::vector<PipelineStage> stages_;
  size_t count_ = 0;
  const char* owner_;
};

}  // namespace ascend_moe
