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

// Static pre-allocation: the activation arena, the paged KV cache, the ACLNN
// workspaces and every `aclTensor` / `aclTensorList` / `aclIntArray` handle the
// pipeline will ever use.
//
// THE INVARIANT THIS CLASS EXISTS TO ENFORCE
// ------------------------------------------
// "The decode loop contains zero malloc, zero descriptor creations, and zero
// CPU-device synchronization calls inside the 43-layer loop."
//
// That is not asserted in a comment here, it is latched. The arena has three
// phases and the transitions are one-way:
//
//   1. RESERVE   `Reserve()` records a named, aligned request. Nothing is
//                allocated and no address exists yet.
//   2. BUILD     `Commit()` performs exactly ONE `aclrtMalloc` for the whole
//                arena and resolves every reservation to an address.
//                Descriptors are created, operators are planned,
//                `NoteWorkspace()` collects the high-water mark, and
//                `CommitWorkspace()` makes the second and last device
//                allocation.
//   3. SEALED    `Seal()` latches. `Reserve`, `CreateTensor`,
//                `CreateTensorList`, `CreateIntArray` and `CommitWorkspace` all
//                throw from here on, and `AssertWorkspaceFits` refuses a plan
//                that would need to grow.
//
// A decode step that tried to allocate or to build a descriptor therefore fails
// loudly instead of quietly costing time, and `dsv4_contract_smoke` verifies
// each refusal.

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "dsv4_aclnn_v5.hpp"
#include "dsv4_config.hpp"
#include "dsv4_device_ops.hpp"

namespace vllm_ascend {
namespace dsv4 {

// Handle returned by Reserve; resolves to an address only after Commit.
using ArenaHandle = size_t;
inline constexpr ArenaHandle kInvalidArenaHandle = static_cast<ArenaHandle>(-1);

struct ArenaReservation {
  const char* name = nullptr;
  size_t offset = 0;
  size_t bytes = 0;
  size_t align = kArenaAlignBytes;
};

struct DescriptorRecord {
  const char* label = nullptr;
  enum class Kind { kTensor, kTensorList, kIntArray, kScalar } kind = Kind::kTensor;
  int32_t dtype = 0;
  int64_t element_count = 0;
  const void* address = nullptr;
};

class DSV4StaticMemoryArena {
 public:
  explicit DSV4StaticMemoryArena(DeviceOps& device);
  ~DSV4StaticMemoryArena();

  DSV4StaticMemoryArena(const DSV4StaticMemoryArena&) = delete;
  DSV4StaticMemoryArena& operator=(const DSV4StaticMemoryArena&) = delete;

  // ---- phase 1: RESERVE -------------------------------------------------

  // `name` must be a string literal: it is kept by pointer for the ledger.
  ArenaHandle Reserve(const char* name, size_t bytes, size_t align = kArenaAlignBytes);
  size_t reserved_bytes() const { return cursor_; }

  // ---- phase 2: BUILD ---------------------------------------------------

  void Commit();
  bool committed() const { return arena_ != nullptr; }
  void* Address(ArenaHandle handle) const;
  template <typename T>
  T* AddressAs(ArenaHandle handle) const {
    return static_cast<T*>(Address(handle));
  }
  size_t Bytes(ArenaHandle handle) const;

  // Descriptors. Every one is created here, once, and destroyed by the
  // destructor; nothing else in the runner owns an acl handle.
  aclTensor* CreateTensor(const char* label, const std::vector<int64_t>& dims, int32_t dtype, void* data);
  // An FP4 (E2M1) view: `dims` count ELEMENTS, two per stored byte, while
  // `data` is the byte base. The audit note in artifacts/cann92_audit/REPORT.md
  // ("FP4 is element-packed (2/byte); aclCreateTensor views must count elements
  // with byte storage offset") is what this encodes.
  aclTensor* CreateFp4Tensor(const char* label, const std::vector<int64_t>& dims, void* data);
  aclTensorList* CreateTensorList(const char* label, const std::vector<aclTensor*>& tensors);
  aclIntArray* CreateIntArray(const char* label, const std::vector<int64_t>& values);
  // A host aclScalar whose backing bytes the arena owns for its lifetime
  // (`aclCreateScalar` keeps reading `value`, so the copy below is what the
  // handle points at). `value` is copied; `dtype` decides how many bytes are
  // read. Used for the aclnnSoftplus beta / threshold scalars, which must be
  // planned once and never rebuilt inside the decode loop.
  aclScalar* CreateScalar(const char* label, int32_t dtype, const void* value);

  // ACLNN workspace high-water mark, collected while planning.
  void NoteWorkspace(uint64_t bytes);
  void CommitWorkspace();
  void* workspace() const { return workspace_; }
  uint64_t workspace_bytes() const { return workspace_bytes_; }
  // Refuses a re-plan that returned more than was reserved. This is the "never
  // grow the workspace inside a decode step" rule from
  // `tools/dsv4_moe_runtime/hardware/v5_ops.py`.
  void AssertWorkspaceFits(uint64_t bytes, const char* what) const;

  // ---- phase 3: SEALED --------------------------------------------------

  void Seal();
  bool sealed() const { return sealed_; }

  // ---- reporting / verification ----------------------------------------

  const std::vector<ArenaReservation>& reservations() const { return reservations_; }
  const std::vector<DescriptorRecord>& descriptors() const { return descriptors_; }
  size_t arena_bytes() const { return arena_bytes_; }
  const void* arena_base() const { return arena_; }

  // Throws unless every reservation sits at its declared alignment, no two
  // overlap, and all of them fit the committed arena.
  void ValidateAlignment() const;

  std::string DescribeLedger() const;

 private:
  void RefuseIfSealed(const char* what) const;
  void RequireCommitted(const char* what) const;
  aclTensor* CreateTensorInternal(const char* label, const std::vector<int64_t>& view_dims, int32_t dtype,
                                  void* data, const std::vector<int64_t>& storage_dims);

  DeviceOps& device_;
  std::vector<ArenaReservation> reservations_;
  size_t cursor_ = 0;
  void* arena_ = nullptr;
  size_t arena_bytes_ = 0;

  std::vector<DescriptorRecord> descriptors_;
  std::vector<aclTensor*> owned_tensors_;
  std::vector<aclTensorList*> owned_lists_;
  std::vector<aclIntArray*> owned_int_arrays_;
  std::vector<aclScalar*> owned_scalars_;
  // Backing bytes of every owned scalar, kept alive for the arena's lifetime
  // (a deque, like descriptor_shapes_: push_back must not move earlier elements).
  std::deque<std::vector<uint8_t>> scalar_values_;
  // Dim / stride / value vectors handed to aclCreateTensor and
  // aclCreateIntArray are kept alive for the arena's lifetime, matching the
  // caution `AclnnIntArray` in csrc/tests/common/aclnn_runtime.hpp already
  // takes. A deque, not a vector: push_back must not move the earlier elements,
  // because the handles were built from pointers into them.
  std::deque<std::vector<int64_t>> descriptor_shapes_;

  void* workspace_ = nullptr;
  uint64_t workspace_bytes_ = 0;
  uint64_t workspace_high_water_ = 0;

  bool sealed_ = false;
};

// Contiguous row-major strides for `dims`.
std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& dims);

int64_t ElementCount(const std::vector<int64_t>& dims);

}  // namespace dsv4
}  // namespace vllm_ascend
