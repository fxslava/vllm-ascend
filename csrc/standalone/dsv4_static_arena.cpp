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

#include "dsv4_static_arena.hpp"

#include <acl/acl_base.h>
#include <aclnn/acl_meta.h>

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "dsv4_acl_check.hpp"
#include "dsv4_expert_layout.hpp"

namespace vllm_ascend {
namespace dsv4 {

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& dims) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (size_t index = dims.size(); index-- > 1;) {
    strides[index - 1] = strides[index] * dims[index];
  }
  return strides;
}

int64_t ElementCount(const std::vector<int64_t>& dims) {
  int64_t count = 1;
  for (int64_t dim : dims) {
    count *= dim;
  }
  return count;
}

DSV4StaticMemoryArena::DSV4StaticMemoryArena(DeviceOps& device) : device_(device) {}

DSV4StaticMemoryArena::~DSV4StaticMemoryArena() {
  for (aclTensorList* list : owned_lists_) {
    aclDestroyTensorList(list);
  }
  for (aclTensor* tensor : owned_tensors_) {
    aclDestroyTensor(tensor);
  }
  for (aclIntArray* array : owned_int_arrays_) {
    aclDestroyIntArray(array);
  }
  if (workspace_ != nullptr) {
    device_.DeviceFree(workspace_);
  }
  if (arena_ != nullptr) {
    device_.DeviceFree(arena_);
  }
}

// ---------------------------------------------------------------------------
// Phase guards
// ---------------------------------------------------------------------------

void DSV4StaticMemoryArena::RefuseIfSealed(const char* what) const {
  DSV4_REQUIRE(!sealed_, "the static arena is sealed for decoding; "
                             << what << " would allocate or build a descriptor inside the 43-layer loop");
}

void DSV4StaticMemoryArena::RequireCommitted(const char* what) const {
  DSV4_REQUIRE(arena_ != nullptr, what << " needs the arena to be committed first (one aclrtMalloc for all of it)");
}

// ---------------------------------------------------------------------------
// Phase 1: RESERVE
// ---------------------------------------------------------------------------

ArenaHandle DSV4StaticMemoryArena::Reserve(const char* name, size_t bytes, size_t align) {
  RefuseIfSealed("Reserve");
  DSV4_REQUIRE(arena_ == nullptr, "Reserve(" << name << ") after Commit: the arena's size is already fixed");
  DSV4_REQUIRE(bytes > 0, "reservation " << name << " is empty");
  DSV4_REQUIRE(align > 0 && (align & (align - 1)) == 0,
               "reservation " << name << " wants a non-power-of-two alignment " << align);
  ArenaReservation reservation;
  reservation.name = name;
  reservation.bytes = bytes;
  reservation.align = align;
  reservation.offset = AlignUp(cursor_, align);
  cursor_ = reservation.offset + bytes;
  reservations_.push_back(reservation);
  return reservations_.size() - 1;
}

// ---------------------------------------------------------------------------
// Phase 2: BUILD
// ---------------------------------------------------------------------------

void DSV4StaticMemoryArena::Commit() {
  RefuseIfSealed("Commit");
  DSV4_REQUIRE(arena_ == nullptr, "the arena is already committed");
  DSV4_REQUIRE(!reservations_.empty(), "nothing was reserved");
  arena_bytes_ = AlignUp(cursor_, kArenaAlignBytes);
  arena_ = device_.DeviceMalloc(arena_bytes_);
  // Zeroing matters for the KV cache and the routing buffers: an unwritten
  // block read as stale HBM is a wrong answer with no symptom.
  device_.DeviceMemset(arena_, arena_bytes_, 0, arena_bytes_);
  ValidateAlignment();
}

void* DSV4StaticMemoryArena::Address(ArenaHandle handle) const {
  RequireCommitted("Address");
  DSV4_REQUIRE(handle < reservations_.size(), "arena handle " << handle << " out of range");
  return static_cast<uint8_t*>(arena_) + reservations_[handle].offset;
}

size_t DSV4StaticMemoryArena::Bytes(ArenaHandle handle) const {
  DSV4_REQUIRE(handle < reservations_.size(), "arena handle " << handle << " out of range");
  return reservations_[handle].bytes;
}

aclTensor* DSV4StaticMemoryArena::CreateTensorInternal(const char* label, const std::vector<int64_t>& view_dims,
                                                       int32_t dtype, void* data,
                                                       const std::vector<int64_t>& storage_dims) {
  RefuseIfSealed("CreateTensor");
  DSV4_REQUIRE(!view_dims.empty(), label << ": a descriptor needs at least one dimension");
  descriptor_shapes_.push_back(view_dims);
  const std::vector<int64_t>& dims = descriptor_shapes_.back();
  descriptor_shapes_.push_back(ContiguousStrides(view_dims));
  const std::vector<int64_t>& strides = descriptor_shapes_.back();
  descriptor_shapes_.push_back(storage_dims);
  const std::vector<int64_t>& storage = descriptor_shapes_.back();

  aclTensor* tensor = aclCreateTensor(dims.data(), dims.size(), static_cast<aclDataType>(dtype), strides.data(), 0,
                                      static_cast<aclFormat>(kAclFormatNd), storage.data(), storage.size(), data);
  DSV4_REQUIRE(tensor != nullptr, label << ": aclCreateTensor returned null");
  owned_tensors_.push_back(tensor);

  DescriptorRecord record;
  record.label = label;
  record.kind = DescriptorRecord::Kind::kTensor;
  record.dtype = dtype;
  record.element_count = ElementCount(view_dims);
  record.address = data;
  descriptors_.push_back(record);
  return tensor;
}

aclTensor* DSV4StaticMemoryArena::CreateTensor(const char* label, const std::vector<int64_t>& dims, int32_t dtype,
                                               void* data) {
  return CreateTensorInternal(label, dims, dtype, data, dims);
}

aclTensor* DSV4StaticMemoryArena::CreateFp4Tensor(const char* label, const std::vector<int64_t>& dims, void* data) {
  DSV4_REQUIRE(dims.back() % kFp4ElementsPerByte == 0,
               label << ": an FP4 view's last dimension (" << dims.back()
                     << " elements) must be even, because two E2M1 values share a byte");
  return CreateTensorInternal(label, dims, kAclFloat4E2m1, data, dims);
}

aclTensorList* DSV4StaticMemoryArena::CreateTensorList(const char* label, const std::vector<aclTensor*>& tensors) {
  RefuseIfSealed("CreateTensorList");
  DSV4_REQUIRE(!tensors.empty(), label << ": an empty tensor list has no meaning to any aclnn op");
  std::vector<const aclTensor*> handles(tensors.begin(), tensors.end());
  aclTensorList* list = aclCreateTensorList(handles.data(), handles.size());
  DSV4_REQUIRE(list != nullptr, label << ": aclCreateTensorList returned null");
  owned_lists_.push_back(list);

  DescriptorRecord record;
  record.label = label;
  record.kind = DescriptorRecord::Kind::kTensorList;
  record.element_count = static_cast<int64_t>(tensors.size());
  descriptors_.push_back(record);
  return list;
}

aclIntArray* DSV4StaticMemoryArena::CreateIntArray(const char* label, const std::vector<int64_t>& values) {
  RefuseIfSealed("CreateIntArray");
  DSV4_REQUIRE(!values.empty(), label << ": an empty int array has no meaning to any aclnn op");
  descriptor_shapes_.push_back(values);
  const std::vector<int64_t>& kept = descriptor_shapes_.back();
  aclIntArray* array = aclCreateIntArray(kept.data(), kept.size());
  DSV4_REQUIRE(array != nullptr, label << ": aclCreateIntArray returned null");
  owned_int_arrays_.push_back(array);

  DescriptorRecord record;
  record.label = label;
  record.kind = DescriptorRecord::Kind::kIntArray;
  record.element_count = static_cast<int64_t>(values.size());
  descriptors_.push_back(record);
  return array;
}

void DSV4StaticMemoryArena::NoteWorkspace(uint64_t bytes) {
  RefuseIfSealed("NoteWorkspace");
  workspace_high_water_ = std::max(workspace_high_water_, bytes);
}

void DSV4StaticMemoryArena::CommitWorkspace() {
  RefuseIfSealed("CommitWorkspace");
  DSV4_REQUIRE(workspace_ == nullptr, "the ACLNN workspace is already committed");
  // One shared workspace sized by the high-water mark across every planned op.
  // Operators run sequentially on the compute stream, so they cannot be using
  // it at the same time; a per-op workspace would multiply the reservation by
  // the op count for no benefit.
  workspace_bytes_ = workspace_high_water_;
  if (workspace_bytes_ == 0) {
    return;  // every planned op asked for zero; nothing to allocate
  }
  workspace_ = device_.DeviceMalloc(static_cast<size_t>(workspace_bytes_));
  device_.DeviceMemset(workspace_, static_cast<size_t>(workspace_bytes_), 0, static_cast<size_t>(workspace_bytes_));
}

void DSV4StaticMemoryArena::AssertWorkspaceFits(uint64_t bytes, const char* what) const {
  DSV4_REQUIRE(bytes <= workspace_bytes_,
               what << " plans " << bytes << " workspace bytes but only " << workspace_bytes_
                    << " were reserved at init; the decode shapes changed -- rebuild the pipeline, never grow the "
                       "workspace inside a step");
}

// ---------------------------------------------------------------------------
// Phase 3: SEALED
// ---------------------------------------------------------------------------

void DSV4StaticMemoryArena::Seal() {
  RequireCommitted("Seal");
  sealed_ = true;
}

// ---------------------------------------------------------------------------
// Verification and reporting
// ---------------------------------------------------------------------------

void DSV4StaticMemoryArena::ValidateAlignment() const {
  size_t previous_end = 0;
  const char* previous_name = "<start>";
  for (const ArenaReservation& reservation : reservations_) {
    DSV4_REQUIRE(reservation.offset % reservation.align == 0,
                 "reservation " << reservation.name << " sits at " << reservation.offset << ", not aligned to "
                                << reservation.align);
    DSV4_REQUIRE(reservation.offset >= previous_end,
                 "reservation " << reservation.name << " at " << reservation.offset << " overlaps " << previous_name
                                << " which ends at " << previous_end);
    DSV4_REQUIRE(reservation.offset + reservation.bytes <= arena_bytes_,
                 "reservation " << reservation.name << " ends at " << (reservation.offset + reservation.bytes)
                                << " beyond the " << arena_bytes_ << "-byte arena");
    previous_end = reservation.offset + reservation.bytes;
    previous_name = reservation.name;
  }
  // The arena base comes from aclrtMalloc, which the toolkit documents as
  // aligned; every reservation's absolute address is base + offset, so an
  // aligned base plus an aligned offset is what makes the descriptor addresses
  // aligned too.
  DSV4_REQUIRE(reinterpret_cast<uintptr_t>(arena_) % kArenaAlignBytes == 0,
               "the committed arena base is not " << kArenaAlignBytes << "-byte aligned");
}

std::string DSV4StaticMemoryArena::DescribeLedger() const {
  const double mib = 1024.0 * 1024.0;
  std::ostringstream out;
  out << std::fixed << std::setprecision(3);
  out << "static device arena: " << arena_bytes_ << " bytes (" << arena_bytes_ / mib << " MiB) in "
      << reservations_.size() << " reservations, " << descriptors_.size() << " descriptors, workspace "
      << workspace_bytes_ << " bytes (" << workspace_bytes_ / mib << " MiB)\n";
  out << "  state: " << (arena_ != nullptr ? "committed" : "uncommitted") << (sealed_ ? ", SEALED" : "") << "\n";
  for (const ArenaReservation& reservation : reservations_) {
    out << "  " << std::left << std::setw(30) << reservation.name << std::right << " offset=" << std::setw(12)
        << reservation.offset << " bytes=" << std::setw(12) << reservation.bytes << " align=" << std::setw(4)
        << reservation.align << "\n";
  }
  return out.str();
}

}  // namespace dsv4
}  // namespace vllm_ascend
