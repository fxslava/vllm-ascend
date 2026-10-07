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

// The symbolic allocator: memory as intervals, not bytes.
//
// DESIGN (per the mock-runtime brief):
//   * aclrtMalloc / aclrtMallocHost hand out monotonically increasing fake
//     addresses, 4096-byte aligned, and register [base, base + size) in a
//     global interval registry. No physical memory backs them -- the whole
//     137 GiB routed-expert hierarchy exists as bookkeeping.
//   * The ONE exception is bounded: host allocations at or below
//     kMockRealHostMaxBytes get real memory, because product code
//     dereferences a few small pinned mailboxes directly (std::memset,
//     `*slot_mailbox_ = position`). Everything the process actually touches
//     stays within a few hundred KiB; "essentially zero" holds.
//   * aclrtMemcpyAsync / aclrtMemcpy / aclrtMemset validate intervals and move
//     nothing: [addr, addr + count) must sit strictly inside ONE registered
//     span, and a partially overlapping source/destination pair is refused as
//     illegal non-inplace aliasing (identical intervals are in-place, which
//     is legal).
//
// What the registry buys: every DMA, every aclSetTensorAddr and every tensor
// region the contract validators look at is proven in-bounds of a span the
// allocator handed out -- a full memory-layout audit of the arena, the paged
// cache and the exclusive slot pool, at zero RAM cost.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace vllm_ascend {
namespace dsv4 {
namespace mock {

// Host allocations this size or smaller are real (mailboxes, transit
// scratch); anything larger is symbolic.
inline constexpr size_t kMockRealHostMaxBytes = 1ull << 20;  // 1 MiB

// Alignment every fake base address carries, mirroring
// aclrtMalloc(..., ACL_MEM_MALLOC_HUGE_FIRST) page alignment.
inline constexpr size_t kMockAllocAlignBytes = 4096;

struct MockMemoryStats {
  uint64_t device_allocations = 0;
  uint64_t host_allocations = 0;
  uint64_t frees = 0;
  uint64_t active_spans = 0;
  uint64_t symbolic_device_bytes = 0;
  uint64_t real_host_bytes = 0;
  uint64_t memcpy_checks = 0;
  uint64_t memset_checks = 0;
  uint64_t setaddr_checks = 0;
  uint64_t slot_map_mismatches = 0;  // aclSetTensorAddr index != captured handle
  uint64_t rejected_operations = 0;
  uint64_t would_copy_bytes = 0;
};

// -- interval registry -------------------------------------------------------

// Registers [base, base + size). Returns false (and registers nothing) if the
// interval overlaps a live span.
bool MockRegisterSpan(uintptr_t base, size_t bytes, bool is_real);

// Removes the span starting exactly at `base`.
bool MockUnregisterSpan(uintptr_t base);

// True iff [addr, addr + bytes) lies strictly inside one registered span.
bool MockSpanContains(uintptr_t addr, size_t bytes);

// True iff the span containing `addr` is one of the small REAL host spans
// (the mailboxes), not a symbolic one.
bool MockSpanIsRealAt(uintptr_t addr);

// True iff [a, a + a_bytes) and [b, b + b_bytes) overlap but are not the
// identical interval -- the illegal non-inplace aliasing.
bool MockPartiallyOverlaps(uintptr_t a, size_t a_bytes, uintptr_t b, size_t b_bytes);

// Validates one transfer: both intervals contained, no partial overlap.
// Returns false with `reason` filled when the contract is violated.
bool MockCheckTransfer(uintptr_t destination, uintptr_t source, size_t count, std::string* reason);

// -- allocator entry points (back the aclrt* C symbols) ---------------------

// Fake, monotonically increasing, 4096-aligned address for a new span.
uintptr_t MockDeviceMalloc(size_t bytes);
uintptr_t MockHostMalloc(size_t bytes);
void MockFree(uintptr_t base);

// -- hooks and reporting ----------------------------------------------------

// D2H seeding: the pipeline's one forced read per MoE layer copies the top-6
// expert ids (and the greedy token) device-to-host; with no-op operators
// there is nothing to copy, so the test installs a seed that fabricates a
// deterministic, in-range payload into the destination. The runtime itself
// stays pure: no seed installed, nothing written.
using MockD2HSeed = void (*)(void* destination, size_t count);
void MockSetD2HSeed(MockD2HSeed seed);

// Total HBM the mock reports through aclrtGetMemInfo (set before the slot
// planner runs; the free figure subtracts the live symbolic spans).
void MockSetReportedHbm(size_t total_bytes);

const MockMemoryStats& MockMemoryStatistics();
void MockNoteSetAddrCheck();
void MockNoteSlotMapMismatch();
void MockResetAllocatorForTest();

}  // namespace mock
}  // namespace dsv4
}  // namespace vllm_ascend
