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

#include "mock_allocator.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

#include "acl/acl.h"

namespace ascend_moe {
namespace mock {
namespace {

struct Span {
  uintptr_t base;
  size_t bytes;
  bool is_real;  // backed by real memory (small host mailboxes only)
};

std::mutex g_mutex;
std::map<uintptr_t, Span> g_spans;  // keyed by base
uintptr_t g_next_symbolic = 0x0000'1000'0000'0000ull;  // far from any real mapping
MockMemoryStats g_stats;
MockD2HSeed g_d2h_seed = nullptr;
size_t g_reported_hbm = 64ull << 30;
bool g_acl_initialized = false;
uint64_t g_next_handle = 1;  // streams / events / contexts

size_t RoundUp4096(size_t bytes) { return (bytes + kMockAllocAlignBytes - 1) & ~(kMockAllocAlignBytes - 1); }

}  // namespace

bool MockRegisterSpan(uintptr_t base, size_t bytes, bool is_real) {
  const std::lock_guard<std::mutex> lock(g_mutex);
  if (bytes == 0) {
    return false;
  }
  const uintptr_t end = base + bytes;  // no overflow: base is page aligned, bytes bounded
  // Overlap test against the neighbours in base order.
  const auto upper = g_spans.upper_bound(base);
  if (upper != g_spans.end() && end > upper->second.base) {
    return false;
  }
  if (upper != g_spans.begin()) {
    const auto lower = std::prev(upper);
    if (lower->first <= base && lower->second.base + lower->second.bytes > base) {
      return false;
    }
  }
  g_spans[base] = Span{base, bytes, is_real};
  ++g_stats.active_spans;
  return true;
}

bool MockUnregisterSpan(uintptr_t base) {
  const std::lock_guard<std::mutex> lock(g_mutex);
  const auto found = g_spans.find(base);
  if (found == g_spans.end()) {
    return false;
  }
  if (found->second.is_real) {
    std::free(reinterpret_cast<void*>(base));
  }
  g_spans.erase(found);
  --g_stats.active_spans;
  ++g_stats.frees;
  return true;
}

bool MockSpanContains(uintptr_t addr, size_t bytes) {
  const std::lock_guard<std::mutex> lock(g_mutex);
  if (g_spans.empty()) {
    return false;
  }
  const auto it = g_spans.upper_bound(addr);
  if (it == g_spans.begin()) {
    return false;  // addr below every span
  }
  const Span& span = std::prev(it)->second;
  if (addr < span.base || addr >= span.base + span.bytes) {
    return false;  // past the span end must not underflow the subtraction below
  }
  return bytes <= span.base + span.bytes - addr;
}

bool MockSpanIsRealAt(uintptr_t addr) {
  const std::lock_guard<std::mutex> lock(g_mutex);
  if (g_spans.empty()) {
    return false;
  }
  const auto it = g_spans.upper_bound(addr);
  if (it == g_spans.begin()) {
    return false;
  }
  const Span& span = std::prev(it)->second;
  return addr >= span.base && addr < span.base + span.bytes && span.is_real;
}

bool MockPartiallyOverlaps(uintptr_t a, size_t a_bytes, uintptr_t b, size_t b_bytes) {
  const bool identical = a == b && a_bytes == b_bytes;
  if (identical) {
    return false;  // in-place
  }
  const bool disjoint = a + a_bytes <= b || b + b_bytes <= a;
  return !disjoint;
}

bool MockCheckTransfer(uintptr_t destination, uintptr_t source, size_t count, std::string* reason) {
  ++g_stats.memcpy_checks;
  g_stats.would_copy_bytes += count;
  if (count == 0) {
    return true;
  }
  if (!MockSpanContains(destination, count)) {
    if (reason != nullptr) {
      std::ostringstream text;
      text << "destination [" << reinterpret_cast<void*>(destination) << ", "
           << reinterpret_cast<void*>(destination + count) << ") is not strictly inside any registered span";
      *reason = text.str();
    }
    ++g_stats.rejected_operations;
    return false;
  }
  if (!MockSpanContains(source, count)) {
    if (reason != nullptr) {
      std::ostringstream text;
      text << "source [" << reinterpret_cast<void*>(source) << ", " << reinterpret_cast<void*>(source + count)
           << ") is not strictly inside any registered span";
      *reason = text.str();
    }
    ++g_stats.rejected_operations;
    return false;
  }
  if (MockPartiallyOverlaps(destination, count, source, count)) {
    if (reason != nullptr) {
      *reason = "source and destination partially overlap (illegal non-inplace aliasing)";
    }
    ++g_stats.rejected_operations;
    return false;
  }
  return true;
}

uintptr_t MockDeviceMalloc(size_t bytes) {
  const size_t width = bytes == 0 ? 1 : bytes;
  const std::lock_guard<std::mutex> lock(g_mutex);
  const uintptr_t base = g_next_symbolic;
  g_next_symbolic = base + RoundUp4096(width) + kMockAllocAlignBytes;  // guard page between spans
  g_spans[base] = Span{base, width, false};
  ++g_stats.active_spans;
  ++g_stats.device_allocations;
  g_stats.symbolic_device_bytes += RoundUp4096(width);
  return base;
}

uintptr_t MockHostMalloc(size_t bytes) {
  const size_t width = bytes == 0 ? 1 : bytes;
  if (width <= kMockRealHostMaxBytes) {
    // The bounded real tier: mailboxes and the pinned transit scratch, which
    // product code dereferences on the host.
    const size_t rounded = RoundUp4096(width);
    void* pointer = std::aligned_alloc(kMockAllocAlignBytes, rounded);
    if (pointer == nullptr) {
      return 0;
    }
    if (!MockRegisterSpan(reinterpret_cast<uintptr_t>(pointer), width, true)) {
      std::free(pointer);
      return 0;
    }
    const std::lock_guard<std::mutex> lock(g_mutex);
    ++g_stats.host_allocations;
    g_stats.real_host_bytes += rounded;
    return reinterpret_cast<uintptr_t>(pointer);
  }
  const std::lock_guard<std::mutex> lock(g_mutex);
  const uintptr_t base = g_next_symbolic;
  g_next_symbolic = base + RoundUp4096(width) + kMockAllocAlignBytes;
  g_spans[base] = Span{base, width, false};
  ++g_stats.active_spans;
  ++g_stats.host_allocations;
  return base;
}

void MockFree(uintptr_t base) { MockUnregisterSpan(base); }

void MockSetD2HSeed(MockD2HSeed seed) {
  const std::lock_guard<std::mutex> lock(g_mutex);
  g_d2h_seed = seed;
}

void MockSetReportedHbm(size_t total_bytes) {
  const std::lock_guard<std::mutex> lock(g_mutex);
  g_reported_hbm = total_bytes;
}

const MockMemoryStats& MockMemoryStatistics() { return g_stats; }

void MockNoteSetAddrCheck() { ++g_stats.setaddr_checks; }

void MockNoteSlotMapMismatch() { ++g_stats.slot_map_mismatches; }

void MockResetAllocatorForTest() {
  const std::lock_guard<std::mutex> lock(g_mutex);
  std::vector<uintptr_t> real;
  for (const auto& entry : g_spans) {
    if (entry.second.is_real) {
      real.push_back(entry.first);
    }
  }
  for (uintptr_t base : real) {
    std::free(reinterpret_cast<void*>(base));
  }
  g_spans.clear();
  g_stats = MockMemoryStats();
  g_d2h_seed = nullptr;
}

// ---------------------------------------------------------------------------
// aclrt* C entry points
// ---------------------------------------------------------------------------

extern "C" {

aclError aclInit(const char*) {
  const std::lock_guard<std::mutex> lock(g_mutex);
  if (g_acl_initialized) {
    return ACL_ERROR_REPEAT_INITIALIZE;
  }
  g_acl_initialized = true;
  return ACL_SUCCESS;
}

aclError aclFinalize(void) { return ACL_SUCCESS; }

aclError aclrtSetDevice(int32_t) { return ACL_SUCCESS; }
aclError aclrtResetDevice(int32_t) { return ACL_SUCCESS; }
aclError aclrtCreateContext(aclrtContext* context, int32_t) {
  if (context == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  const std::lock_guard<std::mutex> lock(g_mutex);
  *context = reinterpret_cast<aclrtContext>(++g_next_handle);
  return ACL_SUCCESS;
}
aclError aclrtDestroyContext(aclrtContext) { return ACL_SUCCESS; }
aclError aclrtSetCurrentContext(aclrtContext) { return ACL_SUCCESS; }

const char* aclrtGetSocName(void) { return "Ascend950PR-Mock"; }

aclError aclrtMalloc(void** device_ptr, size_t size, aclrtMemMallocPolicy) {
  if (device_ptr == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  const uintptr_t base = MockDeviceMalloc(size);
  if (base % kMockAllocAlignBytes != 0) {
    return ACL_ERROR_RT_FAILURE;  // cannot happen; the invariant is the test's
  }
  *device_ptr = reinterpret_cast<void*>(base);
  return ACL_SUCCESS;
}

aclError aclrtFree(void* device_ptr) {
  if (device_ptr == nullptr) {
    return ACL_SUCCESS;
  }
  return MockUnregisterSpan(reinterpret_cast<uintptr_t>(device_ptr)) ? ACL_SUCCESS : ACL_ERROR_INVALID_PARAM;
}

aclError aclrtMallocHost(void** host_ptr, size_t size) {
  if (host_ptr == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  const uintptr_t base = MockHostMalloc(size);
  *host_ptr = reinterpret_cast<void*>(base);
  return base != 0 ? ACL_SUCCESS : ACL_ERROR_RT_FAILURE;
}

aclError aclrtFreeHost(void* host_ptr) { return aclrtFree(host_ptr); }

aclError aclrtMemcpy(void* destination, size_t destination_capacity, const void* source, size_t count,
                     aclrtMemcpyKind kind) {
  std::string reason;
  if (!MockCheckTransfer(reinterpret_cast<uintptr_t>(destination), reinterpret_cast<uintptr_t>(source), count,
                         &reason)) {
    return ACL_ERROR_INVALID_PARAM;
  }
  (void)destination_capacity;
  (void)kind;
  return ACL_SUCCESS;
}

aclError aclrtMemcpyAsync(void* destination, size_t destination_capacity, const void* source, size_t count,
                          aclrtMemcpyKind kind, aclrtStream) {
  std::string reason;
  if (!MockCheckTransfer(reinterpret_cast<uintptr_t>(destination), reinterpret_cast<uintptr_t>(source), count,
                         &reason)) {
    return ACL_ERROR_INVALID_PARAM;
  }
  (void)destination_capacity;
  if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
    // The seed (the test's symbolic oracle for the two forced readbacks) may
    // only ever write REAL memory: the mailboxes. A D2H landing in a symbolic
    // host span stays a pure interval check.
    MockD2HSeed seed;
    {
      const std::lock_guard<std::mutex> lock(g_mutex);
      seed = g_d2h_seed;
    }
    if (seed != nullptr && MockSpanIsRealAt(reinterpret_cast<uintptr_t>(destination))) {
      seed(destination, count);
    }
  }
  return ACL_SUCCESS;
}

aclError aclrtMemset(void* destination, size_t destination_capacity, int32_t value, size_t count) {
  ++g_stats.memset_checks;
  if (count > destination_capacity || !MockSpanContains(reinterpret_cast<uintptr_t>(destination), count)) {
    ++g_stats.rejected_operations;
    return ACL_ERROR_INVALID_PARAM;
  }
  (void)value;
  return ACL_SUCCESS;
}

aclError aclrtCreateStream(aclrtStream* stream) {
  if (stream == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  const std::lock_guard<std::mutex> lock(g_mutex);
  *stream = reinterpret_cast<aclrtStream>(++g_next_handle);
  return ACL_SUCCESS;
}
aclError aclrtDestroyStream(aclrtStream) { return ACL_SUCCESS; }

aclError aclrtCreateEventExWithFlag(aclrtEvent* event, uint32_t) {
  if (event == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  const std::lock_guard<std::mutex> lock(g_mutex);
  *event = reinterpret_cast<aclrtEvent>(++g_next_handle);
  return ACL_SUCCESS;
}
aclError aclrtDestroyEvent(aclrtEvent) { return ACL_SUCCESS; }
aclError aclrtRecordEvent(aclrtEvent, aclrtStream) { return ACL_SUCCESS; }
aclError aclrtStreamWaitEvent(aclrtStream, aclrtEvent) { return ACL_SUCCESS; }
aclError aclrtSynchronizeStream(aclrtStream) { return ACL_SUCCESS; }
aclError aclrtSynchronizeEvent(aclrtEvent) { return ACL_SUCCESS; }

aclError aclrtGetMemInfo(aclrtMemAttr, size_t* free_bytes, size_t* total_bytes) {
  const std::lock_guard<std::mutex> lock(g_mutex);
  size_t used = 0;
  for (const auto& entry : g_spans) {
    if (!entry.second.is_real) {
      used += entry.second.bytes;
    }
  }
  const size_t total = g_reported_hbm;
  if (total_bytes != nullptr) {
    *total_bytes = total;
  }
  if (free_bytes != nullptr) {
    *free_bytes = used < total ? total - used : 0;
  }
  return ACL_SUCCESS;
}

}  // extern "C"

}  // namespace mock
}  // namespace ascend_moe
