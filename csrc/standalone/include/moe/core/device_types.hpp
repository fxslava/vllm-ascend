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

// The value types two backends share: stream / event handles, transfer kinds
// and the DMA counters the contract tests assert on.

#pragma once

#include <cstddef>
#include <cstdint>

namespace ascend_moe {

using DeviceStream = void*;
using DeviceEvent = void*;

enum class MemcpyKind { kHostToHost, kHostToDevice, kDeviceToHost, kDeviceToDevice };

const char* MemcpyKindName(MemcpyKind kind);

struct DmaCounters {
  uint64_t host_to_device_bytes = 0;
  uint64_t device_to_host_bytes = 0;
  uint64_t device_to_device_bytes = 0;
  uint64_t async_copies = 0;
  uint64_t sync_copies = 0;
  uint64_t events_recorded = 0;
  uint64_t stream_waits = 0;
  uint64_t stream_synchronizations = 0;
  uint64_t device_allocations = 0;
  uint64_t host_pinned_allocations = 0;

  void Reset() { *this = DmaCounters(); }
};

}  // namespace ascend_moe
