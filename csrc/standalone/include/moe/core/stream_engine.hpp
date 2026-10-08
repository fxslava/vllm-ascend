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

// IStreamEngine: the ordering half of the backend contract (ISP).
//
// Streams, events, synchronization and DMA transfers. A consumer that only
// orchestrates ordered work -- the exclusive hierarchy's swap engine, the
// router's forced readback -- depends on this interface alone and never sees
// an allocation call. Splitting the old monolithic DeviceOps along this line
// is what lets the staging engine be unit-tested against a fake engine while
// the arena is tested against a fake allocator.

#pragma once

#include <cstddef>
#include <cstdint>

#include "moe/core/device_types.hpp"

namespace ascend_moe {

class IStreamEngine {
 public:
  virtual ~IStreamEngine() = default;

  virtual DeviceStream CreateStream() = 0;
  virtual void DestroyStream(DeviceStream stream) = 0;
  virtual DeviceEvent CreateEvent() = 0;
  virtual void DestroyEvent(DeviceEvent event) = 0;

  virtual void RecordEvent(DeviceEvent event, DeviceStream stream) = 0;
  virtual void StreamWaitEvent(DeviceStream stream, DeviceEvent event) = 0;
  virtual void SynchronizeStream(DeviceStream stream) = 0;

  // Ordered transfers. `destination_capacity` is the caller's proof of room;
  // backends that can validate bounds (the symbolic mock) use it.
  virtual void MemcpyAsync(void* destination, size_t destination_capacity, const void* source, size_t count,
                           MemcpyKind kind, DeviceStream stream) = 0;
  virtual void MemcpySync(void* destination, size_t destination_capacity, const void* source, size_t count,
                          MemcpyKind kind) = 0;
};

}  // namespace ascend_moe
