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

// Mock CANN runtime header: aclrt* entry points backed by the symbolic
// allocator in libopapi_mock (no device, no physical memory for arenas).

#pragma once

#include "acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* aclrtStream;
typedef void* aclrtEvent;
typedef void* aclrtContext;

aclError aclInit(const char* config_path);
aclError aclFinalize(void);
aclError aclrtSetDevice(int32_t device_id);
aclError aclrtResetDevice(int32_t device_id);
aclError aclrtCreateContext(aclrtContext* context, int32_t device_id);
aclError aclrtDestroyContext(aclrtContext context);
aclError aclrtSetCurrentContext(aclrtContext context);
const char* aclrtGetSocName(void);

// The symbolic allocator: fake, monotonically increasing, 4096-aligned
// addresses; every span is registered in the global interval map.
aclError aclrtMalloc(void** device_ptr, size_t size, aclrtMemMallocPolicy policy);
aclError aclrtFree(void* device_ptr);
aclError aclrtMallocHost(void** host_ptr, size_t size);
aclError aclrtFreeHost(void* host_ptr);

// Bounds-validated only: no byte is ever moved. Destination and source
// intervals must sit strictly inside registered spans; a partial overlap of
// the two intervals is refused (illegal non-inplace aliasing).
aclError aclrtMemcpy(void* destination, size_t destination_capacity, const void* source, size_t count,
                     aclrtMemcpyKind kind);
aclError aclrtMemcpyAsync(void* destination, size_t destination_capacity, const void* source, size_t count,
                          aclrtMemcpyKind kind, aclrtStream stream);
aclError aclrtMemset(void* destination, size_t destination_capacity, int32_t value, size_t count);

aclError aclrtCreateStream(aclrtStream* stream);
aclError aclrtDestroyStream(aclrtStream stream);
aclError aclrtCreateEventExWithFlag(aclrtEvent* event, uint32_t flag);
aclError aclrtDestroyEvent(aclrtEvent event);
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream);
aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event);
aclError aclrtSynchronizeStream(aclrtStream stream);
aclError aclrtSynchronizeEvent(aclrtEvent event);

aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t* free_bytes, size_t* total_bytes);

#ifdef __cplusplus
}
#endif
