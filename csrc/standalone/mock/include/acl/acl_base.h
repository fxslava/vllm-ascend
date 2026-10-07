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

// Mock CANN base header. Declares only what csrc/standalone includes; the
// runtime behind it is libopapi_mock (see mock_runtime/). Shapes mirror the
// CANN 9.2.0-beta.2 headers the production build compiles against, so the
// same dsv4 sources translate unchanged.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t aclError;

#define ACL_SUCCESS 0
#define ACL_ERROR_REPEAT_INITIALIZE 100031
#define ACL_ERROR_INVALID_PARAM 100000
#define ACL_ERROR_RT_FAILURE 107000

typedef enum aclrtMemcpyKind {
  ACL_MEMCPY_HOST_TO_HOST = 0,
  ACL_MEMCPY_HOST_TO_DEVICE = 1,
  ACL_MEMCPY_DEVICE_TO_HOST = 2,
  ACL_MEMCPY_DEVICE_TO_DEVICE = 3,
} aclrtMemcpyKind;

typedef enum aclrtMemMallocPolicy {
  ACL_MEM_MALLOC_HUGE_FIRST = 0,
  ACL_MEM_MALLOC_HUGE_ONLY = 1,
  ACL_MEM_MALLOC_NORMAL_FIRST = 2,
  ACL_MEM_MALLOC_NORMAL_ONLY = 3,
} aclrtMemMallocPolicy;

typedef enum aclrtMemAttr { ACL_HBM_MEM = 0, ACL_DDR_MEM = 1 } aclrtMemAttr;

typedef enum aclrtEventWaitStatus { ACL_EVENT_WAIT_STATUS_COMPLETE = 0 } aclrtEventWaitStatus;

#define ACL_EVENT_SYNC 0x0

#ifdef __cplusplus
}
#endif
