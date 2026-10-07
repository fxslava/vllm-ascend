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

// Mock aclnn meta header: the descriptor engine (aclTensor / aclScalar /
// aclIntArray / aclTensorList / aclOpExecutor) and the address-repointing
// entry points the static-arena design rests on.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t aclnnStatus;

typedef enum aclDataType {
  ACL_FLOAT32 = 0,
  ACL_FLOAT16 = 1,
  ACL_INT8 = 2,
  ACL_INT32 = 3,
  ACL_UINT8 = 4,
  ACL_INT16 = 6,
  ACL_UINT16 = 7,
  ACL_INT64 = 9,
  ACL_UINT64 = 10,
  ACL_FLOAT64 = 11,
  ACL_BOOL = 12,
  ACL_DT_UNDEFINED = 17,
  ACL_BF16 = 27,
  ACL_FLOAT8_E5M2 = 35,
  ACL_FLOAT8_E4M3FN = 36,
  ACL_FLOAT8_E8M0 = 37,
  ACL_FP4X2_E2M1 = 40,
} aclDataType;

typedef enum aclFormat { ACL_FORMAT_ND = 2 } aclFormat;

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclTensorList aclTensorList;

aclTensor* aclCreateTensor(const int64_t* view_dims, uint64_t view_dims_num, aclDataType dtype,
                           const int64_t* view_strides, int64_t storage_offset, aclFormat format,
                           const int64_t* storage_dims, uint64_t storage_dims_num, void* device_data);

aclnnStatus aclDestroyTensor(const aclTensor* tensor);

aclTensorList* aclCreateTensorList(const aclTensor* const* tensors, uint64_t size);
aclnnStatus aclDestroyTensorList(const aclTensorList* tensor_list);

aclIntArray* aclCreateIntArray(const int64_t* value, uint64_t size);
aclnnStatus aclDestroyIntArray(const aclIntArray* array);

aclScalar* aclCreateScalar(void* value, aclDataType dtype);
aclnnStatus aclDestroyScalar(const aclScalar* scalar);

// Repoints executor slot `index` at `tensor` living at `address`. The mock
// validates [address, address + tensor bytes) against the registered spans
// before accepting it.
aclnnStatus aclSetTensorAddr(aclOpExecutor* executor, uint64_t index, const aclTensor* tensor, void* address);

aclnnStatus aclSetDynamicTensorAddr(aclOpExecutor* executor, uint64_t ir_index, uint64_t relative_index,
                                    const aclTensorList* tensor_list, void* address);

aclnnStatus aclSetAclOpExecutorRepeatable(aclOpExecutor* executor);
aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor* executor);

#ifdef __cplusplus
}
#endif
