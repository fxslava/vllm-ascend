// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once

#include <cstdint>

#include "dsv4_synthetic_data.hpp"

namespace vllm_ascend {
// Defined in common/dsv4_moe_expert_kernels.cpp, inside the ascendc_library.
void dsv4_moe_expert_impl(void* stream, uint32_t blockDim, void* x, void* w1, void* w2, void* w3, void* w1Scale,
                          void* w2Scale, void* w3Scale, void* gateOut, void* upOut, void* activatedOut, void* downOut,
                          void* workspace, void* tiling);

}  // namespace vllm_ascend
