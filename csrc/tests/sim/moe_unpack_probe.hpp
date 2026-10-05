// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once
#include <cstdint>
extern "C" uint32_t dsv4_unpack_probe_impl(void* stream, void* packed, void* scales, void* decoded, void* activated,
                                           uint32_t elements, uint32_t columns, uint32_t path, uint32_t math);
extern "C" uint32_t dsv4_switch_control_impl(void* stream, void* output, uint32_t iterations, uint32_t simt);
