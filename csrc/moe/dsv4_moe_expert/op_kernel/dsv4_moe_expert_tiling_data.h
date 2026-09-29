/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software and can be redistributed under the terms
 * and conditions of the CANN Open Software License Agreement Version 2.0.
 * See the License for details.
 */

/*!
 * \file dsv4_moe_expert_tiling_data.h
 * \brief Device-side tiling struct for dsv4_moe_expert.
 *
 * Plain mirror of the host-side BEGIN_TILING_DATA_DEF(Dsv4MoeExpertTilingData)
 * in op_host/dsv4_moe_expert_tiling.h: identical field order, all int64, so
 * GET_TILING_DATA_WITH_STRUCT reads the serialized host bytes unchanged.
 */

#ifndef DSV4_MOE_EXPERT_TILING_DATA_H
#define DSV4_MOE_EXPERT_TILING_DATA_H

struct Dsv4MoeExpertTilingData {
    int64_t hiddenSize; // reduction dim of the gate/up leg
    int64_t interSize;  // reduction dim of the down leg
    int64_t blockSize;  // E8M0 scale span, always 32
};

#endif // DSV4_MOE_EXPERT_TILING_DATA_H
