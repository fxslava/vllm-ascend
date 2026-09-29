/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software and can be redistributed under the terms
 * and conditions of the CANN Open Software License Agreement Version 2.0.
 * See the License for details.
 */

/*!
 * \file dsv4_moe_expert_tiling.h
 * \brief Tiling data definition for the DSV4 MoE expert projection kernel.
 */

#ifndef DSV4_MOE_EXPERT_TILING_H
#define DSV4_MOE_EXPERT_TILING_H

#include <cstdint>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Dsv4MoeExpertTilingData)
TILING_DATA_FIELD_DEF(int64_t, hiddenSize); // reduction dim of the gate/up leg
TILING_DATA_FIELD_DEF(int64_t, interSize);  // reduction dim of the down leg
TILING_DATA_FIELD_DEF(int64_t, blockSize);  // E8M0 scale span, always 32
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Dsv4MoeExpert, Dsv4MoeExpertTilingData)

struct Dsv4MoeExpertCompileInfo {
    int64_t coreNum{0};
    int64_t ubSize{0};
};

} // namespace optiling

#endif // DSV4_MOE_EXPERT_TILING_H
