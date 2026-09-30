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
// The gate activation is clamped symmetrically to +/- swigluLimit before the
// activation. This is an architectural constant of DeepSeek-V4, not a tunable
// and not an overflow workaround: a kernel that omits it computes a different
// function from the model. It is carried in tiling rather than hard-coded so a
// model with a different limit can reuse the kernel, and so the value is part
// of the host-side contract instead of being buried in device code.
TILING_DATA_FIELD_DEF(float, swigluLimit);
TILING_DATA_FIELD_DEF(float, tilingReserved); // keeps the struct 8-byte aligned
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Dsv4MoeExpert, Dsv4MoeExpertTilingData)

struct Dsv4MoeExpertCompileInfo {
    int64_t coreNum{0};
    int64_t ubSize{0};
};

} // namespace optiling

#endif // DSV4_MOE_EXPERT_TILING_H
