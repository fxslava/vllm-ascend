/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software and can be redistributed under the terms
 * and conditions of the CANN Open Software License Agreement Version 2.0.
 * See the License for details.
 */

/*!
 * \file dsv4_moe_expert_def.cpp
 * \brief DeepSeek-V4 Flash MoE expert projection operator definition.
 *
 * One launch computes gate/up FP4 GEMMs + SwiGLU + down FP4 GEMM for a single
 * expert, mirroring the operand contract of the PyTorch draft runtime's
 * ExpertKernelRunner seam (tools/dsv4_moe_runtime). All outputs are
 * caller-allocated and written in place: the op allocates nothing.
 */

#include <register/op_def_registry.h>

namespace ops {

class Dsv4MoeExpert : public OpDef {
public:
    explicit Dsv4MoeExpert(const char* name) : OpDef(name)
    {
        // x: [1, hidden] bf16 activation row.
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        // w1/w3 (gate/up): [inter, hidden/2] packed FP4; w2 (down): [hidden, inter/2].
        for (const char* weight : {"w1", "w2", "w3"}) {
            this->Input(weight)
                .ParamType(REQUIRED)
                .DataType({ge::DT_UINT8})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
        }
        // E8M0 block-32 scales carried as raw bytes: w1/w3_scale [inter, hidden/32],
        // w2_scale [hidden, inter/32]. Matches the slot pool's uint8 views.
        for (const char* scale : {"w1_scale", "w2_scale", "w3_scale"}) {
            this->Input(scale)
                .ParamType(REQUIRED)
                .DataType({ge::DT_UINT8})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
        }
        // Outputs: gate_out/up_out/activated [1, inter] bf16, down_out [1, hidden] bf16.
        this->Output("gate_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("up_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("activated")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("down_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // Ascend 950 (arch35) configuration using Regbase, mirroring the
        // proven registration pattern of the other ascend950 MoE ops.
        OpAICoreConfig regbaseCfg;
        regbaseCfg.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "dsv4_moe_expert");
        this->AICore().AddConfig("ascend950", regbaseCfg);
    }
};

OP_ADD(Dsv4MoeExpert);

} // namespace ops
