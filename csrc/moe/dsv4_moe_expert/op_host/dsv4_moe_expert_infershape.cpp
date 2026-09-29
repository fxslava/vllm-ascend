/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software and can be redistributed under the terms
 * and conditions of the CANN Open Software License Agreement Version 2.0.
 * See the License for details.
 */

/*!
 * \file dsv4_moe_expert_infershape.cpp
 * \brief Shape inference for the DSV4 MoE expert projection operator.
 */

#include "graph/utils/type_utils.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t INDEX_INPUT_X = 0;
constexpr size_t INDEX_INPUT_W1 = 1;
constexpr size_t INDEX_INPUT_W2 = 2;
constexpr size_t INDEX_INPUT_W3 = 3;
constexpr size_t INDEX_INPUT_W1_SCALE = 4;
constexpr size_t INDEX_INPUT_W2_SCALE = 5;
constexpr size_t INDEX_INPUT_W3_SCALE = 6;

constexpr size_t INDEX_OUTPUT_GATE = 0;
constexpr size_t INDEX_OUTPUT_UP = 1;
constexpr size_t INDEX_OUTPUT_ACTIVATED = 2;
constexpr size_t INDEX_OUTPUT_DOWN = 3;

constexpr int64_t FP4_BLOCK = 32;
constexpr int64_t FP4_PER_BYTE = 2;
} // namespace

static ge::graphStatus InferShapeForDsv4MoeExpert(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForDsv4MoeExpert");
    const gert::Shape* xShape = context->GetInputShape(INDEX_INPUT_X);
    const gert::Shape* w1Shape = context->GetInputShape(INDEX_INPUT_W1);
    const gert::Shape* w2Shape = context->GetInputShape(INDEX_INPUT_W2);
    const gert::Shape* w3Shape = context->GetInputShape(INDEX_INPUT_W3);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, w1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, w2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, w3Shape);

    OP_CHECK_IF(xShape->GetDimNum() != 2 || xShape->GetDim(0) != 1,
                OP_LOGE(context->GetNodeName(), "x must be [1, hidden], got rank %lu dim0 %ld", xShape->GetDimNum(),
                        xShape->GetDim(0)),
                return ge::GRAPH_FAILED);
    const int64_t hiddenSize = xShape->GetDim(1);

    OP_CHECK_IF(w1Shape->GetDimNum() != 2 || w3Shape->GetDimNum() != 2 || w2Shape->GetDimNum() != 2,
                OP_LOGE(context->GetNodeName(), "weights must be rank 2"),
                return ge::GRAPH_FAILED);
    const int64_t interSize = w1Shape->GetDim(0);
    OP_CHECK_IF(w1Shape->GetDim(1) != hiddenSize / FP4_PER_BYTE || w3Shape->GetDim(0) != interSize ||
                w3Shape->GetDim(1) != hiddenSize / FP4_PER_BYTE,
                OP_LOGE(context->GetNodeName(), "w1/w3 must be [inter, hidden/2] packed FP4"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(w2Shape->GetDim(0) != hiddenSize || w2Shape->GetDim(1) != interSize / FP4_PER_BYTE,
                OP_LOGE(context->GetNodeName(), "w2 must be [hidden, inter/2] packed FP4"),
                return ge::GRAPH_FAILED);

    for (size_t scaleIndex = INDEX_INPUT_W1_SCALE; scaleIndex <= INDEX_INPUT_W3_SCALE; ++scaleIndex) {
        const gert::Shape* scaleShape = context->GetInputShape(scaleIndex);
        OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
        const bool isDown = scaleIndex == INDEX_INPUT_W2_SCALE;
        const int64_t rows = isDown ? hiddenSize : interSize;
        const int64_t cols = isDown ? interSize : hiddenSize;
        OP_CHECK_IF(scaleShape->GetDimNum() != 2 || scaleShape->GetDim(0) != rows ||
                    scaleShape->GetDim(1) != cols / FP4_BLOCK,
                    OP_LOGE(context->GetNodeName(), "scale input %lu must be [%ld, %ld]", scaleIndex, rows,
                            cols / FP4_BLOCK),
                    return ge::GRAPH_FAILED);
    }

    gert::Shape interRow;
    interRow.AppendDim(1);
    interRow.AppendDim(interSize);
    *context->GetOutputShape(INDEX_OUTPUT_GATE) = interRow;
    *context->GetOutputShape(INDEX_OUTPUT_UP) = interRow;
    *context->GetOutputShape(INDEX_OUTPUT_ACTIVATED) = interRow;

    gert::Shape hiddenRow;
    hiddenRow.AppendDim(1);
    hiddenRow.AppendDim(hiddenSize);
    *context->GetOutputShape(INDEX_OUTPUT_DOWN) = hiddenRow;

    OP_LOGI(context->GetNodeName(), "End to do InferShapeForDsv4MoeExpert");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForDsv4MoeExpert(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(INDEX_OUTPUT_GATE, ge::DT_BF16);
    context->SetOutputDataType(INDEX_OUTPUT_UP, ge::DT_BF16);
    context->SetOutputDataType(INDEX_OUTPUT_ACTIVATED, ge::DT_BF16);
    context->SetOutputDataType(INDEX_OUTPUT_DOWN, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Dsv4MoeExpert)
    .InferShape(InferShapeForDsv4MoeExpert)
    .InferDataType(InferDataTypeForDsv4MoeExpert);
} // namespace ops
