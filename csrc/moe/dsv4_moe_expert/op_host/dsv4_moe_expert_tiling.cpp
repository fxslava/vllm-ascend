/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software and can be redistributed under the terms
 * and conditions of the CANN Open Software License Agreement Version 2.0.
 * See the License for details.
 */

/*!
 * \file dsv4_moe_expert_tiling.cpp
 * \brief Tiling for the DSV4 MoE expert projection kernel (single-core milestone).
 *
 * The milestone kernel runs on one AIV core with the whole packed weight
 * matrix resident in UB, so tiling only validates the geometry and rejects
 * anything beyond the single-load budget. The reduced bring-up geometry
 * (hidden=256, inter=128 -> 16 KiB packed per projection) and the production
 * DeepSeek-V4 Flash geometry (hidden=4096, inter=2048 -> 4 MiB packed) differ
 * by 256x; chunked multi-core tiling is the production follow-up and will
 * lift the budget below.
 */

#include "dsv4_moe_expert_tiling.h"

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "graph/types.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INDEX_X = 0;
constexpr size_t INDEX_W1 = 1;
constexpr size_t INDEX_W2 = 2;
constexpr size_t INDEX_W3 = 3;
constexpr size_t INDEX_W1_SCALE = 4;
constexpr size_t INDEX_W2_SCALE = 5;
constexpr size_t INDEX_W3_SCALE = 6;

constexpr int64_t FP4_BLOCK = 32;
constexpr int64_t FP4_PER_BYTE = 2;
// DeepSeek-V4 architectural constant: the gate activation is clamped to
// +/- 10.0 before the activation. Not a tunable and not an overflow guard --
// it is part of the model definition, so the kernel, the CPU reference and the
// goldens all have to apply it or they implement different functions.
constexpr float SWIGLU_LIMIT = 10.0f;
// Milestone budget: packed weight matrix + scales + activations must fit the
// AIV unified buffer in one load (196 KB classic UB, half reserved for
// staging/queues -> 96 KiB for the packed matrix).
constexpr int64_t MAX_PACKED_WEIGHT_BYTES = 96 * 1024;

int64_t DimOr(const gert::Shape *shape, size_t index, int64_t fallback = 1)
{
    return (shape != nullptr && shape->GetDimNum() > index) ? shape->GetDim(index) : fallback;
}
} // namespace

ge::graphStatus Tiling4Dsv4MoeExpert(gert::TilingContext *context)
{
    const gert::Shape *w1Shape = &context->GetInputShape(INDEX_W1)->GetStorageShape();
    const gert::Shape *w2Shape = &context->GetInputShape(INDEX_W2)->GetStorageShape();
    const gert::Shape *xShape = &context->GetInputShape(INDEX_X)->GetStorageShape();
    if (w1Shape == nullptr || w2Shape == nullptr || xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // w1/w3: [inter, hidden/2] packed; w2: [hidden, inter/2] packed.
    const int64_t interSize = DimOr(w1Shape, 0);
    const int64_t hiddenSize = DimOr(w2Shape, 0);

    auto problem = [&](const char *what) {
        OPS_LOG_E(context, "dsv4_moe_expert: %s (w1=[%ld,%ld], w2=[%ld,%ld], x=[%ld,%ld])", what,
                DimOr(w1Shape, 0), DimOr(w1Shape, 1), DimOr(w2Shape, 0), DimOr(w2Shape, 1), DimOr(xShape, 0),
                DimOr(xShape, 1));
    };

    if (DimOr(w1Shape, 1) != hiddenSize / FP4_PER_BYTE) {
        problem("w1 packed cols must equal hidden/2");
        return ge::GRAPH_FAILED;
    }
    if (DimOr(w2Shape, 1) != interSize / FP4_PER_BYTE) {
        problem("w2 packed cols must equal inter/2");
        return ge::GRAPH_FAILED;
    }
    if (DimOr(xShape, 0) != 1 || DimOr(xShape, 1) != hiddenSize) {
        problem("x must be [1, hidden]");
        return ge::GRAPH_FAILED;
    }
    for (size_t scaleIndex = INDEX_W1_SCALE; scaleIndex <= INDEX_W3_SCALE; ++scaleIndex) {
        const gert::StorageShape *scaleStorage = context->GetInputShape(scaleIndex);
        const gert::Shape *scaleShape = &scaleStorage->GetStorageShape();
        const int64_t rows = (scaleIndex == INDEX_W2_SCALE) ? hiddenSize : interSize;
        const int64_t cols = (scaleIndex == INDEX_W2_SCALE) ? interSize : hiddenSize;
        if (scaleShape == nullptr || DimOr(scaleShape, 0) != rows || DimOr(scaleShape, 1) != cols / FP4_BLOCK) {
            problem("scale views must be [rows, cols/32]");
            return ge::GRAPH_FAILED;
        }
    }
    if (hiddenSize % (2 * FP4_BLOCK) != 0 || interSize % (2 * FP4_BLOCK) != 0) {
        problem("hidden/inter must be multiples of 64 (row packing + block alignment)");
        return ge::GRAPH_FAILED;
    }
    const int64_t packedWeightBytes = interSize * hiddenSize / FP4_PER_BYTE;
    if (packedWeightBytes > MAX_PACKED_WEIGHT_BYTES) {
        OPS_LOG_E(context,
                "dsv4_moe_expert: packed weight %ld KiB exceeds the single-load milestone budget %ld KiB; "
                "chunked multi-core tiling is the production follow-up",
                packedWeightBytes / 1024, MAX_PACKED_WEIGHT_BYTES / 1024);
        return ge::GRAPH_FAILED;
    }

    Dsv4MoeExpertTilingData tilingData;
    tilingData.set_hiddenSize(hiddenSize);
    tilingData.set_interSize(interSize);
    tilingData.set_blockSize(FP4_BLOCK);
    tilingData.set_swigluLimit(SWIGLU_LIMIT);
    tilingData.set_tilingReserved(0.0f);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    context->SetTilingKey(0);
    context->SetBlockDim(1); // single AIV core milestone
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    *workspaceSizes = 0;

    OPS_LOG_I(context, "dsv4_moe_expert tiling: hidden=%ld, inter=%ld, coreNum=1", hiddenSize, interSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Dsv4MoeExpert(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Dsv4MoeExpert)
    .Tiling(Tiling4Dsv4MoeExpert)
    .TilingParse<Dsv4MoeExpertCompileInfo>(TilingPrepare4Dsv4MoeExpert);

} // namespace optiling
