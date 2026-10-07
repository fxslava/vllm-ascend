/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software and can be redistributed under the terms
 * and conditions of the CANN Open Software License Agreement Version 2.0.
 * See the License for details.
 */

// Single-AIV column tiling supports production H=7168/I=2048.
// All weights stream through UB; the device's UB size bounds the geometry.

#include "dsv4_moe_expert_tiling.h"

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "graph/types.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{
namespace
{
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
// The envelope the CAModel and device suites have actually validated. These
// are a tested-shape bound, not a capacity bound -- the UB check below is what
// keeps the kernel inside the hardware it is about to run on.
constexpr int64_t MAX_HIDDEN_SIZE = 7168;
constexpr int64_t MAX_INTER_SIZE = 2048;

// The explicit TBuf payload of op_kernel's InitBuffers(), which is what the
// kernel asks TPipe for and TPipe hands out of UB. The 8x512 streaming chunk
// fixes every weight-side buffer, so the total is a constant plus a term
// linear in the geometry. Both halves are derived from the chunk constants
// below rather than written down, so reshaping the chunk reprices the check.
// csrc/tests/common/dsv4_test_oracle.hpp computes the same total for the host
// tier, which has no CANN toolkit to call this function with.
constexpr int64_t UB_ALIGN = 32;
constexpr int64_t CHUNK_FLAT_ELEMS = 4096;                                // 8 rows x 512 columns
constexpr int64_t MAX_CHUNK_COLS = 512;
constexpr int64_t MAX_MERGE_ELEMS = 2 * CHUNK_FLAT_ELEMS / FP4_BLOCK;     // low+high nibble block sums
constexpr int64_t REDUCE_SLOT_ELEMS = UB_ALIGN / static_cast<int64_t>(sizeof(float));
constexpr int64_t OUT_ROW_COUNT = 3;                                      // gate, up, activated

// Weight and padded scale staging; three decode stages plus the products;
// even and odd input columns; two reduction-slot buffers; two merge buffers
// and the decoded scales; the decode predicate scratch; one bf16 chunk row.
constexpr int64_t UB_CHUNK_BYTES = CHUNK_FLAT_ELEMS / FP4_PER_BYTE + MAX_MERGE_ELEMS +
                                   4 * (CHUNK_FLAT_ELEMS * 4) + 2 * (MAX_CHUNK_COLS / FP4_PER_BYTE * 4) +
                                   2 * (MAX_MERGE_ELEMS * REDUCE_SLOT_ELEMS * 4) + 3 * (MAX_MERGE_ELEMS * 4) +
                                   CHUNK_FLAT_ELEMS / FP4_PER_BYTE * 2 + MAX_MERGE_ELEMS;

// The host tier carries this same total as kExpertStreamingScratchBytes, and
// test_host_dsv4_moe_expert.cpp pins the 7168x2048 figure that follows from
// it. This is what stops the two mirrors drifting apart.
static_assert(UB_CHUNK_BYTES == 93696, "chunk payload changed; update dsv4_test_oracle.hpp to match");

// bf16 input, the fp32 input that doubles as the down accumulator, the three
// fp32 leg buffers, and the three bf16 output rows. Both dimensions are
// multiples of 64, so every term is already a multiple of UB_ALIGN.
uint64_t UbFootprintBytes(int64_t hiddenSize, int64_t interSize)
{
    const int64_t maxCols = hiddenSize > interSize ? hiddenSize : interSize;
    const int64_t geometry = 2 * hiddenSize + 4 * maxCols + (3 * 4 + OUT_ROW_COUNT * 2) * interSize;
    return static_cast<uint64_t>(UB_CHUNK_BYTES + geometry);
}

// Ascend910B1..B4 carry 192 KiB of UB per core, the Ascend950 regbase core
// more. When the platform cannot be queried, assume the smaller of the two
// supported parts rather than the one being built for.
constexpr uint64_t FALLBACK_UB_SIZE = 192UL * 1024UL;

uint64_t ResolveUbSize(gert::TilingContext *context)
{
    auto *platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr)
    {
        return FALLBACK_UB_SIZE;
    }
    uint64_t ubSize = 0;
    platform_ascendc::PlatformAscendC(platformInfo).GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    return ubSize == 0 ? FALLBACK_UB_SIZE : ubSize;
}

int64_t DimOr(const gert::Shape *shape, size_t index, int64_t fallback = 1)
{
    return (shape != nullptr && shape->GetDimNum() > index) ? shape->GetDim(index) : fallback;
}
}  // namespace

ge::graphStatus Tiling4Dsv4MoeExpert(gert::TilingContext *context)
{
    const gert::Shape *w1Shape = &context->GetInputShape(INDEX_W1)->GetStorageShape();
    const gert::Shape *w2Shape = &context->GetInputShape(INDEX_W2)->GetStorageShape();
    const gert::Shape *xShape = &context->GetInputShape(INDEX_X)->GetStorageShape();
    if (w1Shape == nullptr || w2Shape == nullptr || xShape == nullptr)
    {
        return ge::GRAPH_FAILED;
    }

    // w1/w3: [inter, hidden/2] packed; w2: [hidden, inter/2] packed.
    const int64_t interSize = DimOr(w1Shape, 0);
    const int64_t hiddenSize = DimOr(w2Shape, 0);

    auto problem = [&](const char *what)
    {
        OPS_LOG_E(context, "dsv4_moe_expert: %s (w1=[%ld,%ld], w2=[%ld,%ld], x=[%ld,%ld])", what, DimOr(w1Shape, 0),
                  DimOr(w1Shape, 1), DimOr(w2Shape, 0), DimOr(w2Shape, 1), DimOr(xShape, 0), DimOr(xShape, 1));
    };

    if (DimOr(w1Shape, 1) != hiddenSize / FP4_PER_BYTE)
    {
        problem("w1 packed cols must equal hidden/2");
        return ge::GRAPH_FAILED;
    }
    if (DimOr(w2Shape, 1) != interSize / FP4_PER_BYTE)
    {
        problem("w2 packed cols must equal inter/2");
        return ge::GRAPH_FAILED;
    }
    if (DimOr(xShape, 0) != 1 || DimOr(xShape, 1) != hiddenSize)
    {
        problem("x must be [1, hidden]");
        return ge::GRAPH_FAILED;
    }
    for (size_t scaleIndex = INDEX_W1_SCALE; scaleIndex <= INDEX_W3_SCALE; ++scaleIndex)
    {
        const gert::StorageShape *scaleStorage = context->GetInputShape(scaleIndex);
        const gert::Shape *scaleShape = &scaleStorage->GetStorageShape();
        const int64_t rows = (scaleIndex == INDEX_W2_SCALE) ? hiddenSize : interSize;
        const int64_t cols = (scaleIndex == INDEX_W2_SCALE) ? interSize : hiddenSize;
        if (scaleShape == nullptr || DimOr(scaleShape, 0) != rows || DimOr(scaleShape, 1) != cols / FP4_BLOCK)
        {
            problem("scale views must be [rows, cols/32]");
            return ge::GRAPH_FAILED;
        }
    }
    if (hiddenSize % (2 * FP4_BLOCK) != 0 || interSize % (2 * FP4_BLOCK) != 0)
    {
        problem("hidden/inter must be multiples of 64 (row packing + block alignment)");
        return ge::GRAPH_FAILED;
    }
    if (hiddenSize <= 0 || interSize <= 0 || hiddenSize > MAX_HIDDEN_SIZE || interSize > MAX_INTER_SIZE)
    {
        OPS_LOG_E(context, "dsv4_moe_expert: hidden=%ld inter=%ld outside validated limits (%ld,%ld)", hiddenSize,
                  interSize, MAX_HIDDEN_SIZE, MAX_INTER_SIZE);
        return ge::GRAPH_FAILED;
    }
    // The UB per core differs between the supported parts, so check the
    // kernel's own allocation against the one it will run on. Without this the
    // geometry limits above would be the only guard, and they were chosen
    // against the larger part.
    const uint64_t ubSize = ResolveUbSize(context);
    const uint64_t footprint = UbFootprintBytes(hiddenSize, interSize);
    if (footprint > ubSize)
    {
        OPS_LOG_E(context,
                  "dsv4_moe_expert: hidden=%ld inter=%ld needs %lu UB bytes, this SoC has %lu per core", hiddenSize,
                  interSize, static_cast<unsigned long>(footprint), static_cast<unsigned long>(ubSize));
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
    context->SetBlockDim(1);  // single AIV baseline
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    *workspaceSizes = 0;

    OPS_LOG_I(context, "dsv4_moe_expert tiling: hidden=%ld, inter=%ld, coreNum=1, ub=%lu/%lu", hiddenSize, interSize,
              static_cast<unsigned long>(footprint), static_cast<unsigned long>(ubSize));
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

}  // namespace optiling
