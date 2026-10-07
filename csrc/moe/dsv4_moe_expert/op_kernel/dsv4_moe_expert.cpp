/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software and can be redistributed under the terms
 * and conditions of the CANN Open Software License Agreement Version 2.0.
 * See the License for details.
 */

/*!
 * DeepSeek-V4 routed expert: BF16 x, E2M1 packed FP4 weights, block-32
 * E8M0 scales, FP32 accumulators, symmetric gate clamp at +/-10, BF16
 * round-to-nearest-even activation before the down projection.
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "dsv4_moe_expert_tiling_data.h"
#include "dsv4_moe_expert_vector_compat.h"

#ifndef CANN_VERSION_MAJOR
#define CANN_VERSION_MAJOR 8
#endif

using namespace AscendC;

namespace Dsv4MoeExpertOp {

__aicore__ inline void VectorDependencyBarrier()
{
#if __CCE_AICORE__ == 220
    PipeBarrier<PIPE_V>();
#endif
}

// Explicit normal-mask repeats avoid CANN 8 arch220's calcount BF16 casts.
// Preserve nearest-even rounding at the activation and output boundaries.
template <typename Dst, typename Src>
__aicore__ inline void CastBf16Boundary(const LocalTensor<Dst>& dst,
                                      const LocalTensor<Src>& src, RoundMode mode, uint32_t count)
{
#if __CCE_AICORE__ == 220
    constexpr uint32_t FLOAT_LANES = 64;
    constexpr uint8_t FLOAT_BLOCKS = 8;
    constexpr uint8_t BF16_BLOCKS = 4;
    const UnaryRepeatParams strides = sizeof(Dst) > sizeof(Src)
        ? UnaryRepeatParams{1, 1, FLOAT_BLOCKS, BF16_BLOCKS}
        : UnaryRepeatParams{1, 1, BF16_BLOCKS, FLOAT_BLOCKS};
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    for (uint32_t base = 0; base < count; base += FLOAT_LANES) {
        const uint32_t remaining = count - base;
        const uint64_t active = remaining < FLOAT_LANES ? remaining : FLOAT_LANES;
        Cast(dst[base], src[base], mode, active, 1, strides);
        PipeBarrier<PIPE_V>();
    }
    ResetMask();
#else
    Cast(dst, src, mode, count);
#endif
}

#if __CCE_AICORE__ == 220
constexpr RoundMode BF16_ROUND_MODE = RoundMode::CAST_RINT;
#else
constexpr RoundMode BF16_ROUND_MODE = RoundMode::CAST_ROUND;
#endif

__aicore__ inline void ReduceHalfBlock(const LocalTensor<float>& dst,
                                      const LocalTensor<float>& src, int32_t count)
{
#if (CANN_VERSION_MAJOR >= 9)
    ReduceRepeat<ReduceType::SUM>(dst, src, count, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
#else
    WholeReduceSum(dst, src, count, 1, 1, 1, 8);
#endif
}

template <typename T>
__aicore__ inline void CompareScalarCompat(const LocalTensor<uint8_t>& dst,
                                         const LocalTensor<T>& src, T value,
                                         CMPMODE mode, uint32_t count)
{
#if (CANN_VERSION_MAJOR >= 9)
    Compares(dst, src, value, mode, count);
#else
    CompareScalar(dst, src, value, mode, count);
#endif
}

constexpr int64_t FP4_BLOCK = 32;       // E8M0 scale span (logical elements)
constexpr int64_t FP4_PER_BYTE = 2;     // two E2M1 nibbles per storage byte
constexpr int64_t BYTES_ALIGN = 32;     // MTE 32-byte granularity
constexpr int64_t OUT_ROW_COUNT = 3;    // gate/up/activated share one staging block
constexpr int64_t CHUNK_FLAT_ELEMS = 4096;
constexpr int64_t MIN_CHUNK_ROWS = 8;
constexpr int64_t MAX_CHUNK_COLS = CHUNK_FLAT_ELEMS / MIN_CHUNK_ROWS;
constexpr int64_t MAX_MERGE_ELEMS = 2 * CHUNK_FLAT_ELEMS / FP4_BLOCK;
constexpr int64_t REDUCE_SLOT_ELEMS = BYTES_ALIGN / sizeof(float);
constexpr int32_t REDUCE_SLOT_BYTE_SHIFT = 5;

constexpr float SWIGLU_BETA = 1.0f;

constexpr uint8_t STAGING_EVENT_ID = 3; // MTE2_V / V_MTE2 weight-staging ring
constexpr uint8_t ROW_EVENT_ID = 3;     // V_MTE3 / MTE3_V bf16-row ring
constexpr uint8_t OUT_EVENT_ID = 4;     // V_MTE3 one-shot before CopyOut
constexpr uint8_t X_EVENT_ID = 4;       // MTE2_V one-shot after the x load

__aicore__ inline int64_t AlignUpBytes(int64_t bytes)
{
    return (bytes + BYTES_ALIGN - 1) / BYTES_ALIGN * BYTES_ALIGN;
}

class Dsv4MoeExpertKernel {
public:
    __aicore__ Dsv4MoeExpertKernel(GM_ADDR x, GM_ADDR w1, GM_ADDR w2, GM_ADDR w3, GM_ADDR w1Scale,
                                   GM_ADDR w2Scale, GM_ADDR w3Scale, GM_ADDR gateOut, GM_ADDR upOut,
                                   GM_ADDR activatedOut, GM_ADDR downOut, GM_ADDR tilingPtr, TPipe *pipe)
        : pipe_(pipe)
    {
        GET_TILING_DATA_WITH_STRUCT(Dsv4MoeExpertTilingData, tilingData, tilingPtr);
        hidden_ = tilingData.hiddenSize;
        inter_ = tilingData.interSize;
        swigluLimit_ = tilingData.swigluLimit;

        xGmU16_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(x));
        wqGm_[0].SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(w1));
        wqGm_[1].SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(w2));
        wqGm_[2].SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(w3));
        wqScaleGm_[0].SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(w1Scale));
        wqScaleGm_[1].SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(w2Scale));
        wqScaleGm_[2].SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(w3Scale));
        legOutGm_[0].SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(gateOut));
        legOutGm_[1].SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(downOut));
        legOutGm_[2].SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(upOut));
        activatedGmU16_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(activatedOut));

        InitBuffers();
        xFloat_ = xFloatBuf_.Get<float>();
        downAcc_ = xFloatBuf_.Get<float>();
        gateAcc_ = gateBuf_.Get<float>();
        upAcc_ = upBuf_.Get<float>();
        activated_ = activatedBuf_.Get<float>();
    }

    __aicore__ void Process()
    {
        CopyInX();
        ProjectLeg(LEG_W1, xFloat_, gateAcc_);
        ProjectLeg(LEG_W3, xFloat_, upAcc_);
        ApplySwiGLU();
        ProjectLeg(LEG_W2, activated_, downAcc_);
        CopyOut();
    }

private:
    static constexpr int32_t LEG_W1 = 0;
    static constexpr int32_t LEG_W2 = 1;
    static constexpr int32_t LEG_W3 = 2;

    static constexpr int32_t E2M1_HALF_BITS = 0x3F000000; // +0.5
    static constexpr int32_t E2M1_ZERO_BITS = 0x00000000;

    __aicore__ void InitBuffers()
    {
        const int64_t maxCols = hidden_ > inter_ ? hidden_ : inter_;
        const int64_t xFloatBytes = AlignUpBytes(maxCols * static_cast<int64_t>(sizeof(float)));

        pipe_->InitBuffer(xBitsBuf_, AlignUpBytes(hidden_ * 2));
        pipe_->InitBuffer(xFloatBuf_, xFloatBytes);
        pipe_->InitBuffer(gateBuf_, AlignUpBytes(inter_ * static_cast<int64_t>(sizeof(float))));
        pipe_->InitBuffer(upBuf_, AlignUpBytes(inter_ * static_cast<int64_t>(sizeof(float))));
        pipe_->InitBuffer(activatedBuf_, AlignUpBytes(inter_ * static_cast<int64_t>(sizeof(float))));
        pipe_->InitBuffer(interRowBuf_, AlignUpBytes(OUT_ROW_COUNT * inter_ * 2));

        pipe_->InitBuffer(wStagingBuf_, AlignUpBytes(CHUNK_FLAT_ELEMS / FP4_PER_BYTE));
        pipe_->InitBuffer(sStagingBuf_, AlignUpBytes(MAX_MERGE_ELEMS));
        pipe_->InitBuffer(stageABuf_, AlignUpBytes(CHUNK_FLAT_ELEMS * 4));
        pipe_->InitBuffer(stageBBuf_, AlignUpBytes(CHUNK_FLAT_ELEMS * 4));
        pipe_->InitBuffer(stageCBuf_, AlignUpBytes(CHUNK_FLAT_ELEMS * 4));
        pipe_->InitBuffer(prodBuf_, CHUNK_FLAT_ELEMS * 4);
        pipe_->InitBuffer(xEvenBuf_, MAX_CHUNK_COLS / FP4_PER_BYTE * 4);
        pipe_->InitBuffer(xOddBuf_, MAX_CHUNK_COLS / FP4_PER_BYTE * 4);
        pipe_->InitBuffer(partHiBuf_, MAX_MERGE_ELEMS * REDUCE_SLOT_ELEMS * 4);
        pipe_->InitBuffer(partRawBuf_, MAX_MERGE_ELEMS * REDUCE_SLOT_ELEMS * 4);
        pipe_->InitBuffer(mergeBuf1_, MAX_MERGE_ELEMS * 4);
        pipe_->InitBuffer(mergeBuf2_, MAX_MERGE_ELEMS * 4);
        pipe_->InitBuffer(scalesF32Buf_, MAX_MERGE_ELEMS * 4);
        pipe_->InitBuffer(decMaskBuf_, AlignUpBytes(CHUNK_FLAT_ELEMS / FP4_PER_BYTE * 2));
        pipe_->InitBuffer(rowBf16Buf_, MAX_MERGE_ELEMS);
    }

    __aicore__ void WaitLoad()
    {
        SetFlag<HardEvent::MTE2_V>(STAGING_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(STAGING_EVENT_ID);
    }

    __aicore__ void CopyInX()
    {
        LocalTensor<uint16_t> xBits = xBitsBuf_.Get<uint16_t>();
        DataCopy(xBits, xGmU16_, hidden_);
        SetFlag<HardEvent::MTE2_V>(X_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(X_EVENT_ID);
        CastBf16Boundary(xFloat_, xBits.ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE,
                         static_cast<uint32_t>(hidden_));
    }

    __aicore__ void ProjectLeg(int32_t legIndex, const LocalTensor<float> &xFull, LocalTensor<float> &legAcc)
    {
        const int64_t fullCols = (legIndex == LEG_W2) ? inter_ : hidden_;
        const int64_t cols = fullCols < MAX_CHUNK_COLS ? fullCols : MAX_CHUNK_COLS;
        const int64_t rows = (legIndex == LEG_W2) ? hidden_ : inter_;
        const int64_t chunkRows = MIN_CHUNK_ROWS;
        const int64_t pow2Blocks = BYTES_ALIGN;
        const int64_t chunkScales = chunkRows * pow2Blocks;

        LocalTensor<uint8_t> packed = wStagingBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> scales = sStagingBuf_.Get<uint8_t>();
        LocalTensor<float> products = prodBuf_.Get<float>();
        LocalTensor<int32_t> codes = prodBuf_.Get<int32_t>();
        LocalTensor<int32_t> stageA = stageABuf_.Get<int32_t>();
        LocalTensor<int32_t> stageB = stageBBuf_.Get<int32_t>();
        LocalTensor<int32_t> stageC = stageCBuf_.Get<int32_t>();
        LocalTensor<uint8_t> decMask = decMaskBuf_.Get<uint8_t>();
        LocalTensor<float> partHi = partHiBuf_.Get<float>();
        LocalTensor<float> xEven = xEvenBuf_.Get<float>();
        LocalTensor<float> xOdd = xOddBuf_.Get<float>();

        LocalTensor<float> partRaw = partRawBuf_.Get<float>();
        LocalTensor<float> merge1 = mergeBuf1_.Get<float>();
        LocalTensor<float> merge2 = mergeBuf2_.Get<float>();
        LocalTensor<uint16_t> rowBf16 = rowBf16Buf_.Get<uint16_t>();

        SetFlag<HardEvent::V_MTE2>(STAGING_EVENT_ID);
        SetFlag<HardEvent::MTE3_V>(ROW_EVENT_ID);
        for (int64_t row0 = 0; row0 < rows; row0 += chunkRows)
        {
            LocalTensor<float> accRow = legAcc[static_cast<uint32_t>(row0)];
            Duplicate(accRow, 0.0f, static_cast<uint32_t>(chunkRows));
            for (int64_t col0 = 0; col0 < fullCols; col0 += cols)
            {
                const int64_t activeCols = fullCols - col0 < cols ? fullCols - col0 : cols;
                const int64_t activeHalfCols = activeCols / FP4_PER_BYTE;
                const int64_t activeBlocks = activeCols / FP4_BLOCK;
                const int64_t activeBytes = chunkRows * activeHalfCols;
                SplitColumns(xEven, xOdd, xFull[static_cast<uint32_t>(col0)], static_cast<uint32_t>(activeCols));
                WaitFlag<HardEvent::V_MTE2>(STAGING_EVENT_ID);
                for (int64_t r = 0; r < chunkRows; ++r)
                {
                    DataCopy(packed[static_cast<uint32_t>(r * activeHalfCols)],
                             wqGm_[legIndex][((row0 + r) * fullCols + col0) / FP4_PER_BYTE], activeHalfCols);
                    DataCopyPad(scales[static_cast<uint32_t>(r * pow2Blocks)],
                                wqScaleGm_[legIndex][((row0 + r) * fullCols + col0) / FP4_BLOCK],
                                DataCopyParams{1, static_cast<uint16_t>(activeBlocks), 0, 0},
                                DataCopyPadParams{true, 0, static_cast<uint8_t>(pow2Blocks - activeBlocks), 0});
                }
                WaitLoad();

                DecodeCodes(packed, codes, stageA, stageB, stageC, decMask, static_cast<uint32_t>(activeBytes));

                for (int64_t r = 0; r < chunkRows; ++r)
                {
                    const uint32_t lowOff = static_cast<uint32_t>(r * activeHalfCols);
                    const uint32_t highOff = static_cast<uint32_t>(activeBytes + r * activeHalfCols);
                    Mul(products[lowOff], products[lowOff], xEven, static_cast<uint32_t>(activeHalfCols));
                    Mul(products[highOff], products[highOff], xOdd, static_cast<uint32_t>(activeHalfCols));
                }

                VectorDependencyBarrier();
                Duplicate(partRaw, 0.0f, static_cast<uint32_t>(MAX_MERGE_ELEMS * REDUCE_SLOT_ELEMS));
                Duplicate(partHi, 0.0f, static_cast<uint32_t>(MAX_MERGE_ELEMS * REDUCE_SLOT_ELEMS));
                VectorDependencyBarrier();
                const int64_t halfBlock = FP4_BLOCK / FP4_PER_BYTE;
                for (int64_t r = 0; r < chunkRows; ++r)
                {
                    const LocalTensor<float> lowRow = products[static_cast<uint32_t>(r * activeHalfCols)];
                    const LocalTensor<float> highRow = products[static_cast<uint32_t>(activeBytes + r * activeHalfCols)];
                    for (int64_t b = 0; b < activeBlocks; ++b)
                    {
                        const uint32_t slot = static_cast<uint32_t>((r * pow2Blocks + b) * REDUCE_SLOT_ELEMS);
                        ReduceHalfBlock(partRaw[slot], lowRow[static_cast<uint32_t>(b * halfBlock)],
                                        static_cast<int32_t>(halfBlock));
                        ReduceHalfBlock(partHi[slot], highRow[static_cast<uint32_t>(b * halfBlock)],
                                        static_cast<int32_t>(halfBlock));
                    }
                }
                VectorDependencyBarrier();
                LocalTensor<int32_t> offsets = scalesF32Buf_.Get<int32_t>();
                CreateGatherIndices(offsets, static_cast<uint32_t>(MAX_MERGE_ELEMS));
                ShiftLeft(offsets, offsets, REDUCE_SLOT_BYTE_SHIFT, static_cast<uint32_t>(MAX_MERGE_ELEMS));
                VectorDependencyBarrier();
                Gather(merge1, partRaw, offsets.ReinterpretCast<uint32_t>(), 0,
                       static_cast<uint32_t>(MAX_MERGE_ELEMS));
                Gather(merge2, partHi, offsets.ReinterpretCast<uint32_t>(), 0,
                       static_cast<uint32_t>(MAX_MERGE_ELEMS));
                VectorDependencyBarrier();
                Add(partRaw, merge1, merge2, static_cast<uint32_t>(MAX_MERGE_ELEMS));
                VectorDependencyBarrier();

                DecodeScales(scales, static_cast<uint32_t>(chunkScales));
                SetFlag<HardEvent::V_MTE2>(STAGING_EVENT_ID);
                const LocalTensor<float> scaleF32 = scalesF32Buf_.Get<float>();
                Mul(partRaw, partRaw, scaleF32, static_cast<uint32_t>(chunkScales));
                VectorDependencyBarrier();

                int64_t mergeLen = chunkRows * pow2Blocks;
                LocalTensor<float> cur = partRaw;
                while (mergeLen > 2 * chunkRows)
                {
                    SplitColumns(merge1, merge2, cur, static_cast<uint32_t>(mergeLen));
                    Add(cur, merge1, merge2, static_cast<uint32_t>(mergeLen / 2));
                    VectorDependencyBarrier();
                    mergeLen /= 2;
                }
                SplitColumns(merge1, merge2, cur, static_cast<uint32_t>(mergeLen));
                Add(merge1, merge1, merge2, static_cast<uint32_t>(chunkRows));
                VectorDependencyBarrier();
                Add(accRow, accRow, merge1, static_cast<uint32_t>(chunkRows));
                VectorDependencyBarrier();
            }

            WaitFlag<HardEvent::MTE3_V>(ROW_EVENT_ID);
            CastBf16Boundary(rowBf16.ReinterpretCast<bfloat16_t>(), accRow, BF16_ROUND_MODE,
                             static_cast<uint32_t>(chunkRows));
            SetFlag<HardEvent::V_MTE3>(ROW_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(ROW_EVENT_ID);
            DataCopyPad(legOutGm_[legIndex][row0], rowBf16, DataCopyParams{1, static_cast<uint16_t>(chunkRows * 2), 0, 0});
            SetFlag<HardEvent::MTE3_V>(ROW_EVENT_ID);
        }
        WaitFlag<HardEvent::V_MTE2>(STAGING_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(ROW_EVENT_ID);
    }

    __aicore__ void BitAnd(const LocalTensor<int32_t>& dst, const LocalTensor<int32_t>& lhs,
                          const LocalTensor<int32_t>& rhs, uint32_t count)
    {
        BitAnd32(dst, lhs, rhs, count);
    }

    __aicore__ void BitOr(const LocalTensor<int32_t>& dst, const LocalTensor<int32_t>& lhs,
                         const LocalTensor<int32_t>& rhs, uint32_t count)
    {
        BitOr32(dst, lhs, rhs, count);
    }

    __aicore__ void SplitColumns(const LocalTensor<float>& even, const LocalTensor<float>& odd,
                                const LocalTensor<float>& src, uint32_t count)
    {
#if __CCE_AICORE__ == 220
        LocalTensor<int32_t> offsets = stageABuf_.Get<int32_t>();
        const uint32_t halfCount = count / FP4_PER_BYTE;
        CreateGatherIndices(offsets, halfCount);
        ShiftLeft(offsets, offsets, 3, halfCount);
        PipeBarrier<PIPE_V>();
        Gather(even, src, offsets.ReinterpretCast<uint32_t>(), 0, halfCount);
        PipeBarrier<PIPE_V>();
        Adds(offsets, offsets, static_cast<int32_t>(sizeof(float)), halfCount);
        PipeBarrier<PIPE_V>();
        Gather(odd, src, offsets.ReinterpretCast<uint32_t>(), 0, halfCount);
        PipeBarrier<PIPE_V>();
#else
        DeInterleave(even, odd, src, static_cast<int32_t>(count));
#endif
    }

    __aicore__ void DecodeCodes(const LocalTensor<uint8_t> &packed, const LocalTensor<int32_t> &codes,
                                const LocalTensor<int32_t> &stageA, const LocalTensor<int32_t> &stageB,
                                const LocalTensor<int32_t> &stageC, const LocalTensor<uint8_t> &decMask,
                                uint32_t byteCount)
    {
        const uint32_t count = byteCount * FP4_PER_BYTE;

#if __CCE_AICORE__ == 220
        WidenBytes32(stageA, packed, stageB, byteCount);
#else
        LocalTensor<uint16_t> wide = stageB.template ReinterpretCast<uint16_t>();
        Cast(wide, packed, RoundMode::CAST_NONE, byteCount);
        VectorDependencyBarrier();
        Cast(stageA.template ReinterpretCast<uint32_t>(), wide, RoundMode::CAST_NONE, byteCount);
        VectorDependencyBarrier();
#endif

        Duplicate(stageB, 0x0F, byteCount);
        VectorDependencyBarrier();
        BitAnd(codes, stageA, stageB, byteCount);
        VectorDependencyBarrier();
        ShiftRight(codes[byteCount], stageA, 4, byteCount);
        VectorDependencyBarrier();

        Duplicate(stageC, 7, count);
        VectorDependencyBarrier();
        BitAnd(stageB, codes, stageC, count); // stageB = mag
        VectorDependencyBarrier();

        ShiftRight(stageA, stageB, 1, count); // e = mag >> 1
        VectorDependencyBarrier();
        Adds(stageA, stageA, 126, count);     // e + 126
        VectorDependencyBarrier();
        ShiftLeft(stageA, stageA, 23, count); // (e + 126) << 23
        VectorDependencyBarrier();

#if __CCE_AICORE__ == 220
        // The low mantissa bit is active only for normal magnitudes (mag >= 2).
        Duplicate(stageC, 1, count);
        VectorDependencyBarrier();
        BitAnd(stageC, stageB, stageC, count); // stageC = mag & 1
        VectorDependencyBarrier();
        ShiftRight(stageB, stageB, 1, count);
        VectorDependencyBarrier();
        Mins(stageB, stageB, 1, count);
        VectorDependencyBarrier();
        BitAnd(stageC, stageC, stageB, count);
        VectorDependencyBarrier();
        ShiftLeft(stageC, stageC, 22, count);
        VectorDependencyBarrier();
        BitOr(stageA, stageA, stageC, count); // assemble magnitude
        VectorDependencyBarrier();

        // Reuse full-sized decode stages, never the smaller predicate buffer.
        // Expand mag != 0 to an all-bit mask before touching IEEE-754 fields.
        Duplicate(stageC, 7, count);
        VectorDependencyBarrier();
        BitAnd(stageB, codes, stageC, count);
        VectorDependencyBarrier();
        Mins(stageB, stageB, 1, count);
        VectorDependencyBarrier();
        Duplicate(stageC, 0, count);
        VectorDependencyBarrier();
        Sub(stageB, stageC, stageB, count);
        VectorDependencyBarrier();
        BitAnd(stageA, stageA, stageB, count);
        VectorDependencyBarrier();
#else
        Duplicate(stageC, 1, count);
        VectorDependencyBarrier();
        BitAnd(stageC, stageB, stageC, count);
        VectorDependencyBarrier();
        ShiftLeft(stageC, stageC, 22, count);
        VectorDependencyBarrier();
        BitOr(stageA, stageA, stageC, count);
        VectorDependencyBarrier();

        Duplicate(stageC, E2M1_HALF_BITS, count);
        VectorDependencyBarrier();
        CompareScalarCompat(decMask, stageB, int32_t{1}, CMPMODE::EQ, count);
        VectorDependencyBarrier();
        Select(stageA, decMask, stageC, stageA, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        VectorDependencyBarrier();

        Duplicate(stageC, E2M1_ZERO_BITS, count);
        VectorDependencyBarrier();
        CompareScalarCompat(decMask, stageB, int32_t{0}, CMPMODE::EQ, count);
        VectorDependencyBarrier();
        Select(stageA, decMask, stageC, stageA, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        VectorDependencyBarrier();
#endif

        // Append Sign: bit 3 into bit 31
        Duplicate(stageC, 8, count);
        VectorDependencyBarrier();
        BitAnd(stageC, codes, stageC, count);
        VectorDependencyBarrier();
        ShiftLeft(stageC, stageC, 28, count);
        VectorDependencyBarrier();
        BitOr(codes, stageA, stageC, count);
        VectorDependencyBarrier();
    }

    __aicore__ void DecodeScales(const LocalTensor<uint8_t> &scales, uint32_t count)
    {
        LocalTensor<int32_t> bits = scalesF32Buf_.Get<int32_t>();
        LocalTensor<int32_t> wrap = mergeBuf2_.Get<int32_t>();
        LocalTensor<int32_t> scratch = mergeBuf1_.Get<int32_t>();
        LocalTensor<uint8_t> special = decMaskBuf_.Get<uint8_t>();

#if __CCE_AICORE__ == 220
        WidenBytes32(bits, scales, wrap, count);
#else
        LocalTensor<uint32_t> bitsU32 = scalesF32Buf_.Get<uint32_t>();
        LocalTensor<uint16_t> wide = mergeBuf2_.Get<uint16_t>();
        Cast(wide, scales, RoundMode::CAST_NONE, count);
        VectorDependencyBarrier();
        Cast(bitsU32, wide, RoundMode::CAST_NONE, count);
        VectorDependencyBarrier();
#endif

        Adds(wrap, bits, 1, count);
        VectorDependencyBarrier();
        Duplicate(scratch, 0xFF, count);
        VectorDependencyBarrier();
        BitAnd(wrap, wrap, scratch, count); // wrap = (b + 1) & 0xFF
        VectorDependencyBarrier();

        ShiftLeft(bits, bits, 23, count);   // b << 23
        VectorDependencyBarrier();

#if __CCE_AICORE__ == 220
        // Find special condition: wrap in {0, 1}
        ShiftRight(scratch, wrap, 1, count);
        VectorDependencyBarrier();
        Mins(scratch, scratch, 1, count); // 0 if special, 1 if normal
        VectorDependencyBarrier();
        // Invert condition: 1 if special, 0 if normal
        Duplicate(wrap, 1, count);
        VectorDependencyBarrier();
        Sub(scratch, wrap, scratch, count);
        VectorDependencyBarrier();
        // Expand to a full-word mask, then insert the subnormal/quiet-NaN bit.
        Duplicate(wrap, ZERO_BITS, count);
        VectorDependencyBarrier();
        Sub(scratch, wrap, scratch, count);
        VectorDependencyBarrier();
        Duplicate(wrap, SUBNORMAL_BITS, count);
        VectorDependencyBarrier();
        BitAnd(scratch, scratch, wrap, count);
        VectorDependencyBarrier();
        BitOr(bits, bits, scratch, count);
        VectorDependencyBarrier();
#else
        CompareScalarCompat(special, wrap, int32_t{2}, CMPMODE::LT, count);
        VectorDependencyBarrier();
        Duplicate(scratch, static_cast<int32_t>(SUBNORMAL_BITS), count);
        VectorDependencyBarrier();
        Select(scratch, special, scratch, static_cast<int32_t>(ZERO_BITS), SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        VectorDependencyBarrier();
        Add(bits, bits, scratch, count);
        VectorDependencyBarrier();
#endif
    }

    __aicore__ void ApplySwiGLU()
    {
        const uint32_t count = static_cast<uint32_t>(inter_);
        LocalTensor<float> clampedGate = xFloatBuf_.Get<float>(count);
        Mins(clampedGate, gateAcc_, swigluLimit_, count);
        VectorDependencyBarrier();
        Maxs(clampedGate, clampedGate, -swigluLimit_, count);
        VectorDependencyBarrier();
        SwiGLU<float, false>(activated_, upAcc_, clampedGate, SWIGLU_BETA, count);
        VectorDependencyBarrier();
        LocalTensor<bfloat16_t> rounded = interRowBuf_.Get<bfloat16_t>(count);
        CastBf16Boundary(rounded, activated_, BF16_ROUND_MODE, count);
        VectorDependencyBarrier();
        CastBf16Boundary(activated_, rounded, RoundMode::CAST_NONE, count);
        VectorDependencyBarrier();
    }

    __aicore__ void CopyOut()
    {
        LocalTensor<uint16_t> interRows = interRowBuf_.Get<uint16_t>();
        LocalTensor<uint16_t> gateRow = interRows;
        LocalTensor<uint16_t> upRow = interRows[static_cast<uint32_t>(inter_)];
        LocalTensor<uint16_t> activatedRow = interRows[static_cast<uint32_t>(2 * inter_)];
        CastBf16Boundary(gateRow.ReinterpretCast<bfloat16_t>(), gateAcc_, BF16_ROUND_MODE,
                         static_cast<uint32_t>(inter_));
        CastBf16Boundary(upRow.ReinterpretCast<bfloat16_t>(), upAcc_, BF16_ROUND_MODE,
                         static_cast<uint32_t>(inter_));
        CastBf16Boundary(activatedRow.ReinterpretCast<bfloat16_t>(), activated_, BF16_ROUND_MODE,
                         static_cast<uint32_t>(inter_));
        SetFlag<HardEvent::V_MTE3>(OUT_EVENT_ID);
        WaitFlag<HardEvent::V_MTE3>(OUT_EVENT_ID);
        DataCopy(legOutGm_[LEG_W1], gateRow, inter_);
        DataCopy(legOutGm_[LEG_W3], upRow, inter_);
        DataCopy(activatedGmU16_, activatedRow, inter_);
        SetFlag<HardEvent::MTE3_V>(OUT_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(OUT_EVENT_ID);
    }

    static constexpr int32_t SUBNORMAL_BITS = 0x00400000;
    static constexpr int32_t ZERO_BITS = 0;

    TPipe *pipe_ = nullptr;
    int64_t hidden_ = 0;
    int64_t inter_ = 0;
    float swigluLimit_ = 0.0f;

    GlobalTensor<uint16_t> xGmU16_;
    GlobalTensor<uint8_t> wqGm_[3];
    GlobalTensor<uint8_t> wqScaleGm_[3];
    GlobalTensor<uint16_t> legOutGm_[3];
    GlobalTensor<uint16_t> activatedGmU16_;

    LocalTensor<float> xFloat_;
    LocalTensor<float> downAcc_;
    LocalTensor<float> gateAcc_;
    LocalTensor<float> upAcc_;
    LocalTensor<float> activated_;

    TBuf<TPosition::VECCALC> xBitsBuf_;
    TBuf<TPosition::VECCALC> xFloatBuf_;
    TBuf<TPosition::VECCALC> gateBuf_;
    TBuf<TPosition::VECCALC> upBuf_;
    TBuf<TPosition::VECCALC> activatedBuf_;
    TBuf<TPosition::VECCALC> interRowBuf_;
    TBuf<TPosition::VECCALC> wStagingBuf_;
    TBuf<TPosition::VECCALC> sStagingBuf_;
    TBuf<TPosition::VECCALC> stageABuf_;
    TBuf<TPosition::VECCALC> stageBBuf_;
    TBuf<TPosition::VECCALC> stageCBuf_;
    TBuf<TPosition::VECCALC> xEvenBuf_;
    TBuf<TPosition::VECCALC> xOddBuf_;
    TBuf<TPosition::VECCALC> partHiBuf_;
    TBuf<TPosition::VECCALC> prodBuf_;
    TBuf<TPosition::VECCALC> partRawBuf_;
    TBuf<TPosition::VECCALC> mergeBuf1_;
    TBuf<TPosition::VECCALC> mergeBuf2_;
    TBuf<TPosition::VECCALC> scalesF32Buf_;
    TBuf<TPosition::VECCALC> decMaskBuf_;
    TBuf<TPosition::VECCALC> rowBf16Buf_;
};
} // namespace Dsv4MoeExpertOp

extern "C" __global__ __aicore__ void dsv4_moe_expert(GM_ADDR x, GM_ADDR w1, GM_ADDR w2, GM_ADDR w3,
                                                      GM_ADDR w1Scale, GM_ADDR w2Scale, GM_ADDR w3Scale,
                                                      GM_ADDR gateOut, GM_ADDR upOut, GM_ADDR activatedOut,
                                                      GM_ADDR downOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }
    REGISTER_TILING_DEFAULT(Dsv4MoeExpertTilingData);
    TPipe pipe;
    Dsv4MoeExpertOp::Dsv4MoeExpertKernel op(x, w1, w2, w3, w1Scale, w2Scale, w3Scale, gateOut, upOut, activatedOut,
                                            downOut, tiling, &pipe);
    op.Process();
    pipe.Destroy();
}
