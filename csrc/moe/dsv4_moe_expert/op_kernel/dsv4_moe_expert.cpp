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
 *
 * One AIV processes eight rows by <=512 columns at a time. Weight/scale
 * loads, vector decode and reductions are streamed; there is no Cube work.
 * FP32 reductions are reassociated relative to the ascending CPU oracle.
 * Hardware numerical parity is required: compilation is not validation.
 *
 * All staging uses TBuf with explicit sync (--cce-auto-sync=off):
 * id 0: MTE2_V / V_MTE2 weight ring, primed, re-armed per column tile, drained.
 * id 1: V_MTE3 / MTE3_V output-row ring, primed, re-armed per row tile, drained.
 * id 2: V_MTE3 output handoff and MTE3_V final output drain.
 * id 3: MTE2_V input handoff. IDs are distinct within each event namespace;
 * no helper calls an external event-using kernel. There are no scalar UB
 * data accesses and no S_* events. Vector arithmetic follows PIPE_V order.
 * E8M0 255 is NaN, not infinity; 0 is the subnormal 2^-127.
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "dsv4_moe_expert_tiling_data.h"

using namespace AscendC;

namespace Dsv4MoeExpertOp {
constexpr int64_t FP4_BLOCK = 32;       // E8M0 scale span (logical elements)
constexpr int64_t FP4_PER_BYTE = 2;     // two E2M1 nibbles per storage byte
constexpr int64_t BYTES_ALIGN = 32;     // MTE 32-byte granularity
constexpr int64_t OUT_ROW_COUNT = 3;    // gate/up/activated share one staging block
// Working-set cap of one dequant chunk, in products (fp32 elements). The chunk
// row count follows from this; the whole vector pipeline below is sized from
// CHUNK_FLAT_ELEMS, so its UB cost is geometry-independent.
constexpr int64_t CHUNK_FLAT_ELEMS = 4096;
// Chunk row counts are multiples of 8 so every accumulator slice they write
// starts on a 32-byte boundary; column tiling keeps eight rows in the working set.
constexpr int64_t MIN_CHUNK_ROWS = 8;
constexpr int64_t MAX_CHUNK_COLS = CHUNK_FLAT_ELEMS / MIN_CHUNK_ROWS;
// Reduce-partial merge buffers hold chunkRows * pow2(blocksPerRow) floats;
// pow2 at most doubles blocksPerRow = CHUNK_FLAT_ELEMS / FP4_BLOCK.
constexpr int64_t MAX_MERGE_ELEMS = 2 * CHUNK_FLAT_ELEMS / FP4_BLOCK;
constexpr int64_t REDUCE_SLOT_ELEMS = BYTES_ALIGN / sizeof(float);
constexpr int32_t REDUCE_SLOT_BYTE_SHIFT = 5;

// The op compiles with --cce-auto-sync=off (op_host/CMakeLists.txt), so the
// compiler inserts no pipe synchronisation at all: every cross-pipe dependency
// is hand-written. Each HardEvent pair owns its own flag namespace, so the ids
// only have to be unique within a pair (see the file-header numbering).
// swish(x) = x / (1 + e^(-beta*x)); DeepSeek-V4 uses beta = 1.
constexpr float SWIGLU_BETA = 1.0f;

constexpr uint8_t STAGING_EVENT_ID = 0; // MTE2_V / V_MTE2 weight-staging ring
constexpr uint8_t ROW_EVENT_ID = 1;     // V_MTE3 / MTE3_V bf16-row ring
constexpr uint8_t OUT_EVENT_ID = 2;     // V_MTE3 one-shot before CopyOut
constexpr uint8_t X_EVENT_ID = 3;       // MTE2_V one-shot after the x load

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
        downAcc_ = xFloatBuf_.Get<float>(); // buffer is dead again by the down leg
        gateAcc_ = gateBuf_.Get<float>();
        upAcc_ = upBuf_.Get<float>();
        activated_ = activatedBuf_.Get<float>();
    }

    __aicore__ void Process()
    {
        CopyInX();
        ProjectLeg(LEG_W1, xFloat_, gateAcc_);    // unpack + gate GEMM
        ProjectLeg(LEG_W3, xFloat_, upAcc_);      // unpack + up GEMM
        ApplySwiGLU();
        ProjectLeg(LEG_W2, activated_, downAcc_); // unpack + down GEMM
        CopyOut();
    }

private:
    // Leg indexing: LEG_* name the wqGm_/wqScaleGm_/legOutGm_ slots. The down
    // leg (w2) swaps the row/reduction roles, hence its own index.
    static constexpr int32_t LEG_W1 = 0;
    static constexpr int32_t LEG_W2 = 1;
    static constexpr int32_t LEG_W3 = 2;

    // E2M1 float bit-pattern assembly constants (see DecodeCodes).
    static constexpr int32_t E2M1_HALF_BITS = 0x3F000000; // +0.5 (the subnormal code)
    static constexpr int32_t E2M1_ZERO_BITS = 0x00000000;

    __aicore__ void InitBuffers()
    {
        const int64_t maxCols = hidden_ > inter_ ? hidden_ : inter_;
        // xFloatBuf_ carries the widened x, then -- dead between the
        // projections and the down leg -- the SwiGLU clamp scratch (inter
        // floats) and finally the down accumulator (maxCols floats); sized
        // for the largest of those.
        const int64_t xFloatBytes = AlignUpBytes(maxCols * static_cast<int64_t>(sizeof(float)));

        pipe_->InitBuffer(xBitsBuf_, AlignUpBytes(hidden_ * 2));
        pipe_->InitBuffer(xFloatBuf_, xFloatBytes);
        pipe_->InitBuffer(gateBuf_, AlignUpBytes(inter_ * static_cast<int64_t>(sizeof(float))));
        pipe_->InitBuffer(upBuf_, AlignUpBytes(inter_ * static_cast<int64_t>(sizeof(float))));
        pipe_->InitBuffer(activatedBuf_, AlignUpBytes(inter_ * static_cast<int64_t>(sizeof(float))));
        pipe_->InitBuffer(interRowBuf_, AlignUpBytes(OUT_ROW_COUNT * inter_ * 2));

        // Per-chunk dequant pipeline. chunkRows * cols <= CHUNK_FLAT_ELEMS by
        // construction (ChunkRows), so these sizes cap the working set. The
        // code/bits assembly stages live in prodBuf_ (int32 views) and the
        // two half-width scratch buffers.
        pipe_->InitBuffer(wStagingBuf_, AlignUpBytes(CHUNK_FLAT_ELEMS / FP4_PER_BYTE));
        pipe_->InitBuffer(sStagingBuf_, AlignUpBytes(MAX_MERGE_ELEMS));
        // One int32 per DECODED element, not per packed byte. DecodeCodes runs
        // at count = chunkElems (up to CHUNK_FLAT_ELEMS); sizing these by
        // CHUNK_FLAT_ELEMS / FP4_PER_BYTE made them exactly half the length the
        // shifts write, which the CPU interpreter catches as "Failed to pass
        // ShiftRight calcount mode check" and silicon would not catch at all.
        pipe_->InitBuffer(stageABuf_, AlignUpBytes(CHUNK_FLAT_ELEMS * 4));
        pipe_->InitBuffer(stageBBuf_, AlignUpBytes(CHUNK_FLAT_ELEMS * 4));
        pipe_->InitBuffer(stageCBuf_, AlignUpBytes(CHUNK_FLAT_ELEMS * 4));
        pipe_->InitBuffer(prodBuf_, CHUNK_FLAT_ELEMS * 4);
        // x split into its even and odd columns once per column tile: the decode emits
        // all low nibbles then all high nibbles, so the low half pairs with the
        // even columns and the high half with the odd ones. See ProjectLeg.
        pipe_->InitBuffer(xEvenBuf_, MAX_CHUNK_COLS / FP4_PER_BYTE * 4);
        pipe_->InitBuffer(xOddBuf_, MAX_CHUNK_COLS / FP4_PER_BYTE * 4);
        pipe_->InitBuffer(partHiBuf_, MAX_MERGE_ELEMS * REDUCE_SLOT_ELEMS * 4);
        // Every reduction destination starts on a 32-byte boundary. The slots
        // are compacted with vector Gather after the reduction stage.
        pipe_->InitBuffer(partRawBuf_, MAX_MERGE_ELEMS * REDUCE_SLOT_ELEMS * 4);
        pipe_->InitBuffer(mergeBuf1_, MAX_MERGE_ELEMS * 4);
        pipe_->InitBuffer(mergeBuf2_, MAX_MERGE_ELEMS * 4);
        pipe_->InitBuffer(scalesF32Buf_, MAX_MERGE_ELEMS * 4);
        pipe_->InitBuffer(decMaskBuf_, AlignUpBytes(CHUNK_FLAT_ELEMS / FP4_PER_BYTE * 2));
        pipe_->InitBuffer(rowBf16Buf_, MAX_MERGE_ELEMS);
    }

    // MTE2 (GM->UB) finished -> the vector pipe may read the staging buffer.
    // Pairs with the V_MTE2 re-arm at the end of each chunk, which is the
    // write-after-read guard for the shared staging buffers.
    __aicore__ void WaitLoad()
    {
        SetFlag<HardEvent::MTE2_V>(STAGING_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(STAGING_EVENT_ID);
    }

    // x row: MTE2 load, then one widening vector cast -- bf16 -> fp32 is
    // exact, so no rounding enters here.
    __aicore__ void CopyInX()
    {
        LocalTensor<uint16_t> xBits = xBitsBuf_.Get<uint16_t>();
        DataCopy(xBits, xGmU16_, hidden_);
        SetFlag<HardEvent::MTE2_V>(X_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(X_EVENT_ID);
        Cast(xFloat_, xBits.ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, static_cast<uint32_t>(hidden_));
    }

    // One projection leg: out[r] = sum_c x[c] * E2M1(w[r, c]) * scale[r, c/32],
    // streamed in row chunks. Each chunk:
    //   MTE2 the packed bytes + scale bytes into the shared staging buffers;
    //   assemble the E2M1 float bit patterns from the nibble codes (vector
    //   int32 ops, natural column order -- see DecodeCodes);
    //   one row-width Mul per row folds x in (exact: short significands);
    //   one ReduceSum per 32-column block; scale + partial merge in fp32;
    //   vector Cast to bf16 and a padded store of the [chunkRows] row.
    __aicore__ void ProjectLeg(int32_t legIndex, const LocalTensor<float> &xFull, LocalTensor<float> &legAcc)
    {
        const int64_t fullCols = (legIndex == LEG_W2) ? inter_ : hidden_;
        const int64_t cols = fullCols < MAX_CHUNK_COLS ? fullCols : MAX_CHUNK_COLS;
        const int64_t rows = (legIndex == LEG_W2) ? hidden_ : inter_;
        const int64_t chunkRows = MIN_CHUNK_ROWS;
        // Each scale DMA row occupies a 32-byte UB block, including its pad.
        const int64_t pow2Blocks = BYTES_ALIGN;
        const int64_t chunkScales = chunkRows * pow2Blocks;

        LocalTensor<uint8_t> packed = wStagingBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> scales = sStagingBuf_.Get<uint8_t>();
        LocalTensor<float> products = prodBuf_.Get<float>();
        LocalTensor<int32_t> codes = prodBuf_.Get<int32_t>();  // bit-assembly stage, dies before the row Muls
        LocalTensor<int32_t> stageA = stageABuf_.Get<int32_t>();
        LocalTensor<int32_t> stageB = stageBBuf_.Get<int32_t>();
        LocalTensor<int32_t> stageC = stageCBuf_.Get<int32_t>();
        LocalTensor<uint8_t> decMask = decMaskBuf_.Get<uint8_t>();
        LocalTensor<float> partHi = partHiBuf_.Get<float>();
        LocalTensor<float> xEven = xEvenBuf_.Get<float>();
        LocalTensor<float> xOdd = xOddBuf_.Get<float>();

        // Split each column tile into even and odd input columns;
        // the decode's two halves pair with these. DeInterleave writes
        // halfCols elements to each destination.
        LocalTensor<float> partRaw = partRawBuf_.Get<float>();
        LocalTensor<float> merge1 = mergeBuf1_.Get<float>();
        LocalTensor<float> merge2 = mergeBuf2_.Get<float>();
        LocalTensor<uint16_t> rowBf16 = rowBf16Buf_.Get<uint16_t>();

        SetFlag<HardEvent::V_MTE2>(STAGING_EVENT_ID);  // prime: staging starts free
        SetFlag<HardEvent::MTE3_V>(ROW_EVENT_ID);      // prime: bf16 row starts free
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
                WaitFlag<HardEvent::V_MTE2>(STAGING_EVENT_ID);  // staging free for refill
                for (int64_t r = 0; r < chunkRows; ++r)
                {
                    DataCopy(packed[static_cast<uint32_t>(r * activeHalfCols)],
                             wqGm_[legIndex][((row0 + r) * fullCols + col0) / FP4_PER_BYTE], activeHalfCols);
                    DataCopyPad(scales[static_cast<uint32_t>(r * pow2Blocks)],
                                wqScaleGm_[legIndex][((row0 + r) * fullCols + col0) / FP4_BLOCK],
                                DataCopyParams{1, static_cast<uint16_t>(activeBlocks), 0, 0},
                                DataCopyPadParams{true, 0, static_cast<uint8_t>(pow2Blocks - activeBlocks), 0});
                }
                WaitLoad();  // MTE2 drain: the vector reads below must see the load

                DecodeCodes(packed, codes, stageA, stageB, stageC, decMask, static_cast<uint32_t>(activeBytes));

                // Fold x in row-wise. The decode emitted all low nibbles (the even
                // columns) in [0, chunkBytes) and all high nibbles (the odd ones)
                // in [chunkBytes, 2*chunkBytes), each row-major with halfCols per
                // row -- so each half multiplies against the matching half of x.
                // Both operands are exact short-significand floats; nothing rounds.
                // Row starts are multiples of halfCols floats, and halfCols >= 32,
                // so every slice is 32-byte aligned.
                for (int64_t r = 0; r < chunkRows; ++r)
                {
                    const uint32_t lowOff = static_cast<uint32_t>(r * activeHalfCols);
                    const uint32_t highOff = static_cast<uint32_t>(activeBytes + r * activeHalfCols);
                    Mul(products[lowOff], products[lowOff], xEven, static_cast<uint32_t>(activeHalfCols));
                    Mul(products[highOff], products[highOff], xOdd, static_cast<uint32_t>(activeHalfCols));
                }

                // Zero first so the pow2 pad slots the merge tree pairs with are
                // exact zero terms; the reduces then fill the real slots. The
                // ReduceRepeat writes one value per aligned 32-byte slot. This avoids
                // the level-2 ReduceSum wrapper's implicit V_S/scalar read.
                //
                // A block of 32 columns is 16 even plus 16 odd, so it takes two
                // reduces -- one per half -- summed at the end. That reassociates
                // the block sum relative to the golden's ascending walk, which is
                // fp32-exact to well under a bf16 ULP.
                Duplicate(partRaw, 0.0f, static_cast<uint32_t>(MAX_MERGE_ELEMS * REDUCE_SLOT_ELEMS));
                Duplicate(partHi, 0.0f, static_cast<uint32_t>(MAX_MERGE_ELEMS * REDUCE_SLOT_ELEMS));
                const int64_t halfBlock = FP4_BLOCK / FP4_PER_BYTE;
                for (int64_t r = 0; r < chunkRows; ++r)
                {
                    const LocalTensor<float> lowRow = products[static_cast<uint32_t>(r * activeHalfCols)];
                    const LocalTensor<float> highRow = products[static_cast<uint32_t>(activeBytes + r * activeHalfCols)];
                    for (int64_t b = 0; b < activeBlocks; ++b)
                    {
                        const uint32_t slot = static_cast<uint32_t>((r * pow2Blocks + b) * REDUCE_SLOT_ELEMS);
                        ReduceRepeat<ReduceType::SUM>(partRaw[slot], lowRow[static_cast<uint32_t>(b * halfBlock)],
                            static_cast<int32_t>(halfBlock), 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
                        ReduceRepeat<ReduceType::SUM>(partHi[slot], highRow[static_cast<uint32_t>(b * halfBlock)],
                            static_cast<int32_t>(halfBlock), 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
                    }
                }
                // Gather only the first float of each aligned 32-byte reduction slot.
                LocalTensor<int32_t> offsets = scalesF32Buf_.Get<int32_t>();
                CreateVecIndex(offsets, static_cast<int32_t>(0), static_cast<uint32_t>(MAX_MERGE_ELEMS));
                ShiftLeft(offsets, offsets, REDUCE_SLOT_BYTE_SHIFT, static_cast<uint32_t>(MAX_MERGE_ELEMS));
                Gather(merge1, partRaw, offsets.ReinterpretCast<uint32_t>(), 0,
                       static_cast<uint32_t>(MAX_MERGE_ELEMS));
                Gather(merge2, partHi, offsets.ReinterpretCast<uint32_t>(), 0,
                       static_cast<uint32_t>(MAX_MERGE_ELEMS));
                Add(partRaw, merge1, merge2, static_cast<uint32_t>(MAX_MERGE_ELEMS));

                // Apply the E8M0 scale to the compact partials. Post-reduce is
                // For finite, normal-range operands the scale is a power of two, so
                // scaling commutes with the reduction; the golden's per-element scaling and this
                // per-block scaling agree exactly.
                DecodeScales(scales, static_cast<uint32_t>(chunkScales));
                SetFlag<HardEvent::V_MTE2>(STAGING_EVENT_ID);  // re-arm after the last staging-buffer read
                const LocalTensor<float> scaleF32 = scalesF32Buf_.Get<float>();
                Mul(partRaw, partRaw, scaleF32, static_cast<uint32_t>(chunkScales));

                // Balanced merge of the per-block partials; the final level lands
                // in the leg accumulator's row slice.
                int64_t mergeLen = chunkRows * pow2Blocks;
                LocalTensor<float> cur = partRaw;
                while (mergeLen > 2 * chunkRows)
                {
                    SplitColumns(merge1, merge2, cur, static_cast<uint32_t>(mergeLen));
                    Add(cur, merge1, merge2, static_cast<uint32_t>(mergeLen / 2));
                    mergeLen /= 2;
                }
                SplitColumns(merge1, merge2, cur, static_cast<uint32_t>(mergeLen));
                Add(merge1, merge1, merge2, static_cast<uint32_t>(chunkRows));
                Add(accRow, accRow, merge1, static_cast<uint32_t>(chunkRows));
            }

            WaitFlag<HardEvent::MTE3_V>(ROW_EVENT_ID);  // previous row store retired
            Cast(rowBf16.ReinterpretCast<bfloat16_t>(), accRow, RoundMode::CAST_ROUND, static_cast<uint32_t>(chunkRows));
            SetFlag<HardEvent::V_MTE3>(ROW_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(ROW_EVENT_ID);  // MTE3 may read the row
            DataCopyPad(legOutGm_[legIndex][row0], rowBf16, DataCopyParams{1, static_cast<uint16_t>(chunkRows * 2), 0, 0});
            SetFlag<HardEvent::MTE3_V>(ROW_EVENT_ID);  // re-arm: row consumed by MTE3
        }
        WaitFlag<HardEvent::V_MTE2>(STAGING_EVENT_ID);  // drain: staging ring closed
        WaitFlag<HardEvent::MTE3_V>(ROW_EVENT_ID);      // drain: row ring closed
    }

    __aicore__ void BitAnd(const LocalTensor<int32_t>& dst, const LocalTensor<int32_t>& lhs,
                          const LocalTensor<int32_t>& rhs, uint32_t count)
    {
#if __CCE_AICORE__ == 220
        // CANN arch220 calcount And reinterprets i32 as i16 without doubling count.
        And(dst.ReinterpretCast<uint16_t>(), lhs.ReinterpretCast<uint16_t>(),
            rhs.ReinterpretCast<uint16_t>(), count * 2);
#else
        And(dst, lhs, rhs, count);
#endif
    }

    __aicore__ void BitOr(const LocalTensor<int32_t>& dst, const LocalTensor<int32_t>& lhs,
                         const LocalTensor<int32_t>& rhs, uint32_t count)
    {
#if __CCE_AICORE__ == 220
        Or(dst.ReinterpretCast<uint16_t>(), lhs.ReinterpretCast<uint16_t>(),
           rhs.ReinterpretCast<uint16_t>(), count * 2);
#else
        Or(dst, lhs, rhs, count);
#endif
    }
    __aicore__ void SplitColumns(const LocalTensor<float>& even, const LocalTensor<float>& odd,
                                const LocalTensor<float>& src, uint32_t count)
    {
#if __CCE_AICORE__ == 220
        // arch220 has no DeInterleave; stageA is dead at both split call sites.
        LocalTensor<int32_t> offsets = stageABuf_.Get<int32_t>();
        const uint32_t halfCount = count / FP4_PER_BYTE;
        CreateVecIndex(offsets, 0, halfCount);
        ShiftLeft(offsets, offsets, 3, halfCount);
        Gather(even, src, offsets.ReinterpretCast<uint32_t>(), 0, halfCount);
        Adds(offsets, offsets, static_cast<int32_t>(sizeof(float)), halfCount);
        Gather(odd, src, offsets.ReinterpretCast<uint32_t>(), 0, halfCount);
#else
        DeInterleave(even, odd, src, static_cast<int32_t>(count));
#endif
    }
    // Packed FP4 bytes -> E2M1 fp32 values, in natural column order, written
    // through the int32 view of the products buffer (bit patterns).
    //
    // Per nibble code c (bit3 sign, bits2..1 exponent, bit0 mantissa):
    //   c == 0      -> +0.0                       (0x00000000)
    //   c == 1      -> +0.5, the E2M1 subnormal   (0x3F000000)
    //   otherwise   -> (1 + m/2) * 2^(e-1), e = (c>>1)&3, m = c&1
    //                  = float bits ((e+126)<<23) | (m<<22)
    //   sign bit 3 OR-ed in last -- orthogonal to the magnitude bits, and it
    //   turns code 8 into -0.0 exactly as the table demands.
    // Every op below is PIPE_V; no barriers (in-order pipe).
    __aicore__ void DecodeCodes(const LocalTensor<uint8_t> &packed, const LocalTensor<int32_t> &codes,
                                const LocalTensor<int32_t> &stageA, const LocalTensor<int32_t> &stageB,
                                const LocalTensor<int32_t> &stageC, const LocalTensor<uint8_t> &decMask,
                                uint32_t byteCount)
    {
        const uint32_t count = byteCount * FP4_PER_BYTE;

        // Widen u8 -> u16 -> u32. Both steps are needed and both destination
        // types matter; this is where two separate bugs lived.
        //
        // The vconv support table (asc/impl/basic_api/dav_3510/
        // kernel_operator_vec_vconv_impl.h, `cast_none`) has Tuple<uint32_t,
        // uint16_t> but NOT Tuple<int32_t, uint16_t>, so casting into an int32
        // view aborts with "illegal type for cast none" -- hence the uint32
        // view here.
        //
        // The table also lists Tuple<uint32_t, uint8_t>, but that direct pair
        // is a NO-OP in practice: it neither aborts nor converts, and leaves
        // the destination untouched, which is the "an unsupported op compiles
        // to nothing" trap this file's header warns about. Measured -- every
        // decoded code came out zero. Only the u8 -> u16 step is trustworthy.
        //
        // stageB's uint16 alias is the staging area; it is not live yet.
#if __CCE_AICORE__ == 220
        // All byte values are exactly representable in both floating formats.
        Cast(stageB.template ReinterpretCast<half>(), packed, RoundMode::CAST_NONE, byteCount);
        Cast(stageC.template ReinterpretCast<float>(), stageB.template ReinterpretCast<half>(),
             RoundMode::CAST_NONE, byteCount);
        Cast(stageA, stageC.template ReinterpretCast<float>(), RoundMode::CAST_RINT, byteCount);
#else
        LocalTensor<uint16_t> wide = stageB.template ReinterpretCast<uint16_t>();
        Cast(wide, packed, RoundMode::CAST_NONE, byteCount);
        Cast(stageA.template ReinterpretCast<uint32_t>(), wide, RoundMode::CAST_NONE, byteCount);
#endif

        // Split the nibbles. Byte i holds element 2i in the low nibble and
        // 2i+1 in the high one, so the two halves are the even and the odd
        // columns. They stay as two contiguous halves rather than being
        // interleaved back: an element-granularity interleave is not
        // expressible here, because every repeat-parameter family strides by
        // 32-byte blocks, not by elements. ProjectLeg pairs each half with the
        // matching half of x instead.
        Duplicate(stageB, 0x0F, byteCount);
        BitAnd(codes, stageA, stageB, byteCount);                     // low nibbles
        ShiftRight(codes[byteCount], stageA, 4, byteCount);        // high nibbles (byte < 256)

        // Everything below keys on the MAGNITUDE code c & 7, not on c: codes
        // 8..15 are the negatives of 0..7 and must take the same magnitude
        // path. Keying on c sends code 9 down the normal branch and decodes it
        // to 0.75 instead of -0.5.
        Duplicate(stageC, 7, count);
        BitAnd(stageB, codes, stageC, count);                         // stageB = magnitude code

        // Normal magnitudes (>= 2): bits = ((e + 126) << 23) | (m << 22),
        // e = mag >> 1 (already in [0, 3], no mask needed), m = mag & 1.
        ShiftRight(stageA, stageB, 1, count);
        Adds(stageA, stageA, 126, count);
        ShiftLeft(stageA, stageA, 23, count);
        Duplicate(stageC, 1, count);
        BitAnd(stageC, stageB, stageC, count);                        // m -- must be masked:
        ShiftLeft(stageC, stageC, 22, count);                      // shifting the whole code
        BitOr(stageA, stageA, stageC, count);                         // corrupts the exponent

        // Magnitude 1 -> +0.5, magnitude 0 -> +0.0. Tensor-TENSOR selects: the
        // else-branch has to preserve stageA. With a scalar else-branch every
        // normal code is overwritten by that scalar, which erases the whole
        // block above and leaves the output all but zero.
        Duplicate(stageC, E2M1_HALF_BITS, count);
        Compares(decMask, stageB, 1, CMPMODE::EQ, count);
#if __CCE_AICORE__ == 220
        Select(stageA.template ReinterpretCast<float>(), decMask, stageC.template ReinterpretCast<float>(),
               stageA.template ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
#else
        Select(stageA, decMask, stageC, stageA, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
#endif
        Duplicate(stageC, E2M1_ZERO_BITS, count);
        Compares(decMask, stageB, 0, CMPMODE::EQ, count);
#if __CCE_AICORE__ == 220
        Select(stageA.template ReinterpretCast<float>(), decMask, stageC.template ReinterpretCast<float>(),
               stageA.template ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
#else
        Select(stageA, decMask, stageC, stageA, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
#endif

        // Sign: bit 3 of the original code into bit 31, which also turns code 8
        // into -0.0. The result lands in `codes` -- the int32 alias of the
        // products buffer -- so the float view downstream sees these patterns.
        Duplicate(stageC, 8, count);
        BitAnd(stageC, codes, stageC, count);
        ShiftLeft(stageC, stageC, 28, count);
        BitOr(codes, stageA, stageC, count);
    }

    // E8M0 byte vector -> fp32 power-of-two vector, in place via the i32 view:
    // bits = b << 23 decodes the healthy range [1, 254] exactly; bytes 0x00 and
    // 0xFF are fixed up by adding 0x00400000 under a predicate, which turns 0
    // into the subnormal 2^-127 (0x00400000) and +inf into the OCP MX quiet
    // NaN (0x7FC00000). The predicate is (b + 1) & 0xFF < 2 -- a single
    // compare that catches both specials via the u8 wraparound. The arithmetic
    // runs on int32, not uint32: the scalar-operand vector family has no
    // uint32 overloads, and every value here (max 255 << 23) stays inside the
    // signed range.
    __aicore__ void DecodeScales(const LocalTensor<uint8_t> &scales, uint32_t count)
    {
        // The merge ping-pong buffers are still idle at decode time, so they
        // host the widen staging and the wraparound predicate; scratch
        // alternates mask / delta duty.
        LocalTensor<int32_t> bits = scalesF32Buf_.Get<int32_t>();
        LocalTensor<int32_t> wrap = mergeBuf2_.Get<int32_t>();
        LocalTensor<int32_t> scratch = mergeBuf1_.Get<int32_t>();
        LocalTensor<uint8_t> special = decMaskBuf_.Get<uint8_t>();
#if __CCE_AICORE__ == 220
        Cast(mergeBuf2_.Get<half>(), scales, RoundMode::CAST_NONE, count);
        Cast(mergeBuf1_.Get<float>(), mergeBuf2_.Get<half>(), RoundMode::CAST_NONE, count);
        Cast(bits, mergeBuf1_.Get<float>(), RoundMode::CAST_RINT, count);
#else
        LocalTensor<uint32_t> bitsU32 = scalesF32Buf_.Get<uint32_t>();
        LocalTensor<uint16_t> wide = mergeBuf2_.Get<uint16_t>();
        Cast(wide, scales, RoundMode::CAST_NONE, count);
        Cast(bitsU32, wide, RoundMode::CAST_NONE, count);
#endif
        Adds(wrap, bits, 1, count);                      // b + 1 (1..256)
        Duplicate(scratch, 0xFF, count);
        BitAnd(wrap, wrap, scratch, count);                 // wrap to the u8 domain
#if __CCE_AICORE__ == 220
        // arch220 integer compares support EQ only; [0,255] converts exactly.
        Cast(scratch.ReinterpretCast<float>(), wrap, RoundMode::CAST_NONE, count);
        Compares(special, scratch.ReinterpretCast<float>(), 2.0f, CMPMODE::LT, count);
#else
        Compares(special, wrap, 2, CMPMODE::LT, count);
#endif
        ShiftLeft(bits, bits, 23, count);                // exact power-of-two bits

        // Build the fixup DELTA, then add it. The Select's destination is the
        // scratch, not `bits`: selecting straight into `bits` with a scalar
        // else-branch overwrites every healthy scale with that scalar, which
        // zeroed the whole scale vector and, through the Mul below, the whole
        // projection. (Here the scalar else-branch is right -- the delta for a
        // healthy byte is 0 -- it was only the destination that was wrong.)
        //
        //   b = 0x00: 0            + 0x00400000 = 2^-127, the fp32 subnormal
        //   b = 0xFF: 0x7F800000   + 0x00400000 = 0x7FC00000, the OCP MX NaN
        Duplicate(scratch, static_cast<int32_t>(SUBNORMAL_BITS), count);
#if __CCE_AICORE__ == 220
        Select(scratch.ReinterpretCast<float>(), special, scratch.ReinterpretCast<float>(), 0.0f,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
#else
        Select(scratch, special, scratch, static_cast<int32_t>(ZERO_BITS), SELMODE::VSEL_TENSOR_SCALAR_MODE,
               count);
#endif
        Add(bits, bits, scratch, count);
    }

    // activated = silu(clamp(gate)) * up, entirely on PIPE_V.
    //
    // The clamp is the DeepSeek-V4 architectural constant carried in tiling,
    // not an overflow guard: at +/-10 the exponential stays far inside fp32
    // range, so the inf path an unclamped kernel relies on -- 1/(1+inf)
    // evaluating to exactly 0 -- never arises.
    //
    // No PipeBarrier between the three calls: PIPE_V is in-order, and a
    // barrier between consecutive vector instructions only flushes the pipe.
    // No flags either: gate/up are vector-produced and activated is
    // vector-consumed, so there is no heterogeneous boundary in this stage.
    __aicore__ void ApplySwiGLU()
    {
        const uint32_t count = static_cast<uint32_t>(inter_);
        // The clamp goes to scratch rather than in place: gate_out is the raw
        // projection, and clamping it would change what that output means.
        // xFloatBuf_ is dead between the projections and the down leg.
        //
        // Sized explicitly to `count`: SwiGLU asserts that dst, src0 and src1
        // all report the SAME tensor size ("Input params.GetSize must be equal
        // with each other", swiglu_3510_impl.h). xFloatBuf_ is allocated for
        // max(hidden, inter) floats, so the default view would be too long
        // whenever hidden > inter.
        LocalTensor<float> clampedGate = xFloatBuf_.Get<float>(count);
        Mins(clampedGate, gateAcc_, swigluLimit_, count);
        Maxs(clampedGate, clampedGate, -swigluLimit_, count);
        // MEASURED, and it is the opposite of what the reference's clamping
        // pattern suggests: SwiGLU(dst, s0, s1, beta, n) computes
        // s0 * swish(s1) -- the swish is applied to the SECOND source.
        SwiGLU<float, false>(activated_, upAcc_, clampedGate, SWIGLU_BETA, count);
        // Preserve FP32 gate/up until activation, then round the down operand to BF16.
        LocalTensor<bfloat16_t> rounded = interRowBuf_.Get<bfloat16_t>(count);
        Cast(rounded, activated_, RoundMode::CAST_ROUND, count);
        Cast(activated_, rounded, RoundMode::CAST_NONE, count);
    }

    __aicore__ void CopyOut()
    {
        LocalTensor<uint16_t> interRows = interRowBuf_.Get<uint16_t>();
        // The three [1, inter] outputs stage side by side rather than reusing
        // one row: sharing a row would need an MTE3_V drain between every
        // cast and the next refill, and getting that wrong corrupts the
        // in-flight copy. inter_ is a multiple of 64, so every row offset is
        // 32-byte aligned. down_out was stored chunk-wise by its leg.
        LocalTensor<uint16_t> gateRow = interRows;
        LocalTensor<uint16_t> upRow = interRows[static_cast<uint32_t>(inter_)];
        LocalTensor<uint16_t> activatedRow = interRows[static_cast<uint32_t>(2 * inter_)];
        Cast(gateRow.ReinterpretCast<bfloat16_t>(), gateAcc_, RoundMode::CAST_ROUND, static_cast<uint32_t>(inter_));
        Cast(upRow.ReinterpretCast<bfloat16_t>(), upAcc_, RoundMode::CAST_ROUND, static_cast<uint32_t>(inter_));
        Cast(activatedRow.ReinterpretCast<bfloat16_t>(), activated_, RoundMode::CAST_ROUND,
             static_cast<uint32_t>(inter_));
        // One handshake covers all three rows (down_out left per-chunk).
        SetFlag<HardEvent::V_MTE3>(OUT_EVENT_ID);
        WaitFlag<HardEvent::V_MTE3>(OUT_EVENT_ID);
        DataCopy(legOutGm_[LEG_W1], gateRow, inter_);
        DataCopy(legOutGm_[LEG_W3], upRow, inter_);
        DataCopy(activatedGmU16_, activatedRow, inter_);
        SetFlag<HardEvent::MTE3_V>(OUT_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(OUT_EVENT_ID); // retire output reads before UB destruction
    }

    static constexpr int32_t SUBNORMAL_BITS = 0x00400000; // 2^-127 (E8M0 byte 0x00)
    static constexpr int32_t ZERO_BITS = 0;

    TPipe *pipe_ = nullptr;
    int64_t hidden_ = 0;
    int64_t inter_ = 0;
    float swigluLimit_ = 0.0f;

    GlobalTensor<uint16_t> xGmU16_;
    GlobalTensor<uint8_t> wqGm_[3];      // w1, w2, w3 packed FP4 bytes
    GlobalTensor<uint8_t> wqScaleGm_[3]; // w1/w2/w3 E8M0 block-scale bytes
    GlobalTensor<uint16_t> legOutGm_[3]; // w1 -> gate_out, w2 -> down_out, w3 -> up_out
    GlobalTensor<uint16_t> activatedGmU16_;

    LocalTensor<float> xFloat_;    // widened x row
    LocalTensor<float> downAcc_;   // down accumulator (xFloatBuf_ view)
    LocalTensor<float> gateAcc_;   // gate leg fp32 accumulator
    LocalTensor<float> upAcc_;     // up leg fp32 accumulator
    LocalTensor<float> activated_; // SwiGLU output, fp32, feeds the down leg

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
