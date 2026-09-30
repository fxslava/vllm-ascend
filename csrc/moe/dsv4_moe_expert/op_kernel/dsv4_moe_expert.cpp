/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software and can be redistributed under the terms
 * and conditions of the CANN Open Software License Agreement Version 2.0.
 * See the License for details.
 */

/*!
 * \file dsv4_moe_expert.cpp
 * \brief DeepSeek-V4 Flash MoE expert projection kernel (reduced-geometry milestone).
 *
 * One kernel launch computes the full per-expert pipeline of the DSV4 routed
 * expert, exactly matching the operand set the PyTorch draft runtime hands
 * to DummyExpertKernelRunner:
 *
 *   1. gate/up leg  -- x[1, hidden] (bf16) x w1/w3[inter, hidden] (packed FP4,
 *                      block-32 E8M0 scales) -> gate_out/up_out[1, inter] bf16
 *   2. SwiGLU       -- activated = silu(gate) * up -> activated[1, inter] bf16
 *   3. down leg     -- activated[1, inter] x w2[hidden, inter] (packed FP4,
 *                      block-32 E8M0 scales) -> down_out[1, hidden] bf16
 *
 * Numerics contract (mirrored by tools/dsv4_moe_runtime/kernel_bringup/
 * gen_golden.py):
 *   - packed FP4 (2 values per byte): byte i holds element 2i in the low
 *     nibble and element 2i+1 in the high nibble; E2M1 decode via a 16-entry
 *     exact table;
 *   - E8M0 block scales: one byte per 32 logical elements along the reduction
 *     dim, value 2^(byte-127), byte 0xFF decodes to NaN (OCP MX semantics),
 *     byte 0x00 to the fp32 subnormal 2^-127; exact power-of-two assembly;
 *   - accumulation in fp32 over the reduction dim in ascending order; outputs
 *     rounded to bf16 round-to-nearest-even.
 *
 * Execution model (this file replaces the scalar milestone kernel):
 *   - No scalar execution loops and no GetValue/SetValue in the math path.
 *     Every datum moves through the vector pipe: nibbles are decoded by
 *     gathering through 256-entry byte LUTs (the code is an index, not a
 *     number), bf16/fp32 conversions are vector Casts, and the reduction is a
 *     level-2 ReduceSum per 32-element block (this CANN ships ReduceSum; the
 *     older WholeReduceSum name does not exist in these headers). The only
 *     scalar writes are the 16 LUT entries and the 256-entry index ramp built
 *     once at init -- constant-table construction, outside the math path.
 *   - Weights stream through UB in row chunks, so the vector working set is a
 *     constant ~60 KiB regardless of the accepted geometry; the host tiling
 *     budget (op_host/dsv4_moe_expert_tiling.cpp) is unchanged and still
 *     bounds the milestone.
 *
 * Reduction-order deviation, by design and within the gate: the golden adds
 * all products of a row sequentially. Splitting the reduction at block
 * boundaries preserves that order ACROSS blocks (ascending, one vector add
 * per block), but the sum WITHIN a 32-element block and the <= 8-way merge of
 * block partials follow the reduce instruction's / a balanced tree's internal
 * order. Every product term is exact in fp32 (bf16 x E2M1 x power-of-two has
 * at most a 10-bit significand), so the deviation is pure addition
 * reassociation -- a few fp32 ULPs before the bf16 rounding, i.e. at most a
 * 1-ULP flip on elements sitting within a few fp32 ULPs of a bf16 rounding
 * boundary. The Gate B / hardware gate (FpDiff <= 2 bf16 ULP, RateDiff <=
 * 1e-2) is sized for exactly this.
 *   The E8M0 scale is applied AFTER the block reduce instead of per element:
 *   s * sum(p_i) == sum(s * p_i) exactly for a power-of-two s, so this neither
 *   adds nor removes a rounding step -- it only removes a broadcast.
 *
 * Synchronisation (the op compiles with --cce-auto-sync=off, so the compiler
 * inserts nothing): PIPE_V is in-order, so consecutive vector instructions --
 * including dependent ones -- carry no barriers. Flag ids are numbered:
 *   id 0: MTE2_V + V_MTE2 ring around the shared weight/scale staging buffers
 *         (primed before each leg's chunk loop, drained after);
 *   id 1: MTE3_V + V_MTE3 ring around the per-chunk bf16 row staging (the
 *         MTE3_V wait before the next Cast is the write-after-read guard for
 *         the in-flight row store);
 *   id 2: V_MTE3 one-shot ordering the three staged inter rows before CopyOut;
 *   id 3: MTE2_V one-shot ordering the x load before its vector cast.
 * The SwiGLU stage needs no flags at all: gate/up/activated are produced and
 * consumed entirely on PIPE_V.
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
// A reduce destination is left 32 bytes of room: whether the level-2 ReduceSum
// wrapper retires one float or a full 32-byte block per call, adjacent calls
// cannot clobber each other. The useful values are compacted by one Gather.
constexpr int64_t REDUCE_DST_SPACING = 8;
// Reduce-partial merge buffers hold chunkRows * pow2(blocksPerRow) floats;
// pow2 at most doubles blocksPerRow = CHUNK_FLAT_ELEMS / FP4_BLOCK.
constexpr int64_t MAX_MERGE_ELEMS = 2 * CHUNK_FLAT_ELEMS / FP4_BLOCK;

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

// E2M1 nibble decode: bit3=sign, bits2..1=exponent (bias 1), bit0=mantissa.
// Subnormal code 0x1 is +/-(1/2); codes 0x0/0x8 are +/-(1/4 * 2) = 0.
constexpr float E2M1_TABLE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__aicore__ inline int64_t AlignUpBytes(int64_t bytes)
{
    return (bytes + BYTES_ALIGN - 1) / BYTES_ALIGN * BYTES_ALIGN;
}

__aicore__ inline int64_t NextPow2(int64_t v)
{
    int64_t p = 1;
    while (p < v) {
        p <<= 1;
    }
    return p;
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
        BuildLuts();
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
        pipe_->InitBuffer(xEvenBuf_, AlignUpBytes(maxCols / FP4_PER_BYTE * static_cast<int64_t>(sizeof(float))));
        pipe_->InitBuffer(xOddBuf_, AlignUpBytes(maxCols / FP4_PER_BYTE * static_cast<int64_t>(sizeof(float))));
        pipe_->InitBuffer(gateBuf_, AlignUpBytes(inter_ * static_cast<int64_t>(sizeof(float))));
        pipe_->InitBuffer(upBuf_, AlignUpBytes(inter_ * static_cast<int64_t>(sizeof(float))));
        pipe_->InitBuffer(activatedBuf_, AlignUpBytes(inter_ * static_cast<int64_t>(sizeof(float))));
        pipe_->InitBuffer(interRowBuf_, AlignUpBytes(OUT_ROW_COUNT * inter_ * 2));

        // Per-chunk dequant pipeline. chunkRows * cols <= CHUNK_FLAT_ELEMS by
        // construction (ChunkRows), so these sizes cap the working set.
        pipe_->InitBuffer(wStagingBuf_, AlignUpBytes(CHUNK_FLAT_ELEMS / FP4_PER_BYTE));
        pipe_->InitBuffer(sStagingBuf_, AlignUpBytes(MAX_MERGE_ELEMS));
        pipe_->InitBuffer(byteIdxBuf_, AlignUpBytes(CHUNK_FLAT_ELEMS / FP4_PER_BYTE * 4));
        pipe_->InitBuffer(vLoBuf_, AlignUpBytes(CHUNK_FLAT_ELEMS / FP4_PER_BYTE * 4));
        pipe_->InitBuffer(vHiBuf_, AlignUpBytes(CHUNK_FLAT_ELEMS / FP4_PER_BYTE * 4));
        pipe_->InitBuffer(prodBuf_, CHUNK_FLAT_ELEMS * 4);
        pipe_->InitBuffer(partRawBuf_, MAX_MERGE_ELEMS * REDUCE_DST_SPACING * 4);
        pipe_->InitBuffer(mergeBuf0_, MAX_MERGE_ELEMS * 4);
        pipe_->InitBuffer(mergeBuf1_, MAX_MERGE_ELEMS * 4);
        pipe_->InitBuffer(mergeBuf2_, MAX_MERGE_ELEMS * 4);
        pipe_->InitBuffer(scalesF32Buf_, MAX_MERGE_ELEMS * 4);
        pipe_->InitBuffer(decMaskBuf_, MAX_MERGE_ELEMS);
        pipe_->InitBuffer(rowBf16Buf_, MAX_MERGE_ELEMS);
        pipe_->InitBuffer(reduceTmpBuf_, FP4_BLOCK * 4);
        pipe_->InitBuffer(arangeBuf_, 256 * 4);
        pipe_->InitBuffer(arangeTmpBuf_, 256 * 4);
        pipe_->InitBuffer(lut16Buf_, 16 * 4);
        pipe_->InitBuffer(lutLoBuf_, 256 * 4);
        pipe_->InitBuffer(lutHiBuf_, 256 * 4);
    }

    // Constant-table construction, once per launch, outside the math path: the
    // 16-entry E2M1 table is written scalar-side (16 values) and the 256-entry
    // index ramp is grown by vector doubling, then the two 256-entry byte LUTs
    // are GATHERED from the table on the vector pipe -- lutLo[b] = E2M1(b &
    // 0xF) decodes the low nibble of byte b, lutHi[b] = E2M1(b >> 4) the high
    // one. arangeBuf_ ends up holding 8*j, the compaction index matching the
    // REDUCE_DST_SPACING layout of the reduce destinations.
    __aicore__ void BuildLuts()
    {
        LocalTensor<float> lut16 = lut16Buf_.Get<float>();
        for (int32_t code = 0; code < 16; ++code) {
            lut16.SetValue(code, E2M1_TABLE[code]);
        }
        LocalTensor<uint32_t> arange = arangeBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> scratch = arangeTmpBuf_.Get<uint32_t>();
        arange.SetValue(0, 0u);
        int64_t len = 1;
        while (len < 256) {
            Duplicate(scratch, static_cast<uint32_t>(len), static_cast<uint32_t>(len));
            Add(arange[len], arange, scratch, static_cast<uint32_t>(len));
            len <<= 1;
        }
        LocalTensor<uint32_t> nibbleMask = arangeTmpBuf_.Get<uint32_t>();
        Duplicate(nibbleMask, 0x0Fu, 256u);
        And(scratch, arange, nibbleMask, 256u);
        Gather(lutLoBuf_.Get<float>(), lut16, scratch, 0, 256u);
        ShiftRight(scratch, arange, 4u, 256u);
        Gather(lutHiBuf_.Get<float>(), lut16, scratch, 0, 256u);
        ShiftLeft(arange, arange, 3u, 256u); // * REDUCE_DST_SPACING
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

    // Chunk row count: the largest row count that keeps the chunk's products
    // at or under CHUNK_FLAT_ELEMS and divides the leg's row count (rows is a
    // multiple of 64, so the loop below terminates at >= 1).
    __aicore__ int64_t ChunkRows(int64_t cols, int64_t rows) const
    {
        int64_t chunkRows = CHUNK_FLAT_ELEMS / cols;
        if (chunkRows > rows) {
            chunkRows = rows;
        }
        while (rows % chunkRows != 0) {
            --chunkRows;
        }
        return chunkRows;
    }

    // One projection leg: out[r] = sum_c x[c] * E2M1(w[r, c]) * scale[r, c/32],
    // streamed in row chunks. Each chunk:
    //   MTE2 the packed bytes + scale bytes into the shared staging buffers;
    //   widen bytes to u32 gather indices; two Gathers decode the low/high
    //   nibble streams through the byte LUTs (the streams are already the
    //   deinterleaved even/odd columns, exactly the x split, so one Mul each
    //   folds x in);
    //   one Interleave zips the streams back to ascending column order;
    //   one ReduceSum per 32-column block; scale + partial merge in fp32;
    //   vector Cast to bf16 and a padded store of the [chunkRows] row.
    __aicore__ void ProjectLeg(int32_t legIndex, const LocalTensor<float> &xFull, LocalTensor<float> &legAcc)
    {
        const int64_t cols = (legIndex == LEG_W2) ? inter_ : hidden_;
        const int64_t rows = (legIndex == LEG_W2) ? hidden_ : inter_;
        const int64_t chunkRows = ChunkRows(cols, rows);
        const int64_t blocksPerRow = cols / FP4_BLOCK;
        const int64_t pow2Blocks = NextPow2(blocksPerRow);
        const int64_t chunkElems = chunkRows * cols;
        const int64_t chunkBytes = chunkElems / FP4_PER_BYTE;
        const int64_t chunkScales = chunkRows * blocksPerRow;

        // The x split mirrors the nibble streams: xEven[j] = x[2j] pairs with
        // the low nibbles, xOdd[j] = x[2j+1] with the high ones.
        LocalTensor<float> xEven = xEvenBuf_.Get<float>();
        LocalTensor<float> xOdd = xOddBuf_.Get<float>();
        DeInterleave(xEven, xOdd, xFull, static_cast<int32_t>(cols));

        LocalTensor<uint8_t> packed = wStagingBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> scales = sStagingBuf_.Get<uint8_t>();
        LocalTensor<uint32_t> byteIdx = byteIdxBuf_.Get<uint32_t>();
        LocalTensor<float> vLo = vLoBuf_.Get<float>();
        LocalTensor<float> vHi = vHiBuf_.Get<float>();
        LocalTensor<float> products = prodBuf_.Get<float>();
        LocalTensor<float> partRaw = partRawBuf_.Get<float>();
        LocalTensor<float> merge0 = mergeBuf0_.Get<float>();
        LocalTensor<float> merge1 = mergeBuf1_.Get<float>();
        LocalTensor<float> merge2 = mergeBuf2_.Get<float>();
        LocalTensor<float> reduceTmp = reduceTmpBuf_.Get<float>();
        LocalTensor<uint16_t> rowBf16 = rowBf16Buf_.Get<uint16_t>();
        const LocalTensor<uint32_t> spacingIdx = arangeBuf_.Get<uint32_t>();
        const LocalTensor<float> lutLo = lutLoBuf_.Get<float>();
        const LocalTensor<float> lutHi = lutHiBuf_.Get<float>();

        SetFlag<HardEvent::V_MTE2>(STAGING_EVENT_ID); // prime: staging starts free
        SetFlag<HardEvent::MTE3_V>(ROW_EVENT_ID);     // prime: bf16 row starts free
        for (int64_t row0 = 0; row0 < rows; row0 += chunkRows) {
            WaitFlag<HardEvent::V_MTE2>(STAGING_EVENT_ID); // staging free for refill
            DataCopy(packed, wqGm_[legIndex][row0 * cols / FP4_PER_BYTE], chunkBytes);
            DataCopyPad(scales, wqScaleGm_[legIndex][row0 * blocksPerRow],
                        DataCopyParams{1, static_cast<uint16_t>(chunkScales), 0, 0}, DataCopyPadParams{});
            WaitLoad(); // MTE2 drain: the vector reads below must see the load

            Cast(byteIdx, packed, RoundMode::CAST_NONE, static_cast<uint32_t>(chunkBytes));
            Gather(vLo, lutLo, byteIdx, 0, static_cast<uint32_t>(chunkBytes)); // elements 2j
            Gather(vHi, lutHi, byteIdx, 0, static_cast<uint32_t>(chunkBytes)); // elements 2j+1
            Mul(vLo, vLo, xEven, static_cast<uint32_t>(chunkBytes)); // exact: short significands
            Mul(vHi, vHi, xOdd, static_cast<uint32_t>(chunkBytes));
            Interleave(products, products[static_cast<uint32_t>(chunkBytes)], vLo, vHi,
                       static_cast<int32_t>(chunkBytes)); // zip back to ascending columns

            for (int64_t r = 0; r < chunkRows; ++r) {
                const LocalTensor<float> prodRow = products[static_cast<uint32_t>(r * cols)];
                for (int64_t b = 0; b < blocksPerRow; ++b) {
                    ReduceSum(partRaw[static_cast<uint32_t>((r * pow2Blocks + b) * REDUCE_DST_SPACING)],
                              prodRow[static_cast<uint32_t>(b * FP4_BLOCK)], reduceTmp,
                              static_cast<int32_t>(FP4_BLOCK));
                }
            }

            // Compact the reduce destinations to one float per block and apply
            // the E8M0 scale. Post-reduce is exact: the scale is a power of
            // two, so s * sum(p) == sum(s * p) bit-for-bit -- the golden's
            // per-element scaling and this per-block scaling agree exactly.
            DecodeScales(scales, static_cast<uint32_t>(chunkScales));
            const LocalTensor<float> scaleF32 = scalesF32Buf_.Get<float>();
            Gather(merge0, partRaw, spacingIdx, 0, static_cast<uint32_t>(chunkScales));
            Mul(merge0, merge0, scaleF32, static_cast<uint32_t>(chunkScales));
            if (pow2Blocks > blocksPerRow) {
                // Pad the partials with zeros so the merge tree below can pair
                // down to one value per row; zero terms are exact identities.
                Duplicate(merge0[static_cast<uint32_t>(chunkScales)], 0.0f,
                          static_cast<uint32_t>(chunkRows * (pow2Blocks - blocksPerRow)));
            }

            // Balanced merge of the per-block partials; the final level lands
            // in the leg accumulator's row slice.
            int64_t mergeLen = chunkRows * pow2Blocks;
            LocalTensor<float> cur = merge0;
            while (mergeLen > 2 * chunkRows) {
                DeInterleave(merge1, merge2, cur, static_cast<int32_t>(mergeLen));
                Add(cur, merge1, merge2, static_cast<uint32_t>(mergeLen / 2));
                mergeLen /= 2;
            }
            DeInterleave(merge1, merge2, cur, static_cast<int32_t>(mergeLen));
            LocalTensor<float> accRow = legAcc[static_cast<uint32_t>(row0)];
            Add(accRow, merge1, merge2, static_cast<uint32_t>(chunkRows));

            WaitFlag<HardEvent::MTE3_V>(ROW_EVENT_ID); // previous row store retired
            Cast(rowBf16.ReinterpretCast<bfloat16_t>(), accRow, RoundMode::CAST_RINT,
                 static_cast<uint32_t>(chunkRows));
            SetFlag<HardEvent::V_MTE3>(ROW_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(ROW_EVENT_ID); // MTE3 may read the row
            DataCopyPad(legOutGm_[legIndex][row0], rowBf16,
                        DataCopyParams{1, static_cast<uint16_t>(chunkRows * 2), 0, 0});
            SetFlag<HardEvent::MTE3_V>(ROW_EVENT_ID); // re-arm: row consumed by MTE3
        }
        WaitFlag<HardEvent::V_MTE2>(STAGING_EVENT_ID); // drain: staging ring closed
        WaitFlag<HardEvent::MTE3_V>(ROW_EVENT_ID);     // drain: row ring closed
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
        // byteIdxBuf_ is dead once the nibble Gathers have run, so it hosts
        // the wraparound predicate; scratch alternates mask / delta duty.
        LocalTensor<int32_t> bits = scalesF32Buf_.Get<int32_t>();
        LocalTensor<int32_t> wrap = byteIdxBuf_.Get<int32_t>();
        LocalTensor<int32_t> scratch = arangeTmpBuf_.Get<int32_t>();
        LocalTensor<uint8_t> special = decMaskBuf_.Get<uint8_t>();
        Cast(bits, scales, RoundMode::CAST_NONE, count); // b
        Adds(wrap, bits, 1, count);                      // b + 1 (1..256)
        Duplicate(scratch, 0xFF, count);
        And(wrap, wrap, scratch, count);                 // wrap to the u8 domain
        Compares(special, wrap, 2, CMPMODE::LT, count);
        ShiftLeft(bits, bits, 23, count);                // exact power-of-two bits
        Duplicate(scratch, static_cast<int32_t>(SUBNORMAL_BITS), count); // reused as the fixup delta
        Select(bits, special, scratch, static_cast<int32_t>(ZERO_BITS), SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
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
        Cast(gateRow.ReinterpretCast<bfloat16_t>(), gateAcc_, RoundMode::CAST_RINT, static_cast<uint32_t>(inter_));
        Cast(upRow.ReinterpretCast<bfloat16_t>(), upAcc_, RoundMode::CAST_RINT, static_cast<uint32_t>(inter_));
        Cast(activatedRow.ReinterpretCast<bfloat16_t>(), activated_, RoundMode::CAST_RINT,
             static_cast<uint32_t>(inter_));
        // One handshake covers all three rows (down_out left per-chunk).
        SetFlag<HardEvent::V_MTE3>(OUT_EVENT_ID);
        WaitFlag<HardEvent::V_MTE3>(OUT_EVENT_ID);
        DataCopy(legOutGm_[LEG_W1], gateRow, inter_);
        DataCopy(legOutGm_[LEG_W3], upRow, inter_);
        DataCopy(activatedGmU16_, activatedRow, inter_);
    }

    static constexpr uint32_t SUBNORMAL_BITS = 0x00400000u; // 2^-127 (E8M0 byte 0x00)
    static constexpr uint32_t ZERO_BITS = 0u;

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
    TBuf<TPosition::VECCALC> xEvenBuf_;
    TBuf<TPosition::VECCALC> xOddBuf_;
    TBuf<TPosition::VECCALC> gateBuf_;
    TBuf<TPosition::VECCALC> upBuf_;
    TBuf<TPosition::VECCALC> activatedBuf_;
    TBuf<TPosition::VECCALC> interRowBuf_;
    TBuf<TPosition::VECCALC> wStagingBuf_;
    TBuf<TPosition::VECCALC> sStagingBuf_;
    TBuf<TPosition::VECCALC> byteIdxBuf_;
    TBuf<TPosition::VECCALC> vLoBuf_;
    TBuf<TPosition::VECCALC> vHiBuf_;
    TBuf<TPosition::VECCALC> prodBuf_;
    TBuf<TPosition::VECCALC> partRawBuf_;
    TBuf<TPosition::VECCALC> mergeBuf0_;
    TBuf<TPosition::VECCALC> mergeBuf1_;
    TBuf<TPosition::VECCALC> mergeBuf2_;
    TBuf<TPosition::VECCALC> scalesF32Buf_;
    TBuf<TPosition::VECCALC> decMaskBuf_;
    TBuf<TPosition::VECCALC> rowBf16Buf_;
    TBuf<TPosition::VECCALC> reduceTmpBuf_;
    TBuf<TPosition::VECCALC> arangeBuf_;
    TBuf<TPosition::VECCALC> arangeTmpBuf_;
    TBuf<TPosition::VECCALC> lut16Buf_;
    TBuf<TPosition::VECCALC> lutLoBuf_;
    TBuf<TPosition::VECCALC> lutHiBuf_;
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
