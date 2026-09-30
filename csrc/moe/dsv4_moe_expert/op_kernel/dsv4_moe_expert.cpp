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
 * Numerics contract (mirrored bit-for-bit by tools/dsv4_moe_runtime/
 * kernel_bringup/gen_golden.py):
 *   - packed FP4 (FP4, 2 values per byte): byte i holds element 2i in the low
 *     nibble and element 2i+1 in the high nibble; E2M1 (E2M1) decode via a
 *     16-entry exact table;
 *   - E8M0 block scales (E8M0 block scale): one byte per 32 logical elements
 *     along the reduction dim, value 2^(byte-127), byte 0xFF decodes to NaN
 *     (OCP MX semantics); exact power-of-two assembly, no libm;
 *   - accumulation in fp32, one multiply-add per element, reduction dim in
 *     ascending order; outputs rounded to bf16 (round-to-nearest-even).
 *
 * Milestone scope (deliberate): single AIV core, whole-matrix resident in UB.
 * The tiling function rejects geometries whose packed weights do not fit the
 * single-load budget; chunked/multi-core tiling is the production follow-up.
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

// The op compiles with --cce-auto-sync=off (op_host/CMakeLists.txt), so the
// compiler inserts no pipe synchronisation at all: every cross-pipe dependency
// below is hand-written. Each HardEvent pair owns its own flag namespace, so
// the ids only have to be unique within a pair.
constexpr uint8_t SIGMOID_EVENT_ID = 7; // vector-Exp <-> scalar (S_V / V_S)
constexpr uint8_t LOAD_EVENT_ID = 0;    // GM->UB load <-> scalar (MTE2_S / S_MTE2)
constexpr uint8_t STORE_EVENT_ID = 1;   // scalar <-> UB->GM store (S_MTE3)

// E2M1 nibble decode: bit3=sign, bits2..1=exponent (bias 1), bit0=mantissa.
// Subnormal code 0x1 is +/-(1/2); codes 0x0/0x8 are +/-(1/4 * 2) = 0.
constexpr float E2M1_TABLE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

union FloatBits {
    uint32_t u;
    float f;
};

// E8M0 exponent byte -> exact fp32 power of two: 2^(b-127); 0xFF is NaN.
__aicore__ inline float E8M0ToScale(uint32_t exponentByte)
{
    FloatBits bits;
    if (exponentByte == 0xFFu) {
        bits.u = 0x7FC00000u; // quiet NaN, matching the OCP MX definition
        return bits.f;
    }
    int32_t exponent = static_cast<int32_t>(exponentByte) - 127; // [-127, 127]
    if (exponent == -127) {
        bits.u = 0x00400000u; // 2^-127: fp32 subnormal, exponent field alone cannot express it
    } else {
        bits.u = static_cast<uint32_t>(exponent + 127) << 23; // exact power-of-two bit pattern
    }
    return bits.f;
}

__aicore__ inline int64_t AlignUpBytes(int64_t bytes)
{
    return (bytes + BYTES_ALIGN - 1) / BYTES_ALIGN * BYTES_ALIGN;
}

// Scalar bf16 <-> fp32 without bfloat16_t casts (the bisheng backend rejects
// them): identical bit semantics to the golden reference's numpy helpers --
// widen by <<16 (exact), narrow with round-to-nearest-even, specials truncated.
__aicore__ inline float Bf16BitsToFloat(uint16_t bits)
{
    FloatBits conv;
    conv.u = static_cast<uint32_t>(bits) << 16;
    return conv.f;
}

__aicore__ inline uint16_t FloatToBf16Bits(float value)
{
    FloatBits conv;
    conv.f = value;
    const uint32_t bits = conv.u;
    if ((bits & 0x7F800000u) == 0x7F800000u) { // inf/NaN: no rounding
        return static_cast<uint16_t>(bits >> 16);
    }
    const uint32_t rounding = 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16);
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

        xGmU16_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(x));
        wqGm_[0].SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(w1));
        wqGm_[1].SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(w2));
        wqGm_[2].SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(w3));
        wqScaleGm_[0].SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(w1Scale));
        wqScaleGm_[1].SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(w2Scale));
        wqScaleGm_[2].SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(w3Scale));
        gateGmU16_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(gateOut));
        upGmU16_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(upOut));
        activatedGmU16_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(activatedOut));
        downGmU16_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(downOut));

        InitBuffers();
    }

    __aicore__ void Process()
    {
        CopyInX();
        ProjectGateUp(); // unpack + gate/up GEMM
        ApplySwiGLU();
        ProjectDown(); // unpack + down GEMM
        CopyOut();
    }

private:
    __aicore__ void InitBuffers()
    {
        const int64_t xBytes = AlignUpBytes(hidden_ * 2);       // bf16 bits via uint16 views
        // Gate/up weights share one staging buffer (loaded sequentially); the
        // down weight has the same packed byte count (inter*hidden/2 either way).
        const int64_t weightBytes = AlignUpBytes(inter_ * hidden_ / FP4_PER_BYTE);
        const int64_t weightScaleBytes = AlignUpBytes(inter_ * hidden_ / FP4_BLOCK);
        const int64_t downScaleBytes = AlignUpBytes(hidden_ * inter_ / FP4_BLOCK);
        // xFloatBuf_ is reused as the SwiGLU exp() scratch, which holds inter_
        // floats -- size it for the larger of the two so a geometry with
        // inter > hidden cannot walk off the end of the allocation.
        const int64_t scratchElems = hidden_ > inter_ ? hidden_ : inter_;
        const int64_t floatHiddenBytes = AlignUpBytes(scratchElems * static_cast<int64_t>(sizeof(float)));
        const int64_t floatInterBytes = AlignUpBytes(inter_ * static_cast<int64_t>(sizeof(float)));
        // gate/up/activated stage side by side so the three stores need one
        // scalar->MTE3 handshake instead of a read-after-write round trip each.
        const int64_t interRowBytes = AlignUpBytes(OUT_ROW_COUNT * inter_ * 2); // bf16 bits via uint16 views
        const int64_t hiddenRowBytes = AlignUpBytes(hidden_ * 2);

        pipe_->InitBuffer(xBuf_, xBytes);
        pipe_->InitBuffer(weightBuf_, weightBytes);
        pipe_->InitBuffer(weightScaleBuf_, weightScaleBytes);
        pipe_->InitBuffer(downScaleBuf_, downScaleBytes);
        pipe_->InitBuffer(xFloatBuf_, floatHiddenBytes);
        pipe_->InitBuffer(gateBuf_, floatInterBytes);
        pipe_->InitBuffer(upBuf_, floatInterBytes);
        pipe_->InitBuffer(activatedBuf_, floatInterBytes);
        pipe_->InitBuffer(interRowBuf_, interRowBytes);
        pipe_->InitBuffer(hiddenRowBuf_, hiddenRowBytes);
    }

    // MTE2 (GM->UB) finished -> the scalar pipe may read the staging buffer.
    __aicore__ void WaitLoad()
    {
        SetFlag<HardEvent::MTE2_S>(LOAD_EVENT_ID);
        WaitFlag<HardEvent::MTE2_S>(LOAD_EVENT_ID);
    }

    // Scalar pipe finished reading the staging buffer -> the next MTE2 load may
    // overwrite it. Without this the gate/up/down legs, which all share
    // weightBuf_, race write-after-read.
    __aicore__ void ReleaseLoadBuffers()
    {
        SetFlag<HardEvent::S_MTE2>(LOAD_EVENT_ID);
        WaitFlag<HardEvent::S_MTE2>(LOAD_EVENT_ID);
    }

    // Scalar pipe finished filling an output row -> MTE3 (UB->GM) may read it.
    __aicore__ void ReleaseStoreBuffers()
    {
        SetFlag<HardEvent::S_MTE3>(STORE_EVENT_ID);
        WaitFlag<HardEvent::S_MTE3>(STORE_EVENT_ID);
    }

    __aicore__ void CopyInX()
    {
        LocalTensor<uint16_t> xBits = xBuf_.Get<uint16_t>();
        DataCopy(xBits, xGmU16_, hidden_);
        WaitLoad(); // MTE2 drain: the scalar reads below must see the loaded row
        LocalTensor<float> xFloat = xFloatBuf_.Get<float>();
        for (int64_t c = 0; c < hidden_; ++c) {
            xFloat.SetValue(c, Bf16BitsToFloat(xBits.GetValue(c)));
        }
    }

    // outRow[r] = sum_c xF[c] * E2M1(nibble) * E8M0Scale(scaleRow r, block c/32)
    __aicore__ void Project(const LocalTensor<uint8_t> &packed, const LocalTensor<uint8_t> &scales,
                            const LocalTensor<float> &xF, LocalTensor<float> out, int64_t rows, int64_t cols)
    {
        const int64_t packedPerRow = cols / FP4_PER_BYTE;
        const int64_t scalesPerRow = cols / FP4_BLOCK;
        const int64_t bytesPerBlock = FP4_BLOCK / FP4_PER_BYTE;
        for (int64_t row = 0; row < rows; ++row) {
            const int64_t packedBase = row * packedPerRow;
            const int64_t scaleBase = row * scalesPerRow;
            float acc = 0.0f;
            // Block-major walk: the E8M0 byte is decoded once per 32 columns
            // instead of once per column, and each packed byte is fetched once
            // for both of its nibbles. Columns are still consumed in ascending
            // order and the accumulate expression is unchanged, so the fp32
            // summation sequence -- and the output bit pattern -- is identical
            // to the column-major form.
            for (int64_t block = 0; block < scalesPerRow; ++block) {
                const float scale = E8M0ToScale(scales.GetValue(scaleBase + block));
                const int64_t byteBase = packedBase + block * bytesPerBlock;
                const int64_t colBase = block * FP4_BLOCK;
                for (int64_t pair = 0; pair < bytesPerBlock; ++pair) {
                    const uint8_t byte = packed.GetValue(byteBase + pair);
                    const int64_t col = colBase + pair * FP4_PER_BYTE;
                    acc += xF.GetValue(col) * (E2M1_TABLE[byte & 0x0Fu] * scale);
                    acc += xF.GetValue(col + 1) * (E2M1_TABLE[byte >> 4] * scale);
                }
            }
            out.SetValue(row, acc);
        }
    }

    __aicore__ void ProjectGateUp()
    {
        LocalTensor<uint8_t> packed = weightBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> scales = weightScaleBuf_.Get<uint8_t>();
        LocalTensor<float> xFloat = xFloatBuf_.Get<float>();
        LocalTensor<float> gate = gateBuf_.Get<float>();
        LocalTensor<float> up = upBuf_.Get<float>();
        // gate leg (w1): inter rows, hidden reduction cols.
        DataCopy(packed, wqGm_[0], inter_ * hidden_ / FP4_PER_BYTE);
        DataCopy(scales, wqScaleGm_[0], inter_ * hidden_ / FP4_BLOCK);
        WaitLoad();
        Project(packed, scales, xFloat, gate, inter_, hidden_);
        // up leg (w3): reuses the staging buffers, so the refill must wait for
        // the gate leg's scalar reads to retire before MTE2 overwrites them.
        ReleaseLoadBuffers();
        DataCopy(packed, wqGm_[2], inter_ * hidden_ / FP4_PER_BYTE);
        DataCopy(scales, wqScaleGm_[2], inter_ * hidden_ / FP4_BLOCK);
        WaitLoad();
        Project(packed, scales, xFloat, up, inter_, hidden_);
    }

    __aicore__ void ApplySwiGLU()
    {
        LocalTensor<float> gate = gateBuf_.Get<float>();
        LocalTensor<float> up = upBuf_.Get<float>();
        LocalTensor<float> activated = activatedBuf_.Get<float>();
        // Device code has no scalar expf: exp(-g) runs through the vector Exp
        // API (mirroring chunk_kda_fwd's RunExp2 event pattern), then the
        // sigmoid combines scalarly. negExp stages in xFloatBuf_, which is sized
        // max(hidden, inter) floats and whose x contents are dead once both
        // projections have run.
        LocalTensor<float> negExp = xFloatBuf_.Get<float>();
        for (int64_t j = 0; j < inter_; ++j) {
            negExp.SetValue(j, -gate.GetValue(j));
        }
        SetFlag<HardEvent::S_V>(SIGMOID_EVENT_ID);
        WaitFlag<HardEvent::S_V>(SIGMOID_EVENT_ID);
        Exp(negExp, negExp, static_cast<uint32_t>(inter_));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(SIGMOID_EVENT_ID);
        WaitFlag<HardEvent::V_S>(SIGMOID_EVENT_ID);
        for (int64_t j = 0; j < inter_; ++j) {
            const float g = gate.GetValue(j);
            const float sigmoid = 1.0f / (1.0f + negExp.GetValue(j)); // silu(g) = g*sigmoid(g)
            activated.SetValue(j, g * sigmoid * up.GetValue(j));
        }
    }

    __aicore__ void ProjectDown()
    {
        LocalTensor<uint8_t> packed = weightBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> scales = downScaleBuf_.Get<uint8_t>();
        // down leg (w2): hidden rows, inter reduction cols. weightBuf_ still
        // carries the up leg's matrix, so release it before MTE2 refills it.
        ReleaseLoadBuffers();
        DataCopy(packed, wqGm_[1], hidden_ * inter_ / FP4_PER_BYTE);
        DataCopy(scales, wqScaleGm_[1], hidden_ * inter_ / FP4_BLOCK);
        WaitLoad();
        LocalTensor<float> activated = activatedBuf_.Get<float>();
        // The SwiGLU scratch is dead once ApplySwiGLU has run, so it doubles as
        // the fp32 down-projection accumulator (hidden_ floats, guaranteed to
        // fit by the InitBuffers sizing).
        LocalTensor<float> downFloat = xFloatBuf_.Get<float>();
        Project(packed, scales, activated, downFloat, hidden_, inter_);
        LocalTensor<uint16_t> downRow = hiddenRowBuf_.Get<uint16_t>();
        for (int64_t row = 0; row < hidden_; ++row) {
            downRow.SetValue(row, FloatToBf16Bits(downFloat.GetValue(row)));
        }
    }

    __aicore__ void CopyOut()
    {
        LocalTensor<float> gate = gateBuf_.Get<float>();
        LocalTensor<float> up = upBuf_.Get<float>();
        LocalTensor<float> activated = activatedBuf_.Get<float>();
        LocalTensor<uint16_t> interRows = interRowBuf_.Get<uint16_t>();
        LocalTensor<uint16_t> downRow = hiddenRowBuf_.Get<uint16_t>();
        // The three [1, inter] outputs stage side by side rather than reusing
        // one row: sharing a row would need an MTE3_S drain between every store
        // and the next scalar refill, and getting that wrong corrupts the
        // in-flight copy. inter_ is a multiple of 64, so every row offset is
        // 32-byte aligned.
        LocalTensor<uint16_t> gateRow = interRows;
        LocalTensor<uint16_t> upRow = interRows[static_cast<uint32_t>(inter_)];
        LocalTensor<uint16_t> activatedRow = interRows[static_cast<uint32_t>(2 * inter_)];

        for (int64_t j = 0; j < inter_; ++j) {
            gateRow.SetValue(j, FloatToBf16Bits(gate.GetValue(j)));
            upRow.SetValue(j, FloatToBf16Bits(up.GetValue(j)));
            activatedRow.SetValue(j, FloatToBf16Bits(activated.GetValue(j)));
        }
        // downRow was filled by ProjectDown; one handshake covers all four rows.
        ReleaseStoreBuffers();
        DataCopy(gateGmU16_, gateRow, inter_);
        DataCopy(upGmU16_, upRow, inter_);
        DataCopy(activatedGmU16_, activatedRow, inter_);
        DataCopy(downGmU16_, downRow, hidden_);
    }

    TPipe *pipe_ = nullptr;
    int64_t hidden_ = 0;
    int64_t inter_ = 0;

    GlobalTensor<uint16_t> xGmU16_;
    GlobalTensor<uint8_t> wqGm_[3];      // w1, w2, w3 packed FP4 bytes
    GlobalTensor<uint8_t> wqScaleGm_[3]; // w1/w2/w3 E8M0 block-scale bytes
    GlobalTensor<uint16_t> gateGmU16_;
    GlobalTensor<uint16_t> upGmU16_;
    GlobalTensor<uint16_t> activatedGmU16_;
    GlobalTensor<uint16_t> downGmU16_;

    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> weightBuf_;
    TBuf<TPosition::VECCALC> weightScaleBuf_;
    TBuf<TPosition::VECCALC> downScaleBuf_;
    TBuf<TPosition::VECCALC> xFloatBuf_;
    TBuf<TPosition::VECCALC> gateBuf_;
    TBuf<TPosition::VECCALC> upBuf_;
    TBuf<TPosition::VECCALC> activatedBuf_;
    TBuf<TPosition::VECCALC> interRowBuf_;
    TBuf<TPosition::VECCALC> hiddenRowBuf_;
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
