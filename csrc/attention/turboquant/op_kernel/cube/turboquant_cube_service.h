/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_CUBE_SERVICE_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_CUBE_SERVICE_H

#include "../common/turboquant_common.h"
#include "../common/turboquant_mode.h"

namespace vllm_ascend {
namespace turboquant {

template <TurboQuantMode MODE>
struct TurboQuantOperandType;

template <>
struct TurboQuantOperandType<TurboQuantMode::KV3_FP4> {
    using Type = fp4x2_e2m1_t;
};
template <>
struct TurboQuantOperandType<TurboQuantMode::KV4_FP8> {
    using Type = fp8_e4m3fn_t;
};
template <>
struct TurboQuantOperandType<TurboQuantMode::KV5_FP8> {
    using Type = fp8_e4m3fn_t;
};

// How the right-hand operand of a Cube GEMM is laid out in L1, which is what decides whether the L0B
// load transposes it. The two are the decode's two GEMMs: the score GEMM's key tile is already N x K
// (LoadData with ifTranspose false), the context GEMM's value tile is K x N and the load transposes it.
// The values are the wire encoding the contract probe's kernel argument carries, so they are fixed.
enum class GemmLayout : uint32_t {
    Transposed = 0,
    Normal = 1,
};

// The K x N context load variants the Cube contract probe (turboquant_cube_gemm_probe_impl) selects by
// number. The decode always issues kCubeLoadDefault.
enum CubeLoadVariant : uint32_t {
    kCubeLoadDefault = 0,
    kCubeLoadBandStrideC0Elems = 1,
    kCubeLoadBandStrideRowElems = 2,
    kCubeLoadMte1BarrierPerBand = 3,
    kCubeLoadNoMte1EventPerBand = 4,
    kCubeLoadKStepByFractalRows = 5,
    kCubeLoadSrcStrideByC0 = 6,
    kCubeLoadSingleLoad = 7,
    kCubeLoadDstStrideByC0 = 8,
    kCubeLoadDoubleOperandsWait = 10,
};

template <TurboQuantMode MODE>
class TurboQuantCubeMm {
public:
    using OperandT = typename TurboQuantOperandType<MODE>::Type;
    static constexpr bool kIsFp4 = TurboQuantModeTraits<MODE>::kOperand == TurboQuantOperand::kFp4E2m1;

    __aicore__ static constexpr uint32_t OperandElems(uint32_t n) { return kIsFp4 ? n / 2 : n; }

    __aicore__ inline void Init(AscendC::TPipe *pipe, uint32_t headSize, uint32_t tileRows)
    {
        headSize_ = headSize;
        tileRows_ = tileRows;

        const uint32_t queryBytes = kCubeTileM * OperandElems(headSize_);
        const uint32_t probsBytes = kCubeTileM * OperandElems(tileRows_);
        const uint32_t tileBytes = tileRows_ * OperandElems(headSize_);
        pipe->InitBuffer(queryL1Buf_, queryBytes);
        pipe->InitBuffer(probsL1Buf_, probsBytes);
        for (uint32_t slot = 0; slot < kCubeSlots; ++slot) {
            pipe->InitBuffer(keyL1Buf_[slot], tileBytes);
            pipe->InitBuffer(valueL1Buf_[slot], tileBytes);
        }

        if ASCEND_IS_AIC {
            pipe->InitBuffer(operandL0aBuf_, queryBytes > probsBytes ? queryBytes : probsBytes);
            pipe->InitBuffer(operandL0bBuf_, tileBytes);
            pipe->InitBuffer(productL0cBuf_, kCubeTileM * headSize_ * sizeof(float));
        }
    }

    __aicore__ inline AscendC::LocalTensor<OperandT> L1Query() { return queryL1Buf_.template Get<OperandT>(); }
    __aicore__ inline AscendC::LocalTensor<OperandT> L1Probs() { return probsL1Buf_.template Get<OperandT>(); }
    __aicore__ inline AscendC::LocalTensor<OperandT> L1Key(uint32_t slot) { return keyL1Buf_[slot].template Get<OperandT>(); }
    __aicore__ inline AscendC::LocalTensor<OperandT> L1Value(uint32_t slot)
    {
        return valueL1Buf_[slot].template Get<OperandT>();
    }
    __aicore__ inline AscendC::LocalTensor<OperandT> L1Weight() { return L1Key(0); }

    __aicore__ static constexpr uint32_t NzOffset(uint32_t row, uint32_t column, uint32_t rows)
    {
        return (column / kOperandC0) * rows * kOperandC0 + row * kOperandC0 + (column % kOperandC0);
    }

    // Query rows x key tile: the M x K left operand in L1, the N x K right operand in L1.
    template <bool DUAL_DST = false>
    __aicore__ inline void GemmScores(const AscendC::LocalTensor<float> &dstUb,
                                      const AscendC::LocalTensor<OperandT> &keyL1, uint32_t m, uint32_t k, uint32_t n)
    {
        activeLeftL1_ = queryL1Buf_.template Get<OperandT>();
        activeRightL1_ = keyL1;
        StageLeftToL0a(m, k);
        StageRightToL0bFromNk(k, n);
        ComputeProductToUb<DUAL_DST>(dstUb, m, k, n);
    }

    // Probability rows x value tile: the M x K left operand in L1, the K x N right operand in L1.
    template <bool DUAL_DST = false>
    __aicore__ inline void GemmContext(const AscendC::LocalTensor<float> &dstUb,
                                       const AscendC::LocalTensor<OperandT> &valueL1, uint32_t m, uint32_t k,
                                       uint32_t n, uint32_t variant = kCubeLoadDefault)
    {
        activeLeftL1_ = probsL1Buf_.template Get<OperandT>();
        activeRightL1_ = valueL1;
        StageLeftToL0a(m, k);
        StageRightToL0bFromKn(k, n, variant);
        ComputeProductToUb<DUAL_DST>(dstUb, m, k, n);
    }

private:
    static constexpr uint16_t kFractalRows = 16;
    static constexpr uint16_t kC0 = 32;
    // The K x N load issues its fractal rows two at a time.
    static constexpr uint16_t kKnBandRows = 2;
    static constexpr uint32_t kDualDstSubcores = 2;
    static constexpr uint8_t kDualDstSplitM = 1;

    __aicore__ inline void StageLeftToL0a(uint32_t m, uint32_t k)
    {
        const AscendC::LocalTensor<OperandT> srcL1 = activeLeftL1_;
        const AscendC::LocalTensor<OperandT> dstL0a = operandL0aBuf_.template Get<OperandT>();
        AscendC::LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = CeilDivU16(m, kFractalRows);
        params.kStep = CeilDivU16(k, kC0);
        params.srcStride = CeilDivU16(m, kFractalRows);
        params.dstStride = CeilDivU16(m, kFractalRows);
        params.ifTranspose = false;
        AscendC::LoadData(dstL0a, srcL1, params);
    }

    __aicore__ inline void StageRightToL0bFromNk(uint32_t k, uint32_t n)
    {
        const AscendC::LocalTensor<OperandT> srcL1 = activeRightL1_;
        const AscendC::LocalTensor<OperandT> dstL0b = operandL0bBuf_.template Get<OperandT>();
        AscendC::LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = CeilDivU16(n, kFractalRows);
        params.kStep = CeilDivU16(k, kC0);
        params.srcStride = CeilDivU16(n, kFractalRows);
        params.dstStride = CeilDivU16(n, kFractalRows);
        params.ifTranspose = false;
        AscendC::LoadData(dstL0b, srcL1, params);
    }

    __aicore__ inline void StageRightToL0bFromKn(uint32_t k, uint32_t n, uint32_t variant = kCubeLoadDefault)
    {
        const AscendC::LocalTensor<OperandT> srcL1 = activeRightL1_;
        const AscendC::LocalTensor<OperandT> dstL0b = operandL0bBuf_.template Get<OperandT>();
        AscendC::LoadData2DParamsV2 params;
        params.kStartPosition = 0;
        params.kStep = (variant == kCubeLoadKStepByFractalRows) ? CeilDivU16(n, kFractalRows) : CeilDivU16(n, kC0);
        params.srcStride = (variant == kCubeLoadSrcStrideByC0) ? CeilDivU16(k, kC0) : CeilDivU16(k, kFractalRows);
        params.dstStride = (variant == kCubeLoadDstStrideByC0) ? CeilDivU16(n, kC0) : CeilDivU16(n, kFractalRows);
        params.ifTranspose = true;

        const uint16_t mSteps = CeilDivU16(k, kFractalRows);

        if (variant == kCubeLoadSingleLoad) {
            params.mStartPosition = 0;
            params.mStep = mSteps;
            AscendC::LoadData(dstL0b, srcL1, params);
            return;
        }

        uint32_t bandStrideElems = static_cast<uint32_t>(CeilDivU16(n, kFractalRows)) * kFractalRows * kC0;
        if (variant == kCubeLoadBandStrideC0Elems) {
            bandStrideElems /= kC0;
        } else if (variant == kCubeLoadBandStrideRowElems) {
            bandStrideElems = static_cast<uint32_t>(CeilDivU16(n, kFractalRows)) * kC0;
        }

        const uint16_t bands = CeilDivU16(mSteps, kKnBandRows);
        params.mStep = kKnBandRows;
        uint32_t dstOffset = 0;
        for (uint16_t band = 0; band < bands; ++band) {
            params.mStartPosition = static_cast<uint32_t>(kKnBandRows) * band;
            AscendC::LoadData(dstL0b[dstOffset], srcL1, params);
            if (variant != kCubeLoadNoMte1EventPerBand) {
                SyncMte1ToMatrix();
            }
            if (variant == kCubeLoadMte1BarrierPerBand) {
                // Probe-only: the contract variant that serialises MTE1 itself instead of handing off.
                AscendC::PipeBarrier<PIPE_MTE1>();
            }
            dstOffset += bandStrideElems;
        }
    }

    template <bool DUAL_DST>
    __aicore__ inline void ComputeProductToUb(const AscendC::LocalTensor<float> &dstUb, uint32_t m, uint32_t k,
                                              uint32_t n)
    {
        const AscendC::LocalTensor<OperandT> leftL0a = operandL0aBuf_.template Get<OperandT>();
        const AscendC::LocalTensor<OperandT> rightL0b = operandL0bBuf_.template Get<OperandT>();
        const AscendC::LocalTensor<float> product = productL0cBuf_.template Get<float>();

        SyncMte1ToMatrix();

        AscendC::Mmad(product, leftL0a, rightL0b,
                      AscendC::MmadParams(static_cast<uint16_t>(m), static_cast<uint16_t>(n),
                                          static_cast<uint16_t>(k), 0, false, true));

        SyncMatrixToFixpipe();

        if constexpr (DUAL_DST) {
            const uint32_t evenM = CeilDivU16(m, kDualDstSubcores) * kDualDstSubcores;
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixParams(
                static_cast<uint16_t>(n), static_cast<uint16_t>(evenM),
                static_cast<uint16_t>(CeilDivU16(evenM, kFractalRows) * kFractalRows), n);
            fixParams.dualDstCtl = kDualDstSplitM;
            fixParams.subBlockId = false;
            AscendC::Fixpipe<float, float, kFixpipeToUb>(dstUb, product, fixParams);
        } else {
            AscendC::Fixpipe<float, float, kFixpipeToUb>(
                dstUb, product,
                AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>(
                    static_cast<uint16_t>(n), static_cast<uint16_t>(m),
                    static_cast<uint16_t>(CeilDivU16(m, kFractalRows) * kFractalRows), n));
        }
        // Drains the Fixpipe before the caller's ready flag releases the vector unit onto its UB rows.
        AscendC::PipeBarrier<PIPE_FIX>();
    }

    AscendC::LocalTensor<OperandT> activeLeftL1_;
    AscendC::LocalTensor<OperandT> activeRightL1_;

    AscendC::TBuf<AscendC::TPosition::A1> queryL1Buf_;
    AscendC::TBuf<AscendC::TPosition::A1> probsL1Buf_;
    AscendC::TBuf<AscendC::TPosition::B1> keyL1Buf_[kCubeSlots];
    AscendC::TBuf<AscendC::TPosition::B1> valueL1Buf_[kCubeSlots];
    AscendC::TBuf<AscendC::TPosition::A2> operandL0aBuf_;
    AscendC::TBuf<AscendC::TPosition::B2> operandL0bBuf_;
    AscendC::TBuf<AscendC::TPosition::CO1> productL0cBuf_;
    uint32_t headSize_ = 0;
    uint32_t tileRows_ = 0;
};

// The Cube half of the fused decode. For every tile the AIV subcores stage, the AIC runs the score
// GEMM and then the context GEMM, each Fixpiped into BOTH subcores' UB (dualDstCtl = 1, M split in
// half), and hands off with the cross-core flags the vector service waits on. Rows are the task's
// head count; the Fixpipe pads M to even and each subcore reads its half at UB offset 0.
template <TurboQuantMode MODE>
struct TurboQuantCubeDecodeService {
    using Mm = TurboQuantCubeMm<MODE>;

    __aicore__ static inline void RunTiles(Mm &mm, const AscendC::LocalTensor<float> &scoresUb,
                                           const AscendC::LocalTensor<float> &contextUb, const uint32_t numTiles,
                                           const uint32_t rows, const uint32_t headSize)
    {
        for (uint32_t tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
            const uint32_t slot = tileIdx % kCubeSlots;
            ComputeScores(mm, scoresUb, slot, rows, headSize);
            ComputeContext(mm, contextUb, slot, rows, headSize);
            if (tileIdx + kCubeSlots < numTiles) {
                AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE1>(static_cast<uint16_t>(kFlagSlotFree + slot));
            }
        }
    }

private:
    __aicore__ static inline void ComputeScores(Mm &mm, const AscendC::LocalTensor<float> &scoresUb,
                                                const uint32_t slot, const uint32_t rows, const uint32_t headSize)
    {
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagSlotReady + slot));
        mm.template GemmScores<true>(scoresUb, mm.L1Key(slot), rows, headSize, kCubeTileRows);
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(kFlagScoresReady);
    }

    __aicore__ static inline void ComputeContext(Mm &mm, const AscendC::LocalTensor<float> &contextUb,
                                                 const uint32_t slot, const uint32_t rows, const uint32_t headSize)
    {
        AscendC::CrossCoreWaitFlag(kFlagProbsReady);
        mm.template GemmContext<true>(contextUb, mm.L1Value(slot), rows, kCubeTileRows, headSize);
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(kFlagContextReady);
    }
};

}
}

#endif
