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
        pipe->InitBuffer(queryA1_, queryBytes);
        pipe->InitBuffer(probsA1_, probsBytes);
        for (uint32_t slot = 0; slot < kCubeSlots; ++slot) {
            pipe->InitBuffer(keyB1_[slot], tileBytes);
            pipe->InitBuffer(valueB1_[slot], tileBytes);
        }

        if ASCEND_IS_AIC {
            pipe->InitBuffer(operandA2_, queryBytes > probsBytes ? queryBytes : probsBytes);
            pipe->InitBuffer(operandB2_, tileBytes);
            pipe->InitBuffer(productCo1_, kCubeTileM * headSize_ * sizeof(float));
        }
    }

    __aicore__ inline AscendC::LocalTensor<OperandT> A1Query() { return queryA1_.template Get<OperandT>(); }
    __aicore__ inline AscendC::LocalTensor<OperandT> A1Probs() { return probsA1_.template Get<OperandT>(); }
    __aicore__ inline AscendC::LocalTensor<OperandT> B1K(uint32_t slot) { return keyB1_[slot].template Get<OperandT>(); }
    __aicore__ inline AscendC::LocalTensor<OperandT> B1V(uint32_t slot)
    {
        return valueB1_[slot].template Get<OperandT>();
    }
    __aicore__ inline AscendC::LocalTensor<OperandT> B1() { return B1K(0); }

    __aicore__ static constexpr uint32_t NzOffset(uint32_t row, uint32_t column, uint32_t rows)
    {
        return (column / kOperandC0) * rows * kOperandC0 + row * kOperandC0 + (column % kOperandC0);
    }

    // Query rows x key tile: M x K operand in A1, N x K operand in B1.
    template <bool DUAL_DST = false>
    __aicore__ inline void GemmScores(const AscendC::LocalTensor<float> &dstUb,
                                      const AscendC::LocalTensor<OperandT> &keyB1, uint32_t m, uint32_t k, uint32_t n)
    {
        activeA1_ = queryA1_.template Get<OperandT>();
        activeB1_ = keyB1;
        StageA2(m, k);
        StageB2FromNk(k, n);
        ComputeProductToUb<DUAL_DST>(dstUb, m, k, n);
    }

    // Probability rows x value tile: M x K operand in A1, K x N operand in B1.
    template <bool DUAL_DST = false>
    __aicore__ inline void GemmContext(const AscendC::LocalTensor<float> &dstUb,
                                       const AscendC::LocalTensor<OperandT> &valueB1, uint32_t m, uint32_t k, uint32_t n,
                                       uint32_t variant = kCubeLoadDefault)
    {
        activeA1_ = probsA1_.template Get<OperandT>();
        activeB1_ = valueB1;
        StageA2(m, k);
        StageB2FromKn(k, n, variant);
        ComputeProductToUb<DUAL_DST>(dstUb, m, k, n);
    }

private:
    static constexpr uint16_t kFractalRows = 16;
    static constexpr uint16_t kC0 = 32;
    // The K x N load issues its fractal rows two at a time.
    static constexpr uint16_t kKnBandRows = 2;
    static constexpr uint32_t kDualDstSubcores = 2;
    static constexpr uint8_t kDualDstSplitM = 1;

    __aicore__ inline void StageA2(uint32_t m, uint32_t k)
    {
        const AscendC::LocalTensor<OperandT> srcA1 = activeA1_;
        const AscendC::LocalTensor<OperandT> dstA2 = operandA2_.template Get<OperandT>();
        AscendC::LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = CeilDivU16(m, kFractalRows);
        params.kStep = CeilDivU16(k, kC0);
        params.srcStride = CeilDivU16(m, kFractalRows);
        params.dstStride = CeilDivU16(m, kFractalRows);
        params.ifTranspose = false;
        AscendC::LoadData(dstA2, srcA1, params);
    }

    __aicore__ inline void StageB2FromNk(uint32_t k, uint32_t n)
    {
        const AscendC::LocalTensor<OperandT> srcB1 = activeB1_;
        const AscendC::LocalTensor<OperandT> dstB2 = operandB2_.template Get<OperandT>();
        AscendC::LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = CeilDivU16(n, kFractalRows);
        params.kStep = CeilDivU16(k, kC0);
        params.srcStride = CeilDivU16(n, kFractalRows);
        params.dstStride = CeilDivU16(n, kFractalRows);
        params.ifTranspose = false;
        AscendC::LoadData(dstB2, srcB1, params);
    }

    __aicore__ inline void StageB2FromKn(uint32_t k, uint32_t n, uint32_t variant = kCubeLoadDefault)
    {
        const AscendC::LocalTensor<OperandT> srcB1 = activeB1_;
        const AscendC::LocalTensor<OperandT> dstB2 = operandB2_.template Get<OperandT>();
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
            AscendC::LoadData(dstB2, srcB1, params);
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
            AscendC::LoadData(dstB2[dstOffset], srcB1, params);
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
        const AscendC::LocalTensor<OperandT> a2 = operandA2_.template Get<OperandT>();
        const AscendC::LocalTensor<OperandT> b2 = operandB2_.template Get<OperandT>();
        const AscendC::LocalTensor<float> product = productCo1_.template Get<float>();

        SyncMte1ToMatrix();

        AscendC::Mmad(product, a2, b2,
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

    AscendC::LocalTensor<OperandT> activeA1_;
    AscendC::LocalTensor<OperandT> activeB1_;

    AscendC::TBuf<AscendC::TPosition::A1> queryA1_;
    AscendC::TBuf<AscendC::TPosition::A1> probsA1_;
    AscendC::TBuf<AscendC::TPosition::B1> keyB1_[kCubeSlots];
    AscendC::TBuf<AscendC::TPosition::B1> valueB1_[kCubeSlots];
    AscendC::TBuf<AscendC::TPosition::A2> operandA2_;
    AscendC::TBuf<AscendC::TPosition::B2> operandB2_;
    AscendC::TBuf<AscendC::TPosition::CO1> productCo1_;
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
        mm.template GemmScores<true>(scoresUb, mm.B1K(slot), rows, headSize, kCubeTileRows);
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(kFlagScoresReady);
    }

    __aicore__ static inline void ComputeContext(Mm &mm, const AscendC::LocalTensor<float> &contextUb,
                                                 const uint32_t slot, const uint32_t rows, const uint32_t headSize)
    {
        AscendC::CrossCoreWaitFlag(kFlagProbsReady);
        mm.template GemmContext<true>(contextUb, mm.B1V(slot), rows, kCubeTileRows, headSize);
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(kFlagContextReady);
    }
};

}
}

#endif
