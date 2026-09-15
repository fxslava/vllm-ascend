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

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_CUBE_MM_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_CUBE_MM_H

#include "kernel_operator.h"
#include "turboquant_mode.h"

namespace vllm_ascend {
namespace turboquant {

constexpr AscendC::FixpipeConfig kFixpipeToUb = {AscendC::CO2Layout::ROW_MAJOR, true};

constexpr uint32_t kCubeTileM = 16;
constexpr uint32_t kCubeKStep = 64;

constexpr uint32_t kSlots = 2;

constexpr uint16_t kFlagOperandsReady = 0;
constexpr uint16_t kFlagProductReady = 1;

constexpr uint16_t kFlagSlotReady = 0;
constexpr uint16_t kFlagSlotFree = 2;
constexpr uint16_t kFlagScoresReady = 4;
constexpr uint16_t kFlagContextReady = 5;
constexpr uint16_t kFlagProbsReady = 6;

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

        const uint32_t qBytes = kCubeTileM * OperandElems(headSize_);
        const uint32_t pBytes = kCubeTileM * OperandElems(tileRows_);
        const uint32_t bBytes = tileRows_ * OperandElems(headSize_);
        pipe->InitBuffer(aQ1_, qBytes);
        pipe->InitBuffer(aP1_, pBytes);
        for (uint32_t slot = 0; slot < kSlots; ++slot) {
            pipe->InitBuffer(bK1_[slot], bBytes);
            pipe->InitBuffer(bV1_[slot], bBytes);
        }

        if ASCEND_IS_AIC {
            pipe->InitBuffer(a2_, qBytes > pBytes ? qBytes : pBytes);
            pipe->InitBuffer(b2_, bBytes);
            pipe->InitBuffer(co1_, kCubeTileM * headSize_ * sizeof(float));
        }
    }

    __aicore__ inline AscendC::LocalTensor<OperandT> A1Query() { return aQ1_.template Get<OperandT>(); }
    __aicore__ inline AscendC::LocalTensor<OperandT> A1Probs() { return aP1_.template Get<OperandT>(); }
    __aicore__ inline AscendC::LocalTensor<OperandT> B1K(uint32_t slot) { return bK1_[slot].template Get<OperandT>(); }
    __aicore__ inline AscendC::LocalTensor<OperandT> B1V(uint32_t slot) { return bV1_[slot].template Get<OperandT>(); }
    __aicore__ inline AscendC::LocalTensor<OperandT> B1() { return B1K(0); }

    __aicore__ static constexpr uint32_t NzOffset(uint32_t r, uint32_t c, uint32_t rows)
    {
        return (c / kOperandC0) * rows * kOperandC0 + r * kOperandC0 + (c % kOperandC0);
    }

    static constexpr uint32_t kOperandC0 = 32;

    __aicore__ inline void GemmScores(const AscendC::LocalTensor<float> &dstUb,
                                      const AscendC::LocalTensor<OperandT> &bL1, uint32_t m, uint32_t k, uint32_t n)
    {
        aActive_ = aQ1_.template Get<OperandT>();
        bActive_ = bL1;
        LoadA(m, k);
        LoadBFromNk(k, n);
        Compute(dstUb, m, k, n);
    }

    __aicore__ inline void GemmContext(const AscendC::LocalTensor<float> &dstUb,
                                       const AscendC::LocalTensor<OperandT> &bL1, uint32_t m, uint32_t k, uint32_t n,
                                       uint32_t variant = 0)
    {
        aActive_ = aP1_.template Get<OperandT>();
        bActive_ = bL1;
        LoadA(m, k);
        LoadBFromKn(k, n, variant);
        Compute(dstUb, m, k, n);
    }

private:
    static constexpr uint16_t kFractalRows = 16;
    static constexpr uint16_t kC0 = 32;
    static constexpr uint16_t kB8MStep = 2;

    __aicore__ static constexpr uint16_t CeilDivU16(uint32_t a, uint32_t b)
    {
        return static_cast<uint16_t>((a + b - 1) / b);
    }

    __aicore__ inline void LoadA(uint32_t m, uint32_t k)
    {
        AscendC::LocalTensor<OperandT> ta1 = aActive_;
        AscendC::LocalTensor<OperandT> ta2 = a2_.template Get<OperandT>();
        AscendC::LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = CeilDivU16(m, kFractalRows);
        p.kStep = CeilDivU16(k, kC0);
        p.srcStride = CeilDivU16(m, kFractalRows);
        p.dstStride = CeilDivU16(m, kFractalRows);
        p.ifTranspose = false;
        AscendC::LoadData(ta2, ta1, p);
    }

    __aicore__ inline void LoadBFromNk(uint32_t k, uint32_t n)
    {
        AscendC::LocalTensor<OperandT> tb1 = bActive_;
        AscendC::LocalTensor<OperandT> tb2 = b2_.template Get<OperandT>();
        AscendC::LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = CeilDivU16(n, kFractalRows);
        p.kStep = CeilDivU16(k, kC0);
        p.srcStride = CeilDivU16(n, kFractalRows);
        p.dstStride = CeilDivU16(n, kFractalRows);
        p.ifTranspose = false;
        AscendC::LoadData(tb2, tb1, p);
    }

    __aicore__ inline void LoadBFromKn(uint32_t k, uint32_t n, uint32_t variant = 0)
    {
        AscendC::LocalTensor<OperandT> tb1 = bActive_;
        AscendC::LocalTensor<OperandT> tb2 = b2_.template Get<OperandT>();
        AscendC::LoadData2DParamsV2 p;
        p.kStartPosition = 0;
        p.kStep = (variant == 5) ? CeilDivU16(n, kFractalRows) : CeilDivU16(n, kC0);
        p.srcStride = (variant == 6) ? CeilDivU16(k, kC0) : CeilDivU16(k, kFractalRows);
        p.dstStride = (variant == 8) ? CeilDivU16(n, kC0) : CeilDivU16(n, kFractalRows);
        p.ifTranspose = true;

        const uint16_t mSteps = CeilDivU16(k, kFractalRows);

        if (variant == 7) {
            p.mStartPosition = 0;
            p.mStep = mSteps;
            AscendC::LoadData(tb2, tb1, p);
            return;
        }

        uint32_t dstStrideElems =
            static_cast<uint32_t>(CeilDivU16(n, kFractalRows)) * kFractalRows * kC0;
        if (variant == 1) {
            dstStrideElems /= kC0;
        } else if (variant == 2) {
            dstStrideElems = static_cast<uint32_t>(CeilDivU16(n, kFractalRows)) * kC0;
        }

        const uint16_t loops = CeilDivU16(mSteps, kB8MStep);
        p.mStep = kB8MStep;
        uint32_t dstOffset = 0;
        AscendC::TPipe *pipe = GetTPipePtr();
        for (uint16_t i = 0; i < loops; ++i) {
            p.mStartPosition = static_cast<uint32_t>(kB8MStep) * i;
            AscendC::LoadData(tb2[dstOffset], tb1, p);
            if (variant != 4) {
                const event_t ev = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::MTE1_M));
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ev);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ev);
            }
            if (variant == 3) {
                AscendC::PipeBarrier<PIPE_MTE1>();
            }
            dstOffset += dstStrideElems;
        }
    }

    __aicore__ inline void Compute(const AscendC::LocalTensor<float> &dstUb, uint32_t m, uint32_t k, uint32_t n)
    {
        AscendC::LocalTensor<OperandT> ta2 = a2_.template Get<OperandT>();
        AscendC::LocalTensor<OperandT> tb2 = b2_.template Get<OperandT>();
        AscendC::LocalTensor<float> tco = co1_.template Get<float>();

        AscendC::TPipe *pipe = GetTPipePtr();
        const event_t mte1ToM = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::MTE1_M));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1ToM);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1ToM);

        AscendC::Mmad(tco, ta2, tb2,
                      AscendC::MmadParams(static_cast<uint16_t>(m), static_cast<uint16_t>(n),
                                          static_cast<uint16_t>(k), 0, false, true));

        const event_t mToFix = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::M_FIX));
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(mToFix);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(mToFix);

        AscendC::Fixpipe<float, float, kFixpipeToUb>(
            dstUb, tco,
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>(
                static_cast<uint16_t>(n), static_cast<uint16_t>(m),
                static_cast<uint16_t>(CeilDivU16(m, kFractalRows) * kFractalRows), n));
        AscendC::PipeBarrier<PIPE_FIX>();
    }

    AscendC::LocalTensor<OperandT> aActive_;
    AscendC::LocalTensor<OperandT> bActive_;

    AscendC::TBuf<AscendC::TPosition::A1> aQ1_;
    AscendC::TBuf<AscendC::TPosition::A1> aP1_;
    AscendC::TBuf<AscendC::TPosition::B1> bK1_[kSlots];
    AscendC::TBuf<AscendC::TPosition::B1> bV1_[kSlots];
    AscendC::TBuf<AscendC::TPosition::A2> a2_;
    AscendC::TBuf<AscendC::TPosition::B2> b2_;
    AscendC::TBuf<AscendC::TPosition::CO1> co1_;
    uint32_t headSize_ = 0;
    uint32_t tileRows_ = 0;
};

}
}

#endif
