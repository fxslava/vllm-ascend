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

#include "kernel_operator.h"

namespace {

constexpr uint32_t kTile = 16;

constexpr uint32_t kHalfC0 = 16;

constexpr uint32_t kResidualFirstStride = kTile;

constexpr uint32_t kEarlyStages = 3;
constexpr uint32_t kFp32PerBlock = 8;
constexpr uint32_t kFp32PerRepeat = 64;

constexpr uint32_t kGatherSrcBase = 0;

constexpr uint32_t kMaxRepeat = 255;

constexpr uint16_t kFlagOperandsReady = 0;
constexpr uint16_t kFlagOperandsFree = 2;
constexpr uint16_t kFlagProductReady = 4;
constexpr uint16_t kFlagProductFree = 6;

constexpr uint32_t kSlots = 2;

constexpr uint32_t kVariantHiLo = 0x1u;
constexpr uint32_t kVariantDualDst = 0x2u;
constexpr uint32_t kVariantLockstep = 0x4u;

constexpr AscendC::FixpipeConfig kFixpipeToUb = {AscendC::CO2Layout::ROW_MAJOR, true};

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
__aicore__ inline uint32_t AlignUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }

__aicore__ inline uint16_t CeilDivU16(uint32_t a, uint32_t b)
{
    return static_cast<uint16_t>((a + b - 1) / b);
}

__aicore__ inline void SyncVectorToMte3()
{
    const event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev);
}

template <AscendC::HardEvent EVENT>
__aicore__ inline void SyncEvent()
{
    const event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVENT));
    AscendC::SetFlag<EVENT>(ev);
    AscendC::WaitFlag<EVENT>(ev);
}

__aicore__ inline void BlockButterfly(const AscendC::LocalTensor<float> &dst,
                                      const AscendC::LocalTensor<float> &src, uint32_t stride, uint32_t count)
{
    const uint32_t groups = count / (2 * stride);
    if (stride > kFp32PerRepeat) {
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t base = g * 2 * stride;
            AscendC::Add(dst[base], src[base], src[base + stride], stride);
            AscendC::Sub(dst[base + stride], src[base], src[base + stride], stride);
        }
        return;
    }
    const uint8_t rep = static_cast<uint8_t>(2 * stride / kFp32PerBlock);
    const AscendC::BinaryRepeatParams params{1, 1, 1, rep, rep, rep};
    const uint64_t mask = static_cast<uint64_t>(stride);
    for (uint32_t done = 0; done < groups; done += kMaxRepeat) {
        const uint32_t batch = (groups - done) < kMaxRepeat ? (groups - done) : kMaxRepeat;
        const uint32_t base = done * 2 * stride;
        AscendC::Add(dst[base], src[base], src[base + stride], mask, static_cast<uint8_t>(batch), params);
        AscendC::Sub(dst[base + stride], src[base], src[base + stride], mask, static_cast<uint8_t>(batch), params);
    }
}

class SimHadamardHybrid {
public:
    __aicore__ inline explicit SimHadamardHybrid(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR h16, GM_ADDR output, uint32_t headDim, uint32_t numVectors,
                                uint32_t vectorsPerChunk, uint32_t variant, float invSqrtDim)
    {
        headDim_ = headDim;
        numVectors_ = numVectors;
        vectorsPerChunk_ = vectorsPerChunk;
        variant_ = variant;
        invSqrtDim_ = invSqrtDim;

        rowsPerVector_ = headDim_ / kTile;
        chunkElems_ = vectorsPerChunk_ * headDim_;
        chunkRows_ = vectorsPerChunk_ * rowsPerVector_;
        numChunks_ = numVectors_ / vectorsPerChunk_;

        paddedRows_ = AlignUp(chunkRows_, kTile);
        paddedElems_ = paddedRows_ * kTile;

        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
        h16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(h16));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));

        for (uint32_t slot = 0; slot < kSlots; ++slot) {
            pipe_->InitBuffer(aHi1_[slot], paddedElems_ * sizeof(half));
            pipe_->InitBuffer(aLo1_[slot], paddedElems_ * sizeof(half));
        }
        pipe_->InitBuffer(b1_, kTile * kTile * sizeof(half));

        if ASCEND_IS_AIC {
            pipe_->InitBuffer(a2Hi_, paddedElems_ * sizeof(half));
            pipe_->InitBuffer(a2Lo_, paddedElems_ * sizeof(half));
            pipe_->InitBuffer(b2_, kTile * kTile * sizeof(half));
            pipe_->InitBuffer(co1_, paddedElems_ * sizeof(float));
        }

        pipe_->InitBuffer(inBuf_, paddedElems_ * sizeof(float));
        pipe_->InitBuffer(castBuf_, paddedElems_ * sizeof(half));
        for (uint32_t slot = 0; slot < kSlots; ++slot) {
            pipe_->InitBuffer(prodBuf_[slot], chunkElems_ * sizeof(float));
            pipe_->InitBuffer(tmpBuf_[slot], paddedElems_ * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if ((variant_ & kVariantLockstep) != 0) {
            ProcessLockstep();
            return;
        }
        if ASCEND_IS_AIV {
            ProcessAiv();
        }
        if ASCEND_IS_AIC {
            ProcessAic();
        }
    }

private:
    __aicore__ inline void ProcessAiv()
    {
        StageOperands(0, 0);
        SignalOperandsReady(0);

        for (uint32_t chunk = 1; chunk < numChunks_; ++chunk) {
            const uint32_t slot = chunk % kSlots;
            const uint32_t prev = (chunk - 1) % kSlots;

            if (chunk >= kSlots) {
                AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagOperandsFree + slot));
            }
            StageOperands(chunk * vectorsPerChunk_, slot);
            SignalOperandsReady(slot);

            AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagProductReady + prev));
            Residual(prev, (chunk - 1) * vectorsPerChunk_);
            if (chunk + 1 < numChunks_) {
                SignalProductFree(prev);
            }
        }

        const uint32_t last = (numChunks_ - 1) % kSlots;
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagProductReady + last));
        Residual(last, (numChunks_ - 1) * vectorsPerChunk_);
    }

    __aicore__ inline void ProcessAic()
    {
        for (uint32_t chunk = 0; chunk < numChunks_; ++chunk) {
            const uint32_t slot = chunk % kSlots;

            AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagOperandsReady + slot));
            LoadCubeOperands(slot);
            if (chunk + kSlots < numChunks_) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE1>(static_cast<uint16_t>(kFlagOperandsFree + slot));
            }
            if (chunk >= kSlots) {
                AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagProductFree + slot));
            }
            MmadAndFixpipe(slot);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(static_cast<uint16_t>(kFlagProductReady + slot));
        }
    }

    __aicore__ inline void ProcessLockstep()
    {
        for (uint32_t chunk = 0; chunk < numChunks_; ++chunk) {
            const uint32_t baseVector = chunk * vectorsPerChunk_;
            if ASCEND_IS_AIV {
                StageOperands(baseVector, 0);
                SignalOperandsReady(0);
            }
            if ASCEND_IS_AIC {
                AscendC::CrossCoreWaitFlag(kFlagOperandsReady);
                LoadCubeOperands(0);
                MmadAndFixpipe(0);
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kFlagProductReady);
            }
            if ASCEND_IS_AIV {
                AscendC::CrossCoreWaitFlag(kFlagProductReady);
                Residual(0, baseVector);
            }
        }
    }

    __aicore__ inline void SignalOperandsReady(uint32_t slot)
    {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(static_cast<uint16_t>(kFlagOperandsReady + slot));
    }

    __aicore__ inline void SignalProductFree(uint32_t slot)
    {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(static_cast<uint16_t>(kFlagProductFree + slot));
    }

    __aicore__ inline bool DualDst() const
    {
        return (variant_ & kVariantDualDst) != 0 && (vectorsPerChunk_ % 2 == 0);
    }

    __aicore__ inline uint32_t MyVectors() const
    {
        if (DualDst()) {
            return vectorsPerChunk_ / 2;
        }
        return AscendC::GetSubBlockIdx() == 0 ? vectorsPerChunk_ : 0;
    }

    __aicore__ inline uint32_t MyChunkOffset() const
    {
        if (DualDst()) {
            return static_cast<uint32_t>(AscendC::GetSubBlockIdx()) * (vectorsPerChunk_ / 2);
        }
        return 0;
    }

    __aicore__ inline void StageOperands(uint32_t baseVector, uint32_t slot)
    {
        AscendC::LocalTensor<float> in = inBuf_.Get<float>();
        AscendC::LocalTensor<half> cast = castBuf_.Get<half>();
        AscendC::LocalTensor<float> tmp = tmpBuf_[slot].Get<float>();

        SyncEvent<AscendC::HardEvent::V_MTE2>();

        if (baseVector == 0) {
            AscendC::DataCopy(cast, h16Gm_, kTile * kTile);
            SyncEvent<AscendC::HardEvent::MTE2_MTE3>();
            AscendC::DataCopy(b1_.Get<half>(), cast, kTile * kTile);
        }

        AscendC::DataCopy(in, inputGm_[static_cast<uint64_t>(baseVector) * headDim_], chunkElems_);
        SyncEvent<AscendC::HardEvent::MTE2_V>();
        if (paddedElems_ > chunkElems_) {
            AscendC::Duplicate(in[chunkElems_], 0.0f, paddedElems_ - chunkElems_);
            AscendC::PipeBarrier<PIPE_V>();
        }

        SyncEvent<AscendC::HardEvent::MTE3_V>();
        AscendC::Cast(cast, in, AscendC::RoundMode::CAST_RINT, paddedElems_);
        SyncVectorToMte3();
        AscendC::DataCopy(aHi1_[slot].Get<half>(), cast, paddedElems_);

        if ((variant_ & kVariantHiLo) != 0) {
            AscendC::Cast(tmp, cast, AscendC::RoundMode::CAST_NONE, paddedElems_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(in, in, tmp, paddedElems_);
            AscendC::PipeBarrier<PIPE_V>();
            SyncEvent<AscendC::HardEvent::MTE3_V>();
            AscendC::Cast(cast, in, AscendC::RoundMode::CAST_RINT, paddedElems_);
            SyncVectorToMte3();
            AscendC::DataCopy(aLo1_[slot].Get<half>(), cast, paddedElems_);
        }
    }

    __aicore__ inline void LoadA(const AscendC::LocalTensor<half> &src, const AscendC::LocalTensor<half> &dst)
    {
        AscendC::LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = CeilDivU16(paddedRows_, kTile);
        p.kStep = CeilDivU16(kTile, kHalfC0);
        p.srcStride = CeilDivU16(paddedRows_, kTile);
        p.dstStride = CeilDivU16(paddedRows_, kTile);
        p.ifTranspose = false;
        AscendC::LoadData(dst, src, p);
    }

    __aicore__ inline void LoadB()
    {
        AscendC::LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = CeilDivU16(kTile, kTile);
        p.kStep = CeilDivU16(kTile, kHalfC0);
        p.srcStride = CeilDivU16(kTile, kTile);
        p.dstStride = CeilDivU16(kTile, kTile);
        p.ifTranspose = false;
        AscendC::LoadData(b2_.Get<half>(), b1_.Get<half>(), p);
    }

    __aicore__ inline void LoadCubeOperands(uint32_t slot)
    {
        SyncEvent<AscendC::HardEvent::M_MTE1>();
        LoadA(aHi1_[slot].Get<half>(), a2Hi_.Get<half>());
        if ((variant_ & kVariantHiLo) != 0) {
            LoadA(aLo1_[slot].Get<half>(), a2Lo_.Get<half>());
        }
        LoadB();

        SyncEvent<AscendC::HardEvent::MTE1_M>();
    }

    __aicore__ inline void MmadAndFixpipe(uint32_t slot)
    {
        const bool hiLo = (variant_ & kVariantHiLo) != 0;

        SyncEvent<AscendC::HardEvent::FIX_M>();

        AscendC::LocalTensor<float> acc = co1_.Get<float>();
        AscendC::Mmad(acc, a2Hi_.Get<half>(), b2_.Get<half>(),
                      AscendC::MmadParams(static_cast<uint16_t>(chunkRows_), static_cast<uint16_t>(kTile),
                                          static_cast<uint16_t>(kTile), 0, false, true));
        if (hiLo) {
            AscendC::Mmad(acc, a2Lo_.Get<half>(), b2_.Get<half>(),
                          AscendC::MmadParams(static_cast<uint16_t>(chunkRows_), static_cast<uint16_t>(kTile),
                                              static_cast<uint16_t>(kTile), 0, false, false));
        }

        SyncEvent<AscendC::HardEvent::M_FIX>();

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fp(
            static_cast<uint16_t>(kTile), static_cast<uint16_t>(chunkRows_),
            static_cast<uint16_t>(paddedRows_), kTile);
        if (DualDst()) {
            fp.dualDstCtl = 0b01;
            fp.subBlockId = false;
        }
        AscendC::Fixpipe<float, float, kFixpipeToUb>(prodBuf_[slot].Get<float>(), acc, fp);
        AscendC::PipeBarrier<PIPE_FIX>();
    }

    __aicore__ inline void Residual(uint32_t slot, uint32_t baseVector)
    {
        const uint32_t mine = MyVectors();
        if (mine == 0) {
            return;
        }
        const uint32_t count = mine * headDim_;
        AscendC::LocalTensor<float> src = prodBuf_[slot].Get<float>();
        AscendC::LocalTensor<float> dst = tmpBuf_[slot].Get<float>();

        for (uint32_t stride = kResidualFirstStride; stride < headDim_; stride <<= 1) {
            BlockButterfly(dst, src, stride, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<float> hold = src;
            src = dst;
            dst = hold;
        }

        AscendC::Muls(src, src, invSqrtDim_, count);
        SyncVectorToMte3();
        const uint64_t out = (static_cast<uint64_t>(baseVector) + MyChunkOffset()) * headDim_;
        AscendC::DataCopy(outputGm_[out], src, count);
    }

    AscendC::TPipe *pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> aHi1_[kSlots];
    AscendC::TBuf<AscendC::TPosition::A1> aLo1_[kSlots];
    AscendC::TBuf<AscendC::TPosition::B1> b1_;
    AscendC::TBuf<AscendC::TPosition::A2> a2Hi_;
    AscendC::TBuf<AscendC::TPosition::A2> a2Lo_;
    AscendC::TBuf<AscendC::TPosition::B2> b2_;
    AscendC::TBuf<AscendC::TPosition::CO1> co1_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> inBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> castBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> prodBuf_[kSlots];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf_[kSlots];
    AscendC::GlobalTensor<float> inputGm_;
    AscendC::GlobalTensor<half> h16Gm_;
    AscendC::GlobalTensor<float> outputGm_;
    uint32_t headDim_ = 0;
    uint32_t numVectors_ = 0;
    uint32_t vectorsPerChunk_ = 0;
    uint32_t variant_ = 0;
    uint32_t rowsPerVector_ = 0;
    uint32_t chunkElems_ = 0;
    uint32_t chunkRows_ = 0;
    uint32_t numChunks_ = 0;
    uint32_t paddedRows_ = 0;
    uint32_t paddedElems_ = 0;
    float invSqrtDim_ = 1.0f;
};

class SimHadamardAiv {
public:
    __aicore__ inline explicit SimHadamardAiv(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR tables, GM_ADDR output, uint32_t headDim,
                                uint32_t numVectors, float invSqrtDim)
    {
        headDim_ = headDim;
        numVectors_ = numVectors;
        invSqrtDim_ = invSqrtDim;
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
        tablesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tables));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));

        pipe_->InitBuffer(xBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(tmpBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(swapBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(tableBuf_, kEarlyStages * 2 * headDim_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            AscendC::LocalTensor<int32_t> tables = tableBuf_.Get<int32_t>();
            AscendC::DataCopy(tables, tablesGm_, kEarlyStages * 2 * headDim_);
            SyncEvent<AscendC::HardEvent::MTE2_V>();

            const uint32_t sub = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
            const uint32_t step = static_cast<uint32_t>(AscendC::GetSubBlockNum());
            for (uint32_t v = sub; v < numVectors_; v += step) {
                Transform(v, tables);
            }
        }
    }

private:
    __aicore__ inline void Transform(uint32_t v, const AscendC::LocalTensor<int32_t> &tables)
    {
        AscendC::LocalTensor<float> x = xBuf_.Get<float>();
        AscendC::LocalTensor<float> tmp = tmpBuf_.Get<float>();
        AscendC::LocalTensor<float> swap = swapBuf_.Get<float>();

        AscendC::DataCopy(x, inputGm_[static_cast<uint64_t>(v) * headDim_], headDim_);
        SyncEvent<AscendC::HardEvent::MTE2_V>();

        for (uint32_t stage = 0; stage < kEarlyStages; ++stage) {
            AscendC::LocalTensor<float> sign = tables[stage * 2 * headDim_].ReinterpretCast<float>();
            AscendC::LocalTensor<uint32_t> offsets =
                tables[(stage * 2 + 1) * headDim_].ReinterpretCast<uint32_t>();
            AscendC::Gather(swap, x, offsets, kGatherSrcBase, headDim_);
            AscendC::Mul(tmp, x, sign, headDim_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(x, swap, tmp, headDim_);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::LocalTensor<float> src = x;
        AscendC::LocalTensor<float> dst = tmp;
        for (uint32_t stride = kFp32PerBlock; stride < headDim_; stride <<= 1) {
            BlockButterfly(dst, src, stride, headDim_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<float> hold = src;
            src = dst;
            dst = hold;
        }

        AscendC::Muls(src, src, invSqrtDim_, headDim_);
        SyncVectorToMte3();
        AscendC::DataCopy(outputGm_[static_cast<uint64_t>(v) * headDim_], src, headDim_);
        SyncEvent<AscendC::HardEvent::MTE3_MTE2>();
        SyncEvent<AscendC::HardEvent::MTE3_V>();
    }

    AscendC::TPipe *pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> swapBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tableBuf_;
    AscendC::GlobalTensor<float> inputGm_;
    AscendC::GlobalTensor<int32_t> tablesGm_;
    AscendC::GlobalTensor<float> outputGm_;
    uint32_t headDim_ = 0;
    uint32_t numVectors_ = 0;
    float invSqrtDim_ = 1.0f;
};

}

extern "C" __global__ __aicore__ void sim_hadamard_hybrid(GM_ADDR input, GM_ADDR h16, GM_ADDR output,
                                                          uint32_t headDim, uint32_t numVectors,
                                                          uint32_t vectorsPerChunk, uint32_t variant,
                                                          float invSqrtDim)
{
    AscendC::TPipe pipe;
    SimHadamardHybrid op(&pipe);
    op.Init(input, h16, output, headDim, numVectors, vectorsPerChunk, variant, invSqrtDim);
    op.Process();
}

extern "C" __global__ __aicore__ void sim_hadamard_aiv(GM_ADDR input, GM_ADDR tables, GM_ADDR output,
                                                       uint32_t headDim, uint32_t numVectors, float invSqrtDim)
{
    AscendC::TPipe pipe;
    SimHadamardAiv op(&pipe);
    op.Init(input, tables, output, headDim, numVectors, invSqrtDim);
    op.Process();
}

namespace vllm_ascend {

void sim_hadamard_hybrid_impl(void *stream, void *input, void *h16, void *output, uint32_t headDim,
                              uint32_t numVectors, uint32_t vectorsPerChunk, uint32_t variant, float invSqrtDim)
{
    sim_hadamard_hybrid<<<1, nullptr, stream>>>(input, h16, output, headDim, numVectors, vectorsPerChunk, variant,
                                                invSqrtDim);
}

void sim_hadamard_aiv_impl(void *stream, void *input, void *tables, void *output, uint32_t headDim,
                           uint32_t numVectors, float invSqrtDim)
{
    sim_hadamard_aiv<<<1, nullptr, stream>>>(input, tables, output, headDim, numVectors, invSqrtDim);
}

}
