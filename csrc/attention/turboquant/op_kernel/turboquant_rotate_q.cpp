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

#include "../../../kernels/types.h"
#include "common/turboquant_codec_950.h"

namespace {

using vllm_ascend::turboquant::CeilDiv;
using vllm_ascend::turboquant::kRotateQTile;
using vllm_ascend::turboquant::MixBlockIdx;
using vllm_ascend::turboquant::RepeatButterfly;
using vllm_ascend::turboquant::RotateQVariant;
using vllm_ascend::turboquant::SyncEvent;
using vllm_ascend::turboquant::SyncVectorToMte3;
using vllm_ascend::turboquant::VecBarrier;

using TurboQuantCodec4 = vllm_ascend::turboquant::TurboQuantCodec<4>;

constexpr uint32_t kHalfC0 = 16;

constexpr uint32_t kResidualFirstStride = kRotateQTile;

constexpr bool kRotateQVecBarriers = false;

constexpr uint16_t kFlagOperandsReady = 0;
constexpr uint16_t kFlagOperandsFree = 2;
constexpr uint16_t kFlagProductReady = 4;
constexpr uint16_t kFlagProductFree = 6;

constexpr uint32_t kSlots = 2;

constexpr AscendC::FixpipeConfig kFixpipeToUb = {AscendC::CO2Layout::ROW_MAJOR, true};

__aicore__ inline uint32_t AlignUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }

__aicore__ inline uint16_t CeilDivU16(uint32_t a, uint32_t b)
{
    return static_cast<uint16_t>((a + b - 1) / b);
}

__aicore__ inline void BlockButterfly(const AscendC::LocalTensor<float> &dst,
                                      const AscendC::LocalTensor<float> &src, uint32_t stride, uint32_t count)
{
    RepeatButterfly(dst, src, stride, count / (2 * stride));
}

template <typename scalar_t>
class TurboQuantRotateQCube {
public:
    __aicore__ inline explicit TurboQuantRotateQCube(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR piSigns, GM_ADDR h16, GM_ADDR queryRot, uint32_t numVectors,
                                uint32_t headSize, uint32_t vectorsPerBlock, uint32_t vectorsPerChunk,
                                uint32_t variant, float invSqrtLen)
    {
        headSize_ = headSize;
        numVectors_ = numVectors;
        vectorsPerChunk_ = vectorsPerChunk;
        variant_ = variant;
        invSqrtLen_ = invSqrtLen;

        blockBase_ = MixBlockIdx() * vectorsPerBlock;
        uint32_t mine = 0;
        if (blockBase_ < numVectors_) {
            mine = numVectors_ - blockBase_;
            if (mine > vectorsPerBlock) {
                mine = vectorsPerBlock;
            }
        }
        blockVectors_ = mine;

        rowsPerVector_ = headSize_ / kRotateQTile;
        chunkElems_ = vectorsPerChunk_ * headSize_;
        chunkRows_ = vectorsPerChunk_ * rowsPerVector_;
        numChunks_ = vectorsPerChunk_ == 0 ? 0 : blockVectors_ / vectorsPerChunk_;

        paddedRows_ = AlignUp(chunkRows_, kRotateQTile);
        paddedElems_ = paddedRows_ * kRotateQTile;

        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(query));
        piSignsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(piSigns), headSize_);
        h16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(h16));
        queryRotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(queryRot));

        for (uint32_t slot = 0; slot < kSlots; ++slot) {
            pipe_->InitBuffer(aHi1_[slot], paddedElems_ * sizeof(half));
            if (HiLo()) {
                pipe_->InitBuffer(aLo1_[slot], paddedElems_ * sizeof(half));
            }
        }
        pipe_->InitBuffer(b1_, kRotateQTile * kRotateQTile * sizeof(half));

        if ASCEND_IS_AIC {
            pipe_->InitBuffer(a2Hi_, paddedElems_ * sizeof(half));
            if (HiLo()) {
                pipe_->InitBuffer(a2Lo_, paddedElems_ * sizeof(half));
            }
            pipe_->InitBuffer(b2_, kRotateQTile * kRotateQTile * sizeof(half));
            pipe_->InitBuffer(co1_, paddedElems_ * sizeof(float));
        }

        pipe_->InitBuffer(qInBuf_, chunkElems_ * sizeof(scalar_t));
        pipe_->InitBuffer(inBuf_, paddedElems_ * sizeof(float));
        pipe_->InitBuffer(castBuf_, paddedElems_ * sizeof(half));
        pipe_->InitBuffer(signBuf_, headSize_ * sizeof(float));
        for (uint32_t slot = 0; slot < kSlots; ++slot) {
            pipe_->InitBuffer(prodBuf_[slot], chunkElems_ * sizeof(float));
            pipe_->InitBuffer(tmpBuf_[slot], paddedElems_ * sizeof(float));
        }
        pipe_->InitBuffer(scaledSignBuf_, headSize_ * sizeof(float));

        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::DataCopy(signs, piSignsGm_, headSize_);
        AscendC::PipeBarrier<PIPE_ALL>();
        if ASCEND_IS_AIV {
            AscendC::Muls(scaledSignBuf_.Get<float>(), signs, invSqrtLen_, headSize_);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Process()
    {
        if (numChunks_ == 0) {
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
            StageOperands(chunk, slot);
            SignalOperandsReady(slot);

            AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagProductReady + prev));
            Residual(prev, chunk - 1);
            if (chunk + 1 < numChunks_) {
                SignalProductFree(prev);
            }
        }

        const uint32_t last = (numChunks_ - 1) % kSlots;
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagProductReady + last));
        Residual(last, numChunks_ - 1);
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

    __aicore__ inline void SignalOperandsReady(uint32_t slot)
    {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(static_cast<uint16_t>(kFlagOperandsReady + slot));
    }

    __aicore__ inline void SignalProductFree(uint32_t slot)
    {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(static_cast<uint16_t>(kFlagProductFree + slot));
    }

    __aicore__ inline bool HiLo() const { return (variant_ & RotateQVariant::kHiLo) != 0; }

    __aicore__ inline bool DualDst() const
    {
        return (variant_ & RotateQVariant::kDualDst) != 0 && (vectorsPerChunk_ % 2 == 0);
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

    __aicore__ inline void StageOperands(uint32_t chunk, uint32_t slot)
    {
        AscendC::LocalTensor<scalar_t> qIn = qInBuf_.Get<scalar_t>();
        AscendC::LocalTensor<float> in = inBuf_.Get<float>();
        AscendC::LocalTensor<half> cast = castBuf_.Get<half>();
        AscendC::LocalTensor<float> tmp = tmpBuf_[slot].Get<float>();
        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();

        if (chunk > 0) {
            SyncEvent<AscendC::HardEvent::V_MTE2>();
        }

        if (!h16Staged_) {
            AscendC::DataCopy(cast, h16Gm_, kRotateQTile * kRotateQTile);
            SyncEvent<AscendC::HardEvent::MTE2_MTE3>();
            AscendC::DataCopy(b1_.Get<half>(), cast, kRotateQTile * kRotateQTile);
            h16Staged_ = true;
        }

        const uint64_t base = static_cast<uint64_t>(blockBase_ + chunk * vectorsPerChunk_) * headSize_;
        AscendC::DataCopy(qIn, queryGm_[base], chunkElems_);
        SyncEvent<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(in, qIn, AscendC::RoundMode::CAST_NONE, chunkElems_);
        if (paddedElems_ > chunkElems_) {
            AscendC::Duplicate(in[chunkElems_], 0.0f, paddedElems_ - chunkElems_);
        }
        VecBarrier<kRotateQVecBarriers>();

        for (uint32_t v = 0; v < vectorsPerChunk_; ++v) {
            AscendC::Mul(in[v * headSize_], in[v * headSize_], signs, headSize_);
        }
        VecBarrier<kRotateQVecBarriers>();

        SyncEvent<AscendC::HardEvent::MTE3_V>();
        AscendC::Cast(cast, in, AscendC::RoundMode::CAST_RINT, paddedElems_);
        SyncVectorToMte3();
        AscendC::DataCopy(aHi1_[slot].Get<half>(), cast, paddedElems_);

        if (HiLo()) {
            AscendC::Cast(tmp, cast, AscendC::RoundMode::CAST_NONE, paddedElems_);
            VecBarrier<kRotateQVecBarriers>();
            AscendC::Sub(in, in, tmp, paddedElems_);
            VecBarrier<kRotateQVecBarriers>();
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
        p.mStep = CeilDivU16(paddedRows_, kRotateQTile);
        p.kStep = CeilDivU16(kRotateQTile, kHalfC0);
        p.srcStride = CeilDivU16(paddedRows_, kRotateQTile);
        p.dstStride = CeilDivU16(paddedRows_, kRotateQTile);
        p.ifTranspose = false;
        AscendC::LoadData(dst, src, p);
    }

    __aicore__ inline void LoadB()
    {
        AscendC::LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = CeilDivU16(kRotateQTile, kRotateQTile);
        p.kStep = CeilDivU16(kRotateQTile, kHalfC0);
        p.srcStride = CeilDivU16(kRotateQTile, kRotateQTile);
        p.dstStride = CeilDivU16(kRotateQTile, kRotateQTile);
        p.ifTranspose = false;
        AscendC::LoadData(b2_.Get<half>(), b1_.Get<half>(), p);
    }

    __aicore__ inline void LoadCubeOperands(uint32_t slot)
    {
        SyncEvent<AscendC::HardEvent::M_MTE1>();
        LoadA(aHi1_[slot].Get<half>(), a2Hi_.Get<half>());
        if (HiLo()) {
            LoadA(aLo1_[slot].Get<half>(), a2Lo_.Get<half>());
        }
        LoadB();
        SyncEvent<AscendC::HardEvent::MTE1_M>();
    }

    __aicore__ inline void MmadAndFixpipe(uint32_t slot)
    {
        SyncEvent<AscendC::HardEvent::FIX_M>();

        AscendC::LocalTensor<float> acc = co1_.Get<float>();
        AscendC::Mmad(acc, a2Hi_.Get<half>(), b2_.Get<half>(),
                      AscendC::MmadParams(static_cast<uint16_t>(chunkRows_), static_cast<uint16_t>(kRotateQTile),
                                          static_cast<uint16_t>(kRotateQTile), 0, false, true));
        if (HiLo()) {
            AscendC::Mmad(acc, a2Lo_.Get<half>(), b2_.Get<half>(),
                          AscendC::MmadParams(static_cast<uint16_t>(chunkRows_), static_cast<uint16_t>(kRotateQTile),
                                              static_cast<uint16_t>(kRotateQTile), 0, false, false));
        }

        SyncEvent<AscendC::HardEvent::M_FIX>();

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fp(
            static_cast<uint16_t>(kRotateQTile), static_cast<uint16_t>(chunkRows_),
            static_cast<uint16_t>(paddedRows_), kRotateQTile);
        if (DualDst()) {
            fp.dualDstCtl = 0b01;
            fp.subBlockId = false;
        }
        AscendC::Fixpipe<float, float, kFixpipeToUb>(prodBuf_[slot].Get<float>(), acc, fp);
        AscendC::PipeBarrier<PIPE_FIX>();
    }

    __aicore__ inline void Residual(uint32_t slot, uint32_t chunk)
    {
        const uint32_t mine = MyVectors();
        if (mine == 0) {
            return;
        }
        const uint32_t count = mine * headSize_;
        AscendC::LocalTensor<float> src = prodBuf_[slot].Get<float>();
        AscendC::LocalTensor<float> dst = tmpBuf_[slot].Get<float>();
        AscendC::LocalTensor<float> scaledSigns = scaledSignBuf_.Get<float>();

        for (uint32_t stride = kResidualFirstStride; stride < headSize_; stride <<= 1) {
            BlockButterfly(dst, src, stride, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<float> hold = src;
            src = dst;
            dst = hold;
        }

        for (uint32_t v = 0; v < mine; ++v) {
            AscendC::Mul(src[v * headSize_], src[v * headSize_], scaledSigns, headSize_);
        }

        SyncVectorToMte3();
        const uint64_t out =
            (static_cast<uint64_t>(blockBase_ + chunk * vectorsPerChunk_) + MyChunkOffset()) * headSize_;
        AscendC::DataCopy(queryRotGm_[out], src, count);
    }

    AscendC::TPipe *pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> aHi1_[kSlots];
    AscendC::TBuf<AscendC::TPosition::A1> aLo1_[kSlots];
    AscendC::TBuf<AscendC::TPosition::B1> b1_;
    AscendC::TBuf<AscendC::TPosition::A2> a2Hi_;
    AscendC::TBuf<AscendC::TPosition::A2> a2Lo_;
    AscendC::TBuf<AscendC::TPosition::B2> b2_;
    AscendC::TBuf<AscendC::TPosition::CO1> co1_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> qInBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> inBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> castBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> signBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> prodBuf_[kSlots];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf_[kSlots];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaledSignBuf_;
    AscendC::GlobalTensor<scalar_t> queryGm_;
    AscendC::GlobalTensor<float> piSignsGm_;
    AscendC::GlobalTensor<half> h16Gm_;
    AscendC::GlobalTensor<float> queryRotGm_;
    uint32_t headSize_ = 0;
    uint32_t numVectors_ = 0;
    uint32_t vectorsPerChunk_ = 0;
    uint32_t variant_ = 0;
    uint32_t blockBase_ = 0;
    uint32_t blockVectors_ = 0;
    uint32_t rowsPerVector_ = 0;
    uint32_t chunkElems_ = 0;
    uint32_t chunkRows_ = 0;
    uint32_t numChunks_ = 0;
    uint32_t paddedRows_ = 0;
    uint32_t paddedElems_ = 0;
    float invSqrtLen_ = 1.0f;
    bool h16Staged_ = false;
};

template <typename scalar_t>
class TurboQuantRotateQAiv {
public:
    __aicore__ inline explicit TurboQuantRotateQAiv(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR piSigns, GM_ADDR rotTables, GM_ADDR queryRot,
                                uint32_t numVectors, uint32_t headSize, uint32_t vectorsPerBlock, float invSqrtLen)
    {
        headSize_ = headSize;
        numVectors_ = numVectors;
        vectorsPerBlock_ = vectorsPerBlock;

        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(query));
        piSignsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(piSigns), headSize_);
        rotTablesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rotTables));
        queryRotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(queryRot));

        pipe_->InitBuffer(qInBuf_, headSize_ * sizeof(scalar_t));
        pipe_->InitBuffer(workBuf_, 2 * headSize_ * sizeof(float));
        pipe_->InitBuffer(signBuf_, headSize_ * sizeof(float));

        codec_.Init(pipe_, headSize_, 1, invSqrtLen, rotTablesGm_);

        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::DataCopy(signs, piSignsGm_, headSize_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const uint32_t base = MixBlockIdx() * vectorsPerBlock_;
            if (base >= numVectors_) {
                return;
            }
            uint32_t end = base + vectorsPerBlock_;
            if (end > numVectors_) {
                end = numVectors_;
            }
            const uint32_t sub = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
            const uint32_t step = static_cast<uint32_t>(AscendC::GetSubBlockNum());
            for (uint32_t v = base + sub; v < end; v += step) {
                Rotate(v);
            }
        }
    }

private:
    __aicore__ inline void Rotate(uint32_t vector)
    {
        AscendC::LocalTensor<scalar_t> qIn = qInBuf_.Get<scalar_t>();
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        AscendC::LocalTensor<float> x = work;
        AscendC::LocalTensor<float> tmp = work[headSize_];
        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();

        AscendC::DataCopy(qIn, queryGm_[static_cast<uint64_t>(vector) * headSize_], headSize_);
        SyncEvent<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(x, qIn, AscendC::RoundMode::CAST_NONE, headSize_);
        VecBarrier<kRotateQVecBarriers>();

        codec_.ApplyPi<kRotateQVecBarriers>(x, tmp, signs, static_cast<int>(headSize_));

        SyncVectorToMte3();
        AscendC::DataCopy(queryRotGm_[static_cast<uint64_t>(vector) * headSize_], x, headSize_);
        SyncEvent<AscendC::HardEvent::MTE3_MTE2>();
        SyncEvent<AscendC::HardEvent::MTE3_V>();
    }

    AscendC::TPipe *pipe_;
    TurboQuantCodec4 codec_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> qInBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> workBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> signBuf_;
    AscendC::GlobalTensor<scalar_t> queryGm_;
    AscendC::GlobalTensor<float> piSignsGm_;
    AscendC::GlobalTensor<int32_t> rotTablesGm_;
    AscendC::GlobalTensor<float> queryRotGm_;
    uint32_t headSize_ = 0;
    uint32_t numVectors_ = 0;
    uint32_t vectorsPerBlock_ = 0;
};

}

#define TURBOQUANT_ROTATE_Q_CUBE_DECLARE(TYPE)                                                                       \
    extern "C" __global__ __aicore__ void turboquant_rotate_q_cube_##TYPE(                                            \
        GM_ADDR query, GM_ADDR piSigns, GM_ADDR h16, GM_ADDR queryRot, uint32_t numVectors, uint32_t headSize,       \
        uint32_t vectorsPerBlock, uint32_t vectorsPerChunk, uint32_t variant, float invSqrtLen)                       \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantRotateQCube<TYPE> op(&pipe);                                                                       \
        op.Init(query, piSigns, h16, queryRot, numVectors, headSize, vectorsPerBlock, vectorsPerChunk, variant,      \
                invSqrtLen);                                                                                         \
        op.Process();                                                                                                \
    }

#define TURBOQUANT_ROTATE_Q_AIV_DECLARE(TYPE)                                                                        \
    extern "C" __global__ __aicore__ void turboquant_rotate_q_aiv_##TYPE(                                             \
        GM_ADDR query, GM_ADDR piSigns, GM_ADDR rotTables, GM_ADDR queryRot, uint32_t numVectors,                    \
        uint32_t headSize, uint32_t vectorsPerBlock, float invSqrtLen)                                                \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantRotateQAiv<TYPE> op(&pipe);                                                                        \
        op.Init(query, piSigns, rotTables, queryRot, numVectors, headSize, vectorsPerBlock, invSqrtLen);             \
        op.Process();                                                                                                \
    }

TURBOQUANT_ROTATE_Q_CUBE_DECLARE(half)
TURBOQUANT_ROTATE_Q_AIV_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
TURBOQUANT_ROTATE_Q_CUBE_DECLARE(bfloat16_t)
TURBOQUANT_ROTATE_Q_AIV_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {

void turboquant_rotate_q_impl(AscendType type, void *stream, uint32_t blockDim, bool useCube, void *query,
                              void *piSigns, void *h16, void *rotTables, void *queryRot, uint32_t numVectors,
                              uint32_t headSize, uint32_t vectorsPerBlock, uint32_t vectorsPerChunk,
                              uint32_t variant, float invSqrtLen)
{
    if (blockDim == 0 || numVectors == 0) {
        return;
    }
    if (type == AscendType::FP16) {
        if (useCube) {
            turboquant_rotate_q_cube_half<<<blockDim, nullptr, stream>>>(
                query, piSigns, h16, queryRot, numVectors, headSize, vectorsPerBlock, vectorsPerChunk, variant,
                invSqrtLen);
        } else {
            turboquant_rotate_q_aiv_half<<<blockDim, nullptr, stream>>>(
                query, piSigns, rotTables, queryRot, numVectors, headSize, vectorsPerBlock, invSqrtLen);
        }
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    } else if (type == AscendType::BF16) {
        if (useCube) {
            turboquant_rotate_q_cube_bfloat16_t<<<blockDim, nullptr, stream>>>(
                query, piSigns, h16, queryRot, numVectors, headSize, vectorsPerBlock, vectorsPerChunk, variant,
                invSqrtLen);
        } else {
            turboquant_rotate_q_aiv_bfloat16_t<<<blockDim, nullptr, stream>>>(
                query, piSigns, rotTables, queryRot, numVectors, headSize, vectorsPerBlock, invSqrtLen);
        }
#endif
    }
}

}
