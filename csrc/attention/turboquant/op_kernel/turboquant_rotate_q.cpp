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

using vllm_ascend::turboquant::AlignUp;
using vllm_ascend::turboquant::CeilDivU16;
using vllm_ascend::turboquant::kFixpipeToUb;
using vllm_ascend::turboquant::kRotateQTile;
using vllm_ascend::turboquant::kSubBlockSyncMode;
using vllm_ascend::turboquant::MixBlockIdx;
using vllm_ascend::turboquant::RepeatButterfly;
using vllm_ascend::turboquant::RotateQVariant;
using vllm_ascend::turboquant::SyncEvent;
using vllm_ascend::turboquant::SyncMatrixToFixpipe;
using vllm_ascend::turboquant::SyncMte1ToMatrix;
using vllm_ascend::turboquant::SyncVectorToMte3;

using TurboQuantCodec4 = vllm_ascend::turboquant::TurboQuantCodec<4>;

// The fp16 operand's C0: a 16 x 16 fractal of the H16 Hadamard factor.
constexpr uint32_t kHalfOperandC0 = 16;

// The Cube applies the lower four butterfly stages (H16); the vector unit starts at stride 16.
constexpr uint32_t kResidualFirstStride = kRotateQTile;

constexpr uint16_t kRotateQFlagOperandsReady = 0;
constexpr uint16_t kRotateQFlagOperandsFree = 2;
constexpr uint16_t kRotateQFlagProductReady = 4;
constexpr uint16_t kRotateQFlagProductFree = 6;

constexpr uint32_t kOperandSlots = 2;

__aicore__ inline void BlockButterfly(const AscendC::LocalTensor<float> &dst, const AscendC::LocalTensor<float> &src,
                                      const uint32_t stride, const uint32_t count)
{
    RepeatButterfly(dst, src, stride, count / (2 * stride));
}

template <typename scalar_t>
class TurboQuantRotateQCube {
public:
    __aicore__ inline explicit TurboQuantRotateQCube(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR piSigns, GM_ADDR h16, GM_ADDR queryRot, const uint32_t numVectors,
                                const uint32_t headSize, const uint32_t vectorsPerBlock, const uint32_t vectorsPerChunk,
                                const uint32_t variant, const float invSqrtLen)
    {
        headSize_ = headSize;
        numVectors_ = numVectors;
        vectorsPerChunk_ = vectorsPerChunk;
        variant_ = variant;
        invSqrtLen_ = invSqrtLen;

        blockBase_ = MixBlockIdx() * vectorsPerBlock;
        uint32_t blockVectors = 0;
        if (blockBase_ < numVectors_) {
            blockVectors = numVectors_ - blockBase_;
            if (blockVectors > vectorsPerBlock) {
                blockVectors = vectorsPerBlock;
            }
        }
        blockVectors_ = blockVectors;

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

        for (uint32_t slot = 0; slot < kOperandSlots; ++slot) {
            pipe_->InitBuffer(queryHiL1Buf_[slot], paddedElems_ * sizeof(half));
            if (HiLo()) {
                pipe_->InitBuffer(queryLoL1Buf_[slot], paddedElems_ * sizeof(half));
            }
        }
        pipe_->InitBuffer(h16L1Buf_, kRotateQTile * kRotateQTile * sizeof(half));

        if ASCEND_IS_AIC {
            pipe_->InitBuffer(queryHiL0aBuf_, paddedElems_ * sizeof(half));
            if (HiLo()) {
                pipe_->InitBuffer(queryLoL0aBuf_, paddedElems_ * sizeof(half));
            }
            pipe_->InitBuffer(h16L0bBuf_, kRotateQTile * kRotateQTile * sizeof(half));
            pipe_->InitBuffer(productL0cBuf_, paddedElems_ * sizeof(float));
        }

        pipe_->InitBuffer(queryInBuf_, chunkElems_ * sizeof(scalar_t));
        pipe_->InitBuffer(queryFloatBuf_, paddedElems_ * sizeof(float));
        pipe_->InitBuffer(queryHalfBuf_, paddedElems_ * sizeof(half));
        pipe_->InitBuffer(signBuf_, headSize_ * sizeof(float));
        for (uint32_t slot = 0; slot < kOperandSlots; ++slot) {
            pipe_->InitBuffer(productBuf_[slot], chunkElems_ * sizeof(float));
            pipe_->InitBuffer(butterflyBuf_[slot], paddedElems_ * sizeof(float));
        }
        pipe_->InitBuffer(scaledSignBuf_, headSize_ * sizeof(float));

        const AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::DataCopy(signs, piSignsGm_, headSize_);
        // Init hand-off: the signs land in UB before the vector unit scales them.
        AscendC::PipeBarrier<PIPE_ALL>();
        if ASCEND_IS_AIV {
            AscendC::Muls(scaledSignBuf_.Get<float>(), signs, invSqrtLen_, headSize_);
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
        SyncOperandsReady(0);

        for (uint32_t chunk = 1; chunk < numChunks_; ++chunk) {
            const uint32_t slot = chunk % kOperandSlots;
            const uint32_t prevSlot = (chunk - 1) % kOperandSlots;

            if (chunk >= kOperandSlots) {
                AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kRotateQFlagOperandsFree + slot));
            }
            StageOperands(chunk, slot);
            SyncOperandsReady(slot);

            AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kRotateQFlagProductReady + prevSlot));
            ComputeResidual(prevSlot, chunk - 1);
            if (chunk + 1 < numChunks_) {
                SyncProductFree(prevSlot);
            }
        }

        const uint32_t lastSlot = (numChunks_ - 1) % kOperandSlots;
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kRotateQFlagProductReady + lastSlot));
        ComputeResidual(lastSlot, numChunks_ - 1);
    }

    __aicore__ inline void ProcessAic()
    {
        for (uint32_t chunk = 0; chunk < numChunks_; ++chunk) {
            const uint32_t slot = chunk % kOperandSlots;

            AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kRotateQFlagOperandsReady + slot));
            StageCubeOperands(slot);
            if (chunk + kOperandSlots < numChunks_) {
                AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE1>(
                    static_cast<uint16_t>(kRotateQFlagOperandsFree + slot));
            }
            if (chunk >= kOperandSlots) {
                AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kRotateQFlagProductFree + slot));
            }
            ComputeProduct(slot);
            AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(
                static_cast<uint16_t>(kRotateQFlagProductReady + slot));
        }
    }

    __aicore__ inline void SyncOperandsReady(const uint32_t slot)
    {
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(static_cast<uint16_t>(kRotateQFlagOperandsReady + slot));
    }

    __aicore__ inline void SyncProductFree(const uint32_t slot)
    {
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(static_cast<uint16_t>(kRotateQFlagProductFree + slot));
    }

    __aicore__ inline bool HiLo() const { return (variant_ & RotateQVariant::kHiLo) != 0; }

    __aicore__ inline bool DualDst() const
    {
        return (variant_ & RotateQVariant::kDualDst) != 0 && (vectorsPerChunk_ % 2 == 0);
    }

    __aicore__ inline uint32_t OwnedVectors() const
    {
        if (DualDst()) {
            return vectorsPerChunk_ / 2;
        }
        return AscendC::GetSubBlockIdx() == 0 ? vectorsPerChunk_ : 0;
    }

    __aicore__ inline uint32_t OwnedChunkOffset() const
    {
        if (DualDst()) {
            return static_cast<uint32_t>(AscendC::GetSubBlockIdx()) * (vectorsPerChunk_ / 2);
        }
        return 0;
    }

    // One chunk of query vectors, sign-flipped and cast to the fp16 operand grid, into L1 slot `slot`. With
    // HiLo the rounding residual is cast into a second operand.
    __aicore__ inline void StageOperands(const uint32_t chunk, const uint32_t slot)
    {
        const AscendC::LocalTensor<scalar_t> queryIn = queryInBuf_.Get<scalar_t>();
        const AscendC::LocalTensor<float> queryFloat = queryFloatBuf_.Get<float>();
        const AscendC::LocalTensor<half> queryHalf = queryHalfBuf_.Get<half>();
        const AscendC::LocalTensor<float> hiWidened = butterflyBuf_[slot].Get<float>();
        const AscendC::LocalTensor<float> signs = signBuf_.Get<float>();

        if (chunk > 0) {
            SyncEvent<AscendC::HardEvent::V_MTE2>();
        }

        if (!h16Staged_) {
            AscendC::DataCopy(queryHalf, h16Gm_, kRotateQTile * kRotateQTile);
            SyncEvent<AscendC::HardEvent::MTE2_MTE3>();
            AscendC::DataCopy(h16L1Buf_.Get<half>(), queryHalf, kRotateQTile * kRotateQTile);
            h16Staged_ = true;
        }

        const uint64_t base = static_cast<uint64_t>(blockBase_ + chunk * vectorsPerChunk_) * headSize_;
        AscendC::DataCopy(queryIn, queryGm_[base], chunkElems_);
        SyncEvent<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(queryFloat, queryIn, AscendC::RoundMode::CAST_NONE, chunkElems_);
        if (paddedElems_ > chunkElems_) {
            AscendC::Duplicate(queryFloat[chunkElems_], 0.0f, paddedElems_ - chunkElems_);
        }

        for (uint32_t vector = 0; vector < vectorsPerChunk_; ++vector) {
            AscendC::Mul(queryFloat[vector * headSize_], queryFloat[vector * headSize_], signs, headSize_);
        }

        SyncEvent<AscendC::HardEvent::MTE3_V>();
        AscendC::Cast(queryHalf, queryFloat, AscendC::RoundMode::CAST_RINT, paddedElems_);
        SyncVectorToMte3();
        AscendC::DataCopy(queryHiL1Buf_[slot].Get<half>(), queryHalf, paddedElems_);

        if (HiLo()) {
            AscendC::Cast(hiWidened, queryHalf, AscendC::RoundMode::CAST_NONE, paddedElems_);
                AscendC::Sub(queryFloat, queryFloat, hiWidened, paddedElems_);
                SyncEvent<AscendC::HardEvent::MTE3_V>();
            AscendC::Cast(queryHalf, queryFloat, AscendC::RoundMode::CAST_RINT, paddedElems_);
            SyncVectorToMte3();
            AscendC::DataCopy(queryLoL1Buf_[slot].Get<half>(), queryHalf, paddedElems_);
        }
    }

    __aicore__ inline void StageToL0a(const AscendC::LocalTensor<half> &srcL1, const AscendC::LocalTensor<half> &dstL0a)
    {
        AscendC::LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = CeilDivU16(paddedRows_, kRotateQTile);
        params.kStep = CeilDivU16(kRotateQTile, kHalfOperandC0);
        params.srcStride = CeilDivU16(paddedRows_, kRotateQTile);
        params.dstStride = CeilDivU16(paddedRows_, kRotateQTile);
        params.ifTranspose = false;
        AscendC::LoadData(dstL0a, srcL1, params);
    }

    __aicore__ inline void StageH16ToL0b()
    {
        AscendC::LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = CeilDivU16(kRotateQTile, kRotateQTile);
        params.kStep = CeilDivU16(kRotateQTile, kHalfOperandC0);
        params.srcStride = CeilDivU16(kRotateQTile, kRotateQTile);
        params.dstStride = CeilDivU16(kRotateQTile, kRotateQTile);
        params.ifTranspose = false;
        AscendC::LoadData(h16L0bBuf_.Get<half>(), h16L1Buf_.Get<half>(), params);
    }

    __aicore__ inline void StageCubeOperands(const uint32_t slot)
    {
        SyncEvent<AscendC::HardEvent::M_MTE1>();
        StageToL0a(queryHiL1Buf_[slot].Get<half>(), queryHiL0aBuf_.Get<half>());
        if (HiLo()) {
            StageToL0a(queryLoL1Buf_[slot].Get<half>(), queryLoL0aBuf_.Get<half>());
        }
        StageH16ToL0b();
        SyncMte1ToMatrix();
    }

    // Query rows x H16 (plus the low residual rows with HiLo), Fixpiped into productBuf_[slot].
    __aicore__ inline void ComputeProduct(const uint32_t slot)
    {
        SyncEvent<AscendC::HardEvent::FIX_M>();

        const AscendC::LocalTensor<float> product = productL0cBuf_.Get<float>();
        AscendC::Mmad(product, queryHiL0aBuf_.Get<half>(), h16L0bBuf_.Get<half>(),
                      AscendC::MmadParams(static_cast<uint16_t>(chunkRows_), static_cast<uint16_t>(kRotateQTile),
                                          static_cast<uint16_t>(kRotateQTile), 0, false, true));
        if (HiLo()) {
            AscendC::Mmad(product, queryLoL0aBuf_.Get<half>(), h16L0bBuf_.Get<half>(),
                          AscendC::MmadParams(static_cast<uint16_t>(chunkRows_), static_cast<uint16_t>(kRotateQTile),
                                              static_cast<uint16_t>(kRotateQTile), 0, false, false));
        }

        SyncMatrixToFixpipe();

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixParams(
            static_cast<uint16_t>(kRotateQTile), static_cast<uint16_t>(chunkRows_),
            static_cast<uint16_t>(paddedRows_), kRotateQTile);
        if (DualDst()) {
            fixParams.dualDstCtl = 0b01;
            fixParams.subBlockId = false;
        }
        AscendC::Fixpipe<float, float, kFixpipeToUb>(productBuf_[slot].Get<float>(), product, fixParams);
        // Drains the Fixpipe before the product-ready flag releases the vector unit onto its UB rows.
        AscendC::PipeBarrier<PIPE_FIX>();
    }

    // The butterfly stages above H16 on this subcore's vectors, the output sign flip and 1 / sqrt(D), and the
    // write to GM.
    __aicore__ inline void ComputeResidual(const uint32_t slot, const uint32_t chunk)
    {
        const uint32_t owned = OwnedVectors();
        if (owned == 0) {
            return;
        }
        const uint32_t count = owned * headSize_;
        AscendC::LocalTensor<float> src = productBuf_[slot].Get<float>();
        AscendC::LocalTensor<float> dst = butterflyBuf_[slot].Get<float>();
        const AscendC::LocalTensor<float> scaledSigns = scaledSignBuf_.Get<float>();

        for (uint32_t stride = kResidualFirstStride; stride < headSize_; stride <<= 1) {
            BlockButterfly(dst, src, stride, count);
            // FWHT ping-pong: this pass's destination is the next pass's source over the same two buffers.
            AscendC::PipeBarrier<PIPE_V>();
            const AscendC::LocalTensor<float> hold = src;
            src = dst;
            dst = hold;
        }

        for (uint32_t vector = 0; vector < owned; ++vector) {
            AscendC::Mul(src[vector * headSize_], src[vector * headSize_], scaledSigns, headSize_);
        }

        SyncVectorToMte3();
        const uint64_t out =
            (static_cast<uint64_t>(blockBase_ + chunk * vectorsPerChunk_) + OwnedChunkOffset()) * headSize_;
        AscendC::DataCopy(queryRotGm_[out], src, count);
    }

    AscendC::TPipe *pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> queryHiL1Buf_[kOperandSlots];
    AscendC::TBuf<AscendC::TPosition::A1> queryLoL1Buf_[kOperandSlots];
    AscendC::TBuf<AscendC::TPosition::B1> h16L1Buf_;
    AscendC::TBuf<AscendC::TPosition::A2> queryHiL0aBuf_;
    AscendC::TBuf<AscendC::TPosition::A2> queryLoL0aBuf_;
    AscendC::TBuf<AscendC::TPosition::B2> h16L0bBuf_;
    AscendC::TBuf<AscendC::TPosition::CO1> productL0cBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> queryInBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> queryFloatBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> queryHalfBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> signBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> productBuf_[kOperandSlots];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> butterflyBuf_[kOperandSlots];
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
                                const uint32_t numVectors, const uint32_t headSize, const uint32_t vectorsPerBlock,
                                const float invSqrtLen)
    {
        headSize_ = headSize;
        numVectors_ = numVectors;
        vectorsPerBlock_ = vectorsPerBlock;

        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(query));
        piSignsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(piSigns), headSize_);
        rotTablesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rotTables));
        queryRotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(queryRot));

        pipe_->InitBuffer(queryInBuf_, headSize_ * sizeof(scalar_t));
        pipe_->InitBuffer(workBuf_, 2 * headSize_ * sizeof(float));
        pipe_->InitBuffer(signBuf_, headSize_ * sizeof(float));

        codec_.Init(pipe_, headSize_, 1, invSqrtLen, rotTablesGm_);

        const AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::DataCopy(signs, piSignsGm_, headSize_);
        // Init hand-off: the signs and the codec tables land in UB before the first rotation.
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
            const uint32_t subcore = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
            const uint32_t subcores = static_cast<uint32_t>(AscendC::GetSubBlockNum());
            for (uint32_t vector = base + subcore; vector < end; vector += subcores) {
                ComputeRotation(vector);
            }
        }
    }

private:
    __aicore__ inline void ComputeRotation(const uint32_t vector)
    {
        const AscendC::LocalTensor<scalar_t> queryIn = queryInBuf_.Get<scalar_t>();
        const AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        const AscendC::LocalTensor<float> x = work;
        const AscendC::LocalTensor<float> tmp = work[headSize_];
        const AscendC::LocalTensor<float> signs = signBuf_.Get<float>();

        AscendC::DataCopy(queryIn, queryGm_[static_cast<uint64_t>(vector) * headSize_], headSize_);
        SyncEvent<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(x, queryIn, AscendC::RoundMode::CAST_NONE, headSize_);

        codec_.ApplyPi(x, tmp, signs, headSize_);

        SyncVectorToMte3();
        AscendC::DataCopy(queryRotGm_[static_cast<uint64_t>(vector) * headSize_], x, headSize_);
        SyncEvent<AscendC::HardEvent::MTE3_MTE2>();
        SyncEvent<AscendC::HardEvent::MTE3_V>();
    }

    AscendC::TPipe *pipe_;
    TurboQuantCodec4 codec_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> queryInBuf_;
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

#define ASCEND_TQ_DECLARE_ROTATE_Q_CUBE(TYPE)                                                                        \
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

#define ASCEND_TQ_DECLARE_ROTATE_Q_AIV(TYPE)                                                                         \
    extern "C" __global__ __aicore__ void turboquant_rotate_q_aiv_##TYPE(                                             \
        GM_ADDR query, GM_ADDR piSigns, GM_ADDR rotTables, GM_ADDR queryRot, uint32_t numVectors,                    \
        uint32_t headSize, uint32_t vectorsPerBlock, float invSqrtLen)                                                \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantRotateQAiv<TYPE> op(&pipe);                                                                        \
        op.Init(query, piSigns, rotTables, queryRot, numVectors, headSize, vectorsPerBlock, invSqrtLen);             \
        op.Process();                                                                                                \
    }

ASCEND_TQ_DECLARE_ROTATE_Q_CUBE(half)
ASCEND_TQ_DECLARE_ROTATE_Q_AIV(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
ASCEND_TQ_DECLARE_ROTATE_Q_CUBE(bfloat16_t)
ASCEND_TQ_DECLARE_ROTATE_Q_AIV(bfloat16_t)
#endif

#undef ASCEND_TQ_DECLARE_ROTATE_Q_AIV
#undef ASCEND_TQ_DECLARE_ROTATE_Q_CUBE

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
