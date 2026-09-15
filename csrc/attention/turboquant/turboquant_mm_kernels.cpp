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
#include "turboquant_codec_mx.h"
#include "turboquant_cube_mm.h"

#include "../../kernels/types.h"

using vllm_ascend::turboquant::DecodeAblationStage;
using vllm_ascend::turboquant::kBrcbDstLanes;
using vllm_ascend::turboquant::kFp32PerBlock;
using vllm_ascend::turboquant::kFp32PerRepeat;
using vllm_ascend::turboquant::kSlots;
using vllm_ascend::turboquant::TurboQuantCodec4;
using vllm_ascend::turboquant::TurboQuantCubeMm;
using vllm_ascend::turboquant::TurboQuantMode;
using vllm_ascend::turboquant::TurboQuantModeCodec;
using vllm_ascend::turboquant::TurboQuantModeTraits;

namespace {

constexpr uint32_t kCubeTileRows = 64;

constexpr uint32_t kUnpackRows = 8;

constexpr uint32_t kUnpackGroups = 2;

constexpr uint32_t kOperandC0 = 32;

constexpr uint32_t kMaxGroupHeads = vllm_ascend::turboquant::kCubeTileM;

constexpr uint32_t kPartialTail = 16;
constexpr uint32_t kPartialMaxLane = 0;
constexpr uint32_t kPartialSumLane = kFp32PerBlock;

constexpr float kNegInf = -3.4028235e38f;

constexpr uint32_t kUbBankPadBytes = 0;

__aicore__ inline uint32_t MixBlockIdx()
{
    return static_cast<uint32_t>(AscendC::GetBlockIdx() / AscendC::GetSubBlockNum());
}

__aicore__ inline void SignalOperandsReady()
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(vllm_ascend::turboquant::kFlagOperandsReady);
}

__aicore__ inline void SignalSlotReady(uint32_t slot)
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
        static_cast<uint16_t>(vllm_ascend::turboquant::kFlagSlotReady + slot));
}

__aicore__ inline void SignalProbsReady()
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(vllm_ascend::turboquant::kFlagProbsReady);
}

__aicore__ inline bool IsPrimarySubcore()
{
    return AscendC::GetSubBlockIdx() == 0;
}

__aicore__ inline void SyncVectorToMte3()
{
    const event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev);
}

__aicore__ inline void SyncMte2ToVector()
{
    const event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev);
}

__aicore__ inline void SyncMte3ToVector()
{
    const event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ev);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ev);
}

template <TurboQuantMode MODE>
__aicore__ inline constexpr float OperandMax()
{
    return TurboQuantModeTraits<MODE>::kOperand == vllm_ascend::turboquant::TurboQuantOperand::kFp4E2m1 ? 6.0f
                                                                                                       : 448.0f;
}

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__aicore__ inline uint32_t ScaleSlotFloats(uint32_t numKvHeads)
{
    return CeilDiv(2 * numKvHeads, kFp32PerBlock) * kFp32PerBlock;
}

__aicore__ inline void BroadcastSub(const AscendC::LocalTensor<float> &dst, const AscendC::LocalTensor<float> &src,
                                    const AscendC::LocalTensor<float> &scalarBlock, uint32_t count)
{
    constexpr uint8_t kRepBlocks = static_cast<uint8_t>(kFp32PerRepeat / kFp32PerBlock);
    const uint32_t repeats = count / kFp32PerRepeat;
    if (repeats > 0) {
        AscendC::Sub(dst, src, scalarBlock, static_cast<uint64_t>(kFp32PerRepeat), static_cast<uint8_t>(repeats),
                     {1, 1, 0, kRepBlocks, kRepBlocks, 0});
    }
    const uint32_t tail = count - repeats * kFp32PerRepeat;
    if (tail > 0) {
        const uint32_t base = repeats * kFp32PerRepeat;
        AscendC::Sub(dst[base], src[base], scalarBlock, static_cast<uint64_t>(tail), 1, {1, 1, 0, 0, 0, 0});
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void BroadcastScalar(const AscendC::LocalTensor<float> &dst, const AscendC::LocalTensor<float> &src)
{
    AscendC::Brcb(dst, src, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
    AscendC::PipeBarrier<PIPE_V>();
}

template <TurboQuantMode MODE, typename scalar_t>
class TurboQuantModeReshapeAndCache {
public:
    using Codec = TurboQuantModeCodec<MODE>;

    __aicore__ inline explicit TurboQuantModeReshapeAndCache(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *key, __gm__ void *value, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *scaleCache, __gm__ void *slotMapping, __gm__ void *piSigns,
                                __gm__ void *rotTables, __gm__ void *modeTables, uint32_t numTokens,
                                uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t tokensPerCore,
                                float invSqrtLen)
    {
        numTokens_ = numTokens;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        blockSize_ = blockSize;
        tokensPerCore_ = tokensPerCore;
        packedBytes_ = Codec::PackedBytes(headSize);
        headPlane_ = numKvHeads_ * headSize_;
        packedPlane_ = numKvHeads_ * packedBytes_;
        scaleSlot_ = ScaleSlotFloats(numKvHeads_);

        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(key));
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(value));
        keyCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(keyCache));
        valueCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(valueCache));
        scaleCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scaleCache));
        slotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(slotMapping), numTokens);
        piSignsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(piSigns), headSize);
        rotTablesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rotTables));
        modeTablesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(modeTables));

        pipe_->InitBuffer(inQueue_, 2, 2 * headPlane_ * sizeof(scalar_t));
        pipe_->InitBuffer(outPacked_, 2, 2 * packedPlane_ * sizeof(int8_t));
        pipe_->InitBuffer(outScale_, 2, scaleSlot_ * sizeof(float));
        pipe_->InitBuffer(workBuf_, 2 * headSize_ * sizeof(float));
        pipe_->InitBuffer(signBuf_, headSize_ * sizeof(float));
        pipe_->InitBuffer(stepBuf_, 2 * numKvHeads_ * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(scaleIdxBuf_, 2 * numKvHeads_ * sizeof(int32_t));

        rotation_.Init(pipe_, headSize_, 1, invSqrtLen, rotTablesGm_);
        codec_.Init(pipe_, headSize_, 1, invSqrtLen, modeTablesGm_);

        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::DataCopy(signs, piSignsGm_, headSize_);

        AscendC::LocalTensor<int32_t> gatherIdx = scaleIdxBuf_.Get<int32_t>();
        AscendC::ArithProgression(gatherIdx, 0, static_cast<int32_t>(kFp32PerBlock * sizeof(float)),
                                  static_cast<int32_t>(2 * numKvHeads_));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        const uint32_t start = static_cast<uint32_t>(AscendC::GetBlockIdx()) * tokensPerCore_;
        uint32_t end = start + tokensPerCore_;
        if (end > numTokens_) {
            end = numTokens_;
        }
        if (start >= end) {
            return;
        }
        const uint32_t total = end - start;

        CopyIn(start, 0);
        for (uint32_t i = 0; i < total; ++i) {
            if (i + 1 < total) {
                CopyIn(start + i + 1, i + 1);
            }
            Compute();
            if (i > 0) {
                CopyOut(i - 1);
            }
        }
        CopyOut(total - 1);
    }

private:
    static constexpr uint32_t kSlotRing = 4;

    __aicore__ inline void CopyIn(uint32_t token, uint32_t step)
    {
        const int32_t slot = slotGm_.GetValue(token);
        const uint32_t ring = step % kSlotRing;
        slotRing_[ring] = slot;
        AscendC::LocalTensor<scalar_t> in = inQueue_.template AllocTensor<scalar_t>();
        if (slot >= 0) {
            const uint64_t base = static_cast<uint64_t>(token) * headPlane_;
            AscendC::DataCopy(in, keyGm_[base], headPlane_);
            AscendC::DataCopy(in[headPlane_], valueGm_[base], headPlane_);
        }
        inQueue_.EnQue(in);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<scalar_t> in = inQueue_.template DeQue<scalar_t>();
        AscendC::LocalTensor<int8_t> packed = outPacked_.template AllocTensor<int8_t>();
        AscendC::LocalTensor<float> scaleOut = outScale_.template AllocTensor<float>();

        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        AscendC::LocalTensor<float> vec = work;
        AscendC::LocalTensor<float> tmp = work[headSize_];
        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::LocalTensor<float> steps = stepBuf_.Get<float>();

        for (uint32_t plane = 0; plane < 2; ++plane) {
            for (uint32_t head = 0; head < numKvHeads_; ++head) {
                const uint32_t src = plane * headPlane_ + head * headSize_;
                AscendC::Cast(vec, in[src], AscendC::RoundMode::CAST_NONE, headSize_);
                AscendC::PipeBarrier<PIPE_V>();
                rotation_.ApplyPi(vec, tmp, signs, static_cast<int>(headSize_));
                codec_.Encode(packed[plane * packedPlane_ + head * packedBytes_], vec,
                              steps[(plane * numKvHeads_ + head) * kFp32PerBlock], static_cast<int>(headSize_));
            }
        }
        inQueue_.FreeTensor(in);

        AscendC::LocalTensor<uint32_t> idx = scaleIdxBuf_.Get<int32_t>().ReinterpretCast<uint32_t>();
        AscendC::Duplicate(scaleOut, 0.0f, scaleSlot_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(scaleOut, steps, idx, vllm_ascend::turboquant::kGatherSrcBase, 2 * numKvHeads_);
        AscendC::PipeBarrier<PIPE_V>();

        outPacked_.EnQue(packed);
        outScale_.EnQue(scaleOut);
    }

    __aicore__ inline void CopyOut(uint32_t step)
    {
        AscendC::LocalTensor<int8_t> packed = outPacked_.template DeQue<int8_t>();
        AscendC::LocalTensor<float> scaleOut = outScale_.template DeQue<float>();
        const int32_t slot = slotRing_[step % kSlotRing];
        if (slot >= 0) {
            const uint64_t row = static_cast<uint64_t>(slot);
            AscendC::DataCopy(keyCacheGm_[row * packedPlane_], packed, packedPlane_);
            AscendC::DataCopy(valueCacheGm_[row * packedPlane_], packed[packedPlane_], packedPlane_);
            AscendC::DataCopy(scaleCacheGm_[row * scaleSlot_], scaleOut, scaleSlot_);
        }
        outPacked_.FreeTensor(packed);
        outScale_.FreeTensor(scaleOut);
    }

    AscendC::TPipe *pipe_;
    TurboQuantCodec4 rotation_;
    Codec codec_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outPacked_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outScale_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> workBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> signBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stepBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleIdxBuf_;
    AscendC::GlobalTensor<scalar_t> keyGm_;
    AscendC::GlobalTensor<scalar_t> valueGm_;
    AscendC::GlobalTensor<int8_t> keyCacheGm_;
    AscendC::GlobalTensor<int8_t> valueCacheGm_;
    AscendC::GlobalTensor<float> scaleCacheGm_;
    AscendC::GlobalTensor<int32_t> slotGm_;
    AscendC::GlobalTensor<float> piSignsGm_;
    AscendC::GlobalTensor<int32_t> rotTablesGm_;
    AscendC::GlobalTensor<int32_t> modeTablesGm_;
    int32_t slotRing_[kSlotRing] = {-1, -1, -1, -1};
    uint32_t numTokens_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t tokensPerCore_ = 0;
    uint32_t packedBytes_ = 0;
    uint32_t headPlane_ = 0;
    uint32_t packedPlane_ = 0;
    uint32_t scaleSlot_ = 0;
};

template <TurboQuantMode MODE, typename scalar_t,
          DecodeAblationStage STAGE = DecodeAblationStage::STAGE_5_FULL_PIPELINE>
class TurboQuantCubeDecodeSplit {
public:
    using Codec = TurboQuantModeCodec<MODE>;
    using Mm = TurboQuantCubeMm<MODE>;
    using OperandT = typename Mm::OperandT;

    __aicore__ static constexpr bool Keeps(DecodeAblationStage stage)
    {
        return static_cast<int32_t>(STAGE) >= static_cast<int32_t>(stage);
    }

    __aicore__ inline explicit TurboQuantCubeDecodeSplit(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *queryRot, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *scaleCache, __gm__ void *blockTables, __gm__ void *contextLens,
                                __gm__ void *modeTables,
                                __gm__ void *workspace, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,
                                float scale, float invSqrtLen)
    {
        numTokens_ = numTokens;
        numHeads_ = numHeads;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        blockSize_ = blockSize;
        maxBlocksPerSeq_ = maxBlocksPerSeq;
        numSplits_ = numSplits;
        scale_ = scale;
        groupHeads_ = numHeads / numKvHeads;
        packedBytes_ = Codec::PackedBytes(headSize);
        packedPlane_ = numKvHeads_ * packedBytes_;
        scaleSlot_ = ScaleSlotFloats(numKvHeads_);
        partialStride_ = headSize + kPartialTail;
        operandElems_ = Mm::OperandElems(headSize_);
        packedGroups_ = packedBytes_ / kOperandC0;
        unpackGroups_ = packedGroups_ < kUnpackGroups ? packedGroups_ : kUnpackGroups;
        while (unpackGroups_ > 1 && (packedGroups_ % unpackGroups_) != 0) {
            --unpackGroups_;
        }
        unpackBytes_ = unpackGroups_ * kCubeTileRows * kOperandC0;

        if constexpr (Codec::kIsAffine) {
            tileParams_ = AscendC::DataCopyParams{static_cast<uint16_t>(kCubeTileRows), 1,
                                                  static_cast<uint16_t>((packedPlane_ - kOperandC0) / 32), 0};
            unpackParams_ = AscendC::DataCopyParams{1, static_cast<uint16_t>(unpackBytes_ / 32), 0, 0};
        } else {
            tileParams_ = AscendC::DataCopyParams{static_cast<uint16_t>(kCubeTileRows),
                                                  static_cast<uint16_t>(packedBytes_ / 32),
                                                  static_cast<uint16_t>((packedPlane_ - packedBytes_) / 32), 0};
            unpackParams_ = AscendC::DataCopyParams{
                static_cast<uint16_t>(operandElems_ / kOperandC0),
                static_cast<uint16_t>(kUnpackRows * kOperandC0 / 32), 0,
                static_cast<uint16_t>((kCubeTileRows - kUnpackRows) * kOperandC0 / 32)};
        }
        qNzParams_ = AscendC::DataCopyParams{static_cast<uint16_t>(headSize_ / kOperandC0), 1, 0,
                                             static_cast<uint16_t>(kMaxGroupHeads - 1)};
        probNzParams_ =
            AscendC::DataCopyParams{static_cast<uint16_t>(Mm::OperandElems(kCubeTileRows) / kOperandC0), 1, 0,
                                    static_cast<uint16_t>(kMaxGroupHeads - 1)};
        const uint8_t accRowBlocks = static_cast<uint8_t>(headSize_ / kFp32PerBlock);
        accRescaleParams_ = AscendC::BinaryRepeatParams{1, 1, 0, accRowBlocks, accRowBlocks, 1};
        scoreScale_ = scale_ / TurboQuantModeTraits<MODE>::kGain;

        queryRotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(queryRot));
        keyCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(keyCache));
        valueCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(valueCache));
        scaleCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scaleCache));
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTables));
        contextLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(contextLens), numTokens);
        modeTablesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(modeTables));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

        mm_.Init(pipe_, headSize_, kCubeTileRows);

        pipe_->InitBuffer(kvBuf_, 2 * kCubeTileRows * packedBytes_);
        pipe_->InitBuffer(scaleTileBuf_, kCubeTileRows * scaleSlot_ * sizeof(float));
        if constexpr (Codec::kIsAffine) {
            pipe_->InitBuffer(operandBuf_, 2 * unpackBytes_);
        } else {
            pipe_->InitBuffer(operandBuf_, kUnpackRows * operandElems_);
        }
        pipe_->InitBuffer(accBuf_, kMaxGroupHeads * headSize_ * sizeof(float));
        pipe_->InitBuffer(qBuf_, 2 * headSize_ * sizeof(float));
        pipe_->InitBuffer(qOperandBuf_, kMaxGroupHeads * operandElems_);
        pipe_->InitBuffer(scoreBuf_, kMaxGroupHeads * kCubeTileRows * sizeof(float));
        pipe_->InitBuffer(probOperandBuf_, kMaxGroupHeads * Mm::OperandElems(kCubeTileRows) + kUbBankPadBytes);
        pipe_->InitBuffer(ctxBuf_, kMaxGroupHeads * headSize_ * sizeof(float));
        pipe_->InitBuffer(stateBuf_, 6 * kMaxGroupHeads * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, 4 * kMaxGroupHeads * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(scaleIdxBuf_, 2 * kCubeTileRows * sizeof(int32_t));
        pipe_->InitBuffer(qInBuf_, groupHeads_ * headSize_ * sizeof(float));

        codec_.Init(pipe_, headSize_,
                    Codec::kIsAffine ? (2 * unpackBytes_ / headSize_) : kUnpackRows, invSqrtLen, modeTablesGm_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process(uint32_t tasksPerCore)
    {
        const uint32_t tasks = numTokens_ * numKvHeads_ * numSplits_;
        uint32_t start = MixBlockIdx() * tasksPerCore;
        uint32_t end = start + tasksPerCore;
        if (end > tasks) {
            end = tasks;
        }
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t split = task % numSplits_;
            const uint32_t rest = task / numSplits_;
            ComputeSplit(rest / numKvHeads_, rest % numKvHeads_, split);
        }
    }

private:
    __aicore__ inline uint64_t PartialOffset(uint32_t token, uint32_t head, uint32_t split) const
    {
        return ((static_cast<uint64_t>(token) * numHeads_ + head) * numSplits_ + split) * partialStride_;
    }

    __aicore__ inline void ComputeSplit(uint32_t token, uint32_t kvHead, uint32_t split)
    {
        AscendC::LocalTensor<float> acc = accBuf_.Get<float>();
        AscendC::LocalTensor<float> state = stateBuf_.Get<float>();
        AscendC::LocalTensor<float> runMax = state;
        AscendC::LocalTensor<float> runSum = state[kMaxGroupHeads * kFp32PerBlock];

        if constexpr (Keeps(DecodeAblationStage::STAGE_5_FULL_PIPELINE)) {
          if ASCEND_IS_AIV {
            AscendC::Duplicate(acc, 0.0f, groupHeads_ * headSize_);
            AscendC::Duplicate(state, 0.0f, 6 * kMaxGroupHeads * kFp32PerBlock);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(runMax, kNegInf, groupHeads_ * kFp32PerBlock);
            AscendC::PipeBarrier<PIPE_V>();
          }
        }

        const int32_t contextLen = contextLenGm_.GetValue(token);
        const uint32_t seqBlocks = contextLen > 0 ? CeilDiv(static_cast<uint32_t>(contextLen), blockSize_) : 0;
        const uint32_t blocksPerSplit = CeilDiv(seqBlocks, numSplits_);
        const uint32_t blockStart = split * blocksPerSplit;
        uint32_t blockEnd = blockStart + blocksPerSplit;
        if (blockEnd > seqBlocks) {
            blockEnd = seqBlocks;
        }

        if (blockStart < blockEnd) {
            const uint32_t ctxLen = static_cast<uint32_t>(contextLen);
            const uint32_t numTiles = CountTiles(token, ctxLen, blockStart, blockEnd);
            if ASCEND_IS_AIV {
                if constexpr (Keeps(DecodeAblationStage::STAGE_2_QUERY_PREP)) {
                    PrepareTask(token, kvHead);
                }
                PipelineAiv(token, ctxLen, blockStart, blockEnd, numTiles, kvHead, acc, state);
            }
            if ASCEND_IS_AIC {
                if constexpr (Keeps(DecodeAblationStage::STAGE_4_SCORE_GEMM)) {
                    PipelineAic(numTiles);
                }
            }
        }

        if constexpr (!Keeps(DecodeAblationStage::STAGE_5_FULL_PIPELINE)) {
            return;
        }

        if ASCEND_IS_AIV {
          if (IsPrimarySubcore()) {
            AscendC::LocalTensor<float> tails = scoreBuf_.Get<float>();
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t h = 0; h < groupHeads_; ++h) {
                AscendC::LocalTensor<float> tail = tails[h * kPartialTail];
                AscendC::Duplicate(tail, 0.0f, kPartialTail);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(tail[kPartialMaxLane], runMax[h * kFp32PerBlock], 0.0f, 1);
                AscendC::Adds(tail[kPartialSumLane], runSum[h * kFp32PerBlock], 0.0f, 1);
            }
            SyncVectorToMte3();
            for (uint32_t h = 0; h < groupHeads_; ++h) {
                const uint32_t head = kvHead * groupHeads_ + h;
                const uint64_t offset = PartialOffset(token, head, split);
                AscendC::DataCopy(workspaceGm_[offset], acc[h * headSize_], headSize_);
                AscendC::DataCopy(workspaceGm_[offset + headSize_], tails[h * kPartialTail], kPartialTail);
            }
          }
        }
    }

    __aicore__ inline void PrepareTask(uint32_t token, uint32_t kvHead)
    {
        AscendC::LocalTensor<float> work = qBuf_.Get<float>();
        AscendC::LocalTensor<float> tmp = work[headSize_];
        AscendC::LocalTensor<OperandT> qOperand = qOperandBuf_.Get<OperandT>();
        AscendC::LocalTensor<float> reduce = reduceBuf_.Get<float>();
        AscendC::LocalTensor<OperandT> qL1 = mm_.A1Query();

        AscendC::Duplicate(qOperand.template ReinterpretCast<int8_t>(), static_cast<int8_t>(0),
                           groupHeads_ * operandElems_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> qIn = qInBuf_.Get<float>();
        AscendC::DataCopy(qIn,
                          queryRotGm_[(static_cast<uint64_t>(token) * numHeads_ +
                                       static_cast<uint64_t>(kvHead) * groupHeads_) * headSize_],
                          groupHeads_ * headSize_);
        SyncMte2ToVector();

        for (uint32_t h = 0; h < groupHeads_; ++h) {
            AscendC::LocalTensor<float> vec = qIn[h * headSize_];

            AscendC::Abs(tmp, vec, headSize_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax<float>(reduce, tmp, scoreBuf_.Get<float>(), headSize_, false);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(reduce, reduce, TurboQuantCodec4::kEps, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(reduce[kFp32PerBlock], OperandMax<MODE>(), 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(reduce[kFp32PerBlock], reduce[kFp32PerBlock], reduce, 1);
            AscendC::PipeBarrier<PIPE_V>();
            BroadcastScalar(reduce[2 * kFp32PerBlock], reduce[kFp32PerBlock]);
            TurboQuantCodec4::BroadcastMul(vec, vec, reduce[2 * kFp32PerBlock], headSize_);
            if constexpr (Keeps(DecodeAblationStage::STAGE_5_FULL_PIPELINE)) {
                qScaleInv_[h] = 1.0f / reduce.GetValue(kFp32PerBlock);
            }

            codec_.CastToOperand(qOperand[h * operandElems_], vec, headSize_);

            if constexpr (Keeps(DecodeAblationStage::STAGE_3_L1_STAGING)) {
                SyncVectorToMte3();
                AscendC::DataCopy(qL1[h * kOperandC0], qOperand[h * operandElems_], qNzParams_);
            }
        }

        if constexpr (!Keeps(DecodeAblationStage::STAGE_5_FULL_PIPELINE)) {
            return;
        }

        const int32_t slotBytes = static_cast<int32_t>(scaleSlot_ * sizeof(float));
        AscendC::LocalTensor<int32_t> idx = scaleIdxBuf_.Get<int32_t>();
        AscendC::ArithProgression(idx, static_cast<int32_t>(kvHead * sizeof(float)), slotBytes,
                                  static_cast<int32_t>(kCubeTileRows));
        AscendC::ArithProgression(idx[kCubeTileRows], static_cast<int32_t>((numKvHeads_ + kvHead) * sizeof(float)),
                                  slotBytes, static_cast<int32_t>(kCubeTileRows));
        AscendC::PipeBarrier<PIPE_V>();
    }

    struct TileCursor {
        uint32_t block = 0;
        uint32_t rows = 0;
        uint32_t base = 0;
        uint32_t physical = 0;
        uint32_t valid = 0;
        bool active = false;
    };

    __aicore__ inline uint32_t CountTiles(uint32_t token, uint32_t contextLen, uint32_t blockStart,
                                          uint32_t blockEnd)
    {
        uint32_t tiles = 0;
        for (uint32_t block = blockStart; block < blockEnd; ++block) {
            const int32_t physical =
                blockTableGm_.GetValue(static_cast<uint64_t>(token) * maxBlocksPerSeq_ + block);
            if (physical < 0) {
                continue;
            }
            uint32_t rows = blockSize_;
            const uint32_t consumed = block * blockSize_;
            if (consumed + rows > contextLen) {
                rows = contextLen - consumed;
            }
            tiles += CeilDiv(rows, kCubeTileRows);
        }
        return tiles;
    }

    __aicore__ inline bool NextTile(TileCursor &cursor, uint32_t token, uint32_t contextLen, uint32_t blockEnd)
    {
        if (cursor.active) {
            cursor.base += kCubeTileRows;
            if (cursor.base < cursor.rows) {
                cursor.valid = cursor.rows - cursor.base;
                if (cursor.valid > kCubeTileRows) {
                    cursor.valid = kCubeTileRows;
                }
                return true;
            }
            cursor.active = false;
            ++cursor.block;
        }
        while (cursor.block < blockEnd) {
            const int32_t physical =
                blockTableGm_.GetValue(static_cast<uint64_t>(token) * maxBlocksPerSeq_ + cursor.block);
            if (physical >= 0) {
                uint32_t rows = blockSize_;
                const uint32_t consumed = cursor.block * blockSize_;
                if (consumed + rows > contextLen) {
                    rows = contextLen - consumed;
                }
                if (rows > 0) {
                    cursor.physical = static_cast<uint32_t>(physical);
                    cursor.rows = rows;
                    cursor.base = 0;
                    cursor.valid = rows < kCubeTileRows ? rows : kCubeTileRows;
                    cursor.active = true;
                    return true;
                }
            }
            ++cursor.block;
        }
        return false;
    }

    __aicore__ inline void StageTile(const TileCursor &tile, uint32_t kvHead, uint32_t l1SlotIdx)
    {
        AscendC::LocalTensor<int8_t> packedTile = kvBuf_.Get<int8_t>();
        AscendC::LocalTensor<float> scaleTile = scaleTileBuf_.Get<float>();
        CopyInTile(packedTile, scaleTile, tile.physical, kvHead, tile.base);
        if constexpr (!Keeps(DecodeAblationStage::STAGE_1_UNPACK)) {
            return;
        }
        UnpackToL1(packedTile, mm_.B1K(l1SlotIdx));
        UnpackToL1(packedTile[kCubeTileRows * packedBytes_], mm_.B1V(l1SlotIdx));
    }

    __aicore__ inline void PipelineAiv(uint32_t token, uint32_t contextLen, uint32_t blockStart, uint32_t blockEnd,
                                       uint32_t numTiles, uint32_t kvHead, const AscendC::LocalTensor<float> &acc,
                                       const AscendC::LocalTensor<float> &state)
    {
        if (numTiles == 0) {
            return;
        }
        AscendC::LocalTensor<float> scaleTile = scaleTileBuf_.Get<float>();
        AscendC::LocalTensor<float> scores = scoreBuf_.Get<float>();
        AscendC::LocalTensor<float> ctx = ctxBuf_.Get<float>();

        TileCursor stageCursor;
        TileCursor consumeCursor;
        stageCursor.block = blockStart;
        consumeCursor.block = blockStart;

        NextTile(stageCursor, token, contextLen, blockEnd);
        StageTile(stageCursor, kvHead, 0);
        if constexpr (Keeps(DecodeAblationStage::STAGE_4_SCORE_GEMM)) {
            SignalSlotReady(0);
        }

        for (uint32_t tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
            const uint32_t nextSlotIdx = (tileIdx + 1) % kSlots;
            if constexpr (Keeps(DecodeAblationStage::STAGE_5_FULL_PIPELINE)) {
                NextTile(consumeCursor, token, contextLen, blockEnd);
            }

            if constexpr (Keeps(DecodeAblationStage::STAGE_4_SCORE_GEMM)) {
                AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagScoresReady);
            }
            if constexpr (Keeps(DecodeAblationStage::STAGE_5_FULL_PIPELINE)) {
                if (IsPrimarySubcore()) {
                    SoftmaxStageProbs(scores, scaleTile, state, consumeCursor.valid);
                }
                SignalProbsReady();
                if (IsPrimarySubcore()) {
                    SoftmaxRescaleAcc(state, acc);
                }
            }

            if (tileIdx + 1 < numTiles) {
                if constexpr (Keeps(DecodeAblationStage::STAGE_4_SCORE_GEMM)) {
                    if (tileIdx + 1 >= kSlots) {
                        AscendC::CrossCoreWaitFlag(
                            static_cast<uint16_t>(vllm_ascend::turboquant::kFlagSlotFree + nextSlotIdx));
                    }
                }
                NextTile(stageCursor, token, contextLen, blockEnd);
                StageTile(stageCursor, kvHead, nextSlotIdx);
                if constexpr (Keeps(DecodeAblationStage::STAGE_4_SCORE_GEMM)) {
                    SignalSlotReady(nextSlotIdx);
                }
            }

            if constexpr (Keeps(DecodeAblationStage::STAGE_5_FULL_PIPELINE)) {
                AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagContextReady);
                if (IsPrimarySubcore()) {
                    Accumulate(acc, ctx, state);
                }
            }
        }
    }

    __aicore__ inline void PipelineAic(uint32_t numTiles)
    {
        AscendC::LocalTensor<float> scores = scoreBuf_.Get<float>();
        AscendC::LocalTensor<float> ctx = ctxBuf_.Get<float>();

        for (uint32_t tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
            const uint32_t l1SlotIdx = tileIdx % kSlots;

            AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(vllm_ascend::turboquant::kFlagSlotReady + l1SlotIdx));
            mm_.GemmScores(scores, mm_.B1K(l1SlotIdx), kMaxGroupHeads, headSize_, kCubeTileRows);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(vllm_ascend::turboquant::kFlagScoresReady);

            if constexpr (Keeps(DecodeAblationStage::STAGE_5_FULL_PIPELINE)) {
                AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagProbsReady);
                mm_.GemmContext(ctx, mm_.B1V(l1SlotIdx), kMaxGroupHeads, kCubeTileRows, headSize_);
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(vllm_ascend::turboquant::kFlagContextReady);
            }

            if (tileIdx + kSlots < numTiles) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE1>(
                    static_cast<uint16_t>(vllm_ascend::turboquant::kFlagSlotFree + l1SlotIdx));
            }
        }
    }

    __aicore__ inline void CopyInTile(const AscendC::LocalTensor<int8_t> &kv,
                                      const AscendC::LocalTensor<float> &scales, uint32_t physical, uint32_t kvHead,
                                      uint32_t rowBase)
    {
        if ASCEND_IS_AIC {
            return;
        }
        const uint64_t row = static_cast<uint64_t>(physical) * blockSize_ + rowBase;
        const uint64_t cacheOff = row * packedPlane_ + static_cast<uint64_t>(kvHead) * packedBytes_;
        const uint64_t scaleOff = row * scaleSlot_;

        if constexpr (Codec::kIsAffine) {
            const uint32_t groupElems = kCubeTileRows * kOperandC0;
            for (uint32_t groupIdx = 0; groupIdx < packedGroups_; ++groupIdx) {
                AscendC::DataCopy(kv[groupIdx * groupElems],
                                  keyCacheGm_[cacheOff + groupIdx * kOperandC0], tileParams_);
                AscendC::DataCopy(kv[kCubeTileRows * packedBytes_ + groupIdx * groupElems],
                                  valueCacheGm_[cacheOff + groupIdx * kOperandC0], tileParams_);
            }
        } else {
            AscendC::DataCopy(kv, keyCacheGm_[cacheOff], tileParams_);
            AscendC::DataCopy(kv[kCubeTileRows * packedBytes_], valueCacheGm_[cacheOff], tileParams_);
        }
        AscendC::DataCopy(scales, scaleCacheGm_[scaleOff], kCubeTileRows * scaleSlot_);
        if constexpr (Keeps(DecodeAblationStage::STAGE_1_UNPACK)) {
            SyncMte2ToVector();
        } else {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void UnpackToL1(const AscendC::LocalTensor<int8_t> &packed,
                                      const AscendC::LocalTensor<OperandT> &l1Dst)
    {
        if ASCEND_IS_AIC {
            return;
        }
        AscendC::LocalTensor<OperandT> unpackedUb = operandBuf_.Get<OperandT>();

        if constexpr (Codec::kIsAffine) {
            const uint32_t groupElems = kCubeTileRows * kOperandC0;
            for (uint32_t groupBase = 0; groupBase < packedGroups_; groupBase += unpackGroups_) {
                vllm_ascend::turboquant::unpack_tq4_to_fp8(codec_, unpackedUb, unpackedUb[unpackBytes_],
                                                           packed[groupBase * groupElems], unpackBytes_);
                if constexpr (Keeps(DecodeAblationStage::STAGE_3_L1_STAGING)) {
                    SyncVectorToMte3();
                    AscendC::DataCopy(l1Dst[groupBase * groupElems], unpackedUb, unpackParams_);
                    AscendC::DataCopy(l1Dst[(packedGroups_ + groupBase) * groupElems],
                                      unpackedUb[unpackBytes_], unpackParams_);
                }
            }
        } else {
            const uint32_t bandElems = kUnpackRows * kOperandC0;
            for (uint32_t bandIdx = 0; bandIdx < kCubeTileRows / kUnpackRows; ++bandIdx) {
                const AscendC::LocalTensor<int8_t> sub_packed = packed[bandIdx * kUnpackRows * packedBytes_];
                if constexpr (MODE == TurboQuantMode::KV5_FP8) {
                    vllm_ascend::turboquant::unpack_tq5_to_fp8(codec_, unpackedUb, sub_packed,
                                                               static_cast<int>(kUnpackRows),
                                                               static_cast<int>(headSize_));
                } else {
                    codec_.Unpack(unpackedUb, sub_packed, static_cast<int>(kUnpackRows),
                                  static_cast<int>(headSize_));
                }
                if constexpr (Keeps(DecodeAblationStage::STAGE_3_L1_STAGING)) {
                    SyncVectorToMte3();
                    AscendC::DataCopy(l1Dst[bandIdx * bandElems], unpackedUb, unpackParams_);
                }
            }
        }
        if constexpr (Keeps(DecodeAblationStage::STAGE_3_L1_STAGING)) {
            SyncMte3ToVector();
        } else {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void SoftmaxStageProbs(const AscendC::LocalTensor<float> &scores,
                                             const AscendC::LocalTensor<float> &scaleTile,
                                             const AscendC::LocalTensor<float> &state, uint32_t valid)
    {
        AscendC::LocalTensor<float> runMax = state;
        AscendC::LocalTensor<float> runSum = state[kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> tileMax = state[2 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> newMax = state[3 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> alpha = state[4 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> probScale = state[5 * kMaxGroupHeads * kFp32PerBlock];

        AscendC::LocalTensor<float> reduce = reduceBuf_.Get<float>();
        AscendC::LocalTensor<float> kScale = reduce;
        AscendC::LocalTensor<float> vScale = reduce[kCubeTileRows];
        AscendC::LocalTensor<float> reduceWork = ctxBuf_.Get<float>();
        AscendC::LocalTensor<OperandT> probOperand = probOperandBuf_.Get<OperandT>();
        AscendC::LocalTensor<float> brcb = reduce[2 * kCubeTileRows];
        AscendC::LocalTensor<float> part = reduce[3 * kCubeTileRows];

        AscendC::LocalTensor<uint32_t> idx = scaleIdxBuf_.Get<int32_t>().ReinterpretCast<uint32_t>();
        AscendC::Gather(kScale, scaleTile, idx, vllm_ascend::turboquant::kGatherSrcBase, kCubeTileRows);
        AscendC::Gather(vScale, scaleTile, idx[kCubeTileRows], vllm_ascend::turboquant::kGatherSrcBase,
                        kCubeTileRows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(kScale, kScale, scoreScale_, kCubeTileRows);
        AscendC::Muls(vScale, vScale, 1.0f / TurboQuantModeTraits<MODE>::kGain, kCubeTileRows);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t h = 0; h < groupHeads_; ++h) {
            AscendC::LocalTensor<float> row = scores[h * kCubeTileRows];
            AscendC::Mul(row, row, kScale, kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(row, row, qScaleInv_[h], kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            if (valid < kCubeTileRows) {
                AscendC::Duplicate(row[valid], kNegInf, kCubeTileRows - valid);
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::ReduceMax<float>(tileMax[h * kFp32PerBlock], row, reduceWork, kCubeTileRows, false);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(newMax[h * kFp32PerBlock], runMax[h * kFp32PerBlock], tileMax[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(alpha[h * kFp32PerBlock], runMax[h * kFp32PerBlock], newMax[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(alpha[h * kFp32PerBlock], alpha[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();

            BroadcastScalar(brcb, newMax[h * kFp32PerBlock]);
            BroadcastSub(row, row, brcb, kCubeTileRows);
            AscendC::Exp(row, row, kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            if (valid < kCubeTileRows) {
                AscendC::Duplicate(row[valid], 0.0f, kCubeTileRows - valid);
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::ReduceSum<float>(part, row, part[kFp32PerBlock], kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(runSum[h * kFp32PerBlock], runSum[h * kFp32PerBlock], alpha[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(runSum[h * kFp32PerBlock], runSum[h * kFp32PerBlock], part, 1);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Mul(row, row, vScale, kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax<float>(part, row, part[kFp32PerBlock], kCubeTileRows, false);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(part, part, TurboQuantCodec4::kEps, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(probScale[h * kFp32PerBlock], OperandMax<MODE>(), 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(probScale[h * kFp32PerBlock], probScale[h * kFp32PerBlock], part, 1);
            AscendC::PipeBarrier<PIPE_V>();
            BroadcastScalar(brcb, probScale[h * kFp32PerBlock]);
            TurboQuantCodec4::BroadcastMul(row, row, brcb, kCubeTileRows);

            codec_.CastToOperand(probOperand[h * Mm::OperandElems(kCubeTileRows)], row, kCubeTileRows);
        }

        AscendC::LocalTensor<OperandT> pL1 = mm_.A1Probs();
        const uint32_t pElems = Mm::OperandElems(kCubeTileRows);
        for (uint32_t h = 0; h < groupHeads_; ++h) {
            SyncVectorToMte3();
            AscendC::DataCopy(pL1[h * kOperandC0], probOperand[h * pElems], probNzParams_);
        }
        SyncMte3ToVector();
    }

    __aicore__ inline void SoftmaxRescaleAcc(const AscendC::LocalTensor<float> &state,
                                             const AscendC::LocalTensor<float> &acc)
    {
        AscendC::LocalTensor<float> runMax = state;
        AscendC::LocalTensor<float> newMax = state[3 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> alpha = state[4 * kMaxGroupHeads * kFp32PerBlock];

        AscendC::LocalTensor<float> alphaBlocks = reduceBuf_.Get<float>()[3 * kCubeTileRows + 2 * kFp32PerBlock];
        for (uint32_t h = 0; h < groupHeads_; ++h) {
            AscendC::Brcb(alphaBlocks[h * kFp32PerBlock], alpha[h * kFp32PerBlock], 1,
                          {1, static_cast<uint16_t>(kFp32PerBlock)});
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t col = 0; col < headSize_; col += kFp32PerRepeat) {
            AscendC::Mul(acc[col], acc[col], alphaBlocks, static_cast<uint64_t>(kFp32PerRepeat),
                         static_cast<uint8_t>(groupHeads_), accRescaleParams_);
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Adds(runMax, newMax, 0.0f, groupHeads_ * kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Accumulate(const AscendC::LocalTensor<float> &acc, const AscendC::LocalTensor<float> &ctx,
                                      const AscendC::LocalTensor<float> &state)
    {
        AscendC::LocalTensor<float> probScale = state[5 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> reduce = reduceBuf_.Get<float>();
        AscendC::LocalTensor<float> inv = reduce[3 * kCubeTileRows];
        AscendC::LocalTensor<float> block = reduce[3 * kCubeTileRows + kFp32PerBlock];
        for (uint32_t h = 0; h < groupHeads_; ++h) {
            BroadcastScalar(block, probScale[h * kFp32PerBlock]);
            AscendC::Duplicate(inv, 1.0f, kFp32PerBlock);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(inv, inv, block, kFp32PerBlock);
            AscendC::PipeBarrier<PIPE_V>();
            TurboQuantCodec4::BroadcastMul(ctx[h * headSize_], ctx[h * headSize_], inv, headSize_);
        }
        AscendC::Add(acc, acc, ctx, groupHeads_ * headSize_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::TPipe *pipe_;
    Codec codec_;
    Mm mm_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> qInBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> kvBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleTileBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> operandBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> qBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> qOperandBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scoreBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> probOperandBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> ctxBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stateBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleIdxBuf_;
    AscendC::GlobalTensor<float> queryRotGm_;
    AscendC::GlobalTensor<int8_t> keyCacheGm_;
    AscendC::GlobalTensor<int8_t> valueCacheGm_;
    AscendC::GlobalTensor<float> scaleCacheGm_;
    AscendC::GlobalTensor<int32_t> blockTableGm_;
    AscendC::GlobalTensor<int32_t> contextLenGm_;
    AscendC::GlobalTensor<int32_t> modeTablesGm_;
    AscendC::GlobalTensor<float> workspaceGm_;
    AscendC::DataCopyParams tileParams_;
    AscendC::DataCopyParams unpackParams_;
    AscendC::DataCopyParams qNzParams_;
    AscendC::DataCopyParams probNzParams_;
    AscendC::BinaryRepeatParams accRescaleParams_;
    float scoreScale_ = 0.0f;
    float qScaleInv_[kMaxGroupHeads] = {};
    uint32_t numTokens_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t groupHeads_ = 1;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlocksPerSeq_ = 0;
    uint32_t numSplits_ = 1;
    uint32_t packedBytes_ = 0;
    uint32_t packedPlane_ = 0;
    uint32_t scaleSlot_ = 0;
    uint32_t partialStride_ = 0;
    uint32_t operandElems_ = 0;
    uint32_t packedGroups_ = 0;
    uint32_t unpackGroups_ = 0;
    uint32_t unpackBytes_ = 0;
    float scale_ = 1.0f;
};

template <typename scalar_t>
class TurboQuantFp16DecodeSplit {
public:
    __aicore__ inline explicit TurboQuantFp16DecodeSplit(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *query, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *blockTables, __gm__ void *contextLens, __gm__ void *workspace,
                                uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize,
                                uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits, float scale)
    {
        numTokens_ = numTokens;
        numHeads_ = numHeads;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        blockSize_ = blockSize;
        maxBlocksPerSeq_ = maxBlocksPerSeq;
        numSplits_ = numSplits;
        scale_ = scale;
        groupHeads_ = numHeads / numKvHeads;
        partialStride_ = headSize + kPartialTail;

        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(query));
        keyCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(keyCache));
        valueCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(valueCache));
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTables));
        contextLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(contextLens), numTokens);
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

        pipe_->InitBuffer(aQ1_, kMaxGroupHeads * headSize_ * sizeof(scalar_t));
        pipe_->InitBuffer(aP1_, kMaxGroupHeads * kCubeTileRows * sizeof(scalar_t));
        pipe_->InitBuffer(b1_, kCubeTileRows * headSize_ * sizeof(scalar_t));
        if ASCEND_IS_AIC {
            pipe_->InitBuffer(a2_, kMaxGroupHeads * headSize_ * sizeof(scalar_t));
            pipe_->InitBuffer(b2_, kCubeTileRows * headSize_ * sizeof(scalar_t));
            pipe_->InitBuffer(co1_, kMaxGroupHeads * headSize_ * sizeof(float));
        }

        pipe_->InitBuffer(accBuf_, kMaxGroupHeads * headSize_ * sizeof(float));
        pipe_->InitBuffer(scoreBuf_, kMaxGroupHeads * kCubeTileRows * sizeof(float));
        pipe_->InitBuffer(probBuf_, kMaxGroupHeads * kCubeTileRows * sizeof(scalar_t));
        pipe_->InitBuffer(ctxBuf_, kMaxGroupHeads * headSize_ * sizeof(float));
        pipe_->InitBuffer(stateBuf_, 5 * kMaxGroupHeads * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, 4 * kMaxGroupHeads * kFp32PerBlock * sizeof(float));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process(uint32_t tasksPerCore)
    {
        const uint32_t tasks = numTokens_ * numKvHeads_ * numSplits_;
        uint32_t start = MixBlockIdx() * tasksPerCore;
        uint32_t end = start + tasksPerCore;
        if (end > tasks) {
            end = tasks;
        }
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t split = task % numSplits_;
            const uint32_t rest = task / numSplits_;
            ComputeSplit(rest / numKvHeads_, rest % numKvHeads_, split);
        }
    }

private:
    __aicore__ inline uint64_t PartialOffset(uint32_t token, uint32_t head, uint32_t split) const
    {
        return ((static_cast<uint64_t>(token) * numHeads_ + head) * numSplits_ + split) * partialStride_;
    }

    __aicore__ inline void ComputeSplit(uint32_t token, uint32_t kvHead, uint32_t split)
    {
        AscendC::LocalTensor<float> acc = accBuf_.Get<float>();
        AscendC::LocalTensor<float> state = stateBuf_.Get<float>();
        AscendC::LocalTensor<float> runMax = state;
        AscendC::LocalTensor<float> runSum = state[kMaxGroupHeads * kFp32PerBlock];

        if ASCEND_IS_AIV {
            AscendC::Duplicate(acc, 0.0f, kMaxGroupHeads * headSize_);
            AscendC::Duplicate(state, 0.0f, 5 * kMaxGroupHeads * kFp32PerBlock);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t h = 0; h < groupHeads_; ++h) {
                AscendC::Duplicate(runMax[h * kFp32PerBlock], kNegInf, 1);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        const int32_t contextLen = contextLenGm_.GetValue(token);
        const uint32_t seqBlocks = contextLen > 0 ? CeilDiv(static_cast<uint32_t>(contextLen), blockSize_) : 0;
        const uint32_t blocksPerSplit = CeilDiv(seqBlocks, numSplits_);
        const uint32_t blockStart = split * blocksPerSplit;
        uint32_t blockEnd = blockStart + blocksPerSplit;
        if (blockEnd > seqBlocks) {
            blockEnd = seqBlocks;
        }

        if (blockStart < blockEnd) {
            if ASCEND_IS_AIC {
                AscendC::LocalTensor<scalar_t> qL1 = aQ1_.template Get<scalar_t>();
                AscendC::DataCopy(qL1,
                                  queryGm_[(static_cast<uint64_t>(token) * numHeads_ + kvHead * groupHeads_) *
                                           headSize_],
                                  AscendC::Nd2NzParams(1, static_cast<uint16_t>(kMaxGroupHeads), headSize_, 0,
                                                       headSize_, static_cast<uint16_t>(kMaxGroupHeads), 1, 0));
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            for (uint32_t block = blockStart; block < blockEnd; ++block) {
                const int32_t physical =
                    blockTableGm_.GetValue(static_cast<uint64_t>(token) * maxBlocksPerSeq_ + block);
                if (physical < 0) {
                    continue;
                }
                uint32_t rows = blockSize_;
                const uint32_t consumed = block * blockSize_;
                if (consumed + rows > static_cast<uint32_t>(contextLen)) {
                    rows = static_cast<uint32_t>(contextLen) - consumed;
                }
                for (uint32_t base = 0; base < rows; base += kCubeTileRows) {
                    uint32_t valid = rows - base;
                    if (valid > kCubeTileRows) {
                        valid = kCubeTileRows;
                    }
                    ProcessTile(static_cast<uint32_t>(physical), kvHead, base, valid, acc, state);
                }
            }
        }

        if ASCEND_IS_AIV {
          if (IsPrimarySubcore()) {
            AscendC::LocalTensor<float> tail = reduceBuf_.Get<float>();
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t h = 0; h < groupHeads_; ++h) {
                const uint32_t head = kvHead * groupHeads_ + h;
                const uint64_t offset = PartialOffset(token, head, split);
                AscendC::Duplicate(tail, 0.0f, kPartialTail);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(tail[kPartialMaxLane], runMax[h * kFp32PerBlock], 0.0f, 1);
                AscendC::Adds(tail[kPartialSumLane], runSum[h * kFp32PerBlock], 0.0f, 1);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(workspaceGm_[offset], acc[h * headSize_], headSize_);
                AscendC::DataCopy(workspaceGm_[offset + headSize_], tail, kPartialTail);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
          }
        }
    }

    __aicore__ inline void ProcessTile(uint32_t physical, uint32_t kvHead, uint32_t rowBase, uint32_t valid,
                                       const AscendC::LocalTensor<float> &acc,
                                       const AscendC::LocalTensor<float> &state)
    {
        AscendC::LocalTensor<float> scores = scoreBuf_.Get<float>();
        AscendC::LocalTensor<float> ctx = ctxBuf_.Get<float>();

        const uint64_t row = static_cast<uint64_t>(physical) * blockSize_ + rowBase;
        const uint64_t cacheOff = (row * numKvHeads_ + kvHead) * headSize_;
        const AscendC::Nd2NzParams tileParams(1, static_cast<uint16_t>(kCubeTileRows), headSize_, 0,
                                              static_cast<uint64_t>(numKvHeads_) * headSize_,
                                              static_cast<uint16_t>(kCubeTileRows), 1, 0);

        if ASCEND_IS_AIC {
            AscendC::LocalTensor<scalar_t> kL1 = b1_.template Get<scalar_t>();
            AscendC::DataCopy(kL1, keyCacheGm_[cacheOff], tileParams);
            AscendC::PipeBarrier<PIPE_ALL>();
            Gemm(scores, aQ1_.template Get<scalar_t>(), kMaxGroupHeads, headSize_, kCubeTileRows,
                 true);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(vllm_ascend::turboquant::kFlagProductReady);
        }
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagProductReady);
            if (IsPrimarySubcore()) {
                Softmax(scores, state, acc, valid);
            }
            SignalOperandsReady();
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagOperandsReady);
            AscendC::LocalTensor<scalar_t> vL1 = b1_.template Get<scalar_t>();
            AscendC::DataCopy(vL1, valueCacheGm_[cacheOff], tileParams);
            AscendC::PipeBarrier<PIPE_ALL>();
            Gemm(ctx, aP1_.template Get<scalar_t>(), kMaxGroupHeads, kCubeTileRows, headSize_,
                 false);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(vllm_ascend::turboquant::kFlagProductReady);
        }
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagProductReady);
            if (IsPrimarySubcore()) {
                AscendC::Add(acc, acc, ctx, groupHeads_ * headSize_);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void Gemm(const AscendC::LocalTensor<float> &dstUb, const AscendC::LocalTensor<scalar_t> &ta1,
                                uint32_t m, uint32_t k, uint32_t n, bool transposeB)
    {
        AscendC::LocalTensor<scalar_t> tb1 = b1_.template Get<scalar_t>();
        AscendC::LocalTensor<scalar_t> ta2 = a2_.template Get<scalar_t>();
        AscendC::LocalTensor<scalar_t> tb2 = b2_.template Get<scalar_t>();
        AscendC::LocalTensor<float> tco = co1_.template Get<float>();

        constexpr uint16_t kFractalRows = 16;
        constexpr uint16_t kC0 = 16;
        AscendC::LoadData2DParamsV2 pa;
        pa.mStep = static_cast<uint16_t>(CeilDiv(m, kFractalRows));
        pa.kStep = static_cast<uint16_t>(CeilDiv(k, kC0));
        pa.srcStride = pa.mStep;
        pa.dstStride = pa.mStep;
        pa.ifTranspose = false;
        AscendC::LoadData(ta2, ta1, pa);

        AscendC::LoadData2DParamsV2 pb;
        if (transposeB) {
            pb.mStep = static_cast<uint16_t>(CeilDiv(n, kFractalRows));
            pb.kStep = static_cast<uint16_t>(CeilDiv(k, kC0));
            pb.srcStride = static_cast<uint16_t>(CeilDiv(n, kFractalRows));
            pb.ifTranspose = false;
        } else {
            pb.mStep = static_cast<uint16_t>(CeilDiv(k, kFractalRows));
            pb.kStep = static_cast<uint16_t>(CeilDiv(n, kC0));
            pb.srcStride = static_cast<uint16_t>(CeilDiv(k, kFractalRows));
            pb.ifTranspose = true;
        }
        pb.dstStride = static_cast<uint16_t>(CeilDiv(n, kFractalRows));
        AscendC::LoadData(tb2, tb1, pb);

        AscendC::TPipe *tpipe = GetTPipePtr();
        const event_t mte1ToM = static_cast<event_t>(tpipe->FetchEventID(AscendC::HardEvent::MTE1_M));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1ToM);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1ToM);
        AscendC::Mmad(tco, ta2, tb2,
                      AscendC::MmadParams(static_cast<uint16_t>(m), static_cast<uint16_t>(n),
                                          static_cast<uint16_t>(k), 0, false, true));
        const event_t mToFix = static_cast<event_t>(tpipe->FetchEventID(AscendC::HardEvent::M_FIX));
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(mToFix);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(mToFix);
        AscendC::Fixpipe<float, float, vllm_ascend::turboquant::kFixpipeToUb>(
            dstUb, tco,
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>(
                static_cast<uint16_t>(n), static_cast<uint16_t>(m),
                static_cast<uint16_t>(CeilDiv(m, kFractalRows) * kFractalRows), n));
        AscendC::PipeBarrier<PIPE_FIX>();
    }

    __aicore__ inline void Softmax(const AscendC::LocalTensor<float> &scores,
                                   const AscendC::LocalTensor<float> &state,
                                   const AscendC::LocalTensor<float> &acc, uint32_t valid)
    {
        AscendC::LocalTensor<float> runMax = state;
        AscendC::LocalTensor<float> runSum = state[kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> tileMax = state[2 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> newMax = state[3 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> alpha = state[4 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> reduce = reduceBuf_.Get<float>();
        AscendC::LocalTensor<float> reduceWork = ctxBuf_.Get<float>();
        AscendC::LocalTensor<scalar_t> probs = probBuf_.Get<scalar_t>();

        for (uint32_t h = 0; h < groupHeads_; ++h) {
            AscendC::LocalTensor<float> row = scores[h * kCubeTileRows];
            AscendC::Muls(row, row, scale_, kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            if (valid < kCubeTileRows) {
                AscendC::Duplicate(row[valid], kNegInf, kCubeTileRows - valid);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::ReduceMax<float>(tileMax[h * kFp32PerBlock], row, reduceWork, kCubeTileRows, false);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(newMax[h * kFp32PerBlock], runMax[h * kFp32PerBlock], tileMax[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(alpha[h * kFp32PerBlock], runMax[h * kFp32PerBlock], newMax[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(alpha[h * kFp32PerBlock], alpha[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<float> brcb = reduce[2 * kFp32PerBlock];
            BroadcastScalar(brcb, newMax[h * kFp32PerBlock]);
            BroadcastSub(row, row, brcb, kCubeTileRows);
            AscendC::Exp(row, row, kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            if (valid < kCubeTileRows) {
                AscendC::Duplicate(row[valid], 0.0f, kCubeTileRows - valid);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::ReduceSum<float>(reduce, row, reduce[kFp32PerBlock], kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(runSum[h * kFp32PerBlock], runSum[h * kFp32PerBlock], alpha[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(runSum[h * kFp32PerBlock], runSum[h * kFp32PerBlock], reduce, 1);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(probs[h * kCubeTileRows], row, AscendC::RoundMode::CAST_RINT, kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::LocalTensor<scalar_t> pL1 = aP1_.template Get<scalar_t>();
        SyncVectorToMte3();
        AscendC::DataCopy(pL1, probs,
                          AscendC::Nd2NzParams(1, static_cast<uint16_t>(kMaxGroupHeads), kCubeTileRows, 0,
                                               kCubeTileRows, static_cast<uint16_t>(kMaxGroupHeads), 1, 0));

        const uint8_t rowBlocks = static_cast<uint8_t>(headSize_ / kFp32PerBlock);
        AscendC::LocalTensor<float> alphaBlocks = reduce[2 * kMaxGroupHeads * kFp32PerBlock];
        for (uint32_t h = 0; h < groupHeads_; ++h) {
            AscendC::Brcb(alphaBlocks[h * kFp32PerBlock], alpha[h * kFp32PerBlock], 1,
                          {1, static_cast<uint16_t>(kFp32PerBlock)});
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t col = 0; col < headSize_; col += kFp32PerRepeat) {
            AscendC::Mul(acc[col], acc[col], alphaBlocks, static_cast<uint64_t>(kFp32PerRepeat),
                         static_cast<uint8_t>(groupHeads_), {1, 1, 0, rowBlocks, rowBlocks, 1});
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t h = 0; h < groupHeads_; ++h) {
            AscendC::Adds(runMax[h * kFp32PerBlock], newMax[h * kFp32PerBlock], 0.0f, 1);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::TPipe *pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> aQ1_;
    AscendC::TBuf<AscendC::TPosition::A1> aP1_;
    AscendC::TBuf<AscendC::TPosition::B1> b1_;
    AscendC::TBuf<AscendC::TPosition::A2> a2_;
    AscendC::TBuf<AscendC::TPosition::B2> b2_;
    AscendC::TBuf<AscendC::TPosition::CO1> co1_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scoreBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> probBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> ctxBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stateBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceBuf_;
    AscendC::GlobalTensor<scalar_t> queryGm_;
    AscendC::GlobalTensor<scalar_t> keyCacheGm_;
    AscendC::GlobalTensor<scalar_t> valueCacheGm_;
    AscendC::GlobalTensor<int32_t> blockTableGm_;
    AscendC::GlobalTensor<int32_t> contextLenGm_;
    AscendC::GlobalTensor<float> workspaceGm_;
    uint32_t numTokens_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t groupHeads_ = 1;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlocksPerSeq_ = 0;
    uint32_t numSplits_ = 1;
    uint32_t partialStride_ = 0;
    float scale_ = 1.0f;
};

template <typename scalar_t>
class TurboQuantPlainCombine {
public:
    __aicore__ inline explicit TurboQuantPlainCombine(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *workspace, __gm__ void *output, uint32_t numTokens, uint32_t numHeads,
                                uint32_t headSize, uint32_t numSplits)
    {
        numTokens_ = numTokens;
        numHeads_ = numHeads;
        headSize_ = headSize;
        numSplits_ = numSplits;
        partialStride_ = headSize + kPartialTail;

        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(output));

        pipe_->InitBuffer(outQueue_, 1, headSize_ * sizeof(scalar_t));
        pipe_->InitBuffer(accBuf_, 2 * headSize_ * sizeof(float));
        pipe_->InitBuffer(stateBuf_, 5 * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(partialBuf_, kPartialTail * sizeof(float));
        pipe_->InitBuffer(brcbBuf_, 4 * kBrcbDstLanes * sizeof(float));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process(uint32_t tasksPerCore)
    {
        const uint32_t tasks = numTokens_ * numHeads_;
        uint32_t start = static_cast<uint32_t>(AscendC::GetBlockIdx()) * tasksPerCore;
        uint32_t end = start + tasksPerCore;
        if (end > tasks) {
            end = tasks;
        }
        for (uint32_t task = start; task < end; ++task) {
            Combine(task / numHeads_, task % numHeads_);
        }
    }

private:
    __aicore__ inline void Combine(uint32_t token, uint32_t head)
    {
        AscendC::LocalTensor<float> acc = accBuf_.Get<float>();
        AscendC::LocalTensor<float> partAcc = acc[headSize_];
        AscendC::LocalTensor<float> state = stateBuf_.Get<float>();
        AscendC::LocalTensor<float> runMax = state[kPartialMaxLane];
        AscendC::LocalTensor<float> runSum = state[kPartialSumLane];
        AscendC::LocalTensor<float> newMax = state[2 * kFp32PerBlock];
        AscendC::LocalTensor<float> alpha = state[3 * kFp32PerBlock];
        AscendC::LocalTensor<float> beta = state[4 * kFp32PerBlock];
        AscendC::LocalTensor<float> brcb = brcbBuf_.Get<float>();
        AscendC::LocalTensor<float> bAlpha = brcb;
        AscendC::LocalTensor<float> bBeta = brcb[kBrcbDstLanes];
        AscendC::LocalTensor<float> bSum = brcb[2 * kBrcbDstLanes];
        AscendC::LocalTensor<float> invSum = brcb[3 * kBrcbDstLanes];
        AscendC::LocalTensor<float> partState = partialBuf_.Get<float>();
        AscendC::LocalTensor<float> partMax = partState[kPartialMaxLane];
        AscendC::LocalTensor<float> partSum = partState[kPartialSumLane];

        AscendC::Duplicate(acc, 0.0f, headSize_);
        AscendC::Duplicate(state, 0.0f, 5 * kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(runMax, kNegInf, 1);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t split = 0; split < numSplits_; ++split) {
            const uint64_t offset =
                ((static_cast<uint64_t>(token) * numHeads_ + head) * numSplits_ + split) * partialStride_;
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(partAcc, workspaceGm_[offset], headSize_);
            AscendC::DataCopy(partState, workspaceGm_[offset + headSize_], kPartialTail);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::Max(newMax, runMax, partMax, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(alpha, runMax, newMax, 1);
            AscendC::Sub(beta, partMax, newMax, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(alpha, alpha, 1);
            AscendC::Exp(beta, beta, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(runSum, runSum, alpha, 1);
            AscendC::Mul(partSum, partSum, beta, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(runSum, runSum, partSum, 1);
            AscendC::PipeBarrier<PIPE_V>();

            BroadcastScalar(bAlpha, alpha);
            BroadcastScalar(bBeta, beta);
            TurboQuantCodec4::BroadcastMul(acc, acc, bAlpha, headSize_);
            TurboQuantCodec4::BroadcastMul(partAcc, partAcc, bBeta, headSize_);
            AscendC::Add(acc, acc, partAcc, headSize_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(runMax, newMax, 0.0f, 1);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Adds(runSum, runSum, TurboQuantCodec4::kEps, 1);
        AscendC::PipeBarrier<PIPE_V>();
        BroadcastScalar(bSum, runSum);
        AscendC::Duplicate(invSum, 1.0f, kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(invSum, invSum, bSum, kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        TurboQuantCodec4::BroadcastMul(acc, acc, invSum, headSize_);

        AscendC::LocalTensor<scalar_t> out = outQueue_.template AllocTensor<scalar_t>();
        AscendC::Cast(out, acc, AscendC::RoundMode::CAST_RINT, headSize_);
        AscendC::PipeBarrier<PIPE_V>();
        outQueue_.EnQue(out);
        out = outQueue_.template DeQue<scalar_t>();
        AscendC::DataCopy(outputGm_[(static_cast<uint64_t>(token) * numHeads_ + head) * headSize_], out, headSize_);
        outQueue_.FreeTensor(out);
    }

    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stateBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> partialBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> brcbBuf_;
    AscendC::GlobalTensor<float> workspaceGm_;
    AscendC::GlobalTensor<scalar_t> outputGm_;
    uint32_t numTokens_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t headSize_ = 0;
    uint32_t numSplits_ = 1;
    uint32_t partialStride_ = 0;
};

class TurboQuantCubeGemmProbe {
public:
    using Mm = TurboQuantCubeMm<TurboQuantMode::KV5_FP8>;
    using OperandT = typename Mm::OperandT;

    __aicore__ inline explicit TurboQuantCubeGemmProbe(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *a, __gm__ void *b, __gm__ void *c, uint32_t headSize,
                                uint32_t tileRows, uint32_t aElems, uint32_t bElems, uint32_t cElems)
    {
        aElems_ = aElems;
        bElems_ = bElems;
        cElems_ = cElems;
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OperandT *>(a));
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OperandT *>(b));
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c));
        mm_.Init(pipe_, headSize, tileRows);
        pipe_->InitBuffer(stageBuf_, aElems > bElems ? aElems : bElems);
        pipe_->InitBuffer(outBuf_, cElems * sizeof(float));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Run(uint32_t m, uint32_t k, uint32_t n, uint32_t bIsNk, uint32_t variant)
    {
        AscendC::LocalTensor<float> out = outBuf_.Get<float>();
        if ASCEND_IS_AIV {
            AscendC::LocalTensor<OperandT> stage = stageBuf_.Get<OperandT>();
            AscendC::DataCopy(stage, aGm_, aElems_);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(bIsNk != 0 ? mm_.A1Query() : mm_.A1Probs(), stage, aElems_);
            AscendC::DataCopy(stage, bGm_, bElems_);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(mm_.B1(), stage, bElems_);
            SignalOperandsReady();
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagOperandsReady);
            if (variant == 10) {
                AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagOperandsReady);
            }
            if (bIsNk != 0) {
                mm_.GemmScores(out, mm_.B1(), m, k, n);
            } else {
                mm_.GemmContext(out, mm_.B1(), m, k, n, variant);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(vllm_ascend::turboquant::kFlagProductReady);
        }
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagProductReady);
            if (IsPrimarySubcore()) {
                AscendC::DataCopy(cGm_, out, cElems_);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    AscendC::TPipe *pipe_;
    Mm mm_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stageBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outBuf_;
    AscendC::GlobalTensor<OperandT> aGm_;
    AscendC::GlobalTensor<OperandT> bGm_;
    AscendC::GlobalTensor<float> cGm_;
    uint32_t aElems_ = 0;
    uint32_t bElems_ = 0;
    uint32_t cElems_ = 0;
};

}

#define TURBOQUANT_MM_RESHAPE_AND_CACHE_DECLARE(MODE_NAME, MODE, TYPE)                                               \
    extern "C" __global__ __aicore__ void turboquant_mm_reshape_and_cache_##MODE_NAME##_##TYPE(                      \
        GM_ADDR key, GM_ADDR value, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR slotMapping,   \
        GM_ADDR piSigns, GM_ADDR rotTables, GM_ADDR modeTables, uint32_t numTokens, uint32_t numKvHeads,             \
        uint32_t headSize, uint32_t blockSize, uint32_t tokensPerCore, float invSqrtLen)                             \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantModeReshapeAndCache<MODE, TYPE> op(&pipe);                                                         \
        op.Init(key, value, keyCache, valueCache, scaleCache, slotMapping, piSigns, rotTables, modeTables,           \
                numTokens, numKvHeads, headSize, blockSize, tokensPerCore, invSqrtLen);                              \
        op.Process();                                                                                                \
    }

#define TURBOQUANT_MM_DECODE_SPLIT_DECLARE(MODE_NAME, MODE, TYPE)                                                    \
    extern "C" __global__ __aicore__ void turboquant_mm_decode_split_##MODE_NAME##_##TYPE(                           \
        GM_ADDR queryRot, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR blockTables,             \
        GM_ADDR contextLens, GM_ADDR modeTables, GM_ADDR workspace,                                                  \
        uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,           \
        uint32_t maxBlocksPerSeq, uint32_t numSplits, uint32_t tasksPerCore, float scale, float invSqrtLen)          \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantCubeDecodeSplit<MODE, TYPE> op(&pipe);                                                             \
        op.Init(queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables,                    \
                workspace, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, scale,  \
                invSqrtLen);                                                                                         \
        op.Process(tasksPerCore);                                                                                    \
    }

#define TURBOQUANT_MM_DECLARE_MODE(MODE_NAME, MODE)                                                                  \
    TURBOQUANT_MM_RESHAPE_AND_CACHE_DECLARE(MODE_NAME, MODE, half)                                                   \
    TURBOQUANT_MM_DECODE_SPLIT_DECLARE(MODE_NAME, MODE, half)

TURBOQUANT_MM_DECLARE_MODE(kv3fp4, TurboQuantMode::KV3_FP4)
TURBOQUANT_MM_DECLARE_MODE(kv4fp8, TurboQuantMode::KV4_FP8)
TURBOQUANT_MM_DECLARE_MODE(kv5fp8, TurboQuantMode::KV5_FP8)

#ifdef VLLM_ASCEND_TQ_DECODE_ABLATION
#define TURBOQUANT_MM_DECODE_ABLATION_DECLARE(STAGE_NAME, STAGE)                                                   \
    extern "C" __global__ __aicore__ void turboquant_mm_decode_ablation_kv4fp8_##STAGE_NAME##_half(                \
        GM_ADDR queryRot, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR blockTables,             \
        GM_ADDR contextLens, GM_ADDR modeTables, GM_ADDR workspace,                                                  \
        uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,           \
        uint32_t maxBlocksPerSeq, uint32_t numSplits, uint32_t tasksPerCore, float scale, float invSqrtLen)          \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantCubeDecodeSplit<TurboQuantMode::KV4_FP8, half, STAGE> op(&pipe);                                   \
        op.Init(queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables,                    \
                workspace, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, scale,  \
                invSqrtLen);                                                                                         \
        op.Process(tasksPerCore);                                                                                    \
    }

TURBOQUANT_MM_DECODE_ABLATION_DECLARE(s0, DecodeAblationStage::STAGE_0_MTE2_ONLY)
TURBOQUANT_MM_DECODE_ABLATION_DECLARE(s1, DecodeAblationStage::STAGE_1_UNPACK)
TURBOQUANT_MM_DECODE_ABLATION_DECLARE(s2, DecodeAblationStage::STAGE_2_QUERY_PREP)
TURBOQUANT_MM_DECODE_ABLATION_DECLARE(s3, DecodeAblationStage::STAGE_3_L1_STAGING)
TURBOQUANT_MM_DECODE_ABLATION_DECLARE(s4, DecodeAblationStage::STAGE_4_SCORE_GEMM)
#endif

#define TURBOQUANT_FP16_DECODE_SPLIT_DECLARE(TYPE)                                                                   \
    extern "C" __global__ __aicore__ void turboquant_fp16_decode_split_##TYPE(                                       \
        GM_ADDR query, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR blockTables, GM_ADDR contextLens,               \
        GM_ADDR workspace, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize,            \
        uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits, uint32_t tasksPerCore, float scale)        \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantFp16DecodeSplit<TYPE> op(&pipe);                                                                    \
        op.Init(query, keyCache, valueCache, blockTables, contextLens, workspace, numTokens, numHeads, numKvHeads,   \
                headSize, blockSize, maxBlocksPerSeq, numSplits, scale);                                             \
        op.Process(tasksPerCore);                                                                                    \
    }

#define TURBOQUANT_PLAIN_COMBINE_DECLARE(TYPE)                                                                       \
    extern "C" __global__ __aicore__ void turboquant_plain_combine_##TYPE(                                           \
        GM_ADDR workspace, GM_ADDR output, uint32_t numTokens, uint32_t numHeads, uint32_t headSize,                 \
        uint32_t numSplits, uint32_t tasksPerCore)                                                                   \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantPlainCombine<TYPE> op(&pipe);                                                                      \
        op.Init(workspace, output, numTokens, numHeads, headSize, numSplits);                                        \
        op.Process(tasksPerCore);                                                                                    \
    }

TURBOQUANT_FP16_DECODE_SPLIT_DECLARE(half)
TURBOQUANT_PLAIN_COMBINE_DECLARE(half)

extern "C" __global__ __aicore__ void turboquant_cube_gemm_probe_fp8(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t m, uint32_t k, uint32_t n, uint32_t headSize, uint32_t tileRows,
    uint32_t aElems, uint32_t bElems, uint32_t cElems, uint32_t bIsNk, uint32_t variant)
{
    AscendC::TPipe pipe;
    TurboQuantCubeGemmProbe op(&pipe);
    op.Init(a, b, c, headSize, tileRows, aElems, bElems, cElems);
    op.Run(m, k, n, bIsNk, variant);
}

namespace vllm_ascend {

void turboquant_mm_reshape_and_cache_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *key,
                                          void *value, void *keyCache, void *valueCache, void *scaleCache,
                                          void *slotMapping, void *piSigns, void *rotTables, void *modeTables,
                                          uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,
                                          uint32_t blockSize, uint32_t tokensPerCore, float invSqrtLen)
{
    if (type != AscendType::FP16) {
        return;
    }
    switch (static_cast<turboquant::TurboQuantMode>(mode)) {
        case turboquant::TurboQuantMode::KV3_FP4:
            turboquant_mm_reshape_and_cache_kv3fp4_half<<<blockDim, nullptr, stream>>>(
                key, value, keyCache, valueCache, scaleCache, slotMapping, piSigns, rotTables, modeTables, numTokens,
                numKvHeads, headSize, blockSize, tokensPerCore, invSqrtLen);
            break;
        case turboquant::TurboQuantMode::KV4_FP8:
            turboquant_mm_reshape_and_cache_kv4fp8_half<<<blockDim, nullptr, stream>>>(
                key, value, keyCache, valueCache, scaleCache, slotMapping, piSigns, rotTables, modeTables, numTokens,
                numKvHeads, headSize, blockSize, tokensPerCore, invSqrtLen);
            break;
        default:
            turboquant_mm_reshape_and_cache_kv5fp8_half<<<blockDim, nullptr, stream>>>(
                key, value, keyCache, valueCache, scaleCache, slotMapping, piSigns, rotTables, modeTables, numTokens,
                numKvHeads, headSize, blockSize, tokensPerCore, invSqrtLen);
            break;
    }
}

void turboquant_mm_decode_split_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *queryRot,
                                     void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
                                     void *contextLens, void *modeTables,
                                     void *workspace, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                     uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                     uint32_t numSplits, uint32_t tasksPerCore, float scale, float invSqrtLen)
{
    if (type != AscendType::FP16) {
        return;
    }
    switch (static_cast<turboquant::TurboQuantMode>(mode)) {
        case turboquant::TurboQuantMode::KV3_FP4:
            turboquant_mm_decode_split_kv3fp4_half<<<blockDim, nullptr, stream>>>(
                queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables,
                workspace, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
                tasksPerCore, scale, invSqrtLen);
            break;
        case turboquant::TurboQuantMode::KV4_FP8:
            turboquant_mm_decode_split_kv4fp8_half<<<blockDim, nullptr, stream>>>(
                queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables,
                workspace, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
                tasksPerCore, scale, invSqrtLen);
            break;
        default:
            turboquant_mm_decode_split_kv5fp8_half<<<blockDim, nullptr, stream>>>(
                queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables,
                workspace, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
                tasksPerCore, scale, invSqrtLen);
            break;
    }
}

#ifdef VLLM_ASCEND_TQ_DECODE_ABLATION
void turboquant_mm_decode_ablation_impl(int32_t stage, AscendType type, void *stream, uint32_t blockDim,
                                        void *queryRot,
                                        void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
                                        void *contextLens, void *modeTables,
                                        void *workspace, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                        uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                        uint32_t numSplits, uint32_t tasksPerCore, float scale, float invSqrtLen)
{
    if (type != AscendType::FP16) {
        return;
    }
    switch (static_cast<turboquant::DecodeAblationStage>(stage)) {
        case turboquant::DecodeAblationStage::STAGE_0_MTE2_ONLY:
            turboquant_mm_decode_ablation_kv4fp8_s0_half<<<blockDim, nullptr, stream>>>(
                queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables,
                workspace, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
                tasksPerCore, scale, invSqrtLen);
            break;
        case turboquant::DecodeAblationStage::STAGE_1_UNPACK:
            turboquant_mm_decode_ablation_kv4fp8_s1_half<<<blockDim, nullptr, stream>>>(
                queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables,
                workspace, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
                tasksPerCore, scale, invSqrtLen);
            break;
        case turboquant::DecodeAblationStage::STAGE_2_QUERY_PREP:
            turboquant_mm_decode_ablation_kv4fp8_s2_half<<<blockDim, nullptr, stream>>>(
                queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables,
                workspace, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
                tasksPerCore, scale, invSqrtLen);
            break;
        case turboquant::DecodeAblationStage::STAGE_3_L1_STAGING:
            turboquant_mm_decode_ablation_kv4fp8_s3_half<<<blockDim, nullptr, stream>>>(
                queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables,
                workspace, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
                tasksPerCore, scale, invSqrtLen);
            break;
        case turboquant::DecodeAblationStage::STAGE_4_SCORE_GEMM:
            turboquant_mm_decode_ablation_kv4fp8_s4_half<<<blockDim, nullptr, stream>>>(
                queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables,
                workspace, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
                tasksPerCore, scale, invSqrtLen);
            break;
        default:
            turboquant_mm_decode_split_kv4fp8_half<<<blockDim, nullptr, stream>>>(
                queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables,
                workspace, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
                tasksPerCore, scale, invSqrtLen);
            break;
    }
}
#endif

void turboquant_fp16_decode_impl(AscendType type, void *stream, uint32_t splitBlockDim, uint32_t combineBlockDim,
                                 void *query, void *keyCache, void *valueCache, void *blockTables,
                                 void *contextLens, void *workspace, void *output, uint32_t numTokens,
                                 uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,
                                 uint32_t maxBlocksPerSeq, uint32_t numSplits, uint32_t splitTasksPerCore,
                                 uint32_t combineTasksPerCore, float scale)
{
    if (type != AscendType::FP16) {
        return;
    }
    turboquant_fp16_decode_split_half<<<splitBlockDim, nullptr, stream>>>(
        query, keyCache, valueCache, blockTables, contextLens, workspace, numTokens, numHeads, numKvHeads, headSize,
        blockSize, maxBlocksPerSeq, numSplits, splitTasksPerCore, scale);
    turboquant_plain_combine_half<<<combineBlockDim, nullptr, stream>>>(
        workspace, output, numTokens, numHeads, headSize, numSplits, combineTasksPerCore);
}

void turboquant_cube_gemm_probe_impl(void *stream, void *a, void *b, void *c, uint32_t m, uint32_t k, uint32_t n,
                                     uint32_t headSize, uint32_t tileRows, uint32_t aElems, uint32_t bElems,
                                     uint32_t cElems, uint32_t bIsNk, uint32_t variant)
{
    turboquant_cube_gemm_probe_fp8<<<1, nullptr, stream>>>(a, b, c, m, k, n, headSize, tileRows, aElems, bElems,
                                                           cElems, bIsNk, variant);
}

}
