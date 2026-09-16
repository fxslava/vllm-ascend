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

// The shipping TurboQuant kernels (built into the wheel for arch32 and arch35): the 4-bit cache write
// and the AIV-only paged-attention decode. One launch per decode step; a context above the fused limit
// is split and reduced inside the same launch.

#include "kernel_operator.h"

#include "../../../kernels/types.h"
#include "common/turboquant_codec_950.h"
#include "vector/turboquant_vector_service.h"

namespace {

using vllm_ascend::turboquant::BroadcastScalar;
using vllm_ascend::turboquant::BroadcastSub;
using vllm_ascend::turboquant::CeilDiv;
using vllm_ascend::turboquant::kBrcbDstLanes;
using vllm_ascend::turboquant::kFp32PerBlock;
using vllm_ascend::turboquant::kFp32PerRepeat;
using vllm_ascend::turboquant::kGatherSrcBase;
using vllm_ascend::turboquant::kPartialMaxLane;
using vllm_ascend::turboquant::kPartialSumLane;
using vllm_ascend::turboquant::kPartialTail;
using vllm_ascend::turboquant::ScaleSlotFloats;
using vllm_ascend::turboquant::SyncMte3ToVector;
using vllm_ascend::turboquant::SyncMte2ToVector;
using vllm_ascend::turboquant::SyncVectorToMte2;
using vllm_ascend::turboquant::SyncVectorToMte3;
using vllm_ascend::turboquant::TurboQuantCodec4;
using vllm_ascend::turboquant::TurboQuantPartialReducer;
using vllm_ascend::turboquant::TurboQuantTileBurst;
using vllm_ascend::turboquant::VecBarrier;
using vllm_ascend::turboquant::WriteNormalizedHeads;

constexpr uint32_t kTileRows = vllm_ascend::turboquant::kAivTileRows;
constexpr float kNegInf = -1.0e30f;
constexpr uint32_t kSlotRing = 4;
constexpr bool kPagedAttentionVecBarriers = false;

template <typename scalar_t>
class TurboQuantReshapeAndCache {
public:
    __aicore__ inline explicit TurboQuantReshapeAndCache(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *key, __gm__ void *value, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *scaleCache, __gm__ void *slotMapping, __gm__ void *piSigns,
                                __gm__ void *tables, uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,
                                uint32_t blockSize, uint32_t tokensPerCore, float invSqrtLen)
    {
        numTokens_ = numTokens;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        blockSize_ = blockSize;
        tokensPerCore_ = tokensPerCore;
        packedBytes_ = headSize / TurboQuantCodec4::kPackFactor;
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
        tablesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tables));

        pipe_->InitBuffer(inQueue_, 2, 2 * headPlane_ * sizeof(scalar_t));
        pipe_->InitBuffer(outPacked_, 2, 2 * packedPlane_ * sizeof(int8_t));
        pipe_->InitBuffer(outScale_, 2, scaleSlot_ * sizeof(float));
        pipe_->InitBuffer(workBuf_, 2 * headSize_ * sizeof(float));
        pipe_->InitBuffer(signBuf_, headSize_ * sizeof(float));
        pipe_->InitBuffer(stepBuf_, 2 * numKvHeads_ * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(scaleIdxBuf_, 2 * numKvHeads_ * sizeof(int32_t));

        codec_.Init(pipe_, headSize_, 1, invSqrtLen, tablesGm_);

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
    __aicore__ inline void CopyIn(uint32_t token, uint32_t step)
    {
        const int32_t slot = slotGm_.GetValue(token);
        const uint32_t ring = step % kSlotRing;
        if (slot < 0) {
            slotValid_[ring] = false;
            cacheOffset_[ring] = 0;
            scaleOffset_[ring] = 0;
        } else {
            const uint64_t row = static_cast<uint64_t>(slot);
            slotValid_[ring] = true;
            cacheOffset_[ring] = row * packedPlane_;
            scaleOffset_[ring] = row * scaleSlot_;
        }

        AscendC::LocalTensor<scalar_t> in = inQueue_.template AllocTensor<scalar_t>();
        const uint64_t offset = static_cast<uint64_t>(token) * headPlane_;
        AscendC::DataCopy(in, keyGm_[offset], headPlane_);
        AscendC::DataCopy(in[headPlane_], valueGm_[offset], headPlane_);
        inQueue_.EnQue(in);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<scalar_t> in = inQueue_.template DeQue<scalar_t>();
        AscendC::LocalTensor<int8_t> packed = outPacked_.template AllocTensor<int8_t>();
        AscendC::LocalTensor<float> scales = outScale_.template AllocTensor<float>();

        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::LocalTensor<float> steps = stepBuf_.Get<float>();
        AscendC::LocalTensor<float> vec = work;
        AscendC::LocalTensor<float> tmp = work[headSize_];

        for (uint32_t plane = 0; plane < 2 * numKvHeads_; ++plane) {
            AscendC::Cast(vec, in[plane * headSize_], AscendC::RoundMode::CAST_NONE, headSize_);
            AscendC::PipeBarrier<PIPE_V>();
            codec_.ApplyPi(vec, tmp, signs, static_cast<int>(headSize_));
            codec_.Quantize4Bit(packed[plane * packedBytes_], vec, steps[plane * kFp32PerBlock],
                                static_cast<int>(headSize_));
        }

        AscendC::LocalTensor<uint32_t> gatherIdx = scaleIdxBuf_.Get<int32_t>().ReinterpretCast<uint32_t>();
        AscendC::Duplicate(scales, 0.0f, scaleSlot_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(scales, steps, gatherIdx, kGatherSrcBase, 2 * numKvHeads_);
        AscendC::PipeBarrier<PIPE_V>();

        inQueue_.FreeTensor(in);
        outPacked_.EnQue(packed);
        outScale_.EnQue(scales);
    }

    __aicore__ inline void CopyOut(uint32_t step)
    {
        AscendC::LocalTensor<int8_t> packed = outPacked_.template DeQue<int8_t>();
        AscendC::LocalTensor<float> scales = outScale_.template DeQue<float>();

        const uint32_t ring = step % kSlotRing;
        if (slotValid_[ring]) {
            AscendC::DataCopy(keyCacheGm_[cacheOffset_[ring]], packed, packedPlane_);
            AscendC::DataCopy(valueCacheGm_[cacheOffset_[ring]], packed[packedPlane_], packedPlane_);
            AscendC::DataCopy(scaleCacheGm_[scaleOffset_[ring]], scales, scaleSlot_);
        }

        outPacked_.FreeTensor(packed);
        outScale_.FreeTensor(scales);
    }

    AscendC::TPipe *pipe_;
    TurboQuantCodec4 codec_;
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
    AscendC::GlobalTensor<int32_t> tablesGm_;
    uint64_t cacheOffset_[kSlotRing] = {0, 0, 0, 0};
    uint64_t scaleOffset_[kSlotRing] = {0, 0, 0, 0};
    bool slotValid_[kSlotRing] = {false, false, false, false};
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

template <typename scalar_t>
class TurboQuantPagedAttentionSplit {
public:
    __aicore__ inline explicit TurboQuantPagedAttentionSplit(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *queryRot, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *scaleCache, __gm__ void *blockTables, __gm__ void *contextLens,
                                __gm__ void *tables, __gm__ void *workspace, __gm__ void *output,
                                uint32_t numTokens, uint32_t numHeads,
                                uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                uint32_t numSplits, uint32_t fusedContextLimit, float scale, float invSqrtLen)
    {
        numTokens_ = numTokens;
        numHeads_ = numHeads;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        blockSize_ = blockSize;
        maxBlocksPerSeq_ = maxBlocksPerSeq;
        numSplits_ = numSplits;
        fusedContextLimit_ = fusedContextLimit;
        scale_ = scale;
        headsPerKv_ = numHeads / numKvHeads;
        packedBytes_ = headSize / TurboQuantCodec4::kPackFactor;
        packedPlane_ = numKvHeads_ * packedBytes_;
        scaleSlot_ = ScaleSlotFloats(numKvHeads_);
        partialStride_ = headSize + kPartialTail;

        queryRotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(queryRot));
        keyCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(keyCache));
        valueCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(valueCache));
        scaleCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scaleCache));
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTables));
        contextLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(contextLens), numTokens);
        tablesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tables));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(output));

        const uint32_t tileElems = kTileRows * headSize_;
        pipe_->InitBuffer(kvQueue_, 2, 2 * kTileRows * packedBytes_ * sizeof(int8_t));
        pipe_->InitBuffer(scaleQueue_, 2, kTileRows * scaleSlot_ * sizeof(float));
        pipe_->InitBuffer(qInQueue_, 1, headSize_ * sizeof(float));

        pipe_->InitBuffer(kvFloatBuf_, 2 * tileElems * sizeof(float));
        pipe_->InitBuffer(qTileBuf_, tileElems * sizeof(float));
        pipe_->InitBuffer(prodBuf_, tileElems * sizeof(float));
        pipe_->InitBuffer(accBuf_, headSize_ * sizeof(float));
        pipe_->InitBuffer(rowBuf_, 6 * kTileRows * sizeof(float));
        pipe_->InitBuffer(brcbBuf_, 2 * kBrcbDstLanes * sizeof(float) + kTileRows * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(stateBuf_, 5 * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(scaleIdxBuf_, 2 * kTileRows * sizeof(int32_t));

        codec_.Init(pipe_, headSize_, kTileRows, invSqrtLen, tablesGm_);
        pipe_->InitBuffer(outBuf_, headSize_ * sizeof(scalar_t));
        burst_.Init(pipe_, numKvHeads_, packedBytes_, kTileRows, 2);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline bool NeedsReduction() const
    {
        if (numSplits_ <= 1) {
            return false;
        }
        for (uint32_t token = 0; token < numTokens_; ++token) {
            if (!IsFused(contextLenGm_.GetValue(token))) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void Process(uint32_t tasksPerCore)
    {
        const uint32_t tasks = numTokens_ * numHeads_ * numSplits_;
        uint32_t start = static_cast<uint32_t>(AscendC::GetBlockIdx()) * tasksPerCore;
        uint32_t end = start + tasksPerCore;
        if (end > tasks) {
            end = tasks;
        }
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t split = task % numSplits_;
            const uint32_t head = (task / numSplits_) % numHeads_;
            const uint32_t token = task / (numSplits_ * numHeads_);
            ComputeSplit(token, head, split);
        }
    }

    // After AscendC::SyncAll: merge the partials of every split token this block owns.
    __aicore__ inline void Reduce(TurboQuantPartialReducer<scalar_t, kPagedAttentionVecBarriers> &reducer,
                                  uint32_t tasksPerCore)
    {
        const uint32_t tasks = numTokens_ * numHeads_;
        const uint32_t start = static_cast<uint32_t>(AscendC::GetBlockIdx()) * tasksPerCore;
        uint32_t end = start + tasksPerCore;
        if (end > tasks) {
            end = tasks;
        }
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t token = task / numHeads_;
            if (!IsFused(contextLenGm_.GetValue(token))) {
                reducer.Reduce(token, task % numHeads_);
            }
        }
    }

private:
    __aicore__ inline uint64_t PartialOffset(uint32_t token, uint32_t head, uint32_t split) const
    {
        return ((static_cast<uint64_t>(token) * numHeads_ + head) * numSplits_ + split) * partialStride_;
    }

    __aicore__ inline bool IsFused(int32_t contextLen) const
    {
        return numSplits_ <= 1 || contextLen <= static_cast<int32_t>(fusedContextLimit_);
    }

    __aicore__ inline void ComputeSplit(uint32_t token, uint32_t head, uint32_t split)
    {
        const int32_t contextLen = contextLenGm_.GetValue(token);
        const bool fused = IsFused(contextLen);
        if (fused && split > 0) {
            return;
        }

        AscendC::LocalTensor<float> state = stateBuf_.Get<float>();
        AscendC::LocalTensor<float> runMax = state[kPartialMaxLane];
        AscendC::LocalTensor<float> runSum = state[kPartialSumLane];
        AscendC::LocalTensor<float> newMax = state[2 * kFp32PerBlock];
        AscendC::LocalTensor<float> alpha = state[3 * kFp32PerBlock];
        AscendC::LocalTensor<float> tileMax = state[4 * kFp32PerBlock];

        AscendC::LocalTensor<float> acc = accBuf_.Get<float>();

        AscendC::Duplicate(acc, 0.0f, headSize_);
        AscendC::Duplicate(state, 0.0f, 5 * kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(runMax, kNegInf, 1);
        AscendC::PipeBarrier<PIPE_V>();

        const uint32_t seqBlocks = contextLen > 0 ? CeilDiv(static_cast<uint32_t>(contextLen), blockSize_) : 0;
        const uint32_t blocksPerSplit = CeilDiv(seqBlocks, fused ? 1u : numSplits_);
        const uint32_t blockStart = split * blocksPerSplit;
        uint32_t blockEnd = blockStart + blocksPerSplit;
        if (blockEnd > seqBlocks) {
            blockEnd = seqBlocks;
        }

        if (blockStart < blockEnd) {
            const uint32_t kvHead = head / headsPerKv_;
            PrepareTask(token, head, kvHead);
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

                CopyInTile(static_cast<uint32_t>(physical), kvHead, 0);
                for (uint32_t base = 0; base < rows; base += kTileRows) {
                    const uint32_t next = base + kTileRows;
                    if (next < rows) {
                        CopyInTile(static_cast<uint32_t>(physical), kvHead, next);
                    }
                    uint32_t valid = rows - base;
                    if (valid > kTileRows) {
                        valid = kTileRows;
                    }
                    AccumulateTile(valid, acc, runMax, runSum, newMax, alpha, tileMax);
                }
            }
        }

        if (fused) {
            WriteOutput(token, head, acc, runSum);
            return;
        }
        const uint64_t offset = PartialOffset(token, head, split);
        SyncVectorToMte3();
        AscendC::DataCopy(workspaceGm_[offset], acc, headSize_);
        AscendC::DataCopy(workspaceGm_[offset + headSize_], state, kPartialTail);
        SyncMte3ToVector();
    }

    __aicore__ inline void WriteOutput(uint32_t token, uint32_t head, const AscendC::LocalTensor<float> &acc,
                                       const AscendC::LocalTensor<float> &runSum)
    {
        const AscendC::LocalTensor<float> brcb = brcbBuf_.Get<float>();
        WriteNormalizedHeads<scalar_t, kPagedAttentionVecBarriers>(
            acc, runSum, 1, headSize_, brcb[2 * kBrcbDstLanes], brcb[3 * kBrcbDstLanes], outBuf_.Get<scalar_t>(),
            outputGm_, (static_cast<uint64_t>(token) * numHeads_ + head) * headSize_);
    }

    __aicore__ inline void PrepareTask(uint32_t token, uint32_t head, uint32_t kvHead)
    {
        AscendC::LocalTensor<float> q = qInQueue_.template AllocTensor<float>();
        AscendC::DataCopy(q, queryRotGm_[(static_cast<uint64_t>(token) * numHeads_ + head) * headSize_], headSize_);
        qInQueue_.EnQue(q);
        q = qInQueue_.template DeQue<float>();

        AscendC::LocalTensor<float> qTile = qTileBuf_.Get<float>();
        for (uint32_t row = 0; row < kTileRows; ++row) {
            AscendC::DataCopy(qTile[row * headSize_], q, headSize_);
        }
        qInQueue_.FreeTensor(q);

        const int32_t slotBytes = static_cast<int32_t>(scaleSlot_ * sizeof(float));
        AscendC::LocalTensor<int32_t> idx = scaleIdxBuf_.Get<int32_t>();
        AscendC::ArithProgression(idx, static_cast<int32_t>(kvHead * sizeof(float)), slotBytes,
                                  static_cast<int32_t>(kTileRows));
        AscendC::ArithProgression(idx[kTileRows], static_cast<int32_t>((numKvHeads_ + kvHead) * sizeof(float)),
                                  slotBytes, static_cast<int32_t>(kTileRows));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInTile(uint32_t physical, uint32_t kvHead, uint32_t rowBase)
    {
        AscendC::LocalTensor<int8_t> kv = kvQueue_.template AllocTensor<int8_t>();
        AscendC::LocalTensor<float> scales = scaleQueue_.template AllocTensor<float>();

        const uint64_t row = static_cast<uint64_t>(physical) * blockSize_ + rowBase;
        const uint64_t rowOff = row * packedPlane_;
        const uint32_t tileBytes = kTileRows * packedBytes_;

        if (burst_.WholeRow()) {
            SyncVectorToMte2();
        }
        burst_.Read(kv, keyCacheGm_, rowOff, kvHead, 0);
        burst_.Read(kv[tileBytes], valueCacheGm_, rowOff, kvHead, 1);
        if (burst_.WholeRow()) {
            SyncMte2ToVector();
            burst_.Permute(kv, kvHead, 0);
            burst_.Permute(kv[tileBytes], kvHead, 1);
        }

        AscendC::DataCopy(scales, scaleCacheGm_[row * scaleSlot_], kTileRows * scaleSlot_);

        kvQueue_.EnQue(kv);
        scaleQueue_.EnQue(scales);
    }

    __aicore__ inline void AccumulateTile(uint32_t valid, const AscendC::LocalTensor<float> &acc,
                                          const AscendC::LocalTensor<float> &runMax,
                                          const AscendC::LocalTensor<float> &runSum,
                                          const AscendC::LocalTensor<float> &newMax,
                                          const AscendC::LocalTensor<float> &alpha,
                                          const AscendC::LocalTensor<float> &tileMax)
    {
        AscendC::LocalTensor<int8_t> kv = kvQueue_.template DeQue<int8_t>();
        AscendC::LocalTensor<float> scaleTile = scaleQueue_.template DeQue<float>();

        AscendC::LocalTensor<float> kf = kvFloatBuf_.Get<float>();
        AscendC::LocalTensor<float> vf = kf[kTileRows * headSize_];
        AscendC::LocalTensor<float> prod = prodBuf_.Get<float>();
        AscendC::LocalTensor<float> qTile = qTileBuf_.Get<float>();

        AscendC::LocalTensor<float> rows = rowBuf_.Get<float>();
        AscendC::LocalTensor<float> scores = rows;
        AscendC::LocalTensor<float> part = rows[kTileRows];
        AscendC::LocalTensor<float> probs = rows[2 * kTileRows];
        AscendC::LocalTensor<float> reduceWork = rows[3 * kTileRows];
        AscendC::LocalTensor<float> kScale = rows[4 * kTileRows];
        AscendC::LocalTensor<float> vScale = rows[5 * kTileRows];

        AscendC::LocalTensor<float> brcb = brcbBuf_.Get<float>();
        AscendC::LocalTensor<float> bMax = brcb;
        AscendC::LocalTensor<float> bAlpha = brcb[kBrcbDstLanes];
        AscendC::LocalTensor<float> probBlocks = brcb[2 * kBrcbDstLanes];

        AscendC::LocalTensor<uint32_t> idx = scaleIdxBuf_.Get<int32_t>().ReinterpretCast<uint32_t>();
        constexpr uint32_t scaleBase = kGatherSrcBase;
        AscendC::Gather(kScale, scaleTile, idx, scaleBase, kTileRows);
        AscendC::Gather(vScale, scaleTile, idx[kTileRows], scaleBase, kTileRows);
        VecBarrier<kPagedAttentionVecBarriers>();

        codec_.Dequantize4Bit(kf, kv, static_cast<int>(kTileRows), static_cast<int>(headSize_));
        AscendC::Mul(prod, kf, qTile, kTileRows * headSize_);
        VecBarrier<kPagedAttentionVecBarriers>();
        RowSums(scores, part, prod);
        AscendC::Mul(scores, scores, kScale, kTileRows);
        VecBarrier<kPagedAttentionVecBarriers>();
        AscendC::Muls(scores, scores, scale_, kTileRows);
        VecBarrier<kPagedAttentionVecBarriers>();
        if (valid < kTileRows) {
            AscendC::Duplicate(scores[valid], kNegInf, kTileRows - valid);
            VecBarrier<kPagedAttentionVecBarriers>();
        }

        AscendC::ReduceMax<float>(tileMax, scores, reduceWork, kTileRows, false);
        VecBarrier<kPagedAttentionVecBarriers>();
        AscendC::Max(newMax, runMax, tileMax, 1);
        VecBarrier<kPagedAttentionVecBarriers>();
        AscendC::Sub(alpha, runMax, newMax, 1);
        VecBarrier<kPagedAttentionVecBarriers>();
        AscendC::Exp(alpha, alpha, 1);
        VecBarrier<kPagedAttentionVecBarriers>();

        BroadcastScalar<kPagedAttentionVecBarriers>(bMax, newMax);
        BroadcastSub<kPagedAttentionVecBarriers>(scores, scores, bMax, kTileRows);
        AscendC::Exp(probs, scores, kTileRows);
        VecBarrier<kPagedAttentionVecBarriers>();
        if (valid < kTileRows) {
            AscendC::Duplicate(probs[valid], 0.0f, kTileRows - valid);
            VecBarrier<kPagedAttentionVecBarriers>();
        }

        AscendC::WholeReduceSum<float>(part, probs, kTileRows, 1, 1, 1, kTileRows / kFp32PerBlock);
        VecBarrier<kPagedAttentionVecBarriers>();
        AscendC::Mul(runSum, runSum, alpha, 1);
        VecBarrier<kPagedAttentionVecBarriers>();
        AscendC::Add(runSum, runSum, part, 1);
        VecBarrier<kPagedAttentionVecBarriers>();

        BroadcastScalar<kPagedAttentionVecBarriers>(bAlpha, alpha);
        TurboQuantCodec4::BroadcastMul<kPagedAttentionVecBarriers>(acc, acc, bAlpha, headSize_);

        AscendC::Mul(probs, probs, vScale, kTileRows);
        VecBarrier<kPagedAttentionVecBarriers>();
        codec_.Dequantize4Bit(vf, kv[kTileRows * packedBytes_], static_cast<int>(kTileRows),
                              static_cast<int>(headSize_));
        AscendC::Brcb(probBlocks, probs, kTileRows / kFp32PerBlock, {1, static_cast<uint16_t>(kFp32PerBlock)});
        VecBarrier<kPagedAttentionVecBarriers>();

        constexpr uint8_t kOneBlock = 1;
        const uint8_t rowBlocks = static_cast<uint8_t>(headSize_ / kFp32PerBlock);
        for (uint32_t col = 0; col < headSize_; col += kFp32PerRepeat) {
            AscendC::Mul(prod[col], vf[col], probBlocks, static_cast<uint64_t>(kFp32PerRepeat),
                         static_cast<uint8_t>(kTileRows), {1, 1, 0, rowBlocks, rowBlocks, kOneBlock});
        }
        VecBarrier<kPagedAttentionVecBarriers>();

        for (uint32_t half = kTileRows / 2; half >= 1; half /= 2) {
            AscendC::Add(prod, prod, prod[half * headSize_], half * headSize_);
            VecBarrier<kPagedAttentionVecBarriers>();
        }
        AscendC::Add(acc, acc, prod, headSize_);
        VecBarrier<kPagedAttentionVecBarriers>();

        AscendC::Adds(runMax, newMax, 0.0f, 1);
        VecBarrier<kPagedAttentionVecBarriers>();

        kvQueue_.FreeTensor(kv);
        scaleQueue_.FreeTensor(scaleTile);
    }

    __aicore__ inline void RowSums(const AscendC::LocalTensor<float> &dst, const AscendC::LocalTensor<float> &part,
                                   const AscendC::LocalTensor<float> &src)
    {
        const uint8_t rowBlocks = static_cast<uint8_t>(headSize_ / kFp32PerBlock);
        AscendC::WholeReduceSum<float>(dst, src, kFp32PerRepeat, static_cast<uint8_t>(kTileRows), 1, 1, rowBlocks);
        VecBarrier<kPagedAttentionVecBarriers>();
        for (uint32_t col = kFp32PerRepeat; col < headSize_; col += kFp32PerRepeat) {
            AscendC::WholeReduceSum<float>(part, src[col], kFp32PerRepeat, static_cast<uint8_t>(kTileRows), 1, 1,
                                           rowBlocks);
            VecBarrier<kPagedAttentionVecBarriers>();
            AscendC::Add(dst, dst, part, kTileRows);
            VecBarrier<kPagedAttentionVecBarriers>();
        }
    }

    AscendC::TPipe *pipe_;
    TurboQuantCodec4 codec_;
    TurboQuantTileBurst burst_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> kvQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> scaleQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qInQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> kvFloatBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> qTileBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> prodBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rowBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> brcbBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stateBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleIdxBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outBuf_;
    AscendC::GlobalTensor<float> queryRotGm_;
    AscendC::GlobalTensor<int8_t> keyCacheGm_;
    AscendC::GlobalTensor<int8_t> valueCacheGm_;
    AscendC::GlobalTensor<float> scaleCacheGm_;
    AscendC::GlobalTensor<int32_t> blockTableGm_;
    AscendC::GlobalTensor<int32_t> contextLenGm_;
    AscendC::GlobalTensor<int32_t> tablesGm_;
    AscendC::GlobalTensor<float> workspaceGm_;
    AscendC::GlobalTensor<scalar_t> outputGm_;
    uint32_t numTokens_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlocksPerSeq_ = 0;
    uint32_t numSplits_ = 1;
    uint32_t fusedContextLimit_ = 0;
    uint32_t headsPerKv_ = 1;
    uint32_t packedBytes_ = 0;
    uint32_t packedPlane_ = 0;
    uint32_t scaleSlot_ = 0;
    uint32_t partialStride_ = 0;
    float scale_ = 1.0f;
};

}

#define TURBOQUANT_RESHAPE_AND_CACHE_DECLARE(TYPE)                                                                   \
    extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_##TYPE(                                       \
        GM_ADDR key, GM_ADDR value, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR slotMapping,   \
        GM_ADDR piSigns, GM_ADDR tables, uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,                 \
        uint32_t blockSize, uint32_t tokensPerCore, float invSqrtLen)                                                \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantReshapeAndCache<TYPE> op(&pipe);                                                                   \
        op.Init(key, value, keyCache, valueCache, scaleCache, slotMapping, piSigns, tables, numTokens, numKvHeads,   \
                headSize, blockSize, tokensPerCore, invSqrtLen);                                                     \
        op.Process();                                                                                                \
    }

#define TURBOQUANT_PAGED_ATTENTION_FUSED_DECLARE(TYPE)                                                               \
    extern "C" __global__ __aicore__ void turboquant_paged_attention_fused_##TYPE(                                   \
        GM_ADDR queryRot, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR blockTables,             \
        GM_ADDR contextLens, GM_ADDR tables, GM_ADDR workspace, GM_ADDR output, uint32_t numTokens,                  \
        uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,     \
        uint32_t numSplits, uint32_t splitTasksPerCore, uint32_t reduceTasksPerCore, uint32_t fusedContextLimit,     \
        float scale, float invSqrtLen)                                                                               \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantPagedAttentionSplit<TYPE> split(&pipe);                                                            \
        split.Init(queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, tables, workspace, output,  \
                   numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,                 \
                   fusedContextLimit, scale, invSqrtLen);                                                            \
        split.Process(splitTasksPerCore);                                                                            \
        if (split.NeedsReduction()) {                                                                                \
            AscendC::SyncAll<true>();                                                                                \
            TurboQuantPartialReducer<TYPE, kPagedAttentionVecBarriers> reducer(&pipe);                               \
            reducer.Init(workspace, output, numHeads, headSize, numSplits);                                          \
            split.Reduce(reducer, reduceTasksPerCore);                                                               \
        }                                                                                                            \
    }

TURBOQUANT_RESHAPE_AND_CACHE_DECLARE(half)
TURBOQUANT_PAGED_ATTENTION_FUSED_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
TURBOQUANT_RESHAPE_AND_CACHE_DECLARE(bfloat16_t)
TURBOQUANT_PAGED_ATTENTION_FUSED_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {

void turboquant_reshape_and_cache_impl(AscendType type, void *stream, uint32_t blockDim, void *key, void *value,
                                       void *keyCache, void *valueCache, void *scaleCache, void *slotMapping,
                                       void *piSigns, void *tables, uint32_t numTokens, uint32_t numKvHeads,
                                       uint32_t headSize, uint32_t blockSize, uint32_t tokensPerCore,
                                       float invSqrtLen)
{
    if (type == AscendType::FP16) {
        turboquant_reshape_and_cache_half<<<blockDim, nullptr, stream>>>(
            key, value, keyCache, valueCache, scaleCache, slotMapping, piSigns, tables, numTokens, numKvHeads,
            headSize, blockSize, tokensPerCore, invSqrtLen);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    } else if (type == AscendType::BF16) {
        turboquant_reshape_and_cache_bfloat16_t<<<blockDim, nullptr, stream>>>(
            key, value, keyCache, valueCache, scaleCache, slotMapping, piSigns, tables, numTokens, numKvHeads,
            headSize, blockSize, tokensPerCore, invSqrtLen);
#endif
    }
}

void turboquant_paged_attention_impl(AscendType type, void *stream, uint32_t blockDim, void *queryRot,
                                     void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
                                     void *contextLens, void *tables, void *workspace, void *output,
                                     uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize,
                                     uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,
                                     uint32_t splitTasksPerCore, uint32_t reduceTasksPerCore,
                                     uint32_t fusedContextLimit, float scale, float invSqrtLen)
{
    if (type == AscendType::FP16) {
        turboquant_paged_attention_fused_half<<<blockDim, nullptr, stream>>>(
            queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, tables, workspace, output,
            numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, splitTasksPerCore,
            reduceTasksPerCore, fusedContextLimit, scale, invSqrtLen);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    } else if (type == AscendType::BF16) {
        turboquant_paged_attention_fused_bfloat16_t<<<blockDim, nullptr, stream>>>(
            queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, tables, workspace, output,
            numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, splitTasksPerCore,
            reduceTasksPerCore, fusedContextLimit, scale, invSqrtLen);
#endif
    }
}

}
