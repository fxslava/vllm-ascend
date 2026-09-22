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
using vllm_ascend::turboquant::AlignUp;
using vllm_ascend::turboquant::CeilDiv;
using vllm_ascend::turboquant::kBrcbDstLanes;
using vllm_ascend::turboquant::kFp32PerBlock;
using vllm_ascend::turboquant::kFp32PerRepeat;
using vllm_ascend::turboquant::kGatherSrcBase;
using vllm_ascend::turboquant::kPartialMaxLane;
using vllm_ascend::turboquant::kPartialSumLane;
using vllm_ascend::turboquant::kPartialTail;
using vllm_ascend::turboquant::ScaleSlotFloats;
using vllm_ascend::turboquant::SyncMte2ToVector;
using vllm_ascend::turboquant::SyncMte3ToVector;
using vllm_ascend::turboquant::SyncScalarToVector;
using vllm_ascend::turboquant::SyncVectorToMte2;
using vllm_ascend::turboquant::SyncVectorToMte3;
using vllm_ascend::turboquant::TurboQuantCodec4;
using vllm_ascend::turboquant::TurboQuantPartialReducer;
using vllm_ascend::turboquant::TurboQuantTileBurst;
using vllm_ascend::turboquant::WriteNormalizedHeads;

constexpr uint32_t kTileRows = vllm_ascend::turboquant::kAivTileRows;
// The logit a row past the end of a partial tile is masked to.
constexpr float kMaskedLogit = -1.0e30f;
constexpr uint32_t kSlotRing = 4;

// Mask the tile lanes at and past `valid`, without ever addressing UB off a 32-byte boundary.
//
// `tile[valid]` is a UB address at byte offset `sizeof(float) * valid`, and the AIV refuses a
// vector access there unless it is a whole 32-byte block: "the address for VEC to access UB is
// not aligned" (error 340, subErrType 0x4). `valid` is the ragged tail of a context length, so
// it is a multiple of kFp32PerBlock only by luck -- a 57-token context leaves 9 rows in the
// last tile, and 9 floats is 36 bytes.
//
// So the vector Duplicate starts at the next whole block, and the handful of lanes below it go
// through the scalar unit, which addresses UB by element and has no such rule. That costs a
// hand-off in each direction, but only on a tile whose tail is ragged -- at most one per block
// of context, not one per tile, and an aligned context still takes the single Duplicate it
// always did.
//
// Everywhere else in these kernels a per-lane offset is strided by kFp32PerBlock precisely so
// the question never arises (the query-scale loop in turboquant_vector_service.h), and the Cube
// decode masks its own tail without indexing at all: ComputeTailMask there builds a whole-tile
// +huge/-huge mask arithmetically from a lane progression and Mins the scores with it. That is
// the better shape where it applies, but it does not cover masking `probs` to zero -- which this
// kernel needs, because a tile whose rows are all masked would otherwise leave exp(0) = 1 behind.
// These two tail masks were the only places in any of the kernels indexing UB by a row count.
__aicore__ inline void MaskTailLanes(const AscendC::LocalTensor<float> &tile, const uint32_t valid,
                                     const float value)
{
    if (valid >= kTileRows) {
        return;
    }
    const uint32_t aligned = AlignUp(valid, kFp32PerBlock);
    if (aligned > valid) {
        // Vector -> scalar: the lanes about to be overwritten were just computed by the vector pipe.
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t lane = valid; lane < aligned; ++lane) {
            tile.SetValue(lane, value);
        }
        // Scalar -> vector: the reductions that follow read the lanes the scalar unit just wrote.
        SyncScalarToVector();
    }
    if (aligned < kTileRows) {
        AscendC::Duplicate(tile[aligned], value, static_cast<int32_t>(kTileRows - aligned));
    }
}

template <typename scalar_t>
class TurboQuantReshapeAndCache {
public:
    __aicore__ inline explicit TurboQuantReshapeAndCache(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *key, __gm__ void *value, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *scaleCache, __gm__ void *slotMapping, __gm__ void *piSigns,
                                __gm__ void *tables, const uint32_t numTokens, const uint32_t numKvHeads,
                                const uint32_t headSize, const uint32_t blockSize, const uint32_t numBlocks,
                                const uint32_t tokensPerCore, const float invSqrtLen)
    {
        numTokens_ = numTokens;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        blockSize_ = blockSize;
        numSlots_ = static_cast<uint64_t>(numBlocks) * blockSize;
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
        pipe_->InitBuffer(scaleIndexBuf_, 2 * numKvHeads_ * sizeof(int32_t));

        codec_.Init(pipe_, headSize_, 1, invSqrtLen, tablesGm_);

        const AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::DataCopy(signs, piSignsGm_, headSize_);

        const AscendC::LocalTensor<int32_t> scaleIndex = scaleIndexBuf_.Get<int32_t>();
        AscendC::ArithProgression(scaleIndex, 0, static_cast<int32_t>(kFp32PerBlock * sizeof(float)),
                                  static_cast<int32_t>(2 * numKvHeads_));
        // Init hand-off: the signs and the codec tables land in UB before the first token's rotation.
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

        StageTokenIn(start, 0);
        for (uint32_t step = 0; step < total; ++step) {
            if (step + 1 < total) {
                StageTokenIn(start + step + 1, step + 1);
            }
            ComputeToken();
            if (step > 0) {
                StageTokenOut(step - 1);
            }
        }
        StageTokenOut(total - 1);
    }

private:
    __aicore__ inline void StageTokenIn(const uint32_t token, const uint32_t step)
    {
        const int32_t slot = slotGm_.GetValue(token);
        const uint32_t ring = step % kSlotRing;
        // A padding token carries a negative slot, and a slot past the last row of the cache is a
        // caller bug. Both are dropped here: the three DataCopy destinations in StageTokenOut are
        // scaled by the slot, so letting either through is a write to an address outside the cache
        // -- an AI core "address for scalar to access GM is invalid" (264) at best, and silent
        // corruption of whatever else that address belongs to at worst.
        if (slot < 0 || static_cast<uint64_t>(slot) >= numSlots_) {
            slotValid_[ring] = false;
            cacheOffset_[ring] = 0;
            scaleOffset_[ring] = 0;
        } else {
            const uint64_t row = static_cast<uint64_t>(slot);
            slotValid_[ring] = true;
            cacheOffset_[ring] = row * packedPlane_;
            scaleOffset_[ring] = row * scaleSlot_;
        }

        const AscendC::LocalTensor<scalar_t> in = inQueue_.template AllocTensor<scalar_t>();
        const uint64_t offset = static_cast<uint64_t>(token) * headPlane_;
        AscendC::DataCopy(in, keyGm_[offset], headPlane_);
        AscendC::DataCopy(in[headPlane_], valueGm_[offset], headPlane_);
        inQueue_.EnQue(in);
    }

    // Rotates and quantises every (plane, kv head) vector of one token, then gathers its per-vector scales
    // into the token's scale slot.
    __aicore__ inline void ComputeToken()
    {
        AscendC::LocalTensor<scalar_t> in = inQueue_.template DeQue<scalar_t>();
        const AscendC::LocalTensor<int8_t> packed = outPacked_.template AllocTensor<int8_t>();
        const AscendC::LocalTensor<float> scales = outScale_.template AllocTensor<float>();

        const AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        const AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        const AscendC::LocalTensor<float> steps = stepBuf_.Get<float>();
        const AscendC::LocalTensor<float> vec = work;
        const AscendC::LocalTensor<float> tmp = work[headSize_];

        for (uint32_t plane = 0; plane < 2 * numKvHeads_; ++plane) {
            AscendC::Cast(vec, in[plane * headSize_], AscendC::RoundMode::CAST_NONE, headSize_);
            codec_.ApplyPi(vec, tmp, signs, headSize_);
            codec_.Quantize4Bit(packed[plane * packedBytes_], vec, steps[plane * kFp32PerBlock], headSize_);
        }

        const AscendC::LocalTensor<uint32_t> scaleIndex = scaleIndexBuf_.Get<int32_t>().ReinterpretCast<uint32_t>();
        AscendC::Duplicate(scales, 0.0f, scaleSlot_);
        // Overlapping write: the Gather overwrites lanes the Duplicate just zeroed.
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(scales, steps, scaleIndex, kGatherSrcBase, 2 * numKvHeads_);

        inQueue_.FreeTensor(in);
        outPacked_.EnQue(packed);
        outScale_.EnQue(scales);
    }

    __aicore__ inline void StageTokenOut(const uint32_t step)
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
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleIndexBuf_;
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
    uint64_t numSlots_ = 0;
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
                                const uint32_t numTokens, const uint32_t numHeads, const uint32_t numKvHeads,
                                const uint32_t headSize, const uint32_t blockSize, const uint32_t maxBlocksPerSeq,
                                const uint32_t numSplits, const uint32_t fusedContextLimit, const float scale,
                                const float invSqrtLen)
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
        pipe_->InitBuffer(queryInQueue_, 1, headSize_ * sizeof(float));

        pipe_->InitBuffer(kvFloatBuf_, 2 * tileElems * sizeof(float));
        pipe_->InitBuffer(queryTileBuf_, tileElems * sizeof(float));
        pipe_->InitBuffer(productBuf_, tileElems * sizeof(float));
        pipe_->InitBuffer(accBuf_, headSize_ * sizeof(float));
        pipe_->InitBuffer(rowBuf_, 6 * kTileRows * sizeof(float));
        pipe_->InitBuffer(blockBuf_, 2 * kBrcbDstLanes * sizeof(float) + kTileRows * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(stateBuf_, kStateLanes * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(scaleIndexBuf_, 2 * kTileRows * sizeof(int32_t));

        codec_.Init(pipe_, headSize_, kTileRows, invSqrtLen, tablesGm_);
        pipe_->InitBuffer(outBuf_, headSize_ * sizeof(scalar_t));
        burst_.Init(pipe_, numKvHeads_, packedBytes_, kTileRows, 2);
        // Init hand-off: the codec tables land in UB before the first split's vector work.
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

    __aicore__ inline void Process(const uint32_t tasksPerCore)
    {
        const uint32_t tasks = numTokens_ * numHeads_ * numSplits_;
        const uint32_t start = static_cast<uint32_t>(AscendC::GetBlockIdx()) * tasksPerCore;
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
    __aicore__ inline void Reduce(TurboQuantPartialReducer<scalar_t> &reducer,
                                  const uint32_t tasksPerCore)
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
    static constexpr uint32_t kStateLanes = 5;

    // The online-softmax state of one split, one kFp32PerBlock lane each; the first two are the partial tail.
    struct SoftmaxState {
        AscendC::LocalTensor<float> runMax;
        AscendC::LocalTensor<float> runSum;
        AscendC::LocalTensor<float> newMax;
        AscendC::LocalTensor<float> alpha;
        AscendC::LocalTensor<float> tileMax;
    };

    __aicore__ inline uint64_t PartialOffset(const uint32_t token, const uint32_t head, const uint32_t split) const
    {
        return ((static_cast<uint64_t>(token) * numHeads_ + head) * numSplits_ + split) * partialStride_;
    }

    __aicore__ inline bool IsFused(const int32_t contextLen) const
    {
        return numSplits_ <= 1 || contextLen <= static_cast<int32_t>(fusedContextLimit_);
    }

    __aicore__ inline void ComputeSplit(const uint32_t token, const uint32_t head, const uint32_t split)
    {
        const int32_t contextLen = contextLenGm_.GetValue(token);
        const bool fused = IsFused(contextLen);
        if (fused && split > 0) {
            return;
        }

        const AscendC::LocalTensor<float> stateLanes = stateBuf_.Get<float>();
        SoftmaxState state;
        state.runMax = stateLanes[kPartialMaxLane];
        state.runSum = stateLanes[kPartialSumLane];
        state.newMax = stateLanes[2 * kFp32PerBlock];
        state.alpha = stateLanes[3 * kFp32PerBlock];
        state.tileMax = stateLanes[4 * kFp32PerBlock];

        const AscendC::LocalTensor<float> acc = accBuf_.Get<float>();

        AscendC::Duplicate(acc, 0.0f, headSize_);
        AscendC::Duplicate(stateLanes, 0.0f, kStateLanes * kFp32PerBlock);
        // Overlapping write: runMax lies inside the state the Duplicate above just zeroed.
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(state.runMax, kMaskedLogit, 1);

        const uint32_t seqBlocks = contextLen > 0 ? CeilDiv(static_cast<uint32_t>(contextLen), blockSize_) : 0;
        const uint32_t blocksPerSplit = CeilDiv(seqBlocks, fused ? 1u : numSplits_);
        const uint32_t blockStart = split * blocksPerSplit;
        uint32_t blockEnd = blockStart + blocksPerSplit;
        if (blockEnd > seqBlocks) {
            blockEnd = seqBlocks;
        }

        if (blockStart < blockEnd) {
            const uint32_t kvHead = head / headsPerKv_;
            StageQuery(token, head, kvHead);
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

                StageTileIn(static_cast<uint32_t>(physical), kvHead, 0);
                for (uint32_t base = 0; base < rows; base += kTileRows) {
                    const uint32_t next = base + kTileRows;
                    if (next < rows) {
                        StageTileIn(static_cast<uint32_t>(physical), kvHead, next);
                    }
                    uint32_t valid = rows - base;
                    if (valid > kTileRows) {
                        valid = kTileRows;
                    }
                    ComputeTile(valid, acc, state);
                }
            }
        }

        if (fused) {
            StageOutput(token, head, acc, state.runSum);
            return;
        }
        const uint64_t offset = PartialOffset(token, head, split);
        SyncVectorToMte3();
        AscendC::DataCopy(workspaceGm_[offset], acc, headSize_);
        AscendC::DataCopy(workspaceGm_[offset + headSize_], stateLanes, kPartialTail);
        SyncMte3ToVector();
    }

    __aicore__ inline void StageOutput(const uint32_t token, const uint32_t head, const AscendC::LocalTensor<float> &acc,
                                       const AscendC::LocalTensor<float> &runSum)
    {
        const AscendC::LocalTensor<float> blocks = blockBuf_.Get<float>();
        WriteNormalizedHeads<scalar_t>(
            acc, runSum, 1, headSize_, blocks[2 * kBrcbDstLanes], blocks[3 * kBrcbDstLanes], outBuf_.Get<scalar_t>(),
            outputGm_, (static_cast<uint64_t>(token) * numHeads_ + head) * headSize_);
    }

    // The rotated query of (token, head) tiled over kTileRows rows, and the tile scale gather index of its
    // kv head.
    __aicore__ inline void StageQuery(const uint32_t token, const uint32_t head, const uint32_t kvHead)
    {
        AscendC::LocalTensor<float> query = queryInQueue_.template AllocTensor<float>();
        AscendC::DataCopy(query, queryRotGm_[(static_cast<uint64_t>(token) * numHeads_ + head) * headSize_],
                          headSize_);
        queryInQueue_.EnQue(query);
        query = queryInQueue_.template DeQue<float>();

        const AscendC::LocalTensor<float> queryTile = queryTileBuf_.Get<float>();
        for (uint32_t row = 0; row < kTileRows; ++row) {
            AscendC::DataCopy(queryTile[row * headSize_], query, headSize_);
        }
        queryInQueue_.FreeTensor(query);

        const int32_t slotBytes = static_cast<int32_t>(scaleSlot_ * sizeof(float));
        const AscendC::LocalTensor<int32_t> scaleIndex = scaleIndexBuf_.Get<int32_t>();
        AscendC::ArithProgression(scaleIndex, static_cast<int32_t>(kvHead * sizeof(float)), slotBytes,
                                  static_cast<int32_t>(kTileRows));
        AscendC::ArithProgression(scaleIndex[kTileRows], static_cast<int32_t>((numKvHeads_ + kvHead) * sizeof(float)),
                                  slotBytes, static_cast<int32_t>(kTileRows));
        // Task boundary: the query tile and gather index are staged before this task's tile queue starts.
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StageTileIn(const uint32_t physical, const uint32_t kvHead, const uint32_t rowBase)
    {
        const AscendC::LocalTensor<int8_t> kv = kvQueue_.template AllocTensor<int8_t>();
        const AscendC::LocalTensor<float> scales = scaleQueue_.template AllocTensor<float>();

        const uint64_t row = static_cast<uint64_t>(physical) * blockSize_ + rowBase;
        const uint64_t rowOffset = row * packedPlane_;
        const uint32_t tileBytes = kTileRows * packedBytes_;

        if (burst_.ReadsWholeRows()) {
            SyncVectorToMte2();
        }
        burst_.ReadTileRows(kv, keyCacheGm_, rowOffset, kvHead, 0);
        burst_.ReadTileRows(kv[tileBytes], valueCacheGm_, rowOffset, kvHead, 1);
        if (burst_.ReadsWholeRows()) {
            SyncMte2ToVector();
            burst_.SelectKvHeadRows(kv, kvHead, 0);
            burst_.SelectKvHeadRows(kv[tileBytes], kvHead, 1);
        }

        AscendC::DataCopy(scales, scaleCacheGm_[row * scaleSlot_], kTileRows * scaleSlot_);

        kvQueue_.EnQue(kv);
        scaleQueue_.EnQue(scales);
    }

    // One tile of the online softmax: scores from the dequantised keys, the running max / log-sum update,
    // and the probability-weighted values added into acc.
    __aicore__ inline void ComputeTile(const uint32_t valid, const AscendC::LocalTensor<float> &acc,
                                       const SoftmaxState &state)
    {
        AscendC::LocalTensor<int8_t> kv = kvQueue_.template DeQue<int8_t>();
        AscendC::LocalTensor<float> scaleTile = scaleQueue_.template DeQue<float>();

        const AscendC::LocalTensor<float> rows = rowBuf_.Get<float>();
        const AscendC::LocalTensor<float> probs = rows[2 * kTileRows];
        const AscendC::LocalTensor<float> valueScale = rows[5 * kTileRows];

        ComputeTileScores(valid, kv, scaleTile);
        ComputeSoftmaxStep(valid, acc, state);

        AscendC::Mul(probs, probs, valueScale, kTileRows);
        ComputeContext(acc, kv[kTileRows * packedBytes_]);

        AscendC::Adds(state.runMax, state.newMax, 0.0f, 1);

        kvQueue_.FreeTensor(kv);
        scaleQueue_.FreeTensor(scaleTile);
    }

    // scores = (q . k) * kScale * scale per tile row, the rows past `valid` masked.
    __aicore__ inline void ComputeTileScores(const uint32_t valid, const AscendC::LocalTensor<int8_t> &kv,
                                             const AscendC::LocalTensor<float> &scaleTile)
    {
        const AscendC::LocalTensor<float> keyFloat = kvFloatBuf_.Get<float>();
        const AscendC::LocalTensor<float> product = productBuf_.Get<float>();
        const AscendC::LocalTensor<float> queryTile = queryTileBuf_.Get<float>();
        const AscendC::LocalTensor<float> rows = rowBuf_.Get<float>();
        const AscendC::LocalTensor<float> scores = rows;
        const AscendC::LocalTensor<float> part = rows[kTileRows];
        const AscendC::LocalTensor<float> keyScale = rows[4 * kTileRows];
        const AscendC::LocalTensor<float> valueScale = rows[5 * kTileRows];

        const AscendC::LocalTensor<uint32_t> scaleIndex = scaleIndexBuf_.Get<int32_t>().ReinterpretCast<uint32_t>();
        AscendC::Gather(keyScale, scaleTile, scaleIndex, kGatherSrcBase, kTileRows);
        AscendC::Gather(valueScale, scaleTile, scaleIndex[kTileRows], kGatherSrcBase, kTileRows);

        codec_.Dequantize4Bit(keyFloat, kv, kTileRows, headSize_);
        AscendC::Mul(product, keyFloat, queryTile, kTileRows * headSize_);
        ComputeRowSums(scores, part, product);
        AscendC::Mul(scores, scores, keyScale, kTileRows);
        AscendC::Muls(scores, scores, scale_, kTileRows);
        MaskTailLanes(scores, valid, kMaskedLogit);
    }

    // newMax, alpha = exp(runMax - newMax), probs = exp(scores - newMax), runSum = runSum * alpha + sum(probs),
    // and acc decayed by alpha.
    __aicore__ inline void ComputeSoftmaxStep(const uint32_t valid, const AscendC::LocalTensor<float> &acc,
                                              const SoftmaxState &state)
    {
        const AscendC::LocalTensor<float> rows = rowBuf_.Get<float>();
        const AscendC::LocalTensor<float> scores = rows;
        const AscendC::LocalTensor<float> part = rows[kTileRows];
        const AscendC::LocalTensor<float> probs = rows[2 * kTileRows];
        const AscendC::LocalTensor<float> reduceWork = rows[3 * kTileRows];

        const AscendC::LocalTensor<float> blocks = blockBuf_.Get<float>();
        const AscendC::LocalTensor<float> newMaxBlock = blocks;
        const AscendC::LocalTensor<float> alphaBlock = blocks[kBrcbDstLanes];

        AscendC::ReduceMax<float>(state.tileMax, scores, reduceWork, kTileRows, false);
        AscendC::Max(state.newMax, state.runMax, state.tileMax, 1);
        AscendC::Sub(state.alpha, state.runMax, state.newMax, 1);
        AscendC::Exp(state.alpha, state.alpha, 1);

        BroadcastScalar(newMaxBlock, state.newMax);
        BroadcastSub(scores, scores, newMaxBlock, kTileRows);
        AscendC::Exp(probs, scores, kTileRows);
        MaskTailLanes(probs, valid, 0.0f);

        AscendC::WholeReduceSum<float>(part, probs, kTileRows, 1, 1, 1, kTileRows / kFp32PerBlock);
        AscendC::Mul(state.runSum, state.runSum, state.alpha, 1);
        AscendC::Add(state.runSum, state.runSum, part, 1);

        BroadcastScalar(alphaBlock, state.alpha);
        TurboQuantCodec4::BroadcastMul(acc, acc, alphaBlock, headSize_);
    }

    // acc += sum over tile rows of probs[row] * V[row], with probs already scaled by the value scales.
    __aicore__ inline void ComputeContext(const AscendC::LocalTensor<float> &acc,
                                          const AscendC::LocalTensor<int8_t> &valuePacked)
    {
        const AscendC::LocalTensor<float> keyFloat = kvFloatBuf_.Get<float>();
        const AscendC::LocalTensor<float> valueFloat = keyFloat[kTileRows * headSize_];
        const AscendC::LocalTensor<float> product = productBuf_.Get<float>();
        const AscendC::LocalTensor<float> probs = rowBuf_.Get<float>()[2 * kTileRows];
        const AscendC::LocalTensor<float> probBlocks = blockBuf_.Get<float>()[2 * kBrcbDstLanes];

        codec_.Dequantize4Bit(valueFloat, valuePacked, kTileRows, headSize_);
        AscendC::Brcb(probBlocks, probs, kTileRows / kFp32PerBlock, {1, static_cast<uint16_t>(kFp32PerBlock)});

        constexpr uint8_t kOneBlock = 1;
        const uint8_t rowBlocks = static_cast<uint8_t>(headSize_ / kFp32PerBlock);
        for (uint32_t col = 0; col < headSize_; col += kFp32PerRepeat) {
            AscendC::Mul(product[col], valueFloat[col], probBlocks, static_cast<uint64_t>(kFp32PerRepeat),
                         static_cast<uint8_t>(kTileRows), {1, 1, 0, rowBlocks, rowBlocks, kOneBlock});
        }

        for (uint32_t halfRows = kTileRows / 2; halfRows >= 1; halfRows /= 2) {
            AscendC::Add(product, product, product[halfRows * headSize_], halfRows * headSize_);
        }
        AscendC::Add(acc, acc, product, headSize_);
    }

    // dst[row] = sum over columns of src[row, col], one 64-lane block reduce at a time.
    __aicore__ inline void ComputeRowSums(const AscendC::LocalTensor<float> &dst,
                                          const AscendC::LocalTensor<float> &part,
                                          const AscendC::LocalTensor<float> &src)
    {
        const uint8_t rowBlocks = static_cast<uint8_t>(headSize_ / kFp32PerBlock);
        AscendC::WholeReduceSum<float>(dst, src, kFp32PerRepeat, static_cast<uint8_t>(kTileRows), 1, 1, rowBlocks);
        for (uint32_t col = kFp32PerRepeat; col < headSize_; col += kFp32PerRepeat) {
            AscendC::WholeReduceSum<float>(part, src[col], kFp32PerRepeat, static_cast<uint8_t>(kTileRows), 1, 1,
                                           rowBlocks);
            AscendC::Add(dst, dst, part, kTileRows);
        }
    }

    AscendC::TPipe *pipe_;
    TurboQuantCodec4 codec_;
    TurboQuantTileBurst burst_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> kvQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> scaleQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> queryInQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> kvFloatBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> queryTileBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> productBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rowBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> blockBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stateBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleIndexBuf_;
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

#define ASCEND_TQ_DECLARE_RESHAPE_AND_CACHE(TYPE)                                                                    \
    extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_##TYPE(                                       \
        GM_ADDR key, GM_ADDR value, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR slotMapping,   \
        GM_ADDR piSigns, GM_ADDR tables, uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,                 \
        uint32_t blockSize, uint32_t numBlocks, uint32_t tokensPerCore, float invSqrtLen)                            \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantReshapeAndCache<TYPE> op(&pipe);                                                                   \
        op.Init(key, value, keyCache, valueCache, scaleCache, slotMapping, piSigns, tables, numTokens, numKvHeads,   \
                headSize, blockSize, numBlocks, tokensPerCore, invSqrtLen);                                          \
        op.Process();                                                                                                \
    }

#define ASCEND_TQ_DECLARE_PAGED_ATTENTION_FUSED(TYPE)                                                                \
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
            TurboQuantPartialReducer<TYPE> reducer(&pipe);                                                           \
            reducer.Init(workspace, output, numHeads, headSize, numSplits);                                          \
            split.Reduce(reducer, reduceTasksPerCore);                                                               \
        }                                                                                                            \
    }

ASCEND_TQ_DECLARE_RESHAPE_AND_CACHE(half)
ASCEND_TQ_DECLARE_PAGED_ATTENTION_FUSED(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
ASCEND_TQ_DECLARE_RESHAPE_AND_CACHE(bfloat16_t)
ASCEND_TQ_DECLARE_PAGED_ATTENTION_FUSED(bfloat16_t)
#endif

#undef ASCEND_TQ_DECLARE_PAGED_ATTENTION_FUSED
#undef ASCEND_TQ_DECLARE_RESHAPE_AND_CACHE

namespace vllm_ascend {

void turboquant_reshape_and_cache_impl(AscendType type, void *stream, uint32_t blockDim, void *key, void *value,
                                       void *keyCache, void *valueCache, void *scaleCache, void *slotMapping,
                                       void *piSigns, void *tables, uint32_t numTokens, uint32_t numKvHeads,
                                       uint32_t headSize, uint32_t blockSize, uint32_t numBlocks,
                                       uint32_t tokensPerCore, float invSqrtLen)
{
    if (type == AscendType::FP16) {
        turboquant_reshape_and_cache_half<<<blockDim, nullptr, stream>>>(
            key, value, keyCache, valueCache, scaleCache, slotMapping, piSigns, tables, numTokens, numKvHeads,
            headSize, blockSize, numBlocks, tokensPerCore, invSqrtLen);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    } else if (type == AscendType::BF16) {
        turboquant_reshape_and_cache_bfloat16_t<<<blockDim, nullptr, stream>>>(
            key, value, keyCache, valueCache, scaleCache, slotMapping, piSigns, tables, numTokens, numKvHeads,
            headSize, blockSize, numBlocks, tokensPerCore, invSqrtLen);
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
