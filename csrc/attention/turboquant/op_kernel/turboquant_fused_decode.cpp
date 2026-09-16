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

// The Cube-native multi-mode TurboQuant kernels: the mode-parameterised cache write, the fused
// single-launch decode, and the fp8 GEMM probe the Cube contract test drives. Test-only today: the
// wheel's library builds turboquant_paged_attention.cpp and turboquant_rotate_q.cpp.

#include "kernel_operator.h"

#include "../../../kernels/types.h"
#include "common/turboquant_codec_mx.h"
#include "cube/turboquant_cube_service.h"
#include "vector/turboquant_vector_service.h"

using vllm_ascend::turboquant::CeilDiv;
using vllm_ascend::turboquant::kCubeSlots;
using vllm_ascend::turboquant::kCubeTileM;
using vllm_ascend::turboquant::kCubeTileRows;
using vllm_ascend::turboquant::kFlagContextReady;
using vllm_ascend::turboquant::kFlagOperandsReady;
using vllm_ascend::turboquant::kFlagProbsReady;
using vllm_ascend::turboquant::kFlagProductReady;
using vllm_ascend::turboquant::kFlagScoresReady;
using vllm_ascend::turboquant::kFlagSlotFree;
using vllm_ascend::turboquant::kFlagSlotReady;
using vllm_ascend::turboquant::kFp32PerBlock;
using vllm_ascend::turboquant::kGatherSrcBase;
using vllm_ascend::turboquant::kVectorSubcoresPerBlock;
using vllm_ascend::turboquant::MixBlockIdx;
using vllm_ascend::turboquant::ScaleSlotFloats;
using vllm_ascend::turboquant::SyncVectorToMte2;
using vllm_ascend::turboquant::TurboQuantCodec4;
using vllm_ascend::turboquant::TurboQuantCubeDecodeService;
using vllm_ascend::turboquant::TurboQuantCubeMm;
using vllm_ascend::turboquant::TurboQuantMode;
using vllm_ascend::turboquant::TurboQuantModeCodec;
using vllm_ascend::turboquant::TurboQuantPartialReducer;
using vllm_ascend::turboquant::TurboQuantTaskHeads;
using vllm_ascend::turboquant::TurboQuantVectorDecodeService;

namespace {

// Intra-pipe vector barriers of the fused decode. The barriered instance is compiled once more, for
// kv4fp8 only, as the bit-exact A/B reference (turboquant_mm_fused_decode_barriered_impl).
constexpr bool kFusedVecBarriers = false;

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
        AscendC::Gather(scaleOut, steps, idx, kGatherSrcBase, 2 * numKvHeads_);
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

// One launch per decode step on the arch35 Cube. A task is (token, kv head, sequence split, head
// chunk). For every tile of a task the two AIV subcores stage K (subcore 0) and V (subcore 1) into L1;
// the AIC runs the score and context GEMMs into both subcores' UB (TurboQuantCubeDecodeService); each
// subcore runs the online softmax for its half of the heads (TurboQuantVectorDecodeService). A token
// whose context fits fusedContextLimit writes its output straight to GM; a longer one leaves
// partials that the same launch reduces after AscendC::SyncAll.
template <TurboQuantMode MODE, typename scalar_t, bool VEC_BARRIERS = kFusedVecBarriers>
class TurboQuantFusedDecode {
public:
    using Mm = TurboQuantCubeMm<MODE>;
    using Codec = TurboQuantModeCodec<MODE>;
    using Vector = TurboQuantVectorDecodeService<MODE, scalar_t, VEC_BARRIERS, Mm, Codec>;
    using Cube = TurboQuantCubeDecodeService<MODE>;
    using Reducer = TurboQuantPartialReducer<scalar_t, VEC_BARRIERS>;

    __aicore__ inline explicit TurboQuantFusedDecode(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *queryRot, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *scaleCache, __gm__ void *blockTables, __gm__ void *contextLens,
                                __gm__ void *modeTables, __gm__ void *workspace, __gm__ void *output,
                                uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize,
                                uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,
                                uint32_t headsPerTask, uint32_t fusedContextLimit, float scale, float invSqrtLen)
    {
        numTokens_ = numTokens;
        numHeads_ = numHeads;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        blockSize_ = blockSize;
        maxBlocksPerSeq_ = maxBlocksPerSeq;
        numSplits_ = numSplits;
        fusedContextLimit_ = fusedContextLimit;
        groupHeads_ = numHeads / numKvHeads;
        headsPerTask_ = headsPerTask < 1 ? 1 : headsPerTask;
        if (headsPerTask_ > groupHeads_) {
            headsPerTask_ = groupHeads_;
        }
        if (headsPerTask_ > kCubeTileM) {
            headsPerTask_ = kCubeTileM;
        }
        chunksPerGroup_ = CeilDiv(groupHeads_, headsPerTask_);

        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTables));
        contextLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(contextLens), numTokens);

        mm_.Init(pipe_, headSize_, kCubeTileRows);
        vector_.Init(pipe_, queryRot, keyCache, valueCache, scaleCache, modeTables, workspace, output, numHeads,
                     numKvHeads, headSize, blockSize, numSplits, scale, invSqrtLen);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process(uint32_t tasksPerBlock)
    {
        const uint32_t tasks = numTokens_ * numKvHeads_ * numSplits_ * chunksPerGroup_;
        const uint32_t start = MixBlockIdx() * tasksPerBlock;
        uint32_t end = start + tasksPerBlock;
        if (end > tasks) {
            end = tasks;
        }
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t chunk = task % chunksPerGroup_;
            const uint32_t rest = task / chunksPerGroup_;
            const uint32_t split = rest % numSplits_;
            const uint32_t pair = rest / numSplits_;
            ComputeTask(pair / numKvHeads_, pair % numKvHeads_, split, chunk);
        }
    }

    __aicore__ inline bool NeedsReduction()
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

    // Reduction tasks are (token, head). The block's range is halved between its two subcores.
    __aicore__ inline void Reduce(Reducer &reducer, uint32_t reduceTasksPerBlock)
    {
        const uint32_t total = numTokens_ * numHeads_;
        const uint32_t start = MixBlockIdx() * reduceTasksPerBlock;
        uint32_t end = start + reduceTasksPerBlock;
        if (end > total) {
            end = total;
        }
        if (start >= end) {
            return;
        }
        const uint32_t firstHalf = CeilDiv(end - start, kVectorSubcoresPerBlock);
        const bool second = AscendC::GetSubBlockIdx() != 0;
        const uint32_t from = second ? start + firstHalf : start;
        const uint32_t to = second ? end : start + firstHalf;
        for (uint32_t task = from; task < to; ++task) {
            const uint32_t token = task / numHeads_;
            if (!IsFused(contextLenGm_.GetValue(token))) {
                reducer.Reduce(token, task % numHeads_);
            }
        }
    }

private:
    struct TileCursor {
        uint32_t block = 0;
        uint32_t rows = 0;
        uint32_t base = 0;
        uint32_t physical = 0;
        uint32_t valid = 0;
        bool active = false;
    };

    __aicore__ inline bool IsFused(int32_t contextLen) const
    {
        return numSplits_ <= 1 || contextLen <= static_cast<int32_t>(fusedContextLimit_);
    }

    __aicore__ inline void ComputeTask(uint32_t token, uint32_t kvHead, uint32_t split, uint32_t chunk)
    {
        const int32_t contextLen = contextLenGm_.GetValue(token);
        const bool fused = IsFused(contextLen);
        if (fused && split > 0) {
            return;
        }
        const uint32_t offset = chunk * headsPerTask_;
        const uint32_t rows = groupHeads_ - offset < headsPerTask_ ? groupHeads_ - offset : headsPerTask_;
        const TurboQuantTaskHeads heads = Vector::Heads(kvHead * groupHeads_ + offset, rows);

        if ASCEND_IS_AIV {
            if constexpr (Vector::kBatched) {
                // Nothing drains the vector unit between tasks, and the previous task's vector work may still
                // read the query buffer and an ingest slot. This task's reads must land behind it.
                SyncVectorToMte2();
            }
            vector_.BeginTask(heads);
        }

        const uint32_t seqBlocks = contextLen > 0 ? CeilDiv(static_cast<uint32_t>(contextLen), blockSize_) : 0;
        const uint32_t blocksPerSplit = CeilDiv(seqBlocks, fused ? 1u : numSplits_);
        const uint32_t blockStart = split * blocksPerSplit;
        uint32_t blockEnd = blockStart + blocksPerSplit;
        if (blockEnd > seqBlocks) {
            blockEnd = seqBlocks;
        }

        if (blockStart < blockEnd) {
            const uint32_t ctxLen = static_cast<uint32_t>(contextLen);
            const uint32_t numTiles = CountTiles(token, ctxLen, blockStart, blockEnd);
            if ASCEND_IS_AIV {
                vector_.PrepareTask(mm_, token, kvHead, heads);
                const uint32_t tail = ctxLen % kCubeTileRows;
                if (heads.mine > 0 && blockEnd == seqBlocks && tail != 0) {
                    vector_.BuildTailMask(tail);
                }
                PipelineAiv(token, ctxLen, blockStart, blockEnd, numTiles, kvHead, heads);
            }
            if ASCEND_IS_AIC {
                if (numTiles > 0) {
                    Cube::RunTiles(mm_, vector_.Scores(), vector_.Context(), numTiles, heads.rows, headSize_);
                }
            }
        }

        if ASCEND_IS_AIV {
            vector_.FinishTask(token, split, heads, fused);
        }
    }

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

    // The AIV side of the tile pipeline. Tile t+1 is staged while the Cube multiplies tile t, so the
    // two L1 slots alternate; every subcore sets every AIV -> AIC flag because the AIC's wait needs both.
    __aicore__ inline void PipelineAiv(uint32_t token, uint32_t contextLen, uint32_t blockStart, uint32_t blockEnd,
                                       uint32_t numTiles, uint32_t kvHead, const TurboQuantTaskHeads &heads)
    {
        if (numTiles == 0) {
            return;
        }
        if constexpr (Vector::kBatched) {
            PipelineAivRing(token, contextLen, blockStart, blockEnd, numTiles, kvHead, heads);
            return;
        }
        TileCursor stageCursor;
        TileCursor consumeCursor;
        stageCursor.block = blockStart;
        consumeCursor.block = blockStart;

        NextTile(stageCursor, token, contextLen, blockEnd);
        vector_.StageTile(mm_, stageCursor.physical, stageCursor.base, kvHead, 0, false);
        SignalSlotReady(0);

        for (uint32_t tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
            const uint32_t nextSlotIdx = (tileIdx + 1) % kCubeSlots;
            NextTile(consumeCursor, token, contextLen, blockEnd);

            AscendC::CrossCoreWaitFlag(kFlagScoresReady);
            vector_.SoftmaxStageProbs(mm_, consumeCursor.valid, heads, 0);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kFlagProbsReady);
            vector_.SoftmaxRescaleAcc(heads);

            if (tileIdx + 1 < numTiles) {
                if (tileIdx + 1 >= kCubeSlots) {
                    AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagSlotFree + nextSlotIdx));
                }
                NextTile(stageCursor, token, contextLen, blockEnd);
                vector_.StageTile(mm_, stageCursor.physical, stageCursor.base, kvHead, nextSlotIdx, true);
                SignalSlotReady(nextSlotIdx);
            }

            AscendC::CrossCoreWaitFlag(kFlagContextReady);
            vector_.Accumulate(heads);
        }
    }

    // The same pipeline with the MTE2 half of the ingest a tile further ahead. Tile t reads into ingest
    // slot t % kCubeSlots; the first two are read up front and tile t + 2 as soon as tile t's softmax has
    // consumed its scale lanes, so the read overlaps tile t + 1's stage, GEMMs and accumulation, and a
    // stage only waits on its own slot's read.
    __aicore__ inline void PipelineAivRing(uint32_t token, uint32_t contextLen, uint32_t blockStart,
                                           uint32_t blockEnd, uint32_t numTiles, uint32_t kvHead,
                                           const TurboQuantTaskHeads &heads)
    {
        TileCursor readCursor;
        TileCursor consumeCursor;
        readCursor.block = blockStart;
        consumeCursor.block = blockStart;

        const uint32_t primed = numTiles < kCubeSlots ? numTiles : kCubeSlots;
        for (uint32_t slot = 0; slot < primed; ++slot) {
            NextTile(readCursor, token, contextLen, blockEnd);
            vector_.ReadTile(readCursor.physical, readCursor.base, kvHead, slot);
        }
        vector_.StageSlot(mm_, kvHead, 0);
        SignalSlotReady(0);

        for (uint32_t tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
            const uint32_t slot = tileIdx % kCubeSlots;
            const uint32_t nextSlot = (tileIdx + 1) % kCubeSlots;
            NextTile(consumeCursor, token, contextLen, blockEnd);

            AscendC::CrossCoreWaitFlag(kFlagScoresReady);
            vector_.SoftmaxStageProbs(mm_, consumeCursor.valid, heads, slot);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kFlagProbsReady);

            if (tileIdx + kCubeSlots < numTiles) {
                SyncVectorToMte2();
                NextTile(readCursor, token, contextLen, blockEnd);
                vector_.ReadTile(readCursor.physical, readCursor.base, kvHead, slot);
            }
            vector_.SoftmaxRescaleAcc(heads);

            if (tileIdx + 1 < numTiles) {
                if (tileIdx + 1 >= kCubeSlots) {
                    AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagSlotFree + nextSlot));
                }
                vector_.StageSlot(mm_, kvHead, nextSlot);
                SignalSlotReady(nextSlot);
            }

            AscendC::CrossCoreWaitFlag(kFlagContextReady);
            vector_.Accumulate(heads);
        }
    }

    __aicore__ static inline void SignalSlotReady(uint32_t slot)
    {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(static_cast<uint16_t>(kFlagSlotReady + slot));
    }

    AscendC::TPipe *pipe_;
    Mm mm_;
    Vector vector_;
    AscendC::GlobalTensor<int32_t> blockTableGm_;
    AscendC::GlobalTensor<int32_t> contextLenGm_;
    uint32_t numTokens_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t groupHeads_ = 1;
    uint32_t headsPerTask_ = 1;
    uint32_t chunksPerGroup_ = 1;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlocksPerSeq_ = 0;
    uint32_t numSplits_ = 1;
    uint32_t fusedContextLimit_ = 0;
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
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kFlagOperandsReady);
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(kFlagOperandsReady);
            if (variant == 10) {
                AscendC::CrossCoreWaitFlag(kFlagOperandsReady);
            }
            if (bIsNk != 0) {
                mm_.GemmScores(out, mm_.B1(), m, k, n);
            } else {
                mm_.GemmContext(out, mm_.B1(), m, k, n, variant);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kFlagProductReady);
        }
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag(kFlagProductReady);
            if (AscendC::GetSubBlockIdx() == 0) {
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

#define TURBOQUANT_MM_FUSED_DECODE_DECLARE(NAME, MODE, TYPE, VEC_BARRIERS)                                           \
    extern "C" __global__ __aicore__ void NAME(                                                                      \
        GM_ADDR queryRot, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR blockTables,             \
        GM_ADDR contextLens, GM_ADDR modeTables, GM_ADDR workspace, GM_ADDR output, uint32_t numTokens,              \
        uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,     \
        uint32_t numSplits, uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,             \
        uint32_t fusedContextLimit, float scale, float invSqrtLen)                                                   \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantFusedDecode<MODE, TYPE, VEC_BARRIERS> op(&pipe);                                                   \
        op.Init(queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output, \
                numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,      \
                fusedContextLimit, scale, invSqrtLen);                                                               \
        op.Process(tasksPerBlock);                                                                                   \
        if (op.NeedsReduction()) {                                                                                   \
            AscendC::SyncAll<false>();                                                                               \
            if ASCEND_IS_AIV {                                                                                       \
                TurboQuantPartialReducer<TYPE, VEC_BARRIERS> reducer(&pipe);                                         \
                reducer.Init(workspace, output, numHeads, headSize, numSplits);                                      \
                op.Reduce(reducer, reduceTasksPerBlock);                                                             \
            }                                                                                                        \
        }                                                                                                            \
    }

#define TURBOQUANT_MM_DECLARE_MODE(MODE_NAME, MODE)                                                                  \
    TURBOQUANT_MM_RESHAPE_AND_CACHE_DECLARE(MODE_NAME, MODE, half)                                                   \
    TURBOQUANT_MM_FUSED_DECODE_DECLARE(turboquant_mm_fused_decode_##MODE_NAME##_half, MODE, half, kFusedVecBarriers)

TURBOQUANT_MM_DECLARE_MODE(kv3fp4, TurboQuantMode::KV3_FP4)
TURBOQUANT_MM_DECLARE_MODE(kv4fp8, TurboQuantMode::KV4_FP8)
TURBOQUANT_MM_DECLARE_MODE(kv5fp8, TurboQuantMode::KV5_FP8)
TURBOQUANT_MM_FUSED_DECODE_DECLARE(turboquant_mm_fused_decode_barriered_kv4fp8_half, TurboQuantMode::KV4_FP8, half,
                                   true)

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

void turboquant_mm_fused_decode_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *queryRot,
                                    void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
                                    void *contextLens, void *modeTables, void *workspace, void *output,
                                    uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize,
                                    uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,
                                    uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,
                                    uint32_t fusedContextLimit, float scale, float invSqrtLen)
{
    if (type != AscendType::FP16 || blockDim == 0) {
        return;
    }
    switch (static_cast<turboquant::TurboQuantMode>(mode)) {
        case turboquant::TurboQuantMode::KV3_FP4:
            turboquant_mm_fused_decode_kv3fp4_half<<<blockDim, nullptr, stream>>>(
                queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,
                numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,
                tasksPerBlock, reduceTasksPerBlock, fusedContextLimit, scale, invSqrtLen);
            break;
        case turboquant::TurboQuantMode::KV4_FP8:
            turboquant_mm_fused_decode_kv4fp8_half<<<blockDim, nullptr, stream>>>(
                queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,
                numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,
                tasksPerBlock, reduceTasksPerBlock, fusedContextLimit, scale, invSqrtLen);
            break;
        default:
            turboquant_mm_fused_decode_kv5fp8_half<<<blockDim, nullptr, stream>>>(
                queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,
                numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,
                tasksPerBlock, reduceTasksPerBlock, fusedContextLimit, scale, invSqrtLen);
            break;
    }
}

void turboquant_mm_fused_decode_barriered_impl(AscendType type, void *stream, uint32_t blockDim, void *queryRot,
                                              void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
                                              void *contextLens, void *modeTables, void *workspace, void *output,
                                              uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                              uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                              uint32_t numSplits, uint32_t headsPerTask, uint32_t tasksPerBlock,
                                              uint32_t reduceTasksPerBlock, uint32_t fusedContextLimit, float scale,
                                              float invSqrtLen)
{
    if (type != AscendType::FP16 || blockDim == 0) {
        return;
    }
    turboquant_mm_fused_decode_barriered_kv4fp8_half<<<blockDim, nullptr, stream>>>(
        queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output, numTokens,
        numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask, tasksPerBlock,
        reduceTasksPerBlock, fusedContextLimit, scale, invSqrtLen);
}

void turboquant_cube_gemm_probe_impl(void *stream, void *a, void *b, void *c, uint32_t m, uint32_t k, uint32_t n,
                                     uint32_t headSize, uint32_t tileRows, uint32_t aElems, uint32_t bElems,
                                     uint32_t cElems, uint32_t bIsNk, uint32_t variant)
{
    turboquant_cube_gemm_probe_fp8<<<1, nullptr, stream>>>(a, b, c, m, k, n, headSize, tileRows, aElems, bElems,
                                                           cElems, bIsNk, variant);
}

}
