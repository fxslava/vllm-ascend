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
// single-launch decode, and the fp8 GEMM probe the Cube contract test drives. The wheel builds this file
// for arch35 and ships kv4fp8 only: the other modes and the probe are compiled under
// VLLM_ASCEND_TQ_TEST_KERNELS, which only csrc/tests defines.

#include "kernel_operator.h"

#include "../../../kernels/types.h"
#include "common/turboquant_codec_mx.h"
#include "cube/turboquant_cube_service.h"
#include "cube/turboquant_query_basis.h"
#include "vector/turboquant_vector_service.h"

using vllm_ascend::turboquant::CeilDiv;
using vllm_ascend::turboquant::GemmLayout;
using vllm_ascend::turboquant::kCubeLoadDoubleOperandsWait;
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
using vllm_ascend::turboquant::kOperandC0;
using vllm_ascend::turboquant::kStoresNzTiles;
using vllm_ascend::turboquant::kSubBlockSyncMode;
using vllm_ascend::turboquant::kVectorSubcoresPerBlock;
using vllm_ascend::turboquant::MixBlockIdx;
using vllm_ascend::turboquant::ScaleSlotFloats;
using vllm_ascend::turboquant::SyncMte2ToVector;
using vllm_ascend::turboquant::SyncVectorToMte2;
using vllm_ascend::turboquant::TurboQuantCodec4;
using vllm_ascend::turboquant::TurboQuantCubeDecodeService;
using vllm_ascend::turboquant::TurboQuantCubeMm;
using vllm_ascend::turboquant::TurboQuantMode;
using vllm_ascend::turboquant::TurboQuantModeCodec;
using vllm_ascend::turboquant::TurboQuantPartialReducer;
using vllm_ascend::turboquant::TurboQuantQueryBasis;
using vllm_ascend::turboquant::TurboQuantTaskHeads;
using vllm_ascend::turboquant::TurboQuantVectorDecodeService;

namespace {

template <TurboQuantMode MODE, typename scalar_t>
class TurboQuantModeReshapeAndCache {
public:
    using Codec = TurboQuantModeCodec<MODE>;

    __aicore__ inline explicit TurboQuantModeReshapeAndCache(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *key, __gm__ void *value, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *scaleCache, __gm__ void *slotMapping, __gm__ void *piSigns,
                                __gm__ void *rotTables, __gm__ void *modeTables, const uint32_t numTokens,
                                const uint32_t numKvHeads, const uint32_t headSize, const uint32_t blockSize,
                                const uint32_t numBlocks, const uint32_t tokensPerCore, const float invSqrtLen)
    {
        numTokens_ = numTokens;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        blockSize_ = blockSize;
        numSlots_ = static_cast<uint64_t>(numBlocks) * blockSize;
        tokensPerCore_ = tokensPerCore;
        packedBytes_ = Codec::PackedBytes(headSize);
        headPlane_ = numKvHeads_ * headSize_;
        packedPlane_ = numKvHeads_ * packedBytes_;
        scaleSlot_ = ScaleSlotFloats(numKvHeads_);
        // Consecutive (kv head, group) cells of one tile row are a tile apart.
        nzTileRowParams_ = AscendC::DataCopyParams{static_cast<uint16_t>(packedPlane_ / kOperandC0), 1, 0,
                                                   static_cast<uint16_t>(kCubeTileRows - 1)};

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
        pipe_->InitBuffer(scaleIndexBuf_, 2 * numKvHeads_ * sizeof(int32_t));

        rotation_.Init(pipe_, headSize_, 1, invSqrtLen, rotTablesGm_);
        codec_.Init(pipe_, headSize_, 1, invSqrtLen, modeTablesGm_);

        const AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::DataCopy(signs, piSignsGm_, headSize_);

        const AscendC::LocalTensor<int32_t> scaleIndex = scaleIndexBuf_.Get<int32_t>();
        AscendC::ArithProgression(scaleIndex, 0, static_cast<int32_t>(kFp32PerBlock * sizeof(float)),
                                  static_cast<int32_t>(2 * numKvHeads_));
        // The signs are the only data this Init moves itself, GM -> UB for the vector unit.
        SyncMte2ToVector();
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
    static constexpr uint32_t kSlotRing = 4;

    __aicore__ inline void StageTokenIn(const uint32_t token, const uint32_t step)
    {
        const int32_t rawSlot = slotGm_.GetValue(token);
        // A padding token carries a negative slot, and a slot past the last row of the cache is a caller
        // bug. Both become -1 here, which is what StageTokenOut reads as "write nothing": every cache
        // address it forms is scaled by the slot, so either one would write outside the cache -- an AI
        // core "address for scalar to access GM is invalid" (264) at best, silent corruption of whatever
        // else lives at that address at worst.
        const int32_t slot = (rawSlot >= 0 && static_cast<uint64_t>(rawSlot) < numSlots_) ? rawSlot : -1;
        slotRing_[step % kSlotRing] = slot;
        const AscendC::LocalTensor<scalar_t> in = inQueue_.template AllocTensor<scalar_t>();
        if (slot >= 0) {
            const uint64_t base = static_cast<uint64_t>(token) * headPlane_;
            AscendC::DataCopy(in, keyGm_[base], headPlane_);
            AscendC::DataCopy(in[headPlane_], valueGm_[base], headPlane_);
        }
        inQueue_.EnQue(in);
    }

    // Rotates and encodes every (plane, kv head) vector of one token, then gathers its per-vector scales into
    // the token's scale slot.
    __aicore__ inline void ComputeToken()
    {
        AscendC::LocalTensor<scalar_t> in = inQueue_.template DeQue<scalar_t>();
        const AscendC::LocalTensor<int8_t> packed = outPacked_.template AllocTensor<int8_t>();
        const AscendC::LocalTensor<float> scaleOut = outScale_.template AllocTensor<float>();

        const AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        const AscendC::LocalTensor<float> vec = work;
        const AscendC::LocalTensor<float> tmp = work[headSize_];
        const AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        const AscendC::LocalTensor<float> steps = stepBuf_.Get<float>();

        for (uint32_t plane = 0; plane < 2; ++plane) {
            for (uint32_t head = 0; head < numKvHeads_; ++head) {
                const uint32_t src = plane * headPlane_ + head * headSize_;
                AscendC::Cast(vec, in[src], AscendC::RoundMode::CAST_NONE, headSize_);
                rotation_.ApplyPi(vec, tmp, signs, headSize_);
                codec_.Encode(packed[plane * packedPlane_ + head * packedBytes_], vec,
                              steps[(plane * numKvHeads_ + head) * kFp32PerBlock], headSize_);
            }
        }
        inQueue_.FreeTensor(in);

        const AscendC::LocalTensor<uint32_t> scaleIndex = scaleIndexBuf_.Get<int32_t>().ReinterpretCast<uint32_t>();
        AscendC::Duplicate(scaleOut, 0.0f, scaleSlot_);
        // Overlapping write: the Gather overwrites lanes the Duplicate just zeroed.
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(scaleOut, steps, scaleIndex, kGatherSrcBase, 2 * numKvHeads_);

        outPacked_.EnQue(packed);
        outScale_.EnQue(scaleOut);
    }

    __aicore__ inline void StageTokenOut(const uint32_t step)
    {
        AscendC::LocalTensor<int8_t> packed = outPacked_.template DeQue<int8_t>();
        AscendC::LocalTensor<float> scaleOut = outScale_.template DeQue<float>();
        const int32_t slot = slotRing_[step % kSlotRing];
        if (slot >= 0) {
            const uint64_t row = static_cast<uint64_t>(slot);
            if constexpr (kStoresNzTiles<MODE>) {
                StageNzTiledPlane(keyCacheGm_, packed, row);
                StageNzTiledPlane(valueCacheGm_, packed[packedPlane_], row);
            } else {
                AscendC::DataCopy(keyCacheGm_[row * packedPlane_], packed, packedPlane_);
                AscendC::DataCopy(valueCacheGm_[row * packedPlane_], packed[packedPlane_], packedPlane_);
            }
            AscendC::DataCopy(scaleCacheGm_[row * scaleSlot_], scaleOut, scaleSlot_);
        }
        outPacked_.FreeTensor(packed);
        outScale_.FreeTensor(scaleOut);
    }

    // Scatters one token's row-major plane into its NZ-tiled cells (turboquant_layout.h,
    // NzTiledPackedByte) in a single strided burst, so the decode never permutes.
    __aicore__ inline void StageNzTiledPlane(AscendC::GlobalTensor<int8_t> &cacheGm,
                                             const AscendC::LocalTensor<int8_t> &plane, const uint64_t slot)
    {
        const uint64_t tileRow = slot % kCubeTileRows;
        AscendC::DataCopy(cacheGm[(slot - tileRow) * packedPlane_ + tileRow * kOperandC0], plane, nzTileRowParams_);
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
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleIndexBuf_;
    AscendC::GlobalTensor<scalar_t> keyGm_;
    AscendC::GlobalTensor<scalar_t> valueGm_;
    AscendC::GlobalTensor<int8_t> keyCacheGm_;
    AscendC::GlobalTensor<int8_t> valueCacheGm_;
    AscendC::GlobalTensor<float> scaleCacheGm_;
    AscendC::GlobalTensor<int32_t> slotGm_;
    AscendC::GlobalTensor<float> piSignsGm_;
    AscendC::GlobalTensor<int32_t> rotTablesGm_;
    AscendC::GlobalTensor<int32_t> modeTablesGm_;
    AscendC::DataCopyParams nzTileRowParams_;
    int32_t slotRing_[kSlotRing] = {-1, -1, -1, -1};
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

// One launch per decode step on the arch35 Cube. A task is (token, kv head, sequence split, head
// chunk). For every tile of a task the two AIV subcores stage K (subcore 0) and V (subcore 1) into L1;
// the AIC runs the score and context GEMMs into both subcores' UB (TurboQuantCubeDecodeService); each
// subcore runs the online softmax for its half of the heads (TurboQuantVectorDecodeService). A token
// whose context fits fusedContextLimit writes its output straight to GM; a longer one leaves
// partials that the same launch reduces after AscendC::SyncAll.
//
// PRE_ROTATED is where Pi q happens. true: the caller hands in the rotated fp32 query (rotate_q, or an
// upstream fusion), the launch has no rotation code or buffer at all, and the output stays in the rotated
// basis. false: the caller hands in the raw query, RotateQuery() rotates every vector once before Process()
// starts the task loop, and the output stage InitBasis() selects runs on every head before its output cast
// (cube/turboquant_query_basis.h).
//
// BYPASS_UNPACK is the unpack ablation described on TurboQuantVectorDecodeService: a timing-only build of
// the same decode with the codec expand removed from the ingest and nothing else changed. Its output is
// all-zero by construction, so only a benchmark instantiates it and only under VLLM_ASCEND_TQ_TEST_KERNELS.
template <TurboQuantMode MODE, typename scalar_t, bool PRE_ROTATED = true, bool BYPASS_UNPACK = false>
class TurboQuantFusedDecode {
public:
    using Mm = TurboQuantCubeMm<MODE>;
    using Codec = TurboQuantModeCodec<MODE>;
    using Vector = TurboQuantVectorDecodeService<MODE, scalar_t, Mm, Codec, BYPASS_UNPACK>;
    using Cube = TurboQuantCubeDecodeService<MODE>;
    using Reducer = TurboQuantPartialReducer<scalar_t>;
    using Basis = TurboQuantQueryBasis<scalar_t, !PRE_ROTATED>;

    __aicore__ inline explicit TurboQuantFusedDecode(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *queryRot, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *scaleCache, __gm__ void *blockTables, __gm__ void *contextLens,
                                __gm__ void *modeTables, __gm__ void *workspace, __gm__ void *output,
                                const uint32_t numTokens, const uint32_t numHeads, const uint32_t numKvHeads,
                                const uint32_t headSize, const uint32_t blockSize, const uint32_t maxBlocksPerSeq,
                                const uint32_t numSplits, const uint32_t headsPerTask,
                                const uint32_t fusedContextLimit, const float scale, const float invSqrtLen)
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
        // Init boundary: drains every pipe before the first task's GM reads and cross-core flags.
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // The raw-query decode's two basis stages. queryRot, the buffer Init() pointed the tasks at, is where the
    // prologue's rotation lands. After Init(), before RotateQuery().
    __aicore__ inline void InitBasis(__gm__ void *query, __gm__ void *piSigns, __gm__ void *rotTables, __gm__ void *h16,
                                     __gm__ void *gate, __gm__ void *queryRot, const uint32_t cubeChunkVectors,
                                     const uint32_t outputStage, const float invSqrtLen)
    {
        static_assert(Basis::kEnabled, "a pre-rotated decode has no query rotation");
        basis_.Init(pipe_, query, piSigns, rotTables, h16, gate, queryRot, numHeads_, headSize_, cubeChunkVectors,
                    outputStage, invSqrtLen);
    }

    // The prologue. Before Process().
    __aicore__ inline void RotateQuery(const uint32_t vectorsPerBlock)
    {
        static_assert(Basis::kEnabled, "a pre-rotated decode has no query rotation");
        basis_.Rotate(numTokens_ * numHeads_, vectorsPerBlock);
    }

    __aicore__ inline void Process(const uint32_t tasksPerBlock)
    {
        const uint32_t tasks = numTokens_ * numKvHeads_ * numSplits_ * chunksPerGroup_;
        const uint32_t start = MixBlockIdx() * tasksPerBlock;
        uint32_t end = start + tasksPerBlock;
        if (end > tasks) {
            end = tasks;
        }
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t chunk = task % chunksPerGroup_;
            const uint32_t splitTask = task / chunksPerGroup_;
            const uint32_t split = splitTask % numSplits_;
            const uint32_t tokenKvHead = splitTask / numSplits_;
            ComputeTask(tokenKvHead / numKvHeads_, tokenKvHead % numKvHeads_, split, chunk);
        }
    }

    // Grid-uniform on purpose: every block reads every token's length, so the SyncAll the caller guards
    // with this is entered by all of them or by none. A per-block predicate here would strand the blocks
    // that answered false in the barrier the rest went on to run.
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
    __aicore__ inline void Reduce(Reducer &reducer, const uint32_t reduceTasksPerBlock)
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
        const bool secondSubcore = AscendC::GetSubBlockIdx() != 0;
        const uint32_t from = secondSubcore ? start + firstHalf : start;
        const uint32_t to = secondSubcore ? end : start + firstHalf;
        for (uint32_t task = from; task < to; ++task) {
            const uint32_t token = task / numHeads_;
            if (!IsFused(contextLenGm_.GetValue(token))) {
                reducer.Reduce(token, task % numHeads_, basis_);
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

    __aicore__ inline bool IsFused(const int32_t contextLen) const
    {
        return numSplits_ <= 1 || contextLen <= static_cast<int32_t>(fusedContextLimit_);
    }

    __aicore__ inline void ComputeTask(const uint32_t token, const uint32_t kvHead, const uint32_t split,
                                       const uint32_t chunk)
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
                vector_.StageQuery(mm_, token, kvHead, heads);
                const uint32_t tail = ctxLen % kCubeTileRows;
                if (heads.mine > 0 && blockEnd == seqBlocks && tail != 0) {
                    vector_.ComputeTailMask(tail);
                }
            }
            // The one condition both cores branch on, so the bypass is visibly lock-step. A block range
            // can hold no tile at all -- every blockTable entry in it negative, which is what a padding or
            // unmapped block carries -- and then neither core may touch the cross-core flags: an AIV that
            // skipped StageTiles while the AIC still ran RunTiles would leave the AIC on kFlagSlotReady
            // with nothing ever to set it, which is a hung launch, not a wrong answer. CountTiles is safe
            // to branch on here because both cores run it over the same GM words with the same indices,
            // so they cannot disagree about it.
            if (numTiles > 0) {
                if ASCEND_IS_AIV {
                    StageTiles(token, ctxLen, blockStart, blockEnd, numTiles, kvHead, heads);
                }
                if ASCEND_IS_AIC {
                    Cube::RunTiles(mm_, vector_.Scores(), vector_.Context(), numTiles, heads.rows, headSize_);
                }
            }
        }

        if ASCEND_IS_AIV {
            vector_.StageTaskOutput(token, split, heads, fused, basis_);
        }
    }

    // The block range and tile cursor below take the task's extents as separate scalars on purpose: packed
    // into a struct they are reloaded from memory on every tile, and the AIV issues more scalar instructions.
    __aicore__ inline uint32_t CountTiles(const uint32_t token, const uint32_t contextLen, const uint32_t blockStart,
                                          const uint32_t blockEnd)
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

    __aicore__ inline bool NextTile(TileCursor &cursor, const uint32_t token, const uint32_t contextLen,
                                    const uint32_t blockEnd)
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

    // numTiles >= 1: ComputeTask bypasses the whole pipeline on both cores when a block range holds no
    // tile, so neither pipeline below has an empty-range case to carry.
    __aicore__ inline void StageTiles(const uint32_t token, const uint32_t contextLen, const uint32_t blockStart,
                                      const uint32_t blockEnd, const uint32_t numTiles, const uint32_t kvHead,
                                      const TurboQuantTaskHeads &heads)
    {
        if constexpr (Vector::kBatched) {
            StageTilesRing(token, contextLen, blockStart, blockEnd, numTiles, kvHead, heads);
        } else {
            StageTilesLockStep(token, contextLen, blockStart, blockEnd, numTiles, kvHead, heads);
        }
    }

    // The AIV side of the tile pipeline. Tile t+1 is staged while the Cube multiplies tile t, so the
    // two L1 slots alternate; every subcore sets every AIV -> AIC flag because the AIC's wait needs both.
    __aicore__ inline void StageTilesLockStep(const uint32_t token, const uint32_t contextLen,
                                              const uint32_t blockStart, const uint32_t blockEnd,
                                              const uint32_t numTiles, const uint32_t kvHead,
                                              const TurboQuantTaskHeads &heads)
    {
        TileCursor stageCursor;
        TileCursor consumeCursor;
        stageCursor.block = blockStart;
        consumeCursor.block = blockStart;

        NextTile(stageCursor, token, contextLen, blockEnd);
        vector_.StageTile(mm_, stageCursor.physical, stageCursor.base, kvHead, 0, false);
        SyncSlotReady(0);

        for (uint32_t tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
            const uint32_t nextSlot = (tileIdx + 1) % kCubeSlots;
            NextTile(consumeCursor, token, contextLen, blockEnd);

            AscendC::CrossCoreWaitFlag(kFlagScoresReady);
            vector_.ComputeSoftmaxAndStageProbs(mm_, consumeCursor.valid, heads, 0);
            AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(kFlagProbsReady);
            vector_.ComputeAccumulatorDecay(heads);

            if (tileIdx + 1 < numTiles) {
                if (tileIdx + 1 >= kCubeSlots) {
                    AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagSlotFree + nextSlot));
                }
                NextTile(stageCursor, token, contextLen, blockEnd);
                vector_.StageTile(mm_, stageCursor.physical, stageCursor.base, kvHead, nextSlot, true);
                SyncSlotReady(nextSlot);
            }

            AscendC::CrossCoreWaitFlag(kFlagContextReady);
            vector_.ComputeAccumulate(heads);
        }
    }

    // The same pipeline with the MTE2 half of the ingest a tile further ahead. Tile t reads into ingest
    // slot t % kCubeSlots; the first two are read up front and tile t + 2 as soon as tile t's softmax has
    // consumed its scale lanes, so the read overlaps tile t + 1's stage, GEMMs and accumulation, and a
    // stage only waits on its own slot's read.
    __aicore__ inline void StageTilesRing(const uint32_t token, const uint32_t contextLen, const uint32_t blockStart,
                                          const uint32_t blockEnd, const uint32_t numTiles, const uint32_t kvHead,
                                          const TurboQuantTaskHeads &heads)
    {
        TileCursor readCursor;
        TileCursor consumeCursor;
        readCursor.block = blockStart;
        consumeCursor.block = blockStart;

        const uint32_t primed = numTiles < kCubeSlots ? numTiles : kCubeSlots;
        for (uint32_t slot = 0; slot < primed; ++slot) {
            NextTile(readCursor, token, contextLen, blockEnd);
            vector_.ReadTileToSlot(readCursor.physical, readCursor.base, kvHead, slot);
        }
        vector_.StageSlotToL1(mm_, kvHead, 0);
        SyncSlotReady(0);

        for (uint32_t tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
            const uint32_t slot = tileIdx % kCubeSlots;
            const uint32_t nextSlot = (tileIdx + 1) % kCubeSlots;
            NextTile(consumeCursor, token, contextLen, blockEnd);

            AscendC::CrossCoreWaitFlag(kFlagScoresReady);
            vector_.ComputeSoftmaxAndStageProbs(mm_, consumeCursor.valid, heads, slot);
            AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(kFlagProbsReady);

            if (tileIdx + kCubeSlots < numTiles) {
                SyncVectorToMte2();
                NextTile(readCursor, token, contextLen, blockEnd);
                vector_.ReadTileToSlot(readCursor.physical, readCursor.base, kvHead, slot);
            }
            vector_.ComputeAccumulatorDecay(heads);

            if (tileIdx + 1 < numTiles) {
                if (tileIdx + 1 >= kCubeSlots) {
                    AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagSlotFree + nextSlot));
                }
                vector_.StageSlotToL1(mm_, kvHead, nextSlot);
                SyncSlotReady(nextSlot);
            }

            AscendC::CrossCoreWaitFlag(kFlagContextReady);
            vector_.ComputeAccumulate(heads);
        }
    }

    __aicore__ static inline void SyncSlotReady(const uint32_t slot)
    {
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(static_cast<uint16_t>(kFlagSlotReady + slot));
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
    // Last, so a pre-rotated decode's members keep their offsets; empty when PRE_ROTATED.
    Basis basis_;
};

#if defined(VLLM_ASCEND_TQ_TEST_KERNELS)
// Test-only: the fp8 GEMM probe test_sim_950pr_cube_gemm drives.
class TurboQuantCubeGemmProbe {
public:
    using Mm = TurboQuantCubeMm<TurboQuantMode::KV5_FP8>;
    using OperandT = typename Mm::OperandT;

    __aicore__ inline explicit TurboQuantCubeGemmProbe(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *leftGm, __gm__ void *rightGm, __gm__ void *outGm,
                                const uint32_t headSize, const uint32_t tileRows, const uint32_t leftElems,
                                const uint32_t rightElems, const uint32_t outElems)
    {
        leftElems_ = leftElems;
        rightElems_ = rightElems;
        outElems_ = outElems;
        leftGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OperandT *>(leftGm));
        rightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OperandT *>(rightGm));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outGm));
        mm_.Init(pipe_, headSize, tileRows);
        pipe_->InitBuffer(stageBuf_, leftElems > rightElems ? leftElems : rightElems);
        pipe_->InitBuffer(outBuf_, outElems * sizeof(float));
        // Init boundary: drains every pipe before the probe's first GM read.
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Run(const uint32_t m, const uint32_t k, const uint32_t n, const GemmLayout layout,
                               const uint32_t variant)
    {
        const bool rightIsNk = layout == GemmLayout::Normal;
        const AscendC::LocalTensor<float> out = outBuf_.Get<float>();
        if ASCEND_IS_AIV {
            const AscendC::LocalTensor<OperandT> stage = stageBuf_.Get<OperandT>();
            AscendC::DataCopy(stage, leftGm_, leftElems_);
            // Probe-only GM -> UB -> L1 hand-off, kept in its original form.
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(rightIsNk ? mm_.L1Query() : mm_.L1Probs(), stage, leftElems_);
            AscendC::DataCopy(stage, rightGm_, rightElems_);
            // Probe-only GM -> UB -> L1 hand-off, kept in its original form.
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(mm_.L1Weight(), stage, rightElems_);
            AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(kFlagOperandsReady);
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(kFlagOperandsReady);
            if (variant == kCubeLoadDoubleOperandsWait) {
                AscendC::CrossCoreWaitFlag(kFlagOperandsReady);
            }
            if (rightIsNk) {
                mm_.GemmScores(out, mm_.L1Weight(), m, k, n);
            } else {
                mm_.GemmContext(out, mm_.L1Weight(), m, k, n, variant);
            }
            AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(kFlagProductReady);
        }
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag(kFlagProductReady);
            if (AscendC::GetSubBlockIdx() == 0) {
                AscendC::DataCopy(outGm_, out, outElems_);
            }
            // Probe-only: the product's GM write lands before the launch returns.
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    AscendC::TPipe *pipe_;
    Mm mm_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stageBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outBuf_;
    AscendC::GlobalTensor<OperandT> leftGm_;
    AscendC::GlobalTensor<OperandT> rightGm_;
    AscendC::GlobalTensor<float> outGm_;
    uint32_t leftElems_ = 0;
    uint32_t rightElems_ = 0;
    uint32_t outElems_ = 0;
};
#endif

}

#define ASCEND_TQ_DECLARE_MM_RESHAPE_AND_CACHE(MODE_NAME, MODE, TYPE)                                                \
    extern "C" __global__ __aicore__ void turboquant_mm_reshape_and_cache_##MODE_NAME##_##TYPE(                      \
        GM_ADDR key, GM_ADDR value, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR slotMapping,   \
        GM_ADDR piSigns, GM_ADDR rotTables, GM_ADDR modeTables, uint32_t numTokens, uint32_t numKvHeads,             \
        uint32_t headSize, uint32_t blockSize, uint32_t numBlocks, uint32_t tokensPerCore, float invSqrtLen)          \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantModeReshapeAndCache<MODE, TYPE> op(&pipe);                                                         \
        op.Init(key, value, keyCache, valueCache, scaleCache, slotMapping, piSigns, rotTables, modeTables,           \
                numTokens, numKvHeads, headSize, blockSize, numBlocks, tokensPerCore, invSqrtLen);                   \
        op.Process();                                                                                                \
    }

#define ASCEND_TQ_DECLARE_MM_FUSED_DECODE(NAME, MODE, TYPE)                                                          \
    ASCEND_TQ_DECLARE_MM_FUSED_DECODE_ABLATED(NAME, MODE, TYPE, false)

// BYPASS_UNPACK selects the unpack ablation of the same kernel; see TurboQuantFusedDecode. The argument list
// and the launch are identical, so the two entries are interchangeable to a benchmark and to msprof.
#define ASCEND_TQ_DECLARE_MM_FUSED_DECODE_ABLATED(NAME, MODE, TYPE, BYPASS_UNPACK)                                   \
    extern "C" __global__ __aicore__ void NAME(                                                                      \
        GM_ADDR queryRot, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR blockTables,             \
        GM_ADDR contextLens, GM_ADDR modeTables, GM_ADDR workspace, GM_ADDR output, uint32_t numTokens,              \
        uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,     \
        uint32_t numSplits, uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,             \
        uint32_t fusedContextLimit, float scale, float invSqrtLen)                                                   \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantFusedDecode<MODE, TYPE, true, BYPASS_UNPACK> op(&pipe);                                            \
        op.Init(queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output, \
                numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,      \
                fusedContextLimit, scale, invSqrtLen);                                                               \
        op.Process(tasksPerBlock);                                                                                   \
        if (op.NeedsReduction()) {                                                                                   \
            AscendC::SyncAll<false>();                                                                               \
            if ASCEND_IS_AIV {                                                                                       \
                TurboQuantPartialReducer<TYPE> reducer(&pipe);                                                       \
                reducer.Init(workspace, output, numHeads, headSize, numSplits);                                      \
                op.Reduce(reducer, reduceTasksPerBlock);                                                             \
            }                                                                                                        \
        }                                                                                                            \
    }

// The raw-query decode: query is the unrotated scalar_t query and queryRot the fp32 buffer the prologue rotates
// it into, sized like a pre-rotated decode's query. h16 is read only with a Cube chunk size, gate only when
// outputStage is kGated; either may be null otherwise.
#define ASCEND_TQ_DECLARE_MM_FUSED_DECODE_RAW_QUERY(NAME, MODE, TYPE)                                                \
    extern "C" __global__ __aicore__ void NAME(                                                                      \
        GM_ADDR query, GM_ADDR piSigns, GM_ADDR rotTables, GM_ADDR h16, GM_ADDR gate, GM_ADDR queryRot,             \
        GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR blockTables, GM_ADDR contextLens,          \
        GM_ADDR modeTables, GM_ADDR workspace, GM_ADDR output, uint32_t numTokens, uint32_t numHeads,                \
        uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,    \
        uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,                                 \
        uint32_t prologueVectorsPerBlock, uint32_t prologueCubeChunkVectors, uint32_t outputStage,                   \
        uint32_t fusedContextLimit, float scale, float invSqrtLen)                                                   \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantFusedDecode<MODE, TYPE, false> op(&pipe);                                                          \
        op.Init(queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output, \
                numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,      \
                fusedContextLimit, scale, invSqrtLen);                                                               \
        op.InitBasis(query, piSigns, rotTables, h16, gate, queryRot, prologueCubeChunkVectors, outputStage,          \
                     invSqrtLen);                                                                                    \
        op.RotateQuery(prologueVectorsPerBlock);                                                                     \
        op.Process(tasksPerBlock);                                                                                   \
        if (op.NeedsReduction()) {                                                                                   \
            AscendC::SyncAll<false>();                                                                               \
            if ASCEND_IS_AIV {                                                                                       \
                TurboQuantPartialReducer<TYPE> reducer(&pipe);                                                       \
                reducer.Init(workspace, output, numHeads, headSize, numSplits);                                      \
                op.Reduce(reducer, reduceTasksPerBlock);                                                             \
            }                                                                                                        \
        }                                                                                                            \
    }

#define ASCEND_TQ_DECLARE_MM_MODE(MODE_NAME, MODE)                                                                   \
    ASCEND_TQ_DECLARE_MM_RESHAPE_AND_CACHE(MODE_NAME, MODE, half)                                                    \
    ASCEND_TQ_DECLARE_MM_FUSED_DECODE(turboquant_mm_fused_decode_##MODE_NAME##_half, MODE, half)

ASCEND_TQ_DECLARE_MM_MODE(kv4fp8, TurboQuantMode::KV4_FP8)
// kv4fp8 only: the one mode the fused decode is validated in (TURBOQUANT_TESTS.md 13.25).
ASCEND_TQ_DECLARE_MM_FUSED_DECODE_RAW_QUERY(turboquant_mm_fused_decode_raw_query_kv4fp8_half, TurboQuantMode::KV4_FP8,
                                            half)
#if defined(VLLM_ASCEND_TQ_TEST_KERNELS)
// The codebook modes have never executed through the fused decode, so only the test library builds them.
ASCEND_TQ_DECLARE_MM_MODE(kv3fp4, TurboQuantMode::KV3_FP4)
ASCEND_TQ_DECLARE_MM_MODE(kv5fp8, TurboQuantMode::KV5_FP8)

// The unpack ablation of the shipping decode: same launch, same traffic, no codec expand, zero output.
ASCEND_TQ_DECLARE_MM_FUSED_DECODE_ABLATED(turboquant_mm_fused_decode_nounpack_kv4fp8_half,
                                          TurboQuantMode::KV4_FP8, half, true)

// rightLayout is a GemmLayout: 0 transposes the K x N right operand into L0B, 1 takes it as N x K.
extern "C" __global__ __aicore__ void turboquant_cube_gemm_probe_fp8(
    GM_ADDR leftGm, GM_ADDR rightGm, GM_ADDR outGm, uint32_t m, uint32_t k, uint32_t n, uint32_t headSize,
    uint32_t tileRows, uint32_t leftElems, uint32_t rightElems, uint32_t outElems, uint32_t rightLayout,
    uint32_t variant)
{
    AscendC::TPipe pipe;
    TurboQuantCubeGemmProbe op(&pipe);
    op.Init(leftGm, rightGm, outGm, headSize, tileRows, leftElems, rightElems, outElems);
    op.Run(m, k, n, static_cast<GemmLayout>(rightLayout), variant);
}
#endif

#undef ASCEND_TQ_DECLARE_MM_MODE
#undef ASCEND_TQ_DECLARE_MM_FUSED_DECODE_RAW_QUERY
#undef ASCEND_TQ_DECLARE_MM_FUSED_DECODE
#undef ASCEND_TQ_DECLARE_MM_RESHAPE_AND_CACHE

namespace vllm_ascend {

void turboquant_mm_reshape_and_cache_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *key,
                                          void *value, void *keyCache, void *valueCache, void *scaleCache,
                                          void *slotMapping, void *piSigns, void *rotTables, void *modeTables,
                                          uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,
                                          uint32_t blockSize, uint32_t numBlocks, uint32_t tokensPerCore,
                                          float invSqrtLen)
{
    if (type != AscendType::FP16) {
        return;
    }
    switch (static_cast<turboquant::TurboQuantMode>(mode)) {
        case turboquant::TurboQuantMode::KV4_FP8:
            turboquant_mm_reshape_and_cache_kv4fp8_half<<<blockDim, nullptr, stream>>>(
                key, value, keyCache, valueCache, scaleCache, slotMapping, piSigns, rotTables, modeTables, numTokens,
                numKvHeads, headSize, blockSize, numBlocks, tokensPerCore, invSqrtLen);
            break;
#if defined(VLLM_ASCEND_TQ_TEST_KERNELS)
        case turboquant::TurboQuantMode::KV3_FP4:
            turboquant_mm_reshape_and_cache_kv3fp4_half<<<blockDim, nullptr, stream>>>(
                key, value, keyCache, valueCache, scaleCache, slotMapping, piSigns, rotTables, modeTables, numTokens,
                numKvHeads, headSize, blockSize, numBlocks, tokensPerCore, invSqrtLen);
            break;
        case turboquant::TurboQuantMode::KV5_FP8:
            turboquant_mm_reshape_and_cache_kv5fp8_half<<<blockDim, nullptr, stream>>>(
                key, value, keyCache, valueCache, scaleCache, slotMapping, piSigns, rotTables, modeTables, numTokens,
                numKvHeads, headSize, blockSize, numBlocks, tokensPerCore, invSqrtLen);
            break;
#endif
        default:
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
        case turboquant::TurboQuantMode::KV4_FP8:
            turboquant_mm_fused_decode_kv4fp8_half<<<blockDim, nullptr, stream>>>(
                queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,
                numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,
                tasksPerBlock, reduceTasksPerBlock, fusedContextLimit, scale, invSqrtLen);
            break;
#if defined(VLLM_ASCEND_TQ_TEST_KERNELS)
        case turboquant::TurboQuantMode::KV3_FP4:
            turboquant_mm_fused_decode_kv3fp4_half<<<blockDim, nullptr, stream>>>(
                queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,
                numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,
                tasksPerBlock, reduceTasksPerBlock, fusedContextLimit, scale, invSqrtLen);
            break;
        case turboquant::TurboQuantMode::KV5_FP8:
            turboquant_mm_fused_decode_kv5fp8_half<<<blockDim, nullptr, stream>>>(
                queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,
                numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,
                tasksPerBlock, reduceTasksPerBlock, fusedContextLimit, scale, invSqrtLen);
            break;
#endif
        default:
            break;
    }
}

#if defined(VLLM_ASCEND_TQ_TEST_KERNELS)
// The unpack ablation's launch. Deliberately a separate entry point rather than a flag on the shipping one:
// nothing outside csrc/tests can reach a kernel whose output is meaningless by construction.
void turboquant_mm_fused_decode_nounpack_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim,
                                             void *queryRot, void *keyCache, void *valueCache, void *scaleCache,
                                             void *blockTables, void *contextLens, void *modeTables, void *workspace,
                                             void *output, uint32_t numTokens, uint32_t numHeads,
                                             uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,
                                             uint32_t maxBlocksPerSeq, uint32_t numSplits, uint32_t headsPerTask,
                                             uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,
                                             uint32_t fusedContextLimit, float scale, float invSqrtLen)
{
    if (type != AscendType::FP16 || blockDim == 0 ||
        static_cast<turboquant::TurboQuantMode>(mode) != turboquant::TurboQuantMode::KV4_FP8) {
        return;
    }
    turboquant_mm_fused_decode_nounpack_kv4fp8_half<<<blockDim, nullptr, stream>>>(
        queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,
        numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,
        tasksPerBlock, reduceTasksPerBlock, fusedContextLimit, scale, invSqrtLen);
}
#endif

void turboquant_mm_fused_decode_raw_query_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim,
                                              void *query, void *piSigns, void *rotTables, void *h16, void *gate,
                                              void *queryRot, void *keyCache, void *valueCache, void *scaleCache,
                                              void *blockTables, void *contextLens, void *modeTables,
                                              void *workspace, void *output, uint32_t numTokens, uint32_t numHeads,
                                              uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,
                                              uint32_t maxBlocksPerSeq, uint32_t numSplits, uint32_t headsPerTask,
                                              uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,
                                              uint32_t prologueVectorsPerBlock, uint32_t prologueCubeChunkVectors,
                                              uint32_t outputStage, uint32_t fusedContextLimit, float scale,
                                              float invSqrtLen)
{
    if (type != AscendType::FP16 || blockDim == 0 ||
        static_cast<turboquant::TurboQuantMode>(mode) != turboquant::TurboQuantMode::KV4_FP8) {
        return;
    }
    turboquant_mm_fused_decode_raw_query_kv4fp8_half<<<blockDim, nullptr, stream>>>(
        query, piSigns, rotTables, h16, gate, queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens,
        modeTables, workspace, output, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
        headsPerTask, tasksPerBlock, reduceTasksPerBlock, prologueVectorsPerBlock, prologueCubeChunkVectors,
        outputStage, fusedContextLimit, scale, invSqrtLen);
}

#if defined(VLLM_ASCEND_TQ_TEST_KERNELS)
void turboquant_cube_gemm_probe_impl(void *stream, void *leftGm, void *rightGm, void *outGm, uint32_t m, uint32_t k,
                                     uint32_t n, uint32_t headSize, uint32_t tileRows, uint32_t leftElems,
                                     uint32_t rightElems, uint32_t outElems, uint32_t rightLayout, uint32_t variant)
{
    turboquant_cube_gemm_probe_fp8<<<1, nullptr, stream>>>(leftGm, rightGm, outGm, m, k, n, headSize, tileRows,
                                                           leftElems, rightElems, outElems, rightLayout, variant);
}
#endif

}
