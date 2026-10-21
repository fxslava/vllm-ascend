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

// The modulo-scheduled fused TurboQuant decode: the same arithmetic as turboquant_fused_decode.cpp's
// kv4fp8 instance, on a three-stage software pipeline instead of the two-stage one.
//
// WHAT CHANGES, AND WHY
//
// The shipping ring schedule (TurboQuantFusedDecode::StageTilesRing) already runs the KV read a tile ahead
// of the stage, so MTE2 overlaps the tile it feeds. What it cannot overlap is the Cube against the vector
// unit: a tile's score GEMM, its softmax, its context GEMM and its accumulate form one chain, and with a
// single Fixpipe score window the next tile's score GEMM cannot start until this tile's softmax has
// consumed -- and overwritten, in place -- the window it would write. The Cube idles through every
// softmax; the vector unit idles through every GEMM pair.
//
// This file breaks the chain with a second score window. In the steady state, iteration k is
//
//   MTE2        reads tile k + kCubePipelineSlots into ingest ring slot k % kCubePipelineSlots
//   AIC Cube    GemmScores(tile k + 1) into score window (k + 1) % kCubeScoreWindows,
//               then GemmContext(tile k) into the context window
//   AIV Vector  the online softmax of tile k out of window k % kCubeScoreWindows, the exponent decay,
//               the L1 stage of tile k + 2, and the accumulate of tile k
//
// and all three are in flight at once: the score GEMM of k + 1 writes the window the softmax of k is not
// reading, the context GEMM of k runs against the vector unit's decay and L1 stage, and the MTE2 burst for
// k + kCubePipelineSlots runs against both.
//
// The accumulate of tile k stays in iteration k rather than sliding to k + 1, which is what keeps the
// online-softmax state single-buffered: alpha and the probability scale are per-tile, held in one state
// field each, and deferring the accumulate would need a second copy of both. Deferring it buys nothing
// here anyway -- the Cube is already covered by the next tile's score GEMM.
//
// WHAT THE SCHEDULE COSTS
//
// Per ring slot: kCubeTileRows * packedBytes of UB for the packed plane and kCubeTileRows * scaleSlot
// floats for the scale lanes (about 10 KB together at headSize 256), plus 2 * kCubeTileRows * headSize
// bytes of L1 for the key and value operands (32 KB). Three slots is 30 KB of the 256 KB UB partition and
// 96 KB of the 512 KB L1, so neither buffer is what bounds the ring: the cross-core flag space is. A flag
// id is four bits wide and the toolkit reserves 11..14, and one id per slot and per window spends
// 2 * kCubePipelineSlots + kCubeScoreWindows + 2 = 10 of the 11 that are left (turboquant_layout.h).
//
// WHAT IS NOT CLAIMED
//
// Nothing here has executed. The schedule is argued, not measured: the token balance is a hand proof over
// the marked graph the two halves form, not a camodel run, and the overlap above is what the dependence
// edges allow, not a profile. Against the shipping decode this is a new kernel, and it has to clear the
// fused goldens (TURBOQUANT_TESTS.md cases (a)-(e)) on the camodel before any claim about it is worth
// making. No build wires this file in yet.

#include "kernel_operator.h"

#include "../../../kernels/types.h"
#include "common/turboquant_codec_mx.h"
#include "common/turboquant_launch.h"
#include "cube/turboquant_cube_service.h"
#include "cube/turboquant_query_basis.h"
#include "vector/turboquant_vector_service.h"

using vllm_ascend::turboquant::CeilDiv;
using vllm_ascend::turboquant::DecodeUnpack;
using vllm_ascend::turboquant::DecodeUnpackGathers;
using vllm_ascend::turboquant::DecodeUnpackIsLloydMax;
using vllm_ascend::turboquant::kBankSkewBytes;
using vllm_ascend::turboquant::kCubePipelineSlots;
using vllm_ascend::turboquant::kCubeScoreWindows;
using vllm_ascend::turboquant::kCubeTileM;
using vllm_ascend::turboquant::kCubeTileRows;
using vllm_ascend::turboquant::kOptFlagContextReady;
using vllm_ascend::turboquant::kOptFlagProbsReady;
using vllm_ascend::turboquant::kOptFlagReservedFirst;
using vllm_ascend::turboquant::kOptFlagScoresReady;
using vllm_ascend::turboquant::kOptFlagSlotFree;
using vllm_ascend::turboquant::kOptFlagSlotReady;
using vllm_ascend::turboquant::kSubBlockSyncMode;
using vllm_ascend::turboquant::kVectorSubcoresPerBlock;
using vllm_ascend::turboquant::MixBlockIdx;
using vllm_ascend::turboquant::SyncVectorToMte2;
using vllm_ascend::turboquant::TurboQuantCubeMm;
using vllm_ascend::turboquant::TurboQuantMode;
using vllm_ascend::turboquant::TurboQuantModeCodec;
using vllm_ascend::turboquant::TurboQuantPartialReducer;
using vllm_ascend::turboquant::TurboQuantQueryBasis;
using vllm_ascend::turboquant::TurboQuantTaskHeads;
using vllm_ascend::turboquant::TurboQuantVectorDecodeService;

namespace {

constexpr uint32_t PipelineGcd(const uint32_t a, const uint32_t b)
{
    return b == 0 ? a : PipelineGcd(b, a % b);
}

// The period of the schedule's whole index state: the ring slot repeats every kCubePipelineSlots tiles and
// the score window every kCubeScoreWindows, so unrolling the loop body this many times makes every index
// in it a compile-time constant. Six, for a three-slot ring and two windows.
constexpr uint32_t kPipelinePhases =
    kCubePipelineSlots * kCubeScoreWindows / PipelineGcd(kCubePipelineSlots, kCubeScoreWindows);

// The modulo-scheduled decode. MODE and UNPACK carry the same meaning they do on TurboQuantFusedDecode,
// and PRE_ROTATED the same two query bases: true takes the rotated fp32 query, false rotates the raw query
// once in a prologue and selects the output stage per launch (cube/turboquant_query_basis.h).
//
// The schedule is RunCubePhase and RunVectorPhase, and the two have to be read together: every flag is one
// edge of the marked graph they form, and the only reason the loop needs no draining barrier is that each
// edge is point-to-point. Both cores derive the same slot and window index from the same template
// parameter, so neither is ever told which one the other picked.
template <TurboQuantMode MODE, typename scalar_t, bool PRE_ROTATED = true,
          DecodeUnpack UNPACK = DecodeUnpack::kNative>
class TurboQuantFusedDecodeOpt {
public:
    static constexpr uint32_t kSlots = kCubePipelineSlots;
    static constexpr uint32_t kWindows = kCubeScoreWindows;
    static constexpr uint32_t kPhases = kPipelinePhases;

    using Mm = TurboQuantCubeMm<MODE, kSlots>;
    using Codec = TurboQuantModeCodec<MODE, DecodeUnpackGathers(UNPACK), DecodeUnpackIsLloydMax(UNPACK)>;
    using Vector =
        TurboQuantVectorDecodeService<MODE, scalar_t, Mm, Codec, UNPACK, kSlots, kWindows, kBankSkewBytes>;
    using Reducer = TurboQuantPartialReducer<scalar_t>;
    using Basis = TurboQuantQueryBasis<scalar_t, !PRE_ROTATED>;

    // The ring ingest and the NZ-tiled plane this schedule streams are kv4fp8's. A codebook mode reads its
    // slots row-major and expands them in bands, which is a different ingest and a different pipeline.
    static_assert(Vector::kBatched, "the modulo schedule is the batched (kv4fp8) ingest only");
    static_assert(Vector::kIngestSlots == kSlots, "the UB ring and the L1 ring have to be the same depth");
    static_assert(Vector::kScoreWindows == kWindows, "the vector service did not take the window count");
    static_assert(Mm::kSlots == kSlots, "the Cube's L1 ring and the ingest ring have to agree");
    static_assert(kOptFlagProbsReady < kOptFlagReservedFirst,
                  "the schedule's flag ids run into the ids the toolkit reserves for its own barriers");
    static_assert(kWindows >= 2, "one score window is the lock-step schedule, not this one");
    static_assert(kSlots >= 3, "a two-slot ring cannot keep a read, a stage and a GEMM pair in flight");

    __aicore__ inline explicit TurboQuantFusedDecodeOpt(AscendC::TPipe *pipe) : pipe_(pipe) {}

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
        mm_.InitUnitScales(pipe_);
        // No launch-boundary flush. The lock-step decode drains every pipe here, which is the one place a
        // PipeBarrier<PIPE_ALL> is cheap but also the one place it hides what it is for. On this
        // instantiation there is nothing in flight to drain: Init only allocates (compile-time offsets, no
        // traffic), kv4fp8's affine codec copies no table at all, InitUnitScales is fp4-only, and the two
        // instantiations whose codec does copy a 64 B table fence it inside their own Init
        // (turboquant_codec_mx.h). The first thing to move data after this point is StageQuery, which
        // posts its own MTE2 -> V edge.
    }

    // The raw-query decode's two basis stages: InitBasis after Init, RotateQuery before Process.
    __aicore__ inline void InitBasis(__gm__ void *query, __gm__ void *piSigns, __gm__ void *rotTables,
                                     __gm__ void *h16, __gm__ void *gate, __gm__ void *queryRot,
                                     const uint32_t cubeChunkVectors, const uint32_t outputStage,
                                     const float invSqrtLen)
    {
        static_assert(Basis::kEnabled, "a pre-rotated decode has no query rotation");
        basis_.Init(pipe_, query, piSigns, rotTables, h16, gate, queryRot, numHeads_, headSize_, cubeChunkVectors,
                    outputStage, invSqrtLen);
    }

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

    // Grid-uniform, like the lock-step decode's: every block reads every token's length, so the SyncAll
    // the caller guards with this is entered by all of them or by none.
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

    // Everything a phase of either half needs that is not its phase index. The cursors advance across
    // phases, so they are carried by reference; holding the rest alongside them keeps an unrolled phase
    // from reloading the task's extents from the frame on every tile.
    struct PipelineTask {
        TileCursor read;
        TileCursor consume;
        uint32_t token = 0;
        uint32_t contextLen = 0;
        uint32_t blockEnd = 0;
        uint32_t numTiles = 0;
        uint32_t kvHead = 0;
        uint32_t tile = 0;
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
            // Nothing drains the vector unit between tasks, and the previous task's vector work may still
            // read the query buffer and an ingest slot. This task's reads must land behind it.
            SyncVectorToMte2();
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
            PipelineTask task;
            task.token = token;
            task.contextLen = static_cast<uint32_t>(contextLen);
            task.blockEnd = blockEnd;
            task.kvHead = kvHead;
            task.numTiles = CountTiles(token, task.contextLen, blockStart, blockEnd);
            task.read.block = blockStart;
            task.consume.block = blockStart;

            if ASCEND_IS_AIV {
                vector_.StageQuery(mm_, token, kvHead, heads);
                const uint32_t tail = task.contextLen % kCubeTileRows;
                if (heads.mine > 0 && blockEnd == seqBlocks && tail != 0) {
                    vector_.ComputeTailMask(tail);
                }
            }
            // The one condition both cores branch on, so the bypass stays lock-step. A block range can
            // hold no tile at all -- every blockTable entry in it negative, which is what a padding or an
            // unmapped block carries -- and then neither core may touch a flag: an AIV that skipped the
            // pipeline while the AIC still entered it would leave the AIC waiting on a slot nothing is
            // ever going to stage, which is a hung launch rather than a wrong answer. CountTiles is safe
            // to branch on here because both cores run it over the same GM words with the same indices.
            if (task.numTiles > 0) {
                if ASCEND_IS_AIV {
                    RunVectorPipeline(task, heads);
                }
                if ASCEND_IS_AIC {
                    RunCubePipeline(task, heads.rows);
                }
            }
        }

        if ASCEND_IS_AIV {
            vector_.StageTaskOutput(token, split, heads, fused, basis_);
        }
    }

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

    // -----------------------------------------------------------------------------------------------
    // The AIC half.
    //
    // Prologue: the score GEMM of tile 0, because the vector unit's first phase waits on its window.
    // Phase k: the score GEMM of tile k + 1 into the window tile k does not own, then the context GEMM of
    // tile k. That order matters -- the score GEMM depends on nothing the vector unit is doing right now,
    // so it goes first and the wait for tile k's probabilities lands as late as it can.
    // -----------------------------------------------------------------------------------------------
    __aicore__ inline void RunCubePipeline(PipelineTask &task, const uint32_t rows)
    {
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagSlotReady));
        mm_.template GemmScores<true>(vector_.Scores(0), mm_.L1Key(0), rows, headSize_, kCubeTileRows);
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(static_cast<uint16_t>(kOptFlagScoresReady));

        while (task.tile + kPhases <= task.numTiles) {
            RunCubePeriod<0>(task, rows);
            task.tile += kPhases;
        }
        RunCubeTail<0>(task, rows, task.numTiles - task.tile);
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunCubePeriod(PipelineTask &task, const uint32_t rows)
    {
        if constexpr (PHASE < kPhases) {
            RunCubePhase<PHASE>(task, rows, task.tile + PHASE);
            RunCubePeriod<PHASE + 1>(task, rows);
        }
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunCubeTail(PipelineTask &task, const uint32_t rows, const uint32_t remaining)
    {
        if constexpr (PHASE < kPhases) {
            if (PHASE < remaining) {
                RunCubePhase<PHASE>(task, rows, task.tile + PHASE);
                RunCubeTail<PHASE + 1>(task, rows, remaining);
            }
        }
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunCubePhase(PipelineTask &task, const uint32_t rows, const uint32_t tile)
    {
        constexpr uint32_t kSlot = PHASE % kSlots;
        constexpr uint32_t kNextSlot = (PHASE + 1) % kSlots;
        constexpr uint32_t kNextWindow = (PHASE + 1) % kWindows;

        if (tile + 1 < task.numTiles) {
            AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagSlotReady + kNextSlot));
            mm_.template GemmScores<true>(vector_.Scores(kNextWindow), mm_.L1Key(kNextSlot), rows, headSize_,
                                          kCubeTileRows);
            AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(
                static_cast<uint16_t>(kOptFlagScoresReady + kNextWindow));
        }

        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagProbsReady));
        mm_.template GemmContext<true>(vector_.Context(), mm_.L1Value(kSlot), rows, kCubeTileRows, headSize_);
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(static_cast<uint16_t>(kOptFlagContextReady));

        // Both of this slot's operands are in L0 now, so the slot is free -- but post the token only if a
        // later tile is going to wait for it. A token nothing consumes is an event leak across launches.
        if (tile + kSlots < task.numTiles) {
            AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE1>(
                static_cast<uint16_t>(kOptFlagSlotFree + kSlot));
        }
    }

    // -----------------------------------------------------------------------------------------------
    // The AIV half.
    //
    // Prologue: prime the ingest ring with up to kSlots reads, then stage tiles 0 and 1 into L1 -- tile 1
    // as well, because the Cube's first phase runs the score GEMM of tile 1 and must not wait on it.
    // Phase k: the softmax of tile k, the MTE2 read of tile k + kSlots into the slot that softmax just
    // finished with, the decay, the L1 stage of tile k + 2, and the accumulate of tile k.
    // -----------------------------------------------------------------------------------------------
    __aicore__ inline void RunVectorPipeline(PipelineTask &task, const TurboQuantTaskHeads &heads)
    {
        const uint32_t primed = task.numTiles < kSlots ? task.numTiles : kSlots;
        for (uint32_t slot = 0; slot < primed; ++slot) {
            NextTile(task.read, task.token, task.contextLen, task.blockEnd);
            vector_.ReadTileToSlot(task.read.physical, task.read.base, task.kvHead, slot);
        }
        StageToL1(task, 0);
        if (task.numTiles > 1) {
            StageToL1(task, 1);
        }

        while (task.tile + kPhases <= task.numTiles) {
            RunVectorPeriod<0>(task, heads);
            task.tile += kPhases;
        }
        RunVectorTail<0>(task, heads, task.numTiles - task.tile);
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunVectorPeriod(PipelineTask &task, const TurboQuantTaskHeads &heads)
    {
        if constexpr (PHASE < kPhases) {
            RunVectorPhase<PHASE>(task, heads, task.tile + PHASE);
            RunVectorPeriod<PHASE + 1>(task, heads);
        }
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunVectorTail(PipelineTask &task, const TurboQuantTaskHeads &heads,
                                         const uint32_t remaining)
    {
        if constexpr (PHASE < kPhases) {
            if (PHASE < remaining) {
                RunVectorPhase<PHASE>(task, heads, task.tile + PHASE);
                RunVectorTail<PHASE + 1>(task, heads, remaining);
            }
        }
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunVectorPhase(PipelineTask &task, const TurboQuantTaskHeads &heads, const uint32_t tile)
    {
        constexpr uint32_t kSlot = PHASE % kSlots;
        constexpr uint32_t kWindow = PHASE % kWindows;
        constexpr uint32_t kStageSlot = (PHASE + 2) % kSlots;

        NextTile(task.consume, task.token, task.contextLen, task.blockEnd);

        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagScoresReady + kWindow));
        vector_.ComputeSoftmaxAndStageProbs(mm_, task.consume.valid, heads, kSlot, kWindow);
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(static_cast<uint16_t>(kOptFlagProbsReady));

        // The run-ahead read. Slot kSlot held tile `tile`: its packed plane was consumed by the L1 stage
        // two phases ago and its scale lanes by the softmax just above, so the slot is dead and the one
        // edge the read needs is V -> MTE2 against this subcore's own vector work.
        if (tile + kSlots < task.numTiles) {
            SyncVectorToMte2();
            NextTile(task.read, task.token, task.contextLen, task.blockEnd);
            vector_.ReadTileToSlot(task.read.physical, task.read.base, task.kvHead, kSlot);
        }

        vector_.ComputeAccumulatorDecay(heads);

        // Tile `tile + 2`'s L1 operands, two tiles ahead, so the Cube's next phase finds them staged. The
        // L1 slot it writes last held tile `tile - 1`, which the Cube released with kOptFlagSlotFree.
        if (tile + 2 < task.numTiles) {
            if (tile + 2 >= kSlots) {
                AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagSlotFree + kStageSlot));
            }
            StageToL1(task, kStageSlot);
        }

        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagContextReady));
        vector_.ComputeAccumulate(heads);
    }

    // The slot's MTE2 read is already posted on the slot's own event id; StageSlotToL1 waits exactly that
    // one, unpacks the plane in UB in NZ order and bursts it into L1. Both subcores set the slot's token,
    // because the AIC's single wait needs one from each of them.
    __aicore__ inline void StageToL1(PipelineTask &task, const uint32_t slot)
    {
        vector_.StageSlotToL1(mm_, task.kvHead, slot);
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(static_cast<uint16_t>(kOptFlagSlotReady + slot));
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

}

#define ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT(NAME, MODE, TYPE)                                                      \
    extern "C" __global__ __aicore__ void NAME(                                                                      \
        GM_ADDR queryRot, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR blockTables,             \
        GM_ADDR contextLens, GM_ADDR modeTables, GM_ADDR workspace, GM_ADDR output, uint32_t numTokens,              \
        uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,      \
        uint32_t numSplits, uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,              \
        uint32_t fusedContextLimit, float scale, float invSqrtLen)                                                   \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantFusedDecodeOpt<MODE, TYPE> op(&pipe);                                                               \
        op.Init(queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,  \
                numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,       \
                fusedContextLimit, scale, invSqrtLen);                                                                \
        op.Process(tasksPerBlock);                                                                                    \
        if (op.NeedsReduction()) {                                                                                    \
            AscendC::SyncAll<false>();                                                                                \
            if ASCEND_IS_AIV {                                                                                        \
                TurboQuantPartialReducer<TYPE> reducer(&pipe);                                                         \
                reducer.Init(workspace, output, numHeads, headSize, numSplits);                                        \
                op.Reduce(reducer, reduceTasksPerBlock);                                                               \
            }                                                                                                         \
        }                                                                                                             \
    }

// The raw-query entry, argument for argument the same as the lock-step decode's, so a benchmark or an
// msprof run can swap one launcher for the other without touching its tiling.
#define ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT_RAW_QUERY(NAME, MODE, TYPE)                                            \
    extern "C" __global__ __aicore__ void NAME(                                                                      \
        GM_ADDR query, GM_ADDR piSigns, GM_ADDR rotTables, GM_ADDR h16, GM_ADDR gate, GM_ADDR queryRot,              \
        GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR blockTables, GM_ADDR contextLens,          \
        GM_ADDR modeTables, GM_ADDR workspace, GM_ADDR output, uint32_t numTokens, uint32_t numHeads,                \
        uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,     \
        uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,                                 \
        uint32_t prologueVectorsPerBlock, uint32_t prologueCubeChunkVectors, uint32_t outputStage,                   \
        uint32_t fusedContextLimit, float scale, float invSqrtLen)                                                   \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantFusedDecodeOpt<MODE, TYPE, false> op(&pipe);                                                        \
        op.Init(queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,  \
                numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,       \
                fusedContextLimit, scale, invSqrtLen);                                                                \
        op.InitBasis(query, piSigns, rotTables, h16, gate, queryRot, prologueCubeChunkVectors, outputStage,           \
                     invSqrtLen);                                                                                     \
        op.RotateQuery(prologueVectorsPerBlock);                                                                       \
        op.Process(tasksPerBlock);                                                                                     \
        if (op.NeedsReduction()) {                                                                                     \
            AscendC::SyncAll<false>();                                                                                 \
            if ASCEND_IS_AIV {                                                                                         \
                TurboQuantPartialReducer<TYPE> reducer(&pipe);                                                          \
                reducer.Init(workspace, output, numHeads, headSize, numSplits);                                         \
                op.Reduce(reducer, reduceTasksPerBlock);                                                                \
            }                                                                                                          \
        }                                                                                                              \
    }

// kv4fp8 only: the NZ-tiled ring ingest the schedule is built on, and the one mode the fused decode is
// validated in at all (TURBOQUANT_TESTS.md 13.25).
ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT(turboquant_mm_fused_decode_opt_kv4fp8_half, TurboQuantMode::KV4_FP8, half)
ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT_RAW_QUERY(turboquant_mm_fused_decode_opt_raw_query_kv4fp8_half,
                                                TurboQuantMode::KV4_FP8, half)

namespace vllm_ascend {

using turboquant::AsBytes;

void turboquant_mm_fused_decode_opt_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim,
                                         void *queryRot, void *keyCache, void *valueCache, void *scaleCache,
                                         void *blockTables, void *contextLens, void *modeTables, void *workspace,
                                         void *output, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                         uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                         uint32_t numSplits, uint32_t headsPerTask, uint32_t tasksPerBlock,
                                         uint32_t reduceTasksPerBlock, uint32_t fusedContextLimit, float scale,
                                         float invSqrtLen)
{
    if (type != AscendType::FP16 || blockDim == 0 ||
        static_cast<turboquant::TurboQuantMode>(mode) != turboquant::TurboQuantMode::KV4_FP8) {
        return;
    }
    turboquant_mm_fused_decode_opt_kv4fp8_half<<<blockDim, nullptr, stream>>>(
        AsBytes(queryRot), AsBytes(keyCache), AsBytes(valueCache), AsBytes(scaleCache), AsBytes(blockTables), AsBytes(contextLens), AsBytes(modeTables), AsBytes(workspace), AsBytes(output),
        numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,
        tasksPerBlock, reduceTasksPerBlock, fusedContextLimit, scale, invSqrtLen);
}

void turboquant_mm_fused_decode_opt_raw_query_impl(
    int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *query, void *piSigns, void *rotTables,
    void *h16, void *gate, void *queryRot, void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
    void *contextLens, void *modeTables, void *workspace, void *output, uint32_t numTokens, uint32_t numHeads,
    uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,
    uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock, uint32_t prologueVectorsPerBlock,
    uint32_t prologueCubeChunkVectors, uint32_t outputStage, uint32_t fusedContextLimit, float scale,
    float invSqrtLen)
{
    if (type != AscendType::FP16 || blockDim == 0 ||
        static_cast<turboquant::TurboQuantMode>(mode) != turboquant::TurboQuantMode::KV4_FP8) {
        return;
    }
    turboquant_mm_fused_decode_opt_raw_query_kv4fp8_half<<<blockDim, nullptr, stream>>>(
        AsBytes(query), AsBytes(piSigns), AsBytes(rotTables), AsBytes(h16), AsBytes(gate), AsBytes(queryRot), AsBytes(keyCache), AsBytes(valueCache), AsBytes(scaleCache), AsBytes(blockTables), AsBytes(contextLens),
        AsBytes(modeTables), AsBytes(workspace), AsBytes(output), numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq,
        numSplits, headsPerTask, tasksPerBlock, reduceTasksPerBlock, prologueVectorsPerBlock,
        prologueCubeChunkVectors, outputStage, fusedContextLimit, scale, invSqrtLen);
}

}
