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
// WHAT THE SCALAR UNIT NO LONGER DOES
//
// The schedule above is only worth its three slots if the SPU can issue ahead of the engines, and
// on dav-3510 it issues in order for the whole core: one integer divide, or one branch on a value
// just loaded from GM, stalls everything behind it -- the MTE2 descriptor for the next ring slot
// and the Mmad issue that keeps the Cube fed included. This operator used to spend, per task, six
// integer div/mods to decode its task id, two more to turn a context length into a logical page
// range, and a loop over that range to count its tiles; and then, per TILE and twice over, a
// NextTile walk with a GM block-table load and two boundary branches, sitting between a tile's
// softmax and the next tile's stage.
//
// All of it is host tiling now (../include/turboquant_fused_decode.hpp). The device reads a
// per-(token, split) descriptor and a per-block dispatch record, both built on the CPU, and what
// is left on the scalar unit is adds, multiplies, shifts, masks and one select per tile. There is
// no integer division anywhere on this kernel's attention path, no loop whose trip count depends
// on device data, and no diagnostic-ABI comparison in the cursor: the isolated single-tile canary
// that used to be `probePage_ != UINT32_MAX`, tested once per tile on every production launch, is
// a descriptor field.
//
// Harness integration: local service snapshot, 64-token KV4/FP8 geometry,

#include "../include/turboquant_fused_decode_opt.hpp"
#include "common/types.h"

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"

#include "common/turboquant_codec.h"
#include "cube/turboquant_cube_service.h"
#include "cube/turboquant_query_basis.h"
#include "vector/turboquant_vector_service.h"

namespace fd = ascend_micro::fused_decode;

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

// A block-local join between tasks and before returning to the scheduler.
// No other participant can reuse the remote Fixpipe landings before this join.
__aicore__ inline void DrainFusedDecodeTask()
{
    constexpr uint16_t flag = ascend_micro::fused_decode_opt::kTaskJoinFlag;
    if ASCEND_IS_AIC {
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::M_S>();
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::FIX_S>();
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(flag);
        AscendC::CrossCoreWaitFlag(flag);
        // Close the MTE1 -> Matrix -> Scalar chain explicitly, including
        // empty tasks, before the join channel can be reused.
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::MTE1_M>();
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::M_S>();
    }
    if ASCEND_IS_AIV {
        AscendC::CrossCoreWaitFlag(flag);
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::MTE2_S>();
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::V_S>();
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::MTE3_S>();
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(flag);
    }
}

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
// Read RunCubeSteadyPhase and RunVectorSteadyPhase together: every flag is one
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
    static_assert(Codec::WorkBufferWords(256, 32) * sizeof(float) ==
        ascend_micro::fused_decode_opt::kUbSizes[27], "codec workspace budget drift");
    static_assert(Vector::kBatched, "the modulo schedule is the batched (kv4fp8) ingest only");
    static_assert(Vector::kIngestSlots == kSlots, "the UB ring and the L1 ring have to be the same depth");
    static_assert(Vector::kScoreWindows == kWindows, "the vector service did not take the window count");
    static_assert(Mm::kSlots == kSlots, "the Cube's L1 ring and the ingest ring have to agree");
    static_assert(kOptFlagProbsReady < kOptFlagReservedFirst,
                  "the schedule's flag ids run into the ids the toolkit reserves for its own barriers");
    static_assert(kSlots == 3 && kWindows == 2 && kPhases == 6, "three-zone schedule geometry");
    static_assert(kCubeTileRows == fd::kDecodeTileRows &&
                      kCubeTileRows == (1u << fd::kDecodeTileRowsShift),
                  "the host tiling contract and the device ring disagree about the tile");
    static_assert(kWindows >= 2, "one score window is the lock-step schedule, not this one");
    static_assert(kSlots >= 3, "a two-slot ring cannot keep a read, a stage and a GEMM pair in flight");

    __aicore__ inline explicit TurboQuantFusedDecodeOpt(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *queryRot, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *scaleCache, __gm__ void *blockTables, __gm__ void *contextLens,
                                __gm__ void *modeTables, __gm__ void *workspace, __gm__ void *output,
                                __gm__ void *tiling, const uint32_t numTokens, const uint32_t numHeads,
                                const uint32_t numKvHeads,
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
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTables));
        contextLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(contextLens), numTokens);
        tilingGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling));

        // THE LAUNCH'S HEAD GEOMETRY, ALSO FROM THE HOST. `numHeads / numKvHeads` and the
        // CeilDiv over it were the last two runtime integer divides on this operator's device
        // side; both quantities are launch-uniform, so every descriptor carries them and this
        // reads the first one. The clamps below are compares and selects, and they match the
        // builder's: valid_launch already refuses heads_per_task > kCubeTileM, so the second
        // clamp is a guard rather than a divergence.
        groupHeads_ = tilingGm_.GetValue(fd::kTilingWordGroupHeads);
        chunksPerGroup_ = tilingGm_.GetValue(fd::kTilingWordChunksPerGroup);
        headsPerTask_ = headsPerTask < 1 ? 1 : headsPerTask;
        if (headsPerTask_ > groupHeads_) {
            headsPerTask_ = groupHeads_;
        }
        if (headsPerTask_ > kCubeTileM) {
            headsPerTask_ = kCubeTileM;
        }

        // THE BLOCK'S DISPATCH RECORD, ONCE, HERE.
        //
        // Process() used to decode its first task id with six runtime integer div/mods -- chunk,
        // splitTask, split, tokenKvHead, kvHead, token -- and then decode EVERY later task of the
        // block the same way, so a 16-task block paid 96 of them, the first six in front of the
        // first tile's MTE2 burst. The host computed the same odometer for every grid block; this
        // reads the block's entry point out of one 32-byte GM line and AdvanceTask() walks it from
        // there. Both roles read the same eight words in the same order, so neither is ever told
        // which task the other picked.
        const uint32_t record = numTokens * numSplits_ * fd::kDecodeTilingWords +
            MixBlockIdx() * fd::kDecodeDispatchWords;
        taskStart_.token = tilingGm_.GetValue(record + 0);
        taskStart_.kvHead = tilingGm_.GetValue(record + 1);
        taskStart_.split = tilingGm_.GetValue(record + 2);
        taskStart_.chunk = tilingGm_.GetValue(record + 3);
        reduceStart_[0].token = tilingGm_.GetValue(record + 4);
        reduceStart_[0].head = tilingGm_.GetValue(record + 5);
        reduceStart_[1].token = tilingGm_.GetValue(record + 6);
        reduceStart_[1].head = tilingGm_.GetValue(record + 7);

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

    // ZERO INTEGER DIVISIONS. `tasks` and `start` are multiplies; the task's four indices come
    // from the host's dispatch record and advance by AdvanceTask's odometer. What was here --
    // `task % chunksPerGroup_`, `task / chunksPerGroup_`, `% numSplits_`, `/ numSplits_`,
    // `/ numKvHeads_`, `% numKvHeads_` -- was six divides by runtime, non-power-of-two values per
    // task, issued on the SPU in front of the task's first MTE2 burst.
    __aicore__ inline void Process(const uint32_t tasksPerBlock)
    {
        const uint32_t tasks = numTokens_ * numKvHeads_ * numSplits_ * chunksPerGroup_;
        const uint32_t start = MixBlockIdx() * tasksPerBlock;
        uint32_t end = start + tasksPerBlock;
        if (end > tasks) {
            end = tasks;
        }
        TaskCursor cursor = taskStart_;
        for (uint32_t task = start; task < end; ++task) {
            ComputeTask(cursor.token, cursor.kvHead, cursor.split, cursor.chunk);
            AdvanceTask(cursor);
        }
    }

    // CAModel-only reduction-network probe schedule. Split zero processes the first 64-token
    // tile of page zero through the real attention pipeline. Every other split publishes the
    // neutral state using the production partial layout and immediately joins the same task
    // drain. The global context length remains full-sized, so every block takes the reduction.
    __aicore__ inline void ProcessReductionProbe(const uint32_t tasksPerBlock)
    {
        const uint32_t tasks = numTokens_ * numKvHeads_ * numSplits_ * chunksPerGroup_;
        const uint32_t start = MixBlockIdx() * tasksPerBlock;
        uint32_t end = start + tasksPerBlock;
        if (end > tasks) end = tasks;
        TaskCursor cursor = taskStart_;
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t offset = cursor.chunk * headsPerTask_;
            const uint32_t rows = groupHeads_ - offset < headsPerTask_ ? groupHeads_ - offset : headsPerTask_;
            const TurboQuantTaskHeads heads = Vector::Heads(cursor.kvHead * groupHeads_ + offset, rows);
            if (cursor.split == 0) {
                // Split zero's single tile is the HOST descriptor's doing now (num_tiles = 1,
                // block_start_idx = 0, tile_origin = 0), not a context-length override argument
                // that the device clipped a page against.
                ComputeTask(cursor.token, cursor.kvHead, cursor.split, cursor.chunk);
            } else {
                vector_.WriteNeutralPartial(cursor.token, cursor.split, heads);
                DrainFusedDecodeTask();
            }
            AdvanceTask(cursor);
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
        // CeilDiv by kVectorSubcoresPerBlock is a shift: it is 2 and a compile-time constant.
        // The (token, head) each half STARTS at was `task / numHeads_` and `task % numHeads_`,
        // divides by a runtime value, once per reduction task; the host put both halves' starts in
        // the dispatch record and this walks a head counter from there.
        const uint32_t firstHalf = CeilDiv(end - start, kVectorSubcoresPerBlock);
        const bool secondSubcore = AscendC::GetSubBlockIdx() != 0;
        const uint32_t from = secondSubcore ? start + firstHalf : start;
        const uint32_t to = secondSubcore ? end : start + firstHalf;
        ReduceCursor cursor = reduceStart_[secondSubcore ? 1 : 0];
        for (uint32_t task = from; task < to; ++task) {
            if (!IsFused(contextLenGm_.GetValue(cursor.token))) {
                reducer.Reduce(cursor.token, cursor.head, basis_);
            }
            if (++cursor.head == numHeads_) {
                cursor.head = 0;
                ++cursor.token;
            }
        }
    }

#if defined(ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE) && ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE
    // The Cube GEMM reduction's entry. Three things differ from Reduce() above, and all three
    // are why it needs its own method rather than a different reducer object:
    //
    //   * It is driven by BOTH roles, not `if ASCEND_IS_AIV`. The Cube does the contraction, so
    //     the caller must not guard the call -- and both roles have to walk the same TPipe
    //     allocation, because the Fixpipe's destination is an address in the vector cores' UB.
    //   * It does its own task slicing. Reduce() hands one (token, head) at a time; the batched
    //     form needs runs of consecutive live tasks, because dnNum walks one uniform source
    //     stride and a fused token's id gap has to end a run. So it takes the whole share and
    //     the predicate, and batches internally.
    //   * The two vector subcores split each batch's rows rather than the share's tasks. The
    //     ceil/floor halving here is the Fixpipe's, not this loop's.
    //
    // IsFused is read through the same contextLenGm_ Reduce() reads, so a token this skips is
    // exactly a token that one skips.
    struct GemmCombineSelect {
        // Non-const: GlobalTensor::GetValue is not a const member in this toolkit.
        TurboQuantFusedDecodeOpt *op;
        __aicore__ inline bool NeedsCombine(const uint32_t token) const
        {
            return !op->IsFused(op->contextLenGm_.GetValue(token));
        }
    };

    template <typename GemmReducer>
    __aicore__ inline void ReduceGemm(GemmReducer &reducer, const uint32_t reduceTasksPerBlock)
    {
        GemmCombineSelect select{this};
        reducer.Reduce(numTokens_ * numHeads_, reduceTasksPerBlock, select, basis_);
    }

    // The attention phase's transient UB, for the reducer to alias its panels onto. Handed out
    // as a reference rather than an address so the reducer's own static_assert is what checks
    // it is long enough.
    __aicore__ inline auto &ReduceGemmSharedPool() { return vector_.TransientPool(); }
#endif

private:
    // The task-dispatch odometer. Fastest-varying first -- chunk, split, kv head, token -- which is
    // the order the six divisions Process() used to issue imposed, so the task stream is the same
    // stream in the same order and the trace census stays comparable across this change.
    struct TaskCursor {
        uint32_t token = 0;
        uint32_t kvHead = 0;
        uint32_t split = 0;
        uint32_t chunk = 0;
    };
    struct ReduceCursor {
        uint32_t token = 0;
        uint32_t head = 0;
    };

    __aicore__ inline void AdvanceTask(TaskCursor &cursor) const
    {
        if (++cursor.chunk < chunksPerGroup_) return;
        cursor.chunk = 0;
        if (++cursor.split < numSplits_) return;
        cursor.split = 0;
        if (++cursor.kvHead < numKvHeads_) return;
        cursor.kvHead = 0;
        ++cursor.token;
    }

    // Everything a phase of either half needs that is not its phase index. The two tile cursors are
    // now plain counters -- the read cursor runs kSlots tiles ahead of the consume cursor, which is
    // what the schedule always meant -- and the rest is the worker's host tiling, held here so an
    // unrolled phase does not reload it from the frame on every tile.
    struct PipelineTask {
        uint32_t readTile = 0;     // MTE2 cursor:    tiles 0 .. numTiles-1, in order
        uint32_t consumeTile = 0;  // softmax cursor: kSlots tiles behind it
        uint32_t token = 0;
        uint32_t kvHead = 0;
        uint32_t numTiles = 0;
        uint32_t lastTile = 0;     // numTiles - 1, so the ragged test is one compare
        uint32_t lastValid = 0;    // tail_rows, or kCubeTileRows for an aligned worker
        uint32_t tileOrigin = 0;   // the tile inside block_start_idx this worker begins at
        uint32_t pageShift = 0;    // log2(blockSize / kCubeTileRows), from the host descriptor
        uint32_t pageMask = 0;     // (1 << pageShift) - 1
        uint32_t tile = 0;         // the steady loop's phase counter, as before
    };

    __aicore__ inline bool IsFused(const int32_t contextLen) const
    {
        return numSplits_ <= 1 || contextLen <= static_cast<int32_t>(fusedContextLimit_);
    }

    // THE TASK'S ONE JOIN. Every path out of a task leaves through the DrainFusedDecodeTask()
    // below, including the paths that do no work at all:
    //
    //   * the duplicate splits of a FUSED token, which this used to leave by a bare `return`
    //     in front of the join -- symmetric across the roles, because both read the same
    //     contextLen word, but the one task exit that did not pass the rendezvous;
    //   * an UNMAPPED split, whose block range is empty because blockStart >= blockEnd;
    //   * a split whose every block-table entry is negative, so numTiles == 0 and neither role
    //     may touch a pipeline flag.
    //
    // The last two already fell through to the join; the first does now. The cost is one
    // mode-2 rendezvous per skipped split of a fused token, and what it buys is that the
    // per-task flag protocol is identical on every exit: the trace auditor's
    // dispatch == completion count then holds over the whole task stream without a case
    // analysis, and a future divergence between the two roles' view of `fused` cannot leave
    // one of them one task ahead of the other. RunTask holds the work and has no join.
    __aicore__ inline void ComputeTask(const uint32_t token, const uint32_t kvHead, const uint32_t split,
                                       const uint32_t chunk)
    {
        // The one GM scalar read left in the task prologue, and it is not arithmetic: `fused`
        // decides whether StageTaskOutput writes the token's output directly or leaves a partial,
        // and the reducers' own predicate reads the SAME word, so hoisting a second copy of it
        // into the descriptor would create exactly the two-sources-of-truth divergence the task
        // join below exists to make impossible.
        const bool fused = IsFused(contextLenGm_.GetValue(token));
        if (!fused || split == 0) {
            RunTask(token, kvHead, split, chunk, fused);
        }
        DrainFusedDecodeTask();
    }

    __aicore__ inline void RunTask(const uint32_t token, const uint32_t kvHead, const uint32_t split,
                                   const uint32_t chunk, const bool fused)
    {
        const uint32_t offset = chunk * headsPerTask_;
        const uint32_t rows = groupHeads_ - offset < headsPerTask_ ? groupHeads_ - offset : headsPerTask_;
        const TurboQuantTaskHeads heads = Vector::Heads(kvHead * groupHeads_ + offset, rows);

        if ASCEND_IS_AIV {
            // Nothing drains the vector unit between tasks, and the previous task's vector work may still
            // read the query buffer and an ingest slot. This task's reads must land behind it.
            SyncVectorToMte2();
            vector_.BeginTask(heads);
        }

        // THE WORKER'S TILING, READ RATHER THAN DERIVED.
        //
        // Six scalar loads out of ONE 64-byte GM line, on both roles, replacing: two integer
        // divisions by runtime values (CeilDiv by blockSize_, then by numSplits_), the
        // multiply-and-clamp that turned those into a logical page range, and CountTiles' loop
        // over that whole range with a block-table load, a sign test, a live-row clamp and a
        // CeilDiv inside it. The host computed all of it from the context lengths it already has.
        const uint32_t descriptor = (token * numSplits_ + split) * fd::kDecodeTilingWords;
        const uint32_t numTiles = tilingGm_.GetValue(descriptor + fd::kTilingWordNumTiles);
        const uint32_t tailRows = tilingGm_.GetValue(descriptor + fd::kTilingWordTailRows);
        const uint32_t blockStart = tilingGm_.GetValue(descriptor + fd::kTilingWordBlockStart);
        const uint32_t blocksInSplit = tilingGm_.GetValue(descriptor + fd::kTilingWordBlocksInSplit);
        const uint32_t tileOrigin = tilingGm_.GetValue(descriptor + fd::kTilingWordTileOrigin);
        const uint32_t pageShift = tilingGm_.GetValue(descriptor + fd::kTilingWordTilesPerPageShift);

        // The split's page entries, fetched once, here, by both roles, over exactly the span the
        // host says is live -- no scan for where the span ends, and nothing read past it. The
        // sweep also decides `pagesMapped_`, which is the kernel's own guard against a device
        // block table that disagrees with what the host validated: an unmapped page inside a live
        // span is not a geometry the closed-form cursor can express, so the task is skipped whole
        // rather than indexed through. Both roles sweep the same words in the same order, so the
        // predicate they branch on below comes from one read each.
        PrefetchBlockTable(token, blockStart, blocksInSplit);

        // The one condition both cores branch on, so the bypass stays lock-step. A worker can hold
        // no tile at all -- a split past the end of the sequence, a duplicate split of a fused
        // token, a zero-length context -- and then neither core may touch a flag: an AIV that
        // skipped the pipeline while the AIC still entered it would leave the AIC waiting on a
        // slot nothing is ever going to stage, which is a hung launch rather than a wrong answer.
        // It is safe to branch on here because it is a WORD BOTH ROLES READ from the same place,
        // which is strictly stronger than the old "both cores run the same loop over the same GM".
        if (numTiles > 0 && pagesMapped_) {
            PipelineTask task;
            task.token = token;
            task.kvHead = kvHead;
            task.numTiles = numTiles;
            task.lastTile = numTiles - 1;
            task.lastValid = tailRows != 0 ? tailRows : kCubeTileRows;
            task.tileOrigin = tileOrigin;
            task.pageShift = pageShift;
            task.pageMask = (1u << pageShift) - 1u;

            if ASCEND_IS_AIV {
                vector_.StageQuery(mm_, token, kvHead, heads);
                // tail_rows is nonzero on exactly the worker that carries the sequence end, which
                // is what `blockEnd == seqBlocks && contextLen % 64` used to establish on device.
                if (heads.mine > 0 && tailRows != 0) {
                    vector_.ComputeTailMask(tailRows);
                }
            }
            if ASCEND_IS_AIV {
                RunVectorPipeline(task, heads);
            }
            if ASCEND_IS_AIC {
                RunCubePipeline(task, heads.rows);
            }
        }

        if ASCEND_IS_AIV {
            vector_.StageTaskOutput(token, split, heads, fused, basis_);
        }
    }

    // THE BLOCK TABLE, OUT OF THE PIPELINE.
    //
    // CountTiles and NextTile both used to read blockTableGm_ with GetValue -- a scalar load
    // from Global Memory -- and NextTile ran INSIDE the steady loop, once per tile on each of
    // the two cursors. So one logical page's entry was fetched three times per task (the count,
    // on both roles; the read cursor; the consume cursor) and two of those three fetches sat
    // between a softmax and the next tile's stage, on the scalar queue, inside the dependence
    // chain the three-zone schedule exists to keep full.
    //
    // The task prologue sweeps the host-declared live span once into this scalar cache and the
    // closed-form cursor reads local memory. The sweep is still GetValue rather than a DataCopy
    // into UB on purpose: the Cube GEMM reduction's overlay already spends the only transient UB
    // this operator has, and a UB landing would make the AIC walk a TPipe allocation the AIV does
    // not -- the fault that made that reduction's first Cube run target an unallocated tensor.
    //
    // WHAT CHANGED HERE. The span is `blocks_in_split` from the descriptor rather than a
    // blockStart/blockEnd pair the device derived, so nothing is read to discover where the span
    // ends and nothing past it is touched; and the sweep ORs the entries' sign bits, which is the
    // density check the closed-form cursor needs and costs one OR per page on a loop that was
    // already loading every one of them.
    //
    // A split longer than the cache falls back to the direct read, per TILE rather than per page,
    // and that fallback is the one scalar GM load still reachable from the steady loop. 64 entries
    // covers every geometry the validation ladder and the benchmark's split sweeps reach except
    // one: 16384 context tokens at 128-token pages is 128 blocks, so splits >= 2 is inside the
    // cache and splits == 1 at that context is the row that falls back. 256 bytes of scalar local
    // memory, which is why it is 64 entries and not 256.
    static constexpr uint32_t kBlockCacheEntries = 64;

    __aicore__ inline void PrefetchBlockTable(const uint32_t token, const uint32_t blockStart,
                                              const uint32_t blocksInSplit)
    {
        blockCacheBase_ = blockStart;
        pagesMapped_ = true;
        blockCacheCount_ = blocksInSplit < kBlockCacheEntries ? blocksInSplit : kBlockCacheEntries;
        if (blockCacheCount_ == 0) {
            return;
        }
        const uint64_t row = static_cast<uint64_t>(token) * maxBlocksPerSeq_;
        int32_t signs = 0;
        for (uint32_t i = 0; i < blockCacheCount_; ++i) {
            const int32_t entry = blockTableGm_.GetValue(row + blockStart + i);
            blockCache_[i] = entry;
            signs |= entry;
        }
        pagesMapped_ = signs >= 0;
    }

    // THE TILE CURSOR, CLOSED FORM.
    //
    // NextTile was a `while (cursor.block < blockEnd)` loop carrying four mutable fields and an
    // `active` flag, with a block-table load, a sign test, a live-row clamp and two boundary
    // branches per call -- and it ran TWICE per tile in the steady state, once for the read cursor
    // and once for the consume cursor, between a tile's softmax and the next tile's stage. The
    // host's descriptor makes the tile-to-page map affine, so a cursor is a counter and this is
    // the whole of the arithmetic:
    //
    //   absolute = tile_origin + tile      one add
    //   page     = absolute >> shift       one shift      (blockSize is a power of two: enforced
    //   row base = (absolute & mask) << 6  one and, shift  by valid_launch and build_decode_tiling)
    //
    // No loop, no branch, no mutable cursor state, no sign test -- an unmapped page inside a live
    // span never reaches here because the prologue refused the task -- and no GM load while the
    // span is inside the scalar cache.
    __aicore__ inline uint32_t TilePhysical(const PipelineTask &task, const uint32_t absolute)
    {
        const uint32_t page = absolute >> task.pageShift;
        if (page < blockCacheCount_) {
            return static_cast<uint32_t>(blockCache_[page]);
        }
        return static_cast<uint32_t>(blockTableGm_.GetValue(
            static_cast<uint64_t>(task.token) * maxBlocksPerSeq_ + blockCacheBase_ + page));
    }

    // Short tasks exist in isolated probes and empty/short Split-K workers.
    // Their bounded dispatch is outside the three-zone fast path.
    __aicore__ inline void RunCubeShortPipeline(PipelineTask &, const uint32_t rows,
                                               const uint32_t count)
    {
        CubeScores<0>(rows);
        if (count > 1) CubeScores<1>(rows);
        CubeContext<0>(rows);
        if (count > 2) CubeScores<2>(rows);
        if (count > 1) CubeContext<1>(rows);
        if (count > 2) CubeContext<2>(rows);
    }

    __aicore__ inline void RunVectorShortPipeline(PipelineTask &task, const TurboQuantTaskHeads &heads)
    {
        ReadPipelineTile<0>(task);
        if (task.numTiles > 1) ReadPipelineTile<1>(task);
        StageToL1(task, 0);
        if (task.numTiles > 2) ReadPipelineTile<2>(task);
        if (task.numTiles > 1) StageToL1(task, 1);
        VectorSoftmax<0>(task, heads);
        vector_.ComputeAccumulatorDecay(heads);
        if (task.numTiles > 2) StageToL1(task, 2);
        VectorAccumulate(heads);
        if (task.numTiles > 1) {
            VectorSoftmax<1>(task, heads);
            vector_.ComputeAccumulatorDecay(heads);
            VectorAccumulate(heads);
        }
        if (task.numTiles > 2) {
            VectorSoftmax<2>(task, heads);
            vector_.ComputeAccumulatorDecay(heads);
            VectorAccumulate(heads);
        }
    }

    // Deep tasks have three zones. Short/empty workers retain the bounded
    // compatibility path; no boundary predicate enters a steady phase.
    __aicore__ inline void RunCubePipeline(PipelineTask &task, const uint32_t rows)
    {
        if (task.numTiles < kSlots + 1) {
            RunCubeShortPipeline(task, rows, task.numTiles);
            return;
        }
        // Slot 2 has never been staged. Prime its initial free credit so even
        // phase zero uses the same unconditional stage wait as every wrap.
        ReleaseCubeSlot<2>();
        CubeScores<0>(rows); // tick -1; ticks -3/-2 belong to MTE2/MTE3.
        const uint32_t steady_limit = task.numTiles - kSlots;
        while (task.tile + kPhases <= steady_limit) {
            RunCubeSteadyPhase<0>(rows);
            RunCubeSteadyPhase<1>(rows);
            RunCubeSteadyPhase<2>(rows);
            RunCubeSteadyPhase<3>(rows);
            RunCubeSteadyPhase<4>(rows);
            RunCubeSteadyPhase<5>(rows);
            task.tile += kPhases;
        }
        RunCubeSteadyTail<0>(rows, steady_limit - task.tile);
    }

    __aicore__ inline void RunVectorPipeline(PipelineTask &task, const TurboQuantTaskHeads &heads)
    {
        if (task.numTiles < kSlots + 1) {
            RunVectorShortPipeline(task, heads);
            return;
        }
        // Exactly three fill ticks; no length checks in the prologue.
        ReadPipelineTile<0>(task);                         // -3: read T0
        ReadPipelineTile<1>(task); StageToL1(task, 0);     // -2: read T1, stage T0
        ReadPipelineTile<2>(task); StageToL1(task, 1);     // -1: read T2, stage T1
        const uint32_t steady_limit = task.numTiles - kSlots;
        while (task.tile + kPhases <= steady_limit) {
            RunVectorSteadyPhase<0>(task, heads);
            RunVectorSteadyPhase<1>(task, heads);
            RunVectorSteadyPhase<2>(task, heads);
            RunVectorSteadyPhase<3>(task, heads);
            RunVectorSteadyPhase<4>(task, heads);
            RunVectorSteadyPhase<5>(task, heads);
            task.tile += kPhases;
        }
        RunVectorSteadyTail<0>(task, heads, steady_limit - task.tile);
    }

    template <uint32_t PHASE>
    __aicore__ inline void CubeScores(const uint32_t rows)
    {
        constexpr uint32_t kSlot = PHASE % kSlots;
        constexpr uint32_t kWindow = PHASE % kWindows;
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagSlotReady + kSlot));
        mm_.template GemmScores<true>(vector_.Scores(kWindow), mm_.L1Key(kSlot), rows, headSize_, kCubeTileRows);
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::FIX_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(
            static_cast<uint16_t>(kOptFlagScoresReady + kWindow));
    }

    template <uint32_t PHASE>
    __aicore__ inline void CubeContext(const uint32_t rows)
    {
        constexpr uint32_t kSlot = PHASE % kSlots;
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagProbsReady));
        mm_.template GemmContext<true>(vector_.Context(), mm_.L1Value(kSlot), rows, kCubeTileRows, headSize_);
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::FIX_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(static_cast<uint16_t>(kOptFlagContextReady));
    }

    // THE SLOT'S FREE CREDIT. The FIX_S drain in front of the announcement is RETAINED, and
    // this comment is why -- the audit found it redundant and the finding was not taken.
    //
    // A SyncEvent<FIX_S> stalls scalar ISSUE until the Fixpipe queue retires, which makes it the
    // most expensive thing this operator can put on the AIC's instruction stream, and there are
    // three per steady phase: CubeScores', CubeContext's, and this one. This one looks redundant
    // at both of its call sites. In RunCubeSteadyPhase it is issued immediately after
    // CubeContext<PHASE>, which has already drained FIX before setting kOptFlagContextReady, and
    // nothing between the two issues a Fixpipe instruction; in RunCubePipeline's prologue it
    // primes slot 2's credit before any Fixpipe work exists at all. Removing it would take one
    // scalar-issue stall per tile off the core that must not stall.
    //
    // WHY IT STAYS. CrossCoreSetFlag issues on the FLOWCTRL/scalar unit in both modes -- mode 2
    // lowers to ffts_cross_core_sync -- and the `pipe` template argument orders it behind
    // nothing (MICROROUTINE_DESIGN_GUIDELINES.md 6.2a). So the argument for removal rests
    // entirely on "the scalar stream is in order and CubeContext's drain is still in force",
    // and section 6.2c records a measured case in this operator family where an announcement
    // overtook its own un-drained pipeline by 1032 and 2179 cycles. That hazard is invisible to
    // every gate this change can run: set/wait counts balance, the id census passes, the
    // capacity-one model works on the schedule rather than the fence, and the output stays
    // bit-exact while a panel is read one generation early.
    //
    // scripts/audit_cross_core_fences.py cannot settle it either. With the drain removed it
    // accepts this announcement as a STRUCTURAL PRIMING SET -- nothing in the enclosing body
    // issues work on the announced pipe -- which is true of the text and false of the steady
    // loop, so the audit would pass for the wrong reason. The gate that would settle it is an
    // A/B scripts/analyze_announce_overtake.py comparison at one geometry and one announcement
    // ordinal, which needs a second full build of this operator; it is recorded as open in
    // reports/fused_decode_host_tiling_audit_20261106.md rather than assumed either way.
    template <uint32_t PHASE>
    __aicore__ inline void ReleaseCubeSlot()
    {
        constexpr uint32_t kSlot = PHASE % kSlots;
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::FIX_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(
            static_cast<uint16_t>(kOptFlagSlotFree + kSlot));
    }

    template <uint32_t PHASE>
    __aicore__ inline void ReadPipelineTile(PipelineTask &task)
    {
        constexpr uint32_t kSlot = PHASE % kSlots;
        const uint32_t absolute = task.tileOrigin + task.readTile;
        ++task.readTile;
        vector_.ReadTileToSlot(TilePhysical(task, absolute),
                               (absolute & task.pageMask) << fd::kDecodeTileRowsShift, task.kvHead, kSlot);
    }

    template <uint32_t PHASE>
    __aicore__ inline void VectorSoftmax(PipelineTask &task, const TurboQuantTaskHeads &heads)
    {
        constexpr uint32_t kSlot = PHASE % kSlots;
        constexpr uint32_t kWindow = PHASE % kWindows;
        // One compare and one select, no branch: every tile but the last has 64 live rows, and
        // which one is last is a host-computed index rather than a page-boundary walk.
        const uint32_t tile = task.consumeTile;
        ++task.consumeTile;
        const uint32_t valid = tile == task.lastTile ? task.lastValid : kCubeTileRows;
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagScoresReady + kWindow));
        vector_.ComputeSoftmaxAndStageProbs(mm_, valid, heads, kSlot, kWindow);
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::MTE3_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(static_cast<uint16_t>(kOptFlagProbsReady));
    }

    __aicore__ inline void VectorAccumulate(const TurboQuantTaskHeads &heads)
    {
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagContextReady));
        vector_.ComputeAccumulate(heads);
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunCubeSteadyPhase(const uint32_t rows)
    {
        CubeScores<PHASE + 1>(rows);
        CubeContext<PHASE>(rows);
        ReleaseCubeSlot<PHASE>();
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunVectorSteadyPhase(PipelineTask &task, const TurboQuantTaskHeads &heads)
    {
        constexpr uint32_t kStageSlot = (PHASE + 2) % kSlots;
        VectorSoftmax<PHASE>(task, heads);
        SyncVectorToMte2();
        ReadPipelineTile<PHASE>(task);
        vector_.ComputeAccumulatorDecay(heads);
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagSlotFree + kStageSlot));
        StageToL1(task, kStageSlot);
        VectorAccumulate(heads);
    }

    // The remainder dispatch selects the drain's phase statically as well.
    // It never resets phase to zero at N-3, including all five ragged residues.
    template <uint32_t PHASE>
    __aicore__ inline void RunCubeSteadyTail(const uint32_t rows, const uint32_t remaining)
    {
        if constexpr (PHASE < kPhases - 1) {
            if (PHASE < remaining) {
                RunCubeSteadyPhase<PHASE>(rows);
                RunCubeSteadyTail<PHASE + 1>(rows, remaining);
                return;
            }
        }
        RunCubeDrain<PHASE>(rows);
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunVectorSteadyTail(PipelineTask &task, const TurboQuantTaskHeads &heads,
                                              const uint32_t remaining)
    {
        if constexpr (PHASE < kPhases - 1) {
            if (PHASE < remaining) {
                RunVectorSteadyPhase<PHASE>(task, heads);
                RunVectorSteadyTail<PHASE + 1>(task, heads, remaining);
                return;
            }
        }
        RunVectorDrain<PHASE>(task, heads);
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunCubeDrain(const uint32_t rows)
    {
        CubeScores<PHASE + 1>(rows); CubeContext<PHASE>(rows);     // N-3: no refill credit
        CubeScores<PHASE + 2>(rows); CubeContext<PHASE + 1>(rows); // N-2
        CubeContext<PHASE + 2>(rows);                            // N-1: no scores
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunVectorDrain(PipelineTask &task, const TurboQuantTaskHeads &heads)
    {
        constexpr uint32_t kStageSlot = (PHASE + 2) % kSlots;
        VectorSoftmax<PHASE>(task, heads);                       // N-3: no read
        vector_.ComputeAccumulatorDecay(heads);
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagSlotFree + kStageSlot));
        StageToL1(task, kStageSlot);
        VectorAccumulate(heads);
        VectorSoftmax<PHASE + 1>(task, heads);                   // N-2: no stage
        vector_.ComputeAccumulatorDecay(heads);
        VectorAccumulate(heads);
        VectorSoftmax<PHASE + 2>(task, heads);                   // N-1: final softmax/context
        vector_.ComputeAccumulatorDecay(heads);
        VectorAccumulate(heads);
    }

    // The slot's MTE2 read is already posted on the slot's own event id; StageSlotToL1 waits exactly that
    // one, unpacks the plane in UB in NZ order and bursts it into L1. Both subcores set the slot's token,
    // because the AIC's single wait needs one from each of them.
    __aicore__ inline void StageToL1(PipelineTask &task, const uint32_t slot)
    {
        vector_.StageSlotToL1(mm_, task.kvHead, slot);
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::MTE3_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(static_cast<uint16_t>(kOptFlagSlotReady + slot));
    }

    // The prefetched block table of the task in flight, over exactly the host-declared live
    // span. blockCacheCount_ == 0 is an empty worker; pagesMapped_ is false when the span held a
    // negative entry, which the closed-form cursor cannot express.
    int32_t blockCache_[kBlockCacheEntries] = {};
    uint32_t blockCacheBase_ = 0;
    uint32_t blockCacheCount_ = 0;
    bool pagesMapped_ = true;
    // The host tiling: descriptors first, then one dispatch record per grid block. Read-only, and
    // read by BOTH roles -- the AIC needs the same words the AIV does, because num_tiles is the
    // predicate the two of them have to agree on.
    AscendC::GlobalTensor<uint32_t> tilingGm_;
    TaskCursor taskStart_;
    ReduceCursor reduceStart_[kVectorSubcoresPerBlock];
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

// ------------------------------------------------------------------------------------------------
// The reduction phase, in one place for both launch macros
// ------------------------------------------------------------------------------------------------
//
// Default: the shipping vector reducer, guarded by `if ASCEND_IS_AIV` because the Cube has no
// part in it -- which is the idleness the Cube GEMM form spends.
//
// ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1 selects TurboQuantBatchedGemmReducer instead: M = 16
// packed query rows against 16-channel slices of D, one dense Mmad per slice, the product split
// between the two vector subcores by a dual-destination Fixpipe. Three differences are visible
// right here and all three are load-bearing:
//
//   * NO `if ASCEND_IS_AIV`. The Cube does the contraction, and both roles have to walk the
//     same TPipe allocation because the Fixpipe's destination is an address in the vector
//     cores' UB. Guarding this is what made the first Cube run of the probe target an
//     unallocated tensor -- su_ccu_mpu_err_t0 on cubecore0 -- while still printing the right
//     numbers.
//   * op.ReduceGemm() rather than op.Reduce(): the batched form slices its own share, because
//     dnNum walks one uniform source stride and a fused token's task-id gap has to end a run.
//   * The reducer is constructed AFTER SyncAll<false>(), as before, but it is not free: see the
//     UB deficit below.
//
// THE SWITCH'S UB COMES FROM AN OVERLAY, NOT FROM HEADROOM. The reducer's panels do not fit
// what this operator has left: it claims 27264 UB bytes against the vector reducer's 3360, a
// net 23904, and only 4256 are free under the 192 KB window -- a 19648-byte deficit. So the
// reducer allocates from a TBufPool aliased onto the attention phase's transient pool, which
// SyncAll<false>() above has made dead. The alias costs zero bytes: InitBufPool's shared
// overload sets the new pool's startAddr_ to the shared one's and does not advance the hardware
// pool. include/turboquant_gemm_reduce.hpp carries the arithmetic and the static_assert that
// the reducer fits inside that pool.
#if defined(ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE) && ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE
#include "cube/turboquant_gemm_reduce_service.h"

// OVERLAY SAFETY: what is checked here, and the one thing no compiler can check.
//
// ReduceGemm's two panel slots live in a TBufPool aliased onto the attention phase's transient
// pool -- kUbSizes[15..17], which are nzOperandBuf_ (the tile ingest's NZ staging operand),
// accBuf_ (the per-task online-softmax accumulator) and queryWorkBuf_ (the query staging). The
// reduction therefore WRITES the same UB bytes the attention phase read. Four checks below,
// and all four are extent checks:
//
//   1. the pool's length is what the contract header says it is, so the two headers that
//      restate it cannot drift (this was the only check here before);
//   2. the reducer's UB claim fits inside the pool the OPERATOR allocates -- kTransientPoolBytes,
//      not the reducer's own restatement of it, so this holds even while the restatement is
//      being changed, and it is the use-site counterpart of the reducer header's own fit
//      assert. (The extent itself needs no assert here: kTransientPoolBytes is DEFINED as
//      kUbSizes[15] + kUbSizes[16] + kUbSizes[17] in the contract header and pinned to 32768
//      there, and check 1 ties the reducer's copy to it. A pool one allocation longer would
//      move kUbEnd and every offset after it, and the Cube's dual-destination Fixpipe windows
//      at UB offsets 0 and kScoreBytes are addressed by offset on three cores with nothing in
//      the toolchain checking that they agree.)
//   3. the launch validator rejects a split count the reduction's partial fractal cannot
//      hold. kMaxSplits is baked into the fractal geometry (one 8-float C0 per query row), so
//      a valid_launch that admitted splits > kMaxSplits would hand the reducer a source
//      stride it has no lane for. valid_launch is constexpr, so the bound is checked by
//      calling it rather than by restating it.
//
// NOT CHECKED, AND NOT CHECKABLE HERE: that those three buffers are DEAD when the reduction
// starts. Nothing in the toolchain verifies liveness. What establishes it is the
// AscendC::SyncAll<false>() in the launch macros -- every block's attention phase has retired
// before any block reduces -- plus the fact that no method reachable from ReduceGemm reads
// them. A future change that reads nzOperandBuf_, accBuf_ or queryWorkBuf_ after the attention
// phase breaks this SILENTLY, with no diagnostic and correct-looking output at geometries that
// do not reduce. The gate that would catch it is a multi-split CAModel rung with its canary
// and exact-comparison checks -- section 9 of
static_assert(ascend_micro::fused_decode_opt::kTransientPoolBytes ==
                  ascend_micro::gemm_reduce::kGemmSharedPoolBytes,
              "turboquant_gemm_reduce.hpp restates the transient pool's length; update "
              "kGemmSharedPoolBytes there to match kTransientPoolBytes");
static_assert(ascend_micro::gemm_reduce::kGemmReducerUbBytes <=
                  ascend_micro::fused_decode_opt::kTransientPoolBytes,
              "the batched reducer's panels do not fit the attention phase's transient pool");
static_assert(ascend_micro::fused_decode_opt::valid_launch(
                  1, 1, 16, 1, 256, 128, 8, ascend_micro::gemm_reduce::kMaxSplits, 16, 8, 16),
              "the launch validator refuses the reduction's own maximum split count");
static_assert(!ascend_micro::fused_decode_opt::valid_launch(
                  1, 1, 16, 1, 256, 128, 8, ascend_micro::gemm_reduce::kMaxSplits + 1, 16, 9, 16),
              "the launch validator admits more splits than the Cube GEMM reduction's partial "
              "fractal has lanes for");

#define ASCEND_TQ_FUSED_DECODE_REDUCE_PHASE(TYPE)                                                                    \
    do {                                                                                                             \
        ascend_micro::gemm_reduce::TurboQuantBatchedGemmReducer<TYPE> reducer(&pipe);                                 \
        reducer.Init(workspace, output, numHeads, headSize, numSplits, op.ReduceGemmSharedPool());                    \
        op.ReduceGemm(reducer, reduceTasksPerBlock);                                                                  \
    } while (0)
#else
// THE DEFAULT PHASE'S ROLE CONTRACT, and why its asymmetry is not a hang.
//
// The reducer object and the Reduce() call stay behind `if ASCEND_IS_AIV`: the Cube has no part
// in the vector recurrence, and -- unlike the Cube GEMM form above -- the reducer's destination
// is the vector cores' own UB, so the two roles do not have to walk the same TPipe allocation.
// The AIC allocates nothing and does nothing in this phase.
//
// That asymmetry was audited for a cross-core join deadlock and does not have one:
//
//   * Both roles reach the launch macro's DrainFusedDecodeTask() after this phase, and that
//     drain is a balanced two-phase rendezvous on ONE flag id. CrossCoreSetFlag(ISASI) mode 2
//     (CANN 9.2, Ascend 950PR): the AIC's set releases the CrossCoreWaitFlag on BOTH AIVs, and
//     the two AIVs' sets TOGETHER release the AIC's wait. So one AIC set and one set from each
//     subcore is exactly the balance DrainFusedDecodeTask() posts, in both directions.
//   * The AIC therefore announcing its join while the AIVs are still reducing is the intended
//     order, not a race: the AIC then blocks in its own wait until both AIVs have finished
//     reducing AND drained MTE3, which the AIV leg's SyncEvent<MTE3_S>() in front of its set is
//     what guarantees.
//   * Nothing on TurboQuantPartialReducer's path takes a cross-core flag or a SyncAll of its
//     own -- its only barrier is a PipeBarrier inside its own Init -- so the AIC's absence
//     cannot block the AIVs either.
//
// What a future change must preserve is that last bullet. A cross-core primitive introduced
// anywhere under op.Reduce() turns this guard into a hang, and the gate that would show it is
// a multi-split CAModel rung and its trace audit, not a build. The join itself stays in the
// launch macros, once, after the phase, so both forms of this macro are followed by the same
// rendezvous and neither has to post one of its own.
#define ASCEND_TQ_FUSED_DECODE_REDUCE_PHASE(TYPE)                                                                    \
    do {                                                                                                             \
        if ASCEND_IS_AIV {                                                                                           \
            TurboQuantPartialReducer<TYPE> reducer(&pipe);                                                           \
            reducer.Init(workspace, output, numHeads, headSize, numSplits);                                          \
            op.Reduce(reducer, reduceTasksPerBlock);                                                                 \
        }                                                                                                            \
    } while (0)
#endif

#define ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT(NAME, MODE, TYPE)                                                      \
    extern "C" __global__ __aicore__ void NAME(                                                                      \
        GM_ADDR queryRot, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR blockTables,             \
        GM_ADDR contextLens, GM_ADDR modeTables, GM_ADDR workspace, GM_ADDR output, GM_ADDR tiling,                  \
        uint32_t numTokens,                                                                                          \
        uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,      \
        uint32_t numSplits, uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,              \
        uint32_t fusedContextLimit, float scale, float invSqrtLen)                                                   \
    {                                                                                                                \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); \
        AscendC::InitSocState(); \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantFusedDecodeOpt<MODE, TYPE> op(&pipe);                                                               \
        op.Init(queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,  \
                tiling, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,             \
                headsPerTask, fusedContextLimit, scale, invSqrtLen);                                                  \
        op.Process(tasksPerBlock);                                                                                    \
        if (op.NeedsReduction()) {                                                                                    \
            AscendC::SyncAll<false>();                                                                                \
            ASCEND_TQ_FUSED_DECODE_REDUCE_PHASE(TYPE);                                                                 \
        }                                                                                                             \
        DrainFusedDecodeTask(); \
    }

// The raw-query entry, argument for argument the same as the lock-step decode's, so a benchmark or an
// msprof run can swap one launcher for the other without touching its tiling.
#define ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT_RAW_QUERY(NAME, MODE, TYPE)                                            \
    extern "C" __global__ __aicore__ void NAME(                                                                      \
        GM_ADDR query, GM_ADDR piSigns, GM_ADDR rotTables, GM_ADDR h16, GM_ADDR gate, GM_ADDR queryRot,              \
        GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR blockTables, GM_ADDR contextLens,          \
        GM_ADDR modeTables, GM_ADDR workspace, GM_ADDR output, GM_ADDR tiling,                                       \
        uint32_t numTokens, uint32_t numHeads,                                                                       \
        uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,     \
        uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,                                 \
        uint32_t prologueVectorsPerBlock, uint32_t prologueCubeChunkVectors, uint32_t outputStage,                   \
        uint32_t fusedContextLimit, float scale, float invSqrtLen)                                                   \
    {                                                                                                                \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); \
        AscendC::InitSocState(); \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantFusedDecodeOpt<MODE, TYPE, false> op(&pipe);                                                        \
        op.Init(queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,  \
                tiling, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,             \
                headsPerTask, fusedContextLimit, scale, invSqrtLen);                                                  \
        op.InitBasis(query, piSigns, rotTables, h16, gate, queryRot, prologueCubeChunkVectors, outputStage,           \
                     invSqrtLen);                                                                                     \
        op.RotateQuery(prologueVectorsPerBlock);                                                                       \
        op.Process(tasksPerBlock);                                                                                     \
        if (op.NeedsReduction()) {                                                                                     \
            AscendC::SyncAll<false>();                                                                                  \
            ASCEND_TQ_FUSED_DECODE_REDUCE_PHASE(TYPE);                                                                  \
        }                                                                                                              \
        DrainFusedDecodeTask(); \
    }

// kv4fp8 only: the NZ-tiled ring ingest the schedule is built on, and the one mode the fused decode is
// validated in at all (TURBOQUANT_TESTS.md 13.25).
//
// TWO ELEMENT TYPES, ONE MODE. The template's `scalar_t` is the QUERY and OUTPUT element type, not the
// cache format: the cache stays KV4/FP8 in both instances, so MODE, the codec, the ring ingest, the
// Cube's score and context operands and every tile descriptor are the same objects in both. What
// scalar_t reaches is (a) the output GM tensor and the one UB row that is cast into it before the burst
// (WriteHeads, vector/turboquant_vector_service.h), (b) the same pair in both Split-K reducers, and (c)
// the raw query and gate the basis service reads, plus its Cube chunk operand
// (cube/turboquant_query_basis.h, which documents why that operand follows scalar_t rather than staying
// fp16). The intervening arithmetic -- scores, the online softmax recurrence, the probabilities, the
// accumulator, the Split-K partials and their merge -- is fp32 in both, so the bf16 instance is the fp16
// instance with a different final rounding and a different input rounding, not a different schedule.
// That is also why the UB map does not move: both types are 16-bit
// (../include/turboquant_fused_decode_opt.hpp pins kUbEnd, kBasisUbBytes and kReducerUbBytes, and the
// sizeof(scalar_t) == 2 static_asserts on both services hold it).
ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT(turboquant_mm_fused_decode_opt_kv4fp8_half, TurboQuantMode::KV4_FP8, half)
ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT_RAW_QUERY(turboquant_mm_fused_decode_opt_raw_query_kv4fp8_half,
                                                TurboQuantMode::KV4_FP8, half)
ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT(turboquant_mm_fused_decode_opt_kv4fp8_bf16, TurboQuantMode::KV4_FP8,
                                      bfloat16_t)
ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT_RAW_QUERY(turboquant_mm_fused_decode_opt_raw_query_kv4fp8_bf16,
                                                TurboQuantMode::KV4_FP8, bfloat16_t)

#elif defined(ASCEND_MICRO_CANN_LAUNCH)
#include "kernel_operator.h"
#include <stdexcept>
extern "C" __global__ __aicore__ void turboquant_mm_fused_decode_opt_kv4fp8_half(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, float, float);
extern "C" __global__ __aicore__ void turboquant_mm_fused_decode_opt_raw_query_kv4fp8_half(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float);
// The bf16 instances. Same signature, same tiling, same cache format: only the query/output
// element type differs, so a caller swaps one symbol for the other and changes nothing else.
extern "C" __global__ __aicore__ void turboquant_mm_fused_decode_opt_kv4fp8_bf16(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, float, float);
extern "C" __global__ __aicore__ void turboquant_mm_fused_decode_opt_raw_query_kv4fp8_bf16(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float);

namespace vllm_ascend {

void turboquant_mm_fused_decode_opt_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim,
                                         void *queryRot, void *keyCache, void *valueCache, void *scaleCache,
                                         void *blockTables, void *contextLens, void *modeTables, void *workspace,
                                         void *output, void *tiling, uint32_t numTokens, uint32_t numHeads,
                                         uint32_t numKvHeads,
                                         uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                         uint32_t numSplits, uint32_t headsPerTask, uint32_t tasksPerBlock,
                                         uint32_t reduceTasksPerBlock, uint32_t fusedContextLimit, float scale,
                                         float invSqrtLen)
{
    // No device fallback: a launch with no host tiling descriptors is the operator this change
    // removed, and silently deriving the tiling again on the SPU is the one outcome that would
    // look correct in a test and cost the Cube its prefetch on hardware.
    if (!tiling) throw std::invalid_argument("fused decode launch without host tiling descriptors");
    if (!ascend_micro::fused_decode_opt::valid_launch(blockDim, numTokens, numHeads,
            numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
            headsPerTask, tasksPerBlock, reduceTasksPerBlock))
        throw std::invalid_argument("unsupported fused decode geometry or incomplete task coverage");
    if (blockDim == 0 ||
        static_cast<turboquant::TurboQuantMode>(mode) != turboquant::TurboQuantMode::KV4_FP8) {
        return;
    }
    // Both element types reach the device through this one dispatch. The tiling, the geometry
    // guards above and the cache format are shared; only the kernel symbol differs.
    if (type == AscendType::FP16) {
        turboquant_mm_fused_decode_opt_kv4fp8_half<<<blockDim, nullptr, stream>>>(
            queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,
            tiling, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,
            tasksPerBlock, reduceTasksPerBlock, fusedContextLimit, scale, invSqrtLen);
    } else if (type == AscendType::BF16) {
        turboquant_mm_fused_decode_opt_kv4fp8_bf16<<<blockDim, nullptr, stream>>>(
            queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, modeTables, workspace, output,
            tiling, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, headsPerTask,
            tasksPerBlock, reduceTasksPerBlock, fusedContextLimit, scale, invSqrtLen);
    }
}

void turboquant_mm_fused_decode_opt_raw_query_impl(
    int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *query, void *piSigns, void *rotTables,
    void *h16, void *gate, void *queryRot, void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
    void *contextLens, void *modeTables, void *workspace, void *output, void *tiling, uint32_t numTokens,
    uint32_t numHeads,
    uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,
    uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock, uint32_t prologueVectorsPerBlock,
    uint32_t prologueCubeChunkVectors, uint32_t outputStage, uint32_t fusedContextLimit, float scale,
    float invSqrtLen)
{
    if (!tiling) throw std::invalid_argument("fused decode launch without host tiling descriptors");
    if (uint64_t{blockDim} * prologueVectorsPerBlock < uint64_t{numTokens} * numHeads ||
        (prologueCubeChunkVectors != 0 && prologueCubeChunkVectors != 16) || outputStage > 2)
        throw std::invalid_argument("unsupported raw query basis tiling");
    if (!ascend_micro::fused_decode_opt::valid_launch(blockDim, numTokens, numHeads,
            numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
            headsPerTask, tasksPerBlock, reduceTasksPerBlock))
        throw std::invalid_argument("unsupported fused decode geometry or incomplete task coverage");
    if (blockDim == 0 ||
        static_cast<turboquant::TurboQuantMode>(mode) != turboquant::TurboQuantMode::KV4_FP8) {
        return;
    }
    // `query` and `gate` are scalar_t and `queryRot` is the fp32 image the prologue writes, in both
    // instances: the basis service's Cube chunk operand follows scalar_t but the H16 table in GM stays
    // fp16 for both, so the raw-query ABI is element-type agnostic apart from the query and gate buffers.
    if (type == AscendType::FP16) {
        turboquant_mm_fused_decode_opt_raw_query_kv4fp8_half<<<blockDim, nullptr, stream>>>(
            query, piSigns, rotTables, h16, gate, queryRot, keyCache, valueCache, scaleCache, blockTables,
            contextLens, modeTables, workspace, output, tiling, numTokens, numHeads, numKvHeads, headSize,
            blockSize, maxBlocksPerSeq, numSplits, headsPerTask, tasksPerBlock, reduceTasksPerBlock,
            prologueVectorsPerBlock, prologueCubeChunkVectors, outputStage, fusedContextLimit, scale, invSqrtLen);
    } else if (type == AscendType::BF16) {
        turboquant_mm_fused_decode_opt_raw_query_kv4fp8_bf16<<<blockDim, nullptr, stream>>>(
            query, piSigns, rotTables, h16, gate, queryRot, keyCache, valueCache, scaleCache, blockTables,
            contextLens, modeTables, workspace, output, tiling, numTokens, numHeads, numKvHeads, headSize,
            blockSize, maxBlocksPerSeq, numSplits, headsPerTask, tasksPerBlock, reduceTasksPerBlock,
            prologueVectorsPerBlock, prologueCubeChunkVectors, outputStage, fusedContextLimit, scale, invSqrtLen);
    }
}

}

// The raw-query entry, through the same convenience ABI. It exists because the only launcher
// this harness had called the PRE-ROTATED kernel, so nothing could reach the query prologue or
// the output stage at all -- neither a test nor a benchmark nor an msprof run.
extern "C" void launch_fused_decode_opt_raw_query(void* stream,
    const ascend_micro::fused_decode_opt::Launch* c,
    const ascend_micro::fused_decode_opt::RawQueryBuffers* b)
{
    if (!c || !b) throw std::invalid_argument("null fused decode raw-query launch descriptor");
    vllm_ascend::turboquant_mm_fused_decode_opt_raw_query_impl(4,
        ascend_micro::fused_decode_opt::ascend_type_of(c->element_type),
        stream, c->grid, b->raw_query, b->pi_signs, b->rot_tables, b->h16, b->gate,
        b->decode.query, b->decode.key, b->decode.value, b->decode.scales, b->decode.block_table,
        b->decode.context_lengths, b->decode.mode_tables, b->decode.workspace, b->decode.output,
        b->decode.tiling,
        c->tokens, c->heads, c->kv_heads, c->head_size, c->block_size, c->max_blocks, c->splits,
        c->heads_per_task, c->tasks_per_block, c->reduce_tasks_per_block,
        c->prologue_vectors_per_block, c->prologue_cube_chunk_vectors, c->output_stage,
        c->fused_context_limit, c->scale, c->inv_sqrt_len);
}

extern "C" void launch_fused_decode_opt(void* stream,
    const ascend_micro::fused_decode_opt::Launch* c,
    const ascend_micro::fused_decode_opt::Buffers* b)
{
    if (!c || !b) throw std::invalid_argument("null fused decode launch descriptor");
    vllm_ascend::turboquant_mm_fused_decode_opt_impl(4,
        ascend_micro::fused_decode_opt::ascend_type_of(c->element_type),
        stream, c->grid, b->query, b->key, b->value, b->scales, b->block_table,
        b->context_lengths, b->mode_tables, b->workspace, b->output, b->tiling, c->tokens,
        c->heads, c->kv_heads, c->head_size, c->block_size, c->max_blocks, c->splits,
        c->heads_per_task, c->tasks_per_block, c->reduce_tasks_per_block,
        c->fused_context_limit, c->scale, c->inv_sqrt_len);
}

#endif
