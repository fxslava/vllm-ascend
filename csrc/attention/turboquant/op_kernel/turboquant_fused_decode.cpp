#include "../include/turboquant_fused_decode_opt.hpp"
#include "common/types.h"

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"

#include "common/turboquant_codec_mx.h"
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

__aicore__ inline void DrainFusedDecodeTask()
{
    constexpr uint16_t flag = ascend_micro::fused_decode_opt::kTaskJoinFlag;
    if ASCEND_IS_AIC {
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::M_S>();
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::FIX_S>();
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(flag);
        AscendC::CrossCoreWaitFlag(flag);
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

constexpr uint32_t kPipelinePhases =
    kCubePipelineSlots * kCubeScoreWindows / PipelineGcd(kCubePipelineSlots, kCubeScoreWindows);

template <TurboQuantMode MODE, typename scalar_t, bool PRE_ROTATED = true,
          DecodeUnpack UNPACK = DecodeUnpack::kNative>
class TurboQuantFusedDecodeOpt {
public:
    static constexpr uint32_t kSlots = kCubePipelineSlots;
    static constexpr uint32_t kWindows = kCubeScoreWindows;
    static constexpr uint32_t kPhases = kPipelinePhases;

    using Mm = TurboQuantCubeMm<MODE, kSlots>;
    using Codec = TurboQuantModeCodec<MODE, DecodeUnpackGathers(UNPACK), DecodeUnpackIsLloydMax(UNPACK)>;
    using Vector = TurboQuantVectorDecodeService<MODE, scalar_t, Mm, Codec, UNPACK, kSlots, kWindows, kBankSkewBytes>;
    using Reducer = TurboQuantPartialReducer<scalar_t>;
    using Basis = TurboQuantQueryBasis<scalar_t, !PRE_ROTATED>;

    static_assert(Codec::WorkBufferWords(256, 32) * sizeof(float) == ascend_micro::fused_decode_opt::kUbSizes[27]);
    static_assert(Vector::kBatched);
    static_assert(Vector::kIngestSlots == kSlots);
    static_assert(Vector::kScoreWindows == kWindows);
    static_assert(Mm::kSlots == kSlots);
    static_assert(kOptFlagProbsReady < kOptFlagReservedFirst);
    static_assert(kSlots == 3 && kWindows == 2 && kPhases == 6);
    static_assert(kCubeTileRows == fd::kDecodeTileRows && kCubeTileRows == (1u << fd::kDecodeTileRowsShift));
    static_assert(kWindows >= 2);
    static_assert(kSlots >= 3);

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

        groupHeads_ = tilingGm_.GetValue(fd::kTilingWordGroupHeads);
        chunksPerGroup_ = tilingGm_.GetValue(fd::kTilingWordChunksPerGroup);
        headsPerTask_ = headsPerTask < 1 ? 1 : headsPerTask;
        if (headsPerTask_ > groupHeads_) {
            headsPerTask_ = groupHeads_;
        }
        if (headsPerTask_ > kCubeTileM) {
            headsPerTask_ = kCubeTileM;
        }

        const uint32_t record = numTokens * numSplits_ * fd::kDecodeTilingWords + MixBlockIdx() * fd::kDecodeDispatchWords;
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
    }

    __aicore__ inline void InitBasis(__gm__ void *query, __gm__ void *piSigns, __gm__ void *rotTables,
                                     __gm__ void *h16, __gm__ void *gate, __gm__ void *queryRot,
                                     const uint32_t cubeChunkVectors, const uint32_t outputStage,
                                     const float invSqrtLen)
    {
        static_assert(Basis::kEnabled);
        basis_.Init(pipe_, query, piSigns, rotTables, h16, gate, queryRot, numHeads_, headSize_, cubeChunkVectors,
                    outputStage, invSqrtLen);
    }

    __aicore__ inline void RotateQuery(const uint32_t vectorsPerBlock)
    {
        static_assert(Basis::kEnabled);
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
        TaskCursor cursor = taskStart_;
        for (uint32_t task = start; task < end; ++task) {
            ComputeTask(cursor.token, cursor.kvHead, cursor.split, cursor.chunk);
            AdvanceTask(cursor);
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
    struct GemmCombineSelect {
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

    __aicore__ inline auto &ReduceGemmSharedPool() { return vector_.TransientPool(); }
#endif

private:
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

    struct PipelineTask {
        uint32_t readTile = 0;
        uint32_t consumeTile = 0;
        uint32_t token = 0;
        uint32_t kvHead = 0;
        uint32_t numTiles = 0;
        uint32_t lastTile = 0;
        uint32_t lastValid = 0;
        uint32_t tileOrigin = 0;
        uint32_t pageShift = 0;
        uint32_t pageMask = 0;
        uint32_t tile = 0;
    };

    __aicore__ inline bool IsFused(const int32_t contextLen) const
    {
        return numSplits_ <= 1 || contextLen <= static_cast<int32_t>(fusedContextLimit_);
    }

    __aicore__ inline void ComputeTask(const uint32_t token, const uint32_t kvHead, const uint32_t split,
                                       const uint32_t chunk)
    {
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
            SyncVectorToMte2();
            vector_.BeginTask(heads);
        }

        const uint32_t descriptor = (token * numSplits_ + split) * fd::kDecodeTilingWords;
        const uint32_t numTiles = tilingGm_.GetValue(descriptor + fd::kTilingWordNumTiles);
        const uint32_t tailRows = tilingGm_.GetValue(descriptor + fd::kTilingWordTailRows);
        const uint32_t blockStart = tilingGm_.GetValue(descriptor + fd::kTilingWordBlockStart);
        const uint32_t blocksInSplit = tilingGm_.GetValue(descriptor + fd::kTilingWordBlocksInSplit);
        const uint32_t tileOrigin = tilingGm_.GetValue(descriptor + fd::kTilingWordTileOrigin);
        const uint32_t pageShift = tilingGm_.GetValue(descriptor + fd::kTilingWordTilesPerPageShift);

        PrefetchBlockTable(token, blockStart, blocksInSplit);

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

    __aicore__ inline uint32_t TilePhysical(const PipelineTask &task, const uint32_t absolute)
    {
        const uint32_t page = absolute >> task.pageShift;
        if (page < blockCacheCount_) {
            return static_cast<uint32_t>(blockCache_[page]);
        }
        return static_cast<uint32_t>(blockTableGm_.GetValue(
            static_cast<uint64_t>(task.token) * maxBlocksPerSeq_ + blockCacheBase_ + page));
    }

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

    __aicore__ inline void RunCubePipeline(PipelineTask &task, const uint32_t rows)
    {
        if (task.numTiles < kSlots + 1) {
            RunCubeShortPipeline(task, rows, task.numTiles);
            return;
        }
        ReleaseCubeSlot<2>();
        CubeScores<0>(rows);
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
        ReadPipelineTile<0>(task);
        ReadPipelineTile<1>(task); StageToL1(task, 0);
        ReadPipelineTile<2>(task); StageToL1(task, 1);
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
        CubeScores<PHASE + 1>(rows); CubeContext<PHASE>(rows);
        CubeScores<PHASE + 2>(rows); CubeContext<PHASE + 1>(rows);
        CubeContext<PHASE + 2>(rows);
    }

    template <uint32_t PHASE>
    __aicore__ inline void RunVectorDrain(PipelineTask &task, const TurboQuantTaskHeads &heads)
    {
        constexpr uint32_t kStageSlot = (PHASE + 2) % kSlots;
        VectorSoftmax<PHASE>(task, heads);
        vector_.ComputeAccumulatorDecay(heads);
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kOptFlagSlotFree + kStageSlot));
        StageToL1(task, kStageSlot);
        VectorAccumulate(heads);
        VectorSoftmax<PHASE + 1>(task, heads);
        vector_.ComputeAccumulatorDecay(heads);
        VectorAccumulate(heads);
        VectorSoftmax<PHASE + 2>(task, heads);
        vector_.ComputeAccumulatorDecay(heads);
        VectorAccumulate(heads);
    }

    __aicore__ inline void StageToL1(PipelineTask &task, const uint32_t slot)
    {
        vector_.StageSlotToL1(mm_, task.kvHead, slot);
        vllm_ascend::turboquant::SyncEvent<AscendC::HardEvent::MTE3_S>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(static_cast<uint16_t>(kOptFlagSlotReady + slot));
    }

    int32_t blockCache_[kBlockCacheEntries] = {};
    uint32_t blockCacheBase_ = 0;
    uint32_t blockCacheCount_ = 0;
    bool pagesMapped_ = true;
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
    Basis basis_;
};

}

#if defined(ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE) && ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE
#include "cube/turboquant_gemm_reduce_service.h"

static_assert(ascend_micro::fused_decode_opt::kTransientPoolBytes == ascend_micro::gemm_reduce::kGemmSharedPoolBytes);
static_assert(ascend_micro::gemm_reduce::kGemmReducerUbBytes <= ascend_micro::fused_decode_opt::kTransientPoolBytes);
static_assert(ascend_micro::fused_decode_opt::valid_launch(1, 1, 16, 1, 256, 128, 8, ascend_micro::gemm_reduce::kMaxSplits, 16, 8, 16));
static_assert(!ascend_micro::fused_decode_opt::valid_launch(1, 1, 16, 1, 256, 128, 8, ascend_micro::gemm_reduce::kMaxSplits + 1, 16, 9, 16));

#define ASCEND_TQ_FUSED_DECODE_REDUCE_PHASE(TYPE)                                                                    \
    do {                                                                                                             \
        ascend_micro::gemm_reduce::TurboQuantBatchedGemmReducer<TYPE> reducer(&pipe);                                 \
        reducer.Init(workspace, output, numHeads, headSize, numSplits, op.ReduceGemmSharedPool());                    \
        op.ReduceGemm(reducer, reduceTasksPerBlock);                                                                  \
    } while (0)
#else
#define ASCEND_TQ_FUSED_DECODE_REDUCE_PHASE(TYPE)                                                                    \
    do {                                                                                                             \
        if ASCEND_IS_AIV {                                                                                           \
            TurboQuantPartialReducer<TYPE> reducer(&pipe);                                                           \
            reducer.Init(workspace, output, numHeads, headSize, numSplits);                                          \
            op.Reduce(reducer, reduceTasksPerBlock);                                                                 \
        }                                                                                                            \
    } while (0)
#endif

#endif  // __CCE_AICORE__ -- everything above is the device-side body

// ------------------------------------------------------------- entry points
// THE MIRROR'S ONE STANDING EXCEPTION. The two `__global__` entry points below
// are compiled in EVERY pass of this translation unit, host and device alike;
// only their *bodies* stay behind __CCE_AICORE__, as empty BODY macros in the
// host pass. turboquant-core keeps the definitions inside the device arm and
// bare prototypes in the host arm, which is correct for ITS build (the legacy
// `ascendc_library`, which generates host_stub.cpp) and wrong for this one.
// Do not re-mirror it away -- see the longer note in
// turboquant_reshape_batch.cpp. Measured on the first build of this port:
// `nm -D` showed both kernels U beside the T launchers that call them.
//
// Guarded for BOTH passes of the kernel build and NEITHER pass of a plain host
// compiler. ASCEND_MICRO_CANN_LAUNCH is set by the root CMakeLists on every pass
// of this translation unit (once in ascendc_compile_definitions, once behind
// -forward-options-to-host-compiler), so the ASC host pass reaches this
// definition and emits the launch stub, while the device pass reaches it through
// __CCE_AICORE__. A device-free gate that `#include`s this file with g++ --
// tests/ut/ops/turboquant/test_contract_gates.cpp does, for the arena
// static_asserts -- defines neither, and so never sees `__global__`,
// `__aicore__`, `__gm__` or `bfloat16_t`, none of which a plain compiler has.
#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_CANN_LAUNCH)
// GM_ADDR is part of the published signature, so kernel_operator.h has to be
// reachable in the host pass too; the include is a no-op in the device passes,
// which already have it.
#include "kernel_operator.h"

#if defined(__CCE_AICORE__)

// The schedules, device-only. In the host pass these expand to nothing, so the entry
// points below are the same function declarations with an empty body -- which is all
// the host pass needs to emit their launch stubs.
#define ASCEND_TQ_FUSED_DECODE_OPT_BODY(MODE, TYPE) \
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
        DrainFusedDecodeTask();

#define ASCEND_TQ_FUSED_DECODE_OPT_RAW_QUERY_BODY(MODE, TYPE) \
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
        DrainFusedDecodeTask();

#else   // the host pass: the launch stub needs the signature, not the schedule

#define ASCEND_TQ_FUSED_DECODE_OPT_BODY(MODE, TYPE)
#define ASCEND_TQ_FUSED_DECODE_OPT_RAW_QUERY_BODY(MODE, TYPE)

#endif  // __CCE_AICORE__

#define ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT(NAME, MODE, TYPE)                                                      \
    extern "C" __global__ __aicore__ void NAME(                                                                      \
        GM_ADDR queryRot, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR blockTables,             \
        GM_ADDR contextLens, GM_ADDR modeTables, GM_ADDR workspace, GM_ADDR output, GM_ADDR tiling,                  \
        uint32_t numTokens,                                                                                          \
        uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,      \
        uint32_t numSplits, uint32_t headsPerTask, uint32_t tasksPerBlock, uint32_t reduceTasksPerBlock,              \
        uint32_t fusedContextLimit, float scale, float invSqrtLen)                                                   \
    {                                                                                                                \
        ASCEND_TQ_FUSED_DECODE_OPT_BODY(MODE, TYPE)                                                                  \
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
        ASCEND_TQ_FUSED_DECODE_OPT_RAW_QUERY_BODY(MODE, TYPE)                                                        \
    }

ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT(turboquant_mm_fused_decode_opt_kv4fp8_bf16, TurboQuantMode::KV4_FP8, bfloat16_t)
ASCEND_TQ_DECLARE_MM_FUSED_DECODE_OPT_RAW_QUERY(turboquant_mm_fused_decode_opt_raw_query_kv4fp8_bf16, TurboQuantMode::KV4_FP8, bfloat16_t)

#endif  // the entry points

#if defined(ASCEND_MICRO_CANN_LAUNCH) && !defined(__CCE_AICORE__)
#include <stdexcept>

// No prototypes for the two kernels here: their definitions above are visible
// in this pass, which is what makes the host launch stubs exist at all, and the
// argument blocks are written once -- in the two entry-point macros -- rather
// than twice, so the two spellings cannot drift apart.

namespace vllm_ascend {

__attribute__((unused)) static inline uint8_t *AsGmAddr(void *pointer)
{
    return static_cast<uint8_t *>(pointer);
}

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
    if (!tiling) throw std::invalid_argument("fused decode launch without host tiling descriptors");
    if (!ascend_micro::fused_decode_opt::valid_launch(blockDim, numTokens, numHeads,
            numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
            headsPerTask, tasksPerBlock, reduceTasksPerBlock))
        throw std::invalid_argument("unsupported fused decode geometry or incomplete task coverage");
    if (blockDim == 0 || static_cast<turboquant::TurboQuantMode>(mode) != turboquant::TurboQuantMode::KV4_FP8) {
        return;
    }
    
    if (type == AscendType::BF16) {
        turboquant_mm_fused_decode_opt_kv4fp8_bf16<<<blockDim, nullptr, stream>>>(
            AsGmAddr(queryRot), AsGmAddr(keyCache), AsGmAddr(valueCache), AsGmAddr(scaleCache),
            AsGmAddr(blockTables), AsGmAddr(contextLens), AsGmAddr(modeTables), AsGmAddr(workspace),
            AsGmAddr(output), AsGmAddr(tiling), numTokens, numHeads, numKvHeads, headSize, blockSize,
            maxBlocksPerSeq, numSplits, headsPerTask, tasksPerBlock, reduceTasksPerBlock,
            fusedContextLimit, scale, invSqrtLen);
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
    if (blockDim == 0 || static_cast<turboquant::TurboQuantMode>(mode) != turboquant::TurboQuantMode::KV4_FP8) {
        return;
    }
    
    if (type == AscendType::BF16) {
        turboquant_mm_fused_decode_opt_raw_query_kv4fp8_bf16<<<blockDim, nullptr, stream>>>(
            AsGmAddr(query), AsGmAddr(piSigns), AsGmAddr(rotTables), AsGmAddr(h16), AsGmAddr(gate),
            AsGmAddr(queryRot), AsGmAddr(keyCache), AsGmAddr(valueCache), AsGmAddr(scaleCache),
            AsGmAddr(blockTables), AsGmAddr(contextLens), AsGmAddr(modeTables), AsGmAddr(workspace),
            AsGmAddr(output), AsGmAddr(tiling), numTokens, numHeads, numKvHeads, headSize,
            blockSize, maxBlocksPerSeq, numSplits, headsPerTask, tasksPerBlock, reduceTasksPerBlock,
            prologueVectorsPerBlock, prologueCubeChunkVectors, outputStage, fusedContextLimit, scale,
            invSqrtLen);
    }
}

}

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