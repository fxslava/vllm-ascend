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

// Vector-unit services of the TurboQuant decodes.
//
//   TurboQuantTileBurst            DMA: one tile of a row-major packed cache in wide GM bursts (>= 128 B)
//   WriteNormalizedHeads           online softmax close-out: 1 / (sum + eps), cast, direct GM write
//   TurboQuantPartialReducer       the in-launch reduction of per-split partials (context > 4096)
//   TurboQuantVectorDecodeService  the AIV half of the Cube decode: query and KV staging into L1,
//                                  the online softmax over the Cube's score rows, the accumulation
//
// The first three are shared by the AIV-only and the Cube decodes. Nothing here names a Cube type,
// so the AIV-only decode (which the wheel also builds for arch32) can include this header; the Cube
// decode service takes its Cube and codec types as template parameters.

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_VECTOR_SERVICE_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_VECTOR_SERVICE_H

#include "../common/turboquant_codec_950.h"
#include "../common/turboquant_common.h"
#include "../common/turboquant_mode.h"

namespace vllm_ascend {
namespace turboquant {

constexpr float kVectorNegInf = -3.4028235e38f;

// ----------------------------------------------------------------------------------------------------
// DMA tile handling
// ----------------------------------------------------------------------------------------------------

// Reads one kv-head plane of a row-major tile in wide contiguous GM bursts and never below kMinBurstBytes
// per descriptor. Three layouts:
//   one kv head          one contiguous tileRows * packedBytes read
//   packedBytes >= 128   one head's packedBytes per row, strided over the other heads
//   packedBytes <  128   the whole row plane into a UB bank, then a UB -> UB strided block move
// ReadTileRows() is the MTE2 half and SelectKvHeadRows() the vector half of the last layout; the caller
// owns the MTE2 -> V event between them.
class TurboQuantTileBurst {
public:
    __aicore__ inline void Init(AscendC::TPipe *pipe, const uint32_t numKvHeads, const uint32_t packedBytes,
                                const uint32_t tileRows, const uint32_t planes)
    {
        numKvHeads_ = numKvHeads;
        packedBytes_ = packedBytes;
        tileRows_ = tileRows;
        planeBytes_ = tileRows * numKvHeads * packedBytes;
        readsWholeRows_ = (numKvHeads > 1 && packedBytes < kMinBurstBytes) ? 1u : 0u;
        kvHeadRowParams_ =
            AscendC::DataCopyParams{static_cast<uint16_t>(tileRows), static_cast<uint16_t>(packedBytes / kOperandC0),
                                    static_cast<uint16_t>((numKvHeads - 1) * packedBytes / kOperandC0), 0};
        if (readsWholeRows_ != 0) {
            pipe->InitBuffer(rowBankBuf_, planes * planeBytes_ * sizeof(int8_t));
        }
    }

    __aicore__ inline bool ReadsWholeRows() const { return readsWholeRows_ != 0; }

    __aicore__ inline void ReadTileRows(const AscendC::LocalTensor<int8_t> &dst, AscendC::GlobalTensor<int8_t> &cacheGm,
                                        const uint64_t rowOffset, const uint32_t kvHead, const uint32_t plane)
    {
        if (numKvHeads_ == 1) {
            AscendC::DataCopy(dst, cacheGm[rowOffset], tileRows_ * packedBytes_);
        } else if (readsWholeRows_ == 0) {
            AscendC::DataCopy(dst, cacheGm[rowOffset + static_cast<uint64_t>(kvHead) * packedBytes_],
                              kvHeadRowParams_);
        } else {
            AscendC::DataCopy(rowBankBuf_.Get<int8_t>()[plane * planeBytes_], cacheGm[rowOffset], planeBytes_);
        }
    }

    __aicore__ inline void SelectKvHeadRows(const AscendC::LocalTensor<int8_t> &dst, const uint32_t kvHead,
                                            const uint32_t plane)
    {
        if (readsWholeRows_ != 0) {
            AscendC::DataCopy(dst, rowBankBuf_.Get<int8_t>()[plane * planeBytes_ + kvHead * packedBytes_],
                              kvHeadRowParams_);
        }
    }

private:
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rowBankBuf_;
    AscendC::DataCopyParams kvHeadRowParams_;
    uint32_t numKvHeads_ = 0;
    uint32_t packedBytes_ = 0;
    uint32_t tileRows_ = 0;
    uint32_t planeBytes_ = 0;
    uint32_t readsWholeRows_ = 0;
};

// ----------------------------------------------------------------------------------------------------
// Online softmax close-out
// ----------------------------------------------------------------------------------------------------

// Normalises `heads` accumulators by 1 / (runSum + eps), casts them to the output type in UB and
// writes them to GM in one burst. runSum holds one kFp32PerBlock lane per head; sums and invs need
// heads * kFp32PerBlock floats each. Requires headSize to be a multiple of kFp32PerRepeat.
template <typename scalar_t>
__aicore__ inline void WriteNormalizedHeads(const AscendC::LocalTensor<float> &acc,
                                            const AscendC::LocalTensor<float> &runSum, const uint32_t heads,
                                            const uint32_t headSize, const AscendC::LocalTensor<float> &sums,
                                            const AscendC::LocalTensor<float> &invs,
                                            const AscendC::LocalTensor<scalar_t> &out,
                                            AscendC::GlobalTensor<scalar_t> &outputGm, const uint64_t outputElem)
{
    const uint8_t rowBlocks = static_cast<uint8_t>(headSize / kFp32PerBlock);
    const AscendC::BinaryRepeatParams rowRepeat{1, 1, 0, rowBlocks, rowBlocks, 1};

    for (uint32_t j = 0; j < heads; ++j) {
        AscendC::Adds(runSum[j * kFp32PerBlock], runSum[j * kFp32PerBlock], TurboQuantCodec4::kEps, 1);
    }
    for (uint32_t j = 0; j < heads; ++j) {
        AscendC::Brcb(sums[j * kFp32PerBlock], runSum[j * kFp32PerBlock], 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
    }
    AscendC::Duplicate(invs, 1.0f, heads * kFp32PerBlock);
    AscendC::Div(invs, invs, sums, heads * kFp32PerBlock);
    for (uint32_t col = 0; col < headSize; col += kFp32PerRepeat) {
        AscendC::Mul(acc[col], acc[col], invs, static_cast<uint64_t>(kFp32PerRepeat), static_cast<uint8_t>(heads),
                     rowRepeat);
    }

    AscendC::Cast(out, acc, AscendC::RoundMode::CAST_RINT, heads * headSize);
    SyncVectorToMte3();
    AscendC::DataCopy(outputGm[outputElem], out, heads * headSize);
    SyncMte3ToVector();
}

// ----------------------------------------------------------------------------------------------------
// In-launch reduction of split partials
// ----------------------------------------------------------------------------------------------------

// A token whose context exceeds the fused limit is decoded in splits, each leaving
// [acc (headSize), runMax lane, runSum lane] in the workspace. This merges a (token, head)'s splits
// with the same online-softmax recurrence the split used and writes the output token. Runs in the
// same launch as the splits, after AscendC::SyncAll.
template <typename scalar_t>
class TurboQuantPartialReducer {
public:
    __aicore__ inline explicit TurboQuantPartialReducer(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *workspace, __gm__ void *output, const uint32_t numHeads,
                                const uint32_t headSize, const uint32_t numSplits)
    {
        numHeads_ = numHeads;
        headSize_ = headSize;
        numSplits_ = numSplits;
        partialStride_ = headSize + kPartialTail;

        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(output));

        pipe_->InitBuffer(accBuf_, (2 * headSize_ + kPartialTail) * sizeof(float));
        pipe_->InitBuffer(stateBuf_, kStateLanes * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(blockBuf_, (2 * kBrcbDstLanes + 2 * kFp32PerBlock) * sizeof(float));
        pipe_->InitBuffer(outBuf_, headSize_ * sizeof(scalar_t));
        // Init boundary: drains every pipe before the reduction's first GM read of the split partials.
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Reduce(const uint32_t token, const uint32_t head)
    {
        const AscendC::LocalTensor<float> acc = accBuf_.Get<float>();
        const AscendC::LocalTensor<float> partAcc = acc[headSize_];
        const AscendC::LocalTensor<float> partState = acc[2 * headSize_];
        const AscendC::LocalTensor<float> partMax = partState[kPartialMaxLane];
        const AscendC::LocalTensor<float> partSum = partState[kPartialSumLane];

        const AscendC::LocalTensor<float> state = stateBuf_.Get<float>();
        const AscendC::LocalTensor<float> runMax = state[kPartialMaxLane];
        const AscendC::LocalTensor<float> runSum = state[kPartialSumLane];
        const AscendC::LocalTensor<float> newMax = state[2 * kFp32PerBlock];
        const AscendC::LocalTensor<float> alpha = state[3 * kFp32PerBlock];
        const AscendC::LocalTensor<float> beta = state[4 * kFp32PerBlock];

        const AscendC::LocalTensor<float> blocks = blockBuf_.Get<float>();
        const AscendC::LocalTensor<float> alphaBlock = blocks;
        const AscendC::LocalTensor<float> betaBlock = blocks[kBrcbDstLanes];
        const AscendC::LocalTensor<float> sums = blocks[2 * kBrcbDstLanes];
        const AscendC::LocalTensor<float> invs = blocks[2 * kBrcbDstLanes + kFp32PerBlock];

        AscendC::Duplicate(acc, 0.0f, headSize_);
        AscendC::Duplicate(state, 0.0f, kStateLanes * kFp32PerBlock);
        // Overlapping write: runMax lies inside the state the Duplicate above just zeroed.
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(runMax, kVectorNegInf, 1);

        for (uint32_t split = 0; split < numSplits_; ++split) {
            SyncVectorToMte2();
            AscendC::DataCopy(partAcc, workspaceGm_[PartialOffset(token, head, split)], partialStride_);
            SyncMte2ToVector();

            AscendC::Max(newMax, runMax, partMax, 1);
            AscendC::Sub(alpha, runMax, newMax, 1);
            AscendC::Sub(beta, partMax, newMax, 1);
            AscendC::Exp(alpha, alpha, 1);
            AscendC::Exp(beta, beta, 1);
            AscendC::Mul(runSum, runSum, alpha, 1);
            AscendC::Mul(partSum, partSum, beta, 1);
            AscendC::Add(runSum, runSum, partSum, 1);

            BroadcastScalar(alphaBlock, alpha);
            BroadcastScalar(betaBlock, beta);
            TurboQuantCodec4::BroadcastMul(acc, acc, alphaBlock, headSize_);
            TurboQuantCodec4::BroadcastMul(partAcc, partAcc, betaBlock, headSize_);
            AscendC::Add(acc, acc, partAcc, headSize_);
            AscendC::Adds(runMax, newMax, 0.0f, 1);
        }

        WriteNormalizedHeads<scalar_t>(acc, runSum, 1, headSize_, sums, invs, outBuf_.Get<scalar_t>(),
                                                     outputGm_,
                                                     (static_cast<uint64_t>(token) * numHeads_ + head) * headSize_);
    }

private:
    static constexpr uint32_t kStateLanes = 5;

    __aicore__ inline uint64_t PartialOffset(const uint32_t token, const uint32_t head, const uint32_t split) const
    {
        return ((static_cast<uint64_t>(token) * numHeads_ + head) * numSplits_ + split) * partialStride_;
    }

    AscendC::TPipe *pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stateBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> blockBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outBuf_;
    AscendC::GlobalTensor<float> workspaceGm_;
    AscendC::GlobalTensor<scalar_t> outputGm_;
    uint32_t numHeads_ = 0;
    uint32_t headSize_ = 0;
    uint32_t numSplits_ = 1;
    uint32_t partialStride_ = 0;
};

// ----------------------------------------------------------------------------------------------------
// The vector half of the Cube decode
// ----------------------------------------------------------------------------------------------------

template <TurboQuantMode MODE>
__aicore__ inline constexpr float OperandMax()
{
    return TurboQuantModeTraits<MODE>::kOperand == TurboQuantOperand::kFp4E2m1 ? 6.0f : 448.0f;
}

// The ring-buffered tile ingest and the head-batched softmax, validated on kv4fp8 only. The other modes
// keep the lock-step ingest and the per-head softmax they were verified with. kv4fp8 batches even one
// head per subcore: the per-head path's Level-2 reduces cost more than the row repeats
// (csrc/tests/TURBOQUANT_TESTS.md 13.28).
template <TurboQuantMode MODE>
constexpr bool kBatchedCubeDecode = MODE == TurboQuantMode::KV4_FP8;

// The heads one task covers, and the half of them this vector subcore owns. The Cube's dual-destination
// Fixpipe splits M rows in half, so subcore 0 owns rows [0, ceil(M/2)) and subcore 1 the rest.
struct TurboQuantTaskHeads {
    uint32_t first = 0;
    uint32_t rows = 0;
    uint32_t base = 0;
    uint32_t mine = 0;
};

// Mm is the Cube BMM service (TurboQuantCubeMm<MODE>) that owns the L1 operand slots, and Codec the
// mode's KV codec. Subcore 0 stages the K plane of every tile and subcore 1 the V plane. An NZ-tiled
// cache (kv4fp8) is read one (tile, kv head) per burst and unpacked straight into NZ order; the
// row-major modes read through TurboQuantTileBurst and unpack in bands.
//
// The batched decode keeps every per-head scalar of the softmax broadcast over its head's 8-lane block:
// one wide op advances all heads, the same field is the block operand of a row op, and lane j * 8 still
// holds head j's value for the per-task readers the two decodes share.
template <TurboQuantMode MODE, typename scalar_t, typename Mm, typename Codec>
class TurboQuantVectorDecodeService {
public:
    using OperandT = typename Mm::OperandT;
    static constexpr bool kBatched = kBatchedCubeDecode<MODE>;
    static constexpr bool kNzTiled = kStoresNzTiles<MODE>;
    static_assert(kNzTiled == Codec::kIsAffine, "only the byte-wise affine expand streams an NZ-tiled plane");
    // Ring slots of the tile ingest. Tile t reads into slot t % kCubeSlots, the slot of its L1 operands.
    static constexpr uint32_t kIngestSlots = kBatched ? kCubeSlots : 1;

    __aicore__ inline void Init(AscendC::TPipe *pipe, __gm__ void *queryRot, __gm__ void *keyCache,
                                __gm__ void *valueCache, __gm__ void *scaleCache, __gm__ void *modeTables,
                                __gm__ void *workspace, __gm__ void *output, const uint32_t numHeads,
                                const uint32_t numKvHeads, const uint32_t headSize, const uint32_t blockSize,
                                const uint32_t numSplits, const float scale, const float invSqrtLen)
    {
        ComputeLayout(numHeads, numKvHeads, headSize, blockSize, numSplits, scale);
        InitGlobalTensors(queryRot, keyCache, valueCache, scaleCache, modeTables, workspace, output);
        InitBuffers(pipe);
        codec_.Init(pipe, headSize_, Codec::kIsAffine ? kUnpackChunkRows : kCubeUnpackRows, invSqrtLen,
                    modeTablesGm_);
    }

    __aicore__ inline AscendC::LocalTensor<float> Scores() { return scoreBuf_.Get<float>(); }
    __aicore__ inline AscendC::LocalTensor<float> Context() { return contextBuf_.Get<float>(); }

    // Which of the task's heads this subcore owns. On the AIC every row is "rows" and nothing is mine.
    __aicore__ static inline TurboQuantTaskHeads Heads(const uint32_t first, const uint32_t rows)
    {
        TurboQuantTaskHeads heads;
        heads.first = first;
        heads.rows = rows;
        const uint32_t firstHalf = CeilDiv(rows, kVectorSubcoresPerBlock);
        heads.mine = firstHalf;
        if ASCEND_IS_AIV {
            if (AscendC::GetSubBlockIdx() != 0) {
                heads.base = firstHalf;
                heads.mine = rows - firstHalf;
            }
        }
        return heads;
    }

    __aicore__ inline void BeginTask(const TurboQuantTaskHeads &heads)
    {
        if (heads.mine == 0) {
            return;
        }
        AscendC::Duplicate(accBuf_.Get<float>(), 0.0f, heads.mine * headSize_);
        AscendC::Duplicate(stateBuf_.Get<float>(), 0.0f, kStateFields * kStateField);
        // Overlapping write: the runMax field lies inside the state the Duplicate above just zeroed.
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(StateField(kStateRunMax), kVectorNegInf, heads.mine * kFp32PerBlock);
    }

    // Quantises this subcore's query heads onto the operand grid, stages them into the Cube's L1 query slot
    // in NZ order, one row per head, and indexes the task's kv head in the tile scale lanes.
    __aicore__ inline void StageQuery(Mm &mm, const uint32_t token, const uint32_t kvHead,
                                      const TurboQuantTaskHeads &heads)
    {
        if (heads.mine == 0) {
            return;
        }
        const AscendC::LocalTensor<float> queryIn = queryInBuf_.Get<float>();
        const AscendC::LocalTensor<OperandT> queryOperand = queryOperandBuf_.Get<OperandT>();
        const AscendC::LocalTensor<OperandT> queryL1 = mm.A1Query();

        AscendC::DataCopy(queryIn,
                          queryRotGm_[(static_cast<uint64_t>(token) * numHeads_ + heads.first + heads.base) *
                                      headSize_],
                          heads.mine * headSize_);
        SyncMte2ToVector();

        if constexpr (kBatched) {
            ComputeQueryOperandRows(heads.mine);
        } else {
            ComputeQueryOperandHeads(heads.mine);
        }

        SyncVectorToMte3();
        for (uint32_t j = 0; j < heads.mine; ++j) {
            AscendC::DataCopy(queryL1[(heads.base + j) * kOperandC0], queryOperand[j * operandElems_],
                              queryRowToL1Params_);
        }

        // Byte offsets of the task's kv head in each tile row's scale slot: K lanes first, V lanes after.
        const int32_t slotBytes = static_cast<int32_t>(scaleSlot_ * sizeof(float));
        const AscendC::LocalTensor<int32_t> scaleIndex = scaleIndexBuf_.Get<int32_t>();
        AscendC::ArithProgression(scaleIndex, static_cast<int32_t>(kvHead * sizeof(float)), slotBytes,
                                  static_cast<int32_t>(kCubeTileRows));
        AscendC::ArithProgression(scaleIndex[kCubeTileRows],
                                  static_cast<int32_t>((numKvHeads_ + kvHead) * sizeof(float)), slotBytes,
                                  static_cast<int32_t>(kCubeTileRows));
    }

    // The mask a partial last tile is Min-ed with: 0-crossing at `valid`, +huge before, -huge after.
    __aicore__ inline void ComputeTailMask(const uint32_t valid)
    {
        const AscendC::LocalTensor<float> mask = maskBuf_.Get<float>();
        const AscendC::LocalTensor<int32_t> lanes = mask[kCubeTileRows].ReinterpretCast<int32_t>();
        AscendC::ArithProgression(lanes, 0, 1, static_cast<int32_t>(kCubeTileRows));
        AscendC::Cast(mask, lanes, AscendC::RoundMode::CAST_NONE, kCubeTileRows);
        AscendC::Adds(mask, mask, kMaskMidpoint - static_cast<float>(valid), kCubeTileRows);
        AscendC::Muls(mask, mask, kMaskGain, kCubeTileRows);
    }

    // Reads this subcore's plane of one tile in a wide burst, unpacks it in UB in NZ order and stages it into
    // the tile's L1 slot in one MTE3 burst. The lock-step ingest of the unbatched modes.
    __aicore__ inline void StageTile(Mm &mm, const uint32_t physical, const uint32_t rowBase, const uint32_t kvHead,
                                     const uint32_t l1Slot, const bool afterSoftmax)
    {
        const AscendC::LocalTensor<OperandT> l1Dst = PlaneOperand(mm, l1Slot);
        if (afterSoftmax) {
            SyncVectorToMte2();
        }

        const AscendC::LocalTensor<int8_t> packed = kvBuf_[0].Get<int8_t>();
        ReadPlane(physical, rowBase, kvHead, 0);
        SyncMte2ToVector();
        SelectKvHeadRows(packed, kvHead, 0);
        StageUnpackedToL1(packed, l1Dst);
    }

    // The MTE2 half of the ring ingest: one tile's plane and scale lanes into ring slot `slot`, posted to
    // the vector unit on an event of the slot's own so the stage can wait for exactly this read. The
    // caller orders it after the vector unit's last read of the slot's previous tile.
    __aicore__ inline void ReadTileToSlot(const uint32_t physical, const uint32_t rowBase, const uint32_t kvHead,
                                          const uint32_t slot)
    {
        ReadPlane(physical, rowBase, kvHead, slot);
        readEvents_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(readEvents_[slot]);
    }

    // The vector half: waits for slot `slot`'s read, unpacks it in UB in NZ order and stages it into the
    // slot's L1 operand in one MTE3 burst. The id is free again once the wait is issued: the next read that
    // can draw it is ordered behind this stage's vector work by the caller's V -> MTE2 edge.
    __aicore__ inline void StageSlotToL1(Mm &mm, const uint32_t kvHead, const uint32_t slot)
    {
        const AscendC::LocalTensor<OperandT> l1Dst = PlaneOperand(mm, slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(readEvents_[slot]);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(readEvents_[slot]);

        const AscendC::LocalTensor<int8_t> packed = kvBuf_[slot].Get<int8_t>();
        SelectKvHeadRows(packed, kvHead, slot);
        StageUnpackedToL1(packed, l1Dst);
    }

    // Scores for this subcore's heads arrived from the Cube: finish the logits, advance the running
    // max / log-sum per head, and stage the probability rows (on the operand grid) into L1. `slot` is the
    // ingest slot holding this tile's scale lanes. One body on purpose: split into helpers, the per-tile
    // tensor views are re-derived in each and the AIV issues more scalar instructions per tile.
    __aicore__ inline void ComputeSoftmaxAndStageProbs(Mm &mm, const uint32_t valid, const TurboQuantTaskHeads &heads,
                                                       const uint32_t slot)
    {
        if (heads.mine == 0) {
            return;
        }
        const AscendC::LocalTensor<float> scaleTile = scaleTileBuf_[slot].Get<float>();
        const AscendC::LocalTensor<float> reduce = reduceBuf_.Get<float>();
        const AscendC::LocalTensor<float> keyScale = reduce;
        const AscendC::LocalTensor<float> valueScale = reduce[kCubeTileRows];
        const AscendC::LocalTensor<OperandT> probOperand = probOperandBuf_.Get<OperandT>();
        const uint32_t probElems = Mm::OperandElems(kCubeTileRows);

        const AscendC::LocalTensor<uint32_t> scaleIndex = scaleIndexBuf_.Get<int32_t>().ReinterpretCast<uint32_t>();
        AscendC::Gather(keyScale, scaleTile, scaleIndex, kGatherSrcBase, kCubeTileRows);
        AscendC::Gather(valueScale, scaleTile, scaleIndex[kCubeTileRows], kGatherSrcBase, kCubeTileRows);
        AscendC::Muls(keyScale, keyScale, scoreScale_, kCubeTileRows);
        AscendC::Muls(valueScale, valueScale, invGain_, kCubeTileRows);

        if constexpr (kBatched) {
            ComputeSoftmaxRows(valid, heads.mine, keyScale, valueScale);
        } else {
            ComputeSoftmaxHeads(valid, heads.mine, keyScale, valueScale);
        }

        const AscendC::LocalTensor<OperandT> probsL1 = mm.A1Probs();
        SyncVectorToMte3();
        for (uint32_t j = 0; j < heads.mine; ++j) {
            AscendC::DataCopy(probsL1[(heads.base + j) * kOperandC0], probOperand[j * probElems], probRowToL1Params_);
        }
        SyncMte3ToVector();
    }

    // Decays the accumulators by alpha = exp(runMax - newMax) before this tile's context lands.
    __aicore__ inline void ComputeAccumulatorDecay(const TurboQuantTaskHeads &heads)
    {
        if (heads.mine == 0) {
            return;
        }
        const AscendC::LocalTensor<float> acc = accBuf_.Get<float>();
        const AscendC::LocalTensor<float> alpha = StateField(kStateAlpha);
        if constexpr (kBatched) {
            MulHeadRows(acc, alpha, heads.mine);
        } else {
            const AscendC::LocalTensor<float> blocks = reduceBuf_.Get<float>()[kReduceBlocks];
            for (uint32_t j = 0; j < heads.mine; ++j) {
                AscendC::Brcb(blocks[j * kFp32PerBlock], alpha[j * kFp32PerBlock], 1,
                              {1, static_cast<uint16_t>(kFp32PerBlock)});
            }
            MulHeadRows(acc, blocks, heads.mine);
        }
        AscendC::Adds(StateField(kStateRunMax), StateField(kStateNewMax), 0.0f, heads.mine * kFp32PerBlock);
    }

    // The Cube's context rows are probs * V on the operand grid: undo each head's prob scale and add.
    __aicore__ inline void ComputeAccumulate(const TurboQuantTaskHeads &heads)
    {
        if (heads.mine == 0) {
            return;
        }
        const AscendC::LocalTensor<float> acc = accBuf_.Get<float>();
        const AscendC::LocalTensor<float> context = contextBuf_.Get<float>();
        const AscendC::LocalTensor<float> probScale = StateField(kStateProbScale);
        if constexpr (kBatched) {
            const AscendC::LocalTensor<float> probScaleInv = StateField(kStateAccInv);
            const uint32_t lanes = heads.mine * kFp32PerBlock;
            AscendC::Duplicate(probScaleInv, 1.0f, lanes);
            AscendC::Div(probScaleInv, probScaleInv, probScale, lanes);
            MulHeadRows(context, probScaleInv, heads.mine);
        } else {
            const AscendC::LocalTensor<float> reduce = reduceBuf_.Get<float>();
            const AscendC::LocalTensor<float> probScaleBlock = reduce[kReduceAccBlock];
            const AscendC::LocalTensor<float> probScaleInv = reduce[kReduceAccInv];
            for (uint32_t j = 0; j < heads.mine; ++j) {
                BroadcastScalar(probScaleBlock, probScale[j * kFp32PerBlock]);
                AscendC::Duplicate(probScaleInv, 1.0f, kFp32PerBlock);
                AscendC::Div(probScaleInv, probScaleInv, probScaleBlock, kFp32PerBlock);
                TurboQuantCodec4::BroadcastMul(context[j * headSize_], context[j * headSize_],
                                                             probScaleInv, headSize_);
            }
        }
        AscendC::Add(acc, acc, context, heads.mine * headSize_);
    }

    // A fused token writes its output straight to GM; a split token leaves its partials for the
    // in-launch reduction.
    __aicore__ inline void StageTaskOutput(const uint32_t token, const uint32_t split, const TurboQuantTaskHeads &heads,
                                           const bool fused)
    {
        if (heads.mine == 0) {
            return;
        }
        const AscendC::LocalTensor<float> reduce = reduceBuf_.Get<float>();
        if (fused) {
            WriteNormalizedHeads<scalar_t>(
                accBuf_.Get<float>(), StateField(kStateRunSum), heads.mine, headSize_, reduce[kReduceBlocks],
                reduce[kReduceInv], outBuf_.Get<scalar_t>(), outputGm_,
                (static_cast<uint64_t>(token) * numHeads_ + heads.first + heads.base) * headSize_);
            return;
        }
        StagePartials(token, split, heads);
    }

private:
    static constexpr uint32_t kHalfRows = kCubeTileM / 2;
    static constexpr uint32_t kUnpackChunkRows = kCubeTileRows / 2;
    static constexpr uint32_t kKeyPlane = 0;
    static constexpr uint32_t kValuePlane = 1;
    static constexpr float kMaskGain = -5.0e36f;
    static constexpr float kMaskMidpoint = 0.5f;

    static constexpr uint32_t kStateRunMax = 0;
    static constexpr uint32_t kStateRunSum = 1;
    static constexpr uint32_t kStateTileMax = 2;
    static constexpr uint32_t kStateNewMax = 3;
    static constexpr uint32_t kStateAlpha = 4;
    static constexpr uint32_t kStateProbScale = 5;
    static constexpr uint32_t kStateAmax = 6;
    static constexpr uint32_t kStateQScale = 7;
    // Batched only: the per-head reciprocals, the row-reduce outputs (one lane per head) and two blocks.
    static constexpr uint32_t kStateQInv = 8;
    static constexpr uint32_t kStateRowReduce = 9;
    static constexpr uint32_t kStatePart = 10;
    static constexpr uint32_t kStateAccInv = 11;
    static constexpr uint32_t kStateFields = kBatched ? 12 : 8;
    static constexpr uint32_t kStateField = kHalfRows * kFp32PerBlock;
    static constexpr uint8_t kTileBlocks = static_cast<uint8_t>(kCubeTileRows / kFp32PerBlock);

    static constexpr uint32_t kReduceFloats = 512;
    static constexpr uint32_t kReduceBlock = 2 * kCubeTileRows;
    static constexpr uint32_t kReducePart = 3 * kCubeTileRows;
    static constexpr uint32_t kReduceBlocks = kReducePart + kFp32PerBlock;
    static constexpr uint32_t kReduceInv = kReduceBlocks + 2 * kCubeTileRows;
    static constexpr uint32_t kReduceAccBlock = kReduceInv + kCubeTileRows;
    static constexpr uint32_t kReduceAccInv = kReduceAccBlock + kCubeTileRows;

    __aicore__ inline void ComputeLayout(const uint32_t numHeads, const uint32_t numKvHeads, const uint32_t headSize,
                                         const uint32_t blockSize, const uint32_t numSplits, const float scale)
    {
        numHeads_ = numHeads;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        blockSize_ = blockSize;
        numSplits_ = numSplits;
        packedBytes_ = Codec::PackedBytes(headSize);
        packedPlane_ = numKvHeads_ * packedBytes_;
        scaleSlot_ = ScaleSlotFloats(numKvHeads_);
        partialStride_ = headSize + kPartialTail;
        operandElems_ = Mm::OperandElems(headSize_);
        tilePlaneBytes_ = kCubeTileRows * packedBytes_;
        unpackChunkBytes_ = kUnpackChunkRows * packedBytes_;
        nzTileToL1Params_ =
            AscendC::DataCopyParams{1, static_cast<uint16_t>(kCubeTileRows * operandElems_ / kOperandC0), 0, 0};
        bandToL1Params_ = AscendC::DataCopyParams{static_cast<uint16_t>(operandElems_ / kOperandC0),
                                                  static_cast<uint16_t>(kCubeUnpackRows), 0,
                                                  static_cast<uint16_t>(kCubeTileRows - kCubeUnpackRows)};
        queryRowToL1Params_ = AscendC::DataCopyParams{static_cast<uint16_t>(headSize_ / kOperandC0), 1, 0,
                                                      static_cast<uint16_t>(kCubeTileM - 1)};
        probRowToL1Params_ = AscendC::DataCopyParams{
            static_cast<uint16_t>(Mm::OperandElems(kCubeTileRows) / kOperandC0), 1, 0,
            static_cast<uint16_t>(kCubeTileM - 1)};
        const uint8_t rowBlocks = static_cast<uint8_t>(headSize_ / kFp32PerBlock);
        rowRepeatParams_ = AscendC::BinaryRepeatParams{1, 1, 0, rowBlocks, rowBlocks, 1};
        tileByBlockParams_ = AscendC::BinaryRepeatParams{1, 1, 0, kTileBlocks, kTileBlocks, 1};
        tileByVectorParams_ = AscendC::BinaryRepeatParams{1, 1, 1, kTileBlocks, kTileBlocks, 0};
        scoreScale_ = scale / TurboQuantModeTraits<MODE>::kGain;
        invGain_ = 1.0f / TurboQuantModeTraits<MODE>::kGain;
    }

    __aicore__ inline void InitGlobalTensors(__gm__ void *queryRot, __gm__ void *keyCache, __gm__ void *valueCache,
                                             __gm__ void *scaleCache, __gm__ void *modeTables, __gm__ void *workspace,
                                             __gm__ void *output)
    {
        queryRotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(queryRot));
        keyCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(keyCache));
        valueCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(valueCache));
        scaleCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scaleCache));
        modeTablesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(modeTables));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(output));
    }

    // The UB layout is allocation order, so this order is part of the kernel's memory contract.
    __aicore__ inline void InitBuffers(AscendC::TPipe *pipe)
    {
        for (uint32_t slot = 0; slot < kIngestSlots; ++slot) {
            pipe->InitBuffer(kvBuf_[slot], kCubeTileRows * packedBytes_);
        }
        for (uint32_t slot = 0; slot < kIngestSlots; ++slot) {
            pipe->InitBuffer(scaleTileBuf_[slot], kCubeTileRows * scaleSlot_ * sizeof(float));
        }
        if constexpr (kNzTiled) {
            pipe->InitBuffer(nzOperandBuf_, kCubeTileRows * operandElems_);
        } else {
            pipe->InitBuffer(bandOperandBuf_, kCubeUnpackRows * operandElems_);
        }
        pipe->InitBuffer(accBuf_, kHalfRows * headSize_ * sizeof(float));
        pipe->InitBuffer(queryWorkBuf_, (kBatched ? kHalfRows : 1) * headSize_ * sizeof(float));
        pipe->InitBuffer(queryInBuf_, kHalfRows * headSize_ * sizeof(float));
        pipe->InitBuffer(queryOperandBuf_, kHalfRows * operandElems_);
        pipe->InitBuffer(scoreBuf_, kHalfRows * kCubeTileRows * sizeof(float));
        pipe->InitBuffer(probOperandBuf_, kHalfRows * Mm::OperandElems(kCubeTileRows));
        pipe->InitBuffer(contextBuf_, kHalfRows * headSize_ * sizeof(float));
        pipe->InitBuffer(stateBuf_, kStateFields * kStateField * sizeof(float));
        pipe->InitBuffer(reduceBuf_, kReduceFloats * sizeof(float));
        pipe->InitBuffer(scaleIndexBuf_, 2 * kCubeTileRows * sizeof(int32_t));
        pipe->InitBuffer(scratchBuf_, (headSize_ > kCubeTileRows ? headSize_ : kCubeTileRows) * sizeof(float));
        pipe->InitBuffer(outBuf_, kHalfRows * headSize_ * sizeof(scalar_t));
        pipe->InitBuffer(maskBuf_, 2 * kCubeTileRows * sizeof(float));
        if constexpr (!kNzTiled) {
            rowMajorBurst_.Init(pipe, numKvHeads_, packedBytes_, kCubeTileRows, kIngestSlots);
        }
    }

    __aicore__ inline AscendC::LocalTensor<float> StateField(const uint32_t field)
    {
        return stateBuf_.Get<float>()[field * kStateField];
    }

    // Subcore 0 stages the K plane and subcore 1 the V plane.
    __aicore__ static inline uint32_t Plane() { return AscendC::GetSubBlockIdx() == 0 ? kKeyPlane : kValuePlane; }

    __aicore__ inline AscendC::LocalTensor<OperandT> PlaneOperand(Mm &mm, const uint32_t slot)
    {
        return Plane() == kKeyPlane ? mm.B1K(slot) : mm.B1V(slot);
    }

    // This subcore's plane of one tile and the tile's scale lanes, into ingest slot `slot`. rowBase is a
    // whole number of tiles into the block, so an NZ-tiled (tile, kv head) starts at
    // (row * kv heads + kvHead * kCubeTileRows) * packedBytes (turboquant_layout.h, NzTiledPackedByte).
    __aicore__ inline void ReadPlane(const uint32_t physical, const uint32_t rowBase, const uint32_t kvHead,
                                     const uint32_t slot)
    {
        const uint64_t row = static_cast<uint64_t>(physical) * blockSize_ + rowBase;
        AscendC::GlobalTensor<int8_t> &cacheGm = Plane() == kKeyPlane ? keyCacheGm_ : valueCacheGm_;
        if constexpr (kNzTiled) {
            const uint64_t tile = (row * numKvHeads_ + static_cast<uint64_t>(kvHead) * kCubeTileRows) * packedBytes_;
            AscendC::DataCopy(kvBuf_[slot].Get<int8_t>(), cacheGm[tile], tilePlaneBytes_);
        } else {
            rowMajorBurst_.ReadTileRows(kvBuf_[slot].Get<int8_t>(), cacheGm, row * packedPlane_, kvHead, slot);
        }
        AscendC::DataCopy(scaleTileBuf_[slot].Get<float>(), scaleCacheGm_[row * scaleSlot_],
                          kCubeTileRows * scaleSlot_);
    }

    // The vector half of a row-major read that had to take the whole row plane. An NZ-tiled tile never
    // needs it.
    __aicore__ inline void SelectKvHeadRows(const AscendC::LocalTensor<int8_t> &packed, const uint32_t kvHead,
                                            const uint32_t slot)
    {
        if constexpr (!kNzTiled) {
            rowMajorBurst_.SelectKvHeadRows(packed, kvHead, slot);
        }
    }

    __aicore__ inline void StageUnpackedToL1(const AscendC::LocalTensor<int8_t> &packed,
                                             const AscendC::LocalTensor<OperandT> &l1Dst)
    {
        if constexpr (kNzTiled) {
            StageNzTileToL1(packed, l1Dst);
        } else {
            StageBandsToL1(packed, l1Dst);
        }
    }

    // rows[j * headSize_ ..] *= blocks[j * kFp32PerBlock] for `heads` rows, one 64-lane sweep per column.
    __aicore__ inline void MulHeadRows(const AscendC::LocalTensor<float> &rows,
                                       const AscendC::LocalTensor<float> &blocks, const uint32_t heads)
    {
        for (uint32_t col = 0; col < headSize_; col += kFp32PerRepeat) {
            AscendC::Mul(rows[col], rows[col], blocks, static_cast<uint64_t>(kFp32PerRepeat),
                         static_cast<uint8_t>(heads), rowRepeatParams_);
        }
    }

    // One sum or max per row of `rows` (kCubeTileRows wide) into rowReduce lanes 0..heads-1, then
    // broadcast over the heads' blocks of `blocks`.
    template <AscendC::ReduceType REDUCE>
    __aicore__ inline void ReduceTileRows(const AscendC::LocalTensor<float> &blocks,
                                          const AscendC::LocalTensor<float> &rows, const uint32_t heads)
    {
        const AscendC::LocalTensor<float> rowReduce = StateField(kStateRowReduce);
        if constexpr (REDUCE == AscendC::ReduceType::SUM) {
            AscendC::ReduceRepeat<REDUCE>(rowReduce, rows, static_cast<int32_t>(kCubeTileRows),
                                          static_cast<int32_t>(heads), 1, 1, kTileBlocks);
        } else {
            AscendC::ReduceRepeat<REDUCE>(rowReduce, rows, static_cast<int32_t>(kCubeTileRows),
                                          static_cast<int32_t>(heads), 1, 1, kTileBlocks,
                                          AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        }
        AscendC::Brcb(blocks, rowReduce, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
    }

    // The unbatched modes: one head at a time, and a scalar readback of each 1 / qScale.
    __aicore__ inline void ComputeQueryOperandHeads(const uint32_t heads)
    {
        const AscendC::LocalTensor<float> queryIn = queryInBuf_.Get<float>();
        const AscendC::LocalTensor<float> queryAbs = queryWorkBuf_.Get<float>();
        const AscendC::LocalTensor<OperandT> queryOperand = queryOperandBuf_.Get<OperandT>();
        const AscendC::LocalTensor<float> amax = StateField(kStateAmax);
        const AscendC::LocalTensor<float> qScale = StateField(kStateQScale);
        const AscendC::LocalTensor<float> qScaleBlock = reduceBuf_.Get<float>()[kReduceBlock];
        const AscendC::LocalTensor<float> scratch = scratchBuf_.Get<float>();

        for (uint32_t j = 0; j < heads; ++j) {
            const AscendC::LocalTensor<float> query = queryIn[j * headSize_];
            const uint32_t lane = j * kFp32PerBlock;
            AscendC::Abs(queryAbs, query, headSize_);
            AscendC::ReduceMax<float>(amax[lane], queryAbs, scratch, headSize_, false);
            AscendC::Adds(amax[lane], amax[lane], TurboQuantCodec4::kEps, 1);
            AscendC::Duplicate(qScale[lane], OperandMax<MODE>(), 1);
            AscendC::Div(qScale[lane], qScale[lane], amax[lane], 1);
            BroadcastScalar(qScaleBlock, qScale[lane]);
            TurboQuantCodec4::BroadcastMul(query, query, qScaleBlock, headSize_);
            codec_.template CastToOperand<OperandT>(queryOperand[j * operandElems_], query, headSize_);
        }
        // Vector -> scalar hand-off: GetValue reads qScale lanes the vector pipe has just written.
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < heads; ++j) {
            qScaleInv_[j] = 1.0f / qScale.GetValue(j * kFp32PerBlock);
        }
    }

    // All of this subcore's query rows at once. The reciprocals stay in UB as head blocks, so nothing here
    // waits on the vector unit.
    __aicore__ inline void ComputeQueryOperandRows(const uint32_t heads)
    {
        const AscendC::LocalTensor<float> queryIn = queryInBuf_.Get<float>();
        const AscendC::LocalTensor<float> queryAbs = queryWorkBuf_.Get<float>();
        const AscendC::LocalTensor<OperandT> queryOperand = queryOperandBuf_.Get<OperandT>();
        const AscendC::LocalTensor<float> rowMax = StateField(kStateRowReduce);
        const AscendC::LocalTensor<float> amax = StateField(kStateAmax);
        const AscendC::LocalTensor<float> qScale = StateField(kStateQScale);
        const AscendC::LocalTensor<float> qInv = StateField(kStateQInv);
        const AscendC::LocalTensor<float> scratch = scratchBuf_.Get<float>();
        const uint32_t lanes = heads * kFp32PerBlock;

        AscendC::Abs(queryAbs, queryIn, heads * headSize_);
        // A row is wider than one register, so this reduce stays per head. It has no scalar readback.
        for (uint32_t j = 0; j < heads; ++j) {
            AscendC::ReduceMax<float>(rowMax[j], queryAbs[j * headSize_], scratch, headSize_, false);
        }
        AscendC::Brcb(amax, rowMax, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
        AscendC::Adds(amax, amax, TurboQuantCodec4::kEps, lanes);
        AscendC::Duplicate(qScale, OperandMax<MODE>(), lanes);
        AscendC::Duplicate(qInv, 1.0f, lanes);
        AscendC::Div(qScale, qScale, amax, lanes);
        AscendC::Div(qInv, qInv, qScale, lanes);
        MulHeadRows(queryIn, qScale, heads);
        codec_.template CastToOperand<OperandT>(queryOperand, queryIn, heads * headSize_);
    }

    __aicore__ inline void ComputeSoftmaxHeads(const uint32_t valid, const uint32_t heads,
                                               const AscendC::LocalTensor<float> &keyScale,
                                               const AscendC::LocalTensor<float> &valueScale)
    {
        const AscendC::LocalTensor<float> scores = scoreBuf_.Get<float>();
        const AscendC::LocalTensor<float> runMax = StateField(kStateRunMax);
        const AscendC::LocalTensor<float> runSum = StateField(kStateRunSum);
        const AscendC::LocalTensor<float> tileMax = StateField(kStateTileMax);
        const AscendC::LocalTensor<float> newMax = StateField(kStateNewMax);
        const AscendC::LocalTensor<float> alpha = StateField(kStateAlpha);
        const AscendC::LocalTensor<float> probScale = StateField(kStateProbScale);
        const AscendC::LocalTensor<float> reduce = reduceBuf_.Get<float>();
        const AscendC::LocalTensor<float> scalarBlock = reduce[kReduceBlock];
        const AscendC::LocalTensor<float> part = reduce[kReducePart];
        const AscendC::LocalTensor<float> scratch = scratchBuf_.Get<float>();
        const AscendC::LocalTensor<float> mask = maskBuf_.Get<float>();
        const AscendC::LocalTensor<OperandT> probOperand = probOperandBuf_.Get<OperandT>();
        const uint32_t probElems = Mm::OperandElems(kCubeTileRows);

        for (uint32_t j = 0; j < heads; ++j) {
            const AscendC::LocalTensor<float> row = scores[j * kCubeTileRows];
            const uint32_t lane = j * kFp32PerBlock;
            AscendC::Mul(row, row, keyScale, kCubeTileRows);
            AscendC::Muls(row, row, qScaleInv_[j], kCubeTileRows);
            if (valid < kCubeTileRows) {
                AscendC::Min(row, row, mask, static_cast<int32_t>(kCubeTileRows));
            }

            AscendC::ReduceMax<float>(tileMax[lane], row, scratch, kCubeTileRows, false);
            AscendC::Max(newMax[lane], runMax[lane], tileMax[lane], 1);
            AscendC::Sub(alpha[lane], runMax[lane], newMax[lane], 1);
            AscendC::Exp(alpha[lane], alpha[lane], 1);

            BroadcastScalar(scalarBlock, newMax[lane]);
            BroadcastSub(row, row, scalarBlock, kCubeTileRows);
            AscendC::Exp(row, row, kCubeTileRows);

            AscendC::ReduceSum<float>(part, row, scratch, kCubeTileRows);
            AscendC::Mul(runSum[lane], runSum[lane], alpha[lane], 1);
            AscendC::Add(runSum[lane], runSum[lane], part, 1);

            AscendC::Mul(row, row, valueScale, kCubeTileRows);
            AscendC::ReduceMax<float>(part, row, scratch, kCubeTileRows, false);
            AscendC::Adds(part, part, TurboQuantCodec4::kEps, 1);
            AscendC::Duplicate(probScale[lane], OperandMax<MODE>(), 1);
            AscendC::Div(probScale[lane], probScale[lane], part, 1);
            BroadcastScalar(scalarBlock, probScale[lane]);
            TurboQuantCodec4::BroadcastMul(row, row, scalarBlock, kCubeTileRows);

            codec_.template CastToOperand<OperandT>(probOperand[j * probElems], row, kCubeTileRows);
        }
    }

    // The same recurrence over every row at once: a row op repeats over the heads, a per-head scalar is a
    // head block, and the row sums and maxima are whole-row reduces with no scalar readback. Logits, one
    // online-softmax step (newMax, alpha, runSum), then the probability rows on the operand grid.
    __aicore__ inline void ComputeSoftmaxRows(const uint32_t valid, const uint32_t heads,
                                              const AscendC::LocalTensor<float> &keyScale,
                                              const AscendC::LocalTensor<float> &valueScale)
    {
        const AscendC::LocalTensor<float> scores = scoreBuf_.Get<float>();
        const AscendC::LocalTensor<float> runMax = StateField(kStateRunMax);
        const AscendC::LocalTensor<float> runSum = StateField(kStateRunSum);
        const AscendC::LocalTensor<float> tileMax = StateField(kStateTileMax);
        const AscendC::LocalTensor<float> newMax = StateField(kStateNewMax);
        const AscendC::LocalTensor<float> alpha = StateField(kStateAlpha);
        const AscendC::LocalTensor<float> probScale = StateField(kStateProbScale);
        const AscendC::LocalTensor<float> qInv = StateField(kStateQInv);
        const AscendC::LocalTensor<float> part = StateField(kStatePart);
        const AscendC::LocalTensor<float> mask = maskBuf_.Get<float>();
        const AscendC::LocalTensor<OperandT> probOperand = probOperandBuf_.Get<OperandT>();
        const uint64_t rowLanes = kCubeTileRows;
        const uint8_t rows = static_cast<uint8_t>(heads);
        const uint32_t lanes = heads * kFp32PerBlock;

        AscendC::Mul(scores, scores, keyScale, rowLanes, rows, tileByVectorParams_);
        AscendC::Mul(scores, scores, qInv, rowLanes, rows, tileByBlockParams_);
        if (valid < kCubeTileRows) {
            AscendC::Min(scores, scores, mask, rowLanes, rows, tileByVectorParams_);
        }

        ReduceTileRows<AscendC::ReduceType::MAX>(tileMax, scores, heads);
        AscendC::Max(newMax, runMax, tileMax, static_cast<int32_t>(lanes));
        AscendC::Sub(alpha, runMax, newMax, lanes);
        AscendC::Sub(scores, scores, newMax, rowLanes, rows, tileByBlockParams_);
        AscendC::Exp(alpha, alpha, lanes);
        AscendC::Exp(scores, scores, heads * kCubeTileRows);

        ReduceTileRows<AscendC::ReduceType::SUM>(part, scores, heads);
        AscendC::Mul(runSum, runSum, alpha, lanes);
        AscendC::Add(runSum, runSum, part, lanes);
        AscendC::Mul(scores, scores, valueScale, rowLanes, rows, tileByVectorParams_);

        ReduceTileRows<AscendC::ReduceType::MAX>(part, scores, heads);
        AscendC::Duplicate(probScale, OperandMax<MODE>(), lanes);
        AscendC::Adds(part, part, TurboQuantCodec4::kEps, lanes);
        AscendC::Div(probScale, probScale, part, lanes);
        AscendC::Mul(scores, scores, probScale, rowLanes, rows, tileByBlockParams_);

        codec_.template CastToOperand<OperandT>(probOperand, scores, heads * kCubeTileRows);
    }

    __aicore__ inline uint64_t PartialOffset(const uint32_t token, const uint32_t head, const uint32_t split) const
    {
        return ((static_cast<uint64_t>(token) * numHeads_ + head) * numSplits_ + split) * partialStride_;
    }

    // An NZ-tiled plane is already the NZ image of its bytes, and the expand is byte-wise, so a chunk's
    // low nibbles land in place on NZ groups [0, groups) and its high nibbles on [groups, 2 * groups).
    // The staging buffer is the L1 operand as unpacked, and one MTE3 burst stages it.
    __aicore__ inline void StageNzTileToL1(const AscendC::LocalTensor<int8_t> &packed,
                                           const AscendC::LocalTensor<OperandT> &l1Dst)
    {
        const AscendC::LocalTensor<OperandT> nzLow = nzOperandBuf_.Get<OperandT>();
        const AscendC::LocalTensor<OperandT> nzHigh = nzLow[tilePlaneBytes_];
        for (uint32_t base = 0; base < tilePlaneBytes_; base += unpackChunkBytes_) {
            codec_.template UnpackAffine<OperandT>(nzLow[base], nzHigh[base], packed[base],
                                                                 unpackChunkBytes_);
        }
        SyncVectorToMte3();
        AscendC::DataCopy(l1Dst, nzLow, nzTileToL1Params_);
        SyncMte3ToVector();
    }

    __aicore__ inline void StageBandsToL1(const AscendC::LocalTensor<int8_t> &packed,
                                          const AscendC::LocalTensor<OperandT> &l1Dst)
    {
        const AscendC::LocalTensor<OperandT> unpacked = bandOperandBuf_.Get<OperandT>();
        const uint32_t bandElems = kCubeUnpackRows * kOperandC0;
        for (uint32_t band = 0; band < kCubeTileRows / kCubeUnpackRows; ++band) {
            codec_.Unpack(unpacked, packed[band * kCubeUnpackRows * packedBytes_], kCubeUnpackRows, headSize_);
            SyncVectorToMte3();
            AscendC::DataCopy(l1Dst[band * bandElems], unpacked, bandToL1Params_);
        }
        SyncMte3ToVector();
    }

    __aicore__ inline void StagePartials(const uint32_t token, const uint32_t split, const TurboQuantTaskHeads &heads)
    {
        const AscendC::LocalTensor<float> acc = accBuf_.Get<float>();
        const AscendC::LocalTensor<float> runMax = StateField(kStateRunMax);
        const AscendC::LocalTensor<float> runSum = StateField(kStateRunSum);
        const AscendC::LocalTensor<float> tails = scoreBuf_.Get<float>();
        AscendC::Duplicate(tails, 0.0f, heads.mine * kPartialTail);
        for (uint32_t j = 0; j < heads.mine; ++j) {
            AscendC::Adds(tails[j * kPartialTail + kPartialMaxLane], runMax[j * kFp32PerBlock], 0.0f, 1);
            AscendC::Adds(tails[j * kPartialTail + kPartialSumLane], runSum[j * kFp32PerBlock], 0.0f, 1);
        }
        SyncVectorToMte3();
        for (uint32_t j = 0; j < heads.mine; ++j) {
            const uint64_t offset = PartialOffset(token, heads.first + heads.base + j, split);
            AscendC::DataCopy(workspaceGm_[offset], acc[j * headSize_], headSize_);
            AscendC::DataCopy(workspaceGm_[offset + headSize_], tails[j * kPartialTail], kPartialTail);
        }
        SyncMte3ToVector();
    }

    Codec codec_;
    TurboQuantTileBurst rowMajorBurst_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> kvBuf_[kCubeSlots];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleTileBuf_[kCubeSlots];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bandOperandBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> nzOperandBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> queryWorkBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> queryInBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> queryOperandBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scoreBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> probOperandBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> contextBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stateBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleIndexBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scratchBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> maskBuf_;
    AscendC::GlobalTensor<float> queryRotGm_;
    AscendC::GlobalTensor<int8_t> keyCacheGm_;
    AscendC::GlobalTensor<int8_t> valueCacheGm_;
    AscendC::GlobalTensor<float> scaleCacheGm_;
    AscendC::GlobalTensor<int32_t> modeTablesGm_;
    AscendC::GlobalTensor<float> workspaceGm_;
    AscendC::GlobalTensor<scalar_t> outputGm_;
    AscendC::DataCopyParams nzTileToL1Params_;
    AscendC::DataCopyParams bandToL1Params_;
    AscendC::DataCopyParams queryRowToL1Params_;
    AscendC::DataCopyParams probRowToL1Params_;
    AscendC::BinaryRepeatParams rowRepeatParams_;
    AscendC::BinaryRepeatParams tileByBlockParams_;
    AscendC::BinaryRepeatParams tileByVectorParams_;
    AscendC::TEventID readEvents_[kCubeSlots] = {};
    float qScaleInv_[kHalfRows] = {};
    float scoreScale_ = 0.0f;
    float invGain_ = 1.0f;
    uint32_t numHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t numSplits_ = 1;
    uint32_t packedBytes_ = 0;
    uint32_t packedPlane_ = 0;
    uint32_t scaleSlot_ = 0;
    uint32_t partialStride_ = 0;
    uint32_t operandElems_ = 0;
    uint32_t tilePlaneBytes_ = 0;
    uint32_t unpackChunkBytes_ = 0;
};

}
}

#endif
