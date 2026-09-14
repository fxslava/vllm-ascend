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

/*
 * TurboQuant 4-bit KV-cache reshape and decode kernels.
 *
 * The file implements the packed-cache layout and the split decode/partial
 * combine path used by the NPU flash-decoding schedule.
 */

#include "kernel_operator.h"

#include "../../kernels/types.h"
#include "turboquant_codec_950.h"

namespace {

using vllm_ascend::turboquant::kBrcbDstLanes;
using vllm_ascend::turboquant::kFp32PerBlock;
using vllm_ascend::turboquant::kFp32PerRepeat;
using vllm_ascend::turboquant::kGatherSrcBase;
using vllm_ascend::turboquant::TurboQuantCodec4;

// Rows of the paged KV block processed per inner iteration.  Sized so that the
// working set (packed tiles, dequantised tiles, the tiled query and the two
// products) stays well inside UB for head_size up to 256, and so that Brcb can
// broadcast the whole tile of probabilities in whole repeats.
constexpr uint32_t kTileRows = 16;
// Softmax running maximum before any block has been seen.
constexpr float kNegInf = -1.0e30f;
// fp32 words of workspace per (token, head, split) partial: the accumulator,
// then one 32B block holding the running max and a second holding the running
// sum.  Two blocks, because every vector operand base must be 32-byte aligned
// and a LocalTensor view at lane 1 is not.
//
// Mirrored by turboquant_adpt::kPartialTail in turboquant_torch_adpt.h and by
// turboquant_host::kPartialTail in csrc/tests/common/turboquant_launch.hpp.
constexpr uint32_t kPartialTail = 2u * kFp32PerBlock;
// Lane offsets within that tail.
constexpr uint32_t kPartialMaxLane = 0;
constexpr uint32_t kPartialSumLane = kFp32PerBlock;
// Depth of the ring that carries a token's cache address from the copy-in stage
// to the copy-out stage.  Four, not two: at steady state the prefetch for i + 1
// and the flush for i - 1 are in flight together, and those two indices share a
// parity.
constexpr uint32_t kSlotRing = 4;

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline uint32_t RoundUp(uint32_t value, uint32_t multiple)
{
    return CeilDiv(value, multiple) * multiple;
}

// fp32 words a single token occupies in the scale plane: K then V for every kv
// head, padded to a whole 32-byte burst.  Mirrored by the host in
// turboquant_torch_adpt.h and by vllm_ascend/attention/turboquant_v1.py.
__aicore__ inline uint32_t ScaleSlotFloats(uint32_t numKvHeads)
{
    return RoundUp(2u * numKvHeads, kFp32PerBlock);
}

// dst[i] = src[i] - scalarBlock[i % 8], for a 32B block whose lanes are equal.
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

// Splay one value across a 32B block so it can be applied with a zero-stride
// Mul/Sub instead of being read back into a scalar register.
__aicore__ inline void BroadcastScalar(const AscendC::LocalTensor<float> &dst, const AscendC::LocalTensor<float> &src)
{
    AscendC::Brcb(dst, src, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
    AscendC::PipeBarrier<PIPE_V>();
}

/*
 * npu_turboquant_reshape_and_cache
 *
 * One AIV core owns a contiguous run of tokens and processes a whole token --
 * every kv head, key and value -- per pipeline stage, so the scatter is three
 * aligned DataCopy bursts.  The loop is a three-stage software pipeline:
 * CopyIn(i + 1) is issued before Compute(i), because DeQue inside Compute waits
 * on the MTE2 flag.
 */
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

        // Two tokens in flight on each side of the compute stage.
        pipe_->InitBuffer(inQueue_, 2, 2 * headPlane_ * sizeof(scalar_t));
        pipe_->InitBuffer(outPacked_, 2, 2 * packedPlane_ * sizeof(int8_t));
        pipe_->InitBuffer(outScale_, 2, scaleSlot_ * sizeof(float));
        pipe_->InitBuffer(workBuf_, 2 * headSize_ * sizeof(float));
        pipe_->InitBuffer(signBuf_, headSize_ * sizeof(float));
        // One 32B-aligned landing slot per scale: Quantize4Bit broadcasts its
        // result with Brcb, which needs an aligned base and eight readable
        // lanes, so the steps cannot be written straight into the packed slot.
        pipe_->InitBuffer(stepBuf_, 2 * numKvHeads_ * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(scaleIdxBuf_, 2 * numKvHeads_ * sizeof(int32_t));

        codec_.Init(pipe_, headSize_, 1, invSqrtLen, tablesGm_);

        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::DataCopy(signs, piSignsGm_, headSize_);

        // Compaction table for the stride-8 step slots. One instruction, once,
        // outside every loop -- not the per-launch table build this file used to
        // do inside the codec.
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

        // Prologue.
        CopyIn(start, 0);
        // Steady state: prefetch i + 1, compute i, flush i - 1.
        for (uint32_t i = 0; i < total; ++i) {
            if (i + 1 < total) {
                CopyIn(start + i + 1, i + 1);
            }
            Compute();
            if (i > 0) {
                CopyOut(i - 1);
            }
        }
        // Epilogue.
        CopyOut(total - 1);
    }

private:
    __aicore__ inline void CopyIn(uint32_t token, uint32_t step)
    {
        // Addressing only: one slot id per token, never a per-element read.
        const int32_t slot = slotGm_.GetValue(token);
        const uint32_t ring = step % kSlotRing;
        if (slot < 0) {
            // A padded token still travels the pipeline so the stage count stays
            // regular; only its flush is suppressed.
            slotValid_[ring] = false;
            cacheOffset_[ring] = 0;
            scaleOffset_[ring] = 0;
        } else {
            // The cache is [num_blocks, block_size, ...] and contiguous, so the
            // slot id is already the flat (block, offset) row index.
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

        // K then V, every kv head: k~ = Pi k and v~ = Pi v, then quantise.
        // The key arrives post-RoPE and the value straight off its projection.
        for (uint32_t plane = 0; plane < 2 * numKvHeads_; ++plane) {
            AscendC::Cast(vec, in[plane * headSize_], AscendC::RoundMode::CAST_NONE, headSize_);
            AscendC::PipeBarrier<PIPE_V>();
            codec_.ApplyPi(vec, tmp, signs, static_cast<int>(headSize_));
            codec_.Quantize4Bit(packed[plane * packedBytes_], vec, steps[plane * kFp32PerBlock],
                                static_cast<int>(headSize_));
        }

        // Compact the stride-8 step slots into the contiguous scale slot, and
        // zero the burst padding so the cache never holds stale UB content.
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
            // Three aligned bursts: K bytes, V bytes, and the whole scale slot.
            // packedPlane_ and scaleSlot_ * 4 are both multiples of 32, and both
            // base offsets are whole multiples of those, so none of these three
            // transfers touches an unaligned global address.
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

/*
 * npu_turboquant_paged_attention, stage one.
 *
 * The sequence blocks of every (token, head) are split across AIV cores.  Each
 * core runs an online softmax over its slice, dequantising K and V on the fly,
 * and writes an un-normalised (max, sum, accumulator) triple to the workspace.
 */
template <typename scalar_t>
class TurboQuantPagedAttentionSplit {
public:
    __aicore__ inline explicit TurboQuantPagedAttentionSplit(AscendC::TPipe *pipe) : pipe_(pipe) {}

    /*
     * `queryRot` is the PRE-ROTATED query, fp32 [num_tokens, num_heads,
     * head_size], produced by npu_turboquant_rotate_q.  This kernel no longer
     * applies Pi to anything: `tables` is here for the codec's dequantiser
     * alone, and pi_signs has gone with the transform.
     */
    __aicore__ inline void Init(__gm__ void *queryRot, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *scaleCache, __gm__ void *blockTables, __gm__ void *contextLens,
                                __gm__ void *tables, __gm__ void *workspace,
                                uint32_t numTokens, uint32_t numHeads,
                                uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                uint32_t numSplits, float scale, float invSqrtLen)
    {
        numTokens_ = numTokens;
        numHeads_ = numHeads;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        blockSize_ = blockSize;
        maxBlocksPerSeq_ = maxBlocksPerSeq;
        numSplits_ = numSplits;
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

        const uint32_t tileElems = kTileRows * headSize_;
        pipe_->InitBuffer(kvQueue_, 2, 2 * kTileRows * packedBytes_ * sizeof(int8_t));
        pipe_->InitBuffer(scaleQueue_, 2, kTileRows * scaleSlot_ * sizeof(float));
        // fp32, not scalar_t: the query arrives rotated.  The queue stays --
        // its EnQue/DeQue is what orders the MTE2 fill against the vector reads
        // that tile it, and an explicit event would only restate that.
        pipe_->InitBuffer(qInQueue_, 1, headSize_ * sizeof(float));

        pipe_->InitBuffer(kvFloatBuf_, 2 * tileElems * sizeof(float));
        pipe_->InitBuffer(qTileBuf_, tileElems * sizeof(float));
        pipe_->InitBuffer(prodBuf_, tileElems * sizeof(float));
        // One head_size, not three.  The other two slices were the rotation's:
        // its in-place working vector and ApplyPi's ping-pong scratch.
        pipe_->InitBuffer(accBuf_, headSize_ * sizeof(float));
        pipe_->InitBuffer(rowBuf_, 6 * kTileRows * sizeof(float));
        pipe_->InitBuffer(brcbBuf_, 2 * kBrcbDstLanes * sizeof(float) + kTileRows * kFp32PerBlock * sizeof(float));
        // Five 32B blocks, one scalar each: every vector operand below is a
        // whole block, never a lane inside one.
        pipe_->InitBuffer(stateBuf_, 5 * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(scaleIdxBuf_, 2 * kTileRows * sizeof(int32_t));

        codec_.Init(pipe_, headSize_, kTileRows, invSqrtLen, tablesGm_);
        AscendC::PipeBarrier<PIPE_ALL>();
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

private:
    __aicore__ inline uint64_t PartialOffset(uint32_t token, uint32_t head, uint32_t split) const
    {
        return ((static_cast<uint64_t>(token) * numHeads_ + head) * numSplits_ + split) * partialStride_;
    }

    __aicore__ inline void ComputeSplit(uint32_t token, uint32_t head, uint32_t split)
    {
        // The first two blocks are laid out exactly as the workspace tail, so
        // the partial goes out with one DataCopy; the other three are scratch.
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

        const int32_t contextLen = contextLenGm_.GetValue(token);
        const uint32_t seqBlocks = contextLen > 0 ? CeilDiv(static_cast<uint32_t>(contextLen), blockSize_) : 0;
        const uint32_t blocksPerSplit = CeilDiv(seqBlocks, numSplits_);
        const uint32_t blockStart = split * blocksPerSplit;
        uint32_t blockEnd = blockStart + blocksPerSplit;
        if (blockEnd > seqBlocks) {
            blockEnd = seqBlocks;
        }

        if (blockStart < blockEnd) {
            const uint32_t kvHead = head / headsPerKv_;
            PrepareTask(token, head, kvHead);
            for (uint32_t block = blockStart; block < blockEnd; ++block) {
                // Addressing only: one physical block id per paged block.
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

                // Same prefetch discipline as the write path: the next tile's
                // MTE2 is issued before the current tile's vector work, so the
                // depth-2 queue actually overlaps. The pipeline drains at each
                // block boundary, which is one tile in blockSize / kTileRows.
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

        const uint64_t offset = PartialOffset(token, head, split);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(workspaceGm_[offset], acc, headSize_);
        AscendC::DataCopy(workspaceGm_[offset + headSize_], state, kPartialTail);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Per-task setup: read the rotated query once, tile it, and build the two
    // scale gather tables. kvHead is fixed for the task, so these are two
    // instructions per task rather than per tile.
    //
    // What used to sit between the read and the tiling was `q~ = Pi q`. It now
    // happens once per (token, head) in npu_turboquant_rotate_q instead of once
    // per (token, head, split) here; everything downstream -- the scores and the
    // value accumulator -- lives in the rotated basis, and so does the output:
    // the combine no longer un-rotates.
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
        const uint64_t cacheOff = row * packedPlane_ + static_cast<uint64_t>(kvHead) * packedBytes_;
        const uint64_t scaleOff = row * scaleSlot_;

        // One tile row per burst; consecutive rows are packedPlane_ apart. Every
        // length and stride here is a whole number of 32B blocks -- packedBytes_
        // is head_size / 2 and head_size is a power of two >= 64 -- so this is
        // plain DataCopy with no unaligned global address in sight.
        const AscendC::DataCopyParams rowsParams{static_cast<uint16_t>(kTileRows),
                                                 static_cast<uint16_t>(packedBytes_ / 32),
                                                 static_cast<uint16_t>((packedPlane_ - packedBytes_) / 32), 0};
        AscendC::DataCopy(kv, keyCacheGm_[cacheOff], rowsParams);
        AscendC::DataCopy(kv[kTileRows * packedBytes_], valueCacheGm_[cacheOff], rowsParams);

        // The whole scale slot for kTileRows tokens, contiguous and aligned; the
        // two lanes this task needs are picked out with a Gather in Accumulate.
        AscendC::DataCopy(scales, scaleCacheGm_[scaleOff], kTileRows * scaleSlot_);

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

        // Pick this task's K and V scale lanes out of the packed slots.
        AscendC::LocalTensor<uint32_t> idx = scaleIdxBuf_.Get<int32_t>().ReinterpretCast<uint32_t>();
        constexpr uint32_t scaleBase = kGatherSrcBase;
        AscendC::Gather(kScale, scaleTile, idx, scaleBase, kTileRows);
        AscendC::Gather(vScale, scaleTile, idx[kTileRows], scaleBase, kTileRows);
        AscendC::PipeBarrier<PIPE_V>();

        // Scores: dot(q_rot, k_levels) per row, then the per-row step.
        codec_.Dequantize4Bit(kf, kv, static_cast<int>(kTileRows), static_cast<int>(headSize_));
        AscendC::Mul(prod, kf, qTile, kTileRows * headSize_);
        AscendC::PipeBarrier<PIPE_V>();
        RowSums(scores, part, prod);
        AscendC::Mul(scores, scores, kScale, kTileRows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(scores, scores, scale_, kTileRows);
        AscendC::PipeBarrier<PIPE_V>();
        if (valid < kTileRows) {
            AscendC::Duplicate(scores[valid], kNegInf, kTileRows - valid);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Online softmax rescale.
        AscendC::ReduceMax<float>(tileMax, scores, reduceWork, kTileRows, false);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max(newMax, runMax, tileMax, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(alpha, runMax, newMax, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(alpha, alpha, 1);
        AscendC::PipeBarrier<PIPE_V>();

        BroadcastScalar(bMax, newMax);
        BroadcastSub(scores, scores, bMax, kTileRows);
        AscendC::Exp(probs, scores, kTileRows);
        AscendC::PipeBarrier<PIPE_V>();
        if (valid < kTileRows) {
            AscendC::Duplicate(probs[valid], 0.0f, kTileRows - valid);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::WholeReduceSum<float>(part, probs, kTileRows, 1, 1, 1, kTileRows / kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(runSum, runSum, alpha, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(runSum, runSum, part, 1);
        AscendC::PipeBarrier<PIPE_V>();

        BroadcastScalar(bAlpha, alpha);
        TurboQuantCodec4::BroadcastMul(acc, acc, bAlpha, headSize_);

        // Value accumulation: fold the per-row step into the probabilities so
        // the scale never has to be broadcast across head_size.
        AscendC::Mul(probs, probs, vScale, kTileRows);
        AscendC::PipeBarrier<PIPE_V>();
        codec_.Dequantize4Bit(vf, kv[kTileRows * packedBytes_], static_cast<int>(kTileRows),
                              static_cast<int>(headSize_));
        AscendC::Brcb(probBlocks, probs, kTileRows / kFp32PerBlock, {1, static_cast<uint16_t>(kFp32PerBlock)});
        AscendC::PipeBarrier<PIPE_V>();

        constexpr uint8_t kOneBlock = 1;
        const uint8_t rowBlocks = static_cast<uint8_t>(headSize_ / kFp32PerBlock);
        for (uint32_t col = 0; col < headSize_; col += kFp32PerRepeat) {
            AscendC::Mul(prod[col], vf[col], probBlocks, static_cast<uint64_t>(kFp32PerRepeat),
                         static_cast<uint8_t>(kTileRows), {1, 1, 0, rowBlocks, rowBlocks, kOneBlock});
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Column-wise sum of the tile by pairwise folding: log2(kTileRows) Adds.
        for (uint32_t half = kTileRows / 2; half >= 1; half /= 2) {
            AscendC::Add(prod, prod, prod[half * headSize_], half * headSize_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Add(acc, acc, prod, headSize_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Adds(runMax, newMax, 0.0f, 1);
        AscendC::PipeBarrier<PIPE_V>();

        kvQueue_.FreeTensor(kv);
        scaleQueue_.FreeTensor(scaleTile);
    }

    // Row-wise sum of a [kTileRows, head_size] tile; one WholeReduceSum per
    // 64-lane column chunk, no scalar accumulation.
    __aicore__ inline void RowSums(const AscendC::LocalTensor<float> &dst, const AscendC::LocalTensor<float> &part,
                                   const AscendC::LocalTensor<float> &src)
    {
        const uint8_t rowBlocks = static_cast<uint8_t>(headSize_ / kFp32PerBlock);
        AscendC::WholeReduceSum<float>(dst, src, kFp32PerRepeat, static_cast<uint8_t>(kTileRows), 1, 1, rowBlocks);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t col = kFp32PerRepeat; col < headSize_; col += kFp32PerRepeat) {
            AscendC::WholeReduceSum<float>(part, src[col], kFp32PerRepeat, static_cast<uint8_t>(kTileRows), 1, 1,
                                           rowBlocks);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(dst, dst, part, kTileRows);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    AscendC::TPipe *pipe_;
    TurboQuantCodec4 codec_;
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
    // fp32, not scalar_t: the query arrives pre-rotated.
    AscendC::GlobalTensor<float> queryRotGm_;
    AscendC::GlobalTensor<int8_t> keyCacheGm_;
    AscendC::GlobalTensor<int8_t> valueCacheGm_;
    AscendC::GlobalTensor<float> scaleCacheGm_;
    AscendC::GlobalTensor<int32_t> blockTableGm_;
    AscendC::GlobalTensor<int32_t> contextLenGm_;
    AscendC::GlobalTensor<int32_t> tablesGm_;
    AscendC::GlobalTensor<float> workspaceGm_;
    uint32_t numTokens_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlocksPerSeq_ = 0;
    uint32_t numSplits_ = 1;
    uint32_t headsPerKv_ = 1;
    uint32_t packedBytes_ = 0;
    uint32_t packedPlane_ = 0;
    uint32_t scaleSlot_ = 0;
    uint32_t partialStride_ = 0;
    float scale_ = 1.0f;
};

/*
 * npu_turboquant_paged_attention, stage two.
 *
 * Merges the per-split (max, sum, accumulator) triples and normalises. That is
 * all it does: the result is written in the ROTATED basis,
 * O~ = softmax(q k^T) V~, and nothing here takes it back out.
 *
 * The inverse rotation used to be the last thing this kernel did -- one ApplyPi
 * per (token, head), with the codec, its constant-table image and a sign buffer
 * held in UB for nothing else. It now lives in the weights: the host folds Pi
 * into the output projection, W_o <- W_o (I_H (x) Pi), so the GEMM that
 * consumes the output un-rotates it for free. A layer whose projection cannot
 * be folded -- an elementwise output gate sits between the attention and W_o --
 * is un-rotated by one more device launch, npu_turboquant_rotate_q on this
 * kernel's output; see vllm_ascend/attention/turboquant_v1.py.
 */
template <typename scalar_t>
class TurboQuantPagedAttentionCombine {
public:
    __aicore__ inline explicit TurboQuantPagedAttentionCombine(AscendC::TPipe *pipe) : pipe_(pipe) {}

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

        // Everything this kernel holds. Against the kernel that un-rotated, per
        // block: accBuf_ loses ApplyPi's ping-pong slice (4 * headSize bytes),
        // signBuf_ goes (4 * headSize), and so does the codec --
        // ConstTableWords(headSize, kTileRows) words of table image plus
        // WorkBufferWords(headSize, kTileRows) words of scratch, 82,272 B at
        // head_size 256. 84,320 B in all at 256 and 42,336 B at 128, pinned by
        // CombineUbFootprint in csrc/tests/common/turboquant_launch.hpp. The
        // GM -> UB copies of the table image and the signs leave the launch
        // prologue with them.
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
    __aicore__ inline uint64_t PartialOffset(uint32_t token, uint32_t head, uint32_t split) const
    {
        return ((static_cast<uint64_t>(token) * numHeads_ + head) * numSplits_ + split) * partialStride_;
    }

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
            const uint64_t offset = PartialOffset(token, head, split);
            // Write-after-read, and it has to be an all-pipe barrier: partAcc
            // and partState are plain TBuf views, so nothing else orders this
            // iteration's MTE2 fill against the previous iteration's vector
            // reads.  Only reachable with numSplits > 1.
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

        // Normalise by the softmax denominator, guarding an empty sequence.
        AscendC::Adds(runSum, runSum, TurboQuantCodec4::kEps, 1);
        AscendC::PipeBarrier<PIPE_V>();
        BroadcastScalar(bSum, runSum);
        AscendC::Duplicate(invSum, 1.0f, kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(invSum, invSum, bSum, kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        TurboQuantCodec4::BroadcastMul(acc, acc, invSum, headSize_);

        // O~, straight out. The folded output projection is the inverse
        // rotation now, so there is nothing left to do in this basis.
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

}  // namespace

/*
 * Kernel entry points take GM_ADDR, not `__gm__ void *`.  The CANN kernel-launch
 * code generator only recognises a parameter as global memory when the
 * declaration carries the cce_global attribute GM_ADDR expands to; a
 * `__gm__ void *` parameter is misclassified as a tiling struct and the
 * generated launcher fails to compile.  Nothing on the host side changes: the
 * generated aclrtlaunch_* wrapper still declares these parameters as `void *`.
 */
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

#define TURBOQUANT_PAGED_ATTENTION_SPLIT_DECLARE(TYPE)                                                               \
    extern "C" __global__ __aicore__ void turboquant_paged_attention_split_##TYPE(                                   \
        GM_ADDR queryRot, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR blockTables,             \
        GM_ADDR contextLens, GM_ADDR tables, GM_ADDR workspace, uint32_t numTokens,                                  \
        uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,     \
        uint32_t numSplits, uint32_t tasksPerCore, float scale, float invSqrtLen)                                    \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantPagedAttentionSplit<TYPE> op(&pipe);                                                               \
        op.Init(queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, tables, workspace,             \
                numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, scale,             \
                invSqrtLen);                                                                                         \
        op.Process(tasksPerCore);                                                                                    \
    }

#define TURBOQUANT_PAGED_ATTENTION_COMBINE_DECLARE(TYPE)                                                             \
    extern "C" __global__ __aicore__ void turboquant_paged_attention_combine_##TYPE(                                 \
        GM_ADDR workspace, GM_ADDR output, uint32_t numTokens, uint32_t numHeads, uint32_t headSize,                 \
        uint32_t numSplits, uint32_t tasksPerCore)                                                                   \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantPagedAttentionCombine<TYPE> op(&pipe);                                                             \
        op.Init(workspace, output, numTokens, numHeads, headSize, numSplits);                                        \
        op.Process(tasksPerCore);                                                                                    \
    }

TURBOQUANT_RESHAPE_AND_CACHE_DECLARE(half)
TURBOQUANT_PAGED_ATTENTION_SPLIT_DECLARE(half)
TURBOQUANT_PAGED_ATTENTION_COMBINE_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
TURBOQUANT_RESHAPE_AND_CACHE_DECLARE(bfloat16_t)
TURBOQUANT_PAGED_ATTENTION_SPLIT_DECLARE(bfloat16_t)
TURBOQUANT_PAGED_ATTENTION_COMBINE_DECLARE(bfloat16_t)
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

// Two launches on one stream.  The split kernel must be globally complete
// before the combine kernel reads the workspace, and stream order is the only
// barrier that holds for an arbitrary grid.
void turboquant_paged_attention_impl(AscendType type, void *stream, uint32_t splitBlockDim, uint32_t combineBlockDim,
                                     void *queryRot, void *keyCache, void *valueCache, void *scaleCache,
                                     void *blockTables, void *contextLens, void *tables, void *workspace,
                                     void *output, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                     uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                     uint32_t numSplits, uint32_t splitTasksPerCore, uint32_t combineTasksPerCore,
                                     float scale, float invSqrtLen)
{
    // Neither launch rotates anything: the split takes a pre-rotated query and
    // the combine writes the rotated output the folded W_o consumes. `tables`
    // is the split codec's dequantiser image and nothing else.
    if (type == AscendType::FP16) {
        turboquant_paged_attention_split_half<<<splitBlockDim, nullptr, stream>>>(
            queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, tables, workspace, numTokens,
            numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, splitTasksPerCore, scale,
            invSqrtLen);
        turboquant_paged_attention_combine_half<<<combineBlockDim, nullptr, stream>>>(
            workspace, output, numTokens, numHeads, headSize, numSplits, combineTasksPerCore);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    } else if (type == AscendType::BF16) {
        turboquant_paged_attention_split_bfloat16_t<<<splitBlockDim, nullptr, stream>>>(
            queryRot, keyCache, valueCache, scaleCache, blockTables, contextLens, tables, workspace, numTokens,
            numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, splitTasksPerCore, scale,
            invSqrtLen);
        turboquant_paged_attention_combine_bfloat16_t<<<combineBlockDim, nullptr, stream>>>(
            workspace, output, numTokens, numHeads, headSize, numSplits, combineTasksPerCore);
#endif
    }
}


/*
 * The combine stage on its own.
 *
 * The Cube path in turboquant_mm_kernels.cpp writes partials in exactly this
 * layout and needs this reduction unchanged -- it is rate-independent, and it
 * reads nothing but the workspace.
 */
void turboquant_paged_attention_combine_impl(AscendType type, void *stream, uint32_t blockDim, void *workspace,
                                             void *output, uint32_t numTokens, uint32_t numHeads, uint32_t headSize,
                                             uint32_t numSplits, uint32_t tasksPerCore)
{
    if (type == AscendType::FP16) {
        turboquant_paged_attention_combine_half<<<blockDim, nullptr, stream>>>(
            workspace, output, numTokens, numHeads, headSize, numSplits, tasksPerCore);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    } else if (type == AscendType::BF16) {
        turboquant_paged_attention_combine_bfloat16_t<<<blockDim, nullptr, stream>>>(
            workspace, output, numTokens, numHeads, headSize, numSplits, tasksPerCore);
#endif
    }
}

}  // namespace vllm_ascend
