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
 * TurboQuant 4-bit KV-cache operators.
 *
 * Layouts are plain contiguous ND -- no FRACTAL_NZ, no 5D views:
 *
 *   key / value        [num_tokens, num_kv_heads, head_size]      fp16 | bf16
 *   key/value cache    [num_blocks, block_size, num_kv_heads, head_size / 2]  int8
 *   key/value scale    [num_blocks, num_kv_heads, block_size]     fp32
 *   query              [num_tokens, num_heads, head_size]         fp16 | bf16
 *   output             [num_tokens, num_heads, head_size]         fp16 | bf16
 *
 * The scale planes are a separate allocation because the packed cache shape is
 * fixed by the backend contract at (2, num_blocks, block_size, num_kv_heads,
 * head_size / 2); at head_size 128 they add 6% on top of the 4-bit payload.
 *
 * Rotation is purely an activation-time transform.  Model weights are never
 * touched: K, V and Q arrive exactly as the projections and RoPE produced them,
 * the kernels apply Pi = D H D in UB, and the decode path applies the same Pi
 * once more to the accumulated output.  Because Pi is a symmetric involution
 * that single routine covers both directions, and RoPE stays correct because it
 * has already been applied by the time anything is rotated.
 */

#include "kernel_operator.h"

#include "../../kernels/types.h"
#include "turboquant_codec_950.h"

namespace {

using vllm_ascend::turboquant::kBrcbDstLanes;
using vllm_ascend::turboquant::kFp32PerBlock;
using vllm_ascend::turboquant::kFp32PerRepeat;
using vllm_ascend::turboquant::TurboQuantCodec4;

// Rows of the paged KV block processed per inner iteration.  Sized so that the
// working set (packed tiles, dequantised tiles, the tiled query and the two
// products) stays well inside UB for head_size up to 256, and so that Brcb can
// broadcast the whole tile of probabilities in whole repeats.
constexpr uint32_t kTileRows = 16;
// Softmax running maximum before any block has been seen.
constexpr float kNegInf = -1.0e30f;
// fp32 words of workspace per (token, head, split) partial: acc plus one padded
// block holding the running max and the running sum.
constexpr uint32_t kPartialTail = kFp32PerBlock;

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
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
 * One AIV core owns a contiguous run of tokens.  For each (token, kv head) the
 * key and value vectors are lifted to fp32, rotated by Pi in UB, quantised to
 * 4 bits and scattered straight into the ND paged cache -- there is no
 * intermediate staging buffer in global memory.
 */
template <typename scalar_t>
class TurboQuantReshapeAndCache {
public:
    __aicore__ inline explicit TurboQuantReshapeAndCache(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *key, __gm__ void *value, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *keyScale, __gm__ void *valueScale, __gm__ void *slotMapping,
                                __gm__ void *piSigns, uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,
                                uint32_t blockSize, uint32_t tokensPerCore, float invSqrtLen)
    {
        numTokens_ = numTokens;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        blockSize_ = blockSize;
        tokensPerCore_ = tokensPerCore;
        packedBytes_ = headSize / TurboQuantCodec4::kPackFactor;

        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(key));
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(value));
        keyCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(keyCache));
        valueCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(valueCache));
        keyScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(keyScale));
        valueScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(valueScale));
        slotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(slotMapping), numTokens);
        piSignsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(piSigns), headSize);

        // Double-buffered so the next (token, head) pair streams in while the
        // current one is being rotated and packed.
        pipe_->InitBuffer(inQueue_, 2, 2 * headSize_ * sizeof(scalar_t));
        pipe_->InitBuffer(outPacked_, 2, 2 * packedBytes_ * sizeof(int8_t));
        pipe_->InitBuffer(outScale_, 2, 2 * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(workBuf_, 2 * headSize_ * sizeof(float));
        pipe_->InitBuffer(signBuf_, headSize_ * sizeof(float));

        codec_.Init(pipe_, headSize_, 1, invSqrtLen);

        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::DataCopy(signs, piSignsGm_, headSize_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        const uint32_t start = static_cast<uint32_t>(AscendC::GetBlockIdx()) * tokensPerCore_;
        uint32_t end = start + tokensPerCore_;
        if (end > numTokens_) {
            end = numTokens_;
        }
        for (uint32_t token = start; token < end; ++token) {
            // Addressing only: a slot id per token, never a per-element read.
            const int32_t slot = slotGm_.GetValue(token);
            if (slot < 0) {
                continue;
            }
            const uint32_t slotId = static_cast<uint32_t>(slot);
            const uint32_t blockIdx = slotId / blockSize_;
            const uint32_t blockOff = slotId % blockSize_;
            for (uint32_t head = 0; head < numKvHeads_; ++head) {
                CopyIn(token, head);
                Compute();
                CopyOut(blockIdx, blockOff, head);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t token, uint32_t head)
    {
        AscendC::LocalTensor<scalar_t> in = inQueue_.template AllocTensor<scalar_t>();
        const uint64_t offset = (static_cast<uint64_t>(token) * numKvHeads_ + head) * headSize_;
        AscendC::DataCopy(in, keyGm_[offset], headSize_);
        AscendC::DataCopy(in[headSize_], valueGm_[offset], headSize_);
        inQueue_.EnQue(in);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<scalar_t> in = inQueue_.template DeQue<scalar_t>();
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::LocalTensor<int8_t> packed = outPacked_.template AllocTensor<int8_t>();
        AscendC::LocalTensor<float> steps = outScale_.template AllocTensor<float>();

        AscendC::LocalTensor<float> vec = work;
        AscendC::LocalTensor<float> tmp = work[headSize_];

        // k~ = Pi k, then quantise.  The key arrives post-RoPE and unmodified.
        AscendC::Cast(vec, in, AscendC::RoundMode::CAST_NONE, headSize_);
        AscendC::PipeBarrier<PIPE_V>();
        codec_.ApplyPi(vec, tmp, signs, static_cast<int>(headSize_));
        codec_.Quantize4Bit(packed, vec, steps, static_cast<int>(headSize_));

        // v~ = Pi v, from the raw value projection.
        AscendC::Cast(vec, in[headSize_], AscendC::RoundMode::CAST_NONE, headSize_);
        AscendC::PipeBarrier<PIPE_V>();
        codec_.ApplyPi(vec, tmp, signs, static_cast<int>(headSize_));
        codec_.Quantize4Bit(packed[packedBytes_], vec, steps[kFp32PerBlock], static_cast<int>(headSize_));

        inQueue_.FreeTensor(in);
        outPacked_.EnQue(packed);
        outScale_.EnQue(steps);
    }

    __aicore__ inline void CopyOut(uint32_t blockIdx, uint32_t blockOff, uint32_t head)
    {
        AscendC::LocalTensor<int8_t> packed = outPacked_.template DeQue<int8_t>();
        AscendC::LocalTensor<float> steps = outScale_.template DeQue<float>();

        const uint64_t cacheOff =
            ((static_cast<uint64_t>(blockIdx) * blockSize_ + blockOff) * numKvHeads_ + head) * packedBytes_;
        const uint64_t scaleOff = (static_cast<uint64_t>(blockIdx) * numKvHeads_ + head) * blockSize_ + blockOff;

        const AscendC::DataCopyExtParams payload{1, packedBytes_ * static_cast<uint32_t>(sizeof(int8_t)), 0, 0, 0};
        const AscendC::DataCopyExtParams scalar{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};

        AscendC::DataCopyPad(keyCacheGm_[cacheOff], packed, payload);
        AscendC::DataCopyPad(valueCacheGm_[cacheOff], packed[packedBytes_], payload);
        AscendC::DataCopyPad(keyScaleGm_[scaleOff], steps, scalar);
        AscendC::DataCopyPad(valueScaleGm_[scaleOff], steps[kFp32PerBlock], scalar);

        outPacked_.FreeTensor(packed);
        outScale_.FreeTensor(steps);
    }

    AscendC::TPipe *pipe_;
    TurboQuantCodec4 codec_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outPacked_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outScale_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> workBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> signBuf_;
    AscendC::GlobalTensor<scalar_t> keyGm_;
    AscendC::GlobalTensor<scalar_t> valueGm_;
    AscendC::GlobalTensor<int8_t> keyCacheGm_;
    AscendC::GlobalTensor<int8_t> valueCacheGm_;
    AscendC::GlobalTensor<float> keyScaleGm_;
    AscendC::GlobalTensor<float> valueScaleGm_;
    AscendC::GlobalTensor<int32_t> slotGm_;
    AscendC::GlobalTensor<float> piSignsGm_;
    uint32_t numTokens_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t tokensPerCore_ = 0;
    uint32_t packedBytes_ = 0;
};

/*
 * npu_turboquant_paged_attention
 *
 * Flash-decoding shape: stage one splits the *sequence blocks* of every
 * (token, head) across AIV cores -- so a batch of one still fills the device --
 * and each core runs an online softmax over its slice, dequantising K and V on
 * the fly in UB behind a double-buffered VECIN queue.  A cross-core barrier
 * then hands the partial (max, sum, accumulator) triples to stage two, which
 * rescales them, normalises, and undoes the rotation once per output vector.
 */
template <typename scalar_t>
class TurboQuantPagedAttention {
public:
    __aicore__ inline explicit TurboQuantPagedAttention(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *query, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *keyScale, __gm__ void *valueScale, __gm__ void *blockTables,
                                __gm__ void *contextLens, __gm__ void *piSigns, __gm__ void *workspace,
                                __gm__ void *output, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
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
        headsPerKv_ = numHeads / numKvHeads;
        packedBytes_ = headSize / TurboQuantCodec4::kPackFactor;
        partialStride_ = headSize + kPartialTail;

        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(query));
        keyCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(keyCache));
        valueCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(valueCache));
        keyScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(keyScale));
        valueScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(valueScale));
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTables));
        contextLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(contextLens), numTokens);
        piSignsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(piSigns), headSize);
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(output));

        const uint32_t tileElems = kTileRows * headSize_;
        pipe_->InitBuffer(kvQueue_, 2, 2 * kTileRows * packedBytes_ * sizeof(int8_t));
        pipe_->InitBuffer(scaleQueue_, 2, 2 * kTileRows * sizeof(float));
        pipe_->InitBuffer(qInQueue_, 1, headSize_ * sizeof(scalar_t));
        pipe_->InitBuffer(outQueue_, 1, headSize_ * sizeof(scalar_t));

        pipe_->InitBuffer(kvFloatBuf_, 2 * tileElems * sizeof(float));
        pipe_->InitBuffer(qTileBuf_, tileElems * sizeof(float));
        pipe_->InitBuffer(prodBuf_, tileElems * sizeof(float));
        pipe_->InitBuffer(accBuf_, 3 * headSize_ * sizeof(float));
        pipe_->InitBuffer(rowBuf_, 4 * kTileRows * sizeof(float));
        pipe_->InitBuffer(brcbBuf_, 3 * kBrcbDstLanes * sizeof(float) + kTileRows * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(stateBuf_, 4 * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(signBuf_, headSize_ * sizeof(float));

        codec_.Init(pipe_, headSize_, kTileRows, invSqrtLen);

        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::DataCopy(signs, piSignsGm_, headSize_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process(uint32_t splitTasksPerCore, uint32_t reduceTasksPerCore)
    {
        const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx());

        const uint32_t splitTasks = numTokens_ * numHeads_ * numSplits_;
        uint32_t start = core * splitTasksPerCore;
        uint32_t end = start + splitTasksPerCore;
        if (end > splitTasks) {
            end = splitTasks;
        }
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t split = task % numSplits_;
            const uint32_t head = (task / numSplits_) % numHeads_;
            const uint32_t token = task / (numSplits_ * numHeads_);
            ComputeSplit(token, head, split);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll();

        const uint32_t reduceTasks = numTokens_ * numHeads_;
        start = core * reduceTasksPerCore;
        end = start + reduceTasksPerCore;
        if (end > reduceTasks) {
            end = reduceTasks;
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

    // Stage one: online softmax over this core's slice of the sequence blocks.
    __aicore__ inline void ComputeSplit(uint32_t token, uint32_t head, uint32_t split)
    {
        AscendC::LocalTensor<float> state = stateBuf_.Get<float>();
        AscendC::LocalTensor<float> runMax = state;                    // [0]
        AscendC::LocalTensor<float> runSum = state[1];                 // [1]
        AscendC::LocalTensor<float> newMax = state[kFp32PerBlock];
        AscendC::LocalTensor<float> alpha = state[2 * kFp32PerBlock];
        AscendC::LocalTensor<float> tileMax = state[3 * kFp32PerBlock];

        AscendC::LocalTensor<float> acc = accBuf_.Get<float>();
        AscendC::LocalTensor<float> qRot = acc[headSize_];
        AscendC::LocalTensor<float> tmp = acc[2 * headSize_];

        AscendC::Duplicate(acc, 0.0f, headSize_);
        AscendC::Duplicate(state, 0.0f, 4 * kFp32PerBlock);
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
            LoadQuery(token, head, qRot, tmp);
            const uint32_t kvHead = head / headsPerKv_;
            for (uint32_t block = blockStart; block < blockEnd; ++block) {
                // Addressing only: one physical block id per paged block.
                const int32_t physical = blockTableGm_.GetValue(static_cast<uint64_t>(token) * maxBlocksPerSeq_ + block);
                if (physical < 0) {
                    continue;
                }
                uint32_t rows = blockSize_;
                const uint32_t consumed = block * blockSize_;
                if (consumed + rows > static_cast<uint32_t>(contextLen)) {
                    rows = static_cast<uint32_t>(contextLen) - consumed;
                }
                for (uint32_t base = 0; base < rows; base += kTileRows) {
                    uint32_t valid = rows - base;
                    if (valid > kTileRows) {
                        valid = kTileRows;
                    }
                    CopyInTile(static_cast<uint32_t>(physical), kvHead, base);
                    AccumulateTile(valid, acc, runMax, runSum, newMax, alpha, tileMax);
                }
            }
        }

        const uint64_t offset = PartialOffset(token, head, split);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(workspaceGm_[offset], acc, headSize_);
        AscendC::DataCopy(workspaceGm_[offset + headSize_], state, kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadQuery(uint32_t token, uint32_t head, const AscendC::LocalTensor<float> &qRot,
                                     const AscendC::LocalTensor<float> &tmp)
    {
        AscendC::LocalTensor<scalar_t> q = qInQueue_.template AllocTensor<scalar_t>();
        AscendC::DataCopy(q, queryGm_[(static_cast<uint64_t>(token) * numHeads_ + head) * headSize_], headSize_);
        qInQueue_.EnQue(q);
        q = qInQueue_.template DeQue<scalar_t>();
        AscendC::Cast(qRot, q, AscendC::RoundMode::CAST_NONE, headSize_);
        AscendC::PipeBarrier<PIPE_V>();
        qInQueue_.FreeTensor(q);

        // q~ = Pi q, once per (token, head, split).  Everything downstream --
        // scores and the value accumulator -- then lives in the rotated basis.
        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::LocalTensor<float> qRotRef = qRot;
        AscendC::LocalTensor<float> tmpRef = tmp;
        codec_.ApplyPi(qRotRef, tmpRef, signs, static_cast<int>(headSize_));

        // Tile the rotated query so the score product is one Mul per tile.
        AscendC::LocalTensor<float> qTile = qTileBuf_.Get<float>();
        for (uint32_t row = 0; row < kTileRows; ++row) {
            AscendC::DataCopy(qTile[row * headSize_], qRot, headSize_);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyInTile(uint32_t physical, uint32_t kvHead, uint32_t rowBase)
    {
        AscendC::LocalTensor<int8_t> kv = kvQueue_.template AllocTensor<int8_t>();
        AscendC::LocalTensor<float> scales = scaleQueue_.template AllocTensor<float>();

        const uint64_t cacheOff =
            ((static_cast<uint64_t>(physical) * blockSize_ + rowBase) * numKvHeads_ + kvHead) * packedBytes_;
        const uint64_t scaleOff = (static_cast<uint64_t>(physical) * numKvHeads_ + kvHead) * blockSize_ + rowBase;

        // One row of the tile per block; consecutive rows are num_kv_heads apart.
        const AscendC::DataCopyExtParams rowsParams{static_cast<uint16_t>(kTileRows), packedBytes_,
                                                    (numKvHeads_ - 1) * packedBytes_, 0, 0};
        const AscendC::DataCopyPadExtParams<int8_t> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(kv, keyCacheGm_[cacheOff], rowsParams, pad);
        AscendC::DataCopyPad(kv[kTileRows * packedBytes_], valueCacheGm_[cacheOff], rowsParams, pad);

        AscendC::DataCopy(scales, keyScaleGm_[scaleOff], kTileRows);
        AscendC::DataCopy(scales[kTileRows], valueScaleGm_[scaleOff], kTileRows);

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
        AscendC::LocalTensor<float> scales = scaleQueue_.template DeQue<float>();

        AscendC::LocalTensor<float> kf = kvFloatBuf_.Get<float>();
        AscendC::LocalTensor<float> vf = kf[kTileRows * headSize_];
        AscendC::LocalTensor<float> prod = prodBuf_.Get<float>();
        AscendC::LocalTensor<float> qTile = qTileBuf_.Get<float>();

        AscendC::LocalTensor<float> rows = rowBuf_.Get<float>();
        AscendC::LocalTensor<float> scores = rows;
        AscendC::LocalTensor<float> part = rows[kTileRows];
        AscendC::LocalTensor<float> probs = rows[2 * kTileRows];
        AscendC::LocalTensor<float> reduceWork = rows[3 * kTileRows];

        AscendC::LocalTensor<float> brcb = brcbBuf_.Get<float>();
        AscendC::LocalTensor<float> bMax = brcb;
        AscendC::LocalTensor<float> bAlpha = brcb[kBrcbDstLanes];
        AscendC::LocalTensor<float> probBlocks = brcb[3 * kBrcbDstLanes];

        // Scores: dot(q_rot, k_levels) per row, then the per-row step.
        codec_.Dequantize4Bit(kf, kv, static_cast<int>(kTileRows), static_cast<int>(headSize_));
        AscendC::Mul(prod, kf, qTile, kTileRows * headSize_);
        AscendC::PipeBarrier<PIPE_V>();
        RowSums(scores, part, prod);
        AscendC::Mul(scores, scores, scales, kTileRows);
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
        AscendC::Mul(probs, probs, scales[kTileRows], kTileRows);
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
        scaleQueue_.FreeTensor(scales);
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

    // Stage two: merge the per-split partials, normalise, undo the rotation.
    __aicore__ inline void Combine(uint32_t token, uint32_t head)
    {
        AscendC::LocalTensor<float> acc = accBuf_.Get<float>();
        AscendC::LocalTensor<float> partAcc = acc[headSize_];
        AscendC::LocalTensor<float> tmp = acc[2 * headSize_];

        AscendC::LocalTensor<float> state = stateBuf_.Get<float>();
        AscendC::LocalTensor<float> runMax = state;
        AscendC::LocalTensor<float> runSum = state[1];
        AscendC::LocalTensor<float> newMax = state[kFp32PerBlock];
        AscendC::LocalTensor<float> alpha = state[2 * kFp32PerBlock];
        AscendC::LocalTensor<float> beta = state[3 * kFp32PerBlock];

        AscendC::LocalTensor<float> brcb = brcbBuf_.Get<float>();
        AscendC::LocalTensor<float> bAlpha = brcb;
        AscendC::LocalTensor<float> bBeta = brcb[kBrcbDstLanes];
        AscendC::LocalTensor<float> bSum = brcb[2 * kBrcbDstLanes];
        // The probability-broadcast region is idle in the reduce stage, so it
        // doubles as scratch for the reciprocal of the softmax denominator.
        AscendC::LocalTensor<float> invSum = brcb[3 * kBrcbDstLanes];

        AscendC::LocalTensor<float> partState = rowBuf_.Get<float>();

        AscendC::Duplicate(acc, 0.0f, headSize_);
        AscendC::Duplicate(state, 0.0f, 4 * kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(runMax, kNegInf, 1);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t split = 0; split < numSplits_; ++split) {
            const uint64_t offset = PartialOffset(token, head, split);
            AscendC::DataCopy(partAcc, workspaceGm_[offset], headSize_);
            AscendC::DataCopy(partState, workspaceGm_[offset + headSize_], kFp32PerBlock);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::Max(newMax, runMax, partState, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(alpha, runMax, newMax, 1);
            AscendC::Sub(beta, partState, newMax, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(alpha, alpha, 1);
            AscendC::Exp(beta, beta, 1);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Mul(runSum, runSum, alpha, 1);
            AscendC::Mul(partState[1], partState[1], beta, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(runSum, runSum, partState[1], 1);
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

        // The accumulator is still in the rotated basis: sum_i p_i (Pi v_i) is
        // Pi (sum_i p_i v_i), so one inverse rotation per output vector suffices.
        // Out = Pi Out~.  Pi is an involution, so the same transform that put
        // K, V and Q into the rotated basis takes the accumulator back out.
        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::LocalTensor<float> accRef = acc;
        AscendC::LocalTensor<float> tmpRef = tmp;
        codec_.ApplyPi(accRef, tmpRef, signs, static_cast<int>(headSize_));

        AscendC::LocalTensor<scalar_t> out = outQueue_.template AllocTensor<scalar_t>();
        AscendC::Cast(out, acc, AscendC::RoundMode::CAST_RINT, headSize_);
        AscendC::PipeBarrier<PIPE_V>();
        outQueue_.EnQue(out);
        out = outQueue_.template DeQue<scalar_t>();
        AscendC::DataCopy(outputGm_[(static_cast<uint64_t>(token) * numHeads_ + head) * headSize_], out, headSize_);
        outQueue_.FreeTensor(out);
    }

    AscendC::TPipe *pipe_;
    TurboQuantCodec4 codec_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> kvQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> scaleQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qInQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> kvFloatBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> qTileBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> prodBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rowBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> brcbBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stateBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> signBuf_;
    AscendC::GlobalTensor<scalar_t> queryGm_;
    AscendC::GlobalTensor<int8_t> keyCacheGm_;
    AscendC::GlobalTensor<int8_t> valueCacheGm_;
    AscendC::GlobalTensor<float> keyScaleGm_;
    AscendC::GlobalTensor<float> valueScaleGm_;
    AscendC::GlobalTensor<int32_t> blockTableGm_;
    AscendC::GlobalTensor<int32_t> contextLenGm_;
    AscendC::GlobalTensor<float> piSignsGm_;
    AscendC::GlobalTensor<float> workspaceGm_;
    AscendC::GlobalTensor<scalar_t> outputGm_;
    uint32_t numTokens_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlocksPerSeq_ = 0;
    uint32_t numSplits_ = 1;
    uint32_t headsPerKv_ = 1;
    uint32_t packedBytes_ = 0;
    uint32_t partialStride_ = 0;
    float scale_ = 1.0f;
};

}  // namespace

#define TURBOQUANT_RESHAPE_AND_CACHE_DECLARE(TYPE)                                                                    \
    extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_##TYPE(                                        \
        __gm__ void *key, __gm__ void *value, __gm__ void *keyCache, __gm__ void *valueCache, __gm__ void *keyScale,  \
        __gm__ void *valueScale, __gm__ void *slotMapping, __gm__ void *piSigns, uint32_t numTokens,                  \
        uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t tokensPerCore, float invSqrtLen)         \
    {                                                                                                                 \
        AscendC::TPipe pipe;                                                                                          \
        TurboQuantReshapeAndCache<TYPE> op(&pipe);                                                                    \
        op.Init(key, value, keyCache, valueCache, keyScale, valueScale, slotMapping, piSigns, numTokens, numKvHeads,   \
                headSize, blockSize, tokensPerCore, invSqrtLen);                                                      \
        op.Process();                                                                                                 \
    }

#define TURBOQUANT_PAGED_ATTENTION_DECLARE(TYPE)                                                                      \
    extern "C" __global__ __aicore__ void turboquant_paged_attention_##TYPE(                                          \
        __gm__ void *query, __gm__ void *keyCache, __gm__ void *valueCache, __gm__ void *keyScale,                    \
        __gm__ void *valueScale, __gm__ void *blockTables, __gm__ void *contextLens, __gm__ void *piSigns,            \
        __gm__ void *workspace, __gm__ void *output, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,      \
        uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,                          \
        uint32_t splitTasksPerCore, uint32_t reduceTasksPerCore, float scale, float invSqrtLen)                       \
    {                                                                                                                 \
        AscendC::TPipe pipe;                                                                                          \
        TurboQuantPagedAttention<TYPE> op(&pipe);                                                                     \
        op.Init(query, keyCache, valueCache, keyScale, valueScale, blockTables, contextLens, piSigns, workspace,      \
                output, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, scale,      \
                invSqrtLen);                                                                                          \
        op.Process(splitTasksPerCore, reduceTasksPerCore);                                                            \
    }

TURBOQUANT_RESHAPE_AND_CACHE_DECLARE(half)
TURBOQUANT_PAGED_ATTENTION_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
TURBOQUANT_RESHAPE_AND_CACHE_DECLARE(bfloat16_t)
TURBOQUANT_PAGED_ATTENTION_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {

void turboquant_reshape_and_cache_impl(AscendType type, void *stream, uint32_t blockDim, void *key, void *value,
                                       void *keyCache, void *valueCache, void *keyScale, void *valueScale,
                                       void *slotMapping, void *piSigns, uint32_t numTokens, uint32_t numKvHeads,
                                       uint32_t headSize, uint32_t blockSize, uint32_t tokensPerCore, float invSqrtLen)
{
    if (type == AscendType::FP16) {
        turboquant_reshape_and_cache_half<<<blockDim, nullptr, stream>>>(key, value, keyCache, valueCache, keyScale,
                                                                         valueScale, slotMapping, piSigns, numTokens,
                                                                         numKvHeads, headSize, blockSize,
                                                                         tokensPerCore, invSqrtLen);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    } else if (type == AscendType::BF16) {
        turboquant_reshape_and_cache_bfloat16_t<<<blockDim, nullptr, stream>>>(
            key, value, keyCache, valueCache, keyScale, valueScale, slotMapping, piSigns, numTokens, numKvHeads,
            headSize, blockSize, tokensPerCore, invSqrtLen);
#endif
    }
}

void turboquant_paged_attention_impl(AscendType type, void *stream, uint32_t blockDim, void *query, void *keyCache,
                                     void *valueCache, void *keyScale, void *valueScale, void *blockTables,
                                     void *contextLens, void *piSigns, void *workspace, void *output,
                                     uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize,
                                     uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits,
                                     uint32_t splitTasksPerCore, uint32_t reduceTasksPerCore, float scale,
                                     float invSqrtLen)
{
    if (type == AscendType::FP16) {
        turboquant_paged_attention_half<<<blockDim, nullptr, stream>>>(
            query, keyCache, valueCache, keyScale, valueScale, blockTables, contextLens, piSigns, workspace, output,
            numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, splitTasksPerCore,
            reduceTasksPerCore, scale, invSqrtLen);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    } else if (type == AscendType::BF16) {
        turboquant_paged_attention_bfloat16_t<<<blockDim, nullptr, stream>>>(
            query, keyCache, valueCache, keyScale, valueScale, blockTables, contextLens, piSigns, workspace, output,
            numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, splitTasksPerCore,
            reduceTasksPerCore, scale, invSqrtLen);
#endif
    }
}

}  // namespace vllm_ascend
