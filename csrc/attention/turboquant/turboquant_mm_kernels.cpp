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
 * npu_turboquant_mm_* -- the Cube-native TurboQuant decode for Ascend 950PR.
 *
 * TOPOLOGY   1 AIC : 2 AIV per MIX block. Both vector subcores run the same
 *            task and stage identical bytes; only subcore 0 consumes the Cube's
 *            product. See TURBOQUANT_TESTS.md 13.14.1-13.14.3.
 *
 * DATA PATH  HBM -> UB -> L1 (ping-pong, kSlots deep) -> L0A/L0B -> L0C -> UB
 *            -> GM.  The AIV owns the query rotation, the K/V unpack onto the
 *            operand grid, the online softmax and the accumulator; the AIC owns
 *            both GEMMs.
 *
 * SCHEDULE   One task is (token, kvHead, split), producing one flash-decoding
 *            partial per query head in the group. Within a task the K/V staging
 *            runs one tile ahead of the GEMMs that consume it: the Cube reads
 *            L1[tile % kSlots] while the AIV fills L1[(tile + 1) % kSlots].
 *            See TURBOQUANT_TESTS.md 13.11.
 */

#include "kernel_operator.h"
#include "turboquant_codec_mx.h"
#include "turboquant_cube_mm.h"

#include "../../kernels/types.h"

using vllm_ascend::turboquant::kBrcbDstLanes;
using vllm_ascend::turboquant::kFp32PerBlock;
using vllm_ascend::turboquant::kFp32PerRepeat;
using vllm_ascend::turboquant::kSlots;
using vllm_ascend::turboquant::TurboQuantCodec4;
using vllm_ascend::turboquant::TurboQuantCubeMm;
using vllm_ascend::turboquant::TurboQuantMode;
using vllm_ascend::turboquant::TurboQuantModeCodec;
using vllm_ascend::turboquant::TurboQuantModeTraits;

namespace {

// Rows of the packed cache one Cube tile covers.  Chosen so the context GEMM's
// reduction length is a multiple of the 32-element C0 an 8-bit Cube operand
// fractal uses, and so it divides block_size (128).
constexpr uint32_t kCubeTileRows = 64;

// Rows the codec expands in one Unpack call, on the CODEBOOK path (kv3fp4,
// kv5fp8).  Not the same as kCubeTileRows: the codec's work buffers scale with
// rows * head_size, so 8 rows costs ~42 KB of UB against ~141 KB for 64.
// Sub-batches are placed into the tile's NZ buffer by a strided DataCopy, so
// the same offset table serves every one.
constexpr uint32_t kUnpackRows = 8;

// C0 column-groups of the packed plane the AFFINE path expands per call.
//
// The affine unpack is byte-major, not row-major: it walks the packed tile one
// 32-byte column-group at a time across all kCubeTileRows rows, because that is
// the run whose expansion is exactly one NZ fractal column of the L1 operand.
// See UnpackToL1 for why that makes both the GM -> UB read and the UB -> L1
// write flat, and CopyInTile for the transposed tile load that sets it up.
//
// Two groups is 4 KB of packed bytes per call at head_size 256, which the codec
// carries in two 16 KB fp32 buffers.  Counting the codec's whole footprint --
// work buffer, constant tables and the unpacked operand -- that is 51.3 KB of UB
// against the codebook path's 60.4 KB, because the affine mode needs no table
// image at all; so this is a 9 KB saving and not a spend.  Four groups would
// halve the call count again and add 32 KB, which was not worth the headroom.
constexpr uint32_t kUnpackGroups = 2;

// Elements of a Cube operand in one C0 block.
constexpr uint32_t kOperandC0 = 32;

// Query heads one task may batch into the GEMM's M dimension.  A model with a
// larger GQA group than this is split across tasks by the host's grid, which is
// why PlanCubeDecode takes headsPerKv rather than assuming it.
constexpr uint32_t kMaxGroupHeads = vllm_ascend::turboquant::kCubeTileM;

// fp32 words appended to a partial: one 32B block for the running max, a second
// for the running sum.  Mirrors turboquant_adpt::kPartialTail and the value in
// turboquant_kernels.cpp; the shared combine reads this layout.
constexpr uint32_t kPartialTail = 16;
constexpr uint32_t kPartialMaxLane = 0;
constexpr uint32_t kPartialSumLane = kFp32PerBlock;

constexpr float kNegInf = -3.4028235e38f;

/*
 * The block index an AIC and its paired AIV subcores agree on.
 *
 * GetBlockIdx() does not mean the same thing on the two halves of a MIX kernel;
 * keying a task loop on it deadlocks any kernel whose tiles carry a cross-core
 * handshake. Not for a pure-vector kernel -- see TurboQuantPlainCombine, which
 * keeps GetBlockIdx(). See TURBOQUANT_TESTS.md 13.14.3.
 */
__aicore__ inline uint32_t MixBlockIdx()
{
    return static_cast<uint32_t>(AscendC::GetBlockIdx() / AscendC::GetSubBlockNum());
}

/*
 * Operand-ready, vector half -> cube half. Issued by BOTH subcores: a mode 0x02
 * flag is satisfied only once every subcore of the pair has set it, so gating
 * this starves the AIC. See TURBOQUANT_TESTS.md 13.14.2.
 */
__aicore__ inline void SignalOperandsReady()
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(vllm_ascend::turboquant::kFlagOperandsReady);
}

/*
 * The pipelined decode's two AIV -> AIC edges. Both ungated, including
 * SignalProbsReady, whose payload only subcore 0 produces: mode 0x02 holds the
 * AIC until that subcore has signalled too. See TURBOQUANT_TESTS.md 13.14.2.
 */
__aicore__ inline void SignalSlotReady(uint32_t slot)
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
        static_cast<uint16_t>(vllm_ascend::turboquant::kFlagSlotReady + slot));
}

__aicore__ inline void SignalProbsReady()
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(vllm_ascend::turboquant::kFlagProbsReady);
}

/*
 * True on the one vector subcore whose UB the Cube's Fixpipe landed in. Gates
 * what CONSUMES a product; never the flag protocol, which stays ungated.
 * See TURBOQUANT_TESTS.md 13.14.1.
 */
__aicore__ inline bool IsPrimarySubcore()
{
    return AscendC::GetSubBlockIdx() == 0;
}

/*
 * Vector -> MTE3, before any DMA of a UB buffer the vector unit just wrote.
 * Sets and immediately waits, so it is correct at a blocking call site and wrong
 * inside a software pipeline. See TURBOQUANT_TESTS.md 13.14.4.
 */
__aicore__ inline void SyncVectorToMte3()
{
    const event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev);
}

// MTE2 -> vector, before reading a UB buffer a GM copy just filled. A
// PipeBarrier<PIPE_MTE2> would NOT do: it orders MTE2 against MTE2 and says
// nothing about the vector pipe. See TURBOQUANT_TESTS.md 13.14.4.
__aicore__ inline void SyncMte2ToVector()
{
    const event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev);
}

// MTE3 -> vector: the write-after-read edge, for a UB buffer the vector unit is
// about to overwrite while a DMA may still be draining it. The opposite
// direction to SyncVectorToMte3, and not covered by it.
__aicore__ inline void SyncMte3ToVector()
{
    const event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ev);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ev);
}

// e4m3fn's largest finite value, and e2m1's.  A query row and a probability row
// are scaled to the top of the operand grid before the cast so the mantissa is
// not thrown away against the grid's subnormal floor, and the fp32 multiplier
// comes back out of the product.
template <TurboQuantMode MODE>
__aicore__ inline constexpr float OperandMax()
{
    return TurboQuantModeTraits<MODE>::kOperand == vllm_ascend::turboquant::TurboQuantOperand::kFp4E2m1 ? 6.0f
                                                                                                       : 448.0f;
}

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// fp32 words one token occupies in the scale plane.  Mirrors ScaleSlotFloats in
// turboquant_kernels.cpp; the two caches share this plane unchanged.
__aicore__ inline uint32_t ScaleSlotFloats(uint32_t numKvHeads)
{
    return CeilDiv(2 * numKvHeads, kFp32PerBlock) * kFp32PerBlock;
}

// dst[i] = src[i] - scalarBlock[i % 8], the subtractive twin of
// TurboQuantCodec4::BroadcastMul.
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

// Splay one fp32 lane across a whole 32B block.
__aicore__ inline void BroadcastScalar(const AscendC::LocalTensor<float> &dst, const AscendC::LocalTensor<float> &src)
{
    AscendC::Brcb(dst, src, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
    AscendC::PipeBarrier<PIPE_V>();
}

/*
 * npu_turboquant_mm_reshape_and_cache
 *
 * One AIV core owns a run of tokens and encodes a whole token -- every kv head,
 * key and value -- per pipeline stage, so the scatter is three aligned bursts.
 * Three-stage software pipeline, as in turboquant_kernels.cpp.
 */
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
        // One 32B-aligned landing slot per scale: Encode broadcasts its result
        // with Brcb, which needs an aligned base and eight readable lanes.
        pipe_->InitBuffer(stepBuf_, 2 * numKvHeads_ * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(scaleIdxBuf_, 2 * numKvHeads_ * sizeof(int32_t));

        // Two codecs: the shipping 4-bit one carries the rotation and nothing
        // else here, and the mode codec carries the rate.  Both at batchRows 1,
        // because the write path handles one vector at a time.
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

        // Compact the 32B-spaced scale slots into the token's contiguous slot,
        // and zero the burst padding so the cache never holds stale UB content.
        // The zeroing covers the whole slot and runs first, because a vector
        // operand base must be 32-byte aligned and the padding does not start
        // on one.
        AscendC::LocalTensor<uint32_t> idx = scaleIdxBuf_.Get<int32_t>().ReinterpretCast<uint32_t>();
        AscendC::Duplicate(scaleOut, 0.0f, scaleSlot_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(scaleOut, steps, idx, vllm_ascend::turboquant::kGatherSrcBase, 2 * numKvHeads_);
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

/*
 * npu_turboquant_mm_decode_split
 *
 * MIX.  One task is (token, kvHead, split) and it produces one flash-decoding
 * partial per query head in that kv head's group.  The AIV owns the query
 * rotation and its fp8 cast, the unpack of every K/V tile onto the operand
 * grid, the online softmax and the accumulator; the AIC owns both GEMMs.
 */
template <TurboQuantMode MODE, typename scalar_t>
class TurboQuantCubeDecodeSplit {
public:
    using Codec = TurboQuantModeCodec<MODE>;
    using Mm = TurboQuantCubeMm<MODE>;
    using OperandT = typename Mm::OperandT;

    __aicore__ inline explicit TurboQuantCubeDecodeSplit(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *query, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *scaleCache, __gm__ void *blockTables, __gm__ void *contextLens,
                                __gm__ void *piSigns, __gm__ void *rotTables, __gm__ void *modeTables,
                                __gm__ void *workspace, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
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
        groupHeads_ = numHeads / numKvHeads;
        packedBytes_ = Codec::PackedBytes(headSize);
        packedPlane_ = numKvHeads_ * packedBytes_;
        scaleSlot_ = ScaleSlotFloats(numKvHeads_);
        partialStride_ = headSize + kPartialTail;
        operandElems_ = Mm::OperandElems(headSize_);
        // C0 column-groups in one vector slot's packed plane, and the run one
        // affine Unpack call consumes.  Computed for every mode and read only by
        // the affine one, which walks groups where a codebook mode walks rows;
        // see kUnpackGroups.
        //
        // The chunk has to DIVIDE the group count, or the last chunk reads past
        // the tile.  head_size is a power of two in [64, 256] so packedGroups_
        // is 1, 2 or 4 and the min alone would do -- the loop is here because
        // nothing in this file enforces that, and an over-read of the packed
        // tile is exactly the class of defect that shows up as an inf/nan flood
        // rather than as a fault.
        packedGroups_ = packedBytes_ / kOperandC0;
        unpackGroups_ = packedGroups_ < kUnpackGroups ? packedGroups_ : kUnpackGroups;
        while (unpackGroups_ > 1 && (packedGroups_ % unpackGroups_) != 0) {
            --unpackGroups_;
        }
        unpackBytes_ = unpackGroups_ * kCubeTileRows * kOperandC0;

        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(query));
        keyCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(keyCache));
        valueCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(valueCache));
        scaleCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scaleCache));
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTables));
        contextLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(contextLens), numTokens);
        piSignsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(piSigns), headSize);
        rotTablesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rotTables));
        modeTablesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(modeTables));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

        mm_.Init(pipe_, headSize_, kCubeTileRows);

        // The packed tile: kCubeTileRows rows of K and of V for one kv head.
        pipe_->InitBuffer(kvBuf_, 2 * kCubeTileRows * packedBytes_);
        pipe_->InitBuffer(scaleTileBuf_, kCubeTileRows * scaleSlot_ * sizeof(float));
        // The unpacked sub-batch.  The codebook path lands kUnpackRows rows in NZ
        // order within their own band; the affine path lands one chunk's two
        // nibble planes end to end, each already a run of whole NZ fractal
        // columns.
        if constexpr (Codec::kIsAffine) {
            pipe_->InitBuffer(operandBuf_, 2 * unpackBytes_);
        } else {
            pipe_->InitBuffer(operandBuf_, kUnpackRows * operandElems_);
        }
        // [kCubeTileM, head_size] fp32 accumulator, plus the query in fp32 and
        // one scratch vector for the rotation.
        pipe_->InitBuffer(accBuf_, kMaxGroupHeads * headSize_ * sizeof(float));
        pipe_->InitBuffer(qBuf_, 2 * headSize_ * sizeof(float));
        pipe_->InitBuffer(qOperandBuf_, kMaxGroupHeads * operandElems_);
        // [kCubeTileM, kCubeTileRows] fp32 scores, and the same shape again for
        // the probabilities before they are cast onto the operand grid.
        pipe_->InitBuffer(scoreBuf_, kMaxGroupHeads * kCubeTileRows * sizeof(float));
        pipe_->InitBuffer(probOperandBuf_, kMaxGroupHeads * Mm::OperandElems(kCubeTileRows));
        // [kCubeTileM, head_size] fp32 context product from the Cube.
        pipe_->InitBuffer(ctxBuf_, kMaxGroupHeads * headSize_ * sizeof(float));
        // Per-head softmax state: running max, running sum, tile max, new max,
        // alpha, and the probability row's operand scale. One 32B block each.
        pipe_->InitBuffer(stateBuf_, 6 * kMaxGroupHeads * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, 4 * kMaxGroupHeads * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(signBuf_, headSize_ * sizeof(float));
        pipe_->InitBuffer(scaleIdxBuf_, 2 * kCubeTileRows * sizeof(int32_t));

        rotation_.Init(pipe_, headSize_, 1, invSqrtLen, rotTablesGm_);
        // The affine codec reads headSize * batchRows as the element count one
        // call produces -- twice the bytes it consumes -- so the chunk size is
        // expressed in the one unit both paths share.  See TurboQuantModeCodec.
        codec_.Init(pipe_, headSize_,
                    Codec::kIsAffine ? (2 * unpackBytes_ / headSize_) : kUnpackRows, invSqrtLen, modeTablesGm_);

        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::DataCopy(signs, piSignsGm_, headSize_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process(uint32_t tasksPerCore)
    {
        const uint32_t tasks = numTokens_ * numKvHeads_ * numSplits_;
        // MixBlockIdx(), not GetBlockIdx(): every tile below carries an
        // AIC <-> AIV handshake, so the two halves of a MIX block have to walk
        // the same task list or they deadlock against each other. See
        // MixBlockIdx.
        uint32_t start = MixBlockIdx() * tasksPerCore;
        uint32_t end = start + tasksPerCore;
        if (end > tasks) {
            end = tasks;
        }
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t split = task % numSplits_;
            const uint32_t rest = task / numSplits_;
            ComputeSplit(rest / numKvHeads_, rest % numKvHeads_, split);
        }
    }

private:
    __aicore__ inline uint64_t PartialOffset(uint32_t token, uint32_t head, uint32_t split) const
    {
        return ((static_cast<uint64_t>(token) * numHeads_ + head) * numSplits_ + split) * partialStride_;
    }

    __aicore__ inline void ComputeSplit(uint32_t token, uint32_t kvHead, uint32_t split)
    {
        AscendC::LocalTensor<float> acc = accBuf_.Get<float>();
        AscendC::LocalTensor<float> state = stateBuf_.Get<float>();
        AscendC::LocalTensor<float> runMax = state;
        AscendC::LocalTensor<float> runSum = state[kMaxGroupHeads * kFp32PerBlock];

        // Vector work, so it is the AIV's alone. Both cores execute this
        // function on a MIX kernel; leaving the accumulator init unguarded made
        // the cube core issue vector stores into UB it does not own, which the
        // part reports as su_ccu_mpu_err rather than as anything legible.
        if ASCEND_IS_AIV {
            AscendC::Duplicate(acc, 0.0f, kMaxGroupHeads * headSize_);
            AscendC::Duplicate(state, 0.0f, 6 * kMaxGroupHeads * kFp32PerBlock);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t h = 0; h < groupHeads_; ++h) {
                AscendC::Duplicate(runMax[h * kFp32PerBlock], kNegInf, 1);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        const int32_t contextLen = contextLenGm_.GetValue(token);
        const uint32_t seqBlocks = contextLen > 0 ? CeilDiv(static_cast<uint32_t>(contextLen), blockSize_) : 0;
        const uint32_t blocksPerSplit = CeilDiv(seqBlocks, numSplits_);
        const uint32_t blockStart = split * blocksPerSplit;
        uint32_t blockEnd = blockStart + blocksPerSplit;
        if (blockEnd > seqBlocks) {
            blockEnd = seqBlocks;
        }

        if (blockStart < blockEnd) {
            // Both halves walk the SAME tile enumeration, independently.  It is
            // a pure function of the block table and the context length, both
            // of which are scalar GM reads either core can make, so the two
            // counts cannot diverge -- and they must not: every flag condition
            // below is a predicate on the tile index and numTiles.
            const uint32_t ctxLen = static_cast<uint32_t>(contextLen);
            const uint32_t numTiles = CountTiles(token, ctxLen, blockStart, blockEnd);
            if ASCEND_IS_AIV {
                PrepareTask(token, kvHead);
                PipelineAiv(token, ctxLen, blockStart, blockEnd, numTiles, kvHead, acc, state);
            }
            if ASCEND_IS_AIC {
                PipelineAic(numTiles);
            }
        }

        // One partial per query head: [head_size] of accumulator then
        // kPartialTail of state, max at lane 0 and sum at lane kFp32PerBlock.
        // Gated because only the Fixpipe's subcore holds this task's
        // accumulator; ASCEND_IS_AIV is a constexpr(...) so the two conditions
        // have to nest rather than &&. See TURBOQUANT_TESTS.md 13.14.1.
        if ASCEND_IS_AIV {
          if (IsPrimarySubcore()) {
            // A tail per head, carved out of the finished score buffer. No two
            // iterations share a landing slot, so the write-after-read the
            // per-iteration barrier used to cover cannot arise, and the whole
            // writeback needs one V -> MTE3 edge instead of one per head.
            AscendC::LocalTensor<float> tails = scoreBuf_.Get<float>();
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t h = 0; h < groupHeads_; ++h) {
                AscendC::LocalTensor<float> tail = tails[h * kPartialTail];
                AscendC::Duplicate(tail, 0.0f, kPartialTail);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(tail[kPartialMaxLane], runMax[h * kFp32PerBlock], 0.0f, 1);
                AscendC::Adds(tail[kPartialSumLane], runSum[h * kFp32PerBlock], 0.0f, 1);
            }
            SyncVectorToMte3();
            for (uint32_t h = 0; h < groupHeads_; ++h) {
                const uint32_t head = kvHead * groupHeads_ + h;
                const uint64_t offset = PartialOffset(token, head, split);
                AscendC::DataCopy(workspaceGm_[offset], acc[h * headSize_], headSize_);
                AscendC::DataCopy(workspaceGm_[offset + headSize_], tails[h * kPartialTail], kPartialTail);
            }
          }
        }
    }

    // Once per task: rotate every query head in the group, scale it to the top
    // of the operand grid, cast, and place it in L1 in NZ order -- eight
    // 32-byte runs per head at a stride of kCubeTileM * 32 bytes.
    __aicore__ inline void PrepareTask(uint32_t token, uint32_t kvHead)
    {
        AscendC::LocalTensor<float> work = qBuf_.Get<float>();
        AscendC::LocalTensor<float> vec = work;
        AscendC::LocalTensor<float> tmp = work[headSize_];
        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::LocalTensor<OperandT> qOperand = qOperandBuf_.Get<OperandT>();
        AscendC::LocalTensor<float> reduce = reduceBuf_.Get<float>();
        AscendC::LocalTensor<OperandT> qL1 = mm_.A1Query();

        const uint32_t c0Blocks = headSize_ / kOperandC0;
        const AscendC::DataCopyParams nzParams{static_cast<uint16_t>(c0Blocks), 1, 0,
                                               static_cast<uint16_t>(kMaxGroupHeads - 1)};

        AscendC::Duplicate(qOperand.template ReinterpretCast<int8_t>(), static_cast<int8_t>(0),
                           kMaxGroupHeads * operandElems_);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t h = 0; h < groupHeads_; ++h) {
            const uint32_t head = kvHead * groupHeads_ + h;
            AscendC::LocalTensor<scalar_t> q = qInQueue_.template AllocTensor<scalar_t>();
            AscendC::DataCopy(q, queryGm_[(static_cast<uint64_t>(token) * numHeads_ + head) * headSize_], headSize_);
            qInQueue_.EnQue(q);
            q = qInQueue_.template DeQue<scalar_t>();
            AscendC::Cast(vec, q, AscendC::RoundMode::CAST_NONE, headSize_);
            AscendC::PipeBarrier<PIPE_V>();
            qInQueue_.FreeTensor(q);

            // q~ = Pi q.  Everything downstream lives in the rotated basis.
            rotation_.ApplyPi(vec, tmp, signs, static_cast<int>(headSize_));

            // Scale into the operand grid.  ReduceMax over |q| leaves the amax
            // in UB; the reciprocal is applied without a scalar round trip, and
            // the factor comes back out of the score row in AccumulateTile.
            AscendC::Abs(tmp, vec, headSize_);
            AscendC::PipeBarrier<PIPE_V>();
            // Work buffer distinct from the source. ReduceMax writes its
            // intermediates into sharedTmpBuffer while still reading src, so
            // passing one tensor as both can corrupt the reduction itself --
            // and this result is qScale_.
            AscendC::ReduceMax<float>(reduce, tmp, scoreBuf_.Get<float>(), headSize_, false);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(reduce, reduce, TurboQuantCodec4::kEps, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(reduce[kFp32PerBlock], OperandMax<MODE>(), 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(reduce[kFp32PerBlock], reduce[kFp32PerBlock], reduce, 1);
            AscendC::PipeBarrier<PIPE_V>();
            // qScale_[h] is the multiplier that was applied; the score row is
            // divided by it, so it never reaches the model.
            BroadcastScalar(reduce[2 * kFp32PerBlock], reduce[kFp32PerBlock]);
            TurboQuantCodec4::BroadcastMul(vec, vec, reduce[2 * kFp32PerBlock], headSize_);
            // Once per head per task, not per tile: the alternative is a
            // full-length Mul per tile instead of a Muls immediate.
            qScale_[h] = reduce.GetValue(kFp32PerBlock);

            codec_.CastToOperand(qOperand[h * operandElems_], vec, headSize_);

            // NZ: head h occupies lane h of every C0 block.
            SyncVectorToMte3();
            AscendC::DataCopy(qL1[h * kOperandC0], qOperand[h * operandElems_], nzParams);
        }

        // The two scale lanes this task reads out of every token's slot.
        const int32_t slotBytes = static_cast<int32_t>(scaleSlot_ * sizeof(float));
        AscendC::LocalTensor<int32_t> idx = scaleIdxBuf_.Get<int32_t>();
        AscendC::ArithProgression(idx, static_cast<int32_t>(kvHead * sizeof(float)), slotBytes,
                                  static_cast<int32_t>(kCubeTileRows));
        AscendC::ArithProgression(idx[kCubeTileRows], static_cast<int32_t>((numKvHeads_ + kvHead) * sizeof(float)),
                                  slotBytes, static_cast<int32_t>(kCubeTileRows));
        AscendC::PipeBarrier<PIPE_V>();
    }

    /*
     * Where the tile walk has got to.  A tile is (physical block, row offset),
     * and the walk skips block-table holes and clamps the last block to the
     * context length -- so it is not an affine function of a tile index and has
     * to be stepped.
     */
    struct TileCursor {
        uint32_t block = 0;     // next block to examine, or the current one
        uint32_t rows = 0;      // valid rows of the current block
        uint32_t base = 0;      // row offset of the current tile within it
        uint32_t physical = 0;  // the current tile's physical block
        uint32_t valid = 0;     // valid rows of the current tile
        bool active = false;    // base is part-way through a resolved block
    };

    // Tiles this split will process.  Cheap -- one scalar block-table read per
    // block, which the walk makes anyway -- and it is what lets both halves
    // express every flag condition as arithmetic on a tile index instead of
    // discovering the end by running off it.
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

    // Advance to the next tile.  False once the split is exhausted; the callers
    // in this file never see that, because they are bounded by CountTiles.
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

    // One tile's whole producer side: the packed read, and both operands onto
    // the grid and into L1[slot].  Every DataCopy into L1 is already guarded by
    // UnpackToL1's SyncVectorToMte3, and the flag the caller posts after this
    // is on PIPE_MTE3, so it cannot outrun them.
    __aicore__ inline void StageTile(const TileCursor &tile, uint32_t kvHead, uint32_t l1SlotIdx)
    {
        AscendC::LocalTensor<int8_t> packedTile = kvBuf_.Get<int8_t>();
        AscendC::LocalTensor<float> scaleTile = scaleTileBuf_.Get<float>();
        CopyInTile(packedTile, scaleTile, tile.physical, kvHead, tile.base);
        UnpackToL1(packedTile, mm_.B1K(l1SlotIdx));
        UnpackToL1(packedTile[kCubeTileRows * packedBytes_], mm_.B1V(l1SlotIdx));
    }

    /*
     * The vector half, software-pipelined one tile deep.
     *
     * The online softmax is a loop-carried dependency and stays on the critical
     * path; what is hoisted a tile ahead is the block-table-only work -- the packed
     * read, the unpack, and the L1 stage.
     *
     * The prefetch sits after the softmax and before the wait on the context
     * product: after, because CopyInTile overwrites scaleTileBuf_ which the softmax
     * reads this tile's scale lanes from; before, because that is the window the
     * Cube is busy in. That single ordering is also why nothing in UB needs a
     * second slot.
     *
     * Every flag set has exactly one wait at any tile count -- these are hardware
     * counters the kernel does not clear on exit. See TURBOQUANT_TESTS.md 13.11.
     */
    __aicore__ inline void PipelineAiv(uint32_t token, uint32_t contextLen, uint32_t blockStart, uint32_t blockEnd,
                                       uint32_t numTiles, uint32_t kvHead, const AscendC::LocalTensor<float> &acc,
                                       const AscendC::LocalTensor<float> &state)
    {
        if (numTiles == 0) {
            return;
        }
        AscendC::LocalTensor<float> scaleTile = scaleTileBuf_.Get<float>();
        AscendC::LocalTensor<float> scores = scoreBuf_.Get<float>();
        AscendC::LocalTensor<float> ctx = ctxBuf_.Get<float>();

        // Two cursors over one enumeration: stageCursor runs a tileIdx ahead of consumeCursor.
        TileCursor stageCursor;
        TileCursor consumeCursor;
        stageCursor.block = blockStart;
        consumeCursor.block = blockStart;

        // Prologue.  Tile 0 has no predecessor to overlap its staging against.
        NextTile(stageCursor, token, contextLen, blockEnd);
        StageTile(stageCursor, kvHead, 0);
        SignalSlotReady(0);

        for (uint32_t tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
            const uint32_t nextSlotIdx = (tileIdx + 1) % kSlots;
            NextTile(consumeCursor, token, contextLen, blockEnd);

            // Both subcores wait -- the counts have to stay balanced -- but
            // scores is UB the Fixpipe wrote into only one of them, and the
            // softmax also stages the probability row this tileIdx's context GEMM
            // reads.  See IsPrimarySubcore.
            AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagScoresReady);
            if (IsPrimarySubcore()) {
                SoftmaxStageProbs(scores, scaleTile, state, consumeCursor.valid);
            }
            // Posted the moment the probability row is in A1Probs, and BEFORE
            // the accumulator rescale, which the context GEMM does not depend
            // on.  Ungated: mode 0x02 needs both subcores, and it is subcore 0's
            // set -- the one that did the MTE3 -- that actually releases the
            // Cube.  See SignalProbsReady and SoftmaxRescaleAcc.
            SignalProbsReady();
            if (IsPrimarySubcore()) {
                SoftmaxRescaleAcc(state, acc);
            }

            if (tileIdx + 1 < numTiles) {
                if (tileIdx + 1 >= kSlots) {
                    // L1[nextSlotIdx] last held tileIdx - 1.  The Cube has to have drained
                    // it into L0B before it can be restaged.
                    AscendC::CrossCoreWaitFlag(
                        static_cast<uint16_t>(vllm_ascend::turboquant::kFlagSlotFree + nextSlotIdx));
                }
                NextTile(stageCursor, token, contextLen, blockEnd);
                StageTile(stageCursor, kvHead, nextSlotIdx);
                SignalSlotReady(nextSlotIdx);
            }

            AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagContextReady);
            if (IsPrimarySubcore()) {
                Accumulate(acc, ctx, state);
            }
        }
    }

    /*
     * The cube half, one tile behind the AIV in steady state, so its wait on
     * kFlagSlotReady costs nothing there. The slot is released after the context
     * load rather than the score load because one slot carries both K and V.
     */
    __aicore__ inline void PipelineAic(uint32_t numTiles)
    {
        AscendC::LocalTensor<float> scores = scoreBuf_.Get<float>();
        AscendC::LocalTensor<float> ctx = ctxBuf_.Get<float>();

        for (uint32_t tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
            const uint32_t l1SlotIdx = tileIdx % kSlots;

            AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(vllm_ascend::turboquant::kFlagSlotReady + l1SlotIdx));
            mm_.GemmScores(scores, mm_.B1K(l1SlotIdx), kMaxGroupHeads, headSize_, kCubeTileRows);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(vllm_ascend::turboquant::kFlagScoresReady);

            // The context GEMM's A operand, which the AIV has only just
            // produced. Lock-step got this ordering free from the V staging
            // sitting between the two GEMMs; the pipelined form does not.
            AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagProbsReady);
            mm_.GemmContext(ctx, mm_.B1V(l1SlotIdx), kMaxGroupHeads, kCubeTileRows, headSize_);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(vllm_ascend::turboquant::kFlagContextReady);

            // Conditioned so that every set has exactly one wait: on the last
            // kSlots tiles no restage of this slot follows.
            if (tileIdx + kSlots < numTiles) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE1>(
                    static_cast<uint16_t>(vllm_ascend::turboquant::kFlagSlotFree + l1SlotIdx));
            }
        }
    }

    __aicore__ inline void CopyInTile(const AscendC::LocalTensor<int8_t> &kv,
                                      const AscendC::LocalTensor<float> &scales, uint32_t physical, uint32_t kvHead,
                                      uint32_t rowBase)
    {
        if ASCEND_IS_AIC {
            return;
        }
        const uint64_t row = static_cast<uint64_t>(physical) * blockSize_ + rowBase;
        const uint64_t cacheOff = row * packedPlane_ + static_cast<uint64_t>(kvHead) * packedBytes_;
        const uint64_t scaleOff = row * scaleSlot_;

        if constexpr (Codec::kIsAffine) {
            /*
             * Group-major, not row-major: the tile lands in UB as [group][row][32], so a
             * group's 64-row run expands element-for-element onto exactly one NZ fractal
             * column and the L1 stage needs no reshape. Costs packedGroups_ descriptors per
             * plane instead of one. See TURBOQUANT_TESTS.md 13.9.
             */
            const AscendC::DataCopyParams groupParams{static_cast<uint16_t>(kCubeTileRows), 1,
                                                      static_cast<uint16_t>((packedPlane_ - kOperandC0) / 32), 0};
            const uint32_t groupElems = kCubeTileRows * kOperandC0;
            for (uint32_t groupIdx = 0; groupIdx < packedGroups_; ++groupIdx) {
                AscendC::DataCopy(kv[groupIdx * groupElems],
                                  keyCacheGm_[cacheOff + groupIdx * kOperandC0], groupParams);
                AscendC::DataCopy(kv[kCubeTileRows * packedBytes_ + groupIdx * groupElems],
                                  valueCacheGm_[cacheOff + groupIdx * kOperandC0], groupParams);
            }
        } else {
            // One tile row per burst; consecutive rows are packedPlane_ apart.
            // Every length and stride is a whole number of 32B blocks.
            const AscendC::DataCopyParams rowsParams{static_cast<uint16_t>(kCubeTileRows),
                                                     static_cast<uint16_t>(packedBytes_ / 32),
                                                     static_cast<uint16_t>((packedPlane_ - packedBytes_) / 32), 0};
            AscendC::DataCopy(kv, keyCacheGm_[cacheOff], rowsParams);
            AscendC::DataCopy(kv[kCubeTileRows * packedBytes_], valueCacheGm_[cacheOff], rowsParams);
        }
        AscendC::DataCopy(scales, scaleCacheGm_[scaleOff], kCubeTileRows * scaleSlot_);
        SyncMte2ToVector();
    }

    /*
     * Expand one tile's packed bytes onto the operand grid and place them in L1.
     *
     *   affine    flat. CopyInTile already grouped the tile, so each chunk is two
     *             plain DataCopy calls at fixed offsets: low nibbles of groups
     *             [g, g+n) to NZ columns [g, g+n), high nibbles to [packedGroups_+g,
     *             +n).
     *   codebook  strided. The codec's Gather emits NZ order within a kUnpackRows
     *             band, so placement is c0Blocks runs at a kCubeTileRows * 32 pitch.
     *
     * Both keep the V -> MTE3 handshake before every copy; see
     * TURBOQUANT_TESTS.md 13.14.4 and 13.9.
     */
    __aicore__ inline void UnpackToL1(const AscendC::LocalTensor<int8_t> &packed,
                                      const AscendC::LocalTensor<OperandT> &l1Dst)
    {
        if ASCEND_IS_AIC {
            return;
        }
        AscendC::LocalTensor<OperandT> unpackedUb = operandBuf_.Get<OperandT>();

        if constexpr (Codec::kIsAffine) {
            const uint32_t groupElems = kCubeTileRows * kOperandC0;
            const AscendC::DataCopyParams flatParams{1, static_cast<uint16_t>(unpackBytes_ / 32), 0, 0};
            for (uint32_t groupBase = 0; groupBase < packedGroups_; groupBase += unpackGroups_) {
                vllm_ascend::turboquant::unpack_tq4_to_fp8(codec_, unpackedUb, unpackedUb[unpackBytes_],
                                                           packed[groupBase * groupElems], unpackBytes_);
                SyncVectorToMte3();
                AscendC::DataCopy(l1Dst[groupBase * groupElems], unpackedUb, flatParams);
                AscendC::DataCopy(l1Dst[(packedGroups_ + groupBase) * groupElems],
                                  unpackedUb[unpackBytes_], flatParams);
            }
        } else {
            const uint32_t c0Blocks = operandElems_ / kOperandC0;
            const uint32_t bandElems = kUnpackRows * kOperandC0;
            const AscendC::DataCopyParams nzParams{
                static_cast<uint16_t>(c0Blocks), static_cast<uint16_t>(bandElems / 32), 0,
                static_cast<uint16_t>((kCubeTileRows - kUnpackRows) * kOperandC0 / 32)};

            for (uint32_t bandIdx = 0; bandIdx < kCubeTileRows / kUnpackRows; ++bandIdx) {
                // Named per mode at the call site; both are one template
                // instantiated twice, differing only in radix and operand grid.
                const AscendC::LocalTensor<int8_t> sub_packed = packed[bandIdx * kUnpackRows * packedBytes_];
                if constexpr (MODE == TurboQuantMode::KV5_FP8) {
                    vllm_ascend::turboquant::unpack_tq5_to_fp8(codec_, unpackedUb, sub_packed,
                                                               static_cast<int>(kUnpackRows),
                                                               static_cast<int>(headSize_));
                } else {
                    codec_.Unpack(unpackedUb, sub_packed, static_cast<int>(kUnpackRows),
                                  static_cast<int>(headSize_));
                }
                SyncVectorToMte3();
                AscendC::DataCopy(l1Dst[bandIdx * bandElems], unpackedUb, nzParams);
            }
        }
        // StageTile calls this twice on one unpackedUb, so the next call's
        // vector expand would otherwise race this one's DMA out of it.
        SyncMte3ToVector();
    }

    // The online softmax, per query head, over one tile's kCubeTileRows scores.
    // The score row arrives as Q_fp8 . K_fp8, so three factors still have to
    // come out of it: the query's operand scale (per head), the per-vector K
    // scale divided by the codebook gain (per column), and the attention scale.
    __aicore__ inline void SoftmaxStageProbs(const AscendC::LocalTensor<float> &scores,
                                             const AscendC::LocalTensor<float> &scaleTile,
                                             const AscendC::LocalTensor<float> &state, uint32_t valid)
    {
        AscendC::LocalTensor<float> runMax = state;
        AscendC::LocalTensor<float> runSum = state[kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> tileMax = state[2 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> newMax = state[3 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> alpha = state[4 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> probScale = state[5 * kMaxGroupHeads * kFp32PerBlock];

        AscendC::LocalTensor<float> reduce = reduceBuf_.Get<float>();
        AscendC::LocalTensor<float> kScale = reduce;
        AscendC::LocalTensor<float> vScale = reduce[kCubeTileRows];
        // Borrowed, not allocated: growing any buffer here exhausts UB, and
        // an InitBuffer that fails hands back a base of 0 whose stores
        // decode as DDR. ctxBuf_ is not written until the context GEMM
        // Fixpipes into it, which is after every use below.
        AscendC::LocalTensor<float> reduceWork = ctxBuf_.Get<float>();
        AscendC::LocalTensor<OperandT> probOperand = probOperandBuf_.Get<OperandT>();

        // The tile's K and V scales, one lane per row, out of the packed slots.
        AscendC::LocalTensor<uint32_t> idx = scaleIdxBuf_.Get<int32_t>().ReinterpretCast<uint32_t>();
        AscendC::Gather(kScale, scaleTile, idx, vllm_ascend::turboquant::kGatherSrcBase, kCubeTileRows);
        AscendC::Gather(vScale, scaleTile, idx[kCubeTileRows], vllm_ascend::turboquant::kGatherSrcBase,
                        kCubeTileRows);
        AscendC::PipeBarrier<PIPE_V>();
        // s / gain, the codebook gain undone; then the attention scale.
        AscendC::Muls(kScale, kScale, scale_ / TurboQuantModeTraits<MODE>::kGain, kCubeTileRows);
        AscendC::Muls(vScale, vScale, 1.0f / TurboQuantModeTraits<MODE>::kGain, kCubeTileRows);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t h = 0; h < groupHeads_; ++h) {
            AscendC::LocalTensor<float> row = scores[h * kCubeTileRows];
            AscendC::Mul(row, row, kScale, kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(row, row, 1.0f / qScale_[h], kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            if (valid < kCubeTileRows) {
                AscendC::Duplicate(row[valid], kNegInf, kCubeTileRows - valid);
                AscendC::PipeBarrier<PIPE_V>();
            }

            // `row` is the score row and is read again by the BroadcastSub
            // below, so it cannot also be the reduction's scratch.
            AscendC::ReduceMax<float>(tileMax[h * kFp32PerBlock], row, reduceWork, kCubeTileRows, false);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(newMax[h * kFp32PerBlock], runMax[h * kFp32PerBlock], tileMax[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(alpha[h * kFp32PerBlock], runMax[h * kFp32PerBlock], newMax[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(alpha[h * kFp32PerBlock], alpha[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<float> brcb = reduce[2 * kCubeTileRows];
            BroadcastScalar(brcb, newMax[h * kFp32PerBlock]);
            BroadcastSub(row, row, brcb, kCubeTileRows);
            AscendC::Exp(row, row, kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            if (valid < kCubeTileRows) {
                AscendC::Duplicate(row[valid], 0.0f, kCubeTileRows - valid);
                AscendC::PipeBarrier<PIPE_V>();
            }

            // The running sum, before the V scale is folded in: the denominator
            // is a sum of probabilities and must not carry the value scale.
            AscendC::LocalTensor<float> part = reduce[3 * kCubeTileRows];
            AscendC::ReduceSum<float>(part, row, part[kFp32PerBlock], kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(runSum[h * kFp32PerBlock], runSum[h * kFp32PerBlock], alpha[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(runSum[h * kFp32PerBlock], runSum[h * kFp32PerBlock], part, 1);
            AscendC::PipeBarrier<PIPE_V>();

            // P * s_v, then up onto the operand grid.  The row's amax after the
            // fold is data-dependent, so the factor is measured rather than
            // assumed, and it is divided back out of the context product.
            AscendC::Mul(row, row, vScale, kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax<float>(part, row, part[kFp32PerBlock], kCubeTileRows, false);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(part, part, TurboQuantCodec4::kEps, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(probScale[h * kFp32PerBlock], OperandMax<MODE>(), 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(probScale[h * kFp32PerBlock], probScale[h * kFp32PerBlock], part, 1);
            AscendC::PipeBarrier<PIPE_V>();
            BroadcastScalar(brcb, probScale[h * kFp32PerBlock]);
            TurboQuantCodec4::BroadcastMul(row, row, brcb, kCubeTileRows);

            codec_.CastToOperand(probOperand[h * Mm::OperandElems(kCubeTileRows)], row, kCubeTileRows);
        }

        // The probability tile is kCubeTileRows wide, which is exactly two C0
        // blocks, so its NZ image is not the same as its ND image and the rows
        // have to be interleaved on the way to L1: lane h of C0 block b.
        AscendC::LocalTensor<OperandT> pL1 = mm_.A1Probs();
        const uint32_t pElems = Mm::OperandElems(kCubeTileRows);
        const uint32_t pBlocks = pElems / kOperandC0;
        const AscendC::DataCopyParams pParams{static_cast<uint16_t>(pBlocks), 1, 0,
                                              static_cast<uint16_t>(kMaxGroupHeads - 1)};
        for (uint32_t h = 0; h < groupHeads_; ++h) {
            SyncVectorToMte3();
            AscendC::DataCopy(pL1[h * kOperandC0], probOperand[h * pElems], pParams);
        }
        SyncMte3ToVector();
    }

    /*
     * The rest of the online softmax: rescale the accumulator by alpha and advance
     * the running max. Split from SoftmaxStageProbs so kFlagProbsReady can post
     * between them -- the context GEMM depends on the probability row, not on this.
     * Safe to run against that GEMM: the AIC only Fixpipes ctxBuf_ and scoreBuf_,
     * neither of which this touches.
     */
    __aicore__ inline void SoftmaxRescaleAcc(const AscendC::LocalTensor<float> &state,
                                             const AscendC::LocalTensor<float> &acc)
    {
        AscendC::LocalTensor<float> runMax = state;
        AscendC::LocalTensor<float> newMax = state[3 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> alpha = state[4 * kMaxGroupHeads * kFp32PerBlock];

        // The accumulator's rescale by alpha, one column chunk at a time: a
        // repeat advances one row of acc and one 32B block of alpha, which is
        // exactly the addressing BinaryRepeatParams can express.
        const uint8_t rowBlocks = static_cast<uint8_t>(headSize_ / kFp32PerBlock);
        AscendC::LocalTensor<float> alphaBlocks = reduceBuf_.Get<float>()[3 * kCubeTileRows + 2 * kFp32PerBlock];
        for (uint32_t h = 0; h < groupHeads_; ++h) {
            AscendC::Brcb(alphaBlocks[h * kFp32PerBlock], alpha[h * kFp32PerBlock], 1,
                          {1, static_cast<uint16_t>(kFp32PerBlock)});
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t col = 0; col < headSize_; col += kFp32PerRepeat) {
            AscendC::Mul(acc[col], acc[col], alphaBlocks, static_cast<uint64_t>(kFp32PerRepeat),
                         static_cast<uint8_t>(groupHeads_), {1, 1, 0, rowBlocks, rowBlocks, 1});
        }
        AscendC::PipeBarrier<PIPE_V>();

        // The running max advances only after alpha has been consumed.
        for (uint32_t h = 0; h < groupHeads_; ++h) {
            AscendC::Adds(runMax[h * kFp32PerBlock], newMax[h * kFp32PerBlock], 0.0f, 1);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // acc += ctx / probScale, per head.
    __aicore__ inline void Accumulate(const AscendC::LocalTensor<float> &acc, const AscendC::LocalTensor<float> &ctx,
                                      const AscendC::LocalTensor<float> &state)
    {
        AscendC::LocalTensor<float> probScale = state[5 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> reduce = reduceBuf_.Get<float>();
        AscendC::LocalTensor<float> inv = reduce[3 * kCubeTileRows];
        AscendC::LocalTensor<float> block = reduce[3 * kCubeTileRows + kFp32PerBlock];
        for (uint32_t h = 0; h < groupHeads_; ++h) {
            // probScale carries the factor in lane 0 only, so splay it across
            // the block before dividing: the other seven lanes are zeros and
            // would divide by zero.
            BroadcastScalar(block, probScale[h * kFp32PerBlock]);
            AscendC::Duplicate(inv, 1.0f, kFp32PerBlock);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(inv, inv, block, kFp32PerBlock);
            AscendC::PipeBarrier<PIPE_V>();
            TurboQuantCodec4::BroadcastMul(ctx[h * headSize_], ctx[h * headSize_], inv, headSize_);
        }
        AscendC::Add(acc, acc, ctx, groupHeads_ * headSize_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::TPipe *pipe_;
    TurboQuantCodec4 rotation_;
    Codec codec_;
    Mm mm_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qInQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> kvBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleTileBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> operandBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> qBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> qOperandBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scoreBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> probOperandBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> ctxBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stateBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> signBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleIdxBuf_;
    AscendC::GlobalTensor<scalar_t> queryGm_;
    AscendC::GlobalTensor<int8_t> keyCacheGm_;
    AscendC::GlobalTensor<int8_t> valueCacheGm_;
    AscendC::GlobalTensor<float> scaleCacheGm_;
    AscendC::GlobalTensor<int32_t> blockTableGm_;
    AscendC::GlobalTensor<int32_t> contextLenGm_;
    AscendC::GlobalTensor<float> piSignsGm_;
    AscendC::GlobalTensor<int32_t> rotTablesGm_;
    AscendC::GlobalTensor<int32_t> modeTablesGm_;
    AscendC::GlobalTensor<float> workspaceGm_;
    float qScale_[kMaxGroupHeads] = {};
    uint32_t numTokens_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t groupHeads_ = 1;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlocksPerSeq_ = 0;
    uint32_t numSplits_ = 1;
    uint32_t packedBytes_ = 0;
    uint32_t packedPlane_ = 0;
    uint32_t scaleSlot_ = 0;
    uint32_t partialStride_ = 0;
    uint32_t operandElems_ = 0;
    // Affine path only: C0 column-groups in one packed slot, how many of them
    // one Unpack call covers, and the packed bytes that comes to.
    uint32_t packedGroups_ = 0;
    uint32_t unpackGroups_ = 0;
    uint32_t unpackBytes_ = 0;
    float scale_ = 1.0f;
};

/*
 * npu_turboquant_fp16_decode_split -- the physical FP16 baseline.
 *
 * aclnnFusedInferAttentionScore V1..V4 are withdrawn on an Ascend950 (planning
 * returns 361001), so the fp16 comparator is built here: Q_fp16 . K_fp16^T and
 * P_fp16 . V_fp16 on the same Cube, over an unquantised fp16 paged cache.
 *
 * Same task decomposition, GQA batching, tile size, online softmax, partial
 * layout and split count as the quantised kernel.  The only difference is where
 * the Cube operands come from: here the AIC's own MTE2 copies them out of the
 * fp16 cache with the ND->NZ conversion DataCopy does for 2-byte types.
 *
 * Its combine is separate from the AIV path's because this accumulator was
 * never rotated.
 */
template <typename scalar_t>
class TurboQuantFp16DecodeSplit {
public:
    __aicore__ inline explicit TurboQuantFp16DecodeSplit(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *query, __gm__ void *keyCache, __gm__ void *valueCache,
                                __gm__ void *blockTables, __gm__ void *contextLens, __gm__ void *workspace,
                                uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize,
                                uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits, float scale)
    {
        numTokens_ = numTokens;
        numHeads_ = numHeads;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        blockSize_ = blockSize;
        maxBlocksPerSeq_ = maxBlocksPerSeq;
        numSplits_ = numSplits;
        scale_ = scale;
        groupHeads_ = numHeads / numKvHeads;
        partialStride_ = headSize + kPartialTail;

        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(query));
        keyCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(keyCache));
        valueCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(valueCache));
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTables));
        contextLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(contextLens), numTokens);
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

        // L1 for the fp16 operands: query [M, D], probabilities [M, tileRows]
        // and the K/V tile [tileRows, D], all in elements of scalar_t.
        // L1 on both halves: aP1_ is written by Softmax, which is AIV work.
        pipe_->InitBuffer(aQ1_, kMaxGroupHeads * headSize_ * sizeof(scalar_t));
        pipe_->InitBuffer(aP1_, kMaxGroupHeads * kCubeTileRows * sizeof(scalar_t));
        pipe_->InitBuffer(b1_, kCubeTileRows * headSize_ * sizeof(scalar_t));
        // L0 on the AIC alone, for the reason spelled out in
        // TurboQuantCubeMm::Init: a vector core has no L0A, L0B or L0C, and the
        // tensors an unconditional InitBuffer hands it there fault on first
        // touch as `ldst_addr: 2`.  Gemm is the only reader and runs under
        // ASCEND_IS_AIC.
        if ASCEND_IS_AIC {
            pipe_->InitBuffer(a2_, kMaxGroupHeads * headSize_ * sizeof(scalar_t));
            pipe_->InitBuffer(b2_, kCubeTileRows * headSize_ * sizeof(scalar_t));
            pipe_->InitBuffer(co1_, kMaxGroupHeads * headSize_ * sizeof(float));
        }

        pipe_->InitBuffer(accBuf_, kMaxGroupHeads * headSize_ * sizeof(float));
        pipe_->InitBuffer(scoreBuf_, kMaxGroupHeads * kCubeTileRows * sizeof(float));
        pipe_->InitBuffer(probBuf_, kMaxGroupHeads * kCubeTileRows * sizeof(scalar_t));
        pipe_->InitBuffer(ctxBuf_, kMaxGroupHeads * headSize_ * sizeof(float));
        pipe_->InitBuffer(stateBuf_, 5 * kMaxGroupHeads * kFp32PerBlock * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, 4 * kMaxGroupHeads * kFp32PerBlock * sizeof(float));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process(uint32_t tasksPerCore)
    {
        const uint32_t tasks = numTokens_ * numKvHeads_ * numSplits_;
        // MixBlockIdx(), not GetBlockIdx(): every tile below carries an
        // AIC <-> AIV handshake, so the two halves of a MIX block have to walk
        // the same task list or they deadlock against each other. See
        // MixBlockIdx.
        uint32_t start = MixBlockIdx() * tasksPerCore;
        uint32_t end = start + tasksPerCore;
        if (end > tasks) {
            end = tasks;
        }
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t split = task % numSplits_;
            const uint32_t rest = task / numSplits_;
            ComputeSplit(rest / numKvHeads_, rest % numKvHeads_, split);
        }
    }

private:
    __aicore__ inline uint64_t PartialOffset(uint32_t token, uint32_t head, uint32_t split) const
    {
        return ((static_cast<uint64_t>(token) * numHeads_ + head) * numSplits_ + split) * partialStride_;
    }

    __aicore__ inline void ComputeSplit(uint32_t token, uint32_t kvHead, uint32_t split)
    {
        AscendC::LocalTensor<float> acc = accBuf_.Get<float>();
        AscendC::LocalTensor<float> state = stateBuf_.Get<float>();
        AscendC::LocalTensor<float> runMax = state;
        AscendC::LocalTensor<float> runSum = state[kMaxGroupHeads * kFp32PerBlock];

        if ASCEND_IS_AIV {
            AscendC::Duplicate(acc, 0.0f, kMaxGroupHeads * headSize_);
            AscendC::Duplicate(state, 0.0f, 5 * kMaxGroupHeads * kFp32PerBlock);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t h = 0; h < groupHeads_; ++h) {
                AscendC::Duplicate(runMax[h * kFp32PerBlock], kNegInf, 1);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        const int32_t contextLen = contextLenGm_.GetValue(token);
        const uint32_t seqBlocks = contextLen > 0 ? CeilDiv(static_cast<uint32_t>(contextLen), blockSize_) : 0;
        const uint32_t blocksPerSplit = CeilDiv(seqBlocks, numSplits_);
        const uint32_t blockStart = split * blocksPerSplit;
        uint32_t blockEnd = blockStart + blocksPerSplit;
        if (blockEnd > seqBlocks) {
            blockEnd = seqBlocks;
        }

        if (blockStart < blockEnd) {
            // The whole query group, GM -> L1 with the ND -> NZ conversion, on
            // the AIC own MTE2: the heads of one kv group are contiguous in
            // [token, head, head_size], so this is a single copy and the vector
            // core is not involved in the fp16 path operand supply at all.
            if ASCEND_IS_AIC {
                AscendC::LocalTensor<scalar_t> qL1 = aQ1_.template Get<scalar_t>();
                AscendC::DataCopy(qL1,
                                  queryGm_[(static_cast<uint64_t>(token) * numHeads_ + kvHead * groupHeads_) *
                                           headSize_],
                                  AscendC::Nd2NzParams(1, static_cast<uint16_t>(kMaxGroupHeads), headSize_, 0,
                                                       headSize_, static_cast<uint16_t>(kMaxGroupHeads), 1, 0));
                AscendC::PipeBarrier<PIPE_ALL>();
            }
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
                for (uint32_t base = 0; base < rows; base += kCubeTileRows) {
                    uint32_t valid = rows - base;
                    if (valid > kCubeTileRows) {
                        valid = kCubeTileRows;
                    }
                    ProcessTile(static_cast<uint32_t>(physical), kvHead, base, valid, acc, state);
                }
            }
        }

        // Only the subcore the Fixpipe wrote into holds this task's
        // accumulator and running state; the other's are whatever its UB had.
        // Ungated, both wrote to the same workspace offsets and the last writer
        // won. See IsPrimarySubcore.
        // ASCEND_IS_AIV expands to constexpr(...), so it cannot be combined
        // with && -- the two conditions have to nest.
        if ASCEND_IS_AIV {
          if (IsPrimarySubcore()) {
            AscendC::LocalTensor<float> tail = reduceBuf_.Get<float>();
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t h = 0; h < groupHeads_; ++h) {
                const uint32_t head = kvHead * groupHeads_ + h;
                const uint64_t offset = PartialOffset(token, head, split);
                AscendC::Duplicate(tail, 0.0f, kPartialTail);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(tail[kPartialMaxLane], runMax[h * kFp32PerBlock], 0.0f, 1);
                AscendC::Adds(tail[kPartialSumLane], runSum[h * kFp32PerBlock], 0.0f, 1);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(workspaceGm_[offset], acc[h * headSize_], headSize_);
                AscendC::DataCopy(workspaceGm_[offset + headSize_], tail, kPartialTail);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
          }
        }
    }

    __aicore__ inline void ProcessTile(uint32_t physical, uint32_t kvHead, uint32_t rowBase, uint32_t valid,
                                       const AscendC::LocalTensor<float> &acc,
                                       const AscendC::LocalTensor<float> &state)
    {
        AscendC::LocalTensor<float> scores = scoreBuf_.Get<float>();
        AscendC::LocalTensor<float> ctx = ctxBuf_.Get<float>();

        const uint64_t row = static_cast<uint64_t>(physical) * blockSize_ + rowBase;
        const uint64_t cacheOff = (row * numKvHeads_ + kvHead) * headSize_;
        // One tile of the fp16 cache, GM -> L1 with ND -> NZ. srcDValue is the
        // row pitch of the [block_size, num_kv_heads, head_size] plane, which is
        // how one kv head rows are picked out without a gather.
        const AscendC::Nd2NzParams tileParams(1, static_cast<uint16_t>(kCubeTileRows), headSize_, 0,
                                              static_cast<uint64_t>(numKvHeads_) * headSize_,
                                              static_cast<uint16_t>(kCubeTileRows), 1, 0);

        if ASCEND_IS_AIC {
            AscendC::LocalTensor<scalar_t> kL1 = b1_.template Get<scalar_t>();
            AscendC::DataCopy(kL1, keyCacheGm_[cacheOff], tileParams);
            AscendC::PipeBarrier<PIPE_ALL>();
            Gemm(scores, aQ1_.template Get<scalar_t>(), kMaxGroupHeads, headSize_, kCubeTileRows,
                 /*bStagedAsNk=*/true);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(vllm_ascend::turboquant::kFlagProductReady);
        }
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagProductReady);
            if (IsPrimarySubcore()) {
                Softmax(scores, state, acc, valid);
            }
            SignalOperandsReady();
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagOperandsReady);
            AscendC::LocalTensor<scalar_t> vL1 = b1_.template Get<scalar_t>();
            AscendC::DataCopy(vL1, valueCacheGm_[cacheOff], tileParams);
            AscendC::PipeBarrier<PIPE_ALL>();
            Gemm(ctx, aP1_.template Get<scalar_t>(), kMaxGroupHeads, kCubeTileRows, headSize_,
                 /*bStagedAsNk=*/false);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(vllm_ascend::turboquant::kFlagProductReady);
        }
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagProductReady);
            if (IsPrimarySubcore()) {
                AscendC::Add(acc, acc, ctx, groupHeads_ * headSize_);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    // The same L1 -> L0 -> Mmad -> Fixpipe sequence TurboQuantCubeMm::Run does,
    // over half operands. <float, half, half> is not one of the arch35
    // microscaled tuples, so this is plain mad, exactly as the fp8 path is.
    __aicore__ inline void Gemm(const AscendC::LocalTensor<float> &dstUb, const AscendC::LocalTensor<scalar_t> &ta1,
                                uint32_t m, uint32_t k, uint32_t n, bool transposeB)  // bStagedAsNk
    {
        AscendC::LocalTensor<scalar_t> tb1 = b1_.template Get<scalar_t>();
        AscendC::LocalTensor<scalar_t> ta2 = a2_.template Get<scalar_t>();
        AscendC::LocalTensor<scalar_t> tb2 = b2_.template Get<scalar_t>();
        AscendC::LocalTensor<float> tco = co1_.template Get<float>();

        // The same contract TurboQuantCubeMm documents, at C0 = 16 because a
        // 2-byte operand's C0 block is 16 elements rather than 32, and without
        // the mStep = 2 chunking, which is a .b8 restriction only.
        constexpr uint16_t kFractalRows = 16;
        constexpr uint16_t kC0 = 16;
        AscendC::LoadData2DParamsV2 pa;
        pa.mStep = static_cast<uint16_t>(CeilDiv(m, kFractalRows));
        pa.kStep = static_cast<uint16_t>(CeilDiv(k, kC0));
        pa.srcStride = pa.mStep;
        pa.dstStride = pa.mStep;
        pa.ifTranspose = false;
        AscendC::LoadData(ta2, ta1, pa);

        AscendC::LoadData2DParamsV2 pb;
        if (transposeB) {
            // B staged [n, k] -- the score GEMM. ifTranspose false; see the
            // contract note in turboquant_cube_mm.h for why that reads backwards.
            pb.mStep = static_cast<uint16_t>(CeilDiv(n, kFractalRows));
            pb.kStep = static_cast<uint16_t>(CeilDiv(k, kC0));
            pb.srcStride = static_cast<uint16_t>(CeilDiv(n, kFractalRows));
            pb.ifTranspose = false;
        } else {
            // B staged [k, n] -- the context GEMM.
            pb.mStep = static_cast<uint16_t>(CeilDiv(k, kFractalRows));
            pb.kStep = static_cast<uint16_t>(CeilDiv(n, kC0));
            pb.srcStride = static_cast<uint16_t>(CeilDiv(k, kFractalRows));
            pb.ifTranspose = true;
        }
        pb.dstStride = static_cast<uint16_t>(CeilDiv(n, kFractalRows));
        AscendC::LoadData(tb2, tb1, pb);

        // MTE1 -> M and M -> FIX, for the reason spelled out in
        // TurboQuantCubeMm::Compute: a PipeBarrier orders a pipe against itself
        // only, and without these the Mmad reads an L0 fragment LoadData has
        // not finished writing.
        AscendC::TPipe *tpipe = GetTPipePtr();
        const event_t mte1ToM = static_cast<event_t>(tpipe->FetchEventID(AscendC::HardEvent::MTE1_M));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1ToM);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1ToM);
        AscendC::Mmad(tco, ta2, tb2,
                      AscendC::MmadParams(static_cast<uint16_t>(m), static_cast<uint16_t>(n),
                                          static_cast<uint16_t>(k), 0, false, true));
        const event_t mToFix = static_cast<event_t>(tpipe->FetchEventID(AscendC::HardEvent::M_FIX));
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(mToFix);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(mToFix);
        AscendC::Fixpipe<float, float, vllm_ascend::turboquant::kFixpipeToUb>(
            dstUb, tco,
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>(
                static_cast<uint16_t>(n), static_cast<uint16_t>(m),
                static_cast<uint16_t>(CeilDiv(m, kFractalRows) * kFractalRows), n));
        AscendC::PipeBarrier<PIPE_FIX>();
    }

    // The online softmax, with no scale to unfold: the fp16 operands carry their
    // own magnitudes, so the score row needs only the attention scale.
    __aicore__ inline void Softmax(const AscendC::LocalTensor<float> &scores,
                                   const AscendC::LocalTensor<float> &state,
                                   const AscendC::LocalTensor<float> &acc, uint32_t valid)
    {
        AscendC::LocalTensor<float> runMax = state;
        AscendC::LocalTensor<float> runSum = state[kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> tileMax = state[2 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> newMax = state[3 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> alpha = state[4 * kMaxGroupHeads * kFp32PerBlock];
        AscendC::LocalTensor<float> reduce = reduceBuf_.Get<float>();
        // Borrowed; see the note in the quantised kernel's Softmax.
        AscendC::LocalTensor<float> reduceWork = ctxBuf_.Get<float>();
        AscendC::LocalTensor<scalar_t> probs = probBuf_.Get<scalar_t>();

        for (uint32_t h = 0; h < groupHeads_; ++h) {
            AscendC::LocalTensor<float> row = scores[h * kCubeTileRows];
            AscendC::Muls(row, row, scale_, kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            if (valid < kCubeTileRows) {
                AscendC::Duplicate(row[valid], kNegInf, kCubeTileRows - valid);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::ReduceMax<float>(tileMax[h * kFp32PerBlock], row, reduceWork, kCubeTileRows, false);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(newMax[h * kFp32PerBlock], runMax[h * kFp32PerBlock], tileMax[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(alpha[h * kFp32PerBlock], runMax[h * kFp32PerBlock], newMax[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(alpha[h * kFp32PerBlock], alpha[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<float> brcb = reduce[2 * kFp32PerBlock];
            BroadcastScalar(brcb, newMax[h * kFp32PerBlock]);
            BroadcastSub(row, row, brcb, kCubeTileRows);
            AscendC::Exp(row, row, kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            if (valid < kCubeTileRows) {
                AscendC::Duplicate(row[valid], 0.0f, kCubeTileRows - valid);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::ReduceSum<float>(reduce, row, reduce[kFp32PerBlock], kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(runSum[h * kFp32PerBlock], runSum[h * kFp32PerBlock], alpha[h * kFp32PerBlock], 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(runSum[h * kFp32PerBlock], runSum[h * kFp32PerBlock], reduce, 1);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(probs[h * kCubeTileRows], row, AscendC::RoundMode::CAST_RINT, kCubeTileRows);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // P -> L1, ND -> NZ. Unlike the fp8 path this overload exists: the
        // UB-side transform is a strided Adds, which has a half instantiation.
        AscendC::LocalTensor<scalar_t> pL1 = aP1_.template Get<scalar_t>();
        SyncVectorToMte3();
        AscendC::DataCopy(pL1, probs,
                          AscendC::Nd2NzParams(1, static_cast<uint16_t>(kMaxGroupHeads), kCubeTileRows, 0,
                                               kCubeTileRows, static_cast<uint16_t>(kMaxGroupHeads), 1, 0));

        const uint8_t rowBlocks = static_cast<uint8_t>(headSize_ / kFp32PerBlock);
        AscendC::LocalTensor<float> alphaBlocks = reduce[2 * kMaxGroupHeads * kFp32PerBlock];
        for (uint32_t h = 0; h < groupHeads_; ++h) {
            AscendC::Brcb(alphaBlocks[h * kFp32PerBlock], alpha[h * kFp32PerBlock], 1,
                          {1, static_cast<uint16_t>(kFp32PerBlock)});
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t col = 0; col < headSize_; col += kFp32PerRepeat) {
            AscendC::Mul(acc[col], acc[col], alphaBlocks, static_cast<uint64_t>(kFp32PerRepeat),
                         static_cast<uint8_t>(groupHeads_), {1, 1, 0, rowBlocks, rowBlocks, 1});
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t h = 0; h < groupHeads_; ++h) {
            AscendC::Adds(runMax[h * kFp32PerBlock], newMax[h * kFp32PerBlock], 0.0f, 1);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::TPipe *pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> aQ1_;
    AscendC::TBuf<AscendC::TPosition::A1> aP1_;
    AscendC::TBuf<AscendC::TPosition::B1> b1_;
    AscendC::TBuf<AscendC::TPosition::A2> a2_;
    AscendC::TBuf<AscendC::TPosition::B2> b2_;
    AscendC::TBuf<AscendC::TPosition::CO1> co1_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scoreBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> probBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> ctxBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> stateBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceBuf_;
    AscendC::GlobalTensor<scalar_t> queryGm_;
    AscendC::GlobalTensor<scalar_t> keyCacheGm_;
    AscendC::GlobalTensor<scalar_t> valueCacheGm_;
    AscendC::GlobalTensor<int32_t> blockTableGm_;
    AscendC::GlobalTensor<int32_t> contextLenGm_;
    AscendC::GlobalTensor<float> workspaceGm_;
    uint32_t numTokens_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t groupHeads_ = 1;
    uint32_t headSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlocksPerSeq_ = 0;
    uint32_t numSplits_ = 1;
    uint32_t partialStride_ = 0;
    float scale_ = 1.0f;
};

/*
 * The fp16 baseline combine: the same flash-decoding reduction the AIV path
 * does, minus the un-rotation.  Separate because this accumulator was never in
 * the rotated basis.  The PipeBarrier<PIPE_ALL> in the per-split loop is there
 * for the reason recorded in turboquant_kernels.cpp: partAcc is a plain TBuf
 * view, so nothing else orders iteration i+1's MTE2 fill against iteration i's
 * vector reads.
 */
template <typename scalar_t>
class TurboQuantPlainCombine {
public:
    __aicore__ inline explicit TurboQuantPlainCombine(AscendC::TPipe *pipe) : pipe_(pipe) {}

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
            const uint64_t offset =
                ((static_cast<uint64_t>(token) * numHeads_ + head) * numSplits_ + split) * partialStride_;
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

        AscendC::Adds(runSum, runSum, TurboQuantCodec4::kEps, 1);
        AscendC::PipeBarrier<PIPE_V>();
        BroadcastScalar(bSum, runSum);
        AscendC::Duplicate(invSum, 1.0f, kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(invSum, invSum, bSum, kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        TurboQuantCodec4::BroadcastMul(acc, acc, invSum, headSize_);

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

/*
 * A bare fp8 Cube GEMM for pinning the fractal contract numerically.
 *
 * It drives TurboQuantCubeMm rather than restating it, so what a test validates
 * is the code the decode runs.  Both operands arrive from GM already in NZ
 * order, so this tests the Cube contract and not the unpack.
 */
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

    // `bIsNk` selects which of the two B forms is exercised: the score GEMM's
    // ([n, k], one load) or the context GEMM's ([k, n], the chunked one).
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
            SignalOperandsReady();
        }
        if ASCEND_IS_AIC {
            // Variant 10: one wait per vector subcore. Both AIVs of a MIX block
            // run this kernel and both set kFlagOperandsReady, so if the AIC's
            // single wait consumes only one notification the surplus carries
            // into the next launch and satisfies its wait before the operands
            // are staged -- which would make the first GEMM of a process exact
            // and every one after it read an unstaged L1.
            AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagOperandsReady);
            if (variant == 10) {
                AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagOperandsReady);
            }
            if (bIsNk != 0) {
                mm_.GemmScores(out, mm_.B1(), m, k, n);
            } else {
                mm_.GemmContext(out, mm_.B1(), m, k, n, variant);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(vllm_ascend::turboquant::kFlagProductReady);
        }
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag(vllm_ascend::turboquant::kFlagProductReady);
            // Both subcores wait -- the flag counts have to stay balanced --
            // but only the one the Fixpipe wrote into has the product.
            if (IsPrimarySubcore()) {
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

}  // namespace

/*
 * Kernel entry points take GM_ADDR, not `__gm__ void *`; see the note at the
 * same place in turboquant_kernels.cpp.
 *
 * One entry point per (mode, dtype).  The mode is a template parameter and not
 * a runtime argument because the codec's thresholds are Adds immediates and its
 * level count decides the loop trip count.  TurboQuantModeIsValid guards the
 * host's choice.
 */
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

#define TURBOQUANT_MM_DECODE_SPLIT_DECLARE(MODE_NAME, MODE, TYPE)                                                    \
    extern "C" __global__ __aicore__ void turboquant_mm_decode_split_##MODE_NAME##_##TYPE(                           \
        GM_ADDR query, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR scaleCache, GM_ADDR blockTables,                \
        GM_ADDR contextLens, GM_ADDR piSigns, GM_ADDR rotTables, GM_ADDR modeTables, GM_ADDR workspace,              \
        uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,           \
        uint32_t maxBlocksPerSeq, uint32_t numSplits, uint32_t tasksPerCore, float scale, float invSqrtLen)          \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantCubeDecodeSplit<MODE, TYPE> op(&pipe);                                                             \
        op.Init(query, keyCache, valueCache, scaleCache, blockTables, contextLens, piSigns, rotTables, modeTables,   \
                workspace, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits, scale,  \
                invSqrtLen);                                                                                         \
        op.Process(tasksPerCore);                                                                                    \
    }

#define TURBOQUANT_MM_DECLARE_MODE(MODE_NAME, MODE)                                                                  \
    TURBOQUANT_MM_RESHAPE_AND_CACHE_DECLARE(MODE_NAME, MODE, half)                                                   \
    TURBOQUANT_MM_DECODE_SPLIT_DECLARE(MODE_NAME, MODE, half)

TURBOQUANT_MM_DECLARE_MODE(kv3fp4, TurboQuantMode::KV3_FP4)
TURBOQUANT_MM_DECLARE_MODE(kv4fp8, TurboQuantMode::KV4_FP8)
TURBOQUANT_MM_DECLARE_MODE(kv5fp8, TurboQuantMode::KV5_FP8)

// The fp16 baseline entry points.  Only `half` is instantiated: this is a
// comparator for the fp8 path, which is fp16-only.
#define TURBOQUANT_FP16_DECODE_SPLIT_DECLARE(TYPE)                                                                   \
    extern "C" __global__ __aicore__ void turboquant_fp16_decode_split_##TYPE(                                       \
        GM_ADDR query, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR blockTables, GM_ADDR contextLens,               \
        GM_ADDR workspace, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize,            \
        uint32_t blockSize, uint32_t maxBlocksPerSeq, uint32_t numSplits, uint32_t tasksPerCore, float scale)        \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantFp16DecodeSplit<TYPE> op(&pipe);                                                                    \
        op.Init(query, keyCache, valueCache, blockTables, contextLens, workspace, numTokens, numHeads, numKvHeads,   \
                headSize, blockSize, maxBlocksPerSeq, numSplits, scale);                                             \
        op.Process(tasksPerCore);                                                                                    \
    }

#define TURBOQUANT_PLAIN_COMBINE_DECLARE(TYPE)                                                                       \
    extern "C" __global__ __aicore__ void turboquant_plain_combine_##TYPE(                                           \
        GM_ADDR workspace, GM_ADDR output, uint32_t numTokens, uint32_t numHeads, uint32_t headSize,                 \
        uint32_t numSplits, uint32_t tasksPerCore)                                                                   \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantPlainCombine<TYPE> op(&pipe);                                                                      \
        op.Init(workspace, output, numTokens, numHeads, headSize, numSplits);                                        \
        op.Process(tasksPerCore);                                                                                    \
    }

TURBOQUANT_FP16_DECODE_SPLIT_DECLARE(half)
TURBOQUANT_PLAIN_COMBINE_DECLARE(half)

// The fp8 Cube GEMM probe.  Its arguments are the fractal contract itself, so a
// test can sweep them.
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

// fp16 only: the query and the softmax row are cast onto an fp8 or fp4 grid,
// and bf16's wider exponent buys nothing against a grid whose top is 448.
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

// The split half of the Cube decode.  The combine is a separate launch on the
// same stream, and it is the AIV-only one in turboquant_kernels.cpp: the
// partials this writes are in exactly that reduction's layout.
void turboquant_mm_decode_split_impl(int32_t mode, AscendType type, void *stream, uint32_t blockDim, void *query,
                                     void *keyCache, void *valueCache, void *scaleCache, void *blockTables,
                                     void *contextLens, void *piSigns, void *rotTables, void *modeTables,
                                     void *workspace, uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                                     uint32_t headSize, uint32_t blockSize, uint32_t maxBlocksPerSeq,
                                     uint32_t numSplits, uint32_t tasksPerCore, float scale, float invSqrtLen)
{
    if (type != AscendType::FP16) {
        return;
    }
    switch (static_cast<turboquant::TurboQuantMode>(mode)) {
        case turboquant::TurboQuantMode::KV3_FP4:
            turboquant_mm_decode_split_kv3fp4_half<<<blockDim, nullptr, stream>>>(
                query, keyCache, valueCache, scaleCache, blockTables, contextLens, piSigns, rotTables, modeTables,
                workspace, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
                tasksPerCore, scale, invSqrtLen);
            break;
        case turboquant::TurboQuantMode::KV4_FP8:
            turboquant_mm_decode_split_kv4fp8_half<<<blockDim, nullptr, stream>>>(
                query, keyCache, valueCache, scaleCache, blockTables, contextLens, piSigns, rotTables, modeTables,
                workspace, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
                tasksPerCore, scale, invSqrtLen);
            break;
        default:
            turboquant_mm_decode_split_kv5fp8_half<<<blockDim, nullptr, stream>>>(
                query, keyCache, valueCache, scaleCache, blockTables, contextLens, piSigns, rotTables, modeTables,
                workspace, numTokens, numHeads, numKvHeads, headSize, blockSize, maxBlocksPerSeq, numSplits,
                tasksPerCore, scale, invSqrtLen);
            break;
    }
}

// The physical fp16 baseline: split then combine, two launches on one stream.
// Same shapes, paging, tile size and split count as the quantised path, so the
// ratio between them measures the codec.
void turboquant_fp16_decode_impl(AscendType type, void *stream, uint32_t splitBlockDim, uint32_t combineBlockDim,
                                 void *query, void *keyCache, void *valueCache, void *blockTables,
                                 void *contextLens, void *workspace, void *output, uint32_t numTokens,
                                 uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,
                                 uint32_t maxBlocksPerSeq, uint32_t numSplits, uint32_t splitTasksPerCore,
                                 uint32_t combineTasksPerCore, float scale)
{
    if (type != AscendType::FP16) {
        return;
    }
    turboquant_fp16_decode_split_half<<<splitBlockDim, nullptr, stream>>>(
        query, keyCache, valueCache, blockTables, contextLens, workspace, numTokens, numHeads, numKvHeads, headSize,
        blockSize, maxBlocksPerSeq, numSplits, splitTasksPerCore, scale);
    turboquant_plain_combine_half<<<combineBlockDim, nullptr, stream>>>(
        workspace, output, numTokens, numHeads, headSize, numSplits, combineTasksPerCore);
}

// The Cube fractal probe.  `mode` is unused -- fp8 e4m3fn is the operand type
// the primary path uses -- and the step arguments are what the caller sweeps.
void turboquant_cube_gemm_probe_impl(void *stream, void *a, void *b, void *c, uint32_t m, uint32_t k, uint32_t n,
                                     uint32_t headSize, uint32_t tileRows, uint32_t aElems, uint32_t bElems,
                                     uint32_t cElems, uint32_t bIsNk, uint32_t variant)
{
    turboquant_cube_gemm_probe_fp8<<<1, nullptr, stream>>>(a, b, c, m, k, n, headSize, tileRows, aElems, bElems,
                                                           cElems, bIsNk, variant);
}

}  // namespace vllm_ascend
