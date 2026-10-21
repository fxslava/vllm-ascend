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

// The Pi basis change at both ends of the raw-query Cube decode (TURBOQUANT_TESTS.md 13.36), a MIX service:
//
//   Rotate()       the query prologue, once per launch before the task loop: Pi q = D H D q for every
//                  (token, head) vector, written as fp32 where a pre-rotated decode reads its query
//   FinishHeads()  the output stage, per normalised fp32 head row before the output cast: Pi once more
//                  (Pi^2 = I), and for a gated layer the division by 1 + exp(-gate)
//
// Block b's prologue share is vectors [b * vectorsPerBlock, (b + 1) * vectorsPerBlock). With a Cube chunk
// size, the share's whole chunks rotate on the Cube: each vector subcore sign-flips its half of the chunk,
// casts it to fp16 (exact for an fp16 query, since the signs and H16 are +-1) and stages it into its half of
// one L1 tile; the AIC multiplies the tile by H16, which is the butterfly strides 1 to 8, and Fixpipes each
// half back to its subcore (dualDstCtl = 0b01), which runs the strides from 16 up, scales by 1 / sqrt(D) and
// flips the signs. The rest of the share rotates on the vector cores with ApplyPi, half per subcore. The
// arithmetic is rotate_q's, stage for stage, on either path.
//
// Every core allocates the same UB list, after the decode's own buffers so those keep their offsets: the
// AIC's Fixpipe names the subcores' product buffer by its own copy's offset. The output stage runs after
// the prologue's SyncAll and reuses its scratch.

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_QUERY_BASIS_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_QUERY_BASIS_H

#include "../common/turboquant_codec_950.h"
#include "../common/turboquant_common.h"
#include "../vector/turboquant_vector_service.h"

namespace vllm_ascend {
namespace turboquant {

template <typename scalar_t, bool ENABLED>
class TurboQuantQueryBasis {
public:
    static constexpr bool kEnabled = true;

    // Call after the decode's Init. cubeChunkVectors is 0 (vector cores only) or an even chunk of at most
    // kScratchRows * 2 vectors whose rows fill whole 16-row fractals; gate is read only for kGated.
    __aicore__ inline void Init(AscendC::TPipe *pipe, __gm__ void *query, __gm__ void *piSigns,
                                __gm__ void *rotTables, __gm__ void *h16, __gm__ void *gate, __gm__ void *queryRot,
                                const uint32_t numHeads, const uint32_t headSize, const uint32_t cubeChunkVectors,
                                const uint32_t outputStage, const float invSqrtLen)
    {
        numHeads_ = numHeads;
        headSize_ = headSize;
        cubeChunk_ = cubeChunkVectors;
        outputStage_ = outputStage;
        invSqrtLen_ = invSqrtLen;
        tileElems_ = cubeChunk_ * headSize_;
        const uint32_t scratchElems = kScratchRows * headSize_;

        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(query));
        piSignsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(piSigns), headSize_);
        rotTablesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rotTables));
        h16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(h16));
        gateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(gate));
        queryRotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(queryRot));

        pipe->InitBuffer(inBuf_, scratchElems * sizeof(scalar_t));
        pipe->InitBuffer(floatBuf_, scratchElems * sizeof(float));
        pipe->InitBuffer(halfBuf_, scratchElems * sizeof(half));
        pipe->InitBuffer(pairBuf_, scratchElems * sizeof(float));
        pipe->InitBuffer(signBuf_, headSize_ * sizeof(float));
        codec_.Init(pipe, headSize_, 1, invSqrtLen, rotTablesGm_);

        if (cubeChunk_ > 0) {
            pipe->InitBuffer(tileL1Buf_, tileElems_ * sizeof(half));
            pipe->InitBuffer(h16L1Buf_, kRotateQH16Elements * sizeof(half));
            if ASCEND_IS_AIC {
                pipe->InitBuffer(tileL0aBuf_, tileElems_ * sizeof(half));
                pipe->InitBuffer(h16L0bBuf_, kRotateQH16Elements * sizeof(half));
                pipe->InitBuffer(productL0cBuf_, tileElems_ * sizeof(float));
            }
        }

        AscendC::DataCopy(signBuf_.Get<float>(), piSignsGm_, headSize_);
        if ASCEND_IS_AIV {
            if (cubeChunk_ > 0) {
                if (AscendC::GetSubBlockIdx() == 0) {
                    const AscendC::LocalTensor<half> stage = halfBuf_.Get<half>();
                    AscendC::DataCopy(stage, h16Gm_, kRotateQH16Elements);
                    SyncEvent<AscendC::HardEvent::MTE2_MTE3>();
                    AscendC::DataCopy(h16L1Buf_.Get<half>(), stage, kRotateQH16Elements);
                }
            }
        }
        // Init hand-off: the signs and the H16 tile land before the first rotation.
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // The prologue over this block's share of the numVectors query vectors. Drains every core afterwards, so
    // no task reads a rotated vector another block has not yet written.
    __aicore__ inline void Rotate(const uint32_t numVectors, const uint32_t vectorsPerBlock)
    {
        const uint32_t start = MixBlockIdx() * vectorsPerBlock;
        uint32_t end = start + vectorsPerBlock;
        if (end > numVectors) {
            end = numVectors;
        }
        const uint32_t share = start < end ? end - start : 0;
        const uint32_t chunks = cubeChunk_ > 0 ? share / cubeChunk_ : 0;

        if ASCEND_IS_AIC {
            for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
                ComputeTileProduct();
            }
        }
        if ASCEND_IS_AIV {
            for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
                RotateTileHalf(start + chunk * cubeChunk_);
            }
            const uint32_t rest = start + chunks * cubeChunk_;
            if (rest < end) {
                const uint32_t firstHalf = CeilDiv(end - rest, kVectorSubcoresPerBlock);
                const bool secondSubcore = AscendC::GetSubBlockIdx() != 0;
                const uint32_t from = secondSubcore ? rest + firstHalf : rest;
                const uint32_t to = secondSubcore ? end : rest + firstHalf;
                for (uint32_t vector = from; vector < to; ++vector) {
                    RotateVector(vector);
                }
            }
        }
        AscendC::SyncAll<false>();
    }

    // The output stage of `heads` normalised rows of `token`, starting at `firstHead`. At most kScratchRows.
    __aicore__ inline void FinishHeads(const AscendC::LocalTensor<float> &acc, const uint32_t token,
                                       const uint32_t firstHead, const uint32_t heads)
    {
        if (outputStage_ == static_cast<uint32_t>(TurboQuantOutputStage::kRotatedBasis)) {
            return;
        }
        const AscendC::LocalTensor<float> tmp = pairBuf_.Get<float>();
        const AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        for (uint32_t j = 0; j < heads; ++j) {
            codec_.ApplyPi(acc[j * headSize_], tmp, signs, headSize_);
        }
        if (outputStage_ != static_cast<uint32_t>(TurboQuantOutputStage::kGated)) {
            return;
        }

        // acc * sigmoid(gate) as acc / (1 + exp(-gate)): one division, no reciprocal buffer.
        const uint32_t elems = heads * headSize_;
        const AscendC::LocalTensor<scalar_t> gateIn = inBuf_.Get<scalar_t>();
        const AscendC::LocalTensor<float> denom = floatBuf_.Get<float>();
        AscendC::DataCopy(gateIn, gateGm_[(static_cast<uint64_t>(token) * numHeads_ + firstHead) * headSize_], elems);
        SyncMte2ToVector();
        AscendC::Cast(denom, gateIn, AscendC::RoundMode::CAST_NONE, elems);
        AscendC::Maxs(denom, denom, kGateLogitFloor, elems);
        AscendC::Muls(denom, denom, -1.0f, elems);
        AscendC::Exp(denom, denom, elems);
        AscendC::Adds(denom, denom, 1.0f, elems);
        AscendC::Div(acc, acc, denom, elems);
    }

private:
    static constexpr uint32_t kScratchRows = kCubeTileM / kVectorSubcoresPerBlock;
    // An fp16 operand's C0: one 16-element group per row of the H16 tile.
    static constexpr uint32_t kHalfOperandC0 = 16;
    static constexpr uint8_t kDualDstSplitM = 0b01;

    // One chunk's AIV half: stage it, wait for its H16 product, finish the transform and write it.
    __aicore__ inline void RotateTileHalf(const uint32_t chunkBase)
    {
        const uint32_t subcore = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
        const uint32_t own = cubeChunk_ / kVectorSubcoresPerBlock;
        const uint32_t elems = own * headSize_;
        const uint64_t firstElem = static_cast<uint64_t>(chunkBase + subcore * own) * headSize_;
        const AscendC::LocalTensor<scalar_t> queryIn = inBuf_.Get<scalar_t>();
        const AscendC::LocalTensor<float> query = floatBuf_.Get<float>();
        const AscendC::LocalTensor<half> operand = halfBuf_.Get<half>();
        const AscendC::LocalTensor<float> signs = signBuf_.Get<float>();

        AscendC::DataCopy(queryIn, queryGm_[firstElem], elems);
        SyncMte2ToVector();
        AscendC::Cast(query, queryIn, AscendC::RoundMode::CAST_NONE, elems);
        for (uint32_t vector = 0; vector < own; ++vector) {
            AscendC::Mul(query[vector * headSize_], query[vector * headSize_], signs, headSize_);
        }
        AscendC::Cast(operand, query, AscendC::RoundMode::CAST_RINT, elems);
        SyncVectorToMte3();
        AscendC::DataCopy(tileL1Buf_.Get<half>()[subcore * elems], operand, elems);
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_MTE3>(kFlagOperandsReady);

        AscendC::CrossCoreWaitFlag(kFlagProductReady);
        AscendC::LocalTensor<float> src = pairBuf_.Get<float>();
        AscendC::LocalTensor<float> dst = query;
        for (uint32_t stride = kRotateQTile; stride < headSize_; stride <<= 1) {
            RepeatButterfly(dst, src, stride, elems / (2 * stride));
            // FWHT ping-pong: this pass's destination is the next pass's source over the same two buffers.
            AscendC::PipeBarrier<PIPE_V>();
            const AscendC::LocalTensor<float> hold = src;
            src = dst;
            dst = hold;
        }
        AscendC::Muls(src, src, invSqrtLen_, elems);
        for (uint32_t vector = 0; vector < own; ++vector) {
            AscendC::Mul(src[vector * headSize_], src[vector * headSize_], signs, headSize_);
        }

        SyncVectorToMte3();
        AscendC::DataCopy(queryRotGm_[firstElem], src, elems);
        SyncEvent<AscendC::HardEvent::MTE3_MTE2>();
        SyncMte3ToVector();
    }

    // One chunk's AIC half: the whole tile times H16, each half Fixpiped into its own subcore's UB.
    __aicore__ inline void ComputeTileProduct()
    {
        const uint32_t rows = tileElems_ / kRotateQTile;

        AscendC::CrossCoreWaitFlag(kFlagOperandsReady);
        SyncEvent<AscendC::HardEvent::M_MTE1>();
        StageTile(rows);
        SyncMte1ToMatrix();

        SyncEvent<AscendC::HardEvent::FIX_M>();
        const AscendC::LocalTensor<float> product = productL0cBuf_.Get<float>();
        AscendC::Mmad(product, tileL0aBuf_.Get<half>(), h16L0bBuf_.Get<half>(),
                      AscendC::MmadParams(static_cast<uint16_t>(rows), static_cast<uint16_t>(kRotateQTile),
                                          static_cast<uint16_t>(kRotateQTile), 0, false, true));
        SyncMatrixToFixpipe();

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixParams(
            static_cast<uint16_t>(kRotateQTile), static_cast<uint16_t>(rows), static_cast<uint16_t>(rows),
            kRotateQTile);
        fixParams.dualDstCtl = kDualDstSplitM;
        fixParams.subBlockId = false;
        AscendC::Fixpipe<float, float, kFixpipeToUb>(pairBuf_.Get<float>(), product, fixParams);
        // Drains the Fixpipe before the product-ready flag releases the vector subcores onto their UB rows.
        AscendC::PipeBarrier<PIPE_FIX>();
        AscendC::CrossCoreSetFlag<kSubBlockSyncMode, PIPE_FIX>(kFlagProductReady);
    }

    __aicore__ inline void StageTile(const uint32_t rows)
    {
        AscendC::LoadData2DParamsV2 tile;
        tile.mStartPosition = 0;
        tile.kStartPosition = 0;
        tile.mStep = CeilDivU16(rows, kRotateQTile);
        tile.kStep = CeilDivU16(kRotateQTile, kHalfOperandC0);
        tile.srcStride = CeilDivU16(rows, kRotateQTile);
        tile.dstStride = CeilDivU16(rows, kRotateQTile);
        tile.ifTranspose = false;
        AscendC::LoadData(tileL0aBuf_.Get<half>(), tileL1Buf_.Get<half>(), tile);

        AscendC::LoadData2DParamsV2 h16;
        h16.mStartPosition = 0;
        h16.kStartPosition = 0;
        h16.mStep = CeilDivU16(kRotateQTile, kRotateQTile);
        h16.kStep = CeilDivU16(kRotateQTile, kHalfOperandC0);
        h16.srcStride = CeilDivU16(kRotateQTile, kRotateQTile);
        h16.dstStride = CeilDivU16(kRotateQTile, kRotateQTile);
        h16.ifTranspose = false;
        AscendC::LoadData(h16L0bBuf_.Get<half>(), h16L1Buf_.Get<half>(), h16);
    }

    // One vector on this subcore alone: read, ApplyPi, write.
    __aicore__ inline void RotateVector(const uint32_t vector)
    {
        const AscendC::LocalTensor<scalar_t> queryIn = inBuf_.Get<scalar_t>();
        const AscendC::LocalTensor<float> x = floatBuf_.Get<float>();
        const AscendC::LocalTensor<float> tmp = pairBuf_.Get<float>();
        const uint64_t elem = static_cast<uint64_t>(vector) * headSize_;

        AscendC::DataCopy(queryIn, queryGm_[elem], headSize_);
        SyncMte2ToVector();
        AscendC::Cast(x, queryIn, AscendC::RoundMode::CAST_NONE, headSize_);

        codec_.ApplyPi(x, tmp, signBuf_.Get<float>(), headSize_);

        SyncVectorToMte3();
        AscendC::DataCopy(queryRotGm_[elem], x, headSize_);
        SyncEvent<AscendC::HardEvent::MTE3_MTE2>();
        SyncMte3ToVector();
    }

    TurboQuantCodec4 codec_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> inBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> floatBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> halfBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> pairBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> signBuf_;
    AscendC::TBuf<AscendC::TPosition::A1> tileL1Buf_;
    AscendC::TBuf<AscendC::TPosition::B1> h16L1Buf_;
    AscendC::TBuf<AscendC::TPosition::A2> tileL0aBuf_;
    AscendC::TBuf<AscendC::TPosition::B2> h16L0bBuf_;
    AscendC::TBuf<AscendC::TPosition::CO1> productL0cBuf_;
    AscendC::GlobalTensor<scalar_t> queryGm_;
    AscendC::GlobalTensor<float> piSignsGm_;
    AscendC::GlobalTensor<int32_t> rotTablesGm_;
    AscendC::GlobalTensor<half> h16Gm_;
    AscendC::GlobalTensor<scalar_t> gateGm_;
    AscendC::GlobalTensor<float> queryRotGm_;
    uint32_t numHeads_ = 0;
    uint32_t headSize_ = 0;
    uint32_t cubeChunk_ = 0;
    uint32_t tileElems_ = 0;
    uint32_t outputStage_ = 0;
    float invSqrtLen_ = 1.0f;
};

// A pre-rotated decode: no rotation buffer, no rotation instruction, and an output left in the rotated basis.
template <typename scalar_t>
class TurboQuantQueryBasis<scalar_t, false> : public TurboQuantRotatedOutput {
public:
    static constexpr bool kEnabled = false;
};

}
}

#endif
