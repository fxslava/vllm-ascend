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

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_CODEC_950_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_CODEC_950_H

#include "turboquant_common.h"

namespace vllm_ascend {
namespace turboquant {

template <int BITS>
class TurboQuantCodec {
    static_assert(BITS == 4, "TurboQuantCodec is only implemented for b = 4");

public:
    static constexpr int kBits = BITS;
    static constexpr int kLevels = 1 << BITS;
    static constexpr uint32_t kPackFactor = 8 / BITS;
    static constexpr float kLevelMax = static_cast<float>(kLevels - 1);
    static constexpr float kPackHigh = static_cast<float>(kLevels);
    static constexpr float kInt8Bias = 128.0f;
    static constexpr float kEps = 1e-20f;
    static constexpr int kEarlyStages = 3;
    static constexpr int kThresholdCount = kLevels - 1;
    static constexpr float kCentroidStride = static_cast<float>(sizeof(float));
    static constexpr uint32_t kSignBitShift = 31;
    static constexpr int kBinLanes = 8;

    __aicore__ static inline float Threshold(int i)
    {
        constexpr float kThresholds[kThresholdCount] = {
            -2.4008033987632817f,  -1.8435318062766393f,  -1.4371387916845330f, -1.0992858269170722f,
            -0.7995497875097155f,  -0.5224037090113789f,  -0.2582216646707196f, 0.0f,
            0.2582216646707196f,   0.5224037090113789f,   0.7995497875097155f,  1.0992858269170722f,
            1.4371387916845330f,   1.8435318062766393f,   2.4008033987632817f};
        return kThresholds[i];
    }

    __aicore__ static inline uint32_t ConstTableWords(uint32_t vecLen, uint32_t batchRows)
    {
        return 7u * vecLen + 2u * vecLen * batchRows + static_cast<uint32_t>(kLevels);
    }

    __aicore__ static inline uint32_t WorkBufferWords(uint32_t vecLen, uint32_t batchRows)
    {
        return 2u * vecLen * batchRows + vecLen + kBrcbDstLanes + kFp32PerBlock +
               static_cast<uint32_t>(kBinLanes) * vecLen;
    }

    __aicore__ inline void Init(AscendC::TPipe *pipe, const uint32_t vecLen, const uint32_t batchRows,
                                const float invSqrtLen, const AscendC::GlobalTensor<int32_t> &tablesGm)
    {
        len_ = vecLen;
        batchLen_ = vecLen * batchRows;
        invSqrtLen_ = invSqrtLen;
        const uint32_t packed = len_ / kPackFactor;
        const uint32_t constWords = ConstTableWords(vecLen, batchRows);

        pipe->InitBuffer(constBuf_, constWords * sizeof(int32_t));
        pipe->InitBuffer(workBuf_, WorkBufferWords(vecLen, batchRows) * sizeof(float));

        const AscendC::LocalTensor<int32_t> pool = constBuf_.Get<int32_t>();
        AscendC::DataCopy(pool, tablesGm, constWords);
        // Init hand-off: the GM tables land in UB before any vector op reads them.
        AscendC::PipeBarrier<PIPE_ALL>();

        const AscendC::LocalTensor<float> poolFloat = pool.template ReinterpretCast<float>();
        uint32_t offset = 0;
        for (int stage = 0; stage < kEarlyStages; ++stage) {
            sign_[stage] = poolFloat[offset];
            offset += len_;
            xorOffset_[stage] = pool[offset].template ReinterpretCast<uint32_t>();
            offset += len_;
        }
        evenOffset_ = pool[offset].template ReinterpretCast<uint32_t>();
        offset += packed;
        oddOffset_ = pool[offset].template ReinterpretCast<uint32_t>();
        offset += packed;
        expandOffset_ = pool[offset].template ReinterpretCast<uint32_t>();
        offset += batchLen_;
        oddSelect_ = poolFloat[offset];
        offset += batchLen_;
        centroid_ = poolFloat[offset];

        const AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        offset = 0;
        swap_ = work[offset];
        offset += batchLen_;
        scratch_ = work[offset];
        offset += batchLen_;
        reduceWork_ = work[offset];
        offset += len_;
        broadcast_ = work[offset];
        offset += kBrcbDstLanes + kFp32PerBlock;
        for (int lane = 0; lane < kBinLanes; ++lane) {
            binLane_[lane] = work[offset];
            offset += len_;
        }
    }

    __aicore__ inline void FastWalshHadamardTransform(const AscendC::LocalTensor<float> &x,
                                                      const AscendC::LocalTensor<float> &tmp, const uint32_t n)
    {
        for (int stage = 0; stage < kEarlyStages; ++stage) {
            const uint32_t stride = 1u << stage;
            if (stride >= n) {
                return;
            }
            ShuffleStage(x, tmp, stage, n);
        }

        AscendC::LocalTensor<float> src = x;
        AscendC::LocalTensor<float> dst = tmp;
        bool inTmp = false;
        for (uint32_t stride = kFp32PerBlock; stride < n; stride <<= 1) {
            RepeatButterfly(dst, src, stride, n / (2 * stride));
            // FWHT ping-pong: this pass's destination is the next pass's source over the same two buffers.
            AscendC::PipeBarrier<PIPE_V>();
            const AscendC::LocalTensor<float> hold = src;
            src = dst;
            dst = hold;
            inTmp = !inTmp;
        }
        if (inTmp) {
            AscendC::DataCopy(x, src, n);
            // FWHT ping-pong: x is scaled in place right after the result is copied back into it.
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Muls(x, x, invSqrtLen_, n);
    }

    __aicore__ inline void ApplyPi(const AscendC::LocalTensor<float> &x, const AscendC::LocalTensor<float> &tmp,
                                   const AscendC::LocalTensor<float> &piSigns, const uint32_t n)
    {
        AscendC::Mul(x, x, piSigns, n);
        FastWalshHadamardTransform(x, tmp, n);
        AscendC::Mul(x, x, piSigns, n);
    }

    __aicore__ inline void Quantize4Bit(const AscendC::LocalTensor<int8_t> &dstPacked,
                                        const AscendC::LocalTensor<float> &src,
                                        const AscendC::LocalTensor<float> &scaleOut, const uint32_t n)
    {
        ComputeInverseScale(src, scaleOut, n);
        BroadcastMul(scratch_, src, broadcast_[kBrcbDstLanes], n);
        // Reinterpretation: reduceWork_ held the ReduceSum's float intermediates and is written as int32 bins next.
        AscendC::PipeBarrier<PIPE_V>();
        ComputeLevelBins(n);
        StagePackedBytes(dstPacked, n / kPackFactor);
    }

    __aicore__ inline void Dequantize4Bit(const AscendC::LocalTensor<float> &dst,
                                          const AscendC::LocalTensor<int8_t> &srcPacked, const uint32_t rows,
                                          const uint32_t len)
    {
        const uint32_t n = rows * len;
        ComputeLevelDigits(dst, srcPacked, n, n / kPackFactor);
        ComputeCentroids(dst, n);
    }

    __aicore__ static inline void BroadcastMul(const AscendC::LocalTensor<float> &dst,
                                               const AscendC::LocalTensor<float> &src,
                                               const AscendC::LocalTensor<float> &scalarBlock, const uint32_t count)
    {
        constexpr uint8_t kRepBlocks = static_cast<uint8_t>(kFp32PerRepeat / kFp32PerBlock);
        const uint32_t repeats = count / kFp32PerRepeat;
        if (repeats > 0) {
            AscendC::Mul(dst, src, scalarBlock, static_cast<uint64_t>(kFp32PerRepeat), static_cast<uint8_t>(repeats),
                         {1, 1, 0, kRepBlocks, kRepBlocks, 0});
        }
        const uint32_t tail = count - repeats * kFp32PerRepeat;
        if (tail > 0) {
            const uint32_t base = repeats * kFp32PerRepeat;
            AscendC::Mul(dst[base], src[base], scalarBlock, static_cast<uint64_t>(tail), 1, {1, 1, 0, 0, 0, 0});
        }
    }

private:
    // scaleOut = RMS(src) + eps, and broadcast_[kBrcbDstLanes] = -1 / scaleOut as one block.
    __aicore__ inline void ComputeInverseScale(const AscendC::LocalTensor<float> &src,
                                               const AscendC::LocalTensor<float> &scaleOut, const uint32_t n)
    {
        AscendC::Mul(scratch_, src, src, n);
        AscendC::ReduceSum<float>(scaleOut, scratch_, reduceWork_, n);
        AscendC::Sqrt(scaleOut, scaleOut, 1);
        AscendC::Muls(scaleOut, scaleOut, invSqrtLen_, 1);
        AscendC::Adds(scaleOut, scaleOut, kEps, 1);

        AscendC::Brcb(broadcast_, scaleOut, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
        AscendC::Duplicate(broadcast_[kBrcbDstLanes], -1.0f, kFp32PerBlock);
        AscendC::Div(broadcast_[kBrcbDstLanes], broadcast_[kBrcbDstLanes], broadcast_, kFp32PerBlock);
    }

    // The level of every coordinate of scratch_ against the thresholds, counted as int32 bins in reduceWork_
    // and cast back to float into scratch_.
    __aicore__ inline void ComputeLevelBins(const uint32_t n)
    {
        const AscendC::LocalTensor<int32_t> bins = reduceWork_.ReinterpretCast<int32_t>();
        AscendC::Duplicate(bins, 0, n);

        for (int base = 0; base < kThresholdCount; base += kBinLanes) {
            const int lanes = (kThresholdCount - base) < kBinLanes ? (kThresholdCount - base) : kBinLanes;

            for (int lane = 0; lane < lanes; ++lane) {
                AscendC::Adds(binLane_[lane], scratch_, Threshold(base + lane), n);
            }
            // Reinterpretation: the bin lanes were just written as float and are shifted as uint32 next.
            AscendC::PipeBarrier<PIPE_V>();

            for (int lane = 0; lane < lanes; ++lane) {
                const AscendC::LocalTensor<uint32_t> signBits = binLane_[lane].ReinterpretCast<uint32_t>();
                AscendC::ShiftRight(signBits, signBits, kSignBitShift, static_cast<int32_t>(n));
            }
            // Reinterpretation: the bin lanes were just written as uint32 and are summed as int32 next.
            AscendC::PipeBarrier<PIPE_V>();

            for (int span = 1; span < lanes; span <<= 1) {
                for (int lane = 0; lane + span < lanes; lane += 2 * span) {
                    const AscendC::LocalTensor<int32_t> sum = binLane_[lane].ReinterpretCast<int32_t>();
                    const AscendC::LocalTensor<int32_t> addend = binLane_[lane + span].ReinterpretCast<int32_t>();
                    AscendC::Add(sum, sum, addend, n);
                }
            }
            AscendC::Add(bins, bins, binLane_[0].ReinterpretCast<int32_t>(), n);
            // Reinterpretation: the bin lanes were last read as int32 and the next pass writes them as float.
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Cast(scratch_, bins, AscendC::RoundMode::CAST_NONE, n);
    }

    // Even and odd levels folded into one byte each, biased to int8 and cast into the packed output.
    __aicore__ inline void StagePackedBytes(const AscendC::LocalTensor<int8_t> &dstPacked, const uint32_t packed)
    {
        AscendC::Gather(swap_, scratch_, evenOffset_, kGatherSrcBase, packed);
        AscendC::Gather(swap_[packed], scratch_, oddOffset_, kGatherSrcBase, packed);
        AscendC::Muls(swap_[packed], swap_[packed], kPackHigh, packed);
        AscendC::Add(swap_, swap_, swap_[packed], packed);
        AscendC::Adds(swap_, swap_, -kInt8Bias, packed);
        // Reinterpretation: scratch_ was read as float by the gathers and is written as half next.
        AscendC::PipeBarrier<PIPE_V>();

        const AscendC::LocalTensor<half> halfView = scratch_.ReinterpretCast<half>();
        AscendC::Cast(halfView, swap_, AscendC::RoundMode::CAST_NONE, packed);
        AscendC::Cast(dstPacked, halfView, AscendC::RoundMode::CAST_RINT, packed);
        // Reinterpretation: scratch_ holds half levels, and the next vector's encode writes it as float.
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Expands every packed byte into its even and odd 4-bit levels, one float per coordinate, into dst.
    __aicore__ inline void ComputeLevelDigits(const AscendC::LocalTensor<float> &dst,
                                              const AscendC::LocalTensor<int8_t> &srcPacked, const uint32_t n,
                                              const uint32_t packed)
    {
        const AscendC::LocalTensor<half> halfView = swap_.ReinterpretCast<half>();
        AscendC::Cast(halfView, srcPacked, AscendC::RoundMode::CAST_NONE, packed);
        AscendC::Cast(scratch_, halfView, AscendC::RoundMode::CAST_NONE, packed);
        AscendC::Adds(scratch_, scratch_, kInt8Bias, packed);
        // Reinterpretation: swap_ was read as half above and the Gather writes it as float.
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Gather(swap_, scratch_, expandOffset_, kGatherSrcBase, n);

        AscendC::Muls(dst, swap_, 1.0f / kPackHigh, n);
        FloorInPlace(dst, n);
        AscendC::Muls(scratch_, dst, -kPackHigh, n);
        AscendC::Add(scratch_, scratch_, swap_, n);

        AscendC::Sub(dst, dst, scratch_, n);
        AscendC::Mul(dst, dst, oddSelect_, n);
        AscendC::Add(dst, dst, scratch_, n);
    }

    // dst holds levels; replaces each with its centroid through a Gather of the centroid table.
    __aicore__ inline void ComputeCentroids(const AscendC::LocalTensor<float> &dst, const uint32_t n)
    {
        AscendC::Muls(scratch_, dst, kCentroidStride, n);
        // Reinterpretation: swap_ was last read as float and is written as int32 offsets next.
        AscendC::PipeBarrier<PIPE_V>();
        const AscendC::LocalTensor<int32_t> offsets = swap_.ReinterpretCast<int32_t>();
        AscendC::Cast(offsets, scratch_, AscendC::RoundMode::CAST_RINT, n);
        // Reinterpretation: the offsets were just written as int32 and the Gather reads them as uint32.
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(dst, centroid_, swap_.ReinterpretCast<uint32_t>(), kGatherSrcBase, n);
        // Reinterpretation: swap_ was read as uint32 offsets, and the next plane's expand writes it as half.
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ShuffleStage(const AscendC::LocalTensor<float> &x, const AscendC::LocalTensor<float> &tmp,
                                        const int stage, const uint32_t n)
    {
        AscendC::Gather(swap_, x, xorOffset_[stage], kGatherSrcBase, n);
        AscendC::Mul(tmp, x, sign_[stage], n);
        AscendC::Add(x, swap_, tmp, n);
    }

    __aicore__ static inline void FloorInPlace(const AscendC::LocalTensor<float> &x, const uint32_t count)
    {
        const AscendC::LocalTensor<int32_t> intView = x.ReinterpretCast<int32_t>();
        // Reinterpretation: x was last written as float and is overwritten in place through an int32 view.
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(intView, x, AscendC::RoundMode::CAST_FLOOR, count);
        // Reinterpretation: the int32 view is cast back over the same lanes as float.
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(x, intView, AscendC::RoundMode::CAST_NONE, count);
    }

    AscendC::TBuf<AscendC::QuePosition::VECCALC> constBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> workBuf_;
    AscendC::LocalTensor<float> sign_[kEarlyStages];
    AscendC::LocalTensor<uint32_t> xorOffset_[kEarlyStages];
    AscendC::LocalTensor<uint32_t> evenOffset_;
    AscendC::LocalTensor<uint32_t> oddOffset_;
    AscendC::LocalTensor<uint32_t> expandOffset_;
    AscendC::LocalTensor<float> oddSelect_;
    AscendC::LocalTensor<float> centroid_;
    AscendC::LocalTensor<float> swap_;
    AscendC::LocalTensor<float> scratch_;
    AscendC::LocalTensor<float> reduceWork_;
    AscendC::LocalTensor<float> broadcast_;
    AscendC::LocalTensor<float> binLane_[kBinLanes];
    uint32_t len_ = 0;
    uint32_t batchLen_ = 0;
    float invSqrtLen_ = 1.0f;
};

using TurboQuantCodec4 = TurboQuantCodec<4>;

}
}

#endif
