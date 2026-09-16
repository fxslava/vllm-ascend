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

    __aicore__ inline void Init(AscendC::TPipe *pipe, uint32_t vecLen, uint32_t batchRows, float invSqrtLen,
                                const AscendC::GlobalTensor<int32_t> &tablesGm)
    {
        len_ = vecLen;
        batchLen_ = vecLen * batchRows;
        invSqrtLen_ = invSqrtLen;
        const uint32_t packed = len_ / kPackFactor;
        const uint32_t constWords = ConstTableWords(vecLen, batchRows);

        pipe->InitBuffer(constBuf_, constWords * sizeof(int32_t));
        pipe->InitBuffer(workBuf_, WorkBufferWords(vecLen, batchRows) * sizeof(float));

        AscendC::LocalTensor<int32_t> pool = constBuf_.Get<int32_t>();
        AscendC::DataCopy(pool, tablesGm, constWords);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::LocalTensor<float> poolF = pool.template ReinterpretCast<float>();
        uint32_t off = 0;
        for (int stage = 0; stage < kEarlyStages; ++stage) {
            sign_[stage] = poolF[off];
            off += len_;
            xorOffset_[stage] = pool[off].template ReinterpretCast<uint32_t>();
            off += len_;
        }
        evenOffset_ = pool[off].template ReinterpretCast<uint32_t>();
        off += packed;
        oddOffset_ = pool[off].template ReinterpretCast<uint32_t>();
        off += packed;
        expandOffset_ = pool[off].template ReinterpretCast<uint32_t>();
        off += batchLen_;
        oddSelect_ = poolF[off];
        off += batchLen_;
        centroid_ = poolF[off];

        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        off = 0;
        swap_ = work[off];
        off += batchLen_;
        scratch_ = work[off];
        off += batchLen_;
        reduceWork_ = work[off];
        off += len_;
        broadcast_ = work[off];
        off += kBrcbDstLanes + kFp32PerBlock;
        for (int lane = 0; lane < kBinLanes; ++lane) {
            binLane_[lane] = work[off];
            off += len_;
        }
    }

    template <bool VEC_BARRIERS = true>
    __aicore__ inline void FastWalshHadamardTransform(AscendC::LocalTensor<float> &x,
                                                      AscendC::LocalTensor<float> &tmp, int len)
    {
        const uint32_t n = static_cast<uint32_t>(len);

        for (int stage = 0; stage < kEarlyStages; ++stage) {
            const uint32_t stride = 1u << stage;
            if (stride >= n) {
                return;
            }
            ShuffleStage<VEC_BARRIERS>(x, tmp, stage, n);
        }

        AscendC::LocalTensor<float> src = x;
        AscendC::LocalTensor<float> dst = tmp;
        bool inTmp = false;
        for (uint32_t stride = kFp32PerBlock; stride < n; stride <<= 1) {
            RepeatButterfly(dst, src, stride, n / (2 * stride));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<float> hold = src;
            src = dst;
            dst = hold;
            inTmp = !inTmp;
        }
        if (inTmp) {
            AscendC::DataCopy(x, src, n);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Muls(x, x, invSqrtLen_, n);
        VecBarrier<VEC_BARRIERS>();
    }

    template <bool VEC_BARRIERS = true>
    __aicore__ inline void ApplyPi(AscendC::LocalTensor<float> &x, AscendC::LocalTensor<float> &tmp,
                                   const AscendC::LocalTensor<float> &piSigns, int len)
    {
        const uint32_t n = static_cast<uint32_t>(len);
        AscendC::Mul(x, x, piSigns, n);
        VecBarrier<VEC_BARRIERS>();
        FastWalshHadamardTransform<VEC_BARRIERS>(x, tmp, len);
        AscendC::Mul(x, x, piSigns, n);
        VecBarrier<VEC_BARRIERS>();
    }

    __aicore__ inline void Quantize4Bit(const AscendC::LocalTensor<int8_t> &dstPacked,
                                        const AscendC::LocalTensor<float> &src,
                                        const AscendC::LocalTensor<float> &scaleOut, int len)
    {
        const uint32_t n = static_cast<uint32_t>(len);
        const uint32_t packed = n / kPackFactor;

        AscendC::Mul(scratch_, src, src, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float>(scaleOut, scratch_, reduceWork_, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(scaleOut, scaleOut, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(scaleOut, scaleOut, invSqrtLen_, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(scaleOut, scaleOut, kEps, 1);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Brcb(broadcast_, scaleOut, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(broadcast_[kBrcbDstLanes], -1.0f, kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(broadcast_[kBrcbDstLanes], broadcast_[kBrcbDstLanes], broadcast_, kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();

        BroadcastMul(scratch_, src, broadcast_[kBrcbDstLanes], n);

        AscendC::LocalTensor<int32_t> bins = reduceWork_.ReinterpretCast<int32_t>();
        AscendC::Duplicate(bins, 0, n);
        AscendC::PipeBarrier<PIPE_V>();

        for (int base = 0; base < kThresholdCount; base += kBinLanes) {
            const int lanes =
                (kThresholdCount - base) < kBinLanes ? (kThresholdCount - base) : kBinLanes;

            for (int lane = 0; lane < lanes; ++lane) {
                AscendC::Adds(binLane_[lane], scratch_, Threshold(base + lane), n);
            }
            AscendC::PipeBarrier<PIPE_V>();

            for (int lane = 0; lane < lanes; ++lane) {
                AscendC::LocalTensor<uint32_t> bits = binLane_[lane].ReinterpretCast<uint32_t>();
                AscendC::ShiftRight(bits, bits, kSignBitShift, static_cast<int32_t>(n));
            }
            AscendC::PipeBarrier<PIPE_V>();

            for (int span = 1; span < lanes; span <<= 1) {
                for (int lane = 0; lane + span < lanes; lane += 2 * span) {
                    AscendC::LocalTensor<int32_t> dst = binLane_[lane].ReinterpretCast<int32_t>();
                    AscendC::LocalTensor<int32_t> src2 =
                        binLane_[lane + span].ReinterpretCast<int32_t>();
                    AscendC::Add(dst, dst, src2, n);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Add(bins, bins, binLane_[0].ReinterpretCast<int32_t>(), n);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Cast(scratch_, bins, AscendC::RoundMode::CAST_NONE, n);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Gather(swap_, scratch_, evenOffset_, kGatherSrcBase, packed);
        AscendC::Gather(swap_[packed], scratch_, oddOffset_, kGatherSrcBase, packed);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(swap_[packed], swap_[packed], kPackHigh, packed);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(swap_, swap_, swap_[packed], packed);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(swap_, swap_, -kInt8Bias, packed);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<half> halfView = scratch_.ReinterpretCast<half>();
        AscendC::Cast(halfView, swap_, AscendC::RoundMode::CAST_NONE, packed);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(dstPacked, halfView, AscendC::RoundMode::CAST_RINT, packed);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Dequantize4Bit(const AscendC::LocalTensor<float> &dst,
                                          const AscendC::LocalTensor<int8_t> &srcPacked, int rows, int len)
    {
        const uint32_t n = static_cast<uint32_t>(rows) * static_cast<uint32_t>(len);
        const uint32_t packed = n / kPackFactor;

        AscendC::LocalTensor<half> halfView = swap_.ReinterpretCast<half>();
        AscendC::Cast(halfView, srcPacked, AscendC::RoundMode::CAST_NONE, packed);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(scratch_, halfView, AscendC::RoundMode::CAST_NONE, packed);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(scratch_, scratch_, kInt8Bias, packed);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Gather(swap_, scratch_, expandOffset_, kGatherSrcBase, n);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Muls(dst, swap_, 1.0f / kPackHigh, n);
        AscendC::PipeBarrier<PIPE_V>();
        FloorInPlace(dst, n);
        AscendC::Muls(scratch_, dst, -kPackHigh, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(scratch_, scratch_, swap_, n);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub(dst, dst, scratch_, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(dst, dst, oddSelect_, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(dst, dst, scratch_, n);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Muls(scratch_, dst, kCentroidStride, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<int32_t> offsets = swap_.ReinterpretCast<int32_t>();
        AscendC::Cast(offsets, scratch_, AscendC::RoundMode::CAST_RINT, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(dst, centroid_, swap_.ReinterpretCast<uint32_t>(), kGatherSrcBase, n);
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <bool VEC_BARRIERS = true>
    __aicore__ static inline void BroadcastMul(const AscendC::LocalTensor<float> &dst,
                                               const AscendC::LocalTensor<float> &src,
                                               const AscendC::LocalTensor<float> &scalarBlock, uint32_t count)
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
        VecBarrier<VEC_BARRIERS>();
    }

private:
    template <bool VEC_BARRIERS>
    __aicore__ inline void ShuffleStage(AscendC::LocalTensor<float> &x, AscendC::LocalTensor<float> &tmp, int stage,
                                        uint32_t n)
    {
        AscendC::Gather(swap_, x, xorOffset_[stage], kGatherSrcBase, n);
        AscendC::Mul(tmp, x, sign_[stage], n);
        VecBarrier<VEC_BARRIERS>();
        AscendC::Add(x, swap_, tmp, n);
        VecBarrier<VEC_BARRIERS>();
    }

    __aicore__ static inline void FloorInPlace(const AscendC::LocalTensor<float> &x, uint32_t count)
    {
        AscendC::LocalTensor<int32_t> intView = x.ReinterpretCast<int32_t>();
        AscendC::Cast(intView, x, AscendC::RoundMode::CAST_FLOOR, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(x, intView, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
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
