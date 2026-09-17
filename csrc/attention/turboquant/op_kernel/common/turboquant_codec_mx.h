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

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_CODEC_MX_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_CODEC_MX_H

#include "kernel_operator.h"
#include "turboquant_codec_950.h"
#include "turboquant_mode.h"

namespace vllm_ascend {
namespace turboquant {

template <TurboQuantMode MODE>
struct TurboQuantPlanes;

template <>
struct TurboQuantPlanes<TurboQuantMode::KV3_FP4> {
    static constexpr int32_t kLowBits = 2;
    static constexpr int32_t kMsbBits = 1;
    static constexpr int32_t kLowRadix = 4;
    static constexpr int32_t kLowPerByte = 4;
};

template <>
struct TurboQuantPlanes<TurboQuantMode::KV4_FP8> {
    static constexpr int32_t kLowBits = 4;
    static constexpr int32_t kMsbBits = 0;
    static constexpr int32_t kLowRadix = 16;
    static constexpr int32_t kLowPerByte = 2;
};

template <>
struct TurboQuantPlanes<TurboQuantMode::KV5_FP8> {
    static constexpr int32_t kLowBits = 4;
    static constexpr int32_t kMsbBits = 1;
    static constexpr int32_t kLowRadix = 16;
    static constexpr int32_t kLowPerByte = 2;
};

constexpr int32_t kMsbPerByte = 8;

template <TurboQuantMode MODE>
class TurboQuantModeCodec {
public:
    using Traits = TurboQuantModeTraits<MODE>;
    using Planes = TurboQuantPlanes<MODE>;

    static constexpr int32_t kBits = Traits::kBits;
    static constexpr int32_t kLevels = Traits::kLevels;
    static constexpr int32_t kThresholdCount = kLevels - 1;
    static constexpr bool kHasMsbPlane = Planes::kMsbBits > 0;
    static constexpr bool kIsFp4 = Traits::kOperand == TurboQuantOperand::kFp4E2m1;
    static constexpr int kBinLanes = TurboQuantCodec4::kBinLanes;
    static constexpr bool kIsAffine = Traits::kIsAffine;
    static constexpr float kAffineBias = Traits::kAffineBias;

    __aicore__ static constexpr uint32_t LowPlaneBytes(uint32_t vecLen)
    {
        return vecLen / static_cast<uint32_t>(Planes::kLowPerByte);
    }
    __aicore__ static constexpr uint32_t MsbPlaneBytes(uint32_t vecLen)
    {
        return kHasMsbPlane ? vecLen / static_cast<uint32_t>(kMsbPerByte) : 0u;
    }
    __aicore__ static constexpr uint32_t PackedBytes(uint32_t vecLen)
    {
        return LowPlaneBytes(vecLen) + MsbPlaneBytes(vecLen);
    }

    __aicore__ static constexpr uint32_t OperandElems(uint32_t vecLen)
    {
        return kIsFp4 ? vecLen / 2 : vecLen;
    }

    __aicore__ static inline uint32_t ConstTableWords(uint32_t vecLen, uint32_t batchRows)
    {
        if constexpr (kIsAffine) {
            return 0u;
        } else {
            return 2u * vecLen * batchRows + kPeriodicWords + vecLen + static_cast<uint32_t>(kLevels);
        }
    }

    __aicore__ static inline uint32_t WorkBufferWords(uint32_t vecLen, uint32_t batchRows)
    {
        const uint32_t shared = kBrcbDstLanes + kFp32PerBlock + static_cast<uint32_t>(kBinLanes) * vecLen;
        if constexpr (kIsAffine) {
            return vecLen * batchRows + 3u * vecLen + shared;
        } else {
            return 4u * vecLen * batchRows + vecLen + shared;
        }
    }

    __aicore__ static inline float Threshold(int i) { return Traits::kThresholds[i]; }

    __aicore__ inline void Init(AscendC::TPipe *pipe, const uint32_t vecLen, const uint32_t batchRows,
                                const float invSqrtLen, const AscendC::GlobalTensor<int32_t> &tablesGm)
    {
        len_ = vecLen;
        rows_ = batchRows;
        batchLen_ = vecLen * batchRows;
        invSqrtLen_ = invSqrtLen;

        pipe->InitBuffer(workBuf_, WorkBufferWords(vecLen, batchRows) * sizeof(float));

        if constexpr (!kIsAffine) {
            const uint32_t constWords = ConstTableWords(vecLen, batchRows);
            pipe->InitBuffer(constBuf_, constWords * sizeof(int32_t));

            const AscendC::LocalTensor<int32_t> pool = constBuf_.Get<int32_t>();
            AscendC::DataCopy(pool, tablesGm, constWords);
            // Init hand-off: the GM tables land in UB before any vector op reads them.
            AscendC::PipeBarrier<PIPE_ALL>();

            const AscendC::LocalTensor<float> poolFloat = pool.template ReinterpretCast<float>();
            uint32_t constOffset = 0;
            lowOffset_ = pool[constOffset].template ReinterpretCast<uint32_t>();
            constOffset += batchLen_;
            msbOffset_ = pool[constOffset].template ReinterpretCast<uint32_t>();
            constOffset += batchLen_;
            lowRecip_ = poolFloat[constOffset];
            constOffset += kFp32PerBlock;
            msbRecip_ = poolFloat[constOffset];
            constOffset += kFp32PerBlock;
            msbWeight_ = poolFloat[constOffset];
            constOffset += kFp32PerBlock;
            packOffset_ = pool[constOffset].template ReinterpretCast<uint32_t>();
            constOffset += len_;
            centroid_ = poolFloat[constOffset];
        }

        const AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        uint32_t offset = 0;
        if constexpr (kIsAffine) {
            expand_ = work[offset];
            offset += batchLen_ / 2u;
            msb_ = work[offset];
            offset += batchLen_ / 2u;
            low_ = work[offset];
            offset += len_;
            scratch_ = work[offset];
            offset += len_;
        } else {
            expand_ = work[offset];
            offset += batchLen_;
            low_ = work[offset];
            offset += batchLen_;
            msb_ = work[offset];
            offset += batchLen_;
            scratch_ = work[offset];
            offset += batchLen_;
        }
        reduceWork_ = work[offset];
        offset += len_;
        broadcast_ = work[offset];
        offset += kBrcbDstLanes + kFp32PerBlock;
        for (int lane = 0; lane < kBinLanes; ++lane) {
            binLane_[lane] = work[offset];
            offset += len_;
        }
    }

    __aicore__ inline void Encode(const AscendC::LocalTensor<int8_t> &dstPacked,
                                  const AscendC::LocalTensor<float> &src,
                                  const AscendC::LocalTensor<float> &scaleOut, const uint32_t n)
    {
        ComputeInverseScale(src, scaleOut, n);
        TurboQuantCodec4::BroadcastMul(scratch_, src, broadcast_[kBrcbDstLanes], n);
        // Reinterpretation: reduceWork_ held the ReduceSum's float intermediates and is written as int32 bins next.
        AscendC::PipeBarrier<PIPE_V>();
        ComputeLevelBins(n);

        if constexpr (kHasMsbPlane) {
            ComputeMsbDigits(n);
        }

        if constexpr (kIsAffine) {
            StageAffinePlane(dstPacked, low_, n);
        } else {
            StageLowPlane(dstPacked, low_, n);
            if constexpr (kHasMsbPlane) {
                StageMsbPlane(dstPacked[LowPlaneBytes(len_)], msb_, n);
            }
        }
    }

    template <typename OperandT>
    __aicore__ inline void Unpack(const AscendC::LocalTensor<OperandT> &dst,
                                  const AscendC::LocalTensor<int8_t> &srcPacked, const uint32_t rows,
                                  const uint32_t len)
    {
        static_assert(!kIsAffine, "an affine mode expands through UnpackAffine; it has no centroid table");
        const uint32_t n = rows * len;
        const uint32_t packedBytes = rows * PackedBytes(len);

        const AscendC::LocalTensor<half> halfView = scratch_.ReinterpretCast<half>();
        AscendC::Cast(halfView, srcPacked, AscendC::RoundMode::CAST_NONE, packedBytes);
        AscendC::Cast(expand_, halfView, AscendC::RoundMode::CAST_NONE, packedBytes);
        AscendC::Adds(expand_, expand_, kInt8Bias, packedBytes);

        AscendC::Gather(low_, expand_, lowOffset_, kGatherSrcBase, n);
        ComputeDigit(low_, lowRecip_, Planes::kLowRadix, n);

        if constexpr (kHasMsbPlane) {
            AscendC::Gather(msb_, expand_, msbOffset_, kGatherSrcBase, n);
            ComputeDigit(msb_, msbRecip_, 2, n);
            AscendC::Muls(msb_, msb_, static_cast<float>(Planes::kLowRadix), n);
            AscendC::Add(low_, low_, msb_, n);
        }

        AscendC::Muls(scratch_, low_, kCentroidStride, n);
        // Reinterpretation: expand_ was last read as float and is written as int32 offsets next.
        AscendC::PipeBarrier<PIPE_V>();
        const AscendC::LocalTensor<int32_t> offsets = expand_.ReinterpretCast<int32_t>();
        AscendC::Cast(offsets, scratch_, AscendC::RoundMode::CAST_RINT, n);
        // Reinterpretation: the offsets were just written as int32 and the Gather reads them as uint32.
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(low_, centroid_, expand_.ReinterpretCast<uint32_t>(), kGatherSrcBase, n);

        CastToOperand(dst, low_, n);
        // Reinterpretation: scratch_ ends the band as float (fp8) or bf16 (fp4); the next band writes it as half.
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename OperandT>
    __aicore__ inline void UnpackAffine(const AscendC::LocalTensor<OperandT> &dstLow,
                                        const AscendC::LocalTensor<OperandT> &dstHigh,
                                        const AscendC::LocalTensor<int8_t> &srcPacked, const uint32_t bytes)
    {
        static_assert(kIsAffine, "UnpackAffine is only defined for a mode with uniform levels");
        static_assert(!kHasMsbPlane, "an affine mode stores one plane; there is no msb digit to fold in");
        static_assert(Planes::kLowPerByte == 2, "the plane split assumes two nibbles per byte");
        const uint32_t elems = 2u * bytes;

        const AscendC::LocalTensor<half> nibbles = msb_.ReinterpretCast<half>();
        AscendC::Cast(nibbles, srcPacked.ReinterpretCast<int4b_t>(), AscendC::RoundMode::CAST_NONE, elems);

        const AscendC::LocalTensor<half> planes = expand_.ReinterpretCast<half>();
        AscendC::DeInterleave(planes, planes[bytes], nibbles, static_cast<int32_t>(elems));

        ComputeNibbleOperands(dstLow, planes, bytes);
        ComputeNibbleOperands(dstHigh, planes[bytes], bytes);
    }

    template <typename OperandT>
    __aicore__ inline void CastToOperand(const AscendC::LocalTensor<OperandT> &dst,
                                         const AscendC::LocalTensor<float> &src, const uint32_t n)
    {
        if constexpr (kIsFp4) {
            const AscendC::LocalTensor<bfloat16_t> bf16View = scratch_.ReinterpretCast<bfloat16_t>();
            AscendC::Cast(bf16View, src, AscendC::RoundMode::CAST_RINT, n);
            AscendC::Cast(dst, bf16View, AscendC::RoundMode::CAST_RINT, n);
        } else {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, n);
        }
    }

    __aicore__ inline uint32_t len() const { return len_; }
    __aicore__ inline uint32_t rows() const { return rows_; }

private:
    static constexpr uint32_t kPeriodicWords = 3u * kFp32PerBlock;

    static constexpr float kInt8Bias = 128.0f;
    static constexpr float kEps = 1e-20f;
    static constexpr float kCentroidStride = static_cast<float>(sizeof(float));
    static constexpr uint32_t kSignBitShift = 31;
    static constexpr float kNibbleSignShift = static_cast<float>(Planes::kLowRadix / 2);
    static constexpr float kSignedLevelOffset = kNibbleSignShift - kAffineBias;

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
    // and cast back to float into low_.
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
        AscendC::Cast(low_, bins, AscendC::RoundMode::CAST_NONE, n);
    }

    // Splits low_'s levels into the low-radix digit (low_) and the msb digit (msb_).
    __aicore__ inline void ComputeMsbDigits(const uint32_t n)
    {
        AscendC::Muls(msb_, low_, 1.0f / static_cast<float>(Planes::kLowRadix), n);
        FloorInPlace(msb_, n);
        AscendC::Muls(scratch_, msb_, -static_cast<float>(Planes::kLowRadix), n);
        AscendC::Add(low_, low_, scratch_, n);
    }

    template <typename OperandT>
    __aicore__ inline void ComputeNibbleOperands(const AscendC::LocalTensor<OperandT> &dst,
                                                 const AscendC::LocalTensor<half> &plane, const uint32_t bytes)
    {
        AscendC::Cast(msb_, plane, AscendC::RoundMode::CAST_NONE, bytes);
        AscendC::Adds(msb_, msb_, kSignedLevelOffset, bytes);
        CastToOperand(dst, msb_, bytes);
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

    // x = floor(x * recip) mod radix.
    __aicore__ inline void ComputeDigit(const AscendC::LocalTensor<float> &x, const AscendC::LocalTensor<float> &recip,
                                        const int32_t radix, const uint32_t n)
    {
        TurboQuantCodec4::BroadcastMul(x, x, recip, n);
        FloorInPlace(x, n);
        AscendC::Muls(scratch_, x, 1.0f / static_cast<float>(radix), n);
        FloorInPlace(scratch_, n);
        AscendC::Muls(scratch_, scratch_, -static_cast<float>(radix), n);
        AscendC::Add(x, x, scratch_, n);
    }

    __aicore__ inline void StageAffinePlane(const AscendC::LocalTensor<int8_t> &dst,
                                            const AscendC::LocalTensor<float> &digits, const uint32_t n)
    {
        const uint32_t bytes = n / 2u;
        AscendC::Adds(digits, digits, -kNibbleSignShift, n);
        const AscendC::LocalTensor<half> levels = scratch_.ReinterpretCast<half>();
        AscendC::Cast(levels, digits, AscendC::RoundMode::CAST_RINT, n);

        const AscendC::LocalTensor<half> woven = expand_.ReinterpretCast<half>();
        AscendC::Interleave(woven, woven[bytes], levels, levels[bytes], static_cast<int32_t>(bytes));
        AscendC::Cast(dst.ReinterpretCast<int4b_t>(), woven, AscendC::RoundMode::CAST_RINT, n);
        // Reinterpretation: scratch_ holds half levels, and the next vector's encode writes it as float.
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StageLowPlane(const AscendC::LocalTensor<int8_t> &dst,
                                         const AscendC::LocalTensor<float> &digits, const uint32_t n)
    {
        constexpr int32_t kDigitsPerByte = Planes::kLowPerByte;
        const uint32_t bytes = n / static_cast<uint32_t>(kDigitsPerByte);

        AscendC::Gather(expand_, digits, packOffset_, kGatherSrcBase, bytes);
        float weight = 1.0f;
        for (int32_t digit = 1; digit < kDigitsPerByte; ++digit) {
            weight *= static_cast<float>(Planes::kLowRadix);
            AscendC::Gather(low_, digits, packOffset_[static_cast<uint32_t>(digit) * bytes], kGatherSrcBase, bytes);
            // Overlapping write: digits is low_, so this Gather rewrites the lanes the next digit gathers from.
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(low_, low_, weight, bytes);
            AscendC::Add(expand_, expand_, low_, bytes);
        }
        StageBytes(dst, bytes);
    }

    __aicore__ inline void StageMsbPlane(const AscendC::LocalTensor<int8_t> &dst,
                                         const AscendC::LocalTensor<float> &digits, const uint32_t n)
    {
        const uint32_t bytes = n / static_cast<uint32_t>(kMsbPerByte);
        TurboQuantCodec4::BroadcastMul(scratch_, digits, msbWeight_, n);
        AscendC::WholeReduceSum<float>(expand_, scratch_, kMsbPerByte, static_cast<uint8_t>(bytes), 1, 1, 1);
        StageBytes(dst, bytes);
    }

    // expand_ holds unsigned byte values as float: bias them to int8 and cast into dst.
    __aicore__ inline void StageBytes(const AscendC::LocalTensor<int8_t> &dst, const uint32_t bytes)
    {
        AscendC::Adds(expand_, expand_, -kInt8Bias, bytes);
        // Reinterpretation: scratch_ was last used as float and is written as half next.
        AscendC::PipeBarrier<PIPE_V>();
        const AscendC::LocalTensor<half> halfView = scratch_.ReinterpretCast<half>();
        AscendC::Cast(halfView, expand_, AscendC::RoundMode::CAST_NONE, bytes);
        AscendC::Cast(dst, halfView, AscendC::RoundMode::CAST_RINT, bytes);
        // Reinterpretation: scratch_ holds half bytes, and the msb plane or the next vector writes it as float.
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::TBuf<AscendC::QuePosition::VECCALC> constBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> workBuf_;

    AscendC::LocalTensor<uint32_t> lowOffset_;
    AscendC::LocalTensor<uint32_t> msbOffset_;
    AscendC::LocalTensor<uint32_t> packOffset_;
    AscendC::LocalTensor<float> lowRecip_;
    AscendC::LocalTensor<float> msbRecip_;
    AscendC::LocalTensor<float> msbWeight_;
    AscendC::LocalTensor<float> centroid_;

    AscendC::LocalTensor<float> expand_;
    AscendC::LocalTensor<float> low_;
    AscendC::LocalTensor<float> msb_;
    AscendC::LocalTensor<float> scratch_;
    AscendC::LocalTensor<float> reduceWork_;
    AscendC::LocalTensor<float> broadcast_;
    AscendC::LocalTensor<float> binLane_[kBinLanes];

    uint32_t len_ = 0;
    uint32_t rows_ = 0;
    uint32_t batchLen_ = 0;
    float invSqrtLen_ = 1.0f;
};

using TurboQuantCodecKv3Fp4 = TurboQuantModeCodec<TurboQuantMode::KV3_FP4>;
using TurboQuantCodecKv4Fp8 = TurboQuantModeCodec<TurboQuantMode::KV4_FP8>;
using TurboQuantCodecKv5Fp8 = TurboQuantModeCodec<TurboQuantMode::KV5_FP8>;

}
}

#endif
