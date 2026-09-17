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

    __aicore__ inline void Init(AscendC::TPipe *pipe, uint32_t vecLen, uint32_t batchRows, float invSqrtLen,
                                const AscendC::GlobalTensor<int32_t> &tablesGm)
    {
        len_ = vecLen;
        rows_ = batchRows;
        batchLen_ = vecLen * batchRows;
        invSqrtLen_ = invSqrtLen;

        pipe->InitBuffer(workBuf_, WorkBufferWords(vecLen, batchRows) * sizeof(float));

        if constexpr (!kIsAffine) {
            const uint32_t constWords = ConstTableWords(vecLen, batchRows);
            pipe->InitBuffer(constBuf_, constWords * sizeof(int32_t));

            AscendC::LocalTensor<int32_t> pool = constBuf_.Get<int32_t>();
            AscendC::DataCopy(pool, tablesGm, constWords);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::LocalTensor<float> poolF = pool.template ReinterpretCast<float>();
            uint32_t coff = 0;
            lowOffset_ = pool[coff].template ReinterpretCast<uint32_t>();
            coff += batchLen_;
            msbOffset_ = pool[coff].template ReinterpretCast<uint32_t>();
            coff += batchLen_;
            lowRecip_ = poolF[coff];
            coff += kFp32PerBlock;
            msbRecip_ = poolF[coff];
            coff += kFp32PerBlock;
            msbWeight_ = poolF[coff];
            coff += kFp32PerBlock;
            packOffset_ = pool[coff].template ReinterpretCast<uint32_t>();
            coff += len_;
            centroid_ = poolF[coff];
        }

        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        uint32_t off = 0;
        if constexpr (kIsAffine) {
            expand_ = work[off];
            off += batchLen_ / 2u;
            msb_ = work[off];
            off += batchLen_ / 2u;
            low_ = work[off];
            off += len_;
            scratch_ = work[off];
            off += len_;
        } else {
            expand_ = work[off];
            off += batchLen_;
            low_ = work[off];
            off += batchLen_;
            msb_ = work[off];
            off += batchLen_;
            scratch_ = work[off];
            off += batchLen_;
        }
        reduceWork_ = work[off];
        off += len_;
        broadcast_ = work[off];
        off += kBrcbDstLanes + kFp32PerBlock;
        for (int lane = 0; lane < kBinLanes; ++lane) {
            binLane_[lane] = work[off];
            off += len_;
        }
    }

    __aicore__ inline void Encode(const AscendC::LocalTensor<int8_t> &dstPacked,
                                  const AscendC::LocalTensor<float> &src,
                                  const AscendC::LocalTensor<float> &scaleOut, int len)
    {
        const uint32_t n = static_cast<uint32_t>(len);

        AscendC::Mul(scratch_, src, src, n);
        ChainBarrier();
        AscendC::ReduceSum<float>(scaleOut, scratch_, reduceWork_, n);
        ChainBarrier();
        AscendC::Sqrt(scaleOut, scaleOut, 1);
        ChainBarrier();
        AscendC::Muls(scaleOut, scaleOut, invSqrtLen_, 1);
        ChainBarrier();
        AscendC::Adds(scaleOut, scaleOut, kEps, 1);
        ChainBarrier();

        AscendC::Brcb(broadcast_, scaleOut, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
        ChainBarrier();
        AscendC::Duplicate(broadcast_[kBrcbDstLanes], -1.0f, kFp32PerBlock);
        ChainBarrier();
        AscendC::Div(broadcast_[kBrcbDstLanes], broadcast_[kBrcbDstLanes], broadcast_, kFp32PerBlock);
        ChainBarrier();

        // The broadcast ends in a barrier on purpose: reduceWork_ held the reduce's float intermediates and
        // is rewritten as int32 bins next.
        TurboQuantCodec4::BroadcastMul(scratch_, src, broadcast_[kBrcbDstLanes], n);

        AscendC::LocalTensor<int32_t> bins = reduceWork_.ReinterpretCast<int32_t>();
        AscendC::Duplicate(bins, 0, n);
        ChainBarrier();

        // Each bin lane is written as float, shifted as uint32 and summed as int32, and the next pass writes
        // it as float again: those three barriers mark reinterpretations and stay.
        for (int base = 0; base < kThresholdCount; base += kBinLanes) {
            const int lanes = (kThresholdCount - base) < kBinLanes ? (kThresholdCount - base) : kBinLanes;
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
                    AscendC::LocalTensor<int32_t> src2 = binLane_[lane + span].ReinterpretCast<int32_t>();
                    AscendC::Add(dst, dst, src2, n);
                }
                ChainBarrier();
            }
            AscendC::Add(bins, bins, binLane_[0].ReinterpretCast<int32_t>(), n);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Cast(low_, bins, AscendC::RoundMode::CAST_NONE, n);
        ChainBarrier();

        if constexpr (kHasMsbPlane) {
            AscendC::Muls(msb_, low_, 1.0f / static_cast<float>(Planes::kLowRadix), n);
            AscendC::PipeBarrier<PIPE_V>();
            FloorInPlace(msb_, n);
            AscendC::Muls(scratch_, msb_, -static_cast<float>(Planes::kLowRadix), n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(low_, low_, scratch_, n);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if constexpr (kIsAffine) {
            PackAffinePlane(dstPacked, low_, n);
        } else {
            PackLowPlane(dstPacked, low_, n);
            if constexpr (kHasMsbPlane) {
                PackMsbPlane(dstPacked[LowPlaneBytes(len_)], msb_, n);
            }
        }
    }

    template <typename OperandT>
    __aicore__ inline void Unpack(const AscendC::LocalTensor<OperandT> &dst,
                                  const AscendC::LocalTensor<int8_t> &srcPacked, int rows, int len)
    {
        static_assert(!kIsAffine, "an affine mode expands through UnpackAffine; it has no centroid table");
        const uint32_t n = static_cast<uint32_t>(rows) * static_cast<uint32_t>(len);
        const uint32_t packedBytes =
            static_cast<uint32_t>(rows) * PackedBytes(static_cast<uint32_t>(len));

        AscendC::LocalTensor<half> halfView = scratch_.ReinterpretCast<half>();
        AscendC::Cast(halfView, srcPacked, AscendC::RoundMode::CAST_NONE, packedBytes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(expand_, halfView, AscendC::RoundMode::CAST_NONE, packedBytes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(expand_, expand_, kInt8Bias, packedBytes);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Gather(low_, expand_, lowOffset_, kGatherSrcBase, n);
        AscendC::PipeBarrier<PIPE_V>();
        ExtractDigit(low_, lowRecip_, Planes::kLowRadix, n);

        if constexpr (kHasMsbPlane) {
            AscendC::Gather(msb_, expand_, msbOffset_, kGatherSrcBase, n);
            AscendC::PipeBarrier<PIPE_V>();
            ExtractDigit(msb_, msbRecip_, 2, n);
            AscendC::Muls(msb_, msb_, static_cast<float>(Planes::kLowRadix), n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(low_, low_, msb_, n);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Muls(scratch_, low_, kCentroidStride, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<int32_t> offsets = expand_.ReinterpretCast<int32_t>();
        AscendC::Cast(offsets, scratch_, AscendC::RoundMode::CAST_RINT, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(low_, centroid_, expand_.ReinterpretCast<uint32_t>(), kGatherSrcBase, n);
        AscendC::PipeBarrier<PIPE_V>();

        CastToOperand(dst, low_, n);
    }

    template <typename OperandT, bool VEC_BARRIERS = true>
    __aicore__ inline void UnpackAffine(const AscendC::LocalTensor<OperandT> &dstLow,
                                        const AscendC::LocalTensor<OperandT> &dstHigh,
                                        const AscendC::LocalTensor<int8_t> &srcPacked, uint32_t bytes)
    {
        static_assert(kIsAffine, "UnpackAffine is only defined for a mode with uniform levels");
        static_assert(!kHasMsbPlane, "an affine mode stores one plane; there is no msb digit to fold in");
        static_assert(Planes::kLowPerByte == 2, "the plane split assumes two nibbles per byte");
        const uint32_t elems = 2u * bytes;

        AscendC::LocalTensor<half> nibbles = msb_.ReinterpretCast<half>();
        AscendC::Cast(nibbles, srcPacked.ReinterpretCast<int4b_t>(), AscendC::RoundMode::CAST_NONE, elems);

        AscendC::LocalTensor<half> planes = expand_.ReinterpretCast<half>();
        AscendC::DeInterleave(planes, planes[bytes], nibbles, static_cast<int32_t>(elems));

        ExpandNibblePlane<OperandT, VEC_BARRIERS>(dstLow, planes, bytes);
        ExpandNibblePlane<OperandT, VEC_BARRIERS>(dstHigh, planes[bytes], bytes);
    }

    template <typename OperandT, bool VEC_BARRIERS = true>
    __aicore__ inline void CastToOperand(const AscendC::LocalTensor<OperandT> &dst,
                                         const AscendC::LocalTensor<float> &src, uint32_t n)
    {
        if constexpr (kIsFp4) {
            AscendC::LocalTensor<bfloat16_t> bf = scratch_.ReinterpretCast<bfloat16_t>();
            AscendC::Cast(bf, src, AscendC::RoundMode::CAST_RINT, n);
            VecBarrier<VEC_BARRIERS>();
            AscendC::Cast(dst, bf, AscendC::RoundMode::CAST_RINT, n);
            VecBarrier<VEC_BARRIERS>();
        } else {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, n);
            VecBarrier<VEC_BARRIERS>();
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

    template <typename OperandT, bool VEC_BARRIERS>
    __aicore__ inline void ExpandNibblePlane(const AscendC::LocalTensor<OperandT> &dst,
                                             const AscendC::LocalTensor<half> &plane, uint32_t bytes)
    {
        AscendC::Cast(msb_, plane, AscendC::RoundMode::CAST_NONE, bytes);
        AscendC::Adds(msb_, msb_, kSignedLevelOffset, bytes);
        CastToOperand<OperandT, VEC_BARRIERS>(dst, msb_, bytes);
    }

    // A barrier between two dependent ops of the encode chain. kv4fp8's chain issues without them and keeps
    // only the barriers that mark a buffer reinterpretation (csrc/tests/TURBOQUANT_TESTS.md 13.29); the
    // codebook modes keep the barriers they were written with.
    __aicore__ static inline void ChainBarrier()
    {
        if constexpr (!kIsAffine) {
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ static inline void FloorInPlace(const AscendC::LocalTensor<float> &x, uint32_t count)
    {
        AscendC::LocalTensor<int32_t> intView = x.ReinterpretCast<int32_t>();
        AscendC::Cast(intView, x, AscendC::RoundMode::CAST_FLOOR, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(x, intView, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ExtractDigit(const AscendC::LocalTensor<float> &x,
                                        const AscendC::LocalTensor<float> &recip, int32_t radix, uint32_t n)
    {
        TurboQuantCodec4::BroadcastMul(x, x, recip, n);
        FloorInPlace(x, n);
        AscendC::Muls(scratch_, x, 1.0f / static_cast<float>(radix), n);
        AscendC::PipeBarrier<PIPE_V>();
        FloorInPlace(scratch_, n);
        AscendC::Muls(scratch_, scratch_, -static_cast<float>(radix), n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(x, x, scratch_, n);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PackAffinePlane(const AscendC::LocalTensor<int8_t> &dst,
                                           const AscendC::LocalTensor<float> &digits, uint32_t n)
    {
        const uint32_t bytes = n / 2u;
        AscendC::Adds(digits, digits, -kNibbleSignShift, n);
        AscendC::LocalTensor<half> levels = scratch_.ReinterpretCast<half>();
        AscendC::Cast(levels, digits, AscendC::RoundMode::CAST_RINT, n);

        AscendC::LocalTensor<half> woven = expand_.ReinterpretCast<half>();
        AscendC::Interleave(woven, woven[bytes], levels, levels[bytes], static_cast<int32_t>(bytes));
        AscendC::Cast(dst.ReinterpretCast<int4b_t>(), woven, AscendC::RoundMode::CAST_RINT, n);
        // scratch_ held half levels here, and the next vector's encode writes it as float.
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PackLowPlane(const AscendC::LocalTensor<int8_t> &dst,
                                        const AscendC::LocalTensor<float> &digits, uint32_t n)
    {
        constexpr int32_t kDpb = Planes::kLowPerByte;
        const uint32_t bytes = n / static_cast<uint32_t>(kDpb);

        AscendC::Gather(expand_, digits, packOffset_, kGatherSrcBase, bytes);
        AscendC::PipeBarrier<PIPE_V>();
        float weight = 1.0f;
        for (int32_t j = 1; j < kDpb; ++j) {
            weight *= static_cast<float>(Planes::kLowRadix);
            AscendC::Gather(low_, digits, packOffset_[static_cast<uint32_t>(j) * bytes], kGatherSrcBase, bytes);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(low_, low_, weight, bytes);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(expand_, expand_, low_, bytes);
            AscendC::PipeBarrier<PIPE_V>();
        }
        EmitBytes(dst, bytes);
    }

    __aicore__ inline void PackMsbPlane(const AscendC::LocalTensor<int8_t> &dst,
                                        const AscendC::LocalTensor<float> &digits, uint32_t n)
    {
        const uint32_t bytes = n / static_cast<uint32_t>(kMsbPerByte);
        TurboQuantCodec4::BroadcastMul(scratch_, digits, msbWeight_, n);
        AscendC::WholeReduceSum<float>(expand_, scratch_, kMsbPerByte, static_cast<uint8_t>(bytes), 1, 1, 1);
        AscendC::PipeBarrier<PIPE_V>();
        EmitBytes(dst, bytes);
    }

    __aicore__ inline void EmitBytes(const AscendC::LocalTensor<int8_t> &dst, uint32_t bytes)
    {
        AscendC::Adds(expand_, expand_, -kInt8Bias, bytes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<half> halfView = scratch_.ReinterpretCast<half>();
        AscendC::Cast(halfView, expand_, AscendC::RoundMode::CAST_NONE, bytes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(dst, halfView, AscendC::RoundMode::CAST_RINT, bytes);
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

__aicore__ inline void unpack_tq4_to_fp8(TurboQuantCodecKv4Fp8 &codec,
                                         const AscendC::LocalTensor<fp8_e4m3fn_t> &dstLow,
                                         const AscendC::LocalTensor<fp8_e4m3fn_t> &dstHigh,
                                         const AscendC::LocalTensor<int8_t> &srcPacked, uint32_t bytes)
{
    codec.UnpackAffine(dstLow, dstHigh, srcPacked, bytes);
}

__aicore__ inline void unpack_tq5_to_fp8(TurboQuantCodecKv5Fp8 &codec,
                                         const AscendC::LocalTensor<fp8_e4m3fn_t> &dst,
                                         const AscendC::LocalTensor<int8_t> &srcPacked, int rows, int len)
{
    codec.Unpack(dst, srcPacked, rows, len);
}

}
}

#endif
