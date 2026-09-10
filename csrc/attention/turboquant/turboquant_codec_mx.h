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
 * The multi-rate TurboQuant codec: encode at 3, 4 or 5 bits, and expand
 * straight onto the Cube's operand grid.  It does not carry the rotation; a
 * kernel using it owns a TurboQuantCodec4 for ApplyPi.
 *
 * A code splits into a `low` digit of kLowBits and an `msb` digit of kMsbBits,
 * stored in separate byte-aligned planes inside the vector slot:
 *
 *   mode     low plane                 msb plane            slot (d=256)
 *   kv3fp4   2 bits, 4 digits/byte      1 bit, 8/byte       64 + 32 =  96 B
 *   kv4fp8   4 bits, 2 digits/byte      --                  128 +  0 = 128 B
 *   kv5fp8   4 bits, 2 digits/byte      1 bit, 8/byte       128 + 32 = 160 B
 *
 * No code straddles a byte.  kv5fp8's low plane is laid out exactly as
 * kv4fp8's whole slot, so the two share one packer and one unpacker.
 *
 * Digit j of a byte is extracted at any radix by
 *
 *     t   = byte * radix^-(p mod digitsPerByte)      one Mul against a block
 *     t   = floor(t)
 *     h   = floor(t / radix)
 *     dig = t - radix * h                            i.e. t mod radix
 *
 * ExtractDigit below is that.  Every period divides the 8 fp32 lanes of a
 * 32-byte block, so the reciprocal and weight tables are a single block applied
 * with src1BlkStride = src1RepStride = 0.
 *
 * The arch35 vector converter has no fp32 <-> fp4 leg -- only
 * bf16 <-> fp4x2_e2m1 -- so kv3fp4 expands fp32 -> bf16 -> fp4x2_e2m1.
 */

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_CODEC_MX_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_CODEC_MX_H

#include "kernel_operator.h"
#include "turboquant_codec_950.h"
#include "turboquant_mode.h"

namespace vllm_ascend {
namespace turboquant {

/*
 * Plane geometry.  Kept out of TurboQuantModeTraits because it is device-side
 * packing detail; the host only needs PackedBytes, which the mode header gives.
 */
template <TurboQuantMode MODE>
struct TurboQuantPlanes;

template <>
struct TurboQuantPlanes<TurboQuantMode::KV3_FP4> {
    static constexpr int32_t kLowBits = 2;
    static constexpr int32_t kMsbBits = 1;
    static constexpr int32_t kLowRadix = 4;    // 1 << kLowBits
    static constexpr int32_t kLowPerByte = 4;  // 8 / kLowBits
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

// Digits the msb plane packs into a byte.  Always 8: every mode's msb digit is
// a single bit, and a wider one would be a different plane split.
constexpr int32_t kMsbPerByte = 8;

/*
 * The multi-rate codec.
 *
 * `vecLen`     head_size, a power of two in [64, 256].
 * `batchRows`  vectors Unpack may expand in one call; the low/msb offset tables
 *              are built for exactly this many and are not valid for more.
 *              Encode always runs one vector at a time.
 *
 * The constant-table image is this codec's own and deliberately does not share
 * TurboQuantCodec<4>::ConstTableWords, which five mirrors pin.
 *
 *   [0 .. B)             lowOffset_   uint32 byte offsets,  B = vecLen*batchRows
 *   [B .. 2B)            msbOffset_   uint32 byte offsets
 *   [2B .. 2B+8)         lowRecip_    fp32 block, radix^-(p mod kLowPerByte)
 *   [2B+8 .. 2B+16)      msbRecip_    fp32 block, 2^-(p mod 8)
 *   [2B+16 .. 2B+24)     msbWeight_   fp32 block, 2^(p mod 8)
 *   [2B+24 .. +vecLen)   packOffset_  uint32, the encoder's low-plane gathers
 *   [.. + kLevels)       centroid_    the mode's stored codebook
 *
 * msbOffset_ is allocated even for kv4fp8, which has no msb plane, so the
 * layout is one formula rather than a conditional.
 */
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

    // Elements of the Cube operand tensor one expanded vector occupies.  fp4 is
    // half a byte and LocalTensor<fp4x2_e2m1_t> counts *pairs*, so a d=256 fp4
    // row is 128 elements and an fp8 row is 256.
    __aicore__ static constexpr uint32_t OperandElems(uint32_t vecLen)
    {
        return kIsFp4 ? vecLen / 2 : vecLen;
    }

    // Same shape of contract as TurboQuantCodec<4>::ConstTableWords: the host
    // writes this exact image and the device never rewrites a word of it.
    __aicore__ static inline uint32_t ConstTableWords(uint32_t vecLen, uint32_t batchRows)
    {
        return 2u * vecLen * batchRows + kPeriodicWords + vecLen + static_cast<uint32_t>(kLevels);
    }

    // Scratch, deliberately uninitialised.  Four full-length buffers carry the
    // expand; reduceWork_, broadcast_ and the boundary lanes belong to the
    // encoder and are sized off vecLen because it only ever sees one vector.
    __aicore__ static inline uint32_t WorkBufferWords(uint32_t vecLen, uint32_t batchRows)
    {
        return 4u * vecLen * batchRows + vecLen + kBrcbDstLanes + kFp32PerBlock +
               static_cast<uint32_t>(kBinLanes) * vecLen;
    }

    // Decision boundaries, consumed as Adds immediates.  Reading them from the
    // traits struct is what a function-local constexpr array becomes once
    // inlined; TurboQuantCodec<4>::Threshold established the pattern compiles
    // clean for dav-c310.
    __aicore__ static inline float Threshold(int i) { return Traits::kThresholds[i]; }

    __aicore__ inline void Init(AscendC::TPipe *pipe, uint32_t vecLen, uint32_t batchRows, float invSqrtLen,
                                const AscendC::GlobalTensor<int32_t> &tablesGm)
    {
        len_ = vecLen;
        rows_ = batchRows;
        batchLen_ = vecLen * batchRows;
        invSqrtLen_ = invSqrtLen;
        const uint32_t constWords = ConstTableWords(vecLen, batchRows);

        pipe->InitBuffer(constBuf_, constWords * sizeof(int32_t));
        pipe->InitBuffer(workBuf_, WorkBufferWords(vecLen, batchRows) * sizeof(float));

        AscendC::LocalTensor<int32_t> pool = constBuf_.Get<int32_t>();
        AscendC::DataCopy(pool, tablesGm, constWords);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::LocalTensor<float> poolF = pool.template ReinterpretCast<float>();
        uint32_t off = 0;
        lowOffset_ = pool[off].template ReinterpretCast<uint32_t>();
        off += batchLen_;
        msbOffset_ = pool[off].template ReinterpretCast<uint32_t>();
        off += batchLen_;
        lowRecip_ = poolF[off];
        off += kFp32PerBlock;
        msbRecip_ = poolF[off];
        off += kFp32PerBlock;
        msbWeight_ = poolF[off];
        off += kFp32PerBlock;
        packOffset_ = pool[off].template ReinterpretCast<uint32_t>();
        off += len_;
        centroid_ = poolF[off];

        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        off = 0;
        expand_ = work[off];
        off += batchLen_;
        low_ = work[off];
        off += batchLen_;
        msb_ = work[off];
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

    /*
     * Encode one rotated vector at kBits.
     *
     *   dstPacked  [PackedBytes(len)] int8, low plane then msb plane
     *   src        [len]              fp32, already rotated
     *   scaleOut   [1]                fp32, ||src||_2 / sqrt(len).  Brcb reads a
     *                                 whole 32B block, so back it with 8 lanes.
     *
     * The codebook gain is not applied here: it lives in the stored centroids
     * and comes back out of the scale on the decode side.
     */
    __aicore__ inline void Encode(const AscendC::LocalTensor<int8_t> &dstPacked,
                                  const AscendC::LocalTensor<float> &src,
                                  const AscendC::LocalTensor<float> &scaleOut, int len)
    {
        const uint32_t n = static_cast<uint32_t>(len);

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

        // scratch_ = -u = -src / scale, so each boundary test is a single Adds.
        TurboQuantCodec4::BroadcastMul(scratch_, src, broadcast_[kBrcbDstLanes], n);

        AscendC::LocalTensor<int32_t> bins = reduceWork_.ReinterpretCast<int32_t>();
        AscendC::Duplicate(bins, 0, n);
        AscendC::PipeBarrier<PIPE_V>();

        // q = sum_i [u > t_i], kBinLanes boundaries in flight.  Same structure
        // and the same reason as TurboQuantCodec<4>::Quantize4Bit -- read the
        // kBinLanes note there for what the serial form cost -- except that the
        // loop now runs over 7, 15 or 31 boundaries rather than always 15.
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
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Add(bins, bins, binLane_[0].ReinterpretCast<int32_t>(), n);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Cast(low_, bins, AscendC::RoundMode::CAST_NONE, n);
        AscendC::PipeBarrier<PIPE_V>();

        // Split q: msb = floor(q / lowRadix), low = q - lowRadix * msb.
        if constexpr (kHasMsbPlane) {
            AscendC::Muls(msb_, low_, 1.0f / static_cast<float>(Planes::kLowRadix), n);
            AscendC::PipeBarrier<PIPE_V>();
            FloorInPlace(msb_, n);
            AscendC::Muls(scratch_, msb_, -static_cast<float>(Planes::kLowRadix), n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(low_, low_, scratch_, n);
            AscendC::PipeBarrier<PIPE_V>();
        }

        PackLowPlane(dstPacked, low_, n);
        if constexpr (kHasMsbPlane) {
            PackMsbPlane(dstPacked[LowPlaneBytes(len_)], msb_, n);
        }
    }

    /*
     * Expand `rows` packed vectors onto the Cube operand grid.
     *
     *   dst        [rows * OperandElems(len)] of the mode's operand type
     *   srcPacked  [rows * PackedBytes(len)]  int8
     *
     * The per-vector scale is not applied: the decode path folds it, and the
     * codebook gain, into the score row (K) or the softmax probabilities (V).
     */
    template <typename OperandT>
    __aicore__ inline void Unpack(const AscendC::LocalTensor<OperandT> &dst,
                                  const AscendC::LocalTensor<int8_t> &srcPacked, int rows, int len)
    {
        const uint32_t n = static_cast<uint32_t>(rows) * static_cast<uint32_t>(len);
        const uint32_t packedBytes =
            static_cast<uint32_t>(rows) * PackedBytes(static_cast<uint32_t>(len));

        // int8 -> fp32 through half, the only route the converter offers, then
        // the -128 bias the packer applied comes back off.
        AscendC::LocalTensor<half> halfView = scratch_.ReinterpretCast<half>();
        AscendC::Cast(halfView, srcPacked, AscendC::RoundMode::CAST_NONE, packedBytes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(expand_, halfView, AscendC::RoundMode::CAST_NONE, packedBytes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(expand_, expand_, kInt8Bias, packedBytes);
        AscendC::PipeBarrier<PIPE_V>();

        // low_[p] = digit (p mod kLowPerByte) of the low plane's byte for p.
        AscendC::Gather(low_, expand_, lowOffset_, kGatherSrcBase, n);
        AscendC::PipeBarrier<PIPE_V>();
        ExtractDigit(low_, lowRecip_, Planes::kLowRadix, n);

        if constexpr (kHasMsbPlane) {
            AscendC::Gather(msb_, expand_, msbOffset_, kGatherSrcBase, n);
            AscendC::PipeBarrier<PIPE_V>();
            ExtractDigit(msb_, msbRecip_, 2, n);
            // q = low + lowRadix * msb.
            AscendC::Muls(msb_, msb_, static_cast<float>(Planes::kLowRadix), n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(low_, low_, msb_, n);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // dst[p] = centroid_[q_p].  Gather counts byte offsets from the start of
        // centroid_ (see kGatherSrcBase), so the index is scaled by the word
        // size before it is cast.  expand_ is spent by here -- it held the
        // packed byte per channel -- so the offset table needs no buffer.
        AscendC::Muls(scratch_, low_, kCentroidStride, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<int32_t> offsets = expand_.ReinterpretCast<int32_t>();
        AscendC::Cast(offsets, scratch_, AscendC::RoundMode::CAST_RINT, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(low_, centroid_, expand_.ReinterpretCast<uint32_t>(), kGatherSrcBase, n);
        AscendC::PipeBarrier<PIPE_V>();

        CastToOperand(dst, low_, n);
    }

    // fp32 -> the mode's Cube operand type.  Both branches are exact.  The fp4
    // branch is two casts: the arch35 converter has no fp32 <-> fp4 leg.
    template <typename OperandT>
    __aicore__ inline void CastToOperand(const AscendC::LocalTensor<OperandT> &dst,
                                         const AscendC::LocalTensor<float> &src, uint32_t n)
    {
        if constexpr (kIsFp4) {
            AscendC::LocalTensor<bfloat16_t> bf = scratch_.ReinterpretCast<bfloat16_t>();
            AscendC::Cast(bf, src, AscendC::RoundMode::CAST_RINT, n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(dst, bf, AscendC::RoundMode::CAST_RINT, n);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, n);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline uint32_t len() const { return len_; }
    __aicore__ inline uint32_t rows() const { return rows_; }

private:
    // lowRecip_, msbRecip_ and msbWeight_, one 32-byte block each.
    static constexpr uint32_t kPeriodicWords = 3u * kFp32PerBlock;

    static constexpr float kInt8Bias = 128.0f;
    static constexpr float kEps = 1e-20f;
    static constexpr float kCentroidStride = static_cast<float>(sizeof(float));
    static constexpr uint32_t kSignBitShift = 31;

    // floor() via the fp32 <-> int32 converter; the dedicated Floor intrinsic
    // is not available on every supported arch.  Same routine as
    // TurboQuantCodec<4>::FloorInPlace, which is private to that class.
    __aicore__ static inline void FloorInPlace(const AscendC::LocalTensor<float> &x, uint32_t count)
    {
        AscendC::LocalTensor<int32_t> intView = x.ReinterpretCast<int32_t>();
        AscendC::Cast(intView, x, AscendC::RoundMode::CAST_FLOOR, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(x, intView, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // x[p] <- (x[p] / radix^(p mod digitsPerByte)) mod radix.  The only per-lane
    // quantity is `recip`, a single 32-byte block whose period divides 8.
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

    /*
     * Fold kLowPerByte consecutive digits into one byte:
     *
     *   byte[b] = sum_j digit[b * dpb + j] * radix^j
     *
     * packOffset_ holds the dpb offset tables end to end, each of len/dpb
     * entries, which is why it is `len` words for every mode.  The result is
     * biased by -128 into int8, matching the shipping 4-bit packer.
     */
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

    // Fold eight 1-bit digits into one byte.  Eight lanes is exactly one 32-byte
    // block, so this is one Mul against msbWeight_ plus one WholeReduceSum.
    __aicore__ inline void PackMsbPlane(const AscendC::LocalTensor<int8_t> &dst,
                                        const AscendC::LocalTensor<float> &digits, uint32_t n)
    {
        const uint32_t bytes = n / static_cast<uint32_t>(kMsbPerByte);
        TurboQuantCodec4::BroadcastMul(scratch_, digits, msbWeight_, n);
        AscendC::WholeReduceSum<float>(expand_, scratch_, kMsbPerByte, static_cast<uint8_t>(bytes), 1, 1, 1);
        AscendC::PipeBarrier<PIPE_V>();
        EmitBytes(dst, bytes);
    }

    // expand_[0, bytes) in [0, 255] -> int8 in [-128, 127].  The decoder adds
    // the bias back numerically, so nothing downstream reads the bit pattern.
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

// The named entry point of the kv5fp8 dequantization stage: packed 5-bit
// indices in `srcPacked` -> fp8_e4m3fn in `dst`, both in local UB memory.
// A thin name over TurboQuantModeCodec<KV5_FP8>::Unpack.
__aicore__ inline void unpack_tq5_to_fp8(TurboQuantCodecKv5Fp8 &codec,
                                         const AscendC::LocalTensor<fp8_e4m3fn_t> &dst,
                                         const AscendC::LocalTensor<int8_t> &srcPacked, int rows, int len)
{
    codec.Unpack(dst, srcPacked, rows, len);
}

}  // namespace turboquant
}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_ATTENTION_TURBOQUANT_CODEC_MX_H
