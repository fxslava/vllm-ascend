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
 * straight onto the Cube's operand grid.
 *
 * WHAT THIS ADDS OVER TurboQuantCodec<4> IN turboquant_codec_950.h, which it
 * neither replaces nor disturbs:
 *
 *   * the rate is a template parameter, so one routine covers all three modes;
 *   * the expand produces fp8_e4m3fn (or fp4 e2m1) in UB rather than fp32,
 *     because the Cube consumes those types directly and an fp32 intermediate
 *     would be a full-length buffer the GEMM never reads.
 *
 * It does NOT carry the rotation.  Pi = D H D is rate-independent, its
 * implementation is where four separate device defects were once found, and a
 * second copy of it is the last thing this file should contain.  A kernel that
 * uses this codec owns a TurboQuantCodec4 for ApplyPi and rotates one vector at
 * a time through it -- which is all the rotation is ever asked to do, so that
 * instance is Init'd at batchRows = 1 and costs ~21 KB of UB rather than the
 * ~82 KB a tile-sized one would.
 *
 * THE PACKED LAYOUT IS TWO PLANES, NOT A BIT STREAM.  A code is split into a
 * `low` digit of kLowBits and an `msb` digit of kMsbBits, and the two are
 * stored in separate byte-aligned planes inside the vector slot:
 *
 *   mode     low plane                 msb plane            slot (d=256)
 *   kv3fp4   2 bits, 4 digits/byte      1 bit, 8/byte       64 + 32 =  96 B
 *   kv4fp8   4 bits, 2 digits/byte      --                  128 +  0 = 128 B
 *   kv5fp8   4 bits, 2 digits/byte      1 bit, 8/byte       128 + 32 = 160 B
 *
 * Two properties fall out of that, and both are load-bearing.  First, no code
 * ever straddles a byte, so an unpack never reassembles one from two loads and
 * the slot length is exact rather than rounded up -- a 40-bit-per-8-codes bit
 * stream gives the same 160 bytes but needs a cross-byte shift by a per-lane
 * amount, which the vector core cannot express.  Second, kv5fp8's low plane is
 * laid out exactly as kv4fp8's whole slot, so the two share one packer and one
 * unpacker and differ only in whether the msb plane exists.
 *
 * DIGIT EXTRACTION IS ONE ROUTINE AT THREE RADICES.  Pulling digit j out of a
 * byte needs a per-lane variable shift, which the vector core does not have.
 * What it does have is a constant indexed by lane, and the digit index within a
 * byte is periodic in the lane with period `digits per byte` -- 2, 4 or 8, all
 * of which divide the 8 fp32 lanes of a 32-byte block.  So
 *
 *     t   = byte * radix^-(p mod digitsPerByte)      one Mul against a block
 *     t   = floor(t)
 *     h   = floor(t / radix)
 *     dig = t - radix * h                            i.e. t mod radix
 *
 * is seven full-length vector instructions per plane at any radix, with no mask
 * register and no scalar round trip.  ExtractDigit below is that.
 *
 * THE PERIODIC TABLES COST ONE 32-BYTE BLOCK EACH, NOT A FULL BUFFER.  Because
 * every period divides 8, the reciprocal and weight tables are a single block
 * applied with src1BlkStride = src1RepStride = 0, which makes the operand
 * repeat as block[i % 8] across the whole vector.  That is exactly the addressing
 * TurboQuantCodec4::BroadcastMul already emits; its doc comment describes the
 * all-lanes-equal case it was written for, but the arithmetic it performs is
 * the general dst[i] = src[i] * block[i % 8], which is what is relied on here.
 * Three full-length tables become 24 words.
 *
 * THE CAST TO THE OPERAND GRID IS CONSTRAINED BY THE CONVERTER.  On arch35 the
 * vector converter admits fp32 -> fp8_e4m3fn under CAST_RINT and
 * fp8_e4m3fn -> fp32 under CAST_NONE, so the fp8 modes expand in one Cast.  fp4
 * has no fp32 leg at all -- the only conversions are bf16 <-> fp4x2_e2m1 -- so
 * kv3fp4 goes fp32 -> bf16 -> fp4x2_e2m1.  bf16's 8 mantissa bits carry every
 * e2m1 value exactly, so the detour costs an instruction and no accuracy.  Read
 * the SupportType lists in
 * .../impl/basic_api/dav_3510/kernel_operator_vec_vconv_impl.h before assuming
 * otherwise.
 *
 * THE GATHER AND THE CAST ARE BOTH EXACT.  Every centroid in
 * TurboQuantModeTraits is a grid point of the mode's operand type by
 * construction (the gain search in scripts/tq_multimode_calibration.py is what
 * arranges that), so the codebook error is entirely in the table and none of it
 * in the expand.
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
 * The constant-table image is this codec's own -- it deliberately does NOT
 * share TurboQuantCodec<4>::ConstTableWords.  Sharing was the obvious thing and
 * is wrong: that image is pinned by five mirrors (the Python builder, the C++
 * host mirror, the CPU reference and two tests), and widening it to carry a
 * 32-entry codebook and the radix tables would move every offset in it.  A
 * separate image leaves the shipping 4-bit cache bit-identical.
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
 * layout is one formula rather than a conditional the host mirror would have to
 * reproduce; B words is a cheap price for that.  vecLen is a power of two >= 64
 * and kLevels is 8, 16 or 32, so every section boundary is a whole 32-byte
 * burst and Init moves the image with one DataCopy.
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
     * The scale never reaches a scalar register, exactly as in
     * TurboQuantCodec<4>::Quantize4Bit.  The codebook gain is NOT applied here:
     * it lives in the stored centroids and comes back out of the scale on the
     * decode side, so the bins are the plain Lloyd-Max bins of N(0,1) and a
     * host reference can check them without knowing the gain.
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
     * The per-vector scale is deliberately not applied: folding it into the
     * score row (K) or into the softmax probabilities (V) is one Mul over
     * `rows` instead of one over `rows * len`, and the fold survives a
     * non-uniform codebook because s * c[q] is still linear in s.  The codebook
     * gain rides along in that same fold -- the decode path multiplies by
     * s / gain -- so it costs nothing here either.
     *
     * unpack_tq5_to_fp8() below is this routine at MODE = KV5_FP8, under the
     * name the pipeline uses.
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

    /*
     * fp32 -> the mode's Cube operand type.
     *
     * Both branches are exact: every centroid is a grid point of the target
     * type, so CAST_RINT never actually rounds.  The fp4 branch is two casts
     * because the arch35 converter has no fp32 <-> fp4 leg -- only
     * bf16 <-> fp4x2_e2m1 -- and bf16 carries every e2m1 value losslessly.
     */
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

    /*
     * x[p] <- (x[p] / radix^(p mod digitsPerByte)) mod radix.
     *
     * Seven instructions, independent of the radix.  The only per-lane quantity
     * is `recip`, a single 32-byte block whose period divides 8.
     */
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
     * A run of 2 or 4 lanes is shorter than a 32-byte block, so WholeReduceSum
     * cannot express its repeat stride and the fold is `dpb` Gathers instead --
     * exactly the deinterleave TurboQuantCodec<4>::Quantize4Bit does at dpb=2.
     * packOffset_ holds the dpb offset tables end to end, each of len/dpb
     * entries, which is why it is `len` words for every mode.
     *
     * The result is biased by -128 into int8, matching the shipping 4-bit
     * packer, so a kv5fp8 low plane and a kv4fp8 slot have the same bit pattern
     * for the same low nibbles.
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

    /*
     * Fold eight 1-bit digits into one byte.
     *
     * Eight lanes is exactly one 32-byte block, so this one *is* expressible as
     * a weighted WholeReduceSum: one Mul against msbWeight_ (2^(p mod 8)) and
     * one reduction whose repeat advances a single block.  Two instructions
     * against the eight Gathers the low-plane form would need.
     */
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

/*
 * The named entry point of the kv5fp8 dequantization stage: packed 5-bit
 * indices in `srcPacked` -> fp8_e4m3fn in `dst`, both in local UB memory.
 *
 * A thin name over TurboQuantModeCodec<KV5_FP8>::Unpack rather than a second
 * implementation, so there is exactly one place the 5-bit layout is decoded and
 * the test that pins it pins the code the decode kernel actually runs.
 */
__aicore__ inline void unpack_tq5_to_fp8(TurboQuantCodecKv5Fp8 &codec,
                                         const AscendC::LocalTensor<fp8_e4m3fn_t> &dst,
                                         const AscendC::LocalTensor<int8_t> &srcPacked, int rows, int len)
{
    codec.Unpack(dst, srcPacked, rows, len);
}

}  // namespace turboquant
}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_ATTENTION_TURBOQUANT_CODEC_MX_H
