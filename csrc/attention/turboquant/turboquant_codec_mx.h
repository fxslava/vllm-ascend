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
 * Multi-rate TurboQuant codec support for the MX packing format.
 *
 * The codec expands low-bit packed KV planes to the operand format used by
 * the Cube-native GEMM kernels and keeps the encode/decode table contract.
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

// The affine rate.  kLowPerByte is 2 as for kv5fp8, but the two digits of a
// byte are coordinates b and b + len/2, not 2b and 2b + 1, and each is a
// signed nibble; see PackAffinePlane.
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
 * `batchRows`  for a codebook mode, vectors Unpack may expand in one call: the
 *              low/msb offset tables are built for exactly this many and are not
 *              valid for more.  For the affine mode there are no offset tables
 *              and UnpackAffine is byte-major rather than row-major, so the
 *              product vecLen * batchRows is read as the number of operand
 *              *elements* one call produces -- twice the bytes it consumes.
 *              Encode always runs one vector at a time.
 *
 * The constant-table image is this codec's own and deliberately does not share
 * TurboQuantCodec<4>::ConstTableWords, which five mirrors pin.  A CODEBOOK mode
 * carries
 *
 *   [0 .. B)             lowOffset_   uint32 byte offsets,  B = vecLen*batchRows
 *   [B .. 2B)            msbOffset_   uint32 byte offsets
 *   [2B .. 2B+8)         lowRecip_    fp32 block, radix^-(p mod kLowPerByte)
 *   [2B+8 .. 2B+16)      msbRecip_    fp32 block, 2^-(p mod 8)
 *   [2B+16 .. 2B+24)     msbWeight_   fp32 block, 2^(p mod 8)
 *   [2B+24 .. +vecLen)   packOffset_  uint32, the encoder's low-plane gathers
 *   [.. + kLevels)       centroid_    the mode's stored codebook
 *
 * msbOffset_ is allocated even for a mode with no msb plane, so the layout is
 * one formula rather than a conditional.
 *
 * The AFFINE mode carries NOTHING: every table above exists to address a byte or
 * a centroid, and it does neither.  ConstTableWords is 0 and Init allocates no
 * constant buffer; turboquant_host::ModeTables still emits one zero block for
 * it, because the launch boundary wants a bindable pointer.
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
    // Uniform levels, so the stored code is the level: no centroid table, and
    // the expand is UnpackAffine rather than Unpack.
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

    // Elements of the Cube operand tensor one expanded vector occupies.  fp4 is
    // half a byte and LocalTensor<fp4x2_e2m1_t> counts *pairs*, so a d=256 fp4
    // row is 128 elements and an fp8 row is 256.
    __aicore__ static constexpr uint32_t OperandElems(uint32_t vecLen)
    {
        return kIsFp4 ? vecLen / 2 : vecLen;
    }

    // Same shape of contract as TurboQuantCodec<4>::ConstTableWords: the host
    // writes this exact image and the device never rewrites a word of it.  Zero
    // for the affine mode, which reads no constant table at all.
    __aicore__ static inline uint32_t ConstTableWords(uint32_t vecLen, uint32_t batchRows)
    {
        if constexpr (kIsAffine) {
            return 0u;
        } else {
            return 2u * vecLen * batchRows + kPeriodicWords + vecLen + static_cast<uint32_t>(kLevels);
        }
    }

    /*
     * Scratch, deliberately uninitialised.
     *
     * A codebook mode carries four full-length buffers through the expand.  The
     * affine mode carries two half-length ones -- it works on packed bytes, of
     * which there are half as many as channels -- which is what pays for the
     * decode unpacking a whole 64-row tile per call instead of an eight-row
     * band.
     *
     * reduceWork_, broadcast_ and the boundary lanes belong to the encoder and
     * are sized off vecLen because it only ever sees one vector.  low_ and
     * scratch_ are the encoder's too on the affine path; the expand there writes
     * into expand_ and msb_ only.
     */
    __aicore__ static inline uint32_t WorkBufferWords(uint32_t vecLen, uint32_t batchRows)
    {
        const uint32_t shared = kBrcbDstLanes + kFp32PerBlock + static_cast<uint32_t>(kBinLanes) * vecLen;
        if constexpr (kIsAffine) {
            return vecLen * batchRows + 3u * vecLen + shared;
        } else {
            return 4u * vecLen * batchRows + vecLen + shared;
        }
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
            // batchLen_ is an element count on this path, so the two byte-major
            // buffers are half of it each; see the class comment.  Every base
            // stays 32-byte aligned because vecLen is a multiple of 64.
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

    /*
     * Encode one rotated vector at kBits.
     *
     *   dstPacked  [PackedBytes(len)] int8, low plane then msb plane
     *   src        [len]              fp32, already rotated
     *   scaleOut   [1]                fp32, ||src||_2 / sqrt(len).  Brcb reads a
     *                                 whole 32B block, so back it with 8 lanes.
     *
     * The codebook gain is not applied here: it lives in the stored centroids
     * and comes back out of the scale on the decode side.  On the affine path
     * there are no stored centroids and the same field is 1 / step, which the
     * uniform thresholds already carry -- so again nothing is applied here.
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

        if constexpr (kIsAffine) {
            PackAffinePlane(dstPacked, low_, n);
        } else {
            PackLowPlane(dstPacked, low_, n);
            if constexpr (kHasMsbPlane) {
                PackMsbPlane(dstPacked[LowPlaneBytes(len_)], msb_, n);
            }
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
        static_assert(!kIsAffine, "an affine mode expands through UnpackAffine; it has no centroid table");
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
     * The affine expand: `bytes` packed bytes -> 2 * `bytes` operand elements,
     * with no Gather, no table and no integer shift.
     *
     *   dstLow    [bytes] coordinates [0, len/2) of each vector the byte run covers
     *   dstHigh   [bytes] coordinates [len/2, len)
     *   srcPacked [bytes] int4x2, two signed nibbles s = q - 8 per byte; see
     *                     PackAffinePlane
     *
     * Eight vector instructions, where the unsigned-nibble shift chain this
     * replaced took thirteen and wrote 44 bytes per packed byte against 26:
     *
     *   * Cast<half, int4b_t> is the vector unit's own signed-nibble converter,
     *     a UNPK4_B8 load plus vcvt_s42f16, over the whole run at once.
     *   * DeInterleave takes even lanes from odd -- the element-stride-2 move
     *     the plane split otherwise has no instruction for -- so both planes
     *     still come out as contiguous runs, and the caller still places each at
     *     a C0-block run of an already-NZ L1 image with a flat DataCopy.
     *   * s + kSignedLevelOffset is q - kAffineBias, so the levels are unchanged
     *     and the quantizer step still lives in TurboQuantModeTraits::kGain.
     *
     * The route stages through fp32 per plane because arch35 has neither an
     * fp32 <- int4 nor an fp8 <- half converter leg.  An unsupported Cast pair
     * compiles clean and emits nothing -- ASCENDC_ASSERT is empty in a device
     * build -- so a new pair has to be checked against the tuple table in
     * dav_3510/kernel_operator_vec_vconv_impl.h, not against the compiler.
     *
     * UB is what the shift chain used: the nibbles are staged in msb_ (2 * bytes
     * halfs), the planes in expand_ (bytes halfs each), and each plane's fp32 in
     * msb_ again once DeInterleave has consumed the nibbles.
     */
    template <typename OperandT>
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
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<half> planes = expand_.ReinterpretCast<half>();
        AscendC::DeInterleave(planes, planes[bytes], nibbles, static_cast<int32_t>(elems));
        AscendC::PipeBarrier<PIPE_V>();

        ExpandNibblePlane(dstLow, planes, bytes);
        ExpandNibblePlane(dstHigh, planes[bytes], bytes);
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
    // The affine storage: a nibble holds q - kNibbleSignShift, which puts the
    // sixteen codes on int4's [-8, 7], and the level is that plus
    // kSignedLevelOffset (8 - 7.5 = 0.5 for kv4fp8).
    static constexpr float kNibbleSignShift = static_cast<float>(Planes::kLowRadix / 2);
    static constexpr float kSignedLevelOffset = kNibbleSignShift - kAffineBias;

    // One plane of UnpackAffine: half -> fp32 -> level -> operand.  msb_ is free
    // again here, because DeInterleave has already consumed the nibbles it held.
    template <typename OperandT>
    __aicore__ inline void ExpandNibblePlane(const AscendC::LocalTensor<OperandT> &dst,
                                             const AscendC::LocalTensor<half> &plane, uint32_t bytes)
    {
        AscendC::Cast(msb_, plane, AscendC::RoundMode::CAST_NONE, bytes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(msb_, msb_, kSignedLevelOffset, bytes);
        AscendC::PipeBarrier<PIPE_V>();
        CastToOperand(dst, msb_, bytes);
    }

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
     * The affine packer: fold the vector's two halves into one plane of signed
     * nibbles, the layout UnpackAffine undoes.
     *
     *     byte[b] = int4x2( q[b] - 8, q[b + n/2] - 8 )
     *
     * Interleave puts the halves on even and odd lanes and Cast<int4b_t> packs
     * each lane pair into one byte -- the same converter pair UnpackAffine reads
     * back through, so which nibble of a byte is lane 0 cannot disagree between
     * the two; the headers do not document it, and nothing on the host reads
     * these bytes.  There is no -128 byte bias any more.  Four instructions,
     * and no packOffset_ Gather -- the interleaved packer below needs one
     * because its digits are kLowPerByte apart.
     *
     * digits is clobbered.  scratch_ holds the n half levels and expand_ the n
     * interleaved ones; Encode is finished with both by the time it calls this.
     */
    __aicore__ inline void PackAffinePlane(const AscendC::LocalTensor<int8_t> &dst,
                                           const AscendC::LocalTensor<float> &digits, uint32_t n)
    {
        const uint32_t bytes = n / 2u;
        AscendC::Adds(digits, digits, -kNibbleSignShift, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<half> levels = scratch_.ReinterpretCast<half>();
        AscendC::Cast(levels, digits, AscendC::RoundMode::CAST_RINT, n);
        AscendC::PipeBarrier<PIPE_V>();

        // dst0 = interleave(src0[:bytes/2], src1[:bytes/2]), dst1 = the rest, so
        // pointing dst1 at dst0 + bytes gives one contiguous interleaved run.
        AscendC::LocalTensor<half> woven = expand_.ReinterpretCast<half>();
        AscendC::Interleave(woven, woven[bytes], levels, levels[bytes], static_cast<int32_t>(bytes));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(dst.ReinterpretCast<int4b_t>(), woven, AscendC::RoundMode::CAST_RINT, n);
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

// The named entry point of the kv4fp8 dequantization stage: packed signed
// INT4 codes in `srcPacked` -- two nibbles per byte, 128 bytes for a d=256
// vector -- to fp8_e4m3fn, in local UB memory throughout.  A thin name over
// TurboQuantModeCodec<KV4_FP8>::UnpackAffine.
//
// There is NO codebook and NO Gather on this path: the stored nibble is the
// reconstruction level less 0.5 (so s + 0.5 = q - 7.5, exact in e4m3fn) and the
// quantizer step is folded into the per-vector scale through
// TurboQuantModeTraits::kGain.  The byte run is expanded by the vector unit's
// int4x2 converter and split into its two planes by one DeInterleave, and
// because the packing is plane-split the two planes come out as contiguous
// runs -- so the caller places each with a flat DataCopy and nothing shuffles.
//
// `bytes` is a byte count, not a row count: the expand is byte-major and does
// not need to know how many vectors the run spans.
__aicore__ inline void unpack_tq4_to_fp8(TurboQuantCodecKv4Fp8 &codec,
                                         const AscendC::LocalTensor<fp8_e4m3fn_t> &dstLow,
                                         const AscendC::LocalTensor<fp8_e4m3fn_t> &dstHigh,
                                         const AscendC::LocalTensor<int8_t> &srcPacked, uint32_t bytes)
{
    codec.UnpackAffine(dstLow, dstHigh, srcPacked, bytes);
}

// The kv5fp8 twin: packed 5-bit indices -> fp8_e4m3fn.  The low plane is laid
// out exactly as kv4fp8's whole slot, so the two share the unpacker.
__aicore__ inline void unpack_tq5_to_fp8(TurboQuantCodecKv5Fp8 &codec,
                                         const AscendC::LocalTensor<fp8_e4m3fn_t> &dst,
                                         const AscendC::LocalTensor<int8_t> &srcPacked, int rows, int len)
{
    codec.Unpack(dst, srcPacked, rows, len);
}

}  // namespace turboquant
}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_ATTENTION_TURBOQUANT_CODEC_MX_H
