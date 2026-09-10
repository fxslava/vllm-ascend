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
 * TurboQuant KV-cache codec for Ascend 950PR (arch35) and 910B (arch32).
 *
 * A KV vector of head_size channels is stored as
 *
 *     packed[c] = int8((q[2c] + 16 * q[2c + 1]) - 128),   c in [0, D/2)
 *     scale     = ||Pi x||_2 / sqrt(D)
 *
 * where q in [0, 15] indexes the 16-level Lloyd-Max quantiser for N(0, 1),
 * u = Pi x / scale, q = sum_{i=1}^{15} [u > t_i], and the reconstruction is
 * scale * c[q].  The rotation is
 *
 *     Pi x = D (H (D x)),    D = diag(+-1),   H = normalised Walsh-Hadamard
 *
 * a symmetric orthogonal involution, so one routine both rotates and
 * un-rotates.  Rotation is applied to activations only; no weight is rewritten.
 *
 * ConstTableWords() is the constant-table layout contract; the host mirror is
 * vllm_ascend/attention/turboquant_v1.py::turboquant_codec_tables.
 */

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_CODEC_950_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_CODEC_950_H

#include "kernel_operator.h"

namespace vllm_ascend {
namespace turboquant {

// fp32 lanes in one 32B UB block; also the smallest butterfly stride that can
// be expressed with block-strided Add/Sub.
constexpr uint32_t kFp32PerBlock = 8;
// fp32 lanes covered by a single vector instruction repeat.
constexpr uint32_t kFp32PerRepeat = 64;
// Brcb consumes eight source lanes and emits eight 32B blocks per repeat, so
// broadcasting one value needs 8 readable source lanes and 64 lanes of
// destination -- only the first block of which is meaningful.
constexpr uint32_t kBrcbSrcLanes = kFp32PerBlock;
constexpr uint32_t kBrcbDstLanes = kFp32PerBlock * kFp32PerBlock;

// Gather's srcBaseAddr is a byte offset *within* srcLocal, not the UB address
// of srcLocal.  Every offset table here already indexes from the start of its
// source tensor, so the correct value is zero.
constexpr uint32_t kGatherSrcBase = 0;

/*
 * b-bit TurboQuant codec.  Only b == 4 is instantiated; the template keeps the
 * level arithmetic in one place should an 8-bit variant ever be added.
 */
template <int BITS>
class TurboQuantCodec {
    static_assert(BITS == 4, "TurboQuantCodec is only implemented for b = 4");

public:
    static constexpr int kBits = BITS;
    static constexpr int kLevels = 1 << BITS;                            // 16
    static constexpr uint32_t kPackFactor = 8 / BITS;                    // codes per byte
    static constexpr float kLevelMax = static_cast<float>(kLevels - 1);  // 15
    static constexpr float kPackHigh = static_cast<float>(kLevels);      // 16
    static constexpr float kInt8Bias = 128.0f;
    // Guards the reciprocal of an all-zero vector; far below fp16 denormals.
    static constexpr float kEps = 1e-20f;
    // Strides 1, 2 and 4 all live inside a single 32B block.
    static constexpr int kEarlyStages = 3;
    // Decision boundaries, one fewer than there are levels.
    static constexpr int kThresholdCount = kLevels - 1;                  // 15
    // Gather consumes byte offsets, and the centroid table is fp32.
    static constexpr float kCentroidStride = static_cast<float>(sizeof(float));
    // Bit position of the fp32 sign, which is how a comparison is turned into
    // an integer 0/1 without a mask register.
    static constexpr uint32_t kSignBitShift = 31;
    // Boundary tests Quantize4Bit keeps in flight, one scratch buffer each.
    static constexpr int kBinLanes = 8;

    /*
     * The 16-level Lloyd-Max quantiser for N(0, 1): the fixed point of
     *
     *     t_i = (c_{i-1} + c_i) / 2,     c_i = E[X | t_i < X < t_{i+1}],
     *
     * re-derivable with
     * scripts/tq_kv_quant_reference.py::lloyd_max_gaussian_table().
     * The centroids reach UB (see ConstTableWords); the thresholds stay
     * compile-time constants because Quantize4Bit uses them as Adds immediates.
     */
    __aicore__ static inline float Threshold(int i)
    {
        constexpr float kThresholds[kThresholdCount] = {
            -2.4008033987632817f,  -1.8435318062766393f,  -1.4371387916845330f, -1.0992858269170722f,
            -0.7995497875097155f,  -0.5224037090113789f,  -0.2582216646707196f, 0.0f,
            0.2582216646707196f,   0.5224037090113789f,   0.7995497875097155f,  1.0992858269170722f,
            1.4371387916845330f,   1.8435318062766393f,   2.4008033987632817f};
        return kThresholds[i];
    }

    /*
     * Constant-table layout, in 4-byte words.  The host writes this exact image
     * and Init() copies it in one DataCopy.
     *
     *   [0 .. 6*len)              sign_[s] then xorOffset_[s], per stage
     *   [6*len .. 7*len)          evenOffset_ then oddOffset_, len/2 words each
     *   [7*len .. 7*len+B)        expandOffset_
     *   [7*len+B .. 7*len+2B)     oddSelect_          (B = len * batchRows)
     *   [7*len+2B .. +kLevels)    centroid_
     *
     * sign_, oddSelect_ and centroid_ are fp32 bit patterns; the offset tables
     * are uint32 byte offsets for Gather.  Every entry is four bytes wide.
     */
    __aicore__ static inline uint32_t ConstTableWords(uint32_t vecLen, uint32_t batchRows)
    {
        return 7u * vecLen + 2u * vecLen * batchRows + static_cast<uint32_t>(kLevels);
    }

    // Scratch, deliberately uninitialised. Purely internal: no host mirror.
    __aicore__ static inline uint32_t WorkBufferWords(uint32_t vecLen, uint32_t batchRows)
    {
        return 2u * vecLen * batchRows + vecLen + kBrcbDstLanes + kFp32PerBlock +
               static_cast<uint32_t>(kBinLanes) * vecLen;
    }

    /*
     * vecLen      head_size, a power of two in [64, 256].
     * batchRows   how many vectors Dequantize4Bit may expand in one call; must
     *             match the batchRows the host built `tablesGm` for.
     * invSqrtLen  1 / sqrt(vecLen), passed in so the kernel needs no scalar sqrt.
     * tablesGm    ConstTableWords(vecLen, batchRows) int32 words.
     */
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

    // In-place normalised fast Walsh-Hadamard transform of x[0, len).
    // Strides 1, 2 and 4 straddle the 32B block and run as a Gather shuffle
    // rather than a block-strided Add/Sub pair.
    __aicore__ inline void FastWalshHadamardTransform(AscendC::LocalTensor<float> &x,
                                                      AscendC::LocalTensor<float> &tmp, int len)
    {
        const uint32_t n = static_cast<uint32_t>(len);

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
            BlockStage(dst, src, stride, n);
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
        AscendC::PipeBarrier<PIPE_V>();
    }

    // In-place Pi x = D (H (D x)).  Pi is a symmetric orthogonal involution, so
    // this is both the rotation and the un-rotation.
    __aicore__ inline void ApplyPi(AscendC::LocalTensor<float> &x, AscendC::LocalTensor<float> &tmp,
                                   const AscendC::LocalTensor<float> &piSigns, int len)
    {
        const uint32_t n = static_cast<uint32_t>(len);
        AscendC::Mul(x, x, piSigns, n);
        AscendC::PipeBarrier<PIPE_V>();
        FastWalshHadamardTransform(x, tmp, len);
        AscendC::Mul(x, x, piSigns, n);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /*
     * Quantise one rotated vector to 4 bits against the Lloyd-Max table.
     *
     *   dstPacked  [len / 2] int8, low nibble = channel 2c, high nibble = 2c+1
     *   src        [len]     fp32, already rotated
     *   scaleOut   [1]       fp32, ||src||_2 / sqrt(len).  Brcb reads a whole
     *                        32B block, so back this with 8 readable lanes.
     *
     * The bin index is q = sum_{i=1}^{15} [u > t_i].  The difference is taken
     * as t_i - u so its sign bit answers the strict `>` the table's boundaries
     * are defined with, which is the tie-break the host reference takes.
     */
    __aicore__ inline void Quantize4Bit(const AscendC::LocalTensor<int8_t> &dstPacked,
                                        const AscendC::LocalTensor<float> &src,
                                        const AscendC::LocalTensor<float> &scaleOut, int len)
    {
        const uint32_t n = static_cast<uint32_t>(len);
        const uint32_t packed = n / kPackFactor;

        // scale = ||src||_2 / sqrt(len), the RMS the Lloyd-Max table is stated
        // in.  invSqrtLen_ is the same reciprocal the Hadamard normalises with,
        // so the kernel still needs no scalar sqrt of its own.
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

        // broadcast_[0..7] = scale; broadcast_[64..71] = -1 / scale.
        AscendC::Brcb(broadcast_, scaleOut, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(broadcast_[kBrcbDstLanes], -1.0f, kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(broadcast_[kBrcbDstLanes], broadcast_[kBrcbDstLanes], broadcast_, kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();

        // scratch_ = -u = -src / scale.
        BroadcastMul(scratch_, src, broadcast_[kBrcbDstLanes], n);

        // reduceWork_ is the ReduceSum scratch, and the reduction is finished:
        // reusing it as the bin accumulator keeps the codec's UB footprint
        // unchanged by the switch to a non-uniform table.
        AscendC::LocalTensor<int32_t> bins = reduceWork_.ReinterpretCast<int32_t>();
        AscendC::Duplicate(bins, 0, n);
        AscendC::PipeBarrier<PIPE_V>();

        for (int base = 0; base < kThresholdCount; base += kBinLanes) {
            const int lanes =
                (kThresholdCount - base) < kBinLanes ? (kThresholdCount - base) : kBinLanes;

            // t_i - u, written the way round that puts the answer to the strict
            // comparison in the sign bit.  Independent across lanes.
            for (int lane = 0; lane < lanes; ++lane) {
                AscendC::Adds(binLane_[lane], scratch_, Threshold(base + lane), n);
            }
            AscendC::PipeBarrier<PIPE_V>();

            // Sign bit -> integer 0/1, in place, still independent across lanes.
            for (int lane = 0; lane < lanes; ++lane) {
                AscendC::LocalTensor<uint32_t> bits = binLane_[lane].ReinterpretCast<uint32_t>();
                AscendC::ShiftRight(bits, bits, kSignBitShift, static_cast<int32_t>(n));
            }
            AscendC::PipeBarrier<PIPE_V>();

            // Pairwise tree over the lanes: log2(lanes) barriers rather than one
            // per boundary.  Every Add in a round is independent of the others.
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
        // q lands in [0, 15] by construction -- there are fifteen boundaries --
        // so the uniform grid's clamp has nothing left to do.
        AscendC::Cast(scratch_, bins, AscendC::RoundMode::CAST_NONE, n);
        AscendC::PipeBarrier<PIPE_V>();

        // Deinterleave even/odd channels with two Gathers, then build the byte.
        AscendC::Gather(swap_, scratch_, evenOffset_, kGatherSrcBase, packed);
        AscendC::Gather(swap_[packed], scratch_, oddOffset_, kGatherSrcBase, packed);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(swap_[packed], swap_[packed], kPackHigh, packed);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(swap_, swap_, swap_[packed], packed);
        AscendC::PipeBarrier<PIPE_V>();
        // [0, 255] -> [-128, 127].  The decoder adds the bias back numerically,
        // so nothing here depends on the int8 bit pattern.
        AscendC::Adds(swap_, swap_, -kInt8Bias, packed);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<half> halfView = scratch_.ReinterpretCast<half>();
        AscendC::Cast(halfView, swap_, AscendC::RoundMode::CAST_NONE, packed);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(dstPacked, halfView, AscendC::RoundMode::CAST_RINT, packed);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /*
     * Expand `rows` packed vectors into Lloyd-Max centroids c[q], fp32.
     *
     * The per-vector `scale` is not applied here: the decode path folds it into
     * the score vector (K) or the softmax probabilities (V), which is valid
     * because scale * c[q] is linear in the scale.
     */
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

        // swap_[p] = byte[p >> 1] for every output channel p.
        AscendC::Gather(swap_, scratch_, expandOffset_, kGatherSrcBase, n);
        AscendC::PipeBarrier<PIPE_V>();

        // high = floor(byte / 16); low = byte - 16 * high.
        AscendC::Muls(dst, swap_, 1.0f / kPackHigh, n);
        AscendC::PipeBarrier<PIPE_V>();
        FloorInPlace(dst, n);
        AscendC::Muls(scratch_, dst, -kPackHigh, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(scratch_, scratch_, swap_, n);
        AscendC::PipeBarrier<PIPE_V>();

        // Odd channels take the high nibble: q = low + oddSelect * (high - low).
        AscendC::Sub(dst, dst, scratch_, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(dst, dst, oddSelect_, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(dst, dst, scratch_, n);
        AscendC::PipeBarrier<PIPE_V>();

        // Gather counts byte offsets from the start of centroid_, so the index
        // is scaled by the word size before it is cast.
        AscendC::Muls(scratch_, dst, kCentroidStride, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<int32_t> offsets = swap_.ReinterpretCast<int32_t>();
        AscendC::Cast(offsets, scratch_, AscendC::RoundMode::CAST_RINT, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(dst, centroid_, swap_.ReinterpretCast<uint32_t>(), kGatherSrcBase, n);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // dst[i] = src[i] * scalarBlock[i % 8], for a 32B block whose lanes are all
    // equal.  Used to apply a reduction result without reading it to a scalar.
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
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    __aicore__ inline void ShuffleStage(AscendC::LocalTensor<float> &x, AscendC::LocalTensor<float> &tmp, int stage,
                                        uint32_t n)
    {
        AscendC::Gather(swap_, x, xorOffset_[stage], kGatherSrcBase, n);
        AscendC::Mul(tmp, x, sign_[stage], n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(x, swap_, tmp, n);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // One butterfly stage for stride >= 8, as a single Add/Sub pair.  A stride
    // wider than one repeat (> 64 lanes) falls back to a short contiguous loop.
    __aicore__ inline void BlockStage(AscendC::LocalTensor<float> &dst, AscendC::LocalTensor<float> &src,
                                      uint32_t stride, uint32_t n)
    {
        const uint32_t groups = n / (2 * stride);
        if (stride <= kFp32PerRepeat) {
            const uint8_t rep = static_cast<uint8_t>(2 * stride / kFp32PerBlock);
            const AscendC::BinaryRepeatParams params{1, 1, 1, rep, rep, rep};
            const uint64_t mask = static_cast<uint64_t>(stride);
            AscendC::Add(dst, src, src[stride], mask, static_cast<uint8_t>(groups), params);
            AscendC::Sub(dst[stride], src, src[stride], mask, static_cast<uint8_t>(groups), params);
            return;
        }
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t base = g * 2 * stride;
            AscendC::Add(dst[base], src[base], src[base + stride], stride);
            AscendC::Sub(dst[base + stride], src[base], src[base + stride], stride);
        }
    }

    // floor() via the fp32 <-> int32 converter, which is available on every
    // supported arch; the dedicated Floor intrinsic is not.
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

}  // namespace turboquant
}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_ATTENTION_TURBOQUANT_CODEC_950_H
