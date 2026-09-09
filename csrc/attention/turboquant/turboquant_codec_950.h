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
 * where q in [0, 15] indexes the 16-level Lloyd-Max quantiser for N(0, 1) and
 * the rotated vector is
 *
 *     Pi x = D (H (D x)),    D = diag(+-1),   H = normalised Walsh-Hadamard
 *
 * with u = Pi_x / scale, q = sum_{i=1}^{15} [u > t_i] and the reconstruction
 * scale * c[q].  The rotation makes the coordinates of u very close to i.i.d.
 * N(0, 1) -- measured |excess kurtosis| < 0.12 on real activations at every
 * layer -- which is exactly the distribution the table is optimal for, so the
 * codec quantises against the density it actually sees rather than against a
 * uniform grid sized by the single largest coordinate.  The RMS scale is the
 * matching metric: absmax sizes its step off an outlier and leaves the bulk
 * crushed, and on real activations that inflates the reconstructed norm by up
 * to 2.17x.  A sign flip is applied *before* the Hadamard
 * because a bare Hadamard has adversarial inputs -- a vector aligned with one
 * of its rows concentrates all energy into a single coordinate -- and the
 * random diagonal removes them.  The second flip makes Pi symmetric:
 *
 *     Pi^T = D^T H^T D^T = D H D = Pi,      Pi^2 = D H D D H D = D H H D = I
 *
 * so Pi is an orthogonal involution and is its own inverse.  One routine both
 * rotates and un-rotates, and because Pi is orthogonal (Pi q) . (Pi k) == q . k,
 * leaving attention scores untouched; only the quantisation error changes.
 *
 * The reconstruction stays *linear in the scale* -- scale * c[q] rather than
 * scale * (q - 7.5) -- which is the property the decode path depends on:
 * Dequantize4Bit still returns scale-free values and the per-vector scale is
 * still folded into the attention score row (for K) or into the softmax
 * probabilities (for V), one Mul over `rows` instead of over `rows * D`.
 *
 * The 4-bit width is deliberate.  Two codes fill exactly one byte, D/2 bytes
 * stay 32-byte aligned for every D >= 64, and the packed layout is the one the
 * INT4 Cube GEMM path already expects.
 *
 * Everything below stays in the Vector pipeline.  There is no per-element
 * GetValue/SetValue in this header: the sign patterns, shuffle tables and the
 * nibble split are Gather-driven, the bin index comes from a sign bit rather
 * than a comparison mask, and the per-vector RMS scale is broadcast with Brcb
 * rather than read into a scalar register.
 *
 * The constant tables are *not* generated here.  They are a pure function of
 * (head_size, batch_rows), so the host builds them once at engine start and
 * Init() pulls the finished image into UB with a single aligned DataCopy.
 * Generating them per launch cost roughly forty dependent vector instructions
 * plus their pipe barriers on the critical path of every decode step.
 * ConstTableWords() is the layout contract; the host mirror lives in
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

/*
 * `srcBaseAddr` for every AscendC::Gather in this file and in
 * turboquant_kernels.cpp.
 *
 * Gather's fourth argument is a byte offset *within* srcLocal - it names where
 * in the source tensor the offset table starts counting from - and NOT the UB
 * address of srcLocal. Every offset table here already indexes from the start
 * of its source tensor, so the correct value is zero.
 *
 * This was a helper that returned srcLocal->GetPhyAddr(), and the effect was
 * that the tensor's own UB offset got added twice: the gather then read from
 * 2 * base + offset, which lands in whichever buffer happens to sit there.
 * It only produced correct results when the source was the very first UB
 * allocation, where the address is zero and the double-count is invisible -
 * which is why it survived code review and why it reproduces in a unit probe
 * only once a second buffer is allocated ahead of the source. Measured on both
 * the Ascend910B1 and the Ascend950PR_9599 camodel, so it is not arch-specific.
 *
 * Symptom, if it comes back: ApplyPi's stride-1/2/4 stages and the codec's
 * nibble split silently return the contents of an unrelated buffer, the packed
 * cache fills with the -128 that an all-zero nibble pair encodes, and the
 * decode output is uncorrelated with the reference.
 * TurboQuantKernels.ReshapeAndCacheMatchesTheCpuReference is the test that
 * catches it.
 */
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

    /*
     * The 16-level Lloyd-Max quantiser for N(0, 1): the fixed point of
     *
     *     t_i = (c_{i-1} + c_i) / 2,     c_i = E[X | t_i < X < t_{i+1}],
     *
     * solved to machine precision with the closed-form truncated-Gaussian
     * moments rather than from a histogram, and symmetrised exactly.  Its
     * distortion E[(X - Q(X))^2] is 0.0095010080, i.e. 20.222 dB, against
     * 0.01388 for the uniform mid-rise grid this replaced.  Re-derivable with
     * scripts/tq_kv_quant_reference.py::lloyd_max_gaussian_table().
     *
     * The centroids live in UB (see ConstTableWords) because Dequantize4Bit
     * reads them with a Gather.  The thresholds are compile-time constants
     * because Quantize4Bit consumes them as Adds immediates -- putting them in
     * UB would cost fifteen scalar loads and their pipeline stalls to buy
     * nothing, since they are the same on every launch.
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
     * Constant-table layout, in 4-byte words.  The host writes this exact image;
     * Init() copies it in one burst and never rewrites a word of it.
     *
     *   [0 .. 6*len)              sign_[s] then xorOffset_[s], per stage
     *   [6*len .. 7*len)          evenOffset_ then oddOffset_, len/2 words each
     *   [7*len .. 7*len+B)        expandOffset_
     *   [7*len+B .. 7*len+2B)     oddSelect_          (B = len * batchRows)
     *   [7*len+2B .. +kLevels)    centroid_
     *
     * sign_, oddSelect_ and centroid_ are fp32 bit patterns; the offset tables
     * are uint32 byte offsets for Gather.  Every entry is four bytes wide, so
     * one int32 DataCopy moves the lot and the device needs no cast and no
     * arithmetic.
     *
     * The centroid table is appended rather than prepended so that adding it
     * left every pre-existing offset in the image unchanged.  vecLen is a
     * power of two >= 64, so every section boundary -- the centroid table's
     * included -- is a whole 32-byte burst, which is what lets Init() move the
     * image with a single DataCopy and lets Gather read the centroids from an
     * aligned base.
     */
    __aicore__ static inline uint32_t ConstTableWords(uint32_t vecLen, uint32_t batchRows)
    {
        return 7u * vecLen + 2u * vecLen * batchRows + static_cast<uint32_t>(kLevels);
    }

    // Scratch, deliberately uninitialised: swap_, scratch_, reduceWork_, broadcast_.
    __aicore__ static inline uint32_t WorkBufferWords(uint32_t vecLen, uint32_t batchRows)
    {
        return 2u * vecLen * batchRows + vecLen + kBrcbDstLanes + kFp32PerBlock;
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
    }

    /*
     * In-place normalised fast Walsh-Hadamard transform of x[0, len).
     *
     * Stages with stride >= 8 are exactly one Add plus one Sub: the butterfly
     * is expressed with block strides, so a whole stage is a single strided
     * instruction pair and the halves ping-pong between x and tmp.
     *
     * Stages with stride in {1, 2, 4} straddle the 32B block, so those block
     * strides are not integral.  They become a register shuffle instead:
     * Gather materialises x[p ^ stride] in one VGATHER and the butterfly
     * collapses to  out[p] = x[p ^ s] + sign_s[p] * x[p]  -- two more vector
     * instructions, still with no scalar memory round-trip.
     */
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

    /*
     * In-place Pi x = D (H (D x)).
     *
     * Pi is a symmetric orthogonal involution, so this single routine is both
     * the rotation applied to K, V and Q on the way in and the un-rotation
     * applied to the attention output on the way out.  There is no separate
     * inverse to keep in step with it, and no basis the kernel can be left in
     * by accident.
     */
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
     *                        32B block, so the caller must back this with 8
     *                        readable lanes.
     *
     * The scale never reaches a scalar register: ReduceSum leaves the sum of
     * squares in UB, Sqrt and the sqrt(len) normalisation are applied to that
     * single lane, Brcb splays it across a 32B block, and the reciprocal is
     * applied with a zero-stride Mul.
     *
     * The bin index is  q = sum_{i=1}^{15} [u > t_i]  over the fifteen decision
     * boundaries, evaluated without a mask register: for each threshold,
     *
     *     d = t_i - u              one Adds against an immediate
     *     b = uint32(d) >> 31      the fp32 sign bit, 1 exactly when u > t_i
     *     q += b                   an int32 accumulate
     *
     * Three vector instructions per boundary and no scalar round trip.  The
     * subtraction is written t_i - u rather than u - t_i so that the sign bit
     * answers the strict `>` the table's boundaries are defined with; a
     * coordinate sitting exactly on t_i yields +0.0 and rounds down, which is
     * the same tie-break the host reference takes.
     *
     * u is materialised as its own negation, so the Adds above needs no second
     * operand negation per boundary.
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
        AscendC::LocalTensor<uint32_t> signBits = swap_.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<int32_t> signInts = swap_.ReinterpretCast<int32_t>();
        AscendC::Duplicate(bins, 0, n);
        AscendC::PipeBarrier<PIPE_V>();
        for (int i = 0; i < kThresholdCount; ++i) {
            AscendC::Adds(swap_, scratch_, Threshold(i), n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftRight(signBits, signBits, kSignBitShift, static_cast<int32_t>(n));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(bins, bins, signInts, n);
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
     * The per-vector `scale` is deliberately not applied here.  Folding it into
     * the score vector (for K) or into the softmax probabilities (for V) costs
     * one Mul over `rows` elements instead of one over `rows * len`, and keeps
     * the hot loop free of per-element scale broadcasts.  That fold survives
     * the non-uniform table only because scale * c[q] is still linear in the
     * scale; a codec whose reconstruction were affine in it would not be
     * expressible this way, and the decode path would need a broadcast per
     * element.
     *
     * The final step is the codec's only data-dependent Gather -- every other
     * one in this file indexes a host-built constant table -- and it replaces
     * the uniform grid's single Adds(-7.5) with Muls, Cast and Gather: two
     * extra full-length vector instructions on the decode critical path, about
     * +22% on this routine.
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

        // dst[p] = centroid_[q_p].  Gather counts byte offsets from the start
        // of centroid_ (see kGatherSrcBase), so the index is scaled by the word
        // size before it is cast.  swap_ and scratch_ are both spent by here --
        // swap_ held the packed byte per channel, scratch_ the low nibble --
        // so the offset table needs no buffer of its own.
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

    /*
     * One butterfly stage for stride >= 8, as a single Add/Sub pair.
     *
     * Element j of group i lives at i * 2 * stride + j, so a repeat covers
     * `stride` lanes and advances 2 * stride lanes -- both whole 32B blocks
     * once stride >= 8.  A stride wider than one repeat (> 64 lanes) falls back
     * to a short loop of contiguous Add/Sub: still one instruction pair per
     * group, still entirely in the vector pipe.
     */
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
    uint32_t len_ = 0;
    uint32_t batchLen_ = 0;
    float invSqrtLen_ = 1.0f;
};

using TurboQuantCodec4 = TurboQuantCodec<4>;

}  // namespace turboquant
}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_ATTENTION_TURBOQUANT_CODEC_950_H
