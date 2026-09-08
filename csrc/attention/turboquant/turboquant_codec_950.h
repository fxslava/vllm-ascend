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
 *     step      = absmax(Pi x) / 7.5
 *
 * where q in [0, 15] are the 16 uniform levels of the rotated vector
 *
 *     Pi x = D (H (D x)),    D = diag(+-1),   H = normalised Walsh-Hadamard
 *
 * and q = clamp(round(Pi_x / step + 7.5), 0, 15) is a mid-rise grid, so all
 * sixteen levels carry data.  A sign flip is applied *before* the Hadamard
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
 * The 4-bit width is deliberate.  Two codes fill exactly one byte, D/2 bytes
 * stay 32-byte aligned for every D >= 64, and the packed layout is the one the
 * INT4 Cube GEMM path already expects.
 *
 * Everything below stays in the Vector pipeline.  There is no per-element
 * GetValue/SetValue in this header: the sign patterns, shuffle tables and the
 * nibble split are built with ArithProgression / Cast / Gather, and the
 * per-vector scale is broadcast with Brcb rather than read into a scalar
 * register.
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

__aicore__ inline uint32_t UbByteAddr(const AscendC::LocalTensor<float> &tensor)
{
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tensor.GetPhyAddr()));
}

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
    static constexpr float kZeroPoint = kLevelMax * 0.5f;                // 7.5
    static constexpr float kPackHigh = static_cast<float>(kLevels);      // 16
    static constexpr float kInt8Bias = 128.0f;
    // Guards the reciprocal of an all-zero vector; far below fp16 denormals.
    static constexpr float kEps = 1e-20f;

    /*
     * vecLen      head_size, a power of two in [64, 256].
     * batchRows   how many vectors Dequantize4Bit may expand in one call; sizes
     *             the shuffle tables used by the batched path.
     * invSqrtLen  1 / sqrt(vecLen), passed in so the kernel needs no scalar sqrt.
     */
    __aicore__ inline void Init(AscendC::TPipe *pipe, uint32_t vecLen, uint32_t batchRows, float invSqrtLen)
    {
        len_ = vecLen;
        batchLen_ = vecLen * batchRows;
        invSqrtLen_ = invSqrtLen;
        const uint32_t packed = len_ / kPackFactor;

        pipe->InitBuffer(constBuf_, ConstBufferBytes(vecLen, batchRows));
        AscendC::LocalTensor<float> pool = constBuf_.Get<float>();

        uint32_t off = 0;
        for (int stage = 0; stage < kEarlyStages; ++stage) {
            sign_[stage] = pool[off];
            off += len_;
            xorOffset_[stage] = pool[off];
            off += len_;
        }
        evenOffset_ = pool[off];
        off += packed;
        oddOffset_ = pool[off];
        off += packed;
        expandOffset_ = pool[off];
        off += batchLen_;
        oddSelect_ = pool[off];
        off += batchLen_;
        swap_ = pool[off];
        off += batchLen_;
        scratch_ = pool[off];
        off += batchLen_;
        reduceWork_ = pool[off];
        off += len_;
        broadcast_ = pool[off];

        BuildTables();
    }

    __aicore__ static inline uint32_t ConstBufferBytes(uint32_t vecLen, uint32_t batchRows)
    {
        const uint32_t batchLen = vecLen * batchRows;
        const uint32_t elems = kEarlyStages * 2 * vecLen     // sign_ + xorOffset_
                               + 2 * (vecLen / kPackFactor)  // evenOffset_ + oddOffset_
                               + 4 * batchLen                // expandOffset_, oddSelect_, swap_, scratch_
                               + vecLen                      // reduceWork_
                               + kBrcbDstLanes + kFp32PerBlock;  // broadcast_
        return elems * static_cast<uint32_t>(sizeof(float));
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
     * Quantise one rotated vector to 4 bits.
     *
     *   dstPacked  [len / 2] int8, low nibble = channel 2c, high nibble = 2c+1
     *   src        [len]     fp32, already rotated
     *   stepOut    [1]       fp32, absmax / 7.5.  Brcb reads a whole 32B block,
     *                        so the caller must back this with 8 readable lanes.
     *
     * The absmax never reaches a scalar register: ReduceMax leaves it in UB,
     * Brcb splays it across a 32B block, and the reciprocal is applied with a
     * zero-stride Mul.
     */
    __aicore__ inline void Quantize4Bit(const AscendC::LocalTensor<int8_t> &dstPacked,
                                        const AscendC::LocalTensor<float> &src,
                                        const AscendC::LocalTensor<float> &stepOut, int len)
    {
        const uint32_t n = static_cast<uint32_t>(len);
        const uint32_t packed = n / kPackFactor;

        AscendC::Abs(scratch_, src, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceMax<float>(stepOut, scratch_, reduceWork_, n, false);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(stepOut, stepOut, kEps, 1);
        AscendC::PipeBarrier<PIPE_V>();

        // broadcast_[0..7] = absmax; broadcast_[64..71] = 7.5 / absmax.
        AscendC::Brcb(broadcast_, stepOut, 1, {1, static_cast<uint16_t>(kFp32PerBlock)});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(broadcast_[kBrcbDstLanes], kZeroPoint, kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(broadcast_[kBrcbDstLanes], broadcast_[kBrcbDstLanes], broadcast_, kFp32PerBlock);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Muls(stepOut, stepOut, 1.0f / kZeroPoint, 1);

        BroadcastMul(scratch_, src, broadcast_[kBrcbDstLanes], n);
        AscendC::Adds(scratch_, scratch_, kZeroPoint, n);
        AscendC::PipeBarrier<PIPE_V>();
        RoundToLevels(scratch_, n);

        // Deinterleave even/odd channels with two Gathers, then build the byte.
        AscendC::Gather(swap_, scratch_, evenOffset_.ReinterpretCast<uint32_t>(), UbByteAddr(scratch_), packed);
        AscendC::Gather(swap_[packed], scratch_, oddOffset_.ReinterpretCast<uint32_t>(), UbByteAddr(scratch_), packed);
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
     * Expand `rows` packed vectors into centred levels (q - 7.5), fp32.
     *
     * The per-vector `step` is deliberately not applied here.  Folding it into
     * the score vector (for K) or into the softmax probabilities (for V) costs
     * one Mul over `rows` elements instead of one over `rows * len`, and keeps
     * the hot loop free of per-element scale broadcasts.
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
        AscendC::Gather(swap_, scratch_, expandOffset_.ReinterpretCast<uint32_t>(), UbByteAddr(scratch_), n);
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
        AscendC::Adds(dst, dst, -kZeroPoint, n);
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
    // Strides 1, 2 and 4 all live inside a single 32B block.
    static constexpr int kEarlyStages = 3;

    __aicore__ inline void ShuffleStage(AscendC::LocalTensor<float> &x, AscendC::LocalTensor<float> &tmp, int stage,
                                        uint32_t n)
    {
        AscendC::Gather(swap_, x, xorOffset_[stage].ReinterpretCast<uint32_t>(), UbByteAddr(x), n);
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

    __aicore__ inline void RoundToLevels(const AscendC::LocalTensor<float> &x, uint32_t count)
    {
        AscendC::LocalTensor<int32_t> intView = swap_.ReinterpretCast<int32_t>();
        AscendC::Cast(intView, x, AscendC::RoundMode::CAST_RINT, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(x, intView, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(x, x, 0.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(x, x, kLevelMax, count);
        AscendC::PipeBarrier<PIPE_V>();
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

    /*
     * Build every constant table with vector instructions only.
     *
     *   sign_[s][p]      = 1 - 2 * ((p / s) & 1)
     *   xorOffset_[s][p] = 4 * (p ^ s) = 4 * (p + s * sign_[s][p])
     *   evenOffset_[c]   = 8 * c;  oddOffset_[c] = 8 * c + 4
     *   expandOffset_[p] = 4 * (p >> 1)
     *   oddSelect_[p]    = p & 1
     */
    __aicore__ inline void BuildTables()
    {
        AscendC::ArithProgression(scratch_, 0.0f, 1.0f, static_cast<int32_t>(batchLen_));
        AscendC::PipeBarrier<PIPE_V>();

        for (int stage = 0; stage < kEarlyStages; ++stage) {
            const float stride = static_cast<float>(1u << stage);
            // parity = floor(p / s) - 2 * floor(p / 2s)
            AscendC::Muls(swap_, scratch_, 1.0f / stride, len_);
            AscendC::PipeBarrier<PIPE_V>();
            FloorInPlace(swap_, len_);
            AscendC::Muls(sign_[stage], swap_, 0.5f, len_);
            AscendC::PipeBarrier<PIPE_V>();
            FloorInPlace(sign_[stage], len_);
            AscendC::Muls(sign_[stage], sign_[stage], -2.0f, len_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(sign_[stage], sign_[stage], swap_, len_);
            AscendC::PipeBarrier<PIPE_V>();
            // sign = 1 - 2 * parity
            AscendC::Muls(sign_[stage], sign_[stage], -2.0f, len_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(sign_[stage], sign_[stage], 1.0f, len_);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Muls(xorOffset_[stage], sign_[stage], stride, len_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(xorOffset_[stage], xorOffset_[stage], scratch_, len_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(xorOffset_[stage], xorOffset_[stage], static_cast<float>(sizeof(float)), len_);
            AscendC::PipeBarrier<PIPE_V>();
            ToByteOffsets(xorOffset_[stage], len_);

            if (stage == 0) {
                // The stride-1 parity doubles as the nibble selector.
                AscendC::Muls(oddSelect_, sign_[stage], -0.5f, len_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(oddSelect_, oddSelect_, 0.5f, len_);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t row = len_; row < batchLen_; row += len_) {
                    AscendC::DataCopy(oddSelect_[row], oddSelect_, len_);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        const uint32_t packed = len_ / kPackFactor;
        const float pairBytes = static_cast<float>(kPackFactor * sizeof(float));
        AscendC::ArithProgression(evenOffset_, 0.0f, pairBytes, static_cast<int32_t>(packed));
        AscendC::PipeBarrier<PIPE_V>();
        ToByteOffsets(evenOffset_, packed);
        AscendC::ArithProgression(oddOffset_, static_cast<float>(sizeof(float)), pairBytes,
                                  static_cast<int32_t>(packed));
        AscendC::PipeBarrier<PIPE_V>();
        ToByteOffsets(oddOffset_, packed);

        AscendC::Muls(expandOffset_, scratch_, 1.0f / static_cast<float>(kPackFactor), batchLen_);
        AscendC::PipeBarrier<PIPE_V>();
        FloorInPlace(expandOffset_, batchLen_);
        AscendC::Muls(expandOffset_, expandOffset_, static_cast<float>(sizeof(float)), batchLen_);
        AscendC::PipeBarrier<PIPE_V>();
        ToByteOffsets(expandOffset_, batchLen_);
    }

    // Gather consumes uint32 byte offsets.  The tables are built in fp32 -- every
    // value they hold is exactly representable -- and converted in place.
    __aicore__ static inline void ToByteOffsets(const AscendC::LocalTensor<float> &table, uint32_t count)
    {
        AscendC::LocalTensor<int32_t> intView = table.ReinterpretCast<int32_t>();
        AscendC::Cast(intView, table, AscendC::RoundMode::CAST_RINT, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::TBuf<AscendC::QuePosition::VECCALC> constBuf_;
    AscendC::LocalTensor<float> sign_[kEarlyStages];
    AscendC::LocalTensor<float> xorOffset_[kEarlyStages];
    AscendC::LocalTensor<float> evenOffset_;
    AscendC::LocalTensor<float> oddOffset_;
    AscendC::LocalTensor<float> expandOffset_;
    AscendC::LocalTensor<float> oddSelect_;
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
