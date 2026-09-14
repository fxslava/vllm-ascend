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
 * npu_turboquant_rotate_q -- the query rotation, lifted out of the decode.
 *
 * TurboQuant's rotation is Pi x = D (H (D x)) with H the normalised
 * Walsh-Hadamard transform of the head dimension and D a fixed +-1 sign vector.
 * Until now both decode splits applied it per task, inside the latency-critical
 * loop, once per (token, kv_head, split) and once more per query head of the
 * group. This kernel applies it once per (token, head) and writes the rotated
 * query to GM; the splits read it and do no transform at all.
 *
 * The KV cache is already stored rotated (turboquant_reshape_and_cache), and Pi
 * is orthogonal, so <Pi q, Pi k> = <q, k> and nothing downstream changes basis.
 * The output's inverse rotation is folded into the output projection's weights;
 * a layer that cannot fold it -- an elementwise output gate between attention
 * and W_o -- launches this same kernel on the attention output instead, which
 * is correct because Pi is an involution.
 *
 * TWO PATHS, ONE OPERATOR
 *
 * The host picks between them; see PlanRotateQ in turboquant_torch_adpt.h.
 *
 *   Cube   N >= 16 and N % 16 == 0. Sylvester factorises H_D = H_R (x) H_16, so
 *          for D = 16 R the four butterfly stages of strides 1, 2, 4 and 8 are a
 *          right multiply by a static 16 x 16 constant -- one Mmad over a whole
 *          chunk of vectors, at any D -- and only the log2(R) row-butterflies
 *          above stride 8 stay on the vector unit. Ported from the spike in
 *          csrc/tests/sim/sim_hadamard_hybrid_kernels.cpp, which ran this shape
 *          on both the arch35 camodel and silicon.
 *
 *   AIV    everything else. All log2(D) stages on the vector unit, through
 *          TurboQuantCodec4::ApplyPi -- the same routine, the same call, the
 *          same order of operations the decode used to make per head. The
 *          output is therefore BIT-IDENTICAL to the pre-refactor decode, which
 *          is what makes the AIV path the reference the Cube path is gated
 *          against.
 *
 * WHY THE CUBE PATH NEEDS 16 VECTORS AND NOT 2
 *
 * A Mmad against a 16 x 16 constant costs the same whether the A operand is one
 * fractal row or a hundred, and the staging around it -- the fp16 cast, the L1
 * copy, the LoadData pair, the Fixpipe -- is a fixed cost per chunk. The device
 * sweep in hadamard_benchmark_results.csv puts the break-even at 16 vectors: at
 * D = 256 the Cube form is within noise of the vector form at 8 and 1.36x ahead
 * at 32. Below 16 the host does not select it, and the gate is exact
 * divisibility rather than a threshold because a chunk that does not divide the
 * block's share would need a short final Mmad.
 *
 * PRECISION
 *
 * Mmad's fp16 path is <float, half, half>: half operands, fp32 accumulate. H_16
 * holds only +-1, both exact, so the Cube stage contributes no error of its own
 * and the whole question is whether the A operand survives the fp32 -> fp16
 * cast.
 *
 *   * A half query does. The sign multiply is by +-1, so D x is exactly
 *     representable in half whenever x is, and one Mmad is exact.
 *   * A bfloat16 query does not, in general: its 8-bit exponent reaches values
 *     half cannot hold, and although activations at inference scale sit well
 *     inside half's range, "well inside" is not a property this kernel can
 *     check. kVariantHiLo splits the operand as x = hi + lo with hi = half(x)
 *     and issues two Mmads accumulating into one L0C (cmatrixInitVal false on
 *     the second), which put the spike's error at 2.38e-7 against 3.58e-4 for
 *     about 2% more ticks.
 *
 * The host sets the bit; the kernel does not infer it from scalar_t, so a
 * caller that wants the safe path for half data can ask for it.
 *
 * TWO SUBCORES
 *
 * FixpipeParamsArch3510::dualDstCtl = 0b01 writes M/2 x N into EACH subcore's
 * private UB, so the Cube produces a chunk once and each subcore finishes its
 * own half of the residual with no exchange and no GM round trip. The split
 * lands at row M/2, which is a vector boundary only when the chunk holds an even
 * number of vectors, so DualDst() tests that rather than honouring the bit
 * wrongly. Without it the product lands in subcore 0 alone and that subcore runs
 * the residual for the whole chunk.
 */

#include "kernel_operator.h"

#include "../../kernels/types.h"
#include "turboquant_codec_950.h"
#include "turboquant_rotate_q.h"

namespace {

using vllm_ascend::turboquant::kFp32PerBlock;
using vllm_ascend::turboquant::kRotateQTile;
using vllm_ascend::turboquant::RotateQVariant;

using TurboQuantCodec4 = vllm_ascend::turboquant::TurboQuantCodec<4>;

// Elements of a half Cube operand in one C0 block: 32 bytes over 2. The fp8
// decode's 32 is the same rule at a different element size. Both operands here
// are 16 columns wide, so c / C0 is always 0 and the NZ address collapses to
// r * 16 + c -- the staging is a contiguous copy with no reorder to pay for.
constexpr uint32_t kHalfC0 = 16;

// The first butterfly stride the Cube does NOT absorb. Strides 1, 2, 4 and 8 are
// the right multiply by H_16.
constexpr uint32_t kResidualFirstStride = kRotateQTile;

// A vector instruction repeats at most 255 times, so a butterfly whose group
// count exceeds that is issued in several calls.
constexpr uint32_t kMaxRepeat = 255;

/*
 * Cross-core flag ids. Two directions, two reverse edges, one id per slot.
 *
 * The reverse edges are what a double-buffered loop needs and a lockstep one
 * gets by accident: kFlagOperandsFree says the Cube's MTE1 has drained L1[slot]
 * into L0A/L0B and the AIV may restage it; kFlagProductFree says the residual
 * and its writeback are done with UB[slot] and the Fixpipe may land in it again.
 *
 * Ids 11..14 belong to CANN's own SyncAll (dav_3510/kernel_operator_sync_impl.h)
 * and GetffstMsg masks the id to four bits, so 0..7 is the range this kernel may
 * spend. It is a separate launch from the decode, so there is no conflict with
 * the ids in turboquant_cube_mm.h.
 */
constexpr uint16_t kFlagOperandsReady = 0;  // 0, 1    AIV -> AIC
constexpr uint16_t kFlagOperandsFree = 2;   // 2, 3    AIC -> AIV
constexpr uint16_t kFlagProductReady = 4;   // 4, 5    AIC -> AIV
constexpr uint16_t kFlagProductFree = 6;    // 6, 7    AIV -> AIC

// Depth of the L1/UB ping-pong, and the modulus every slot index is taken at.
constexpr uint32_t kSlots = 2;

// L0C -> UB row major. No predefined FixpipeConfig sets isToUB, which exists
// only on arch35; the decode's kFixpipeToUb is the same constant.
constexpr AscendC::FixpipeConfig kFixpipeToUb = {AscendC::CO2Layout::ROW_MAJOR, true};

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
__aicore__ inline uint32_t AlignUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }

__aicore__ inline uint16_t CeilDivU16(uint32_t a, uint32_t b)
{
    return static_cast<uint16_t>((a + b - 1) / b);
}

// Use the MIX block index consistently across the AIC/AIV pair. GetBlockIdx()
// on a vector core counts subcores, so the two halves of one MIX block would
// disagree about which vectors they own -- and every flag below is a predicate
// on the chunk index within that share.
__aicore__ inline uint32_t MixBlockIdx()
{
    return static_cast<uint32_t>(AscendC::GetBlockIdx() / AscendC::GetSubBlockNum());
}

/*
 * Vector -> MTE3, before any DMA of a UB buffer the vector unit just wrote.
 *
 * PipeBarrier<PIPE_V> orders V against V only. Without this event the DMA stages
 * whatever the buffer held before, the Cube multiplies an empty operand and the
 * Fixpipe writes an all-zero product. Nothing faults. See TURBOQUANT_TESTS.md
 * 13.14.4.
 */
__aicore__ inline void SyncVectorToMte3()
{
    const event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev);
}

// The same rule for the other hand-offs these kernels make.
template <AscendC::HardEvent EVENT>
__aicore__ inline void SyncEvent()
{
    const event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVENT));
    AscendC::SetFlag<EVENT>(ev);
    AscendC::WaitFlag<EVENT>(ev);
}

/*
 * One butterfly stage of stride `stride` over `count` contiguous fp32 lanes.
 *
 * dst[i] = src[i] + src[i + stride] and dst[i + stride] = src[i] - src[i + stride]
 * for every group of 2 * stride lanes. The two writes are to disjoint halves of
 * the group, which is what makes the pair a single Add and a single Sub rather
 * than a masked read-modify-write.
 *
 * Every stride used here divides the vector length and the vectors are laid out
 * contiguously, so the pattern is periodic across a whole chunk and one call
 * covers all of it.
 */
__aicore__ inline void BlockButterfly(const AscendC::LocalTensor<float> &dst,
                                      const AscendC::LocalTensor<float> &src, uint32_t stride, uint32_t count)
{
    const uint32_t groups = count / (2 * stride);
    constexpr uint32_t kFp32PerRepeat = 64;
    if (stride > kFp32PerRepeat) {
        // Wider than one repeat: the block-stride form cannot express it, so
        // each group is its own contiguous pair.
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t base = g * 2 * stride;
            AscendC::Add(dst[base], src[base], src[base + stride], stride);
            AscendC::Sub(dst[base + stride], src[base], src[base + stride], stride);
        }
        return;
    }
    const uint8_t rep = static_cast<uint8_t>(2 * stride / kFp32PerBlock);
    const AscendC::BinaryRepeatParams params{1, 1, 1, rep, rep, rep};
    const uint64_t mask = static_cast<uint64_t>(stride);
    for (uint32_t done = 0; done < groups; done += kMaxRepeat) {
        const uint32_t batch = (groups - done) < kMaxRepeat ? (groups - done) : kMaxRepeat;
        const uint32_t base = done * 2 * stride;
        AscendC::Add(dst[base], src[base], src[base + stride], mask, static_cast<uint8_t>(batch), params);
        AscendC::Sub(dst[base + stride], src[base], src[base + stride], mask, static_cast<uint8_t>(batch), params);
    }
}

/*
 * The Cube path: strides 1, 2, 4, 8 on the Cube, the rest on the AIV.
 *
 * One MIX block owns `vectorsPerBlock` consecutive vectors and walks them in
 * chunks of `vectorsPerChunk`. The host guarantees that the chunk divides the
 * block's share and that the share divides numVectors, so every chunk is full
 * and there is no short final Mmad.
 */
template <typename scalar_t>
class TurboQuantRotateQCube {
public:
    __aicore__ inline explicit TurboQuantRotateQCube(AscendC::TPipe *pipe) : pipe_(pipe) {}

    /*
     * headSize         a power of two, at least 64, so R = headSize / 16 is at
     *                  least 4 and the M split of the dual-destination Fixpipe
     *                  lands on a vector boundary for any even chunk
     * vectorsPerBlock  divides numVectors
     * vectorsPerChunk  divides vectorsPerBlock
     * invSqrtLen       1 / sqrt(headSize), so the device needs no square root
     */
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR piSigns, GM_ADDR h16, GM_ADDR queryRot, uint32_t numVectors,
                                uint32_t headSize, uint32_t vectorsPerBlock, uint32_t vectorsPerChunk,
                                uint32_t variant, float invSqrtLen)
    {
        headSize_ = headSize;
        numVectors_ = numVectors;
        vectorsPerChunk_ = vectorsPerChunk;
        variant_ = variant;
        invSqrtLen_ = invSqrtLen;

        blockBase_ = MixBlockIdx() * vectorsPerBlock;
        uint32_t mine = 0;
        if (blockBase_ < numVectors_) {
            mine = numVectors_ - blockBase_;
            if (mine > vectorsPerBlock) {
                mine = vectorsPerBlock;
            }
        }
        blockVectors_ = mine;

        rowsPerVector_ = headSize_ / kRotateQTile;
        chunkElems_ = vectorsPerChunk_ * headSize_;
        chunkRows_ = vectorsPerChunk_ * rowsPerVector_;
        numChunks_ = vectorsPerChunk_ == 0 ? 0 : blockVectors_ / vectorsPerChunk_;

        // A LoadData of mStep fractals reads whole 16-row fractals out of L1
        // whatever m is, so every Cube buffer is sized on the padded row count
        // and the pad is zeroed before staging. At every head size this operator
        // accepts the chunk already fills whole fractals; the alignment is kept
        // so a future head size or chunk rule cannot silently read past the end.
        paddedRows_ = AlignUp(chunkRows_, kRotateQTile);
        paddedElems_ = paddedRows_ * kRotateQTile;

        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(query));
        piSignsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(piSigns), headSize_);
        h16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(h16));
        queryRotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(queryRot));

        // L1 on both halves of the MIX kernel: the AIV stages into these and the
        // AIC reads them. A1/B1 are the only Cube buffers a vector core may
        // hold -- InitBuffer on A2/B2/CO1 hands it a base it cannot reach and the
        // first touch is `pem_lsu: unrecognize ldst addr`.
        //
        // A1 is the operand ping-pong; B1 holds H_16, which is the same 512
        // bytes for every chunk and stays single-buffered.
        for (uint32_t slot = 0; slot < kSlots; ++slot) {
            pipe_->InitBuffer(aHi1_[slot], paddedElems_ * sizeof(half));
            pipe_->InitBuffer(aLo1_[slot], paddedElems_ * sizeof(half));
        }
        pipe_->InitBuffer(b1_, kRotateQTile * kRotateQTile * sizeof(half));

        // L0A/L0B/L0C stay single-buffered: the Cube stage is one or two Mmads
        // against a 16 x 16 constant and is already far shorter than the vector
        // work it overlaps. What that costs instead is two explicit events per
        // chunk -- M_MTE1 before the load and FIX_M before the Mmad.
        if ASCEND_IS_AIC {
            pipe_->InitBuffer(a2Hi_, paddedElems_ * sizeof(half));
            pipe_->InitBuffer(a2Lo_, paddedElems_ * sizeof(half));
            pipe_->InitBuffer(b2_, kRotateQTile * kRotateQTile * sizeof(half));
            pipe_->InitBuffer(co1_, paddedElems_ * sizeof(float));
        }

        // UB, per subcore. prodBuf_ is the Fixpipe destination, sized for the
        // whole chunk even under dual destination where only half of it is
        // written -- the spare half costs nothing and keeps one sizing rule.
        //
        // prodBuf_ and tmpBuf_ are both per slot, and tmpBuf_ has to be: the AIV
        // stages chunk k BEFORE running the residual of chunk k - 1, and the
        // staging's hi/lo split uses tmpBuf_ as its fp32 scratch. Sharing one
        // would have the stage of k overwrite the buffer the residual of k - 1
        // is about to ping-pong through.
        //
        // tmpBuf_ is sized on paddedElems_, not chunkElems_: the hi/lo residual
        // Cast covers the fractal pad as well.
        pipe_->InitBuffer(qInBuf_, chunkElems_ * sizeof(scalar_t));
        pipe_->InitBuffer(inBuf_, paddedElems_ * sizeof(float));
        pipe_->InitBuffer(castBuf_, paddedElems_ * sizeof(half));
        pipe_->InitBuffer(signBuf_, headSize_ * sizeof(float));
        for (uint32_t slot = 0; slot < kSlots; ++slot) {
            pipe_->InitBuffer(prodBuf_[slot], chunkElems_ * sizeof(float));
            pipe_->InitBuffer(tmpBuf_[slot], paddedElems_ * sizeof(float));
        }

        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::DataCopy(signs, piSignsGm_, headSize_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        if (numChunks_ == 0) {
            // A block past the end of the grid, which the host's exact-division
            // planning does not produce -- but a kernel that walks a task list
            // has to be safe when handed none, and both halves must agree to do
            // nothing or the one that ran would wait on a flag nobody sets.
            return;
        }
        // Not one interleaved loop with guards: the two halves no longer visit
        // the same chunk at the same time, and writing them as one body would
        // only hide that.
        if ASCEND_IS_AIV {
            ProcessAiv();
        }
        if ASCEND_IS_AIC {
            ProcessAic();
        }
    }

private:
    /*
     * The vector half of the macro-pipeline.
     *
     * Per iteration it stages chunk k and then runs the residual of chunk k - 1,
     * in that order. Both halves of that overlap Cube work: the staging runs
     * against the Mmad and Fixpipe of chunk k - 1, and the residual against
     * those of chunk k, whose operands the staging just signalled.
     *
     * Every flag set is conditioned so that it has exactly one wait. A surplus
     * set is not harmless: these are hardware counters the kernel does not clear
     * on exit, and a leftover count lets a later launch's wait fall straight
     * through.
     */
    __aicore__ inline void ProcessAiv()
    {
        StageOperands(0, 0);
        SignalOperandsReady(0);

        for (uint32_t chunk = 1; chunk < numChunks_; ++chunk) {
            const uint32_t slot = chunk % kSlots;
            const uint32_t prev = (chunk - 1) % kSlots;

            if (chunk >= kSlots) {
                // L1[slot] last held chunk - kSlots; the Cube has to have read
                // it into L0A/L0B before it can be restaged.
                AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagOperandsFree + slot));
            }
            // BEFORE the previous chunk's residual, and that ordering is the
            // whole pipeline: the Cube is running chunk - 1 while this copy
            // happens and can start chunk the moment the flag below lands.
            StageOperands(chunk, slot);
            SignalOperandsReady(slot);

            AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagProductReady + prev));
            Residual(prev, chunk - 1);
            if (chunk + 1 < numChunks_) {
                // Matched by the AIC's wait at chunk + 1, which reuses UB[prev].
                SignalProductFree(prev);
            }
        }

        const uint32_t last = (numChunks_ - 1) % kSlots;
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagProductReady + last));
        Residual(last, numChunks_ - 1);
    }

    /*
     * The Cube half. One chunk behind the AIV in steady state.
     *
     * L1[slot] is released as soon as MTE1 has copied it into L0A/L0B, long
     * before the Fixpipe -- that early release is what lets the AIV stage
     * chunk + kSlots against this chunk's Mmad rather than after its Fixpipe.
     */
    __aicore__ inline void ProcessAic()
    {
        for (uint32_t chunk = 0; chunk < numChunks_; ++chunk) {
            const uint32_t slot = chunk % kSlots;

            AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagOperandsReady + slot));
            LoadCubeOperands(slot);
            if (chunk + kSlots < numChunks_) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE1>(static_cast<uint16_t>(kFlagOperandsFree + slot));
            }
            if (chunk >= kSlots) {
                // UB[slot] still holds chunk - kSlots until the AIV's residual
                // and writeback have drained it.
                AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagProductFree + slot));
            }
            MmadAndFixpipe(slot);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(static_cast<uint16_t>(kFlagProductReady + slot));
        }
    }

    /*
     * Both subcores set every AIV -> AIC flag, and that is not an oversight: the
     * arch35 flag in mode 0x02 is satisfied only once all subcores of the pair
     * have set it, so a set from one of them hangs the Cube.
     *
     * It is also what makes kFlagProductFree correct under single-destination
     * Fixpipe, where subcore 1 owns no vectors, returns from Residual at once
     * and signals immediately: the AIC is still held until subcore 0 -- the one
     * that actually holds the product -- has signalled too.
     */
    __aicore__ inline void SignalOperandsReady(uint32_t slot)
    {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(static_cast<uint16_t>(kFlagOperandsReady + slot));
    }

    __aicore__ inline void SignalProductFree(uint32_t slot)
    {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(static_cast<uint16_t>(kFlagProductFree + slot));
    }

    // The dual-destination Fixpipe splits the product at row chunkRows_ / 2,
    // which is a vector boundary only for an even chunk. At an odd one the bit
    // is ignored rather than honoured wrongly.
    __aicore__ inline bool DualDst() const
    {
        return (variant_ & RotateQVariant::kDualDst) != 0 && (vectorsPerChunk_ % 2 == 0);
    }

    // Vectors of the current chunk this subcore owns, and where they start
    // within it. Under dual destination each subcore holds its half at offset 0
    // of its own UB; otherwise the whole chunk is in subcore 0's UB.
    __aicore__ inline uint32_t MyVectors() const
    {
        if (DualDst()) {
            return vectorsPerChunk_ / 2;
        }
        return AscendC::GetSubBlockIdx() == 0 ? vectorsPerChunk_ : 0;
    }

    __aicore__ inline uint32_t MyChunkOffset() const
    {
        if (DualDst()) {
            return static_cast<uint32_t>(AscendC::GetSubBlockIdx()) * (vectorsPerChunk_ / 2);
        }
        return 0;
    }

    /*
     * scalar_t GM -> fp32 UB -> D x -> fp16 UB -> L1, in NZ order, which for a
     * 16-column half image is plain row major.
     *
     * Both subcores stage the whole chunk rather than half each. The AIC's
     * single CrossCoreWaitFlag is released by the pair of sets, so a split would
     * be safe only if that quorum reading is exact; duplicating a copy this
     * small is the cheaper way not to depend on it.
     */
    __aicore__ inline void StageOperands(uint32_t chunk, uint32_t slot)
    {
        AscendC::LocalTensor<scalar_t> qIn = qInBuf_.Get<scalar_t>();
        AscendC::LocalTensor<float> in = inBuf_.Get<float>();
        AscendC::LocalTensor<half> cast = castBuf_.Get<half>();
        AscendC::LocalTensor<float> tmp = tmpBuf_[slot].Get<float>();
        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();

        // The previous chunk's Cast and Sub read inBuf_ on PIPE_V and the load
        // below refills qInBuf_ on PIPE_MTE2. Nothing else orders that pair once
        // the loop is pipelined.
        SyncEvent<AscendC::HardEvent::V_MTE2>();

        // H_16 is the same for every chunk, so it is staged once per block.
        if (!h16Staged_) {
            AscendC::DataCopy(cast, h16Gm_, kRotateQTile * kRotateQTile);
            SyncEvent<AscendC::HardEvent::MTE2_MTE3>();
            AscendC::DataCopy(b1_.Get<half>(), cast, kRotateQTile * kRotateQTile);
            h16Staged_ = true;
        }

        const uint64_t base = static_cast<uint64_t>(blockBase_ + chunk * vectorsPerChunk_) * headSize_;
        AscendC::DataCopy(qIn, queryGm_[base], chunkElems_);
        SyncEvent<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(in, qIn, AscendC::RoundMode::CAST_NONE, chunkElems_);
        AscendC::PipeBarrier<PIPE_V>();

        if (paddedElems_ > chunkElems_) {
            // The rows the fractal load reads past the chunk. Left undefined
            // they would reach the Mmad as fp16 garbage and could raise inf/nan
            // on rows the Fixpipe never copies out.
            AscendC::Duplicate(in[chunkElems_], 0.0f, paddedElems_ - chunkElems_);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // The first D of Pi = D H D. Per vector, not over the whole chunk: the
        // sign vector is headSize_ long, a batched Mul would read past it, and
        // no BinaryRepeatParams can cycle a src1 whose period spans several
        // repeats. vectorsPerChunk_ instructions against log2(D) butterfly
        // stages is not a trade worth a second table format.
        for (uint32_t v = 0; v < vectorsPerChunk_; ++v) {
            AscendC::Mul(in[v * headSize_], in[v * headSize_], signs, headSize_);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // hi = half(x). castBuf_ was last read by an MTE3 copy -- into B1 on the
        // first chunk, into A1 on every later one -- and MTE2_V orders MTE2
        // against V and says nothing about MTE3, so overwriting it needs its own
        // event or that image is still in flight when the cast lands on it.
        SyncEvent<AscendC::HardEvent::MTE3_V>();
        AscendC::Cast(cast, in, AscendC::RoundMode::CAST_RINT, paddedElems_);
        SyncVectorToMte3();
        AscendC::DataCopy(aHi1_[slot].Get<half>(), cast, paddedElems_);

        if ((variant_ & RotateQVariant::kHiLo) != 0) {
            // lo = half(x - float(hi)). castBuf_ is still being read by the DMA
            // above, so the residual is formed in fp32 first and the cast buffer
            // is only reused once MTE3 has released it.
            AscendC::Cast(tmp, cast, AscendC::RoundMode::CAST_NONE, paddedElems_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(in, in, tmp, paddedElems_);
            AscendC::PipeBarrier<PIPE_V>();
            SyncEvent<AscendC::HardEvent::MTE3_V>();
            AscendC::Cast(cast, in, AscendC::RoundMode::CAST_RINT, paddedElems_);
            SyncVectorToMte3();
            AscendC::DataCopy(aLo1_[slot].Get<half>(), cast, paddedElems_);
        }
    }

    /*
     * A is [paddedRows_, 16] in L1 and L0A wants the same: a straight load.
     *
     * The fractal contract is TurboQuantCubeMm::LoadA's, which CANN's
     * load_to_l0a_load2dV2.h pins: mStep counts 16-row fractals, kStep counts C0
     * elements, and srcStride / dstStride are the leading extents over 16 and
     * are not optional -- zero for either silently reads the wrong fractals.
     */
    __aicore__ inline void LoadA(const AscendC::LocalTensor<half> &src, const AscendC::LocalTensor<half> &dst)
    {
        AscendC::LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = CeilDivU16(paddedRows_, kRotateQTile);
        p.kStep = CeilDivU16(kRotateQTile, kHalfC0);
        p.srcStride = CeilDivU16(paddedRows_, kRotateQTile);
        p.dstStride = CeilDivU16(paddedRows_, kRotateQTile);
        p.ifTranspose = false;
        AscendC::LoadData(dst, src, p);
    }

    /*
     * B is H_16 staged [n, k], which is the form that loads with one LoadData
     * and ifTranspose FALSE -- the flag reads as the opposite of what it does.
     * H_16 is symmetric, so the [n, k] image and the [k, n] image are the same
     * bytes and the product is X . H_16^T either way.
     */
    __aicore__ inline void LoadB()
    {
        AscendC::LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = CeilDivU16(kRotateQTile, kRotateQTile);
        p.kStep = CeilDivU16(kRotateQTile, kHalfC0);
        p.srcStride = CeilDivU16(kRotateQTile, kRotateQTile);
        p.dstStride = CeilDivU16(kRotateQTile, kRotateQTile);
        p.ifTranspose = false;
        AscendC::LoadData(b2_.Get<half>(), b1_.Get<half>(), p);
    }

    /*
     * L1[slot] -> L0A/L0B. The first half of the Cube stage, split out from the
     * Mmad so the caller can release L1[slot] to the AIV between the two.
     *
     * L0A/L0B are single-buffered, so MTE1 may not refill them until the
     * PREVIOUS chunk's Mmad has read them -- hence the explicit M_MTE1. MTE1_M
     * is equally mandatory: PipeBarrier orders a pipe against itself only, and
     * without it the Mmad reads an L0A/L0B that LoadData is still filling. The
     * symptom of getting either wrong is a partly-zero product, not a fault.
     */
    __aicore__ inline void LoadCubeOperands(uint32_t slot)
    {
        SyncEvent<AscendC::HardEvent::M_MTE1>();
        LoadA(aHi1_[slot].Get<half>(), a2Hi_.Get<half>());
        if ((variant_ & RotateQVariant::kHiLo) != 0) {
            LoadA(aLo1_[slot].Get<half>(), a2Lo_.Get<half>());
        }
        LoadB();
        SyncEvent<AscendC::HardEvent::MTE1_M>();
    }

    // The chunk's lower four butterfly stages, as one Mmad -- or two when the
    // hi/lo split is on, accumulating into the same L0C -- and the Fixpipe that
    // hands the result to UB[slot].
    __aicore__ inline void MmadAndFixpipe(uint32_t slot)
    {
        // L0C is single-buffered too, so the previous chunk's Fixpipe has to
        // have drained it before this Mmad overwrites it.
        SyncEvent<AscendC::HardEvent::FIX_M>();

        AscendC::LocalTensor<float> acc = co1_.Get<float>();
        AscendC::Mmad(acc, a2Hi_.Get<half>(), b2_.Get<half>(),
                      AscendC::MmadParams(static_cast<uint16_t>(chunkRows_), static_cast<uint16_t>(kRotateQTile),
                                          static_cast<uint16_t>(kRotateQTile), 0, false, true));
        if ((variant_ & RotateQVariant::kHiLo) != 0) {
            // cmatrixInitVal false: accumulate onto the hi product already in
            // L0C. Back-to-back Mmads on one accumulator are what CANN's own
            // k-loop issues, and they need no barrier between them.
            AscendC::Mmad(acc, a2Lo_.Get<half>(), b2_.Get<half>(),
                          AscendC::MmadParams(static_cast<uint16_t>(chunkRows_), static_cast<uint16_t>(kRotateQTile),
                                              static_cast<uint16_t>(kRotateQTile), 0, false, false));
        }

        SyncEvent<AscendC::HardEvent::M_FIX>();

        // srcStride is m rounded UP to a multiple of 16 -- a different
        // convention from LoadData's m / 16, two lines away. dstStride is the UB
        // row pitch in elements.
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fp(
            static_cast<uint16_t>(kRotateQTile), static_cast<uint16_t>(chunkRows_),
            static_cast<uint16_t>(paddedRows_), kRotateQTile);
        if (DualDst()) {
            // 0b01: split in M, chunkRows_ / 2 x 16 into EACH subcore's UB. The
            // assert in FixpipeL0cToUB wants an even M; DualDst() has already
            // established the split lands on a vector boundary, which is the
            // stronger condition.
            fp.dualDstCtl = 0b01;
            fp.subBlockId = false;
        }
        AscendC::Fixpipe<float, float, kFixpipeToUb>(prodBuf_[slot].Get<float>(), acc, fp);
        AscendC::PipeBarrier<PIPE_FIX>();
    }

    /*
     * The stages above stride 8: in the R x 16 tile they are butterflies over
     * whole rows. log2(R) Add/Sub pairs for this subcore's share of the chunk,
     * then the 1 / sqrt(D) the transform is normalised by, then the second D of
     * Pi = D H D, then the writeback.
     *
     * An odd number of residual stages leaves the result in the scratch buffer,
     * so which of the two holds it is tracked rather than assumed -- D = 128 and
     * D = 512 end in tmp, D = 64 and D = 256 in prod.
     */
    __aicore__ inline void Residual(uint32_t slot, uint32_t chunk)
    {
        const uint32_t mine = MyVectors();
        if (mine == 0) {
            return;
        }
        const uint32_t count = mine * headSize_;
        AscendC::LocalTensor<float> src = prodBuf_[slot].Get<float>();
        AscendC::LocalTensor<float> dst = tmpBuf_[slot].Get<float>();
        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();

        // Under dual destination each subcore's half starts at offset 0 of its
        // own prodBuf_, so the residual always runs from the base.
        for (uint32_t stride = kResidualFirstStride; stride < headSize_; stride <<= 1) {
            BlockButterfly(dst, src, stride, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<float> hold = src;
            src = dst;
            dst = hold;
        }

        AscendC::Muls(src, src, invSqrtLen_, count);
        AscendC::PipeBarrier<PIPE_V>();

        // The second D of Pi = D H D, per vector for the same reason as the
        // first.
        for (uint32_t v = 0; v < mine; ++v) {
            AscendC::Mul(src[v * headSize_], src[v * headSize_], signs, headSize_);
        }

        SyncVectorToMte3();
        const uint64_t out =
            (static_cast<uint64_t>(blockBase_ + chunk * vectorsPerChunk_) + MyChunkOffset()) * headSize_;
        AscendC::DataCopy(queryRotGm_[out], src, count);
        // No MTE3 -> V here, and the two buffers this DMA may still be reading
        // are covered elsewhere. prodBuf_[slot] is next written by the Fixpipe,
        // which the AIC does not issue until kFlagProductFree -- set on
        // PIPE_MTE3 after this copy -- has arrived. tmpBuf_[slot] is next
        // written by the staging of chunk + kSlots, which begins with its own
        // V -> MTE2 and MTE3 -> V edges. Draining the writeback here would stall
        // the AIV at precisely the point the pipeline exists to keep busy.
    }

    AscendC::TPipe *pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> aHi1_[kSlots];
    AscendC::TBuf<AscendC::TPosition::A1> aLo1_[kSlots];
    AscendC::TBuf<AscendC::TPosition::B1> b1_;
    AscendC::TBuf<AscendC::TPosition::A2> a2Hi_;
    AscendC::TBuf<AscendC::TPosition::A2> a2Lo_;
    AscendC::TBuf<AscendC::TPosition::B2> b2_;
    AscendC::TBuf<AscendC::TPosition::CO1> co1_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> qInBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> inBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> castBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> signBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> prodBuf_[kSlots];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf_[kSlots];
    AscendC::GlobalTensor<scalar_t> queryGm_;
    AscendC::GlobalTensor<float> piSignsGm_;
    AscendC::GlobalTensor<half> h16Gm_;
    AscendC::GlobalTensor<float> queryRotGm_;
    uint32_t headSize_ = 0;
    uint32_t numVectors_ = 0;
    uint32_t vectorsPerChunk_ = 0;
    uint32_t variant_ = 0;
    uint32_t blockBase_ = 0;
    uint32_t blockVectors_ = 0;
    uint32_t rowsPerVector_ = 0;
    uint32_t chunkElems_ = 0;
    uint32_t chunkRows_ = 0;
    uint32_t numChunks_ = 0;
    uint32_t paddedRows_ = 0;
    uint32_t paddedElems_ = 0;
    float invSqrtLen_ = 1.0f;
    bool h16Staged_ = false;
};

/*
 * The vector-only path, and the numerical reference for the other one.
 *
 * All log2(D) stages through TurboQuantCodec4::ApplyPi -- the same routine, in
 * the same order, on the same fp32 data that TurboQuantPagedAttentionSplit and
 * TurboQuantCubeDecodeSplit used to call per query head. Its output is therefore
 * bit-identical to the pre-refactor decode's internal rotation, which is the
 * property the sim and device tests gate on.
 *
 * The two subcores take alternate vectors, which is the only free parallelism
 * this form has.
 */
template <typename scalar_t>
class TurboQuantRotateQAiv {
public:
    __aicore__ inline explicit TurboQuantRotateQAiv(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR piSigns, GM_ADDR rotTables, GM_ADDR queryRot,
                                uint32_t numVectors, uint32_t headSize, uint32_t vectorsPerBlock, float invSqrtLen)
    {
        headSize_ = headSize;
        numVectors_ = numVectors;
        vectorsPerBlock_ = vectorsPerBlock;

        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(query));
        piSignsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(piSigns), headSize_);
        rotTablesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rotTables));
        queryRotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(queryRot));

        pipe_->InitBuffer(qInBuf_, headSize_ * sizeof(scalar_t));
        pipe_->InitBuffer(workBuf_, 2 * headSize_ * sizeof(float));
        pipe_->InitBuffer(signBuf_, headSize_ * sizeof(float));

        // batchRows 1: this path rotates one vector per call, exactly as the
        // decode did. The dequantiser's batchRows term in ConstTableWords is
        // what sizes the tables, and the host builds them at 1 for the same
        // reason the decode's rotation_ does.
        codec_.Init(pipe_, headSize_, 1, invSqrtLen, rotTablesGm_);

        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();
        AscendC::DataCopy(signs, piSignsGm_, headSize_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        // Both halves of a MIX block enter this function and every statement in
        // it is vector work, so the whole body is guarded. Unguarded, the cube
        // core issues UB reads and a GM writeback it does not own -- which shows
        // up as an arithmetic error with a silent exception dump, not as a fault.
        if ASCEND_IS_AIV {
            const uint32_t base = MixBlockIdx() * vectorsPerBlock_;
            if (base >= numVectors_) {
                return;
            }
            uint32_t end = base + vectorsPerBlock_;
            if (end > numVectors_) {
                end = numVectors_;
            }
            const uint32_t sub = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
            const uint32_t step = static_cast<uint32_t>(AscendC::GetSubBlockNum());
            for (uint32_t v = base + sub; v < end; v += step) {
                Rotate(v);
            }
        }
    }

private:
    __aicore__ inline void Rotate(uint32_t vector)
    {
        AscendC::LocalTensor<scalar_t> qIn = qInBuf_.Get<scalar_t>();
        AscendC::LocalTensor<float> work = workBuf_.Get<float>();
        AscendC::LocalTensor<float> x = work;
        AscendC::LocalTensor<float> tmp = work[headSize_];
        AscendC::LocalTensor<float> signs = signBuf_.Get<float>();

        AscendC::DataCopy(qIn, queryGm_[static_cast<uint64_t>(vector) * headSize_], headSize_);
        SyncEvent<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(x, qIn, AscendC::RoundMode::CAST_NONE, headSize_);
        AscendC::PipeBarrier<PIPE_V>();

        codec_.ApplyPi(x, tmp, signs, static_cast<int>(headSize_));

        SyncVectorToMte3();
        AscendC::DataCopy(queryRotGm_[static_cast<uint64_t>(vector) * headSize_], x, headSize_);
        // MTE3 -> MTE2 AND MTE3 -> V: the next vector's first touch of qInBuf_
        // is the input DataCopy, which is MTE2, and ApplyPi's Gather and Mul are
        // V. Ordering the writeback against only one of the two leaves the other
        // racing it.
        SyncEvent<AscendC::HardEvent::MTE3_MTE2>();
        SyncEvent<AscendC::HardEvent::MTE3_V>();
    }

    AscendC::TPipe *pipe_;
    TurboQuantCodec4 codec_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> qInBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> workBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> signBuf_;
    AscendC::GlobalTensor<scalar_t> queryGm_;
    AscendC::GlobalTensor<float> piSignsGm_;
    AscendC::GlobalTensor<int32_t> rotTablesGm_;
    AscendC::GlobalTensor<float> queryRotGm_;
    uint32_t headSize_ = 0;
    uint32_t numVectors_ = 0;
    uint32_t vectorsPerBlock_ = 0;
};

}  // namespace

/*
 * Entry points take GM_ADDR, not `__gm__ void *`: the generated launcher does
 * not compile otherwise. Same note as in turboquant_kernels.cpp.
 */
#define TURBOQUANT_ROTATE_Q_CUBE_DECLARE(TYPE)                                                                       \
    extern "C" __global__ __aicore__ void turboquant_rotate_q_cube_##TYPE(                                            \
        GM_ADDR query, GM_ADDR piSigns, GM_ADDR h16, GM_ADDR queryRot, uint32_t numVectors, uint32_t headSize,       \
        uint32_t vectorsPerBlock, uint32_t vectorsPerChunk, uint32_t variant, float invSqrtLen)                       \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantRotateQCube<TYPE> op(&pipe);                                                                       \
        op.Init(query, piSigns, h16, queryRot, numVectors, headSize, vectorsPerBlock, vectorsPerChunk, variant,      \
                invSqrtLen);                                                                                         \
        op.Process();                                                                                                \
    }

#define TURBOQUANT_ROTATE_Q_AIV_DECLARE(TYPE)                                                                        \
    extern "C" __global__ __aicore__ void turboquant_rotate_q_aiv_##TYPE(                                             \
        GM_ADDR query, GM_ADDR piSigns, GM_ADDR rotTables, GM_ADDR queryRot, uint32_t numVectors,                    \
        uint32_t headSize, uint32_t vectorsPerBlock, float invSqrtLen)                                                \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        TurboQuantRotateQAiv<TYPE> op(&pipe);                                                                        \
        op.Init(query, piSigns, rotTables, queryRot, numVectors, headSize, vectorsPerBlock, invSqrtLen);             \
        op.Process();                                                                                                \
    }

TURBOQUANT_ROTATE_Q_CUBE_DECLARE(half)
TURBOQUANT_ROTATE_Q_AIV_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
TURBOQUANT_ROTATE_Q_CUBE_DECLARE(bfloat16_t)
TURBOQUANT_ROTATE_Q_AIV_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {

/*
 * One launch, one path. The host has already decided which -- see PlanRotateQ --
 * and passes `useCube` rather than re-deriving the gate here, so the plan the
 * operator validates and the plan the kernel runs cannot drift apart.
 */
void turboquant_rotate_q_impl(AscendType type, void *stream, uint32_t blockDim, bool useCube, void *query,
                              void *piSigns, void *h16, void *rotTables, void *queryRot, uint32_t numVectors,
                              uint32_t headSize, uint32_t vectorsPerBlock, uint32_t vectorsPerChunk,
                              uint32_t variant, float invSqrtLen)
{
    if (blockDim == 0 || numVectors == 0) {
        return;
    }
    if (type == AscendType::FP16) {
        if (useCube) {
            turboquant_rotate_q_cube_half<<<blockDim, nullptr, stream>>>(
                query, piSigns, h16, queryRot, numVectors, headSize, vectorsPerBlock, vectorsPerChunk, variant,
                invSqrtLen);
        } else {
            turboquant_rotate_q_aiv_half<<<blockDim, nullptr, stream>>>(
                query, piSigns, rotTables, queryRot, numVectors, headSize, vectorsPerBlock, invSqrtLen);
        }
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    } else if (type == AscendType::BF16) {
        if (useCube) {
            turboquant_rotate_q_cube_bfloat16_t<<<blockDim, nullptr, stream>>>(
                query, piSigns, h16, queryRot, numVectors, headSize, vectorsPerBlock, vectorsPerChunk, variant,
                invSqrtLen);
        } else {
            turboquant_rotate_q_aiv_bfloat16_t<<<blockDim, nullptr, stream>>>(
                query, piSigns, rotTables, queryRot, numVectors, headSize, vectorsPerBlock, invSqrtLen);
        }
#endif
    }
}

}  // namespace vllm_ascend
