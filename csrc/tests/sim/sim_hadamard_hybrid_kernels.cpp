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
 * EXPLORATORY SPIKE - test-owned device code, not part of the decode.
 *
 * Nothing in csrc/attention/turboquant/ includes this file and nothing here is
 * reachable from the wheel: the source is listed only by csrc/tests's kernel
 * library, and the only callers are
 * csrc/tests/sim/test_sim_950pr_cube_hadamard.cpp,
 * csrc/tests/device/test_device_950pr_cube_hadamard.cpp and
 * csrc/tests/device/bench_950pr_cube_hadamard.cpp. Do not promote any of it
 * into the decode path without re-measuring.
 *
 * WHAT IS BEING PROTOTYPED
 *
 * TurboQuant's rotation is Pi x = D (H (D x)) with H the normalised
 * Walsh-Hadamard transform of the head dimension. The AIV evaluates all
 * log2(D) butterfly stages: three Gather shuffles (strides 1, 2, 4, which
 * straddle the 32 B block) and the rest as block-strided Add/Sub pairs. See
 * TurboQuantCodec::FastWalshHadamardTransform in
 * csrc/attention/turboquant/turboquant_codec_950.h.
 *
 * Sylvester's construction factorises that transform. For D = R * 16, write the
 * flat index as n = 16 a + b and reshape x into an R x 16 tile
 * X[a][b] = x[16 a + b]. Then
 *
 *     H_D = H_R (x) H_16     and     Y = H_R . X . H_16^T
 *
 * with H_16 symmetric, so H_16^T = H_16. The four stages of strides 1, 2, 4, 8
 * move only the low nibble b: they are the right multiply X . H_16^T, and they
 * are the SAME four stages against the SAME 16 x 16 constant at every D. The
 * remaining log2(R) stages, strides 16 up to D/2, move only a: they are a
 * butterfly over whole 16-element rows - block aligned, no shuffle.
 *
 * So four stages of the transform collapse into one Mmad against a static
 * 16 x 16 constant no matter how long the vector is, and everything above stays
 * on the vector unit. At D = 64 that is 4 stages of 6 on the Cube; at D = 512
 * it is 4 of 9, which is the reason to expect the win to shrink with D.
 *
 * THREE THINGS THAT FELL OUT OF THE LAYOUT, ALL LOAD-BEARING
 *
 *  1. NZ IS ROW MAJOR HERE. The Cube consumes L1 operands in NZ order,
 *     nz(r, c) = (c / C0) * rows * C0 + r * C0 + (c % C0), and C0 is
 *     32 / sizeof(T) - 16 for half (Impl::B16_C0SIZE), not the 32 the fp8
 *     decode uses. Both operands are 16 columns wide, so c / C0 is always 0
 *     and nz(r, c) collapses to r * 16 + c. The staging is a contiguous copy;
 *     there is no reorder to pay for and no Gather table to build.
 *
 *  2. A WHOLE CHUNK OF VECTORS IS ONE Mmad. Stacking V tiles vertically gives
 *     A = [V * R, 16] and the same [16, 16] B, so V vectors' worth of the lower
 *     four stages is a single m = V * R, k = 16, n = 16 instruction.
 *
 *  3. THE PRODUCT COMES BACK IN THE FLAT VECTOR ORDER. Fixpipe writes
 *     [V * R, 16] fp32 row major, and row m = R v + a at column b lands at
 *     v * D + a * 16 + b - exactly x's own layout. The residual stages then
 *     read it with no permutation at all.
 *
 * OPERAND PRECISION
 *
 * Mmad's fp16 path is <float, half, half>: half operands, fp32 accumulate.
 * H_16 is +-1 and exact, so the only error is the fp32 -> fp16 cast of the
 * input, and after the full normalised transform that lands near 1e-3 for
 * unit-scale data - an order of magnitude past the 1e-4 this spike has to
 * clear. kVariantHiLo splits the input as x = hi + lo with hi = fp16(x) and
 * lo = fp16(x - hi) and issues two Mmads accumulating into one L0C
 * (cmatrixInitVal false on the second). Two instructions instead of one,
 * against four vector stages, and the error drops to the fp32 floor: measured
 * 3.58e-4 -> 2.38e-7 at D = 256 for about 2% of the camodel ticks.
 *
 * THE TWO VECTOR SUBCORES
 *
 * A 950PR AIC has two AIV subcores with PRIVATE UB. The natural-looking split
 * of the residual butterfly - one subcore takes the A + B halves, the other the
 * A - B halves, to disjoint destinations - is a real shape for ONE stage and
 * does not chain: stage k+1 reads both halves and neither subcore can see the
 * other's UB.
 *
 * The obstacle is the copy, not the barrier. arch35 does have an intra-block
 * flag - CrossCoreSetFlag<0x4, PIPE_V>, which lowers to set_intra_block and is
 * already used between subcores in csrc/attention/chunk_kda_fwd/op_kernel/arch35
 * - so the two subcores can be made to wait for each other. What they cannot do
 * is see each other's results: every one of the log2(R) - 1 inter-stage
 * boundaries would have to move half a chunk out to L1 or GM and back, against
 * an Add or a Sub of the same size. The transfer is the larger half of that
 * trade by a wide margin, which is why this file does not implement the split.
 *
 * What does work is splitting by vector, and arch35's Fixpipe can hand the
 * halves out directly: FixpipeParamsArch3510 carries dualDstCtl, and 0b01
 * writes M/2 x N into EACH subcore's UB (kernel_operator_fixpipe_impl.h,
 * FixpipeL0cToUB). kVariantDualDst selects it, so the Cube produces the whole
 * chunk once and each subcore finishes its own half with no exchange and no GM
 * round trip. Without the bit the product lands in subcore 0 alone - the
 * behaviour IsPrimarySubcore() in turboquant_mm_kernels.cpp documents - and that
 * subcore runs the residual for every vector in the chunk.
 *
 * The M split lands at row M/2, which is a vector boundary only when the chunk
 * holds an EVEN number of vectors. At an odd chunk - which in practice means a
 * batch of one - the bit is ignored and the kernel takes the single-destination
 * path, because splitting one vector's tile between two subcores would leave
 * neither able to run a row butterfly.
 *
 * CHUNKING, AND THE TWO-STAGE MACRO-PIPELINE OVER IT
 *
 * UB is the binding constraint: the working set holds fp32 input, an fp16 cast,
 * the fp32 product and an fp32 ping-pong buffer, so 14 bytes per element. The
 * host sizes a chunk and the kernel loops over chunks. The host guarantees
 * numVectors is a multiple of vectorsPerChunk.
 *
 * A chunk used to be a self-contained stage -> Mmad -> Fixpipe -> residual ->
 * writeback, with the AIV and the AIC alternately idle: the AIC waited out both
 * of the AIV's stages and the AIV waited out the whole Cube stage. Each chunk
 * now runs against its neighbours instead, on a two-slot ping-pong keyed on
 * chunk % 2:
 *
 *   AIV   stage(k) -> signal      | residual(k-1) -> writeback -> release UB
 *   AIC                Mmad + Fixpipe(k-1)        |  Mmad + Fixpipe(k)
 *
 * so both of the AIV's phases overlap a Cube stage. What each slot needs is the
 * REVERSE edge of the handshake the lockstep form never had to state, because
 * in lockstep each side's wait for the other also happened to keep it off the
 * other's buffer. Those are kFlagOperandsFree and kFlagProductFree below.
 * Cube-internal buffers stay single: L0A/L0B/L0C are re-used every chunk under
 * explicit M_MTE1 and FIX_M events, which is enough because the Cube stage is
 * far shorter than the vector work it hides behind.
 *
 * kVariantLockstep selects the old loop, at the same chunking, so the overlap
 * can be measured rather than asserted.
 *
 * sim_hadamard_aiv is the comparison: the same transform, all log2(D) stages on
 * the vector unit, in the shape the production codec runs it (one vector at a
 * time, Gather for the sub-block strides), split across the two subcores.
 */

#include "kernel_operator.h"

namespace {

// The Cube's fractal quantum, and the width of both operands. Sylvester makes
// any power-of-two D a multiple of it, so the tile never needs column padding.
constexpr uint32_t kTile = 16;

// Elements of a half Cube operand in one C0 block: 32 bytes over 2. The fp8
// decode's 32 is the same rule at a different element size.
constexpr uint32_t kHalfC0 = 16;

// The first butterfly stride the Cube does NOT absorb. Strides 1, 2, 4 and 8
// are the right multiply by H_16.
constexpr uint32_t kResidualFirstStride = kTile;

// Sub-block strides need a Gather shuffle; from 8 up a butterfly is a pair of
// block-strided Add/Sub. Mirrors kEarlyStages / kFp32PerBlock in
// turboquant_codec_950.h.
constexpr uint32_t kEarlyStages = 3;
constexpr uint32_t kFp32PerBlock = 8;
constexpr uint32_t kFp32PerRepeat = 64;

// Gather's srcBaseAddr is a byte offset within srcLocal, and every table here
// already indexes from the start of the tensor it is handed.
constexpr uint32_t kGatherSrcBase = 0;

// A vector instruction repeats at most 255 times, so a butterfly whose group
// count exceeds that is issued in several calls.
constexpr uint32_t kMaxRepeat = 255;

/*
 * Cross-core flag ids: two directions, two reverse edges, one id per slot.
 *
 * The lockstep form needed only two, "operands are in L1" and "the product is
 * in UB", because each side's wait for the other was ALSO what kept it off the
 * buffer the other was still using. Double-buffering removes that accident, so
 * the reverse edges have to be named:
 *
 *   kFlagOperandsFree  AIC -> AIV, the Cube's MTE1 has drained L1[slot] into
 *                      L0A/L0B and the AIV may restage it
 *   kFlagProductFree   AIV -> AIC, the residual and its writeback are done with
 *                      UB[slot] and the Fixpipe may land in it again
 *
 * Each id is a base plus slot = chunk % kSlots. Ids 11..14 belong to CANN's own
 * SyncAll (SYNC_AIC_FLAG, SYNC_AIV_FLAG, SYNC_AIC_AIV_FLAG, SYNC_AIV_ONLY_ALL
 * in dav_3510/kernel_operator_sync_impl.h) and GetffstMsg masks the id to four
 * bits, so 0..7 is the range this kernel may spend.
 */
constexpr uint16_t kFlagOperandsReady = 0;  // 0, 1    AIV -> AIC
constexpr uint16_t kFlagOperandsFree = 2;   // 2, 3    AIC -> AIV
constexpr uint16_t kFlagProductReady = 4;   // 4, 5    AIC -> AIV
constexpr uint16_t kFlagProductFree = 6;    // 6, 7    AIV -> AIC

// Depth of the L1/UB ping-pong, and the modulus every slot index is taken at.
constexpr uint32_t kSlots = 2;

// Variant bits. Selected by the host so one build can sweep them.
constexpr uint32_t kVariantHiLo = 0x1u;      // two Mmads, x = hi + lo
constexpr uint32_t kVariantDualDst = 0x2u;   // Fixpipe dualDstCtl = 0b01
constexpr uint32_t kVariantLockstep = 0x4u;  // the un-pipelined loop, for measurement

// L0C -> UB row major. No predefined FixpipeConfig sets isToUB, which exists
// only on arch35; the decode's kFixpipeToUb is the same constant.
constexpr AscendC::FixpipeConfig kFixpipeToUb = {AscendC::CO2Layout::ROW_MAJOR, true};

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
__aicore__ inline uint32_t AlignUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }

__aicore__ inline uint16_t CeilDivU16(uint32_t a, uint32_t b)
{
    return static_cast<uint16_t>((a + b - 1) / b);
}

/*
 * Vector -> MTE3, before any DMA of a UB buffer the vector unit just wrote.
 *
 * Transcribed from SyncVectorToMte3 in turboquant_mm_kernels.cpp, where it is
 * the fix for a whole session's worth of all-zero Cube products: the operand is
 * produced on PIPE_V and moved by PIPE_MTE3, PipeBarrier<PIPE_V> orders V
 * against V only, and without the event the DMA stages whatever the buffer held
 * before. Nothing faults.
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
 * Every stride used here divides the vector length, so the pattern is periodic
 * across a whole batch of vectors and one call covers all of them.
 */
__aicore__ inline void BlockButterfly(const AscendC::LocalTensor<float> &dst,
                                      const AscendC::LocalTensor<float> &src, uint32_t stride, uint32_t count)
{
    const uint32_t groups = count / (2 * stride);
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
 * The hybrid transform: strides 1, 2, 4, 8 on the Cube, the rest on the AIV.
 *
 * One MIX block, both vector subcores. `numVectors` vectors of `headDim` fp32
 * arrive in GM and leave in GM, normalised by invSqrtDim.
 */
class SimHadamardHybrid {
public:
    __aicore__ inline explicit SimHadamardHybrid(AscendC::TPipe *pipe) : pipe_(pipe) {}

    /*
     * headDim         a power of two, at least 32, so R = headDim / 16 is at
     *                 least 2 and the M split of the dual-destination Fixpipe
     *                 can land on a vector boundary
     * vectorsPerChunk divides numVectors; the host sizes it so one chunk's UB
     *                 working set fits
     * invSqrtDim      1 / sqrt(headDim), computed on the host so the device
     *                 does not need a square root
     */
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR h16, GM_ADDR output, uint32_t headDim, uint32_t numVectors,
                                uint32_t vectorsPerChunk, uint32_t variant, float invSqrtDim)
    {
        headDim_ = headDim;
        numVectors_ = numVectors;
        vectorsPerChunk_ = vectorsPerChunk;
        variant_ = variant;
        invSqrtDim_ = invSqrtDim;

        rowsPerVector_ = headDim_ / kTile;
        chunkElems_ = vectorsPerChunk_ * headDim_;
        chunkRows_ = vectorsPerChunk_ * rowsPerVector_;
        numChunks_ = numVectors_ / vectorsPerChunk_;

        // A LoadData of mStep fractals reads whole 16-row fractals out of L1
        // whatever m is, so every Cube buffer is sized on the padded row count
        // and the pad is zeroed before staging. At V = 1, D = 64 the chunk is
        // four rows and the pad is the other twelve.
        paddedRows_ = AlignUp(chunkRows_, kTile);
        paddedElems_ = paddedRows_ * kTile;

        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
        h16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(h16));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));

        // L1 on both halves of the MIX kernel: the AIV stages into these and
        // the AIC reads them. A1/B1 are the only Cube buffers a vector core may
        // hold - InitBuffer on A2/B2/CO1 hands it a base it cannot reach and
        // the first touch is `pem_lsu: unrecognize ldst addr`.
        //
        // A1 is the operand ping-pong. Both slots together are 4 * paddedElems_
        // halves - 32 KB at the D = 256, V = 16 shape, against an L1 measured in
        // megabytes - so the doubling is not a sizing question. B1 holds H_16,
        // which is the same 512 bytes for every chunk and stays single.
        for (uint32_t slot = 0; slot < kSlots; ++slot) {
            pipe_->InitBuffer(aHi1_[slot], paddedElems_ * sizeof(half));
            pipe_->InitBuffer(aLo1_[slot], paddedElems_ * sizeof(half));
        }
        pipe_->InitBuffer(b1_, kTile * kTile * sizeof(half));

        // L0A/L0B/L0C stay single-buffered: the Cube stage is one or two Mmads
        // against a 16 x 16 constant and is already far shorter than the vector
        // work it overlaps, so a second set of fractal buffers would buy
        // nothing. What it costs instead is two explicit events per chunk -
        // M_MTE1 before the load and FIX_M before the Mmad, in
        // LoadCubeOperands and MmadAndFixpipe.
        if ASCEND_IS_AIC {
            pipe_->InitBuffer(a2Hi_, paddedElems_ * sizeof(half));
            pipe_->InitBuffer(a2Lo_, paddedElems_ * sizeof(half));
            pipe_->InitBuffer(b2_, kTile * kTile * sizeof(half));
            pipe_->InitBuffer(co1_, paddedElems_ * sizeof(float));
        }

        // UB, per subcore. prodBuf_ is the Fixpipe destination and is sized for
        // the whole chunk even under kVariantDualDst, where only half of it is
        // written - the spare half costs nothing and keeps one sizing rule.
        //
        // prodBuf_ and tmpBuf_ are both per slot, and tmpBuf_ has to be: the
        // AIV runs the staging of chunk k BEFORE the residual of chunk k - 1,
        // and staging's hi/lo split uses tmpBuf_ as its fp32 scratch. Sharing
        // one tmp would have the stage of chunk k overwrite the buffer the
        // residual of chunk k - 1 is about to ping-pong through. Indexing both
        // by chunk % kSlots puts them in different slots and there is no
        // aliasing to order.
        //
        // tmpBuf_ is sized on paddedElems_, not chunkElems_: the hi/lo residual
        // Cast covers the fractal pad as well, so at any shape where the chunk
        // does not fill a whole 16-row fractal - V = 1, D = 64 is the smallest -
        // a chunkElems_ buffer is written past its end.
        pipe_->InitBuffer(inBuf_, paddedElems_ * sizeof(float));
        pipe_->InitBuffer(castBuf_, paddedElems_ * sizeof(half));
        for (uint32_t slot = 0; slot < kSlots; ++slot) {
            pipe_->InitBuffer(prodBuf_[slot], chunkElems_ * sizeof(float));
            pipe_->InitBuffer(tmpBuf_[slot], paddedElems_ * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if ((variant_ & kVariantLockstep) != 0) {
            ProcessLockstep();
            return;
        }
        // Not one interleaved loop with guards any more: the two halves no
        // longer visit the same chunk at the same time, so writing them as one
        // body would only hide that.
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
     * against the Mmad and Fixpipe of chunk k - 1, and the residual runs against
     * the Mmad and Fixpipe of chunk k, whose operands the staging just signalled.
     * In the lockstep form the AIV was idle for the whole of every Cube stage.
     *
     * The flag sets are conditioned so that every set has exactly one wait. A
     * surplus set is not harmless: these are hardware counters that the kernel
     * does not clear on exit, and the test binary launches this kernel several
     * times in a row, so a leftover count would let a later launch's wait fall
     * straight through.
     */
    __aicore__ inline void ProcessAiv()
    {
        // Prolog. Nothing to overlap chunk 0's staging against yet.
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
            // happens, and it can start chunk the moment the flag below lands.
            StageOperands(chunk * vectorsPerChunk_, slot);
            SignalOperandsReady(slot);

            AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagProductReady + prev));
            Residual(prev, (chunk - 1) * vectorsPerChunk_);
            if (chunk + 1 < numChunks_) {
                // Matched by the AIC's wait at chunk + 1, which reuses UB[prev].
                // On the last two chunks no such wait follows.
                SignalProductFree(prev);
            }
        }

        // Epilogue: the Cube has no successor to overlap the last residual with.
        const uint32_t last = (numChunks_ - 1) % kSlots;
        AscendC::CrossCoreWaitFlag(static_cast<uint16_t>(kFlagProductReady + last));
        Residual(last, (numChunks_ - 1) * vectorsPerChunk_);
    }

    /*
     * The cube half. One chunk behind the AIV in steady state.
     *
     * L1[slot] is released as soon as MTE1 has copied it into L0A/L0B, which is
     * long before the Fixpipe - that early release is what lets the AIV stage
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
     * The un-pipelined loop this kernel used to be, kept as the measurement
     * baseline rather than as a fallback.
     *
     * Everything is slot 0, so the AIV cannot stage chunk k until the Fixpipe
     * of chunk k - 1 has retired and the Cube sits idle through both of the
     * AIV's stages. The two paths compute the same thing; what differs is how
     * much of it overlaps, which is the only reason to keep this one around -
     * comparing the pipelined kernel at N chunks against itself at one chunk
     * would confound the pipeline with the chunking.
     */
    __aicore__ inline void ProcessLockstep()
    {
        for (uint32_t chunk = 0; chunk < numChunks_; ++chunk) {
            const uint32_t baseVector = chunk * vectorsPerChunk_;
            if ASCEND_IS_AIV {
                StageOperands(baseVector, 0);
                SignalOperandsReady(0);
            }
            if ASCEND_IS_AIC {
                AscendC::CrossCoreWaitFlag(kFlagOperandsReady);
                LoadCubeOperands(0);
                MmadAndFixpipe(0);
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kFlagProductReady);
            }
            if ASCEND_IS_AIV {
                // Also what keeps the next chunk's staging off an L1 the Cube
                // is still reading: the AIV cannot reach the next iteration
                // until the Fixpipe has retired.
                AscendC::CrossCoreWaitFlag(kFlagProductReady);
                Residual(0, baseVector);
            }
        }
    }

    /*
     * Both subcores set every AIV -> AIC flag, and that is not an oversight to
     * be tidied away: the arch35 flag in mode 0x02 is satisfied only once all
     * subcores of the pair have set it, so a set from one of them hangs the
     * Cube. See the addendum in turboquant_mm_kernels.cpp's SignalOperandsReady.
     *
     * It is what makes kFlagProductFree correct under single-destination
     * Fixpipe too, where subcore 1 owns no vectors, returns from Residual
     * immediately and signals at once: the AIC is still held until subcore 0 -
     * the one that actually holds the product - has signalled as well.
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
        return (variant_ & kVariantDualDst) != 0 && (vectorsPerChunk_ % 2 == 0);
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
     * fp32 GM -> fp16 UB -> L1, in NZ order, which for a 16-column half image
     * is plain row major (see the file header).
     *
     * Both subcores stage the whole chunk rather than half each. The AIC's
     * single CrossCoreWaitFlag is released by the pair of sets, so a split
     * would be safe only if that quorum reading is exact; duplicating a copy
     * this small is the cheaper way to not depend on it. Splitting it is the
     * first thing to try if the staging ever shows up in a profile.
     *
     * inBuf_ and castBuf_ stay single-buffered: each is written and fully
     * consumed inside one call, and the events below carry them across calls.
     * tmpBuf_ is per slot because the residual of the PREVIOUS chunk runs after
     * this call and ping-pongs through its own slot's copy.
     */
    __aicore__ inline void StageOperands(uint32_t baseVector, uint32_t slot)
    {
        AscendC::LocalTensor<float> in = inBuf_.Get<float>();
        AscendC::LocalTensor<half> cast = castBuf_.Get<half>();
        AscendC::LocalTensor<float> tmp = tmpBuf_[slot].Get<float>();

        // The previous chunk's Cast and Sub read inBuf_ on PIPE_V and the load
        // below refills it on PIPE_MTE2. Nothing else orders that pair: the
        // lockstep form got away with it because a whole Cube stage separated
        // the two, and the pipeline is exactly the change that removes the gap.
        SyncEvent<AscendC::HardEvent::V_MTE2>();

        // H_16 is the same for every chunk, so it is staged once.
        if (baseVector == 0) {
            AscendC::DataCopy(cast, h16Gm_, kTile * kTile);
            SyncEvent<AscendC::HardEvent::MTE2_MTE3>();
            AscendC::DataCopy(b1_.Get<half>(), cast, kTile * kTile);
        }

        AscendC::DataCopy(in, inputGm_[static_cast<uint64_t>(baseVector) * headDim_], chunkElems_);
        SyncEvent<AscendC::HardEvent::MTE2_V>();
        if (paddedElems_ > chunkElems_) {
            // The rows the fractal load reads past the chunk. Left undefined
            // they would reach the Mmad as fp16 garbage and could raise
            // inf/nan on rows the Fixpipe never copies out.
            AscendC::Duplicate(in[chunkElems_], 0.0f, paddedElems_ - chunkElems_);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // hi = fp16(x). castBuf_ was last read by an MTE3 copy - into B1 on the
        // first chunk, into A1 on every later one - and MTE2_V orders MTE2
        // against V and says nothing about MTE3, so overwriting it needs its
        // own event or that image is still in flight when the cast lands on it.
        SyncEvent<AscendC::HardEvent::MTE3_V>();
        AscendC::Cast(cast, in, AscendC::RoundMode::CAST_RINT, paddedElems_);
        SyncVectorToMte3();
        AscendC::DataCopy(aHi1_[slot].Get<half>(), cast, paddedElems_);

        if ((variant_ & kVariantHiLo) != 0) {
            // lo = fp16(x - fp32(hi)). castBuf_ is still being read by the DMA
            // above, so the residual is formed in fp32 first and the cast
            // buffer is only reused once MTE3 has released it.
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
     * The fractal contract is transcribed from TurboQuantCubeMm::LoadA, which
     * CANN's load_to_l0a_load2dV2.h pins: mStep counts 16-row fractals, kStep
     * counts C0 elements, and srcStride / dstStride are the leading extents
     * over 16 and are not optional - zero for either silently reads the wrong
     * fractals.
     */
    __aicore__ inline void LoadA(const AscendC::LocalTensor<half> &src, const AscendC::LocalTensor<half> &dst)
    {
        AscendC::LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = CeilDivU16(paddedRows_, kTile);
        p.kStep = CeilDivU16(kTile, kHalfC0);
        p.srcStride = CeilDivU16(paddedRows_, kTile);
        p.dstStride = CeilDivU16(paddedRows_, kTile);
        p.ifTranspose = false;
        AscendC::LoadData(dst, src, p);
    }

    /*
     * B is H_16 staged [n, k], which is the form that loads with one LoadData
     * and ifTranspose FALSE - the flag reads as the opposite of what it does.
     * H_16 is symmetric, so the [n, k] image and the [k, n] image are the same
     * bytes and the product is X . H_16^T either way.
     */
    __aicore__ inline void LoadB()
    {
        AscendC::LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = CeilDivU16(kTile, kTile);
        p.kStep = CeilDivU16(kTile, kHalfC0);
        p.srcStride = CeilDivU16(kTile, kTile);
        p.dstStride = CeilDivU16(kTile, kTile);
        p.ifTranspose = false;
        AscendC::LoadData(b2_.Get<half>(), b1_.Get<half>(), p);
    }

    /*
     * L1[slot] -> L0A/L0B. The first half of the Cube stage, split out from the
     * Mmad so the caller can release L1[slot] to the AIV between the two.
     *
     * L0A/L0B are single-buffered, so MTE1 may not refill them until the
     * PREVIOUS chunk's Mmad has read them. In the lockstep form that held by
     * accident - the AIC could not reach this point until an AIV that was
     * itself waiting on the Fixpipe had signalled - and the pipeline is exactly
     * the change that removes the accident, so M_MTE1 is now explicit.
     */
    __aicore__ inline void LoadCubeOperands(uint32_t slot)
    {
        SyncEvent<AscendC::HardEvent::M_MTE1>();
        LoadA(aHi1_[slot].Get<half>(), a2Hi_.Get<half>());
        if ((variant_ & kVariantHiLo) != 0) {
            LoadA(aLo1_[slot].Get<half>(), a2Lo_.Get<half>());
        }
        LoadB();

        // MTE1 -> M and M -> FIX, both mandatory: PipeBarrier orders a pipe
        // against itself only, so without these the Mmad reads an L0A/L0B that
        // LoadData is still filling and the Fixpipe reads an L0C the Mmad has
        // not finished writing. The symptom of getting this wrong is a
        // partly-zero product, not a fault.
        SyncEvent<AscendC::HardEvent::MTE1_M>();
    }

    // The chunk's lower four butterfly stages, as one Mmad - or two when the
    // hi/lo split is on, accumulating into the same L0C - and the Fixpipe that
    // hands the result to UB[slot].
    __aicore__ inline void MmadAndFixpipe(uint32_t slot)
    {
        const bool hiLo = (variant_ & kVariantHiLo) != 0;

        // L0C is single-buffered too, so the previous chunk's Fixpipe has to
        // have drained it before this Mmad overwrites it. Same reasoning as
        // M_MTE1 above: implicit in lockstep, explicit once pipelined.
        SyncEvent<AscendC::HardEvent::FIX_M>();

        AscendC::LocalTensor<float> acc = co1_.Get<float>();
        AscendC::Mmad(acc, a2Hi_.Get<half>(), b2_.Get<half>(),
                      AscendC::MmadParams(static_cast<uint16_t>(chunkRows_), static_cast<uint16_t>(kTile),
                                          static_cast<uint16_t>(kTile), 0, false, true));
        if (hiLo) {
            // cmatrixInitVal false: accumulate onto the hi product already in
            // L0C. Back-to-back Mmads on one accumulator are what CANN's own
            // k-loop issues, and they need no barrier between them.
            AscendC::Mmad(acc, a2Lo_.Get<half>(), b2_.Get<half>(),
                          AscendC::MmadParams(static_cast<uint16_t>(chunkRows_), static_cast<uint16_t>(kTile),
                                              static_cast<uint16_t>(kTile), 0, false, false));
        }

        SyncEvent<AscendC::HardEvent::M_FIX>();

        // srcStride is m rounded UP to a multiple of 16 - a different
        // convention from LoadData's m / 16, two lines away. dstStride is the
        // UB row pitch in elements.
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fp(
            static_cast<uint16_t>(kTile), static_cast<uint16_t>(chunkRows_),
            static_cast<uint16_t>(paddedRows_), kTile);
        if (DualDst()) {
            // 0b01: split in M, chunkRows_ / 2 x 16 into EACH subcore's UB. The
            // assert in FixpipeL0cToUB wants an even M; DualDst() has already
            // established that the split lands on a vector boundary, which is
            // the stronger condition.
            fp.dualDstCtl = 0b01;
            fp.subBlockId = false;
        }
        AscendC::Fixpipe<float, float, kFixpipeToUb>(prodBuf_[slot].Get<float>(), acc, fp);
        AscendC::PipeBarrier<PIPE_FIX>();
    }

    /*
     * The stages above stride 8: in the R x 16 tile they are butterflies over
     * whole rows. log2(R) Add/Sub pairs for this subcore's share of the chunk,
     * then the 1 / sqrt(D) the transform is normalised by.
     *
     * An odd number of stages leaves the result in the scratch buffer, so which
     * of the two holds it is tracked rather than assumed - D = 128 and D = 512
     * end in tmp, D = 64 and D = 256 in prod.
     */
    __aicore__ inline void Residual(uint32_t slot, uint32_t baseVector)
    {
        const uint32_t mine = MyVectors();
        if (mine == 0) {
            return;
        }
        const uint32_t count = mine * headDim_;
        AscendC::LocalTensor<float> src = prodBuf_[slot].Get<float>();
        AscendC::LocalTensor<float> dst = tmpBuf_[slot].Get<float>();

        for (uint32_t stride = kResidualFirstStride; stride < headDim_; stride <<= 1) {
            BlockButterfly(dst, src, stride, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<float> hold = src;
            src = dst;
            dst = hold;
        }

        AscendC::Muls(src, src, invSqrtDim_, count);
        SyncVectorToMte3();
        const uint64_t out = (static_cast<uint64_t>(baseVector) + MyChunkOffset()) * headDim_;
        AscendC::DataCopy(outputGm_[out], src, count);
        // No MTE3 -> V here, and the two buffers this DMA may still be reading
        // are covered elsewhere. prodBuf_[slot] is next written by the Fixpipe,
        // which the AIC does not issue until kFlagProductFree - set on
        // PIPE_MTE3 after this copy - has arrived. tmpBuf_[slot] is next
        // written by the staging of chunk + kSlots, which begins with its own
        // MTE3 -> V. Draining the writeback here instead would stall the AIV at
        // precisely the point the pipeline exists to keep busy.
    }

    AscendC::TPipe *pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> aHi1_[kSlots];
    AscendC::TBuf<AscendC::TPosition::A1> aLo1_[kSlots];
    AscendC::TBuf<AscendC::TPosition::B1> b1_;
    AscendC::TBuf<AscendC::TPosition::A2> a2Hi_;
    AscendC::TBuf<AscendC::TPosition::A2> a2Lo_;
    AscendC::TBuf<AscendC::TPosition::B2> b2_;
    AscendC::TBuf<AscendC::TPosition::CO1> co1_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> inBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> castBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> prodBuf_[kSlots];
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf_[kSlots];
    AscendC::GlobalTensor<float> inputGm_;
    AscendC::GlobalTensor<half> h16Gm_;
    AscendC::GlobalTensor<float> outputGm_;
    uint32_t headDim_ = 0;
    uint32_t numVectors_ = 0;
    uint32_t vectorsPerChunk_ = 0;
    uint32_t variant_ = 0;
    uint32_t rowsPerVector_ = 0;
    uint32_t chunkElems_ = 0;
    uint32_t chunkRows_ = 0;
    uint32_t numChunks_ = 0;
    uint32_t paddedRows_ = 0;
    uint32_t paddedElems_ = 0;
    float invSqrtDim_ = 1.0f;
};

/*
 * The comparison: all log2(D) stages on the vector unit, in the shape the
 * production codec runs them - one vector at a time, three Gather shuffles for
 * the strides that straddle a 32 B block and block-strided Add/Sub pairs above
 * it. The two subcores take alternate vectors, which is the only free
 * parallelism the AIV-only form has.
 *
 * `tables` is the same image TurboQuantCodec reads: for each of stages 0, 1, 2,
 * headDim fp32 sign lanes followed by headDim uint32 Gather byte offsets.
 */
class SimHadamardAiv {
public:
    __aicore__ inline explicit SimHadamardAiv(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR tables, GM_ADDR output, uint32_t headDim,
                                uint32_t numVectors, float invSqrtDim)
    {
        headDim_ = headDim;
        numVectors_ = numVectors;
        invSqrtDim_ = invSqrtDim;
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
        tablesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tables));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));

        pipe_->InitBuffer(xBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(tmpBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(swapBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(tableBuf_, kEarlyStages * 2 * headDim_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        // Both halves of a MIX block enter this function and every statement in
        // it is vector work, so the whole body is guarded. Unguarded, the cube
        // core issues UB reads and a GM writeback it does not own: measured
        // here as a max|err| of 2.03 against cpu_fwht with a silent
        // excp_log.dump, which reads exactly like an arithmetic bug.
        if ASCEND_IS_AIV {
            AscendC::LocalTensor<int32_t> tables = tableBuf_.Get<int32_t>();
            AscendC::DataCopy(tables, tablesGm_, kEarlyStages * 2 * headDim_);
            SyncEvent<AscendC::HardEvent::MTE2_V>();

            const uint32_t sub = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
            const uint32_t step = static_cast<uint32_t>(AscendC::GetSubBlockNum());
            for (uint32_t v = sub; v < numVectors_; v += step) {
                Transform(v, tables);
            }
        }
    }

private:
    __aicore__ inline void Transform(uint32_t v, const AscendC::LocalTensor<int32_t> &tables)
    {
        AscendC::LocalTensor<float> x = xBuf_.Get<float>();
        AscendC::LocalTensor<float> tmp = tmpBuf_.Get<float>();
        AscendC::LocalTensor<float> swap = swapBuf_.Get<float>();

        AscendC::DataCopy(x, inputGm_[static_cast<uint64_t>(v) * headDim_], headDim_);
        SyncEvent<AscendC::HardEvent::MTE2_V>();

        // Strides 1, 2, 4: the partner is inside the 32 B block, so the pair is
        // a Gather of the XOR partner plus a signed Add.
        for (uint32_t stage = 0; stage < kEarlyStages; ++stage) {
            AscendC::LocalTensor<float> sign = tables[stage * 2 * headDim_].ReinterpretCast<float>();
            AscendC::LocalTensor<uint32_t> offsets =
                tables[(stage * 2 + 1) * headDim_].ReinterpretCast<uint32_t>();
            AscendC::Gather(swap, x, offsets, kGatherSrcBase, headDim_);
            AscendC::Mul(tmp, x, sign, headDim_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(x, swap, tmp, headDim_);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Strides 8 .. D/2: log2(D / 8) of them, so which buffer holds the
        // result depends on D and has to be tracked rather than assumed.
        AscendC::LocalTensor<float> src = x;
        AscendC::LocalTensor<float> dst = tmp;
        for (uint32_t stride = kFp32PerBlock; stride < headDim_; stride <<= 1) {
            BlockButterfly(dst, src, stride, headDim_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<float> hold = src;
            src = dst;
            dst = hold;
        }

        AscendC::Muls(src, src, invSqrtDim_, headDim_);
        SyncVectorToMte3();
        AscendC::DataCopy(outputGm_[static_cast<uint64_t>(v) * headDim_], src, headDim_);
        // MTE3 -> MTE2 AND MTE3 -> V: the next vector's first touch of x is the
        // input DataCopy, which is MTE2, and its Gather and Mul are V. Ordering
        // the writeback against only one of the two leaves the other racing it.
        SyncEvent<AscendC::HardEvent::MTE3_MTE2>();
        SyncEvent<AscendC::HardEvent::MTE3_V>();
    }

    AscendC::TPipe *pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> swapBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tableBuf_;
    AscendC::GlobalTensor<float> inputGm_;
    AscendC::GlobalTensor<int32_t> tablesGm_;
    AscendC::GlobalTensor<float> outputGm_;
    uint32_t headDim_ = 0;
    uint32_t numVectors_ = 0;
    float invSqrtDim_ = 1.0f;
};

}  // namespace

// Entry points take GM_ADDR, not `__gm__ void *`: the generated launcher does
// not compile otherwise. Same note as in turboquant_kernels.cpp.
extern "C" __global__ __aicore__ void sim_hadamard_hybrid(GM_ADDR input, GM_ADDR h16, GM_ADDR output,
                                                          uint32_t headDim, uint32_t numVectors,
                                                          uint32_t vectorsPerChunk, uint32_t variant,
                                                          float invSqrtDim)
{
    AscendC::TPipe pipe;
    SimHadamardHybrid op(&pipe);
    op.Init(input, h16, output, headDim, numVectors, vectorsPerChunk, variant, invSqrtDim);
    op.Process();
}

extern "C" __global__ __aicore__ void sim_hadamard_aiv(GM_ADDR input, GM_ADDR tables, GM_ADDR output,
                                                       uint32_t headDim, uint32_t numVectors, float invSqrtDim)
{
    AscendC::TPipe pipe;
    SimHadamardAiv op(&pipe);
    op.Init(input, tables, output, headDim, numVectors, invSqrtDim);
    op.Process();
}

namespace vllm_ascend {

void sim_hadamard_hybrid_impl(void *stream, void *input, void *h16, void *output, uint32_t headDim,
                              uint32_t numVectors, uint32_t vectorsPerChunk, uint32_t variant, float invSqrtDim)
{
    sim_hadamard_hybrid<<<1, nullptr, stream>>>(input, h16, output, headDim, numVectors, vectorsPerChunk, variant,
                                                invSqrtDim);
}

void sim_hadamard_aiv_impl(void *stream, void *input, void *tables, void *output, uint32_t headDim,
                           uint32_t numVectors, float invSqrtDim)
{
    sim_hadamard_aiv<<<1, nullptr, stream>>>(input, tables, output, headDim, numVectors, invSqrtDim);
}

}  // namespace vllm_ascend
