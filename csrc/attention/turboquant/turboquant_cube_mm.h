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
 * The Cube half of the TurboQuant decode: two GEMMs per K/V tile, with the
 * operand type chosen by the mode.
 *
 * AscendC::Mmad on arch35 dispatches on <DstT, Src0T, Src1T>:
 *
 *   <float, fp8_e4m3fn_t, fp8_e4m3fn_t>   -> mad       (kv4fp8, kv5fp8)
 *   <float, fp4x2_e2m1_t, fp4x2_e2m1_t>   -> mad_mx    (kv3fp4)
 *
 * so the instruction follows from TurboQuantModeTraits<MODE>::kOperand.
 *
 * The task decomposition is (token, kvHead, split): the Cube's M granularity is
 * 16, so the query heads sharing one kv head are batched into M to make a
 * [headsPerKv, D] x [D, rows] GEMM.
 *
 * Operands reach L1 already in NZ -- DataCopy(L1, UB, Nd2NzParams) does not
 * exist for 1-byte operands on arch35 -- so the unpack emits NZ order directly.
 * For C0 = 32 / sizeof(operand) elements,
 *
 *     nz(r, c) = (c / C0) * rows * C0 + r * C0 + (c % C0)
 *
 * and a plain contiguous DataCopy then moves UB -> L1.  NzOffset() below is
 * that formula.
 *
 * The accumulator stays in UB, not in L0C: flash decoding rescales it by
 * exp(m_old - m_new) at every tile and that multiply has no expression on the
 * Cube, so each tile's PV product is Fixpipe'd to UB.
 */

#ifndef VLLM_ASCEND_ATTENTION_TURBOQUANT_CUBE_MM_H
#define VLLM_ASCEND_ATTENTION_TURBOQUANT_CUBE_MM_H

#include "kernel_operator.h"
#include "turboquant_mode.h"

namespace vllm_ascend {
namespace turboquant {

// L0C -> UB, row major.  No predefined FixpipeConfig sets isToUB, which exists
// only on arch35 and 5102; on any other arch this header does not compile, and
// that is correct -- there is no Cube path for them here.
constexpr AscendC::FixpipeConfig kFixpipeToUb = {AscendC::CO2Layout::ROW_MAJOR, true};

// The Cube's M/N granularity for 8-bit operands, and the K granularity one
// LoadData2D step covers.  A GEMM whose M is smaller is padded, not split.
constexpr uint32_t kCubeTileM = 16;
constexpr uint32_t kCubeKStep = 64;

// Cross-core flag ids.  Two are enough: the AIV signals "operands are in L1"
// and the AIC signals "the product is in UB".  They alternate within a tile, so
// a third would only be needed if two tiles were ever in flight at once.
constexpr uint16_t kFlagOperandsReady = 0;
constexpr uint16_t kFlagProductReady = 1;

// The operand type for a mode, as a C++ type.  fp4x2_e2m1_t names a *pair* of
// fp4 values, so size fp4 buffers through TurboQuantModeCodec::OperandElems and
// never through the head size.
template <TurboQuantMode MODE>
struct TurboQuantOperandType;

template <>
struct TurboQuantOperandType<TurboQuantMode::KV3_FP4> {
    using Type = fp4x2_e2m1_t;
};
template <>
struct TurboQuantOperandType<TurboQuantMode::KV4_FP8> {
    using Type = fp8_e4m3fn_t;
};
template <>
struct TurboQuantOperandType<TurboQuantMode::KV5_FP8> {
    using Type = fp8_e4m3fn_t;
};

/*
 * The Cube stage for one mode.  Owns L1 on both halves of the MIX kernel and
 * L0A, L0B and L0C on the AIC alone -- see Init, where that split is not a
 * micro-optimisation; the AIV's unpack buffers belong to the codec, and the two
 * halves meet at StageA/StageB (UB -> L1, an AIV-side MTE3) and at the Fixpipe
 * (L0C -> UB).
 *
 *   maxM   rows of the A operand, i.e. the GQA group padded to kCubeTileM
 *   maxN   columns of the B operand; kTileRows for the score GEMM and head_size
 *          for the context GEMM, so this is sized off the larger
 *   maxK   the reduction length; head_size for scores, kTileRows for context
 */
template <TurboQuantMode MODE>
class TurboQuantCubeMm {
public:
    using OperandT = typename TurboQuantOperandType<MODE>::Type;
    static constexpr bool kIsFp4 = TurboQuantModeTraits<MODE>::kOperand == TurboQuantOperand::kFp4E2m1;

    // Operand elements for `n` logical coordinates.  fp4 packs two per element.
    __aicore__ static constexpr uint32_t OperandElems(uint32_t n) { return kIsFp4 ? n / 2 : n; }

    /*
     * headSize   the reduction length of the score GEMM and the width of the
     *            context GEMM; a power of two in [64, 256]
     * tileRows   K/V rows per tile; the width of the score GEMM and the
     *            reduction length of the context GEMM
     */
    __aicore__ inline void Init(AscendC::TPipe *pipe, uint32_t headSize, uint32_t tileRows)
    {
        headSize_ = headSize;
        tileRows_ = tileRows;

        // Two A1 buffers -- the query and the probability row, both live at
        // once -- and one B1 that holds K and then V within a tile.
        const uint32_t qBytes = kCubeTileM * OperandElems(headSize_);
        const uint32_t pBytes = kCubeTileM * OperandElems(tileRows_);
        const uint32_t bBytes = tileRows_ * OperandElems(headSize_);
        // L1 on both halves of the MIX kernel: the AIV stages into these and
        // the AIC reads them.
        pipe->InitBuffer(aQ1_, qBytes);
        pipe->InitBuffer(aP1_, pBytes);
        pipe->InitBuffer(b1_, bBytes);

        // L0 on the AIC alone.  A vector core has no L0A, L0B or L0C, so an
        // unconditional InitBuffer here hands the AIV tensors whose base is not
        // an address it can reach; the part reports the first touch as
        // `pem_lsu: unrecognize ldst addr` with `ldst_addr: 2` and the whole
        // decode stalls there.  Nothing on the AIV side reads these -- LoadA,
        // LoadBFrom*, and Compute all run under ASCEND_IS_AIC at their call
        // sites -- so leaving them unallocated on that core costs nothing.
        if ASCEND_IS_AIC {
            pipe->InitBuffer(a2_, qBytes > pBytes ? qBytes : pBytes);
            // L0B holds the [k, n] form's chunks end to end: ceil(k/16)/2 of
            // them, each ceil(n/16) * 16 * 32 elements. For the shapes here
            // that is the same bBytes the tile occupies, but it is written out
            // rather than assumed because the chunking rounds k up to a
            // multiple of 32.
            pipe->InitBuffer(b2_, bBytes);
            // L0C carries [kCubeTileM, headSize] fp32, which also covers
            // [kCubeTileM, tileRows] since tileRows <= headSize on every shape
            // here.
            pipe->InitBuffer(co1_, kCubeTileM * headSize_ * sizeof(float));
        }
    }

    // AIV side: the three L1 landing buffers, exposed so the caller can place
    // into them with whatever DataCopy stride it needs.  Everything written
    // here must already be in NZ order, and the writes are MTE3 from the vector
    // core's point of view, so the flag that follows them is set on PIPE_MTE3.
    __aicore__ inline AscendC::LocalTensor<OperandT> A1Query() { return aQ1_.template Get<OperandT>(); }
    __aicore__ inline AscendC::LocalTensor<OperandT> A1Probs() { return aP1_.template Get<OperandT>(); }
    __aicore__ inline AscendC::LocalTensor<OperandT> B1() { return b1_.template Get<OperandT>(); }

    // The NZ position of logical coordinate (r, c) in a `rows` x `cols` tile,
    // transcribed from TransND2NZ in
    // .../impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h.  The host
    // mirror is turboquant_host::NzOffset and a test pins the two together.
    __aicore__ static constexpr uint32_t NzOffset(uint32_t r, uint32_t c, uint32_t rows)
    {
        return (c / kOperandC0) * rows * kOperandC0 + r * kOperandC0 + (c % kOperandC0);
    }

    // Elements of OperandT in one C0 block: 32 bytes over the element size, so
    // 32 for fp8 and 32 fp4-pairs (64 fp4 coordinates) for fp4.
    static constexpr uint32_t kOperandC0 = 32;

    /*
     * AIC side, GEMM 1: scores[m, n] = Q[m, k] . K[n, k]^T, fp32 accumulate.
     * K is staged as [n, k], which loads with one LoadData at ifTranspose
     * false.  See LoadBFromNk.
     *
     * NOT YET VERIFIED, and the only part of this file that is not: this B form
     * does not reproduce the host product in test_sim_950pr_cube_gemm, so the
     * defect is confined to LoadBFromNk.  The fix is to stage K transposed and
     * route through the proven [k, n] form of GemmContext.
     */
    __aicore__ inline void GemmScores(const AscendC::LocalTensor<float> &dstUb, uint32_t m, uint32_t k, uint32_t n)
    {
        aActive_ = aQ1_.template Get<OperandT>();
        LoadA(m, k);
        LoadBFromNk(k, n);
        Compute(dstUb, m, k, n);
    }

    // AIC side, GEMM 2: ctx[m, n] = P[m, k] . V[k, n], fp32 accumulate.  V is
    // staged as [k, n], which for the B operand needs ifTranspose true and, for
    // an 8-bit type, a loop.  See LoadBFromKn.
    __aicore__ inline void GemmContext(const AscendC::LocalTensor<float> &dstUb, uint32_t m, uint32_t k, uint32_t n,
                                       uint32_t variant = 0)
    {
        aActive_ = aP1_.template Get<OperandT>();
        LoadA(m, k);
        LoadBFromKn(k, n, variant);
        Compute(dstUb, m, k, n);
    }

private:
    /*
     * THE FRACTAL CONTRACT for LoadData2DParamsV2, transcribed from
     *
     *   .../impl/adv_api/detail/matmul/stage/split/load_to_l0a/load_to_l0a_load2dV2.h
     *   .../impl/adv_api/detail/matmul/stage/split/load_to_l0b/load_to_l0b_load2dV2.h
     *
     * and pinned numerically by csrc/tests/sim/test_sim_950pr_cube_gemm.cpp.
     *
     *   * mStep counts 16-row fractals; kStep counts C0 elements, and C0 is 32
     *     for every 8-bit type (AuxGetC0Size), not 64.
     *   * srcStride and dstStride are not optional.  srcStride is the L1
     *     matrix's leading extent over 16 and dstStride is the L0 fragment's
     *     over 16; zero for either silently reads the wrong fractals.
     *   * ifTranspose on the B operand means the opposite of what it reads as.
     *     B staged [n, k] loads with ifTranspose FALSE; B staged [k, n] loads
     *     with ifTranspose TRUE.
     *   * an 8-bit B operand in the [k, n] orientation must be loaded in
     *     mStep = 2 chunks; a single call with mStep = 4 raises
     *     mte_instr_addr_misalign.
     */
    static constexpr uint16_t kFractalRows = 16;
    // C0 elements of an 8-bit Cube operand. fp4 packs two per element, so a
    // LocalTensor<fp4x2> of C0 elements covers 2 * C0 coordinates.
    static constexpr uint16_t kC0 = 32;
    // The only mStep an 8-bit [k, n] B operand accepts.
    static constexpr uint16_t kB8MStep = 2;

    __aicore__ static constexpr uint16_t CeilDivU16(uint32_t a, uint32_t b)
    {
        return static_cast<uint16_t>((a + b - 1) / b);
    }

    // A is [m, k] in L1; L0A wants [m, k]. Straight load.
    __aicore__ inline void LoadA(uint32_t m, uint32_t k)
    {
        AscendC::LocalTensor<OperandT> ta1 = aActive_;
        AscendC::LocalTensor<OperandT> ta2 = a2_.template Get<OperandT>();
        AscendC::LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = CeilDivU16(m, kFractalRows);
        p.kStep = CeilDivU16(k, kC0);
        p.srcStride = CeilDivU16(m, kFractalRows);
        p.dstStride = CeilDivU16(m, kFractalRows);
        p.ifTranspose = false;
        AscendC::LoadData(ta2, ta1, p);
    }

    // B is [n, k] in L1. One load, no loop, ifTranspose false.
    __aicore__ inline void LoadBFromNk(uint32_t k, uint32_t n)
    {
        AscendC::LocalTensor<OperandT> tb1 = b1_.template Get<OperandT>();
        AscendC::LocalTensor<OperandT> tb2 = b2_.template Get<OperandT>();
        AscendC::LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = CeilDivU16(n, kFractalRows);
        p.kStep = CeilDivU16(k, kC0);
        p.srcStride = CeilDivU16(n, kFractalRows);
        p.dstStride = CeilDivU16(n, kFractalRows);
        p.ifTranspose = false;
        AscendC::LoadData(tb2, tb1, p);
    }

    /*
     * B is [k, n] in L1, loaded with ifTranspose true and, for an 8-bit
     * operand, in mStep = 2 chunks.
     *
     * `variant` exists so test_sim_950pr_cube_gemm can sweep the parameters
     * this transcription is uncertain about rather than assert a guess; 0 is
     * the shipping path and the only one production ever passes. See the
     * variant table in that test.
     */
    __aicore__ inline void LoadBFromKn(uint32_t k, uint32_t n, uint32_t variant = 0)
    {
        AscendC::LocalTensor<OperandT> tb1 = b1_.template Get<OperandT>();
        AscendC::LocalTensor<OperandT> tb2 = b2_.template Get<OperandT>();
        AscendC::LoadData2DParamsV2 p;
        p.kStartPosition = 0;
        p.kStep = (variant == 5) ? CeilDivU16(n, kFractalRows) : CeilDivU16(n, kC0);
        p.srcStride = (variant == 6) ? CeilDivU16(k, kC0) : CeilDivU16(k, kFractalRows);
        p.dstStride = (variant == 8) ? CeilDivU16(n, kC0) : CeilDivU16(n, kFractalRows);
        p.ifTranspose = true;

        const uint16_t mSteps = CeilDivU16(k, kFractalRows);

        // Variant 7: no chunking at all -- one call at the full mStep. The b8
        // restriction this loop exists for may not apply on every CANN.
        if (variant == 7) {
            p.mStartPosition = 0;
            p.mStep = mSteps;
            AscendC::LoadData(tb2, tb1, p);
            return;
        }

        // CANN's dstAddrStride is CeilAlign(madN, ALIGN_NUM) * ONE_BLK_SIZE,
        // and CeilAlign rounds UP to a multiple, so at n = 256 that is
        // 256 * 32 = 8192 elements -- exactly one chunk's k-rows x n.
        uint32_t dstStrideElems =
            static_cast<uint32_t>(CeilDivU16(n, kFractalRows)) * kFractalRows * kC0;
        if (variant == 1) {
            dstStrideElems /= kC0;  // the same figure read as 32-byte blocks
        } else if (variant == 2) {
            dstStrideElems = static_cast<uint32_t>(CeilDivU16(n, kFractalRows)) * kC0;
        }

        const uint16_t loops = CeilDivU16(mSteps, kB8MStep);
        p.mStep = kB8MStep;
        uint32_t dstOffset = 0;
        AscendC::TPipe *pipe = GetTPipePtr();
        for (uint16_t i = 0; i < loops; ++i) {
            p.mStartPosition = static_cast<uint32_t>(kB8MStep) * i;
            AscendC::LoadData(tb2[dstOffset], tb1, p);
            /*
             * An MTE1 -> M set/wait after EVERY chunk, and it is not redundant
             * with the one Compute issues.
             *
             * Compute's single set/wait is enough on the first Cube GEMM of a
             * process and not on the ones after it: with a dirty event state
             * left by an earlier launch, the Mmad issues against an L0B whose
             * second chunk LoadData has not finished writing, and reads zeros
             * there. The failure is total rather than partial -- the whole
             * product comes back zero -- which is what makes it read as a
             * layout error rather than a race.
             *
             * Measured by test_sim_950pr_cube_gemm's LoadBFromKnVariantSweep:
             * with this sync the context GEMM is exact at every position in the
             * launch order; without it, exact only when it runs first. Variant 4
             * of that sweep is this path without the sync, kept so the pair can
             * be re-measured.
             */
            if (variant != 4) {
                const event_t ev = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::MTE1_M));
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ev);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ev);
            }
            if (variant == 3) {
                AscendC::PipeBarrier<PIPE_MTE1>();
            }
            dstOffset += dstStrideElems;
        }
    }

    /*
     * cmatrixInitVal is always true: every call starts a fresh product, because
     * the accumulator lives in UB (see the file header) rather than in L0C.
     */
    __aicore__ inline void Compute(const AscendC::LocalTensor<float> &dstUb, uint32_t m, uint32_t k, uint32_t n)
    {
        AscendC::LocalTensor<OperandT> ta2 = a2_.template Get<OperandT>();
        AscendC::LocalTensor<OperandT> tb2 = b2_.template Get<OperandT>();
        AscendC::LocalTensor<float> tco = co1_.template Get<float>();

        // Cross-pipe synchronisation, and it is not optional: PipeBarrier<MTE1>
        // orders MTE1 against MTE1 and says nothing about the M pipe, so
        // without these the Mmad reads an L0A/L0B that LoadData is still
        // filling.  Same for M -> FIX.
        AscendC::TPipe *pipe = GetTPipePtr();
        const event_t mte1ToM = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::MTE1_M));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1ToM);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1ToM);

        AscendC::Mmad(tco, ta2, tb2,
                      AscendC::MmadParams(static_cast<uint16_t>(m), static_cast<uint16_t>(n),
                                          static_cast<uint16_t>(k), 0, false, true));

        const event_t mToFix = static_cast<event_t>(pipe->FetchEventID(AscendC::HardEvent::M_FIX));
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(mToFix);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(mToFix);

        // dstStride is the UB row pitch in elements, i.e. n.  srcStride is m
        // rounded UP to a multiple of 16 -- not m over 16, which is the
        // convention LoadData uses two lines earlier.
        AscendC::Fixpipe<float, float, kFixpipeToUb>(
            dstUb, tco,
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>(
                static_cast<uint16_t>(n), static_cast<uint16_t>(m),
                static_cast<uint16_t>(CeilDivU16(m, kFractalRows) * kFractalRows), n));
        AscendC::PipeBarrier<PIPE_FIX>();
    }

    AscendC::LocalTensor<OperandT> aActive_;

    AscendC::TBuf<AscendC::TPosition::A1> aQ1_;
    AscendC::TBuf<AscendC::TPosition::A1> aP1_;
    AscendC::TBuf<AscendC::TPosition::B1> b1_;
    AscendC::TBuf<AscendC::TPosition::A2> a2_;
    AscendC::TBuf<AscendC::TPosition::B2> b2_;
    AscendC::TBuf<AscendC::TPosition::CO1> co1_;
    uint32_t headSize_ = 0;
    uint32_t tileRows_ = 0;
};

}  // namespace turboquant
}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_ATTENTION_TURBOQUANT_CUBE_MM_H
