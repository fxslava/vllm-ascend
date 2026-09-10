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
 * WHICH INSTRUCTION RUNS IS DECIDED BY THE OPERAND TUPLE, NOT BY THIS FILE.
 * AscendC::Mmad on arch35 inspects <DstT, Src0T, Src1T> and dispatches:
 *
 *   <float, fp8_e4m3fn_t, fp8_e4m3fn_t>   -> mad       (kv4fp8, kv5fp8)
 *   <float, fp4x2_e2m1_t, fp4x2_e2m1_t>   -> mad_mx    (kv3fp4)
 *
 * so both modes go through the same call here and the hardware instruction
 * follows from TurboQuantModeTraits<MODE>::kOperand.  The dispatch lives in
 * .../impl/basic_api/dav_3510/kernel_operator_mm_impl.h::MmadCal; the fp8 tuple
 * is deliberately NOT one of the mx_fp8_* microscaled types, because this
 * pipeline already carries a per-vector scale that is better conditioned than a
 * per-32-element e8m0 exponent, and folding that scale into the score row costs
 * one Mul over `rows` instead of a scale matrix per tile.
 *
 * THE M DIMENSION IS THE GQA GROUP.  A decode step is a matrix-vector product
 * per query head, and the Cube's M granularity is 16, so a per-head task would
 * waste 15/16 of every Mmad.  Batching the query heads that share one kv head
 * into M turns it into a [headsPerKv, D] x [D, rows] GEMM.  On Qwen3.5-2B that
 * is 4 heads over 2 kv heads, so the waste drops from 16x to 4x -- the task
 * decomposition is therefore (token, kvHead, split), not (token, head, split),
 * and that is the one structural difference between this decode and the AIV-only
 * one in turboquant_kernels.cpp.  It is not a tuning choice: a per-head task
 * cannot use the Cube usefully at all.
 *
 * THE OPERANDS REACH L1 ALREADY IN NZ, AND THAT COSTS NOTHING.  The obvious
 * route is DataCopy(L1, UB, Nd2NzParams), and on arch35 it does not exist for
 * 1-byte operands: CANN implements the UB-side ND->NZ transform as a strided
 * Adds (TransND2NZ in .../dav_3510/kernel_operator_data_copy_impl.h), Adds has
 * no int8/fp8 overload, and the mask it builds narrows for a 32-element block.
 * Both failures are compile-time; neither is a shape this path can avoid.
 *
 * Routing the tile through GM instead -- AIV writes unpacked fp8, AIC reads it
 * back with the GlobalTensor Nd2Nz overload -- compiles, and would triple the
 * decode's GM traffic: a d=256 tile is 8 KB of fp8 written and read against the
 * 5 KB of packed codes it came from.  Tripling the traffic to save arithmetic
 * inverts the entire point of a compressed KV cache.
 *
 * So the unpack emits NZ order directly.  Its last step is a Gather over the
 * centroid table, and a Gather's offset table is arbitrary by construction, so
 * writing the output in NZ order rather than row-major is a different constant
 * table and not a single extra instruction.  The layout is the one TransND2NZ
 * above spells out: for C0 = 32 / sizeof(operand) elements,
 *
 *     nz(r, c) = (c / C0) * rows * C0 + r * C0 + (c % C0)
 *
 * and then a plain contiguous DataCopy moves UB -> L1.  NzOffset() below is
 * that formula, and the host table builder is its only caller.
 *
 * The periodic reciprocal tables survive this untouched, which is not luck but
 * is worth stating: the digit index within a packed byte is c mod dpb, and
 * every dpb here (2, 4, 8) divides C0 = 32, so c mod dpb == nz mod dpb and the
 * period-8 blocks are still correct against the permuted positions.
 *
 * THE ACCUMULATOR STAYS IN UB, NOT IN L0C.  Flash decoding rescales the running
 * accumulator by exp(m_old - m_new) at every tile, and that multiply has no
 * expression on the Cube; keeping the accumulator in L0C would mean reading it
 * out, rescaling and writing it back every tile anyway.  So each tile's PV
 * product is Fixpipe'd to UB and the AIV does the rescale-and-add it already
 * knows how to do.  What the Cube takes over is both O(S*D) products, which is
 * the whole of the arithmetic; what stays on the AIV is O(S) softmax and O(D)
 * accumulator maintenance.
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

/*
 * The operand type for a mode, as a C++ type.
 *
 * fp4x2_e2m1_t names a *pair* of fp4 values, so a LocalTensor of it is half as
 * long as the vector it holds.  Everything that sizes an fp4 buffer therefore
 * has to go through TurboQuantModeCodec::OperandElems rather than through the
 * head size, and mixing the two up produces a buffer that is exactly twice as
 * large as it needs to be with no error anywhere.
 */
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
 * The Cube stage for one mode.
 *
 * Owns L1, L0A, L0B and L0C.  It does NOT own the UB side: the AIV's unpack
 * buffers belong to the codec, and the two halves meet at StageA/StageB (UB ->
 * L1, an AIV-side MTE3) and at the Fixpipe (L0C -> UB, an AIC-side write).
 *
 *   maxM   rows of the A operand, i.e. the GQA group padded to kCubeTileM
 *   maxN   columns of the B operand; kTileRows for the score GEMM and head_size
 *          for the context GEMM, so this is sized off the larger
 *   maxK    the reduction length; head_size for scores, kTileRows for context
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

        // Two A1 buffers -- the query, staged once per task, and the
        // probability row, staged once per tile -- and one B1 that holds K and
        // then V within a tile.  No double buffering: it would cost another
        // bBytes of L1 and there is no measurement yet saying the stage is
        // where the time goes.
        const uint32_t qBytes = kCubeTileM * OperandElems(headSize_);
        const uint32_t pBytes = kCubeTileM * OperandElems(tileRows_);
        const uint32_t bBytes = tileRows_ * OperandElems(headSize_);
        pipe->InitBuffer(aQ1_, qBytes);
        pipe->InitBuffer(aP1_, pBytes);
        pipe->InitBuffer(b1_, bBytes);
        pipe->InitBuffer(a2_, qBytes > pBytes ? qBytes : pBytes);
        // L0B holds the [k, n] form's chunks end to end: ceil(k/16)/2 of them,
        // each ceil(n/16) * 16 * 32 elements. For the shapes here that is the
        // same bBytes the tile occupies, but it is written out rather than
        // assumed because the chunking rounds k up to a multiple of 32.
        pipe->InitBuffer(b2_, bBytes);
        // L0C carries [kCubeTileM, headSize] fp32, which also covers
        // [kCubeTileM, tileRows] since tileRows <= headSize on every shape here.
        pipe->InitBuffer(co1_, kCubeTileM * headSize_ * sizeof(float));
    }

    /*
     * AIV side: the three L1 landing buffers, handed out so the caller can
     * place into them with whatever DataCopy stride it needs.
     *
     * They are exposed rather than wrapped because the placement is not one
     * shape.  The query is written a head at a time (eight 32-byte runs, stride
     * headSize*... -- see the NZ formula below), the probability tile arrives
     * dense because its width is exactly one C0 block, and the K/V tile arrives
     * one unpack sub-batch at a time.  A Stage() that tried to cover all three
     * would take the DataCopyParams as an argument anyway.
     *
     * Everything written here must already be in NZ order; see the file header
     * for why that is the unpack's job and costs nothing.  The writes are MTE3
     * from the vector core's point of view, which is why the flag that follows
     * them is set on PIPE_MTE3.
     *
     * A1 is two buffers, not one: the score GEMM's A operand is the query and
     * the context GEMM's is the probability row, and they are live at the same
     * time -- the query is staged once per task and the probabilities once per
     * tile, so sharing one buffer would mean re-staging the query every tile.
     */
    __aicore__ inline AscendC::LocalTensor<OperandT> A1Query() { return aQ1_.template Get<OperandT>(); }
    __aicore__ inline AscendC::LocalTensor<OperandT> A1Probs() { return aP1_.template Get<OperandT>(); }
    __aicore__ inline AscendC::LocalTensor<OperandT> B1() { return b1_.template Get<OperandT>(); }

    /*
     * The NZ position of logical coordinate (r, c) in a `rows` x `cols` tile.
     *
     * Transcribed from TransND2NZ in
     * .../impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h, which is the
     * only statement of this layout the toolkit makes for arch35.  The host
     * mirror is turboquant_host::NzOffset; a test pins the two against each
     * other, because getting it wrong produces a wrong answer and not a fault.
     */
    __aicore__ static constexpr uint32_t NzOffset(uint32_t r, uint32_t c, uint32_t rows)
    {
        return (c / kOperandC0) * rows * kOperandC0 + r * kOperandC0 + (c % kOperandC0);
    }

    // Elements of OperandT in one C0 block: 32 bytes over the element size, so
    // 32 for fp8 and 32 fp4-pairs (64 fp4 coordinates) for fp4.
    static constexpr uint32_t kOperandC0 = 32;

    /*
     * AIC side, GEMM 1: scores[m, n] = Q[m, k] . K[n, k]^T, fp32 accumulate.
     *
     * K is staged as [n, k] -- [rows, head_size], the orientation the cache
     * already has -- which reads as the cheap form for an 8-bit B operand: one
     * LoadData with ifTranspose false. See LoadBFromNk.
     *
     * NOT YET VERIFIED, AND THE ONLY PART OF THIS FILE THAT IS NOT.
     * test_sim_950pr_cube_gemm pins both B forms against a host product on the
     * camodel. The [k, n] form (GemmContext) reproduces it *exactly* -- max|err|
     * 0 over a 16x64x256 product of exact fp8 values -- which means the NZ
     * layout, Mmad's argument order, Fixpipe's parameters and the MTE1 -> M ->
     * FIX synchronisation are all right. This one does not: at m=16 k=256 n=64
     * it leaves most of the output zero. Since everything the two share is
     * proven by the case that passes, what is wrong is confined to LoadBFromNk.
     *
     * The fix is known and is not a parameter tweak: route this GEMM through
     * the proven [k, n] form by staging K transposed, K^T [head_size, rows].
     * That costs nothing at run time -- the unpack's last step is a Gather and
     * its offset table is arbitrary, so emitting the transposed NZ order is a
     * different host-built constant and not an instruction -- but it needs the
     * unpack re-tiled from (8 rows x head_size) to (32 rows x 64 channels) so
     * that each chunk lands as one contiguous run in the transposed image, and
     * one offset table per channel chunk. Until that lands, the kv5fp8 decode
     * computes a wrong score row and the multi-mode smoke test says so.
     */
    __aicore__ inline void GemmScores(const AscendC::LocalTensor<float> &dstUb, uint32_t m, uint32_t k, uint32_t n)
    {
        aActive_ = aQ1_.template Get<OperandT>();
        LoadA(m, k);
        LoadBFromNk(k, n);
        Compute(dstUb, m, k, n);
    }

    /*
     * AIC side, GEMM 2: ctx[m, n] = P[m, k] . V[k, n], fp32 accumulate.
     *
     * V is staged as [k, n] -- [rows, head_size] again, the same orientation --
     * which for the B operand is the form that needs ifTranspose true and, for
     * an 8-bit type, a loop. See LoadBFromKn.
     */
    __aicore__ inline void GemmContext(const AscendC::LocalTensor<float> &dstUb, uint32_t m, uint32_t k, uint32_t n)
    {
        aActive_ = aP1_.template Get<OperandT>();
        LoadA(m, k);
        LoadBFromKn(k, n);
        Compute(dstUb, m, k, n);
    }

private:
    /*
     * THE FRACTAL CONTRACT, AND WHERE IT COMES FROM.
     *
     * LoadData2DParamsV2's fields are passed straight through to
     * load_cbuf_to_ca / load_cbuf_to_cb with no documentation, and the toolkit's
     * own matmul is the only statement of what they mean.  Everything below is
     * transcribed from
     *
     *   .../impl/adv_api/detail/matmul/stage/split/load_to_l0a/load_to_l0a_load2dV2.h
     *   .../impl/adv_api/detail/matmul/stage/split/load_to_l0b/load_to_l0b_load2dV2.h
     *
     * and is pinned numerically by csrc/tests/sim/test_sim_950pr_cube_gemm.cpp.
     * Three of the four facts are ones a reasonable reading gets wrong:
     *
     *   * mStep counts 16-row fractals; kStep counts C0 elements, and C0 is 32
     *     for every 8-bit type (AuxGetC0Size), not 64.
     *   * srcStride and dstStride are NOT optional.  srcStride is the L1
     *     matrix's leading extent over 16 and dstStride is the L0 fragment's
     *     over 16; passing zero for either silently reads the wrong fractals.
     *   * ifTranspose on the B operand means the OPPOSITE of what it reads as.
     *     B staged [n, k] loads with ifTranspose FALSE; B staged [k, n] -- the
     *     orientation that looks untransposed -- loads with ifTranspose TRUE.
     *     The toolkit's own names say so the other way round (its
     *     TransLoadDataToL0 sets ifTranspose = false), which is why this is
     *     written out rather than inferred.
     *
     * And one that is not a reading at all: an 8-bit B operand in the [k, n]
     * orientation must be loaded in mStep = 2 chunks, because the instruction
     * only accepts an even mStep for .b8.  A single call with mStep = 4 raises
     * mte_instr_addr_misalign -- measured on the camodel, and the reason the
     * first version of this file faulted.
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

    // B is [k, n] in L1. ifTranspose true, and for an 8-bit operand the load
    // runs in mStep = 2 chunks whose L0B destinations are
    // ceil(n / 16) * 32 elements apart.
    __aicore__ inline void LoadBFromKn(uint32_t k, uint32_t n)
    {
        AscendC::LocalTensor<OperandT> tb1 = b1_.template Get<OperandT>();
        AscendC::LocalTensor<OperandT> tb2 = b2_.template Get<OperandT>();
        AscendC::LoadData2DParamsV2 p;
        p.kStartPosition = 0;
        p.kStep = CeilDivU16(n, kC0);
        p.srcStride = CeilDivU16(k, kFractalRows);
        p.dstStride = CeilDivU16(n, kFractalRows);
        p.ifTranspose = true;

        const uint16_t mSteps = CeilDivU16(k, kFractalRows);
        const uint16_t loops = CeilDivU16(mSteps, kB8MStep);
        const uint32_t dstStrideElems =
            static_cast<uint32_t>(CeilDivU16(n, kFractalRows)) * kFractalRows * kC0;
        p.mStep = kB8MStep;
        uint32_t dstOffset = 0;
        for (uint16_t i = 0; i < loops; ++i) {
            p.mStartPosition = static_cast<uint32_t>(kB8MStep) * i;
            AscendC::LoadData(tb2[dstOffset], tb1, p);
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

        // Cross-pipe synchronisation, and it is not optional.
        //
        // PipeBarrier<PIPE_MTE1> orders MTE1 against MTE1 and says nothing
        // about the M pipe, so without these the Mmad issues while LoadData is
        // still filling L0A/L0B and reads whatever was there -- which on a
        // freshly allocated buffer is zeros, so the symptom is an output that
        // is partly or entirely zero rather than a fault. Same for M -> FIX:
        // the Fixpipe would copy an L0C the Mmad has not finished writing.
        // CANN's own matmul does exactly this with FetchEventID; see
        // .../detail/matmul/matmul_impl.h.
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

        // dstStride is the UB row pitch in elements, which is n because the AIV
        // reads the result as a dense [m, n] fp32 tile.
        //
        // srcStride is m rounded UP to a multiple of 16 -- not m over 16, which
        // is the convention LoadData uses two lines earlier and the one this
        // read as at first. CANN's own matmul sets
        // params_.srcStride = CeilAlign(mSize, BLOCK_CUBE)
        // (copy_cube_out_utils.h). With m/16 the copy reads a sixteenth of the
        // L0C and the rest of the destination stays whatever it was.
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
