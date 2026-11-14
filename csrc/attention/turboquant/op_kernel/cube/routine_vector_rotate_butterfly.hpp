#pragma once

// The shared radix-2 randomised-Hadamard butterfly leaf (roadmap kernel No.6).
// Ascend 950PR / arch35 / dav-3510 / CANN 9.2.
//
// What it computes
// ----------------
//   Pi = (H8 (x) I32) . diag(piSigns) . (I8 (x) H32) / 16 = A . D . B
//
// with A the channel-group factor (strides 32, 64, 128), D the sign diagonal
// and B the in-group factor (strides 1, 2, 4, 8, 16). A radix-2 stage at
// stride s is
//
//   out[i] = in[i ^ s] + sgn_s(i) . in[i],      sgn_s(i) = (i & s) ? -1 : +1.
//
// The two directions are NOT the same stage order, and that is the single most
// error-prone line in this family (roadmap section 1.0):
//
//   x . Pi  (egress of the decode operators)            = ((x.A).D).B
//           strides 32, 64, 128, 1, 2, 4, 8, 16   sign after stage 3
//   Pi . x  (ingress of turboquant_reshape_scatter_aiv) = A.(D.(B.x))
//           strides 1, 2, 4, 8, 16, 32, 64, 128   sign after stage 5
//
// Reusing the egress table for the ingress is a silent wrong answer: both
// orders are orthogonal maps, so the result is plausible and never faults.
//
// The 1/16 is NOT applied unless the caller asks. The decode egress passes
// kRotOrthoScale; the cache writer passes 1.0f and folds invSqrtLen into the
// stored quantisation scale instead -- scaling Y and absmax together leaves
// every nibble unchanged, which is what keeps the quantiser bit-exact against
// an integer oracle (RFC section 4.0 H-A).
//
// Three implementations, and why this header carries all three
// ------------------------------------------------------------
// This leaf is the whole arithmetic budget of an AIV-only ingress kernel, so
// which permutation strategy the eight stages use is a measurement, not a
// preference. The three are selected by `RotateImpl` and are bit-identical by
// construction -- same stage order, same associativity, FP32 throughout:
//
//   Blocked     Approach A. Level-2 continuous APIs over UNIFIED BUFFER
//               operands. Stages with s >= 8 are two repeat-form instructions
//               with `BinaryRepeatParams` strides; stages with s in {1, 2, 4}
//               have the partner lane inside the 8-lane FP32 DataBlock, where
//               a 4/8/16-byte tensor offset is not 32-byte aligned and the
//               masked form cannot reach it either, so they pay a `Gather`
//               against a 256-entry UB offset table plus a sign multiply.
//               Three offset and three sign vectors, 6 KiB. This is the shape
//               routine_vector_egress_rotate.hpp ships today, generalised to
//               both directions. Every stage writes its result back to UB.
//
//   RegGather   Approach B, register-indexed shuffle. One `__simd_vf__` body:
//               the 256 FP32 channels are four 64-lane vector registers and
//               stay there for all eight stages. Strides 1..32 flip a LANE
//               bit, so they are one `Reg::Gather` (dst[lane] =
//               src[idx[lane]], a register-to-register lane permutation --
//               the shfl_xor-equivalent on this ISA) plus one
//               `Reg::MulAddDst`. Strides 64 and 128 flip a REGISTER index
//               bit, so they are straight-line inter-register `Add`/`Sub`
//               with no permutation at all. The index and sign registers are
//               built in-register from `Arange`, `Xor` and a compare-select,
//               so no UB offset table exists.
//
//   RegShuffle  Approach B, isomorphic shuffle. Same register residency, but
//               the lane-bit stages use `DeInterleave` + `Select` and carry NO
//               index registers whatsoever. See "The shuffle recursion".
//
// The zero-spill invariant applies to RegGather and RegShuffle only. Blocked
// round-trips every stage through Unified Buffer by construction; it is here
// as the measured baseline the other two have to beat, and the production
// scatter kernel must not select it.
//
// The shuffle recursion
// ---------------------
// Inside one 64-lane FP32 register, let P be the permutation (P x)[k] =
// x[rotl6(k)] -- a cyclic left-rotation of the six lane-index bits -- and let
// B_t be the butterfly on lane bit t. Then
//
//   DeInterleave(E, O, V, V);  Add(S, E, O);  Sub(D, E, O);
//   Select(V, S, D, maskH)                              == P . B_0
//
// because DeInterleave puts the even lanes in E and the odd lanes in O (each
// duplicated across both halves of its destination), so S and D hold the two
// butterfly wings duplicated, and `maskH` (MaskPattern::H, the low half)
// splices the sums into lanes 0..31 and the differences into lanes 32..63 --
// which is exactly the decimated order. Six such steps give
//
//   (P B_0)^6 = P^6 . B_5 B_4 B_3 B_2 B_1 B_0 = B_5 ... B_0
//
// since P^6 = I and conjugating B_0 by P^t yields B_t. So six uniform
// four-instruction steps compute the whole 64-point Hadamard with one static
// mask, no index table, no sign vector and no Unified Buffer traffic. The
// identity is not taken on faith: tests/cpu/test_rotate_butterfly_probe.cpp
// runs this exact recursion against the dense 256 x 256 matrix.
//
// The cost is that the sign diagonal does not sit at a step boundary. After t
// steps the data is in P^t order, and the ingress needs D between stride 16
// (bit 4) and stride 32 (bit 5), i.e. at t = 5. The leaf therefore pre-permutes
// the SIGN registers by P^5 once per call -- five applications of the
// permutation-only form (`DeInterleave` + `Select`, two instructions) -- and
// multiplies in the permuted frame. Hoisted out of the row loop that is 40
// instructions per call against four per row.
//
// RegShuffle implements the INGRESS direction only. The egress order puts bit
// 5 before the sign and bits 0..4 after it, which an increasing-order
// recursion cannot express without pre-rotating the DATA by P^5 on every row
// (ten instructions per register per row) -- not competitive, and the egress
// already has a verified Blocked implementation. Asking for Egress with
// RegShuffle is a compile error rather than a silently wrong answer.
//
// Numerics. Every stage is an FP32 add or subtract of values the previous
// stage produced; there is no reciprocal, no exponential and no reduction
// reordering. For inputs whose rotated panel stays inside 2^24 -- which
// integer-valued BF16 activations at d = 256 do, since |y| <= 256 * 128 * 2 --
// every intermediate is exact and the three implementations are bit-identical
// to each other and to the integer host oracle. That is what makes the
// benchmark a fair comparison and the production gate a bit-exact one.

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "kernel_operator.h"
#endif

namespace ascend_micro {

// ---------------------------------------------------------------- geometry
constexpr std::uint32_t kRotChannels = 256;      // d, one KV head
constexpr std::uint32_t kRotVecLanes = 64;       // FP32 lanes in VECTOR_REG_WIDTH
constexpr std::uint32_t kRotRegisters = kRotChannels / kRotVecLanes;  // 4
constexpr std::uint32_t kRotLanesPerBlock = 8;   // FP32 per 32-byte DataBlock
constexpr std::uint32_t kRotStages = 8;
constexpr std::uint32_t kRotLaneStages = 6;  // strides 1..32 flip a lane bit
constexpr std::uint32_t kRotRegStages = 2;   // strides 64, 128 flip a register bit

// Pi's orthonormalisation, for the caller that wants it applied here.
constexpr float kRotOrthoScale = 1.0f / 16.0f;

enum class RotateDirection : unsigned {
    Egress = 0,   // x . Pi
    Ingress = 1,  // Pi . x
};

enum class RotateImpl : unsigned {
    Blocked = 0,     // Approach A: UB operands, BinaryRepeatParams + UB Gather
    RegGather = 1,   // Approach B: register-resident, Reg::Gather lane permute
    RegShuffle = 2,  // Approach B: register-resident, DeInterleave/Select only
};

// The stage tables. Mirrored exactly by `rotate_dense` and `rotate_staged` in
// tests/common/rotate_butterfly_oracle.hpp, which the Tier-1 probe checks
// against the dense 256 x 256 matrix in both directions.
constexpr std::uint32_t kRotStrideOf[2][kRotStages] = {
    {32, 64, 128, 1, 2, 4, 8, 16},  // Egress
    {1, 2, 4, 8, 16, 32, 64, 128},  // Ingress
};
// How many stages run BEFORE the sign diagonal.
constexpr std::uint32_t kRotSignAfterStage[2] = {3, 5};

// Approach A's sub-DataBlock stages, in the order each direction reaches them.
constexpr std::uint32_t kRotGatherStages = 3;
constexpr std::uint32_t kRotGatherStrideOf[kRotGatherStages] = {1, 2, 4};
// Bytes of Unified Buffer the Blocked tables cost, for the budget arithmetic
// in the kernels that choose between the implementations.
constexpr std::uint32_t kRotBlockedTableBytes =
    kRotGatherStages * kRotChannels * 4u * 2u;  // offsets + signs

constexpr unsigned RotDirIndex(RotateDirection d) { return static_cast<unsigned>(d); }

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)

// ======================================================================== A
// Approach A -- blocked SIMD over Unified Buffer operands.
//
// Compile-time shape of a block-aligned stage. For s >= 8 the partner lane
// i ^ s is a whole number of DataBlocks away, so the stage is two repeat-form
// instructions over contiguous runs with the second source offset by s
// elements (s * 4 >= 32 bytes, therefore aligned).
template <std::uint32_t S>
constexpr std::uint32_t kRotLanesPerRepeat = S < kRotVecLanes ? S : kRotVecLanes;
template <std::uint32_t S>
constexpr std::uint32_t kRotRepeats = (kRotChannels / 2u) / kRotLanesPerRepeat<S>;
template <std::uint32_t S>
constexpr std::uint32_t kRotRepStrideBlocks =
    S <= kRotVecLanes ? (2u * S / kRotLanesPerBlock) : (kRotVecLanes / kRotLanesPerBlock);

struct NativeRotateBlockedEngine {
    // ---- inputs ----------------------------------------------------------
    AscendC::LocalTensor<float> pi_sign;  // kRotChannels, +-1
    // Three offset vectors and three sign vectors, one pair per sub-DataBlock
    // stage, each kRotChannels entries, concatenated in stride order {1, 2, 4}.
    AscendC::LocalTensor<std::uint32_t> gather_offsets;
    AscendC::LocalTensor<float> gather_signs;

    // ---- scratch, all kRotChannels FP32 ----------------------------------
    AscendC::LocalTensor<float> buffer_a;  // the row arrives here
    AscendC::LocalTensor<float> buffer_b;  // nine writes later, the result
    AscendC::LocalTensor<float> permuted;
    AscendC::LocalTensor<float> signed_tmp;

    static constexpr std::int32_t kChannelLanes = static_cast<std::int32_t>(kRotChannels);

    // Within each 2S-lane group the lower S lanes get lo + hi and the upper S
    // lanes get lo - hi. `dst` must not alias `src`: the lower half's write
    // would otherwise clobber an operand of the upper half's.
    template <std::uint32_t S>
    __aicore__ inline void block_stage(const AscendC::LocalTensor<float>& dst,
                                       const AscendC::LocalTensor<float>& src) const
    {
        constexpr std::int32_t mask = static_cast<std::int32_t>(kRotLanesPerRepeat<S>);
        constexpr std::uint8_t repeats = static_cast<std::uint8_t>(kRotRepeats<S>);
        constexpr std::uint8_t stride = static_cast<std::uint8_t>(kRotRepStrideBlocks<S>);
        const AscendC::BinaryRepeatParams params(1, 1, 1, stride, stride, stride);
        AscendC::Add(dst, src, src[S], mask, repeats, params);
        AscendC::Sub(dst[S], src, src[S], mask, repeats, params);
    }

    // `Index` selects the stride's offset and sign vectors out of the
    // concatenated tables.
    template <std::uint32_t Index>
    __aicore__ inline void gather_stage(const AscendC::LocalTensor<float>& dst,
                                        const AscendC::LocalTensor<float>& src) const
    {
        constexpr std::uint32_t table = Index * kRotChannels;
        AscendC::Gather(permuted, src, gather_offsets[table], 0u, kRotChannels);
        AscendC::Mul(signed_tmp, src, gather_signs[table], kChannelLanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(dst, permuted, signed_tmp, kChannelLanes);
    }

    __aicore__ inline void sign_stage(const AscendC::LocalTensor<float>& dst,
                                      const AscendC::LocalTensor<float>& src) const
    {
        AscendC::Mul(dst, src, pi_sign, kChannelLanes);
    }

    // The whole rotation, ping-ponging between buffer_a and buffer_b. Eight
    // stages plus the sign is nine writes, an odd count, so the first reads
    // `src` directly and the last writes `dst` directly -- the ping-pong pair
    // never has to be primed by a copy, which would have made the Approach A
    // measurement two passes worse than the method deserves. `dst` must not
    // alias `src`: the final stage's Add writes the lower half of each group
    // while its Sub still reads it.
    template <RotateDirection D>
    __aicore__ inline void rotate(const AscendC::LocalTensor<float>& dst,
                                  const AscendC::LocalTensor<float>& src)
    {
        if constexpr (D == RotateDirection::Egress) {
            block_stage<32>(buffer_b, src);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<64>(buffer_a, buffer_b);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<128>(buffer_b, buffer_a);
            AscendC::PipeBarrier<PIPE_V>();
            sign_stage(buffer_a, buffer_b);
            AscendC::PipeBarrier<PIPE_V>();
            gather_stage<0>(buffer_b, buffer_a);  // stride 1
            AscendC::PipeBarrier<PIPE_V>();
            gather_stage<1>(buffer_a, buffer_b);  // stride 2
            AscendC::PipeBarrier<PIPE_V>();
            gather_stage<2>(buffer_b, buffer_a);  // stride 4
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<8>(buffer_a, buffer_b);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<16>(dst, buffer_a);
        } else {
            gather_stage<0>(buffer_b, src);       // stride 1
            AscendC::PipeBarrier<PIPE_V>();
            gather_stage<1>(buffer_a, buffer_b);  // stride 2
            AscendC::PipeBarrier<PIPE_V>();
            gather_stage<2>(buffer_b, buffer_a);  // stride 4
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<8>(buffer_a, buffer_b);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<16>(buffer_b, buffer_a);
            AscendC::PipeBarrier<PIPE_V>();
            sign_stage(buffer_a, buffer_b);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<32>(buffer_b, buffer_a);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<64>(buffer_a, buffer_b);
            AscendC::PipeBarrier<PIPE_V>();
            block_stage<128>(dst, buffer_a);
        }
    }

    // The 1/16, or 1.0f for the writer. A separate pass over Unified Buffer
    // because the Blocked form has nowhere to fold it: every stage's
    // destination is already distinct from its sources, and folding a multiply
    // into one of them would make that stage in-place. The register-resident
    // forms fold it into four `Muls` on live registers instead, and that
    // asymmetry is part of what the benchmark measures.
    __aicore__ inline void scale(const AscendC::LocalTensor<float>& dst, float factor)
    {
        AscendC::Muls(dst, dst, factor, kChannelLanes);
    }
};

// ======================================================================== B
// Approach B -- the register-resident bodies.
//
// Lane map: channel i lives in register i >> 6, lane i & 63. So strides 1..32
// flip a lane bit (an in-register permutation) and strides 64 and 128 flip a
// register-index bit (no permutation at all, only which register pairs with
// which).

struct RotRegSet {
    AscendC::Reg::RegTensor<float> v0;
    AscendC::Reg::RegTensor<float> v1;
    AscendC::Reg::RegTensor<float> v2;
    AscendC::Reg::RegTensor<float> v3;
};

// Stride 64: i ^ 64 flips register bit 0, so (v0, v1) and (v2, v3) pair up.
// Stride 128: i ^ 128 flips register bit 1, so (v0, v2) and (v1, v3) pair up.
// In both cases the register with the bit CLEAR takes the sum and the one with
// it SET takes the difference, which is sgn_s(i) = (i & s) ? -1 : +1.
__simd_callee__ inline void rot_reg_stage64(RotRegSet& v, AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<float> s0, d0, s1, d1;
    AscendC::Reg::Add<float>(s0, v.v0, v.v1, mask);
    AscendC::Reg::Sub<float>(d0, v.v0, v.v1, mask);
    AscendC::Reg::Add<float>(s1, v.v2, v.v3, mask);
    AscendC::Reg::Sub<float>(d1, v.v2, v.v3, mask);
    v.v0 = s0;
    v.v1 = d0;
    v.v2 = s1;
    v.v3 = d1;
}

__simd_callee__ inline void rot_reg_stage128(RotRegSet& v, AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<float> s0, d0, s1, d1;
    AscendC::Reg::Add<float>(s0, v.v0, v.v2, mask);
    AscendC::Reg::Sub<float>(d0, v.v0, v.v2, mask);
    AscendC::Reg::Add<float>(s1, v.v1, v.v3, mask);
    AscendC::Reg::Sub<float>(d1, v.v1, v.v3, mask);
    v.v0 = s0;
    v.v2 = d0;
    v.v1 = s1;
    v.v3 = d1;
}

// ---------------------------------------------------------------- B1 tables
// The six lane-bit stages' index and sign registers, built once per VF call
// and never read from Unified Buffer. Named individually rather than held in
// an array: an array of RegTensor indexed by a loop variable is what gives the
// compiler a reason to put the set in memory, which is the one thing this
// implementation exists to avoid.
struct RotGatherTables {
    AscendC::Reg::RegTensor<std::uint32_t> i0, i1, i2, i3, i4, i5;
    // The sign registers are held as the FP32 BIT PATTERNS of +-1.0f and read
    // back through a reference reinterpret, because that is how they are
    // built: see rot_one_table.
    AscendC::Reg::RegTensor<std::uint32_t> s0, s1, s2, s3, s4, s5;
};

// idx[lane] = lane ^ s and sgn[lane] = (lane & s) ? -1.0f : +1.0f, for
// s = 1 << k. `Reg::Gather` wraps an index modulo VL / sizeof(T) = 64, and
// lane ^ s with s < 64 is already in range, so no clamp is needed.
//
// The sign vector is built by bit arithmetic rather than by a compare and a
// select, for a reason worth recording: +1.0f is 0x3F800000 and -1.0f is
// 0xBF800000, so the whole vector is
//
//   ((lane & s) << (31 - k)) | 0x3F800000
//
// -- the stride's own bit, moved to the FP32 sign position. Three instructions
// with no comparison, no mask register and no conversion. The alternative
// needed `Compares` plus `Select` and two more live constant registers.
__simd_callee__ inline void rot_one_table(AscendC::Reg::RegTensor<std::uint32_t>& idx,
                                          AscendC::Reg::RegTensor<std::uint32_t>& sgn_bits,
                                          AscendC::Reg::RegTensor<std::uint32_t>& lane,
                                          AscendC::Reg::RegTensor<std::uint32_t>& one_bits,
                                          std::uint32_t k, AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<std::uint32_t> stride, bit, shifted;
    AscendC::Reg::Duplicate<std::uint32_t>(stride, 1u << k);
    AscendC::Reg::Xor<std::uint32_t>(idx, lane, stride, mask);
    AscendC::Reg::And<std::uint32_t>(bit, lane, stride, mask);
    // `ShiftLefts` takes its shift count as an int16_t on this part and
    // nothing else -- the static_assert is `SupportType<U, int16_t>`.
    AscendC::Reg::ShiftLefts<std::uint32_t>(shifted, bit,
                                            static_cast<std::int16_t>(31u - k), mask);
    AscendC::Reg::Or<std::uint32_t>(sgn_bits, shifted, one_bits, mask);
}

__simd_callee__ inline void rot_build_gather_tables(RotGatherTables& t,
                                                    AscendC::Reg::MaskReg& mask)
{
    // `Arange` has no unsigned form on dav-3510 -- its `vci` intrinsic is
    // declared for s8/s16/s32/f16/f32 only -- while `Reg::Gather` requires an
    // index register whose element type is uint8_t, uint16_t or uint32_t of
    // the same width as the data. A reference reinterpret between two
    // RegTensors of one register width is the vendor's own idiom for exactly
    // this: kernel_reg_compute_vec_vconv_impl.h writes
    // `(RegTensor<uint16_t>&)srcReg` in the bf16 -> fp8e8m0 path. The ramp
    // 0..63 reads identically either way.
    AscendC::Reg::RegTensor<std::int32_t> ramp;
    AscendC::Reg::Arange<std::int32_t>(ramp, 0);
    auto& lane = reinterpret_cast<AscendC::Reg::RegTensor<std::uint32_t>&>(ramp);
    AscendC::Reg::RegTensor<std::uint32_t> one_bits;
    AscendC::Reg::Duplicate<std::uint32_t>(one_bits, 0x3F800000u);  // +1.0f
    rot_one_table(t.i0, t.s0, lane, one_bits, 0u, mask);   // stride 1
    rot_one_table(t.i1, t.s1, lane, one_bits, 1u, mask);   // stride 2
    rot_one_table(t.i2, t.s2, lane, one_bits, 2u, mask);   // stride 4
    rot_one_table(t.i3, t.s3, lane, one_bits, 3u, mask);   // stride 8
    rot_one_table(t.i4, t.s4, lane, one_bits, 4u, mask);   // stride 16
    rot_one_table(t.i5, t.s5, lane, one_bits, 5u, mask);   // stride 32
}

// One lane-bit stage on one register: gather the partner lane, then fold the
// signed self term in with a single fused multiply-add. Two instructions.
__simd_callee__ inline void rot_gather_one(AscendC::Reg::RegTensor<float>& v,
                                           AscendC::Reg::RegTensor<std::uint32_t>& idx,
                                           AscendC::Reg::RegTensor<std::uint32_t>& sgn_bits,
                                           AscendC::Reg::MaskReg& mask)
{
    auto& sgn = reinterpret_cast<AscendC::Reg::RegTensor<float>&>(sgn_bits);
    AscendC::Reg::RegTensor<float> partner;
    AscendC::Reg::Gather<float, std::uint32_t>(partner, v, idx);
    AscendC::Reg::MulAddDst<float>(partner, v, sgn, mask);
    v = partner;
}

// One lane-bit stage across the whole 256-channel row.
__simd_callee__ inline void rot_gather_stage(RotRegSet& v,
                                             AscendC::Reg::RegTensor<std::uint32_t>& idx,
                                             AscendC::Reg::RegTensor<std::uint32_t>& sgn,
                                             AscendC::Reg::MaskReg& mask)
{
    rot_gather_one(v.v0, idx, sgn, mask);
    rot_gather_one(v.v1, idx, sgn, mask);
    rot_gather_one(v.v2, idx, sgn, mask);
    rot_gather_one(v.v3, idx, sgn, mask);
}

// --------------------------------------------------------------- B2 shuffle
// P . B_0: one butterfly on lane bit 0 followed by the bit rotation. Four
// instructions, one static mask, no table.
__simd_callee__ inline void rot_shuffle_one(AscendC::Reg::RegTensor<float>& v,
                                            AscendC::Reg::MaskReg& mask,
                                            AscendC::Reg::MaskReg& half)
{
    AscendC::Reg::RegTensor<float> even, odd, sum, diff;
    AscendC::Reg::DeInterleave<float>(even, odd, v, v);
    AscendC::Reg::Add<float>(sum, even, odd, mask);
    AscendC::Reg::Sub<float>(diff, even, odd, mask);
    AscendC::Reg::Select<float>(v, sum, diff, half);
}

__simd_callee__ inline void rot_shuffle_stage(RotRegSet& v, AscendC::Reg::MaskReg& mask,
                                              AscendC::Reg::MaskReg& half)
{
    rot_shuffle_one(v.v0, mask, half);
    rot_shuffle_one(v.v1, mask, half);
    rot_shuffle_one(v.v2, mask, half);
    rot_shuffle_one(v.v3, mask, half);
}

// P alone, no butterfly: the same splice with the wings replaced by the two
// halves themselves. Two instructions. Used to carry the sign vector into the
// permuted frame the recursion leaves the data in.
__simd_callee__ inline void rot_shuffle_permute(AscendC::Reg::RegTensor<float>& v,
                                                AscendC::Reg::MaskReg& half)
{
    AscendC::Reg::RegTensor<float> even, odd;
    AscendC::Reg::DeInterleave<float>(even, odd, v, v);
    AscendC::Reg::Select<float>(v, even, odd, half);
}

// --------------------------------------------------------- shared row helpers
__simd_callee__ inline void rot_load_row(RotRegSet& v, __ubuf__ float* in)
{
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(v.v0, in);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(v.v1, in + kRotVecLanes);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(v.v2,
                                                                      in + 2u * kRotVecLanes);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(v.v3,
                                                                      in + 3u * kRotVecLanes);
}

__simd_callee__ inline void rot_store_row(__ubuf__ float* out, RotRegSet& v,
                                          AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(out, v.v0, mask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(out + kRotVecLanes,
                                                                            v.v1, mask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
        out + 2u * kRotVecLanes, v.v2, mask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
        out + 3u * kRotVecLanes, v.v3, mask);
}

__simd_callee__ inline void rot_scale_row(RotRegSet& v, float factor,
                                          AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::Muls<float, float>(v.v0, v.v0, factor, mask);
    AscendC::Reg::Muls<float, float>(v.v1, v.v1, factor, mask);
    AscendC::Reg::Muls<float, float>(v.v2, v.v2, factor, mask);
    AscendC::Reg::Muls<float, float>(v.v3, v.v3, factor, mask);
}

struct RotSignRegs {
    AscendC::Reg::RegTensor<float> p0, p1, p2, p3;
};

__simd_callee__ inline void rot_load_signs(RotSignRegs& p, __ubuf__ float* sign)
{
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(p.p0, sign);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(p.p1,
                                                                      sign + kRotVecLanes);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
        p.p2, sign + 2u * kRotVecLanes);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
        p.p3, sign + 3u * kRotVecLanes);
}

__simd_callee__ inline void rot_apply_signs(RotRegSet& v, RotSignRegs& p,
                                            AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::Mul<float>(v.v0, v.v0, p.p0, mask);
    AscendC::Reg::Mul<float>(v.v1, v.v1, p.p1, mask);
    AscendC::Reg::Mul<float>(v.v2, v.v2, p.p2, mask);
    AscendC::Reg::Mul<float>(v.v3, v.v3, p.p3, mask);
}

// ---------------------------------------------------- the per-row sequences
//
// These are the leaf proper: eight stages on one row that is already in
// registers, and nothing else. They are `__simd_callee__` rather than
// `__simd_vf__` so a caller with more to do per row -- the scatter ingress
// operator quantises in the same VF entry -- can keep the row resident across
// the rotation AND its own epilogue instead of handing it back through Unified
// Buffer in between. The VF wrappers below are the benchmark's callers and
// nothing more.

template <RotateDirection D>
__simd_callee__ inline void rot_row_gather(RotRegSet& v, RotGatherTables& t, RotSignRegs& p,
                                           AscendC::Reg::MaskReg& mask)
{
    if constexpr (D == RotateDirection::Egress) {
        rot_gather_stage(v, t.i5, t.s5, mask);  // 32
        rot_reg_stage64(v, mask);               // 64
        rot_reg_stage128(v, mask);              // 128
        rot_apply_signs(v, p, mask);
        rot_gather_stage(v, t.i0, t.s0, mask);  // 1
        rot_gather_stage(v, t.i1, t.s1, mask);  // 2
        rot_gather_stage(v, t.i2, t.s2, mask);  // 4
        rot_gather_stage(v, t.i3, t.s3, mask);  // 8
        rot_gather_stage(v, t.i4, t.s4, mask);  // 16
    } else {
        rot_gather_stage(v, t.i0, t.s0, mask);  // 1
        rot_gather_stage(v, t.i1, t.s1, mask);  // 2
        rot_gather_stage(v, t.i2, t.s2, mask);  // 4
        rot_gather_stage(v, t.i3, t.s3, mask);  // 8
        rot_gather_stage(v, t.i4, t.s4, mask);  // 16
        rot_apply_signs(v, p, mask);
        rot_gather_stage(v, t.i5, t.s5, mask);  // 32
        rot_reg_stage64(v, mask);               // 64
        rot_reg_stage128(v, mask);              // 128
    }
}

// Ingress only -- see the header comment. `p` must already be in the P^5
// frame, which `rot_permute_signs_p5` puts it in once per VF entry.
__simd_callee__ inline void rot_row_shuffle(RotRegSet& v, RotSignRegs& p,
                                            AscendC::Reg::MaskReg& mask,
                                            AscendC::Reg::MaskReg& half)
{
    // Five steps: lane bits 0..4, i.e. strides 1, 2, 4, 8, 16.
    rot_shuffle_stage(v, mask, half);
    rot_shuffle_stage(v, mask, half);
    rot_shuffle_stage(v, mask, half);
    rot_shuffle_stage(v, mask, half);
    rot_shuffle_stage(v, mask, half);
    rot_apply_signs(v, p, mask);
    // The sixth step: lane bit 5 (stride 32). P^6 = I returns the data to
    // natural order, which is what makes the two register stages below read
    // the channels they think they are reading.
    rot_shuffle_stage(v, mask, half);
    rot_reg_stage64(v, mask);
    rot_reg_stage128(v, mask);
}

__simd_callee__ inline void rot_permute_signs_p5(RotSignRegs& p, AscendC::Reg::MaskReg& half)
{
    for (std::uint32_t k = 0; k < 5u; ++k) {
        rot_shuffle_permute(p.p0, half);
        rot_shuffle_permute(p.p1, half);
        rot_shuffle_permute(p.p2, half);
        rot_shuffle_permute(p.p3, half);
    }
}

// ------------------------------------------------------------- the VF bodies
//
// Both bodies transform `rows` consecutive kRotChannels-lane FP32 rows from
// `src` to `dst`, which may be the same address: all four registers of a row
// are loaded before any store. `sign` is the natural-order piSigns vector and
// is read once per call, not once per row.

template <RotateDirection D>
__simd_vf__ inline void rotate_butterfly_gather_vf(__ubuf__ float* dst, __ubuf__ float* src,
                                                   __ubuf__ float* sign, float scale,
                                                   std::uint16_t rows)
{
    AscendC::Reg::MaskReg mask =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    RotGatherTables t;
    rot_build_gather_tables(t, mask);
    RotSignRegs p;
    rot_load_signs(p, sign);

    for (std::uint16_t r = 0; r < rows; ++r) {
        __ubuf__ float* in = src + static_cast<std::uint32_t>(r) * kRotChannels;
        __ubuf__ float* out = dst + static_cast<std::uint32_t>(r) * kRotChannels;
        RotRegSet v;
        rot_load_row(v, in);
        rot_row_gather<D>(v, t, p, mask);
        rot_scale_row(v, scale, mask);
        rot_store_row(out, v, mask);
    }
}

// Ingress only -- see the header comment. The sign registers are carried into
// the P^5 frame once, outside the row loop.
__simd_vf__ inline void rotate_butterfly_shuffle_vf(__ubuf__ float* dst, __ubuf__ float* src,
                                                    __ubuf__ float* sign, float scale,
                                                    std::uint16_t rows)
{
    AscendC::Reg::MaskReg mask =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg half = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::H>();
    RotSignRegs p;
    rot_load_signs(p, sign);
    rot_permute_signs_p5(p, half);

    for (std::uint16_t r = 0; r < rows; ++r) {
        __ubuf__ float* in = src + static_cast<std::uint32_t>(r) * kRotChannels;
        __ubuf__ float* out = dst + static_cast<std::uint32_t>(r) * kRotChannels;
        RotRegSet v;
        rot_load_row(v, in);
        rot_row_shuffle(v, p, mask, half);
        rot_scale_row(v, scale, mask);
        rot_store_row(out, v, mask);
    }
}

#endif  // native

}  // namespace ascend_micro
