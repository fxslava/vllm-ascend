#include <cstdint>

// turboquant_reshape_scatter_aiv -- sub-strip BF16 KV ingestion, AIV only.
// Roadmap kernel No.1. Ascend 950PR / arch35 / dav-3510 / CANN 9.2,
// KERNEL_TYPE_AIV_ONLY.
//
// What it is for
// --------------
// Every decode step writes B x (1 + k) tokens into B different pages at
// arbitrary, non-consecutive slots, and every request's prefill tail writes
// n < 16 tokens. The aligned prefill writer cannot serve either: its contract
// is num_tokens % 16 == 0 with 16 consecutive slots, and a decode step has no
// slot run to offer. This kernel serves 1..16 tokens per work item with
// arbitrary non-negative slots.
//
// Unit of work: (request, kv_head)
// --------------------------------
// One KV head of one request, d = 256, owned exclusively by one physical AIV.
// No intra-head splitting: the head is the butterfly's operand and the
// butterfly does not decompose across cores.
//
// (request, kv_head) rather than (token, kv_head) because all 1 + k tokens of
// one request share a strip in the MTP case, so folding them into one work
// item makes the value-plane read-modify-write a single transaction and
// removes the "no two work items collide on a strip" assertion that the
// per-token form would need and that MTP would violate.
//
// Why Vector and not Cube
// -----------------------
// This is the one place the Cube-centric verdict inverts. At M = 1 the Cube
// prologue would stage 32 KiB of rotation tables to transform 512 bytes of
// payload, and GEMV.md records that an M = 1 Mmad leaves only 32-64 bytes
// valid in each 1024-byte L0C sub-vector -- a 16:1 waste. So Pi . x runs as
// the Vector butterfly: eight radix-2 stages over 256 lanes with the sign
// diagonal between them.
//
// Which butterfly, and why that one
// ---------------------------------
// `RotateImpl::RegGather` from routines/routine_vector_rotate_butterfly.hpp,
// selected by measurement rather than preference. All three implementations
// are bit-identical; on the CAModel micro-benchmark over 16 rows
//
//   variant     vector instr   RVECEX   RV_VLD   RV_VST   vector span (ticks)
//   Blocked           11250      2512     3072     1728                 30215
//   RegShuffle         1981      1834       64       64                  1965
//   RegGather          1203      1057       64       64                  1850
//
// 4 loads and 4 stores per row and not one more, for both register-resident
// variants: that is the zero-UB-spill invariant as a measured number rather
// than an intention. Approach A spills 1664 extra registers over the same
// work and runs 16.3x longer.
//
// The layout contract (RFC section 3.1, normative)
// ------------------------------------------------
//   key   nibble = ((block*H + head)*S + token)*d + channel
//   value nibble = ((block*H + head)*S)*d
//                  + (token / R)*(R*d) + channel*R + (token % R)
//   scale index  = ((kv*P + block)*H + head)*S + token     kv 0 key, 1 value
//   nibbles       two per byte, low nibble first, signed INT4 in [-8, 7]
//
// with R = 16 the strip height. The two planes are written completely
// differently and the layout is why:
//
//   KEY needs no read-modify-write. It is token-major, so one token is 256
//   consecutive nibbles = 128 consecutive bytes at a 32-byte aligned offset
//   (token * d/2 = token * 128). A run of consecutive tokens is one burst of
//   n * 128 bytes; a single token is four 32-byte bursts.
//
//   VALUE is a read-modify-write of one 2048-byte strip. Channel-major inside
//   the strip puts one token's 256 nibbles at stride 16 -- 128 bytes spread
//   over 2048 at half-byte granularity -- and there is no aligned-burst
//   expression of that. So the strip is read in one 2048-byte burst (64
//   DataBlocks), the owned nibbles are spliced in by register bitwise masks,
//   and the whole strip goes back in one burst.
//
// The read-modify-write ALWAYS PRESERVES. Lanes the work item does not own go
// back unchanged. That is not defensive: in decode a strip accumulates over
// 16 successive steps, so zeroing the unowned lanes would destroy the previous
// fifteen tokens. It also makes the canary protocol stronger, because an
// untouched lane must still read back as canary.
//
// The absmax == 0 branch is mandatory here (RFC I15)
// --------------------------------------------------
// The scale is the power of two 2^(e-2) extracted by masking the exponent
// field of the FP32 absmax. For absmax == 0 that field is 0, so 2^(e-2) is
// built from a borrowed exponent and the stored scale is garbage. The codes
// are unaffected -- every y is zero, so every code is zero whatever the finite
// reciprocal is -- which is exactly why the case is silent without the branch.
// A zero K or V row never occurs with integer fixtures and occurs immediately
// in production, so the scale is selected to 1.0f when absmax == 0.
//
// Synchronisation: intra-core HardEvents only
// -------------------------------------------
// No Cube, no L1, no cross-core flag, no mode-4 id, no modulo ring. Straight
// line per work item, one buffer set, with the four WAR back edges a single
// buffer set needs:
//
//   MTE2_V    raw activation rows filled / strip filled
//   V_MTE2    Vector finished reading the raw rows -> MTE2 may refill
//   V_MTE3    packed key page / merged strip / scale slots ready
//   MTE3_V    MTE3 drained the packed page and the scale slots
//   MTE3_MTE2 MTE3 drained the strip -> MTE2 may read the next one
//
// MTE3_MTE2 is the one that is easy to get wrong: the strip is written by
// MTE2, mutated by Vector and read by MTE3, so the reuse edge runs from MTE3
// straight back to MTE2 and routing it through Vector would not bound the
// MTE2 queue at all.

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "vector/routine_vector_rotate_butterfly.hpp"

namespace turboquant_scatter {

using ascend_micro::kRotChannels;

// ---------------------------------------------------------------- geometry
constexpr std::uint32_t kHeadSize = kRotChannels;  // 256, compile-time
constexpr std::uint32_t kStripHeight = 16;         // R, the strip height
constexpr std::uint32_t kMaxRun = kStripHeight;    // tokens per RMW transaction
constexpr std::uint32_t kStripBytes = kStripHeight * kHeadSize / 2u;  // 2048
constexpr std::uint32_t kStripWords = kStripBytes / 4u;               // 512
constexpr std::uint32_t kKeyTokenBytes = kHeadSize / 2u;              // 128
constexpr std::uint32_t kVecLanes = ascend_micro::kRotVecLanes;  // 64

// ------------------------------------------------------- UB layout, AKA3003
constexpr std::uint32_t kBlock = 32;
constexpr std::uint32_t kBankSkew = 32;
constexpr std::uint32_t kVecReg = 256;  // VECTOR_REG_WIDTH on dav-3510
constexpr std::uint32_t kUbWindow = 192u * 1024u;

constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}
constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkew;
}

constexpr std::uint32_t kPanelElems = kMaxRun * kHeadSize;  // 4096
constexpr std::uint32_t kUbRaw = 0;                                        // bf16 in
constexpr std::uint32_t kUbAct = NextSlot(kUbRaw, kPanelElems * 2u);       // fp32
constexpr std::uint32_t kUbQ = NextSlot(kUbAct, kPanelElems * 4u);         // fp32 codes
constexpr std::uint32_t kUbQuant = NextSlot(kUbQ, kPanelElems * 4u);       // int32 codes
constexpr std::uint32_t kUbHalf = NextSlot(kUbQuant, kPanelElems * 4u);    // half staging
constexpr std::uint32_t kUbPacked = NextSlot(kUbHalf, kPanelElems * 2u);   // key nibbles
constexpr std::uint32_t kUbStrip = NextSlot(kUbPacked, kMaxRun * kKeyTokenBytes);
constexpr std::uint32_t kUbSign = NextSlot(kUbStrip, kStripBytes);
// One whole vector-register slot per row, lane 0 holding that row's scale. A
// VL1-masked store writes exactly four bytes, and the slot keeps every row's
// scale 32-byte aligned so MTE3 can read it without a gather.
constexpr std::uint32_t kUbScale = NextSlot(kUbSign, kHeadSize * 4u);
// The launch probe: four DataBlocks, one broadcast scalar each.
constexpr std::uint32_t kProbeLanes = 32;
constexpr std::uint32_t kUbProbe = NextSlot(kUbScale, kMaxRun * kVecLanes * 4u);
constexpr std::uint32_t kArenaEnd = kUbProbe + kProbeLanes * 4u;

static_assert(kUbRaw % kBlock == 0u && kUbAct % kBlock == 0u && kUbQ % kBlock == 0u &&
                  kUbQuant % kBlock == 0u && kUbHalf % kBlock == 0u &&
                  kUbPacked % kBlock == 0u && kUbStrip % kBlock == 0u &&
                  kUbSign % kBlock == 0u && kUbScale % kBlock == 0u &&
                  kUbProbe % kBlock == 0u,
              "every UB operand base must be 32-byte aligned");
// Rule A3: level-2 continuous ops read a whole 256-byte vector register from
// each source base, unmasked, so the arena needs that much pad behind it.
static_assert(kArenaEnd + kVecReg <= kUbWindow, "UB arena leaves no read pad");
// The register-resident bodies address the panels as raw __ubuf__ pointers, so
// every row base must be a whole number of vector registers from the panel
// base as well as DataBlock aligned.
static_assert(kHeadSize * 4u % kVecReg == 0u, "a 256-channel FP32 row is whole registers");
static_assert(kVecLanes * 4u == kVecReg, "64 FP32 lanes is one vector register");
// Every DMA length below is a whole number of 32-byte bursts.
static_assert(kStripBytes % kBlock == 0u && kKeyTokenBytes % kBlock == 0u &&
                  kHeadSize * 2u % kBlock == 0u,
              "DMA lengths must be whole 32-byte bursts");
// The strip is addressed as uint32 words during the merge, so its GM byte
// offset -- a multiple of kStripBytes -- must be 4-byte aligned.
static_assert(kStripBytes % 4u == 0u, "the strip must be addressable as uint32");

// Event ids, static code 0..5 (6 and 7 reserved). One token in flight per
// (route, id).
constexpr std::int32_t kEvtRawReady = 0;    // MTE2_V
constexpr std::int32_t kEvtStripReady = 1;  // MTE2_V
constexpr std::int32_t kEvtRawFree = 0;     // V_MTE2
constexpr std::int32_t kEvtPackReady = 0;   // V_MTE3
constexpr std::int32_t kEvtStripDone = 1;   // V_MTE3
constexpr std::int32_t kEvtPackFree = 0;    // MTE3_V
constexpr std::int32_t kEvtStripFree = 0;   // MTE3_MTE2
constexpr std::int32_t kEvtDrain = 0;       // MTE3_S

constexpr auto kUb = AscendC::TPosition::VECCALC;

template <class T>
__aicore__ inline AscendC::LocalTensor<T> View(std::uint32_t base, std::uint32_t elems)
{
    return AscendC::LocalTensor<T>{kUb, base, elems};
}

// ===================================================== the quantiser VF
//
// Rotate one row and quantise it, register-resident from the load to the
// clamp. The butterfly leaf's per-row sequence is a __simd_callee__ precisely
// so this epilogue can share the row's four registers with it instead of
// handing them back through Unified Buffer in between.
//
// The absmax is a butterfly all-reduce rather than a `Reduce` plus a
// broadcast: six rounds of (gather the XOR partner, take the max) over the
// index registers the rotation already built leave the row maximum in EVERY
// lane of the result, which is the form the quantise multiply needs anyway.
// Fifteen instructions, no reduction API, no broadcast step and no new
// constant.
__simd_callee__ inline void absmax_broadcast(AscendC::Reg::RegTensor<float>& m,
                                             ascend_micro::RotRegSet& v,
                                             ascend_micro::RotGatherTables& t,
                                             AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<float> a0, a1, a2, a3, partner;
    AscendC::Reg::Abs<float>(a0, v.v0, mask);
    AscendC::Reg::Abs<float>(a1, v.v1, mask);
    AscendC::Reg::Abs<float>(a2, v.v2, mask);
    AscendC::Reg::Abs<float>(a3, v.v3, mask);
    AscendC::Reg::Max<float>(m, a0, a1, mask);
    AscendC::Reg::Max<float>(m, m, a2, mask);
    AscendC::Reg::Max<float>(m, m, a3, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i0);
    AscendC::Reg::Max<float>(m, m, partner, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i1);
    AscendC::Reg::Max<float>(m, m, partner, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i2);
    AscendC::Reg::Max<float>(m, m, partner, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i3);
    AscendC::Reg::Max<float>(m, m, partner, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i4);
    AscendC::Reg::Max<float>(m, m, partner, mask);
    AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i5);
    AscendC::Reg::Max<float>(m, m, partner, mask);
}

// `qout` receives y * 2^(2-e) clamped from above at 7.0f -- NOT rounded. The
// single rounding of the whole path is the level-2 Cast to INT32 the caller
// performs next, on an exactly representable product, which is what lets the
// host oracle reproduce it in integer arithmetic. `scaleOut` receives one
// vector-register slot per row with the stored scale in lane 0.
__simd_vf__ inline void scatter_rotate_quantise_vf(__ubuf__ float* qout,
                                                   __ubuf__ float* scaleOut,
                                                   __ubuf__ float* src,
                                                   __ubuf__ float* sign, float inv_sqrt_len,
                                                   std::uint16_t rows)
{
    AscendC::Reg::MaskReg mask =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg lane0 =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
    ascend_micro::RotGatherTables t;
    ascend_micro::rot_build_gather_tables(t, mask);
    ascend_micro::RotSignRegs p;
    ascend_micro::rot_load_signs(p, sign);

    // Power-of-two constants. A float 2^k carries exponent field k + 127, so
    // with 2^e <= absmax < 2^(e+1):
    //   exp_bits = bits(absmax) & 0x7F800000   -> 2^e,     field e + 127
    //   pow_bits = exp_bits - (2 << 23)        -> 2^(e-2), field e + 125
    //   inv_bits = (254 << 23) - pow_bits      -> 2^(2-e), field 129 - e
    // The last line is the exact reciprocal of a power of two: the two fields
    // sum to 254. No division, no logarithm, no hardware reciprocal.
    AscendC::Reg::RegTensor<std::uint32_t> exp_mask, two_pow, inv_seed;
    AscendC::Reg::Duplicate<std::uint32_t>(exp_mask, 0x7F800000u);
    AscendC::Reg::Duplicate<std::uint32_t>(two_pow, 0x01000000u);   // 2 << 23
    AscendC::Reg::Duplicate<std::uint32_t>(inv_seed, 0x7F000000u);  // 254 << 23
    AscendC::Reg::RegTensor<float> one, inv_sqrt, zero;
    AscendC::Reg::Duplicate<float>(one, 1.0f);
    AscendC::Reg::Duplicate<float>(inv_sqrt, inv_sqrt_len);
    AscendC::Reg::Duplicate<float>(zero, 0.0f);

    for (std::uint16_t r = 0; r < rows; ++r) {
        const std::uint32_t row = static_cast<std::uint32_t>(r);
        ascend_micro::RotRegSet v;
        ascend_micro::rot_load_row(v, src + row * kHeadSize);
        ascend_micro::rot_row_gather<ascend_micro::RotateDirection::Ingress>(v, t, p, mask);

        AscendC::Reg::RegTensor<float> absmax;
        absmax_broadcast(absmax, v, t, mask);

        auto& abs_bits = reinterpret_cast<AscendC::Reg::RegTensor<std::uint32_t>&>(absmax);
        AscendC::Reg::RegTensor<std::uint32_t> exp_bits, pow_bits, inv_bits;
        AscendC::Reg::And<std::uint32_t>(exp_bits, abs_bits, exp_mask, mask);
        AscendC::Reg::Sub<std::uint32_t>(pow_bits, exp_bits, two_pow, mask);
        AscendC::Reg::Sub<std::uint32_t>(inv_bits, inv_seed, pow_bits, mask);
        auto& inv = reinterpret_cast<AscendC::Reg::RegTensor<float>&>(inv_bits);
        auto& pow_two = reinterpret_cast<AscendC::Reg::RegTensor<float>&>(pow_bits);

        // |y * 2^(2-e)| < 8 and round-to-nearest-even can reach +8, which
        // INT4 cannot hold, so clamp from above only; -8 is in range. The
        // clamp commutes with the rounding here because everything it touches
        // -- [7, 8) -- maps to 7 either way.
        AscendC::Reg::Mul<float>(v.v0, v.v0, inv, mask);
        AscendC::Reg::Mul<float>(v.v1, v.v1, inv, mask);
        AscendC::Reg::Mul<float>(v.v2, v.v2, inv, mask);
        AscendC::Reg::Mul<float>(v.v3, v.v3, inv, mask);
        AscendC::Reg::Mins<float, float>(v.v0, v.v0, 7.0f, mask);
        AscendC::Reg::Mins<float, float>(v.v1, v.v1, 7.0f, mask);
        AscendC::Reg::Mins<float, float>(v.v2, v.v2, 7.0f, mask);
        AscendC::Reg::Mins<float, float>(v.v3, v.v3, 7.0f, mask);
        ascend_micro::rot_store_row(qout + row * kHeadSize, v, mask);

        // invSqrtLen folds into the STORED SCALE and never into the code: the
        // quantised panel is scale invariant, because scaling y and absmax
        // together leaves every code unchanged, and that is what keeps the
        // quantiser bit-exact against an integer oracle.
        AscendC::Reg::RegTensor<float> scaled, scale;
        AscendC::Reg::Mul<float>(scaled, pow_two, inv_sqrt, mask);
        AscendC::Reg::MaskReg zero_row;
        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(zero_row, absmax, zero, mask);
        AscendC::Reg::Select<float>(scale, one, scaled, zero_row);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            scaleOut + row * kVecLanes, scale, lane0);
    }
}

// ===================================================== the value-plane merge
//
// Splice one token's 256 nibbles into the 2048-byte strip without touching a
// neighbouring token's.
//
// The arithmetic. Inside the strip a channel owns 16 consecutive nibbles, one
// per token, so a channel is 8 bytes and the strip is 256 of them. As uint32
// words, word 2c holds channel c's tokens 0..7 and word 2c + 1 its tokens
// 8..15, with token t at bit 4 * (t % 8). So one token owns
//
//   words  { 2c + t/8 : c = 0..255 }      -- every other word
//   bits   [4*(t%8) + 3 : 4*(t%8)]        -- the same nibble in every one
//
// `Half` is t / 8 and is a template parameter: the two cases differ only in
// which half of an `Interleave` the mask comes from, and keeping the branch
// outside the VF body costs one instantiation and no runtime decision.
//
// `Interleave(e0, e1, s, s)` duplicates each code into BOTH words of its
// channel; the nibble mask then deletes the copy in the word this token does
// not own, so no second interleave and no lane-parity constant is needed.
template <std::uint32_t Half>
__simd_vf__ inline void scatter_merge_token_vf(__ubuf__ std::uint32_t* strip,
                                               __ubuf__ std::int32_t* codes,
                                               std::int16_t shift)
{
    AscendC::Reg::MaskReg mask =
        AscendC::Reg::CreateMask<std::uint32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<std::uint32_t> zero, fifteen, ones, nib;
    AscendC::Reg::Duplicate<std::uint32_t>(zero, 0u);
    AscendC::Reg::Duplicate<std::uint32_t>(fifteen, 0xFu);
    AscendC::Reg::Duplicate<std::uint32_t>(ones, 0xFFFFFFFFu);
    AscendC::Reg::Duplicate<std::uint32_t>(
        nib, 0xFu << static_cast<std::uint32_t>(shift));

    AscendC::Reg::RegTensor<std::uint32_t> keep, keep_hi, drop;
    if constexpr (Half == 0u) {
        AscendC::Reg::Interleave<std::uint32_t>(keep, keep_hi, nib, zero);
    } else {
        AscendC::Reg::Interleave<std::uint32_t>(keep, keep_hi, zero, nib);
    }
    // Both Interleave outputs are the same constant pattern, because both
    // sources are lane-invariant; only `keep` is used.
    AscendC::Reg::Xor<std::uint32_t>(drop, keep, ones, mask);

    for (std::uint32_t r = 0; r < ascend_micro::kRotRegisters; ++r) {
        AscendC::Reg::RegTensor<std::int32_t> ci;
        AscendC::Reg::LoadAlign<std::int32_t, AscendC::Reg::LoadDist::DIST_NORM>(
            ci, codes + r * kVecLanes);
        auto& cu = reinterpret_cast<AscendC::Reg::RegTensor<std::uint32_t>&>(ci);
        AscendC::Reg::RegTensor<std::uint32_t> nibs, placed, e0, e1, s0, s1;
        AscendC::Reg::And<std::uint32_t>(nibs, cu, fifteen, mask);
        AscendC::Reg::ShiftLefts<std::uint32_t>(placed, nibs, shift, mask);
        AscendC::Reg::Interleave<std::uint32_t>(e0, e1, placed, placed);
        AscendC::Reg::And<std::uint32_t>(e0, e0, keep, mask);
        AscendC::Reg::And<std::uint32_t>(e1, e1, keep, mask);
        __ubuf__ std::uint32_t* w0 = strip + (2u * r) * kVecLanes;
        __ubuf__ std::uint32_t* w1 = strip + (2u * r + 1u) * kVecLanes;
        AscendC::Reg::LoadAlign<std::uint32_t, AscendC::Reg::LoadDist::DIST_NORM>(s0, w0);
        AscendC::Reg::LoadAlign<std::uint32_t, AscendC::Reg::LoadDist::DIST_NORM>(s1, w1);
        AscendC::Reg::And<std::uint32_t>(s0, s0, drop, mask);
        AscendC::Reg::And<std::uint32_t>(s1, s1, drop, mask);
        AscendC::Reg::Or<std::uint32_t>(s0, s0, e0, mask);
        AscendC::Reg::Or<std::uint32_t>(s1, s1, e1, mask);
        AscendC::Reg::StoreAlign<std::uint32_t, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            w0, s0, mask);
        AscendC::Reg::StoreAlign<std::uint32_t, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            w1, s1, mask);
    }
}

// ============================================================ the operator
struct Geometry {
    std::uint32_t num_kv_heads;
    std::uint32_t head_size;
    std::uint32_t block_size;  // S
    std::uint32_t num_blocks;  // P
    float inv_sqrt_len;
};

// One (block, head, strip) read-modify-write transaction: `run` consecutive
// tokens whose token-in-block indices are consecutive and which all land in
// the same strip.
struct Run {
    std::uint32_t token_begin;  // index into the flat token list
    std::uint32_t run;          // 1..kMaxRun
    std::uint32_t head;
    std::uint32_t block;
    std::uint32_t token_in_block;  // of token_begin
    std::uint32_t strip;
};

// The strip's offset in the value plane, in uint32 words. Derived in ONE place:
// the read and the write-back have to name the same 2048 bytes, and two copies
// of this expression is exactly the shape of a read-modify-write that reads one
// strip and overwrites another.
__aicore__ inline std::uint64_t StripWordOffset(const Run& item, const Geometry& geom)
{
    const std::uint64_t nibble =
        (static_cast<std::uint64_t>(item.block) * geom.num_kv_heads + item.head) *
            geom.block_size * geom.head_size +
        static_cast<std::uint64_t>(item.strip) * kStripHeight * geom.head_size;
    return nibble / 8u;  // 8 nibbles per uint32 word
}

// Stage `run` rows of one plane into the BF16 landing buffer. The activation
// tensors are [num_tokens, H, d], so a row is 256 BF16 = 16 DataBlocks and
// consecutive rows are H * d * 2 bytes apart.
__aicore__ inline void StageRows(const AscendC::GlobalTensor<bfloat16_t>& plane,
                                 const Run& item, const Geometry& geom)
{
    AscendC::DataCopyParams rows;
    rows.blockCount = static_cast<std::uint16_t>(item.run);
    rows.blockLen = static_cast<std::uint16_t>(geom.head_size * 2u / kBlock);
    rows.srcStride =
        static_cast<std::uint16_t>((geom.num_kv_heads - 1u) * geom.head_size * 2u / kBlock);
    rows.dstStride = 0;
    const std::uint64_t base =
        (static_cast<std::uint64_t>(item.token_begin) * geom.num_kv_heads + item.head) *
        geom.head_size;
    AscendC::DataCopy(View<bfloat16_t>(kUbRaw, kMaxRun * kHeadSize), plane[base], rows);
}

// The rotation and the quantiser, from the BF16 landing buffer to the INT32
// codes and the scale slots. Shared by both planes.
__aicore__ inline void RotateQuantise(const Run& item, const Geometry& geom)
{
    const std::uint32_t elems = item.run * geom.head_size;
    AscendC::Cast(View<float>(kUbAct, kPanelElems), View<bfloat16_t>(kUbRaw, kPanelElems),
                  AscendC::RoundMode::CAST_NONE, static_cast<std::int32_t>(elems));
    AscendC::PipeBarrier<PIPE_V>();
    scatter_rotate_quantise_vf(
        reinterpret_cast<__ubuf__ float*>(View<float>(kUbQ, kPanelElems).GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(
            View<float>(kUbScale, kMaxRun * kVecLanes).GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(View<float>(kUbAct, kPanelElems).GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(View<float>(kUbSign, kHeadSize).GetPhyAddr()),
        geom.inv_sqrt_len, static_cast<std::uint16_t>(item.run));
    AscendC::PipeBarrier<PIPE_V>();
    // The single rounding of the whole path.
    AscendC::Cast(View<std::int32_t>(kUbQuant, kPanelElems), View<float>(kUbQ, kPanelElems),
                  AscendC::RoundMode::CAST_RINT, static_cast<std::int32_t>(elems));
}

// ---------------------------------------------------------------- the planes
__aicore__ inline void WriteKeyPlane(const AscendC::GlobalTensor<std::int8_t>& key_cache,
                                     const AscendC::GlobalTensor<float>& scale_cache,
                                     const Run& item, const Geometry& geom)
{
    const std::uint32_t elems = item.run * geom.head_size;
    // INT32 -> FP16 -> INT4. The hardware Cast to int4b_t takes a two-byte
    // source, the staged values are exact integers in [-8, 7], so both
    // narrowing steps are exact and the INT32 cast above remains the only
    // rounding. The INT16 source form compiles to a deprecated vcvt_s162s4
    // intrinsic, which is the only reason this goes through FP16.
    AscendC::Cast(View<half>(kUbHalf, kPanelElems), View<std::int32_t>(kUbQuant, kPanelElems),
                  AscendC::RoundMode::CAST_RINT, static_cast<std::int32_t>(elems));
    AscendC::PipeBarrier<PIPE_V>();
    auto packed = View<std::int8_t>(kUbPacked, kMaxRun * kKeyTokenBytes);
    auto nibbles = packed.ReinterpretCast<AscendC::int4b_t>();
    AscendC::Cast(nibbles, View<half>(kUbHalf, kPanelElems), AscendC::RoundMode::CAST_RINT,
                  static_cast<std::int32_t>(elems));

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtPackReady);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtPackReady);

    // Token-major: one token is 128 contiguous bytes at a 32-byte aligned
    // offset, and a run of consecutive tokens is one burst.
    const std::uint64_t key_base =
        ((static_cast<std::uint64_t>(item.block) * geom.num_kv_heads + item.head) *
             geom.block_size +
         item.token_in_block) *
        (geom.head_size / 2u);
    AscendC::DataCopy(key_cache[key_base], packed, item.run * kKeyTokenBytes);

    // The scale plane is planar, so a run's tokens are consecutive FP32 lanes
    // of one (kv, block, head) slice -- but one token is four bytes, which is
    // not a burst. DataCopyPad per token, from a 32-byte aligned slot.
    const std::uint64_t scale_base =
        ((static_cast<std::uint64_t>(0u) * geom.num_blocks + item.block) * geom.num_kv_heads +
         item.head) *
            geom.block_size +
        item.token_in_block;
    AscendC::DataCopyExtParams one_float;
    one_float.blockCount = 1;
    one_float.blockLen = 4u;
    one_float.srcStride = 0;
    one_float.dstStride = 0;
    one_float.rsv = 0;
    for (std::uint32_t r = 0; r < item.run; ++r)
        AscendC::DataCopyPad(scale_cache[scale_base + r],
                             View<float>(kUbScale + r * kVecLanes * 4u, 1u), one_float);
}

__aicore__ inline void WriteValuePlane(
    const AscendC::GlobalTensor<std::uint32_t>& value_words,
    const AscendC::GlobalTensor<float>& scale_cache, const Run& item, const Geometry& geom)
{
    // Splice every owned token into the strip image MTE2 has already read.
    for (std::uint32_t r = 0; r < item.run; ++r) {
        const std::uint32_t token = (item.token_in_block + r) % kStripHeight;
        const auto shift = static_cast<std::int16_t>(4u * (token % 8u));
        __ubuf__ std::uint32_t* strip = reinterpret_cast<__ubuf__ std::uint32_t*>(
            View<std::uint32_t>(kUbStrip, kStripWords).GetPhyAddr());
        __ubuf__ std::int32_t* codes = reinterpret_cast<__ubuf__ std::int32_t*>(
            View<std::int32_t>(kUbQuant + r * kHeadSize * 4u, kHeadSize).GetPhyAddr());
        if (token < 8u)
            scatter_merge_token_vf<0u>(strip, codes, shift);
        else
            scatter_merge_token_vf<1u>(strip, codes, shift);
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtStripDone);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtStripDone);

    AscendC::DataCopy(value_words[StripWordOffset(item, geom)],
                      View<std::uint32_t>(kUbStrip, kStripWords), kStripWords);

    const std::uint64_t scale_base =
        ((static_cast<std::uint64_t>(1u) * geom.num_blocks + item.block) * geom.num_kv_heads +
         item.head) *
            geom.block_size +
        item.token_in_block;
    AscendC::DataCopyExtParams one_float;
    one_float.blockCount = 1;
    one_float.blockLen = 4u;
    one_float.srcStride = 0;
    one_float.dstStride = 0;
    one_float.rsv = 0;
    for (std::uint32_t r = 0; r < item.run; ++r)
        AscendC::DataCopyPad(scale_cache[scale_base + r],
                             View<float>(kUbScale + r * kVecLanes * 4u, 1u), one_float);
}

// --------------------------------------------------------------- the body
__aicore__ inline void ProcessRun(const AscendC::GlobalTensor<bfloat16_t>& key,
                                  const AscendC::GlobalTensor<bfloat16_t>& value,
                                  const AscendC::GlobalTensor<std::int8_t>& key_cache,
                                  const AscendC::GlobalTensor<std::uint32_t>& value_words,
                                  const AscendC::GlobalTensor<float>& scale_cache,
                                  const Run& item, const Geometry& geom)
{
    // ---- key plane: no read-modify-write ---------------------------------
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtRawFree);
    StageRows(key, item, geom);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtRawReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtRawReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtPackFree);
    RotateQuantise(item, geom);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtRawFree);
    AscendC::PipeBarrier<PIPE_V>();
    WriteKeyPlane(key_cache, scale_cache, item, geom);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtPackFree);

    // ---- value plane: read-modify-write of one 2048-byte strip -----------
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtRawFree);
    StageRows(value, item, geom);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtRawReady);

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kEvtStripFree);
    AscendC::DataCopy(View<std::uint32_t>(kUbStrip, kStripWords),
                      value_words[StripWordOffset(item, geom)], kStripWords);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtStripReady);

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtRawReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtStripReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtPackFree);
    RotateQuantise(item, geom);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtRawFree);
    AscendC::PipeBarrier<PIPE_V>();
    WriteValuePlane(value_words, scale_cache, item, geom);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kEvtStripFree);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtPackFree);
}

__aicore__ inline void Run_(__gm__ bfloat16_t* key_gm, __gm__ bfloat16_t* value_gm,
                            __gm__ std::int8_t* key_cache_gm,
                            __gm__ std::int8_t* value_cache_gm, __gm__ float* scale_cache_gm,
                            __gm__ std::int32_t* slot_gm, __gm__ float* pi_signs_gm,
                            __gm__ std::int32_t* cu_seqlens_gm, __gm__ std::int32_t* probe_gm,
                            std::uint32_t num_requests, std::uint32_t num_kv_heads,
                            std::uint32_t head_size, std::uint32_t block_size,
                            std::uint32_t num_blocks, std::uint32_t item_begin,
                            std::uint32_t item_count, float inv_sqrt_len)
{
    // `headSize` is a runtime operand and `kHeadSize` is the butterfly's
    // compile-time width, so the kernel is only defined where they agree: the
    // leaf's four resident registers, the 2048-byte strip geometry and the
    // merge's (word, bit) derivation are all 256 channels by construction. The
    // adapter checks `head_size == 256` on the host; this is the device half of
    // the same contract, because a silently wrong head size would read one
    // head's rows and write another head's slice.
    if (head_size != kHeadSize) return;
    if (block_size % kStripHeight != 0u) return;

    Geometry geom;
    geom.num_kv_heads = num_kv_heads;
    geom.head_size = head_size;
    geom.block_size = block_size;
    geom.num_blocks = num_blocks;
    geom.inv_sqrt_len = inv_sqrt_len;

    AscendC::GlobalTensor<bfloat16_t> key, value;
    AscendC::GlobalTensor<std::int8_t> key_cache;
    AscendC::GlobalTensor<std::uint32_t> value_words;
    AscendC::GlobalTensor<float> scale_cache, pi_signs;
    AscendC::GlobalTensor<std::int32_t> slots, cu_seqlens;
    key.SetGlobalBuffer(key_gm);
    value.SetGlobalBuffer(value_gm);
    key_cache.SetGlobalBuffer(key_cache_gm);
    value_words.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint32_t*>(value_cache_gm));
    scale_cache.SetGlobalBuffer(scale_cache_gm);
    pi_signs.SetGlobalBuffer(pi_signs_gm);
    slots.SetGlobalBuffer(slot_gm);
    if (cu_seqlens_gm != nullptr) cu_seqlens.SetGlobalBuffer(cu_seqlens_gm);

    const std::uint32_t block_idx = static_cast<std::uint32_t>(AscendC::GetBlockIdx());
    const std::uint32_t block_num = static_cast<std::uint32_t>(AscendC::GetBlockNum());

    // The roadmap's open item: what GetBlockIdx / GetBlockNum /
    // GetSubBlockNum report under KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)
    // is OBSERVED here rather than assumed, because the safe MIX idiom
    // (GetBlockIdx() / GetSubBlockNum()) and the AIV-only one differ by
    // exactly this. Block 0 of every launch publishes all four.
    //
    // Four DataBlocks rather than four lanes: a scalar reaches Unified Buffer
    // through `Duplicate`, which writes a whole 32-byte block from a
    // vector-pipe broadcast, and that avoids both a LocalTensor::SetValue and
    // the V_S fence a scalar UB write would need. The runner reads lanes 0, 8,
    // 16 and 24.
    if (probe_gm != nullptr && block_idx == 0u) {
        AscendC::GlobalTensor<std::int32_t> probe;
        probe.SetGlobalBuffer(probe_gm);
        auto probe_ub = View<std::int32_t>(kUbProbe, kProbeLanes);
        AscendC::Duplicate(probe_ub, static_cast<std::int32_t>(block_idx), 8);
        AscendC::Duplicate(probe_ub[8], static_cast<std::int32_t>(block_num), 8);
        AscendC::Duplicate(probe_ub[16], static_cast<std::int32_t>(AscendC::GetSubBlockNum()),
                           8);
        AscendC::Duplicate(probe_ub[24], static_cast<std::int32_t>(item_count), 8);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtPackReady);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtPackReady);
        AscendC::DataCopy(probe[0], probe_ub, kProbeLanes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    }

    // Pi's sign diagonal, once per launch: 256 FP32 read by the butterfly's
    // own sign stage. `pi_signs` is live for this operator, unlike for the
    // aligned writer, which folds the diagonal into its Cube tables and never
    // reads the operand (RFC section 4.0 H-B).
    AscendC::DataCopy(View<float>(kUbSign, head_size), pi_signs[0], head_size);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtRawReady);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtRawReady);

    // Prime the three reuse routes: both landing buffers and the strip start
    // free. The post-loop waits drain them, so no token leaks across launches.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kEvtRawFree);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kEvtPackFree);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kEvtStripFree);

    const std::uint32_t item_end = item_begin + item_count;
    for (std::uint32_t item = item_begin + block_idx; item < item_end; item += block_num) {
        const std::uint32_t request = item / num_kv_heads;
        const std::uint32_t head = item % num_kv_heads;
        if (request >= num_requests) continue;
        std::uint32_t token = cu_seqlens_gm != nullptr
                                  ? static_cast<std::uint32_t>(cu_seqlens.GetValue(request))
                                  : request;
        const std::uint32_t token_end =
            cu_seqlens_gm != nullptr
                ? static_cast<std::uint32_t>(cu_seqlens.GetValue(request + 1u))
                : request + 1u;

        while (token < token_end) {
            const std::int32_t slot0 = slots.GetValue(token);
            if (slot0 < 0) {  // a padded token owns nothing
                ++token;
                continue;
            }
            Run active;
            active.token_begin = token;
            active.head = head;
            active.block = static_cast<std::uint32_t>(slot0) / block_size;
            active.token_in_block = static_cast<std::uint32_t>(slot0) % block_size;
            active.strip = active.token_in_block / kStripHeight;
            active.run = 1u;
            // Extend the transaction while the slots stay consecutive AND
            // inside the same strip: that is exactly the set of tokens one
            // 2048-byte read-modify-write and one key burst can carry.
            while (active.token_begin + active.run < token_end && active.run < kMaxRun) {
                const std::int32_t next = slots.GetValue(active.token_begin + active.run);
                if (next != slot0 + static_cast<std::int32_t>(active.run)) break;
                const std::uint32_t tib = static_cast<std::uint32_t>(next) % block_size;
                if (static_cast<std::uint32_t>(next) / block_size != active.block) break;
                if (tib / kStripHeight != active.strip) break;
                ++active.run;
            }
            ProcessRun(key, value, key_cache, value_words, scale_cache, active, geom);
            token = active.token_begin + active.run;
        }
    }

    // Drain the primed tokens.
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kEvtRawFree);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kEvtPackFree);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kEvtStripFree);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
}

}  // namespace turboquant_scatter

#endif  // __CCE_AICORE__ -- everything above is the device-side body

// ------------------------------------------------------------- entry point
// Compiled in EVERY pass of this translation unit, host and device alike; only
// the body stays behind __CCE_AICORE__. Under the CANN 9.2 ASC language
// toolchain the host-side launch stub of a `__global__` function is emitted by
// the host pass from the function's DEFINITION, so a definition hidden behind
// __CCE_AICORE__ leaves the triple-chevron launch below referring to a symbol
// nothing in libvllm_ascend_turboquant.so defines. See the longer note in
// turboquant_reshape_batch.cpp.
extern "C" __global__ __aicore__ void turboquant_reshape_scatter_aiv(
    __gm__ bfloat16_t* __restrict__ key, __gm__ bfloat16_t* __restrict__ value,
    __gm__ std::int8_t* __restrict__ keyCache, __gm__ std::int8_t* __restrict__ valueCache,
    __gm__ float* __restrict__ scaleCache, __gm__ std::int32_t* __restrict__ slotMapping,
    __gm__ float* __restrict__ piSigns, __gm__ std::int32_t* __restrict__ cuSeqlens,
    __gm__ std::int32_t* __restrict__ launchProbe, uint32_t numRequests,
    uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize, uint32_t numBlocks,
    uint32_t itemBegin, uint32_t itemCount, float invSqrtLen)
{
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    turboquant_scatter::Run_(key, value, keyCache, valueCache, scaleCache, slotMapping,
                             piSigns, cuSeqlens, launchProbe, numRequests, numKvHeads,
                             headSize, blockSize, numBlocks, itemBegin, itemCount,
                             invSqrtLen);
#endif
}

#if defined(ASCEND_MICRO_CANN_LAUNCH) && !defined(__CCE_AICORE__)

// No prototype for the kernel here any more: its definition above is visible in
// this pass, which is what makes the host launch stub exist at all, and the one
// argument block is now written once rather than twice.
extern "C" void launch_turboquant_reshape_scatter_aiv(
    void* stream, std::uint32_t block_dim, std::uint16_t* key, std::uint16_t* value,
    std::int8_t* key_cache, std::int8_t* value_cache, float* scale_cache,
    std::int32_t* slot_mapping, float* pi_signs, std::int32_t* cu_seqlens,
    std::int32_t* launch_probe, std::uint32_t num_requests, std::uint32_t num_kv_heads,
    std::uint32_t head_size, std::uint32_t block_size, std::uint32_t num_blocks,
    std::uint32_t item_begin, std::uint32_t item_count, float inv_sqrt_len)
{
    turboquant_reshape_scatter_aiv<<<block_dim, nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(key), reinterpret_cast<bfloat16_t*>(value), key_cache,
        value_cache, scale_cache, slot_mapping, pi_signs, cu_seqlens, launch_probe,
        num_requests, num_kv_heads, head_size, block_size, num_blocks, item_begin,
        item_count, inv_sqrt_len);
}

#endif
