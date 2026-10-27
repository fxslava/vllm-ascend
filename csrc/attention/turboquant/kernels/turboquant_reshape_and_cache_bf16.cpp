// TurboQuant A8 prefill reshape-and-cache, Ascend 950PR / arch35 / dav-3510 /
// CANN 9.2, MIX_AIC_1_2: one Cube core and BOTH Vector cores in a four-stage
// depth-4 modulo pipeline over the eight chunks of one 16-token strip.
//
// What this proves
// ----------------
//   * a four-stage ring (MTE2 -> Cube/Fixpipe -> Vector -> MTE3) in which all
//     four engines plus both Vector cores are in flight together;
//   * the *whole* randomised Hadamard rotation as INT8 x INT8 -> INT32 Cube
//     work, driven straight out of a Global Memory page staged into L1 with no
//     Unified Buffer hop. No part of the orthogonal transform runs on the
//     Vector cores: no sign diagonal, no butterfly cascade;
//   * a bit-exact INT4 KV cache: the whole quantisation path is integer and
//     power-of-two, so the host oracle reproduces every nibble exactly for an
//     arbitrary INT8 input, not just for a crafted fixture.
//
// What it deliberately omits
// --------------------------
//   * ragged multi-core tiling. Each block owns one aligned 16-token strip;
//     the launcher uses numTokens / tokensPerCore blocks.
//   * ragged shapes. The strip height (kTokens == 16), headSize == 256 and
//     numKvHeads == 4 are compile-time constants (kTokens, kHeadSize,
//     kHeads); the scalar arguments drive the physical cache addressing only.
//     blockSize is a genuine runtime value and may be any multiple of kTokens,
//     128 included; numTokens must be a multiple of kTokens.
//   * a slot mapping that is not kTokens-aligned and consecutive within a
//     strip (see "Slot mapping contract" below). The strip need not start a
//     page: blockSize may be any multiple of kTokens.
//
// Target: Ascend950PR_9599 CAModel, __NPU_ARCH__ == 3510, CANN 9.2.0-beta.2.
// Grid:   8 chunks over a depth-4 L1 ring and depth-2 per-core UB rings, so
//         every ring wraps at least twice in one run and every buffer-reuse
//         back edge is contended rather than merely primed and drained
//         (MICROROUTINE_DESIGN_GUIDELINES.md section 3.1).
//
//   engine        owner        work
//   ------------  -----------  --------------------------------------------
//   MTE2          AIC          GM -> L1 page, 4 KiB per chunk, no UB hop
//   MTE1/M/FIX    AIC          L1 -> L0, 64 INT8 Mmads, ONE Fixpipe -> AIV UB
//   Vector        AIV0 + AIV1  absmax, power-of-two scale, INT4 nibble pack
//   MTE3          AIV0 + AIV1  2 KiB packed page + 64 B scale slice -> GM
//
// The rotation, and why it is one Cube pass
// -----------------------------------------
// Per head of kHeadSize channels the transform is the randomised Hadamard
//
//   Y = M . x,   M = (H8 (x) I32) . diag(piSigns) . (I8 (x) H32)
//
// normalised by invSqrtLen = 1 / sqrt(headSize). Writing the channel index as
// ch = c1 * kGroup + c0, the entries of M are
//
//   M[c1*32 + c0][g*32 + i] = H8[c1][g] . piSigns[g*32 + c0] . H32[c0][i]      (1)
//
// - one signed entry per (output channel, input channel) pair, because the
// sign diagonal sits *between* the two factors and is therefore indexed by the
// intermediate channel (g, c0). M M^T = 256 I exactly; the Tier-1 host probe
// measures that rather than asserting it.
//
// An earlier revision split (1) across the engine classes: the Cube applied
// the inner factor I8 (x) H32 as one Mmad per channel group, and the Vector
// cores applied diag(piSigns) and then the outer factor H8 (x) I32 as three
// radix-2 butterfly stages. That is the cheap factorisation in multiply-adds
// and the expensive one in wall clock. Measured on this probe
// (reports/reshape_and_cache_a8.md):
//
//   * the Cube spent 33.6% of its attributed cycles in WAIT_INTRA_BLOCK
//     waiting for a Vector core to release a landing panel, against 2.8% in
//     MMAD - it waited twelve times longer than it multiplied;
//   * the Vector cost was register traffic, not arithmetic: RV_VLD + RV_VST
//     was 48.6% of the Vector cycles against 24.4% for the whole execute pipe,
//     and the sign multiply plus three butterfly stages are four full-panel
//     passes of exactly that traffic.
//
// So this revision folds all of (1) into the Cube. The contraction is split
// over the eight source groups g, and the outer factor's sign H8[c1][g] is
// folded into the *resident operand* rather than applied afterwards: for
// output group c1 and source group g the Cube multiplies the page's group-g
// fractal by the 32x32 block
//
//   T[s][g][c0][i] = (s ? -1 : +1) . piSigns[g*32 + c0] . H32[c0][i],
//   s = parity(c1 & g) = (H8[c1][g] < 0)                                      (2)
//
// and accumulates the eight products for one c1 into one L0C region. There are
// only 2 * kGroups = 16 distinct blocks, 16 KiB of INT8, because the dependence
// on c1 is a single sign; `tables` carries them and they stay resident in both
// L0A and L0B for the whole kernel. The 8 x 8 = 64 Mmads per chunk cost eight
// times the multiply-adds of the factored form - that is the price of folding a
// factor that is diagonal in 32 channels into a GEMM - and the Cube had the
// headroom: it was idle for two thirds of its attributed cycles.
//
// Three consequences, all of them the point:
//
//   * the Vector stage loses the sign multiply and the three butterfly stages,
//     four of its thirteen full-panel passes, and the three ping-pong panels
//     and two sign panels they needed (65 KiB of Unified Buffer);
//   * the Cube emits ONE Fixpipe per chunk instead of one per channel group,
//     because the whole 16 x 256 panel is finished in L0C before anything
//     leaves it. FIX_L0C_TO_DST plus its MOV_SPR_XN configuration writes were
//     43.7% of the Cube's attributed cycles at eight per chunk;
//   * the landing-panel back edge is waited *after* the Mmads rather than
//     before them, since only the Fixpipe touches Unified Buffer. The Cube's
//     64 multiplies for chunk j now overlap the Vector core's quantisation of
//     chunk j - 1 instead of queueing behind it.
//
// `piSigns` is consumed by whoever builds `tables` and is not read by the
// device any more. The runner builds (2) from the same sign function it
// uploads, and the Tier-1 probe checks that the 16 blocks composed the way the
// kernel composes them reproduce M entry for entry.
//
// Both factors are exact in INT32 and in FP32: each row of M has 256 signed
// entries and |x| <= 128, so |Y| <= 32768 and every intermediate is an integer
// below 2^24. The whole rotation is therefore bit-exact against an integer
// oracle for *any* INT8 input.
//
// kGroup is 32 because that is the INT8 K0: the arch-3510 INT8 fractal is
// 16 x 32, so a 32-point contraction is exactly one K fractal and needs no
// zero padding, and a 32 x 32 resident block is exactly two L0 fractals.
//
// One L1 image, two operand roles
// -------------------------------
// For INT8 the NZ image of a 16 x 256 page and the ZN image of its 256 x 16
// transpose are the same bytes: NZ is (M1, K1, M0=16, K0=32) with the fractal
// stored as 16 rows of 32 bytes, and ZN is (K1, N1, N0=16, K0=32) with the
// fractal stored as 16 rows of 32 bytes as well. Channel group c1 is fractal
// c1 of both, at byte offset c1 * 512. So one GM -> L1 ND -> NZ copy per chunk
// produces a page that serves as
//
//   A operand, NZ(16, 32)  -> C[t][c0out]   for the key chunks, and
//   B operand, ZN(32, 16)  -> C[c0out][t]   for the value chunks,
//
// which is exactly the orientation each cache layout needs.
//
// This survives the widening to 16 bits with K0 = 16, where a 32-wide group is
// two fractals rather than one. The page is NZ(16, 256), so M1 = 1 and its
// image is n1*256 + t*16 + n0 with channel n1*16 + n0; the group-G slice read
// as ZN(32, 16) has N1 = 1 and asks for k1*256 + t*16 + k0 with channel
// G*32 + k1*16 + k0. In both readings the only fractal-grid axis is the
// channel axis and the group's two sub-fractals are adjacent along it, so the
// bytes coincide. The value chunks need no new load path, and
// tests/camodel/test_micro_layout_bf16.cpp measures that rather than assuming
// it.
//
// A second symmetry covers the resident blocks, and ONE HALF OF IT DOES NOT
// SURVIVE THE WIDENING. With W = T[s][g] of (2):
//
//   value chunks  A = NZ load of W     -> A[c0][i] = W[c0][i]
//   key chunks    B = ZN load of W     -> B[i][c0] = W[c0][i]
//
// and (1) needs exactly W[c0][i] in both roles. The part that holds at any
// width is that NZ(32, 32) of W and ZN(32, 32) of W-as-B are the SAME image,
// differing only in the name of the second index, so one L1 image still feeds
// both L0A and L0B through two byte-identity loads.
//
// The part that does not hold is the claim the INT8 file could make -- that a
// row-major 32 x 32 block already IS that image. At C0 = 32 it is: the fractal
// is a whole 32-wide row and the grid is one fractal along K. At C0 = 16 both
// grid axes are live, the required image is four 16 x 16 row-major sub-blocks
// ordered column-block major, and row-major is not it. So the 16 blocks are a
// flat row-major blob in GM -- the ABI is unchanged -- and are converted
// ND -> NZ on the way into L1 by routines/routine_stage_page_zn_bf16.hpp. The
// prologue of run_cube_core carries the derivation and the evidence.
//
// Layout contracts
// ----------------
//   keyCache   key-major. Within the (block, head) page the nibble index is
//              key * headSize + d, so one chunk is 2048 contiguous bytes.
//   valueCache channel-major within a kTokens strip, strips laid end to end.
//              Within the (block, head) page the nibble index is
//                (key / kTokens) * headSize * kTokens + ch * kTokens
//                  + (key % kTokens),
//              so one chunk is 2048 contiguous bytes. At blockSize == kTokens
//              this is identical to plain channel-major (ch * blockSize + key),
//              which is the form the CAModel probe verified; strip blocking is
//              what keeps the chunk a single aligned burst once a page holds
//              more than one strip. See run_vector_core for why the literal
//              channel-major reading cannot be written safely then.
//   scaleCache per (kv, block, head) a contiguous run of blockSize FP32, one
//              scale per token: index
//              ((kv * numBlocks + block) * numKvHeads + head) * blockSize + key
//              with kv 0 for the key tensor and 1 for the value tensor.
//   Nibbles    two per byte, low nibble first (even element index in bits
//              [3:0], odd in [7:4]), signed INT4 in [-8, 7]. This is the
//              packing `routines/routine_turboquant_dequant4.hpp` unpacks, and
//              the hardware Cast to int4b_t produces it directly.
//   tables     kTableBlocks = 16 row-major 32 x 32 INT8 blocks, block s*8 + g
//              holding T[s][g] of (2). 16384 bytes.
//
// Work sharing: one chunk per Vector core, not one token half
// ----------------------------------------------------------
// The natural reading of the four-stage schedule gives both Vector cores half
// the tokens of the *same* chunk. That is not expressible against the value
// cache contract above. Channel-major means a channel's blockSize nibbles are
// adjacent, so one channel of a 16-key page is 8 bytes; a core holding only 8
// of the 16 keys owns 4 bytes of every such group and can never assemble a
// 32-byte MTE3 burst, which is the egress granularity. Splitting instead by
// chunk - chunk c is owned by Vector core c & 1 - gives each core whole
// 16-token chunks, so both cache layouts come out as one 2048-byte burst, and
// adjacent chunks still sit on different cores, so the two Vector cores are in
// flight together. Each core then owns two value chunks and two key chunks.
//
// The quantiser is power-of-two, and that is a verification decision
// ------------------------------------------------------------------
// With 2^e <= absmax < 2^(e+1) the scale is 2^(e-2) and the quantised value is
// min(rint(Y * 2^(2-e)), 7), clamped only from above because |Y * 2^(2-e)| < 8
// and rint can reach +8. Y is an integer and 2^(2-e) a power of two, so the
// product is exact in FP32 and round-to-nearest-even on it is exact: the
// device and an integer host oracle agree bit for bit. A 7/absmax scale would
// put the rounding of a hardware `Div` inside the verification contract, where
// a 1-ULP reciprocal can flip a tie and change a nibble. The exponent is
// extracted with integer ops on the FP32 bit pattern - mask the mantissa, and
// form 2^(2-e) as one integer subtract - so no division or logarithm runs.
// invSqrtLen folds into the stored scale, not into the quantised value, since
// scaling Y and absmax together leaves every nibble unchanged.
//
// Slot mapping contract
// ---------------------
// A prefill strip is kTokens consecutive slots aligned to kTokens:
// slotMapping[i] must equal slotMapping[0] + i, and slotMapping[0] must be a
// multiple of kTokens. blockSize must be a multiple of kTokens, which the
// launcher enforces; a strip then lies wholly inside one page, at the in-page
// offset derived in run_vector_core, and need not start one. The
// kernel reads slotMapping[0] once, on the scalar unit, before the pipeline
// starts; that is a scalar read of caller-supplied Global Memory metadata, the
// same category as the online-softmax leaf's GM-resident index vector, and not
// a scalar read of computed Unified Buffer data (guidelines section 1.2). The
// runner asserts the contract.
//
// Synchronisation
// ---------------
// Marked graph, one token per free slot, every feedback cycle primed before
// the schedule and drained after it. Inside a core the edges are HardEvent
// tokens on the route whose *consumer queue* must stall; across cores they are
// mode-4 intra-block flags. Every cross-core set that means "my pipe work has
// finished" is preceded by a HardEvent::x_S drain of that pipe, because
// set_intra_block issues on the scalar stream and the pipe argument does not
// order it - guidelines section 6.2a, discovered by the eight-page fused
// attention run.
//
// Every cross-core event also gets its own flag id, set once and waited once:
// a mode-4 flag is not a token a wait consumes, so an id per ring slot lets
// the second announcement on that id be released by the first. See "cross-core
// flag ids" below for the trace that forced that.
//
// Forward edges:
//   AIC  intra   MTE2_MTE1     page staged into L1
//   AIC  intra   MTE1_M        page fractals staged into L0
//   AIC  intra   M_FIX         the 64 Mmads of a chunk have retired into L0C.
//                              The factored revision carried this dependency
//                              on matching unitFlag = 3 between one Mmad and
//                              its one Fixpipe; 64 accumulating Mmads feeding
//                              one Fixpipe get an explicit event instead, so
//                              nothing rests on how the unit flag pairs a
//                              group of Mmads.
//   AIC -> AIVk  kFlagProductReady  the landing panel holds this chunk
//   AIVk intra   V_MTE3        the packed page and its scale slice are ready
//
// Back edges, all contended in this run:
//   AIC  intra   MTE1_MTE2     L1 page slot free, after MTE1 read it
//   AIC  intra   M_MTE1        L0A/L0B page region free, after the Mmads
//   AIC  intra   FIX_M         L0C free, after the Fixpipe
//   AIVk -> AIC  kFlagPanelFree   landing panel free, after the FP32 cast.
//                              A back edge in the marked graph but *not* a
//                              reused flag id: each release carries its own
//                              id, so the ring's two slots cost six ids per
//                              core rather than two.
//   AIVk intra   MTE3_V        packed page slot free, after the MTE3 burst

#include <cstdint>
#include <utility>

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
// The resident-block ingress. At 16 bits a row-major 32 x 32 block is no
// longer its own Nz image, so the blocks are converted on the way into L1
// rather than copied flat; see the prologue of run_cube_core and
// reports/micro_layout_bf16.md.
#include "routines/routine_stage_page_zn_bf16.hpp"
#endif

// The sparse-sampling predicate the CAModel probe used (ASCEND_MICRO_SPARSE_SAMPLING)
// is not carried into this tree. It exists so a verification harness and the device
// agree, block for block, about which blocks execute, and there is no such harness
// here; this is a production build only, so the branch and its include are gone
// rather than guarded. The kernel body is otherwise the file the CAModel runs
// verified, with the in-page addressing generalisation noted at run_vector_core.

namespace turboquant_bf16 {

// ------------------------------------------------------------ strip geometry
constexpr std::uint32_t kTokens = 16;   // tokens per strip == blockSize
constexpr std::uint32_t kHeads = 4;     // numKvHeads
constexpr std::uint32_t kHeadSize = 256;
constexpr std::uint32_t kGroup = 32;    // channels per Cube group == INT8 K0
constexpr std::uint32_t kGroups = kHeadSize / kGroup;  // 8, order of the outer factor
constexpr std::uint32_t kChunks = 2u * kHeads;         // 4 value chunks then 4 key chunks
constexpr std::uint32_t kPanel = kTokens * kHeadSize;  // 4096 elements per chunk
constexpr std::uint32_t kPackedBytes = kPanel / 2u;    // 2048 bytes of nibbles
constexpr std::uint32_t kScaleFloats = kTokens;        // one scale per token

// ----------------------------------------------------- element width, arch35
// This file is a clone of the INT8 kernel with one activation element widened
// from 1 byte to 2. That makes a distinction load bearing which the INT8 file
// could afford to blur: an element count and a byte count were the same number
// there and are not here. Raw L1 and Unified Buffer addresses are *byte*
// offsets, while an offset added to a __cbuf__/__ca__/__cb__ bfloat16_t*
// advances *elements*. Every constant below says which it is in its name.
constexpr std::uint32_t kElemBytes = 2u;  // sizeof(bfloat16_t)
//
// The fractal is 16 x (32B / sizeof(T)), 512 bytes
// (cube_compute_fractal_intro/key_fractal_format_detailed_explanation.md), so
// the INT8 K0 of 32 becomes 16 at 16 bits and a 32-wide group is TWO fractals
// rather than one.
//
// kGroup stays 32 regardless. It is the width of the *algorithm's* Hadamard
// group, not of the hardware's contraction step, and keeping it at 32 leaves
// header equation (1), the 16 resident blocks and the host oracle byte for
// byte unchanged, with the chunk still 64 Mmads of k = 32. Dropping it to 16 to
// match K0 would refactor the transform to (H16 (x) I16) . diag(piSigns) .
// (I16 (x) H16): 32 new resident blocks, a second golden oracle, and 256 Mmads
// per chunk instead of 64. The layout machinery derives C0 from the element
// type -- MakeFrameLayout's LayoutTraitDefault<T> "specifies the data type T,
// with C0 calculated automatically" -- so a 32-wide group tile is described the
// same way it was and simply resolves to two fractals.
constexpr std::uint32_t kFractalK = 32u / kElemBytes;            // 16
constexpr std::uint32_t kFractalsPerGroup = kGroup / kFractalK;  // 2
static_assert(kFractalsPerGroup * kFractalK == kGroup,
              "a Hadamard group must be a whole number of hardware fractals");

// ------------------------------------------------------- rotation table shape
// One 32x32 bfloat16_t block per (sign, source group) pair: block s * kGroups +
// g is T[s][g] of equation (2) in the header. The sign is the only way the
// outer factor's H8[c1][g] enters, so 16 blocks cover all 64 (output group,
// source group) pairs of a chunk. The entries are +-1, exactly representable.
constexpr std::uint32_t kBlockElems = kGroup * kGroup;              // 1024
constexpr std::uint32_t kTableBlocks = 2u * kGroups;                // 16
constexpr std::uint32_t kTableElems = kTableBlocks * kBlockElems;   // 16384
constexpr std::uint32_t kTableBytes = kTableElems * kElemBytes;     // 32768
constexpr std::uint32_t kTableRows = kTableBlocks * kGroup;     // 512, as one ND blob

// Ring depths. The L1 ring is the pipeline depth; the per-core UB rings are
// half of it because each core owns every other chunk.
constexpr std::uint32_t kStageDepth = 4;
constexpr std::uint32_t kLandDepth = 2;
constexpr std::uint32_t kPackDepth = 2;

static_assert(kHeadSize == kGroups * kGroup);
static_assert(kTableRows * kGroup == kTableElems,
              "the table blob must be a whole number of 32-wide rows");
static_assert(kChunks / 2u == kLandDepth * 2u,
              "each core must wrap its landing ring at least twice");

}  // namespace turboquant_bf16

#if defined(__CCE_AICORE__)

// Mode-4 intra-block flags and a Fixpipe that picks a Vector core both need a
// non-1:1 AIC:AIV ratio. At 1:1 there is one Vector core per Cube core, the
// sub-block index is meaningless and the library reroutes cross-core traffic.
#if defined(__MIX_CORE_AIC_RATION__) && (__MIX_CORE_AIC_RATION__ == 1)
#error "MIX_AIC_1_2 is required: a 1:1 AIC:AIV ratio has no second Vector core"
#endif

namespace turboquant_bf16 {

// --------------------------------------------------------------- chunk traits
// Chunks 0..3 are the value heads, 4..7 the key heads, which is the order the
// four-stage schedule runs them in. Every trait is a compile-time value, so no
// column of the schedule contains a runtime predicate or a modulo.
template <std::uint32_t C>
constexpr bool kIsKey = (C >= kHeads);
template <std::uint32_t C>
constexpr std::uint32_t kChunkHead = C % kHeads;
template <std::uint32_t C>
constexpr std::uint32_t kOwner = C & 1u;
template <std::uint32_t C>
constexpr std::uint32_t kL1Slot = C % kStageDepth;
template <std::uint32_t C>
constexpr std::uint32_t kLandSlot = (C / 2u) % kLandDepth;
template <std::uint32_t C>
constexpr std::uint32_t kPackSlot = (C / 2u) % kPackDepth;

// H8[c1][g] = (-1)^popcount(c1 & g), so the block for (c1, g) is the one whose
// sign flag is that parity. Resolved at compile time, once per Mmad site.
constexpr std::uint32_t ParityOf(std::uint32_t v)
{
    std::uint32_t parity = 0;
    while (v != 0u) {
        parity ^= (v & 1u);
        v >>= 1;
    }
    return parity;
}
template <std::uint32_t C1, std::uint32_t G>
constexpr std::uint32_t kSignBlock = ParityOf(C1 & G) * kGroups + G;

// --------------------------------------------------------------- UB arena
// Rule AKA3003: every operand base 32-byte aligned, one DataBlock of bank skew
// between live slots, and VECTOR_REG_WIDTH of addressable UB behind the last
// operand because a level-2 vector op reads a whole vector register per source
// unmasked. One map serves both Vector cores, and the Cube forms its Fixpipe
// destination from the same constants.
constexpr std::uint32_t kBlock = 32u;
constexpr std::uint32_t kBankSkewBytes = 32u;
constexpr std::uint32_t kReadPadBytes = 256u;  // VECTOR_REG_WIDTH on dav-3510
constexpr std::uint32_t kUbWindow = 192u * 1024u;
constexpr std::uint32_t kLanesPerBlock = kBlock / sizeof(float);  // 8

constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}
constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkewBytes;
}

constexpr std::uint32_t kPanelBytes = kPanel * 4u;        // 16384, FP32 or INT32
constexpr std::uint32_t kLandSlotBytes = kPanelBytes + kBankSkewBytes;
constexpr std::uint32_t kPackSlotBytes =
    RoundUpBlock(kPackedBytes + kScaleFloats * 4u) + kBankSkewBytes;

// The Fixpipe landing ring stays at UB 0 so the Cube forms a raw __ubuf__
// pointer without depending on any Vector-side allocation.
constexpr std::uint32_t kUbLandRing = 0u;
constexpr std::uint32_t kUbLandBytes = kLandDepth * kLandSlotBytes;

// Working panels. Single buffered: reuse across chunks is a same-pipe Vector
// dependency, which the Vector queue orders in issue order, not a cross-engine
// hazard (guidelines section 3.2).
//   kUbCast   FP32 image of the INT32 landing panel, later the INT32 quantised
//             panel, which is the same 16 KiB read as 4-byte lanes
//   kUbWork   |Y|, then the value-layout fold in place, then Y * inv
//
// The factored revision also needed kUbWorkA/kUbWorkB as butterfly ping-pong
// panels and kUbSignKey/kUbSignValue as replicated sign panels. All four are
// gone: the rotation finishes in L0C.
constexpr std::uint32_t kUbCast = NextSlot(kUbLandRing, kUbLandBytes);
constexpr std::uint32_t kUbWork = NextSlot(kUbCast, kPanelBytes);

// Row-wise max ladder for the key layout, where the reduction runs along the
// fast axis. Two halving stages at the full 64-lane mask take 256 partials per
// token down to 64, and two BlockReduceMax passes take 64 down to 8 and 8 down
// to 1 - a DataBlock at a time, so no stage wastes most of a vector register
// the way the old mask-32/16/8 stages did. The value layout folds in place and
// needs none of these.
constexpr std::uint32_t kUbFold128 = NextSlot(kUbWork, kPanelBytes);
constexpr std::uint32_t kUbFold64 = NextSlot(kUbFold128, kTokens * 128u * 4u);
constexpr std::uint32_t kUbFold8 = NextSlot(kUbFold64, kTokens * 64u * 4u);

// FP16 staging for the nibble pack: the hardware Cast to int4b_t takes a
// 2-byte source. The staged values are exact integers in [-8, 7], so the
// INT32 -> FP16 -> INT4 chain rounds once, in the INT32 cast, and the two
// narrowing steps are exact. The INT16 source form compiles to a deprecated
// vcvt_s162s4 intrinsic, which is the only reason this goes through FP16.
constexpr std::uint32_t kUbHalf = NextSlot(kUbFold8, kTokens * 8u * 4u);

// Logically scalar state, one lane-replicated DataBlock each where the API
// reads whole blocks. kScaleFloats is 16 == two DataBlocks.
constexpr std::uint32_t kUbAbsMax = NextSlot(kUbHalf, kPanel * 2u);
constexpr std::uint32_t kUbExpBits = NextSlot(kUbAbsMax, kScaleFloats * 4u);
constexpr std::uint32_t kUbPow = NextSlot(kUbExpBits, kScaleFloats * 4u);
constexpr std::uint32_t kUbInv = NextSlot(kUbPow, kScaleFloats * 4u);
constexpr std::uint32_t kUbInvRep = NextSlot(kUbInv, kScaleFloats * 4u);
constexpr std::uint32_t kUbInvBcast = NextSlot(kUbInvRep, 4u * kScaleFloats * 4u);
constexpr std::uint32_t kUbOne = NextSlot(kUbInvBcast, kTokens * kLanesPerBlock * 4u);
constexpr std::uint32_t kUbExpMask = NextSlot(kUbOne, kScaleFloats * 4u);
constexpr std::uint32_t kUbInvSeed = NextSlot(kUbExpMask, kScaleFloats * 4u);

// Packed egress ring: 2 KiB of nibbles plus the 64-byte scale slice, so one
// back edge covers both MTE3 bursts of a chunk.
constexpr std::uint32_t kUbPackRing = NextSlot(kUbInvSeed, kScaleFloats * 4u);
constexpr std::uint32_t kArenaEnd = kUbPackRing + kPackDepth * kPackSlotBytes;

static_assert(kUbLandRing % kBlock == 0u && kUbCast % kBlock == 0u &&
                  kUbWork % kBlock == 0u && kUbFold128 % kBlock == 0u &&
                  kUbFold64 % kBlock == 0u && kUbFold8 % kBlock == 0u &&
                  kUbHalf % kBlock == 0u && kUbAbsMax % kBlock == 0u &&
                  kUbExpBits % kBlock == 0u && kUbPow % kBlock == 0u &&
                  kUbInv % kBlock == 0u &&
                  kUbInvRep % kBlock == 0u && kUbInvBcast % kBlock == 0u &&
                  kUbOne % kBlock == 0u && kUbExpMask % kBlock == 0u &&
                  kUbInvSeed % kBlock == 0u && kUbPackRing % kBlock == 0u,
              "every UB operand base must be 32-byte aligned");
static_assert(kLandSlotBytes % kBlock == 0u && kPackSlotBytes % kBlock == 0u,
              "every ring slot stride must be a whole number of DataBlocks");
static_assert(kLandSlotBytes % 256u != 0u && kPackSlotBytes % 256u != 0u,
              "a 256-byte ring stride puts every slot on the same bank phase");
static_assert(kArenaEnd + kReadPadBytes <= kUbWindow,
              "UB arena leaves no VECTOR_REG_WIDTH read pad behind the state");
static_assert(kUbInvSeed + kReadPadBytes <= kUbWindow,
              "the last state block must have a whole vector register behind it");
static_assert(kPackedBytes % kBlock == 0u && kScaleFloats * 4u % kBlock == 0u,
              "both MTE3 bursts must be whole 32-byte bursts");
static_assert(kTokens * kLanesPerBlock * 4u == kTokens * kBlock,
              "a Brcb repeat writes eight DataBlocks per source element");

template <std::uint32_t Slot>
constexpr std::uint32_t kLandBase = kUbLandRing + Slot * kLandSlotBytes;
template <std::uint32_t Slot>
constexpr std::uint32_t kPackBase = kUbPackRing + Slot * kPackSlotBytes;
template <std::uint32_t Slot>
constexpr std::uint32_t kScaleBase = kPackBase<Slot> + kPackedBytes;

// --------------------------------------------------------------- L1 arena
// The 16 rotation blocks are page invariant and reach L1 once, as one flat
// 16 KiB blob. The page ring has one slot per pipeline stage. L1 operands need
// 32-byte alignment; the 512-byte rule applies to L0A and L0B, and the fractal
// offsets inside a slot are multiples of 512 from a 32-byte aligned base,
// which is the layout the fused attention kernel validated.
constexpr std::uint32_t kL1Table = 0u;
constexpr std::uint32_t kL1PageElems = kTokens * kHeadSize;          // 4096
constexpr std::uint32_t kL1PageBytes = kL1PageElems * kElemBytes;    // 8192
constexpr std::uint32_t kL1Ring = NextSlot(kL1Table, kTableBytes);
constexpr std::uint32_t kL1SlotBytes = kL1PageBytes + kBankSkewBytes;
constexpr std::uint32_t kL1End = kL1Ring + kStageDepth * kL1SlotBytes;
// Element stride for a typed L1 pointer, and byte stride for a raw L1 address.
// Both are needed: the group's L1 base is computed as an integer byte address
// before the reinterpret_cast, while an L0 operand offset is added to a typed
// pointer and therefore counts elements.
constexpr std::uint32_t kFractalElems = kTokens * kGroup;            // 512
constexpr std::uint32_t kFractalBytes = kFractalElems * kElemBytes;  // 1024

static_assert(kL1Table % kBlock == 0u && kL1Ring % kBlock == 0u &&
                  kL1SlotBytes % kBlock == 0u,
              "every L1 operand base must be 32-byte aligned");
static_assert(kL1SlotBytes % 256u != 0u, "L1 slot stride collapses the bank phase");
// 32 KiB of resident blocks plus four 8 KiB page slots is about 64 KiB, so the
// doubled footprint is nowhere near the budget. The assert stays at the figure
// the INT8 kernel used for this part.
static_assert(kL1End <= 512u * 1024u, "L1 ring exceeds the L1 buffer");
static_assert(kFractalElems * kGroups == kL1PageElems,
              "the page must be a whole number of Hadamard groups");

template <std::uint32_t Slot>
constexpr std::uint32_t kL1PageBase = kL1Ring + Slot * kL1SlotBytes;

// --------------------------------------------------------------- L0 arena
// The 16 rotation blocks stay resident in both L0A and L0B: the key chunks
// take them as the B operand and the value chunks as the A operand, and
// nothing writes that region after the prologue. The page fractals land behind
// it. 16 KiB of table plus 4 KiB of page is 20 KiB of the 64 KiB each of L0A
// and L0B holds on arch35.
constexpr std::uint32_t kL0TableElems = kTableElems;              // 16384
constexpr std::uint32_t kL0PageElems = kGroups * kFractalElems;   // 4096
constexpr std::uint32_t kL0Elems = kL0TableElems + kL0PageElems;  // 20480 elements
constexpr std::uint32_t kL0CElems = kGroups * kTokens * kGroup;   // 4096 FP32
// The 512-byte rules are about bytes, so they take the element width now.
// 32 KiB of resident blocks plus 8 KiB of page is 40 KiB of the 64 KiB each of
// L0A and L0B holds, against 20 KiB for INT8.
static_assert(kL0TableElems * kElemBytes % 512u == 0u,
              "the L0 page region must stay 512-byte aligned");
static_assert(kBlockElems * kElemBytes % 512u == 0u,
              "every resident block must be 512-byte aligned");
static_assert(kL0Elems * kElemBytes <= 64u * 1024u, "L0A/L0B hold 64 KiB on arch35");

// ------------------------------------------------- cross-core flag ids
// Mode 4 ids are per sub block: 0..15 belong to AIV0 and 16..31 to AIV1, which
// is how a participant names the *other* sub block's channel in both
// directions. 13, 14 and their +16 images are library reserved.
//
// **One id per cross-core event, never reused.** The first revision used one
// id per ring slot - Owner * 16 + Slot for "panel ready" - so each id carried
// two announcements over the eight chunks. That is not a token a wait
// consumes: the trace caught AIV1's *second* wait on its slot-1 id passing
// 2179 cycles before the announcement it was waiting for, released by the set
// its *first* wait had supposedly taken, and the panel it then read was 87%
// the previous occupant of that landing slot. The mechanism is untested -
// whether the flag is a level bit a wait tests without clearing, or something
// subtler - so this revision stops depending on it and labels every event
// uniquely instead. Guidelines section 6.2c.
//
// The first revision passed anyway, twice, because its Cube finished all eight
// chunks by cycle 6121 of a 19331-cycle run: a stale "ready" always pointed at
// a panel that had in fact been written long before. Balancing the pipeline is
// what made the bug reachable.
//
// Per Vector core, over its four chunks j = Chunk / 2:
//   productReady  Owner*16 + j            4 ids, j = 0..3, one per chunk
//   panelFree     Owner*16 + 4 + j        the Cube's wait for chunk j
//   panelFree     Owner*16 + 4 + j + 2    the AIV's release after chunk j
// so ids 4 and 5 are the two priming tokens, 6 and 7 the releases the Cube
// consumes for its last two chunks, and 8 and 9 the releases the epilogue
// drains. Ten ids per core, every one set exactly once and waited exactly
// once, all clear of 11..14.
template <std::uint32_t Chunk>
constexpr std::uint16_t kFlagProductReady =
    static_cast<std::uint16_t>(kOwner<Chunk> * 16u + Chunk / 2u);
template <std::uint32_t Owner, std::uint32_t Index>
constexpr std::uint16_t kFlagPanelFree =
    static_cast<std::uint16_t>(Owner * 16u + 4u + Index);
template <std::uint32_t Chunk>
constexpr std::uint16_t kFlagPanelFreeWait = kFlagPanelFree<kOwner<Chunk>, Chunk / 2u>;
template <std::uint32_t Chunk>
constexpr std::uint16_t kFlagPanelFreeSet =
    kFlagPanelFree<kOwner<Chunk>, Chunk / 2u + kLandDepth>;

static_assert(4u + kChunks / 2u + kLandDepth <= 11u,
              "every mode-4 flag id must stay below the library's reserved 11");
static_assert(kFlagPanelFreeSet<0u> == kFlagPanelFreeWait<2u * kLandDepth>,
              "the release after a chunk must be the token the Cube waits for "
              "when that landing slot comes round again");

// Intra-core HardEvent ids. The table-copy barriers finish before slot 0 is
// primed, so their route-local ID 0 cannot collide with the ring's slot 0.
constexpr std::int32_t kEvtL0 = 0;
constexpr std::int32_t kEvtEpilogue = 0;

// ------------------------------------------------- compile-time column list
using ChunkSeq = std::make_integer_sequence<std::uint32_t, kChunks>;
using GroupSeq = std::make_integer_sequence<std::uint32_t, kGroups>;

// =========================================================== AIC: Cube core
// MTE2 ingress of the raw INT8 page into the L1 ring, then per chunk eight
// L1 -> L0 fractal loads, 64 accumulating INT8 Mmads and one Fixpipe into the
// owning Vector core's landing panel.
struct CubeEngine {
    __gm__ const bfloat16_t* key;
    __gm__ const bfloat16_t* value;
    std::uint32_t row_pitch;   // numKvHeads * headSize, elements
    std::uint32_t head_size;   // headSize, elements
};

// MTE2: one strided ND -> NZ burst per chunk. The source is the headSize-wide
// window of head h inside the kTokens x (numKvHeads * headSize) activation
// block, so the row pitch is a runtime stride and the window is expressed with
// MakePatternLayout rather than MakeFrameLayout, whose NDExt row pitch is
// pinned to the column count.
template <std::uint32_t Chunk>
__aicore__ inline void StagePage(CubeEngine& engine)
{
    using namespace AscendC::Te;
    constexpr std::uint32_t slot = kL1Slot<Chunk>;
    constexpr std::int32_t evt = static_cast<std::int32_t>(slot);

    __gm__ const bfloat16_t* base = kIsKey<Chunk> ? engine.key : engine.value;
    auto* src_ptr = const_cast<__gm__ bfloat16_t*>(base) +
                    kChunkHead<Chunk> * engine.head_size;
    auto* dst_ptr = reinterpret_cast<__cbuf__ bfloat16_t*>(kL1PageBase<slot>);

    auto shape = MakeShape(MakeShape(_1{}, kTokens), MakeShape(_1{}, kHeadSize));
    auto stride = MakeStride(MakeStride(_0{}, engine.row_pitch), MakeStride(_0{}, _1{}));
    auto src = MakeTensor(MakeMemPtr(src_ptr),
                          MakePatternLayout<NDExtLayoutPtn,
                                            LayoutTrait<AscendC::Std::ignore_t, _1>>(shape, stride));
    auto dst = MakeTensor(MakeMemPtr(dst_ptr),
                          MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTokens, kHeadSize));

    // The L1 slot is free once MTE1 has read the page it held kStageDepth
    // chunks ago. WaitFlag<MTE1_MTE2> stalls the MTE2 queue, which is the
    // queue that must not run ahead here: nothing else gates it.
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(evt);
    Copy(MakeCopy(CopyGM2L1{}), dst, src);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(evt);
}

// MTE1 for one channel group. Key and value chunks differ only in which L0
// buffer the page fractal belongs to, because the page is the A operand of the
// key chunks and the B operand of the value chunks.
template <std::uint32_t Chunk, std::uint32_t G>
__aicore__ inline void GemmGroupLoad(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b)
{
    using namespace AscendC::Te;
    constexpr std::uint32_t slot = kL1Slot<Chunk>;
    auto* page = reinterpret_cast<__cbuf__ bfloat16_t*>(kL1PageBase<slot> + G * kFractalBytes);

    if constexpr (kIsKey<Chunk>) {
        // A = the page group, NZ(kTokens, kGroup): A[t][i] = x[t][G*32 + i].
        auto src = MakeTensor(MakeMemPtr(page),
                              MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTokens, kGroup));
        auto dst = MakeTensor(MakeMemPtr(l0a + kL0TableElems + G * kFractalElems),
                              MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTokens, kGroup));
        Copy(MakeCopy(CopyL12L0A{}), dst, src);
    } else {
        // B = the page group, ZN(kGroup, kTokens): B[i][t] = x[t][G*32 + i],
        // the same bytes read with the transposed meaning.
        auto src = MakeTensor(MakeMemPtr(page),
                              MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTokens));
        auto dst = MakeTensor(MakeMemPtr(l0b + kL0TableElems + G * kFractalElems),
                              MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTokens));
        Copy(MakeCopy(CopyL12L0B{}), dst, src);
    }
}

// One accumulating Mmad: output channel group C1 takes the contribution of
// source channel group G through the resident block kSignBlock<C1, G>. G == 0
// initialises the L0C region and the other seven accumulate into it, so one
// L0C region ends up holding
//
//   sum_G  sum_i  T[s(C1,G)][G][c0][i] . x[t][G*32 + i]
//
// which is row C1*32 + c0 of M . x by equation (1). cmatrixInitVal is the
// compile-time flag that picks "C = A B" over "C += A B"; the Cube pipe is
// in-order, so nothing else orders the eight accumulations.
template <std::uint32_t Chunk, std::uint32_t C1, std::uint32_t G>
__aicore__ inline void GemmBlock(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                 __cc__ float* l0c)
{
    using namespace AscendC::Te;
    constexpr std::uint32_t block = kSignBlock<C1, G> * kBlockElems;
    constexpr bool init = (G == 0u);
    auto* c_ptr = l0c + C1 * kTokens * kGroup;

    if constexpr (kIsKey<Chunk>) {
        auto a = MakeTensor(MakeMemPtr(l0a + kL0TableElems + G * kFractalElems),
                            MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTokens, kGroup));
        auto b = MakeTensor(MakeMemPtr(l0b + block),
                            MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kGroup));
        auto c = MakeTensor(MakeMemPtr(c_ptr),
                            MakeFrameLayout<NZLayoutPtn, 16>(kTokens, kGroup));
        MmadParams mp(static_cast<std::uint16_t>(kTokens), static_cast<std::uint16_t>(kGroup),
                      static_cast<std::uint16_t>(kGroup), 0, init);
        Mmad(MakeMmad(MmadOperation{}, MmadTraitDefault{}).with(mp), c, a, b);
    } else {
        auto a = MakeTensor(MakeMemPtr(l0a + block),
                            MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kGroup, kGroup));
        auto b = MakeTensor(MakeMemPtr(l0b + kL0TableElems + G * kFractalElems),
                            MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTokens));
        auto c = MakeTensor(MakeMemPtr(c_ptr),
                            MakeFrameLayout<NZLayoutPtn, 16>(kGroup, kTokens));
        MmadParams mp(static_cast<std::uint16_t>(kGroup), static_cast<std::uint16_t>(kTokens),
                      static_cast<std::uint16_t>(kGroup), 0, init);
        Mmad(MakeMmad(MmadOperation{}, MmadTraitDefault{}).with(mp), c, a, b);
    }
}

// One Fixpipe per chunk. The whole rotation is finished in L0C, so the panel
// leaves as a single L0C -> UB transfer: NZ(16, 256) -> ND for the token-major
// key layout and NZ(256, 16) -> ND for the channel-major value layout. The
// factored revision needed one Fixpipe per channel group, with a strided ND
// destination for the key chunks, and those eight transfers plus their
// MOV_SPR_XN configuration writes were the Cube's largest single cost.
template <std::uint32_t Chunk>
__aicore__ inline void FixpipeChunk(__cc__ float* l0c)
{
    using namespace AscendC::Te;
    constexpr std::uint32_t land = kLandSlot<Chunk>;
    auto* ub = reinterpret_cast<__ubuf__ float*>(kLandBase<land>);

    FixpipeParams fp;
    // No unit flag: M_FIX carries the L0C dependency explicitly, because a
    // group of 64 accumulating Mmads feeds this one transfer.
    fp.unitFlag = 0;
    fp.subBlockId = (kOwner<Chunk> == 1u);

    if constexpr (kIsKey<Chunk>) {
        auto c = MakeTensor(MakeMemPtr(l0c),
                            MakeFrameLayout<NZLayoutPtn, 16>(kTokens, kHeadSize));
        auto y = MakeTensor(MakeMemPtr(ub),
                            MakeFrameLayout<NDExtLayoutPtn>(kTokens, kHeadSize));
        Copy(MakeCopy(CopyL0C2UB{}).with(fp), y, c);
    } else {
        auto c = MakeTensor(MakeMemPtr(l0c),
                            MakeFrameLayout<NZLayoutPtn, 16>(kHeadSize, kTokens));
        auto y = MakeTensor(MakeMemPtr(ub),
                            MakeFrameLayout<NDExtLayoutPtn>(kHeadSize, kTokens));
        Copy(MakeCopy(CopyL0C2UB{}).with(fp), y, c);
    }
}

template <std::uint32_t Chunk, std::uint32_t... Gs>
__aicore__ inline void GemmGroupLoads(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                      std::integer_sequence<std::uint32_t, Gs...>)
{
    (GemmGroupLoad<Chunk, Gs>(l0a, l0b), ...);
}

template <std::uint32_t Chunk, std::uint32_t C1, std::uint32_t... Gs>
__aicore__ inline void GemmBlockRow(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                    __cc__ float* l0c,
                                    std::integer_sequence<std::uint32_t, Gs...>)
{
    (GemmBlock<Chunk, C1, Gs>(l0a, l0b, l0c), ...);
}

template <std::uint32_t Chunk, std::uint32_t... C1s>
__aicore__ inline void GemmBlockRows(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                     __cc__ float* l0c,
                                     std::integer_sequence<std::uint32_t, C1s...>)
{
    (GemmBlockRow<Chunk, C1s>(l0a, l0b, l0c, GroupSeq{}), ...);
}

template <std::uint32_t Chunk>
__aicore__ inline void GemmChunk(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                 __cc__ float* l0c)
{
    constexpr std::uint32_t slot = kL1Slot<Chunk>;

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<std::int32_t>(slot));
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
    GemmGroupLoads<Chunk>(l0a, l0b, GroupSeq{});
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtL0);
    // Issued on MTE1 and behind the eight fractal loads, so the page slot is
    // announced free only once they have read it.
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<std::int32_t>(slot));

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtL0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEvtL0);
    AscendC::SetMMRowMajor();
    GemmBlockRows<Chunk>(l0a, l0b, l0c, GroupSeq{});
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEvtL0);

    // The landing panel is waited here rather than before the Mmads: only the
    // Fixpipe writes Unified Buffer, so the 64 multiplies for this chunk run
    // while the owning Vector core is still finishing the previous one. A
    // cross-core wait stalls the scalar stream, so everything issued *after*
    // it is behind the flag and everything before it is not.
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagPanelFreeWait<Chunk>);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEvtL0);
    FixpipeChunk<Chunk>(l0c);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEvtL0);

    // A mode-4 set runs on the scalar stream, so drain the Fixpipe into it
    // first; without this the announcement overtakes the transfer it reports
    // (guidelines section 6.2a).
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtL0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtL0);
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(kFlagProductReady<Chunk>);
}

// Column j of the schedule: MTE2 stages chunk j while the Cube multiplies
// chunk j - 1. Neither depends on the other, so MTE2, MTE1, Cube and Fixpipe
// are all free to be in flight together.
template <std::uint32_t Column>
__aicore__ inline void StepColumnCube(CubeEngine& engine, __ca__ bfloat16_t* l0a,
                                      __cb__ bfloat16_t* l0b, __cc__ float* l0c)
{
    if constexpr (Column < kChunks) StagePage<Column>(engine);
    if constexpr (Column >= 1u) GemmChunk<Column - 1u>(l0a, l0b, l0c);
}

template <std::uint32_t... Columns>
__aicore__ inline void StepColumnsCube(CubeEngine& engine, __ca__ bfloat16_t* l0a,
                                       __cb__ bfloat16_t* l0b, __cc__ float* l0c,
                                       std::integer_sequence<std::uint32_t, Columns...>)
{
    (StepColumnCube<Columns>(engine, l0a, l0b, l0c), ...);
}

__aicore__ inline void run_cube_core(__gm__ const bfloat16_t* key,
                                     __gm__ const bfloat16_t* value,
                                     __gm__ const std::int32_t* tables,
                                     std::uint32_t num_kv_heads,
                                     std::uint32_t head_size)
{
    using namespace AscendC::Te;
    __ca__ bfloat16_t l0_a[kL0Elems] __attribute__((aligned(512)));
    __cb__ bfloat16_t l0_b[kL0Elems] __attribute__((aligned(512)));
    __cc__ float l0_c[kL0CElems] __attribute__((aligned(1024)));

    CubeEngine engine{key, value, num_kv_heads * head_size, head_size};

    // The 16 rotation blocks reach L1 once, converted ND -> Nz one block at a
    // time, and each of L0A and L0B then takes the whole blob once as a byte
    // identity.
    //
    // The INT8 kernel copied the blob in flat, because at C0 = 32 a row-major
    // 32 x 32 block IS its own Nz(32, 32) image: the fractal is a whole 32-wide
    // row and the grid is one fractal along K. At C0 = 16 both grid axes are
    // live and the image the Cube reads is
    //
    //   offset = (c/16)*512 + (r/16)*256 + (r%16)*16 + (c%16)   holds W[r][c]
    //
    // -- four 16 x 16 sub-blocks, each row-major, ordered column-block major --
    // against row-major's r*32 + c. A flat ingress therefore hands the Cube a
    // permuted operand in BOTH roles, so the key chunks are as wrong as the
    // value chunks. tests/camodel/test_micro_layout_bf16.cpp reproduces that
    // misread bit-exactly from the index map and then shows this ingress
    // removing it.
    //
    // MTE1 cannot repair it downstream: the L1 -> L0 path's copy unit is the
    // 512-byte fractal and its only rearrangement is an intra-fractal
    // transpose, so ingress is the last stage that can reorder sub-fractals.
    //
    // Block b still lands at b * kBlockElems, so every operand base below and
    // the GM contract for `tables` (16 row-major 32 x 32 blocks) are unchanged.
    // The two L1 -> L0 copies stay identities for a different reason than the
    // INT8 file gave: source and destination carry the same layout descriptor
    // over the same kTableElems elements, so element i moves to element i
    // whatever order the bytes are in. Inside the blob, block b then reads as
    // NZ(32, 32) from L0A and as ZN(32, 32) from L0B -- the same image, since
    // those two differ only in the name of the second index -- which are the
    // two operand roles equation (1) needs.
    auto* l1_table = reinterpret_cast<__cbuf__ bfloat16_t*>(kL1Table);
    ascend_micro::stage_blocks_nz<kTableBlocks, kGroup>(
        reinterpret_cast<__gm__ const bfloat16_t*>(tables), kL1Table);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtEpilogue);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtEpilogue);

    auto ta_src = MakeTensor(MakeMemPtr(l1_table),
                             MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTableRows, kGroup));
    auto ta_dst = MakeTensor(MakeMemPtr(l0_a),
                             MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTableRows, kGroup));
    Copy(MakeCopy(CopyL12L0A{}), ta_dst, ta_src);
    auto tb_src = MakeTensor(MakeMemPtr(l1_table),
                             MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTableRows));
    auto tb_dst = MakeTensor(MakeMemPtr(l0_b),
                             MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTableRows));
    Copy(MakeCopy(CopyL12L0B{}), tb_dst, tb_src);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtEpilogue);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtEpilogue);

    // Feedback cycle priming: every L1 page slot starts free, and L0A/L0B and
    // L0C are free for the first chunk.
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(3);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEvtL0);

    StepColumnsCube(engine, l0_a, l0_b, l0_c,
                    std::make_integer_sequence<std::uint32_t, kChunks + 1u>{});

    // Drain every primed token and every panel-free token the Vector cores
    // handed back, so no event bit and no intra-block flag leaks into the next
    // launch. These waits also retire the last Fixpipe before the kernel ends.
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(3);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEvtL0);
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagPanelFree<0u, kChunks / 2u>);
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagPanelFree<0u, kChunks / 2u + 1u>);
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagPanelFree<1u, kChunks / 2u>);
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagPanelFree<1u, kChunks / 2u + 1u>);
}

// ======================================================= AIV: Vector cores
// Both cores run this body; the chunk list is filtered by owner, so each core
// handles two value chunks and two key chunks and nothing else. The rotation
// arrives finished, so the stage is absmax, a power-of-two scale, and the INT4
// pack - nothing else.
struct VectorEngine {
    AscendC::GlobalTensor<std::int8_t> key_cache;
    AscendC::GlobalTensor<std::int8_t> value_cache;
    AscendC::GlobalTensor<float> scale_cache;
    std::uint32_t page_bytes;    // byte base of (block, head 0) in either cache
    std::uint32_t head_bytes;    // blockSize * headSize / 2
    // Byte offset of this strip *inside* its (block, head) page. Zero when a
    // strip is the whole page (blockSize == kTokens); otherwise the two cache
    // layouts place the strip differently, so each plane carries its own offset.
    // See run_vector_core for the derivation.
    std::uint32_t key_token_bytes;    // keyCache:   token-major, kTokens * headSize / 2 per strip
    std::uint32_t value_token_bytes;  // valueCache: strip-blocked, kPackedBytes per strip
    std::uint32_t key_scale;     // float base of (kv 0, block, head 0) + token offset
    std::uint32_t value_scale;   // float base of (kv 1, block, head 0) + token offset
    std::uint32_t head_scale;    // blockSize
    float inv_sqrt_len;
};

constexpr auto kUb = AscendC::TPosition::VECCALC;

template <class T>
__aicore__ inline AscendC::LocalTensor<T> View(std::uint32_t base, std::uint32_t count)
{
    return AscendC::LocalTensor<T>(kUb, base, count);
}

// Row-wise max ladder for the key layout: fold N partials per token down to
// N/2, from a SrcPitch-strided source into a DstPitch-strided destination.
// Both uses run at the full 64-lane mask.
template <std::uint32_t N, std::uint32_t SrcPitch, std::uint32_t DstPitch,
          std::uint32_t... Cs>
__aicore__ inline void KeyFoldEmit(const AscendC::LocalTensor<float>& dst,
                                   const AscendC::LocalTensor<float>& src,
                                   std::integer_sequence<std::uint32_t, Cs...>)
{
    constexpr std::uint32_t half = N / 2u;
    constexpr std::int32_t mask = static_cast<std::int32_t>(half > 64u ? 64u : half);
    AscendC::BinaryRepeatParams rp;
    rp.dstBlkStride = 1;
    rp.src0BlkStride = 1;
    rp.src1BlkStride = 1;
    rp.dstRepStride = static_cast<std::uint8_t>(DstPitch / kLanesPerBlock);
    rp.src0RepStride = static_cast<std::uint8_t>(SrcPitch / kLanesPerBlock);
    rp.src1RepStride = rp.src0RepStride;
    (AscendC::Max<float>(dst[Cs * mask], src[Cs * mask], src[half + Cs * mask], mask,
                         static_cast<std::uint8_t>(kTokens), rp),
     ...);
}

template <std::uint32_t N, std::uint32_t SrcPitch, std::uint32_t DstPitch>
__aicore__ inline void KeyFold(const AscendC::LocalTensor<float>& dst,
                               const AscendC::LocalTensor<float>& src)
{
    constexpr std::uint32_t half = N / 2u;
    constexpr std::uint32_t mask = half > 64u ? 64u : half;
    static_assert(mask == 64u, "both key folds must run at the full vector mask");
    static_assert(DstPitch >= kLanesPerBlock, "a fold destination pitch is a block count");
    KeyFoldEmit<N, SrcPitch, DstPitch>(
        dst, src, std::make_integer_sequence<std::uint32_t, half / mask>{});
}

// In-place halving fold for the value layout, where the reduction axis is the
// slow one and every step is one contiguous level-2 op.
template <std::uint32_t... Steps>
__aicore__ inline void ValueFoldEmit(const AscendC::LocalTensor<float>& buf,
                                     std::integer_sequence<std::uint32_t, Steps...>)
{
    // Step s folds kPanel >> s elements onto kPanel >> (s + 1).
    (AscendC::Max(buf, buf, buf[kPanel >> (Steps + 1u)],
                  static_cast<std::int32_t>(kPanel >> (Steps + 1u))),
     ...);
}

// One level-0 op per vector register of a token row: the key panel has the
// token index outermost, so the per-token inverse block advances once per
// repeat (src1RepStride 1) and is read by every DataBlock of that repeat
// (src1BlkStride 0).
template <std::uint32_t... Qs>
__aicore__ inline void KeyQuantiseEmit(const AscendC::LocalTensor<float>& dst,
                                       const AscendC::LocalTensor<float>& src,
                                       const AscendC::LocalTensor<float>& inv_bcast,
                                       const AscendC::BinaryRepeatParams& rp,
                                       std::integer_sequence<std::uint32_t, Qs...>)
{
    (AscendC::Mul<float>(dst[Qs * 64u], src[Qs * 64u], inv_bcast, 64,
                         static_cast<std::uint8_t>(kTokens), rp),
     ...);
}

// --- the Vector stage --------------------------------------------------------
template <std::uint32_t Chunk>
__aicore__ inline void RotateAndQuantise(VectorEngine& engine)
{
    constexpr std::uint32_t land = kLandSlot<Chunk>;
    constexpr std::uint32_t pack = kPackSlot<Chunk>;
    constexpr std::int32_t pack_evt = static_cast<std::int32_t>(pack);
    constexpr std::int32_t all = static_cast<std::int32_t>(kPanel);

    auto landing = View<float>(kLandBase<land>, kPanel);
    auto cast = View<float>(kUbCast, kPanel);
    auto work = View<float>(kUbWork, kPanel);

    AscendC::CrossCoreWaitFlag<4, PIPE_V>(kFlagProductReady<Chunk>);
    // BF16 x BF16 accumulates to FP32 in L0C, so the panel lands as FP32 and
    // the INT8 kernel's INT32 -> FP32 cast is not needed. What is still needed
    // is to get the panel *out* of the landing ring slot in one pass, so the
    // slot can be released immediately and the Cube can refill it; this is that
    // copy, expressed as a multiply by one so it stays a single contiguous
    // Vector op on the same pipe the cast used.
    //
    // Exactness is preserved for integer-valued BF16 activations, which is what
    // the fixture supplies: a BF16 significand is 8 bits, so integers up to 256
    // are exact, a product of two is at most 2^16 and exact in FP32, and a sum
    // of 256 such products is an integer below 2^24 -- exact whatever order the
    // Cube accumulates in. With general BF16 inputs that order is unspecified
    // and no integer oracle could reproduce it bit for bit.
    AscendC::Muls(cast, landing, 1.0f, all);
    // The landing panel is dead from here on, so release it before the rest of
    // the stage runs and let the Cube refill it immediately. The drain makes
    // the scalar-stream set wait for the Vector queue to retire the copy.
    AscendC::SetFlag<AscendC::HardEvent::V_S>(pack_evt);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(pack_evt);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagPanelFreeSet<Chunk>);

    // Row-wise absolute maximum, one value per token, left contiguous so the
    // scale slice is a single 64-byte MTE3 burst.
    auto abs_max = View<float>(kUbAbsMax, kScaleFloats);
    auto one = View<float>(kUbOne, kScaleFloats);
    AscendC::Abs(work, cast, all);
    if constexpr (kIsKey<Chunk>) {
        auto f128 = View<float>(kUbFold128, kTokens * 128u);
        auto f64 = View<float>(kUbFold64, kTokens * 64u);
        auto f8 = View<float>(kUbFold8, kTokens * 8u);
        KeyFold<256u, 256u, 128u>(f128, work);
        KeyFold<128u, 128u, 64u>(f64, f128);
        // 64 partials per token remain, eight DataBlocks each, and
        // BlockReduceMax writes one result per block contiguously: 64 -> 8 at
        // one repeat per token, then 8 -> 1 over the two registers that holds.
        AscendC::BlockReduceMax<float>(f8, f64, static_cast<std::uint8_t>(kTokens), 64,
                                       1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float>(abs_max, f8, 2, 64, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max(abs_max, abs_max, one, static_cast<std::int32_t>(kScaleFloats));
    } else {
        ValueFoldEmit(work, std::make_integer_sequence<std::uint32_t, 8u>{});
        AscendC::PipeBarrier<PIPE_V>();
        // Floor at 1.0 and copy out of the fold buffer in one op: every value
        // is an integer, so 1.0 is the smallest non-zero magnitude and an
        // all-zero token gets scale 2^-2 rather than a division by zero.
        AscendC::Max(abs_max, work, one, static_cast<std::int32_t>(kScaleFloats));
    }

    // Power-of-two scale, by integer arithmetic on the FP32 bit pattern. A
    // float 2^k carries the exponent field k + 127, so with
    // 2^e <= absmax < 2^(e+1):
    //   exp_bits  = absmax & 0x7F800000       -> 2^e,       field e + 127
    //   pow_bits  = exp_bits - (2 << 23)      -> 2^(e-2),   field e + 125
    //   inv_bits  = (254 << 23) - pow_bits    -> 2^(2-e),   field 129 - e
    // The last line is the exact reciprocal of a power of two: the two fields
    // sum to 254, which is what (254 << 23) - x computes. No division, no
    // logarithm, and no hardware reciprocal rounding enters the contract.
    auto exp_bits = View<std::int32_t>(kUbExpBits, kScaleFloats);
    auto exp_mask = View<std::int32_t>(kUbExpMask, kScaleFloats);
    auto inv_seed = View<std::int32_t>(kUbInvSeed, kScaleFloats);
    auto pow_bits = View<std::int32_t>(kUbPow, kScaleFloats);
    auto pow_two = View<float>(kUbPow, kScaleFloats);
    auto inv_bits = View<std::int32_t>(kUbInv, kScaleFloats);
    auto inv = View<float>(kUbInv, kScaleFloats);
    constexpr std::int32_t lanes = static_cast<std::int32_t>(kScaleFloats);

    AscendC::And(exp_bits, abs_max.template ReinterpretCast<std::int32_t>(), exp_mask, lanes);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(pow_bits, exp_bits, static_cast<std::int32_t>(-0x01000000), lanes);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(inv_bits, inv_seed, pow_bits, lanes);

    // Broadcast inv along the panel's fast axis and quantise. The value layout
    // has the token index innermost, so one level-0 op with a repeat-invariant
    // second operand covers the whole panel; the key layout has the token index
    // outermost, so the per-token block advances once per repeat and a repeat
    // covers a quarter of a token row.
    auto quant = View<std::int32_t>(kUbCast, kPanel);
    auto staged = View<half>(kUbHalf, kPanel);
    auto packed = View<std::int8_t>(kPackBase<pack>, kPackedBytes);
    AscendC::BinaryRepeatParams bc;
    bc.dstBlkStride = 1;
    bc.src0BlkStride = 1;
    bc.dstRepStride = 8;
    bc.src0RepStride = 8;
    if constexpr (kIsKey<Chunk>) {
        auto inv_bcast = View<float>(kUbInvBcast, kTokens * kLanesPerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(inv_bcast, inv, 2, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        bc.src1BlkStride = 0;
        bc.src1RepStride = 1;
        bc.dstRepStride = static_cast<std::uint8_t>(kHeadSize / kLanesPerBlock);
        bc.src0RepStride = bc.dstRepStride;
        KeyQuantiseEmit(work, cast, inv_bcast, bc,
                        std::make_integer_sequence<std::uint32_t, kHeadSize / 64u>{});
    } else {
        // Four copies of the 16-token block fill one vector register, and the
        // register repeats unchanged across the whole panel.
        auto inv_rep = View<float>(kUbInvRep, 4u * kScaleFloats);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(inv_rep, inv, 1.0f, lanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(inv_rep[kScaleFloats], inv_rep, 1.0f, lanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(inv_rep[2u * kScaleFloats], inv_rep, 1.0f,
                      static_cast<std::int32_t>(2u * kScaleFloats));
        AscendC::PipeBarrier<PIPE_V>();
        bc.src1BlkStride = 1;
        bc.src1RepStride = 0;
        AscendC::Mul<float>(work, cast, inv_rep, 64,
                            static_cast<std::uint8_t>(kPanel / 64u), bc);
    }

    // |Y * inv| < 8 and round-to-nearest-even can reach +8, which INT4 cannot
    // hold, so clamp from above only; -8 is in range. The INT32 cast is the
    // only rounding in the whole path, and it rounds an exactly representable
    // product, so the host oracle reproduces it in integer arithmetic.
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mins(work, work, 7.0f, all);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(quant, work, AscendC::RoundMode::CAST_RINT, kPanel);

    // The packed slot and its scale slice are one ring slot, so one back edge
    // covers both: its previous occupant is still being read by MTE3 until now.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pack_evt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(staged, quant, AscendC::RoundMode::CAST_RINT, kPanel);
    AscendC::PipeBarrier<PIPE_V>();
    auto nibbles = packed.template ReinterpretCast<AscendC::int4b_t>();
    AscendC::Cast(nibbles, staged, AscendC::RoundMode::CAST_RINT, kPanel);
    // The quantised panel is scale invariant - scaling Y and absmax together
    // leaves every nibble unchanged - so invSqrtLen folds into the stored scale
    // alone, and q * scale reconstructs the orthonormal rotation.
    auto scale = View<float>(kScaleBase<pack>, kScaleFloats);
    AscendC::Muls(scale, pow_two, engine.inv_sqrt_len, lanes);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pack_evt);
}

// --- the MTE3 stage ----------------------------------------------------------
template <std::uint32_t Chunk>
__aicore__ inline void StoreChunk(VectorEngine& engine)
{
    constexpr std::uint32_t pack = kPackSlot<Chunk>;
    constexpr std::int32_t pack_evt = static_cast<std::int32_t>(pack);
    auto packed = View<std::int8_t>(kPackBase<pack>, kPackedBytes);
    auto scale = View<float>(kScaleBase<pack>, kScaleFloats);

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pack_evt);
    const std::uint32_t page = engine.page_bytes + kChunkHead<Chunk> * engine.head_bytes +
                               (kIsKey<Chunk> ? engine.key_token_bytes : engine.value_token_bytes);
    const std::uint32_t slice =
        (kIsKey<Chunk> ? engine.key_scale : engine.value_scale) +
        kChunkHead<Chunk> * engine.head_scale;
    if constexpr (kIsKey<Chunk>) {
        AscendC::DataCopy(engine.key_cache[page], packed, kPackedBytes);
    } else {
        AscendC::DataCopy(engine.value_cache[page], packed, kPackedBytes);
    }
    AscendC::DataCopy(engine.scale_cache[slice], scale, kScaleFloats);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pack_evt);
}

// Column j of the schedule on Vector core k: the quantisation of chunk j - 2
// and the egress of chunk j - 3, each only if core k owns that chunk. Both are
// emitted at compile time, so a core's instruction stream contains only its
// own chunks and no predicate.
template <std::uint32_t Column, std::uint32_t Core>
__aicore__ inline void StepColumnVector(VectorEngine& engine)
{
    if constexpr (Column >= 2u && Column - 2u < kChunks &&
                  kOwner<(Column >= 2u ? Column - 2u : 0u)> == Core) {
        RotateAndQuantise<Column - 2u>(engine);
    }
    if constexpr (Column >= 3u && Column - 3u < kChunks &&
                  kOwner<(Column >= 3u ? Column - 3u : 0u)> == Core) {
        StoreChunk<Column - 3u>(engine);
    }
}

template <std::uint32_t Core, std::uint32_t... Columns>
__aicore__ inline void StepColumnsVector(VectorEngine& engine,
                                         std::integer_sequence<std::uint32_t, Columns...>)
{
    (StepColumnVector<Columns, Core>(engine), ...);
}

template <std::uint32_t Core>
__aicore__ inline void run_vector_core(__gm__ std::int8_t* key_cache,
                                       __gm__ std::int8_t* value_cache,
                                       __gm__ float* scale_cache,
                                       __gm__ const std::int32_t* slot_mapping,
                                       std::uint32_t num_kv_heads,
                                       std::uint32_t head_size,
                                       std::uint32_t block_size,
                                       std::uint32_t num_blocks,
                                       float inv_sqrt_len)
{
    // One scalar read of caller-supplied Global Memory metadata, before the
    // pipeline. A strip is kTokens consecutive slots, so its page index and its
    // offset within that page are all the schedule needs and no stage reads the
    // mapping again.
    //
    // Where the strip sits inside its page
    // ------------------------------------
    // The CAModel probe ran blockSize == kTokens == 16, where a strip *is* a
    // whole page and every in-page offset is zero. A vLLM page is blockSize 128
    // (any multiple of 16), so one page holds blockSize / kTokens strips and
    // each plane needs the strip's offset inside it. The three planes place it
    // differently, because their in-page index functions differ:
    //
    //   keyCache    nibble index = key * headSize + d. The strip's kTokens rows
    //               are adjacent, so it is ONE run of kPackedBytes at byte
    //               offset t0 * headSize / 2.
    //
    //   valueCache  channel-major. Taken literally as ch * blockSize + key, a
    //               strip owns kTokens of each channel's blockSize nibbles --
    //               256 runs of 8 bytes on a blockSize / 2 byte stride. Those
    //               runs are neither 32-byte bursts nor 32-byte aligned, and
    //               two strips of one page would share a 32-byte word, so
    //               concurrent strips would tear each other's writes. The plane
    //               is therefore strip-blocked:
    //
    //                 nibble index = (key / kTokens) * headSize * kTokens
    //                              + ch * kTokens + (key % kTokens)
    //
    //               channel-major *within* a strip, strips laid end to end. The
    //               strip is again ONE run of kPackedBytes, at byte offset
    //               (t0 / kTokens) * kPackedBytes. This is not a new packing
    //               step: the Vector stage already folds a value chunk into
    //               exactly (ch, key-within-strip) order, which is why the old
    //               store was a single DataCopy at all -- at blockSize ==
    //               kTokens the two readings coincide byte for byte, so the
    //               verified behaviour is unchanged and this only fixes where
    //               the run lands when a page holds more than one strip.
    //
    //   scaleCache  one fp32 per token in a blockSize run, so the strip is
    //               kTokens contiguous floats at offset t0.
    //
    // All three offsets are multiples of kTokens, so every burst stays 32-byte
    // aligned (2048 B for the packed planes, 64 B for the scale slice) and no
    // two strips of a page touch the same 32-byte word. The launcher enforces
    // blockSize % kTokens == 0; a slot0 that is not kTokens-aligned would put a
    // strip across two of these runs, which is the caller's contract.
    const std::uint32_t slot0 = static_cast<std::uint32_t>(slot_mapping[0]);
    const std::uint32_t block = slot0 / block_size;
    const std::uint32_t token_offset = slot0 % block_size;
    const std::uint32_t head_bytes = block_size * head_size / 2u;

    VectorEngine engine;
    engine.key_cache.SetGlobalBuffer(key_cache);
    engine.value_cache.SetGlobalBuffer(value_cache);
    engine.scale_cache.SetGlobalBuffer(scale_cache);
    engine.page_bytes = block * num_kv_heads * head_bytes;
    engine.head_bytes = head_bytes;
    engine.key_token_bytes = token_offset * head_size / 2u;
    engine.value_token_bytes = (token_offset / kTokens) * kPackedBytes;
    engine.key_scale = block * num_kv_heads * block_size + token_offset;
    engine.value_scale = (num_blocks + block) * num_kv_heads * block_size + token_offset;
    engine.head_scale = block_size;
    engine.inv_sqrt_len = inv_sqrt_len;

    // ---- prologue: the quantiser's three immediate constants --------------
    // The factored revision also built two replicated sign panels here, from a
    // Global Memory read of piSigns. The signs are in the Cube's resident
    // blocks now, so the Vector prologue reads no Global Memory at all and
    // every constant below is immediate sourced, which carries no tensor
    // hazard.
    AscendC::Duplicate(View<float>(kUbOne, kScaleFloats), 1.0f,
                       static_cast<std::int32_t>(kScaleFloats));
    AscendC::Duplicate(View<std::int32_t>(kUbExpMask, kScaleFloats),
                       static_cast<std::int32_t>(0x7F800000),
                       static_cast<std::int32_t>(kScaleFloats));
    AscendC::Duplicate(View<std::int32_t>(kUbInvSeed, kScaleFloats),
                       static_cast<std::int32_t>(0x7F000000),
                       static_cast<std::int32_t>(kScaleFloats));
    AscendC::PipeBarrier<PIPE_V>();

    // Feedback cycle priming: both packed page slots start free, and both
    // landing panels are free for the Cube to write.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagPanelFree<Core, 0u>);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagPanelFree<Core, 1u>);

    StepColumnsVector<Core>(
        engine, std::make_integer_sequence<std::uint32_t, kChunks + 3u>{});

    // Drain the primed tokens. These waits also retire the last MTE3 bursts
    // before the kernel returns, so Global Memory is visible to the host.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
}

}  // namespace turboquant_bf16

// The input pointers lose their const at this boundary and get it back one
// line into the body. CANN 9.2 generates host_stub.cpp with a `const void*`
// parameter for a `__gm__ const T*` kernel parameter and then assigns it to a
// `void* ` field of the argument block, which does not compile. Dropping const
// on the five lines below is the only way to keep the published argument order
// and types; every device-side function still takes `__gm__ const`, so nothing
// inside the kernel can write an input.
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_bf16(
    // The activations are bfloat16_t; the caches stay INT8, because they hold
    // packed INT4 nibbles and the output contract does not move with the input
    // precision. `key += strip * numKvHeads * headSize` below therefore still
    // advances exactly one strip: the pointer is typed, so the arithmetic is in
    // elements either way.
    __gm__ bfloat16_t* __restrict__ key,
    __gm__ bfloat16_t* __restrict__ value,
    __gm__ std::int8_t* __restrict__ keyCache,
    __gm__ std::int8_t* __restrict__ valueCache,
    __gm__ float* __restrict__ scaleCache,
    __gm__ std::int32_t* __restrict__ slotMapping,
    __gm__ float* __restrict__ piSigns,
    __gm__ std::int32_t* __restrict__ tables,
    // Unqualified uint32_t, not std::uint32_t: CANN copies this parameter list
    // verbatim into aclrtlaunch_triple_chevrons_func.h, which has no
    // using-declaration for std, so a qualified scalar type breaks the host leg
    // of the build while the device legs compile cleanly. Pointer parameters
    // are rewritten to void* and are unaffected.
    uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,
    uint32_t blockSize, uint32_t numBlocks, uint32_t tokensPerCore,
    float invSqrtLen)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    // The pipeline is specialised to one 16-token strip per physical AI Core.
    // Input and metadata bases advance together, while cache addresses come
    // from the strip's first slot.
    // GetBlockIdx() is NOT the grid block index on a Vector core of a MIX
    // launch. CANN 9.2 dav_3510 (GetBlockIdxImpl in
    // asc/impl/basic_api/dav_3510/kernel_operator_sys_var_impl.h) returns
    //
    //   AIC:  get_block_idx()
    //   AIV:  get_block_idx() * get_subblockdim() + get_subblockid()
    //
    // and the API reference says the same: at a 1:2 AIC:AIV ratio the AIC
    // range is [0, numBlocks) and the AIV range is [0, 2 * numBlocks). Taken
    // raw, the two Vector cores of block b therefore addressed strips 2b and
    // 2b + 1 while their Cube read strip b: only AIV0 of block 0 agreed with
    // its own Cube, every other core wrote another strip's page, and above
    // numBlocks / 2 the page index left the cache entirely. Dividing by
    // GetSubBlockNum() -- 1 on the AIC, the sub-block count on an AIV -- is
    // the one expression both roles agree on, and it is what the vendored
    // baseline's MixBlockIdx() has always done.
    //
    // At blockDim 1 the quotient is 0 either way, which is why eight chunks of
    // one strip passed on CAModel and on silicon while 32 strips did not.
    const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx() /
                                                 AscendC::GetSubBlockNum());
    // numTokens reaches the device only so the host and the kernel agree about
    // the grid: the launcher derives blockDim from numTokens / tokensPerCore and
    // nothing inside a block needs the total.
    (void)numTokens;
    const uint32_t strip = block * tokensPerCore;
    key += strip * numKvHeads * headSize;
    value += strip * numKvHeads * headSize;
    slotMapping += strip;
    // piSigns is folded into `tables` by whoever builds it -- the sign diagonal
    // sits between the two Hadamard factors, so it is a property of the
    // resident rotation blocks (header equation (2)) and not of a Vector-side
    // multiply any more. The argument stays in the published signature; the
    // device does not read it.
    (void)piSigns;
    // The only role dispatch in the kernel, outside every pipeline stage.
    if ASCEND_IS_AIC {
        turboquant_bf16::run_cube_core(key, value, tables, numKvHeads, headSize);
    } else if (AscendC::GetSubBlockIdx() == 0) {
        turboquant_bf16::run_vector_core<0u>(keyCache, valueCache, scaleCache, slotMapping,
                                           numKvHeads, headSize, blockSize,
                                           numBlocks, invSqrtLen);
    } else {
        turboquant_bf16::run_vector_core<1u>(keyCache, valueCache, scaleCache, slotMapping,
                                           numKvHeads, headSize, blockSize,
                                           numBlocks, invSqrtLen);
    }
}

#else  // host leg of the same ASC translation unit

// The activations are 2-byte here, and this leg has no CANN device headers --
// they live behind `#if defined(__CCE_AICORE__)` -- so `bfloat16_t` is not
// available. std::uint16_t stands in for it: every pointer becomes void* in the
// generated argument block, and the host side holds BF16 as bit patterns
// anyway. The `tables` parameter keeps the std::int32_t* spelling the kernel
// publishes and is reinterpreted device side.
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_bf16(
    __gm__ std::uint16_t*, __gm__ std::uint16_t*, __gm__ std::int8_t*,
    __gm__ std::int8_t*, __gm__ float*, __gm__ std::int32_t*,
    __gm__ float*, __gm__ std::int32_t*, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, float);

// One launch of `block_dim` strips. The grid is NOT derived here: the window
// loop in op_adapter/turboquant_torch_adpt.h owns how many strips a launch
// carries and advances the input bases between launches, so this wrapper is
// exactly the triple-chevron call plus the const strip the generated argument
// block forces. `num_tokens` is the token count of THIS launch, so
// num_tokens / tokens_per_core == block_dim; it is passed for the kernel's own
// grid bookkeeping and is otherwise unread.
extern "C" void turboquant_reshape_and_cache_bf16_impl(
    void* stream, std::uint32_t block_dim, const std::uint16_t* key,
    const std::uint16_t* value, std::int8_t* key_cache, std::int8_t* value_cache,
    float* scale_cache, const std::int32_t* slot_mapping, const float* pi_signs,
    const std::int32_t* tables, std::uint32_t num_tokens, std::uint32_t num_kv_heads,
    std::uint32_t head_size, std::uint32_t block_size, std::uint32_t num_blocks,
    std::uint32_t tokens_per_core, float inv_sqrt_len)
{
    turboquant_reshape_and_cache_bf16<<<block_dim, nullptr, stream>>>(
        const_cast<std::uint16_t*>(key), const_cast<std::uint16_t*>(value), key_cache,
        value_cache, scale_cache, const_cast<std::int32_t*>(slot_mapping),
        const_cast<float*>(pi_signs), const_cast<std::int32_t*>(tables), num_tokens,
        num_kv_heads, head_size, block_size, num_blocks, tokens_per_core,
        inv_sqrt_len);
}

#endif
