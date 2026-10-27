// Fused TurboQuant paged attention, Ascend 950PR / arch35 / dav-3510 / CANN 9.2.
//
// MIX_AIC_1_2: one Cube core and BOTH Vector cores, three-stage modulo
// software pipeline over runtime-selected key pages, pure streaming online softmax, and
// a hardware Cube-Vector transfer that keeps the dequantised key page on chip.
//
//   engine        owner        work
//   ------------  -----------  --------------------------------------------
//   MTE2          AIV0 + AIV1  packed 4-bit page ingress
//   Vector        AIV0         V dequant, online-softmax recurrence, outflow
//   Vector        AIV1         K dequant
//   MTE3          AIV1         UB -> L1 direct key staging (the CV path)
//   MTE1/Cube/FIX AIC          L1 -> L0, Mmad, Fixpipe -> AIV0 UB
//   MTE3          AIV0         one normalised write-out, after the sequence
//
// Leaves, all used unmodified through their published entry points:
//   routines/routine_turboquant_dequant4.hpp      int4 -> FP16 / FP32
//   routines/routine_cv_transfer_ub_to_l1.hpp     UB -> L1, ND -> NZ placement
//   routines/routine_cube_matmul.hpp              L1 -> L0 -> L0C -> UB
//   routines/routine_vector_online_softmax.hpp    the (m, l, Acc) recurrence
//   routines/routine_vector_normalize_store.hpp   Acc / l, cast, one MTE3
//
// No Global Memory workspace
// --------------------------
// The dequantised key page never leaves the chip. AIV1 dequantises it into UB
// and stages it straight into the Cube L1 ring with the dav-3510 UB->L1 MTE3
// channel; see routine_cv_transfer_ub_to_l1.hpp for the API evidence and for
// why the Nd2NzParams overload is unusable in a kernel with no TPipe. This
// replaces the UB -> GM workspace -> L1 round trip of the previous revision
// and removes the ws_k argument entirely. Section 6.1 of
// MICROROUTINE_DESIGN_GUIDELINES.md, which states that no UB->L1 transfer
// exists, described the Tensor API copy surface of an earlier CANN and is
// superseded: CANN 9.2 has both Te CopyUB2L1 and the basic_api DataCopy used
// here.
//
// Three-stage modulo schedule
// ------------------------------------------------------------
// Column j of the schedule, with slots taken modulo kStageDepth at compile
// time, runs three mutually independent stages:
//
//   stage 0  ingress + dequant + CV stage of tile j + 2   slot (j + 2) % 3
//   stage 1  L1 -> L0, Mmad, Fixpipe of tile j + 1        slot (j + 1) % 3
//   stage 2  online-softmax recurrence of tile j          slot  j      % 3
//
// Nothing in a column depends on anything else in that column, so MTE2,
// Vector, MTE3, MTE1, Cube and Fixpipe are all free to be in flight together.
// Up to kMaxTiles columns are emitted by a fold over a compile-time index
// sequence, one specialisation per column. Runtime guards select the requested
// page count; slot indices remain compile-time constants. Synchronisation is
// HardEvent tokens inside a core and
// mode-4 intra-block flags between cores; there is no PipeBarrier<PIPE_ALL>.
//
// The runtime page count selects two useful boundaries:
//
//   tiles == 3    the depth boundary. No ring slot is ever reused, so the
//                 reuse back edges are primed and drained but never contended,
//                 and the span is a pipeline fill, not a throughput interval.
//   tiles == 8    wraps the depth-3 ring twice. Slot s is refilled for tile
//                 s + 3 and again for s + 6, so every back edge below carries
//                 real contention and the per-column interval is measurable.
//
// The back edges that only do work once the ring wraps:
//
//   AIC  -> AIV1   kFlagL1Free     L1 key slot free, after MTE1 read it
//   AIV0 -> AIC    kFlagPanelFree  Fixpipe score panel free, after recurrence
//   AIV1 intra     MTE3_V          FP16 key page free, after the CV burst
//   AIV1 intra     V_MTE2          packed key page free, after the dequant
//   AIV0 intra     V_MTE2          packed value page and value page free
//
// AIV1's V_MTE2 edge was missing before this revision and the N = 3 probe
// could not see it: WaitFlag<MTE3_V> stalls the V queue, not MTE2, so nothing
// gated the packed-key refill. With three pages the three MTE2 bursts use
// three different ring slots and three different event ids, so there was
// neither a WAR hazard nor a double set. With eight the MTE2 queue, which has
// no other gate on this core, runs ahead of the throttled V queue, refills a
// packed slot a page before its dequant has read it, and sets MTE2_V for the
// same id twice before the first wait consumes it. See
// reports/fused_attention_integration.md.
//
// A mode-4 set is a scalar instruction, so it needs a drain
// ---------------------------------------------------------
// CrossCoreSetFlag<4, PIPE_x> lowers to set_intra_block(PIPE_x, id), and in
// the CAModel trace that instruction issues on the FLOWCTRL unit, which runs
// the scalar stream: the PIPE_x argument does **not** hold the set back until
// pipe x has drained past the work in front of it. At N = 3 that is invisible,
// because every producer happens to be ahead of its consumer anyway. At N = 8
// the Cube's scalar stream ran through six columns of flags in 130 cycles and
// released six L1-slot-free tokens by cycle 1402, while its first L1 -> L0B
// load only moved at cycle 2862; AIV1 refilled L1 slot 0 with page 3 at cycle
// 2344 and the Cube then multiplied page 3's keys into page 0's score panel.
//
// So every cross-core set that is supposed to mean "the work on pipe x is
// finished" is preceded here by a drain of pipe x into the scalar pipe:
//
//   SetFlag<HardEvent::x_S>(id); WaitFlag<HardEvent::x_S>(id);
//   CrossCoreSetFlag<4, PIPE_x>(flag);
//
// The wait stalls the scalar stream until pipe x has retired the instruction
// the matching set was queued behind, which is exactly the ordering the pipe
// argument looks like it provides. The library's own SyncAllImpl gets there
// with PipeBarrier<PIPE_ALL> in front of its intra-block handshake; a full
// barrier per page would serialise the whole pipeline, and one route per edge
// costs two instructions. There is no MTE1_S route, so the Cube's L1-free edge
// rides the FIX_S drain: the Fixpipe is paired to the Mmad by unitFlag = 3 and
// the Mmad waits MTE1_M, so a retired Fixpipe proves the L1 -> L0 loads are
// done. Cross-core *waits* need no such treatment: wait_intra_block also
// stalls the scalar stream, and everything the consumer issues afterwards is
// therefore behind the flag.
//
// Mode-4 double handshake
// -----------------------
// set_intra_block / wait_intra_block ids form one space per AI Core block in
// which the library reserves the upper half for sub block 1
// (GetIntraFlagId adds subBlockId * INTRA_NUM, and SyncAllImpl waits both
// SYNC_AIV_FLAG and SYNC_AIV_FLAG + SYNC_FLAG_ID_MAX). This kernel follows
// that convention: ids 0..10 are AIV0 traffic, ids 16..22 are AIV1 traffic,
// and the Cube waits one flag from each Vector core before every GEMM, so a
// page is multiplied only once both Vector cores have finished their half of
// the staging. In the trace an AIV reports sync_id inside its own half while
// the Cube reports the absolute id, so AIV1's id 16 reads as sync_id:0 on
// core0.veccore1 and sync_id:16 on core0.cubecore0.
//
// Host packing contract
// ---------------------
//   k_packed nibble index (key * kDim + d) = quantised K[key][d]
//   v_packed nibble index (ch  * kKeys + key) = quantised V[key][ch]
// Two nibbles per byte, low nibble first, matching the dequant4 contract.
//
// The key order is the one change from the previous revision. For a single
// 16x16 half fractal, Nz is row major and Zn is its transpose, so the Zn image
// of K^T that the Cube B operand needs is byte for byte the Nz image of the ND
// page K[key][d]. Packing the page in key-major order therefore lets the
// dequantised UB buffer be placed into L1 by the CV leaf with no element
// permutation anywhere: the KV cache is stored in the fractal order the Cube
// wants, which is what a production paged-attention cache does. Scores are
// unaffected: S = Q * K^T exactly as before, and the runner oracle is
// unchanged.
//
// Scales are compile time; the page count is a scalar kernel argument.

// Status in vllm-ascend: COMPILED, NOT REGISTERED
// ------------------------------------------------
// This translation unit is carried here so the schedule it validates keeps
// building against the in-tree toolchain, and for no other reason. It is a
// CAModel pipeline probe, not a paged-attention operator, and it cannot read
// the TurboQuant cache that turboquant_reshape_and_cache_bf16.cpp writes:
//
//   * kDim is 16. It is the head dimension and it sizes every accumulator and
//     output buffer below. The TurboQuant cache in this tree is head_size 256.
//   * there is no scale operand. kKeyScale and kValueScale are constexpr 0.5f
//     and 0.25f and are handed straight to the dequant leaf, so the planar
//     [2, num_blocks, num_kv_heads, block_size] fp32 scale plane that the
//     writer fills is never read. Per-token scales are the whole point of the
//     quantiser, and a fixed scale is only valid for the probe fixture.
//   * there is no block table and no context length. `offsets` is a flat
//     per-tile page offset array, `tiles` is 3..8, and the launch is
//     <<<1, ...>>>, so the reachable context is 48 to 128 keys on one AI core.
//   * the four outputs are intermediate state for a harness to check, not an
//     attention result.
//
// So no operator schema is registered for it. Doing that would publish an
// entry point under _C_ascend that no caller could feed correctly, which is
// the failure mode the ingestion commit avoided. Making the schedule below
// into a decode operator means reworking it against the real cache geometry
// and a runtime scale plane; that is kernel work, not integration work.
//
// What this file does buy: the schedule compiles and links for dav-3510 beside
// the ingestion kernel in one library, so the two cannot drift apart silently.

#include <cstdint>
#include <utility>

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "routines/routine_turboquant_dequant4.hpp"
#include "routines/routine_cv_transfer_ub_to_l1.hpp"
#include "routines/routine_cube_matmul.hpp"
#include "routines/routine_vector_online_softmax.hpp"
#include "routines/routine_vector_normalize_store.hpp"
#endif

namespace fused_attention {
constexpr std::uint32_t kStageDepth = 3;
// The score archive and unrolled column list need compile-time bounds. Eight
// pages covers two depth-3 ring wraps and is the largest supported fixture.
constexpr std::uint32_t kMaxTiles = 8;
constexpr std::uint32_t kDim = 16;   // head dim, channel count and Cube M
constexpr std::uint32_t kKeys = 16;  // keys per page
constexpr std::uint32_t kPanel = kDim * kKeys;
constexpr std::uint32_t kPackedBytes = kPanel / 2;  // two nibbles per byte
}  // namespace fused_attention

#if defined(__CCE_AICORE__)

// The direct UB->L1 channel exists only when the AIC:AIV ratio is not 1:1; at
// 1:1 the same DataCopy silently becomes a KFC relay through Global Memory,
// which would reintroduce the workspace this kernel exists to remove.
#if defined(__MIX_CORE_AIC_RATION__) && (__MIX_CORE_AIC_RATION__ == 1)
#error "MIX_AIC_1_2 is required: a 1:1 AIC:AIV ratio routes UB->L1 through GM"
#endif

namespace fused_attention {

// ---------------------------------------------------------------- UB layout
// Rule AKA3003. Every base is 32-byte aligned, consecutive slots carry one
// DataBlock of bank skew, and the arena keeps VECTOR_REG_WIDTH of addressable
// UB behind the last operand because a level-2 vector op reads a whole vector
// register per source, unmasked.
//
// Both Vector cores have their own 256 KB UB, so one arena map serves both and
// each core touches only the slots its role uses. Keeping a single map means a
// single set of alignment and capacity assertions, and it lets the Cube form
// the Fixpipe destination address from the same constant the recurrence reads.
constexpr std::uint32_t kBlock = 32;
constexpr std::uint32_t kBankSkewBytes = 32u;
constexpr std::uint32_t kReadPadBytes = 256u;  // VECTOR_REG_WIDTH on dav-3510
constexpr std::uint32_t kUbWindow = 192u * 1024u;
constexpr std::uint32_t kStateLanes = ascend_micro::kStateLanes;          // 8
constexpr std::uint32_t kBroadcastLanes = ascend_micro::kBroadcastLanes;  // 64

constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}
constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkewBytes;
}

constexpr std::uint32_t kF32 = sizeof(float);
constexpr std::uint32_t kI16 = sizeof(std::int16_t);
constexpr std::uint32_t kF16 = 2u;

constexpr std::uint32_t kPanelF32Bytes = kPanel * kF32;   // 1024
constexpr std::uint32_t kPanelF16Bytes = kPanel * kF16;   // 512
constexpr std::uint32_t kRingSlotF32 = kPanelF32Bytes + kBankSkewBytes;  // 1056
constexpr std::uint32_t kRingSlotF16 = kPanelF16Bytes + kBankSkewBytes;  // 544
constexpr std::uint32_t kRingSlotPacked = kPackedBytes + kBankSkewBytes; // 160

// The Fixpipe landing ring stays at UB 0 so the Cube side can form a raw
// __ubuf__ pointer without depending on any Vector-side allocation. It is
// kStageDepth panels deep and no deeper: that is what makes the panel-free
// back edge a real WAR guard once the ring wraps. Widening it to kMaxTiles would
// let the whole sequence be checked without the archive below, but it would
// also uncontend the edge this revision exists to contend.
constexpr std::uint32_t kUbScoreRing = 0u;
constexpr std::uint32_t kUbScoreRingBytes = kStageDepth * kRingSlotF32;

// Packed ingress rings, one per role. MTE2 runs ahead of the Vector pipe, so
// these must be ring buffered even though the dequant scratch need not be.
constexpr std::uint32_t kUbPackedKRing = NextSlot(kUbScoreRing, kUbScoreRingBytes);
constexpr std::uint32_t kUbPackedVRing =
    NextSlot(kUbPackedKRing, kStageDepth * kRingSlotPacked);

// Dequantised page rings. The FP16 key ring is the CV transfer source; the
// FP32 value ring is the recurrence operand.
constexpr std::uint32_t kUbKeyRing =
    NextSlot(kUbPackedVRing, kStageDepth * kRingSlotPacked);
constexpr std::uint32_t kUbValueRing =
    NextSlot(kUbKeyRing, kStageDepth * kRingSlotF16);

// Dequant scratch, single buffered. Reuse across pages is a same-pipe Vector
// dependency, which the Vector queue orders in issue order; it is not a
// cross-engine hazard. The slot is sized for the widest user, the FP32 value
// path, so the FP16 key path shares it without a second allocation.
constexpr std::uint32_t kUbDqSigned = NextSlot(kUbValueRing, kStageDepth * kRingSlotF32);
constexpr std::uint32_t kUbDqMask = NextSlot(kUbDqSigned, kPanel * kI16);
constexpr std::uint32_t kUbDqUnsigned = NextSlot(kUbDqMask, kPanel * kI16);
constexpr std::uint32_t kUbDqConverted = NextSlot(kUbDqUnsigned, kPanel * kI16);

// Online-softmax operands. Logically scalar state is a lane-replicated
// 32-byte DataBlock; reduction destinations hold lane 0 only.
constexpr std::uint32_t kUbOffsets = NextSlot(kUbDqConverted, kPanel * kF32);
constexpr std::uint32_t kUbMPrev = NextSlot(kUbOffsets, kPanel * kF32);
constexpr std::uint32_t kUbLPrev = NextSlot(kUbMPrev, kStateLanes * kF32);
constexpr std::uint32_t kUbMNew = NextSlot(kUbLPrev, kStateLanes * kF32);
constexpr std::uint32_t kUbLNew = NextSlot(kUbMNew, kStateLanes * kF32);
constexpr std::uint32_t kUbAlphaDelta = NextSlot(kUbLNew, kStateLanes * kF32);
constexpr std::uint32_t kUbAlpha = NextSlot(kUbAlphaDelta, kStateLanes * kF32);
constexpr std::uint32_t kUbLScaled = NextSlot(kUbAlpha, kStateLanes * kF32);
constexpr std::uint32_t kUbTileMaxRaw = NextSlot(kUbLScaled, kStateLanes * kF32);
constexpr std::uint32_t kUbPSumRaw = NextSlot(kUbTileMaxRaw, kStateLanes * kF32);
constexpr std::uint32_t kUbAccPrev = NextSlot(kUbPSumRaw, kStateLanes * kF32);
constexpr std::uint32_t kUbAccNew = NextSlot(kUbAccPrev, kDim * kF32);
constexpr std::uint32_t kUbAccScaled = NextSlot(kUbAccNew, kDim * kF32);
constexpr std::uint32_t kUbCentered = NextSlot(kUbAccScaled, kDim * kF32);
constexpr std::uint32_t kUbP = NextSlot(kUbCentered, kDim * kF32);
constexpr std::uint32_t kUbWeightedSum = NextSlot(kUbP, kDim * kF32);
constexpr std::uint32_t kUbTileMaxBc = NextSlot(kUbWeightedSum, kDim * kF32);
constexpr std::uint32_t kUbMBc = NextSlot(kUbTileMaxBc, kBroadcastLanes * kF32);
constexpr std::uint32_t kUbPSumBc = NextSlot(kUbMBc, kBroadcastLanes * kF32);
constexpr std::uint32_t kUbAlphaBc = NextSlot(kUbPSumBc, kBroadcastLanes * kF32);
constexpr std::uint32_t kUbPRepeated = NextSlot(kUbAlphaBc, kBroadcastLanes * kF32);
constexpr std::uint32_t kUbWeighted = NextSlot(kUbPRepeated, kPanel * kF32);

// Outflow operands.
constexpr std::uint32_t kUbLBroadcast = NextSlot(kUbWeighted, kPanel * kF32);
constexpr std::uint32_t kUbNormalized = NextSlot(kUbLBroadcast, kDim * kF32);
constexpr std::uint32_t kUbContext = NextSlot(kUbNormalized, kDim * kF32);

// Score archive, one slot per page, written by the recurrence from the panel
// it has just consumed and read back by the single end-of-sequence outflow.
// The Fixpipe ring is only kStageDepth deep, so without this the panels of
// tiles 0..tiles - 4 would be overwritten before the outflow could read them
// and only the last three would be checkable. The copy is a Vector Muls by
// 1.0f, which is bit exact for every finite value including signed zeros, so
// the archived panel is the Fixpipe output bit for bit.
constexpr std::uint32_t kUbScoreArchive = NextSlot(kUbContext, kDim * kF16);
constexpr std::uint32_t kArenaEnd = kUbScoreArchive + kMaxTiles * kRingSlotF32;

static_assert(kUbScoreRing % kBlock == 0 && kUbPackedKRing % kBlock == 0 &&
              kUbPackedVRing % kBlock == 0 && kUbKeyRing % kBlock == 0 &&
              kUbValueRing % kBlock == 0 && kUbDqSigned % kBlock == 0 &&
              kUbDqMask % kBlock == 0 && kUbDqUnsigned % kBlock == 0 &&
              kUbDqConverted % kBlock == 0 && kUbOffsets % kBlock == 0 &&
              kUbMPrev % kBlock == 0 && kUbAccPrev % kBlock == 0 &&
              kUbTileMaxBc % kBlock == 0 && kUbPRepeated % kBlock == 0 &&
              kUbWeighted % kBlock == 0 && kUbLBroadcast % kBlock == 0 &&
              kUbNormalized % kBlock == 0 && kUbContext % kBlock == 0 &&
              kUbScoreArchive % kBlock == 0,
              "every UB operand base must be 32-byte aligned");
static_assert(kRingSlotF32 % kBlock == 0 && kRingSlotF16 % kBlock == 0 &&
              kRingSlotPacked % kBlock == 0,
              "every ring slot stride must be a whole number of DataBlocks");
static_assert(kRingSlotF32 % 256u != 0u && kRingSlotF16 % 256u != 0u &&
              kRingSlotPacked % 256u != 0u,
              "a 256-byte ring stride puts every slot on the same bank phase");
static_assert(kArenaEnd + kReadPadBytes <= kUbWindow,
              "UB arena leaves no VECTOR_REG_WIDTH read pad behind the state");
static_assert(kUbLPrev + kReadPadBytes <= kUbWindow,
              "Duplicate over-reads 256 bytes from the l denominator base");
static_assert(kBroadcastLanes * kF32 == 8u * kBlock,
              "a Brcb repeat writes eight DataBlocks");
static_assert(kPanelF16Bytes % kBlock == 0 && kDim * kF16 % kBlock == 0,
              "FP16 MTE3 lengths must be whole 32-byte bursts");

// Slot addresses and flag ids are variable templates, not constexpr functions:
// in the CCE dialect a plain constexpr function is __host__, so an __aicore__
// function may not call one. A variable template is a compile-time value.
template <std::uint32_t Slot>
constexpr std::uint32_t kScoreSlot = kUbScoreRing + Slot * kRingSlotF32;
template <std::uint32_t Slot>
constexpr std::uint32_t kValueSlot = kUbValueRing + Slot * kRingSlotF32;
template <std::uint32_t Slot>
constexpr std::uint32_t kKeySlot = kUbKeyRing + Slot * kRingSlotF16;
template <std::uint32_t Slot>
constexpr std::uint32_t kPackedKSlot = kUbPackedKRing + Slot * kRingSlotPacked;
template <std::uint32_t Slot>
constexpr std::uint32_t kPackedVSlot = kUbPackedVRing + Slot * kRingSlotPacked;
template <std::uint32_t Tile>
constexpr std::uint32_t kArchiveSlot = kUbScoreArchive + Tile * kRingSlotF32;

// ---------------------------------------------------------------- L1 layout
// Q is page invariant and reaches L1 once. The key ring has one slot per
// pipeline stage, so the Cube never reads a slot the Vector core is refilling.
// L1 operands need 32-byte alignment; the 512-byte rule applies to L0A/L0B.
// The 544-byte stride is one 16x16 FP16 fractal plus a DataBlock of bank skew,
// the layout the Hadamard leaf validated.
constexpr std::uint32_t kL1Q = 0u;
constexpr std::uint32_t kL1KeyRing = NextSlot(kL1Q, kPanelF16Bytes);
constexpr std::uint32_t kL1SlotStride = kRingSlotF16;
constexpr std::uint32_t kL1End = kL1KeyRing + kStageDepth * kL1SlotStride;

static_assert(kL1Q % kBlock == 0 && kL1KeyRing % kBlock == 0 &&
              kL1SlotStride % kBlock == 0,
              "every L1 operand base must be 32-byte aligned");
static_assert(kL1End <= 512u * 1024u, "L1 ring exceeds the L1 buffer");

template <std::uint32_t Slot>
constexpr std::uint32_t kL1KeySlot = kL1KeyRing + Slot * kL1SlotStride;

// --------------------------------------------------------- quantisation
constexpr float kKeyScale = 0.5f;
constexpr float kValueScale = 0.25f;

// ------------------------------------------------- cross-core flag ids
// Mode 4 is set_intra_block / wait_intra_block inside one AI Core block. Ids
// 0..15 are sub block 0 traffic and 16..31 sub block 1 traffic, the split the
// CANN library itself uses. 12..15 and 28..31 are left clear of the library
// reserved ids (SYNC_AIC_AIV_FLAG 13, SYNC_AIV_ONLY_ALL 14).
template <std::uint32_t Slot>
constexpr std::uint16_t kFlagSlotReadyValue = 0u + Slot;   //  0..2  AIV0 -> AIC
template <std::uint32_t Slot>
constexpr std::uint16_t kFlagProductReady = 4u + Slot;     //  4..6  AIC  -> AIV0
template <std::uint32_t Slot>
constexpr std::uint16_t kFlagPanelFree = 8u + Slot;        //  8..10 AIV0 -> AIC
template <std::uint32_t Slot>
constexpr std::uint16_t kFlagSlotReadyKey = 16u + Slot;    // 16..18 AIV1 -> AIC
template <std::uint32_t Slot>
constexpr std::uint16_t kFlagL1Free = 20u + Slot;          // 20..22 AIC  -> AIV1

// Intra-core HardEvent ids. IDs 6 and 7 are reserved, so slot ids stay 0..2
// and the epilogue uses 3.
constexpr std::int32_t kEvtEpilogue = 3;

// ------------------------------------------------- compile-time column list
// The schedule runs tiles columns. Column j is the triple
//
//   ingress slot (j + 2) % kStageDepth   present while j + 2 < tiles
//   GEMM    slot (j + 1) % kStageDepth   present while j + 1 < tiles
//   recur   slot  j      % kStageDepth   always
//
// and every one of those five values is a constant expression in a template
// argument list, evaluated by the front end. A fold over an index sequence
// instantiates one column body per supported column, in order, with runtime
// guards for presence. This is the same idiom routine_cv_transfer_ub_to_l1.hpp uses to
// unroll its burst strips.
template <std::uint32_t Column>
constexpr std::uint32_t kIngressSlot = (Column + 2u) % kStageDepth;
template <std::uint32_t Column>
constexpr std::uint32_t kGemmSlot = (Column + 1u) % kStageDepth;
template <std::uint32_t Column>
constexpr std::uint32_t kRecurSlot = Column % kStageDepth;
using ColumnSeq = std::make_integer_sequence<std::uint32_t, kMaxTiles>;

// ============================================================ AIV1: key path
// MTE2 ingress of the packed key page, Vector dequantisation to FP16, and the
// direct UB -> L1 staging burst. This is the whole CV path; it touches Global
// Memory exactly once, to read the packed page.
struct KeyEngine {
    AscendC::GlobalTensor<std::int8_t> k_packed;
};

template <std::uint32_t Slot>
__aicore__ inline void StageKeyPage(KeyEngine& engine, std::uint32_t tile)
{
    static_assert(Slot < kStageDepth);
    constexpr auto ub = AscendC::TPosition::VECCALC;
    constexpr std::int32_t evt = static_cast<std::int32_t>(Slot);

    ascend_micro::NativeDequant4Engine<half> dq;
    dq.packed = {ub, kPackedKSlot<Slot>, kPackedBytes};
    dq.signed_nibbles = {ub, kUbDqSigned, kPanel};
    dq.nibble_mask = {ub, kUbDqMask, kPanel};
    dq.unsigned_nibbles = {ub, kUbDqUnsigned, kPanel};
    dq.converted = {ub, kUbDqConverted, kPanel};
    dq.output = {ub, kKeySlot<Slot>, kPanel};
    dq.row_scale = static_cast<half>(kKeyScale);

    ascend_micro::NativeCvTransferEngine<half> cv;
    cv.ub_position = ub;
    cv.l1_position = AscendC::TPosition::B1;
    cv.ub_base_bytes = kKeySlot<Slot>;
    cv.l1_base_bytes = kL1KeySlot<Slot>;

    // Two back edges serialise this slot body against its own incarnation
    // kStageDepth tiles earlier, and they gate two different queues:
    //
    //   V_MTE2  the dequant of tile - kStageDepth has read the packed page,
    //           so MTE2 may refill it. WaitFlag<V_MTE2> stalls the MTE2 queue.
    //   MTE3_V  the CV burst of tile - kStageDepth has read the FP16 page, so
    //           the dequant may overwrite it. WaitFlag<MTE3_V> stalls V.
    //
    // Without the first one nothing on this core gates MTE2 at all: the MTE2
    // queue free runs past the V queue, which is throttled by the L1-free flag
    // from the Cube. Different slots remain free to overlap.
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evt);
    AscendC::DataCopy(dq.packed, engine.k_packed[tile * kPackedBytes], kPackedBytes);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evt);
    ascend_micro::turboquant_dequant4(dq, kPanel);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evt);
    // The Cube must have finished with L1 slot Slot before the staging burst
    // overwrites it. The Cube releases the slot after the Fixpipe of the page
    // that read it, which is one Mmad later than strictly necessary but is an
    // ordering the hardware actually guarantees. Primed once per slot.
    AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(kFlagL1Free<Slot>);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evt);
    ascend_micro::cv_transfer_nd2nz<kKeys, kDim, kDim, 1u, 1u>(cv);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evt);
    // Drain MTE3 into the scalar pipe before announcing the L1 slot: the
    // cross-core set is a scalar instruction and would otherwise overtake the
    // staging burst it is reporting.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evt);
    AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(kFlagSlotReadyKey<Slot>);
}

template <std::uint32_t Column>
__aicore__ inline void StepColumnKey(KeyEngine& engine, std::uint32_t tiles)
{
    static_assert(kIngressSlot<Column> < kStageDepth);
    if (Column + 2u < tiles)
        StageKeyPage<kIngressSlot<Column>>(engine, Column + 2u);
}

template <std::uint32_t... Columns>
__aicore__ inline void StepColumnsKey(
    KeyEngine& engine, std::uint32_t tiles,
    std::integer_sequence<std::uint32_t, Columns...>)
{
    (StepColumnKey<Columns>(engine, tiles), ...);
}

__aicore__ inline void run_key_core(__gm__ std::int8_t* gm_k_packed,
                                    std::uint32_t tiles)
{
    KeyEngine engine;
    engine.k_packed.SetGlobalBuffer(gm_k_packed);

    // Feedback cycle priming: every ring slot starts drained, on both the
    // MTE2 refill edge and the dequant overwrite edge.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(2);

    StageKeyPage<0>(engine, 0);
    StageKeyPage<1>(engine, 1);

    StepColumnsKey(engine, tiles, ColumnSeq{});

    // Drain every primed token and every token the Cube handed back, so no
    // event bit and no intra-block flag leaks into the next launch. These
    // waits also retire the in-flight MTE3 bursts before the kernel returns.
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(2);
    AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(kFlagL1Free<0>);
    AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(kFlagL1Free<1>);
    AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(kFlagL1Free<2>);
}

// ====================================================== AIV0: value path
// MTE2 ingress of the packed value page, Vector dequantisation to FP32, the
// online-softmax recurrence over the Cube score panels, and the single
// end-of-sequence outflow.
struct ValueEngine {
    AscendC::GlobalTensor<std::int8_t> v_packed;
};

template <std::uint32_t Slot>
__aicore__ inline void StageValuePage(ValueEngine& engine, std::uint32_t tile)
{
    static_assert(Slot < kStageDepth);
    constexpr auto ub = AscendC::TPosition::VECCALC;
    constexpr std::int32_t evt = static_cast<std::int32_t>(Slot);

    ascend_micro::NativeDequant4Engine<float> dq;
    dq.packed = {ub, kPackedVSlot<Slot>, kPackedBytes};
    dq.signed_nibbles = {ub, kUbDqSigned, kPanel};
    dq.nibble_mask = {ub, kUbDqMask, kPanel};
    dq.unsigned_nibbles = {ub, kUbDqUnsigned, kPanel};
    dq.converted = {ub, kUbDqConverted, kPanel};
    dq.output = {ub, kValueSlot<Slot>, kPanel};
    dq.row_scale = kValueScale;

    // The recurrence releases the slot when it has consumed the value page.
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evt);
    AscendC::DataCopy(dq.packed, engine.v_packed[tile * kPackedBytes], kPackedBytes);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evt);
    ascend_micro::turboquant_dequant4(dq, kPanel);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(evt);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagSlotReadyValue<Slot>);
}

// Pure streaming online softmax. The carried state (m, l, Acc) lives in UB
// across the whole page sequence: there is no Split-K partial, no workspace
// and no store of any kind in this stage.
template <std::uint32_t Slot, std::uint32_t Tile>
__aicore__ inline void RecurPage(ValueEngine&)
{
    static_assert(Slot < kStageDepth && Tile < kMaxTiles);
    constexpr auto ub = AscendC::TPosition::VECCALC;
    constexpr std::int32_t evt = static_cast<std::int32_t>(Slot);

    ascend_micro::NativeOnlineSoftmaxEngine es;
    es.scores = {ub, kScoreSlot<Slot>, kKeys};  // query row 0 of the panel
    es.v_by_channel = {ub, kValueSlot<Slot>, kPanel};
    es.p_repeat_offsets = {ub, kUbOffsets, kPanel};
    es.m_prev = {ub, kUbMPrev, kStateLanes};
    es.l_prev = {ub, kUbLPrev, kStateLanes};
    es.m_new = {ub, kUbMNew, kStateLanes};
    es.l_new = {ub, kUbLNew, kStateLanes};
    es.alpha_delta = {ub, kUbAlphaDelta, kStateLanes};
    es.alpha = {ub, kUbAlpha, kStateLanes};
    es.l_scaled = {ub, kUbLScaled, kStateLanes};
    es.tile_max_raw = {ub, kUbTileMaxRaw, kStateLanes};
    es.p_sum_raw = {ub, kUbPSumRaw, kStateLanes};
    es.acc_prev = {ub, kUbAccPrev, kDim};
    es.acc_new = {ub, kUbAccNew, kDim};
    es.acc_scaled = {ub, kUbAccScaled, kDim};
    es.centered = {ub, kUbCentered, kDim};
    es.p = {ub, kUbP, kDim};
    es.weighted_sum = {ub, kUbWeightedSum, kDim};
    es.tile_max_bc = {ub, kUbTileMaxBc, kBroadcastLanes};
    es.m_bc = {ub, kUbMBc, kBroadcastLanes};
    es.p_sum_bc = {ub, kUbPSumBc, kBroadcastLanes};
    es.alpha_bc = {ub, kUbAlphaBc, kBroadcastLanes};
    es.p_repeated = {ub, kUbPRepeated, kPanel};
    es.weighted = {ub, kUbWeighted, kPanel};

    // The whole panel, not just the query row the recurrence reads, so the
    // outflow can check all tiles * kPanel entries from one UB region after
    // the Fixpipe ring has been reused. Muls by 1.0f is an exact FP32 copy and
    // stays on the Vector pipe, so the cross-core set below, which is issued
    // on PIPE_V, is ordered after it: the Cube cannot overwrite the panel
    // until this read has retired.
    AscendC::LocalTensor<float> panel{ub, kScoreSlot<Slot>, kPanel};
    AscendC::LocalTensor<float> archived{ub, kArchiveSlot<Tile>, kPanel};

    AscendC::CrossCoreWaitFlag<4, PIPE_V>(kFlagProductReady<Slot>);
    ascend_micro::vector_online_softmax(es);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(es.m_prev, es.m_new, 0.0f, static_cast<std::int32_t>(kStateLanes));
    AscendC::Adds(es.l_prev, es.l_new, 0.0f, static_cast<std::int32_t>(kStateLanes));
    AscendC::Adds(es.acc_prev, es.acc_new, 0.0f, static_cast<std::int32_t>(kDim));
    AscendC::Muls(archived, panel, 1.0f, static_cast<std::int32_t>(kPanel));
    AscendC::PipeBarrier<PIPE_V>();
    // The value page and the score panel are both consumed now. The panel is
    // the one the Cube is waiting to overwrite, so the scalar pipe waits for
    // the Vector queue to retire the archive copy before the flag goes out.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evt);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(evt);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagPanelFree<Slot>);
}

template <std::uint32_t Column>
__aicore__ inline void StepColumnValue(ValueEngine& engine, std::uint32_t tiles)
{
    static_assert(kIngressSlot<Column> < kStageDepth &&
                  kRecurSlot<Column> < kStageDepth);
    if (Column + 2u < tiles)
        StageValuePage<kIngressSlot<Column>>(engine, Column + 2u);
    if (Column < tiles)
        RecurPage<kRecurSlot<Column>, Column>(engine);
}

template <std::uint32_t... Columns>
__aicore__ inline void StepColumnsValue(
    ValueEngine& engine, std::uint32_t tiles,
    std::integer_sequence<std::uint32_t, Columns...>)
{
    (StepColumnValue<Columns>(engine, tiles), ...);
}

__aicore__ inline void run_value_core(__gm__ std::int8_t* gm_v_packed,
                                      __gm__ std::uint32_t* gm_offsets,
                                      __gm__ float* gm_out_scores,
                                      __gm__ float* gm_out_state,
                                      __gm__ float* gm_out_normalized,
                                      __gm__ half* gm_out_context,
                                      std::uint32_t tiles)
{
    constexpr auto ub = AscendC::TPosition::VECCALC;

    ValueEngine engine;
    engine.v_packed.SetGlobalBuffer(gm_v_packed);

    AscendC::GlobalTensor<std::uint32_t> offsets;
    AscendC::GlobalTensor<float> out_scores, out_state, out_normalized;
    offsets.SetGlobalBuffer(gm_offsets);
    out_scores.SetGlobalBuffer(gm_out_scores);
    out_state.SetGlobalBuffer(gm_out_state);
    out_normalized.SetGlobalBuffer(gm_out_normalized);

    AscendC::LocalTensor<std::uint32_t> offsets_ub{ub, kUbOffsets, kPanel};
    AscendC::DataCopy(offsets_ub, offsets, kPanel);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvtEpilogue);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvtEpilogue);

    // Seed the replicated-state invariant from immediate scalars, which never
    // read UB and so carry none of the tensor-source hazards.
    AscendC::LocalTensor<float> m_prev{ub, kUbMPrev, kStateLanes};
    AscendC::LocalTensor<float> l_prev{ub, kUbLPrev, kStateLanes};
    AscendC::LocalTensor<float> acc_prev{ub, kUbAccPrev, kDim};
    AscendC::Duplicate(m_prev, 0.0f, static_cast<std::int32_t>(kStateLanes));
    AscendC::Duplicate(l_prev, 0.0f, static_cast<std::int32_t>(kStateLanes));
    AscendC::Duplicate(acc_prev, 0.0f, static_cast<std::int32_t>(kDim));

    // Feedback cycle priming: every value slot is free, and every score panel
    // in the Fixpipe ring is free for the Cube to write.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagPanelFree<0>);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagPanelFree<1>);
    AscendC::CrossCoreSetFlag<4, PIPE_V>(kFlagPanelFree<2>);

    StageValuePage<0>(engine, 0);
    StageValuePage<1>(engine, 1);

    StepColumnsValue(engine, tiles, ColumnSeq{});

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);

    // ---- Outflow, once, after the whole page sequence ------------------
    // Nothing above this point stores to Global Memory. The score archive is
    // read back here as one strided MTE3 burst, tiles panels of
    // kPanelF32Bytes separated by the archive bank skew, so the runner can
    // check every entry of every page without the kernel ever dumping a panel
    // per page. The V -> MTE3 edge below covers the archive writes too: the
    // Vector queue is in order, so the token set after vector_normalize is set
    // after the last archive Muls has retired.
    ascend_micro::NativeNormalizeStoreEngine<half> en;
    en.acc = acc_prev;
    en.l = l_prev;
    en.l_broadcast = {ub, kUbLBroadcast, kDim};
    en.normalized = {ub, kUbNormalized, kDim};
    en.ub_output = {ub, kUbContext, kDim};
    en.gm_output.SetGlobalBuffer(gm_out_context);

    ascend_micro::vector_normalize(en, kDim);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEvtEpilogue);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEvtEpilogue);
    ascend_micro::vector_store(en, kDim);

    AscendC::LocalTensor<float> score_archive{ub, kUbScoreArchive,
                                              kMaxTiles * kPanel};
    AscendC::DataCopyParams panels;
    panels.blockCount = static_cast<std::uint16_t>(tiles);
    panels.blockLen = static_cast<std::uint16_t>(kPanelF32Bytes / kBlock);
    panels.srcStride =
        static_cast<std::uint16_t>((kRingSlotF32 - kPanelF32Bytes) / kBlock);
    panels.dstStride = 0;
    AscendC::DataCopy(out_scores, score_archive, panels);

    AscendC::DataCopy(out_state[0], m_prev, kStateLanes);
    AscendC::DataCopy(out_state[kStateLanes], l_prev, kStateLanes);
    AscendC::DataCopy(out_normalized[0], en.normalized, kDim);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtEpilogue);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtEpilogue);
}

// ============================================================== AIC: Cube
struct CubeEngine {
    __cbuf__ half* l1_q;
    __ca__ half* l0a;
    __cb__ half* l0b;
    __cc__ float* l0c;
};

template <std::uint32_t Slot>
__aicore__ inline void GemmPage(CubeEngine& engine, std::uint32_t)
{
    static_assert(Slot < kStageDepth);
    constexpr std::int32_t evt = static_cast<std::int32_t>(Slot);
    auto* l1_k = reinterpret_cast<__cbuf__ half*>(kL1KeySlot<Slot>);
    auto* ub_scores = reinterpret_cast<__ubuf__ float*>(kScoreSlot<Slot>);

    ascend_micro::NativeCubeMatmulEngine<kDim> e{engine.l1_q, l1_k,
                                                 engine.l0a, engine.l0b,
                                                 engine.l0c, ub_scores};

    // Double handshake quorum: one flag from each Vector core. The page is
    // multiplied only once AIV1 has staged the key fractal into L1 slot Slot
    // and AIV0 has dequantised the matching value page, so neither Vector core
    // can be outrun by the Cube.
    AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(kFlagSlotReadyKey<Slot>);
    AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(kFlagSlotReadyValue<Slot>);

    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(evt);  // L0A/L0B free
    ascend_micro::cube_matmul_stage_l0(e);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(evt);

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(evt);  // L0C free
    // The recurrence must have consumed the previous occupant of this score
    // panel before Fixpipe overwrites it.
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagPanelFree<Slot>);
    // One Mmad paired with one Fixpipe under matching unitFlag = 3, so the L0C
    // producer/consumer dependency rides in the instruction encoding.
    ascend_micro::cube_matmul_compute_dump(e);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(evt);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(evt);
    // One drain for both outgoing flags. A retired Fixpipe proves the Mmad
    // retired, and the Mmad waited MTE1_M, so it also proves the L1 -> L0
    // loads have read the L1 key slot and the staging core may refill it.
    // There is no MTE1_S route to express that more tightly, and releasing the
    // slot one Mmad later costs the staging core nothing it can use: its next
    // burst into this slot is three pages away.
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(evt);
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(kFlagProductReady<Slot>);
    AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(kFlagL1Free<Slot>);
}

template <std::uint32_t Column>
__aicore__ inline void StepColumnCube(CubeEngine& engine, std::uint32_t tiles)
{
    static_assert(kGemmSlot<Column> < kStageDepth);
    if (Column + 1u < tiles)
        GemmPage<kGemmSlot<Column>>(engine, Column + 1u);
}

template <std::uint32_t... Columns>
__aicore__ inline void StepColumnsCube(
    CubeEngine& engine, std::uint32_t tiles,
    std::integer_sequence<std::uint32_t, Columns...>)
{
    (StepColumnCube<Columns>(engine, tiles), ...);
}

__aicore__ inline void run_cube_core(__gm__ half* gm_q, std::uint32_t tiles)
{
    using namespace AscendC::Te;
    __ca__ half l0_a[kDim * kKeys] __attribute__((aligned(512)));
    __cb__ half l0_b[kDim * kKeys] __attribute__((aligned(512)));
    __cc__ float l0_c[kDim * kKeys] __attribute__((aligned(1024)));

    CubeEngine engine{reinterpret_cast<__cbuf__ half*>(kL1Q), l0_a, l0_b, l0_c};

    // Q is page invariant, so only the B operand is refreshed per page.
    auto src_q = MakeTensor(MakeMemPtr(gm_q), MakeFrameLayout<NDExtLayoutPtn>(kDim, kKeys));
    auto dst_q = MakeTensor(MakeMemPtr(engine.l1_q),
                            MakeFrameLayout<NZLayoutPtn, half>(kDim, kKeys));
    Copy(MakeCopy(CopyGM2L1{}), dst_q, src_q);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtEpilogue);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtEpilogue);

    // Feedback cycle priming: L0A/L0B and L0C are free, and every L1 key slot
    // is free for the staging core to fill.
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
    AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(kFlagL1Free<0>);
    AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(kFlagL1Free<1>);
    AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(kFlagL1Free<2>);

    GemmPage<0>(engine, 0);

    StepColumnsCube(engine, tiles, ColumnSeq{});

    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagPanelFree<0>);
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagPanelFree<1>);
    AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(kFlagPanelFree<2>);
}

}  // namespace fused_attention

extern "C" __global__ __aicore__ void camodel_fused_attention(
    __gm__ half* q, __gm__ std::int8_t* k_packed, __gm__ std::int8_t* v_packed,
    __gm__ std::uint32_t* offsets, __gm__ float* out_scores,
    __gm__ float* out_state, __gm__ float* out_normalized,
    __gm__ half* out_context, uint32_t tiles)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    if (tiles < fused_attention::kStageDepth || tiles > fused_attention::kMaxTiles)
        return;
    // The only role dispatch in the kernel, outside every pipeline stage.
    if ASCEND_IS_AIC {
        fused_attention::run_cube_core(q, tiles);
    } else if (AscendC::GetSubBlockIdx() == 0) {
        fused_attention::run_value_core(v_packed, offsets, out_scores,
                                        out_state, out_normalized, out_context, tiles);
    } else {
        fused_attention::run_key_core(k_packed, tiles);
    }
}

#else  // host leg of the same ASC translation unit

#include <cassert>
#include <stdexcept>

extern "C" __global__ __aicore__ void camodel_fused_attention(
    __gm__ half*, __gm__ std::int8_t*, __gm__ std::int8_t*, __gm__ std::uint32_t*,
    __gm__ float*, __gm__ float*, __gm__ float*, __gm__ half*, uint32_t);

extern "C" void launch_camodel_fused_attention(
    void* stream, std::uint16_t* q, std::int8_t* k_packed, std::int8_t* v_packed,
    std::uint32_t* offsets, float* out_scores, float* out_state,
    float* out_normalized, std::uint16_t* out_context, std::uint32_t tiles)
{
    // The fixed UB archive and unrolled schedule are sized for kMaxTiles.
    assert(tiles >= fused_attention::kStageDepth && tiles <= fused_attention::kMaxTiles);
    if (tiles < fused_attention::kStageDepth || tiles > fused_attention::kMaxTiles)
        throw std::invalid_argument("fused attention tiles must be 3..8");
    camodel_fused_attention<<<1, nullptr, stream>>>(
        reinterpret_cast<half*>(q), k_packed, v_packed, offsets, out_scores,
        out_state, out_normalized, reinterpret_cast<half*>(out_context), tiles);
}

#endif
