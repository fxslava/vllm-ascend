#include <cstdint>
#include <utility>

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "cube/routine_stage_page_zn_bf16.hpp"

#define TQ_AICORE __aicore__
#else
#define TQ_AICORE
#endif

#include "common/turboquant_wave_protocol.hpp"
#include "common/turboquant_layout.h"

#ifndef TURBOQUANT_INGRESS_NZ
#define TURBOQUANT_INGRESS_NZ 0
#endif

namespace turboquant_bf16 {

constexpr std::uint32_t kTokens = 16;
constexpr std::uint32_t kHeads = 4;
constexpr std::uint32_t kHeadSize = 256;
constexpr std::uint32_t kGroup = 32;
constexpr std::uint32_t kGroups = kHeadSize / kGroup;
constexpr std::uint32_t kChunks = 2u * kHeads;
constexpr std::uint32_t kPanel = kTokens * kHeadSize;
constexpr std::uint32_t kPackedBytes = kPanel / 2u;
constexpr std::uint32_t kScaleFloats = kTokens;

constexpr std::uint32_t kElemBytes = 2u;
constexpr std::uint32_t kFractalK = 32u / kElemBytes;
constexpr std::uint32_t kFractalsPerGroup = kGroup / kFractalK;

constexpr std::uint32_t kBlockElems = kGroup * kGroup;
constexpr std::uint32_t kTableBlocks = 2u * kGroups;
constexpr std::uint32_t kTableElems = kTableBlocks * kBlockElems;
constexpr std::uint32_t kTableBytes = kTableElems * kElemBytes;
constexpr std::uint32_t kTableRows = kTableBlocks * kGroup;

constexpr std::uint32_t kStageDepth = 4;
constexpr std::uint32_t kLandDepth = 2;
constexpr std::uint32_t kPackDepth = 2;

constexpr std::uint32_t kPhases = kChunks;
constexpr std::uint32_t kIngressLookahead = kStageDepth - 1u;

#if TURBOQUANT_INGRESS_NZ
// ----------------------------------------------- write-time NZ-tiled ingress (the geometry)
// TURBOQUANT_INGRESS_NZ=1 makes this writer emit the NZ-fractal cache the fused decode reads
// instead of the RFC 3.1 row-major strip. A physical block becomes
//
//   [kNzTileRows-row tile][kv head][kNzC0-byte column group][tile row][byte]
//
// with the kv4fp8 nibble pairing -- one byte carries channels (c, c + d/2) of ONE token, low
// nibble first -- and the scale plane interleaved slot-major rather than plane-major.
// turboquant_layout.h is the normative contract. Every constant here and the open-coded address
// arithmetic below are tied back to it by static_assert, so a writer that drifted from the
// reader cannot compile; tests/cpu/test_layout_bridge.cpp is the same gate from the other side.
//
// The 16-token strip this operator quantises in one pass is a QUARTER of a tile, and the four
// quarters are independent: each writes its own 16 tile rows of each column group, so no two
// strips -- and no two grid blocks -- ever touch the same byte. That is why this layout change
// needs no cross-block protocol and no read-modify-write.
constexpr std::uint32_t kNzTileRows = 64u;
constexpr std::uint32_t kNzC0 = 32u;
constexpr std::uint32_t kNzRowBytes = kHeadSize / 2u;              // packedBytes of one row
constexpr std::uint32_t kNzGroups = kNzRowBytes / kNzC0;           // column groups in a row
constexpr std::uint32_t kNzHeadBytes = kNzTileRows * kNzRowBytes;  // one (tile, kv head) plane
constexpr std::uint32_t kNzGroupBytes = kNzTileRows * kNzC0;       // one column group of a tile
constexpr std::uint32_t kNzScaleSlot = ((2u * kHeads + 7u) / 8u) * 8u;  // NzScaleSlotFloats(H)
constexpr std::uint32_t kNzScaleSlabFloats = kTokens * 8u;         // the 16 x 32-byte lane image

namespace tq_contract = vllm_ascend::turboquant;

constexpr std::uint32_t kNzBurstBytes = kTokens * kNzC0;

// The NZ byte a strip starts at, given the GLOBAL row of its first token. The per-head step
// from it is kNzHeadBytes, the per-group step kNzGroupBytes and the per-tile-row step kNzC0 --
// which is the whole address arithmetic this writer needs, because a strip's slice of one
// column group is 16 CONTIGUOUS tile rows of 32 bytes.
TQ_AICORE constexpr std::uint32_t NzStripBase(std::uint32_t slot, std::uint32_t heads)
{
    return (slot - slot % kNzTileRows) * heads * kNzRowBytes + (slot % kNzTileRows) * kNzC0;
}

// The open-coded base, head step, group step and row step against NzTiledPackedByte itself,
// over every (head, group, tile row, byte) of a strip. Checked for each tile row a 16-token
// strip can start on, and once far up the cache so the tile term is exercised too.
TQ_AICORE constexpr bool NzByteAgrees(std::uint32_t slot, std::uint32_t head, std::uint32_t group,
                            std::uint32_t row, std::uint32_t byte)
{
    return static_cast<std::uint64_t>(NzStripBase(slot, kHeads)) + head * kNzHeadBytes +
               group * kNzGroupBytes + row * kNzC0 + byte ==
           tq_contract::NzTiledPackedByte(slot + row, head, group * kNzC0 + byte, kHeads,
                                          kNzRowBytes);
}

TQ_AICORE constexpr bool NzStripAgrees(std::uint32_t slot)
{
    for (std::uint32_t head = 0; head < kHeads; ++head)
        for (std::uint32_t group = 0; group < kNzGroups; ++group)
            for (std::uint32_t row = 0; row < kTokens; ++row)
                for (std::uint32_t byte = 0; byte < kNzC0; ++byte)
                    if (!NzByteAgrees(slot, head, group, row, byte)) return false;
    for (std::uint32_t kv = 0; kv < 2u; ++kv)
        for (std::uint32_t head = 0; head < kHeads; ++head)
            for (std::uint32_t row = 0; row < kTokens; ++row)
                if (slot * kNzScaleSlot + kv * kHeads + head + row * kNzScaleSlot !=
                    tq_contract::NzTiledScaleIndex(slot + row, head, kv, kHeads))
                    return false;
    return true;
}

static_assert(NzStripAgrees(0u) && NzStripAgrees(kTokens) && NzStripAgrees(2u * kTokens) &&
                  NzStripAgrees(3u * kTokens) && NzStripAgrees(kNzTileRows) &&
                  NzStripAgrees(4096u + 3u * kTokens),
              "the writer's NZ addressing disagrees with turboquant_layout.h");
static_assert(kTokens == tq_contract::kStripTokens, "the strip height is the contract's R");
static_assert(kNzTileRows == tq_contract::kCubeTileRows, "a tile is the contract's 64 rows");
static_assert(kNzC0 == tq_contract::kOperandC0, "a column group is the contract's C0");
static_assert(kNzRowBytes == kHeadSize / tq_contract::kPackFactor, "two nibbles to a byte");
static_assert(kNzScaleSlot == tq_contract::NzScaleSlotFloats(kHeads),
              "the scale slot width is NzScaleSlotFloats(numKvHeads)");
static_assert(kNzScaleSlabFloats == kTokens * tq_contract::kFp32PerBlock,
              "the scale lane image is one FP32 DataBlock a token");
#endif

}

#if defined(__CCE_AICORE__)

namespace turboquant_bf16 {

template <std::uint32_t C> constexpr bool kIsKey = (C >= kHeads);
template <std::uint32_t C> constexpr std::uint32_t kChunkHead = C % kHeads;
template <std::uint32_t C> constexpr std::uint32_t kOwner = C & 1u;
template <std::uint32_t C> constexpr std::uint32_t kL1Slot = C % kStageDepth;
template <std::uint32_t C> constexpr std::uint32_t kLandSlot = (C / 2u) % kLandDepth;
template <std::uint32_t C> constexpr std::uint32_t kPackSlot = (C / 2u) % kPackDepth;

constexpr std::uint32_t ParityOf(std::uint32_t v) {
    std::uint32_t parity = 0;
    while (v != 0u) { parity ^= (v & 1u); v >>= 1; }
    return parity;
}
template <std::uint32_t C1, std::uint32_t G>
constexpr std::uint32_t kSignBlock = ParityOf(C1 & G) * kGroups + G;

using turboquant_wave::kBankSkewBytes;
using turboquant_wave::kBlock;
using turboquant_wave::kReadPadBytes;
using turboquant_wave::kUbWindow;
using turboquant_wave::NextSlot;
using turboquant_wave::RoundUpBlock;
constexpr std::uint32_t kLanesPerBlock = kBlock / sizeof(float);

constexpr std::uint32_t kPanelBytes = kPanel * 4u;
constexpr std::uint32_t kLandSlotBytes = kPanelBytes + kBankSkewBytes;
#if TURBOQUANT_INGRESS_NZ
// A pack slot carries the NZ-ordered 2048 bytes plus the slot-major scale image: the 16 scale
// floats at a 32-byte pitch, which is the source frame the strided MTE3 descriptor reads lane 0
// of. The strip writer's slot carries the same bytes and 16 CONTIGUOUS floats instead.
constexpr std::uint32_t kPackSlotPayload = kPackedBytes + kNzScaleSlabFloats * 4u;
#else
constexpr std::uint32_t kPackSlotPayload = kPackedBytes + kScaleFloats * 4u;
#endif
constexpr std::uint32_t kPackSlotBytes = RoundUpBlock(kPackSlotPayload) + kBankSkewBytes;

constexpr std::uint32_t kUbLandRing = 0u;
constexpr std::uint32_t kUbLandBytes = kLandDepth * kLandSlotBytes;
constexpr std::uint32_t kUbCast = NextSlot(kUbLandRing, kUbLandBytes);
constexpr std::uint32_t kUbWork = NextSlot(kUbCast, kPanelBytes);
constexpr std::uint32_t kUbFold128 = NextSlot(kUbWork, kPanelBytes);
constexpr std::uint32_t kUbFold64 = NextSlot(kUbFold128, kTokens * 128u * 4u);
constexpr std::uint32_t kUbFold8 = NextSlot(kUbFold64, kTokens * 64u * 4u);
constexpr std::uint32_t kUbHalf = NextSlot(kUbFold8, kTokens * 8u * 4u);
constexpr std::uint32_t kUbAbsMax = NextSlot(kUbHalf, kPanel * 2u);
constexpr std::uint32_t kUbExpBits = NextSlot(kUbAbsMax, kScaleFloats * 4u);
constexpr std::uint32_t kUbPow = NextSlot(kUbExpBits, kScaleFloats * 4u);
constexpr std::uint32_t kUbInv = NextSlot(kUbPow, kScaleFloats * 4u);
constexpr std::uint32_t kUbInvRep = NextSlot(kUbInv, kScaleFloats * 4u);
constexpr std::uint32_t kUbInvBcast = NextSlot(kUbInvRep, 4u * kScaleFloats * 4u);
constexpr std::uint32_t kUbOne = NextSlot(kUbInvBcast, kTokens * kLanesPerBlock * 4u);
constexpr std::uint32_t kUbExpMask = NextSlot(kUbOne, kScaleFloats * 4u);
constexpr std::uint32_t kUbInvSeed = NextSlot(kUbExpMask, kScaleFloats * 4u);
#if TURBOQUANT_INGRESS_NZ
// The three NZ staging panels, each 4096 half elements. kUbNzTmp is the ping-pong partner of
// kUbHalf for the value plane's four DeInterleave passes; kUbNzSrc holds the two nibble planes
// in NZ order, low at 0 and high at + kNzPlaneElems; kUbWoven holds their element-wise weave,
// which is the nibble sequence the int4b_t Cast packs straight into the pack slot.
//
// None of the three is ringed. Each is written and consumed inside ONE RotateAndQuantise, and
// the only buffer that has to outlive it -- because MTE3 drains it while the next chunk's
// vector work runs -- is the pack slot, which keeps its depth-2 ring unchanged.
constexpr std::uint32_t kUbNzTmp = NextSlot(kUbInvSeed, kScaleFloats * 4u);
constexpr std::uint32_t kUbNzSrc = NextSlot(kUbNzTmp, kPanel * 2u);
constexpr std::uint32_t kUbWoven = NextSlot(kUbNzSrc, kPanel * 2u);
constexpr std::uint32_t kUbPackRing = NextSlot(kUbWoven, kPanel * 2u);
#else
constexpr std::uint32_t kUbPackRing = NextSlot(kUbInvSeed, kScaleFloats * 4u);
#endif
constexpr std::uint32_t kArenaEnd = kUbPackRing + kPackDepth * kPackSlotBytes;

template <std::uint32_t Slot> constexpr std::uint32_t kLandBase = kUbLandRing + Slot * kLandSlotBytes;
template <std::uint32_t Slot> constexpr std::uint32_t kPackBase = kUbPackRing + Slot * kPackSlotBytes;
template <std::uint32_t Slot> constexpr std::uint32_t kScaleBase = kPackBase<Slot> + kPackedBytes;

constexpr std::uint32_t kL1Table = 0u;
constexpr std::uint32_t kL1PageElems = kTokens * kHeadSize;
constexpr std::uint32_t kL1PageBytes = kL1PageElems * kElemBytes;
constexpr std::uint32_t kL1Ring = NextSlot(kL1Table, kTableBytes);
constexpr std::uint32_t kL1SlotBytes = kL1PageBytes + kBankSkewBytes;
constexpr std::uint32_t kL1End = kL1Ring + kStageDepth * kL1SlotBytes;
constexpr std::uint32_t kFractalElems = kTokens * kGroup;
constexpr std::uint32_t kFractalBytes = kFractalElems * kElemBytes;

template <std::uint32_t Slot> constexpr std::uint32_t kL1PageBase = kL1Ring + Slot * kL1SlotBytes;

constexpr std::uint32_t kL0TableElems = kTableElems;
constexpr std::uint32_t kL0PageElems = kGroups * kFractalElems;
constexpr std::uint32_t kL0Elems = kL0TableElems + kL0PageElems;
constexpr std::uint32_t kL0CElems = kGroups * kTokens * kGroup;

using turboquant_wave::kSync2AicAnnounce;
using turboquant_wave::kSync2AicWait;
using turboquant_wave::kSync2AivAnnounce;
using turboquant_wave::kSync2AivWait;
using turboquant_wave::kSync2PageDrained;
using turboquant_wave::kSync2PanelFree;
using turboquant_wave::kSync2PanelFreePrime;
using turboquant_wave::kSync2PanelReady;

constexpr std::uint32_t kGenerations = kChunks / 2u;
template <std::uint32_t C> constexpr bool kOpensGeneration = (C % 2u) == 0u;
template <std::uint32_t C> constexpr bool kClosesGeneration = (C % 2u) == 1u;

constexpr std::int32_t kEvtL0 = 0;
constexpr std::int32_t kEvtEpilogue = 0;
constexpr std::int32_t kEvtDrain = 0;
constexpr std::int32_t kEvtRendezvous = 2;

using ChunkSeq = std::make_integer_sequence<std::uint32_t, kChunks>;
using GroupSeq = std::make_integer_sequence<std::uint32_t, kGroups>;

static_assert(kFractalsPerGroup * kFractalK == kGroup);
static_assert(kTableRows * kGroup == kTableElems);
static_assert(kChunks / 2u == kLandDepth * 2u);

static_assert(kUbLandRing % kBlock == 0u && kUbCast % kBlock == 0u &&
              kUbWork % kBlock == 0u && kUbFold128 % kBlock == 0u &&
              kUbFold64 % kBlock == 0u && kUbFold8 % kBlock == 0u &&
              kUbHalf % kBlock == 0u && kUbAbsMax % kBlock == 0u &&
              kUbExpBits % kBlock == 0u && kUbPow % kBlock == 0u &&
              kUbInv % kBlock == 0u && kUbInvRep % kBlock == 0u && 
              kUbInvBcast % kBlock == 0u && kUbOne % kBlock == 0u && 
              kUbExpMask % kBlock == 0u && kUbInvSeed % kBlock == 0u && 
              kUbPackRing % kBlock == 0u);
static_assert(kLandSlotBytes % kBlock == 0u && kPackSlotBytes % kBlock == 0u);
static_assert(kLandSlotBytes % 256u != 0u && kPackSlotBytes % 256u != 0u);
static_assert(kArenaEnd + turboquant_wave::kReadPadBytes <= turboquant_wave::kUbWindow);
static_assert(kUbInvSeed + turboquant_wave::kReadPadBytes <= turboquant_wave::kUbWindow);
static_assert(kPackedBytes % kBlock == 0u && kScaleFloats * 4u % kBlock == 0u);
#if TURBOQUANT_INGRESS_NZ
static_assert(kUbNzTmp % kBlock == 0u && kUbNzSrc % kBlock == 0u && kUbWoven % kBlock == 0u);
static_assert(kNzScaleSlabFloats * 4u % kBlock == 0u && kNzBurstBytes % kBlock == 0u &&
              kNzGroupBytes % kBlock == 0u && kNzHeadBytes % kBlock == 0u);
#endif
static_assert(kTokens * kLanesPerBlock * 4u == kTokens * kBlock);

static_assert(kL1Table % kBlock == 0u && kL1Ring % kBlock == 0u && kL1SlotBytes % kBlock == 0u);
static_assert(kL1SlotBytes % 256u != 0u);
static_assert(kL1End <= 512u * 1024u);
static_assert(kFractalElems * kGroups == kL1PageElems);

static_assert(kL0TableElems * kElemBytes % 512u == 0u);
static_assert(kBlockElems * kElemBytes % 512u == 0u);
static_assert(kL0Elems * kElemBytes <= 64u * 1024u);

static_assert(turboquant_wave::Mode2LandingBudget<kLandDepth>::kFits);
static_assert(kGenerations * 2u == kChunks);
static_assert(kGenerations % kLandDepth == 0u);
static_assert(kChunks % kStageDepth == 0u);
static_assert(kGenerations % kPackDepth == 0u);
static_assert(kTokens == turboquant_wave::kStripTokens);
static_assert(kIngressLookahead >= 1u && kIngressLookahead <= kStageDepth);
static_assert(kIngressLookahead < kChunks);

#if TURBOQUANT_INGRESS_NZ
// The NZ stage works in half, which is what the int4b_t Cast consumes and what halves the UB
// traffic of every rearrangement below relative to the fp32 panel.
constexpr std::uint32_t kHalfPerBlock = kBlock / 2u;        // 16 halves to a DataBlock
constexpr std::uint32_t kNzPlaneElems = kPanel / 2u;        // 2048, one nibble plane of a strip
static_assert(kNzPlaneElems == kTokens * kNzRowBytes);
static_assert(kNzPlaneElems % kHalfPerBlock == 0u, "both nibble planes must start 32B aligned");
static_assert(kHeadSize % kHalfPerBlock == 0u && kNzC0 % kHalfPerBlock == 0u);
static_assert(kNzC0 / kHalfPerBlock <= 255u && kHeadSize / kHalfPerBlock <= 255u,
              "a repeat stride is one byte");
static_assert(kTokens <= 255u, "a repeat count is one byte");
#endif

struct CubeState {
    __gm__ const bfloat16_t* key;
    __gm__ const bfloat16_t* value;
    std::uint32_t row_pitch;
    std::uint32_t head_size;
    std::uint32_t strip_elems;
    __ca__ bfloat16_t* l0_a;
    __cb__ bfloat16_t* l0_b;
    __cc__ float* l0_c;
};

struct VectorState {
    AscendC::GlobalTensor<std::int8_t> key_cache;
    AscendC::GlobalTensor<std::int8_t> value_cache;
    AscendC::GlobalTensor<float> scale_cache;
    std::uint32_t page_bytes;
    std::uint32_t head_bytes;
    std::uint32_t key_scale;
    std::uint32_t value_scale;
    std::uint32_t head_scale;
    float inv_sqrt_len;
};

struct StripAddr {
    std::uint32_t page_bytes;
    std::uint32_t key_scale;
    std::uint32_t value_scale;
};

#if TURBOQUANT_INGRESS_NZ
// The NZ form of the same three bases. `slot0` is the GLOBAL row of the strip's first token,
// which is what NzTiledPackedByte and NzTiledScaleIndex are defined on: no separate page base is
// added, because (slot0 - slot0 % 64) already carries it. StoreChunk adds this chunk's head and
// plane steps (kNzHeadBytes, and one float a head above the plane's lane base).
//
// The strip writer's `head_bytes`, `block_size` and `num_blocks` have no NZ counterpart: a tile
// is addressed by its own first row, so the page stride and the plane-major scale split both
// disappear. The signature is kept so ExecuteVector reads the same for both layouts.
__aicore__ inline StripAddr CalculateStripAddr(__gm__ const std::int32_t* slot_mapping,
                                               std::uint32_t strip, std::uint32_t num_kv_heads,
                                               std::uint32_t head_bytes, std::uint32_t block_size,
                                               std::uint32_t num_blocks) {
    (void)head_bytes;
    (void)block_size;
    (void)num_blocks;
    const std::uint32_t slot0 = static_cast<std::uint32_t>(slot_mapping[strip * kTokens]);
    StripAddr a;
    a.page_bytes = NzStripBase(slot0, num_kv_heads);
    a.key_scale = slot0 * kNzScaleSlot;
    a.value_scale = a.key_scale + num_kv_heads;
    return a;
}
#else
__aicore__ inline StripAddr CalculateStripAddr(__gm__ const std::int32_t* slot_mapping,
                                               std::uint32_t strip, std::uint32_t num_kv_heads,
                                               std::uint32_t head_bytes, std::uint32_t block_size,
                                               std::uint32_t num_blocks) {
    const std::uint32_t slot0 = static_cast<std::uint32_t>(slot_mapping[strip * kTokens]);
    const std::uint32_t block_idx = slot0 / block_size;
    const std::uint32_t strip_in_block = (slot0 % block_size) / kTokens;
    const std::uint32_t strip_bytes = strip_in_block * kPackedBytes;
    StripAddr a;
    a.page_bytes = block_idx * num_kv_heads * head_bytes + strip_bytes;
    a.key_scale = block_idx * num_kv_heads * block_size + strip_in_block * kTokens;
    a.value_scale = (num_blocks + block_idx) * num_kv_heads * block_size + strip_in_block * kTokens;
    return a;
}
#endif

constexpr auto kUb = AscendC::TPosition::VECCALC;

template <class T>
__aicore__ inline AscendC::LocalTensor<T> View(std::uint32_t base, std::uint32_t count) {
    return AscendC::LocalTensor<T>(kUb, base, count);
}

template <std::uint32_t N, std::uint32_t SrcPitch, std::uint32_t DstPitch, std::uint32_t... Cs>
__aicore__ inline void KeyFoldEmit(const AscendC::LocalTensor<float>& dst,
                                   const AscendC::LocalTensor<float>& src,
                                   std::integer_sequence<std::uint32_t, Cs...>) {
    constexpr std::uint32_t half = N / 2u;
    constexpr std::int32_t mask = static_cast<std::int32_t>(half > 64u ? 64u : half);
    AscendC::BinaryRepeatParams rp;
    rp.dstBlkStride = 1; rp.src0BlkStride = 1; rp.src1BlkStride = 1;
    rp.dstRepStride = static_cast<std::uint8_t>(DstPitch / kLanesPerBlock);
    rp.src0RepStride = static_cast<std::uint8_t>(SrcPitch / kLanesPerBlock);
    rp.src1RepStride = rp.src0RepStride;
    (AscendC::Max<float>(dst[Cs * mask], src[Cs * mask], src[half + Cs * mask], mask, static_cast<std::uint8_t>(kTokens), rp), ...);
}

template <std::uint32_t N, std::uint32_t SrcPitch, std::uint32_t DstPitch>
__aicore__ inline void KeyFold(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src) {
    constexpr std::uint32_t half = N / 2u;
    KeyFoldEmit<N, SrcPitch, DstPitch>(dst, src, std::make_integer_sequence<std::uint32_t, half / 64u>{});
}

template <std::uint32_t... Steps>
__aicore__ inline void ValueFoldEmit(const AscendC::LocalTensor<float>& buf, std::integer_sequence<std::uint32_t, Steps...>) {
    (AscendC::Max(buf, buf, buf[kPanel >> (Steps + 1u)], static_cast<std::int32_t>(kPanel >> (Steps + 1u))), ...);
}

#if TURBOQUANT_INGRESS_NZ
// --------------------------------------------- the NZ column transposition, inside UB
// One nibble plane of a 16 x 256 token-major panel into NZ order:
//
//   nz[group * (16 * 32) + row * 32 + byte] = panel[row * 256 + group * 32 + byte]
//
// which is the "local transpose of the 16 x 128 byte strip into kNzGroups contiguous column
// groups of 16 x 32" the GM layout wants, done as kNzGroups high-dimensional-split copies: one
// instruction a column group, 16 iterations of 32 elements, the source stepping a whole panel
// row (kHeadSize / kHalfPerBlock DataBlocks) between iterations and the destination stepping
// one 32-byte tile row (kNzC0 / kHalfPerBlock). The scalar is 1.0: these values are integers in
// [-8, 7], exactly representable in half, so the multiply is a pure move.
//
// Nothing here is a GM access. The whole spatial transformation lands in UB before MTE3 is
// armed, which is the point: the alternative -- one scattered 32-byte MTE3 descriptor a tile row
// -- is 64 strided bursts a chunk against 4 contiguous ones.
template <std::uint32_t... Gs>
__aicore__ inline void NzGroupEmit(const AscendC::LocalTensor<half>& dst,
                                   const AscendC::LocalTensor<half>& src,
                                   const AscendC::UnaryRepeatParams& rp,
                                   std::integer_sequence<std::uint32_t, Gs...>) {
    (AscendC::Muls(dst[Gs * kTokens * kNzC0], src[Gs * kNzC0], static_cast<half>(1.0f),
                   static_cast<std::uint64_t>(kNzC0), static_cast<std::uint8_t>(kTokens), rp),
     ...);
}

__aicore__ inline void NzGroupGather(const AscendC::LocalTensor<half>& dst,
                                     const AscendC::LocalTensor<half>& src) {
    AscendC::UnaryRepeatParams rp;
    rp.dstBlkStride = 1;
    rp.srcBlkStride = 1;
    rp.dstRepStride = static_cast<std::uint8_t>(kNzC0 / kHalfPerBlock);
    rp.srcRepStride = static_cast<std::uint8_t>(kHeadSize / kHalfPerBlock);
    NzGroupEmit(dst, src, rp, std::make_integer_sequence<std::uint32_t, kNzGroups>{});
}

// The value plane's Fixpipe lands [channel][token]; the NZ cache is token-major for BOTH planes
// (turboquant_layout.h: the decode reads one address function and the golden oracle dequantises
// through one). One DeInterleave over the whole panel splits even from odd elements, which on a
// contiguous destination is a rotation of the panel index by one bit: the element at
// (channel, token) index c * 16 + t moves to t0 c7..c0 t3..t1. Four passes therefore carry all
// four token bits above the channel and leave exactly t * 256 + c -- the key plane's layout --
// with no transpose instruction and no address table.
__aicore__ inline void NzTokenMajor(const AscendC::LocalTensor<half>& panel,
                                    const AscendC::LocalTensor<half>& scratch) {
    constexpr std::int32_t all = static_cast<std::int32_t>(kPanel);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DeInterleave(scratch, scratch[kNzPlaneElems], panel, all);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DeInterleave(panel, panel[kNzPlaneElems], scratch, all);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DeInterleave(scratch, scratch[kNzPlaneElems], panel, all);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DeInterleave(panel, panel[kNzPlaneElems], scratch, all);
}

// Four passes is log2(kTokens); the chain only lands token-major because the strip is 16 tokens.
static_assert(kTokens == 16u, "NzTokenMajor is four DeInterleave passes, i.e. log2(kTokens)");
#endif

template <std::uint32_t... Qs>
__aicore__ inline void KeyQuantiseEmit(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
                                       const AscendC::LocalTensor<float>& inv_bcast, const AscendC::BinaryRepeatParams& rp,
                                       std::integer_sequence<std::uint32_t, Qs...>) {
    (AscendC::Mul<float>(dst[Qs * 64u], src[Qs * 64u], inv_bcast, 64, static_cast<std::uint8_t>(kTokens), rp), ...);
}

template <typename scalar_t>
class TurboQuantReshapePipeline {
public:
    __aicore__ inline void InitCube(__gm__ const bfloat16_t* key, __gm__ const bfloat16_t* value,
                                    std::uint32_t num_kv_heads, std::uint32_t head_size) {
        cube_.key = key;
        cube_.value = value;
        cube_.row_pitch = num_kv_heads * head_size;
        cube_.head_size = head_size;
        cube_.strip_elems = 0;
    }

    __aicore__ inline void InitVector(__gm__ std::int8_t* key_cache, __gm__ std::int8_t* value_cache,
                                      __gm__ float* scale_cache, std::uint32_t block_size,
                                      std::uint32_t head_size, float inv_sqrt_len) {
        vector_.key_cache.SetGlobalBuffer(key_cache);
        vector_.value_cache.SetGlobalBuffer(value_cache);
        vector_.scale_cache.SetGlobalBuffer(scale_cache);
#if TURBOQUANT_INGRESS_NZ
        // An NZ (tile, kv head) plane is kNzHeadBytes whatever the block size, and a kv head is
        // ONE float inside a slot's record rather than a whole block-size plane.
        (void)block_size;
        (void)head_size;
        vector_.head_bytes = kNzHeadBytes;
        vector_.head_scale = 1u;
#else
        vector_.head_bytes = block_size * head_size / 2u;
        vector_.head_scale = block_size;
#endif
        vector_.inv_sqrt_len = inv_sqrt_len;
    }

    template <std::uint32_t PHASE>
    __aicore__ inline void StagePage(bool stage_next) {
        using namespace AscendC::Te;
        constexpr std::uint32_t slot = kL1Slot<PHASE>;
        constexpr std::int32_t evt = static_cast<std::int32_t>(slot);
        std::uint32_t offset = cube_.strip_elems;
        if (stage_next) offset += kTokens * cube_.row_pitch;

        __gm__ const bfloat16_t* base = kIsKey<PHASE> ? cube_.key : cube_.value;
        auto* src_ptr = const_cast<__gm__ bfloat16_t*>(base) + offset + kChunkHead<PHASE> * cube_.head_size;
        auto* dst_ptr = reinterpret_cast<__cbuf__ bfloat16_t*>(kL1PageBase<slot>);

        auto shape = MakeShape(MakeShape(_1{}, kTokens), MakeShape(_1{}, kHeadSize));
        auto stride = MakeStride(MakeStride(_0{}, cube_.row_pitch), MakeStride(_0{}, _1{}));
        auto src = MakeTensor(MakeMemPtr(src_ptr), MakePatternLayout<NDExtLayoutPtn, LayoutTrait<AscendC::Std::ignore_t, _1>>(shape, stride));
        auto dst = MakeTensor(MakeMemPtr(dst_ptr), MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTokens, kHeadSize));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(evt);
        Copy(MakeCopy(CopyGM2L1{}), dst, src);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(evt);
    }

    template <std::uint32_t PHASE, std::uint32_t G>
    __aicore__ inline void GemmGroupLoad(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b) {
        using namespace AscendC::Te;
        constexpr std::uint32_t slot = kL1Slot<PHASE>;
        auto* page = reinterpret_cast<__cbuf__ bfloat16_t*>(kL1PageBase<slot> + G * kFractalBytes);
        if constexpr (kIsKey<PHASE>) {
            auto src = MakeTensor(MakeMemPtr(page), MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTokens, kGroup));
            auto dst = MakeTensor(MakeMemPtr(l0a + kL0TableElems + G * kFractalElems), MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTokens, kGroup));
            Copy(MakeCopy(CopyL12L0A{}), dst, src);
        } else {
            auto src = MakeTensor(MakeMemPtr(page), MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTokens));
            auto dst = MakeTensor(MakeMemPtr(l0b + kL0TableElems + G * kFractalElems), MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTokens));
            Copy(MakeCopy(CopyL12L0B{}), dst, src);
        }
    }

    template <std::uint32_t PHASE, std::uint32_t C1, std::uint32_t G>
    __aicore__ inline void GemmBlock(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b, __cc__ float* l0c) {
        using namespace AscendC::Te;
        constexpr std::uint32_t block = kSignBlock<C1, G> * kBlockElems;
        constexpr bool init = (G == 0u);
        auto* c_ptr = l0c + C1 * kTokens * kGroup;

        if constexpr (kIsKey<PHASE>) {
            auto a = MakeTensor(MakeMemPtr(l0a + kL0TableElems + G * kFractalElems), MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTokens, kGroup));
            auto b = MakeTensor(MakeMemPtr(l0b + block), MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kGroup));
            auto c = MakeTensor(MakeMemPtr(c_ptr), MakeFrameLayout<NZLayoutPtn, 16>(kTokens, kGroup));
            MmadParams mp(static_cast<std::uint16_t>(kTokens), static_cast<std::uint16_t>(kGroup), static_cast<std::uint16_t>(kGroup), 0, init);
            Mmad(MakeMmad(MmadOperation{}, MmadTraitDefault{}).with(mp), c, a, b);
        } else {
            auto a = MakeTensor(MakeMemPtr(l0a + block), MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kGroup, kGroup));
            auto b = MakeTensor(MakeMemPtr(l0b + kL0TableElems + G * kFractalElems), MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTokens));
            auto c = MakeTensor(MakeMemPtr(c_ptr), MakeFrameLayout<NZLayoutPtn, 16>(kGroup, kTokens));
            MmadParams mp(static_cast<std::uint16_t>(kGroup), static_cast<std::uint16_t>(kTokens), static_cast<std::uint16_t>(kGroup), 0, init);
            Mmad(MakeMmad(MmadOperation{}, MmadTraitDefault{}).with(mp), c, a, b);
        }
    }

    template <std::uint32_t PHASE>
    __aicore__ inline void FixpipeChunk(__cc__ float* l0c) {
        using namespace AscendC::Te;
        constexpr std::uint32_t land = kLandSlot<PHASE>;
        auto* ub = reinterpret_cast<__ubuf__ float*>(kLandBase<land>);
        FixpipeParams fp;
        fp.unitFlag = 0;
        fp.subBlockId = (kOwner<PHASE> == 1u);
        if constexpr (kIsKey<PHASE>) {
            auto c = MakeTensor(MakeMemPtr(l0c), MakeFrameLayout<NZLayoutPtn, 16>(kTokens, kHeadSize));
            auto y = MakeTensor(MakeMemPtr(ub), MakeFrameLayout<NDExtLayoutPtn>(kTokens, kHeadSize));
            Copy(MakeCopy(CopyL0C2UB{}).with(fp), y, c);
        } else {
            auto c = MakeTensor(MakeMemPtr(l0c), MakeFrameLayout<NZLayoutPtn, 16>(kHeadSize, kTokens));
            auto y = MakeTensor(MakeMemPtr(ub), MakeFrameLayout<NDExtLayoutPtn>(kHeadSize, kTokens));
            Copy(MakeCopy(CopyL0C2UB{}).with(fp), y, c);
        }
    }

    template <std::uint32_t PHASE, std::uint32_t... Gs>
    __aicore__ inline void GemmGroupLoads(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b, std::integer_sequence<std::uint32_t, Gs...>) {
        (GemmGroupLoad<PHASE, Gs>(l0a, l0b), ...);
    }

    template <std::uint32_t PHASE, std::uint32_t C1, std::uint32_t... Gs>
    __aicore__ inline void GemmBlockRow(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b, __cc__ float* l0c, std::integer_sequence<std::uint32_t, Gs...>) {
        (GemmBlock<PHASE, C1, Gs>(l0a, l0b, l0c), ...);
    }

    template <std::uint32_t PHASE, std::uint32_t... C1s>
    __aicore__ inline void GemmBlockRows(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b, __cc__ float* l0c, std::integer_sequence<std::uint32_t, C1s...>) {
        (GemmBlockRow<PHASE, C1s>(l0a, l0b, l0c, GroupSeq{}), ...);
    }

    template <std::uint32_t PHASE>
    __aicore__ inline void GemmChunk(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b, __cc__ float* l0c) {
        constexpr std::uint32_t slot = kL1Slot<PHASE>;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<std::int32_t>(slot));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
        GemmGroupLoads<PHASE>(l0a, l0b, GroupSeq{});
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtL0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<std::int32_t>(slot));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtL0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEvtL0);
        AscendC::SetMMRowMajor();
        GemmBlockRows<PHASE>(l0a, l0b, l0c, GroupSeq{});
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEvtL0);
        if constexpr (kOpensGeneration<PHASE>) {
            AscendC::CrossCoreWaitFlag<2, kSync2AicWait>(kSync2PanelFree);
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtRendezvous);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtRendezvous);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEvtL0);
        FixpipeChunk<PHASE>(l0c);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEvtL0);
        if constexpr (kClosesGeneration<PHASE>) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtL0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtL0);
            AscendC::CrossCoreSetFlag<2, kSync2AicAnnounce>(kSync2PanelReady);
        }
    }

    template <std::uint32_t PHASE>
    __aicore__ inline void RotateAndQuantise() {
        constexpr std::uint32_t land = kLandSlot<PHASE>;
        constexpr std::uint32_t pack = kPackSlot<PHASE>;
        constexpr std::int32_t pack_evt = static_cast<std::int32_t>(pack);
        constexpr std::int32_t all = static_cast<std::int32_t>(kPanel);
        auto landing = View<float>(kLandBase<land>, kPanel);
        auto cast = View<float>(kUbCast, kPanel);
        auto work = View<float>(kUbWork, kPanel);

        AscendC::CrossCoreWaitFlag<2, kSync2AivWait>(kSync2PanelReady);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtRendezvous);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtRendezvous);
        AscendC::Muls(cast, landing, 1.0f, all);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(pack_evt);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(pack_evt);
        AscendC::CrossCoreSetFlag<2, kSync2AivAnnounce>(kSync2PanelFree);

        auto abs_max = View<float>(kUbAbsMax, kScaleFloats);
        auto one = View<float>(kUbOne, kScaleFloats);
        AscendC::Abs(work, cast, all);
        if constexpr (kIsKey<PHASE>) {
            auto f128 = View<float>(kUbFold128, kTokens * 128u);
            auto f64 = View<float>(kUbFold64, kTokens * 64u);
            auto f8 = View<float>(kUbFold8, kTokens * 8u);
            KeyFold<256u, 256u, 128u>(f128, work);
            KeyFold<128u, 128u, 64u>(f64, f128);
            AscendC::BlockReduceMax<float>(f8, f64, static_cast<std::uint8_t>(kTokens), 64, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceMax<float>(abs_max, f8, 2, 64, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(abs_max, abs_max, one, static_cast<std::int32_t>(kScaleFloats));
        } else {
            ValueFoldEmit(work, std::make_integer_sequence<std::uint32_t, 8u>{});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(abs_max, work, one, static_cast<std::int32_t>(kScaleFloats));
        }

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

        auto quant = View<std::int32_t>(kUbCast, kPanel);
        auto staged = View<half>(kUbHalf, kPanel);
        auto packed = View<std::int8_t>(kPackBase<pack>, kPackedBytes);
        AscendC::BinaryRepeatParams bc;
        bc.dstBlkStride = 1; bc.src0BlkStride = 1; bc.dstRepStride = 8; bc.src0RepStride = 8;
        if constexpr (kIsKey<PHASE>) {
            auto inv_bcast = View<float>(kUbInvBcast, kTokens * kLanesPerBlock);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(inv_bcast, inv, 2, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            bc.src1BlkStride = 0; bc.src1RepStride = 1;
            bc.dstRepStride = static_cast<std::uint8_t>(kHeadSize / kLanesPerBlock);
            bc.src0RepStride = bc.dstRepStride;
            KeyQuantiseEmit(work, cast, inv_bcast, bc, std::make_integer_sequence<std::uint32_t, kHeadSize / 64u>{});
        } else {
            auto inv_rep = View<float>(kUbInvRep, 4u * kScaleFloats);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(inv_rep, inv, 1.0f, lanes);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(inv_rep[kScaleFloats], inv_rep, 1.0f, lanes);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(inv_rep[2u * kScaleFloats], inv_rep, 1.0f, static_cast<std::int32_t>(2u * kScaleFloats));
            AscendC::PipeBarrier<PIPE_V>();
            bc.src1BlkStride = 1; bc.src1RepStride = 0;
            AscendC::Mul<float>(work, cast, inv_rep, 64, static_cast<std::uint8_t>(kPanel / 64u), bc);
        }

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(work, work, 7.0f, all);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(quant, work, AscendC::RoundMode::CAST_RINT, kPanel);
#if TURBOQUANT_INGRESS_NZ
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(staged, quant, AscendC::RoundMode::CAST_RINT, kPanel);

        // ---- the write-time NZ stage, entirely in UB -------------------------------------
        // (a) the value plane's panel to token-major, so both planes enter the weave in the
        //     same orientation; (b) the 256 channels split at kNzRowBytes into a low half and a
        //     high half, each gathered into NZ column-group order; (c) woven element-wise, so
        //     the int4b_t Cast puts channel c in the low nibble of its byte and channel
        //     c + kNzRowBytes in the high one -- the kv4fp8 pairing the decode's DeInterleave
        //     expand is the inverse of. The pack slot then holds the GM image byte for byte.
        if constexpr (!kIsKey<PHASE>) {
            NzTokenMajor(staged, View<half>(kUbNzTmp, kPanel));
        }
        auto nz = View<half>(kUbNzSrc, kPanel);
        AscendC::PipeBarrier<PIPE_V>();
        NzGroupGather(nz, staged);
        NzGroupGather(nz[kNzPlaneElems], staged[kNzRowBytes]);

        auto woven = View<half>(kUbWoven, kPanel);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pack_evt);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Interleave(woven, woven[kNzPlaneElems], nz, nz[kNzPlaneElems],
                            static_cast<std::int32_t>(kNzPlaneElems));
        AscendC::PipeBarrier<PIPE_V>();
        auto nibbles = packed.template ReinterpretCast<AscendC::int4b_t>();
        AscendC::Cast(nibbles, woven, AscendC::RoundMode::CAST_RINT, kPanel);

        // The scale lanes in their GM frame: one float a token at a 32-byte pitch, which is the
        // source frame of the strided descriptor StoreChunk issues. Brcb lays the pitch down and
        // the fold of invSqrtLen follows on the whole image -- the same product, lane for lane,
        // that the strip writer's single Muls computes.
        auto slab = View<float>(kScaleBase<pack>, kNzScaleSlabFloats);
        AscendC::Brcb(slab, pow_two, static_cast<std::uint8_t>(kTokens / kLanesPerBlock),
                      {1, static_cast<std::uint16_t>(kLanesPerBlock)});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(slab, slab, vector_.inv_sqrt_len,
                      static_cast<std::int32_t>(kNzScaleSlabFloats));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pack_evt);
#else
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pack_evt);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(staged, quant, AscendC::RoundMode::CAST_RINT, kPanel);
        AscendC::PipeBarrier<PIPE_V>();
        auto nibbles = packed.template ReinterpretCast<AscendC::int4b_t>();
        AscendC::Cast(nibbles, staged, AscendC::RoundMode::CAST_RINT, kPanel);
        auto scale = View<float>(kScaleBase<pack>, kScaleFloats);
        AscendC::Muls(scale, pow_two, vector_.inv_sqrt_len, lanes);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pack_evt);
#endif
    }

    template <std::uint32_t PHASE>
    __aicore__ inline void StoreChunk() {
        constexpr std::uint32_t pack = kPackSlot<PHASE>;
        constexpr std::int32_t pack_evt = static_cast<std::int32_t>(pack);
        auto packed = View<std::int8_t>(kPackBase<pack>, kPackedBytes);
#if TURBOQUANT_INGRESS_NZ
        auto slab = View<float>(kScaleBase<pack>, kNzScaleSlabFloats);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pack_evt);
        const std::uint32_t page = vector_.page_bytes + kChunkHead<PHASE> * kNzHeadBytes;
        const std::uint32_t slice = (kIsKey<PHASE> ? vector_.key_scale : vector_.value_scale) + kChunkHead<PHASE> * vector_.head_scale;

        // kNzGroups bursts, one a column group: kNzBurstBytes of contiguous UB into
        // kNzBurstBytes of contiguous GM, kNzGroupBytes apart. A strip's 16 tile rows of a group
        // ARE adjacent in the fractal, so each burst is 512 whole DataBlocks' worth of payload
        // and the write bus never sees a partial beat. Nothing here is a per-row descriptor.
        AscendC::DataCopyExtParams plane{static_cast<std::uint16_t>(kNzGroups), kNzBurstBytes, 0,
                                         kNzGroupBytes - kNzBurstBytes, 0};
        if constexpr (kIsKey<PHASE>) {
            AscendC::DataCopyPad(vector_.key_cache[page], packed, plane);
        } else {
            AscendC::DataCopyPad(vector_.value_cache[page], packed, plane);
        }

        // The one write in this operator that cannot be widened: a slot's record is
        // kNzScaleSlot floats and this chunk owns exactly one of them -- the other heads' lanes
        // and the other plane's lanes are written by the other seven chunks of the same strip,
        // on both subcores, so a wider burst would be a race rather than a throughput win.
        // kTokens records of one float, kNzScaleSlot apart; UB side reads lane 0 of each
        // 32-byte block Brcb laid down.
        AscendC::DataCopyExtParams lanes{static_cast<std::uint16_t>(kTokens),
                                         static_cast<std::uint32_t>(sizeof(float)), 0,
                                         (kNzScaleSlot - 1u) * static_cast<std::uint32_t>(sizeof(float)), 0};
        AscendC::DataCopyPad(vector_.scale_cache[slice], slab, lanes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pack_evt);
#else
        auto scale = View<float>(kScaleBase<pack>, kScaleFloats);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pack_evt);
        const std::uint32_t page = vector_.page_bytes + kChunkHead<PHASE> * vector_.head_bytes;
        const std::uint32_t slice = (kIsKey<PHASE> ? vector_.key_scale : vector_.value_scale) + kChunkHead<PHASE> * vector_.head_scale;
        if constexpr (kIsKey<PHASE>) {
            AscendC::DataCopy(vector_.key_cache[page], packed, kPackedBytes);
        } else {
            AscendC::DataCopy(vector_.value_cache[page], packed, kPackedBytes);
        }
        AscendC::DataCopy(vector_.scale_cache[slice], scale, kScaleFloats);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pack_evt);
#endif
    }

    template <std::uint32_t PHASE>
    __aicore__ inline void RunCubeSteadyPhase(bool stage_next) {
        constexpr std::uint32_t ahead = (PHASE + kIngressLookahead) % kPhases;
        constexpr bool crosses = (PHASE + kIngressLookahead) >= kPhases;
        if (!crosses || stage_next) {
            StagePage<ahead>(crosses);
        }
        GemmChunk<PHASE>(cube_.l0_a, cube_.l0_b, cube_.l0_c);
    }

    template <std::uint32_t PHASE, std::uint32_t Core>
    __aicore__ inline void RunVectorSteadyPhase(bool quant_next) {
        constexpr std::uint32_t ahead = (PHASE + 1) % kPhases;
        constexpr bool crosses = (PHASE + 1) >= kPhases;
        if constexpr (kOwner<ahead> == Core) {
            if (!crosses || quant_next) {
                RotateAndQuantise<ahead>();
            }
        }
        if constexpr (kOwner<PHASE> == Core) {
            StoreChunk<PHASE>();
        }
    }

    __aicore__ inline void ExecuteCube(std::uint32_t strips, __gm__ const std::int32_t* tables) {
        auto* l1_table = reinterpret_cast<__cbuf__ bfloat16_t*>(kL1Table);
        ascend_micro::stage_blocks_nz<kTableBlocks, kGroup>(reinterpret_cast<__gm__ const bfloat16_t*>(tables), kL1Table);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtEpilogue);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtEpilogue);

        using namespace AscendC::Te;
        auto ta_src = MakeTensor(MakeMemPtr(l1_table), MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTableRows, kGroup));
        auto ta_dst = MakeTensor(MakeMemPtr(cube_.l0_a), MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTableRows, kGroup));
        Copy(MakeCopy(CopyL12L0A{}), ta_dst, ta_src);
        auto tb_src = MakeTensor(MakeMemPtr(l1_table), MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTableRows));
        auto tb_dst = MakeTensor(MakeMemPtr(cube_.l0_b), MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTableRows));
        Copy(MakeCopy(CopyL12L0B{}), tb_dst, tb_src);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtEpilogue);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtEpilogue);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEvtL0);

        StagePage<0>(false);
        StagePage<1>(false);
        StagePage<2>(false);

        for (std::uint32_t strip = 0u; strip + 1u < strips; ++strip) {
            RunCubeSteadyPhase<0>(true);
            RunCubeSteadyPhase<1>(true);
            RunCubeSteadyPhase<2>(true);
            RunCubeSteadyPhase<3>(true);
            RunCubeSteadyPhase<4>(true);
            RunCubeSteadyPhase<5>(true);
            RunCubeSteadyPhase<6>(true);
            RunCubeSteadyPhase<7>(true);
            cube_.strip_elems += kTokens * cube_.row_pitch;
        }
        RunCubeSteadyPhase<0>(false);
        RunCubeSteadyPhase<1>(false);
        RunCubeSteadyPhase<2>(false);
        RunCubeSteadyPhase<3>(false);
        RunCubeSteadyPhase<4>(false);
        RunCubeSteadyPhase<5>(false);
        RunCubeSteadyPhase<6>(false);
        RunCubeSteadyPhase<7>(false);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEvtL0);

        AscendC::CrossCoreWaitFlag<2, kSync2AicWait>(kSync2PanelFree);
        AscendC::CrossCoreWaitFlag<2, kSync2AicWait>(kSync2PanelFree);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(kEvtDrain);
        AscendC::SetFlag<AscendC::HardEvent::M_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::M_S>(kEvtDrain);
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtDrain);
        AscendC::CrossCoreSetFlag<2, kSync2AicAnnounce>(kSync2PageDrained);
    }

    template <std::uint32_t Core>
    __aicore__ inline void ExecuteVector(__gm__ const std::int32_t* slot_mapping, std::uint32_t num_kv_heads,
                                         std::uint32_t block_size, std::uint32_t num_blocks, std::uint32_t strips) {
        AscendC::Duplicate(View<float>(kUbOne, kScaleFloats), 1.0f, static_cast<std::int32_t>(kScaleFloats));
        AscendC::Duplicate(View<std::int32_t>(kUbExpMask, kScaleFloats), static_cast<std::int32_t>(0x7F800000), static_cast<std::int32_t>(kScaleFloats));
        AscendC::Duplicate(View<std::int32_t>(kUbInvSeed, kScaleFloats), static_cast<std::int32_t>(0x7F000000), static_cast<std::int32_t>(kScaleFloats));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
        AscendC::CrossCoreSetFlag<2, kSync2AivAnnounce>(kSync2PanelFreePrime);
        AscendC::CrossCoreSetFlag<2, kSync2AivAnnounce>(kSync2PanelFreePrime);

        if constexpr (kOwner<0> == Core) RotateAndQuantise<0>();

        for (std::uint32_t strip = 0u; strip + 1u < strips; ++strip) {
            const StripAddr a = CalculateStripAddr(slot_mapping, strip, num_kv_heads, vector_.head_bytes, block_size, num_blocks);
            vector_.page_bytes = a.page_bytes;
            vector_.key_scale = a.key_scale;
            vector_.value_scale = a.value_scale;
            RunVectorSteadyPhase<0, Core>(true);
            RunVectorSteadyPhase<1, Core>(true);
            RunVectorSteadyPhase<2, Core>(true);
            RunVectorSteadyPhase<3, Core>(true);
            RunVectorSteadyPhase<4, Core>(true);
            RunVectorSteadyPhase<5, Core>(true);
            RunVectorSteadyPhase<6, Core>(true);
            RunVectorSteadyPhase<7, Core>(true);
        }
        {
            const StripAddr a = CalculateStripAddr(slot_mapping, strips - 1u, num_kv_heads, vector_.head_bytes, block_size, num_blocks);
            vector_.page_bytes = a.page_bytes;
            vector_.key_scale = a.key_scale;
            vector_.value_scale = a.value_scale;
            RunVectorSteadyPhase<0, Core>(false);
            RunVectorSteadyPhase<1, Core>(false);
            RunVectorSteadyPhase<2, Core>(false);
            RunVectorSteadyPhase<3, Core>(false);
            RunVectorSteadyPhase<4, Core>(false);
            RunVectorSteadyPhase<5, Core>(false);
            RunVectorSteadyPhase<6, Core>(false);
            RunVectorSteadyPhase<7, Core>(false);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);

        AscendC::CrossCoreWaitFlag<2, kSync2AivWait>(kSync2PageDrained);
        if constexpr (Core == 0u) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtDrain);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtDrain);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(kEvtDrain);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(kEvtDrain);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtDrain);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtDrain);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(kEvtDrain);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(kEvtDrain);
        }
    }

    CubeState cube_;
    VectorState vector_;
};

__aicore__ inline void KernelBody(
    __gm__ bfloat16_t* __restrict__ key,
    __gm__ bfloat16_t* __restrict__ value,
    __gm__ std::int8_t* __restrict__ keyCache,
    __gm__ std::int8_t* __restrict__ valueCache,
    __gm__ float* __restrict__ scaleCache,
    __gm__ std::int32_t* __restrict__ slotMapping,
    __gm__ float* __restrict__ piSigns,
    __gm__ std::int32_t* __restrict__ tables,
    uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,
    uint32_t blockSize, uint32_t numBlocks, uint32_t tokensPerCore,
    float invSqrtLen) {

    const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx() / AscendC::GetSubBlockNum());
    const uint32_t total_blocks = turboquant_wave::TotalStrips(numTokens, tokensPerCore);
    if (turboquant_wave::BlockOutOfStrips(block, total_blocks)) return;
    const uint32_t strips = turboquant_wave::StripsPerBlock(tokensPerCore);
    if (strips == 0u) return;

    AscendC::InitSocState();

    const uint32_t first_token = block * tokensPerCore;
    key += first_token * numKvHeads * headSize;
    value += first_token * numKvHeads * headSize;
    slotMapping += first_token;
    (void)piSigns;

    TurboQuantReshapePipeline<bfloat16_t> op;

    if ASCEND_IS_AIC {
        __ca__ bfloat16_t l0_a[kL0Elems] __attribute__((aligned(512)));
        __cb__ bfloat16_t l0_b[kL0Elems] __attribute__((aligned(512)));
        __cc__ float l0_c[kL0CElems] __attribute__((aligned(1024)));
        op.cube_.l0_a = l0_a;
        op.cube_.l0_b = l0_b;
        op.cube_.l0_c = l0_c;
        op.InitCube(key, value, numKvHeads, headSize);
        op.ExecuteCube(strips, tables);
    } else {
        op.InitVector(keyCache, valueCache, scaleCache, blockSize, headSize, invSqrtLen);
        if (AscendC::GetSubBlockIdx() == 0) {
            op.ExecuteVector<0>(slotMapping, numKvHeads, blockSize, numBlocks, strips);
        } else {
            op.ExecuteVector<1>(slotMapping, numKvHeads, blockSize, numBlocks, strips);
        }
    }
}
}

#endif  // __CCE_AICORE__ -- everything above is the device-side body

// ------------------------------------------------------------- entry point
// THE ONE STANDING EXCEPTION TO THE MIRROR. The `__global__` entry point below
// is compiled in EVERY pass of this translation unit, host and device alike;
// only its *body* stays behind __CCE_AICORE__. turboquant-core keeps the
// definition inside the device arm and a bare prototype in the host arm, which
// is correct for ITS build and wrong for this one. Do not re-mirror it away.
//
// CANN 9.2 builds this file with the ASC language toolchain (bisheng,
// --npu-arch=dav-3510, see the root CMakeLists), which is single-source: one
// driver invocation runs a host pass and two device passes over the same text,
// and the host-side launch stub of a `__global__` function is emitted by the
// host pass from that function's DEFINITION. A prototype is not enough. The
// legacy `ascendc_library` the standalone repo uses does not care, because
// there a generated host_stub.cpp supplies the stubs instead.
//
// With the definition inside the device-only region the host pass sees only a
// prototype, emits no stub, and the triple-chevron launch below leaves
// `turboquant_reshape_and_cache_bf16` UNDEFINED in
// libvllm_ascend_turboquant.so itself. The library still links and every
// launch_* entry is present, so the only symptom is an undefined symbol when
// Python dlopen()s the extension. `nm -D` on the library is what shows it,
// with the kernels marked U beside the T launchers that call them -- measured
// exactly that way on the first build of this port.
// Guarded for BOTH passes of the kernel build and NEITHER pass of a plain host
// compiler. ASCEND_MICRO_CANN_LAUNCH is set by the root CMakeLists on every pass
// of this translation unit (once in ascendc_compile_definitions, once behind
// -forward-options-to-host-compiler), so the ASC host pass reaches this
// definition and emits the launch stub, while the device pass reaches it through
// __CCE_AICORE__. A device-free gate that `#include`s this file with g++ --
// tests/ut/ops/turboquant/test_contract_gates.cpp does, for the arena
// static_asserts -- defines neither, and so never sees `__global__`,
// `__aicore__`, `__gm__` or `bfloat16_t`, none of which a plain compiler has.
#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_CANN_LAUNCH)
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_bf16(
    // The activations are bfloat16_t; the caches stay INT8, because they hold
    // packed INT4 nibbles and the output contract does not move with the input
    // precision.
    __gm__ bfloat16_t* __restrict__ key, __gm__ bfloat16_t* __restrict__ value,
    __gm__ std::int8_t* __restrict__ keyCache, __gm__ std::int8_t* __restrict__ valueCache,
    __gm__ float* __restrict__ scaleCache, __gm__ std::int32_t* __restrict__ slotMapping,
    __gm__ float* __restrict__ piSigns, __gm__ std::int32_t* __restrict__ tables,
    uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,
    uint32_t blockSize, uint32_t numBlocks, uint32_t tokensPerCore, float invSqrtLen) {
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    turboquant_bf16::KernelBody(key, value, keyCache, valueCache, scaleCache, slotMapping,
                                piSigns, tables, numTokens, numKvHeads, headSize, blockSize,
                                numBlocks, tokensPerCore, invSqrtLen);
#else
    (void)key; (void)value; (void)keyCache; (void)valueCache; (void)scaleCache;
    (void)slotMapping; (void)piSigns; (void)tables; (void)numTokens; (void)numKvHeads;
    (void)headSize; (void)blockSize; (void)numBlocks; (void)tokensPerCore; (void)invSqrtLen;
#endif
}

#endif  // the entry point

#if defined(ASCEND_MICRO_CANN_LAUNCH) && !defined(__CCE_AICORE__)
#include <stdexcept>
// No prototype for the kernel here: its definition above is visible in this
// pass, which is what makes the host launch stub exist at all.

extern "C" void launch_turboquant_reshape_and_cache_bf16(
    void* stream, const std::uint16_t* key, const std::uint16_t* value,
    std::int8_t* key_cache, std::int8_t* value_cache, float* scale_cache,
    const std::int32_t* slot_mapping, const float* pi_signs, const std::int32_t* tables,
    std::uint32_t num_tokens, std::uint32_t num_kv_heads, std::uint32_t head_size,
    std::uint32_t block_size, std::uint32_t num_blocks, std::uint32_t tokens_per_core, float inv_sqrt_len) {
    if (!turboquant_wave::WholeStripShare(tokens_per_core))
        throw std::invalid_argument("reshape_and_cache_bf16: tokensPerCore error");
    if (!turboquant_wave::WholeStrips(num_tokens, tokens_per_core))
        throw std::invalid_argument("reshape_and_cache_bf16: numTokens error");
#if TURBOQUANT_INGRESS_NZ
    // The NZ cache lays a physical block out as whole 64-row tiles, and the writer's chunk
    // geometry is compiled for kHeads kv heads of kHeadSize channels. Both are refused here
    // rather than producing a cache the decode would read at the wrong byte.
    if (block_size % turboquant_bf16::kNzTileRows != 0u)
        throw std::invalid_argument("reshape_and_cache_bf16: blockSize must be a multiple of 64 "
                                    "for the NZ-tiled ingress");
    if (num_kv_heads != turboquant_bf16::kHeads || head_size != turboquant_bf16::kHeadSize)
        throw std::invalid_argument("reshape_and_cache_bf16: the NZ-tiled ingress is compiled "
                                    "for 4 kv heads of 256 channels");
#endif
    const std::uint32_t grid = turboquant_wave::TotalStrips(num_tokens, tokens_per_core);
    turboquant_reshape_and_cache_bf16<<<grid, nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(const_cast<std::uint16_t*>(key)),
        reinterpret_cast<bfloat16_t*>(const_cast<std::uint16_t*>(value)), key_cache,
        value_cache, scale_cache, const_cast<std::int32_t*>(slot_mapping), const_cast<float*>(pi_signs),
        const_cast<std::int32_t*>(tables), num_tokens, num_kv_heads, head_size, block_size, num_blocks,
        tokens_per_core, inv_sqrt_len);
}
#endif
