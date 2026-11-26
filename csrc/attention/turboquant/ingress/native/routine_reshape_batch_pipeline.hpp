#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "routine_stage_page_zn_bf16.hpp"
#include "routine_vector_rotate_butterfly.hpp"

#define TQ_AICORE __aicore__
#else
#define TQ_AICORE
#endif

#include "../../common/turboquant_wave_protocol.hpp"
#include "../../common/turboquant_layout.h"
#include "../../common/turboquant_mode.h"

#ifndef TURBOQUANT_INGRESS_NZ
#define TURBOQUANT_INGRESS_NZ 0
#endif

namespace turboquant_bf16 {

[[maybe_unused]] constexpr std::uint32_t kTokens = 16;
[[maybe_unused]] constexpr std::uint32_t kHeads = 4;
[[maybe_unused]] constexpr std::uint32_t kHeadSize = 256;
[[maybe_unused]] constexpr std::uint32_t kGroup = 32;
[[maybe_unused]] constexpr std::uint32_t kGroups = kHeadSize / kGroup;
[[maybe_unused]] constexpr std::uint32_t kChunks = 2u * kHeads;
[[maybe_unused]] constexpr std::uint32_t kPanel = kTokens * kHeadSize;
[[maybe_unused]] constexpr std::uint32_t kPackedBytes = kPanel / 2u;
[[maybe_unused]] constexpr std::uint32_t kScaleFloats = kTokens;

[[maybe_unused]] constexpr std::uint32_t kElemBytes = 2u;
[[maybe_unused]] constexpr std::uint32_t kFractalK = 32u / kElemBytes;
[[maybe_unused]] constexpr std::uint32_t kFractalsPerGroup = kGroup / kFractalK;

[[maybe_unused]] constexpr std::uint32_t kBlockElems = kGroup * kGroup;
[[maybe_unused]] constexpr std::uint32_t kTableBlocks = 2u * kGroups;
[[maybe_unused]] constexpr std::uint32_t kTableElems = kTableBlocks * kBlockElems;
[[maybe_unused]] constexpr std::uint32_t kTableBytes = kTableElems * kElemBytes;
[[maybe_unused]] constexpr std::uint32_t kTableRows = kTableBlocks * kGroup;

[[maybe_unused]] constexpr std::uint32_t kStageDepth = 4;
[[maybe_unused]] constexpr std::uint32_t kLandDepth = 2;
[[maybe_unused]] constexpr std::uint32_t kPackDepth = 2;

[[maybe_unused]] constexpr std::uint32_t kPhases = kChunks;
[[maybe_unused]] constexpr std::uint32_t kIngressLookahead = kStageDepth - 1u;

#if TURBOQUANT_INGRESS_NZ

[[maybe_unused]] constexpr std::uint32_t kNzTileRows = 64u;
[[maybe_unused]] constexpr std::uint32_t kNzC0 = 32u;
[[maybe_unused]] constexpr std::uint32_t kNzRowBytes = kHeadSize / 2u;
[[maybe_unused]] constexpr std::uint32_t kNzGroups = kNzRowBytes / kNzC0;
[[maybe_unused]] constexpr std::uint32_t kNzHeadBytes = kNzTileRows * kNzRowBytes;
[[maybe_unused]] constexpr std::uint32_t kNzGroupBytes = kNzTileRows * kNzC0;
[[maybe_unused]] constexpr std::uint32_t kNzScaleSlot = ((2u * kHeads + 7u) / 8u) * 8u;
[[maybe_unused]] constexpr std::uint32_t kNzScaleSlabFloats = kTokens * 8u;

namespace tq_contract = vllm_ascend::turboquant;

[[maybe_unused]] constexpr std::uint32_t kNzBurstBytes = kTokens * kNzC0;

TQ_AICORE constexpr std::uint64_t NzStripBaseWide(std::uint32_t slot, std::uint32_t heads)
{
    return static_cast<std::uint64_t>(slot - slot % kNzTileRows) * heads * kNzRowBytes +
           (slot % kNzTileRows) * kNzC0;
}

TQ_AICORE constexpr bool NzByteAgrees(std::uint32_t slot, std::uint32_t head, std::uint32_t group,
                            std::uint32_t row, std::uint32_t byte)
{
    return NzStripBaseWide(slot, kHeads) + head * kNzHeadBytes +
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
                if (static_cast<std::uint64_t>(slot) * kNzScaleSlot + kv * kHeads + head + row * kNzScaleSlot !=
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

template <typename Offset>
struct StripAddress {
    Offset page_bytes;
    Offset key_scale;
    Offset value_scale;
};

using StripAddr = StripAddress<std::uint64_t>;

TQ_AICORE constexpr StripAddr CalculateStripOffsets(std::uint32_t slot0, std::uint32_t num_kv_heads,
                                                   std::uint64_t head_bytes, std::uint32_t block_size,
                                                   std::uint32_t num_blocks) {
#if TURBOQUANT_INGRESS_NZ
    (void)head_bytes;
    (void)block_size;
    (void)num_blocks;
    const std::uint64_t key_scale = static_cast<std::uint64_t>(slot0) * kNzScaleSlot;
    return {NzStripBaseWide(slot0, num_kv_heads), key_scale, key_scale + num_kv_heads};
#else
    const std::uint64_t block_idx = slot0 / block_size;
    const std::uint32_t strip_in_block = (slot0 % block_size) / kTokens;
    const std::uint32_t strip_bytes = strip_in_block * kPackedBytes;
    return {block_idx * num_kv_heads * head_bytes + strip_bytes,
            block_idx * num_kv_heads * block_size + strip_in_block * kTokens,
            (num_blocks + block_idx) * num_kv_heads * block_size + strip_in_block * kTokens};
#endif
}

#if !defined(__CCE_AICORE__)
#if TURBOQUANT_INGRESS_NZ
constexpr std::uint32_t kFrozenPageMultiple = kNzTileRows;
#else
constexpr std::uint32_t kFrozenPageMultiple = kTokens;
#endif

[[maybe_unused]] constexpr bool FrozenBatchGeometry(
    std::uint32_t num_tokens, std::uint32_t num_kv_heads, std::uint32_t head_size,
    std::uint32_t block_size, std::uint32_t num_blocks, std::uint32_t tokens_per_core)
{
    constexpr std::uint64_t addressable = std::uint64_t{1} << 32u;
    constexpr std::uint64_t packed_bytes_per_slot = kHeads * (kHeadSize / 2u);

    return num_tokens != 0u && num_kv_heads == kHeads && head_size == kHeadSize &&
           block_size != 0u && num_blocks != 0u && block_size % kFrozenPageMultiple == 0u &&
           tokens_per_core <= 6u * kTokens && turboquant_wave::WholeStripShare(tokens_per_core) &&
           turboquant_wave::WholeStrips(num_tokens, tokens_per_core) &&
           static_cast<std::uint64_t>(block_size) * num_blocks <=
               addressable / packed_bytes_per_slot &&
           static_cast<std::uint64_t>(num_tokens) * kHeads * kHeadSize <= addressable;
}

constexpr std::uint32_t kFrozenTestBlocks = (1u << 23u) / kFrozenPageMultiple;
static_assert(FrozenBatchGeometry(96u, 4u, 256u, kFrozenPageMultiple, kFrozenTestBlocks, 96u));
static_assert(!FrozenBatchGeometry(96u, 4u, 256u, kFrozenPageMultiple, kFrozenTestBlocks + 1u, 96u));
static_assert(FrozenBatchGeometry(1u << 22u, 4u, 256u, kFrozenPageMultiple, kFrozenTestBlocks, 16u));
static_assert(!FrozenBatchGeometry((1u << 22u) + 16u, 4u, 256u, kFrozenPageMultiple, kFrozenTestBlocks, 16u));
static_assert(!FrozenBatchGeometry(112u, 4u, 256u, kFrozenPageMultiple, kFrozenTestBlocks, 112u));
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

[[maybe_unused]] constexpr std::uint32_t ParityOf(std::uint32_t v) {
    std::uint32_t parity = 0;
    while (v != 0u) { parity ^= (v & 1u); v >>= 1; }
    return parity;
}
template <std::uint32_t C1, std::uint32_t G>
[[maybe_unused]] constexpr std::uint32_t kSignBlock = ParityOf(C1 & G) * kGroups + G;

using turboquant_wave::kBankSkewBytes;
using turboquant_wave::kBlock;
using turboquant_wave::kReadPadBytes;
using turboquant_wave::kUbWindow;
using turboquant_wave::NextSlot;
using turboquant_wave::RoundUpBlock;
[[maybe_unused]] constexpr std::uint32_t kLanesPerBlock = kBlock / sizeof(float);

[[maybe_unused]] constexpr std::uint32_t kPanelBytes = kPanel * 4u;
[[maybe_unused]] constexpr std::uint32_t kLandSlotBytes = kPanelBytes + kBankSkewBytes;
#if TURBOQUANT_INGRESS_NZ

[[maybe_unused]] constexpr std::uint32_t kPackSlotPayload = kPackedBytes + kNzScaleSlabFloats * 4u;
#else
[[maybe_unused]] constexpr std::uint32_t kPackSlotPayload = kPackedBytes + kScaleFloats * 4u;
#endif
[[maybe_unused]] constexpr std::uint32_t kPackSlotBytes = RoundUpBlock(kPackSlotPayload) + kBankSkewBytes;

[[maybe_unused]] constexpr std::uint32_t kUbLandRing = 0u;
[[maybe_unused]] constexpr std::uint32_t kUbLandBytes = kLandDepth * kLandSlotBytes;
[[maybe_unused]] constexpr std::uint32_t kUbCast = NextSlot(kUbLandRing, kUbLandBytes);
[[maybe_unused]] constexpr std::uint32_t kUbWork = NextSlot(kUbCast, kPanelBytes);
[[maybe_unused]] constexpr std::uint32_t kUbFold128 = NextSlot(kUbWork, kPanelBytes);
[[maybe_unused]] constexpr std::uint32_t kUbFold64 = NextSlot(kUbFold128, kTokens * 128u * 4u);
[[maybe_unused]] constexpr std::uint32_t kUbFold8 = NextSlot(kUbFold64, kTokens * 64u * 4u);
[[maybe_unused]] constexpr std::uint32_t kUbHalf = NextSlot(kUbFold8, kTokens * 8u * 4u);
[[maybe_unused]] constexpr std::uint32_t kUbAbsMax = NextSlot(kUbHalf, kPanel * 2u);
[[maybe_unused]] constexpr std::uint32_t kUbExpBits = NextSlot(kUbAbsMax, kScaleFloats * 4u);
[[maybe_unused]] constexpr std::uint32_t kUbPow = NextSlot(kUbExpBits, kScaleFloats * 4u);
[[maybe_unused]] constexpr std::uint32_t kUbInv = NextSlot(kUbPow, kScaleFloats * 4u);
[[maybe_unused]] constexpr std::uint32_t kUbInvRep = NextSlot(kUbInv, kScaleFloats * 4u);
[[maybe_unused]] constexpr std::uint32_t kUbInvBcast = NextSlot(kUbInvRep, 4u * kScaleFloats * 4u);
[[maybe_unused]] constexpr std::uint32_t kUbOne = NextSlot(kUbInvBcast, kTokens * kLanesPerBlock * 4u);
[[maybe_unused]] constexpr std::uint32_t kUbExpMask = NextSlot(kUbOne, kScaleFloats * 4u);
[[maybe_unused]] constexpr std::uint32_t kUbInvSeed = NextSlot(kUbExpMask, kScaleFloats * 4u);
#if TURBOQUANT_INGRESS_NZ

[[maybe_unused]] constexpr std::uint32_t kUbNzTmp = NextSlot(kUbInvSeed, kScaleFloats * 4u);
[[maybe_unused]] constexpr std::uint32_t kUbNzSrc = NextSlot(kUbNzTmp, kPanel * 2u);
[[maybe_unused]] constexpr std::uint32_t kUbWoven = NextSlot(kUbNzSrc, kPanel * 2u);
[[maybe_unused]] constexpr std::uint32_t kUbPackRing = NextSlot(kUbWoven, kPanel * 2u);
#else
[[maybe_unused]] constexpr std::uint32_t kUbPackRing = NextSlot(kUbInvSeed, kScaleFloats * 4u);
#endif
[[maybe_unused]] constexpr std::uint32_t kArenaEnd = kUbPackRing + kPackDepth * kPackSlotBytes;

template <std::uint32_t Slot> constexpr std::uint32_t kLandBase = kUbLandRing + Slot * kLandSlotBytes;
template <std::uint32_t Slot> constexpr std::uint32_t kPackBase = kUbPackRing + Slot * kPackSlotBytes;
template <std::uint32_t Slot> constexpr std::uint32_t kScaleBase = kPackBase<Slot> + kPackedBytes;

[[maybe_unused]] constexpr std::uint32_t kL1Table = 0u;
[[maybe_unused]] constexpr std::uint32_t kL1PageElems = kTokens * kHeadSize;
[[maybe_unused]] constexpr std::uint32_t kL1PageBytes = kL1PageElems * kElemBytes;
[[maybe_unused]] constexpr std::uint32_t kL1Ring = NextSlot(kL1Table, kTableBytes);
[[maybe_unused]] constexpr std::uint32_t kL1SlotBytes = kL1PageBytes + kBankSkewBytes;
[[maybe_unused]] constexpr std::uint32_t kL1End = kL1Ring + kStageDepth * kL1SlotBytes;
[[maybe_unused]] constexpr std::uint32_t kFractalElems = kTokens * kGroup;
[[maybe_unused]] constexpr std::uint32_t kFractalBytes = kFractalElems * kElemBytes;

template <std::uint32_t Slot> constexpr std::uint32_t kL1PageBase = kL1Ring + Slot * kL1SlotBytes;

[[maybe_unused]] constexpr std::uint32_t kL0TableElems = kTableElems;
[[maybe_unused]] constexpr std::uint32_t kL0PageElems = kGroups * kFractalElems;
[[maybe_unused]] constexpr std::uint32_t kL0Elems = kL0TableElems + kL0PageElems;
[[maybe_unused]] constexpr std::uint32_t kL0CElems = kGroups * kTokens * kGroup;

using turboquant_wave::kSync2AicAnnounce;
using turboquant_wave::kSync2AicWait;
using turboquant_wave::kSync2AivAnnounce;
using turboquant_wave::kSync2AivWait;
using turboquant_wave::kSync2PageDrained;
using turboquant_wave::kSync2PanelFree;
using turboquant_wave::kSync2PanelFreePrime;
using turboquant_wave::kSync2PanelReady;

[[maybe_unused]] constexpr std::uint32_t kGenerations = kChunks / 2u;
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

[[maybe_unused]] constexpr std::uint32_t kHalfPerBlock = kBlock / 2u;
[[maybe_unused]] constexpr std::uint32_t kNzPlaneElems = kPanel / 2u;
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

template <typename Offset>
struct VectorState {
    AscendC::GlobalTensor<std::int8_t> key_cache;
    AscendC::GlobalTensor<std::int8_t> value_cache;
    AscendC::GlobalTensor<float> scale_cache;
    Offset page_bytes;
    Offset head_bytes;
    Offset key_scale;
    Offset value_scale;
    std::uint32_t head_scale;

    std::uint32_t quant_mode;
    float inv_sqrt_len;
};

using LloydMaxKv4 =
    vllm_ascend::turboquant::TurboQuantLloydMaxTraits<vllm_ascend::turboquant::TurboQuantMode::KV4_FP8>;

using FrozenStripAddr = StripAddress<std::uint32_t>;

#if TURBOQUANT_INGRESS_NZ
TQ_AICORE constexpr std::uint32_t NzStripBase(std::uint32_t slot, std::uint32_t heads)
{
    return (slot - slot % kNzTileRows) * heads * kNzRowBytes + (slot % kNzTileRows) * kNzC0;
}
#endif

__aicore__ inline FrozenStripAddr CalculateStripAddr(__gm__ const std::int32_t* slot_mapping,
                                                      std::uint32_t strip, std::uint32_t num_kv_heads,
                                                      std::uint32_t head_bytes, std::uint32_t block_size,
                                                      std::uint32_t num_blocks) {
#if TURBOQUANT_INGRESS_NZ
    (void)head_bytes;
    (void)block_size;
    (void)num_blocks;
    const std::uint32_t slot0 = static_cast<std::uint32_t>(slot_mapping[strip * kTokens]);
    FrozenStripAddr a;
    a.page_bytes = NzStripBase(slot0, num_kv_heads);
    a.key_scale = slot0 * kNzScaleSlot;
    a.value_scale = a.key_scale + num_kv_heads;
    return a;
#else
    const std::uint32_t slot0 = static_cast<std::uint32_t>(slot_mapping[strip * kTokens]);
    const std::uint32_t block_idx = slot0 / block_size;
    const std::uint32_t strip_in_block = (slot0 % block_size) / kTokens;
    const std::uint32_t strip_bytes = strip_in_block * kPackedBytes;
    FrozenStripAddr a;
    a.page_bytes = block_idx * num_kv_heads * head_bytes + strip_bytes;
    a.key_scale = block_idx * num_kv_heads * block_size + strip_in_block * kTokens;
    a.value_scale = (num_blocks + block_idx) * num_kv_heads * block_size + strip_in_block * kTokens;
    return a;
#endif
}

__aicore__ inline StripAddr CalculateStripAddr(__gm__ const std::int32_t* slot_mapping,
                                               std::uint32_t strip, std::uint32_t num_kv_heads,
                                               std::uint64_t head_bytes, std::uint32_t block_size,
                                               std::uint32_t num_blocks) {
    const std::uint32_t slot0 = static_cast<std::uint32_t>(slot_mapping[strip * kTokens]);
    return CalculateStripOffsets(slot0, num_kv_heads, head_bytes, block_size, num_blocks);
}

[[maybe_unused]] constexpr auto kUb = AscendC::TPosition::VECCALC;

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

template <std::uint32_t... Steps>
__aicore__ inline void ValueFoldSumEmit(const AscendC::LocalTensor<float>& buf,
                                        std::integer_sequence<std::uint32_t, Steps...>) {
    (AscendC::Add(buf, buf, buf[kPanel >> (Steps + 1u)], static_cast<std::int32_t>(kPanel >> (Steps + 1u))), ...);
}

__simd_vf__ inline void __attribute__((used))
LloydMaxSumsqRowsVf(__ubuf__ float* sumsq_out, __ubuf__ float* squares, std::uint16_t rows)
{
    AscendC::Reg::MaskReg mask =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg lane0 =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
    ascend_micro::RotGatherTables t;
    ascend_micro::rot_build_gather_tables(t, mask);
    for (std::uint16_t r = 0; r < rows; ++r) {
        __ubuf__ float* row = squares + static_cast<std::uint32_t>(r) * kHeadSize;
        AscendC::Reg::RegTensor<float> m, v, partner;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(m, row);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(v, row + 64u);
        AscendC::Reg::Add<float>(m, m, v, mask);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(v, row + 128u);
        AscendC::Reg::Add<float>(m, m, v, mask);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(v, row + 192u);
        AscendC::Reg::Add<float>(m, m, v, mask);
        AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i0);
        AscendC::Reg::Add<float>(m, m, partner, mask);
        AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i1);
        AscendC::Reg::Add<float>(m, m, partner, mask);
        AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i2);
        AscendC::Reg::Add<float>(m, m, partner, mask);
        AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i3);
        AscendC::Reg::Add<float>(m, m, partner, mask);
        AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i4);
        AscendC::Reg::Add<float>(m, m, partner, mask);
        AscendC::Reg::Gather<float, std::uint32_t>(partner, m, t.i5);
        AscendC::Reg::Add<float>(m, m, partner, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            sumsq_out + static_cast<std::uint32_t>(r) * 8u, m, lane0);
    }
}

#if TURBOQUANT_INGRESS_NZ

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

static_assert(kTokens == 16u, "NzTokenMajor is four DeInterleave passes, i.e. log2(kTokens)");
#endif

template <std::uint32_t... Qs>
__aicore__ inline void KeyQuantiseEmit(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
                                       const AscendC::LocalTensor<float>& inv_bcast, const AscendC::BinaryRepeatParams& rp,
                                       std::integer_sequence<std::uint32_t, Qs...>) {
    (AscendC::Mul<float>(dst[Qs * 64u], src[Qs * 64u], inv_bcast, 64, static_cast<std::uint8_t>(kTokens), rp), ...);
}

template <typename Offset>
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
                                      std::uint32_t head_size, float inv_sqrt_len,
                                      std::uint32_t quant_mode) {
        vector_.key_cache.SetGlobalBuffer(key_cache);
        vector_.value_cache.SetGlobalBuffer(value_cache);
        vector_.scale_cache.SetGlobalBuffer(scale_cache);
        vector_.quant_mode = quant_mode;
#if TURBOQUANT_INGRESS_NZ

        (void)block_size;
        (void)head_size;
        vector_.head_bytes = kNzHeadBytes;
        vector_.head_scale = 1u;
#else
        if constexpr (std::is_same_v<Offset, std::uint32_t>) {
            vector_.head_bytes = block_size * head_size / 2u;
        } else {
            vector_.head_bytes = static_cast<std::uint64_t>(block_size) * head_size / 2u;
        }
        vector_.head_scale = block_size;
#endif
        vector_.inv_sqrt_len = inv_sqrt_len;
    }

    template <std::uint32_t PHASE>
    __aicore__ inline void StagePage(bool stage_next) {
        using namespace asc::te;
        constexpr std::uint32_t slot = kL1Slot<PHASE>;
        constexpr std::int32_t evt = static_cast<std::int32_t>(slot);
        std::uint32_t offset = cube_.strip_elems;
        if (stage_next) offset += kTokens * cube_.row_pitch;

        __gm__ const bfloat16_t* base = kIsKey<PHASE> ? cube_.key : cube_.value;
        auto* src_ptr = const_cast<__gm__ bfloat16_t*>(base) + offset + kChunkHead<PHASE> * cube_.head_size;
        auto* dst_ptr = reinterpret_cast<__cbuf__ bfloat16_t*>(kL1PageBase<slot>);

        auto shape = make_shape(make_shape(_1{}, kTokens), make_shape(_1{}, kHeadSize));
        auto stride = make_stride(make_stride(_0{}, cube_.row_pitch), make_stride(_0{}, _1{}));
        auto src = make_tensor(make_mem_ptr(src_ptr), make_pattern_layout<nd_ext_layout_ptn, layout_trait<AscendC::Std::ignore_t, _1>>(shape, stride));
        auto dst = make_tensor(make_mem_ptr(dst_ptr), make_frame_layout<nz_layout_ptn, bfloat16_t>(kTokens, kHeadSize));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(evt);
        copy(make_copy(copy_gm_to_l1{}), dst, src);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(evt);
    }

    template <std::uint32_t PHASE, std::uint32_t G>
    __aicore__ inline void GemmGroupLoad(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b) {
        using namespace asc::te;
        constexpr std::uint32_t slot = kL1Slot<PHASE>;
        auto* page = reinterpret_cast<__cbuf__ bfloat16_t*>(kL1PageBase<slot> + G * kFractalBytes);
        if constexpr (kIsKey<PHASE>) {
            auto src = make_tensor(make_mem_ptr(page), make_frame_layout<nz_layout_ptn, bfloat16_t>(kTokens, kGroup));
            auto dst = make_tensor(make_mem_ptr(l0a + kL0TableElems + G * kFractalElems), make_frame_layout<nz_layout_ptn, bfloat16_t>(kTokens, kGroup));
            copy(make_copy(copy_l1_to_l0a{}), dst, src);
        } else {
            auto src = make_tensor(make_mem_ptr(page), make_frame_layout<zn_layout_ptn, bfloat16_t>(kGroup, kTokens));
            auto dst = make_tensor(make_mem_ptr(l0b + kL0TableElems + G * kFractalElems), make_frame_layout<zn_layout_ptn, bfloat16_t>(kGroup, kTokens));
            copy(make_copy(copy_l1_to_l0b{}), dst, src);
        }
    }

    template <std::uint32_t PHASE, std::uint32_t C1, std::uint32_t G>
    __aicore__ inline void GemmBlock(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b, __cc__ float* l0c) {
        using namespace asc::te;
        constexpr std::uint32_t block = kSignBlock<C1, G> * kBlockElems;
        constexpr bool init = (G == 0u);
        auto* c_ptr = l0c + C1 * kTokens * kGroup;

        if constexpr (kIsKey<PHASE>) {
            auto a = make_tensor(make_mem_ptr(l0a + kL0TableElems + G * kFractalElems), make_frame_layout<nz_layout_ptn, bfloat16_t>(kTokens, kGroup));
            auto b = make_tensor(make_mem_ptr(l0b + block), make_frame_layout<zn_layout_ptn, bfloat16_t>(kGroup, kGroup));
            auto c = make_tensor(make_mem_ptr(c_ptr), make_frame_layout<nz_layout_ptn, 16>(kTokens, kGroup));
            mmad_params mp(static_cast<std::uint16_t>(kTokens), static_cast<std::uint16_t>(kGroup), static_cast<std::uint16_t>(kGroup), unit_flag_mode::disable, init);
            mmad(make_mmad(mmad_operation{}, mmad_trait_default{}).with(mp), c, a, b);
        } else {
            auto a = make_tensor(make_mem_ptr(l0a + block), make_frame_layout<nz_layout_ptn, bfloat16_t>(kGroup, kGroup));
            auto b = make_tensor(make_mem_ptr(l0b + kL0TableElems + G * kFractalElems), make_frame_layout<zn_layout_ptn, bfloat16_t>(kGroup, kTokens));
            auto c = make_tensor(make_mem_ptr(c_ptr), make_frame_layout<nz_layout_ptn, 16>(kGroup, kTokens));
            mmad_params mp(static_cast<std::uint16_t>(kGroup), static_cast<std::uint16_t>(kTokens), static_cast<std::uint16_t>(kGroup), unit_flag_mode::disable, init);
            mmad(make_mmad(mmad_operation{}, mmad_trait_default{}).with(mp), c, a, b);
        }
    }

    template <std::uint32_t PHASE>
    __aicore__ inline void FixpipeChunk(__cc__ float* l0c) {
        using namespace asc::te;
        constexpr std::uint32_t land = kLandSlot<PHASE>;
        auto* ub = reinterpret_cast<__ubuf__ float*>(kLandBase<land>);
        l0c_to_ub_params fp;
        fp.unit_flag = unit_flag_mode::disable;
        fp.sub_block_id = (kOwner<PHASE> == 1u);
        if constexpr (kIsKey<PHASE>) {
            auto c = make_tensor(make_mem_ptr(l0c), make_frame_layout<nz_layout_ptn, 16>(kTokens, kHeadSize));
            auto y = make_tensor(make_mem_ptr(ub), make_frame_layout<nd_ext_layout_ptn>(kTokens, kHeadSize));
            copy(make_copy(copy_l0c_to_ub{}).with(fp), y, c);
        } else {
            auto c = make_tensor(make_mem_ptr(l0c), make_frame_layout<nz_layout_ptn, 16>(kHeadSize, kTokens));
            auto y = make_tensor(make_mem_ptr(ub), make_frame_layout<nd_ext_layout_ptn>(kHeadSize, kTokens));
            copy(make_copy(copy_l0c_to_ub{}).with(fp), y, c);
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
    __aicore__ inline void LloydMaxQuantisePanel() {
        constexpr float kLmEps = 1e-20f;
        static constexpr AscendC::SqrtConfig kSqrt0 = {AscendC::SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};
        static constexpr AscendC::DivConfig kDiv0 = {AscendC::DivAlgo::DIFF_COMPENSATION};
        constexpr std::int32_t all = static_cast<std::int32_t>(kPanel);
        constexpr std::int32_t lanes = static_cast<std::int32_t>(kScaleFloats);

        static_assert(kUbFold128 + kPanel * 4u <= kUbHalf + kPanel * 2u,
                      "the Lloyd-Max binning scratch must fit the fold and half buffers");
        auto cast = View<float>(kUbCast, kPanel);
        auto work = View<float>(kUbWork, kPanel);
        auto sumsq = View<float>(kUbAbsMax, kScaleFloats);
        auto sigma = View<float>(kUbPow, kScaleFloats);
        auto invn = View<float>(kUbInv, kScaleFloats);
        auto quant = View<std::int32_t>(kUbCast, kPanel);

        AscendC::Mul(work, cast, cast, all);
        if constexpr (kIsKey<PHASE>) {

            auto f64 = View<float>(kUbFold64, kTokens * 64u);
            LloydMaxSumsqRowsVf(reinterpret_cast<__ubuf__ float*>(f64.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float*>(work.GetPhyAddr()),
                                static_cast<std::uint16_t>(kTokens));
            AscendC::PipeBarrier<PIPE_V>();

            auto gather_offsets = View<std::int32_t>(kUbExpBits, kScaleFloats);
            AscendC::ArithProgression(gather_offsets, 0, 32, static_cast<std::int32_t>(kTokens));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(sumsq, f64, gather_offsets.ReinterpretCast<std::uint32_t>(), 0u,
                            static_cast<std::int32_t>(kTokens));
        } else {
            ValueFoldSumEmit(work, std::make_integer_sequence<std::uint32_t, 8u>{});
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Muls(sumsq, work, 1.0f, lanes);
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Adds(sumsq, sumsq, kLmEps, lanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt<float, kSqrt0>(sigma, sumsq, lanes);
        AscendC::Muls(sigma, sigma, vector_.inv_sqrt_len, lanes);
        AscendC::Duplicate(invn, -1.0f, lanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div<float, kDiv0>(invn, invn, sigma, lanes);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::BinaryRepeatParams bc;
        bc.dstBlkStride = 1; bc.src0BlkStride = 1; bc.dstRepStride = 8; bc.src0RepStride = 8;
        if constexpr (kIsKey<PHASE>) {
            auto inv_bcast = View<float>(kUbInvBcast, kTokens * kLanesPerBlock);
            AscendC::Brcb(inv_bcast, invn, 2, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            bc.src1BlkStride = 0; bc.src1RepStride = 1;
            bc.dstRepStride = static_cast<std::uint8_t>(kHeadSize / kLanesPerBlock);
            bc.src0RepStride = bc.dstRepStride;
            KeyQuantiseEmit(work, cast, inv_bcast, bc, std::make_integer_sequence<std::uint32_t, kHeadSize / 64u>{});
        } else {
            auto inv_rep = View<float>(kUbInvRep, 4u * kScaleFloats);
            AscendC::Muls(inv_rep, invn, 1.0f, lanes);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(inv_rep[kScaleFloats], inv_rep, 1.0f, lanes);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(inv_rep[2u * kScaleFloats], inv_rep, 1.0f, static_cast<std::int32_t>(2u * kScaleFloats));
            AscendC::PipeBarrier<PIPE_V>();
            bc.src1BlkStride = 1; bc.src1RepStride = 0;
            AscendC::Mul<float>(work, cast, inv_rep, 64, static_cast<std::uint8_t>(kPanel / 64u), bc);
        }
        AscendC::PipeBarrier<PIPE_V>();

        auto tmp = View<float>(kUbFold128, kPanel);
        AscendC::Duplicate(quant, -8, all);
        AscendC::PipeBarrier<PIPE_V>();
        for (int i = 0; i < LloydMaxKv4::kThresholdCount; ++i) {
            AscendC::Adds(tmp, work, LloydMaxKv4::kThresholds[i], all);

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftRight(tmp.ReinterpretCast<std::uint32_t>(),
                                tmp.ReinterpretCast<std::uint32_t>(), 31u, all);

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(quant, quant, tmp.ReinterpretCast<std::int32_t>(), all);

            AscendC::PipeBarrier<PIPE_V>();
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

        auto quant = View<std::int32_t>(kUbCast, kPanel);
        auto staged = View<half>(kUbHalf, kPanel);
        auto packed = View<std::int8_t>(kPackBase<pack>, kPackedBytes);

        auto pow_two = View<float>(kUbPow, kScaleFloats);
        if (vector_.quant_mode == 0u) {
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
            auto inv_bits = View<std::int32_t>(kUbInv, kScaleFloats);
            auto inv = View<float>(kUbInv, kScaleFloats);
            constexpr std::int32_t lanes = static_cast<std::int32_t>(kScaleFloats);

            AscendC::And(exp_bits, abs_max.template ReinterpretCast<std::int32_t>(), exp_mask, lanes);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(pow_bits, exp_bits, static_cast<std::int32_t>(-0x01000000), lanes);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(inv_bits, inv_seed, pow_bits, lanes);

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
        } else {

            LloydMaxQuantisePanel<PHASE>();
        }
#if TURBOQUANT_INGRESS_NZ
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(staged, quant, AscendC::RoundMode::CAST_RINT, kPanel);

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
        AscendC::Muls(scale, pow_two, vector_.inv_sqrt_len,
                      static_cast<std::int32_t>(kScaleFloats));
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
        const Offset page = vector_.page_bytes + kChunkHead<PHASE> * kNzHeadBytes;
        const Offset slice = (kIsKey<PHASE> ? vector_.key_scale : vector_.value_scale) + kChunkHead<PHASE> * vector_.head_scale;

        AscendC::DataCopyExtParams plane{static_cast<std::uint16_t>(kNzGroups), kNzBurstBytes, 0,
                                         kNzGroupBytes - kNzBurstBytes, 0};
        if constexpr (kIsKey<PHASE>) {
            AscendC::DataCopyPad(vector_.key_cache[page], packed, plane);
        } else {
            AscendC::DataCopyPad(vector_.value_cache[page], packed, plane);
        }

        AscendC::DataCopyExtParams lanes{static_cast<std::uint16_t>(kTokens),
                                         static_cast<std::uint32_t>(sizeof(float)), 0,
                                         (kNzScaleSlot - 1u) * static_cast<std::uint32_t>(sizeof(float)), 0};
        AscendC::DataCopyPad(vector_.scale_cache[slice], slab, lanes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pack_evt);
#else
        auto scale = View<float>(kScaleBase<pack>, kScaleFloats);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pack_evt);
        const Offset page = vector_.page_bytes + kChunkHead<PHASE> * vector_.head_bytes;
        const Offset slice = (kIsKey<PHASE> ? vector_.key_scale : vector_.value_scale) + kChunkHead<PHASE> * vector_.head_scale;
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

        using namespace asc::te;
        auto ta_src = make_tensor(make_mem_ptr(l1_table), make_frame_layout<nz_layout_ptn, bfloat16_t>(kTableRows, kGroup));
        auto ta_dst = make_tensor(make_mem_ptr(cube_.l0_a), make_frame_layout<nz_layout_ptn, bfloat16_t>(kTableRows, kGroup));
        copy(make_copy(copy_l1_to_l0a{}), ta_dst, ta_src);
        auto tb_src = make_tensor(make_mem_ptr(l1_table), make_frame_layout<zn_layout_ptn, bfloat16_t>(kGroup, kTableRows));
        auto tb_dst = make_tensor(make_mem_ptr(cube_.l0_b), make_frame_layout<zn_layout_ptn, bfloat16_t>(kGroup, kTableRows));
        copy(make_copy(copy_l1_to_l0b{}), tb_dst, tb_src);
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

        StripAddress<Offset> current = CalculateStripAddr(slot_mapping, 0u, num_kv_heads,
            vector_.head_bytes, block_size, num_blocks);
        if constexpr (kOwner<0> == Core) RotateAndQuantise<0>();

        for (std::uint32_t strip = 0u; strip + 1u < strips; ++strip) {
            vector_.page_bytes = current.page_bytes;
            vector_.key_scale = current.key_scale;
            vector_.value_scale = current.value_scale;
            RunVectorSteadyPhase<0, Core>(true);
            RunVectorSteadyPhase<1, Core>(true);
            RunVectorSteadyPhase<2, Core>(true);
            RunVectorSteadyPhase<3, Core>(true);
            RunVectorSteadyPhase<4, Core>(true);

            const StripAddress<Offset> next = CalculateStripAddr(slot_mapping, strip + 1u,
                num_kv_heads, vector_.head_bytes, block_size, num_blocks);
            RunVectorSteadyPhase<5, Core>(true);
            RunVectorSteadyPhase<6, Core>(true);
            RunVectorSteadyPhase<7, Core>(true);
            current = next;
        }
        {
            vector_.page_bytes = current.page_bytes;
            vector_.key_scale = current.key_scale;
            vector_.value_scale = current.value_scale;
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
    VectorState<Offset> vector_;
};

template <typename Offset>
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
    float invSqrtLen, uint32_t quantMode) {

    const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx() / AscendC::GetSubBlockNum());
    const uint32_t total_blocks = turboquant_wave::TotalStrips(numTokens, tokensPerCore);
    if (turboquant_wave::BlockOutOfStrips(block, total_blocks)) return;
    const uint32_t strips = turboquant_wave::StripsPerBlock(tokensPerCore);
    if (strips == 0u) return;

    if (quantMode > 1u) return;

    AscendC::InitSocState();

    const uint32_t first_token = block * tokensPerCore;
    if constexpr (std::is_same_v<Offset, std::uint32_t>) {
        key += first_token * numKvHeads * headSize;
        value += first_token * numKvHeads * headSize;
    } else {
        const std::uint64_t first_activation = turboquant_wave::ActivationOffset(first_token, numKvHeads, headSize);
        key += first_activation;
        value += first_activation;
    }
    slotMapping += first_token;
    (void)piSigns;

    TurboQuantReshapePipeline<Offset> op;

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
        op.InitVector(keyCache, valueCache, scaleCache, blockSize, headSize, invSqrtLen, quantMode);
        if (AscendC::GetSubBlockIdx() == 0) {
            op.template ExecuteVector<0>(slotMapping, numKvHeads, blockSize, numBlocks, strips);
        } else {
            op.template ExecuteVector<1>(slotMapping, numKvHeads, blockSize, numBlocks, strips);
        }
    }
}
}

#endif

#undef TQ_AICORE
