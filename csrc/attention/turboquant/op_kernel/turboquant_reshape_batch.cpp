#include <cstdint>
#include <utility>

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "cube/routine_stage_page_zn_bf16.hpp"
#endif

#include "common/turboquant_wave_protocol.hpp"

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
constexpr std::uint32_t kPackSlotBytes = RoundUpBlock(kPackedBytes + kScaleFloats * 4u) + kBankSkewBytes;

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
constexpr std::uint32_t kUbPackRing = NextSlot(kUbInvSeed, kScaleFloats * 4u);
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
        vector_.head_bytes = block_size * head_size / 2u;
        vector_.head_scale = block_size;
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
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pack_evt);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(staged, quant, AscendC::RoundMode::CAST_RINT, kPanel);
        AscendC::PipeBarrier<PIPE_V>();
        auto nibbles = packed.template ReinterpretCast<AscendC::int4b_t>();
        AscendC::Cast(nibbles, staged, AscendC::RoundMode::CAST_RINT, kPanel);
        auto scale = View<float>(kScaleBase<pack>, kScaleFloats);
        AscendC::Muls(scale, pow_two, vector_.inv_sqrt_len, lanes);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pack_evt);
    }

    template <std::uint32_t PHASE>
    __aicore__ inline void StoreChunk() {
        constexpr std::uint32_t pack = kPackSlot<PHASE>;
        constexpr std::int32_t pack_evt = static_cast<std::int32_t>(pack);
        auto packed = View<std::int8_t>(kPackBase<pack>, kPackedBytes);
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
    const std::uint32_t grid = turboquant_wave::TotalStrips(num_tokens, tokens_per_core);
    turboquant_reshape_and_cache_bf16<<<grid, nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(const_cast<std::uint16_t*>(key)),
        reinterpret_cast<bfloat16_t*>(const_cast<std::uint16_t*>(value)), key_cache,
        value_cache, scale_cache, const_cast<std::int32_t*>(slot_mapping), const_cast<float*>(pi_signs),
        const_cast<std::int32_t*>(tables), num_tokens, num_kv_heads, head_size, block_size, num_blocks,
        tokens_per_core, inv_sqrt_len);
}
#endif