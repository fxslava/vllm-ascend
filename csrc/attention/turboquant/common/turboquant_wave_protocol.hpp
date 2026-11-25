#pragma once
#include <cstdint>

#ifndef TQ_WAVE_FN
#  if defined(__CCE_AICORE__)
#    if defined(__aicore__)
#      define TQ_WAVE_FN __aicore__ inline
#    else
#      define TQ_WAVE_FN [aicore] inline
#    endif
#  else
#    define TQ_WAVE_FN inline
#  endif
#endif

namespace turboquant_wave {

constexpr std::uint32_t kStripTokens = 16u;

TQ_WAVE_FN constexpr std::uint64_t ActivationOffset(std::uint32_t token,
                                                     std::uint32_t num_kv_heads,
                                                     std::uint32_t head_size)
{
    return static_cast<std::uint64_t>(token) * num_kv_heads * head_size;
}

static_assert(ActivationOffset(1u << 22u, 4u, 256u) == (1ull << 32u));

constexpr bool FitsLaunchU32(std::int64_t value)
{
    return value >= 0 && static_cast<std::uint64_t>(value) <= UINT32_MAX;
}

TQ_WAVE_FN constexpr std::uint32_t TotalStrips(std::uint32_t num_tokens,
                                              std::uint32_t tokens_per_core)
{
    return tokens_per_core == 0u ? 0u : num_tokens / tokens_per_core;
}

TQ_WAVE_FN constexpr bool BlockOutOfStrips(std::uint32_t block, std::uint32_t total_strips)
{
    return block >= total_strips;
}

TQ_WAVE_FN constexpr bool BlockOutOfRange(std::uint32_t block, std::uint32_t num_tokens,
                                          std::uint32_t tokens_per_core)
{
    return BlockOutOfStrips(block, TotalStrips(num_tokens, tokens_per_core));
}

TQ_WAVE_FN constexpr bool WholeStrips(std::uint32_t num_tokens, std::uint32_t tokens_per_core)
{
    return tokens_per_core != 0u && num_tokens % tokens_per_core == 0u;
}

TQ_WAVE_FN constexpr std::uint32_t StripsPerBlock(std::uint32_t tokens_per_core)
{
    return tokens_per_core / kStripTokens;
}

TQ_WAVE_FN constexpr bool WholeStripShare(std::uint32_t tokens_per_core)
{
    return tokens_per_core != 0u && tokens_per_core % kStripTokens == 0u;
}

static_assert(TotalStrips(1024u, 16u) == 64u, "64 strips of 16 tokens");
static_assert(TotalStrips(16u, 16u) == 1u, "one strip is one block");
static_assert(TotalStrips(528u, 16u) == 33u, "the first block of wave 1 exists");
static_assert(TotalStrips(2048u, 16u) == 128u, "four waves dispatch");
static_assert(TotalStrips(16u, 0u) == 0u, "a zero share width guards every block");
static_assert(TotalStrips(1024u, 64u) == 16u, "four strips a block is sixteen blocks");
static_assert(BlockOutOfRange(32u, 512u, 16u), "block 32 of a 32-block grid is out");
static_assert(!BlockOutOfRange(32u, 528u, 16u), "block 32 of a 33-block grid is in");
static_assert(BlockOutOfRange(0u, 0u, 16u), "an empty context runs no block");

static_assert(BlockOutOfStrips(32u, TotalStrips(512u, 16u)) == BlockOutOfRange(32u, 512u, 16u) &&
                  BlockOutOfStrips(32u, TotalStrips(528u, 16u)) ==
                      BlockOutOfRange(32u, 528u, 16u) &&
                  BlockOutOfStrips(0u, TotalStrips(0u, 16u)) == BlockOutOfRange(0u, 0u, 16u),
              "BlockOutOfStrips must be BlockOutOfRange with the divide hoisted");
static_assert(BlockOutOfStrips(0u, 0u), "a zero strip count guards every block");

static_assert(WholeStrips(16u, 16u) && WholeStrips(528u, 16u) && WholeStrips(2048u, 16u) &&
                  WholeStrips(1584u, 16u),
              "a whole number of shares is an arbitrary context this operator runs");
static_assert(!WholeStrips(1000u, 16u), "1000 tokens is 62 strips and 8 lost tokens");
static_assert(!WholeStrips(17u, 16u), "a single trailing token is still a partial page");
static_assert(!WholeStrips(16u, 0u), "a zero share width is not a whole context");
static_assert(WholeStrips(0u, 16u), "an empty context is zero whole strips");

static_assert(StripsPerBlock(16u) == 1u && StripsPerBlock(128u) == 8u,
              "a share is counted in strips");
static_assert(WholeStripShare(64u) && !WholeStripShare(24u) && !WholeStripShare(0u),
              "a share that is not a whole number of strips has no pipeline");

struct Mode2IdBudget {
    static constexpr std::uint16_t kIds = 16u;

    static constexpr std::uint16_t kSyncAllFirst = 11u;
    static constexpr std::uint16_t kSyncAllLast = 14u;

    static constexpr std::uint16_t kMatmulLast = 7u;
    static constexpr std::uint16_t kCounterMax = 15u;

    static constexpr bool Free(std::uint16_t id)
    {
        return id < kIds && id > kMatmulLast && (id < kSyncAllFirst || id > kSyncAllLast);
    }
};

constexpr std::uint16_t kSync2PanelReady = 8u;
constexpr std::uint16_t kSync2PanelFree = 9u;

constexpr std::uint16_t kSync2PanelFreePrime = kSync2PanelFree;
constexpr std::uint16_t kSync2PageDrained = 10u;

static_assert(Mode2IdBudget::Free(kSync2PanelReady),
              "the release channel must not alias SyncAll or Matmul");
static_assert(Mode2IdBudget::Free(kSync2PanelFree),
              "the readiness channel must not alias SyncAll or Matmul");
static_assert(Mode2IdBudget::Free(kSync2PageDrained),
              "the inter-wave drain channel must not alias SyncAll or Matmul");
static_assert(kSync2PanelReady != kSync2PanelFree && kSync2PanelReady != kSync2PageDrained &&
                  kSync2PanelFree != kSync2PageDrained,
              "readiness, release and the drain join must never share an id");

template <std::uint32_t LandDepth>
struct Mode2LandingBudget {
    static constexpr std::uint16_t kMaxOutstanding = static_cast<std::uint16_t>(LandDepth + 1u);
    static constexpr bool kFits = kMaxOutstanding <= Mode2IdBudget::kCounterMax;
};
static_assert(Mode2LandingBudget<2u>::kFits,
              "a mode-2 flag counter overflow is a reported error, not a wrap");

#if defined(__CCE_AICORE__)

constexpr bool Mode2AivPipe(pipe_t pipe)
{
    return pipe == PIPE_V || pipe == PIPE_MTE2 || pipe == PIPE_MTE3;
}

constexpr bool Mode2AicPipe(pipe_t pipe)
{
    return pipe == PIPE_MTE1 || pipe == PIPE_MTE2 || pipe == PIPE_FIX || pipe == PIPE_M;
}

constexpr pipe_t kSync2AicWait = PIPE_FIX;
constexpr pipe_t kSync2AicAnnounce = PIPE_FIX;
constexpr pipe_t kSync2AivWait = PIPE_V;
constexpr pipe_t kSync2AivAnnounce = PIPE_V;

static_assert(Mode2AicPipe(kSync2AicWait) && Mode2AicPipe(kSync2AicAnnounce),
              "an AIC mode-2 call naming a Vector-only pipe compiles to nothing");
static_assert(Mode2AivPipe(kSync2AivWait) && Mode2AivPipe(kSync2AivAnnounce),
              "an AIV mode-2 call naming a Cube-only pipe compiles to nothing");
#endif

constexpr std::uint32_t kBlock = 32u;
constexpr std::uint32_t kBankSkewBytes = 32u;
constexpr std::uint32_t kReadPadBytes = 256u;
constexpr std::uint32_t kUbWindow = 192u * 1024u;
constexpr std::uint32_t kL1Window = 512u * 1024u;
constexpr std::uint32_t kL0Window = 64u * 1024u;

TQ_WAVE_FN constexpr std::uint32_t RoundUpBlock(std::uint32_t bytes)
{
    return (bytes + kBlock - 1u) / kBlock * kBlock;
}

TQ_WAVE_FN constexpr std::uint32_t NextSlot(std::uint32_t prev_base, std::uint32_t prev_bytes)
{
    return RoundUpBlock(prev_base + prev_bytes) + kBankSkewBytes;
}

TQ_WAVE_FN constexpr bool SlotStrideSkewed(std::uint32_t slot_bytes)
{
    return slot_bytes % kBlock == 0u && slot_bytes % 256u != 0u;
}

TQ_WAVE_FN constexpr bool BaseAligned(std::uint32_t base) { return base % kBlock == 0u; }

static_assert(RoundUpBlock(0u) == 0u && RoundUpBlock(1u) == 32u && RoundUpBlock(32u) == 32u &&
                  RoundUpBlock(33u) == 64u,
              "RoundUpBlock must be the identity on whole DataBlocks");
static_assert(NextSlot(0u, 0u) == 32u, "a following slot carries the bank skew");
static_assert(BaseAligned(NextSlot(31u, 1u)), "NextSlot must land 32-byte aligned");
static_assert(SlotStrideSkewed(16384u + kBankSkewBytes),
              "the 16 KiB landing slot stride must break the bank phase");
static_assert(!SlotStrideSkewed(16384u), "a bare 16 KiB stride collapses the phase");
static_assert(!SlotStrideSkewed(48u), "a stride must be a whole number of DataBlocks");

}
