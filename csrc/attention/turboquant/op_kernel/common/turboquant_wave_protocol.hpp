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

// The wave protocol of the MIX prefill operator, single-sourced: the mode-2 cross-core channel
// map, the strip boundary every core role has to agree on, and the UB/L1/L0 arena algebra.
//
// Three sections, none of them depending on a kernel toolchain beyond `pipe_t`:
//
//   1. STRIP BOUNDARY. A launch is cut into blocks of `tokens_per_core` tokens; a grid wider
//      than the context has blocks that own no strip, no physical page and no slotMapping
//      entry, and reading slotMapping there kills the Vector core and hangs its Cube on the
//      next rendezvous. All three roles derive the boundary from the same two hardware
//      registers and the same two host scalars, so a guarded group announces nothing its peers
//      could wait for.
//
//   2. MODE-2 CHANNELS. Mode 2 lowers to ffts_cross_core_sync and is the fused "AIC and all
//      AIVs" channel: the Cube's counter advances only once BOTH Vector cores have announced,
//      and one Cube announcement releases both. Each id is a counter in 0..15 that banks
//      unconsumed announcements, so reusing one across pipeline generations is sound by
//      construction, and both participants name the SAME id -- mode 2 does not address an
//      individual Vector core, so there is no per-sub-block rebase.
//
//   3. ARENA ALGEBRA. Rule AKA3003: every operand base 32-byte aligned, one DataBlock of bank
//      skew between live ring slots, and VECTOR_REG_WIDTH of addressable buffer behind the last
//      operand, because a level-2 vector op reads a whole vector register per source unmasked.

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

// =============================================================== strip boundary

// The strip: the token group one Cube pass rotates and one Vector pass quantises. It is
// kStripTokens in turboquant_layout.h; restated here so this header stays free of the layout
// contract, and pinned to it by a static_assert in each kernel translation unit.
constexpr std::uint32_t kStripTokens = 16u;

// The true task boundary, computed identically by all three core roles. The zero guard is not
// decoration: tokens_per_core is a caller-supplied scalar, and an integer divide by zero on the
// scalar unit is as fatal as the out-of-range slotMapping read this boundary prevents. Returning
// 0 makes every block take the early return, which is the safe direction.
TQ_WAVE_FN constexpr std::uint32_t TotalStrips(std::uint32_t num_tokens,
                                              std::uint32_t tokens_per_core)
{
    return tokens_per_core == 0u ? 0u : num_tokens / tokens_per_core;
}

// The guard itself, against a strip count the caller has already computed. The kernel bodies use
// this form rather than the three-argument one so that a launch costs exactly one scalar divide.
TQ_WAVE_FN constexpr bool BlockOutOfStrips(std::uint32_t block, std::uint32_t total_strips)
{
    return block >= total_strips;
}

TQ_WAVE_FN constexpr bool BlockOutOfRange(std::uint32_t block, std::uint32_t num_tokens,
                                          std::uint32_t tokens_per_core)
{
    return BlockOutOfStrips(block, TotalStrips(num_tokens, tokens_per_core));
}

// Whether the context is a whole number of blocks' shares. A partial trailing share is a
// partially filled page, a shape this operator does not have. TotalStrips truncates, which is
// the safe direction for the guard but silent data loss for a launch -- 1000 tokens at 16 a
// strip is 62 strips, and tokens 992..999 would never be written, with every canary intact and
// every other token bit-exact. The launcher refuses that geometry on this predicate instead.
TQ_WAVE_FN constexpr bool WholeStrips(std::uint32_t num_tokens, std::uint32_t tokens_per_core)
{
    return tokens_per_core != 0u && num_tokens % tokens_per_core == 0u;
}

// Strips in one block's share. The share is pipelined as one uninterrupted chunk stream, so it
// has to be a whole number of strips.
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

// The two-argument guard must agree with the three-argument one, or the kernels' one-divide form
// would be a different predicate from the documented one.
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

// ============================================================ mode-2 channels

// The id space. Each id is a counter in 0..15; overflow is a reported error, not a wrap.
struct Mode2IdBudget {
    static constexpr std::uint16_t kIds = 16u;
    // SyncAll's hardware path claims 11..14.
    static constexpr std::uint16_t kSyncAllFirst = 11u;
    static constexpr std::uint16_t kSyncAllLast = 14u;
    // The Matmul high-level API claims [0, 2N - 1] for N <= 4 objects. The prefill operator
    // defines none, but the budget is asserted against the worst case.
    static constexpr std::uint16_t kMatmulLast = 7u;
    static constexpr std::uint16_t kCounterMax = 15u;

    static constexpr bool Free(std::uint16_t id)
    {
        return id < kIds && id > kMatmulLast && (id < kSyncAllFirst || id > kSyncAllLast);
    }
};

// The prefill operator's three channels. The rendezvous is per landing GENERATION: generation g
// owns one chunk per Vector core and landing slot g % LandDepth in each core's Unified Buffer.
//
//   kSync2PanelReady   AIC -> both AIVs, one broadcast per generation, issued after the Fixpipe
//                      of the generation's second chunk so that it reports both panels.
//   kSync2PanelFree    both AIVs -> AIC, one quorum wait per generation. Each core's first
//                      LandDepth announcements are priming sets.
//   kSync2PageDrained  AIC -> both AIVs, the inter-wave epilogue join.
//
// Readiness and release never share an id.
constexpr std::uint16_t kSync2PanelReady = 8u;
constexpr std::uint16_t kSync2PanelFree = 9u;
// The same channel as kSync2PanelFree, named so that a fence audit can tell a priming set --
// "the landing ring starts empty" -- from an announcement of earlier pipe work, which is the
// only kind needing a drain.
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

// Worst-case outstanding announcements on either landing channel. Core k's g-th readiness
// announcement is behind the AIC's (g - LandDepth)-th release and the AIC's g-th release behind
// its g-th readiness wait, so neither side runs more than LandDepth generations ahead of the
// other, plus the one announcement in flight. 1-safety of the landing ring is exactly this
// bound: a counter that cannot exceed kCounterMax never wraps a stale token into a live slot.
template <std::uint32_t LandDepth>
struct Mode2LandingBudget {
    static constexpr std::uint16_t kMaxOutstanding = static_cast<std::uint16_t>(LandDepth + 1u);
    static constexpr bool kFits = kMaxOutstanding <= Mode2IdBudget::kCounterMax;
};
static_assert(Mode2LandingBudget<2u>::kFits,
              "a mode-2 flag counter overflow is a reported error, not a wrap");

#if defined(__CCE_AICORE__)
// Which pipe each participant may name. The library gates both calls on IsSplitVectorPipe /
// IsSplitCubePipe, and on __NPU_ARCH__ == 3510 the two sets are nearly disjoint: a call naming
// the other side's pipe compiles to NOTHING, silently and with no diagnostic. PIPE_S is excluded
// because mode 2 does not support it.
constexpr bool Mode2AivPipe(pipe_t pipe)
{
    return pipe == PIPE_V || pipe == PIPE_MTE2 || pipe == PIPE_MTE3;
}

constexpr bool Mode2AicPipe(pipe_t pipe)
{
    return pipe == PIPE_MTE1 || pipe == PIPE_MTE2 || pipe == PIPE_FIX || pipe == PIPE_M;
}

// The AIC waits and announces on FIX, the pipe that WRITES the landing panel; both Vector cores
// on V, the pipe that READS it. One AIC wait covers the generation's whole Fixpipe pair: the
// Fixpipe queue is in order, so gating its issuance gates both transfers, and the L0C dependency
// behind each one is carried explicitly by M_FIX / FIX_M.
constexpr pipe_t kSync2AicWait = PIPE_FIX;
constexpr pipe_t kSync2AicAnnounce = PIPE_FIX;
constexpr pipe_t kSync2AivWait = PIPE_V;
constexpr pipe_t kSync2AivAnnounce = PIPE_V;

static_assert(Mode2AicPipe(kSync2AicWait) && Mode2AicPipe(kSync2AicAnnounce),
              "an AIC mode-2 call naming a Vector-only pipe compiles to nothing");
static_assert(Mode2AivPipe(kSync2AivWait) && Mode2AivPipe(kSync2AivAnnounce),
              "an AIV mode-2 call naming a Cube-only pipe compiles to nothing");
#endif

// ============================================================= arena algebra

constexpr std::uint32_t kBlock = 32u;
constexpr std::uint32_t kBankSkewBytes = 32u;
constexpr std::uint32_t kReadPadBytes = 256u;  // VECTOR_REG_WIDTH on dav-3510
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

// A ring slot stride that is a multiple of 256 puts every slot of the ring on the same bank
// phase, which is the AKA3003 case the 32-byte skew exists to break. Checked against the stride
// rather than against the payload, because it is the stride that decides floor(offset/32) mod 16.
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

}  // namespace turboquant_wave
