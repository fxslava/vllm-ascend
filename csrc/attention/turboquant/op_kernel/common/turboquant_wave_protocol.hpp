#pragma once
// The MIX_AIC_1_2 inter-wave protocol, shared by both reshape-and-cache
// instances. Pure C++17: <cstdint> and nothing else, no CCE intrinsic and no
// AscendC header, so a host translation unit can compile this file under
// -Wall -Wextra -Werror and every static_assert below is a gate that runs
// without a CANN toolkit (REFINEMENT_OF_COMBAT_KERNELS.md section 2).
//
// Why this file exists
// --------------------
// A KERNEL_TYPE_MIX_AIC_1_2 launch dispatches its grid across physical AI Core
// blocks of one Cube plus two Vector cores. Above the physical block count the
// scheduler runs the grid in *waves*: blocks 0..31 are wave 0, blocks 32..63
// wave 1, and a physical block's three sub-cores execute wave 1's block after
// wave 0's. The hardware scheduler does not drain that block's engine queues
// between the two, and the mode-4 intra-block flag registers are local to the
// block. So wave 1 inherits whatever wave 0 left behind:
//
//   * unretired MTE2/MTE3 transfers, whose completion then races wave 1's
//     reuse of the same L1 / UB / L0 addresses;
//   * HardEvent credits never moved to a terminal sink;
//   * a block that read metadata it does not own -- slotMapping[block] for
//     block >= numTokens / tokensPerCore -- faults an AIV, after which the
//     Cube's next CrossCoreWaitFlag never returns and the launch hangs.
//
// Three rules answer that, and this header carries the parts of them that are
// arithmetic rather than instruction sequence:
//
//   1. boundary and guard isolation   -- TotalStrips() below, used for an
//      early return taken before InitSocState(), before any tensor or SOC
//      state exists, and before any GM address is formed;
//   2. deterministic intra-block channel allocation -- the id algebra below,
//      bounded by a static_assert against the library's reserved range rather
//      than by inspection;
//   3. the mandatory inter-wave epilogue drain -- the kFlagPageDrained ids
//      below, plus the x_S drain sequences in each kernel's epilogue.
//
// Why the channels recycle, and what had to be settled first
// ----------------------------------------------------------
// MICROROUTINE_DESIGN_GUIDELINES.md sections 6.2a, 6.2c and 6.5 carry the full
// record; it is worth knowing here because this file shipped the opposite
// default first.
//
// A recycled two-slot READY/FREE ring -- strict bilateral SET, WAIT, SET, WAIT
// per id, two landing slots, five ids per sub block -- was synthesised and
// exhaustively proved 1-safe and deadlock free over 41,844,109 modulo states
// (`safe_pipeline_proof.json`), under an explicit consuming one-bit latch
// semantics. Integrated into turboquant_reshape_and_cache_a8.cpp and run on
// CAModel it produced bit-exact caches, and its trace appeared to refute that
// premise: on AIV1, READY channel 16 was set at cycle 4906, waited at 4916,
// waited AGAIN at 7589, and only set for the second chunk at 8621 -- a second
// wait retiring 1032 cycles before its matching notification. Section 6.2c
// measured the same shape independently at 2179 cycles on an earlier revision.
// Both were read as "a mode-4 flag is not a consuming latch", and the single-
// shot allocation below was adopted because it is correct under either reading.
//
// physical block, 1 Cube + 2 Vector cores, no data, a cyclic bilateral
// handshake over recycled ids 0, 1, 16 and 17, and an `x_S` drain in front of
// every announcement -- `V_S` on each Vector core, `MTE1_M` then `M_S` on the
// Cube. It completed cleanly on physical silicon and on CAModel
// (Ascend950PR_9599, CANN 9.2.0-beta.2; 176,061 ticks), with no early wait
// retirement in the trace. The diagnosis that survives both runs is NOT that
// the latch fails to consume: it is section 6.2a's hazard, an announcement
// issued on the scalar unit overtaking the pipeline work it reports, because
// `CrossCoreSetFlag<4, PIPE_x>` is not ordered by its own `pipe` argument. A
// recycled id turns that reordering into a *stale release*, which is why
// recycling is where it showed and single-shot allocation hid it.
//
// So mode-4 flags do consume, and recycling is sound when every announcement
// is behind its `SetFlag`/`WaitFlag<HardEvent::x_S>` pair.
// and the probe, with negative controls, because it is invisible to every
// other gate here: set and wait counts balance without it, the id census
// passes, and the numbers can come out bit-exact while a panel is read one
// generation early. RecycledChannels is therefore the production policy, and
// SingleShotChannels stays compiled and dispatchable as the regression
// fallback -- it is the variant with the longer passing history, it is what a
// `--mode4-single-shot` run bisects against, and keeping both means a future
// difference on silicon isolates id recycling and nothing else.
//
// What that leaves for the wave boundary is the drain, which is where the
// correctness of a multi-wave grid actually lives: every id is set and waited
// an equal number of times inside one block's body with at most one
// notification outstanding, so a block leaves no pending notification for the
// next wave to mistake for its own, and the epilogue moves every asynchronous
// queue into the scalar stream before the block retires.

#include <cstdint>

// ===================================================== device callability
// Directive 1 / the device link error this header caused:
//
//   ld.lld: undefined symbol: turboquant_wave::TotalStrips
//
// `constexpr` already implies `inline` in C++17, so every function below has
// vague linkage and a weak definition in each translation unit that uses it.
// That was never the problem. The problem is the *core type*: TotalStrips is
// called with runtime scalars from inside the `__global__ __aicore__` kernel
// body, and a function with no core-type attribute is host code, so BiSheng
// emits a call to a host symbol that the device link cannot resolve.
//
// The same two functions are also called from the host launch wrapper, which
// computes the grid. Both callers live in the same file but never in the same
// compilation: the kernel body is under `#if defined(__CCE_AICORE__)` and the
// launcher under the `#elif defined(ASCEND_MICRO_CANN_LAUNCH)` arm of the same
// conditional. `__CCE_AICORE__` is defined only in the device pass, so one
// macro serves both -- the device pass gets `[aicore] inline`, the host pass
// plain `inline` -- and there is one body rather than a device copy and a host
// copy that could drift apart.
//
// Marking every function, not just the two that are called at runtime today:
// the rest are consumed in constant expressions only, where the attribute is
// inert, and a future runtime call to any of them would otherwise reintroduce
// device objects define and reference no `turboquant_wave` symbol at all.
//
// `__aicore__` is a macro the CCE compiler and kernel_operator.h provide; both
// reshape kernels now include this header after kernel_operator.h so it is
// defined here. The literal `[aicore]` fallback covers a device translation
// unit that includes this header first, which is how the broken include order
// produced a link error rather than a compile error.
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

// =========================================================== strip boundary
// The true task boundary, computed identically by all three core roles so that
// a block beyond it returns from every role and leaves no peer waiting. The
// zero guard is not decoration: tokensPerCore is a caller-supplied scalar, and
// an integer divide by zero on the scalar unit is as fatal as the out-of-range
// slotMapping read this boundary exists to prevent. Returning 0 makes every
// block take the early return, which is the safe direction.
TQ_WAVE_FN constexpr std::uint32_t TotalStrips(std::uint32_t num_tokens,
                                    std::uint32_t tokens_per_core)
{
    return tokens_per_core == 0u ? 0u : num_tokens / tokens_per_core;
}

// The guard itself, against a strip count the caller has already computed.
// The kernel bodies use this form rather than the three-argument one below so
// that a launch costs exactly one scalar divide: `tokensPerCore` is a runtime
// scalar, so TotalStrips is a real division, and the three-argument guard
// would make it two.
TQ_WAVE_FN constexpr bool BlockOutOfStrips(std::uint32_t block,
                                           std::uint32_t total_strips)
{
    return block >= total_strips;
}

TQ_WAVE_FN constexpr bool BlockOutOfRange(std::uint32_t block, std::uint32_t num_tokens,
                               std::uint32_t tokens_per_core)
{
    return BlockOutOfStrips(block, TotalStrips(num_tokens, tokens_per_core));
}

// Whether the context is a whole number of strips.
//
// A strip is one 16-token page of the paged KV cache, so a partial trailing
// strip is a partially filled page -- a shape this operator does not have
// (see the "no ragged shapes" note at the head of both kernels). TotalStrips
// truncates, which is the safe direction for the *guard* but silent data loss
// for a *launch*: 1000 tokens at 16 per strip is 62 strips, and tokens 992..999
// would simply never be written to the cache, with every canary intact and
// every other token bit-exact. So the launchers refuse that geometry rather
// than dispatching it, and this is the predicate they refuse on. Arbitrary
// context length means any number of strips, over any number of waves, which
// is what the guard and the inter-wave drain deliver; it does not mean a
// partial page.
TQ_WAVE_FN constexpr bool WholeStrips(std::uint32_t num_tokens,
                                      std::uint32_t tokens_per_core)
{
    return tokens_per_core != 0u && num_tokens % tokens_per_core == 0u;
}

static_assert(TotalStrips(1024u, 16u) == 64u, "64 strips of 16 tokens");
static_assert(TotalStrips(16u, 16u) == 1u, "one strip is one block");
static_assert(TotalStrips(528u, 16u) == 33u, "the first block of wave 1 exists");
static_assert(TotalStrips(2048u, 16u) == 128u, "four waves dispatch");
static_assert(TotalStrips(16u, 0u) == 0u, "a zero strip width guards every block");
static_assert(BlockOutOfRange(32u, 512u, 16u), "block 32 of a 32-strip grid is out");
static_assert(!BlockOutOfRange(32u, 528u, 16u), "block 32 of a 33-strip grid is in");
static_assert(BlockOutOfRange(0u, 0u, 16u), "an empty context runs no block");

// The two-argument guard must agree with the three-argument one everywhere the
// three-argument one is asserted above, or the kernels' one-divide form would
// be a different predicate from the documented one.
static_assert(BlockOutOfStrips(32u, TotalStrips(512u, 16u)) ==
                  BlockOutOfRange(32u, 512u, 16u) &&
              BlockOutOfStrips(32u, TotalStrips(528u, 16u)) ==
                  BlockOutOfRange(32u, 528u, 16u) &&
              BlockOutOfStrips(0u, TotalStrips(0u, 16u)) ==
                  BlockOutOfRange(0u, 0u, 16u),
              "BlockOutOfStrips must be BlockOutOfRange with the divide hoisted");
static_assert(BlockOutOfStrips(0u, 0u), "a zero strip count guards every block");

// Arbitrary context length is any number of strips, including counts that are
// not a multiple of the 32-block wave width; it is not a partial strip.
static_assert(WholeStrips(16u, 16u) && WholeStrips(528u, 16u) &&
              WholeStrips(2048u, 16u) && WholeStrips(1584u, 16u),
              "a whole number of strips is an arbitrary context this operator runs");
static_assert(!WholeStrips(1000u, 16u), "1000 tokens is 62 strips and 8 lost tokens");
static_assert(!WholeStrips(17u, 16u), "a single trailing token is still a partial page");
static_assert(!WholeStrips(16u, 0u), "a zero strip width is not a whole context");
static_assert(WholeStrips(0u, 16u), "an empty context is zero whole strips");

// ===================================================== mode-4 flag id space
// Ids are per sub block: 0..15 name AIV0's channel and 16..31 AIV1's, and the
// `+ kIntraNum` offset is how a participant names the *other* sub block's
// channel in both directions (guidelines sections 6.2 and 6.2d -- renumbering
// AIV1 into the low half deadlocks the kernel). The library claims
// SYNC_AIC_AIV_FLAG 13 and SYNC_AIV_ONLY_ALL 14, and ids 11 and 12 are not
// safe either, so a kernel's own channels live in 0..kMaxSafeLocalId.
constexpr std::uint16_t kIntraNum = 16u;
constexpr std::uint16_t kMaxSafeLocalId = 10u;

TQ_WAVE_FN constexpr std::uint16_t IntraId(std::uint32_t owner, std::uint32_t local)
{
    return static_cast<std::uint16_t>(owner * kIntraNum + local);
}

// Two channel policies, selected by a non-type template parameter the kernel
// entry points instantiate. They differ ONLY in how physical ids are handed
// out: the schedule, the wait placement, the drain sequence and the priming
// are identical, so a difference in behaviour between the two variants on
// silicon isolates id recycling and nothing else. That is the whole point of
// having both.
//
// `Chunks` is the schedule length and `LandDepth` the landing ring depth; both
// instances are 8 and 2.
//
// ---- policy A, SingleShotChannels: one id per cross-core event ------------
// The regression fallback. No id carries two announcements at all, so no
// reading of the flag mechanism can make a second wait pass early -- which is
// why it is what `--mode4-single-shot` bisects against.
//
//   productReady[j]        AIC -> AIVk, "landing panel holds chunk j"
//     j in [0, Chunks/2)                                      ids 0 .. 3
//   panelFree[i]           AIVk -> AIC, "landing panel is free again"
//     i in [0, Chunks/2 + LandDepth)                          ids 4 .. 9
//       i < LandDepth                 the prologue priming credits
//       i = j + LandDepth             AIVk's release after its chunk j
//       i >= Chunks/2                 the releases the Cube's epilogue drains
//   pageDrained            AIC -> AIVk, "every Cube queue is in the scalar
//                          stream; the strip is retired"       id 10
//
// Eleven ids per sub block, each set exactly once and waited exactly once in
// one block's body, and the highest is exactly kMaxSafeLocalId: the 8-chunk
// schedule saturates the legal space. That is why the budget is a
// static_assert below rather than a comment -- a ninth chunk or a second
// temporal segment does not fit, and must not be made to fit by renumbering.
//
// ---- policy B, RecycledChannels: a bounded bilateral two-slot ring --------
// The production default. Five ids per sub block regardless of schedule
// length, so a production-length loop over pages or strips fits with room to
// spare:
//
//   productReady[slot]     AIC -> AIVk        ids 0 .. LandDepth-1   (0, 1)
//   panelFree[slot]        AIVk -> AIC        ids LandDepth .. 2*LandDepth-1
//                                                                   (2, 3)
//   pageDrained            AIC -> AIVk        id 2*LandDepth         (4)
//
// Each id alternates strictly SET, WAIT, SET, WAIT, and that is enforced by
// the paired channel rather than asserted: the Cube's second set on
// productReady[s] is behind its own wait on panelFree[s], which is behind the
// AIV's set, which is behind the AIV's wait on the first productReady[s]. So
// no id ever carries two outstanding notifications. safe_pipeline_proof.json
// explores that net to 41,844,109 modulo states with no overwrite, race or
// deadlock and every marking co-reachable -- under an explicit *consuming
// one-bit latch* semantics.
//
// That premise is settled for this SoC and toolkit, in the direction the
// recycled handshake on physical silicon and on CAModel with `x_S` drains in
// front of every announcement and measured no early retirement; the two
// earlier CAModel counterexamples (1032 cycles in safe_pipeline.md, 2179 in
// guidelines section 6.2c) are now attributed to section 6.2a's scalar-unit
// reordering, which recycling exposes and single-shot allocation masks. The
// fence is the precondition, not an optimisation:
// revision that drops a drain must be expected to reproduce the stale release
// rather than to disprove this paragraph.
template <std::uint32_t Chunks, std::uint32_t LandDepth>
struct SingleShotChannels {
    static constexpr bool kRecycles = false;
    static constexpr std::uint32_t kPanelFreeBase = Chunks / 2u;
    static constexpr std::uint32_t kPanelFreeCount = Chunks / 2u + LandDepth;
    static constexpr std::uint32_t kMaxLocalId = kPanelFreeBase + kPanelFreeCount;

    static TQ_WAVE_FN constexpr std::uint16_t ProductReady(std::uint32_t owner, std::uint32_t chunk)
    {
        return IntraId(owner, chunk / 2u);
    }
    static TQ_WAVE_FN constexpr std::uint16_t PanelFree(std::uint32_t owner, std::uint32_t index)
    {
        return IntraId(owner, kPanelFreeBase + index);
    }
    // The Cube's wait before the Fixpipe of `chunk`.
    static TQ_WAVE_FN constexpr std::uint16_t PanelFreeWait(std::uint32_t owner, std::uint32_t chunk)
    {
        return PanelFree(owner, chunk / 2u);
    }
    // AIVk's release after `chunk`: the token the Cube consumes LandDepth
    // generations later, which is why the index is shifted by the ring depth.
    static TQ_WAVE_FN constexpr std::uint16_t PanelFreeSet(std::uint32_t owner, std::uint32_t chunk)
    {
        return PanelFree(owner, chunk / 2u + LandDepth);
    }
    // One prologue credit per landing slot.
    static TQ_WAVE_FN constexpr std::uint16_t PanelFreePrime(std::uint32_t owner, std::uint32_t slot)
    {
        return PanelFree(owner, slot);
    }
    // The releases from each core's last LandDepth chunks: nothing in the
    // schedule waits for them, so the Cube's epilogue drains them.
    static TQ_WAVE_FN constexpr std::uint16_t EpilogueDrain(std::uint32_t owner, std::uint32_t k)
    {
        return PanelFree(owner, Chunks / 2u + k);
    }
    static TQ_WAVE_FN constexpr std::uint16_t PageDrained(std::uint32_t owner)
    {
        return IntraId(owner, kMaxLocalId);
    }

    static_assert(kMaxLocalId <= kMaxSafeLocalId,
                  "the schedule's cross-core channels do not fit below the "
                  "library's reserved mode-4 ids; a longer schedule needs a "
                  "verified channel-recycling mechanism, not renumbering");
    static_assert(Chunks / 2u * 2u == Chunks,
                  "two chunks per landing generation: one per Vector core");
    static_assert(LandDepth >= 1u && LandDepth <= Chunks / 2u,
                  "the landing ring must wrap at least once per core");
};

template <std::uint32_t Chunks, std::uint32_t LandDepth>
struct RecycledChannels {
    static constexpr bool kRecycles = true;
    static constexpr std::uint32_t kPanelFreeBase = LandDepth;
    static constexpr std::uint32_t kPanelFreeCount = LandDepth;
    static constexpr std::uint32_t kMaxLocalId = 2u * LandDepth;

    static TQ_WAVE_FN constexpr std::uint16_t ProductReady(std::uint32_t owner, std::uint32_t chunk)
    {
        return IntraId(owner, (chunk / 2u) % LandDepth);
    }
    static TQ_WAVE_FN constexpr std::uint16_t PanelFree(std::uint32_t owner, std::uint32_t index)
    {
        return IntraId(owner, kPanelFreeBase + index % LandDepth);
    }
    static TQ_WAVE_FN constexpr std::uint16_t PanelFreeWait(std::uint32_t owner, std::uint32_t chunk)
    {
        return PanelFree(owner, chunk / 2u);
    }
    // The release and the wait are the SAME id here: that is the recycling.
    static TQ_WAVE_FN constexpr std::uint16_t PanelFreeSet(std::uint32_t owner, std::uint32_t chunk)
    {
        return PanelFree(owner, chunk / 2u);
    }
    static TQ_WAVE_FN constexpr std::uint16_t PanelFreePrime(std::uint32_t owner, std::uint32_t slot)
    {
        return PanelFree(owner, slot);
    }
    // The last generation's releases come back on the same LandDepth ids the
    // prologue primed, so the epilogue drains those.
    static TQ_WAVE_FN constexpr std::uint16_t EpilogueDrain(std::uint32_t owner, std::uint32_t k)
    {
        return PanelFree(owner, k);
    }
    static TQ_WAVE_FN constexpr std::uint16_t PageDrained(std::uint32_t owner)
    {
        return IntraId(owner, kMaxLocalId);
    }

    static_assert(kMaxLocalId <= kMaxSafeLocalId,
                  "even the recycled policy must stay below the reserved ids");
    static_assert(Chunks / 2u * 2u == Chunks,
                  "two chunks per landing generation: one per Vector core");
    static_assert(LandDepth >= 1u && LandDepth <= Chunks / 2u,
                  "the landing ring must wrap at least once per core");
};

// The non-type template parameter the kernel entry points select on.
// ReuseMode4Flags == true is the default everywhere; see kReshapeDefaultReuse.
template <bool ReuseMode4Flags, std::uint32_t Chunks, std::uint32_t LandDepth>
struct ChannelPolicySelect {
    using type = SingleShotChannels<Chunks, LandDepth>;
};
template <std::uint32_t Chunks, std::uint32_t LandDepth>
struct ChannelPolicySelect<true, Chunks, LandDepth> {
    using type = RecycledChannels<Chunks, LandDepth>;
};
template <bool ReuseMode4Flags, std::uint32_t Chunks, std::uint32_t LandDepth>
using ChannelPolicy =
    typename ChannelPolicySelect<ReuseMode4Flags, Chunks, LandDepth>::type;

// An id is legal when it is inside a sub block's half and clear of the
// library's reserved range in that half.
TQ_WAVE_FN constexpr bool LegalIntraId(std::uint16_t id)
{
    return (id % kIntraNum) <= kMaxSafeLocalId && id < 2u * kIntraNum;
}

// ------------------------------------------------- the production instances
// Both reshape-and-cache kernels run eight chunks over a depth-2 landing ring.
constexpr std::uint32_t kReshapeChunks = 8u;
constexpr std::uint32_t kReshapeLandDepth = 2u;
// The canonical policy, shared by the kernels' default entry points, the
// harness defaults and the trace analyser so none of them can drift from the
// others. Flipping this constant flips the default everywhere; the other
// policy stays compiled either way.
constexpr bool kReshapeDefaultReuse = true;

using ReshapeSingleShot = ChannelPolicy<false, kReshapeChunks, kReshapeLandDepth>;
using ReshapeRecycled = ChannelPolicy<true, kReshapeChunks, kReshapeLandDepth>;
using ReshapeDefault =
    ChannelPolicy<kReshapeDefaultReuse, kReshapeChunks, kReshapeLandDepth>;

static_assert(!ReshapeSingleShot::kRecycles && ReshapeRecycled::kRecycles,
              "the policy selector must not collapse the two variants");
static_assert(ReshapeDefault::kRecycles == kReshapeDefaultReuse,
              "the default alias must follow kReshapeDefaultReuse");
static_assert(ReshapeSingleShot::kMaxLocalId == kMaxSafeLocalId,
              "the eight-chunk single-shot schedule is expected to saturate the "
              "id space; if this changes, re-derive the budget rather than "
              "relaxing it");
static_assert(ReshapeRecycled::kMaxLocalId == 2u * kReshapeLandDepth,
              "the recycled policy must cost 2 * LandDepth + 1 ids, whatever "
              "the schedule length");
static_assert(ReshapeRecycled::kMaxLocalId < ReshapeSingleShot::kMaxLocalId,
              "recycling must buy id headroom or it buys nothing");

// Every id either policy can emit, on both sub blocks, is legal.
template <class P>
TQ_WAVE_FN constexpr bool AllIdsLegal()
{
    for (std::uint32_t owner = 0; owner < 2u; ++owner) {
        if (!LegalIntraId(P::PageDrained(owner))) return false;
        for (std::uint32_t chunk = 0; chunk < kReshapeChunks; ++chunk) {
            if (!LegalIntraId(P::ProductReady(owner, chunk))) return false;
            if (!LegalIntraId(P::PanelFreeWait(owner, chunk))) return false;
            if (!LegalIntraId(P::PanelFreeSet(owner, chunk))) return false;
        }
        for (std::uint32_t k = 0; k < kReshapeLandDepth; ++k) {
            if (!LegalIntraId(P::PanelFreePrime(owner, k))) return false;
            if (!LegalIntraId(P::EpilogueDrain(owner, k))) return false;
        }
    }
    return true;
}
static_assert(AllIdsLegal<ReshapeSingleShot>(), "a single-shot id reaches the reserved range");
static_assert(AllIdsLegal<ReshapeRecycled>(), "a recycled id reaches the reserved range");

// The three roles are disjoint channels in both policies: no productReady id is
// a panelFree id and neither is the pageDrained id, per sub block. A collision
// here is the one error class that balances every set/wait census and still
// corrupts a panel, so it is asserted rather than read off the tables above.
template <class P>
TQ_WAVE_FN constexpr bool RolesDisjoint()
{
    std::uint16_t highest_ready = 0;
    std::uint16_t lowest_free = static_cast<std::uint16_t>(kIntraNum);
    std::uint16_t highest_free = 0;
    for (std::uint32_t chunk = 0; chunk < kReshapeChunks; ++chunk) {
        const std::uint16_t ready = P::ProductReady(0u, chunk);
        if (ready > highest_ready) highest_ready = ready;
        const std::uint16_t frees[2] = {P::PanelFreeWait(0u, chunk),
                                        P::PanelFreeSet(0u, chunk)};
        for (const std::uint16_t id : frees) {
            if (id < lowest_free) lowest_free = id;
            if (id > highest_free) highest_free = id;
        }
    }
    for (std::uint32_t k = 0; k < kReshapeLandDepth; ++k) {
        const std::uint16_t frees[2] = {P::PanelFreePrime(0u, k),
                                        P::EpilogueDrain(0u, k)};
        for (const std::uint16_t id : frees) {
            if (id < lowest_free) lowest_free = id;
            if (id > highest_free) highest_free = id;
        }
    }
    return highest_ready < lowest_free && highest_free < P::PageDrained(0u);
}
static_assert(RolesDisjoint<ReshapeSingleShot>(), "single-shot channel roles overlap");
static_assert(RolesDisjoint<ReshapeRecycled>(), "recycled channel roles overlap");

// The release after chunk c is the token the Cube waits for when that landing
// slot comes round again, LandDepth generations later. True in both policies,
// by different arithmetic: the single-shot policy shifts the index, the
// recycled one relies on the modulo closing the ring.
static_assert(ReshapeSingleShot::PanelFreeSet(0u, 0u) == ReshapeSingleShot::PanelFreeWait(0u, 4u) &&
                  ReshapeSingleShot::PanelFreeSet(1u, 1u) == ReshapeSingleShot::PanelFreeWait(1u, 5u),
              "a single-shot release does not line up with its consuming wait");
static_assert(ReshapeRecycled::PanelFreeSet(0u, 0u) ==
                      ReshapeRecycled::PanelFreeWait(0u, 4u) &&
                  ReshapeRecycled::PanelFreeSet(1u, 1u) ==
                      ReshapeRecycled::PanelFreeWait(1u, 5u),
              "a recycled release does not line up with its consuming wait");
// Each core's last LandDepth releases are the ones only the epilogue drains:
// for core 0 chunks 4 and 6, for core 1 chunks 5 and 7.
static_assert(ReshapeSingleShot::PanelFreeSet(0u, 4u) == ReshapeSingleShot::EpilogueDrain(0u, 0u) &&
                  ReshapeSingleShot::PanelFreeSet(0u, 6u) == ReshapeSingleShot::EpilogueDrain(0u, 1u) &&
                  ReshapeSingleShot::PanelFreeSet(1u, 5u) == ReshapeSingleShot::EpilogueDrain(1u, 0u) &&
                  ReshapeSingleShot::PanelFreeSet(1u, 7u) == ReshapeSingleShot::EpilogueDrain(1u, 1u),
              "the epilogue does not drain the final single-shot releases");
static_assert(ReshapeRecycled::PanelFreeSet(0u, 4u) ==
                      ReshapeRecycled::EpilogueDrain(0u, 0u) &&
                  ReshapeRecycled::PanelFreeSet(0u, 6u) ==
                      ReshapeRecycled::EpilogueDrain(0u, 1u),
              "the epilogue does not drain the final recycled releases");
// The two sub blocks never share a physical id, in either policy.
static_assert(ReshapeSingleShot::PageDrained(0u) + kIntraNum == ReshapeSingleShot::PageDrained(1u) &&
                  ReshapeRecycled::PageDrained(0u) + kIntraNum ==
                      ReshapeRecycled::PageDrained(1u),
              "the sub blocks must occupy disjoint id halves");

// ============================================================ arena algebra
// Rule AKA3003 (guidelines section 2): every operand base 32-byte aligned, one
// DataBlock of bank skew between live ring slots, and VECTOR_REG_WIDTH of
// addressable buffer behind the last operand, because a level-2 vector op
// reads a whole vector register per source unmasked.
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

// A ring slot stride that is a multiple of 256 puts every slot of the ring on
// the same bank phase, which is the AKA3003 case the 32-byte skew exists to
// break. Checked against the stride rather than against the payload, because
// it is the stride that decides floor(offset / 32) mod 16.
TQ_WAVE_FN constexpr bool SlotStrideSkewed(std::uint32_t slot_bytes)
{
    return slot_bytes % kBlock == 0u && slot_bytes % 256u != 0u;
}

TQ_WAVE_FN constexpr bool BaseAligned(std::uint32_t base) { return base % kBlock == 0u; }

static_assert(RoundUpBlock(0u) == 0u && RoundUpBlock(1u) == 32u &&
                  RoundUpBlock(32u) == 32u && RoundUpBlock(33u) == 64u,
              "RoundUpBlock must be the identity on whole DataBlocks");
static_assert(NextSlot(0u, 0u) == 32u, "a following slot carries the bank skew");
static_assert(BaseAligned(NextSlot(31u, 1u)), "NextSlot must land 32-byte aligned");
static_assert(SlotStrideSkewed(16384u + kBankSkewBytes),
              "the 16 KiB landing slot stride must break the bank phase");
static_assert(!SlotStrideSkewed(16384u), "a bare 16 KiB stride collapses the phase");
static_assert(!SlotStrideSkewed(48u), "a stride must be a whole number of DataBlocks");

}  // namespace turboquant_wave
