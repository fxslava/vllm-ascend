#pragma once
// Compile-time sparse boundary sampling of the grid, shared by the kernel
// under test and the harness that checks it.
//
// Why
// ---
// The device shapes are 32 strips (512 tokens) and 64 (1024). On the
// cycle-accurate model a full 64-strip run is ~550 s of wall clock, and the
// strips are independent, so almost all of that time re-proves the same thing.
// This header names a subset of grid blocks that actually execute; every other
// block returns from the kernel entry before it touches anything, and the
// harness then requires its pages to still hold their canary.
//
// What the subset is, per wave of kBlocksPerWave blocks:
//
//   first kSampleEdge      the cold start of the wave
//   kSampleCenter straddling the midpoint   the steady-state interior step
//   last kSampleEdge       the wave's tail
//
// For kBlocksPerWave = 32, kSampleEdge = 4, kSampleCenter = 4 that is indices
// 0-3, 14-17 and 28-31 of each wave: blocks 0-3, 14-17, 28-31 in wave 0 and
// 32-35, 46-49, 60-63 in wave 1. 24 of 64 blocks, so ~2.7x off the model time,
// and the same index positions in every wave.
//
// What it proves, and what it does not
// ------------------------------------
//   * **Addressing at the extremes and in the interior.** The sampled set
//     includes the lowest and highest block indices of the grid, which is
//     where the multi-block bug this harness was built for did its damage: a
//     Vector core whose index exceeded the core count computed a page past the
//     end of the cache. It also includes an interior step in each wave, which
//     is the inductive case -- block t's addressing as a function of t alone,
//     uniform in t.
//   * **Written-next-to-canary adjacency.** A center block is surrounded by
//     *skipped* blocks, so the pages on either side of the page it writes
//     belong to strips that never ran. Any interconnect or L2 write-combining
//     effect that spilled past a page boundary lands on a canary and is named,
//     not averaged away. The boundary blocks cannot show this on their own:
//     they are contiguous, so only the unowned even pages separate them.
//   * **Second-wave execution.** Sampled blocks exist above kBlocksPerWave, so
//     with a grid of 64 the kernel runs on a physical core that has already
//     executed it once in this launch, re-setting every mode-4 flag id a
//     previous block on that core used.
//   * **NOT** which physical core runs which block. Block placement is the
//     runtime's, not ours: it is only under a static (modulo) placement that a
//     sampled wave-1 block is guaranteed to land on a core that ran a sampled
//     wave-0 block. The same index positions are sampled in every wave so that
//     this holds when placement is static, and nothing here observes it, so a
//     second-wave pass is evidence of inter-wave cleanliness rather than proof
//     of it. The full-count soak (sampling compiled out) remains the stronger
//     gate.
//   * **NOT** anything about resource high-water marks at full count, which a
//     collapsed grid understates by construction.
//
// The predicate is consistent across the roles of a MIX launch by
// construction: it is a function of the grid block index -- the one both the
// Cube and its Vector cores agree on -- and of numTokens / tokensPerCore, two
// host-supplied scalars. That matters because an inconsistent predicate would
// let the Cube skip a block while its Vector cores ran it, which is a deadlock
// rather than a wrong answer. It deliberately does not use GetBlockNum(),
// whose value differs between AIC and AIV at a 1:2 ratio for the same reason
// GetBlockIdx() does.

#include <cstdint>

// One function body, two callers. The CCE compiler treats an unannotated
// function as host-only and rejects a runtime device call to one -- "reference
// to __host__ function 'is_sampled' in __global__ __aicore__ function" -- so
// the device build takes the `__aicore__ inline` spelling the vendored
// baseline's MixBlockIdx() uses, and the host build takes `inline constexpr`
// so the sample set can be pinned by static_assert. The *text* of each
// function is shared either way, which is the property that matters: host and
// device cannot disagree about which blocks run.
//
// (An unannotated constexpr function is fine for the device compiler as long
// as it is only ever evaluated at compile time -- the kernel's own ParityOf()
// is one. It is the runtime call that is rejected.)
#if defined(__CCE_AICORE__)
#define TQ_SAMPLING_API __aicore__ inline
#else
#define TQ_SAMPLING_API inline constexpr
#endif

namespace tq_sampling {

// One wave of the grid: the target's AI Core count. The Ascend950PR_9599
// CAModel reports 32 AI Cores, each with two Vector cores
// ("AicWrapper attach AIC 0..31, num_vec_core=2").
#ifndef TQ_BLOCKS_PER_WAVE
#define TQ_BLOCKS_PER_WAVE 32
#endif
constexpr int kBlocksPerWave = TQ_BLOCKS_PER_WAVE;

// Blocks sampled at each end of a wave, and across its midpoint.
constexpr int kSampleEdge = 4;
constexpr int kSampleCenter = 4;

// A wave with no room for a distinguishable interior runs whole: there is
// nothing to skip, and the collapsed grids used for iteration (1, 2 and 3
// strips) must behave exactly as they do with sampling compiled out.
constexpr int kMinSampledWidth = 2 * kSampleEdge + kSampleCenter;

TQ_SAMPLING_API int wave_of(int block) { return block / kBlocksPerWave; }
TQ_SAMPLING_API int index_in_wave(int block) { return block % kBlocksPerWave; }

// The last wave of a grid may be narrower than kBlocksPerWave.
TQ_SAMPLING_API int wave_width(int block, int total_blocks)
{
    const int rest = total_blocks - wave_of(block) * kBlocksPerWave;
    return rest < kBlocksPerWave ? rest : kBlocksPerWave;
}

TQ_SAMPLING_API int center_begin(int width) { return width / 2 - kSampleCenter / 2; }

TQ_SAMPLING_API bool is_sampled(int block, int total_blocks)
{
    if (block < 0 || total_blocks <= 0) return true;
    const int width = wave_width(block, total_blocks);
    if (width <= kMinSampledWidth) return true;
    const int i = index_in_wave(block);
    const int c0 = center_begin(width);
    return i < kSampleEdge || i >= width - kSampleEdge ||
           (i >= c0 && i < c0 + kSampleCenter);
}

// Host only, both of them: the device never counts the sample set, and the
// static_asserts need the constexpr spelling that only the host build has.
#if !defined(__CCE_AICORE__)

// How many of `total_blocks` actually execute.
constexpr int sampled_count(int total_blocks)
{
    int n = 0;
    for (int b = 0; b < total_blocks; ++b) n += is_sampled(b, total_blocks) ? 1 : 0;
    return n;
}

static_assert(kSampleEdge > 0 && kSampleCenter > 0, "an empty sample set runs nothing");
static_assert(kBlocksPerWave >= kMinSampledWidth,
              "a wave narrower than its own sample set can never sample sparsely");
// The sample set of a full wave, spelled out, so a change to the constants
// that silently drops the interior or an end is caught here rather than by a
// confusing canary failure.
static_assert(is_sampled(0, 64) && is_sampled(3, 64) && !is_sampled(4, 64),
              "the leading edge of wave 0 is 0..3");
static_assert(!is_sampled(13, 64) && is_sampled(14, 64) && is_sampled(17, 64) &&
                  !is_sampled(18, 64),
              "the interior step of a 32-wide wave is 14..17");
static_assert(!is_sampled(27, 64) && is_sampled(28, 64) && is_sampled(31, 64),
              "the trailing edge of wave 0 is 28..31");
static_assert(is_sampled(32, 64) && is_sampled(46, 64) && is_sampled(49, 64) &&
                  is_sampled(63, 64) && !is_sampled(50, 64),
              "wave 1 is sampled at the same index positions as wave 0");
static_assert(sampled_count(64) == 24, "24 of 64 blocks execute");
static_assert(sampled_count(32) == 12, "12 of 32 blocks execute");
static_assert(sampled_count(1) == 1 && sampled_count(2) == 2 && sampled_count(3) == 3,
              "the collapsed grids must be unaffected by sampling");

#endif  // !__CCE_AICORE__

}  // namespace tq_sampling
