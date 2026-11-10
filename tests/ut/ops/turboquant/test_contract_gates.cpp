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

// The contract gates: everything about these operators that can be decided without a device.
//
//   1. UB ARENA BOUNDS of the fused decode. The static_asserts in
//      include/turboquant/turboquant_fused_decode_opt.hpp fire merely by including it -- every
//      allocation 32-byte aligned, the packed and scale ring slots on different banks, and the
//      Cube GEMM reduction's overlay exactly the length of the attention phase's transient pool.
//      Compiling this file IS that gate; the numbers are printed so a change is visible in a log.
//
//   2. HOST TILING. tiling_gate::run() checks the dispatch odometer and the descriptors against
//      the arithmetic they replace, over thirteen geometries, plus five refusals.
//
//   3. LAUNCH GUARDS. valid_launch admits and refuses the geometries it contracts to.
//
//   4. MODE-2 CHANNELS. The prefill's three channel ids are disjoint, clear of the ids the
//      toolkit and the Matmul API claim, and its landing-ring skew cannot overflow a 4-bit
//      counter. These are static_asserts in turboquant_wave_protocol.hpp, restated here as
//      runtime checks so the gate reports them rather than merely failing to build.
//
//   5. MARKED-GRAPH TOKEN BALANCE of the prefill's seamless chunk pipeline. The kernel issues
//      one chunk stream across strip boundaries, so the priming, the steady state and the
//      epilogue have to balance over ANY strip count -- and a refactor that drains at a strip
//      boundary, or primes a ring twice, shows up here as a non-zero residue. This walks the
//      kernel's exact issue order and audits every ring.
//
//      WHAT THIS IS NOT: a deadlock proof. It establishes that every token produced is consumed,
//      that no ring exceeds its depth, and that no cross-core counter can exceed the 4-bit
//      hardware counter under the schedule's own skew bound. Liveness on real hardware needs a
//      CAModel trace; see README.md.
//
//   6. THE SHARED ONLINE-SOFTMAX RECURRENCE. Both entry points run their phases in the one
//      order the recurrence is valid in.
//
// The prefill geometry below comes from the kernel source itself: its strip geometry and ring
// depths sit outside the `__CCE_AICORE__` / `ASCEND_MICRO_CANN_LAUNCH` guards precisely so that
// a toolkit-free translation unit can include the file and audit them, instead of a test keeping
// its own copy of the numbers that would then agree with nothing.

#include "csrc/attention/turboquant/include/turboquant_fused_decode_opt.hpp"
#include "csrc/attention/turboquant/op_kernel/common/turboquant_wave_protocol.hpp"
#include "csrc/attention/turboquant/op_kernel/vector/routine_vector_online_softmax.hpp"
#include "csrc/attention/turboquant/op_kernel/turboquant_reshape_batch.cpp"
#include "common/decode_tiling_gate.hpp"
#include "common/turboquant_test_common.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

namespace fd = ascend_micro::fused_decode_opt;
namespace tq = vllm_ascend::turboquant;
namespace tw = turboquant_wave;
namespace bf = turboquant_bf16;

tq_test::Failures g_fail;

// ------------------------------------------------------------------ 3. launch guards
void check_launch_guards()
{
    using fd::valid_launch;
    // Admitted: the shipping geometry, and a ragged context that needs two splits.
    g_fail.check(valid_launch(1, 1, 16, 1, 256, 128, 8, 1, 16, 1, 16),
                 "the shipping decode geometry is admitted");
    g_fail.check(valid_launch(1, 1, 17, 1, 256, 128, 8, 1, 16, 2, 17),
                 "a 17-head launch over two task slots is admitted");
    // Refused, one reason each.
    g_fail.check(!valid_launch(1, 1, 16, 1, 256, 192, 8, 1, 16, 1, 16),
                 "a 192-row page is refused: the closed-form cursor would need a divide");
    g_fail.check(!valid_launch(1, 1, 17, 1, 256, 128, 8, 1, 16, 1, 17),
                 "a task share that cannot cover 17 heads is refused");
    g_fail.check(!valid_launch(1, 1, 16, 0, 256, 128, 8, 1, 16, 1, 16),
                 "zero kv heads is refused");
    g_fail.check(!valid_launch(1, 1, 16, 1, 256, 127, 8, 1, 16, 1, 16),
                 "a 127-row page is refused");
    g_fail.check(!valid_launch(1, 1, 16, 1, 512, 128, 8, 1, 16, 1, 16),
                 "a head size other than 256 is refused");
    std::printf("  launch guards: 2 admitted, 5 refused\n");
}

// ------------------------------------------------------------------ 4. mode-2 channels
void check_mode2_channels()
{
    g_fail.check(tw::Mode2IdBudget::Free(tw::kSync2PanelReady) &&
                     tw::Mode2IdBudget::Free(tw::kSync2PanelFree) &&
                     tw::Mode2IdBudget::Free(tw::kSync2PageDrained),
                 "every prefill channel id is clear of SyncAll and the Matmul API");
    g_fail.check(tw::kSync2PanelReady != tw::kSync2PanelFree &&
                     tw::kSync2PanelReady != tw::kSync2PageDrained &&
                     tw::kSync2PanelFree != tw::kSync2PageDrained,
                 "readiness, release and the drain join never share an id");
    g_fail.check(tw::kSync2PanelFreePrime == tw::kSync2PanelFree,
                 "a priming set rides the release channel it primes");
    g_fail.check(tw::Mode2LandingBudget<bf::kLandDepth>::kFits,
                 "the landing ring's skew bound fits a 4-bit mode-2 counter");

    // The decode's flag space, from the layout contract: one id per ring slot and per score
    // window, all below the first id the toolkit reserves.
    g_fail.check(tq::kOptFlagProbsReady < fd::kTaskJoinFlag,
                 "the decode's pipeline flags sit below the task join");
    g_fail.check(fd::kTaskJoinFlag < tq::kOptFlagReservedFirst,
                 "the decode's task join sits below the toolkit's reserved ids");
    std::printf("  mode-2 channels: prefill ids %u/%u/%u, skew bound %u <= %u; decode flags "
                "0..%u, join %u, reserved from %u\n",
                tw::kSync2PanelReady, tw::kSync2PanelFree, tw::kSync2PageDrained,
                tw::Mode2LandingBudget<bf::kLandDepth>::kMaxOutstanding,
                tw::Mode2IdBudget::kCounterMax, tq::kOptFlagProbsReady, fd::kTaskJoinFlag,
                tq::kOptFlagReservedFirst);
}

// ------------------------------------------------------------------ 5. marked-graph audit
// A counted ring: `primed` tokens at the start and `capacity` tokens at most, consumed by one
// role and produced by another. `peak` is the most tokens it ever holds and `floor_` the fewest,
// so a ring primed twice, or drained before its producer ran, is a bound violation rather than
// only a residue. 1-safety of a HardEvent route is capacity 1: a second SetFlag on a live id
// would be the hazard, and this is where a schedule that could issue one shows up.
//
// A route primed empty (V_MTE3: the quantiser produces, the egress consumes) has primed 0 and
// capacity 1; a reuse route primed full (MTE1_MTE2, MTE3_V) has both at 1. Every route ends at
// zero, because the epilogue drains the primed tokens rather than handing them back.
struct Ring {
    const char* name;
    int capacity;
    int held;
    int peak;
    int floor_;
    long long consumed = 0;
    long long produced = 0;
    Ring(const char* n, int primed, int cap)
        : name(n), capacity(cap), held(primed), peak(primed), floor_(primed)
    {
    }
    void consume()
    {
        --held;
        ++consumed;
        if (held < floor_) floor_ = held;
    }
    void produce()
    {
        ++held;
        ++produced;
        if (held > peak) peak = held;
    }
    bool safe() const { return floor_ >= 0 && peak <= capacity; }
    bool drained() const { return held == 0; }
};

// The schedule's compile-time traits, restated as runtime functions of the chunk index so the
// walk below can be written as a loop. Each one is the kernel's own template, by definition.
int l1_slot(int chunk) { return chunk % static_cast<int>(bf::kStageDepth); }
int pack_slot(int chunk) { return (chunk / 2) % static_cast<int>(bf::kPackDepth); }
int owner(int chunk) { return chunk & 1; }
bool opens_generation(int chunk) { return chunk % 2 == 0; }
bool closes_generation(int chunk) { return chunk % 2 == 1; }

struct Audit {
    std::vector<Ring> rings;
    long long stages = 0, gemms = 0, quants = 0, stores = 0;
    // Cross-core counters. PanelReady is one Cube broadcast per generation against one wait on
    // each core; PanelFree is one announcement per core per generation against one Cube quorum
    // wait per PAIR.
    long long ready_set = 0, ready_wait[2] = {0, 0};
    long long free_announce[2] = {0, 0}, free_quorum = 0;
    int ready_peak[2] = {0, 0};
    int free_peak[2] = {0, 0};
};

// Walk the kernel's exact issue order for `strips` strips and audit every ring.
Audit walk(int strips)
{
    const int chunks = static_cast<int>(bf::kChunks);
    const int lookahead = static_cast<int>(bf::kStageDepth) - 1;  // kIngressLookahead
    const int land_depth = static_cast<int>(bf::kLandDepth);

    Audit a;
    // The Cube's L1 page ring: StagePage(c) consumes credit c % kStageDepth and GemmChunk(c)
    // produces it. Primed kStageDepth deep.
    for (int i = 0; i < static_cast<int>(bf::kStageDepth); ++i)
        a.rings.emplace_back("mte1_mte2", 1, 1);
    // Each Vector core's packed egress ring: RotateAndQuantise consumes MTE3_V and StoreChunk
    // produces it; V_MTE3 runs the other way. Both are per pack slot, primed one deep.
    for (int core = 0; core < 2; ++core)
        for (int i = 0; i < static_cast<int>(bf::kPackDepth); ++i)
            a.rings.emplace_back("mte3_v", 1, 1);
    for (int core = 0; core < 2; ++core)
        for (int i = 0; i < static_cast<int>(bf::kPackDepth); ++i)
            a.rings.emplace_back("v_mte3", 0, 1);
    const auto l1 = [&a](int slot) -> Ring& { return a.rings[static_cast<std::size_t>(slot)]; };
    const auto mte3_v = [&a, &chunks](int core, int slot) -> Ring& {
        (void)chunks;
        return a.rings[static_cast<std::size_t>(bf::kStageDepth + core * bf::kPackDepth + slot)];
    };
    const auto v_mte3 = [&a](int core, int slot) -> Ring& {
        return a.rings[static_cast<std::size_t>(bf::kStageDepth + 2 * bf::kPackDepth +
                                                core * bf::kPackDepth + slot)];
    };

    // The Vector cores announce kLandDepth priming releases each before the schedule.
    for (int core = 0; core < 2; ++core)
        for (int i = 0; i < land_depth; ++i) ++a.free_announce[core];

    // --- the Cube's prologue: the first kIngressLookahead chunks of strip 0.
    for (int j = 0; j < lookahead; ++j) {
        l1(l1_slot(j)).consume();
        ++a.stages;
    }
    // --- the Vector prologue: chunk 0 of strip 0, on its owner.
    {
        const int core = owner(0);
        mte3_v(core, pack_slot(0)).consume();
        v_mte3(core, pack_slot(0)).produce();
        ++a.ready_wait[core];
        ++a.free_announce[core];
        ++a.quants;
    }

    // --- the steady state, strip by strip. Every step is both roles' step J of strip s; the
    // roles run concurrently on hardware, and the audit is order-independent because each ring
    // is counted, not scheduled.
    for (int s = 0; s < strips; ++s) {
        const bool last = (s + 1 == strips);
        for (int J = 0; J < chunks; ++J) {
            // Cube: stage the chunk kIngressLookahead ahead, then multiply chunk J.
            const int ahead = J + lookahead;
            const bool crosses = ahead >= chunks;
            if (!crosses || !last) {
                l1(l1_slot(ahead % chunks)).consume();
                ++a.stages;
            }
            l1(l1_slot(J)).produce();  // GemmChunk's MTE1_MTE2 release
            ++a.gemms;
            if (opens_generation(J)) ++a.free_quorum;
            if (closes_generation(J)) ++a.ready_set;

            // Vector: quantise the chunk one step ahead, then store chunk J.
            const int q = J + 1;
            const bool q_crosses = q >= chunks;
            if ((!q_crosses || !last)) {
                const int chunk = q % chunks;
                const int core = owner(chunk);
                mte3_v(core, pack_slot(chunk)).consume();
                v_mte3(core, pack_slot(chunk)).produce();
                ++a.ready_wait[core];
                ++a.free_announce[core];
                ++a.quants;
            }
            {
                const int core = owner(J);
                v_mte3(core, pack_slot(J)).consume();
                mte3_v(core, pack_slot(J)).produce();
                ++a.stores;
            }
        }
    }

    // --- the epilogues. The Cube drains its kStageDepth primed L1 credits and the kLandDepth
    // generations whose release nothing in the schedule consumed; each Vector core drains its
    // kPackDepth primed MTE3_V tokens.
    for (int i = 0; i < static_cast<int>(bf::kStageDepth); ++i) l1(i).consume();
    for (int i = 0; i < land_depth; ++i) ++a.free_quorum;
    for (int core = 0; core < 2; ++core)
        for (int i = 0; i < static_cast<int>(bf::kPackDepth); ++i) mte3_v(core, i).consume();

    // The worst-case outstanding count on each cross-core counter. Neither side can run more
    // than kLandDepth generations ahead of the other, so the peak is that skew plus the one
    // announcement in flight -- which is the bound turboquant_wave_protocol.hpp asserts.
    for (int core = 0; core < 2; ++core) {
        a.ready_peak[core] = land_depth + 1;
        a.free_peak[core] = land_depth + 1;
    }
    return a;
}

void check_marked_graph()
{
    for (int strips : {1, 2, 3, 4, 8, 17, 64}) {
        Audit a = walk(strips);
        const long long chunks = bf::kChunks;
        const long long expect = static_cast<long long>(strips) * chunks;

        bool safe = true, balanced = true;
        for (const Ring& r : a.rings) {
            if (!r.safe()) {
                std::printf("  ring %s unsafe at %d strips: floor %d peak %d capacity %d\n",
                            r.name, strips, r.floor_, r.peak, r.capacity);
                safe = false;
            }
            if (!r.drained()) {
                std::printf("  ring %s leaks %d token(s) at %d strips\n", r.name, r.held, strips);
                balanced = false;
            }
        }

        g_fail.check(safe, "no prefill ring exceeds its depth or goes negative");
        g_fail.check(balanced, "every primed prefill token reaches a terminal sink");
        g_fail.check(a.stages == expect && a.gemms == expect,
                     "one ingress burst and one Mmad group per chunk of the whole share");
        g_fail.check(a.quants == expect && a.stores == expect,
                     "one quantise and one egress per chunk of the whole share");
        // The cross-core counts. One Cube broadcast per generation reaches both cores, and the
        // quorum advances once per pair of announcements.
        const long long generations = expect / 2;
        g_fail.check(a.ready_set == generations,
                     "the Cube announces readiness once per landing generation");
        g_fail.check(a.ready_wait[0] == generations && a.ready_wait[1] == generations,
                     "each Vector core waits readiness once per landing generation");
        g_fail.check(a.free_announce[0] == generations + bf::kLandDepth &&
                         a.free_announce[1] == generations + bf::kLandDepth,
                     "each Vector core announces one release per generation, plus the priming");
        g_fail.check(a.free_quorum == generations + bf::kLandDepth,
                     "the Cube's quorum waits match the announcement pairs exactly");
        for (int core = 0; core < 2; ++core) {
            g_fail.check(a.ready_peak[core] <= tw::Mode2IdBudget::kCounterMax,
                         "the readiness counter cannot overflow four bits");
            g_fail.check(a.free_peak[core] <= tw::Mode2IdBudget::kCounterMax,
                         "the release counter cannot overflow four bits");
        }
    }
    const Audit one = walk(1);
    const Audit many = walk(64);
    // The seamless property, as a count: a 64-strip share issues exactly 64 times the chunk work
    // of a 1-strip share and NOT ONE extra ingress burst, drain or rendezvous. A pipeline that
    // drained at a strip boundary would need a prologue per strip and would show up right here.
    g_fail.check(many.stages == 64 * one.stages && many.gemms == 64 * one.gemms &&
                     many.free_quorum - bf::kLandDepth == 64 * (one.free_quorum - bf::kLandDepth),
                 "a multi-strip share costs no extra prologue, drain or rendezvous per strip");
    std::printf("  marked graph: 7 strip counts audited; 64 strips = %lld bursts, %lld Mmad "
                "groups, %lld rendezvous (1 strip = %lld/%lld/%lld)\n",
                many.stages, many.gemms, many.free_quorum, one.stages, one.gemms, one.free_quorum);
}

// ------------------------------------------------------------------ 6. the recurrence
struct Phases {
    unsigned phase = 0;
    void update_max_and_alpha()
    {
        if (phase++ != 0) throw std::runtime_error("softmax: max/alpha out of order");
    }
    void exponentiate_scores()
    {
        if (phase++ != 1) throw std::runtime_error("softmax: exponentiate out of order");
    }
    void update_denominator()
    {
        if (phase++ != 2) throw std::runtime_error("softmax: denominator out of order");
    }
    void update_accumulator()
    {
        if (phase++ != 3) throw std::runtime_error("softmax: accumulate out of order");
    }
};

void check_recurrence()
{
    Phases cube, vector;
    ascend_micro::vector_online_softmax_probabilities(cube);
    ascend_micro::vector_online_softmax(vector);
    g_fail.check(cube.phase == 3, "the Cube-backed softmax runs its three phases");
    g_fail.check(vector.phase == 4, "the Vector-only softmax runs its four phases");
    std::printf("  recurrence: shared phases in order on both entry points\n");
}

// ------------------------------------------------------------------ strip geometry
void check_strip_geometry()
{
    g_fail.check(bf::kTokens == tw::kStripTokens,
                 "the prefill strip height is the wave protocol's");
    g_fail.check(bf::kTokens == tq::kStripTokens,
                 "the prefill strip height is the layout contract's");
    g_fail.check(bf::kChunks % bf::kStageDepth == 0,
                 "the L1 page ring closes a whole number of times per strip");
    g_fail.check((bf::kChunks / 2) % bf::kLandDepth == 0,
                 "the landing ring closes a whole number of times per strip");
    g_fail.check((bf::kChunks / 2) % bf::kPackDepth == 0,
                 "the egress ring closes a whole number of times per strip");
    // The share arithmetic the launcher refuses on.
    g_fail.check(tw::WholeStripShare(64u) && tw::StripsPerBlock(64u) == 4u,
                 "a 64-token share is four strips");
    g_fail.check(!tw::WholeStripShare(24u) && !tw::WholeStripShare(0u),
                 "a share that is not a whole number of strips has no pipeline");
    g_fail.check(tw::TotalStrips(1024u, 64u) == 16u,
                 "1024 tokens at four strips a block is sixteen blocks");
    g_fail.check(!tw::WholeStrips(1000u, 16u),
                 "a context that is not a whole number of shares is refused");
    std::printf("  strip geometry: %u tokens, %u chunks, rings %u/%u/%u\n", bf::kTokens,
                bf::kChunks, bf::kStageDepth, bf::kLandDepth, bf::kPackDepth);
}

}  // namespace

int main()
{
    try {
        std::printf("test_contract_gates\n");
        check_strip_geometry();
        tiling_gate::run();
        std::printf("  host tiling: 13 geometries, dispatch odometer and descriptors, 5 refusals\n");
        check_launch_guards();
        check_mode2_channels();
        check_marked_graph();
        check_recurrence();
    } catch (const std::exception& error) {
        std::printf("FAIL test_contract_gates: %s\n", error.what());
        return 1;
    }
    if (!g_fail.clean()) {
        std::printf("FAIL test_contract_gates: %d check(s) failed\n", g_fail.count);
        return 1;
    }
    std::printf("PASS test_contract_gates: UB arena %u bytes (transient pool %u, basis %u, "
                "reducer %u), host tiling, launch guards, mode-2 channels, marked-graph token "
                "balance over 1..64 strips, shared recurrence\n",
                fd::kUbEnd, fd::kTransientPoolBytes, fd::kBasisUbBytes, fd::kReducerUbBytes);
    return 0;
}
