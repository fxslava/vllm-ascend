# TurboQuant test suites: a review

Everything that validates the TurboQuant 4-bit rotated KV cache, organised the
way the suite is now organised: by **where a test can run**. What each binary is
for, what hardware it targets, what its exit criterion is, what each case
actually proves and where its bound came from — and, as important, what none of
them cover.

Companion documents: [README.md](README.md) for how to build and run the suite,
[COVERAGE.md](COVERAGE.md) for how the C++ suite compares to the Python one.

---

## Contents

- [1. The three tiers](#1-the-three-tiers)
- [2. Architecture isolation: 310P under a 950PR build](#2-architecture-isolation-310p-under-a-950pr-build)
- [3. What is under test](#3-what-is-under-test)
- [4. The oracle, and what it is not](#4-the-oracle-and-what-it-is-not)
- [5. Tier 1 — `host/`](#5-tier-1--host)
- [6. Tier 2 — `sim/`](#6-tier-2--sim)
- [7. Tier 3 — `device/`](#7-tier-3--device)
- [8. The fp16 baseline, and the V2 → V5 migration](#8-the-fp16-baseline-and-the-v2--v5-migration)
- [9. Build and run matrix](#9-build-and-run-matrix)
- [10. Bounds, and where each number came from](#10-bounds-and-where-each-number-came-from)
- [11. What none of this covers](#11-what-none-of-this-covers)
- [12. What has actually been executed](#12-what-has-actually-been-executed)
- [13. Multi-mode TurboQuant and the Cube-native decode](#13-multi-mode-turboquant-and-the-cube-native-decode)
  - [13.24 Corrections from silicon profiling](#1324-corrections-from-silicon-profiling-2026-09-16)
  - [13.25 The fused Cube decode](#1325-the-fused-cube-decode-turboquantfuseddecode)
  - [13.26 Filled MIX blocks, the ingest ring and the head-batched softmax](#1326-filled-mix-blocks-the-ingest-ring-and-the-head-batched-softmax)
  - [13.27 CAModel cycle profile of 13.26](#1327-camodel-cycle-profile-of-1326-2026-09-17)

---

## 1. The three tiers

A test is defined by where it can run, and the three answers need different link
lines — not just different runtime gates. Each is a subdirectory with its own
`CMakeLists.txt`, and no binary belongs to two of them.

| Tier | Directory | Binaries | Target hardware | Links |
| --- | --- | --- | --- | --- |
| **1. host** | `host/` | `test_host_*` | none — any CPU | no CANN, no NPU runtime |
| **2. sim** | `sim/` | `test_sim_950pr_*` | CAModel emulator | `libruntime_camodel.so` |
| **3. device** | `device/` | `test_device_950pr_*`, `bench_device_950pr_*` | physical Ascend 950PR | real `libruntime.so` + `libascendcl.so` |

```
csrc/tests/
├── CMakeLists.txt          tier dispatch, SoC gating, shared helpers
├── common/                 shared infrastructure (runtime, operators, harness)
├── reference/              the CPU oracle
├── data/golden_layer3/     the Qwen3.5 layer-3 dump (Git LFS)
├── turboquant/             ascendc_library() for the Ascend C kernels
│
├── host/                   TIER 1  ── no CANN at all
│   └── test_host_turboquant_fidelity.cpp
│
├── sim/                    TIER 2  ── CAModel only
│   ├── test_sim_950pr_turboquant_kernels.cpp
│   └── test_sim_950pr_turboquant_decode.cpp
│
├── device/                 TIER 3  ── physical 950PR silicon
│   ├── test_device_950pr_turboquant.cpp
│   ├── test_device_950pr_matmul.cpp
│   ├── test_device_950pr_rmsnorm.cpp
│   ├── test_device_950pr_rotary_embedding.cpp
│   ├── test_device_950pr_activation_swiglu.cpp
│   ├── test_device_950pr_qwen_layer_golden.cpp
│   ├── test_device_950pr_benchmark_harness.cpp
│   ├── bench_device_950pr_turboquant.cpp
│   ├── bench_device_950pr_turboquant_ablation.cpp
│   └── prof_device_950pr_msprof_trace.cpp
│
└── device_310p/            the 310P leg — a different part, not a tier
    ├── test_*_310p.cpp
    └── bench_*_310p.cpp
```

**The exclusivity is structural, not a naming convention.** `ascendc_library()`
builds one kernel library per `RUN_MODE`, so a single tree cannot hold both a
camodel and a silicon build of it. `RUN_MODE` therefore selects which of `sim/`
and `device/` is configured at all:

| Configuration | Tiers configured |
| --- | --- |
| `-DRUN_MODE=npu` (default) | `host/` + `device/` |
| `-DRUN_MODE=sim` | `host/` + `sim/` |
| `-DVLLM_ASCEND_TESTS_HOST_ONLY=ON` | `host/` alone |

That is what makes the device tier's "no CAModel fallback" a guarantee rather
than an aspiration: in a device build there is no camodel on the link line, none
on the `RUNPATH`, and nothing to fall back to.

Every 950PR binary prints its own tier and whether a camodel is loaded, so a
result can never be misattributed after the fact:

```
[ascend-test] tier: device
[ascend-test] CAModel loaded: no (physical runtime)
```

---

## 2. Architecture isolation: 310P under a 950PR build

**When `SOC_VERSION` names a 950PR, `device_310p/` is not configured, not
compiled, and contributes no target.** The parent `CMakeLists.txt` never calls
`add_subdirectory()` on it, so a `*310*` binary cannot exist in a 950PR build
tree — `ctest -N` there has nothing to list. Configure output says so
explicitly:

```
-- vllm-ascend tests: tiers -> host=TRUE sim=FALSE device=TRUE device_310p=FALSE
-- vllm-ascend tests: 310P targets are excluded from this build entirely
```

The reverse holds too: on a 310P SoC the 950PR tiers are not configured.

Two deliberate decisions inside that isolation:

- **The 310P binaries keep their names** (`test_matmul_310p`,
  `bench_matmul_310p`, …). `.agents/skills/ascend-310p-csrc-tests/SKILL.md` and
  the shipped `dist/ascend310p3-aarch64-tests/run_tests.sh` both drive them by
  name; they already carry their part in the name; and they are unreachable from
  a 950PR build, so nothing can collide. What the isolation needed was a
  directory and a configure-time gate, not a rename.
- **`common/aclnn_ops.{hpp,cpp}` is *not* a 310P file** despite its history and
  is deliberately still shared. It is the suite's whole operator prototype
  table, and the 950PR tests read it too — `ops::kMatmul`, `ops::kRmsNorm`,
  `ops::kSwiGlu`, `ops::kScatterPaKvCache` and `ops::kApplyRotaryPosEmbV2` all
  resolve through it. What the isolation excludes is the 310P *targets*.
  The one genuinely v200-specific thing it carried, the `ASCEND_PLATFORM_310P`
  define, now lives in `device_310p/CMakeLists.txt` so a 950PR tree never
  carries a define claiming it is a 310P.

---

## 3. What is under test

The shipped codec, as `csrc/attention/turboquant/turboquant_codec_950.h` and
`turboquant_kernels.cpp` implement it:

```
Pi x = D (H (D x))       D = diag(+-1) from an explicit LCG, H = normalised Walsh-Hadamard
scale = ||Pi x||_2 / sqrt(D)
q_c   = sum_{i=1..15} [ (Pi x)_c / scale > t_i ]        16-level Lloyd-Max for N(0, 1)
byte  = int8( q_{2c} + 16 q_{2c+1} - 128 )
```

`Pi` is symmetric and an involution, so one routine both rotates and un-rotates;
the decode runs entirely in the rotated basis and applies `Pi` once more to the
accumulated output. Nothing rewrites a weight — the rotation is an activation
transform, which is why RoPE stays correct.

Storage is plain contiguous ND:

| Plane | Shape | Type |
| --- | --- | --- |
| key / value in | `[num_tokens, num_kv_heads, head_size]` | fp16 or bf16 |
| key / value cache | `[num_blocks, block_size, num_kv_heads, head_size / 2]` | int8 |
| scale cache | `[num_blocks, block_size, scale_slot]` | fp32 |
| query / output | `[num_tokens, num_heads, head_size]` | fp16 or bf16 |

with `scale_slot = round_up(2 * num_kv_heads, 8)`. That padding is why every DMA
on this path is an aligned `DataCopy` and never a `DataCopyPad`, and several
cases exist purely to keep it that way.

**Both kernels are vector-only.** No `Matmul`, no `Mmad`, no L1 copy, no `[aic]`
half anywhere in the source. That is a design point, not an omission, and it is
what `bench_device_950pr_turboquant` measures.

---

## 4. The oracle, and what it is not

`csrc/tests/reference/turbo_quant_cpu.h` is the reference every device case is
compared against. It mirrors `turboquant_codec_950.h` instruction for
instruction: the same LCG for the `Pi` diagonal, the same Lloyd-Max table, the
same `-128` store bias, the same rotated-basis decode.

**It is not an independent reimplementation and does not pretend to be.** A
comparison against it establishes that *the kernel does on the device what the
reference does on the host* — a strong statement about the port, and no
statement at all about whether the algorithm is right. The algorithm's own
accuracy is measured separately against exact fp32 attention, by the host tier
(on real activations) and by the sim tier (on the device).

Two consequences to internalise before reading any bound below:

1. **Device and host cannot be bit-exact in general.** The RMS scale is a sum of
   squares; the device reduces it in a tree and the host sums it serially, so
   the two differ in the last bits. A coordinate sitting on a Lloyd-Max decision
   boundary can therefore legitimately fall either side of it, which is worth
   exactly one bin. Every packed-cache comparison is stated in *bin indices*
   with a one-bin tolerance for that reason — never in reconstructed values,
   because the Lloyd-Max levels are unevenly spaced (one bin is worth anywhere
   between 0.257 and 0.664), so a value-space bound tight enough to catch a
   two-bin slip at the centre of the table would reject a legitimate tie-break
   at its edge.
2. **Except where the test removes the tie-break.**
   `PackedCacheIsByteIdenticalOnRotationExactInputs` constructs inputs on which
   no rounding difference is possible at all, and then demands byte equality
   with no tolerance. See §7.3.

---

## 5. Tier 1 — `host/`

| | |
| --- | --- |
| **Binary** | `test_host_turboquant_fidelity` |
| **Purpose** | Verify the codec's *algorithm*: the Lloyd-Max grid and its decision boundaries, the `Pi` involution and orthogonality, the nibble packing, the read/write extents of every codec entry point, and the codec's measured fidelity on real Qwen3.5 activations. |
| **Target hardware** | None. Any CPU. |
| **Links** | `vllm_ascend_test_host` + `gtest_main`. Neither `libascendcl.so` nor any NPU runtime — it links `gtest_main` rather than one of the suite's `main()`s precisely because those initialise the ACL runtime. |
| **Exit criterion** | **Every case passes.** There is no "skipped, no hardware" state in this tier, which is exactly why it is the one CI can gate on. The single conditional skip is a golden dump that was never fetched from Git LFS, and it names `git lfs pull` when it happens. |

**Assertion policy.** This binary is an *analytical reporter*. It prints cosine
similarity, SNR and relative L2 and asserts on none of them, so a change in
quantiser behaviour shows up as a number that moved rather than as a red build.
The only `EXPECT`s are on exact integer or involution identities that cannot
drift with the data.

### 5.1 Structural invariants — `TurboQuantCodecInvariants`

| Case | Validation target |
| --- | --- |
| `PiIsAnInvolution` | `Pi(Pi(x)) == x` to fp32 rounding at `d = 256`. If this fails, the decode's un-rotation is not the inverse of the write's rotation and every output is in the wrong basis. |
| `PiPreservesDotProducts` | `Pi` is orthogonal, so `<Pi q, Pi k> == <q, k>`. This is the identity that lets the decode compute scores in the rotated basis without ever un-rotating K. |
| `PiSignVectorMatchesThePythonReference` | The `+-1` diagonal from the C++ LCG equals the one `vllm_ascend/attention/turboquant_v1.py` generates, channel for channel, at 128 and 256. A cache written by one half of the project has to be readable by the other. |
| `NibblePackRoundTripsExactly` | Every one of the 16 codes survives both nibble positions, including the extremes that make the byte `+127` and `-128`. |

### 5.2 Bounds and memory discipline — `TurboQuantPoison`

The codec runs in UB with scratch buffers sized for a full `batchRows`, and a
call with fewer live rows must confine itself to the live prefix. Five cases
poison the destination, the padding behind it and the scratch with two disjoint
alphabets and require that nothing poisoned reaches the output and nothing
outside the licensed slice is written:
`QuantizeIgnoresPoisonedDestinationAndPadding`,
`QuantizeWritesNoByteOutsideItsSlice`,
`DequantizePartialBatchIgnoresPoisonedTail`,
`DequantizeWritesNoByteOutsideItsSlice`, `ApplyPiIgnoresSurroundingMemory`.

**Validation target:** the read and write extents of every codec entry point —
the class of bug a fidelity metric cannot see, because a stray write shows up as
someone else's corruption rather than as a worse SNR.

### 5.3 The table-driven path — `TurboQuantTableDriven`

The device does not compute the butterfly pairing or the nibble split; it
gathers them from a host-built constant-table image. Two cases run the same
gather-driven algorithm on the host over `len in {64, 128, 256}` and
`batch_rows in {4, 8}` with partial batches of 1, 3 and 4 rows, and require it to
agree with the direct reference **bit for bit** (`FloatToBits` equality), with
poisoned scratch throughout.

**Validation target:** the table layout itself. A wrong offset in the image
produces a plausible-looking but wrong vector, and this is where that is caught
rather than on the device.

### 5.4 Numerical edge cases — `TurboQuantEdgeCases`

| Case | Validation target |
| --- | --- |
| `AllZeroVectorIsNumericallyStable` | The `kEps` floor. The table has no exact zero, so an all-zero vector reconstructs to about `-1.3e-21` rather than `0.0`; what matters is that it does not divide by zero or produce a NaN. |
| `SingleDominantOutlier` | The regime the rotation exists for: one channel far outside the rest. |
| `ClampingAndSaturationExtremes` | Values past the top and bottom of the table saturate into bins 0 and 15 rather than wrapping the packed byte. |
| `EveryNibblePairRoundTrips` | All 256 `(low, high)` nibble pairs. |
| `PartialBatchesMatchPerRowQuantisation` | A batched call equals row-by-row calls. |
| `InvolutionOnCanonicalBasisVectors` | `Pi^2 e_i == e_i` and `||Pi e_i|| == 1` for every basis vector, reported as a worst case. |

### 5.5 Fidelity reports

| Case | What it measures | Known limitation |
| --- | --- | --- |
| `TurboQuantGoldenFidelity.Qwen35Layer3DecodeReport` | The codec on **real Qwen3.5 layer-3 activations** from `data/golden_layer3`: post-RoPE Q and K and the V slice of `tap_qkv`. | The dump is `pos=0, ctx_len=1`, so the softmax is over a single score and the attention context is just V. It measures the codec on real activations and **not** the online-softmax accumulation. |
| `TurboQuantSyntheticFidelity.LongContextDecodeReport` | A 300-position context spanning three paged blocks, the last partial, at `d = 128`, 8 heads over 2 kv heads. | Synthetic activations. It exists to cover exactly what the `ctx=1` golden dump cannot: accumulation across block boundaries. |

---

## 6. Tier 2 — `sim/`

| | |
| --- | --- |
| **Binaries** | `test_sim_950pr_turboquant_kernels`, `test_sim_950pr_turboquant_decode` |
| **Purpose** | Functional validation of the Ascend C kernels against the CPU reference, with no hardware. |
| **Target hardware** | The CANN camodel — a cycle-level functional model. These binaries link `libruntime_camodel.so` and carry the simulator directory first on their `RUNPATH`, so they run on a machine with no NPU **and cannot run on silicon**. That is the tier's definition, not a limitation of it. |
| **Exit criterion** | Every case passes, or skips naming an operator the camodel has no binary kernel for. A timeout is a failure of the machine, not of the kernel — re-run before believing it. |

**Nothing in this tier is timed, and nothing in it may ever be presented as a
performance number.** A cycle-level simulator's wall clock measures the
simulator. Shapes are deliberately the smallest that still exercise the feature
under test — `head_size 64`, `block_size 16`, `context 32`, 4 heads over 2 kv
heads — because a single decode pass here is two to four *minutes*. ctest gets a
5400 s timeout for that reason. Each shape is still the smallest that keeps its
feature live: 64 is the bottom of the supported range and still a four-stage
Walsh-Hadamard, 16 is exactly one `kTileRows` tile, 32 is two blocks read
through a non-identity block table, and the GQA group of two makes the
head→kv_head mapping non-trivial.

### 6.1 `test_sim_950pr_turboquant_kernels`

Two quite different things live in this binary.

**The launch contract — `TurboQuantLaunchContract`, no device needed.** The
scale-plane geometry, the codec table image and the grid arithmetic exist in
**five places** that can drift independently: the Ascend C kernel, the Torch
adapter, the Python backend, the CPU reference, and the Torch-free test shim in
`common/turboquant_launch.hpp`.

| Case | Validation target |
| --- | --- |
| `ScaleSlotMatchesTheCpuReference` | `scale_slot` agrees between the shim and the reference for `num_kv_heads` 1..16, is always a whole 32-byte burst, and always fits K and V. |
| `CodecTableWordsMatchTheLayoutContract` | The image is `7D + 2DB + 16` words for `D in {64, 128, 256}` and `B in {1, 16}`, the built image is that long, and the length is a whole burst — `Init()` moves it with one `DataCopy`. |
| `CodecTablesHaveTheDocumentedLayout` | Every section of the image against its documented meaning: the three butterfly stages' signs and XOR partners, the even/odd nibble offsets, the batched expansion offsets and selectors, and the 16 Lloyd-Max centroids. |
| `PiSignsMatchTheCpuReference` | The sign vector at 64, 128 and 256, plus the first eight values at 128 recomputed from the documented LCG rather than transcribed — so it stays a check on the generator, not on a copied literal. |
| `GridPlansMatchTheAdapterArithmetic` | Coverage and non-over-subscription of the write grid; the split cap, the per-sequence block cap, the workspace size and `combine_block_dim <= split_block_dim` for the decode; and that zero tokens produces a no-op rather than a zero-sized grid the runtime would reject. |

**The kernels — `TurboQuantKernels`, gated on `REQUIRE_ASCEND_950PR`.**

| Case | Validation target | Bound |
| --- | --- | --- |
| `ReshapeAndCacheMatchesTheCpuReference` | The write path: rotate, quantise, scatter, for both planes and the scale plane. | Bin drift `<= 1`, differing channels `<= 2%`, scale relative error `<= 1e-5`. Measured drift on the camodel is **0.000**. |
| `ReshapeAndCacheLeavesNegativeSlotsUntouched` | A `-1` slot is how the scheduler marks a padded token. The kernel must skip it entirely — not write zeros, not write it elsewhere — or a padded step corrupts whatever block `-1` aliases to. | The skipped slot is byte-for-byte still zero. |
| `PagedAttentionMatchesTheCpuReference` | The split/combine decode, reading back **the cache the device itself wrote**, so a disagreement is the decode and not the write path. | `cos > 0.9995`, `relL2 < 5e-3`. Tight because both sides keep an fp32 accumulator and round once at the store. |

### 6.2 `test_sim_950pr_turboquant_decode`

One decode step, and the only case that compares the quantised device output
against *exact fp32 attention* rather than against the CPU TurboQuant reference.

Three outputs are compared: **quantised** (the two kernels on the device),
**control** (the same decode with an unquantised fp16 KV cache through
`aclnnFusedInferAttentionScore*` — *attempted, never required*), and **exact**
(fp32 attention over the same fp16 inputs, on the host; always available, which
is why the assertions hang off it).

| Assertion | Bound | Why it sits there |
| --- | --- | --- |
| quantised vs the CPU TurboQuant reference | `cos > 0.999`, `relL2 < 5e-3` | Same algorithm both sides; anything beyond fp16 output rounding is a kernel bug. This is the regression detector. |
| quantised vs exact fp32 | `cos > 0.97`, `SNR >= 12 dB`, `relL2 < 0.30` | A **floor**, not a target. This is the scheme's error, not the kernel's: the shape measures ~16.7 dB on the camodel and ~18 dB on the host at a different seed, and the bound sits well clear because the number moves with the data. It catches gross breakage — a dropped un-rotation, a scale read from the wrong lane, a block table ignored — all of which collapse the cosine rather than nudging the SNR. |

**Two things this case explicitly cannot show.** It does not discriminate "the
`Pi` rotation was dropped": on i.i.d. Gaussian channels a bare absmax quantiser
scores the same ~18 dB, because what the rotation buys is protection against
anisotropic, outlier-heavy channels — measured on real activations by the host
tier instead. And any timing printed here is the simulator's speed, which is why
it is printed and never asserted.

---

## 7. Tier 3 — `device/`

| | |
| --- | --- |
| **Binaries** | `test_device_950pr_turboquant`, `test_device_950pr_{matmul,rmsnorm,rotary_embedding,activation_swiglu,qwen_layer_golden}`, `test_device_950pr_benchmark_harness`, `bench_device_950pr_turboquant`, `bench_device_950pr_turboquant_ablation`, and `prof_device_950pr_msprof_trace`, which is not a ctest entry (§7.7) |
| **Purpose** | Correctness at the shapes the model actually decodes at, and **every timing number this project quotes**. |
| **Target hardware** | Physical Ascend 950PR. Real `libruntime.so` and `libascendcl.so`, no camodel on the link line, none on the `RUNPATH`, no fallback. |
| **Exit criterion** | Every correctness case passes; the benchmarks exit 0 having produced a table, or exit 77 when no 950PR is attached, which ctest records as a skip. A benchmark that fails its checksum is a **correctness** failure wearing a benchmark's clothes — the launches stopped agreeing with each other — and must be treated as one. |

`bench_device_950pr_turboquant` additionally refuses to run at all if it finds a
camodel in its address space, exiting 77 rather than printing a table: a
benchmark is the one kind of binary for which a simulator produces a
plausible-looking number that means nothing.

**Shape:** `head_dim 256`, 8 query heads over 2 kv heads (a GQA group of four),
`block_size 128`, context `512 / 1024 / 2048`, one query token. `head_dim 256` is
the top of the range the codec sizes UB for and a width the 310P cannot run at
all. The block pool is four times the blocks a context needs, so the block table
is a genuine scatter rather than a consecutive run that would stay resident.

### 7.1 Why the camodel is excluded at run time as well

Under `RUN_MODE=sim` the camodel reports a genuine `Ascend950PR_*` bin from
`aclrtGetSocName()` — that is what makes the simulator useful, and exactly why
the SoC name cannot distinguish it from silicon. So `REQUIRE_ASCEND_950PR` would
pass and the suite would start, then run for days on the `S=2048` cases, looking
like a hang rather than a skip.

`REQUIRE_PHYSICAL_ASCEND_950PR` asks a different question: it scans
`/proc/self/maps` for `libruntime_camodel.so` or anything loaded out of a
`tools/simulator/` directory, and skips with the offending path in the message.
Silicon is the default assumption — a host whose `/proc` cannot be read reports
no evidence rather than skipping. Since the tier no longer *links* a camodel
this is now defence in depth rather than the only guard, but it still catches an
`LD_LIBRARY_PATH` or `LD_PRELOAD` that would otherwise go unnoticed.

Observed skip message on a camodel build:

```
Test targets a physical Ascend 950PR and this process has the CANN camodel loaded
(/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/simulator/dav_3510/lib/libruntime_camodel.so).
Rebuild with -DRUN_MODE=npu and run on the part, or set ASCEND_TEST_ALLOW_SIMULATOR=1
to run it here anyway (hours).
```

Two environment overrides exist, both for smoke-checking a binary and neither
for producing results:

| Variable | Effect |
| --- | --- |
| `ASCEND_TEST_ALLOW_SIMULATOR=1` | Runs the bare-metal suite under a camodel anyway. |
| `ASCEND_TQ_BARE_METAL_CONTEXTS=16` | Replaces the `64,512,1024,2048` sweep, so such a run is merely slow rather than pointless. |

### 7.2 `test_device_950pr_turboquant`, case by case

| # | Case | Purpose | Validation target |
| --- | --- | --- | --- |
| 1 | `DeviceIsPhysicalSiliconAndReportsItsTopology` | The case that says "these numbers came from a part". | `aclGetDeviceCapability(ACL_DEVICE_INFO_VECTOR_CORE_NUM)` answers, the count is positive, and `aclrtGetMemInfo(ACL_HBM_MEM, ...)` reports free HBM. The core-count check is an `EXPECT`, not an `ASSERT`: a grid is only a work split, so a wrong count changes how many blocks launch and not what they compute — that should be loud, not fatal. |
| 2 | `WritePathMatchesTheCpuReferenceAcrossContexts` | The write path at every context in the sweep. | Bin drift `<= 1` and differing channels `<= 2%` per plane, plus worst scale relative error `<= 1e-5`. Bounds are **not** relaxed for the larger shape: nothing about a longer context makes a bin slip more likely, because quantisation is per vector. |
| 3 | `PackedCacheIsByteIdenticalOnRotationExactInputs` | The strict test the one-bin tolerance has to exist without. | Byte-for-byte `memcmp` equality of every written cache row against the CPU reference, and a scale of exactly `1.0`. No tolerance, no drift count, no statistics. See §7.3. |
| 4 | `DecodeMatchesTheCpuReferenceAcrossContexts` | The split/combine pipeline at production context lengths, reading back the cache the device itself wrote. | `cos > 0.999`, `relL2 < 5e-3`. |
| 5 | `CacheGeometryAndAlignmentMatchTheDocumentedLayout` | The memory-layout contract, on the device. | `scale_slot` is a whole 32-byte burst; a token's packed bytes across all kv heads are a whole number of bursts (the claim that makes the scatter one aligned `DataCopy`); the shim and the reference agree; allocation sizes match the production host's arithmetic; **every device base the kernels are handed is 32-byte aligned**; and the fp16-to-4-bit footprint ratio is above 3x, stated in bytes rather than as a ratio to be trusted. |
| 6 | `WritePathTouchesNoByteOutsideItsSlotMapping` | Addressing, at scale, on real hardware. | Both caches are pre-filled with `0x5A`. Every eighth token is marked `-1`. Afterwards: every byte of every row the mapping does not name still reads `0x5A`, **and** no row the mapping does name is still all-poison — so a kernel that wrote nothing at all cannot pass by leaving the poison intact everywhere. |
| 7 | `RepeatedDecodeLaunchesAreBitIdentical` | Determinism, at the longest context in the sweep, where the grid uses the most sequence splits and the combine stage has the most partials to reduce. | Eight decode launches from one filled cache, compared on the raw fp16 bit patterns. A difference is a race or an uninitialised workspace, not rounding. This is a regression test for a class of defect that was invisible at one launch. |

### 7.3 How case 3 gets to be bit-exact

`Pi` is an involution, so feeding the write path `Pi u` hands the codec back
`u`. Choosing `u` in `{-1, +1}^D` makes every subsequent step exact on both
sides:

- the sum of squares is exactly `D` **in any summation order**, so the device's
  tree reduction and the host's serial loop cannot disagree, and the scale is
  exactly `1.0` on both;
- every normalised coordinate is exactly `+-1.0`, which sits 0.2 away from the
  nearest Lloyd-Max decision boundary (`0.7995` and `1.0992`) — no tie-break is
  even in reach;
- `Pi u` is itself a multiple of `1/8` bounded by `sqrt(D)`, so it survives the
  fp16 store the device reads it through without rounding.

That last point is a premise, not a hope, so the case **asserts** it — every
channel must round-trip through fp16 unchanged before anything else runs. The
construction needs `1/sqrt(D)` to be a power of two, which holds for `D = 64`
and `D = 256` and is stated as such at the construction site.

### 7.4 The other device binaries

| Binary | Purpose | Notes |
| --- | --- | --- |
| `test_device_950pr_matmul`, `_rmsnorm`, `_rotary_embedding`, `_activation_swiglu` | Stock CANN operators for the four non-attention stages of a Qwen3.5 layer. | No camodel path exists for these at all — the simulator ships no binary kernels for them — so silicon is their first run anywhere. |
| `test_device_950pr_qwen_layer_golden` | End-to-end parity: nine stages of a real decoder layer against seven intermediate taps plus the final output. | Also carries 13 host-only cases that need no NPU, which check the dump, the loader and the CPU references. Needs `data/golden_layer3`; `QWEN_GOLDEN_LAYER3_DIR` overrides the baked-in path. |
| `test_device_950pr_benchmark_harness` | The arithmetic a benchmark run does *around* its launches: sample validation, the `ASCEND_BENCH_*` parses, the alignment rules, the checksum comparison. | **Needs no device attached.** It lives in this tier rather than in `host/` because it includes `acl.h` and links `libascendcl.so`, which the host tier's entry criterion forbids. Every failure it covers produced wrong numbers on healthy hardware. |

### 7.5 `bench_device_950pr_turboquant`

This binary is an **end-to-end audit**, not a kernel microbenchmark. It asks one
question, twice — once for prefill and once for decode:

> does a TurboQuant layer finish an attention step sooner than the same layer
> built on the stock CANN operator, **after** every transformation the 4-bit
> rotated cache imposes has been paid for?

Everything that does not serve that question has been taken out. There is no leg
comparing TurboQuant against a hypothetical unquantised fp16 memory layout, and
no leg comparing one TurboQuant kernel against another: those are *ablations*,
they live in `bench_device_950pr_turboquant_ablation` (§7.6), and mixing them
with a competitive comparison invites reading an internal ratio as a product
claim. What is left is a component-by-component breakdown of the shipping path
and one native baseline to divide it by.

#### Which native operator, and why it is not the one that was asked for

The specification for this audit named `aclnnPromptFlashAttentionV5` for the
prefill role and `aclnnFusionIncrementalAttention` for the decode role.
**Neither symbol exists.** CANN 9.2.0-beta.2 ships:

| family | highest version in `include/aclnnop/` |
| --- | --- |
| `aclnn_prompt_flash_attention` | **V3** |
| `aclnn_incre_flash_attention` | **V4** |
| `aclnn_fused_infer_attention_score` | **V5** |

and `aclnnFusionIncrementalAttention` appears in no header, and in no exported
symbol, anywhere in the install. What *does* exist — and what "the native V5
suite" means on an Ascend950 — is **`aclnnFusedInferAttentionScore`**, whose V5
interface is the unified successor that replaces both families (§8: V1..V4 of it
return `361001` on this part). One operator covers both roles, selected by its
argument list:

| role | `sparseMode` | mask | K/V | `blockSize` | `actualSeqLengths` / `…Kv` |
| --- | --- | --- | --- | --- | --- |
| **prompt** | 3 (right-down causal) | 2048×2048 compressed `int8` | contiguous TND | 0 — no paging | cumulative over `C` / cumulative over `S` |
| **incremental** | 0 | none | paged pool as a 1-entry `aclTensorList` + block table | 128 | cumulative query tokens / context length **per sequence** |

`aclnnFusedInferAttentionScoreV2` remains the fallback, and every table names the
operator that actually planned.

Guessing a prototype for the two symbols that were asked for was considered and
rejected. A hand-declared argument list behind `dlsym` that does not match the
operator is undefined behaviour **at launch**, not a planning refusal — it takes
the stream down and every case queued behind it. This suite's rule is that every
`aclnn` prototype is transcribed from a header it can point at; see §8 and
`common/aclnn_ops_950pr.hpp`.

#### The accounting

```
T_TQ_E2E = T_rot_q + T_attn_core + T_rot_o
```

measured with ACL events **per component and again as one composite region over
the whole graph**. Both are reported. The components say where the time goes; the
composite says what the caller waits for; the difference between them is the
inter-stage launch overhead the component view cannot see, and the footer under
each table prints that residual as a median percentage rather than leaving a
reader to subtract.

| term | prefill | decode |
| --- | --- | --- |
| `T_rot_q` | `npu_turboquant_rotate_q` over the chunk's query | the same, over the step's query |
| `T_attn_core` | FIA V5 over the rotated basis `(Q~, K~, V~)` | `TurboQuantCubeDecodeSplit` + `TurboQuantPagedAttentionCombine`, or the AIV launcher that submits both |
| `T_rot_o` | `npu_turboquant_rotate_q` over the attention output, before the sigmoid gate | the same |

`T_rot_o` is **exactly `0.00 us` for a folded layer** — `W_o' = W_o (I ⊗ Π)`
absorbs the de-rotation offline — and it is printed as a number rather than a
dash, because it is a fact about the layer and not a missing sample. The CSV
flags it as `rot_o_is_structural_zero`. Qwen3.5 cannot fold: `attn_output_gate`
multiplies the context by `sigmoid(gate)` between attention and `o_proj`, so its
column is a measured kernel on the critical path.

Neither side's KV-cache **write** is inside its E2E figure, and both are visible:
TurboQuant's is Table A's `TQ Ingest / Reshape` column, and the native write
(`aclnnScatterPaKvCache`) is timed as `pf_v5_ingest` and carried in the prefill
CSV alongside a `net_speedup_with_ingest` column that includes both.

#### Why prefill is chunked

A step processes `C = min(S, 2048)` query tokens against the `S`-token prefix,
which is what vLLM's scheduler actually submits. Without it a 1M-context row is
`O(S²)` attention — on the order of `8 × 10¹⁸` FLOPs for DeepSeek-V4-Flash, hours
per iteration — and the table would be empty where it matters most. The
`Context (S)` column prints `S/C` whenever `C < S`, and plain `S` when the chunk
is the whole prefill. `ASCEND_BENCH_TQ_AUDIT_CHUNK` changes it.

Chunking is also what makes `T_rot_q` meaningful in prefill at all. A chunk's
query attends over a prefix that is already in the cache, and the cache stores
`Π k`; the query has to be rotated to match. Π is orthogonal, so attention in the
rotated basis is attention — `softmax(Q~ K~ᵀ) V~ = Π · softmax(Q Kᵀ) V` — which is
the identity the first device check below asserts.

#### The workload

Three flagship families at TP=8, one NPU's share of each. Serving ranks, not a
geometric sweep: a dense cartesian product of head counts produces hundreds of
shapes nobody deploys and buries the nine that matter.

| model | `D` | `H_Q` | `H_KV` | fold | decode path | why |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B | 128 | 4 | 1 | **no** | **Cube** (AIV before 13.31) | GQA 4:1; `attn_output_gate` blocks the fold |
| DeepSeek-V4-Flash-284B | 256 | 16 | 1 | yes | **Cube** | MLA decoupled latent KV; 16:1 exactly fills the Cube's M fractal |
| GLM-5.2-744B | 128 | 8 | 1 | yes | **Cube** (AIV before 13.31) | ultra-wide GQA |

| regime | `S` | `B` | warmup | timed |
| --- | --- | --- | --- | --- |
| short / interactive | 2 048 | 1, 4, 8 | 5 | 20 |
| enterprise long | 32 768 | 1, 4, 8 | 5 | 20 |
| ultra-long | 262 144 | 1, 2 | 1 | 3 |
| extreme needle / 1M | 1 048 576 | 1 | 1 | 3 |

3 families × 9 `(S, B)` pairs = **27 configurations**, each audited in both
phases: 27 rows in Table A and 27 in Table B.

**Two iteration budgets means two `BenchmarkRunner`s.** The generic per-case table
stamps one `warmup= iterations= pipeline_batch=` line on the whole of itself, so a
regime with a different budget gets its own runner and its own table rather than
rows its header misdescribes. `BenchmarkRunner::set_options` exists for this and
throws if it is called after a case has run.

#### The decode path is chosen by the GQA group, not by a flag

The Cube decode batches a kv head's query heads into the GEMM's `M` dimension, and
that fractal is 16 rows wide (`turboquant_host::kCubeTileM`). **Superseded 2026-09-17
(13.31):** every model now takes the Cube decode. Silicon measured it at 1.98x-2.18x and
the AIV-only path at 0.05x-0.19x, narrow groups included. `ASCEND_BENCH_TQ_AUDIT_PATH=aiv`
forces the AIV-only path.

**The AIV path's `T_DecodeSplit` is derived, and is marked `~`.** There is no
exported entry point for the AIV split alone — `turboquant_paged_attention_impl`
submits both stages — so that column is the measured core minus the separately
measured combine. The Cube path's split is measured directly and carries no
marker. The CSV carries the flag as `split_is_derived`.

#### The two device checks, before anything is timed

| check | what it asserts | bound |
| --- | --- | --- |
| **rotated-basis identity** | FIA over `(Q~, K~, V~)`, un-rotated per head on the host, equals FIA over `(Q, K, V)`. The device-level statement that the prefill core is computing *the layer's* attention, and what licenses timing the folded path with `T_rot_o = 0`. | cos > 0.999 |
| **decode tie-point** | the TurboQuant decode's output, un-rotated, against the native V5 decode over the same context in the same slots. | cos > 0.90 |

The second bound is loose on purpose: 4 bits against 16 will not do better, and
the fidelity gates are §5, §6 and `test_device_950pr_turboquant`. What it catches
is the one failure the checksums are blind to — **timing a kernel that is reading
an empty cache**. It runs on the smallest configuration whose context also fits an
exact host image (`B·S·H_KV·D ≤ 2²²`), which is what lets the fp16 paged cache be
built slot-exact rather than tiled. Both checks are recorded as ctest-visible
failures when they miss.

#### Cases, per configuration

| leg | what is timed |
| --- | --- |
| `pf_ingest` | the TurboQuant cache write over the chunk — the Cube-native codec on the Cube path, the Lloyd-Max one on the AIV path |
| `pf_rot_q` | `npu_turboquant_rotate_q` over the chunk's query |
| `pf_attn_core` | FIA V5 over `(Q~, K~, V~)` |
| `pf_rot_o` | the O de-rotation — skipped with a reason on a folded layer |
| `pf_e2e` | the composite: rot-q, core, and rot-o when the layer needs it |
| `pf_v5` | FIA V5 over `(Q, K, V)`, native uncompressed tensors |
| `pf_v5_ingest` | `aclnnScatterPaKvCache`, the baseline's own cache write |
| `dec_rot_q` | `npu_turboquant_rotate_q` over the step's query |
| `dec_split` | `TurboQuantCubeDecodeSplit` — Cube path only; skipped with a reason on the AIV path |
| `dec_combine` | `TurboQuantPagedAttentionCombine`, on both paths |
| `dec_attn_core` | split + combine, or the AIV launcher |
| `dec_rot_o` | the O de-rotation — skipped with a reason on a folded layer |
| `dec_e2e` | the composite |
| `dec_v5` | FIA V5 paged decode over the fp16 pool |

**Decode legs run before prefill legs, per configuration, and the order is load
bearing.** `pf_ingest` rewrites the tail of the TurboQuant cache from the chunk's
own K/V and `pf_v5_ingest` rewrites part of the fp16 pool; running them first
would leave the decode legs reading a cache the setup phase did not build. Neither
rewrite changes any latency — both write the same bytes to the same slots every
launch — but the order keeps the decode measuring a cache whose provenance is the
one the tie-point checked.

Every case carries a checksum the harness verifies after warmup and again after
the last timed iteration. One consequence is worth naming: `dec_combine` timed on
its own would reduce a workspace of zeros — a zero softmax denominator, so a
non-finite checksum and a spurious failure — so `Scenario::Prime` runs one untimed
pass of the attention core and both FIA plans before any case is registered,
leaving the workspace and every output buffer holding real values.

#### What is still not priced, said here rather than hidden

1. **The `fp32 → fp16` narrowing between the rotation kernel and FIA, in the
   prefill core only.** `npu_turboquant_rotate_q` writes fp32; FIA reads fp16.
   Production does the cast with a torch op; this suite has no header-verified
   `aclnnCast` prototype and will not guess one, so the rotated tensors FIA is
   handed are prepared on the host in fp16. The modelled byte cost of both casts
   is in the prefill CSV as `untimed_cast_bytes`. **The decode core has no such
   gap** — the split kernels consume the fp32 rotation directly.
2. **The sigmoid gate itself, for Qwen3.5.** `T_rot_o` stops where the
   specification says it stops, and the gate is the same elementwise cost on both
   sides of the comparison.
3. **Accuracy.** TurboQuant stores 4 bits per coordinate and the native leg stores
   16; the outputs are not the same numbers and are not meant to be. The decode
   tie-point above is a provenance check, not a fidelity gate.

#### Traffic, and what the GB/s columns mean

Every byte figure is **compulsory traffic**: each distinct byte the step must move
across HBM, counted once, under one convention for every leg. A cached KV row is
counted once per *kv* head and not once per query head, whichever path reads it,
so the ratio between two legs' byte counts is the ratio between their storage
formats and not between their task decompositions. Re-reads that L2 may or may not
absorb are not guessed at in either direction. `HBM Bandwidth` (Table A) and
`Effective BW` (Table B) are the pipeline's compulsory traffic over its measured
**composite** time.

`Memory Compression Ratio` is fp16 KV residency over TurboQuant's, **scale plane
included**. At `H_KV = 1` a token's scale slot is `round_up(2·H_KV, 8) = 8` fp32
lanes for 2 live ones, so 24 of its 32 bytes are burst padding — which is most of
the gap between the measured ratio and a naive 4.00×, and why every row in this
sweep lands near 3.2× (`D = 128`) or 3.6× (`D = 256`) rather than at 4.

#### Reported

Two summary tables, printed after every case has run, plus the generic per-case
table and CSV from the shared harness. The column definitions are §7.5.1 below.

**Table A — prefill pipeline performance**

```
Model | Context (S) | B | H_Q/H_KV | D | TQ Ingest | T_rot_q | T_attn_core |
T_rot_o | TQ_E2E | V5_Native | Net Speedup | HBM GB/s | KV Saved MB
```

**Table B — decode latency breakdown**

```
Model | Context | B | Path | T_rot_q | T_DecodeSplit | T_Combine | T_rot_o |
TQ_E2E | V5_Decode | Net Speedup | Eff GB/s | Compression
```

`Net Speedup` is `V5 / TQ_E2E` in both: **above 1.000× is TurboQuant ahead.** An
absent leg prints a dash; `0.00` would read as a measurement, except where it is
one (a folded `T_rot_o`).

#### 7.5.1 Column definitions

| column | table | definition |
| --- | --- | --- |
| `Model` | A, B | the family label from the catalogue at the head of the file |
| `Context (S)` | A | KV prefix length; `S/C` when the step is a `C`-token chunk of it |
| `Context` | B | KV length per sequence |
| `B` | A, B | sequences in the batch. In decode that is also the query token count |
| `H_Q / H_KV` | A | query and kv heads on one NPU at TP=8 |
| `D` | A | head size |
| `Path Mode` | B | `Cube` or `AIV`, chosen by `H_Q/H_KV ≥ 16` |
| `TQ Ingest / Reshape` | A | `pf_ingest`: rotate, quantise and scatter the chunk. **Not inside `TQ_E2E`** |
| `T_rot_q` | A, B | `pf_rot_q` / `dec_rot_q` |
| `T_attn_core` | A | `pf_attn_core`: FIA V5 over the rotated basis |
| `T_DecodeSplit` | B | `dec_split` on the Cube path; core − combine, marked `~`, on the AIV path |
| `T_Combine` | B | `dec_combine`, measured directly on both paths |
| `T_rot_o` | A, B | `pf_rot_o` / `dec_rot_o`; exactly `0.00` on a folded layer |
| `TQ_E2E` | A, B | `pf_e2e` / `dec_e2e`, **one composite ACL-event region**, not a sum |
| `V5_Native` / `V5_Decode` | A, B | `pf_v5` / `dec_v5` on native uncompressed tensors |
| `Net Speedup` | A, B | `V5 / TQ_E2E` |
| `HBM Bandwidth` / `Effective BW` | A, B | compulsory E2E traffic ÷ measured composite time |
| `KV Cache Memory Saved` | A | `(fp16 residency − TurboQuant residency)` for `B·S` tokens, MiB |
| `Memory Compression Ratio` | B | fp16 residency ÷ TurboQuant residency, scale plane included |

The CSVs carry everything above plus, per row: the regime label, the chunk length,
the split count, `tq_e2e_sum_of_parts_us` beside the composite, per-leg GB/s and
TFLOP/s, p95s, the raw byte and FLOP counts, `untimed_cast_bytes`, the flags
`rot_o_is_structural_zero` and `split_is_derived`, the warmup/iteration budget the
row ran under, and which timing mode the median came from. An absent leg leaves
its fields **empty** rather than writing 0: a CSV is the artifact most likely to be
read without its banner.

#### Knobs

| variable | default | effect |
| --- | --- | --- |
| `ASCEND_BENCH_TQ_AUDIT_MODELS` | all | `qwen35,dsv4,glm52` |
| `ASCEND_BENCH_TQ_AUDIT_S` | all | restrict the context regimes |
| `ASCEND_BENCH_TQ_AUDIT_B` | all | restrict the batches |
| `ASCEND_BENCH_TQ_AUDIT_PHASES` | both | `prefill`, `decode` |
| `ASCEND_BENCH_TQ_AUDIT_LEGS` | all | restrict the legs by name |
| `ASCEND_BENCH_TQ_AUDIT_CHUNK` | 2048 | the prefill chunk `C` |
| `ASCEND_BENCH_TQ_AUDIT_PATH` | `cube` | `aiv` forces the AIV-only decode (13.31) |
| `ASCEND_BENCH_TQ_AUDIT_GLM_D` | 128 | GLM-5.2 is specified at 128 **or** 256 |
| `ASCEND_BENCH_TQ_AUDIT_WARMUP` / `_ITERS` | 5 / 20 | the `S ≤ 32K` budget |
| `ASCEND_BENCH_TQ_AUDIT_ULTRA_WARMUP` / `_ULTRA_ITERS` | 1 / 3 | the `S ≥ 262K` budget |
| `ASCEND_BENCH_TQ_AUDIT_PREFILL_CSV` | unset | Table A as a CSV |
| `ASCEND_BENCH_TQ_AUDIT_DECODE_CSV` | unset | Table B as a CSV |
| `ASCEND_BENCH_TQ_FIA` | on | `0` drops every native V5 leg |

plus the shared `ASCEND_BENCH_*` set (`common/benchmark.hpp`). Note that
`ASCEND_BENCH_MODES` selects **three** timing modes by default and each is a full
warmup-plus-iterations run: on the ultra-long rows that is the difference between
minutes and tens of minutes. `pipeline_batch` is pinned to 1 regardless of
`ASCEND_BENCH_BATCH`, because these shapes are orders of magnitude heavier than
the microbenchmarks the shared default of 10 was chosen for.

A configuration whose estimated footprint exceeds 85% of free HBM is skipped with
both numbers in the reason, and every one of its legs gets a recorded skip so the
hole is visible in the report.

> **NOTHING THIS BINARY PRINTS HAS EVER BEEN TAKEN ON SILICON.** No Ascend 950PR
> part has been available to this project. The kernels are verified on the arch35
> camodel (§13.8, §13.10) and the FIA V5 argument list has never been through a
> planning call on hardware (§8). Every number is hypothetical until that changes.

### 7.6 `bench_device_950pr_turboquant_ablation`

**What it answers:** where the kv4fp8 Cube split's latency goes. The split is
cut at six compile-time stages -- `DecodeAblationStage` in `turboquant_mode.h`, a
template parameter on `TurboQuantCubeDecodeSplit` tested only under
`if constexpr`. Each rung runs everything below it plus one piece, and the top
rung *is* `turboquant_mm_decode_split_kv4fp8_half`, not a copy of it. A rung's
median minus its neighbour's is that piece's cost in place, including what it
does to pipe overlap.

| Stage | Adds | Grows with |
| --- | --- | --- |
| `stage0_mte2` | `CopyInTile`: packed K/V and scale tile, GM -> UB, then `PipeBarrier<PIPE_ALL>` (below) | tiles |
| `stage1_unpack` | the `MTE2_V` edge, `UnpackAffine` per chunk, `PipeBarrier<PIPE_ALL>` per pass (below) | tiles |
| `stage2_query_prep` | pre-rotated query GM read, amax and scale, fp8 cast | **tasks, not S** |
| `stage3_l1_staging` | V -> MTE3 edge and `DataCopy` per chunk, `MTE3_V` per pass, the query's NZ staging | tiles |
| `stage4_score_gemm` | `PipelineAic`'s score half -- two `LoadData`, `Mmad`, `Fixpipe` -- and the `SlotReady` / `ScoresReady` / `SlotFree` flags | tiles |
| `stage5_full` | acc/state init, `qScaleInv_`, softmax, `ProbsReady`, context GEMM, `ContextReady`, accumulate, partial writeback | tiles |

**Stage 2 no longer contains a Walsh-Hadamard transform at all.** It used to:
the cache is written already rotated, so the split's only transform was the
query's, run once per query head of each kv head *and once per sequence split*.
That rotation moved out to `npu_turboquant_rotate_q` -- see 13.22 -- and what is
left in this rung is the query's GM read, its amax and the operand cast. The
rung keeps its number, which crosses the launch boundary, and the benchmark
enqueues the rotation once before any stage so no rung's delta carries it.

The output's inverse rotation is in the combine, which the ladder does not time,
and is unchanged.

**Four places the cut kernel differs from the shipping one**, each on purpose:

1. Stages 0–3 contain no Cube instruction, so the AIC half only walks the task
   list. It is still launched as a MIX block, like `sim_hadamard_aiv`, so the grid
   and `MixBlockIdx()` mean what they mean in production.
2. **No cut leaves a HardEvent wait with nothing on its waiting pipe.** Stage 0
   ends `CopyInTile` with `PipeBarrier<PIPE_ALL>` instead of `SyncMte2ToVector()`,
   because it has no vector reader. Stages 1–2 end each `UnpackToL1` with the same
   barrier instead of `SyncVectorToMte3()`, because they issue no MTE3 copy --
   but something still has to order the next tile's MTE2 read after this tile's
   unpack, since both touch `kvBuf_`. Stage 1 swaps stage 0's barrier for the
   edge its unpack consumes; stage 3 swaps stage 2's for production's
   `SyncVectorToMte3()` per chunk plus `SyncMte3ToVector()`. So stage 3's delta is
   the copies and the difference in synchronisation.

   The first version used the bare event edges here. They were not what hung --
   the camodel ran stages 0–2 in 39, 124 and 158 s with every exception dump
   empty, and in CANN 9.2.0-beta.2 `SetFlagImpl` / `WaitFlagImpl`
   (`asc/impl/basic_api/kernel_event.h`) issue each side as its own pipe
   instruction, while `FetchEventID` only picks the first free id and occupies
   nothing. They were replaced because "every wait has a consumer" is the rule the
   shipping kernel keeps, and silicon is not the camodel.
3. `qScaleInv_` -- a scalar UB read per query head -- and the scale-index
   `ArithProgression` only feed the softmax, so they belong to stage 5.
4. Every flag set is gated together with the wait that balances it (13.14.2),
   **on both halves**: `SlotReady`, `ScoresReady` and `SlotFree` from stage 4;
   `ProbsReady` and `ContextReady` only at stage 5. The first version got this
   wrong on the Cube side, and it deadlocked -- see below.

#### The stage 4 deadlock

The first version gated the AIV's `SignalProbsReady()` and its `kFlagContextReady`
wait to stage 5, and gated the call to `PipelineAic` to stage 4 -- but not
`PipelineAic`'s body. Under a stage 4 cut the Cube therefore still waited
`kFlagProbsReady` after the first score GEMM, and would then have set a
`kFlagContextReady` nobody waits for.

It hung on physical 950PR silicon, and it reproduced on the arch35 camodel
(`test_sim_950pr_turboquant_ablation`, S = 256, one split of four tiles): stages
0–3 returned in 39 / 124 / 158 / 161 s with empty exception dumps, and stage 4 was
still running 39 minutes later with every control unit spinning in the same
instruction:

| core | CCU log, repeating | flag id |
| --- | --- | --- |
| `cubecore0` (AIC) | `WAIT_FLAG_DEV pc=0x10d12d9c` | **6**, `kFlagProbsReady` |
| `veccore0`, `veccore1` | `WAIT_FLAG_DEV pc=0x10d28d48` | **4**, `kFlagScoresReady` |

The AIC posted tile 0's `ScoresReady` and blocked on id 6; both vector subcores
consumed it, staged tile 1, and blocked on id 4. Every shape deadlocks this way,
including one tile per task, which is why silicon hung immediately.

**The fix** gates the `kFlagProbsReady` wait, the context GEMM and the
`kFlagContextReady` set inside `PipelineAic` to stage 5, the mirror of the AIV's
gates. The audit of every stage after it:

| stage | AIV sets | AIV waits | AIC sets | AIC waits |
| --- | --- | --- | --- | --- |
| 0–3 | none | none | none | none |
| 4 | `SlotReady` x numTiles | `ScoresReady` x numTiles, `SlotFree` x max(numTiles - 2, 0) | `ScoresReady` x numTiles, `SlotFree` x max(numTiles - 2, 0) | `SlotReady` x numTiles |
| 5 | as 4, plus `ProbsReady` x numTiles | as 4, plus `ContextReady` x numTiles | as 4, plus `ContextReady` x numTiles | as 4, plus `ProbsReady` x numTiles |

Two things the audit brief asked about do not exist in this kernel:
`ProductIsMine()`, `SignalOperandsReady()`, `kFlagOperandsReady` and
`kFlagProductReady` belong to the lock-step `TurboQuantFp16DecodeSplit` and the
GEMM probe, not to `TurboQuantCubeDecodeSplit`, whose Fixpipe consumer is gated
with `IsPrimarySubcore()` and whose flags are the pipelined map in
`turboquant_cube_mm.h`. And stage 4's two back-to-back sets on different pipes
(`ScoresReady` on FIX, then `SlotFree` on MTE1) are safe: `CrossCoreSetFlag`
issues an `ffts_cross_core_sync` instruction on its own pipe
(`dav_3510/kernel_operator_sync_impl.h`, `NotifyEventImpl`) rather than loading a
single pending register. Production has the same adjacency (`ContextReady` then
`SlotFree`) whenever a task walks three tiles or more.

**Checks.** Every stage runs against a workspace freshly uploaded as zeros, and
a stage 0–4 case fails if it is not still zero afterwards -- a leaked writeback.
Straight after stage 5 the combine runs once, untimed, and the output is held to
cos >= 0.90 against an fp32 host attention. A shape that misses is printed
UNTRUSTED and the suite exits 1. Because each check is local to its stage, any
order `--stage=` names is valid.

**Hang guard.** The shared harness synchronises without a deadline, so each
stage's first launch is issued by the suite itself and synchronised with
`aclrtSynchronizeStreamWithTimeout` before the harness sees the case. A miss
names the stage, prints the waterfall so far, and exits **3** without the shared
report or any stream teardown -- a deadlocked stream does not come back, and
destroying it blocks too. `--stage=4` runs one stage in its own process.

**D = 512 is outside what `CheckHeadSize` lets a model reach**, and was asked for
anyway. By static count the split's UB comes to 234,624 B of the part's
`ub_size` 253,952 (`Ascend950PR_9599.ini`): 137,728 B of decode buffers, 41,312 B
of rotation codec, 55,584 B of affine codec. L1 needs 140,288 B of 524,288, L0B
32,768 of 65,536, L0C 32,768 of 262,144. It fits on paper with 19 KB to spare and
has never run. The fidelity gate is what decides whether its rows mean anything,
and every D = 256 shape runs first so a D = 512 fault cannot take those numbers
with it.

**Output:** the shared per-case table, then a waterfall per shape -- median,
delta, share of stage 5, P95 -- and a delta-by-context table per head size, where
stage 2 should read flat.

**Knobs:** `--stage=<list>` (stages 0..5, in order) and
`--sync-timeout-ms=<ms>` (default 30000; 0 waits forever) on the command line,
from `device/bench_main_950pr_ablation.cpp`, which rejects a malformed value
rather than falling back -- `--stage=4x` silently running all six would relaunch
the stage it meant to isolate. Each flag sets `ASCEND_BENCH_TQ_ABLATION_STAGES` /
`ASCEND_BENCH_TQ_ABLATION_SYNC_TIMEOUT_MS`. Plus the shared `ASCEND_BENCH_*` set,
`ASCEND_BENCH_TQ_ABLATION_DIMS` (powers of two in [64, 512]) and
`ASCEND_BENCH_TQ_ABLATION_CONTEXTS` (positive multiples of 8, see 13.20). An
invalid environment value falls back to its default, with a message.

**Build:** the stage 0–4 entry points exist only under
`VLLM_ASCEND_TQ_DECODE_ABLATION`, which the wheel's library does not define.
`csrc/tests/turboquant/CMakeLists.txt` has to set it twice, because
`ascendc_library` compiles the source twice: `ascendc_compile_definitions` for
the device precompile, and `ascendc_compile_options(...
-forward-options-to-host-compiler ...)` for the host stub. With only the first,
the five kernels and their launchers build and
`turboquant_mm_decode_ablation_impl` does not, so the benchmark fails to link.

**Status, 2026-09-14.** Clean under `-DVLLM_ASCEND_TESTS_WERROR=ON` in three
configurations on CANN 9.2.0-beta.2 -- device (`RUN_MODE=npu`, with benchmarks),
sim (`RUN_MODE=sim`) and host-only -- with the five kernels, their launchers and
`turboquant_mm_decode_ablation_impl` exported. The benchmark has not been re-run
on silicon since the fix; it refuses the camodel by construction.

What has run is `sim/test_sim_950pr_turboquant_ablation` on the arch35 camodel, at
B = 1, 8 query over 2 kv heads, head_size 256, block 64, S = 256 -- one split
walking four tiles, the smallest shape at which every flag id fires, both
`SlotFree` slots included. Each stage checks that it returns inside a watchdog,
that the workspace still holds its sentinel (stages 0–4) or was fully written
(stage 5), and that no exception dump holds a byte; the run script repeats the
dump census after the process exits.

| run | cache | stages, in one process | seconds per stage | workspace | exception dumps after exit | exit |
| --- | --- | --- | --- | --- | --- | --- |
| before the fix | written on device | 0, 1, 2, 3, 4 | 39 / 124 / 158 / 161 / **hung** | untouched through 3 | 0 bytes through 3 | killed in stage 4 |
| A, after the fix | host-filled | 0, 1, 2, 3, 4 | 43 / 143 / 183 / 160 / **197** | untouched, all five | 32 files, 0 bytes | **0** |
| B, after the fix | written on device, 1,400 s | 4, 5 | 180 / **230** | untouched at 4; every partial written at 5 | 32 files, 0 bytes | **0** |

Stage 4 now returns in 197 s against 161 s for stage 3. Run B's stage 5 is the
shipping kernel walking four tiles in one task, which makes it **the first time the
`kFlagSlotFree` path has executed anywhere** -- 13.11 put it out of reach of every
default shape. Its output is 2048 of 2048 finite at **cos 0.989226** against the
fp32 host attention. `ASCEND_TQ_ABLATION_HOST_FILL=1`
fills the packed cache from the host instead of launching the 1,232 s write:
nothing on the flag protocol reads the values, so it is the right cache for a
hang check and the wrong one for a cosine.

#### The unpack bypass

**What it answers:** how much of the full split is `UnpackAffine`, measured by
removing it rather than by subtracting ladder rungs. `BYPASS_UNPACK`, a fourth
template parameter on `TurboQuantCubeDecodeSplit` (stage 5, affine mode only),
builds `turboquant_mm_decode_bypass_unpack_kv4fp8_half`. It takes a
pre-unpacked **fp8 operand** cache, `[blocks, block, kv_heads, head_size]`, in
place of the packed one.

**What changes, and what does not.**
- Changed: `CopyInTile` reads that cache GM -> UB in 32-byte group bursts, and
  `StageOperandsToL1` hands it to L1 with the same `SyncVectorToMte3()` +
  `DataCopy` pairs and `SyncMte3ToVector()` as `UnpackToL1`.
- Unchanged: query prep, softmax, every cross-core flag, both GEMMs, the partial
  writeback and the combine.
- The 2 x 64 x `head_size` tile buffer is allocated after every other buffer, so
  the shared ones keep their UB base addresses (13.21).
- fp8 rather than fp16, because the Cube operand is fp8 and there is no
  half -> fp8 cast.
- The bypass moves **twice the GM bytes** of the packed read, so its saving is a
  lower bound on the unpack's cost.
- At `head_size` 512 the extra 65,536 B would exceed UB, so the bench skips
  that size.

**The cache it runs on.** `common/turboquant_mirrored_cache.hpp` mirrors each
K/V vector's halves, `x[j] == x[j + D/2]`. Every packed byte therefore holds
two equal nibbles, and the fp8 image is the codec's expansion under either
nibble order. Unwritten slots hold +0.5, the expansion of a zero byte. Both
splits run on one such cache, so their partials must match **bit for bit**.
`test_host_turboquant_fidelity` pins the builder.

**Camodel smoke,** `TurboQuantDecodeAblation.BypassUnpackKeepsThePipelineOnTheCamodel`
(4 query heads over 1 kv head, D 256, block 64, one split, 2026-09-15):

| S | path | split wall | cos vs fp32 | partials | exception dumps | exit |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | standard | 136.2 s | 0.999496 | written | 0 bytes | 0 |
| 256 | bypass | 96.8 s (0.711x) | 0.999496 | **0 of 1088 words differ** | 0 bytes | 0 |
| 2048 | standard | 883.0 s, 290,297 ticks over a setup-only process | 0.999530 | written | 0 bytes | 0 |
| 2048 | bypass | stopped before it returned | -- | -- | -- | -- |

On the camodel, `aclrtEventElapsedTime` reads 0.000 ms around every launch, so
the only measures there are wall clock and the simulator's per-process
`Total tick`. Wall clock is not device time. **Keep camodel runs at S <= 256.**
The timing question belongs to the bench: `--bypass-unpack` adds the pair
after the ladder, and `--bypass-unpack=only` runs the pair alone. It reports
standard and bypass medians, the ratio, the removed share, both tile reads,
partial identity and both cosines. Setup failures, bit mismatches, or a cos
below 0.90 are recorded as failures and marked UNTRUSTED.

### 7.7 `prof_device_950pr_msprof_trace`

**What it answers:** where the time between launches goes. §7.5 reduces the
composite-versus-sum-of-parts gap to one median percentage; this binary produces
the timeline that gap comes from. It is a trace harness, not a benchmark: no
warmup, no timing loop, no statistics. It is **not a ctest entry**, because
without msprof attached it records nothing.

Per shape -- model x S x B, and phase under `--mode=both` -- in one process:

| Step | Ranges and marks | Contents |
| --- | --- | --- |
| setup | `setup` | buffers and descriptors; for decode, the whole-context TurboQuant cache write, synchronised |
| leg 1 | `TQ_Pipeline_Start`, `TQ_Pipeline`, `TQ_Pipeline_End` | decode: `TQ_rotate_q`; `TQ_DecodeSplit` and `TQ_Combine` (Cube) or `TQ_PagedAttention_SplitCombine` (AIV); `TQ_rotate_o` when W_o is unfolded; `TQ_sync`. Prefill: `TQ_rotate_q`, `TQ_FIA_V5_GetWorkspaceSize`, `TQ_FIA_V5_Launch`, `TQ_rotate_o` when unfolded, `TQ_sync` |
| gap | -- | 10 ms host sleep |
| leg 2 | `V5_Native_Start`, `V5_Native`, `V5_Native_End` | `V5_GetWorkspaceSize` (the plan and its workspace allocation), `V5_Launch`, `V5_sync` |
| after | `readback`, `teardown` | absolute sum over the leading 2^20 elements of each leg's output, then every buffer freed |
| gap | -- | 10 ms host sleep before the next shape |

A leg that throws gets a `_Aborted` mark instead of its `_End`, and the shape's
log line says `FAILED` with the CANN message.

The shapes, the path routing and both FIA V5 argument lists are the audit's.
The model table, `SelectPath` and `PrefillChunk` moved into
`common/turboquant_audit_models.hpp`, which both binaries include, so every trace
is of a Table A or Table B row by construction, and
`ASCEND_BENCH_TQ_AUDIT_PATH`, `_GLM_D` and `_CHUNK` steer both.

**Markers.** mstx ranges and marks from `libms_tools_ext.so`
(`tools/mstx/lib64`), resolved with `dlopen` like every other CANN symbol this
suite calls rather than linked. Launch ranges are passed the stream, so each is
stamped on the host when the stage is enqueued and on the device when the stream
reaches it. The planning and synchronisation ranges get a null stream, because
they are host work. If mstx does not resolve, the harness falls back to
`aclprofMarkEx`, which `libmsprofiler.so` exports and `libascendcl.so` already
loads, and every range becomes a `_Start`/`_End` pair of marks. Either way it is
`msprof --msproftx=on` that records them:

```bash
msprof --output=./prof/tq_trace --msproftx=on --task-time=l1 --runtime-api=on ./build/dev/device/prof_device_950pr_msprof_trace --models=DeepSeek-V4-Flash,Qwen3.5-9B,GLM-5.2-744B --contexts=2048,32768 --batches=1,4 --mode=decode
```

The binary prints that line for its own flags before it touches the device.

**Caveats, each deliberate:**

1. **No warmup**, so the first shape carries every kernel image load and the
   operator's first plan. Read it as the cold-start profile, or put the shape
   that matters second.
2. **The AIV split is not separable.** `turboquant_paged_attention_impl` enqueues
   split and combine through one launcher and exports no split-only entry point,
   so one range covers both. msprof's task timeline still shows the two kernel
   tasks under it. The Cube path's two launches get a range each.
3. **Prefill shares FIA between the legs.** Leg 1 plans and launches FIA V5 over
   the rotated basis, leg 2 over the plain one, so leg 2 may reuse state leg 1
   created. And as in §7.5, FIA reads a host-rotated fp16 query: the fp32 -> fp16
   narrowing after rotate_q is not launched.
4. **Stream-bound markers are device tasks too.** They add a few tiny tasks per
   shape to the timeline they label.
5. **The readback check is not a fidelity bound.** It fails a shape whose output
   is all zero or non-finite, which is what a leg that dispatched nothing looks
   like, and nothing more.

Exit status: 0 when every shape dispatched both legs with non-empty output, 1
otherwise, 2 on a bad flag, 77 with no 950PR attached or a camodel loaded.

---

## 8. The fp16 baseline, and the V2 → V5 migration

The fp16 leg is what turns the benchmark from "how fast is TurboQuant" into "what
does TurboQuant cost". It was empty, and this is why.

**What was wrong.** The leg called `aclnnFusedInferAttentionScoreV2`. Measured on
CANN 9.1.0, its planning call returns `361001`:

```
AclNN_Runtime_Error(EZ9903): Interface aclnnFusedInferAttentionScore
versions V1 to V4 are no longer supported on Ascend950.
```

The whole V1..V4 family is withdrawn on this part, so every `fp16_decode_s*` case
skipped and the `fp16 us` and `speedup` columns came out empty.

**The fix.** `aclnnFusedInferAttentionScoreV5` is what replaces the family. The
benchmark now resolves **V5 first and keeps V2 as the fallback**, so the same
source works on a CANN release that predates V5 and on any part where V2 still
exists; the report names whichever one it used, and prints why the column is
empty when neither resolves.

**What V5 adds over V2** — seven optional inputs after
`actualSharedPrefixLenOptional`, and two `int64_t` scalars after
`valueAntiquantMode`:

| Added parameter | What it is for | Value here |
| --- | --- | --- |
| `queryRopeOptional`, `keyRopeOptional`, `keyRopeAntiquantScaleOptional` | MLA's split-RoPE path, where the rotary half of Q/K is a separate tensor | `nullptr` — the RoPE is already folded into the cached K |
| `dequantScaleQueryOptional` | per-query dequant for a quantised query | `nullptr` — the query is fp16 |
| `learnableSinkOptional` | the learned attention-sink logit some long-context models carry | `nullptr` |
| `qStartIdxOptional`, `kvStartIdxOptional` | window offsets for the sliding-window sparse modes | `nullptr` — `sparseMode` is 0 |
| `queryQuantMode` | whether the query is quantised | `0` (`shapes950::kFiaQueryQuantModeNone`) |
| `pseType` | how a positional-encoding shift combines with the scores | `1` (`shapes950::kFiaPseTypeDefault`), torch_npu's default. Unused because `pseShift` is null, but the operator still range-checks it |

The geometry is unchanged and is identical to the TurboQuant leg's — `head_dim`
256, 8 heads over 2 kv heads, `block_size` 128, layout `TND` with a block table,
matching the tensor shapes the benchmark builds. That identity is what makes the
two legs' numbers comparable at all.

The prototype is transcribed from
`$ASCEND_TOOLKIT_HOME/include/aclnnop/aclnn_fused_infer_attention_score_v5.h`
into `common/aclnn_ops_950pr.hpp` and, like every declaration in that file, is
resolved with `dlsym` rather than linked — **so the compiler cannot check it**.
A mismatch surfaces as a non-zero planning status with the CANN diagnostic
attached, not as corruption.

> **NOT YET PLANNED ON HARDWARE.** No 950PR has been available, so the V5
> argument list has never been through a planning call. If the first silicon run
> returns non-zero here, the argument list is the first suspect — and within it,
> `kFiaPseTypeDefault` and the seven optional tensors are the two things to vary
> first. Until that run happens, **there is no measured `V5_Native`, no measured
> `V5_Decode` and no measured net speedup** in either of §7.5's tables. What
> exists is the traffic model behind their byte and compression columns, which is
> exact arithmetic over the two layouts and not a measurement.

> **The geometry above is the old fixed shape.** Since the audit rewrite, V5 is
> planned per configuration across three model families and both roles — the
> prompt role with `sparseMode` 3 and a compressed causal mask, the incremental
> role with `sparseMode` 0 and a block table — rather than at one `head_dim` 256
> / 8-over-2 shape. §7.5 has the current argument lists and the reason the two
> operators the audit was specified against do not exist.

---

## 9. Build and run matrix

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

| Configuration | Command | Tiers |
| --- | --- | --- |
| host only | `cmake -S csrc/tests -B build/host -DVLLM_ASCEND_TESTS_HOST_ONLY=ON` | `host/` |
| 950PR silicon | `cmake -S csrc/tests -B build/dev -G "Unix Makefiles" -DSOC_VERSION=Ascend950PR_9599` | `host/` + `device/` |
| 950PR camodel | `cmake -S csrc/tests -B build/sim -G "Unix Makefiles" -DSOC_VERSION=Ascend950PR_9599 -DRUN_MODE=sim` | `host/` + `sim/` |
| 310P | `cmake -S csrc/tests -B build/310p -DSOC_VERSION=Ascend310P3` | `host/` + `device_310p/` |

Running:

```bash
ctest --test-dir build/dev -L host                 # tier 1 only
ctest --test-dir build/dev -L device -LE benchmark # tier 3 correctness
ctest --test-dir build/dev -L benchmark            # tier 3 timings
```

Every test carries its tier as a ctest label, so the three are selectable
without knowing the binary names.

The work-in-progress Cube decode is off in all of the above. It is compiled
everywhere - it has to keep building clean under `-Werror` - but nothing
launches it unless asked:

```bash
VLLM_ASCEND_TQ_CUBE_WIP=1 ctest --test-dir build/sim -L simulator
cmake -S csrc/tests -B build/cube ... -DVLLM_ASCEND_TESTS_ENABLE_WIP_CUBE=ON
```

The variable wins over the CMake option in both directions. See §13.7 and
§13.8.

For the camodel:

```bash
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH
./build/sim/sim/test_sim_950pr_turboquant_decode
```

Use the **Makefiles generator**. With Ninja, `ascendc_library()`'s nested
ExternalProjects hit a CANN 9.1.0 bug in `extract_host_stub.py` and the build
dies with a bare Python `KeyError`; the sub-builds are pinned for that reason in
`turboquant/CMakeLists.txt`.

**Incremental kernel builds break intermittently.** Re-running the build after
touching anything the `ascendc_library()` sub-projects consume can fail with
`ld.lld: ... unknown file type` on a `turboquant_kernels.cpp.o` that compiled
cleanly moments earlier. It is not a source problem: delete the build directory
and configure again.

The camodel writes several hundred `core*.dump` and `profile_*.toml` files into
the **process working directory**, and nothing in `.gitignore` covers them. Run
from a scratch directory.

---

## 10. Bounds, and where each number came from

| Bound | Value | Provenance |
| --- | --- | --- |
| bin drift, device vs host packed cache | `<= 1` | The RMS scale is a tree reduction on the device and a serial sum on the host; one bin is what a coordinate on a decision boundary can cost. Measured drift is 0.000. |
| differing-channel fraction | `<= 2%` | Same. Measured is 0%. |
| scale relative error | `<= 1e-5` | One fp32 division of a reduced sum of squares; anything larger means the reduction disagreed, not that rounding did. |
| decode vs CPU reference | `cos > 0.999` (`0.9995` at the small shape), `relL2 < 5e-3` | Both sides keep an fp32 accumulator and round once at the store. |
| decode vs exact fp32 | `cos > 0.97`, `SNR >= 12 dB`, `relL2 < 0.30` | A floor. Measured: 16.70 dB on the camodel at the small shape and seed, 18.01 dB on the host at another seed — that spread is the width of the variation the bound has to survive. |
| footprint ratio | `> 3.0` | The layout arithmetic: 2048 B vs 544 B per token at `D = 256`, `kv = 2`, i.e. 3.76x. |
| bit-exact packed cache | exact | Constructed so no tie-break is possible; see §7.3. |

**Traffic model.** At `head_dim 256`, `kv_heads 2`, per token:

| Quantity | fp16 | TurboQuant 4-bit | Ratio |
| --- | --- | --- | --- |
| cache footprint per token | `2 * kv * D * 2` = 2048 B | `2 * kv * D/2` + `scale_slot * 4` = 512 + 32 = 544 B | **3.76x** |
| decode read per token per head | `2 * D * 2` = 1024 B | `2 * D/2` + `scale_slot * 4` = 256 + 32 = 288 B | **3.56x** |

The footprint ratio is exactly 3.76x at every length. **The decode-read ratio is
not**: the per-token figure is an asymptote, and the model also counts the query
and output rows and the flash-decoding partials, which are fixed in `S` and
dilute it at short contexts. Modelled end to end at 64 vector cores it comes out
around 3.3x at `S = 512`, 3.35x at 1024 and 3.45x at 2048, and only 2.23x at the
`S = 16` shape a camodel smoke run uses. The binary prints the exact figure for
whatever it ran; do not carry a number out of this table without the length it
belongs to.

The scale plane is 5.9% of the 4-bit cache. At `num_kv_heads = 2` a token's slot
is 8 fp32 lanes carrying 4 live ones — `round_up(2 * 2, 8)` — so **half** of that
plane is burst padding, 2.9% of the whole 4-bit cache, and that is what buys
every DMA on this path being an aligned `DataCopy`.

**The 4-bit fidelity ceiling is information-theoretic, not an implementation
gap.** Scalar Lloyd-Max delivers 20.22 dB per coordinate and *is* the scalar
optimum on a Gaussian, which the rotation makes the marginals. Adding scalar
entropy coding (~0.4 dB) and a practical lattice VQ gain (~0.75 dB) reaches only
~21.4 dB. Do not read a bound above as headroom someone could tune into
existence.

---

## 11. What none of this covers

- **Silicon.** As of this document every TurboQuant device case has executed on
  the arch35 camodel and none on a physical 950PR. The device tier is written
  *for* that run and skips until it happens; that is its status, not a defect.
- **Any measured fp16 comparison.** The V5 migration in §8 makes the leg
  *runnable*, and nothing more: no measured `fp16 us`, no measured speedup, no
  measured fp16 GB/s exists yet, and the V5 argument list has never been planned.
- **bf16.** Both kernels take an `AscendType` and every case passes `FP16`.
- **Prefill.** Every decode case is one query token. The write path is exercised
  at prefill *lengths*, but no attention case has `num_tokens > 1`.
- **Batch.** One sequence, one block table.
- **head_size other than 64, 128 and 256**, and block sizes other than 16 and 128.
- **E2M1 / E8M0 (MX-FP4) grids.** The host tier verifies the Lloyd-Max grid and
  the uniform mid-rise grid it replaced. No microscaling format is implemented in
  this tree, so there is nothing to sweep; adding one is a codec change first and
  a test change second.
- **A rotation ablation on real activations at the device level.** The sim tier's
  bounds cannot tell `Pi` from a bare absmax quantiser on i.i.d. Gaussian
  channels; only the host tier's golden case sees real activation structure, and
  it is a report rather than an assertion.
- **Performance regression thresholds.** The benchmarks report numbers and check
  that the kernels still produce their output. Nothing fails on a slowdown.

---

## 12. What has actually been executed

Recorded so the difference between "written" and "run" stays visible.

Environment: WSL Ubuntu 22.04 with CANN 9.2.0-beta.2 native for the camodel work,
and the official `quay.io/ascend/cann:9.1.0-950-ubuntu22.04-py3.10` container
(Ubuntu 22.04.5, CANN 9.1.0, g++ 11.4, cmake 3.22.1) for the silicon build.
The camodel reports `soc='Ascend950PR_9589'` with 64 vector cores and 40 GiB HBM.
**No physical part has been available in this environment.** One silicon result
exists and was reported from outside it: the first `bench_device_950pr_turboquant_ablation`
hung on a physical 950PR, which §7.6 traces to stage 4 and fixes. No silicon run
is recorded in the table below.

**Build**, clean under `-DVLLM_ASCEND_TESTS_WERROR=ON`, in all four
configurations: host-only, 310P, 950PR device, 950PR sim.

**Executed:**

| Binary | Where | Result |
| --- | --- | --- |
| `test_host_turboquant_fidelity` | host | 19/19 pass. Golden layer-3 report: attention context `cos 0.995593`, `20.54 dB`; layer output vs the fp16 dump `cos 0.998752`, `26.02 dB`. Synthetic 300-token context: `cos 0.995849`, `20.80 dB`. |
| `test_device_950pr_benchmark_harness` | host, no device | 17/17 pass. |
| `test_device_950pr_turboquant` | camodel, default gate | 7/7 **skip**, each naming the camodel library found in `/proc/self/maps`. The gate works. |
| `test_device_950pr_turboquant` | camodel, `ASCEND_TEST_ALLOW_SIMULATOR=1 ASCEND_TQ_BARE_METAL_CONTEXTS=16`, determinism case excluded | **6/6 pass.** Write path: max bin drift **0**, 0 of 8192 channels differ, worst scale relative error `2.478e-07`. Bit-exact case: **8192 packed bytes byte-identical** to the CPU reference, scale exactly 1.0. Decode: `cos 1.000000`, `73.97 dB`, `relL2 2.0e-4`. Layout: 32.0 KiB fp16 vs 8.5 KiB 4-bit, **3.76x**, scale plane 5.9%. Bounds: 996 poisoned rows survived, 28 live rows written, 0 live rows left unwritten. |
| `bench_device_950pr_turboquant` | camodel, `S=16`, one iteration | Traffic model and summary render; the fp16 leg skipped with the `361001` refusal quoted in full. That run predates the V5 migration and was taken before the device tier began refusing to run under a camodel at all. |
| `test_sim_950pr_turboquant_ablation` | camodel, B=1, 8/2 heads, head_size 256, block 64, S=256 (one split, 4 tiles), `ASCEND_TQ_ABLATION_HOST_FILL=1`, stages 0–4 in one process | **Pass**, exit 0, 726 s. Every stage returned -- 43 / 143 / 183 / 160 / 197 s -- with the workspace sentinel intact after each, and all 32 exception dumps at 0 bytes after exit. Before the §7.6 fix the same shape hung in stage 4, the AIC spinning on flag 6 and both vector subcores on flag 4. |
| `test_sim_950pr_turboquant_ablation` | camodel, same shape, cache written on device, stages 4 and 5 in one process | **Pass**, exit 0, 1,843 s. Write 1,400 s. Stage 4 180 s with the workspace sentinel intact; stage 5 230 s with every partial written, output 2048/2048 finite, `cos 0.989226` against the fp32 host attention; all 32 exception dumps at 0 bytes after exit. The first execution anywhere of the shipping kernel's `kFlagSlotFree` path. |

**Not executed:** `RepeatedDecodeLaunchesAreBitIdentical` (nine decode passes is
hours under the camodel); every case at `S = 512 / 1024 / 2048`; and every
timing. All wait on silicon, which is what the device tier is for.

`prof_device_950pr_msprof_trace` (§7.7) has **not executed at all**. It builds
clean under `-Werror` (host 4, sim 35 and npu 41 unique targets, 0 diagnostics;
the host and sim sets are unchanged and npu gained only this binary), but in the
`tq950-sim` build container it aborts before `main()` with
`basic_string::_S_construct null not valid`, the no-driver static-initialisation
failure described below. It still aborts with the camodel runtime and the
sim-mode kernel library shadowing their silicon counterparts on
`LD_LIBRARY_PATH`, so not even its argument parser has run.

Defects found by running the above and fixed in the shared harness:

- `BenchmarkRunner::WarmUp` took the reference checksum with no launch behind it
  at `ASCEND_BENCH_WARMUP=0`, so every case with a checksum failed at the end of
  the timed loop claiming "the launches are not all computing the same thing"
  when the baseline had simply never been computed. A case with a checksum now
  always gets at least one warmup launch.
- The binaries that link `libvllm_ascend_turboquant.so` in silicon mode abort
  during ctest's `PRE_TEST` GTest discovery on a host with no NPU driver, which
  ended the whole enumeration and made `ctest -N` list nothing for the entire
  project. They are now registered one ctest entry per binary.
- CANN's `ascendc_kernel_cmake` adds its `tools/simulator/${SOC_VERSION}/lib`
  directory as a PUBLIC link directory on every `ascendc_library()` target
  regardless of `RUN_MODE`, and it was reaching the `RUNPATH` of silicon
  binaries. Every device executable now links with `BUILD_WITH_INSTALL_RPATH` and
  an explicit rpath list. (`BUILD_RPATH_USE_LINK_PATH FALSE` was measured **not**
  to suppress it on CMake 3.22.)

---

## 13. Multi-mode TurboQuant and the Cube-native decode

Added after §12 and not folded into it, because none of what follows changes
the AIV-only 4-bit path: `turboquant_kernels.cpp` is untouched except for one
additive host launcher, and every number in §1–§12 still stands.

### 13.1 The three modes

`csrc/attention/turboquant/turboquant_mode.h` carries the enum, the packing
geometry and the codebooks. All three modes share one pipeline and differ in the
rate, the Cube operand type, and — since the INT4 change below — whether the
stored code is an index or a level:

| mode | bits | levels | group | bytes / vector (d=256) | 32B bursts | operand grid | Cube instruction | code is |
|---|---|---|---|---|---|---|---|---|
| `kv3fp4` | 3 | 8  | 8 elems / 3 B | 96  | 3 | fp4 e2m1   | `mad_mx` | Lloyd-Max index |
| `kv4fp8` | 4 | 16 | 2 elems / 1 B | 128 | 4 | fp8 e4m3fn | `mad` | **the level** |
| `kv5fp8` | 5 | 32 | 8 elems / 5 B | 160 | 5 | fp8 e4m3fn | `mad` | Lloyd-Max index |

**Which instruction runs is not a choice this project makes.** `AscendC::Mmad`
on arch35 inspects the `<DstT, Src0T, Src1T>` tuple and dispatches `mad_mx` for
`<float, fp4x2_e2m1_t, fp4x2_e2m1_t>` and plain `mad` for
`<float, fp8_e4m3fn_t, fp8_e4m3fn_t>`; the tuples that reach `mad_mx` on the fp8
side are the *microscaled* `mx_fp8_*` types, which this pipeline does not use
because it already carries a better-conditioned per-vector scale. The dispatch
is in `.../impl/basic_api/dav_3510/kernel_operator_mm_impl.h::MmadCal`.

**Mixed operand types do not exist here.** `fp4x2_e2m1 x fp8_e4m3fn` is rejected
by that same `static_assert` — measured, `build/perf/cubeprobe/err_mix_e2m1_e4m3.txt`
— so `kv3fp4` must put the query and the softmax row on the fp4 grid too.

### 13.2 The packed layout is two planes, not a bit stream

A code splits into a `low` digit and an `msb` digit, stored in separate
byte-aligned planes inside the vector slot:

| mode | low plane | msb plane | slot |
|---|---|---|---|
| `kv3fp4` | 2 bits, 4 digits/byte | 1 bit, 8/byte | 64 + 32 = 96 B |
| `kv4fp8` | 4 bits, 2 digits/byte | — | 128 + 0 = 128 B |
| `kv5fp8` | 4 bits, 2 digits/byte | 1 bit, 8/byte | 128 + 32 = 160 B |

No code straddles a byte, so an unpack never reassembles one from two loads and
the slot length is exact rather than rounded up — a 40-bit-per-8-codes bit stream
gives the same 160 bytes but needs a per-lane variable shift the vector core
cannot express.

Digit extraction on the **codebook** path (`kv3fp4`, `kv5fp8`) is one routine at
two radices, because the digit index within a byte is periodic in the lane with a
period (2 or 4) that divides the eight fp32 lanes of a 32-byte block. Seven
vector instructions per plane at any radix, with no mask register and no scalar
round trip. The reciprocal and weight tables are therefore a single 32-byte block
each rather than full-length buffers — three tables become 24 words.

**`kv4fp8` no longer shares that packer.** Its two digits per byte are the
coordinates `b` and `b + d/2` rather than `2b` and `2b + 1` — a plane split, not
an interleave — which is what makes both nibble planes contiguous runs and lets
the expand be the vector unit's signed-nibble converter plus one `DeInterleave`
over the whole run instead of a lane-indexed Gather. §13.9 is the whole of that
change.

### 13.3 The codebooks, and why there is a gain

A Lloyd-Max codebook is stated on N(0,1) and its centroids are not grid points
of e2m1 or e4m3fn. The stored table is `cast(g * c)` for a per-mode constant `g`
and the decoder reconstructs `(s / g) * lut[q]`, folding `1/g` into the
per-vector scale it already multiplies by — so the gain costs no instruction and
no UB. Without it the 3-bit table loses its innermost pair to zero (0.2451 sits
between e2m1's 0 and 0.5), which is two of its eight levels.

`kv4fp8` keeps the field and changes what it means. Its levels are the integers
`q - 7.5`, every one of which is already exact in e4m3fn, so there is no grid to
place a table on; `g` is instead `1 / step` for the optimal uniform quantizer,
and `(s / g) * (q - 7.5)` reconstructs the coordinate with the step riding the
per-vector scale exactly as the placement gain did.

`scripts/tq_multimode_calibration.py --emit-header` regenerates the tables;
`--report` measures what they deliver.

| mode | gain | D (fp32 Lloyd-Max) | D (as stored) | cost vs the fp32 ideal |
|---|---|---|---|---|
| `kv3fp4` | 2.7882 | 0.0345478 | 0.0384443 | 0.464 dB (the e2m1 cast) |
| `kv4fp8` | 2.9833 | 0.0095010 | 0.0115429 | 0.845 dB (uniform levels; the cast is free) |
| `kv5fp8` | 2.2065 | 0.0025047 | 0.0028687 | 0.589 dB (the e4m3fn cast) |

The two columns mean different things per row, which is the point of the last
one: for a codebook rate the loss is the cast onto the operand grid, and for
`kv4fp8` it is the whole price of giving up Lloyd-Max — 0.746 dB against the same
16 levels placed optimally and cast (0.0097204, the figure this table carried
before), and 0.845 dB against the fp32 ideal.

### 13.4 Fidelity: `kv5fp8` clears the gate

End-to-end attention, d=256, S=512, 8 heads, 4 seeds, fixture
`OUTLIER_KAPPA = 4.0` (post-rotation excess kurtosis −0.063, inside the band
measured on real Qwen3.5-2B). `kv4fp8`'s row is the affine INT4 form it now
ships as; §13.9 has the comparison against the codebook it replaced:

| mode | cos mean | cos min | SNR | gate `cos > 0.995` | memory vs fp16 |
|---|---|---|---|---|---|
| `kv3fp4` | 0.92813 | 0.91682 | 8.44 dB | **FAIL** by −0.078 | 5.33x |
| `kv4fp8` | 0.98752 | 0.98629 | 16.06 dB | **FAIL** by −0.0087 | 4.00x |
| `kv5fp8` | **0.99613** | **0.99539** | **21.13 dB** | **PASS** by +0.00039 | 3.20x |

The margin on the minimum is thin — four ten-thousandths — and worth stating as
such: `kv5fp8` clears the gate, it does not clear it comfortably.

Attribution for `kv5fp8`, same fixture and seeds: 5-bit storage with fp32 GEMMs
gives `cos 0.99677`, no KV quantisation with fp8 Cube GEMMs gives `0.99920`,
both together `0.99613`. The Cube stage costs about 0.0007 cosine, which
reproduces the earlier fp8-vs-fp32 finding: **the compute precision is not where
the error is, the storage rate is.**

Attribution for `kv3fp4` is the more interesting one, because the obvious
reading of its 0.93 is wrong: 3-bit storage with fp32 GEMMs gives `0.95966`,
while *no* KV quantisation with fp4 Cube GEMMs gives `0.97130`. The fp4 operand
grid — which the mode is forced onto because mixed fp4 x fp8 `Mmad` does not
exist — costs more than the 3-bit codebook does, and the two compound.
`kv3fp4` is scaffolded, not recommended.

### 13.5 The Cube stage, and the fractal contract

`turboquant_cube_mm.h` holds L1, L0A, L0B and L0C. Three structural decisions:

**The M dimension is the GQA group.** A decode step is a matrix-vector product
per query head and the Cube's M granularity is 16, so a per-head task would
waste fifteen of every sixteen MAC slots. Batching the query heads that share a
kv head into M makes it a real GEMM — 4x waste instead of 16x on Qwen3.5-2B —
so the task is `(token, kvHead, split)`, not `(token, head, split)`. That is the
one structural difference from the AIV-only decode.

**The operands reach L1 already in NZ, and there is no alternative.** The
L1 → L0 path is MTE1 `load2d`, which addresses L1 as an image of 16 × C0
fractals: `LoadData2DParamsV2::srcStride` is a *fractal* count, and fractal
`(m, k)` sits at `(k * srcStride + m) * 512` bytes. A fractal is not contiguous
in a row-major buffer, so **a flat ND L1 image cannot be sliced by `load2d` at
any parameterisation.** CANN's own matmul agrees: for `CubeFormat::ND` input it
calls `CopyND2NZForInt8` on the **GM → L1** copy
(`.../copy_cube_in/copy_tile_to_cube/copy_tile_to_cube_common.h`) rather than
loading ND out of L1. `CubeFormat` describes the global tensor, never L1.

Who applies the permutation is the part that is open, because
`DataCopy(L1, UB, Nd2NzParams)` does not exist for 1-byte operands on arch35:
CANN implements that UB-side transform as a strided `Adds`, which has no
int8/fp8 overload, and the mask it builds narrows for a 32-element block. Both
failures are compile-time. Routing the tile through GM instead compiles and
would **triple** the decode's GM traffic — 8 KB of fp8 written and read against
the 5 KB of codes it came from — which inverts the entire point of a compressed
cache.

On the **codebook** path the unpack's last step is a Gather, a Gather's offset
table is arbitrary, and the permutation is therefore a different constant rather
than an instruction; a strided `DataCopy` then places each band into L1. On the
**affine** path there is no Gather left to ride, and §13.9 is where the
permutation went instead.

**The accumulator stays in UB.** Flash decoding rescales by `exp(m_old - m_new)`
every tile and that multiply has no expression on the Cube. What the Cube takes
over is both O(S·D) products; what stays on the AIV is O(S) softmax and O(D)
accumulator maintenance.

`test_sim_950pr_cube_gemm` pins the fractal contract numerically, against a host
product of exactly-representable fp8 values, so the comparison is exact rather
than approximate. It exists because every part of that contract is undocumented
in the public headers; the authoritative statement is CANN's own matmul in
`.../detail/matmul/stage/split/load_to_l0{a,b}/load_to_l0{a,b}_load2dV2.h`, and
`TurboQuantCubeMm` is a transcription of it. Four facts that a reasonable
reading gets wrong:

- `mStep` counts 16-row fractals; `kStep` counts C0 elements, and C0 is **32**
  for every 8-bit type (`AuxGetC0Size` → `B8_C0SIZE`), not 64.
- `srcStride` and `dstStride` are **not optional**. Passing zero silently reads
  the wrong fractals.
- Fixpipe's `srcStride` is `m` rounded **up** to a multiple of 16, not `m/16` —
  a different convention from `LoadData`'s two lines earlier
  (`CeilAlign(mSize, BLOCK_CUBE)` in `copy_cube_out_utils.h`). With `m/16` the
  copy reads a sixteenth of the L0C.
- `ifTranspose` on the B operand means the **opposite** of what it reads as. B
  staged `[n, k]` loads with `ifTranspose` FALSE; B staged `[k, n]` loads with
  `ifTranspose` TRUE, and for an 8-bit operand in `mStep = 2` chunks, because
  the instruction only accepts an even `mStep` for `.b8`.

And one that is not a reading at all: **`PipeBarrier` orders a pipe against
itself and nothing else.** Without explicit `MTE1 → M` and `M → FIX` event
flags the `Mmad` issues while `LoadData` is still filling L0A/L0B and reads
whatever was there — which on a fresh buffer is zeros, so the symptom is a
partly-zero output rather than a fault. This is the same class of defect as the
`PipeBarrier<PIPE_ALL>` the combine kernel needs at the top of its reduction
loop (§ the four defects in the 2026-09-08 run).

### 13.6 The fp16 baseline is now a kernel, not an operator

> **Superseded by §7.5 / §13.7.** `turboquant_fp16_decode` is still built, still
> exported and still declared in `common/turboquant_launch.hpp`, but it **no
> longer has a leg in `bench_device_950pr_turboquant`**. That binary is an
> end-to-end audit against the native V5 operator now, and a Cube decode over an
> unquantised cache measures the *codec* — an ablation, not a competitive claim.
> Everything below is still an accurate description of the kernel; it is no
> longer a description of what the benchmark reports.

§8 records that `aclnnFusedInferAttentionScore` V1–V4 are withdrawn on an
Ascend950 — the planning call returns `361001` — and that the suite migrated to
V5 with V2 as a fallback. V5 is exported but has never been planned on hardware,
so a benchmark resting on it prints an empty column on the first machine that
runs it, which is what the previous `bench_device_950pr_turboquant` did.

The baseline is therefore built rather than borrowed.
`turboquant_fp16_decode_split` + `turboquant_plain_combine` are the same Cube
decode with the codec removed: same task decomposition, same GQA batching, same
64-row tile, same online softmax, same flash-decoding split count, same partial
layout. The only difference is where the Cube's operands come from — the fp16
leg copies them straight out of its cache with the AIC's own MTE2 and the ND→NZ
conversion `DataCopy` does for 2-byte types, while the quantised leg has the AIV
unpack them from 5-bit codes first.

That makes the ratio between them a measurement of the codec rather than of two
different kernels, and it makes the column impossible to leave empty: it depends
on no operator that can be withdrawn. It is also a *fair* baseline and not a
weak one — the fp16 leg does no vector work per tile at all, so it starts ahead
and the quantised leg has to win the difference back on bandwidth.

The combine is a separate kernel from the AIV path's for one reason: the fp16
accumulator was never rotated, so un-rotating it would be wrong.

### 13.7 What the benchmark now reports

`bench_device_950pr_turboquant` **is no longer a kernel comparison table.** It is
an end-to-end audit of the shipping path against the native CANN V5 operator, and
the full specification — workload, accounting, checks, columns and knobs — is
§7.5. This section records what changed and why, for anyone holding an older run.

#### What was removed, and why

| removed | why |
| --- | --- |
| `fp16_decode_s<S>` — `turboquant_fp16_decode`, the Cube decode with the codec removed | It measured the **codec** against a physical fp16 cache nobody would deploy. That is an ablation, not a competitive claim, and a reader seeing `vs fp16` beside `vs fia` in one table has no way to know only the second is a product statement. |
| `tq4_write_s<S>` / `tq4_decode_s<S>` beside `kv4fp8_*` | The AIV-vs-Cube ratio is a kernel comparison. Since §13.9 it also moved the *quantiser* and the *compute unit* together — Lloyd-Max on one side, affine INT4 on the other — so it had stopped isolating anything. The path is now **chosen** per model by the GQA group (§7.5) and only the chosen one is timed. |
| `pf_fp16_write` / `pf_fp16_pipeline` | Same reason: a pipeline built on an unquantised write is not what the comparison is against. `aclnnScatterPaKvCache` survives as `pf_v5_ingest`, which is the **native baseline's own** cache write and therefore a real cost on the other side of the ledger. |
| the 270-shape prefill sweep (`S`×`B`×`D`×`H_Q`×`H_KV`) | A dense cartesian product of head counts produces hundreds of shapes nobody deploys and buries the ones that matter. Replaced by 27 serving ranks across three flagship families at TP=8. |
| the standalone traffic-model tables | Folded into the two summary tables as the `KV Cache Memory Saved` and `Memory Compression Ratio` columns, and into the CSVs as raw byte counts. |

#### What was added

- **A strict E2E definition**, `T_rot_q + T_attn_core + T_rot_o`, measured per
  component *and* as one composite ACL-event region, with the residual between
  them printed. Nothing about the rotated cache is outside a timed region any
  more except the one narrowing named in §7.5 and repeated in the file header.
- **The folded / unfolded distinction as a first-class column.** Qwen3.5's
  `attn_output_gate` is the reason `T_rot_o` exists as a measurement; DeepSeek-V4
  and GLM-5.2 fold it into `W_o` and report an exact `0.00 us`.
- **Chunked prefill**, `C = min(S, 2048)`, without which the 262K and 1M rows are
  hours of `O(S²)` attention per iteration and the table is empty exactly where
  the memory argument is strongest.
- **Adaptive iteration budgets**, 5/20 below 262K and 1/3 at or above it, which is
  why the binary now prints **two** generic per-case tables — one per budget, each
  with a header that describes its own rows. `BenchmarkRunner::set_options` was
  added for this and refuses to run after a case has been recorded.
- **Two device checks before any timing**: the rotated-basis identity
  (`cos > 0.999`) and the decode tie-point against the native operator
  (`cos > 0.90`). The second exists to catch a benchmark timing a kernel that is
  reading an empty cache, which no checksum can detect.
- **`H_KV = 1` everywhere.** All three families are MQA-shaped at TP=8. That makes
  the scale plane's burst padding the dominant term in the compression ratio —
  24 of every 32 scale-plane bytes are padding — and it is why the ratios land
  near 3.2× and 3.6× rather than at 4.

#### The operator the specification asked for does not exist

The audit was specified against `aclnnPromptFlashAttentionV5` and
`aclnnFusionIncrementalAttention`. CANN 9.2.0-beta.2 ships PFA up to **V3**, IFA
up to **V4**, and no symbol named `aclnnFusionIncrementalAttention` at all.
`aclnnFusedInferAttentionScoreV5` is the unified interface that replaces both
families on an Ascend950 (§8), and it is what both roles are planned against —
`sparseMode 3` with the compressed causal mask for the prompt role, `sparseMode 0`
with a block table for the incremental one. Guessing a prototype for a symbol that
is not in a header is undefined behaviour at launch rather than a planning
refusal, and this suite does not do it. See §7.5 and §8.

#### Still true, and still the headline

**kv3fp4 and kv5fp8 have no leg here.** Both are still built, still dispatched and
still covered by the sim tier. Neither is on the 4-bit path this binary measures.
`kv5fp8`'s fidelity — the one that clears `cos > 0.995` — is measured by
`scripts/tq_multimode_calibration.py` and recorded in §13.3.

`test_device_950pr_turboquant`, the bare-metal correctness binary, is unchanged
and needs no switch of its own: it drives `turboquant_kernels.cpp` only and calls
nothing in `turboquant_mm_kernels.cpp`. Keep it that way.

`VLLM_ASCEND_TQ_CUBE_WIP` no longer gates anything in this binary. The Cube legs
it used to hold shut are the *selected path* for one of the three families now,
and a family whose group does not fill the Cube fractal never launches a Cube
kernel in the first place. `ASCEND_BENCH_TQ_AUDIT_PATH=aiv` is what to reach for
if a future Cube defect faults: it moves every family onto the vector path before
any launch. `-DVLLM_ASCEND_TESTS_ENABLE_WIP_CUBE=ON` remains sim-tier only.

> **NO NUMBER THIS BINARY PRINTS HAS EVER BEEN TAKEN ON SILICON**, and the banner
> says so on every run. No Ascend 950PR part has been available to this project;
> the kernels are verified on the arch35 camodel (§13.8, §13.10), and the FIA V5
> argument list has never been through a planning call on hardware (§8). The
> tables are a measurement *apparatus*, complete and self-describing; the
> measurements do not exist yet.

### 13.8 What has actually been executed, and what has not

| binary | where | result |
|---|---|---|
| `scripts/tq_multimode_calibration.py --report` | host, numpy + real `torch.float8_e4m3fn` casts | **Run.** The §13.3 and §13.4 tables are its output. |
| `test_sim_950pr_cube_gemm` | arch35 camodel | **Run.** Both B forms reproduce the host product **exactly**, `max|err| = 0`: `ContextGemmBStagedKn` (`m=16 k=64 n=256`, B staged `[k,n]`, mStep = 2 chunks) and `ScoreGemmBStagedNk` (B staged `[n,k]`, one load), plus a repeated-launch case. See the correction below. |
| `test_sim_950pr_turboquant_multimode` | arch35 camodel | **Run.** `kv4fp8` completes and clears its `cos > 0.90` bound; see the table below and §13.9. `kv5fp8` and `kv3fp4` stay behind `VLLM_ASCEND_TQ_CUBE_WIP=1`. |
| `bench_device_950pr_turboquant` | — | **Compiled for `RUN_MODE=npu`, never executed.** Its three Cube legs are additionally gated off by default (§13.7), so even the first run on silicon reports the 4-bit path alone until the defects below are closed. No Ascend 950PR silicon has been available to this project at any point; the camodel is explicitly barred from multi-iteration timing. Every number in §13.7's table is therefore a column the binary will fill on the first machine that has the part, and none of them exist yet. |

**CLOSED, and the diagnosis above was wrong.** This section previously said the
score GEMM's `[n, k]` B operand does not reproduce the host product and
prescribed staging K transposed with the unpack re-tiled. That is a large
redesign for a defect that does not exist, and it misdirected a full session of
work. **Both B forms were always correct.** `test_sim_950pr_cube_gemm` now
reports `max|err| = 0` for the score GEMM (`[n,k]`, one load) and the context
GEMM (`[k,n]`, mStep = 2 chunks), plus a repeated-launch case. A sweep of nine
alternative parameterisations of `LoadBFromKn` produced an all-zero L0B for every
one of them, and CANN's own `load_to_l0b_load2dV2.h` confirms the shipping
parameters exactly (`HW_M0 = ALIGN_NUM = BLOCK_CUBE = 16`, and
`CeilAlign(a,b) = Ceil(a,b)*b`, so `dstAddrStride` is `CeilAlign(n,16)*32`).

**What was actually wrong: no V -> MTE3 synchronisation on any UB -> L1 staging
copy.** Every Cube operand is produced by vector work -- the codec's unpack, the
query cast, the softmax's probability row -- and then moved UB -> L1 by
`DataCopy`, which is MTE3. `PipeBarrier<PIPE_V>` orders V against V and nothing
else, exactly as `PipeBarrier<PIPE_MTE1>` fails to order MTE1 against M (§13.5).
Without an explicit `HardEvent::V_MTE3` set/wait the DMA read each buffer before
the vector writes landed and staged **zeros**, so `Mmad` multiplied an empty
operand and Fixpiped an all-zero product. Nothing faulted, and an all-zero
product is indistinguishable from a wrong fractal layout -- which is how this
section came to blame the wrong thing.

`SyncVectorToMte3()` now guards all four hand-offs: the query in `PrepareTask`,
each unpacked K **and V** band in `UnpackToL1`, and the probability row in both
kernels' `Softmax`. The same idiom is in CANN's `matmul_client.h`.

**Result, measured on the arch35 camodel at S = 64, head_dim 256, block 128:**

| metric | before | after |
|---|---|---|
| `cos` vs the fp32 host reference | `nan` | **0.982351** |
| `vec_err_idata_inf_nan` | 310 | **0** |
| hard faults / `core*.excp_log.dump` | 0 / 0 B | 0 / 0 B |

`scripts/tq_multimode_calibration.py` measures 0.98785 for this rate at S = 512,
so the device is landing where the host model says it should. The `kv4fp8` leg
of `test_sim_950pr_turboquant_multimode` passes its `cos > 0.90` structural
bound.

**Still unmeasured:** every context length above 64, `kv3fp4` through the
corrected path, and every timing number -- no Ascend 950PR silicon has been
available to this project at any point.

Multi-tile addressing was **deliberately not** taken to the camodel. An S=512
pass was started and halted: it is most of an hour per rate per shape, and the
decision recorded here is that multi-tile and deep-context validation happens on
silicon, through the matrix in §13.10, rather than by spending simulator hours on
it. That is a choice about where to spend verification, not a claim that the tile
loop is known good -- at S=64 the decode runs a single 64-row tile, so the tile
*loop* has been executed exactly once per launch and never iterated.

Four device defects were found by the first camodel run of the multi-mode decode
and fixed, all of the kind the source reads as correct on:

1. **A vector store at a 16-byte offset.** Zeroing only the scale slot's burst
   padding starts the store at `2 * num_kv_heads` lanes — 16 bytes at
   `num_kv_heads = 2` — and a vector operand base must be 32-byte aligned. The
   part raises `vec_err_instr_misalign` and the camodel asserts. Zero the whole
   slot from offset zero, then gather the live lanes over it, which is what the
   shipping 4-bit kernel does and why.
2. **The AIC running the AIV's half of a MIX kernel.** Both cores execute the
   function; the accumulator init, the task setup and the partial writeback are
   vector work and were unguarded, so the cube core issued vector stores and a
   GM scatter it does not own. Reported as `su_ccu_mpu_err`, which names nothing.
3. **Dividing a whole 32-byte block by a block with one live lane.** The
   probability row's operand scale is written to lane 0 only; its other seven
   lanes are the zeros the state buffer was cleared to. `vec_err_div_by_zero`
   followed by thirty-two `vec_err_idata_inf_nan`, all from one missing `Brcb`.
4. **The missing MTE1 → M → FIX synchronisation** described in §13.5. This is
   the one that produced a plausible partly-zero answer rather than a fault, and
   the one the exact-match test caught.

A fifth was open and is now **fixed**. `TurboQuantCubeMm::Init` ran on both
halves of the MIX kernel and allocated L0A, L0B and L0C unconditionally; those
pools do not exist on a vector core, so the tensors it handed back there were
nonsense addresses, and `ldst_addr: 2` is what one of them looks like when
something touches it. The allocation is now split: L1 (`A1`, `B1`) on both
cores, because the AIV stages into it and the AIC reads it, and L0A / L0B / L0C
under `ASCEND_IS_AIC`. Nothing on the vector side reads them - `LoadA`,
`LoadBFrom*` and `Compute` all run under `ASCEND_IS_AIC` at their call sites -
so the AIV simply does not allocate them.

`TurboQuantFp16DecodeSplit::Init` had the same defect and got the same fix. It
does not share `TurboQuantCubeMm` - it carries its own L1/L0 set and its own
`Gemm`, because its operands are `half` and come from the AIC's own MTE2 rather
than from an unpack - so the two had to be corrected separately. Its `aP1_` is
the one L1 buffer genuinely written by the vector core, in `Softmax`, which is
why all three L1 buffers stay allocated on both halves there too.

**A sixth, which the fifth was hiding: the MIX block index.** With the L0 fault
gone the decode stopped faulting and started *hanging* - one core spinning in a
flag wait, every other camodel worker parked on `futex_wait`, and
`core*.excp_log.dump` empty on all of them, because nothing had gone wrong with
any address.

`AscendC::GetBlockIdx()` does not mean the same thing on the two halves of a MIX
kernel. From `kernel_operator_sys_var_impl.h` on arch35:

```
AIV   get_block_idx() * get_subblockdim() + get_subblockid()
AIC   get_block_idx()
```

A 950PR reports `num_vec_core=2`, so the AIC of block `c` sees `c` while its own
two vector subcores see `2c` and `2c + 1`. Both decode kernels opened their task
loop with

```cpp
uint32_t start = static_cast<uint32_t>(AscendC::GetBlockIdx()) * tasksPerCore;
```

so an AIC and the AIVs physically paired with it walked **different task lists,
of different lengths**. Every `ProcessTile` carries a four-phase handshake, so
the counts diverged on the first tile: the AIC blocked in
`CrossCoreWaitFlag(kFlagOperandsReady)` for operands its own AIVs were never
going to stage, and those AIVs blocked in
`CrossCoreWaitFlag(kFlagProductReady)` for a product no AIC would compute.

`MixBlockIdx()` - `GetBlockIdx() / GetSubBlockNum()`, which is `c` on the AIC
(divisor 1) and `c` on both AIVs (divisor 2) - collapses both halves onto the
block index, which is what the host's grid is expressed in anyway:
`PlanCubeDecode` sizes `split_block_dim` in tasks per *block*, and `<<<blockDim>>>`
on a MIX kernel counts AIC blocks. Applied to `TurboQuantCubeDecodeSplit` and
`TurboQuantFp16DecodeSplit`.

Two things worth not rediscovering. **The flags were never the problem**, though
they look like it: `CrossCoreSetFlag<0x2, PIPE_MTE3>` appears outside any
`ASCEND_IS_*` guard in `ProcessTile` and is nonetheless AIV-only, because
`NotifyEventImpl` self-guards by pipe class - on arch35 `IsSplitVectorPipe` is
`{PIPE_S, PIPE_V, PIPE_MTE2, PIPE_MTE3}` and `IsSplitCubePipe` is
`{PIPE_S, PIPE_MTE1, PIPE_MTE2, PIPE_FIX, PIPE_M}`. And **this is why
`test_sim_950pr_cube_gemm` passes while the decode hung**: the probe is
single-shot with no block-indexed task loop, so its handshake cannot
desynchronise. It is also the reason both vector subcores are left running the
same task rather than half of it each - that is the arrangement the probe
proves, and the AIV -> AIC flag is satisfied only when every subcore of the pair
has set it.

> `TurboQuantCubeDecodeSplit::ProcessTile` no longer exists: section 13.11
> replaced it with `PipelineAiv` / `PipelineAic`. Nothing above changes — the
> per-tile handshake is still what desynchronises if the two halves of a MIX
> block walk different task lists, and `MixBlockIdx()` is still what stops them.
> `TurboQuantFp16DecodeSplit::ProcessTile`, the other kernel the fix was applied
> to, is unchanged.

**A seventh, in the test rather than the kernel: `block_size` must be a multiple
of `kCubeTileRows`.** `kCubeTileRows` is 64 and `CopyInTile` issues one
unconditional 64-row `DataCopy` per tile, so a smaller block size makes every
tile read past the end of the block it was handed - into the next blocks of the
pool, and past the end of the cache and scale-plane allocations entirely when
the block is the last one. That garbage reaches the per-row scale lanes, which
are multiplied into the score row *before* the invalid columns are masked to
`-inf`, and the run fills with `vec_err_idata_inf_nan` on the softmax's
`RV_VMAX`/`VSUB`/`VEXP`. It raises no `ldst_addr`, because the over-read stays
inside the allocation's page slack - which is exactly what makes it read as a
numerical defect rather than an addressing one.

`test_sim_950pr_turboquant_multimode` was running `block_size = 16`, inherited
from the tier's habit of picking the smallest shape that still exercises the
feature. It is now 128, the production value. At `S <= 128` that costs nothing:
the tile loop still runs a single 64-row tile, because `rows` is clamped to the
context length.

### 13.9 INT4 storage, and where the NZ permutation went

> **Superseded by 13.24 rule 1 (silicon, 2026-09-16).** The group-major 32 B GM -> UB tile
> read described below throttles the HBM controller to 44.1 GB/s. It is not an acceptable
> trade. The fused decode reads GM in wide bursts. Since 13.28 the kv4fp8 cache is stored
> NZ-tiled, so the permutation happens once, at cache-write time.

Two changes, and they are one change: the second is forced by the first.

**Milestone 1 — the stored code is the level, not an index.** `kv4fp8` used to
store a 4-bit index into a 16-entry Lloyd-Max table and expand it with a UB
`Gather` through that table. It now stores a symmetric INT4 code and the expand
is `q - 7.5`, a single `Adds`. The sixteen levels `±0.5 … ±7.5` are all exact in
`fp8_e4m3fn`, so the cast onto the operand grid is lossless and the quantizer
step folds into `kGain`, which the decode already divides the per-vector scale
by — the reconstruction path is unchanged, only the constant differs. The
encoder is unchanged too: its bin is still `sum_i [u > t_i]`, against uniform
boundaries instead of Lloyd-Max ones.

What that costs, from `scripts/tq_multimode_calibration.py` (d=256, 8 heads,
4 seeds):

| | codebook SNR | D | cos mean / min, S=64 | cos mean / min, S=512 |
|---|---|---|---|---|
| Lloyd-Max 16 + e4m3fn cast | 20.12 dB | 0.0097204 | 0.99083 / 0.98944 | 0.98785 / 0.98629 |
| affine INT4, step 0.3352006 | 19.38 dB | 0.0115429 | 0.98831 / 0.98638 | 0.98752 / 0.98629 |

0.746 dB of codebook SNR; 0.003 of attention cosine at S=64 and, on the minimum,
nothing at all at S=512 — the two agree to five decimals there. Well clear of the
`cos > 0.90` structural bound, and §13.4's verdict is unchanged: `kv4fp8` misses
the `cos > 0.995` gate by 0.0087 either way, and `kv5fp8` is still the rate that
clears it — still a codebook rate.

**The 0.75 dB per-coordinate gap does not survive to the output, and this
document does not claim to know why.** It is consistent with an earlier finding
on this project that per-vector MSE predicts attention-output error poorly, but
that is an observation about two rates at one shape, not a mechanism. Treat it as
a measured result for `kv4fp8` at d=256, not as a reason to expect uniform levels
to be free at another rate — `kv5fp8`'s codebook has not been re-measured against
a uniform alternative at all.

**Milestone 2 — the vector path no longer touches NZ.** Removing the codebook
`Gather` also removes the *other* one. The old expand needed a first `Gather` to
bring each channel's packed byte to its own lane, and that table, being
arbitrary, was where the NZ permutation lived for free (§13.5). With no Gather
there is nothing to fold it into, so the permutation has to go somewhere — and
the flat L1 staging the milestone asked for only works if it goes **upstream**,
not downstream:

1. **The packing is plane-split.** Byte `b` of a slot holds coordinates `b` and
   `b + d/2`, not `2b` and `2b + 1`. An interleaved pair needs a stride-2 scatter
   to separate, and `DataCopy` has no element-stride-2 move — `blkStride`
   counts 32-byte blocks — which is exactly why the old layout needed a Gather.
   Split by plane, both nibble planes are contiguous runs. The split was first
   `>> 4` and `(x << 28) >> 28` on a `uint32` view of the run; it is now the
   `int4x2` converter plus a vector `DeInterleave` — see the update at the end
   of this section.
2. **`CopyInTile` reads the tile group-major.** The tile lands in UB as
   `[group][row][32]` instead of `[row][group][32]`, one `DataCopy` per 32-byte
   column-group across all 64 rows. Group `g` carries coordinates `[32g, 32g+32)`
   in its low nibbles and `[d/2 + 32g, +32)` in its high ones, and both of those
   are whole C0 columns of the NZ operand — so a group's expansion lands, element
   for element and in order, on exactly one NZ fractal column. The row/column
   interleave is done here, by an MTE2 read that was strided anyway.
3. **`UnpackToL1` is then flat.** Two `DataCopy` calls per chunk, each a single
   unbroken burst, at fixed offsets: the low plane to NZ columns `[g, g+n)` and
   the high plane to `[d/64 + g, +n)`. `kUnpackRows`, `dstStride =
   kCubeTileRows - kUnpackRows` and the strided `nzParams` are gone from this
   path.

**What was asked for and is not possible: `MTE1 Load2D` slicing fractals from a
flat ND L1 buffer.** See §13.5 — `load2d` addresses L1 as a fractal image and has
no ND source mode, on this part or in CANN's own matmul. The achievable form is
the one above: L1 is still NZ, but nothing in the vector path reshapes anything,
and the UB → L1 copy is sequential. CANN's own unwired
`DataCopyUB2L1ND2NZImplV2` (`dav_3510/kernel_operator_data_copy_impl.h`,
commented "Ascend950 MTE3 UB→L1") reaches the same conclusion from the other
direction: it does the interleave with a loop of plain `CopyUbufToCbuf`, i.e.
with strided DMA, not with MTE1.

**UB, which is what bounds the chunk size.** The affine codec carries two
half-length fp32 buffers (it works on packed bytes, of which there are half as
many as channels) against the codebook path's four full-length ones, and needs
no constant-table image at all — `ConstTableWords` is 0, `Init` allocates no
constant buffer, and `ModeTables` emits a single zero block only so the launch
has a bindable pointer. At d=256:

| | work buffer | table image | operand | total |
|---|---|---|---|---|
| codebook, 8-row band | 42 272 B | 17 568 B | 2 048 B | **60.4 KB** |
| affine, `kUnpackGroups = 2` | 44 320 B | 0 | 8 192 B | **51.3 KB** |

so the change is a 9 KB UB saving even though the chunk is four times the band
and the operand buffer four times the size — the table image is what pays for it.
`kUnpackGroups = 4` would halve the call count again for 32 KB more, which the
headroom did not justify.

**The codebook modes are untouched.** `kv3fp4` and `kv5fp8` keep the interleaved
packing, the two Gathers and the strided band staging; `CopyInTile` and
`UnpackToL1` branch on `TurboQuantModeTraits::kIsAffine`.

**Update 2026-09-14 — signed `int4x2` nibbles, and an eight-instruction expand.**
The ablation ladder put `stage1_unpack` at about 30% of the decode, so the
shift chain above was replaced. No DMA engine can do this instead: MTE1, MTE2 and
MTE3 only move bytes, and CANN's 4-bit DMA overloads re-type `fp4x2` as a B8 or
B16 container. What arch35 does have is on the vector unit:

- `Cast<half, int4b_t>` — a `UNPK4_B8` load plus `vcvt_s42f16`, a **signed**
  nibble expand — and its inverse `Cast<int4b_t, half>`;
- a public `DeInterleave` / `Interleave` for half, which is the element-stride-2
  move the plane split was avoiding.

So a nibble now stores `s = q − 8` in two's complement instead of the unsigned
`q` under a −128 byte bias, and the expand is `s + 0.5`. The levels are the same
sixteen, so `kGain`, the scale plane and everything after the unpack are
unchanged. The packing is still plane-split, so `CopyInTile`'s group-major read
and `UnpackToL1`'s flat copies are unchanged too.

| | vector instructions per chunk | bytes written per packed byte | UB |
|---|---|---|---|
| shift chain (13.9 above) | 13 | 44 | `expand_` + `msb_` |
| `int4x2` + `DeInterleave` | 8 | 26 | the same two buffers |

The encoder's `PackAffinePlane` is four instructions: `Adds(−8)`, a cast to half,
`Interleave`, and the cast to `int4b_t`. Pack and unpack go through the same
converter pair, so which nibble of a byte is lane 0 — the headers do not say —
cannot disagree between them. No host code reads `kv4fp8` bytes; the `−128`
references elsewhere in `csrc/tests` are the AIV-only `tq4` cache, which is a
separate codec and still unsigned.

**A cache written before this change decodes wrong, silently:** the byte layout is
the same, only the nibble meaning changed. There is no persisted `kv4fp8` cache
to migrate, but a device run that reuses a pre-change cache image will read
garbage without faulting.

**A trap found while doing this:** an unsupported AscendC `Cast` pair compiles
clean and emits no instruction, because `ASCENDC_ASSERT` is empty in a device
build (`impl/basic_api/kernel_log.h`). Probes for `fp32 ← int4` and
`fp8 ← half` both compiled. Check a new pair against the tuple table in
`dav_3510/kernel_operator_vec_vconv_impl.h::CastImpl`, and the object for a
`CastIntrinsicsImplVF<dst, src>` symbol, never against the compiler alone.

### 13.10 The device validation matrix

`test_device_950pr_turboquant_multimode` compiles the **sim tier's source** —
deliberately, so the cos bound the simulator enforces and the one silicon
enforces cannot drift apart. What differs is the default sweep, and only that:

| | simulator | silicon |
|---|---|---|
| batch | 1 | **1, 8** |
| context | 16 | **64, 512, 1024, 2048** |
| query heads | 4 over 2 kv | **8 over 2 kv** (`shapes950::kNumHeads`) |
| head dim | 256 | 256 |
| block size | 128 | 128 |
| gate | `cos > 0.90` per shape per rate | same |

A camodel pass is cycle-level — seven minutes at the single-tile shape, most of
an hour at S=512 — so there the sweep is **one** shape and the loop runs exactly
once. The `#ifdef VLLM_ASCEND_TEST_TIER_DEVICE` that selects between the two
lists is set only on the device target; a string literal cannot be tested with
`#ifdef`, which is why that target carries a second definition alongside
`VLLM_ASCEND_TEST_TIER="device"`.

`ASCEND_TQ_SIM_CONTEXT` and `ASCEND_TQ_SIM_BATCH` override either list on either
tier, as a comma-separated set of positive integers. That is how a camodel run
covers a second shape — one process per shape — and how a silicon run narrows to
the one that regressed. A partly parseable list keeps the entries that parsed
rather than silently substituting the default.

**What the batch axis buys, and why the cosine is reported twice.** With
`batch > 1` each sequence gets its own run of blocks out of one shuffled pool, so
no two sequences share a block and the block table is a scatter in both axes.
The check then reports the cosine over the whole batch **and** the worst
sequence, and gates on the worst: a batched figure alone would hide one wrong
sequence in seven right ones, which is exactly the failure a batch axis
introduces. The kernel already supported it — `PlanCubeDecode` takes the token
count and the task loop is keyed on `(token, kvHead, split)` — so this is test
coverage catching up with the kernel, not a kernel change.

**`test_device_950pr_turboquant`, the bare-metal AIV-only binary, gained `S=64`**
for the same reason the benchmark did: it is the one context the Cube path's
numerics have been checked at, so the two tiers now overlap at one shape instead
of meeting nowhere. It is also the only entry in that file smaller than
`block_size`, so it is the one case there where the sequence occupies a partial
paged block. Its own gate stays at `cos > 0.999` — it compares against the CPU
reference running the identical algorithm, not against fp32 attention, so 0.90
would be far too loose there.

**None of this has run.** No Ascend 950PR silicon has been available to this
project at any point. The matrix above is what the first machine with the part
will execute; what is verified today is the simulator column, at `B=1` and
`S in {16, 64}`.

### 13.11 The decode's AIV/AIC pipeline is double-buffered

> **See 13.24 rule 3.** At B=1 and S <= 4096 the split/combine chain this section tunes is
> replaced by one fused launch. The legacy split keeps this pipeline as the A/B reference.

`TurboQuantCubeDecodeSplit` used to run its tiles in lock-step: the Cube sat
idle through the MTE2 read, the INT4 unpack and the MTE3 staging of every tile,
twice per tile, because the flag it was waiting on was the one the AIV posted at
the end of that work. The staging is now a **tile ahead** of the GEMMs that
consume it, with `L1` operand B partitioned into `kSlots = 2` ping-pong slots of
K and V each.

**What moved, and what could not.** Flash decoding's online softmax is a
loop-carried dependency: the probability row of tile `i` comes out of the score
GEMM of tile `i` and goes straight back in as the A operand of tile `i`'s
context GEMM, and the running max it advances is what tile `i + 1` rescales
against. So `Softmax` and `Accumulate` stay on the critical path, and **the Cube
still stalls on the softmax between its two GEMMs** — closing that gap needs a
second task in flight, not a deeper buffer. What moved is everything that
depends only on the block table: `CopyInTile`, both `UnpackToL1` calls, and the
flat MTE3 into L1. That is the bulk of the AIV's per-tile work.

**The prefetch sits after the softmax, and that position is load-bearing twice
over.** It has to be after, or `CopyInTile` overwrites `scaleTileBuf_` while
`Softmax` still needs this tile's K and V scale lanes out of it — a silent wrong
answer, not a fault, and the reason nothing in UB needed a second slot. It has
to be before the wait on the context product, or it has no Cube work to overlap.

**Two orderings that used to come free now have flags.** `kFlagProbsReady`
(AIV → AIC) replaces what the V-unpack's position between the two GEMMs used to
guarantee; without it the Cube multiplies the *previous* tile's probability row.
`kFlagSlotFree` (AIC → AIV, one id per slot) is the reverse edge that lock-step
did not need, because each side's wait for the other was also what kept it off
the buffer the other was still reading. The full map is in
`turboquant_cube_mm.h`; it spends seven of the eight ids the four-bit mask
leaves after CANN's own `SyncAll` takes 11..14.

**Every flag set has exactly one wait**, which is a correctness requirement and
not tidiness: these are hardware counters the kernel does not clear on exit and a
decode runs many tasks per core, so a surplus set would satisfy a *later* task's
wait before its operands were staged. The two conditioned edges —
`tile + 1 >= kSlots` on the AIV's free-wait and `tile + kSlots < numTiles` on the
AIC's free-post — are what make the pair balance at every tile count, including
`numTiles = 1` where neither fires.

Both halves need the tile count up front for those predicates, so both call
`CountTiles` and step the walk with `NextTile` rather than discovering the end by
running off it. The enumeration is a pure function of the block table and the
context length — scalar GM reads either core can make — so the two cannot
diverge.

**What this is verified against.** The protocol arithmetic (counter balance,
deadlock-freedom, and that tile `t`'s GEMMs read tile `t`'s operands) and the
`CountTiles`/`NextTile` equivalence to the old nested loop were both swept in
host-side models, the latter over ~37k split geometries including block-table
holes and ragged final blocks. Neither is a substitute for the numeric gate:
that is `test_sim_950pr_turboquant_multimode` on the CAModel, and the multi-tile
shapes need `ASCEND_TQ_SIM_CONTEXT=128` or higher, since the simulator default
of `S = 16` runs a single tile and so exercises the prologue and the epilogue but
never the steady state.

**Which shapes actually reach the credit edge.** `kFlagSlotFree` is posted only
when `tile + kSlots < numTiles`, so a task carrying two tiles or fewer never
exercises the reverse edge at all — and `num_splits` rises with the block count
in `PlanCubeDecode`, which cancels the growth in context length almost exactly.
At batch 1 the tiles-per-task figure is **2 for every context length from 128
through 1024** (identical at `aiv_num` 32, 48 and 64, so this is not a
platform-count artefact). `S = 128` therefore covers the prologue, one prefetch
and `kFlagProbsReady`, but leaves the free-credit path — the likeliest place for
a hang or a leaked counter — untouched. Firing it needs **batch 1 at S >= 1536**,
or **batch 8 at S >= 512**, which is why the device matrix rather than the
simulator default is where that edge gets covered.

### 13.12 Pairing the softmax across heads: measured, and rejected

Every dependent pair of vector ops in the online softmax carries a
`PipeBarrier<PIPE_V>`, and a barrier is a full vector-pipe flush. One head's
softmax is a chain of 17 of them. The heads of a GQA group are independent, so
the obvious move is to issue two heads against each flush -- 34 flushes for two
heads down to 17 -- and the obvious move does not pay.

**It was built and measured.** Single tile (`ASCEND_TQ_SIM_CONTEXT=64`, batch 1,
`kv4fp8`) on the `Ascend950PR_9599` CAModel:

| | cos vs fp32 host | total tick |
|---|---|---|
| per-head (shipping) | 0.986033 | 68584, 68579 |
| paired, 17 barriers shared | 0.986033 | 69318, 69336 |

so about **+1.1%, in the wrong direction**, against a run-to-run noise floor of
~18 ticks measured by running one unchanged binary twice. (The tick count is
NOT deterministic across runs of the same binary -- worth knowing before reading
any single number here as exact.)

**Why counting barriers was the wrong metric.** Classifying each shared barrier
by the width of the op it guards:

```
barriers guarding a 64-element op : 7
barriers guarding a 1-element op  : 10
```

Ten of the seventeen guard single-lane work -- `Max`, `Sub`, `Exp`, `Add`,
`Adds`, `Duplicate` and `Div` at `count = 1`, the running-max and scale
bookkeeping. A one-element vector op retires almost immediately, so its barrier
is nearly free and sharing it buys nothing. Against that, pairing costs 22
scalar branches and about 14 extra `LocalTensor` address computations per pair,
every one of them real. Only 7 barriers guard 64-element work and only those
were ever worth amortising.

**What the experiment did establish.** cos came back *bit-identical*, which is
the point worth keeping: the paired form shared flushes without moving a single
dependency edge, so it changed no arithmetic. The version of this idea that
deletes barriers and trusts the hardware scoreboard to cover the hazard is a
different proposition entirely, and a fidelity shift is exactly how it would
announce itself.

**If it is revisited**, pair only the seven wide ops and leave the one-element
chain per-head, and measure at a shape where the softmax runs more than twice --
`S = 64` is one tile and one head pair, so the per-tile saving is amortised over
a single invocation while the added scalar cost is paid in full. The patch and
these numbers are the whole of what was learned; the shipping kernel keeps the
per-head loop.

**`Div` was kept, and not for want of trying `Rcp`.** There is no vector division
in this kernel to eliminate. The only `Div` in the softmax is
`probScale = OperandMax / amax(row)`, a single lane per head per tile, and the
probabilities are never divided by the running sum here at all -- `runSum` rides
in the partial's tail lane and the combine stage normalises, so an
`inv_sum = Rcp(sum)` folded in here would double-normalise. On the one-lane
`Div` that does exist, `Rcp` trades one instruction for two, on one element, off
the critical path, and gives up the property that makes the op safe: `Div`
guarantees `row * probScale <= OperandMax` exactly, where a reciprocal that
rounds up puts the scaled row over the operand grid ceiling -- 448 for e4m3, 6
for e2m1 -- and `CastToOperand` saturates. That is the `vec_err_idata_inf_nan`
flood of section 13.8, bought for nothing.

### 13.13 Sub-chunk pipelining the ingest: measured, and rejected

The affine ingest is three pipes run one at a time -- the whole tile GM -> UB on
MTE2, then the INT4 -> FP8 expand on V, then the flat stage into L1 on MTE3 --
each idle while the other two work. The obvious move is to cut the tile into
sub-chunks and issue the MTE2 for chunk k+1 ahead of the unpack of chunk k. It
was built, on `TQue` depth 2 for both the packed input and the two-plane output,
and it does not pay.

**A sub-chunk had to be a column-group pair, not a row range.** `CopyInTile`
lands the tile group-major, `[group][row][32]`, precisely so one group's 64-row
run maps element-for-element onto a single NZ fractal column (section 13.9).
Slicing by rows cuts across every group, so each sub-chunk would need
`packedGroups_` descriptors on both the GM read and the L1 write instead of one
-- the descriptor overhead of section 13.12, reintroduced. Slicing by group
keeps both ends contiguous and still gives four chunks per tile at the
production shape: K's two group pairs, then V's.

**`SyncVectorToMte3` cannot appear inside such a loop.** It sets the V -> MTE3
event and immediately waits on it, draining the vector pipe at the call site; in
a pipelined loop that serialises the very stages being overlapped. The edge it
enforces has to become deferred instead -- posted by `EnQue` after the unpack's
vector writes, waited by `DeQue` before the `DataCopy`. Anyone revisiting this
should understand that the invariant is the *edge*, not the helper.

**UB was unchanged**, which is worth recording because it is not obvious:

```
lock-step  kvBuf_ 2*64*128 + operandBuf_ 2*4096        = 24576 B
pipelined  ingest 2*4096   + operand     2*2*4096      = 24576 B
```

What the input side gives up by never holding a whole tile, the output side takes
back by having to keep two nibble planes live per slot.

**It failed the gate on both counts.** Single tile (`ASCEND_TQ_SIM_CONTEXT=64`,
batch 1, `kv4fp8`):

| | cos | total tick | excp_log |
|---|---|---|---|
| lock-step ingest (shipping) | 0.986033 | 68584, 68579 | 32 files, 0 non-empty |
| sub-chunk pipelined | 0.986033 | 69135 | **32 files, 2 non-empty** |

`core0` and `core1` each dropped 6962 B of `su_ccu_mpu_err_t0`, on both vector
subcores:

```
LD_XD_XN dtype:B8, XN:X3=0, ... accessUb:0, accessDdr:1
```

scalar loads at **base 0, resolving to DDR rather than UB** -- the signature of a
`TBuf`/`TQue` whose `InitBuffer` did not take, whose base comes back 0 and whose
accesses then decode as DDR. The byte budget is not the cause; it is identical
either way. The suspicion is TPipe *resource* exhaustion rather than capacity:
two more `TQue` objects on top of `qInQueue_` draw queue and event ids from a
limited per-position pool, and a queue that fails to allocate still answers
`AllocTensor` with a zero-based tensor.

**Note that gtest passed and cos was bit-identical while this was happening.**
The numeric path was undisturbed; only the exception logs showed it. That is the
argument for keeping the `excp_log` check in the gate rather than trusting cos
alone.

**If it is revisited**: hand-rolled ping-pong on a single `TBuf` with fixed event
ids, avoiding new queue objects entirely, would test the resource hypothesis. But
note the tick went the wrong way by ~550 even with the exceptions, and `S = 64`
is the shape where this optimisation should look *best*, not worst -- at one tile
`tile + 1 < numTiles` is false, the prefetch never runs, and the whole ingest sits
exposed in the prologue on the Cube's critical path. It was measured where it had
the most to gain.

### 13.14 The four rules the MIX decode cannot be written without

These are the hardware facts the decode kernel is shaped around. They were
discovered expensively, none of them fault when broken, and until this section
existed three of the four lived only as comment blocks inside the function
bodies that obey them. The kernel now carries one-line pointers here instead.

#### 13.14.1 The Fixpipe lands in ONE subcore's UB

> **See 13.24 rule 2.** The fact below still holds. The conclusion "gate every consumer on
> subcore 0" is a correctness workaround that idles AIV1. The fused decode splits each task's
> heads into two contiguous halves and feeds both subcores with `dualDstCtl = 0b01`.

`IsPrimarySubcore()` -- `GetSubBlockIdx() == 0`.

The accumulator, the score row and the context row live in UB, and **UB is
per-subcore**. The Fixpipe that fills them is issued by the AIC, which writes
into a single subcore's UB and not into both. Every vector subcore of the pair
then believes it holds the product; the one that does not holds whatever its
buffer contained before.

Ungated, both subcores wrote their copy to the same GM address and the last
writer won -- which is why one GEMM alternated exact / all-zero / exact across
launches of a single process (`test_sim_950pr_cube_gemm`'s
`ContextGemmRepeated`). It reads as a layout defect and is not one: the product
in L0C was correct every time.

**The predicate gates only what CONSUMES the product.** The flag protocol stays
ungated -- see 13.14.2.

#### 13.14.2 Every AIV -> AIC flag must be set by BOTH subcores

An arch35 cross-core flag in mode `0x02` is satisfied only once *all* subcores of
the pair have set it. Gating a set to subcore 0 starves the AIC: the first tile
never completes and the launch hangs with an empty exception log. CANN's own MIX
idiom (`WaitPreTaskEndLooseV3Impl`) issues it ungated for the same reason.

This is why a signal whose payload only subcore 0 produced -- `kFlagProbsReady`
after the softmax -- is still posted by both. Mode `0x02` holds the AIC until the
subcore that actually did the MTE3 has signalled, so the other's early set cannot
release it.

#### 13.14.3 `GetBlockIdx()` means different things on the two halves

```
AIV   get_block_idx() * get_subblockdim() + get_subblockid()
AIC   get_block_idx()
```

At `subblockdim = 2` the AIC of block `c` sees `c` while its own two vector
subcores see `2c` and `2c + 1`. A task loop keyed on `GetBlockIdx()` puts the two
halves of one MIX block on *different tasks*, and any kernel whose tiles carry a
cross-core handshake deadlocks the moment the counts diverge. `MixBlockIdx()` --
`GetBlockIdx() / GetSubBlockNum()` -- collapses both halves onto the block index.
Not for a pure-vector kernel, where the 2N subcores are 2N independent workers;
see `TurboQuantPlainCombine`, which keeps `GetBlockIdx()`. Full post-mortem in
13.8.

#### 13.14.4 Vector -> MTE3 needs an explicit event

Every operand staged into L1 is produced by vector work and moved by `DataCopy`,
which is MTE3. `PipeBarrier<PIPE_V>` orders V against V only. Without the
explicit event the DMA reads the buffer before the vector writes land and stages
whatever was there -- zeros on a fresh buffer -- so the Cube multiplies an empty
operand and Fixpipes an all-zero product. Nothing faults.

`SyncVectorToMte3()` sets and immediately waits, which is correct at a call site
that is about to block anyway and wrong inside a software pipeline, where it
drains the vector pipe and serialises the stages being overlapped. In a pipelined
loop the same edge must be carried deferred -- posted after the vector writes,
waited before the DMA. See 13.13.

### 13.15 Where the 68,576 ticks actually go

> **Overridden by 13.24.** Every percentage below is a camodel tick share. The "reject"
> verdict on splitting the softmax across both AIV subcores, and the reading that
> barriers are cheap, do not survive the silicon profile.

Measured, not estimated. The CAModel writes a per-instruction log per core with
issue-cycle timestamps and a unit class; charging the gap between one
instruction and the next to the first gives a cycle attribution. Commit
`c74fc1221`, `S = 64` batch 1 `kv4fp8`, `Ascend950PR_9599`.

| core | span | SCALAR | RVECEX (vec ALU) | MTE2 | MTE3 | CUBE | FIXP |
|---|---|---|---|---|---|---|---|
| AIV block 0, subcore 0 | 67,199 | **41.0%** | 12.9% | 2.1% | 0.6% | - | - |
| AIV block 0, subcore 1 | 66,843 | **39.2%** | 12.2% | 2.0% | 0.5% | - | - |
| AIC block 0 | 53,521 | **93.6%** | - | 0.0% | - | **0.1%** | 4.7% |

Total kernel 68,576 ticks, so the AIV span *is* the kernel.

**The Cube does 79 cycles of arithmetic.** Seventeen `CUBE` instructions, plus
21 `MTE1` totalling 18 cycles and 10 Fixpipes totalling 2,537. The two GEMMs are
262,144 MACs each; at the part's 8-bit rate that is ~64 cycles, so the Cube is
essentially at theoretical speed and idle for 99.9% of the launch. Its
53,521-cycle span is 93.6% scalar, of which `ST_XD_XN_IMM` alone is 80.4% --
descriptor issue and blocking on the AIV.

**Nothing is memory-bound.** MTE2 is 1,436 cycles over 27 instructions and MTE3
is 420 over 32: the entire ingest DMA is **2.7%** of the AIV span.

**Nothing is vector-ALU-bound either.** `RVECEX` is 12.9%. The affine unpack,
the FWHT butterflies and the whole softmax live inside that.

**The kernel is instruction-issue and synchronisation bound.** On the AIV,
SCALAR 41.0% + RVECST 17.6% + FLOWCTRL 7.3% + PUSHQ 5.6% + RVECSU 5.5% is ~77%
of the span, against ~13% of actual arithmetic. `RV_SMEM_BAR` is 6.0% and `DCCI`
4.2% on subcore 0 and 9.6% on subcore 1 -- 10-16% in barrier and cache-coherence
traffic alone.

**This explains 13.12 and 13.13 in one line.** Both targeted a category worth
under 13% of the profile, and both paid for it in the category worth ~80%. The
softmax pairing traded pipeline flushes for scalar branches; the sub-chunk
ingest traded 2.7% of DMA for more loop and descriptor overhead. Neither could
have won.

**Subcore 1 is not idle, so splitting the softmax across both is near-worthless.**
`IsPrimarySubcore()` keeps the softmax, the accumulate and the partial writeback
off subcore 1, which costs it 1,660 instructions out of 35,035 -- and **356
cycles out of 67,199, 0.5%**. Both subcores pay the same staging and
synchronisation bill, and that bill is the kernel. The ceiling on perfectly
parallelising the vector work across the pair is half a percent, against
breaking the invariant in 13.14.1.

#### Ranking the candidate vectors

| vector | ceiling | risk | verdict |
|---|---|---|---|
| Split softmax across 2 AIV | **0.5%** (356 cycles) | high -- breaks 13.14.1; needs dual-destination Fixpipe or a UB hop | reject |
| Hadamard onto Cube MMAD | **~4.6%** (`RV_VADD` 1,694 + `RV_VSUB` 1,421) | high -- new numerics, more descriptor issue in the dominant cost class, extra handshakes | defer |
| `PipeUtilization` on silicon | n/a -- a measurement | low | worth one run to confirm the profile transfers, but it is validation, not leverage |

#### Recommendation: cut the barriers, not the arithmetic

Nine `PipeBarrier<PIPE_ALL>` sit in the decode class. Three are inside the
per-head partial writeback, executed `groupHeads_` times per task; three more
run per tile (`CopyInTile`, `UnpackToL1` twice) and one per tile in the softmax
staging. Each drains *every* pipe, and most guard a narrow dependency that a
targeted barrier or a single event would cover.

Replace each with the narrowest correct edge, starting with the writeback loop.
Unlike 13.12 this removes work rather than reshuffling it, and unlike 13.13 it
adds no new state.

**Verification criteria**

- `cos == 0.986033` exactly at `S = 64`. A dropped edge shows up here; this is
  the check, not a formality.
- 32/32 `core*.excp_log.dump` at 0 bytes.
- Total tick **< 65,000** (>5%) to count, two samples per side against the
  18-tick noise floor.
- Re-profile `core0.veccore0` and confirm the `RV_SMEM_BAR` + `DCCI` share falls.
  If ticks drop without that share moving, the win came from somewhere else and
  the change is not understood.

**Caveats.** This is CAModel data at one tile. The model is cycle-accurate but
may weigh instruction issue differently from silicon, and at `numTiles = 1` the
per-task costs (`PrepareTask`, the writeback) are amortised over a single tile,
inflating their share against a long-context decode. The *ranking* --
issue/sync >> arithmetic -- is robust; the percentages are shape-specific. The
silicon `PipeUtilization` run above is how to confirm the first.

### 13.16 Barrier narrowing: it worked, and not for the reason 13.15 gave

Eight of the nine `PipeBarrier<PIPE_ALL>` in `TurboQuantCubeDecodeSplit` were
replaced with the narrowest correct edge. The ninth is `Init`'s and runs once
per kernel.

| | cos | total tick |
|---|---|---|
| before (`c74fc1221`) | 0.986033 | 68,576 / 68,584 / 68,579 |
| after | 0.986033 | 66,188 / 66,196 |

Fidelity is bit-identical and all 32 `core*.excp_log.dump` stay empty.

**13.15 predicted the win would show up as a fall in `RV_SMEM_BAR` and `DCCI`,
and it did not.** Re-profiling `core0.veccore0` across the change:

| | before | after | delta |
|---|---|---|---|
| span | 67,199 | 64,512 | **-2,687** |
| SCALAR | 27,532 | 22,776 | **-4,756** |
| `ST_XD_XN_IMM` | 12,420 | 10,250 | -2,170 |
| `RV_SMEM_BAR` | 4,065 | 4,111 | +46 |
| `DCCI` | 2,818 | 3,318 | +500 |
| instructions | 35,035 | 35,036 | +1 |

The barrier counters did not move; `DCCI` rose. The whole saving came out of
SCALAR. **`PipeBarrier<PIPE_ALL>` on this part lowers to scalar pipe-control
stores, not to the dedicated barrier opcode**, so reading `RV_SMEM_BAR` as "time
spent in barriers" was wrong. The instruction count is unchanged to within one:
the same work now issues 2,687 cycles faster, so what was removed was stall, not
instructions.

This also caps the lever. 13.15 projected 5-12% on the strength of the
`RV_SMEM_BAR` + `DCCI` share; the measured 3.5% is what was actually there, and
with eight of nine barriers already gone there is no more of it to take.

**Where the remaining headroom is.** SCALAR is still ~35% of the AIV span, and
after this pass it is almost entirely `LocalTensor` address arithmetic and DMA
descriptor construction -- `ST_XD_XN_IMM` 15.9%, `LD_XD_XN_IMM` 3.7%, `MOVK`
3.4%, `AND` 2.4%. Those are per-tile and per-head loop bodies recomputing
offsets that are invariant across the loop. Hoisting them is the next lever and
it is a different kind of change from this one: fewer instructions rather than
fewer stalls.

#### Three narrowings that were specified wrong, and why

Worth recording because each would have passed a build and two of them would
have passed `cos` as well.

- **`CopyInTile` -> `PipeBarrier<PIPE_MTE2>`.** The barrier guards a GM -> UB
  copy whose consumers are *vector* ops. `PipeBarrier<PIPE_MTE2>` orders MTE2
  against MTE2 and says nothing about the vector pipe, exactly as
  `PipeBarrier<PIPE_MTE1>` fails to order MTE1 against M. Correct edge is
  `HardEvent::MTE2_V`.
- **Deleting `UnpackToL1`'s trailing barrier.** `StageTile` calls it twice on one
  `operandBuf_`, so the second call's vector expand races the first call's DMA
  out of that buffer. That is MTE3 -> V; `SyncVectorToMte3` is V -> MTE3, the
  opposite direction, and does not cover it. Correct edge is
  `HardEvent::MTE3_V`.
- **Double-buffering the writeback `tail` by `h & 1`.** Two slots only push the
  write-after-read out by one iteration. It is safe at `groupHeads_ = 2` (the
  simulator tier, two iterations, no reuse) and racy at `groupHeads_ = 4` (the
  device tier). Giving each head its own slot in the finished `scoreBuf_` removes
  the hazard at any group width, and lets the whole writeback run on one
  V -> MTE3 edge instead of one per head.

### 13.17 UB placement beats instruction count, and a query buffer that was never allocated

Three changes, aimed at the SCALAR share 13.16 pointed at. The headline result is
not the one that was aimed for.

| | cos | total tick |
|---|---|---|
| after 13.16 | 0.986033 | 66,188 / 66,196 |
| this pass | 0.986033 | **63,832 / 63,820** |

-3.6%, cumulative -6.9% from 68,580. 32/32 `core*.excp_log.dump` empty, clean
under `-Werror` in both run modes.

#### `qInQueue_` was declared, used, and never InitBuffer'd

`TurboQuantCubeDecodeSplit` ran its per-head query load through a
`TQue<VECIN, 1>` that appears nowhere in `Init`'s allocation list. `AllocTensor`
on an unallocated queue hands back a base of 0, so the load was writing to -- and
the cast reading from -- **the bottom of UB, aliasing whatever buffer sits
there**. It produced the right answer because nothing happened to touch that
region between the copy and the cast.

This is the failure mode of 13.13 reached by a different route, and unlike 13.13
it raised no exception: `su_ccu_mpu_err` fires when the aliased address is
*unmapped*, not when it merely belongs to something else. cos was exact
throughout. **Neither the fidelity gate nor the exception gate can see this
class of bug** -- only reading the allocation list can.

Replaced with a real `TBuf`, and since the group's heads are consecutive in GM
the whole group now loads in one burst instead of one per head, with one
MTE2 -> V edge instead of the (absent) per-head ones.

#### 8 KB at the front of the allocation order cost 8.3%

The first version sized the new buffer for `kMaxGroupHeads` and allocated it
before `qBuf_`. Identical code, one measurement:

| `qInBuf_` | size | position | tick |
|---|---|---|---|
| first attempt | `kMaxGroupHeads * headSize_` (8 KB) | before `qBuf_` | **71,707** |
| shipped | `groupHeads_ * headSize_` (1 KB) | last | **63,832** |

**7,875 ticks, 12%, from moving one `InitBuffer` call and sizing it for the
actual group.** Inserting a buffer at the front shifts the base address of every
buffer after it, and this kernel is sensitive to those addresses far beyond
anything instruction-level attempted in this session. Anyone adding a UB buffer
here should append it and size it to the shape actually in use; doing otherwise
is worth more than every optimisation in 13.12 through 13.16 combined, in the
wrong direction.

#### The win was PUSHQ, not SCALAR

The pass targeted the SCALAR share. It did not move it:

| AIV subcore 0 | before | after | delta |
|---|---|---|---|
| span | 64,512 | 62,799 | -1,713 |
| SCALAR | 22,776 (35.3%) | 23,059 (**36.7%**) | **+283** |
| PUSHQ | 3,483 | 1,995 | **-1,488** |
| RVECST | 11,776 | 11,165 | -611 |
| instructions | 35,036 | 34,911 | -125 |

SCALAR rose in both absolute and relative terms. The saving is `PUSHQ` -- the
queue-push unit -- which is what deleting `4 * groupHeads_` `TQue` operations
should hit, plus vector stores.

**That is twice running that the predicted mechanism was wrong while the change
still worked** (13.16 predicted `RV_SMEM_BAR`, got SCALAR; this predicted SCALAR,
got `PUSHQ`). The counter shares are useful for finding *what is large*, and
unreliable for predicting *what a given edit will move*. Measure the edit; do not
reason forward from the share.

#### What was not done, and why

- **`qScale_[h] = reduce.GetValue(...)`** stays. Two scalar reads per *task*.
  Holding the factor in UB instead replaces a `Muls` immediate with a broadcast
  `Mul` over every score row, per head, **per tile** -- a wash at `S = 64` where
  there is one tile, and a regression at production context lengths.
- **Batching `ApplyPi` across heads** stays undone. The rotation is `Init`'d at
  `batchRows = 1` and its Gather offset tables are sized to that, so batching is
  surgery on a codec the encode path shares. It is also the only remaining
  candidate that could plausibly reach 63,000: `RV_VADD` + `RV_VSUB` is ~4.9% of
  the span. It needs a multi-tile gate first.
- **Hoisting invariant address arithmetic**, the nominal target of this pass, was
  dropped after profiling. The PC histogram shows the loop bodies that do that
  arithmetic running at **0.76-1.03 cycles per instruction**; the expensive PC
  windows execute 51-128 times at 8-40 cycles each. There is no hot loop to hoist
  out of, only stalls.

### 13.18 Batching the rotation: audit, and why the Cube variant is the wrong one

#### What exists

| artefact | what it is |
|---|---|
| `csrc/tests/common/hadamard_spike.hpp` | host helpers: `Hadamard16Half`, `EarlyStageTables`, chunk sizing, `HadamardDualDstApplies` |
| `csrc/tests/common/hadamard_spike_kernels.cpp` | 954 lines. AIV-batched butterfly **and** Cube-factorised variants, double-buffered across kSlots |
| `csrc/tests/device/bench_950pr_cube_hadamard.cpp` | the silicon benchmark |
| `hadamard_benchmark_results.csv` | 181 rows of real 950PR device measurements, 100 samples per case |
| `csrc/attention/turboquant/turboquant_codec_950.h` | the shipping `ApplyPi` / `FastWalshHadamardTransform`, at `batchRows = 1` |

#### The Cube factorisation does not apply at the decode's shape

`d256`, device mode, median us:

| vectors | AIV | single | hilo | dualdst |
|---|---|---|---|---|
| v1 | 8.649 | 8.708 | 8.573 | - |
| v8 | 9.847 | 9.737 | 9.775 | 8.863 |
| v16 | 12.762 | 9.366 | 9.543 | 9.926 |
| v32 | 24.501 | 17.182 | 17.471 | 10.645 |

stddev runs 0.68-1.30 across these, so **at v1 every variant is inside one
standard deviation of the AIV baseline**. The Cube only pulls away at v16 (-27%)
and v32 (-57%), where its fixed staging cost amortises.

The decode rotates `groupHeads_` query vectors per task -- **2 at the simulator
tier, 4 at the device tier**. That is below every crossover measured. Offloading
the rotation to the Cube at this shape buys nothing and adds cross-core
handshakes.

#### What the same table says about AIV batching

Read the AIV column alone: **8 vectors cost 1.139x what 1 vector costs**. The
AIV path is fixed-overhead-dominated at low vector counts, so `groupHeads_`
separate single-vector calls collapse to roughly the cost of one. Interpolating,
2 vectors are ~1.02x one and 4 are ~1.06x.

Against the decode's measured `RV_VADD` 1,694 + `RV_VSUB` 1,421 = 3,115 cycles of
butterfly, batching at `groupHeads_ = 2` should return ~1,500 cycles, and the
per-call overhead it also folds -- the `ShuffleStage` Gathers and Muls, and the
~11 `PipeBarrier<PIPE_V>` a 256-wide `ApplyPi` issues -- roughly as much again.
Call it **2,000-2,500 cycles, ~3-4%**, which would put `S = 64` near 61,500 and
clear 63,000. This is an estimate from a benchmark at a different shape, not a
measurement of the decode.

#### What the code change actually is

`FastWalshHadamardTransform` is closer to batchable than it looks. Laying B
vectors of length L contiguously as `[B][L]`:

- **`ShuffleStage`** is a Gather plus two elementwise ops over `n`. The butterfly
  pairs index `i` with `i XOR s`; because each vector's base is L-aligned and
  `s < L`, that never leaves the vector. The stage batches by lengthening
  `xorOffset_` and `sign_` to `B * L` -- and `Init` already takes `batchRows`
  and sizes the tables through `ConstTableWords(vecLen, batchRows)`.
- **`BlockStage`** butterflies within `2 * stride`-sized groups, and those groups
  tile each L-length vector exactly (both powers of two, `stride < L`). It
  batches at `n = B * L` **with no change at all**.
- **The one real bug** is the loop bound. `for (stride = kFp32PerBlock; stride <
  n; stride <<= 1)` with `n = B * L` keeps going past `stride = L`, and a stage
  at `stride >= L` pairs across vectors. At L=256, B=2 it would run a spurious
  `stride = 256`. Both that bound and the early-stage `if (stride >= n) return;`
  must test the *per-vector* length, not the batched one.

So: two loop bounds, a `batchRows` parameter threaded through `ApplyPi`, and
longer tables. `common/hadamard_spike_kernels.cpp` already runs a batched butterfly
of this shape on both CAModel and silicon, so this is porting a proven form
rather than inventing one.

#### The risk is UB placement, not correctness

Batching grows three things: `constBuf_` by `2 * vecLen * (batchRows - 1)` words,
`workBuf_` by `vecLen * (batchRows - 1)`, and `qBuf_` from `2 * headSize_` floats
to `2 * groupHeads_ * headSize_` -- about +3 KB at `groupHeads_ = 2`, +9 KB at 4.

13.17 measured an 8 KB buffer allocated at the front of the order costing
**12%**, which is larger than this optimisation's entire upside. `rotation_.Init`
sits in the middle of the decode's allocation sequence, so growing it shifts
every buffer after it.

**Mitigation is known and cheap**: size by `groupHeads_` rather than
`kMaxGroupHeads`, and append new allocations rather than inserting them. Both are
the lesson of 13.17 applied directly.

#### Recommendation

Integrate the **AIV-batched** variant; do not pursue the Cube factorisation at
this shape. Order the work so the dominant risk is measured first:

1. Thread `batchRows` through `ApplyPi` / `FastWalshHadamardTransform`, fixing
   the two stride bounds. Keep the encode path at `batchRows = 1` via a default,
   since `TurboQuantCodec4` is shared with `TurboQuantModeReshapeAndCache`.
2. Grow `qBuf_` and re-`Init` `rotation_` at `groupHeads_`, **appending** the
   growth.
3. Gate on exact `cos == 0.986033` -- a stride bound that leaks across vectors
   changes the rotated basis and shows up there immediately -- plus 32/32 empty
   exception logs.
4. Take a UB-placement A/B before believing any tick number, per 13.17, and
   confirm the win in `RV_VADD` + `RV_VSUB` rather than in total ticks alone.
   13.16 and 13.17 each moved a different counter than predicted; this one
   should be checked against the counter it targets.

Worth stating plainly: the estimate above rests on a v1-to-v8 interpolation from
a standalone benchmark. The decode's rotation runs inside a task with different
cache and issue pressure, and the honest confidence interval on "2,000-2,500
cycles" is wide.

### 13.19 Correction to 13.18: batching cannot move RV_VADD / RV_VSUB

13.18 recommended integrating a batched AIV butterfly and estimated 2,000-2,500
cycles. Reading the codec closely enough to write the patch turned up two errors
in that audit. Both point the same way, and the integration was not done.

#### Error 1: `batchRows` does not size the shuffle tables

13.18 claimed `ShuffleStage` "batches by lengthening `xorOffset_` and `sign_` ...
and `Init` already takes `batchRows` and sizes the tables through
`ConstTableWords`". It does not. The layout is

```
ConstTableWords(vecLen, batchRows) = 7*vecLen + 2*vecLen*batchRows + kLevels
```

and `Init` consumes it as `kEarlyStages * (sign_ len_ + xorOffset_ len_)` --
**`len_`, fixed, batchRows-independent** -- then `evenOffset_`/`oddOffset_`, then
`expandOffset_` and `oddSelect_` at `batchLen_` each. The header says so at the
layout comment: `B = len * batchRows` labels those last two. **The `batchRows`
term feeds the dequantiser, not the transform.**

So `ShuffleStage(x, tmp, stage, B*L)` reads `sign_[stage]` and `xorOffset_[stage]`
past their `vecLen` extent, into whatever table follows -- and `xorOffset_` feeds
a `Gather`, so the over-read becomes wild source offsets rather than merely wrong
arithmetic. Batching the shuffle stages requires a new table format: the device
layout, the `CodecTableWords` host mirror in `turboquant_torch_adpt.h`,
`CheckCodecTables`, the builder that fills the tensor, and every test that
constructs one.

#### Error 2, and the decisive one: the ALU work does not shrink

The 2,000-2,500 estimate came from reading the standalone benchmark's AIV column
-- 8 vectors costing 1.139x one vector -- as evidence that per-head calls
collapse. That extrapolation does not transfer to the decode.

A fast Walsh-Hadamard transform of B vectors of length L is `B * L * log2(L)`
butterflies. **Batching changes how those are issued, not how many there are.**
`RV_VADD` and `RV_VSUB` count element-work, so they are invariant under batching:

```
per task, groupHeads_ = 2, L = 256:
  2 heads * (3 shuffle adds * 256 + 5 block adds * 128) = 2,816 element-adds
  measured RV_VADD 1,694 cycles  ->  ~1.66 elements/cycle
```

Batched, it is the same 2,816 element-adds. What the benchmark's v1 point is
dominated by is *fixed* cost -- kernel launch, the GM table load into UB, setup
-- which the decode pays **once per task, not once per head**. The head loop
never re-pays it, so there is nothing there for batching to amortise.

What batching does save is real but small: one call's worth of instruction issue
and the ~11 `PipeBarrier<PIPE_V>` a 256-wide `ApplyPi` emits, times
`groupHeads_ - 1`. At `groupHeads_ = 2` that is on the order of **200-400
cycles**, not 2,000-2,500.

#### Consequence for the stated gates

A verification criterion of "confirm the cycle drop originates specifically from
`RV_VADD` and `RV_VSUB`" **cannot be met by AIV batching**, because AIV batching
does not reduce butterfly element-work. The only transformation that does is
moving butterflies onto the Cube -- and 13.18's own table shows that losing to
the AIV baseline at every vector count below 16, while the decode runs 2 or 4.

So at `d256, v = 2..4` there is no available change that reduces the decode's
Hadamard ALU cycles. The rotation is already at the fast transform's operation
count, on the right unit for its shape.

#### What is still worth doing

Batching remains worth roughly 200-400 cycles at `groupHeads_ = 2` and more at 4,
from issue and barrier overhead alone, and it needs no table change **if it is
restricted to the block stages** (strides >= `kFp32PerBlock`), which use no
tables: their `2*stride` groups tile each L-length vector exactly, so they batch
at `n = B*L` unchanged. The shuffle stages stay per-head. The stride loop bound
still has to test the per-vector length rather than the batched one, or a
spurious cross-vector stage runs at `stride = L`.

That is a real but modest win against a wide confidence interval, and it does not
reach the 62,000 target. Recorded rather than built, pending a decision on
whether a few hundred cycles justifies the change.

### 13.20 A context length that is not a multiple of 8 misaligns the tail mask

Found by a corner-case sweep of the simulator tier at `51469c5d4` (codegen
identical to `3ee5f7f0d`; the commit between them is comment-only).

| shape | tiles | tail `valid` | cos | excp_log |
|---|---|---|---|---|
| S=16 | 1 | 16 | 0.991817 | 32/32 empty |
| S=64 | 1 | 64 (no mask) | 0.986033 | 32/32 empty |
| **S=65** | 2 | **1** | 0.988713 | **2/32 non-empty** |
| S=192 | 2 | 64 | 0.987147 | 32/32 empty |

S=65 run alone on a clean build reproduces it exactly: cos 0.988713, 87,320
ticks, `core0` and `core1` each 764 B of

```
vec_err_instr_misalign_t0  RV_VST  Sn[10]=0xc004
```

Four records per core -- `groupHeads_` heads times the two masking `Duplicate`
calls.

#### The cause

`SoftmaxStageProbs` masks the invalid tail of a partial tile:

```cpp
if (valid < kCubeTileRows) {
    Duplicate(row[valid], kNegInf, kCubeTileRows - valid);   // and later, 0.0f
}
```

`row[valid]` is 32-byte aligned only when `valid` is a multiple of 8 floats. At
S=65 the second tile carries `valid = 1`, so the store base lands 4 bytes past a
block boundary -- `0xc004` in the dump, which is the signature.

**This is pre-existing.** The masking predates the barrier narrowing, the
writeback restructure and the query-buffer fix. It has gone unseen because every
context length this kernel has ever been run at -- 16, 64, 128, 512, 1024, 2048,
and the production shapes -- leaves `valid` a multiple of 8. It takes a context
like 65, 100 or 2049 to expose it, and a decode serving arbitrary sequence
lengths will hit those constantly.

Note what the gates did and did not catch. **cos stayed plausible at 0.988713**,
so a fidelity-only gate passes this. Only the exception log saw it -- the same
lesson as 13.13, from the opposite direction: there the fault was loud and the
numbers were right; here the numbers are right and the fault is quiet but real.

#### The fix, not yet applied

Mask from an aligned base with a lane predicate rather than from `row[valid]`
directly: take `base = valid & ~(kFp32PerBlock - 1)` and use the mask form of
`Duplicate`, whose bitmask can exclude the lanes in `[base, valid)` that must
survive. The naive alternatives are both wrong -- rounding the base down
overwrites live scores, rounding it up leaves `[valid, alignedUp)` unmasked and
those lanes feed the row's `ReduceMax`.

Until it is fixed, **the simulator and device sweeps should include at least one
context that is not a multiple of 8**, or this stays invisible.

### 13.21 The UB bank map, and a descriptor pass that is not measured

Two changes on top of `23fb38218` (codegen identical to `3ee5f7f0d`; the two
commits between them are comment-only): every shape-invariant `DataCopyParams`
moved into `Init`, and the two per-task `Duplicate` calls narrowed to the group
actually in use. Plus an audit of where the kernel's UB buffers land in the bank
geometry, which is the part worth reading.

**NOT MEASURED.** No camodel pass has been run on it. Everything below that is a
number is either static arithmetic over the allocation list or a count of edits;
nothing is a tick. `cos` and the exception logs are likewise unverified.

#### The bank geometry, and where this kernel's buffers sit

The arch35 UB is 256 KB shaped `(depth 512, banks 2, bank_groups 8, block 32)`
at strides `(512, 256, 32, 1)` -- `UB_BANK_STRIDE` and friends in
`attention/lightning_indexer/op_kernel/lightning_indexer_common.h`. A byte's
bank is therefore **bit 8 of its address**: two operands whose bases differ by a
multiple of 512 read the same bank at every element of the walk.

Every one of the fourteen buffers `TurboQuantCubeDecodeSplit::Init` allocates is
a whole multiple of 512 bytes at `head_size` 128 and 256 -- every shape either
tier runs -- so **all fourteen start at (bank 0, bank group 0)**. At `head_size`
64, `signBuf_` (256 B) is the first that is not, and only the two cold buffers
after it change bank.

`kv4fp8`, `head_size` 256, `num_kv_heads` 2:

| buffer | offset | size | bank |
|---|---|---|---|
| `kvBuf_` | 0 | 16384 | 0 |
| `scaleTileBuf_` | 16384 | 2048 | 0 |
| `operandBuf_` | 18432 | 8192 | 0 |
| `accBuf_` | 26624 | 16384 | 0 |
| `qBuf_` | 43008 | 2048 | 0 |
| `qOperandBuf_` | 45056 | 4096 | 0 |
| `scoreBuf_` | 49152 | 4096 | 0 |
| `probOperandBuf_` | 53248 | 1024 | 0 |
| `ctxBuf_` | 54272 | 16384 | 0 |
| `stateBuf_` | 70656 | 3072 | 0 |
| `reduceBuf_` | 73728 | 2048 | 0 |
| `signBuf_` | 75776 | 1024 | 0 |
| `scaleIdxBuf_` | 76800 | 512 | 0 |
| `qInBuf_` | 77312 | 1024 at `groupHeads_` 2 | 0 |

The consequence for the hot binary ops:

| op | operand bases | bank |
|---|---|---|
| `Accumulate`: `Add(acc, acc, ctx)` | 26624, 54272 | **same, whole length** |
| `SoftmaxStageProbs`: `Mul(row, row, kScale)` | 49152 + 256h, 73728 | same for even `h` |
| `SoftmaxRescaleAcc`: `Mul(acc[col], acc, alphaBlocks)` | 26624 + 256c, 74560 | same for odd `c` |
| softmax `Max` / `Sub` / `Mul` on `stateBuf_` | fields 512 B apart | **always same** |

The `stateBuf_` fields are the worst case in the layout -- the field stride is
`kMaxGroupHeads * kFp32PerBlock * sizeof(float)`, exactly 512 -- but every op on
them is one element, so a bank conflict is not what those cost. The one that is
worth anything is `Add(acc, acc, ctx)`: two source streams of
`groupHeads_ * head_size` fp32, 2 KB each on the simulator tier and 4 KB on
silicon, in one bank from end to end.

#### 13.17's regression was not a bank conflict

That pass is sometimes read as having proved bank conflicts dominate here. It
did not, and the geometry says so directly: the buffer inserted in 13.17 was
**8192 bytes, which is 16 x 512**. A shift by a multiple of 512 leaves every
buffer's bank *and* bank group exactly where it was. Whatever cost 7,875 ticks
there, it was not bank placement -- and that measurement conflated two variables
anyway, since the same attempt also sized the buffer at `kMaxGroupHeads` instead
of `groupHeads_`.

So 13.17's finding stands, unchanged and still unexplained: **this kernel is
extremely sensitive to UB base addresses, by a mechanism not yet identified.**
That is why the pad below ships at 0.

#### The knob

`kUbBankPadBytes`, at the top of `turboquant_mm_kernels.cpp`, is added to
`probOperandBuf_` -- the allocation immediately before `ctxBuf_`. At 256 it
splits `acc` and `ctx` into different banks and leaves every other pair's
conflict count unchanged: half the `kScale` / `vScale` and rescale pairs swap
which side conflicts, and the `stateBuf_` fields, 512 apart internally, do not
move relative to each other at any pad. It rides on an existing buffer rather
than one of its own so that nothing before `ctxBuf_` shifts.

It is **0**, which is byte-for-byte the allocation that measured 63,832. Sweep
`0 / 256 / 768` on the camodel before drawing any conclusion; on 13.17's
evidence the right prior is that it could go either way by 10%.

#### Phase 1: what moved into `Init`

Per tile, at `head_size` 256:

| site | descriptor | was built |
|---|---|---|
| `CopyInTile` | tile read params | 1 per tile |
| `UnpackToL1` | UB to L1 params | 2 per tile, K and V |
| `SoftmaxStageProbs` | probs to A1 params | 1 per tile |
| `SoftmaxRescaleAcc` | `BinaryRepeatParams` | 4 per tile, `head_size / 64` |

Eight descriptor constructions per tile, each 4 to 6 immediate field stores, now
one construction each in `Init`. Per task: the query's NZ descriptor, and
`scale_ / kGain`, which was an fp32 divide per tile.

`qScale_` became `qScaleInv_`, holding `1 / qScale_[h]` from the moment
`PrepareTask` computes it. `SoftmaxStageProbs` was doing
`Muls(row, row, 1.0f / qScale_[h])` once per head **per tile**; the reciprocal is
now taken once per head per task. Bit-identical -- the same divide on the same
operand, fewer times -- and worth `(tiles - 1) * groupHeads_` scalar divides,
which is nothing at S=64 and 31 x `groupHeads_` at S=2048.

#### Phase 1: two `Duplicate` calls sized for the wrong group

Both wrote `kMaxGroupHeads` rows where only `groupHeads_` are ever read. This is
13.17's own lesson -- size it to the shape actually in use -- applied to the two
places that still did not:

| call | was | now | saved per task at `groupHeads_` 2 |
|---|---|---|---|
| `ComputeSplit`, `Duplicate(acc, 0)` | 16 x 256 fp32, 16384 B | `groupHeads_` x 256 | 14336 B |
| `PrepareTask`, `Duplicate(qOperand, 0)` | 16 x 256 B, 4096 B | `groupHeads_` x 256 B | 3584 B |

Safe by inspection rather than by measurement, and the argument is worth writing
down, because "the buffer is sized for 16 heads" is a different question from
"are rows 2..15 read":

- `acc` rows past the group are touched by nothing. `SoftmaxRescaleAcc`'s `Mul`
  repeats `groupHeads_` times, `Accumulate`'s `Add` is `groupHeads_ * head_size`
  long, and the partial writeback walks `h < groupHeads_`.
- `qOperand` slots past the group are never a `DataCopy` source -- the staging
  loop is `h < groupHeads_` -- and inside the group `CastToOperand` overwrites
  all `operandElems_` bytes of the slot before it is read. Those zeros never
  reached L1, so dropping them cannot change what the Cube sees. The A operand's
  rows `groupHeads_..15` are stale in L1 either way; that is pre-existing and
  has nothing to do with this `Duplicate`.

`Duplicate(state, 0)` stays at `6 * kMaxGroupHeads * kFp32PerBlock`: its six
fields are strided by the padded group width, so there is no contiguous prefix
to narrow to.

#### What 13.17 predicts about all of this

That the descriptor half will not move the needle. 13.17 profiled the loop
bodies doing exactly this arithmetic at 0.76-1.03 cycles per instruction,
concluded there is no hot loop to hoist out of, only stalls, and dropped the
hoisting work for that reason; it also watched SCALAR *rise* while instruction
count fell. This is the same target reached by a different route and should be
expected to behave the same way.

The two `Duplicate` narrowings are a different animal -- 17,920 fewer bytes of
vector store per task -- and are the part of this section most likely to show up
in a tick count.

Measure it. Do not reason forward from this section.

### 13.22 The query rotation moved out of both decode splits

`npu_turboquant_rotate_q` applies `Pi q = D (H (D q))` to a whole decode step's
query in its own launch, and both decode splits -- the AIV
`TurboQuantPagedAttentionSplit` the wheel ships and the Cube
`TurboQuantCubeDecodeSplit` -- now read the result. Neither contains a
Walsh-Hadamard transform of any kind. The cache is still written rotated by
`reshape_and_cache`, and the output is still un-rotated by the combine; only the
query's rotation moved.

#### What it removes from the decode

`PrepareTask` used to run, per query head of the GQA group: a sign multiply,
three `Gather` shuffle stages, `log2(head_size / 8)` block-strided butterflies, a
normalisation, a second sign multiply, and the roughly eleven
`PipeBarrier<PIPE_V>` a 256-wide `ApplyPi` emits. It ran **once per task**, and a
task is `(token, kv_head, split)` -- so the same transform ran `num_splits` times
over. At the benchmark shape that is 1x at S=64 and **8x at S >= 1024**.

The redundancy is self-limiting and should not be oversold: `num_splits =
CeilDiv(aivNum, base_tasks)` exceeds 1 only when cores are idle, so the repeated
work ran on cores with nothing else to do. What the move buys at B=1 is latency
off every split block's critical path, not throughput.

#### Dispatch

`turboquant::PlanRotateQ` picks the path on the host, and the kernel is told
which rather than re-deriving it, so the plan that is validated is the plan that
runs.

| gate | path | why |
|---|---|---|
| `N >= 16` and `N % 16 == 0` | Cube | Sylvester factorises `H_D = H_R (x) H_16`, so strides 1, 2, 4 and 8 are one Mmad against a static 16x16 constant at any `D`, and only the `log2(R)` whole-row stages stay on the vector unit |
| otherwise | AIV | all `log2(D)` stages through `TurboQuantCodec4::ApplyPi` -- the same routine the decode used to call per head |

`N = num_tokens * num_heads`. The gate is exact divisibility rather than a
threshold because a block share that is not a whole number of tiles would need a
short final Mmad, and a short Mmad is the one piece of the spike that was never
run. The planner then guarantees two invariants the Cube kernel relies on and
does not check: `vectors_per_block` divides `N`, and `vectors_per_chunk` divides
`vectors_per_block`. The torch operator asserts both.

**Where the Cube path pays, and where it does not.** The device sweep in
`hadamard_benchmark_results.csv` puts the crossover at 16 vectors per block: at
`D = 256` the marginal cost per vector is 0.611 us on the vector unit and 0.074
us on the Cube with `dualDstCtl`, but the Cube's fixed staging only amortises
from 16 up. A standalone kernel also has a lever the in-decode rotation never
had -- it can put one vector on each of many blocks -- so for *latency* at small
`N` the vector path spread wide still wins. 13.18 and 13.19 reached that verdict
for the in-decode case; it holds here for the same arithmetic. The Cube path is
in the tree because MLA at `H_KV = 1` and continuous batching at `B = 4..16` put
`N` well past the gate, which is a throughput regime, not a `B = 1` one.

#### Precision, and why a half query needs one Mmad and not two

`H_16` holds only `+-1`, both exact in fp16, and the sign multiply is by `+-1`.
So a **half** query is exactly representable in the Cube's operand grid after
`D x`, and a single Mmad with its fp32 accumulator introduces no error at all.
A bfloat16 query is not -- its 8-bit exponent reaches values half cannot hold --
so `PlanRotateQ` sets `kHiLo` for it and the kernel issues two Mmads
accumulating into one L0C. The host decides; the kernel does not infer it from
`scalar_t`.

#### What was measured

`test_sim_950pr_turboquant_rotate_q` on the arch35 camodel, `D = 256`,
`H = 8`, 425 s, exit 0, 32/32 `excp_log` dumps at 0 bytes:

| case | path | shape | result |
|---|---|---|---|
| `RotateQPlan.Dispatch` | host only | ten `(N, D, cores)` points | every path, chunk and divisibility invariant as specified |
| `RotateQVector.SparseBatchIsBitIdentical...` | AIV | `N = 8` | **0 of 2048 elements differ** from `cpu_apply_pi` |
| `DenseBatches/RotateQCube.MatchesTheReference/0` | Cube | `N = 16`, 1 block x 16, chunk 16 | **0 of 4096 differ**, `max abs err 0.000e+00` |
| `DenseBatches/RotateQCube.MatchesTheReference/1` | Cube | `N = 64`, 4 blocks x 16, chunk 16 | **0 of 16384 differ** |
| `RotateQCube.AgreesWithTheVectorPath...` | both | `N = 64` | **0 of 16384 differ** between the two paths |

The vector path being bit-identical was expected: it is the reference's own
algorithm, on the same fp32 data, in the same order.

**The Cube path being bit-identical was not.** It evaluates four butterfly
stages as a 16x16 matrix product where the reference evaluates them as four
passes of pairwise adds, which reassociates sixteen fp32 sums -- the test's
`kCubeAbsTolerance` of 1e-5 exists for exactly that and was not needed. Treat
this as a *measurement at these shapes*, not a guarantee: a different head size,
a bfloat16 query on the `kHiLo` path, or data at a different scale can
reassociate differently, and the tolerance stays for that reason.

`test_sim_950pr_turboquant_multimode`, kv4fp8, end to end through the decoupled
decode -- write the cache, rotate, split, combine, compare against fp32 host
attention:

| shape | `N` | rotation path | result |
|---|---|---|---|
| `B=1 S=16`, 4 heads / 2 kv heads | 4 | AIV | **cos 0.991817**, 580 s, 32/32 dumps empty |

#### UB freed in the decode

Static arithmetic over the allocation lists, `head_size` 256:

| kernel | removed | grown | net |
|---|---|---|---|
| `TurboQuantCubeDecodeSplit` | `rotation_` (const 9,280 + work 11,552), `signBuf_` 1,024 | `qInBuf_` fp16 -> fp32, +2,048 at `groupHeads_` 4 | **-19,808 B** |
| `TurboQuantPagedAttentionSplit` | `signBuf_` 1,024, two thirds of `accBuf_` | `qInQueue_` fp16 -> fp32 | **-1,536 B** |

The Cube split's figure is 51,552 - 4,096 = **-47,456 B at `head_size` 512**,
which takes 7.6's static count from 234,624 B of the part's 253,952 to about
187,000 and its headroom from 19 KB to about 67 KB. That is the difference
between "fits on paper" and fits, and it is the argument for revisiting
`CheckHeadSize`'s 256 cap for MLA -- *after* the `D = 512` fidelity gate has
actually been run, which 7.6 is explicit it has not.

#### Two things this change did NOT do, deliberately

**It did not keep the allocation lists byte-identical.** 13.17 measured an 8 KB
buffer inserted at the front of the Cube split's allocation order costing 7,875
ticks -- 12% -- and 13.21 showed the mechanism is still unidentified. The one
allocation this refactor grows in that kernel, `qInBuf_`, is the **last** in the
order, so nothing moves behind it; but removing `signBuf_` shifts the two after
it, and the AIV split's list moves more than that. Neither kernel's UB placement
has been A/B'd since. **Do that before believing any tick number from either.**

**It did not price the extra launch.** `tq_rotate_s<S>` in
`bench_device_950pr_turboquant` is the instrument: one launch, the rotation
alone, flat in S by construction. Subtracting it from `tq4_decode_s<S>` --
which now enqueues rotate, split and combine -- is what prices the operator
invocation the contract costs. It has not been run: this host has no NPU, and
the ~8.6 us floor in `hadamard_benchmark_results.csv` is a `replan-per-launch`
harness artefact, not the graph-replay figure a decode step pays. Until that
number exists, the latency case for the split rests on an estimate.

### 13.23 The output de-rotation moved into W_o, and prefill against FIA V5

The combine kernel no longer applies Pi. `TurboQuantPagedAttentionCombine` merges
the per-split `(max, sum, accumulator)` triples, normalises, and writes the
**rotated** output `O~ = softmax(q k^T) V~`. The inverse lives in the weights of
the output projection instead, folded offline. The query's rotation stays where
13.22 put it, in `npu_turboquant_rotate_q`; K, V and Q still rotate as
activations, so RoPE never sees the rotated basis.

#### The fold

With `W_o` of shape `[hidden, H * D]` acting as `y = W_o o`, `o` the
concatenation of `H` head vectors, and each head's output `o_h = Pi o~_h`:

```
y = W_o (I_H (x) Pi) o~ = W_o' o~,        W_o' = W_o (I_H (x) Pi)
```

Every `D`-wide input block of every row is multiplied by Pi -- `(W_h Pi)[r] =
Pi W_h[r]` because Pi is symmetric -- so the fold is one `apply_pi` over the
trailing axis of `W_o.view(hidden, H, D)`. In the row-vector convention it is
`Pi^T W_o = Pi W_o`. Two properties make it deployable:

- it is **block diagonal over heads**, so it commutes with `RowParallelLinear`'s
  sharding of `o_proj`'s input dimension, which always cuts on head boundaries;
- the **bias is untouched**.

`vllm_ascend/attention/turboquant_rotation.py` holds it
(`fold_pi_into_output_projection`, `validate_output_projection_fold`), and imports
torch alone so `scripts/tq_fold_output_rotation.py` runs on a host with no
torch_npu and no vLLM. The script rewrites every
`<prefix>.self_attn.o_proj.weight` of a safetensors checkpoint, validates each,
and records the fold in `config.json`:

```json
"turboquant_output_rotation": {"format_version": 1, "pi_seed": 1597463007,
                               "head_size": 128, "folded_modules": ["model.layers.0.self_attn", ...]}
```

The seed and head size are there because a fold is only valid for the Pi it was
made with, and nothing else in a checkpoint says which one that was.

**Measured, float64 reference, 256 random rotated outputs per case** (host,
torch 2.13):

| check | D=128 H=32 | D=256 H=8 | D=128 H=64 | D=256 H=16 |
|---|---|---|---|---|
| `|fold - W_o block_diag(Pi)| / |W_o|` | 4.05e-16 | 5.56e-16 | 4.05e-16 | 5.57e-16 |
| TP shards folded separately vs whole | 0 | 0 | 0 | 0 |
| worst cos, `W_o'` stored fp32 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 |
| worst cos, `W_o'` stored fp16 | 0.999999976 | 0.999999976 | 0.999999977 | 0.999999977 |
| worst cos, `W_o'` stored bf16 | 0.999998454 | 0.999998405 | 0.999998524 | 0.999998490 |

The bound the tool enforces is `cos > 0.9999`; bf16's rounding of the rotated
weights is the largest error and sits two orders of magnitude inside it.
End to end, `softmax(q k^T) V~` projected by `W_o'` against `softmax(q k^T) V`
projected by `W_o` (D=256, H=8, S=300, float64): cos `0.9999999999999996`,
relative error `1.8e-15`.

#### Where the fold is wrong: the output gate

Qwen3-Next and Qwen3.5 -- this suite's own golden layer, see
`scripts/dump_qwen35_layer3.py` -- multiply the attention output by
`sigmoid(gate)` **before** `o_proj`. An elementwise gate does not commute with Pi:

```
o_proj( O * g )  !=  o_proj'( (O Pi) * g )
```

Measured with the same random layer and a random gate: **cos 0.859**. So those
layers are never folded, and vLLM makes this easy to get wrong -- it reads
`attn_output_gate` with a default of `True`, so a config that omits the key is
gated. The script refuses when the gate is truthy **or absent on a model type
that gates by default**, refuses model types it has not checked unless told
`--assume-no-output-gate`, and refuses a quantised `o_proj` (a non-float weight
or companion tensors such as `weight_scale`): rotating a quantised weight needs a
requantisation, so fold first.

#### The runtime contract

Two ways to get the basis wrong, and both produce plausible-looking text rather
than an error: un-rotating into a folded projection, or not un-rotating into an
unfolded one. So the backend decides per layer, from the checkpoint, and refuses
rather than guesses:

| `turboquant_output_rotation` in the config | layer listed | `output_rotation_folded` | decode | prefill |
|---|---|---|---|---|
| absent | -- | `False` | rotate Q, split, combine, **rotate the output** (Pi is an involution) and copy it back | unchanged |
| present, seed and head size match | yes | `True` | rotate Q, split, combine; output stays `O~` | **rotate V** before FIA |
| present | no, or seed / head size differ | load fails with the reason | -- | -- |

- The default is the unfolded path because it is correct for every model. It
  costs one more `npu_turboquant_rotate_q` launch per decode step, the size of
  `tq_rotate_s<S>`, and lands in the rotated query's own buffer: the split has
  consumed it by then and stream order guarantees it, so no second high-water
  buffer can grow during a capture.
- A folded prefill rotates **V**, not the output: `softmax(q k^T)(V Pi) = O Pi`,
  and V is `H_KV` vectors per token where the output is `H_Q`. Q and K stay
  unrotated, so RoPE and the scores are untouched.
- `activate_turboquant_backend` now sets `rotated_query` and `_hadamard16`. A
  class swap does not run `__init__`, and before this change a real activation
  would have raised `AttributeError` on the first decode -- the unit tests built
  their impls by hand and never saw it.
- `npu_turboquant_paged_attention` loses `pi_signs`: twelve arguments, and nothing
  in either kernel reads a sign. Schema, meta signature, adapter, Python caller
  and unit tests changed together.

#### UB freed in the combine

From the allocation list, per block. `TurboQuantPagedAttentionCombine::Init` and
`turboquant_host::PlanCombineUb` quote the same numbers, and
`TurboQuantLaunchContract.CombineHoldsNoRotationState` pins them:

| head_size | codec table image | codec scratch | signs | ApplyPi ping-pong | released | combine before | combine after |
|---|---|---|---|---|---|---|---|
| 128 | 20,032 B | 21,280 B | 512 B | 512 B | **42,336 B** | 44,864 B | 2,528 B |
| 256 | 40,000 B | 42,272 B | 1,024 B | 1,024 B | **84,320 B** | 88,128 B | 3,808 B |

The combine keeps 4.3% of the UB it held at head_size 256. The launch prologue
also loses the GM -> UB copies of the table image (10,000 words at head_size 256) and of the signs,
per block, per launch, and each `(token, head)` task loses a whole `ApplyPi`:
two sign multiplies, three `Gather` shuffle stages, `log2(D / 8)` block-strided
butterflies, a normalisation and their `PipeBarrier<PIPE_V>`s.

**Not A/B'd for placement.** 13.17 measured UB placement moving a kernel by 12%.
`accBuf_` shrinks by 4 * head_size bytes, which slides `stateBuf_`,
`partialBuf_` and `brcbBuf_` down; the codec and `signBuf_` were at the end of
the list and move nothing. No tick number for the combine should be believed
until that has been measured.

#### Prefill against FIA V5

> **Superseded by §7.5 / §13.7.** This describes the 270-shape prefill sweep as it
> was first built. It has been replaced by the end-to-end audit: 27 serving ranks
> across three flagship families instead of a cartesian product, chunked prefill
> instead of full-context, and `pf_fp16_write` / `pf_fp16_pipeline` dropped as
> unquantised-memory comparisons. The fold-identity check, the tiled context data,
> the 85%-of-free-HBM guard and the untimed-cast caveat all survive into the
> audit; the footprint table below is unchanged arithmetic and is still correct.
> The `ASCEND_BENCH_TQ_PREFILL_*` variables named here no longer exist — see
> §7.5's knob table for the `ASCEND_BENCH_TQ_AUDIT_*` set that replaced them.

`bench_device_950pr_turboquant` runs a second suite after the decode summary, in
its own `BenchmarkRunner` on the same stream, with its own budget (warmup 3,
10 iterations, pipeline batch 1 by default) because the largest shape is four
8,192-token sequences, 32,768 tokens in one launch. Failures are mirrored into the decode runner so the exit
code still reflects them.

A TurboQuant prefill is `PrefillNoCache`: the cache write, then full-precision
attention over the batch's own K/V. So the pipelines differ only in the write and,
folded, in the rotation of V:

| leg | launches | what |
|---|---|---|
| `pf_tq4_write` | 1 | `turboquant_reshape_and_cache` over `B * S` tokens |
| `pf_rotate_v` | 1 | `npu_turboquant_rotate_q` on V, `[B*S, H_KV, D]` -> fp32 |
| `pf_fp16_write` | 1 | `aclnnScatterPaKvCache`, the baseline's own write |
| `pf_fia_v5` | 1 | `aclnnFusedInferAttentionScoreV5`, TND, `sparse_mode` 3, the 2048x2048 int8 causal mask; V2 as a fallback |
| `pf_fp16_pipeline` | 2 | fp16 write + FIA |
| `pf_tq4_pipeline` | 2 | tq4 write + FIA -- an unfolded layer |
| `pf_tq4_folded_pipeline` | 3 | tq4 write + rotate V + FIA over `V~` -- a folded layer |

The sweep is the cartesian product of S {512, 1024, 2048, 4096, 8192}, B {1, 2,
4}, D {128, 256}, H_Q {32, 64, 128} and H_KV {1, 2, 8}: 270 shapes, every axis an
`ASCEND_BENCH_TQ_PREFILL_*` comma list. `H_KV = 1` is the MQA form DeepSeek MLA's
single latent cache takes through this operator -- a shape proxy, not the MLA
kernel. A shape whose estimated footprint exceeds 85% of free HBM is skipped
with both numbers in the reason. Each shape is built, timed and destroyed before
the next, and 1,024 tokens of random data are tiled across the context rather
than 1e9 fp16 values generated on the host.

Reported per shape: device time from ACL events (the device-events median,
pipelined when that mode is not selected), GB/s over each leg's own traffic
model, TFLOP/s for the attention legs, tq4 and folded pipeline latency over the
fp16 pipeline, and the KV-cache footprint:

| D | H_KV | fp16 B/token | tq4 B/token | kv8 B/token | fp16/tq4 | fp16/kv8 |
|---|---|---|---|---|---|---|
| 128 | 1 | 512 | 160 | 288 | 3.20x | 1.78x |
| 128 | 2 | 1,024 | 288 | 544 | 3.56x | 1.88x |
| 128 | 8 | 4,096 | 1,088 | 2,112 | 3.76x | 1.94x |
| 256 | 1 | 1,024 | 288 | 544 | 3.56x | 1.88x |
| 256 | 2 | 2,048 | 544 | 1,056 | 3.76x | 1.94x |
| 256 | 8 | 8,192 | 2,112 | 4,160 | 3.88x | 1.97x |

tq4 is the shipping layout: `D / 2` packed bytes per vector plus a scale plane of
`round_up(2 * H_KV, 8)` fp32 per token, whose burst padding is why small `H_KV`
compresses less. **kv8 is layout arithmetic only** -- a byte per coordinate and
the same scale plane. `TurboQuantCodec` is instantiated for `b = 4` alone, so
there is no 8-bit kernel and no 8-bit leg.

Before any timing, on the smallest shape, the suite checks the fold identity on
the device: FIA over `V~`, un-rotated on the host, against FIA over V, cos above
0.9999. A failure is recorded as a failure, not a skip.

**What the prefill suite does not measure, deliberately:**

- **The folded pipeline's fp32 -> fp16 cast of `V~`.** The backend does it with a
  torch cast; this suite has no header-verified `aclnnCast` prototype, so FIA
  reads a host-prepared fp16 `V~` and the cast is in no column. It is one
  elementwise pass over `B * S * H_KV * D` values.
- **`npu_fusion_attention` (`aclnnFlashAttentionScore`).** Its argument list could
  not be read from a CANN header here: the 9.1.0 toolkit image carries no
  `aclnnop` operator headers at all, only `aclnn_util.h`. The symbol is exported
  by `libopapi.so`, but a guessed prototype behind `dlsym` is undefined behaviour
  at launch, not a planning refusal. Adding the leg needs the header, nothing
  else; the FIA V5 prototype this suite uses was transcribed from one.
- **FIA's workspace traffic.** GB/s counts what each operator is handed and
  writes, the decode suite's convention.

`scripts/run_950pr_npu_baseline.sh` sets `ASCEND_BENCH_TQ_PREFILL=0` for its
decode baseline and its msprof stage: the sweep is hours, and neither stage is
about prefill.

#### What was executed, and what was not

Toolchain: `ghcr.io/fxslava/vllm-ascend-950pr:x86_64-offline`, CANN 9.1.0, g++
11.4, arch35 CAModel `Ascend950PR_9599`.

**Build.** Clean under `-DVLLM_ASCEND_TESTS_WERROR=ON`, zero compiler warnings,
in three configurations: host-only, `SOC_VERSION=Ascend950PR_9599 RUN_MODE=npu`
(benchmarks on, so the prefill suite compiles) and `RUN_MODE=sim`. The only
"warning" text in the logs is the ascendc generator's pre-existing "multiple
kernel functions per file" notice. The exported symbols carry the new
signatures: `turboquant_paged_attention_combine_impl` takes ten parameters, with
no `piSigns`, `tables` or `invSqrtLen`.

**The wheel's adapter, syntax only.** `turboquant_torch_adpt.h` and
`torch_binding_meta.cpp` compile with `-fsyntax-only -Wall -Wextra -Werror`
against the image's torch 2.10 and torch_npu headers, with a function-pointer
check pinning `npu_turboquant_paged_attention` to the twelve-argument schema
order. The full wheel was not built.

**Camodel**, 32/32 `excp_log` dumps at 0 bytes after every run:

| binary | shape | result |
|---|---|---|
| `test_sim_950pr_turboquant_kernels` | D=64, 4 / 2 heads, S=32 | **10/10 pass**, 191 s. Includes `CombineHoldsNoRotationState` and `UnrotateHeadsIsPiPerHead`. Write -> decode -> combine, un-rotated on the host, vs the CPU reference: **cos 1.000000, 72.87 dB, relL2 2.27e-4** |
| `test_sim_950pr_turboquant_multimode`, `ASCEND_TQ_SIM_MODES=kv4fp8` | D=256, 4 / 2 heads, B=1, S=16 | **pass**, 104 s. Cube split + the shared combine vs fp32 host attention: **cos 0.991817** -- the same six digits 13.22 measured with the un-rotation still in the kernel |
| `test_sim_950pr_turboquant_decode` | D=64, 4 / 2 heads, S=32, 2 splits | **pass**, 84 s. vs the CPU reference **cos 1.000000, 74.00 dB, relL2 1.99e-4**; vs exact fp32 cos 0.991182, 17.54 dB. The fp16 control still skips on the V2 `361001` refusal |

The multimode figure agreeing to six digits is the check that matters: moving
the un-rotation from the kernel's fp32 accumulator to after its fp16 store adds
at most one fp16 rounding, and Pi is orthogonal, so it cannot move the result
beyond that.

**Unit tests.** `tests/ut/attention/test_turboquant_v1.py`, **61/61**, run under
unittest in the same image with `tests/ut/conftest.py` executed first (its CPU
torch_npu fakes) and `TORCH_DEVICE_BACKEND_AUTOLOAD=0`. New: the fold against a
dense block-diagonal Pi, TP-shard commutation, fp32 / fp16 / bf16 acceptance,
the output-gate counterexample, the fold record's checks, activation per layer,
decode un-rotation computed through a Pi-implementing mock of rotate_q, and
the folded prefill's rotated V.

**The offline tool**, on synthetic sharded fp16 checkpoints: an ungated `qwen3`
config folds three projections (worst cos 0.999999976), every other tensor and
dtype comes back byte-identical, and the written record round-trips through the
backend's resolver; a `qwen3_5` config with no `attn_output_gate` key, a
`weight_scale` beside `o_proj`, an unknown model type and an already-folded
checkpoint are each refused with exit 2 and nothing written.

**Not executed:**

- **Any timing.** No physical 950PR is attached, and the device tier refuses
  the camodel by construction. Neither the decode legs since this change nor
  any `pf_*` leg has produced a number, and the FIA V5 prefill argument list --
  `sparse_mode` 3, TND, no block table, the int8 2048x2048 mask -- has never been
  through a planning call.
- **The un-rotation launch on the output**, and the rotation of prefill V, on
  the camodel. Both are `npu_turboquant_rotate_q`, which 13.22 verified on
  queries, but not on `B*S*H_KV` vectors at prefill scale, and not on a bf16
  input's `kHiLo` path.
- **A real checkpoint** through the tool, a multi-rank TP load of a folded one,
  or a vLLM model load exercising the fold record end to end.
- **UB placement A/B** for the combine; see above.

### 13.24 Corrections from silicon profiling (2026-09-16)

The four rules below come from a user-supplied profile of the TurboQuant decode on a physical
950PR. **They are reported numbers, not traces this suite produced**; they are recorded here
because they overturn design conclusions this document drew from the camodel, which models
neither HBM throughput nor dispatch cost.

| measurement (silicon, user-reported) | value |
|---|---|
| native `aclnnFusedInferAttentionScoreV5` decode | **14.16 us** |
| TurboQuant rotate -> split -> combine chain | **~850 us** |
| of which inter-kernel dispatch bubbles and stream syncs | 48.4 us |
| of which `Combine` re-reading GM partials | ~11 us |
| vector time inflated by `PipeBarrier<PIPE_V>` | 31.3 us |
| effective HBM throughput of the 32 B group-major tile read | **44.1 GB/s** (peak 1.6 TB/s) |

1. **MTE2 bursts stay contiguous and >= 128-256 B.** The group-major read of 13.9 fragments
   every tile into 32 B descriptors. A permutation belongs in the vector unit: a
   `DataCopy(UB, UB, DataCopyParams)` block move (lowered on `PIPE_V` as `CopyUbufToUbuf` in
   CANN 9.2.0's `dav_3510/kernel_operator_data_copy_impl.h`), `DeInterleave`, or a `Gather`
   from contiguous UB. Never through fragmented GM access.
2. **Both AIV subcores work in decode.** Heads split by `GetSubBlockIdx()`: subcore 0 takes the
   first `ceil(M / 2)` heads of a task and subcore 1 the rest (the split `dualDstCtl = 0b01`
   makes in M), and both GEMM products are Fixpiped with `dualDstCtl = 0b01`. 13.14.1's per-subcore UB fact is why dual-destination is needed. It is
   not a reason to gate.
3. **No Split-K and no combine at B=1, S <= 4096.** One fused launch over grid `[B, H_Q]`:
   online softmax in UB, direct GM writeback of the output token, no workspace. Above 4096 the
   reduction runs inside the same launch after an `AscendC::SyncAll`, never as a second
   host-dispatched kernel.
4. **No intra-loop `PipeBarrier<PIPE_V>` between arithmetic ops.** The stated reason is that
   ccec tracks intra-pipe RAW/WAR hazards on the vector registers. Explicit event pairs stay
   at the cross-pipe interfaces (MTE2 -> V, V -> MTE3, FIX -> V).
   **Open question:** CANN's own `sigmoid_v100_impl.h` puts `PipeBarrier<PIPE_V>` between
   dependent vector ops. The fused kernel therefore compiles the barriers behind a template
   switch, and a test-only barriered instance runs as an A/B against it (13.25).

### 13.25 The fused Cube decode (`TurboQuantFusedDecode`)

> **See 13.26.** The tail-tile mask path and the in-launch reduction listed as not executed below
> now run on the camodel, and the kv4fp8 instance's ingest and softmax changed shape.

`turboquant_mm_kernels.cpp` now carries `TurboQuantFusedDecode<MODE, scalar_t, VEC_BARRIERS>`
and `turboquant_mm_fused_decode_impl`, which apply 13.24's four rules to the Cube multi-mode
decode. The legacy `TurboQuantCubeDecodeSplit` + combine stays, unchanged and barriered, as the
A/B reference. The dead fp16 reference chain (`TurboQuantFp16DecodeSplit`,
`TurboQuantPlainCombine`, `turboquant_fp16_decode_impl`) is gone. It had no caller.

| rule | what the kernel does |
|---|---|
| 1. wide bursts | One contiguous row-major GM read per tile plane (16 * D/2 B at H_KV = 1; one head's D/2 B per row at D >= 256; the whole row plane into UB otherwise), unpacked in UB and permuted to NZ with a UB -> UB `DataCopyParams` block move, then one MTE3 burst into L1 |
| 2. both subcores | A task is (token, kv head, split, head chunk). Subcore 0 owns the first `ceil(M/2)` heads and stages the K plane, subcore 1 owns the rest and stages the V plane. `GemmScores<true>` / `GemmContext<true>` Fixpipe with `dualDstCtl = 1` over an even M. No `IsPrimarySubcore()` anywhere in it |
| 3. one launch | Tokens with `context_len <= fusedContextLimit` run one split, normalise in UB and write the output token to GM. Longer ones write partials that the same launch reduces after `AscendC::SyncAll` |
| 4. no intra-loop barriers | Every dependent vector op goes through `VecBarrier<VEC_BARRIERS>`, `kFusedVecBarriers = false`. Cross-pipe edges keep explicit `SetFlag`/`WaitFlag` events |

Host planning is `PlanFusedDecode` in `common/turboquant_launch.cpp`. The helpers it shares with the
legacy split (`BroadcastMul`, `UnpackAffine`, `CastToOperand`, `GemmScores`, `GemmContext`) took a
template switch whose default is the old behaviour, so the legacy instance compiles as before.

The shipping AIV decode (`turboquant_kernels.cpp`) took rule 4 in the same change.
`AccumulateTile`, `RowSums` and `WriteOutput` switch 26 vector barriers behind
`kPagedAttentionVecBarriers = false`, plus the barriers inside the three broadcast helpers they
call. `WriteOutput`'s GM writeback uses a V -> MTE3 / MTE3 -> V event pair instead of two
`PIPE_ALL` barriers. `ComputeSplit`'s initialisation barrier stays, because it orders two
overlapping `Duplicate` writes. The combine and the reshape keep theirs.

**Executed (camodel, Ascend950PR_9589, one process each, every `excp_log.dump` 0 B):**

| test | shape | result |
|---|---|---|
| `test_sim_950pr_turboquant_fused` | kv4fp8, H_Q 4, H_KV 2, D 256, block 64, S 64, aiv 64 | fused: 1 launch, splits 1, block_dim 2, 2 heads/task, no sentinel left, cos 0.999407 vs fp32. **0 of 1024 elements differ from the legacy split + combine** (max abs err 0) |
| `test_sim_950pr_turboquant_kernels` | AIV decode, barriers off | 10/10; decode vs CPU reference cos 1.000000, SNR 72.87 dB, relL2 2.27e-4, the same to printed precision as the barriered build |

All three tiers build clean under `-Werror` (4 / 36 / 41 targets). The msprof trace harness's Cube
leg now enqueues `TQ_FusedDecode` (one launch) in place of `TQ_DecodeSplit` + `TQ_Combine`.

**Not executed:** kv3fp4 and kv5fp8 through the fused kernel (compiled only); any shape with a
tail tile (`context_len % 64 != 0`, the `Min` mask path); head chunking (`H_Q / H_KV > 16`); the
in-launch reduction (`context_len > 4096`, which the camodel policy keeps off the simulator); and
anything on silicon. The camodel wall times in the test output are not latency.

### 13.26 Filled MIX blocks, the ingest ring and the head-batched softmax

Three changes aimed at the fused decode's silicon latency: 126.34 us per decode step against a
~34 us native `FusedInferAttentionScore` target, both user-reported. **Nothing here was timed.**
The camodel shows that the outputs are correct and that the pipes neither hang nor fault. It
cannot show whether the changes pay off.

The two kernel changes apply to the kv4fp8 instances only (`kBatchedCubeDecode<MODE>`). kv3fp4 and
kv5fp8 still compile the per-head, lock-step decode of 13.25 and are still not executed.

#### 1. Host tiling fills the MIX blocks

`PlanFusedDecode` takes a `FusedSplitPolicy`. Under the default, `kFillBlocks`, a context inside
the fused limit is split anyway whenever every task can still have a block of its own:

```
K = min(kMaxSequenceSplits, blocks_per_seq, mix_blocks / unsplit_tasks)     split when K > 1
```

`unsplit_tasks` is `tokens * kv_heads * head_chunks`, and splitting does not change the chunking
(the planner comment says why). A split grid returns `fused_context_limit = 0`, so `IsFused()` is
false, `NeedsReduction()` is true, and the same launch reduces the partials after
`AscendC::SyncAll`. Callers now pass `grid.fused_context_limit` to the kernel, not the constant.
A caller that passes the constant with a split grid still gets the right answer, because the
kernel then treats the context as fused and returns early from every task with `split > 0`. It
just wastes the blocks. `kContextOnly` is the old behaviour.

| shape, aiv 64 | before | after |
|---|---|---|
| B 1, H_Q 16, H_KV 8, 32 blocks | 8 tasks, 8 blocks | 4 splits, 32 tasks, 32 blocks, limit 0 |
| DeepSeek-V4-Flash, B 1, S 2048, block 128 (the trace's Cube leg) | 8 tasks, 8 blocks | 4 splits x 8 two-head chunks, 32 blocks |
| B 4 or S > 4096 | unchanged | unchanged |

`test_host_turboquant_tiling` pins both rows and checks the policy's contract over the whole sweep.
**Open on silicon:** the reduction reads every partial back from GM, `tokens * H_Q * K * (D + 16)`
floats. Whether 4x the parallel tiles buys more than that costs at S <= 4096 has not been measured.

#### 2. The ingest is a two-slot ring

`kvBuf_` and `scaleTileBuf_` each have `kCubeSlots` slots, and tile t uses slot `t % 2`. That is
the same slot as its L1 operands. The ingest is split in two:

- `ReadTile` is the MTE2 half. It reads the tile's plane and scale lanes into the slot, then sets
  an MTE2 -> V event taken with `AllocEventID`. `FetchEventID` only peeks at the first free id and
  would give two in-flight reads the same id.
- `StageSlot` is the vector half. It waits on that slot's event and releases the id, then unpacks,
  permutes and stages the tile into L1 with MTE3.

Tiles 0 and 1 are read before anything is staged. Tile t + 2 is read right after tile t's softmax,
behind a V -> MTE2 edge, so the read overlaps tile t + 1's stage, both GEMMs and the accumulation.
The scale lanes need the second slot as much as the packed plane does, because tile t + 2 lands
in tile t's slot. The same edge is what lets an id be reused: the read that takes a released id is
always issued behind the vector work of the stage that released it.

Every AIV task now starts with a V -> MTE2 edge as well. Once the scalar readback below was gone,
nothing drained the vector unit between tasks. The next task's query and tile reads could then
land under vector work of the previous task that still reads those buffers.

This differs from 13.13's rejected attempt in that it adds no queue object. The buffers are plain
`TBuf`s and the ids come from the pipe's event pool, which is the "hand-rolled ping-pong" 13.13
suggested. UB grows by one tile plane and one scale tile, 10 KB at D 256. The whole-row burst
layout (`packedBytes < 128` with H_KV > 1) now holds two planes.

#### 3. The softmax batches the heads and never reads back to the scalar core

Every per-head scalar of the kv4fp8 softmax is now kept broadcast over its head's 8-lane block.
That has three effects:

- One `m * 8`-lane op advances all `m` heads.
- The same state field is the block operand of a row op
  (`BinaryRepeatParams{1, 1, 0, 8, 8, 1}` over `m` repeats).
- Lane `j * 8` still holds head j's value, so `WriteNormalizedHeads` and `WritePartials` are
  unchanged.

Two scalar stalls are gone:

| stall | where | frequency |
|---|---|---|
| `PipeBarrier<PIPE_V>` + `qScale.GetValue` for `qScaleInv_` | `PrepareTask` | per task |
| Level-2 `ReduceSum<float>`, which on dav_3510 ends in a V -> S event and a scalar load of the sum | the softmax | **per head, per tile** |

The row sums and maxima are now `ReduceRepeat<SUM/MAX>`. That makes one register reduce per row,
one lane per head, with no readback, and it matches the per-head bits exactly. The reciprocals
stay in UB as head blocks. Per tile, the kv4fp8 softmax issues 21 vector ops whatever the head
count (22 with the tail mask); before, it issued about 21 per head plus a stall per head. The
query's row max stays a per-head Level-2 `ReduceMax`, because a D 256 row is wider than one
register. That reduce has no readback.

#### Harness

`test_sim_950pr_turboquant_fused` runs four shapes. Each runs through the barrier-free and the
barriered instance, and each output is hashed with FNV-1a over its fp16 bit patterns:

| case | shape | covers | golden (b48ed2951) |
|---|---|---|---|
| (a) | H_Q 4, H_KV 2, S 64, block 64 | one tile | `0x6176461416358ec1`, cos 0.999407 |
| (b) | H_Q 4, H_KV 2, S 256, `kContextOnly` | four tiles in one task, the slot-free edge | `0x470dad36e6708da5`, cos 0.999400 |
| (c) | (b) under `kFillBlocks` | 4 splits on 8 blocks, the in-launch reduction | `0xda6a2c77e20ff4a1`, cos 0.999400 |
| (d) | H_Q 8, H_KV 2, S 120, planned for 4 AIVs | a masked tail tile (`valid` 56), two heads per subcore | `0xcc1fcdfed39b48e5`, cos 0.999517 |

The goldens were recorded from b48ed2951 before any of the three changes, with (c) planned at
`fused_context_limit = 0`. That produces the same grid `kFillBlocks` does. (c) must also agree
with (b) to cos >= 0.999999, and must stay within 1e-6 of (b)'s cos against fp32.

**Executed (camodel `Ascend950PR_9589`, aiv 64, `quay.io/ascend/vllm-ascend:v0.26.0rc1-a5`,
one process per run, 32 `excp_log.dump` files, all 0 B):**

| build | (a) | (b) | (c) | (d) |
|---|---|---|---|---|
| b48ed2951 + harness | golden | golden | golden; vs (b) cos 1.000000000, fp32 cos 0.999400043 vs 0.999400044 | golden |
| + fill policy | golden | golden | golden, now from `kFillBlocks` | golden |
| + ring + batched softmax | golden | golden | golden | golden |

Fused matched barriered bit for bit in every run. The ring and the batched softmax were verified
together, not one at a time. All three tiers build clean under `-Werror` (6 / 45 / 40 targets).
`test_host_turboquant_tiling` passes 6/6 and `test_host_turboquant_fidelity` 20/20. No run needed
the teardown watchdog.

**Not executed:** anything on silicon, the msprof trace and the bench included (they build, and
they plan and launch the filled grid); kv3fp4 and kv5fp8; the whole-row burst layout;
3 to 8 heads per subcore; head chunking; contexts above 4096.

### 13.27 CAModel cycle profile of 13.26 (2026-09-17)

Silicon was not available, so this profiles the camodel. Each profiled process ran one fused decode
launch and nothing else, with the rotated query drawn on the host so no rotation launch ran either.
Cycles come from the per-core `coreN.{veccore0,veccore1,cubecore0}` logs in the working directory:

- The `instr_log` records each instruction's issue cycle and unit class. The gap to the next
  issued instruction is charged to the unit.
- The `ccu_log` records every issue attempt. Retries of one instruction id are its blocked cycles,
  and they are subtracted from that charge.
- `WAIT_FLAG_DEV` is issued on the cycle its cross-core flag is released, which gives a per-tile
  handshake timeline.

The `profile_*_log0.toml` tables stayed empty and were not used. A launch's span (first to last
instruction over all cores) was reproduced exactly by repeated runs; `Total tick` moved by up to 7.
**These are camodel cycles, not silicon latency.**

"Baseline kernel" below is the current tree with `turboquant_fused_decode.cpp` and
`turboquant_vector_service.h` taken from b48ed2951. The planner and the harness are the same in
both builds, so the grid is a separate variable. kv4fp8, D 256, block 64, 64 AIVs, launch span in
cycles:

| shape | grid | baseline kernel | current kernel | delta |
|---|---|---:|---:|---:|
| H_Q 16, H_KV 8, S 256 | unsplit: 8 blocks, 4 tiles per task | 35,339 | 35,242 | -0.3% |
| H_Q 16, H_KV 8, S 256 | fill: 32 blocks, K 4, 1 tile per task | 22,142 | 23,551 | **+6.4%** |
| H_Q 16, H_KV 1, S 256, one block | 8 heads per subcore, 4 tiles | 59,400 | 50,949 | **-14.2%** |
| H_Q 16, H_KV 8, S 1024 | unsplit: 8 blocks, 16 tiles per task | 113,695 | 100,860 | -11.3% |
| H_Q 16, H_KV 8, S 1024 | fill: 32 blocks, K 4, 4 tiles per task | 41,505 | 43,069 | +3.8% |

**The grid is the large lever.** Filling the blocks cuts the span 1.60x at S 256 and 2.74x at
S 1024 on the baseline kernel, and 1.50x and 2.34x on the current one. From the b48ed2951 state
(unsplit grid, old kernel) to the current state (fill grid, new kernel), the span falls 1.50x at
S 256 and 2.64x at S 1024.

**The two kernel changes pull in opposite directions at one head per subcore.** An ablation at
S 256 built each change on its own:

| | unsplit | fill | 8 heads per subcore |
|---|---:|---:|---:|
| ring only | -5.3% | -0.8% | -0.7% |
| batched softmax only | +5.5% | +9.0% | -13.0% |
| both (the current kernel) | -0.3% | +6.4% | -14.2% |

- **The ring pays in steady state.** On the unsplit S 256 grid (block 0), a tile's slot is staged
  4,048 cycles after the previous tile's probabilities, against ~5,330 before. The tile period falls
  from 6,521 to 5,434 cycles (-17%). At 8 heads per subcore it falls from 11,769 to 9,266 (-21%).
  The ring issues each MTE2 read 4-8 K cycles ahead of its stage, where the baseline issues it
  1 cycle ahead.
- **Most of that is lost on short tasks.** The ring's prologue costs more, and a 1-tile task (the
  fill grid at S 256) is all prologue.
- **The batched softmax only pays with more than one head per subcore.** At 8 heads the softmax
  falls from 5,275 to 3,793 cycles per tile. At one head it rises from 920 to 1,129: its `Brcb` and
  `ReduceRepeat` calls cost more than the per-head path's readback. The readbacks do go away:
  V -> S events fall from 112 to 32 on the unsplit grid.
- **The score GEMM is 257 cycles per tile.** Staging the next tile's operands on the vector pipe,
  4-7 K cycles, is what the Cube waits on.

**Outputs.** At S 256 all four builds give one bit pattern per shape and grid. **At S 1024 the
baseline and current kernels differ bitwise** (`0xbf9622c0a6ff6d35` vs `0xead7bf890165bacd`), while
cos against fp32 is 0.999414 for both. Within each kernel, the fill grid reproduces the unsplit grid
bit for bit. The source of the difference was not investigated; the 13.26 goldens only cover
S <= 256. Every run left all 32 `excp_log.dump` files at 0 B.

**Follow-ups this points at:**
- Take the batched softmax only when `heads.mine > 1`. At one head the per-head path is 5-9%
  faster, and the ring alone keeps its gain. (Implemented and measured in 13.28, where it did not
  reproduce: on cases (a) to (c) the gate cost 2-25%, and it was dropped.)
- Find the S 1024 divergence before extending the goldens.
- Treat the vector-pipe unpack as the critical path at every shape profiled.

### 13.28 NZ-tiled kv4fp8 storage, the kernel writer round trip, a dropped softmax gate (2026-09-17)

Three changes follow up on 13.27, all kv4fp8 only:
- The NZ permutation moved out of the decode and into the cache write.
- A one-head softmax gate was measured, found to be a regression, and dropped.
- A fifth fused case now runs the kernel cache writer.

kv3fp4 and kv5fp8 keep their code paths. Their object code was not compared, and they were not run.

#### 1. The cache is stored in the Cube's order

`kStoresNzTiles<MODE>` (`turboquant_mode.h`) is true for kv4fp8, and `NzTiledPackedByte`
(`turboquant_layout.h`) defines the layout:

- A physical block is `[64-row tile][kv head][32 B column group][tile row][byte]`, so the packed
  plane of each (tile, kv head) is the contiguous `[D/64, 64, 32]` fractal image.
- The cache and the scale plane keep their sizes, and the scale plane stays row-major.
- The block size must be a multiple of 64, which the Cube decode already required.

The three kernel paths:

- **Write.** `TurboQuantModeReshapeAndCache::WriteNzTiled` issues one strided UB -> GM burst per
  plane per token, `DataCopyParams{H_kv * D/64, 1, 0, 63}`. Consecutive (kv head, group) cells of one
  tile row are exactly a tile apart. The 32 B cells are paid once per token.
- **Read.** `ReadPlane` issues one contiguous GM -> UB burst of `64 * D/2` B (8,192 B at D 256) at
  `(row * H_kv + kvHead * 64) * D/2`, whatever the block size. `TurboQuantTileBurst` is no longer
  used on this path, and nothing reads GM below 128 B (13.24 rule 1).
- **Unpack.** The `int4x2` cast, the `DeInterleave` and the plane expand all work byte by byte.
  `UnpackPlaneToL1` therefore writes each chunk's low nibbles straight onto NZ groups `[0, D/64)` and
  its high nibbles one tile plane later, onto `[D/64, D/32)`. The 16 UB -> UB permute copies per tile
  are gone, and so is the separate 8 KB operand buffer. One MTE3 burst still stages the tile.

**Pinned on the host.** `TurboQuantNzTiledCache.TilesAreContiguousNzImagesAndTheWriterBurstHitsThem`
(`test_host_turboquant_fidelity`) covers H_kv 1/2/8, D 64/128/256 and block 64/128. It checks three
things:
- the layout is a bijection;
- every cell lies in its (tile, kv head) read window, at exactly the NZ offset (`NzOffset`) the
  unpack gives it;
- the writer's strided burst lands on every cell.

`BuildMirroredKvCache` now writes NZ tiles. Its own test moved to block 128, D 128 and S 150, so the
context crosses a tile and a block.

#### 2. Case (e): the kernel writer round trip

`TurboQuantFusedDecode.DecodesTheCacheTheKernelWriterWrote` is (a)'s shape with
`FusedShape::kernel_writer`. The scenario works in four steps:

1. It draws **independent** vector halves (`BuildMirroredKvCache(..., mirror_halves = false)`), so a
   swapped nibble lane changes the data.
2. It unrotates the host's dequantised vectors, converts them to fp16, and has
   `turboquant_mm_reshape_and_cache_impl` (kv4fp8) write them into a zeroed cache.
3. The writer rotates them back. Each coordinate sits half a quantiser step from both neighbouring
   thresholds, so it lands on the level it came from.
4. The test then decodes that cache.

The writer's scale is the RMS of the vectors it was given, not of the draw, so the exact-attention
reference is rescaled to the scales the writer stored.

**Executed (the camodel and container of 13.27):**

| check | result |
|---|---|
| packed GM bytes vs the host encoder | **0 of 131,072 differ**; with the two nibble lanes of every byte swapped, 29,700 differ, so lane 0 is the low nibble, as `MirroredPackedPair` assumes |
| scale plane | 2,048 lanes, 0 pad or unwritten-slot writes, max rel err 1.95e-4 against the host RMS |
| decode | fused = barriered bit for bit, no sentinel left, cos 0.999576 vs fp32, golden `0x9ffe02efde1506cf` (now pinned) |
| exception dumps | 32 files, 0 B |

This is the first execution of the NZ-tiled writer. The multimode suite, which also runs the writer,
was out of scope. Its kv4fp8 figure in `docs/ascend_c_optimization_skills.md` §4.5 predates this
layout.

#### 3. The one-head softmax gate: measured, and dropped

13.27's follow-up proposed taking the per-head softmax whenever a subcore owns one head. It was
implemented as `BatchesHeads = heads.mine > 1` and measured against the same tree with the gate off.
Gate on against gate off:

| case | span | Cube idle on the probability flag |
|---|---:|---:|
| (a) | +19% | +45-76% |
| (b) | +25% | +45-76% |
| (c) | +1.6% | +45-76% |

The gate was dropped: kv4fp8 batches the softmax at every head count, which is HEAD's code. The
per-head path's Level-2 reduces cost more than the row repeats on this camodel. 13.27's number came
from a single-change ablation on H_Q 16 / H_KV 8, a shape not run here.

#### 4. Case (d) regressed, and why

With the gate in (unused in (d), which has two heads per subcore), (d) was 4.4% slower than HEAD
although its tile stage was 30% faster.

**The mechanism.** The softmax instruction lists were identical in both builds, yet the first tile's
softmax took 3,722 cycles against 2,158.

- **Two kinds of gap.** An AIV core has *flag waits*, which are gaps that end in `WAIT_FLAG_DEV`.
  Every other gap of 200+ cycles is a *stall*: neither its scalar nor its vector unit issues
  anything.
- **Where the stalls end.** Most end on a `PUSHQ VF` record, a synchronous vector-function call such
  as the `ReduceRepeat` loop. The record's `vf_execute_time` is up to ~450 cycles, while the body's
  instructions run in ~40.
- **What was ruled out.** No cache miss, hazard retry or flag explains the dead time.
- **How it moved.** Across builds, the same 332 calls with the same bodies averaged 214 (HEAD), 238
  (gated), 220 and 214 (below) cycles. Stall time on (d)'s AIV cores moved with them: 18,892 ->
  28,130 -> 16,960 -> 15,543 core-cycles.

So the regression is camodel vector-function dispatch latency, and code layout moves it. It is not
work the kernel added.

**Ruled out as causes:**
- **UB addresses.** A control build kept HEAD's addresses and gave (d) 22,476.
- **The tail mask and the ring.** For a two-tile task the ring issues no mid-task read and no
  slot-free wait, and tile 1's stage waits only on its own read event, which was set at task start.
- **The MTE2/MTE3 timeline.** It had the same shape in both builds, with no DMA inside the stalled
  softmax.

**What cleared it.** `BuildTailMask` did carry four unconditional `PipeBarrier<PIPE_V>` calls between
dependent ops on one buffer. They became switchable in the build where (d) recovered, together with
the gate removal. These runs cannot separate those two edits. 13.29 then removed the switch as well.

#### Results

Each test ran in its own process, and all 32 `excp_log.dump` files stayed at 0 B.

- **Goldens.** Every build reproduces (a) `0x6176461416358ec1`, (b) `0x470dad36e6708da5`,
  (c) `0xda6a2c77e20ff4a1` and (d) `0xcc1fcdfed39b48e5`.
- **Fidelity.** Cos is unchanged: 0.999407 / 0.999400 / 0.999400 / 0.999517.
- **A/B twin.** Fused matches barriered bit for bit in every case, and (c) agrees with (b) at cos
  1.000000000.
- **Builds.** Clean under `-Werror` with zero compiler warnings: host 6 targets, sim 37, npu 42,
  the same target sets as HEAD.
- **Host tests.** Pass: fidelity 21/21, tiling 6/6.

**How the cycles were measured.** `build/nz_profile.py` (gitignored) reads the per-core `instr_log`
and splits a process's launches at the Cube prologue. `Total tick` is **not** a launch metric: the
camodel keeps ticking while the host works between launches. The columns are:

- **Span.** From the prologue to the last instruction of the launch, over all cores.
- **Flag release cycles.** Taken from `WAIT_FLAG_DEV` (flag id in XT), which is issued on the cycle
  its flag is released.
- **Tile stage.** From the vector unit's MTE2 -> V wait that opens `StageSlot` to the MTE3
  `MOV_UB_TO_L1` burst of the tile.
- **Stalls and VF latency.** `build/nz_stalls.py` and `build/nz_vf.py`.

A repeated process reproduced every figure exactly. Builds that differ only in code layout still move
a span by up to ~3% through the dispatch latency above, so smaller differences are noise.

Fused launch, cycles. "Shipped" is this section plus 13.29.

| | case | HEAD 41c2012e2 | NZ + one-head gate (dropped) | shipped |
|---|---|---:|---:|---:|
| span | (a) S 64, 1 head/subcore | 14,933 | 15,413 | **13,558 (-9.2%)** |
| | (b) S 256 unsplit, 4 tiles | 31,082 | 30,993 | **24,888 (-19.9%)** |
| | (c) (b) on the fill grid | 23,132 | 21,360 | **21,041 (-9.0%)** |
| | (d) S 120, 2 heads/subcore | 21,628 | 22,580 | **18,460 (-14.6%)** |
| | (e) (a) from the kernel writer | - | - | 12,824 |
| tile stage, mean | (a) / (b) / (c) / (d) | 2,757 / 3,128 / 2,741 / 3,234 | 1,762 / 2,475 / 1,731 / 2,264 | 1,766 / 2,219 / 1,778 / 2,108 |
| first slot ready | (a) / (b) / (c) / (d) | 8,859 / 9,466 / 8,770 / 9,784 | 7,560 / 8,150 / 7,542 / 8,561 | 7,622 / 8,318 / 7,776 / 8,583 |
| Cube idle on slot flags | (a) / (b) / (c) / (d) | 12,169 / 35,775 / 49,877 / 22,355 | 10,807 / 29,003 / 43,074 / 19,733 | 10,783 / 24,481 / 44,301 / 16,442 |
| Cube idle on probs | (a) / (b) / (c) / (d) | 4,451 / 10,923 / 17,426 / 7,869 | 5,107 / 14,816 / 16,484 / 12,379 | 2,853 / 9,512 / 11,567 / 6,625 |
| AIV instructions | (a) / (b) / (c) / (d) | 36,764 / 108,688 / 153,072 / 62,964 | 32,220 / 94,772 / 135,040 / 60,872 | 35,388 / 104,348 / 147,684 / 60,588 |
| probs(t) -> slot(t+1) | (b) / (d) | 4,004 / 4,328 | 3,106 / 3,754 | 2,342 / 2,193 |
| tile period | (b) / (d) | 5,358 / 6,264 | 4,998 / 6,534 | 3,696 / 4,110 |

**What moved, and what did not:**
- **Faster.** A tile stages in 29-36% fewer cycles, the first tile reaches the Cube 11-14% sooner,
  and the tile period falls 31-34%.
- **Less idle.** The Cube's idle wait on the slot flags falls 11-32%, but it does not collapse.
- **Target missed.** Staging did not reach the 800-cycle target. What remains (1.7-2.2 K cycles) is
  the unpack itself: two chunks of eight vector instructions each, over 4,096 packed bytes (8,192
  nibbles) per chunk.
- **Dominated by setup.** The first-slot figure is still mostly the ~5-6 K cycles before the first
  stage: `Init`, the query, the cursor and the reads.

**Not run, by scope:** S 1024, the H_Q 16 / H_KV 8 and 8-heads-per-subcore shapes of 13.27, kv3fp4 /
kv5fp8, the multimode suite, and anything on silicon.

**Follow-ups this points at:**
- The unpack is now the staging cost. Two structural options each remove about half of it:
  - Fold the `+0.5` level offset into the Cube product as a per-head additive correction. This
    changes the bits, so the goldens would need re-recording.
  - Pack adjacent NZ columns per byte instead of plane-split, so `Cast<half, int4b_t>` emits NZ order
    and `DeInterleave` goes away. `DataCopy`'s 32 B granularity cannot scatter the 16 B cells such a
    layout needs, so the writer would need another mechanism.
- Take the pre-stage setup (~5-6 K cycles of the first slot) apart the same way the stage was.
- Measure vector-function dispatch latency on silicon before trusting camodel spans closer than ~3%.

### 13.29 Intra-pipe barriers removed from the kv4fp8 decode and writer (2026-09-17)

The user's rules, from silicon profiling (+31.3 us attributed to vector barriers):
1. No `PipeBarrier<PIPE_V>` between vector arithmetic; ccec tracks RAW/WAR/WAW itself.
2. Cross-pipe hand-offs use `HardEvent` set/wait.
3. A vector barrier stays only for an aliased or reinterpreted buffer, or for back-to-back reductions
   sharing one scratch buffer.

**Audit, before the change:**

- **The decode.** Its 67 switchable `VecBarrier` calls in `turboquant_vector_service.h` were already
  compiled out of the production kernel (`kFusedVecBarriers = false`). Only the barriered A/B
  instance issued them.
- **Unconditional vector barriers on kv4fp8's paths:**
  - 2 in the decode (`BeginTask` and `TurboQuantPartialReducer::Reduce`), each separating two
    overlapping `Duplicate` writes;
  - 4 in `BuildTailMask`, made switchable in 13.28;
  - about 37 per encoded vector in the cache writer (the rotation, `Compute`, `Encode` and
    `PackAffinePlane`). The purge removes 29 of them and keeps 8 plus the FWHT's ping-pong barriers.

**What changed:**

- **`turboquant_vector_service.h`.** All 67 `VecBarrier` calls were deleted, from every chain:
  softmax, logits, query quantisation, accumulation, tail mask and close-out. The two
  overlapping-write barriers stay (rule 3, aliased write). The barrier before the per-head path's
  scalar readback (`QuantizeQueryHeads`, kv3fp4 / kv5fp8 only) is untouched.
- **`turboquant_codec_mx.h`.**
  - **Unpack.** `UnpackAffine` and `ExpandNibblePlane` issue barrier-free.
  - **Encode.** Its chain barriers became `ChainBarrier()`, which is empty for kv4fp8 and unchanged
    for the codebook modes. Four barriers stay unconditional, each at a reinterpretation:
    - the reduce's float scratch becomes int32 bins;
    - float bin lanes become uint32;
    - uint32 bin lanes become int32, and are written as float by the next pass;
    - half levels in `scratch_` are written as float by the next vector's encode.
  - **Pack.** `PackAffinePlane` keeps only that last one.
  - **Cast.** `CastToOperand` keeps its switch, because the codebook modes' unpack calls it
    barriered.
- **Writer (`turboquant_fused_decode.cpp`), for kv4fp8 (`kBarrierFreeWriter<MODE>`):**
  - The rotation runs `ApplyPi<false>`, the same barrier-free rotation the production query rotation
    uses. The FWHT's ping-pong barriers stay, as a documented race.
  - The `Cast -> ApplyPi` and `Gather -> EnQue` barriers are gone.
  - The `Duplicate -> Gather` barrier over overlapping scale lanes stays.
  - `Init` hands the signs over with `SyncMte2ToVector()` instead of `PipeBarrier<PIPE_ALL>`.

**Cross-pipe events.** Every GM -> UB, UB -> L1 and UB -> GM hand-off in the decode already uses
`MTE2_V` / `V_MTE3` / `MTE3_V` events, and the writer's queues use their own. **`HardEvent::FIX_V`
is not usable here:** `kernel_event.h` lowers `FIX_V` / `V_FIX` only under `__NPU_ARCH__ == 5102`.
On arch35 (3510) both fall to the empty `default:` assert and emit nothing, the same silent trap as
an unsupported `Cast` (13.9). The Fixpipe product also lands on another core, the AIV subcores. So
the decode keeps the cross-core flags the AIC sets on `PIPE_FIX` (`kFlagScoresReady`,
`kFlagContextReady`), which is the hand-off that works on this part.

**Results (same camodel, one process per test, 0 B dumps everywhere):**

- **Correctness.** All five goldens hold, fused = barriered in every case, and case (e)'s written
  cache is still byte-identical to the host encoder, with the same scale error.
- **Decode spans.** They moved by -1.8% to +1.4% against the build before the purge. That is
  expected, since its removed barriers were already compiled out, and within layout noise.
- **Writer.** The kv4fp8 writer launch (64 tokens, 64 AIV cores) went from 29,237 to 28,814 cycles
  (-1.4%). It issued the same 7,424 vector-function calls with the same mean latency (264 -> 265).
  Its barrier-class instructions fell by only 128 across 256 encoded vectors, so most of the 29
  barriers per vector emitted nothing to begin with. The writer's time is vector-function dispatch.

The +31.3 us silicon figure could therefore not be reproduced as camodel cycles. Whether the purge
buys it back needs the device trace. The barriered decode instance now differs from the fused one
only inside the broadcast and cast helpers that still take the switch. It remains a (weaker) A/B.

### 13.30 Tree-wide barrier hygiene and the retired barriered twin (2026-09-17)

The 13.29 rules applied to every TurboQuant kernel source, not only the kv4fp8 decode and writer:
`turboquant_paged_attention.cpp`, `turboquant_rotate_q.cpp`, both codecs, the cube and vector services.

**Barriers.** Vector barrier call sites (`PipeBarrier<PIPE_V>`, `VecBarrier<>`, `ChainBarrier`) went from
132 to 34 across the tree. The switch itself (`VecBarrier`, every `VEC_BARRIERS` template parameter,
`ChainBarrier`, `kBarrierFreeWriter`, `kPagedAttentionVecBarriers`, `kRotateQVecBarriers`) is gone. A kept
barrier sits at one of:
- a UB reinterpretation (float/int32/uint32/half/bf16 views of one buffer: the bin lanes, `FloorInPlace`,
  the offset gathers, the half scratch between vectors or bands);
- an overlapping write (`Duplicate` then an overlapping `Duplicate`/`Gather`; the in-place low-digit
  `Gather` of the codebook pack);
- the FWHT ping-pong (butterfly passes, the copy back into `x`);
- the vector -> scalar `GetValue` readback of the per-head query path.

Each carries a one-line comment naming its case. The `PIPE_ALL` init barriers, the `PIPE_FIX` barrier
after each Fixpipe and the probe's `PIPE_MTE1` variant are unchanged and now commented. The codebook
writer's `Init` hands its signs over with `SyncMte2ToVector()` like kv4fp8's. `FloorInPlace` takes its
entry barrier itself instead of relying on the caller's.

For kv4fp8's fused decode and writer the barrier set is the one 13.29 left: the fused instance issued no
switchable barrier, and the writer's encode keeps the same four reinterpretation barriers. The shipping
AIV path (`Quantize4Bit`, `Dequantize4Bit`, the rotation, `AccumulateTile`) and the kv3fp4 / kv5fp8
codec chains lost their arithmetic barriers here; **none of those were executed after this change** (the
test scope is fused (a)-(e), kv4fp8 only).

**The barriered twin is retired.** With no switch left, `turboquant_mm_fused_decode_barriered_kv4fp8_half`
compiled to the same launch, so it and `turboquant_mm_fused_decode_barriered_impl`, `RunBarriered` and the
fused-vs-barriered comparison were removed. The goldens are the bit-exact reference. Case (a) is renamed
`DecodesASingleTileToItsGolden`.

**Structure.** Helpers split out with `Stage*` / `Compute*` / `Sync*` names (e.g.
`ComputeSoftmaxAndStageProbs`, `ComputeRowSoftmaxStep`, `StageNzTileToL1`, `ComputeTile`), read-only
tensors and extents taken as `const`, `TurboQuantTaskSpan` for the tile pipeline, shared `kFixpipeToUb` /
`CeilDivU16` / `AlignUp` / `kSubBlockSyncMode` in `turboquant_common.h`, named `CubeLoadVariant`s for the
probe, and `ASCEND_TQ_`-prefixed declare macros that are `#undef`ed after use. The dead `unpack_tq4_to_fp8`
/ `unpack_tq5_to_fp8` wrappers were removed.

**Results (camodel, one process per test):** host / sim / npu clean under `-Werror` with target sets equal
to 13.29's. Fused (a)-(e) pass: all five goldens, cosines equal to the recorded values, (c) vs (b) cos
1.000000000, (e) 0 of 131,072 written bytes differ (29,700 with lanes swapped), 0 B exception dumps.

**Spans, not conclusive.** Launch spans read higher than 13.29's (a +3.8%, b +19%, c +45%, d +7.6%,
e +12%, writer +11%). Vector-function call counts and body-size histograms are identical to 13.29, while
the mean VF dispatch latency moved (writer 265 -> 291 cycles, b 180 -> 214). 13.29's own build shows the
same latency swinging between identical instances (c 193 vs c_bar 149). The AIV instruction count rose by
212 in (a) and 1,044 in (b), consistent with the extra per-tile `TBuf::Get` lookups of the split softmax
helpers. A sequential same-session A/B was started and stopped by user direction. Latency is to be judged
on silicon (`msprof`), not on these spans.

### 13.31 Every model on the Cube decode; the scalar overhead of 13.30 removed (2026-09-17)

**Routing.** Silicon measured the Cube decode at 1.98x-2.18x and the AIV-only decode at 0.05x-0.19x (user
report). `SelectPath` (`turboquant_audit_models.hpp`) therefore returns `kCube` for every model: Qwen3.5-9B
(D 128, 4:1) and GLM-5.2 (8:1) move off the AIV path. `ASCEND_BENCH_TQ_AUDIT_PATH=aiv` still forces the
AIV-only decode for an A/B. `PlanFusedDecode` had no D or group gate and is unchanged, so the K = 4
context splits of the 32-block grid still apply. The wheel's adapter is not touched and still dispatches
`turboquant_paged_attention`; the Cube decode stays test-only.

**D = 128 needs no kernel change.** Every size in the kv4fp8 Cube path follows from `headSize` and the
32 B C0: a 64 x 64 B NZ tile burst, two 2,048 B unpack chunks, an 8,192 B L1 tile in 256 C0 blocks,
query rows of 4 C0 groups, a context Fixpipe with n = 128, and 64 B per kv head in the writer. **D = 128
has not executed on the Cube** (fused (a)-(e) are D = 256); its first run is the silicon bench.

**Not implemented: packing kv heads into M = 16.** A score GEMM multiplies the task's query rows by ONE
kv head's key tile (`B1K(slot)` holds a single (tile, kv head) plane), so query rows from different kv
heads cannot share it. Packing them would score them against the wrong keys. Qwen3.5 and GLM-5.2 have
H_KV = 1 anyway, so at batch 1 there is nothing to pack. Tasks with M < 16 already run: the Fixpipe pads
M to the 16-row fractal ((a) runs M = 2 and (d) M = 4).

**Scalar instructions.** 13.30 added 212 AIV instructions to (a) and 1,044 to (b). Two changes, measured
on the camodel's `aiv_instr` (`build/nz_profile.py`):

| build | (a) | (b) | (c) |
| --- | --- | --- | --- |
| 13.29 (3a0b70797) | 35,388 | 104,348 | 147,684 |
| 13.30 (67fa2f2d7) | 35,600 | 105,392 | 148,412 |
| every TBuf view cached once in `Init` | 41,724 | 111,824 | 172,896 |
| softmax helpers folded back into one body | 35,508 | 104,720 | 148,228 |
| + task span and cursor back to scalar arguments | **35,388** | **104,348** | **147,684** |

- **Caching `TBuf::Get` views does not help.** It added ~96 instructions per AIV core of `Init` and saved
  nothing per tile, so `Get` is not what costs. That change was reverted.
- **The cost was structure.** `ComputeSoftmaxAndStageProbs` was split into `ComputeTileScales`,
  `ComputeRowLogits`, `ComputeRowSoftmaxStep`, `ComputeRowProbOperands` and `StageProbRowsToL1`, and each
  re-derived its views and extents. The task extents were packed into `TurboQuantTaskSpan`, whose fields
  `NextTile` reloaded from memory on every tile. Both are back in their 13.29 form, commented as
  deliberate, and the instruction counts equal 13.29's exactly.

**Results.** host / sim / npu clean under `-Werror` with unchanged target sets. Fused (a)-(e) pass: goldens,
(c) vs (b), (e) 0 of 131,072 written bytes differ, 0 B exception dumps.

### 13.32 Fused case (f): D = 128 on the Cube (2026-09-17)

`DecodesHeadSize128OnAQwenGroup` covers Qwen3.5's shape: H_Q 4, H_KV 1, D 128, S 64, block 64, with a
host-built cache. It is the first execution of the D = 128 Cube fractals: 4 C0 groups per query row, an
8,192 B L1 tile, and a context Fixpipe with n = 128. The planner gives 2 tasks and 2 heads per task.

Camodel, one process: golden `0xedc60ea1714295f1`, cos vs exact fp32 **0.999543** (the D = 256 cases sit at
0.99940-0.99958), no sentinel left, 0 B exception dumps, ~70 s. A second build reproduced both pinned values.
The case also holds an absolute floor of cos >= 0.99, so a wrong stride or C0 mapping fails it even
before any golden exists.

**Not covered:** the kv4fp8 kernel writer at D = 128 (64 B per kv head); split reduction, tail masks and
wider groups at D = 128; GLM-5.2's 8:1 group. Run with `build/nz_runs.sh <tag> <snap> f`.

### 13.33 PRE_ROTATED: the raw-query prologue, and full-model KV Saved MB (2026-09-17)

`TurboQuantFusedDecode<MODE, scalar_t, PRE_ROTATED = true>` now says where Pi q happens.

- **`PRE_ROTATED = true`** (every existing entry point). The caller hands in the rotated fp32 query, as before.
  The prologue member is the empty specialisation of `TurboQuantQueryPrologue`, declared last so no other
  member moves. It allocates no UB and issues no instruction. Cases (a)-(f) reproduce every golden unchanged.
- **`PRE_ROTATED = false`** (`turboquant_mm_fused_decode_raw_query_kv4fp8_half`, kv4fp8 only). The launch is
  handed the fp16 query, Pi signs and rotation tables. `RotateQuery()` runs after `Init()` and before
  `Process()`. Each MIX block's two subcores take contiguous halves of `[block * v, (block + 1) * v)` query
  vectors (`FusedDecodeGrid::prologue_vectors_per_block`). Each vector is read from GM, rotated in UB with
  `TurboQuantCodec4::ApplyPi`, the same body as rotate_q's AIV path, and written as fp32 into the buffer the
  tasks read. One `SyncAll` then drains every core, so no split task on another block can read an unwritten
  vector. The rotation therefore runs once per (token, head), not once per split. That matters because
  `kFillBlocks` gives 4 splits at B = 1.
- The prologue adds about 24 KB of UB at D = 256: the codec's tables and work area plus 3.6 KB of its own.
  It is allocated after the decode's buffers, so the decode's UB layout does not move.

**Case (g)** `RotatesARawQueryOnceAheadOfItsSplits` runs (c)'s grid (S 256, 4 splits, 8 blocks, fused limit 0,
in-launch reduction) through the raw-query entry. Camodel, one process each, against a clean `-Werror` sim tier:

| case | golden | cos vs fp32 | notes |
|---|---|---|---|
| (a) | `0x6176461416358ec1` | 0.999407 | 76 s |
| (b) / (c) | `0x470dad36e6708da5` / `0xda6a2c77e20ff4a1` | 0.999400 / 0.999400 | 260 s |
| (d) | `0xcc1fcdfed39b48e5` | 0.999517 | 112 s |
| (e) | `0x9ffe02efde1506cf` | 0.999576 | written cache 0 of 131,072 bytes differ; 505 s |
| (f) | `0xedc60ea1714295f1` | 0.999543 | 66 s |
| (g) | `0xda6a2c77e20ff4a1` (= (c)) | 0.999400 | in-launch rotation 0 of 1,024 fp32 words differ from rotate_q; 148 s |

Every run left 0 B of exception dumps. Run (g) with `build/nz_runs.sh <tag> <build> g`, or all seven with `ag`.

**Not covered:** the raw-query entry at D = 128, at B > 1 and on bf16; no camodel span A/B of the prologue
against a separate rotate_q launch (the launch-dispatch saving it exists for is a silicon number); no Python
or adapter wiring, since the shipping `npu_turboquant_paged_attention` still takes the pre-rotated query.

**Bench (`bench_device_950pr_turboquant`, Table A).** The Context column prints S alone. The chunk C
(`ASCEND_BENCH_TQ_AUDIT_CHUNK`) moved to the legend. KV Saved MB is now whole-model: the benchmarked kv head's
saving times `kv_layers x model_kv_heads / num_kv_heads`, from each checkpoint's config.json. Qwen3.5-9B has
32 layers, but only 8 are full attention, with H_KV 4. DeepSeek-V4-Flash has 43 layers and H_KV 1.
GLM-5.2 has 78 layers of MLA, and its latent counts as one slot. The legend prints the factors. The CSV
`kv_saved_mib` keeps the per-slice value. Only the npu tier compiled this; no silicon run exists.

### 13.34 Adaptive split policy: split by grid saturation, not by context length (2026-09-17)

`PlanFusedDecode` defaults to `FusedSplitPolicy::kAdaptive`. `kFillBlocks` and `kContextOnly` stay selectable.

- **Base tasks.** `base_tasks = tokens x kv_heads x chunks`, where chunks is the count the heads-per-task
  rounding realises. It is not the count `chunks_for()` asks for: 16 heads asked into 5 chunks make 4.
  The first draft used the asked count, and the host contract test caught it calling an under-filled grid
  saturated (aiv 40, kv 4, group 16: 16 tasks on 20 blocks).
- **Saturated grid** (`base_tasks >= mix_blocks`). K = 1 at every context length. `IsFused()` is true for
  every token, so `NeedsReduction()` is false and the launch has no `SyncAll`, workspace or reducer.
  `reduce_tasks_per_block` and `workspace_floats` are 0.
- **Under-filled grid.** K = min(ceil(mix_blocks / base_tasks), kMaxSequenceSplits = 8, blocks), and the fused
  limit is 0 whenever K > 1. This holds at any S, so B = 1 at S <= 4096 keeps its fill splits (the 2026-09-17 permanence
  rule for the context splits of 41c2012e2). The directive's S > 4096 condition and its cap of 4 were dropped by user decision.
- **The 4096 limit was not a numeric bound.** It came from the wording of 13.24 rule 3. The online-softmax
  recurrence is length-agnostic, and an unsplit multi-tile task is what case (b) runs. What has never run,
  on the camodel or on silicon, is an unsplit task over more than 4096 rows.

Bench decode rows, aiv 64, block 128, H_KV 1 (K and tasks on 32 MIX blocks):

| model | S | B | fill K / tasks | adaptive K / tasks |
|---|---|---|---|---|
| Qwen3.5 4:1 D128 | 2048 | 1 / 4 / 8 | 8/16, 4/32, 2/32 | unchanged |
| Qwen3.5 4:1 D128 | 32768 | 1 / 4 / 8 | 8/16, 8/32, 8/64 | 8/16, **4/32, 2/32** |
| GLM-5.2 8:1 D128 | 2048 | 1 / 4 / 8 | 8/32, 2/32, 1/32 | unchanged |
| GLM-5.2 8:1 D128 | 32768 | 1 / 4 / 8 | 8/32, 8/32, 8/64 | 8/32, **2/32, 1/32** |
| DeepSeek-V4-Flash 16:1 D256 | 2048 | 1 / 4 / 8 | 4/32, 1/32, 1/32 | unchanged |
| DeepSeek-V4-Flash 16:1 D256 | 32768 | 1 / 4 / 8 | 8/32, 8/32, 8/64 | **4/32, 1/32, 1/32** |

At S <= 4096 the two policies agree on every row. Qwen 4:1 at H_KV 1 never saturates 32 blocks by B = 8
(16 base tasks), so it keeps 2 splits there. That is the literal saturation test the user chose, over
forcing K = 1 at B >= 4.

**Tests.** `test_host_turboquant_tiling` holds the adaptive contract over the whole sweep. It pins the rows above
(`FusedDecodeAdaptiveSplitsOnlyAnUnderfilledGrid`) and checks that every camodel fused case plans the same grid
under adaptive as under the policy its golden was recorded with (`FusedDecodeAdaptiveKeepsTheCamodelCaseGrids`).
The kernel is unchanged, so no camodel case was re-run. `test_sim_950pr_turboquant_multimode` plans with the
default policy, so its grid may move; per the test-scope rule it was built, not run.

**Bench.** `ASCEND_BENCH_TQ_AUDIT_SPLIT=adaptive|fill|context` (default adaptive) selects the policy for the
decode leg, its traffic model and the msprof trace. Table B gains a K column and prints the policy. The decode
CSV gains a trailing `split_policy` column. Silicon A/B:
`ASCEND_BENCH_TQ_AUDIT_MODELS=qwen35,glm52 ASCEND_BENCH_TQ_AUDIT_S=2048,32768`, once per policy, with
`ASCEND_BENCH_TQ_AUDIT_DECODE_CSV` set per run. No silicon number exists yet.

### 13.35 Two-tier adaptive split policy, dispatch telemetry and the rotation-mode A/B (2026-09-17)

**Why.** The user benchmarked 13.34's first adaptive draft on 950PR silicon at S = 32768, B = 4. Their numbers
(no trace from this project backs them):

| model | K = 8 (fill) | adaptive draft | draft K |
|---|---|---|---|
| DeepSeek-V4-Flash | 306.10 us, 124.4 GB/s | 1241.47 us, 30.4 GB/s | 1 |
| GLM-5.2-744B | 171.27 us, 121.5 GB/s | 491.94 us, 42.5 GB/s | 2 |
| Qwen3.5-9B | 147.17 us, 136.3 GB/s | 253.81 us, 80.6 GB/s | 4 |

Their reading: one MTE2 stream per sequence cannot keep enough DMA transactions in flight to load the
multi-channel HBM controller. That costs hundreds of microseconds, against 2-4 us for the partial reduction.

**Policy.** `kAdaptive` now has two tiers, keyed on the longest context the grid is planned for,
`blocks x block_size`.
- **Latency tier (every context).** Unchanged from 13.34: K = ceil(mix_blocks / base_tasks) on an
  under-filled grid, K = 1 on a saturated one.
- **Bandwidth tier (context >= `kBandwidthSplitContext` = 8192).** K = CeilPow2(max(latency K,
  ceil(S / `kBandwidthSplitRows` = 2048))). This splits a saturated grid too.
- **Caps.** Both tiers are capped at `min(kMaxSequenceSplits, blocks)`. The fused limit is 0 whenever K > 1.

Two changes to the directive's literal formula, both from the planner's own arithmetic:
- **The max with the latency tier.** Without it, K is not monotonic in S. Qwen B = 1 plans 8 splits at 8064,
  would get ceil(8192 / 2048) = 4 at 8192, and 8 again at 16384.
- **The power-of-two rounding.** Blocks take tasks in equal runs, so 5 to 7 splits of power-of-two tasks
  leave blocks idle. A literal ceil(12288 / 2048) = 6 puts 48 tasks on 24 of 32 blocks for Qwen B = 4, and
  36 tasks on 18 blocks for DeepSeek B = 1. With the rounding, the tier is 4 splits at exactly 8192 tokens and 8 beyond.

`DecodeNeedsReduction(K, max context, fused limit)` is the host side of both kernels' `NeedsReduction()`.

**Schedule.** aiv 64 (32 MIX blocks), block 128, H_KV 1; `K/Tsk/Blk/Red`:

| model | S | B 1 | B 2 | B 4 | B 8 |
|---|---|---|---|---|---|
| Qwen3.5 4:1 D128 | 2048 | 8/16/16/ON | 8/32/32/ON | 4/32/32/ON | 2/32/32/ON |
| Qwen3.5 4:1 D128 | 32768, 262144, 1048576 | 8/16/16/ON | 8/32/32/ON | 8/32/32/ON (draft 4) | 8/64/32/ON (draft 2) |
| DeepSeek-V4-Flash 16:1 D256 | 2048 | 4/32/32/ON | 2/32/32/ON | 1/32/32/OFF | 1/32/32/OFF |
| DeepSeek-V4-Flash 16:1 D256 | 32768, 262144, 1048576 | 8/32/32/ON (draft 4) | 8/32/32/ON (draft 2) | 8/32/32/ON (draft 1) | 8/64/32/ON (draft 1) |
| GLM-5.2 8:1 D128 | 2048 | 8/32/32/ON | 4/32/32/ON | 2/32/32/ON | 1/32/32/OFF |
| GLM-5.2 8:1 D128 | 32768, 262144, 1048576 | 8/32/32/ON | 8/32/32/ON (draft 4) | 8/32/32/ON (draft 2) | 8/64/32/ON (draft 1) |

- **Bench rows.** On every row the adaptive grid is now the fill policy's: same K, tasks, heads per task
  and blocks. Only the launch limit differs, 0 against 4096, and a uniform batch beyond 4096 reduces either way.
- **The 2048 rows** are 13.34's, pruning included.
- **Where adaptive and fill still differ, off the bench.** At S in (4096, 8192) a saturated grid gets K = 1
  against fill's >= 2. At S in [8192, 16384], adaptive gives 8 against fill's 4 at B = 8, and 4 against 8 for
  DeepSeek at B = 1, 8192.

**Telemetry (bench).**
- **Table B columns:** `Model | Context | B | Path | Cfg [K/Tsk/Blk/Red] | T_rot_q | T_FusedDecode | Launches |
  T_rot_o | TQ_E2E | V5_Decode | Speedup | Eff GB/s`.
- **Cfg:** Tsk is the launch's task count (Cube B x H_KV x K x head chunks; AIV B x H_Q x K). Blk counts
  MIX blocks (Cube) or vector cores (AIV); the legend prints the device totals.
- **Compression** left the table; it stays in the CSV as `compression_ratio`.
- **Alignment fix.** The old rows printed T_FusedDecode and Launches one column narrower than their headers.
- **Decode CSV.** One row per (configuration, rotation mode). New columns: `rotation_mode, split_k, total_tasks,
  active_blocks, device_blocks, needs_reduction`. `num_splits` became `split_k`.
- **Consistency check.** Each configuration fails a `dispatch_*` case if the scenario's launched grid differs
  from what the table reports.
- **Traffic model.** The partials term follows `needs_reduction`, not K > 1.

**Rotation mode (bench, user directive).** `ASCEND_BENCH_TQ_ROTATION_MODE=separate|fused_prologue|both`
(default `both`).
- **`fused_prologue`.** Path `Cube-FusedQ` feeds the raw fp16 query, the Pi signs and `CodecTables(D, 1)` to
  `turboquant_mm_fused_decode_raw_query_impl` over the same grid, and passes `prologue_vectors_per_block`.
- **New legs.** `dec_fq_attn_core` and `dec_fq_e2e`. T_rot_q is a structural 0.00 ("in-launch").
  T_FusedDecode includes the rotation. Launches is one fewer than `Cube-Sep`.
- **Shared with `-Sep`.** T_rot_o, V5 and Cfg.
- **Delta block.** After the legend, one line per configuration gives TQ_E2E(-FusedQ) - TQ_E2E(-Sep), and the
  T_FusedDecode difference against the rotate_q launch it replaces.
- **AIV path.** It has no raw entry, so its `-FusedQ` legs are skipped.
- **Agreement gate.** Before timing, every configuration that times `-FusedQ` runs both modes once.
  - It prints how many fp32 words of the in-launch rotation differ from rotate_q's. This count is not gated:
    rotate_q stages the Cube from two tokens on, and its rounding differs.
  - It fails `rotation_mode_agreement_*` if the two decode outputs agree below cos 0.9999.
  - Silicon is the first place the raw entry runs at B > 1 or D = 128.

**Tests.**
- **Host tiling test.**
  - The sweep contract covers both tiers, with blocks 130 and 512 added.
  - `FusedDecodeAdaptivePinsTheBenchSchedule` pins all 48 rows above and their equality with fill.
  - `FusedDecodeAdaptiveCrossesIntoTheBandwidthTier` pins the boundary rows.
  - `FusedDecodeAdaptiveNeverDipsOrStrandsBlocks` walks 1..512 blocks for B in {1..8, 16}, kv heads
    {1, 2, 4, 8} and groups {1, 2, 4, 8, 16}: K never falls, and a power-of-two batch never leaves a block
    idle.
  - `FusedDecodeKeepsTheCamodelCaseGrids` pins (a)-(g)'s grids under their recorded policies. Those policies
    are explicit in the fused test, so the adaptive policy was never on their path.
  - `DecodeNeedsReductionMirrorsTheKernels` checks the host mirror.
- **Kernel.** No kernel source changed.

**Verification.** Host, sim and npu tiers built clean under `-Werror`: 0 diagnostic lines, target sets
unchanged at 6 / 37 / 42. `test_host_turboquant_tiling` passed 11 of 11. Fused (a)-(g) ran on the camodel
against the new sim tier, one process each (`build/nz_runs.sh bw /workspace/build_sim ag`), with the grids
the host test pins:

| case | K / blocks / heads per task / limit | golden | cos vs fp32 | notes |
|---|---|---|---|---|
| (a) | 1 / 2 / 2 / 4096 | `0x6176461416358ec1` | 0.999407 | 78 s |
| (b) / (c) | 1 / 2 / 2 / 4096, 4 / 8 / 2 / 0 | `0x470dad36e6708da5` / `0xda6a2c77e20ff4a1` | 0.999400 / 0.999400 | (c) vs (b) cos 1.000000000; 250 s |
| (d) | 1 / 2 / 4 / 4096 | `0xcc1fcdfed39b48e5` | 0.999517 | 102 s |
| (e) | 1 / 2 / 2 / 4096 | `0x9ffe02efde1506cf` | 0.999576 | written cache 0 of 131,072 bytes differ; 446 s |
| (f) | 1 / 2 / 2 / 4096 | `0xedc60ea1714295f1` | 0.999543 | 71 s |
| (g) | 4 / 8 / 2 / 0 | `0xda6a2c77e20ff4a1` | 0.999400 | in-launch rotation 0 of 1,024 words differ; 164 s |

Every run left 0 B of exception dumps. `build/bandwidth_tiers.sh` and `build/bandwidth_sim_ag.sh` repeat the
two steps. `build/split_ab_compare.py` now keys on the rotation mode as well and reads `split_k` (or an older
CSV's `num_splits`).

**Not covered.**
- No silicon number exists for either tier or either rotation mode.
- The msprof trace harness plans with the new policy but has no rotation mode.
- `test_sim_950pr_turboquant_multimode` (contexts <= 2048, so latency tier only) was built, not run.

### 13.36 Downstream fusion: un-rotation and output gate in the raw-query launch, and the Python wiring (2026-09-17)

**What moved into the launch.** The raw-query Cube decode (`PRE_ROTATED = false`, 13.33) now changes basis at
both ends. `cube/turboquant_query_basis.h` (`TurboQuantQueryBasis`, which replaces `TurboQuantQueryPrologue`)
holds both stages:
- **`Rotate()`, the prologue.** It rotates Pi q = D H D q once per (token, head) before the task loop. A block
  whose share holds at least `kQueryBasisMinCubeVectors` = 16 vectors rotates whole 16-vector chunks on the
  Cube. The AIV subcores sign-flip and stage their halves into one L1 tile, the AIC multiplies it by H16
  (butterfly strides 1-8) and dual-destination Fixpipes the halves back, and each subcore runs the strides from
  16 up. The remainder, and any share below 16, rotates on the vector cores with `ApplyPi`, as before. The host
  plans the chunk with `QueryBasisCubeChunk` into `FusedDecodeGrid::prologue_cube_chunk_vectors`.
- **`FinishHeads()`, the output stage.** It runs on every normalised fp32 head row before the output cast, chosen
  per launch by `TurboQuantOutputStage`: `kRotatedBasis` (0, folded W_o, unchanged), `kUnrotated` (1, Pi once
  more) or `kGated` (2, un-rotated, then divided by 1 + exp(-gate), with the gate logit floored at
  `kGateLogitFloor` = -80). The vector service's fused path and the Split-K reducer both call the finisher, so a
  split decode and an unsplit one share the stage.

The raw-query entry takes two new pointers, `h16` and `gate`, and two new sizes, `prologueCubeChunkVectors` and
`outputStage`. For an unfolded layer, a decode step is now one launch: no rotate_q before it, no rotate_q(o),
copy_, sigmoid or mul after it.

**Wheel and Python.**
- **Root `CMakeLists.txt`.** For `ascend950` SOCs it adds `turboquant_fused_decode.cpp` to the shipped
  `vllm_ascend_turboquant` library and defines `VLLM_ENABLE_TURBOQUANT_CUBE`. The kv3fp4 / kv5fp8 modes and the
  fp8 GEMM probe compile only under `VLLM_ASCEND_TQ_TEST_KERNELS`, which only `csrc/tests` sets.
- **Operators** (`torch_binding.cpp`, adapter, Meta no-ops): `npu_turboquant_cube_reshape_and_cache`,
  `npu_turboquant_cube_decode` and `npu_turboquant_cube_workspace_size`.
- **Backend.** `turboquant_v1.py` selects the Cube path when `VLLM_ASCEND_TURBOQUANT_CUBE_DECODE` (default 1), the
  build and float16 all allow it. The writer follows the decode, because the two layouts differ.
- **Gated layers.** `vllm::turboquant_gated_attention` (`vllm_ascend/ops/turboquant_attention.py`, a graph
  splitting op with a fake impl) hands a Qwen3.5 gated layer's raw gate to the impl. `patch_qwen3_5.py` then
  skips the model's own sigmoid and multiply.

**Tests.**
- **Host tiling.** `FusedDecodePrologueRoutesWholeChunksToTheCube` pins (h)'s grids. 12 of 12 pass.
- **Fused case (h).** `UnrotatesAndGatesARawQueryBatch`: H_Q 8, H_KV 2, D 256, S 128, block 64, B = 3, raw-query
  entry. The 24 query vectors put one 16-vector chunk on the Cube and 8 on the vector cores. The case runs three
  launches over one cache:
  1. rotated basis, unsplit;
  2. gated, unsplit;
  3. gated, filled to 2 splits.

  It gates on four checks. The prologue must match rotate_q within 1e-5. The kernel's output stage must agree
  with the host applying Pi and the gate to launch 1 at cos >= 0.99999. The split and unsplit gated outputs must
  agree at cos >= 0.999999. Goldens and cosines are pinned.
- **Python.**
  - `TURBOQUANT_CUBE_OP_SCHEMAS` pins the three schemas against `torch_binding.cpp`.
  - Deliberately, there is no CPU kernel for them. `turboquant_cube_meta_ops()` defines the schemas with shape-only
    `Meta` kernels, and `TestCubeOpsOnMeta` sends the backend's decode (all three stages) and cache write through the
    real dispatcher on meta tensors.
  - `TestCubeDecode` (mocked ops) and `tests/ut/ops/test_turboquant_attention.py` cover the call contract and the
    gated op.
- **Bench.**
  - The `-FusedQ` path passes `h16` and the stage (`kUnrotated` unless W_o is folded). Its T_rot_o is an in-launch
    0.00, and its decode is one launch.
  - The rotation agreement gate now compares the model-basis outputs.
  - Eff GB/s and the decode CSV's `e2e_bytes` use `dec_fused_q_e2e` on `-FusedQ` rows. Before this, those rows
    divided by the separate path's bytes.

**Verification.**
- **Tiers.** Host, sim and npu built clean under `-Werror` (0 diagnostic lines). The npu tier was rebuilt after the
  bench fix with its 42 targets unchanged.
- **Adapter.** The adapter, `torch_binding.cpp` and `torch_binding_meta.cpp` pass a syntax check against the vendor
  image's torch / torch_npu headers, with and without `VLLM_ENABLE_TURBOQUANT_CUBE`.
- **Python** (`quay.io/ascend/vllm-ascend:v0.26.0rc1-a5`). The five TurboQuant files plus
  `tests/ut/ops/test_turboquant_attention.py` give 128 passed. `tests/ut/{attention,ops,patch}`, `test_platform`,
  `test_utils` and `test_envs` give 201 failed / 841 passed, against 201 / 822 at HEAD, with an identical set of
  failing test ids. Both sides share two collection errors (`test_select_experts.py`, `test_comm_utils.py`).

Camodel, one process per case (`build/nz_runs.sh dualh /workspace/build_sim ah`):

| case | K / blocks / heads per task / limit | golden | cos vs fp32 | notes |
|---|---|---|---|---|
| (a) | 1 / 2 / 2 / 4096 | `0x6176461416358ec1` | 0.999407 | 85 s, 22,697 ticks |
| (b) / (c) | 1 / 2 / 2 / 4096, 4 / 8 / 2 / 0 | `0x470dad36e6708da5` / `0xda6a2c77e20ff4a1` | 0.999400 / 0.999400 | 270 s, 68,369 ticks |
| (d) | 1 / 2 / 4 / 4096 | `0xcc1fcdfed39b48e5` | 0.999517 | 110 s, 29,310 ticks |
| (e) | 1 / 2 / 2 / 4096 | `0x9ffe02efde1506cf` | 0.999576 | written cache 0 of 131,072 bytes differ; 491 s, 52,935 ticks |
| (f) | 1 / 2 / 2 / 4096 | `0xedc60ea1714295f1` | 0.999543 | 75 s, 20,587 ticks |
| (g) | 4 / 8 / 2 / 0 | `0xda6a2c77e20ff4a1` | 0.999400 | in-launch rotation 0 of 1,024 words differ; 185 s, 43,604 ticks |
| (h) rotated | 1 / 12 / 2 / 4096 | `0x964ba5db49bdfb2d` | 0.999510 | in-launch rotation 0 of 6,144 words differ (max err 0) |
| (h) gated | 1 / 12 / 2 / 4096 | `0x80ddb42451590ba1` | 0.999507 | output stage vs host stage cos 0.999999957 |
| (h) gated split | 2 / 24 / 2 / 0 | `0x80ddb42451590ba1` | 0.999507 | vs unsplit cos 1.000000000; (h) total 836 s, 151,183 ticks |

(a)-(g) are unchanged from 13.35. Every run left 0 B of exception dumps. Tick counts are per process and include
host time between launches (see the camodel run policy), so they are not per-launch spans. A clean sim
rebuild with (h)'s goldens pinned (0 diagnostics, 37 targets unchanged) reproduced all three hashes and cosines
(`build/final_sim_h.sh`, 770 s, 151,176 ticks, 0 B dumps).

**Not covered.**
- No silicon number exists for the one-launch step, or for the gate.
- The wheel's own CMake path (`ascend950` SOC, `VLLM_ENABLE_TURBOQUANT_CUBE`) has not been built. Only the adapter
  and binding sources were syntax-checked.
- Raw-query output stages at D = 128, in bf16, and with a Cube chunk on more than one block.
- The Python Cube path and `patch_qwen3_5.py`'s gated branch have not run on a device.
- No camodel launch spans were taken with `build/nz_profile.py`.

### 13.37 The `lse` out-tensor, and uncompressed sinks on both decodes (2026-09-24)

**What the decodes now return.** Both take an optional `lse` out-tensor, last and defaulted, so every existing
call site was left alone: `[numTokens, numHeads, kLseStride]` fp32, carrying each (token, head)'s running
softmax maximum in lane `kPartialMaxLane` and its mass `sum_i exp(s_i - max)` in lane `kPartialSumLane`.
`turboquant_layout.h` owns the stride and the Python side mirrors it (`TURBOQUANT_LSE_STRIDE`).

- **Why it cannot be recovered on the host.** An attention output is a convex combination of the values, so it
  is invariant to a rescaling of the weights, and the mass is exactly what that rescaling destroys. No number of
  extra launches over the operator's output gets it back. The kernels already had it.
- **`TurboQuantLseWriter`** (`vector/turboquant_vector_service.h`) holds one `GlobalTensor` and an `enabled_`
  launch constant. `enabled_` is identical on every core, so branching on it cannot desync the two halves of a
  mixed Cube launch the way a per-task predicate would, and a null pointer costs a launch a branch.
- **Three writers, one per path, and every (token, head) written exactly once.** The AIV split writes its state
  lanes in the fused branch (`turboquant_paged_attention.cpp`); the Cube fused branch stages its pairs through
  `StageLse` into the dead half of `scoreBuf_` (a `static_assert` pins that they fit); a split token's pair comes
  out of `TurboQuantPartialReducer` instead. All three write *before* `NormalizeHeads`, which adds its epsilon to
  the running sum in place.
- **A whole `kPartialTail` per (token, head), not one float.** Every `DataCopy` these kernels make to global
  memory moves whole 32-byte bursts. At this stride the offset of any (token, head) is a multiple of 64 bytes
  whatever `numHeads` is, and the pair is already laid out exactly as a split's partial tail, so all three
  writers hand the same sixteen words to the same copy. It costs 64 B per head per decode step.
- **The pair, not `max + log(mass)`.** An empty context leaves the mass at zero and a log of it would put a NaN
  in an out-tensor the kernel has no way to flag. An empty context writes `(kVectorNegInf, 0)` and every reader
  branches on the mass, never on the max.

**What it is for: uncompressed attention sinks, phase two.** Phase one kept the first N tokens of a sequence at
activation precision in a side-car plane and merged them into the decode by recomputing the quantised context's
softmax denominator on the host -- `O(context)` per decode step -- and by dequantising the sink rows to subtract
what they had contributed. Both halves are gone:

- **The weight comes out of the kernel.** `quantized_context_softmax_stats` is now a test oracle and nothing on a
  serving path calls it.
- **The quantised copies leave through the scale plane.** Every mode reconstructs a cached vector as code times
  the token's scale, and both decodes apply that scale on the vector unit after the product. Zeroing a sink
  token's K and V scale lanes (`TurboQuantSinkCache.neutralize_scale_rows`, run after the packed write) makes its
  key exactly the zero vector and its value contribution exactly zero, on any codebook and in any byte order,
  without a nibble being unpacked. The row then contributes exactly one unit of absolute mass -- a known integer
  -- and nothing to the numerator, so the merge is `w_q O_q + sum_j w_j v_j` over
  `(w_q - n exp(-m')) + sum_j w_j`: an append of two streams, not a subtraction of a measured one, and better
  conditioned for it.
- **So the kv4fp8 Cube decode is no longer refused.** 13.36's NZ-tiled affine planes were unreadable to a host
  reader of the packed cache (LongBench 57.6% at 0 sinks against 14.1% at 4, which is what the refusal in
  `0c4d9b3b2` was for). The merge reads no packed byte now, so both TurboQuant layouts serve it, and neither
  `tools/tq_longbench` nor the plugin gives up static decode or graph capture for it. What the plugin still gives
  up is the single-launch fusion: the merge has to see the raw rotated accumulator, so a sink layer decodes at
  `kRotatedBasis` and re-applies the un-rotation and the gate afterwards.
- **The guard moved to the build.** `_refuse_sinks_without_softmax_stats` refuses `VLLM_ASCEND_TQ_SINK_TOKENS`
  against an extension whose decode schema has no `lse` argument, before a plane is allocated, rather than
  leaving it a `TypeError` about an argument count inside a decode step.

**Verification.**
- **Tiers.** Host, sim and npu built clean under `-Werror`, 0 diagnostic lines, target sets unchanged (6 / 37 / 42
  unique `Built target` lines, identical to the pre-port builds).
- **Camodel** (`build/lse_sim_smoke.sh`, one process, one case). Fused case (a), kv4fp8, S = 64: its FNV-1a golden
  and `cos vs fp32 0.999407` both match 13.36's recorded values, 0 B of exception dumps, 68.8 s host wall. Every
  launch in `csrc/tests` passes `lse = nullptr`, so this is what says the added writer is inert when nobody asks
  for the statistics. The S = 256 cases were not re-run: see the camodel run policy.
- **Python** (`quay.io/ascend/vllm-ascend:v0.26.0rc1-a5`). `tests/ut/attention` plus `test_platform_turboquant.py`:
  418 passed against 408 at HEAD, with an identical set of 152 failing ids (all pre-existing, sfa/mla/deepseek).
  The five TurboQuant files give 241 passed. `tests/ut/_tools/test_tq_longbench.py`: 59 failed / 278 passed
  against 59 / 279, the failing set identical (the pre-existing "vLLM leaked into the process" family).
  `test_turboquant_sink_cache.py` drives the whole feature through the CPU operators, which now write the
  out-tensor too, so the pair the merge consumes there is the pair the kernels write.
- **The guard was a no-op when it was written.** `torch.ops` hands out an `OpOverloadPacket`, which carries no
  `_schema` at all -- the schema is on the overload -- so reading one off the packet passed on every build there
  has ever been. `_registered_schema` reads `default._schema` and insists on a real `FunctionSchema`;
  `test_the_build_check_reads_the_schema_that_is_actually_registered` pins it.

**Not covered.**
- No silicon or camodel run has passed a non-null `lse`. The three in-kernel writers are exercised only by the
  CPU stand-in, whose `npu_turboquant_paged_attention` computes the same pair in one pass rather than tile by
  tile; the Cube `StageLse` and the reducer's write have never executed.
- No camodel case covers the sink merge end to end, and nothing measures what the merge's extra launches cost.
- The LongBench retrieval number that motivated the feature has not been re-taken with the phase-two merge, on
  either backend.


### 13.38 The `lse` out-tensor on silicon: two device cases and the turnkey run-scripts (2026-09-25)

13.37 shipped the out-tensor with one hole named in its own "Not covered": **no launch anywhere had
ever passed a non-null `lse`.** Every `csrc/tests` call site passed `nullptr`, so what the camodel
gate proved was only that the added writer is inert when disabled. This closes the hole on the
device tier and packages the hardware run so it needs no judgement at the console.

**Two new bare-metal cases** (`device/test_device_950pr_turboquant.cpp`, the AIV decode):

- **`DecodeReportsItsSoftmaxStatisticsToTheLseOutTensor`.** Allocates the
  `[numTokens, numHeads, kLseStride]` fp32 buffer, **poisons it with -1e30** so a `(token, head)`
  no writer touched is distinguishable from one written badly -- which an all-zero buffer cannot be,
  because zero is exactly an empty context's mass -- runs the decode with it, and compares against
  an independent host derivation. `cpu_paged_attention_turboquant` gained two trailing defaulted
  out-parameters, `softmax_max` and `softmax_mass`, reported **before** its `kEps` floor, because
  the kernels write the pair before `NormalizeHeads` adds theirs.

  What it gates, and why in this shape:
  - the whole log-mass `m + log(L)` to 0.02, which is 2% of the total softmax mass. `m` alone is
    *not* the meaningful quantity: two scores within fp rounding can swap the argmax between the
    kernel's tile-wise online reduction and the reference's one-pass loop, and `L` then compensates
    exactly. `m` is gated loosely (0.05) as a check that it is a score of this context at all;
  - `L >= 1`, exactly and structurally: the maximising element contributes `exp(0)`;
  - every padding lane of the pair's two blocks is exactly `0`. A poison there means the copy was
    short; anything else means the write ran off its own group, which is the failure the
    `kLseStride` padding exists to make impossible;
  - which writer the context reached is *reported* (`writer=fused` or `writer=reducer`) rather than
    inferred. This matters more than it looks: the kernel's `IsFused` is
    `numSplits <= 1 || contextLen <= fusedContextLimit`, and the default ladder tops out at 2048
    against a limit of 4096, so **the ladder alone reaches the fused writer only** and would have
    said nothing about `TurboQuantPartialReducer`'s write. The case therefore sweeps one context
    above the limit as well (`ASCEND_TQ_LSE_SPLIT_CONTEXT`, default 8192) and carries a non-fatal
    expectation that at least one context actually split, so a run that covered one writer cannot
    be read as covering both.
- **`DecodeWithNoLseTensorLeavesTheOutputUnchanged`.** The same scenario decoded with and without
  the out-tensor, compared **bit for bit** on the fp16 output. The writer is gated on a launch
  constant every core agrees on, so attaching it cannot change what the rest of the kernel computes;
  "optional costs nothing" has to include the answer, not just the time.

**Two turnkey scripts**, both `set -euo pipefail` with `--help`:

- **`scripts/run_950pr_baremetal_validation.sh`** -- stage 1, no Python and no checkpoint.
  `npu-smi` preflight; binaries present (with the cmake line to fix it if not); the AIV decode and
  both `lse` cases at `ASCEND_TQ_BARE_METAL_CONTEXTS`; the kv4fp8 Cube decode; the unpack ablation
  under `ASCEND_BENCH_TQ_UNPACK=both`.

  Three things it gets right that are easy to get wrong:
  - **`REQUIRE_PHYSICAL_ASCEND_950PR` is a `GTEST_SKIP`, so a run with no part exits 0.** The
    script treats any `SKIPPED` case as a failure; otherwise the gate is a false green.
  - **two cosines, two different gates.** `test_device_950pr_turboquant`'s `cos` is the kernel
    against a CPU reference running the same arithmetic over the same quantised cache -- a
    correctness gate, and 0.999 is right. The multimode binary's "cos vs fp32 host reference" is the
    4-bit cache's *own* quantisation error against exact fp32 over unquantised K/V, measured at
    **0.98629** worst by `scripts/tq_multimode_calibration.py`. Gating that at 0.999 would fail
    every correct build, so it is gated at 0.98 and the distinction is stated in-file.
  - the decode-cosine and log-mass bounds are re-asserted in the script, not merely trusted to the
    binary's own `EXPECT`s, so the run halts on them even if a later build loosens a constant.
- **`scripts/run_950pr_sink_lse_benchmark.sh`** -- stage 2, the ablation. Sweeps
  `--sink-tokens 0,4` over `--contexts 2048,8192,32768` through `run_benchmark.py`, optionally
  `run_longbench.py` for task scores, into `reports/npu_950pr_sink_lse_<stamp>/` (JSONL, summary,
  progress log, `run_manifest.txt` with the git head and the environment, `npu-smi.log`).
  It pins `--prefill-mode dense_staging`, because the per-rung policy can choose `batched_decode` at
  a long rung and `RunnerConfig` then refuses the run minutes in; it refuses that combination itself,
  up front. It does *not* pass `--skip-preflight` by default: the probe is what negotiates the
  staging pool's backend, and this part refuses op-plugin's FIA (EZ9903) and has to stage through
  `cann_dense`. `VLLM_ASCEND_TQ_SINK_TOKENS` is left alone -- the harness publishes it per rung.

  The banner says which numbers to read: the sinks>0 rows are for their **scores**. Their decode p50
  is not comparable with the sinks=0 rows, because the merge adds launches to every step and a sink
  layer gives up the single-launch Cube fusion.

**Verification.**
- **Tiers, from the `feat/tq-sink-lse-fusion` worktree.** host and npu built clean under `-Werror`,
  0 diagnostic lines; 6 and 42 unique targets. `nm` confirms every TurboQuant symbol the three
  device binaries import is exported by `libvllm_ascend_turboquant.so` -- 7, 3 and 3 imports, 0
  unresolved -- and both new case names are in the correctness binary.
- **The adapter and the schema.** `tools/tq_longbench/build_turboquant_ops.py --soc-version
  Ascend950PR_9599` compiled and linked `turboquant_standalone_binding.cpp`, which is one of only
  two files that include `turboquant_torch_adpt.h`. This is the **first time** `CheckLse` and the two
  `lse` schema lines have been compiled at all: the three `-Werror` tiers never include that header.
  The vendor image cannot *load* the result (no driver), so the registered schema text stays covered
  by `test_schemas_match_the_compiled_extension`, which pins it to `turboquant_torch_ops.h`.
- **The scripts.** `bash -n` clean. Exercised for: `--help`; an unknown flag; no `npu-smi` (exit 2);
  a missing build tree (stage 1 names it and exits 1); a missing `--model-path`; a malformed sink
  list; and `--sink-tokens 4` with `--prefill-mode batched_decode`, which is refused before anything
  loads. A `--dry-run` of the full matrix emits both argv lists, and `build/lse_argv_check.py` feeds
  them to `run_benchmark.build_parser()` and `run_longbench.build_parser()` -- 23 and 24 words,
  both parsed -- then constructs `RunnerConfig` for all four (backend, sink count) pairs.
  `turboquant_cube` with sinks is accepted, which is what 13.37 unlocked.

**Not covered.**
- **Nothing here has run on the part.** The two device cases have been compiled and linked, not
  executed; the scripts have been argument-checked, not run against silicon. Their first execution
  is the 950PR session they exist for, and the tolerances in the new cases are therefore predicted
  rather than measured.
- The camodel was not asked to run a non-null `lse` either: the sim tier's gate is the fused case
  set, and widening it is a camodel-cost decision (see the camodel run policy).
- The wheel's own CMake path still has not been built; the standalone binding is the adapter's only
  compile gate.
- Nothing prices the sink merge's extra launches, on the camodel or on silicon.


**Addendum 2026-09-25: which library the harness launches, and refusing a stale one.**
On a shared container with an older extension reachable on the Python path, stage 2 died
with `_C_ascend::npu_turboquant_cube_decode() expected at most 17 argument(s) but received
18` -- the pre-13.37 schema, registered by something else before the harness looked.

**Two facts shape the fix, and the first one is not obvious.**

1. **Registration is global and permanent.** Once any library has `def`-ed
   `_C_ascend::npu_turboquant_cube_decode`, a second library defining the same name
   *raises* rather than replacing it. Loading a fresh `.so` afterwards cannot correct a
   stale schema. The only fix is to stop the stale library loading; the only useful thing
   the harness can do is say which file it was, before a launch rather than during one.
2. **A `csrc/tests` build tree contains no torch binding.** Its
   `lib/libvllm_ascend_turboquant.so` is the Ascend C kernel library and nothing else --
   `readelf -d` shows no `libc10` and no `libtorch_npu` among its `NEEDED` entries, because
   that tier is configured without Python, PyTorch or torch_npu on purpose. Handing it to
   `torch.ops.load_library` succeeds and registers nothing. The operators come only from
   `libvllm_turboquant_cube.so`, which `tools/tq_longbench/build_turboquant_ops.py`
   produces (and which does carry `libc10`, `libtorch_npu` and
   `libvllm_turboquant_cube_kernels.so`). Neither path needs `setup.py` or `pip install`.

**What changed.**
- `turboquant_library_candidates()` searches `$ASCEND_TQ_BUILD_DIR` first, then the build
  script's install directory, then `build/csrc-tests-npu/` and `build/`. A directory
  candidate is tried both flat and under `lib/`, because a CMake tree puts its libraries in
  the latter and a staged directory in the former. `$TURBOQUANT_LIB_PATH` still replaces
  the search outright; `$ASCEND_TQ_BUILD_DIR` only adds to it.
- `assert_turboquant_schema_is_fresh()` reads each decode's registered schema off its
  *overload* -- an `OpOverloadPacket` carries no `_schema`, so reading one off the packet is
  a check that can only ever pass -- and refuses a registration missing the `lse` argument.
  The message carries the schema it found, every mapped file named `_C_ascend` or
  `turboquant` from `/proc/self/maps` (the dispatcher records no provenance for a schema),
  the note that loading another library will not help, and every path a current one would
  be looked for at. `ascend_ops()` runs it once per registration, so it is on the path every
  backend takes rather than beside it.
- `tools/tq_longbench/check_turboquant_ops.py` is the same check as a CLI: seconds, no
  checkpoint, no dataset, no device. Exit 0 fresh, 1 stale, 2 nothing registered.
  `run_950pr_sink_lse_benchmark.sh` runs it before the sweep and stops on it, exports
  `ASCEND_TQ_BUILD_DIR` and prepends the build tree to `LD_LIBRARY_PATH`, and keeps the
  whole preflight output in the report as `turboquant_ops_check.log`.
- **`npu-smi` is no longer fatal in either script.** It goes through DCMI, which returns
  -8005 inside a shared container on hosts whose devices a run can still open; refusing
  there blocks a working setup on the strength of a management interface. What decides
  whether the operators can run is the runtime's own device open -- the test binaries refuse
  a CAModel themselves, the benchmark exits 77, and a suite that quietly skipped is caught
  by stage 2's SKIPPED count, which is a failure there.

**Verification.** `tests/ut/_tools/test_tq_longbench.py` gained
`TestTurboQuantSchemaFreshness` and three search-order cases: 59 failed / 288 passed
against 59 / 278, the failing set unchanged (the pre-existing "vLLM leaked into the
process" family) and the ten new cases green. The stale path is exercised by registering
the real pre-13.37 schema under its own name and asserting on the refusal. On Linux,
`loaded_library_paths` was confirmed against procfs to return real `libtorch` paths, so a
stale file will be named. The CLI was run in the vendor image for the
nothing-registered path (exit 2) and for a synthetic fresh registration (exit 0); the
stale path has not been reproduced against a real older `_C_ascend`, because no host here
has one installed.
