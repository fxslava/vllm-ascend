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
│   └── bench_device_950pr_turboquant_ablation.cpp
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
| **Binaries** | `test_device_950pr_turboquant`, `test_device_950pr_{matmul,rmsnorm,rotary_embedding,activation_swiglu,qwen_layer_golden}`, `test_device_950pr_benchmark_harness`, `bench_device_950pr_turboquant`, `bench_device_950pr_turboquant_ablation` |
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

**Why "AIV-only" is the headline.** Nothing in `turboquant_kernels.cpp` touches
the Cube: the score row is a vector `Mul` plus a `ReduceSum`, the value
accumulation is a vector FMA, the Walsh-Hadamard is `Add`/`Sub` and `Gather`, the
codec is compares and a `Gather` over a 16-entry table. Grep the source for
`Matmul`, `Mmad` or an `[aic]` block and it comes back empty. This is the
baseline any future Cube-assisted INT4 path has to beat, and measuring it before
that path exists is the only way the later comparison means anything.

**Cases**, per context length `S in {512, 1024, 2048}`:

| Case | What is timed | Notes |
| --- | --- | --- |
| `tq4_write_s<S>` | The cache write path: `S` tokens rotated, quantised and scattered. One launch. | Prefill-shaped — a decode step writes one token — and here because the cache has to be filled before it can be read and the fill is not free. |
| `tq4_decode_s<S>` | The decode: split then combine, **two** launches on one stream. | The cache is written once during setup, so the timed region is the read path only. `tasks_per_launch = 2`, so the harness's queue-depth accounting knows this case submits two tasks per iteration. |
| `fp16_decode_s<S>` | The same decode with an unquantised fp16 paged KV cache, through `aclnnFusedInferAttentionScoreV5` (or V2 where that is what resolves). | See §8. |

Each case carries a checksum the harness verifies after warmup and again after
the last timed iteration — the standard defence against timing an operator that
has stopped writing its output. The write path's checksum is the whole scale
plane, so a scale written outside the slot mapping is caught too.

**Reported metrics.** The shared harness reports min / median / mean / P95 / P99
/ stddev per case across three timing modes (`pipelined`, `device`, `host`), plus
TFLOP/s and GB/s. On top of that this suite prints its own summary: median
pipelined latency, **decode steps per second**, KV GB/s from the traffic model's
decode-read bytes over the measured time, TFLOP/s, and the ratio to the fp16 leg.

`flops_per_iteration` counts the attention math only — `QK^T` and the value
accumulation, two operations per element of each. The rotation and the codec are
**not** counted, so TFLOP/s is directly comparable between the two legs, and the
gap between that figure and the kernel's real instruction count is exactly what
the codec costs.

**Knobs:** everything in `common/benchmark.hpp` (`ASCEND_BENCH_WARMUP`,
`ASCEND_BENCH_ITERS`, `ASCEND_BENCH_BATCH`, `ASCEND_BENCH_MODES`,
`ASCEND_BENCH_CSV`, `ASCEND_BENCH_REPEATABLE`), plus `ASCEND_BENCH_TQ_CONTEXTS`
to restrict the context sweep.

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
| `stage2_hadamard` | query GM read, cast, `ApplyPi`, amax and scale, fp8 cast | **tasks, not S** |
| `stage3_l1_staging` | V -> MTE3 edge and `DataCopy` per chunk, `MTE3_V` per pass, the query's NZ staging | tiles |
| `stage4_score_gemm` | `PipelineAic`'s score half -- two `LoadData`, `Mmad`, `Fixpipe` -- and the `SlotReady` / `ScoresReady` / `SlotFree` flags | tiles |
| `stage5_full` | acc/state init, `qScaleInv_`, softmax, `ProbsReady`, context GEMM, `ContextReady`, accumulate, partial writeback | tiles |

**Stage 2 is not a tile-loop FWHT.** The cache is written already rotated, so
the split's only Walsh-Hadamard is the query's, once per query head of each kv
head. The output's inverse rotation is in the combine, which the ladder does not
time.

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
> first. Until that run happens, **there are no measured `fp16 us`, no measured
> speedup and no measured fp16 GB/s.** What exists is the traffic model in §7.5,
> which is exact arithmetic over the two layouts and not a measurement.

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
the expand be two integer shifts over the whole run instead of a lane-indexed
Gather. §13.10 is the whole of that change.

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

`bench_device_950pr_turboquant`, per `S in {64, 512, 1024, 2048}`, warmup 20,
100 iterations, at `B=1 H=8 D=256` over 2 kv heads with `block_size` 128:

| case | what it times | default |
|---|---|---|
| `tq4_write_s<S>` / `tq4_decode_s<S>` | the AIV-only 4-bit path, unchanged | **runs** |
| `kv4fp8_write_s<S>` | the 4-bit cache write through the multi-rate codec | **runs** |
| `kv4fp8_decode_s<S>` | the Cube-native 4-bit decode, split then combine | **runs** |
| `fp16_decode_s<S>` | the physical fp16 Cube baseline | **runs** |
| `fia_decode_s<S>` | the **stock CANN operator** baseline over the same fp16 paged cache | **attempted** |

`S=64` is in the sweep so the timings start at the one shape whose numerics have
been checked anywhere (§13.9); without it every timed shape is one whose fidelity
is unmeasured.

**The operator leg is `aclnnFusedInferAttentionScoreV5`, falling back to
`aclnnFusedInferAttentionScoreV2`, and it is expected to be absent on some
builds.** V1..V4 are withdrawn on an Ascend950 — planning returns 361001,
measured on CANN 9.1.0 — and V5's argument list has never been planned on
hardware. Both are tried at construction, before any timing, and the outcome is
recorded either way: a latency if one planned, or the operator name and the
planning status as a ctest-visible **skip** if neither did. A configuration where
no stock operator exists is a fact about the part, not a defect in this binary,
which is why it is a skip and not a failure.

It reads the **same** fp16 caches `fp16_decode` built rather than allocating its
own, so both sides of that comparison see identical bytes through identical
paging, and it is planned once at construction through `PlanAclnn` so the timed
region is the launch rather than the aclnn planner.

**All four legs run by default.** The gate that used to hold the Cube legs shut
was §13.8's fifth defect - `TurboQuantCubeMm::Init` allocating L0A/L0B/L0C on
both halves of the MIX kernel, which faulted the AIV half rather than returning
a wrong number - and that defect is closed. Both Cube kernels now allocate L1 on
both cores and L0 under `ASCEND_IS_AIC`; the same split was applied to
`TurboQuantFp16DecodeSplit::Init`, which had it too and would have taken the
fp16 baseline down with it.

What is still open is the score GEMM's operand form, and that is a wrong number
rather than a dead stream: the launch completes and the checksum is stable, so
it cannot take the queue down behind it. **The benchmark says so on every run**,
in the banner and again at the foot of the summary, because the summary is the
part that gets pasted into a report. Treat every Cube column as provisional
until §13.8's open defect is closed.

`VLLM_ASCEND_TQ_CUBE_WIP=0` drops the Cube legs, which is what to reach for if a
future defect does fault: a faulting launch takes the stream down and every case
queued behind it, so the switch is before the launch, and before *construction*
too - a dropped run never allocates the second and third KV caches either.

```bash
VLLM_ASCEND_TQ_CUBE_WIP=0 ./bench_device_950pr_turboquant   # AIV-only baseline
```

`-DVLLM_ASCEND_TESTS_ENABLE_WIP_CUBE=ON` is now **sim-tier only**: it flips the
default that `REQUIRE_CUBE_WIP_OPT_IN` reads in `test_sim_950pr_cube_gemm` and
`test_sim_950pr_turboquant_multimode`, and no longer has anything to say about
this binary. The environment variable is still read by both tiers, so setting it
explicitly keeps them in step. The Cube sources are **compiled in every
configuration** regardless; the switch decides what launches, not what builds.

**kv3fp4 and kv5fp8 have no leg here.** Both are still built, still dispatched
and still covered by the sim tier. Neither is on the 4-bit path this binary
measures, and a column for each was another KV cache of the same shape per
context length for a comparison nothing was asking of a benchmark whose subject
is the 4-bit ecosystem. `kv5fp8`'s fidelity - the one that clears cos > 0.995 -
is measured by `scripts/tq_multimode_calibration.py` and recorded in §13.3, not
here.

`test_device_950pr_turboquant`, the bare-metal correctness binary, needs no
switch of its own: it drives `turboquant_kernels.cpp` only and calls nothing in
`turboquant_mm_kernels.cpp`. Keep it that way.

The summary prints three tables, all keyed on the shape.

**1. Latency**, one row per `S`, columns `kv4 us` / `tq4 us` / `fp16 us` /
`fia us`. A leg with no sample prints a dash; `0.00` would read as a measurement.

**2. Speedups**, every one `kv4` against a denominator, because `kv4` is the
thing being proposed and the other three are what it would replace:

| column | what it is | what it isolates |
|---|---|---|
| `vs fp16` | against the unquantised fp16 Cube decode | the **codec**, and nothing else: same kernel, codec removed |
| `vs tq4` | against the AIV-only 4-bit decode | the Cube against the vector cores — but see below |
| `vs fia` | against the stock CANN operator | what a caller actually gains by switching |

**3. Derived rates**, now per leg rather than for `kv4fp8` alone: steps/s, GB/s
and TFLOP/s. Each leg's GB/s is **its own** traffic model over **its own**
measured time, so the column compares a leg to its own memory demand and not to
another leg's — a 4-bit cache and an fp16 one do not read the same bytes, which
is the entire point of the change.

**`vs tq4` no longer holds the codec fixed, and the old claim that it did is
now wrong.** It used to be true that `tq4` and `kv4fp8` stored identical nibbles
against identical thresholds. Since §13.9, `kv4fp8` bins against uniform
thresholds and packs coordinates `b` / `b + d/2` into a byte, while `tq4` still
bins against Lloyd-Max thresholds and interleaves `2b` / `2b + 1`. The cache
geometry is identical and the contents are not, so that ratio now moves the
quantiser and the compute unit together. Separating them again would mean keeping
a Lloyd-Max Cube leg alive purely as a benchmark control, which has not been
done.

A failure in the fp16 case is recorded as a **failure** and not a skip: it is a
kernel in the same binary, so if it does not run, something is broken rather than
missing. That is a different thing from the dropped-leg skip, which is a decision
taken before any launch, and from the operator leg's skip, which is a fact about
the CANN build.

**The comparison CSV.** `ASCEND_BENCH_TQ_COMPARE_CSV=<path>` writes one row per
shape with `batch,context_len,num_heads,head_dim,num_kv_heads,block_size`, each
leg's median, the operator that planned, the three speedups, per-leg GB/s and
TFLOP/s, per-leg p95, and both traffic-model byte counts. It is separate from the
generic `ASCEND_BENCH_CSV`, which is one row per (case, mode) and knows nothing
about which cases are baselines for which: that one is the raw record, this one is
the comparison. An absent leg leaves its fields **empty** rather than writing 0,
because this is the artifact most likely to be read without its banner.

**Every row is `B=1`.** The benchmark times one decode step per launch. The batch
axis is *validated* rather than timed, by `test_device_950pr_turboquant_multimode`
at batch `{1, 8}`; sweeping it here would multiply four context lengths of KV
cache by the batch and is a separate change.

The traffic model is printed either way and is still not a measurement. The kv4
rate gets its own table rather than two more columns on the 4-bit one, because
one line of it is easy to misread: the Cube decode reads each cached row once
per *kv* head rather than once per query head, since the query heads of a group
are batched into M. That is a property of the task decomposition, not of the
rate, and it is why the kv4 decode-read column is not the tq4 one - **the two
cache footprints are identical**, byte for byte, and only the decode-read column
separates them.

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
   to separate, and AscendC has no element-stride-2 vector move — `blkStride`
   counts 32-byte blocks — which is exactly why the old layout needed a Gather.
   Split by plane, both nibble planes are contiguous runs and the split is
   `>> 4` and `(x << 28) >> 28` on a `uint32` view of the run.
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
| `csrc/tests/sim/sim_hadamard_hybrid_kernels.cpp` | 954 lines. AIV-batched butterfly **and** Cube-factorised variants, double-buffered across kSlots |
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
longer tables. `sim_hadamard_hybrid_kernels.cpp` already runs a batched butterfly
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
