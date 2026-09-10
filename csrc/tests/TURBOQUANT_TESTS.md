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
│   └── bench_device_950pr_turboquant.cpp
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
| **Binaries** | `test_device_950pr_turboquant`, `test_device_950pr_{matmul,rmsnorm,rotary_embedding,activation_swiglu,qwen_layer_golden}`, `test_device_950pr_benchmark_harness`, `bench_device_950pr_turboquant` |
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
| `ASCEND_TQ_BARE_METAL_CONTEXTS=16` | Replaces the `512,1024,2048` sweep, so such a run is merely slow rather than pointless. |

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
**No physical part has been available.**

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
