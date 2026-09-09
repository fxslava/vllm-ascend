# TurboQuant test suites: a review

Everything that validates the TurboQuant 4-bit rotated KV cache, in one place:
what each binary is for, what each case actually proves, what its bound is and
why it sits where it does, and — as important — what none of them cover.

Companion documents: [README.md](README.md) for how to build and run the suite,
[COVERAGE.md](COVERAGE.md) for how the C++ suite compares to the Python one.

---

## Contents

- [1. What is under test](#1-what-is-under-test)
- [2. The binary map](#2-the-binary-map)
- [3. The oracle, and what it is not](#3-the-oracle-and-what-it-is-not)
- [4. `test_turbo_quant_fidelity` — host-only](#4-test_turbo_quant_fidelity--host-only)
- [5. `test_turboquant_kernels_950pr` — contract and kernels](#5-test_turboquant_kernels_950pr--contract-and-kernels)
- [6. `test_turboquant_npu_simulator` — one decode pass end to end](#6-test_turboquant_npu_simulator--one-decode-pass-end-to-end)
- [7. `test_turboquant_bare_metal_950pr` — production shapes on silicon](#7-test_turboquant_bare_metal_950pr--production-shapes-on-silicon)
- [8. `bench_turboquant_950pr` — the AIV-only performance baseline](#8-bench_turboquant_950pr--the-aiv-only-performance-baseline)
- [9. Build and run matrix](#9-build-and-run-matrix)
- [10. Bounds, and where each number came from](#10-bounds-and-where-each-number-came-from)
- [11. What none of this covers](#11-what-none-of-this-covers)
- [12. What has actually been executed](#12-what-has-actually-been-executed)

---

## 1. What is under test

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

with `scale_slot = round_up(2 * num_kv_heads, 8)`. That padding is the reason
every DMA on this path is an aligned `DataCopy` and never a `DataCopyPad`, and
several cases below exist purely to keep it that way.

**Both kernels are vector-only.** There is no `Matmul`, no `Mmad`, no L1 copy
and no `[aic]` half anywhere in the source. That is a deliberate design point,
not an omission, and it is what `bench_turboquant_950pr` measures.

---

## 2. The binary map

| Binary | Runs on | Needs | The question it answers |
| --- | --- | --- | --- |
| `test_turbo_quant_fidelity` | any host | nothing — no CANN, no NPU | Is the *codec* right, and how accurate is it on real Qwen activations? |
| `test_turboquant_kernels_950pr` | camodel or silicon | Ascend C toolchain | Does the *device* do what the CPU reference does, and does the launch contract still agree across its five copies? |
| `test_turboquant_npu_simulator` | camodel or silicon | Ascend C toolchain | Does one complete decode step, quantised, land near exact fp32 attention? |
| `test_turboquant_bare_metal_950pr` | **silicon only** | Ascend C toolchain, a real 950PR | Does all of that still hold at the shapes the model actually decodes at, on the part? |
| `bench_turboquant_950pr` | silicon (usable), camodel (buildable) | Ascend C toolchain | What does the AIV-only path cost, and what does the 4-bit cache buy? |

The first three predate this document and are unchanged by it. The last two are
new, and exist because the first three are all bounded by what a cycle-level
simulator can finish: every shape in them is the smallest that still exercises
the thing it is there to exercise.

---

## 3. The oracle, and what it is not

`csrc/tests/reference/turbo_quant_cpu.h` is the reference every device case is
compared against. It mirrors `turboquant_codec_950.h` instruction for
instruction: the same LCG for the `Pi` diagonal, the same Lloyd-Max table, the
same `-128` store bias, the same rotated-basis decode.

**It is not an independent reimplementation and does not pretend to be.** What a
comparison against it establishes is that *the kernel does on the device what
the reference does on the host* — a strong statement about the port, and no
statement at all about whether the algorithm is the right one. The algorithm's
own accuracy is measured separately, against exact fp32 attention, by
`test_turbo_quant_fidelity` (on real activations) and
`test_turboquant_npu_simulator` (on the device).

Two consequences worth internalising before reading any bound below:

1. **Device and host cannot be bit-exact in general.** The RMS scale is a sum of
   squares; the device reduces it in a tree and the host sums it serially, so
   the two differ in the last bits. A coordinate sitting on a Lloyd-Max decision
   boundary can therefore legitimately fall either side of it, which is worth
   exactly one bin. Every packed-cache comparison is stated in *bin indices*
   with a one-bin tolerance for that reason — and never in reconstructed values,
   because the Lloyd-Max levels are unevenly spaced (one bin is worth anywhere
   between 0.257 and 0.664) so a value-space bound tight enough to catch a
   two-bin slip at the centre of the table would reject a legitimate tie-break
   at its edge.
2. **Except where the test removes the tie-break.**
   `PackedCacheIsByteIdenticalOnRotationExactInputs` constructs inputs on which
   no rounding difference is possible at all, and then demands byte equality
   with no tolerance. See §7.3.

---

## 4. `test_turbo_quant_fidelity` — host-only

**Purpose.** Measure and pin the codec itself, with no device in the loop, so
its accuracy on real activations can be tracked in CI on a build machine.

**Links neither `libascendcl.so` nor a kernel.** It is built even under
`-DVLLM_ASCEND_TESTS_HOST_ONLY=ON`, and it links `gtest_main` rather than
`common/main.cpp` precisely because that `main` initialises the ACL runtime.

**Assertion policy.** This binary is an *analytical reporter*. It prints cosine
similarity, SNR and relative L2 and asserts on none of them, so a change in
quantiser behaviour shows up as a number that moved rather than as a red build.
The only `EXPECT`s are on exact integer or involution identities that cannot
drift with the data.

### 4.1 Structural invariants — `TurboQuantCodecInvariants`

| Case | Validation target |
| --- | --- |
| `PiIsAnInvolution` | `Pi(Pi(x)) == x` to fp32 rounding at `d = 256`. If this fails the decode's un-rotation is not the inverse of the write's rotation and every output is in the wrong basis. |
| `PiPreservesDotProducts` | `Pi` is orthogonal, so `<Pi q, Pi k> == <q, k>`. This is the identity that lets the decode compute scores in the rotated basis without ever un-rotating K. |
| `PiSignVectorMatchesThePythonReference` | The `+-1` diagonal from the C++ LCG equals the one `vllm_ascend/attention/turboquant_v1.py` generates, channel for channel, at 128 and 256. A cache written by one half of the project has to be readable by the other. |
| `NibblePackRoundTripsExactly` | Every one of the 16 codes survives both nibble positions, including the extremes that make the byte `+127` and `-128`. |

### 4.2 Bounds and memory discipline — `TurboQuantPoison`

The codec runs in UB with scratch buffers sized for a full `batchRows`, and a
call with fewer live rows must confine itself to the live prefix. These five
cases poison the destination, the padding behind it and the scratch with two
disjoint alphabets and require that nothing poisoned reaches the output and
nothing outside the licensed slice is written:

`QuantizeIgnoresPoisonedDestinationAndPadding`,
`QuantizeWritesNoByteOutsideItsSlice`,
`DequantizePartialBatchIgnoresPoisonedTail`,
`DequantizeWritesNoByteOutsideItsSlice`,
`ApplyPiIgnoresSurroundingMemory`.

**Validation target:** the read and write extents of every codec entry point,
which is the class of bug a fidelity metric cannot see — a stray write shows up
as someone else's corruption, not as a worse SNR.

### 4.3 The table-driven path — `TurboQuantTableDriven`

The device does not compute the butterfly pairing or the nibble split; it
gathers them from a host-built constant-table image. These two cases run the
same gather-driven algorithm on the host, over `len in {64, 128, 256}` and
`batch_rows in {4, 8}` with partial batches of 1, 3 and 4 rows, and require it to
agree with the direct reference **bit for bit** (`FloatToBits` equality), with
poisoned scratch throughout.

**Validation target:** the table layout itself. A wrong offset in the image
produces a plausible-looking but wrong vector, and this is where that is caught
rather than on the device.

### 4.4 Numerical edge cases — `TurboQuantEdgeCases`

| Case | Validation target |
| --- | --- |
| `AllZeroVectorIsNumericallyStable` | The `kEps` floor. The table has no exact zero, so an all-zero vector reconstructs to about `-1.3e-21` rather than `0.0`; what matters is that it does not divide by zero or produce a NaN. |
| `SingleDominantOutlier` | The regime the rotation exists for: one channel far outside the rest. |
| `ClampingAndSaturationExtremes` | Values past the top and bottom of the table saturate into bins 0 and 15 rather than wrapping the packed byte. |
| `EveryNibblePairRoundTrips` | All 256 `(low, high)` nibble pairs. |
| `PartialBatchesMatchPerRowQuantisation` | A batched call equals row-by-row calls. |
| `InvolutionOnCanonicalBasisVectors` | `Pi^2 e_i == e_i` and `||Pi e_i|| == 1` for every basis vector, reported as a worst-case. |

### 4.5 Fidelity reports

| Case | What it measures | Known limitation |
| --- | --- | --- |
| `TurboQuantGoldenFidelity.Qwen35Layer3DecodeReport` | The codec on **real Qwen3.5 layer-3 activations** from `data/golden_layer3`: post-RoPE Q and K and the V slice of `tap_qkv`. | The dump is `pos=0, ctx_len=1`, so the softmax is over a single score and the attention context is just V. It measures the codec on real activations and **not** the online-softmax accumulation. Skips with a `git lfs pull` message when the dump was never fetched. |
| `TurboQuantSyntheticFidelity.LongContextDecodeReport` | A 300-position context spanning three paged blocks, the last partial, at `d = 128`, 8 heads over 2 kv heads. | Synthetic activations. It exists to cover exactly what the `ctx=1` golden dump cannot: the accumulation across block boundaries. |

---

## 5. `test_turboquant_kernels_950pr` — contract and kernels

Two quite different things live in this binary.

### 5.1 The launch contract — `TurboQuantLaunchContract`, no device needed

The scale-plane geometry, the codec table image and the grid arithmetic exist in
**five places** that can drift independently: the Ascend C kernel, the Torch
adapter, the Python backend, the CPU reference, and the Torch-free test shim in
`common/turboquant_launch.hpp`. These five cases pin the ones that are pure
functions of the shapes, and run with no device at all.

| Case | Validation target |
| --- | --- |
| `ScaleSlotMatchesTheCpuReference` | `scale_slot` agrees between the shim and the reference for `num_kv_heads` 1..16, is always a whole 32-byte burst, and always fits K and V. |
| `CodecTableWordsMatchTheLayoutContract` | The image is `7D + 2DB + 16` words for `D in {64, 128, 256}` and `B in {1, 16}`, the built image is that long, and the length is a whole burst — `Init()` moves it with one `DataCopy`. |
| `CodecTablesHaveTheDocumentedLayout` | Every section of the image, read back and checked against its documented meaning: the three butterfly stages' signs and XOR partners, the even/odd nibble offsets, the batched expansion offsets and selectors, and the 16 Lloyd-Max centroids. |
| `PiSignsMatchTheCpuReference` | The sign vector at 64, 128 and 256, plus the first eight values at 128 recomputed from the documented LCG rather than transcribed — so it stays a check on the generator, not on a copied literal. |
| `GridPlansMatchTheAdapterArithmetic` | Coverage and non-over-subscription of the write grid; the split cap, the per-sequence block cap, the workspace size and `combine_block_dim <= split_block_dim` for the decode; and that zero tokens produces a no-op rather than a zero-sized grid the runtime would reject. |

### 5.2 The kernels — `TurboQuantKernels`, gated on `REQUIRE_ASCEND_950PR`

Shapes are deliberately tiny — `head_size 64`, `block_size 16`, `context 32`,
4 heads over 2 kv heads — because this file has to finish under the camodel.
Each one is still the smallest that keeps its feature live: 64 is the bottom of
the supported range and still a four-stage Walsh-Hadamard, 16 is exactly one
`kTileRows` tile, 32 is two blocks read through a non-identity block table, and
the GQA group of two makes the head→kv_head mapping non-trivial.

| Case | Validation target | Bound |
| --- | --- | --- |
| `ReshapeAndCacheMatchesTheCpuReference` | The write path: rotate, quantise, scatter, for both planes and the scale plane. | Bin drift `<= 1`, differing channels `<= 2%`, scale relative error `<= 1e-5`. Measured drift on the camodel is **0.000** — all 4096 channels land on the same code. |
| `ReshapeAndCacheLeavesNegativeSlotsUntouched` | A `-1` slot is how the scheduler marks a padded token. The kernel must skip it entirely — not write zeros, not write it elsewhere — or a padded step corrupts whatever block `-1` aliases to. | The skipped slot is byte-for-byte still zero. |
| `PagedAttentionMatchesTheCpuReference` | The split/combine decode, reading back **the cache the device itself wrote**, so a disagreement is the decode and not the write path. | `cos > 0.9995`, `relL2 < 5e-3`. Tight because both sides keep an fp32 accumulator and round once at the store. |

---

## 6. `test_turboquant_npu_simulator` — one decode pass end to end

**Purpose.** The only case that compares the quantised device output against
*exact fp32 attention* rather than against the CPU TurboQuant reference. One
decode step, one pass, at the same tiny shape and for the same reason.

Three outputs are compared:

- **quantised** — the two kernels on the device.
- **control** — the same decode with an unquantised fp16 KV cache, through
  `aclnnFusedInferAttentionScoreV2`. *Attempted, never required.*
- **exact** — fp32 attention over the same fp16 inputs, on the host. Always
  available, which is why the assertions hang off it.

| Assertion | Bound | Why it sits there |
| --- | --- | --- |
| quantised vs the CPU TurboQuant reference | `cos > 0.999`, `relL2 < 5e-3` | Same algorithm both sides; anything beyond fp16 output rounding is a kernel bug. This is the regression detector. |
| quantised vs exact fp32 | `cos > 0.97`, `SNR >= 12 dB`, `relL2 < 0.30` | A **floor**, not a target. This is the scheme's error, not the kernel's: the shape measures ~16.7 dB on the camodel and ~18 dB on the host at a different seed, and the bound sits well clear because the number moves with the data. It catches gross breakage — a dropped un-rotation, a scale read from the wrong lane, a block table ignored — all of which collapse the cosine rather than nudging the SNR. |

**Two things this case explicitly cannot show.** It does not discriminate "the
`Pi` rotation was dropped": on i.i.d. Gaussian channels a bare absmax quantiser
scores the same ~18 dB, because what the rotation buys is protection against
anisotropic, outlier-heavy channels — which is measured on real activations by
`test_turbo_quant_fidelity` instead. And timing printed here is the simulator's
speed, not the part's, which is why it is printed and never asserted.

**The fp16 control does not run on an Ascend950.** Measured on CANN 9.1.0, the
planning call returns `361001`: *"Interface aclnnFusedInferAttentionScore
versions V1 to V4 are no longer supported on Ascend950"*. The leg is kept because
it is the right control wherever the operator exists, and because a test that
silently omits its control is worse than one that prints why it has none. See
`common/aclnn_ops_950pr.hpp`.

---

## 7. `test_turboquant_bare_metal_950pr` — production shapes on silicon

**Purpose.** Everything above is bounded by what a cycle-level simulator can
finish. This binary is what runs on the part, at the shapes Qwen3.5-2B actually
decodes at.

**Shape:** `head_dim 256`, 8 query heads over 2 kv heads (a GQA group of four),
`block_size 128`, context `512 / 1024 / 2048`, one query token. `head_dim 256` is
the top of the range the codec sizes UB for and a width the 310P cannot run at
all. The block pool is four times the blocks a context needs, so the block table
is a genuine scatter rather than a consecutive run that would stay resident.

### 7.1 Why the camodel is excluded rather than merely slow

Under `RUN_MODE=sim` the camodel reports a genuine `Ascend950PR_*` bin from
`aclrtGetSocName()` — that is what makes the simulator useful, and it is exactly
why the SoC name cannot distinguish it from silicon. So `REQUIRE_ASCEND_950PR`
would pass and the suite would start, and then run for days on the `S=2048`
cases, looking like a hang rather than a skip.

`REQUIRE_PHYSICAL_ASCEND_950PR` asks a different question: it scans
`/proc/self/maps` for `libruntime_camodel.so` or anything loaded out of a
`tools/simulator/` directory, and skips with the offending path in the message.
Silicon is the default assumption — a host whose `/proc` cannot be read reports
no evidence rather than skipping.

Observed skip message on this workstation's camodel build:

```
Test targets a physical Ascend 950PR and this process has the CANN camodel loaded
(/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/simulator/dav_3510/lib/libruntime_camodel.so).
Rebuild with -DRUN_MODE=npu and run on the part, or set ASCEND_TEST_ALLOW_SIMULATOR=1
to run it here anyway (hours).
```

Two environment overrides exist, both for smoke-checking the binary and neither
for producing results:

| Variable | Effect |
| --- | --- |
| `ASCEND_TEST_ALLOW_SIMULATOR=1` | Runs the suite under the camodel anyway. |
| `ASCEND_TQ_BARE_METAL_CONTEXTS=16` | Replaces the `512,1024,2048` sweep, so such a run is merely slow rather than pointless. |

### 7.2 Case by case

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
channel is required to round-trip through fp16 unchanged before anything else
runs. The construction needs `1/sqrt(D)` to be a power of two, which holds for
`D = 64` and `D = 256` and is stated as such at the construction site.

---

## 8. `bench_turboquant_950pr` — the AIV-only performance baseline

**Purpose.** Establish what the vector cores alone can do on this path, before
any Cube-assisted INT4 GEMM path exists, so that a later comparison means
something.

**Why "AIV-only" is the headline.** Nothing in `turboquant_kernels.cpp` touches
the Cube: the score row is a vector `Mul` plus a `ReduceSum`, the value
accumulation is a vector FMA, the Walsh-Hadamard is `Add`/`Sub` and `Gather`, the
codec is compares and a `Gather` over a 16-entry table. Grep the source for
`Matmul`, `Mmad` or an `[aic]` block and it comes back empty.

**Cases**, per context length `S in {512, 1024, 2048}`:

| Case | What is timed | Notes |
| --- | --- | --- |
| `tq4_write_s<S>` | The cache write path: `S` tokens rotated, quantised and scattered. One launch. | Prefill-shaped — a decode step writes one token — and here because the cache has to be filled before it can be read and the fill is not free. |
| `tq4_decode_s<S>` | The decode: split then combine, **two** launches on one stream. | The cache is written once during setup, so the timed region is the read path only. `tasks_per_launch = 2`, so the harness's queue-depth accounting knows this case submits two tasks per iteration. |
| `fp16_decode_s<S>` | The same decode with an unquantised fp16 paged KV cache, through `aclnnFusedInferAttentionScoreV2`. | Expected to **skip** on this part; see below. |

Each case carries a checksum the harness verifies after warmup and again after
the last timed iteration — the standard defence against timing an operator that
has stopped writing its output. The write path's checksum is the whole scale
plane, so a scale written outside the slot mapping is caught too.

### 8.1 The fp16 baseline, and what carries the comparison in its absence

The fp16 leg is attempted and, on an Ascend950, refused at planning time with
the same `361001` the simulator test records. That is a fact about CANN, not a
broken benchmark, so it is registered as a ctest-visible **skip with the reason
attached** rather than quietly omitted.

What carries the comparison in the meantime is the **traffic model** the binary
prints before the table. It is not a measurement and does not pretend to be:
it is exact arithmetic over the two cache layouts. A decode step is
bandwidth-bound — it streams the whole context and does two vector operations
per element it reads — so the ratio of bytes moved is the ratio the latency
tends toward once the pipeline is full.

Two quantities, deliberately kept apart:

- **footprint** — what the KV cache for `S` tokens occupies in HBM. The capacity
  win. Independent of how attention reads it.
- **decode read** — what one decode step streams. The bandwidth win. Larger per
  byte of cache than the footprint suggests, because every query head re-reads
  its kv head's rows: with 8 heads over 2 kv heads each cached row is read four
  times. Whether L2 absorbs some of those re-reads is a hardware question the
  model deliberately does not guess at — it counts what the kernel *asks the
  memory system for*, under the same convention as the fp16 leg, so the ratio is
  fair even where the absolute figures are pessimistic.

At `head_dim 256`, `kv_heads 2`, per token:

| Quantity | fp16 | TurboQuant 4-bit | Ratio |
| --- | --- | --- | --- |
| cache footprint per token | `2 * kv * D * 2` = 2048 B | `2 * kv * D/2` + `scale_slot * 4` = 512 + 32 = 544 B | **3.76x** |
| decode read per token per head | `2 * D * 2` = 1024 B | `2 * D/2` + `scale_slot * 4` = 256 + 32 = 288 B | **3.56x** |

The footprint ratio is exactly 3.76x at every length. **The decode-read ratio is
not**: the per-token figure above is an asymptote, and the model also counts the
query and output rows and the flash-decoding partials, which are fixed in `S`
and dilute it at short contexts. Modelled end to end at 64 vector cores it comes
out around 3.3x at `S = 512`, 3.35x at 1024 and 3.45x at 2048, and only 2.23x at
the `S = 16` shape a camodel smoke run uses. The binary prints the exact figure
for whatever it actually ran; do not carry a number out of this table without
the length it belongs to.

The scale plane is 5.9% of the 4-bit cache. At `num_kv_heads = 2` a token's slot
is 8 fp32 lanes carrying 4 live ones — `round_up(2 * 2, 8)` — so **half** of that
plane is burst padding, 2.9% of the whole 4-bit cache, and that is what buys
every DMA on this path being an aligned `DataCopy` rather than a `DataCopyPad`.
The binary derives all of it at run time from the same `ScaleSlotFloats` the
kernel uses, so these figures cannot drift from the table without the table
being wrong.

(The layout note at the top of `turboquant_kernels.cpp` quotes this overhead as
"25% at `num_kv_heads = 2`" and "12.5% from 4 upward". Neither matches
`round_up(2 * kv, 8)`, which pads 4 lanes to 8 at `kv = 2` and pads nothing at
all at `kv = 4`. The layout is right and its comment's percentages are not;
believe the arithmetic.)

### 8.2 Reported metrics

The shared harness reports min / median / mean / P95 / P99 / stddev per case
across three timing modes (`pipelined`, `device`, `host`), plus TFLOP/s and GB/s.
On top of that this suite prints its own summary: median pipelined latency,
**decode steps per second**, KV GB/s from the traffic model's decode-read bytes
over the measured time, TFLOP/s, and the ratio to the fp16 leg where that leg
produced numbers.

`flops_per_iteration` counts the attention math only — `QK^T` and the value
accumulation, two operations per element of each. The rotation and the codec are
**not** counted, so TFLOP/s is directly comparable between the two legs, and the
gap between that figure and the kernel's real instruction count is exactly what
the codec costs.

### 8.3 Knobs

Everything in `common/benchmark.hpp` (`ASCEND_BENCH_WARMUP`, `ASCEND_BENCH_ITERS`,
`ASCEND_BENCH_BATCH`, `ASCEND_BENCH_MODES`, `ASCEND_BENCH_CSV`,
`ASCEND_BENCH_REPEATABLE`), plus:

| Variable | Effect |
| --- | --- |
| `ASCEND_BENCH_TQ_CONTEXTS=512,1024` | Restricts the context sweep. |

Under `RUN_MODE=sim` the benchmark is built but **disabled in ctest**: a
default-sized run is ~1300 launches per case, which is days on a cycle-level
simulator. Run the binary by hand with the knobs above if you need a smoke
check.

---

## 9. Build and run matrix

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**Host-only — no CANN, no NPU:**

```bash
cmake -S csrc/tests -B build/csrc-tests-host -DVLLM_ASCEND_TESTS_HOST_ONLY=ON
cmake --build build/csrc-tests-host -j
./build/csrc-tests-host/test_turbo_quant_fidelity
```

**950PR, silicon (`RUN_MODE` defaults to `npu`):**

```bash
cmake -S csrc/tests -B build/csrc-tests-950pr -G "Unix Makefiles" \
      -DENABLE_ASCEND_950PR=ON -DSOC_VERSION=Ascend950PR_9599
cmake --build build/csrc-tests-950pr -j

ctest --test-dir build/csrc-tests-950pr -R turboquant -LE benchmark
./build/csrc-tests-950pr/bench_turboquant_950pr
```

**950PR, camodel:**

```bash
cmake -S csrc/tests -B build/csrc-tests-950pr-sim -G "Unix Makefiles" \
      -DENABLE_ASCEND_950PR=ON -DSOC_VERSION=Ascend950PR_9599 -DRUN_MODE=sim
cmake --build build/csrc-tests-950pr-sim -j

export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH
./build/csrc-tests-950pr-sim/test_turboquant_npu_simulator
```

Use the **Makefiles generator**. With Ninja, `ascendc_library()`'s nested
ExternalProjects hit a CANN 9.1.0 bug in `extract_host_stub.py` and the build
dies with a bare Python `KeyError`; the sub-builds are pinned for that reason in
`kernels/ascend/turboquant/CMakeLists.txt`.

**Incremental kernel builds break intermittently.** Re-running the build after
touching anything the `ascendc_library()` sub-projects consume can fail with
`ld.lld: ... unknown file type` on a `turboquant_kernels.cpp.o` that compiled
cleanly moments earlier. It is not a source problem: delete the build directory
and configure again. Only the kernel sub-builds are affected, so editing a test
or the benchmark alone is usually safe.

The camodel writes several hundred `core*.dump` and `profile_*.toml` files into
the **process working directory**, and nothing in `.gitignore` covers them. Run
from a scratch directory.

Which binaries appear where:

| Configuration | fidelity | kernels | simulator | bare-metal | bench |
| --- | --- | --- | --- | --- | --- |
| `HOST_ONLY=ON` | built, runs | — | — | — | — |
| 310P default | built, runs | — | — | — | — |
| `ENABLE_ASCEND_950PR=ON`, `RUN_MODE=npu` | built, runs | built, runs on the part | built, runs on the part | built, **runs on the part** | built, runs on the part |
| `ENABLE_ASCEND_950PR=ON`, `RUN_MODE=sim` | built, runs | built, runs on the camodel | built, runs on the camodel | built, **skips** | built, ctest-disabled |
| `BUILD_TURBOQUANT_KERNELS=OFF` | built, runs | — | — | — | — |

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

**The 4-bit fidelity ceiling is information-theoretic, not an implementation
gap.** Scalar Lloyd-Max delivers 20.22 dB per coordinate and *is* the scalar
optimum on a Gaussian, which the rotation makes the marginals. Adding scalar
entropy coding (~0.4 dB) and a practical lattice VQ gain (~0.75 dB) reaches only
~21.4 dB. Do not read a bound above as headroom someone could tune into
existence.

---

## 11. What none of this covers

- **Silicon.** As of this document every TurboQuant device case has executed on
  the arch35 camodel and none on a physical 950PR.
  `test_turboquant_bare_metal_950pr` and `bench_turboquant_950pr` are written
  *for* that run and skip until it happens; that is their status, not a defect.
- **An fp16 latency baseline on this part.** `aclnnFusedInferAttentionScore`
  V1–V4 are withdrawn on an Ascend950 and no V5+ interface is exposed by the
  release this suite builds against. Until one is, the fp16 comparison is the
  traffic model, and the traffic model is bytes rather than time.
- **bf16.** Both kernels take an `AscendType` and every case passes `FP16`.
- **Prefill.** Every decode case is one query token. The write path is exercised
  at prefill *lengths*, but no attention case has `num_tokens > 1`.
- **Batch.** One sequence, one block table.
- **head_size other than 64, 128 and 256**, and block sizes other than 16 and
  128.
- **A rotation ablation on real activations at the device level.** The
  simulator's bounds cannot tell `Pi` from a bare absmax quantiser on i.i.d.
  Gaussian channels; only `test_turbo_quant_fidelity`'s golden case sees real
  activation structure, and it is a report rather than an assertion.
- **Performance regression thresholds.** The benchmarks report numbers and check
  that the kernels still produce their output. Nothing fails on a slowdown.

---

## 12. What has actually been executed

Recorded so the difference between "written" and "run" stays visible.

Environment: WSL Ubuntu 22.04, CANN 9.2.0-beta.2 native, `Ascend950PR_9599`
platform config, camodel reporting `soc='Ascend950PR_9589'` with 64 vector cores
and 40 GiB HBM. No physical part.

**Build, clean under `-DVLLM_ASCEND_TESTS_WERROR=ON` with zero warnings, in all
four configurations:** `HOST_ONLY=ON`; the 310P default;
`ENABLE_ASCEND_950PR=ON -DSOC_VERSION=Ascend950PR_9599` with `RUN_MODE=npu`; and
the same with `RUN_MODE=sim`. `ctest -N` enumerates 384 tests in the npu tree.

**Executed:**

| Binary | Where | Result |
| --- | --- | --- |
| `test_turbo_quant_fidelity` | host | 19/19 pass. Golden layer-3 report: attention context `cos 0.995593`, `20.54 dB`; layer output vs the fp16 dump `cos 0.998752`, `26.02 dB`. Synthetic 300-token context: `cos 0.995849`, `20.80 dB`. |
| `test_benchmark_harness` | host, no device | 17/17 pass after the harness changes below. |
| `test_turboquant_bare_metal_950pr` | camodel, `RUN_MODE=sim`, default gate | 7/7 **skip**, each naming the camodel library found in `/proc/self/maps`. The gate works. |
| `test_turboquant_bare_metal_950pr` | camodel, `ASCEND_TEST_ALLOW_SIMULATOR=1 ASCEND_TQ_BARE_METAL_CONTEXTS=16`, determinism case excluded | **6/6 pass.** Write path: max bin drift **0**, 0 of 8192 channels differ, worst scale relative error `2.478e-07`. Bit-exact case: **8192 packed bytes byte-identical** to the CPU reference, scale exactly 1.0. Decode: `cos 1.000000`, `73.97 dB`, `relL2 2.0e-4`. Layout: 32.0 KiB fp16 vs 8.5 KiB 4-bit, **3.76x**, scale plane 5.9%. Bounds: 996 poisoned rows survived, 28 live rows written, 0 live rows left unwritten. |
| `bench_turboquant_950pr` | camodel, `S=16`, one iteration, host mode | Traffic model and the summary print; the fp16 leg skips with the `361001` refusal quoted in full. Timings are the simulator's and are not results. |

**Not executed:** `RepeatedDecodeLaunchesAreBitIdentical` (nine decode passes is
hours under the camodel), and every case at `S = 512 / 1024 / 2048`. Both wait on
silicon, which is what this binary is for.

Two defects were found by running the above and fixed in the shared harness:

- `BenchmarkRunner::WarmUp` took the reference checksum with no launch behind it
  at `ASCEND_BENCH_WARMUP=0`, so every case with a checksum failed at the end of
  the timed loop claiming "the launches are not all computing the same thing"
  when the baseline had simply never been computed. A case with a checksum now
  always gets at least one warmup launch.
- The TurboQuant binaries built with `RUN_MODE=npu` abort during ctest's
  `PRE_TEST` GTest discovery on a host with no NPU driver, which ended the whole
  enumeration and made `ctest -N` list nothing for the entire project. They are
  now registered one ctest entry per binary; see the note at
  `vllm_ascend_register_correctness_test` in `CMakeLists.txt`.
