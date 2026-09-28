# QJL 3+1 codec — Phase 1: audit, specification, CPU reference, and the Phase 2 blueprint

Branch `feat/tq-qjl-3plus1-codec`, off `feat/tq-v0.26-clean`. 2026-09-25.

Companion to [`QJL_VS_INT4_FINDINGS.md`](QJL_VS_INT4_FINDINGS.md) (2026-09-22), which this
phase audits, confirms, and extends in the one direction it did not cover.

## Verdict

**The layout half of the brief is already satisfied by the bytes in the cache, and the
unbiasedness half is not a layout property at all.** Those are two separate findings and they
point in opposite directions:

1. **Zero layout drift is achievable because there is nothing to move.** The shipped 4-bit word
   already *is* 1 sign bit plus a 3-bit magnitude. This phase pins that as a compile-time proof
   and a runtime test rather than a claim. The cost of the guarantee is that the reconstruction
   is unchanged too: a relabelling cannot improve fidelity, and the 2026-09-22 measurements
   (bit-identical, max abs diff `0.00e+00`) stand.
2. **`E[<q̃, k̃>] = <q, k>` is reachable, exactly — but through the encoder, not the packing.**
   Every codec in the 2026-09-22 study is deterministic, and no deterministic quantizer can be
   unbiased: `E[k̂ | k] = Q(k) ≠ k` for any bit layout. Stochastic (dithered) rounding of the
   level index gives exact unbiasedness, costs one uniform draw per coordinate in the encoder,
   costs the decoder nothing, and changes no stored byte. That is genuinely new on this tree.

The trade it makes is measured here and is not free: **√2 on the single-draw error**. Whether
that is worth paying on attention output is a Phase 2 measurement, not a claim this phase makes.

## Step 1 — audit of the existing QJL artifacts

| artifact | state |
| --- | --- |
| `tests/research/qjl_vs_int4_ablation.py` | present, 8 codecs (`c_int4_*`, `c_qjl_*`), numpy, untracked in git |
| `tests/research/QJL_VS_INT4_FINDINGS.md` | present, full result tables incl. the 2026-09-23 softmax-KL pass |
| `scripts/tq_kv_quant_reference.py` | the rotation (`apply_pi`, `pi_signs`) and both shipped codecs, pinned bit-exact against `csrc/tests/reference/turbo_quant_cpu.h` |
| **CUDA digital twin** | **does not exist.** `find . -name "*.cu"` returns nothing; there is no CUDA source anywhere in the repository. The harness above is the only digital twin, and it is numpy on the host. |

### Projection matrix generation

There is no Rademacher/WHT *sign sketch* to extract, because the tree does not use one. What it
uses is the orthogonal rotation `Pi = D H D` — `apply_pi` / `pi_signs` in
`scripts/tq_kv_quant_reference.py`, `ApplyPi` in `turboquant_codec_950.h`, `turboquant_rotation.py`
on the host. `D` is a deterministic per-channel sign pattern and `H` the normalised FWHT; it is
applied to the *full* head dimension and is invertible (`Pi² = I`), so it loses nothing and
projects nothing away. The QJL construction's random projection to `m < d` dimensions has no
counterpart here, and adding one would not preserve the layout the brief requires.

This matters for the estimator: because `Pi` is orthogonal, `<Pi q, Pi k> = <q, k>` exactly, so
an encoder that is unbiased in the rotated basis is unbiased on the original inner product. That
is what makes the stochastic-rounding route work without touching the rotation at all.

### 3-bit magnitude centroid tables

Both shipped grids are symmetric, so their magnitude halves are exactly the 3-bit tables the
brief asks for — extracted, not re-derived:

| grid | magnitude table | source |
| --- | --- | --- |
| mid-rise (KV4_FP8, Cube) | `m + 0.5` for `m` in `[0, 8)` — no table needed, one `Adds` | `TurboQuantModeTraits<KV4_FP8>::kCentroids` |
| Lloyd-Max (AIV default) | `0.12840, 0.38805, 0.65676, 0.94234, 1.25623, 1.61805, 2.06902, 2.73259` | `TURBOQUANT_LLOYD_MAX_CENTROIDS[8:]` |

## Step 2 — codec specification

`csrc/attention/turboquant/op_kernel/common/qjl_codec.h` — placed beside `turboquant_layout.h`
(which is where that file actually lives) and dependency-free for the same reason: the Torch-free
host tests and `op_host/turboquant_tiling.h` include it without a kernel toolchain.

### The correction the brief needs

The brief's decomposition is `sign = (val >> 3) & 0x1; magnitude_idx = val & 0x7`. The sign is
right. **The magnitude is not** — `val & 0x7` is the magnitude only on one side of zero. Half the
code space counts outward from the grid centre and half counts inward, so one half has to be
complemented. Branch-free, no select:

```c
mask = (0 - sign_bit) & 0x7      /* 0x00 or 0x07, the sign bit replicated */
mag  = (val ^ mask) & 0x7        /* ones'-complement abs, 3 bits wide     */
```

Taking `val & 0x7` alone decodes every negative coordinate to the mirrored magnitude
(e.g. `-3.5` reads as `-4.5`) and nothing raises.

### The two sign words — the trap for Phase 2

Both shipped writers store a bin index `b` in `[0, 16)`, and **they do not store the same bits**:

| plane | low nibble | high nibble | level of bin `b` | bit 3 set means |
| --- | --- | --- | --- | --- |
| AIV `TurboQuantCodec<4>`, int8 `lo + 16·hi − 128` | `b_even` | `b_odd ^ 8` | Lloyd-Max `C[b]` | **positive** (low), negative (high) |
| Cube `TurboQuantModeCodec<KV4_FP8>`, `int4b_t` pairs | `b_even ^ 8` | `b_odd ^ 8` | `b − 7.5` | **negative** |

The AIV byte's `−128` bias is the cause: `−128 ≡ +128 (mod 256)`, so the bias flips bit 7 — the
MSB of the *high* nibble only. The Cube plane stages `n = b − 8` through `int4b_t`, whose
two's-complement nibble carries the bits of `b ^ 8` in both halves. An unpack that assumes one
convention on the other plane decodes odd coordinates off by 8 and returns plausible numbers.

Both polarities are named (`SignPolarity`) and both are proven over the whole code space.

### What is proven, and where

Six `static_assert`s in the header itself, so a Phase 2 edit to the bit arithmetic **cannot
compile**:

- the 3+1 read of the KV4_FP8 word equals `TurboQuantModeTraits<KV4_FP8>::kCentroids`;
- the 3+1 read of the AIV word equals `TURBOQUANT_LLOYD_MAX_CENTROIDS`;
- `(sign, magnitude) → nibble → (sign, magnitude)` is the identity on all 16 codes, both
  polarities;
- the AIV byte's `−128` bias is confined to bit 7;
- the 4-bit footprint constants have not moved.

The header restates the level tables in its own terms, so it is *also* held to the tables the
kernels compile against, at runtime, by `TurboQuantQjlCodec.SignMagnitudeReadsTheShippedGrids` in
`csrc/tests/host/test_host_turboquant_fidelity.cpp`. That test is what gives the header a
`-Werror` gate at all: a header nobody includes compiles zero times.

## Step 3 — CPU reference and the measurements

`vllm_ascend/attention/qjl_reference.py` (torch, no NPU) and
`tests/ut/attention/test_qjl_codec.py` (9 tests).

### Layout: bit-exact, against the shipped writers and the shipped reader

- The packed plane for the AIV word is **byte-identical** to `low + 16·high − 128`, and for the
  Cube word to `int4b_t` pairs of `b − 8`.
- `turboquant_layout.turboquant_dequantize` — the host's only reader of the packed format —
  decodes the reference's bytes to exactly what the sign/magnitude split decodes.
- A full cache plane still packs to `(num_blocks, block_size, num_kv_heads, head_size // 2)`
  int8, `numel / 2` bytes.
- The 3+1 decode reproduces the shipped mid-rise reconstruction with **max abs diff exactly
  `0.0`** — the 2026-09-22 relabelling verdict, now a regression test.

### Estimator: inner-product bias, D=128, 512 keys

Measured **per key**, deliberately not pooled: on a symmetric key distribution the deterministic
error averages towards zero *across* keys, so a pooled mean reads almost unbiased while every
individual logit is still off. `E[k̂ | k]` is what the softmax sees.

| encoder | per-key bias rms | as a fraction of mean \|⟨q,k⟩\| = 8.467 |
| --- | --- | --- |
| 3+1 deterministic (= shipped mid-rise) | 1.2095 | 14.3% |
| asymmetric uniform (affine min/max + zero point) | 1.0384 | 12.3% |
| **3+1 stochastic, `E[·]` in closed form** | **5.7e-06 (max)** | **6.8e-07** |

The unbiased row is the *expectation*, evaluated in closed form rather than sampled, so the
residual is only fp32 round-off. Sampling converges on it at exactly the expected rate:

| trials | per-key bias rms | ratio vs ¼ the trials |
| --- | --- | --- |
| 32 | 0.2946 | — |
| 128 | 0.1500 | 1.96 |
| 512 | 0.0706 | 2.12 |
| 2048 | 0.0368 | 1.92 |

(`1/√4 = 2.00` expected.) On long-tail keys (Student-t, df=3, with 16× outlier channels) the
deterministic bias does **not** average out even pooled — mean bias `+1.204` against a mean
`|⟨q,k⟩|` of 33.8 — while the closed-form stochastic bias stays at `2.4e-04`, i.e. round-off.

### The cost

Single-draw error rms, same fixture: stochastic **1.7052** vs deterministic **1.2095** = **1.41×**,
the √2 the theory predicts (per-coordinate error power rises from ≈`step²/12` to ≈`step²/6`).
Unbiasedness buys accuracy *in aggregate over many keys*; a single decoded key is noisier. For a
softmax over a long context that is the right trade in principle, and it is exactly the thing
Phase 2 has to measure on `O` rather than assume — this tree has already established twice that
only about a third of a reconstruction-side gain survives the softmax.

### Why the default scale is absmax

Exact unbiasedness holds **only while nothing clips**. `compute_scale` therefore rounds the scale
*up* through fp16 (round-to-nearest can land below `absmax/7.5` and push the peak coordinate past
the outermost level). With `ScaleMode.RMS` the tail does clip, and the clipped coordinates keep a
bias no amount of dithering removes; that asymmetry is pinned as a test so the default cannot
drift silently.

## Step 4 — Phase 2 integration blueprint

### The thing to check before anything else

**The word this header's mid-rise path targets is test-only.** Confirmed against the two CMake
source lists:

- the wheel (`CMakeLists.txt`, `VLLM_ASCEND_TURBOQUANT_KERNEL_SOURCES`) builds
  `op_kernel/turboquant_paged_attention.cpp` (shipping AIV 4-bit) and
  `op_kernel/turboquant_rotate_q.cpp`;
- `csrc/tests/CMakeLists.txt` adds `op_kernel/turboquant_fused_decode.cpp` — the Cube-native
  KV4_FP8 path — which the wheel does **not** build.

So a stochastic encoder landed only in `StageAffinePlane` would never run in production. Decide
the target first.

### Files, in dependency order

| # | file | change | risk |
| --- | --- | --- | --- |
| 1 | `op_kernel/common/qjl_codec.h` | **done in Phase 1.** No further change needed for the decode. | — |
| 2 | `op_kernel/common/turboquant_codec_950.h` → `ComputeLevelBins` / `Quantize4Bit` | the **shipping** encoder. Stochastic variant of the bin search. | high — see below |
| 3 | `op_kernel/common/turboquant_codec_mx.h` → `ComputeLevelBins` / `StageAffinePlane` | the **test-only** Cube encoder. Cheapest place to prototype: the grid is uniform. | low |
| 4 | `op_host/turboquant_tiling.{h,cpp}` | one extra UB buffer for the RNG state; `WorkBufferWords` grows by `vecLen`. | medium — UB headroom is the known pressure point on this kernel family (the kv4fp8 split's query FWHT already reaches 92% of UB at D=512), so budget it before writing it |
| 5 | `op_adapter/turboquant_torch_adpt.h` + `turboquant_torch_ops.h` | a seed argument on `npu_turboquant_reshape_and_cache`. | ABI change |
| 6 | `vllm_ascend/attention/turboquant_v1.py` | pass the seed; nothing else. `turboquant_layout.py` needs **no** change. | low |
| 7 | `csrc/tests/reference/turbo_quant_cpu.h` | the host twin of the new encoder, for the goldens. | low |

**`turboquant_vector_service.h` and `turboquant_cube_service.h` do not need to change.** They call
`codec_.Unpack` / `codec_.UnpackAffine`, and the *decode* is untouched by all of this — the stored
bytes and their meaning are identical. The brief lists them as Phase 2 targets; they are only in
scope if a kernel ever wants the sign as a separate tensor (a mask or a sketch), which nothing
currently does.

**The WHT prefill does not need to change either.** `Pi` is orthogonal, so unbiasedness in the
rotated basis is unbiasedness on `<q, k>`.

### The high-risk item, quantified

The shipping AIV encoder quantizes on the **Lloyd-Max** grid with an **RMS** scale. Two
consequences, both measured here (D=256, 4096 rotated Gaussian vectors):

- The grid's outermost centroid is `2.7326σ`, so an RMS scale leaves **0.60% of coordinates
  outside the grid**. Those clip, and stochastic rounding cannot debias a clipped coordinate:
  residual `max |E[x̂] − x| = 1.95`, mean `1.8e-03`. Widening to `absmax/2.7326` drops the
  residual to `2.4e-07` — fp32 round-off — but *widening a Lloyd-Max grid wastes the resolution
  it was optimised for*, so that is a fidelity/bias trade, not a free fix.
- Stochastic rounding on a **non-uniform** codebook needs both bracketing centroids to form the
  up-step probability `(u − c_lo)/(c_hi − c_lo)`, i.e. a **second data-dependent Gather** on top
  of the one `ComputeCentroids` already issues. The tree has never measured a data-dependent
  Gather's throughput (every other one uses a host-built constant offset table), and that
  unmeasured cost is already flagged against the Lloyd-Max decode itself.

On the **uniform mid-rise** grid neither problem exists: the probability is just `t − floor(t)`,
no Gather, and an `absmax/7.5` scale clips nothing by construction. That is a real argument for
prototyping on the Cube path first even though it is test-only.

### Verification ladder

1. `-Werror` host tier — already green:
   `cmake -S csrc/tests -B <build> -DVLLM_ASCEND_TESTS_HOST_ONLY=ON -DVLLM_ASCEND_TESTS_WERROR=ON`,
   target `test_host_turboquant_fidelity`. **22/22 pass** including the new
   `TurboQuantQjlCodec` case.
2. Python UT in the pinned vendor image — already green: **224 passed**
   (`tests/ut/attention/test_qjl_codec.py` plus all five TurboQuant UT files, no regression).
3. The standalone ops build in the vendor image — required *only* if step 5 of the table above is
   taken, because the three `-Werror` tiers never compile `turboquant_torch_adpt.h`.
4. Camodel: `test_sim_950pr_turboquant_fused` cases (a)–(e), `kv4fp8`, S ≤ 256, one test per
   process via `build/nz_runs.sh`. Case (e) is the only one that executes the kernel cache
   **writer**, so it is the gate for any encoder change. Do not run `..._multimode`.
5. Fidelity on `O` — the measurement that actually decides this. Not available on this host: there
   is no NPU in either WSL distro, and no `datasets/` or `~/models` for a LongBench run.

## Reproduce

```bash
python -m pytest tests/ut/attention/test_qjl_codec.py -q
```

```bash
python -m ruff check vllm_ascend/attention/qjl_reference.py tests/ut/attention/test_qjl_codec.py
```

The prior ablation, unchanged by this phase:

```bash
python tests/research/qjl_vs_int4_ablation.py --seeds 8 --kappa 1 8 64
```

## Recommendation

Do **not** build a sign-magnitude unpack path on fidelity grounds — it decodes the same grid, and
`UnpackAffine`'s native `int4b_t` `Cast` + `DeInterleave` is already cheaper than the shift/mask
sequence (two vector instructions for both planes, against three per plane plus the widen).
`qjl_codec.h` is worth keeping regardless: it is the decoder identity, proven, and it costs
nothing at runtime.

The open question Phase 2 should answer is the one this phase opened rather than the one the brief
assumed: **does removing the per-key bias improve attention output, and does it survive the √2
variance cost?** Prototype it on the uniform mid-rise grid (`turboquant_codec_mx.h`) where the
encoder change is one `Adds` and one random draw, measure on `O` and not on SNR, and only then
decide whether it is worth the Gather and the clipping trade on the shipping Lloyd-Max path.

## Scope and limits

- Nothing here ran on an NPU or on the camodel. Every number is host arithmetic.
- The bias tables are D=128, 512 keys, synthetic (Gaussian and Student-t df=3 with 16× outlier
  channels). No real-activation pass: the estimator claim is analytic and the closed-form check
  proves it exactly, but the *value* of unbiasedness on real attention output is unmeasured.
- The √2 variance cost is measured on the dot product, not on `O`. Given this tree's repeated
  finding that only about a third of a score-side change reaches the output, the effect on `O`
  could be materially smaller in either direction.
