# QJL (1-bit sign + 3-bit magnitude) vs baseline int4, at a fixed 4-bit footprint

Host-side numerical ablation, 2026-09-22, branch `feat/tq-v0.26-clean`.
Harness: [`tests/research/qjl_vs_int4_ablation.py`](qjl_vs_int4_ablation.py) (numpy;
torch/transformers only for the real-activation source). No Ascend C device kernel was
touched, and nothing was run on an NPU or on the camodel.

## Verdict

**The QJL layout is not a different quantizer from the int4 baseline — it is the same
quantizer with the code word relabelled, so there is nothing here to build.** Two of the
three QJL variants are *bit-identical* to a regime already in the tree, and the third
(the one that is genuinely different) loses by 3.8 dB.

At a fixed 4 bits per coordinate and one fp16 scale per vector, a sign bit plus a 3-bit
magnitude spans a symmetric 16-point grid. So does a signed int4 word. Which one the
hardware stores changes nothing about the reconstruction, and the measurements below
confirm that to the last decimal place:

| QJL variant | is exactly | max abs diff | coords differing |
| --- | --- | --- | --- |
| Q1 sign + mid-tread magnitude, `s = absmax/7` | B1, the briefed int4 `[-8,7]` baseline | **0.00e+00** | 0 |
| Q2 sign + mid-rise magnitude, `s = absmax/7.5` | B3, the shipped uniform int4 codec | 3.84e-01 | **4.8e-07** (exact ties only) |
| Q3 sign + 3-bit Lloyd-Max magnitude, RMS scale | B4, the shipped default (16-level Lloyd-Max) | **0.00e+00** | 0 |

Q2's only disagreement with B3 is on inputs that land exactly on a decision boundary
(`|x|/s` exactly 1.0 or 5.0), where the two encoders round the tie to different but
equally valid neighbours; 4.8e-07 of coordinates, and the metrics agree to five decimals.

Q3 is an identity for a reason worth stating once: the optimal 16-level scalar quantizer of
a symmetric density is itself symmetric, so its levels already pair up as `±c_i`. "Exact
sign + optimal 3-bit magnitude" *is* the symmetric 16-level Lloyd-Max quantizer, and it
inherits the same 20.2 dB scalar-Gaussian ceiling, not a better one.

### The shipped code word is already a sign-magnitude code

Stronger than reconstruction equality: the shipped mid-rise int4 word, read bit by bit, is
a 1-bit sign plus a 3-bit ones'-complement magnitude. With `q` in `[0,15]`,

    sign      = q >> 3                              (1 = positive)
    magnitude = (q & 7) ^ (7 if sign == 0 else 0)
    value     = ±(magnitude + 0.5) * s

is bit-identical to `(q - 7.5) * s` (max abs diff **0.00e+00**, D=256, rotated Gaussian),
and the stored sign bit equals `sign(x)` for **100.0000%** of coordinates. The "1 exact
sign bit" the QJL layout is supposed to deliver is already there, at no cost, in the codec
that ships. An Ascend C kernel that unpacked sign and magnitude separately would be
decoding the same bits it decodes today, one extra XOR later.

## What the numbers say

### Real activations — Qwen3.5-2B, post-RoPE K/V, D=256, S=512, mean of all 6 full-attention layers

`O = softmax(Q K̂ᵀ/√d + causal) V̂`, Q kept exact, K and V quantized in the rotated basis
(`Pi = D H D`). The Q/K/V recomputation is validated against each attention module's own
output first; worst case over the 6 layers is 4.7e-07 relative.

| regime | mean cos (K) | mean cos (V) | QKᵀ NMSE | max &#124;Δlogit&#124; | cos (O) | rel L2 (O) | KV SNR dB | levels/vec | sign lost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 int4 `[-8,7]`, `s=absmax/7` (briefed baseline) | 0.992367 | 0.992303 | 2.027e-03 | 1.3199 | 0.997662 | 0.0681 | 18.06 | 13.85 | 16.78% |
| Q1 QJL 1b sign + 3b mag, mid-tread | 0.992367 | 0.992303 | 2.027e-03 | 1.3199 | 0.997662 | 0.0681 | 18.06 | 13.85 | 16.78% |
| B2 int4 `[-8,7]`, `s=absmax/8` | 0.993935 | 0.993860 | 1.605e-03 | 1.1915 | 0.998076 | 0.0609 | 19.10 | 15.07 | 14.69% |
| B3 int4 mid-rise, `s=absmax/7.5` (shipped uniform) | 0.993344 | 0.993248 | 1.760e-03 | 1.1969 | 0.997341 | 0.0696 | 18.66 | 14.70 | 0% |
| Q2 QJL 1b sign + 3b mag, mid-rise | 0.993344 | 0.993248 | 1.760e-03 | 1.1969 | 0.997341 | 0.0696 | 18.66 | 14.70 | 0% |
| B4 int4 Lloyd-Max, RMS scale (shipped default) | 0.995415 | 0.995367 | 1.291e-03 | 1.0565 | 0.998288 | 0.0586 | 20.31 | 15.79 | 0% |
| Q3 QJL 1b sign + 3b Lloyd-Max mag, RMS | 0.995415 | 0.995367 | 1.291e-03 | 1.0565 | 0.998288 | 0.0586 | 20.31 | 15.79 | 0% |
| Q4 QJL 1b sign + 3b **log** magnitude | 0.981110 | 0.981133 | 6.270e-03 | 2.0247 | 0.991496 | 0.1317 | 14.22 | 14.86 | 1.86% |

`levels/vec` is the mean number of distinct reconstruction values a single vector resolves
to, out of the 16 a 4-bit code can reach. `sign lost` is the fraction of coordinates whose
reconstruction is exactly zero, i.e. whose sign bit — exact or not — does not survive into
the dot product.

### Synthetic post-Hadamard vectors, both head dimensions, 8 seeds

Rotated i.i.d. Gaussian channels (`kappa = 1`), H=8, KV=2, S=512, 64 query rows.

| regime | D | mean cos (K) | QKᵀ NMSE | max &#124;Δlogit&#124; | cos (O) | rel L2 (O) |
| --- | --- | --- | --- | --- | --- | --- |
| B1 / Q1 (identical) | 128 | 0.993252 | 1.377e-02 | 0.6700 | 0.986294 | 0.16863 |
| B3 / Q2 (identical) | 128 | 0.994110 | 1.201e-02 | 0.6106 | 0.988241 | 0.15605 |
| B4 / Q3 (identical) | 128 | 0.995404 | 9.341e-03 | 0.5750 | 0.990620 | 0.13801 |
| Q4 log magnitude | 128 | 0.981736 | 3.708e-02 | 0.9639 | 0.962892 | 0.27360 |
| B1 / Q1 (identical) | 256 | 0.992172 | 1.590e-02 | 0.6883 | 0.984593 | 0.17919 |
| B3 / Q2 (identical) | 256 | 0.993168 | 1.384e-02 | 0.6347 | 0.986545 | 0.16709 |
| B4 / Q3 (identical) | 256 | 0.995342 | 9.344e-03 | 0.5674 | 0.990800 | 0.13633 |
| Q4 log magnitude | 256 | 0.981121 | 3.777e-02 | 0.9566 | 0.963022 | 0.27161 |

Paired t over the 8 seeds, Q1 against B1: **exactly 0.00** on every metric. Q2 against B1:
+82 on K cosine, −98 on dot NMSE, +20 on O cosine — a real effect, but it is the mid-rise
grid's effect, not the sign bit's, and the repository already has it.

## The three findings that are not tautologies

1. **The briefed baseline wastes a code, and the QJL mid-tread layout wastes it in exactly
   the same place.** With `s = absmax/7`, `|v|/s ≤ 7`, so the code `-8` is unreachable:
   `[-8,7]` is a 15-level grid in practice (13.85 distinct levels realised per vector). The
   sign-magnitude layout loses the same level to a duplicated signed zero. Both then
   annihilate **16.8%** of their own sign bits, because a coordinate that quantizes to
   magnitude 0 reconstructs as exactly 0 and its sign — however exactly stored — never
   reaches the dot product. This is the one place the "1 exact sign bit" claim could have
   paid off, and it pays off only if the magnitude grid has no zero level — which is the
   mid-rise grid the tree already ships (0% sign loss, +0.6 dB, dot NMSE −13% on real
   activations). Note that the reconstruction gain does not reach the output: on real
   activations mid-rise posts a slightly *worse* mean O cosine than the briefed baseline
   (0.997341 vs 0.997662), all of it from layer 11 (0.993915 vs 0.997243), even though it
   reconstructs K and V better at every layer. On synthetic vectors it wins on O as well
   (0.986545 vs 0.984593 at D=256). Score codec work on O, never on SNR.
2. **Widening the baseline's scale to `absmax/8` beats it, and beats the shipped uniform
   grid.** B2 clips the single peak coordinate by 12.5% and buys a 12.5% finer step
   everywhere else: +1.03 dB over B1, +0.44 dB over B3, and the best O cosine of the three
   uniform regimes on real activations (0.998076). It is not a QJL result, but it is the
   cheapest change on this table — a constant in the encoder.
3. **A log-spaced magnitude grid is the only genuinely new quantizer the layout unlocks,
   and it loses by 3.8 dB.** Q4 is what sign-magnitude packing is actually good for: a
   non-uniform magnitude codebook with no lookup table. On rotated (hence near-Gaussian)
   coordinates it is far too coarse in the bulk: 14.22 dB against the baseline's 18.06 dB,
   O cosine 0.9915 against 0.9977, and max logit error 2.02 against 1.32. Under heavy
   pre-rotation outliers (`kappa = 64`, D=256) it collapses to O cosine 0.768.

## Softmax KL — `D_KL(P_golden || P_quant)`, in nats

Added 2026-09-23. KL is score-only: it depends on `K̂` alone, so it reads the survival of the
key *ranking* rather than the reconstruction magnitudes, and a codec can move it without
moving `O` at all.

Implementation checks before the numbers: an identity codec through the same rotation and
metric path gives KL **2.4e-14** (fp32 round-off), and a hand-computed two-point case matches
the harness to 12 decimals.

### Real activations — Qwen3.5-2B, per full-attention layer, mean row KL

| regime | L3 | L7 | L11 | L15 | L19 | L23 | mean | p99 | worst row | vs B1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 int4 `[-8,7]` absmax/7 (baseline) | 2.333e-02 | 2.716e-02 | 2.455e-02 | 2.832e-02 | 2.073e-02 | 1.942e-02 | 2.392e-02 | 6.18e-02 | 1.313e-01 | — |
| Q1 QJL sign + 3b mag, mid-tread | 2.333e-02 | 2.716e-02 | 2.455e-02 | 2.832e-02 | 2.073e-02 | 1.942e-02 | 2.392e-02 | 6.18e-02 | 1.313e-01 | **+0.0%** |
| B2 int4 `[-8,7]` absmax/8 | 2.165e-02 | 2.278e-02 | 1.697e-02 | 2.391e-02 | 1.519e-02 | 1.580e-02 | 1.939e-02 | 5.21e-02 | 1.102e-01 | −19.0% |
| B3 int4 mid-rise (shipped uniform) | 2.331e-02 | 2.342e-02 | 2.350e-02 | 2.583e-02 | 1.875e-02 | 1.792e-02 | 2.212e-02 | 5.93e-02 | 1.248e-01 | −7.5% |
| Q2 QJL sign + 3b mag, mid-rise | 2.331e-02 | 2.342e-02 | 2.350e-02 | 2.583e-02 | 1.875e-02 | 1.792e-02 | 2.212e-02 | 5.93e-02 | 1.248e-01 | −7.5% |
| B4 int4 Lloyd-Max RMS (shipped default) | 1.598e-02 | 1.748e-02 | 1.582e-02 | 2.078e-02 | 1.310e-02 | 1.381e-02 | 1.616e-02 | 4.15e-02 | 1.097e-01 | −32.4% |
| Q3 QJL sign + 3b Lloyd-Max mag | 1.598e-02 | 1.748e-02 | 1.582e-02 | 2.078e-02 | 1.310e-02 | 1.381e-02 | 1.616e-02 | 4.15e-02 | 1.097e-01 | −32.4% |
| Q4 QJL sign + 3b log mag | 6.793e-02 | 8.056e-02 | 7.000e-02 | 9.181e-02 | 5.936e-02 | 6.408e-02 | 7.229e-02 | 1.83e-01 | 4.406e-01 | **+202%** |

### Synthetic post-Hadamard, mean row KL, 8 seeds

| regime | D=128 κ=1 | D=128 κ=8 | D=128 κ=64 | D=256 κ=1 | D=256 κ=8 | D=256 κ=64 |
| --- | --- | --- | --- | --- | --- | --- |
| B1 / Q1 (identical) | 6.905e-03 | 1.217e-02 | 1.214e-01 | 7.966e-03 | 1.418e-02 | 2.211e-01 |
| B2 absmax/8 | 5.585e-03 | 8.839e-03 | 8.729e-02 | 6.275e-03 | 1.159e-02 | 3.244e-01 |
| B3 / Q2 (identical) | 6.028e-03 | 9.336e-03 | 1.250e-01 | 6.927e-03 | 1.249e-02 | 1.436e-01 |
| B4 / Q3 (identical) | 4.727e-03 | 1.165e-02 | 5.505e-02 | 4.692e-03 | 1.187e-02 | **1.601e+00** |
| Q4 log magnitude | 1.882e-02 | 6.392e-02 | 7.354e-02 | 1.900e-02 | 6.319e-02 | **1.466e+01** |

Paired t over 8 seeds at κ=1, against B1: Q1 **+0.00** (identical), Q2 −72.2, Q3 −126.6,
Q4 +204.3 on mean KL; on worst-row KL, −4.6, −3.7 and +17.9.

**KL adds no new ranking and no new argument for the QJL layout.** It orders the regimes
exactly as SNR and dot NMSE do, the three identity pairs are identical on it to every printed
digit, and the one regime the sign-magnitude packing actually enables (log magnitude) is 3×
worse on the mean and 3.4× worse on the worst row. The two genuine gains are again the ones
already in the tree or one constant away: Lloyd-Max −32.4%, `absmax/8` −19.0%.

One asymmetry worth noting: KL responds more strongly than `O` does. Lloyd-Max cuts KL 32%
but improves O rel-L2 only 14% (0.0681 → 0.0586) on the same tensors — the softmax absorbs
most of a score-side gain, which is the same "only about a third survives" effect recorded
earlier on this tree. Do not promote a codec on KL alone.

## Side result worth not forgetting

Under heavy pre-rotation outlier structure (`kappa = 64`, 3 channels of 256 scaled 64×) the
shipped **Lloyd-Max + RMS-scale** default inverts and becomes the worst regime on the
output: O cosine **0.9215** against the mid-rise absmax grid's **0.9803**, rel L2 0.383 vs
0.185, max logit error 119 vs 59, and mean softmax KL **1.601 vs 0.144** — an order of
magnitude. The rotation does not fully Gaussianise a 64× channel,
and an RMS scale then clips the surviving tail that an absmax scale would have covered. The
real Qwen3.5-2B activations are nowhere near that regime (Lloyd-Max wins every layer there),
so this is a caveat about models with worse outlier structure, not a reason to change the
default.

## Scope and limits

- **No real D=128 source.** The only local checkpoint with a softmax KV cache is
  Qwen3.5-2B, whose head dim is 256. D=128 is measured on synthetic post-Hadamard vectors
  only. The post-rotation coordinate distribution is near-Gaussian at every layer of the
  real model (established previously: |excess kurtosis| < 0.12), which is what makes the
  synthetic source a fair proxy for a codec comparison, but a D=128 checkpoint would be
  better evidence.
- **Nothing was measured on device.** No kernel, no camodel, no silicon; this is a
  fidelity study only. Since every QJL variant that is worth anything turns out to be a
  regime already implemented, there is no new kernel cost to price.
- The 4-bit fidelity ceiling established earlier on this tree still applies: cos > 0.995 on
  attention output is unreachable at 4 bits/coordinate by any scalar codec, and the scalar
  optimum (Lloyd-Max, 20.22 dB) is 2.9 dB short of the gate. Nothing in this ablation moves
  that bound — a sign-magnitude relabelling cannot.

## Reproduce

```bash
python tests/research/qjl_vs_int4_ablation.py --seeds 8 --kappa 1 8 64
python tests/research/qjl_vs_int4_ablation.py --source qwen35 --seq-len 512 \
    --model-path C:\Qwen3.5-2B
```

The rotation and the two shipped codecs are imported from
`scripts/tq_kv_quant_reference.py`, which is pinned bit-exact against
`csrc/tests/reference/turbo_quant_cpu.h`; none of it is re-implemented in the harness.
