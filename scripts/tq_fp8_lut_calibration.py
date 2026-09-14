# -----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
"""Calibrate the 16-level Lloyd-Max -> fp8_e4m3fn LUT for the 950PR KV cache,
and simulate the all-Cube FP8 attention pipeline end to end.

The pipeline under test (CLAUDE.md, section 1):

  1. AIV   rotate K/V/Q by Pi = D H D              (outlier mitigation)
  2. AIV   quantize rotated K/V to 4-bit Lloyd-Max indices, per-vector RMS scale
  3. AIV   stream 4-bit tiles from HBM, expand through a 16-entry LUT into
           fp8_e4m3fn in UB
  4. Cube  Q_fp8 x K_fp8^T and P_fp8 x V_fp8, hardware fp8 mad, fp32 accumulate

``--report`` produces the deliverables: the calibrated centroids and
thresholds, their fp8_e4m3fn images, the end-to-end cosine/SNR over four seeds,
and an ablation attributing the error between the 4-bit and the fp8 stages.

Run:
    python scripts/tq_fp8_lut_calibration.py --report
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from scipy.stats import norm

# Qwen3.5 hybrid-attention geometry: head_dim is 256, not 128, and only 6 of
# the 24 layers carry a softmax KV cache at all.
HEAD_DIM = 256
CONTEXT_LEN = 512
NUM_HEADS = 8
NUM_QUERIES = 8
SEEDS = (0, 1, 2, 3)

# The fidelity gate this pipeline is asked to clear.
COS_GATE = 0.995

# fp8_e4m3fn: 1-4-3, no infinities, max finite 448, min subnormal 2^-9.
FP8_MAX = 448.0

# Post-rotation excess kurtosis measured on real Qwen3.5-2B activations across
# the six full-attention layers: |g2| < 0.12 everywhere except V at layer 23.
TARGET_EXCESS_KURTOSIS = 0.10

# Pre-rotation outlier structure: a few channels carry the heavy tail.  KAPPA is
# not free -- it is calibrated so that the *post*-rotation excess kurtosis of the
# RMS-normalized coordinates lands inside the band measured on real Qwen3.5-2B
# activations (|g2| < 0.12 across the six full-attention layers).  KAPPA=4 gives
# -0.063; KAPPA=16 would give -0.57, which is not a Qwen-like distribution and
# would make every number below an artifact of the fixture.
OUTLIER_CHANNELS = 3
OUTLIER_KAPPA = 4.0


# ------------------------------------------------------------------ rotation --
def pi_signs(d: int) -> np.ndarray:
    """The +-1 diagonal of Pi = D H D, from the LCG both host and device use.

    Mirrors ``turboquant_pi_signs`` in vllm_ascend/attention/turboquant_rotation.py
    and ``cpu_pi_sign_vector`` in csrc/tests/reference/turbo_quant_cpu.h, so a
    cache written by one side is readable by the other.
    """
    state = (0x5F3759DF + d) & 0xFFFFFFFF
    out = np.empty(d, dtype=np.float32)
    for i in range(d):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = 1.0 if (state >> 16) & 1 else -1.0
    return out


def fwht(x: np.ndarray) -> np.ndarray:
    """Normalized fast Walsh-Hadamard transform over the last axis."""
    d = x.shape[-1]
    assert d & (d - 1) == 0, "FWHT needs a power-of-two length"
    shape = x.shape
    y = x.astype(np.float32).reshape(-1, d).copy()
    h = 1
    while h < d:
        for i in range(0, d, 2 * h):
            a = y[:, i:i + h].copy()
            b = y[:, i + h:i + 2 * h].copy()
            y[:, i:i + h] = a + b
            y[:, i + h:i + 2 * h] = a - b
        h *= 2
    return (y / math.sqrt(d)).reshape(shape)


def apply_pi(v: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Pi = D H D.  Symmetric and an involution, so this is its own inverse."""
    return fwht(v * signs) * signs


# ----------------------------------------------------------- Lloyd-Max solve --
def lloyd_max_gaussian(n_levels: int = 16, iters: int = 20000, tol: float = 1e-15):
    """Closed-form Lloyd-Max fixed point for N(0,1).

    Uses the exact truncated-Gaussian conditional mean rather than a histogram,
    so the fixed point is reached to fp64 resolution instead of to sampling
    noise:  E[X | a<X<b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a)).

    Returns (thresholds[n-1], centroids[n], distortion).
    """
    cen = np.linspace(-3.5, 3.5, n_levels).astype(np.float64)
    for _ in range(iters):
        thr = 0.5 * (cen[:-1] + cen[1:])
        edges = np.concatenate(([-np.inf], thr, [np.inf]))
        lo, hi = edges[:-1], edges[1:]
        mass = norm.cdf(hi) - norm.cdf(lo)
        new = (norm.pdf(lo) - norm.pdf(hi)) / np.maximum(mass, 1e-300)
        if np.max(np.abs(new - cen)) < tol:
            cen = new
            break
        cen = new
    thr = 0.5 * (cen[:-1] + cen[1:])
    edges = np.concatenate(([-np.inf], thr, [np.inf]))
    mass = norm.cdf(edges[1:]) - norm.cdf(edges[:-1])
    distortion = 1.0 - float(np.sum(mass * cen ** 2))
    return thr, cen, distortion


def lloyd_max_empirical(sample: np.ndarray, n_levels: int = 16, iters: int = 500):
    """Lloyd-Max on an empirical sample, for the kurtosis-adjusted variant."""
    x = np.sort(sample.astype(np.float64).ravel())
    cen = np.quantile(x, (np.arange(n_levels) + 0.5) / n_levels)
    for _ in range(iters):
        thr = 0.5 * (cen[:-1] + cen[1:])
        idx = np.searchsorted(thr, x)
        new = cen.copy()
        for k in range(n_levels):
            m = idx == k
            if m.any():
                new[k] = x[m].mean()
        if np.max(np.abs(new - cen)) < 1e-14:
            cen = new
            break
        cen = new
    thr = 0.5 * (cen[:-1] + cen[1:])
    idx = np.searchsorted(thr, x)
    return thr, cen, float(np.mean((x - cen[idx]) ** 2))


# ------------------------------------------------------------- fp8 machinery --
def to_fp8(x: np.ndarray) -> np.ndarray:
    """Round-trip through real fp8_e4m3fn (torch), returning fp32."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return t.to(torch.float8_e4m3fn).to(torch.float32).numpy()


def build_fp8_lut(centroids: np.ndarray) -> np.ndarray:
    """Stage 3: the 16 centroids as they exist in UB after the LUT expand."""
    return to_fp8(centroids.astype(np.float32))


def lut_distortion(thresholds: np.ndarray, centroids: np.ndarray) -> float:
    """E[(X - c[q(X)])^2] on N(0,1) for an arbitrary (possibly fp8) codebook.

    Recomputed from the Gaussian measure rather than reusing the Lloyd-Max
    distortion, because once the centroids move onto the fp8 grid they are no
    longer the conditional means and the closed-form identity stops holding.
    """
    edges = np.concatenate(([-np.inf], thresholds.astype(np.float64), [np.inf]))
    lo, hi = edges[:-1], edges[1:]
    mass = norm.cdf(hi) - norm.cdf(lo)
    # E[X | bin] and E[X^2 | bin] for a truncated Gaussian.
    m1 = norm.pdf(lo) - norm.pdf(hi)
    lo_f = np.where(np.isfinite(lo), lo, 0.0)
    hi_f = np.where(np.isfinite(hi), hi, 0.0)
    m2 = mass + (lo_f * norm.pdf(lo) - hi_f * norm.pdf(hi))
    c = centroids.astype(np.float64)
    return float(np.sum(m2 - 2.0 * c * m1 + mass * c ** 2))


# ------------------------------------------------------------------- codec ----
def quantize_4bit(v: np.ndarray, thresholds: np.ndarray):
    """Rotated vectors -> (4-bit indices, per-vector RMS scale).

    The scale is ||v|| / sqrt(d), matching ``Quantize4Bit`` on the AIV, and it
    is deliberately not folded into the stored codes: dequantization stays
    linear in the scale, so the scale can be folded into the score row (K) or
    into the softmax probabilities (V) downstream.
    """
    d = v.shape[-1]
    s = np.linalg.norm(v, axis=-1, keepdims=True) / math.sqrt(d) + 1e-20
    codes = np.searchsorted(thresholds, (v / s).astype(np.float32)).astype(np.uint8)
    return codes, s.astype(np.float32)


def dequantize_lut(codes: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Stage 3: 4-bit code -> fp8 value straight out of the 16-entry LUT."""
    return lut[codes]


# ------------------------------------------------------------------ fixture ---
def make_activations(rng: np.random.Generator, heads: int, s_len: int, d: int):
    """Qwen-like pre-rotation K/V: near-Gaussian bulk plus per-channel outliers.

    The rotation is what has to remove the outliers, so the fixture must contain
    some.  A fixture drawn i.i.d. Gaussian would make Pi a no-op and measure a
    gain that is an artifact of the fixture rather than a property of the model.
    """
    df = 6.0 / TARGET_EXCESS_KURTOSIS + 4.0

    def draw():
        x = rng.standard_t(df, size=(heads, s_len, d)).astype(np.float32)
        x /= math.sqrt(df / (df - 2.0))
        chans = rng.choice(d, size=OUTLIER_CHANNELS, replace=False)
        x[..., chans] *= OUTLIER_KAPPA
        return x

    return draw(), draw()


def excess_kurtosis(x: np.ndarray) -> float:
    z = (x - x.mean()) / x.std()
    return float(np.mean(z ** 4) - 3.0)


# ---------------------------------------------------------------- attention ---
def _softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def attention_reference(q, k, v):
    """fp32 oracle: the attention the quantized path has to reproduce."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    logits = np.einsum("hqd,hkd->hqk", q, k).astype(np.float32) * scale
    return np.einsum("hqk,hkd->hqd", _softmax(logits), v).astype(np.float32)


def attention_quantized(q, k, v, thresholds, lut, signs,
                        fp8_compute=True, quantize_kv=True):
    """The pipeline: rotate, 4-bit store, LUT->fp8 expand, fp8 Cube GEMMs.

    ``fp8_compute`` and ``quantize_kv`` are the ablation switches, so the 4-bit
    storage error and the fp8 compute error can be attributed separately.
    """
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    # Stage 1. Pi is orthogonal, so Q.K^T is invariant under it; the rotation is
    # there to Gaussianize the coordinates the codec sees, nothing more.
    q_rot = apply_pi(q, signs)
    k_rot = apply_pi(k, signs)
    v_rot = apply_pi(v, signs)

    if quantize_kv:
        k_codes, k_scale = quantize_4bit(k_rot, thresholds)
        v_codes, v_scale = quantize_4bit(v_rot, thresholds)
        k_hat = dequantize_lut(k_codes, lut)   # already fp8-valued
        v_hat = dequantize_lut(v_codes, lut)
    else:
        k_hat, v_hat = k_rot, v_rot
        k_scale = np.ones((*k_rot.shape[:-1], 1), np.float32)
        v_scale = np.ones((*v_rot.shape[:-1], 1), np.float32)

    if fp8_compute:
        # Q carries real magnitudes and must be scaled into the e4m3 range
        # before the cast; the Cube consumes the scale as an fp32 multiplier.
        q_amax = np.abs(q_rot).max(axis=(-2, -1), keepdims=True) + 1e-20
        q_sf = (FP8_MAX / q_amax).astype(np.float32)
        q_in = to_fp8(q_rot * q_sf)
        k_in = to_fp8(k_hat)                   # a no-op when quantize_kv
    else:
        q_sf = np.ones((q.shape[0], 1, 1), np.float32)
        q_in, k_in = q_rot, k_hat

    # Cube GEMM 1: fp8 x fp8, fp32 accumulate. The per-vector K scale folds into
    # the score row here, which is why it never had to enter the LUT.
    logits = np.einsum("hqd,hkd->hqk",
                       q_in.astype(np.float32),
                       k_in.astype(np.float32)).astype(np.float32)
    logits = logits / q_sf * np.swapaxes(k_scale, -1, -2) * scale

    p = _softmax(logits)

    # The per-vector V scale folds into the softmax probabilities, so the row
    # the Cube multiplies is P * s_v rather than P.
    p_scaled = (p * np.swapaxes(v_scale, -1, -2)).astype(np.float32)

    if fp8_compute:
        # P is in [0,1]; scaled up so the mantissa is not thrown away against
        # e4m3's 2^-9 subnormal floor.
        p_amax = np.abs(p_scaled).max(axis=-1, keepdims=True) + 1e-20
        p_sf = (FP8_MAX / p_amax).astype(np.float32)
        p_in = to_fp8(p_scaled * p_sf)
        v_in = to_fp8(v_hat)
    else:
        p_sf = np.ones((*p_scaled.shape[:-1], 1), np.float32)
        p_in, v_in = p_scaled, v_hat

    # Cube GEMM 2, then the single un-rotation of the accumulator. Pi^2 = I is
    # what makes that the same routine as the input rotation.
    out_rot = np.einsum("hqk,hkd->hqd",
                        p_in.astype(np.float32),
                        v_in.astype(np.float32)).astype(np.float32) / p_sf
    return apply_pi(out_rot, signs)


# ------------------------------------------------------------------ metrics ---
def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def snr_db(ref, got):
    ref, got = ref.astype(np.float64), got.astype(np.float64)
    err = float(np.sum((ref - got) ** 2))
    return float(10.0 * np.log10(float(np.sum(ref ** 2)) / max(err, 1e-300)))


# ------------------------------------------------------- gate back-solve ------
def gate_snr_requirement(signs, seeds=(0, 1), lo=10.0, hi=40.0, tol=0.02):
    """Per-coordinate SNR at which attention cosine crosses COS_GATE.

    Injects orthogonal noise of a known per-coordinate SNR into K and V and
    bisects on the resulting attention cosine.  This is codec-independent: it
    says what any 4-bit scheme would have to deliver, rather than what ours
    happens to deliver.
    """
    def cos_at(target_db):
        vals = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            k, v = make_activations(rng, NUM_HEADS, CONTEXT_LEN, HEAD_DIM)
            q = rng.standard_normal((NUM_HEADS, NUM_QUERIES, HEAD_DIM)).astype(np.float32)
            ref = attention_reference(q, k, v)
            sigma = 10.0 ** (-target_db / 20.0)
            kn = k + sigma * k.std() * rng.standard_normal(k.shape).astype(np.float32)
            vn = v + sigma * v.std() * rng.standard_normal(v.shape).astype(np.float32)
            vals.append(cosine(ref, attention_reference(q, kn, vn)))
        return float(np.mean(vals))

    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if cos_at(mid) < COS_GATE:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------- report ---
def _fmt_row(cells, widths):
    return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true", help="print the full report")
    ap.add_argument("--gate-bound", action="store_true",
                    help="also back-solve the per-coordinate SNR the gate needs")
    ap.add_argument("--context-len", type=int, default=CONTEXT_LEN)
    ap.add_argument("--heads", type=int, default=NUM_HEADS)
    args = ap.parse_args()

    signs = pi_signs(HEAD_DIM)

    # -- 1. calibration ------------------------------------------------------
    thr_g, cen_g, dist_g = lloyd_max_gaussian(16)

    rng = np.random.default_rng(1234)
    k_cal, v_cal = make_activations(rng, args.heads, args.context_len, HEAD_DIM)
    rot_cal = np.concatenate([apply_pi(k_cal, signs), apply_pi(v_cal, signs)])
    s_cal = np.linalg.norm(rot_cal, axis=-1, keepdims=True) / math.sqrt(HEAD_DIM)
    norm_cal = (rot_cal / s_cal).astype(np.float32)
    thr_e, cen_e, dist_e = lloyd_max_empirical(norm_cal, 16)

    lut_g = build_fp8_lut(cen_g)
    lut_e = build_fp8_lut(cen_e)
    dist_g_fp8 = lut_distortion(thr_g, lut_g)
    dist_e_fp8 = lut_distortion(thr_e, lut_e)

    print("=" * 78)
    print("1. LLOYD-MAX CALIBRATION  (16 levels, head_dim=%d)" % HEAD_DIM)
    print("=" * 78)
    print("Pre-rotation excess kurtosis (K)      : %+.4f" % excess_kurtosis(k_cal))
    print("Post-rotation, RMS-normalized         : %+.4f" % excess_kurtosis(norm_cal))
    print("Post-rotation std                     : %.6f" % float(norm_cal.std()))
    print()
    w = [4, 22, 22, 14]
    print(_fmt_row(["i", "centroid (fp32)", "centroid (fp8_e4m3fn)", "threshold"], w))
    print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for i in range(16):
        t = "%+.10f" % thr_g[i] if i < 15 else "--"
        print(_fmt_row(["%2d" % i, "%+.10f" % cen_g[i], "%+.10f" % lut_g[i], t], w))
    print()
    print("Distortion D on N(0,1), fp32 centroids : %.10f  (%.3f dB SNR)"
          % (dist_g, -10.0 * math.log10(dist_g)))
    print("Distortion D after the fp8_e4m3fn cast : %.10f  (%.3f dB SNR)"
          % (dist_g_fp8, -10.0 * math.log10(dist_g_fp8)))
    print("Cost of the fp8 LUT quantization       : %.4f dB"
          % (10.0 * math.log10(dist_g_fp8 / dist_g)))
    print()
    print("Kurtosis-adjusted (empirical) variant  : D = %.10f -> fp8 %.10f"
          % (dist_e, dist_e_fp8))
    print("  gain over the N(0,1) table           : %.4f dB"
          % (10.0 * math.log10(dist_g_fp8 / dist_e_fp8)))

    # -- 2. end-to-end simulation -------------------------------------------
    print()
    print("=" * 78)
    print("2. END-TO-END ATTENTION  (d=%d, S=%d, heads=%d, %d seeds)"
          % (HEAD_DIM, args.context_len, args.heads, len(SEEDS)))
    print("=" * 78)

    regimes = {
        "A  4-bit LUT -> fp8, fp8 Cube GEMMs  (SHIPPING PIPELINE)":
            dict(quantize_kv=True, fp8_compute=True),
        "B  4-bit LUT, fp32 GEMMs (isolates the 4-bit stage)":
            dict(quantize_kv=True, fp8_compute=False),
        "C  no KV quant, fp8 Cube GEMMs (isolates the fp8 stage)":
            dict(quantize_kv=False, fp8_compute=True),
    }

    results = {}
    for label, kw in regimes.items():
        cs, ss = [], []
        for seed in SEEDS:
            r = np.random.default_rng(seed)
            k, v = make_activations(r, args.heads, args.context_len, HEAD_DIM)
            q = r.standard_normal((args.heads, NUM_QUERIES, HEAD_DIM)).astype(np.float32)
            ref = attention_reference(q, k, v)
            got = attention_quantized(q, k, v, thr_g, lut_g, signs, **kw)
            cs.append(cosine(ref, got))
            ss.append(snr_db(ref, got))
        results[label] = (np.array(cs), np.array(ss))

    w = [56, 10, 10, 9, 7]
    print(_fmt_row(["regime", "cos mean", "cos min", "SNR dB", "gate"], w))
    print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for label, (cs, ss) in results.items():
        verdict = "PASS" if cs.min() > COS_GATE else "FAIL"
        print(_fmt_row([label, "%.6f" % cs.mean(), "%.6f" % cs.min(),
                        "%.2f" % ss.mean(), verdict], w))
    print()
    print("Per-seed cosine, shipping pipeline: "
          + "  ".join("s%d=%.6f" % (s, c)
                      for s, c in zip(SEEDS, list(results.values())[0][0])))

    # -- 3. rate sweep: what rate would clear the gate -----------------------
    print()
    print("=" * 78)
    print("4. RATE SWEEP  (same pipeline, wider codebook; fp8 compute throughout)")
    print("=" * 78)
    w = [26, 10, 10, 9, 10, 7]
    print(_fmt_row(["storage for K/V", "cos mean", "cos min", "SNR dB",
                    "bytes/tok", "gate"], w))
    print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for bits in (4, 5, 6):
        t, c, _ = lloyd_max_gaussian(1 << bits)
        lut_b = build_fp8_lut(c)
        cs, ss = [], []
        for seed in SEEDS:
            r = np.random.default_rng(seed)
            k, v = make_activations(r, args.heads, args.context_len, HEAD_DIM)
            q = r.standard_normal((args.heads, NUM_QUERIES, HEAD_DIM)).astype(np.float32)
            ref = attention_reference(q, k, v)
            got = attention_quantized(q, k, v, t, lut_b, signs,
                                      fp8_compute=True, quantize_kv=True)
            cs.append(cosine(ref, got))
            ss.append(snr_db(ref, got))
        cs, ss = np.array(cs), np.array(ss)
        # 2 planes (K and V) * head_dim coords * bits, plus the E8M0 scale byte.
        per_tok = 2 * HEAD_DIM * bits / 8.0 + 2
        print(_fmt_row(["%d-bit Lloyd-Max LUT" % bits, "%.6f" % cs.mean(),
                        "%.6f" % cs.min(), "%.2f" % ss.mean(), "%.0f" % per_tok,
                        "PASS" if cs.min() > COS_GATE else "FAIL"], w))
    # fp8 KV storage: no 4-bit stage at all, K/V held directly as e4m3.
    cs, ss = [], []
    for seed in SEEDS:
        r = np.random.default_rng(seed)
        k, v = make_activations(r, args.heads, args.context_len, HEAD_DIM)
        q = r.standard_normal((args.heads, NUM_QUERIES, HEAD_DIM)).astype(np.float32)
        ref = attention_reference(q, k, v)
        got = attention_quantized(q, k, v, thr_g, lut_g, signs,
                                  fp8_compute=True, quantize_kv=False)
        cs.append(cosine(ref, got))
        ss.append(snr_db(ref, got))
    cs, ss = np.array(cs), np.array(ss)
    print(_fmt_row(["fp8_e4m3fn direct", "%.6f" % cs.mean(), "%.6f" % cs.min(),
                    "%.2f" % ss.mean(), "%.0f" % (2 * HEAD_DIM),
                    "PASS" if cs.min() > COS_GATE else "FAIL"], w))
    print()
    print("Baseline for the byte column: fp16 KV is %d bytes/token/head."
          % (2 * HEAD_DIM * 2))

    if args.gate_bound:
        print()
        print("=" * 78)
        print("3. WHAT THE GATE ACTUALLY REQUIRES")
        print("=" * 78)
        need = gate_snr_requirement(signs)
        print("Per-coordinate SNR needed for cos > %.3f : %.2f dB" % (COS_GATE, need))
        print("Scalar Lloyd-Max at 4 bits delivers      : %.2f dB"
              % (-10.0 * math.log10(dist_g)))
        print("Shannon bound D(R)=2^-2R at 4 bits       : %.2f dB" % (6.0206 * 4))
        print("Shortfall of the scalar optimum          : %.2f dB"
              % (need + 10.0 * math.log10(dist_g)))


if __name__ == "__main__":
    main()
