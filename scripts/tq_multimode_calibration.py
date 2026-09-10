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
"""Fidelity calibration for the three TurboQuantMode storage rates.

    kv3fp4   8 Lloyd-Max centroids  -> fp4 e2m1 LUT -> Cube mad_mx (FP4 x FP4)
    kv4fp8  16 Lloyd-Max centroids  -> fp8 e4m3fn LUT -> Cube mad  (FP8 x FP8)
    kv5fp8  32 Lloyd-Max centroids  -> fp8 e4m3fn LUT -> Cube mad  (FP8 x FP8)

This is the host-side twin of TurboQuantModeTraits in
csrc/attention/turboquant/turboquant_mode.h: the centroid tables printed here
are the tables that header ships, and --emit-header regenerates them.

WHAT THIS ADDS OVER scripts/tq_fp8_lut_calibration.py, which it imports its
fixture, rotation and metrics from rather than restating them:

  * the codebook is no longer fixed at 16 levels, so the rate is a parameter
    and the gate is a curve rather than a single verdict;

  * the LUT is placed on the *target* float grid with a global gain.  Casting a
    Lloyd-Max codebook straight onto e2m1 is not viable -- the smallest 3-bit
    centroid is 0.2451 and e2m1's neighbours there are 0 and 0.5, so the
    innermost pair collapses onto zero and the codebook loses two of its eight
    levels.  A gain g applied before the cast and divided back out of the
    per-vector scale costs nothing at run time (the scale is already a
    multiply) and is what makes a 3-bit fp4 codebook exist at all.  The same
    search runs for e4m3fn, where it turns out to be worth little -- that grid
    is fine enough near 1 that the untuned table is already close;

  * the operand types are per mode.  fp4x2_e2m1 x fp8_e4m3fn is NOT a supported
    Mmad tuple on arch35 (measured: build/perf/cubeprobe/err_mix_e2m1_e4m3.txt),
    so kv3fp4 has to put Q on the fp4 grid too, and that -- not the 3-bit
    storage -- is what dominates its error.  kv4fp8 and kv5fp8 keep Q in
    e4m3fn.

Usage:
    python scripts/tq_multimode_calibration.py --report
    python scripts/tq_multimode_calibration.py --emit-header
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tq_fp8_lut_calibration import (  # noqa: E402
    COS_GATE as FIDELITY_GATE,
    FP8_MAX,
    HEAD_DIM,
    NUM_HEADS,
    OUTLIER_KAPPA,
    SEEDS,
    _softmax,
    apply_pi,
    attention_reference,
    cosine,
    excess_kurtosis,
    lloyd_max_gaussian,
    lut_distortion,
    make_activations,
    pi_signs,
    snr_db,
    to_fp8,
)

CONTEXT_LEN = 512

# fp16 KV bytes per token per head at head_dim 256, the denominator of every
# memory-reduction figure below.
FP16_BYTES_PER_VECTOR = HEAD_DIM * 2


# ------------------------------------------------------------- fp4 machinery --
# e2m1: 1 sign, 2 exponent, 1 mantissa bit.  Fifteen distinct values, no
# infinities and no NaN.  torch has no float4 dtype, so the grid is written out
# and the cast is a nearest-neighbour search against it -- which is exactly what
# the hardware converter does, ties away from zero being unreachable here
# because no midpoint of the grid is ever a codebook entry after the gain
# search.
FP4_E2M1_GRID = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64
)
FP4_MAX = 6.0


def to_fp4_e2m1(x: np.ndarray) -> np.ndarray:
    """Round-trip through fp4 e2m1, returning fp32."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    idx = np.abs(a[..., None] - FP4_E2M1_GRID).argmin(axis=-1)
    return (np.sign(x) * FP4_E2M1_GRID[idx]).astype(np.float32)


# --------------------------------------------------------------- mode table --
class ModeSpec:
    """One TurboQuantMode: rate, target float grid, and packing geometry."""

    def __init__(self, name, bits, cast, grid_max, elems_per_group,
                 bytes_per_group, cube_op, query_cast):
        self.name = name
        self.bits = bits
        self.levels = 1 << bits
        self.cast = cast
        self.grid_max = grid_max
        self.elems_per_group = elems_per_group
        self.bytes_per_group = bytes_per_group
        self.cube_op = cube_op
        self.query_cast = query_cast

    def packed_bytes(self, d: int = HEAD_DIM) -> int:
        return d * self.bytes_per_group // self.elems_per_group


MODES = {
    # 8 codes / 3 bytes: three bit-planes, 96 bytes for a d=256 vector, and 96
    # is three whole 32-byte bursts.
    "kv3fp4": ModeSpec("kv3fp4", 3, to_fp4_e2m1, FP4_MAX, 8, 3,
                       "mad_mx (FP4 x FP4)", to_fp4_e2m1),
    # 2 codes / byte, the shipping layout: 128 bytes, four bursts.
    "kv4fp8": ModeSpec("kv4fp8", 4, to_fp8, FP8_MAX, 2, 1,
                       "mad (FP8 x FP8)", to_fp8),
    # 8 codes / 5 bytes: five bit-planes, 160 bytes, five bursts.
    "kv5fp8": ModeSpec("kv5fp8", 5, to_fp8, FP8_MAX, 8, 5,
                       "mad (FP8 x FP8)", to_fp8),
}


# --------------------------------------------------------------- LUT fitting --
def fit_lut(spec: ModeSpec, gains=None):
    """Place a Lloyd-Max codebook on the mode's float grid.

    Returns (thresholds, lut, gain, distortion).

    The stored codebook is  lut[q] = cast(g * c[q])  and the decoder
    reconstructs  (s / g) * lut[q], so the gain never costs an instruction --
    it is folded into the per-vector scale, which the decode path already
    multiplies by.  The distortion is therefore measured against  lut / g,
    which is the codebook the model actually sees.

    The thresholds are *not* re-derived after the cast.  They are the Lloyd-Max
    boundaries of the fp32 codebook, and the encoder on the AIV compares
    against them as immediates; moving them would mean the device and this
    script no longer agree on which bin a coordinate falls in.  The cast
    therefore perturbs the reconstruction levels only, which is why
    lut_distortion() has to recompute D from the Gaussian measure rather than
    reuse the Lloyd-Max fixed-point identity.
    """
    thr, cen, _ = lloyd_max_gaussian(spec.levels)
    if gains is None:
        # The useful range is bounded above by the grid's top: a gain that
        # pushes the outermost centroid past grid_max clips it, which costs far
        # more than the interior placement gains.
        hi = spec.grid_max / float(np.max(np.abs(cen)))
        gains = np.concatenate([
            np.linspace(0.25, min(hi, 8.0), 512),
            [1.0],
        ])

    best = None
    for g in gains:
        # `stored` is what lands in UB and what the Cube multiplies; the
        # codebook the model sees is stored/g, and that is what D is measured
        # against.  Returning `stored` rather than the effective codebook is
        # deliberate: it is the array the C++ header carries, and the single
        # place the gain is undone is the per-vector scale.
        stored = spec.cast((cen * g).astype(np.float32)).astype(np.float64)
        d = lut_distortion(thr, stored / g)
        if best is None or d < best[2]:
            best = (float(g), stored, d)
    gain, stored, dist = best
    return thr, stored.astype(np.float32), gain, dist


# ------------------------------------------------------------------- codec ----
def quantize(v: np.ndarray, thresholds: np.ndarray):
    """Rotated vectors -> (code indices, per-vector RMS scale).

    Identical to Quantize4Bit's arithmetic on the AIV and to
    tq_fp8_lut_calibration.quantize_4bit, generalized off 16 levels: the bin is
    sum_i [u > t_i], the scale is ||v||/sqrt(d), and the scale is deliberately
    left out of the stored codes so dequantization stays linear in it.
    """
    d = v.shape[-1]
    s = np.linalg.norm(v, axis=-1, keepdims=True) / math.sqrt(d) + 1e-20
    codes = np.searchsorted(thresholds, (v / s).astype(np.float32)).astype(np.uint8)
    return codes, s.astype(np.float32)


# ---------------------------------------------------------------- attention ---
def attention_mode(q, k, v, spec: ModeSpec, thresholds, lut, gain, signs,
                   quantize_kv=True, cube_compute=True):
    """The full mode pipeline: rotate, store at `bits`, expand to the Cube grid.

    A transcription of what the device does, in the same order:

      AIV   Pi = D H D on Q, K, V; encode K/V at `bits`; expand the packed
            codes back through the LUT into the operand grid in UB.
      Cube  two GEMMs, fp32 accumulate.  The per-vector K scale folds into the
            score row and the per-vector V scale into the softmax
            probabilities, so neither ever enters the LUT.
      AIV   one un-rotation of the accumulator, which is the same routine as
            the input rotation because Pi^2 = I.
    """
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    q_rot = apply_pi(q, signs)
    k_rot = apply_pi(k, signs)
    v_rot = apply_pi(v, signs)

    if quantize_kv:
        k_codes, k_scale = quantize(k_rot, thresholds)
        v_codes, v_scale = quantize(v_rot, thresholds)
        # The LUT holds cast(g*c); the gain comes back out through the scale.
        k_hat = lut[k_codes]
        v_hat = lut[v_codes]
        k_scale = k_scale / gain
        v_scale = v_scale / gain
    else:
        k_hat, v_hat = k_rot, v_rot
        k_scale = np.ones((*k_rot.shape[:-1], 1), np.float32)
        v_scale = np.ones((*v_rot.shape[:-1], 1), np.float32)

    if cube_compute:
        # Q carries real magnitudes and has to be scaled into the operand
        # grid's range before the cast; the fp32 multiplier comes back out of
        # the score row.  For kv3fp4 this is an fp4 cast, because
        # fp4 x fp8 is not a supported Mmad tuple on arch35.
        q_amax = np.abs(q_rot).max(axis=(-2, -1), keepdims=True) + 1e-20
        q_sf = (spec.grid_max / q_amax).astype(np.float32)
        q_in = spec.query_cast(q_rot * q_sf)
        k_in = k_hat  # already on the grid by construction
    else:
        q_sf = np.ones((q.shape[0], 1, 1), np.float32)
        q_in, k_in = q_rot, k_hat

    logits = np.einsum("hqd,hkd->hqk",
                       q_in.astype(np.float32),
                       k_in.astype(np.float32)).astype(np.float32)
    logits = logits / q_sf * np.swapaxes(k_scale, -1, -2) * scale

    p = _softmax(logits)
    p_scaled = (p * np.swapaxes(v_scale, -1, -2)).astype(np.float32)

    if cube_compute:
        # P is in [0,1] and would land in the subnormals without this.
        p_amax = np.abs(p_scaled).max(axis=-1, keepdims=True) + 1e-20
        p_sf = (spec.grid_max / p_amax).astype(np.float32)
        p_in = spec.query_cast(p_scaled * p_sf)
        v_in = v_hat
    else:
        p_sf = np.ones((*p_scaled.shape[:-1], 1), np.float32)
        p_in, v_in = p_scaled, v_hat

    out_rot = np.einsum("hqk,hkd->hqd",
                        p_in.astype(np.float32),
                        v_in.astype(np.float32)).astype(np.float32) / p_sf
    return apply_pi(out_rot, signs)


def sweep_mode(spec, thresholds, lut, gain, signs, heads, s_len, **kw):
    cos, snr = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        k, v = make_activations(rng, heads, s_len, HEAD_DIM)
        q = rng.standard_normal((heads, 1, HEAD_DIM)).astype(np.float32)
        ref = attention_reference(q, k, v)
        got = attention_mode(q, k, v, spec, thresholds, lut, gain, signs, **kw)
        cos.append(cosine(ref, got))
        snr.append(snr_db(ref, got))
    return np.array(cos), np.array(snr)


# ------------------------------------------------------------------ layout ----
def layout_rows():
    """Packing geometry per mode, at d=256, with the burst check."""
    rows = []
    for spec in MODES.values():
        packed = spec.packed_bytes()
        rows.append(dict(
            mode=spec.name,
            bits=spec.bits,
            levels=spec.levels,
            group="%d elems / %d B" % (spec.elems_per_group, spec.bytes_per_group),
            packed=packed,
            bursts32=packed / 32.0,
            aligned32=(packed % 32 == 0),
            aligned64=(packed % 64 == 0),
            ratio=FP16_BYTES_PER_VECTOR / packed,
        ))
    return rows


# ------------------------------------------------------------------ header ----
def emit_header(results):
    """The centroid tables as the C++ header wants them."""
    for name, r in results.items():
        spec = MODES[name]
        print("// %s: %d levels, gain %.10f, D = %.10f (%.3f dB)"
              % (name, spec.levels, r["gain"], r["dist"],
                 -10.0 * math.log10(r["dist"])))
        print("static constexpr float k%sCentroids[%d] = {"
              % (name.upper().replace("KV", "Kv"), spec.levels))
        for i in range(0, spec.levels, 4):
            chunk = r["lut"][i:i + 4]
            print("    " + ", ".join("%+.10ff" % c for c in chunk) + ",")
        print("};")
        print("static constexpr float k%sThresholds[%d] = {"
              % (name.upper().replace("KV", "Kv"), spec.levels - 1))
        for i in range(0, spec.levels - 1, 4):
            chunk = r["thr"][i:i + 4]
            print("    " + ", ".join("%+.10ff" % t for t in chunk) + ",")
        print("};")
        print()


# ------------------------------------------------------------------ report ----
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--emit-header", action="store_true")
    ap.add_argument("--context-len", type=int, default=CONTEXT_LEN)
    ap.add_argument("--heads", type=int, default=NUM_HEADS)
    args = ap.parse_args()

    signs = pi_signs(HEAD_DIM)
    results = {}
    for name, spec in MODES.items():
        thr, lut, gain, dist = fit_lut(spec)
        results[name] = dict(thr=thr, lut=lut, gain=gain, dist=dist)

    if args.emit_header:
        emit_header(results)
        return 0

    print("=" * 78)
    print("TurboQuant multi-mode calibration   d=%d  S=%d  heads=%d  seeds=%d"
          % (HEAD_DIM, args.context_len, args.heads, len(SEEDS)))
    print("fixture OUTLIER_KAPPA = %.1f   gate: cos > %.3f" % (OUTLIER_KAPPA, FIDELITY_GATE))
    print("=" * 78)

    # -- 1. packing layout ---------------------------------------------------
    print()
    print("1. PACKED LAYOUT  (one d=%d vector slot, scale plane counted separately)" % HEAD_DIM)
    print("-" * 78)
    print("  %-8s %5s %7s %-16s %8s %8s %7s %8s"
          % ("mode", "bits", "levels", "group", "bytes", "32B", "64B", "vs fp16"))
    for r in layout_rows():
        print("  %-8s %5d %7d %-16s %8d %8s %7s %7.2fx"
              % (r["mode"], r["bits"], r["levels"], r["group"], r["packed"],
                 "%.0f" % r["bursts32"] if r["aligned32"] else "NO",
                 "yes" if r["aligned64"] else "no", r["ratio"]))
    print()
    print("  32B column is the burst count when the slot is a whole number of them.")
    print("  All three are 32-byte aligned, so every DMA on the path stays a")
    print("  DataCopy; 96 and 160 are not multiples of 64, so a 64-byte burst")
    print("  covers two vector slots rather than one -- which is why the packed")
    print("  cache is indexed by token and not by vector.")

    # -- 2. codebooks --------------------------------------------------------
    print()
    print("2. CODEBOOK PLACEMENT ON THE OPERAND GRID")
    print("-" * 78)
    print("  %-8s %-22s %8s %14s %10s %10s"
          % ("mode", "grid", "gain", "D (fp32 LM)", "D (cast)", "cast cost"))
    for name, spec in MODES.items():
        r = results[name]
        _, cen, dist_fp32 = lloyd_max_gaussian(spec.levels)
        grid = "fp4 e2m1" if spec.bits == 3 else "fp8 e4m3fn"
        print("  %-8s %-22s %8.4f %14.10f %10.7f %8.3f dB"
              % (name, grid, r["gain"], dist_fp32, r["dist"],
                 10.0 * math.log10(r["dist"] / dist_fp32)))
    print()
    print("  D is E[(X - Q(X))^2] on N(0,1); 'cast cost' is what moving the")
    print("  Lloyd-Max centroids onto the operand grid costs, gain included.")

    print()
    print("  kv5fp8 codebook (the primary target), 32 levels:")
    lut5 = results["kv5fp8"]["lut"]
    thr5 = results["kv5fp8"]["thr"]
    g5 = results["kv5fp8"]["gain"]
    for i in range(0, 32, 4):
        cells = []
        for j in range(i, min(i + 4, 32)):
            cells.append("c[%2d]=%+.6f" % (j, lut5[j] / g5))
        print("    " + "  ".join(cells))
    print("    stored as fp8_e4m3fn with gain %.6f; the decoder divides it back" % g5)
    print("    out of the per-vector scale, so no instruction is spent on it.")
    del thr5

    # -- 3. end-to-end -------------------------------------------------------
    print()
    print("3. END-TO-END ATTENTION FIDELITY")
    print("-" * 78)
    print("  %-8s %-26s %10s %10s %10s %8s"
          % ("mode", "regime", "cos mean", "cos min", "SNR dB", "gate"))

    verdict = {}
    for name, spec in MODES.items():
        r = results[name]
        cos, snr = sweep_mode(spec, r["thr"], r["lut"], r["gain"], signs,
                              args.heads, args.context_len)
        verdict[name] = (cos, snr)
        print("  %-8s %-26s %10.5f %10.5f %10.2f %8s"
              % (name, spec.cube_op, cos.mean(), cos.min(), snr.mean(),
                 "PASS" if cos.min() > FIDELITY_GATE else "FAIL"))

    # The attribution the primary target needs: storage rate versus Cube grid.
    print()
    print("  attribution for kv5fp8 (same fixture, same seeds):")
    spec5 = MODES["kv5fp8"]
    r5 = results["kv5fp8"]
    cos_s, snr_s = sweep_mode(spec5, r5["thr"], r5["lut"], r5["gain"], signs,
                              args.heads, args.context_len, cube_compute=False)
    print("    5-bit storage, fp32 GEMMs (isolates the codec) : cos %.5f  %.2f dB"
          % (cos_s.mean(), snr_s.mean()))
    cos_c, snr_c = sweep_mode(spec5, r5["thr"], r5["lut"], r5["gain"], signs,
                              args.heads, args.context_len, quantize_kv=False)
    print("    no KV quant, fp8 Cube GEMMs (isolates the Cube) : cos %.5f  %.2f dB"
          % (cos_c.mean(), snr_c.mean()))
    print("    both stages                                     : cos %.5f  %.2f dB"
          % (verdict["kv5fp8"][0].mean(), verdict["kv5fp8"][1].mean()))

    # kv3fp4's error is not where it looks.  Its Cube tuple has to be
    # fp4 x fp4 -- arch35 rejects fp4x2_e2m1 x fp8_e4m3fn -- so Q and the
    # softmax row go onto the e2m1 grid too, and that dominates the 3-bit
    # storage by a wide margin.  Worth printing because the obvious reading of
    # the 0.93 above is "3 bits is too few", and it is not the whole story.
    print()
    print("  attribution for kv3fp4 (why fp4 operands, not 3-bit storage, dominate):")
    spec3 = MODES["kv3fp4"]
    r3 = results["kv3fp4"]
    cos_s3, snr_s3 = sweep_mode(spec3, r3["thr"], r3["lut"], r3["gain"], signs,
                                args.heads, args.context_len, cube_compute=False)
    cos_c3, snr_c3 = sweep_mode(spec3, r3["thr"], r3["lut"], r3["gain"], signs,
                                args.heads, args.context_len, quantize_kv=False)
    print("    3-bit storage, fp32 GEMMs (isolates the codec) : cos %.5f  %.2f dB"
          % (cos_s3.mean(), snr_s3.mean()))
    print("    no KV quant, fp4 Cube GEMMs (isolates the Cube) : cos %.5f  %.2f dB"
          % (cos_c3.mean(), snr_c3.mean()))
    print("    both stages                                     : cos %.5f  %.2f dB"
          % (verdict["kv3fp4"][0].mean(), verdict["kv3fp4"][1].mean()))

    # -- 4. verdict ----------------------------------------------------------
    print()
    print("4. VERDICT")
    print("-" * 78)
    for name in MODES:
        cos, _ = verdict[name]
        spec = MODES[name]
        margin = cos.min() - FIDELITY_GATE
        print("  %-8s cos_min %.5f  %s gate by %+.5f   %.2fx memory vs fp16"
              % (name, cos.min(), "clears" if margin > 0 else "misses",
                 margin, FP16_BYTES_PER_VECTOR / spec.packed_bytes()))
    print()
    ok = verdict["kv5fp8"][0].min() > FIDELITY_GATE
    print("  PRIMARY TARGET kv5fp8: %s" % ("PASS" if ok else "FAIL"))
    print("  Post-rotation excess kurtosis of the fixture: %+.4f (Qwen band |g2| < 0.12)"
          % excess_kurtosis(
              (lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) / math.sqrt(HEAD_DIM)))(
                  apply_pi(make_activations(np.random.default_rng(1234), args.heads,
                                            args.context_len, HEAD_DIM)[0], signs))))
    return 0 if ok else 1


if __name__ == "__main__":
    torch.set_num_threads(4)
    sys.exit(main())
