#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""CPU reference codecs for the KV-cache quantization regimes, numpy only.

This is the host reference the fidelity studies import.  It carries no torch,
no vllm and no CANN dependency so it runs on a build machine with no NPU and
no vLLM install, which is what lets an ablation be run against real model
activations anywhere.

Provenance.  Two of the four regimes are *shipped* code and their arithmetic is
pinned elsewhere in the tree; the other two exist only as research baselines:

  * ``pi_signs`` / ``fwht`` / ``apply_pi`` mirror, element for element,
    ``cpu_pi_sign_vector`` / ``cpu_fwht`` / ``cpu_apply_pi`` in
    ``csrc/tests/reference/turbo_quant_cpu.h`` and ``turboquant_pi_signs`` /
    ``walsh_hadamard`` / ``apply_pi`` in
    ``vllm_ascend/attention/turboquant_rotation.py``.
  * ``q_absmax`` mirrors ``cpu_quantize_4bit`` + ``cpu_dequantize_4bit`` from
    the same header: symmetric mid-rise grid, ``scale = absmax / 7.5``.
  * ``q_spherical``, ``q_qjl`` and ``q_lloyd_max`` are research baselines
    (spherical L2, QJL 3b+1b, and the optimal non-uniform 4-bit scalar
    quantizer for a Gaussian).  Nothing on the device implements them; they
    exist so the shipped codec can be measured against the published
    alternatives.

The rotation is ``Pi x = D (H (D x))``.  Pi is symmetric and an involution, so
one routine is both the rotation and the un-rotation -- see the module docstring
of ``vllm_ascend/attention/turboquant_v1.py`` for why that matters on device.
"""

import math

import numpy as np

TURBOQUANT_PI_SEED = 0x5F3759DF


def fwht(x):
    """Normalised FWHT over the last axis (length must be a power of two)."""
    x = x.astype(np.float32).copy()
    d = x.shape[-1]
    assert d & (d - 1) == 0
    h = 1
    while h < d:
        x = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        a, b = x[..., 0, :].copy(), x[..., 1, :].copy()
        x[..., 0, :], x[..., 1, :] = a + b, a - b
        x = x.reshape(*x.shape[:-3], d)
        h *= 2
    return x / math.sqrt(d)


def pi_signs(d, seed=TURBOQUANT_PI_SEED):
    """The exact +-1 diagonal cpu_pi_sign_vector() derives (LCG, seed + d)."""
    s = np.uint32((seed + d) & 0xFFFFFFFF)
    out = np.empty(d, np.float32)
    for i in range(d):
        s = np.uint32((np.uint64(s) * np.uint64(1664525) + np.uint64(1013904223)) & 0xFFFFFFFF)
        out[i] = 1.0 if ((s >> np.uint32(16)) & np.uint32(1)) else -1.0
    return out


def apply_pi(x, signs):
    """``Pi x = D (H (D x))`` over the last axis.  Its own inverse."""
    return fwht(x * signs) * signs


def q_absmax(v, bits=4):
    """Symmetric mid-rise grid, scale = absmax / zp.  Matches cpu_quantize_4bit."""
    lv = (1 << bits) - 1
    zp = lv * 0.5
    amax = np.abs(v).max(-1, keepdims=True) + 1e-20
    step = amax / zp
    q = np.clip(np.rint(v / step + zp), 0, lv)
    return (q - zp) * step


def q_fixed(v, bits, step):
    """Mid-rise grid with a caller-supplied step (clipping quantizer)."""
    lv = (1 << bits) - 1
    zp = lv * 0.5
    q = np.clip(np.rint(v / step + zp), 0, lv)
    return (q - zp) * step


def mmse_gain(v, vhat):
    """Per-vector scalar alpha minimising ||v - alpha*vhat||: the compensation."""
    num = (v * vhat).sum(-1, keepdims=True)
    den = (vhat * vhat).sum(-1, keepdims=True) + 1e-30
    return num / den


def q_spherical(v, bits=4, clip_sigma=3.0, compensate=True):
    """L2 unit-norm, fixed grid sized to clip_sigma/sqrt(d), norm folded back."""
    d = v.shape[-1]
    nrm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-30
    u = v / nrm
    step = (clip_sigma / math.sqrt(d)) / ((1 << bits) - 1) * 2.0
    uh = q_fixed(u, bits, step)
    if compensate:
        uh = uh * mmse_gain(u, uh)
    return uh * nrm


def q_qjl(v, clip_sigma=3.0):
    """3-bit coarse grid + 1-bit JL sign residual, 4 bits/coord total."""
    d = v.shape[-1]
    nrm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-30
    u = v / nrm
    step = (clip_sigma / math.sqrt(d)) / 7.0 * 2.0
    base = q_fixed(u, 3, step)
    resid = u - base
    sgn = np.sign(resid)
    sgn[sgn == 0] = 1.0
    beta = np.abs(resid).mean(-1, keepdims=True)
    uh = base + beta * sgn
    uh = uh * mmse_gain(u, uh)
    return uh * nrm


LLOYD_MAX_4BIT_CENTROIDS = np.array([
    -2.732589570995171, -2.069017226531392, -1.6180463860218863, -1.2562311973471796,
    -0.9423404564869651, -0.6567591185324659, -0.38804829949029185, -0.12839502985114726,
    0.12839502985114726, 0.38804829949029185, 0.6567591185324659, 0.9423404564869651,
    1.2562311973471796, 1.6180463860218863, 2.069017226531392, 2.732589570995171,
], dtype=np.float32)

LLOYD_MAX_4BIT_THRESHOLDS = np.array([
    -2.4008033987632817, -1.8435318062766393, -1.437138791684533, -1.0992858269170722,
    -0.7995497875097155, -0.5224037090113789, -0.25822166467071955, 0.0,
    0.25822166467071955, 0.5224037090113789, 0.7995497875097155, 1.0992858269170722,
    1.437138791684533, 1.8435318062766393, 2.4008033987632817,
], dtype=np.float32)

LLOYD_MAX_4BIT_DISTORTION = 0.009501008008191723


def lloyd_max_gaussian_table(n_levels=16, iters=10000, tol=1e-15):
    """Re-derive the Lloyd-Max table for N(0,1).  Needs scipy; tests only.

    Returns (thresholds[n-1], centroids[n], distortion).  Uses the closed-form
    truncated-Gaussian moments rather than a histogram, so the fixed point is
    reached to machine precision instead of to sampling noise:

        E[X   | a<X<b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a))
        E[X^2 | a<X<b] = 1 + (a phi(a) - b phi(b)) / (Phi(b) - Phi(a))
    """
    from scipy.stats import norm

    c = np.linspace(-3.0, 3.0, n_levels)
    for _ in range(iters):
        t = np.concatenate(([-np.inf], 0.5 * (c[:-1] + c[1:]), [np.inf]))
        pa, pb = norm.pdf(t[:-1]), norm.pdf(t[1:])
        mass = norm.cdf(t[1:]) - norm.cdf(t[:-1])
        c_new = (pa - pb) / mass
        if np.max(np.abs(c_new - c)) < tol:
            c = c_new
            break
        c = c_new
    c = 0.5 * (c - c[::-1])
    t = np.concatenate(([-np.inf], 0.5 * (c[:-1] + c[1:]), [np.inf]))
    pa, pb = norm.pdf(t[:-1]), norm.pdf(t[1:])
    mass = norm.cdf(t[1:]) - norm.cdf(t[:-1])
    fin = lambda x, p: np.where(np.isfinite(x), x, 0.0) * p  # noqa: E731
    ex2 = mass + fin(t[:-1], pa) - fin(t[1:], pb)
    ex1 = pa - pb
    dist = float(np.sum(ex2 - 2.0 * c * ex1 + c ** 2 * mass))
    return t[1:-1], c, dist


def q_lloyd_max(v, centroids=None, thresholds=None):
    """Non-uniform 4-bit Lloyd-Max quantizer, per-vector Gaussian scale.

    The scale is the per-vector RMS, s = ||v||_2 / sqrt(d), which is the
    maximum-likelihood sigma of a zero-mean Gaussian and -- unlike an absmax --
    is not set by a single outlier coordinate.  Codes are the bin index of
    v / s under the Lloyd-Max boundaries; the reconstruction is s * c[q].

    The reconstruction stays *linear in s*, exactly as (q - 7.5) * step does for
    the uniform grid.  That is what lets the device keep folding the per-vector
    scale into the score row rather than broadcasting it over head_size; see
    Dequantize4Bit in csrc/attention/turboquant/op_kernel/common/turboquant_codec_950.h.
    """
    cen = LLOYD_MAX_4BIT_CENTROIDS if centroids is None else centroids
    thr = LLOYD_MAX_4BIT_THRESHOLDS if thresholds is None else thresholds
    d = v.shape[-1]
    s = np.linalg.norm(v, axis=-1, keepdims=True) / math.sqrt(d) + 1e-20
    q = np.searchsorted(thr, v / s)
    return (s * cen[q]).astype(np.float32)


def make_regimes(d, clip4=3.0, clip4r=3.0, clipq=3.0):
    """The four evaluated regimes at a matched 4 bits/coord, keyed by label.

    ``d`` is the head dimension; the Pi diagonal is derived from it, so this
    must be called with the model's real ``head_dim``.
    """
    s = pi_signs(d)
    rot = lambda v: apply_pi(v, s)  # noqa: E731 - Pi is an involution ...
    unrot = rot
    return {
        "R1 absmax L_inf 4b (baseline)": lambda v: q_absmax(v, 4),
        "R2 spherical L2 4b": lambda v: q_spherical(v, 4, clip4),
        "R3a Hadamard + absmax 4b (shipped)": lambda v: unrot(q_absmax(rot(v), 4)),
        "R3b Hadamard + spherical L2 4b": lambda v: unrot(q_spherical(rot(v), 4, clip4r)),
        "R3-LM Hadamard + Lloyd-Max 4b": lambda v: unrot(q_lloyd_max(rot(v))),
        "R4 QJL Hadamard 3b + 1b JL": lambda v: unrot(q_qjl(rot(v), clipq)),
    }


CLIP_TUNED = {
    "R2 spherical L2 4b": "clip4",
    "R3b Hadamard + spherical L2 4b": "clip4r",
    "R4 QJL Hadamard 3b + 1b JL": "clipq",
}
