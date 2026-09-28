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
"""QJL-style 1-bit sign + 3-bit magnitude vs baseline int4, at a fixed 4 bits.

Host-side only: numpy for the codecs and the metrics, torch/transformers only
for the optional real-activation source.  No CANN, no NPU and no device kernel
is touched or needed -- this answers "is the sign-magnitude layout worth a
kernel" *before* anything is written in Ascend C.

What is compared, every codec at exactly 4 bits per coordinate with one fp16
scale per vector (the scale is rounded through float16 in every regime, so the
memory footprint is identical by construction):

  baseline int4    a signed 4-bit integer code in [-8, 7], reconstruction q * s
  QJL candidate    1 exact sign bit + a 3-bit magnitude code in [0, 7]

Both are measured on *post-Hadamard* vectors: the study quantizes ``Pi x`` with
``Pi = D H D`` exactly as the shipped codec does (see
``vllm_ascend/attention/turboquant_v1.py``), so the distribution the codebooks
see is the rotated one.

Metrics, per the brief:

  * mean per-vector cosine similarity of the reconstruction against fp32,
  * NMSE of the dot products ``Q K^T``,
  * max absolute error on the attention logits ``Q K^T / sqrt(d)``,

plus the attention output ``O = softmax(logits) V`` (cosine and relative L2).
``O`` is carried because an earlier study on this tree established that
per-vector and logit-level error do *not* predict output error -- only about a
third of a reconstruction-SNR gain survives the softmax.

Usage:

    python tests/research/qjl_vs_int4_ablation.py                   # synthetic
    python tests/research/qjl_vs_int4_ablation.py --kappa 1 8 64     # outliers
    python tests/research/qjl_vs_int4_ablation.py --source qwen35 \
        --model-path C:\\Qwen3.5-2B --seq-len 512                    # real K/V
"""

import argparse
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

# The rotation and the two shipped codecs come from the repository's own host
# reference, which is pinned bit-exact against
# csrc/tests/reference/turbo_quant_cpu.h.  None of it is re-implemented here.
from tq_kv_quant_reference import (  # noqa: E402
    LLOYD_MAX_4BIT_CENTROIDS,
    LLOYD_MAX_4BIT_THRESHOLDS,
    apply_pi,
    pi_signs,
)

EPS = np.float32(1e-20)


def fp16(x):
    """Round a per-vector scale through float16, as the cache stores it."""
    return np.asarray(x, np.float16).astype(np.float32)


# ------------------------------------------------------------------ codecs --
# Every codec takes v [..., d] (already rotated) and returns the fp32
# reconstruction plus the integer codes, so code utilisation can be audited.


def absmax(v):
    return np.abs(v).max(-1, keepdims=True) + EPS


def rms(v):
    return np.linalg.norm(v, axis=-1, keepdims=True) / math.sqrt(v.shape[-1]) + EPS


def c_int4_midtread(v):
    """Baseline as briefed: int4 in [-8, 7], s = absmax / 7, recon = q * s."""
    s = fp16(absmax(v) / 7.0)
    q = np.clip(np.rint(v / s), -8.0, 7.0)
    return (q * s).astype(np.float32), q.astype(np.int8)


def c_int4_midtread_d8(v):
    """Same code space, s = absmax / 8: the only scale that can emit -8."""
    s = fp16(absmax(v) / 8.0)
    q = np.clip(np.rint(v / s), -8.0, 7.0)
    return (q * s).astype(np.float32), q.astype(np.int8)


def c_int4_midrise(v):
    """The shipped uniform grid: 16 levels +-{0.5..7.5} * s, s = absmax / 7.5."""
    s = fp16(absmax(v) / 7.5)
    q = np.clip(np.rint(v / s + 7.5), 0.0, 15.0)
    return ((q - 7.5) * s).astype(np.float32), q.astype(np.int8)


def c_int4_lloydmax(v):
    """The shipped default codec: symmetric 16-level Lloyd-Max, RMS scale."""
    s = fp16(rms(v))
    q = np.searchsorted(LLOYD_MAX_4BIT_THRESHOLDS, v / s)
    return (s * LLOYD_MAX_4BIT_CENTROIDS[q]).astype(np.float32), q.astype(np.int8)


def _sign(v):
    """The exact 1-bit sign.  sign(0) is stored as +, as a sign bit must be."""
    return np.where(v < 0.0, np.float32(-1.0), np.float32(1.0))


def c_qjl_midtread(v):
    """QJL candidate: sign bit + magnitude rint(|v|/s) in [0,7], s = absmax/7."""
    s = fp16(absmax(v) / 7.0)
    m = np.clip(np.rint(np.abs(v) / s), 0.0, 7.0)
    code = (m + np.where(v < 0.0, 8.0, 0.0)).astype(np.int8)
    return (_sign(v) * m * s).astype(np.float32), code


def c_qjl_midrise(v):
    """QJL candidate, sign-preserving: recon = sign * (m + 0.5) * s.

    The magnitude grid has no exact zero, so the exact sign bit is never
    annihilated by its own magnitude -- this is the layout that actually
    delivers what "1-bit exact sign" promises.
    """
    s = fp16(absmax(v) / 7.5)
    m = np.clip(np.rint(np.abs(v) / s - 0.5), 0.0, 7.0)
    code = (m + np.where(v < 0.0, 8.0, 0.0)).astype(np.int8)
    return (_sign(v) * (m + 0.5) * s).astype(np.float32), code


# The optimal 8-level quantizer of a half-normal magnitude is the positive half
# of the optimal symmetric 16-level quantizer of the normal it came from, so the
# table is not re-derived: it is sliced out of the shipped one.
HALFNORMAL_CENTROIDS = LLOYD_MAX_4BIT_CENTROIDS[8:].copy()
HALFNORMAL_THRESHOLDS = LLOYD_MAX_4BIT_THRESHOLDS[8:].copy()


def c_qjl_lloydmax(v):
    """QJL candidate with the optimal magnitude codebook, RMS scale."""
    s = fp16(rms(v))
    m = np.searchsorted(HALFNORMAL_THRESHOLDS, np.abs(v) / s)
    code = (m + np.where(v < 0.0, 8, 0)).astype(np.int8)
    return (_sign(v) * s * HALFNORMAL_CENTROIDS[m]).astype(np.float32), code


QJL_LOG_LEVELS = np.array(
    [0.0, 1.0 / 64, 1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0],
    dtype=np.float32,
)


def c_qjl_log(v):
    """QJL candidate with an exponential (fp-like) magnitude grid, absmax scale.

    Included because a sign-magnitude layout makes a non-uniform magnitude
    codebook nearly free, and a log grid is the one such codebook that costs no
    table: it is the alternative the packing is supposed to unlock.
    """
    s = fp16(absmax(v))
    u = np.abs(v) / s
    bnd = 0.5 * (QJL_LOG_LEVELS[:-1] + QJL_LOG_LEVELS[1:])
    m = np.searchsorted(bnd, u)
    code = (m + np.where(v < 0.0, 8, 0)).astype(np.int8)
    return (_sign(v) * s * QJL_LOG_LEVELS[m]).astype(np.float32), code


CODECS = OrderedDict(
    [
        ("B1 int4 [-8,7] absmax/7 (briefed baseline)", c_int4_midtread),
        ("B2 int4 [-8,7] absmax/8", c_int4_midtread_d8),
        ("B3 int4 mid-rise absmax/7.5 (shipped uniform)", c_int4_midrise),
        ("B4 int4 Lloyd-Max, RMS scale (shipped default)", c_int4_lloydmax),
        ("Q1 QJL 1b sign + 3b mag, mid-tread", c_qjl_midtread),
        ("Q2 QJL 1b sign + 3b mag, mid-rise", c_qjl_midrise),
        ("Q3 QJL 1b sign + 3b Lloyd-Max mag, RMS", c_qjl_lloydmax),
        ("Q4 QJL 1b sign + 3b log mag", c_qjl_log),
    ]
)

BASELINE = "B1 int4 [-8,7] absmax/7 (briefed baseline)"


def rotated_codec(fn, signs):
    """Quantize in the rotated basis and come back: Pi is an involution."""

    def run(v):
        shape = v.shape
        flat = v.reshape(-1, shape[-1])
        rot = apply_pi(flat, signs)
        hat, code = fn(rot)
        # The un-rotated reconstruction is what attention consumes; the rotated
        # one is the only basis in which the code grid is still visible, so the
        # code audit has to read that, not the output.
        return apply_pi(hat, signs).reshape(shape), hat, rot, code

    return run


# ----------------------------------------------------------------- metrics --
def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def log_softmax(x):
    """Stable log-softmax in float64 over the last axis."""
    x = np.asarray(x, np.float64)
    m = x.max(-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def softmax_kl(lg_ref, lg_hat, valid=None):
    """Per-row ``D_KL(P_golden || P_quant)`` in nats over the softmax rows.

    KL is a score-only metric: it depends on Khat alone, so a codec can move it
    without touching the attention output at all -- which is exactly what an
    earlier study on this tree measured (a ~27% KL cut that never reached O).
    It is reported here because it is the metric that says whether the *ranking*
    of keys, not their magnitudes, survives quantization.
    """
    lp_ref = log_softmax(lg_ref)
    lp_hat = log_softmax(lg_hat)
    p_ref = np.exp(lp_ref)
    term = p_ref * (lp_ref - lp_hat)
    if valid is not None:
        term = np.where(valid, term, 0.0)
    return term.sum(-1)


def vec_cos(a, b):
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30
    return num / den


def code_stats(code, hat, ref, sample=512):
    """Code utilisation, distinct levels per vector, and lost sign bits.

    ``levels`` is the mean number of distinct reconstruction values a single
    vector actually resolves to.  A 4-bit code can reach 16; a grid that wastes
    a code (an unreachable ``-8``, or a duplicated signed zero) reaches 15, and
    that is the whole difference between the two briefed layouts.
    """
    flat = hat.reshape(-1, hat.shape[-1])[:sample]
    scale = np.abs(flat).max(-1, keepdims=True) + 1e-30
    per_vec = [np.unique(np.round(row, 6)).size for row in (flat / scale)]
    dead = float(np.mean((hat == 0.0) & (ref != 0.0)))
    return {"codes_used": float(np.unique(code).size), "levels": float(np.mean(per_vec)), "sign_lost": dead}


def evaluate(q, k, v, scale, codec, signs, causal):
    """One (codec, tensor set) measurement.  q [H,R,d]; k,v [KV,S,d]."""
    run = rotated_codec(codec, signs)
    k_hat, k_rec_rot, k_rot, k_code = run(k)
    v_hat, v_rec_rot, v_rot, v_code = run(v)

    h_n, r_n, _ = q.shape
    kv_n, s_n, _ = k.shape
    grp = h_n // kv_n

    mask = None
    if causal:
        rows = np.arange(s_n - r_n, s_n)
        mask = np.arange(s_n)[None, :] <= rows[:, None]

    se_dot = 0.0
    en_dot = 0.0
    max_logit = 0.0
    o_cos = []
    o_num = 0.0
    o_den = 0.0
    kl_rows = []
    for h in range(h_n):
        kv = h // grp
        ref = (q[h] @ k[kv].T).astype(np.float32)
        hat = (q[h] @ k_hat[kv].T).astype(np.float32)
        if mask is None:
            diff, base = ref - hat, ref
        else:
            diff, base = (ref - hat)[mask], ref[mask]
        se_dot += float((diff.astype(np.float64) ** 2).sum())
        en_dot += float((base.astype(np.float64) ** 2).sum())
        max_logit = max(max_logit, float(np.abs(diff).max()) * scale)

        lg_ref = ref * scale
        lg_hat = hat * scale
        if mask is not None:
            neg = np.float32(-3.0e38)
            lg_ref = np.where(mask, lg_ref, neg)
            lg_hat = np.where(mask, lg_hat, neg)
        kl_rows.append(softmax_kl(lg_ref, lg_hat, mask))
        o_ref = softmax(lg_ref) @ v[kv]
        o_hat = softmax(lg_hat) @ v_hat[kv]
        o_cos.append(vec_cos(o_hat, o_ref))
        o_num += float(((o_hat - o_ref) ** 2).sum())
        o_den += float((o_ref**2).sum())

    out = {
        "k_cos": float(np.mean(vec_cos(k_hat, k))),
        "v_cos": float(np.mean(vec_cos(v_hat, v))),
        "dot_nmse": se_dot / (en_dot + 1e-30),
        "logit_max_abs": max_logit,
        "o_cos": float(np.mean(np.concatenate(o_cos))),
        "o_rel_l2": math.sqrt(o_num / (o_den + 1e-30)),
        "kl_mean": float(np.mean(np.concatenate(kl_rows))),
        "kl_p99": float(np.percentile(np.concatenate(kl_rows), 99.0)),
        "kl_max": float(np.max(np.concatenate(kl_rows))),
        "kv_snr_db": -10.0
        * math.log10(
            (float(((k_hat - k) ** 2).sum()) + float(((v_hat - v) ** 2).sum()))
            / (float((k**2).sum()) + float((v**2).sum()) + 1e-30)
            + 1e-30
        ),
    }
    out.update(
        code_stats(
            np.concatenate([k_code, v_code]), np.concatenate([k_rec_rot, v_rec_rot]), np.concatenate([k_rot, v_rot])
        )
    )
    return out


# ----------------------------------------------------------------- sources --
def synthetic(d, seed, s_n=512, r_n=64, h_n=8, kv_n=2, kappa=1.0):
    """Gaussian channels with `kappa`-scaled outlier channels, pre-rotation.

    The codec always sees ``Pi x``; kappa controls how much per-channel outlier
    structure exists *before* the rotation, which is the only thing the
    rotation is there to destroy.
    """
    rng = np.random.default_rng(seed)
    gain = np.ones(d, np.float32)
    gain[rng.choice(d, size=max(1, d // 85), replace=False)] = kappa
    q = rng.standard_normal((h_n, r_n, d)).astype(np.float32) * gain
    k = rng.standard_normal((kv_n, s_n, d)).astype(np.float32) * gain
    v = rng.standard_normal((kv_n, s_n, d)).astype(np.float32) * gain
    return q, k, v, 1.0 / math.sqrt(d), False


def qwen35_tensors(model_path, seq_len, layers=None, verbose=True):
    """Real post-RoPE Q/K/V from the local Qwen3.5-2B checkpoint, per layer.

    Only the 6 full_attention layers of this hybrid have a softmax KV cache.
    The recomputation is validated against each module's own output before any
    number derived from it is reported.
    """
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.eval()

    text = (
        "The Ascend 950PR pairs a Cube unit with two vector subcores. "
        "Quantization of the key-value cache trades reconstruction "
        "fidelity for bandwidth, and the trade is only worth making when "
        "the attention output survives it. "
    ) * 32
    ids = tok(text, return_tensors="pt").input_ids
    while ids.shape[1] < seq_len:
        text = text + text
        ids = tok(text, return_tensors="pt").input_ids
    ids = ids[:, :seq_len]

    mods = sorted(
        ((m.layer_idx, m) for m in model.modules() if type(m).__name__ == "Qwen3_5Attention"), key=lambda p: p[0]
    )
    store = {}
    handles = []

    def pre(mod, args, kwargs):
        store.setdefault(mod.layer_idx, {})["in"] = (kwargs["hidden_states"], kwargs["position_embeddings"])

    def post(mod, args, kwargs, output):
        store[mod.layer_idx]["out"] = output[0].detach()

    for _, m in mods:
        handles.append(m.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(m.register_forward_hook(post, with_kwargs=True))
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()

    out = []
    for layer_idx, mod in mods:
        if layers and layer_idx not in layers:
            continue
        hs, (cos, sin) = store[layer_idx]["in"]
        with torch.no_grad():
            shp = hs.shape[:-1]
            hshape = (shp[0], shp[1], -1, mod.head_dim)
            qq, gate = torch.chunk(mod.q_proj(hs).view(shp[0], shp[1], -1, mod.head_dim * 2), 2, dim=-1)
            q = mod.q_norm(qq.view(hshape)).transpose(1, 2)
            k = mod.k_norm(mod.k_proj(hs).view(hshape)).transpose(1, 2)
            v = mod.v_proj(hs).view(hshape).transpose(1, 2)
            q, k = M.apply_rotary_pos_emb(q, k, cos, sin)
            # Validate the recomputation against the module's real output.
            grp = q.shape[1] // k.shape[1]
            kx = k.repeat_interleave(grp, dim=1)
            vx = v.repeat_interleave(grp, dim=1)
            s_n = q.shape[2]
            causal = torch.triu(torch.full((s_n, s_n), float("-inf")), diagonal=1)
            att = torch.softmax((q @ kx.transpose(-1, -2)) * mod.scaling + causal, dim=-1) @ vx
            att = att.transpose(1, 2).reshape(shp[0], shp[1], -1)
            att = att * torch.sigmoid(gate.reshape(shp[0], shp[1], -1))
            ours = mod.o_proj(att)
            ref = store[layer_idx]["out"]
            err = float((ours - ref).abs().max() / (ref.abs().max() + 1e-30))
        if verbose:
            print("  layer {}: o_recon_max_rel = {:.2e}".format(layer_idx, err))
        assert err < 1e-4, "layer {} recomputation diverged ({:.2e})".format(layer_idx, err)
        out.append(
            (
                layer_idx,
                q[0].numpy().astype(np.float32),
                k[0].numpy().astype(np.float32),
                v[0].numpy().astype(np.float32),
                float(mod.scaling),
            )
        )
    return out


# --------------------------------------------------------------- reporting --
def paired_t(a, b):
    """Paired t statistic of (a - b) over matched trials; numpy only."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    if d.size < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0.0:
        return 0.0
    return float(d.mean() / (sd / math.sqrt(d.size)))


COLUMNS = [
    ("mean cos (K)", "k_cos", "{:.6f}"),
    ("mean cos (V)", "v_cos", "{:.6f}"),
    ("QK^T NMSE", "dot_nmse", "{:.3e}"),
    ("max |d logit|", "logit_max_abs", "{:.4f}"),
    ("softmax KL", "kl_mean", "{:.3e}"),
    ("KL p99", "kl_p99", "{:.3e}"),
    ("KL max", "kl_max", "{:.3e}"),
    ("cos (O)", "o_cos", "{:.6f}"),
    ("rel L2 (O)", "o_rel_l2", "{:.5f}"),
    ("KV SNR dB", "kv_snr_db", "{:.2f}"),
    ("levels", "levels", "{:.1f}"),
    ("sign lost", "sign_lost", "{:.4f}"),
]


def table(rows, title):
    head = "| regime | " + " | ".join(c[0] for c in COLUMNS) + " |"
    sep = "| --- " * (len(COLUMNS) + 1) + "|"
    lines = ["### " + title, "", head, sep]
    for name, m in rows.items():
        cells = [fmt.format(m[key]) for _, key, fmt in COLUMNS]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def aggregate(per_trial):
    out = OrderedDict()
    for name, trials in per_trial.items():
        agg = {}
        for key in trials[0]:
            vals = [t[key] for t in trials]
            agg[key] = float(np.mean(vals))
            agg[key + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[name] = agg
    return out


def t_block(per_trial):
    """Paired t of every regime against the briefed int4 baseline."""
    base = per_trial[BASELINE]
    keys = ["k_cos", "dot_nmse", "logit_max_abs", "kl_mean", "kl_max", "o_cos", "o_rel_l2"]
    lines = [
        "#### Paired t vs " + BASELINE + " (negative = regime lower)",
        "",
        "| regime | " + " | ".join(keys) + " |",
        "| --- " * (len(keys) + 1) + "|",
    ]
    for name, trials in per_trial.items():
        if name == BASELINE:
            continue
        cells = ["{:+.2f}".format(paired_t([t[k] for t in trials], [b[k] for b in base])) for k in keys]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def identity_report(d, seed=0, tol=1e-4):
    """Which regimes are the *same quantizer* under a relabelling of codes.

    Two codecs that reconstruct the same value for every input are the same
    quantizer however their bits are arranged, and no ablation can separate
    them.  This runs first, on rotated Gaussian input, so the reader knows
    which rows of the tables below are measurements and which are tautologies.
    """
    rng = np.random.default_rng(seed)
    v = apply_pi(rng.standard_normal((8192, d)).astype(np.float32), pi_signs(d))
    recon = OrderedDict((name, fn(v)[0]) for name, fn in CODECS.items())
    names = list(CODECS)
    lines = [
        "### Pairwise reconstruction identity (rotated N(0,1), D={})".format(d),
        "",
        "| a | b | max abs diff | coords differing |",
        "| --- | --- | --- | --- |",
    ]
    hits = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            delta = np.abs(recon[names[i]] - recon[names[j]])
            dd = float(delta.max())
            frac = float(np.mean(delta > 0.0))
            # Either bit-identical, or identical except where an input lands
            # exactly on a decision boundary and the two encoders round the tie
            # to different -- equally valid -- neighbours.
            if dd < 1e-6 or frac < tol:
                hits += 1
                lines.append("| {} | {} | **{:.2e}** | {:.2e} |".format(names[i], names[j], dd, frac))
    if not hits:
        lines.append("| (no pair agrees to within {:g}) | | | |".format(tol))
    return "\n".join(lines) + "\n"


def bit_layout_report(d=256, seed=1):
    """Read the shipped mid-rise int4 code as a sign bit plus a magnitude.

    The claim being checked is stronger than "the two codecs reconstruct the
    same values": the shipped 4-bit code word *already is* a 1-bit sign plus a
    3-bit magnitude, in ones'-complement.  With ``q`` in [0,15],

        sign     = q >> 3            (1 = positive)
        magnitude = (q & 7) ^ (7 if sign == 0 else 0)
        value    = +-(magnitude + 0.5) * s

    If that reconstruction is bit-identical to ``(q - 7.5) * s`` then the QJL
    layout is a relabelling of the shipped one, not a different quantizer, and
    no measurement can separate them.
    """
    rng = np.random.default_rng(seed)
    v = apply_pi(rng.standard_normal((8192, d)).astype(np.float32), pi_signs(d))
    s = fp16(absmax(v) / 7.5)
    hat, q = c_int4_midrise(v)
    q = q.astype(np.int32)
    sign_bit = (q >> 3) & 1
    mag = (q & 7) ^ np.where(sign_bit == 1, 0, 7)
    recon = np.where(sign_bit == 1, 1.0, -1.0) * (mag + 0.5) * s
    agree = float(np.abs(recon.astype(np.float32) - hat).max())
    sign_exact = float(np.mean((sign_bit == 1) == (v >= 0.0)))
    return (
        "### Bit layout of the shipped mid-rise int4 code (D={})\n\n"
        "| check | value |\n| --- | --- |\n"
        "| max abs diff, (q-7.5)*s vs sign-magnitude read of the same q "
        "| **{:.2e}** |\n"
        "| fraction of coords whose stored sign bit equals sign(x) "
        "| **{:.6f}** |\n"
        "| magnitude codes in use | {} of 8 |\n"
    ).format(d, agree, sign_exact, int(np.unique(mag).size))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "qwen35"], default="synthetic")
    ap.add_argument("--dims", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--kappa", type=float, nargs="+", default=[1.0])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--query-rows", type=int, default=64)
    ap.add_argument("--model-path", default=os.environ.get("QWEN35_PATH", r"C:\Qwen3.5-2B"))
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    report = []
    payload = {"source": args.source, "codecs": list(CODECS), "results": {}}

    if args.source == "synthetic":
        report.append(identity_report(args.dims[-1]))
        report.append(bit_layout_report(args.dims[-1]))
        for d in args.dims:
            signs = pi_signs(d)
            for kappa in args.kappa:
                per_trial = OrderedDict((n, []) for n in CODECS)
                for s in range(args.seeds):
                    q, k, v, scale, causal = synthetic(d, 1234 + s, s_n=args.seq_len, r_n=args.query_rows, kappa=kappa)
                    for name, fn in CODECS.items():
                        per_trial[name].append(evaluate(q, k, v, scale, fn, signs, causal))
                agg = aggregate(per_trial)
                tag = "synthetic D={}, S={}, kappa={:g}, {} seeds".format(d, args.seq_len, kappa, args.seeds)
                report.append(table(agg, tag))
                report.append(t_block(per_trial))
                payload["results"][tag] = agg
    else:
        for layer_idx, q, k, v, scale in qwen35_tensors(args.model_path, args.seq_len, args.layers):
            d = q.shape[-1]
            signs = pi_signs(d)
            rows = min(args.query_rows, q.shape[1])
            per_trial = OrderedDict()
            for name, fn in CODECS.items():
                per_trial[name] = [evaluate(q[:, -rows:, :], k, v, scale, fn, signs, True)]
            agg = aggregate(per_trial)
            tag = "Qwen3.5-2B layer {}, real post-RoPE K/V, D={}, S={}".format(layer_idx, d, args.seq_len)
            report.append(table(agg, tag))
            payload["results"][tag] = agg

    text = "\n".join(report)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print("[wrote {}]".format(args.json_out))


if __name__ == "__main__":
    main()
