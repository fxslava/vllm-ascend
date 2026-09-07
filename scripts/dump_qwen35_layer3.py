#!/usr/bin/env python3
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#
"""Golden dumps for Qwen3.5 layer 3 (the first ``full_attention`` block).

Writes the weights, the input, per-stage activation taps and the final output of
a single-token decode step, so ``test_qwen_layer_golden`` can check a CUDA
pipeline against PyTorch stage by stage rather than only end to end.

Numerics
--------
The point of the taps is to localise a divergence, which only works if the
reference makes the same rounding decisions as the kernels. Every CUDA kernel in
``csrc/tests/kernels/cuda`` reads fp16, computes in fp32 and rounds once on the
store; cuBLAS is called with ``CUDA_R_16F`` operands and ``CUBLAS_COMPUTE_32F``.
So each stage here upcasts its fp16 input to fp32, computes, and rounds back to
fp16 - and the *rounded* tensor is what both feeds the next stage and gets
written to disk. Two consequences worth knowing:

* Everything runs on CPU. A float32 matmul on an Ampere-or-newer GPU may be
  taken through TF32, which truncates the mantissa to 10 bits and would make the
  reference *less* accurate than the fp16 kernel it is checking.
* Weights are drawn with stddev ``1/sqrt(in_features)``, which is both how they
  are really initialised and what keeps every activation near unit magnitude.
  At magnitude ~1 an fp16 ULP is ~9.8e-4, so ``atol=rtol=1e-3`` is roughly a
  one-ULP bar. Draw N(0,1) weights instead and a 2048-wide dot product lands at
  magnitude ~45, where one ULP is 0.03 and the tolerance would be measuring fp16
  storage rather than the kernels.

Layout
------
All artifacts are raw little-endian fp16, C-contiguous, no header. Linear
weights keep the torch ``[out_features, in_features]`` layout, which is exactly
the ``transpose_b=true`` case of ``CublasGemmFp16``.

A note on what pos=0 does and does not test
-------------------------------------------
At position 0 the rotary tables are cos=1, sin=0, so RoPE is the identity and
``tap_rope_q == q``. With a single token the softmax runs over one position, so
it is exactly 1.0 and the attention context is just V. Both stages are therefore
shape-and-plumbing checks at this position, not numerical ones. Pass ``--pos``
(and a larger ``--ctx-len``) to generate a set where they carry weight; the
kernels and the test do not care which set they are given.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

# --- Layer 3 configuration ---------------------------------------------------

HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 6144
NUM_ATTENTION_HEADS = 8
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = 256
PARTIAL_ROTARY_FACTOR = 0.25
ATTN_OUTPUT_GATE = True

Q_DIM = NUM_ATTENTION_HEADS * HEAD_DIM  # 2048
KV_DIM = NUM_KEY_VALUE_HEADS * HEAD_DIM  # 512
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)  # 64
RMS_NORM_EPS = 1e-6
ATTENTION_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def h(tensor: torch.Tensor) -> torch.Tensor:
    """Round to fp16, the way a kernel's store does."""
    return tensor.to(torch.float16)


def f(tensor: torch.Tensor) -> torch.Tensor:
    """Widen to fp32, the way a kernel's load does."""
    return tensor.to(torch.float32)


def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """y = x * rstd * gamma, rstd = 1/sqrt(mean(x^2) + eps). Matches LaunchRmsNormHalf."""
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rstd * gamma


def build_rope_tables(position: int, rotary_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin widened to [1, rotary_dim] as concat(v, v).

    This is the layout reference::GatherFullCosSin produces and
    LaunchApplyRotaryPosEmbHalfMode consumes: the kernel reads the angle for
    element k and for element k + rotary_dim/2 separately, and they are equal by
    construction.
    """
    half = rotary_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float64) * 2.0 / rotary_dim))
    angles = float(position) * inv_freq
    cos = torch.cat([angles.cos(), angles.cos()]).to(torch.float32).unsqueeze(0)
    sin = torch.cat([angles.sin(), angles.sin()]).to(torch.float32).unsqueeze(0)
    return cos, sin


def apply_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    """Rotate the first `rotary_dim` channels of every head; pass the rest through.

    x is [tokens, heads, head_dim] fp32. The pairing is rotate_half within the
    rotary slice - element k with element k + rotary_dim/2 - which is what the
    kernel does and what HF's partial-rotary path does.
    """
    half = rotary_dim // 2
    rotated = x.clone()

    first = x[..., :half]
    second = x[..., half:rotary_dim]

    cos_first = cos[:, None, :half]
    sin_first = sin[:, None, :half]
    cos_second = cos[:, None, half:rotary_dim]
    sin_second = sin[:, None, half:rotary_dim]

    rotated[..., :half] = first * cos_first - second * sin_first
    rotated[..., half:rotary_dim] = second * cos_second + first * sin_second
    # rotated[..., rotary_dim:] keeps the clone's pass-through values.
    return rotated


def decode_attention(
    query: torch.Tensor,
    key_context: torch.Tensor,
    value_context: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Single-token decode attention, fp32 throughout.

    query          [heads, head_dim]
    key/value      [ctx_len, kv_heads, head_dim]
    returns        [heads, head_dim]

    GQA follows vLLM and LaunchPagedAttentionDecodeV1Half: query head hq reads
    kv head hq // (heads // kv_heads).
    """
    heads, head_dim = query.shape
    ctx_len, kv_heads, _ = key_context.shape
    group = heads // kv_heads

    out = torch.zeros(heads, head_dim, dtype=torch.float32)
    for hq in range(heads):
        kv = hq // group
        scores = torch.empty(ctx_len, dtype=torch.float32)
        for pos in range(ctx_len):
            scores[pos] = torch.dot(query[hq], key_context[pos, kv]) * scale
        weights = torch.softmax(scores, dim=0)
        for pos in range(ctx_len):
            out[hq] += weights[pos] * value_context[pos, kv]
    return out


def normal(generator: torch.Generator, *shape: int, std: float) -> torch.Tensor:
    return torch.empty(*shape, dtype=torch.float32).normal_(0.0, std, generator=generator)


def write(path: Path, tensor: torch.Tensor) -> None:
    array = h(tensor).contiguous().cpu().numpy()
    assert array.dtype.name == "float16", array.dtype
    path.write_bytes(array.tobytes())
    print(f"  {path.name:<24} {str(tuple(tensor.shape)):<16} {array.nbytes:>10,} B")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "csrc/tests/data/golden_layer3",
    )
    parser.add_argument("--seed", type=int, default=0x33355157, help="0x33355157 = '3.5QW'")
    parser.add_argument("--pos", type=int, default=0, help="decode position; 0 makes RoPE the identity")
    parser.add_argument("--ctx-len", type=int, default=1, help="context length including the new token")
    parser.add_argument("--rope-theta", type=float, default=1000000.0)
    args = parser.parse_args()

    if args.ctx_len < 1:
        parser.error("--ctx-len must be at least 1")

    torch.set_grad_enabled(False)
    generator = torch.Generator().manual_seed(args.seed)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Qwen3.5 layer 3 golden dump -> {out_dir}")
    print(
        f"  hidden={HIDDEN_SIZE} ffn={INTERMEDIATE_SIZE} heads={NUM_ATTENTION_HEADS}"
        f"/{NUM_KEY_VALUE_HEADS}kv head_dim={HEAD_DIM} rotary_dim={ROTARY_DIM}"
        f" gate={ATTN_OUTPUT_GATE} pos={args.pos} ctx={args.ctx_len}\n"
    )

    # --- parameters ----------------------------------------------------------
    # fp16 on disk and fp16 as the stage input, so the reference and the device
    # start from bit-identical values.
    x_in = h(normal(generator, 1, HIDDEN_SIZE, std=1.0))
    gamma1 = h(torch.empty(HIDDEN_SIZE, dtype=torch.float32).normal_(1.0, 0.1, generator=generator))
    gamma2 = h(torch.empty(HIDDEN_SIZE, dtype=torch.float32).normal_(1.0, 0.1, generator=generator))

    hid_std = 1.0 / math.sqrt(HIDDEN_SIZE)
    ffn_std = 1.0 / math.sqrt(INTERMEDIATE_SIZE)

    w_q = h(normal(generator, Q_DIM, HIDDEN_SIZE, std=hid_std))
    w_k = h(normal(generator, KV_DIM, HIDDEN_SIZE, std=hid_std))
    w_v = h(normal(generator, KV_DIM, HIDDEN_SIZE, std=hid_std))
    w_gate_attn = h(normal(generator, Q_DIM, HIDDEN_SIZE, std=hid_std))
    w_out = h(normal(generator, HIDDEN_SIZE, Q_DIM, std=hid_std))
    w_gate = h(normal(generator, INTERMEDIATE_SIZE, HIDDEN_SIZE, std=hid_std))
    w_up = h(normal(generator, INTERMEDIATE_SIZE, HIDDEN_SIZE, std=hid_std))
    w_down = h(normal(generator, HIDDEN_SIZE, INTERMEDIATE_SIZE, std=ffn_std))

    cos_tab, sin_tab = build_rope_tables(args.pos, ROTARY_DIM, args.rope_theta)
    cos_tab, sin_tab = h(cos_tab), h(sin_tab)

    # Prior context, if any. The new token always occupies the last slot, which
    # is what a decode step does.
    past = args.ctx_len - 1
    k_past = h(normal(generator, past, NUM_KEY_VALUE_HEADS, HEAD_DIM, std=1.0)) if past else None
    v_past = h(normal(generator, past, NUM_KEY_VALUE_HEADS, HEAD_DIM, std=1.0)) if past else None

    # --- stage 1: input RMSNorm ----------------------------------------------
    norm1 = h(rms_norm(f(x_in), f(gamma1), RMS_NORM_EPS))

    # --- stage 2: Q/K/V and the attention gate -------------------------------
    q = h(f(norm1) @ f(w_q).t())
    k = h(f(norm1) @ f(w_k).t())
    v = h(f(norm1) @ f(w_v).t())
    attn_gate = h(f(norm1) @ f(w_gate_attn).t())
    qkv = torch.cat([q, k, v], dim=-1)

    # --- stage 3: partial RoPE ------------------------------------------------
    q_rope = h(
        apply_partial_rope(
            f(q).view(1, NUM_ATTENTION_HEADS, HEAD_DIM), f(cos_tab), f(sin_tab), ROTARY_DIM
        )
    ).view(1, Q_DIM)
    k_rope = h(
        apply_partial_rope(
            f(k).view(1, NUM_KEY_VALUE_HEADS, HEAD_DIM), f(cos_tab), f(sin_tab), ROTARY_DIM
        )
    ).view(1, KV_DIM)

    # --- stage 4: paged decode attention -------------------------------------
    k_new = k_rope.view(1, NUM_KEY_VALUE_HEADS, HEAD_DIM)
    v_new = v.view(1, NUM_KEY_VALUE_HEADS, HEAD_DIM)
    k_ctx = torch.cat([k_past, k_new], dim=0) if past else k_new
    v_ctx = torch.cat([v_past, v_new], dim=0) if past else v_new

    context = h(
        decode_attention(
            f(q_rope).view(NUM_ATTENTION_HEADS, HEAD_DIM), f(k_ctx), f(v_ctx), ATTENTION_SCALE
        )
    ).view(1, Q_DIM)

    # --- stage 5: output gate, out projection, residual -----------------------
    gated = h(f(context) * torch.sigmoid(f(attn_gate))) if ATTN_OUTPUT_GATE else context
    attn_out = h(f(gated) @ f(w_out).t())
    x_after_attn = h(f(x_in) + f(attn_out))

    # --- stage 6: post-attention RMSNorm --------------------------------------
    norm2 = h(rms_norm(f(x_after_attn), f(gamma2), RMS_NORM_EPS))

    # --- stage 7: SwiGLU MLP --------------------------------------------------
    mlp_gate = h(f(norm2) @ f(w_gate).t())
    mlp_up = h(f(norm2) @ f(w_up).t())
    # silu(v) = v / (1 + exp(-v)), the form SiluFloat in swiglu_kernel.cu uses.
    swiglu = h(f(mlp_gate) / (1.0 + torch.exp(-f(mlp_gate))) * f(mlp_up))
    mlp_out = h(f(swiglu) @ f(w_down).t())

    # --- stage 8: final residual ----------------------------------------------
    golden = h(f(x_after_attn) + f(mlp_out))

    # --- write ----------------------------------------------------------------
    print("inputs and weights")
    write(out_dir / "input_x.bin", x_in)
    write(out_dir / "input_norm_gamma.bin", gamma1)
    write(out_dir / "post_attn_norm_gamma.bin", gamma2)
    write(out_dir / "w_q.bin", w_q)
    write(out_dir / "w_k.bin", w_k)
    write(out_dir / "w_v.bin", w_v)
    write(out_dir / "w_gate_attn.bin", w_gate_attn)
    write(out_dir / "w_out.bin", w_out)
    write(out_dir / "w_gate.bin", w_gate)
    write(out_dir / "w_up.bin", w_up)
    write(out_dir / "w_down.bin", w_down)
    write(out_dir / "cos_tab_d64.bin", cos_tab)
    write(out_dir / "sin_tab_d64.bin", sin_tab)

    print("activation taps")
    write(out_dir / "tap_norm1.bin", norm1)
    write(out_dir / "tap_qkv.bin", qkv)
    write(out_dir / "tap_rope_q.bin", q_rope)
    write(out_dir / "tap_rope_k.bin", k_rope)
    write(out_dir / "tap_attn_out.bin", attn_out)
    write(out_dir / "tap_norm2.bin", norm2)
    write(out_dir / "tap_swiglu.bin", swiglu)

    print("golden")
    write(out_dir / "golden_output.bin", golden)

    if args.pos == 0:
        print(
            "\nnote: pos=0 makes cos=1, sin=0, so tap_rope_q == q and tap_rope_k == k."
            "\n      With ctx_len=1 the softmax is exactly 1.0 and the context is V."
            "\n      Re-run with --pos/--ctx-len for a set that exercises both."
        )


if __name__ == "__main__":
    main()
