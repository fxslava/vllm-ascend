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
"""Real-weight golden dump for Qwen3.5 layer 3 (the first ``full_attention`` block).

The sibling script ``dump_qwen35_layer3.py`` dumps the same layer with random
weights at pos=0, ctx_len=1. That set is a plumbing check: at position 0 the
rotary tables are cos=1 / sin=0 so RoPE is the identity, and over a
single-element context the softmax is exactly 1.0 whatever the score is, which
makes the layer output independent of Q, K and RoPE altogether.

This script removes both blind spots and the synthetic weights with them:

* Parameters come out of the real ``F:\\AI\\Qwen3.5-2B`` checkpoint.
* The layer input is the hidden state the real model produces at layer 3 for a
  real tokenised prompt - layers 0..2 are ``linear_attention`` (gated DeltaNet)
  blocks, and running them is the only way to get the state layer 3 actually
  sees.
* The decode step runs at ``--pos`` (64 by default) over a KV cache filled from
  the same prompt, so the rotary angles are non-trivial and the softmax runs
  over ``pos + 1`` competing positions.

What the checkpoint actually stores
-----------------------------------
Three things differ from the synthetic dump and all three are load-bearing:

1. ``q_proj`` is ``[2 * num_heads * head_dim, hidden]``. The forward pass views
   its output as ``[..., num_heads, 2 * head_dim]`` and chunks the last axis, so
   the rows interleave per head: head h owns rows ``[h*512, h*512+256)`` for Q
   and ``[h*512+256, (h+1)*512)`` for the attention output gate. Splitting the
   matrix in half down the middle instead gets you head 0..3's Q and gate mixed
   together, which is the single easiest way to get a plausible-looking wrong
   answer here. ``w_q.bin`` and ``w_gate_attn.bin`` are the de-interleaved
   halves.

2. ``q_norm`` / ``k_norm`` are per-head RMSNorms over ``head_dim``, applied
   after the projection and before RoPE. The synthetic layer has no equivalent.

3. ``Qwen3_5RMSNorm`` scales by ``1 + weight``, not by ``weight`` (see the
   comment in ``modeling_qwen3_5.py`` citing huggingface/transformers#29402).
   Every gamma written here is already ``1 + weight``, so the CUDA RMSNorm
   kernel's plain ``x * rstd * gamma`` is the right operation.

Numerics
--------
Unchanged from the synthetic dump, and for the same reason: taps only localise
a divergence if the reference rounds where the kernels round. Every CUDA kernel
in ``csrc/tests/kernels/cuda`` reads fp16, computes in fp32 and rounds once on
the store; cuBLAS is called with ``CUDA_R_16F`` operands and
``CUBLAS_COMPUTE_32F``. So each stage here upcasts its fp16 input to fp32,
computes, and rounds back to fp16 - and the rounded tensor is what both feeds
the next stage and reaches disk. The reference runs on CPU so that a float32
matmul cannot be quietly taken through TF32.

The layer-3 prefix runs at the checkpoint's own bfloat16. That is what a real
deployment computes, and it only sets ``input_x`` and the KV cache; every stage
this dump asserts on is fp16 from there.

Layout
------
All artifacts are raw little-endian fp16, C-contiguous, no header, and are
tracked with Git LFS. Linear weights keep the torch ``[out_features,
in_features]`` layout, which is exactly the ``transpose_b=true`` case of
``CublasGemmFp16``. ``meta.json`` records the position, the context length and
the prompt, and is plain text on purpose.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

# --- Layer 3 configuration, from config.json ---------------------------------
# text_config.layer_types[3] == "full_attention", the first one in the stack.

LAYER_INDEX = 3

HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 6144
NUM_ATTENTION_HEADS = 8
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = 256
PARTIAL_ROTARY_FACTOR = 0.25
ROPE_THETA = 10_000_000.0
RMS_NORM_EPS = 1e-6

Q_DIM = NUM_ATTENTION_HEADS * HEAD_DIM  # 2048
KV_DIM = NUM_KEY_VALUE_HEADS * HEAD_DIM  # 512
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)  # 64
ATTENTION_SCALE = 1.0 / math.sqrt(HEAD_DIM)  # 0.0625, exact in fp16

# Long enough that --pos 128 still has a real token at every cached position.
DEFAULT_PROMPT = (
    "Kernel verification matters. The capital of France is Paris, and the capital of Japan "
    "is Tokyo. In modern computer architecture, a graphics processing unit executes many "
    "threads concurrently, which makes it well suited to the dense linear algebra at the "
    "heart of a transformer language model. The attention mechanism computes a weighted "
    "average over previously seen tokens, so a decode step must read every key and every "
    "value that the prefill stage placed into the paged cache. Numerical parity between a "
    "hand-written CUDA kernel and the reference implementation is therefore checked stage "
    "by stage, because a single mismatched rounding rule, a transposed weight, or a "
    "rotary table built from the wrong base will quietly change the answer that the model "
    "produces without ever raising an error. Measuring cosine similarity alongside the "
    "absolute error catches the direction of the drift as well as its size, which is what "
    "matters once the residual stream has grown past unit magnitude and a fixed absolute "
    "tolerance stops meaning very much at all."
)


def h(tensor: torch.Tensor) -> torch.Tensor:
    """Round to fp16, the way a kernel's store does."""
    return tensor.to(torch.float16)


def f(tensor: torch.Tensor) -> torch.Tensor:
    """Widen to fp32, the way a kernel's load does."""
    return tensor.to(torch.float32)


def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """y = x * rstd * gamma, rstd = 1/sqrt(mean(x^2) + eps). Matches LaunchRmsNormHalf.

    `gamma` is the already-shifted `1 + weight`; see the module docstring.
    """
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rstd * gamma


def build_rope_tables(positions: torch.Tensor, rotary_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin as [len(positions), rotary_dim], each row concat(v, v).

    This is the layout LaunchApplyRotaryPosEmbHalfMode consumes: the kernel
    reads the angle for element k and for element k + rotary_dim/2 separately,
    and they are equal by construction.

    The checkpoint uses interleaved MRoPE with sections [11, 11, 10]. For a
    text-only prompt the T/H/W position ids are all equal, so every section
    contributes the same angle and the interleave is the identity - this is
    plain RoPE, which is what the CUDA kernel implements.
    """
    half = rotary_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))
    angles = positions.to(torch.float64)[:, None] * inv_freq[None, :]
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(torch.float32)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(torch.float32)
    return cos, sin


def apply_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    """Rotate the first `rotary_dim` channels of every head; pass the rest through.

    x is [tokens, heads, head_dim] fp32. The pairing is rotate_half within the
    rotary slice - element k with element k + rotary_dim/2 - which is what the
    kernel does and what the partial-rotary path in modeling_qwen3_5.py does.
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
    kv head hq // (heads // kv_heads). The new token is the last context slot,
    so every earlier position is visible and no causal mask is needed.
    """
    heads, head_dim = query.shape
    ctx_len, kv_heads, _ = key_context.shape
    group = heads // kv_heads

    out = torch.zeros(heads, head_dim, dtype=torch.float32)
    for hq in range(heads):
        kv = hq // group
        scores = (key_context[:, kv, :] @ query[hq]) * scale
        weights = torch.softmax(scores, dim=0)
        out[hq] = weights @ value_context[:, kv, :]
    return out


def split_fused_q_proj(fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """De-interleave q_proj into (query, output gate), both [Q_DIM, HIDDEN].

    `fused` is [2 * Q_DIM, HIDDEN]. The forward pass does
    ``q_proj(x).view(*batch, -1, 2 * head_dim).chunk(2, dim=-1)``, so the row
    block owned by head h is [h * 2 * head_dim, (h+1) * 2 * head_dim) and splits
    down the middle into that head's query rows and its gate rows.
    """
    per_head = fused.view(NUM_ATTENTION_HEADS, 2 * HEAD_DIM, HIDDEN_SIZE)
    query = per_head[:, :HEAD_DIM, :].reshape(Q_DIM, HIDDEN_SIZE)
    gate = per_head[:, HEAD_DIM:, :].reshape(Q_DIM, HIDDEN_SIZE)
    return query.contiguous(), gate.contiguous()


def write(path: Path, tensor: torch.Tensor) -> None:
    array = h(tensor).contiguous().cpu().numpy()
    assert array.dtype.name == "float16", array.dtype
    path.write_bytes(array.tobytes())
    print(f"  {path.name:<24} {str(tuple(tensor.shape)):<18} {array.nbytes:>12,} B")


def load_layer3_prefix(model_dir: Path, prompt: str, tokens_needed: int) -> tuple[torch.Tensor, list[int]]:
    """Hidden states entering layer 3, for every position of the tokenised prompt.

    Returns ([tokens, hidden] fp32, token_ids). ``output_hidden_states`` yields
    one entry per layer boundary, so index LAYER_INDEX is the residual stream as
    layer 3 receives it - after the three gated-DeltaNet blocks below it.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if input_ids.shape[1] < tokens_needed:
        raise SystemExit(
            f"prompt tokenises to {input_ids.shape[1]} tokens but --pos needs at least "
            f"{tokens_needed}; pass a longer --prompt or a smaller --pos"
        )
    input_ids = input_ids[:, :tokens_needed]

    print(f"  prompt -> {input_ids.shape[1]} tokens, running layers 0..{LAYER_INDEX - 1}")
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), device_map="cpu")
    outputs = model.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[LAYER_INDEX][0].to(torch.float32)
    del model
    return hidden, input_ids[0].tolist()


def load_layer3_weights(model_dir: Path) -> dict[str, torch.Tensor]:
    """Layer 3's parameters, straight out of the shards, as fp32.

    Read through safe_open rather than by instantiating the model: only twelve
    tensors are wanted and the checkpoint is 4.5 GB.
    """
    import json as _json

    from safetensors import safe_open

    index_path = model_dir / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as handle:
        weight_map = _json.load(handle)["weight_map"]

    prefix = f"model.language_model.layers.{LAYER_INDEX}."
    wanted = {
        "input_layernorm": prefix + "input_layernorm.weight",
        "post_attention_layernorm": prefix + "post_attention_layernorm.weight",
        "q_proj": prefix + "self_attn.q_proj.weight",
        "k_proj": prefix + "self_attn.k_proj.weight",
        "v_proj": prefix + "self_attn.v_proj.weight",
        "o_proj": prefix + "self_attn.o_proj.weight",
        "q_norm": prefix + "self_attn.q_norm.weight",
        "k_norm": prefix + "self_attn.k_norm.weight",
        "gate_proj": prefix + "mlp.gate_proj.weight",
        "up_proj": prefix + "mlp.up_proj.weight",
        "down_proj": prefix + "mlp.down_proj.weight",
    }

    missing = [name for name in wanted.values() if name not in weight_map]
    if missing:
        raise SystemExit(f"checkpoint is missing {missing}; is layer {LAYER_INDEX} a full_attention block?")

    by_shard: dict[str, list[tuple[str, str]]] = {}
    for key, name in wanted.items():
        by_shard.setdefault(weight_map[name], []).append((key, name))

    weights: dict[str, torch.Tensor] = {}
    for shard, entries in by_shard.items():
        with safe_open(str(model_dir / shard), framework="pt") as handle:
            for key, name in entries:
                weights[key] = handle.get_tensor(name).to(torch.float32)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path(r"F:\AI\Qwen3.5-2B"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "csrc/tests/data/real_qwen_layer3",
    )
    parser.add_argument(
        "--pos",
        type=int,
        default=64,
        help="decode position of the new token; the KV cache then holds positions 0..pos-1",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    args = parser.parse_args()

    if args.pos < 1:
        parser.error("--pos must be at least 1; pos=0 makes RoPE the identity and the softmax degenerate")
    if not args.model.is_dir():
        parser.error(f"--model {args.model} is not a directory")

    torch.set_grad_enabled(False)
    context_len = args.pos + 1

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Qwen3.5 layer {LAYER_INDEX} real-weight dump -> {out_dir}")
    print(
        f"  hidden={HIDDEN_SIZE} ffn={INTERMEDIATE_SIZE} heads={NUM_ATTENTION_HEADS}"
        f"/{NUM_KEY_VALUE_HEADS}kv head_dim={HEAD_DIM} rotary_dim={ROTARY_DIM}"
        f" theta={ROPE_THETA:g} pos={args.pos} ctx={context_len}"
    )

    # --- parameters ----------------------------------------------------------
    weights = load_layer3_weights(args.model)

    # `1 +` because Qwen3_5RMSNorm scales by (1 + weight); see the docstring.
    gamma1 = h(1.0 + weights["input_layernorm"])
    gamma2 = h(1.0 + weights["post_attention_layernorm"])
    q_norm_gamma = h(1.0 + weights["q_norm"])
    k_norm_gamma = h(1.0 + weights["k_norm"])

    w_q, w_gate_attn = split_fused_q_proj(weights["q_proj"])
    w_q, w_gate_attn = h(w_q), h(w_gate_attn)
    w_k = h(weights["k_proj"])
    w_v = h(weights["v_proj"])
    w_out = h(weights["o_proj"])
    w_gate = h(weights["gate_proj"])
    w_up = h(weights["up_proj"])
    w_down = h(weights["down_proj"])

    # --- the layer's real input ----------------------------------------------
    hidden_states, token_ids = load_layer3_prefix(args.model, args.prompt, context_len)
    x_all = h(hidden_states)  # [ctx_len, hidden], fp16 from here on
    x_in = x_all[args.pos : args.pos + 1]  # the token being decoded

    positions = torch.arange(context_len, dtype=torch.int64)
    cos_all, sin_all = build_rope_tables(positions, ROTARY_DIM, ROPE_THETA)
    cos_all, sin_all = h(cos_all), h(sin_all)
    cos_tab = cos_all[args.pos : args.pos + 1]
    sin_tab = sin_all[args.pos : args.pos + 1]

    # --- the KV cache the prefill left behind --------------------------------
    # Positions 0..pos-1, run through exactly the stages the decode step runs:
    # input RMSNorm, the K/V projections, k_norm, and RoPE at each token's own
    # position. Keys are cached rotated, which is what the model does and what
    # LaunchPagedCacheScatterHalf stores.
    past = args.pos
    norm1_past = h(rms_norm(f(x_all[:past]), f(gamma1), RMS_NORM_EPS))
    k_past = h(f(norm1_past) @ f(w_k).t()).view(past, NUM_KEY_VALUE_HEADS, HEAD_DIM)
    k_past = h(rms_norm(f(k_past), f(k_norm_gamma), RMS_NORM_EPS))
    k_past = h(apply_partial_rope(f(k_past), f(cos_all[:past]), f(sin_all[:past]), ROTARY_DIM))
    v_past = h(f(norm1_past) @ f(w_v).t()).view(past, NUM_KEY_VALUE_HEADS, HEAD_DIM)

    # --- stage 1: input RMSNorm ----------------------------------------------
    norm1 = h(rms_norm(f(x_in), f(gamma1), RMS_NORM_EPS))

    # --- stage 2: Q/K/V and the attention gate -------------------------------
    q = h(f(norm1) @ f(w_q).t())
    k = h(f(norm1) @ f(w_k).t())
    v = h(f(norm1) @ f(w_v).t())
    attn_gate = h(f(norm1) @ f(w_gate_attn).t())
    qkv = torch.cat([q, k, v], dim=-1)

    # --- stage 3: per-head Q/K RMSNorm ---------------------------------------
    q_normed = h(rms_norm(f(q).view(1, NUM_ATTENTION_HEADS, HEAD_DIM), f(q_norm_gamma), RMS_NORM_EPS))
    k_normed = h(rms_norm(f(k).view(1, NUM_KEY_VALUE_HEADS, HEAD_DIM), f(k_norm_gamma), RMS_NORM_EPS))
    qk_norm = torch.cat([q_normed.view(1, Q_DIM), k_normed.view(1, KV_DIM)], dim=-1)

    # --- stage 4: partial RoPE ------------------------------------------------
    q_rope = h(apply_partial_rope(f(q_normed), f(cos_tab), f(sin_tab), ROTARY_DIM)).view(1, Q_DIM)
    k_rope = h(apply_partial_rope(f(k_normed), f(cos_tab), f(sin_tab), ROTARY_DIM)).view(1, KV_DIM)

    # --- stage 5: paged decode attention -------------------------------------
    k_ctx = torch.cat([k_past, k_rope.view(1, NUM_KEY_VALUE_HEADS, HEAD_DIM)], dim=0)
    v_ctx = torch.cat([v_past, v.view(1, NUM_KEY_VALUE_HEADS, HEAD_DIM)], dim=0)
    context = h(
        decode_attention(
            f(q_rope).view(NUM_ATTENTION_HEADS, HEAD_DIM), f(k_ctx), f(v_ctx), ATTENTION_SCALE
        )
    ).view(1, Q_DIM)

    # --- stage 6: output gate, out projection, residual -----------------------
    gated = h(f(context) * torch.sigmoid(f(attn_gate)))
    attn_out = h(f(gated) @ f(w_out).t())
    x_after_attn = h(f(x_in) + f(attn_out))

    # --- stage 7: post-attention RMSNorm --------------------------------------
    norm2 = h(rms_norm(f(x_after_attn), f(gamma2), RMS_NORM_EPS))

    # --- stage 8: SwiGLU MLP --------------------------------------------------
    mlp_gate = h(f(norm2) @ f(w_gate).t())
    mlp_up = h(f(norm2) @ f(w_up).t())
    # silu(v) = v / (1 + exp(-v)), the form SiluFloat in swiglu_kernel.cu uses.
    swiglu = h(f(mlp_gate) / (1.0 + torch.exp(-f(mlp_gate))) * f(mlp_up))
    mlp_out = h(f(swiglu) @ f(w_down).t())

    # --- stage 9: final residual ----------------------------------------------
    golden = h(f(x_after_attn) + f(mlp_out))

    # An fp16 overflow anywhere upstream reaches the output as an inf and would
    # otherwise be read as a kernel bug on the C++ side.
    for name, tensor in (
        ("attn_out", attn_out), ("swiglu", swiglu), ("golden_output", golden)
    ):
        if not torch.isfinite(f(tensor)).all():
            raise SystemExit(f"{name} contains non-finite values in fp16; this dump is unusable")

    # --- write ----------------------------------------------------------------
    print("\ninputs and weights")
    write(out_dir / "input_x.bin", x_in)
    write(out_dir / "input_norm_gamma.bin", gamma1)
    write(out_dir / "post_attn_norm_gamma.bin", gamma2)
    write(out_dir / "q_norm_gamma.bin", q_norm_gamma)
    write(out_dir / "k_norm_gamma.bin", k_norm_gamma)
    write(out_dir / "w_q.bin", w_q)
    write(out_dir / "w_gate_attn.bin", w_gate_attn)
    write(out_dir / "w_k.bin", w_k)
    write(out_dir / "w_v.bin", w_v)
    write(out_dir / "w_out.bin", w_out)
    write(out_dir / "w_gate.bin", w_gate)
    write(out_dir / "w_up.bin", w_up)
    write(out_dir / "w_down.bin", w_down)
    write(out_dir / "cos_tab_d64.bin", cos_tab)
    write(out_dir / "sin_tab_d64.bin", sin_tab)

    print("kv cache for positions 0..pos-1")
    write(out_dir / "k_cache.bin", k_past)
    write(out_dir / "v_cache.bin", v_past)

    print("activation taps")
    write(out_dir / "tap_norm1.bin", norm1)
    write(out_dir / "tap_qkv.bin", qkv)
    write(out_dir / "tap_qk_norm.bin", qk_norm)
    write(out_dir / "tap_attn_gate.bin", attn_gate)
    write(out_dir / "tap_rope_q.bin", q_rope)
    write(out_dir / "tap_rope_k.bin", k_rope)
    write(out_dir / "tap_attn_ctx.bin", context)
    write(out_dir / "tap_attn_out.bin", attn_out)
    write(out_dir / "tap_norm2.bin", norm2)
    write(out_dir / "tap_swiglu.bin", swiglu)

    print("golden")
    write(out_dir / "golden_output.bin", golden)

    meta = {
        "model": str(args.model),
        "layer_index": LAYER_INDEX,
        "layer_type": "full_attention",
        "position": args.pos,
        "context_len": context_len,
        "rope_theta": ROPE_THETA,
        "rotary_dim": ROTARY_DIM,
        "rms_norm_eps": RMS_NORM_EPS,
        "attention_scale": ATTENTION_SCALE,
        "token_ids": token_ids,
        "prompt": args.prompt,
        "note": (
            "Gammas are already 1 + weight, matching Qwen3_5RMSNorm. w_q/w_gate_attn are the "
            "de-interleaved halves of the fused q_proj. k_cache holds post-RoPE keys."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\n  meta.json                {'':<18} {(out_dir / 'meta.json').stat().st_size:>12,} B")

    # A quick sanity read on the values themselves: a dump whose softmax has
    # collapsed onto one position, or whose rotary angles are all zero, would
    # pass every parity check while testing nothing.
    scores_span = (f(k_ctx)[:, 0, :] @ f(q_rope).view(NUM_ATTENTION_HEADS, HEAD_DIM)[0]) * ATTENTION_SCALE
    weights_span = torch.softmax(scores_span, dim=0)
    print(
        f"\n  softmax over {context_len} positions: max weight {weights_span.max():.4f}, "
        f"entropy {-(weights_span * weights_span.clamp_min(1e-20).log()).sum():.3f} nats"
    )
    print(f"  rotary angle at pos {args.pos}: cos[0]={cos_tab[0, 0]:.5f} sin[0]={sin_tab[0, 0]:.5f}")


if __name__ == "__main__":
    main()
