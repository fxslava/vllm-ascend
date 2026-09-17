#
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
"""Per-layer quantisation loss on a real checkpoint: which layers the 4-bit cache hurts.

    python tools/tq_longbench/diagnose.py --model-path F:/AI/Qwen2.5-3B-Instruct --context 8192

When a long-context answer degrades, the first question is whether the decode is
wrong or the cache is simply lossy, and an end-to-end score cannot tell the two
apart.  This can: it runs the prefill once, then at the decode step hands **the
same query** to the dense backend and to the quantised one and reports the
cosine between their outputs, layer by layer.

Isolated, deliberately.  The model continues on the dense backend's answer, so
each layer's number is that layer's own loss rather than the accumulation of
every layer before it.  A uniform band near the 4-bit floor means the decode is
right and the cache is lossy; a single layer far below its neighbours is a
finding about the model, and one layer near zero while the rest are fine is a
bug in the backend.

Measured on Qwen2.5-3B-Instruct at context 6925 (2026-09-18), which is the
baseline a regression would show against:

    median 0.9933, max 0.9968, and two outliers -- layer 0 at 0.8953 and
    layer 20 at 0.8608.

Layer 0 is the expected one: its attention is dominated by the sink token, whose
key and value are far outside the distribution the per-vector RMS scale is
chosen for, so 4 bits cost the most there.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> None:
    """Appended, so the standard library keeps precedence over ``tools/bisect``."""
    parent = str(Path(__file__).resolve().parents[1])
    if parent not in sys.path:
        sys.path.append(parent)


_bootstrap_path()

import argparse  # noqa: E402  (after the path bootstrap above)

import torch  # noqa: E402

from tq_longbench.engine import RunnerConfig, StandaloneModelRunner  # noqa: E402
from tq_longbench.layers import ForwardBatch  # noqa: E402
from tq_longbench.tasks import needle_in_a_haystack  # noqa: E402

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

#: Below this a layer is called out. Well under the ~0.99 band 4-bit sits in, so
#: it flags an outlier rather than the floor itself.
OUTLIER_COSINE = 0.95

_BAR_WIDTH = 40


def cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    left = actual.flatten().to(torch.float64)
    right = expected.flatten().to(torch.float64)
    return float(torch.dot(left, right) / (left.norm() * right.norm()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/tq_longbench/diagnose.py",
        description="Per-layer cosine between the quantised and unquantised decode, on real weights.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--context", type=int, default=8192, help="approximate prompt tokens")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=sorted(_DTYPES))
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--attn-output-gate", action="store_true")
    return parser


@torch.inference_mode()
def layer_profile(runner: StandaloneModelRunner, token_ids: torch.Tensor) -> dict[int, float]:
    """Prefill, then compare the two backends on one decode step, layer by layer.

    The layer's own ``project`` produces the query, so what the two backends are
    asked is exactly what :meth:`~tq_longbench.layers.Attention.forward` would
    ask them -- no second copy of the projection to drift.
    """
    dense, quantised = runner.prefill_backend, runner.decode_backend
    if dense is quantised:
        raise ValueError(
            "the diagnostic needs two backends to compare. Run with --prefill-mode dense_staging and a "
            "quantised --backend, which is what this script's RunnerConfig asks for."
        )

    from tq_longbench.engine import RunMetrics

    logits = runner.prefill(token_ids, RunMetrics())
    position = token_ids.numel()
    step = torch.tensor([int(torch.argmax(logits[-1]))], dtype=torch.int64, device=runner.device)

    batch = ForwardBatch(
        positions=torch.arange(position, position + 1, dtype=torch.int64, device=runner.device),
        slots=quantised.cache.slot_mapping(position, 1),
        prefix_end=position + 1,
        is_decode=True,
    )

    profile: dict[int, float] = {}
    hidden = runner.model.embed_tokens(step)
    for layer in runner.model.layers:
        attn = layer.self_attn
        normed = layer.input_layernorm(hidden)
        query, key, value, gate = attn.project(normed, batch, runner.model.rope)
        for backend in runner.write_backends:
            backend.write_kv(attn.layer_index, key, value, batch.slots)
        reference = attn.attend(query, batch, dense, gate)
        measured = attn.attend(query, batch, quantised, gate)
        profile[attn.layer_index] = cosine(measured, reference)
        # Continue on the reference trajectory, so each layer's number is its own.
        hidden = hidden + attn.o_proj(reference.reshape(1, -1))
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return profile


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    item = needle_in_a_haystack(args.context, depth=0.0, seed=0)
    token_ids = tokenizer(item.prompt, return_tensors="pt").input_ids[0].to(torch.int64)

    runner = StandaloneModelRunner(
        RunnerConfig(
            model_path=args.model_path,
            backend="turboquant_reference",
            prefill_mode="dense_staging",
            max_seq_len=token_ids.numel() + 64,
            chunk_size=args.chunk_size,
            block_size=args.block_size,
            dtype=_DTYPES[args.dtype],
            device=args.device,
            attn_output_gate=args.attn_output_gate,
        )
    )
    print(
        f"{args.model_path}: {runner.shape.num_layers} layers, {runner.shape.num_heads} heads / "
        f"{runner.shape.num_kv_heads} kv, head_size {runner.shape.head_size}"
    )
    print(f"prefill {token_ids.numel()} tokens on {args.device}, then one decode step\n")

    profile = layer_profile(runner, token_ids.to(runner.device))
    for index in sorted(profile):
        value = profile[index]
        bar = "#" * int(max(0.0, value) * _BAR_WIDTH)
        flag = "  <-- outlier" if value < OUTLIER_COSINE else ""
        print(f"  layer {index:>3}: {value: .6f}  {bar}{flag}")

    values = sorted(profile.values())
    outliers = [index for index in sorted(profile) if profile[index] < OUTLIER_COSINE]
    print(
        f"\n  min {values[0]:.6f}   median {values[len(values) // 2]:.6f}   max {values[-1]:.6f}"
        f"\n  below {OUTLIER_COSINE}: {outliers if outliers else 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
