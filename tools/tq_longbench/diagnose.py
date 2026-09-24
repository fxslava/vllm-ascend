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
    python tools/tq_longbench/diagnose.py --model-path /models/glm-4-9b-chat-1m --device npu:0 \
        --backend turboquant_cube --context 8192

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

The quantised side of the comparison is ``--backend`` -- the Cube decode where
there is an NPU to run it, its torch reference where there is not -- and the
dense side is the ``dense_staging`` prefill pool.  Which backend that pool runs
through is not assumed on a device: :func:`negotiate_dense_staging` probes one
synthetic layer before any weight loads, because an Ascend 950 refuses
op-plugin's fused attention outright (``EZ9903``) and stages through
``cann_dense`` instead.  Without it a 950 run reads as a device error forty
layers into a prefill rather than as a named refusal.
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

from tq_longbench.engine import (  # noqa: E402
    RunnerConfig,
    StandaloneModelRunner,
    read_checkpoint_shape,
)
from tq_longbench.families import family_for  # noqa: E402
from tq_longbench.glm4 import read_config  # noqa: E402
from tq_longbench.layers import ForwardBatch  # noqa: E402
from tq_longbench.ops import (  # noqa: E402
    DENSE_ATTENTION_APIS,
    DENSE_BACKENDS,
    LayerShape,
    reference_equivalent,
)
from tq_longbench.preflight import select_dense_backend  # noqa: E402
from tq_longbench.smoke_glm import (  # noqa: E402
    encode,
    load_tokenizer,
    quantised_backends,
    resolve_device,
)
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


def default_device() -> str:
    """The best accelerator present, preferring the one the kernels are written for.

    ``torch_npu`` has to be imported before ``torch.npu`` exists, which is why
    the probe is an import rather than a ``hasattr``: on an Ascend host that has
    not imported it yet ``hasattr(torch, "npu")`` is False and the NPU goes
    unnoticed. Where it is not installed the ``ImportError`` is the answer, so a
    developer host reads ``cpu`` here and nothing below asks it for a device.
    """
    try:
        import torch_npu  # noqa: F401  (registers torch.npu)

        if torch.npu.is_available():
            return "npu:0"
    except (ImportError, AttributeError):
        pass
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def default_backend(device: str) -> str:
    """The Cube decode where there is an NPU to run it, its torch reference where there is not."""
    return "turboquant_cube" if device.startswith("npu") else reference_equivalent("turboquant_cube")


def build_parser() -> argparse.ArgumentParser:
    device = default_device()
    parser = argparse.ArgumentParser(
        prog="python tools/tq_longbench/diagnose.py",
        description="Per-layer cosine between the quantised and unquantised decode, on real weights.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--context", type=int, default=8192, help="approximate prompt tokens")
    parser.add_argument("--device", default=device, help="npu, npu:N, cuda:N or cpu (default: the best one present)")
    parser.add_argument(
        "--backend",
        default=default_backend(device),
        choices=quantised_backends(),
        help="the quantised decode under test; the dense side is always the dense_staging pool "
        "(default: the Cube decode on an NPU, its torch reference elsewhere)",
    )
    parser.add_argument("--dtype", default="float16", choices=sorted(_DTYPES))
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--attn-output-gate", action="store_true")
    parser.add_argument(
        "--fold-output-rotation",
        action="store_true",
        help="fold Pi into o_proj at ingestion, so the decode returns its output in the rotated basis",
    )
    parser.add_argument(
        "--dense-staging-backend",
        default=None,
        choices=sorted(DENSE_BACKENDS),
        help="pin the unquantised prefill pool rather than letting the pre-flight negotiate it",
    )
    parser.add_argument(
        "--dense-attention-api",
        default=None,
        choices=sorted(DENSE_ATTENTION_APIS),
        help="pin native_v5's torch_npu entry point rather than letting the pre-flight negotiate it",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the synthetic one-layer probe of the unquantised prefill path",
    )
    return parser


def prompt_tokens(model_path: str, prompt: str) -> torch.Tensor:
    """``prompt`` through the checkpoint's own tokenizer, GLM-4's remote-code one included.

    GLM-4's tokenizer is remote code, which a plain ``from_pretrained`` refuses,
    and on ``transformers`` 4.28 it needs the special tokens registering that
    :func:`~tq_longbench.smoke_glm.load_tokenizer` registers. The fallback is for
    a checkpoint that loader has no answer for: the same ``AutoTokenizer`` call,
    with the trust those configs ask for.

    ``use_chat_template=False`` keeps the bare document this script has always
    measured -- the numbers in the module docstring are that prompt's, and turn
    markers would move every token id underneath them.
    """
    family = family_for(read_config(model_path))
    try:
        tokenizer = load_tokenizer(model_path, family)
    except Exception as error:
        print(f"  {family.name} tokenizer loader failed ({error}); falling back to AutoTokenizer", file=sys.stderr)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return encode(tokenizer, prompt, use_chat_template=False, family=family)


def negotiate_dense_staging(args, device: torch.device) -> tuple[str | None, str | None]:
    """Which unquantised backend and entry point the dense_staging prefill runs through here.

    Probed on one synthetic layer at the checkpoint's own head counts, before a
    weight loads, and only on an NPU: elsewhere there is a single candidate
    (``dense_reference``) and nothing to negotiate. A pinned ``--dense-*`` flag is
    the answer and is not second-guessed, and a pre-flight that finds no path is
    not fatal -- the runner still has its own default, and the error it raises
    then is the one worth reading.
    """
    pinned = args.dense_staging_backend is not None or args.dense_attention_api is not None
    if device.type != "npu" or args.skip_preflight or pinned:
        return args.dense_staging_backend, args.dense_attention_api

    shape = read_checkpoint_shape(args.model_path, args.attn_output_gate)
    selection = select_dense_backend(
        device,
        LayerShape(shape.num_heads, shape.num_kv_heads, shape.head_size, shape.scale),
        _DTYPES[args.dtype],
        args.block_size,
        role="prefill",
        output_rotation_folded=args.fold_output_rotation,
    )
    if not selection.passed:
        for result in selection.results:
            print(f"  {result.describe()}", file=sys.stderr)
        print("  no unquantised prefill path passed the pre-flight; leaving the choice to the runner", file=sys.stderr)
        return None, args.dense_attention_api
    return selection.backend, selection.api


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
            f"the diagnostic needs two backends to compare, and --backend {runner.config.backend} decodes out "
            "of the same pool the dense_staging prefill filled. Choose a quantised --backend."
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
    device = resolve_device(args.device)

    item = needle_in_a_haystack(args.context, depth=0.0, seed=0)
    token_ids = prompt_tokens(args.model_path, item.prompt)

    dense_staging_backend, dense_attention_api = negotiate_dense_staging(args, device)

    runner = StandaloneModelRunner(
        RunnerConfig(
            model_path=args.model_path,
            backend=args.backend,
            prefill_mode="dense_staging",
            max_seq_len=token_ids.numel() + 64,
            chunk_size=args.chunk_size,
            block_size=args.block_size,
            dtype=_DTYPES[args.dtype],
            device=str(device),
            attn_output_gate=args.attn_output_gate,
            fold_output_rotation=args.fold_output_rotation,
            dense_staging_backend=dense_staging_backend,
            dense_attention_api=dense_attention_api,
        )
    )
    print(
        f"{args.model_path}: {runner.shape.num_layers} layers, {runner.shape.num_heads} heads / "
        f"{runner.shape.num_kv_heads} kv, head_size {runner.shape.head_size}"
    )
    print(f"dense {runner.prefill_backend.name} against quantised {runner.decode_backend.name}")
    print(f"prefill {token_ids.numel()} tokens on {device}, then one decode step\n")

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
