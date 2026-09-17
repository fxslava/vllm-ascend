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
"""Smoke-test a real checkpoint through the TurboQuant cache, and say what it cost.

    python tools/tq_longbench/smoke_hf.py --model-path F:/AI/Qwen3.5-2B

Four things are established, in order, because each is only meaningful if the
one before it held:

1. **The weights load clean.** Missing, unexpected and mismatched keys are all
   reported; a silently skipped projection is fluent nonsense, not a crash.
2. **The bridge is in the path.** The hook is counted. A backend that was never
   called agrees with the baseline perfectly, so agreement proves nothing until
   this does.
3. **The bridge is faithful.** With the dense backend the model must reproduce
   its own unpatched output. Any gap here is the bridge's, and would otherwise
   be charged to quantisation.
4. **The quantisation cost.** Only then is the 4-bit path's cosine against the
   unpatched model a measurement of the thing under test.

Run with ``--device cuda:0`` (the default where CUDA is present) to exercise the
torch reference backends; they compute what the kernels compute and say nothing
about what the kernels cost.
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
import time  # noqa: E402

import torch  # noqa: E402

from tq_longbench.hf_bridge import TurboQuantAttentionBridge, discover_full_attention_layers  # noqa: E402

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

DEFAULT_PROMPT = (
    "Notes. "
    + "Routine maintenance and weather logs were filed. " * 40
    + "\nImportant: the vault access code is 84731.\n"
    + "More notes. "
    + "Nothing of consequence occurred. " * 40
    + "\nQuestion: what is the vault access code?\nAnswer:"
)

#: The needle in DEFAULT_PROMPT. Retrieval across the whole prefix is the point;
#: the continuation after it is unconstrained and diverges freely.
DEFAULT_NEEDLE = "84731"


def cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    left = actual.flatten().to(torch.float64)
    right = expected.flatten().to(torch.float64)
    return float(torch.dot(left, right) / (left.norm() * right.norm()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/tq_longbench/smoke_hf.py",
        description="Run a real checkpoint with its full-attention layers on the TurboQuant cache.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(_DTYPES))
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prompt", default=None, help=f"default: a needle prompt containing {DEFAULT_NEEDLE}")
    parser.add_argument("--needle", default=None, help="string the continuation must contain")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device, dtype = args.device, _DTYPES[args.dtype]
    prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPT
    needle = args.needle if args.needle is not None else (DEFAULT_NEEDLE if args.prompt is None else None)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=== 1. weights ===")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model, loading = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, device_map=device, output_loading_info=True
    )
    model.eval()
    print(f"  {type(model).__name__} in {time.perf_counter() - started:.1f}s on {device}")
    clean = True
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = loading.get(key, [])
        clean &= not values
        print(f"  {key}: {len(values)}{'' if not values else '  ' + str(values[:4])}")
    print(f"  clean: {clean}")

    attentions = discover_full_attention_layers(model)
    probe = next(iter(attentions.values()))
    print(
        f"  full-attention layers claimed: {sorted(attentions)} "
        f"(head_dim={probe.head_dim}, heads={probe.o_proj.in_features // probe.head_dim}, "
        f"kv_heads={probe.k_proj.out_features // probe.head_dim})"
    )

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    print(f"  prompt tokens: {ids.numel()}")

    pristine = probe.config._attn_implementation

    @torch.inference_mode()
    def generate(bridge: TurboQuantAttentionBridge | None) -> tuple[list[int], str, int]:
        calls = {"n": 0}
        if bridge is None:
            out = model.generate(
                ids, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        else:
            inner = bridge._attention

            def counting(*call_args, **call_kwargs):
                calls["n"] += 1
                return inner(*call_args, **call_kwargs)

            bridge._attention = counting
            with bridge:
                bridge.reset()
                out = model.generate(
                    ids, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
            bridge._attention = inner
            if probe.config._attn_implementation != pristine:
                raise RuntimeError(
                    f"the bridge leaked: attention implementation is {probe.config._attn_implementation!r}, "
                    f"expected {pristine!r}. Every later measurement would compare a backend against itself."
                )
        produced = out[0, ids.shape[1] :]
        return produced.tolist(), tokenizer.decode(produced, skip_special_tokens=True), calls["n"]

    @torch.inference_mode()
    def last_logits(bridge: TurboQuantAttentionBridge | None) -> torch.Tensor:
        if bridge is None:
            return model(ids).logits[0, -1].float().cpu()
        with bridge:
            bridge.reset()
            return model(ids).logits[0, -1].float().cpu()

    def bridge_for(backend: str) -> TurboQuantAttentionBridge:
        return TurboQuantAttentionBridge(model, backend, max_seq_len=args.max_seq_len, block_size=args.block_size)

    print("\n=== 2. the bridge is in the path ===")
    baseline_tokens, baseline_text = generate(None)[:2]
    print(f"  unpatched:            {baseline_text!r}")
    dense_tokens, dense_text, dense_calls = generate(bridge_for("dense_reference"))
    print(f"  dense_reference:      {dense_text!r}")
    print(f"  hook calls: {dense_calls} = {len(attentions)} layers x {dense_calls // len(attentions)} forwards")
    if not dense_calls:
        raise RuntimeError("the bridge was never called; nothing below this line means anything")

    print("\n=== 3. the bridge is faithful ===")
    unpatched_logits = last_logits(None)
    dense_logits = last_logits(bridge_for("dense_reference"))
    print(f"  cos(dense_reference, unpatched) = {cosine(dense_logits, unpatched_logits):.8f}")
    print(f"  greedy tokens agree: {dense_tokens == baseline_tokens}")

    print("\n=== 4. what 4 bits cost ===")
    quantised = bridge_for("turboquant_reference")
    turboquant_tokens, turboquant_text, _ = generate(quantised)
    turboquant_logits = last_logits(bridge_for("turboquant_reference"))
    agreed = 0
    for mine, theirs in zip(turboquant_tokens, baseline_tokens):
        if mine != theirs:
            break
        agreed += 1
    print(f"  turboquant_reference: {turboquant_text!r}")
    print(f"  cos(turboquant, unpatched) = {cosine(turboquant_logits, unpatched_logits):.8f}")
    print(f"  argmax agrees: {int(turboquant_logits.argmax()) == int(unpatched_logits.argmax())}")
    print(f"  leading greedy tokens matching the baseline: {agreed}/{len(baseline_tokens)}")
    if needle:
        print(
            f"  needle {needle!r} retrieved: baseline={needle in baseline_text} turboquant={needle in turboquant_text}"
        )

    report = quantised.memory_report()
    print(
        f"\n=== 5. memory over {report['claimed_layers']} layers at max_seq_len={args.max_seq_len} ===\n"
        f"  turboquant {report['kv_cache_mb']:.2f} MB vs dense {report['dense_equivalent_mb']:.2f} MB "
        f"({report['dense_equivalent_mb'] / report['kv_cache_mb']:.2f}x), "
        f"{report['bytes_per_token_per_layer']:.0f} B/token/layer"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
