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
"""Smoke-test a dense GQA checkpoint through the standalone runner, against eager Hugging Face.

    python tools/tq_longbench/smoke_dense.py --model-path F:/AI/Qwen2.5-3B-Instruct

The whole stack is the harness's own here -- :mod:`tq_longbench.layers` builds
the model, :mod:`tq_longbench.engine` loads the weights and runs the loop, and
nothing of ``transformers`` is in the path but the tokenizer.  That is the
arrangement :mod:`tq_longbench.hf_bridge` exists to avoid needing, and it only
works for a model that is dense GQA all the way down.

Three claims, in order:

1. the shape the harness inferred is the checkpoint's, including the two things
   no config states (``qk_norm``, ``qkv_bias``);
2. the unquantised path reproduces eager Hugging Face token for token, so any
   later gap is the cache's and not the model's;
3. the 4-bit path's cost, in tokens and in megabytes.

Hugging Face is loaded and freed before the runner is built: two copies of a 3B
checkpoint do not fit beside two KV pools on a 12 GB card.

Measured on Qwen2.5-3B-Instruct, RTX 5070, float16 (2026-09-18): both paths
16/16 against eager HF on a short prompt, 288 B/token/layer against the dense
1024, and needle-in-a-haystack at 8192 retrieving 5/5 unquantised against 1/5
quantised -- the loss ``diagnose.py`` attributes to layers 0 and 20.
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
import gc  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from tq_longbench._ascend import assert_no_vllm_imported  # noqa: E402
from tq_longbench.engine import RunnerConfig, StandaloneModelRunner  # noqa: E402
from tq_longbench.tasks import needle_in_a_haystack  # noqa: E402

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

DEFAULT_PROMPT = "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is"

NIAH_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/tq_longbench/smoke_dense.py",
        description="Run a dense GQA checkpoint through the standalone runner and compare with eager HF.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=sorted(_DTYPES))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--niah",
        type=int,
        default=None,
        metavar="TOKENS",
        help="also run a needle-in-a-haystack sweep at roughly this context (e.g. 8192)",
    )
    return parser


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_mib() -> float:
    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def eager_reference(args, token_ids: torch.Tensor) -> list[int]:
    """Greedy continuation from ``transformers`` itself, then give the memory back."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=_DTYPES[args.dtype], device_map=args.device, attn_implementation="eager"
    ).eval()
    with torch.inference_mode():
        out = model.generate(
            token_ids.unsqueeze(0).to(args.device),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )
    produced = out[0, token_ids.numel() :].tolist()
    del model, out
    _free()
    return produced


def runner_for(args, backend: str, max_seq_len: int) -> StandaloneModelRunner:
    runner = StandaloneModelRunner(
        RunnerConfig(
            model_path=args.model_path,
            backend=backend,
            prefill_mode="dense_staging",
            max_seq_len=max_seq_len,
            chunk_size=args.chunk_size,
            block_size=args.block_size,
            dtype=_DTYPES[args.dtype],
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            seed=0,
        )
    )
    assert_no_vllm_imported()
    return runner


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    token_ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0].to(torch.int64)

    print("=== 1. eager Hugging Face ===")
    started = time.perf_counter()
    reference = eager_reference(args, token_ids)
    print(f"  {tokenizer.decode(reference, skip_special_tokens=True)!r}  ({time.perf_counter() - started:.1f}s)")

    print("\n=== 2. the standalone runner ===")
    backends = ("dense_reference", "turboquant_reference")
    reports = {}
    for backend in backends:
        _free()
        runner = runner_for(args, backend, max_seq_len=token_ids.numel() + args.max_new_tokens + 64)
        if backend == backends[0]:
            shape = runner.shape
            print(
                f"  inferred: {shape.num_layers} layers, {shape.num_heads} heads / {shape.num_kv_heads} kv, "
                f"head_size {shape.head_size}, ffn {shape.intermediate_size}"
            )
            print(
                f"  off the checkpoint: qk_norm={shape.qk_norm} qkv_bias={shape.qkv_bias} "
                f"attn_output_gate={shape.attn_output_gate} tied={shape.tie_word_embeddings}"
            )
        produced, metrics = runner.generate(token_ids.to(runner.device), max_new_tokens=args.max_new_tokens)
        agreed = sum(mine == theirs for mine, theirs in zip(produced, reference))
        report = runner.memory_report()
        reports[backend] = report
        print(
            f"  {backend:22s} {agreed:2d}/{len(reference)} vs eager HF   "
            f"{tokenizer.decode(produced, skip_special_tokens=True)!r}"
        )
        print(
            f"  {'':22s} decode kv {report['decode_cache_mb']:.1f} MB, "
            f"{report['bytes_per_token_per_layer']:.0f} B/token/layer, "
            f"decode p50 {metrics.summary()['decode_p50_us']:.0f} us, peak VRAM {_peak_mib():.0f} MiB"
        )
        del runner
        _free()

    dense, quantised = (reports[name]["bytes_per_token_per_layer"] for name in backends)
    print(f"\n  cache: {quantised:.0f} B/token/layer against {dense:.0f} ({dense / quantised:.2f}x)")

    if args.niah:
        print(f"\n=== 3. needle in a haystack at ~{args.niah} tokens ===")
        for backend in backends:
            _free()
            runner = runner_for(args, backend, max_seq_len=args.niah + 512)
            hits = 0
            for depth in NIAH_DEPTHS:
                item = needle_in_a_haystack(args.niah, depth=depth, seed=int(depth * 100))
                prompt_ids = tokenizer(item.prompt, return_tensors="pt").input_ids[0]
                prompt_ids = prompt_ids[: args.niah + 512 - args.max_new_tokens].to(torch.int64)
                produced, _ = runner.generate(prompt_ids.to(runner.device), max_new_tokens=args.max_new_tokens)
                answer = tokenizer.decode(produced, skip_special_tokens=True)
                found = item.answers[0] in answer
                hits += found
                print(
                    f"    depth {depth:<5} tokens={prompt_ids.numel():>6} found={str(found):5s} {answer.strip()[:44]!r}"
                )
            print(f"  {backend:22s} {hits}/{len(NIAH_DEPTHS)} retrieved, peak VRAM {_peak_mib():.0f} MiB")
            del runner
            _free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
