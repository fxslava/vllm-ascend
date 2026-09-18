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
"""CLI entry point: evaluate one backend on one task and write a JSONL record per item.

    python tools/tq_longbench/run_eval.py \\
        --model-path /path/to/weights \\
        --dataset longbench_narrativeqa \\
        --backend turboquant_cube \\
        --max-seq-len 131072 \\
        --out-file results.jsonl

One record per item, written as it completes rather than at the end, so a sweep
that dies at 100k still leaves the 99k that worked.

Run it as the script above, or as ``python -m tq_longbench.run_eval`` **with
``tools/`` appended** to ``PYTHONPATH``.  Appended, not prepended:
``tools/bisect/`` is a package whose name shadows the standard library's
``bisect``, and prepending ``tools/`` breaks ``import random`` -- that is,
Python itself -- before this module is reached.  :func:`_bootstrap_path` below
appends, for the same reason.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> None:
    """Make ``tq_longbench`` importable when this file is run as a script.

    Appended so the standard library keeps precedence over ``tools/bisect``; see
    the module docstring.
    """
    parent = str(Path(__file__).resolve().parents[1])
    if parent not in sys.path:
        sys.path.append(parent)


_bootstrap_path()

import argparse  # noqa: E402  (after the path bootstrap above)
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from tq_longbench.engine import (  # noqa: E402
    DENSE_STAGING_RECOMMENDED_MAX_SEQ,
    PREFILL_MODES,
    RunnerConfig,
    StandaloneModelRunner,
)
from tq_longbench.glm4 import glm4_prefill_mode, is_glm4_checkpoint  # noqa: E402
from tq_longbench.ops import DENSE_BACKENDS, backend_names, reference_equivalent  # noqa: E402
from tq_longbench.tasks import NIAH_TASK, EvalItem, load_longbench, niah_sweep, score  # noqa: E402

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tq_longbench.run_eval",
        description="LongBench evaluation over a TurboQuant or dense KV cache, with no vLLM in the process.",
    )
    parser.add_argument("--model-path", required=True, help="directory holding config.json and safetensors shards")
    parser.add_argument(
        "--dataset",
        default=NIAH_TASK,
        help="'niah' for the synthetic retrieval sweep, or longbench_<task> (e.g. longbench_narrativeqa)",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=backend_names(),
        help="default: turboquant_cube on an NPU, turboquant_reference elsewhere",
    )
    parser.add_argument(
        "--prefill-mode",
        default=None,
        choices=list(PREFILL_MODES),
        help=f"default: dense_staging at or below {DENSE_STAGING_RECOMMENDED_MAX_SEQ} tokens, batched_decode above",
    )
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=sorted(_DTYPES))
    parser.add_argument("--device", default=None, help="default: npu:0, else cuda:0, else cpu")
    parser.add_argument("--limit", type=int, default=None, help="evaluate at most this many items")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="override the task's own budget")
    parser.add_argument("--attn-output-gate", action="store_true", help="Qwen3.5-style [query | gate] projection")
    parser.add_argument(
        "--fold-output-rotation",
        action="store_true",
        help="rewrite o_proj to W_o (I (x) Pi) so the decode's output needs no un-rotation",
    )
    parser.add_argument(
        "--random-weights", action="store_true", help="skip the checkpoint; for shape and plumbing checks only"
    )
    parser.add_argument(
        "--niah-context",
        type=int,
        default=None,
        help="approximate prompt tokens for the niah sweep (default: max-seq-len minus headroom)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-file", type=Path, default=None, help="JSONL results; stdout if omitted")
    return parser


def default_device() -> str:
    """The best accelerator present, preferring the one the kernels are written for.

    ``torch_npu`` has to be imported before ``torch.npu`` exists, which is why
    the probe is an import rather than a ``hasattr``.
    """
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu:0"
    except (ImportError, AttributeError):
        pass
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def default_backend(device: torch.device) -> str:
    """The Cube decode where it exists, its torch reference where it does not."""
    return "turboquant_cube" if device.type == "npu" else reference_equivalent("turboquant_cube")


def default_prefill_mode(max_seq_len: int, backend: str, glm4_checkpoint: bool = False) -> str:
    """dense_staging while the fp16 pool is affordable, batched_decode past it.

    The threshold is about HBM, not accuracy: dense_staging gives the exact
    prefill and is what the NIAH alignment check wants, so it is preferred right
    up to the point where the unquantised pool stops fitting. A GLM-4 checkpoint
    switches at 32768 itself rather than above it (:func:`glm4_prefill_mode`).
    """
    if glm4_checkpoint:
        return glm4_prefill_mode(max_seq_len, backend)
    if backend in DENSE_BACKENDS:
        return "dense_staging"
    return "dense_staging" if max_seq_len <= DENSE_STAGING_RECOMMENDED_MAX_SEQ else "batched_decode"


def load_items(args: argparse.Namespace) -> list[EvalItem]:
    if args.dataset == NIAH_TASK:
        context = args.niah_context or max(1024, args.max_seq_len - 512)
        items = list(niah_sweep(context, seed=args.seed))
    else:
        items = list(load_longbench(args.dataset, limit=args.limit))
    if args.limit is not None:
        items = items[: args.limit]
    return items


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device or default_device())
    backend = args.backend or default_backend(device)
    prefill_mode = args.prefill_mode or default_prefill_mode(
        args.max_seq_len, backend, glm4_checkpoint=is_glm4_checkpoint(args.model_path)
    )

    config = RunnerConfig(
        model_path=args.model_path,
        backend=backend,
        prefill_mode=prefill_mode,
        max_seq_len=args.max_seq_len,
        chunk_size=args.chunk_size,
        block_size=args.block_size,
        dtype=_DTYPES[args.dtype],
        device=str(device),
        attn_output_gate=args.attn_output_gate,
        fold_output_rotation=args.fold_output_rotation,
        random_weights=args.random_weights,
        seed=args.seed,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    runner = StandaloneModelRunner(config)
    memory = runner.memory_report()
    print(
        f"[tq_longbench] backend={backend} device={device} prefill={prefill_mode} "
        f"max_seq_len={args.max_seq_len} kv={memory['kv_cache_mb']:.1f} MB "
        f"(dense equivalent {memory['dense_equivalent_mb']:.1f} MB)",
        file=sys.stderr,
    )

    items = load_items(args)
    sink = args.out_file.open("w", encoding="utf-8") if args.out_file else sys.stdout
    scores: list[float] = []
    try:
        for index, item in enumerate(items):
            token_ids = tokenizer(item.prompt, return_tensors="pt").input_ids[0]
            budget = args.max_new_tokens or item.max_new_tokens
            if token_ids.numel() + budget > args.max_seq_len:
                # Truncate from the middle, as LongBench does: the head carries
                # the instructions and the tail carries the question.
                keep = args.max_seq_len - budget
                half = keep // 2
                token_ids = torch.cat((token_ids[:half], token_ids[-(keep - half) :]))
            token_ids = token_ids.to(torch.int64).to(runner.device)

            started = time.perf_counter()
            produced, metrics = runner.generate(token_ids, max_new_tokens=budget)
            text = tokenizer.decode(produced, skip_special_tokens=True)
            value = score(text, item.answers, item.metric)
            scores.append(value)

            record = {
                "index": index,
                "backend": backend,
                "prefill_mode": prefill_mode,
                "metric": item.metric,
                "score": round(value, 4),
                "prediction": text,
                "answers": item.answers,
                "wall_seconds": round(time.perf_counter() - started, 3),
                **item.extra,
                **metrics.summary(),
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
    finally:
        if args.out_file:
            sink.close()

    if scores:
        mean = sum(scores) / len(scores)
        print(f"[tq_longbench] {args.dataset}: {mean:.4f} over {len(scores)} items", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
