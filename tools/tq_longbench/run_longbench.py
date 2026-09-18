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
"""LongBench tasks across a context ladder, with the layer probe on demand.

    python tools/tq_longbench/run_longbench.py --model-path /models/glm-4-9b-chat-1m \\
        --tasks narrativeqa,qasper,gov_report,multifieldqa_en \\
        --contexts 4096,8192,16384,32768,65536 --out-file longbench.jsonl

Three CLIs now, and they answer different questions. ``run_eval.py`` runs one
task at one length. ``run_benchmark.py`` runs the synthetic needle up a ladder,
where the right answer is known exactly and retrieval is unambiguous. This one
runs the *published* tasks up the same ladder, where the score is a real number
against real references and the interesting quantity is how it moves as the
context grows.

**Context scaling on a fixed dataset is truncation**, and it has to be: a
LongBench item is as long as it is. Each item's prompt is cut to the rung's
length the way the suite itself cuts -- from the middle, so the instructions at
the head and the question at the tail both survive -- which means a short rung
is the same item with less of its middle. That makes the ladder a measure of how
much of the middle the model needed, which is the question worth asking of a KV
cache, and it makes the scores incomparable to published numbers at any rung
that actually truncated. The record says how many items were cut.

**The prompt and the stop set are ``smoke_glm``'s**, so a task's score is not
quietly measuring a missing ``<|assistant|>`` marker or a continuation that ran
its whole budget.  ``--profile-layers`` additionally runs
:mod:`tq_longbench.probe` over the first item of each (backend, context, task),
against an fp16 ``cann_dense`` prefill of the same prompt, and prints the
layer-by-layer table.
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
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import torch  # noqa: E402

from tq_longbench.run_benchmark import (  # noqa: E402
    build_runner,
    cache_tokens,
    free_device,
    parse_backends,
    parse_int_list,
    plan_for_backend,
    rung_prefill_mode,
)
from tq_longbench.smoke_glm import (  # noqa: E402
    CONTEXT_HEADROOM_TOKENS,
    decode_text,
    encode,
    eos_ids_for,
    load_tokenizer,
    prompt_route,
    quiet_tensorflow,
    resolve_device,
    stop_at_eos,
    truncate_middle,
)

quiet_tensorflow()

from tq_longbench.engine import DEFAULT_CHUNK_SIZE, PREFILL_MODES, RunMetrics  # noqa: E402
from tq_longbench.glm4 import is_glm4_config, read_config  # noqa: E402
from tq_longbench.layers import FOLD_SITES  # noqa: E402
from tq_longbench.ops import backend_names  # noqa: E402
from tq_longbench.probe import LayerProbe  # noqa: E402
from tq_longbench.tasks import LONGBENCH_MAX_NEW_TOKENS, LONGBENCH_PROMPTS, load_longbench, score  # noqa: E402

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

#: The ladder, when ``--contexts`` is left alone.
DEFAULT_CONTEXTS = "4096,8192,16384,32768,65536"

#: The four the objective asks for, in the order they stress different things: two
#: single-document QA tasks, one multi-field, one long-form summarisation.
DEFAULT_TASKS = "narrativeqa,qasper,gov_report,multifieldqa_en"

#: Items per task per rung. A full LongBench split at five rungs is thousands of
#: prefills; the default is a sweep you can read the same day.
DEFAULT_LIMIT = 20

#: Which unquantised backend the ``--profile-layers`` reference prefill uses.
PROBE_REFERENCE_BACKEND = "cann_dense"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/tq_longbench/run_longbench.py",
        description="LongBench sub-tasks across a context ladder, with TTFT, decode percentiles and peak HBM.",
    )
    parser.add_argument("--model-path", required=True, help="directory holding config.json and safetensors shards")
    parser.add_argument("--device", default="npu", help="npu, npu:N, cuda:N or cpu (default: npu)")
    parser.add_argument("--backends", default="turboquant_cube", help=f"comma-separated; one of {backend_names()}")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help=f"comma-separated; known: {sorted(LONGBENCH_PROMPTS)}")
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS, help="comma-separated prompt lengths in tokens")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="items per task per rung")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="override the task's own generation budget (LongBench sets one per task)",
    )
    parser.add_argument("--prefill-mode", default=None, choices=list(PREFILL_MODES))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=sorted(_DTYPES))
    parser.add_argument("--no-fold-output-rotation", action="store_true")
    parser.add_argument("--raw-prompt", action="store_true", help="skip the chat template")
    parser.add_argument("--fold-site", default="auto", choices=list(FOLD_SITES))
    parser.add_argument(
        "--profile-layers",
        action="store_true",
        help="run the layer probe over the first item of each rung, against a cann_dense prefill of the "
        "same prompt, and print where the two parted company",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-file", type=Path, default=None, help="JSONL results; stdout if omitted")
    return parser


def parse_tasks(text: str) -> tuple[str, ...]:
    names = tuple(part.strip().removeprefix("longbench_") for part in text.split(",") if part.strip())
    unknown = [name for name in names if name not in LONGBENCH_PROMPTS]
    if not names or unknown:
        raise ValueError(
            f"--tasks: no prompt template for {unknown or 'nothing given'}; known {sorted(LONGBENCH_PROMPTS)}"
        )
    return names


@dataclass
class TaskRung:
    """One (backend, context, task): every item scored, and what they cost."""

    backend: str
    context: int
    task: str
    metric: str
    records: list[dict] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return statistics.mean(r["score"] for r in self.records) if self.records else 0.0

    @property
    def truncated(self) -> int:
        """Items whose middle was cut to reach this rung -- the ones not comparable to published scores."""
        return sum(r["truncated"] for r in self.records)

    def median(self, key: str) -> float:
        return statistics.median([r[key] for r in self.records]) if self.records else 0.0

    def peak(self, key: str) -> float:
        return max((r[key] for r in self.records), default=0.0)


def run_item(args, runner, tokenizer, eos_ids, item, keep: int, glm4: bool) -> tuple[dict, torch.Tensor]:
    """One LongBench item, truncated to the rung, scored by its own metric."""
    full = encode(tokenizer, item.prompt, not args.raw_prompt, glm4=glm4)
    prompt_ids = truncate_middle(full, keep)
    budget = args.max_new_tokens or item.max_new_tokens
    started = time.perf_counter()
    produced, metrics = runner.generate(prompt_ids.to(runner.device), max_new_tokens=budget, stop_ids=eos_ids)
    text, failure = decode_text(tokenizer, stop_at_eos(produced, eos_ids))
    record = {
        "backend": runner.config.backend,
        "task": item.extra.get("task"),
        "index": item.extra.get("index"),
        "metric": item.metric,
        "score": round(score(text, item.answers, item.metric), 4),
        "prediction": text,
        "answers": item.answers,
        "decode_failure": failure,
        "truncated": int(full.numel() > keep),
        "untruncated_tokens": int(full.numel()),
        "wall_seconds": round(time.perf_counter() - started, 3),
        **metrics.summary(),
    }
    return record, prompt_ids


def profile_item(args, runner, tokenizer, prompt_ids: torch.Tensor, item, plan) -> str:
    """The layer table for one item, against a ``cann_dense`` prefill of the same prompt.

    The reference is prefilled and thrown away: only its per-layer hidden states
    are kept, which is what ``hidden_cosine`` needs and all it needs. Two prefills
    of a 64k prompt is what this costs, which is why it runs on one item a rung.
    """
    target = _target_token(tokenizer, item)
    reference_runner = build_runner(
        args, PROBE_REFERENCE_BACKEND, "dense_staging", cache_tokens(args, int(prompt_ids.numel())), plan
    )
    with LayerProbe(reference_runner) as baseline:
        reference_runner.prefill(prompt_ids.to(reference_runner.device), RunMetrics())
    hidden = dict(baseline.hidden)
    del reference_runner
    free_device(args.device)

    with LayerProbe(runner, target_token=target, reference=hidden) as probe:
        runner.prefill(prompt_ids.to(runner.device), RunMetrics())
    return probe.report().describe(limit=12)


def summarise(rungs: list[TaskRung]) -> str:
    """One line per (backend, context, task)."""
    header = (
        f"{'backend':20s} {'task':17s} {'context':>8s} {'items':>6s} {'cut':>4s} {'score':>7s} "
        f"{'ttft ms':>10s} {'p50 us':>9s} {'p99 us':>9s} {'kv MB':>8s} {'peak MB':>8s}"
    )
    lines = [header, "-" * len(header)]
    for rung in rungs:
        lines.append(
            f"{rung.backend:20s} {rung.task:17s} {rung.context:>8d} {len(rung.records):>6d} "
            f"{rung.truncated:>4d} {rung.mean_score:>7.4f} {rung.median('ttft_ms'):>10.0f} "
            f"{rung.median('decode_p50_us'):>9.0f} {rung.median('decode_p99_us'):>9.0f} "
            f"{rung.median('kv_cache_mb'):>8.1f} {rung.peak('device_peak_mb'):>8.0f}"
        )
    lines.append("")
    lines.append("score is the task's own metric (f1, or rouge_l for gov_report), averaged over its items.")
    lines.append("cut counts items whose middle was truncated to reach the rung: those are not comparable")
    lines.append("to published LongBench numbers, which score the whole item.")
    lines.append("ttft/p50/p99/kv are medians over the items; peak MB is the high-water mark over them.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.device = resolve_device(args.device)
    backends = parse_backends(args.backends)
    tasks = parse_tasks(args.tasks)
    contexts = parse_int_list(args.contexts, "contexts")
    config = read_config(args.model_path)
    glm4 = is_glm4_config(config)
    tokenizer = load_tokenizer(args.model_path) if glm4 else _auto_tokenizer(args.model_path)
    eos_ids = eos_ids_for(config, tokenizer)
    print(f"prompt: {prompt_route(tokenizer, not args.raw_prompt, glm4)}", file=sys.stderr)
    print(f"stop ids: {sorted(eos_ids) or 'NONE -- every item will run its whole budget'}", file=sys.stderr)

    # Loaded once and reused at every rung: the rung truncates the same items.
    items = {task: list(load_longbench(task, limit=args.limit)) for task in tasks}
    for task, loaded in items.items():
        print(f"{task}: {len(loaded)} items", file=sys.stderr)

    sink = args.out_file.open("w", encoding="utf-8") if args.out_file else sys.stdout
    rungs: list[TaskRung] = []
    try:
        for backend in backends:
            plan = plan_for_backend(args, backend, contexts, glm4)
            if plan.results or plan.notes:
                print(plan.report(), file=sys.stderr)
            runner = None
            for context in contexts:
                mode = rung_prefill_mode(args, plan, backend, context, glm4)
                max_seq_len = context + _largest_budget(args, tasks) + CONTEXT_HEADROOM_TOKENS
                runner = None
                free_device(args.device)
                runner = build_runner(args, backend, mode, max_seq_len, plan)
                print(f"  {backend} at {context} ({mode}, cache {max_seq_len})", file=sys.stderr)
                keep = max_seq_len - CONTEXT_HEADROOM_TOKENS - _largest_budget(args, tasks)
                for task in tasks:
                    rung = TaskRung(backend, context, task, items[task][0].metric if items[task] else "f1")
                    for position, item in enumerate(items[task]):
                        record, prompt_ids = run_item(args, runner, tokenizer, eos_ids, item, keep, glm4)
                        record.update(context_requested=context, prefill_mode=mode)
                        rung.records.append(record)
                        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                        sink.flush()
                        if args.profile_layers and position == 0:
                            print(f"    layer probe, {task} item 0 at {context}:", file=sys.stderr)
                            print(profile_item(args, runner, tokenizer, prompt_ids, item, plan), file=sys.stderr)
                    rungs.append(rung)
                    print(
                        f"    {task:17s} {rung.mean_score:.4f} over {len(rung.records)} items "
                        f"({rung.truncated} cut), ttft {rung.median('ttft_ms'):.0f} ms",
                        file=sys.stderr,
                    )
            runner = None
            free_device(args.device)
    finally:
        if args.out_file:
            sink.close()

    print()
    print(summarise(rungs))
    return 0


def _target_token(tokenizer, item) -> int | None:
    """The first token of the first reference answer, for the logit lens to rank.

    A LongBench item has no single right token the way a needle has, so this is
    the nearest honest stand-in: the token the reference answer opens with. A
    rank that falls through the stack says the layers stopped heading toward the
    reference; it does not say they headed somewhere wrong.
    """
    if not item.answers:
        return None
    ids = tokenizer(item.answers[0], return_tensors="pt").input_ids[0]
    return int(ids[-1]) if ids.numel() else None


def _largest_budget(args, tasks: tuple[str, ...]) -> int:
    """The longest continuation any task at this rung can ask for.

    One cache size per rung rather than one per task, so the rung's memory
    numbers describe the rung. gov_report's 512 tokens set it for the others.
    """
    if args.max_new_tokens:
        return args.max_new_tokens
    return max(LONGBENCH_MAX_NEW_TOKENS[task] for task in tasks)


def _auto_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


if __name__ == "__main__":
    raise SystemExit(main())
