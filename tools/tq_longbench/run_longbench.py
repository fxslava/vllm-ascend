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

``--context-tier`` is the way out of that. ``prepare_buckets.py`` selects items
that already *are* 32k, 256k or 1M tokens long, and a tier run is a single rung
at the tier's ceiling over those items -- so nothing is cut, ``truncated`` is
zero throughout, and the score is about long context rather than about the
scissors. The arena follows the tier (``tier_tokens + max_gen_tokens +
headroom``), ``--max-tokens-override`` sets per-task generation budgets, and
``--max-context`` refuses to provision an arena nobody asked for::

    python tools/tq_longbench/run_longbench.py --model-path ~/models/glm-4-9b-chat-1m \\
        --context-tier 256k --max-tokens-override gov_report=1024,qasper=256 --decode-graph on

``--dry-run`` prints what each rung would provision, from ``config.json`` alone,
and loads no weights. Every record already carries ``kv_cache_mb``,
``dense_equivalent_mb`` and ``kv_saved_mb``; at a tier those are the numbers the
compression claim rests on, so they are on the table too.

**The prompt and the stop set are ``smoke_glm``'s**, so a task's score is not
quietly measuring a missing ``<|assistant|>`` marker or a continuation that ran
its whole budget.  ``--eval-fidelity`` adds the output-embedding cosine against
an unquantised prefill of the same prompt -- a different question from the
score, and capped at 32k because it needs both pools at once.
``--profile-layers`` additionally runs
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
from collections.abc import Iterator  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import torch  # noqa: E402

from tq_longbench.run_benchmark import (  # noqa: E402
    build_runner,
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
    model_directory,
    prompt_route,
    quiet_tensorflow,
    resolve_device,
    stop_at_eos,
    truncate_middle,
)

quiet_tensorflow()

from tq_longbench.engine import (  # noqa: E402
    DECODE_GRAPH_MODES,
    DEFAULT_CHUNK_SIZE,
    PREFILL_MODES,
    RunMetrics,
    read_checkpoint_shape,
)
from tq_longbench.families import ModelFamily, family_for  # noqa: E402
from tq_longbench.glm4 import read_config  # noqa: E402
from tq_longbench.kv_cache import CacheGeometry, dense_equivalent_bytes  # noqa: E402
from tq_longbench.layers import FOLD_SITES  # noqa: E402
from tq_longbench.metrics import (  # noqa: E402
    TASK_METRICS,
    as_percentage,
    metric_label,
    score_prediction,
    segmentation_backend,
)
from tq_longbench.ops import backend_names  # noqa: E402
from tq_longbench.prepare_buckets import DEFAULT_OUT_DIR as DEFAULT_TIER_DIR  # noqa: E402
from tq_longbench.prepare_buckets import TIERS  # noqa: E402
from tq_longbench.probe import LayerProbe, cosine  # noqa: E402
from tq_longbench.tasks import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    EvalItem,
    config_source,
    load_longbench_jsonl,
    local_task_path,
    longbench_config,
)

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

#: How many stand-in items a task gets when its JSONL is not on disk. Few, on
#: purpose: the fallback exists to prove the pipeline runs, not to produce a
#: number anyone might mistake for a benchmark.
STUB_ITEMS = 3

#: The arena a run will provision without being asked twice, in tokens. The 32k
#: and 256k tiers clear it; the 1M tier does not, and has to be let through with
#: an explicit ``--max-context``. A 1M-token arena is tens of gigabytes before a
#: single weight loads, and the failure mode without a guard is an allocator
#: abort a long way into a run that had already been paid for.
DEFAULT_MAX_CONTEXT = 262144

#: Above this, ``--eval-fidelity``'s cosine is refused. The number needs a second,
#: *unquantised* prefill of the same prompt held in memory beside the first --
#: which is the one thing a 256k or 1M run has no room for, and refusing is
#: better than discovering it after the dense pool is allocated. The dataset
#: metrics and the telemetry are unaffected and run at every tier.
FIDELITY_MAX_CONTEXT = 32768

#: Which unquantised backend ``--eval-fidelity`` measures against. The same one
#: ``--profile-layers`` uses, so the two numbers mean the same thing.
FIDELITY_REFERENCE_BACKEND = PROBE_REFERENCE_BACKEND

#: What a task whose budget nobody wrote down is given. LongBench sets one per
#: task and the tiered corpora carry their own; this is the floor under both.
DEFAULT_BUDGET = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/tq_longbench/run_longbench.py",
        description="LongBench sub-tasks across a context ladder, with TTFT, decode percentiles and peak HBM.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=model_directory,
        help="directory holding config.json and safetensors shards; a Windows path in either slash is fine",
    )
    parser.add_argument("--device", default="npu", help="npu, npu:N, cuda:N or cpu (default: npu)")
    parser.add_argument("--backends", default="turboquant_cube", help=f"comma-separated; one of {backend_names()}")
    parser.add_argument(
        "--tasks",
        default=None,
        help=f"comma-separated; known: {sorted(TASK_METRICS)} (default: {DEFAULT_TASKS}, or whatever "
        "--context-tier's manifest holds)",
    )
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS, help="comma-separated prompt lengths in tokens")
    parser.add_argument(
        "--context-tier",
        default=None,
        choices=sorted(TIERS),
        help="run one tier of untruncated items built by prepare_buckets.py instead of the --contexts ladder; "
        "the rung is the tier's ceiling and the arena is sized to it plus the generation budget",
    )
    parser.add_argument(
        "--tiered-dir",
        type=Path,
        default=Path(DEFAULT_TIER_DIR),
        help=f"where prepare_buckets.py wrote the tiers (default: {DEFAULT_TIER_DIR})",
    )
    parser.add_argument(
        "--max-context-len",
        type=int,
        default=None,
        help="run one length instead of the --contexts ladder; prompts longer than this are cut from the middle",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(DEFAULT_DATASET_DIR),
        help=f"directory of {{task}}.jsonl files (default: {DEFAULT_DATASET_DIR}); falls back to synthetic "
        "stubs where a file is missing",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="items per task per rung")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="override the task's own generation budget (LongBench sets one per task)",
    )
    parser.add_argument(
        "--max-tokens-override",
        default=None,
        help="per-bucket generation budgets: a bare number for every task, or comma-separated task=N pairs "
        "(gov_report=1024,qasper=256). Beats --max-new-tokens and the task's own budget.",
    )
    parser.add_argument(
        "--eval-fidelity",
        action="store_true",
        help=f"also measure each item's output embedding against a {FIDELITY_REFERENCE_BACKEND} prefill of the "
        f"same prompt; refused above {FIDELITY_MAX_CONTEXT} tokens, where the second prefill does not fit",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=DEFAULT_MAX_CONTEXT,
        help=f"refuse to provision an arena larger than this many tokens (default: {DEFAULT_MAX_CONTEXT}); "
        "raise it deliberately for the 1M tier",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the arena each rung would provision and what it would cost, then stop without loading weights",
    )
    parser.add_argument(
        "--decode-graph",
        default="auto",
        choices=list(DECODE_GRAPH_MODES),
        help="'on' refuses to fall back to an eager decode, which is what a benchmark wants (default: auto)",
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
    """``all`` for every known task, otherwise the named ones."""
    if text.strip() == "all":
        return tuple(TASK_METRICS)
    names = tuple(_task_name(part.strip()) for part in text.split(",") if part.strip())
    unknown = [name for name in names if name not in TASK_METRICS]
    if not names or unknown:
        raise ValueError(f"--tasks: no LongBench metric for {unknown or 'nothing given'}; known {sorted(TASK_METRICS)}")
    return names


def _task_name(part: str) -> str:
    """``longbench_qasper`` is ``qasper``; ``longbench_v2`` is itself.

    The prefix strip is a convenience for anyone who types the suite's own
    ``longbench_`` names, and it predates there being a task whose real name
    starts that way. The known name wins, so ``longbench_v2`` is not quietly
    turned into a ``v2`` that no metric answers to.
    """
    if part in TASK_METRICS:
        return part
    return part.removeprefix("longbench_")


def parse_budget_overrides(text: str | None) -> dict[str | None, int]:
    """``--max-tokens-override``: a bare number, or ``task=N`` pairs.

    Returned with ``None`` as the key of the bare form, so one lookup answers
    both: a task's own entry if it has one, else the blanket one, else nothing
    and the task's published budget stands. A budget is part of what a score
    means -- ``gov_report`` scored with 64 tokens of room is not ``gov_report``
    -- so an override travels into every record.
    """
    if text is None:
        return {}
    text = text.strip()
    if not text:
        return {}
    if text.isdigit():
        return {None: int(text)}
    overrides: dict[str | None, int] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        task, _, value = part.partition("=")
        task, value = task.strip(), value.strip()
        if not value.isdigit() or int(value) <= 0:
            raise ValueError(f"--max-tokens-override: {part.strip()!r} is not `task=N` with a positive N")
        name = _task_name(task)
        if name not in TASK_METRICS:
            raise ValueError(f"--max-tokens-override: no LongBench metric for {task!r}; known {sorted(TASK_METRICS)}")
        overrides[name] = int(value)
    if not overrides:
        raise ValueError(f"--max-tokens-override: {text!r} is neither a number nor any `task=N` pair")
    return overrides


def item_budget(args: argparse.Namespace, item: EvalItem) -> int:
    """How many tokens this item may generate, in the order the flags override each other."""
    task = item.extra.get("task")
    if task in args.budget_overrides:
        return args.budget_overrides[task]
    if None in args.budget_overrides:
        return args.budget_overrides[None]
    return args.max_new_tokens or item.max_new_tokens


def tier_tasks(directory: Path) -> tuple[str, ...]:
    """Which tasks a prepared tier actually holds, from its manifest or its files.

    The manifest first, because it is what ``prepare_buckets.py`` wrote and it
    records the tasks that admitted at least one item. Globbing is the fallback
    for a directory assembled by hand.
    """
    manifest = directory / "manifest.json"
    if manifest.is_file():
        held = json.loads(manifest.read_text(encoding="utf-8")).get("tasks", {})
        if held:
            return tuple(sorted(held))
    return tuple(sorted(path.stem for path in directory.glob("*.jsonl")))


def apply_tier(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """``--context-tier``: the rung and the tasks it implies, or the ladder's own.

    A tier is a **single** rung at its ceiling, not a ladder: its items were
    selected for already being that long, so a shorter rung would start cutting
    them again and throw away the one property the tier was built for.
    """
    if args.context_tier is None:
        contexts = (args.max_context_len,) if args.max_context_len else parse_int_list(args.contexts, "contexts")
        return contexts, parse_tasks(args.tasks or DEFAULT_TASKS)

    tier = TIERS[args.context_tier]
    directory = args.tiered_dir / tier.name
    if not directory.is_dir():
        raise SystemExit(
            f"--context-tier {tier.name}: {directory} does not exist. Build it first:\n"
            f"  python tools/tq_longbench/prepare_buckets.py --model-path {args.model_path} "
            f"--out-dir {args.tiered_dir} --tiers {tier.name}"
        )
    args.dataset_dir = directory
    tasks = parse_tasks(args.tasks) if args.tasks else tier_tasks(directory)
    if not tasks:
        raise SystemExit(f"--context-tier {tier.name}: {directory} holds no task JSONL; re-run prepare_buckets.py")
    return (tier.max_tokens,), tasks


def guard_arena(args: argparse.Namespace, context: int, max_seq_len: int) -> None:
    """Refuse an arena past ``--max-context`` before anything is allocated.

    The 1M tier is the case this exists for. Provisioning its cache is tens of
    gigabytes of device memory committed up front, and without a guard the way
    that goes wrong is an allocator abort partway through a run whose weights
    had already loaded -- ten minutes to learn something the token count knew at
    the start.
    """
    if max_seq_len <= args.max_context:
        return
    raise SystemExit(
        f"a {context}-token rung needs a {max_seq_len}-token arena, past the {args.max_context}-token "
        f"--max-context guard. Re-run with --max-context {max_seq_len} once you have checked the device has "
        f"room for it, or with --dry-run to see what it would cost."
    )


def arena_report(args: argparse.Namespace, contexts: tuple[int, ...], tasks: tuple[str, ...]) -> str:
    """``--dry-run``: what each rung would provision, from the config alone.

    Reads ``config.json`` and nothing else -- no weights, no device, no
    tokenizer -- because the point is to answer "will this fit" before paying
    for the run that would otherwise answer it. The dense figure is exact
    arithmetic over the geometry; what the TurboQuant pool actually holds is
    measured per item and lands in the records as ``kv_cache_mb``.
    """
    shape = read_checkpoint_shape(args.model_path)
    budget = _largest_budget(args, tasks)
    lines = [
        f"{'Rung':>9} {'Arena':>9} {'Dense KV (MB)':>14} {'Guard':>9}",
        "-" * 45,
    ]
    for context in contexts:
        max_seq_len = context + budget + CONTEXT_HEADROOM_TOKENS
        geometry = CacheGeometry(
            num_layers=shape.num_layers,
            num_kv_heads=shape.num_kv_heads,
            head_size=shape.head_size,
            block_size=args.block_size,
            max_seq_len=max_seq_len,
        )
        dense_mb = dense_equivalent_bytes(geometry, _DTYPES[args.dtype]) / (1024 * 1024)
        verdict = "ok" if max_seq_len <= args.max_context else "REFUSED"
        lines.append(f"{context:>9} {max_seq_len:>9} {dense_mb:>14.0f} {verdict:>9}")
    lines += [
        "",
        f"tasks: {', '.join(tasks)} (generation budget {budget} tokens, --max-context {args.max_context})",
        "Dense KV is what the same arena would cost unquantised, from the checkpoint geometry; the "
        "TurboQuant pool's own size is measured per run and reported as kv_cache_mb.",
    ]
    if any(context + budget + CONTEXT_HEADROOM_TOKENS > args.max_context for context in contexts):
        lines.append("A REFUSED rung needs an explicit --max-context to run.")
    return "\n".join(lines)


def load_items(task: str, dataset_dir: Path, limit: int) -> tuple[list[EvalItem], str]:
    """``(items, where they came from)``: the local JSONL if it is there, else a stub.

    Falling back rather than failing, because the ladder is useful before the
    corpus arrives -- but the source travels into every record and onto the
    table, because a score computed over three synthetic items is not a LongBench
    score and the two must never be read as the same number.
    """
    if local_task_path(task, dataset_dir) is not None:
        return list(load_longbench_jsonl(task, dataset_dir, limit=limit)), "jsonl"
    return list(synthetic_items(task, dataset_dir, limit=min(limit, STUB_ITEMS))), "stub"


def synthetic_items(task: str, dataset_dir: Path, limit: int) -> Iterator[EvalItem]:
    """Stand-in items in the task's own prompt template, for a host without the corpus.

    Deliberately trivial: the answer is in the context verbatim, so a working
    pipeline scores near the top and a broken one scores zero. It measures the
    harness, never the model.
    """
    prompts, budgets = longbench_config(dataset_dir)
    for index in range(limit):
        answer = f"stub answer {index}"
        yield EvalItem(
            prompt=prompts[task].format(context=f"The answer to question {index} is {answer}. " * 8, input="what?"),
            answers=[answer],
            metric=task,
            max_new_tokens=budgets.get(task, 64),
            extra={"task": task, "index": index, "all_classes": [answer, "other"], "synthetic": True},
        )


@dataclass
class TaskRung:
    """One (backend, context, task): every item scored, and what they cost."""

    backend: str
    context: int
    task: str
    metric: str
    #: ``"jsonl"`` or ``"stub"``. On the table, because a stub row is not a score.
    source: str = "jsonl"
    records: list[dict] = field(default_factory=list)

    @property
    def percentage(self) -> float:
        """The suite's aggregate: the mean, times 100, to two places."""
        return as_percentage(record["score"] for record in self.records)

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

    @property
    def fidelity(self) -> float | None:
        """The worst output-embedding cosine over the items, or ``None`` if none was measured.

        The worst rather than the mean: a rung whose items sit at 0.999 except
        for one at 0.4 has a problem, and an average of 0.94 is exactly the
        number that would let it through.
        """
        measured = [r["output_cosine"] for r in self.records if r.get("output_cosine") is not None]
        return min(measured) if measured else None


def run_item(args, runner, tokenizer, eos_ids, item, keep: int, family: ModelFamily) -> tuple[dict, torch.Tensor]:
    """One LongBench item, truncated to the rung, scored by its own metric."""
    full = encode(tokenizer, item.prompt, not args.raw_prompt, family)
    prompt_ids = truncate_middle(full, keep)
    budget = item_budget(args, item)
    task = item.extra.get("task")
    started = time.perf_counter()
    produced, metrics = runner.generate(prompt_ids.to(runner.device), max_new_tokens=budget, stop_ids=eos_ids)
    text, failure = decode_text(tokenizer, stop_at_eos(produced, eos_ids))
    record = {
        "backend": runner.config.backend,
        "task": task,
        "index": item.extra.get("index"),
        "metric": metric_label(task),
        "synthetic": bool(item.extra.get("synthetic")),
        "score": round(score_prediction(task, text, item.answers, item.extra.get("all_classes")), 4),
        "prediction": text,
        "answers": item.answers,
        "decode_failure": failure,
        "truncated": int(full.numel() > keep),
        "untruncated_tokens": int(full.numel()),
        "max_new_tokens": budget,
        "wall_seconds": round(time.perf_counter() - started, 3),
        **metrics.summary(),
    }
    return record, prompt_ids


def fidelity_cosine(args, runner, prompt_ids: torch.Tensor, plan) -> float:
    """One item's output embedding against an unquantised prefill of the same prompt.

    The last layer's hidden state for the last prefill token -- the vector the
    logits that choose the first generated token are read off -- taken from this
    run and from a :data:`FIDELITY_REFERENCE_BACKEND` run of the same ids, and
    compared with the same :func:`~tq_longbench.probe.cosine` the layer table
    uses.

    This is a different question from the score beside it, and that is the point:
    a score says whether the answer was right, and a right answer can come out of
    a stack that has drifted a long way from the dense one. The cosine says how
    far it drifted, at the one point where the drift becomes the answer.

    The reference runner is built, prefilled and dropped per item, so the two
    pools are never held at once -- which is the whole reason the flag is capped
    at :data:`FIDELITY_MAX_CONTEXT`.
    """
    # Sized for the prompt and nothing else: the reference is prefilled and
    # dropped, never generated from, so a generation budget would be arena it
    # holds for no reason at the one moment two pools are closest to coexisting.
    reference_runner = build_runner(
        args, FIDELITY_REFERENCE_BACKEND, "dense_staging", int(prompt_ids.numel()) + CONTEXT_HEADROOM_TOKENS, plan
    )
    try:
        with LayerProbe(reference_runner) as baseline:
            reference_runner.prefill(prompt_ids.to(reference_runner.device), RunMetrics())
        reference = dict(baseline.hidden)
    finally:
        del reference_runner
        free_device(args.device)

    with LayerProbe(runner) as probe:
        runner.prefill(prompt_ids.to(runner.device), RunMetrics())
    last = max(probe.hidden) if probe.hidden else None
    if last is None or last not in reference:
        return float("nan")
    return round(cosine(probe.hidden[last], reference[last]), 6)


def profile_item(args, runner, tokenizer, prompt_ids: torch.Tensor, item, plan) -> str:
    """The layer table for one item, against a ``cann_dense`` prefill of the same prompt.

    The reference is prefilled and thrown away: only its per-layer hidden states
    are kept, which is what ``hidden_cosine`` needs and all it needs. Two prefills
    of a 64k prompt is what this costs, which is why it runs on one item a rung.
    """
    target = _target_token(tokenizer, item)
    # The reference only ever prefills, so it is sized for the prompt rather than
    # through ``cache_tokens`` -- which reads ``--max-new-tokens`` and is ``None``
    # whenever the budget came from the task or from ``--max-tokens-override``.
    reference_runner = build_runner(
        args, PROBE_REFERENCE_BACKEND, "dense_staging", int(prompt_ids.numel()) + CONTEXT_HEADROOM_TOKENS, plan
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
    """The markdown table the run exists to produce, one block per (backend, context).

    Per backend and rung rather than one table for everything, because a score
    only means something next to the length it was measured at: ``narrativeqa``
    at 4k and at 64k are different numbers about the same items, and a single
    column would average them into one that is about neither.

    ``Overall Average`` is the mean of the task scores, which is what LongBench
    reports -- not the mean over items, which would weight a task by how many of
    them it happens to have.
    """
    widths = (20, 11, 9, 7, 9, 12, 13, 10, 10, 10, 9)
    blocks = []
    for (backend, context), group in _grouped(rungs):
        header = (
            f"| {'Task':<20} | {'Metric':<11} | {'Score (%)':>9} | {'Samples':>7} | "
            f"{'TTFT (ms)':>9} | {'Dec p50 (us)':>12} | {'Peak HBM (MB)':>13} | "
            f"{'KV (MB)':>10} | {'Dense (MB)':>10} | {'Saved (MB)':>10} | {'Fidelity':>9} |"
        )
        rule = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
        lines = [f"### {backend} @ {context} tokens", "", header, rule]
        for rung in group:
            note = " *" if rung.source == "stub" else ""
            fidelity = f"{rung.fidelity:.6f}" if rung.fidelity is not None else "-"
            lines.append(
                f"| {rung.task + note:<20} | {rung.metric:<11} | {rung.percentage:>8.1f}% | "
                f"{len(rung.records):>7} | {rung.median('ttft_ms'):>9.0f} | "
                f"{rung.median('decode_p50_us'):>12.0f} | {rung.peak('device_peak_mb'):>13.0f} | "
                f"{rung.peak('kv_cache_mb'):>10.0f} | {rung.peak('dense_equivalent_mb'):>10.0f} | "
                f"{rung.peak('kv_saved_mb'):>10.0f} | {fidelity:>9} |"
            )
        average = as_percentage([rung.percentage / 100 for rung in group])
        lines.append(
            f"| {'Overall Average':<20} | {'':<11} | {average:>8.1f}% | {'':>7} | {'':>9} | {'':>12} | "
            f"{'':>13} | {'':>10} | {'':>10} | {'':>10} | {'':>9} |"
        )
        blocks.append("\n".join(lines))

    notes = [
        "",
        "Score is the task's own LongBench metric as a percentage: the mean over items x 100.",
        "Overall Average is the mean of the task scores, not of the items, which is how the suite reports it.",
        "TTFT and Dec p50 are medians over a task's items; Peak HBM is the high-water mark over them.",
        "KV is what the quantised pool holds, Dense what the same arena would cost unquantised, Saved the "
        "difference -- all three sized for the rung's arena rather than for the prompt that filled it.",
    ]
    if any(rung.fidelity is not None for rung in rungs):
        notes.append(
            f"Fidelity is the *worst* output-embedding cosine over a task's items against a "
            f"{FIDELITY_REFERENCE_BACKEND} prefill of the same prompt -- the drift at the vector the first "
            "generated token is chosen from, not whether the answer was right."
        )
    if any(rung.source == "stub" for rung in rungs):
        notes.append("* no JSONL on disk for this task: it ran on synthetic stubs and measures the harness only.")
    if any(record.get("truncated") for rung in rungs for record in rung.records):
        notes.append(
            "Items longer than the rung were cut from the middle, so those rows are not comparable with "
            "published LongBench numbers; every JSONL record carries its own `truncated` flag."
        )
    return "\n\n".join(blocks) + "\n" + "\n".join(notes)


def _grouped(rungs: list[TaskRung]):
    """``((backend, context), rungs)``, in the order they were run."""
    order, grouped = [], {}
    for rung in rungs:
        key = (rung.backend, rung.context)
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append(rung)
    return [(key, grouped[key]) for key in order]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.device = resolve_device(args.device)
    args.budget_overrides = parse_budget_overrides(args.max_tokens_override)
    backends = parse_backends(args.backends)
    contexts, tasks = apply_tier(args)
    if args.eval_fidelity and max(contexts) > FIDELITY_MAX_CONTEXT:
        raise SystemExit(
            f"--eval-fidelity needs a second, unquantised prefill of the same prompt held beside the first, "
            f"which does not fit above {FIDELITY_MAX_CONTEXT} tokens; the longest rung here is {max(contexts)}. "
            "Drop the flag -- the dataset metrics and the telemetry are measured at every tier -- or run the "
            "32k tier."
        )
    if args.dry_run:
        print(arena_report(args, contexts, tasks))
        return 0
    for context in contexts:
        guard_arena(args, context, context + _largest_budget(args, tasks) + CONTEXT_HEADROOM_TOKENS)
    config = read_config(args.model_path)
    family = family_for(config)
    tokenizer = load_tokenizer(args.model_path, family)
    eos_ids = eos_ids_for(config, tokenizer, family)
    print(f"prompt: {prompt_route(tokenizer, not args.raw_prompt, family)}", file=sys.stderr)
    print(f"stop ids: {sorted(eos_ids) or 'NONE -- every item will run its whole budget'}", file=sys.stderr)

    print(f"prompts: {config_source(args.dataset_dir)}", file=sys.stderr)
    print(f"chinese segmentation: {segmentation_backend()}", file=sys.stderr)

    # Loaded once and reused at every rung: the rung truncates the same items.
    items, sources = {}, {}
    for task in tasks:
        items[task], sources[task] = load_items(task, args.dataset_dir, args.limit)
        where = str(args.dataset_dir / f"{task}.jsonl") if sources[task] == "jsonl" else "SYNTHETIC STUBS"
        print(f"{task}: {len(items[task])} items from {where}", file=sys.stderr)
    if any(source == "stub" for source in sources.values()):
        print(
            "warning: some tasks fell back to stubs -- those rows measure the harness, not the model",
            file=sys.stderr,
        )

    sink = args.out_file.open("w", encoding="utf-8") if args.out_file else sys.stdout
    rungs: list[TaskRung] = []
    try:
        for backend in backends:
            plan = plan_for_backend(args, backend, contexts, family)
            if plan.results or plan.notes:
                print(plan.report(), file=sys.stderr)
            runner = None
            for context in contexts:
                mode = rung_prefill_mode(args, plan, backend, context, family)
                max_seq_len = context + _largest_budget(args, tasks) + CONTEXT_HEADROOM_TOKENS
                runner = None
                free_device(args.device)
                runner = build_runner(args, backend, mode, max_seq_len, plan)
                print(f"  {backend} at {context} ({mode}, cache {max_seq_len})", file=sys.stderr)
                keep = max_seq_len - CONTEXT_HEADROOM_TOKENS - _largest_budget(args, tasks)
                for task in tasks:
                    rung = TaskRung(backend, context, task, metric_label(task), sources[task])
                    for position, item in enumerate(items[task]):
                        record, prompt_ids = run_item(args, runner, tokenizer, eos_ids, item, keep, family)
                        record.update(context_requested=context, prefill_mode=mode)
                        if args.eval_fidelity:
                            record["output_cosine"] = fidelity_cosine(args, runner, prompt_ids, plan)
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


def _largest_budget(args: argparse.Namespace, tasks: tuple[str, ...]) -> int:
    """The longest continuation any task at this rung can ask for.

    One cache size per rung rather than one per task, so the rung's memory
    numbers describe the rung. gov_report's 512 tokens set it for the others,
    unless ``--max-tokens-override`` set a longer one for some task -- which is
    the point of the flag, and would be an arena short by the difference if it
    were not read here.
    """
    overrides = getattr(args, "budget_overrides", {})
    if None in overrides:
        return overrides[None]
    _, budgets = longbench_config(args.dataset_dir)
    published = args.max_new_tokens or 0
    if not published:
        published = max((budgets.get(task, DEFAULT_BUDGET) for task in tasks), default=DEFAULT_BUDGET)
    return max([published, *(overrides[task] for task in tasks if task in overrides)])


if __name__ == "__main__":
    raise SystemExit(main())
