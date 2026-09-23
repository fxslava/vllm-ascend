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
"""Partition LongBench into context tiers by *measured* length, never by truncation.

    python tools/tq_longbench/prepare_buckets.py --model-path ~/models/glm-4-9b-chat-1m \\
        --local-dir datasets/longbench --out-dir datasets/tiered --tiers 32k,256k,1m

``run_longbench.py``'s ladder is truncation: one dataset, cut from the middle to
each rung, which measures how much of an item's middle the model needed. That is
a real question and it is not this one. **This script builds three corpora whose
items are already the length they are meant to be**, so a 256k row is a document
that is genuinely 256k tokens long rather than a 32k document padded or a 1M
document with its heart cut out -- and a score on such a tier is a score on
long-context behaviour rather than on the harness's scissors.

The tiers, and where each is drawn from:

======  =====================  ========================================
Tier    Measured length        Corpus
======  =====================  ========================================
32k     8k <= L <= 32k         LongBench v1 (QA, summarisation, code)
256k    64k <= L <= 256k       LongBench-v2
1M      500k <= L <= 1M        LongBench-v2's tail, InfiniteBench
======  =====================  ========================================

Three things this does that a length field in the source rows cannot:

1. **The length is this checkpoint's.** ``L`` is measured by encoding the fully
   rendered prompt -- template, chat markers and all -- with the tokenizer at
   ``--model-path``, because that is the number that decides whether the item
   fits the KV arena. LongBench's own ``length`` column is a word count on the
   raw context and disagrees with it by a factor that varies per task.
2. **Nothing is clipped.** An item is admitted only if its whole rendered prompt
   lands inside the tier's ceiling, so the retained window contains the entire
   document including whatever span the answer is in. ``--require-answer`` goes
   further on the extractive tasks and checks the reference text actually occurs
   in the context; see :func:`answer_survives`.
3. **The ceiling is the prompt's, not the arena's.** The generation budget and
   the tokenizer headroom are the runner's problem and it adds them
   (``cache_size = tier_tokens + max_gen_tokens``); what is written here is a
   prompt that fits ``tier_tokens``.

Output is ``{out_dir}/{tier}/{task}.jsonl`` in LongBench v1's own row shape --
``context``, ``input``, ``answers``, ``all_classes`` -- so the existing loader
reads it unchanged, plus a ``tq`` sub-object carrying the measured length, the
source row's id and the tier. Each tier directory also gets a ``manifest.json``
with the per-task counts and the length distribution, which is what makes a run
reproducible without re-tokenising a million tokens to find out what was in it.

**Downloads are opt-in.** Without ``--download`` this reads only what is already
under ``--local-dir`` and says plainly which sources were missing; with it, the
missing ones are pulled through ``datasets`` and cached as JSONL under
``--local-dir`` so the second run needs no network. An evaluation host that has
the files never needs one.
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
from collections.abc import Callable, Iterable, Iterator, Sequence  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

from tq_longbench.metrics import TASK_METRICS, normalize_answer  # noqa: E402
from tq_longbench.tasks import DEFAULT_DATASET_DIR, longbench_config  # noqa: E402

#: Where the tiered corpora are written, relative to the repository root.
DEFAULT_OUT_DIR = "datasets/tiered"

#: The prompt LongBench-v2 items are rendered with. v2 ships no
#: ``dataset2prompt.json``; this is its paper's instruction, with the four
#: choices laid out one per line and the model asked for the letter alone --
#: which is what :func:`~tq_longbench.metrics.multiple_choice_score` reads.
LONGBENCH_V2_PROMPT = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "{input}\n\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)

#: InfiniteBench's long-book QA, rendered the way its own harness does: the book,
#: then the question, then a demand for brevity.
INFINITEBENCH_QA_PROMPT = (
    "Read the book below and answer a question.\n\n{context}\n\nQuestion: {input}\n\nBe very concise. Answer:"
)

#: Generation budgets for the two corpora that have none of their own. v2 wants a
#: letter and is given room for the sentence around it; InfiniteBench's long-book
#: QA references are phrases.
EXTRA_MAX_NEW_TOKENS = {"longbench_v2": 128, "infinitebench_qa": 64}

#: Tasks whose references are verbatim spans of the context, and so are the ones
#: :func:`answer_survives` can meaningfully check. Summarisation references are
#: written, not extracted; a multiple-choice reference is a letter; code
#: completion's is the next line, which may legitimately be novel.
EXTRACTIVE_TASKS = frozenset(
    {"narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", "triviaqa"}
)

#: A character-per-token band wide enough to hold every tokenizer and language in
#: the corpora, used only to skip an item before paying to encode it. The lower
#: bound decides what is too short to reach the floor, the upper what is too long
#: to fit the ceiling; both are deliberately loose, because a wrong skip is a
#: silently missing item and a wrong keep only costs one encode.
_MIN_CHARS_PER_TOKEN = 0.5
_MAX_CHARS_PER_TOKEN = 12.0


@dataclass(frozen=True)
class Source:
    """One corpus file under ``--local-dir``, and how to read a row out of it."""

    #: The task name a row becomes, which is also the JSONL stem on both sides.
    #: It has to be a :data:`~tq_longbench.metrics.TASK_METRICS` key, because
    #: that is what the runner scores on.
    task: str
    #: ``datasets`` coordinates for ``--download``. ``config`` is v1's per-task
    #: split name; v2 and InfiniteBench have one dataset each.
    repo: str
    config: str | None
    split: str
    #: Turns a source row into ``(context, input, answers, all_classes)``.
    reader: Callable[[dict], SourceRow]


@dataclass(frozen=True)
class SourceRow:
    """A corpus row reduced to the four fields every tier row is built from."""

    context: str
    input: str
    answers: list
    all_classes: list | None = None


def read_v1_row(row: dict) -> SourceRow:
    """A LongBench v1 row, which is already in the shape the harness wants."""
    return SourceRow(
        context=str(row.get("context", "")),
        input=str(row.get("input", "")),
        answers=[str(answer) for answer in row.get("answers", [])],
        all_classes=row.get("all_classes"),
    )


def read_v2_row(row: dict) -> SourceRow:
    """A LongBench-v2 row: the four choices are folded into ``input``, the answer is a letter.

    v2's row keeps ``question`` and ``choice_A`` through ``choice_D`` apart, and
    ``answer`` is the letter. Laying the choices out inside ``input`` rather than
    inside the template is what lets the row be stored, and later read, in v1's
    shape -- the loader formats ``{context}`` and ``{input}`` and knows nothing
    about choices.
    """
    choices = "\n".join(f"({letter}) {row.get('choice_' + letter, '')}" for letter in ("A", "B", "C", "D"))
    answer = str(row.get("answer", "")).strip().upper()[:1]
    return SourceRow(
        context=str(row.get("context", "")),
        input=f"What is the correct answer to this question: {row.get('question', '')}\nChoices:\n{choices}",
        answers=[answer] if answer else [],
    )


def read_infinitebench_row(row: dict) -> SourceRow:
    """An InfiniteBench row, whose ``answer`` is a string on some tasks and a list on others."""
    answer = row.get("answer", row.get("answers", []))
    answers = [str(answer)] if isinstance(answer, str) else [str(one) for one in answer or []]
    return SourceRow(context=str(row.get("context", "")), input=str(row.get("input", "")), answers=answers)


#: Every corpus this knows how to read, keyed by the task name it produces.
SOURCES = {
    **{
        task: Source(task=task, repo="THUDM/LongBench", config=task, split="test", reader=read_v1_row)
        for task in ("narrativeqa", "gov_report", "qasper", "multifieldqa_en", "trec", "lcc", "hotpotqa", "2wikimqa")
    },
    "longbench_v2": Source(
        task="longbench_v2", repo="THUDM/LongBench-v2", config=None, split="train", reader=read_v2_row
    ),
    "infinitebench_qa": Source(
        task="infinitebench_qa",
        repo="xinrongzhang2022/InfiniteBench",
        config=None,
        split="longbook_qa_eng",
        reader=read_infinitebench_row,
    ),
}


@dataclass(frozen=True)
class Tier:
    """One context bucket: the length band that defines it, and what it is drawn from."""

    name: str
    #: Inclusive, in tokens of the ``--model-path`` tokenizer, measured over the
    #: rendered prompt.
    min_tokens: int
    max_tokens: int
    #: :data:`SOURCES` keys, in the order they are read.
    sources: tuple
    why: str

    def admits(self, tokens: int) -> bool:
        return self.min_tokens <= tokens <= self.max_tokens


#: The three tiers, and why each band is where it is.
TIERS = {
    "32k": Tier(
        name="32k",
        min_tokens=8192,
        max_tokens=32768,
        sources=("narrativeqa", "gov_report", "qasper", "multifieldqa_en", "trec", "lcc"),
        why="v1's own range: the longest items the published suite contains, none of them truncated here",
    ),
    "256k": Tier(
        name="256k",
        min_tokens=65536,
        max_tokens=262144,
        sources=("longbench_v2",),
        why="v2's short and medium splits: multi-chapter reasoning, papers and repositories",
    ),
    "1m": Tier(
        name="1m",
        min_tokens=524288,
        max_tokens=1048576,
        sources=("longbench_v2", "infinitebench_qa"),
        why="v2's long split and InfiniteBench's books, inside GLM-4-9B-Chat-1M's native window",
    ),
}


@dataclass
class TierReport:
    """What one tier ended up holding, and what it turned away."""

    tier: Tier
    #: ``task -> [measured token lengths]``, in the order the rows were written.
    lengths: dict = field(default_factory=dict)
    #: ``reason -> count``, over every source row considered.
    rejected: dict = field(default_factory=dict)

    @property
    def kept(self) -> int:
        return sum(len(values) for values in self.lengths.values())

    def record(self, task: str, tokens: int) -> None:
        self.lengths.setdefault(task, []).append(tokens)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def manifest(self) -> dict:
        """The tier's ``manifest.json``: enough to reproduce it without re-tokenising."""
        every = [length for values in self.lengths.values() for length in values]
        return {
            "tier": self.tier.name,
            "min_tokens": self.tier.min_tokens,
            "max_tokens": self.tier.max_tokens,
            "why": self.tier.why,
            "items": self.kept,
            "tasks": {
                task: {
                    "items": len(values),
                    "min_tokens": min(values),
                    "max_tokens": max(values),
                    "median_tokens": int(statistics.median(values)),
                    "metric": _metric_name(task),
                    "max_new_tokens": _budget(task),
                }
                for task, values in sorted(self.lengths.items())
                if values
            },
            "token_length": {
                "min": min(every),
                "median": int(statistics.median(every)),
                "max": max(every),
            }
            if every
            else None,
            "rejected": dict(sorted(self.rejected.items())),
        }


def _metric_name(task: str) -> str:
    metric = TASK_METRICS.get(task)
    return getattr(metric, "__name__", "unknown")


def _budget(task: str, dataset_dir: Path | None = None) -> int:
    """The task's generation budget: the suite's where it has one, ours otherwise."""
    _, budgets = longbench_config(dataset_dir)
    if task in budgets:
        return int(budgets[task])
    return EXTRA_MAX_NEW_TOKENS.get(task, 64)


# ------------------------------------------------------------------ reading


def local_source_path(task: str, local_dir: Path) -> Path | None:
    """``{local_dir}/{task}.jsonl`` if it is there, else ``None``."""
    path = Path(local_dir) / f"{task}.jsonl"
    return path if path.is_file() else None


def iter_source_rows(task: str, local_dir: Path) -> Iterator[dict]:
    """Stream the raw rows of one corpus out of its local JSONL.

    Rows, not lines, so a blank line neither raises nor shifts the index a
    written item is traced back to its source by.
    """
    path = local_source_path(task, local_dir)
    if path is None:
        raise FileNotFoundError(f"{Path(local_dir) / f'{task}.jsonl'} does not exist; pass --download to fetch it")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def download_source(task: str, local_dir: Path) -> Path:
    """Fetch one corpus through ``datasets`` and cache it as JSONL under ``local_dir``.

    Written out as JSONL rather than left in the ``datasets`` cache so that the
    next run -- and every run on an air-gapped evaluation host the files are
    copied to -- needs nothing but this file. ``datasets`` is imported here and
    nowhere else in the harness, which is what keeps it off the dependency list
    for anyone who already has the corpus.
    """
    source = SOURCES[task]
    from datasets import load_dataset

    rows = load_dataset(source.repo, source.config, split=source.split)
    path = Path(local_dir) / f"{task}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return path


def ensure_sources(tasks: Iterable[str], local_dir: Path, download: bool) -> tuple:
    """``(present, missing)``, downloading the missing ones when asked to.

    Separated from the partitioning pass so that a missing corpus is reported
    before the tokenizer loads rather than after: the load is a minute, and the
    thing it is waiting to do cannot be done.
    """
    present, missing = [], []
    for task in tasks:
        if local_source_path(task, local_dir) is not None:
            present.append(task)
        elif download:
            print(f"downloading {SOURCES[task].repo} -> {local_dir / (task + '.jsonl')}", file=sys.stderr)
            download_source(task, local_dir)
            present.append(task)
        else:
            missing.append(task)
    return present, missing


# ------------------------------------------------------------- partitioning


def render_prompt(task: str, row: SourceRow, prompts: dict) -> str:
    """The item's full prompt text, through whichever template the task owns."""
    if task == "longbench_v2":
        template = LONGBENCH_V2_PROMPT
    elif task == "infinitebench_qa":
        template = INFINITEBENCH_QA_PROMPT
    else:
        template = prompts[task]
    return template.format(context=row.context, input=row.input)


def plausible_length(text: str, tier: Tier) -> bool:
    """Whether ``text`` could possibly tokenise into ``tier``, judged on characters alone.

    A 1M-token tier over a corpus of mostly 100k-token rows spends nearly all of
    its time encoding documents it is about to throw away. This is the cheap
    filter in front of that: a document too short to reach the floor even at the
    most generous characters-per-token, or too long to fit the ceiling even at
    the least generous, cannot land in the tier and is never encoded.

    Loose on purpose, in the direction that costs an encode rather than the one
    that loses an item.
    """
    characters = len(text)
    if characters < tier.min_tokens * _MIN_CHARS_PER_TOKEN:
        return False
    return characters <= tier.max_tokens * _MAX_CHARS_PER_TOKEN


def answer_survives(task: str, row: SourceRow) -> bool:
    """Whether an extractive task's reference is actually present in the retained context.

    Nothing here truncates, so the usual way an answer goes missing cannot
    happen. What this catches is the other way: a source row whose context does
    not contain its own answer -- ``narrativeqa`` references are written from the
    whole book and some of them are paraphrases, and an item nobody could answer
    from the text in front of them measures the model's priors, not its context.

    Only the extractive tasks are judged (:data:`EXTRACTIVE_TASKS`); everything
    else passes, because a summarisation reference or a choice letter appearing
    verbatim in the context would mean nothing either way.
    """
    if task not in EXTRACTIVE_TASKS or not row.answers:
        return True
    haystack = normalize_answer(row.context)
    return any(normalize_answer(answer) in haystack for answer in row.answers if answer.strip())


def tier_row(task: str, row: SourceRow, tokens: int, index: int, tier: Tier) -> dict:
    """One written row: LongBench v1's shape, plus what this script measured.

    The measurements go inside a ``tq`` sub-object rather than beside
    ``context``, so a row stays readable by anything that reads the suite's own
    JSONL and so nothing here can collide with a field a future corpus adds.
    """
    return {
        "context": row.context,
        "input": row.input,
        "answers": row.answers,
        "all_classes": row.all_classes,
        "length": tokens,
        "tq": {
            "tier": tier.name,
            "task": task,
            "source_index": index,
            "prompt_tokens": tokens,
            "max_new_tokens": _budget(task),
        },
    }


def partition(
    tier: Tier,
    local_dir: Path,
    out_dir: Path,
    measure: Callable[[str], int],
    limit: int | None,
    require_answer: bool,
    available: Sequence[str],
) -> TierReport:
    """Read every source of one tier, keep the rows that land in it, write them out.

    Streaming throughout: a 1M-token corpus does not fit in memory twice, and the
    per-task file is opened only once a row is actually going into it, so a tier
    that admits nothing from a task leaves no empty file to be mistaken for a
    task that scored zero.
    """
    prompts, _ = longbench_config(local_dir)
    report = TierReport(tier)
    directory = Path(out_dir) / tier.name
    directory.mkdir(parents=True, exist_ok=True)
    for task in tier.sources:
        if task not in available:
            report.reject(f"{task}: corpus not present")
            continue
        handle = None
        try:
            for index, raw in enumerate(iter_source_rows(task, local_dir)):
                if limit is not None and len(report.lengths.get(task, ())) >= limit:
                    break
                row = SOURCES[task].reader(raw)
                if not row.answers:
                    report.reject("no reference answer")
                    continue
                text = render_prompt(task, row, prompts)
                if not plausible_length(text, tier):
                    report.reject("outside the band (by character estimate)")
                    continue
                tokens = measure(text)
                if not tier.admits(tokens):
                    report.reject("outside the band (measured)")
                    continue
                if require_answer and not answer_survives(task, row):
                    report.reject("reference not found in context")
                    continue
                if handle is None:
                    handle = (directory / f"{task}.jsonl").open("w", encoding="utf-8")
                handle.write(json.dumps(tier_row(task, row, tokens, index, tier), ensure_ascii=False) + "\n")
                report.record(task, tokens)
        finally:
            if handle is not None:
                handle.close()
    (directory / "manifest.json").write_text(
        json.dumps(report.manifest(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


# --------------------------------------------------------------- the tokenizer


def build_measure(model_path: str, raw_prompt: bool) -> Callable[[str], int]:
    """A ``text -> tokens`` that encodes exactly the way the runner will.

    Through ``smoke_glm.encode`` and the checkpoint's own family rather than a
    bare ``tokenizer(text)``, because the chat template is worth hundreds of
    tokens on some checkpoints and is part of what has to fit. Imported inside
    the function so that ``--dry-run`` and the tests can exercise the
    partitioning with a measure of their own, on a host with neither the
    checkpoint nor transformers.
    """
    from tq_longbench.families import family_for
    from tq_longbench.glm4 import read_config
    from tq_longbench.smoke_glm import encode, load_tokenizer

    family = family_for(read_config(model_path))
    tokenizer = load_tokenizer(model_path, family)

    def measure(text: str) -> int:
        return int(encode(tokenizer, text, not raw_prompt, family).numel())

    return measure


# ---------------------------------------------------------------------- CLI


def parse_tiers(text: str) -> tuple:
    names = tuple(part.strip().lower() for part in text.split(",") if part.strip())
    unknown = [name for name in names if name not in TIERS]
    if not names or unknown:
        raise ValueError(f"--tiers: no such tier {unknown or 'nothing given'}; known {sorted(TIERS)}")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/tq_longbench/prepare_buckets.py",
        description="Partition LongBench v1/v2 and InfiniteBench into 32k, 256k and 1M context tiers by "
        "measured token length, without truncating any item.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="checkpoint whose tokenizer measures the lengths; the tiers are only valid for this one",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(DEFAULT_DATASET_DIR),
        help=f"directory of raw {{task}}.jsonl corpora (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR), help=f"default: {DEFAULT_OUT_DIR}")
    parser.add_argument("--tiers", default=",".join(TIERS), help=f"comma-separated; known {sorted(TIERS)}")
    parser.add_argument(
        "--download",
        action="store_true",
        help="fetch any corpus missing from --local-dir through `datasets`, and cache it there as JSONL",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap the items kept per task per tier")
    parser.add_argument(
        "--no-require-answer",
        action="store_true",
        help="keep extractive items whose reference does not occur in their context (they are dropped by default)",
    )
    parser.add_argument("--raw-prompt", action="store_true", help="measure without the chat template")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report which corpora are present and what each tier would read, and load no tokenizer",
    )
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    tiers = [TIERS[name] for name in parse_tiers(args.tiers)]
    wanted = sorted({task for tier in tiers for task in tier.sources})

    present, missing = ensure_sources(wanted, args.local_dir, args.download)
    for task in present:
        print(f"{task}: {local_source_path(task, args.local_dir)}", file=sys.stderr)
    for task in missing:
        print(f"{task}: MISSING from {args.local_dir} -- re-run with --download to fetch it", file=sys.stderr)
    if not present:
        print("no corpus present and none downloaded; nothing to partition", file=sys.stderr)
        return 1

    if args.dry_run:
        for tier in tiers:
            readable = [task for task in tier.sources if task in present]
            print(f"{tier.name}: {tier.min_tokens}-{tier.max_tokens} tokens from {readable or 'nothing'}")
        return 0

    measure = build_measure(args.model_path, args.raw_prompt)
    reports = []
    for tier in tiers:
        print(f"partitioning {tier.name} ({tier.min_tokens}-{tier.max_tokens} tokens)...", file=sys.stderr)
        reports.append(
            partition(
                tier,
                args.local_dir,
                args.out_dir,
                measure,
                args.limit,
                not args.no_require_answer,
                present,
            )
        )
    print()
    print(summarise(reports, args.out_dir))
    return 0


def summarise(reports: Sequence[TierReport], out_dir: Path) -> str:
    """The table the run exists to produce: what each tier holds and how long it is."""
    widths = (6, 18, 7, 11, 11, 11)
    header = f"| {'Tier':<6} | {'Task':<18} | {'Items':>7} | {'Min tok':>11} | {'Median tok':>11} | {'Max tok':>11} |"
    lines = [header, "|" + "|".join("-" * (width + 2) for width in widths) + "|"]
    for report in reports:
        manifest = report.manifest()
        if not manifest["tasks"]:
            lines.append(
                f"| {report.tier.name:<6} | {'(nothing admitted)':<18} | {0:>7} | {'':>11} | {'':>11} | {'':>11} |"
            )
            continue
        for task, entry in manifest["tasks"].items():
            lines.append(
                f"| {report.tier.name:<6} | {task:<18} | {entry['items']:>7} | {entry['min_tokens']:>11} | "
                f"{entry['median_tokens']:>11} | {entry['max_tokens']:>11} |"
            )
    notes = [
        "",
        f"Written to {out_dir}/{{tier}}/{{task}}.jsonl, with a manifest.json per tier.",
        "Lengths are the rendered prompt measured by the --model-path tokenizer, so they are that "
        "checkpoint's and no other's.",
        "No item was truncated: a row is in a tier because it already had that length.",
    ]
    for report in reports:
        if report.rejected:
            dropped = ", ".join(f"{count} {reason}" for reason, count in sorted(report.rejected.items()))
            notes.append(f"{report.tier.name} turned away: {dropped}.")
    return "\n".join(lines) + "\n" + "\n".join(notes)


if __name__ == "__main__":
    raise SystemExit(main())
