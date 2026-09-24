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
"""The evaluation items and the scores, with no framework in between.

Two sources of items:

* **LongBench**, loaded through ``datasets``.  Each task carries its own prompt
  template and its own metric, which is why the metric travels with the task
  rather than being chosen on the command line.
* **Needle in a haystack**, generated here.  It exists because it is the one
  task whose correct answer is known exactly, at any context length, without a
  checkpoint or a download -- which makes it the retrieval gate the harness can
  hold itself to rather than a benchmark it reports.
"""

from __future__ import annotations

import json
import random
import string
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

#: LongBench's own prompt templates and per-task generation budgets, kept as data
#: rather than as Python. Two reasons, and the second is the real one:
#:
#: * a prompt full of newlines and quotes is a bad thing to escape into a source
#:   file, and a prompt that is subtly wrong scores differently without failing;
#: * this file is ``dataset2prompt.json`` and ``dataset2maxlen.json`` in the
#:   suite's own shape, so it can be diffed against the copy that ships with a
#:   download -- and :func:`longbench_config` prefers that copy where there is
#:   one. The table here is a transcription and is treated as one.
PROMPT_TABLE = Path(__file__).with_name("longbench_prompts.json")

#: The corpora that are **not** LongBench v1, and so are deliberately not in
#: :data:`PROMPT_TABLE` -- that file is the suite's own two config files and has
#: to stay diffable against the copy a download ships. These live here instead,
#: and :func:`longbench_config` merges them in, because everything downstream
#: (``load_longbench_jsonl``, ``prepare_buckets``, the runner) asks that one
#: function what a task's prompt is and a task with no answer there cannot be
#: loaded at all.
#:
#: ``longbench_v2`` is four-way multiple choice. The choices are folded into
#: ``input`` when the row is written (``prepare_buckets.read_v2_row``) rather
#: than kept as separate ``choice_A``..``choice_D`` fields, which is what lets a
#: v2 row be stored and read in v1's ``{context}``/``{input}`` shape. The closing
#: instruction is v2's own, and is what
#: :func:`~tq_longbench.metrics.multiple_choice_score` reads back.
EXTRA_PROMPTS = {
    "longbench_v2": (
        "Please read the following text and answer the question below.\n\n"
        "<text>\n{context}\n</text>\n\n"
        "{input}\n\n"
        'Format your response as follows: "The correct answer is (insert answer here)".'
    ),
    "infinitebench_qa": (
        "Read the book below and answer a question.\n\n{context}\n\nQuestion: {input}\n\nBe very concise. Answer:"
    ),
}

#: Generation budgets for the same two. v2 wants a letter and is given room for
#: the sentence around it; InfiniteBench's long-book QA references are phrases.
EXTRA_MAX_NEW_TOKENS = {"longbench_v2": 128, "infinitebench_qa": 64}


def longbench_config(dataset_dir: str | Path | None = None) -> tuple[dict, dict]:
    """``(prompts, budgets)``, from the downloaded suite's own config where it has one.

    A LongBench checkout keeps ``config/dataset2prompt.json`` and
    ``config/dataset2maxlen.json`` next to the data. Where ``dataset_dir`` has
    them they win outright: they are the file the published numbers were produced
    with, and this module's copy is a transcription that can only be as good as
    the transcribing. Where it does not, the transcription is used and the caller
    is told which -- :func:`config_source`.

    :data:`EXTRA_PROMPTS` is merged in underneath both, so the non-v1 corpora the
    context tiers are drawn from can be loaded. A downloaded config still wins
    over them, on the same reasoning: whatever a corpus ships with is the thing
    its numbers were produced with.
    """
    shipped = json.loads(PROMPT_TABLE.read_text(encoding="utf-8"))
    prompts = {**shipped["dataset2prompt"], **EXTRA_PROMPTS}
    budgets = {**shipped["dataset2maxlen"], **EXTRA_MAX_NEW_TOKENS}
    for name, target in (("dataset2prompt", prompts), ("dataset2maxlen", budgets)):
        path = _config_path(dataset_dir, name)
        if path is not None:
            target.update(json.loads(path.read_text(encoding="utf-8")))
    return prompts, budgets


def config_source(dataset_dir: str | Path | None = None) -> str:
    """Which config a run used, for the banner. Never left to be inferred from a score."""
    found = [name for name in ("dataset2prompt", "dataset2maxlen") if _config_path(dataset_dir, name)]
    if len(found) == 2:
        return f"{dataset_dir}/config (the suite's own)"
    if found:
        return f"{dataset_dir}/config for {found[0]}, this harness's transcription for the rest"
    return "this harness's transcription of dataset2prompt.json"


def _config_path(dataset_dir: str | Path | None, name: str) -> Path | None:
    if dataset_dir is None:
        return None
    for candidate in (Path(dataset_dir) / "config" / f"{name}.json", Path(dataset_dir) / f"{name}.json"):
        if candidate.is_file():
            return candidate
    return None


LONGBENCH_PROMPTS, LONGBENCH_MAX_NEW_TOKENS = longbench_config()

NIAH_TASK = "niah"


@dataclass
class EvalItem:
    """One prompt, its references, and how to score a continuation of it."""

    prompt: str
    answers: list[str]
    metric: str
    max_new_tokens: int
    #: Free-form, carried into the results file: context length, needle depth.
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ metrics


#: SQuAD normalisation drops these wherever they stand alone.
_ARTICLES = frozenset(("a", "an", "the"))

_PUNCTUATION = frozenset(string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD normalisation: lowercase, strip articles, punctuation and extra space.

    The reference implementation substitutes ``\\b(a|an|the)\\b`` with a space and
    then collapses whitespace.  Punctuation is already gone by that point, so the
    word boundaries can only be spaces -- which makes the substitution exactly a
    filter over the split tokens, and saves a regex dependency.
    """
    stripped = "".join(character for character in text.lower() if character not in _PUNCTUATION)
    return " ".join(word for word in stripped.split() if word not in _ARTICLES)


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-overlap F1, the LongBench QA metric."""
    predicted = normalize_answer(prediction).split()
    reference = normalize_answer(ground_truth).split()
    common = Counter(predicted) & Counter(reference)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def rouge_l(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F-measure from the longest common subsequence.

    Written out rather than pulled from ``rouge_score``: the harness's whole
    premise is a short dependency list, and LCS over two token lists is six
    lines.
    """
    predicted = normalize_answer(prediction).split()
    reference = normalize_answer(ground_truth).split()
    if not predicted or not reference:
        return 0.0
    table = [[0] * (len(reference) + 1) for _ in range(len(predicted) + 1)]
    for i, left in enumerate(predicted, start=1):
        for j, right in enumerate(reference, start=1):
            table[i][j] = table[i - 1][j - 1] + 1 if left == right else max(table[i - 1][j], table[i][j - 1])
    lcs = table[-1][-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(predicted)
    recall = lcs / len(reference)
    return 2 * precision * recall / (precision + recall)


def substring_match(prediction: str, ground_truth: str) -> float:
    """Retrieval: did the needle come back at all, anywhere in the continuation."""
    return float(normalize_answer(ground_truth) in normalize_answer(prediction))


def needle_match(prediction: str, ground_truth: str) -> float:
    """The needle as a **whole token** of the continuation, not as a substring of one.

    :func:`substring_match` is what LongBench-style retrieval usually means, and
    for a passcode it is one character too generous: the needle ``894772`` is a
    substring of ``1894772`` and of ``8947720``, so a model that emitted a longer
    number it invented would score as having retrieved. Tokenising both sides
    first closes that, and closes nothing else -- a continuation that carries the
    passcode anywhere still counts, which is the point of the task.

    It does not, and cannot, distinguish a retrieved needle from a recited one:
    a model that copies the whole needle sentence out of the prompt has still
    read it out of the context. :func:`needle_report` is what separates answering
    from rambling.
    """
    return float(normalize_answer(ground_truth) in normalize_answer(prediction).split())


METRICS = {
    "f1": f1_score,
    "rouge_l": rouge_l,
    "exact_match": exact_match,
    "substring_match": substring_match,
    "needle_match": needle_match,
}

_TASK_METRICS = {
    "narrativeqa": "f1",
    "qasper": "f1",
    "hotpotqa": "f1",
    "2wikimqa": "f1",
    "multifieldqa_en": "f1",
    # A summarisation task, so overlap of the longest common subsequence rather
    # than of the bag of tokens: gov_report's references are paragraphs.
    "gov_report": "rouge_l",
}


def score(prediction: str, answers: list[str], metric: str) -> float:
    """Best score over the references, which is how LongBench aggregates."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose one of {sorted(METRICS)}")
    scorer = METRICS[metric]
    return max((scorer(prediction, answer) for answer in answers), default=0.0)


# -------------------------------------------------------------------- items


@dataclass(frozen=True)
class NeedleReport:
    """What a continuation did with the needle, past the yes/no of finding it.

    ``found`` alone is what made a run of answers like ``"894772. The secret
    passcode for "`` and ``"581850. 581850. 581850. "`` print as ``5/5``: every
    one of them *had* retrieved the passcode, and the table had no column in
    which the rest of the sentence could appear. These are those columns.

    ``found`` is still the retrieval verdict, and still the only one used to
    score. The rest describe the continuation around it:

    * ``answered_first`` -- the needle is the first number in the continuation.
      A model that answers and stops has it; a model that wanders through the
      haystack and hits the passcode on the way does not.
    * ``first_number`` -- what it did lead with, so a miss can be read.
    * ``echoed_prompt`` -- the needle sentence or the question came back with it,
      which is recitation rather than an answer, though still retrieval.
    * ``degenerate`` -- the continuation is a loop. Not a retrieval failure and
      not scored as one, but it is what a collapsing decode looks like, so it
      does not belong hidden inside a passing row.
    """

    found: bool
    answered_first: bool
    first_number: str | None
    echoed_prompt: bool
    degenerate: bool

    @property
    def clean(self) -> bool:
        """Retrieved, led with it, and did not fall into a loop."""
        return self.found and self.answered_first and not self.degenerate

    def describe(self) -> str:
        """The one-word verdict for a results table."""
        if not self.found:
            return "missed"
        flags = [name for name, on in (("late", not self.answered_first), ("loop", self.degenerate)) if on]
        return "found" if not flags else "found+" + "+".join(flags)


def needle_report(prediction: str, needle: str, city: str | None = None) -> NeedleReport:
    """Score one continuation against one passcode, and describe what else it did.

    ``needle`` is the passcode itself. ``city`` is the needle sentence's subject
    when there is one; it only sharpens ``echoed_prompt``, which is a diagnostic,
    never the verdict.
    """
    normalized = normalize_answer(prediction)
    words = normalized.split()
    numbers = [word for word in words if word.isdigit()]
    target = normalize_answer(needle)
    stem = normalize_answer(_NIAH_NEEDLE.format(city=city or "", code="")).replace(" is", "").strip()
    return NeedleReport(
        found=target in words,
        answered_first=bool(numbers) and numbers[0] == target,
        first_number=numbers[0] if numbers else None,
        echoed_prompt=bool(stem) and stem in normalized,
        degenerate=_is_degenerate(words),
    )


#: A continuation this long or longer is judged for repetition; below it there is
#: not enough of one to tell a loop from an answer that happens to repeat a word.
_DEGENERATE_MIN_WORDS = 6

#: Distinct words over total, under which the continuation is a loop rather than a
#: sentence. "581850. 581850. 581850. 581850." is 0.25; ordinary English prose does
#: not come near it until it is hundreds of words long, which a needle answer is not.
_DEGENERATE_DISTINCT_RATIO = 0.35


def _is_degenerate(words: list[str]) -> bool:
    """Whether a continuation is repeating itself rather than saying something."""
    if len(words) < _DEGENERATE_MIN_WORDS:
        return False
    return len(set(words)) / len(words) < _DEGENERATE_DISTINCT_RATIO


_NIAH_NEEDLE = "The secret passcode for {city} is {code}."
_NIAH_QUESTION = "What is the secret passcode for {city}?"

# Filler that says nothing the question could be answered from, so a hit is
# retrieval rather than a plausible guess.
_NIAH_FILLER = (
    "The grass is green and the sky is blue. Trees grow slowly near the river. "
    "People walk along the path in the morning and return before the evening. "
)

_NIAH_CITIES = ("Harbin", "Lisbon", "Nairobi", "Quito", "Reykjavik", "Tashkent", "Valencia", "Wellington")


def needle_in_a_haystack(
    approximate_tokens: int,
    depth: float = 0.5,
    seed: int = 0,
    chars_per_token: int = 4,
) -> EvalItem:
    """A haystack of roughly ``approximate_tokens`` with one retrievable fact at ``depth``.

    ``depth`` is the fraction of the haystack before the needle, so 0.0 puts it
    first and 1.0 last.  The length is approximate because it is built in
    characters and tokenised later; the harness reports the tokenised length it
    actually ran, not this estimate.
    """
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must lie in [0, 1], got {depth}")
    rng = random.Random(seed)
    city = rng.choice(_NIAH_CITIES)
    code = f"{rng.randrange(10**5, 10**6)}"
    needle = _NIAH_NEEDLE.format(city=city, code=code)

    target_chars = max(len(needle) * 2, approximate_tokens * chars_per_token)
    repeats = max(1, target_chars // len(_NIAH_FILLER))
    haystack = _NIAH_FILLER * repeats
    cut = int(len(haystack) * depth)
    # Land on a sentence boundary so the needle is not spliced mid-word.
    cut = haystack.rfind(" ", 0, cut) + 1 if cut else 0
    document = haystack[:cut] + needle + " " + haystack[cut:]

    prompt = (
        "Read the document and answer the question using only the document.\n\n"
        f"Document: {document}\n\n"
        f"Question: {_NIAH_QUESTION.format(city=city)}\nAnswer:"
    )
    return EvalItem(
        prompt=prompt,
        answers=[code],
        metric="needle_match",
        max_new_tokens=16,
        extra={"task": NIAH_TASK, "depth": depth, "city": city, "approximate_tokens": approximate_tokens},
    )


def niah_sweep(
    context_tokens: int, depths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0), seed: int = 0
) -> Iterator[EvalItem]:
    """One item per depth, which is how a retrieval regression shows itself.

    A needle only ever placed mid-document hides the failure that matters: a
    decode whose context length is short by a block loses the *end* of the
    haystack and nothing else.
    """
    for index, depth in enumerate(depths):
        yield needle_in_a_haystack(context_tokens, depth=depth, seed=seed + index)


#: Where a downloaded LongBench lives, relative to the repository root.
DEFAULT_DATASET_DIR = "datasets/longbench"


def local_task_path(task: str, dataset_dir: str | Path) -> Path | None:
    """``{dataset_dir}/{task}.jsonl`` if it is there, else ``None``."""
    path = Path(dataset_dir) / f"{task}.jsonl"
    return path if path.is_file() else None


def load_longbench_jsonl(task: str, dataset_dir: str | Path, limit: int | None = None) -> Iterator[EvalItem]:
    """Stream a task out of a local ``{task}.jsonl``, through its own prompt template.

    The suite ships one JSON object per line with ``context``, ``input``,
    ``answers`` and -- for the classification tasks -- ``all_classes``. Read with
    :func:`json.loads` and nothing else: ``datasets`` is not a harness dependency,
    and an evaluation host that has the files should not need a network to use
    them.

    ``all_classes`` travels on the item because ``trec`` and ``lsht`` cannot be
    scored without it; a classification task whose classes went missing would
    score zero everywhere rather than raise, which is the kind of result that
    gets believed.

    ``EvalItem.metric`` is set to the **task name** rather than a metric name,
    because that is what :func:`tq_longbench.metrics.score_prediction` dispatches
    on: the metric, the first-line cut and the class list are all a function of
    the task, and splitting them across two names is how they come apart.
    """
    prompts, budgets = longbench_config(dataset_dir)
    if task not in prompts:
        raise ValueError(f"no prompt template for LongBench task {task!r}; known: {sorted(prompts)}")
    path = local_task_path(task, dataset_dir)
    if path is None:
        raise FileNotFoundError(f"{Path(dataset_dir) / f'{task}.jsonl'} does not exist")
    with path.open(encoding="utf-8") as handle:
        # Rows, not lines: a blank line must neither use up ``limit`` nor shift
        # the ``index`` a record is joined back to its item by.
        rows = (json.loads(line) for line in handle if line.strip())
        for index, row in enumerate(rows):
            if limit is not None and index >= limit:
                return
            yield EvalItem(
                prompt=prompts[task].format(context=row.get("context", ""), input=row.get("input", "")),
                answers=list(row.get("answers", [])),
                metric=task,
                max_new_tokens=budgets.get(task, 64),
                extra={
                    "task": task,
                    "index": index,
                    "length": row.get("length"),
                    "all_classes": row.get("all_classes"),
                    "language": row.get("language"),
                },
            )


def load_longbench(task: str, limit: int | None = None) -> Iterator[EvalItem]:
    """Stream a LongBench task through its own prompt template.

    ``datasets`` is imported here rather than at module scope so that the NIAH
    path -- the one the harness's tests use -- needs nothing installed beyond
    torch.
    """
    name = task.removeprefix("longbench_")
    # Gated on the tasks this path can score, not on the prompt table: the table
    # holds all 21 v1 tasks, :func:`score` knows six of them, and admitting the
    # rest here would fail a whole download later on a KeyError.
    if name not in _TASK_METRICS:
        raise ValueError(f"no prompt template for LongBench task {name!r}; known: {sorted(_TASK_METRICS)}")
    template = LONGBENCH_PROMPTS[name]
    from datasets import load_dataset

    rows = load_dataset("THUDM/LongBench", name, split="test")
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            return
        yield EvalItem(
            prompt=template.format(context=row["context"], input=row["input"]),
            answers=list(row["answers"]),
            metric=_TASK_METRICS[name],
            max_new_tokens=LONGBENCH_MAX_NEW_TOKENS[name],
            extra={"task": name, "index": index, "length": row.get("length")},
        )
