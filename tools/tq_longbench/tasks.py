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

import random
import string
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field

# LongBench's own prompt templates, abbreviated to the tasks this harness runs.
# Verbatim from the suite: a reworded prompt scores differently and would make
# the numbers incomparable to published ones.
LONGBENCH_PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\nStory: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, using a single phrase "
        "if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you "
        "can, using a single phrase or sentence if possible. If the question cannot be answered based "
        'on the information in the article, write "unanswerable". If the question is a yes/no '
        'question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\n'
        "Article: {context}\n\n Answer the question based on the above article as concisely as you "
        "can, using a single phrase or sentence if possible. If the question cannot be answered based "
        'on the information in the article, write "unanswerable". If the question is a yes/no '
        'question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\n'
        "Question: {input}\n\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer and do "
        "not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
}

#: LongBench's own per-task generation budgets.
LONGBENCH_MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "multifieldqa_en": 64,
}

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


METRICS = {
    "f1": f1_score,
    "rouge_l": rouge_l,
    "exact_match": exact_match,
    "substring_match": substring_match,
}

_TASK_METRICS = {
    "narrativeqa": "f1",
    "qasper": "f1",
    "hotpotqa": "f1",
    "2wikimqa": "f1",
    "multifieldqa_en": "f1",
}


def score(prediction: str, answers: list[str], metric: str) -> float:
    """Best score over the references, which is how LongBench aggregates."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose one of {sorted(METRICS)}")
    scorer = METRICS[metric]
    return max((scorer(prediction, answer) for answer in answers), default=0.0)


# -------------------------------------------------------------------- items

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
        metric="substring_match",
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


def load_longbench(task: str, limit: int | None = None) -> Iterator[EvalItem]:
    """Stream a LongBench task through its own prompt template.

    ``datasets`` is imported here rather than at module scope so that the NIAH
    path -- the one the harness's tests use -- needs nothing installed beyond
    torch.
    """
    from datasets import load_dataset

    name = task.removeprefix("longbench_")
    if name not in LONGBENCH_PROMPTS:
        raise ValueError(f"no prompt template for LongBench task {name!r}; known: {sorted(LONGBENCH_PROMPTS)}")
    template = LONGBENCH_PROMPTS[name]
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
