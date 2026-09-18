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
"""LongBench's own scoring, written out rather than imported.

Transcribed from ``THUDM/LongBench``'s ``metrics.py`` and ``eval.py`` so that a
number this harness prints can be compared with a published one. The suite's
scorer does three things per item, and all three are here because leaving any of
them out changes the score:

1. for the few-shot tasks (:data:`FIRST_LINE_TASKS`) the prediction is cut to its
   first non-empty line, because those prompts are ``example\\nexample\\nquery``
   and a model that keeps going invents the next example;
2. each metric is taken against **every** reference and the best one kept;
3. the mean over items is multiplied by 100 and rounded to two places.

Written out rather than imported for the reason the rest of the harness is: the
dependency list is torch, transformers and nothing that has to be resolved on an
air-gapped NPU host. That costs fidelity in exactly two places, both named below
and both asserted in :mod:`tests.ut._tools.test_longbench_metrics`:

* **ROUGE-L.** The suite calls the ``rouge`` package. Its ROUGE-L for a single
  sentence pair is the F-measure of the longest common subsequence with
  ``alpha=0.5``, which is plain LCS F1 -- :func:`rouge_l_score` computes exactly
  that. Where the package differs is on multi-sentence input, which it splits and
  averages; a LongBench summary reference is one block of text, so the two agree
  on the suite's own data and can differ on text that is not.
* **Chinese word segmentation.** The suite calls ``jieba``. Where ``jieba`` is
  importable this uses it and matches; where it is not, the fallback is
  character-level, which is what the objective asked for and is *not* the same
  number. :func:`segmentation_backend` says which one a run used, and the CLI
  prints it, so a Chinese score is never ambiguous about what produced it.

Everything else -- ``normalize_answer``'s order of operations, the
classification tie-break, the code metric's choice of line, the retrieval
metric's digit counting -- is the published behaviour including its quirks.
"""

from __future__ import annotations

import difflib
import re
import string
from collections.abc import Callable, Iterable, Sequence

#: The few-shot and classification tasks, whose prompts are worked examples. The
#: suite keeps only the first non-empty line of the prediction for these: the
#: model has been shown ``question\\nanswer`` pairs and carries on producing them.
FIRST_LINE_TASKS = frozenset({"trec", "triviaqa", "samsum", "lsht"})

#: Punctuation ``normalize_zh_answer`` strips on top of ASCII's, verbatim from the suite.
_CHINESE_PUNCTUATION = (
    "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､"
    "、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
)

_ASCII_PUNCTUATION = frozenset(string.punctuation)


def _is_punctuation(character: str) -> bool:
    """ASCII punctuation, the suite's CJK list, or anything Unicode calls punctuation.

    The third clause is a deliberate widening of the published set, and it is here
    because the published set is sixty hand-typed codepoints and this transcription
    of it is missing at least one that matters: it carries the halfwidth full stop
    ``U+FF61`` but not the fullwidth ``U+3002``, which is the one Chinese text is
    actually written with. Rather than guess at which others are missing, anything
    in a Unicode punctuation category goes too.

    This can differ from the suite, in one direction only: a mark the published
    list forgot is stripped here and kept there, which can only raise a score by
    removing a token neither side meant to compare. Named here rather than left to
    be discovered from a score that does not reproduce.
    """
    import unicodedata

    return (
        character in _ASCII_PUNCTUATION
        or character in _CHINESE_PUNCTUATION
        or unicodedata.category(character).startswith("P")
    )


#: SQuAD normalisation drops these wherever they stand alone.
_ARTICLES = frozenset(("a", "an", "the"))


# ------------------------------------------------------------- normalisation


def normalize_answer(text: str) -> str:
    """SQuAD normalisation: lowercase, strip punctuation, strip articles, collapse space.

    The order matters and is the suite's: punctuation goes before articles, so
    ``"the,"`` becomes ``"the"`` and is then dropped, where the other order would
    leave it.
    """
    lowered = text.lower()
    stripped = "".join(character for character in lowered if character not in _ASCII_PUNCTUATION)
    return " ".join(word for word in stripped.split() if word not in _ARTICLES)


def normalize_zh_answer(text: str) -> str:
    """The Chinese normalisation: lowercase, strip both punctuation sets, remove *all* space.

    Whitespace is removed rather than collapsed, because Chinese is not written
    with it and a space in a prediction is an artefact of the model rather than a
    token boundary.
    """
    lowered = text.lower()
    return "".join(character for character in lowered if not _is_punctuation(character) and not character.isspace())


def segmentation_backend() -> str:
    """``"jieba"`` where the suite's segmenter is importable, else ``"character"``.

    Printed by the CLI next to any Chinese score. The two are different metrics,
    not two implementations of one, and a table that did not say which it used
    would invite exactly the comparison that cannot be made.
    """
    try:
        import jieba  # noqa: F401
    except ImportError:
        return "character"
    return "jieba"


def _zh_tokens(text: str) -> list[str]:
    """Chinese tokens: ``jieba``'s words where it is installed, single characters otherwise."""
    if segmentation_backend() == "jieba":
        import jieba

        pieces = jieba.cut(text, cut_all=False)
    else:
        pieces = text
    return [token for token in (normalize_zh_answer(piece) for piece in pieces) if token]


# ------------------------------------------------------------------ measures


def _f1(predicted: Sequence[str], reference: Sequence[str]) -> float:
    """Token-overlap F1 over two token lists, counting multiplicity."""
    from collections import Counter

    common = Counter(predicted) & Counter(reference)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def _longest_common_subsequence(left: Sequence[str], right: Sequence[str]) -> int:
    """Length of the LCS, by the usual table. Six lines, and it saves a dependency."""
    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, a in enumerate(left, start=1):
        for j, b in enumerate(right, start=1):
            table[i][j] = table[i - 1][j - 1] + 1 if a == b else max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def _best(prediction: str, ground_truths: Iterable[str], measure: Callable[[str, str], float]) -> float:
    """The suite aggregates over references by taking the best, not the mean."""
    return max((measure(prediction, reference) for reference in ground_truths), default=0.0)


# ------------------------------------------------------------------- metrics


def qa_f1_score(prediction: str, ground_truths: Iterable[str]) -> float:
    """Token-level F1 after SQuAD normalisation: the single- and multi-doc QA metric."""

    def measure(predicted: str, reference: str) -> float:
        return _f1(normalize_answer(predicted).split(), normalize_answer(reference).split())

    return _best(prediction, ground_truths, measure)


def qa_f1_zh_score(prediction: str, ground_truths: Iterable[str]) -> float:
    """The same, over Chinese tokens. See :func:`segmentation_backend` for which tokens."""

    def measure(predicted: str, reference: str) -> float:
        return _f1(_zh_tokens(predicted), _zh_tokens(reference))

    return _best(prediction, ground_truths, measure)


def rouge_l_score(prediction: str, ground_truths: Iterable[str]) -> float:
    """ROUGE-L F1: the F-measure of the longest common subsequence, ``alpha=0.5``."""

    def measure(predicted: str, reference: str) -> float:
        left, right = predicted.split(), reference.split()
        if not left or not right:
            return 0.0
        lcs = _longest_common_subsequence(left, right)
        if lcs == 0:
            return 0.0
        precision, recall = lcs / len(left), lcs / len(right)
        return 2 * precision * recall / (precision + recall)

    return _best(prediction, ground_truths, measure)


def rouge_l_zh_score(prediction: str, ground_truths: Iterable[str]) -> float:
    """ROUGE-L over Chinese tokens, which the suite reaches by segmenting and re-joining."""

    def measure(predicted: str, reference: str) -> float:
        return rouge_l_score(" ".join(_zh_tokens(predicted)), [" ".join(_zh_tokens(reference))])

    return _best(prediction, ground_truths, measure)


def exact_match_score(prediction: str, ground_truths: Iterable[str]) -> float:
    """Normalised exact match: 1.0 when the whole normalised prediction is a reference."""

    def measure(predicted: str, reference: str) -> float:
        return float(normalize_answer(predicted) == normalize_answer(reference))

    return _best(prediction, ground_truths, measure)


def classification_score(prediction: str, ground_truths: Iterable[str], all_classes: Sequence[str] | None) -> float:
    """Label extraction: which of ``all_classes`` the prediction names, split over ties.

    The suite's rule, quirks included. Every class that appears anywhere in the
    prediction is a candidate; a candidate that is a *substring* of the reference
    without being it is dropped, so ``"sport"`` does not count against
    ``"sports"``; and a correct answer among ``n`` candidates scores ``1/n``
    rather than 1, so naming every label scores near zero instead of always
    winning.
    """
    if not all_classes:
        return 0.0

    def measure(predicted: str, reference: str) -> float:
        candidates = [name for name in all_classes if name in predicted]
        kept = [name for name in candidates if not (name in reference and name != reference)]
        return 1.0 / len(kept) if reference in kept else 0.0

    return _best(prediction, ground_truths, measure)


#: One departure in :func:`classification_score`, recorded because it is a
#: departure: the suite drops its substring candidates with ``list.remove`` while
#: iterating that same list, which skips the element after each removal. This
#: keeps every one it meant to. The two can only differ when two classes in a row
#: are both proper substrings of the reference, which no v1 label set contains --
#: ``trec``'s coarse types and ``lsht``'s categories are mutually non-nesting.
_CLASSIFICATION_KEEPS_EVERY_DROP = True


def code_similarity_score(prediction: str, ground_truths: Iterable[str]) -> float:
    """Edit-distance similarity on the prediction's first line of actual code.

    Comments and fenced markdown are skipped -- the suite takes the first line
    carrying no backtick, ``#`` or ``//`` -- because a model that explains itself
    before completing the line would otherwise be scored on the explanation.

    ``difflib``'s ratio is the same quantity ``fuzzywuzzy.fuzz.ratio`` reports,
    ``2M/T`` over the matching characters, on a 0-1 scale rather than 0-100.
    """
    line = _first_code_line(prediction)

    def measure(predicted: str, reference: str) -> float:
        return difflib.SequenceMatcher(None, predicted, reference).ratio()

    return _best(line, ground_truths, measure)


def _first_code_line(prediction: str) -> str:
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            return line
    return ""


def retrieval_score(prediction: str, ground_truths: Iterable[str]) -> float:
    """Passage retrieval: of the numbers the prediction names, how many are the right one.

    Not "did it say the right number" but the fraction of the numbers it said
    that were right, so a prediction listing every paragraph scores ``1/n``.
    """
    return _retrieval(prediction, ground_truths, r"Paragraph (\d+)")


def retrieval_zh_score(prediction: str, ground_truths: Iterable[str]) -> float:
    """The Chinese passage-retrieval task, whose references say 段落 rather than Paragraph."""
    return _retrieval(prediction, ground_truths, r"段落(\d+)")


def _retrieval(prediction: str, ground_truths: Iterable[str], pattern: str) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0

    def measure(_predicted: str, reference: str) -> float:
        wanted = re.findall(pattern, reference)
        if not wanted:
            return 0.0
        return sum(number == wanted[0] for number in numbers) / len(numbers)

    return _best(prediction, ground_truths, measure)


def count_score(prediction: str, ground_truths: Iterable[str]) -> float:
    """``passage_count``: the same digit counting, against a bare number reference."""
    return _retrieval(prediction, ground_truths, r"(\d+)")


# ------------------------------------------------------------ the task table


def _needs_classes(metric: Callable) -> bool:
    return metric is classification_score


#: Every LongBench v1 task and the metric the suite scores it with, verbatim from
#: ``dataset2metric``. Two of these differ from the objective's summary of them
#: and the suite wins, because the point of the file is comparability:
#:
#: * ``triviaqa`` is **F1**, not exact match. It is a few-shot task, so its
#:   prediction is cut to the first line, which is what makes F1 behave like one.
#: * ``passage_retrieval_zh`` is the **retrieval** metric -- the fraction of the
#:   numbers named that were right -- not a Chinese F1 or ROUGE.
#:
#: :func:`exact_match_score` is implemented and tested regardless; no v1 task uses it.
TASK_METRICS: dict[str, Callable] = {
    # Single-document QA
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    # Multi-document QA
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_l_zh_score,
    # Summarisation
    "gov_report": rouge_l_score,
    "qmsum": rouge_l_score,
    "multi_news": rouge_l_score,
    "vcsum": rouge_l_zh_score,
    # Few-shot learning
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_l_score,
    "lsht": classification_score,
    # Synthetic
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "passage_retrieval_zh": retrieval_zh_score,
    # Code
    "lcc": code_similarity_score,
    "repobench-p": code_similarity_score,
}

#: What to call each metric in a results table.
_METRIC_LABELS = {
    qa_f1_score: "F1",
    qa_f1_zh_score: "F1-zh",
    rouge_l_score: "ROUGE-L",
    rouge_l_zh_score: "ROUGE-L-zh",
    exact_match_score: "EM",
    classification_score: "Accuracy",
    code_similarity_score: "Edit-Sim",
    retrieval_score: "Retrieval",
    retrieval_zh_score: "Retrieval-zh",
    count_score: "Count",
}


def metric_label(task: str) -> str:
    """The column heading for ``task``'s metric."""
    return _METRIC_LABELS.get(TASK_METRICS.get(task), "F1")


def score_prediction(
    task: str,
    prediction: str,
    ground_truths: Sequence[str],
    all_classes: Sequence[str] | None = None,
) -> float:
    """One item's score in ``[0, 1]``, the way ``eval.py`` computes it.

    The first-line cut for the few-shot tasks happens here rather than in the
    caller, because it is part of the metric: the same prediction scores
    differently on ``trec`` and on ``qasper``, and a caller that forgot would
    quietly report a number nobody else's harness would reproduce.
    """
    metric = TASK_METRICS.get(task)
    if metric is None:
        raise ValueError(f"no LongBench metric for task {task!r}; known: {sorted(TASK_METRICS)}")
    if task in FIRST_LINE_TASKS:
        prediction = prediction.lstrip("\n").split("\n")[0]
    if _needs_classes(metric):
        return metric(prediction, ground_truths, all_classes)
    return metric(prediction, ground_truths)


def as_percentage(scores: Iterable[float]) -> float:
    """The suite's aggregate: the mean, times 100, to two places."""
    values = list(scores)
    if not values:
        return 0.0
    return round(100 * sum(values) / len(values), 2)
