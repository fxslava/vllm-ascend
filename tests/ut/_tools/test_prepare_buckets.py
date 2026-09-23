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
"""The tiered corpus builder, held to the one property the whole thing exists for.

``prepare_buckets`` makes exactly one promise that a length column in the source
data does not: **an item is in a tier because it already had that length**, and
nothing was cut to put it there.  Everything here is a way of checking that the
promise survives the paths that could break it -- a band boundary, a cheap
character filter in front of the expensive encode, an extractive reference that
is not in its own context, and a corpus that is simply not on disk.

The tokenizer is injected (:func:`~tq_longbench.prepare_buckets.partition` takes
``measure``), so none of this needs a checkpoint, transformers or a device: a
stand-in that counts whitespace words is enough to exercise the partitioning,
and a stand-in that counts how many times it was called is what proves the
character pre-filter is actually saving the encode it claims to.

Runs under pytest in CI and under ``python -m unittest`` anywhere.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "tools"))

from tq_longbench.metrics import multiple_choice_score  # noqa: E402
from tq_longbench.prepare_buckets import (  # noqa: E402
    TIERS,
    SourceRow,
    Tier,
    TierReport,
    answer_survives,
    parse_tiers,
    partition,
    plausible_length,
    read_v2_row,
    render_prompt,
    summarise,
    tier_row,
)
from tq_longbench.tasks import load_longbench_jsonl, longbench_config  # noqa: E402


def word_measure(text: str) -> int:
    """A stand-in tokenizer: one token per whitespace-separated word."""
    return len(text.split())


#: What ``qasper``'s own template costs before any context goes into it. The
#: bands below are stated relative to it, because the thing being partitioned is
#: the *rendered* prompt and a band that ignored the wrapper would be a band
#: about a different number than the one the script measures.
_TEMPLATE_WORDS = word_measure(
    render_prompt("qasper", SourceRow(context="", input="what?", answers=[]), longbench_config()[0])
)


class TierBandTest(unittest.TestCase):
    """The bands themselves, which are the definition of the whole artefact."""

    def test_bands_are_the_objective(self):
        self.assertEqual((TIERS["32k"].min_tokens, TIERS["32k"].max_tokens), (8192, 32768))
        self.assertEqual((TIERS["256k"].min_tokens, TIERS["256k"].max_tokens), (65536, 262144))
        self.assertEqual((TIERS["1m"].min_tokens, TIERS["1m"].max_tokens), (524288, 1048576))

    def test_bands_do_not_overlap(self):
        ordered = [TIERS["32k"], TIERS["256k"], TIERS["1m"]]
        for lower, upper in zip(ordered, ordered[1:]):
            self.assertLess(lower.max_tokens, upper.min_tokens)

    def test_admits_is_inclusive_at_both_ends(self):
        tier = TIERS["32k"]
        self.assertTrue(tier.admits(8192))
        self.assertTrue(tier.admits(32768))
        self.assertFalse(tier.admits(8191))
        self.assertFalse(tier.admits(32769))

    def test_parse_tiers_refuses_an_unknown_name(self):
        self.assertEqual(parse_tiers("32k,1m"), ("32k", "1m"))
        with self.assertRaises(ValueError):
            parse_tiers("64k")
        with self.assertRaises(ValueError):
            parse_tiers("")


class PlausibleLengthTest(unittest.TestCase):
    """The character pre-filter: loose, and loose in the safe direction."""

    def test_it_skips_what_cannot_reach_the_floor(self):
        self.assertFalse(plausible_length("word " * 100, TIERS["1m"]))

    def test_it_skips_what_cannot_fit_the_ceiling(self):
        # Twelve characters per token is the most generous the band allows, so
        # anything past that many cannot tokenise down into the tier.
        self.assertFalse(plausible_length("x" * (32768 * 12 + 1), TIERS["32k"]))

    def test_it_keeps_anything_that_could_land(self):
        self.assertTrue(plausible_length("x" * (16384 * 4), TIERS["32k"]))

    def test_the_filter_saves_the_encode(self):
        """The point of the filter is not to be right, it is to not tokenise a book."""
        calls = []

        def counted(text: str) -> int:
            calls.append(len(text))
            return word_measure(text)

        # A band far above the template's own cost, so the short row is out of
        # reach on characters alone and the long one is not.
        tier = Tier(name="t", min_tokens=5000, max_tokens=10000, sources=("qasper",), why="test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_v1_corpus(root / "raw", "qasper", [_row(words=4), _row(words=6000)])
            partition(tier, root / "raw", root / "out", counted, None, False, ["qasper"])
        # The four-word row never reached the tokenizer; the six-thousand-word one did.
        self.assertEqual(len(calls), 1)


class AnswerSurvivesTest(unittest.TestCase):
    """The 'without clipping' half of the objective, which here means 'answerable at all'."""

    def test_an_extractive_reference_must_be_in_the_context(self):
        present = SourceRow(context="the passcode is 894772 and nothing else", input="q", answers=["894772"])
        absent = SourceRow(context="a document about something else", input="q", answers=["894772"])
        self.assertTrue(answer_survives("qasper", present))
        self.assertFalse(answer_survives("qasper", absent))

    def test_normalisation_applies_to_both_sides(self):
        row = SourceRow(context="The Answer, is: Foo Bar.", input="q", answers=["foo bar"])
        self.assertTrue(answer_survives("qasper", row))

    def test_a_written_reference_is_not_judged(self):
        """gov_report references are summaries; requiring them verbatim would empty the tier."""
        row = SourceRow(context="a long report", input="q", answers=["a summary written from the report"])
        self.assertTrue(answer_survives("gov_report", row))
        self.assertTrue(answer_survives("longbench_v2", SourceRow(context="c", input="q", answers=["C"])))


class LongBenchV2Test(unittest.TestCase):
    """v2 is multiple choice, and is stored in v1's shape so the existing loader reads it."""

    def test_the_choices_are_folded_into_input(self):
        row = read_v2_row(
            {
                "context": "the text",
                "question": "which one?",
                "choice_A": "alpha",
                "choice_B": "beta",
                "choice_C": "gamma",
                "choice_D": "delta",
                "answer": "C",
            }
        )
        self.assertEqual(row.answers, ["C"])
        self.assertEqual(row.context, "the text")
        for letter, choice in (("A", "alpha"), ("B", "beta"), ("C", "gamma"), ("D", "delta")):
            self.assertIn(f"({letter}) {choice}", row.input)

    def test_the_rendered_prompt_carries_both_halves(self):
        row = read_v2_row({"context": "CONTEXT", "question": "Q", "choice_A": "a", "answer": "a"})
        text = render_prompt("longbench_v2", row, {})
        self.assertIn("CONTEXT", text)
        self.assertIn("Q", text)

    def test_the_metric_reads_a_letter_out_of_a_sentence(self):
        self.assertEqual(multiple_choice_score("The correct answer is (C).", ["C"]), 1.0)
        self.assertEqual(multiple_choice_score("C", ["C"]), 1.0)
        self.assertEqual(multiple_choice_score("The correct answer is (B).", ["C"]), 0.0)

    def test_the_metric_does_not_read_a_letter_out_of_a_word(self):
        """'Answer' opens with an A, and a prediction naming no choice has not answered."""
        self.assertEqual(multiple_choice_score("Answer: none of these", ["A"]), 0.0)
        self.assertEqual(multiple_choice_score("I cannot tell from the text.", ["A"]), 0.0)


class PartitionTest(unittest.TestCase):
    """The end to end: rows in, a tier directory and a manifest out."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        self.tier = Tier(
            name="test",
            min_tokens=_TEMPLATE_WORDS + 50,
            max_tokens=_TEMPLATE_WORDS + 100,
            sources=("qasper",),
            why="test",
        )

    def test_only_the_rows_inside_the_band_are_written(self):
        rows = [_row(words=10), _row(words=60), _row(words=200), _row(words=80)]
        _write_v1_corpus(self.root / "raw", "qasper", rows)
        report = partition(self.tier, self.root / "raw", self.root / "out", word_measure, None, False, ["qasper"])

        self.assertEqual(report.kept, 2)
        written = _read_jsonl(self.root / "out" / "test" / "qasper.jsonl")
        for item in written:
            self.assertTrue(self.tier.admits(item["tq"]["prompt_tokens"]))

    def test_nothing_written_is_shorter_than_its_source(self):
        """The promise: a tier row is the whole item, not a cut one."""
        rows = [_row(words=60, marker="UNIQUE-MIDDLE")]
        _write_v1_corpus(self.root / "raw", "qasper", rows)
        partition(self.tier, self.root / "raw", self.root / "out", word_measure, None, False, ["qasper"])

        written = _read_jsonl(self.root / "out" / "test" / "qasper.jsonl")[0]
        self.assertEqual(written["context"], rows[0]["context"])
        self.assertIn("UNIQUE-MIDDLE", written["context"])

    def test_a_row_traces_back_to_its_source_line(self):
        _write_v1_corpus(self.root / "raw", "qasper", [_row(words=10), _row(words=60)])
        partition(self.tier, self.root / "raw", self.root / "out", word_measure, None, False, ["qasper"])

        written = _read_jsonl(self.root / "out" / "test" / "qasper.jsonl")[0]
        self.assertEqual(written["tq"]["source_index"], 1)
        self.assertEqual(written["tq"]["tier"], "test")

    def test_the_limit_caps_a_task(self):
        _write_v1_corpus(self.root / "raw", "qasper", [_row(words=60) for _ in range(5)])
        report = partition(self.tier, self.root / "raw", self.root / "out", word_measure, 2, False, ["qasper"])
        self.assertEqual(report.kept, 2)

    def test_a_task_that_admits_nothing_leaves_no_file(self):
        """An empty file would read as a task that scored zero rather than one with no items."""
        _write_v1_corpus(self.root / "raw", "qasper", [_row(words=10)])
        partition(self.tier, self.root / "raw", self.root / "out", word_measure, None, False, ["qasper"])
        self.assertFalse((self.root / "out" / "test" / "qasper.jsonl").exists())

    def test_a_missing_corpus_is_recorded_not_raised(self):
        (self.root / "raw").mkdir(parents=True)
        report = partition(self.tier, self.root / "raw", self.root / "out", word_measure, None, False, [])
        self.assertEqual(report.kept, 0)
        self.assertIn("qasper: corpus not present", report.rejected)

    def test_the_manifest_describes_what_was_written(self):
        _write_v1_corpus(self.root / "raw", "qasper", [_row(words=60), _row(words=90), _row(words=10)])
        partition(self.tier, self.root / "raw", self.root / "out", word_measure, None, False, ["qasper"])

        manifest = json.loads((self.root / "out" / "test" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["items"], 2)
        self.assertEqual(manifest["tasks"]["qasper"]["items"], 2)
        self.assertLessEqual(manifest["tasks"]["qasper"]["min_tokens"], manifest["tasks"]["qasper"]["max_tokens"])
        self.assertEqual(manifest["min_tokens"], self.tier.min_tokens)
        self.assertEqual(manifest["rejected"]["outside the band (measured)"], 1)
        self.assertEqual(manifest["tasks"]["qasper"]["metric"], "qa_f1_score")

    def test_require_answer_drops_an_unanswerable_item(self):
        rows = [_row(words=60, answer="not in the text at all")]
        _write_v1_corpus(self.root / "raw", "qasper", rows)
        kept = partition(self.tier, self.root / "raw", self.root / "out", word_measure, None, False, ["qasper"])
        dropped = partition(self.tier, self.root / "raw", self.root / "out2", word_measure, None, True, ["qasper"])
        self.assertEqual(kept.kept, 1)
        self.assertEqual(dropped.kept, 0)
        self.assertIn("reference not found in context", dropped.rejected)

    def test_a_written_row_keeps_the_suite_shape(self):
        row = tier_row("qasper", SourceRow(context="c", input="i", answers=["a"]), 60, 3, self.tier)
        for field in ("context", "input", "answers", "all_classes", "length"):
            self.assertIn(field, row)
        self.assertEqual(row["length"], row["tq"]["prompt_tokens"])

    def test_the_written_rows_load_back_through_the_existing_loader(self):
        """The reason a tier row is stored in v1's shape: nothing new has to read it."""
        _write_v1_corpus(self.root / "raw", "qasper", [_row(words=60, marker="haystack", answer="haystack")])
        partition(self.tier, self.root / "raw", self.root / "out", word_measure, None, False, ["qasper"])

        items = list(load_longbench_jsonl("qasper", self.root / "out" / "test"))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].answers, ["haystack"])
        self.assertIn("haystack", items[0].prompt)


class SummariseTest(unittest.TestCase):
    """The table, including the tier that admitted nothing -- which is a real outcome."""

    def test_an_empty_tier_says_so_rather_than_vanishing(self):
        full = TierReport(TIERS["32k"])
        full.record("qasper", 9000)
        full.record("qasper", 30000)
        empty = TierReport(TIERS["1m"])
        table = summarise([full, empty], Path("datasets/tiered"))

        self.assertIn("qasper", table)
        self.assertIn("nothing admitted", table)
        self.assertIn("No item was truncated", table)

    def test_what_was_turned_away_is_on_the_table(self):
        report = TierReport(TIERS["32k"])
        report.record("qasper", 9000)
        report.reject("outside the band (measured)")
        self.assertIn("1 outside the band (measured)", summarise([report], Path("out")))


def _row(words: int, marker: str = "filler", answer: str = "filler") -> dict:
    """A v1 source row whose *rendered* prompt is roughly ``words`` words long.

    Roughly, because qasper's template wraps it in instructions; the tests that
    care about the exact band use margins wider than the wrapper.
    """
    return {"context": " ".join([marker] * words), "input": "what?", "answers": [answer]}


def _write_v1_corpus(directory: Path, task: str, rows: list) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _read_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    unittest.main()
