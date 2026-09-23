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
"""LongBench's scoring, held to values worked out by hand from its definitions.

Not to another implementation of the same thing: the point of
:mod:`tq_longbench.metrics` is that a number here is comparable with a published
one, and a test that compared it with a second copy of the same code would pass
whatever both of them did. Every expected value below is arithmetic that can be
checked on paper, and the cases are the ones where LongBench's scorer does
something a reasonable person would not have guessed -- the first-line cut, the
classification tie-break, the code metric's choice of line, retrieval counting
*all* the digits a prediction names.

Runs under pytest and under ``python -m unittest`` anywhere: nothing here needs
torch, a device, or the corpus.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Appended, never prepended: tools/bisect/ shadows the standard library's bisect.
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "tools"))

from tq_longbench.metrics import (  # noqa: E402
    FIRST_LINE_TASKS,
    TASK_METRICS,
    as_percentage,
    classification_score,
    code_similarity_score,
    count_score,
    exact_match_score,
    metric_label,
    normalize_answer,
    normalize_zh_answer,
    qa_f1_score,
    qa_f1_zh_score,
    retrieval_score,
    retrieval_zh_score,
    rouge_l_score,
    rouge_l_zh_score,
    score_prediction,
    segmentation_backend,
)


class TestNormalisation(unittest.TestCase):
    """SQuAD normalisation, in the order the suite applies it."""

    def test_it_lowercases_strips_punctuation_and_articles(self):
        self.assertEqual(normalize_answer("The Eiffel Tower, in Paris!"), "eiffel tower in paris")

    def test_punctuation_goes_before_articles_so_a_comma_cannot_save_one(self):
        """``"the,"`` has to become ``"the"`` and then vanish, not survive as ``"the,"``."""
        self.assertEqual(normalize_answer("the, dog"), "dog")
        self.assertEqual(normalize_answer("A. AN, THE!"), "")

    def test_an_article_inside_a_word_is_left_alone(self):
        self.assertEqual(normalize_answer("theatre another"), "theatre another")

    def test_whitespace_is_collapsed_not_removed(self):
        self.assertEqual(normalize_answer("  red   \n dog "), "red dog")

    def test_the_chinese_normalisation_removes_whitespace_entirely(self):
        self.assertEqual(normalize_zh_answer("你好， 世界！"), "你好世界")
        self.assertEqual(normalize_zh_answer("A B\tC"), "abc")


class TestQaF1(unittest.TestCase):
    """The metric for six of the twenty-one tasks, so the one worth pinning hardest."""

    def test_an_exact_answer_scores_one_through_the_normalisation(self):
        self.assertEqual(qa_f1_score("The Eiffel Tower", ["eiffel tower"]), 1.0)

    def test_a_partial_answer_is_the_harmonic_mean_of_precision_and_recall(self):
        """``"the big red dog"`` -> [big, red, dog]; ``"a red dog"`` -> [red, dog].

        Two of three predicted tokens overlap and two of two reference tokens do,
        so precision 2/3, recall 1, F1 = 2(2/3)/(2/3 + 1) = 0.8.
        """
        self.assertAlmostEqual(qa_f1_score("the big red dog", ["a red dog"]), 0.8)

    def test_repeated_tokens_count_only_as_often_as_the_reference_has_them(self):
        """``"dog dog dog"`` against ``"dog"``: overlap 1, precision 1/3, recall 1, F1 0.5."""
        self.assertAlmostEqual(qa_f1_score("dog dog dog", ["dog"]), 0.5)

    def test_no_overlap_scores_zero(self):
        self.assertEqual(qa_f1_score("nothing at all", ["red dog"]), 0.0)

    def test_the_best_reference_wins_rather_than_the_mean(self):
        """LongBench takes the max over answers; a mean would punish having alternatives."""
        self.assertEqual(qa_f1_score("red dog", ["blue cat", "red dog"]), 1.0)

    def test_an_empty_prediction_scores_zero_without_dividing_by_it(self):
        self.assertEqual(qa_f1_score("", ["red dog"]), 0.0)
        self.assertEqual(qa_f1_score("red dog", []), 0.0)


class TestRougeL(unittest.TestCase):
    """The summarisation metric: F1 of the longest common subsequence."""

    def test_identical_text_scores_one(self):
        self.assertEqual(rouge_l_score("a b c d", ["a b c d"]), 1.0)

    def test_it_is_a_subsequence_not_a_substring(self):
        """``"a x b y c"`` and ``"a b c"`` share the subsequence a,b,c though not contiguously.

        LCS 3 of 5 predicted and 3 of 3 reference: precision 0.6, recall 1,
        F1 = 2(0.6)/(1.6) = 0.75.
        """
        self.assertAlmostEqual(rouge_l_score("a x b y c", ["a b c"]), 0.75)

    def test_order_matters_where_it_does_not_for_f1(self):
        """The distinction that makes ROUGE-L worth having: F1 would call this perfect."""
        self.assertEqual(qa_f1_score("c b a", ["a b c"]), 1.0)
        self.assertAlmostEqual(rouge_l_score("c b a", ["a b c"]), 1 / 3)

    def test_no_common_subsequence_scores_zero(self):
        self.assertEqual(rouge_l_score("x y z", ["a b c"]), 0.0)

    def test_an_empty_side_scores_zero(self):
        self.assertEqual(rouge_l_score("", ["a b"]), 0.0)
        self.assertEqual(rouge_l_score("a b", [""]), 0.0)


class TestExactMatch(unittest.TestCase):
    def test_it_compares_the_whole_normalised_string(self):
        self.assertEqual(exact_match_score("The, Answer.", ["answer"]), 1.0)
        self.assertEqual(exact_match_score("the answer is", ["answer"]), 0.0)

    def test_the_best_reference_wins(self):
        self.assertEqual(exact_match_score("yes", ["no", "yes"]), 1.0)


class TestClassification(unittest.TestCase):
    """Label extraction, with the tie-break and the substring rule the suite has."""

    CLASSES = ("sports", "politics", "sport")

    def test_naming_one_label_correctly_scores_one(self):
        self.assertEqual(classification_score("this is politics", ["politics"], self.CLASSES), 1.0)

    def test_naming_several_splits_the_credit(self):
        """A prediction hedging over n labels scores 1/n, so listing them all is near zero."""
        prediction = "could be politics"
        self.assertAlmostEqual(classification_score(prediction, ["politics"], self.CLASSES), 1.0)
        # "politics or sports" names three of these classes, not two: "sport" is
        # inside "sports". The substring rule drops a candidate only when it is
        # inside the *reference*, which "sport" is not, so the credit splits three
        # ways. That is the suite's arithmetic, and it is why the rule is easy to
        # misread as "drop near-duplicates".
        both = "politics or sports"
        self.assertAlmostEqual(classification_score(both, ["politics"], self.CLASSES), 1 / 3)

    def test_a_label_that_is_a_substring_of_the_reference_is_not_a_rival(self):
        """``"sport"`` appears inside ``"sports"``; counting it would halve a correct answer."""
        self.assertEqual(classification_score("sports", ["sports"], self.CLASSES), 1.0)

    def test_a_wrong_label_scores_zero(self):
        self.assertEqual(classification_score("politics", ["sports"], self.CLASSES), 0.0)

    def test_no_class_list_scores_zero_rather_than_raising(self):
        self.assertEqual(classification_score("sports", ["sports"], None), 0.0)
        self.assertEqual(classification_score("sports", ["sports"], []), 0.0)


class TestCodeSimilarity(unittest.TestCase):
    """Edit similarity on the first line of the prediction that is actually code."""

    def test_an_exact_line_scores_one(self):
        self.assertEqual(code_similarity_score("return a + b", ["return a + b"]), 1.0)

    def test_comments_and_fences_are_skipped_to_reach_the_code(self):
        """A model that explains itself first would otherwise be scored on the explanation."""
        for preamble in ("# here is the line\n", "```python\n", "// the completion\n"):
            with self.subTest(preamble=preamble.strip()):
                self.assertEqual(code_similarity_score(preamble + "return a + b", ["return a + b"]), 1.0)

    def test_a_near_miss_scores_between_zero_and_one(self):
        value = code_similarity_score("return a - b", ["return a + b"])
        self.assertGreater(value, 0.8)
        self.assertLess(value, 1.0)

    def test_a_prediction_of_only_comments_has_no_code_line(self):
        self.assertEqual(code_similarity_score("# nothing\n// nothing", ["return a + b"]), 0.0)


class TestRetrievalAndCount(unittest.TestCase):
    """The synthetic tasks, which count every number a prediction names."""

    def test_the_only_number_being_right_scores_one(self):
        self.assertEqual(retrieval_score("Paragraph 3", ["Paragraph 3 is the answer"]), 1.0)

    def test_naming_extra_numbers_dilutes_the_score(self):
        """Not "did it say 3" but "of what it said, how much was 3"."""
        self.assertAlmostEqual(retrieval_score("Paragraph 3 or 5", ["Paragraph 3"]), 0.5)
        self.assertAlmostEqual(retrieval_score("1 2 3 4", ["Paragraph 3"]), 0.25)

    def test_naming_no_number_scores_zero(self):
        self.assertEqual(retrieval_score("the first one", ["Paragraph 3"]), 0.0)

    def test_the_chinese_variant_reads_its_own_reference_format(self):
        self.assertEqual(retrieval_zh_score("段落4", ["答案是段落4"]), 1.0)
        self.assertEqual(retrieval_zh_score("段落4", ["Paragraph 4"]), 0.0)

    def test_count_reads_a_bare_number(self):
        self.assertEqual(count_score("5", ["5"]), 1.0)
        self.assertAlmostEqual(count_score("4 or 5", ["5"]), 0.5)


class TestChineseTokenisation(unittest.TestCase):
    """The one place this deliberately may not match the suite, so it says which it did."""

    def test_the_backend_is_reported_and_is_one_of_two(self):
        self.assertIn(segmentation_backend(), ("jieba", "character"))

    def test_identical_chinese_text_scores_one_either_way(self):
        self.assertEqual(qa_f1_zh_score("北京是中国的首都", ["北京是中国的首都"]), 1.0)
        self.assertEqual(rouge_l_zh_score("北京是中国的首都", ["北京是中国的首都"]), 1.0)

    def test_punctuation_does_not_change_a_chinese_score(self):
        """Including the fullwidth full stop the published list happens not to carry.

        The suite's CJK punctuation set has the halfwidth ``U+FF61`` and not the
        fullwidth ``U+3002`` that Chinese is actually written with, so this module
        widens the set to Unicode's punctuation categories. That widening is what
        this case is really pinning.
        """
        self.assertEqual(qa_f1_zh_score("北京，是首都。", ["北京是首都"]), 1.0)
        self.assertEqual(normalize_zh_answer("北京。"), "北京")
        self.assertEqual(normalize_zh_answer("北京｡"), "北京")

    def test_disjoint_chinese_text_scores_zero(self):
        self.assertEqual(qa_f1_zh_score("上海", ["广州"]), 0.0)


class TestScorerBehaviour(unittest.TestCase):
    """``score_prediction``: the parts of ``eval.py`` that are not the metric itself."""

    def test_a_few_shot_task_is_scored_on_its_first_line_only(self):
        """The same prediction, two tasks, two scores -- and that is the suite's doing.

        A few-shot prompt is worked examples, so a model that answers and then
        invents the next question would be scored on the invention too.
        """
        prediction = "Paris\nQuestion: what is the capital of Italy?\nAnswer: Rome"
        self.assertEqual(score_prediction("triviaqa", prediction, ["Paris"]), 1.0)
        self.assertLess(score_prediction("qasper", prediction, ["Paris"]), 1.0)

    def test_the_leading_newlines_are_stripped_before_the_first_line_is_taken(self):
        self.assertEqual(score_prediction("triviaqa", "\n\nParis\nand more", ["Paris"]), 1.0)

    def test_every_first_line_task_is_a_task_the_table_knows(self):
        self.assertTrue(set(TASK_METRICS) >= FIRST_LINE_TASKS)

    def test_a_classification_task_is_given_its_classes(self):
        self.assertEqual(score_prediction("trec", "sports", ["sports"], ["sports", "politics"]), 1.0)
        self.assertEqual(score_prediction("trec", "sports", ["sports"], None), 0.0)

    def test_an_unknown_task_is_refused_by_name(self):
        with self.assertRaisesRegex(ValueError, "no LongBench metric for task"):
            score_prediction("not_a_task", "x", ["x"])

    def test_the_two_documented_departures_from_the_objective_are_the_suites_own(self):
        """triviaqa is F1 and passage_retrieval_zh is the retrieval metric.

        The objective's summary said exact match and a Chinese F1. Where the two
        disagree the suite wins, because a score that is not the suite's is not
        comparable with a published one -- which is the whole reason this module
        exists. Pinned so the choice is visible rather than buried.
        """
        self.assertIs(TASK_METRICS["triviaqa"], qa_f1_score)
        self.assertIs(TASK_METRICS["passage_retrieval_zh"], retrieval_zh_score)

    #: The corpora that are deliberately in the table but are not LongBench v1,
    #: and so are not part of the "exactly the published 21" check below. They
    #: are the 256k and 1M context tiers -- see ``prepare_buckets.py`` -- and a
    #: score on either is not comparable with a published v1 number.
    NON_V1_TASKS = {"longbench_v2", "infinitebench_qa"}

    def test_every_v1_task_has_a_metric_and_a_label(self):
        expected = {
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
            "hotpotqa", "2wikimqa", "musique", "dureader",
            "gov_report", "qmsum", "multi_news", "vcsum",
            "trec", "triviaqa", "samsum", "lsht",
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
            "lcc", "repobench-p",
        }  # fmt: skip
        self.assertEqual(set(TASK_METRICS) - self.NON_V1_TASKS, expected)
        for task in expected | self.NON_V1_TASKS:
            with self.subTest(task=task):
                self.assertTrue(metric_label(task))

    def test_the_labels_name_the_metric_the_task_uses(self):
        self.assertEqual(metric_label("narrativeqa"), "F1")
        self.assertEqual(metric_label("gov_report"), "ROUGE-L")
        self.assertEqual(metric_label("trec"), "Accuracy")
        self.assertEqual(metric_label("lcc"), "Edit-Sim")
        self.assertEqual(metric_label("vcsum"), "ROUGE-L-zh")


class TestAggregate(unittest.TestCase):
    def test_the_aggregate_is_a_percentage_to_two_places(self):
        self.assertEqual(as_percentage([1.0, 0.5, 0.0]), 50.0)
        self.assertEqual(as_percentage([1.0, 1.0]), 100.0)
        self.assertEqual(as_percentage([1 / 3]), 33.33)

    def test_no_items_is_zero_rather_than_a_division(self):
        self.assertEqual(as_percentage([]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
