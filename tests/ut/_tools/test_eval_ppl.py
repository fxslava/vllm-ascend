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
"""Perplexity, held to the one property chunking it could break.

``sequence_perplexity`` walks a sequence in pieces because a whole sequence of
logits does not fit, and the only way that goes wrong quietly is at the seams:
a chunk boundary that scores its last token twice, or not at all, moves the
number by a fraction of a percent and looks like a result.  So the test is an
identity rather than a threshold -- **the chunked answer is the one-shot answer**
-- checked at three chunk sizes against the formula written out in full.

Two layers of random weights on the CPU, through the runner's own
``random_weights`` path: no checkpoint, no transformers, no device.  The model
is nonsense and that is fine, because nothing here is a claim about a model.

Runs under pytest in CI and under ``python -m unittest`` anywhere.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "tools"))

from tq_longbench.engine import RunnerConfig, StandaloneModelRunner  # noqa: E402
from tq_longbench.eval_ppl import PerplexityReading, sequence_perplexity  # noqa: E402
from tq_longbench.layers import ForwardBatch, ModelShape  # noqa: E402

#: Two layers, D=64 because Pi needs a power of two, and a tiny untied vocabulary.
TINY = ModelShape(
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    head_size=64,
    hidden_size=256,
    intermediate_size=192,
    vocab_size=320,
    rms_norm_eps=1e-6,
    rope_theta=5.0e6,
    tie_word_embeddings=False,
    attn_output_gate=False,
    qk_norm=False,
    qkv_bias=True,
)

TOKENS = 300


def build_runner(chunk_size: int, backend: str = "dense_reference", mode: str = "dense_staging"):
    config = RunnerConfig(
        model_path="<random weights>",
        backend=backend,
        prefill_mode=mode,
        max_seq_len=1024,
        chunk_size=chunk_size,
        block_size=128,
        dtype=torch.float32,
        device="cpu",
        random_weights=True,
        seed=7,
    )
    return StandaloneModelRunner(config, model_shape=TINY)


@torch.inference_mode()
def reference_perplexity(runner, ids: torch.Tensor) -> float:
    """The formula in full, in one forward: ``exp(cross_entropy(logits[:-1], ids[1:]))``.

    Written out here rather than imported, so that the thing under test is
    checked against the definition and not against itself.
    """
    batch = ForwardBatch(
        positions=torch.arange(ids.numel(), dtype=torch.int64),
        slots=runner.decode_backend.cache.slot_mapping(0, int(ids.numel())),
        prefix_end=int(ids.numel()),
        is_decode=False,
    )
    logits = runner.model(ids, batch, runner.write_backends, runner.prefill_backend, last_token_only=False)
    return float(torch.exp(torch.nn.functional.cross_entropy(logits[:-1].float(), ids[1:])))


class ReadingTest(unittest.TestCase):
    """The arithmetic between a summed loss and a perplexity."""

    def test_loss_becomes_perplexity(self):
        self.assertAlmostEqual(PerplexityReading(4, 4 * math.log(3.0)).perplexity, 3.0)

    def test_nothing_scored_is_not_a_perplexity_of_one(self):
        self.assertTrue(math.isnan(PerplexityReading(0, 0.0).perplexity))

    def test_a_loss_that_would_overflow_exp_reports_infinity(self):
        self.assertEqual(PerplexityReading(1, 10_000.0).perplexity, float("inf"))

    def test_it_describes_itself_with_the_loss_beside_the_ratio(self):
        described = PerplexityReading(10, 23.0).describe()
        self.assertIn("ppl ", described)
        self.assertIn("10 tokens", described)


class SequencePerplexityTest(unittest.TestCase):
    """The seam, which is the only thing chunking can get wrong quietly."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.ids = torch.randint(0, TINY.vocab_size, (TOKENS,), dtype=torch.int64)
        cls.runner = build_runner(chunk_size=512)
        cls.expected = reference_perplexity(cls.runner, cls.ids)

    def test_one_chunk_is_the_formula(self):
        self.assertAlmostEqual(sequence_perplexity(self.runner, self.ids).perplexity, self.expected, places=3)

    def test_every_chunking_is_the_same_number(self):
        """The property: where the chunk boundaries fall cannot change the answer."""
        for chunk in (64, 37, 7):
            with self.subTest(chunk=chunk):
                reading = sequence_perplexity(self.runner, self.ids, chunk_size=chunk)
                self.assertAlmostEqual(reading.perplexity, self.expected, places=3)

    def test_every_token_but_the_first_is_scored_exactly_once(self):
        for chunk in (512, 64, 7):
            with self.subTest(chunk=chunk):
                self.assertEqual(sequence_perplexity(self.runner, self.ids, chunk_size=chunk).tokens, TOKENS - 1)

    def test_a_chunk_larger_than_the_runners_own_is_capped(self):
        """The launch buffers were reserved for the runner's chunk; a bigger one would overrun them."""
        runner = build_runner(chunk_size=64)
        reading = sequence_perplexity(runner, self.ids, chunk_size=100_000)
        self.assertAlmostEqual(reading.perplexity, reference_perplexity(runner, self.ids), places=2)

    def test_it_runs_through_a_quantised_pool_too(self):
        """``batched_decode`` prefills out of the 4-bit cache, which is what makes the number about it."""
        runner = build_runner(chunk_size=128, backend="turboquant_reference", mode="batched_decode")
        reading = sequence_perplexity(runner, self.ids)
        self.assertTrue(math.isfinite(reading.perplexity))
        self.assertGreater(reading.perplexity, 0.0)
        self.assertEqual(reading.tokens, TOKENS - 1)

    def test_a_sequence_with_no_prediction_in_it_is_refused(self):
        with self.assertRaises(ValueError):
            sequence_perplexity(self.runner, torch.tensor([5], dtype=torch.int64))

    def test_a_sequence_past_the_cache_is_refused(self):
        with self.assertRaises(ValueError):
            sequence_perplexity(self.runner, torch.zeros(2000, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
