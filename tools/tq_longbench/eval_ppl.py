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
"""Perplexity over a prompt, measured through the same forward the run prefills with.

A LongBench score says whether an answer was right, and a right answer is a
coarse instrument: a cache can lose a great deal before the letter changes.
Perplexity is the fine one. It asks what probability the model put on the text
it was actually given, token by token, which moves long before any answer does.

    from tq_longbench.eval_ppl import sequence_perplexity

    reading = sequence_perplexity(runner, prompt_ids)
    print(reading.describe())

**Which pool is being measured is the caller's choice, and it matters.** The
forward attends through ``runner.prefill_backend`` by default -- the same path
:meth:`~tq_longbench.engine.StandaloneModelRunner.prefill` takes, so this costs
one extra prefill and no more. Under ``--prefill-mode dense_staging`` that pool
is the *unquantised* one, and the number is the dense model's perplexity: useful
as a baseline, and not a statement about 4-bit KV. Under ``batched_decode`` the
prefill reads the quantised pool, and the same number becomes exactly that
statement. Passing ``backend=runner.decode_backend`` forces the quantised
reading in either mode, at the cost of the generic
:meth:`~tq_longbench.ops.AttentionBackend.prefill_chunk`, which re-presents the
chunk as one decode per token and is orders of magnitude slower.

**Chunked twice**, for two different reasons. The sequence is walked in
``chunk_size`` pieces because that is how the cache is filled and because a
whole sequence of logits is not affordable: at 32k tokens and a 152k vocabulary,
``[tokens, vocab]`` in float16 is nine gigabytes. Then each chunk's cross-entropy
is taken :data:`CROSS_ENTROPY_ROWS` rows at a time, because the float32 the loss
needs would otherwise double the chunk's block. Nothing here holds more than
``CROSS_ENTROPY_ROWS x vocab`` of float32 at once.

The chunk boundary is not a seam: a chunk's last row predicts the first token of
the next chunk, so every token of the sequence but the first is scored exactly
once, and the total is a token-weighted mean rather than a mean of chunk means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tq_longbench.layers import ForwardBatch

#: How many rows of a chunk's ``[tokens, vocab]`` logits are cast to float32 at a
#: time. The cast is what the loss needs and what costs the memory: 256 rows of a
#: 152k vocabulary is 156 MB, which is affordable beside the block it is read
#: from, and a whole 2048-row chunk would not be.
CROSS_ENTROPY_ROWS = 256

#: Above this a perplexity is reported as infinite rather than overflowed.
#: ``exp`` runs out of float at 709, and a loss anywhere near it is a broken run
#: rather than a large number -- the ceiling for a real one is ``ln(vocab)``,
#: about 12.
_MAX_LOG_PERPLEXITY = 700.0


@dataclass(frozen=True)
class PerplexityReading:
    """One sequence's cross-entropy, and the perplexity it implies."""

    #: Tokens actually scored: the sequence minus its first, which nothing predicts.
    tokens: int
    #: Summed cross-entropy over those tokens, in nats.
    nats: float

    @property
    def loss(self) -> float:
        """Mean cross-entropy per scored token."""
        return self.nats / self.tokens if self.tokens else float("nan")

    @property
    def perplexity(self) -> float:
        loss = self.loss
        if math.isnan(loss):
            return float("nan")
        return math.exp(loss) if loss < _MAX_LOG_PERPLEXITY else float("inf")

    def describe(self) -> str:
        return f"ppl {self.perplexity:.4f} (loss {self.loss:.4f} nats over {self.tokens} tokens)"


@torch.inference_mode()
def sequence_perplexity(
    runner,
    token_ids: torch.Tensor,
    chunk_size: int | None = None,
    backend=None,
) -> PerplexityReading:
    """Perplexity of ``token_ids`` under ``runner``'s model, chunk by chunk.

    ``chunk_size`` defaults to the runner's own and is capped by it: the launch
    buffers the backends reserved were sized for that number, and a larger chunk
    would overrun them. A smaller one is always safe and is the lever to pull
    when the logits block is what does not fit.

    ``backend`` is which pool the forward attends through; see the module
    docstring for why the default is the prefill one and when it is not the one
    you want.

    The cache is refilled from position 0 by this call, exactly as a prefill
    would, so a runner is left holding this sequence's KV afterwards -- the same
    contract ``prefill`` has, and the reason a caller that needs the previous
    contents back re-prefills.
    """
    total = int(token_ids.numel())
    if total < 2:
        raise ValueError(f"perplexity needs at least two tokens to have a prediction in it, got {total}")
    if total > runner.config.max_seq_len:
        raise ValueError(f"a {total}-token sequence does not fit the {runner.config.max_seq_len}-token cache")

    attend = backend if backend is not None else runner.prefill_backend
    chunk = min(chunk_size or runner.config.chunk_size, runner.config.chunk_size)
    if chunk <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    ids = token_ids.to(runner.device)
    nats, scored = 0.0, 0
    for start in range(0, total, chunk):
        count = min(chunk, total - start)
        batch = ForwardBatch(
            positions=torch.arange(start, start + count, dtype=torch.int64, device=runner.device),
            slots=runner.decode_backend.cache.slot_mapping(start, count),
            prefix_end=start + count,
            is_decode=False,
        )
        logits = runner.model(
            ids[start : start + count],
            batch,
            runner.write_backends,
            attend,
            last_token_only=False,
        )
        # Row i of this chunk predicts token start+i+1, which for the last row of
        # a non-final chunk is the first token of the next one -- so the labels
        # run one past the chunk wherever there is a token there to run to.
        labels = ids[start + 1 : start + count + 1]
        if labels.numel():
            nats += _chunk_cross_entropy(logits[: labels.numel()], labels)
            scored += int(labels.numel())
    return PerplexityReading(scored, nats)


def _chunk_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Summed cross-entropy of one chunk, in float32, ``CROSS_ENTROPY_ROWS`` at a time.

    Summed rather than averaged so the caller can weight by token and not by
    chunk: the last chunk of a sequence is nearly always short, and a mean of
    chunk means would give its tokens more weight than the rest.
    """
    total = 0.0
    for begin in range(0, int(labels.numel()), CROSS_ENTROPY_ROWS):
        rows = logits[begin : begin + CROSS_ENTROPY_ROWS].to(torch.float32)
        target = labels[begin : begin + CROSS_ENTROPY_ROWS]
        total += float(torch.nn.functional.cross_entropy(rows, target, reduction="sum"))
    return total
