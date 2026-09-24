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
"""Uncompressed attention sinks beside the TurboQuant 4-bit cache, on a host with no NPU.

:func:`~tests.ut.attention.turboquant_cpu_ops.turboquant_cpu_ops` serves the operators from
the CPU, so what runs here is the code that ships: the side-car allocation, the ingestion
that rides the packed write, the scale rows it neutralises afterwards, the merge that
appends the two streams from the decode's own ``lse`` out-tensor, and the un-rotation that
follows it.  The point of the harness is that it can compare *numbers*: every run below is
held against dense float64 attention over the same unquantised K and V, which is what an
fp16 baseline converges to.

The CPU ``npu_turboquant_paged_attention`` writes that out-tensor too, from the same online
softmax it already computed, so the pair the merge consumes here is the pair the kernels
write rather than a stand-in for it.

The sequences are built so that the sinks matter.  Random K gives no token a dominant
score, and then a 4-bit sink is no worse than a 4-bit anything else; a trained model's
sinks attract most of the mass, which is what makes their quantisation error the whole
error.  :func:`_plant_sinks` reproduces that by aligning the first few keys with the decode
query, so removing their quantisation error is measurable rather than a rounding.
"""

import os
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.attention.backend import AttentionType  # type: ignore

from tests.ut.attention.turboquant_cpu_ops import cpu_fused_infer_attention_score, dequantize, turboquant_cpu_ops
from tests.ut.base import TestBase
from vllm_ascend.attention import turboquant_v1 as tq_module
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionState, AscendMetadata
from vllm_ascend.attention.turboquant_layout import (
    TURBOQUANT_LLOYD_MAX_CENTROIDS,
    TURBOQUANT_LSE_STRIDE,
    TURBOQUANT_PACK_FACTOR,
    turboquant_dequantize,
    turboquant_scale_slot,
    turboquant_split_lse,
)
from vllm_ascend.attention.turboquant_rotation import apply_pi, turboquant_pi_signs
from vllm_ascend.attention.turboquant_sink import (
    TURBOQUANT_MAX_SINK_TOKENS,
    TURBOQUANT_SINK_ALIGN_BYTES,
    TurboQuantSinkCache,
    TurboQuantSinkConfig,
    fuse_sink_stream,
    quantized_context_softmax_stats,
    sink_plane_bytes,
    sink_validity_mask,
)
from vllm_ascend.attention.turboquant_v1 import AscendTurboQuantAttentionBackendImpl, activate_turboquant_backend
from vllm_ascend.attention.utils import needs_layer_aware_fia_graph_replay
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

CPU = torch.device("cpu")
DTYPE = torch.bfloat16

NUM_HEADS = 8
HEAD_SIZE = 128
BLOCK_SIZE = 128
# 2048 prompt tokens plus their decode token need 17 blocks; the rest keep the two-sequence
# cases from sharing any block, so a misindexed anchor reads unwritten memory rather than
# the other sequence's sinks.
NUM_BLOCKS = 24

SINK_TOKENS = 4
LAYER_NAME = "model.layers.0.self_attn.attn"

SHORT_PROMPT = 128
MEDIUM_PROMPT = 2048

# How far above the rest of the context a planted sink's logit sits. High enough that 4-bit
# error on the sinks dominates the decode's error, which is the regime the feature exists
# for, and scaled by the head group so a wide GQA group is as sink-dominated as a narrow one.
SINK_LOGIT_GAIN = 16.0

# Dense float64 attention is the target; these are cosine similarities against it.
# An all-4-bit decode over a sink-dominated context sits below the first and an
# uncompressed-sink decode above the second, with the gap being the whole point.
MAX_QUANTIZED_SINK_COSINE = 0.996
MIN_UNCOMPRESSED_SINK_COSINE = 0.9999
# A sequence with only a couple of tokens has no sink-dominated softmax to restore, so it is
# held to the ordinary 4-bit bound on the one token that is still quantised.
MIN_SHORT_SEQUENCE_COSINE = 0.99


def _slot(block_table, seq: int, position: int) -> int:
    return block_table[seq][position // BLOCK_SIZE] * BLOCK_SIZE + position % BLOCK_SIZE


def _block_table(prompt_lens, num_kv_heads: int):
    """One row per sequence, blocks handed out in order and never shared."""
    del num_kv_heads
    rows, next_block = [], 0
    width = max((length // BLOCK_SIZE) + 1 for length in prompt_lens)
    for length in prompt_lens:
        needed = (length // BLOCK_SIZE) + 1
        rows.append(tuple(range(next_block, next_block + needed)) + (0,) * (width - needed))
        next_block += needed
    assert next_block <= NUM_BLOCKS, f"{next_block} blocks needed, cache holds {NUM_BLOCKS}"
    return tuple(rows)


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(actual.to(torch.float64), expected.to(torch.float64), dim=-1)


def _exact_attention(query, key, value, num_kv_heads: int, causal: bool) -> torch.Tensor:
    """Dense float64 attention, each kv head shared by a head group."""
    group = NUM_HEADS // num_kv_heads
    q = query.to(torch.float64)
    k = key.to(torch.float64).repeat_interleave(group, dim=1)
    v = value.to(torch.float64).repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) * HEAD_SIZE**-0.5
    if causal:
        allowed = torch.ones(q.shape[0], k.shape[0], dtype=torch.bool).tril(k.shape[0] - q.shape[0])
        scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v)


def _plant_sinks(key, value, decode_query, prompt_lens, num_kv_heads: int, num_sinks: int) -> None:
    """Make the first ``num_sinks`` keys of each prompt dominate that sequence's decode.

    A sink is a key every query is strongly aligned with, so its score sits far above the
    rest of the context and its value carries most of the output.  Under GQA one kv head
    serves a whole group of query heads, and a key aligned with one of them is orthogonal to
    the others -- so the direction is the group's mean, which every head in the group has a
    positive component along, and the gain is scaled by ``sqrt(group)`` to make up for how
    much averaging ``group`` random directions shortens that component.  Without this the
    sinks dominate one head per group and the other heads carry the ordinary 4-bit error,
    which is what the measurement would then be reading.

    The *values* are left random and large: the output is essentially their weighted mean, so
    an error in their direction is what a cosine against dense attention can see, where an
    error in the magnitude of a value parallel to the key would cancel out of it.
    """
    group = NUM_HEADS // num_kv_heads
    gain = SINK_LOGIT_GAIN * group**0.5
    generator = torch.Generator().manual_seed(4242)
    start = 0
    for seq, length in enumerate(prompt_lens):
        for kv_head in range(num_kv_heads):
            heads = decode_query[seq, kv_head * group : (kv_head + 1) * group].to(torch.float32)
            direction = heads.mean(dim=0)
            direction = direction / direction.norm()
            for sink in range(num_sinks):
                key[start + sink, kv_head] = (direction * gain).to(key.dtype)
                value[start + sink, kv_head] = (3.0 * torch.randn(HEAD_SIZE, generator=generator)).to(value.dtype)
        start += length


@dataclass
class _Run:
    impl: AscendTurboQuantAttentionBackendImpl
    kv_cache: tuple
    prompt_key: torch.Tensor
    prompt_value: torch.Tensor
    decode_query: torch.Tensor
    decode_key: torch.Tensor
    decode_value: torch.Tensor
    decode_output: torch.Tensor


class _TurboQuantSinkHarness(TestBase):
    """Prefill then decode through the shipping backend, with the operators served from CPU."""

    def setUp(self):
        self.vllm_config = MagicMock()
        self.vllm_config.kv_transfer_config = None
        self.vllm_config.quant_config = None
        self.vllm_config.model_config.hf_config = SimpleNamespace(model_type="glm4")
        self.vllm_config.model_config.dtype = DTYPE
        patchers = (
            patch("vllm_ascend.attention.attention_v1.get_current_vllm_config", return_value=self.vllm_config),
            patch("vllm_ascend.attention.utils.get_current_vllm_config", return_value=self.vllm_config),
            patch.dict(os.environ, {"ENABLE_TURBOQUANT": "1"}),
            patch.object(
                tq_module.torch_npu,
                "npu_fused_infer_attention_score",
                cpu_fused_infer_attention_score,
                create=True,
            ),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        needs_layer_aware_fia_graph_replay.cache_clear()
        self.addCleanup(needs_layer_aware_fia_graph_replay.cache_clear)
        ops = turboquant_cpu_ops()
        ops.__enter__()
        self.addCleanup(ops.__exit__, None, None, None)

    def _build_impl(self, num_kv_heads: int, sink_tokens: int) -> AscendTurboQuantAttentionBackendImpl:
        with patch.dict(os.environ, {"VLLM_ASCEND_TQ_SINK_TOKENS": str(sink_tokens)}):
            impl = AscendTurboQuantAttentionBackendImpl(
                num_heads=NUM_HEADS,
                head_size=HEAD_SIZE,
                scale=HEAD_SIZE**-0.5,
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="turboquant_4bit_nc",
                logits_soft_cap=None,
                attn_type=AttentionType.DECODER,
                kv_sharing_target_layer_name=None,
            )
            layer = SimpleNamespace(layer_name=LAYER_NAME, impl=impl)
            activate_turboquant_backend(layer)
            impl.process_weights_after_loading(DTYPE)
        self.assertFalse(impl.cube_decode, "the CPU harness serves the AIV operators only")
        return impl

    @staticmethod
    def _allocate_kv_cache(num_kv_heads: int) -> tuple:
        shape = AscendAttentionBackend.get_kv_cache_shape(NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, HEAD_SIZE)
        unpacked_bytes = NUM_BLOCKS * BLOCK_SIZE * num_kv_heads * HEAD_SIZE
        return tuple(
            NPUModelRunner._view_raw_kv_cache(torch.zeros(unpacked_bytes, dtype=torch.int8), torch.uint8, shape[1:])
            for _ in range(2)
        )

    def _run(
        self,
        *,
        prompt_lens,
        num_kv_heads: int,
        sink_tokens: int,
        plant: bool = True,
        seed: int = 0,
        impl: AscendTurboQuantAttentionBackendImpl | None = None,
    ) -> _Run:
        impl = impl if impl is not None else self._build_impl(num_kv_heads, sink_tokens)
        kv_cache = self._allocate_kv_cache(num_kv_heads)
        block_table = _block_table(prompt_lens, num_kv_heads)
        generator = torch.Generator().manual_seed(seed)
        num_prompt_tokens = sum(prompt_lens)
        num_seqs = len(prompt_lens)

        def normal(*shape):
            return torch.randn(*shape, generator=generator).to(DTYPE)

        prompt_query = normal(num_prompt_tokens, NUM_HEADS, HEAD_SIZE)
        prompt_key = normal(num_prompt_tokens, num_kv_heads, HEAD_SIZE)
        prompt_value = normal(num_prompt_tokens, num_kv_heads, HEAD_SIZE)
        decode_query = normal(num_seqs, NUM_HEADS, HEAD_SIZE)
        decode_key = normal(num_seqs, num_kv_heads, HEAD_SIZE)
        decode_value = normal(num_seqs, num_kv_heads, HEAD_SIZE)
        if plant:
            _plant_sinks(prompt_key, prompt_value, decode_query, prompt_lens, num_kv_heads, sink_tokens or SINK_TOKENS)

        ends = torch.tensor(prompt_lens).cumsum(dim=0)
        prefill_metadata = AscendMetadata(
            attn_state=AscendAttentionState.PrefillNoCache,
            num_actual_tokens=num_prompt_tokens,
            num_prefills=num_seqs,
            seq_lens=torch.tensor(prompt_lens),
            seq_lens_list=list(prompt_lens),
            actual_seq_lengths_q=ends.tolist(),
            query_start_loc=torch.cat((torch.zeros(1, dtype=ends.dtype), ends)),
            max_query_len=max(prompt_lens),
            block_tables=torch.tensor(block_table, dtype=torch.int32),
            slot_mapping=torch.tensor(
                [_slot(block_table, seq, p) for seq, length in enumerate(prompt_lens) for p in range(length)]
            ),
        )
        prefill_output = torch.zeros(num_prompt_tokens, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        impl.forward(
            SimpleNamespace(layer_name=LAYER_NAME),
            prompt_query,
            prompt_key,
            prompt_value,
            kv_cache,
            prefill_metadata,
            prefill_output,
        )

        decode_metadata = AscendMetadata(
            attn_state=AscendAttentionState.DecodeOnly,
            num_actual_tokens=num_seqs,
            num_decode_tokens=num_seqs,
            num_decodes=num_seqs,
            seq_lens=torch.tensor(prompt_lens) + 1,
            seq_lens_list=[length + 1 for length in prompt_lens],
            actual_seq_lengths_q=list(range(1, num_seqs + 1)),
            query_start_loc=torch.arange(num_seqs + 1),
            max_query_len=1,
            block_tables=torch.tensor(block_table, dtype=torch.int32),
            slot_mapping=torch.tensor([_slot(block_table, s, length) for s, length in enumerate(prompt_lens)]),
        )
        decode_output = torch.zeros(num_seqs, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        impl.forward(
            SimpleNamespace(layer_name=LAYER_NAME),
            decode_query,
            decode_key,
            decode_value,
            kv_cache,
            decode_metadata,
            decode_output,
        )
        return _Run(
            impl=impl,
            kv_cache=kv_cache,
            prompt_key=prompt_key,
            prompt_value=prompt_value,
            decode_query=decode_query,
            decode_key=decode_key,
            decode_value=decode_value,
            decode_output=decode_output,
        )

    def _decode_cosine(self, run: _Run, prompt_lens, num_kv_heads: int) -> torch.Tensor:
        """Every sequence's decode output against dense float64 attention over unquantised K/V."""
        cosines, start = [], 0
        for seq, length in enumerate(prompt_lens):
            end = start + length
            keys = torch.cat((run.prompt_key[start:end], run.decode_key[seq : seq + 1]))
            values = torch.cat((run.prompt_value[start:end], run.decode_value[seq : seq + 1]))
            exact = _exact_attention(run.decode_query[seq : seq + 1], keys, values, num_kv_heads, causal=False)
            cosines.append(_cosine(run.decode_output[seq : seq + 1], exact))
            start = end
        return torch.cat(cosines)


class TestTurboQuantSinkConfig(TestBase):
    def test_the_range_is_enforced(self):
        self.assertFalse(TurboQuantSinkConfig().enabled)
        self.assertTrue(TurboQuantSinkConfig(num_sink_tokens=4).enabled)
        for value in (-1, TURBOQUANT_MAX_SINK_TOKENS + 1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "VLLM_ASCEND_TQ_SINK_TOKENS"):
                TurboQuantSinkConfig(num_sink_tokens=value)

    def test_the_default_is_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_ASCEND_TQ_SINK_TOKENS", None)
            self.assertEqual(TurboQuantSinkConfig.from_env().num_sink_tokens, 0)
        with patch.dict(os.environ, {"VLLM_ASCEND_TQ_SINK_TOKENS": "4"}):
            self.assertEqual(TurboQuantSinkConfig.from_env().num_sink_tokens, 4)

    def test_a_sink_count_past_the_block_is_refused(self):
        """Every sink lives in logical block 0; the anchor-row arithmetic depends on it."""
        with self.assertRaisesRegex(ValueError, "do not fit"):
            TurboQuantSinkCache(
                num_blocks=2,
                block_size=2,
                num_sink_tokens=4,
                num_kv_heads=1,
                head_size=HEAD_SIZE,
                dtype=DTYPE,
                device=CPU,
            )


class TestTurboQuantSinkFusionMath(TestBase):
    """The merge on its own, against a softmax written out directly.

    The decode it stands in for has had its sink rows neutralised in the scale plane, so
    what it saw at those positions is a score of exactly zero and a value of exactly zero.
    That is what the fixtures below reproduce: the "operator" attends over
    ``[0, 0, ..., tail]`` and the merge has to turn its answer into attention over
    ``[dense sinks, tail]``.
    """

    @staticmethod
    def _neutralized_decode(tail_scores, tail_values, sinks):
        """``(output, running_max, mass)`` of a decode whose first ``sinks`` rows are zeroed."""
        tokens, _, heads = tail_scores.shape
        head_size = tail_values.shape[-1]
        scores = torch.cat((torch.zeros(tokens, sinks, heads), tail_scores), dim=1)
        values = torch.cat((torch.zeros(tokens, sinks, heads, head_size), tail_values), dim=1)
        running_max = scores.amax(dim=1)
        mass = torch.exp(scores - running_max.unsqueeze(1)).sum(dim=1)
        weights = torch.softmax(scores.to(torch.float64), dim=1)
        return torch.einsum("tnh,tnhd->thd", weights, values.to(torch.float64)), running_max, mass

    @staticmethod
    def _exact_softmax(sink_scores, sink_values, tail_scores, tail_values):
        """Dense float64 attention over the sinks followed by the tail."""
        weights = torch.softmax(torch.cat((sink_scores, tail_scores), dim=1).to(torch.float64), dim=1)
        values = torch.cat((sink_values, tail_values), dim=1).to(torch.float64)
        return torch.einsum("tnh,tnhd->thd", weights, values)

    def test_appending_the_exact_sinks_reproduces_the_exact_softmax(self):
        tokens, heads, sinks, context = 3, 4, 4, 64
        generator = torch.Generator().manual_seed(7)

        def normal(*shape):
            return torch.randn(*shape, generator=generator, dtype=torch.float32)

        # The tail of the context, which the "operator" and the reference share verbatim.
        tail_scores = normal(tokens, context, heads)
        tail_values = normal(tokens, context, heads, HEAD_SIZE)
        # The sinks as they really are. Their logits sit far above the tail's, which is the
        # regime the feature exists for and the one where the rescaling has to be right.
        dense_scores = normal(tokens, sinks, heads) + 8.0
        dense_values = normal(tokens, sinks, heads, HEAD_SIZE)
        valid = torch.ones(tokens, sinks, dtype=torch.bool)

        operator_output, running_max, mass = self._neutralized_decode(tail_scores, tail_values, sinks)
        expected = self._exact_softmax(dense_scores, dense_values, tail_scores, tail_values)

        fused = fuse_sink_stream(
            attention_output=operator_output.to(torch.float32),
            running_max=running_max,
            mass=mass,
            neutralized=valid.sum(dim=1, keepdim=True),
            dense_scores=dense_scores,
            dense_values=dense_values,
            valid=valid,
        )
        # The merge is exact up to fp32; the reference runs in float64.
        torch.testing.assert_close(fused.to(torch.float64), expected, rtol=2e-4, atol=2e-4)
        # And it is a real correction: the operator's own answer is measurably further away.
        operator_error = (operator_output - expected).abs().max().item()
        fused_error = (fused.to(torch.float64) - expected).abs().max().item()
        self.assertGreater(operator_error, 10 * fused_error)

    def test_only_the_rows_the_decode_neutralized_are_subtracted(self):
        """A sequence shorter than the slot count zeroed fewer cache rows than there are slots.

        A merge that subtracted the slot count rather than the filled count would take mass
        the decode never carried, and the residual would be short by ``exp(-m')`` per
        unfilled slot -- small, silent, and in the denominator of every head.
        """
        heads, context = 2, 8
        generator = torch.Generator().manual_seed(17)

        def normal(*shape):
            return torch.randn(*shape, generator=generator, dtype=torch.float32)

        for filled in (1, 2, 4):
            tail_scores = normal(1, context, heads)
            tail_values = normal(1, context, heads, HEAD_SIZE)
            dense_scores = normal(1, filled, heads) + 4.0
            dense_values = normal(1, filled, heads, HEAD_SIZE)
            output, running_max, mass = self._neutralized_decode(tail_scores, tail_values, filled)
            fused = fuse_sink_stream(
                attention_output=output.to(torch.float32),
                running_max=running_max,
                mass=mass,
                neutralized=torch.tensor([[filled]]),
                dense_scores=dense_scores,
                dense_values=dense_values,
                valid=torch.ones(1, filled, dtype=torch.bool),
            )
            expected = self._exact_softmax(dense_scores, dense_values, tail_scores, tail_values)
            with self.subTest(filled=filled):
                torch.testing.assert_close(fused.to(torch.float64), expected, rtol=2e-4, atol=2e-4)

    def test_a_token_with_no_valid_sink_is_left_alone(self):
        tokens, heads, sinks = 2, 2, 4
        output = torch.randn(tokens, heads, HEAD_SIZE)
        valid = torch.zeros(tokens, sinks, dtype=torch.bool)
        valid[0] = True
        fused = fuse_sink_stream(
            attention_output=output,
            running_max=torch.zeros(tokens, heads),
            mass=torch.ones(tokens, heads),
            neutralized=valid.sum(dim=1, keepdim=True),
            dense_scores=torch.zeros(tokens, sinks, heads),
            dense_values=torch.zeros(tokens, sinks, heads, HEAD_SIZE),
            valid=valid,
        )
        torch.testing.assert_close(fused[1], output[1])

    def test_a_token_whose_decode_carried_no_mass_is_left_alone(self):
        """An empty context: the max lane holds whatever the kernel initialised, not a logit.

        The merge must branch on the mass and never on the max, or it would take the
        accumulator's sentinel for a score and rescale everything else against it.
        """
        tokens, heads, sinks = 1, 2, 2
        output = torch.randn(tokens, heads, HEAD_SIZE)
        fused = fuse_sink_stream(
            attention_output=output,
            running_max=torch.full((tokens, heads), -3.0e38),
            mass=torch.zeros(tokens, heads),
            neutralized=torch.zeros(tokens, 1),
            dense_scores=torch.zeros(tokens, sinks, heads),
            dense_values=torch.randn(tokens, sinks, heads, HEAD_SIZE),
            valid=torch.ones(tokens, sinks, dtype=torch.bool),
        )
        torch.testing.assert_close(fused, output)
        self.assertTrue(bool(torch.isfinite(fused).all()))

    def test_a_context_that_is_nothing_but_sinks_is_the_dense_sinks_alone(self):
        """Every row the decode read was neutralised, so the residual is exactly zero.

        The previous shape of this merge subtracted the quantised sinks from a mass they
        dominated, and what was left had to be clamped rather than trusted.  Here the
        subtracted term is ``n`` against a decode that summed ``n`` rows of ``exp(0)``, so
        the residual is zero by construction and the answer is the dense sinks' own softmax.
        """
        tokens, heads, sinks = 1, 1, 2
        scores = torch.tensor([[[3.0], [1.0]]])
        values = torch.randn(tokens, sinks, heads, HEAD_SIZE)
        # What a neutralised decode over nothing but sinks produces: every score 0, every
        # value 0, and a mass of exactly one unit per row.
        fused = fuse_sink_stream(
            attention_output=torch.zeros(tokens, heads, HEAD_SIZE),
            running_max=torch.zeros(tokens, heads),
            mass=torch.full((tokens, heads), float(sinks)),
            neutralized=torch.full((tokens, 1), float(sinks)),
            dense_scores=scores,
            dense_values=values,
            valid=torch.ones(tokens, sinks, dtype=torch.bool),
        )
        expected = torch.einsum("tnh,tnhd->thd", torch.softmax(scores, dim=1), values)
        torch.testing.assert_close(fused, expected, rtol=1e-5, atol=1e-5)
        self.assertTrue(bool(torch.isfinite(fused).all()))


class TestTurboQuantSinkCachePlane(TestBase):
    def test_the_plane_holds_the_sequence_prefix_at_its_anchor_block(self):
        num_kv_heads, num_blocks = 2, 8
        cache = TurboQuantSinkCache(
            num_blocks=num_blocks,
            block_size=BLOCK_SIZE,
            num_sink_tokens=SINK_TOKENS,
            num_kv_heads=num_kv_heads,
            head_size=HEAD_SIZE,
            dtype=DTYPE,
            device=CPU,
        )
        generator = torch.Generator().manual_seed(11)
        key = torch.randn(10, num_kv_heads, HEAD_SIZE, generator=generator).to(DTYPE)
        value = torch.randn(10, num_kv_heads, HEAD_SIZE, generator=generator).to(DTYPE)
        anchors = torch.tensor([5, 2], dtype=torch.int32)

        stored = cache.ingest(
            key,
            value,
            anchor_blocks=anchors,
            query_starts=[0, 6],
            query_lens=[6, 4],
            context_lens=[0, 0],
        )
        self.assertEqual(stored, 2 * SINK_TOKENS)
        torch.testing.assert_close(cache.planes[0, 5], key[:SINK_TOKENS])
        torch.testing.assert_close(cache.planes[1, 5], value[:SINK_TOKENS])
        torch.testing.assert_close(cache.planes[0, 2], key[6 : 6 + SINK_TOKENS])
        # Nothing else was touched -- a misindexed anchor would show up here.
        untouched = [block for block in range(num_blocks) if block not in (5, 2)]
        self.assertEqual(int(cache.planes[:, untouched].count_nonzero()), 0)

        gathered_key, gathered_value = cache.gather(anchors)
        torch.testing.assert_close(gathered_key[0], key[:SINK_TOKENS])
        torch.testing.assert_close(gathered_value[1], value[6 : 6 + SINK_TOKENS])

    def test_a_prefix_that_arrives_across_steps_is_still_ingested(self):
        """A prompt shorter than the sink count keeps filling its sinks from decode tokens."""
        cache = TurboQuantSinkCache(
            num_blocks=4,
            block_size=BLOCK_SIZE,
            num_sink_tokens=SINK_TOKENS,
            num_kv_heads=1,
            head_size=HEAD_SIZE,
            dtype=DTYPE,
            device=CPU,
        )
        generator = torch.Generator().manual_seed(12)
        key = torch.randn(4, 1, HEAD_SIZE, generator=generator).to(DTYPE)
        anchors = torch.tensor([1], dtype=torch.int32)
        self.assertEqual(cache.ingest(key, key, anchor_blocks=anchors, query_starts=[0], query_lens=[2],
                                      context_lens=[0]), 2)  # fmt: skip
        self.assertEqual(cache.ingest(key, key, anchor_blocks=anchors, query_starts=[2], query_lens=[1],
                                      context_lens=[2]), 1)  # fmt: skip
        # Past the sinks nothing is copied any more, which is what makes steady-state
        # decode free.
        self.assertEqual(cache.ingest(key, key, anchor_blocks=anchors, query_starts=[3], query_lens=[1],
                                      context_lens=[SINK_TOKENS]), 0)  # fmt: skip
        torch.testing.assert_close(cache.planes[0, 1, :3], key[:3])
        self.assertEqual(int(cache.planes[0, 1, 3].count_nonzero()), 0)

    def test_ingesting_by_slot_agrees_with_ingesting_by_request(self):
        """The two writers must leave identical planes, or the harness measures another cache.

        ``ingest`` is driven by host-side request boundaries, which is what a ragged vLLM
        batch needs; ``ingest_by_slot`` reads the slots themselves in one fixed-shape
        device operation, which is what ``tools/tq_longbench`` needs to keep a chunk free
        of device-to-host copies. A drift between them would not raise -- it would put a
        different prefix in the side-car on one of the two paths.
        """
        generator = torch.Generator().manual_seed(31)
        key = torch.randn(16, 2, HEAD_SIZE, generator=generator).to(DTYPE)
        value = torch.randn(16, 2, HEAD_SIZE, generator=generator).to(DTYPE)

        def fresh():
            return TurboQuantSinkCache(
                num_blocks=1,
                block_size=BLOCK_SIZE,
                num_sink_tokens=SINK_TOKENS,
                num_kv_heads=2,
                head_size=HEAD_SIZE,
                dtype=DTYPE,
                device=CPU,
            )

        by_request, by_slot = fresh(), fresh()
        # One sequence in block 0, arriving in two chunks that straddle the sink count.
        for start, count in ((0, 3), (3, 13)):
            by_request.ingest(
                key[start : start + count],
                value[start : start + count],
                anchor_blocks=torch.zeros(1, dtype=torch.int32),
                query_starts=[0],
                query_lens=[count],
                context_lens=[start],
            )
            by_slot.ingest_by_slot(
                key[start : start + count],
                value[start : start + count],
                torch.arange(start, start + count),
                block=0,
            )
        self.assertTrue(torch.equal(by_request.planes, by_slot.planes))
        self.assertTrue(torch.equal(by_slot.planes[0, 0], key[:SINK_TOKENS]))

    def test_the_footprint_is_a_few_kilobytes_per_sequence(self):
        cache = TurboQuantSinkCache(
            num_blocks=64,
            block_size=BLOCK_SIZE,
            num_sink_tokens=SINK_TOKENS,
            num_kv_heads=8,
            head_size=HEAD_SIZE,
            dtype=torch.float16,
            device=CPU,
        )
        # 2 planes * 4 sinks * 8 kv heads * 128 lanes * 2 bytes.
        self.assertEqual(cache.bytes_per_sequence, 16384)
        payload = 64 * cache.bytes_per_sequence
        # nbytes is what the arena paid, which is the payload plus the alignment slack.
        self.assertEqual(cache.nbytes, payload + TURBOQUANT_SINK_ALIGN_BYTES)
        self.assertEqual(
            cache.nbytes,
            sink_plane_bytes(
                num_blocks=64,
                num_sink_tokens=SINK_TOKENS,
                num_kv_heads=8,
                head_size=HEAD_SIZE,
                element_size=2,
            ),
        )
        # And what the payload is against the packed cache it accompanies: 4 * N / block_size.
        packed_bytes = 2 * 64 * BLOCK_SIZE * 8 * (HEAD_SIZE // TURBOQUANT_PACK_FACTOR)
        self.assertAlmostEqual(payload / packed_bytes, 4 * SINK_TOKENS / BLOCK_SIZE, places=6)

    def test_the_plane_starts_on_a_whole_dma_line(self):
        """The slack is there to be used: a view at the first aligned element, not at zero."""
        for num_kv_heads, dtype in ((2, torch.float16), (8, torch.bfloat16), (1, torch.float32)):
            cache = TurboQuantSinkCache(
                num_blocks=3,
                block_size=BLOCK_SIZE,
                num_sink_tokens=SINK_TOKENS,
                num_kv_heads=num_kv_heads,
                head_size=HEAD_SIZE,
                dtype=dtype,
                device=CPU,
            )
            with self.subTest(num_kv_heads=num_kv_heads, dtype=dtype):
                self.assertEqual(cache.planes.data_ptr() % TURBOQUANT_SINK_ALIGN_BYTES, 0)
                self.assertTrue(cache.planes.is_contiguous())
                self.assertEqual(int(cache.planes.count_nonzero()), 0)

    def test_a_token_whose_rows_are_not_a_whole_group_is_refused(self):
        """The base alignment says nothing about a row unless the row length carries it."""
        with self.assertRaisesRegex(ValueError, "whole group"):
            TurboQuantSinkCache(
                num_blocks=1,
                block_size=BLOCK_SIZE,
                num_sink_tokens=1,
                num_kv_heads=1,
                head_size=48,
                dtype=DTYPE,
                device=CPU,
            )

    def test_neutralizing_zeroes_exactly_the_scale_rows_the_sinks_were_written_to(self):
        """The quantised copies leave the decode's arithmetic through the scale plane.

        Zeroing a token's K and V scale lanes makes its cached key exactly the zero vector
        and its cached value contribute exactly nothing, on whichever codebook and byte
        order the writer used -- which is what lets the merge be layout-independent.  The
        selection has to be the one :meth:`ingest` made, to the row.
        """
        num_kv_heads, num_blocks = 2, 8
        cache = TurboQuantSinkCache(
            num_blocks=num_blocks,
            block_size=BLOCK_SIZE,
            num_sink_tokens=SINK_TOKENS,
            num_kv_heads=num_kv_heads,
            head_size=HEAD_SIZE,
            dtype=DTYPE,
            device=CPU,
        )
        slot_floats = turboquant_scale_slot(num_kv_heads)
        scale_cache = torch.ones(num_blocks, BLOCK_SIZE, slot_floats, dtype=torch.float32)
        anchors = torch.tensor([5, 2], dtype=torch.int32)
        selection = {
            "anchor_blocks": anchors,
            "query_starts": [0, 6],
            "query_lens": [6, 4],
            "context_lens": [0, 0],
        }

        zeroed = cache.neutralize_scale_rows(scale_cache, **selection)
        self.assertEqual(zeroed, 2 * SINK_TOKENS)
        for block in (5, 2):
            self.assertEqual(int(scale_cache[block, :SINK_TOKENS].count_nonzero()), 0)
            self.assertEqual(int((scale_cache[block, SINK_TOKENS:] != 1.0).sum()), 0)
        untouched = [block for block in range(num_blocks) if block not in (5, 2)]
        self.assertEqual(int((scale_cache[untouched] != 1.0).sum()), 0)

        # Idempotent: a prompt split across steps, and a prefix-cache hit, both re-run it.
        scale_cache[5, :SINK_TOKENS] = 1.0
        self.assertEqual(cache.neutralize_scale_rows(scale_cache, **selection), 2 * SINK_TOKENS)
        self.assertEqual(int(scale_cache[5, :SINK_TOKENS].count_nonzero()), 0)

    def test_neutralizing_a_step_with_no_sinks_left_touches_nothing(self):
        cache = TurboQuantSinkCache(
            num_blocks=2,
            block_size=BLOCK_SIZE,
            num_sink_tokens=SINK_TOKENS,
            num_kv_heads=1,
            head_size=HEAD_SIZE,
            dtype=DTYPE,
            device=CPU,
        )
        scale_cache = torch.ones(2, BLOCK_SIZE, turboquant_scale_slot(1), dtype=torch.float32)
        zeroed = cache.neutralize_scale_rows(
            scale_cache,
            anchor_blocks=torch.tensor([1], dtype=torch.int32),
            query_starts=[0],
            query_lens=[1],
            context_lens=[SINK_TOKENS],
        )
        self.assertEqual(zeroed, 0)
        self.assertEqual(int((scale_cache != 1.0).sum()), 0)

    def test_neutralizing_by_slot_agrees_with_neutralizing_by_request(self):
        """The two selections have to name the same rows, or one harness measures another cache."""
        num_kv_heads = 2
        slot_floats = turboquant_scale_slot(num_kv_heads)

        def fresh():
            return TurboQuantSinkCache(
                num_blocks=1,
                block_size=BLOCK_SIZE,
                num_sink_tokens=SINK_TOKENS,
                num_kv_heads=num_kv_heads,
                head_size=HEAD_SIZE,
                dtype=DTYPE,
                device=CPU,
            )

        by_request_scales = torch.ones(1, BLOCK_SIZE, slot_floats, dtype=torch.float32)
        by_slot_scales = torch.ones(1, BLOCK_SIZE, slot_floats, dtype=torch.float32)
        by_request, by_slot = fresh(), fresh()
        for start, count in ((0, 3), (3, 13)):
            by_request.neutralize_scale_rows(
                by_request_scales,
                anchor_blocks=torch.zeros(1, dtype=torch.int32),
                query_starts=[0],
                query_lens=[count],
                context_lens=[start],
            )
            by_slot.neutralize_scale_rows_by_slot(by_slot_scales, block=0)
        self.assertTrue(torch.equal(by_request_scales, by_slot_scales))
        self.assertEqual(int(by_slot_scales[0, :SINK_TOKENS].count_nonzero()), 0)
        self.assertEqual(int((by_slot_scales[0, SINK_TOKENS:] != 1.0).sum()), 0)

    def test_the_validity_mask_follows_the_sequence_length(self):
        mask, any_valid = sink_validity_mask(torch.tensor([0, 2, 9]), 4, SINK_TOKENS, CPU)
        self.assertTrue(any_valid)
        self.assertEqual(mask.tolist(), [[False] * 4, [True, True, False, False], [True] * 4, [False] * 4])
        empty, none_valid = sink_validity_mask(torch.tensor([0, 0]), 2, SINK_TOKENS, CPU)
        self.assertFalse(none_valid)
        self.assertEqual(int(empty.count_nonzero()), 0)


class TestTurboQuantSinkDecode(_TurboQuantSinkHarness):
    def test_sinks_off_is_the_unmodified_path(self):
        """Zero must be free: no plane, no ingestion, and the operator's own output."""
        prompts = (SHORT_PROMPT,)
        off = self._run(prompt_lens=prompts, num_kv_heads=2, sink_tokens=0, plant=False)
        self.assertIsNone(off.impl.sink_cache)
        self.assertFalse(off.impl.sink_config.enabled)

        with patch.object(tq_module, "fuse_decode_sinks", side_effect=AssertionError("fusion must not run")):
            again = self._run(prompt_lens=prompts, num_kv_heads=2, sink_tokens=0, plant=False)
        # Bit-identical, not merely close: nothing on the path was reached.
        self.assertTrue(bool(torch.equal(off.decode_output, again.decode_output)))

    def test_a_short_sequence_tracks_the_dense_baseline(self):
        prompts = (SHORT_PROMPT, SHORT_PROMPT)
        self._assert_sinks_recover_the_baseline(prompts, num_kv_heads=2)

    def test_a_medium_sequence_tracks_the_dense_baseline(self):
        self._assert_sinks_recover_the_baseline((MEDIUM_PROMPT,), num_kv_heads=2)

    def test_gqa_and_mqa_head_expansion_hold(self):
        # 2 kv heads is a group of 4, 8 is one head per group, 1 is MQA.
        for num_kv_heads in (1, 2, 8):
            with self.subTest(num_kv_heads=num_kv_heads):
                self._assert_sinks_recover_the_baseline((SHORT_PROMPT,), num_kv_heads=num_kv_heads)

    def _assert_sinks_recover_the_baseline(self, prompts, *, num_kv_heads: int) -> None:
        quantized = self._run(prompt_lens=prompts, num_kv_heads=num_kv_heads, sink_tokens=0)
        with_sinks = self._run(prompt_lens=prompts, num_kv_heads=num_kv_heads, sink_tokens=SINK_TOKENS)
        self.assertIsNotNone(with_sinks.impl.sink_cache)
        self.assertEqual(with_sinks.impl.sink_cache.num_sink_tokens, SINK_TOKENS)

        before = self._decode_cosine(quantized, prompts, num_kv_heads).min().item()
        after = self._decode_cosine(with_sinks, prompts, num_kv_heads).min().item()
        self.assertLess(before, MAX_QUANTIZED_SINK_COSINE, "the harness has to be able to see the sink error")
        self.assertGreater(after, MIN_UNCOMPRESSED_SINK_COSINE)
        self.assertGreater(after, before)

    def test_the_plane_holds_the_prompt_prefix_after_a_real_prefill(self):
        prompts = (SHORT_PROMPT, SHORT_PROMPT)
        run = self._run(prompt_lens=prompts, num_kv_heads=2, sink_tokens=SINK_TOKENS)
        block_table = _block_table(prompts, 2)
        start = 0
        for seq, length in enumerate(prompts):
            anchor = block_table[seq][0]
            keys, values = run.impl.sink_cache.gather(torch.tensor([anchor], dtype=torch.int32))
            torch.testing.assert_close(keys[0], run.prompt_key[start : start + SINK_TOKENS])
            torch.testing.assert_close(values[0], run.prompt_value[start : start + SINK_TOKENS])
            start += length

    def test_a_sequence_shorter_than_the_sink_count_decodes(self):
        """Two prompt tokens, four sink slots: the two unfilled slots must not be read."""
        run = self._run(prompt_lens=(2,), num_kv_heads=2, sink_tokens=SINK_TOKENS, plant=False)
        cosine = self._decode_cosine(run, (2,), 2)
        self.assertTrue(bool(torch.isfinite(run.decode_output.to(torch.float32)).all()))
        self.assertGreater(cosine.min().item(), MIN_SHORT_SEQUENCE_COSINE)

    def test_the_decode_still_reuses_its_scratch(self):
        """The merge must not reintroduce a per-step allocation the capture path forbids."""
        run = self._run(prompt_lens=(SHORT_PROMPT,), num_kv_heads=2, sink_tokens=SINK_TOKENS)
        workspace, rotated_query = run.impl.decode_workspace, run.impl.rotated_query
        plane, lse = run.impl.sink_cache.planes, run.impl.sink_lse
        self.assertIsNotNone(lse)
        again = self._run(prompt_lens=(SHORT_PROMPT,), num_kv_heads=2, sink_tokens=SINK_TOKENS, impl=run.impl)
        self.assertIs(again.impl.decode_workspace, workspace)
        self.assertIs(again.impl.rotated_query, rotated_query)
        self.assertIs(again.impl.sink_cache.planes, plane)
        self.assertIs(again.impl.sink_lse, lse)

    def test_no_statistics_buffer_is_allocated_with_sinks_off(self):
        """The out-tensor is optional in the schema so that this costs a launch nothing."""
        run = self._run(prompt_lens=(SHORT_PROMPT,), num_kv_heads=2, sink_tokens=0, plant=False)
        self.assertIsNone(run.impl.sink_lse)
        self.assertIsNone(run.impl._sink_lse(1, CPU))

    def test_the_decodes_own_statistics_match_the_host_oracle(self):
        """What the kernel wrote against an independent derivation of the same pair.

        This is the whole of phase two: the merge's weight used to be recomputed on the
        host at ``O(context)`` per step, and is now read out of the decode.  The two have to
        agree, or the merge rescales against a denominator the operator never used -- which
        does not raise, it just returns a plausible wrong answer.

        The oracle reads the packed planes row-major on the Lloyd-Max grid, which is what
        this harness's writer leaves; it is a test instrument, not a path.
        """
        prompts = (SHORT_PROMPT,)
        # A folded layer, so that the rotated-query buffer still holds Pi q when this reads
        # it: the un-rotation writes the output back through the same scratch.
        impl = self._build_impl(2, SINK_TOKENS)
        impl.output_rotation_folded = True
        run = self._run(prompt_lens=prompts, num_kv_heads=2, sink_tokens=SINK_TOKENS, impl=impl)
        block_tables = torch.tensor(_block_table(prompts, 2), dtype=torch.int32)
        seq_lens = torch.tensor(prompts) + 1

        self.assertEqual(impl.sink_lse.numel() % (NUM_HEADS * TURBOQUANT_LSE_STRIDE), 0)
        written = impl.sink_lse[: NUM_HEADS * TURBOQUANT_LSE_STRIDE].view(1, NUM_HEADS, TURBOQUANT_LSE_STRIDE)
        kernel_max, kernel_mass = turboquant_split_lse(written)

        oracle_max, oracle_mass = quantized_context_softmax_stats(
            rotated_query=impl.rotated_query[: NUM_HEADS * HEAD_SIZE].view(1, NUM_HEADS, HEAD_SIZE),
            key_cache=run.kv_cache[0],
            scale_cache=impl.scale_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            num_kv_heads=2,
            num_heads=NUM_HEADS,
            scale_value=HEAD_SIZE**-0.5,
        )
        torch.testing.assert_close(kernel_max, oracle_max, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(kernel_mass, oracle_mass, rtol=1e-5, atol=1e-5)

        # The sink rows were neutralised in the scale plane, so the decode scored each of
        # them at exactly zero -- which is what the merge subtracts, as a known integer.
        rows = torch.tensor(
            [_block_table(prompts, 2)[0][sink // BLOCK_SIZE] * BLOCK_SIZE + sink for sink in range(SINK_TOKENS)]
        )
        scales = impl.scale_cache.view(-1, turboquant_scale_slot(2))[rows]
        self.assertEqual(int(scales.count_nonzero()), 0)
        self.assertGreaterEqual(kernel_mass.min().item(), float(SINK_TOKENS))

        # And the lanes the kernel does not use stay zero, so a reader that took the whole
        # group for data would see nothing plausible in them.
        for lane in range(TURBOQUANT_LSE_STRIDE):
            if lane in (0, TURBOQUANT_LSE_STRIDE // 2):
                continue
            self.assertEqual(int(written[..., lane].count_nonzero()), 0)

    def test_the_host_oracle_zero_extends_a_short_length_vector(self):
        """Defensive: a metadata that describes fewer rows than it runs reads as no context."""
        impl = self._build_impl(2, SINK_TOKENS)
        run = self._run(prompt_lens=(SHORT_PROMPT,), num_kv_heads=2, sink_tokens=SINK_TOKENS, impl=impl)
        running_max, mass = quantized_context_softmax_stats(
            rotated_query=torch.zeros(2, NUM_HEADS, HEAD_SIZE),
            key_cache=run.kv_cache[0],
            scale_cache=impl.scale_cache,
            block_tables=torch.tensor(_block_table((SHORT_PROMPT,), 2) * 2, dtype=torch.int32),
            seq_lens=torch.tensor([SHORT_PROMPT]),
            num_kv_heads=2,
            num_heads=NUM_HEADS,
            scale_value=HEAD_SIZE**-0.5,
        )
        self.assertGreater(mass[0].min().item(), 0.0)
        self.assertEqual(int(mass[1].count_nonzero()), 0)
        self.assertTrue(bool(torch.isfinite(running_max).all()))
        # The host dequantiser agrees with the one the CPU operators use, so the oracle and
        # the kernel it is checked against are reading the same grid.
        packed = run.kv_cache[0].view(-1, 2, HEAD_SIZE // TURBOQUANT_PACK_FACTOR)[:4]
        centroids = torch.tensor(TURBOQUANT_LLOYD_MAX_CENTROIDS, dtype=torch.float32)
        torch.testing.assert_close(turboquant_dequantize(packed), dequantize(packed, centroids))

    def test_the_build_check_reads_the_schema_that_is_actually_registered(self):
        """Otherwise the guard below is a no-op, which is what it was at first.

        The schema lives on the operator's *overload*. ``torch.ops`` hands out an
        ``OpOverloadPacket``, which carries no ``_schema`` at all, so a guard that read one
        off the packet passed on every build there has ever been -- including the ones it
        exists to refuse.
        """
        for name in ("npu_turboquant_paged_attention",):
            schema = tq_module._registered_schema(name)
            with self.subTest(name=name):
                self.assertIsNotNone(schema, f"{name} is registered, so its schema has to be readable")
                self.assertIn(tq_module.TURBOQUANT_LSE_ARGUMENT, [a.name for a in schema.arguments])
        # And a name nothing registered answers nothing rather than an empty argument list.
        self.assertIsNone(tq_module._registered_schema("npu_turboquant_no_such_operator"))

    def test_a_build_whose_decode_cannot_report_its_softmax_is_refused(self):
        """Without the ``lse`` out-tensor there is nothing to merge against.

        The pair cannot be recovered from the operator's output at any number of extra
        launches -- the output is a convex combination of the values, so it is invariant to
        a rescaling of the weights, and the mass is exactly what that rescaling destroys.
        Left to the call it would be a TypeError about an argument count, raised from inside
        a decode step on a configuration that had already allocated a plane per layer.
        """
        stale = SimpleNamespace(
            _schema=torch._C.parse_schema(
                "_C_ascend::npu_turboquant_paged_attention(Tensor query_rot, Tensor! out) -> ()"
            )
        )
        impl = self._build_impl(2, SINK_TOKENS)
        with patch.object(torch.ops._C_ascend, "npu_turboquant_paged_attention", stale):
            for call in (
                lambda: impl.process_weights_after_loading(DTYPE),
                lambda: impl._ensure_sink_cache(self._allocate_kv_cache(2), DTYPE),
            ):
                with self.subTest(call=call), self.assertRaisesRegex(RuntimeError, "optional 'lse' out-tensor"):
                    call()
        self.assertIsNone(impl.sink_cache)

    def test_the_build_check_is_skipped_with_sinks_off(self):
        """``sink_tokens=0`` is the shipping path and must not have acquired a guard."""
        stale = SimpleNamespace(_schema=torch._C.parse_schema("_C_ascend::whatever(Tensor a) -> ()"))
        impl = self._build_impl(2, 0)
        with patch.object(torch.ops._C_ascend, "npu_turboquant_paged_attention", stale):
            impl.process_weights_after_loading(DTYPE)
            impl._ensure_sink_cache(self._allocate_kv_cache(2), DTYPE)
        self.assertIsNone(impl.sink_cache)

    def test_the_cube_decode_is_no_longer_refused(self):
        """The merge reads no packed byte, so the NZ-tiled affine kv4fp8 cache serves it too.

        What used to make this combination wrong was a host reader of the packed planes:
        the merge recomputed the softmax denominator by dequantising the context row-major
        on the Lloyd-Max grid, and the Cube writer leaves neither layout nor codebook.  Both
        halves of that are gone -- the statistics come out of the kernel and the quantised
        sinks leave through the scale plane -- so the guard is now about the *build*, and
        this build has the out-tensor.
        """
        impl = self._build_impl(2, SINK_TOKENS)
        impl.cube_decode = True
        impl.process_weights_after_loading(DTYPE)
        impl._ensure_sink_cache(self._allocate_kv_cache(2), DTYPE)
        self.assertIsNotNone(impl.sink_cache)
        # And the build check follows the decode that will actually run.
        stale = SimpleNamespace(_schema=torch._C.parse_schema("_C_ascend::whatever(Tensor a) -> ()"))
        with (
            patch.object(torch.ops._C_ascend, "npu_turboquant_cube_decode", stale, create=True),
            self.assertRaisesRegex(RuntimeError, "npu_turboquant_cube_decode"),
        ):
            impl.process_weights_after_loading(DTYPE)

    def test_a_sink_plane_the_arena_cannot_hold_is_refused_where_it_is_named(self):
        """Out of memory several layers later names neither the plane nor the flag that asked."""
        impl = self._build_impl(2, SINK_TOKENS)
        kv_cache = self._allocate_kv_cache(2)
        needed = sink_plane_bytes(
            num_blocks=NUM_BLOCKS,
            num_sink_tokens=SINK_TOKENS,
            num_kv_heads=2,
            head_size=HEAD_SIZE,
            element_size=torch.empty((), dtype=DTYPE).element_size(),
        )
        with (
            patch.object(tq_module, "_device_free_memory", return_value=needed),
            self.assertRaisesRegex(RuntimeError, "uncompressed sink plane"),
        ):
            impl._ensure_sink_cache(kv_cache, DTYPE)
        self.assertIsNone(impl.sink_cache)
        # Room to spare is allowed, and a device that answers nothing is not second-guessed.
        with patch.object(tq_module, "_device_free_memory", return_value=None):
            impl._ensure_sink_cache(kv_cache, DTYPE)
        self.assertIsNotNone(impl.sink_cache)

    def test_the_fusion_runs_before_the_output_un_rotation(self):
        """A merge applied after Pi would be a rescaling of the wrong basis.

        ``Pi`` is linear, so the *values* may be merged in either basis, but the merge also
        divides by a new denominator -- and the output the operator wrote is rotated. The
        run below pins the ordering by making the un-rotation observable.
        """
        seen: list[str] = []
        prompts = (SHORT_PROMPT,)
        impl = self._build_impl(2, SINK_TOKENS)
        real_fuse = tq_module.fuse_decode_sinks
        real_unrotate = AscendTurboQuantAttentionBackendImpl._unrotate_output

        def recording_fuse(**kwargs):
            seen.append("fuse")
            return real_fuse(**kwargs)

        def recording_unrotate(self, *args):
            seen.append("unrotate")
            return real_unrotate(self, *args)

        with (
            patch.object(tq_module, "fuse_decode_sinks", recording_fuse),
            patch.object(AscendTurboQuantAttentionBackendImpl, "_unrotate_output", recording_unrotate),
        ):
            self._run(prompt_lens=prompts, num_kv_heads=2, sink_tokens=SINK_TOKENS, impl=impl)
        self.assertEqual(seen, ["fuse", "unrotate"])

    def test_an_empty_request_does_not_shift_the_anchor_blocks(self):
        """The anchor blocks arrive as one row per request, so the lists must stay aligned.

        A request that contributes no tokens -- a zero-length row, or the dummy the FIA TND
        layout appends -- used to be dropped from the ingestion's lists while the block rows
        kept their positions, which stored the second sequence's sinks under the third
        sequence's anchor. Nothing about that failure raises; it reads another sequence's
        prefix at decode.
        """
        impl = self._build_impl(2, SINK_TOKENS)
        impl.sink_cache = TurboQuantSinkCache(
            num_blocks=NUM_BLOCKS,
            block_size=BLOCK_SIZE,
            num_sink_tokens=SINK_TOKENS,
            num_kv_heads=2,
            head_size=HEAD_SIZE,
            dtype=DTYPE,
            device=CPU,
        )
        slot_floats = turboquant_scale_slot(2)
        impl.scale_cache = torch.ones(NUM_BLOCKS, BLOCK_SIZE, slot_floats, dtype=torch.float32)
        generator = torch.Generator().manual_seed(21)
        key = torch.randn(12, 2, HEAD_SIZE, generator=generator).to(DTYPE)
        value = torch.randn(12, 2, HEAD_SIZE, generator=generator).to(DTYPE)
        # Requests of 6, 0 and 6 tokens, anchored at three different blocks.
        metadata = AscendMetadata(
            attn_state=AscendAttentionState.PrefillNoCache,
            num_actual_tokens=12,
            actual_seq_lengths_q=[6, 6, 12],
            seq_lens=torch.tensor([6, 0, 6]),
            seq_lens_list=[6, 0, 6],
            block_tables=torch.tensor([[7, 0], [3, 0], [5, 0]], dtype=torch.int32),
        )
        self.assertEqual(impl._ingest_sinks(key, value, metadata), 2 * SINK_TOKENS)
        torch.testing.assert_close(impl.sink_cache.planes[0, 7], key[:SINK_TOKENS])
        torch.testing.assert_close(impl.sink_cache.planes[0, 5], key[6 : 6 + SINK_TOKENS])
        self.assertEqual(int(impl.sink_cache.planes[:, 3].count_nonzero()), 0)
        # The neutralisation follows the same rows, so the empty request cannot shift it
        # either: the third sequence's block keeps its scales and the other two lose theirs.
        for block in (7, 5):
            self.assertEqual(int(impl.scale_cache[block, :SINK_TOKENS].count_nonzero()), 0)
            self.assertEqual(int((impl.scale_cache[block, SINK_TOKENS:] != 1.0).sum()), 0)
        self.assertEqual(int((impl.scale_cache[3] != 1.0).sum()), 0)

    def test_padding_rows_of_a_batch_are_left_alone(self):
        """A captured shape runs query rows the scheduler never filled.

        The operators want one context length per query row, so a padded batch spells a
        padding row as a sequence of no length rather than as a missing entry. Such a row has
        no sinks, no context and nothing for the merge to say: it has to come back exactly as
        the operator wrote it, not divided by a denominator assembled out of zeros.
        """
        impl = self._build_impl(2, SINK_TOKENS)
        run = self._run(prompt_lens=(SHORT_PROMPT,), num_kv_heads=2, sink_tokens=SINK_TOKENS, impl=impl)
        padded_rows = 3
        query = torch.cat((run.decode_query, torch.zeros(padded_rows, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)))
        block_table = _block_table((SHORT_PROMPT,), 2)
        metadata = AscendMetadata(
            attn_state=AscendAttentionState.DecodeOnly,
            num_actual_tokens=1,
            num_decode_tokens=1,
            num_decodes=1,
            seq_lens=torch.tensor([SHORT_PROMPT + 1] + [0] * padded_rows),
            seq_lens_list=[SHORT_PROMPT + 1] + [0] * padded_rows,
            actual_seq_lengths_q=list(range(1, 2 + padded_rows)),
            query_start_loc=torch.arange(2 + padded_rows),
            max_query_len=1,
            block_tables=torch.tensor(block_table * (1 + padded_rows), dtype=torch.int32),
            slot_mapping=torch.tensor([_slot(block_table, 0, SHORT_PROMPT)]),
        )
        output = torch.zeros(1 + padded_rows, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        impl.forward_paged_attention(query, metadata, output)
        self.assertTrue(bool(torch.isfinite(output.to(torch.float32)).all()))
        self.assertEqual(int(output[1:].count_nonzero()), 0)
        run.decode_output = output[:1]
        self.assertGreater(self._decode_cosine(run, (SHORT_PROMPT,), 2).min().item(), MIN_UNCOMPRESSED_SINK_COSINE)

    def test_a_folded_layer_leaves_the_merged_output_rotated(self):
        """With Pi folded into o_proj the merge is the last thing that touches the output."""
        prompts = (SHORT_PROMPT,)
        impl = self._build_impl(2, SINK_TOKENS)
        impl.output_rotation_folded = True
        run = self._run(prompt_lens=prompts, num_kv_heads=2, sink_tokens=SINK_TOKENS, impl=impl)
        keys = torch.cat((run.prompt_key, run.decode_key))
        values = torch.cat((run.prompt_value, run.decode_value))
        exact = _exact_attention(run.decode_query, keys, values, 2, causal=False)
        rotated = apply_pi(exact, turboquant_pi_signs(HEAD_SIZE, CPU).to(torch.float64))
        self.assertGreater(_cosine(run.decode_output, rotated).min().item(), MIN_UNCOMPRESSED_SINK_COSINE)


if __name__ == "__main__":
    unittest.main()
