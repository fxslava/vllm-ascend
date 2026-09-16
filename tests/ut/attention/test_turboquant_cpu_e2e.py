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
"""Prefill then decode through the TurboQuant backend on a host with no NPU.

:func:`~tests.ut.attention.turboquant_cpu_ops.turboquant_cpu_ops` serves the operators from the
CPU, so everything above them is the code that ships: ``forward``, ``reshape_and_cache``, the
persistent decode buffers, the output un-rotation, ``activate_turboquant_backend``, and the model
runner's view of a packed cache. The stand-ins enforce the adapter's checks and compute what the
device computes, so a run is held to the exact attention over the same inputs rather than to
"nothing raised": a slot mapping that disagrees with the block table decodes another sequence's
keys, and the cosine collapses.

Q, K and V arrive as ``Glm4Attention`` hands them over -- column slices of one fused QKV
projection, viewed per head. None of the three is contiguous, and V never passes through RoPE, so
nothing on the way would pack it.
"""

import os
import re
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.attention.backend import AttentionType  # type: ignore

from tests.ut.attention.turboquant_cpu_ops import (
    CPU_VECTOR_CORES,
    TURBOQUANT_OP_SCHEMAS,
    cpu_fused_infer_attention_score,
    dequantize,
    paged_attention_workspace_floats,
    turboquant_cpu_ops,
)
from tests.ut.base import TestBase
from vllm_ascend.attention import turboquant_v1 as tq_module
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionState, AscendMetadata
from vllm_ascend.attention.turboquant_rotation import (
    TURBOQUANT_OUTPUT_ROTATION_CONFIG_KEY,
    apply_pi,
    output_rotation_marker,
    turboquant_pi_signs,
)
from vllm_ascend.attention.turboquant_v1 import (
    TURBOQUANT_LLOYD_MAX_CENTROIDS,
    TURBOQUANT_PACK_FACTOR,
    TURBOQUANT_TILE_ROWS,
    AscendTurboQuantAttentionBackendImpl,
    activate_turboquant_backend,
    turboquant_codec_tables,
    turboquant_hadamard16,
    turboquant_scale_slot,
)
from vllm_ascend.attention.utils import needs_layer_aware_fia_graph_replay
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

CPU = torch.device("cpu")
DTYPE = torch.bfloat16

NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_SIZE = 128
BLOCK_SIZE = 128
NUM_BLOCKS = 6

# Both prompts cross a block boundary, and their physical blocks interleave, so a slot computed
# from the wrong row or the wrong column lands on the other sequence's tokens rather than on
# unwritten memory. Block 5 is never named: nothing may be written there.
PROMPT_LENS = (131, 260)
BLOCK_TABLE = ((3, 0, 0), (1, 4, 2))
UNUSED_BLOCK = 5
# Padding past num_actual_tokens, as data-parallel padding leaves it.
PAD_TOKENS = 3
PAD_SLOT_ID = -1

LAYER_PREFIX = "model.layers.0.self_attn"
LAYER_NAME = f"{LAYER_PREFIX}.attn"

# bfloat16 rounding of an exact dense prefill.
MIN_PREFILL_COSINE = 0.999
# One 4-bit Lloyd-Max vector against the value it was quantised from.
MIN_CACHE_COSINE = 0.98
# A decode over a whole quantised context.
MIN_DECODE_COSINE = 0.98
# What reading the wrong sequence's blocks has to cost.
MAX_MISALIGNED_COSINE = 0.9

TORCH_BINDING = Path(__file__).resolve().parents[3] / "csrc" / "torch_binding.cpp"
_STRING_LITERAL = r'"(?:[^"\\]|\\.)*"'


def _slot(block_table: tuple[tuple[int, ...], ...], seq: int, position: int) -> int:
    return block_table[seq][position // BLOCK_SIZE] * BLOCK_SIZE + position % BLOCK_SIZE


def _glm4_qkv(num_tokens: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q, K and V as ``Glm4Attention.forward`` splits them and ``Attention`` views them."""
    q_size = NUM_HEADS * HEAD_SIZE
    kv_size = NUM_KV_HEADS * HEAD_SIZE
    qkv = torch.randn(num_tokens, q_size + 2 * kv_size, generator=generator).to(DTYPE)
    query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)
    return (
        query.view(-1, NUM_HEADS, HEAD_SIZE),
        key.view(-1, NUM_KV_HEADS, HEAD_SIZE),
        value.view(-1, NUM_KV_HEADS, HEAD_SIZE),
    )


def _exact_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal: bool) -> torch.Tensor:
    """Dense float64 attention over unquantised K/V, with each kv head shared by a head group."""
    group = NUM_HEADS // NUM_KV_HEADS
    q = query.to(torch.float64)
    k = key.to(torch.float64).repeat_interleave(group, dim=1)
    v = value.to(torch.float64).repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) * HEAD_SIZE**-0.5
    if causal:
        allowed = torch.ones(q.shape[0], k.shape[0], dtype=torch.bool).tril(k.shape[0] - q.shape[0])
        scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v)


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(actual.to(torch.float64), expected.to(torch.float64), dim=-1)


def _prefill_metadata() -> AscendMetadata:
    """Both prompts in one PrefillNoCache step, with padding past the last real token."""
    ends = torch.tensor(PROMPT_LENS).cumsum(dim=0)
    slots = [
        _slot(BLOCK_TABLE, seq, position) for seq, length in enumerate(PROMPT_LENS) for position in range(length)
    ]
    return AscendMetadata(
        attn_state=AscendAttentionState.PrefillNoCache,
        num_actual_tokens=int(ends[-1]),
        num_prefills=len(PROMPT_LENS),
        seq_lens=torch.tensor(PROMPT_LENS),
        seq_lens_list=list(PROMPT_LENS),
        actual_seq_lengths_q=ends.tolist(),
        query_start_loc=torch.cat((torch.zeros(1, dtype=ends.dtype), ends)),
        max_query_len=max(PROMPT_LENS),
        block_tables=torch.tensor(BLOCK_TABLE, dtype=torch.int32),
        slot_mapping=torch.tensor(slots + [PAD_SLOT_ID] * PAD_TOKENS),
    )


def _decode_metadata(block_table: tuple[tuple[int, ...], ...]) -> AscendMetadata:
    """One new token per sequence, at the position right after its prompt.

    The slot mapping always names the true block table: it is what the write path uses, and
    pointing the read somewhere else is exactly the misalignment this file has to catch.
    """
    num_decodes = len(PROMPT_LENS)
    return AscendMetadata(
        attn_state=AscendAttentionState.DecodeOnly,
        num_actual_tokens=num_decodes,
        num_decode_tokens=num_decodes,
        num_decodes=num_decodes,
        seq_lens=torch.tensor(PROMPT_LENS) + 1,
        seq_lens_list=[length + 1 for length in PROMPT_LENS],
        actual_seq_lengths_q=list(range(1, num_decodes + 1)),
        query_start_loc=torch.arange(num_decodes + 1),
        max_query_len=1,
        block_tables=torch.tensor(block_table, dtype=torch.int32),
        slot_mapping=torch.tensor([_slot(BLOCK_TABLE, seq, length) for seq, length in enumerate(PROMPT_LENS)]),
    )


@dataclass
class _Trace:
    """What one prefill-then-decode run consumed and produced."""

    layer: SimpleNamespace
    kv_cache: tuple[torch.Tensor, torch.Tensor]
    prefill_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    prefill_output: torch.Tensor
    decode_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    decode_output: torch.Tensor
    folded: bool


def _in_output_basis(attention: torch.Tensor, folded: bool) -> torch.Tensor:
    """A folded ``o_proj`` un-rotates for itself, so the backend has to leave the output rotated."""
    if not folded:
        return attention
    return apply_pi(attention, turboquant_pi_signs(HEAD_SIZE, CPU).to(torch.float64))


def _prefill_cosine(trace: _Trace) -> torch.Tensor:
    query, key, value = trace.prefill_qkv
    cosines = []
    start = 0
    for length in PROMPT_LENS:
        end = start + length
        exact = _exact_attention(query[start:end], key[start:end], value[start:end], causal=True)
        cosines.append(_cosine(trace.prefill_output[start:end], _in_output_basis(exact, trace.folded)))
        start = end
    return torch.cat(cosines)


def _decode_cosine(trace: _Trace) -> torch.Tensor:
    _, prefill_key, prefill_value = trace.prefill_qkv
    decode_query, decode_key, decode_value = trace.decode_qkv
    cosines = []
    start = 0
    for seq, length in enumerate(PROMPT_LENS):
        end = start + length
        keys = torch.cat((prefill_key[start:end], decode_key[seq : seq + 1]))
        values = torch.cat((prefill_value[start:end], decode_value[seq : seq + 1]))
        exact = _exact_attention(decode_query[seq : seq + 1], keys, values, causal=False)
        cosines.append(_cosine(trace.decode_output[seq : seq + 1], _in_output_basis(exact, trace.folded)))
        start = end
    return torch.cat(cosines)


def _written_rows() -> torch.Tensor:
    """Cache rows the two steps name, prompt tokens then the decode token, per sequence."""
    rows = [
        _slot(BLOCK_TABLE, seq, position)
        for seq, length in enumerate(PROMPT_LENS)
        for position in range(length + 1)
    ]
    return torch.tensor(rows)


def _cache_cosine(trace: _Trace) -> torch.Tensor:
    """How well the packed cache holds ``Pi K`` and ``Pi V`` for every token written to it."""
    _, prefill_key, prefill_value = trace.prefill_qkv
    _, decode_key, decode_value = trace.decode_qkv
    keys, values = [], []
    start = 0
    for seq, length in enumerate(PROMPT_LENS):
        keys += [prefill_key[start : start + length], decode_key[seq : seq + 1]]
        values += [prefill_value[start : start + length], decode_value[seq : seq + 1]]
        start += length
    planes = torch.cat((torch.cat(keys), torch.cat(values)), dim=1)

    rows = _written_rows()
    packed_shape = (-1, NUM_KV_HEADS, HEAD_SIZE // TURBOQUANT_PACK_FACTOR)
    packed = torch.cat(
        (trace.kv_cache[0].view(packed_shape)[rows], trace.kv_cache[1].view(packed_shape)[rows]), dim=1
    )
    scales = trace.layer.impl.scale_cache.view(-1, turboquant_scale_slot(NUM_KV_HEADS))[rows, : 2 * NUM_KV_HEADS]
    centroids = torch.tensor(TURBOQUANT_LLOYD_MAX_CENTROIDS, dtype=torch.float32)
    recovered = dequantize(packed, centroids) * scales.unsqueeze(-1)
    return _cosine(recovered, apply_pi(planes.to(torch.float32), turboquant_pi_signs(HEAD_SIZE, CPU)))


class TestTurboQuantCpuOps(TestBase):
    """The stand-ins: the extension's own schemas, the adapter's strictness, and no leftovers."""

    @unittest.skipUnless(TORCH_BINDING.is_file(), "needs csrc/torch_binding.cpp")
    def test_schemas_match_the_compiled_extension(self):
        """A schema that drifts from csrc would mock an operator the device does not have."""
        source = TORCH_BINDING.read_text(encoding="utf-8")
        compiled = {}
        for literals in re.findall(rf"ops\.def\(\s*((?:{_STRING_LITERAL}\s*)+)\)", source):
            schema = " ".join("".join(re.findall(rf"{_STRING_LITERAL}", literals)).replace('"', "").split())
            if schema.startswith("npu_turboquant_"):
                compiled[schema.partition("(")[0]] = schema
        self.assertEqual(compiled, TURBOQUANT_OP_SCHEMAS)

    def test_kernels_refuse_what_the_adapter_refuses(self):
        num_tokens = 4
        signs = turboquant_pi_signs(HEAD_SIZE, CPU)
        tables = turboquant_codec_tables(HEAD_SIZE, 1, CPU)
        tile_tables = turboquant_codec_tables(HEAD_SIZE, TURBOQUANT_TILE_ROWS, CPU)
        cache = torch.zeros(1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE // TURBOQUANT_PACK_FACTOR, dtype=torch.int8)
        scales = torch.zeros(1, BLOCK_SIZE, turboquant_scale_slot(NUM_KV_HEADS))
        slots = torch.arange(num_tokens, dtype=torch.int32)
        key = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_SIZE).to(DTYPE)
        strided_key = torch.randn(num_tokens, HEAD_SIZE, NUM_KV_HEADS).to(DTYPE).transpose(1, 2)
        strided_query = torch.randn(num_tokens, HEAD_SIZE, NUM_HEADS).to(DTYPE).transpose(1, 2)
        query_rot = torch.zeros(num_tokens, NUM_HEADS, HEAD_SIZE)
        block_tables = torch.zeros(num_tokens, 1, dtype=torch.int32)
        context_lens = torch.ones(num_tokens, dtype=torch.int32)
        workspace = torch.zeros(1 << 16)
        ops = torch.ops._C_ascend

        def write(k, v, kv_cache=cache, slot_mapping=slots):
            ops.npu_turboquant_reshape_and_cache(k, v, kv_cache, kv_cache, scales, slot_mapping, signs, tables)

        def decode(out, codec=tile_tables, tables_=block_tables):
            ops.npu_turboquant_paged_attention(
                query_rot, cache, cache, scales, tables_, context_lens, codec, workspace,
                NUM_KV_HEADS, NUM_HEADS, HEAD_SIZE**-0.5, out,
            )  # fmt: skip

        cases = (
            ("key/value must be contiguous", lambda: write(strided_key, strided_key)),
            ("the packed 4-bit kv cache must be int8", lambda: write(key, key, kv_cache=cache.to(torch.int16))),
            ("slot_mapping must be int32", lambda: write(key, key, slot_mapping=slots.to(torch.int64))),
            ("supports float16 and bfloat16 activations", lambda: write(key.float(), key.float())),
            (
                "query must be contiguous",
                lambda: ops.npu_turboquant_rotate_q(
                    strided_query, signs, tables, turboquant_hadamard16(CPU), query_rot
                ),
            ),
            ("out must be float16 or bfloat16", lambda: decode(torch.zeros(num_tokens, NUM_HEADS, HEAD_SIZE))),
            (
                "codec tables must hold",
                lambda: decode(torch.zeros(num_tokens, NUM_HEADS, HEAD_SIZE, dtype=DTYPE), codec=tables),
            ),
            (
                "block_tables and context_lens must be contiguous",
                lambda: decode(
                    torch.zeros(num_tokens, NUM_HEADS, HEAD_SIZE, dtype=DTYPE),
                    tables_=torch.zeros(num_tokens, 2, dtype=torch.int32)[:, ::2],
                ),
            ),
        )
        with turboquant_cpu_ops():
            for message, call in cases:
                with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, re.escape(message)):
                    call()

    def test_only_a_context_past_the_fused_limit_needs_a_workspace(self):
        """The decode is one launch: a context that fits the fused limit writes its output directly."""
        fused_limit_blocks, block_size = 32, 128
        with turboquant_cpu_ops():
            workspace_size = torch.ops._C_ascend.npu_turboquant_workspace_size
            at_limit = workspace_size(1, NUM_HEADS, HEAD_SIZE, fused_limit_blocks, block_size)
            wide_batch = workspace_size(64, NUM_HEADS, HEAD_SIZE, 1, BLOCK_SIZE)
            past_limit = workspace_size(1, NUM_HEADS, HEAD_SIZE, 2 * fused_limit_blocks, block_size)
        self.assertEqual(at_limit, 0)
        self.assertEqual(wide_batch, 0)
        self.assertGreater(past_limit, 0)
        self.assertEqual(past_limit % (NUM_HEADS * (HEAD_SIZE + 16)), 0)
        self.assertEqual(
            past_limit,
            paged_attention_workspace_floats(
                1, NUM_HEADS, HEAD_SIZE, 2 * fused_limit_blocks, block_size, CPU_VECTOR_CORES
            ),
        )

    def test_leaving_the_block_removes_every_registration(self):
        """Otherwise a later test that mocks torch.ops._C_ascend would inherit these kernels."""
        before = {name: hasattr(torch.ops._C_ascend, name) for name in TURBOQUANT_OP_SCHEMAS}
        with turboquant_cpu_ops():
            self.assertEqual(torch.ops._C_ascend.npu_turboquant_vector_core_num(), CPU_VECTOR_CORES)
        after = {name: hasattr(torch.ops._C_ascend, name) for name in TURBOQUANT_OP_SCHEMAS}
        self.assertEqual(after, before)


class TestTurboQuantCpuEndToEnd(TestBase):
    def setUp(self):
        self.vllm_config = MagicMock()
        self.vllm_config.kv_transfer_config = None
        self.vllm_config.quant_config = None
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

    def _build_layer(self, folded: bool) -> SimpleNamespace:
        """An attention layer as the TurboQuant scheme leaves it once the weights are loaded."""
        hf_config = SimpleNamespace(model_type="glm4")
        if folded:
            setattr(
                hf_config, TURBOQUANT_OUTPUT_ROTATION_CONFIG_KEY, output_rotation_marker([LAYER_PREFIX], HEAD_SIZE)
            )
        self.vllm_config.model_config.hf_config = hf_config
        needs_layer_aware_fia_graph_replay.cache_clear()

        impl = AscendTurboQuantAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_SIZE,
            scale=HEAD_SIZE**-0.5,
            num_kv_heads=NUM_KV_HEADS,
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
        return layer

    @staticmethod
    def _allocate_kv_cache() -> tuple[torch.Tensor, torch.Tensor]:
        """K and V caches viewed the way ``NPUModelRunner`` views its raw per-layer buffers.

        The KV cache spec still sizes a page from the unpacked head size, so each raw buffer holds
        twice what the packed shape needs and the runner keeps the leading half.
        """
        shape = AscendAttentionBackend.get_kv_cache_shape(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        unpacked_bytes = NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE
        caches = tuple(
            NPUModelRunner._view_raw_kv_cache(torch.zeros(unpacked_bytes, dtype=torch.int8), torch.uint8, shape[1:])
            for _ in range(2)
        )
        return caches

    def _run(self, folded: bool, decode_block_table: tuple[tuple[int, ...], ...] = BLOCK_TABLE) -> _Trace:
        layer = self._build_layer(folded)
        kv_cache = self._allocate_kv_cache()
        generator = torch.Generator().manual_seed(0)
        num_prompt_tokens = sum(PROMPT_LENS)

        prefill_qkv = _glm4_qkv(num_prompt_tokens + PAD_TOKENS, generator)
        prefill_output = torch.zeros(num_prompt_tokens + PAD_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        layer.impl.forward(layer, *prefill_qkv, kv_cache, _prefill_metadata(), prefill_output)

        decode_qkv = _glm4_qkv(len(PROMPT_LENS), generator)
        decode_output = torch.zeros(len(PROMPT_LENS), NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        layer.impl.forward(layer, *decode_qkv, kv_cache, _decode_metadata(decode_block_table), decode_output)

        return _Trace(
            layer=layer,
            kv_cache=kv_cache,
            prefill_qkv=prefill_qkv,
            prefill_output=prefill_output,
            decode_qkv=decode_qkv,
            decode_output=decode_output,
            folded=folded,
        )

    def test_a_fused_qkv_split_reaches_the_backend_strided(self):
        """The premise of every run below, and of the contiguous() calls the backend makes."""
        query, key, value = _glm4_qkv(len(PROMPT_LENS), torch.Generator().manual_seed(0))
        self.assertFalse(query.is_contiguous())
        self.assertFalse(key.is_contiguous())
        self.assertFalse(value.is_contiguous())

    def test_the_prefill_packs_the_strided_split_before_the_dense_attention(self):
        """The dense prefill reads the same fused-QKV slices ``reshape_and_cache`` does.

        ``npu_fused_infer_attention_score`` is handed q, k and v straight from the projection,
        so whatever arrives strided has to be packed here as well.  The stand-in computes over
        strides without complaint -- it is ``scaled_dot_product_attention`` underneath -- so the
        layout handed over is asserted directly rather than inferred from the numbers.
        """
        seen: list[tuple[bool, bool, bool]] = []

        def recording_fia(query, key, value, **kwargs):
            seen.append((query.is_contiguous(), key.is_contiguous(), value.is_contiguous()))
            return cpu_fused_infer_attention_score(query, key, value, **kwargs)

        # Folded rotates V through its own operator and unfolded passes it along; both reach
        # the same call, so both have to arrive packed.
        for folded in (False, True):
            with self.subTest(folded=folded):
                seen.clear()
                with patch.object(tq_module.torch_npu, "npu_fused_infer_attention_score", recording_fia, create=True):
                    self._run(folded)
                self.assertEqual(seen, [(True, True, True)])

    def test_prefill_then_decode_track_exact_attention(self):
        for folded in (False, True):
            with self.subTest(folded=folded):
                trace = self._run(folded)
                self.assertIs(trace.layer.impl.output_rotation_folded, folded)
                self.assertGreater(_prefill_cosine(trace).min().item(), MIN_PREFILL_COSINE)
                self.assertGreater(_decode_cosine(trace).min().item(), MIN_DECODE_COSINE)

    def test_the_cache_holds_the_tokens_the_slot_mapping_named(self):
        trace = self._run(folded=False)
        key_cache, value_cache = trace.kv_cache
        self.assertEqual(key_cache.dtype, torch.int8)
        self.assertEqual(
            tuple(key_cache.shape), (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE // TURBOQUANT_PACK_FACTOR)
        )
        self.assertGreater(_cache_cosine(trace).min().item(), MIN_CACHE_COSINE)

        written = torch.zeros(NUM_BLOCKS * BLOCK_SIZE, dtype=torch.bool)
        written[_written_rows()] = True
        scale_rows = trace.layer.impl.scale_cache.view(-1, turboquant_scale_slot(NUM_KV_HEADS))
        # Every named slot carries a positive K and V scale, and nothing else was touched: not the
        # padding tokens, whose slot is -1, and not block 5, which no block table names.
        self.assertTrue(bool((scale_rows[written, : 2 * NUM_KV_HEADS] > 0).all()))
        self.assertEqual(int(scale_rows[~written].count_nonzero()), 0)
        # The scale slot is padded to a whole burst; the padding lanes stay zero.
        self.assertEqual(int(scale_rows[:, 2 * NUM_KV_HEADS :].count_nonzero()), 0)
        for cache in (key_cache, value_cache):
            rows = cache.view(-1, NUM_KV_HEADS, HEAD_SIZE // TURBOQUANT_PACK_FACTOR)
            self.assertEqual(int(rows[~written].count_nonzero()), 0)
            self.assertEqual(int(cache[UNUSED_BLOCK].count_nonzero()), 0)

    def test_a_block_table_that_disagrees_with_the_slot_mapping_is_caught(self):
        """The harness has teeth: decoding through the other sequence's blocks stops tracking."""
        trace = self._run(folded=False, decode_block_table=BLOCK_TABLE[::-1])
        self.assertLess(_decode_cosine(trace).max().item(), MAX_MISALIGNED_COSINE)

    def test_the_decode_reuses_its_scratch_rather_than_allocating(self):
        """Decode must not allocate: under graph capture the replay owns the captured addresses."""
        trace = self._run(folded=False)
        impl = trace.layer.impl
        workspace, rotated_query = impl.decode_workspace, impl.rotated_query
        self.assertIsNotNone(workspace)
        self.assertIsNotNone(rotated_query)

        decode_output = torch.zeros(len(PROMPT_LENS), NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        impl.forward(trace.layer, *trace.decode_qkv, trace.kv_cache, _decode_metadata(BLOCK_TABLE), decode_output)
        self.assertIs(impl.decode_workspace, workspace)
        self.assertIs(impl.rotated_query, rotated_query)
