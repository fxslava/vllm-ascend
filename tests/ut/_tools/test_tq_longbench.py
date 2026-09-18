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
"""The ``tools/tq_longbench`` harness, held to exact attention on a host with no NPU.

``tests/ut/attention/turboquant_cpu_ops.py`` serves ``torch.ops._C_ascend`` from
the CPU, so everything above the operators is the code that ships: the static
pools, the paging arithmetic, the tie point, the Pi round trip and the greedy
loop.  The stand-ins compute what the device computes, so a run is held to the
attention over the same inputs rather than to "nothing raised" -- a block table
that disagrees with the slot mapping decodes another token's keys, and the
cosine collapses.

Two boundaries this file does not cross:

* **The Cube decode has no CPU kernel**, by design -- what that launch computes
  is verified on the camodel (``test_sim_950pr_turboquant_fused``).  What is
  checked here is the host-side half: which output stage a layer selects, and
  which combinations are refused.
* **Retrieval is tested at the cache, not at a checkpoint.**  A needle placed as
  a distinctive K/V pair at a known slot exercises exactly what a 128k NIAH
  prompt would -- the write, the block table, the context length and the decode
  lining up across the whole prefix -- and needs no weights to be unambiguous.

Runs under pytest in CI and under ``python -m unittest`` anywhere, because it
imports neither ``tests.ut.base`` (which reaches vLLM) nor pytest itself.
"""

from __future__ import annotations
import __future__

import contextlib
import ctypes
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

# Appended, never prepended: tools/bisect/ is a package whose name shadows the
# standard library's bisect, and prepending tools/ breaks `import random`.
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "tools"))

from tq_longbench import _ascend, build_turboquant_ops  # noqa: E402
from tq_longbench._ascend import assert_no_vllm_imported, turboquant_layout, turboquant_rotation  # noqa: E402
from tq_longbench.cpu_reference import cpu_turboquant_ops  # noqa: E402
from tq_longbench.engine import RunMetrics, RunnerConfig, StandaloneModelRunner  # noqa: E402
from tq_longbench.glm4 import (  # noqa: E402
    GLM4_BATCHED_DECODE_MIN_TOKENS,
    glm4_checkpoint_targets,
    glm4_model_shape,
    glm4_prefill_mode,
    is_glm4_config,
)
from tq_longbench.kv_cache import CacheGeometry, DenseKVCache, TurboQuantKVCache, dense_equivalent_bytes  # noqa: E402
from tq_longbench.layers import ModelShape, RotaryEmbedding  # noqa: E402
from tq_longbench.ops import (  # noqa: E402
    LayerShape,
    TurboQuantAivBackend,
    TurboQuantCubeBackend,
    ascend_ops,
    build_backend,
    cube_decode_available,
)
from tq_longbench.reference import DenseReferenceBackend, TurboQuantReferenceBackend  # noqa: E402
from tq_longbench.tasks import f1_score, needle_in_a_haystack, rouge_l, score  # noqa: E402

CPU = torch.device("cpu")
DTYPE = torch.bfloat16

NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_SIZE = 128
BLOCK_SIZE = 128

# A needle key this much longer than the distractors clears the log(S) that a
# 128k softmax accumulates over them; below it retrieval is a coin toss and the
# test would measure the margin rather than the plumbing.
NEEDLE_GAIN = 4.0

# Attention over the dequantized cache is the kernel's own arithmetic, so it
# tracks closely; against the original K/V the 4-bit floor sits near 0.99, which
# is quantisation loss rather than error.
EXACT_COSINE = 0.9999
RETRIEVAL_COSINE = 0.99

# The golden bound for the torch reference against the operator stand-ins. The
# two agree to float ordering, not to quantisation: measured 0.99999995 at
# Qwen2.5-3B's geometry (16 heads / 2 kv / head_size 128) over a 4096-token
# context on 2026-09-18. Held well inside that, and orders of magnitude tighter
# than the ~0.99 band 4 bits sit in, so it gates the implementation rather than
# the codec.
REFERENCE_FIDELITY_COSINE = 0.999999

# Qwen2.5-3B-Instruct's attention geometry, as the CUDA smoke test runs it.
QWEN25_3B_HEADS = 16
QWEN25_3B_KV_HEADS = 2
QWEN25_3B_HEAD_SIZE = 128


def cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    left = actual.flatten().to(torch.float64)
    right = expected.flatten().to(torch.float64)
    return float(torch.dot(left, right) / (left.norm() * right.norm()))


def _geometry(max_seq_len: int, num_kv_heads: int = NUM_KV_HEADS, head_size: int = HEAD_SIZE) -> CacheGeometry:
    return CacheGeometry(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=BLOCK_SIZE,
        max_seq_len=max_seq_len,
    )


class TestSharedLayout(unittest.TestCase):
    """The harness must build the plugin's tables, not its own."""

    def test_no_vllm_is_imported(self):
        assert_no_vllm_imported()

    def test_the_codec_tables_are_the_shipped_ones(self):
        # Loaded twice by two different routes: the harness's file-path borrow,
        # and a direct read of the module the plugin imports. A drift between
        # them would not raise anywhere -- it would decode the cache against the
        # wrong centroids -- so it is asserted rather than assumed.
        import importlib.util

        path = REPO_ROOT / "vllm_ascend" / "attention" / "turboquant_layout.py"
        spec = importlib.util.spec_from_file_location("_shipped_turboquant_layout", path)
        shipped = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shipped)

        borrowed = turboquant_layout()
        for head_size in (64, 128, 256):
            for rows in (1, borrowed.TURBOQUANT_TILE_ROWS):
                self.assertTrue(
                    torch.equal(
                        borrowed.turboquant_codec_tables(head_size, rows, CPU),
                        shipped.turboquant_codec_tables(head_size, rows, CPU),
                    ),
                    f"codec tables differ at head_size={head_size}, batch_rows={rows}",
                )
        self.assertTrue(torch.equal(borrowed.turboquant_hadamard16(CPU), shipped.turboquant_hadamard16(CPU)))
        self.assertEqual(borrowed.TURBOQUANT_LLOYD_MAX_CENTROIDS, shipped.TURBOQUANT_LLOYD_MAX_CENTROIDS)

    def test_pi_is_an_involution(self):
        rotation = turboquant_rotation()
        signs = rotation.turboquant_pi_signs(HEAD_SIZE, CPU)
        x = torch.randn(4, HEAD_SIZE, dtype=torch.float32)
        self.assertLess((rotation.apply_pi(rotation.apply_pi(x, signs), signs) - x).abs().max().item(), 1e-5)


class TestStaticKVCache(unittest.TestCase):
    def test_the_packed_pool_is_shaped_as_the_backend_budgets_it(self):
        geometry = _geometry(4096)
        cache = TurboQuantKVCache(geometry, CPU)
        packed = (geometry.num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE // 2)
        self.assertEqual(tuple(cache.key_planes[0].shape), packed)
        self.assertEqual(cache.key_planes[0].dtype, torch.int8)
        self.assertEqual(tuple(cache.scale_planes[0].shape), (geometry.num_blocks, BLOCK_SIZE, geometry.scale_slot))
        self.assertEqual(cache.scale_planes[0].dtype, torch.float32)
        # round_up(2 * num_kv_heads, 8)
        self.assertEqual(geometry.scale_slot, 8)

    def test_the_saving_is_the_one_the_layout_implies(self):
        geometry = _geometry(4096, num_kv_heads=8)
        packed = TurboQuantKVCache(geometry, CPU)
        dense = DenseKVCache(geometry, CPU, torch.float16)
        # 2 planes * 8 heads * 64 packed bytes + 16 fp32 scale lanes = 1088 B;
        # dense is 2 * 8 * 128 * 2 = 4096 B.
        self.assertEqual(packed.bytes_per_token_per_layer(), 1088.0)
        self.assertEqual(dense.bytes_per_token_per_layer(), 4096.0)
        self.assertEqual(dense_equivalent_bytes(geometry, torch.float16), dense.bytes_allocated())

    def test_the_block_table_narrows_to_the_context(self):
        cache = TurboQuantKVCache(_geometry(4096), CPU)
        self.assertEqual(cache.block_table(1).shape[1], 1)
        self.assertEqual(cache.block_table(300).shape[1], 3)
        self.assertEqual(cache.block_table(4096).shape[1], 32)
        self.assertEqual(cache.block_table(300).flatten().tolist(), [0, 1, 2])
        self.assertEqual(cache.slot_mapping(128, 4).tolist(), [128, 129, 130, 131])
        self.assertEqual(cache.slot_mapping(0, 1).dtype, torch.int32)

    def test_a_geometry_the_kernels_cannot_read_is_refused(self):
        with self.assertRaisesRegex(ValueError, "power-of-two head_size"):
            _geometry(128, head_size=96)
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            CacheGeometry(1, NUM_KV_HEADS, HEAD_SIZE, 24, 128)
        with self.assertRaisesRegex(ValueError, "64-row tiles"):
            CacheGeometry(1, NUM_KV_HEADS, HEAD_SIZE, 32, 128).require_cube_tiling()
        cache = TurboQuantKVCache(_geometry(256), CPU)
        with self.assertRaisesRegex(ValueError, "outside"):
            cache.slot_mapping(0, 300)
        with self.assertRaisesRegex(ValueError, "outside"):
            cache.block_table(300)


class _TurboQuantCase(unittest.TestCase):
    """Shared setup: the CPU operators, live for the duration of each test."""

    def setUp(self):
        self._ops = cpu_turboquant_ops()
        self.stand_ins = self._ops.__enter__()
        self.addCleanup(self._ops.__exit__, None, None, None)
        torch.manual_seed(0)

    def backend(
        self, max_seq_len: int, num_kv_heads: int = NUM_KV_HEADS, head_size: int = HEAD_SIZE, num_heads: int = NUM_HEADS
    ) -> TurboQuantAivBackend:
        geometry = _geometry(max_seq_len, num_kv_heads, head_size)
        shape = LayerShape(num_heads, num_kv_heads, head_size, head_size**-0.5)
        return TurboQuantAivBackend(geometry, shape, CPU)

    def attention_over_the_cache(self, backend, query: torch.Tensor, context_len: int) -> torch.Tensor:
        """Exact attention over what the cache actually holds, in the output's basis."""
        layout, rotation = turboquant_layout(), turboquant_rotation()
        centroids = torch.tensor(layout.TURBOQUANT_LLOYD_MAX_CENTROIDS, dtype=torch.float32)
        heads = backend.shape
        key_plane, value_plane, scale_plane = backend.cache.planes(0)
        rows = torch.arange(context_len)
        packed = (-1, heads.num_kv_heads, heads.head_size // 2)
        scales = scale_plane.view(-1, scale_plane.shape[-1])[rows]
        keys = self.stand_ins.dequantize(key_plane.view(packed)[rows], centroids) * scales[
            :, : heads.num_kv_heads
        ].unsqueeze(-1)
        values = self.stand_ins.dequantize(value_plane.view(packed)[rows], centroids) * scales[
            :, heads.num_kv_heads : 2 * heads.num_kv_heads
        ].unsqueeze(-1)
        group = heads.num_heads // heads.num_kv_heads
        keys = keys.repeat_interleave(group, dim=1)
        values = values.repeat_interleave(group, dim=1)
        signs = rotation.turboquant_pi_signs(heads.head_size, CPU)
        # The cache holds Pi k and Pi v, so the query enters the rotated basis
        # and the result leaves it -- Pi is an involution, so both are apply_pi.
        rotated_query = rotation.apply_pi(query.to(torch.float32), signs)
        scores = torch.einsum("thd,lhd->thl", rotated_query, keys) * heads.scale
        attended = torch.einsum("thl,lhd->thd", torch.softmax(scores, dim=-1), values)
        return rotation.apply_pi(attended, signs)


class TestTurboQuantDecode(_TurboQuantCase):
    def test_the_decode_tracks_attention_over_the_cache_it_read(self):
        length = 512
        backend = self.backend(length)
        key = torch.randn(length, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE)
        value = torch.randn(length, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE)
        for start in range(0, length, 256):
            backend.write_kv(
                0, key[start : start + 256], value[start : start + 256], backend.cache.slot_mapping(start, 256)
            )

        query = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        out = torch.empty(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        backend.decode(0, query, torch.full((1,), length, dtype=torch.int32), out)
        self.assertGreater(cosine(out, self.attention_over_the_cache(backend, query, length)), EXACT_COSINE)

    def test_chunking_the_write_does_not_change_the_cache(self):
        """A chunked prefill and a single-shot one must leave identical bytes.

        This is the alignment the NIAH gate depends on: if a chunk boundary
        shifted a slot, the decode would still run and still return numbers.
        """
        length = 512
        key = torch.randn(length, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE)
        value = torch.randn(length, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE)

        whole = self.backend(length)
        whole.write_kv(0, key, value, whole.cache.slot_mapping(0, length))
        for chunk in (64, 128, 256):
            pieces = self.backend(length)
            for start in range(0, length, chunk):
                count = min(chunk, length - start)
                pieces.write_kv(
                    0, key[start : start + count], value[start : start + count], pieces.cache.slot_mapping(start, count)
                )
            for left, right in zip(whole.cache.planes(0), pieces.cache.planes(0)):
                self.assertTrue(torch.equal(left, right), f"a {chunk}-token chunking changed the cache")

    def test_the_tie_point_refuses_a_context_that_is_not_the_prefix(self):
        backend = self.backend(512)
        backend.check_tie_point(torch.full((1,), 512, dtype=torch.int32), 512)
        for wrong in (511, 513):
            with self.assertRaisesRegex(RuntimeError, "tie-point"):
                backend.check_tie_point(torch.full((1,), wrong, dtype=torch.int32), 512)

    def test_a_prefill_chunk_is_bottom_right_causal(self):
        """Query token i must see exactly the prefix through its own position."""
        length, chunk = 256, 64
        backend = self.backend(length)
        key = torch.randn(length, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE)
        value = torch.randn(length, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE)
        backend.write_kv(0, key, value, backend.cache.slot_mapping(0, length))

        query = torch.randn(chunk, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        out = torch.empty(chunk, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        backend.prefill_chunk(0, query, chunk, out)
        for row in (0, chunk // 2, chunk - 1):
            single = torch.empty(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
            backend.decode(0, query[row : row + 1], torch.full((1,), row + 1, dtype=torch.int32), single)
            self.assertTrue(torch.equal(out[row], single[0]), f"row {row} of the chunk is not its own decode")


class TestNeedleInAHaystack(_TurboQuantCase):
    """Retrieval across a full-length prefix, at the cache rather than at a checkpoint."""

    HEAD_SIZE = 64

    def _haystack(self, length: int, depth: float):
        backend = self.backend(length, num_kv_heads=1, head_size=self.HEAD_SIZE, num_heads=1)
        position = min(length - 1, int(length * depth))
        direction = torch.randn(self.HEAD_SIZE)
        direction = direction / direction.norm() * (self.HEAD_SIZE**0.5)
        needle = torch.randn(self.HEAD_SIZE) * 3.0

        for start in range(0, length, 4096):
            count = min(4096, length - start)
            key = torch.randn(count, 1, self.HEAD_SIZE)
            value = torch.randn(count, 1, self.HEAD_SIZE)
            if start <= position < start + count:
                key[position - start, 0] = direction * NEEDLE_GAIN
                value[position - start, 0] = needle
            backend.write_kv(0, key.to(DTYPE), value.to(DTYPE), backend.cache.slot_mapping(start, count))
        query = direction.view(1, 1, self.HEAD_SIZE).to(DTYPE)
        return backend, query, needle, position

    def _retrieve(self, backend, query, context_len):
        out = torch.empty(1, 1, self.HEAD_SIZE, dtype=DTYPE)
        backend.check_tie_point(torch.full((1,), context_len, dtype=torch.int32), context_len)
        backend.decode(0, query, torch.full((1,), context_len, dtype=torch.int32), out)
        return out

    def test_retrieval_at_32k(self):
        for depth in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(depth=depth):
                backend, query, needle, _ = self._haystack(32768, depth)
                found = self._retrieve(backend, query, 32768)
                self.assertGreater(cosine(found, needle), RETRIEVAL_COSINE)

    def test_retrieval_at_128k(self):
        for depth in (0.0, 0.5, 1.0):
            with self.subTest(depth=depth):
                backend, query, needle, _ = self._haystack(131072, depth)
                found = self._retrieve(backend, query, 131072)
                self.assertGreater(cosine(found, needle), RETRIEVAL_COSINE)

    def test_the_decode_returns_what_the_cache_holds_not_what_was_written(self):
        """The needle comes back un-rotated and dequantized, to the kernel's own precision."""
        backend, query, needle, position = self._haystack(32768, 0.5)
        found = self._retrieve(backend, query, 32768)
        layout, rotation = turboquant_layout(), turboquant_rotation()
        centroids = torch.tensor(layout.TURBOQUANT_LLOYD_MAX_CENTROIDS, dtype=torch.float32)
        _, value_plane, scale_plane = backend.cache.planes(0)
        stored = self.stand_ins.dequantize(value_plane.view(-1, 1, self.HEAD_SIZE // 2)[position], centroids)[0]
        stored = stored * scale_plane.view(-1, scale_plane.shape[-1])[position, 1]
        stored = rotation.apply_pi(stored, rotation.turboquant_pi_signs(self.HEAD_SIZE, CPU))
        self.assertGreater(cosine(found, stored), EXACT_COSINE)

    def test_a_context_that_stops_before_the_needle_loses_it(self):
        """The negative control: without it the retrieval assertions could not fail."""
        backend, query, needle, position = self._haystack(32768, 0.5)
        truncated = torch.empty(1, 1, self.HEAD_SIZE, dtype=DTYPE)
        backend.decode(0, query, torch.full((1,), position, dtype=torch.int32), truncated)
        self.assertLess(cosine(truncated, needle), 0.5)


class TestFullStack(_TurboQuantCase):
    """Prefill and decode through the whole model, against the same weights run eagerly."""

    SHAPE = ModelShape(
        num_layers=2,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        hidden_size=256,
        intermediate_size=512,
        vocab_size=512,
        rms_norm_eps=1e-6,
        rope_theta=1.0e6,
        tie_word_embeddings=False,
        attn_output_gate=False,
    )
    PROMPT_TOKENS = 300
    NEW_TOKENS = 8

    def _runner(self, shape: ModelShape, **overrides) -> StandaloneModelRunner:
        config = RunnerConfig(
            model_path="<random weights>",
            backend="turboquant_aiv",
            prefill_mode="batched_decode",
            max_seq_len=1024,
            chunk_size=128,
            block_size=BLOCK_SIZE,
            dtype=DTYPE,
            device="cpu",
            max_new_tokens=self.NEW_TOKENS,
            random_weights=True,
            attn_output_gate=shape.attn_output_gate,
            **overrides,
        )
        return StandaloneModelRunner(config, model_shape=shape)

    def _eager(self, model, shape: ModelShape, token_ids: torch.Tensor, new_tokens: int) -> list[int]:
        """The same modules with full-precision attention and no cache at all."""
        produced: list[int] = []
        ids = token_ids.clone()
        group = shape.num_heads // shape.num_kv_heads
        for _ in range(new_tokens):
            count = ids.numel()
            positions = torch.arange(count, dtype=torch.int64)
            hidden = model.embed_tokens(ids)
            for layer in model.layers:
                normed = layer.input_layernorm(hidden)
                attn = layer.self_attn
                query = attn.q_proj(normed)
                gate = None
                if shape.attn_output_gate:
                    query, gate = query.chunk(2, dim=-1)
                    gate = gate.view(count, shape.num_heads, shape.head_size)
                query = attn.q_norm(query.view(count, shape.num_heads, shape.head_size))
                key = attn.k_norm(attn.k_proj(normed).view(count, shape.num_kv_heads, shape.head_size))
                value = attn.v_proj(normed).view(count, shape.num_kv_heads, shape.head_size)
                query = model.rope.apply(query, positions)
                key = model.rope.apply(key, positions)
                attended = (
                    torch.nn.functional.scaled_dot_product_attention(
                        query.to(torch.float32).transpose(0, 1),
                        key.to(torch.float32).transpose(0, 1).repeat_interleave(group, dim=0),
                        value.to(torch.float32).transpose(0, 1).repeat_interleave(group, dim=0),
                        is_causal=True,
                        scale=shape.scale,
                    )
                    .transpose(0, 1)
                    .to(hidden.dtype)
                )
                if gate is not None:
                    attended = attended * torch.sigmoid(gate)
                hidden = hidden + attn.o_proj(attended.reshape(count, -1))
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            produced.append(int(torch.argmax(model.lm_head(model.norm(hidden))[-1])))
            ids = torch.cat((ids, torch.tensor([produced[-1]], dtype=torch.int64)))
        return produced

    def _agreement(self, shape: ModelShape) -> tuple[list[int], list[int]]:
        runner = self._runner(shape)
        token_ids = torch.randint(0, shape.vocab_size, (self.PROMPT_TOKENS,), dtype=torch.int64)
        produced, _ = runner.generate(token_ids, max_new_tokens=self.NEW_TOKENS)
        return produced, self._eager(runner.model, shape, token_ids, self.NEW_TOKENS)

    def test_generation_matches_eager_pytorch(self):
        produced, reference = self._agreement(self.SHAPE)
        self.assertEqual(produced, reference)

    def test_a_gated_layer_matches_eager_pytorch(self):
        """Qwen3.5's [query | gate] projection, with the AIV path applying the sigmoid in torch."""
        gated = ModelShape(**{**self.SHAPE.__dict__, "attn_output_gate": True})
        produced, reference = self._agreement(gated)
        self.assertEqual(produced, reference)

    def test_the_run_reports_what_it_allocated(self):
        runner = self._runner(self.SHAPE)
        token_ids = torch.randint(0, self.SHAPE.vocab_size, (self.PROMPT_TOKENS,), dtype=torch.int64)
        _, metrics = runner.generate(token_ids, max_new_tokens=self.NEW_TOKENS)
        summary = metrics.summary()
        self.assertEqual(summary["prompt_tokens"], self.PROMPT_TOKENS)
        self.assertEqual(summary["generated_tokens"], self.NEW_TOKENS)
        self.assertGreater(summary["kv_saved_mb"], 0.0)
        self.assertGreater(summary["prefill_tokens_per_second"], 0.0)
        # Percentiles are ordered by construction; a regression here means the
        # latency figures a sweep reports are not percentiles at all.
        self.assertLessEqual(summary["decode_p50_us"], summary["decode_p90_us"])
        self.assertLessEqual(summary["decode_p90_us"], summary["decode_p99_us"])

    def test_batched_decode_never_allocates_the_dense_pool(self):
        runner = self._runner(self.SHAPE)
        self.assertIs(runner.prefill_backend, runner.decode_backend)
        # Written once, not twice: a second write would requantise the first.
        self.assertEqual(len(runner.write_backends), 1)

    def test_native_v5_cannot_decode_out_of_a_quantised_cache(self):
        with self.assertRaisesRegex(ValueError, "dense_staging"):
            RunnerConfig(model_path="<random weights>", backend="native_v5", prefill_mode="batched_decode")


class TestCubeDispatch(unittest.TestCase):
    """The Cube decode's host-side half. Its arithmetic is the camodel gate's job."""

    def _backend(self, folded: bool) -> TurboQuantCubeBackend:
        geometry = _geometry(512)
        shape = LayerShape(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE**-0.5)
        return TurboQuantCubeBackend(geometry, shape, CPU, output_rotation_folded=folded)

    def test_the_output_stage_follows_the_layer(self):
        stages = turboquant_layout().TurboQuantOutputStage
        with cpu_turboquant_ops() as stand_ins, stand_ins.turboquant_cube_meta_ops():
            unfolded, folded = self._backend(False), self._backend(True)
            gate = torch.zeros(1, NUM_HEADS, HEAD_SIZE)
            self.assertEqual(unfolded._output_stage(None), stages.UNROTATED)
            self.assertEqual(unfolded._output_stage(gate), stages.GATED)
            self.assertEqual(folded._output_stage(None), stages.ROTATED_BASIS)
            # A folded layer with a gate is a contradiction, not a fallback:
            # the gate sits where the output is still rotated.
            with self.assertRaisesRegex(ValueError, "folded"):
                folded._output_stage(gate)
            self.assertTrue(unfolded.fuses_output_gate)
            self.assertFalse(folded.fuses_output_gate)

    def test_the_cube_decode_is_float16_only(self):
        geometry = _geometry(512)
        shape = LayerShape(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE**-0.5)
        with self.assertRaisesRegex(ValueError, "float16 only"):
            build_backend("turboquant_cube", geometry, shape, CPU, torch.bfloat16)

    def test_an_unbuilt_cube_decode_says_so(self):
        geometry = _geometry(512)
        shape = LayerShape(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE**-0.5)
        # Outside turboquant_cube_meta_ops the operator is not registered at all, and
        # no standalone library is on offer to load it from.
        missing = str(REPO_ROOT / "build" / "no-such-dir" / _ascend.TURBOQUANT_LIB_NAME)
        no_library = mock.patch.dict(os.environ, {_ascend.TURBOQUANT_LIB_ENV: missing})
        with no_library, self.assertRaisesRegex(RuntimeError, "not registered"):
            TurboQuantCubeBackend(geometry, shape, CPU)

    def test_an_ascend_backend_is_refused_with_no_operators_registered(self):
        """Nothing is serving them here, so a CPU run cannot have them either."""
        geometry = _geometry(512)
        shape = LayerShape(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE**-0.5)
        with self.assertRaisesRegex(ValueError, "cannot run on cpu"):
            build_backend("turboquant_aiv", geometry, shape, CPU, torch.float16)


class TestTaskScoring(unittest.TestCase):
    def test_f1_and_rouge_agree_on_the_easy_cases(self):
        self.assertEqual(f1_score("the answer is blue", "The Answer Is Blue"), 1.0)
        self.assertEqual(f1_score("blue", "red"), 0.0)
        self.assertEqual(rouge_l("a b c d", "a b c d"), 1.0)
        self.assertGreater(rouge_l("a x b y c", "a b c"), 0.5)

    def test_a_score_takes_the_best_reference(self):
        self.assertEqual(score("Paris", ["London", "Paris"], "f1"), 1.0)
        self.assertEqual(score("the code is 12345", ["12345"], "substring_match"), 1.0)
        self.assertEqual(score("no idea", ["12345"], "substring_match"), 0.0)

    def test_the_needle_is_where_the_depth_puts_it(self):
        for depth in (0.0, 0.5, 1.0):
            item = needle_in_a_haystack(2048, depth=depth)
            self.assertEqual(len(item.answers), 1)
            body = item.prompt[: item.prompt.index("Question:")]
            found = body.index(item.answers[0])
            self.assertAlmostEqual(found / len(body), depth, delta=0.15)


class TestReferenceMatchesTheStandIns(_TurboQuantCase):
    """The torch backends carry the whole correctness argument for a run with no NPU.

    ``turboquant_cpu_ops`` is held to the kernels' own goldens; this holds
    :mod:`tq_longbench.reference` to *it*.  Without that link a number produced
    on CUDA would be a number produced by a second, unvalidated implementation
    of the codec, which is exactly the drift the shared layout module exists to
    prevent on the host side.
    """

    LENGTH = 512

    def _both(self):
        geometry = _geometry(self.LENGTH)
        shape = LayerShape(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE**-0.5)
        return TurboQuantAivBackend(geometry, shape, CPU), TurboQuantReferenceBackend(geometry, shape, CPU)

    def _fill(self, *backends):
        key = torch.randn(self.LENGTH, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE)
        value = torch.randn(self.LENGTH, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE)
        for start in range(0, self.LENGTH, 128):
            for backend in backends:
                backend.write_kv(
                    0, key[start : start + 128], value[start : start + 128], backend.cache.slot_mapping(start, 128)
                )
        return key, value

    def test_the_packed_planes_are_byte_identical(self):
        """Same rotation, same RMS scale, same Lloyd-Max bins, same nibble packing."""
        kernel, reference = self._both()
        self._fill(kernel, reference)
        for name, left, right in zip(("key", "value", "scale"), kernel.cache.planes(0), reference.cache.planes(0)):
            self.assertTrue(torch.equal(left, right), f"the {name} plane differs from the stand-in's")

    def test_the_decode_agrees_at_uniform_and_ragged_contexts(self):
        kernel, reference = self._both()
        self._fill(kernel, reference)
        query = torch.randn(4, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        for label, lengths in (("uniform", [self.LENGTH] * 4), ("ragged", [1, 129, 384, self.LENGTH])):
            with self.subTest(contexts=label):
                context_lens = torch.tensor(lengths, dtype=torch.int32)
                from_kernel = torch.empty(4, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
                from_reference = torch.empty(4, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
                kernel.decode(0, query, context_lens, from_kernel)
                reference.decode(0, query, context_lens, from_reference)
                self.assertGreater(cosine(from_reference, from_kernel), EXACT_COSINE)

    def test_the_streaming_softmax_does_not_depend_on_the_window(self):
        """A 128k context is reduced window by window; the answer must not know that.

        The running max, denominator and numerator are the whole reason a long
        context never materialises a long score matrix, and a rescaling bug
        there would show up only at lengths the quick tests do not reach.
        """
        import tq_longbench.reference as reference_module

        _, reference = self._both()
        self._fill(reference)
        query = torch.randn(2, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        context_lens = torch.tensor([self.LENGTH, self.LENGTH // 3], dtype=torch.int32)

        wide = torch.empty(2, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        reference.decode(0, query, context_lens, wide)
        saved = reference_module._CONTEXT_WINDOW
        try:
            reference_module._CONTEXT_WINDOW = 64
            narrow = torch.empty(2, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
            reference.decode(0, query, context_lens, narrow)
        finally:
            reference_module._CONTEXT_WINDOW = saved
        self.assertGreater(cosine(narrow, wide), 0.99999)

    def test_the_dense_reference_is_exact_attention(self):
        geometry = _geometry(self.LENGTH)
        shape = LayerShape(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE**-0.5)
        dense = DenseReferenceBackend(geometry, shape, CPU, DTYPE)
        key, value = self._fill(dense)
        query = torch.randn(2, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        out = torch.empty(2, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        dense.decode(0, query, torch.tensor([self.LENGTH] * 2, dtype=torch.int32), out)

        group = NUM_HEADS // NUM_KV_HEADS
        expected = torch.nn.functional.scaled_dot_product_attention(
            query.to(torch.float32).transpose(0, 1),
            key.to(torch.float32).transpose(0, 1).repeat_interleave(group, dim=0),
            value.to(torch.float32).transpose(0, 1).repeat_interleave(group, dim=0),
            scale=shape.scale,
        ).transpose(0, 1)
        self.assertGreater(cosine(out, expected), EXACT_COSINE)

    def test_the_golden_bound_holds_at_a_real_model_geometry(self):
        """The bound the CUDA smoke test's numbers rest on, at the geometry it runs.

        The short cases above use 8 heads and a 512-token context. A real decode
        is 16 heads over thousands of positions, which is where the streaming
        softmax actually streams and where the group broadcast is widest --
        neither of which the short cases reach. Recorded as a golden bound
        because every claim made from a CUDA run is a claim about this number.
        """
        length = 4096
        geometry = CacheGeometry(
            num_layers=1,
            num_kv_heads=QWEN25_3B_KV_HEADS,
            head_size=QWEN25_3B_HEAD_SIZE,
            block_size=BLOCK_SIZE,
            max_seq_len=length,
        )
        shape = LayerShape(QWEN25_3B_HEADS, QWEN25_3B_KV_HEADS, QWEN25_3B_HEAD_SIZE, QWEN25_3B_HEAD_SIZE**-0.5)
        kernel = TurboQuantAivBackend(geometry, shape, CPU)
        reference = TurboQuantReferenceBackend(geometry, shape, CPU)
        key = torch.randn(length, QWEN25_3B_KV_HEADS, QWEN25_3B_HEAD_SIZE, dtype=torch.float16)
        value = torch.randn(length, QWEN25_3B_KV_HEADS, QWEN25_3B_HEAD_SIZE, dtype=torch.float16)
        for start in range(0, length, 1024):
            for backend in (kernel, reference):
                backend.write_kv(
                    0,
                    key[start : start + 1024],
                    value[start : start + 1024],
                    backend.cache.slot_mapping(start, 1024),
                )
        for name, left, right in zip(("key", "value", "scale"), kernel.cache.planes(0), reference.cache.planes(0)):
            self.assertTrue(torch.equal(left, right), f"the {name} plane differs at the real geometry")

        query = torch.randn(3, QWEN25_3B_HEADS, QWEN25_3B_HEAD_SIZE, dtype=torch.float16)
        for label, lengths in (("uniform", [length] * 3), ("ragged", [1, 2049, length])):
            with self.subTest(contexts=label):
                context_lens = torch.tensor(lengths, dtype=torch.int32)
                from_kernel = torch.empty(3, QWEN25_3B_HEADS, QWEN25_3B_HEAD_SIZE, dtype=torch.float16)
                from_reference = torch.empty_like(from_kernel)
                kernel.decode(0, query, context_lens, from_kernel)
                reference.decode(0, query, context_lens, from_reference)
                self.assertGreater(cosine(from_reference, from_kernel), REFERENCE_FIDELITY_COSINE)

    def test_an_ascend_backend_is_refused_where_its_operators_cannot_run(self):
        """Refused, not quietly swapped: a latency table torch produced would be a lie.

        A CUDA device can never serve these operators, so the refusal holds even
        with the CPU stand-ins live -- which they are, inside this case.
        """
        geometry = _geometry(512)
        shape = LayerShape(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE**-0.5)
        with self.assertRaisesRegex(ValueError, "cannot run on cuda"):
            build_backend("turboquant_cube", geometry, shape, torch.device("cuda"), torch.float16)
        # The CPU is allowed only because the stand-ins are serving them here.
        self.assertIsInstance(build_backend("turboquant_aiv", geometry, shape, CPU, DTYPE), TurboQuantAivBackend)


@unittest.skipUnless(torch.cuda.is_available(), "no CUDA device")
class TestReferenceOnCuda(unittest.TestCase):
    """The same arithmetic, on a device whose float ordering is not the CPU's."""

    LENGTH = 1024

    def test_cuda_and_cpu_agree(self):
        torch.manual_seed(0)
        geometry = _geometry(self.LENGTH)
        shape = LayerShape(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE**-0.5)
        key = torch.randn(self.LENGTH, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16)
        value = torch.randn(self.LENGTH, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16)
        query = torch.randn(4, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
        context_lens = torch.tensor([1, 300, 777, self.LENGTH], dtype=torch.int32)

        answers = {}
        for name in ("cpu", "cuda:0"):
            device = torch.device(name)
            backend = TurboQuantReferenceBackend(geometry, shape, device)
            backend.write_kv(0, key.to(device), value.to(device), backend.cache.slot_mapping(0, self.LENGTH))
            out = torch.empty(4, NUM_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
            backend.decode(0, query.to(device), context_lens.to(device), out)
            answers[name] = out.cpu()
            for plane in backend.cache.planes(0):
                self.assertEqual(plane.device.type, device.type)
                self.assertTrue(plane.is_contiguous())
        self.assertGreater(cosine(answers["cuda:0"], answers["cpu"]), 0.99999)


class _FakeAttention(torch.nn.Module):
    """The structural shape :func:`discover_full_attention_layers` looks for."""

    def __init__(self, config, layer_idx: int, heads: int, kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        hidden = heads * head_dim
        self.q_proj = torch.nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(heads * head_dim, hidden, bias=False)


class _FakeLinearAttention(torch.nn.Module):
    """Gated DeltaNet's shape: a fused qkv projection, no o_proj, no KV cache."""

    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.in_proj_qkv = torch.nn.Linear(8, 8, bias=False)
        self.conv1d = torch.nn.Conv1d(8, 8, 4, groups=8)


class _FakeHybridModel(torch.nn.Module):
    """A hybrid stack: full attention every fourth layer, as Qwen3.5 is."""

    def __init__(self, layers: int = 8, interval: int = 4) -> None:
        super().__init__()
        from types import SimpleNamespace

        # One config object shared by every layer, which is what transformers
        # does and what the install/uninstall regression below turns on.
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.layers = torch.nn.ModuleList(
            [
                _FakeAttention(self.config, index, NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE)
                if index % interval == interval - 1
                else _FakeLinearAttention(index)
                for index in range(layers)
            ]
        )


def _has_attention_registry() -> bool:
    """Whether the installed ``transformers`` has a registry the bridge can install into.

    The same lookup ``hf_bridge._register_attention`` makes: an
    ``AttentionInterface`` with ``register``, or the older
    ``ALL_ATTENTION_FUNCTIONS`` dict. ``transformers`` 4.28 on the NPU host has
    neither, and the bridge refuses to install there by design.
    """
    try:
        import transformers
        from transformers import modeling_utils
    except ImportError:
        return False
    interface = getattr(transformers, "AttentionInterface", None) or getattr(modeling_utils, "AttentionInterface", None)
    return hasattr(interface, "register") or hasattr(modeling_utils, "ALL_ATTENTION_FUNCTIONS")


_NO_ATTENTION_REGISTRY = "transformers version lacks attention registry"


class TestHuggingFaceBridge(unittest.TestCase):
    """Claiming the right layers, and giving the model back exactly as it was."""

    def setUp(self):
        from tq_longbench.hf_bridge import TurboQuantAttentionBridge, discover_full_attention_layers

        self.bridge_cls = TurboQuantAttentionBridge
        self.discover = discover_full_attention_layers
        self.model = _FakeHybridModel()

    def test_only_full_attention_layers_are_claimed(self):
        """Linear attention keeps a recurrent state, not a KV cache, so it is not ours."""
        found = self.discover(self.model)
        self.assertEqual(sorted(found), [3, 7])
        for module in found.values():
            self.assertIsInstance(module, _FakeAttention)

    def test_the_planes_are_indexed_by_claimed_layer_not_model_layer(self):
        bridge = self.bridge_cls(self.model, "turboquant_reference", max_seq_len=512, block_size=BLOCK_SIZE)
        self.assertEqual(bridge.plane_of, {3: 0, 7: 1})
        self.assertEqual(bridge.geometry.num_layers, 2)

    @unittest.skipIf(not _has_attention_registry(), _NO_ATTENTION_REGISTRY)
    def test_uninstall_restores_the_shared_config(self):
        """Regression: every layer of a model shares one config object.

        Recording the previous implementation per layer reads back the bridge's
        own name from the second layer onward, so uninstall leaves the model on
        the bridge it was meant to be leaving. Nothing raises -- the next
        measurement just compares the backend against itself, which is how this
        was found.
        """
        pristine = self.model.config._attn_implementation
        first = self.bridge_cls(self.model, "turboquant_reference", max_seq_len=512, block_size=BLOCK_SIZE)
        with first:
            self.assertEqual(self.model.config._attn_implementation, first._name)
        self.assertEqual(self.model.config._attn_implementation, pristine)

        second = self.bridge_cls(self.model, "dense_reference", max_seq_len=512, block_size=BLOCK_SIZE)
        self.assertNotEqual(second._name, first._name)
        with second:
            self.assertEqual(self.model.config._attn_implementation, second._name)
        self.assertEqual(self.model.config._attn_implementation, pristine)

    @unittest.skipIf(not _has_attention_registry(), _NO_ATTENTION_REGISTRY)
    def test_installing_twice_is_refused(self):
        bridge = self.bridge_cls(self.model, "turboquant_reference", max_seq_len=512, block_size=BLOCK_SIZE)
        with bridge, self.assertRaisesRegex(RuntimeError, "already installed"):
            bridge.install()
        self.assertEqual(self.model.config._attn_implementation, "sdpa")

    @staticmethod
    def _legacy_transformers(registry: Optional[dict]):  # noqa: UP045  (Optional survives a 3.9 runtime eval)
        """``transformers`` as an older release has it: no ``AttentionInterface`` anywhere."""
        package = types.ModuleType("transformers")
        package.__version__ = "4.48.0"
        modeling_utils = types.ModuleType("transformers.modeling_utils")
        if registry is not None:
            modeling_utils.ALL_ATTENTION_FUNCTIONS = registry
        package.modeling_utils = modeling_utils
        return mock.patch.dict(sys.modules, {"transformers": package, "transformers.modeling_utils": modeling_utils})

    def test_an_older_transformers_registers_through_the_dict(self):
        registry: dict = {}
        bridge = self.bridge_cls(self.model, "turboquant_reference", max_seq_len=512, block_size=BLOCK_SIZE)
        with self._legacy_transformers(registry), bridge:
            self.assertEqual(registry, {bridge._name: bridge._attention})
            self.assertEqual(self.model.config._attn_implementation, bridge._name)
        self.assertEqual(self.model.config._attn_implementation, "sdpa")

    def test_a_transformers_with_no_registry_is_refused_at_install(self):
        bridge = self.bridge_cls(self.model, "turboquant_reference", max_seq_len=512, block_size=BLOCK_SIZE)
        with self._legacy_transformers(None), self.assertRaisesRegex(RuntimeError, "no attention registry"):
            bridge.install()
        self.assertEqual(self.model.config._attn_implementation, "sdpa")

    def test_the_hook_attends_over_the_prefix_the_layer_cached(self):
        """Prefill then decode through the hook's own signature, as transformers calls it."""
        bridge = self.bridge_cls(self.model, "turboquant_reference", max_seq_len=512, block_size=BLOCK_SIZE)
        module = self.discover(self.model)[3]
        prompt = 64
        key = torch.randn(1, NUM_KV_HEADS, prompt, HEAD_SIZE)
        value = torch.randn(1, NUM_KV_HEADS, prompt, HEAD_SIZE)
        query = torch.randn(1, NUM_HEADS, prompt, HEAD_SIZE)

        out, weights = bridge._attention(module, query, key, value, None, scaling=module.scaling)
        self.assertEqual(tuple(out.shape), (1, prompt, NUM_HEADS, HEAD_SIZE))
        self.assertIsNone(weights)

        # One more token, with the layer's cache grown by one as transformers grows it.
        step_key = torch.cat((key, torch.randn(1, NUM_KV_HEADS, 1, HEAD_SIZE)), dim=2)
        step_value = torch.cat((value, torch.randn(1, NUM_KV_HEADS, 1, HEAD_SIZE)), dim=2)
        step_query = torch.randn(1, NUM_HEADS, 1, HEAD_SIZE)
        step, _ = bridge._attention(module, step_query, step_key, step_value, None, scaling=module.scaling)
        self.assertEqual(tuple(step.shape), (1, 1, NUM_HEADS, HEAD_SIZE))

    def test_a_padded_batch_is_refused(self):
        """The bridge replaces the attention mask with context lengths, so it cannot pad."""
        bridge = self.bridge_cls(self.model, "turboquant_reference", max_seq_len=512, block_size=BLOCK_SIZE)
        module = self.discover(self.model)[3]
        shape = (2, NUM_KV_HEADS, 8, HEAD_SIZE)
        with self.assertRaisesRegex(NotImplementedError, "one sequence at a time"):
            bridge._attention(
                module,
                torch.randn(2, NUM_HEADS, 8, HEAD_SIZE),
                torch.randn(shape),
                torch.randn(shape),
                None,
                scaling=module.scaling,
            )


class TestPython39Compatibility(unittest.TestCase):
    """The NPU host runs Python 3.9, where ``torch.Tensor | None`` in a signature is a TypeError.

    Postponed evaluation (PEP 563) is what makes those annotations legal there,
    so the compiled flag is asserted rather than the source text.
    """

    MODULES = (
        "vllm_ascend/attention/turboquant_rotation.py",
        "vllm_ascend/attention/turboquant_layout.py",
        "tests/ut/attention/turboquant_cpu_ops.py",
        # This file too: the 3.9 host imports it before any test can run.
        "tests/ut/_tools/test_tq_longbench.py",
        *sorted(path.relative_to(REPO_ROOT).as_posix() for path in (REPO_ROOT / "tools" / "tq_longbench").glob("*.py")),
    )

    def test_every_harness_module_postpones_its_annotations(self):
        flag = __future__.annotations.compiler_flag
        for relative in self.MODULES:
            with self.subTest(module=relative):
                path = REPO_ROOT / relative
                source = path.read_text(encoding="utf-8")
                if "| None" not in source:
                    continue
                code = compile(source, str(path), "exec", dont_inherit=True)
                self.assertTrue(code.co_flags & flag, f"{relative} uses PEP 604 unions without postponed annotations")


# glm-4-9b-chat-1m's config.json as published, fetched 2026-09-18. Its
# multi_query_group_num is 4; glm-4-9b-chat's is 2.
GLM4_9B_CHAT_1M_CONFIG = {
    "model_type": "chatglm",
    "add_bias_linear": False,
    "add_qkv_bias": True,
    "apply_residual_connection_post_layernorm": False,
    "ffn_hidden_size": 13696,
    "hidden_size": 4096,
    "kv_channels": 128,
    "layernorm_epsilon": 1.5625e-07,
    "multi_query_attention": True,
    "multi_query_group_num": 4,
    "num_attention_heads": 32,
    "num_layers": 40,
    "original_rope": True,
    "padded_vocab_size": 151552,
    "post_layer_norm": True,
    "rmsnorm": True,
    "rope_ratio": 10000,
    "seq_length": 1048576,
    "tie_word_embeddings": False,
}

# A two-layer GLM-4 small enough to run eagerly: GQA 4/2, D=64 (Pi needs a power of two).
TINY_GLM = {
    **GLM4_9B_CHAT_1M_CONFIG,
    "ffn_hidden_size": 192,
    "hidden_size": 256,
    "kv_channels": 64,
    "multi_query_group_num": 2,
    "num_attention_heads": 4,
    "num_layers": 2,
    "padded_vocab_size": 320,
    "rope_ratio": 500,
}


def _chatglm_rope(x: torch.Tensor, positions: torch.Tensor, head_size: int, rope_ratio: float) -> torch.Tensor:
    """ChatGLM's ``RotaryEmbedding`` + ``apply_rotary_pos_emb``, transcribed from modeling_chatglm.py."""
    n_elem = head_size // 2
    theta = 1.0 / ((10000 * rope_ratio) ** (torch.arange(0, n_elem, 2, dtype=torch.float32) / n_elem))
    idx_theta = torch.outer(positions.to(torch.float32), theta)
    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)  # [s, n_elem / 2, 2]
    rot_dim = cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    xshaped = x.reshape(x.shape[0], x.shape[1], rot_dim // 2, 2)
    cache = cache.view(x.shape[0], 1, rot_dim // 2, 2)
    out = torch.stack(
        [
            xshaped[..., 0] * cache[..., 0] - xshaped[..., 1] * cache[..., 1],
            xshaped[..., 1] * cache[..., 0] + xshaped[..., 0] * cache[..., 1],
        ],
        -1,
    ).flatten(2)
    return torch.cat((out, x_pass), dim=-1)


def _hf_glm_rope(x: torch.Tensor, positions: torch.Tensor, head_size: int, theta: float) -> torch.Tensor:
    """HF ``modeling_glm``'s partial rotary: cos/sin repeat-interleaved, interleaved rotate_half."""
    rotary_dim = head_size // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[..., : emb.shape[-1] // 2].repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = emb.sin()[..., : emb.shape[-1] // 2].repeat_interleave(2, dim=-1).unsqueeze(1)
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated_half = torch.stack((-x_rot[..., 1::2], x_rot[..., 0::2]), dim=-1).flatten(-2)
    return torch.cat((x_rot * cos + rotated_half * sin, x_pass), dim=-1)


def _chatglm_checkpoint(config: dict, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random weights under chatglm's tensor names, fused qkv and gate/up included."""
    generator = torch.Generator().manual_seed(seed)
    hidden, heads, groups = config["hidden_size"], config["num_attention_heads"], config["multi_query_group_num"]
    head_size, ffn, vocab = config["kv_channels"], config["ffn_hidden_size"], config["padded_vocab_size"]

    def rand(*shape, scale=0.05):
        return torch.randn(*shape, generator=generator) * scale

    tensors = {
        "transformer.embedding.word_embeddings.weight": rand(vocab, hidden, scale=1.0),
        "transformer.encoder.final_layernorm.weight": 1.0 + rand(hidden),
        "transformer.output_layer.weight": rand(vocab, hidden),
        "transformer.rotary_pos_emb.inv_freq": torch.ones(head_size // 4),
    }
    qkv_rows = (heads + 2 * groups) * head_size
    for index in range(config["num_layers"]):
        prefix = f"transformer.encoder.layers.{index}."
        tensors.update(
            {
                prefix + "input_layernorm.weight": 1.0 + rand(hidden),
                prefix + "post_attention_layernorm.weight": 1.0 + rand(hidden),
                prefix + "self_attention.query_key_value.weight": rand(qkv_rows, hidden),
                prefix + "self_attention.query_key_value.bias": rand(qkv_rows, scale=0.5),
                prefix + "self_attention.dense.weight": rand(hidden, heads * head_size),
                prefix + "mlp.dense_h_to_4h.weight": rand(2 * ffn, hidden),
                prefix + "mlp.dense_4h_to_h.weight": rand(hidden, ffn),
            }
        )
    return tensors


def _as_hf_glm(config: dict, tensors: dict[str, torch.Tensor]) -> tuple[dict, dict[str, torch.Tensor]]:
    """The same model under HF ``glm``'s config keys and split-q/k/v tensor names."""
    heads, groups, head_size = config["num_attention_heads"], config["multi_query_group_num"], config["kv_channels"]
    hf_config = {
        "model_type": "glm",
        "num_hidden_layers": config["num_layers"],
        "num_attention_heads": heads,
        "num_key_value_heads": groups,
        "head_dim": head_size,
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["ffn_hidden_size"],
        "vocab_size": config["padded_vocab_size"],
        "rms_norm_eps": config["layernorm_epsilon"],
        "rope_theta": 10000.0 * config["rope_ratio"],
        "partial_rotary_factor": 0.5,
        "attention_bias": True,
        "tie_word_embeddings": False,
    }
    if not tensors:
        return hf_config, {}
    hf = {
        "model.embed_tokens.weight": tensors["transformer.embedding.word_embeddings.weight"],
        "model.norm.weight": tensors["transformer.encoder.final_layernorm.weight"],
        "lm_head.weight": tensors["transformer.output_layer.weight"],
    }
    sizes = (heads * head_size, groups * head_size, groups * head_size)
    for index in range(config["num_layers"]):
        old, new = f"transformer.encoder.layers.{index}.", f"model.layers.{index}."
        for kind in ("weight", "bias"):
            fused = tensors[old + f"self_attention.query_key_value.{kind}"]
            for name, piece in zip(("q_proj", "k_proj", "v_proj"), fused.split(sizes, dim=0)):
                hf[new + f"self_attn.{name}.{kind}"] = piece.clone()
        hf[new + "self_attn.o_proj.weight"] = tensors[old + "self_attention.dense.weight"]
        hf[new + "mlp.gate_up_proj.weight"] = tensors[old + "mlp.dense_h_to_4h.weight"]
        hf[new + "mlp.down_proj.weight"] = tensors[old + "mlp.dense_4h_to_h.weight"]
        hf[new + "input_layernorm.weight"] = tensors[old + "input_layernorm.weight"]
        hf[new + "post_attention_layernorm.weight"] = tensors[old + "post_attention_layernorm.weight"]
    return hf_config, hf


def _chatglm_eager_logits(config: dict, tensors: dict[str, torch.Tensor], token_ids: torch.Tensor) -> torch.Tensor:
    """ChatGLM's forward pass in float32, written from modeling_chatglm.py rather than from layers.py."""
    heads, groups, head_size = config["num_attention_heads"], config["multi_query_group_num"], config["kv_channels"]
    eps = config["layernorm_epsilon"]
    count = token_ids.numel()
    positions = torch.arange(count)

    def rms(x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    hidden = tensors["transformer.embedding.word_embeddings.weight"][token_ids]
    for index in range(config["num_layers"]):
        prefix = f"transformer.encoder.layers.{index}."
        normed = rms(hidden, tensors[prefix + "input_layernorm.weight"])
        mixed = normed @ tensors[prefix + "self_attention.query_key_value.weight"].T
        mixed = mixed + tensors[prefix + "self_attention.query_key_value.bias"]
        query, key, value = mixed.split((heads * head_size, groups * head_size, groups * head_size), dim=-1)
        query = _chatglm_rope(query.view(count, heads, head_size), positions, head_size, config["rope_ratio"])
        key = _chatglm_rope(key.view(count, groups, head_size), positions, head_size, config["rope_ratio"])
        value = value.view(count, groups, head_size)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(0, 1),
            key.transpose(0, 1).repeat_interleave(heads // groups, dim=0),
            value.transpose(0, 1).repeat_interleave(heads // groups, dim=0),
            is_causal=True,
        ).transpose(0, 1)
        hidden = hidden + attended.reshape(count, -1) @ tensors[prefix + "self_attention.dense.weight"].T
        normed = rms(hidden, tensors[prefix + "post_attention_layernorm.weight"])
        gate, up = (normed @ tensors[prefix + "mlp.dense_h_to_4h.weight"].T).chunk(2, dim=-1)
        hidden = hidden + (torch.nn.functional.silu(gate) * up) @ tensors[prefix + "mlp.dense_4h_to_h.weight"].T
    final = rms(hidden, tensors["transformer.encoder.final_layernorm.weight"])
    return final @ tensors["transformer.output_layer.weight"].T


class TestGlm4Config(unittest.TestCase):
    def test_the_published_1m_config_reads_as_its_attention_contract(self):
        shape = glm4_model_shape(GLM4_9B_CHAT_1M_CONFIG)
        self.assertEqual((shape.num_heads, shape.num_kv_heads, shape.head_size), (32, 4, 128))
        self.assertEqual((shape.num_layers, shape.hidden_size, shape.intermediate_size), (40, 4096, 13696))
        self.assertEqual(shape.vocab_size, 151552)
        self.assertEqual(shape.rope_theta, 1.0e8)
        self.assertEqual(shape.rotary_dim, 64)
        self.assertTrue(shape.rope_interleaved)
        self.assertTrue(shape.qkv_bias)
        self.assertFalse(shape.qk_norm)
        self.assertFalse(shape.attn_output_gate)
        self.assertFalse(shape.tie_word_embeddings)

    def test_the_hf_export_reads_as_the_same_shape(self):
        hf_config, _ = _as_hf_glm(GLM4_9B_CHAT_1M_CONFIG, {})
        self.assertEqual(glm4_model_shape(hf_config), glm4_model_shape(GLM4_9B_CHAT_1M_CONFIG))

    def test_a_layer_variant_the_harness_does_not_build_is_refused(self):
        for key, value in (("rmsnorm", False), ("add_bias_linear", True), ("original_rope", False)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                glm4_model_shape({**GLM4_9B_CHAT_1M_CONFIG, key: value})
        with self.assertRaisesRegex(ValueError, "no attention output gate"):
            glm4_model_shape(GLM4_9B_CHAT_1M_CONFIG, attn_output_gate=True)

    def test_glm4_0414_is_refused_by_name(self):
        self.assertTrue(is_glm4_config(GLM4_9B_CHAT_1M_CONFIG))
        self.assertFalse(is_glm4_config({"model_type": "qwen3"}))
        with self.assertRaisesRegex(ValueError, "0414"):
            is_glm4_config({"model_type": "glm4"})


class TestGlm4Rope(unittest.TestCase):
    """Half the head, adjacent pairs: checked against both upstream formulations."""

    def _rope(self, head_size: int, theta: float) -> RotaryEmbedding:
        return RotaryEmbedding(head_size, 4096, theta, torch.float32, CPU, rotary_dim=head_size // 2, interleaved=True)

    def test_it_is_chatglms_rotation(self):
        positions = torch.tensor([0, 1, 7, 1000, 4095])
        x = torch.randn(positions.numel(), 4, 128)
        expected = _chatglm_rope(x, positions, 128, rope_ratio=10000)
        torch.testing.assert_close(self._rope(128, 1.0e8).apply(x, positions), expected, rtol=1e-5, atol=1e-5)

    def test_it_is_hf_glms_rotation(self):
        positions = torch.tensor([0, 3, 511, 4095])
        x = torch.randn(positions.numel(), 2, 128)
        expected = _hf_glm_rope(x, positions, 128, theta=5.0e6)
        torch.testing.assert_close(self._rope(128, 5.0e6).apply(x, positions), expected, rtol=1e-5, atol=1e-5)

    def test_the_second_half_of_each_head_passes_through(self):
        positions = torch.tensor([5, 900])
        x = torch.randn(2, 3, 128)
        self.assertTrue(torch.equal(self._rope(128, 1.0e8).apply(x, positions)[..., 64:], x[..., 64:]))

    def test_the_neox_default_is_unchanged(self):
        """Qwen's full-width rotate_half, restated independently of layers.py."""
        positions = torch.tensor([0, 17, 2048])
        x = torch.randn(3, 2, 64)
        inv_freq = 1.0 / (1.0e6 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        emb = torch.outer(positions.to(torch.float32), inv_freq).repeat(1, 2)
        rotated = torch.cat((-x[..., 32:], x[..., :32]), dim=-1)
        expected = x * emb.cos().unsqueeze(1) + rotated * emb.sin().unsqueeze(1)
        rope = RotaryEmbedding(64, 4096, 1.0e6, torch.float32, CPU)
        torch.testing.assert_close(rope.apply(x, positions), expected, rtol=1e-5, atol=1e-5)


class TestGlm4Checkpoint(unittest.TestCase):
    """A tiny GLM-4 on disk, loaded by the engine exactly as the 9B one would be."""

    PROMPT_TOKENS = 40
    NEW_TOKENS = 6

    def setUp(self):
        torch.manual_seed(0)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.tensors = _chatglm_checkpoint(TINY_GLM)
        self.token_ids = torch.randint(0, TINY_GLM["padded_vocab_size"], (self.PROMPT_TOKENS,), dtype=torch.int64)

    def _write(self, name: str, config: dict, tensors: dict[str, torch.Tensor]) -> str:
        from safetensors.torch import save_file

        path = self.root / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        save_file({key: value.contiguous() for key, value in tensors.items()}, str(path / "model.safetensors"))
        return str(path)

    def _runner(self, model_path: str, backend: str = "dense_reference", **overrides) -> StandaloneModelRunner:
        settings = {
            "backend": backend,
            "prefill_mode": "dense_staging",
            "max_seq_len": 256,
            "chunk_size": 16,
            "block_size": BLOCK_SIZE,
            "dtype": torch.float32,
            "device": "cpu",
            "max_new_tokens": self.NEW_TOKENS,
            **overrides,
        }
        return StandaloneModelRunner(RunnerConfig(model_path=model_path, **settings))

    def _eager_greedy(self, tensors: dict[str, torch.Tensor]) -> list[int]:
        ids, produced = self.token_ids.clone(), []
        for _ in range(self.NEW_TOKENS):
            produced.append(int(torch.argmax(_chatglm_eager_logits(TINY_GLM, tensors, ids)[-1])))
            ids = torch.cat((ids, torch.tensor([produced[-1]])))
        return produced

    def test_the_chatglm_checkpoint_runs_as_chatglm_does(self):
        runner = self._runner(self._write("chatglm", TINY_GLM, self.tensors))
        self.assertEqual((runner.shape.num_heads, runner.shape.num_kv_heads, runner.shape.head_size), (4, 2, 64))
        logits = runner.prefill(self.token_ids, RunMetrics())
        expected = _chatglm_eager_logits(TINY_GLM, self.tensors, self.token_ids)[-1:]
        torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)
        produced, _ = self._runner(runner.config.model_path).generate(self.token_ids)
        self.assertEqual(produced, self._eager_greedy(self.tensors))

    def test_the_hf_export_loads_to_the_same_weights(self):
        chatglm = self._runner(self._write("chatglm", TINY_GLM, self.tensors))
        hf_config, hf_tensors = _as_hf_glm(TINY_GLM, self.tensors)
        hf = self._runner(self._write("glm", hf_config, hf_tensors))
        self.assertEqual(hf.shape, chatglm.shape)
        mine, theirs = dict(chatglm.model.named_parameters()), dict(hf.model.named_parameters())
        self.assertEqual(sorted(mine), sorted(theirs))
        for name, parameter in mine.items():
            self.assertTrue(torch.equal(parameter, theirs[name]), name)

    def test_o_proj_is_folded_as_it_is_ingested(self):
        path = self._write("chatglm", TINY_GLM, self.tensors)
        folded = self._runner(path, backend="turboquant_reference", fold_output_rotation=True)
        fold = turboquant_rotation().fold_pi_into_output_projection
        for index, layer in enumerate(folded.model.layers):
            original = self.tensors[f"transformer.encoder.layers.{index}.self_attention.dense.weight"]
            self.assertTrue(torch.equal(layer.self_attn.o_proj.weight, fold(original, TINY_GLM["kv_channels"])))
        # The folded projection un-rotates what the decode leaves rotated, so the
        # run is the unfolded one's up to the fold's own rounding. dense_staging
        # prefills through the *dense* pool, so this also holds its output to
        # O Pi: left unrotated, the tokens diverge from the first one.
        unfolded = self._runner(path, backend="turboquant_reference")
        self.assertTrue(folded.decode_backend.output_rotation_folded)
        self.assertEqual(folded.generate(self.token_ids)[0], unfolded.generate(self.token_ids)[0])

    def test_the_cube_decode_is_asked_for_the_rotated_basis(self):
        """kv4fp8 Cube, folded, ungated: stage 0, the kernel passes its output through."""
        from tq_longbench.smoke_glm import check_contract

        path = self._write("chatglm", TINY_GLM, self.tensors)
        with cpu_turboquant_ops() as stand_ins, stand_ins.turboquant_cube_meta_ops():
            folded = self._runner(path, backend="turboquant_cube", dtype=torch.float16, fold_output_rotation=True)
            self.assertEqual(check_contract(folded), "ROTATED_BASIS (stage 0)")
            unfolded = self._runner(path, backend="turboquant_cube", dtype=torch.float16)
            self.assertEqual(check_contract(unfolded), "UNROTATED (stage 1)")

    def test_dummy_prompt_runs_without_a_tokenizer(self):
        """``--dummy-prompt`` end to end: 64 ones through the runner, and no tokenizer is ever loaded."""
        from tq_longbench import smoke_glm

        path = self._write("chatglm", TINY_GLM, self.tensors)
        argv = ["--model-path", path, "--device", "cpu", "--backend", "dense_reference", "--dtype", "float32"]
        argv += ["--dummy-prompt", "--max-new-tokens", "4", "--tokens", "8192"]
        refuse = mock.patch.object(smoke_glm, "load_tokenizer", side_effect=AssertionError("tokenizer loaded"))
        with refuse, contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(smoke_glm.main(argv), 0)
        self.assertIn("ids [", out.getvalue())
        self.assertNotIn("needle in a haystack", out.getvalue())
        prompt = smoke_glm.dummy_prompt_ids()
        self.assertEqual((prompt.shape, prompt.dtype, int(prompt.sum())), ((64,), torch.long, 64))


class TestGlm4LegacyTokenizer(unittest.TestCase):
    """The special-token registration ``transformers`` 4.28 needs for GLM-4, on a stand-in tokenizer.

    Checked against the real ChatGLM4Tokenizer on 4.28.0 and 4.44.0 on
    2026-09-18: identical ids for the short prompt and a 1717-token haystack.
    """

    DECLARED = {"151329": {"content": "<|endoftext|>"}, "151331": {"content": "[gMASK]"}}

    def _tokenizer(self, registered: dict[str, int]):
        tokenizer = types.SimpleNamespace(
            added_tokens_encoder=dict(registered),
            added_tokens_decoder={index: token for token, index in registered.items()},
            unique_no_split_tokens=[],
            _create_trie=mock.Mock(),
        )
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": self.DECLARED}))
        return tokenizer, str(root)

    def test_a_tokenizer_that_ignored_the_config_gets_its_tokens_at_their_ids(self):
        from tq_longbench.smoke_glm import register_declared_special_tokens

        tokenizer, root = self._tokenizer({})
        self.assertEqual(register_declared_special_tokens(tokenizer, root), ["<|endoftext|>", "[gMASK]"])
        self.assertEqual(tokenizer.added_tokens_encoder, {"<|endoftext|>": 151329, "[gMASK]": 151331})
        self.assertEqual(tokenizer.added_tokens_decoder, {151329: "<|endoftext|>", 151331: "[gMASK]"})
        self.assertEqual(tokenizer.unique_no_split_tokens, ["<|endoftext|>", "[gMASK]"])
        tokenizer._create_trie.assert_called_once_with(["<|endoftext|>", "[gMASK]"])

    def test_a_tokenizer_that_read_the_config_is_left_alone(self):
        from tq_longbench.smoke_glm import register_declared_special_tokens

        tokenizer, root = self._tokenizer({"<|endoftext|>": 151329, "[gMASK]": 151331})
        self.assertEqual(register_declared_special_tokens(tokenizer, root), [])
        tokenizer._create_trie.assert_not_called()


class TestGlm4CheckpointNames(unittest.TestCase):
    SHAPE = glm4_model_shape(TINY_GLM)
    QKV = "transformer.encoder.layers.1.self_attention.query_key_value.weight"

    def test_the_fused_qkv_splits_query_first(self):
        rows = (4 + 2 * 2) * 64
        fused = torch.arange(rows, dtype=torch.float32).unsqueeze(1).expand(rows, 3)
        targets = dict(glm4_checkpoint_targets(self.QKV, fused, self.SHAPE))
        self.assertEqual(sorted(targets), [f"layers.1.self_attn.{p}_proj.weight" for p in "kqv"])
        self.assertEqual(targets["layers.1.self_attn.q_proj.weight"][0, 0], 0)
        self.assertEqual(targets["layers.1.self_attn.k_proj.weight"][0, 0], 256)
        self.assertEqual(targets["layers.1.self_attn.v_proj.weight"][0, 0], 384)

    def test_a_qkv_that_disagrees_with_the_kv_heads_is_refused(self):
        with self.assertRaisesRegex(ValueError, "multi_query_group_num"):
            glm4_checkpoint_targets(self.QKV, torch.zeros(10, 3), self.SHAPE)

    def test_an_unmodelled_tensor_raises_instead_of_being_skipped(self):
        self.assertEqual(glm4_checkpoint_targets("transformer.rotary_pos_emb.inv_freq", torch.zeros(4), self.SHAPE), [])
        with self.assertRaisesRegex(KeyError, "does not model"):
            glm4_checkpoint_targets(
                "transformer.encoder.layers.0.self_attention.dense.bias", torch.zeros(4), self.SHAPE
            )


class TestGlm4PrefillPolicy(unittest.TestCase):
    def test_batched_decode_from_32k_up(self):
        self.assertEqual(GLM4_BATCHED_DECODE_MIN_TOKENS, 32768)
        self.assertEqual(glm4_prefill_mode(32767, "turboquant_cube"), "dense_staging")
        self.assertEqual(glm4_prefill_mode(32768, "turboquant_cube"), "batched_decode")
        self.assertEqual(glm4_prefill_mode(1048576, "turboquant_aiv"), "batched_decode")
        # A dense baseline decodes out of the fp16 pool; it has no other mode.
        self.assertEqual(glm4_prefill_mode(131072, "native_v5"), "dense_staging")

    def test_run_eval_routes_a_glm_checkpoint_by_the_same_rule(self):
        from tq_longbench.run_eval import default_prefill_mode

        self.assertEqual(default_prefill_mode(32768, "turboquant_cube", glm4_checkpoint=True), "batched_decode")
        self.assertEqual(default_prefill_mode(32768, "turboquant_cube"), "dense_staging")

    def test_the_smoke_runner_takes_its_flags(self):
        from tq_longbench.smoke_glm import build_parser, choose_prefill_mode, parse_depths, truncate_middle

        parser = build_parser()
        args = parser.parse_args(
            ["--model-path", "/m", "--device", "npu", "--backend", "turboquant_cube", "--tokens", "131072"]
        )
        self.assertEqual((args.device, args.backend, args.tokens), ("npu", "turboquant_cube", 131072))
        self.assertEqual(choose_prefill_mode(args), "batched_decode")
        args = parser.parse_args(["--model-path", "/m", "--tokens", "131072", "--prefill-mode", "dense_staging"])
        with contextlib.redirect_stderr(io.StringIO()) as warning:
            self.assertEqual(choose_prefill_mode(args), "dense_staging")
        self.assertIn("HBM", warning.getvalue())
        self.assertEqual(parse_depths("0,0.5,1"), (0.0, 0.5, 1.0))
        with self.assertRaises(ValueError):
            parse_depths("1.5")
        self.assertEqual(truncate_middle(torch.arange(10), 4).tolist(), [0, 1, 8, 9])


class TestStandaloneTurboQuantLibrary(unittest.TestCase):
    """Finding and loading ``libvllm_turboquant_cube.so`` when nothing registered the operators.

    ``torch.ops.load_library`` is mocked: its stand-in registers the CPU operators,
    which is what the real library's static registrations do on the NPU host. The
    real library was built and link-checked in the vendor image (CANN 9.1.0,
    torch 2.10); opening it needs a device driver, which no host here has.
    """

    def setUp(self):
        self.enterPatch(mock.patch.dict(os.environ))
        os.environ.pop(_ascend.TURBOQUANT_LIB_ENV, None)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.library = Path(directory.name) / _ascend.TURBOQUANT_LIB_NAME
        self.library.write_bytes(b"not a real library")
        self.registrations = contextlib.ExitStack()
        self.addCleanup(self.registrations.close)

    def enterPatch(self, patcher):  # noqa: N802  (unittest's camelCase)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def _registers_the_operators(self, path: str) -> None:
        """What dlopen of the real library does: register every TurboQuant op, Cube included."""
        stand_ins = self.registrations.enter_context(cpu_turboquant_ops())
        self.registrations.enter_context(stand_ins.turboquant_cube_meta_ops())

    def _cube_backend(self) -> TurboQuantCubeBackend:
        shape = LayerShape(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE**-0.5)
        return TurboQuantCubeBackend(_geometry(512), shape, CPU)

    def test_the_default_search_is_the_build_script_output_then_build(self):
        candidates = _ascend.turboquant_library_candidates()
        self.assertEqual(candidates[0], REPO_ROOT / "tools" / "tq_longbench" / "lib" / _ascend.TURBOQUANT_LIB_NAME)
        self.assertEqual(candidates[1], REPO_ROOT / "build" / _ascend.TURBOQUANT_LIB_NAME)

    def test_the_environment_replaces_the_search(self):
        os.environ[_ascend.TURBOQUANT_LIB_ENV] = str(self.library)
        self.assertEqual(_ascend.turboquant_library_candidates(), [self.library])
        # A directory names the library inside it.
        os.environ[_ascend.TURBOQUANT_LIB_ENV] = str(self.library.parent)
        self.assertEqual(_ascend.turboquant_library_candidates(), [self.library])

    def test_the_cube_backend_loads_the_library_when_the_decode_is_missing(self):
        os.environ[_ascend.TURBOQUANT_LIB_ENV] = str(self.library)
        loader = self.enterPatch(
            mock.patch.object(torch.ops, "load_library", side_effect=self._registers_the_operators)
        )
        backend = self._cube_backend()
        loader.assert_called_once_with(str(self.library))
        self.assertTrue(cube_decode_available())
        self.assertIsNotNone(backend.cache)

    def test_the_operator_namespace_loads_the_library_for_the_aiv_path(self):
        os.environ[_ascend.TURBOQUANT_LIB_ENV] = str(self.library)
        loader = self.enterPatch(
            mock.patch.object(torch.ops, "load_library", side_effect=self._registers_the_operators)
        )
        self.assertTrue(hasattr(ascend_ops(), "npu_turboquant_reshape_and_cache"))
        loader.assert_called_once_with(str(self.library))

    def test_nothing_is_loaded_when_the_operators_are_registered(self):
        os.environ[_ascend.TURBOQUANT_LIB_ENV] = str(self.library)
        loader = self.enterPatch(mock.patch.object(torch.ops, "load_library"))
        with cpu_turboquant_ops() as stand_ins, stand_ins.turboquant_cube_meta_ops():
            self._cube_backend()
            ascend_ops()
        loader.assert_not_called()

    def test_a_library_that_fails_to_open_is_reported_with_the_loader_error(self):
        os.environ[_ascend.TURBOQUANT_LIB_ENV] = str(self.library)
        self.enterPatch(mock.patch.object(torch.ops, "load_library", side_effect=OSError("undefined symbol: _Zfoo")))
        with self.assertRaisesRegex(RuntimeError, "not registered") as raised:
            self._cube_backend()
        self.assertIn("undefined symbol: _Zfoo", str(raised.exception))
        self.assertIn(str(self.library), str(raised.exception))

    def test_a_library_without_the_decode_is_reported(self):
        os.environ[_ascend.TURBOQUANT_LIB_ENV] = str(self.library)
        self.enterPatch(mock.patch.object(torch.ops, "load_library"))
        with self.assertRaisesRegex(RuntimeError, "does not register the operator"):
            self._cube_backend()

    def test_no_library_means_nothing_is_loaded(self):
        os.environ[_ascend.TURBOQUANT_LIB_ENV] = str(self.library.parent / "missing.so")
        loader = self.enterPatch(mock.patch.object(torch.ops, "load_library"))
        with self.assertRaisesRegex(RuntimeError, r"not registered.*\$TURBOQUANT_LIB_PATH"):
            self._cube_backend()
        loader.assert_not_called()


class TestBuildTurboQuantOps(unittest.TestCase):
    """The standalone build's inputs; the build itself was run in the vendor image."""

    VARIANTS = ["Ascend950PR_9589", "Ascend950PR_9599"]

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SOC_VERSION", None)

    def test_a_soc_variant_is_required_not_a_family(self):
        with self.assertRaisesRegex(SystemExit, "family, not a variant"):
            build_turboquant_ops.resolve_soc_version("Ascend950PR", self.VARIANTS)
        with self.assertRaisesRegex(SystemExit, "Ascend 950 only"):
            build_turboquant_ops.resolve_soc_version("ascend910b1", self.VARIANTS)
        with self.assertRaisesRegex(SystemExit, "no platform config"):
            build_turboquant_ops.resolve_soc_version("Ascend950PR_0000", self.VARIANTS)
        # The spelling CANN ships wins over the one typed.
        self.assertEqual(
            build_turboquant_ops.resolve_soc_version("ascend950pr_9599", self.VARIANTS), "Ascend950PR_9599"
        )

    def test_the_soc_falls_back_to_the_environment_then_the_device(self):
        detected = mock.Mock(return_value="Ascend950PR_9589")
        self.assertEqual(build_turboquant_ops.resolve_soc_version(None, self.VARIANTS, detected), "Ascend950PR_9589")
        os.environ["SOC_VERSION"] = "ascend950pr_9599"
        self.assertEqual(build_turboquant_ops.resolve_soc_version(None, self.VARIANTS, detected), "Ascend950PR_9599")
        detected.assert_called_once()

    def test_what_the_device_reports_is_matched_against_the_platform_configs(self):
        with self.assertRaisesRegex(SystemExit, "no platform config"):
            build_turboquant_ops.resolve_soc_version(None, self.VARIANTS, lambda: "Ascend950PR_957d")
        with self.assertRaisesRegex(SystemExit, "no SoC variant"):
            build_turboquant_ops.resolve_soc_version(None, self.VARIANTS, lambda: None)

    @staticmethod
    def _libascendcl(names, init_status=0):
        """A stand-in for ``ctypes.CDLL(libascendcl.so)``: ``aclrtGetSocName`` answers from ``names`` in turn."""
        answers = iter(names)
        acl = mock.Mock()
        acl.aclrtGetSocName = mock.Mock(side_effect=lambda: next(answers))
        acl.aclInit = mock.Mock(return_value=init_status)
        return acl

    def test_the_soc_name_comes_straight_from_acl(self):
        acl = self._libascendcl([b"Ascend950PR_9599"])
        loader = mock.Mock(return_value=acl)
        self.assertEqual(
            build_turboquant_ops.acl_soc_name(Path("/cann/lib64/libascendcl.so"), loader), "Ascend950PR_9599"
        )
        loader.assert_called_once_with(str(Path("/cann/lib64/libascendcl.so")))
        acl.aclInit.assert_not_called()
        # The return type is set before the call, or ctypes would hand back an int.
        self.assertIs(acl.aclrtGetSocName.restype, ctypes.c_char_p)

    def test_acl_is_initialised_only_when_the_bare_call_has_no_answer(self):
        acl = self._libascendcl([None, b"Ascend950PR_9589"])
        self.assertEqual(
            build_turboquant_ops.acl_soc_name(Path("acl.so"), mock.Mock(return_value=acl)), "Ascend950PR_9589"
        )
        acl.aclInit.assert_called_once_with(None)
        acl.aclFinalize.assert_called_once()

    def test_acl_initialised_by_someone_else_is_not_finalised(self):
        acl = self._libascendcl([None, None], init_status=100002)  # ACL_ERROR_REPEAT_INITIALIZE
        self.assertIsNone(build_turboquant_ops.acl_soc_name(Path("acl.so"), mock.Mock(return_value=acl)))
        acl.aclFinalize.assert_not_called()

    def test_a_libascendcl_that_will_not_load_is_no_answer(self):
        loader = mock.Mock(side_effect=OSError("libascend_hal.so: cannot open shared object file"))
        self.assertIsNone(build_turboquant_ops.acl_soc_name(Path("acl.so"), loader))

    def test_detection_falls_back_to_torch_npu_and_never_runs_npu_smi(self):
        with (
            mock.patch.object(build_turboquant_ops, "acl_soc_name", return_value=None),
            mock.patch.object(build_turboquant_ops, "torch_npu_soc_name", return_value="Ascend950PR_9599"),
            mock.patch.object(build_turboquant_ops.subprocess, "run") as run,
        ):
            self.assertEqual(build_turboquant_ops.detect_soc_version(Path("/cann")), "Ascend950PR_9599")
        run.assert_not_called()

    def test_a_source_tree_under_a_dot_directory_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "TooFewObj"):
            build_turboquant_ops.refuse_dot_directories(Path("/work/.claude/worktrees/x/csrc"))
        build_turboquant_ops.refuse_dot_directories(Path("/work/vllm-ascend/csrc"))

    def test_the_cmake_invocation(self):
        args = build_turboquant_ops.build_parser().parse_args(["--jobs", "3"])
        configure, build, install = build_turboquant_ops.cmake_commands(
            "Ascend950PR_9599", Path("/cann"), Path("/py/torch"), Path("/py/torch_npu"), Path("/b"), Path("/o"), args
        )
        self.assertEqual(configure[configure.index("-G") + 1], "Unix Makefiles")
        self.assertIn("-DSOC_VERSION=Ascend950PR_9599", configure)
        self.assertIn(f"-DCMAKE_PREFIX_PATH={Path('/py/torch') / 'share' / 'cmake'}", configure)
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", configure)
        self.assertEqual(configure[configure.index("-S") + 1], str(build_turboquant_ops.STANDALONE_SOURCE_DIR))
        self.assertEqual(build[-2:], ["-j", "3"])
        self.assertEqual(install, ["cmake", "--install", str(Path("/b")), "--component", "vllm_turboquant"])

    def test_only_this_projects_component_is_installed(self):
        """``ascendc_library()``'s asc-devkit rules would nest lib/ and include/ in the output."""
        args = build_turboquant_ops.build_parser().parse_args([])
        *_, install = build_turboquant_ops.cmake_commands(
            "Ascend950PR_9599", Path("/cann"), Path("/t"), Path("/n"), Path("/b"), Path("/o"), args
        )
        self.assertEqual(install[-2:], ["--component", build_turboquant_ops.INSTALL_COMPONENT])
        cmake = (build_turboquant_ops.STANDALONE_SOURCE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
        rule = cmake[cmake.index("install(TARGETS") :]
        rule = rule[: rule.index(")") + 1]
        self.assertIn(f"COMPONENT {build_turboquant_ops.INSTALL_COMPONENT}", rule)
        self.assertIn("DESTINATION .", rule)
        self.assertNotIn("DESTINATION lib", rule)

    def test_the_output_dir_is_created_and_an_old_nested_install_removed(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        output = Path(directory.name) / "tools" / "tq_longbench" / "lib"
        self.assertEqual(build_turboquant_ops.prepare_output_dir(output), [])
        self.assertTrue(output.is_dir())

        (output / "lib").mkdir()
        (output / "lib" / "libvllm_turboquant_cube_kernels.so").write_bytes(b"old")
        (output / "include" / "vllm_turboquant_cube_kernels").mkdir(parents=True)
        (output / "include" / "vllm_turboquant_cube_kernels" / "aclrtlaunch_x.h").write_text("x")
        (output / "include" / "mine.h").write_text("kept")
        (output / build_turboquant_ops.LIBRARY_NAME).write_bytes(b"current")
        removed = build_turboquant_ops.prepare_output_dir(output)
        self.assertEqual(len(removed), 2)
        self.assertFalse((output / "lib").exists())
        self.assertTrue((output / "include" / "mine.h").is_file())
        self.assertFalse((output / "include" / "vllm_turboquant_cube_kernels").exists())
        self.assertTrue((output / build_turboquant_ops.LIBRARY_NAME).is_file())

    def test_only_an_empty_or_cmake_build_dir_is_wiped(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "keep").mkdir()
        (root / "keep" / "precious.txt").write_text("x")
        with self.assertRaisesRegex(SystemExit, "refusing to delete"):
            build_turboquant_ops.fresh_build_dir(root / "keep")
        (root / "tree").mkdir()
        (root / "tree" / "CMakeCache.txt").write_text("x")
        (root / "tree" / "stale.o").write_text("x")
        build_turboquant_ops.fresh_build_dir(root / "tree")
        self.assertEqual(list((root / "tree").iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
