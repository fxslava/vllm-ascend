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
import math
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
from tq_longbench import engine as engine_module  # noqa: E402
from tq_longbench import ops as ops_module  # noqa: E402
from tq_longbench import preflight as preflight_module  # noqa: E402
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
from tq_longbench.layers import (  # noqa: E402
    ModelShape,
    RotaryEmbedding,
    fold_on_device,
    fold_output_projection_in_place,
    fold_precision,
    pi_matrix,
)
from tq_longbench.ops import (  # noqa: E402
    DENSE_BACKENDS,
    CANNDenseBackend,
    LayerShape,
    NativeV5Backend,
    TurboQuantAivBackend,
    TurboQuantCubeBackend,
    ascend_ops,
    build_backend,
    cube_decode_available,
    reference_equivalent,
)
from tq_longbench.preflight import (  # noqa: E402
    exact_attention,
    probe_backend,
    select_dense_api,
    select_dense_backend,
)
from tq_longbench.probe import (  # noqa: E402
    DIVERGENCE_COSINE,
    LayerProbe,
    cached_keys,
    needle_slot_positions,
)
from tq_longbench.reference import DenseReferenceBackend, TurboQuantReferenceBackend  # noqa: E402
from tq_longbench.smoke_glm import (  # noqa: E402
    encode,
    eos_ids_for,
    glm4_stop_ids,
    prompt_route,
    special_token_id,
    stop_at_eos,
)
from tq_longbench.tasks import (  # noqa: E402
    f1_score,
    needle_in_a_haystack,
    needle_match,
    needle_report,
    rouge_l,
    score,
    substring_match,
)

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


class TestNeedleScoring(unittest.TestCase):
    """What ``found`` means, and what it deliberately does not.

    Written after a run printed ``5/5 retrieved`` over continuations that had
    answered correctly and then recited the haystack. The verdict was right --
    every one of those had the passcode -- and the table had nowhere to say the
    rest. These cases pin both halves: the scoring is strict about the passcode,
    and the report is explicit about the rambling.
    """

    NEEDLE = "894772"

    def test_a_longer_number_containing_the_needle_is_not_the_needle(self):
        """The one way a substring match over digits can be generous, closed."""
        for invented in ("1894772", "8947720", "18947720"):
            with self.subTest(prediction=invented):
                self.assertEqual(substring_match(f"the code is {invented}", self.NEEDLE), 1.0)
                self.assertEqual(needle_match(f"the code is {invented}", self.NEEDLE), 0.0)
        self.assertEqual(needle_match(f"the code is {self.NEEDLE}", self.NEEDLE), 1.0)

    def test_the_boilerplate_without_the_passcode_scores_nothing(self):
        """Reciting the needle's sentence is not answering with the needle."""
        for recited in (
            "The secret passcode for Valencia is",
            "the secret passcode for Valencia is not in the document",
            "I could not find a passcode.",
        ):
            with self.subTest(prediction=recited):
                self.assertEqual(needle_match(recited, self.NEEDLE), 0.0)
                self.assertFalse(needle_report(recited, self.NEEDLE, "Valencia").found)

    def test_the_niah_item_scores_on_the_token_boundary_metric(self):
        item = needle_in_a_haystack(1024, depth=0.5)
        self.assertEqual(item.metric, "needle_match")
        self.assertEqual(score(f"the code is {item.answers[0]}", item.answers, item.metric), 1.0)
        self.assertEqual(score(f"the code is 9{item.answers[0]}", item.answers, item.metric), 0.0)

    def test_an_answer_followed_by_a_ramble_is_found_but_not_clean(self):
        """The depth-0.75 shape: right answer, then the haystack's own filler."""
        report = needle_report(f"{self.NEEDLE}. blue. Trees grow slowly near the river.", self.NEEDLE, "Valencia")
        self.assertTrue(report.found)
        self.assertTrue(report.answered_first)
        self.assertEqual(report.first_number, self.NEEDLE)
        self.assertFalse(report.degenerate)
        self.assertTrue(report.clean)
        self.assertEqual(report.describe(), "found")

    def test_a_repetition_loop_is_found_but_never_clean(self):
        """The depth-1.0 shape: the passcode, then the passcode, then the passcode."""
        report = needle_report(f"{self.NEEDLE}. " * 8, self.NEEDLE, "Valencia")
        self.assertTrue(report.found)
        self.assertTrue(report.degenerate)
        self.assertFalse(report.clean)
        self.assertEqual(report.describe(), "found+loop")

    def test_a_passcode_reached_late_is_marked_late(self):
        """Wandering the haystack and hitting the number on the way is not answering."""
        report = needle_report(f"There are 12 trees and 7 paths. {self.NEEDLE}", self.NEEDLE, "Valencia")
        self.assertTrue(report.found)
        self.assertFalse(report.answered_first)
        self.assertEqual(report.first_number, "12")
        self.assertEqual(report.describe(), "found+late")

    def test_an_ordinary_sentence_is_not_a_loop(self):
        """The degeneracy rule has to leave real answers alone."""
        for prose in (
            f"The secret passcode for Valencia is {self.NEEDLE}.",
            f"{self.NEEDLE}",
            f"Based on the document, the passcode you asked about is {self.NEEDLE}.",
        ):
            with self.subTest(prediction=prose):
                self.assertFalse(needle_report(prose, self.NEEDLE, "Valencia").degenerate)

    def test_echoing_the_needle_sentence_is_flagged_but_still_retrieval(self):
        report = needle_report(f"The secret passcode for Valencia is {self.NEEDLE}.", self.NEEDLE, "Valencia")
        self.assertTrue(report.echoed_prompt)
        self.assertTrue(report.found)


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
        # These too: the 3.9 host imports them before any test can run.
        "tests/ut/_tools/test_tq_longbench.py",
        "tests/ut/_tools/test_attention_layer_cpu_equiv.py",
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


class _Glm4TinyCheckpoint(unittest.TestCase):
    """A tiny GLM-4 written to disk, and the runner that loads it.

    Split from the cases below so a second class can reuse the fixture without
    inheriting -- and re-running -- the first one's tests.
    """

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


class TestGlm4Checkpoint(_Glm4TinyCheckpoint):
    """The tiny checkpoint loaded and run exactly as the 9B one would be."""

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


class _ByteTokenizer:
    """The least a tokenizer can be and still drive the ladder: characters as ids.

    The benchmark's own tests are about the harness -- the rungs, the records,
    the metrics -- not about what a tiny random checkpoint retrieves, so the
    tokenizer only has to be reversible and to keep inside the vocabulary.

    It declares GLM-4's special tokens at ids this vocabulary actually has, near
    the top of it. That is not a convenience: ``special_token_id`` resolves each
    marker through the tokenizer before falling back to the published id, and a
    stub that declared none would have the 151xxx constants assembled into a
    prompt for a 320-token embedding. Which is the same thing that would happen
    to a real checkpoint that numbered its tokens differently, so resolving by
    name is what is under test here as well.
    """

    SPECIAL = {"[gMASK]": 310, "<sop>": 311, "<|user|>": 312, "<|assistant|>": 313, "<|observation|>": 314}

    eos_token_id = None
    chat_template = None
    unk_token_id = 0

    def __init__(self):
        self.added_tokens_encoder = dict(self.SPECIAL, **{"<|endoftext|>": 315})

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [ord(character) % 256 for character in text]
        return types.SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.int64))

    def convert_tokens_to_ids(self, token):
        return self.added_tokens_encoder.get(token, self.unk_token_id)

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(int(token)) for token in token_ids if int(token) < 256)


class TestRunBenchmark(unittest.TestCase):
    """The context ladder end to end, on the same tiny GLM-4 the checkpoint tests use."""

    CONTEXTS = (64, 128)
    DEPTHS = (0.0, 1.0)
    NEW_TOKENS = 2

    def setUp(self):
        torch.manual_seed(0)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.model_path = self.root / "chatglm"
        self.model_path.mkdir()
        (self.model_path / "config.json").write_text(json.dumps(TINY_GLM), encoding="utf-8")
        from safetensors.torch import save_file

        tensors = _chatglm_checkpoint(TINY_GLM)
        save_file(
            {key: value.contiguous() for key, value in tensors.items()},
            str(self.model_path / "model.safetensors"),
        )

    def _run(self, *extra):
        from tq_longbench import run_benchmark

        out_file = self.root / "bench.jsonl"
        argv = [
            "--model-path", str(self.model_path),
            "--device", "cpu",
            "--backends", "dense_reference",
            "--contexts", ",".join(str(context) for context in self.CONTEXTS),
            "--depths", ",".join(str(depth) for depth in self.DEPTHS),
            "--max-new-tokens", str(self.NEW_TOKENS),
            "--dtype", "float32",
            "--chunk-size", "16",
            "--block-size", str(BLOCK_SIZE),
            "--out-file", str(out_file),
            *extra,
        ]  # fmt: skip
        tokenizer = mock.patch.object(run_benchmark, "load_tokenizer", return_value=_ByteTokenizer())
        with tokenizer, contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_benchmark.main(argv), 0)
        records = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
        return records, out.getvalue()

    def test_every_rung_and_depth_leaves_a_record(self):
        records, _ = self._run()
        self.assertEqual(len(records), len(self.CONTEXTS) * len(self.DEPTHS))
        self.assertEqual(
            [(record["context_requested"], record["depth"]) for record in records],
            [(context, depth) for context in self.CONTEXTS for depth in self.DEPTHS],
        )
        for record in records:
            with self.subTest(context=record["context_requested"], depth=record["depth"]):
                self.assertEqual(record["backend"], "dense_reference")
                self.assertEqual(record["prefill_mode"], "dense_staging")
                # At most the budget, and fewer when a stop token came first --
                # which is now a legitimate ending rather than a trimmed one.
                self.assertGreater(record["generated_tokens"], 0)
                self.assertLessEqual(record["generated_tokens"], self.NEW_TOKENS)
                self.assertEqual(record["hit_token_budget"], record["stopped_on_token"] is None)
                self.assertIsInstance(record["found"], bool)
                self.assertIsNone(record["decode_failure"])
                # The verdict columns travel with the record, not just the count:
                # a results file that says only "found" cannot be re-read later.
                self.assertIn(record["verdict"], ("missed", "found", "found+late", "found+loop", "found+late+loop"))
                self.assertEqual(record["clean"], record["found"] and "+" not in record["verdict"])

    def test_the_record_carries_the_metrics_the_ladder_exists_to_report(self):
        """TTFT, the decode percentiles and the memory, each measured rather than derived."""
        records, _ = self._run()
        for record in records:
            with self.subTest(context=record["context_requested"]):
                for key in ("ttft_ms", "decode_p50_us", "decode_p99_us", "kv_cache_mb", "dense_equivalent_mb"):
                    self.assertIn(key, record)
                # Time to first token is the prefill plus the argmax over its
                # logits and nothing else, so it brackets the prefill from above
                # and the whole item from below.
                self.assertGreaterEqual(record["ttft_ms"], record["prefill_seconds"] * 1e3)
                self.assertLessEqual(record["ttft_ms"], record["wall_seconds"] * 1e3)
                self.assertGreaterEqual(record["decode_p99_us"], record["decode_p50_us"])
                # No allocator on the CPU to ask, and a zero says so rather than guessing.
                self.assertEqual(record["device_peak_mb"], 0.0)

    def test_a_rungs_pool_is_sized_for_that_rung_not_for_the_ladder(self):
        """Which is the whole reason a runner is rebuilt per rung: a shared one reports one size."""
        records, _ = self._run()
        by_context = {}
        for record in records:
            by_context.setdefault(record["context_requested"], record["kv_cache_mb"])
        self.assertEqual(sorted(by_context), sorted(self.CONTEXTS))
        self.assertLess(by_context[self.CONTEXTS[0]], by_context[self.CONTEXTS[1]])

        shared, _ = self._run("--reuse-runner")
        held = {record["kv_cache_mb"] for record in shared}
        self.assertEqual(len(held), 1, "--reuse-runner should report the one pool it allocated")

    def test_the_summary_has_a_line_per_rung(self):
        _, printed = self._run()
        lines = [line for line in printed.splitlines() if line.startswith("dense_reference")]
        self.assertEqual(len(lines), len(self.CONTEXTS))
        for line, context in zip(lines, self.CONTEXTS):
            self.assertIn(str(context), line)
            self.assertIn(f"/{len(self.DEPTHS)}", line)
        # Retrieved and clean are both columns, so a rambling run cannot read as
        # an unqualified pass. This is the regression for that reading.
        header = next(line for line in printed.splitlines() if "retrieved" in line)
        self.assertIn("clean", header)
        self.assertIn("retrieved > clean", printed)

    def test_tensorflow_is_off_before_transformers_can_be_imported(self):
        """A TensorFlow that predates NumPy 2 makes ``import transformers`` raise, not warn."""
        from tq_longbench import run_benchmark, smoke_glm

        for name, value in smoke_glm.TENSORFLOW_OFF.items():
            self.assertEqual(os.environ.get(name), value)
        self.assertIsNot(run_benchmark.quiet_tensorflow, None)
        # The harness never imports it, so nothing here can have reached it.
        self.assertNotIn("tensorflow", sys.modules)


class _ChatGLM4LikeTokenizer:
    """ChatGLM4Tokenizer's shape on ``transformers`` 4.28, which is where this matters.

    It prepends ``[gMASK]<sop>`` to everything it encodes, declares its special
    tokens in ``added_tokens_encoder`` (where
    :func:`register_declared_special_tokens` puts them), has no ``chat_template``
    and no ``apply_chat_template``, and -- the part that made generation never
    stop -- reports ``eos_token_id`` as ``None``.
    """

    PREFIX = [151331, 151333]
    added_tokens_encoder = {
        "<|endoftext|>": 151329,
        "[gMASK]": 151331,
        "<sop>": 151333,
        "<|user|>": 151336,
        "<|assistant|>": 151337,
        "<|observation|>": 151338,
    }
    eos_token_id = None
    unk_token_id = 0

    def _body(self, text):
        return [ord(character) % 256 for character in text]

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = (self.PREFIX + self._body(text)) if add_special_tokens else self._body(text)
        return types.SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.int64))

    def convert_tokens_to_ids(self, token):
        return self.added_tokens_encoder.get(token, self.unk_token_id)

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(int(token)) for token in token_ids if int(token) < 256)


class TestGenerationStops(unittest.TestCase):
    """That a stop token ends the loop, and that GLM-4's stop set is never empty.

    Both halves were silently absent. ``glm-4-9b-chat-1m``'s ``config.json``
    declares no ``eos_token_id`` and the 4.28 tokenizer reports ``None`` for it,
    so the stop set was ``{}``; and ``stop_at_eos`` trimmed the text *after* the
    fact, so even a correct stop set would have let every continuation run its
    whole budget first. A run could only be read as a model that had more to say.
    """

    BUDGET = 6

    def test_the_glm4_stop_set_is_never_empty(self):
        """The three ids that end an assistant turn, from a tokenizer that declares none."""
        tokenizer = _ChatGLM4LikeTokenizer()
        self.assertIsNone(tokenizer.eos_token_id)
        self.assertNotIn("eos_token_id", GLM4_9B_CHAT_1M_CONFIG)
        self.assertEqual(eos_ids_for(GLM4_9B_CHAT_1M_CONFIG, tokenizer), {151329, 151336, 151338})

    def test_the_stop_ids_come_from_the_tokenizer_before_the_constants(self):
        """A checkpoint that numbers its special tokens differently still stops."""
        tokenizer = _ChatGLM4LikeTokenizer()
        tokenizer.added_tokens_encoder = dict(tokenizer.added_tokens_encoder, **{"<|user|>": 999})
        self.assertIn(999, glm4_stop_ids(tokenizer))
        self.assertNotIn(151336, glm4_stop_ids(tokenizer))

    def test_an_unknown_special_token_falls_back_rather_than_becoming_unk(self):
        """``<unk>`` is not a stop token; treating it as one truncates at the first odd word."""

        class Ignorant:
            added_tokens_encoder: dict = {}
            unk_token_id = 0

            def convert_tokens_to_ids(self, token):
                return 0

        self.assertEqual(special_token_id(Ignorant(), "<|user|>", 151336), 151336)

    def test_a_non_glm4_checkpoint_keeps_its_own_stop_tokens(self):
        """GLM-4's ids are GLM-4's: nothing here leaks into another model's vocabulary."""
        qwen = {"model_type": "qwen2", "eos_token_id": 151645}
        self.assertEqual(eos_ids_for(qwen, _ChatGLM4LikeTokenizer()), {151645})

    def test_stopping_turns_the_reported_loop_into_a_clean_answer(self):
        """The observed depth-1.0 continuation, with and without the loop being ended.

        ``581850. 581850. 581850. ...`` is what a correct answer looks like when
        nothing stops the model afterwards. Ending the loop at the stop token it
        emitted leaves the answer, and the verdict goes from found+loop to clean.

        What this pins is the composition -- generate stops, stop_at_eos trims,
        needle_report scores -- not that GLM-4 emits a stop token there. That is
        a property of the model and the prompt, and only silicon can report it.
        """
        needle = "581850"
        unbounded = f"{needle}. " * 8
        self.assertEqual(needle_report(unbounded, needle, None).describe(), "found+loop")
        self.assertFalse(needle_report(unbounded, needle, None).clean)

        stopped = f"{needle}."
        self.assertEqual(needle_report(stopped, needle, None).describe(), "found")
        self.assertTrue(needle_report(stopped, needle, None).clean)


class TestStopTokenEndsTheLoop(_Glm4TinyCheckpoint):
    """The loop itself, on the tiny checkpoint the other GLM-4 cases use."""

    def _first_token(self, runner) -> int:
        produced, _ = runner.generate(self.token_ids, max_new_tokens=1)
        return produced[0]

    def test_a_stop_token_ends_the_loop_not_just_the_text(self):
        """Told to stop on what it is about to produce, the run produces one token and stops.

        The decode-step list is the assertion that matters: an empty one means no
        step was ever launched past the stop, which is what "halts immediately"
        has to mean if the latency percentiles are to describe tokens the model
        meant to emit.
        """
        path = self._write("chatglm", TINY_GLM, self.tensors)
        stop = self._first_token(self._runner(path))
        produced, metrics = self._runner(path).generate(self.token_ids, max_new_tokens=8, stop_ids={stop})
        self.assertEqual(produced, [stop])
        self.assertEqual(metrics.stopped_on_token, stop)
        self.assertEqual(metrics.decode_step_us, [])
        self.assertFalse(metrics.summary()["hit_token_budget"])
        # The text is what precedes the stop token, so the two agree about the end.
        self.assertEqual(stop_at_eos(produced, {stop}), [])

    def test_without_a_stop_set_the_whole_budget_is_spent(self):
        """The old behaviour, kept as the contrast: this is what an empty stop set looks like."""
        path = self._write("chatglm", TINY_GLM, self.tensors)
        produced, metrics = self._runner(path).generate(self.token_ids, max_new_tokens=self.BUDGET_TOKENS)
        self.assertEqual(len(produced), self.BUDGET_TOKENS)
        self.assertIsNone(metrics.stopped_on_token)
        self.assertTrue(metrics.summary()["hit_token_budget"])

    def test_a_stop_id_the_model_never_emits_changes_nothing(self):
        path = self._write("chatglm", TINY_GLM, self.tensors)
        plain, _ = self._runner(path).generate(self.token_ids, max_new_tokens=self.BUDGET_TOKENS)
        guarded, metrics = self._runner(path).generate(
            self.token_ids, max_new_tokens=self.BUDGET_TOKENS, stop_ids={TINY_GLM["padded_vocab_size"] + 1}
        )
        self.assertEqual(guarded, plain)
        self.assertIsNone(metrics.stopped_on_token)

    BUDGET_TOKENS = 6


class TestGlm4ChatPrompt(unittest.TestCase):
    """The turn markers, for a ``transformers`` that cannot apply the template itself."""

    def test_the_prompt_is_the_documented_format(self):
        """``[gMASK]<sop><|user|>\\n{prompt}\\n<|assistant|>``, assembled by id."""
        tokenizer = _ChatGLM4LikeTokenizer()
        ids = encode(tokenizer, "HI", use_chat_template=True, glm4=True).tolist()
        self.assertEqual(ids[:3], [151331, 151333, 151336])
        self.assertEqual(ids[-1], 151337)
        self.assertEqual(ids[3:-1], [ord("\n"), ord("H"), ord("I"), ord("\n")])

    def test_the_tokenizers_own_prefix_does_not_appear_twice(self):
        """ChatGLM4Tokenizer prepends [gMASK]<sop> to everything; the body must not carry it."""
        ids = encode(_ChatGLM4LikeTokenizer(), "HI", use_chat_template=True, glm4=True).tolist()
        self.assertEqual(ids.count(151331), 1)
        self.assertEqual(ids.count(151333), 1)

    def test_a_tokenizer_without_the_flag_has_its_prefix_measured(self):
        """``add_special_tokens`` is the direct way to decline the prefix; not every call takes it."""

        class NoFlag(_ChatGLM4LikeTokenizer):
            def __call__(self, text, return_tensors=None):
                return types.SimpleNamespace(
                    input_ids=torch.tensor([self.PREFIX + self._body(text)], dtype=torch.int64)
                )

        with_flag = encode(_ChatGLM4LikeTokenizer(), "HI", use_chat_template=True, glm4=True)
        measured = encode(NoFlag(), "HI", use_chat_template=True, glm4=True)
        self.assertEqual(measured.tolist(), with_flag.tolist())

    def test_the_tokenizers_own_template_wins_where_it_exists(self):
        """A transcription is a fallback, never a preference: the checkpoint's own format rules."""

        class Modern(_ChatGLM4LikeTokenizer):
            chat_template = "{{ 'x' }}"

            def apply_chat_template(self, messages, **kwargs):
                return {"input_ids": torch.tensor([[7, 7, 7]], dtype=torch.int64)}

        self.assertEqual(encode(Modern(), "HI", use_chat_template=True, glm4=True).tolist(), [7, 7, 7])

    def test_raw_prompt_still_means_raw(self):
        ids = encode(_ChatGLM4LikeTokenizer(), "HI", use_chat_template=False, glm4=True).tolist()
        self.assertNotIn(151336, ids)

    def test_the_route_is_named_rather_than_inferred(self):
        """Which of the three routes ran is printed, because it used to be invisible."""
        plain = _ChatGLM4LikeTokenizer()
        self.assertIn("assembled here", prompt_route(plain, use_chat_template=True, glm4=True))
        self.assertIn("raw", prompt_route(plain, use_chat_template=False, glm4=True))
        self.assertIn("not GLM-4", prompt_route(plain, use_chat_template=True, glm4=False))


class TestOutputProjectionFold(unittest.TestCase):
    """Pi as a matrix, and the fold as a matmul instead of a transform.

    The fold was seven strided float64 passes per layer on the host, in front of
    every run. The same map written as a right multiply by a ``[D, D]`` matrix is
    what a Cube is built for -- and, it turns out, what a BLAS ``gemm`` is too, so
    the matmul wins on a CPU as well despite ``O(D^2)`` against ``O(D log D)``.
    """

    HEAD_SIZE = 128

    def _pi(self, dtype=torch.float64):
        return pi_matrix(self.HEAD_SIZE, CPU, dtype)

    def test_pi_is_built_from_the_shipped_transform(self):
        """Not assembled here from a Hadamard and the signs: there is one definition.

        A second copy of Pi would not raise when it drifted -- it would rotate the
        cache into a basis the kernels do not share and return plausible numbers.
        """
        rotation = turboquant_rotation()
        signs = rotation.turboquant_pi_signs(self.HEAD_SIZE, CPU).to(torch.float64)
        pi = self._pi()
        torch.manual_seed(0)
        x = torch.randn(5, 3, self.HEAD_SIZE, dtype=torch.float64)
        # The matrix *is* the transform: applying one is multiplying by the other.
        torch.testing.assert_close(rotation.apply_pi(x, signs), x @ pi)

    def test_pi_is_a_symmetric_involution(self):
        """Which is why one matrix serves both the fold and the un-rotation."""
        pi = self._pi()
        torch.testing.assert_close(pi, pi.T)
        torch.testing.assert_close(pi @ pi, torch.eye(self.HEAD_SIZE, dtype=torch.float64))

    def test_the_matmul_fold_reproduces_the_shipped_one_exactly(self):
        """At float64 the two routes agree to the last bit, on both stored dtypes."""
        rotation = turboquant_rotation()
        torch.manual_seed(0)
        for dtype in (torch.float16, torch.float32):
            with self.subTest(stored=dtype):
                weight = torch.randn(256, 512, dtype=dtype)
                shipped = rotation.fold_pi_into_output_projection(weight, self.HEAD_SIZE)
                mine = weight.clone()
                fold_output_projection_in_place(mine, self.HEAD_SIZE, self._pi())
                self.assertTrue(torch.equal(shipped, mine))

    def test_float32_accumulation_stays_inside_one_float16_ulp(self):
        """What an NPU will actually do, held to what the host would have done.

        float32 is not float64 and the two do not agree bit for bit. What matters
        is that the disagreement is smaller than the float16 the result is stored
        in, so the fold's only error is still that dtype's rounding.
        """
        rotation = turboquant_rotation()
        torch.manual_seed(0)
        weight = torch.randn(512, 512, dtype=torch.float16)
        shipped = rotation.fold_pi_into_output_projection(weight, self.HEAD_SIZE)
        mine = weight.clone()
        fold_output_projection_in_place(mine, self.HEAD_SIZE, self._pi(torch.float32))
        largest = float((shipped.to(torch.float64) - mine.to(torch.float64)).abs().max())
        self.assertLessEqual(largest, 2 * float(torch.finfo(torch.float16).eps))
        self.assertGreater(cosine(mine, shipped), 0.99999999)

    def test_the_precision_follows_the_device(self):
        """float64 where it is fast, float32 where there is no fast float64."""
        self.assertIs(fold_precision(CPU), torch.float64)
        self.assertIs(fold_precision(types.SimpleNamespace(type="npu")), torch.float32)
        self.assertEqual(pi_matrix(self.HEAD_SIZE, CPU).dtype, torch.float64)

    def test_where_the_fold_runs_follows_the_site(self):
        npu = types.SimpleNamespace(type="npu")
        self.assertFalse(fold_on_device("auto", CPU))
        self.assertTrue(fold_on_device("auto", npu))
        self.assertFalse(fold_on_device("host", npu))
        self.assertTrue(fold_on_device("device", CPU))
        with self.assertRaisesRegex(ValueError, "unknown fold site"):
            fold_on_device("elsewhere", CPU)

    def test_a_misshapen_projection_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not divide"):
            fold_output_projection_in_place(torch.zeros(4, 130), self.HEAD_SIZE, self._pi())
        with self.assertRaisesRegex(ValueError, "Pi is"):
            fold_output_projection_in_place(torch.zeros(4, 256), self.HEAD_SIZE, self._pi()[:4, :4])


class TestFoldSiteEndToEnd(_Glm4TinyCheckpoint):
    """The two sites through the whole loader, on the tiny checkpoint."""

    def _folded(self, site: str):
        # Written once per test: _write creates the directory, so a second call
        # for the second site would fail before anything was folded.
        if not hasattr(self, "_path"):
            self._path = self._write("chatglm", TINY_GLM, self.tensors)
        return self._runner(self._path, backend="turboquant_reference", fold_output_rotation=True, fold_site=site)

    def test_both_sites_fold_every_layer_to_the_same_weights(self):
        host, device = self._folded("host"), self._folded("device")
        fold = turboquant_rotation().fold_pi_into_output_projection
        for index, (left, right) in enumerate(zip(host.model.layers, device.model.layers)):
            with self.subTest(layer=index):
                original = self.tensors[f"transformer.encoder.layers.{index}.self_attention.dense.weight"]
                expected = fold(original.to(torch.float32), TINY_GLM["kv_channels"])
                self.assertTrue(torch.equal(left.self_attn.o_proj.weight, expected))
                self.assertTrue(torch.equal(right.self_attn.o_proj.weight, expected))

    def test_both_sites_generate_the_same_tokens(self):
        host = self._folded("host").generate(self.token_ids)[0]
        device = self._folded("device").generate(self.token_ids)[0]
        self.assertEqual(host, device)

    def test_the_run_says_where_it_folded_and_what_it_cost(self):
        for site in ("host", "device"):
            with self.subTest(site=site):
                report = self._folded(site).fold_report
                self.assertEqual(report.folded, TINY_GLM["num_layers"])
                self.assertEqual(report.site, site)
                self.assertIn(f"on the {site}", report.describe())

    def test_a_run_that_folds_nothing_still_has_a_report(self):
        """Printed unconditionally, so it must exist before load_weights runs."""
        self.assertEqual(self._runner(self._write("chatglm", TINY_GLM, self.tensors)).fold_report.folded, 0)

    def test_an_unknown_fold_site_is_refused_at_config_time(self):
        with self.assertRaisesRegex(ValueError, "unknown fold_site"):
            RunnerConfig(model_path="<random weights>", fold_site="somewhere")

    def test_a_device_fold_that_does_not_reproduce_the_host_one_is_refused(self):
        """The check that catches a device whose float32 matmul is not float32.

        Simulated by folding through a Pi that is not Pi -- which is the shape of
        the failure a reduced-precision matmul would produce, and which nothing
        downstream would raise on: the run would be fluent and wrong.
        """
        self._folded("host")  # writes the checkpoint
        blunt = mock.patch.object(
            engine_module,
            "pi_matrix",
            return_value=torch.eye(TINY_GLM["kv_channels"], dtype=torch.float64),
        )
        with blunt, self.assertRaisesRegex(RuntimeError, "does not reproduce the host fold"):
            self._folded("device")


class TestLayerProbe(_Glm4TinyCheckpoint):
    """The layer-wise diagnostic, on the tiny checkpoint.

    What is under test is that the four columns are read off the run rather than
    modelled, that a divergence put in on purpose is the layer the table points
    at, and -- the one that would otherwise rot -- that a run without the probe
    carries no hooks at all.
    """

    CONTEXT = 200
    NEEDLE = slice(40, 46)

    def _prompt(self):
        ids = torch.randint(0, TINY_GLM["padded_vocab_size"], (self.CONTEXT,), dtype=torch.int64)
        return ids, needle_slot_positions(ids, ids[self.NEEDLE].clone())

    def _probe(self, backend, ids, slots, reference=None, target=None):
        path = getattr(self, "_path", None) or self._write("chatglm", TINY_GLM, self.tensors)
        self._path = path
        runner = self._runner(path, backend=backend, chunk_size=64, max_seq_len=512)
        probe = LayerProbe(runner, needle_slots=slots, target_token=target, reference=reference)
        with probe:
            runner.prefill(ids, RunMetrics())
        return runner, probe

    def test_a_run_without_the_probe_carries_no_hooks(self):
        """Zero overhead is a claim about the hook dict, so that is what is asserted.

        torch skips its whole hook machinery when the dicts are empty, so an
        unprobed run costs exactly what it did before this module existed. A
        probe that left a handle behind would make every later run pay for a
        diagnostic nobody asked for.
        """
        ids, slots = self._prompt()
        runner, probe = self._probe("dense_reference", ids, slots)
        for layer in runner.model.layers:
            with self.subTest(module="layer"):
                self.assertEqual(len(layer._forward_hooks), 0)
                self.assertEqual(len(layer.self_attn._forward_hooks), 0)
        self.assertEqual(probe._handles, [])

    def test_the_hooks_exist_only_inside_the_context(self):
        ids, slots = self._prompt()
        path = self._write("chatglm", TINY_GLM, self.tensors)
        self._path = path
        runner = self._runner(path, backend="dense_reference", chunk_size=64, max_seq_len=512)
        layer = runner.model.layers[0]
        self.assertEqual(len(layer._forward_hooks), 0)
        with LayerProbe(runner, needle_slots=slots):
            self.assertEqual(len(layer._forward_hooks), 1)
            self.assertEqual(len(layer.self_attn._forward_hooks), 1)
        self.assertEqual(len(layer._forward_hooks), 0)
        self.assertEqual(len(layer.self_attn._forward_hooks), 0)

    def test_every_layer_reports_every_column(self):
        ids, slots = self._prompt()
        base_runner, base = self._probe("dense_reference", ids, slots, target=int(ids[-1]))
        _, test = self._probe("dense_reference", ids, slots, reference=base.hidden, target=int(ids[-1]))
        report = test.report()
        self.assertEqual(len(report.readings), TINY_GLM["num_layers"])
        self.assertEqual(report.context, self.CONTEXT)
        for reading in report.readings:
            with self.subTest(layer=reading.layer):
                self.assertIsNotNone(reading.needle_mass)
                self.assertIsNotNone(reading.attention_entropy)
                self.assertIsNotNone(reading.target_rank)
                self.assertIsNotNone(reading.hidden_cosine)
                # A probability mass and an entropy inside their own ranges.
                self.assertGreaterEqual(reading.needle_mass, 0.0)
                self.assertLessEqual(reading.needle_mass, 1.0)
                self.assertGreater(reading.attention_entropy, 0.0)
                self.assertLessEqual(reading.attention_entropy, math.log(self.CONTEXT) + 1e-6)
        # The same run against itself: every layer identical.
        self.assertGreater(min(r.hidden_cosine for r in report.readings), 0.999999)
        self.assertIs(base_runner.model.layers[0].self_attn.shape, base_runner.model.layers[0].self_attn.shape)

    def test_the_table_points_at_a_layer_that_was_made_to_diverge(self):
        """The assertion with teeth: break one layer and see whether L* finds it.

        A probe that only ever sees healthy runs says nothing about what it would
        do with a broken one. Layer 1's output projection is scaled hard enough
        that everything downstream of it leaves the baseline, and L* has to be 1
        rather than 0 or the last one.
        """
        ids, slots = self._prompt()
        _, base = self._probe("dense_reference", ids, slots)

        path, broken = self._path, 1
        runner = self._runner(path, backend="dense_reference", chunk_size=64, max_seq_len=512)
        with torch.no_grad():
            runner.model.layers[broken].self_attn.o_proj.weight.mul_(50.0)
        with LayerProbe(runner, needle_slots=slots, reference=base.hidden) as probe:
            runner.prefill(ids, RunMetrics())

        report = probe.report()
        self.assertTrue(report.compared)
        self.assertEqual(report.breakdown_layer(), broken)
        self.assertGreater(report.readings[0].hidden_cosine, DIVERGENCE_COSINE)
        self.assertLess(report.readings[broken].hidden_cosine, DIVERGENCE_COSINE)
        self.assertIn("left the dense baseline", report.describe())

    def test_a_run_with_no_baseline_says_so_rather_than_claiming_divergence(self):
        ids, slots = self._prompt()
        _, probe = self._probe("dense_reference", ids, slots)
        report = probe.report()
        self.assertFalse(report.compared)
        self.assertTrue(all(r.hidden_cosine is None for r in report.readings))
        self.assertIn("No dense baseline was run", report.describe())

    def test_the_quantised_cache_is_read_back_in_the_models_own_basis(self):
        """A 4-bit pool holds Pi k; the probe has to undo that or measure nonsense.

        Held to the dense pool's own keys: the two differ by the codec and by
        nothing else, so a missing un-rotation shows up immediately -- Pi scatters
        a vector across all 128 channels, and the cosine would collapse.
        """
        ids, slots = self._prompt()
        with cpu_turboquant_ops():
            dense_runner, _ = self._probe("dense_reference", ids, slots)
            quant_runner, _ = self._probe("turboquant_reference", ids, slots)
            dense = cached_keys(dense_runner.decode_backend, 0, self.CONTEXT)
            quantised = cached_keys(quant_runner.decode_backend, 0, self.CONTEXT)
        self.assertEqual(dense.shape, quantised.shape)
        self.assertGreater(cosine(quantised, dense), 0.98)

    def test_an_unknown_cache_degrades_rather_than_guessing(self):
        self.assertIsNone(cached_keys(types.SimpleNamespace(cache=None, shape=None), 0, 8))


class TestNeedleSlots(unittest.TestCase):
    """Finding the needle's positions in the tokenised prompt."""

    def test_every_occurrence_is_found(self):
        """A haystack built by repetition can carry the answer twice by accident."""
        prompt = torch.tensor([9, 1, 2, 3, 9, 9, 1, 2, 3, 9], dtype=torch.int64)
        needle = torch.tensor([1, 2, 3], dtype=torch.int64)
        self.assertEqual(needle_slot_positions(prompt, needle).tolist(), [1, 2, 3, 6, 7, 8])

    def test_a_needle_that_is_not_there_is_empty_not_wrong(self):
        prompt = torch.arange(10, dtype=torch.int64)
        self.assertEqual(needle_slot_positions(prompt, torch.tensor([99, 98])).numel(), 0)
        self.assertEqual(needle_slot_positions(prompt, torch.empty(0, dtype=torch.int64)).numel(), 0)
        self.assertEqual(needle_slot_positions(torch.tensor([1]), torch.arange(5)).numel(), 0)


class TestRunLongBench(unittest.TestCase):
    """The LongBench ladder end to end, with the dataset stubbed.

    ``datasets`` is not a harness dependency and downloading LongBench is not a
    unit test's business, so ``load_longbench`` is replaced with items built from
    the suite's own prompt templates. What is under test is the sweep -- tasks
    times rungs, one runner per rung, the record and the table -- not the corpus.
    """

    TASKS = ("narrativeqa", "gov_report")
    CONTEXTS = (128, 256)
    ITEMS = 2

    def setUp(self):
        torch.manual_seed(0)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.model_path = self.root / "chatglm"
        self.model_path.mkdir()
        (self.model_path / "config.json").write_text(json.dumps(TINY_GLM), encoding="utf-8")
        from safetensors.torch import save_file

        save_file(
            {key: value.contiguous() for key, value in _chatglm_checkpoint(TINY_GLM).items()},
            str(self.model_path / "model.safetensors"),
        )

    def _items(self, task, limit=None):
        from tq_longbench.tasks import _TASK_METRICS, LONGBENCH_MAX_NEW_TOKENS, LONGBENCH_PROMPTS, EvalItem

        for index in range(limit or self.ITEMS):
            yield EvalItem(
                prompt=LONGBENCH_PROMPTS[task].format(context="alpha beta gamma delta " * 20, input="what?"),
                answers=["alpha beta"],
                metric=_TASK_METRICS[task],
                max_new_tokens=LONGBENCH_MAX_NEW_TOKENS[task],
                extra={"task": task, "index": index},
            )

    def _run(self, *extra):
        from tq_longbench import run_longbench

        out_file = self.root / "longbench.jsonl"
        argv = [
            "--model-path", str(self.model_path),
            "--device", "cpu",
            "--backends", "dense_reference",
            "--tasks", ",".join(self.TASKS),
            "--contexts", ",".join(str(c) for c in self.CONTEXTS),
            "--limit", str(self.ITEMS),
            "--max-new-tokens", "3",
            "--dtype", "float32",
            "--chunk-size", "32",
            "--out-file", str(out_file),
            *extra,
        ]  # fmt: skip
        with (
            mock.patch.object(run_longbench, "load_tokenizer", return_value=_ByteTokenizer()),
            mock.patch.object(run_longbench, "load_longbench", self._items),
            contextlib.redirect_stdout(io.StringIO()) as out,
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            self.assertEqual(run_longbench.main(argv), 0)
        records = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
        return records, out.getvalue(), err.getvalue()

    def test_every_task_at_every_rung_leaves_its_items(self):
        records, _, _ = self._run()
        self.assertEqual(len(records), len(self.TASKS) * len(self.CONTEXTS) * self.ITEMS)
        seen = {(r["context_requested"], r["task"]) for r in records}
        self.assertEqual(seen, {(c, t) for c in self.CONTEXTS for t in self.TASKS})

    def test_each_record_carries_the_score_and_the_cost(self):
        records, _, _ = self._run()
        for record in records:
            with self.subTest(task=record["task"], context=record["context_requested"]):
                self.assertEqual(record["metric"], "rouge_l" if record["task"] == "gov_report" else "f1")
                self.assertGreaterEqual(record["score"], 0.0)
                self.assertLessEqual(record["score"], 1.0)
                for key in ("ttft_ms", "decode_p50_us", "decode_p99_us", "kv_cache_mb", "device_peak_mb"):
                    self.assertIn(key, record)
                # These prompts are far longer than the rungs, so every one was cut.
                self.assertEqual(record["truncated"], 1)
                self.assertGreater(record["untruncated_tokens"], record["prompt_tokens"])

    def test_a_rungs_cache_is_sized_for_the_rung(self):
        """One runner per rung, so the memory column describes the rung and not the ladder."""
        records, _, _ = self._run()
        held = {}
        for record in records:
            held.setdefault(record["context_requested"], record["kv_cache_mb"])
        self.assertLess(held[self.CONTEXTS[0]], held[self.CONTEXTS[1]])

    def test_the_table_has_a_line_per_task_and_rung(self):
        _, printed, _ = self._run()
        lines = [line for line in printed.splitlines() if line.startswith("dense_reference")]
        self.assertEqual(len(lines), len(self.TASKS) * len(self.CONTEXTS))
        self.assertIn("cut counts items whose middle was truncated", printed)

    def test_the_prompt_route_and_stop_set_are_reported(self):
        """The same two lines smoke_glm prints: a score must not be measuring a missing marker."""
        _, _, logged = self._run()
        self.assertIn("GLM-4 turn markers", logged)
        self.assertIn("stop ids: [", logged)

    def test_profile_layers_prints_the_table_and_is_off_by_default(self):
        _, _, quiet = self._run()
        self.assertNotIn("layer probe", quiet)
        _, _, loud = self._run("--profile-layers")
        self.assertIn("layer probe", loud)
        self.assertIn("hidden_cos", loud)
        # One probe per (rung, task), on the first item only.
        self.assertEqual(loud.count("layer probe"), len(self.TASKS) * len(self.CONTEXTS))

    def test_an_unknown_task_is_refused_by_name(self):
        from tq_longbench.run_longbench import parse_tasks

        self.assertEqual(parse_tasks("longbench_qasper,gov_report"), ("qasper", "gov_report"))
        with self.assertRaisesRegex(ValueError, "no prompt template"):
            parse_tasks("narrativeqa,not_a_task")
        with self.assertRaisesRegex(ValueError, "no prompt template"):
            parse_tasks("")


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

    def test_a_decode_that_raises_leaves_the_ids_rather_than_ending_the_run(self):
        """GLM-4's tokenizer raises for an id outside its tiktoken ranks.

        Which is exactly what a run worth looking at produces: a wrong basis or a
        truncated context emits ids no vocabulary has. Taking the run down there
        would lose the latency and the memory the item had already measured, and
        the ids themselves, which are the evidence.
        """
        from tq_longbench.smoke_glm import decode_text, render

        class _Raises:
            def decode(self, token_ids, skip_special_tokens=True):
                raise KeyError("<|endoftext|>")

        self.assertEqual(decode_text(_Raises(), [7, 8]), ("", "KeyError: '<|endoftext|>'"))
        rendered = render(_Raises(), [7, 8])
        self.assertIn("undecodable", rendered)
        self.assertIn("ids [7, 8]", rendered)
        # No tokenizer at all (--dummy-prompt) is not a failure, just ids.
        self.assertEqual(render(None, [7, 8]), "ids [7, 8]")

    def test_a_decode_that_works_is_the_text_and_no_failure(self):
        from tq_longbench.smoke_glm import decode_text, render

        class _Works:
            def decode(self, token_ids, skip_special_tokens=True):
                return "hello"

        self.assertEqual(decode_text(_Works(), [1]), ("hello", None))
        self.assertEqual(render(_Works(), [1]), "'hello'")


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

    def _with_kernel_library(self) -> Path:
        kernels = self.library.parent / _ascend.TURBOQUANT_KERNELS_LIB_NAME
        kernels.write_bytes(b"not a real library either")
        os.environ[_ascend.TURBOQUANT_LIB_ENV] = str(self.library)
        return kernels

    def test_the_kernel_library_is_preloaded_globally_before_the_binding(self):
        """A binding whose $ORIGIN did not survive the link still finds its kernels by SONAME."""
        kernels = self._with_kernel_library()
        order = mock.Mock()
        order.attach_mock(self.enterPatch(mock.patch.object(_ascend.ctypes, "CDLL")), "cdll")
        order.attach_mock(
            self.enterPatch(mock.patch.object(torch.ops, "load_library", side_effect=self._registers_the_operators)),
            "load_library",
        )
        self._cube_backend()
        self.assertEqual(
            order.mock_calls,
            [
                mock.call.cdll(str(kernels), mode=ctypes.RTLD_GLOBAL),
                mock.call.load_library(str(self.library)),
            ],
        )

    def test_without_a_kernel_library_beside_it_nothing_is_preloaded(self):
        os.environ[_ascend.TURBOQUANT_LIB_ENV] = str(self.library)
        cdll = self.enterPatch(mock.patch.object(_ascend.ctypes, "CDLL"))
        self.enterPatch(mock.patch.object(torch.ops, "load_library", side_effect=self._registers_the_operators))
        self._cube_backend()
        cdll.assert_not_called()

    def test_a_failed_preload_is_reported_and_the_binding_left_alone(self):
        kernels = self._with_kernel_library()
        self.enterPatch(
            mock.patch.object(_ascend.ctypes, "CDLL", side_effect=OSError("libc_sec.so: cannot open shared object"))
        )
        loader = self.enterPatch(mock.patch.object(torch.ops, "load_library"))
        with self.assertRaisesRegex(RuntimeError, "not registered") as raised:
            self._cube_backend()
        self.assertIn("preloading the kernels beside", str(raised.exception))
        self.assertIn("libc_sec.so: cannot open shared object", str(raised.exception))
        self.assertTrue(kernels.is_file())
        loader.assert_not_called()

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


# The keywords torch_npu 2.10.0.post4 declares for each entry point (its npu:: schemas),
# so a call spelled for another version is refused here as it would be there.
_FIA_V2_SCHEMA_WORDS = (
    "query key value query_rope key_rope pse_shift atten_mask actual_seq_qlen actual_seq_kvlen block_table "
    "dequant_scale_query dequant_scale_key dequant_offset_key dequant_scale_value dequant_offset_value "
    "dequant_scale_key_rope quant_scale_out quant_offset_out quant_scale_p learnable_sink num_query_heads "
    "num_key_value_heads softmax_scale pre_tokens next_tokens input_layout sparse_mode block_size "
    "query_quant_mode key_quant_mode value_quant_mode inner_precise return_softmax_lse query_dtype key_dtype "
    "value_dtype query_rope_dtype key_rope_dtype key_shared_prefix_dtype value_shared_prefix_dtype "
    "dequant_scale_query_dtype dequant_scale_key_dtype dequant_scale_value_dtype dequant_scale_key_rope_dtype "
    "out_dtype"
)
_FIA_V2_KWARGS = frozenset(_FIA_V2_SCHEMA_WORDS.split())
_FIA_V1_SCHEMA_WORDS = (
    "query key value pse_shift atten_mask actual_seq_lengths actual_seq_lengths_kv dequant_scale1 quant_scale1 "
    "dequant_scale2 quant_scale2 quant_offset2 antiquant_scale antiquant_offset key_antiquant_scale "
    "key_antiquant_offset value_antiquant_scale value_antiquant_offset block_table query_padding_size "
    "kv_padding_size key_shared_prefix value_shared_prefix actual_shared_prefix_len query_rope key_rope "
    "key_rope_antiquant_scale num_heads scale pre_tokens next_tokens input_layout num_key_value_heads "
    "sparse_mode inner_precise block_size antiquant_mode key_antiquant_mode value_antiquant_mode "
    "softmax_lse_flag"
)
_FIA_V1_KWARGS = frozenset(_FIA_V1_SCHEMA_WORDS.split())
_PFA_SCHEMA_WORDS = (
    "query key value padding_mask atten_mask pse_shift actual_seq_lengths deq_scale1 quant_scale1 deq_scale2 "
    "quant_scale2 quant_offset2 num_heads scale_value pre_tokens next_tokens input_layout num_key_value_heads "
    "actual_seq_lengths_kv sparse_mode"
)
_PFA_KWARGS = frozenset(_PFA_SCHEMA_WORDS.split())
_EZ9903 = (
    "AclNN_Runtime_Error(EZ9903): Interface aclnnFusedInferAttentionScore versions V1 to V4 are no longer "
    "supported on Ascend950."
)


def _dense_attention(query, key, value, scale, causal):
    """``[q, H, D]`` over ``[k, H_kv, D]``; bottom-right causal when ``causal``."""
    if causal:
        return exact_attention(query, key, value, scale)
    group = query.shape[1] // key.shape[1]
    key, value = key.repeat_interleave(group, dim=1).float(), value.repeat_interleave(group, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", query.float(), key) * scale
    return torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), value)


def fake_torch_npu(v5=True, v1=True, pfa=True, refused=(), ignore_causal=False):
    """torch_npu's three attention entry points, computing what the device computes.

    ``refused`` names entry points that raise EZ9903, as V1-V4 do on Ascend 950;
    ``ignore_causal`` makes FIA V5 read sparse_mode 3 as no mask at all -- a call
    that runs and returns plausible numbers, which only the probe's comparison catches.
    """
    module = types.ModuleType("torch_npu")

    def check(name, allowed, kwargs):
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise TypeError(f"{name}() got unexpected keyword arguments {unknown}")
        if name in refused:
            raise RuntimeError(_EZ9903)

    def paged(cache, table_row, length, kv_heads):
        return cache[table_row.long()].reshape(-1, kv_heads, cache.shape[-1] // kv_heads)[:length]

    def fia(query, key, value, *, q_ends, kv_lens, table, kv_heads, scale, sparse_mode, mask):
        if sparse_mode == 3 and (mask is None or tuple(mask.shape) != (2048, 2048)):
            raise RuntimeError("sparse_mode 3 needs the compressed 2048 x 2048 atten_mask")
        causal = sparse_mode == 3 and not ignore_causal
        out, start = torch.empty_like(query), 0
        for row, (end, length) in enumerate(zip(q_ends, kv_lens)):
            keys, values = (paged(cache, table[row], length, kv_heads) for cache in (key, value))
            out[start:end] = _dense_attention(query[start:end], keys, values, scale, causal).to(out.dtype)
            start = end
        return out, torch.empty(0)

    if v5:

        def npu_fused_infer_attention_score_v2(query, key, value, **kwargs):
            check("fia_v5", _FIA_V2_KWARGS, kwargs)
            assert kwargs["input_layout"] == "TND"
            return fia(
                query, key, value, q_ends=kwargs["actual_seq_qlen"], kv_lens=kwargs["actual_seq_kvlen"],
                table=kwargs["block_table"], kv_heads=kwargs["num_key_value_heads"], scale=kwargs["softmax_scale"],
                sparse_mode=kwargs.get("sparse_mode", 0), mask=kwargs.get("atten_mask"),
            )  # fmt: skip

        module.npu_fused_infer_attention_score_v2 = npu_fused_infer_attention_score_v2
    if v1:

        def npu_fused_infer_attention_score(**kwargs):
            check("fia_v1", _FIA_V1_KWARGS, kwargs)
            return fia(
                kwargs["query"], kwargs["key"], kwargs["value"], q_ends=kwargs["actual_seq_lengths"],
                kv_lens=kwargs["actual_seq_lengths_kv"], table=kwargs["block_table"],
                kv_heads=kwargs["num_key_value_heads"], scale=kwargs["scale"],
                sparse_mode=kwargs.get("sparse_mode", 0), mask=kwargs.get("atten_mask"),
            )  # fmt: skip

        module.npu_fused_infer_attention_score = npu_fused_infer_attention_score
    if pfa:

        def npu_prompt_flash_attention(query, key, value, **kwargs):
            check("pfa", _PFA_KWARGS, kwargs)
            assert kwargs["input_layout"] == "BSH" and kwargs["sparse_mode"] == 3
            heads, kv_heads = kwargs["num_heads"], kwargs["num_key_value_heads"]
            q = query[0].view(query.shape[1], heads, -1)
            k, v = (t[0].view(t.shape[1], kv_heads, -1) for t in (key, value))
            out = _dense_attention(q, k, v, kwargs["scale_value"], causal=True).to(query.dtype)
            return out.reshape(query.shape)

        module.npu_prompt_flash_attention = npu_prompt_flash_attention
    return module


# GLM-4 9B-chat-1M's attention geometry, which the pre-flight probes with.
GLM4_LAYER = LayerShape(32, 4, 128, 128**-0.5)


class TestCANNDenseBackend(unittest.TestCase):
    """The unquantised baseline that is not a fused kernel: matmul, softmax, matmul.

    Held to exact attention in float32 rather than to the fused path, because the
    reason it exists is that on Ascend 950 there is no fused path to compare it
    with. Everything here runs on the CPU, where the same torch operators
    dispatch to the CPU rather than to the Cube and Vector units -- what is under
    test is the masking, the grouping and the tiling, none of which are the
    device's.
    """

    LENGTH = 300
    SHAPE = LayerShape(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE**-0.5)

    def _filled(self, length=None):
        torch.manual_seed(0)
        length = length or self.LENGTH
        backend = CANNDenseBackend(_geometry(max(BLOCK_SIZE, length)), self.SHAPE, CPU, torch.float16)
        key = torch.randn(length, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16)
        value = torch.randn(length, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16)
        backend.write_kv(0, key, value, backend.cache.slot_mapping(0, length))
        return backend, key, value

    def test_the_cache_it_decodes_out_of_is_unquantised(self):
        backend, key, _ = self._filled()
        self.assertIsInstance(backend.cache, DenseKVCache)
        self.assertEqual(backend.cache.key_planes[0].dtype, torch.float16)
        # Not a codec, a copy: what was written is what is there, bit for bit.
        held = backend.cache.key_planes[0].view(-1, NUM_KV_HEADS, HEAD_SIZE)[: self.LENGTH]
        self.assertTrue(torch.equal(held, key))

    def test_a_decode_is_exact_attention_over_each_rows_own_context(self):
        backend, key, value = self._filled()
        query = torch.randn(3, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
        lengths = torch.tensor([1, 177, self.LENGTH], dtype=torch.int32)
        out = torch.empty(3, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
        backend.decode(0, query, lengths, out)
        for row, length in enumerate(lengths.tolist()):
            with self.subTest(context=length):
                expected = exact_attention(query[row : row + 1], key[:length], value[:length], self.SHAPE.scale)
                self.assertGreater(cosine(out[row : row + 1], expected), EXACT_COSINE)

    def test_a_prefill_chunk_is_bottom_right_causal(self):
        """A chunk arriving after a populated cache: row r sees the prefix plus r of its own."""
        written, count = 200, 100
        backend, key, value = self._filled(written + count)
        query = torch.randn(count, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
        out = torch.empty(count, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
        backend.prefill_chunk(0, query, written + count, out)
        expected = exact_attention(query, key, value, self.SHAPE.scale)
        self.assertGreater(cosine(out, expected), EXACT_COSINE)
        # Row by row too, so a mask that is causal only on average cannot pass.
        for row in (0, count // 2, count - 1):
            with self.subTest(row=row):
                visible = written + row + 1
                one = exact_attention(query[row : row + 1], key[:visible], value[:visible], self.SHAPE.scale)
                self.assertGreater(cosine(out[row : row + 1], one), EXACT_COSINE)

    def test_tiling_the_score_matrix_does_not_change_the_answer(self):
        """The tile size is an HBM budget, so it has to be invisible in the result."""
        backend, _, _ = self._filled()
        query = torch.randn(64, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
        answers = []
        for budget in (ops_module._SCORE_TILE_ELEMENTS, NUM_HEADS * self.LENGTH):
            with mock.patch.object(ops_module, "_SCORE_TILE_ELEMENTS", budget):
                out = torch.empty(64, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
                backend.prefill_chunk(0, query, self.LENGTH, out)
                answers.append(out.clone())
        # All 64 rows in one tile against one row per tile: float ordering apart, the same.
        self.assertGreater(cosine(answers[0], answers[1]), 0.9999999)

    def test_a_row_with_no_context_is_refused_rather_than_returning_nan(self):
        """An all-masked softmax row is NaN, and nothing downstream raises on NaN."""
        backend, _, _ = self._filled()
        query = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
        out = torch.empty(1, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "attends to nothing"):
            backend.decode(0, query, torch.tensor([0], dtype=torch.int32), out)

    def test_a_context_past_the_pool_is_refused(self):
        backend, _, _ = self._filled()
        query = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
        out = torch.empty(1, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, "exceeds the"):
            backend.decode(0, query, torch.tensor([10**6], dtype=torch.int32), out)

    def test_it_is_built_where_the_ascend_backends_are_refused(self):
        """No Ascend C operator to be missing, so no device to be refused on.

        This is why it is a backend of its own rather than another ``native_v5``
        entry point: it is the unquantised path that is always there, and
        ``build_backend`` must not treat it like a kernel. The TurboQuant
        backends beside it stay refused, operators registered or not.
        """
        geometry = _geometry(512)
        self.assertIn("cann_dense", DENSE_BACKENDS)
        self.assertEqual(reference_equivalent("cann_dense"), "dense_reference")
        built = build_backend("cann_dense", geometry, self.SHAPE, CPU, torch.float16)
        self.assertIsInstance(built, CANNDenseBackend)
        with self.assertRaisesRegex(ValueError, "cannot run on cuda"):
            build_backend("turboquant_aiv", geometry, self.SHAPE, torch.device("cuda"), torch.float16)

    def test_it_clears_the_preflight_on_a_glm4_layer(self):
        """What smoke_glm runs before any weights load, at GLM-4's own head counts."""
        result = probe_backend("cann_dense", GLM4_LAYER, CPU, torch.float16, BLOCK_SIZE)
        self.assertTrue(result.passed, result.describe())
        self.assertIn("prefill cos", result.detail)

    def test_the_staging_pool_falls_back_to_it_when_no_fused_path_runs(self):
        """The Ascend 950 case, with the fused candidate failing as it does there.

        ``native_v5`` is offered first and cannot even be built here (no
        torch_npu), which is the same verdict a 950 reaches by another route --
        every entry point refused. What matters is that the selection does not
        stop there but settles on ``cann_dense``, and still reports the failure.
        """
        candidates = mock.patch.object(
            preflight_module, "dense_backend_candidates", return_value=["native_v5", "cann_dense"]
        )
        with candidates:
            selection = select_dense_backend(CPU, GLM4_LAYER, torch.float16, BLOCK_SIZE, role="decode")
        self.assertTrue(selection.passed, [result.describe() for result in selection.results])
        self.assertEqual(selection.backend, "cann_dense")
        self.assertTrue(any(not result.passed for result in selection.results), "the failed probe is not reported")


class _FakeNpuCase(unittest.TestCase):
    def setUp(self):
        self._serving = False

    def use(self, module) -> None:
        """Make ``module`` the torch_npu this test sees; may be called again to swap it."""
        patcher = mock.patch.dict(sys.modules, {"torch_npu": module})
        patcher.start()
        self.addCleanup(patcher.stop)
        # native_v5 builds on a CPU only while something serves the TurboQuant operators.
        # Registered once: torch 2.10 refuses a second Python kernel for the same key.
        if not self._serving:
            ops = cpu_turboquant_ops()
            ops.__enter__()
            self.addCleanup(ops.__exit__, None, None, None)
            self._serving = True


class TestNativeAttentionApis(_FakeNpuCase):
    """NativeV5Backend's three torch_npu entry points, against exact attention."""

    CONTEXT, CHUNK = 48, 16

    def _filled(self, api):
        geometry = _geometry(BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        backend = NativeV5Backend(geometry, LayerShape(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE**-0.5), CPU,
                                  torch.float32, attention_api=api)  # fmt: skip
        key = torch.randn(self.CONTEXT, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(self.CONTEXT, NUM_KV_HEADS, HEAD_SIZE)
        backend.write_kv(0, key, value, backend.cache.slot_mapping(0, self.CONTEXT))
        return backend, key, value

    def _check_prefill(self, api):
        backend, key, value = self._filled(api)
        query = torch.randn(self.CHUNK, NUM_HEADS, HEAD_SIZE)
        out = torch.empty_like(query)
        backend.prefill_chunk(0, query, self.CONTEXT, out)
        torch.testing.assert_close(out, exact_attention(query, key, value, HEAD_SIZE**-0.5), rtol=1e-4, atol=1e-5)

    def _check_decode(self, api):
        backend, key, value = self._filled(api)
        query = torch.randn(3, NUM_HEADS, HEAD_SIZE)
        lengths = torch.tensor([self.CONTEXT, 20, 7], dtype=torch.int32)
        out = torch.empty_like(query)
        backend.decode(0, query, lengths, out)
        for row, length in enumerate(lengths.tolist()):
            expected = exact_attention(query[row : row + 1], key[:length], value[:length], HEAD_SIZE**-0.5)
            torch.testing.assert_close(out[row : row + 1], expected, rtol=1e-4, atol=1e-5)

    def test_v5_is_the_default_where_torch_npu_has_it(self):
        self.use(fake_torch_npu())
        self.assertEqual(self._filled(None)[0].attention_api, "fia_v5")
        self.use(fake_torch_npu(v5=False))
        self.assertEqual(self._filled(None)[0].attention_api, "fia_v1")

    def test_fia_v5_prefills_bottom_right_causal_and_decodes_ragged_rows(self):
        self.use(fake_torch_npu())
        self._check_prefill("fia_v5")
        self._check_decode("fia_v5")

    def test_fia_v1_now_passes_the_mask_sparse_mode_3_reads(self):
        self.use(fake_torch_npu())
        self._check_prefill("fia_v1")
        self._check_decode("fia_v1")

    def test_pfa_prefills_the_contiguous_prefix_and_refuses_to_decode(self):
        self.use(fake_torch_npu())
        self._check_prefill("pfa")
        with self.assertRaisesRegex(NotImplementedError, "no block table"):
            self._check_decode("pfa")


class TestPreflight(_FakeNpuCase):
    """The synthetic one-layer probe, at GLM-4's head counts."""

    def test_on_ascend_950_the_v5_entry_point_is_chosen(self):
        self.use(fake_torch_npu(refused={"fia_v1"}))
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                chosen = select_dense_api("native_v5", GLM4_LAYER, CPU, torch.float32, BLOCK_SIZE, role=role)
                self.assertTrue(chosen.passed)
                self.assertEqual(chosen.api, "fia_v5")

    def test_a_torch_npu_without_v5_prefills_through_pfa_and_cannot_decode(self):
        self.use(fake_torch_npu(v5=False, refused={"fia_v1"}))
        prefill = select_dense_api("native_v5", GLM4_LAYER, CPU, torch.float32, BLOCK_SIZE, role="prefill")
        self.assertEqual((prefill.passed, prefill.api), (True, "pfa"))
        self.assertIn("EZ9903", prefill.results[0].detail)
        decode = select_dense_api("native_v5", GLM4_LAYER, CPU, torch.float32, BLOCK_SIZE, role="decode")
        self.assertFalse(decode.passed)

    def test_a_call_that_runs_but_ignores_the_causal_mask_fails_the_probe(self):
        self.use(fake_torch_npu(ignore_causal=True))
        result = probe_backend("native_v5", GLM4_LAYER, CPU, torch.float32, BLOCK_SIZE, dense_attention_api="fia_v5")
        self.assertFalse(result.passed)
        self.assertIn("prefill cos", result.detail)

    def test_the_quantised_backends_clear_their_floor(self):
        self.use(fake_torch_npu())
        for name, folded in (("turboquant_aiv", False), ("turboquant_aiv", True), ("turboquant_reference", True)):
            with self.subTest(backend=name, folded=folded):
                result = probe_backend(name, GLM4_LAYER, CPU, torch.bfloat16, BLOCK_SIZE, output_rotation_folded=folded)
                self.assertTrue(result.passed, result.describe())

    def test_a_backend_that_cannot_be_built_is_a_verdict_not_a_crash(self):
        refused = probe_backend("turboquant_cube", GLM4_LAYER, CPU, torch.float16, BLOCK_SIZE)
        self.assertFalse(refused.passed)
        self.assertIn("cannot run on cpu", refused.detail)
        # Operators served, but no Cube library anywhere to load it from.
        self.use(fake_torch_npu())
        missing = str(REPO_ROOT / "build" / "no-such-dir" / _ascend.TURBOQUANT_LIB_NAME)
        with mock.patch.dict(os.environ, {_ascend.TURBOQUANT_LIB_ENV: missing}):
            unregistered = probe_backend("turboquant_cube", GLM4_LAYER, CPU, torch.float16, BLOCK_SIZE)
        self.assertFalse(unregistered.passed)
        self.assertIn("not registered", unregistered.detail)


class TestSmokeGlmPreflight(unittest.TestCase):
    """How smoke_glm acts on the probes: switch, stop, or carry the chosen entry point into the runners."""

    def _args(self, *extra):
        from tq_longbench.smoke_glm import build_parser

        args = build_parser().parse_args(["--model-path", "/m", "--device", "cpu", *extra])
        args.device = CPU
        return args

    def _plan(self, args, prefill_mode, dense_passes, decode_passes=True, dense_backend="native_v5"):
        """``plan_attention`` with every probe answered, so only its branching is under test.

        Both selectors are stubbed with the same verdict: an explicitly named
        dense backend goes through ``select_dense_api``, the staging pool through
        ``select_dense_backend``, and which one a case exercises is the case's point.
        """
        from tq_longbench import smoke_glm
        from tq_longbench.preflight import DenseSelection, ProbeResult

        dense = DenseSelection(dense_passes, "fia_v5" if dense_passes else None,
                               [ProbeResult("native_v5", "fia_v5", dense_passes, 0.01, _EZ9903)],
                               dense_backend if dense_passes else None)  # fmt: skip
        decode = ProbeResult(args.backend, None, decode_passes, 0.01, "decode cos 0.99")
        with (
            mock.patch.object(smoke_glm, "select_dense_api", return_value=dense),
            mock.patch.object(smoke_glm, "select_dense_backend", return_value=dense),
            mock.patch.object(smoke_glm, "probe_backend", return_value=decode),
        ):
            return smoke_glm.plan_attention(args, GLM4_9B_CHAT_1M_CONFIG, prefill_mode)

    def test_a_failed_default_dense_prefill_switches_to_batched_decode(self):
        plan = self._plan(self._args("--backend", "turboquant_reference"), "dense_staging", dense_passes=False)
        self.assertEqual(plan.prefill_mode, "batched_decode")
        self.assertIn("batched_decode", plan.notes[0])

    def test_an_explicit_dense_staging_stops_before_the_weights(self):
        from tq_longbench.smoke_glm import PreflightFailed

        args = self._args("--backend", "turboquant_reference", "--prefill-mode", "dense_staging")
        with self.assertRaisesRegex(PreflightFailed, "--prefill-mode batched_decode") as raised:
            self._plan(args, "dense_staging", dense_passes=False)
        self.assertIn("EZ9903", str(raised.exception))

    def test_a_failed_decode_stops_before_the_weights(self):
        from tq_longbench.smoke_glm import PreflightFailed

        with self.assertRaisesRegex(PreflightFailed, "decode does not run here"):
            self._plan(self._args("--backend", "turboquant_reference"), "batched_decode", True, decode_passes=False)

    def test_the_entry_point_that_passed_is_the_one_the_runners_get(self):
        plan = self._plan(self._args("--backend", "turboquant_reference"), "dense_staging", dense_passes=True)
        self.assertEqual(plan.prefill_mode, "dense_staging")
        self.assertEqual(plan.dense_api_for("turboquant_reference"), "fia_v5")
        self.assertEqual(plan.staging_backend, "native_v5")
        native = self._plan(self._args("--backend", "native_v5"), "dense_staging", dense_passes=True)
        self.assertEqual(native.dense_api_for("native_v5"), "fia_v5")

    def test_a_staging_pool_that_fell_back_is_named_and_carried(self):
        """The 950 case: no fused entry point runs, so the pool stages through cann_dense.

        The prefill mode is untouched -- the pool exists, it is just not the
        preferred one -- and which backend it is has to reach the runner, or the
        engine would build the one that just failed.
        """
        args = self._args("--backend", "turboquant_reference")
        plan = self._plan(args, "dense_staging", dense_passes=True, dense_backend="cann_dense")
        self.assertEqual(plan.prefill_mode, "dense_staging")
        self.assertEqual(plan.staging_backend, "cann_dense")
        self.assertTrue(any("cann_dense" in note for note in plan.notes), plan.notes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
