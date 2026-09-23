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

import contextlib
import ctypes
import io
import json
import math
import os
import re
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
from tq_longbench import layers as layers_module  # noqa: E402
from tq_longbench import ops as ops_module  # noqa: E402
from tq_longbench import preflight as preflight_module  # noqa: E402
from tq_longbench._ascend import assert_no_vllm_imported, turboquant_layout, turboquant_rotation  # noqa: E402
from tq_longbench.cpu_reference import cpu_turboquant_ops  # noqa: E402
from tq_longbench.engine import (  # noqa: E402
    GraphCaptureUnavailable,
    RunMetrics,
    RunnerConfig,
    StandaloneModelRunner,
    graph_windows,
    read_checkpoint_shape,
    should_cast_to_nz,
)
from tq_longbench.families import (  # noqa: E402
    BODY,
    GENERIC,
    GLM4,
    QWEN2,
    ModelFamily,
    family_for,
    special_token_id,
)
from tq_longbench.glm4 import (  # noqa: E402
    GLM4_BATCHED_DECODE_MIN_TOKENS,
    glm4_checkpoint_targets,
    glm4_model_shape,
    glm4_prefill_mode,
    is_glm4_config,
)
from tq_longbench import kv_dump  # noqa: E402
from tq_longbench.kv_cache import CacheGeometry, DenseKVCache, TurboQuantKVCache, dense_equivalent_bytes  # noqa: E402
from tq_longbench.kv_dump import (  # noqa: E402
    collect_tensors,
    default_run_name,
    dump_quantised_cache,
    normalise_dump_dir,
    run_directory,
)
from tq_longbench.layers import (  # noqa: E402
    MLP,
    CausalLM,
    DecodeWorkspace,
    ModelShape,
    RMSNorm,
    RotaryEmbedding,
    fold_on_device,
    fold_output_projection_in_place,
    fold_precision,
    npu_operator,
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
    cached_extent,
    encode,
    eos_ids_for,
    model_directory,
    prompt_route,
    require_dumpable,
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
                # The projections are fused in the module tree; here they are
                # taken apart again, so this stays a transcription of the model
                # rather than a second call into the thing under test.
                query, key, value = attn.qkv_proj(normed).split(attn.qkv_split, dim=-1)
                gate = None
                if shape.attn_output_gate:
                    query, gate = query.chunk(2, dim=-1)
                    gate = gate.reshape(count, shape.num_heads, shape.head_size)
                query = attn.q_norm(query.reshape(count, shape.num_heads, shape.head_size))
                key = attn.k_norm(key.reshape(count, shape.num_kv_heads, shape.head_size))
                value = value.reshape(count, shape.num_kv_heads, shape.head_size)
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


# Qwen2.5-3B-Instruct's published config.json, verbatim. The one checkpoint
# these cases are about that is too large to write down.
QWEN2_5_3B_INSTRUCT_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 32768,
    "max_window_layers": 70,
    "model_type": "qwen2",
    "num_attention_heads": 16,
    "num_hidden_layers": 36,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 32768,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}

# A two-layer Qwen2 small enough to run eagerly, at the 3B's 8:1 GQA ratio.
# D=64 because Pi needs a power of two; the vocabulary is tiny and untied, so
# that a wrong lm_head cannot hide behind the embedding.
TINY_QWEN2 = {
    **QWEN2_5_3B_INSTRUCT_CONFIG,
    "hidden_size": 256,
    "intermediate_size": 192,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rope_theta": 5.0e6,
    "tie_word_embeddings": False,
    "vocab_size": 320,
}


def _qwen2_rope(x: torch.Tensor, positions: torch.Tensor, head_size: int, theta: float) -> torch.Tensor:
    """HF ``modeling_qwen2``'s rotary, transcribed: NeoX halves over the whole head.

    The contrast with :func:`_chatglm_rope` is the whole of what ``model_type``
    decides about position here -- the same frequencies, paired ``(i, i + d/2)``
    instead of ``(2i, 2i+1)``, and applied to all of the head rather than its
    first half. Pairing them the other way does not raise; it scrambles position.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size))
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)
    half = head_size // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def _qwen2_checkpoint(config: dict, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random weights under Qwen2's tensor names: split q/k/v, with the biases only it carries."""
    generator = torch.Generator().manual_seed(seed)
    hidden, heads = config["hidden_size"], config["num_attention_heads"]
    groups, ffn, vocab = config["num_key_value_heads"], config["intermediate_size"], config["vocab_size"]
    head_size = hidden // heads

    def rand(*shape, scale=0.05):
        return torch.randn(*shape, generator=generator) * scale

    tensors = {
        "model.embed_tokens.weight": rand(vocab, hidden, scale=1.0),
        "model.norm.weight": 1.0 + rand(hidden),
        "lm_head.weight": rand(vocab, hidden),
    }
    for index in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{index}."
        tensors.update(
            {
                prefix + "input_layernorm.weight": 1.0 + rand(hidden),
                prefix + "post_attention_layernorm.weight": 1.0 + rand(hidden),
                prefix + "self_attn.q_proj.weight": rand(heads * head_size, hidden),
                prefix + "self_attn.q_proj.bias": rand(heads * head_size, scale=0.5),
                prefix + "self_attn.k_proj.weight": rand(groups * head_size, hidden),
                prefix + "self_attn.k_proj.bias": rand(groups * head_size, scale=0.5),
                prefix + "self_attn.v_proj.weight": rand(groups * head_size, hidden),
                prefix + "self_attn.v_proj.bias": rand(groups * head_size, scale=0.5),
                prefix + "self_attn.o_proj.weight": rand(hidden, heads * head_size),
                prefix + "mlp.gate_proj.weight": rand(ffn, hidden),
                prefix + "mlp.up_proj.weight": rand(ffn, hidden),
                prefix + "mlp.down_proj.weight": rand(hidden, ffn),
            }
        )
    return tensors


def _qwen2_eager_logits(config: dict, tensors: dict[str, torch.Tensor], token_ids: torch.Tensor) -> torch.Tensor:
    """Qwen2's forward pass in float32, written from modeling_qwen2.py rather than from layers.py."""
    heads, groups = config["num_attention_heads"], config["num_key_value_heads"]
    head_size = config["hidden_size"] // heads
    eps, theta = config["rms_norm_eps"], config["rope_theta"]
    count = token_ids.numel()
    positions = torch.arange(count)

    def rms(x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    hidden = tensors["model.embed_tokens.weight"][token_ids]
    for index in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{index}."

        def project(name, width, x, prefix=prefix):
            out = x @ tensors[prefix + f"self_attn.{name}.weight"].T + tensors[prefix + f"self_attn.{name}.bias"]
            return out.view(count, width, head_size)

        normed = rms(hidden, tensors[prefix + "input_layernorm.weight"])
        query = _qwen2_rope(project("q_proj", heads, normed), positions, head_size, theta)
        key = _qwen2_rope(project("k_proj", groups, normed), positions, head_size, theta)
        value = project("v_proj", groups, normed)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(0, 1),
            key.transpose(0, 1).repeat_interleave(heads // groups, dim=0),
            value.transpose(0, 1).repeat_interleave(heads // groups, dim=0),
            is_causal=True,
        ).transpose(0, 1)
        hidden = hidden + attended.reshape(count, -1) @ tensors[prefix + "self_attn.o_proj.weight"].T
        normed = rms(hidden, tensors[prefix + "post_attention_layernorm.weight"])
        gate = normed @ tensors[prefix + "mlp.gate_proj.weight"].T
        up = normed @ tensors[prefix + "mlp.up_proj.weight"].T
        hidden = hidden + (torch.nn.functional.silu(gate) * up) @ tensors[prefix + "mlp.down_proj.weight"].T
    return rms(hidden, tensors["model.norm.weight"]) @ tensors["lm_head.weight"].T


class TestModelFamilies(unittest.TestCase):
    """Which family a ``config.json`` is, and what each one lends a run.

    The dispatch is the whole of what ``model_type`` decides outside the shape,
    and every branch of it has a way of failing silently: a GLM-4 stop set lent
    to a Qwen checkpoint stops on ids that mean nothing there, a Qwen one lent to
    GLM-4 never stops at all, and a family invented for an unknown checkpoint
    would do either.
    """

    def test_each_model_type_finds_its_own_family(self):
        self.assertIs(family_for({"model_type": "qwen2"}), QWEN2)
        self.assertIs(family_for({"model_type": "chatglm"}), GLM4)
        self.assertIs(family_for({"model_type": "glm"}), GLM4)
        self.assertIs(family_for({"model_type": "qwen3"}), GENERIC)
        self.assertIs(family_for({}), GENERIC)

    def test_a_glm_the_harness_cannot_build_is_still_refused_by_name(self):
        """GLM-4-0414's sandwich norms: matching it as a family would load two norms short."""
        with self.assertRaisesRegex(ValueError, "post-attention and post-MLP norms"):
            family_for({"model_type": "glm4"})

    def test_the_two_families_lend_each_other_nothing(self):
        self.assertFalse(set(GLM4.stop_tokens) & set(QWEN2.stop_tokens) - {"<|endoftext|>"})
        # Both call it <|endoftext|>; they do not agree on what it is numbered.
        self.assertNotEqual(GLM4.stop_tokens["<|endoftext|>"], QWEN2.stop_tokens["<|endoftext|>"])
        self.assertEqual(GENERIC.stop_ids(None), set())

    def test_an_unsized_family_never_holds_a_second_kv_pool(self):
        """GENERIC stages at no length: what an fp16 pool would cost is what is not known."""
        for context in (1024, 4096, 131072):
            self.assertEqual(GENERIC.prefill_mode(context, "turboquant_cube"), "batched_decode")
        # Except where the decode itself reads that pool, which is not a policy.
        self.assertEqual(GENERIC.prefill_mode(1024, "cann_dense"), "dense_staging")

    def test_a_sized_family_stages_below_its_threshold_and_not_above(self):
        for family in (GLM4, QWEN2):
            with self.subTest(family=family.name):
                self.assertEqual(family.prefill_mode(16384, "turboquant_cube"), "dense_staging")
                self.assertEqual(family.prefill_mode(32768, "turboquant_cube"), "batched_decode")

    def test_a_turn_that_never_says_where_the_prompt_goes_is_refused(self):
        """The one way to transcribe a template wrongly that would not raise at run time."""
        with self.assertRaisesRegex(ValueError, "BODY exactly once"):
            ModelFamily(name="x", model_types=frozenset(), template_tokens={"<a>": 1}, turn=("<a>",))
        with self.assertRaisesRegex(ValueError, "BODY exactly once"):
            ModelFamily(name="x", model_types=frozenset(), template_tokens={"<a>": 1}, turn=("<a>", BODY, BODY))
        with self.assertRaisesRegex(ValueError, "not a turn"):
            ModelFamily(name="x", model_types=frozenset(), turn=("hello ", BODY))

    def test_a_marker_with_no_id_is_refused(self):
        """It would go through the tokenizer as six characters of text and encode to something."""
        with self.assertRaisesRegex(ValueError, r"\['<\|im_end\|>'\]"):
            ModelFamily(
                name="x",
                model_types=frozenset(),
                template_tokens={"<|im_start|>": 1},
                turn=("<|im_start|>", BODY, "<|im_end|>"),
            )


class TestQwen2Config(unittest.TestCase):
    """The published Qwen2.5-3B-Instruct config, as the harness reads it.

    No family reads this one: ``qwen2`` goes through ``ModelShape.from_hf_config``
    like any HF decoder, and the two fields the config does not state are read off
    the checkpoint's tensor names. So what is pinned here is that the ordinary
    path gets Qwen2.5 right -- the full-width NeoX RoPE in particular, which is
    the default and would therefore never announce itself as a choice.
    """

    def _write_index(self, root: Path, config: dict, tensors) -> str:
        """The config and a weight map, with no shard behind it.

        ``checkpoint_features`` reads names, never tensors, so an index is the
        whole of what a shape needs -- and writing one keeps the 3B's real tensor
        names in the test instead of a 6 GB download.
        """
        path = root / "qwen2"
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        weight_map = {name: "model-00001-of-00001.safetensors" for name in tensors}
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
        return str(path)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        names = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            *(
                f"model.layers.{index}.self_attn.{part}"
                for index in range(QWEN2_5_3B_INSTRUCT_CONFIG["num_hidden_layers"])
                for part in ("q_proj.weight", "q_proj.bias", "k_proj.bias", "v_proj.bias", "o_proj.weight")
            ),
        ]
        self.path = self._write_index(Path(self._tmp.name), QWEN2_5_3B_INSTRUCT_CONFIG, names)

    def test_the_published_config_reads_as_its_attention_contract(self):
        shape = read_checkpoint_shape(self.path)
        self.assertEqual((shape.num_heads, shape.num_kv_heads, shape.head_size), (16, 2, 128))
        self.assertEqual((shape.num_layers, shape.hidden_size, shape.intermediate_size), (36, 2048, 11008))
        self.assertEqual(shape.vocab_size, 151936)
        self.assertEqual(shape.rope_theta, 1.0e6)
        # The two that separate it from GLM-4: all of each head, NeoX halves.
        self.assertIsNone(shape.rotary_dim)
        self.assertFalse(shape.rope_interleaved)
        self.assertFalse(shape.attn_output_gate)
        self.assertTrue(shape.tie_word_embeddings)

    def test_the_bias_and_the_norms_are_read_off_the_checkpoint_not_the_type(self):
        """``attention_bias`` is absent from this config and the biases are there anyway."""
        self.assertNotIn("attention_bias", QWEN2_5_3B_INSTRUCT_CONFIG)
        shape = read_checkpoint_shape(self.path)
        self.assertTrue(shape.qkv_bias)
        # Qwen3 norms each head and Qwen2.5 does not; only the tensors say so.
        self.assertFalse(shape.qk_norm)


class _Qwen2TinyCheckpoint(unittest.TestCase):
    """A tiny Qwen2 written to disk, and the runner that loads it."""

    PROMPT_TOKENS = 40
    NEW_TOKENS = 6

    def setUp(self):
        torch.manual_seed(0)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.tensors = _qwen2_checkpoint(TINY_QWEN2)
        self.token_ids = torch.randint(0, TINY_QWEN2["vocab_size"], (self.PROMPT_TOKENS,), dtype=torch.int64)
        self.model_path = self._write("qwen2", TINY_QWEN2, self.tensors)

    def _write(self, name: str, config: dict, tensors: dict[str, torch.Tensor]) -> str:
        from safetensors.torch import save_file

        path = self.root / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        save_file({key: value.contiguous() for key, value in tensors.items()}, str(path / "model.safetensors"))
        return str(path)

    def _runner(self, backend: str = "dense_reference", **overrides) -> StandaloneModelRunner:
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
        return StandaloneModelRunner(RunnerConfig(model_path=self.model_path, **settings))


class TestQwen2Checkpoint(_Qwen2TinyCheckpoint):
    """The tiny checkpoint loaded and run exactly as Qwen2.5-3B-Instruct would be.

    Eager PyTorch is the reference rather than another path through this harness:
    what is under test is that ``model_type: qwen2`` builds the model the
    checkpoint is, and a comparison against the harness's own reading of the
    config could not tell a wrong RoPE from a consistently wrong one.
    """

    def test_the_qwen2_checkpoint_runs_as_qwen2_does(self):
        runner = self._runner()
        self.assertEqual((runner.shape.num_heads, runner.shape.num_kv_heads, runner.shape.head_size), (4, 2, 64))
        self.assertTrue(runner.shape.qkv_bias)
        logits = runner.prefill(self.token_ids, RunMetrics())
        expected = _qwen2_eager_logits(TINY_QWEN2, self.tensors, self.token_ids)[-1:]
        torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)

    def test_the_greedy_continuation_is_the_eager_one(self):
        ids, expected = self.token_ids.clone(), []
        for _ in range(self.NEW_TOKENS):
            expected.append(int(torch.argmax(_qwen2_eager_logits(TINY_QWEN2, self.tensors, ids)[-1])))
            ids = torch.cat((ids, torch.tensor([expected[-1]])))
        produced, _ = self._runner().generate(self.token_ids)
        self.assertEqual(produced, expected)

    def test_a_tied_lm_head_is_loaded_from_the_embedding(self):
        """Qwen2.5-3B ships no ``lm_head.weight``; an untied harness would run it at random."""
        tied = dict(TINY_QWEN2, tie_word_embeddings=True)
        tensors = {name: value for name, value in self.tensors.items() if name != "lm_head.weight"}
        self.model_path = self._write("qwen2-tied", tied, tensors)
        runner = self._runner()
        self.assertTrue(
            torch.equal(runner.model.lm_head.weight, runner.model.embed_tokens.weight),
        )

    def test_o_proj_is_folded_as_it_is_ingested(self):
        """The folded-o_proj contract is the family's, not GLM-4's: ungated is ungated."""
        folded = self._runner(backend="turboquant_reference", fold_output_rotation=True)
        fold = turboquant_rotation().fold_pi_into_output_projection
        head_size = TINY_QWEN2["hidden_size"] // TINY_QWEN2["num_attention_heads"]
        for index, layer in enumerate(folded.model.layers):
            original = self.tensors[f"model.layers.{index}.self_attn.o_proj.weight"]
            self.assertTrue(torch.equal(layer.self_attn.o_proj.weight, fold(original, head_size)))
        unfolded = self._runner(backend="turboquant_reference")
        self.assertEqual(folded.generate(self.token_ids)[0], unfolded.generate(self.token_ids)[0])

    def test_the_smoke_runner_takes_a_qwen2_checkpoint_end_to_end(self):
        """``smoke_glm.main`` on a ``qwen2`` checkpoint: family, shape, pre-flight, contract, generation.

        ``--dummy-prompt`` because the tiny checkpoint has no tokenizer; what the
        stages report is what is being checked, and the first line of stage 1 is
        the one that used to read "not a GLM-4 checkpoint" and exit.
        """
        from tq_longbench import smoke_glm

        argv = ["--model-path", self.model_path, "--device", "cpu", "--backend", "dense_reference"]
        argv += ["--dtype", "float32", "--dummy-prompt", "--max-new-tokens", "4", "--tokens", "0"]
        refuse = mock.patch.object(smoke_glm, "load_tokenizer", side_effect=AssertionError("tokenizer loaded"))
        with refuse, contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(smoke_glm.main(argv), 0)
        printed = out.getvalue()
        self.assertIn("Qwen2.5 ('qwen2')", printed)
        self.assertIn("4 heads / 2 kv", printed)
        self.assertIn("interleaved=False", printed)
        self.assertIn("rotary_dim None of 64", printed)


class TestQwen2ChatPrompt(unittest.TestCase):
    """ChatML, for a ``transformers`` that cannot apply the template itself."""

    class Tokenizer:
        """Characters as ids, with ChatML's markers declared where this vocabulary has room."""

        SPECIAL = {"<|im_start|>": 300, "<|im_end|>": 301, "<|endoftext|>": 302}

        chat_template = None
        eos_token_id = None
        unk_token_id = 0

        def __init__(self):
            self.added_tokens_encoder = dict(self.SPECIAL)

        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            ids = [ord(character) % 256 for character in text]
            return types.SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.int64))

        def convert_tokens_to_ids(self, token):
            return self.added_tokens_encoder.get(token, self.unk_token_id)

    def test_the_prompt_is_the_documented_format(self):
        """``<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n``, assembled by id."""
        tokenizer = self.Tokenizer()
        ids = encode(tokenizer, "HI", use_chat_template=True, family=QWEN2).tolist()
        start, end = tokenizer.SPECIAL["<|im_start|>"], tokenizer.SPECIAL["<|im_end|>"]
        text = lambda s: [ord(character) for character in s]  # noqa: E731
        self.assertEqual(ids, [start, *text("user\nHI"), end, *text("\n"), start, *text("assistant\n")])

    def test_the_markers_resolve_through_the_tokenizer_before_the_constants(self):
        """A vocabulary of 320 cannot hold 151644, and nothing here assumes it can."""
        ids = encode(self.Tokenizer(), "HI", use_chat_template=True, family=QWEN2).tolist()
        self.assertNotIn(151644, ids)
        self.assertEqual(ids[0], self.Tokenizer.SPECIAL["<|im_start|>"])

    def test_the_body_is_tokenised_with_the_text_around_it(self):
        """``user\\n`` and the prompt are one encode: a BPE tokenizer merges across the join."""
        calls = []

        class Counting(self.Tokenizer):
            def __call__(self, text, return_tensors=None, add_special_tokens=True):
                calls.append(text)
                return super().__call__(text, return_tensors, add_special_tokens)

        encode(Counting(), "HI", use_chat_template=True, family=QWEN2)
        self.assertEqual(calls, ["user\nHI", "\n", "assistant\n"])

    def test_the_tokenizers_own_template_wins_where_it_exists(self):
        class Modern(self.Tokenizer):
            chat_template = "{{ 'x' }}"

            def apply_chat_template(self, messages, **kwargs):
                return {"input_ids": torch.tensor([[7, 7, 7]], dtype=torch.int64)}

        self.assertEqual(encode(Modern(), "HI", use_chat_template=True, family=QWEN2).tolist(), [7, 7, 7])

    def test_the_route_is_named_rather_than_inferred(self):
        plain = self.Tokenizer()
        self.assertIn("Qwen2.5 turn markers", prompt_route(plain, use_chat_template=True, family=QWEN2))
        self.assertIn("raw", prompt_route(plain, use_chat_template=False, family=QWEN2))

    def test_the_stop_set_carries_both_ends_of_a_turn(self):
        """``config.json`` names one of the two; a run that took it at its word would over-run."""
        config = dict(QWEN2_5_3B_INSTRUCT_CONFIG)
        self.assertEqual(config["eos_token_id"], 151645)
        tokenizer = self.Tokenizer()
        ids = eos_ids_for(config, tokenizer)
        self.assertEqual(ids, {151645, tokenizer.SPECIAL["<|im_end|>"], tokenizer.SPECIAL["<|endoftext|>"]})


class TestFusedProjections(unittest.TestCase):
    """q/k/v in one tensor and gate/up in another, and the checkpoint names that write into them.

    The fusion is worth two GEMMs a layer and, because it puts q's heads next to
    k's, one RoPE call instead of two. What it must not be worth is a row in the
    wrong place: a q/k/v boundary off by a head loads without complaint and
    generates fluent nonsense, which is the failure this class exists to make
    impossible.
    """

    SHAPE = ModelShape(
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_size=64,
        hidden_size=256,
        intermediate_size=192,
        vocab_size=320,
        rms_norm_eps=1e-6,
        rope_theta=1.0e6,
        tie_word_embeddings=False,
        attn_output_gate=False,
        qk_norm=False,
        qkv_bias=True,
    )

    def _model(self, shape: ModelShape | None = None) -> CausalLM:
        return CausalLM(shape or self.SHAPE, max_seq_len=128, dtype=torch.float32, device=CPU)

    def test_the_fused_tensor_is_exactly_as_tall_as_its_three_parts(self):
        attention = self._model().layers[0].self_attn
        shape = self.SHAPE
        query_rows = shape.num_heads * shape.head_size
        kv_rows = shape.num_kv_heads * shape.head_size
        self.assertEqual(attention.qkv_split, (query_rows, kv_rows, kv_rows))
        self.assertEqual(attention.qkv_proj.weight.shape, (query_rows + 2 * kv_rows, shape.hidden_size))
        self.assertEqual(attention.qkv_proj.bias.shape, (query_rows + 2 * kv_rows,))

    def test_a_checkpoints_separate_names_write_into_the_fused_rows(self):
        """``q_proj.weight`` is a view of the fused tensor, so loading one writes the other."""
        model = self._model()
        destinations = model.load_destinations()
        attention = model.layers[0].self_attn
        query_rows, kv_rows, _ = attention.qkv_split

        marks = {"q_proj": 1.0, "k_proj": 2.0, "v_proj": 3.0}
        with torch.no_grad():
            for name, value in marks.items():
                destinations[f"layers.0.self_attn.{name}.weight"].fill_(value)
        fused = attention.qkv_proj.weight
        self.assertTrue(torch.all(fused[:query_rows] == 1.0))
        self.assertTrue(torch.all(fused[query_rows : query_rows + kv_rows] == 2.0))
        self.assertTrue(torch.all(fused[query_rows + kv_rows :] == 3.0))

    def test_gate_comes_before_up_which_is_the_order_the_checkpoints_fuse_them_in(self):
        model = self._model()
        destinations = model.load_destinations()
        with torch.no_grad():
            destinations["layers.0.mlp.gate_proj.weight"].fill_(1.0)
            destinations["layers.0.mlp.up_proj.weight"].fill_(2.0)
        fused = model.layers[0].mlp.gate_up_proj.weight
        half = self.SHAPE.intermediate_size
        self.assertTrue(torch.all(fused[:half] == 1.0))
        self.assertTrue(torch.all(fused[half:] == 2.0))

    def test_every_parameter_has_a_name_a_checkpoint_could_fill(self):
        """The loader refuses an unfilled parameter, so the table must reach all of them.

        Counted rather than listed: a fused tensor is one parameter under two or
        three names, so the table is *longer* than the parameter list and the
        check that matters is that no parameter's storage is left out of it.
        """
        model = self._model()
        destinations = model.load_destinations()
        reachable = {tensor.data_ptr() for tensor in destinations.values()}
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIn(parameter.data_ptr(), reachable)
        self.assertNotIn("layers.0.self_attn.qkv_proj.weight", destinations)
        self.assertIn("layers.0.self_attn.q_proj.weight", destinations)

    def test_the_fused_projection_computes_what_three_would_have(self):
        model = self._model()
        attention = model.layers[0].self_attn
        torch.manual_seed(0)
        with torch.no_grad():
            attention.qkv_proj.weight.normal_()
            attention.qkv_proj.bias.normal_()
        hidden = torch.randn(3, self.SHAPE.hidden_size)
        weights = attention.qkv_proj.weight.split(attention.qkv_split, dim=0)
        biases = attention.qkv_proj.bias.split(attention.qkv_split, dim=0)
        separate = [torch.nn.functional.linear(hidden, w, b) for w, b in zip(weights, biases)]
        fused = attention.qkv_proj(hidden).split(attention.qkv_split, dim=-1)
        for name, mine, theirs in zip(("q", "k", "v"), fused, separate):
            with self.subTest(projection=name):
                torch.testing.assert_close(mine, theirs, rtol=0, atol=0)

    def test_q_and_k_are_adjacent_so_one_rotation_serves_both(self):
        """The combined RoPE is only sound because the fusion put them side by side."""
        attention = self._model().layers[0].self_attn
        shape = self.SHAPE
        self.assertTrue(attention._rotates_together)
        self.assertEqual(attention._rotary_columns, (shape.num_heads + shape.num_kv_heads) * shape.head_size)

    def test_a_gate_between_q_and_k_is_rotated_separately(self):
        """``[q | gate | k | v]``: rotating across the gate would rotate the gate."""
        gated = ModelShape(**{**self.SHAPE.__dict__, "attn_output_gate": True})
        self.assertFalse(self._model(gated).layers[0].self_attn._rotates_together)

    def test_normed_heads_are_rotated_separately(self):
        """q_norm/k_norm copy each out of the buffer, so there is no shared view left."""
        normed = ModelShape(**{**self.SHAPE.__dict__, "qk_norm": True})
        self.assertFalse(self._model(normed).layers[0].self_attn._rotates_together)


class TestRotaryInPlace(unittest.TestCase):
    """``apply_`` writes the rotation back over the projection buffer it read."""

    def _rope(self, **kwargs) -> RotaryEmbedding:
        return RotaryEmbedding(64, 128, 1.0e6, torch.float32, CPU, **kwargs)

    def test_it_is_apply_written_back(self):
        for interleaved in (False, True):
            for rotary_dim in (None, 32):
                with self.subTest(interleaved=interleaved, rotary_dim=rotary_dim):
                    rope = self._rope(interleaved=interleaved, rotary_dim=rotary_dim)
                    x = torch.randn(3, 4, 64)
                    positions = torch.arange(3)
                    expected = rope.apply(x, positions)
                    written = rope.apply_(x.clone(), positions)
                    torch.testing.assert_close(written, expected, rtol=1e-6, atol=1e-6)

    def test_it_returns_the_tensor_it_was_given(self):
        rope = self._rope()
        x = torch.randn(2, 4, 64)
        self.assertIs(rope.apply_(x, torch.arange(2)), x)

    def test_the_pass_through_channels_are_left_alone(self):
        """A partial rotary rewrites the leading channels and nothing after them."""
        rope = self._rope(rotary_dim=32)
        x = torch.randn(2, 4, 64)
        tail = x[..., 32:].clone()
        rope.apply_(x, torch.arange(2))
        self.assertTrue(torch.equal(x[..., 32:], tail))

    def test_rotating_q_and_k_together_is_rotating_each(self):
        """What the decode does once instead of twice, held to the two calls it replaces."""
        rope = self._rope()
        heads, kv_heads, head_size = 4, 2, 64
        fused = torch.randn(1, (heads + kv_heads) * head_size)
        positions = torch.arange(1)
        query = fused[:, : heads * head_size].view(1, heads, head_size).clone()
        key = fused[:, heads * head_size :].view(1, kv_heads, head_size).clone()
        expected = (rope.apply(query, positions), rope.apply(key, positions))

        rope.apply_(fused.view(1, heads + kv_heads, head_size), positions)
        torch.testing.assert_close(fused[:, : heads * head_size].view(1, heads, head_size), expected[0])
        torch.testing.assert_close(fused[:, heads * head_size :].view(1, kv_heads, head_size), expected[1])


class TestFusedRMSNorm(unittest.TestCase):
    """The library kernel, against the expression it replaced."""

    def test_the_two_agree_exactly_on_the_row_a_decode_norms(self):
        """One token is the decode's whole batch, and there the two are bit for bit."""
        if not layers_module._HAS_FUSED_RMS_NORM:
            self.skipTest("this torch has no torch.nn.functional.rms_norm")
        torch.manual_seed(0)
        norm = RMSNorm(256, 1e-6, torch.float32, CPU)
        with torch.no_grad():
            norm.weight.normal_(mean=1.0, std=0.1)
        x = torch.randn(1, 256)
        fused = norm(x)
        layers_module._HAS_FUSED_RMS_NORM = False
        try:
            explicit = norm(x)
        finally:
            layers_module._HAS_FUSED_RMS_NORM = True
        self.assertTrue(torch.equal(fused, explicit))

    def test_the_accumulation_is_still_float32_under_a_float16_weight(self):
        """The property the explicit expression existed to state, held to on the kernel."""
        if not layers_module._HAS_FUSED_RMS_NORM:
            self.skipTest("this torch has no torch.nn.functional.rms_norm")
        norm = RMSNorm(4096, 1e-6, torch.float16, CPU)
        # Values whose squares overflow float16 (max 65504) but not float32.
        x = torch.full((1, 4096), 300.0, dtype=torch.float16)
        out = norm(x)
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out, torch.ones_like(out), rtol=1e-3, atol=1e-3)


class TestDecodeWorkspace(unittest.TestCase):
    """The buffers a decode step writes into, and the three integers that aim them."""

    SHAPE = TestFusedProjections.SHAPE

    def test_every_buffer_is_the_shape_one_token_needs(self):
        shape = self.SHAPE
        workspace = DecodeWorkspace.build(shape, torch.float32, CPU)
        query_rows = shape.num_heads * shape.head_size
        kv_rows = shape.num_kv_heads * shape.head_size
        self.assertEqual(workspace.hidden.shape, (1, shape.hidden_size))
        self.assertEqual(workspace.qkv.shape, (1, query_rows + 2 * kv_rows))
        self.assertEqual(workspace.attention.shape, (1, shape.num_heads, shape.head_size))
        self.assertEqual(workspace.gate_up.shape, (1, 2 * shape.intermediate_size))
        self.assertEqual(workspace.logits.shape, (1, shape.vocab_size))
        self.assertEqual(workspace.slots.dtype, torch.int32)
        self.assertEqual(workspace.context_lens.dtype, torch.int32)
        self.assertEqual(workspace.token_ids.dtype, torch.int64)

    def test_a_gated_layer_gets_room_for_the_gate(self):
        gated = ModelShape(**{**self.SHAPE.__dict__, "attn_output_gate": True})
        workspace = DecodeWorkspace.build(gated, torch.float32, CPU)
        rows = gated.num_heads * gated.head_size
        self.assertEqual(workspace.qkv.shape, (1, 2 * rows + 2 * gated.num_kv_heads * gated.head_size))

    def test_the_buffers_are_budgeted_like_the_pools_are(self):
        """Small beside a KV cache, and on the record rather than left to the allocator."""
        workspace = DecodeWorkspace.build(self.SHAPE, torch.float32, CPU)
        counted = sum(
            t.numel() * t.element_size()
            for t in (workspace.hidden, workspace.qkv, workspace.attention, workspace.gate_up, workspace.logits)
        )
        self.assertEqual(workspace.bytes_allocated(), counted)

    def test_seating_a_step_writes_the_buffers_rather_than_replacing_them(self):
        """A captured graph reads these addresses, so nothing here may be rebound."""
        workspace = DecodeWorkspace.build(self.SHAPE, torch.float32, CPU)
        pointers = [t.data_ptr() for t in (workspace.token_ids, workspace.positions, workspace.slots)]
        workspace.seat(token=17, position=9, context_len=10)
        self.assertEqual(int(workspace.token_ids), 17)
        self.assertEqual(int(workspace.positions), 9)
        self.assertEqual(int(workspace.slots), 9)
        self.assertEqual(int(workspace.context_lens), 10)
        self.assertEqual([t.data_ptr() for t in (workspace.token_ids, workspace.positions, workspace.slots)], pointers)


class TestDecodeWindow(unittest.TestCase):
    """The fixed slot count that replaced reading the context length back to the host."""

    def test_the_ladder_covers_the_cache_and_doubles(self):
        self.assertEqual(graph_windows(800, 128), (128, 256, 512, 800))
        self.assertEqual(graph_windows(4096, 128), (128, 256, 512, 1024, 2048, 4096))
        # A cache no longer than one block is one window.
        self.assertEqual(graph_windows(128, 128), (128,))

    def test_every_window_is_at_least_as_long_as_the_context_it_serves(self):
        windows = graph_windows(4096, 128)
        for context in (1, 127, 128, 129, 1000, 4096):
            with self.subTest(context=context):
                chosen = next(w for w in windows if w >= context)
                self.assertGreaterEqual(chosen, context)
                self.assertLess(chosen, max(2 * context, windows[0] + 1))

    def test_an_over_wide_window_changes_no_number(self):
        """Slots past a row's own length are masked off, so rounding up only costs reads."""
        geometry = CacheGeometry(num_layers=1, num_kv_heads=2, head_size=64, block_size=BLOCK_SIZE, max_seq_len=512)
        shape = LayerShape(num_heads=4, num_kv_heads=2, head_size=64, scale=0.125)
        backend = build_backend("cann_dense", geometry, shape, CPU, torch.float32)
        torch.manual_seed(0)
        context = 40
        key = torch.randn(context, 2, 64)
        value = torch.randn(context, 2, 64)
        backend.write_kv(0, key, value, backend.cache.slot_mapping(0, context))
        query = torch.randn(1, 4, 64)
        lengths = torch.full((1,), context, dtype=torch.int32)

        exact = torch.empty(1, 4, 64)
        backend.decode(0, query, lengths, exact, window=context)
        for window in (context + 1, 128, 512):
            with self.subTest(window=window):
                wide = torch.empty(1, 4, 64)
                backend.decode(0, query, lengths, wide, window=window)
                torch.testing.assert_close(wide, exact, rtol=0, atol=0)
        # And the same as letting the backend ask the host for the length.
        asked = torch.empty(1, 4, 64)
        backend.decode(0, query, lengths, asked, window=None)
        torch.testing.assert_close(asked, exact, rtol=0, atol=0)

    def test_the_backends_say_whether_they_can_run_window_free(self):
        """``native_v5`` takes its kv lengths as a Python list, so it never can."""
        from tq_longbench.ops import CANNDenseBackend, NativeV5Backend, TurboQuantCubeBackend

        self.assertFalse(NativeV5Backend.supports_static_decode)
        self.assertTrue(CANNDenseBackend.supports_static_decode)
        self.assertTrue(TurboQuantCubeBackend.supports_static_decode)
        self.assertTrue(DenseReferenceBackend.supports_static_decode)
        self.assertTrue(TurboQuantReferenceBackend.supports_static_decode)

    def test_running_window_free_and_being_capturable_are_different_questions(self):
        """The torch reference of the quantised cache decodes statically and still cannot be captured.

        Its ``write_kv`` drops padding slots with a boolean mask, so the number
        of rows it writes depends on the slots' values -- a host read, and one
        the kernels it stands in for do not make. Conflating the two flags would
        have had this backend attempt a capture that fails mid-record.
        """
        self.assertTrue(TurboQuantReferenceBackend.supports_static_decode)
        self.assertFalse(TurboQuantReferenceBackend.supports_graph_capture)
        self.assertTrue(DenseReferenceBackend.supports_graph_capture)


class TestStaticDecodePath(_Qwen2TinyCheckpoint):
    """The workspace path on a real (if tiny) checkpoint, against the path it replaced."""

    def test_a_run_is_what_it_was_before_the_buffers(self):
        """Allocation-free is a property of how the work is submitted, not of the answer."""
        runner = self._runner()
        static, _ = runner.generate(self.token_ids, max_new_tokens=self.NEW_TOKENS)

        # The same model, stepped the way the loop did before the workspace:
        # a fresh batch and fresh tensors every step.
        eager: list[int] = []
        ids = self.token_ids.clone()
        fresh = self._runner()
        logits = fresh.prefill(ids, RunMetrics())
        position = ids.numel()
        for _ in range(self.NEW_TOKENS):
            token = int(torch.argmax(logits[-1]))
            eager.append(token)
            step = torch.tensor([token], dtype=torch.int64)
            batch = fresh._batch(position, 1, is_decode=True)
            logits = fresh.model(step, batch, fresh.write_backends, fresh.decode_backend)
            position += 1
        self.assertEqual(static, eager)

    def test_the_loop_reports_which_path_it_took(self):
        """Eager and captured differ by an order of magnitude and by nothing in the text."""
        runner = self._runner()
        _, metrics = runner.generate(self.token_ids, max_new_tokens=2)
        summary = metrics.summary()
        self.assertFalse(summary["decode_graph"])
        self.assertEqual(summary["captures"], 0)
        self.assertGreater(summary["decode_tokens_per_second"], 0)
        self.assertIsNotNone(runner.decode_graph_refusal)

    def test_a_cpu_run_that_demands_a_graph_is_refused_rather_than_slowed(self):
        """``on`` is the benchmark's mode: a quiet fallback would read as slow kernels."""
        with self.assertRaisesRegex(GraphCaptureUnavailable, "cpu"):
            self._runner(decode_graph="on")
        # And 'auto' takes the fallback without complaint.
        self.assertIsNone(self._runner(decode_graph="auto").decode_graph)

    def test_an_unknown_mode_is_refused_at_the_config(self):
        with self.assertRaisesRegex(ValueError, "decode_graph"):
            RunnerConfig(model_path=self.model_path, decode_graph="maybe")


@unittest.skipUnless(torch.cuda.is_available(), "no CUDA device")
class TestCapturedDecodeGraph(_Qwen2TinyCheckpoint):
    """Capture and replay, against the eager path that recorded it.

    The tiny checkpoint rather than a real one, because what is under test is the
    machinery -- that a replay reads the buffers as they are now and not as they
    were at capture, that a window boundary is crossed cleanly, and that the
    tokens are the eager path's. What a 3B model costs is a benchmark, not a test.
    """

    def _cuda_runner(self, **overrides) -> StandaloneModelRunner:
        return self._runner(device="cuda:0", dtype=torch.float16, **overrides)

    def test_a_replayed_decode_produces_the_eager_tokens(self):
        eager, _ = self._cuda_runner(decode_graph="off").generate(self.token_ids.cuda(), max_new_tokens=self.NEW_TOKENS)
        runner = self._cuda_runner(decode_graph="on")
        graphed, metrics = runner.generate(self.token_ids.cuda(), max_new_tokens=self.NEW_TOKENS)
        self.assertEqual(graphed, eager)
        self.assertTrue(metrics.summary()["decode_graph"])
        self.assertGreaterEqual(metrics.summary()["captures"], 1)

    def test_a_replay_reads_the_buffers_as_they_are_now(self):
        """The whole risk of capture: a value frozen at record time would pass silently.

        Two steps at different positions from the same graph. If the replay had
        baked in the capture's position, the second would attend over the wrong
        prefix and write its K/V to the wrong slot -- and would still return a
        plausible token, which is why this is checked against the eager path
        rather than against nothing.
        """
        runner = self._cuda_runner(decode_graph="on")
        tokens, _ = runner.generate(self.token_ids.cuda(), max_new_tokens=6)
        reference, _ = self._cuda_runner(decode_graph="off").generate(self.token_ids.cuda(), max_new_tokens=6)
        self.assertEqual(tokens, reference)
        # One window served every step here, so the graph really was replayed.
        self.assertEqual(len(runner.decode_graph.captured), 1)

    def test_crossing_a_window_boundary_captures_a_second_graph(self):
        """A run long enough to outgrow its window records another and keeps going."""
        block, max_seq_len = 32, 256
        first, second = graph_windows(max_seq_len, block)[:2]
        # A prompt that sits just inside the first window, and a budget that
        # carries it past -- derived from the ladder so neither can drift.
        prompt = torch.randint(0, TINY_QWEN2["vocab_size"], (first - 2,), dtype=torch.int64).cuda()
        budget = second - first
        settings = {"max_seq_len": max_seq_len, "block_size": block}
        runner = self._cuda_runner(decode_graph="on", **settings)
        graphed, metrics = runner.generate(prompt, max_new_tokens=budget)
        reference, _ = self._cuda_runner(decode_graph="off", **settings).generate(prompt, max_new_tokens=budget)
        self.assertEqual(graphed, reference)
        self.assertEqual(runner.decode_graph.captured, [first, second])
        # The capture steps are kept out of the steady-state percentiles.
        self.assertEqual(len(metrics.capture_us), 2)

    def test_the_captures_are_reported_rather_than_averaged_in(self):
        runner = self._cuda_runner(decode_graph="on")
        _, metrics = runner.generate(self.token_ids.cuda(), max_new_tokens=5)
        summary = metrics.summary()
        self.assertEqual(summary["captures"], len(metrics.capture_us))
        self.assertGreater(summary["capture_ms"], 0)
        self.assertNotIn(metrics.capture_us[0], metrics.decode_step_us)

    def test_a_backend_that_reads_its_lengths_back_cannot_be_captured(self):
        """The refusal is by capability, not by device: native_v5 is the one that cannot."""
        graph = mock.patch.object(ops_module.CANNDenseBackend, "supports_static_decode", False)
        with graph, self.assertRaisesRegex(GraphCaptureUnavailable, "context lengths back to the host"):
            self._cuda_runner(decode_graph="on", backend="cann_dense")

    def test_replaying_two_windows_in_turn_keeps_both_correct(self):
        """The shared pool's one real risk: a later graph landing on an earlier one's memory.

        Within a run the window only grows, so alternation needs two runs on the
        same runner -- the second starts at position zero again and replays the
        small window after the large one has been captured over the same pool.
        If anything a replay needs had been allocated in that pool, the second
        pass through the small window would read what the large one left.
        """
        block, max_seq_len = 32, 256
        first, second = graph_windows(max_seq_len, block)[:2]
        settings = {"max_seq_len": max_seq_len, "block_size": block}
        short = torch.randint(0, TINY_QWEN2["vocab_size"], (8,), dtype=torch.int64).cuda()
        long = torch.randint(0, TINY_QWEN2["vocab_size"], (first + 4,), dtype=torch.int64).cuda()

        eager = self._cuda_runner(decode_graph="off", **settings)
        expected_short, _ = eager.generate(short, max_new_tokens=4)
        expected_long, _ = eager.generate(long, max_new_tokens=4)

        runner = self._cuda_runner(decode_graph="on", **settings)
        self.assertEqual(runner.generate(short, max_new_tokens=4)[0], expected_short)
        self.assertEqual(runner.generate(long, max_new_tokens=4)[0], expected_long)
        # ...and back to the first window, after the second was captured over it.
        self.assertEqual(runner.generate(short, max_new_tokens=4)[0], expected_short)
        self.assertEqual(runner.decode_graph.captured, [first, second])

    def test_on_refuses_a_failing_capture_rather_than_falling_back(self):
        """The mode a benchmark uses: a quiet eager fallback would report the wrong number."""
        runner = self._cuda_runner(decode_graph="on")
        broken = mock.patch.object(
            runner.decode_graph, "capture", side_effect=RuntimeError("this device will not record that")
        )
        with broken, self.assertRaisesRegex(RuntimeError, "will not record that"):
            runner.generate(self.token_ids.cuda(), max_new_tokens=4)
        self.assertIsNotNone(runner.decode_graph)

    def test_a_backend_with_a_value_dependent_shape_is_refused_too(self):
        graph = mock.patch.object(ops_module.CANNDenseBackend, "supports_graph_capture", False)
        with graph, self.assertRaisesRegex(GraphCaptureUnavailable, "depends on a tensor's values"):
            self._cuda_runner(decode_graph="on", backend="cann_dense")
        # And 'auto' takes the eager path instead of failing the run.
        with mock.patch.object(ops_module.CANNDenseBackend, "supports_graph_capture", False):
            runner = self._cuda_runner(decode_graph="auto", backend="cann_dense")
        self.assertIsNone(runner.decode_graph)
        self.assertIn("depends on a tensor's values", runner.decode_graph_refusal)

    def test_a_capture_that_fails_finishes_the_run_eagerly_and_says_so(self):
        """No device here refuses a capture, so one is made to -- the fallback is the point.

        An NPU that has the graph API but will not record this step is the case
        this exists for, and it cannot be produced from a machine with no NPU.
        What is checked is the behaviour: the token the capture was for is still
        generated, the rest of the run continues, and the reason is on the record
        rather than showing up as slow kernels.
        """
        runner = self._cuda_runner(decode_graph="auto")
        reference, _ = self._cuda_runner(decode_graph="off").generate(self.token_ids.cuda(), max_new_tokens=4)
        broken = mock.patch.object(
            runner.decode_graph, "capture", side_effect=RuntimeError("this device will not record that")
        )
        with broken:
            tokens, metrics = runner.generate(self.token_ids.cuda(), max_new_tokens=4)
        self.assertEqual(tokens, reference)
        self.assertIsNone(runner.decode_graph)
        self.assertFalse(metrics.summary()["decode_graph"])
        self.assertIn("will not record that", runner.decode_graph_refusal)


class _FakeTorchNpu(types.SimpleNamespace):
    """The slice of ``torch_npu`` this harness reaches for, with torch underneath.

    Mocked rather than skipped, because every one of these call sites is a branch
    only an NPU would take and this host has none: the fallbacks are what runs
    here, so without a stand-in the NPU code would be typed and never executed.
    What the stand-in cannot check is what the real operators compute; what it
    does check is the shape of the call -- the arguments, the unpacking, and that
    the fallback runs when the operator is absent.

    The operators are set as *attributes* rather than defined as methods, so a
    test can replace one, delete one to play an older torch_npu, or compare the
    bound object the resolver returned against the one it should have found.
    Each computes what torch would, so the NPU route can be held to the route it
    replaces.
    """

    def __init__(self, soc_version: int = 251, **overrides):
        super().__init__(**overrides)
        self.npu = types.SimpleNamespace(get_soc_version=lambda: soc_version)
        self.calls: dict[str, int] = {}
        self._formats: dict[int, int] = {}
        self.npu_rms_norm = self._npu_rms_norm
        self.npu_swiglu = self._npu_swiglu
        self.npu_format_cast = self._npu_format_cast
        self.get_npu_format = self._get_npu_format

    def _count(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def _npu_rms_norm(self, x, gamma, epsilon=1e-6):
        """CANN returns ``(normed, rstd)``; a caller that forgets the unpack gets a tuple."""
        self._count("npu_rms_norm")
        promoted = x.to(torch.float32)
        rstd = torch.rsqrt(promoted.pow(2).mean(-1, keepdim=True) + epsilon)
        return (promoted * rstd * gamma.to(torch.float32)).to(x.dtype), rstd.to(x.dtype)

    def _npu_swiglu(self, x):
        """Halves the trailing axis and returns ``silu(first) * second``."""
        self._count("npu_swiglu")
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

    def _npu_format_cast(self, tensor, acl_format, **kwargs):
        self._count("npu_format_cast")
        cast = tensor.clone()
        self._formats[cast.data_ptr()] = acl_format
        return cast

    def _get_npu_format(self, tensor):
        return self._formats.get(tensor.data_ptr(), 2)


@contextlib.contextmanager
def _fake_torch_npu(**kwargs):
    """Put a stand-in ``torch_npu`` where :func:`npu_operator` looks for one."""
    fake = _FakeTorchNpu(**kwargs)
    with mock.patch.dict(sys.modules, {"torch_npu": fake}):
        yield fake


class _NpuDevice:
    """A device object that claims to be an NPU without one being present.

    Only ever handed to the resolvers, which read ``.type`` and nothing else; the
    tensors in these tests stay on the CPU, because what is under test is which
    branch is taken and not where it runs.
    """

    type = "npu"


class TestNpuOperatorResolution(unittest.TestCase):
    """Which operator a module binds at construction, and when it binds none."""

    def test_nothing_is_bound_off_an_npu(self):
        with _fake_torch_npu():
            self.assertIsNone(npu_operator("npu_rms_norm", CPU))
            self.assertIsNone(npu_operator("npu_swiglu", torch.device("cuda:0")))

    def test_the_operator_is_bound_on_one(self):
        with _fake_torch_npu() as fake:
            self.assertIs(npu_operator("npu_rms_norm", _NpuDevice()), fake.npu_rms_norm)

    def test_an_operator_this_torch_npu_lacks_binds_nothing(self):
        """An older torch_npu is a fallback, not a crash halfway through a run."""
        with _fake_torch_npu():
            self.assertIsNone(npu_operator("npu_something_invented", _NpuDevice()))

    def test_torch_npu_is_never_imported_to_answer(self):
        """A CUDA or CPU run must not drag it in, and an NPU run has it already."""
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("torch_npu", None)
            self.assertIsNone(npu_operator("npu_rms_norm", _NpuDevice()))


class TestNpuRMSNorm(unittest.TestCase):
    """CANN's norm in place of the expression, and the unpack it needs."""

    def _norm(self, hidden: int = 256) -> RMSNorm:
        torch.manual_seed(0)
        norm = RMSNorm(hidden, 1e-6, torch.float32, CPU)
        with torch.no_grad():
            norm.weight.normal_(mean=1.0, std=0.1)
        return norm

    def test_the_operator_is_used_and_its_second_return_dropped(self):
        norm = self._norm()
        x = torch.randn(1, 256)
        expected = norm(x)
        with _fake_torch_npu() as fake:
            norm._npu_rms_norm = fake.npu_rms_norm
            out = norm(x)
        self.assertEqual(fake.calls["npu_rms_norm"], 1)
        # A forgotten [0] would return a tuple and fail here, loudly, which is
        # the point of asserting the type before the values.
        self.assertIsInstance(out, torch.Tensor)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_epsilon_reaches_the_operator_as_a_keyword(self):
        """``npu_rms_norm(x, gamma, epsilon=...)`` -- passing it positionally is a different op."""
        captured = {}

        def recording(x, gamma, epsilon=None):
            captured["epsilon"] = epsilon
            return x, x

        norm = self._norm()
        norm.eps = 1e-5
        norm._npu_rms_norm = recording
        norm(torch.randn(1, 256))
        self.assertEqual(captured["epsilon"], 1e-5)

    def test_without_the_operator_the_torch_route_still_runs(self):
        norm = self._norm()
        self.assertIsNone(norm._npu_rms_norm)
        self.assertEqual(norm(torch.randn(1, 256)).shape, (1, 256))


class TestNpuSwiGLU(unittest.TestCase):
    """CANN's SwiGLU in place of the split, the silu and the multiply."""

    SHAPE = TestFusedProjections.SHAPE

    def _mlp(self) -> MLP:
        torch.manual_seed(0)
        mlp = MLP(self.SHAPE, torch.float32, CPU)
        with torch.no_grad():
            mlp.gate_up_proj.weight.normal_(std=0.05)
            mlp.down_proj.weight.normal_(std=0.05)
        return mlp

    def test_it_computes_what_the_split_and_multiply_did(self):
        """Gate first, up second: the op halves the trailing axis the way this layout does."""
        mlp = self._mlp()
        x = torch.randn(1, self.SHAPE.hidden_size)
        expected = mlp(x)
        with _fake_torch_npu() as fake:
            mlp._npu_swiglu = fake.npu_swiglu
            out = mlp(x)
        self.assertEqual(fake.calls["npu_swiglu"], 1)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_it_is_handed_the_whole_fused_output_not_a_half(self):
        """The op does the splitting; handing it one half would silently halve the FFN."""
        captured = {}
        mlp = self._mlp()

        def recording(fused):
            captured["width"] = fused.shape[-1]
            return fused[..., : self.SHAPE.intermediate_size]

        mlp._npu_swiglu = recording
        mlp(torch.randn(1, self.SHAPE.hidden_size))
        self.assertEqual(captured["width"], 2 * self.SHAPE.intermediate_size)

    @torch.inference_mode()
    def test_the_workspace_path_takes_it_too(self):
        """The decode is the path that matters, and it reaches the op through the buffer.

        Under ``inference_mode`` because that is the workspace path's contract:
        it writes through ``out=``, which autograd refuses to track, and both
        callers (``generate`` and a graph capture) are already inside it.
        """
        mlp = self._mlp()
        workspace = DecodeWorkspace.build(self.SHAPE, torch.float32, CPU)
        x = torch.randn(1, self.SHAPE.hidden_size)
        expected = mlp(x, workspace)
        with _fake_torch_npu() as fake:
            mlp._npu_swiglu = fake.npu_swiglu
            out = mlp(x, workspace)
        self.assertEqual(fake.calls["npu_swiglu"], 1)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


class TestWeightNzPolicy(unittest.TestCase):
    """When a fused projection goes into FRACTAL_NZ, which is vllm_ascend's question.

    ``auto`` follows ``_should_trans_nz``: never for float32, always on a 310P,
    and -- for float16 and bfloat16 anywhere else -- only at ``weight_nz_mode 2``,
    which the shipped default is not. So on a 910 or a 950 the plugin leaves dense
    half-precision weights in ND, and copying that policy is deliberate: without
    an NPU to measure on, following the vendor beats guessing that more native
    must mean faster.
    """

    def test_off_is_off_everywhere(self):
        with _fake_torch_npu():
            for device in (CPU, _NpuDevice()):
                cast, reason = should_cast_to_nz("off", torch.float16, device)
                self.assertFalse(cast)
                self.assertIn("off", reason)

    def test_a_device_with_no_fractal_layout_never_casts(self):
        cast, reason = should_cast_to_nz("on", torch.float16, CPU)
        self.assertFalse(cast)
        self.assertIn("fractal", reason)

    def test_float32_never_casts_even_when_asked(self):
        with _fake_torch_npu():
            cast, reason = should_cast_to_nz("on", torch.float32, _NpuDevice())
            self.assertFalse(cast)
            self.assertIn("float32", reason)

    def test_a_310p_casts_on_auto(self):
        for soc in (200, 203, 205):
            with self.subTest(soc=soc), _fake_torch_npu(soc_version=soc):
                cast, reason = should_cast_to_nz("auto", torch.float16, _NpuDevice())
                self.assertTrue(cast)
                self.assertIn("310P", reason)

    def test_a_950_does_not_cast_half_precision_on_auto(self):
        """The plugin's default is quantised weights only, and this mirrors it."""
        with _fake_torch_npu(soc_version=260):
            cast, reason = should_cast_to_nz("auto", torch.bfloat16, _NpuDevice())
            self.assertFalse(cast)
            self.assertIn("weight_nz_mode 2", reason)
            # ...and 'on' is how to ask for it anyway.
            self.assertTrue(should_cast_to_nz("on", torch.bfloat16, _NpuDevice())[0])

    def test_a_torch_npu_that_cannot_name_the_soc_does_not_guess(self):
        with _fake_torch_npu() as fake:
            fake.npu = types.SimpleNamespace()
            cast, reason = should_cast_to_nz("auto", torch.float16, _NpuDevice())
            self.assertFalse(cast)
            self.assertIn("SoC version", reason)

    def test_an_unknown_mode_is_refused_at_the_config(self):
        with self.assertRaisesRegex(ValueError, "weight_nz"):
            RunnerConfig(model_path="/m", weight_nz="fractal")


class TestWeightNzCast(_Qwen2TinyCheckpoint):
    """The cast itself: what it touches, and the check that it did not change the answer.

    The policy is :class:`TestWeightNzPolicy`'s; here it is forced, because
    whether a device is an NPU and whether the mechanics are right are separate
    questions and only the second can be asked on a host with no NPU. The tensors
    stay on the CPU and the stand-in's ``npu_format_cast`` copies rather than
    retiling -- which is the honest limit of this: it exercises every line, and
    it cannot tell anyone what a real Cube does with the result.
    """

    LAYERS = TINY_QWEN2["num_hidden_layers"]

    @contextlib.contextmanager
    def _forced(self, reason: str = "test"):
        with mock.patch.object(engine_module, "should_cast_to_nz", return_value=(True, reason)):
            yield

    def test_nothing_is_cast_off_an_npu_and_the_reason_is_recorded(self):
        runner = self._runner()
        self.assertEqual(runner.layout_report.cast, 0)
        self.assertIn("fractal", runner.layout_report.describe())

    def test_it_casts_both_fused_projections_of_every_layer(self):
        """qkv and gate_up, and not o_proj -- which carries a fold done on ND values."""
        runner = self._runner()
        o_proj = [layer.self_attn.o_proj.weight.data_ptr() for layer in runner.model.layers]
        with _fake_torch_npu() as fake, self._forced():
            report = runner._cast_fused_projections()
        self.assertEqual(report.cast, 2 * self.LAYERS)
        self.assertEqual(fake.calls["npu_format_cast"], 2 * self.LAYERS)
        self.assertEqual(report.confirmed_format, 29)
        self.assertGreater(report.cosine, 0.9999)
        self.assertIn("FRACTAL_NZ", report.describe())
        # o_proj and down_proj were left where they were.
        self.assertEqual([layer.self_attn.o_proj.weight.data_ptr() for layer in runner.model.layers], o_proj)

    def test_the_model_still_generates_what_it_did(self):
        """A cast that reorders bytes must not move a token."""
        runner = self._runner()
        before, _ = runner.generate(self.token_ids, max_new_tokens=self.NEW_TOKENS)
        with _fake_torch_npu(), self._forced():
            runner._cast_fused_projections()
        after, _ = runner.generate(self.token_ids, max_new_tokens=self.NEW_TOKENS)
        self.assertEqual(after, before)

    def test_a_cast_that_changes_the_product_fails_the_load(self):
        """A layout the matmul reads wrongly returns numbers; only the probe notices."""
        runner = self._runner()
        with _fake_torch_npu() as fake, self._forced():
            fake.npu_format_cast = lambda tensor, acl_format, **kwargs: tensor.flip(-1).clone()
            with self.assertRaisesRegex(RuntimeError, "changed what the first fused projection computes"):
                runner._cast_fused_projections()

    def test_a_torch_npu_without_the_operator_falls_back_unless_on_asked(self):
        runner = self._runner()
        with _fake_torch_npu() as fake, self._forced():
            del fake.npu_format_cast
            runner.config.weight_nz = "auto"
            self.assertEqual(runner._cast_fused_projections().cast, 0)
            runner.config.weight_nz = "on"
            with self.assertRaisesRegex(RuntimeError, "no npu_format_cast"):
                runner._cast_fused_projections()

    def test_a_torch_npu_that_cannot_name_a_format_still_casts(self):
        """``get_npu_format`` is a confirmation, not a prerequisite."""
        runner = self._runner()
        with _fake_torch_npu() as fake, self._forced():
            del fake.get_npu_format
            report = runner._cast_fused_projections()
        self.assertEqual(report.cast, 2 * self.LAYERS)
        self.assertIsNone(report.confirmed_format)


class TestBlockTableCache(unittest.TestCase):
    """The block table a paged decode hands its kernel, built once per shape.

    It was a fresh allocation and a copy per layer per step, for a tensor whose
    contents depend on nothing but the window. Under capture that is a pointer
    into the graph's pool; cached, it is an address the replay already holds.
    """

    def _backend(self):
        geometry = CacheGeometry(num_layers=1, num_kv_heads=2, head_size=64, block_size=BLOCK_SIZE, max_seq_len=512)
        shape = LayerShape(num_heads=4, num_kv_heads=2, head_size=64, scale=0.125)
        return build_backend("cann_dense", geometry, shape, CPU, torch.float32)

    def test_the_same_window_returns_the_same_tensor(self):
        backend = self._backend()
        first = backend._block_tables(1, 256)
        self.assertIs(backend._block_tables(1, 256), first)

    def test_a_different_window_gets_its_own(self):
        backend = self._backend()
        narrow, wide = backend._block_tables(1, 128), backend._block_tables(1, 512)
        self.assertIsNot(narrow, wide)
        self.assertLess(narrow.shape[1], wide.shape[1])

    def test_it_is_the_table_the_uncached_expression_built(self):
        backend = self._backend()
        for tokens, window in ((1, 128), (1, 512), (4, 256)):
            with self.subTest(tokens=tokens, window=window):
                expected = backend.cache.block_table(window).expand(tokens, -1).contiguous()
                torch.testing.assert_close(backend._block_tables(tokens, window), expected, rtol=0, atol=0)


class TestKvDumpPaths(unittest.TestCase):
    """``--dump-kv-dir``: one spelling, and a directory that exists afterwards."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()

    def test_either_slash_and_a_trailing_one_name_the_same_directory(self):
        native = str(self.root)
        self.assertEqual(normalise_dump_dir(native.replace("\\", "/")), self.root)
        self.assertEqual(normalise_dump_dir(native + os.sep), self.root)

    def test_the_path_is_absolute_so_the_run_reports_where_it_wrote(self):
        self.assertTrue(normalise_dump_dir(".").is_absolute())

    def test_the_home_directory_is_expanded(self):
        self.assertEqual(normalise_dump_dir("~"), Path.home().resolve())

    def test_a_missing_destination_is_created_parents_and_all(self):
        directory = run_directory(self.root / "nowhere" / "deeper", "a_run")
        self.assertTrue(directory.is_dir())
        self.assertEqual(directory.name, "a_run")
        # Asking twice is not an error; a second depth must not blow up on it.
        self.assertEqual(run_directory(self.root / "nowhere" / "deeper", "a_run"), directory)

    def test_the_run_name_says_what_the_run_was(self):
        name = default_run_name("F:/AI/models/Qwen2.5-3B-Instruct", "turboquant_cube", 4096)
        self.assertTrue(name.startswith("qwen2.5-3b-instruct_turboquant_cube_4096_"), name)
        # Nothing in it needs quoting on a filesystem.
        self.assertRegex(name, r"^[0-9a-z._-]+$")

    def test_two_runs_do_not_land_on_one_directory(self):
        with mock.patch.object(kv_dump.time, "strftime", side_effect=["20260920-120000", "20260920-120001"]):
            first = default_run_name("/m/model", "turboquant_aiv", 8192)
            second = default_run_name("/m/model", "turboquant_aiv", 8192)
        self.assertNotEqual(first, second)


class TestKvDumpRefusals(unittest.TestCase):
    """The two ways a dump would come up empty, refused before the weights load."""

    def _args(self, **overrides):
        settings = {"dump_kv_dir": "F:/AI/kvcache", "backend": "turboquant_cube", "tokens": 4096, **overrides}
        return types.SimpleNamespace(**settings)

    def test_a_dense_backend_has_no_quantised_cache_to_write(self):
        with self.assertRaisesRegex(SystemExit, "has none"):
            require_dumpable(self._args(backend="cann_dense"))

    def test_the_needle_stage_is_what_builds_the_cache(self):
        with self.assertRaisesRegex(SystemExit, "--tokens above 0"):
            require_dumpable(self._args(tokens=0))

    def test_a_run_that_asked_for_no_dump_is_never_refused(self):
        self.assertIsNone(require_dumpable(self._args(dump_kv_dir=None, backend="cann_dense", tokens=0)))

    def test_a_quantised_backend_at_a_real_context_is_allowed(self):
        for backend in ("turboquant_cube", "turboquant_aiv", "turboquant_reference"):
            with self.subTest(backend=backend):
                self.assertIsNone(require_dumpable(self._args(backend=backend)))


class TestCachedExtent(unittest.TestCase):
    """How much of the pool a generation actually filled -- counted, not inferred.

    The loop appends a token and only then forwards it, so the last one it
    produced was chosen and never written. Counting produced tokens would put one
    row of zeros at the end of the context, which in a compression study reads as
    a suspiciously compressible final position rather than as a bug.
    """

    def test_it_counts_steps_rather_than_tokens(self):
        # Ended on a stop token: four produced, three forwarded.
        metrics = RunMetrics(decode_step_us=[1.0, 2.0, 3.0], stopped_on_token=7)
        self.assertEqual(cached_extent(100, metrics), 103)

    def test_a_capture_step_wrote_a_slot_too(self):
        """A captured step is still a step; it is only timed in a different list."""
        metrics = RunMetrics(decode_step_us=[1.0, 2.0], capture_us=[50.0])
        self.assertEqual(cached_extent(100, metrics), 103)

    def test_a_prefill_with_no_decode_is_the_prompt(self):
        self.assertEqual(cached_extent(4096, RunMetrics()), 4096)


class TestKvDumpContents(unittest.TestCase):
    """What the file holds, and that it reads back with nothing of this package loaded."""

    KV_HEADS = 2
    HEADS = 4
    HEAD_SIZE = 128
    LAYERS = 2
    TOKENS = 40

    def setUp(self):
        torch.manual_seed(0)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.geometry = CacheGeometry(
            num_layers=self.LAYERS,
            num_kv_heads=self.KV_HEADS,
            head_size=self.HEAD_SIZE,
            block_size=BLOCK_SIZE,
            max_seq_len=4 * BLOCK_SIZE,
        )
        shape = LayerShape(self.HEADS, self.KV_HEADS, self.HEAD_SIZE, self.HEAD_SIZE**-0.5)
        self.backend = build_backend("turboquant_reference", self.geometry, shape, CPU, torch.float16)
        self.key = torch.randn(self.TOKENS, self.KV_HEADS, self.HEAD_SIZE)
        self.value = torch.randn(self.TOKENS, self.KV_HEADS, self.HEAD_SIZE)
        for layer in range(self.LAYERS):
            self.backend.write_kv(layer, self.key, self.value, self.backend.cache.slot_mapping(0, self.TOKENS))
        self.runner = self._runner()

    def _runner(self):
        """A stand-in with the attributes the dump reads, and no checkpoint behind it."""
        shape = ModelShape(
            num_layers=self.LAYERS,
            num_heads=self.HEADS,
            num_kv_heads=self.KV_HEADS,
            head_size=self.HEAD_SIZE,
            hidden_size=512,
            intermediate_size=1024,
            vocab_size=320,
            rms_norm_eps=1e-6,
            rope_theta=1.0e6,
            tie_word_embeddings=False,
            attn_output_gate=False,
        )
        config = types.SimpleNamespace(
            backend="turboquant_reference", prefill_mode="dense_staging", dtype=torch.float16
        )
        return types.SimpleNamespace(decode_backend=self.backend, geometry=self.geometry, shape=shape, config=config)

    def _dump(self, **overrides):
        settings = {
            "model_path": "F:/AI/models/Qwen2.5-3B-Instruct",
            "family": "Qwen2.5",
            "cached_tokens": self.TOKENS,
            "context_tokens": self.TOKENS - 8,
            "prompt": "where is the passcode?",
            "needle": {"depth": 0.5, "city": "Valencia", "answer": "894772", "retrieved": True, "clean": True},
            **overrides,
        }
        return dump_quantised_cache(self.runner, self.root / "run", **settings)

    def _blob(self):
        return torch.load(self._dump().directory / "kv_cache_tensors.pt", map_location="cpu", weights_only=True)

    def test_all_three_files_are_written(self):
        report = self._dump()
        for name in ("kv_cache_tensors.pt", "metadata.json", "README.md"):
            with self.subTest(file=name):
                self.assertTrue((report.directory / name).is_file())
        self.assertGreater(report.bytes_written, 0)
        self.assertIn("kv dump", report.describe())

    def test_it_loads_on_a_cpu_with_weights_only(self):
        """The consumer has torch and nothing else -- no custom class may be in the pickle."""
        blob = self._blob()
        self.assertIsInstance(blob, dict)
        for key in ("key_planes", "value_planes", "scale_planes", "centroids", "pi", "context_lens"):
            with self.subTest(key=key):
                self.assertIsInstance(blob[key], torch.Tensor)
                self.assertEqual(blob[key].device.type, "cpu")

    def test_the_planes_are_layer_major_and_trimmed_to_the_context(self):
        blob = self._blob()
        blocks = -(-self.TOKENS // BLOCK_SIZE)
        packed = (self.LAYERS, blocks, BLOCK_SIZE, self.KV_HEADS, self.HEAD_SIZE // 2)
        self.assertEqual(tuple(blob["key_planes"].shape), packed)
        self.assertEqual(tuple(blob["value_planes"].shape), packed)
        self.assertEqual(
            tuple(blob["scale_planes"].shape), (self.LAYERS, blocks, BLOCK_SIZE, self.geometry.scale_slot)
        )
        # The pool is four blocks; only the one the context reached was written out.
        self.assertLess(blocks, self.geometry.num_blocks)

    def test_the_dumped_tensors_are_copies_the_run_cannot_move(self):
        """Cloned and detached: a later decode must not rewrite what was handed over."""
        blob = self._blob()
        before = blob["key_planes"][0].clone()
        self.backend.write_kv(
            0, torch.randn_like(self.key), self.value, self.backend.cache.slot_mapping(0, self.TOKENS)
        )
        self.assertTrue(torch.equal(blob["key_planes"][0], before))

    def test_the_codec_and_the_rotation_travel_with_it(self):
        """Without these the codes are unreadable, and nothing about the shapes says so."""
        blob = self._blob()
        self.assertEqual(tuple(blob["centroids"].shape), (16,))
        self.assertEqual(tuple(blob["thresholds"].shape), (15,))
        self.assertEqual(tuple(blob["pi"].shape), (self.HEAD_SIZE, self.HEAD_SIZE))
        # Pi is its own inverse, which is what makes one matmul enough.
        identity = torch.eye(self.HEAD_SIZE, dtype=blob["pi"].dtype)
        torch.testing.assert_close(blob["pi"] @ blob["pi"], identity, rtol=1e-5, atol=1e-5)

    def test_the_readme_recipe_actually_reconstructs_k_and_v(self):
        """The guide is executed here, on this cache, against the tensors that were written.

        The reason this exists rather than a proofread: every way of getting the
        reconstruction wrong -- skipping Pi, reading the high nibble first,
        taking the V scales for the K ones -- produces a tensor of exactly the
        right shape and dtype full of plausible numbers. A consumer following a
        stale README would get -0.005 and no error, so the README's own steps are
        held to the cache they describe.
        """
        blob = self._blob()
        centroids, pi = blob["centroids"], blob["pi"]
        tokens = int(blob["context_lens"][0])
        kv_heads, head_dim = self.KV_HEADS, self.HEAD_SIZE

        def rebuild(planes, scales, layer):
            packed = planes[layer].reshape(-1, kv_heads, head_dim // 2)[:tokens]
            byte = packed.to(torch.int64) + 128
            codes = torch.stack((byte % 16, byte // 16), dim=-1).flatten(-2)
            return (centroids[codes] * scales.unsqueeze(-1)) @ pi

        flat = blob["scale_planes"].reshape(self.LAYERS, -1, blob["scale_planes"].shape[-1])
        key_hat = rebuild(blob["key_planes"], flat[0, :tokens, :kv_heads], 0)
        value_hat = rebuild(blob["value_planes"], flat[0, :tokens, kv_heads : 2 * kv_heads], 0)

        # 4 bits, so ~0.995 -- close, and not the 1.0 an unquantised copy would give.
        self.assertGreater(_cosine_of(key_hat, self.key), 0.99)
        self.assertGreater(_cosine_of(value_hat, self.value), 0.99)
        self.assertLess(_cosine_of(key_hat, self.key), 0.9999)

    def test_skipping_the_rotation_is_silent_and_wrong(self):
        """The number the README quotes, measured rather than asserted in prose."""
        blob = self._blob()
        tokens = int(blob["context_lens"][0])
        packed = blob["key_planes"][0].reshape(-1, self.KV_HEADS, self.HEAD_SIZE // 2)[:tokens]
        byte = packed.to(torch.int64) + 128
        codes = torch.stack((byte % 16, byte // 16), dim=-1).flatten(-2)
        scales = blob["scale_planes"].reshape(self.LAYERS, -1, blob["scale_planes"].shape[-1])
        rotated = blob["centroids"][codes] * scales[0, :tokens, : self.KV_HEADS].unsqueeze(-1)
        self.assertLess(abs(_cosine_of(rotated, self.key)), 0.05)

    def test_the_metadata_carries_the_model_the_run_and_the_needle(self):
        report = self._dump()
        metadata = json.loads((report.directory / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["model_name"], "Qwen2.5-3B-Instruct")
        self.assertEqual(metadata["model_family"], "Qwen2.5")
        self.assertEqual(metadata["num_layers"], self.LAYERS)
        self.assertEqual(metadata["num_kv_heads"], self.KV_HEADS)
        self.assertEqual(metadata["head_dim"], self.HEAD_SIZE)
        self.assertEqual(metadata["num_attention_heads"], self.HEADS)
        self.assertEqual(metadata["backend"], "turboquant_reference")
        self.assertEqual(metadata["block_size"], BLOCK_SIZE)
        self.assertEqual(metadata["allocated_blocks"], self.geometry.num_blocks)
        self.assertTrue(metadata["niah"]["retrieved"])
        self.assertEqual(metadata["niah"]["depth"], 0.5)
        self.assertIn("passcode", metadata["prompt"])

    def test_the_prompt_length_and_the_cache_length_are_both_recorded(self):
        """They differ by what the run generated, and reading one for the other misleads."""
        metadata = json.loads((self._dump().directory / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["cached_tokens"], self.TOKENS)
        self.assertEqual(metadata["context_tokens"], self.TOKENS - 8)

    def test_the_readme_describes_this_dump_rather_than_some_other(self):
        readme = (self._dump().directory / "README.md").read_text(encoding="utf-8")
        self.assertIn("Qwen2.5-3B-Instruct", readme)
        self.assertIn("weights_only=True", readme)
        for key in ("key_planes", "value_planes", "scale_planes", "centroids", "pi"):
            with self.subTest(key=key):
                self.assertIn(f"`{key}`", readme)
        # The shapes in the table are this cache's.
        blocks = -(-self.TOKENS // BLOCK_SIZE)
        self.assertIn(str((self.LAYERS, blocks, BLOCK_SIZE, self.KV_HEADS, self.HEAD_SIZE // 2)), readme)

    def test_a_dense_cache_is_refused_by_the_collector_too(self):
        """Belt and braces on the CLI's own check, for a caller that skipped it."""
        shape = LayerShape(self.HEADS, self.KV_HEADS, self.HEAD_SIZE, self.HEAD_SIZE**-0.5)
        self.runner.decode_backend = build_backend("dense_reference", self.geometry, shape, CPU, torch.float16)
        with self.assertRaisesRegex(TypeError, "no quantised cache"):
            self._dump()

    def test_every_tensor_has_a_line_in_the_readme_table(self):
        """A tensor added to the dump without a description would ship unexplained."""
        self.assertEqual(sorted(collect_tensors(self.runner, self.TOKENS)), sorted(kv_dump._PURPOSE))


def _cosine_of(actual: torch.Tensor, expected: torch.Tensor) -> float:
    left, right = actual.flatten().double(), expected.flatten().double()
    return float(torch.dot(left, right) / (left.norm() * right.norm()))


class TestModelPathNormalisation(unittest.TestCase):
    """``--model-path``: one spelling for the whole run, and a sentence when it is wrong."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_either_slash_and_a_trailing_one_are_the_same_directory(self):
        """What a Windows shell hands over, in the three spellings it hands it over in."""
        native = str(self.root)
        forward = native.replace("\\", "/")
        self.assertEqual(model_directory(forward), native)
        self.assertEqual(model_directory(native), native)
        self.assertEqual(model_directory(native + os.sep), native)

    def test_the_home_directory_is_expanded(self):
        self.assertEqual(model_directory("~"), str(Path.home()))

    def test_nothing_is_opened_while_the_parser_is_still_being_built(self):
        """An argparse type that stats would make every parser need a checkpoint on disk."""
        missing = str(self.root / "nowhere")
        self.assertEqual(model_directory(missing), missing)

    def test_a_path_that_is_not_a_checkpoint_is_one_sentence_not_a_traceback(self):
        from tq_longbench.smoke_glm import require_checkpoint

        with self.assertRaisesRegex(SystemExit, "is not a directory"):
            require_checkpoint(str(self.root / "nowhere"))
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(SystemExit, "holds no config.json"):
            require_checkpoint(str(empty))


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
        self.assertIn(999, GLM4.stop_ids(tokenizer))
        self.assertNotIn(151336, GLM4.stop_ids(tokenizer))

    def test_an_unknown_special_token_falls_back_rather_than_becoming_unk(self):
        """``<unk>`` is not a stop token; treating it as one truncates at the first odd word."""

        class Ignorant:
            added_tokens_encoder: dict = {}
            unk_token_id = 0

            def convert_tokens_to_ids(self, token):
                return 0

        self.assertEqual(special_token_id(Ignorant(), "<|user|>", 151336), 151336)

    def test_a_non_glm4_checkpoint_keeps_its_own_stop_tokens(self):
        """GLM-4's ids are GLM-4's: nothing here leaks into another model's vocabulary.

        The tokenizer handed over is GLM-4's, so it answers for ``<|endoftext|>``
        -- and at *its* id, 151329, not the 151643 Qwen2.5 publishes, because the
        tokenizer is asked before the constant. What must not appear either way
        is ``<|user|>`` or ``<|observation|>``: those are GLM-4 turn markers, and
        nothing about a ``qwen2`` config asks for them.
        """
        qwen = {"model_type": "qwen2", "eos_token_id": 151645}
        tokenizer = _ChatGLM4LikeTokenizer()
        ids = eos_ids_for(qwen, tokenizer)
        self.assertEqual(ids, {151645, tokenizer.added_tokens_encoder["<|endoftext|>"]})
        self.assertFalse(ids & {151336, 151338})

    def test_an_unclaimed_model_type_borrows_no_ids_at_all(self):
        """GENERIC is the honest answer to a checkpoint nothing here has been told about."""
        llama = {"model_type": "llama", "eos_token_id": [2, 32000]}
        self.assertIs(family_for(llama), GENERIC)
        self.assertEqual(eos_ids_for(llama, _ChatGLM4LikeTokenizer()), {2, 32000})

    def test_how_a_run_ended_is_reported_rather_than_read_off_the_text(self):
        """A stopped continuation and a truncated one are the same sentence on the page."""
        from tq_longbench.smoke_glm import describe_stop

        class Naming:
            def convert_ids_to_tokens(self, token):
                return {151645: "<|im_end|>"}[token]

        stopped = RunMetrics(stopped_on_token=151645)
        self.assertEqual(describe_stop(stopped, Naming(), 32), "stopped on '<|im_end|>' (151645)")
        # No tokenizer, or one that cannot name the id: the id is still the answer.
        self.assertEqual(describe_stop(stopped, None, 32), "stopped on token 151645")
        self.assertEqual(describe_stop(RunMetrics(stopped_on_token=7), Naming(), 32), "stopped on token 7")

    def test_a_run_that_never_stopped_says_so_in_those_words(self):
        """The failure this exists for: a stop set that does not hold what the model emits."""
        from tq_longbench.smoke_glm import describe_stop

        self.assertIn("ran the whole 32-token budget", describe_stop(RunMetrics(), None, 32))

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
        ids = encode(tokenizer, "HI", use_chat_template=True, family=GLM4).tolist()
        self.assertEqual(ids[:3], [151331, 151333, 151336])
        self.assertEqual(ids[-1], 151337)
        self.assertEqual(ids[3:-1], [ord("\n"), ord("H"), ord("I"), ord("\n")])

    def test_the_tokenizers_own_prefix_does_not_appear_twice(self):
        """ChatGLM4Tokenizer prepends [gMASK]<sop> to everything; the body must not carry it."""
        ids = encode(_ChatGLM4LikeTokenizer(), "HI", use_chat_template=True, family=GLM4).tolist()
        self.assertEqual(ids.count(151331), 1)
        self.assertEqual(ids.count(151333), 1)

    def test_a_tokenizer_without_the_flag_has_its_prefix_measured(self):
        """``add_special_tokens`` is the direct way to decline the prefix; not every call takes it."""

        class NoFlag(_ChatGLM4LikeTokenizer):
            def __call__(self, text, return_tensors=None):
                return types.SimpleNamespace(
                    input_ids=torch.tensor([self.PREFIX + self._body(text)], dtype=torch.int64)
                )

        with_flag = encode(_ChatGLM4LikeTokenizer(), "HI", use_chat_template=True, family=GLM4)
        measured = encode(NoFlag(), "HI", use_chat_template=True, family=GLM4)
        self.assertEqual(measured.tolist(), with_flag.tolist())

    def test_the_tokenizers_own_template_wins_where_it_exists(self):
        """A transcription is a fallback, never a preference: the checkpoint's own format rules."""

        class Modern(_ChatGLM4LikeTokenizer):
            chat_template = "{{ 'x' }}"

            def apply_chat_template(self, messages, **kwargs):
                return {"input_ids": torch.tensor([[7, 7, 7]], dtype=torch.int64)}

        self.assertEqual(encode(Modern(), "HI", use_chat_template=True, family=GLM4).tolist(), [7, 7, 7])

    def test_raw_prompt_still_means_raw(self):
        ids = encode(_ChatGLM4LikeTokenizer(), "HI", use_chat_template=False, family=GLM4).tolist()
        self.assertNotIn(151336, ids)

    def test_the_route_is_named_rather_than_inferred(self):
        """Which of the three routes ran is printed, because it used to be invisible."""
        plain = _ChatGLM4LikeTokenizer()
        self.assertIn("assembled here", prompt_route(plain, use_chat_template=True, family=GLM4))
        self.assertIn("raw", prompt_route(plain, use_chat_template=False, family=GLM4))
        self.assertIn("no turn is transcribed", prompt_route(plain, use_chat_template=True, family=GENERIC))


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
    """The LongBench pipeline end to end, over JSONL written the way the suite writes it.

    No ``datasets`` and no download: the loader reads ``{task}.jsonl`` with
    ``json.loads``, which is the whole point of it, so a test can hand it four
    lines and exercise the real path rather than a mock of it.
    """

    TASKS = ("narrativeqa", "gov_report", "trec", "lcc")
    CONTEXT = 192
    ITEMS = 3

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
        self.dataset_dir = self.root / "longbench"
        self.dataset_dir.mkdir()
        self.answers = {
            "narrativeqa": ["alpha beta"],
            "gov_report": ["alpha beta gamma"],
            "trec": ["sports"],
            "lcc": ["return a + b"],
        }
        for task in self.TASKS:
            extra = {"all_classes": ["sports", "politics"]} if task == "trec" else {}
            rows = [
                {
                    "context": "alpha beta gamma delta " * 12,
                    "input": "what?",
                    "answers": self.answers[task],
                    "length": 60,
                    **extra,
                }
                for _ in range(self.ITEMS)
            ]
            path = self.dataset_dir / f"{task}.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def _run(self, *extra, tasks=None):
        from tq_longbench import run_longbench

        out_file = self.root / "longbench.jsonl"
        argv = [
            "--model-path", str(self.model_path),
            "--device", "cpu",
            "--backends", "dense_reference",
            "--tasks", ",".join(tasks or self.TASKS),
            "--dataset-dir", str(self.dataset_dir),
            "--max-context-len", str(self.CONTEXT),
            "--limit", str(self.ITEMS),
            "--max-new-tokens", "4",
            "--dtype", "float32",
            "--chunk-size", "48",
            "--out-file", str(out_file),
            *extra,
        ]  # fmt: skip
        with (
            mock.patch.object(run_longbench, "load_tokenizer", return_value=_ByteTokenizer()),
            contextlib.redirect_stdout(io.StringIO()) as out,
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            self.assertEqual(run_longbench.main(argv), 0)
        records = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
        return records, out.getvalue(), err.getvalue()

    def test_the_items_come_off_disk_through_the_tasks_own_template(self):
        records, _, logged = self._run()
        self.assertEqual(len(records), len(self.TASKS) * self.ITEMS)
        for task in self.TASKS:
            with self.subTest(task=task):
                self.assertIn(f"{task}.jsonl", logged)
                self.assertFalse(any(r["synthetic"] for r in records if r["task"] == task))

    def test_each_task_is_scored_by_its_own_official_metric(self):
        """The metric is a property of the task, so the record has to name it."""
        records, _, _ = self._run()
        seen = {record["task"]: record["metric"] for record in records}
        self.assertEqual(seen["narrativeqa"], "F1")
        self.assertEqual(seen["gov_report"], "ROUGE-L")
        self.assertEqual(seen["trec"], "Accuracy")
        self.assertEqual(seen["lcc"], "Edit-Sim")

    def test_a_missing_jsonl_falls_back_to_stubs_and_says_so_everywhere(self):
        """Loudly: a stub row is not a LongBench score and must not read as one."""
        records, printed, logged = self._run(tasks=("narrativeqa", "qasper"))
        self.assertIn("SYNTHETIC STUBS", logged)
        self.assertIn("measure the harness, not the model", logged)
        self.assertTrue(all(r["synthetic"] for r in records if r["task"] == "qasper"))
        self.assertFalse(any(r["synthetic"] for r in records if r["task"] == "narrativeqa"))
        self.assertIn("qasper *", printed)
        self.assertIn("synthetic stubs", printed)

    def test_the_markdown_table_has_a_row_per_task_and_an_average(self):
        _, printed, _ = self._run()
        self.assertIn("| Task                 | Metric", printed)
        for task in self.TASKS:
            with self.subTest(task=task):
                self.assertRegex(printed, rf"\|\s*{re.escape(task)}\s*\|")
        self.assertIn("Overall Average", printed)
        self.assertIn("### dense_reference @ 192 tokens", printed)
        # Every body row carries a number followed by a percent sign; the header
        # says "Score (%)" and must not be counted as one.
        rows = [line for line in printed.splitlines() if re.search(r"\|\s+\d+\.\d%", line)]
        self.assertEqual(len(rows), len(self.TASKS) + 1)

    def test_a_perfect_prediction_scores_a_hundred_percent(self):
        """The scoring wired end to end, with the model replaced by the right answer.

        A tiny random checkpoint scores zero on everything, which cannot tell a
        working scorer from one that returns zero. Handing the loop the reference
        answer proves the other end of the range, and with it the whole chain:
        loader, template, truncation, generation, decode, metric, aggregate.

        One task per metric family -- F1, ROUGE-L, classification, edit
        similarity -- because each has its own normalisation, and a family
        whose ideal answer scored 99% would be a scorer bug hiding at the top.
        """
        from tq_longbench import run_longbench

        # Items run task by task in --tasks order, so the reference answers can
        # be handed out in that same order.
        references = iter([self.answers[task][0] for task in self.TASKS for _ in range(self.ITEMS)])

        def perfect(tokenizer, token_ids):
            return next(references), None

        with mock.patch.object(run_longbench, "decode_text", perfect):
            records, printed, _ = self._run()
        self.assertEqual(len(records), len(self.TASKS) * self.ITEMS)
        for record in records:
            with self.subTest(task=record["task"], index=record["index"]):
                self.assertEqual(record["prediction"], self.answers[record["task"]][0])
                self.assertEqual(record["score"], 1.0)
        average = next(line for line in printed.splitlines() if "Overall Average" in line)
        self.assertIn("100.0%", average)

    def test_the_prompt_route_the_config_and_the_segmenter_are_all_reported(self):
        """Three things that silently change a score, so none of them is left to be guessed."""
        _, _, logged = self._run()
        self.assertIn("GLM-4 turn markers", logged)
        self.assertIn("stop ids: [", logged)
        self.assertIn("prompts:", logged)
        self.assertIn("chinese segmentation:", logged)

    def test_the_suites_own_config_wins_over_the_transcription(self):
        """A downloaded LongBench ships the prompts it was scored with; they beat ours."""
        from tq_longbench.tasks import config_source, longbench_config

        mine, _ = longbench_config(self.dataset_dir)
        config = self.dataset_dir / "config"
        config.mkdir()
        (config / "dataset2prompt.json").write_text(
            json.dumps({"narrativeqa": "OVERRIDDEN {context} {input}"}), encoding="utf-8"
        )
        theirs, _ = longbench_config(self.dataset_dir)
        self.assertNotEqual(mine["narrativeqa"], theirs["narrativeqa"])
        self.assertEqual(theirs["narrativeqa"], "OVERRIDDEN {context} {input}")
        # Tasks the override does not mention keep the transcription.
        self.assertEqual(mine["gov_report"], theirs["gov_report"])
        self.assertIn("config", config_source(self.dataset_dir))

    def test_profile_layers_is_off_by_default(self):
        _, _, quiet = self._run(tasks=("narrativeqa",))
        self.assertNotIn("layer probe", quiet)
        _, _, loud = self._run("--profile-layers", tasks=("narrativeqa",))
        self.assertIn("layer probe", loud)

    def test_an_unknown_task_is_refused_and_all_means_all(self):
        from tq_longbench.metrics import TASK_METRICS
        from tq_longbench.run_longbench import parse_tasks

        self.assertEqual(parse_tasks("longbench_qasper,gov_report"), ("qasper", "gov_report"))
        self.assertEqual(set(parse_tasks("all")), set(TASK_METRICS))
        with self.assertRaisesRegex(ValueError, "no LongBench metric"):
            parse_tasks("narrativeqa,not_a_task")


class TestContextTiers(unittest.TestCase):
    """``--context-tier`` and the flags that keep a 1M arena from being provisioned by accident.

    Everything here stops short of allocating one. A tier run's whole point is an
    arena three orders of magnitude past what this suite can hold, so what is
    checked is the arithmetic and the refusals that stand in front of it: which
    rung a tier implies, which tasks it reads, what an override does to the cache
    size, and that the guard, the fidelity cap and the dry run all fire before
    a single weight is loaded.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.model_path = self.root / "chatglm"
        self.model_path.mkdir()
        (self.model_path / "config.json").write_text(json.dumps(TINY_GLM), encoding="utf-8")
        self.tiered = self.root / "tiered"

    def _args(self, *extra):
        from tq_longbench.run_longbench import build_parser, parse_budget_overrides

        args = build_parser().parse_args(
            ["--model-path", str(self.model_path), "--tiered-dir", str(self.tiered), *extra]
        )
        args.budget_overrides = parse_budget_overrides(args.max_tokens_override)
        return args

    def _write_tier(self, tier, tasks=("qasper",), items=1, manifest=True):
        directory = self.tiered / tier
        directory.mkdir(parents=True)
        for task in tasks:
            rows = [{"context": "alpha beta", "input": "what?", "answers": ["alpha"]} for _ in range(items)]
            (directory / f"{task}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        if manifest:
            (directory / "manifest.json").write_text(
                json.dumps({"tier": tier, "tasks": {task: {"items": items} for task in tasks}}), encoding="utf-8"
            )
        return directory

    # ------------------------------------------------------------ the budgets

    def test_a_bare_number_overrides_every_task(self):
        from tq_longbench.run_longbench import parse_budget_overrides

        self.assertEqual(parse_budget_overrides("256"), {None: 256})

    def test_pairs_override_one_task_each(self):
        from tq_longbench.run_longbench import parse_budget_overrides

        self.assertEqual(parse_budget_overrides("gov_report=1024,qasper=256"), {"gov_report": 1024, "qasper": 256})

    def test_a_task_whose_real_name_starts_with_longbench_survives_the_prefix_strip(self):
        """``longbench_v2`` is a task, not ``v2`` with a prefix on it."""
        from tq_longbench.run_longbench import parse_budget_overrides, parse_tasks

        self.assertEqual(parse_tasks("longbench_v2"), ("longbench_v2",))
        self.assertEqual(parse_tasks("longbench_qasper"), ("qasper",))
        self.assertEqual(parse_budget_overrides("longbench_v2=32"), {"longbench_v2": 32})

    def test_a_malformed_override_is_refused(self):
        from tq_longbench.run_longbench import parse_budget_overrides

        for text in ("gov_report", "gov_report=x", "gov_report=0", "not_a_task=64"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_budget_overrides(text)

    def test_the_budget_falls_through_the_flags_in_order(self):
        from tq_longbench.run_longbench import item_budget
        from tq_longbench.tasks import EvalItem

        item = EvalItem(prompt="p", answers=["a"], metric="qasper", max_new_tokens=128, extra={"task": "qasper"})
        args = types.SimpleNamespace(budget_overrides={}, max_new_tokens=None)
        self.assertEqual(item_budget(args, item), 128)
        args.max_new_tokens = 16
        self.assertEqual(item_budget(args, item), 16)
        args.budget_overrides = {None: 32}
        self.assertEqual(item_budget(args, item), 32)
        args.budget_overrides = {None: 32, "qasper": 64}
        self.assertEqual(item_budget(args, item), 64)

    def test_an_override_widens_the_arena_it_needs(self):
        """An arena sized off the published budget would be short by the difference."""
        from tq_longbench.run_longbench import _largest_budget

        args = self._args("--max-tokens-override", "qasper=2048")
        self.assertEqual(_largest_budget(args, ("qasper", "gov_report")), 2048)
        # gov_report's published 512 still sets it where no override applies.
        self.assertEqual(_largest_budget(self._args(), ("qasper", "gov_report")), 512)

    # -------------------------------------------------------------- the tiers

    def test_a_tier_is_one_rung_at_its_ceiling(self):
        """Not a ladder: the items were chosen for being that long, so a shorter rung would cut them."""
        from tq_longbench.prepare_buckets import TIERS
        from tq_longbench.run_longbench import apply_tier

        self._write_tier("256k", tasks=("longbench_v2",))
        args = self._args("--context-tier", "256k")
        contexts, tasks = apply_tier(args)
        self.assertEqual(contexts, (TIERS["256k"].max_tokens,))
        self.assertEqual(tasks, ("longbench_v2",))
        self.assertEqual(args.dataset_dir, self.tiered / "256k")

    def test_the_tasks_come_from_the_manifest_and_fall_back_to_the_files(self):
        from tq_longbench.run_longbench import tier_tasks

        with_manifest = self._write_tier("32k", tasks=("qasper", "gov_report"))
        self.assertEqual(tier_tasks(with_manifest), ("gov_report", "qasper"))
        without = self._write_tier("256k", tasks=("longbench_v2",), manifest=False)
        self.assertEqual(tier_tasks(without), ("longbench_v2",))

    def test_an_explicit_tasks_flag_still_wins_at_a_tier(self):
        from tq_longbench.run_longbench import apply_tier

        self._write_tier("32k", tasks=("qasper", "gov_report"))
        _, tasks = apply_tier(self._args("--context-tier", "32k", "--tasks", "qasper"))
        self.assertEqual(tasks, ("qasper",))

    def test_a_tier_that_was_never_built_says_how_to_build_it(self):
        from tq_longbench.run_longbench import apply_tier

        with self.assertRaises(SystemExit) as refusal:
            apply_tier(self._args("--context-tier", "1m"))
        self.assertIn("prepare_buckets.py", str(refusal.exception))

    def test_an_empty_tier_directory_is_refused_rather_than_run_on_nothing(self):
        from tq_longbench.run_longbench import apply_tier

        (self.tiered / "32k").mkdir(parents=True)
        with self.assertRaises(SystemExit) as refusal:
            apply_tier(self._args("--context-tier", "32k"))
        self.assertIn("no task JSONL", str(refusal.exception))

    def test_without_a_tier_the_ladder_is_unchanged(self):
        from tq_longbench.run_longbench import DEFAULT_CONTEXTS, apply_tier

        contexts, tasks = apply_tier(self._args())
        self.assertEqual(contexts, tuple(int(part) for part in DEFAULT_CONTEXTS.split(",")))
        self.assertIn("narrativeqa", tasks)

    # ------------------------------------------------------------- the guards

    def test_the_arena_guard_names_the_flag_that_lifts_it(self):
        from tq_longbench.run_longbench import guard_arena

        args = self._args()
        guard_arena(args, 1024, 2048)  # well under the default, nothing happens
        with self.assertRaises(SystemExit) as refusal:
            guard_arena(args, 1048576, 1049088)
        message = str(refusal.exception)
        self.assertIn("--max-context 1049088", message)
        self.assertIn("--dry-run", message)

    def test_the_guard_lifts_when_it_is_lifted_deliberately(self):
        from tq_longbench.run_longbench import guard_arena

        guard_arena(self._args("--max-context", "2000000"), 1048576, 1049088)

    def test_the_1m_tier_does_not_run_without_being_asked_twice(self):
        from tq_longbench import run_longbench

        self._write_tier("1m", tasks=("longbench_v2",))
        argv = [
            "--model-path", str(self.model_path),
            "--tiered-dir", str(self.tiered),
            "--context-tier", "1m",
            "--device", "cpu",
        ]  # fmt: skip
        with self.assertRaises(SystemExit) as refusal:
            run_longbench.main(argv)
        self.assertIn("--max-context", str(refusal.exception))

    def test_fidelity_is_refused_above_the_length_its_second_prefill_fits(self):
        from tq_longbench import run_longbench
        from tq_longbench.run_longbench import FIDELITY_MAX_CONTEXT

        self._write_tier("256k", tasks=("longbench_v2",))
        argv = [
            "--model-path", str(self.model_path),
            "--tiered-dir", str(self.tiered),
            "--context-tier", "256k",
            "--eval-fidelity",
            "--device", "cpu",
        ]  # fmt: skip
        with self.assertRaises(SystemExit) as refusal:
            run_longbench.main(argv)
        message = str(refusal.exception)
        self.assertIn(str(FIDELITY_MAX_CONTEXT), message)
        self.assertIn("dataset metrics and the telemetry are measured at every tier", message)

    def test_the_dry_run_prices_the_arena_without_loading_a_weight(self):
        """config.json is the whole input: the point is to answer 'will this fit' before paying."""
        from tq_longbench import run_longbench

        self._write_tier("1m", tasks=("longbench_v2",))
        argv = [
            "--model-path", str(self.model_path),
            "--tiered-dir", str(self.tiered),
            "--context-tier", "1m",
            "--dry-run",
            "--device", "cpu",
        ]  # fmt: skip
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = run_longbench.main(argv)
        printed = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("REFUSED", printed)
        self.assertIn("Dense KV (MB)", printed)
        self.assertIn("longbench_v2", printed)

    def test_the_dry_run_passes_a_tier_that_clears_the_guard(self):
        from tq_longbench import run_longbench

        self._write_tier("32k", tasks=("qasper",))
        argv = [
            "--model-path", str(self.model_path),
            "--tiered-dir", str(self.tiered),
            "--context-tier", "32k",
            "--dry-run",
            "--device", "cpu",
        ]  # fmt: skip
        with contextlib.redirect_stdout(io.StringIO()) as out:
            run_longbench.main(argv)
        printed = out.getvalue()
        self.assertNotIn("REFUSED", printed)
        self.assertRegex(printed, r"\s32768\s")


class TestLongBenchJsonlLoader(unittest.TestCase):
    """``load_longbench_jsonl`` on its own: the file shape the suite ships, and its failure modes."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dataset_dir = Path(self._tmp.name)

    def _write(self, task, rows, blank_lines=False):
        lines = [json.dumps(row, ensure_ascii=False) for row in rows]
        separator = "\n\n" if blank_lines else "\n"
        (self.dataset_dir / f"{task}.jsonl").write_text(separator.join(lines) + "\n", encoding="utf-8")

    def test_a_row_becomes_an_item_through_the_tasks_template(self):
        from tq_longbench.tasks import load_longbench_jsonl, longbench_config

        row = {"context": "CTX-MARKER", "input": "INPUT-MARKER", "answers": ["x", "y"], "length": 7}
        self._write("hotpotqa", [row])
        (item,) = load_longbench_jsonl("hotpotqa", self.dataset_dir)
        prompts, budgets = longbench_config(self.dataset_dir)
        self.assertEqual(item.prompt, prompts["hotpotqa"].format(context="CTX-MARKER", input="INPUT-MARKER"))
        self.assertEqual(item.answers, ["x", "y"])
        self.assertEqual(item.metric, "hotpotqa")
        self.assertEqual(item.max_new_tokens, budgets["hotpotqa"])
        self.assertEqual(item.extra["length"], 7)

    def test_classification_classes_and_language_travel_on_the_item(self):
        """``trec`` cannot be scored without its class list, so the loader must not drop it."""
        from tq_longbench.metrics import score_prediction
        from tq_longbench.tasks import load_longbench_jsonl

        classes = ["Abbreviation", "Entity", "Location"]
        self._write("trec", [{"context": "c", "input": "i", "answers": ["Location"], "all_classes": classes}])
        (item,) = load_longbench_jsonl("trec", self.dataset_dir)
        self.assertEqual(item.extra["all_classes"], classes)
        self.assertEqual(score_prediction("trec", "Location", item.answers, item.extra["all_classes"]), 1.0)

    def test_chinese_rows_round_trip_unescaped(self):
        from tq_longbench.tasks import load_longbench_jsonl

        self._write("multifieldqa_zh", [{"context": "上下文", "input": "问题", "answers": ["答案"], "language": "zh"}])
        (item,) = load_longbench_jsonl("multifieldqa_zh", self.dataset_dir)
        self.assertIn("上下文", item.prompt)
        self.assertEqual(item.answers, ["答案"])
        self.assertEqual(item.extra["language"], "zh")

    def test_limit_counts_rows_and_blank_lines_are_skipped(self):
        from tq_longbench.tasks import load_longbench_jsonl

        rows = [{"context": str(n), "input": "q", "answers": [str(n)]} for n in range(5)]
        self._write("qasper", rows, blank_lines=True)
        loaded = [item.answers for item in load_longbench_jsonl("qasper", self.dataset_dir)]
        self.assertEqual(loaded, [[str(n)] for n in range(5)])
        limited = list(load_longbench_jsonl("qasper", self.dataset_dir, limit=2))
        self.assertEqual([item.extra["index"] for item in limited], [0, 1])

    def test_a_missing_file_and_an_unknown_task_are_refused_by_name(self):
        from tq_longbench.tasks import load_longbench_jsonl, local_task_path

        self.assertIsNone(local_task_path("narrativeqa", self.dataset_dir))
        with self.assertRaisesRegex(FileNotFoundError, "narrativeqa.jsonl"):
            next(load_longbench_jsonl("narrativeqa", self.dataset_dir))
        with self.assertRaisesRegex(ValueError, "not_a_task"):
            next(load_longbench_jsonl("not_a_task", self.dataset_dir))

    def test_the_hf_loader_refuses_a_task_it_cannot_score(self):
        """The prompt table has all 21 tasks, but ``score`` knows six: refuse before downloading."""
        from tq_longbench.tasks import load_longbench

        with self.assertRaisesRegex(ValueError, "no prompt template"):
            next(load_longbench("longbench_trec"))


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
        self.assertEqual(choose_prefill_mode(args, GLM4), "batched_decode")
        args = parser.parse_args(["--model-path", "/m", "--tokens", "131072", "--prefill-mode", "dense_staging"])
        with contextlib.redirect_stderr(io.StringIO()) as warning:
            self.assertEqual(choose_prefill_mode(args, GLM4), "dense_staging")
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
            return smoke_glm.plan_attention(args, glm4_model_shape(GLM4_9B_CHAT_1M_CONFIG), prefill_mode)

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
