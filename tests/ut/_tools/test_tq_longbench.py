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

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

# Appended, never prepended: tools/bisect/ is a package whose name shadows the
# standard library's bisect, and prepending tools/ breaks `import random`.
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "tools"))

from tq_longbench._ascend import assert_no_vllm_imported, turboquant_layout, turboquant_rotation  # noqa: E402
from tq_longbench.cpu_reference import cpu_turboquant_ops  # noqa: E402
from tq_longbench.engine import RunnerConfig, StandaloneModelRunner  # noqa: E402
from tq_longbench.kv_cache import CacheGeometry, DenseKVCache, TurboQuantKVCache, dense_equivalent_bytes  # noqa: E402
from tq_longbench.layers import ModelShape  # noqa: E402
from tq_longbench.ops import LayerShape, TurboQuantAivBackend, TurboQuantCubeBackend, build_backend  # noqa: E402
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
        # Outside turboquant_cube_meta_ops the operator is not registered at all.
        with self.assertRaisesRegex(RuntimeError, "not registered"):
            TurboQuantCubeBackend(geometry, shape, CPU)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
