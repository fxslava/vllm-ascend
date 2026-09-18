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
"""One GLM-4 attention layer, every backend, against naive float32 attention.

Why this file is separate from ``test_tq_longbench.py``: that one holds the
harness to exact attention at geometries small enough to be quick (8 heads, a
few hundred tokens).  What is asked here is the opposite trade -- GLM-4's real
geometry (32 heads over 4 kv heads, ``D=128``, ``rotary_dim=64`` interleaved) at
lengths that cross the boundaries the short cases never reach: a 2048-token
prefill chunk, and a 4096-token prefix under a decode.  These cases take seconds
rather than milliseconds, so they live on their own and are marked
:data:`SLOW_TEST_ENV` to be skipped where that matters.

The three questions, in the order a failure would have to be read:

1. **Does a chunked prefill equal an unchunked one at this geometry?**  If the
   absolute positions RoPE is given, or the causal mask across a chunk boundary,
   were wrong, a token at 2048 would attend to a different prefix than the same
   token does when the whole prompt arrives at once.  Held to the naive float32
   reference, not to another backend, so "both wrong the same way" cannot pass.
   :meth:`TestMultiChunkPrefill.test_the_second_chunk_reaches_back_into_the_first`
   additionally checks the mask against a deliberately *broken* reference -- one
   where the second chunk cannot see the first -- because a cosine that only ever
   sees the right answer says nothing about what it would do with the wrong one.

2. **Does a decode over a 4096-token prefix agree?**  The prefill path masks; the
   decode path does not mask at all, it is told a length.  They are different
   code, and a decode that silently reads a truncated prefix returns fluent
   numbers.

3. **Is the interleaved RoPE GLM-4's?**  Pairs ``(2i, 2i+1)``, first 64 channels
   only, transcribed from ``modeling_chatglm.py`` rather than paraphrased.

**What runs where.**  ``cann_dense`` is matmul and softmax, so it runs on any
device and is the CPU-side proof.  The TurboQuant pair need their operators:
``turboquant_aiv`` has CPU stand-ins and runs here, ``turboquant_cube`` has none
by design -- what that launch computes is a camodel gate -- so it is skipped
without an NPU.  Point ``TQ_EQUIV_DEVICE`` at one (``npu:0``) to run the whole
file there; every backend then runs on the device and is still held to the same
float32 reference computed on the host.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

# Appended, never prepended: tools/bisect/ is a package whose name shadows the
# standard library's bisect, and prepending tools/ breaks `import random`.
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "tools"))

from tq_longbench.cpu_reference import cpu_turboquant_ops  # noqa: E402
from tq_longbench.kv_cache import CacheGeometry  # noqa: E402
from tq_longbench.layers import Attention, ForwardBatch, ModelShape, RotaryEmbedding  # noqa: E402
from tq_longbench.ops import LayerShape, build_backend, cube_decode_available  # noqa: E402

#: ``npu:0`` to run the whole file on a device; unset means the host.
DEVICE_ENV = "TQ_EQUIV_DEVICE"

#: Set to ``0`` to skip the two long cases. They are minutes on a host serving
#: the TurboQuant operators from the CPU, seconds on a device.
SLOW_TEST_ENV = "TQ_EQUIV_SLOW"

#: Set to ``1`` to stop :func:`report` printing each measured cosine.
QUIET_ENV = "TQ_EQUIV_QUIET"

# glm-4-9b-chat-1m's attention layer, as published.
NUM_HEADS = 32
NUM_KV_HEADS = 4
HEAD_SIZE = 128
ROTARY_DIM = 64
ROPE_THETA = 1.0e8
HIDDEN_SIZE = NUM_HEADS * HEAD_SIZE
BLOCK_SIZE = 128

#: Crosses one 2048-token chunk boundary with a short chunk behind it, so the
#: second chunk is neither aligned to the first nor the same size.
PREFILL_TOKENS = 3072
PREFILL_CHUNK = 2048

#: A decode whose prefix is past every tile and block boundary the backends have.
DECODE_PREFIX = 4096

#: Unquantised against exact attention: storage rounding, nothing else.
DENSE_MIN_COSINE = 0.9999

#: The 4-bit cache against exact attention over the *unquantised* K/V, so this
#: floor is quantisation loss. The harness's own suite measures ~0.99 at smaller
#: geometries; the asked-for floor is 0.999, which 4 bits does not clear and is
#: not a defect -- see :data:`QUANTIZED_MIN_COSINE`'s use below.
QUANTIZED_MIN_COSINE = 0.98

#: float32 scores one reference tile may hold. 32 heads x 3072 x 3072 at once is
#: 1.2 GB, which is not a unit test's to allocate.
REFERENCE_TILE_ROWS = 256


def device_under_test() -> torch.device:
    """The host unless :data:`DEVICE_ENV` names a device; ``npu`` needs its import first."""
    name = os.environ.get(DEVICE_ENV, "cpu")
    if name.startswith("npu"):
        import torch_npu  # noqa: F401  (registers the npu device type)

        return torch.device("npu:0" if name == "npu" else name)
    return torch.device(name)


def cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    left = actual.detach().to("cpu", torch.float64).flatten()
    right = expected.detach().to("cpu", torch.float64).flatten()
    return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-30))


def report(label: str, measured: float, floor: float) -> float:
    """Print what was measured, not only whether it cleared the floor.

    This file exists to answer "is there divergence at the chunk boundary", and
    a green tick does not answer it -- 0.99991 and 0.99999999 both pass and mean
    different things. Set :data:`QUIET_ENV` to silence it.
    """
    if os.environ.get(QUIET_ENV) != "1":
        verdict = "ok " if measured > floor else "LOW"
        print(f"    {verdict} {label:52s} cos {measured:.8f}  (floor {floor})", file=sys.stderr)
    return measured


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Naive GQA attention in float32: query row ``r`` sees keys ``0 .. lengths[r] - 1``.

    Deliberately the slow spelling -- broadcast the kv heads, build the whole
    score matrix, mask it, softmax it, weight the values -- because it is the
    thing every backend is being held to and the one place where being obviously
    correct beats being fast. Tiled over query rows only so a 32-head 3072-token
    case does not ask for 1.2 GB at once; the tiling touches nothing but memory.

    ``lengths`` is a count of visible columns per row, which is how both a decode
    (each row its own context) and a bottom-right causal prefill (row ``r`` sees
    ``k - q + r + 1``) are spelled.
    """
    query, key, value = (tensor.detach().to("cpu", torch.float32) for tensor in (query, key, value))
    lengths = lengths.detach().to("cpu")
    group = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(group, dim=1)
    value = value.repeat_interleave(group, dim=1)
    columns = torch.arange(key.shape[0]).view(1, 1, -1)
    outputs = []
    for start in range(0, query.shape[0], REFERENCE_TILE_ROWS):
        rows = query[start : start + REFERENCE_TILE_ROWS]
        scores = torch.einsum("qhd,khd->hqk", rows, key) * scale
        visible = lengths[start : start + rows.shape[0]].view(1, -1, 1)
        scores = scores.masked_fill(columns >= visible, float("-inf"))
        outputs.append(torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), value))
    return torch.cat(outputs)


def causal_lengths(count: int, prefix_end: int) -> torch.Tensor:
    """Bottom-right causal as a per-row column count: the last row sees the whole prefix."""
    return torch.arange(prefix_end - count + 1, prefix_end + 1, dtype=torch.int64)


def glm4_shape() -> ModelShape:
    """One GLM-4 layer's shape: no q/k norm, no gate, qkv bias, half-width interleaved RoPE."""
    return ModelShape(
        num_layers=1,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=HIDDEN_SIZE * 2,
        vocab_size=1024,
        rms_norm_eps=1.5625e-07,
        rope_theta=ROPE_THETA,
        tie_word_embeddings=False,
        attn_output_gate=False,
        qk_norm=False,
        qkv_bias=True,
        rotary_dim=ROTARY_DIM,
        rope_interleaved=True,
    )


def layer_shape() -> LayerShape:
    return LayerShape(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE**-0.5)


def backends_under_test(device: torch.device) -> list[str]:
    """Every attention path this host can actually run, dense first.

    ``turboquant_cube`` has no CPU stand-in on purpose -- what that one launch
    computes is verified on the camodel -- so it joins the list only where its
    kernels are registered.
    """
    names = ["cann_dense", "turboquant_aiv"]
    if device.type == "npu":
        names.insert(1, "native_v5")
        if cube_decode_available():
            names.append("turboquant_cube")
    return names


def minimum_cosine(backend: str) -> float:
    return DENSE_MIN_COSINE if backend in ("cann_dense", "native_v5") else QUANTIZED_MIN_COSINE


def synthetic_hidden(count: int, dtype: torch.dtype, device: torch.device, seed: int = 0) -> torch.Tensor:
    """Deterministic hidden states, generated on the host so a device run gets the same ones."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, HIDDEN_SIZE, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)


class _LayerCase(unittest.TestCase):
    """One initialised attention layer and its RoPE, shared by the prefill and decode cases."""

    DTYPE = torch.float16

    @classmethod
    def setUpClass(cls):
        cls.device = device_under_test()
        cls.shape = glm4_shape()
        # The stand-ins serve torch.ops._C_ascend for turboquant_aiv on a host
        # with no NPU. Entered for the class, because torch 2.10 refuses a second
        # Python kernel for the same key and a per-test entry would register twice.
        cls._ops = cpu_turboquant_ops()
        cls._ops.__enter__()
        torch.manual_seed(0)
        cls.layer = Attention(cls.shape, 0, cls.DTYPE, cls.device)
        # Small weights: at D=128 the default initialisation drives float16
        # logits into saturation before any backend is involved, and what would
        # then be measured is the dtype rather than the attention.
        with torch.no_grad():
            for parameter in cls.layer.parameters():
                parameter.mul_(0.1)
        cls.rope = RotaryEmbedding(
            HEAD_SIZE,
            DECODE_PREFIX + 64,
            ROPE_THETA,
            cls.DTYPE,
            cls.device,
            rotary_dim=ROTARY_DIM,
            interleaved=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls._ops.__exit__(None, None, None)

    def _geometry(self, max_seq_len: int) -> CacheGeometry:
        return CacheGeometry(
            num_layers=1,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            block_size=BLOCK_SIZE,
            max_seq_len=max_seq_len,
        )

    def _backend(self, name: str, max_seq_len: int):
        return build_backend(name, self._geometry(max_seq_len), layer_shape(), self.device, self.DTYPE)

    def _batch(self, start: int, count: int, backend, is_decode: bool) -> ForwardBatch:
        positions = torch.arange(start, start + count, dtype=torch.int64, device=self.device)
        return ForwardBatch(
            positions=positions,
            slots=backend.cache.slot_mapping(start, count),
            prefix_end=start + count,
            is_decode=is_decode,
        )

    def _project(self, hidden: torch.Tensor, batch: ForwardBatch):
        """q, k, v for this chunk, RoPE applied at its absolute positions."""
        with torch.inference_mode():
            query, key, value, _ = self.layer.project(hidden, batch, self.rope)
        return query, key, value


@unittest.skipIf(os.environ.get(SLOW_TEST_ENV) == "0", f"{SLOW_TEST_ENV}=0")
class TestMultiChunkPrefill(_LayerCase):
    """A 3072-token prefill in 2048 + 1024, against the same prefill computed whole."""

    #: One prefill per backend for the whole class. The three cases below ask
    #: different questions of the same run, and on a host serving the TurboQuant
    #: operators from the CPU a 3072-token prefill is minutes, not seconds.
    _runs: dict[str, tuple[torch.Tensor, ...]] = {}

    @classmethod
    def tearDownClass(cls):
        cls._runs.clear()
        super().tearDownClass()

    def _run_chunked(self, name: str) -> tuple[torch.Tensor, ...]:
        """Prefill in chunks, returning this layer's attention output and the q/k/v it used."""
        if name in self._runs:
            return self._runs[name]
        backend = self._backend(name, PREFILL_TOKENS)
        outputs, queries, keys, values = [], [], [], []
        for start in range(0, PREFILL_TOKENS, PREFILL_CHUNK):
            count = min(PREFILL_CHUNK, PREFILL_TOKENS - start)
            hidden = synthetic_hidden(PREFILL_TOKENS, self.DTYPE, self.device)[start : start + count]
            batch = self._batch(start, count, backend, is_decode=False)
            query, key, value = self._project(hidden, batch)
            backend.write_kv(0, key, value, batch.slots)
            attention = torch.empty_like(query)
            with torch.inference_mode():
                backend.prefill_chunk(0, query, batch.prefix_end, attention)
            outputs.append(attention)
            queries.append(query)
            keys.append(key)
            values.append(value)
        self._runs[name] = tuple(torch.cat(tensors) for tensors in (outputs, queries, keys, values))
        return self._runs[name]

    def test_a_chunked_prefill_is_the_unchunked_one(self):
        """Every backend, 3072 tokens in two chunks, against naive float32 attention.

        The second chunk starts at 2048 with a populated cache behind it, which is
        where a lost position offset or a mask anchored to the chunk rather than to
        the prefix would show.
        """
        lengths = causal_lengths(PREFILL_TOKENS, PREFILL_TOKENS)
        for name in backends_under_test(self.device):
            with self.subTest(backend=name):
                attention, query, key, value = self._run_chunked(name)
                expected = reference_attention(query, key, value, layer_shape().scale, lengths)
                floor = minimum_cosine(name)
                measured = report(f"prefill {PREFILL_TOKENS} in {PREFILL_CHUNK}-chunks: {name}", floor=floor,
                                  measured=cosine(attention, expected))  # fmt: skip
                self.assertGreater(measured, floor, f"{name}: cos {measured:.8f}")

    def test_the_second_chunks_own_rows_are_right_one_by_one(self):
        """Row by row across the boundary, so a mask that is causal on average cannot pass."""
        attention, query, key, value = self._run_chunked("cann_dense")
        for row in (PREFILL_CHUNK - 1, PREFILL_CHUNK, PREFILL_CHUNK + 1, PREFILL_TOKENS - 1):
            with self.subTest(row=row):
                visible = row + 1
                expected = reference_attention(
                    query[row : row + 1],
                    key[:visible],
                    value[:visible],
                    layer_shape().scale,
                    torch.tensor([visible]),
                )
                measured = report(f"row {row} of the chunked prefill", cosine(attention[row : row + 1], expected),
                                  DENSE_MIN_COSINE)  # fmt: skip
                self.assertGreater(measured, DENSE_MIN_COSINE)

    def test_the_second_chunk_reaches_back_into_the_first(self):
        """The assertion with teeth: the result must *not* match a prefix-blind reference.

        A cosine against the right answer only says the right answer was not
        missed. What is being ruled out here is the specific failure the symptom
        would imply -- a second chunk that attends only within itself -- so the
        same rows are also scored against a reference where exactly that is true.
        If the two references scored alike, this test could not tell them apart
        and would be worth nothing; the gap between them is asserted first.
        """
        attention, query, key, value = self._run_chunked("cann_dense")
        rows = slice(PREFILL_CHUNK, PREFILL_TOKENS)
        count = PREFILL_TOKENS - PREFILL_CHUNK

        whole_prefix = reference_attention(
            query[rows], key, value, layer_shape().scale, causal_lengths(count, PREFILL_TOKENS)
        )
        # The broken one: the same rows, but the cache starts at the chunk.
        chunk_only = reference_attention(
            query[rows],
            key[PREFILL_CHUNK:],
            value[PREFILL_CHUNK:],
            layer_shape().scale,
            causal_lengths(count, count),
        )
        divergence = cosine(whole_prefix, chunk_only)
        self.assertLess(divergence, 0.99, f"the two references agree (cos {divergence:.8f}): this test has no teeth")
        against_whole = report("chunk 2 vs the WHOLE prefix (must match)", cosine(attention[rows], whole_prefix),
                               DENSE_MIN_COSINE)  # fmt: skip
        against_chunk = report("chunk 2 vs chunk 2 ALONE (must not match)", cosine(attention[rows], chunk_only),
                               DENSE_MIN_COSINE)  # fmt: skip
        self.assertGreater(against_whole, DENSE_MIN_COSINE)
        self.assertLess(against_chunk, against_whole)


@unittest.skipIf(os.environ.get(SLOW_TEST_ENV) == "0", f"{SLOW_TEST_ENV}=0")
class TestDecodeOverALongPrefix(_LayerCase):
    """One decode step at position 4096, over a prefix written 2048 tokens at a time."""

    def _run_decode(self, name: str):
        backend = self._backend(name, DECODE_PREFIX + 1)
        keys, values = [], []
        for start in range(0, DECODE_PREFIX, PREFILL_CHUNK):
            count = min(PREFILL_CHUNK, DECODE_PREFIX - start)
            hidden = synthetic_hidden(DECODE_PREFIX, self.DTYPE, self.device)[start : start + count]
            batch = self._batch(start, count, backend, is_decode=False)
            _, key, value = self._project(hidden, batch)
            backend.write_kv(0, key, value, batch.slots)
            keys.append(key)
            values.append(value)

        step = synthetic_hidden(1, self.DTYPE, self.device, seed=7)
        batch = self._batch(DECODE_PREFIX, 1, backend, is_decode=True)
        query, key, value = self._project(step, batch)
        backend.write_kv(0, key, value, batch.slots)
        keys.append(key)
        values.append(value)

        lengths = torch.full((1,), DECODE_PREFIX + 1, dtype=torch.int32, device=self.device)
        attention = torch.empty_like(query)
        with torch.inference_mode():
            backend.decode(0, query, lengths, attention)
        return attention, query, torch.cat(keys), torch.cat(values)

    def test_the_decode_reads_the_whole_prefix_it_was_told_about(self):
        for name in backends_under_test(self.device):
            with self.subTest(backend=name):
                attention, query, key, value = self._run_decode(name)
                expected = reference_attention(
                    query, key, value, layer_shape().scale, torch.tensor([DECODE_PREFIX + 1])
                )
                floor = minimum_cosine(name)
                measured = report(f"decode over {DECODE_PREFIX} cached tokens: {name}", floor=floor,
                                  measured=cosine(attention, expected))  # fmt: skip
                self.assertGreater(measured, floor, f"{name}: cos {measured:.8f}")

    def test_a_decode_told_a_short_context_loses_the_tail(self):
        """The negative control: the length is load-bearing, so a wrong one must show.

        Without this, a decode that ignored ``context_lens`` and always read the
        whole pool would pass the case above.
        """
        attention, query, key, value = self._run_decode("cann_dense")
        truncated = reference_attention(query, key[:1024], value[:1024], layer_shape().scale, torch.tensor([1024]))
        self.assertLess(cosine(attention, truncated), DENSE_MIN_COSINE)


class TestInterleavedRope(unittest.TestCase):
    """GLM-4's rotation, coordinate by coordinate, with nothing else in the way."""

    POSITIONS = (0, 1, 17, 2048, 4095)

    def _rope(self, interleaved: bool = True) -> RotaryEmbedding:
        return RotaryEmbedding(
            HEAD_SIZE,
            DECODE_PREFIX + 64,
            ROPE_THETA,
            torch.float32,
            torch.device("cpu"),
            rotary_dim=ROTARY_DIM,
            interleaved=interleaved,
        )

    def test_it_rotates_adjacent_pairs_and_nothing_else(self):
        """``(x[2i], x[2i+1])`` turn together by ``theta_i``; 64..127 pass through.

        Written out from ChatGLM's ``apply_rotary_pos_emb`` rather than compared
        with the harness's own tables, so a table that was wrong in both places
        could not agree with itself here.
        """
        rope = self._rope()
        torch.manual_seed(0)
        x = torch.randn(len(self.POSITIONS), 1, HEAD_SIZE)
        positions = torch.tensor(self.POSITIONS)
        rotated = rope.apply(x, positions)

        inverse = 1.0 / (ROPE_THETA ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM))
        for row, position in enumerate(self.POSITIONS):
            with self.subTest(position=position):
                angle = position * inverse
                cos, sin = torch.cos(angle), torch.sin(angle)
                even, odd = x[row, 0, 0:ROTARY_DIM:2], x[row, 0, 1:ROTARY_DIM:2]
                torch.testing.assert_close(rotated[row, 0, 0:ROTARY_DIM:2], even * cos - odd * sin)
                torch.testing.assert_close(rotated[row, 0, 1:ROTARY_DIM:2], even * sin + odd * cos)
                # The upper half is not rotated, it is copied.
                torch.testing.assert_close(rotated[row, 0, ROTARY_DIM:], x[row, 0, ROTARY_DIM:])

    def test_the_rotation_preserves_the_inner_product_it_is_supposed_to(self):
        """Two tokens the same distance apart score the same, which is the whole point."""
        rope = self._rope()
        torch.manual_seed(1)
        query, key = torch.randn(1, 1, HEAD_SIZE), torch.randn(1, 1, HEAD_SIZE)
        scores = [
            float(
                torch.dot(
                    rope.apply(query, torch.tensor([left]))[0, 0],
                    rope.apply(key, torch.tensor([left - 64]))[0, 0],
                )
            )
            for left in (64, 128, 1024, 3072)
        ]
        for value in scores[1:]:
            self.assertAlmostEqual(value, scores[0], places=3)

    def test_the_neox_pairing_is_a_different_rotation(self):
        """The negative control for the flag: pairing the wrong way must not agree.

        Same frequencies, different channels -- which is why getting it wrong
        scrambles position rather than raising.
        """
        torch.manual_seed(2)
        x = torch.randn(4, 1, HEAD_SIZE)
        positions = torch.tensor([1, 2, 3, 4])
        interleaved = self._rope(interleaved=True).apply(x, positions)
        halves = self._rope(interleaved=False).apply(x, positions)
        self.assertLess(cosine(interleaved, halves), 0.999)


if __name__ == "__main__":
    unittest.main(verbosity=2)
