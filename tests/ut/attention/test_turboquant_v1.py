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
"""Device-free tests for the TurboQuant 4-bit KV cache backend.

These pin down the arithmetic contract the Ascend C kernel has to honour: Pi is
a symmetric orthogonal involution, the packed byte round-trips exactly, the
operators are called with pure-runtime signatures, q/k/v are never rewritten,
and the output projection is the one weight Pi may be folded into -- with the
decode output left rotated for a folded layer and un-rotated for any other.
"""

import json
import math
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.attention.turboquant_cpu_ops import TURBOQUANT_CUBE_OP_SCHEMAS, turboquant_cube_meta_ops
from tests.ut.base import TestBase
from vllm_ascend.attention import turboquant_rotation as rotation_module
from vllm_ascend.attention import turboquant_trace as tq_trace
from vllm_ascend.attention import turboquant_v1 as tq_module
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.turboquant_rotation import (
    TURBOQUANT_FOLD_MIN_COSINE,
    TURBOQUANT_OUTPUT_ROTATION_CONFIG_KEY,
    TURBOQUANT_PI_SEED,
    apply_pi,
    fold_pi_into_output_projection,
    output_rotation_is_folded,
    output_rotation_marker,
    turboquant_pi_signs,
    validate_output_projection_fold,
    walsh_hadamard,
)
from vllm_ascend.attention.turboquant_trace import (
    TURBOQUANT_CASES_FILENAME,
    TURBOQUANT_CRASH_FILENAME,
    TURBOQUANT_RING_CAPACITY,
    TURBOQUANT_TRACE_ALIGNMENTS,
    TURBOQUANT_TRACE_PREVIEW,
    cache_planes,
    compact_tensor,
    host_seq_lengths,
    describe_indices,
    describe_tensor,
    reset_turboquant_tracer,
    turboquant_tracer,
)
from vllm_ascend.attention.turboquant_v1 import (
    TURBOQUANT_BURST_FLOATS,
    TURBOQUANT_PACK_FACTOR,
    TURBOQUANT_TILE_ROWS,
    AscendTurboQuantAttentionBackend,
    AscendTurboQuantAttentionBackendImpl,
    turboquant_codec_tables,
    turboquant_hadamard16,
    turboquant_scale_slot,
)
from vllm_ascend.attention.turboquant_v1 import (
    TURBOQUANT_LEVELS,
    TURBOQUANT_LLOYD_MAX_CENTROIDS,
    TURBOQUANT_LLOYD_MAX_THRESHOLDS,
)

HEAD_SIZE = 128
PACK_HIGH = 16.0
INT8_BIAS = 128.0
# TurboQuantCodec<4>::kEps -- the scale floor that keeps an all-zero vector from
# dividing by zero.
TURBOQUANT_EPS = 1e-20

CENTROIDS = torch.tensor(TURBOQUANT_LLOYD_MAX_CENTROIDS, dtype=torch.float64)
THRESHOLDS = torch.tensor(TURBOQUANT_LLOYD_MAX_THRESHOLDS, dtype=torch.float64)
# Bin 7 is where a zero coordinate lands: it clears the seven negative
# boundaries and fails the one at exactly 0, which the codec compares strictly.
ZERO_BIN = 7

CPU = torch.device("cpu")

# An arbitrary workspace size for the mocked operator to report. The real number
# depends on the device's vector core count, which is exactly why the backend
# asks the operator for it instead of computing it here.
WORKSPACE_FLOATS = 4096

# The block size of the decode fixtures' caches; the workspace plan needs it to bound the context.
DECODE_BLOCK_SIZE = 128


def _capture_in_progress():
    """Make ``tq_module._is_capturing()`` answer True.

    Both halves of the signal have to be faked, because the backend deliberately
    needs both: ``ACLGraphWrapper`` leaves ``forward_context.capturing`` set for the
    rest of the forward once it has captured a piece, so the flag alone is also True
    while the attention layers run *outside* the capture that piecewise compilation
    splits them out of. Only the stream says whether a capture is open right now --
    and on a host with no device it says no, which is why a test that patches one
    has to patch the other.
    """
    return (
        patch.object(tq_module, "_EXTRA_CTX", SimpleNamespace(capturing=True)),
        patch.object(torch.npu, "is_current_stream_capturing", lambda: True),
    )


def _ops_mock(workspace_floats: int = WORKSPACE_FLOATS) -> MagicMock:
    """A stand-in for ``torch.ops._C_ascend``.

    ``npu_turboquant_workspace_size`` has to answer with a real integer: the
    backend sizes its persistent buffer from it, so a bare MagicMock would fail
    the ``int()`` rather than exercise the path under test.
    """
    ops = MagicMock()
    ops.npu_turboquant_workspace_size.return_value = workspace_floats
    return ops


def _numeric_ops(rotated_output: torch.Tensor) -> MagicMock:
    """Operators that compute: rotate_q is Pi, and the decode writes a known
    rotated output. What the backend then leaves in ``output`` is the thing under
    test, not which mocks it happened to call."""
    ops = _ops_mock()

    def rotate_q(x, signs, tables, h16, out):
        out.copy_(apply_pi(x.to(torch.float32), signs))

    def paged_attention(*args):
        args[-1].copy_(rotated_output)

    ops.npu_turboquant_rotate_q.side_effect = rotate_q
    ops.npu_turboquant_paged_attention.side_effect = paged_attention
    return ops

# The first eight channels of the shared LCG sign vector. Hard-coded so this
# test and csrc/tests/reference/turbo_quant_cpu.h pin the same constant from
# both sides: a cache written by one must be readable by the other.
EXPECTED_SIGNS_128 = [1, -1, 1, -1, -1, -1, 1, 1]
EXPECTED_SIGNS_256 = [-1, 1, -1, -1, -1, -1, 1, 1]


def _quantize(vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for TurboQuantCodec<4>::Quantize4Bit.

    The scale is the per-vector RMS ``||v||_2 / sqrt(d)``, which is the metric
    the Lloyd-Max table is stated in, and the bin is the strict
    ``sum_i [u > t_i]`` the kernel evaluates from a sign bit.  The ``kEps``
    floor mirrors the codec's: without it an all-zero vector divides by zero and
    every bin comes back NaN.
    """
    d = vec.shape[-1]
    scale = vec.pow(2).sum(dim=-1, keepdim=True).sqrt() / math.sqrt(d) + TURBOQUANT_EPS
    unit = vec / scale
    bins = (unit.unsqueeze(-1) > THRESHOLDS.to(unit.dtype)).sum(-1).to(unit.dtype)
    packed = bins[..., 0::2] + PACK_HIGH * bins[..., 1::2] - INT8_BIAS
    return packed.round().to(torch.int8), scale.squeeze(-1)


def _dequantize_levels(packed: torch.Tensor) -> torch.Tensor:
    """Reference for TurboQuantCodec<4>::Dequantize4Bit (bare centroids)."""
    return CENTROIDS.to(torch.float64)[_dequantize_bins(packed).to(torch.int64)]


def _dequantize_bins(packed: torch.Tensor) -> torch.Tensor:
    """The bin indices ``Dequantize4Bit`` gathers the centroids with."""
    byte = packed.to(torch.float64) + INT8_BIAS
    expanded = byte.repeat_interleave(TURBOQUANT_PACK_FACTOR, dim=-1)
    high = torch.floor(expanded / PACK_HIGH)
    low = expanded - PACK_HIGH * high
    odd = (torch.arange(expanded.shape[-1]) % 2).to(expanded.dtype)
    return low + odd * (high - low)


class TestWalshHadamard(TestBase):
    def test_orthonormal_involution(self):
        eye = torch.eye(HEAD_SIZE, dtype=torch.float64)
        h = walsh_hadamard(eye, dim=-1)
        torch.testing.assert_close(h @ h.T, eye, atol=1e-10, rtol=0)
        torch.testing.assert_close(walsh_hadamard(h, dim=-1), eye, atol=1e-10, rtol=0)

    def test_rejects_non_power_of_two(self):
        with self.assertRaises(ValueError):
            walsh_hadamard(torch.zeros(3, dtype=torch.float64), dim=-1)

    def test_transforms_the_requested_axis(self):
        x = torch.randn(5, HEAD_SIZE, 3, dtype=torch.float64)
        torch.testing.assert_close(
            walsh_hadamard(x, dim=1),
            walsh_hadamard(x.movedim(1, -1), dim=-1).movedim(-1, 1),
        )


class TestPiRotation(TestBase):
    """Pi = D H D. The properties below are what let one kernel routine both
    rotate the activations and un-rotate the attention output."""

    def setUp(self):
        self.signs = turboquant_pi_signs(HEAD_SIZE, CPU).to(torch.float64)

    def test_signs_match_the_cpp_reference_lcg(self):
        for head_size, expected in ((128, EXPECTED_SIGNS_128), (256, EXPECTED_SIGNS_256)):
            signs = turboquant_pi_signs(head_size, CPU)
            self.assertEqual(signs.numel(), head_size)
            self.assertTrue(bool((signs.abs() == 1).all()))
            self.assertEqual(signs[: len(expected)].to(torch.int32).tolist(), expected)

    def test_signs_are_deterministic(self):
        torch.testing.assert_close(turboquant_pi_signs(HEAD_SIZE, CPU).to(torch.float64), self.signs)

    def test_pi_is_an_involution(self):
        x = torch.randn(8, HEAD_SIZE, dtype=torch.float64)
        torch.testing.assert_close(apply_pi(apply_pi(x, self.signs), self.signs), x, atol=1e-9, rtol=0)

    def test_pi_is_symmetric_and_orthogonal(self):
        eye = torch.eye(HEAD_SIZE, dtype=torch.float64)
        pi = apply_pi(eye, self.signs)
        torch.testing.assert_close(pi, pi.T, atol=1e-10, rtol=0)
        torch.testing.assert_close(pi @ pi.T, eye, atol=1e-10, rtol=0)

    def test_preserves_dot_products(self):
        q = torch.randn(16, HEAD_SIZE, dtype=torch.float64)
        k = torch.randn(16, HEAD_SIZE, dtype=torch.float64)
        torch.testing.assert_close(
            (apply_pi(q, self.signs) * apply_pi(k, self.signs)).sum(-1),
            (q * k).sum(-1),
            atol=1e-9,
            rtol=0,
        )


class TestCodecRoundTrip(TestBase):
    def setUp(self):
        self.signs = turboquant_pi_signs(HEAD_SIZE, CPU).to(torch.float64)

    def test_packed_byte_round_trips_exactly(self):
        bins = torch.randint(0, TURBOQUANT_LEVELS, (64, HEAD_SIZE)).to(torch.float64)
        packed = (bins[..., 0::2] + PACK_HIGH * bins[..., 1::2] - INT8_BIAS).round().to(torch.int8)
        self.assertEqual(packed.shape[-1], HEAD_SIZE // TURBOQUANT_PACK_FACTOR)
        torch.testing.assert_close(_dequantize_bins(packed), bins)
        torch.testing.assert_close(_dequantize_levels(packed), CENTROIDS[bins.to(torch.int64)])

    def test_rotation_shrinks_heavy_tailed_error(self):
        torch.manual_seed(0)
        x = torch.randn(512, HEAD_SIZE, dtype=torch.float64)
        # One dominant channel is exactly the case absmax quantisation handles
        # worst, and the case the rotation exists to fix.
        x[:, 0] *= 40.0

        packed, step = _quantize(x)
        plain = (_dequantize_levels(packed) * step.unsqueeze(-1) - x).norm(dim=-1) / x.norm(dim=-1)

        packed, step = _quantize(apply_pi(x, self.signs))
        recovered = apply_pi(_dequantize_levels(packed) * step.unsqueeze(-1), self.signs)
        rotated = (recovered - x).norm(dim=-1) / x.norm(dim=-1)

        self.assertLess(rotated.mean().item(), 0.2)
        self.assertLess(rotated.mean().item(), plain.mean().item() / 3.0)

    def test_quantised_attention_tracks_exact_attention(self):
        torch.manual_seed(0)
        num_ctx, num_heads = 256, 4
        q = torch.randn(num_heads, HEAD_SIZE, dtype=torch.float64) / HEAD_SIZE**0.5
        k = torch.randn(num_ctx, HEAD_SIZE, dtype=torch.float64)
        v = torch.randn(num_ctx, HEAD_SIZE, dtype=torch.float64)
        scale = HEAD_SIZE**-0.5

        exact = torch.softmax((q @ k.T) * scale, dim=-1) @ v

        k_packed, k_step = _quantize(apply_pi(k, self.signs))
        v_packed, v_step = _quantize(apply_pi(v, self.signs))
        # The kernel folds the per-vector step into the score row and into the
        # softmax probabilities, never into a per-channel broadcast.
        scores = (apply_pi(q, self.signs) @ _dequantize_levels(k_packed).T) * k_step * scale
        probs = torch.softmax(scores, dim=-1) * v_step
        # What the combine writes: the accumulator, still rotated.
        rotated = probs @ _dequantize_levels(v_packed)

        # Un-rotated on the device, for a layer whose o_proj is not folded.
        approx = apply_pi(rotated, self.signs)
        cosine = torch.nn.functional.cosine_similarity(approx, exact, dim=-1)
        self.assertGreater(cosine.min().item(), 0.98)

        # Or never un-rotated at all: the folded projection of the rotated output
        # is the original projection of the exact one.
        o_proj = torch.randn(3 * HEAD_SIZE, num_heads * HEAD_SIZE, dtype=torch.float64)
        folded = fold_pi_into_output_projection(o_proj, HEAD_SIZE)
        projected = torch.nn.functional.cosine_similarity(
            rotated.reshape(-1) @ folded.T, exact.reshape(-1) @ o_proj.T, dim=0
        )
        self.assertGreater(projected.item(), 0.98)


class TestPureRuntimeContract(TestBase):
    """The invariants: Q, K and V rotate in the kernel and never in a weight, the
    operators carry no basis-selection flag, and the output projection is the
    only weight Pi is ever folded into."""

    def test_only_the_output_projection_has_a_fold_helper(self):
        for module in (tq_module, rotation_module):
            for name in (
                "fold_pi_into_attention_projections",
                "fold_pi_into_qkv_projection",
                "maybe_trans_nz_with_pi",
            ):
                self.assertFalse(
                    hasattr(module, name), f"{name} must not come back: folding Pi into q/k/v moves RoPE's basis"
                )
        self.assertTrue(callable(rotation_module.fold_pi_into_output_projection))

    def test_impl_carries_no_rotation_toggle(self):
        self.assertFalse(hasattr(AscendTurboQuantAttentionBackendImpl, "apply_rotation"))

    def _make_impl(self) -> AscendTurboQuantAttentionBackendImpl:
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        impl.head_size = HEAD_SIZE
        impl.num_heads = 8
        impl.num_kv_heads = 2
        impl.scale = HEAD_SIZE**-0.5
        impl.attn_type = "decoder"
        impl.kv_sharing_target_layer_name = None
        impl.is_kv_producer = False
        impl.key_cache = None
        impl.value_cache = None
        impl.scale_cache = None
        impl._pi_signs = None
        impl.decode_workspace = None
        impl._workspace_floats = {}
        impl.rotated_query = None
        impl._hadamard16 = None
        impl.output_rotation_folded = False
        return impl

    def _decode_fixture(self, impl, num_tokens):
        impl.key_cache = torch.zeros(1, DECODE_BLOCK_SIZE, 2, HEAD_SIZE // TURBOQUANT_PACK_FACTOR, dtype=torch.int8)
        impl.value_cache = impl.key_cache
        impl.scale_cache = torch.zeros(1, DECODE_BLOCK_SIZE, turboquant_scale_slot(2))
        query = torch.randn(num_tokens, impl.num_heads, HEAD_SIZE)
        output = torch.zeros(num_tokens, impl.num_heads, HEAD_SIZE)
        metadata = MagicMock(
            block_tables=torch.zeros(num_tokens, 1, dtype=torch.int64),
            seq_lens=torch.ones(num_tokens, dtype=torch.int64),
        )
        return query, output, metadata

    def test_reshape_and_cache_passes_activations_through_unrotated(self):
        impl = self._make_impl()
        num_tokens, num_blocks, block_size = 4, 3, 128
        key = torch.randn(num_tokens, impl.num_kv_heads, HEAD_SIZE)
        value = torch.randn_like(key)
        cache = torch.zeros(
            num_blocks, block_size, impl.num_kv_heads, HEAD_SIZE // TURBOQUANT_PACK_FACTOR, dtype=torch.int8
        )
        metadata = MagicMock(num_actual_tokens=num_tokens, slot_mapping=torch.arange(num_tokens, dtype=torch.int64))

        ops = MagicMock()
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl.reshape_and_cache(None, key, value, (cache, cache), metadata, None)

        ops.npu_turboquant_reshape_and_cache.assert_called_once()
        args = ops.npu_turboquant_reshape_and_cache.call_args.args
        # key, value, k_cache, v_cache, scale_cache, slots, pi_signs,
        # codec_tables -- and no rotation flag anywhere.
        self.assertEqual(len(args), 8)
        torch.testing.assert_close(args[0], key)
        torch.testing.assert_close(args[1], value)
        self.assertEqual(args[5].dtype, torch.int32)
        torch.testing.assert_close(args[6], turboquant_pi_signs(HEAD_SIZE, CPU))
        # The codec tables are host-built, not regenerated per launch.
        torch.testing.assert_close(args[7], turboquant_codec_tables(HEAD_SIZE, 1, CPU))

    def test_paged_attention_rotates_the_query_in_its_own_operator(self):
        impl = self._make_impl()
        impl.output_rotation_folded = True
        num_tokens = 2
        query, output, metadata = self._decode_fixture(impl, num_tokens)

        ops = _ops_mock()
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl.forward_paged_attention(query, metadata, output)

        # The rotation is its own launch, and it comes first: the split kernel
        # applies no Walsh-Hadamard transform any more, so a decode that skipped
        # this call would read an unrotated query and produce plausible-looking
        # nonsense rather than fail. With o_proj folded it is the only one.
        ops.npu_turboquant_rotate_q.assert_called_once()
        rot_args = ops.npu_turboquant_rotate_q.call_args.args
        # query, pi_signs, codec_tables, hadamard16, query_rot.
        self.assertEqual(len(rot_args), 5)
        torch.testing.assert_close(rot_args[0], query)
        torch.testing.assert_close(rot_args[1], turboquant_pi_signs(HEAD_SIZE, CPU))
        # batch_rows 1: the rotation expands one vector per call, as the write
        # path does -- not the decode's kTileRows image.
        torch.testing.assert_close(rot_args[2], turboquant_codec_tables(HEAD_SIZE, 1, CPU))
        torch.testing.assert_close(rot_args[3], turboquant_hadamard16(CPU))
        self.assertEqual(rot_args[4].dtype, torch.float32)
        self.assertEqual(rot_args[4].shape, (num_tokens, impl.num_heads, HEAD_SIZE))

        ops.npu_turboquant_paged_attention.assert_called_once()
        args = ops.npu_turboquant_paged_attention.call_args.args
        # query_rot, k_cache, v_cache, scale_cache, block_tables, context_lens,
        # codec_tables, workspace, num_kv_heads, num_heads, scale, out. No
        # pi_signs: neither kernel rotates, so nothing would read it.
        self.assertEqual(len(args), 12)
        # The decode reads the rotation's output, not the model's query. Same
        # tensor object, so the two launches cannot disagree about the buffer.
        self.assertIs(args[0], rot_args[4])
        torch.testing.assert_close(args[6], turboquant_codec_tables(HEAD_SIZE, TURBOQUANT_TILE_ROWS, CPU))
        self.assertEqual(args[8], impl.num_kv_heads)
        self.assertEqual(args[9], impl.num_heads)
        self.assertFalse(any(isinstance(a, bool) for a in args))
        self.assertFalse(any(a is turboquant_pi_signs(HEAD_SIZE, CPU) for a in args))

    def test_folded_decode_leaves_the_output_rotated(self):
        """A folded o_proj un-rotates, so the backend must not."""
        impl = self._make_impl()
        impl.output_rotation_folded = True
        query, output, metadata = self._decode_fixture(impl, 2)
        rotated = torch.randn_like(output)

        ops = _numeric_ops(rotated)
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl.forward_paged_attention(query, metadata, output)

        torch.testing.assert_close(output, rotated)
        self.assertEqual(ops.npu_turboquant_rotate_q.call_count, 1)

    def test_unfolded_decode_unrotates_its_output_on_the_device(self):
        """Without a fold the output reaching o_proj has to be O = Pi O~, and the
        un-rotation is a launch of the same rotation operator, not host code."""
        impl = self._make_impl()
        query, output, metadata = self._decode_fixture(impl, 3)
        rotated = torch.randn_like(output)
        signs = turboquant_pi_signs(HEAD_SIZE, CPU)

        ops = _numeric_ops(rotated)
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl.forward_paged_attention(query, metadata, output)

        torch.testing.assert_close(output, apply_pi(rotated, signs), atol=1e-5, rtol=0)
        self.assertEqual(ops.npu_turboquant_rotate_q.call_count, 2)
        first, second = ops.npu_turboquant_rotate_q.call_args_list
        # The un-rotation lands in the rotated query's own buffer: the split has
        # consumed it by then, and a second high-water buffer would be a second
        # thing that can grow during a capture.
        self.assertIs(second.args[4], first.args[4])
        self.assertEqual(second.args[4].data_ptr(), impl.rotated_query.data_ptr())

    def test_rotated_query_buffer_is_preallocated_and_reused(self):
        """The rotation never allocates on a decode step, for the same reason
        the reduction workspace does not: a graph replays captured addresses."""
        impl = self._make_impl()
        impl.key_cache = torch.zeros(1, DECODE_BLOCK_SIZE, 2, HEAD_SIZE // TURBOQUANT_PACK_FACTOR, dtype=torch.int8)
        impl.value_cache = impl.key_cache
        impl.scale_cache = torch.zeros(1, DECODE_BLOCK_SIZE, turboquant_scale_slot(2))

        def run(num_tokens):
            query = torch.randn(num_tokens, impl.num_heads, HEAD_SIZE)
            output = torch.zeros(num_tokens, impl.num_heads, HEAD_SIZE)
            metadata = MagicMock(
                block_tables=torch.zeros(num_tokens, 1, dtype=torch.int64),
                seq_lens=torch.ones(num_tokens, dtype=torch.int64),
            )
            ops = _ops_mock(workspace_floats=WORKSPACE_FLOATS)
            with patch.object(torch.ops, "_C_ascend", ops, create=True):
                impl.forward_paged_attention(query, metadata, output)
            return ops.npu_turboquant_rotate_q.call_args.args[4]

        wide = run(4)
        self.assertIsNotNone(impl.rotated_query)
        storage = impl.rotated_query
        # A narrower step reuses the same allocation rather than making a new
        # one, and takes a correctly shaped view of its prefix.
        narrow = run(2)
        self.assertIs(impl.rotated_query, storage)
        self.assertEqual(narrow.shape, (2, impl.num_heads, HEAD_SIZE))
        self.assertEqual(wide.shape, (4, impl.num_heads, HEAD_SIZE))

    def test_the_rotated_query_buffer_is_reserved_at_the_configured_ceiling(self):
        """A decode carries one token per sequence, so ``max_num_seqs`` is the ceiling.
        Reserving it up front means no graph capture is ever the step that grows it."""
        impl = self._make_impl()
        impl.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=32),
            model_config=SimpleNamespace(max_model_len=1024),
            cache_config=SimpleNamespace(block_size=DECODE_BLOCK_SIZE),
        )
        view = impl._rotated_query(2, CPU)
        self.assertEqual(view.shape, (2, impl.num_heads, HEAD_SIZE))
        self.assertEqual(impl.rotated_query.numel(), 32 * impl.num_heads * HEAD_SIZE)

        reserved = impl.rotated_query
        flag, stream = _capture_in_progress()
        with flag, stream:
            widest = impl._rotated_query(32, CPU)
        self.assertIs(impl.rotated_query, reserved)
        self.assertEqual(widest.shape, (32, impl.num_heads, HEAD_SIZE))

    def test_paged_attention_is_handed_a_preallocated_workspace(self):
        """The operator never allocates its own reduction scratch."""
        impl = self._make_impl()
        num_tokens = 2
        query, output, metadata = self._decode_fixture(impl, num_tokens)

        ops = _ops_mock(workspace_floats=WORKSPACE_FLOATS)
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl.forward_paged_attention(query, metadata, output)

        workspace = ops.npu_turboquant_paged_attention.call_args.args[7]
        self.assertIs(workspace, impl.decode_workspace)
        self.assertEqual(workspace.dtype, torch.float32)
        self.assertEqual(workspace.numel(), WORKSPACE_FLOATS)
        # Sized from the operator's own arithmetic, not a copy of it here.
        ops.npu_turboquant_workspace_size.assert_called_once_with(
            num_tokens, impl.num_heads, impl.head_size, 1, DECODE_BLOCK_SIZE
        )


class TestCacheWriteValidation(TestBase):
    """The write path's guard rails.

    Every address the writer forms is scaled by the token's slot, so a slot the cache
    cannot hold is a write outside it -- which the AI core reports as "the address for
    scalar to access GM is invalid" (error 264) naming nothing at all. The kernels drop
    such a slot rather than writing it, the same way they drop the negative slot vLLM
    marks a padding token with. Dropping is silent, so
    ``VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS`` exists to name it; it is off by default
    because reading the values costs a device synchronisation per layer per step.
    """

    NUM_BLOCKS = 3
    BLOCK_SIZE = 128
    NUM_TOKENS = 4

    def _make_impl(self) -> AscendTurboQuantAttentionBackendImpl:
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        impl.head_size = HEAD_SIZE
        impl.num_heads = 8
        impl.num_kv_heads = 2
        impl.scale = HEAD_SIZE**-0.5
        impl.attn_type = "decoder"
        impl.kv_sharing_target_layer_name = None
        impl.is_kv_producer = False
        impl.cube_decode = False
        impl.key_cache = None
        impl.value_cache = None
        impl.scale_cache = None
        impl._pi_signs = None
        impl._hadamard16 = None
        return impl

    def _write(self, slots: torch.Tensor, validate: bool, key: torch.Tensor | None = None) -> MagicMock:
        """Run one cache write; returns the operator mock it was pushed through."""
        impl = self._make_impl()
        cache = torch.zeros(
            self.NUM_BLOCKS,
            self.BLOCK_SIZE,
            impl.num_kv_heads,
            HEAD_SIZE // TURBOQUANT_PACK_FACTOR,
            dtype=torch.int8,
        )
        if key is None:
            key = torch.randn(slots.numel(), impl.num_kv_heads, HEAD_SIZE)
        metadata = MagicMock(num_actual_tokens=slots.numel(), slot_mapping=slots)
        ops = MagicMock()
        with (
            patch.dict("os.environ", {"VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS": "1" if validate else "0"}),
            patch.object(torch.ops, "_C_ascend", ops, create=True),
        ):
            impl.reshape_and_cache(None, key, torch.randn_like(key), (cache, cache), metadata, None)
        return ops

    def _write_with(self, impl, slots: torch.Tensor) -> MagicMock:
        """One cache write through a caller-supplied impl, so its state survives the call."""
        cache = torch.zeros(
            self.NUM_BLOCKS,
            self.BLOCK_SIZE,
            impl.num_kv_heads,
            HEAD_SIZE // TURBOQUANT_PACK_FACTOR,
            dtype=torch.int8,
        )
        key = torch.randn(slots.numel(), impl.num_kv_heads, HEAD_SIZE)
        metadata = MagicMock(num_actual_tokens=slots.numel(), slot_mapping=slots)
        ops = MagicMock()
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl.reshape_and_cache(None, key, torch.randn_like(key), (cache, cache), metadata, None)
        return ops

    def _slots(self) -> torch.Tensor:
        return torch.arange(self.NUM_TOKENS, dtype=torch.int64)

    def test_slots_inside_the_cache_are_written(self):
        ops = self._write(self._slots(), validate=True)
        ops.npu_turboquant_reshape_and_cache.assert_called_once()

    def test_a_slot_past_the_last_row_is_named(self):
        rows = self.NUM_BLOCKS * self.BLOCK_SIZE
        slots = self._slots()
        slots[2] = rows
        with self.assertRaises(RuntimeError) as caught:
            self._write(slots, validate=True)
        message = str(caught.exception)
        self.assertIn(str(rows), message, message)
        self.assertIn("slot_mapping", message, message)

    def test_an_out_of_range_slot_is_not_checked_on_the_serving_path(self):
        """The guard that keeps it off the bus is the kernel's, not this one.

        Checking here costs a synchronisation, so by default the writer is launched
        and drops the row; this test pins that the hot path does not pay for the check.
        """
        slots = self._slots()
        slots[2] = self.NUM_BLOCKS * self.BLOCK_SIZE
        ops = self._write(slots, validate=False)
        ops.npu_turboquant_reshape_and_cache.assert_called_once()

    def test_a_padding_slot_warns_rather_than_raising(self):
        """-1 is how vLLM marks a token with no cache row; the kernels skip it."""
        slots = self._slots()
        slots[1] = -1
        with patch.object(tq_module, "logger") as logger:
            ops = self._write(slots, validate=True)
        ops.npu_turboquant_reshape_and_cache.assert_called_once()
        logger.warning_once.assert_called_once()

    def test_the_writer_is_handed_packed_operands(self):
        """A fused QKV split hands the backend a strided view; the kernel indexes raw memory."""
        qkv = torch.randn(self.NUM_TOKENS, 3, 2, HEAD_SIZE)
        key = qkv[:, 1]
        self.assertFalse(key.is_contiguous())
        ops = self._write(self._slots(), validate=True, key=key)
        args = ops.npu_turboquant_reshape_and_cache.call_args.args
        self.assertTrue(args[0].is_contiguous())
        self.assertTrue(args[1].is_contiguous())
        self.assertEqual(args[5].dtype, torch.int32)
        torch.testing.assert_close(args[0], key)

    def test_an_operand_that_starts_mid_burst_is_named(self):
        """``.contiguous()`` keeps a view's storage offset, so contiguity is not alignment."""
        impl = self._make_impl()
        elements = self.NUM_TOKENS * impl.num_kv_heads * HEAD_SIZE
        flat = torch.randn(elements + 1)
        key = flat[1:].view(self.NUM_TOKENS, impl.num_kv_heads, HEAD_SIZE)
        self.assertTrue(key.is_contiguous())
        # Precondition rather than an assumption: a one-float offset from an allocator
        # that aligns to at least a burst cannot land on a burst boundary.
        self.assertNotEqual(key.data_ptr() % tq_module.TURBOQUANT_GM_BURST_BYTES, 0)
        with self.assertRaises(RuntimeError) as caught:
            self._write(self._slots(), validate=True, key=key)
        self.assertIn("burst", str(caught.exception))

    def test_the_burst_matches_the_kernel_constant(self):
        """turboquant_layout.h: kFp32PerBlock fp32 lanes is one 32-byte burst."""
        self.assertEqual(tq_module.TURBOQUANT_GM_BURST_BYTES, 32)
        self.assertEqual(tq_module.TURBOQUANT_GM_BURST_BYTES, TURBOQUANT_BURST_FLOATS * 4)

    def test_the_write_is_fenced_on_device_and_nowhere_else(self):
        """A trap in the writer must not be reported against a later operator.

        The fence is what makes the failing launch legible: without it an Ascend kernel
        that traps fails a *subsequent* launch API call, which is how a write fault comes
        back naming an operator the writer never calls.
        """
        impl = self._make_impl()
        with patch.object(tq_module.torch, "npu", MagicMock(), create=True) as npu:
            impl._fence_cache_write(torch.device("cpu"))
            npu.synchronize.assert_not_called()
            impl._fence_cache_write(SimpleNamespace(type="npu"))
            npu.synchronize.assert_called_once()

    def test_the_serving_path_is_not_fenced(self):
        """The fence is a synchronisation per layer per step; it rides the diagnostic flag."""
        impl = self._make_impl()
        with patch.object(impl, "_fence_cache_write") as fence:
            with patch.dict("os.environ", {"VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS": "0"}):
                self._write_with(impl, self._slots())
            fence.assert_not_called()
            with patch.dict("os.environ", {"VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS": "1"}):
                self._write_with(impl, self._slots())
            fence.assert_called_once()


class TestDecodeWorkspace(TestBase):
    """The persistent reduction buffer.

    Decode must not allocate: an allocation is host latency on the critical path
    and, under graph capture, an address the replay has no reason to still own.
    The buffer is therefore grown to a high-water mark and then reused.
    """

    def _make_impl(self) -> AscendTurboQuantAttentionBackendImpl:
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        impl.head_size = HEAD_SIZE
        impl.num_heads = 8
        impl.decode_workspace = None
        impl._workspace_floats = {}
        return impl

    def test_second_decode_of_the_same_shape_allocates_nothing(self):
        impl = self._make_impl()
        ops = _ops_mock()
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            first = impl._decode_workspace(2, 1, DECODE_BLOCK_SIZE, CPU)
            second = impl._decode_workspace(2, 1, DECODE_BLOCK_SIZE, CPU)

        self.assertIs(first, second)
        # The size is memoised per shape, so the steady-state step costs a dict
        # lookup rather than an operator dispatch.
        ops.npu_turboquant_workspace_size.assert_called_once()

    def test_a_narrower_decode_reuses_the_high_water_buffer(self):
        impl = self._make_impl()
        ops = _ops_mock()
        ops.npu_turboquant_workspace_size.side_effect = [WORKSPACE_FLOATS, WORKSPACE_FLOATS // 4]
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            wide = impl._decode_workspace(8, 4, DECODE_BLOCK_SIZE, CPU)
            narrow = impl._decode_workspace(2, 1, DECODE_BLOCK_SIZE, CPU)

        self.assertIs(wide, narrow)
        self.assertEqual(narrow.numel(), WORKSPACE_FLOATS)

    def test_a_wider_decode_grows_the_buffer(self):
        impl = self._make_impl()
        ops = _ops_mock()
        ops.npu_turboquant_workspace_size.side_effect = [WORKSPACE_FLOATS, WORKSPACE_FLOATS * 2]
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            narrow = impl._decode_workspace(2, 1, DECODE_BLOCK_SIZE, CPU)
            wide = impl._decode_workspace(8, 4, DECODE_BLOCK_SIZE, CPU)

        self.assertIsNot(narrow, wide)
        self.assertEqual(wide.numel(), WORKSPACE_FLOATS * 2)
        self.assertIs(wide, impl.decode_workspace)

    def test_a_smaller_batch_that_needs_more_still_grows_the_buffer(self):
        """The buffer tracks the requirement, not the batch size.

        A decode too small to fill the cores is split further along the
        sequence, so num_splits rises as num_tokens falls and the scratch a
        1-token step needs can exceed what a 3-token step needs. Sizing off
        "the widest batch seen" would under-allocate exactly there.
        """
        impl = self._make_impl()
        ops = _ops_mock()
        ops.npu_turboquant_workspace_size.side_effect = [WORKSPACE_FLOATS, WORKSPACE_FLOATS * 2]
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            big_batch = impl._decode_workspace(3, 8, DECODE_BLOCK_SIZE, CPU)
            small_batch = impl._decode_workspace(1, 8, DECODE_BLOCK_SIZE, CPU)

        self.assertIsNot(big_batch, small_batch)
        self.assertEqual(small_batch.numel(), WORKSPACE_FLOATS * 2)

    def test_the_size_memo_is_bounded(self):
        """An uncaptured run drifts through shapes; the memo must not grow forever."""
        impl = self._make_impl()
        impl._workspace_floats = {(0, i): 1 for i in range(tq_module._WORKSPACE_MEMO_LIMIT)}
        ops = _ops_mock()
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl._decode_workspace(2, 1, DECODE_BLOCK_SIZE, CPU)

        self.assertEqual(len(impl._workspace_floats), 1)
        self.assertEqual(impl._workspace_floats[(2, 1)], WORKSPACE_FLOATS)

    def test_growth_during_a_capture_is_refused(self):
        """A buffer swapped mid-capture would be replayed as a dangling pointer."""
        impl = self._make_impl()
        ops = _ops_mock()
        ops.npu_turboquant_workspace_size.side_effect = [WORKSPACE_FLOATS, WORKSPACE_FLOATS * 2]
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl._decode_workspace(2, 1, DECODE_BLOCK_SIZE, CPU)
            flag, stream = _capture_in_progress()
            with flag, stream, self.assertRaisesRegex(RuntimeError, "during a graph capture"):
                impl._decode_workspace(8, 4, DECODE_BLOCK_SIZE, CPU)

    def test_a_capture_that_needs_no_growth_is_allowed(self):
        impl = self._make_impl()
        ops = _ops_mock()
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            warmed = impl._decode_workspace(2, 1, DECODE_BLOCK_SIZE, CPU)
            flag, stream = _capture_in_progress()
            with flag, stream:
                captured = impl._decode_workspace(2, 1, DECODE_BLOCK_SIZE, CPU)

        self.assertIs(warmed, captured)

    def test_a_missing_forward_context_does_not_read_as_a_capture(self):
        """Outside a forward context there is no capture to protect against."""

        class _NoContext:
            @property
            def capturing(self):
                raise AssertionError("no forward context")

        with patch.object(tq_module, "_EXTRA_CTX", _NoContext()):
            self.assertFalse(tq_module._is_capturing())

    def test_a_stale_capture_flag_between_pieces_does_not_read_as_a_capture(self):
        """The piecewise case, which is how TurboQuant is meant to be served.

        ``ACLGraphWrapper`` sets ``forward_context.capturing`` before capturing a
        piece and only the next step clears it. Attention is split out of those
        pieces, so from the first captured piece onwards every TurboQuant call in
        that forward sees the flag set while running outside any capture -- and
        must be free to grow a buffer. Only the stream can tell the two apart.
        """
        with (
            patch.object(tq_module, "_EXTRA_CTX", SimpleNamespace(capturing=True)),
            patch.object(torch.npu, "is_current_stream_capturing", lambda: False),
        ):
            self.assertFalse(tq_module._is_capturing())

    def test_an_unanswerable_stream_is_treated_as_a_capture(self):
        """Not knowing is not the same as knowing it is safe to allocate."""

        def unavailable():
            raise RuntimeError("no device")

        with (
            patch.object(tq_module, "_EXTRA_CTX", SimpleNamespace(capturing=True)),
            patch.object(torch.npu, "is_current_stream_capturing", unavailable),
        ):
            self.assertTrue(tq_module._is_capturing())

    def test_the_stream_is_not_asked_when_no_capture_is_running(self):
        """The common path stays off the device: the flag alone answers it."""
        asked = []

        def record():
            asked.append(True)
            return False

        with (
            patch.object(tq_module, "_EXTRA_CTX", SimpleNamespace(capturing=False)),
            patch.object(torch.npu, "is_current_stream_capturing", record),
        ):
            self.assertFalse(tq_module._is_capturing())
        self.assertEqual(asked, [])


class TestNumericalEdgeCases(TestBase):
    """The corners of the codec: an all-zero vector, a lone outlier, and inputs
    that land exactly on the boundary levels."""

    def test_all_zero_vector_is_numerically_stable(self):
        for head_size in (64, 128, 256):
            zeros = torch.zeros(head_size, dtype=torch.float64)
            packed, step = _quantize(zeros)

            # The eps floor keeps the reciprocal finite and the scale positive.
            self.assertTrue(torch.isfinite(step).all(), f"scale not finite at D={head_size}")
            self.assertGreater(step.item(), 0.0)
            self.assertAlmostEqual(step.item(), TURBOQUANT_EPS, delta=1e-30)

            # The Lloyd-Max table has no exact zero either: u = 0 fails the
            # strict comparison at the boundary t = 0 and clears the seven below
            # it, so every channel lands on bin 7. What has to hold is that the
            # reconstruction is numerically zero.
            levels = _dequantize_levels(packed)
            self.assertTrue(torch.isfinite(levels).all())
            torch.testing.assert_close(
                levels, torch.full_like(levels, CENTROIDS[ZERO_BIN].item()), atol=0, rtol=0
            )

            recon = levels * step
            self.assertTrue(torch.isfinite(recon).all())
            self.assertLess(recon.abs().max().item(), 1e-18)

    def test_single_dominant_outlier(self):
        head_size = 128
        for magnitude in (1.0e4, -1.0e4):
            vec = torch.zeros(head_size, dtype=torch.float64)
            vec[0] = magnitude
            packed, step = _quantize(vec)

            # The RMS scale moves by 1/sqrt(d) for a lone outlier rather than
            # keying on it. That is the deliberate difference from the absmax
            # codec: absmax sized the grid off this one coordinate and crushed
            # every other channel, where the RMS scale keeps the bulk resolved
            # and clips the outlier instead.
            self.assertTrue(torch.isfinite(step).all())
            self.assertAlmostEqual(
                step.item(), abs(magnitude) / math.sqrt(head_size), delta=abs(magnitude) * 1e-9
            )

            levels = _dequantize_levels(packed)
            extreme_bin = TURBOQUANT_LEVELS - 1 if magnitude > 0 else 0
            self.assertAlmostEqual(levels[0].item(), CENTROIDS[extreme_bin].item(), places=6)
            torch.testing.assert_close(
                levels[1:], torch.full_like(levels[1:], CENTROIDS[ZERO_BIN].item()), atol=0, rtol=0
            )

            # Packing an extreme beside a mid bin must stay inside int8.
            self.assertGreaterEqual(int(packed.min()), -128)
            self.assertLessEqual(int(packed.max()), 127)

            # The outlier is clipped to the outermost centroid, so it returns at
            # c[15] / sqrt(d) of its magnitude, not at its magnitude.
            recon = (levels * step)[0].item()
            clipped = magnitude * CENTROIDS[TURBOQUANT_LEVELS - 1].item() / math.sqrt(head_size)
            self.assertAlmostEqual(recon, clipped, delta=abs(clipped) * 1e-5)
            self.assertLess(abs(recon), abs(magnitude))

    def test_clamping_and_saturation_extremes(self):
        # Both nibbles at 15 give byte 255, which the -128 bias maps to +127;
        # both at 0 give -128. Those are the exact int8 endpoints, so a signed
        # overflow or a truncating cast shows up here and nowhere else.
        #
        # Reaching those bins takes a *sparse* outlier under an RMS scale. A
        # vector whose channels all share a magnitude normalises to u = +-1,
        # nowhere near the +-2.4008 outermost boundary; two live channels among
        # head_size - 2 zeros normalise to +-sqrt(head_size / 2) and saturate.
        head_size = 64
        amplitude = 3.0
        cases = (
            ("(+A, +A)", amplitude, amplitude, 15, 15, 127),
            ("(-A, -A)", -amplitude, -amplitude, 0, 0, -128),
            ("(+A, -A)", amplitude, -amplitude, 15, 0, -113),
            ("(-A, +A)", -amplitude, amplitude, 0, 15, 112),
        )
        quiet_byte = ZERO_BIN + int(PACK_HIGH) * ZERO_BIN - int(INT8_BIAS)
        for name, first, second, want_low, want_high, want_byte in cases:
            vec = torch.zeros(head_size, dtype=torch.float64)
            vec[0] = first
            vec[1] = second
            packed, scale = _quantize(vec)

            # If the sparse outlier stops saturating, the endpoints are untested
            # rather than merely differently encoded.
            self.assertGreater(amplitude / scale.item(), THRESHOLDS[-1].item(), name)

            self.assertEqual(int(packed[0]), want_byte, name)
            self.assertEqual(packed[1:].unique().tolist(), [quiet_byte], name)
            self.assertGreaterEqual(int(packed.min()), -128, name)
            self.assertLessEqual(int(packed.max()), 127, name)

            bins = _dequantize_bins(packed)
            self.assertEqual(int(bins[0]), want_low, name)
            self.assertEqual(int(bins[1]), want_high, name)
            self.assertEqual(bins[2:].unique().tolist(), [float(ZERO_BIN)], name)

    def test_every_nibble_pair_round_trips(self):
        # All 256 (low, high) combinations, endpoints included.
        lows = torch.arange(16).repeat_interleave(16)
        highs = torch.arange(16).repeat(16)
        packed = (lows + PACK_HIGH * highs - INT8_BIAS).round().to(torch.int8).unsqueeze(-1)
        self.assertGreaterEqual(int(packed.min()), -128)
        self.assertLessEqual(int(packed.max()), 127)

        bins = _dequantize_bins(packed)
        torch.testing.assert_close(bins[:, 0], lows.to(bins.dtype), atol=0, rtol=0)
        torch.testing.assert_close(bins[:, 1], highs.to(bins.dtype), atol=0, rtol=0)

        levels = _dequantize_levels(packed)
        torch.testing.assert_close(levels[:, 0], CENTROIDS[lows], atol=0, rtol=0)
        torch.testing.assert_close(levels[:, 1], CENTROIDS[highs], atol=0, rtol=0)

    def test_involution_on_canonical_basis_vectors(self):
        for head_size in (64, 128, 256):
            signs = turboquant_pi_signs(head_size, CPU).to(torch.float64)
            for index in (0, 1, 7, 8, head_size - 1):
                basis = torch.zeros(head_size, dtype=torch.float64)
                basis[index] = 1.0

                rotated = apply_pi(basis, signs)
                # Pi is orthogonal: a unit basis vector stays unit, and its
                # energy is spread evenly instead of sitting in one channel.
                self.assertAlmostEqual(rotated.norm().item(), 1.0, delta=1e-6)
                torch.testing.assert_close(
                    rotated.abs(),
                    torch.full_like(rotated, head_size**-0.5),
                    atol=1e-9,
                    rtol=0,
                )
                torch.testing.assert_close(apply_pi(rotated, signs), basis, atol=1e-6, rtol=0)


class TestBufferPoisoning(TestBase):
    """The kernel sizes its scratch for a full ``batchRows`` and TPipe hands that
    pool over uninitialised, so a call with fewer rows runs with live garbage
    right behind its payload. These mirror that on the host: neighbouring memory
    is filled with values no arithmetic can produce, and a leak shows up either
    as a poisoned value in the payload or as a difference between two runs whose
    padding was poisoned differently."""

    # Chosen so a leak survives an isnan() filter: a quiet NaN, an infinity, and
    # the large finite negative float whose bit pattern is 0xDEADBEEF.
    POISON = (float("nan"), float("-inf"), -6.2598534e18)
    POISON_ALT = (float("inf"), float("nan"), -1.0e30)

    def _poison_floats(self, shape: tuple[int, ...], variant: int) -> torch.Tensor:
        table = self.POISON if variant == 0 else self.POISON_ALT
        count = 1
        for dim in shape:
            count *= dim
        values = [table[i % len(table)] for i in range(count)]
        return torch.tensor(values, dtype=torch.float64).reshape(shape)

    def _poison_bytes(self, shape: tuple[int, ...], variant: int) -> torch.Tensor:
        # 0x7F and 0x80, the int8 endpoints, swapped between variants.
        count = 1
        for dim in shape:
            count *= dim
        first, second = (127, -128) if variant == 0 else (-128, 127)
        values = [first if i % 2 == 0 else second for i in range(count)]
        return torch.tensor(values, dtype=torch.int8).reshape(shape)

    def test_dequantize_payload_is_independent_of_neighbouring_rows(self):
        # The live rows are handed to the codec inside a buffer sized for a full
        # batch, with the padding poisoned differently on each pass. Identical
        # payload output across both passes means the padding was never read.
        #
        # batch_rows is fixed at the kernel's full TPipe pool rather than swept:
        # what the property needs is that padding exists at all, and a second
        # pool size only repeats the same 1-live-row and 3-live-row cases.
        batch_rows = 8
        for head_size in (64, 128, 256):
            for rows in (1, 3):
                packed_stride = head_size // TURBOQUANT_PACK_FACTOR
                live = ((torch.arange(rows * packed_stride) % 256) - 128).to(torch.int8)
                live = live.reshape(rows, packed_stride)
                isolated = _dequantize_levels(live)

                payloads = []
                for variant in (0, 1):
                    buffer = self._poison_bytes((batch_rows, packed_stride), variant)
                    buffer[:rows] = live
                    expanded = _dequantize_levels(buffer)
                    payload = expanded[:rows]
                    self.assertTrue(
                        torch.isfinite(payload).all(),
                        f"D={head_size} rows={rows} batch_rows={batch_rows}",
                    )
                    torch.testing.assert_close(payload, isolated, atol=0, rtol=0)
                    payloads.append(payload.clone())
                torch.testing.assert_close(payloads[0], payloads[1], atol=0, rtol=0)

    def test_dequantize_rows_are_independent(self):
        # Row r of the output must depend only on row r of the input. A shared
        # scratch buffer that is not reset between rows breaks exactly this.
        head_size, rows = 128, 4
        packed_stride = head_size // TURBOQUANT_PACK_FACTOR
        packed = ((torch.arange(rows * packed_stride) % 256) - 128).to(torch.int8)
        packed = packed.reshape(rows, packed_stride)

        batched = _dequantize_levels(packed)
        for row in range(rows):
            single = _dequantize_levels(packed[row : row + 1])
            torch.testing.assert_close(batched[row : row + 1], single, atol=0, rtol=0)

    def test_writing_a_prefix_leaves_the_padding_untouched(self):
        # Whatever the codec writes must stop at the live rows; the padding the
        # kernel never initialised has to come back bit for bit.
        head_size, batch_rows, rows = 128, 8, 3
        packed_stride = head_size // TURBOQUANT_PACK_FACTOR
        packed = ((torch.arange(rows * packed_stride) % 256) - 128).to(torch.int8)
        packed = packed.reshape(rows, packed_stride)

        out = self._poison_floats((batch_rows, head_size), 0)
        before = out.clone()
        out[:rows] = _dequantize_levels(packed)

        tail_after = out[rows:].view(torch.int64)
        tail_before = before[rows:].view(torch.int64)
        torch.testing.assert_close(tail_after, tail_before, atol=0, rtol=0)

    def test_quantize_ignores_a_poisoned_destination(self):
        # A destination pre-filled with the int8 endpoints must be fully
        # overwritten within the payload and untouched outside it.
        head_size, batch_rows = 128, 4
        packed_stride = head_size // TURBOQUANT_PACK_FACTOR
        torch.manual_seed(5)
        vec = torch.randn(head_size, dtype=torch.float64) * 2.5
        reference, reference_step = _quantize(vec)

        for variant in (0, 1):
            buffer = self._poison_bytes((batch_rows, packed_stride), variant)
            tail_before = buffer[1:].clone()
            packed, step = _quantize(vec)
            buffer[0] = packed

            torch.testing.assert_close(buffer[0], reference, atol=0, rtol=0)
            self.assertAlmostEqual(step.item(), reference_step.item(), delta=0.0)
            torch.testing.assert_close(buffer[1:], tail_before, atol=0, rtol=0)

    def test_apply_pi_ignores_surrounding_memory(self):
        for head_size in (64, 128, 256):
            signs = turboquant_pi_signs(head_size, CPU).to(torch.float64)
            torch.manual_seed(3)
            clean = torch.randn(head_size, dtype=torch.float64)
            expected = apply_pi(clean, signs)

            for variant in (0, 1):
                arena = self._poison_floats((4, head_size), variant)
                arena[0] = clean
                rotated = apply_pi(arena[0], signs)

                self.assertTrue(torch.isfinite(rotated).all(), f"D={head_size} variant={variant}")
                torch.testing.assert_close(rotated, expected, atol=0, rtol=0)

    def test_quantize_scale_is_per_row_not_shared(self):
        # A leaked scale across rows would show up as an RMS that is right for
        # one row and wrong for the others. A constant-magnitude row has
        # RMS == that magnitude, which is what makes the expectation exact.
        head_size = 64
        amplitudes = (1.0, 16.0, 256.0)
        vectors = torch.stack([torch.full((head_size,), amplitude, dtype=torch.float64) for amplitude in amplitudes])
        _, steps = _quantize(vectors)
        self.assertEqual(tuple(steps.shape), (len(amplitudes),))
        for row, amplitude in enumerate(amplitudes):
            self.assertAlmostEqual(steps[row].item(), amplitude, delta=amplitude * 1e-9)


class TestCodecTables(TestBase):
    """The codec's constant tables are built here and copied into UB verbatim.

    Nothing on the device regenerates or even touches them, so the image this
    produces *is* the contract; every field is checked against the formula the
    kernel assumes.
    """

    def _split(self, head_size: int, batch_rows: int):
        tables = turboquant_codec_tables(head_size, batch_rows, CPU)
        batch = head_size * batch_rows
        cursor = 0
        signs, xor_offsets = [], []
        for _ in range(3):
            signs.append(tables[cursor : cursor + head_size].view(torch.float32))
            cursor += head_size
            xor_offsets.append(tables[cursor : cursor + head_size])
            cursor += head_size
        half = head_size // TURBOQUANT_PACK_FACTOR
        even = tables[cursor : cursor + half]
        cursor += half
        odd = tables[cursor : cursor + half]
        cursor += half
        expand = tables[cursor : cursor + batch]
        cursor += batch
        odd_select = tables[cursor : cursor + batch].view(torch.float32)
        cursor += batch
        centroids = tables[cursor : cursor + TURBOQUANT_LEVELS].view(torch.float32)
        return signs, xor_offsets, even, odd, expand, odd_select, centroids

    def test_image_length_matches_the_kernel_contract(self):
        for head_size in (64, 128, 256):
            for batch_rows in (1, TURBOQUANT_TILE_ROWS):
                tables = turboquant_codec_tables(head_size, batch_rows, CPU)
                self.assertEqual(tables.dtype, torch.int32)
                self.assertTrue(tables.is_contiguous())
                self.assertEqual(
                    tables.numel(), 7 * head_size + 2 * head_size * batch_rows + TURBOQUANT_LEVELS
                )
                # One DataCopy moves the image, so it must be a whole burst.
                self.assertEqual(tables.numel() % 8, 0)

    def test_sign_and_xor_tables_encode_the_butterfly(self):
        head_size = 128
        signs, xor_offsets, _, _, _, _, _ = self._split(head_size, 1)
        channels = torch.arange(head_size)
        for stage in range(3):
            stride = 1 << stage
            expected_sign = (1 - 2 * ((channels // stride) & 1)).to(torch.float32)
            torch.testing.assert_close(signs[stage], expected_sign, atol=0, rtol=0)
            # Gather consumes byte offsets, so p ^ stride scaled by sizeof(float).
            torch.testing.assert_close(xor_offsets[stage], (4 * (channels ^ stride)).to(torch.int32), atol=0, rtol=0)

    def test_nibble_tables_deinterleave_and_expand(self):
        head_size, batch_rows = 128, TURBOQUANT_TILE_ROWS
        _, _, even, odd, expand, odd_select, _ = self._split(head_size, batch_rows)
        pairs = torch.arange(head_size // TURBOQUANT_PACK_FACTOR)
        torch.testing.assert_close(even, (8 * pairs).to(torch.int32), atol=0, rtol=0)
        torch.testing.assert_close(odd, (8 * pairs + 4).to(torch.int32), atol=0, rtol=0)

        batch = torch.arange(head_size * batch_rows)
        torch.testing.assert_close(expand, (4 * (batch // 2)).to(torch.int32), atol=0, rtol=0)
        torch.testing.assert_close(odd_select, (batch % 2).to(torch.float32), atol=0, rtol=0)

    def test_centroid_table_is_the_lloyd_max_fixed_point(self):
        # The last 16 words are what Dequantize4Bit gathers with; a table that
        # is merely close still decodes, it just stops being the quantiser the
        # thresholds were derived for.
        _, _, _, _, _, _, centroids = self._split(128, TURBOQUANT_TILE_ROWS)
        torch.testing.assert_close(
            centroids, CENTROIDS.to(torch.float32), atol=0, rtol=0
        )
        # Strictly increasing, or the threshold scan's bin index would not
        # select the nearest centroid.
        self.assertTrue(bool((centroids[1:] > centroids[:-1]).all()))
        # Symmetric about zero, as the fixed point for a symmetric density is.
        torch.testing.assert_close(centroids, -centroids.flip(0), atol=0, rtol=1e-7)
        # Every threshold is exactly the midpoint of the centroids it separates.
        midpoints = 0.5 * (CENTROIDS[:-1] + CENTROIDS[1:])
        torch.testing.assert_close(THRESHOLDS, midpoints, atol=1e-7, rtol=0)

    def test_tables_are_cached_per_shape(self):
        first = turboquant_codec_tables(128, 1, CPU)
        self.assertIs(turboquant_codec_tables(128, 1, CPU), first)
        self.assertIsNot(turboquant_codec_tables(128, TURBOQUANT_TILE_ROWS, CPU), first)


class TestBackendContract(TestBase):
    def test_kv_cache_shape_is_packed_nd(self):
        shape = AscendTurboQuantAttentionBackend.get_kv_cache_shape(7, 128, 2, HEAD_SIZE)
        self.assertEqual(shape, (2, 7, 128, 2, HEAD_SIZE // TURBOQUANT_PACK_FACTOR))

    def test_backend_identity(self):
        self.assertEqual(AscendTurboQuantAttentionBackend.get_name(), "ASCEND_TURBOQUANT")
        self.assertIs(AscendTurboQuantAttentionBackend.get_impl_cls(), AscendTurboQuantAttentionBackendImpl)
        self.assertFalse(AscendTurboQuantAttentionBackend.supports_pcp())

    @staticmethod
    def _layer(hf_config, layer_name="model.layers.3.self_attn.attn"):
        layer = MagicMock()
        layer.layer_name = layer_name
        layer.impl = MagicMock()
        layer.impl.__class__ = MagicMock
        layer.impl.head_size = HEAD_SIZE
        layer.impl.vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))
        return layer

    def test_activation_swaps_the_impl(self):
        layer = self._layer(SimpleNamespace())
        tq_module.activate_turboquant_backend(layer)
        self.assertIs(layer.impl.__class__, AscendTurboQuantAttentionBackendImpl)
        self.assertEqual(layer.kv_cache_torch_dtype, torch.int8)
        # A swapped-in impl starts with no scratch of its own: whatever the
        # previous class left behind was sized for a different cache layout.
        self.assertIsNone(layer.impl.decode_workspace)
        self.assertEqual(layer.impl._workspace_floats, {})
        # __init__ does not run on a class swap. Every attribute it sets has to
        # be set here too, or the first decode is an AttributeError.
        self.assertIsNone(layer.impl.rotated_query)
        self.assertIsNone(layer.impl._hadamard16)
        # No marker: the checkpoint's o_proj is as trained, so the decode must
        # un-rotate its own output.
        self.assertFalse(layer.impl.output_rotation_folded)

    def test_activation_reads_the_fold_record_per_layer(self):
        marker = output_rotation_marker(["model.layers.3.self_attn"], HEAD_SIZE)
        hf_config = SimpleNamespace(**{TURBOQUANT_OUTPUT_ROTATION_CONFIG_KEY: marker})

        folded = self._layer(hf_config)
        tq_module.activate_turboquant_backend(folded)
        self.assertTrue(folded.impl.output_rotation_folded)

        # A marked checkpoint that does not name this layer is refused, not run
        # unfolded: either answer could be the wrong one, and both are silent.
        with self.assertRaises(ValueError):
            tq_module.activate_turboquant_backend(self._layer(hf_config, "model.layers.4.self_attn.attn"))

    def test_loading_primes_the_device_registry(self):
        """The driver query happens at load time, never on a decode step."""
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        ops = _ops_mock()
        with (
            patch.object(torch.ops, "_C_ascend", ops, create=True),
            patch.object(tq_module.AscendAttentionBackendImpl, "process_weights_after_loading"),
        ):
            impl.process_weights_after_loading(torch.float16)

        ops.npu_turboquant_vector_core_num.assert_called_once_with()

    # The scale plane's own arithmetic and allocation are not retested here: the
    # slot table lives in test_turboquant_page_budget.TestTurboQuantScaleSlot and
    # _ensure_scale_cache in test_turboquant_scale_plane.TestScalePlaneHandover,
    # which is the module that owns the hand-over.

    def _prefill(self, folded: bool):
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        impl.head_size = HEAD_SIZE
        impl.num_heads = 4
        impl.num_kv_heads = 2
        impl.scale = HEAD_SIZE**-0.5
        impl._pi_signs = None
        impl._hadamard16 = None
        impl.output_rotation_folded = folded
        num_tokens = 5
        query = torch.randn(num_tokens, impl.num_heads, HEAD_SIZE)
        key = torch.randn(num_tokens, impl.num_kv_heads, HEAD_SIZE)
        value = torch.randn(num_tokens, impl.num_kv_heads, HEAD_SIZE)
        output = torch.zeros(num_tokens, impl.num_heads, HEAD_SIZE)
        metadata = MagicMock(actual_seq_lengths_q=[num_tokens], attn_mask=None)

        ops = _numeric_ops(torch.zeros(0))
        fia = MagicMock(return_value=(torch.zeros(num_tokens, impl.num_heads, HEAD_SIZE), None))
        with (
            patch.object(torch.ops, "_C_ascend", ops, create=True),
            patch.object(tq_module.torch_npu, "npu_fused_infer_attention_score", fia, create=True),
        ):
            impl._forward_prefill_no_cache(query, key, value, metadata, output)
        return value, ops, fia.call_args.kwargs

    def test_folded_prefill_rotates_value_before_attention(self):
        """softmax(q k^T) (V Pi) = O Pi, which is what a folded o_proj expects."""
        value, ops, kwargs = self._prefill(folded=True)
        ops.npu_turboquant_rotate_q.assert_called_once()
        torch.testing.assert_close(kwargs["value"], apply_pi(value, turboquant_pi_signs(HEAD_SIZE, CPU)))
        self.assertEqual(kwargs["value"].dtype, value.dtype)

    def test_unfolded_prefill_is_untouched(self):
        value, ops, kwargs = self._prefill(folded=False)
        ops.npu_turboquant_rotate_q.assert_not_called()
        torch.testing.assert_close(kwargs["value"], value)

    def test_unsupported_states_are_rejected(self):
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        metadata = MagicMock(attn_state=AscendAttentionState.ChunkedPrefill)
        with self.assertRaises(NotImplementedError):
            impl.forward_impl(None, None, None, (), metadata, None)


CUBE_WORKSPACE_FLOATS = 2048


def _cube_ops(stage_output: torch.Tensor | None = None) -> MagicMock:
    """Operators with the Cube path registered; the decode writes ``stage_output`` if given."""
    ops = _ops_mock()
    ops.npu_turboquant_cube_workspace_size.return_value = CUBE_WORKSPACE_FLOATS
    if stage_output is not None:

        def cube_decode(*args):
            args[-1].copy_(stage_output)

        ops.npu_turboquant_cube_decode.side_effect = cube_decode
    return ops


class TestCubeDecode(TestBase):
    """The kv4fp8 Cube decode: one launch per step that rotates the raw query and hands o_proj its basis."""

    def _make_impl(self, folded: bool = False) -> AscendTurboQuantAttentionBackendImpl:
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        impl.head_size = HEAD_SIZE
        impl.num_heads = 8
        impl.num_kv_heads = 2
        impl.scale = HEAD_SIZE**-0.5
        impl.attn_type = "decoder"
        impl.kv_sharing_target_layer_name = None
        impl.is_kv_producer = False
        impl.key_cache = torch.zeros(1, DECODE_BLOCK_SIZE, 2, HEAD_SIZE // TURBOQUANT_PACK_FACTOR, dtype=torch.int8)
        impl.value_cache = torch.zeros_like(impl.key_cache)
        impl.scale_cache = torch.zeros(1, DECODE_BLOCK_SIZE, turboquant_scale_slot(2))
        impl._pi_signs = None
        impl.decode_workspace = None
        impl._workspace_floats = {}
        impl.rotated_query = None
        impl._hadamard16 = None
        impl.output_rotation_folded = folded
        impl.cube_decode = True
        return impl

    def _decode(self, impl, ops, num_tokens=3, gate=None):
        query = torch.randn(num_tokens, impl.num_heads, HEAD_SIZE)
        output = torch.zeros(num_tokens, impl.num_heads, HEAD_SIZE)
        metadata = MagicMock(
            attn_state=AscendAttentionState.DecodeOnly,
            block_tables=torch.zeros(num_tokens, 2, dtype=torch.int64),
            seq_lens=torch.ones(num_tokens, dtype=torch.int64),
        )
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            result = impl.forward_impl(query, None, None, (), metadata, output, gate)
        return query, output, result

    def test_a_decode_step_is_one_launch_over_the_raw_query(self):
        impl = self._make_impl()
        ops = _cube_ops()
        query, output, _ = self._decode(impl, ops)

        ops.npu_turboquant_cube_decode.assert_called_once()
        ops.npu_turboquant_rotate_q.assert_not_called()
        ops.npu_turboquant_paged_attention.assert_not_called()
        args = ops.npu_turboquant_cube_decode.call_args.args
        # query, gate, pi_signs, codec_tables, hadamard16, k_cache, v_cache, scale_cache, block_tables,
        # context_lens, workspace, query_rot, num_kv_heads, num_heads, scale, output_stage, out.
        self.assertEqual(len(args), 17)
        # The raw query, not a rotation of it: the launch rotates it itself.
        torch.testing.assert_close(args[0], query)
        self.assertIsNone(args[1])
        torch.testing.assert_close(args[2], turboquant_pi_signs(HEAD_SIZE, CPU))
        torch.testing.assert_close(args[3], turboquant_codec_tables(HEAD_SIZE, 1, CPU))
        torch.testing.assert_close(args[4], turboquant_hadamard16(CPU))
        self.assertIs(args[5], impl.key_cache)
        self.assertIs(args[6], impl.value_cache)
        self.assertEqual(args[8].dtype, torch.int32)
        self.assertEqual(args[9].dtype, torch.int32)
        self.assertIs(args[10], impl.decode_workspace)
        # The in-launch rotation lands in the same persistent buffer the separate rotation used.
        self.assertEqual(args[11].dtype, torch.float32)
        self.assertEqual(args[11].data_ptr(), impl.rotated_query.data_ptr())
        self.assertEqual(args[12], impl.num_kv_heads)
        self.assertEqual(args[13], impl.num_heads)
        self.assertEqual(args[15], int(tq_module.TurboQuantOutputStage.UNROTATED))
        self.assertEqual(args[16].data_ptr(), output.data_ptr())

    def test_the_workspace_is_sized_by_the_cube_planner(self):
        impl = self._make_impl()
        ops = _cube_ops()
        self._decode(impl, ops, num_tokens=4)
        ops.npu_turboquant_workspace_size.assert_not_called()
        ops.npu_turboquant_cube_workspace_size.assert_called_once_with(
            4, impl.num_heads, impl.num_kv_heads, impl.head_size, 2, DECODE_BLOCK_SIZE
        )
        self.assertEqual(impl.decode_workspace.numel(), CUBE_WORKSPACE_FLOATS)

    def test_a_folded_layer_keeps_the_rotated_basis(self):
        impl = self._make_impl(folded=True)
        ops = _cube_ops()
        self._decode(impl, ops)
        self.assertEqual(
            ops.npu_turboquant_cube_decode.call_args.args[15], int(tq_module.TurboQuantOutputStage.ROTATED_BASIS)
        )
        self.assertFalse(impl.fuses_output_gate)

    def test_a_gate_is_applied_in_the_launch_and_nowhere_else(self):
        impl = self._make_impl()
        written = torch.randn(2, impl.num_heads, HEAD_SIZE)
        gate = torch.randn(2, impl.num_heads, HEAD_SIZE)
        ops = _cube_ops(stage_output=written)
        _, output, result = self._decode(impl, ops, num_tokens=2, gate=gate)

        args = ops.npu_turboquant_cube_decode.call_args.args
        self.assertEqual(args[15], int(tq_module.TurboQuantOutputStage.GATED))
        torch.testing.assert_close(args[1], gate)
        # What the launch wrote is what o_proj reads: the host neither re-rotates nor re-gates it.
        torch.testing.assert_close(output, written)
        self.assertIs(result, output)
        self.assertTrue(impl.fuses_output_gate)

    def test_a_folded_layer_refuses_a_gate(self):
        impl = self._make_impl(folded=True)
        with self.assertRaises(ValueError):
            self._decode(impl, _cube_ops(), gate=torch.randn(3, impl.num_heads, HEAD_SIZE))

    def test_the_writer_follows_the_decode(self):
        impl = self._make_impl()
        impl.key_cache = None
        key = torch.randn(4, impl.num_kv_heads, HEAD_SIZE)
        cache = torch.zeros(
            3, DECODE_BLOCK_SIZE, impl.num_kv_heads, HEAD_SIZE // TURBOQUANT_PACK_FACTOR, dtype=torch.int8
        )
        metadata = MagicMock(num_actual_tokens=4, slot_mapping=torch.arange(4, dtype=torch.int64))
        ops = _cube_ops()
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl.reshape_and_cache(None, key, torch.randn_like(key), (cache, cache), metadata, None)

        ops.npu_turboquant_reshape_and_cache.assert_not_called()
        ops.npu_turboquant_cube_reshape_and_cache.assert_called_once()
        args = ops.npu_turboquant_cube_reshape_and_cache.call_args.args
        self.assertEqual(len(args), 8)
        torch.testing.assert_close(args[0], key)
        torch.testing.assert_close(args[7], turboquant_codec_tables(HEAD_SIZE, 1, CPU))

    def test_a_block_that_is_not_whole_cube_tiles_is_refused(self):
        impl = self._make_impl()
        impl.scale_cache = None
        cache = torch.zeros(3, 16, 2, HEAD_SIZE // TURBOQUANT_PACK_FACTOR, dtype=torch.int8)
        with self.assertRaises(ValueError):
            impl._ensure_scale_cache((cache, cache))

    def test_the_aiv_decode_applies_a_handed_gate_in_torch(self):
        impl = self._make_impl()
        impl.cube_decode = False
        rotated = torch.randn(2, impl.num_heads, HEAD_SIZE)
        gate = torch.randn(2, impl.num_heads, HEAD_SIZE)
        _, output, _ = self._decode(impl, _numeric_ops(rotated), num_tokens=2, gate=gate)
        expected = apply_pi(rotated, turboquant_pi_signs(HEAD_SIZE, CPU)) * torch.sigmoid(gate)
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=0)
        self.assertFalse(impl.fuses_output_gate)

    def test_a_prefill_applies_a_handed_gate_in_torch(self):
        impl = self._make_impl()
        num_tokens = 5
        attended = torch.randn(num_tokens, impl.num_heads, HEAD_SIZE)
        gate = torch.randn(num_tokens, impl.num_heads, HEAD_SIZE)
        output = torch.zeros(num_tokens, impl.num_heads, HEAD_SIZE)
        metadata = MagicMock(
            attn_state=AscendAttentionState.PrefillNoCache, actual_seq_lengths_q=[num_tokens], attn_mask=None
        )
        fia = MagicMock(return_value=(attended, None))
        with (
            patch.object(torch.ops, "_C_ascend", _cube_ops(), create=True),
            patch.object(tq_module.torch_npu, "npu_fused_infer_attention_score", fia, create=True),
        ):
            impl.forward_impl(
                torch.randn(num_tokens, impl.num_heads, HEAD_SIZE),
                torch.randn(num_tokens, 2, HEAD_SIZE),
                torch.randn(num_tokens, 2, HEAD_SIZE),
                (),
                metadata,
                output,
                gate,
            )
        torch.testing.assert_close(output, attended * torch.sigmoid(gate))

    def test_the_cube_path_needs_the_switch_the_build_and_float16(self):
        cube_ops = SimpleNamespace(npu_turboquant_cube_decode=object())
        aiv_ops = SimpleNamespace()
        cases = (
            ("1", cube_ops, torch.float16, True),
            ("0", cube_ops, torch.float16, False),
            ("1", aiv_ops, torch.float16, False),
            ("1", cube_ops, torch.bfloat16, False),
            ("1", cube_ops, None, False),
        )
        for switch, ops, dtype, expected in cases:
            with (
                patch.dict("os.environ", {"VLLM_ASCEND_TURBOQUANT_CUBE_DECODE": switch}),
                patch.object(torch.ops, "_C_ascend", ops, create=True),
            ):
                self.assertEqual(
                    tq_module.turboquant_cube_decode_selected(dtype),
                    expected,
                    f"switch {switch} dtype {dtype} ops {ops}",
                )

    def test_the_output_stages_match_the_kernel_layout(self):
        # turboquant_layout.h: TurboQuantOutputStage.
        self.assertEqual(int(tq_module.TurboQuantOutputStage.ROTATED_BASIS), 0)
        self.assertEqual(int(tq_module.TurboQuantOutputStage.UNROTATED), 1)
        self.assertEqual(int(tq_module.TurboQuantOutputStage.GATED), 2)


META = torch.device("meta")


class TestCubeOpsOnMeta(TestBase):
    """The backend's Cube calls through the real dispatcher against the pinned schemas, on meta tensors only.

    TestCubeDecode replaces ``torch.ops._C_ascend`` with a mock, which accepts any argument list. Here every call
    has to bind to the schema in TURBOQUANT_CUBE_OP_SCHEMAS (pinned to turboquant_torch_ops.h) and pass the meta
    kernels' shape checks. Nothing is computed: the launch's numbers are the camodel's to verify.
    """

    NUM_HEADS = 8
    NUM_KV_HEADS = 2
    NUM_BLOCKS = 3

    def _make_impl(self, folded: bool = False) -> AscendTurboQuantAttentionBackendImpl:
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        impl.head_size = HEAD_SIZE
        impl.num_heads = self.NUM_HEADS
        impl.num_kv_heads = self.NUM_KV_HEADS
        impl.scale = HEAD_SIZE**-0.5
        impl.attn_type = "decoder"
        impl.kv_sharing_target_layer_name = None
        impl.is_kv_producer = False
        cache_shape = (self.NUM_BLOCKS, DECODE_BLOCK_SIZE, self.NUM_KV_HEADS, HEAD_SIZE // TURBOQUANT_PACK_FACTOR)
        impl.key_cache = torch.empty(cache_shape, dtype=torch.int8, device=META)
        impl.value_cache = torch.empty(cache_shape, dtype=torch.int8, device=META)
        impl.scale_cache = torch.empty(
            self.NUM_BLOCKS, DECODE_BLOCK_SIZE, turboquant_scale_slot(self.NUM_KV_HEADS), device=META
        )
        impl._pi_signs = None
        impl.decode_workspace = None
        impl._workspace_floats = {}
        impl.rotated_query = None
        impl._hadamard16 = None
        impl.output_rotation_folded = folded
        impl.cube_decode = True
        return impl

    def _decode(self, impl, num_tokens: int, gate: torch.Tensor | None = None) -> torch.Tensor:
        query = torch.empty(num_tokens, impl.num_heads, HEAD_SIZE, dtype=torch.float16, device=META)
        output = torch.empty_like(query)
        metadata = MagicMock(
            attn_state=AscendAttentionState.DecodeOnly,
            block_tables=torch.empty(num_tokens, 2, dtype=torch.int64, device=META),
            seq_lens=torch.empty(num_tokens, dtype=torch.int64, device=META),
        )
        with turboquant_cube_meta_ops():
            return impl.forward_impl(query, None, None, (), metadata, output, gate)

    def test_a_decode_step_binds_to_the_pinned_schema_in_every_stage(self):
        for folded, gated in ((True, False), (False, False), (False, True)):
            with self.subTest(folded=folded, gated=gated):
                impl = self._make_impl(folded=folded)
                gate = torch.empty(4, impl.num_heads, HEAD_SIZE, dtype=torch.float16, device=META) if gated else None
                result = self._decode(impl, num_tokens=4, gate=gate)
                self.assertEqual(result.shape, (4, impl.num_heads, HEAD_SIZE))
                self.assertEqual(result.device, META)
                self.assertEqual(impl.rotated_query.dtype, torch.float32)

    def test_a_malformed_call_is_refused_without_a_cpu_kernel(self):
        impl = self._make_impl()
        # A cache whose kv head count is not the layer's.
        impl.key_cache = torch.empty(
            self.NUM_BLOCKS, DECODE_BLOCK_SIZE, 3, HEAD_SIZE // TURBOQUANT_PACK_FACTOR, dtype=torch.int8, device=META
        )
        impl.value_cache = impl.key_cache
        with self.assertRaisesRegex(RuntimeError, "cache heads"):
            self._decode(impl, num_tokens=2)

    def test_the_writer_binds_to_the_pinned_schema(self):
        impl = self._make_impl()
        key = torch.empty(4, self.NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=META)
        metadata = MagicMock(num_actual_tokens=4, slot_mapping=torch.empty(4, dtype=torch.int64, device=META))
        with turboquant_cube_meta_ops(), patch.object(tq_module, "notify_kv_cache_written"):
            impl.reshape_and_cache(None, key, torch.empty_like(key), (impl.key_cache, impl.value_cache), metadata, None)

    def test_the_stubs_leave_nothing_behind(self):
        before = {name: hasattr(torch.ops._C_ascend, name) for name in TURBOQUANT_CUBE_OP_SCHEMAS}
        with turboquant_cube_meta_ops():
            self.assertTrue(all(hasattr(torch.ops._C_ascend, name) for name in TURBOQUANT_CUBE_OP_SCHEMAS))
        after = {name: hasattr(torch.ops._C_ascend, name) for name in TURBOQUANT_CUBE_OP_SCHEMAS}
        self.assertEqual(before, after)


class TestOutputProjectionFold(TestBase):
    """W_o' = W_o (I_H (x) Pi): the one weight rewrite TurboQuant does."""

    NUM_HEADS = 8
    HIDDEN = 512

    def setUp(self):
        torch.manual_seed(0)
        self.weight = torch.randn(self.HIDDEN, self.NUM_HEADS * HEAD_SIZE, dtype=torch.float64) * 0.02
        self.pi = apply_pi(torch.eye(HEAD_SIZE, dtype=torch.float64), turboquant_pi_signs(HEAD_SIZE, CPU).double())

    def test_fold_is_the_block_diagonal_right_multiply(self):
        dense = self.weight @ torch.block_diag(*([self.pi] * self.NUM_HEADS))
        torch.testing.assert_close(fold_pi_into_output_projection(self.weight, HEAD_SIZE), dense, atol=1e-12, rtol=0)

    def test_folded_projection_of_the_rotated_output_is_the_original_projection(self):
        signs = turboquant_pi_signs(HEAD_SIZE, CPU).double()
        rotated = torch.randn(16, self.NUM_HEADS, HEAD_SIZE, dtype=torch.float64)
        reference = apply_pi(rotated, signs).reshape(16, -1) @ self.weight.T
        folded = rotated.reshape(16, -1) @ fold_pi_into_output_projection(self.weight, HEAD_SIZE).T
        torch.testing.assert_close(folded, reference, atol=1e-10, rtol=0)

    def test_fold_commutes_with_tensor_parallel_sharding(self):
        """RowParallelLinear shards o_proj's input by whole heads, and the fold is
        block diagonal over heads, so folding before or after the split agrees."""
        whole = fold_pi_into_output_projection(self.weight, HEAD_SIZE)
        for tp in (2, 4, 8):
            shards = [fold_pi_into_output_projection(s, HEAD_SIZE) for s in self.weight.chunk(tp, dim=1)]
            torch.testing.assert_close(torch.cat(shards, dim=1), whole, atol=0, rtol=0)

    def test_stored_dtypes_clear_the_acceptance_bound(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            stored = self.weight.to(dtype)
            folded = fold_pi_into_output_projection(stored, HEAD_SIZE)
            self.assertEqual(folded.dtype, dtype)
            report = validate_output_projection_fold(stored, folded, HEAD_SIZE, num_samples=128)
            self.assertTrue(report.passed(), f"{dtype}: cos_min {report.min_cosine}")
            self.assertGreater(report.min_cosine, TURBOQUANT_FOLD_MIN_COSINE)

    def test_validation_catches_a_wrong_fold(self):
        wrong = self.weight.flip(dims=(1,))
        report = validate_output_projection_fold(self.weight, wrong, HEAD_SIZE)
        self.assertFalse(report.passed())

    def test_an_output_gate_breaks_the_fold(self):
        """Why the offline tool refuses Qwen3-Next and Qwen3.5: an elementwise gate
        between attention and o_proj does not commute with Pi."""
        signs = turboquant_pi_signs(HEAD_SIZE, CPU).double()
        output = torch.randn(self.NUM_HEADS * HEAD_SIZE, dtype=torch.float64)
        gate = torch.sigmoid(torch.randn(self.NUM_HEADS * HEAD_SIZE, dtype=torch.float64))
        rotated = apply_pi(output.view(self.NUM_HEADS, HEAD_SIZE), signs).reshape(-1)
        reference = (output * gate) @ self.weight.T
        folded = (rotated * gate) @ fold_pi_into_output_projection(self.weight, HEAD_SIZE).T
        cosine = torch.nn.functional.cosine_similarity(folded, reference, dim=0)
        self.assertLess(cosine.item(), TURBOQUANT_FOLD_MIN_COSINE)

    def test_fold_refuses_what_it_cannot_rotate_exactly(self):
        with self.assertRaises(ValueError):
            fold_pi_into_output_projection(self.weight.to(torch.int8), HEAD_SIZE)
        with self.assertRaises(ValueError):
            fold_pi_into_output_projection(self.weight[:, :-1], HEAD_SIZE)
        with self.assertRaises(ValueError):
            fold_pi_into_output_projection(self.weight.view(-1), HEAD_SIZE)

    def test_fold_record_is_checked_against_this_build(self):
        name = "model.layers.0.self_attn.attn"
        marker = output_rotation_marker(["model.layers.0.self_attn"], HEAD_SIZE)
        self.assertEqual(marker["pi_seed"], TURBOQUANT_PI_SEED)
        self.assertTrue(output_rotation_is_folded(SimpleNamespace(**{TURBOQUANT_OUTPUT_ROTATION_CONFIG_KEY: marker}),
                                                  name, HEAD_SIZE))
        self.assertFalse(output_rotation_is_folded(SimpleNamespace(), name, HEAD_SIZE))
        for field, value in (("pi_seed", TURBOQUANT_PI_SEED + 1), ("head_size", 2 * HEAD_SIZE), ("format_version", 0)):
            broken = dict(marker, **{field: value})
            with self.assertRaises(ValueError, msg=field):
                output_rotation_is_folded(SimpleNamespace(**{TURBOQUANT_OUTPUT_ROTATION_CONFIG_KEY: broken}),
                                          name, HEAD_SIZE)


class TestDiagnosticCapture(TestBase):
    """The spy: what it records, what it costs when it is off, and what a dry run skips.

    An Ascend launch is asynchronous, so the operator that faults is not the one the
    runtime names, and the engine dies before anything has written down what it was
    handed. These pin the two halves of the answer to that: a record that reaches disk
    before the launch it describes, and a mode that skips every launch so a whole run's
    configuration can be collected in one pass.
    """

    NUM_BLOCKS = 3

    def setUp(self):
        super().setUp()
        self._trace_dir = tempfile.TemporaryDirectory()
        self.trace_path = os.path.join(self._trace_dir.name, "turboquant_trace.log")
        reset_turboquant_tracer()
        self.addCleanup(self._trace_dir.cleanup)
        self.addCleanup(reset_turboquant_tracer)

    def _tracing(self, capture: str = "1", dry_run: str = "0"):
        """Rebind the tracer to this test's temp file. The variables are read once, when
        the tracer is built, so the reset is what makes them take effect."""
        reset_turboquant_tracer()
        return patch.dict(
            os.environ,
            {
                "TURBOQUANT_CAPTURE_SIGNATURES": capture,
                "TURBOQUANT_DRY_RUN": dry_run,
                "TURBOQUANT_TRACE_PATH": self.trace_path,
            },
        )

    def _records(self, event: str | None = None) -> list:
        turboquant_tracer().close()
        if not os.path.exists(self.trace_path):
            return []
        with open(self.trace_path, encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        return [r for r in records if event is None or r["event"] == event]

    def _make_impl(self, cube: bool = False) -> AscendTurboQuantAttentionBackendImpl:
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        impl.head_size = HEAD_SIZE
        impl.num_heads = 8
        impl.num_kv_heads = 2
        impl.scale = HEAD_SIZE**-0.5
        impl.attn_type = "decoder"
        impl.kv_sharing_target_layer_name = None
        impl.is_kv_producer = False
        cache_shape = (self.NUM_BLOCKS, DECODE_BLOCK_SIZE, 2, HEAD_SIZE // TURBOQUANT_PACK_FACTOR)
        impl.key_cache = torch.zeros(cache_shape, dtype=torch.int8)
        impl.value_cache = torch.zeros(cache_shape, dtype=torch.int8)
        impl.scale_cache = torch.zeros(self.NUM_BLOCKS, DECODE_BLOCK_SIZE, turboquant_scale_slot(2))
        impl._pi_signs = None
        impl.decode_workspace = None
        impl._workspace_floats = {}
        impl.rotated_query = None
        impl._hadamard16 = None
        impl.output_rotation_folded = False
        impl.cube_decode = cube
        impl.sliding_window = None
        return impl

    def _decode_metadata(self, num_tokens: int) -> MagicMock:
        return MagicMock(
            attn_state=AscendAttentionState.DecodeOnly,
            num_actual_tokens=num_tokens,
            block_tables=torch.zeros(num_tokens, 2, dtype=torch.int64),
            seq_lens=torch.ones(num_tokens, dtype=torch.int64),
            slot_mapping=torch.arange(num_tokens, dtype=torch.int64),
        )

    def _write_cache(self, impl, ops, slots: torch.Tensor):
        num_tokens = slots.numel()
        key = torch.randn(num_tokens, impl.num_kv_heads, HEAD_SIZE)
        value = torch.randn_like(key)
        metadata = MagicMock(num_actual_tokens=num_tokens, slot_mapping=slots)
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl.reshape_and_cache(None, key, value, (impl.key_cache, impl.value_cache), metadata, None)

    # -- off by default -------------------------------------------------------

    def test_capture_is_off_by_default_and_writes_nothing(self):
        """The cost of a disabled tracer is one attribute read, and no file exists."""
        with patch.dict(os.environ, {"TURBOQUANT_TRACE_PATH": self.trace_path}, clear=False):
            for name in ("TURBOQUANT_CAPTURE_SIGNATURES", "TURBOQUANT_DRY_RUN"):
                os.environ.pop(name, None)
            reset_turboquant_tracer()
            tracer = turboquant_tracer()
            self.assertFalse(tracer.capture)
            self.assertFalse(tracer.dry_run)
            self.assertFalse(tracer.active)

            impl = self._make_impl()
            ops = _ops_mock()
            self._write_cache(impl, ops, torch.arange(4, dtype=torch.int64))

        ops.npu_turboquant_reshape_and_cache.assert_called_once()
        self.assertFalse(os.path.exists(self.trace_path))

    def test_a_disabled_tracer_never_touches_the_index_tensors(self):
        """The slot summary is a device-to-host copy; nothing may take it when off."""
        with self._tracing(capture="0", dry_run="0"):
            impl = self._make_impl()
            slots = MagicMock(wraps=torch.arange(4, dtype=torch.int64))
            slots.to.return_value = torch.arange(4, dtype=torch.int32)
            metadata = MagicMock(num_actual_tokens=4, slot_mapping=slots)
            key = torch.randn(4, impl.num_kv_heads, HEAD_SIZE)
            ops = _ops_mock()
            with patch.object(torch.ops, "_C_ascend", ops, create=True):
                impl.reshape_and_cache(None, key, key.clone(), (impl.key_cache, impl.value_cache), metadata, None)
            slots.cpu.assert_not_called()
            slots.min.assert_not_called()

    # -- what a record carries ------------------------------------------------

    def test_a_cache_write_records_every_operand_signature(self):
        impl = self._make_impl()
        with self._tracing():
            ops = _ops_mock()
            self._write_cache(impl, ops, torch.arange(6, dtype=torch.int64))

        ops.npu_turboquant_reshape_and_cache.assert_called_once()
        (record,) = self._records("reshape_and_cache")
        self.assertEqual(record["operator"], "npu_turboquant_reshape_and_cache")
        # Every operand the writer is handed, by the parameter name it arrives under.
        for name in ("key", "value", "key_cache", "value_cache", "scale_cache", "slot_mapping", "pi_signs"):
            self.assertIn(name, record["tensors"], name)
        key = record["tensors"]["key"]
        self.assertEqual(key["shape"], [6, impl.num_kv_heads, HEAD_SIZE])
        self.assertEqual(key["stride"], [impl.num_kv_heads * HEAD_SIZE, HEAD_SIZE, 1])
        self.assertEqual(key["dtype"], "torch.float32")
        self.assertTrue(key["contiguous"])
        self.assertTrue(key["data_ptr"].startswith("0x"))
        # The alignment of the pointer, not just the pointer: an operand can arrive
        # contiguous and still start part-way through a 32-byte burst.
        self.assertEqual(sorted(key["align"]), sorted(str(n) for n in TURBOQUANT_TRACE_ALIGNMENTS))
        self.assertEqual(key["align"]["32"], int(key["data_ptr"], 16) % 32)

    def test_a_cache_write_records_its_scalar_launch_arguments(self):
        impl = self._make_impl()
        with self._tracing():
            self._write_cache(impl, _ops_mock(), torch.arange(6, dtype=torch.int64))

        (record,) = self._records("reshape_and_cache")
        args = record["launch_args"]
        self.assertEqual(args["num_tokens"], 6)
        self.assertEqual(args["num_scheduled_tokens"], 6)
        self.assertEqual(args["num_kv_heads"], impl.num_kv_heads)
        self.assertEqual(args["head_dim"], HEAD_SIZE)
        self.assertEqual(args["num_blocks"], self.NUM_BLOCKS)
        self.assertEqual(args["block_size"], DECODE_BLOCK_SIZE)
        # numBlocks * blockSize: the row capacity a slot is dropped for exceeding.
        self.assertEqual(args["capacity"], self.NUM_BLOCKS * DECODE_BLOCK_SIZE)

    def test_a_cache_write_records_the_slots_that_would_be_dropped(self):
        """Both ends of the slot range, because the writer drops both silently: a
        negative slot is vLLM's padding marker, and one past the last row would
        otherwise scribble on whatever else owns that address."""
        impl = self._make_impl()
        capacity = self.NUM_BLOCKS * DECODE_BLOCK_SIZE
        slots = torch.tensor([-1, -1, 0, 5, capacity, capacity + 1], dtype=torch.int64)
        with self._tracing():
            self._write_cache(impl, _ops_mock(), slots)

        (record,) = self._records("reshape_and_cache")
        summary = record["slot_mapping"]
        self.assertEqual(summary["count"], 6)
        self.assertEqual(summary["min"], -1)
        self.assertEqual(summary["max"], capacity + 1)
        self.assertEqual(summary["negative"], 2)
        self.assertEqual(summary["past_capacity"], 2)
        self.assertEqual(summary["capacity"], capacity)
        self.assertEqual(summary["first"], slots.tolist())

    def test_a_decode_records_its_page_geometry_and_workspace_plan(self):
        impl = self._make_impl()
        with self._tracing():
            query = torch.randn(3, impl.num_heads, HEAD_SIZE)
            output = torch.zeros_like(query)
            ops = _ops_mock()
            with patch.object(torch.ops, "_C_ascend", ops, create=True):
                impl.forward_paged_attention(query, self._decode_metadata(3), output)

        ops.npu_turboquant_paged_attention.assert_called_once()
        (record,) = self._records("decode_attention")
        self.assertEqual(record["operator"], "npu_turboquant_paged_attention")
        args = record["launch_args"]
        self.assertEqual(args["num_tokens"], 3)
        self.assertEqual(args["num_heads"], impl.num_heads)
        self.assertEqual(args["num_kv_heads"], impl.num_kv_heads)
        self.assertEqual(args["head_dim"], HEAD_SIZE)
        self.assertEqual(args["num_blocks"], self.NUM_BLOCKS)
        self.assertEqual(args["block_size"], DECODE_BLOCK_SIZE)
        self.assertEqual(args["max_blocks_per_seq"], 2)
        # The size the operator itself planned, not a copy of its arithmetic here.
        self.assertEqual(args["workspace_floats"], WORKSPACE_FLOATS)
        # A block table is bounded by the block count, a slot mapping by the row count.
        self.assertEqual(record["block_tables"]["capacity"], self.NUM_BLOCKS)
        self.assertEqual(record["seq_lens"]["capacity"], self.NUM_BLOCKS * DECODE_BLOCK_SIZE)
        self.assertEqual(record["tensors"]["rotated_query"]["dtype"], "torch.float32")

    def test_a_cube_decode_records_its_output_stage_and_gate(self):
        """The Cube decode's stage is the one launch argument that decides whether
        o_proj is handed a rotated or an un-rotated basis, so a trace that omits it
        cannot explain the layer's output."""
        impl = self._make_impl(cube=True)
        gate = torch.randn(3, impl.num_heads, HEAD_SIZE)
        with self._tracing():
            query = torch.randn(3, impl.num_heads, HEAD_SIZE)
            output = torch.zeros_like(query)
            ops = _cube_ops()
            with patch.object(torch.ops, "_C_ascend", ops, create=True):
                impl.forward_cube_decode(query, self._decode_metadata(3), output, gate)

        ops.npu_turboquant_cube_decode.assert_called_once()
        (record,) = self._records("decode_attention")
        self.assertEqual(record["operator"], "npu_turboquant_cube_decode")
        self.assertEqual(record["decode_path"], "cube")
        self.assertEqual(record["launch_args"]["output_stage"], int(tq_module.TurboQuantOutputStage.GATED))
        self.assertEqual(record["tensors"]["output_gate"]["shape"], [3, impl.num_heads, HEAD_SIZE])

    def test_forward_records_the_layer_and_a_per_layer_step_counter(self):
        """A trace is only readable if a record can be attributed to a layer and a
        step: the interesting failures are where layer 0 step 1 is fine and a later
        one is not."""
        impl = self._make_impl()
        layer = SimpleNamespace(layer_name="model.layers.7.self_attn.attn")
        with self._tracing():
            for _ in range(3):
                query = torch.randn(2, impl.num_heads, HEAD_SIZE)
                output = torch.zeros_like(query)
                ops = _ops_mock()
                with patch.object(torch.ops, "_C_ascend", ops, create=True):
                    impl.forward(layer, query, None, None, (), self._decode_metadata(2), output)

        records = self._records("forward")
        self.assertEqual([r["step"] for r in records], [1, 2, 3])
        self.assertEqual({r["layer"] for r in records}, {"model.layers.7.self_attn.attn"})
        self.assertTrue(all(r["is_decode"] and not r["is_prefill"] for r in records))
        self.assertEqual(records[0]["launch_args"]["num_tokens"], 2)

    def test_forward_labels_a_prefill_as_a_prefill(self):
        impl = self._make_impl()
        num_tokens = 4
        metadata = MagicMock(
            attn_state=AscendAttentionState.PrefillNoCache,
            num_actual_tokens=num_tokens,
            slot_mapping=torch.arange(num_tokens, dtype=torch.int64),
            actual_seq_lengths_q=[num_tokens],
            attn_mask=None,
        )
        query = torch.randn(num_tokens, impl.num_heads, HEAD_SIZE)
        key = torch.randn(num_tokens, impl.num_kv_heads, HEAD_SIZE)
        attn_out = torch.zeros(num_tokens, impl.num_heads * HEAD_SIZE)
        with self._tracing(), patch.object(tq_module, "torch_npu", MagicMock()) as npu:
            npu.npu_fused_infer_attention_score.return_value = (attn_out, None)
            output = torch.zeros(num_tokens, impl.num_heads, HEAD_SIZE)
            with patch.object(torch.ops, "_C_ascend", _ops_mock(), create=True):
                impl.forward(SimpleNamespace(layer_name="l0"), query, key, key.clone(), (), metadata, output)

        (record,) = self._records("forward")
        self.assertEqual(record["phase"], "prefill")
        self.assertTrue(record["is_prefill"])
        self.assertFalse(record["is_decode"])

    def test_forward_records_the_profile_run_that_carries_no_metadata(self):
        """The step that sizes the KV cache enters every layer with nothing to do, and
        'this layer was entered and did nothing' is one of the things a trace answers."""
        impl = self._make_impl()
        with self._tracing():
            output = torch.ones(2, impl.num_heads, HEAD_SIZE)
            result = impl.forward(SimpleNamespace(layer_name="l0"), output.clone(), None, None, (), None, output)

        self.assertEqual(self._records("forward")[0]["phase"], "no_metadata")
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_the_profile_runs_empty_tensor_cache_is_described_not_evaluated(self):
        """Regression: ``determine_available_memory`` -> ``profile_run`` -> ``_dummy_run``
        hands the backend a bare empty tensor where a served step hands a tuple of planes.
        ``bool()`` on it raises "Boolean value of Tensor with no values is ambiguous", so a
        trace that asked whether the cache was empty crashed the startup it was capturing."""
        impl = self._make_impl()
        with self._tracing():
            output = torch.ones(2, impl.num_heads, HEAD_SIZE)
            result = impl.forward(
                SimpleNamespace(layer_name="l0"), output.clone(), None, None, torch.tensor([]), None, output
            )

        (record,) = self._records("forward")
        # Described as the one plane it is, rather than refused or silently dropped.
        self.assertEqual(record["launch_args"]["kv_cache_planes"], 1)
        self.assertEqual(record["tensors"]["kv_cache[0]"]["numel"], 0)
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_every_shape_a_kv_cache_arrives_in_is_described(self):
        """A tensor answers neither ``bool()`` nor, when 0-d, ``len()``, so the planes
        are taken by asking what the argument is rather than by assuming."""
        planes = (torch.zeros(2), torch.zeros(3))
        self.assertEqual(cache_planes(None), ())
        self.assertEqual(cache_planes(planes), planes)
        self.assertEqual(cache_planes(list(planes)), planes)
        # The profile run's empty tensor, and a 0-d one, which len() also refuses.
        self.assertEqual(len(cache_planes(torch.tensor([]))), 1)
        self.assertEqual(len(cache_planes(torch.tensor(0.0))), 1)
        # Anything else is described as the single object it is, not raised about.
        self.assertEqual(cache_planes(object.__new__(object)).__len__(), 1)

    # -- dry run --------------------------------------------------------------

    def test_a_dry_run_launches_no_writer_and_still_announces_the_write(self):
        """Skipping the launch is the point; skipping the notification is not. A KV
        connector that never sees the event stops the run before it has traced much."""
        impl = self._make_impl()
        with self._tracing(dry_run="1"):
            ops = _ops_mock()
            with patch.object(tq_module, "notify_kv_cache_written") as notify:
                self._write_cache(impl, ops, torch.arange(4, dtype=torch.int64))

        ops.npu_turboquant_reshape_and_cache.assert_not_called()
        ops.npu_turboquant_cube_reshape_and_cache.assert_not_called()
        notify.assert_called_once()
        self.assertTrue(self._records("reshape_and_cache")[0]["bypassed"])

    def test_a_dry_run_launches_no_decode_and_zeroes_the_output(self):
        """Zeroed rather than left alone: ``output`` is whatever the runner last put
        there, so an untouched buffer would make a dry run look like it computed."""
        for cube in (False, True):
            with self.subTest(cube=cube):
                impl = self._make_impl(cube=cube)
                with self._tracing(dry_run="1"):
                    query = torch.randn(3, impl.num_heads, HEAD_SIZE)
                    output = torch.full((3, impl.num_heads, HEAD_SIZE), 7.0)
                    ops = _cube_ops() if cube else _ops_mock()
                    with patch.object(torch.ops, "_C_ascend", ops, create=True):
                        result = impl.forward_impl(query, None, None, (), self._decode_metadata(3), output)

                ops.npu_turboquant_cube_decode.assert_not_called()
                ops.npu_turboquant_paged_attention.assert_not_called()
                ops.npu_turboquant_rotate_q.assert_not_called()
                torch.testing.assert_close(result, torch.zeros_like(result))

    def test_a_dry_run_completes_a_whole_forward_over_many_layers_and_steps(self):
        """The reason the mode exists: a real run aborts at the first faulting launch,
        so the configuration of every later layer and step is never seen."""
        layers = [SimpleNamespace(layer_name=f"model.layers.{i}.self_attn.attn") for i in range(4)]
        impls = [self._make_impl() for _ in layers]
        with self._tracing(dry_run="1"):
            for step in range(3):
                for layer, impl in zip(layers, impls):
                    num_tokens = 2
                    metadata = self._decode_metadata(num_tokens)
                    query = torch.randn(num_tokens, impl.num_heads, HEAD_SIZE)
                    key = torch.randn(num_tokens, impl.num_kv_heads, HEAD_SIZE)
                    output = torch.zeros(num_tokens, impl.num_heads, HEAD_SIZE)
                    cache = (impl.key_cache, impl.value_cache)
                    ops = _ops_mock()
                    with patch.object(torch.ops, "_C_ascend", ops, create=True):
                        impl.forward(layer, query, key, key.clone(), cache, metadata, output)
                    self.assertEqual(ops.npu_turboquant_reshape_and_cache.call_count, 0, step)
                    self.assertEqual(ops.npu_turboquant_paged_attention.call_count, 0, step)

        self.assertEqual(len(self._records("forward")), 12)
        self.assertEqual(len(self._records("reshape_and_cache")), 12)
        self.assertEqual(len(self._records("decode_attention")), 12)
        # Every layer reached every step, which is what a live run cannot do.
        self.assertEqual({r["step"] for r in self._records("forward")}, {1, 2, 3})
        self.assertEqual(len({r["layer"] for r in self._records("forward")}), 4)

    def test_a_dry_run_skips_the_prefill_rotation_and_keeps_the_dense_attention(self):
        """The V rotation is the only TurboQuant launch a prefill makes; the dense
        attention beside it is a stock operator and stays."""
        impl = self._make_impl()
        impl.output_rotation_folded = True
        value = torch.randn(4, impl.num_kv_heads, HEAD_SIZE)
        with self._tracing(dry_run="1"):
            ops = _ops_mock()
            with patch.object(torch.ops, "_C_ascend", ops, create=True):
                rotated = impl._rotate_value_for_prefill(value)

        ops.npu_turboquant_rotate_q.assert_not_called()
        torch.testing.assert_close(rotated, value)
        self.assertTrue(self._records("prefill_rotate_value")[0]["bypassed"])

    def test_a_live_capture_records_without_bypassing(self):
        """Capture and dry-run are independent: a capture of a live run has to launch."""
        impl = self._make_impl()
        with self._tracing(capture="1", dry_run="0"):
            ops = _ops_mock()
            self._write_cache(impl, ops, torch.arange(4, dtype=torch.int64))

        ops.npu_turboquant_reshape_and_cache.assert_called_once()
        self.assertFalse(self._records("reshape_and_cache")[0]["bypassed"])

    def test_a_dry_run_without_capture_writes_no_file(self):
        impl = self._make_impl()
        with self._tracing(capture="0", dry_run="1"):
            ops = _ops_mock()
            self._write_cache(impl, ops, torch.arange(4, dtype=torch.int64))

        ops.npu_turboquant_reshape_and_cache.assert_not_called()
        self.assertEqual(self._records(), [])

    # -- the diagnostic must never become the failure -------------------------

    def test_the_file_is_line_buffered_so_a_record_survives_an_abort(self):
        """Nothing flushes or closes this file on the path that matters: the process is
        killed by the runtime. The record has to be on disk when it is written."""
        impl = self._make_impl()
        with self._tracing():
            self._write_cache(impl, _ops_mock(), torch.arange(4, dtype=torch.int64))
            # Read it back without closing the tracer -- exactly what a post-mortem does.
            with open(self.trace_path, encoding="utf-8") as handle:
                lines = [line for line in handle if line.strip()]
        self.assertTrue(lines)
        self.assertEqual(json.loads(lines[0])["event"], "reshape_and_cache")

    def test_an_unopenable_path_falls_back_to_stderr_rather_than_raising(self):
        impl = self._make_impl()
        reset_turboquant_tracer()
        unusable = os.path.join(self.trace_path, "no", "such", "directory", "trace.log")
        with patch.dict(
            os.environ,
            {"TURBOQUANT_CAPTURE_SIGNATURES": "1", "TURBOQUANT_DRY_RUN": "0", "TURBOQUANT_TRACE_PATH": unusable},
        ):
            ops = _ops_mock()
            with patch("sys.stderr", new_callable=MagicMock) as stderr:
                self._write_cache(impl, ops, torch.arange(4, dtype=torch.int64))

        # The run continued and the writer still launched.
        ops.npu_turboquant_reshape_and_cache.assert_called_once()
        self.assertTrue(stderr.write.called)

    def test_a_tensor_that_cannot_be_described_is_recorded_not_raised(self):
        broken = MagicMock(spec=torch.Tensor)
        type(broken).shape = property(lambda self: (_ for _ in ()).throw(RuntimeError("no shape")))
        self.assertIn("describe_failed", describe_tensor(broken))
        self.assertIn("describe_failed", describe_indices(broken))
        # And the values the operators legitimately take are described, not refused.
        self.assertIsNone(describe_tensor(None))
        self.assertIsNone(describe_indices(None))
        self.assertIn("not_a_tensor", describe_tensor(3))

    def test_an_index_summary_is_bounded_and_survives_an_empty_tensor(self):
        long_slots = torch.arange(1000, dtype=torch.int64)
        summary = describe_indices(long_slots, capacity=500)
        self.assertEqual(len(summary["first"]), TURBOQUANT_TRACE_PREVIEW)
        self.assertEqual(summary["past_capacity"], 500)
        empty = describe_indices(torch.zeros(0, dtype=torch.int64))
        self.assertEqual(empty["count"], 0)
        self.assertNotIn("min", empty)


class TestDecodeIndexOperandDevice(TestBase):
    """The decode's index operands have to reach the kernel in global memory.

    ``AscendMetadata.seq_lens`` is deliberately a *host* tensor: the metadata builder
    takes it from ``seq_lens_cpu``, or copies it to the host if it has to, because the
    CANN paged attention the stock backend calls reads the lengths host-side to tile
    with.  The TurboQuant kernels read it with ``contextLenGm_.GetValue(token)`` -- a
    scalar read from global memory -- so a host tensor reaches a vector core as an
    invalid address (error 264), reported from whichever later launch the runtime was
    working on.

    The device here is ``meta`` rather than ``npu``: what is under test is that the
    staging happens and where it lands, which needs a second device and no driver.
    """

    NUM_BLOCKS = 2

    def _make_impl(self, device: torch.device) -> AscendTurboQuantAttentionBackendImpl:
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        impl.head_size = HEAD_SIZE
        impl.num_heads = 8
        impl.num_kv_heads = 2
        impl.scale = HEAD_SIZE**-0.5
        impl.attn_type = "decoder"
        impl.kv_sharing_target_layer_name = None
        impl.is_kv_producer = False
        cache_shape = (self.NUM_BLOCKS, DECODE_BLOCK_SIZE, 2, HEAD_SIZE // TURBOQUANT_PACK_FACTOR)
        impl.key_cache = torch.empty(cache_shape, dtype=torch.int8, device=device)
        impl.value_cache = torch.empty(cache_shape, dtype=torch.int8, device=device)
        impl.scale_cache = torch.empty(
            self.NUM_BLOCKS, DECODE_BLOCK_SIZE, turboquant_scale_slot(2), device=device
        )
        impl._pi_signs = None
        impl.decode_workspace = None
        impl._workspace_floats = {}
        impl.rotated_query = None
        impl._hadamard16 = None
        impl._device_index_buffers = None
        impl.output_rotation_folded = False
        impl.cube_decode = False
        return impl

    def test_the_int32_cast_alone_does_not_move_a_host_tensor(self):
        """Why the bug was invisible: the operands already satisfy every other check.
        ``.to(torch.int32).contiguous()`` on a contiguous int32 host tensor returns the
        very same object -- right dtype, right shape, right strides, wrong memory."""
        host = torch.arange(4, dtype=torch.int32)
        self.assertIs(host.to(torch.int32).contiguous(), host)

    def test_a_host_index_operand_is_staged_onto_the_query_device(self):
        impl = self._make_impl(META)
        host_seq_lens = torch.tensor([17, 4, 900], dtype=torch.int32)
        staged = impl._device_index(host_seq_lens, META, "seq_lens")

        self.assertEqual(staged.device, META)
        self.assertEqual(staged.dtype, torch.int32)
        self.assertEqual(staged.shape, host_seq_lens.shape)
        self.assertTrue(staged.is_contiguous())

    def test_an_operand_already_on_the_device_is_handed_over_untouched(self):
        """The path that was always correct must not start paying for a copy."""
        impl = self._make_impl(META)
        resident = torch.empty(4, dtype=torch.int32, device=META)
        self.assertIs(impl._device_index(resident, META, "seq_lens"), resident)
        self.assertIsNone(impl._device_index_buffers)

    def test_the_staging_buffer_is_persistent_and_per_operand(self):
        """Allocation-free in the steady state, for the reason the decode workspace is:
        a graph replays the addresses it was captured with. Two operands are staged per
        decode, so one shared buffer would have them overwrite each other."""
        impl = self._make_impl(META)
        impl._device_index(torch.zeros(4, dtype=torch.int32), META, "seq_lens")
        impl._device_index(torch.zeros(4, 2, dtype=torch.int32), META, "block_tables")
        buffers = dict(impl._device_index_buffers)
        # Keyed by operand *and* dtype, so a buffer cannot be reused under a dtype it
        # was not allocated for.
        lengths = ("seq_lens", tq_module.TURBOQUANT_INDEX_DTYPE)
        pages = ("block_tables", tq_module.TURBOQUANT_INDEX_DTYPE)
        self.assertEqual(sorted(buffers), sorted([pages, lengths]))
        self.assertIsNot(buffers[lengths], buffers[pages])

        # A second step of the same shape reuses both allocations.
        impl._device_index(torch.zeros(4, dtype=torch.int32), META, "seq_lens")
        self.assertIs(impl._device_index_buffers[lengths], buffers[lengths])
        # A narrower step takes a view of the prefix rather than reallocating.
        narrow = impl._device_index(torch.zeros(2, dtype=torch.int32), META, "seq_lens")
        self.assertIs(impl._device_index_buffers[lengths], buffers[lengths])
        self.assertEqual(narrow.shape, (2,))
        # A wider one has to grow.
        impl._device_index(torch.zeros(64, dtype=torch.int32), META, "seq_lens")
        self.assertIsNot(impl._device_index_buffers[lengths], buffers[lengths])

    def test_growing_the_staging_buffer_during_a_capture_is_refused(self):
        impl = self._make_impl(META)
        flag, stream = _capture_in_progress()
        with flag, stream, self.assertRaisesRegex(RuntimeError, "during a graph capture"):
            impl._device_index(torch.zeros(4, dtype=torch.int32), META, "seq_lens")

    def test_the_runners_device_lengths_are_used_without_a_copy(self):
        """The whole point: staging seq_lens was one host-to-device transfer per
        layer per decode step. The runner already holds the same batch's lengths
        as a persistent int32 device buffer, so preferring it costs nothing --
        no copy, no allocation, and the operand handed to the kernel *is* the
        runner's tensor."""
        impl = self._make_impl(META)
        runner_lengths = torch.arange(4, dtype=torch.int32, device=META)
        metadata = SimpleNamespace(
            seq_lens=torch.arange(4, dtype=torch.int32),
            seq_lens_device=runner_lengths,
        )
        staged = impl._decode_seq_lens(metadata, META)
        self.assertIs(staged, runner_lengths)
        # Nothing was staged, so no buffer was ever allocated for it.
        self.assertIsNone(impl._device_index_buffers)

    def test_metadata_without_the_device_lengths_still_stages_the_host_tensor(self):
        """The field is new; a metadata that predates it must still decode."""
        impl = self._make_impl(META)
        host = torch.arange(4, dtype=torch.int32)
        metadata = SimpleNamespace(seq_lens=host, seq_lens_device=None)
        staged = impl._decode_seq_lens(metadata, META)
        self.assertEqual(staged.device.type, META.type)
        self.assertIsNot(staged, host)

        bare = SimpleNamespace(seq_lens=host)
        self.assertEqual(impl._decode_seq_lens(bare, META).device.type, META.type)

    def test_the_device_lengths_are_still_held_to_the_index_contract(self):
        """Routed through _device_index rather than used raw, so an operand that
        is on the device but the wrong dtype is narrowed rather than reinterpreted
        by a kernel reading it as int32."""
        impl = self._make_impl(META)
        wide = torch.arange(4, dtype=torch.int64, device=META)
        staged = impl._decode_seq_lens(SimpleNamespace(seq_lens=None, seq_lens_device=wide), META)
        self.assertEqual(staged.dtype, tq_module.TURBOQUANT_INDEX_DTYPE)

    def test_the_staging_buffers_are_reserved_at_the_configured_ceiling(self):
        """So the capture is never the step that has to grow one.

        vLLM warms each graph size up eagerly before capturing it, which usually
        sizes the buffer first. Reserving the configured maximum on the first
        allocation makes that hold without depending on the warmup order, for two
        buffers that are one int32 per sequence and one block-table row per
        sequence."""
        impl = self._make_impl(META)
        impl.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=64),
            model_config=SimpleNamespace(max_model_len=1024),
            cache_config=SimpleNamespace(block_size=DECODE_BLOCK_SIZE),
        )
        impl._device_index(torch.zeros(2, dtype=torch.int32), META, "seq_lens")
        impl._device_index(torch.zeros(2, 1, dtype=torch.int32), META, "block_tables")
        buffers = impl._device_index_buffers
        lengths = buffers[("seq_lens", tq_module.TURBOQUANT_INDEX_DTYPE)]
        pages = buffers[("block_tables", tq_module.TURBOQUANT_INDEX_DTYPE)]
        self.assertEqual(lengths.numel(), 64)
        # 1024 tokens over 128-token blocks is 8 block-table entries per sequence.
        self.assertEqual(pages.numel(), 64 * 8)

        # And the largest decode the config allows then finds them already big enough.
        flag, stream = _capture_in_progress()
        with flag, stream:
            impl._device_index(torch.zeros(64, dtype=torch.int32), META, "seq_lens")
            impl._device_index(torch.zeros(64, 8, dtype=torch.int32), META, "block_tables")
        self.assertIs(buffers[("seq_lens", tq_module.TURBOQUANT_INDEX_DTYPE)], lengths)
        self.assertIs(buffers[("block_tables", tq_module.TURBOQUANT_INDEX_DTYPE)], pages)

    def test_a_config_that_names_no_limits_reserves_nothing(self):
        """An impl with no engine around it -- the CPU harness, a profile run before
        the config is complete -- sizes by the call in hand rather than guessing."""
        impl = self._make_impl(META)
        self.assertEqual(impl._decode_capacity(), (0, 0))
        impl.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=MagicMock()),
            model_config=SimpleNamespace(max_model_len=1024),
            cache_config=SimpleNamespace(block_size=DECODE_BLOCK_SIZE),
        )
        self.assertEqual(impl._decode_capacity(), (0, 0))
        impl._device_index(torch.zeros(3, dtype=torch.int32), META, "seq_lens")
        self.assertEqual(impl._device_index_buffers[("seq_lens", tq_module.TURBOQUANT_INDEX_DTYPE)].numel(), 3)

    def test_an_unknown_operand_is_sized_by_the_call_that_stages_it(self):
        """Only the two the decode stages have a ceiling the config knows."""
        impl = self._make_impl(META)
        impl.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=64),
            model_config=SimpleNamespace(max_model_len=1024),
            cache_config=SimpleNamespace(block_size=DECODE_BLOCK_SIZE),
        )
        self.assertEqual(impl._reserved_index_words("slot_mapping"), 0)
        impl._device_index(torch.zeros(5, dtype=torch.int32), META, "slot_mapping")
        self.assertEqual(impl._device_index_buffers[("slot_mapping", tq_module.TURBOQUANT_INDEX_DTYPE)].numel(), 5)

    def _decode(self, impl, ops, num_tokens=3, seq_lens=None, block_tables=None):
        query = torch.empty(num_tokens, impl.num_heads, HEAD_SIZE, device=META)
        output = torch.empty_like(query)
        metadata = MagicMock(
            attn_state=AscendAttentionState.DecodeOnly,
            block_tables=(
                block_tables if block_tables is not None else torch.empty(num_tokens, 2, dtype=torch.int32, device=META)
            ),
            seq_lens=seq_lens if seq_lens is not None else torch.ones(num_tokens, dtype=torch.int32),
        )
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl.forward_impl(query, None, None, (), metadata, output)
        return metadata

    def test_the_paged_decode_is_handed_device_resident_lengths(self):
        impl = self._make_impl(META)
        ops = _ops_mock()
        host_seq_lens = torch.tensor([9, 9, 9], dtype=torch.int32)
        self._decode(impl, ops, seq_lens=host_seq_lens)

        # query_rot, k_cache, v_cache, scale_cache, block_tables, context_lens, ...
        context_lens = ops.npu_turboquant_paged_attention.call_args.args[5]
        self.assertEqual(context_lens.device, META)
        self.assertIsNot(context_lens, host_seq_lens)
        block_tables = ops.npu_turboquant_paged_attention.call_args.args[4]
        self.assertEqual(block_tables.device, META)

    def test_the_cube_decode_is_handed_device_resident_lengths(self):
        impl = self._make_impl(META)
        impl.cube_decode = True
        ops = _cube_ops()
        self._decode(impl, ops, seq_lens=torch.tensor([9, 9, 9], dtype=torch.int32))

        # query, gate, pi_signs, codec_tables, hadamard16, k_cache, v_cache,
        # scale_cache, block_tables, context_lens, ...
        args = ops.npu_turboquant_cube_decode.call_args.args
        self.assertEqual(args[9].device, META)
        self.assertEqual(args[8].device, META)

    def test_an_int64_host_length_reaches_the_kernel_as_int32(self):
        """The metadata's seq_lens is int64 on some paths; the kernel reads int32."""
        impl = self._make_impl(META)
        ops = _ops_mock()
        self._decode(impl, ops, seq_lens=torch.ones(3, dtype=torch.int64))

        context_lens = ops.npu_turboquant_paged_attention.call_args.args[5]
        self.assertEqual(context_lens.dtype, torch.int32)
        self.assertEqual(context_lens.device, META)

    def test_the_narrowing_happens_before_the_transfer_not_across_it(self):
        """An int64 operand is narrowed on its own device, so the host-to-device copy
        is always same-dtype and never asks the runtime to convert across the bus."""
        impl = self._make_impl(META)
        host_int64 = torch.tensor([17, 4, 900], dtype=torch.int64)
        copied: list = []

        original = torch.Tensor.copy_

        def record(dst, src, *args, **kwargs):
            copied.append((src.dtype, src.device, dst.dtype, dst.device))
            return original(dst, src, *args, **kwargs)

        with patch.object(torch.Tensor, "copy_", record):
            staged = impl._device_index(host_int64, META, "seq_lens")

        self.assertEqual(len(copied), 1)
        src_dtype, src_device, dst_dtype, dst_device = copied[0]
        # The source was already int32 by the time it crossed, and on the host.
        self.assertEqual(src_dtype, tq_module.TURBOQUANT_INDEX_DTYPE)
        self.assertEqual(src_device.type, "cpu")
        self.assertEqual(dst_dtype, src_dtype)
        self.assertEqual(dst_device, META)
        self.assertEqual(staged.dtype, tq_module.TURBOQUANT_INDEX_DTYPE)

    def test_the_staging_buffer_is_allocated_in_the_operand_dtype(self):
        impl = self._make_impl(META)
        impl._device_index(torch.zeros(4, dtype=torch.int64), META, "seq_lens")
        ((key, buffer),) = impl._device_index_buffers.items()
        # Keyed by dtype as well as operand, so a buffer cannot be reused under a
        # dtype it was not allocated for.
        self.assertEqual(key, ("seq_lens", tq_module.TURBOQUANT_INDEX_DTYPE))
        self.assertEqual(buffer.dtype, tq_module.TURBOQUANT_INDEX_DTYPE)

    def test_a_staging_buffer_of_the_wrong_dtype_is_refused_not_reinterpreted(self):
        """The copy would otherwise reinterpret the bytes the kernel indexes with."""
        impl = self._make_impl(META)
        impl._device_index_buffers = {
            ("seq_lens", tq_module.TURBOQUANT_INDEX_DTYPE): torch.empty(8, dtype=torch.int16, device=META)
        }
        with self.assertRaisesRegex(RuntimeError, "staging buffer is"):
            impl._device_index(torch.zeros(4, dtype=torch.int64), META, "seq_lens")


class TestDecodeFence(TestBase):
    """507035 is ACL_ERROR_RT_VECTOR_CORE_EXCEPTION, not a copy error.

    An Ascend launch is asynchronous, so a decode that traps does not fail its own
    launch. The first call after it that synchronises is the *next* step's staging
    copy in ``_device_index``, which is why the fault comes back from
    ``copy_between_host_and_device_opapi`` -- a host-to-device transfer of a handful
    of int32 lengths, which touches no AI core and cannot have raised it. The fence
    is what moves the report back onto the launch that caused it.
    """

    def _make_impl(self, cube: bool = False) -> AscendTurboQuantAttentionBackendImpl:
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        impl.head_size = HEAD_SIZE
        impl.num_heads = 8
        impl.num_kv_heads = 2
        impl.scale = HEAD_SIZE**-0.5
        impl.key_cache = torch.zeros(1, DECODE_BLOCK_SIZE, 2, HEAD_SIZE // TURBOQUANT_PACK_FACTOR, dtype=torch.int8)
        impl.value_cache = impl.key_cache
        impl.scale_cache = torch.zeros(1, DECODE_BLOCK_SIZE, turboquant_scale_slot(2))
        impl._pi_signs = None
        impl.decode_workspace = None
        impl._workspace_floats = {}
        impl.rotated_query = None
        impl._hadamard16 = None
        impl._device_index_buffers = None
        impl.output_rotation_folded = False
        impl.cube_decode = cube
        return impl

    def _decode(self, impl, ops):
        query = torch.randn(2, impl.num_heads, HEAD_SIZE)
        output = torch.zeros_like(query)
        metadata = MagicMock(
            attn_state=AscendAttentionState.DecodeOnly,
            block_tables=torch.zeros(2, 1, dtype=torch.int32),
            seq_lens=torch.ones(2, dtype=torch.int32),
        )
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl.forward_impl(query, None, None, (), metadata, output)

    def test_the_fence_drains_on_device_and_nowhere_else(self):
        impl = self._make_impl()
        with patch.object(tq_module.torch, "npu", MagicMock(), create=True) as npu:
            impl._fence_decode(torch.device("cpu"))
            npu.synchronize.assert_not_called()
            impl._fence_decode(SimpleNamespace(type="npu"))
            npu.synchronize.assert_called_once()

    def test_both_decodes_are_fenced_only_under_the_diagnostic_flag(self):
        """One synchronisation per layer per step; it rides the same flag the write
        fence does and is never a serving default."""
        for cube in (False, True):
            with self.subTest(cube=cube):
                impl = self._make_impl(cube=cube)
                ops = _cube_ops() if cube else _ops_mock()
                with patch.object(impl, "_fence_decode") as fence:
                    with patch.dict("os.environ", {"VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS": "0"}):
                        self._decode(impl, ops)
                    fence.assert_not_called()
                    with patch.dict("os.environ", {"VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS": "1"}):
                        self._decode(impl, ops)
                    fence.assert_called_once()


class TestFlightRecorder(TestBase):
    """The in-memory ring, its crash dump, and the unique-configuration harvester.

    The ring is the cheap half of the capture layer: it holds the last N dispatches in a
    deque and writes nothing until the process leaves. What it costs per dispatch is a
    handful of host-side attribute reads, which is why the sequence lengths it records are
    taken from the metadata's own host tensor rather than from the staged device copy.
    """

    NUM_BLOCKS = 3

    def setUp(self):
        super().setUp()
        self._dir = tempfile.TemporaryDirectory()
        self.trace_path = os.path.join(self._dir.name, "turboquant_trace.log")
        self.crash_path = os.path.join(self._dir.name, TURBOQUANT_CRASH_FILENAME)
        self.cases_path = os.path.join(self._dir.name, TURBOQUANT_CASES_FILENAME)
        reset_turboquant_tracer()
        self.addCleanup(self._dir.cleanup)
        self.addCleanup(reset_turboquant_tracer)

    def _env(self, **overrides):
        env = {
            "TURBOQUANT_TRACE_PATH": self.trace_path,
            "TURBOQUANT_CAPTURE_SIGNATURES": "0",
            "TURBOQUANT_DRY_RUN": "0",
            "TURBOQUANT_FLIGHT_RECORDER": "0",
            "TURBOQUANT_RECORD_CASES": "0",
        }
        env.update(overrides)
        reset_turboquant_tracer()
        return patch.dict(os.environ, env)

    def _make_impl(self, cube: bool = False) -> AscendTurboQuantAttentionBackendImpl:
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        impl.head_size = HEAD_SIZE
        impl.num_heads = 8
        impl.num_kv_heads = 2
        impl.scale = HEAD_SIZE**-0.5
        impl.attn_type = "decoder"
        impl.kv_sharing_target_layer_name = None
        impl.is_kv_producer = False
        shape = (self.NUM_BLOCKS, DECODE_BLOCK_SIZE, 2, HEAD_SIZE // TURBOQUANT_PACK_FACTOR)
        impl.key_cache = torch.zeros(shape, dtype=torch.int8)
        impl.value_cache = torch.zeros(shape, dtype=torch.int8)
        impl.scale_cache = torch.zeros(self.NUM_BLOCKS, DECODE_BLOCK_SIZE, turboquant_scale_slot(2))
        impl._pi_signs = None
        impl.decode_workspace = None
        impl._workspace_floats = {}
        impl.rotated_query = None
        impl._hadamard16 = None
        impl._device_index_buffers = None
        impl.output_rotation_folded = False
        impl.cube_decode = cube
        return impl

    def _decode(self, impl, ops, seq_len=57, num_tokens=1):
        query = torch.randn(num_tokens, impl.num_heads, HEAD_SIZE)
        output = torch.zeros_like(query)
        metadata = MagicMock(
            attn_state=AscendAttentionState.DecodeOnly,
            block_tables=torch.zeros(num_tokens, 2, dtype=torch.int32),
            seq_lens=torch.full((num_tokens,), seq_len, dtype=torch.int32),
        )
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl.forward_impl(query, None, None, (), metadata, output)

    def _write(self, impl, ops, num_tokens=4):
        key = torch.randn(num_tokens, impl.num_kv_heads, HEAD_SIZE)
        metadata = MagicMock(num_actual_tokens=num_tokens, slot_mapping=torch.arange(num_tokens, dtype=torch.int64))
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl.reshape_and_cache(None, key, key.clone(), (impl.key_cache, impl.value_cache), metadata, None)

    def _ring(self):
        return list(turboquant_tracer()._ring or ())

    # -- off by default -------------------------------------------------------

    def test_the_recorder_is_off_by_default(self):
        with self._env():
            tracer = turboquant_tracer()
            self.assertFalse(tracer.active)
            self.assertFalse(tracer.recording)
            self.assertFalse(tracer.record_cases)
            impl = self._make_impl()
            self._decode(impl, _ops_mock())
            self.assertEqual(self._ring(), [])
        self.assertFalse(os.path.exists(self.crash_path))
        self.assertFalse(os.path.exists(self.cases_path))

    def test_the_ring_never_reads_an_index_tensor_off_the_device(self):
        """A synchronisation per dispatch is exactly what the ring exists not to cost, so a
        device-resident seq_lens is skipped rather than copied back."""
        self.assertIsNone(host_seq_lengths(torch.ones(3, dtype=torch.int32, device=META)))
        self.assertIsNone(host_seq_lengths(None))
        self.assertEqual(host_seq_lengths(torch.tensor([5, 9], dtype=torch.int32)), [5, 9])

    # -- the ring -------------------------------------------------------------

    def test_a_decode_and_a_write_are_both_recorded_under_their_own_names(self):
        with self._env(TURBOQUANT_FLIGHT_RECORDER="1"):
            impl = self._make_impl()
            self._write(impl, _ops_mock())
            self._decode(impl, _ops_mock())
            cube = self._make_impl(cube=True)
            self._decode(cube, _cube_ops())
            self.assertEqual([e["op"] for e in self._ring()], ["reshape_and_cache", "decode_aiv", "decode_cube"])

    def test_a_decode_records_the_ragged_tail_that_decides_the_last_tile(self):
        """seq_len % block_size is the quantity a tail-masking kernel turns on -- the shape
        that produced AI core error 340 -- so it is recorded rather than left to be derived."""
        with self._env(TURBOQUANT_FLIGHT_RECORDER="1"):
            impl = self._make_impl()
            self._decode(impl, _ops_mock(), seq_len=57, num_tokens=1)

        (entry,) = [e for e in self._ring() if e["op"] == "decode_aiv"]
        self.assertEqual(entry["seq_len"], 57)
        self.assertEqual(entry["block_size"], DECODE_BLOCK_SIZE)
        self.assertEqual(entry["seq_len_mod_block"], 57 % DECODE_BLOCK_SIZE)
        self.assertEqual(entry["num_tokens"], 1)
        self.assertEqual(entry["num_heads"], impl.num_heads)
        self.assertEqual(entry["num_kv_heads"], impl.num_kv_heads)
        self.assertEqual(entry["head_dim"], HEAD_SIZE)

    def test_every_recorded_tensor_carries_its_shape_dtype_and_both_alignments(self):
        with self._env(TURBOQUANT_FLIGHT_RECORDER="1"):
            impl = self._make_impl()
            self._decode(impl, _ops_mock())

        (entry,) = [e for e in self._ring() if e["op"] == "decode_aiv"]
        # The ring stores raw snapshots -- shape, stride, dtype, device, pointer, straight off
        # the tensor. Rendering them into JSON is what a dump does, once, at the end.
        query = tq_trace.render_entry(entry)["tensors"]["query"]
        self.assertEqual(query["shape"], [1, impl.num_heads, HEAD_SIZE])
        self.assertEqual(query["stride"], [impl.num_heads * HEAD_SIZE, HEAD_SIZE, 1])
        self.assertEqual(query["dtype"], "float32")
        self.assertIn("a32", query)
        self.assertIn("a512", query)
        self.assertLess(query["a32"], 32)
        self.assertLess(query["a512"], 512)

    def test_the_ring_is_bounded_and_keeps_the_most_recent(self):
        """A flight recorder that grows without bound is a memory leak, not a diagnostic."""
        with self._env(TURBOQUANT_FLIGHT_RECORDER="4"):
            impl = self._make_impl()
            for step in range(9):
                self._decode(impl, _ops_mock(), seq_len=40 + step)
            ring = self._ring()

        self.assertEqual(len(ring), 4)
        self.assertEqual([e["seq_len"] for e in ring], [45, 46, 47, 48])

    def test_the_default_depth_is_the_documented_one(self):
        with self._env(TURBOQUANT_FLIGHT_RECORDER="1"):
            self.assertEqual(turboquant_tracer()._ring.maxlen, TURBOQUANT_RING_CAPACITY)

    # -- the crash dump -------------------------------------------------------

    def test_the_ring_is_dumped_as_json_lines_with_a_reason(self):
        with self._env(TURBOQUANT_FLIGHT_RECORDER="1"):
            impl = self._make_impl()
            self._decode(impl, _ops_mock(), seq_len=57)
            written = turboquant_tracer().dump_ring(reason="RuntimeError: aiv exception")

        self.assertEqual(written, self.crash_path)
        with open(self.crash_path, encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        header, *entries = lines
        self.assertEqual(header["reason"], "RuntimeError: aiv exception")
        self.assertEqual(header["entries"], len(entries))
        self.assertEqual(entries[-1]["seq_len"], 57)

    def test_an_empty_ring_writes_no_crash_file(self):
        """A clean run that never dispatched must not leave a crash report behind."""
        with self._env(TURBOQUANT_FLIGHT_RECORDER="1"):
            self.assertIsNone(turboquant_tracer().dump_ring(reason="process exit"))
        self.assertFalse(os.path.exists(self.crash_path))

    def test_the_hooks_are_installed_only_once_a_dispatch_has_been_seen(self):
        """A worker that never enables the recorder must not have its excepthook rewritten."""
        original = sys.excepthook
        with self._env(TURBOQUANT_FLIGHT_RECORDER="1"):
            tracer = turboquant_tracer()
            self.assertFalse(tracer._hooks_installed)
            self.assertIs(sys.excepthook, original)
            try:
                impl = self._make_impl()
                self._decode(impl, _ops_mock())
                self.assertTrue(tracer._hooks_installed)
                self.assertIsNot(sys.excepthook, original)
            finally:
                sys.excepthook = original

    def test_replacing_the_tracer_gives_the_interpreter_its_hooks_back(self):
        """Regression: every tracer used to leave its atexit hook behind, so a process that
        built several spent its shutdown writing dumps for runs that were long over."""
        original = sys.excepthook
        with self._env(TURBOQUANT_FLIGHT_RECORDER="1"):
            impl = self._make_impl()
            self._decode(impl, _ops_mock())
            tracer = turboquant_tracer()
            self.assertTrue(tracer._hooks_installed)
            tracer.close()
            self.assertFalse(tracer._hooks_installed)
            self.assertIs(sys.excepthook, original)
            # atexit.unregister is idempotent, so a second close is not an error.
            tracer.close()

    def test_a_chained_excepthook_is_left_alone(self):
        """Something that installed itself after us owns the chain; restoring over it would
        silently drop whatever it does."""
        original = sys.excepthook
        with self._env(TURBOQUANT_FLIGHT_RECORDER="1"):
            impl = self._make_impl()
            self._decode(impl, _ops_mock())
            later = MagicMock()
            sys.excepthook = later
            try:
                turboquant_tracer().close()
                self.assertIs(sys.excepthook, later)
            finally:
                sys.excepthook = original

    # -- the case harvester ---------------------------------------------------

    def test_a_repeated_configuration_is_written_once(self):
        with self._env(TURBOQUANT_RECORD_CASES="1"):
            impl = self._make_impl()
            for _ in range(5):
                self._decode(impl, _ops_mock(), seq_len=57)
            turboquant_tracer().close()

        with open(self.cases_path, encoding="utf-8") as handle:
            cases = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["op"], "decode_aiv")
        self.assertEqual(cases[0]["case"], 1)

    def test_a_new_shape_is_a_new_case_but_a_new_step_is_not(self):
        """Keyed on the ragged tail, not the sequence length: a decode's length grows by one
        every step, so keying on it would append a line per step forever."""
        with self._env(TURBOQUANT_RECORD_CASES="1"):
            impl = self._make_impl()
            # Same tail, different lengths -- one case.
            self._decode(impl, _ops_mock(), seq_len=57)
            self._decode(impl, _ops_mock(), seq_len=57 + DECODE_BLOCK_SIZE)
            # A different token count is a different launch shape.
            self._decode(impl, _ops_mock(), seq_len=57, num_tokens=2)
            turboquant_tracer().close()

        with open(self.cases_path, encoding="utf-8") as handle:
            cases = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(cases), 2)
        self.assertEqual([c["num_tokens"] for c in cases], [1, 2])

    def test_a_harvested_case_carries_what_an_offline_replay_needs(self):
        with self._env(TURBOQUANT_RECORD_CASES="1"):
            impl = self._make_impl()
            self._decode(impl, _ops_mock(), seq_len=57)
            turboquant_tracer().close()

        with open(self.cases_path, encoding="utf-8") as handle:
            (case,) = [json.loads(line) for line in handle if line.strip()]
        for field in ("op", "num_tokens", "block_size", "seq_len", "seq_len_mod_block", "num_blocks"):
            self.assertIn(field, case)
        for name in ("query", "block_tables", "seq_lens", "workspace", "output"):
            self.assertIn(name, case["tensors"], name)
        self.assertEqual(case["tensors"]["query"]["shape"], [1, impl.num_heads, HEAD_SIZE])

    def test_the_two_recorders_are_independent(self):
        """Harvesting without the ring writes cases and no crash file; the ring without
        harvesting writes neither until the process leaves."""
        with self._env(TURBOQUANT_RECORD_CASES="1"):
            impl = self._make_impl()
            self._decode(impl, _ops_mock())
            self.assertFalse(turboquant_tracer().recording)
            self.assertEqual(self._ring(), [])
            turboquant_tracer().close()
        self.assertTrue(os.path.exists(self.cases_path))

        with self._env(TURBOQUANT_FLIGHT_RECORDER="1"):
            impl = self._make_impl()
            self._decode(impl, _ops_mock())
            self.assertTrue(self._ring())
        self.assertFalse(os.path.exists(self.crash_path))

    def test_a_recorder_failure_never_reaches_the_caller(self):
        """The diagnostic must never become the failure it exists to describe."""
        with self._env(TURBOQUANT_FLIGHT_RECORDER="1"):
            impl = self._make_impl()
            with (
                patch.object(tq_trace, "snapshot_tensor", side_effect=RuntimeError("boom")),
                patch("sys.stderr", new_callable=MagicMock),
            ):
                self._decode(impl, _ops_mock())
            # The launch still happened; only the record was dropped.
            self.assertEqual(self._ring(), [])

    def test_compact_tensor_handles_what_the_operators_legitimately_pass(self):
        self.assertIsNone(compact_tensor(None))
        self.assertIn("not_a_tensor", compact_tensor(7))
        described = compact_tensor(torch.zeros(2, 3))
        self.assertEqual(described["shape"], [2, 3])
        self.assertEqual(described["dtype"], "float32")
