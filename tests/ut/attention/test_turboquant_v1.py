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

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.attention.turboquant_cpu_ops import TURBOQUANT_CUBE_OP_SCHEMAS, turboquant_cube_meta_ops
from tests.ut.base import TestBase
from vllm_ascend.attention import turboquant_rotation as rotation_module
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
from vllm_ascend.attention.turboquant_v1 import (
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
            with patch.object(tq_module, "_EXTRA_CTX", SimpleNamespace(capturing=True)):
                with self.assertRaisesRegex(RuntimeError, "during a graph capture"):
                    impl._decode_workspace(8, 4, DECODE_BLOCK_SIZE, CPU)

    def test_a_capture_that_needs_no_growth_is_allowed(self):
        impl = self._make_impl()
        ops = _ops_mock()
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            warmed = impl._decode_workspace(2, 1, DECODE_BLOCK_SIZE, CPU)
            with patch.object(tq_module, "_EXTRA_CTX", SimpleNamespace(capturing=True)):
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
        for head_size in (64, 128, 256):
            for batch_rows in (4, 8):
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

    def test_scale_cache_is_indexed_by_token_and_burst_aligned(self):
        # Indexing by token rather than by head is what lets the kernel scatter a
        # token's scales in one aligned burst; the slot is padded to 8 fp32 so
        # every write lands on a 32-byte boundary.
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        impl.scale_cache = None
        cache = torch.zeros(3, 128, 2, HEAD_SIZE // TURBOQUANT_PACK_FACTOR, dtype=torch.int8)
        impl._ensure_scale_cache((cache, cache))
        self.assertEqual(tuple(impl.scale_cache.shape), (3, 128, 8))
        self.assertEqual(impl.scale_cache.dtype, torch.float32)

    def test_scale_slot_is_a_whole_burst(self):
        for num_kv_heads, expected in ((1, 8), (2, 8), (4, 8), (5, 16), (8, 16)):
            slot = turboquant_scale_slot(num_kv_heads)
            self.assertEqual(slot, expected, f"num_kv_heads={num_kv_heads}")
            self.assertGreaterEqual(slot, 2 * num_kv_heads)
            self.assertEqual(slot % 8, 0)

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
