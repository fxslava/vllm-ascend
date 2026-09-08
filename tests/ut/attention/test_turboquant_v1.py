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
operators are called with pure-runtime signatures, and nothing rewrites a
projection weight.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention import turboquant_v1 as tq_module
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.turboquant_v1 import (
    TURBOQUANT_PACK_FACTOR,
    TURBOQUANT_TILE_ROWS,
    AscendTurboQuantAttentionBackend,
    AscendTurboQuantAttentionBackendImpl,
    apply_pi,
    turboquant_codec_tables,
    turboquant_pi_signs,
    turboquant_scale_slot,
    walsh_hadamard,
)

HEAD_SIZE = 128
LEVEL_MAX = 15.0
ZERO_POINT = 7.5
PACK_HIGH = 16.0
INT8_BIAS = 128.0
# TurboQuantCodec<4>::kEps -- the absmax floor that keeps an all-zero vector
# from dividing by zero.
TURBOQUANT_EPS = 1e-20

CPU = torch.device("cpu")

# An arbitrary workspace size for the mocked operator to report. The real number
# depends on the device's vector core count, which is exactly why the backend
# asks the operator for it instead of computing it here.
WORKSPACE_FLOATS = 4096


def _ops_mock(workspace_floats: int = WORKSPACE_FLOATS) -> MagicMock:
    """A stand-in for ``torch.ops._C_ascend``.

    ``npu_turboquant_workspace_size`` has to answer with a real integer: the
    backend sizes its persistent buffer from it, so a bare MagicMock would fail
    the ``int()`` rather than exercise the path under test.
    """
    ops = MagicMock()
    ops.npu_turboquant_workspace_size.return_value = workspace_floats
    return ops

# The first eight channels of the shared LCG sign vector. Hard-coded so this
# test and csrc/tests/reference/turbo_quant_cpu.h pin the same constant from
# both sides: a cache written by one must be readable by the other.
EXPECTED_SIGNS_128 = [1, -1, 1, -1, -1, -1, 1, 1]
EXPECTED_SIGNS_256 = [-1, 1, -1, -1, -1, -1, 1, 1]


def _quantize(vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for TurboQuantCodec<4>::Quantize4Bit.

    The ``kEps`` floor mirrors the codec's: without it an all-zero vector
    divides by zero and every level comes back NaN.
    """
    absmax = vec.abs().amax(dim=-1, keepdim=True) + TURBOQUANT_EPS
    levels = torch.clamp(torch.round(vec * (ZERO_POINT / absmax) + ZERO_POINT), 0.0, LEVEL_MAX)
    packed = levels[..., 0::2] + PACK_HIGH * levels[..., 1::2] - INT8_BIAS
    return packed.round().to(torch.int8), (absmax / ZERO_POINT).squeeze(-1)


def _dequantize_levels(packed: torch.Tensor) -> torch.Tensor:
    """Reference for TurboQuantCodec<4>::Dequantize4Bit (centred levels)."""
    byte = packed.to(torch.float64) + INT8_BIAS
    expanded = byte.repeat_interleave(TURBOQUANT_PACK_FACTOR, dim=-1)
    high = torch.floor(expanded / PACK_HIGH)
    low = expanded - PACK_HIGH * high
    odd = (torch.arange(expanded.shape[-1]) % 2).to(expanded.dtype)
    return low + odd * (high - low) - ZERO_POINT


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
        levels = torch.randint(0, 16, (64, HEAD_SIZE)).to(torch.float64)
        packed = (levels[..., 0::2] + PACK_HIGH * levels[..., 1::2] - INT8_BIAS).round().to(torch.int8)
        self.assertEqual(packed.shape[-1], HEAD_SIZE // TURBOQUANT_PACK_FACTOR)
        torch.testing.assert_close(_dequantize_levels(packed) + ZERO_POINT, levels)

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
        # One Pi on the accumulator is the un-rotation; there is no separate
        # inverse transform anywhere in the pipeline.
        approx = apply_pi(probs @ _dequantize_levels(v_packed), self.signs)

        cosine = torch.nn.functional.cosine_similarity(approx, exact, dim=-1)
        self.assertGreater(cosine.min().item(), 0.98)


class TestPureRuntimeContract(TestBase):
    """The refactor's invariant: rotation happens in the kernel, never in a
    weight, and the operators carry no basis-selection flag."""

    def test_no_weight_folding_helpers_are_exported(self):
        for name in (
            "fold_pi_into_attention_projections",
            "fold_pi_into_qkv_projection",
            "fold_pi_into_out_projection",
            "maybe_trans_nz_with_pi",
        ):
            self.assertFalse(hasattr(tq_module, name), f"{name} must not come back: rotation is runtime-only")

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
        return impl

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

    def test_paged_attention_passes_post_rope_query_and_no_flag(self):
        impl = self._make_impl()
        num_tokens = 2
        impl.key_cache = torch.zeros(1, 128, 2, HEAD_SIZE // TURBOQUANT_PACK_FACTOR, dtype=torch.int8)
        impl.value_cache = impl.key_cache
        impl.scale_cache = torch.zeros(1, 128, turboquant_scale_slot(2))
        query = torch.randn(num_tokens, impl.num_heads, HEAD_SIZE)
        output = torch.zeros(num_tokens, impl.num_heads, HEAD_SIZE)
        metadata = MagicMock(
            block_tables=torch.zeros(num_tokens, 1, dtype=torch.int64),
            seq_lens=torch.ones(num_tokens, dtype=torch.int64),
        )

        ops = _ops_mock()
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl.forward_paged_attention(query, metadata, output)

        ops.npu_turboquant_paged_attention.assert_called_once()
        args = ops.npu_turboquant_paged_attention.call_args.args
        # query, k_cache, v_cache, scale_cache, block_tables, context_lens,
        # pi_signs, codec_tables, workspace, num_kv_heads, num_heads, scale, out.
        self.assertEqual(len(args), 13)
        torch.testing.assert_close(args[0], query)
        torch.testing.assert_close(args[7], turboquant_codec_tables(HEAD_SIZE, TURBOQUANT_TILE_ROWS, CPU))
        self.assertEqual(args[9], impl.num_kv_heads)
        self.assertEqual(args[10], impl.num_heads)
        self.assertFalse(any(isinstance(a, bool) for a in args))

    def test_paged_attention_is_handed_a_preallocated_workspace(self):
        """The operator never allocates its own reduction scratch."""
        impl = self._make_impl()
        num_tokens = 2
        impl.key_cache = torch.zeros(1, 128, 2, HEAD_SIZE // TURBOQUANT_PACK_FACTOR, dtype=torch.int8)
        impl.value_cache = impl.key_cache
        impl.scale_cache = torch.zeros(1, 128, turboquant_scale_slot(2))
        query = torch.randn(num_tokens, impl.num_heads, HEAD_SIZE)
        output = torch.zeros(num_tokens, impl.num_heads, HEAD_SIZE)
        metadata = MagicMock(
            block_tables=torch.zeros(num_tokens, 1, dtype=torch.int64),
            seq_lens=torch.ones(num_tokens, dtype=torch.int64),
        )

        ops = _ops_mock(workspace_floats=WORKSPACE_FLOATS)
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl.forward_paged_attention(query, metadata, output)

        workspace = ops.npu_turboquant_paged_attention.call_args.args[8]
        self.assertIs(workspace, impl.decode_workspace)
        self.assertEqual(workspace.dtype, torch.float32)
        self.assertEqual(workspace.numel(), WORKSPACE_FLOATS)
        # Sized from the operator's own arithmetic, not a copy of it here.
        ops.npu_turboquant_workspace_size.assert_called_once_with(
            num_tokens, impl.num_heads, impl.head_size, 1
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
            first = impl._decode_workspace(2, 1, CPU)
            second = impl._decode_workspace(2, 1, CPU)

        self.assertIs(first, second)
        # The size is memoised per shape, so the steady-state step costs a dict
        # lookup rather than an operator dispatch.
        ops.npu_turboquant_workspace_size.assert_called_once()

    def test_a_narrower_decode_reuses_the_high_water_buffer(self):
        impl = self._make_impl()
        ops = _ops_mock()
        ops.npu_turboquant_workspace_size.side_effect = [WORKSPACE_FLOATS, WORKSPACE_FLOATS // 4]
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            wide = impl._decode_workspace(8, 4, CPU)
            narrow = impl._decode_workspace(2, 1, CPU)

        self.assertIs(wide, narrow)
        self.assertEqual(narrow.numel(), WORKSPACE_FLOATS)

    def test_a_wider_decode_grows_the_buffer(self):
        impl = self._make_impl()
        ops = _ops_mock()
        ops.npu_turboquant_workspace_size.side_effect = [WORKSPACE_FLOATS, WORKSPACE_FLOATS * 2]
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            narrow = impl._decode_workspace(2, 1, CPU)
            wide = impl._decode_workspace(8, 4, CPU)

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
            big_batch = impl._decode_workspace(3, 8, CPU)
            small_batch = impl._decode_workspace(1, 8, CPU)

        self.assertIsNot(big_batch, small_batch)
        self.assertEqual(small_batch.numel(), WORKSPACE_FLOATS * 2)

    def test_the_size_memo_is_bounded(self):
        """An uncaptured run drifts through shapes; the memo must not grow forever."""
        impl = self._make_impl()
        impl._workspace_floats = {(0, i): 1 for i in range(tq_module._WORKSPACE_MEMO_LIMIT)}
        ops = _ops_mock()
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl._decode_workspace(2, 1, CPU)

        self.assertEqual(len(impl._workspace_floats), 1)
        self.assertEqual(impl._workspace_floats[(2, 1)], WORKSPACE_FLOATS)

    def test_growth_during_a_capture_is_refused(self):
        """A buffer swapped mid-capture would be replayed as a dangling pointer."""
        impl = self._make_impl()
        ops = _ops_mock()
        ops.npu_turboquant_workspace_size.side_effect = [WORKSPACE_FLOATS, WORKSPACE_FLOATS * 2]
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl._decode_workspace(2, 1, CPU)
            with patch.object(tq_module, "_EXTRA_CTX", SimpleNamespace(capturing=True)):
                with self.assertRaisesRegex(RuntimeError, "during a graph capture"):
                    impl._decode_workspace(8, 4, CPU)

    def test_a_capture_that_needs_no_growth_is_allowed(self):
        impl = self._make_impl()
        ops = _ops_mock()
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            warmed = impl._decode_workspace(2, 1, CPU)
            with patch.object(tq_module, "_EXTRA_CTX", SimpleNamespace(capturing=True)):
                captured = impl._decode_workspace(2, 1, CPU)

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

            # absmax + eps keeps the reciprocal finite and the scale positive.
            self.assertTrue(torch.isfinite(step).all(), f"step not finite at D={head_size}")
            self.assertGreater(step.item(), 0.0)
            self.assertAlmostEqual(step.item(), TURBOQUANT_EPS / ZERO_POINT, delta=1e-30)

            # The mid-rise grid has no exact zero: round-half-to-even sends 7.5
            # to level 8, so the centred level is +0.5 rather than 0.0. What has
            # to hold is that the reconstruction is numerically zero.
            levels = _dequantize_levels(packed)
            self.assertTrue(torch.isfinite(levels).all())
            torch.testing.assert_close(levels, torch.full_like(levels, 0.5), atol=0, rtol=0)

            recon = levels * step
            self.assertTrue(torch.isfinite(recon).all())
            self.assertLess(recon.abs().max().item(), 1e-18)

    def test_single_dominant_outlier(self):
        head_size = 128
        for magnitude in (1.0e4, -1.0e4):
            vec = torch.zeros(head_size, dtype=torch.float64)
            vec[0] = magnitude
            packed, step = _quantize(vec)

            # The grid must key on the outlier and must not collapse.
            self.assertTrue(torch.isfinite(step).all())
            self.assertAlmostEqual(step.item(), abs(magnitude) / ZERO_POINT, delta=abs(magnitude) * 1e-9)

            levels = _dequantize_levels(packed)
            expected_extreme = ZERO_POINT if magnitude > 0 else -ZERO_POINT
            self.assertAlmostEqual(levels[0].item(), expected_extreme, places=6)
            torch.testing.assert_close(levels[1:], torch.full_like(levels[1:], 0.5), atol=0, rtol=0)

            # Packing an extreme beside a mid level must stay inside int8.
            self.assertGreaterEqual(int(packed.min()), -128)
            self.assertLessEqual(int(packed.max()), 127)

            recon = (levels * step)[0].item()
            self.assertAlmostEqual(recon, magnitude, delta=abs(magnitude) * 1e-5)

    def test_clamping_and_saturation_extremes(self):
        # Both nibbles at 15 give byte 255, which the -128 bias maps to +127;
        # both at 0 give -128. Those are the exact int8 endpoints, so a signed
        # overflow or a truncating cast shows up here and nowhere else.
        head_size = 64
        amplitude = 3.0
        cases = (
            ("all +A", amplitude, amplitude, 15, 15, 127),
            ("all -A", -amplitude, -amplitude, 0, 0, -128),
            ("alt +-A", amplitude, -amplitude, 15, 0, -113),
            ("alt -+A", -amplitude, amplitude, 0, 15, 112),
        )
        channels = torch.arange(head_size)
        for name, even, odd, want_low, want_high, want_byte in cases:
            vec = torch.where(channels % 2 == 0, even, odd).to(torch.float64)
            packed, _ = _quantize(vec)
            self.assertEqual(packed.unique().tolist(), [want_byte], name)
            self.assertGreaterEqual(int(packed.min()), -128, name)
            self.assertLessEqual(int(packed.max()), 127, name)

            levels = _dequantize_levels(packed) + ZERO_POINT
            self.assertEqual(levels[0::2].unique().tolist(), [float(want_low)], name)
            self.assertEqual(levels[1::2].unique().tolist(), [float(want_high)], name)

    def test_every_nibble_pair_round_trips(self):
        # All 256 (low, high) combinations, endpoints included.
        lows = torch.arange(16).repeat_interleave(16)
        highs = torch.arange(16).repeat(16)
        packed = (lows + PACK_HIGH * highs - INT8_BIAS).round().to(torch.int8).unsqueeze(-1)
        self.assertGreaterEqual(int(packed.min()), -128)
        self.assertLessEqual(int(packed.max()), 127)

        levels = _dequantize_levels(packed) + ZERO_POINT
        torch.testing.assert_close(levels[:, 0], lows.to(levels.dtype), atol=0, rtol=0)
        torch.testing.assert_close(levels[:, 1], highs.to(levels.dtype), atol=0, rtol=0)

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
        # A leaked absmax across rows would show up as a step that is right for
        # one row and wrong for the others.
        head_size = 64
        amplitudes = (1.0, 16.0, 256.0)
        vectors = torch.stack([torch.full((head_size,), amplitude, dtype=torch.float64) for amplitude in amplitudes])
        _, steps = _quantize(vectors)
        self.assertEqual(tuple(steps.shape), (len(amplitudes),))
        for row, amplitude in enumerate(amplitudes):
            self.assertAlmostEqual(steps[row].item(), amplitude / ZERO_POINT, delta=amplitude * 1e-9)


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
        return signs, xor_offsets, even, odd, expand, odd_select

    def test_image_length_matches_the_kernel_contract(self):
        for head_size in (64, 128, 256):
            for batch_rows in (1, TURBOQUANT_TILE_ROWS):
                tables = turboquant_codec_tables(head_size, batch_rows, CPU)
                self.assertEqual(tables.dtype, torch.int32)
                self.assertTrue(tables.is_contiguous())
                self.assertEqual(tables.numel(), 7 * head_size + 2 * head_size * batch_rows)
                # One DataCopy moves the image, so it must be a whole burst.
                self.assertEqual(tables.numel() % 8, 0)

    def test_sign_and_xor_tables_encode_the_butterfly(self):
        head_size = 128
        signs, xor_offsets, _, _, _, _ = self._split(head_size, 1)
        channels = torch.arange(head_size)
        for stage in range(3):
            stride = 1 << stage
            expected_sign = (1 - 2 * ((channels // stride) & 1)).to(torch.float32)
            torch.testing.assert_close(signs[stage], expected_sign, atol=0, rtol=0)
            # Gather consumes byte offsets, so p ^ stride scaled by sizeof(float).
            torch.testing.assert_close(xor_offsets[stage], (4 * (channels ^ stride)).to(torch.int32), atol=0, rtol=0)

    def test_nibble_tables_deinterleave_and_expand(self):
        head_size, batch_rows = 128, TURBOQUANT_TILE_ROWS
        _, _, even, odd, expand, odd_select = self._split(head_size, batch_rows)
        pairs = torch.arange(head_size // TURBOQUANT_PACK_FACTOR)
        torch.testing.assert_close(even, (8 * pairs).to(torch.int32), atol=0, rtol=0)
        torch.testing.assert_close(odd, (8 * pairs + 4).to(torch.int32), atol=0, rtol=0)

        batch = torch.arange(head_size * batch_rows)
        torch.testing.assert_close(expand, (4 * (batch // 2)).to(torch.int32), atol=0, rtol=0)
        torch.testing.assert_close(odd_select, (batch % 2).to(torch.float32), atol=0, rtol=0)

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

    def test_activation_swaps_the_impl(self):
        layer = MagicMock()
        layer.impl = MagicMock()
        layer.impl.__class__ = MagicMock
        tq_module.activate_turboquant_backend(layer)
        self.assertIs(layer.impl.__class__, AscendTurboQuantAttentionBackendImpl)
        self.assertEqual(layer.kv_cache_torch_dtype, torch.int8)
        # A swapped-in impl starts with no scratch of its own: whatever the
        # previous class left behind was sized for a different cache layout.
        self.assertIsNone(layer.impl.decode_workspace)
        self.assertEqual(layer.impl._workspace_floats, {})

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

    def test_unsupported_states_are_rejected(self):
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        metadata = MagicMock(attn_state=AscendAttentionState.ChunkedPrefill)
        with self.assertRaises(NotImplementedError):
            impl.forward_impl(None, None, None, (), metadata, None)
