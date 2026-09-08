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

from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention import turboquant_v1 as tq_module
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.turboquant_v1 import (
    TURBOQUANT_PACK_FACTOR,
    AscendTurboQuantAttentionBackend,
    AscendTurboQuantAttentionBackendImpl,
    apply_pi,
    turboquant_pi_signs,
    walsh_hadamard,
)

HEAD_SIZE = 128
LEVEL_MAX = 15.0
ZERO_POINT = 7.5
PACK_HIGH = 16.0
INT8_BIAS = 128.0

CPU = torch.device("cpu")

# The first eight channels of the shared LCG sign vector. Hard-coded so this
# test and csrc/tests/reference/turbo_quant_cpu.h pin the same constant from
# both sides: a cache written by one must be readable by the other.
EXPECTED_SIGNS_128 = [1, -1, 1, -1, -1, -1, 1, 1]
EXPECTED_SIGNS_256 = [-1, 1, -1, -1, -1, -1, 1, 1]


def _quantize(vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for TurboQuantCodec<4>::Quantize4Bit."""
    absmax = vec.abs().amax(dim=-1, keepdim=True)
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
        impl.key_scale_cache = None
        impl.value_scale_cache = None
        impl._pi_signs = None
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
        # key, value, k_cache, v_cache, k_scale, v_scale, slots, pi_signs -- and
        # no trailing rotation flag.
        self.assertEqual(len(args), 8)
        torch.testing.assert_close(args[0], key)
        torch.testing.assert_close(args[1], value)
        self.assertEqual(args[6].dtype, torch.int32)
        torch.testing.assert_close(args[7], turboquant_pi_signs(HEAD_SIZE, CPU))

    def test_paged_attention_passes_post_rope_query_and_no_flag(self):
        impl = self._make_impl()
        num_tokens = 2
        impl.key_cache = torch.zeros(1, 128, 2, HEAD_SIZE // TURBOQUANT_PACK_FACTOR, dtype=torch.int8)
        impl.value_cache = impl.key_cache
        impl.key_scale_cache = torch.zeros(1, 2, 128)
        impl.value_scale_cache = impl.key_scale_cache
        query = torch.randn(num_tokens, impl.num_heads, HEAD_SIZE)
        output = torch.zeros(num_tokens, impl.num_heads, HEAD_SIZE)
        metadata = MagicMock(
            block_tables=torch.zeros(num_tokens, 1, dtype=torch.int64),
            seq_lens=torch.ones(num_tokens, dtype=torch.int64),
        )

        ops = MagicMock()
        with patch.object(torch.ops, "_C_ascend", ops, create=True):
            impl.forward_paged_attention(query, metadata, output)

        ops.npu_turboquant_paged_attention.assert_called_once()
        args = ops.npu_turboquant_paged_attention.call_args.args
        # query, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens,
        # pi_signs, num_kv_heads, num_heads, scale, out.
        self.assertEqual(len(args), 12)
        torch.testing.assert_close(args[0], query)
        self.assertEqual(args[8], impl.num_kv_heads)
        self.assertEqual(args[9], impl.num_heads)
        self.assertFalse(any(isinstance(a, bool) for a in args))


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

    def test_scale_cache_shape_follows_the_packed_cache(self):
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        impl.key_scale_cache = None
        impl.value_scale_cache = None
        cache = torch.zeros(3, 128, 2, HEAD_SIZE // TURBOQUANT_PACK_FACTOR, dtype=torch.int8)
        impl._ensure_scale_caches((cache, cache))
        self.assertEqual(tuple(impl.key_scale_cache.shape), (3, 2, 128))
        self.assertEqual(impl.key_scale_cache.dtype, torch.float32)

    def test_unsupported_states_are_rejected(self):
        impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
        metadata = MagicMock(attn_state=AscendAttentionState.ChunkedPrefill)
        with self.assertRaises(NotImplementedError):
            impl.forward_impl(None, None, None, (), metadata, None)
