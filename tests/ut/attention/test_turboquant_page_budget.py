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
"""Page budgeting for the TurboQuant 4-bit KV cache.

A page has to hold exactly what the backend writes into it: two packed int8
planes and a burst-aligned fp32 scale plane. The tests here hold
:class:`AscendTurboQuantAttentionSpec` to the shapes the backend itself
reports, so the two cannot drift apart, and pin the two places the arithmetic
is easy to get wrong -- the scale plane's burst padding at small KV-head
counts, and the comparison against the stock unpacked page.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.turboquant_v1 import (
    AscendTurboQuantAttentionBackend,
    TURBOQUANT_PACK_FACTOR,
    turboquant_scale_slot,
)
from vllm_ascend.core.kv_cache_interface import AscendTurboQuantAttentionSpec
from vllm_ascend.utils import TURBOQUANT_KV_CACHE_DTYPE, is_turboquant_cache_dtype

HEAD_SIZE = 128
BLOCK_SIZE = 128
# The kernels take any head count; 1 to 3 are the ones whose scale burst is
# padded, and 4 upward are the ones where it is not.
KV_HEAD_COUNTS = (1, 2, 3, 4, 5, 6, 8, 16)


def _spec(num_kv_heads: int, block_size: int = BLOCK_SIZE, head_size: int = HEAD_SIZE, **kwargs):
    return AscendTurboQuantAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
        **kwargs,
    )


class TestTurboQuantScaleSlot(TestBase):
    def test_the_spec_and_the_backend_agree_on_the_scale_slot(self):
        """The spec reimplements turboquant_scale_slot; hold the two together."""
        for num_kv_heads in KV_HEAD_COUNTS:
            self.assertEqual(
                _spec(num_kv_heads).scale_slot_floats,
                turboquant_scale_slot(num_kv_heads),
                f"scale slot disagrees at num_kv_heads={num_kv_heads}",
            )

    def test_the_scale_slot_is_a_whole_burst_and_covers_every_lane(self):
        """The arithmetic itself, by value: 2 * num_kv_heads rounded up to a burst.

        Spelled out rather than recomputed, so a change to the rounding has to be
        written down here to pass. The two invariants underneath it -- every lane
        kept, every slot burst-aligned -- are what the kernel's one aligned
        ``DataCopy`` per token depends on.
        """
        expected_slots = {1: 8, 2: 8, 3: 8, 4: 8, 5: 16, 6: 16, 8: 16, 16: 32}
        self.assertEqual(sorted(expected_slots), sorted(KV_HEAD_COUNTS))
        for num_kv_heads, expected in expected_slots.items():
            slot = turboquant_scale_slot(num_kv_heads)
            self.assertEqual(slot, expected, f"num_kv_heads={num_kv_heads}")
            self.assertEqual(slot % 8, 0, f"num_kv_heads={num_kv_heads} is not burst-aligned")
            self.assertGreaterEqual(
                slot, 2 * num_kv_heads, f"num_kv_heads={num_kv_heads} loses scale lanes"
            )


class TestTurboQuantPageSize(TestBase):
    def test_the_page_holds_exactly_what_the_backend_writes(self):
        """The decisive test: page bytes == the tensors the kernels are handed.

        The backend reports its packed shape and the kernels require a
        ``(num_blocks, block_size, scale_slot)`` fp32 plane beside it. One
        page must be those two, to the byte.
        """
        for num_kv_heads in KV_HEAD_COUNTS:
            spec = _spec(num_kv_heads)
            packed = AscendTurboQuantAttentionBackend.get_kv_cache_shape(
                1, BLOCK_SIZE, num_kv_heads, HEAD_SIZE
            )
            # shape is (2, num_blocks, block_size, num_kv_heads, head_size // 2) int8
            payload = 1
            for dim in packed:
                payload *= dim
            scale = BLOCK_SIZE * turboquant_scale_slot(num_kv_heads) * get_dtype_size(torch.float32)
            self.assertEqual(
                spec.real_page_size_bytes,
                payload + scale,
                f"page does not match the backend's own shapes at num_kv_heads={num_kv_heads}",
            )

    def test_the_payload_is_half_the_unpacked_head(self):
        for num_kv_heads in KV_HEAD_COUNTS:
            spec = _spec(num_kv_heads)
            self.assertEqual(
                spec.payload_bytes,
                2 * BLOCK_SIZE * num_kv_heads * (HEAD_SIZE // TURBOQUANT_PACK_FACTOR),
            )

    def test_the_reference_shape_budgets_the_documented_page(self):
        """head_size 128 / 8 heads / block 128: 128 KiB payload + 8 KiB scales."""
        spec = _spec(8)
        self.assertEqual(spec.payload_bytes, 128 * 1024)
        self.assertEqual(spec.scale_plane_bytes, 8 * 1024)
        self.assertEqual(spec.real_page_size_bytes, 136 * 1024)

    def test_it_beats_the_stock_unpacked_page(self):
        """The point of the feature: a bf16 FullAttentionSpec page is ~3.8x larger."""
        stock = FullAttentionSpec(
            block_size=BLOCK_SIZE, num_kv_heads=8, head_size=HEAD_SIZE, dtype=torch.bfloat16
        )
        packed = _spec(8)
        self.assertEqual(stock.page_size_bytes, 512 * 1024)
        self.assertLess(packed.page_size_bytes, stock.page_size_bytes)
        ratio = stock.page_size_bytes / packed.page_size_bytes
        self.assertAlmostEqual(ratio, 3.7647, places=3)


class TestAgainstUpstreamInt4Mode(TestBase):
    """vLLM's own int4_per_token_head budget, which this spec exists to correct."""

    @staticmethod
    def _upstream(num_kv_heads: int) -> int:
        spec = FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=num_kv_heads,
            head_size=HEAD_SIZE,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
        )
        return spec.page_size_bytes

    def test_it_matches_upstream_where_the_burst_needs_no_padding(self):
        for num_kv_heads in (4, 8, 16):
            self.assertEqual(
                _spec(num_kv_heads).page_size_bytes,
                self._upstream(num_kv_heads),
                f"should agree with int4_per_token_head at num_kv_heads={num_kv_heads}",
            )

    def test_it_exceeds_upstream_exactly_by_the_burst_padding(self):
        """Below 4 heads upstream budgets fewer lanes than the kernels scatter."""
        for num_kv_heads in (1, 2, 3):
            ours = _spec(num_kv_heads).page_size_bytes
            theirs = self._upstream(num_kv_heads)
            padding_lanes = turboquant_scale_slot(num_kv_heads) - 2 * num_kv_heads
            self.assertEqual(
                ours - theirs,
                BLOCK_SIZE * padding_lanes * get_dtype_size(torch.float32),
                f"the gap at num_kv_heads={num_kv_heads} is not the burst padding",
            )

    def test_upstream_is_never_larger(self):
        """A page smaller than this layout needs would be an out-of-bounds write."""
        for num_kv_heads in KV_HEAD_COUNTS:
            self.assertGreaterEqual(_spec(num_kv_heads).page_size_bytes, self._upstream(num_kv_heads))


class TestPagePadding(TestBase):
    def test_an_explicit_padded_page_is_honoured(self):
        spec = _spec(8, page_size_padded=200 * 1024)
        self.assertEqual(spec.page_size_bytes, 200 * 1024)
        self.assertEqual(spec.real_page_size_bytes, 136 * 1024)

    def test_a_padded_page_below_the_real_one_is_refused(self):
        spec = _spec(8, page_size_padded=64 * 1024)
        with self.assertRaises(AssertionError):
            spec.page_size_bytes

    def test_merge_keeps_the_packing(self):
        merged = AscendTurboQuantAttentionSpec.merge([_spec(8), _spec(8)])
        self.assertEqual(merged.pack_factor, TURBOQUANT_PACK_FACTOR)
        self.assertEqual(merged.real_page_size_bytes, 136 * 1024)


class TestBlockSizes(TestBase):
    def test_every_kernel_legal_block_size_budgets_proportionally(self):
        """The kernels take any multiple of 16, so the page scales with it."""
        base = _spec(8, block_size=16).real_page_size_bytes
        for block_size in (16, 32, 64, 128):
            spec = _spec(8, block_size=block_size)
            self.assertEqual(spec.real_page_size_bytes, base * (block_size // 16))


class TestPackedShapeFollowsEveryRoute(TestBase):
    """The route into the layout and the shape the cache is viewed as must agree.

    ``turboquant_enabled`` opens the layout three ways, and a page is budgeted by
    :class:`AscendTurboQuantAttentionSpec` for all of them. The backend the model runner
    actually asks for the shape is the registered ``AscendAttentionBackend``, so it has to
    report the *packed* shape for the same set -- otherwise a plane allocated for a packed
    page is viewed as an unpacked one and ``_reshape_kv_cache_tensors`` fails outright.
    ``--kv-cache-dtype int4_per_token_head`` is the route that does not carry the
    ``turboquant_`` prefix, which is exactly how the two drifted apart.
    """

    PACKED = HEAD_SIZE // TURBOQUANT_PACK_FACTOR

    def test_int4_per_token_head_is_a_turboquant_cache_dtype(self):
        self.assertTrue(is_turboquant_cache_dtype(TURBOQUANT_KV_CACHE_DTYPE))
        self.assertEqual(TURBOQUANT_KV_CACHE_DTYPE, "int4_per_token_head")

    def test_every_turboquant_cache_dtype_reports_the_packed_shape(self):
        for cache_dtype in (TURBOQUANT_KV_CACHE_DTYPE, "turboquant_4bit_nc"):
            with patch.dict("os.environ", {"ENABLE_TURBOQUANT": "0"}):
                shape = AscendAttentionBackend.get_kv_cache_shape(2, BLOCK_SIZE, 8, HEAD_SIZE, cache_dtype)
            self.assertEqual(shape[-1], self.PACKED, f"{cache_dtype} did not pack the head")
            self.assertEqual(
                shape,
                AscendTurboQuantAttentionBackend.get_kv_cache_shape(2, BLOCK_SIZE, 8, HEAD_SIZE),
                f"{cache_dtype} disagrees with the TurboQuant backend's own shape",
            )

    def test_a_stock_cache_dtype_keeps_the_unpacked_shape(self):
        for cache_dtype in ("auto", "int8", ""):
            with patch.dict("os.environ", {"ENABLE_TURBOQUANT": "0"}):
                shape = AscendAttentionBackend.get_kv_cache_shape(2, BLOCK_SIZE, 8, HEAD_SIZE, cache_dtype)
            self.assertEqual(shape[-1], HEAD_SIZE, f"{cache_dtype!r} should not pack the head")

    def test_the_budgeted_page_holds_the_shape_that_route_reports(self):
        """The decisive one: bytes budgeted == bytes the runner then views."""
        for num_kv_heads in KV_HEAD_COUNTS:
            with patch.dict("os.environ", {"ENABLE_TURBOQUANT": "0"}):
                packed = AscendAttentionBackend.get_kv_cache_shape(
                    1, BLOCK_SIZE, num_kv_heads, HEAD_SIZE, TURBOQUANT_KV_CACHE_DTYPE
                )
            payload = 1
            for dim in packed:
                payload *= dim
            scale = BLOCK_SIZE * turboquant_scale_slot(num_kv_heads) * get_dtype_size(torch.float32)
            self.assertEqual(
                _spec(num_kv_heads).real_page_size_bytes,
                payload + scale,
                f"the int4_per_token_head route mis-sizes the page at num_kv_heads={num_kv_heads}",
            )

    def test_the_impl_follows_the_same_route_as_the_layout(self):
        """The shape and the impl have to agree, or a stock impl writes a packed plane.

        ``get_impl_cls`` takes no config, so it reads the current one; the dtype route is
        the one that reaches it that way rather than through the environment.
        """
        from vllm_ascend.attention.turboquant_v1 import AscendTurboQuantAttentionBackendImpl

        config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=TURBOQUANT_KV_CACHE_DTYPE))
        with (
            patch.dict("os.environ", {"ENABLE_TURBOQUANT": "0"}),
            patch("vllm.config.get_current_vllm_config_or_none", return_value=config),
            patch("vllm_ascend.attention.attention_v1.enable_dcp", return_value=False),
        ):
            self.assertIs(AscendAttentionBackend.get_impl_cls(), AscendTurboQuantAttentionBackendImpl)

    def test_a_stock_cache_dtype_keeps_the_stock_impl(self):
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

        config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="auto"))
        with (
            patch.dict("os.environ", {"ENABLE_TURBOQUANT": "0"}),
            patch("vllm.config.get_current_vllm_config_or_none", return_value=config),
            patch("vllm_ascend.attention.attention_v1.enable_dcp", return_value=False),
        ):
            self.assertIs(AscendAttentionBackend.get_impl_cls(), AscendAttentionBackendImpl)
