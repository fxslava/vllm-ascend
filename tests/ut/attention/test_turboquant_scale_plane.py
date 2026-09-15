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
"""The fp32 scale plane: where it comes from, and that it travels with its blocks.

The scales used to be a side allocation the backend made on first use, which
the block allocator could not see and block relocation left behind. The model
runner now carves the plane out of the same page budget and hands it over as
``kv_cache[2]``. These tests pin that hand-over, the refusals that catch a
mis-shaped plane, and the block moves that would otherwise separate codes from
the scales that decode them.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.turboquant_v1 import (
    AscendTurboQuantAttentionBackend,
    AscendTurboQuantAttentionBackendImpl,
    turboquant_scale_slot,
)

NUM_BLOCKS = 4
BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_SIZE = 64


def _impl() -> AscendTurboQuantAttentionBackendImpl:
    """A bare impl: _ensure_scale_cache reads nothing but its argument."""
    impl = AscendTurboQuantAttentionBackendImpl.__new__(AscendTurboQuantAttentionBackendImpl)
    impl.scale_cache = None
    return impl


def _packed_planes() -> tuple[torch.Tensor, torch.Tensor]:
    shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE // 2)
    return torch.zeros(shape, dtype=torch.int8), torch.zeros(shape, dtype=torch.int8)


def _scale_plane(slot: int | None = None) -> torch.Tensor:
    slot = turboquant_scale_slot(NUM_KV_HEADS) if slot is None else slot
    return torch.zeros((NUM_BLOCKS, BLOCK_SIZE, slot), dtype=torch.float32)


class TestScalePlaneHandover(TestBase):
    def test_a_provided_plane_is_adopted_rather_than_allocated(self):
        key, value = _packed_planes()
        provided = _scale_plane()
        impl = _impl()
        impl._ensure_scale_cache((key, value, provided))
        # the same storage, not a copy: the runner owns this memory
        self.assertIs(impl.scale_cache, provided)

    def test_the_expected_slot_is_the_burst_aligned_one(self):
        """num_kv_heads=2 needs 8 lanes, not the 4 an unpadded count would give."""
        self.assertEqual(turboquant_scale_slot(NUM_KV_HEADS), 8)
        key, value = _packed_planes()
        impl = _impl()
        impl._ensure_scale_cache((key, value, _scale_plane(8)))
        self.assertEqual(impl.scale_cache.shape[2], 8)

    def test_an_unpadded_plane_is_refused(self):
        """Exactly the out-of-bounds case: 2 * num_kv_heads lanes, unrounded."""
        key, value = _packed_planes()
        impl = _impl()
        with self.assertRaises(RuntimeError) as caught:
            impl._ensure_scale_cache((key, value, _scale_plane(2 * NUM_KV_HEADS)))
        self.assertIn("expected", str(caught.exception))

    def test_a_wrong_dtype_plane_is_refused(self):
        key, value = _packed_planes()
        wrong = _scale_plane().to(torch.float16)
        impl = _impl()
        with self.assertRaises(RuntimeError) as caught:
            impl._ensure_scale_cache((key, value, wrong))
        self.assertIn("float32", str(caught.exception))

    def test_a_two_tuple_still_allocates_for_the_cpu_harness(self):
        key, value = _packed_planes()
        impl = _impl()
        impl._ensure_scale_cache((key, value))
        self.assertEqual(
            impl.scale_cache.shape,
            (NUM_BLOCKS, BLOCK_SIZE, turboquant_scale_slot(NUM_KV_HEADS)),
        )
        self.assertEqual(impl.scale_cache.dtype, torch.float32)

    def test_allocating_during_a_graph_capture_is_refused(self):
        """The workspace and rotated-query buffers refuse this; so must the plane."""
        key, value = _packed_planes()
        impl = _impl()
        with patch("vllm_ascend.attention.turboquant_v1._is_capturing", return_value=True):
            with self.assertRaises(RuntimeError) as caught:
                impl._ensure_scale_cache((key, value))
        self.assertIn("graph capture", str(caught.exception))

    def test_a_provided_plane_is_fine_during_a_capture(self):
        """Nothing is allocated, so there is nothing to refuse."""
        key, value = _packed_planes()
        provided = _scale_plane()
        impl = _impl()
        with patch("vllm_ascend.attention.turboquant_v1._is_capturing", return_value=True):
            impl._ensure_scale_cache((key, value, provided))
        self.assertIs(impl.scale_cache, provided)


class TestBlocksMoveWithTheirScales(TestBase):
    @staticmethod
    def _filled_cache(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE // 2)
        key = torch.randint(-128, 127, shape, dtype=torch.int8, generator=generator)
        value = torch.randint(-128, 127, shape, dtype=torch.int8, generator=generator)
        scale = torch.rand(
            (NUM_BLOCKS, BLOCK_SIZE, turboquant_scale_slot(NUM_KV_HEADS)), generator=generator
        )
        return key, value, scale

    def test_copy_blocks_carries_the_scale_plane(self):
        cache = self._filled_cache(1)
        key, value, scale = cache
        src, dst = 0, 3
        AscendTurboQuantAttentionBackend.copy_blocks(
            [cache], torch.tensor([[src, dst]], dtype=torch.int64)
        )
        self.assertTrue(torch.equal(key[dst], key[src]))
        self.assertTrue(torch.equal(value[dst], value[src]))
        self.assertTrue(
            torch.equal(scale[dst], scale[src]),
            "the codes moved but their scales did not -- this is the silent-corruption case",
        )

    def test_swap_blocks_carries_the_scale_plane(self):
        source = self._filled_cache(2)
        destination = (
            torch.zeros_like(source[0]),
            torch.zeros_like(source[1]),
            torch.zeros_like(source[2]),
        )
        AscendTurboQuantAttentionBackend.swap_blocks(
            list(source), list(destination), torch.tensor([[1, 2]], dtype=torch.int64)
        )
        self.assertTrue(torch.equal(destination[0][2], source[0][1]))
        self.assertTrue(torch.equal(destination[1][2], source[1][1]))
        self.assertTrue(torch.equal(destination[2][2], source[2][1]))

    def test_a_two_plane_cache_still_moves(self):
        """Backwards compatible with a cache that has no separate scale plane."""
        key, value, _ = self._filled_cache(3)
        AscendTurboQuantAttentionBackend.copy_blocks(
            [(key, value)], torch.tensor([[0, 1]], dtype=torch.int64)
        )
        self.assertTrue(torch.equal(key[1], key[0]))


class TestSpecConversion(TestBase):
    """_as_turboquant_spec reads nothing off the runner but vllm_config."""

    @staticmethod
    def _convert(spec, cache_dtype="int4_per_token_head"):
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        runner = SimpleNamespace(
            vllm_config=SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=cache_dtype))
        )
        return NPUModelRunner._as_turboquant_spec(runner, "layer.0", spec)

    @staticmethod
    def _full_spec(**kwargs):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        defaults = dict(
            block_size=BLOCK_SIZE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            dtype=torch.bfloat16,
        )
        defaults.update(kwargs)
        return FullAttentionSpec(**defaults)

    def test_a_dense_spec_becomes_a_turboquant_spec(self):
        from vllm_ascend.core.kv_cache_interface import AscendTurboQuantAttentionSpec

        converted = self._convert(self._full_spec())
        self.assertIsInstance(converted, AscendTurboQuantAttentionSpec)
        self.assertLess(converted.page_size_bytes, self._full_spec().page_size_bytes)

    def test_the_converted_page_holds_the_planes_and_the_scales(self):
        converted = self._convert(self._full_spec())
        payload = 2 * BLOCK_SIZE * NUM_KV_HEADS * (HEAD_SIZE // 2)
        scale = BLOCK_SIZE * turboquant_scale_slot(NUM_KV_HEADS) * 4
        self.assertEqual(converted.page_size_bytes, payload + scale)

    def test_it_leaves_other_runs_alone(self):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        spec = self._full_spec()
        self.assertIs(self._convert(spec, cache_dtype="auto"), spec)
        self.assertIsInstance(self._convert(spec, cache_dtype="auto"), FullAttentionSpec)

    def test_a_sliding_window_layer_is_left_alone(self):
        """TurboQuant refuses sliding window in forward; do not resize its cache."""
        spec = self._full_spec(sliding_window=256)
        self.assertIs(self._convert(spec), spec)
