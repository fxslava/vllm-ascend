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
"""Static KV-cache pools, allocated once and never resized.

One sequence at a time, so the block table is the identity: block ``i`` holds
positions ``[i * block_size, (i + 1) * block_size)`` and a token's slot *is* its
position.  That is not a simplification of the paged layout, it is a
one-sequence instance of it -- the operators still receive a real block table
and a real slot mapping, and everything they check about paging is still
checked.  What it removes is the allocator, the free list and the eviction
policy, none of which an evaluation of a single prompt needs.

Two pools implement :class:`StaticKVCache`:

* :class:`DenseKVCache` -- ``(num_blocks, block_size, num_kv_heads, head_size)``
  per plane per layer, in the model dtype.  What
  ``npu_fused_infer_attention_score`` reads, once viewed as
  ``(num_blocks, block_size, -1)``.
* :class:`TurboQuantKVCache` -- the packed 4-bit planes plus the fp32 scale
  plane, laid out exactly as ``AscendTurboQuantAttentionBackend`` budgets them.

Both are allocated for the full ``max_seq_len`` up front.  A decode step that
allocates is a decode step that cannot be captured into a graph and whose
latency percentiles measure the allocator, so :meth:`StaticKVCache.allocate` is
the only place either pool grows.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch

from tq_longbench._ascend import turboquant_layout

# ``torch.int8`` payload, ``torch.float32`` scales; the operators fix both.
_PACKED_DTYPE = torch.int8
_SCALE_DTYPE = torch.float32

# What the operators require of the geometry, mirrored from the CPU stand-ins
# and the tiling host code so a bad shape is refused here rather than inside a
# kernel launch.
_MIN_HEAD_SIZE = 64
_MAX_HEAD_SIZE = 256
_MEGABYTE = 1024 * 1024


@dataclass(frozen=True)
class CacheGeometry:
    """The shape of a run's cache, fixed before the first token is written."""

    num_layers: int
    num_kv_heads: int
    head_size: int
    block_size: int
    max_seq_len: int

    def __post_init__(self) -> None:
        layout = turboquant_layout()
        if self.num_layers <= 0 or self.num_kv_heads <= 0:
            raise ValueError(
                f"num_layers and num_kv_heads must be positive, got {self.num_layers} and {self.num_kv_heads}"
            )
        if not _MIN_HEAD_SIZE <= self.head_size <= _MAX_HEAD_SIZE:
            raise ValueError(
                f"TurboQuant requires {_MIN_HEAD_SIZE} <= head_size <= {_MAX_HEAD_SIZE}, got {self.head_size}"
            )
        if self.head_size & (self.head_size - 1):
            raise ValueError(
                f"TurboQuant needs a power-of-two head_size for the Walsh-Hadamard transform, got {self.head_size}"
            )
        if self.block_size % layout.TURBOQUANT_TILE_ROWS:
            raise ValueError(f"block_size must be a multiple of {layout.TURBOQUANT_TILE_ROWS}, got {self.block_size}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")

    @property
    def num_blocks(self) -> int:
        """Blocks reserved for the one sequence, rounded up to cover ``max_seq_len``."""
        return -(-self.max_seq_len // self.block_size)

    @property
    def scale_slot(self) -> int:
        """fp32 lanes one token occupies in the TurboQuant scale plane."""
        return turboquant_layout().turboquant_scale_slot(self.num_kv_heads)

    def require_cube_tiling(self) -> None:
        """Refuse a block size the Cube decode cannot read.

        It reads a block one 64-row tile at a time, so a block that is not a
        whole number of tiles would have the kernel walking off the end of the
        last one.  ``AscendTurboQuantAttentionBackendImpl._ensure_scale_cache``
        raises the same way.
        """
        tile = turboquant_layout().TURBOQUANT_CUBE_TILE_ROWS
        if self.block_size % tile:
            raise ValueError(
                f"the Cube decode reads {tile}-row tiles, so block_size must be a multiple of {tile}, "
                f"got {self.block_size}. Use --backend turboquant_aiv to decode this cache on the vector path."
            )


class StaticKVCache(abc.ABC):
    """A pool sized for one sequence of ``geometry.max_seq_len`` tokens.

    Subclasses own their planes; what is shared is the paging arithmetic, which
    is the part the operators see.
    """

    def __init__(self, geometry: CacheGeometry, device: torch.device) -> None:
        self.geometry = geometry
        self.device = device
        self._block_table: torch.Tensor | None = None
        self.allocate()

    @abc.abstractmethod
    def allocate(self) -> None:
        """Reserve every byte this pool will ever use."""

    @abc.abstractmethod
    def bytes_allocated(self) -> int:
        """Total device bytes held, across all layers."""

    def megabytes_allocated(self) -> float:
        return self.bytes_allocated() / _MEGABYTE

    def bytes_per_token_per_layer(self) -> float:
        """What one token costs, for the saving the run reports."""
        return self.bytes_allocated() / (self.geometry.num_layers * self.geometry.num_blocks * self.geometry.block_size)

    def slot_mapping(self, start: int, count: int) -> torch.Tensor:
        """int32 slots for ``count`` tokens beginning at position ``start``.

        Identity paging, so a slot is a position -- but it goes through this
        method rather than being written inline at the call sites, because the
        one thing that must never disagree is the slot a token is written to and
        the block table row the decode reads it back through.  A mismatch does
        not raise; it silently attends to another token's keys.
        """
        end = start + count
        if start < 0 or end > self.geometry.max_seq_len:
            raise ValueError(f"positions [{start}, {end}) fall outside the {self.geometry.max_seq_len}-token cache")
        return torch.arange(start, end, dtype=torch.int32, device=self.device)

    def block_table(self, context_len: int) -> torch.Tensor:
        """int32 ``(1, blocks)`` covering ``context_len`` tokens.

        Narrowed to the blocks the context actually reaches: the decode's
        workspace is sized from ``max_blocks_per_seq``, so handing it the full
        128k table while decoding at 8k would reserve scratch for a context that
        is not there.  The full table is allocated once and sliced.
        """
        if context_len < 0 or context_len > self.geometry.max_seq_len:
            raise ValueError(f"context_len {context_len} is outside [0, {self.geometry.max_seq_len}]")
        if self._block_table is None:
            self._block_table = torch.arange(self.geometry.num_blocks, dtype=torch.int32, device=self.device).view(
                1, -1
            )
        blocks = max(1, -(-context_len // self.geometry.block_size))
        return self._block_table[:, :blocks]


class DenseKVCache(StaticKVCache):
    """Unquantised paged cache: the ``native_v5`` baseline, and the prefill staging pool.

    ``npu_fused_infer_attention_score`` wants each plane as
    ``(num_blocks, block_size, num_kv_heads * head_size)``; it is stored with the
    heads split out so a write can index them, and :meth:`flat` does the view.
    """

    def __init__(self, geometry: CacheGeometry, device: torch.device, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.key_planes: list[torch.Tensor] = []
        self.value_planes: list[torch.Tensor] = []
        super().__init__(geometry, device)

    def allocate(self) -> None:
        shape = (
            self.geometry.num_blocks,
            self.geometry.block_size,
            self.geometry.num_kv_heads,
            self.geometry.head_size,
        )
        self.key_planes = [
            torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in range(self.geometry.num_layers)
        ]
        self.value_planes = [
            torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in range(self.geometry.num_layers)
        ]

    def bytes_allocated(self) -> int:
        per_plane = self.key_planes[0].numel() * self.key_planes[0].element_size()
        return 2 * self.geometry.num_layers * per_plane

    def flat(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(key, value)`` as the attention operator reads them."""
        blocks, block_size = self.geometry.num_blocks, self.geometry.block_size
        return (
            self.key_planes[layer].view(blocks, block_size, -1),
            self.value_planes[layer].view(blocks, block_size, -1),
        )

    def write(self, layer: int, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor) -> None:
        """Scatter ``[num_tokens, num_kv_heads, head_size]`` K/V into the named slots."""
        rows = slots.to(torch.int64)
        packed = (-1, self.geometry.num_kv_heads, self.geometry.head_size)
        self.key_planes[layer].view(packed)[rows] = key.to(self.dtype)
        self.value_planes[layer].view(packed)[rows] = value.to(self.dtype)


class TurboQuantKVCache(StaticKVCache):
    """The 4-bit packed cache and its fp32 scale plane.

    Three planes per layer, shaped as ``AscendTurboQuantAttentionBackend``
    budgets them:

    * ``key`` / ``value``: ``(num_blocks, block_size, num_kv_heads, head_size // 2)``
      int8, two 4-bit codes per byte;
    * ``scale``: ``(num_blocks, block_size, scale_slot)`` fp32, token-major --
      a token's K scales for every kv head, then its V scales, padded to a whole
      32-byte burst so the kernel scatters them with one aligned copy.

    The scale plane is not optional and not derivable: packed codes without
    their scales decode to plausible wrong numbers, which is why
    :meth:`StaticKVCache.slot_mapping` is shared and why the planes are
    allocated together.
    """

    def __init__(self, geometry: CacheGeometry, device: torch.device) -> None:
        self.key_planes: list[torch.Tensor] = []
        self.value_planes: list[torch.Tensor] = []
        self.scale_planes: list[torch.Tensor] = []
        super().__init__(geometry, device)

    def allocate(self) -> None:
        layout = turboquant_layout()
        packed_shape = (
            self.geometry.num_blocks,
            self.geometry.block_size,
            self.geometry.num_kv_heads,
            self.geometry.head_size // layout.TURBOQUANT_PACK_FACTOR,
        )
        scale_shape = (self.geometry.num_blocks, self.geometry.block_size, self.geometry.scale_slot)
        layers = range(self.geometry.num_layers)
        self.key_planes = [torch.zeros(packed_shape, dtype=_PACKED_DTYPE, device=self.device) for _ in layers]
        self.value_planes = [torch.zeros(packed_shape, dtype=_PACKED_DTYPE, device=self.device) for _ in layers]
        self.scale_planes = [torch.zeros(scale_shape, dtype=_SCALE_DTYPE, device=self.device) for _ in layers]

    def bytes_allocated(self) -> int:
        packed = self.key_planes[0].numel() * self.key_planes[0].element_size()
        scales = self.scale_planes[0].numel() * self.scale_planes[0].element_size()
        return self.geometry.num_layers * (2 * packed + scales)

    def planes(self, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(key, value, scale)`` for one layer, as the operators take them."""
        return self.key_planes[layer], self.value_planes[layer], self.scale_planes[layer]


def dense_equivalent_bytes(geometry: CacheGeometry, dtype: torch.dtype) -> int:
    """What the same context would cost unquantised, for the saving a run reports."""
    element = torch.empty((), dtype=dtype).element_size()
    tokens = geometry.num_blocks * geometry.block_size
    return 2 * geometry.num_layers * tokens * geometry.num_kv_heads * geometry.head_size * element
