# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# This file is a part of the vllm-ascend project.
"""Byte layout of the TurboQuant three-plane KV cache page.

One page of the TurboQuant cache holds three things, and exactly one module
describes all three so they cannot drift apart:

* the packed key plane, ``block_size * num_kv_heads * head_size/2`` INT8,
* the packed value plane, the same again,
* the fp32 scale plane, ``2 * num_kv_heads * block_size`` floats --- one scale
  per (plane, head, token), plane-major, the layout the staged kernels'
  reference uses (``tq_oracle::Reference::scale_offset``).

Three consumers share it: the KV cache spec
(:class:`vllm_ascend.core.kv_cache_interface.AscendTurboQuantAttentionSpec`)
budgets a page from it, the model runner carves and views the planes with it,
and the backend (:mod:`vllm_ascend.attention.turboquant_dry_run`) writes the
bytes it describes. A page budget that disagreed with the bytes written is
silent corruption rather than an error, which is why the arithmetic lives here
once.

Deliberately a leaf module: ``torch`` only, no ``vllm`` and no other
``vllm_ascend`` import, so the cache-spec module can use it without importing
an attention backend.
"""

import os
from typing import Final

import torch

# Two signed INT4 codes per byte, low nibble first.
TURBOQUANT_PACK_FACTOR: Final[int] = 2

# Plane 0 is the key tensor, plane 1 the value tensor, in the packed cache and
# in the scale cache alike.
TURBOQUANT_PLANES: Final[int] = 2
TURBOQUANT_KEY_PLANE: Final[int] = 0
TURBOQUANT_VALUE_PLANE: Final[int] = 1

# The INT8 K0 of the arch-3510 fractal: the inner Hadamard factor is 32 wide
# because one Cube contraction is exactly one K fractal. A head that is not a
# whole number of groups has no rotation.
TURBOQUANT_GROUP: Final[int] = 32

# Every operand base address the kernels read or write is one aligned GM burst.
TURBOQUANT_GM_BURST_BYTES: Final[int] = 32

# Environment switch. ENABLE_TURBOQUANT selects the TurboQuant backend
# (vllm_ascend/platform.py) and with it the packed cache spec and the
# three-plane carve (vllm_ascend/worker/model_runner_v1.py).
TURBOQUANT_ENABLE_ENV: Final[str] = "ENABLE_TURBOQUANT"

# One fp32 scale per (plane, head, token).
TURBOQUANT_SCALE_DTYPE: Final[torch.dtype] = torch.float32
TURBOQUANT_SCALE_DTYPE_SIZE: Final[int] = 4


def turboquant_enabled() -> bool:
    """Whether ``ENABLE_TURBOQUANT=1`` selected the TurboQuant cache."""
    return os.getenv(TURBOQUANT_ENABLE_ENV, "0") == "1"


def turboquant_packed_bytes(head_size: int) -> int:
    """Bytes one (token, head) of one packed plane occupies."""
    if head_size <= 0 or head_size % TURBOQUANT_GROUP:
        raise ValueError(f"TurboQuant needs head_size to be a positive multiple of {TURBOQUANT_GROUP}, got {head_size}")
    return head_size // TURBOQUANT_PACK_FACTOR


def turboquant_packed_plane_bytes(block_size: int, num_kv_heads: int, head_size: int) -> int:
    """Bytes one page of one packed plane occupies."""
    return block_size * num_kv_heads * turboquant_packed_bytes(head_size)


def turboquant_scale_plane_bytes(block_size: int, num_kv_heads: int) -> int:
    """Bytes one page of the fp32 scale plane occupies (both K and V scales)."""
    return TURBOQUANT_PLANES * num_kv_heads * block_size * TURBOQUANT_SCALE_DTYPE_SIZE


def turboquant_page_size_bytes(block_size: int, num_kv_heads: int, head_size: int) -> int:
    """Bytes one TurboQuant page occupies: both packed planes and the scales."""
    return TURBOQUANT_PLANES * turboquant_packed_plane_bytes(
        block_size, num_kv_heads, head_size
    ) + turboquant_scale_plane_bytes(block_size, num_kv_heads)


def turboquant_packed_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[int, ...]:
    """``(2, num_blocks, block_size, num_kv_heads, head_size // 2)`` INT8.

    Token-major inside a ``(page, head)`` panel. The fp32 scale plane is not
    part of this shape: it is the third plane, shaped by
    :func:`turboquant_scale_cache_shape`.
    """
    return (
        TURBOQUANT_PLANES,
        num_blocks,
        block_size,
        num_kv_heads,
        turboquant_packed_bytes(head_size),
    )


def turboquant_scale_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int) -> tuple[int, ...]:
    """Shape of the fp32 scale plane: ``(2, num_blocks, num_kv_heads, block_size)``.

    Plane-major, then page, then head, then token -- the staged kernels' layout
    (``tq_oracle::Reference::scale_offset``). A token's scales for one head are
    therefore *not* contiguous; the AIV scatter writer uses a burst-padded
    per-token slot instead, and that difference is one of the things a kernel
    bring-up has to reconcile.
    """
    return (TURBOQUANT_PLANES, num_blocks, num_kv_heads, block_size)


def turboquant_carve_planes(
    buffer: torch.Tensor,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Carve one layer's raw INT8 allocation into the three planes.

    ``buffer`` is the layer's own contiguous byte region, as the allocator hands
    it over: a page budget, not a shape. The planes are taken from its head in
    physical order (packed keys, packed values, scales) so a page padded to a
    larger common size -- which is what a hybrid model does, see
    :meth:`NPUModelRunner._as_turboquant_spec` -- simply leaves its tail unused.

    Each plane starts on an aligned GM burst, which the kernels require of every
    operand base address, and is a whole number of pages, which the views
    require.
    """
    if buffer.dtype is not torch.int8:
        raise ValueError(f"the TurboQuant page budget must be raw int8 bytes, got {buffer.dtype}")
    if buffer.ndim != 1 or buffer.stride(0) != 1:
        raise ValueError("the TurboQuant page budget must be a contiguous 1-D int8 view")
    if num_blocks <= 0:
        raise ValueError(f"the TurboQuant cache needs at least one block, got {num_blocks}")

    packed_bytes = num_blocks * turboquant_packed_plane_bytes(block_size, num_kv_heads, head_size)
    scale_bytes = num_blocks * turboquant_scale_plane_bytes(block_size, num_kv_heads)
    needed = TURBOQUANT_PLANES * packed_bytes + scale_bytes
    if buffer.numel() < needed:
        raise ValueError(
            f"the TurboQuant three-plane carve needs {needed} bytes for {num_blocks} blocks, "
            f"but the allocation is {buffer.numel()} bytes"
        )

    planes = (
        buffer[:packed_bytes],
        buffer[packed_bytes : 2 * packed_bytes],
        buffer[2 * packed_bytes : 2 * packed_bytes + scale_bytes],
    )
    for name, plane in zip(("key", "value", "scale"), planes):
        if plane.data_ptr() % TURBOQUANT_GM_BURST_BYTES:
            raise ValueError(
                f"the TurboQuant {name} plane starts at {plane.data_ptr()}, which is not an "
                f"aligned {TURBOQUANT_GM_BURST_BYTES}-byte burst"
            )
    return planes


def turboquant_plane_views(
    planes: tuple[torch.Tensor, ...] | list[torch.Tensor],
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """View the three carved planes as the backend reads them.

    ``(key, value)`` are ``(num_blocks, block_size, num_kv_heads, head_size/2)``
    INT8 and the scale plane is ``(2, num_blocks, num_kv_heads, block_size)``
    fp32. The block count is derived from the planes themselves rather than
    passed in, so a padded page budget cannot silently produce a view over the
    padding.
    """
    if len(planes) != TURBOQUANT_PLANES + 1:
        raise ValueError(f"TurboQuant expects three planes, got {len(planes)}")
    raw_k, raw_v, raw_scale = planes
    page_bytes = turboquant_packed_plane_bytes(block_size, num_kv_heads, head_size)
    scale_page_bytes = turboquant_scale_plane_bytes(block_size, num_kv_heads)
    if raw_k.numel() != raw_v.numel():
        raise ValueError(f"the TurboQuant packed planes differ in size: {raw_k.numel()} and {raw_v.numel()}")
    if raw_k.numel() % page_bytes or raw_scale.numel() % scale_page_bytes:
        raise ValueError(
            f"the TurboQuant planes are not a whole number of pages: {raw_k.numel()} packed bytes "
            f"and {raw_scale.numel()} scale bytes for a {page_bytes}+{scale_page_bytes}-byte page"
        )
    num_blocks = raw_k.numel() // page_bytes
    if raw_scale.numel() // scale_page_bytes != num_blocks:
        raise ValueError(
            f"the TurboQuant scale plane covers {raw_scale.numel() // scale_page_bytes} pages, "
            f"but the packed planes cover {num_blocks}"
        )
    packed_shape = turboquant_packed_cache_shape(num_blocks, block_size, num_kv_heads, head_size)[1:]
    return (
        raw_k.view(torch.int8).view(packed_shape),
        raw_v.view(torch.int8).view(packed_shape),
        raw_scale.view(TURBOQUANT_SCALE_DTYPE).view(turboquant_scale_cache_shape(num_blocks, block_size, num_kv_heads)),
    )
