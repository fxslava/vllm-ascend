# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the TurboQuant three-plane page layout.

Deliberately imports nothing but ``torch`` and the leaf layout module: the page
budget the KV cache spec asks for and the bytes the backend writes are the same
arithmetic, and this file is where that arithmetic is pinned.
"""

import pytest
import torch

from vllm_ascend.attention.turboquant_layout import (
    TURBOQUANT_PACK_FACTOR,
    TURBOQUANT_PLANES,
    turboquant_carve_planes,
    turboquant_packed_bytes,
    turboquant_packed_plane_bytes,
    turboquant_page_size_bytes,
    turboquant_plane_views,
    turboquant_scale_cache_shape,
    turboquant_scale_plane_bytes,
)

BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_SIZE = 64
NUM_BLOCKS = 3


def _page_budget(num_blocks: int = NUM_BLOCKS, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    return torch.zeros(
        num_blocks * turboquant_page_size_bytes(block_size, NUM_KV_HEADS, HEAD_SIZE),
        dtype=torch.int8,
    )


def test_page_is_two_packed_planes_and_the_scales():
    assert turboquant_packed_bytes(HEAD_SIZE) == HEAD_SIZE // TURBOQUANT_PACK_FACTOR
    assert turboquant_packed_plane_bytes(BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE) == BLOCK_SIZE * NUM_KV_HEADS * 32
    # One fp32 scale per (plane, head, token).
    assert turboquant_scale_plane_bytes(BLOCK_SIZE, NUM_KV_HEADS) == TURBOQUANT_PLANES * NUM_KV_HEADS * BLOCK_SIZE * 4
    assert turboquant_page_size_bytes(BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE) == (
        TURBOQUANT_PLANES * turboquant_packed_plane_bytes(BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        + turboquant_scale_plane_bytes(BLOCK_SIZE, NUM_KV_HEADS)
    )
    # A dense BF16 page of the same geometry: the packed page is the point.
    dense_page = BLOCK_SIZE * NUM_KV_HEADS * 2 * HEAD_SIZE * 2
    assert turboquant_page_size_bytes(BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE) * 3 < dense_page


@pytest.mark.parametrize("head_size", [0, -32, 48, 33])
def test_packed_bytes_rejects_a_head_that_is_not_whole_groups(head_size):
    with pytest.raises(ValueError, match="multiple of 32"):
        turboquant_packed_bytes(head_size)


def test_scale_cache_shape_is_plane_major():
    assert turboquant_scale_cache_shape(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS) == (
        TURBOQUANT_PLANES,
        NUM_BLOCKS,
        NUM_KV_HEADS,
        BLOCK_SIZE,
    )


def test_carve_splits_the_page_budget_without_overlap():
    buffer = _page_budget()
    raw_k, raw_v, raw_scale = turboquant_carve_planes(buffer, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)

    packed = NUM_BLOCKS * turboquant_packed_plane_bytes(BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    scales = NUM_BLOCKS * turboquant_scale_plane_bytes(BLOCK_SIZE, NUM_KV_HEADS)
    assert (raw_k.numel(), raw_v.numel(), raw_scale.numel()) == (packed, packed, scales)
    assert raw_k.numel() + raw_v.numel() + raw_scale.numel() == buffer.numel()

    base = buffer.data_ptr()
    assert [plane.data_ptr() - base for plane in (raw_k, raw_v, raw_scale)] == [0, packed, 2 * packed]

    # Each plane is a window on the one allocation, and only its own window.
    raw_k.fill_(1)
    raw_scale.fill_(3)
    assert int(raw_v.abs().sum()) == 0
    assert int(buffer[:packed].min()) == 1
    assert int(buffer[2 * packed :].min()) == 3


def test_carve_leaves_a_padded_tail_untouched():
    """A hybrid model pads the page to the common size; the tail stays unused."""
    needed = NUM_BLOCKS * turboquant_page_size_bytes(BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    padding = 4096
    buffer = torch.zeros(needed + padding, dtype=torch.int8)
    planes = turboquant_carve_planes(buffer, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)

    assert sum(plane.numel() for plane in planes) == needed
    for plane in planes:
        plane.fill_(7)
    assert int(buffer[needed:].abs().sum()) == 0


def test_carve_rejects_an_allocation_it_cannot_serve():
    short = torch.zeros(
        NUM_BLOCKS * turboquant_page_size_bytes(BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE) - 1,
        dtype=torch.int8,
    )
    with pytest.raises(ValueError, match="three-plane carve needs"):
        turboquant_carve_planes(short, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    with pytest.raises(ValueError, match="raw int8 bytes"):
        turboquant_carve_planes(_page_budget().view(torch.uint8), NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    with pytest.raises(ValueError, match="contiguous 1-D"):
        turboquant_carve_planes(_page_budget()[::2], NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    with pytest.raises(ValueError, match="at least one block"):
        turboquant_carve_planes(_page_budget(), 0, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)


def test_plane_views_write_through_to_the_carved_bytes():
    buffer = _page_budget()
    planes = turboquant_carve_planes(buffer, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    key, value, scale = turboquant_plane_views(planes, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)

    packed_shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE // TURBOQUANT_PACK_FACTOR)
    assert tuple(key.shape) == tuple(value.shape) == packed_shape
    assert (key.dtype, value.dtype, scale.dtype) == (torch.int8, torch.int8, torch.float32)
    # What the backend asserts about the allocator's third plane.
    assert tuple(scale.shape) == turboquant_scale_cache_shape(key.shape[0], key.shape[1], key.shape[2])

    key[1, 2, 0, 3] = 5
    scale[1, 2, 0, 3] = 0.5
    assert int(planes[0][((1 * BLOCK_SIZE + 2) * NUM_KV_HEADS + 0) * (HEAD_SIZE // 2) + 3]) == 5
    assert int(planes[1].abs().sum()) == 0
    assert float(planes[2].view(torch.float32)[((1 * NUM_BLOCKS + 2) * NUM_KV_HEADS + 0) * BLOCK_SIZE + 3]) == 0.5


def test_plane_views_follow_the_kernel_block_size():
    """A scheduler block larger than the kernel block is viewed as kernel blocks."""
    scheduler_block_size = 4 * BLOCK_SIZE
    buffer = _page_budget(num_blocks=NUM_BLOCKS, block_size=scheduler_block_size)
    planes = turboquant_carve_planes(buffer, NUM_BLOCKS, scheduler_block_size, NUM_KV_HEADS, HEAD_SIZE)
    key, _, scale = turboquant_plane_views(planes, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)

    assert tuple(key.shape) == (4 * NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE // TURBOQUANT_PACK_FACTOR)
    assert tuple(scale.shape) == (TURBOQUANT_PLANES, 4 * NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE)
    assert key.numel() == planes[0].numel()


def test_plane_views_reject_a_mismatched_carve():
    planes = turboquant_carve_planes(_page_budget(), NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    with pytest.raises(ValueError, match="three planes"):
        turboquant_plane_views(planes[:2], BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    with pytest.raises(ValueError, match="differ in size"):
        turboquant_plane_views((planes[0], planes[1][:-32], planes[2]), BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    with pytest.raises(ValueError, match="whole number of pages"):
        turboquant_plane_views((planes[0][:-32], planes[1][:-32], planes[2]), BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
