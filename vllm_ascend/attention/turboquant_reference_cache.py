# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch reference writes, gathers and indexing for TurboQuant cache layouts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from vllm_ascend.attention.turboquant_codec import TurboQuantEmulator
from vllm_ascend.attention.turboquant_config import (
    TURBOQUANT_NZ_FP32_PER_BLOCK,
    TURBOQUANT_NZ_OPERAND_C0,
    TURBOQUANT_NZ_TILE_ROWS,
    TURBOQUANT_STRIP_TOKENS,
    TurboQuantValueLayout,
)
from vllm_ascend.attention.turboquant_layout import (
    E8_HEAD_SIZE,
    E8_LABEL_ROW_BYTES,
    E8_LABELS_PER_ROW,
    E8_PACKET_BYTES,
    E8_PACKET_ROWS,
    E8_SCALE_OFFSET,
    TURBOQUANT_KEY_PLANE,
    TURBOQUANT_PACK_FACTOR,
    TURBOQUANT_SCALE_DTYPE_SIZE,
    TURBOQUANT_VALUE_PLANE,
)

E8_LATTICE_DIMENSIONS = 8
E8_CODEBOOK_SIZE = 1 << 16
_E8_GOLDEN_QUERY_CHUNK = 16


def e8_label_coordinates(labels: torch.Tensor) -> torch.Tensor:
    """Decode each UINT16 label into eight doubled lattice coordinates."""
    labels = labels.to(torch.int32)
    a = 2 * ((labels >> 14) & 1) + ((labels >> 15) & 1) - 2
    b, c, d, e, f, g, h = (((labels >> bit) & 3) - 2 for bit in range(0, 14, 2))
    projected = torch.stack(
        (
            a,
            a + 2 * b,
            a - 2 * b + 2 * c,
            a - 2 * c + 2 * d,
            a - 2 * d + 2 * e,
            a - 2 * e + 2 * f,
            a - 2 * f + 2 * g + 2 * h,
            a - 2 * g + 2 * h,
        ),
        dim=-1,
    )
    shifted = projected + 4
    integer = (shifted & 7) - 4
    half = torch.where(integer < 0, integer + 4, integer - 4)
    parity = torch.sum((shifted >> 3) & 1, dim=-1, keepdim=True, dtype=torch.int32) & 1
    half_parity = parity ^ (torch.sum(integer < 0, dim=-1, keepdim=True, dtype=torch.int32) & 1)
    coords = torch.arange(E8_LATTICE_DIMENSIONS, device=labels.device, dtype=torch.int32)

    def repair(candidate: torch.Tensor, parity: torch.Tensor) -> torch.Tensor:
        positive = candidate >= 0
        # The last negative tie or first nonnegative tie wins, with nonnegative preferred.
        priority = (candidate.abs() << 8) | torch.where(positive, 128 + (7 - coords), coords)
        winner = priority.argmax(dim=-1, keepdim=True)
        return candidate + (coords == winner) * parity * torch.where(positive, -8, 8)

    integer = repair(integer, parity)
    half = repair(half, half_parity)
    int_dist = (integer * integer).sum(dim=-1, dtype=torch.int32)
    half_dist = (half * half).sum(dim=-1, dtype=torch.int32)
    use_half = (half_dist < int_dist) | ((half_dist == int_dist) & (half[..., 0] < integer[..., 0]))
    return torch.where(use_half.unsqueeze(-1), half, integer)


def decode_e8_labels(
    labels: torch.Tensor, scales: torch.Tensor, *, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Decode packet rows in cache basis using the SIMT component's rounding."""
    labels = labels.view(torch.int16) if labels.dtype == torch.uint16 else labels
    coordinates = e8_label_coordinates(labels).flatten(-2)
    values = (coordinates.to(torch.float32) * 0.5) * scales.unsqueeze(-1)
    if dtype == torch.int8:
        return values.clamp(-128, 127).round().to(torch.int8)
    return values.to(dtype)


def encode_e8_rotated(rotated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU golden for r3/revision2's unnormalized rotated FP32 input rows."""
    if rotated.device.type != "cpu" or rotated.shape[-1:] != (E8_HEAD_SIZE,):
        raise ValueError(f"E8 golden encoding needs CPU D={E8_HEAD_SIZE} rotated rows")
    values = rotated.detach().to(torch.float32).reshape(-1, E8_HEAD_SIZE).numpy() * np.float32(1 / 16)
    if not np.isfinite(values).all():
        raise ValueError("E8 golden encoding requires finite rows")
    squares = np.square(values).reshape(-1, 2, 16, E8_LATTICE_DIMENSIONS)
    sums = np.zeros_like(squares[:, :, 0])
    for group in range(squares.shape[2]):
        sums += squares[:, :, group]
    sums = sums.reshape(-1, 2 * E8_LATTICE_DIMENSIONS)
    while sums.shape[-1] > 1:
        sums = sums[:, 0::2] + sums[:, 1::2]
    scales = np.sqrt(sums[:, 0] * np.float32(1 / E8_HEAD_SIZE))
    scales[scales == 0] = 1
    if not np.isfinite(scales).all():
        raise ValueError("E8 golden encoding requires finite FP32 scales")
    queries = (values / scales[:, None]).reshape(-1, E8_LATTICE_DIMENSIONS)
    points = e8_label_coordinates(torch.arange(E8_CODEBOOK_SIZE, dtype=torch.int32, device="cpu"))
    points = points.numpy().astype(np.float64) * 0.5
    labels = np.empty(len(queries), dtype=np.uint16)
    for begin in range(0, len(queries), _E8_GOLDEN_QUERY_CHUNK):
        end = begin + _E8_GOLDEN_QUERY_CHUNK
        distance = np.square(queries[begin:end, None, :] - points[None, :, :])
        # Adjacent-pair FP64 summation and the smallest label on ties are part of the golden.
        while distance.shape[-1] > 1:
            distance = distance[..., 0::2] + distance[..., 1::2]
        labels[begin:end] = distance[..., 0].argmin(axis=1)
    return (
        torch.from_numpy(labels).reshape(*rotated.shape[:-1], E8_LABELS_PER_ROW),
        torch.from_numpy(scales).reshape(rotated.shape[:-1]),
    )


def write_e8_slots(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    emulator: TurboQuantEmulator,
) -> None:
    """Encode CPU K/V rows into their packet offsets, preserving untouched rows."""
    if any(t.device.type != "cpu" for t in (key, value, key_cache, value_cache, slot_mapping)):
        raise RuntimeError("TurboQuant E8 device ingress is gated; load host-packed E8 packets to use the reader")
    blocks, tiles, heads, _ = key_cache.shape
    if key.shape != value.shape or key.shape != (slot_mapping.numel(), heads, E8_HEAD_SIZE):
        raise ValueError("E8 K/V rows and slot mapping must agree with the cache geometry")
    keep = slot_mapping >= 0
    slots = slot_mapping[keep].to(torch.int64)
    if slots.numel() == 0:
        return
    if int(slots.max()) >= blocks * tiles * E8_PACKET_ROWS or slots.unique().numel() != slots.numel():
        raise ValueError("E8 slots must be unique and within cache capacity")
    packet = ((slots // E8_PACKET_ROWS) * heads)[:, None] + torch.arange(heads, dtype=torch.int64, device="cpu")
    row = (slots % E8_PACKET_ROWS)[:, None, None]
    label_pos = row * E8_LABEL_ROW_BYTES + torch.arange(E8_LABEL_ROW_BYTES, device="cpu")
    scale_pos = E8_SCALE_OFFSET + row * TURBOQUANT_SCALE_DTYPE_SIZE
    scale_pos = scale_pos + torch.arange(TURBOQUANT_SCALE_DTYPE_SIZE, device="cpu")
    labels, scales = encode_e8_rotated(emulator.rotate(torch.stack((key[keep], value[keep]))))
    for cache, plane_labels, plane_scales in zip((key_cache, value_cache), labels, scales):
        packets = cache.view(-1, E8_PACKET_BYTES)
        packets[packet.unsqueeze(-1), label_pos] = plane_labels.contiguous().view(torch.uint8)
        packets[packet.unsqueeze(-1), scale_pos] = plane_scales.unsqueeze(-1).view(torch.uint8)


def write_slots(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    emulator: TurboQuantEmulator,
    *,
    rotate: Callable[[torch.Tensor, TurboQuantEmulator], torch.Tensor],
    layout: TurboQuantValueLayout,
    lloyd_max_folds: tuple[str, str] = ("key", "value"),
) -> None:
    codes_key, key_scales = emulator.quantize(rotate(key, emulator), lloyd_max_folds[0])
    codes_value, value_scales = emulator.quantize(rotate(value, emulator), lloyd_max_folds[1])

    block_size = key_cache.shape[1]
    slots = slot_mapping.to(torch.int64)
    valid = slots >= 0
    if not bool(valid.all()):
        slots = slots[valid]
        codes_key = codes_key[valid]
        codes_value = codes_value[valid]
        key_scales = key_scales[valid]
        value_scales = value_scales[valid]
    if slots.numel() == 0:
        return

    num_blocks = key_cache.shape[0]
    slot_capacity = num_blocks * block_size
    max_slot = int(slots.max())
    if max_slot >= slot_capacity:
        raise ValueError(
            f"slot_mapping is out of range for the TurboQuant cache: expected slots in "
            f"[0, {slot_capacity}) ({num_blocks} blocks activations {block_size} tokens, from key_cache.shape="
            f"{tuple(key_cache.shape)}), but the largest non-negative slot is {max_slot}."
        )

    pages = torch.div(slots, block_size, rounding_mode="floor")
    offsets = slots - pages * block_size

    if layout is TurboQuantValueLayout.CHANNEL_MAJOR:
        write_strip_planes(codes_key, codes_value, key_cache, value_cache, pages, offsets)
    elif layout is TurboQuantValueLayout.NZ_TILED:
        _write_nz_planes(codes_key, codes_value, key_cache, value_cache, slots)
    else:
        for packed, cache in (
            (emulator.pack(codes_key), key_cache),
            (emulator.pack(codes_value), value_cache),
        ):
            _flat_slot_view(cache)[slots] = packed

    if layout is TurboQuantValueLayout.NZ_TILED:
        _write_nz_scales(key_scales, value_scales, scale_cache, slots)
    else:
        for plane, scales in (
            (TURBOQUANT_KEY_PLANE, key_scales),
            (TURBOQUANT_VALUE_PLANE, value_scales),
        ):
            scale_cache[plane][pages, :, offsets] = scales.to(scale_cache.dtype)


def write_strip_planes(
    codes_key: torch.Tensor,
    codes_value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
) -> None:
    num_blocks, block_size, num_kv_heads, packed_bytes = key_cache.shape
    head_size = packed_bytes * TURBOQUANT_PACK_FACTOR
    if block_size % TURBOQUANT_STRIP_TOKENS:
        raise ValueError(
            f"the strip cache is written 16 tokens at a time, so block_size {block_size} must be a "
            f"multiple of {TURBOQUANT_STRIP_TOKENS}"
        )
    strips_per_page = block_size // TURBOQUANT_STRIP_TOKENS

    key_pages = _require_contiguous(key_cache, "key").view(num_blocks, num_kv_heads, block_size, packed_bytes)
    key_pages[pages, :, offsets] = TurboQuantEmulator.pack(codes_key)

    strips = torch.div(offsets, TURBOQUANT_STRIP_TOKENS, rounding_mode="floor")
    lanes = offsets - strips * TURBOQUANT_STRIP_TOKENS
    value_pages = _require_contiguous(value_cache, "value").view(
        num_blocks, num_kv_heads, strips_per_page, head_size, TURBOQUANT_STRIP_TOKENS // TURBOQUANT_PACK_FACTOR
    )
    touched, inverse = torch.unique(pages, return_inverse=True)
    panel = TurboQuantEmulator.unpack(value_pages[touched])
    panel[inverse, :, strips, :, lanes] = codes_value
    value_pages[touched] = TurboQuantEmulator.pack(panel)


def nz_byte_offsets(slots: torch.Tensor, num_kv_heads: int, packed_bytes: int) -> torch.Tensor:
    tile_rows = TURBOQUANT_NZ_TILE_ROWS
    c0 = TURBOQUANT_NZ_OPERAND_C0
    groups = packed_bytes // c0
    device = slots.device
    tile_row = slots % tile_rows
    base = (slots - tile_row) * (num_kv_heads * packed_bytes) + tile_row * c0
    heads = torch.arange(num_kv_heads, device=device) * (tile_rows * packed_bytes)
    group = torch.arange(groups, device=device) * (tile_rows * c0)
    byte = torch.arange(c0, device=device)
    return base[..., None, None, None] + heads.view(-1, 1, 1) + group.view(-1, 1) + byte


def _write_nz_planes(
    codes_key: torch.Tensor,
    codes_value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    _, block_size, num_kv_heads, packed_bytes = key_cache.shape
    head_size = packed_bytes * TURBOQUANT_PACK_FACTOR
    if block_size % TURBOQUANT_NZ_TILE_ROWS:
        raise ValueError(
            f"the NZ-tiled cache is laid out in {TURBOQUANT_NZ_TILE_ROWS}-row tiles, so block_size "
            f"{block_size} must be a multiple of {TURBOQUANT_NZ_TILE_ROWS}"
        )
    offsets = nz_byte_offsets(slots, num_kv_heads, packed_bytes)

    def nz_bytes(codes: torch.Tensor) -> torch.Tensor:
        low, high = codes[..., :packed_bytes], codes[..., packed_bytes:]
        woven = torch.stack((low, high), dim=-1).reshape(*codes.shape[:-1], head_size)
        return TurboQuantEmulator.pack(woven).reshape(*codes.shape[:-1], -1, TURBOQUANT_NZ_OPERAND_C0)

    for codes, cache, name in (
        (codes_key, key_cache, "key"),
        (codes_value, value_cache, "value"),
    ):
        _require_contiguous(cache, name).view(-1)[offsets] = nz_bytes(codes).to(cache.dtype)


def nz_scale_slot_floats(num_kv_heads: int) -> int:
    lanes = TURBOQUANT_PACK_FACTOR * num_kv_heads
    return -(-lanes // TURBOQUANT_NZ_FP32_PER_BLOCK) * TURBOQUANT_NZ_FP32_PER_BLOCK


def _write_nz_scales(
    key_scales: torch.Tensor,
    value_scales: torch.Tensor,
    scale_cache: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    num_kv_heads = scale_cache.shape[2]
    slot_floats = nz_scale_slot_floats(num_kv_heads)
    flat = _require_contiguous(scale_cache, "scale").view(-1)
    total_slots = scale_cache.shape[1] * scale_cache.shape[3]
    if flat.numel() < total_slots * slot_floats:
        raise ValueError(
            f"the NZ scale plane needs a whole {slot_floats}-float record per slot "
            f"(NzScaleSlotFloats({num_kv_heads})), {total_slots * slot_floats} floats over the cache; "
            f"the allocated plane holds {flat.numel()}, which is exact only when 2 * num_kv_heads "
            f"is a multiple of {TURBOQUANT_NZ_FP32_PER_BLOCK}"
        )
    lanes = torch.arange(num_kv_heads, device=slots.device)
    records = slots[:, None] * slot_floats + lanes
    flat[records] = key_scales.to(scale_cache.dtype)
    flat[records + num_kv_heads] = value_scales.to(scale_cache.dtype)


def gather_pages(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_tables: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block_size = key_cache.shape[1]
    pages = block_tables.to(torch.int64)
    batch_size, max_blocks = pages.shape
    within = torch.arange(block_size, device=pages.device, dtype=torch.int64)
    slots = (pages.unsqueeze(-1) * block_size + within).reshape(batch_size, max_blocks * block_size)

    packed_key = _flat_slot_view(key_cache)[slots]
    packed_value = _flat_slot_view(value_cache)[slots]
    num_kv_heads = scale_cache.shape[2]

    def gathered_scales(plane: int) -> torch.Tensor:
        return scale_cache[plane][pages].transpose(-1, -2).reshape(batch_size, -1, num_kv_heads)

    return (
        packed_key,
        gathered_scales(TURBOQUANT_KEY_PLANE),
        packed_value,
        gathered_scales(TURBOQUANT_VALUE_PLANE),
    )


def gather_pages_strip(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_tables: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks, block_size, num_kv_heads, packed_bytes = key_cache.shape
    head_size = packed_bytes * TURBOQUANT_PACK_FACTOR
    if block_size % TURBOQUANT_STRIP_TOKENS:
        raise ValueError(
            f"the strip cache is laid out 16 tokens at a time, so block_size {block_size} must be a "
            f"multiple of {TURBOQUANT_STRIP_TOKENS}"
        )
    strips_per_page = block_size // TURBOQUANT_STRIP_TOKENS
    pages = block_tables.to(torch.int64)
    batch_size = pages.shape[0]

    key_pages = _require_contiguous(key_cache, "key").view(num_blocks, num_kv_heads, block_size, packed_bytes)
    packed_key = key_pages[pages].permute(0, 1, 3, 2, 4).reshape(batch_size, -1, num_kv_heads, packed_bytes)

    value_pages = _require_contiguous(value_cache, "value").view(
        num_blocks, num_kv_heads, strips_per_page, head_size, TURBOQUANT_STRIP_TOKENS // TURBOQUANT_PACK_FACTOR
    )
    codes = TurboQuantEmulator.unpack(value_pages[pages])
    token_major = codes.permute(0, 1, 3, 5, 2, 4).reshape(batch_size, -1, num_kv_heads, head_size)
    packed_value = TurboQuantEmulator.pack(token_major.contiguous())

    def gathered_scales(plane: int) -> torch.Tensor:
        return scale_cache[plane][pages].transpose(-1, -2).reshape(batch_size, -1, num_kv_heads)

    return (
        packed_key,
        gathered_scales(TURBOQUANT_KEY_PLANE),
        packed_value,
        gathered_scales(TURBOQUANT_VALUE_PLANE),
    )


def gather_pages_nz(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_tables: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, block_size, num_kv_heads, packed_bytes = key_cache.shape
    if block_size % TURBOQUANT_NZ_TILE_ROWS:
        raise ValueError(
            f"the NZ-tiled cache is laid out in {TURBOQUANT_NZ_TILE_ROWS}-row tiles, so block_size "
            f"{block_size} must be a multiple of {TURBOQUANT_NZ_TILE_ROWS}"
        )
    pages = block_tables.to(torch.int64)
    batch_size, max_blocks = pages.shape
    within = torch.arange(block_size, device=pages.device, dtype=torch.int64)
    slots = (pages.unsqueeze(-1) * block_size + within).reshape(batch_size, max_blocks * block_size)

    offsets = nz_byte_offsets(slots, num_kv_heads, packed_bytes)

    def read_plane(cache: torch.Tensor, name: str) -> torch.Tensor:
        gathered = _require_contiguous(cache, name).view(-1)[offsets]
        packed = gathered.reshape(batch_size, -1, num_kv_heads, packed_bytes)
        woven = TurboQuantEmulator.unpack(packed)
        codes = torch.cat((woven[..., 0::2], woven[..., 1::2]), dim=-1)
        return TurboQuantEmulator.pack(codes.contiguous())

    slot_floats = nz_scale_slot_floats(num_kv_heads)
    flat_scales = _require_contiguous(scale_cache, "scale").view(-1)
    lanes = torch.arange(num_kv_heads, device=slots.device)
    records = slots[..., None] * slot_floats + lanes

    return (
        read_plane(key_cache, "key"),
        flat_scales[records],
        read_plane(value_cache, "value"),
        flat_scales[records + num_kv_heads],
    )


def _flat_slot_view(cache: torch.Tensor) -> torch.Tensor:
    if not cache.is_contiguous():
        raise RuntimeError(
            "the TurboQuant packed cache plane must be contiguous to be addressed by slot; "
            f"got strides {tuple(cache.stride())} for shape {tuple(cache.shape)}"
        )
    return cache.view(cache.shape[0] * cache.shape[1], *cache.shape[2:])


def _require_contiguous(cache: torch.Tensor, name: str) -> torch.Tensor:
    if not cache.is_contiguous():
        raise RuntimeError(
            f"the TurboQuant {name} plane must be contiguous to be addressed by page; "
            f"got strides {tuple(cache.stride())} for shape {tuple(cache.shape)}"
        )
    return cache
