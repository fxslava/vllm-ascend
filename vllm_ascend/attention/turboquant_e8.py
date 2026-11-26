# SPDX-License-Identifier: Apache-2.0
"""Prepared operations for the D256 TurboQuant E8 packet cache.

D256 packets contain 64 rows: 32 little-endian UINT16 labels per row
(two bits per coordinate), followed by 64 FP32 scales at byte 4096.
The reader takes six cache-basis FP32 query heads padded to 16 rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index

import torch

from vllm_ascend.attention.turboquant_layout import (
    E8_HEAD_SIZE,
    E8_LABELS_PER_ROW,
    E8_PACKET_ROWS,
)
from vllm_ascend.attention.turboquant_layout import E8_PACKET_BYTES as E8_PACKET_BYTES
from vllm_ascend.attention.turboquant_layout import E8_SCALE_OFFSET as E8_SCALE_OFFSET

E8_OPS_NAMESPACE = "turboquant_e8_ops"
E8_READER_HEADS = 6
E8_READER_PADDED_HEADS = 16
E8_READER_MIN_TILES = 4
E8_READER_MAX_TILES = 256
E8_DEQUANT_ROWS = 16


@dataclass(frozen=True)
class E8RequestPlan:
    query_start: int
    query_end: int
    context_length: int
    packet_ids: torch.Tensor
    reader_offsets: torch.Tensor | None


@dataclass(frozen=True)
class E8ReaderBuffers:
    query: torch.Tensor
    output: torch.Tensor
    key_witness: torch.Tensor
    value_witness: torch.Tensor
    score_witness: torch.Tensor


@dataclass(frozen=True)
class E8BatchPlan:
    requests: tuple[E8RequestPlan, ...]
    num_blocks: int
    block_size: int
    reader_buffers: E8ReaderBuffers | None


def prepare_batch(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    *,
    num_tokens: int,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    device: torch.device,
    use_reader: bool,
) -> E8BatchPlan:
    """Prepare valid physical packet IDs and aligned reader offsets from the scheduler's CPU mirrors."""
    if any(x.device.type != "cpu" for x in (block_table, seq_lens, query_lens)):
        raise ValueError("E8 metadata requires CPU block-table and length mirrors")
    if block_size <= 0 or block_size % E8_PACKET_ROWS or num_blocks <= 0 or num_kv_heads <= 0 or num_tokens < 0:
        raise ValueError("Invalid E8 cache geometry")
    if block_table.ndim != 2 or block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("E8 block tables must be two-dimensional integer tensors")
    tiles_per_block = block_size // E8_PACKET_ROWS
    if num_blocks * tiles_per_block * num_kv_heads * E8_PACKET_BYTES > torch.iinfo(torch.int64).max:
        raise ValueError("E8 cache byte offsets exceed INT64 capacity")
    tile_capacity = block_table.size(1) * tiles_per_block
    chunks = []
    descriptors = []
    query_start = 0
    max_reader_tiles = 0
    uploaded_words = 0
    context_lengths = seq_lens.tolist()
    num_request_rows = min(len(context_lengths), block_table.size(0))
    for request, queries in enumerate(query_lens.tolist()):
        if query_start == num_tokens:
            # Sequence parallelism can append a dummy query row beyond the CPU block table.
            break
        if queries < 0:
            raise ValueError(f"Invalid E8 query length for request {request}")
        if queries == 0:
            continue
        if request >= num_request_rows:
            raise ValueError(f"E8 live request {request} requires a block table row and context length")
        context = context_lengths[request]
        query_end = min(query_start + queries, num_tokens)
        if not query_end - query_start <= context <= tile_capacity * E8_PACKET_ROWS:
            raise ValueError(f"Invalid E8 context length for request {request}")
        tiles = (context + E8_PACKET_ROWS - 1) // E8_PACKET_ROWS
        logical_tiles = torch.arange(tiles, dtype=torch.int64, device="cpu")
        physical_blocks = block_table[request, logical_tiles // tiles_per_block].to(torch.int64)
        if bool(((physical_blocks < 0) | (physical_blocks >= num_blocks)).any()):
            raise ValueError(f"E8 request {request} uses a block outside the cache")
        physical_tiles = physical_blocks * tiles_per_block + logical_tiles % tiles_per_block
        packet_ids = physical_tiles[:, None] * num_kv_heads + torch.arange(num_kv_heads, device="cpu")
        chunks.append(packet_ids.reshape(-1))
        uploaded_words += packet_ids.numel()
        offset_width = 0
        if use_reader and query_end - query_start == 1 and reader_supports_tiles(tiles):
            max_reader_tiles = max(tiles, max_reader_tiles)
            offset_width = (tiles + 3) // 4 * 4
            # Each head row and each request starts on a 32-byte DMA boundary.
            padding = -uploaded_words % 4
            if padding:
                chunks.append(torch.zeros(padding, dtype=torch.int64, device="cpu"))
            reader_offsets = torch.zeros((num_kv_heads, offset_width), dtype=torch.int64, device="cpu")
            reader_offsets[:, :tiles] = packet_ids.T * E8_PACKET_BYTES
            chunks.append(reader_offsets.flatten())
            uploaded_words += padding + reader_offsets.numel()
        descriptors.append((query_start, query_end, context, tiles, offset_width))
        query_start = query_end
    if query_start != num_tokens:
        raise ValueError("E8 query lengths do not cover the live token count")

    requests = []
    if chunks:
        upload = torch.cat(chunks)
        if device.type != "cpu":
            upload = upload.pin_memory()
            upload = upload.to(device, non_blocking=upload.is_pinned())
        start = 0
        for query_start, query_end, context, tiles, offset_width in descriptors:
            packet_ids = upload.narrow(0, start, tiles * num_kv_heads).view(tiles, num_kv_heads)
            start += packet_ids.numel()
            reader_offsets = None
            if offset_width:
                start += -start % 4
                reader_offsets = upload.narrow(0, start, num_kv_heads * offset_width).view(num_kv_heads, offset_width)
                start += reader_offsets.numel()
            requests.append(E8RequestPlan(query_start, query_end, context, packet_ids, reader_offsets))
    buffers = None
    if max_reader_tiles:
        witness_shape = (max_reader_tiles, E8_PACKET_ROWS, E8_HEAD_SIZE)
        buffers = E8ReaderBuffers(
            query=torch.empty((E8_READER_PADDED_HEADS, E8_HEAD_SIZE), dtype=torch.float32, device=device),
            output=torch.empty((E8_READER_HEADS, E8_HEAD_SIZE), dtype=torch.bfloat16, device=device),
            key_witness=torch.empty(witness_shape, dtype=torch.bfloat16, device=device),
            value_witness=torch.empty(witness_shape, dtype=torch.bfloat16, device=device),
            score_witness=torch.empty(
                (max_reader_tiles, E8_READER_HEADS, E8_PACKET_ROWS), dtype=torch.float32, device=device
            ),
        )
    return E8BatchPlan(tuple(requests), num_blocks, block_size, buffers)


def pack_packets(labels: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pack already encoded labels on CPU, outside replay; this is no encoder."""
    if labels.device.type != "cpu" or scales.device.type != "cpu":
        raise ValueError("E8 packet preparation requires CPU tensors")
    if labels.dtype != torch.uint16 or labels.ndim != 3 or labels.shape[1:] != (E8_PACKET_ROWS, E8_LABELS_PER_ROW):
        raise ValueError("E8 labels must be UINT16 [packets, 64, 32]")
    if scales.dtype != torch.float32 or scales.shape != labels.shape[:2]:
        raise ValueError("E8 scales must be FP32 [packets, 64]")
    packets = torch.empty((labels.shape[0], E8_PACKET_BYTES), dtype=torch.uint8)
    packets[:, :E8_SCALE_OFFSET] = labels.contiguous().view(torch.uint8).reshape(-1, E8_SCALE_OFFSET)
    packets[:, E8_SCALE_OFFSET:] = scales.contiguous().view(torch.uint8)
    return packets


def unpack_packets(packets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return label/scale views with the external 4352-byte packet stride."""
    if packets.dtype != torch.uint8 or packets.ndim != 2 or packets.shape[1] != E8_PACKET_BYTES:
        raise ValueError("E8 packets must be UINT8 [packets, 4352]")
    if not packets.is_contiguous():
        raise ValueError("E8 packets must be contiguous")
    labels = packets[:, :E8_SCALE_OFFSET].view(torch.uint16).reshape(-1, E8_PACKET_ROWS, E8_LABELS_PER_ROW)
    scales = packets[:, E8_SCALE_OFFSET:].view(torch.float32)
    return labels, scales


def packet_byte_offsets(packet_ids: Sequence[int], packet_count: int) -> torch.Tensor:
    """Prepare validated INT64 offsets on the CPU, outside the runtime path."""
    try:
        packet_ids = [index(p) for p in packet_ids]
        packet_count = index(packet_count)
    except TypeError as exc:
        raise ValueError("E8 packet ids and packet count must be integers") from exc
    if not reader_supports_tiles(len(packet_ids)):
        raise ValueError("E8 requires an even packet count in [4, 256]")
    if packet_count <= 0 or any(p < 0 or p >= packet_count for p in packet_ids):
        raise ValueError("E8 packet id outside its plane")
    offsets = [p * E8_PACKET_BYTES for p in packet_ids]
    offsets.extend([0] * (-len(offsets) % 4))
    return torch.tensor(offsets, dtype=torch.int64)


def reader_supports_tiles(tiles: int) -> bool:
    return E8_READER_MIN_TILES <= tiles <= E8_READER_MAX_TILES and tiles % 2 == 0


def reader(
    query: torch.Tensor,
    key_packets: torch.Tensor,
    value_packets: torch.Tensor,
    packet_offsets: torch.Tensor,
    output: torch.Tensor,
    key_witness: torch.Tensor,
    value_witness: torch.Tensor,
    score_witness: torch.Tensor,
    tiles: int,
    last_rows: int,
    scale: float,
) -> None:
    """Launch into caller-owned buffers; query/output remain in cache basis."""
    torch.ops.turboquant_e8_ops.reader(
        query,
        key_packets,
        value_packets,
        packet_offsets,
        output,
        key_witness,
        value_witness,
        score_witness,
        tiles,
        last_rows,
        scale,
    )


def dequant(labels: torch.Tensor, scales: torch.Tensor, bf16: torch.Tensor, int8: torch.Tensor) -> None:
    """Decode the verified 16-row frame into preallocated BF16 and INT8 outputs."""
    torch.ops.turboquant_e8_ops.dequant(labels, scales, bf16, int8)


def dequant_packet(packets: torch.Tensor, bf16: torch.Tensor, int8: torch.Tensor) -> None:
    """Decode whole E8 packets into caller-owned BF16 and INT8 tensors."""
    torch.ops.turboquant_e8_ops.dequant_packet(packets, bf16, int8)
