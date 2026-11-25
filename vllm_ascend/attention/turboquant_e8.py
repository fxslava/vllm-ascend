# SPDX-License-Identifier: Apache-2.0
"""Prepared Wave-3 E8 operations, separate from native BITS == 4 attention.

D256 packets contain 64 rows: 32 little-endian UINT16 labels per row
(two bits per coordinate), followed by 64 FP32 scales at byte 4096.
The reader takes six cache-basis FP32 query heads padded to 16 rows. Query
padding and packet byte offsets must be prepared before entering replay.
There is no verified raw BF16 encoder or automatic E8 attention promotion.
"""

from __future__ import annotations

from collections.abc import Sequence
from operator import index

import torch

E8_PACKET_BYTES = 4352
E8_SCALE_OFFSET = 4096


def pack_packets(labels: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pack already encoded labels on CPU, outside replay; this is no encoder."""
    if labels.device.type != "cpu" or scales.device.type != "cpu":
        raise ValueError("E8 packet preparation requires CPU tensors")
    if labels.dtype != torch.uint16 or labels.ndim != 3 or labels.shape[1:] != (64, 32):
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
    labels = packets[:, :E8_SCALE_OFFSET].view(torch.uint16).reshape(-1, 64, 32)
    scales = packets[:, E8_SCALE_OFFSET:].view(torch.float32)
    return labels, scales


def packet_byte_offsets(packet_ids: Sequence[int], packet_count: int) -> torch.Tensor:
    """Prepare validated INT64 offsets on the CPU, outside the runtime path."""
    try:
        packet_ids = [index(p) for p in packet_ids]
        packet_count = index(packet_count)
    except TypeError as exc:
        raise ValueError("E8 packet ids and packet count must be integers") from exc
    if not 4 <= len(packet_ids) <= 256 or len(packet_ids) % 2:
        raise ValueError("E8 requires an even packet count in [4, 256]")
    if packet_count <= 0 or any(p < 0 or p >= packet_count for p in packet_ids):
        raise ValueError("E8 packet id outside its plane")
    offsets = [p * E8_PACKET_BYTES for p in packet_ids]
    offsets.extend([0] * (-len(offsets) % 4))
    return torch.tensor(offsets, dtype=torch.int64)


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
