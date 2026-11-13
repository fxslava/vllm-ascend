# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused decode's ingress contract, and a bridge to it from the strip cache.

NOT a mirrored file. The kernels under ``csrc/attention/turboquant`` are a
verbatim mirror of the ascend-micro-harness and must not be edited here; this
module is the repo's own host-side statement of the layout the *fused decode*
requires, plus the conversion from the layout the two audited writers produce.

Why it exists
-------------

The two cache writers and the fused decode were built against two different
cache contracts, and nothing converted between them, so
``torch.ops.turboquant_ops.fused_decode`` could not read a page either writer
had written. The binding says so itself (``_kernel_fused_decode``'s docstring in
:mod:`vllm_ascend.attention.turboquant_dry_run`) and refuses the launch rather
than attending over bytes in the wrong order. They disagree on four independent
axes, every one of them read off production code rather than inferred:

1. **Byte order.** The writers lay a ``(page, head)`` panel out row-major,
   ``((page * numHeads + head) * blockSize + row) * headSize / 2``
   (the ``Layout contracts`` block of
   ``op_kernel/turboquant_reshape_batch.cpp``, restated for a host reader as
   ``key_nibble`` / ``value_nibble`` in
   ``tests/ut/ops/turboquant/common/turboquant_test_common.hpp``).
   The decode reads an NZ fractal, ``[tile][kvHead][columnGroup][tileRow][byte]``
   with a 64-row tile and 32-byte column groups (``NzTiledPackedByte`` in
   ``op_kernel/common/turboquant_layout.h``, issued as strided DMA by
   ``ReadPlane`` in ``op_kernel/vector/turboquant_vector_service.h``).

2. **Nibble pairing.** The writers pair *adjacent channels* -- channel ``2c`` in
   the low nibble of byte ``c`` and ``2c + 1`` in the high one. The decode pairs
   channel ``c`` with channel ``c + headSize / 2``: ``UnpackAffine``
   (``op_kernel/common/turboquant_codec_mx.h``) widens a tile's bytes and
   DeInterleaves them into a low plane and a high plane of equal length, and
   ``StageNzTileToL1`` stages the low plane on NZ groups ``[0, groups)`` and the
   high plane on ``[groups, 2 * groups)``. So packed column ``col`` carries
   channel ``col`` low and channel ``col + 128`` high.

   This is the pairing ``NzTiledChannelByte`` and ``NzTiledChannelNibble`` in
   ``op_kernel/common/turboquant_layout.h`` describe. They used to be left out
   of that header, on the grounds that a reader might take them for the
   *writers'* contract; they are in it now, because
   ``tests/ut/ops/turboquant/test_layout_bridge.cpp`` is a device-free gate that
   states both contracts in C++ and proves their co-byte pairings structurally
   incompatible. A test says it, so a comment no longer has to.

3. **The value plane's token axis.** The strip cache stores V channel-major
   inside a 16-token strip, nibble
   ``(token // 16) * 16 * headSize + channel * 16 + token % 16``
   (``value_nibble``). The decode reads both planes through the *same* NZ
   addressing, so that transposition is a strip-writer artifact and is undone
   here.

4. **The scale plane.** The writers produce a plane-major
   ``(2, num_blocks, num_kv_heads, block_size)`` buffer with K and V in separate
   planes (``turboquant_scale_cache_shape``). ``ReadPlane`` reads
   ``scaleCacheGm_[row * scaleSlotFloats]`` for ``kCubeTileRows *
   scaleSlotFloats`` floats, i.e. one contiguous per-slot record holding *both*
   lanes, ``slot * scaleSlotFloats + kv * numKvHeads + kvHead``, with
   ``scaleSlotFloats = ceil(2 * numKvHeads / 8) * 8``. The two buffers are not
   even the same size when ``2 * num_kv_heads`` is not a multiple of 8.

What this module claims, and what it does not
---------------------------------------------

:func:`turboquant_repack_strip_to_nz_tiled` is a reference conversion. It makes
the fused decode reachable from a cache the audited writers produced, and it is
the executable specification that a write-time NZ ingress kernel has to match.
That kernel has landed: the mirrored batch writer
(``csrc/attention/turboquant/op_kernel/turboquant_reshape_batch.cpp``) is built
with ``TURBOQUANT_INGRESS_NZ=1`` and its NZ addressing is tied to
``turboquant_layout.h`` by ``static_assert``. This module remains the bridge
for caches written before the ingress existed and the host-side oracle for the
contract. This is not a serving-path optimisation: it is a whole extra pass
over the pages it touches, and the scatter writer's one token per launch would
make a per-step repack absurd. Use it to exercise and gate the decode, not to
serve.

``block_size`` has to satisfy both sides at once: a multiple of 16 for the
writers and a power-of-two multiple of 64 for the decode's closed-form tile
cursor. So 64, 128, 256, ... qualify, and 16 and 32 do not.
"""

from __future__ import annotations

import torch

from vllm_ascend.attention.turboquant_dry_run import (
    TURBOQUANT_PACK_FACTOR,
    TURBOQUANT_STRIP_TOKENS,
    TurboQuantEmulator,
)

__all__ = [
    "TURBOQUANT_NZ_OPERAND_C0",
    "TURBOQUANT_NZ_TILE_ROWS",
    "turboquant_codes_to_nz_tiled",
    "turboquant_nz_channel_nibble",
    "turboquant_nz_packed_byte",
    "turboquant_nz_scale_index",
    "turboquant_nz_scale_slot_floats",
    "turboquant_repack_strip_to_nz_tiled",
    "turboquant_scales_to_nz_tiled",
    "turboquant_strip_cache_to_codes",
]

# kCubeTileRows and kOperandC0 of op_kernel/common/turboquant_layout.h. Restated
# rather than imported because that header is C++;
# tests/ut/attention/test_turboquant_nz_ingress.py reads the header and fails if
# either constant drifts from it.
TURBOQUANT_NZ_TILE_ROWS: int = 64
TURBOQUANT_NZ_OPERAND_C0: int = 32

# kFp32PerBlock, the fp32 block a scale record is rounded up to.
TURBOQUANT_FP32_PER_BLOCK: int = 8


# ---------------------------------------------------------------------------
# The decode's addressing, as scalar formulas
# ---------------------------------------------------------------------------
#
# The host-readable statement of what ReadPlane issues. Deliberately scalar and
# allocation-free: the tests use these as the oracle for the vectorised repack
# below, so the two must not share code.


def turboquant_nz_packed_byte(slot: int, kv_head: int, column: int, num_kv_heads: int, packed_bytes: int) -> int:
    """Flat byte index of packed ``column`` of ``(slot, kv_head)``.

    ``slot`` is the global row, ``page * block_size + token``. This is
    ``NzTiledPackedByte`` of ``op_kernel/common/turboquant_layout.h``.
    """
    tile_row = slot % TURBOQUANT_NZ_TILE_ROWS
    return (
        (slot - tile_row) * num_kv_heads * packed_bytes
        + kv_head * TURBOQUANT_NZ_TILE_ROWS * packed_bytes
        + ((column // TURBOQUANT_NZ_OPERAND_C0) * TURBOQUANT_NZ_TILE_ROWS + tile_row) * TURBOQUANT_NZ_OPERAND_C0
        + column % TURBOQUANT_NZ_OPERAND_C0
    )


def turboquant_nz_channel_nibble(channel: int, head_size: int) -> tuple[int, int]:
    """``(column, nibble)`` holding ``channel``; 0 is the low nibble, 1 the high.

    The woven pairing of ``UnpackAffine`` plus ``StageNzTileToL1``: channel
    ``c`` and channel ``c + head_size / 2`` share one byte.
    """
    packed_bytes = head_size // TURBOQUANT_PACK_FACTOR
    return channel % packed_bytes, 0 if channel < packed_bytes else 1


def turboquant_nz_scale_slot_floats(num_kv_heads: int) -> int:
    """``ScaleSlotFloats`` of ``op_kernel/common/turboquant_common.h``."""
    lanes = TURBOQUANT_PACK_FACTOR * num_kv_heads
    return -(-lanes // TURBOQUANT_FP32_PER_BLOCK) * TURBOQUANT_FP32_PER_BLOCK


def turboquant_nz_scale_index(slot: int, kv_head: int, kv: int, num_kv_heads: int) -> int:
    """Flat fp32 index of the ``kv`` lane of ``(slot, kv_head)``.

    ``kv`` is 0 for the key plane and 1 for the value plane. This is the index
    ``ReadPlane`` bursts from, ``scaleCacheGm_[row * scaleSlotFloats]``.
    """
    return slot * turboquant_nz_scale_slot_floats(num_kv_heads) + kv * num_kv_heads + kv_head


# ---------------------------------------------------------------------------
# The conversion
# ---------------------------------------------------------------------------


def _check_geometry(block_size: int, head_size: int) -> None:
    if block_size % TURBOQUANT_NZ_TILE_ROWS or block_size & (block_size - 1):
        raise ValueError(
            f"block_size {block_size} cannot be read by the fused decode: its closed-form tile "
            f"cursor turns a tile index into (page, row base) with one shift and one mask, so a "
            f"page must be a power-of-two multiple of {TURBOQUANT_NZ_TILE_ROWS}. The writers only "
            f"require a multiple of {TURBOQUANT_STRIP_TOKENS}, which is why a cache can be "
            f"writable and unreadable at once."
        )
    if head_size % (TURBOQUANT_PACK_FACTOR * TURBOQUANT_NZ_OPERAND_C0):
        raise ValueError(
            f"head_size {head_size} must be a multiple of "
            f"{TURBOQUANT_PACK_FACTOR * TURBOQUANT_NZ_OPERAND_C0} so a packed row divides into "
            f"whole {TURBOQUANT_NZ_OPERAND_C0}-byte NZ column groups"
        )


def turboquant_strip_cache_to_codes(
    key_cache: torch.Tensor, value_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover INT4 codes from the strip cache the audited writers produce.

    Both planes are the buffers ``turboquant_packed_cache_shape`` allocates,
    ``(num_blocks, block_size, num_kv_heads, head_size / 2)`` int8, *addressed*
    the way the writers address them -- head before token. Returns two
    ``[num_blocks, num_kv_heads, block_size, head_size]`` int8 tensors in
    ``[-8, 7]``, indexed ``[page, head, token, channel]``.

    The key plane is token-major, so it unpacks directly. The value plane is
    channel-major inside a 16-token strip and is transposed back here.
    """
    if key_cache.shape != value_cache.shape:
        raise ValueError(
            f"the two planes must be identically shaped, got {tuple(key_cache.shape)} and {tuple(value_cache.shape)}"
        )
    num_blocks, block_size, num_kv_heads, packed_bytes = key_cache.shape
    head_size = packed_bytes * TURBOQUANT_PACK_FACTOR
    if block_size % TURBOQUANT_STRIP_TOKENS:
        raise ValueError(
            f"the strip cache is written {TURBOQUANT_STRIP_TOKENS} tokens at a time, so block_size "
            f"{block_size} must be a multiple of {TURBOQUANT_STRIP_TOKENS}"
        )
    strips_per_page = block_size // TURBOQUANT_STRIP_TOKENS

    panel = (num_blocks, num_kv_heads, block_size, packed_bytes)
    codes_key = TurboQuantEmulator.unpack(key_cache.reshape(panel))

    # [page, head, strip, channel, token % 16] -> [page, head, token, channel].
    strip_bytes = TURBOQUANT_STRIP_TOKENS // TURBOQUANT_PACK_FACTOR
    value_strips = value_cache.reshape(num_blocks, num_kv_heads, strips_per_page, head_size, strip_bytes)
    codes_value = (
        TurboQuantEmulator.unpack(value_strips)
        .permute(0, 1, 2, 4, 3)
        .reshape(num_blocks, num_kv_heads, block_size, head_size)
    )
    return codes_key.contiguous(), codes_value.contiguous()


def turboquant_codes_to_nz_tiled(codes: torch.Tensor) -> torch.Tensor:
    """Pack ``[num_blocks, num_kv_heads, block_size, head_size]`` codes NZ-tiled.

    Returns a flat int8 buffer of ``num_blocks * block_size * num_kv_heads *
    head_size / 2`` bytes -- the same byte count as the strip plane -- in which
    the byte at :func:`turboquant_nz_packed_byte` carries the two channels
    :func:`turboquant_nz_channel_nibble` names.
    """
    num_blocks, num_kv_heads, block_size, head_size = codes.shape
    _check_geometry(block_size, head_size)
    packed_bytes = head_size // TURBOQUANT_PACK_FACTOR
    tiles_per_page = block_size // TURBOQUANT_NZ_TILE_ROWS
    groups = packed_bytes // TURBOQUANT_NZ_OPERAND_C0

    # [page, head, token, channel] -> [tile, head, tileRow, channel]. The global
    # slot's tile index is page * tiles_per_page + token // kCubeTileRows, so the
    # tile axis comes out of the page axis and the token axis together.
    tiled = (
        codes.reshape(num_blocks, num_kv_heads, tiles_per_page, TURBOQUANT_NZ_TILE_ROWS, head_size)
        .permute(0, 2, 1, 3, 4)
        .reshape(num_blocks * tiles_per_page, num_kv_heads, TURBOQUANT_NZ_TILE_ROWS, head_size)
    )

    # The weave: interleave channel c with channel c + packed_bytes so that
    # pack()'s (low, high) pair per byte is exactly the decode's pair.
    low, high = tiled[..., :packed_bytes], tiled[..., packed_bytes:]
    woven = TurboQuantEmulator.pack(torch.stack((low, high), dim=-1).reshape(*tiled.shape[:-1], head_size))

    # [tile, head, tileRow, column] -> [tile, head, group, tileRow, byte].
    return (
        woven.reshape(woven.shape[0], num_kv_heads, TURBOQUANT_NZ_TILE_ROWS, groups, TURBOQUANT_NZ_OPERAND_C0)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1)
        .contiguous()
    )


def turboquant_scales_to_nz_tiled(scale_cache: torch.Tensor) -> torch.Tensor:
    """Plane-major ``(2, num_blocks, num_kv_heads, block_size)`` -> per-slot records.

    Returns ``[num_blocks * block_size, scale_slot_floats]`` fp32, the buffer
    ``ReadPlane`` bursts a tile's 64 rows out of in one go. Lanes past
    ``2 * num_kv_heads`` are the alignment pad and are zeroed.
    """
    if scale_cache.dim() != 4 or scale_cache.shape[0] != TURBOQUANT_PACK_FACTOR:
        raise ValueError(
            f"the strip scale plane is (2, num_blocks, num_kv_heads, block_size), got {tuple(scale_cache.shape)}"
        )
    _, num_blocks, num_kv_heads, block_size = scale_cache.shape
    slot_floats = turboquant_nz_scale_slot_floats(num_kv_heads)
    lanes = TURBOQUANT_PACK_FACTOR * num_kv_heads

    # (plane, page, head, token) -> (page, token, plane, head), so a record's
    # lane index is kv * num_kv_heads + kv_head.
    records = scale_cache.permute(1, 3, 0, 2).reshape(num_blocks * block_size, lanes)
    out = scale_cache.new_zeros((num_blocks * block_size, slot_floats))
    out[:, :lanes] = records
    return out


def turboquant_repack_strip_to_nz_tiled(
    key_cache: torch.Tensor, value_cache: torch.Tensor, scale_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a whole strip cache into the three planes the fused decode reads.

    ``key_cache`` and ``value_cache`` are the writers' int8 planes and
    ``scale_cache`` their plane-major fp32 plane. Returns
    ``(nz_key, nz_value, nz_scale)``, ready to pass straight to
    ``torch.ops.turboquant_ops.fused_decode``.

    The two packed planes come back in the *allocated* shape,
    ``(num_blocks, block_size, num_kv_heads, head_size / 2)``, and the same byte
    count as the inputs. Those dimensions are nominal: the binding reads
    ``block_size`` off ``size(1)`` and TORCH_CHECKs that both planes are 4-D and
    identically shaped, and the kernel then takes the buffer as a ``GM_ADDR`` and
    addresses it with :func:`turboquant_nz_packed_byte`. So the shape is what
    gets the launch accepted and the fractal inside it is what the decode
    actually reads; do not index these tensors by those dimensions.

    ``nz_scale`` is ``[num_blocks * block_size, scale_slot_floats]`` fp32, which
    is *not* the same size as the input scale plane unless ``2 * num_kv_heads``
    is already a multiple of 8. The binding only checks its dtype.

    Raises ``ValueError`` when ``block_size`` is writable but not readable -- see
    the note at the top of this module.
    """
    codes_key, codes_value = turboquant_strip_cache_to_codes(key_cache, value_cache)
    allocated = tuple(key_cache.shape)
    return (
        turboquant_codes_to_nz_tiled(codes_key).reshape(allocated),
        turboquant_codes_to_nz_tiled(codes_value).reshape(allocated),
        turboquant_scales_to_nz_tiled(scale_cache),
    )
