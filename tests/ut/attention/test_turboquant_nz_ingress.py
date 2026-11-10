# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The strip-cache -> NZ-tiled bridge, against both sides' addressing formulas.

Device-free: torch only, no torch_npu, no CANN, no NPU. Registered in
``run_turboquant_device_free.py``.

The thing under test is the vectorised repack in
:mod:`vllm_ascend.attention.turboquant_nz_ingress`. The oracle is a pair of
scalar writers/readers in this file, transcribed from the two C++ headers that
state the contracts -- the ``Layout contracts`` block of
``op_kernel/turboquant_reshape_batch.cpp`` for what the audited writers produce
and ``op_kernel/common/turboquant_layout.h`` plus
``op_kernel/vector/turboquant_vector_service.h`` for what the fused decode
reads. ``tests/ut/ops/turboquant/test_layout_bridge.cpp`` is the C++ gate over
the same two contracts. The repack and the oracle share no code, so agreement between them is a
real cross-check rather than a restatement.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import torch

from vllm_ascend.attention.turboquant_dry_run import (
    TURBOQUANT_PACK_FACTOR,
    TURBOQUANT_STRIP_TOKENS,
    TurboQuantEmulator,
)
from vllm_ascend.attention.turboquant_nz_ingress import (
    TURBOQUANT_FP32_PER_BLOCK,
    TURBOQUANT_NZ_OPERAND_C0,
    TURBOQUANT_NZ_TILE_ROWS,
    turboquant_codes_to_nz_tiled,
    turboquant_nz_channel_nibble,
    turboquant_nz_packed_byte,
    turboquant_nz_scale_index,
    turboquant_nz_scale_slot_floats,
    turboquant_repack_strip_to_nz_tiled,
    turboquant_scales_to_nz_tiled,
    turboquant_strip_cache_to_codes,
)

REPO = pathlib.Path(__file__).resolve().parents[3]
KERNEL_COMMON = REPO / "csrc" / "attention" / "turboquant" / "op_kernel" / "common"


# ---------------------------------------------------------------------------
# The oracle: the two layouts as scalar index arithmetic
# ---------------------------------------------------------------------------


def _strip_key_nibble(token: int, channel: int, head_size: int) -> int:
    """``key_nibble``: token-major inside the panel."""
    return token * head_size + channel


def _strip_value_nibble(token: int, channel: int, head_size: int) -> int:
    """``value_nibble``: channel-major inside a 16-token strip."""
    return (
        (token // TURBOQUANT_STRIP_TOKENS) * head_size * TURBOQUANT_STRIP_TOKENS
        + channel * TURBOQUANT_STRIP_TOKENS
        + token % TURBOQUANT_STRIP_TOKENS
    )


def _write_strip_plane_scalar(
    codes: torch.Tensor, nibble_of, num_kv_heads: int, block_size: int, head_size: int
) -> torch.Tensor:
    """Lay ``codes`` out the way the audited writers do, one nibble at a time.

    ``codes`` is ``[num_blocks, num_kv_heads, block_size, head_size]``. The
    result is the writers' int8 plane, shaped as the backend allocates it:
    ``(num_blocks, block_size, num_kv_heads, head_size / 2)``.
    """
    num_blocks = codes.shape[0]
    packed_bytes = head_size // TURBOQUANT_PACK_FACTOR
    flat = [0] * (num_blocks * block_size * num_kv_heads * packed_bytes)
    panel_bytes = block_size * packed_bytes
    nested = codes.tolist()
    for page in range(num_blocks):
        for head in range(num_kv_heads):
            # The writers' panel base: head before token, so a (page, head) panel is
            # one contiguous MTE3 burst.
            base = (page * num_kv_heads + head) * panel_bytes
            rows = nested[page][head]
            for token in range(block_size):
                row = rows[token]
                for channel in range(head_size):
                    nibble = nibble_of(token, channel, head_size)
                    flat[base + nibble // 2] |= (row[channel] & 0xF) << (4 * (nibble % 2))
    return (
        torch.tensor(flat, dtype=torch.uint8)
        .view(torch.int8)
        # The same flat bytes the backend allocates as
        # (num_blocks, block_size, num_kv_heads, packed_bytes) and the writers
        # address head-before-token.
        .reshape(num_blocks, block_size, num_kv_heads, packed_bytes)
    )


def _read_nz_code(plane: torch.Tensor, slot, kv_head, channel, num_kv_heads, head_size) -> int:
    """Read one channel out of an NZ-tiled plane through the decode's formula."""
    packed_bytes = head_size // TURBOQUANT_PACK_FACTOR
    column, nibble = turboquant_nz_channel_nibble(channel, head_size)
    byte = turboquant_nz_packed_byte(slot, kv_head, column, num_kv_heads, packed_bytes)
    raw = int(plane[byte].view(torch.uint8))
    code = (raw >> (4 * nibble)) & 0xF
    return code - 16 if code >= 8 else code


def _random_codes(num_blocks, num_kv_heads, block_size, head_size, seed) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        -8, 8, (num_blocks, num_kv_heads, block_size, head_size), generator=generator, dtype=torch.int8
    )


# ---------------------------------------------------------------------------
# The constants are the kernels'
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("constant", "value"),
    [
        ("kCubeTileRows", TURBOQUANT_NZ_TILE_ROWS),
        ("kOperandC0", TURBOQUANT_NZ_OPERAND_C0),
        ("kFp32PerBlock", TURBOQUANT_FP32_PER_BLOCK),
    ],
)
def test_nz_constants_match_the_layout_header(constant: str, value: int) -> None:
    """This module restates three C++ constants; drift must be a test failure."""
    text = (KERNEL_COMMON / "turboquant_layout.h").read_text(encoding="utf-8")
    match = re.search(rf"constexpr uint32_t {constant} = (\d+);", text)
    assert match, f"{constant} is no longer declared the way this test reads it"
    assert int(match.group(1)) == value


def test_scale_slot_floats_matches_the_common_header() -> None:
    """``ScaleSlotFloats`` rounds 2 * num_kv_heads up to an fp32 block."""
    text = (KERNEL_COMMON / "turboquant_common.h").read_text(encoding="utf-8")
    assert "CeilDiv(2 * numKvHeads, kFp32PerBlock) * kFp32PerBlock" in text
    assert turboquant_nz_scale_slot_floats(1) == 8
    assert turboquant_nz_scale_slot_floats(2) == 8
    assert turboquant_nz_scale_slot_floats(4) == 8
    assert turboquant_nz_scale_slot_floats(5) == 16


# ---------------------------------------------------------------------------
# The addressing itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
@pytest.mark.parametrize("block_size", [64, 128])
def test_packed_byte_is_a_permutation_of_the_plane(num_kv_heads: int, block_size: int) -> None:
    """Every (slot, head, column) names a distinct byte, and together all of them.

    A layout that dropped or aliased a byte would still look plausible in a
    spot check; this is the structural property that rules it out.
    """
    head_size, num_blocks = 128, 2
    packed_bytes = head_size // TURBOQUANT_PACK_FACTOR
    slots = num_blocks * block_size
    seen = [
        turboquant_nz_packed_byte(slot, head, column, num_kv_heads, packed_bytes)
        for slot in range(slots)
        for head in range(num_kv_heads)
        for column in range(packed_bytes)
    ]
    assert sorted(seen) == list(range(slots * num_kv_heads * packed_bytes))


def test_channel_nibble_is_the_woven_pairing() -> None:
    """Channel c shares a byte with c + head_size / 2, not with c ^ 1.

    This is the axis the two sides disagree on most quietly: both pairings fill
    the plane completely, so only the pairing itself distinguishes them.
    """
    head_size = 256
    half = head_size // TURBOQUANT_PACK_FACTOR
    for channel in range(half):
        assert turboquant_nz_channel_nibble(channel, head_size) == (channel, 0)
        assert turboquant_nz_channel_nibble(channel + half, head_size) == (channel, 1)


def test_scale_index_packs_both_lanes_into_one_slot_record() -> None:
    """K and V of one slot are in the same record, unlike the writers' planes."""
    num_kv_heads = 2
    slot_floats = turboquant_nz_scale_slot_floats(num_kv_heads)
    for slot in (0, 1, 63, 64):
        for head in range(num_kv_heads):
            key = turboquant_nz_scale_index(slot, head, 0, num_kv_heads)
            value = turboquant_nz_scale_index(slot, head, 1, num_kv_heads)
            assert key == slot * slot_floats + head
            assert value == key + num_kv_heads
            assert value < (slot + 1) * slot_floats


# ---------------------------------------------------------------------------
# The repack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [64, 128])
def test_repack_puts_every_code_where_the_decode_reads_it(block_size: int) -> None:
    """Exhaustive: strip-write, repack, then read back through the NZ formula.

    Both planes, every (page, head, token, channel). The strip side is written
    by the scalar oracle above, so the only thing that can make this pass is the
    repack agreeing with both contracts at once.
    """
    num_blocks, num_kv_heads, head_size = 2, 2, 128
    codes_key = _random_codes(num_blocks, num_kv_heads, block_size, head_size, seed=11)
    codes_value = _random_codes(num_blocks, num_kv_heads, block_size, head_size, seed=12)

    key_plane = _write_strip_plane_scalar(codes_key, _strip_key_nibble, num_kv_heads, block_size, head_size)
    value_plane = _write_strip_plane_scalar(codes_value, _strip_value_nibble, num_kv_heads, block_size, head_size)

    # The strip plane really is the writers' layout, so the recovery step is
    # under test too, not assumed.
    recovered_key, recovered_value = turboquant_strip_cache_to_codes(key_plane, value_plane)
    assert torch.equal(recovered_key, codes_key)
    assert torch.equal(recovered_value, codes_value)

    for codes, recovered in ((codes_key, recovered_key), (codes_value, recovered_value)):
        nz = turboquant_codes_to_nz_tiled(recovered)
        assert nz.numel() == key_plane.numel()
        for page in range(num_blocks):
            for head in range(num_kv_heads):
                for token in range(block_size):
                    slot = page * block_size + token
                    for channel in range(head_size):
                        got = _read_nz_code(nz, slot, head, channel, num_kv_heads, head_size)
                        assert got == int(codes[page, head, token, channel]), (
                            f"page {page} head {head} token {token} channel {channel}"
                        )


def test_repack_at_the_production_geometry() -> None:
    """head_size 256, block_size 128, 4 kv heads -- sampled rather than exhaustive."""
    num_blocks, num_kv_heads, block_size, head_size = 3, 4, 128, 256
    codes = _random_codes(num_blocks, num_kv_heads, block_size, head_size, seed=13)
    nz = turboquant_codes_to_nz_tiled(codes)
    assert nz.numel() == num_blocks * block_size * num_kv_heads * head_size // TURBOQUANT_PACK_FACTOR

    generator = torch.Generator().manual_seed(17)
    for _ in range(4000):
        page = int(torch.randint(num_blocks, (1,), generator=generator))
        head = int(torch.randint(num_kv_heads, (1,), generator=generator))
        token = int(torch.randint(block_size, (1,), generator=generator))
        channel = int(torch.randint(head_size, (1,), generator=generator))
        slot = page * block_size + token
        got = _read_nz_code(nz, slot, head, channel, num_kv_heads, head_size)
        assert got == int(codes[page, head, token, channel])


def test_value_transposition_is_undone_not_carried_through() -> None:
    """The strip V plane is transposed; the NZ V plane must not be.

    A repack that forgot the de-transposition would still produce a full,
    self-consistent plane -- it would just attend over the wrong tokens.
    """
    num_blocks, num_kv_heads, block_size, head_size = 1, 1, 64, 128
    codes = _random_codes(num_blocks, num_kv_heads, block_size, head_size, seed=19)
    value_plane = _write_strip_plane_scalar(codes, _strip_value_nibble, num_kv_heads, block_size, head_size)
    key_plane = _write_strip_plane_scalar(codes, _strip_key_nibble, num_kv_heads, block_size, head_size)
    # The same codes written through the two different strip formulas. Both
    # recoveries must return those codes, which is only true if the value side
    # undoes the transposition.
    from_key, from_value = turboquant_strip_cache_to_codes(key_plane, value_plane)
    assert torch.equal(from_key, codes)
    assert torch.equal(from_value, codes)
    assert torch.equal(turboquant_codes_to_nz_tiled(from_key), turboquant_codes_to_nz_tiled(from_value))
    # ... and the two strip planes are genuinely different bytes, so none of the
    # above is comparing a buffer with itself.
    assert not torch.equal(key_plane, value_plane)


def test_scale_records_match_the_decode_index() -> None:
    num_blocks, num_kv_heads, block_size = 2, 2, 64
    generator = torch.Generator().manual_seed(23)
    scales = torch.rand((TURBOQUANT_PACK_FACTOR, num_blocks, num_kv_heads, block_size), generator=generator)
    records = turboquant_scales_to_nz_tiled(scales)

    slot_floats = turboquant_nz_scale_slot_floats(num_kv_heads)
    assert records.shape == (num_blocks * block_size, slot_floats)
    flat = records.reshape(-1)
    for page in range(num_blocks):
        for token in range(block_size):
            slot = page * block_size + token
            for head in range(num_kv_heads):
                for kv in range(TURBOQUANT_PACK_FACTOR):
                    index = turboquant_nz_scale_index(slot, head, kv, num_kv_heads)
                    assert flat[index] == scales[kv, page, head, token]
    # The pad lanes are the decode's burst tail; they must be defined.
    assert torch.equal(
        records[:, TURBOQUANT_PACK_FACTOR * num_kv_heads :],
        torch.zeros_like(records[:, TURBOQUANT_PACK_FACTOR * num_kv_heads :]),
    )


def test_a_tile_of_scales_is_one_contiguous_burst() -> None:
    """``ReadPlane`` bursts ``kCubeTileRows * scaleSlotFloats`` floats from row."""
    num_blocks, num_kv_heads, block_size = 2, 4, 128
    scales = torch.rand((TURBOQUANT_PACK_FACTOR, num_blocks, num_kv_heads, block_size))
    records = turboquant_scales_to_nz_tiled(scales)
    slot_floats = turboquant_nz_scale_slot_floats(num_kv_heads)
    for row_base in range(0, num_blocks * block_size, TURBOQUANT_NZ_TILE_ROWS):
        burst = records.reshape(-1)[row_base * slot_floats : (row_base + TURBOQUANT_NZ_TILE_ROWS) * slot_floats]
        assert burst.numel() == TURBOQUANT_NZ_TILE_ROWS * slot_floats


# ---------------------------------------------------------------------------
# The geometries that cannot be bridged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [16, 32])
def test_block_sizes_the_writers_take_but_the_decode_cannot(block_size: int) -> None:
    """16 and 32 are writable and unreadable; the bridge says so instead of lying."""
    codes = _random_codes(1, 1, block_size, 128, seed=29)
    with pytest.raises(ValueError, match="power-of-two multiple of 64"):
        turboquant_codes_to_nz_tiled(codes)


@pytest.mark.parametrize("block_size", [192, 320])
def test_non_power_of_two_pages_are_refused(block_size: int) -> None:
    """A 192-row page is a multiple of 64 but needs a divide on the SPU."""
    codes = _random_codes(1, 1, block_size, 128, seed=31)
    with pytest.raises(ValueError, match="power-of-two multiple of 64"):
        turboquant_codes_to_nz_tiled(codes)


def test_head_size_must_divide_into_whole_column_groups() -> None:
    codes = _random_codes(1, 1, 64, 32, seed=37)
    with pytest.raises(ValueError, match="whole 32-byte NZ column groups"):
        turboquant_codes_to_nz_tiled(codes)


def test_mismatched_planes_are_refused() -> None:
    with pytest.raises(ValueError, match="identically shaped"):
        turboquant_strip_cache_to_codes(
            torch.zeros((1, 64, 1, 64), dtype=torch.int8),
            torch.zeros((1, 64, 2, 64), dtype=torch.int8),
        )


def test_scale_plane_shape_is_checked() -> None:
    with pytest.raises(ValueError, match=r"\(2, num_blocks, num_kv_heads, block_size\)"):
        turboquant_scales_to_nz_tiled(torch.rand((3, 2, 2, 64)))


# ---------------------------------------------------------------------------
# End to end, through the adapter's own strip writer
# ---------------------------------------------------------------------------


def test_repack_bridges_the_adapter_strip_writer() -> None:
    """Tie the bridge to this repo's own writer restatement, not just the oracle.

    ``_write_strip_planes`` is what ``tests/ut/attention/test_turboquant_strip_cache.py``
    gates against the kernels' nibble formula, so going through it means the
    bridge starts from the layout the audited writers are held to.
    """
    from vllm_ascend.attention.turboquant_dry_run import TurboQuantOpsAdapter

    num_blocks, num_kv_heads, block_size, head_size = 2, 2, 64, 128
    packed_bytes = head_size // TURBOQUANT_PACK_FACTOR
    tokens = num_blocks * block_size

    generator = torch.Generator().manual_seed(41)
    codes_key = torch.randint(-8, 8, (tokens, num_kv_heads, head_size), generator=generator, dtype=torch.int8)
    codes_value = torch.randint(-8, 8, (tokens, num_kv_heads, head_size), generator=generator, dtype=torch.int8)

    key_cache = torch.zeros((num_blocks, block_size, num_kv_heads, packed_bytes), dtype=torch.int8)
    value_cache = torch.zeros_like(key_cache)
    slots = torch.arange(tokens)
    pages = torch.div(slots, block_size, rounding_mode="floor")
    offsets = slots - pages * block_size

    adapter = TurboQuantOpsAdapter.__new__(TurboQuantOpsAdapter)
    TurboQuantOpsAdapter._write_strip_planes(adapter, codes_key, codes_value, key_cache, value_cache, pages, offsets)

    nz_key, nz_value, _ = turboquant_repack_strip_to_nz_tiled(
        key_cache,
        value_cache,
        torch.rand((TURBOQUANT_PACK_FACTOR, num_blocks, num_kv_heads, block_size)),
    )

    # The planes come back in the allocated 4-D shape so the binding accepts
    # them; the fractal inside is flat, which is how the kernel reads it.
    assert nz_key.shape == key_cache.shape
    assert nz_value.shape == value_cache.shape

    for plane, codes in ((nz_key.reshape(-1), codes_key), (nz_value.reshape(-1), codes_value)):
        for token in range(0, tokens, 7):
            for head in range(num_kv_heads):
                for channel in range(0, head_size, 11):
                    got = _read_nz_code(plane, token, head, channel, num_kv_heads, head_size)
                    assert got == int(codes[token, head, channel])


def test_packed_bytes_survive_a_round_trip_through_codes() -> None:
    """pack/unpack is the audited codec; the bridge must not perturb it."""
    codes = _random_codes(2, 2, 64, 128, seed=43)
    packed = TurboQuantEmulator.pack(codes)
    assert torch.equal(TurboQuantEmulator.unpack(packed), codes)


def test_repack_output_satisfies_the_fused_decode_binding() -> None:
    """The three planes must pass the binding's own cache TORCH_CHECKs.

    Read off csrc/attention/turboquant/torch_bindings.cpp rather than restated,
    so that tightening a check there without updating the bridge fails here
    instead of at a launch on hardware nobody has.
    """
    bindings = (REPO / "csrc" / "attention" / "turboquant" / "torch_bindings.cpp").read_text(encoding="utf-8")
    assert "key_cache.dim() == 4 && key_cache.sizes() == value_cache.sizes()" in bindings
    assert "const std::int64_t block_size = key_cache.size(1);" in bindings
    assert "block_size % layout::kCubeTileRows == 0" in bindings
    assert "key_cache.scalar_type() == at::kChar && value_cache.scalar_type() == at::kChar" in bindings
    assert "scale_cache.scalar_type() == at::kFloat" in bindings

    num_blocks, num_kv_heads, block_size, head_size = 2, 2, 128, 256
    packed_bytes = head_size // TURBOQUANT_PACK_FACTOR
    shape = (num_blocks, block_size, num_kv_heads, packed_bytes)
    nz_key, nz_value, nz_scale = turboquant_repack_strip_to_nz_tiled(
        torch.zeros(shape, dtype=torch.int8),
        torch.zeros(shape, dtype=torch.int8),
        torch.zeros((TURBOQUANT_PACK_FACTOR, num_blocks, num_kv_heads, block_size)),
    )

    assert nz_key.dim() == 4 and nz_key.shape == nz_value.shape
    assert nz_key.size(1) == block_size
    assert nz_key.size(1) % TURBOQUANT_NZ_TILE_ROWS == 0
    assert nz_key.size(1) & (nz_key.size(1) - 1) == 0
    assert nz_key.dtype is torch.int8 and nz_value.dtype is torch.int8
    assert nz_scale.dtype is torch.float32
    assert nz_key.is_contiguous() and nz_value.is_contiguous() and nz_scale.is_contiguous()
