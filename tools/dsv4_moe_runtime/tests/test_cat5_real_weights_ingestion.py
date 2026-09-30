"""CAT 5 -- real-weight ingestion: DirectStorage streaming, integrity, FP4/E8M0.

Generates a minimal *valid* safetensors checkpoint on disk (dense-backbone
slices plus 16 routed experts with known bit patterns and E8M0 scales), then
verifies the DirectStorage pipeline end to end: layout binding with exact
13,369,344-byte slot accounting, streaming ingestion with strict
zero-allocation, pipelined throughput, byte-level HBM read-back integrity,
and FP4 nibble / E8M0 decode accuracy against the CANN conversion table.
"""

from __future__ import annotations

import json
import struct
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from ..benchmarks.telemetry import full_fingerprints
from ..core.config import DeepSeekV4MoEConfig
from ..core.layout import SLOT_REGION_ALIGN_BYTES, ExpertTensorLayout
from ..core.slot_pool import StaticExpertSlotPool
from ..hardware.runtime import CpuRuntime
from ..hardware.safetensors_provider import (
    SAFETENSORS_HEADER_LENGTH_BYTES,
    SafetensorsExpertProvider,
    WeightLayoutMismatchError,
    bind_expert_spans,
    parse_safetensors_header,
)
from ..hardware.weight_loader import StreamingWeightLoader
from .conftest import forbid_torch_allocations

FP4_LOOKUP = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
NUM_EXPERTS = 16
# Full production expert geometry: the 13,369,344-byte slot contract only
# holds at hidden 4096 / intermediate 2048 (vocab is irrelevant here).
CAT5_CONFIG = DeepSeekV4MoEConfig(vocab_size=1024)
CAT5_LAYER_IDS = [0]
DENSE_DEFINITIONS = {
    "layers.0.attn.wq_a.weight": 4096,  # fp8 bytes, [1024, 4096] flattened
    "layers.0.attn_norm.weight": 8192,  # bf16 bytes, [4096]
}


_PATTERN_TILE_BYTES = 4096


def _weight_bytes(param_index: int, expert_id: int, rows: int, cols: int) -> bytes:
    """Deterministic FP4 bit pattern per (param, expert), tiled for speed."""
    tile = bytes((expert_id * 61 + param_index * 17 + i) & 0xFF for i in range(_PATTERN_TILE_BYTES))
    total = rows * cols
    return (tile * (total // _PATTERN_TILE_BYTES + 1))[:total]


def _scale_bytes(param_index: int, expert_id: int, rows: int, cols: int) -> bytes:
    """E8M0 exponents cycling 2^-1 .. 2^5 (bias-127 bytes 126..132)."""
    tile = bytes(126 + ((expert_id + param_index + i) % 7) for i in range(_PATTERN_TILE_BYTES))
    total = rows * cols
    return (tile * (total // _PATTERN_TILE_BYTES + 1))[:total]


def _expert_payload(param_index: int, spec, expert_id: int) -> bytes:
    rows, cols = spec.view_shape
    if spec.kind == "packed_fp4":
        return _weight_bytes(param_index, expert_id, rows, cols)
    return _scale_bytes(param_index, expert_id, rows, cols)


def _payload_tensor(param_index: int, spec, expert_id: int) -> torch.Tensor:
    return torch.frombuffer(bytearray(_expert_payload(param_index, spec, expert_id)), dtype=torch.uint8)


def _write_synthetic_checkpoint(path: Path, layout: ExpertTensorLayout, num_experts: int) -> dict[str, bytes]:
    """Write a minimal valid safetensors file; returns the exact dense bytes.

    Alignment is a *file-absolute* property: the 8-byte length prefix plus the
    JSON header shift every payload, so the header (whose ``data_offsets`` are
    data-section-relative) is serialized first and the 128-byte file padding
    is computed afterwards per tensor.
    """
    entries: list[tuple[str, int, bytes]] = []
    cursor = 0

    def append(name: str, payload: bytes) -> None:
        nonlocal cursor
        entries.append((name, cursor, payload))
        cursor += len(payload)

    dense_payloads = {name: bytes((i * 31 + 5) & 0xFF for i in range(size)) for name, size in DENSE_DEFINITIONS.items()}
    for name, payload in dense_payloads.items():
        append(name, payload)
    for expert_id in range(num_experts):
        for param_index, spec in enumerate(layout.specs):
            kind = "weight" if spec.kind == "packed_fp4" else "scale"
            name = f"layers.0.ffn.experts.{expert_id}.{spec.name}.{kind}"
            append(name, _expert_payload(param_index, spec, expert_id))

    header = {
        name: {"dtype": "U8", "shape": [1, len(payload)], "data_offsets": [start, start + len(payload)]}
        for name, start, payload in entries
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # safetensors permits trailing whitespace inside the header: pad the JSON
    # so the data section starts 128-byte aligned. Every entry length is a
    # 128-multiple, so all spans stay aligned without per-entry gaps.
    while (SAFETENSORS_HEADER_LENGTH_BYTES + len(header_bytes)) % SLOT_REGION_ALIGN_BYTES:
        header_bytes += b" "
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for _name, _start, payload in entries:
            handle.write(payload)
    return dense_payloads


def _staged_six(expert_id: int) -> list[int]:
    """Six unique staged expert ids anchored at ``expert_id``."""
    return [(expert_id + offset) % NUM_EXPERTS for offset in range(6)]


def _verify_slot_bytes(pool: StaticExpertSlotPool, layout: ExpertTensorLayout, slot: int, expert_id: int) -> None:
    for param_index, spec in enumerate(layout.specs):
        view = pool.weight_views(slot)[spec.name] if spec.kind == "packed_fp4" else pool.scale_views(slot)[spec.name]
        expected = _payload_tensor(param_index, spec, expert_id).view(spec.view_shape)
        assert torch.equal(view, expected), (slot, spec.param_key)


@pytest.fixture
def cat5_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "dsv4_experts_staging.safetensors"
    _write_synthetic_checkpoint(path, ExpertTensorLayout.for_deepseek_v4_flash(CAT5_CONFIG), NUM_EXPERTS)
    return path


def test_synthetic_checkpoint_binds_exact_slot_layout(cat5_checkpoint: Path) -> None:
    layout = ExpertTensorLayout.for_deepseek_v4_flash(CAT5_CONFIG)
    header, data_start, file_size = parse_safetensors_header(str(cat5_checkpoint))
    spans = bind_expert_spans(header, data_start, layout, layer_ids=[0], num_experts=NUM_EXPERTS, file_size=file_size)

    assert len(spans) == NUM_EXPERTS
    for params in spans.values():
        total = sum(size for _begin, size in params.values())
        assert total == layout.slot_num_bytes
        assert total == 12.75 * 1024 * 1024  # the exact DeepSeek-V4 slot size
        for begin, _end in params.values():
            assert begin % SLOT_REGION_ALIGN_BYTES == 0


def test_layout_binding_rejects_mismatched_spans(cat5_checkpoint: Path, tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.safetensors"
    payload = cat5_checkpoint.read_bytes()
    truncated.write_bytes(payload[: len(payload) // 2])  # cuts the last experts' spans

    layout = ExpertTensorLayout.for_deepseek_v4_flash(CAT5_CONFIG)
    header, data_start, file_size = parse_safetensors_header(str(truncated))
    with pytest.raises(WeightLayoutMismatchError):
        bind_expert_spans(header, data_start, layout, layer_ids=[0], num_experts=NUM_EXPERTS, file_size=file_size)


def test_streaming_ingestion_zero_allocation(cat5_checkpoint: Path, sanity_config: DeepSeekV4MoEConfig) -> None:
    """Stream all 16 experts through the provider into the pool: no allocation."""
    del sanity_config  # CAT5 uses its own smaller geometry
    layout = ExpertTensorLayout.for_deepseek_v4_flash(CAT5_CONFIG)
    provider = SafetensorsExpertProvider(
        CpuRuntime(), str(cat5_checkpoint), layout, layer_ids=CAT5_LAYER_IDS, num_experts=NUM_EXPERTS
    )
    pool = StaticExpertSlotPool(CAT5_CONFIG, num_slots=NUM_EXPERTS, layout=layout)
    fingerprint = full_fingerprints(pool)

    with forbid_torch_allocations():
        for expert_id in range(NUM_EXPERTS):
            requested = _staged_six(expert_id)
            reservation = pool.acquire_for_step(0, requested, provider)
            try:
                assert [int(value) for value in reservation.device_slot_ids] == list(reservation.slot_ids)
                for expert, slot in zip(requested, reservation.slot_ids):
                    _verify_slot_bytes(pool, layout, slot, expert)
            finally:
                pool.release_step(reservation)
    assert full_fingerprints(pool) == fingerprint
    assert provider.experts_served == NUM_EXPERTS


def test_dense_backbone_streaming(cat5_checkpoint: Path) -> None:
    layout = ExpertTensorLayout.for_deepseek_v4_flash(CAT5_CONFIG)
    runtime = CpuRuntime()
    loader = StreamingWeightLoader(runtime, str(cat5_checkpoint), chunk_bytes=1 << 20, num_chunks=4)
    provider = SafetensorsExpertProvider(
        runtime,
        str(cat5_checkpoint),
        layout,
        layer_ids=CAT5_LAYER_IDS,
        num_experts=NUM_EXPERTS,
        loader=loader,
        dense_names=tuple(DENSE_DEFINITIONS),
    )

    backbone = torch.empty(DENSE_DEFINITIONS["layers.0.attn_norm.weight"], dtype=torch.uint8)
    moved = provider.stream_dense_into("layers.0.attn_norm.weight", backbone)
    assert moved == 8192
    assert bytes(backbone.tolist()) == bytes((i * 31 + 5) & 0xFF for i in range(8192))
    with pytest.raises(KeyError):
        provider.stream_dense_into("layers.9.attn.wq_a.weight", backbone)
    loader.close()


def test_ingestion_throughput_reported(cat5_checkpoint: Path) -> None:
    """Pipelined streaming throughput (GB/s), measured over all 16 experts."""
    layout = ExpertTensorLayout.for_deepseek_v4_flash(CAT5_CONFIG)
    runtime = CpuRuntime()
    loader = StreamingWeightLoader(runtime, str(cat5_checkpoint), chunk_bytes=1 << 20, num_chunks=4)
    provider = SafetensorsExpertProvider(
        runtime,
        str(cat5_checkpoint),
        layout,
        layer_ids=CAT5_LAYER_IDS,
        num_experts=NUM_EXPERTS,
        loader=loader,
        dense_names=tuple(DENSE_DEFINITIONS),
    )

    slot_buffer = torch.empty(layout.slot_num_bytes, dtype=torch.uint8)
    total_bytes = 0
    started = time.perf_counter()
    for expert_id in range(NUM_EXPERTS):
        for param_index, spec in enumerate(layout.specs):
            begin, size = provider.expert_spans[(0, expert_id)][spec.param_key]
            total_bytes += loader.stream_into(slot_buffer.view(-1)[:size], begin, size)
            del param_index
    elapsed = time.perf_counter() - started
    throughput_gbps = total_bytes / elapsed / 1e9
    print(f"\nCAT5 ingestion throughput: {throughput_gbps:.2f} GB/s over {total_bytes} bytes in {elapsed:.3f}s")
    assert total_bytes == NUM_EXPERTS * layout.slot_num_bytes
    assert throughput_gbps > 0.05, "streaming throughput collapsed"
    loader.close()


def test_byte_integrity_and_fp4_e8m0_decode(cat5_checkpoint: Path) -> None:
    """HBM read-back equals raw disk bytes; FP4 nibbles and E8M0 decode exactly."""
    layout = ExpertTensorLayout.for_deepseek_v4_flash(CAT5_CONFIG)
    provider = SafetensorsExpertProvider(
        CpuRuntime(), str(cat5_checkpoint), layout, layer_ids=CAT5_LAYER_IDS, num_experts=NUM_EXPERTS
    )
    pool = StaticExpertSlotPool(CAT5_CONFIG, num_slots=8, layout=layout)

    expert_id, sample_byte_offset = 7, 1234
    requested = _staged_six(expert_id)
    reservation = pool.acquire_for_step(0, requested, provider)
    slot = reservation.slot_ids[0]

    raw_file = cat5_checkpoint.read_bytes()
    header, data_start, _file_size = parse_safetensors_header(str(cat5_checkpoint))
    for param_index, spec in enumerate(layout.specs):
        view = pool.weight_views(slot)[spec.name] if spec.kind == "packed_fp4" else pool.scale_views(slot)[spec.name]
        kind = "weight" if spec.kind == "packed_fp4" else "scale"
        disk_name = f"layers.0.ffn.experts.{expert_id}.{spec.name}.{kind}"
        begin, end = header[disk_name]["data_offsets"]
        expected = raw_file[data_start + begin : data_start + end]
        assert bytes(view.flatten().tolist()) == expected, spec.name

        byte = view.flatten()[sample_byte_offset].item()
        pattern_byte = (expert_id * 61 + param_index * 17 + sample_byte_offset) & 0xFF
        if spec.kind == "packed_fp4":
            # CANN e2m1 table: low nibble first, then high nibble.
            assert FP4_LOOKUP[byte & 0xF] == FP4_LOOKUP[pattern_byte & 0xF]
            assert FP4_LOOKUP[byte >> 4] == FP4_LOOKUP[pattern_byte >> 4]
        else:
            decoded = 2.0 ** (byte - 127)  # E8M0: bias-127 pure exponent
            scale_index = sample_byte_offset % _PATTERN_TILE_BYTES
            pattern_exponent = 126 + ((expert_id + param_index + scale_index) % 7)
            assert decoded == 2.0 ** (pattern_exponent - 127)
    pool.release_step(reservation)


def test_provider_rejects_unknown_expert(cat5_checkpoint: Path) -> None:
    layout = ExpertTensorLayout.for_deepseek_v4_flash(CAT5_CONFIG)
    provider = SafetensorsExpertProvider(
        CpuRuntime(), str(cat5_checkpoint), layout, layer_ids=CAT5_LAYER_IDS, num_experts=NUM_EXPERTS
    )
    with pytest.raises(KeyError):
        provider.ensure_staged(0, NUM_EXPERTS + 1)
    with pytest.raises(KeyError):
        provider.ensure_staged(1, 0)  # only layer 0 exists in the staging file


def _iterate_all() -> Iterator[int]:  # pragma: no cover - documentation helper
    yield from range(NUM_EXPERTS)
