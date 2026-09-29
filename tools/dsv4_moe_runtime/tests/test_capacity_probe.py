"""Capacity probe: synthetic OOM detection/recovery and zero-alloc thrash."""

from __future__ import annotations

import pytest
import torch

from dsv4_moe_runtime.benchmarks.capacity_probe import (
    CapacityProbeConfig,
    DummyLoopbackWeightProvider,
    run_capacity_probe,
)
from dsv4_moe_runtime.core.layout import ExpertTensorLayout
from dsv4_moe_runtime.hardware.runtime import CpuRuntime


def test_dummy_loopback_provider_fills_pattern(sanity_layout: ExpertTensorLayout) -> None:
    provider = DummyLoopbackWeightProvider(CpuRuntime(), sanity_layout)
    views = {spec.param_key: torch.zeros(spec.view_shape, dtype=torch.uint8) for spec in sanity_layout.specs}

    provider.ensure_staged(31, 255)  # loopback: every expert is staged, never raises
    moved = provider.fill_slot_params(0, 0, views)
    assert moved == sanity_layout.slot_num_bytes
    assert provider.bytes_copied == moved and provider.fill_count == 1
    for param_key, view in views.items():
        assert torch.equal(view, provider.pinned_cpu_weight(3, 200, param_key))  # one shared buffer


def test_probe_detects_synthetic_oom_and_recovers() -> None:
    calls = {"count": 0}

    def failing_allocator(num_bytes: int, device: str) -> torch.Tensor:
        calls["count"] += 1
        if calls["count"] > 3:  # backbone + two probe batches succeed, then synthetic OOM
            raise torch.OutOfMemoryError("synthetic OOM from the test allocator")
        return torch.empty(num_bytes, dtype=torch.uint8, device=device)

    config = CapacityProbeConfig(
        device="cpu",
        small_geometry=True,
        backbone_bytes=1 << 20,
        probe_slot_batch=64,
        thrash_cycles=4,
        slots_swept_per_cycle=32,
        safety_margin=0.5,
    )
    report = run_capacity_probe(config, allocator=failing_allocator)

    assert report.oom_detected and report.oom_error is not None
    assert "synthetic OOM" in report.oom_error
    assert report.max_contiguous_slots == 2 * 64  # two successful probe batches
    assert report.oom_recovered is True  # ceiling pool allocated after the OOM
    assert report.ceiling_slots == int(2 * 64 * 0.5)
    assert report.thrash is not None
    assert report.thrash.cycles == 4 and report.thrash.steps > 0
    assert report.thrash.fills == report.thrash.steps * 6  # loopback forces a miss every step
    assert report.allocation_violation_cycles == []
    assert report.fingerprints_stable
    assert report.loopback_verified is not None and report.loopback_verified[0]
    assert report.passed


def test_thrash_preserves_zero_allocation() -> None:
    config = CapacityProbeConfig(
        device="cpu",
        small_geometry=True,
        backbone_bytes=32 << 20,
        probe_slot_batch=128,
        probe_max_slots=512,  # cap: CPU hosts have no natural OOM boundary
        thrash_cycles=10,
        slots_swept_per_cycle=48,
        safety_margin=0.9,
    )
    report = run_capacity_probe(config)

    assert report.max_contiguous_slots == 512
    assert report.oom_detected is False
    assert report.ceiling_slots == int(512 * 0.9)
    assert report.thrash is not None and report.thrash.cycles == 10
    assert report.thrash.fills >= report.ceiling_slots  # full-pool turnover happened
    # CPU runtime has no device allocator; the data_ptr fingerprints carry the
    # invariant, so the allocator delta stays None and violations stay empty.
    assert report.allocated_delta_bytes is None
    assert report.allocation_violation_cycles == []
    assert report.fingerprints_stable
    assert report.passed


def test_probe_skips_ceiling_below_one_topk() -> None:
    def always_oom(num_bytes: int, device: str) -> torch.Tensor:
        if num_bytes != (1 << 20):  # only the backbone succeeds; every probe batch fails
            raise torch.OutOfMemoryError("synthetic OOM: first probe batch")
        return torch.empty(num_bytes, dtype=torch.uint8, device=device)

    config = CapacityProbeConfig(
        device="cpu",
        small_geometry=True,
        backbone_bytes=1 << 20,
        probe_slot_batch=2,  # 2 slots < top_k (6)
        thrash_cycles=1,
    )
    report = run_capacity_probe(config, allocator=always_oom)

    assert report.oom_detected and report.max_contiguous_slots == 0
    assert report.ceiling_slots == 0 and report.thrash is None
    assert any("below one top-k" in note for note in report.notes)
    assert report.oom_recovered is True  # a small allocation succeeded after the OOM
    assert report.passed


@pytest.mark.parametrize("margin", [0.0, 1.0, -0.5])
def test_probe_cli_rejects_invalid_margin(margin: float) -> None:
    from dsv4_moe_runtime.benchmarks.capacity_probe import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--device", "cpu", "--small-geometry", "--safety-margin", str(margin)])
    assert excinfo.value.code == 2
