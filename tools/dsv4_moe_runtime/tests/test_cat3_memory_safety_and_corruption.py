"""CAT 3 -- memory safety & corruption: capacity probing, zero-allocation
invariants, pointer stability under churn.

Covers the device-memory safety leg: synthetic OOM boundary detection and
recovery, allocator-booby-trapped decode loops (no ``torch.empty``/``clone``/
``cat`` at serve time), and ``data_ptr`` fingerprint stability of every arena
and pre-sliced view across LRU churn at both the pool and the full-engine
level.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

import pytest
import torch

from ..benchmarks.capacity_probe import (
    CapacityProbeConfig,
    run_capacity_probe,
)
from ..core.config import DeepSeekV4MoEConfig
from ..core.layout import ExpertTensorLayout
from ..core.slot_pool import StaticExpertSlotPool
from ..draft_inference.backends import MockV5Backend
from ..draft_inference.engine import DraftInferenceEngine
from ..protocols.provider import WeightProviderProtocol

NUM_DECODE_STEPS = 50
NUM_SLOTS_FOR_CHURN_TEST = 16


def churn_expert_ids(step: int, top_k: int, num_experts: int) -> list[int]:
    """Deterministic decode-step request mixing loads, LRU hits and evictions.

    Steps come in groups of five that request the same six experts (four pure
    LRU-hit steps after the initial loads); every fifth step shifts the window
    by eight experts, forcing the pool to evict the LRU-oldest residents.
    """
    window_base = (step // 5) * 8
    return [(window_base + 3 * expert_pos) % num_experts for expert_pos in range(top_k)]


def test_zero_runtime_allocations(
    sanity_config: DeepSeekV4MoEConfig,
    sanity_layout: ExpertTensorLayout,
    mock_provider_factory: Callable[..., WeightProviderProtocol],
    allocation_guard: Callable[..., AbstractContextManager[None]],
) -> None:
    pool = StaticExpertSlotPool(sanity_config, num_slots=NUM_SLOTS_FOR_CHURN_TEST, layout=sanity_layout)
    pilot_layer = 3
    provider = mock_provider_factory(
        sanity_layout, layer_ids=(pilot_layer,), num_experts=sanity_config.num_routed_experts
    )

    def pool_fingerprints() -> list[int]:
        tensors = [pool.slot_arena, pool.expert_slot_table, pool.step_slot_ids_buffer]
        for slot_id in range(pool.num_slots):
            tensors.extend(pool.weight_views(slot_id).values())
            tensors.extend(pool.scale_views(slot_id).values())
        return [tensor.data_ptr() for tensor in tensors]

    baseline = pool_fingerprints()

    with allocation_guard():
        for step in range(NUM_DECODE_STEPS):
            expert_ids = churn_expert_ids(step, sanity_config.top_k, sanity_config.num_routed_experts)
            reservation = pool.acquire_for_step(pilot_layer, expert_ids, provider)
            try:
                assert len(set(reservation.slot_ids)) == sanity_config.top_k
                assert all(0 <= slot < pool.num_slots for slot in reservation.slot_ids)
                assert [int(value) for value in reservation.device_slot_ids] == list(reservation.slot_ids)
                for expert_id, slot_id in zip(expert_ids, reservation.slot_ids):
                    assert pool.slot_of(pilot_layer, expert_id) == slot_id
                    for name in ("w1", "w2", "w3"):
                        assert torch.equal(
                            pool.weight_views(slot_id)[name], provider.pinned_cpu_weight(pilot_layer, expert_id, name)
                        ), f"step {step}: wrong bytes in slot {slot_id} for {name}"
                        assert torch.equal(
                            pool.scale_views(slot_id)[name],
                            provider.pinned_cpu_weight(pilot_layer, expert_id, name + "_scale"),
                        ), f"step {step}: wrong scale bytes in slot {slot_id} for {name}"
                assert pool_fingerprints() == baseline, f"step {step}: arena moved"
            finally:
                pool.release_step(reservation)

    assert pool.resident_expert_count == NUM_SLOTS_FOR_CHURN_TEST
    assert pool.stats.hits >= 1, "expected cross-step LRU hits in the sliding window"
    assert pool.stats.hits + pool.stats.loads == NUM_DECODE_STEPS * sanity_config.top_k
    assert pool.stats.evictions == pool.stats.loads - NUM_SLOTS_FOR_CHURN_TEST
    assert pool_fingerprints() == baseline


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
    from ..benchmarks.capacity_probe import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--device", "cpu", "--small-geometry", "--safety-margin", str(margin)])
    assert excinfo.value.code == 2


def test_draft_engine_zero_allocation_50_steps(
    sanity_config: DeepSeekV4MoEConfig,
    sanity_layout: ExpertTensorLayout,
    mock_provider_factory,
    allocation_guard,
) -> None:
    """Engine-level pointer stability: scratchpad + pool arenas never move,
    and the surrogate expert kernel itself allocates nothing per execution."""
    provider = mock_provider_factory(
        sanity_layout, layer_ids=tuple(range(sanity_config.num_layers)), num_experts=sanity_config.num_routed_experts
    )
    engine = DraftInferenceEngine(
        config=sanity_config,
        layout=sanity_layout,
        provider=provider,
        backend=MockV5Backend(),
        num_slots=32,
        device="cpu",
    )
    engine.warm_up_routing_tables()
    fingerprint = engine.arena_fingerprint()

    with allocation_guard():
        for step_index in range(50):
            engine.prepare_step(token_id=(step_index * 13) % sanity_config.vocab_size)
            report = engine.step()
            assert report.layers_executed == sanity_config.num_layers
            assert engine.arena_fingerprint() == fingerprint, f"step {step_index}: an arena moved"

    assert report.layer_misses >= 0
    assert engine.arena_fingerprint() == fingerprint
