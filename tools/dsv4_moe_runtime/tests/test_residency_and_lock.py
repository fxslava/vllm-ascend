"""Zero-allocation decode loops and transactional step-level locking."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

import pytest
import torch

from ..core.config import DeepSeekV4MoEConfig
from ..core.layout import ExpertTensorLayout
from ..core.ledger import SlotExhaustionError
from ..core.slot_pool import StaticExpertSlotPool
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


def test_step_lock_prevents_eviction_collision(
    sanity_config: DeepSeekV4MoEConfig,
    sanity_layout: ExpertTensorLayout,
    mock_provider_factory: Callable[..., WeightProviderProtocol],
) -> None:
    pool = StaticExpertSlotPool(sanity_config, num_slots=sanity_config.top_k, layout=sanity_layout)
    provider = mock_provider_factory(sanity_layout, layer_ids=(0,), num_experts=sanity_config.num_routed_experts)

    first_ids = [10, 11, 12, 13, 14, 15]
    reservation = pool.acquire_for_step(0, first_ids, provider)
    first_slots = {expert_id: pool.slot_of(0, expert_id) for expert_id in first_ids}
    assert len(set(first_slots.values())) == sanity_config.top_k
    baseline = [pool.slot_arena.data_ptr(), pool.expert_slot_table.data_ptr(), pool.step_slot_ids_buffer.data_ptr()]

    # Same step, all 6 slots locked, 6 fresh experts requested: must refuse
    # atomically instead of evicting any locked slot.
    with pytest.raises(SlotExhaustionError, match="evictable"):
        pool.acquire_for_step(0, [100, 101, 102, 103, 104, 105], provider)

    # State is untouched: residency, bytes, counters.
    assert {e: pool.slot_of(0, e) for e in first_ids} == first_slots
    assert pool.stats.evictions == 0
    assert pool.resident_expert_count == sanity_config.top_k
    for expert_id, slot_id in first_slots.items():
        assert torch.equal(pool.weight_views(slot_id)["w1"], provider.pinned_cpu_weight(0, expert_id, "w1"))
    assert [
        pool.slot_arena.data_ptr(),
        pool.expert_slot_table.data_ptr(),
        pool.step_slot_ids_buffer.data_ptr(),
    ] == baseline

    pool.release_step(reservation)

    # Locks released: the same request now succeeds by evicting exactly the
    # 6 (now-unlocked) LRU-oldest slots, reusing all of them.
    second_ids = [100, 101, 102, 103, 104, 105]
    second = pool.acquire_for_step(0, second_ids, provider)
    assert set(second.slot_ids) == set(first_slots.values())
    assert all(pool.slot_of(0, expert_id) == -1 for expert_id in first_ids)
    assert pool.stats.evictions == sanity_config.top_k
    for expert_id, slot_id in zip(second_ids, second.slot_ids):
        assert torch.equal(pool.weight_views(slot_id)["w2"], provider.pinned_cpu_weight(0, expert_id, "w2"))
    pool.release_step(second)

    # --- Free slots do not weaken the lock: 8 slots, 6 locked, 2 free. ---
    wide_pool = StaticExpertSlotPool(sanity_config, num_slots=8, layout=sanity_layout)
    held = wide_pool.acquire_for_step(0, [0, 1, 2, 3, 4, 5], provider)
    assert wide_pool.resident_expert_count == 6
    with pytest.raises(SlotExhaustionError):
        wide_pool.acquire_for_step(0, [6, 7, 8, 9, 10, 11], provider)  # 2 free < 6 needed, rest locked
    wide_pool.release_step(held)

    evicting = wide_pool.acquire_for_step(0, [6, 7, 8, 9, 10, 11], provider)
    assert all(wide_pool.slot_of(0, expert_id) == -1 for expert_id in (0, 1, 2, 3))  # LRU-oldest evicted
    survivors = [wide_pool.slot_of(0, expert_id) for expert_id in (4, 5)]
    assert -1 not in survivors
    assert sorted(list(evicting.slot_ids) + survivors) == list(range(8))
    wide_pool.release_step(evicting)

    # --- Resident hits are never evicted to make room for co-requested misses. ---
    hit_pool = StaticExpertSlotPool(sanity_config, num_slots=sanity_config.top_k, layout=sanity_layout)
    warmup = hit_pool.acquire_for_step(0, [0, 1, 2, 3, 4, 5], provider)
    hit_pool.release_step(warmup)
    mixed = hit_pool.acquire_for_step(0, [2, 3, 4, 5, 6, 7], provider)
    assert [hit_pool.slot_of(0, expert_id) for expert_id in (2, 3, 4, 5)] == [
        warmup.slot_ids[2],
        warmup.slot_ids[3],
        warmup.slot_ids[4],
        warmup.slot_ids[5],
    ]
    assert all(hit_pool.slot_of(0, expert_id) == -1 for expert_id in (0, 1))  # only true LRU victims
    assert hit_pool.stats.evictions == 2
    hit_pool.release_step(mixed)
