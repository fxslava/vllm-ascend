"""Generational radix-sorted eviction policy: aging, immunity, ordering.

Mandated checks:
* hash-layer (0-2) experts survive full forward passes through layer 42
  without eviction while active in Gen 0;
* correct generational aging T -> T+1 -> T+2 (with cold saturation);
* radix key ordering invariants (Gen2 before Gen1 before Gen0-cold before
  Gen0-hot, pinned hash core strictly last).
"""

from __future__ import annotations

import torch

from ..benchmarks.capacity_probe import DummyLoopbackWeightProvider
from ..core.config import SANITY_GEOMETRY
from ..core.generational_policy import (
    GENERATION_COLD_CAP,
    PRIORITY_COLD,
    PRIORITY_GEN0,
    PRIORITY_HASH,
    PRIORITY_PINNED,
    GenerationalRadixPolicy,
)
from ..core.layout import ExpertTensorLayout
from ..core.legacy_lru_policy import LegacyLruPolicy
from ..core.slot_pool import StaticExpertSlotPool
from ..hardware.runtime import CpuRuntime
from ..protocols.residency_policy import AdmissionDecision


def _seed_resident(policy: GenerationalRadixPolicy, layer: int, expert: int, slot: int) -> None:
    """Mark one expert resident without going through a pool acquisition."""
    policy.commit_decision(
        AdmissionDecision(
            hits=(),
            admissions=(((layer, expert), slot),),
            evictions=(),
            request_order=((layer, expert),),
            free_slot_count=0,
        )
    )


def _policy() -> GenerationalRadixPolicy:
    return GenerationalRadixPolicy(SANITY_GEOMETRY, num_slots=16)


def test_generational_aging_advances_and_saturates() -> None:
    policy = _policy()
    policy.register_access(5, 10, token_idx=0)  # score layer: Gen0 after token 0
    policy.advance_generation(0)
    assert policy.generation_of(5, 10) == 0  # accessed in the completed token -> Gen 0

    policy.advance_generation(1)  # not accessed during token 1 -> Gen 1
    assert policy.generation_of(5, 10) == 1
    policy.advance_generation(2)  # inactive for 2 tokens -> cold
    assert policy.generation_of(5, 10) == GENERATION_COLD_CAP
    policy.advance_generation(3)  # cold saturates
    assert policy.generation_of(5, 10) == GENERATION_COLD_CAP

    policy.advance_generation(0)  # stale boundary: idempotent, no rewinding
    assert policy.generation_of(5, 10) == GENERATION_COLD_CAP


def test_hash_layer_gen0_survives_full_forward_pass(
    sanity_config, sanity_layout: ExpertTensorLayout, mock_provider_factory
) -> None:
    """Late-layer churn (layers 3-42) must never displace the Gen0 hash core."""
    provider = mock_provider_factory(
        sanity_layout, layer_ids=tuple(range(sanity_config.num_layers)), num_experts=sanity_config.num_routed_experts
    )
    pool = StaticExpertSlotPool(sanity_config, num_slots=16, layout=sanity_layout)

    hash_experts = [0, 1, 2, 3, 4, 5]
    reservation = pool.acquire_for_step(0, hash_experts, provider)  # token 0, hash layer 0
    hash_slots = dict(zip(hash_experts, reservation.slot_ids))
    pool.release_step(reservation)
    pool.advance_generation(0)

    for layer_idx in range(3, sanity_config.num_layers):  # token 1 decodes layers 3..42
        expert_ids = [(layer_idx * 7 + offset) % sanity_config.num_routed_experts for offset in range(6)]
        churn = pool.acquire_for_step(layer_idx, expert_ids, provider)
        pool.release_step(churn)
        for expert_id, slot in hash_slots.items():
            assert pool.slot_of(0, expert_id) == slot, f"layer {layer_idx} evicted the hash core"
            for name in ("w1", "w2"):
                assert torch.equal(pool.weight_views(slot)[name], provider.pinned_cpu_weight(0, expert_id, name)), (
                    f"layer {layer_idx} corrupted hash-core bytes"
                )

    assert pool.slot_of(0, 0) == hash_slots[0]  # hash core fully intact after layer 42


def test_radix_key_ordering_invariants() -> None:
    policy = _policy()
    # Seed five residents with distinct generations/frequencies and one pinned
    # hash resident; then force full-pool evictions one request at a time.
    plan_residents = {
        (20, 0): 0,  # Gen2 cold (inactive 3 tokens, low freq)
        (20, 1): 1,  # Gen1 (inactive 1 token)
        (20, 2): 2,  # Gen0 cold (freshly accessed once)
        (20, 3): 3,  # Gen0 hot (accessed 5x)
        (0, 0): 4,  # pinned hash Gen0
    }
    for (layer, expert), slot in plan_residents.items():
        _seed_resident(policy, layer, expert, slot)
    policy.register_access(20, 0, token_idx=0)
    policy.advance_generation(0)
    policy.register_access(20, 1, token_idx=1)
    policy.register_access(20, 2, token_idx=2)
    for _ in range(5):
        policy.register_access(20, 3, token_idx=2)
    policy.register_access(0, 0, token_idx=2)
    policy.advance_generation(1)
    policy.advance_generation(2)  # (20,0) -> Gen 2; (20,1) -> Gen 1; core -> Gen 0

    assert policy.ranking_key(0, 0) >> 48 == PRIORITY_PINNED
    assert policy.ranking_key(20, 3) >> 48 == PRIORITY_GEN0
    assert policy.ranking_key(20, 2) >> 48 == PRIORITY_GEN0
    assert policy.ranking_key(20, 1) >> 48 == PRIORITY_COLD
    assert policy.generation_of(20, 0) == GENERATION_COLD_CAP
    assert policy.generation_of(20, 1) == 1
    assert policy.generation_of(20, 2) == 0
    assert (policy.ranking_key(20, 0) >> 32) & 0xFFFF == GENERATION_COLD_CAP
    assert policy.ranking_key(0, 0) > policy.ranking_key(20, 3)  # pinned > hot Gen0
    assert policy.ranking_key(0, 0) >> 48 == PRIORITY_HASH + 1  # pinned tops the ladder

    # Eviction order: Gen2 -> Gen1 -> Gen0-cold -> Gen0-hot -> pinned (last).
    eviction_order: list[tuple[int, int]] = []
    expected = [(20, 0), (20, 1), (20, 2), (20, 3), (0, 0)]
    remaining = set(plan_residents)
    while remaining:
        decision = policy.plan_admissions([(30, 99)], {k: plan_residents[k] for k in remaining}, [])
        victim_key, _victim_slot = decision.evictions[0]
        eviction_order.append(victim_key)
        remaining.discard(victim_key)
        policy.commit_decision(decision)
        policy.register_access(30, 99, token_idx=2)
    assert eviction_order == expected


def test_policy_swap_preserves_pool_contract(
    sanity_config, sanity_layout: ExpertTensorLayout, mock_provider_factory
) -> None:
    """LegacyLruPolicy and GenerationalRadixPolicy are drop-in interchangeable."""
    provider = mock_provider_factory(sanity_layout, layer_ids=(0,), num_experts=sanity_config.num_routed_experts)
    baseline_slots = None
    for policy in (LegacyLruPolicy(sanity_config, num_slots=8), GenerationalRadixPolicy(sanity_config, 8)):
        pool = StaticExpertSlotPool(sanity_config, num_slots=8, layout=sanity_layout, policy=policy)
        first = pool.acquire_for_step(0, [0, 1, 2, 3, 4, 5], provider)
        pool.release_step(first)
        resident = {expert: pool.slot_of(0, expert) for expert in range(6)}
        for expert, slot in resident.items():
            for param_key, view in pool.param_views(slot).items():
                assert torch.equal(view, provider.pinned_cpu_weight(0, expert, param_key))
        if baseline_slots is None:
            baseline_slots = resident
        assert resident == baseline_slots  # identical slot slicing under both policies


def test_pinned_core_survives_under_full_pool(sanity_config, sanity_layout: ExpertTensorLayout) -> None:
    """Gen0 hash residents rank last: exhausted pools evict them only at the very end."""
    provider = DummyLoopbackWeightProvider(CpuRuntime(), sanity_layout)
    pool = StaticExpertSlotPool(sanity_config, num_slots=sanity_config.top_k, layout=sanity_layout)
    core = pool.acquire_for_step(0, [0, 1, 2, 3, 4, 5], provider)  # hash layer 0
    pool.release_step(core)
    pool.advance_generation(0)
    core_slots = {expert: pool.slot_of(0, expert) for expert in range(6)}

    churn = pool.acquire_for_step(10, [10, 11, 12, 13, 14, 15], provider)  # late score layer
    pool.release_step(churn)
    # With pinned candidates as the ONLY alternative, the last-resort eviction
    # is allowed -- immunity is a rank, never a deadlock.
    assert all(pool.slot_of(0, expert) == -1 for expert in range(6))
    assert all(pool.slot_of(10, expert) >= 0 for expert in range(10, 16))
    del core_slots
