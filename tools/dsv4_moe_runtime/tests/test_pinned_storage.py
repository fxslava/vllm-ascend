"""DMA-capable pinned host storage and transactional streamed fills."""

from __future__ import annotations

import pytest
import torch

from ..core.layout import ExpertTensorLayout
from ..core.slot_pool import StaticExpertSlotPool
from ..hardware.pinned_storage import AscendPinnedHostStorage
from ..hardware.runtime import CpuRuntime
from ..protocols.provider import SlotFillProviderProtocol


def test_pinned_storage_stages_and_fills_views(sanity_layout: ExpertTensorLayout) -> None:
    runtime = CpuRuntime()
    storage = AscendPinnedHostStorage(runtime, sanity_layout, layer_ids=[0, 1], experts_per_layer=8)
    assert isinstance(storage, SlotFillProviderProtocol)
    assert storage.staged_expert_count == 16
    assert storage.staged_bytes() == 16 * sanity_layout.slot_num_bytes

    storage.ensure_staged(1, 7)
    with pytest.raises(KeyError):
        storage.ensure_staged(2, 0)  # layer 2 not staged
    with pytest.raises(KeyError):
        storage.ensure_staged(1, 8)  # expert id beyond staged range

    views = {spec.param_key: torch.zeros(spec.view_shape, dtype=torch.uint8) for spec in sanity_layout.specs}
    moved = storage.fill_slot_params(1, 7, views)
    assert moved == sanity_layout.slot_num_bytes
    assert storage.bytes_copied == sanity_layout.slot_num_bytes
    for param_key, view in views.items():
        assert torch.equal(view, storage.pinned_cpu_weight(1, 7, param_key))


def test_streamed_fill_is_transactional_on_unstaged_expert(sanity_config, sanity_layout: ExpertTensorLayout) -> None:
    pool = StaticExpertSlotPool(sanity_config, num_slots=6)
    storage = AscendPinnedHostStorage(CpuRuntime(), sanity_layout, layer_ids=[0], experts_per_layer=8)

    first = pool.acquire_for_step(0, [0, 1, 2, 3, 4, 5], storage)
    assert pool.stats.loads == 6
    assert pool.stats.bytes_staged == 6 * sanity_layout.slot_num_bytes
    resident = {expert: pool.slot_of(0, expert) for expert in range(6)}
    for expert, slot in resident.items():
        for param_key, view in pool.param_views(slot).items():
            assert torch.equal(view, storage.pinned_cpu_weight(0, expert, param_key))
    pool.release_step(first)

    # Expert ids 100+ are not staged: ensure_staged must fail BEFORE the ledger
    # commits, so residency, tables and counters are untouched.
    with pytest.raises(KeyError):
        pool.acquire_for_step(0, [100, 101, 102, 103, 104, 105], storage)
    assert {expert: pool.slot_of(0, expert) for expert in range(6)} == resident
    assert pool.stats.evictions == 0 and pool.stats.loads == 6
    assert pool.stats.bytes_staged == 6 * sanity_layout.slot_num_bytes
