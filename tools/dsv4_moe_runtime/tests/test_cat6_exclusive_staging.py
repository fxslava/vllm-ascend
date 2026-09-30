"""CAT 6 -- exclusive staging: the bounded transit window replaces the monolithic host copy.

Regression class: the benchmark once staged ALL experts in pinned host DDR
(137 GiB for the full model -- instant OOM on <=128 GiB hosts). These tests
pin the replacement guarantees: the window stays bounded regardless of trace
coverage, evictions land in it keyed, promotions consume their entry
(exclusive, non-inclusive), drop-oldest recycles safely (including in-flight
entries), and the decode-step path never allocates.
"""

from __future__ import annotations

import pytest
import torch

from ..benchmarks.offload_stress import OffloadStressHarness, parse_config
from ..benchmarks.synthetic_source import SyntheticExpertSource
from ..core.layout import ExpertTensorLayout
from ..core.slot_pool import StaticExpertSlotPool
from ..hardware.exchange_buffer import ExchangeBufferFullError, TransitExchangeBuffer
from ..hardware.exclusive_staging import ExclusiveStagingProvider
from ..hardware.runtime import CpuRuntime
from ..protocols.provider import SlotFillProviderProtocol


def _make_source(layout: ExpertTensorLayout, num_layers: int = 2, num_experts: int = 8, seed: int = 5):
    return SyntheticExpertSource(layout, num_layers=num_layers, num_experts=num_experts, seed=seed)


def _zero_views(layout: ExpertTensorLayout) -> dict[str, torch.Tensor]:
    return {spec.param_key: torch.zeros(spec.view_shape, dtype=torch.uint8) for spec in layout.specs}


def test_exchange_buffer_keyed_lookup_consume_and_sync(sanity_layout: ExpertTensorLayout) -> None:
    buffer = TransitExchangeBuffer(CpuRuntime(), sanity_layout, host_slots=2, overflow="drop_oldest")
    staged = buffer.acquire_fill_slot((0, 5))
    staged.copy_(torch.arange(sanity_layout.slot_num_bytes, dtype=torch.uint8) % 251)
    assert buffer.lookup((0, 5)) == 0

    victim = torch.full((sanity_layout.slot_num_bytes,), 0x5A, dtype=torch.uint8)
    buffer.stage_eviction(victim, key=(1, 2))
    assert buffer.lookup((1, 2)) == 1

    assert buffer.consume_key((0, 5)) == 0
    assert buffer.lookup((0, 5)) is None  # consumed entries leave the keyed index
    assert buffer.in_flight == 2  # ...but stay occupied while the DMA is in flight
    buffer.synchronize()
    assert buffer.in_flight == 1 and buffer.state_of(0) == 0  # FIFO release
    assert buffer.consumed_entries == 1 and buffer.dropped_entries == 0


def test_exchange_buffer_drop_oldest_recycles_within_bound(sanity_layout: ExpertTensorLayout) -> None:
    buffer = TransitExchangeBuffer(CpuRuntime(), sanity_layout, host_slots=2, overflow="drop_oldest")
    fingerprint = buffer.fingerprint()
    buffer.stage_eviction(torch.full((sanity_layout.slot_num_bytes,), 1, dtype=torch.uint8), key=(0, 0))
    buffer.stage_eviction(torch.full((sanity_layout.slot_num_bytes,), 2, dtype=torch.uint8), key=(0, 1))

    buffer.acquire_fill_slot((0, 2))  # full ring: the oldest entry (0,0) is recycled
    assert buffer.lookup((0, 0)) is None
    assert buffer.lookup((0, 2)) is not None
    assert buffer.in_flight == 2 and buffer.dropped_entries == 1
    assert buffer.fingerprint() == fingerprint  # host arena never moved


def test_exchange_buffer_superseded_key_keeps_newest_entry(sanity_layout: ExpertTensorLayout) -> None:
    buffer = TransitExchangeBuffer(CpuRuntime(), sanity_layout, host_slots=3, overflow="drop_oldest")
    region = torch.zeros(sanity_layout.slot_num_bytes, dtype=torch.uint8)
    buffer.stage_eviction(region, key=(7, 7))  # slot 0
    buffer.stage_eviction(region, key=(7, 7))  # slot 1 supersedes it in the index
    assert buffer.lookup((7, 7)) == 1

    assert buffer.consume_key((7, 7)) == 1
    buffer.stage_eviction(region, key=(8, 8))  # slot 2: ring full
    buffer.stage_eviction(region, key=(9, 9))  # drops slot 0 (the stale anonymous copy)
    buffer.synchronize()  # releases slot 1 (the consumed (7,7))
    assert buffer.lookup((7, 7)) is None and buffer.lookup((8, 8)) == 2 and buffer.lookup((9, 9)) == 0
    assert buffer.in_flight == 2 and buffer.dropped_entries == 1


def test_exchange_buffer_error_mode_still_backpressures(sanity_layout: ExpertTensorLayout) -> None:
    buffer = TransitExchangeBuffer(CpuRuntime(), sanity_layout, host_slots=1)
    buffer.stage_eviction(torch.zeros(sanity_layout.slot_num_bytes, dtype=torch.uint8), key=(0, 0))
    with pytest.raises(ExchangeBufferFullError):
        buffer.acquire_fill_slot((0, 1))
    with pytest.raises(ExchangeBufferFullError):
        buffer.stage_eviction(torch.zeros(sanity_layout.slot_num_bytes, dtype=torch.uint8))
    with pytest.raises(ValueError, match="overflow"):
        TransitExchangeBuffer(CpuRuntime(), sanity_layout, host_slots=1, overflow="spill")


def test_synthetic_source_tiles_deterministic_and_distinct(sanity_layout: ExpertTensorLayout) -> None:
    source = _make_source(sanity_layout)
    assert source.contains(1, 7) and not source.contains(2, 0) and not source.contains(0, 8)

    tile = source.read_param(0, 3, "w1")
    assert torch.equal(tile, source.read_param(0, 3, "w1"))  # deterministic
    assert not torch.equal(tile, source.read_param(0, 2, "w1"))  # distinct per expert
    assert not torch.equal(tile, source.read_param(1, 3, "w1"))  # distinct per layer

    region = torch.zeros(sanity_layout.slot_num_bytes, dtype=torch.uint8)
    source.fill_slot(region, 0, 3)
    flat = tile.reshape(-1)
    assert torch.equal(region.view(-1)[: flat.numel()], flat)
    with pytest.raises(KeyError):
        source.fill_slot(region, 2, 0)


def test_exclusive_provider_streams_through_window_and_consumes(sanity_layout: ExpertTensorLayout) -> None:
    runtime = CpuRuntime()
    buffer = TransitExchangeBuffer(runtime, sanity_layout, host_slots=3, overflow="drop_oldest")
    source = _make_source(sanity_layout)
    provider = ExclusiveStagingProvider(runtime, sanity_layout, buffer, source)
    assert isinstance(provider, SlotFillProviderProtocol)
    assert provider.window_bytes == 3 * sanity_layout.slot_num_bytes

    provider.ensure_staged(0, 1)
    provider.ensure_staged(0, 1)  # second look: window hit, no source fill
    assert provider.window_hits == 1 and provider.window_source_fills == 1

    views = _zero_views(sanity_layout)
    moved = provider.fill_slot_params(0, 1, views)
    assert moved == sanity_layout.slot_num_bytes
    for param_key, view in views.items():
        assert torch.equal(view, source.read_param(0, 1, param_key))
    assert provider.bytes_copied == moved and provider.dma_copy_count == len(views)
    assert buffer.lookup((0, 1)) is None  # exclusive: the promotion consumed the entry
    assert buffer.in_flight == 1  # INFLIGHT: DMA source protected until synchronize
    provider.synchronize()
    assert buffer.in_flight == 0

    with pytest.raises(KeyError):
        provider.fill_slot_params(0, 1, views)  # nothing staged anymore
    with pytest.raises(KeyError):
        provider.ensure_staged(5, 0)  # outside source coverage: raises before mutating
    assert buffer.in_flight == 0  # the failed ensure_staged mutated nothing


def test_reclaimed_entry_is_refilled_from_the_source(sanity_layout: ExpertTensorLayout) -> None:
    """A step whose staged entry the ring drops must still promote correct bytes.

    Regression: with ``drop_oldest``, the evictions a step stages *after*
    ``ensure_staged`` can reclaim the very entry that step is about to promote
    (the common case is a window hit on a long-evicted expert, which sits at the
    ring head). The fill used to raise mid-admission; it now re-reads the
    authoritative source, which is what makes window entries droppable caches.
    """
    runtime = CpuRuntime()
    buffer = TransitExchangeBuffer(runtime, sanity_layout, host_slots=2, overflow="drop_oldest")
    source = _make_source(sanity_layout)
    provider = ExclusiveStagingProvider(runtime, sanity_layout, buffer, source)

    provider.ensure_staged(0, 1)  # the promotee lands at the ring head
    victim = torch.full((sanity_layout.slot_num_bytes,), 0x5A, dtype=torch.uint8)
    buffer.stage_eviction(victim, key=(0, 6))
    buffer.stage_eviction(victim, key=(0, 7))  # ring full -> drops the oldest entry
    assert buffer.lookup((0, 1)) is None and buffer.dropped_entries == 1

    views = _zero_views(sanity_layout)
    moved = provider.fill_slot_params(0, 1, views)
    assert moved == sanity_layout.slot_num_bytes
    assert provider.window_refills == 1
    for param_key, view in views.items():
        assert torch.equal(view, source.read_param(0, 1, param_key))

    provider.synchronize()
    with pytest.raises(KeyError):  # a key no step prepared is still a hard error
        provider.fill_slot_params(0, 1, views)


def test_pool_window_hit_avoids_source_refill(sanity_config, sanity_layout: ExpertTensorLayout) -> None:
    runtime = CpuRuntime()
    buffer = TransitExchangeBuffer(runtime, sanity_layout, host_slots=2 * sanity_config.top_k, overflow="drop_oldest")
    source = _make_source(sanity_layout, num_layers=1, num_experts=sanity_config.num_routed_experts, seed=9)
    provider = ExclusiveStagingProvider(runtime, sanity_layout, buffer, source)
    pool = StaticExpertSlotPool(sanity_config, num_slots=6, layout=sanity_layout, exchange_buffer=buffer)

    first = pool.acquire_for_step(0, [0, 1, 2, 3, 4, 5], provider)
    pool.release_step(first)
    second = pool.acquire_for_step(0, [100, 101, 102, 103, 104, 105], provider)  # evicts 0-5 -> window
    pool.release_step(second)
    fills_after_evictions = provider.window_source_fills
    assert fills_after_evictions == 12

    third = pool.acquire_for_step(0, [0, 1, 2, 3, 4, 5], provider)  # window hit: no source refill
    pool.release_step(third)
    assert provider.window_source_fills == fills_after_evictions
    assert provider.window_hits == sanity_config.top_k
    for expert in range(6):
        slot = pool.slot_of(0, expert)
        for param_key, view in pool.param_views(slot).items():
            assert torch.equal(view, source.read_param(0, expert, param_key))


def test_harness_bounded_window_regression() -> None:
    """The defect: full-model staging (137 GiB) instead of a bounded window."""
    config = parse_config(["--dry-run", "--steps", "60", "--verify-samples", "6", "--buckets", "4"])
    report = OffloadStressHarness(config).run()
    assert len(report.metrics) == 60 and report.invariants_ok()
    assert report.host_window_bytes == config.transit_slots * report.slot_num_bytes  # the ONLY host staging
    assert report.window_cap_ok
    assert report.transit_peak_in_flight <= config.transit_slots
    assert report.window_source_fills > 0
    assert report.window_dropped > 0  # churn actually recycled the bounded window
    assert report.verification is not None and report.verification[0]


def test_decode_step_path_is_allocation_free(
    sanity_config, sanity_layout: ExpertTensorLayout, allocation_guard
) -> None:
    runtime = CpuRuntime()
    buffer = TransitExchangeBuffer(runtime, sanity_layout, host_slots=12, overflow="drop_oldest")
    source = SyntheticExpertSource(
        sanity_layout, num_layers=sanity_config.num_layers, num_experts=sanity_config.num_routed_experts, seed=1
    )
    provider = ExclusiveStagingProvider(runtime, sanity_layout, buffer, source)
    pool = StaticExpertSlotPool(sanity_config, num_slots=8, layout=sanity_layout, exchange_buffer=buffer)

    with allocation_guard():
        for step, layer in enumerate(range(6)):
            expert_ids = [(step * 3 + offset) % sanity_config.num_routed_experts for offset in range(6)]
            reservation = pool.acquire_for_step(layer, expert_ids, provider)
            provider.synchronize()
            pool.release_step(reservation)
            pool.advance_generation(step)
    assert pool.stats.loads >= 6 and pool.stats.evictions > 0
    assert buffer.dropped_entries > 0  # the window recycled under churn, allocation-free


def test_harness_verification_is_nonvacuous() -> None:
    """A pool large enough to hold every touched expert: sampled views are
    actually byte-checked against the source (not all skipped as evicted)."""
    config = parse_config(["--dry-run", "--steps", "8", "--pool-slots", "64", "--verify-samples", "4"])
    report = OffloadStressHarness(config).run()
    assert report.invariants_ok()
    params_per_expert = len(config.layout.specs)
    assert report.verification == (True, 4 * config.model_config.top_k * params_per_expert)


def test_transit_window_cap_rejects_oversize() -> None:
    with pytest.raises(SystemExit):
        parse_config(["--transit-slots", "2571"])  # 2571 x 12.75 MiB > 32 GiB
    assert parse_config(["--transit-slots", "2570"]).transit_slots == 2570  # exactly at the cap
    with pytest.raises(SystemExit):
        parse_config(["--dry-run", "--transit-slots", "8"])  # below one full exchange step (2 x top-k)
