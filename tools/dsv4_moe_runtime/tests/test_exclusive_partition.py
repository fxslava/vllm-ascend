"""Exclusive coverage, byte preservation, stream ordering and reader sealing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ..core.config import MoEGeometry
from ..core.layout import ExpertTensorLayout
from ..core.slot_pool import StaticExpertSlotPool
from ..hardware.exclusive_partition import ExclusiveExpertPartition, plan_vram_slots
from ..hardware.runtime import make_runtime
from ..hardware.weight_loader import MODE_MMAP, StreamingWeightLoader, _RawFileReader


class StartupSource:
    def __init__(self, layout):
        self.layout = layout
        self.closed = False
        self.reads = 0

    def fill_slot(self, destination, layer, expert):
        assert not self.closed
        destination.fill_(layer * 17 + expert + 1)
        self.reads += 1

    def fill_slot_params(self, layer, expert, views):
        assert not self.closed
        for view in views.values():
            view.fill_(layer * 17 + expert + 1)
        self.reads += 1
        return self.layout.slot_num_bytes

    def synchronize(self):
        pass

    def close(self):
        self.closed = True


def make_partition(device):
    config = MoEGeometry(
        hidden_size=64,
        moe_intermediate_size=32,
        num_routed_experts=4,
        top_k=2,
        num_layers=2,
        num_hash_layers=0,
        vocab_size=32,
    )
    layout = ExpertTensorLayout.for_dense_bf16(config)
    pool = StaticExpertSlotPool(config, 2, layout, device)
    source = StartupSource(layout)
    keys = [(layer, expert) for layer in range(2) for expert in range(4)]
    provider = ExclusiveExpertPartition(make_runtime(device), layout, pool, keys, source, chunk_bytes=1024)
    return pool, provider, source


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_every_expert_retains_exact_bytes_after_repeated_exchanges(device):
    pool, provider, source = make_partition(device)
    try:
        pointers = (pool.slot_arena.data_ptr(), *provider.pointers())
        assert source.closed and source.reads == 8
        assert provider.host_slots == 6
        if device.startswith("cuda"):
            assert all(block.is_pinned() for block in provider.host_arenas)
        for token in range(3):
            for layer, experts in [(0, [2, 3]), (1, [0, 1]), (0, [0, 1]), (1, [2, 3])]:
                with pool.step(layer, experts, provider):
                    provider.validate_residency(pool)
                    for expert in experts:
                        region = pool.slot_region(pool.slot_of(layer, expert))
                        assert bool((region == layer * 17 + expert + 1).all())
                pool.advance_generation(token)
                for key, slot in provider.host_residents.items():
                    assert bool((provider.host_regions[slot] == key[0] * 17 + key[1] + 1).all())
            assert pointers == (pool.slot_arena.data_ptr(), *provider.pointers())
        assert source.reads == 8
        assert provider.h2d_bytes == provider.d2h_bytes == provider.swaps * pool.layout.slot_num_bytes
        assert bool((provider.host_scratch == 0).all())
        assert bool((provider.device_scratch == 0).all())
        with pytest.raises(RuntimeError, match="disk reads are forbidden"):
            provider.read_param(0, 0, "w1")
    finally:
        provider.close()


def test_vram_hit_has_no_transfer_and_swap_failure_is_fail_stop():
    pool, provider, source = make_partition("cpu")
    with pool.step(0, [0, 1], provider):
        assert provider.swaps == provider.h2d_bytes == provider.d2h_bytes == 0
        assert pool.stats.hits == 2 and pool.stats.loads == 0

    def fail(*args):
        raise RuntimeError("injected DMA failure")

    provider._swap = fail
    with pytest.raises(RuntimeError, match="injected DMA"):
        pool.acquire_for_step(0, [2, 3], provider)
    with pytest.raises(RuntimeError, match="unusable"):
        provider.ensure_staged(0, 2)
    assert source.reads == 8


def test_closed_reader_cannot_reopen_file_or_mapping(tmp_path: Path):
    path = tmp_path / "weights.bin"
    path.write_bytes(bytes(4096))
    reader = _RawFileReader(str(path), MODE_MMAP)
    reader.close()
    chunk = torch.empty(4096, dtype=torch.uint8)
    for action in (
        lambda: reader.read_into(chunk, 0, 4096),
        lambda: reader.read_tail_into(chunk, 0, 4),
        lambda: reader.mmap_view(0, 4),
    ):
        with pytest.raises(RuntimeError, match="forbidden"):
            action()
    assert reader._fd is None and reader._tail_fd is None and reader._mmap_fd is None


def test_raw_cpu_ingestion_rejects_typed_destinations(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(bytes(4096))
    loader = StreamingWeightLoader(make_runtime("cpu"), str(path), chunk_bytes=4096, num_chunks=1, io_mode=MODE_MMAP)
    try:
        with pytest.raises(ValueError, match="raw uint8"):
            loader.stream_into(torch.empty(4096, dtype=torch.float32), 0, 4096)
    finally:
        loader.close()
    with pytest.raises(RuntimeError, match="forbidden"):
        loader.stream_into(torch.empty(4096, dtype=torch.uint8), 0, 4096)


def test_capacity_planning_accounts_for_fixed_weights_and_reserves():
    assert plan_vram_slots(12 * 1024**3, 3 * 1024**3, 16 * 1024**2, 1664, 512 * 1024**2, None) == 543
    with pytest.raises(MemoryError, match="requested"):
        plan_vram_slots(12 * 1024**3, 3 * 1024**3, 16 * 1024**2, 1664, 512 * 1024**2, 600)
