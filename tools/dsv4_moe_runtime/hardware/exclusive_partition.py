"""Complete, disk-free routed weight partition with chunked exclusive swaps.

Resident expert keys are disjoint at every acquisition boundary. Two bounded
transfer chunks are transient scratch, not additional resident expert slots.
CUDA copies cannot exchange overlapping source/destination slots directly.
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Sequence

import psutil
import torch

from ..protocols.residency_policy import AdmissionDecision

TRANSFER_CHUNK_BYTES = 4 * 1024**2
HOST_RESERVE_BYTES = 2 * 1024**3
HOST_BLOCK_BYTES = 1024**3


def plan_vram_slots(
    free_bytes: int, fixed_bytes: int, slot_bytes: int, total_experts: int, reserve_bytes: int, requested: int | None
) -> int:
    maximum = min(
        total_experts, max(0, (free_bytes - fixed_bytes - reserve_bytes - TRANSFER_CHUNK_BYTES) // slot_bytes)
    )
    selected = maximum if requested is None else requested
    if selected < 6 or selected > maximum:
        raise MemoryError(f"VRAM plan permits 6..{maximum} routed slots, requested {selected}")
    return selected


class ExclusiveExpertPartition:
    """One authoritative location per routed expert, no runtime file source."""

    def __init__(
        self,
        runtime,
        layout,
        pool,
        keys: Sequence[tuple[int, int]],
        disk_source,
        chunk_bytes: int = TRANSFER_CHUNK_BYTES,
    ):
        self.runtime = runtime
        self.layout = layout
        self.keys = frozenset(keys)
        if len(self.keys) != len(keys) or not pool.num_slots <= len(keys):
            raise ValueError("partition keys must be unique and cover the VRAM pool")
        if chunk_bytes <= 0:
            raise ValueError("transfer chunk size must be positive")
        if runtime.device != "cpu" and not runtime.device.startswith("cuda"):
            raise ValueError("exclusive full-duplex partition currently supports CUDA and CPU tests")
        self.host_slots = len(keys) - pool.num_slots
        host_bytes = self.host_slots * layout.slot_num_bytes
        if host_bytes + chunk_bytes + HOST_RESERVE_BYTES > psutil.virtual_memory().available:
            raise MemoryError(f"exclusive partition needs {host_bytes} pinned bytes plus host reserve")
        # A single 26-GiB pin request can exceed a driver allocation limit or
        # round up excessively in the host caching allocator. Fixed blocks
        # still contain exactly N-K slots and every block is page-locked.
        slots_per_block = max(1, HOST_BLOCK_BYTES // layout.slot_num_bytes)
        self.host_arenas = tuple(
            torch.empty(
                min(slots_per_block, self.host_slots - start) * layout.slot_num_bytes,
                dtype=torch.uint8,
                pin_memory=runtime.device != "cpu",
            )
            for start in range(0, self.host_slots, slots_per_block)
        )
        self.host_regions = tuple(
            block.narrow(0, offset, layout.slot_num_bytes)
            for block in self.host_arenas
            for offset in range(0, block.numel(), layout.slot_num_bytes)
        )
        self.host_residents: dict[tuple[int, int], int] = {}
        self.chunk_bytes = min(chunk_bytes, layout.slot_num_bytes)
        self.host_scratch = torch.empty(self.chunk_bytes, dtype=torch.uint8, pin_memory=runtime.device != "cpu")
        self.device_scratch = torch.empty(self.chunk_bytes, dtype=torch.uint8, device=runtime.device)
        self.h2d_stream = runtime.make_stream()
        self.d2h_stream = runtime.make_stream()
        self.stage_stream = runtime.make_stream()
        self.poisoned = False
        self.swap_seconds = self.h2d_seconds = self.d2h_seconds = self.duplex_seconds = 0.0
        self.h2d_bytes = self.d2h_bytes = self.swaps = 0
        self.disk_read_bytes = 0
        self.disk_read_requests = 0
        if runtime.device.startswith("cuda"):
            self.events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(4))
        else:
            self.events = ()
        self.startup_bytes = 0
        # No mmap views are created: all payloads flow through the loader's
        # bounded pinned chunks into their final resident slots exactly once.
        for slot, key in enumerate(keys[: pool.num_slots]):
            self.startup_bytes += disk_source.fill_slot_params(*key, pool.param_views(slot))
        for slot, key in enumerate(keys[pool.num_slots :]):
            disk_source.fill_slot(self.host_regions[slot], *key)
            self.startup_bytes += layout.slot_num_bytes
            self.host_residents[key] = slot
        disk_source.synchronize()
        pool.initialize_residency(keys[: pool.num_slots])
        disk_source.close()
        self._disk_source = disk_source  # sealed metadata retained for guard verification
        self.host_scratch.zero_()
        self.device_scratch.zero_()
        runtime.synchronize_device()
        self.validate_residency(pool)
        self.fingerprint = self.pointers()

    def pointers(self):
        return (
            *[block.data_ptr() for block in self.host_arenas],
            self.host_scratch.data_ptr(),
            self.device_scratch.data_ptr(),
        )

    @property
    def host_bytes(self):
        return sum(block.numel() for block in self.host_arenas)

    def ensure_staged(self, layer_idx, expert_id):
        if self.poisoned:
            raise RuntimeError("exclusive partition is unusable after a failed transfer")
        if (layer_idx, expert_id) not in self.host_residents:
            raise KeyError("requested missing expert has no authoritative RAM slot")

    def pinned_cpu_weight(self, layer_idx, expert_id, param_key):
        slot = self.host_residents[(layer_idx, expert_id)]
        spec = self.layout.spec_for(param_key)
        return self.host_regions[slot].narrow(0, spec.offset_bytes, spec.num_bytes).view(spec.view_shape)

    def fill_slot_params(self, *args, **kwargs):
        raise RuntimeError("exclusive promotion requires a paired eviction, never an independent fill")

    def read_param(self, *args, **kwargs):
        self.disk_read_requests += 1
        raise RuntimeError("runtime disk reads are forbidden in the exclusive hierarchy")

    def validate_residency(self, pool):
        gpu_keys = pool.resident_keys()
        ram_keys = frozenset(self.host_residents)
        if gpu_keys & ram_keys or gpu_keys | ram_keys != self.keys:
            raise RuntimeError("exclusive partition coverage or overlap invariant violated")
        if len(gpu_keys) != pool.num_slots or len(ram_keys) != self.host_slots:
            raise RuntimeError("exclusive partition occupancy changed")
        if len(set(self.host_residents.values())) != self.host_slots:
            raise RuntimeError("multiple experts alias one host slot")

    def exchange_admissions(self, decision: AdmissionDecision, pool) -> int:
        if self.poisoned:
            raise RuntimeError("exclusive partition is unusable after a failed transfer")
        victims = {slot: key for key, slot in decision.evictions}
        if decision.free_slot_count or len(victims) != len(decision.admissions):
            raise RuntimeError("exclusive hierarchy requires a full paired slot exchange")
        for key, slot in decision.admissions:
            self.ensure_staged(*key)
            if slot not in victims:
                raise RuntimeError("admission has no matching eviction victim")
        try:
            for key, slot in decision.admissions:
                host_slot = self.host_residents[key]
                self._swap(pool.slot_region(slot), self.host_regions[host_slot])
                del self.host_residents[key]
                self.host_residents[victims[slot]] = host_slot
                self.swaps += 1
        except Exception:
            # An in-place exchange cannot safely retry after partial DMA.
            self.poisoned = True
            raise
        return len(decision.admissions) * self.layout.slot_num_bytes

    def _swap(self, gpu, host):
        started = time.perf_counter()
        self.runtime.adopt_current_stream(self.stage_stream)
        for offset in range(0, self.layout.slot_num_bytes, self.chunk_bytes):
            length = min(self.chunk_bytes, self.layout.slot_num_bytes - offset)
            gpu_chunk = gpu.narrow(0, offset, length)
            host_chunk = host.narrow(0, offset, length)
            host_temp = self.host_scratch.narrow(0, 0, length)
            device_temp = self.device_scratch.narrow(0, 0, length)
            # A CPU memcpy avoids PyTorch's threaded CPU copy overhead and
            # preserves the pinned incoming source before D2H overwrites it.
            ctypes.memmove(host_temp.data_ptr(), host_chunk.data_ptr(), length)
            with self.runtime.stream_context(self.stage_stream):
                device_temp.copy_(gpu_chunk, non_blocking=True)
            self.runtime.synchronize_stream(self.stage_stream)
            duplex_started = time.perf_counter()
            with self.runtime.stream_context(self.h2d_stream):
                if self.events:
                    self.events[0].record(self.h2d_stream)
                gpu_chunk.copy_(host_temp, non_blocking=True)
                if self.events:
                    self.events[1].record(self.h2d_stream)
            with self.runtime.stream_context(self.d2h_stream):
                if self.events:
                    self.events[2].record(self.d2h_stream)
                host_chunk.copy_(device_temp, non_blocking=True)
                if self.events:
                    self.events[3].record(self.d2h_stream)
            self.runtime.synchronize_stream(self.h2d_stream)
            self.runtime.synchronize_stream(self.d2h_stream)
            self.duplex_seconds += time.perf_counter() - duplex_started
            if self.events:
                self.h2d_seconds += self.events[0].elapsed_time(self.events[1]) / 1000
                self.d2h_seconds += self.events[2].elapsed_time(self.events[3]) / 1000
        self.h2d_bytes += self.layout.slot_num_bytes
        self.d2h_bytes += self.layout.slot_num_bytes
        # Erase transient bytes at the acquisition boundary; no resident-sized
        # mirror or stale transfer payload survives the exchange.
        self.host_scratch.zero_()
        with self.runtime.stream_context(self.stage_stream):
            self.device_scratch.zero_()
        self.runtime.synchronize_stream(self.stage_stream)
        self.swap_seconds += time.perf_counter() - started

    def close(self):
        self.runtime.synchronize_stream(self.h2d_stream)
        self.runtime.synchronize_stream(self.d2h_stream)
        self.runtime.synchronize_stream(self.stage_stream)
